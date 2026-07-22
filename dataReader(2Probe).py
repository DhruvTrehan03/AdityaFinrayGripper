import argparse
import math
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyeit.eit.protocol as protocol


NUM_ELECTRODES = 16
EPS = 1e-12
FORCE_CALIBRATION_DIVISOR = 3.0
TRAPEZIUM_LEFT_HEIGHT = 20.0
TRAPEZIUM_RIGHT_HEIGHT = 15.0
TRAPEZIUM_WIDTH = 60.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse collected two-probe measurements as taxel heatmaps."
    )
    parser.add_argument(
        "--csv_file",
        default="/home/dhruv/AdityaFinrayGripper/RawData/2 Probe/2026-07-06_17-15-22_4Forces_40x10(2Probe).csv",
        help="Path to the two-probe CSV file.",
    )
    parser.add_argument(
        "--min_force",
        type=float,
        default=None,
        help="Only display contacts with actual_force_N >= min_force.",
    )
    parser.add_argument(
        "--max_force",
        type=float,
        default=None,
        help="Only display contacts with actual_force_N <= max_force.",
    )
    parser.add_argument(
        "--mode",
        choices=("delta", "percent", "raw", "max-decrease"),
        default="delta",
        help=(
            "Heatmap value: contact-baseline delta, percent change, raw contact, "
            "or baseline decrease with the largest-decrease taxel column attenuated."
        ),
    )
    parser.add_argument(
        "--row-sensitivity",
        type=float,
        default=1.0,
        help=(
            "In max-decrease mode, divide the rest of the winning row by this factor "
            "while preserving the winning taxel."
        ),
    )
    parser.add_argument(
        "--column-sensitivity",
        type=float,
        default=4.0,
        help=(
            "In max-decrease mode, divide the rest of the winning column by this factor "
            "while preserving the winning taxel. Use larger values to reduce column spread."
        ),
    )
    parser.add_argument(
        "--column-active-threshold",
        type=float,
        default=0.35,
        help=(
            "In max-decrease mode, also dampen columns whose strongest taxel is at "
            "least this fraction of the strongest touch. Prevents weak columns/noise "
            "from being dampened."
        ),
    )
    parser.add_argument(
        "--max-dampened-columns",
        type=int,
        default=3,
        help=(
            "Maximum number of active columns to dampen in max-decrease mode. "
            "Use 0 for no cap."
        ),
    )
    parser.add_argument(
        "--no_pair_average",
        action="store_true",
        help="Keep raw EIT channels instead of averaging duplicate protocol read pairs.",
    )
    parser.add_argument(
        "--condensed_csv_file",
        default=None,
        help="Optional path to write a pair-averaged CSV when loading raw 208-channel data.",
    )
    parser.add_argument(
        "--max_heatmaps",
        type=int,
        default=12,
        help="Maximum number of force heatmaps to show per location.",
    )
    parser.add_argument(
        "--plot_signal",
        action="store_true",
        help=(
            "Plot a force-stepped signal browser and response metrics instead "
            "of the location heatmap browser."
        ),
    )
    parser.add_argument(
        "--snr_threshold_db",
        type=float,
        default=6.0,
        help="SNR threshold used to mark the first significant force in signal mode.",
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=3.0,
        help="Per-channel z-score threshold used by the active-channel metric.",
    )
    parser.add_argument(
        "--signal_force_column",
        choices=("actual", "target"),
        default="actual",
        help="Force column used for arrow-key stepping in signal mode.",
    )
    parser.add_argument(
        "--signal_force_round",
        type=int,
        default=3,
        help="Decimal places used when grouping force values in signal mode.",
    )
    parser.add_argument(
        "--noise_max_force",
        type=float,
        default=0.05,
        help=(
            "Fallback noise estimate uses measurements at or below this actual "
            "force when explicit baseline rows are unavailable."
        ),
    )
    return parser.parse_args()


def find_eit_columns(columns):
    pattern = re.compile(r"^eit_(\d+)$")
    matches = []
    for col in columns:
        match = pattern.match(col)
        if match:
            matches.append((int(match.group(1)), col))
    if not matches:
        raise ValueError("No eit_n columns found.")
    return [col for _, col in sorted(matches)]


def build_channel_pair_map(num_electrodes=NUM_ELECTRODES):
    protocol_obj = protocol.create(
        num_electrodes,
        dist_exc=1,
        step_meas=1,
        parser_meas="rotate_meas",
    )

    channel_pairs = []
    for meas_pairs in protocol_obj.meas_mat:
        for m, n in meas_pairs:
            pair = tuple(sorted((int(m) + 1, int(n) + 1)))
            channel_pairs.append(pair)

    return channel_pairs


def pair_sort_key(pair):
    a, b = pair
    if a == 1 and b == NUM_ELECTRODES:
        return (NUM_ELECTRODES, b)
    return (a, b)


def condense_duplicate_pair_columns(df, eit_columns, channel_pairs):
    if len(eit_columns) != len(channel_pairs):
        print(
            "Pair averaging skipped: "
            f"{len(eit_columns)} EIT columns do not match "
            f"{len(channel_pairs)} protocol channels."
        )
        return df, eit_columns, None

    pair_to_columns = defaultdict(list)
    for col, pair in zip(eit_columns, channel_pairs):
        pair_to_columns[pair].append(col)

    unique_pairs = sorted(pair_to_columns, key=pair_sort_key)
    condensed = df.drop(columns=eit_columns).copy()
    insert_at = min(df.columns.get_loc(col) for col in eit_columns)

    averaged_columns = []
    pair_labels = {}
    for idx, pair in enumerate(unique_pairs):
        new_col = f"eit_{idx}"
        source_cols = pair_to_columns[pair]
        averaged_columns.append(new_col)
        pair_labels[new_col] = pair
        condensed.insert(
            insert_at + idx,
            new_col,
            df[source_cols].astype(float).mean(axis=1),
        )

    print(
        "Averaged duplicate EIT pair columns: "
        f"{len(eit_columns)} raw channels -> {len(averaged_columns)} unique pairs."
    )

    return condensed, averaged_columns, pair_labels


def odd_even_pair_labels(num_electrodes=NUM_ELECTRODES):
    odd = list(range(1, num_electrodes + 1, 2))
    even = list(range(2, num_electrodes + 1, 2))
    labels = [(a, b) for a in odd for b in even]
    return odd, even, labels


def infer_pair_labels(eit_columns, pair_labels):
    if pair_labels:
        return [pair_labels[col] for col in eit_columns]

    if len(eit_columns) == (NUM_ELECTRODES // 2) ** 2:
        _, _, labels = odd_even_pair_labels()
        return labels

    if len(eit_columns) == NUM_ELECTRODES:
        return [
            (electrode, electrode + 1)
            for electrode in range(1, NUM_ELECTRODES)
        ] + [(1, NUM_ELECTRODES)]

    return [(int(re.search(r"(\d+)$", col).group(1)), None) for col in eit_columns]


def heatmap_shape(n_channels):
    odd_even_side = NUM_ELECTRODES // 2
    if n_channels == odd_even_side * odd_even_side:
        return odd_even_side, odd_even_side

    side = int(math.sqrt(n_channels))
    if side * side == n_channels:
        return side, side

    cols = int(math.ceil(math.sqrt(n_channels)))
    rows = int(math.ceil(n_channels / cols))
    return rows, cols


def values_to_grid(values):
    values = np.asarray(values, dtype=float)
    rows, cols = heatmap_shape(len(values))
    grid = np.full((rows, cols), np.nan, dtype=float)
    grid.flat[: len(values)] = values
    return grid


def render_grid(values):
    return np.flipud(values_to_grid(values))


def trapezium_edge_mesh(
    rows,
    cols,
    left_height=TRAPEZIUM_LEFT_HEIGHT,
    right_height=TRAPEZIUM_RIGHT_HEIGHT,
    width=TRAPEZIUM_WIDTH,
):
    x_edges = np.zeros((rows + 1, cols + 1), dtype=float)
    y_edges = np.zeros((rows + 1, cols + 1), dtype=float)
    for col_index in range(cols + 1):
        t = col_index / cols if cols else 0.0
        height = left_height + (right_height - left_height) * t
        bottom = -0.5 * height
        top = 0.5 * height
        x_edges[:, col_index] = width * t
        y_edges[:, col_index] = np.linspace(bottom, top, rows + 1)
    return x_edges, y_edges


def is_taxel_grid(n_channels):
    rows, cols = heatmap_shape(n_channels)
    return n_channels == rows * cols and cols == NUM_ELECTRODES // 2


def taxel_labels(n_channels):
    return [str(idx) for idx in range(1, n_channels + 1)]


def axis_labels(n_channels, labels):
    rows, cols = heatmap_shape(n_channels)
    if is_taxel_grid(n_channels):
        xlabels = [str(idx) for idx in range(1, cols + 1)]
        ylabels = [str(row * cols + 1) for row in reversed(range(rows))]
        return xlabels, ylabels, "Taxel column", "Taxel row start"

    flat = []
    for pair in labels:
        if pair[1] is None:
            flat.append(str(pair[0]))
        else:
            flat.append(f"{pair[0]}-{pair[1]}")

    xlabels = []
    ylabels = []
    for c in range(cols):
        idx = c
        xlabels.append(flat[idx] if idx < len(flat) else "")
    for r in range(rows):
        idx = r * cols
        ylabels.append(flat[idx] if idx < len(flat) else "")
    return xlabels, ylabels, "Channel / pair", "Channel / pair"


def load_data(csv_file, average_duplicate_pairs=True, condensed_csv_file=None):
    df = pd.read_csv(csv_file)
    eit_columns = find_eit_columns(df.columns)
    pair_labels = None

    if average_duplicate_pairs:
        channel_pairs = build_channel_pair_map()
        df, eit_columns, pair_labels = condense_duplicate_pair_columns(
            df,
            eit_columns,
            channel_pairs,
        )

    if condensed_csv_file:
        df.to_csv(condensed_csv_file, index=False)
        print(f"Pair-averaged CSV written to {condensed_csv_file}")

    return df, eit_columns, pair_labels


def make_contact_records(
    df,
    eit_columns,
    mode,
    row_sensitivity=1.0,
    column_sensitivity=4.0,
    column_active_threshold=0.35,
    max_dampened_columns=3,
):
    records = []
    global_baseline = None

    if len(df) and str(df.iloc[0].get("row_type", "")).lower() == "baseline":
        if pd.isna(df.iloc[0].get("measurement_index", np.nan)):
            global_baseline = df.iloc[0][eit_columns].astype(float).values

    if "row_type" in df.columns and "measurement_index" in df.columns:
        baselines_by_index = {}
        for _, row in df.iterrows():
            row_type = str(row.get("row_type", "")).lower()
            measurement_index = row.get("measurement_index", np.nan)
            if pd.isna(measurement_index):
                continue

            key = int(measurement_index)
            values = row[eit_columns].astype(float).values
            if row_type == "baseline":
                baselines_by_index[key] = values
            elif row_type == "contact":
                baseline = baselines_by_index.get(key, global_baseline)
                if baseline is None:
                    baseline = np.zeros_like(values)
                records.append(
                    row_to_record(
                        row,
                        values,
                        baseline,
                        mode,
                        row_sensitivity=row_sensitivity,
                        column_sensitivity=column_sensitivity,
                        column_active_threshold=column_active_threshold,
                        max_dampened_columns=max_dampened_columns,
                    )
                )
    else:
        if global_baseline is None:
            global_baseline = df.iloc[0][eit_columns].astype(float).values
        for _, row in df.iloc[1:].iterrows():
            values = row[eit_columns].astype(float).values
            records.append(
                row_to_record(
                    row,
                    values,
                    global_baseline,
                    mode,
                    row_sensitivity=row_sensitivity,
                    column_sensitivity=column_sensitivity,
                    column_active_threshold=column_active_threshold,
                    max_dampened_columns=max_dampened_columns,
                )
            )

    return records


def active_column_winners(grid, threshold=0.35, max_columns=3):
    finite_grid = np.where(np.isfinite(grid), grid, -np.inf)
    global_peak = float(np.nanmax(finite_grid))
    if global_peak <= 0.0:
        return []

    column_peaks = np.nanmax(finite_grid, axis=0)
    active_columns = [
        col
        for col, peak in enumerate(column_peaks)
        if np.isfinite(peak) and peak > 0.0 and peak >= global_peak * threshold
    ]
    active_columns.sort(key=lambda col: column_peaks[col], reverse=True)
    if max_columns > 0:
        active_columns = active_columns[:max_columns]

    winners = []
    for col in active_columns:
        row = int(np.nanargmax(finite_grid[:, col]))
        winners.append((row, col))
    return winners


def attenuate_active_columns(grid, winners, row_factor=1.0, col_factor=1.0):
    shaped = grid.copy()
    preserved = {
        (row_index, col_index): shaped[row_index, col_index]
        for row_index, col_index in winners
    }

    if row_factor > 1.0:
        for row_index, _ in winners:
            shaped[row_index, :] /= row_factor
    if col_factor > 1.0:
        for _, col_index in winners:
            shaped[:, col_index] /= col_factor

    for (row_index, col_index), value in preserved.items():
        shaped[row_index, col_index] = value
    return shaped


def row_to_record(
    row,
    contact_values,
    baseline_values,
    mode,
    row_sensitivity=1.0,
    column_sensitivity=4.0,
    column_active_threshold=0.35,
    max_dampened_columns=3,
):
    if mode == "raw":
        heatmap_values = contact_values
    elif mode == "max-decrease":
        decrease = baseline_values - contact_values
        heatmap_values = decrease.copy()
        finite = np.isfinite(decrease)
        if np.any(finite):
            side = NUM_ELECTRODES // 2
            heatmap_grid = heatmap_values.reshape(side, side)
            winners = active_column_winners(
                heatmap_grid,
                threshold=column_active_threshold,
                max_columns=max_dampened_columns,
            )
            if winners:
                heatmap_values = attenuate_active_columns(
                    heatmap_grid,
                    winners,
                    row_factor=row_sensitivity,
                    col_factor=column_sensitivity,
                ).reshape(-1)
    elif mode == "percent":
        heatmap_values = 100.0 * (contact_values - baseline_values) / np.maximum(
            np.abs(baseline_values),
            EPS,
        )
    else:
        heatmap_values = contact_values - baseline_values

    return {
        "x": float(row["target_x_mm"]),
        "y": float(row["target_y_mm"]),
        "target_force": float(row["target_force_N"]) / FORCE_CALIBRATION_DIVISOR,
        "actual_force": float(row["actual_force_N"]) / FORCE_CALIBRATION_DIVISOR,
        "values": heatmap_values,
    }


def filter_records(records, min_force, max_force):
    filtered = []
    for record in records:
        force = record["actual_force"]
        if min_force is not None and force < min_force:
            continue
        if max_force is not None and force > max_force:
            continue
        filtered.append(record)
    return filtered


def build_location_index(records):
    locations = defaultdict(list)
    for idx, record in enumerate(records):
        locations[(record["x"], record["y"])].append(idx)
    return dict(locations)


def average_records_by_target_force(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["target_force"]].append(record)

    averaged = []
    for target_force in sorted(groups):
        group = groups[target_force]
        values = np.mean([r["values"] for r in group], axis=0)
        actual_force = float(np.mean([r["actual_force"] for r in group]))
        averaged.append(
            {
                "x": group[0]["x"],
                "y": group[0]["y"],
                "target_force": target_force,
                "actual_force": actual_force,
                "values": values,
                "n": len(group),
            }
        )
    return averaged


def color_limits(records, positive_only=False):
    if not records:
        return (0.0, 1.0) if positive_only else (-1.0, 1.0)
    all_values = np.concatenate([np.asarray(r["values"], dtype=float) for r in records])
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return (0.0, 1.0) if positive_only else (-1.0, 1.0)
    if positive_only:
        vmax = float(np.nanpercentile(finite, 98))
        if vmax <= EPS:
            vmax = 1.0
        return 0.0, vmax

    limit = float(np.nanpercentile(np.abs(finite), 98))
    if limit <= EPS:
        limit = 1.0
    return -limit, limit


def make_filter_description(min_force, max_force):
    parts = []
    if min_force is not None:
        parts.append(f">= {min_force:.2f} N")
    if max_force is not None:
        parts.append(f"<= {max_force:.2f} N")
    return " and ".join(parts)


def estimate_baseline_noise(df, eit_columns, records=None, noise_max_force=0.05):
    if "row_type" not in df.columns:
        baseline_values = None
    else:
        baseline_rows = df[df["row_type"].astype(str).str.lower() == "baseline"]
        baseline_values = baseline_rows[eit_columns].astype(float).to_numpy()

    if baseline_values is None or baseline_values.shape[0] < 2:
        if records is None:
            return np.full(len(eit_columns), np.nan, dtype=float)

        low_force_values = [
            r["values"]
            for r in records
            if np.isfinite(r["actual_force"]) and r["actual_force"] <= noise_max_force
        ]
        if len(low_force_values) < 2:
            return np.full(len(eit_columns), np.nan, dtype=float)
        baseline_values = np.asarray(low_force_values, dtype=float)

    noise = np.nanstd(baseline_values, axis=0, ddof=1)
    return np.maximum(noise, EPS)


def group_records_by_force(records, force_column, force_round):
    groups = defaultdict(list)
    force_key = "actual_force" if force_column == "actual" else "target_force"
    for record in records:
        force = record[force_key]
        if force_round is not None:
            force = round(force, force_round)
        groups[force].append(record)
    return [(force, groups[force]) for force in sorted(groups)]


def force_group_stats(force_groups, baseline_noise, z_threshold):
    noise_rms = float(np.sqrt(np.nanmean(np.square(baseline_noise))))
    if not np.isfinite(noise_rms) or noise_rms <= EPS:
        noise_rms = float("nan")

    stats = []
    for force, group in force_groups:
        values = np.asarray([r["values"] for r in group], dtype=float)
        mean_signal = np.nanmean(values, axis=0)
        std_signal = np.nanstd(values, axis=0, ddof=1) if len(group) > 1 else np.zeros_like(mean_signal)

        rms = float(np.sqrt(np.nanmean(np.square(mean_signal))))
        peak_abs = float(np.nanmax(np.abs(mean_signal)))
        l2_norm = float(np.linalg.norm(np.nan_to_num(mean_signal)))

        if np.isfinite(noise_rms) and noise_rms > EPS:
            snr = rms / noise_rms
            snr_db = 20.0 * math.log10(max(snr, EPS))
        else:
            snr = float("nan")
            snr_db = float("nan")

        z_scores = np.abs(mean_signal) / baseline_noise
        finite_z = z_scores[np.isfinite(z_scores)]
        if finite_z.size:
            max_abs_z = float(np.nanmax(finite_z))
            active_fraction = float(np.mean(finite_z >= z_threshold))
        else:
            max_abs_z = float("nan")
            active_fraction = float("nan")

        actual_force = float(np.nanmean([r["actual_force"] for r in group]))
        target_force = float(np.nanmean([r["target_force"] for r in group]))
        locations = sorted({(r["x"], r["y"]) for r in group})
        stats.append(
            {
                "force": float(force),
                "actual_force": actual_force,
                "target_force": target_force,
                "mean_signal": mean_signal,
                "std_signal": std_signal,
                "rms": rms,
                "peak_abs": peak_abs,
                "l2_norm": l2_norm,
                "snr": snr,
                "snr_db": snr_db,
                "max_abs_z": max_abs_z,
                "active_fraction": active_fraction,
                "n": len(group),
                "locations": locations,
            }
        )
    return stats


def first_significant_force(stats, snr_threshold_db, z_threshold):
    for item in stats:
        snr_ok = np.isfinite(item["snr_db"]) and item["snr_db"] >= snr_threshold_db
        z_ok = np.isfinite(item["max_abs_z"]) and item["max_abs_z"] >= z_threshold
        if snr_ok and z_ok:
            return item
    return None


def plot_signal_browser(
    records,
    df,
    eit_columns,
    labels,
    mode,
    min_force,
    max_force,
    snr_threshold_db,
    z_threshold,
    signal_force_column,
    signal_force_round,
    noise_max_force,
):
    baseline_noise = estimate_baseline_noise(
        df,
        eit_columns,
        records=records,
        noise_max_force=noise_max_force,
    )
    force_groups = group_records_by_force(records, signal_force_column, signal_force_round)
    stats = force_group_stats(force_groups, baseline_noise, z_threshold)
    significant = first_significant_force(stats, snr_threshold_db, z_threshold)

    selected_idx = 0
    if is_taxel_grid(len(eit_columns)):
        channel_labels = taxel_labels(len(eit_columns))
        channel_axis_label = "Taxel / channel"
    else:
        channel_labels = [
            f"{a}-{b}" if b is not None else str(a)
            for a, b in labels
        ]
        channel_axis_label = "Electrode pair / channel"
    x = np.arange(len(channel_labels))

    fig = plt.figure(figsize=(14, 9))

    def metric_arrays(key):
        return np.asarray([item[key] for item in stats], dtype=float)

    forces = metric_arrays("force")
    force_axis_label = (
        "Actual force (N)" if signal_force_column == "actual" else "Target force (N)"
    )

    def plot_current_force():
        nonlocal selected_idx
        fig.clear()
        item = stats[selected_idx]

        ax_signal = fig.add_subplot(2, 2, 1)
        ax_snr = fig.add_subplot(2, 2, 2)
        ax_response = fig.add_subplot(2, 2, 3)
        ax_z = fig.add_subplot(2, 2, 4)

        mean_signal = item["mean_signal"]
        std_signal = item["std_signal"]
        ax_signal.plot(x, mean_signal, marker="o", linewidth=1.6, label="mean signal")
        if item["n"] > 1:
            ax_signal.fill_between(
                x,
                mean_signal - std_signal,
                mean_signal + std_signal,
                alpha=0.18,
                label="1 SD across repeats",
            )
        ax_signal.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax_signal.set_xticks(x)
        ax_signal.set_xticklabels(channel_labels, rotation=45, ha="right", fontsize=8)
        ax_signal.set_xlabel(channel_axis_label)
        ax_signal.set_ylabel(mode)
        ax_signal.set_title(
            f"Signal at {signal_force_column} {item['force']:.3g} N "
            f"(target {item['target_force']:.2f} N, "
            f"actual {item['actual_force']:.2f} N, n={item['n']})"
        )
        ax_signal.grid(True, alpha=0.25)
        ax_signal.legend(loc="best", fontsize=8)

        ax_snr.plot(forces, metric_arrays("snr_db"), marker="o", label="SNR (dB)")
        ax_snr.axhline(
            snr_threshold_db,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"{snr_threshold_db:.1f} dB threshold",
        )
        ax_snr.axvline(item["force"], color="black", alpha=0.25)
        ax_snr.set_xlabel(force_axis_label)
        ax_snr.set_ylabel("SNR (dB)")
        ax_snr.set_title("Signal-to-noise ratio")
        ax_snr.grid(True, alpha=0.25)
        ax_snr.legend(loc="best", fontsize=8)

        ax_response.plot(forces, metric_arrays("rms"), marker="o", label="RMS response")
        ax_response.plot(forces, metric_arrays("peak_abs"), marker="s", label="Peak abs response")
        ax_response.plot(forces, metric_arrays("l2_norm"), marker="^", label="L2 norm")
        ax_response.axvline(item["force"], color="black", alpha=0.25)
        ax_response.set_xlabel(force_axis_label)
        ax_response.set_ylabel(mode)
        ax_response.set_title("Response magnitude")
        ax_response.grid(True, alpha=0.25)
        ax_response.legend(loc="best", fontsize=8)

        ax_z.plot(forces, metric_arrays("max_abs_z"), marker="o", label="Max channel z")
        ax_z.axhline(
            z_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"{z_threshold:.1f} sigma threshold",
        )
        ax_z_twin = ax_z.twinx()
        ax_z_twin.plot(
            forces,
            100.0 * metric_arrays("active_fraction"),
            color="tab:green",
            marker="s",
            label="Active channels",
        )
        ax_z.axvline(item["force"], color="black", alpha=0.25)
        ax_z.set_xlabel(force_axis_label)
        ax_z.set_ylabel("Max abs z-score")
        ax_z_twin.set_ylabel("Active channels (%)")
        ax_z.set_title("Channel significance")
        ax_z.grid(True, alpha=0.25)
        lines, names = ax_z.get_legend_handles_labels()
        twin_lines, twin_names = ax_z_twin.get_legend_handles_labels()
        ax_z.legend(lines + twin_lines, names + twin_names, loc="best", fontsize=8)

        locations = item["locations"]
        if locations:
            first_location = locations[0]
            location_text = (
                f"location shown only: x={first_location[0]:.2f} mm, "
                f"y={first_location[1]:.2f} mm"
            )
            if len(locations) > 1:
                location_text += f" (+{len(locations) - 1} more)"
        else:
            location_text = "location shown only: unavailable"

        filter_desc = make_filter_description(min_force, max_force)
        title = (
            f"2-Probe Signal Metrics - Force {selected_idx + 1}/{len(stats)} - "
            f"{location_text}"
        )
        if significant:
            title += (
                f"\nfirst significant force: {signal_force_column} "
                f"{significant['force']:.3g} N, "
                f"target {significant['target_force']:.2f} N, "
                f"actual {significant['actual_force']:.2f} N "
                f"(SNR {significant['snr_db']:.1f} dB, "
                f"max z {significant['max_abs_z']:.1f})"
            )
        else:
            title += "\nno force crossed the configured SNR/z thresholds"
        if filter_desc:
            title += f" - filter {filter_desc}"

        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal selected_idx
        if event.key in ("right", "up") and selected_idx < len(stats) - 1:
            selected_idx += 1
            plot_current_force()
        elif event.key in ("left", "down") and selected_idx > 0:
            selected_idx -= 1
            plot_current_force()
        elif event.key == "home":
            selected_idx = 0
            plot_current_force()
        elif event.key == "end":
            selected_idx = len(stats) - 1
            plot_current_force()

    plot_current_force()
    fig.canvas.mpl_connect("key_press_event", on_key)

    print()
    print("Signal metric summary:")
    print(
        f"  stepping by {signal_force_column}_force_N "
        f"(rounded to {signal_force_round} decimals)"
    )
    print("  force_N  target_N  actual_N  snr_dB  rms  peak_abs  max_z  active_%  n")
    for item in stats:
        print(
            "  "
            f"{item['force']:7.3f}  "
            f"{item['target_force']:8.2f}  "
            f"{item['actual_force']:8.2f}  "
            f"{item['snr_db']:6.1f}  "
            f"{item['rms']:.4g}  "
            f"{item['peak_abs']:.4g}  "
            f"{item['max_abs_z']:5.1f}  "
            f"{100.0 * item['active_fraction']:8.1f}  "
            f"{item['n']}"
        )
    if significant:
        print(
            "  First significant force: "
            f"{signal_force_column} {significant['force']:.2f} N, "
            f"target {significant['target_force']:.2f} N, "
            f"actual {significant['actual_force']:.2f} N "
            f"(SNR {significant['snr_db']:.1f} dB, "
            f"max z {significant['max_abs_z']:.1f})"
        )
    else:
        print("  No force crossed the configured SNR/z thresholds.")
    print("Controls:")
    print("  Right/Up   : increase force")
    print("  Left/Down  : decrease force")
    print("  Home       : lowest force")
    print("  End        : highest force")
    print()

    plt.show()


def main():
    args = parse_args()
    if args.row_sensitivity < 1.0:
        raise ValueError("--row-sensitivity must be >= 1.0")
    if args.column_sensitivity < 1.0:
        raise ValueError("--column-sensitivity must be >= 1.0")
    if not 0.0 < args.column_active_threshold <= 1.0:
        raise ValueError("--column-active-threshold must be > 0 and <= 1")
    if args.max_dampened_columns < 0:
        raise ValueError("--max-dampened-columns must be >= 0")

    try:
        df, eit_columns, pair_labels = load_data(
            args.csv_file,
            average_duplicate_pairs=not args.no_pair_average,
            condensed_csv_file=args.condensed_csv_file,
        )
        records = make_contact_records(
            df,
            eit_columns,
            args.mode,
            row_sensitivity=args.row_sensitivity,
            column_sensitivity=args.column_sensitivity,
            column_active_threshold=args.column_active_threshold,
            max_dampened_columns=args.max_dampened_columns,
        )
        records = filter_records(records, args.min_force, args.max_force)
    except Exception as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("No contact measurements found after filtering.")
        return 1

    labels = infer_pair_labels(eit_columns, pair_labels)

    if args.plot_signal:
        plot_signal_browser(
            records,
            df,
            eit_columns,
            labels,
            args.mode,
            args.min_force,
            args.max_force,
            args.snr_threshold_db,
            args.z_threshold,
            args.signal_force_column,
            args.signal_force_round,
            args.noise_max_force,
        )
        return 0

    location_index = build_location_index(records)
    location_keys = sorted(location_index)
    location_idx = 0

    xlabels, ylabels, xlabel, ylabel = axis_labels(len(eit_columns), labels)
    positive_heatmap = args.mode == "max-decrease"
    vmin, vmax = color_limits(records, positive_only=positive_heatmap)
    cmap = "viridis" if positive_heatmap else "RdBu_r"

    fig = plt.figure(figsize=(13, 8))

    def plot_location():
        fig.clear()
        location = location_keys[location_idx]
        location_records = [records[i] for i in location_index[location]]
        heatmaps = average_records_by_target_force(location_records)
        heatmaps = heatmaps[: args.max_heatmaps]

        if not heatmaps:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No measurements for this location", ha="center", va="center")
            ax.axis("off")
            fig.canvas.draw_idle()
            return

        n = len(heatmaps)
        cols = min(4, n)
        rows = int(math.ceil(n / cols))
        image = None

        for idx, record in enumerate(heatmaps):
            ax = fig.add_subplot(rows, cols, idx + 1)
            grid = render_grid(record["values"])
            edge_x, edge_y = trapezium_edge_mesh(*grid.shape)
            image = ax.pcolormesh(
                edge_x,
                edge_y,
                grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="flat",
            )
            ax.set_title(
                f"target {record['target_force']:.2f} N\n"
                f"actual {record['actual_force']:.2f} N, n={record['n']}",
                fontsize=9,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0.0, TRAPEZIUM_WIDTH)
            ax.set_ylim(-TRAPEZIUM_LEFT_HEIGHT * 0.55, TRAPEZIUM_LEFT_HEIGHT * 0.55)
            ax.set_xticks(np.linspace(
                TRAPEZIUM_WIDTH / (2 * grid.shape[1]),
                TRAPEZIUM_WIDTH - TRAPEZIUM_WIDTH / (2 * grid.shape[1]),
                len(xlabels),
            ))
            ax.set_yticks(np.linspace(
                -TRAPEZIUM_LEFT_HEIGHT * 0.5 + TRAPEZIUM_LEFT_HEIGHT / (2 * grid.shape[0]),
                TRAPEZIUM_LEFT_HEIGHT * 0.5 - TRAPEZIUM_LEFT_HEIGHT / (2 * grid.shape[0]),
                len(ylabels),
            ))
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(ylabels, fontsize=8)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)

        for idx in range(n, rows * cols):
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.axis("off")

        if image is not None:
            fig.colorbar(image, ax=fig.axes, shrink=0.82, label=args.mode)

        filter_desc = make_filter_description(args.min_force, args.max_force)
        title = (
            f"2-Probe Heatmaps - Location {location_idx + 1}/{len(location_keys)} "
            f"x={location[0]:.2f} mm, y={location[1]:.2f} mm"
        )
        if filter_desc:
            title += f" ({filter_desc})"
        if len(average_records_by_target_force(location_records)) > args.max_heatmaps:
            title += f" - first {args.max_heatmaps} force groups shown"

        fig.suptitle(title, fontsize=13)
        fig.subplots_adjust(
            left=0.06,
            right=0.86,
            bottom=0.08,
            top=0.88,
            wspace=0.45,
            hspace=0.65,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal location_idx
        if event.key == "right" and location_idx < len(location_keys) - 1:
            location_idx += 1
            plot_location()
        elif event.key == "left" and location_idx > 0:
            location_idx -= 1
            plot_location()
        elif event.key == "home":
            location_idx = 0
            plot_location()
        elif event.key == "end":
            location_idx = len(location_keys) - 1
            plot_location()

    plot_location()
    fig.canvas.mpl_connect("key_press_event", on_key)

    print()
    print("Controls:")
    print("  Left/Right : previous/next location")
    print("  Home       : first location")
    print("  End        : last location")
    print()

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
