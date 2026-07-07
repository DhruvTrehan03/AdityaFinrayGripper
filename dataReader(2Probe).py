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
        choices=("delta", "percent", "raw"),
        default="delta",
        help="Heatmap value: contact-baseline delta, percent change, or raw contact.",
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


def axis_labels(n_channels, labels):
    rows, cols = heatmap_shape(n_channels)
    if n_channels == (NUM_ELECTRODES // 2) ** 2:
        odd, even, _ = odd_even_pair_labels()
        return [str(x) for x in even], [str(x) for x in odd], "Even electrode B/N", "Odd electrode A/M"

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


def make_contact_records(df, eit_columns, mode):
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
                records.append(row_to_record(row, values, baseline, mode))
    else:
        if global_baseline is None:
            global_baseline = df.iloc[0][eit_columns].astype(float).values
        for _, row in df.iloc[1:].iterrows():
            values = row[eit_columns].astype(float).values
            records.append(row_to_record(row, values, global_baseline, mode))

    return records


def row_to_record(row, contact_values, baseline_values, mode):
    if mode == "raw":
        heatmap_values = contact_values
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
        "target_force": float(row["target_force_N"]),
        "actual_force": float(row["actual_force_N"]),
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


def color_limits(records):
    if not records:
        return -1.0, 1.0
    all_values = np.concatenate([np.asarray(r["values"], dtype=float) for r in records])
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return -1.0, 1.0
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


def main():
    args = parse_args()

    try:
        df, eit_columns, pair_labels = load_data(
            args.csv_file,
            average_duplicate_pairs=not args.no_pair_average,
            condensed_csv_file=args.condensed_csv_file,
        )
        records = make_contact_records(df, eit_columns, args.mode)
        records = filter_records(records, args.min_force, args.max_force)
    except Exception as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("No contact measurements found after filtering.")
        return 1

    labels = infer_pair_labels(eit_columns, pair_labels)
    location_index = build_location_index(records)
    location_keys = sorted(location_index)
    location_idx = 0

    xlabels, ylabels, xlabel, ylabel = axis_labels(len(eit_columns), labels)
    vmin, vmax = color_limits(records)

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
            grid = values_to_grid(record["values"])
            image = ax.imshow(grid, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(
                f"target {record['target_force']:.2f} N\n"
                f"actual {record['actual_force']:.2f} N, n={record['n']}",
                fontsize=9,
            )
            ax.set_xticks(range(len(xlabels)))
            ax.set_yticks(range(len(ylabels)))
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
