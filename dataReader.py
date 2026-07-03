import argparse
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
import pyeit.eit.bp as bp


NUM_ELECTRODES = 16
MESH_ELEMENT_SIZE = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View overlaid EIT signals for each location."
    )

    parser.add_argument(
        "--csv_file",
        default="RawData/Press_1027/2026-06-12_12-59-23_6Forces_40x10.csv",
        help="Path to CSV file.",
    )

    parser.add_argument(
        "--min_force",
        type=float,
        default=None,
        help="Only display signals with actual_force_N >= min_force.",
    )

    parser.add_argument(
        "--max_force",
        type=float,
        default=None,
        help="Only display signals with actual_force_N <= max_force.",
    )

    parser.add_argument(
        "--no_pair_average",
        action="store_true",
        help="Keep the original eit_n columns instead of averaging duplicate electrode pairs.",
    )

    parser.add_argument(
        "--condensed_csv_file",
        default=None,
        help="Optional path to write the pair-averaged CSV.",
    )

    return parser.parse_args()


def find_eit_columns(columns):
    pattern = re.compile(r"^eit_(\d+)$")

    matches = []
    for col in columns:
        m = pattern.match(col)
        if m:
            matches.append((int(m.group(1)), col))

    if not matches:
        raise ValueError("No eit_n columns found.")

    return [col for _, col in sorted(matches)]


def sort_rows_by_xy_then_force(
    df,
    x_col="target_x_mm",
    y_col="target_y_mm",
    force_col="actual_force_N",
):
    return (
        df.sort_values(
            [x_col, y_col, force_col],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


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
    for col in averaged_columns:
        pair = pair_labels[col]
        print(f"{col}: pair=({pair[0]}, {pair[1]}) from {len(pair_to_columns[pair])} channels")

    return condensed, averaged_columns, pair_labels


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

    # Global baseline: first row of the CSV (before any per-location sorting)
    global_baseline = df.iloc[0][eit_columns].astype(float).values
    # Remove baseline row from dataframe
    df = df.iloc[1:].reset_index(drop=True)
    df = sort_rows_by_xy_then_force(df)
    print("Rows sorted by target_x_mm, target_y_mm, actual_force_N.")
    eit_columns = find_eit_columns(df.columns)

    return df, eit_columns, global_baseline, pair_labels


def build_location_index(df):
    """
    Group rows by spatial location only.
    Each location contains all force measurements.
    """
    locations = {}

    for idx, row in df.iterrows():
        key = (row["target_x_mm"], row["target_y_mm"])

        if key not in locations:
            locations[key] = []

        locations[key].append(idx)

    return locations


def make_filter_description(min_force, max_force):
    parts = []

    if min_force is not None:
        parts.append(f"≥ {min_force:.2f} N")

    if max_force is not None:
        parts.append(f"≤ {max_force:.2f} N")

    return " and ".join(parts)


def setup_eit_solver():
    """Build mesh, protocol and BP solver once at startup."""
    mesh_obj = mesh.create(NUM_ELECTRODES, h0=MESH_ELEMENT_SIZE)
    protocol_obj = protocol.create(
        NUM_ELECTRODES,
        dist_exc=1,
        step_meas=1,
        parser_meas="rotate_meas",
    )
    solver = bp.BP(mesh_obj, protocol_obj)
    solver.setup(weight="none")
    return mesh_obj, protocol_obj, solver


def print_protocol_pair_summary(channel_pairs):
    pair_to_channels = defaultdict(list)
    for channel, pair in enumerate(channel_pairs):
        pair_to_channels[pair].append(channel)

    print("Unique read pairs in protocol:")
    for pair in sorted(pair_to_channels, key=pair_sort_key):
        channels = pair_to_channels[pair]
        print(
            f"pair=({pair[0]}, {pair[1]}): "
            f"{len(channels)} channels "
            f"(eit_{channels[0]}..eit_{channels[-1]})"
        )


def field_to_face(values, pts, tri):
    """Convert node-based or element-based field to per-face values."""
    n_pts = pts.shape[0]
    n_tri = tri.shape[0]
    if values.shape[0] == n_pts:
        return values[tri].mean(axis=1)
    elif values.shape[0] == n_tri:
        return values
    else:
        raise RuntimeError(
            f"Unexpected field size: {values.shape[0]}, "
            f"expected {n_pts} (per-vertex) or {n_tri} (per-face)"
        )


def main():
    args = parse_args()

    channel_pairs = build_channel_pair_map()
    print_protocol_pair_summary(channel_pairs)

    try:
        df, eit_columns, global_baseline, pair_labels = load_data(
            args.csv_file,
            average_duplicate_pairs=not args.no_pair_average,
            condensed_csv_file=args.condensed_csv_file,
        )
    except Exception as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 1

    if len(df) == 0:
        print("CSV contains no rows.")
        return 1

    locations = build_location_index(df)
    location_keys = sorted(locations.keys())

    if not location_keys:
        print("No valid locations found.")
        return 1

    channel_indices = [
        int(re.search(r"(\d+)$", col).group(1))
        for col in eit_columns
    ]
    channel_labels = [
        f"{pair_labels[col][0]}-{pair_labels[col][1]}"
        if pair_labels and col in pair_labels
        else str(channel)
        for channel, col in zip(channel_indices, eit_columns)
    ]

    # --- EIT solver setup ---
    print("Setting up EIT solver...")
    mesh_obj, protocol_obj, eit_solver = setup_eit_solver()
    pts = mesh_obj.node
    tri = mesh_obj.element

    # Ensure no zeros in baseline
    safe_baseline = np.array(
        [x if x > 1e-12 else 1e-12 for x in global_baseline]
    )

    # --- State ---
    location_idx = 0
    reconstruction_mode = False  # False = signal plot, True = EIT reconstruction

    fig, ax = plt.subplots(figsize=(12, 8))
    # Extra axes for reconstruction (hidden initially)
    recon_ax = fig.add_axes(ax.get_position(), sharey=None)
    recon_ax.set_visible(False)

    # Colorbar handle so we can remove/redraw it
    _cbar_obj = [None]

    def get_filtered_signals(location_idx):
        location = location_keys[location_idx]
        row_indices = locations[location]

        filtered = []
        for idx in row_indices:
            force = float(df.iloc[idx]["actual_force_N"])
            if args.min_force is not None and force < args.min_force:
                continue
            if args.max_force is not None and force > args.max_force:
                continue
            filtered.append(idx)

        signals = []
        for row_idx in filtered:
            row = df.iloc[row_idx]
            force = float(row["actual_force_N"])
            values = row[eit_columns].astype(float).values
            signals.append((force, values))

        signals.sort(key=lambda x: x[0])
        return signals

    def plot_signals(signals, x_mm, y_mm):
        ax.set_visible(True)
        recon_ax.set_visible(False)

        # Remove old colorbar if present
        if _cbar_obj[0] is not None:
            _cbar_obj[0].remove()
            _cbar_obj[0] = None

        ax.clear()

        if not signals:
            filter_desc = make_filter_description(args.min_force, args.max_force)
            ax.text(
                0.5, 0.5,
                f"No measurements satisfy\n{filter_desc or 'current filter'}",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=12,
            )
        else:
            baseline_subtracted = [
                (f, v - global_baseline) for f, v in signals
            ]
            all_vals = np.concatenate([v for _, v in baseline_subtracted])
            global_min, global_max = all_vals.min(), all_vals.max()
            pad = 0.05 * (global_max - global_min) or 1.0

            cmap = plt.get_cmap("viridis")
            colours = cmap(np.linspace(0, 1, max(len(signals), 2)))

            for colour, (force, values) in zip(colours, baseline_subtracted):
                ax.plot(
                    channel_indices, values,
                    color=colour, linewidth=1.5, label=f"{force:.2f} N",
                )

            ax.set_ylim(global_min - pad, global_max + pad)
            ax.set_xlabel("EIT Channel")
            ax.set_ylabel("Signal (baseline-subtracted)")
            ax.set_xticks(channel_indices)
            ax.set_xticklabels(channel_labels, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)
            ax.legend(
                title="Force",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
            )

        filter_desc = make_filter_description(args.min_force, args.max_force)
        title = f"x={x_mm:.1f} mm, y={y_mm:.1f} mm\n{len(signals)} force levels shown"
        if filter_desc:
            title += f" ({filter_desc})"
        ax.set_title(title)

    def plot_reconstruction(signals, x_mm, y_mm):
        ax.set_visible(False)
        recon_ax.set_visible(True)
        recon_ax.set_position(ax.get_position())

        # Remove old colorbar if present
        if _cbar_obj[0] is not None:
            _cbar_obj[0].remove()
            _cbar_obj[0] = None

        recon_ax.clear()

        if not signals:
            filter_desc = make_filter_description(args.min_force, args.max_force)
            recon_ax.text(
                0.5, 0.5,
                f"No measurements satisfy\n{filter_desc or 'current filter'}",
                ha="center", va="center",
                transform=recon_ax.transAxes, fontsize=12,
            )
        else:
            expected_measurements = sum(len(group) for group in protocol_obj.meas_mat)
            if len(signals[0][1]) != expected_measurements:
                recon_ax.text(
                    0.5, 0.5,
                    "Reconstruction needs the full protocol channel count.\n"
                    f"Loaded {len(signals[0][1])} averaged pair channels; "
                    f"solver expects {expected_measurements}.",
                    ha="center", va="center",
                    transform=recon_ax.transAxes, fontsize=12,
                )
                recon_ax.set_title("Reconstruction unavailable for pair-averaged data")
                fig.canvas.draw_idle()
                return

            # Average all signals at this location into one reconstruction
            all_frames = np.array([v for _, v in signals])
            mean_frame = np.mean(all_frames, axis=0)
            safe_frame = np.array(
                [x if x > 1e-12 else 1e-12 for x in mean_frame]
            )

            ds = eit_solver.solve(safe_frame, safe_baseline, normalize=True)
            vals = np.real(ds)
            vals_face = field_to_face(vals, pts, tri)

            # Build matplotlib triangulation from pyeit mesh nodes
            triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)

            # Map face values to node values for tripcolor (or use tripcolor with tri directly)
            v_abs = max(np.max(np.abs(vals_face)), 1e-12)
            tcf = recon_ax.tripcolor(
                triang,
                facecolors=vals_face,
                cmap="RdBu_r",
                vmin=-v_abs,
                vmax=v_abs,
                shading="flat",
            )
            _cbar_obj[0] = fig.colorbar(tcf, ax=recon_ax, label="Δσ (normalised)")

            # Draw electrode positions as dots
            n = NUM_ELECTRODES
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            el_x = np.cos(angles)
            el_y = np.sin(angles)
            recon_ax.scatter(el_x, el_y, c="white", s=30, zorder=5)

            recon_ax.set_aspect("equal")
            recon_ax.axis("off")

        filter_desc = make_filter_description(args.min_force, args.max_force)
        title = (
            f"EIT Reconstruction — x={x_mm:.1f} mm, y={y_mm:.1f} mm\n"
            f"{len(signals)} force levels averaged"
        )
        if filter_desc:
            title += f" ({filter_desc})"
        recon_ax.set_title(title)

    def update_plot():
        location = location_keys[location_idx]
        x_mm, y_mm = location
        signals = get_filtered_signals(location_idx)

        if reconstruction_mode:
            plot_reconstruction(signals, x_mm, y_mm)
        else:
            plot_signals(signals, x_mm, y_mm)

        fig.suptitle(
            f"Location {location_idx + 1}/{len(location_keys)}"
            + ("  [R: signal view]" if reconstruction_mode else "  [R: reconstruction view]"),
            fontsize=14,
        )

        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal location_idx, reconstruction_mode

        if event.key == "right":
            if location_idx < len(location_keys) - 1:
                location_idx += 1
                update_plot()

        elif event.key == "left":
            if location_idx > 0:
                location_idx -= 1
                update_plot()

        elif event.key == "home":
            location_idx = 0
            update_plot()

        elif event.key == "end":
            location_idx = len(location_keys) - 1
            update_plot()

        elif event.key == "r":
            reconstruction_mode = not reconstruction_mode
            update_plot()

    update_plot()

    fig.canvas.mpl_connect("key_press_event", on_key)

    print()
    print("Controls:")
    print("  Left/Right : previous/next location")
    print("  Home       : first location")
    print("  End        : last location")
    print("  R          : toggle signal / EIT reconstruction view")

    filter_desc = make_filter_description(args.min_force, args.max_force)
    if filter_desc:
        print(f"  Force filter: {filter_desc}")

    print()

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
