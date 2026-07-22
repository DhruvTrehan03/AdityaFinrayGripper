import argparse
import os
import queue
import sys
import threading
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import serial
from vispy import app, color, scene

from two_probe_amodo_eit import NUM_ELECTRODES, TwoProbeAmodoEITDevice
from utils import print_error, print_info, print_warning


APP_NAME = "Two-Probe Amodo Taxel Live View"
EPS = 1e-12
TRAPEZIUM_LEFT_HEIGHT = 20.0
TRAPEZIUM_RIGHT_HEIGHT = 15.0
TRAPEZIUM_WIDTH = 60.0


class MockTwoProbeSource(threading.Thread):
    """Synthetic odd/even two-probe source for previewing the UI."""

    def __init__(self, q_out, num_electrodes=NUM_ELECTRODES):
        super().__init__(name="MockTwoProbeSource", daemon=True)
        self.q = q_out
        self.num_electrodes = num_electrodes
        self.stop_evt = threading.Event()
        self.start_time = time.monotonic()
        odd_electrodes, even_electrodes, _ = build_pair_labels(num_electrodes)
        self.n_rows = len(odd_electrodes)
        self.n_cols = len(even_electrodes)

    def _frame(self):
        t = time.monotonic() - self.start_time
        rows = np.arange(self.n_rows, dtype=float)[:, None]
        cols = np.arange(self.n_cols, dtype=float)[None, :]
        centre_row = (self.n_rows - 1) * (0.5 + 0.35 * np.sin(t * 0.8))
        centre_col = (self.n_cols - 1) * (0.5 + 0.35 * np.cos(t * 0.6))
        bump = np.exp(-((rows - centre_row) ** 2 + (cols - centre_col) ** 2) / 5.0)
        ripple = 0.03 * np.sin(rows * 1.7 + cols * 0.9 + t * 3.0)
        return (0.15 + 0.08 * bump + ripple).reshape(-1)

    def run(self):
        baseline = self._frame()
        self.q.put(
            {
                "t": time.monotonic(),
                "readings": baseline.tolist(),
                "clipping": False,
                "baseline": True,
            }
        )
        while not self.stop_evt.is_set():
            frame = self._frame()
            sample = {
                "t": time.monotonic(),
                "readings": frame.tolist(),
                "clipping": False,
            }
            try:
                self.q.put_nowait(sample)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put_nowait(sample)
            time.sleep(1.0 / 60.0)

    def stop(self):
        self.stop_evt.set()


def build_pair_labels(num_electrodes):
    """Match TwoProbeAmodoEITDevice._build_two_probe_configurations order."""
    odd_electrodes = list(range(1, num_electrodes + 1, 2))
    even_electrodes = list(range(2, num_electrodes + 1, 2))
    pair_labels = [
        (odd, even)
        for odd in odd_electrodes
        for even in even_electrodes
    ]
    return odd_electrodes, even_electrodes, pair_labels


def safe_frame(readings):
    frame = np.asarray(readings, dtype=float)
    return np.maximum(frame, EPS)


def readings_to_grid(readings, num_electrodes):
    odd_electrodes, even_electrodes, _ = build_pair_labels(num_electrodes)
    expected = len(odd_electrodes) * len(even_electrodes)
    values = safe_frame(readings)
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} two-probe readings for {num_electrodes} electrodes, "
            f"got {values.size}."
        )
    return values.reshape(len(odd_electrodes), len(even_electrodes))


def render_grid(grid):
    return np.flipud(grid)


def taxel_layout_description(num_electrodes):
    cols = num_electrodes // 2
    return (
        "trapezium left height 20, right height 15, center aligned; "
        f"display columns 1-{cols}"
    )


def trapezium_vertices(
    rows,
    cols,
    left_height=TRAPEZIUM_LEFT_HEIGHT,
    right_height=TRAPEZIUM_RIGHT_HEIGHT,
    width=TRAPEZIUM_WIDTH,
):
    vertices = []
    for col_index in range(cols + 1):
        t = col_index / cols if cols else 0.0
        height = left_height + (right_height - left_height) * t
        bottom = -0.5 * height
        top = 0.5 * height
        x = width * t
        for row_index in range(rows + 1):
            y = bottom + height * row_index / rows if rows else 0.0
            vertices.append((float(x), float(y), 0.0))
    return np.asarray(vertices, dtype=np.float32)


def trapezium_faces(rows, cols):
    faces = []
    for col_index in range(cols):
        for row_index in range(rows):
            lower_left = col_index * (rows + 1) + row_index
            upper_left = lower_left + 1
            lower_right = (col_index + 1) * (rows + 1) + row_index
            upper_right = lower_right + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    return np.asarray(faces, dtype=np.uint32)


def trapezium_grid_lines(
    rows,
    cols,
    left_height=TRAPEZIUM_LEFT_HEIGHT,
    right_height=TRAPEZIUM_RIGHT_HEIGHT,
    width=TRAPEZIUM_WIDTH,
):
    segments = []
    for col_index in range(cols + 1):
        t = col_index / cols if cols else 0.0
        height = left_height + (right_height - left_height) * t
        bottom = -0.5 * height
        top = 0.5 * height
        x = width * t
        segments.extend([(float(x), bottom, 0.01), (float(x), top, 0.01)])

    for row_index in range(rows + 1):
        line = []
        for col_index in range(cols + 1):
            t = col_index / cols if cols else 0.0
            height = left_height + (right_height - left_height) * t
            bottom = -0.5 * height
            y = bottom + height * row_index / rows if rows else 0.0
            line.append((float(width * t), float(y), 0.01))
        for start, end in zip(line, line[1:]):
            segments.extend([start, end])
    return np.asarray(segments, dtype=np.float32)


def grid_face_colors(grid, cmap, clim):
    vmin, vmax = clim
    span = max(vmax - vmin, EPS)
    values = np.asarray(grid, dtype=float).T.reshape(-1)
    normalized = np.clip((values - vmin) / span, 0.0, 1.0)
    normalized[~np.isfinite(values)] = 0.0
    colors = cmap.map(normalized)
    colors[~np.isfinite(values), 3] = 0.0
    return np.repeat(colors, 2, axis=0)


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


def display_grid(
    frame_grid,
    baseline_grid,
    mode,
    row_sensitivity=1.0,
    column_sensitivity=6.0,
    column_active_threshold=0.35,
    max_dampened_columns=3,
):
    if mode == "raw":
        return frame_grid
    if mode == "max-decrease":
        decrease = baseline_grid - frame_grid
        winners = active_column_winners(
            decrease,
            threshold=column_active_threshold,
            max_columns=max_dampened_columns,
        )
        if winners:
            return attenuate_active_columns(
                decrease,
                winners,
                row_factor=row_sensitivity,
                col_factor=column_sensitivity,
            )
        return decrease
    if mode == "delta":
        return frame_grid - baseline_grid
    if mode == "percent":
        return 100.0 * (frame_grid - baseline_grid) / np.maximum(np.abs(baseline_grid), EPS)
    if mode == "abs-percent":
        return np.abs(
            100.0 * (frame_grid - baseline_grid) / np.maximum(np.abs(baseline_grid), EPS)
        )
    raise ValueError(f"Unknown display mode: {mode}")


def drain_latest(q_in):
    latest = None
    while True:
        try:
            latest = q_in.get_nowait()
        except queue.Empty:
            return latest


def color_limits(grid, symmetric):
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return (-2.0, 2.0) if symmetric else (0.0, 1.0)

    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 98))
        if limit <= EPS:
            limit = 1.0
        return -limit, limit

    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if vmin >= 0.0:
        vmin = 0.0
    if abs(vmax - vmin) <= EPS:
        pad = max(abs(vmax) * 0.05, 1.0)
        vmax += pad
    return vmin, vmax


def initial_color_limits(args, grid, symmetric):
    if args.scale_limit is not None:
        limit = abs(float(args.scale_limit))
        if limit <= EPS:
            raise ValueError("--scale-limit must be greater than zero.")
        return (-limit, limit) if symmetric else (0.0, limit)
    return color_limits(grid, symmetric)


def set_heatmap_data(mesh, cbar, grid, cmap, clim):
    rows, cols = grid.shape
    mesh.set_data(
        vertices=trapezium_vertices(rows, cols),
        faces=trapezium_faces(rows, cols),
        face_colors=grid_face_colors(grid, cmap, clim),
        color=(1.0, 1.0, 1.0, 1.0),
    )
    cbar.clim = clim


def make_source(args, q_eit):
    if args.demo:
        print_warning("Running in --demo mode (no board required).")
        return MockTwoProbeSource(q_eit, num_electrodes=args.num_electrodes)
    return TwoProbeAmodoEITDevice(q_eit, num_electrodes=args.num_electrodes)


def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the UI with synthetic data and no connected board.",
    )
    parser.add_argument(
        "--mode",
        choices=("percent", "abs-percent", "delta", "raw", "max-decrease"),
        default="percent",
        help="Quantity shown in the heatmap.",
    )
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="Keep the initial color scale instead of adapting it to incoming frames.",
    )
    parser.add_argument(
        "--scale-limit",
        type=float,
        default=None,
        help=(
            "Use an absolute heatmap color scale. Percent/delta modes use +/- this "
            "value; abs-percent/raw modes use 0 to this value. Also implies --fixed-scale."
        ),
    )
    parser.add_argument(
        "--row-sensitivity",
        type=float,
        default=1.0,
        help=(
            "In max-decrease mode, divide the rest of the winning row by this factor "
            "while preserving the winning taxel. Use values >1 to stop a whole row "
            "from lighting up."
        ),
    )
    parser.add_argument(
        "--column-sensitivity",
        type=float,
        default=6.0,
        help=(
            "In max-decrease mode, divide the rest of the winning column by this factor "
            "while preserving the winning taxel."
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
        "--num-electrodes",
        type=int,
        default=NUM_ELECTRODES,
        help="Number of electrodes in the two-probe configuration.",
    )
    args = parser.parse_args()
    if args.row_sensitivity < 1.0:
        parser.error("--row-sensitivity must be >= 1.0")
    if args.column_sensitivity < 1.0:
        parser.error("--column-sensitivity must be >= 1.0")
    if not 0.0 < args.column_active_threshold <= 1.0:
        parser.error("--column-active-threshold must be > 0 and <= 1")
    if args.max_dampened_columns < 0:
        parser.error("--max-dampened-columns must be >= 0")

    q_eit = queue.Queue(maxsize=4)
    reader = make_source(args, q_eit)
    reader.start()

    print_info("Waiting for baseline frame...")
    baseline_sample = None
    while baseline_sample is None:
        sample = q_eit.get()
        if sample.get("baseline"):
            baseline_sample = sample

    odd_electrodes, even_electrodes, pair_labels = build_pair_labels(args.num_electrodes)
    n_measurements = len(pair_labels)
    baseline_frame = safe_frame(baseline_sample["readings"])
    baseline_grid = readings_to_grid(baseline_frame, args.num_electrodes)
    plot_grid = render_grid(display_grid(
        baseline_grid,
        baseline_grid,
        args.mode,
        row_sensitivity=args.row_sensitivity,
        column_sensitivity=args.column_sensitivity,
        column_active_threshold=args.column_active_threshold,
        max_dampened_columns=args.max_dampened_columns,
    ))
    symmetric = args.mode in ("percent", "delta")
    cmap = "diverging" if symmetric else "viridis"

    backend = app.use_app("pyqt6")
    qapp = backend.native
    qapp.setApplicationName(APP_NAME)
    qapp.setApplicationDisplayName(APP_NAME)

    canvas = scene.SceneCanvas(
        keys="interactive",
        show=True,
        bgcolor="black",
        size=(1200, 700),
        title=APP_NAME,
    )
    vertical_grid = canvas.central_widget.add_grid(margin=0)
    vertical_grid.spacing = 5

    heatmap_grid = vertical_grid.add_grid(row=0, col=0)
    heatmap_grid.spacing = 0
    heatmap_view = heatmap_grid.add_view(row=0, col=0)
    heatmap_view.camera = "panzoom"
    rows, cols = plot_grid.shape
    heatmap_view.camera.set_range(
        x=(0.0, TRAPEZIUM_WIDTH),
        y=(-TRAPEZIUM_LEFT_HEIGHT * 0.55, TRAPEZIUM_LEFT_HEIGHT * 0.55),
    )
    heatmap_view.camera.aspect = 1.0

    clim = initial_color_limits(args, plot_grid, symmetric)
    color_map = color.get_colormap(cmap)
    mesh = scene.visuals.Mesh(
        vertices=trapezium_vertices(rows, cols),
        faces=trapezium_faces(rows, cols),
        face_colors=grid_face_colors(plot_grid, color_map, clim),
        color=(1.0, 1.0, 1.0, 1.0),
        parent=heatmap_view.scene,
    )
    mesh.set_gl_state(depth_test=False, cull_face=False)
    scene.visuals.Line(
        pos=trapezium_grid_lines(rows, cols),
        color=(0.85, 0.85, 0.85, 0.55),
        width=1.0,
        connect="segments",
        parent=heatmap_view.scene,
    )

    cbar = scene.ColorBarWidget(
        cmap=cmap,
        orientation="left",
        border_width=0.0,
        label_color="white",
        padding=(0.05, 0.3),
    )
    cbar.clim = clim
    for tick in cbar.ticks:
        tick.font_size = 8
    heatmap_grid.add_widget(cbar, row=0, col=1)

    line_grid = vertical_grid.add_grid(row=1, col=0, spacing=0, margin=0)
    line_view = line_grid.add_view(row=0, col=1, margin=0, padding=0)
    line_view.camera = scene.cameras.PanZoomCamera(aspect=None, interactive=False)

    yaxis = scene.AxisWidget(
        orientation="left",
        tick_font_size=6,
        tick_label_margin=5,
        axis_width=1,
    )
    yaxis.width_max = 60
    line_grid.add_widget(yaxis, row=0, col=0)
    yaxis.link_view(line_view)

    line_x = np.arange(n_measurements)
    line_plot = scene.visuals.Line(
        pos=np.column_stack((line_x, baseline_frame)),
        color="cyan",
        width=1,
        parent=line_view.scene,
    )

    title = scene.Label(
        f"{APP_NAME} | {taxel_layout_description(args.num_electrodes)}",
        color="white",
        font_size=10,
        anchor_x="left",
        anchor_y="top",
    )
    title.parent = canvas.scene
    title.pos = (5, canvas.size[1] - 5)

    status = scene.Label(
        "Frame 0 | R: reset baseline",
        color="white",
        font_size=10,
        anchor_x="left",
        anchor_y="bottom",
    )
    status.parent = canvas.scene
    status.pos = (5, 5)

    state = {
        "baseline_frame": baseline_frame,
        "baseline_grid": baseline_grid,
        "frame_count": 0,
        "fps_count": 0,
        "fps_start": time.perf_counter(),
        "last_sample": baseline_sample,
    }

    def set_baseline_from_sample(sample):
        frame = safe_frame(sample["readings"])
        state["baseline_frame"] = frame
        state["baseline_grid"] = readings_to_grid(frame, args.num_electrodes)
        if sample.get("clipping"):
            print_warning("Clipping detected in baseline")
        print_info("Baseline frame updated.")

    def capture_baseline():
        sample = state["last_sample"]
        if sample is None:
            print_warning("No frame available to capture baseline.")
            return
        set_baseline_from_sample(sample)

    def on_key_press(event):
        if event.key == "R":
            capture_baseline()

    canvas.connect(on_key_press)

    def update(event):
        sample = drain_latest(q_eit)
        if sample is None:
            return

        state["last_sample"] = sample
        frame = safe_frame(sample["readings"])
        frame_grid = readings_to_grid(frame, args.num_electrodes)
        plot_grid_now = render_grid(display_grid(
            frame_grid,
            state["baseline_grid"],
            args.mode,
            row_sensitivity=args.row_sensitivity,
            column_sensitivity=args.column_sensitivity,
            column_active_threshold=args.column_active_threshold,
            max_dampened_columns=args.max_dampened_columns,
        ))

        clim_now = cbar.clim
        if args.scale_limit is None and not args.fixed_scale:
            clim_now = color_limits(plot_grid_now, symmetric)
        set_heatmap_data(mesh, cbar, plot_grid_now, color_map, clim_now)

        line_pos = np.column_stack((line_x, frame))
        line_plot.set_data(pos=line_pos)
        y_min = float(np.nanmin(frame))
        y_max = float(np.nanmax(frame))
        # y_min = 0.05
        # y_max = 0.08

        y_range = max(y_max - y_min, EPS)
        line_view.camera.aspect = None
        line_view.camera.set_range(
            x=(0, n_measurements - 1),
            y=(y_min - y_range * 0.05, y_max + y_range * 0.05),
            margin=1e-4,
        )

        state["frame_count"] += 1
        state["fps_count"] += 1
        now = time.perf_counter()
        if now - state["fps_start"] >= 1.0:
            fps = state["fps_count"] / (now - state["fps_start"])
            status.text = (
                f"Frame: {state['frame_count']} ({fps:.1f} Hz) - "
                f"{'CLIPPING!' if sample.get('clipping') else 'OK'} - "
                f"mode: {args.mode} - R: reset baseline"
            )
            print_info(f"Framerate: {fps:.1f} FPS")
            state["fps_count"] = 0
            state["fps_start"] = now

    timer = app.Timer(interval=1 / 60.0, connect=update, start=True)

    try:
        print_info("Starting two-probe taxel live view (close window to exit)...")
        app.run()
    finally:
        timer.stop()
        reader.stop()
        reader.join(timeout=2.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("Stopped by user.")
    except serial.SerialException as exc:
        print_error(f"Serial error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
