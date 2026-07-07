import argparse
import os
import queue
import sys
import threading
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import serial
from vispy import app, scene

from two_probe_amodo_eit import NUM_ELECTRODES, TwoProbeAmodoEITDevice
from utils import print_error, print_info, print_warning


APP_NAME = "Two-Probe Amodo Taxel Live View"
EPS = 1e-12


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


def display_grid(frame_grid, baseline_grid, mode):
    if mode == "raw":
        return frame_grid
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
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 98))
        if limit <= EPS:
            limit = 1.0
        return -limit, limit

    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if abs(vmax - vmin) <= EPS:
        pad = max(abs(vmax) * 0.05, 1.0)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def set_image_clim(image, cbar, clim):
    image.clim = clim
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
        choices=("percent", "abs-percent", "delta", "raw"),
        default="percent",
        help="Quantity shown in the heatmap.",
    )
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="Keep the initial color scale instead of adapting it to incoming frames.",
    )
    parser.add_argument(
        "--num-electrodes",
        type=int,
        default=NUM_ELECTRODES,
        help="Number of electrodes in the two-probe configuration.",
    )
    args = parser.parse_args()

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
    plot_grid = display_grid(baseline_grid, baseline_grid, args.mode)
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
    heatmap_view.camera.set_range(
        x=(-0.5, len(even_electrodes) - 0.5),
        y=(-0.5, len(odd_electrodes) - 0.5),
    )
    heatmap_view.camera.aspect = 1.0

    image = scene.visuals.Image(
        plot_grid,
        cmap=cmap,
        clim=color_limits(plot_grid, symmetric),
        parent=heatmap_view.scene,
    )

    cbar = scene.ColorBarWidget(
        cmap=cmap,
        orientation="left",
        border_width=0.0,
        label_color="white",
        padding=(0.05, 0.3),
    )
    cbar.clim = image.clim
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
        f"{APP_NAME} | rows odd A/M {odd_electrodes} | cols even B/N {even_electrodes}",
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
        plot_grid_now = display_grid(frame_grid, state["baseline_grid"], args.mode)

        image.set_data(plot_grid_now)
        if not args.fixed_scale:
            set_image_clim(image, cbar, color_limits(plot_grid_now, symmetric))

        line_pos = np.column_stack((line_x, frame))
        line_plot.set_data(pos=line_pos)
        y_min = float(np.nanmin(frame))
        y_max = float(np.nanmax(frame))
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
