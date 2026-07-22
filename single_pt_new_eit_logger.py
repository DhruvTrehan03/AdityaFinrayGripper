import argparse
import csv
import os
import queue
import re
import threading
import time
from datetime import datetime

import numpy as np
import serial
import amodo_eit as eit
from utils import print_info

# ---------------------------------------------------------------------------
# User-configurable experiment settings
# ---------------------------------------------------------------------------
FILE_NAME = "RepeatedContact_80_30"     # prefix for output CSV file

TARGET_POINT = (80.0, 30.0)             # (x, y) in printer coordinate space, fixed

RETRACT_Z = 25.0                        # mm, fully-retracted / rest height
Z_START = 22.6                          # mm, first probing target z
Z_END = 21.5                            # mm, last probing target z (approx — see NUM_CONTACTS)
Z_STEP = 0.01                           # mm, step between successive contacts
NUM_CONTACTS = 111                      # 22.60, 22.59, ..., 21.50 inclusive

DWELL_AT_TARGET_S = 5.0                 # hold time once probe reaches target z
REST_AFTER_RETRACT_S = 3.0              # hold time once probe is back at RETRACT_Z

APPROACH_FEED = 1000                    # mm/min — homing / initial positioning moves
FAST_FEED = 3000                        # mm/min — feed rate used for continuous descend/retract moves
CSV_FLUSH_EVERY_ROWS = 100              # avoid making disk flush the sample-rate limiter
CSV_FLUSH_EVERY_S = 0.5

Z_ARRIVAL_TOLERANCE_MM = 0.005          # mm — how close the firmware-reported Z must be to the
                                         # commanded target before wait_for_position() considers the
                                         # move "arrived". Kept tighter than Z_STEP so consecutive
                                         # contact targets can't be confused with each other.

# EIT device settings
NUM_ELECTRODES = 16                     # 8 odd (excitation) x 8 even (measurement) = 64 channels
STIM_FREQ_KHZ = 100
PERIODS_PER_MEASUREMENT = 100
TX_GAIN = 32
RX_GAIN = 2


# ---------------------------------------------------------------------------
# EIT device thread — pushes every fresh frame onto the shared sensor queue,
# unmodified, with the same monotonic clock used by force samples.
# ---------------------------------------------------------------------------

class AmodoEITDevice(threading.Thread):
    def __init__(self, q_out, group=None):
        super().__init__(group=group, name="AmodoEITDevice", daemon=True)
        self.devices = eit.get_connected_devices()
        self.q = q_out
        self.stop_evt = threading.Event()
        if not self.devices:
            print_info("No Amodo EIT devices connected.")
            raise SystemExit(1)
        if len(self.devices) > 1:
            print_info("Multiple Amodo EIT devices detected.")
        self.device = self.devices[0]

    def _build_two_probe_configurations(self):
        """Odd electrodes as excitation (A), even electrodes as measurement
        (B): 8 x 8 = 64 configurations for NUM_ELECTRODES=16."""
        electrode_configurations = []
        odd_electrodes = range(1, NUM_ELECTRODES + 1, 2)
        even_electrodes = range(2, NUM_ELECTRODES + 1, 2)
        for A in odd_electrodes:
            for B in even_electrodes:
                configuration = (
                    A,
                    B,
                    A,
                    B,
                    TX_GAIN,
                    RX_GAIN,
                )
                print_info(f"  Configuration: A={A}, B={B}, TX_GAIN={TX_GAIN}, RX_GAIN={RX_GAIN}")
                electrode_configurations.append(configuration)
        return electrode_configurations

    def _configure_and_start_streaming(self):
        print_info(
            f"Using device: {self.device.port}, "
            f"version {self.device.version}, build {self.device.build_date_time}"
        )
        self.device.set_stimulation_frequency(STIM_FREQ_KHZ)

        print_info("Loading two-probe electrode configuration...")
        electrode_configurations = self._build_two_probe_configurations()

        self.device.set_electrode_configurations(electrode_configurations)
        print_info(
            f"Two-probe electrode configuration loaded "
            f"({len(electrode_configurations)} configurations).\n"
        )
        self.device.set_num_periods_to_sample_per_measurement(PERIODS_PER_MEASUREMENT)

        print_info("Starting EIT streaming...")
        self.device.start_streaming()

        while self.device.latest_frame is None and not self.stop_evt.is_set():
            time.sleep(0.01)

    def run(self):
        last_frame = None
        try:
            with self.device:
                self._configure_and_start_streaming()
                while not self.stop_evt.is_set():
                    latest = self.device.latest_frame
                    if latest is None:
                        time.sleep(0.002)
                        continue
                    frame, clipping = latest
                    if frame is None:
                        time.sleep(0.002)
                        continue
                    frame_arr = np.asarray(frame, dtype=float)
                    if last_frame is not None and np.array_equal(frame_arr, last_frame):
                        time.sleep(0.002)
                        continue
                    last_frame = frame_arr.copy()
                    sample = {
                        "row_type": "eit",
                        "t": time.monotonic(),
                        "readings": frame_arr.tolist(),
                        "clipping": clipping,
                    }
                    try:
                        self.q.put_nowait(sample)
                    except queue.Full:
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                        self.q.put_nowait(sample)
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print_info("\n\nStopped by user")
        except Exception as e:
            print_info(f"Unexpected error in EIT reader thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.device.stop_streaming()
            except Exception:
                pass

    def stop(self):
        self.stop_evt.set()
        try:
            self.device.stop_streaming()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Force reader thread — continuously writes every complete force line to the
# force queue. Force values are logged only on force rows; EIT rows stay
# force-blank and can be aligned later with sensor_time_s.
# ---------------------------------------------------------------------------

class ForceReader(threading.Thread):
    def __init__(self, force_serial, q_out, debug_lines=0, group=None):
        super().__init__(group=group, name="ForceReader", daemon=True)
        self.force_serial = force_serial
        self.q = q_out
        self.debug_lines = debug_lines
        self.stop_evt = threading.Event()
        self.record_evt = threading.Event()
        self.samples_read = 0
        self.samples_logged = 0
        self.invalid_lines = 0
        self.latest_force = None

    def _put_sample(self, sample):
        try:
            self.q.put_nowait(sample)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(sample)

    def _handle_line(self, line):
        sample_time = time.monotonic()
        text = line.decode(errors="ignore").strip()
        if not text:
            return
        try:
            force_value = float(text)
        except ValueError:
            self.invalid_lines += 1
            if self.invalid_lines <= self.debug_lines:
                print_info(f"Force raw invalid: {text!r}")
            return
        self.samples_read += 1
        self.latest_force = force_value
        if self.samples_read <= self.debug_lines:
            print_info(f"Force raw: {text!r} -> {force_value:.6g}")
        if self.record_evt.is_set():
            self.samples_logged += 1
            self._put_sample({
                "row_type": "force",
                "t": sample_time,
                "force_sample": force_value,
            })

    def run(self):
        try:
            self.force_serial.reset_input_buffer()
        except AttributeError:
            self.force_serial.flushInput()
        while not self.stop_evt.is_set():
            try:
                line = self.force_serial.readline()
                if not line:
                    continue
                self._handle_line(line)
            except Exception as exc:
                print_info(f"Unexpected error in force reader thread: {exc}")
                time.sleep(0.05)

    def set_recording(self, enabled):
        if enabled:
            self.record_evt.set()
        else:
            self.record_evt.clear()

    def stop(self):
        self.stop_evt.set()


# ---------------------------------------------------------------------------
# Position store — thread-safe holder for the most recently CONFIRMED real
# X/Y/Z. It is only ever written by PrinterController.wait_for_position(),
# from an actual M114 R response line — never guessed, never left stale
# during a move.
# ---------------------------------------------------------------------------

class PositionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._z = None
        self._xyz = None

    def update(self, x, y, z):
        with self._lock:
            self._xyz = (x, y, z)
            self._z = z

    @property
    def latest_z(self):
        with self._lock:
            return self._z

    @property
    def latest_xyz(self):
        with self._lock:
            return self._xyz


# ---------------------------------------------------------------------------
# Printer controller — the SOLE owner of printer_serial (no separate reader
# thread anymore, so there's no lock contention and no risk of concurrent
# queries interfering with move commands).
#
# wait_for_position() blocks until the firmware's response to M114 R
# contains "Count" AND (if target_z is given) the parsed actual Z has
# genuinely converged to it. The "Count" check alone isn't enough: with
# closely-chained small moves the firmware can echo a valid M114 response
# while still blending into the next segment, so a bare "Count" only means
# "a response arrived", not "the head is physically at the target". Passing
# target_z makes this re-query until the reported Z actually matches, which
# is what keeps the subsequent dwell from starting early (above target).
# ---------------------------------------------------------------------------

class PrinterController:
    def __init__(self, printer_serial, position_store):
        self.printer_serial = printer_serial
        self.position_store = position_store

    def write(self, command):
        self.printer_serial.write(command)
        self.printer_serial.flush()

    def wait_for_position(self, target_z=None, tolerance=Z_ARRIVAL_TOLERANCE_MM,
                           timeout=30.0, retry_interval=0.05):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.printer_serial.flush()
            self.printer_serial.write(b"M114 R\r\n")
            self.printer_serial.flush()

            line_deadline = time.monotonic() + 2.0
            got_count = False
            actual_z = None
            while time.monotonic() < line_deadline:
                line = self.printer_serial.readline()
                if not line:
                    continue
                decoded = line.decode(errors="ignore").strip()
                match = re.search(
                    r"X:([-\d.]+)\s+Y:([-\d.]+)\s+Z:([-\d.]+)", decoded
                )
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    self.position_store.update(x, y, z)
                    actual_z = z
                if "Count" in decoded:
                    got_count = True
                    break

            if got_count:
                if target_z is None or (actual_z is not None and abs(actual_z - target_z) <= tolerance):
                    return
            time.sleep(retry_interval)

        print_info(
            f"Warning: wait_for_position() timed out"
            f"{f' waiting for Z≈{target_z:.3f} mm' if target_z is not None else ''} "
            f"(last reading: {self.position_store.latest_z})"
        )

    def close(self):
        try:
            self.printer_serial.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared state — the single source of truth for "what is the printer doing
# right now". The EIT writer thread reads a snapshot of this on every single
# frame so that the never-stopping data stream can be tagged with the
# correct bunch/phase after the fact.
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.bunch_index = -1     # -1 == not yet in a contact (setup/baseline)
        self.target_z = None
        self.phase = "setup"      # setup | baseline | descend | dwell | retract | rest
        self.phase_start = time.monotonic()

    def set(self, bunch_index=None, target_z=None, phase=None):
        with self._lock:
            if bunch_index is not None:
                self.bunch_index = bunch_index
            if target_z is not None:
                self.target_z = target_z
            if phase is not None:
                self.phase = phase
            self.phase_start = time.monotonic()

    def snapshot(self):
        with self._lock:
            return {
                "bunch_index": self.bunch_index,
                "target_z": self.target_z,
                "phase": self.phase,
                "phase_elapsed_s": time.monotonic() - self.phase_start,
            }


def get_one_fresh_eit_frame(q_eit):
    """Discard any stale queued frames, then block until exactly one fresh
    EIT frame arrives and return it. Used to grab the single baseline
    reading before continuous logging begins."""
    while True:
        try:
            q_eit.get_nowait()
        except queue.Empty:
            break
    while True:
        sample = q_eit.get()   # blocks until a new EIT sample is pushed
        if sample.get("row_type") == "eit":
            return sample


def drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def build_sensor_row(sample_index, tag, sample, channel_count, target_x, target_y,
                     experiment_start, actual_z):
    """Build a single CSV row from either an EIT frame or force sample.
    Force and EIT keep their own native arrival rates, but share one
    experiment-relative monotonic timebase for downstream alignment."""
    sensor_time_s = sample["t"] - experiment_start
    row_type = sample.get("row_type", "eit")
    readings = sample.get("readings", [])
    force_sample = sample.get("force_sample")
    row = [
        sample_index, row_type,
        tag["bunch_index"], tag["phase"], tag["target_z"],
        actual_z if actual_z is not None else "",
        f'{tag["phase_elapsed_s"]:.4f}',
        target_x, target_y,
        force_sample if force_sample is not None else "",
        f"{sensor_time_s:.4f}", sample.get("clipping", ""),
    ]
    row.extend(
        f"{readings[i]:.12f}" if i < len(readings) else ""
        for i in range(channel_count)
    )
    return row


# ---------------------------------------------------------------------------
# Sensor writer thread — continuously drains the separate force/EIT queues and
# appends every sample to the CSV, tagged with whatever SharedState/position
# say at that instant. This is what lets force and EIT both run at their
# maximum rates without forcing them onto one fake sampling grid.
# ---------------------------------------------------------------------------

class SensorWriter(threading.Thread):
    def __init__(self, q_eit, q_force, state, csv_path, channel_count, target_x, target_y,
                 experiment_start, position_store, start_index=0, group=None):
        super().__init__(group=group, name="SensorWriter", daemon=True)
        self.q_eit = q_eit
        self.q_force = q_force
        self.state = state
        self.channel_count = channel_count
        self.target_x = target_x
        self.target_y = target_y
        self.experiment_start = experiment_start
        self.position_store = position_store
        self.stop_evt = threading.Event()
        self.sample_index = start_index
        self._f = open(csv_path, "a", newline="")
        self._writer = csv.writer(self._f)
        self._last_flush = time.monotonic()

    def _write_sample(self, sample):
        tag = self.state.snapshot()
        actual_z = self.position_store.latest_z
        row = build_sensor_row(
            self.sample_index, tag, sample, self.channel_count,
            self.target_x, self.target_y, self.experiment_start,
            actual_z,
        )
        self._writer.writerow(row)
        self.sample_index += 1

    def run(self):
        while not self.stop_evt.is_set():
            try:
                sample = self.q_force.get_nowait()
            except queue.Empty:
                try:
                    sample = self.q_eit.get(timeout=0.5)
                except queue.Empty:
                    continue
            self._write_sample(sample)
            now = time.monotonic()
            if (
                self.sample_index % CSV_FLUSH_EVERY_ROWS == 0
                or now - self._last_flush >= CSV_FLUSH_EVERY_S
            ):
                self._f.flush()
                self._last_flush = now

    def stop(self):
        self.stop_evt.set()

    def close(self):
        try:
            while True:
                try:
                    self._write_sample(self.q_force.get_nowait())
                except queue.Empty:
                    break
            while True:
                try:
                    self._write_sample(self.q_eit.get_nowait())
                except queue.Empty:
                    break
            self._f.flush()
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------

def write_csv_header(csv_path, channel_count):
    header = [
        "sample_index", "row_type",
        "bunch_index", "phase", "target_z_mm", "actual_z_mm", "phase_elapsed_s",
        "target_x_mm", "target_y_mm", "force_N",
        "sensor_time_s", "eit_clipping",
    ]
    header.extend([f"eit_{i}" for i in range(channel_count)])
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)


# ---------------------------------------------------------------------------
# Printer moves
# ---------------------------------------------------------------------------

def setup(printer, position_store, target_x, target_y):
    """Home, then move to a safe height above the target point and settle
    at RETRACT_Z directly above it."""
    printer.write(f"G1 Z{RETRACT_Z} F{APPROACH_FEED}\r\n".encode())
    print_info("Homing...")
    printer.write(b"G28\r\n")
    printer.wait_for_position()
    print_info("Homed.")

    printer.write(f"G1 Z{RETRACT_Z + 50} F{APPROACH_FEED}\r\n".encode())
    printer.wait_for_position(target_z=RETRACT_Z + 50)

    print_info(f"Moving to X{target_x} Y{target_y}...")
    printer.write(f"G1 X{target_x} Y{target_y} F{APPROACH_FEED}\r\n".encode())
    printer.wait_for_position()
    print_info(f"Arrived at XY (actual reading: {position_store.latest_xyz})")

    printer.write(f"G1 Z{RETRACT_Z} F{APPROACH_FEED}\r\n".encode())
    printer.wait_for_position(target_z=RETRACT_Z)
    print_info(f"Setup complete — probe at rest height (actual Z={position_store.latest_z}).")


def remove_probe(printer, position_store, target_x, target_y):
    printer.write(f"G1 Z{RETRACT_Z + 20} F{APPROACH_FEED}\r\n".encode())
    printer.wait_for_position(target_z=RETRACT_Z + 20)


def move_z_continuous(printer, position_store, to_z, feed):
    """Command one continuous Z move and poll real printer position until the
    firmware-reported Z has reached the target."""
    printer.write(f"G1 Z{to_z:.4f} F{feed}\r\n".encode())
    printer.wait_for_position(target_z=to_z)
    return position_store.latest_z


# ---------------------------------------------------------------------------
# Core protocol: 111 repeated contacts at the same (x, y)
# ---------------------------------------------------------------------------

def run_repeated_contact_protocol(printer, state, position_store):
    target_zs = [round(Z_START - i * Z_STEP, 3) for i in range(NUM_CONTACTS)]
    current_z = RETRACT_Z

    for i, tz in enumerate(target_zs):
        print_info(f"--- Contact {i + 1}/{NUM_CONTACTS}  target Z={tz:.3f} mm ---")

        # One continuous move down from the current rest height to this
        # contact's target z. Logging continues independently while the move
        # is in progress.
        state.set(bunch_index=i, target_z=tz, phase="descend")
        actual_z = move_z_continuous(printer, position_store, tz, FAST_FEED)
        current_z = tz
        print_info(f"  Reached target (actual Z={actual_z}, target {tz:.3f} mm)")

        # Dwell at target for DWELL_AT_TARGET_S seconds
        state.set(phase="dwell")
        time.sleep(DWELL_AT_TARGET_S)

        # One continuous retract back to rest height.
        state.set(phase="retract")
        actual_z = move_z_continuous(printer, position_store, RETRACT_Z, FAST_FEED)
        current_z = RETRACT_Z
        print_info(f"  Retracted (actual Z={actual_z}, rest {RETRACT_Z:.3f} mm)")

        # Rest / recovery pause once back at RETRACT_Z
        state.set(phase="rest")
        time.sleep(REST_AFTER_RETRACT_S)

    print_info(f"All {NUM_CONTACTS} contacts complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Repeated single-location contact EIT collector (continuous streaming, "
                     "actual-Z + force tagging)"
    )
    parser.add_argument("--printer-port", default="/dev/ttyUSB0")
    parser.add_argument("--force-port", default="/dev/ttyACM0")
    parser.add_argument(
        "--force-debug-lines",
        type=int,
        default=0,
        help="Print this many raw force serial lines and their parsed values.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--x", type=float, default=TARGET_POINT[0])
    parser.add_argument("--y", type=float, default=TARGET_POINT[1])
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "RawData", "RepeatedContact")
    os.makedirs(output_dir, exist_ok=True)

    experiment_start = time.monotonic()

    print_info("Connecting to printer controller...")
    printer_serial = serial.Serial(args.printer_port, 115200)
    print_info("Connecting to force sensor...")
    force_serial = serial.Serial(args.force_port, 115200, timeout=0.01)

    q_eit = queue.Queue(maxsize=20000)
    q_force = queue.Queue(maxsize=20000)

    eit_reader = AmodoEITDevice(q_out=q_eit)
    eit_reader.start()

    force_reader = ForceReader(
        force_serial,
        q_out=q_force,
        debug_lines=args.force_debug_lines,
    )
    force_reader.start()
    print_info("EIT and force readers started.")

    # Determine channel count from the first available EIT frame
    channel_count = 0
    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            samp = q_eit.get(timeout=1.0)
            if (
                samp
                and samp.get("row_type") == "eit"
                and "readings" in samp
                and len(samp["readings"]) > 0
            ):
                channel_count = len(samp["readings"])
                print_info(f"Detected EIT channel count: {channel_count}")
                break
        except queue.Empty:
            continue

    if channel_count == 0:
        print_info("No EIT data detected — exiting.")
        eit_reader.stop()
        force_reader.stop()
        force_reader.join(timeout=2.0)
        printer_serial.close()
        force_serial.close()
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(output_dir, f"{timestamp}_{FILE_NAME}.csv")
    write_csv_header(csv_path, channel_count)

    state = SharedState()
    position_store = PositionStore()
    printer = PrinterController(printer_serial, position_store)

    sensor_writer = None

    try:
        setup(printer, position_store, args.x, args.y)

        # Baseline: exactly one EIT value, captured while the probe is
        # stationary at rest height, before any contact motion begins.
        print_info("Collecting single baseline sample...")
        baseline_frame = get_one_fresh_eit_frame(q_eit)
        baseline_frame["row_type"] = "baseline_eit"
        baseline_tag = {"bunch_index": -1, "phase": "baseline",
                         "target_z": None, "phase_elapsed_s": 0.0}
        baseline_row = build_sensor_row(
            0, baseline_tag, baseline_frame, channel_count,
            args.x, args.y, experiment_start,
            position_store.latest_z,
        )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(baseline_row)
        print_info("Baseline sample collected.")

        drain_queue(q_force)
        force_reader.set_recording(True)
        print_info("Force logging started.")

        # Start the never-stopping CSV writer for everything from here on.
        # It writes force samples and EIT frames at their own arrival rates,
        # using one shared experiment timebase.
        sensor_writer = SensorWriter(
            q_eit=q_eit, q_force=q_force,
            state=state, csv_path=csv_path, channel_count=channel_count,
            target_x=args.x, target_y=args.y, experiment_start=experiment_start,
            position_store=position_store,
            start_index=1,
        )
        sensor_writer.start()

        run_repeated_contact_protocol(printer, state, position_store)
    except KeyboardInterrupt:
        print_info("Stopped by user.")
    except Exception as e:
        print_info(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            remove_probe(printer, position_store, args.x, args.y)
        except Exception:
            pass

        eit_reader.stop()
        force_reader.stop()
        eit_reader.join(timeout=2.0)
        force_reader.join(timeout=2.0)
        print_info(
            f"Force reader parsed {force_reader.samples_read} samples, "
            f"logged {force_reader.samples_logged} "
            f"({force_reader.invalid_lines} invalid lines skipped)."
        )

        total_rows = 1  # baseline row, if it was reached
        if sensor_writer is not None:
            sensor_writer.stop()
            sensor_writer.join(timeout=2.0)
            sensor_writer.close()
            total_rows = sensor_writer.sample_index

        printer.close()
        force_serial.close()
        print_info(f"Data saved to {csv_path}")
        print_info(f"Total sensor rows written: {total_rows}")
        print_info("All threads stopped and resources cleaned up.")


if __name__ == "__main__":
    main()
