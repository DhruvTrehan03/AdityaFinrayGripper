import queue
import threading
import time

import numpy as np

import amodo_eit as eit
from utils import print_info


# Defaults match the current logger configuration.
NUM_ELECTRODES = 16
STIM_FREQ_KHZ = 50
PERIODS_PER_MEASUREMENT = 50
TX_GAIN = 32
RX_GAIN = 2


class TwoProbeAmodoEITDevice(threading.Thread):
    """Amodo EIT reader that uses two-probe configurations only.

    Each configuration injects current through the same two electrodes used for
    voltage measurement: (A, B, M, N) = (A, B, A, B). The pair list covers
    every odd-numbered electrode paired with every even-numbered electrode.
    """

    def __init__(
        self,
        q_out,
        group=None,
        num_electrodes=NUM_ELECTRODES,
        stim_freq_khz=STIM_FREQ_KHZ,
        periods_per_measurement=PERIODS_PER_MEASUREMENT,
        tx_gain=TX_GAIN,
        rx_gain=RX_GAIN,
    ):
        super().__init__(group=group, name="TwoProbeAmodoEITDevice", daemon=True)
        self.devices = eit.get_connected_devices()
        self.q = q_out
        self.stop_evt = threading.Event()
        if not self.devices:
            print_info("No Amodo EIT devices connected.")
            raise SystemExit(1)
        if len(self.devices) > 1:
            print_info("Multiple Amodo EIT devices detected.")
        self.device = self.devices[0]

        self.num_electrodes = num_electrodes
        self.stim_freq_khz = stim_freq_khz
        self.periods_per_measurement = periods_per_measurement
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain

        self.baseline_frame = None
        self.baseline_clipping = None

    def _build_two_probe_configurations(self):
        electrode_configurations = []
        odd_electrodes = range(1, self.num_electrodes + 1, 2)
        even_electrodes = range(2, self.num_electrodes + 1, 2)
        for A in odd_electrodes:
            for B in even_electrodes:
                configuration = (
                    A,
                    B,
                    A,
                    B,
                    self.tx_gain,
                    self.rx_gain,
                )
                electrode_configurations.append(configuration)
        return electrode_configurations

    def _configure_and_start_streaming(self):
        print_info(
            f"Using device: {self.device.port}, "
            f"version {self.device.version}, build {self.device.build_date_time}"
        )
        self.device.set_stimulation_frequency(self.stim_freq_khz)

        print_info("Loading two-probe electrode configuration...")
        electrode_configurations = self._build_two_probe_configurations()

        self.device.set_electrode_configurations(electrode_configurations)
        print_info(
            f"Two-probe electrode configuration loaded "
            f"({len(electrode_configurations)} configurations).\n"
        )
        self.device.set_num_periods_to_sample_per_measurement(self.periods_per_measurement)

        print_info("Capturing baseline frame...")
        self.device.start_streaming()

        while self.device.latest_frame is None and not self.stop_evt.is_set():
            time.sleep(0.01)

        if self.device.latest_frame is not None:
            baseline_frame, baseline_clipping = self.device.latest_frame
            baseline_frame = np.array(
                [x if x > 1e-12 else 1e-12 for x in baseline_frame]
            )
            if baseline_clipping:
                print_info("Clipping detected in baseline")
            print_info(f"Baseline captured: {len(baseline_frame)} measurements\n")
            self.baseline_frame = baseline_frame
            self.baseline_clipping = baseline_clipping
            baseline_sample = {
                "t": time.monotonic(),
                "readings": baseline_frame.tolist(),
                "clipping": baseline_clipping,
                "baseline": True,
            }
            try:
                self.q.put_nowait(baseline_sample)
            except queue.Full:
                pass

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
