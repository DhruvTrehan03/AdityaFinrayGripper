# single_pt_new_eit_logger CSV Format

This file documents the CSV produced by `single_pt_new_eit_logger.py` so the data can be parsed later without re-reading the logger code.

## Experiment Summary

The logger runs repeated contacts at one fixed printer XY point.

- Default point: `(target_x_mm, target_y_mm) = (80.0, 30.0)`
- Rest height: `RETRACT_Z = 25.0 mm`
- Target Z sequence: `22.60, 22.59, ..., 21.50 mm`
- Number of contacts: `111`
- Per contact:
  - continuously move from rest height down to `target_z_mm`
  - dwell at target for `5.0 s`
  - continuously retract to `25.0 mm`
  - rest for `3.0 s`

Force samples and EIT frames are logged as separate rows at their own arrival rates. They share the same `sensor_time_s` clock, so parse by time rather than by assuming force and EIT rows are one-to-one.

## Output Location

Default output directory:

```text
RawData/RepeatedContact/
```

Default filename pattern:

```text
YYYY-MM-DD_HH-MM-SS_RepeatedContact_80_30.csv
```

## Timebase

`sensor_time_s` is the primary alignment column.

It is computed from Python's monotonic clock:

```python
sensor_time_s = sample_time_monotonic - experiment_start_monotonic
```

Use `sensor_time_s` to align force and EIT. Do not use row number as time.

## Row Types

The CSV is event-based.

| row_type | Meaning | EIT columns | force_N |
| --- | --- | --- | --- |
| `baseline_eit` | Single EIT baseline captured at rest height before the repeated contacts | filled | blank |
| `eit` | One EIT frame from the Amodo device | filled | blank |
| `force` | One force serial sample | blank | measured force for that sample |

For force-vs-EIT alignment, use the `force` rows and interpolate or nearest-neighbor match them to `eit` rows by `sensor_time_s`. EIT rows intentionally do not carry copied force values.

## Phases

| phase | Meaning |
| --- | --- |
| `baseline` | Initial single baseline EIT frame at rest height |
| `descend` | Continuous motion from rest height to current target Z |
| `dwell` | Probe is held at current target Z for 5 seconds |
| `retract` | Continuous motion from current target Z back to rest height |
| `rest` | Probe is held at rest height for 3 seconds before the next target |

`bunch_index` is the contact index:

- `-1` means setup/baseline before the repeated contacts.
- `0` to `110` correspond to the 111 target Z contacts.

## Columns

| Column | Type | Notes |
| --- | --- | --- |
| `sample_index` | integer | Incrementing CSV row index after the header. Baseline is `0`. |
| `row_type` | string | One of `baseline_eit`, `eit`, `force`. |
| `bunch_index` | integer | Contact index, or `-1` for baseline/setup. |
| `phase` | string | `baseline`, `descend`, `dwell`, `retract`, or `rest`. |
| `target_z_mm` | float or blank | Current commanded target Z for the contact. Blank for baseline. |
| `actual_z_mm` | float or blank | Latest firmware-reported Z from `M114 R`. Updated when printer position is polled. |
| `phase_elapsed_s` | float | Seconds since the current phase was set. |
| `target_x_mm` | float | Fixed target X coordinate for the run. |
| `target_y_mm` | float | Fixed target Y coordinate for the run. |
| `force_N` | float or blank | Measured force for `force` rows only. Blank for EIT rows. |
| `sensor_time_s` | float | Shared experiment-relative monotonic timestamp. Use this for alignment. |
| `eit_clipping` | boolean or blank | EIT clipping flag for EIT rows. Blank for force rows. |
| `eit_0` ... `eit_N` | float or blank | EIT channel values for EIT rows. Blank for force rows. |

The current logger uses a two-probe EIT configuration with 16 electrodes:

- Odd electrodes are excitation candidates: `1, 3, 5, ..., 15`
- Even electrodes are measurement candidates: `2, 4, 6, ..., 16`
- Each configuration is `(A, B, A, B, TX_GAIN, RX_GAIN)`
- Expected channel count is `8 x 8 = 64`

The header is generated from the first detected EIT frame, so parsers should discover EIT columns dynamically with `col.startswith("eit_")`.

## Recommended Pandas Parsing

```python
import pandas as pd

csv_path = "RawData/RepeatedContact/YYYY-MM-DD_HH-MM-SS_RepeatedContact_80_30.csv"
df = pd.read_csv(csv_path)

eit_cols = [c for c in df.columns if c.startswith("eit_")]

force = (
    df[df["row_type"] == "force"]
    .copy()
    .sort_values("sensor_time_s")
)

eit = (
    df[df["row_type"].isin(["baseline_eit", "eit"])]
    .copy()
    .sort_values("sensor_time_s")
)

# Optional: attach nearest force sample to each EIT frame.
eit_with_force = pd.merge_asof(
    eit,
    force[["sensor_time_s", "force_N"]].rename(columns={"force_N": "nearest_force_N"}),
    on="sensor_time_s",
    direction="nearest",
)
```

## Contact-Level Parsing

To analyze only the steady hold at each target, filter to `row_type == "eit"` and `phase == "dwell"`.

```python
dwell_eit = df[
    (df["row_type"] == "eit")
    & (df["phase"] == "dwell")
].copy()

by_contact = {
    int(contact): group.sort_values("sensor_time_s")
    for contact, group in dwell_eit.groupby("bunch_index")
}
```

To include the full motion history for each contact, group all rows where `bunch_index >= 0` by `bunch_index`.

## Important Parsing Notes

- Force and EIT rows are intentionally not one-to-one.
- `sample_index` is row order, not sample time.
- `actual_z_mm` is only updated when the printer is polled by `wait_for_position()`. During a continuous move, it is the latest confirmed printer position, not a high-frequency Z trace.
- `phase_elapsed_s` is useful for slicing within each phase, but `sensor_time_s` is the global alignment clock.
- Blank EIT cells on force rows should be treated as missing values.
- Blank `eit_clipping` on force rows should be treated as missing or false depending on the downstream analysis.
