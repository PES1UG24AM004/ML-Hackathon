# PathSnap dataset exploration — generated from supplied ZIP

Only the `train` and `student_test` partitions present in the supplied ZIP were inspected. No evaluator-testing partition was present or used.

## train

- Trips: **80**
- GPS observations per trip (min / median / max): **[361, 1529, 19313]**
- Road nodes per trip (min / median / max): **[299, 9608, 155003]**
- Road segments per trip (min / median / max): **[649, 19526, 307392]**
- Reference-route arcs per trip (min / median / max): **[59, 280, 2266]**
- Mean fraction of exactly 1-second intervals: **0.938**
- Largest timestamp gap in this partition: **153064 seconds**
- Trips with at least one gap over 5 seconds: **45**
- Non-positive timestamp steps / invalid GPS rows: **0 / 0**
- All reference-route IDs and arc endpoint IDs are in range: **True / True**
- Reference-route directed continuity (min / median): **0.9813 / 1.0000**

## student_test

- Trips: **10**
- GPS observations per trip (min / median / max): **[262, 1420, 3958]**
- Road nodes per trip (min / median / max): **[1770, 6596, 26635]**
- Road segments per trip (min / median / max): **[3577, 13145, 56027]**
- Reference-route arcs per trip (min / median / max): **[95, 250, 1323]**
- Mean fraction of exactly 1-second intervals: **0.982**
- Largest timestamp gap in this partition: **210 seconds**
- Trips with at least one gap over 5 seconds: **2**
- Non-positive timestamp steps / invalid GPS rows: **0 / 0**
- All reference-route IDs and arc endpoint IDs are in range: **True / True**
- Reference-route directed continuity (min / median): **0.9973 / 1.0000**

## Initial observations

1. The raw files are whitespace-separated and have no headers: `.track` = longitude, latitude, timestamp; `.nodes` = longitude, latitude; `.arcs` = from-node ID, to-node ID; `.route` = chronological arc IDs.
2. Timestamp gaps need explicit handling. The training partition contains trips with very large gaps; do not assume every GPS row is exactly one second apart.
3. Reference-route IDs should be validated against arc-row indices, and arc endpoints against node-row indices before modelling.
4. The nearest-road baseline in this kit does not consume `.route`; route files are only read by this EDA script for structural checks.

## Files

- `trip_inventory.csv` — per-trip inventory and data-quality flags.
- `summary.json` — aggregate statistics.
- `gps_observations_by_trip.png` and `max_timestamp_gaps.png` — EDA plots.
