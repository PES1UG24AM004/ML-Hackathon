# First-round checkpoint — SRNs

**Members:** PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007  
**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction

## Demonstrable work in this package

- Validated the supplied raw-data layout across 80 training trips and 10 student-testing trips.
- Generated a per-trip inventory, EDA summary, data-quality checks and plots.
- Built a working nearest-road-segment baseline from GPS observations and the directed road graph.
- Added a FastAPI service, simple browser UI, a self-contained offline SVG trajectory map, an interactive Folium export and inference logs.
- Ran the geometric baseline over all 10 student-testing trips and saved `outputs/student_test_baseline_summary.csv`.
- Smoke-tested the API, trip list, inference endpoint and generated map endpoint successfully.

## Actual dataset findings

| Finding | Training | Student testing |
|---|---:|---:|
| Trips included | 80 | 10 |
| GPS observations per trip, median | 1,529 | 1,420 |
| Road nodes per trip, median | 9,608 | 6,597 |
| Road segments per trip, median | 19,527 | 13,145 |
| Mean of per-trip fractions with exact 1-second intervals | 93.8% | 98.2% |
| Largest timestamp gap | 153,064 s | 210 s |
| Trips with at least one gap over 5 seconds | 45 | 2 |

The large training gaps mean the pipeline must use actual timestamps and handle irregular sampling. Do not calculate speed or transition features under a fixed 1 Hz assumption without checking `dt`.

## Baseline run: student trip 00000052

| Metric | Actual result |
|---|---:|
| GPS observations | 262 |
| Road segments in graph | 3,577 |
| Median distance from GPS to selected segment | 3.49 m |
| 95th percentile distance | 12.54 m |
| Directed edge-to-edge continuity | 24.1% |

**Interpretation:** the geometric snap is close for this trip, but independent nearest-edge predictions often fail the directed connectivity check. That is a concrete motivation for sequence decoding (HMM/Viterbi). Snap distance is not prediction accuracy; no accuracy/F1 claim is made here.

## Baseline sweep across all 10 development trips

| Trip ID | Median snap offset | 95th percentile | Directed continuity |
|---|---:|---:|---:|
| 00000000 | 6.02 m | 31.15 m | 25.5% |
| 00000007 | 4.60 m | 27.54 m | 32.8% |
| 00000009 | 4.14 m | 16.25 m | 32.6% |
| 00000040 | 5.07 m | 33.17 m | 24.7% |
| 00000052 | 3.49 m | 12.54 m | 24.1% |
| 00000056 | 4.46 m | 15.07 m | 28.4% |
| 00000065 | 4.31 m | 16.57 m | 21.3% |
| 00000076 | 6.14 m | 18.84 m | 33.7% |
| 00000077 | 5.98 m | 19.57 m | 30.3% |
| 00000091 | 3.79 m | 13.17 m | 26.6% |

These are geometric baseline diagnostics, not accuracy/F1 or ground-truth route scores. Full precision values and processing times are in `outputs/student_test_baseline_summary.csv`.

## Not yet complete

- Five or more trained model families and their measured comparison.
- Group-aware cross-validation and student-test model comparison.
- Bayesian hyperparameter optimisation and trial record.
- HMM/Viterbi implementation and classifier-only vs decoded route comparison.
- Final evaluator-facing deployment across two laptops.
- Consolidated final technical report and public GitHub repository.

## Important dataset boundary

The supplied ZIP contains only `train` and `student_test`. There is no `evaluator_test` folder in the uploaded ZIP. Do not invent or reconstruct evaluator trips. The baseline's inference path does not read `.route`; the EDA script reads reference routes only for dataset-integrity statistics.
