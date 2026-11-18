# SRNs — first-round evaluation runbook

## What the evaluator should see

The organiser email gives the first evaluation window as **11:30 AM–12:40 PM**, before the lunch break. The problem statement was released at 9:30 AM. Prioritise a working, explainable checkpoint over claiming the full final system.

**Goal for this checkpoint:** demonstrate that the team has understood the raw PathSnap data and has a real, reproducible first map-matching baseline running end-to-end.

### 3–4 minute demo flow

1. **Problem (20 sec):** GPS observations are noisy; the task is to match them to road segments and reconstruct a plausible connected route.
2. **Dataset (40 sec):** show `outputs/eda/eda_report.md` and `trip_inventory.csv`. Explain the four headerless files:
   - `.track`: longitude, latitude, timestamp in seconds.
   - `.nodes`: longitude, latitude; zero-based row index is node ID.
   - `.arcs`: from-node ID, to-node ID; zero-based row index is arc ID.
   - `.route`: chronological sequence of reference arc IDs.
3. **EDA finding (30 sec):** training has 80 trips and student-testing has 10. Timestamp steps are mostly one second, but training includes long gaps (largest observed: 153,064 seconds); therefore time gaps must be detected instead of assuming uniform sampling.
4. **Live run (60–90 sec):** open `http://127.0.0.1:8000`, select trip `00000052`, click the button, then explain the map layers and the median / p95 snap offsets.
5. **Engineering limitation and next step (30 sec):** nearest-edge matching is the baseline. It does not enforce graph continuity; next is candidate features + five-model comparison + Bayesian optimisation + HMM/Viterbi.

## Four-slide speaking outline (no separate deck required)

1. **Problem and objective:** noisy GPS → realistic road route; show the challenge illustration in the Main Guidelines, page 1.
2. **Dataset and EDA:** 80 train + 10 student-test trips, four raw file types, timestamp gaps; show EDA chart/report.
3. **Baseline architecture:** parse files → local metre projection → KD-tree candidate edges → exact point-to-segment projection → Folium map / FastAPI response.
4. **Live output and next step:** show trip 00000052, actual snap-offset stats and weak directed continuity; motivate HMM/Viterbi.

## Suggested opening script

"We are SRNs: PES1UG24AM004, 005, 006 and 007. Our project is PathSnap, a map-matching pipeline for noisy GPS trajectories. We started by validating the supplied data rather than assuming the files are uniform. Each trip gives GPS observations, road nodes, directed road arcs and a reference route. We have built a data-inventory and quality-check pipeline, then a working nearest-road baseline served through FastAPI. The current demo shows how raw GPS observations are projected onto the local road network. This is our baseline; we are not presenting it as the final learned solution. Our next phase is candidate feature engineering, five-model comparison, Bayesian tuning and HMM/Viterbi route decoding."

## What to point at on screen

- Browser: `http://127.0.0.1:8000`
- Live server terminal: `INFERENCE trip=... gps_points=... median_offset_m=... p95_offset_m=... continuity=...`
- EDA report: `outputs/eda/eda_report.md`
- EDA charts: `outputs/eda/gps_observations_by_trip.png`, `outputs/eda/max_timestamp_gaps.png`

## Likely viva questions and concise answers

**Why not use Euclidean distance in longitude/latitude degrees?**  
Because degrees are angular units and longitude scale varies with latitude. The baseline converts coordinates into a local metre-based equirectangular plane before measuring point-to-segment distance.

**What does an arc ID mean?**  
The row index in `.arcs`; that row stores the directed connection `from_node_id → to_node_id`.

**Does a GPS row map directly to a node or arc row?**  
No. GPS observations and road-network tables are separate structures. Candidate road segments must be generated spatially.

**Why use a KD-tree?**  
It quickly finds nearby segment midpoints; exact point-to-segment distance is then calculated for those candidates.

**What is the current method's weakness?**  
Independent nearest-edge choices can jump between disconnected road segments. The directed continuity metric exposes this; HMM/Viterbi is a planned sequence-level correction.

**What is the difference between `.route` and our predicted route?**  
`.route` is supplied reference information, encoded as a chronological list of arc IDs. The baseline predictor does not read it. The reference labels are not sent to the browser.

**How are you avoiding leakage?**  
The prediction path reads only `.track`, `.nodes` and `.arcs`. The supplied ZIP has `train` and `student_test`; the separate evaluator-testing partition is not present in it. Evaluator labels must remain untouched when provided.

**Have you completed the five-model comparison and Bayesian optimisation?**  
Not in this first-round baseline package. The current checkpoint validates the dataset and establishes a functioning map-matching baseline. The next stage will train and compare five models on the 80 training trips, use the 10 student-testing trips for development comparison, and record Bayesian search trials.

## Four-member work split (suggested)

| SRN | Ownership for this checkpoint | Next-phase ownership |
|---|---|---|
| PES1UG24AM004 | Lead presenter; architecture and integration | Experiment tracking and final model selection |
| PES1UG24AM005 | Dataset exploration; inventory and quality findings | Feature engineering and candidate generation |
| PES1UG24AM006 | Road graph interpretation; nearest-edge baseline | HMM/Viterbi and graph-continuity analysis |
| PES1UG24AM007 | FastAPI, browser UI, maps and run instructions | API/UI integration and deployment validation |

All four members should understand the data layout, baseline limitations and demo flow. Change the ownership if your actual work split differs.

## Do not claim

- A learned model has been selected.
- Five models have been trained and compared.
- Bayesian optimisation has been completed.
- HMM/Viterbi has improved results.
- The final evaluator-facing two-laptop system is complete.
- Accuracy or F1 values not actually measured by this implementation.
