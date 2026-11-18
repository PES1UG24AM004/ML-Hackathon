# 4-slide first-round presentation outline

**Team:** SRNs — PES1UG24AM004 / 005 / 006 / 007

## Slide 1 — PathSnap: the problem
- GPS can drift, jump across buildings or land on parallel roads.
- Objective: match noisy GPS observations to directed road segments and reconstruct a plausible route.
- Visual: use the example on page 1 of the Main Guidelines PDF (blue raw GPS points vs green reference path).

## Slide 2 — Understanding the data
- 80 training trips + 10 student-testing trips in the supplied ZIP.
- `.track`: longitude, latitude, timestamp; `.nodes`: longitude, latitude; `.arcs`: directed node links; `.route`: ordered arc IDs.
- EDA checks file sizes/row counts, time gaps, route IDs and graph references.
- Finding: 45/80 training trips have at least one timestamp gap above five seconds; the largest is 153,064 seconds.

## Slide 3 — Working baseline architecture
- Load trip files.
- Convert lon/lat to local metres (rather than measuring degrees).
- KD-tree gets nearby road-segment midpoint candidates.
- Exact point-to-segment distance chooses one candidate per GPS point.
- FastAPI serves the inference endpoint; Folium renders raw GPS, nearby road edges and predicted snaps.

## Slide 4 — Demo, evidence and next stage
- Live demo: `http://127.0.0.1:8000`, trip `00000052`.
- Median snap offset: 3.49 m; p95: 12.54 m; directed continuity: 24.1%.
- Interpretation: geometric snap can be close while consecutive edges are often disconnected.
- Next: feature engineering, five model families, group-aware CV, Bayesian optimisation, HMM/Viterbi, final two-laptop API/UI.

**Do not put fabricated accuracy, F1, model comparison or HMM improvement numbers on any slide.**
