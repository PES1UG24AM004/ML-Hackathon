# PathSnap — Distributed Map-Matching & Route Reconstruction

**Team:** SRNs  
**Members:**
- PES1UG24AM004 — Architecture, integration, baseline workflows, and testing
- PES1UG24AM005 — Dataset exploration and EDA
- PES1UG24AM006 — Geospatial processing and map-matching
- PES1UG24AM007 — Backend API, UI, and visualisation

**Course:** UE24CS352A — Machine Learning  
**Target Repository:** [https://github.com/PES1UG24AM004/ML-Hackathon.git](https://github.com/PES1UG24AM004/ML-Hackathon.git)

---

## 1. Project Overview

This repository contains the **first-round baseline demonstration** and verification suite for the PathSnap map-matching challenge. It is designed to be fully reproducible, transparent, and grounded in the supplied dataset.

> [!NOTE]
> This is a first-round evaluation deliverable establishing a validated baseline and dataset inventory. It is not represented as the final learned ML system. Five-model comparison, Bayesian optimisation, and HMM/Viterbi route decoding are planned for subsequent phases.

---

## 2. What Is Implemented & Verified

1. **Dataset Ingestion & Integrity Validation:** Ingestion and validation scripts for the supplied `train` (80 trips) and `student_test` (10 trips) partitions.
2. **Exploratory Data Analysis (EDA):** Per-trip observation counts, node/arc/route sizes, 1-second interval distribution, timestamp gaps, invalid rows, reference-route ID bounds, and graph endpoint bounds.
3. **Nearest-Road Map-Matching Baseline:** An end-to-end baseline consuming `.track`, `.nodes`, and `.arcs` at inference time. Coordinates are projected into a local metre-based equirectangular plane, candidate road-segment midpoints are indexed with a KD-tree, and exact point-to-segment orthogonal projections determine the nearest road edge.
4. **Visualisation & Offline Maps:**
   - Interactive Folium HTML maps displaying raw GPS points, nearby road segments, and predicted route snaps.
   - Self-contained, zero-dependency offline SVG maps rendered directly in the browser.
5. **Batch Cross-Trip Testing:** Full baseline evaluation executed across all 10 student-testing trips, recorded in `outputs/student_test_baseline_summary.csv`.
6. **FastAPI Service & Web UI:** Live HTTP service exposing `/health`, `/api/trips`, and `/api/match/{trip_id}` with an embedded dashboard and server-side request logging.

---

## 3. Quick Start (Windows / Linux / macOS)

### Running on Windows (One-Click)
1. Ensure Python 3.10+ is installed.
2. Place `Student folder.zip` into the `data/` directory (see [data/README.md](data/README.md)).
3. Double-click `START_DEMO.bat`.
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
5. Select a student-testing trip (default: `00000052`) and click **Run nearest-road baseline**.

### Running from Command Line
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Extract dataset (requires data/Student folder.zip)
python scripts/extract_dataset.py

# 4. Generate EDA reports and plots
python scripts/run_eda.py

# 5. Run single-trip baseline demonstration
python scripts/run_baseline_demo.py 00000052

# 6. Run batch baseline across all 10 student-test trips
python scripts/summarize_student_baseline.py

# 7. Start FastAPI web service
python -m uvicorn pathsnap.api:app --app-dir src --host 127.0.0.1 --port 8000
```

---

## 4. Key Documentation & Artifacts

- **[CONTRIBUTIONS.md](CONTRIBUTIONS.md):** Module ownership register, team roles, and individual review status.
- **[CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md):** Onboarding guide, environment setup, and dedicated review tasks for each member.
- **[FIRST_ROUND_RUNBOOK.md](FIRST_ROUND_RUNBOOK.md):** 3–4 minute evaluator presentation flow, viva Q&A guide, and live demo checklist.
- **[FIRST_ROUND_STATUS.md](FIRST_ROUND_STATUS.md):** Measured baseline diagnostics, dataset findings, and explicit project boundaries.
- **[SLIDE_OUTLINE.md](SLIDE_OUTLINE.md):** 4-slide first-round presentation outline.
- **[outputs/eda/](outputs/eda/):** Generated dataset inventory (`trip_inventory.csv`), statistical summary (`summary.json`), markdown report (`eda_report.md`), and distribution charts.
- **[outputs/student_test_baseline_summary.csv](outputs/student_test_baseline_summary.csv):** Benchmark results across all 10 development trips.

---

## 5. Baseline Evaluation Summary (Student Testing Trips)

| Trip ID | GPS Observations | Road Segments | Median Snap Offset | 95th Percentile Offset | Directed Continuity |
|:-------:|:----------------:|:-------------:|:------------------:|:----------------------:|:-------------------:|
| 00000000 | 2,503 | 18,196 | 6.02 m | 31.15 m | 25.5% |
| 00000007 | 1,354 | 4,369 | 4.60 m | 27.54 m | 32.8% |
| 00000009 | 1,049 | 13,106 | 4.14 m | 16.25 m | 32.6% |
| 00000040 | 2,995 | 27,026 | 5.07 m | 33.17 m | 24.7% |
| **00000052** | **262** | **3,577** | **3.49 m** | **12.54 m** | **24.1%** |
| 00000056 | 3,958 | 27,631 | 4.46 m | 15.07 m | 28.4% |
| 00000065 | 768 | 13,184 | 4.31 m | 16.57 m | 21.3% |
| 00000076 | 1,486 | 56,027 | 6.14 m | 18.84 m | 33.7% |
| 00000077 | 3,911 | 9,622 | 5.98 m | 19.57 m | 30.3% |
| 00000091 | 874 | 12,303 | 3.79 m | 13.17 m | 26.6% |

> [!IMPORTANT]
> The snap offsets measure geometric distance to the closest road segment. Directed edge continuity measures whether consecutive snapped segments form a connected path in the road graph. Low continuity (~21–34%) illustrates the known limitation of point-by-point geometric snapping and motivates sequence decoding (HMM/Viterbi) in the next phase.

---

## 6. Project Boundaries & Integrity Declaration

- **No Reference Leakage:** The baseline inference logic uses only `.track`, `.nodes`, and `.arcs`. The reference `.route` file is not read during inference and is never transmitted to the client.
- **Partition Scope:** The supplied archive contains only `train` and `student_test`. No evaluator test partition was provided or used.
- **No Fabricated Metrics:** Accuracy, F1-scores, and model comparisons are not fabricated. All metrics reported above are genuine geometric outputs from the implemented codebase.

---

## 7. Repository Layout

```text
ML-Hackathon/
├── .gitignore
├── CONTRIBUTIONS.md
├── CONTRIBUTOR_GUIDE.md
├── FIRST_ROUND_RUNBOOK.md
├── FIRST_ROUND_STATUS.md
├── README.md
├── requirements.txt
├── SLIDE_OUTLINE.md
├── START_DEMO.bat
├── data/
│   └── README.md
├── scripts/
│   ├── extract_dataset.py
│   ├── run_baseline_demo.py
│   ├── run_eda.py
│   └── summarize_student_baseline.py
├── src/
│   └── pathsnap/
│       ├── __init__.py
│       ├── api.py
│       ├── core.py
│       └── visuals.py
└── outputs/
    ├── baseline_demo_00000052.json
    ├── student_test_baseline_summary.csv
    ├── eda/
    │   ├── eda_report.md
    │   ├── gps_observations_by_trip.png
    │   ├── max_timestamp_gaps.png
    │   ├── summary.json
    │   └── trip_inventory.csv
    ├── maps/
    │   └── .gitkeep
    └── offline_maps/
        ├── 00000007.html
        ├── 00000052.html
        └── 00000065.html
```
