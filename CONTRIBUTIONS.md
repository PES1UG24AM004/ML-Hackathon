# PathSnap Contribution Register & Proposed Ownership

**Course:** UE24CS352A — Machine Learning  
**Project:** PathSnap — Distributed Map-Matching & Route Reconstruction  
**Team:** SRNs  

---

## 1. Governance & Contribution Policy

This repository represents the collaborative effort of Team SRNs for the first-round baseline evaluation.

> [!IMPORTANT]
> **Proposed Ownership Notice:** The module assignments documented below represent proposed technical ownership and review areas, **not** unilateral claims of historical authorship. Commits made during initial repository setup are published under the repository owner's GitHub identity (`PES1UG24AM004`). Each teammate has been allocated dedicated modules to inspect, refine, and commit using their own authenticated GitHub account.

---

## 2. Proposed Module Ownership

### PES1UG24AM004 (Integration & Architecture Lead)
- **Role:** System architecture, baseline integration, launcher automation, cross-trip testing, and repository curation.
- **Assigned Modules & Artifacts:**
  - `START_DEMO.bat` — Windows virtual environment setup and one-click demo launcher
  - `scripts/run_baseline_demo.py` — Single-trip baseline CLI demonstration
  - `scripts/summarize_student_baseline.py` — Cross-trip evaluation over student-testing dataset
  - `FIRST_ROUND_RUNBOOK.md` — Demonstration script and viva briefing
  - `FIRST_ROUND_STATUS.md` — Measured checkpoint status and boundary declaration
  - `SLIDE_OUTLINE.md` — Four-slide presentation outline
  - `CONTRIBUTOR_GUIDE.md` — Teammate onboarding and contribution workflows

### PES1UG24AM005 (Dataset Exploration & EDA Lead)
- **Role:** Dataset ingestion, schema validation, timestamp discontinuity detection, and exploratory data analysis.
- **Assigned Modules & Artifacts:**
  - `scripts/extract_dataset.py` — Dataset extraction and integrity verification
  - `scripts/run_eda.py` — Full EDA sweep over 80 training and 10 testing trips
  - `outputs/eda/eda_report.md` — Detailed statistical findings across partitions
  - `outputs/eda/summary.json` — Aggregated dataset statistics
  - `outputs/eda/trip_inventory.csv` — Per-trip observation and quality flags
  - `outputs/eda/gps_observations_by_trip.png` — Observation count distribution plot
  - `outputs/eda/max_timestamp_gaps.png` — Sampling gap distribution plot

### PES1UG24AM006 (Geospatial & Map-Matching Lead)
- **Role:** Geospatial projection, KD-tree spatial indexing, candidate road search, and geometric projection.
- **Assigned Modules & Artifacts:**
  - `src/pathsnap/core.py` — `Trip` data model, `load_trip()`, equirectangular local metre projection (`_project`, `_unproject`), and `nearest_edge_match()` algorithm

### PES1UG24AM007 (Backend, UI & Visualisation Lead)
- **Role:** API service development, browser user interface, offline SVG generation, and Folium map rendering.
- **Assigned Modules & Artifacts:**
  - `src/pathsnap/api.py` — FastAPI service endpoints (`/health`, `/api/trips`, `/api/match/{trip_id}`) and embedded dashboard UI
  - `src/pathsnap/visuals.py` — Interactive Folium HTML map generator (`make_map`) and self-contained SVG map generator (`make_offline_map`)

---

## 3. Individual Verification and Review

Each member will independently review their assigned module, perform their designated improvement task, and confirm below upon pushing their contribution from their personal GitHub account:

- [x] **PES1UG24AM004** — Integration, launcher validation, baseline verification, and repository structure verified.
- [ ] **PES1UG24AM005** — Ingestion logic, timestamp gap analysis, and EDA visualisations reviewed and verified.
- [ ] **PES1UG24AM006** — Metre-based equirectangular projection and KD-tree candidate nearest-edge logic reviewed and verified.
- [ ] **PES1UG24AM007** — FastAPI endpoints, offline SVG renderer, and web UI reviewed and verified.
