# PathSnap Dataset Setup & Specification

This document details the expected dataset structure, required file formats, local configuration, and dependencies for the PathSnap map-matching challenge.

---

## 1. Dataset Overview & Provenance

The raw dataset consists of GPS trajectory recordings matched against urban road networks across two partitions:
- **`train/`** (80 trips): Trajectories used for exploratory data analysis, feature extraction, candidate generation, and 5-fold cross-validation.
- **`student_test/`** (10 development trips): Independent holdout evaluation partition used for model benchmarking, Bayesian optimisation verification, and sequential HMM route reconstruction.

> [!NOTE]
> Due to university academic distribution guidelines and file size limits, the raw dataset archives (`Student folder.zip` or raw folder structures) are excluded from this Git repository. Trained models (`artifacts/final_mlp_model.npz`) and HMM transition configs (`artifacts/hmm_config.json`) are bundled directly with the repository so that the inference API, UI, and test suites run out-of-the-box.

---

## 2. Expected Local Directory Layout

To run dataset processing, candidate generation, or model retraining, place the raw dataset files into this directory under `data/`:

```text
PathSnap_Submission/
└── data/
    ├── README.md
    └── Student folder/          <- (Or set PATHSNAP_DATA_DIR environment variable)
        ├── train/
        │   ├── 00000001/
        │   │   ├── 00000001.track
        │   │   ├── 00000001.nodes
        │   │   ├── 00000001.arcs
        │   │   └── 00000001.route
        │   └── ... (80 trips total)
        └── student_test/
            ├── 00000000/
            │   ├── 00000000.track
            │   ├── 00000000.nodes
            │   ├── 00000000.arcs
            │   └── 00000000.route
            ├── 00000052/
            ├── 00000056/
            └── ... (10 trips total)
```

Alternatively, set the environment variable to point to any custom path:
```bash
# Windows (PowerShell)
$env:PATHSNAP_DATA_DIR = "C:\Path\To\Student folder"

# Linux / macOS
export PATHSNAP_DATA_DIR="/path/to/Student folder"
```

---

## 3. Required Raw File Formats

Each trip directory contains four headerless, whitespace-delimited text files:

| File Extension | Columns | Types | Description |
|---|---|---|---|
| `.track` | 3 | `float, float, float` | `[longitude, latitude, timestamp_seconds]`. Raw sequential GPS points. |
| `.nodes` | 2 | `float, float` | `[longitude, latitude]`. Road graph vertices; 0-based row index is `node_id`. |
| `.arcs` | 2 | `int, int` | `[from_node_id, to_node_id]`. Directed graph edges; 0-based row index is `arc_id`. |
| `.route` | 1 | `int` | `[arc_id]`. Chronological ground-truth sequence of traversed road arcs. |

---

## 4. Regenerating Intermediate Candidate Data

The training pipeline generates compressed candidate feature matrices (`train_candidates.npz` and `test_candidates.npz`) containing 16 geometric, kinematic, and topological features per candidate arc.

To regenerate these intermediate files from the raw dataset:

```bash
python src/feature_engineering.py --data-dir "data/Student folder" --artifacts-dir "artifacts" --top-k 8
```

This will produce:
- `artifacts/train_candidates.npz` (~55 MB uncompressed, contains $X, y, \text{group}, \text{obs\_idx}, \text{cand\_arcs}$)
- `artifacts/test_candidates.npz` (~7 MB uncompressed)

---

## 5. Script Dependencies on Dataset

| Script | Depends on Raw Dataset? | Depends on Candidate NPZ? | Description |
|---|:---:|:---:|---|
| `src/data_processing.py` | Yes | No | Reads raw trips and generates `results/eda_trip_summary.csv` and `results/eda_summary.json`. |
| `src/feature_engineering.py` | Yes | No | Reads raw trips and writes candidate NPZ files. |
| `src/train_evaluate.py` | For evaluation arcs only | Yes | Requires candidate NPZ files for 5-fold CV and training all 6 model families. |
| `src/bayesian_opt.py` | For evaluation arcs only | Yes | Requires candidate NPZ files for Gaussian Process hyperparameter optimisation. |
| `src/hmm_viterbi.py` | Yes (`student_test`) | No | Uses trained `artifacts/final_mlp_model.npz` directly on raw test trips. |
| `src/visualizations.py` | Optional (only for geo maps) | No | Visualizes recorded results from `results/`. Uses raw test trips for Figure 6 and Leaflet HTML maps. |
| `src/api/main.py` | Yes (`student_test`) | No | Live server reads test trips dynamically and performs real-time inference using `final_mlp_model.npz`. |
| `src/build_report.py` | No | No | Reads verified outputs from `results/` and generates `results/consolidated_report.html`. |

---

## 6. Quick Setup (Extraction)

1. Place `Student folder.zip` in this directory (`data/`).
2. Run the extraction script:
   ```bash
   python scripts/extract_dataset.py
   ```
   Or launch `START_DEMO.bat`, which extracts the archive automatically if not already present.

> The `evaluator_test` partition described in the hackathon guidelines is not present in the student distribution package and is not used.
