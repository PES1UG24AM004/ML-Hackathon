# Dataset Directory

This directory holds the raw dataset files for the PathSnap map-matching task.

## Data Layout Requirements

To run the pipeline and demonstration, place the supplied `Student folder.zip` into this folder:

```text
data/
├── README.md
├── Student folder.zip       <- (Place supplied archive here)
└── Student folder/          <- (Extracted automatically by scripts/extract_dataset.py)
    ├── train/               <- 80 training trips (with .track, .nodes, .arcs, .route)
    └── student_test/        <- 10 development trips (with .track, .nodes, .arcs, .route)
```

## Setup Instructions

1. Place `Student folder.zip` in this directory (`data/`).
2. Run the extraction script:
   ```bash
   python scripts/extract_dataset.py
   ```
   Or launch `START_DEMO.bat`, which extracts the archive automatically if not already present.

## Note on Repository Boundaries
- The raw dataset files (`Student folder.zip` and `Student folder/`) are ignored by Git (`.gitignore`) to prevent committing large binary data and adhering to dataset usage guidelines.
- The `evaluator_test` partition described in the hackathon guidelines is not present in the student distribution package and is not used.
