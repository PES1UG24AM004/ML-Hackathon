# PathSnap Contributor Guide & Teammate Workflow

Welcome to the **PathSnap** first-round codebase!

This guide provides step-by-step instructions for each team member to clone the repository, run the baseline, inspect their assigned module, implement their designated review/improvement task, and commit using their own authenticated GitHub account.

---

## 1. Quick Start & Local Setup

### Step 1: Clone the Repository
Clone using HTTPS or SSH with your own GitHub account:
```bash
git clone https://github.com/PES1UG24AM004/ML-Hackathon.git
cd ML-Hackathon
```

### Step 2: Set Up Dataset
Due to redistribution restrictions and repository size guidelines, the raw dataset archive `Student folder.zip` is not committed to the repository:
1. Copy `Student folder.zip` into the `data/` directory.
2. Run the extraction script:
   ```bash
   python scripts/extract_dataset.py
   ```
   This will unpack `train` (80 trips) and `student_test` (10 trips) into `data/Student folder/`.

### Step 3: Install Dependencies
Create a virtual environment and install the required dependencies:
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Run the Application
You can run the full baseline demo in two ways:
- **Windows One-Click Launcher:** Double-click `START_DEMO.bat`
- **Manual Launch:**
  ```bash
  # Run baseline demo on default trip 00000052
  python scripts/run_baseline_demo.py 00000052

  # Launch FastAPI server and interactive UI
  python -m uvicorn pathsnap.api:app --app-dir src --host 127.0.0.1 --port 8000
  ```
  Open `http://127.0.0.1:8000` in your web browser.

---

## 2. Git Identity Configuration

Before making any commits, verify that your local Git identity matches your personal GitHub account so that GitHub correctly attributes your commits:

```bash
git config --local user.name "Your Full Name"
git config --local user.email "your_github_registered_email@example.com"
```

Verify your configuration:
```bash
git config --local user.name
git config --local user.email
```

---

## 3. Dedicated Review & Improvement Tasks

Each teammate has a concrete, actionable improvement task aligned with their module ownership:

### PES1UG24AM005 (EDA & Preprocessing)
- **Assigned Module:** `scripts/run_eda.py`, `scripts/extract_dataset.py`
- **Review Task:**
  1. Inspect the summary outputs in `outputs/eda/eda_report.md` and `outputs/eda/trip_inventory.csv`.
  2. Implement an additional quality check in `scripts/run_eda.py`: calculate the 90th percentile of sampling time intervals per trip (`np.percentile(dt, 90)`) to quantify interval jitter alongside the existing exact 1-second fraction check.
  3. Re-run `python scripts/run_eda.py` to regenerate the artifacts.
- **Branch Name:** `feature/eda-sampling-jitter`

### PES1UG24AM006 (Geospatial & Map-Matching)
- **Assigned Module:** `src/pathsnap/core.py`
- **Review Task:**
  1. Inspect `nearest_edge_match()` in `src/pathsnap/core.py`.
  2. Add input validation or parameter configuration: allow `candidate_k` to be passed dynamically (with bounds checking: `1 <= candidate_k <= 50`) and expose bearing / orientation helper utilities for future directional feature engineering.
  3. Verify with `python scripts/run_baseline_demo.py 00000052` that metrics remain consistent and correct.
- **Branch Name:** `feature/geospatial-candidate-bounds`

### PES1UG24AM007 (Backend, UI & Visualisation)
- **Assigned Module:** `src/pathsnap/api.py`, `src/pathsnap/visuals.py`
- **Review Task:**
  1. Inspect the frontend template in `src/pathsnap/api.py` and the offline SVG builder in `src/pathsnap/visuals.py`.
  2. Add an enhancement to the web UI: display processing latency (elapsed time in milliseconds) alongside the existing snap offset metrics when a trip inference completes.
  3. Verify by opening `http://127.0.0.1:8000`, running inference on trip `00000052`, and checking that the response time renders clearly.
- **Branch Name:** `feature/ui-latency-display`

---

## 4. Contributing Your Changes

1. **Create and switch to your feature branch:**
   ```bash
   git checkout -b <your-feature-branch>
   ```
2. **Implement your changes and test locally:**
   ```bash
   # Run verification
   python scripts/run_baseline_demo.py 00000052
   ```
3. **Stage and commit your changes:**
   ```bash
   git status
   git add <modified-files>
   git commit -m "feat(<module>): <description of your improvement>"
   ```
4. **Push your branch to GitHub:**
   ```bash
   git push -u origin <your-feature-branch>
   ```
5. **Open a Pull Request:**
   Go to `https://github.com/PES1UG24AM004/ML-Hackathon/pulls` and open a Pull Request into `main` or `develop`.
6. **Update the Contribution Register:**
   Mark your checkbox as completed in `CONTRIBUTIONS.md`.
