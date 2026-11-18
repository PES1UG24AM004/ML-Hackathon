from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP = DATA / "Student folder.zip"
TARGET = DATA / "Student folder"

if TARGET.exists() and (TARGET / "train").is_dir() and (TARGET / "student_test").is_dir():
    print(f"Dataset already extracted: {TARGET}")
else:
    if not ZIP.exists():
        raise FileNotFoundError(f"Missing {ZIP}")
    print("Extracting supplied dataset (train + student_test only)...")
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(DATA)
    if not (TARGET / "train").is_dir() or not (TARGET / "student_test").is_dir():
        raise RuntimeError("Unexpected archive layout; expected data/Student folder/{train,student_test}")
    print(f"Extracted to {TARGET}")
