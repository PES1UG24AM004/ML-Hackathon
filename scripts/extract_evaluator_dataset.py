"""
PathSnap — Evaluator Dataset Extraction and Strict Isolation Pipeline
Extracts evaluator trips with strict separation of input features and private reference labels.
"""

import os
import sys
import zipfile
from pathlib import Path

DEFAULT_ZIP_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "evaluator_test.zip")
EXPECTED_TRIPS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]


def resolve_project_root():
    # If run from scripts/, project root is parent
    current_dir = Path(__file__).resolve().parent
    if (current_dir / "src").is_dir():
        return current_dir
    if (current_dir.parent / "src").is_dir():
        return current_dir.parent
    return Path.cwd()


def extract_and_isolate_evaluator_dataset(zip_path: str = DEFAULT_ZIP_PATH, project_root: Path = None):
    if project_root is None:
        project_root = resolve_project_root()

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Evaluator dataset ZIP not found at: {zip_path}")

    evaluator_data_dir = project_root / "evaluator_data"
    inputs_dir = evaluator_data_dir / "inputs"
    labels_dir = evaluator_data_dir / "private_labels"
    outputs_dir = project_root / "evaluator_outputs"

    # Create target directories
    inputs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "reports").mkdir(parents=True, exist_ok=True)

    print(f"[*] Extracting evaluator dataset from: {zip_path}")
    print(f"[*] Destination: {evaluator_data_dir}")

    extracted_trips = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            parts = name.replace("\\", "/").strip("/").split("/")
            # Expected format: evaluator_test/<trip_id>/<trip_id>.<ext> or <trip_id>/<trip_id>.<ext>
            filename = parts[-1]
            ext = os.path.splitext(filename)[1].lower()
            trip_id = None
            for p in parts:
                if p in EXPECTED_TRIPS:
                    trip_id = p
                    break
            if not trip_id and len(parts) >= 2 and parts[-2].isdigit():
                trip_id = parts[-2]

            if not trip_id:
                continue

            extracted_trips.add(trip_id)
            content = zf.read(member.filename)

            if ext in [".track", ".nodes", ".arcs"]:
                # Input features go to evaluator_data/inputs/<trip_id>/
                trip_in_dir = inputs_dir / trip_id
                trip_in_dir.mkdir(parents=True, exist_ok=True)
                dest = trip_in_dir / filename
                with open(dest, "wb") as f:
                    f.write(content)
            elif ext == ".route":
                # Private ground-truth labels go strictly to evaluator_data/private_labels/
                dest = labels_dir / f"{trip_id}.route"
                with open(dest, "wb") as f:
                    f.write(content)

    print(f"[*] Extracted files for {len(extracted_trips)} trips.")
    summary = validate_extracted_dataset(project_root)
    return summary


def validate_extracted_dataset(project_root: Path = None):
    if project_root is None:
        project_root = resolve_project_root()

    inputs_dir = project_root / "evaluator_data" / "inputs"
    labels_dir = project_root / "evaluator_data" / "private_labels"

    if not inputs_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError("Evaluator directories do not exist.")

    trip_stats = []
    missing_errors = []

    print("\n" + "=" * 78)
    print(f"{'Trip ID':^10} | {'GPS Fixes (.track)':^20} | {'Road Nodes (.nodes)':^20} | {'Road Arcs (.arcs)':^18}")
    print("-" * 78)

    for trip_id in sorted(EXPECTED_TRIPS):
        trip_dir = inputs_dir / trip_id
        track_path = trip_dir / f"{trip_id}.track"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        arcs_path = trip_dir / f"{trip_id}.arcs"
        route_path = labels_dir / f"{trip_id}.route"

        # Check existence
        if not track_path.exists():
            missing_errors.append(f"Trip {trip_id}: missing .track")
        if not nodes_path.exists():
            missing_errors.append(f"Trip {trip_id}: missing .nodes")
        if not arcs_path.exists():
            missing_errors.append(f"Trip {trip_id}: missing .arcs")
        if not route_path.exists():
            missing_errors.append(f"Trip {trip_id}: missing private .route")

        # Validate file sizes and line counts
        n_track = sum(1 for line in open(track_path, "r", encoding="utf-8") if line.strip()) if track_path.exists() else 0
        n_nodes = sum(1 for line in open(nodes_path, "r", encoding="utf-8") if line.strip()) if nodes_path.exists() else 0
        n_arcs = sum(1 for line in open(arcs_path, "r", encoding="utf-8") if line.strip()) if arcs_path.exists() else 0
        n_route = sum(1 for line in open(route_path, "r", encoding="utf-8") if line.strip()) if route_path.exists() else 0

        if n_track == 0 or n_nodes == 0 or n_arcs == 0 or n_route == 0:
            missing_errors.append(f"Trip {trip_id}: empty file detected (track={n_track}, nodes={n_nodes}, arcs={n_arcs}, route={n_route})")

        trip_stats.append({
            "trip_id": trip_id,
            "track_count": n_track,
            "nodes_count": n_nodes,
            "arcs_count": n_arcs,
            "route_count": n_route,
        })
        print(f"{trip_id:^10} | {n_track:^20,} | {n_nodes:^20,} | {n_arcs:^18,}")

    print("=" * 78)

    if missing_errors:
        print("\n[ERROR] Evaluator dataset validation failed with issues:")
        for err in missing_errors:
            print(f"  - {err}")
        raise ValueError(f"Evaluator dataset validation failed: {missing_errors}")

    print(f"\n[SUCCESS] All {len(trip_stats)} evaluator trips validated successfully.")
    print(f"[SECURITY] Input features are in: {inputs_dir}")
    print(f"[SECURITY] Private reference labels strictly isolated in: {labels_dir}")
    return trip_stats


if __name__ == "__main__":
    zip_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP_PATH
    extract_and_isolate_evaluator_dataset(zip_arg)
