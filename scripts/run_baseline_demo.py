from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.core import load_trip, nearest_edge_match
from pathsnap.visuals import make_map, make_offline_map

DATA = ROOT / "data" / "Student folder"
OUT = ROOT / "outputs"
trip_id = sys.argv[1] if len(sys.argv) > 1 else "00000052"
trip = load_trip(DATA, "student_test", trip_id, include_route=False)
result = nearest_edge_match(trip)
map_path = make_map(trip, result, OUT / "maps" / f"{trip_id}.html")
offline_map_path = make_offline_map(trip, result, OUT / "offline_maps" / f"{trip_id}.html")
summary = {
    "trip_id": trip_id,
    "method": "nearest road-segment baseline; KD-tree midpoint candidates + exact point-to-segment projection",
    "inference_inputs": [".track", ".nodes", ".arcs"],
    "reference_route_used": False,
    "metrics": result["metrics"],
    "map": str(map_path.relative_to(ROOT)),
    "offline_map": str(offline_map_path.relative_to(ROOT)),
    "interpretation": "Snap offsets measure geometric distance to selected road segments. Directed continuity is the fraction of consecutive predicted directed edges that connect end-to-start. Neither metric is route accuracy or F1-score."
}
out_file = OUT / f"baseline_demo_{trip_id}.json"
out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
