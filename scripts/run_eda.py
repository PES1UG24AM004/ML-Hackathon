from __future__ import annotations

from pathlib import Path
import csv, json, sys, math
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.core import load_trip

DATA = ROOT / "data" / "Student folder"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for partition in ("train", "student_test"):
    part_dir = DATA / partition
    for folder in sorted(p for p in part_dir.iterdir() if p.is_dir()):
        tid = folder.name
        trip = load_trip(DATA, partition, tid, include_route=True)
        track, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
        dt = np.diff(track[:, 2]) if len(track) > 1 else np.array([], dtype=float)
        route_in_range = bool(route is not None and np.all((route >= 0) & (route < len(arcs))))
        arc_nodes_in_range = bool(np.all((arcs >= 0) & (arcs < len(nodes))))
        route_continuity = float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) if route is not None and len(route) > 1 and route_in_range else float("nan")
        rows.append({
            "partition": partition, "trip_id": tid,
            "gps_observations": int(len(track)), "road_nodes": int(len(nodes)),
            "road_segments": int(len(arcs)), "reference_route_arcs": int(len(route) if route is not None else 0),
            "one_second_interval_fraction": float(np.mean(dt == 1.0)) if len(dt) else 0.0,
            "max_timestamp_gap_seconds": float(np.max(dt)) if len(dt) else 0.0,
            "gaps_over_5_seconds": int(np.sum(dt > 5.0)),
            "nonpositive_timestamp_steps": int(np.sum(dt <= 0.0)),
            "invalid_gps_rows": int(np.sum(~np.isfinite(track).all(axis=1))),
            "route_arc_ids_in_range": route_in_range,
            "arc_node_ids_in_range": arc_nodes_in_range,
            "reference_route_directed_continuity": route_continuity,
            "min_longitude": float(np.min(track[:, 0])), "max_longitude": float(np.max(track[:, 0])),
            "min_latitude": float(np.min(track[:, 1])), "max_latitude": float(np.max(track[:, 1])),
        })

csv_path = OUT / "trip_inventory.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

summary = {}
for partition in ("train", "student_test"):
    rs = [r for r in rows if r["partition"] == partition]
    def vals(key): return np.asarray([r[key] for r in rs], dtype=float)
    summary[partition] = {
        "trip_count": len(rs),
        "gps_observations_min_median_max": [int(np.min(vals("gps_observations"))), int(np.median(vals("gps_observations"))), int(np.max(vals("gps_observations")))],
        "nodes_min_median_max": [int(np.min(vals("road_nodes"))), int(np.median(vals("road_nodes"))), int(np.max(vals("road_nodes")))],
        "arcs_min_median_max": [int(np.min(vals("road_segments"))), int(np.median(vals("road_segments"))), int(np.max(vals("road_segments")))],
        "route_arcs_min_median_max": [int(np.min(vals("reference_route_arcs"))), int(np.median(vals("reference_route_arcs"))), int(np.max(vals("reference_route_arcs")))],
        "one_second_interval_fraction_mean": float(np.mean(vals("one_second_interval_fraction"))),
        "max_timestamp_gap_seconds_max": float(np.max(vals("max_timestamp_gap_seconds"))),
        "trips_with_gap_over_5_seconds": int(np.sum(vals("gaps_over_5_seconds") > 0)),
        "timestamp_nonpositive_steps_total": int(np.sum(vals("nonpositive_timestamp_steps"))),
        "invalid_gps_rows_total": int(np.sum(vals("invalid_gps_rows"))),
        "route_ids_in_range_all": bool(all(r["route_arc_ids_in_range"] for r in rs)),
        "arc_node_ids_in_range_all": bool(all(r["arc_node_ids_in_range"] for r in rs)),
        "route_directed_continuity_min_median": [float(np.nanmin(vals("reference_route_directed_continuity"))), float(np.nanmedian(vals("reference_route_directed_continuity")))],
        "worst_gap_trip_ids": sorted(rs, key=lambda r: r["max_timestamp_gap_seconds"], reverse=True)[:3],
    }

(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Plot 1: GPS observations by trip
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, partition in enumerate(("train", "student_test")):
    vals = [r["gps_observations"] for r in rows if r["partition"] == partition]
    ax.scatter(range(len(vals)), vals, label=partition, s=22)
ax.set(title="GPS observations per trip", xlabel="Trip index within partition", ylabel="GPS observations")
ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "gps_observations_by_trip.png", dpi=160); plt.close(fig)

# Plot 2: sampling gap distribution on a log scale
fig, ax = plt.subplots(figsize=(9, 4.5))
for partition in ("train", "student_test"):
    vals = [r["max_timestamp_gap_seconds"] for r in rows if r["partition"] == partition]
    ax.hist(np.log10(np.asarray(vals) + 1), bins=18, alpha=.65, label=partition)
ax.set(title="Per-trip maximum timestamp gap (log10(seconds + 1))", xlabel="log10(max gap + 1)", ylabel="Trip count")
ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "max_timestamp_gaps.png", dpi=160); plt.close(fig)

md = ["# PathSnap dataset exploration — generated from supplied ZIP", "",
      "Only the `train` and `student_test` partitions present in the supplied ZIP were inspected. No evaluator-testing partition was present or used.", ""]
for partition, s in summary.items():
    md += [f"## {partition}", "",
           f"- Trips: **{s['trip_count']}**",
           f"- GPS observations per trip (min / median / max): **{s['gps_observations_min_median_max']}**",
           f"- Road nodes per trip (min / median / max): **{s['nodes_min_median_max']}**",
           f"- Road segments per trip (min / median / max): **{s['arcs_min_median_max']}**",
           f"- Reference-route arcs per trip (min / median / max): **{s['route_arcs_min_median_max']}**",
           f"- Mean fraction of exactly 1-second intervals: **{s['one_second_interval_fraction_mean']:.3f}**",
           f"- Largest timestamp gap in this partition: **{s['max_timestamp_gap_seconds_max']:.0f} seconds**",
           f"- Trips with at least one gap over 5 seconds: **{s['trips_with_gap_over_5_seconds']}**",
           f"- Non-positive timestamp steps / invalid GPS rows: **{s['timestamp_nonpositive_steps_total']} / {s['invalid_gps_rows_total']}**",
           f"- All reference-route IDs and arc endpoint IDs are in range: **{s['route_ids_in_range_all']} / {s['arc_node_ids_in_range_all']}**",
           f"- Reference-route directed continuity (min / median): **{s['route_directed_continuity_min_median'][0]:.4f} / {s['route_directed_continuity_min_median'][1]:.4f}**", ""]
md += ["## Initial observations", "",
       "1. The raw files are whitespace-separated and have no headers: `.track` = longitude, latitude, timestamp; `.nodes` = longitude, latitude; `.arcs` = from-node ID, to-node ID; `.route` = chronological arc IDs.",
       "2. Timestamp gaps need explicit handling. The training partition contains trips with very large gaps; do not assume every GPS row is exactly one second apart.",
       "3. Reference-route IDs should be validated against arc-row indices, and arc endpoints against node-row indices before modelling.",
       "4. The nearest-road baseline in this kit does not consume `.route`; route files are only read by this EDA script for structural checks.",
       "", "## Files", "", "- `trip_inventory.csv` — per-trip inventory and data-quality flags.", "- `summary.json` — aggregate statistics.", "- `gps_observations_by_trip.png` and `max_timestamp_gaps.png` — EDA plots.", ""]
(OUT / "eda_report.md").write_text("\n".join(md), encoding="utf-8")
print("EDA complete:", OUT)
print(json.dumps({k: {"trips": v["trip_count"], "median_gps": v["gps_observations_min_median_max"][1], "max_gap_s": v["max_timestamp_gap_seconds_max"], "1s_fraction_mean": round(v["one_second_interval_fraction_mean"], 3)} for k,v in summary.items()}, indent=2))
