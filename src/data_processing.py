"""
PathSnap - Data Processing & Exploratory Data Analysis
Handles dataset inspection, per-trip Local AEQD metric projection,
dynamic programming route alignment, and cross-partition EDA summary generation.

Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import CRS, Transformer


def list_trips(partition_dir: str):
    """Return sorted list of trip IDs in a partition directory."""
    if not os.path.isdir(partition_dir):
        return []
    return [
        d for d in sorted(os.listdir(partition_dir))
        if os.path.isdir(os.path.join(partition_dir, d))
    ]


def load_trip(trip_dir: str, trip_id: str, load_route: bool = True):
    """Load .track, .nodes, .arcs, and optionally .route for a single trip."""
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")

    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Missing track file: {track_path}")

    track = np.loadtxt(track_path, dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)

    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)

    arcs = np.loadtxt(arcs_path, dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route = None
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    if load_route and os.path.exists(route_path):
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = np.array([int(route)], dtype=np.int64)

    return {
        "trip_id": trip_id,
        "track": track,   # (N_gps, 3): lon, lat, timestamp
        "nodes": nodes,   # (N_nodes, 2): lon, lat
        "arcs": arcs,     # (N_arcs, 2): from_node_id, to_node_id
        "route": route,   # (N_route,) or None
    }


def make_local_aeqd_transformer(lon0: float, lat0: float):
    """
    Create a per-trip Local Azimuthal Equidistant (AEQD) projection centered at (lon0, lat0)
    so all coordinates are in true meters regardless of geographical latitude.
    """
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +datum=WGS84 +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    to_local = Transformer.from_crs(wgs84, local_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(local_crs, wgs84, always_xy=True)
    return to_local, to_wgs84


def project_trip_to_meters(track: np.ndarray, nodes: np.ndarray):
    """Project both track and road-network nodes into a trip-centered AEQD metric frame."""
    lon0 = float(np.median(track[:, 0]))
    lat0 = float(np.median(track[:, 1]))
    to_local, to_wgs84 = make_local_aeqd_transformer(lon0, lat0)
    tx, ty = to_local.transform(track[:, 0], track[:, 1])
    nx, ny = to_local.transform(nodes[:, 0], nodes[:, 1])
    track_xy = np.column_stack([np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)])
    nodes_xy = np.column_stack([np.asarray(nx, dtype=np.float64), np.asarray(ny, dtype=np.float64)])
    return track_xy, nodes_xy, (lon0, lat0), to_wgs84


def point_to_segments_dist(px, py, ax, ay, bx, by):
    """
    Vectorized distance from a single point (px, py) to M line segments (ax, ay) -> (bx, by).
    Returns:
      dists: (M,) Euclidean distance in meters
      proj_t: (M,) clamped projection parameter in [0, 1]
      proj_x, proj_y: (M,) closest point on each segment
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / safe_len_sq
    t = np.where(seg_len_sq > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dists = np.hypot(px - proj_x, py - proj_y)
    return dists, t, proj_x, proj_y


def align_gps_to_route(track_xy: np.ndarray, nodes_xy: np.ndarray, arcs: np.ndarray, route: np.ndarray):
    """
    Monotonically align each GPS observation to an arc in the ground-truth .route sequence
    using Dynamic Programming (minimizing total point-to-segment distance while respecting
    chronological order along the route).
    """
    n_gps = len(track_xy)
    n_route = len(route)
    route_arcs = arcs[route]
    ax = nodes_xy[route_arcs[:, 0], 0]
    ay = nodes_xy[route_arcs[:, 0], 1]
    bx = nodes_xy[route_arcs[:, 1], 0]
    by = nodes_xy[route_arcs[:, 1], 1]

    cost = np.empty((n_gps, n_route), dtype=np.float64)
    for i in range(n_gps):
        d, _, _, _ = point_to_segments_dist(track_xy[i, 0], track_xy[i, 1], ax, ay, bx, by)
        cost[i] = d

    dp = np.full((n_gps, n_route), np.inf, dtype=np.float64)
    back = np.zeros((n_gps, n_route), dtype=np.int32)
    dp[0] = cost[0]

    for i in range(1, n_gps):
        running_min = np.minimum.accumulate(dp[i - 1])
        cur_val = dp[i - 1, 0]
        cur_idx = 0
        best_k = np.zeros(n_route, dtype=np.int32)
        for j in range(n_route):
            if dp[i - 1, j] < cur_val:
                cur_val = dp[i - 1, j]
                cur_idx = j
            best_k[j] = cur_idx
        dp[i] = running_min + cost[i]
        back[i] = best_k

    obs_route_idx = np.zeros(n_gps, dtype=np.int32)
    obs_route_idx[-1] = int(np.argmin(dp[-1]))
    for i in range(n_gps - 2, -1, -1):
        obs_route_idx[i] = back[i + 1, obs_route_idx[i + 1]]

    obs_arc_ids = route[obs_route_idx]
    obs_dists = cost[np.arange(n_gps), obs_route_idx]
    return obs_arc_ids, obs_route_idx, obs_dists


def run_eda(base_data_dir: str, output_dir: str):
    """Run comprehensive EDA across train (80) and student_test (10) partitions."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for partition in ["train", "student_test"]:
        part_dir = os.path.join(base_data_dir, partition)
        trips = list_trips(part_dir)
        if not trips:
            print(f"Warning: No trips found in {part_dir}")
            continue

        for tid in trips:
            tdir = os.path.join(part_dir, tid)
            data = load_trip(tdir, tid, load_route=True)
            track = data["track"]
            nodes = data["nodes"]
            arcs = data["arcs"]
            route = data["route"]

            n_gps = len(track)
            n_nodes = len(nodes)
            n_arcs = len(arcs)
            n_route = len(route) if route is not None else 0

            n_nan_track = int(np.isnan(track).sum() + np.isinf(track).sum())
            n_nan_nodes = int(np.isnan(nodes).sum() + np.isinf(nodes).sum())

            ts = track[:, 2]
            dt = np.diff(ts)
            non_monotonic = int((dt <= 0).sum())
            dt_eq_1 = int(np.isclose(dt, 1.0, atol=1e-3).sum()) if len(dt) > 0 else 0
            dt_gt_1 = int((dt > 1.0 + 1e-3).sum()) if len(dt) > 0 else 0
            max_gap = float(dt.max()) if len(dt) > 0 else 0.0
            min_dt = float(dt.min()) if len(dt) > 0 else 0.0
            duration_s = float(ts[-1] - ts[0]) if n_gps > 1 else 0.0

            lon_min, lon_max = float(track[:, 0].min()), float(track[:, 0].max())
            lat_min, lat_max = float(track[:, 1].min()), float(track[:, 1].max())

            track_xy, nodes_xy, _, _ = project_trip_to_meters(track, nodes)
            step_dists = np.hypot(np.diff(track_xy[:, 0]), np.diff(track_xy[:, 1])) if n_gps > 1 else np.array([0.0])
            total_gps_len = float(step_dists.sum())

            adj = csr_matrix(
                (np.ones(n_arcs, dtype=np.int8), (arcs[:, 0], arcs[:, 1])),
                shape=(n_nodes, n_nodes)
            )
            n_comp, _ = connected_components(adj, directed=False)
            deg_out = np.bincount(arcs[:, 0], minlength=n_nodes)
            deg_in = np.bincount(arcs[:, 1], minlength=n_nodes)
            isolated_nodes = int(((deg_out + deg_in) == 0).sum())
            self_loops = int((arcs[:, 0] == arcs[:, 1]).sum())

            route_discont = 0
            if route is not None and len(route) > 1:
                r_arcs = arcs[route]
                route_discont = int((r_arcs[:-1, 1] != r_arcs[1:, 0]).sum())

            obs_dists = np.zeros(n_gps, dtype=np.float64)
            if route is not None and len(route) > 0:
                _, _, obs_dists = align_gps_to_route(track_xy, nodes_xy, arcs, route)

            records.append({
                "partition": partition,
                "trip_id": tid,
                "n_gps": n_gps,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "n_route_arcs": n_route,
                "n_nan_track": n_nan_track,
                "n_nan_nodes": n_nan_nodes,
                "non_monotonic_ts": non_monotonic,
                "dt_eq_1s": dt_eq_1,
                "dt_gt_1s": dt_gt_1,
                "min_dt_s": min_dt,
                "max_gap_s": max_gap,
                "duration_s": duration_s,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "gps_length_m": total_gps_len,
                "n_components": int(n_comp),
                "isolated_nodes": isolated_nodes,
                "self_loops": self_loops,
                "route_discontinuities": route_discont,
                "mean_gps_to_route_m": float(obs_dists.mean()),
                "median_gps_to_route_m": float(np.median(obs_dists)),
                "p95_gps_to_route_m": float(np.percentile(obs_dists, 95)),
                "max_gps_to_route_m": float(obs_dists.max()),
            })

    if not records:
        print("No trip data found to generate EDA report.")
        return None, None

    df = pd.DataFrame(records)
    csv_out = os.path.join(output_dir, "eda_trip_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved EDA trip summary: {csv_out}")

    summary = {}
    for part in ["train", "student_test"]:
        sub = df[df["partition"] == part]
        if len(sub) == 0:
            continue
        total_steps = int(sub["n_gps"].sum() - len(sub))
        summary[part] = {
            "num_trips": int(len(sub)),
            "total_gps_obs": int(sub["n_gps"].sum()),
            "gps_obs_min": int(sub["n_gps"].min()),
            "gps_obs_max": int(sub["n_gps"].max()),
            "gps_obs_mean": round(float(sub["n_gps"].mean()), 2),
            "total_nan": int(sub["n_nan_track"].sum() + sub["n_nan_nodes"].sum()),
            "non_monotonic_ts_total": int(sub["non_monotonic_ts"].sum()),
            "pct_1hz_steps": round(100.0 * int(sub["dt_eq_1s"].sum()) / max(total_steps, 1), 2),
            "trips_with_gaps_gt_5s": int((sub["max_gap_s"] > 5.0).sum()),
            "max_gap_s_overall": float(sub["max_gap_s"].max()),
            "lon_range": [round(float(sub["lon_min"].min()), 5), round(float(sub["lon_max"].max()), 5)],
            "lat_range": [round(float(sub["lat_min"].min()), 5), round(float(sub["lat_max"].max()), 5)],
            "nodes_mean": round(float(sub["n_nodes"].mean()), 1),
            "arcs_mean": round(float(sub["n_arcs"].mean()), 1),
            "isolated_nodes_total": int(sub["isolated_nodes"].sum()),
            "self_loops_total": int(sub["self_loops"].sum()),
            "route_arcs_mean": round(float(sub["n_route_arcs"].mean()), 1),
            "route_discontinuities_total": int(sub["route_discontinuities"].sum()),
            "mean_gps_to_route_m": round(float(sub["mean_gps_to_route_m"].mean()), 2),
            "p95_gps_to_route_m": round(float(sub["p95_gps_to_route_m"].mean()), 2),
            "max_gps_to_route_m": round(float(sub["max_gps_to_route_m"].max()), 2),
        }

    json_out = os.path.join(output_dir, "eda_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved EDA summary JSON: {json_out}")

    return df, summary


def resolve_default_data_dir():
    """Locate dataset folder using environment variable or sensible relative paths."""
    env_dir = os.environ.get("PATHSNAP_DATA_DIR")
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(project_root, "data", "Student folder", "Student folder"),
        os.path.join(project_root, "data", "Student folder"),
        os.path.join(project_root, "data"),
        os.path.join(project_root, "..", "PathSnap_SRNs_First_Round_Kit", "data", "Student folder"),
    ]
    for c in candidates:
        if os.path.isdir(c) and (os.path.isdir(os.path.join(c, "student_test")) or os.path.isdir(os.path.join(c, "train"))):
            return os.path.abspath(c)
    return os.path.join(project_root, "data", "Student folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap - Dataset Processing & EDA")
    parser.add_argument("--data-dir", default=None, help="Path to base data directory containing train and student_test")
    parser.add_argument("--output-dir", default=None, help="Directory to save EDA outputs")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = args.data_dir if args.data_dir else resolve_default_data_dir()
    out_dir = args.output_dir if args.output_dir else os.path.join(project_root, "results")

    print(f"PathSnap Data Processing & EDA")
    print(f"  Data directory:   {data_dir}")
    print(f"  Output directory: {out_dir}")

    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}")
        print("Please place the dataset in data/Student folder or pass --data-dir <path>.")
        sys.exit(1)

    run_eda(data_dir, out_dir)
