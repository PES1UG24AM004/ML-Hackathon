"""
PathSnap - Candidate Generation & Feature Engineering
Constructs spatial KD-tree over road segment midpoints and extracts
16 geometric, kinematic, and topological features per candidate arc.

Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import sys
import time
import argparse
import numpy as np
from scipy.spatial import cKDTree

from data_processing import (
    list_trips,
    load_trip,
    project_trip_to_meters,
    point_to_segments_dist,
    align_gps_to_route,
    resolve_default_data_dir,
)

FEATURE_NAMES = [
    "perp_dist",
    "dist_rank",
    "dist_diff_min",
    "dist_ratio_min",
    "heading_diff_deg",
    "cos_heading_diff",
    "proj_t",
    "arc_length_m",
    "gps_speed_mps",
    "dt_prev",
    "step_dist_prev",
    "prev_dist_to_cand",
    "next_dist_to_cand",
    "local_mean_dist",
    "cand_out_degree",
    "cand_in_degree",
]


def build_network_index(nodes_xy: np.ndarray, arcs: np.ndarray):
    """Build spatial KD-tree on arc midpoints and node-to-arc adjacency structures."""
    ax = nodes_xy[arcs[:, 0], 0]
    ay = nodes_xy[arcs[:, 0], 1]
    bx = nodes_xy[arcs[:, 1], 0]
    by = nodes_xy[arcs[:, 1], 1]

    midpoints = np.column_stack([0.5 * (ax + bx), 0.5 * (ay + by)])
    mid_tree = cKDTree(midpoints)
    node_tree = cKDTree(nodes_xy)

    arc_dx = bx - ax
    arc_dy = by - ay
    arc_len = np.hypot(arc_dx, arc_dy)
    arc_bearing = np.arctan2(arc_dy, arc_dx)  # radians in [-pi, pi]

    n_nodes = len(nodes_xy)
    out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
    in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)

    node_to_arcs = [[] for _ in range(n_nodes)]
    for arc_id, (u, v) in enumerate(arcs):
        node_to_arcs[u].append(arc_id)
        if v != u:
            node_to_arcs[v].append(arc_id)

    return {
        "ax": ax, "ay": ay, "bx": bx, "by": by,
        "arc_len": arc_len,
        "arc_bearing": arc_bearing,
        "mid_tree": mid_tree,
        "node_tree": node_tree,
        "node_to_arcs": node_to_arcs,
        "out_deg": out_deg,
        "in_deg": in_deg,
    }


def compute_gps_kinematics(track_xy: np.ndarray, timestamps: np.ndarray):
    """Compute per-observation GPS heading (radians), speed (m/s), dt_prev, and step_dist_prev."""
    n = len(track_xy)
    headings = np.zeros(n, dtype=np.float64)
    speeds = np.zeros(n, dtype=np.float64)
    dt_prev = np.ones(n, dtype=np.float64)
    step_dist = np.zeros(n, dtype=np.float64)

    if n == 1:
        return headings, speeds, dt_prev, step_dist

    dx_step = np.diff(track_xy[:, 0])
    dy_step = np.diff(track_xy[:, 1])
    ds_step = np.hypot(dx_step, dy_step)
    dt_step = np.maximum(np.diff(timestamps), 1e-3)

    dt_prev[1:] = dt_step
    step_dist[1:] = ds_step

    for i in range(n):
        if i == 0:
            dx = track_xy[1, 0] - track_xy[0, 0]
            dy = track_xy[1, 1] - track_xy[0, 1]
            dt = max(timestamps[1] - timestamps[0], 1e-3)
        elif i == n - 1:
            dx = track_xy[-1, 0] - track_xy[-2, 0]
            dy = track_xy[-1, 1] - track_xy[-2, 1]
            dt = max(timestamps[-1] - timestamps[-2], 1e-3)
        else:
            dx = track_xy[i + 1, 0] - track_xy[i - 1, 0]
            dy = track_xy[i + 1, 1] - track_xy[i - 1, 1]
            dt = max(timestamps[i + 1] - timestamps[i - 1], 1e-3)
        dist = np.hypot(dx, dy)
        speeds[i] = dist / dt
        if dist > 0.2:
            headings[i] = np.arctan2(dy, dx)
        elif i > 0:
            headings[i] = headings[i - 1]

    return headings, speeds, dt_prev, step_dist


def generate_candidates_for_trip(
    track_xy: np.ndarray,
    timestamps: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    top_k: int = 8,
    search_radius_m: float = 120.0,
    net_idx=None,
):
    """
    Generate top_k candidate road arcs and 16-dim feature vectors for each GPS observation.
    Returns:
      X: (M, 16) float32 feature matrix
      obs_idx: (M,) int32 GPS observation index (0..N_gps-1)
      cand_arc_ids: (M,) int64 arc_id for each candidate row
      obs_cand_counts: (N_gps,) number of candidates for each GPS point
    """
    if net_idx is None:
        net_idx = build_network_index(nodes_xy, arcs)

    ax, ay, bx, by = net_idx["ax"], net_idx["ay"], net_idx["bx"], net_idx["by"]
    arc_len = net_idx["arc_len"]
    arc_bearing = net_idx["arc_bearing"]
    mid_tree = net_idx["mid_tree"]
    node_tree = net_idx["node_tree"]
    node_to_arcs = net_idx["node_to_arcs"]
    out_deg = net_idx["out_deg"]
    in_deg = net_idx["in_deg"]

    headings, speeds, dt_prev, step_dist = compute_gps_kinematics(track_xy, timestamps)
    n_gps = len(track_xy)
    n_arcs = len(arcs)

    k_mid = min(40, n_arcs)
    k_node = min(25, len(nodes_xy))
    _, near_mids = mid_tree.query(track_xy, k=k_mid)
    _, near_nodes = node_tree.query(track_xy, k=k_node)

    feat_rows = []
    obs_idx_list = []
    cand_arc_list = []
    obs_cand_counts = np.zeros(n_gps, dtype=np.int32)

    for i in range(n_gps):
        px, py = track_xy[i, 0], track_xy[i, 1]
        cand_set = set(np.atleast_1d(near_mids[i]).tolist())
        for nid in np.atleast_1d(near_nodes[i]):
            cand_set.update(node_to_arcs[nid])
        cand_ids = np.fromiter(cand_set, dtype=np.int64)

        dists, proj_t, _, _ = point_to_segments_dist(
            px, py, ax[cand_ids], ay[cand_ids], bx[cand_ids], by[cand_ids]
        )

        order = np.argsort(dists)
        within = order[dists[order] <= search_radius_m]
        if len(within) >= top_k:
            sel = within[:top_k]
        elif len(within) > 0:
            sel = order[:min(top_k, len(order))]
        else:
            sel = order[:min(top_k, len(order))]

        sel_arcs = cand_ids[sel]
        sel_dists = dists[sel]
        sel_t = proj_t[sel]

        px_prev, py_prev = (track_xy[i - 1, 0], track_xy[i - 1, 1]) if i > 0 else (px, py)
        px_next, py_next = (track_xy[i + 1, 0], track_xy[i + 1, 1]) if i < n_gps - 1 else (px, py)

        d_prev, _, _, _ = point_to_segments_dist(
            px_prev, py_prev, ax[sel_arcs], ay[sel_arcs], bx[sel_arcs], by[sel_arcs]
        )
        d_next, _, _, _ = point_to_segments_dist(
            px_next, py_next, ax[sel_arcs], ay[sel_arcs], bx[sel_arcs], by[sel_arcs]
        )

        min_d = sel_dists[0]
        ranks = np.arange(len(sel_arcs), dtype=np.float64)

        dtheta = np.abs(headings[i] - arc_bearing[sel_arcs])
        dtheta = np.minimum(dtheta, 2.0 * np.pi - dtheta)
        heading_diff_deg = np.degrees(dtheta)
        cos_h = np.cos(dtheta)

        u_nodes = arcs[sel_arcs, 0]
        v_nodes = arcs[sel_arcs, 1]

        feats = np.column_stack([
            sel_dists,
            ranks,
            sel_dists - min_d,
            sel_dists / (min_d + 1.0),
            heading_diff_deg,
            cos_h,
            sel_t,
            arc_len[sel_arcs],
            np.full(len(sel_arcs), speeds[i]),
            np.full(len(sel_arcs), dt_prev[i]),
            np.full(len(sel_arcs), step_dist[i]),
            d_prev,
            d_next,
            (d_prev + sel_dists + d_next) / 3.0,
            out_deg[v_nodes].astype(np.float64),
            in_deg[u_nodes].astype(np.float64),
        ])

        feat_rows.append(feats)
        obs_idx_list.append(np.full(len(sel_arcs), i, dtype=np.int32))
        cand_arc_list.append(sel_arcs)
        obs_cand_counts[i] = len(sel_arcs)

    X = np.vstack(feat_rows).astype(np.float32)
    obs_idx = np.concatenate(obs_idx_list)
    cand_arc_ids = np.concatenate(cand_arc_list)
    return X, obs_idx, cand_arc_ids, obs_cand_counts


def build_partition_dataset(base_data_dir: str, partition: str, top_k: int = 8):
    """Build candidate-level dataset for all trips in a partition."""
    part_dir = os.path.join(base_data_dir, partition)
    trips = list_trips(part_dir)
    if not trips:
        raise FileNotFoundError(f"No trip subdirectories found in: {part_dir}")

    X_list, y_list, trip_id_list, obs_idx_list, cand_arc_list, gt_arc_list = [], [], [], [], [], []
    total_obs = 0
    missed_obs = 0
    t0 = time.time()

    for idx, tid in enumerate(trips):
        data = load_trip(os.path.join(part_dir, tid), tid, load_route=True)
        track_xy, nodes_xy, _, _ = project_trip_to_meters(data["track"], data["nodes"])
        gt_obs_arcs, _, _ = align_gps_to_route(track_xy, nodes_xy, data["arcs"], data["route"])

        X, obs_idx, cand_arcs, _ = generate_candidates_for_trip(
            track_xy, data["track"][:, 2], nodes_xy, data["arcs"], top_k=top_k
        )
        gt_for_cand = gt_obs_arcs[obs_idx]
        y = (cand_arcs == gt_for_cand).astype(np.int32)

        n_gps = len(track_xy)
        obs_has_hit = np.zeros(n_gps, dtype=bool)
        hit_obs = obs_idx[y == 1]
        obs_has_hit[hit_obs] = True
        total_obs += n_gps
        missed_obs += int((~obs_has_hit).sum())

        X_list.append(X)
        y_list.append(y)
        trip_id_list.append(np.full(len(y), idx, dtype=np.int32))
        obs_idx_list.append(obs_idx)
        cand_arc_list.append(cand_arcs)
        gt_arc_list.append(gt_for_cand)

    elapsed = time.time() - t0
    miss_rate = 100.0 * missed_obs / max(total_obs, 1)
    print(f"[{partition}] Trips={len(trips)}, Obs={total_obs}, CandidateRows={sum(len(y) for y in y_list)}, "
          f"MissedObs={missed_obs} ({miss_rate:.2f}%), Time={elapsed:.1f}s")

    return {
        "X": np.vstack(X_list),
        "y": np.concatenate(y_list),
        "trip_group": np.concatenate(trip_id_list),
        "obs_idx": np.concatenate(obs_idx_list),
        "cand_arcs": np.concatenate(cand_arc_list),
        "gt_arcs": np.concatenate(gt_arc_list),
        "trip_ids": trips,
        "total_obs": total_obs,
        "missed_obs": missed_obs,
        "miss_rate_pct": miss_rate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap - Candidate Generation & Feature Extraction")
    parser.add_argument("--data-dir", default=None, help="Base dataset directory containing train/ and student_test/")
    parser.add_argument("--artifacts-dir", default=None, help="Output directory for compressed candidate npz files")
    parser.add_argument("--top-k", type=int, default=8, help="Number of candidate segments per observation (default: 8)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = args.data_dir if args.data_dir else resolve_default_data_dir()
    art_dir = args.artifacts_dir if args.artifacts_dir else os.path.join(project_root, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    print(f"PathSnap Feature Engineering")
    print(f"  Data directory:      {data_dir}")
    print(f"  Artifacts directory: {art_dir}")
    print(f"  Top-K candidates:    {args.top_k}")

    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}")
        print("Please place the dataset in data/Student folder or pass --data-dir <path>.")
        sys.exit(1)

    train_ds = build_partition_dataset(data_dir, "train", top_k=args.top_k)
    test_ds = build_partition_dataset(data_dir, "student_test", top_k=args.top_k)

    train_out = os.path.join(art_dir, "train_candidates.npz")
    test_out = os.path.join(art_dir, "test_candidates.npz")
    np.savez_compressed(train_out, **train_ds)
    np.savez_compressed(test_out, **test_ds)
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")
