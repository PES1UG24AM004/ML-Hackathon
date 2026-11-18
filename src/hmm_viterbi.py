"""
PathSnap - HMM Emission/Transition Modelling & Viterbi Route Reconstruction
Combines MLP classifier emissions with graph topology transition probabilities
to decode continuous, plausible topological routes from discrete GPS candidates.

Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

from data_processing import (
    list_trips,
    load_trip,
    project_trip_to_meters,
    align_gps_to_route,
    resolve_default_data_dir,
)
from feature_engineering import generate_candidates_for_trip, build_network_index
from models import MLPClassifierNP


def load_final_mlp(model_path: str):
    """Load serialized MLP model weights and scaler statistics from npz archive."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model artifact not found: {model_path}")
    d = np.load(model_path, allow_pickle=True)
    params = json.loads(str(d["best_params_json"]))
    model = MLPClassifierNP(
        hidden_dim=params["hidden_dim"],
        lr=params["lr"],
        epochs=params["epochs"],
        weight_decay=params["weight_decay"],
    )
    model.W1 = d["W1"]
    model.b1 = d["b1"]
    model.W2 = d["W2"]
    model.b2 = d["b2"]
    mean = d["scaler_mean"]
    std = d["scaler_std"]
    return model, mean, std, params


def build_two_hop_set(arcs: np.ndarray):
    """Build fast adjacency dict: node -> set of outgoing nodes, for 1-hop and 2-hop checks."""
    out_nodes = {}
    for u, v in arcs:
        u_i, v_i = int(u), int(v)
        if u_i not in out_nodes:
            out_nodes[u_i] = set()
        out_nodes[u_i].add(v_i)
    return out_nodes


def viterbi_decode_trip(
    cand_probs: np.ndarray,
    obs_idx: np.ndarray,
    cand_arcs: np.ndarray,
    proj_t_all: np.ndarray,
    arcs: np.ndarray,
    nodes_xy: np.ndarray,
    track_xy: np.ndarray,
    timestamps: np.ndarray,
    out_nodes: dict,
    emission_weight: float = 1.0,
    hop1_penalty: float = 0.25,
    hop2_penalty: float = 1.2,
    disc_penalty: float = 5.5,
):
    """
    Viterbi decoding in log-probability space over per-observation candidate road arcs.
    Handles GPS gaps (dt > 3s) and disconnected segments gracefully.
    """
    n_obs = int(obs_idx.max()) + 1
    split_pts = np.flatnonzero(np.diff(obs_idx)) + 1
    groups = np.split(np.arange(len(obs_idx)), split_pts)

    V = []
    backptrs = []

    # Initialize t = 0
    g0 = groups[0]
    p0 = np.clip(cand_probs[g0], 1e-6, 1.0 - 1e-6)
    V.append(emission_weight * np.log(p0))

    for t in range(1, n_obs):
        g_prev = groups[t - 1]
        g_cur = groups[t]
        arcs_prev = cand_arcs[g_prev]
        arcs_cur = cand_arcs[g_cur]
        pt_prev = proj_t_all[g_prev]
        pt_cur = proj_t_all[g_cur]

        dt = max(float(timestamps[t] - timestamps[t - 1]), 1.0)
        gap_scale = 1.0 / (1.0 + 0.15 * max(0.0, dt - 2.0))

        k_prev = len(arcs_prev)
        k_cur = len(arcs_cur)
        trans = np.full((k_prev, k_cur), -disc_penalty * gap_scale, dtype=np.float64)

        for i in range(k_prev):
            a_i = int(arcs_prev[i])
            u_i, v_i = int(arcs[a_i, 0]), int(arcs[a_i, 1])
            v_i_out = out_nodes.get(v_i, ())

            for j in range(k_cur):
                a_j = int(arcs_cur[j])
                u_j = int(arcs[a_j, 0])

                if a_i == a_j:
                    if pt_cur[j] >= pt_prev[i] - 0.05:
                        trans[i, j] = 0.0
                    else:
                        trans[i, j] = -0.5
                elif u_j == v_i:
                    trans[i, j] = -hop1_penalty
                elif u_j in v_i_out:
                    trans[i, j] = -hop2_penalty

        p_cur = np.clip(cand_probs[g_cur], 1e-6, 1.0 - 1e-6)
        log_emis = emission_weight * np.log(p_cur)

        scores = V[t - 1][:, None] + trans
        best_prev = np.argmax(scores, axis=0)
        best_score = scores[best_prev, np.arange(k_cur)] + log_emis

        V.append(best_score)
        backptrs.append(best_prev)

    best_last = int(np.argmax(V[-1]))
    path = [best_last]
    for t in range(n_obs - 1, 0, -1):
        prev = int(backptrs[t - 1][path[-1]])
        path.append(prev)
    path.reverse()

    pred_arcs = np.empty(n_obs, dtype=np.int64)
    for t, sel_idx in enumerate(path):
        pred_arcs[t] = cand_arcs[groups[t][sel_idx]]

    return pred_arcs


def collapse_consecutive(arr):
    """Collapse adjacent duplicate elements in sequence."""
    out = []
    for x in arr:
        xi = int(x)
        if len(out) == 0 or out[-1] != xi:
            out.append(xi)
    return out


def compute_lcs_length(seq_a, seq_b):
    """
    Compute length of Longest Common Subsequence (LCS) between two integer sequences
    using space-optimized dynamic programming. Order-sensitive evaluation metric.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def sequence_metrics(pred_seq: np.ndarray, gt_route: np.ndarray, arcs: np.ndarray):
    """
    Compute Route Continuity, Unique Arc Jaccard/F1 (set-based),
    and Longest Common Subsequence (LCS) metrics (order-sensitive).
    """
    n = len(pred_seq)
    same = pred_seq[:-1] == pred_seq[1:]
    connected = arcs[pred_seq[:-1], 1] == arcs[pred_seq[1:], 0]
    continuity = float(np.mean(same | connected)) if n > 1 else 1.0

    pred_collapsed = collapse_consecutive(pred_seq)
    gt_list = [int(x) for x in gt_route]

    # 1. Set-based unique arc metrics (unordered coverage)
    set_p, set_g = set(pred_collapsed), set(gt_list)
    inter = len(set_p & set_g)
    prec = inter / max(len(set_p), 1)
    rec = inter / max(len(set_g), 1)
    arc_f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    jaccard = inter / max(len(set_p | set_g), 1)

    # 2. Order-sensitive sequence metrics via Longest Common Subsequence (LCS)
    lcs_len = compute_lcs_length(pred_collapsed, gt_list)
    lcs_prec = lcs_len / max(len(pred_collapsed), 1)
    lcs_rec = lcs_len / max(len(gt_list), 1)
    lcs_f1 = (2 * lcs_prec * lcs_rec) / max(lcs_prec + lcs_rec, 1e-12)

    return {
        "continuity": continuity,
        "arc_f1": float(arc_f1),
        "arc_jaccard": float(jaccard),
        "lcs_length": int(lcs_len),
        "lcs_f1": float(lcs_f1),
        "lcs_ratio": float(lcs_rec),
        "pred_unique_arcs": len(set_p),
        "gt_unique_arcs": len(set_g),
    }


def run_hmm_evaluation(data_dir: str, artifacts_dir: str, results_dir: str):
    """Run HMM/Viterbi evaluation across student-testing partition trips."""
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "final_mlp_model.npz")
    model, mean, std, _ = load_final_mlp(model_path)

    student_test_dir = os.path.join(data_dir, "student_test")
    trips = list_trips(student_test_dir)
    if not trips:
        print(f"Error: No student_test trips found in: {student_test_dir}")
        sys.exit(1)

    rows = []
    t0 = time.time()

    for tid in trips:
        trip_dir = os.path.join(student_test_dir, tid)
        data = load_trip(trip_dir, tid, load_route=True)
        track_xy, nodes_xy, _, _ = project_trip_to_meters(data["track"], data["nodes"])
        gt_obs_arcs, _, _ = align_gps_to_route(track_xy, nodes_xy, data["arcs"], data["route"])

        net_idx = build_network_index(nodes_xy, data["arcs"])
        X, obs_idx, cand_arcs, _ = generate_candidates_for_trip(
            track_xy, data["track"][:, 2], nodes_xy, data["arcs"], top_k=8, net_idx=net_idx
        )
        X_scaled = ((X - mean) / std).astype(np.float32)
        probs = model.predict_proba(X_scaled)[:, 1]
        proj_t_all = X[:, 6]

        # 1. Classifier-only predictions (greedy argmax per observation)
        n_obs = len(track_xy)
        split_pts = np.flatnonzero(np.diff(obs_idx)) + 1
        groups = np.split(np.arange(len(obs_idx)), split_pts)
        clf_pred = np.array([cand_arcs[g[int(np.argmax(probs[g]))]] for g in groups], dtype=np.int64)

        # 2. Classifier + HMM/Viterbi predictions
        out_nodes = build_two_hop_set(data["arcs"])
        hmm_pred = viterbi_decode_trip(
            probs, obs_idx, cand_arcs, proj_t_all, data["arcs"], nodes_xy,
            track_xy, data["track"][:, 2], out_nodes,
            emission_weight=1.0, hop1_penalty=0.25, hop2_penalty=1.2, disc_penalty=5.5
        )

        clf_top1 = float(np.mean(clf_pred == gt_obs_arcs))
        hmm_top1 = float(np.mean(hmm_pred == gt_obs_arcs))
        clf_seq = sequence_metrics(clf_pred, data["route"], data["arcs"])
        hmm_seq = sequence_metrics(hmm_pred, data["route"], data["arcs"])

        rows.append({
            "trip_id": tid,
            "n_gps": n_obs,
            "clf_top1_acc": round(clf_top1, 4),
            "hmm_top1_acc": round(hmm_top1, 4),
            "clf_continuity": round(clf_seq["continuity"], 4),
            "hmm_continuity": round(hmm_seq["continuity"], 4),
            "clf_arc_f1": round(clf_seq["arc_f1"], 4),
            "hmm_arc_f1": round(hmm_seq["arc_f1"], 4),
            "clf_jaccard": round(clf_seq["arc_jaccard"], 4),
            "hmm_jaccard": round(hmm_seq["arc_jaccard"], 4),
            "clf_lcs_f1": round(clf_seq["lcs_f1"], 4),
            "hmm_lcs_f1": round(hmm_seq["lcs_f1"], 4),
        })

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    csv_out = os.path.join(results_dir, "hmm_vs_classifier_trips.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    summary = {
        "eval_time_s": round(elapsed, 2),
        "hmm_params": {
            "emission_weight": 1.0,
            "hop1_penalty": 0.25,
            "hop2_penalty": 1.2,
            "disc_penalty": 5.5,
        },
        "classifier_only": {
            "mean_top1_acc": round(float(df["clf_top1_acc"].mean()), 4),
            "mean_continuity": round(float(df["clf_continuity"].mean()), 4),
            "mean_arc_f1": round(float(df["clf_arc_f1"].mean()), 4),
            "mean_jaccard": round(float(df["clf_jaccard"].mean()), 4),
            "mean_lcs_f1": round(float(df["clf_lcs_f1"].mean()), 4),
        },
        "classifier_plus_hmm": {
            "mean_top1_acc": round(float(df["hmm_top1_acc"].mean()), 4),
            "mean_continuity": round(float(df["hmm_continuity"].mean()), 4),
            "mean_arc_f1": round(float(df["hmm_arc_f1"].mean()), 4),
            "mean_jaccard": round(float(df["hmm_jaccard"].mean()), 4),
            "mean_lcs_f1": round(float(df["hmm_lcs_f1"].mean()), 4),
        },
    }
    json_out = os.path.join(results_dir, "hmm_comparison_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_out}")

    cfg_out = os.path.join(artifacts_dir, "hmm_config.json")
    with open(cfg_out, "w") as f:
        json.dump(summary["hmm_params"], f, indent=2)
    print(f"Saved: {cfg_out}")

    print("\nHMM Comparison Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap - HMM & Viterbi Route Reconstruction")
    parser.add_argument("--data-dir", default=None, help="Base dataset directory containing student_test/")
    parser.add_argument("--artifacts-dir", default=None, help="Directory containing final_mlp_model.npz")
    parser.add_argument("--results-dir", default=None, help="Directory to save HMM evaluation results")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = args.data_dir if args.data_dir else resolve_default_data_dir()
    art_dir = args.artifacts_dir if args.artifacts_dir else os.path.join(project_root, "artifacts")
    res_dir = args.results_dir if args.results_dir else os.path.join(project_root, "results")

    run_hmm_evaluation(data_dir, art_dir, res_dir)
