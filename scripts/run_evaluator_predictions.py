"""
PathSnap — Evaluator Prediction Pipeline (Strict Label Isolation)
Discovers evaluator trips, loads ONLY .track, .nodes, .arcs, runs trained PathSnap
inference pipeline, and saves prediction artifacts independently per trip.
CRITICAL: Never accesses evaluator_data/private_labels or any .route files!
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing import load_trip, project_trip_to_meters
from feature_engineering import generate_candidates_for_trip, build_network_index
from hmm_viterbi import load_final_mlp, build_two_hop_set, viterbi_decode_trip, collapse_consecutive


def run_evaluator_predictions():
    inputs_dir = PROJECT_ROOT / "evaluator_data" / "inputs"
    outputs_dir = PROJECT_ROOT / "evaluator_outputs" / "predictions"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "artifacts" / "final_mlp_model.npz"
    hmm_cfg_path = PROJECT_ROOT / "artifacts" / "hmm_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not hmm_cfg_path.exists():
        raise FileNotFoundError(f"Missing HMM config: {hmm_cfg_path}")
    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"Evaluator inputs directory not found: {inputs_dir}")

    # Load model and config
    print(f"[*] Loading trained MLP model from: {model_path}")
    model, scaler_mean, scaler_std, model_params = load_final_mlp(str(model_path))

    with open(hmm_cfg_path, "r", encoding="utf-8") as f:
        hmm_cfg = json.load(f)

    # Discover evaluator trips
    trip_dirs = sorted([d for d in inputs_dir.iterdir() if d.is_dir()])
    print(f"[*] Found {len(trip_dirs)} evaluator trips in {inputs_dir}")

    results_summary = []
    total_t0 = time.time()

    print("\n" + "=" * 80)
    print(f"{'Trip ID':^10} | {'GPS Fixes':^12} | {'Unique Arcs':^14} | {'Continuity':^14} | {'Inference Time':^16}")
    print("-" * 80)

    for trip_path in trip_dirs:
        trip_id = trip_path.name
        t0 = time.time()
        try:
            # STRICT: load_route is False; only reads .track, .nodes, .arcs
            data = load_trip(str(trip_path), trip_id, load_route=False)
            track_wgs = data["track"]
            nodes_wgs = data["nodes"]
            arcs = data["arcs"]

            # Project coordinates to local metric plane
            track_xy, nodes_xy, _, _ = project_trip_to_meters(track_wgs, nodes_wgs)
            net_idx = build_network_index(nodes_xy, arcs)

            # Generate candidate features
            X, obs_idx, cand_arcs, _ = generate_candidates_for_trip(
                track_xy, track_wgs[:, 2], nodes_xy, arcs, top_k=8, net_idx=net_idx
            )

            # Feature scaling using fixed training scaler parameters
            X_scaled = ((X - scaler_mean) / scaler_std).astype(np.float32)
            probs = model.predict_proba(X_scaled)[:, 1]

            # Point-wise classifier predictions (greedy argmax)
            split_pts = np.flatnonzero(np.diff(obs_idx)) + 1
            groups = np.split(np.arange(len(obs_idx)), split_pts)
            clf_preds = np.array([cand_arcs[g[int(np.argmax(probs[g]))]] for g in groups], dtype=np.int64)

            # HMM Viterbi decoding
            out_nodes = build_two_hop_set(arcs)
            hmm_preds = viterbi_decode_trip(
                probs, obs_idx, cand_arcs, X[:, 6], arcs, nodes_xy,
                track_xy, track_wgs[:, 2], out_nodes,
                emission_weight=hmm_cfg.get("emission_weight", 1.0),
                hop1_penalty=hmm_cfg.get("hop1_penalty", 0.25),
                hop2_penalty=hmm_cfg.get("hop2_penalty", 1.20),
                disc_penalty=hmm_cfg.get("disc_penalty", 5.50),
            )

            collapsed_route = collapse_consecutive(hmm_preds)
            elapsed = round(time.time() - t0, 3)

            # Calculate prediction-time diagnostics (without ground truth)
            # Directed route continuity of predicted sequence
            if len(collapsed_route) > 1:
                cont_matches = sum(
                    arcs[collapsed_route[i], 1] == arcs[collapsed_route[i + 1], 0]
                    for i in range(len(collapsed_route) - 1)
                )
                pred_continuity = round(float(cont_matches / (len(collapsed_route) - 1)), 4)
            else:
                pred_continuity = 1.0

            # Save individual trip predictions
            pred_file = outputs_dir / f"{trip_id}_prediction.json"
            pred_route_txt = outputs_dir / f"{trip_id}.route"

            prediction_payload = {
                "trip_id": trip_id,
                "n_gps_observations": int(len(track_wgs)),
                "n_road_nodes": int(len(nodes_wgs)),
                "n_road_arcs": int(len(arcs)),
                "inference_time_s": elapsed,
                "predicted_arc_sequence": [int(a) for a in collapsed_route],
                "point_level_hmm_preds": [int(a) for a in hmm_preds],
                "point_level_clf_preds": [int(a) for a in clf_preds],
                "diagnostics": {
                    "unique_predicted_arcs": len(set(collapsed_route)),
                    "collapsed_sequence_length": len(collapsed_route),
                    "raw_observation_count": len(hmm_preds),
                    "predicted_topological_continuity": pred_continuity,
                }
            }

            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(prediction_payload, f, indent=2)

            # Save predicted route in official headerless newline format
            np.savetxt(pred_route_txt, collapsed_route, fmt="%d")

            summary_item = {
                "trip_id": trip_id,
                "n_gps": int(len(track_wgs)),
                "unique_arcs": len(set(collapsed_route)),
                "continuity": pred_continuity,
                "inference_time_s": elapsed,
                "status": "SUCCESS"
            }
            results_summary.append(summary_item)

            print(f"{trip_id:^10} | {len(track_wgs):^12,} | {len(set(collapsed_route)):^14,} | {pred_continuity*100:^13.1f}% | {elapsed:^15.3f}s")

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"{trip_id:^10} | {'FAILED':^12} | {'-':^14} | {'-':^14} | {elapsed:^15.3f}s -- Error: {e}")
            results_summary.append({
                "trip_id": trip_id,
                "status": "FAILED",
                "error": str(e),
                "inference_time_s": elapsed
            })

    total_time = round(time.time() - total_t0, 3)
    print("=" * 80)
    print(f"[SUMMARY] Processed {len(results_summary)} trips in {total_time}s.")

    # Save summary manifest
    summary_path = outputs_dir / "prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_trips": len(results_summary),
            "successful_trips": sum(1 for s in results_summary if s.get("status") == "SUCCESS"),
            "total_inference_time_s": total_time,
            "mean_inference_time_s": round(float(np.mean([s["inference_time_s"] for s in results_summary if "inference_time_s" in s])), 3),
            "trips": results_summary
        }, f, indent=2)

    print(f"[SAVED] Predictions saved to: {outputs_dir}")
    print(f"[SAVED] Execution summary saved to: {summary_path}")
    return results_summary


if __name__ == "__main__":
    run_evaluator_predictions()
