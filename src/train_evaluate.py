"""
PathSnap - Multi-Model Training, Group-Aware 5-Fold CV & Student-Test Evaluation
Evaluates six distinct models across five algorithm families on candidate road arcs.

Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

from data_processing import load_trip, resolve_default_data_dir
from models import (
    StandardScalerNP,
    LogisticRegressionNP,
    GaussianNaiveBayesNP,
    DistanceWeightedKNN,
    RandomForestNP,
    GradientBoostingNP,
    MLPClassifierNP,
)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Compute binary classification metrics and 2x2 confusion matrix."""
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = (tp + tn) / max(len(y_true), 1)
    prec_1 = tp / max(tp + fp, 1)
    rec_1 = tp / max(tp + fn, 1)
    f1_1 = (2 * prec_1 * rec_1) / max(prec_1 + rec_1, 1e-12)

    prec_0 = tn / max(tn + fn, 1)
    rec_0 = tn / max(tn + fp, 1)
    f1_0 = (2 * prec_0 * rec_0) / max(prec_0 + rec_0, 1e-12)

    macro_f1 = 0.5 * (f1_1 + f1_0)
    p_clip = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
    log_loss = float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1.0 - p_clip)))

    return {
        "accuracy": float(acc),
        "precision_pos": float(prec_1),
        "recall_pos": float(rec_1),
        "f1_pos": float(f1_1),
        "macro_f1": float(macro_f1),
        "log_loss": log_loss,
        "cm": [[tn, fp], [fn, tp]],
    }


def compute_observation_top1_and_continuity(
    y_prob: np.ndarray,
    trip_group: np.ndarray,
    obs_idx: np.ndarray,
    cand_arcs: np.ndarray,
    gt_arcs: np.ndarray,
    arcs_by_trip: dict = None,
):
    """Compute top-1 selection accuracy and route continuity across observations."""
    unique_trips = np.unique(trip_group)
    correct_obs = 0
    total_obs = 0
    cont_ok = 0
    cont_total = 0

    for tg in unique_trips:
        t_mask = (trip_group == tg)
        t_obs = obs_idx[t_mask]
        t_prob = y_prob[t_mask]
        t_cand = cand_arcs[t_mask]
        t_gt = gt_arcs[t_mask]

        n_obs = int(t_obs.max()) + 1
        pred_seq = np.zeros(n_obs, dtype=np.int64)
        gt_seq = np.zeros(n_obs, dtype=np.int64)

        split_pts = np.flatnonzero(np.diff(t_obs)) + 1
        groups = np.split(np.arange(len(t_obs)), split_pts)
        for g in groups:
            o = t_obs[g[0]]
            best = g[int(np.argmax(t_prob[g]))]
            pred_seq[o] = t_cand[best]
            gt_seq[o] = t_gt[best]

        correct_obs += int(np.sum(pred_seq == gt_seq))
        total_obs += n_obs

        if arcs_by_trip is not None and int(tg) in arcs_by_trip and n_obs > 1:
            arcs = arcs_by_trip[int(tg)]
            same = pred_seq[:-1] == pred_seq[1:]
            connected = arcs[pred_seq[:-1], 1] == arcs[pred_seq[1:], 0]
            cont_ok += int(np.sum(same | connected))
            cont_total += (n_obs - 1)

    top1_acc = correct_obs / max(total_obs, 1)
    continuity = (cont_ok / max(cont_total, 1)) if cont_total > 0 else 0.0
    return float(top1_acc), float(continuity)


def get_model_configs():
    """Return model specification dictionaries for all six model families."""
    return [
        {
            "name": "Logistic Regression",
            "family": "Linear and probabilistic",
            "needs_scaling": True,
            "key_config": "L2 C=1.0, L-BFGS-B, max_iter=60",
            "builder": lambda: LogisticRegressionNP(C=1.0, max_iter=60),
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Linear and probabilistic",
            "needs_scaling": True,
            "key_config": "var_smoothing=1e-6",
            "builder": lambda: GaussianNaiveBayesNP(var_smoothing=1e-6),
        },
        {
            "name": "Distance-Weighted kNN",
            "family": "Instance-based",
            "needs_scaling": True,
            "key_config": "k=15, Euclidean, 1/d weights, index=25k",
            "builder": lambda: DistanceWeightedKNN(n_neighbors=15, max_index_samples=25000, random_state=42),
        },
        {
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "needs_scaling": False,
            "key_config": "n_estimators=15, max_depth=8, max_features=4",
            "builder": lambda: RandomForestNP(n_estimators=15, max_depth=8, min_samples_leaf=25, max_features=4, max_samples=60000, random_state=42),
        },
        {
            "name": "Gradient Boosted Trees",
            "family": "Boosting",
            "needs_scaling": False,
            "key_config": "n_estimators=35, lr=0.1, max_depth=4, subsample=0.6",
            "builder": lambda: GradientBoostingNP(n_estimators=35, learning_rate=0.1, max_depth=4, min_samples_leaf=30, subsample=0.6, max_train_samples=80000, random_state=42),
        },
        {
            "name": "Shallow MLP (Neural Net)",
            "family": "Neural networks",
            "needs_scaling": True,
            "key_config": "hidden=64 ReLU, Adam lr=1e-3, epochs=12, wd=1e-4",
            "builder": lambda: MLPClassifierNP(hidden_dim=64, lr=1e-3, epochs=12, batch_size=8192, weight_decay=1e-4, random_state=42),
        },
    ]


def run_training_pipeline(data_dir: str, artifacts_dir: str, results_dir: str):
    """Execute 5-fold cross-validation and evaluation across all 6 models."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    train_npz_path = os.path.join(artifacts_dir, "train_candidates.npz")
    test_npz_path = os.path.join(artifacts_dir, "test_candidates.npz")

    if not os.path.exists(train_npz_path) or not os.path.exists(test_npz_path):
        print("Candidate datasets not found in artifacts directory:")
        print(f"  Expected: {train_npz_path}")
        print(f"  Expected: {test_npz_path}")
        print("Please run `python src/feature_engineering.py` first to generate candidates from raw data.")
        sys.exit(1)

    print("Loading candidate datasets...")
    train_npz = np.load(train_npz_path, allow_pickle=True)
    test_npz = np.load(test_npz_path, allow_pickle=True)

    X_train, y_train = train_npz["X"], train_npz["y"]
    g_train = train_npz["trip_group"]
    obs_train, cand_train, gt_train = train_npz["obs_idx"], train_npz["cand_arcs"], train_npz["gt_arcs"]

    X_test, y_test = test_npz["X"], test_npz["y"]
    g_test = test_npz["trip_group"]
    obs_test, cand_test, gt_test = test_npz["obs_idx"], test_npz["cand_arcs"], test_npz["gt_arcs"]
    test_trip_ids = list(test_npz["trip_ids"])

    test_arcs_by_trip = {}
    for idx, tid in enumerate(test_trip_ids):
        trip_path = os.path.join(data_dir, "student_test", str(tid))
        if os.path.isdir(trip_path):
            d = load_trip(trip_path, str(tid), load_route=False)
            test_arcs_by_trip[idx] = d["arcs"]

    unique_groups = np.unique(g_train)
    rng = np.random.default_rng(42)
    shuffled_groups = rng.permutation(unique_groups)
    folds = np.array_split(shuffled_groups, 5)

    configs = get_model_configs()
    summary_rows = []
    cv_details = {}
    cms = {}
    mlp_hist = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== Evaluating {name} ({cfg['family']}) ===", flush=True)
        fold_f1s, fold_losses, fold_top1s = [], [], []

        for f_idx, val_groups in enumerate(folds):
            val_mask = np.isin(g_train, val_groups)
            tr_mask = ~val_mask

            X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
            X_va, y_va = X_train[val_mask], y_train[val_mask]

            if cfg["needs_scaling"]:
                scaler = StandardScalerNP()
                X_tr_in = scaler.fit_transform(X_tr)
                X_va_in = scaler.transform(X_va)
            else:
                X_tr_in, X_va_in = X_tr, X_va

            model = cfg["builder"]()
            model.fit(X_tr_in, y_tr)
            p_va = model.predict_proba(X_va_in)[:, 1]
            m_va = compute_binary_metrics(y_va, p_va)
            t1_va, _ = compute_observation_top1_and_continuity(
                p_va, g_train[val_mask], obs_train[val_mask], cand_train[val_mask], gt_train[val_mask]
            )
            fold_f1s.append(m_va["macro_f1"])
            fold_losses.append(m_va["log_loss"])
            fold_top1s.append(t1_va)

        if cfg["needs_scaling"]:
            scaler_full = StandardScalerNP()
            X_tr_full = scaler_full.fit_transform(X_train)
            X_te_full = scaler_full.transform(X_test)
        else:
            X_tr_full, X_te_full = X_train, X_test

        model_full = cfg["builder"]()
        t0 = time.time()
        if isinstance(model_full, MLPClassifierNP):
            # Internal holdout from training groups to monitor training convergence without leaking student-test
            val_group_internal = folds[0]
            val_mask_internal = np.isin(g_train, val_group_internal)
            model_full.fit(
                X_tr_full[~val_mask_internal],
                y_train[~val_mask_internal],
                X_val=X_tr_full[val_mask_internal],
                y_val=y_train[val_mask_internal],
            )
            mlp_hist = model_full.history_
        else:
            model_full.fit(X_tr_full, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        p_test = model_full.predict_proba(X_te_full)[:, 1]
        inf_time = time.time() - t1

        p_train_eval = model_full.predict_proba(X_tr_full[:150000])[:, 1]
        m_train_sample = compute_binary_metrics(y_train[:150000], p_train_eval)

        m_test = compute_binary_metrics(y_test, p_test)
        top1_test, cont_test = compute_observation_top1_and_continuity(
            p_test, g_test, obs_test, cand_test, gt_test, test_arcs_by_trip
        )

        row = {
            "model": name,
            "family": cfg["family"],
            "key_config": cfg["key_config"],
            "cv_macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
            "cv_macro_f1_std": round(float(np.std(fold_f1s)), 4),
            "cv_top1_mean": round(float(np.mean(fold_top1s)), 4),
            "train_macro_f1": round(m_train_sample["macro_f1"], 4),
            "test_accuracy": round(m_test["accuracy"], 4),
            "test_precision": round(m_test["precision_pos"], 4),
            "test_recall": round(m_test["recall_pos"], 4),
            "test_f1_pos": round(m_test["f1_pos"], 4),
            "test_macro_f1": round(m_test["macro_f1"], 4),
            "test_log_loss": round(m_test["log_loss"], 4),
            "test_top1_acc": round(top1_test, 4),
            "test_route_continuity": round(cont_test, 4),
            "train_time_s": round(train_time, 3),
            "inference_time_s": round(inf_time, 4),
        }
        summary_rows.append(row)
        cv_details[name] = {
            "fold_macro_f1": [round(float(x), 4) for x in fold_f1s],
            "fold_log_loss": [round(float(x), 4) for x in fold_losses],
            "fold_top1": [round(float(x), 4) for x in fold_top1s],
        }
        cms[name] = m_test["cm"]
        print(f"  CV Macro-F1={row['cv_macro_f1_mean']:.4f} (+/-{row['cv_macro_f1_std']:.4f}) | "
              f"Test Macro-F1={row['test_macro_f1']:.4f} | Test Top-1={row['test_top1_acc']:.4f} | "
              f"Cont={row['test_route_continuity']:.4f} | LogLoss={row['test_log_loss']:.4f} | "
              f"Train={row['train_time_s']}s, Inf={row['inference_time_s']}s", flush=True)

    df_res = pd.DataFrame(summary_rows)
    df_res.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    with open(os.path.join(results_dir, "cv_fold_results.json"), "w") as f:
        json.dump(cv_details, f, indent=2)
    with open(os.path.join(results_dir, "confusion_matrices.json"), "w") as f:
        json.dump(cms, f, indent=2)
    with open(os.path.join(results_dir, "mlp_history.json"), "w") as f:
        json.dump(mlp_hist, f, indent=2)

    print("\nSaved model comparison outputs to results/ directory.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap - Multi-Model Training & Evaluation")
    parser.add_argument("--data-dir", default=None, help="Base dataset directory containing train/ and student_test/")
    parser.add_argument("--artifacts-dir", default=None, help="Directory containing train_candidates.npz and test_candidates.npz")
    parser.add_argument("--results-dir", default=None, help="Directory to save experimental results")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = args.data_dir if args.data_dir else resolve_default_data_dir()
    art_dir = args.artifacts_dir if args.artifacts_dir else os.path.join(project_root, "artifacts")
    res_dir = args.results_dir if args.results_dir else os.path.join(project_root, "results")

    run_training_pipeline(data_dir, art_dir, res_dir)
