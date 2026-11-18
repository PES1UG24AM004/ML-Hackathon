"""
PathSnap - Bayesian Hyperparameter Optimisation (Gaussian Process + Expected Improvement)
Optimises the selected Shallow MLP Classifier using Group-Aware CV on the 80 training trips.

Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import cdist

from data_processing import load_trip, resolve_default_data_dir
from models import StandardScalerNP, MLPClassifierNP
from train_evaluate import compute_binary_metrics, compute_observation_top1_and_continuity


def gp_predict(X_obs: np.ndarray, y_obs: np.ndarray, X_cand: np.ndarray, length_scale=0.35, noise=1e-4):
    """Gaussian Process posterior mean & std with RBF kernel in normalized [0,1]^d space."""
    K = np.exp(-0.5 * cdist(X_obs, X_obs, "sqeuclidean") / (length_scale ** 2))
    K += noise * np.eye(len(X_obs))
    K_s = np.exp(-0.5 * cdist(X_obs, X_cand, "sqeuclidean") / (length_scale ** 2))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs - y_obs.mean()))
    mu = y_obs.mean() + K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    var = np.maximum(1.0 - np.sum(v ** 2, axis=0), 1e-9)
    return mu, np.sqrt(var)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.005):
    """Expected Improvement acquisition function for GP exploration vs exploitation."""
    imp = mu - best_y - xi
    Z = imp / np.maximum(sigma, 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def decode_params(u: np.ndarray):
    """Map unit hypercube u in [0,1]^4 to MLP hyperparameters."""
    hidden_dim = int(round(32 + u[0] * (128 - 32)))
    lr = float(10.0 ** (-3.7 + u[1] * (-2.3 - (-3.7))))
    weight_decay = float(10.0 ** (-5.5 + u[2] * (-2.5 - (-5.5))))
    epochs = int(round(8 + u[3] * (15 - 8)))
    return {
        "hidden_dim": hidden_dim,
        "lr": round(lr, 6),
        "weight_decay": round(weight_decay, 7),
        "epochs": epochs,
    }


def evaluate_cv(params: dict, X_train, y_train, g_train, folds):
    """Evaluate candidate hyperparameter set across CV folds."""
    fold_f1s = []
    fold_losses = []
    for val_groups in folds:
        val_mask = np.isin(g_train, val_groups)
        tr_mask = ~val_mask
        tr_idx = np.flatnonzero(tr_mask)
        if len(tr_idx) > 250000:
            tr_idx = np.random.default_rng(42).choice(tr_idx, size=100000, replace=False)
        scaler = StandardScalerNP()
        X_tr = scaler.fit_transform(X_train[tr_idx])
        va_idx = np.flatnonzero(val_mask)
        if len(va_idx) > 100000:
            va_idx = np.random.default_rng(42).choice(va_idx, size=100000, replace=False)
        X_va = scaler.transform(X_train[va_idx])

        model = MLPClassifierNP(
            hidden_dim=params["hidden_dim"],
            lr=params["lr"],
            epochs=params["epochs"],
            batch_size=16384,
            weight_decay=params["weight_decay"],
            random_state=42,
        )
        model.fit(X_tr, y_train[tr_idx])
        p_va = model.predict_proba(X_va)[:, 1]
        m = compute_binary_metrics(y_train[va_idx], p_va)
        fold_f1s.append(m["macro_f1"])
        fold_losses.append(m["log_loss"])
    return float(np.mean(fold_f1s)), float(np.std(fold_f1s)), float(np.mean(fold_losses))


def run_bayesian_optimization(data_dir: str, artifacts_dir: str, results_dir: str):
    """Execute GP-EI Bayesian hyperparameter search on the MLP classifier."""
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

    X_train, y_train, g_train = train_npz["X"], train_npz["y"], train_npz["trip_group"]
    X_test, y_test, g_test = test_npz["X"], test_npz["y"], test_npz["trip_group"]
    obs_test, cand_test, gt_test = test_npz["obs_idx"], test_npz["cand_arcs"], test_npz["gt_arcs"]

    test_arcs = {}
    for i, tid in enumerate(test_npz["trip_ids"]):
        trip_path = os.path.join(data_dir, "student_test", str(tid))
        if os.path.isdir(trip_path):
            test_arcs[i] = load_trip(trip_path, str(tid), False)["arcs"]

    rng = np.random.default_rng(42)
    unique_groups = np.unique(g_train)
    folds = np.array_split(rng.permutation(unique_groups), 3)

    n_init = 4
    n_bo = 6
    total_trials = n_init + n_bo

    U_obs = []
    y_obs = []
    trials = []
    t_bo_start = time.time()

    for t in range(total_trials):
        if t < n_init:
            u = rng.uniform(0.0, 1.0, size=4)
            phase = "random_init"
        else:
            U_cand = rng.uniform(0.0, 1.0, size=(2000, 4))
            mu, sigma = gp_predict(np.array(U_obs), np.array(y_obs), U_cand)
            ei = expected_improvement(mu, sigma, float(np.max(y_obs)))
            u = U_cand[int(np.argmax(ei))]
            phase = "gp_ei"

        params = decode_params(u)
        t0 = time.time()
        cv_f1, cv_std, cv_loss = evaluate_cv(params, X_train, y_train, g_train, folds)
        dt = time.time() - t0

        U_obs.append(u)
        y_obs.append(cv_f1)
        trial_rec = {
            "trial": t + 1,
            "phase": phase,
            **params,
            "cv_macro_f1": round(cv_f1, 4),
            "cv_macro_f1_std": round(cv_std, 4),
            "cv_log_loss": round(cv_loss, 4),
            "trial_time_s": round(dt, 2),
        }
        trials.append(trial_rec)
        print(f"Trial {t+1:02d}/{total_trials} [{phase}] {params} -> CV Macro-F1={cv_f1:.4f} (+/-{cv_std:.4f}), {dt:.1f}s", flush=True)

    bo_total_time = time.time() - t_bo_start
    df_trials = pd.DataFrame(trials)
    trials_csv = os.path.join(results_dir, "bayesian_opt_trials.csv")
    df_trials.to_csv(trials_csv, index=False)
    print(f"Saved: {trials_csv}")

    best_row = df_trials.loc[df_trials["cv_macro_f1"].idxmax()].to_dict()
    best_params = {
        "hidden_dim": int(best_row["hidden_dim"]),
        "lr": float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
        "epochs": int(best_row["epochs"]),
    }
    print(f"\nBest Bayesian Opt Config: {best_params} (CV Macro-F1={best_row['cv_macro_f1']})", flush=True)

    scaler_final = StandardScalerNP()
    X_tr_scaled = scaler_final.fit_transform(X_train)
    X_te_scaled = scaler_final.transform(X_test)

    final_model = MLPClassifierNP(
        hidden_dim=best_params["hidden_dim"],
        lr=best_params["lr"],
        epochs=best_params["epochs"],
        batch_size=16384,
        weight_decay=best_params["weight_decay"],
        random_state=42,
    )
    t0 = time.time()
    # Internal validation from training folds to monitor convergence without leaking student-test labels
    val_group_internal = folds[0]
    val_mask_internal = np.isin(g_train, val_group_internal)
    final_model.fit(
        X_tr_scaled[~val_mask_internal],
        y_train[~val_mask_internal],
        X_val=X_tr_scaled[val_mask_internal],
        y_val=y_train[val_mask_internal],
    )
    train_time = time.time() - t0

    t1 = time.time()
    p_test = final_model.predict_proba(X_te_scaled)[:, 1]
    inf_time = time.time() - t1

    m_test = compute_binary_metrics(y_test, p_test)
    top1_test, cont_test = compute_observation_top1_and_continuity(
        p_test, g_test, obs_test, cand_test, gt_test, test_arcs
    )

    final_model_path = os.path.join(artifacts_dir, "final_mlp_model.npz")
    np.savez(
        final_model_path,
        W1=final_model.W1,
        b1=final_model.b1,
        W2=final_model.W2,
        b2=final_model.b2,
        scaler_mean=scaler_final.mean_,
        scaler_std=scaler_final.std_,
        best_params_json=json.dumps(best_params),
    )
    print(f"Saved: {final_model_path}")

    df_comp_path = os.path.join(results_dir, "model_comparison.csv")
    untuned_metrics = {}
    if os.path.exists(df_comp_path):
        df_comp = pd.read_csv(df_comp_path)
        mlp_rows = df_comp[df_comp["model"] == "Shallow MLP (Neural Net)"]
        if len(mlp_rows) > 0:
            untuned = mlp_rows.iloc[0].to_dict()
            untuned_metrics = {
                "cv_macro_f1": untuned["cv_macro_f1_mean"],
                "test_top1_acc": untuned["test_top1_acc"],
                "test_macro_f1": untuned["test_macro_f1"],
                "test_log_loss": untuned["test_log_loss"],
                "test_route_continuity": untuned["test_route_continuity"],
                "inference_time_s": untuned["inference_time_s"],
            }

    comparison = {
        "bo_total_time_s": round(bo_total_time, 2),
        "best_params": best_params,
        "untuned": untuned_metrics,
        "tuned": {
            "cv_macro_f1": round(float(best_row["cv_macro_f1"]), 4),
            "test_top1_acc": round(top1_test, 4),
            "test_macro_f1": round(m_test["macro_f1"], 4),
            "test_log_loss": round(m_test["log_loss"], 4),
            "test_route_continuity": round(cont_test, 4),
            "train_time_s": round(train_time, 3),
            "inference_time_s": round(inf_time, 4),
        },
        "tuned_history": final_model.history_,
    }
    bo_summary_path = os.path.join(results_dir, "bayesian_opt_summary.json")
    with open(bo_summary_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Saved: {bo_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap - Bayesian Hyperparameter Optimization")
    parser.add_argument("--data-dir", default=None, help="Base dataset directory containing train/ and student_test/")
    parser.add_argument("--artifacts-dir", default=None, help="Directory containing candidates and output final_mlp_model.npz")
    parser.add_argument("--results-dir", default=None, help="Directory to save BO trial results")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = args.data_dir if args.data_dir else resolve_default_data_dir()
    art_dir = args.artifacts_dir if args.artifacts_dir else os.path.join(project_root, "artifacts")
    res_dir = args.results_dir if args.results_dir else os.path.join(project_root, "results")

    run_bayesian_optimization(data_dir, art_dir, res_dir)
