"""
PathSnap - EDA, Model Comparison, BO, HMM & Folium-Compatible Geographical Visualisations
Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from data_processing import load_trip, project_trip_to_meters, align_gps_to_route
from feature_engineering import generate_candidates_for_trip, build_network_index
from hmm_viterbi import load_final_mlp, build_two_hop_set, viterbi_decode_trip

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})


def plot_eda(res_dir: str, fig_dir: str):
    df = pd.read_csv(os.path.join(res_dir, "eda_trip_summary.csv"))
    tr = df[df["partition"] == "train"]
    te = df[df["partition"] == "student_test"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # 1. GPS Observations per trip
    ax = axes[0, 0]
    ax.hist(tr["n_gps"], bins=20, alpha=0.75, color="#1f77b4", label="Train (80 trips)")
    ax.hist(te["n_gps"], bins=10, alpha=0.80, color="#ff7f0e", label="Student Test (10 trips)")
    ax.set_title("Distribution of GPS Observations per Trip")
    ax.set_xlabel("Number of GPS Points")
    ax.set_ylabel("Trip Count")
    ax.legend()

    # 2. Mean GPS-to-Route Noise
    ax = axes[0, 1]
    ax.hist(tr["mean_gps_to_route_m"], bins=20, alpha=0.75, color="#2ca02c", label="Mean Noise (Train)")
    ax.hist(te["mean_gps_to_route_m"], bins=8, alpha=0.85, color="#d62728", label="Mean Noise (Test)")
    ax.set_title("Mean GPS-to-Reference-Route Distance (m)")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Trip Count")
    ax.legend()

    # 3. Road Network Size (Nodes vs Arcs)
    ax = axes[1, 0]
    ax.scatter(tr["n_nodes"], tr["n_arcs"], alpha=0.75, s=35, color="#1f77b4", label="Train")
    ax.scatter(te["n_nodes"], te["n_arcs"], alpha=0.90, s=55, marker="^", color="#ff7f0e", label="Student Test")
    ax.set_title("Road Network Size: Nodes vs. Directed Arcs")
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Number of Directed Arcs")
    ax.legend()

    # 4. Global Geographic Span of Trips
    ax = axes[1, 1]
    tr_lon = 0.5 * (tr["lon_min"] + tr["lon_max"])
    tr_lat = 0.5 * (tr["lat_min"] + tr["lat_max"])
    te_lon = 0.5 * (te["lon_min"] + te["lon_max"])
    te_lat = 0.5 * (te["lat_min"] + te["lat_max"])
    ax.scatter(tr_lon, tr_lat, alpha=0.8, s=40, color="#1f77b4", label="Train Centroids")
    ax.scatter(te_lon, te_lat, alpha=0.95, s=65, marker="^", color="#d62728", label="Student Test Centroids")
    ax.set_title("Global Geographic Distribution of Trip Centroids")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim(-120, 150)
    ax.set_ylim(-45, 70)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_overview.png"), bbox_inches="tight")
    plt.close()


def plot_model_comparison(res_dir: str, fig_dir: str):
    df = pd.read_csv(os.path.join(res_dir, "model_comparison.csv"))
    models = df["model"].tolist()
    short_names = [
        "LogReg", "GaussNB", "Dist-kNN", "RandForest", "GradBoost", "Shallow MLP"
    ]
    x = np.arange(len(models))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#1b9e77"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # 1. CV Macro F1 with std error bars
    ax = axes[0, 0]
    bars = ax.bar(x, df["cv_macro_f1_mean"], yerr=df["cv_macro_f1_std"], capsize=4, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("5-Fold Group-Aware CV Macro F1-Score")
    ax.set_ylabel("Macro F1-Score")
    for b, v in zip(bars, df["cv_macro_f1_mean"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # 2. Student-Test Top-1 Accuracy & Macro F1
    ax = axes[0, 1]
    w = 0.36
    b1 = ax.bar(x - w / 2, df["test_macro_f1"], width=w, label="Macro F1", color="#2b5c8f")
    b2 = ax.bar(x + w / 2, df["test_top1_acc"], width=w, label="Obs Top-1 Acc", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Student-Test Macro F1 & Observation Top-1 Accuracy")
    ax.set_ylabel("Score")
    ax.legend()
    for b, v in zip(b2, df["test_top1_acc"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)

    # 3. Student-Test Log Loss (lower is better)
    ax = axes[1, 0]
    bars = ax.bar(x, df["test_log_loss"], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15)
    ax.set_title("Student-Test Log Loss (Lower is Better)")
    ax.set_ylabel("Binary Cross-Entropy Log Loss")
    for b, v in zip(bars, df["test_log_loss"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}", ha="center", fontsize=8.5)

    # 4. Inference Time (log scale)
    ax = axes[1, 1]
    bars = ax.bar(x, df["inference_time_s"], color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15)
    ax.set_title("Student-Test Inference Time (153,280 Candidates)")
    ax.set_ylabel("Time (seconds, log scale)")
    for b, v in zip(bars, df["inference_time_s"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.3f}s", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison.png"), bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(res_dir: str, fig_dir: str):
    with open(os.path.join(res_dir, "confusion_matrices.json")) as f:
        cms = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    for idx, (name, cm) in enumerate(cms.items()):
        ax = axes[idx]
        arr = np.array(cm, dtype=np.float64)
        norm_arr = arr / arr.sum(axis=1, keepdims=True)
        im = ax.imshow(norm_arr, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                val = int(arr[i, j])
                pct = 100.0 * norm_arr[i, j]
                color = "white" if norm_arr[i, j] > 0.55 else "black"
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=8.5)

    plt.suptitle("Student-Test Confusion Matrices Across All Six Models (153,280 Candidate Pairs)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"), bbox_inches="tight")
    plt.close()


def plot_mlp_and_bo(res_dir: str, fig_dir: str):
    with open(os.path.join(res_dir, "mlp_history.json")) as f:
        untuned_hist = json.load(f)
    with open(os.path.join(res_dir, "bayesian_opt_summary.json")) as f:
        bo_sum = json.load(f)
    df_bo = pd.read_csv(os.path.join(res_dir, "bayesian_opt_trials.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # 1. Epoch-wise Loss Curves (Untuned & Tuned MLP)
    ax = axes[0]
    ep_u = np.arange(1, len(untuned_hist["train_loss"]) + 1)
    ax.plot(ep_u, untuned_hist["train_loss"], "o--", color="#1f77b4", label="Untuned MLP Train Loss")
    ax.plot(ep_u, untuned_hist["val_loss"], "o-", color="#1f77b4", linewidth=2, label="Untuned MLP Val Loss")
    ep_t = np.arange(1, len(bo_sum["tuned_history"]["train_loss"]) + 1)
    ax.plot(ep_t, bo_sum["tuned_history"]["train_loss"], "s--", color="#d62728", alpha=0.8, label="BO-Tuned MLP Train Loss")
    ax.plot(ep_t, bo_sum["tuned_history"]["val_loss"], "s-", color="#d62728", linewidth=2, label="BO-Tuned MLP Val Loss")
    ax.set_title("MLP Epoch-Wise Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend(fontsize=8.5)

    # 2. Bayesian Optimization Convergence
    ax = axes[1]
    trials = df_bo["trial"].values
    scores = df_bo["cv_macro_f1"].values
    best_so_far = np.maximum.accumulate(scores)
    init_mask = df_bo["phase"] == "random_init"
    gp_mask = df_bo["phase"] == "gp_ei"

    ax.scatter(trials[init_mask], scores[init_mask], s=65, color="#ff7f0e", label="Random Init Trials", zorder=3)
    ax.scatter(trials[gp_mask], scores[gp_mask], s=65, color="#2ca02c", marker="^", label="GP-EI Acquisition Trials", zorder=3)
    ax.plot(trials, best_so_far, "-", color="#1f77b4", linewidth=2.2, label="Best CV Macro-F1 So Far")
    ax.set_title("Bayesian Hyperparameter Optimisation Convergence")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("3-Fold Group-CV Macro F1-Score")
    ax.set_xticks(trials)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mlp_and_bo.png"), bbox_inches="tight")
    plt.close()


def plot_hmm_comparison(res_dir: str, fig_dir: str):
    df = pd.read_csv(os.path.join(res_dir, "hmm_vs_classifier_trips.csv"))
    x = np.arange(len(df))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    metrics = [
        ("clf_top1_acc", "hmm_top1_acc", "Observation Top-1 Accuracy"),
        ("clf_continuity", "hmm_continuity", "Route Continuity"),
        ("clf_arc_f1", "hmm_arc_f1", "Sequence Route Arc F1"),
    ]
    for ax, (c_col, h_col, title) in zip(axes, metrics):
        ax.bar(x - w / 2, df[c_col], width=w, color="#7f7f7f", label="Classifier Only")
        ax.bar(x + w / 2, df[h_col], width=w, color="#1b9e77", label="Classifier + HMM/Viterbi")
        ax.set_xticks(x)
        ax.set_xticklabels(df["trip_id"].astype(str), rotation=45, fontsize=8)
        ax.set_ylim(0.55, 1.03)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)

    plt.suptitle("Classifier-Only vs. Classifier + HMM/Viterbi Across 10 Student-Test Trips", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hmm_comparison.png"), bbox_inches="tight")
    plt.close()


def generate_folium_leaflet_html(
    trip_id: str,
    track_wgs: np.ndarray,
    nodes_wgs: np.ndarray,
    arcs: np.ndarray,
    gt_route: np.ndarray,
    pred_arcs: np.ndarray,
    local_arc_ids: np.ndarray,
    out_html_path: str,
    metrics: dict,
):
    """
    Generate an interactive Folium-compatible Leaflet.js HTML map with layer controls,
    legend, distinguishable colors, GPS observations, road network, reference route, and predicted route.
    """
    center_lat = float(np.mean(track_wgs[:, 1]))
    center_lon = float(np.mean(track_wgs[:, 0]))

    # Subsample GPS points if very dense for smooth browser rendering
    step = max(1, len(track_wgs) // 400)
    gps_pts = [[round(float(track_wgs[i, 1]), 6), round(float(track_wgs[i, 0]), 6)] for i in range(0, len(track_wgs), step)]

    def arcs_to_polylines(arc_id_list):
        lines = []
        seen = set()
        for aid in arc_id_list:
            ai = int(aid)
            if ai in seen:
                continue
            seen.add(ai)
            u, v = int(arcs[ai, 0]), int(arcs[ai, 1])
            lines.append([
                [round(float(nodes_wgs[u, 1]), 6), round(float(nodes_wgs[u, 0]), 6)],
                [round(float(nodes_wgs[v, 1]), 6), round(float(nodes_wgs[v, 0]), 6)],
            ])
        return lines

    net_lines = arcs_to_polylines(local_arc_ids[:1500])
    gt_lines = arcs_to_polylines(gt_route)
    pred_lines = arcs_to_polylines(pred_arcs)

    if HAS_FOLIUM:
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

        net_fg = folium.FeatureGroup(name="Local Road Network", show=True)
        for seg in net_lines:
            folium.PolyLine(seg, color="#999999", weight=1.5, opacity=0.45).add_to(net_fg)
        net_fg.add_to(fmap)

        gt_fg = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        for seg in gt_lines:
            folium.PolyLine(seg, color="#2ca02c", weight=5, opacity=0.85).add_to(gt_fg)
        gt_fg.add_to(fmap)

        pred_fg = folium.FeatureGroup(name="Predicted Route (MLP + HMM)", show=True)
        for seg in pred_lines:
            folium.PolyLine(seg, color="#d62728", weight=3, dash_array="6, 4", opacity=0.95).add_to(pred_fg)
        pred_fg.add_to(fmap)

        gps_fg = folium.FeatureGroup(name="Raw GPS Observations", show=True)
        for pt in gps_pts:
            folium.CircleMarker(pt, radius=3.5, color="#1f77b4", fill=True, fill_color="#1f77b4", fill_opacity=0.85).add_to(gps_fg)
        gps_fg.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.save(out_html_path)
        return

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PathSnap Map - Trip {trip_id}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; font-family: Arial, sans-serif; }}
    .legend {{
      background: white; padding: 10px 14px; border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.25); line-height: 1.6; font-size: 13px;
    }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
    .line {{ display:inline-block; width:18px; height:3px; vertical-align:middle; margin-right:6px; }}
  </style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map').setView([{center_lat:.6f}, {center_lon:.6f}], 14);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  var netLayer = L.layerGroup();
  var netData = {json.dumps(net_lines)};
  netData.forEach(function(seg) {{
    L.polyline(seg, {{color: '#999999', weight: 1.5, opacity: 0.45}}).addTo(netLayer);
  }});
  netLayer.addTo(map);

  var gtLayer = L.layerGroup();
  var gtData = {json.dumps(gt_lines)};
  gtData.forEach(function(seg) {{
    L.polyline(seg, {{color: '#2ca02c', weight: 5, opacity: 0.85}}).addTo(gtLayer);
  }});
  gtLayer.addTo(map);

  var predLayer = L.layerGroup();
  var predData = {json.dumps(pred_lines)};
  predData.forEach(function(seg) {{
    L.polyline(seg, {{color: '#d62728', weight: 3, dashArray: '6, 4', opacity: 0.95}}).addTo(predLayer);
  }});
  predLayer.addTo(map);

  var gpsLayer = L.layerGroup();
  var gpsData = {json.dumps(gps_pts)};
  gpsData.forEach(function(pt) {{
    L.circleMarker(pt, {{radius: 3.5, color: '#1f77b4', fillColor: '#1f77b4', fillOpacity: 0.85}}).addTo(gpsLayer);
  }});
  gpsLayer.addTo(map);

  L.control.layers(null, {{
    "Local Road Network": netLayer,
    "Reference Route (Ground Truth)": gtLayer,
    "Predicted Route (MLP + HMM/Viterbi)": predLayer,
    "Raw GPS Observations": gpsLayer
  }}, {{collapsed: false}}).addTo(map);

  var legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function(map) {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML = '<b>Trip {trip_id}</b><br>' +
      'Top-1 Acc: {(metrics["hmm_top1_acc"]*100):.1f}% | Continuity: {(metrics["hmm_continuity"]*100):.1f}%<br>' +
      '<span class="dot" style="background:#1f77b4;"></span>Raw GPS Observations<br>' +
      '<span class="line" style="background:#999999;"></span>Road Network<br>' +
      '<span class="line" style="background:#2ca02c;height:4px;"></span>Reference Route<br>' +
      '<span class="line" style="background:#d62728;"></span>Predicted Route (HMM)';
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>"""
    with open(out_html_path, "w") as f:
        f.write(html)


def plot_geographical_trip(
    base_data: str,
    trip_id: str,
    model,
    mean,
    std,
    fig_dir: str,
    map_dir: str,
    trip_metrics: dict,
):
    data = load_trip(os.path.join(base_data, "student_test", trip_id), trip_id, load_route=True)
    track_xy, nodes_xy, _, _ = project_trip_to_meters(data["track"], data["nodes"])
    net_idx = build_network_index(nodes_xy, data["arcs"])
    X, obs_idx, cand_arcs, _ = generate_candidates_for_trip(
        track_xy, data["track"][:, 2], nodes_xy, data["arcs"], top_k=8, net_idx=net_idx
    )
    X_scaled = ((X - mean) / std).astype(np.float32)
    probs = model.predict_proba(X_scaled)[:, 1]
    out_nodes = build_two_hop_set(data["arcs"])
    hmm_pred = viterbi_decode_trip(
        probs, obs_idx, cand_arcs, X[:, 6], data["arcs"], nodes_xy,
        track_xy, data["track"][:, 2], out_nodes
    )

    # Local bounding box in meters around the trajectory
    xmin, ymin = track_xy.min(axis=0) - 180.0
    xmax, ymax = track_xy.max(axis=0) + 180.0
    ax_xy = nodes_xy[data["arcs"][:, 0]]
    bx_xy = nodes_xy[data["arcs"][:, 1]]
    in_box = (
        (np.minimum(ax_xy[:, 0], bx_xy[:, 0]) <= xmax) &
        (np.maximum(ax_xy[:, 0], bx_xy[:, 0]) >= xmin) &
        (np.minimum(ax_xy[:, 1], bx_xy[:, 1]) <= ymax) &
        (np.maximum(ax_xy[:, 1], bx_xy[:, 1]) >= ymin)
    )
    local_arc_ids = np.flatnonzero(in_box)

    # Save interactive Folium-compatible HTML map
    generate_folium_leaflet_html(
        trip_id, data["track"], data["nodes"], data["arcs"],
        data["route"], hmm_pred, local_arc_ids,
        os.path.join(map_dir, f"trip_{trip_id}_map.html"),
        trip_metrics,
    )

    # Save high-res static PNG for PDF Technical Report
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for aid in local_arc_ids[:2500]:
        u, v = data["arcs"][aid]
        ax.plot([nodes_xy[u, 0], nodes_xy[v, 0]], [nodes_xy[u, 1], nodes_xy[v, 1]],
                color="#cccccc", linewidth=0.7, zorder=1)

    for aid in np.unique(data["route"]):
        u, v = data["arcs"][aid]
        ax.plot([nodes_xy[u, 0], nodes_xy[v, 0]], [nodes_xy[u, 1], nodes_xy[v, 1]],
                color="#2ca02c", linewidth=3.5, alpha=0.85, zorder=2)

    for aid in np.unique(hmm_pred):
        u, v = data["arcs"][aid]
        ax.plot([nodes_xy[u, 0], nodes_xy[v, 0]], [nodes_xy[u, 1], nodes_xy[v, 1]],
                color="#d62728", linewidth=1.8, linestyle="--", alpha=0.95, zorder=3)

    step = max(1, len(track_xy) // 350)
    ax.scatter(track_xy[::step, 0], track_xy[::step, 1], s=10, color="#1f77b4", alpha=0.85, zorder=4)

    ax.set_xlim(xmin + 80, xmax - 80)
    ax.set_ylim(ymin + 80, ymax - 80)
    ax.set_aspect("equal")
    ax.set_title(
        f"Geographical Route Reconstruction — Student Test Trip {trip_id}\n"
        f"(Top-1 Acc: {trip_metrics['hmm_top1_acc']*100:.1f}%, Continuity: {trip_metrics['hmm_continuity']*100:.1f}%, "
        f"Arc F1: {trip_metrics['hmm_arc_f1']*100:.1f}%)"
    )
    ax.set_xlabel("Local Easting (meters)")
    ax.set_ylabel("Local Northing (meters)")

    legend_elems = [
        Line2D([0], [0], color="#cccccc", lw=1.5, label="Local Road Network"),
        Line2D([0], [0], color="#2ca02c", lw=3.5, label="Reference Route (Ground Truth)"),
        Line2D([0], [0], color="#d62728", lw=2.0, ls="--", label="Predicted Route (MLP + HMM)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=6, label="Raw GPS Observations"),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"geo_map_trip_{trip_id}.png"), bbox_inches="tight")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PathSnap - Visualizations and Figures Generator")
    parser.add_argument("--data-dir", default=None, help="Base dataset directory containing student_test/")
    parser.add_argument("--results-dir", default=None, help="Directory containing experimental CSV/JSON results")
    parser.add_argument("--artifacts-dir", default=None, help="Directory containing final_mlp_model.npz")
    parser.add_argument("--figures-dir", default=None, help="Directory to save output figures")
    parser.add_argument("--maps-dir", default=None, help="Directory to save interactive HTML maps")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    from data_processing import resolve_default_data_dir
    base_data = args.data_dir if args.data_dir else resolve_default_data_dir()
    res_dir = args.results_dir if args.results_dir else os.path.join(project_root, "results")
    art_dir = args.artifacts_dir if args.artifacts_dir else os.path.join(project_root, "artifacts")
    fig_dir = args.figures_dir if args.figures_dir else os.path.join(project_root, "figures")
    map_dir = args.maps_dir if args.maps_dir else os.path.join(project_root, "maps")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)

    print("Generating PathSnap figures and visualizations...")
    if os.path.exists(os.path.join(res_dir, "eda_trip_summary.csv")):
        plot_eda(res_dir, fig_dir)
        print("  - Generated eda_overview.png")

    if os.path.exists(os.path.join(res_dir, "model_comparison.csv")):
        plot_model_comparison(res_dir, fig_dir)
        print("  - Generated model_comparison.png")

    if os.path.exists(os.path.join(res_dir, "confusion_matrices.json")):
        plot_confusion_matrices(res_dir, fig_dir)
        print("  - Generated confusion_matrices.png")

    if os.path.exists(os.path.join(res_dir, "mlp_history.json")) and os.path.exists(os.path.join(res_dir, "bayesian_opt_trials.csv")):
        plot_mlp_and_bo(res_dir, fig_dir)
        print("  - Generated mlp_and_bo.png")

    if os.path.exists(os.path.join(res_dir, "hmm_vs_classifier_trips.csv")):
        plot_hmm_comparison(res_dir, fig_dir)
        print("  - Generated hmm_comparison.png")

    # Generate geographical route maps if trip data and model artifact exist
    model_path = os.path.join(art_dir, "final_mlp_model.npz")
    hmm_csv = os.path.join(res_dir, "hmm_vs_classifier_trips.csv")
    if os.path.exists(model_path) and os.path.exists(hmm_csv):
        model, mean, std, _ = load_final_mlp(model_path)
        df_hmm = pd.read_csv(hmm_csv, dtype={"trip_id": str})
        for tid in ["00000056", "00000007"]:
            trip_folder = os.path.join(base_data, "student_test", tid)
            if os.path.isdir(trip_folder):
                matching = df_hmm[df_hmm["trip_id"] == tid]
                if len(matching) > 0:
                    m = matching.iloc[0].to_dict()
                    plot_geographical_trip(base_data, tid, model, mean, std, fig_dir, map_dir, m)
                    print(f"  - Generated geo map and interactive Leaflet map for trip {tid}")

    print("All visualizations complete.")


if __name__ == "__main__":
    main()
