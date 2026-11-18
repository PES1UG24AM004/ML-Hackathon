"""
PathSnap - FastAPI Backend (Laptop 1)
Processes evaluation trip requests, runs MLP + HMM/Viterbi inference,
performs server-side evaluation without exposing ground-truth labels, and logs all requests.
Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_processing import list_trips, load_trip, project_trip_to_meters, align_gps_to_route
from feature_engineering import generate_candidates_for_trip, build_network_index
from hmm_viterbi import load_final_mlp, build_two_hop_set, viterbi_decode_trip, sequence_metrics, collapse_consecutive

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))


def resolve_trips_dir():
    """Locate student_test evaluation trips directory using environment variable or sensible relative paths."""
    env_dir = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "Student folder", "Student folder", "student_test"),
        os.path.join(PROJECT_ROOT, "data", "Student folder", "student_test"),
        os.path.join(PROJECT_ROOT, "data", "student_test"),
        os.path.join(PROJECT_ROOT, "..", "PathSnap_SRNs_First_Round_Kit", "data", "Student folder", "student_test"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.join(PROJECT_ROOT, "data", "Student folder", "student_test")


DEFAULT_TRIPS_DIR = resolve_trips_dir()
EVALUATOR_INPUTS_DIR = os.path.join(PROJECT_ROOT, "evaluator_data", "inputs")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "Student folder", "train")
if not os.path.isdir(TRAIN_DIR):
    fallback_train = os.path.join(PROJECT_ROOT, "..", "PathSnap_SRNs_First_Round_Kit", "data", "Student folder", "train")
    if os.path.isdir(fallback_train):
        TRAIN_DIR = fallback_train


def get_dir_for_dataset(dataset_name: str = "student_test"):
    if dataset_name in ["evaluator_test", "evaluator"]:
        return EVALUATOR_INPUTS_DIR
    elif dataset_name in ["train", "training"]:
        return TRAIN_DIR
    return DEFAULT_TRIPS_DIR
MODEL_PATH = os.environ.get("PATHSNAP_MODEL_PATH", os.path.join(PROJECT_ROOT, "artifacts", "final_mlp_model.npz"))
HMM_CFG_PATH = os.environ.get("PATHSNAP_HMM_CFG", os.path.join(PROJECT_ROOT, "artifacts", "hmm_config.json"))
LOG_PATH = os.path.join(PROJECT_ROOT, "results", "api_inference_logs.jsonl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pathsnap_api")

app = FastAPI(title="PathSnap Map-Matching API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL, SCALER_MEAN, SCALER_STD, MODEL_PARAMS = load_final_mlp(MODEL_PATH)
with open(HMM_CFG_PATH) as f:
    HMM_CFG = json.load(f)


@app.get("/")
def index():
    ui_path = os.path.join(SRC_DIR, "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "PathSnap API is running. Access /health or /trips."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "trips_dir": DEFAULT_TRIPS_DIR,
        "model_params": MODEL_PARAMS,
        "hmm_config": HMM_CFG,
    }


MAPS_DIR = os.path.join(PROJECT_ROOT, "maps")
os.makedirs(MAPS_DIR, exist_ok=True)
UI_DIR = os.path.join(SRC_DIR, "ui")
os.makedirs(UI_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/style.css")
def get_style():
    style_path = os.path.join(UI_DIR, "style.css")
    if os.path.exists(style_path):
        return FileResponse(style_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js")
def get_app_js():
    js_path = os.path.join(UI_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/api/logs")
def get_api_logs(limit: int = 50):
    """Retrieve recent inference log entries."""
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line.strip()))
                except Exception:
                    pass
    return {"count": len(logs), "logs": logs}


def extract_trip_geometry(data, max_segments: int = 15000):
    """
    Extract genuine connected road-network segments in the spatial bounding box of the trip.
    Returns network_segments ([[lat_u, lon_u], [lat_v, lon_v]]), gps_points, and geographic bounds.
    """
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]

    lon_min = float(np.min(track[:, 0]))
    lon_max = float(np.max(track[:, 0]))
    lat_min = float(np.min(track[:, 1]))
    lat_max = float(np.max(track[:, 1]))

    # Add 25% padding or at least 0.008 degrees (~800m) around track bounds
    dlon = max(lon_max - lon_min, 0.008)
    dlat = max(lat_max - lat_min, 0.008)
    pad_lon = max(dlon * 0.25, 0.008)
    pad_lat = max(dlat * 0.25, 0.008)

    box_lon_min, box_lon_max = lon_min - pad_lon, lon_max + pad_lon
    box_lat_min, box_lat_max = lat_min - pad_lat, lat_max + pad_lat

    u_idx = arcs[:, 0]
    v_idx = arcs[:, 1]
    u_lon, u_lat = nodes[u_idx, 0], nodes[u_idx, 1]
    v_lon, v_lat = nodes[v_idx, 0], nodes[v_idx, 1]

    # Include any arc where endpoints intersect the expanded trip bounding box
    in_box = (
        (np.minimum(u_lon, v_lon) <= box_lon_max) &
        (np.maximum(u_lon, v_lon) >= box_lon_min) &
        (np.minimum(u_lat, v_lat) <= box_lat_max) &
        (np.maximum(u_lat, v_lat) >= box_lat_min)
    )

    box_arcs = arcs[in_box]
    # Deduplicate bidirectional arcs (min(u,v), max(u,v)) for efficient vector display
    unique_edges = {}
    for u, v in box_arcs:
        key = (min(u, v), max(u, v))
        if key not in unique_edges:
            unique_edges[key] = (u, v)

    selected_edges = list(unique_edges.values())
    if len(selected_edges) > max_segments:
        step = max(1, len(selected_edges) // max_segments)
        selected_edges = selected_edges[::step]

    net_segments = []
    for u, v in selected_edges:
        net_segments.append([
            [round(float(nodes[u, 1]), 6), round(float(nodes[u, 0]), 6)],
            [round(float(nodes[v, 1]), 6), round(float(nodes[v, 0]), 6)],
        ])

    # Sample GPS observations (up to 400 points for smooth vector markers)
    gps_step = max(1, len(track) // 400)
    gps_points = [
        [round(float(track[i, 1]), 6), round(float(track[i, 0]), 6)]
        for i in range(0, len(track), gps_step)
    ]

    bounds = [
        [round(lat_min, 6), round(lon_min, 6)],
        [round(lat_max, 6), round(lon_max, 6)],
    ]

    return {
        "network_segments": net_segments,
        "gps_points": gps_points,
        "bounds": bounds,
    }


@app.get("/offline-maps/{trip_id}.html")
@app.get("/offline-maps/{trip_id}")
def get_offline_map(trip_id: str):
    """Generate self-contained vector SVG map without external network access."""
    clean_id = trip_id.replace(".html", "")
    trip_dir = None
    for candidate_dir in [EVALUATOR_INPUTS_DIR, DEFAULT_TRIPS_DIR, TRAIN_DIR]:
        c = os.path.join(candidate_dir, clean_id)
        if os.path.isdir(c):
            trip_dir = c
            break

    if not trip_dir:
        raise HTTPException(status_code=404, detail=f"Trip '{clean_id}' not found")

    data = load_trip(trip_dir, clean_id, load_route=False)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]

    track_xy, nodes_xy, _, _ = project_trip_to_meters(track, nodes)
    min_x, min_y = float(np.min(track_xy[:, 0])), float(np.min(track_xy[:, 1]))
    max_x, max_y = float(np.max(track_xy[:, 0])), float(np.max(track_xy[:, 1]))
    span_x = max(max_x - min_x, 10.0)
    span_y = max(max_y - min_y, 10.0)

    w, h, pad = 960, 580, 40
    scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)

    def to_svg(x, y):
        sx = pad + (x - min_x) * scale
        sy = h - (pad + (y - min_y) * scale)
        return round(float(sx), 1), round(float(sy), 1)

    svg_lines = []
    # Subtle background coordinate grid
    for gx in range(pad, w - pad, 60):
        svg_lines.append(f'<line x1="{gx}" y1="{pad}" x2="{gx}" y2="{h-pad}" stroke="#F1F5F9" stroke-width="1"/>')
    for gy in range(pad, h - pad, 60):
        svg_lines.append(f'<line x1="{pad}" y1="{gy}" x2="{w-pad}" y2="{gy}" stroke="#F1F5F9" stroke-width="1"/>')

    # Bounding box filtered road network arcs
    u_idx, v_idx = arcs[:, 0], arcs[:, 1]
    in_box = (
        (np.minimum(nodes_xy[u_idx, 0], nodes_xy[v_idx, 0]) <= max_x + 300) &
        (np.maximum(nodes_xy[u_idx, 0], nodes_xy[v_idx, 0]) >= min_x - 300) &
        (np.minimum(nodes_xy[u_idx, 1], nodes_xy[v_idx, 1]) <= max_y + 300) &
        (np.maximum(nodes_xy[u_idx, 1], nodes_xy[v_idx, 1]) >= min_y - 300)
    )
    box_arcs = arcs[in_box]
    step_arc = max(1, len(box_arcs) // 2500)
    for ai in range(0, len(box_arcs), step_arc):
        u, v = int(box_arcs[ai, 0]), int(box_arcs[ai, 1])
        ux, uy = to_svg(nodes_xy[u, 0], nodes_xy[u, 1])
        vx, vy = to_svg(nodes_xy[v, 0], nodes_xy[v, 1])
        if 0 <= ux <= w and 0 <= uy <= h and 0 <= vx <= w and 0 <= vy <= h:
            svg_lines.append(f'<line x1="{ux}" y1="{uy}" x2="{vx}" y2="{vy}" stroke="#94A3B8" stroke-width="1.2" opacity="0.65"/>')

    # GPS Track
    step_gps = max(1, len(track_xy) // 300)
    gps_pts = [to_svg(track_xy[i, 0], track_xy[i, 1]) for i in range(0, len(track_xy), step_gps)]
    poly_pts = " ".join(f"{x},{y}" for x, y in gps_pts)
    svg_lines.append(f'<polyline points="{poly_pts}" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-dasharray="4,2" opacity="0.6"/>')
    for x, y in gps_pts:
        svg_lines.append(f'<circle cx="{x}" cy="{y}" r="2.8" fill="#2563EB" stroke="#FFFFFF" stroke-width="0.8"/>')

    body = "\n".join(svg_lines)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PathSnap Offline Vector Map — Trip {clean_id}</title>
  <style>
    body {{ margin: 0; background: #F8FAFC; color: #0F172A; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }}
    .frame {{ border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; background: #FFFFFF; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }}
    .bar {{ display: flex; justify-content: space-between; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 12px; font-weight: 600; font-size: 13px; color: #0F172A; }}
    .footer {{ display: flex; gap: 20px; border-top: 1px solid #E2E8F0; padding-top: 8px; margin-top: 12px; font-size: 12px; color: #64748B; }}
    .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #2563EB; margin-right: 4px; }}
    .line {{ display: inline-block; width: 14px; height: 3px; background: #2563EB; vertical-align: middle; margin-right: 4px; }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="bar">
      <span>PATHSNAP OFFLINE VECTOR MAP // TRIP {clean_id}</span>
      <span>SCALE: 1px = {1/scale:.2f}m</span>
    </div>
    <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="border: 1px solid #E2E8F0; border-radius: 6px; background: #F8FAFC;">
      {body}
    </svg>
    <div class="footer">
      <span><span class="dot"></span>GPS Fixes ({len(track)} obs)</span>
      <span><span class="line"></span>Trajectory Path</span>
      <span><span class="line" style="background:#94A3B8;"></span>Road Arcs ({len(arcs)} edges)</span>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/api/datasets")
def get_datasets():
    """List available dataset partitions."""
    available = []
    if os.path.isdir(DEFAULT_TRIPS_DIR):
        available.append({
            "id": "student_test",
            "name": "Student Test (10 development trips)",
            "count": len(list_trips(DEFAULT_TRIPS_DIR))
        })
    if os.path.isdir(EVALUATOR_INPUTS_DIR):
        available.append({
            "id": "evaluator_test",
            "name": "Evaluator Test (10 holdout trips)",
            "count": len(list_trips(EVALUATOR_INPUTS_DIR))
        })
    if os.path.isdir(TRAIN_DIR):
        available.append({
            "id": "train",
            "name": "Training Partition (80 trips)",
            "count": len(list_trips(TRAIN_DIR))
        })
    return {"datasets": available}


@app.get("/trips")
@app.get("/api/trips")
def get_trips(dataset: str = "student_test"):
    """Enumerate trips for a specific dataset partition."""
    target_dir = get_dir_for_dataset(dataset)
    if not os.path.isdir(target_dir):
        return {"count": 0, "dataset": dataset, "trips": []}
    trips = list_trips(target_dir)
    return {"count": len(trips), "dataset": dataset, "trips": trips}


@app.get("/api/evaluator/summary")
def get_evaluator_summary():
    """Return evaluator holdout scoring summary if compiled."""
    summary_path = os.path.join(PROJECT_ROOT, "evaluator_outputs", "metrics", "evaluator_aggregate_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "not_scored_yet", "message": "Run RUN_EVALUATOR_SCORING.bat to compile evaluator scores."}


@app.get("/api/trip/{trip_id}")
def get_trip_details(trip_id: str, dataset: str = None):
    """
    Retrieve genuine trip geometry (bounding road network and GPS observations)
    for pre-inference display and inspection.
    """
    trip_dir = None
    actual_dataset = "student_test"
    if dataset:
        target = get_dir_for_dataset(dataset)
        candidate = os.path.join(target, trip_id)
        if os.path.isdir(candidate):
            trip_dir = candidate
            actual_dataset = dataset

    if not trip_dir:
        if os.path.isdir(os.path.join(EVALUATOR_INPUTS_DIR, trip_id)):
            trip_dir = os.path.join(EVALUATOR_INPUTS_DIR, trip_id)
            actual_dataset = "evaluator_test"
        elif os.path.isdir(os.path.join(DEFAULT_TRIPS_DIR, trip_id)):
            trip_dir = os.path.join(DEFAULT_TRIPS_DIR, trip_id)
            actual_dataset = "student_test"
        elif os.path.isdir(os.path.join(TRAIN_DIR, trip_id)):
            trip_dir = os.path.join(TRAIN_DIR, trip_id)
            actual_dataset = "train"
        else:
            raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in any dataset")

    # STRICT: load_route is False; only reads .track, .nodes, .arcs
    data = load_trip(trip_dir, trip_id, load_route=False)
    geom = extract_trip_geometry(data)

    return {
        "trip_id": trip_id,
        "dataset": actual_dataset,
        "n_gps": int(len(data["track"])),
        "n_road_arcs": int(len(data["arcs"])),
        "n_nodes": int(len(data["nodes"])),
        "bounds": geom["bounds"],
        "gps_points": geom["gps_points"],
        "network_segments": geom["network_segments"],
    }


@app.post("/predict/{trip_id}")
@app.post("/api/match/{trip_id}")
@app.get("/predict/{trip_id}")
@app.get("/api/match/{trip_id}")
def predict_trip(trip_id: str, dataset: str = None):
    # Locate trip directory across datasets
    trip_dir = None
    actual_dataset = "student_test"
    if dataset:
        target = get_dir_for_dataset(dataset)
        candidate = os.path.join(target, trip_id)
        if os.path.isdir(candidate):
            trip_dir = candidate
            actual_dataset = dataset

    if not trip_dir:
        if os.path.isdir(os.path.join(EVALUATOR_INPUTS_DIR, trip_id)):
            trip_dir = os.path.join(EVALUATOR_INPUTS_DIR, trip_id)
            actual_dataset = "evaluator_test"
        elif os.path.isdir(os.path.join(DEFAULT_TRIPS_DIR, trip_id)):
            trip_dir = os.path.join(DEFAULT_TRIPS_DIR, trip_id)
            actual_dataset = "student_test"
        elif os.path.isdir(os.path.join(TRAIN_DIR, trip_id)):
            trip_dir = os.path.join(TRAIN_DIR, trip_id)
            actual_dataset = "train"
        else:
            raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in any dataset")

    t0 = time.time()
    # STRICT: load_route is False; only reads .track, .nodes, .arcs
    data = load_trip(trip_dir, trip_id, load_route=False)
    track_xy, nodes_xy, _, _ = project_trip_to_meters(data["track"], data["nodes"])
    net_idx = build_network_index(nodes_xy, data["arcs"])

    X, obs_idx, cand_arcs, _ = generate_candidates_for_trip(
        track_xy, data["track"][:, 2], nodes_xy, data["arcs"], top_k=8, net_idx=net_idx
    )
    X_scaled = ((X - SCALER_MEAN) / SCALER_STD).astype(np.float32)
    probs = MODEL.predict_proba(X_scaled)[:, 1]

    # Classifier-only greedy selection
    split_pts = np.flatnonzero(np.diff(obs_idx)) + 1
    groups = np.split(np.arange(len(obs_idx)), split_pts)
    clf_pred = np.array([cand_arcs[g[int(np.argmax(probs[g]))]] for g in groups], dtype=np.int64)

    # HMM + Viterbi decoding
    out_nodes = build_two_hop_set(data["arcs"])
    hmm_pred = viterbi_decode_trip(
        probs, obs_idx, cand_arcs, X[:, 6], data["arcs"], nodes_xy,
        track_xy, data["track"][:, 2], out_nodes,
        emission_weight=HMM_CFG["emission_weight"],
        hop1_penalty=HMM_CFG["hop1_penalty"],
        hop2_penalty=HMM_CFG["hop2_penalty"],
        disc_penalty=HMM_CFG["disc_penalty"],
    )

    collapsed_seq = collapse_consecutive(hmm_pred)
    elapsed = round(time.time() - t0, 3)

    # Compute prediction diagnostics (no ground-truth needed)
    if len(collapsed_seq) > 1:
        cont_matches = sum(
            data["arcs"][collapsed_seq[i], 1] == data["arcs"][collapsed_seq[i + 1], 0]
            for i in range(len(collapsed_seq) - 1)
        )
        pred_continuity = round(float(cont_matches / (len(collapsed_seq) - 1)), 4)
    else:
        pred_continuity = 1.0

    eval_metrics = {}
    # CRITICAL: For evaluator dataset, NEVER load private .route file!
    if actual_dataset == "evaluator_test":
        per_trip_metrics_path = os.path.join(PROJECT_ROOT, "evaluator_outputs", "metrics", "evaluator_metrics_per_trip.json")
        if os.path.exists(per_trip_metrics_path):
            try:
                with open(per_trip_metrics_path, "r", encoding="utf-8") as mf:
                    all_scores = json.load(mf)
                    for sc in all_scores:
                        if sc.get("trip_id") == trip_id:
                            eval_metrics = {
                                "scored": True,
                                "hmm_top1_acc": sc.get("hmm_top1_accuracy"),
                                "clf_top1_acc": sc.get("clf_top1_accuracy"),
                                "hmm_continuity": sc.get("hmm_route_continuity"),
                                "hmm_arc_f1": sc.get("hmm_arc_f1"),
                                "hmm_jaccard": sc.get("hmm_arc_jaccard"),
                                "lcs_f1": sc.get("hmm_lcs_f1"),
                            }
                            break
            except Exception:
                pass
        if not eval_metrics:
            eval_metrics = {
                "scored": False,
                "predicted_continuity": pred_continuity,
                "status": "Private reference labels isolated. Run RUN_EVALUATOR_SCORING.bat to score."
            }
    else:
        # For student_test / train, perform evaluation if .route exists
        route_path = os.path.join(trip_dir, f"{trip_id}.route")
        if os.path.exists(route_path):
            gt_route = np.loadtxt(route_path, dtype=np.int64)
            if gt_route.ndim == 0:
                gt_route = np.array([int(gt_route)], dtype=np.int64)
            gt_obs_arcs, _, _ = align_gps_to_route(track_xy, nodes_xy, data["arcs"], gt_route)
            clf_seq = sequence_metrics(clf_pred, gt_route, data["arcs"])
            hmm_seq = sequence_metrics(hmm_pred, gt_route, data["arcs"])
            eval_metrics = {
                "scored": True,
                "clf_top1_acc": round(float(np.mean(clf_pred == gt_obs_arcs)), 4),
                "hmm_top1_acc": round(float(np.mean(hmm_pred == gt_obs_arcs)), 4),
                "clf_continuity": round(clf_seq["continuity"], 4),
                "hmm_continuity": round(hmm_seq["continuity"], 4),
                "clf_arc_f1": round(clf_seq["arc_f1"], 4),
                "hmm_arc_f1": round(hmm_seq["arc_f1"], 4),
                "hmm_jaccard": round(hmm_seq["arc_jaccard"], 4),
            }

    # Build genuine geometry payload for offline UI map rendering
    geom = extract_trip_geometry(data)
    nodes_wgs = data["nodes"]
    arcs = data["arcs"]

    pred_segments = []
    seen = set()
    for aid in collapsed_seq:
        ai = int(aid)
        if ai not in seen:
            seen.add(ai)
            u, v = int(arcs[ai, 0]), int(arcs[ai, 1])
            pred_segments.append([
                [round(float(nodes_wgs[u, 1]), 6), round(float(nodes_wgs[u, 0]), 6)],
                [round(float(nodes_wgs[v, 1]), 6), round(float(nodes_wgs[v, 0]), 6)],
            ])

    start_point = None
    end_point = None
    if len(collapsed_seq) > 0:
        first_arc = int(collapsed_seq[0])
        u0 = int(arcs[first_arc, 0])
        start_point = [round(float(nodes_wgs[u0, 1]), 6), round(float(nodes_wgs[u0, 0]), 6)]
        last_arc = int(collapsed_seq[-1])
        v1 = int(arcs[last_arc, 1])
        end_point = [round(float(nodes_wgs[v1, 1]), 6), round(float(nodes_wgs[v1, 0]), 6)]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trip_id": trip_id,
        "dataset": actual_dataset,
        "n_gps": int(len(data["track"])),
        "predicted_unique_arcs": len(set(collapsed_seq)),
        "inference_time_s": elapsed,
        "diagnostics": {"predicted_continuity": pred_continuity},
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as lf:
        lf.write(json.dumps(log_entry) + "\n")

    return {
        "trip_id": trip_id,
        "dataset": actual_dataset,
        "n_gps": int(len(data["track"])),
        "n_road_arcs": int(len(data["arcs"])),
        "n_nodes": int(len(data["nodes"])),
        "inference_time_s": elapsed,
        "predicted_arc_sequence": [int(a) for a in collapsed_seq],
        "diagnostics": {
            "predicted_continuity": pred_continuity,
            "unique_arcs": len(set(collapsed_seq)),
            "sequence_length": len(collapsed_seq)
        },
        "metrics": eval_metrics,
        "bounds": geom["bounds"],
        "gps_points": geom["gps_points"],
        "predicted_segments": pred_segments,
        "network_segments": geom["network_segments"],
        "start_point": start_point,
        "end_point": end_point,
    }
