from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray   # longitude, latitude, timestamp
    nodes: np.ndarray   # longitude, latitude
    arcs: np.ndarray    # from_node_id, to_node_id; row number is arc_id
    route: np.ndarray | None = None  # reference route; not used by predictor


def load_trip(data_root: Path, partition: str, trip_id: str, include_route: bool = False) -> Trip:
    if partition not in {"train", "student_test"}:
        raise ValueError("Only train and student_test partitions are included in this package.")
    folder = data_root / partition / trip_id
    if not folder.is_dir():
        raise FileNotFoundError(f"Trip folder not found: {folder}")
    prefix = folder / trip_id

    def load(name: str, columns: int, dtype=float):
        arr = np.loadtxt(str(prefix) + name, dtype=dtype, ndmin=2)
        if arr.ndim != 2 or arr.shape[1] != columns:
            raise ValueError(f"Unexpected shape in {prefix}{name}: {arr.shape}; expected {columns} columns")
        return arr

    track = load(".track", 3, float)
    nodes = load(".nodes", 2, float)
    arcs = load(".arcs", 2, int)
    route = None
    if include_route and (Path(str(prefix) + ".route").exists()):
        route = np.loadtxt(str(prefix) + ".route", dtype=int, ndmin=1).reshape(-1)
    return Trip(trip_id, track, nodes, arcs, route)


def _project(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Local equirectangular projection in metres, adequate for a single trip area."""
    lat0_rad = math.radians(float(lat0))
    x = (lonlat[:, 0] - lon0) * 111320.0 * math.cos(lat0_rad)
    y = (lonlat[:, 1] - lat0) * 110540.0
    return np.column_stack((x, y))


def _unproject(xy: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lat0_rad = math.radians(float(lat0))
    lon = xy[:, 0] / (111320.0 * math.cos(lat0_rad)) + lon0
    lat = xy[:, 1] / 110540.0 + lat0
    return np.column_stack((lon, lat))


def nearest_edge_match(trip: Trip, candidate_k: int = 12) -> dict[str, Any]:
    """Nearest-road-segment baseline. Does not read or use the .route reference labels."""
    if len(trip.track) == 0 or len(trip.arcs) == 0:
        raise ValueError("Trip has no GPS observations or road segments")
    if np.any(trip.arcs < 0) or np.any(trip.arcs >= len(trip.nodes)):
        raise ValueError("Road graph contains an out-of-range node reference")

    origin_lon = float(np.mean(trip.track[:, 0]))
    origin_lat = float(np.mean(trip.track[:, 1]))
    gps_xy = _project(trip.track[:, :2], origin_lon, origin_lat)
    node_xy = _project(trip.nodes, origin_lon, origin_lat)
    edge_start = node_xy[trip.arcs[:, 0]]
    edge_end = node_xy[trip.arcs[:, 1]]
    edge_vec = edge_end - edge_start
    edge_len2 = np.einsum("ij,ij->i", edge_vec, edge_vec)
    midpoints = (edge_start + edge_end) / 2.0

    k = max(1, min(int(candidate_k), len(trip.arcs)))
    tree = cKDTree(midpoints)
    _, candidate_ids = tree.query(gps_xy, k=k, workers=-1)
    if k == 1:
        candidate_ids = candidate_ids[:, None]

    n = len(gps_xy)
    predicted_arc_ids = np.empty(n, dtype=int)
    snapped_xy = np.empty((n, 2), dtype=float)
    offsets = np.empty(n, dtype=float)

    for i, point in enumerate(gps_xy):
        ids = np.asarray(candidate_ids[i], dtype=int)
        starts = edge_start[ids]
        vecs = edge_vec[ids]
        denom = np.maximum(edge_len2[ids], 1e-9)
        t = np.clip(np.einsum("ij,ij->i", point - starts, vecs) / denom, 0.0, 1.0)
        projections = starts + vecs * t[:, None]
        distances = np.linalg.norm(projections - point, axis=1)
        best = int(np.argmin(distances))
        predicted_arc_ids[i] = ids[best]
        snapped_xy[i] = projections[best]
        offsets[i] = distances[best]

    snapped_lonlat = _unproject(snapped_xy, origin_lon, origin_lat)
    if n > 1:
        connected = trip.arcs[predicted_arc_ids[:-1], 1] == trip.arcs[predicted_arc_ids[1:], 0]
        continuity = float(np.mean(connected))
    else:
        continuity = 1.0

    return {
        "trip_id": trip.trip_id,
        "predicted_arc_ids": predicted_arc_ids,
        "snapped_lonlat": snapped_lonlat,
        "offsets_m": offsets,
        "candidate_ids": np.asarray(candidate_ids),
        "gps_xy": gps_xy,
        "node_xy": node_xy,
        "edge_start": edge_start,
        "edge_end": edge_end,
        "midpoints": midpoints,
        "metrics": {
            "gps_points": int(n),
            "road_segments": int(len(trip.arcs)),
            "mean_offset_m": float(np.mean(offsets)),
            "median_offset_m": float(np.median(offsets)),
            "p95_offset_m": float(np.percentile(offsets, 95)),
            "max_offset_m": float(np.max(offsets)),
            "directed_edge_continuity": continuity,
        },
    }
