from __future__ import annotations

from pathlib import Path
import numpy as np
import folium
from folium import FeatureGroup, LayerControl

from .core import Trip


def make_map(trip: Trip, result: dict, out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw = trip.track[:, :2]
    snapped = result["snapped_lonlat"]
    center = [float(np.mean(raw[:, 1])), float(np.mean(raw[:, 0]))]
    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)

    raw_layer = FeatureGroup(name="Raw GPS observations", show=True)
    folium.PolyLine([[float(lat), float(lon)] for lon, lat in raw],
                    color="#2563eb", weight=3, opacity=0.65,
                    tooltip="Raw GPS trajectory").add_to(raw_layer)
    stride = max(1, len(raw) // 100)
    for lon, lat in raw[::stride]:
        folium.CircleMarker([float(lat), float(lon)], radius=2.5,
                            color="#2563eb", fill=True, fill_opacity=0.8,
                            tooltip="GPS observation").add_to(raw_layer)
    raw_layer.add_to(m)

    # Show a small local subset of road graph edges around the observed trajectory.
    road_layer = FeatureGroup(name="Nearby road-network segments", show=True)
    edge_start = result["edge_start"]
    edge_end = result["edge_end"]
    distances, near_ids = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(result["midpoints"]).query(
        result["gps_xy"][::max(1, len(raw)//350)], k=3, workers=-1
    )
    keep = set()
    for drow, ids in zip(np.atleast_2d(distances), np.atleast_2d(near_ids)):
        for dist, edge_id in zip(drow, ids):
            if float(dist) <= 180.0:
                keep.add(int(edge_id))
            if len(keep) >= 1200:
                break
        if len(keep) >= 1200:
            break
    for edge_id in keep:
        a, b = edge_start[edge_id], edge_end[edge_id]
        # Convert projected node coordinates back to lon/lat using the same trip origin.
        lon0, lat0 = float(np.mean(raw[:, 0])), float(np.mean(raw[:, 1]))
        lat_rad = np.deg2rad(lat0)
        coords = [
            [float(a[1] / 110540.0 + lat0), float(a[0] / (111320.0 * np.cos(lat_rad)) + lon0)],
            [float(b[1] / 110540.0 + lat0), float(b[0] / (111320.0 * np.cos(lat_rad)) + lon0)],
        ]
        folium.PolyLine(coords, color="#9ca3af", weight=1.4, opacity=0.55).add_to(road_layer)
    road_layer.add_to(m)

    match_layer = FeatureGroup(name="Nearest-road baseline", show=True)
    folium.PolyLine([[float(lat), float(lon)] for lon, lat in snapped],
                    color="#dc2626", weight=4, opacity=0.9,
                    tooltip="Nearest-road-segment prediction").add_to(match_layer)
    for idx in [0, len(snapped)-1]:
        lon, lat = snapped[idx]
        folium.CircleMarker([float(lat), float(lon)], radius=5, color="#dc2626",
                            fill=True, fill_opacity=1,
                            tooltip="Predicted route start" if idx == 0 else "Predicted route end").add_to(match_layer)
    match_layer.add_to(m)

    LayerControl(collapsed=False).add_to(m)
    bounds = [[float(np.min(raw[:, 1])), float(np.min(raw[:, 0]))],
              [float(np.max(raw[:, 1])), float(np.max(raw[:, 0]))]]
    m.fit_bounds(bounds)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;top:12px;left:50px;z-index:9999;background:white;'
        'padding:8px 12px;border:1px solid #bbb;border-radius:6px;font:13px Arial">'
        '<b>PathSnap baseline</b> · Blue: GPS · Red: snapped prediction · Grey: nearby graph edges'
        '</div>'
    ))
    m.save(str(out_file))
    return out_file


def make_offline_map(trip: Trip, result: dict, out_file: Path) -> Path:
    """Self-contained SVG map for demonstrations without internet/CDN access."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    gps_xy = np.asarray(result["gps_xy"], dtype=float)
    snap_lonlat = np.asarray(result["snapped_lonlat"], dtype=float)
    # Reproject snapped coordinates to the same local metre plane.
    from .core import _project
    origin_lon = float(np.mean(trip.track[:, 0]))
    origin_lat = float(np.mean(trip.track[:, 1]))
    snap_xy = _project(snap_lonlat, origin_lon, origin_lat)

    edge_start = np.asarray(result["edge_start"], dtype=float)
    edge_end = np.asarray(result["edge_end"], dtype=float)
    candidate_ids = np.asarray(result["candidate_ids"], dtype=int)
    candidate_sample = candidate_ids[::max(1, len(candidate_ids) // 300)].reshape(-1)
    road_ids = set(int(x) for x in candidate_sample)
    if len(road_ids) > 1800:
        road_ids = set(sorted(road_ids)[:1800])

    all_xy = np.vstack([gps_xy, snap_xy])
    if road_ids:
        ids = np.fromiter(road_ids, dtype=int)
        all_xy = np.vstack([all_xy, edge_start[ids], edge_end[ids]])
    min_x, min_y = np.min(all_xy, axis=0)
    max_x, max_y = np.max(all_xy, axis=0)
    span_x = max(float(max_x - min_x), 1.0)
    span_y = max(float(max_y - min_y), 1.0)
    width, height, margin = 1120.0, 760.0, 42.0
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    draw_w, draw_h = span_x * scale, span_y * scale
    offset_x = (width - draw_w) / 2.0
    offset_y = (height - draw_h) / 2.0

    def xy_to_svg(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        sx = offset_x + (pts[:, 0] - min_x) * scale
        sy = height - (offset_y + (pts[:, 1] - min_y) * scale)
        return np.column_stack([sx, sy])

    gps_svg = xy_to_svg(gps_xy)
    snap_svg = xy_to_svg(snap_xy)
    road_svg_start = xy_to_svg(edge_start[np.fromiter(road_ids, dtype=int)]) if road_ids else np.empty((0,2))
    road_svg_end = xy_to_svg(edge_end[np.fromiter(road_ids, dtype=int)]) if road_ids else np.empty((0,2))

    def points_attr(points: np.ndarray) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    roads = "\n".join(
        f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" />'
        for a, b in zip(road_svg_start, road_svg_end)
    )
    markers = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" />' for x, y in gps_svg[::max(1, len(gps_svg)//140)]
    )
    m = result["metrics"]
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathSnap offline map — {trip.trip_id}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f8fafc;color:#172033;font:14px/1.4 Arial,sans-serif}}
.top{{padding:12px 16px;background:#111827;color:white}}.top b{{font-size:16px}}.top span{{color:#cbd5e1;margin-left:12px}}
main{{padding:12px}}svg{{width:100%;height:calc(100vh - 100px);min-height:420px;background:#fff;border:1px solid #dbe2ea;border-radius:8px;touch-action:none}}
#roads line{{stroke:#aab4c3;stroke-width:1.5;opacity:.7}}#gps polyline{{fill:none;stroke:#2563eb;stroke-width:2.5;stroke-opacity:.78}}#gps circle{{fill:#2563eb;stroke:white;stroke-width:1}}
#pred polyline{{fill:none;stroke:#dc2626;stroke-width:4;stroke-opacity:.9;stroke-linejoin:round;stroke-linecap:round}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 4px;color:#475569}}.swatch{{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:6px}}
.hint{{margin-left:auto;color:#64748b}}
</style></head><body><div class="top"><b>PathSnap · Offline trajectory view</b><span>Trip {trip.trip_id} · {m['gps_points']} GPS points · median snap {m['median_offset_m']:.2f} m</span></div>
<main><svg id="map" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="GPS and predicted route map">
<g id="roads">{roads}</g>
<g id="gps"><polyline points="{points_attr(gps_svg)}" />{markers}</g>
<g id="pred"><polyline points="{points_attr(snap_svg)}" /></g>
</svg><div class="legend"><span><i class="swatch" style="background:#2563eb"></i>Raw GPS</span><span><i class="swatch" style="background:#dc2626"></i>Nearest-road prediction</span><span><i class="swatch" style="background:#aab4c3"></i>Nearby road edges</span><span class="hint">Mouse wheel: zoom · Drag: pan</span></div></main>
<script>
const svg=document.getElementById('map');let vb={{x:0,y:0,w:{width},h:{height}}},drag=null;
function apply(){{svg.setAttribute('viewBox',`${{vb.x}} ${{vb.y}} ${{vb.w}} ${{vb.h}}`)}}
svg.addEventListener('wheel',e=>{{e.preventDefault();const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width,py=(e.clientY-r.top)/r.height;const f=e.deltaY<0?.85:1.18;const nw=vb.w*f,nh=vb.h*f;vb.x+=(vb.w-nw)*px;vb.y+=(vb.h-nh)*py;vb.w=nw;vb.h=nh;apply()}},{{passive:false}});
svg.addEventListener('pointerdown',e=>{{drag={{x:e.clientX,y:e.clientY,vx:vb.x,vy:vb.y}};svg.setPointerCapture(e.pointerId)}});
svg.addEventListener('pointermove',e=>{{if(!drag)return;const r=svg.getBoundingClientRect();vb.x=drag.vx-(e.clientX-drag.x)*vb.w/r.width;vb.y=drag.vy-(e.clientY-drag.y)*vb.h/r.height;apply()}});
svg.addEventListener('pointerup',()=>drag=null);
</script></body></html>'''
    out_file.write_text(html, encoding="utf-8")
    return out_file
