from __future__ import annotations

from pathlib import Path
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .core import load_trip, nearest_edge_match
from .visuals import make_map, make_offline_map

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "Student folder"))
MAP_ROOT = ROOT / "outputs" / "maps"
OFFLINE_MAP_ROOT = ROOT / "outputs" / "offline_maps"
MAP_ROOT.mkdir(parents=True, exist_ok=True)
OFFLINE_MAP_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("pathsnap.api")
app = FastAPI(title="PathSnap SRNs Demo API", version="0.1.0")
app.mount("/maps", StaticFiles(directory=str(MAP_ROOT)), name="maps")
app.mount("/offline-maps", StaticFiles(directory=str(OFFLINE_MAP_ROOT)), name="offline-maps")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathSnap | SRNs</title>
<style>
:root{--ink:#172033;--muted:#657189;--line:#dce2ec;--blue:#2457d6;--bg:#f5f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif;color:var(--ink)}
header{background:#111827;color:#fff;padding:22px max(22px,calc((100vw - 1160px)/2))}header small{color:#b7c4da;letter-spacing:.14em;text-transform:uppercase;font-weight:700}h1{margin:4px 0;font-size:26px}header p{margin:0;color:#d1d9e6}
main{max-width:1160px;margin:24px auto;padding:0 18px}.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 4px 18px #1720330a}
.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:block;color:var(--muted);font-weight:650;font-size:13px;margin-bottom:6px}select,button{height:42px;border:1px solid var(--line);border-radius:9px;padding:0 13px;font:inherit}select{min-width:220px;background:white}button{background:var(--blue);color:white;border:0;font-weight:700;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.metric{border:1px solid var(--line);padding:13px;border-radius:10px}.metric span{display:block;color:var(--muted);font-size:12px}.metric b{font-size:21px}.status{color:var(--muted);margin-left:5px}iframe{width:100%;height:590px;border:1px solid var(--line);border-radius:10px;background:white}.foot{font-size:13px;color:var(--muted)}
@media(max-width:800px){.metrics{grid-template-columns:repeat(2,1fr)}iframe{height:460px}}
</style></head><body>
<header><small>UE24CS352A · Machine Learning Hackathon</small><h1>PathSnap — Distributed Map-Matching & Route Reconstruction</h1><p>SRNs · PES1UG24AM004 / 005 / 006 / 007 · First-round baseline demo</p></header>
<main>
<section class="panel"><div class="controls"><div><label for="trip">Choose a student-testing trip</label><select id="trip"><option>Loading trips…</option></select></div><button id="run">Run nearest-road baseline</button><span id="status" class="status">Ready</span></div></section>
<section class="panel"><div class="metrics" id="metrics">
<div class="metric"><span>GPS observations</span><b>—</b></div><div class="metric"><span>Road segments</span><b>—</b></div><div class="metric"><span>Median snap offset</span><b>—</b></div><div class="metric"><span>95th percentile offset</span><b>—</b></div><div class="metric"><span>Directed continuity</span><b>—</b></div></div></section>
<section class="panel"><h2 style="margin:0 0 12px">Trajectory map</h2><iframe id="map" title="PathSnap trajectory map"></iframe><p class="foot">Blue = raw GPS observations · Red = nearest-road-segment prediction · Grey = nearby road-network edges. This is a baseline, not the final learned/HMM system. Reference labels are not sent to the browser.</p></section>
<section class="panel"><b>Demo note</b><p class="foot">This proof-of-work uses only the trip's GPS observations and road graph at inference time. The supplied .route file is not used by the prediction function or displayed in this interface.</p></section>
</main>
<script>
const tripSel=document.getElementById('trip'),runBtn=document.getElementById('run'),statusEl=document.getElementById('status'),metricsEl=document.getElementById('metrics'),mapEl=document.getElementById('map');
async function init(){try{const r=await fetch('/api/trips');const d=await r.json();tripSel.innerHTML='';d.trips.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;tripSel.appendChild(o)});if(d.trips.includes('00000052'))tripSel.value='00000052';}catch(e){statusEl.textContent='Could not load trips: '+e}}
runBtn.addEventListener('click',async()=>{runBtn.disabled=true;statusEl.textContent='Matching GPS points to road segments…';try{const id=tripSel.value;const r=await fetch('/api/match/'+id,{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.detail||r.statusText);const m=d.metrics;const vals=[m.gps_points.toLocaleString(),m.road_segments.toLocaleString(),m.median_offset_m.toFixed(1)+' m',m.p95_offset_m.toFixed(1)+' m',(m.directed_edge_continuity*100).toFixed(1)+'%'];const labels=['GPS observations','Road segments','Median snap offset','95th percentile offset','Directed continuity'];metricsEl.innerHTML=vals.map((v,i)=>`<div class="metric"><span>${labels[i]}</span><b>${v}</b></div>`).join('');mapEl.src=d.map_url+'?v='+Date.now();statusEl.textContent='Completed · '+id;}catch(e){statusEl.textContent='Error: '+e.message;}finally{runBtn.disabled=false;}});
init();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok", "data_root_exists": DATA_ROOT.exists()}


@app.get("/api/trips")
def list_trips():
    folder = DATA_ROOT / "student_test"
    if not folder.exists():
        raise HTTPException(500, f"Dataset not found at {folder}. Run extract_dataset.py first.")
    trips = sorted(p.name for p in folder.iterdir() if p.is_dir())
    return {"partition": "student_test", "trips": trips}


@app.post("/api/match/{trip_id}")
def match_trip(trip_id: str):
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(400, "Trip ID must be an eight-digit ID")
    try:
        trip = load_trip(DATA_ROOT, "student_test", trip_id, include_route=False)
        result = nearest_edge_match(trip)
        map_file = MAP_ROOT / f"{trip_id}.html"
        make_map(trip, result, map_file)
        offline_map_file = OFFLINE_MAP_ROOT / f"{trip_id}.html"
        make_offline_map(trip, result, offline_map_file)
        logger.info("INFERENCE trip=%s gps_points=%d road_segments=%d median_offset_m=%.2f p95_offset_m=%.2f continuity=%.4f",
                    trip_id, result["metrics"]["gps_points"], result["metrics"]["road_segments"],
                    result["metrics"]["median_offset_m"], result["metrics"]["p95_offset_m"],
                    result["metrics"]["directed_edge_continuity"])
        return {"trip_id": trip_id, "metrics": result["metrics"], "map_url": f"/offline-maps/{trip_id}.html", "folium_map_url": f"/maps/{trip_id}.html"}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to process trip %s", trip_id)
        raise HTTPException(500, f"Inference failed: {exc}") from exc
