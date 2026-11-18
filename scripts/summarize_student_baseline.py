from pathlib import Path
import csv, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.core import load_trip, nearest_edge_match
DATA = ROOT / "data" / "Student folder"
OUT = ROOT / "outputs"
rows=[]
for folder in sorted((DATA / "student_test").iterdir()):
    if not folder.is_dir(): continue
    trip_id=folder.name
    started=time.perf_counter()
    trip=load_trip(DATA,"student_test",trip_id,include_route=False)
    result=nearest_edge_match(trip)
    elapsed=time.perf_counter()-started
    m=result['metrics']
    rows.append({
        'trip_id':trip_id,'gps_observations':m['gps_points'],'road_segments':m['road_segments'],
        'mean_offset_m':round(m['mean_offset_m'],3),'median_offset_m':round(m['median_offset_m'],3),
        'p95_offset_m':round(m['p95_offset_m'],3),'directed_edge_continuity':round(m['directed_edge_continuity'],4),
        'processing_seconds':round(elapsed,3)
    })
path=OUT/'student_test_baseline_summary.csv'
with path.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader();w.writerows(rows)
print('Saved',path)
for r in rows: print(r)
