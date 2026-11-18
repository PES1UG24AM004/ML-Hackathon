"""
PathSnap - Unit Tests for PES1UG24AM007 (API, UI & Maps)
Validates FastAPI endpoints, offline vendor asset delivery, zero-tile security,
road-network bounding geometry extraction, and map rendering response formats.
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient
from api.main import app


class TestAPIAndUIMaps(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.ui_dir = PROJECT_ROOT / "src" / "ui"

    def test_local_offline_vendor_leaflet_assets(self):
        """Verify Leaflet JS and CSS are served locally with zero CDN dependency."""
        res_js = self.client.get("/static/vendor/leaflet.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertGreater(len(res_js.content), 50000)

        res_css = self.client.get("/static/vendor/leaflet.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertGreater(len(res_css.content), 5000)

    def test_frontend_zero_external_tile_dependencies(self):
        """Ensure no external Carto, Mapbox, or unpkg CDN dependencies exist in UI code."""
        app_js = (self.ui_dir / "app.js").read_text(encoding="utf-8")
        index_html = (self.ui_dir / "index.html").read_text(encoding="utf-8")

        self.assertNotIn("cartocdn", app_js.lower())
        self.assertNotIn("tile.openstreetmap", app_js.lower())
        self.assertNotIn("L.tileLayer", app_js)
        self.assertNotIn("unpkg.com/leaflet", index_html)

    def test_api_trip_geometry_endpoint(self):
        """Verify GET /api/trip/{trip_id} returns genuine road network and GPS points."""
        res = self.client.get("/api/trip/00000052?dataset=student_test")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["trip_id"], "00000052")
        self.assertGreater(len(data["network_segments"]), 500)
        self.assertGreater(len(data["gps_points"]), 50)
        self.assertEqual(len(data["bounds"]), 2)

    def test_api_match_returns_start_end_markers(self):
        """Verify POST /api/match/{trip_id} returns predicted_segments, start_point, and end_point."""
        res = self.client.post("/api/match/00000052?dataset=student_test")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("predicted_segments", data)
        self.assertIn("start_point", data)
        self.assertIn("end_point", data)

    def test_offline_vector_svg_map_endpoint(self):
        """Verify GET /offline-maps/{trip_id}.html returns standalone SVG map."""
        res = self.client.get("/offline-maps/00000052.html")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<svg", res.text)


if __name__ == "__main__":
    unittest.main()
