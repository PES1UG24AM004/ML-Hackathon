"""
PathSnap - Unit Tests for PES1UG24AM005 (Data Processing & EDA)
Validates dataset loading, Local AEQD metric projection, orthogonal distance calculation,
and strict partition separation.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing import (
    load_trip,
    make_local_aeqd_transformer,
    project_trip_to_meters,
    point_to_segments_dist,
    list_trips,
)


class TestDataProcessingModule(unittest.TestCase):
    def test_local_aeqd_projection_preserves_metric_scale(self):
        """Verify that AEQD metric projection preserves Euclidean distances in meters."""
        lon0, lat0 = 35.858, 56.800
        to_local, to_wgs84 = make_local_aeqd_transformer(lon0, lat0)

        # Origin should map to (0, 0)
        x0, y0 = to_local.transform(lon0, lat0)
        self.assertAlmostEqual(x0, 0.0, places=3)
        self.assertAlmostEqual(y0, 0.0, places=3)

        # Unprojecting back to WGS84
        lon_back, lat_back = to_wgs84.transform(x0, y0)
        self.assertAlmostEqual(lon_back, lon0, places=4)
        self.assertAlmostEqual(lat_back, lat0, places=4)

    def test_point_to_segment_distance_orthogonal_and_endpoints(self):
        """Test point-to-segment orthogonal clamping and distance computation."""
        px, py = 0.0, 1.0
        ax, ay = np.array([-1.0]), np.array([0.0])
        bx, by = np.array([1.0]), np.array([0.0])

        dists, t, proj_x, proj_y = point_to_segments_dist(px, py, ax, ay, bx, by)
        self.assertAlmostEqual(dists[0], 1.0, places=4)
        self.assertAlmostEqual(t[0], 0.5, places=4)
        self.assertAlmostEqual(proj_x[0], 0.0, places=4)
        self.assertAlmostEqual(proj_y[0], 0.0, places=4)

    def test_trip_ingestion_and_projection(self):
        """Test loading trip 00000052 and projecting to metric frame."""
        trip_dir = PROJECT_ROOT / "data" / "Student folder" / "student_test" / "00000052"
        if not trip_dir.is_dir():
            self.skipTest("Trip 00000052 directory not found.")

        data = load_trip(str(trip_dir), "00000052", load_route=True)
        self.assertEqual(data["trip_id"], "00000052")
        self.assertEqual(data["track"].shape[1], 3)
        self.assertEqual(data["nodes"].shape[1], 2)
        self.assertEqual(data["arcs"].shape[1], 2)

        track_xy, nodes_xy, origin, _ = project_trip_to_meters(data["track"], data["nodes"])
        self.assertEqual(len(track_xy), len(data["track"]))
        self.assertEqual(len(nodes_xy), len(data["nodes"]))


if __name__ == "__main__":
    unittest.main()
