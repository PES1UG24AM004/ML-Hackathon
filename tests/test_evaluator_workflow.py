"""
PathSnap — Unit & Integration Test Suite for Evaluator Dataset Workflow
Validates strict label isolation, prediction generation, evaluation scoring, and API compatibility.
"""

import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing import load_trip, project_trip_to_meters
from feature_engineering import generate_candidates_for_trip, build_network_index
from hmm_viterbi import load_final_mlp, sequence_metrics, collapse_consecutive


class TestEvaluatorDatasetIsolation(unittest.TestCase):
    def setUp(self):
        self.inputs_dir = PROJECT_ROOT / "evaluator_data" / "inputs"
        self.labels_dir = PROJECT_ROOT / "evaluator_data" / "private_labels"
        self.expected_trips = [
            "00000015", "00000017", "00000023", "00000030", "00000049",
            "00000059", "00000067", "00000073", "00000087", "00000094"
        ]

    def test_all_ten_evaluator_trips_present(self):
        self.assertTrue(self.inputs_dir.is_dir(), "evaluator_data/inputs/ directory must exist")
        discovered = sorted([d.name for d in self.inputs_dir.iterdir() if d.is_dir()])
        self.assertEqual(discovered, self.expected_trips, "All 10 evaluator trips must be discovered")

    def test_strict_label_isolation_in_inputs(self):
        """CRITICAL: Ensure zero .route files exist anywhere inside evaluator_data/inputs/."""
        route_leaks = []
        for root, _, files in os.walk(self.inputs_dir):
            for f in files:
                if f.endswith(".route"):
                    route_leaks.append(os.path.join(root, f))
        self.assertEqual(route_leaks, [], f"Data leakage detected! .route files found in inputs: {route_leaks}")

    def test_required_input_files_exist_and_non_empty(self):
        for trip_id in self.expected_trips:
            tdir = self.inputs_dir / trip_id
            for ext in [".track", ".nodes", ".arcs"]:
                fpath = tdir / f"{trip_id}{ext}"
                self.assertTrue(fpath.is_file(), f"Missing input file: {fpath}")
                self.assertGreater(fpath.stat().st_size, 0, f"Empty file: {fpath}")

    def test_private_labels_isolated_and_valid(self):
        self.assertTrue(self.labels_dir.is_dir(), "Private labels directory must exist")
        for trip_id in self.expected_trips:
            rpath = self.labels_dir / f"{trip_id}.route"
            self.assertTrue(rpath.is_file(), f"Missing isolated label: {rpath}")
            with open(rpath, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertGreater(len(lines), 0, f"Empty label file: {rpath}")


class TestEvaluatorPredictionAndScoring(unittest.TestCase):
    def setUp(self):
        self.preds_dir = PROJECT_ROOT / "evaluator_outputs" / "predictions"
        self.metrics_dir = PROJECT_ROOT / "evaluator_outputs" / "metrics"
        self.model_path = PROJECT_ROOT / "artifacts" / "final_mlp_model.npz"

    def test_model_artifacts_available(self):
        self.assertTrue(self.model_path.is_file(), "final_mlp_model.npz must exist in artifacts/")
        model, mean, std, params = load_final_mlp(str(self.model_path))
        self.assertIsNotNone(model)
        self.assertEqual(len(mean), 16, "Feature dimension must be 16")
        self.assertEqual(len(std), 16, "Feature dimension must be 16")

    def test_prediction_output_manifest(self):
        summary_file = self.preds_dir / "prediction_summary.json"
        self.assertTrue(summary_file.is_file(), "prediction_summary.json must be generated")
        with open(summary_file) as f:
            data = json.load(f)
        self.assertEqual(data.get("successful_trips"), 10, "All 10 trips must be successfully predicted")

    def test_scoring_output_artifacts(self):
        csv_file = self.metrics_dir / "evaluator_metrics_per_trip.csv"
        agg_file = self.metrics_dir / "evaluator_aggregate_summary.json"
        self.assertTrue(csv_file.is_file(), "evaluator_metrics_per_trip.csv must exist")
        self.assertTrue(agg_file.is_file(), "evaluator_aggregate_summary.json must exist")

        with open(agg_file) as f:
            agg = json.load(f)
        m = agg.get("metrics_summary", {})
        self.assertIn("hmm_top1_accuracy", m)
        self.assertGreater(m["hmm_top1_accuracy"]["mean"], 0.70, "HMM accuracy should exceed 70%")
        self.assertGreater(m["hmm_route_continuity"]["mean"], 0.90, "HMM continuity should exceed 90%")


class TestAPICompatibility(unittest.TestCase):
    def setUp(self):
        import urllib.request
        self.base = "http://127.0.0.1:8000"
        self.urlopen = urllib.request.urlopen

    def test_health_endpoint(self):
        res = json.loads(self.urlopen(f"{self.base}/health").read().decode())
        self.assertEqual(res.get("status"), "ok")

    def test_datasets_endpoint(self):
        res = json.loads(self.urlopen(f"{self.base}/api/datasets").read().decode())
        ds_ids = [d["id"] for d in res.get("datasets", [])]
        self.assertIn("student_test", ds_ids)
        self.assertIn("evaluator_test", ds_ids)

    def test_evaluator_trips_enumeration(self):
        res = json.loads(self.urlopen(f"{self.base}/api/trips?dataset=evaluator_test").read().decode())
        self.assertEqual(res.get("count"), 10)
        self.assertEqual(len(res.get("trips", [])), 10)

    def test_evaluator_match_preserves_privacy(self):
        """Verify POST /api/match/{trip_id} on evaluator trip returns diagnostics and does not leak gt_route."""
        import urllib.request
        req = urllib.request.Request(f"{self.base}/api/match/00000015?dataset=evaluator_test", data=b"", method="POST")
        res = json.loads(self.urlopen(req).read().decode())
        self.assertEqual(res.get("trip_id"), "00000015")
        self.assertIn("predicted_arc_sequence", res)
        self.assertIn("predicted_segments", res)
        # Verify ground-truth route sequence is NOT leaked
        self.assertNotIn("gt_route", res)
        self.assertNotIn("ground_truth_sequence", res)


if __name__ == "__main__":
    unittest.main()
