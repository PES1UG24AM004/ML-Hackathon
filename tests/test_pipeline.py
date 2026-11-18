"""
PathSnap - Verification & Smoke Test Suite
Tests data processing, projection, model classes, serialized artifacts, and HMM utilities.
"""

import os
import sys
import unittest
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_processing import point_to_segments_dist, make_local_aeqd_transformer
from models import (
    StandardScalerNP,
    LogisticRegressionNP,
    GaussianNaiveBayesNP,
    DistanceWeightedKNN,
    RandomForestNP,
    GradientBoostingNP,
    MLPClassifierNP,
)
from hmm_viterbi import load_final_mlp, collapse_consecutive, sequence_metrics


class TestDataProcessing(unittest.TestCase):
    def test_point_to_segment_distance(self):
        # Point at (0, 1), segment from (-1, 0) to (1, 0)
        px, py = 0.0, 1.0
        ax, ay = np.array([-1.0]), np.array([0.0])
        bx, by = np.array([1.0]), np.array([0.0])

        dists, t, proj_x, proj_y = point_to_segments_dist(px, py, ax, ay, bx, by)
        self.assertAlmostEqual(dists[0], 1.0, places=5)
        self.assertAlmostEqual(t[0], 0.5, places=5)
        self.assertAlmostEqual(proj_x[0], 0.0, places=5)
        self.assertAlmostEqual(proj_y[0], 0.0, places=5)

    def test_aeqd_transformer(self):
        to_local, to_wgs84 = make_local_aeqd_transformer(lon0=8.68, lat0=50.11)
        x, y = to_local.transform(8.68, 50.11)
        self.assertAlmostEqual(x, 0.0, places=2)
        self.assertAlmostEqual(y, 0.0, places=2)
        lon, lat = to_wgs84.transform(x, y)
        self.assertAlmostEqual(lon, 8.68, places=4)
        self.assertAlmostEqual(lat, 50.11, places=4)


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(0, 1, size=(200, 16)).astype(np.float32)
        self.y = rng.integers(0, 2, size=200).astype(np.int32)

    def test_scaler(self):
        scaler = StandardScalerNP()
        X_scaled = scaler.fit_transform(self.X)
        self.assertAlmostEqual(float(np.mean(X_scaled)), 0.0, places=3)
        self.assertAlmostEqual(float(np.std(X_scaled)), 1.0, places=1)

    def test_all_six_models(self):
        models = [
            ("Logistic Regression", LogisticRegressionNP(max_iter=10)),
            ("Gaussian Naive Bayes", GaussianNaiveBayesNP()),
            ("Distance-Weighted kNN", DistanceWeightedKNN(n_neighbors=5, max_index_samples=100)),
            ("Random Forest", RandomForestNP(n_estimators=3, max_depth=4)),
            ("Gradient Boosted Trees", GradientBoostingNP(n_estimators=5, max_depth=3)),
            ("Shallow MLP", MLPClassifierNP(hidden_dim=16, epochs=3, batch_size=64)),
        ]
        for name, m in models:
            m.fit(self.X, self.y)
            probs = m.predict_proba(self.X)
            self.assertEqual(probs.shape, (len(self.X), 2), f"{name} invalid shape")
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)), f"{name} probs out of bounds")
            self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4), f"{name} probs sum != 1")


class TestArtifactsAndHMM(unittest.TestCase):
    def test_load_final_mlp_artifact(self):
        art_path = os.path.join(SRC_DIR, "..", "artifacts", "final_mlp_model.npz")
        self.assertTrue(os.path.exists(art_path), f"Artifact missing at {art_path}")
        model, mean, std, params = load_final_mlp(art_path)
        self.assertEqual(len(mean), 16)
        self.assertEqual(len(std), 16)
        self.assertEqual(model.hidden_dim, 123)

        # Test forward pass with dummy observation
        x_dummy = np.zeros((1, 16), dtype=np.float32)
        x_scaled = ((x_dummy - mean) / std).astype(np.float32)
        p = model.predict_proba(x_scaled)
        self.assertEqual(p.shape, (1, 2))
        self.assertTrue(0.0 <= p[0, 1] <= 1.0)

    def test_collapse_consecutive(self):
        raw = [1, 1, 2, 3, 3, 3, 2, 2, 4]
        collapsed = collapse_consecutive(raw)
        self.assertEqual(collapsed, [1, 2, 3, 2, 4])

    def test_sequence_metrics(self):
        # Arcs: 0: (0->1), 1: (1->2), 2: (2->3)
        arcs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        pred = np.array([0, 0, 1, 2], dtype=np.int64)
        gt = np.array([0, 1, 2], dtype=np.int64)

        metrics = sequence_metrics(pred, gt, arcs)
        self.assertEqual(metrics["continuity"], 1.0)
        self.assertEqual(metrics["arc_f1"], 1.0)
        self.assertEqual(metrics["arc_jaccard"], 1.0)
        self.assertEqual(metrics["lcs_length"], 3)
        self.assertEqual(metrics["lcs_f1"], 1.0)
        self.assertEqual(metrics["lcs_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
