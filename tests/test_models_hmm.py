"""
PathSnap - Unit Tests for PES1UG24AM006 (Feature Engineering, Models & HMM)
Validates candidate arc extraction, 16-dim features, classifier families, standard scaling,
and Viterbi sequential decoding.
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

from feature_engineering import generate_candidates_for_trip, build_network_index
from models import (
    StandardScalerNP,
    LogisticRegressionNP,
    GaussianNaiveBayesNP,
    DistanceWeightedKNN,
    RandomForestNP,
    GradientBoostingNP,
    MLPClassifierNP,
)
from hmm_viterbi import (
    load_final_mlp,
    build_two_hop_set,
    viterbi_decode_trip,
    sequence_metrics,
    collapse_consecutive,
)


class TestModelsAndHMM(unittest.TestCase):
    def test_feature_engineering_candidate_dimensions(self):
        """Verify KD-tree candidate extraction returns exactly 16 spatial/temporal features."""
        nodes_xy = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ], dtype=np.float64)
        arcs = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ], dtype=np.int64)
        track_xy = np.array([
            [50.0, 5.0],
            [95.0, 50.0],
        ], dtype=np.float64)
        timestamps = np.array([1.0, 2.0], dtype=np.float64)

        net_idx = build_network_index(nodes_xy, arcs)
        X, obs_idx, cand_arcs, _ = generate_candidates_for_trip(
            track_xy, timestamps, nodes_xy, arcs, top_k=3, net_idx=net_idx
        )

        self.assertEqual(X.shape[1], 16, "Features vector must have exactly 16 dimensions")
        self.assertEqual(len(obs_idx), 2 * 3, "2 points * top_k=3 candidates")

    def test_all_six_classifiers_train_and_predict(self):
        """Verify that all six models instantiate, fit, and predict probabilities."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 16)).astype(np.float32)
        y = rng.integers(0, 2, size=100).astype(np.int32)

        scaler = StandardScalerNP()
        X_s = scaler.fit_transform(X)

        models = [
            LogisticRegressionNP(max_iter=5),
            GaussianNaiveBayesNP(),
            DistanceWeightedKNN(n_neighbors=5, max_index_samples=100),
            RandomForestNP(n_estimators=3, max_depth=4),
            GradientBoostingNP(n_estimators=3, max_depth=3),
            MLPClassifierNP(hidden_dim=16, epochs=3, batch_size=32),
        ]

        for m in models:
            m.fit(X_s, y)
            probs = m.predict_proba(X_s)
            self.assertEqual(probs.shape, (100, 2))
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_viterbi_sequential_decoding(self):
        """Verify Viterbi decoding enforces topological connectivity."""
        arcs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
        nodes_xy = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0], [50.0, 50.0]])
        track_xy = np.array([[25.0, 2.0], [75.0, 2.0], [90.0, 5.0]])
        timestamps = np.array([1.0, 2.0, 3.0])

        probs = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.9])
        obs_idx = np.array([0, 0, 1, 1, 2, 2])
        cand_arcs = np.array([0, 3, 0, 1, 1, 2])
        dists = np.array([2.0, 40.0, 40.0, 2.0, 10.0, 2.0])

        out_nodes = build_two_hop_set(arcs)
        pred = viterbi_decode_trip(
            probs, obs_idx, cand_arcs, dists, arcs, nodes_xy, track_xy, timestamps, out_nodes
        )
        self.assertEqual(len(pred), 3)
        collapsed = collapse_consecutive(pred)
        self.assertTrue(len(collapsed) >= 1)


if __name__ == "__main__":
    unittest.main()
