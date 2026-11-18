"""
PathSnap - Six Distinct ML Models Across Five Algorithm Families
Implemented in pure NumPy / SciPy for reproducibility and zero heavy-dependency lock-in.
Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit


class StandardScalerNP:
    """Standardize features using training-partition mean and std only (leakage-free)."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0, dtype=np.float64)
        self.std_ = np.std(X, axis=0, dtype=np.float64)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# 1. Logistic Regression (L2-regularized via L-BFGS-B) — Linear & Probabilistic
# ---------------------------------------------------------------------------
class LogisticRegressionNP:
    def __init__(self, C: float = 1.0, max_iter: int = 100, class_weight_pos: float = 1.0):
        self.C = C
        self.max_iter = max_iter
        self.class_weight_pos = class_weight_pos
        self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X64 = X.astype(np.float64)
        y64 = y.astype(np.float64)
        n, d = X64.shape
        weights = np.where(y64 == 1, self.class_weight_pos, 1.0)
        reg = 1.0 / (2.0 * self.C * n)

        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = np.clip(X64 @ w + b, -30.0, 30.0)
            p = expit(z)
            bce = -np.mean(weights * (y64 * np.log(p + 1e-12) + (1.0 - y64) * np.log(1.0 - p + 1e-12)))
            loss = bce + reg * np.dot(w, w)
            err = weights * (p - y64) / n
            gw = X64.T @ err + 2.0 * reg * w
            gb = np.sum(err)
            return loss, np.append(gw, gb)

        init = np.zeros(d + 1, dtype=np.float64)
        res = minimize(loss_and_grad, init, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.w = res.x[:-1]
        self.b = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = expit(np.clip(X.astype(np.float64) @ self.w + self.b, -30.0, 30.0))
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# 2. Gaussian Naive Bayes — Linear & Probabilistic
# ---------------------------------------------------------------------------
class GaussianNaiveBayesNP:
    def __init__(self, var_smoothing: float = 1e-6):
        self.var_smoothing = var_smoothing
        self.means_ = None
        self.vars_ = None
        self.log_priors_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X64 = X.astype(np.float64)
        self.means_ = np.zeros((2, X64.shape[1]), dtype=np.float64)
        self.vars_ = np.zeros((2, X64.shape[1]), dtype=np.float64)
        self.log_priors_ = np.zeros(2, dtype=np.float64)
        eps = self.var_smoothing * np.var(X64, axis=0).max()
        for c in (0, 1):
            Xc = X64[y == c]
            self.means_[c] = Xc.mean(axis=0)
            self.vars_[c] = Xc.var(axis=0) + eps
            self.log_priors_[c] = np.log(len(Xc) / len(X64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X64 = X.astype(np.float64)
        log_lik = np.zeros((len(X64), 2), dtype=np.float64)
        for c in (0, 1):
            ll = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.vars_[c]) + ((X64 - self.means_[c]) ** 2) / self.vars_[c],
                axis=1,
            )
            log_lik[:, c] = self.log_priors_[c] + ll
        # Log-sum-exp trick
        m = np.max(log_lik, axis=1, keepdims=True)
        probs = np.exp(log_lik - m)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


# ---------------------------------------------------------------------------
# 3. Distance-Weighted k-Nearest Neighbors — Instance-based
# ---------------------------------------------------------------------------
class DistanceWeightedKNN:
    def __init__(self, n_neighbors: int = 15, max_index_samples: int = 80000, random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.max_index_samples = max_index_samples
        self.random_state = random_state
        self.tree_ = None
        self.y_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        if len(X) > self.max_index_samples:
            idx = rng.choice(len(X), size=self.max_index_samples, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]
        else:
            X_sub = X
            y_sub = y
        self.tree_ = cKDTree(X_sub)
        self.y_train_ = y_sub.astype(np.float64)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dists, idxs = self.tree_.query(X, k=self.n_neighbors, workers=-1)
        if self.n_neighbors == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        weights = 1.0 / (dists + 1e-4)
        neighbor_y = self.y_train_[idxs]
        p1 = np.sum(weights * neighbor_y, axis=1) / np.sum(weights, axis=1)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# Fast Vectorized Histogram CART Tree (used by Random Forest & Gradient Boosting)
# ---------------------------------------------------------------------------
class _FastTreeNode:
    __slots__ = ("feat", "thresh", "left", "right", "val")

    def __init__(self, feat=-1, thresh=0.0, left=None, right=None, val=0.0):
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.val = val


class FastHistogramTree:
    """
    Fast binned decision tree supporting both Gini classification (mode='gini')
    and MSE regression (mode='mse') for Gradient Boosting.
    """
    def __init__(self, max_depth=6, min_samples_leaf=20, max_features=None, n_bins=32, mode="gini", rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_bins = n_bins
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.root = None

    def _fit_node(self, X, target, bin_edges, depth):
        n, d = X.shape
        if n == 0:
            return _FastTreeNode(val=0.5)
        mean_val = float(target.mean())
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return _FastTreeNode(val=mean_val)
        if self.mode == "gini" and (mean_val <= 1e-7 or mean_val >= 1.0 - 1e-7):
            return _FastTreeNode(val=mean_val)

        m_feats = d if self.max_features is None else min(d, max(1, int(self.max_features)))
        feat_idxs = self.rng.choice(d, size=m_feats, replace=False) if m_feats < d else np.arange(d)

        best_gain = 1e-9
        best_feat = -1
        best_thresh = 0.0

        total_sum = float(target.sum())
        if self.mode == "gini":
            p_all = total_sum / n
            parent_imp = 2.0 * p_all * (1.0 - p_all)
        else:
            parent_score = (total_sum * total_sum) / n

        for f in feat_idxs:
            edges = bin_edges[f]
            if len(edges) == 0:
                continue
            col = X[:, f]
            b_idx = np.searchsorted(edges, col, side="right")
            nb = len(edges) + 1
            counts = np.bincount(b_idx, minlength=nb)
            sums = np.bincount(b_idx, weights=target, minlength=nb)

            c_left = np.cumsum(counts[:-1])
            s_left = np.cumsum(sums[:-1])
            c_right = n - c_left
            s_right = total_sum - s_left

            valid = (c_left >= self.min_samples_leaf) & (c_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            if self.mode == "gini":
                p_l = s_left[valid] / c_left[valid]
                p_r = s_right[valid] / c_right[valid]
                imp_l = 2.0 * p_l * (1.0 - p_l)
                imp_r = 2.0 * p_r * (1.0 - p_r)
                w_imp = (c_left[valid] * imp_l + c_right[valid] * imp_r) / n
                gains = parent_imp - w_imp
            else:
                scores = (s_left[valid] ** 2) / c_left[valid] + (s_right[valid] ** 2) / c_right[valid]
                gains = (scores - parent_score) / n

            idx_best = int(np.argmax(gains))
            if gains[idx_best] > best_gain:
                best_gain = float(gains[idx_best])
                best_feat = int(f)
                valid_positions = np.flatnonzero(valid)
                best_thresh = float(edges[valid_positions[idx_best]])

        if best_feat == -1:
            return _FastTreeNode(val=mean_val)

        mask = X[:, best_feat] <= best_thresh
        if np.all(mask) or not np.any(mask):
            return _FastTreeNode(val=mean_val)
        left = self._fit_node(X[mask], target[mask], bin_edges, depth + 1)
        right = self._fit_node(X[~mask], target[~mask], bin_edges, depth + 1)
        return _FastTreeNode(feat=best_feat, thresh=best_thresh, left=left, right=right, val=mean_val)

    def fit(self, X, target, bin_edges=None):
        if bin_edges is None:
            bin_edges = []
            qs = np.linspace(0, 100, self.n_bins + 2)[1:-1]
            for f in range(X.shape[1]):
                u = np.unique(np.percentile(X[:, f], qs))
                bin_edges.append(u)
        self.root = self._fit_node(X, target, bin_edges, 0)
        return self

    def _predict_batch(self, node, X, out, idx):
        if node.feat == -1 or len(idx) == 0:
            out[idx] = node.val
            return
        mask = X[idx, node.feat] <= node.thresh
        self._predict_batch(node.left, X, out, idx[mask])
        self._predict_batch(node.right, X, out, idx[~mask])

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full(len(X), 0.5, dtype=np.float64)
        self._predict_batch(self.root, X, out, np.arange(len(X)))
        return out


# ---------------------------------------------------------------------------
# 4. Random Forest Classifier — Ensemble (Bagging)
# ---------------------------------------------------------------------------
class RandomForestNP:
    def __init__(self, n_estimators=25, max_depth=10, min_samples_leaf=20, max_features=4,
                 max_samples=200000, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        n = len(X)
        m = min(n, self.max_samples)
        # Precompute bin edges once on a subsample
        sub_idx = rng.choice(n, size=min(n, 50000), replace=False)
        qs = np.linspace(0, 100, 34)[1:-1]
        bin_edges = [np.unique(np.percentile(X[sub_idx, f], qs)) for f in range(X.shape[1])]

        y64 = y.astype(np.float64)
        for _ in range(self.n_estimators):
            boot = rng.choice(n, size=m, replace=True)
            tree = FastHistogramTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                mode="gini",
                rng=rng,
            )
            tree.fit(X[boot], y64[boot], bin_edges=bin_edges)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.mean([t.predict(X) for t in self.trees_], axis=0)
        preds = np.nan_to_num(preds, nan=0.5)
        preds = np.clip(preds, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - preds, preds])


# ---------------------------------------------------------------------------
# 5. Gradient Boosted Trees (LogitBoost / GBM) — Boosting
# ---------------------------------------------------------------------------
class GradientBoostingNP:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=4, min_samples_leaf=30,
                 subsample=0.7, max_train_samples=250000, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.trees_ = []
        self.init_logit_ = 0.0
        self.train_loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.train_loss_history_ = []

        if len(X) > self.max_train_samples:
            idx_base = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X_fit = X[idx_base]
            y_fit = y[idx_base].astype(np.float64)
        else:
            X_fit = X
            y_fit = y.astype(np.float64)

        n = len(X_fit)
        p_pos = np.clip(y_fit.mean(), 1e-5, 1.0 - 1e-5)
        self.init_logit_ = float(np.log(p_pos / (1.0 - p_pos)))
        F = np.full(n, self.init_logit_, dtype=np.float64)

        sub_idx = rng.choice(n, size=min(n, 50000), replace=False)
        qs = np.linspace(0, 100, 34)[1:-1]
        bin_edges = [np.unique(np.percentile(X_fit[sub_idx, f], qs)) for f in range(X_fit.shape[1])]

        m = max(100, int(n * self.subsample))
        for _ in range(self.n_estimators):
            p = expit(np.clip(F, -20.0, 20.0))
            residuals = y_fit - p  # negative gradient of binary cross-entropy

            s_idx = rng.choice(n, size=m, replace=False) if self.subsample < 1.0 else np.arange(n)
            tree = FastHistogramTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                mode="mse",
                rng=rng,
            )
            tree.fit(X_fit[s_idx], residuals[s_idx], bin_edges=bin_edges)
            # Newton-Raphson step scaling (approximate 2nd order scaling: r / (p*(1-p)) clipped)
            update = tree.predict(X_fit)
            F += self.learning_rate * 2.5 * update
            self.trees_.append(tree)
            p_now = expit(np.clip(F, -20.0, 20.0))
            bce = -np.mean(y_fit * np.log(p_now + 1e-12) + (1.0 - y_fit) * np.log(1.0 - p_now + 1e-12))
            self.train_loss_history_.append(float(bce))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.init_logit_, dtype=np.float64)
        scale = self.learning_rate * 2.5
        for tree in self.trees_:
            F += scale * tree.predict(X)
        p1 = expit(np.clip(F, -20.0, 20.0))
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# 6. Shallow MLP Neural Network (1 Hidden Layer ReLU + Adam) — Neural Networks
# ---------------------------------------------------------------------------
class MLPClassifierNP:
    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3, epochs: int = 20,
                 batch_size: int = 4096, weight_decay: float = 1e-4, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.history_ = {"train_loss": [], "val_loss": []}

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, self.hidden_dim)).astype(np.float64)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / self.hidden_dim), size=(self.hidden_dim, 1)).astype(np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)

        params = [self.W1, self.b1, self.W2, self.b2]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        y_col = y.astype(np.float64).reshape(-1, 1)
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                Xb = X[idx].astype(np.float64)
                yb = y_col[idx]
                bs = len(Xb)

                # Forward
                z1 = Xb @ self.W1 + self.b1
                h1 = np.maximum(0.0, z1)
                z2 = np.clip(h1 @ self.W2 + self.b2, -25.0, 25.0)
                p = expit(z2)

                bce = -np.mean(yb * np.log(p + 1e-12) + (1.0 - yb) * np.log(1.0 - p + 1e-12))
                epoch_loss += bce
                n_batches += 1

                # Backward
                dz2 = (p - yb) / bs
                gW2 = h1.T @ dz2 + self.weight_decay * self.W2
                gb2 = np.sum(dz2, axis=0)
                dh1 = dz2 @ self.W2.T
                dz1 = dh1 * (z1 > 0)
                gW1 = Xb.T @ dz1 + self.weight_decay * self.W1
                gb1 = np.sum(dz1, axis=0)

                grads = [gW1, gb1, gW2, gb2]
                step += 1
                for k in range(4):
                    m_adam[k] = beta1 * m_adam[k] + (1.0 - beta1) * grads[k]
                    v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * (grads[k] ** 2)
                    m_hat = m_adam[k] / (1.0 - beta1 ** step)
                    v_hat = v_adam[k] / (1.0 - beta2 ** step)
                    params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            self.history_["train_loss"].append(float(epoch_loss / max(n_batches, 1)))
            if X_val is not None and y_val is not None:
                pv = self.predict_proba(X_val)[:, 1]
                yv = y_val.astype(np.float64)
                v_bce = -np.mean(yv * np.log(pv + 1e-12) + (1.0 - yv) * np.log(1.0 - pv + 1e-12))
                self.history_["val_loss"].append(float(v_bce))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, X.astype(np.float64) @ self.W1 + self.b1)
        z2 = np.clip(h1 @ self.W2 + self.b2, -25.0, 25.0)
        p1 = expit(z2).ravel()
        return np.column_stack([1.0 - p1, p1])
