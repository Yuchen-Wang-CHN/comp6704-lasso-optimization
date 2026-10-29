"""Common objective, KKT checks, and primal-dual certificates."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import eigvalsh


def soft_threshold(x, threshold):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@dataclass
class LassoProblem:
    X: np.ndarray
    y: np.ndarray
    backend: str
    gram: np.ndarray | None
    cross: np.ndarray
    y_squared: float
    common_setup_sec: float

    @classmethod
    def from_data(cls, X, y, backend="gram"):
        X = np.asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or len(y) != len(X) or min(X.shape) < 1:
            raise ValueError("X must be nonempty (n,d), y must have shape (n,)")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must be finite")
        if backend not in ("gram", "direct"):
            raise ValueError("backend must be gram or direct")
        started = perf_counter()
        cross = X.T @ y / len(y)
        gram = X.T @ X / len(y) if backend == "gram" else None
        return cls(X, y, backend, gram, cross, float(y @ y / len(y)), perf_counter() - started)

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def d(self):
        return self.X.shape[1]

    @property
    def lambda_max(self):
        return float(np.max(np.abs(self.cross)))

    def gradient(self, w):
        if self.gram is not None:
            return self.gram @ w - self.cross
        return self.X.T @ (self.X @ w - self.y) / self.n

    def lipschitz(self):
        # An exact dense spectral calculation, charged to each PG solver's setup.
        if self.gram is not None:
            value = float(eigvalsh(self.gram, subset_by_index=[self.d - 1, self.d - 1])[0])
        else:
            value = float(np.linalg.svd(self.X, compute_uv=False)[0] ** 2 / self.n)
        return max(value * (1 + 1e-12), np.finfo(float).eps)

    def certificate(self, w, lam, direct=False):
        """Return primal, feasible dual, dual gap, relative gap, and KKT infinity norm.

        theta = scale * (y-Xw)/n, with ||X.T theta||_inf <= lam.
        Gram certificates avoid repeated data passes. Direct certificates are used
        for final verification to detect cancellation in the quadratic expression.
        """
        if direct or self.gram is None:
            residual = self.y - self.X @ w
            corr = self.X.T @ residual / self.n
            r2 = float(residual @ residual / self.n)
            yr = float(self.y @ residual / self.n)
        else:
            gw = self.gram @ w
            corr = self.cross - gw
            r2 = float(self.y_squared - 2 * self.cross @ w + w @ gw)
            yr = float(self.y_squared - self.cross @ w)
            floor = 1e-10 * max(1.0, self.y_squared)
            if r2 < -floor:
                raise FloatingPointError("Invalid Gram residual; use the direct backend")
            r2 = max(0.0, r2)  # Only roundoff-sized negative squared norms.
        corr_norm = float(np.max(np.abs(corr)))
        # Tiny inward margin protects the dual constraint against rounding.
        scale = min(1.0, lam / corr_norm) * (1 - 1e-14) if corr_norm else 1.0
        primal = 0.5 * r2 + lam * float(np.abs(w).sum())
        dual = scale * yr - 0.5 * scale * scale * r2
        gap = primal - dual
        if gap < -1e-10 * max(1.0, primal, self.y_squared):
            raise FloatingPointError("Negative dual gap exceeds rounding tolerance")
        gap = max(0.0, gap)
        active = w != 0
        violation = np.maximum(np.abs(corr) - lam, 0.0)
        violation[active] = np.abs(corr[active] - lam * np.sign(w[active]))
        return {"objective": primal, "dual": dual, "dual_gap": gap,
                "relative_gap": gap / max(1.0, abs(primal)),
                "kkt_inf": float(np.max(violation))}
