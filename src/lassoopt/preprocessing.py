"""Train-only centering/scaling, with an unpenalized intercept."""

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    target_mean: float

    @classmethod
    def fit(cls, X, y):
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=0)
        # Constant columns become zeros after centering, without division by zero.
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean, scale, float(np.mean(y)))

    def transform(self, X, y=None):
        transformed = np.asarray((X - self.mean) / self.scale, dtype=np.float64, order="C")
        return transformed if y is None else (transformed, np.asarray(y - self.target_mean, dtype=np.float64))

    def original_parameters(self, w):
        coefficients = w / self.scale
        return coefficients, self.target_mean - float(self.mean @ coefficients)


def split_and_standardize(X, y, seed):
    indices = np.arange(len(y))
    train, holdout = train_test_split(indices, test_size=0.30, random_state=seed)
    valid, test = train_test_split(holdout, test_size=0.50, random_state=seed)
    scaler = Standardizer.fit(X[train], y[train])
    return {"train": scaler.transform(X[train], y[train]),
            "valid": scaler.transform(X[valid], y[valid]),
            "test": scaler.transform(X[test], y[test]),
            "indices": {"train": train, "valid": valid, "test": test}, "scaler": scaler}
