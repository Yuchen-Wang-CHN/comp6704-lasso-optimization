import numpy as np
from lassoopt.preprocessing import Standardizer, split_and_standardize


def test_train_only_statistics_disjoint_splits_and_intercept():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(200, 5)) + 4
    X[:, -1] = 7
    y = 3 + X @ np.arange(5) + rng.normal(size=200)
    split = split_and_standardize(X, y, 7)
    train, valid, test = (split["indices"][key] for key in ("train", "valid", "test"))
    assert (len(train), len(valid), len(test)) == (140, 30, 30)
    assert len(set(train) & set(valid)) == len(set(train) & set(test)) == len(set(valid) & set(test)) == 0
    assert len(set(np.concatenate([train, valid, test]))) == len(y)
    np.testing.assert_allclose(split["scaler"].mean, X[train].mean(axis=0))
    np.testing.assert_allclose(split["train"][0].mean(axis=0), 0, atol=1e-14)
    assert np.isfinite(split["train"][0]).all()
    w = np.arange(5.0)
    coef, intercept = split["scaler"].original_parameters(w)
    np.testing.assert_allclose(X @ coef + intercept,
                               split["scaler"].transform(X) @ w + split["scaler"].target_mean)
    X_changed = X.copy()
    X_changed[valid] += 1000
    np.testing.assert_array_equal(Standardizer.fit(X[train], y[train]).mean,
                                   Standardizer.fit(X_changed[train], y[train]).mean)
