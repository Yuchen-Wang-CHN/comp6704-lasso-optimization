import numpy as np
import pytest
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits
from lassoopt import LassoProblem, ista, fista, admm, soft_threshold
from lassoopt.data import synthetic


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


SOLVERS = [ista, fista, lambda *a, **k: fista(*a, restart=True, **k), admm]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("backend", ["gram", "direct"])
def test_orthogonal_closed_form(solver, backend):
    n = 8
    X = np.sqrt(n) * np.eye(n)
    y = np.array([-3, -1, -0.2, 0, 0.2, 0.5, 1, 3.0])
    problem = LassoProblem.from_data(X, y, backend)
    lam = 0.15
    expected = soft_threshold(X.T @ y / n, lam)
    result = solver(problem, lam, tol=1e-10, check_every=1)
    assert result.converged
    np.testing.assert_allclose(result.coefficients, expected, atol=2e-8)
    assert problem.certificate(result.coefficients, lam, direct=True)["relative_gap"] < 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_matches_trusted_reference_and_dual_bound(solver, rank_deficient):
    X, y, _ = synthetic(n=120, d=30, correlation=0.85, seed=4)
    X -= X.mean(axis=0)
    y -= y.mean()
    if rank_deficient:
        X[:, -1] = X[:, 0]
    problem = LassoProblem.from_data(X, y)
    lam = 0.03 * problem.lambda_max
    ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-13, max_iter=100000).fit(X, y)
    result = solver(problem, lam, tol=1e-9, check_every=5, max_iter=30000)
    assert result.converged
    cert = problem.certificate(result.coefficients, lam, direct=True)
    f_ref = problem.certificate(ref.coef_, lam, direct=True)["objective"]
    assert cert["dual"] <= f_ref + 1e-9
    assert abs(cert["objective"] - f_ref) <= 2e-8 * max(1, f_ref)
    np.testing.assert_allclose(X @ result.coefficients, X @ ref.coef_, atol=5e-4)
    if solver is ista:
        assert np.all(np.diff([h["objective"] for h in result.history]) <= 1e-10)
    if solver is admm:
        final = result.history[-1]
        assert final["primal_residual"] <= final["eps_primal"]
        assert final["dual_residual"] <= final["eps_dual"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_lambda_max_and_warm_start(solver):
    X, y, _ = synthetic(n=80, d=15, seed=12)
    problem = LassoProblem.from_data(X, y)
    result = solver(problem, 1.1 * problem.lambda_max, w0=np.ones(15), tol=1e-9, check_every=1)
    assert result.converged
    assert np.max(np.abs(result.coefficients)) < 1e-6
    assert solver(problem, problem.lambda_max).iterations == 0


def test_gram_and_direct_certificate_and_gradient():
    X, y, _ = synthetic(n=160, d=35, seed=13)
    gram = LassoProblem.from_data(X, y)
    direct = LassoProblem.from_data(X, y, "direct")
    w = np.random.default_rng(8).normal(size=35)
    np.testing.assert_allclose(gram.gradient(w), direct.gradient(w), atol=1e-12)
    for key, value in gram.certificate(w, 0.1).items():
        assert value == pytest.approx(direct.certificate(w, 0.1)[key], abs=1e-10)
    eps = 1e-6
    j = 7
    wp, wm = w.copy(), w.copy()
    wp[j] += eps
    wm[j] -= eps
    numeric = ((np.linalg.norm(X @ wp - y)**2 - np.linalg.norm(X @ wm - y)**2) / (4 * eps * len(y)))
    assert gram.gradient(w)[j] == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_iteration_limit_is_not_success(solver):
    X, y, _ = synthetic(n=100, d=30, correlation=0.95, seed=8)
    result = solver(LassoProblem.from_data(X, y), 0.01, max_iter=1, tol=1e-14)
    assert not result.converged
    assert result.iterations == 1


@pytest.mark.parametrize("lam", [-1, 0, float("nan"), float("inf")])
def test_invalid_regularization(lam):
    with pytest.raises(ValueError):
        ista(LassoProblem.from_data(np.eye(2), np.ones(2)), lam)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        LassoProblem.from_data(np.ones((3, 2)), np.ones(2))
    with pytest.raises(ValueError):
        LassoProblem.from_data(np.array([[np.nan]]), np.ones(1))
    with pytest.raises(ValueError):
        admm(LassoProblem.from_data(np.eye(2), np.ones(2)), 0.1, rho=0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_zero_design(solver):
    problem = LassoProblem.from_data(np.zeros((10, 3)), np.arange(10.0))
    result = solver(problem, 0.2, w0=np.ones(3), tol=1e-10, check_every=1)
    assert result.converged
    np.testing.assert_allclose(result.coefficients, 0, atol=1e-9)
