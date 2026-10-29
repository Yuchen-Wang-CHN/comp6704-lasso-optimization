"""Original NumPy/SciPy implementations; no library estimator in these solvers."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .problem import LassoProblem, soft_threshold


@dataclass
class SolverResult:
    coefficients: np.ndarray
    history: list[dict]
    converged: bool
    iterations: int
    common_setup_sec: float
    solver_setup_sec: float
    iteration_sec: float
    total_sec: float
    restarts: int = 0


def _validate(problem, lam, tol, max_iter, check_every, w0):
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive; lambda=0 is outside this LASSO benchmark")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(max_iter, int) or max_iter < 1 or not isinstance(check_every, int) or check_every < 1:
        raise ValueError("max_iter and check_every must be positive integers")
    w = np.zeros(problem.d) if w0 is None else np.array(w0, dtype=float, copy=True)
    if w.shape != (problem.d,) or not np.isfinite(w).all():
        raise ValueError("w0 must be a finite vector with shape (d,)")
    return w


def _pg(problem, lam, tol, max_iter, check_every, w0, accelerated, restart):
    w = _validate(problem, lam, tol, max_iter, check_every, w0)
    setup_start = perf_counter()
    L = problem.lipschitz()
    setup_sec = perf_counter() - setup_start
    start = perf_counter()
    offset = problem.common_setup_sec + setup_sec
    history = []
    y = w.copy()
    momentum = 1.0
    restarts = 0
    converged = False
    for iteration in range(max_iter + 1):
        if iteration % check_every == 0 or iteration == max_iter:
            cert = problem.certificate(w, lam)
            history.append({"iteration": iteration, "elapsed_sec": offset + perf_counter() - start,
                            **cert, "restarts": restarts})
            if cert["relative_gap"] <= tol:
                converged = True
                break
        if iteration == max_iter:
            break
        new = soft_threshold(y - problem.gradient(y) / L, lam / L)
        if accelerated:
            # Composite gradient restart: (y_k - w_{k+1}) . (w_{k+1} - w_k) > 0.
            # The accepted proximal point is retained; only momentum is reset.
            if restart and float((y - new) @ (new - w)) > 0:
                new_momentum = 1.0
                y = new.copy()
                restarts += 1
            else:
                new_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
                y = new + (momentum - 1) / new_momentum * (new - w)
            momentum = new_momentum
        else:
            y = new
        w = new
    elapsed = perf_counter() - start
    return SolverResult(w, history, converged, iteration, problem.common_setup_sec,
                        setup_sec, elapsed, offset + elapsed, restarts)


def ista(problem: LassoProblem, lam, *, tol=1e-6, max_iter=20000, check_every=5, w0=None):
    return _pg(problem, lam, tol, max_iter, check_every, w0, False, False)


def fista(problem: LassoProblem, lam, *, restart=False, tol=1e-6, max_iter=20000, check_every=5, w0=None):
    return _pg(problem, lam, tol, max_iter, check_every, w0, True, restart)


def admm(problem: LassoProblem, lam, *, rho=1.0, tol=1e-6, max_iter=20000,
         check_every=5, abs_tol=1e-6, rel_tol=1e-5, w0=None):
    z = _validate(problem, lam, tol, max_iter, check_every, w0)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    if not all(np.isfinite(v) and v > 0 for v in (abs_tol, rel_tol)):
        raise ValueError("ADMM residual tolerances must be finite and positive")
    setup_start = perf_counter()
    gram = problem.gram if problem.gram is not None else problem.X.T @ problem.X / problem.n
    factor = cho_factor(gram + rho * np.eye(problem.d), lower=True, check_finite=False)
    setup_sec = perf_counter() - setup_start
    start = perf_counter()
    offset = problem.common_setup_sec + setup_sec
    w, u = z.copy(), np.zeros(problem.d)
    primal_residual = dual_residual = 0.0
    history = []
    converged = False
    for iteration in range(max_iter + 1):
        if iteration % check_every == 0 or iteration == max_iter:
            eps_primal = np.sqrt(problem.d) * abs_tol + rel_tol * max(np.linalg.norm(w), np.linalg.norm(z))
            eps_dual = np.sqrt(problem.d) * abs_tol + rel_tol * np.linalg.norm(rho * u)
            cert = problem.certificate(z, lam)
            history.append({"iteration": iteration, "elapsed_sec": offset + perf_counter() - start,
                            **cert, "primal_residual": primal_residual, "dual_residual": dual_residual,
                            "eps_primal": eps_primal, "eps_dual": eps_dual, "restarts": 0})
            if cert["relative_gap"] <= tol and primal_residual <= eps_primal and dual_residual <= eps_dual:
                converged = True
                break
        if iteration == max_iter:
            break
        w = cho_solve(factor, problem.cross + rho * (z - u), check_finite=False)
        previous_z = z
        z = soft_threshold(w + u, lam / rho)
        u += w - z
        primal_residual = float(np.linalg.norm(w - z))
        dual_residual = float(rho * np.linalg.norm(z - previous_z))
    elapsed = perf_counter() - start
    # Report z: it is sparse, and its unsplit LASSO objective is comparable to PG.
    return SolverResult(z, history, converged, iteration, problem.common_setup_sec,
                        setup_sec, elapsed, offset + elapsed)
