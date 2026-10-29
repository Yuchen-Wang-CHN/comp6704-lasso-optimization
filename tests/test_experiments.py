import numpy as np
from threadpoolctl import threadpool_limits
from experiments.run import reference_solution
from lassoopt import LassoProblem
from lassoopt.data import synthetic


def test_reference_gram_scaling():
    X, y, _ = synthetic(n=150, d=20, seed=3)
    X -= X.mean(axis=0)
    y -= y.mean()
    with threadpool_limits(limits=1):
        problem = LassoProblem.from_data(X, y)
        coefficients, cert, _ = reference_solution(problem, 0.1, 1e-12)
    assert coefficients.shape == (20,)
    assert cert["relative_gap"] < 1e-9
    assert cert["kkt_inf"] < 1e-8
