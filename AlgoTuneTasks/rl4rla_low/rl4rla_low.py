import logging

import numpy as np
import scipy.linalg as la
from AlgoTuneTasks.base import register_task, Task


def _orthonormal_gauss(rng, num_rows, num_cols):
    G = rng.standard_normal((num_rows, num_cols))
    Q, R = la.qr(G, overwrite_a=True, pivoting=False, mode="economic")
    return Q * np.sign(np.diag(R))


def _orthonormal_mvt(rng, num_rows, num_cols, df=0.1):
    G = rng.standard_normal((num_rows, num_cols))
    c = rng.chisquare(df, size=(num_rows, 1))
    G = G * np.sqrt(1.0 / c)
    Q, R = la.qr(G, overwrite_a=True, pivoting=False, mode="economic")
    return Q * np.sign(np.diag(R))


@register_task("rl4rla_low")
class Solution(Task):
    """Overdetermined least-squares, m = 200 n."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        rng = np.random.default_rng(random_seed)
        m = n * 200

        # Well-conditioned spectrum: linspace [1, 2]
        sigma = np.linspace(1.0, 2.0, n) + 1e-6
        U  = _orthonormal_mvt(rng, m, n)
        Vt = _orthonormal_gauss(rng, n, n)
        A  = (U * sigma) @ Vt

        b0 = rng.standard_normal(m)
        b  = U @ (U.T @ b0)
        b *= np.mean(sigma) / (la.norm(b) + 1e-12)

        return {"A": A, "b": b}

    def solve(self, problem: dict) -> dict:
        x, _, _, _ = la.lstsq(problem["A"], problem["b"])
        return {"x": x}

    def is_solution(self, problem: dict, solution: dict) -> bool:
        x = solution.get("x")
        if x is None:
            logging.error("Missing key 'x'.")
            return False
        try:
            x = np.asarray(x, dtype=float)
        except Exception:
            logging.error("Cannot convert 'x' to array.")
            return False
        A, b = problem["A"], problem["b"]
        if x.shape != (A.shape[1],):
            logging.error(f"Wrong shape: expected ({A.shape[1]},), got {x.shape}.")
            return False
        if not np.isfinite(x).all():
            logging.error("Solution contains NaN or Inf.")
            return False
        rel_res = la.norm(A @ x - b) / (la.norm(b) + 1e-12)
        if rel_res >= 0.01:
            logging.error(f"Relative residual {rel_res:.4e} >= 0.01.")
            return False
        return True
