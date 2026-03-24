import logging

import numpy as np
import scipy.linalg as la
from AlgoTuneTasks.base import register_task, Task


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


@register_task("rl4rla_logistic")
class Solution(Task):
    """Logistic regression, m = 100 n."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        rng = np.random.default_rng(random_seed)
        m = n * 100
        sep = 5.0

        w_true = rng.standard_normal(n)
        w_true /= np.linalg.norm(w_true)

        A = rng.standard_normal((m, n))
        noise = rng.standard_normal(m) / sep
        y = np.where(A @ w_true + noise > 0, 1.0, 0.0)

        return {"A": A, "y": y}

    def solve(self, problem: dict) -> dict:
        A, y = problem["A"], problem["y"]
        n = A.shape[1]
        x = np.zeros(n)
        for _ in range(50):
            p = _sigmoid(A @ x)
            g = A.T @ (p - y) / len(y)
            W = p * (1.0 - p)
            H = (A.T * W) @ A / len(y) + 1e-8 * np.eye(n)
            x -= la.solve(H, g)
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
        A, y = problem["A"], problem["y"]
        if x.shape != (A.shape[1],):
            logging.error(f"Wrong shape: expected ({A.shape[1]},), got {x.shape}.")
            return False
        if not np.isfinite(x).all():
            logging.error("Solution contains NaN or Inf.")
            return False
        p = _sigmoid(A @ x)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if loss >= 0.5:
            logging.error(f"Cross-entropy {loss:.4f} >= 0.5.")
            return False
        return True
