"""
AlgoTune Task: rl4rla_ls

Overdetermined least-squares problem
=====================================

Given A ∈ R^{m×n} with m >> n, find x* that minimises

    (1/2) ||A x - b||²

The matrix A is constructed with a chi-squared singular-value spectrum
(condition number ~10⁴) and near-uniform statistical leverage scores.
The right-hand side b lies in the column space of A.

Input:  A  (m × n  float64 array)
        b  (m-vector float64)

Output: x  (n-vector float64)

Acceptance criterion:  ||A x - b|| / ||b|| < 0.1
"""

import logging

import numpy as np
import scipy.linalg as la
from AlgoTuneTasks.base import register_task, Task


# ---------------------------------------------------------------------------
# Matrix-generation helpers (internal)
# ---------------------------------------------------------------------------

def _orthonormal_gauss(rng, num_rows, num_cols):
    """Orthonormal matrix via QR of a Gaussian matrix."""
    G = rng.standard_normal((num_rows, num_cols))
    Q, R = la.qr(G, overwrite_a=True, pivoting=False, mode="economic")
    return Q * np.sign(np.diag(R))


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@register_task("rl4rla_ls")
class Solution(Task):
    """
    Overdetermined least-squares with a moderately ill-conditioned matrix
    (chi-squared singular-value spectrum, m = 200 n).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        """
        Generate an overdetermined least-squares problem.

        :param n:           Number of columns (problem size).
        :param random_seed: Reproducibility seed.
        :return: dict with keys:
                   A         — (m, n) float64 array
                   b         — (m,)  float64 array
                   num_rows  — m
                   num_cols  — n
        """
        rng = np.random.default_rng(random_seed)
        num_cols = n
        num_rows = n * 200

        # Singular values: chi-squared (df=1) distribution
        sigma = rng.standard_normal(num_cols) ** 2 + 1e-6   # (n,)

        # Left and right singular vectors (Gaussian QR → uniform leverage)
        U  = _orthonormal_gauss(rng, num_rows, num_cols)   # (m, n)
        Vt = _orthonormal_gauss(rng, num_cols, num_cols)   # (n, n)

        # A = U diag(sigma) Vt
        A = (U * sigma) @ Vt

        # Consistent RHS: b in the column space of A
        b0      = rng.standard_normal(num_rows)
        b_proj  = U @ (U.T @ b0)
        b_orth  = b0 - b_proj
        mu      = float(np.mean(sigma))
        b_proj  = b_proj  * (mu / (la.norm(b_proj)  + 1e-12))
        b_orth  = b_orth  * (mu / (la.norm(b_orth)  + 1e-12))
        b       = b_proj + b_orth          # prop_range = 1.0 → b = b_proj

        return {
            "A":        A,
            "b":        b,
            # "num_rows": num_rows,
            # "num_cols": num_cols,
        }

    # ------------------------------------------------------------------
    def solve(self, problem: dict) -> dict:
        """
        Reference solver: direct least-squares via scipy.linalg.lstsq.

        :param problem: dict from generate_problem.
        :return: dict with key "x".
        """
        A = problem["A"]
        b = problem["b"]
        x, _, _, _ = la.lstsq(A, b)
        return {"x": x}

    # ------------------------------------------------------------------
    def is_solution(self, problem: dict, solution: dict) -> bool:
        """
        Accept a candidate solution if ||Ax - b|| / ||b|| < 0.01.

        :param problem:  dict from generate_problem.
        :param solution: dict with key "x".
        :return: True if the solution is accurate enough.
        """
        x = solution.get("x")
        if x is None:
            logging.error("Solution missing key 'x'.")
            return False
        try:
            x = np.asarray(x, dtype=float)
        except Exception:
            logging.error("Cannot convert solution['x'] to numpy array.")
            return False
        n = problem["num_cols"]
        if x.shape != (n,):
            logging.error(f"Wrong shape: expected ({n},), got {x.shape}.")
            return False
        if not np.isfinite(x).all():
            logging.error("Solution contains NaN or Inf.")
            return False

        A, b = problem["A"], problem["b"]
        rel_res = la.norm(A @ x - b) / (la.norm(b) + 1e-12)
        if rel_res >= 0.1:
            logging.error(f"Relative residual {rel_res:.4e} >= 0.01.")
            return False
        return True
