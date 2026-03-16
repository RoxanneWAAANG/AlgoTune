"""
AlgoTune Task: rl4rla_subsampled_least_squares_gd
==================================================
Corresponds to:
  Config:            DiscoverNLA/hp/mcgs_ucd/sketch_n_solve_low_cond/2.yml
  Target algorithm:  "sketch and solve - low cond"
                     (Subsampled Least-Squares GD with leverage-score row selection)
  Reference (solve): gradient descent  [starting algorithm for this curriculum]
  System type:       LOW_COND — linearly spaced singular values in [1, 2],
                     condition number = 2, high-leverage left singular vectors
                     (multivariate-t), Gaussian right singular vectors

Problem
-------
Find x* that minimises (1/2) ||A x - b||^2,  A in R^{m x n},  m >> n.

The target algorithm ("sketch and solve - low cond") does ONCE:
    compute leverage scores  lev_i = ||A_i R^{-1}||^2  (proxied by row norms^2)
    sample S_0 subset of batch_size rows proportional to lev_i
then every iteration:
    residual  = A_S x - b_S
    gradient  = A_S^T residual
    x_{t+1}  = x_t - lr * gradient

Key parameters (from YAML):
  num-cols  : n  (problem size parameter)
  num-rows  : n * 200   (= 10 000 when n = 50)
  lev       : high   (heavy-tailed left singular vectors)
  vt-dis    : gauss  (Gaussian right singular vectors)
  prop-range: 1.0
  condition-number: 2.0  (LOW_COND)
  sampling-factor: 8,  batch-size: 100
  max-iterations:  10, learning-rate: 0.2
  reward-type:     log  (high_loss=0.1)

Quality threshold: ||Ax - b|| / ||b|| < 0.1
"""

import logging

import numpy as np
import scipy.linalg as la

from AlgoTuneTasks.base import register_task, Task


# ---------------------------------------------------------------------------
# Matrix-generation helpers
# ---------------------------------------------------------------------------

def _orthonormal_gauss(rng, num_rows, num_cols):
    """Orthonormal matrix via Gaussian QR  (vt_dis='gauss')."""
    Q = rng.standard_normal((num_rows, num_cols))
    Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
    Q = Q * np.sign(np.diag(R))
    return Q


def _orthonormal_mvt(rng, num_rows, num_cols):
    """Orthonormal matrix via multivariate-t QR  (lev='high', df=0.1)."""
    if num_rows < num_cols:
        return _orthonormal_mvt(rng, num_cols, num_rows).T
    Q = rng.standard_normal((num_rows, num_cols))
    c = rng.chisquare(df=0.1, size=(num_rows, 1))
    Q = Q * np.sqrt(1.0 / c)
    Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
    Q = Q * np.sign(np.diag(R))
    return Q


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

@register_task("rl4rla_subsampled_least_squares_gd")
class Solution(Task):
    """
    Overdetermined least-squares on a LOW_COND system (condition number = 2)
    with high-leverage rows (heavy-tailed left singular vectors).

    generate_problem replicates
      DiscoverNLA/_create_low_condition_system with lev='high', vt_dis='gauss'.

    solve() implements gradient descent (the RL starting algorithm).
    For LOW_COND the YAML learning rate lr=0.2 is safe (sigma_max^2 ≈ 4 + ε,
    lr * sigma_max^2 ≈ 0.8 < 2), so convergence is guaranteed.

    is_solution() checks ||Ax-b|| / ||b|| < 0.1.

    The target algorithm ("sketch and solve - low cond") fixes a leverage-score
    subsample at setup time and runs gradient descent on that sub-problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        """
        Generate a LOW_COND least-squares problem A x = b.

        :param n: Number of columns.  n=50 → paper dimensions.
        :param random_seed: Reproducibility seed.
        :return: dict with keys A, b, x_opt, num_rows, num_cols,
                 condition_number, sampling_factor, batch_size,
                 max_iterations, learning_rate.
        """
        rng = np.random.default_rng(random_seed)
        num_cols = n
        num_rows = n * 200
        prop_range = 1.0
        condition_number = 2.0

        # LOW_COND spectrum: linearly spaced in [1, 2] (condition number = 2)
        min_ev = 1.0
        max_ev = float(condition_number)   # = 2.0
        scaling = np.linspace(0.0, 1.0, num_cols)
        spectrum = min_ev + scaling * (max_ev - min_ev) + 1e-6   # (n,)

        # Left singular vectors: high-leverage (multivariate-t, df=0.1)
        U = _orthonormal_mvt(rng, num_rows, num_cols)    # (m, n)

        # Right singular vectors: Gaussian orthonormal
        Vt = _orthonormal_gauss(rng, num_cols, num_cols)  # (n, n)

        # A = U diag(spectrum) Vt
        A = (U * spectrum) @ Vt   # (m, n)

        # Consistent RHS: b in column space of A
        b0 = rng.standard_normal(num_rows)
        b_range  = U @ (U.T @ b0)
        b_orthog = b0 - b_range
        mean_spec = float(np.mean(spectrum))
        b_range  = b_range  * (mean_spec / (la.norm(b_range)  + 1e-12))
        b_orthog = b_orthog * (mean_spec / (la.norm(b_orthog) + 1e-12))
        b = prop_range * b_range + (1.0 - prop_range) * b_orthog

        # Optimal solution: x* = Vt^T diag(1/spectrum) U^T b
        x_opt = (Vt.T / spectrum) @ (U.T @ b)

        return {
            "A": A,
            "b": b,
            "x_opt": x_opt,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "condition_number": condition_number,
            "sampling_factor": 8,
            "batch_size": 100,
            "max_iterations": 10,
            "learning_rate": 0.2,
        }

    # ------------------------------------------------------------------
    def solve(self, problem: dict) -> dict:
        """
        Reference solver: gradient descent  (starting algorithm in the curriculum).

        For LOW_COND (condition number = 2) the YAML learning rate lr=0.2 is
        safe (singular values in [1, 2], so sigma_max^2 ≈ 4 and
        lr * sigma_max^2 = 0.8 < 2).  Convergence factor per step ≤ 0.8^t.

        x_{t+1} = x_t - lr * A^T (A x_t - b)

        :param problem: dict from generate_problem.
        :return: dict with key "x".
        """
        A = problem["A"]
        b = problem["b"]
        lr = problem["learning_rate"]   # 0.2 — safe for LOW_COND

        x = np.zeros(problem["num_cols"])
        for _ in range(500):
            grad = A.T @ (A @ x - b)
            x = x - lr * grad

        return {"x": x}

    # ------------------------------------------------------------------
    def is_solution(self, problem: dict, solution: dict) -> bool:
        """
        Verify candidate solution quality.

        Checks:
          1. Key "x" present, convertible to float64.
          2. Shape (num_cols,).
          3. No NaN / Inf.
          4. ||Ax - b|| / ||b|| < 0.1.

        :param problem:  dict from generate_problem.
        :param solution: dict with key "x".
        :return: True if all checks pass.
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
        if x.shape != (problem["num_cols"],):
            logging.error(
                f"Wrong shape: expected ({problem['num_cols']},), got {x.shape}."
            )
            return False
        if not np.isfinite(x).all():
            logging.error("Solution contains NaN or Inf.")
            return False

        A, b = problem["A"], problem["b"]
        rel_res = la.norm(A @ x - b) / (la.norm(b) + 1e-12)
        if rel_res >= 0.01:
            logging.error(
                f"Relative residual {rel_res:.4e} >= threshold 0.01."
            )
            return False
        return True
