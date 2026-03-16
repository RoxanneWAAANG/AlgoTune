"""
AlgoTune Task: rl4rla_block_randomized_kaczmarz

export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

================================================
Corresponds to:
  Config:            DiscoverNLA/hp/mcgs_ucd/sketch_n_project/3.yml
  Target algorithm:  "sketch and project"
                     (Block Randomized Kaczmarz — sketch-and-project method)
  Reference (solve): gradient descent  [starting algorithm for this curriculum]
  System type:       MID_COND — chi-squared singular-value spectrum,
                     LOW-leverage left singular vectors (Gaussian QR),
                     Gaussian right singular vectors

Problem
-------
Find x* that minimises (1/2) ||A x - b||^2,  A in R^{m x n},  m >> n.

The target (Block Randomized Kaczmarz) update per iteration is:
    sample S_t rows uniformly at random (batch_size rows)
    x_{t+1} = x_t - lr * A_S^T (A_S A_S^T)^{-1} (A_S x_t - b_S)

Key parameters (from YAML):
  num-cols  : n  (problem size parameter)
  num-rows  : n * 200   (= 10 000 when n = 50)
  lev       : low    (Gaussian left singular vectors  → uniform leverage)
  vt-dis    : gauss  (Gaussian orthonormal Vt)
  prop-range: 1.0
  sampling-factor: 8,  batch-size: 100
  max-iterations:  10, learning-rate: 0.5
  reward-type:     log  (high_loss=0.1 → quality threshold used in is_solution)

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
    """Orthonormal matrix via Gaussian QR  (lev='low')."""
    Q = rng.standard_normal((num_rows, num_cols))
    Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
    Q = Q * np.sign(np.diag(R))
    return Q


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

@register_task("rl4rla_block_randomized_kaczmarz")
class Solution(Task):
    """
    Overdetermined least-squares on a MID_COND system with low-leverage rows
    (Gaussian left singular vectors → near-uniform statistical leverage scores).

    generate_problem replicates
      DiscoverNLA/_create_mid_condition_system with lev='low', vt_dis='gauss'.

    solve() implements gradient descent (the RL starting algorithm).

    is_solution() checks ||Ax-b|| / ||b|| < 0.1
    (DiscoverNLA reward-type: log, high_loss = 0.1).

    The target algorithm (Block Randomized Kaczmarz) does per iteration:
      S  = random subset of batch_size rows of A
      d  = A_S^T (A_S A_S^T)^{-1} (A_S x - b_S)
      x  = x - lr * d
    which matches the DiscoverNLA "sketch and project" update_loop:
      R1 = A_S A_S^T,  R1 = R1^{-1},  V1 = R1 V1,  V1 = A_S^T V1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        """
        Generate a MID_COND least-squares problem A x = b.

        :param n: Number of columns.  n=50 → paper dimensions.
        :param random_seed: Reproducibility seed.
        :return: dict with keys A, b, x_opt, num_rows, num_cols,
                 sampling_factor, batch_size, max_iterations, learning_rate.
        """
        rng = np.random.default_rng(random_seed)
        num_cols = n
        num_rows = n * 200
        prop_range = 1.0

        # MID_COND spectrum: chi-squared (df=1)
        spectrum = rng.standard_normal(num_cols) ** 2 + 1e-6   # (n,)

        # Left singular vectors: low-leverage (Gaussian QR)
        U = _orthonormal_gauss(rng, num_rows, num_cols)    # (m, n)

        # Right singular vectors: Gaussian orthonormal
        Vt = _orthonormal_gauss(rng, num_cols, num_cols)   # (n, n)

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
            "sampling_factor": 8,
            "batch_size": 100,
            "max_iterations": 10,
            "learning_rate": 0.5,
        }

    # ------------------------------------------------------------------
    def solve(self, problem: dict) -> dict:
        """
        Reference solver: exact least-squares (quality bar for AlgoTune).

        MID_COND matrices can have condition numbers ~10^4, so plain gradient
        descent (the RL starting algorithm) may not converge in a practical
        number of steps.  scipy.linalg.lstsq is used as the reference quality
        bar.  AlgoTuner should implement the TARGET algorithm
        ("sketch and project" / Block Randomized Kaczmarz) and beat gradient
        descent in speed.

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
        Verify candidate solution quality.

        Checks:
          1. Key "x" present, convertible to float64.
          2. Shape (num_cols,).
          3. No NaN / Inf.
          4. ||Ax - b|| / ||b|| < 0.1  (log-reward high_loss threshold).

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
