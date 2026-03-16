"""
AlgoTune Task: rl4rla_preconditioned_weighted_sgd
=================================================
Corresponds to:
  Config:            DiscoverNLA/hp/mcgs_ucd/leverage_score_subsampling/6.yml
  Target algorithm:  "leverage score subsampling"
                     (sketch-and-precondition SGD with leverage-score row sampling)
  Reference (solve): gradient descent  [starting algorithm for this curriculum]
  System type:       MID_COND — chi-squared singular-value spectrum,
                     high-leverage left singular vectors (multivariate-t),
                     Gaussian right singular vectors

Problem
-------
Find x* that minimises (1/2) ||A x - b||^2,  A in R^{m x n},  m >> n.

Key parameters (from YAML):
  num-cols  : n  (problem size parameter)
  num-rows  : n * 200   (= 10 000 when n = 50)
  lev       : high   (heavy-tailed U via chi-square df=0.1 QR)
  vt-dis    : gauss  (Gaussian orthonormal Vt)
  prop-range: 1.0
  sampling-factor: 8,  batch-size: 100
  max-iterations:  10, learning-rate: 0.5
  reward-type:     log  (high_loss=0.1 → quality threshold used in is_solution)

Quality threshold: ||Ax - b|| / ||b|| < 0.1
  (matches the 'high_loss' boundary of the log-type reward in DiscoverNLA)
"""

import logging

import numpy as np
import scipy.linalg as la
import scipy.sparse as spar

from AlgoTuneTasks.base import register_task, Task


# ---------------------------------------------------------------------------
# Matrix-generation helpers — exact replicas of LinearSystemFactory internals
# ---------------------------------------------------------------------------


def _orthonormal_gauss(rng, num_rows, num_cols):
    """Orthonormal matrix via Gaussian QR  (lev='low' variant)."""
    Q = rng.standard_normal((num_rows, num_cols))
    Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
    Q = Q * np.sign(np.diag(R))
    return Q


def _orthonormal_mvt(rng, num_rows, num_cols):
    """Orthonormal matrix via multivariate-t QR  (lev='high', df=0.1)."""
    if num_rows < num_cols:
        return _orthonormal_mvt(rng, num_cols, num_rows).T
    Q = rng.standard_normal((num_rows, num_cols))
    c = rng.chisquare(df=0.1, size=(num_rows, 1))  # heavy-tailed scaling
    Q = Q * np.sqrt(1.0 / c)
    Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
    Q = Q * np.sign(np.diag(R))
    return Q


def _create_sjlt(rng, num_sketch_rows, num_cols, vec_nnz=8):
    """Sparse Johnson-Lindenstrauss Transform  (SJLT, used by target algorithm)."""
    if num_cols >= num_sketch_rows:
        bad_size = num_sketch_rows < vec_nnz
        row_vecs = []
        for _ in range(num_cols):
            rows = rng.choice(num_sketch_rows, vec_nnz, replace=bad_size)
            row_vecs.append(rows)
        rows_idx = np.concatenate(row_vecs)
        cols_idx = np.repeat(np.arange(num_cols), vec_nnz)
        vals = np.where(rng.random(num_cols * vec_nnz) <= 0.5, -1.0, 1.0)
        vals /= np.sqrt(vec_nnz)
        S = spar.coo_matrix((vals, (rows_idx, cols_idx)), shape=(num_sketch_rows, num_cols)).tocsc()
    else:
        S = _create_sjlt(rng, num_cols, num_sketch_rows, vec_nnz)
        S = S.T.tocsr()
    return S


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


@register_task("rl4rla_preconditioned_weighted_sgd")
class Solution(Task):
    """
    Overdetermined least-squares on a MID_COND system with high-leverage rows.

    generate_problem replicates the matrix generation from
      DiscoverNLA/infrastructure/linear_algebra/data_generation.py
      (method: _create_mid_condition_system + _create_system_from_spectrum)

    solve() implements gradient descent (the RL starting algorithm),
    using a stable step size so it converges on the chi-squared spectrum.

    is_solution() checks ||Ax-b|| / ||b|| < 0.1
    (DiscoverNLA reward-type: log, high_loss = 0.1).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        """
        Generate a MID_COND least-squares problem A x = b.

        :param n: Number of columns (features).  n=50 replicates the paper.
        :param random_seed: Seed for reproducibility.
        :return: dict with keys
                   A            (num_rows x num_cols, float64)
                   b            (num_rows,)
                   x_opt        (num_cols,)  — pseudoinverse solution
                   num_rows, num_cols
                   sampling_factor, batch_size, max_iterations, learning_rate
        """
        rng = np.random.default_rng(random_seed)
        num_cols = n
        num_rows = n * 200  # m = 200 n  →  10 000 when n = 50
        prop_range = 1.0

        # MID_COND spectrum: chi-squared (df=1) = squared standard normal
        spectrum = rng.standard_normal(num_cols) ** 2 + 1e-6  # (n,)  positive

        # Left singular vectors: high-leverage (multivariate-t, df=0.1)
        U = _orthonormal_mvt(rng, num_rows, num_cols)  # (m, n)

        # Right singular vectors: Gaussian orthonormal
        Vt = _orthonormal_gauss(rng, num_cols, num_cols)  # (n, n)

        # A = U diag(spectrum) Vt  — singular values = spectrum
        A = (U * spectrum) @ Vt  # (m, n)

        # RHS: b in the column space of A  (consistent system, prop_range=1)
        b0 = rng.standard_normal(num_rows)
        b_range = U @ (U.T @ b0)
        b_orthog = b0 - b_range
        mean_spec = float(np.mean(spectrum))
        b_range = b_range * (mean_spec / (la.norm(b_range) + 1e-12))
        b_orthog = b_orthog * (mean_spec / (la.norm(b_orthog) + 1e-12))
        b = prop_range * b_range + (1.0 - prop_range) * b_orthog  # (m,)

        # Optimal solution: x* = A^+ b = Vt^T diag(1/spectrum) U^T b
        x_opt = (Vt.T / spectrum) @ (U.T @ b)  # (n,)

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

        The RL curriculum starts from plain gradient descent, which diverges on
        ill-conditioned MID_COND systems (condition numbers ~10^4).  For AlgoTune
        we use scipy.linalg.lstsq as the reference quality bar — it returns the
        minimum-norm least-squares solution that the target algorithm should match.
        AlgoTuner should implement the TARGET algorithm
        ("leverage score subsampling") and beat gradient descent in speed.

        :param problem: dict from generate_problem.
        :return: dict with key "x" (the minimum-norm LS solution).
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
          1. Key "x" present and convertible to float64 array.
          2. Shape matches (num_cols,).
          3. No NaN / Inf.
          4. Relative residual ||Ax - b|| / ||b|| < 0.1
             (matches DiscoverNLA log-reward high_loss = 0.1).

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
            logging.error(f"Wrong shape: expected ({problem['num_cols']},), got {x.shape}.")
            return False
        if not np.isfinite(x).all():
            logging.error("Solution contains NaN or Inf.")
            return False

        A, b = problem["A"], problem["b"]
        rel_res = la.norm(A @ x - b) / (la.norm(b) + 1e-12)
        if rel_res >= 0.01:
            logging.error(f"Relative residual {rel_res:.4e} >= threshold 0.01.")
            return False
        return True
