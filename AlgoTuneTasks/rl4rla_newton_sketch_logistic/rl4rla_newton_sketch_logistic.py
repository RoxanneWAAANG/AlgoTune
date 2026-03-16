"""
AlgoTune Task: rl4rla_newton_sketch_logistic
=============================================
Corresponds to:
  Config:            DiscoverNLA/hp/mcgs_ucd/final_newton_sketch/2b_newton_sketch_composed.yml
  Target algorithm:  "C3_newton_sketch_composed"  (Newton Sketch for logistic regression)
  Reference (solve): "C2_full_newton_composed"    (Full Newton — starting algorithm)
  System type:       LOGISTIC_REG — Gaussian features, binary labels from noisy logits

Problem
-------
Minimise the binary cross-entropy loss over (A, y):
    L(x) = -(1/m) sum_i [ y_i log σ(a_i^T x) + (1-y_i) log(1 - σ(a_i^T x)) ]
where A in R^{m x n}, y in {0,1}^m.

Key parameters (from YAML):
  num-cols     : n  (problem size parameter)
  num-rows     : n * 100   (= 10 000 when n = 100)
  separability : 5.0   (higher → more linearly separable classes)
  max-iterations:  20, learning-rate: 0.5
  sampling-factor: 4,  batch-size: 200
  reward-type:     cross_entropy

Reference algorithm  (Full Newton — C2_full_newton_composed):
    For each iteration t:
      σ   = sigmoid(A x_t)
      r   = σ - y                         (residual)
      w   = σ (1 - σ)                     (Hessian diagonal weights)
      Ã   = D^{1/2} A  where D = diag(w) (weighted feature matrix)
      H   = Ã^T Ã  = A^T D A             (Hessian of L)
      g   = A^T r                         (gradient of L)
      d   = H^{-1} g                      (Newton direction)
      x_{t+1} = x_t - lr * d

Target algorithm  (Newton Sketch — C3_newton_sketch_composed):
    Same as above but replace H with sketched approximation:
      Ã   = D^{1/2} A
      Ã_S = S Ã  where S ~ SJLT of size (sampling_factor*n) x m
      Ĥ   = Ã_S^T Ã_S                    (sketched Hessian)
      d   = Ĥ^{-1} g

Quality threshold (from complexity_reward check: loss < 0.5):
    cross-entropy loss < 0.5
    (log(2) ≈ 0.693 = random-guess loss; < 0.5 means genuine convergence)
"""

import logging

import numpy as np
import scipy.linalg as la
import scipy.sparse as spar

from AlgoTuneTasks.base import register_task, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def _cross_entropy(A, y, x):
    """
    Binary cross-entropy loss with log-sum-exp trick for stability.
      L = (1/m) mean[ max(z,0) - y*z + log(1 + exp(-|z|)) ]
    where z = A @ x.
    """
    z = A @ x
    losses = np.maximum(z, 0.0) - y * z + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(losses))


def _create_sjlt(rng, num_sketch_rows, num_cols, vec_nnz=8):
    """Sparse Johnson-Lindenstrauss Transform  (used by target algorithm)."""
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
        S = spar.coo_matrix((vals, (rows_idx, cols_idx)),
                            shape=(num_sketch_rows, num_cols)).tocsc()
    else:
        S = _create_sjlt(rng, num_cols, num_sketch_rows, vec_nnz)
        S = S.T.tocsr()
    return S


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

@register_task("rl4rla_newton_sketch_logistic")
class Solution(Task):
    """
    Binary logistic regression on a Gaussian feature matrix with controlled
    separability.

    generate_problem replicates
      DiscoverNLA/_create_logistic_regression_system:
        - w_true ~ N(0,I), normalised
        - A ~ N(0,I)^{m x n}
        - logits = A w_true
        - noisy_logits = logits + N(0, 1/separability^2)
        - y = (noisy_logits > 0).astype(float)

    solve() implements the Full Newton method (C2_full_newton_composed),
    which is the starting algorithm for the Newton Sketch curriculum.

    is_solution() checks cross-entropy loss < 0.5
    (DiscoverNLA complexity_reward threshold: loss < 0.5).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    def generate_problem(self, n: int, random_seed: int = 1) -> dict:
        """
        Generate a logistic regression problem (A, y).

        :param n: Number of columns (features).  n=100 → paper dimensions.
        :param random_seed: Reproducibility seed.
        :return: dict with keys
                   A            (num_rows x num_cols, float64)
                   y            (num_rows,)  binary labels in {0, 1}
                   w_true       (num_cols,)  true weight vector
                   num_rows, num_cols, separability
                   sampling_factor, batch_size, max_iterations, learning_rate
        """
        rng = np.random.default_rng(random_seed)
        num_cols = n
        num_rows = n * 100       # m = 100 n  →  10 000 when n = 100
        separability = 5.0

        # True weights (normalised)
        w_true = rng.standard_normal(num_cols)
        w_true = w_true / np.linalg.norm(w_true)

        # Feature matrix
        A = rng.standard_normal((num_rows, num_cols))

        # Noisy logits → binary labels
        logits = A @ w_true
        noise = rng.standard_normal(num_rows) / separability
        noisy_logits = logits + noise
        y = np.where(noisy_logits > 0, 1.0, 0.0)

        return {
            "A": A,
            "y": y,
            "w_true": w_true,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "separability": separability,
            "sampling_factor": 4,
            "batch_size": 200,
            "max_iterations": 20,
            "learning_rate": 0.5,
        }

    # ------------------------------------------------------------------
    def solve(self, problem: dict) -> dict:
        """
        Reference solver: Full Newton method  (C2_full_newton_composed).

        Per iteration:
          σ  = sigmoid(A x_t)
          r  = σ - y
          w  = σ (1 - σ)
          Ã  = diag(sqrt(w)) A      [BUILD_WEIGHTED_MATRIX]
          H  = Ã^T Ã               [COMPUTE_INVERSE_HESSIAN → H^{-1}]
          g  = A^T r
          d  = pinv(H) g
          x_{t+1} = x_t - lr * d

        :param problem: dict from generate_problem.
        :return: dict with key "x".
        """
        A = problem["A"]
        y = problem["y"]
        lr = problem["learning_rate"]     # 0.5
        max_iter = problem["max_iterations"]  # 20

        x = np.zeros(problem["num_cols"])
        for _ in range(max_iter):
            z = A @ x
            p = _sigmoid(z)               # σ
            r = p - y                     # residual = σ - y
            w = p * (1.0 - p)            # Hessian weights σ(1-σ)

            sqrt_w = np.sqrt(w)
            A_tilde = sqrt_w[:, np.newaxis] * A   # D^{1/2} A  (m, n)

            H = A_tilde.T @ A_tilde       # A^T D A  (n, n)   [Hessian]
            H_inv = la.pinv(H)            # H^{-1}

            g = A.T @ r                   # gradient  (n,)
            d = H_inv @ g                 # Newton direction
            x = x - lr * d

        return {"x": x}

    # ------------------------------------------------------------------
    def is_solution(self, problem: dict, solution: dict) -> bool:
        """
        Verify candidate solution quality.

        Checks:
          1. Key "x" present, convertible to float64.
          2. Shape (num_cols,).
          3. No NaN / Inf.
          4. Cross-entropy loss < 0.5
             (DiscoverNLA complexity_reward threshold: loss < 0.5,
              random-guess loss ≈ log(2) ≈ 0.693).

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

        A, y = problem["A"], problem["y"]
        loss = _cross_entropy(A, y, x)
        if loss >= 0.5:
            logging.error(
                f"Cross-entropy loss {loss:.4f} >= 0.5 threshold "
                f"(random-guess ≈ log(2) ≈ 0.693)."
            )
            return False
        return True
