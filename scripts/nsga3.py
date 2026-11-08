"""
Mô hình 2 - NSGA-III (Deb & Jain 2014): tiến hóa đa/nhiều mục tiêu với reference directions.

Biến quyết định x in [0,1]^N, ánh xạ lên simplex w = x / sum(x).
Mục tiêu (đều là minimize), chọn tùy ý trong:
  'return'     -> -μ'w                       (lợi nhuận năm)
  'volatility' -> sqrt(w'Σw)                 (độ lệch chuẩn năm)
  'mdd'        -> max drawdown trên chuỗi in-sample (không lồi, không khả vi -> lợi thế của EA)
  'cvar'       -> CVaR lịch sử theo ngày
  'hhi'        -> sum w^2 (mức độ tập trung)
Mặc định 3 mục tiêu (return, volatility, mdd) - NSGA-III được thiết kế cho >= 3 mục tiêu.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


def _to_simplex(X):
    s = X.sum(axis=1, keepdims=True)
    W = np.where(s > 0, X / np.where(s > 0, s, 1), 1 / X.shape[1])
    return W


def batch_mdd(R, W):
    """Max drawdown cho nhiều danh mục: R (T x N) return ngày, W (P x N)."""
    wealth = np.cumprod(1 + R @ W.T, axis=0)
    wealth = np.vstack([np.ones((1, W.shape[0])), wealth])
    return -(wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0)


def batch_cvar(R, W, alpha=0.95):
    P = R @ W.T
    k = max(int(np.floor((1 - alpha) * len(R))), 1)
    return -np.sort(P, axis=0)[:k].mean(axis=0)


class PortfolioProblem(Problem):
    def __init__(self, R, mu, cov, objectives=("return", "volatility", "mdd"), alpha=0.95):
        self.R, self.mu, self.cov, self.objectives, self.alpha = R, mu, cov, tuple(objectives), alpha
        n = R.shape[1]
        super().__init__(n_var=n, n_obj=len(objectives), xl=np.zeros(n), xu=np.ones(n))

    def objective_values(self, W):
        cols = []
        for o in self.objectives:
            if o == "return":
                cols.append(-(W @ self.mu))
            elif o == "volatility":
                cols.append(np.sqrt(np.einsum("ij,jk,ik->i", W, self.cov, W)))
            elif o == "mdd":
                cols.append(batch_mdd(self.R, W))
            elif o == "cvar":
                cols.append(batch_cvar(self.R, W, self.alpha))
            elif o == "hhi":
                cols.append((W ** 2).sum(axis=1))
            else:
                raise ValueError(o)
        return np.column_stack(cols)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.objective_values(_to_simplex(X))


class _History(Callback):
    """Lưu tập nghiệm không bị trội (W) ở mỗi thế hệ để vẽ đường hội tụ."""
    def __init__(self):
        super().__init__()
        self.W = []

    def notify(self, algorithm):
        self.W.append(_to_simplex(algorithm.opt.get("X")))


@dataclass
class NSGA3Result:
    W: np.ndarray            # trọng số tập Pareto (P x N)
    F: np.ndarray            # giá trị mục tiêu (P x M), 'return' đã đổi dấu về dương
    objectives: tuple
    sharpe: np.ndarray
    n_eval: int
    history: list = field(default_factory=list)

    def select(self, rule: str = "max_sharpe") -> np.ndarray:
        if rule == "max_sharpe":
            return self.W[int(np.argmax(self.sharpe))]
        if rule == "max_calmar":
            ret = self.F[:, self.objectives.index("return")]
            mdd = self.F[:, self.objectives.index("mdd")]
            return self.W[int(np.argmax(ret / np.maximum(mdd, 1e-9)))]
        raise ValueError(rule)


def run_nsga3(R, mu, cov, rf=0.0, objectives=("return", "volatility", "mdd"),
              n_partitions=12, n_gen=200, seed=0, track_history=False) -> NSGA3Result:
    """R: ma trận return ngày (T x N); mu, cov: đã năm hóa."""
    m = len(objectives)
    ref_dirs = get_reference_directions("das-dennis", m, n_partitions=n_partitions)
    pop = int(np.ceil(len(ref_dirs) / 4) * 4)
    problem = PortfolioProblem(np.asarray(R), mu, cov, objectives)
    cb = _History() if track_history else None
    kw = {"callback": cb} if cb else {}
    res = minimize(problem, NSGA3(pop_size=pop, ref_dirs=ref_dirs), ("n_gen", n_gen),
                   seed=seed, verbose=False, **kw)
    W = _to_simplex(np.atleast_2d(res.X))
    F = problem.objective_values(W)
    if "return" in objectives:
        F[:, objectives.index("return")] *= -1
    ret = W @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return NSGA3Result(W, F, tuple(objectives), (ret - rf) / vol, res.algorithm.evaluator.n_eval,
                       cb.W if cb else [])


def nsga3_weights(R, mu, cov, rf=0.0, rule="max_sharpe", **kw) -> np.ndarray:
    return run_nsga3(R, mu, cov, rf, **kw).select(rule)
