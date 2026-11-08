"""
Mô hình 1 - Markowitz Mean-Variance (quy hoạch toàn phương, giải bằng SLSQP).

- Min Variance:  min w'Σw              s.t. 1'w = 1, 0 <= w <= w_max
- Max Sharpe:    max (μ'w - rf)/sqrt(w'Σw)  -> không lồi, nên dùng phép đổi biến
                 y = w / ((μ-rf)'w) để đưa về QP lồi:
                 min y'Σy  s.t. (μ-rf)'y = 1, y >= 0   rồi w = y / 1'y
- Efficient frontier: min w'Σw s.t. μ'w = target
- Mean-variance utility (dùng cho Black-Litterman): max w'μ - δ/2 w'Σw
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_OPTS = {"ftol": 1e-12, "maxiter": 500}


def portfolio_stats(w, mu, cov, rf: float = 0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def _clean(w):
    w = np.clip(w, 0, None)
    w[w < 1e-6] = 0.0
    return w / w.sum()


def min_variance(cov, w_max: float = 1.0):
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                   method="SLSQP", bounds=[(0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options=_OPTS)
    return _clean(res.x)


def max_sharpe(mu, cov, rf: float = 0.0, w_max: float = 1.0):
    """Tangency portfolio qua QP lồi. Nếu không tài sản nào có μ > rf thì trả về Min Variance."""
    ex = mu - rf
    n = len(mu)
    if ex.max() <= 0:
        return min_variance(cov, w_max)
    if ex.mean() > 0:
        y0 = np.full(n, 1 / n) / ex.mean()
    else:
        y0 = np.zeros(n)
        k = int(np.argmax(ex))
        y0[k] = 1 / ex[k]
    cons = [{"type": "eq", "fun": lambda y: ex @ y - 1, "jac": lambda y: ex}]
    if w_max < 1:  # y_i <= w_max * 1'y
        cons.append({"type": "ineq", "fun": lambda y: w_max * y.sum() - y,
                     "jac": lambda y: w_max * np.ones((n, n)) - np.eye(n)})
    res = minimize(lambda y: y @ cov @ y, y0, jac=lambda y: 2 * cov @ y, method="SLSQP",
                   bounds=[(0, None)] * n, constraints=cons, options=_OPTS)
    return _clean(res.x)


def max_feasible_return(mu, w_max: float = 1.0) -> float:
    """max μ'w trên {1'w=1, 0<=w<=w_max}: lấp đầy tham lam các tài sản μ cao nhất."""
    left, total = 1.0, 0.0
    for m in np.sort(mu)[::-1]:
        take = min(w_max, left)
        total += take * m
        left -= take
        if left <= 1e-12:
            break
    return float(total)


def efficient_frontier(mu, cov, n_points: int = 50, w_max: float = 1.0):
    """Trả về (vols, rets, weights) dọc đường biên hiệu quả."""
    n = len(mu)
    w_mv = min_variance(cov, w_max)
    r_lo, r_hi = w_mv @ mu, max_feasible_return(mu, w_max)
    vols, rets, W = [], [], []
    w0 = w_mv
    for target in np.linspace(r_lo, r_hi * 0.999, n_points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t, "jac": lambda w: mu}]
        res = minimize(lambda w: w @ cov @ w, w0, jac=lambda w: 2 * cov @ w, method="SLSQP",
                       bounds=[(0, w_max)] * n, constraints=cons, options=_OPTS)
        if res.success:
            w0 = res.x
            w = _clean(res.x)
            vols.append(np.sqrt(w @ cov @ w)); rets.append(w @ mu); W.append(w)
    return np.array(vols), np.array(rets), np.array(W)


def mean_variance_utility(mu, cov, delta: float = 2.5, w_max: float = 1.0):
    n = len(mu)
    res = minimize(lambda w: -(w @ mu) + 0.5 * delta * w @ cov @ w, np.full(n, 1 / n),
                   jac=lambda w: -mu + delta * cov @ w, method="SLSQP", bounds=[(0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options=_OPTS)
    return _clean(res.x)
