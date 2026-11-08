"""
Mô hình 5 - Mean-CVaR (Rockafellar & Uryasev 2000): quy hoạch tuyến tính trên T kịch bản lịch sử.

CVaR_β(w) = min_α  α + 1/((1-β)T) Σ_s max(0, -r_s'w - α)

Min-CVaR (LP):   min  α + c·Σ u_s
                 s.t. u_s >= -r_s'w - α,  u_s >= 0,  1'w = 1,  μ'w >= target,  0 <= w <= w_max
Max STARR = (μ'w - rf)/CVaR(w): tỷ số phân thức -> biến đổi Charnes-Cooper (y = t·w) thành LP.
Giải bằng HiGHS (scipy.optimize.linprog).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def portfolio_cvar(w, R, beta=0.95):
    p = np.sort(np.asarray(R) @ w)
    k = max(int(np.floor((1 - beta) * len(p))), 1)
    return float(-p[:k].mean())


def _clean(w):
    w = np.clip(w, 0, None)
    w[w < 1e-6] = 0
    return w / w.sum()


def min_cvar(R, beta=0.95, target_return=None, w_max=1.0):
    """R: return ngày (T x N). target_return: return trung bình ngày tối thiểu."""
    R = np.asarray(R)
    T, N = R.shape
    c = np.r_[np.zeros(N), 1.0, np.full(T, 1 / ((1 - beta) * T))]
    # -R w - α - u <= 0
    A_ub = sparse.hstack([sparse.csr_matrix(-R), -np.ones((T, 1)), -sparse.eye(T)]).tocsr()
    b_ub = np.zeros(T)
    if target_return is not None:
        mu = R.mean(axis=0)
        A_ub = sparse.vstack([A_ub, sparse.csr_matrix(np.r_[-mu, 0, np.zeros(T)])])
        b_ub = np.r_[b_ub, -target_return]
    A_eq = sparse.csr_matrix(np.r_[np.ones(N), 0, np.zeros(T)])
    bounds = [(0, w_max)] * N + [(None, None)] + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        return None
    return _clean(res.x[:N])


def max_starr(R, rf_daily=0.0, beta=0.95, w_max=1.0):
    """Tối đa (μ'w - rf)/CVaR(w) bằng Charnes-Cooper. Fallback Min-CVaR nếu mọi μ <= rf."""
    R = np.asarray(R)
    T, N = R.shape
    ex = R.mean(axis=0) - rf_daily
    if ex.max() <= 0:
        return min_cvar(R, beta, w_max=w_max)
    # biến: [y (N), α' (1), u' (T), t (1)]
    c = np.r_[np.zeros(N), 1.0, np.full(T, 1 / ((1 - beta) * T)), 0.0]
    A1 = sparse.hstack([sparse.csr_matrix(-R), -np.ones((T, 1)), -sparse.eye(T), np.zeros((T, 1))])
    blocks, b_ub = [A1], [np.zeros(T)]
    if w_max < 1:  # y_i - w_max t <= 0
        blocks.append(sparse.hstack([sparse.eye(N), np.zeros((N, 1 + T)), -w_max * np.ones((N, 1))]))
        b_ub.append(np.zeros(N))
    A_ub = sparse.vstack(blocks).tocsr()
    A_eq = sparse.csr_matrix(np.vstack([np.r_[ex, 0, np.zeros(T), 0],              # (μ-rf)'y = 1
                                        np.r_[np.ones(N), 0, np.zeros(T), -1]]))    # 1'y = t
    bounds = [(0, None)] * N + [(None, None)] + [(0, None)] * T + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.concatenate(b_ub), A_eq=A_eq, b_eq=[1.0, 0.0],
                  bounds=bounds, method="highs")
    if not res.success or res.x[-1] <= 1e-12:
        return min_cvar(R, beta, w_max=w_max)
    return _clean(res.x[:N] / res.x[-1])


def cvar_frontier(R, beta=0.95, n_points=30, w_max=1.0):
    """Trả về (cvars, mean_returns_daily, weights) dọc đường biên Mean-CVaR."""
    R = np.asarray(R)
    mu = R.mean(axis=0)
    w0 = min_cvar(R, beta, w_max=w_max)
    lo, hi = w0 @ mu, mu.max()
    out_c, out_r, out_w = [], [], []
    for t in np.linspace(lo, hi * 0.999 if hi > 0 else hi, n_points):
        w = min_cvar(R, beta, target_return=t, w_max=w_max)
        if w is not None:
            out_c.append(portfolio_cvar(w, R, beta)); out_r.append(w @ mu); out_w.append(w)
    return np.array(out_c), np.array(out_r), np.array(out_w)
