"""
Mô hình 7 - Black-Litterman (1992), dạng của He & Litterman (1999).

1. Prior cân bằng (reverse optimization):  π = δ Σ w_mkt
2. Quan điểm (views):  P μ = Q + ε,  ε ~ N(0, Ω),   Ω = diag(τ P Σ P')
3. Posterior:  M = [(τΣ)^-1 + P'Ω^-1 P]^-1
               μ_BL = M [(τΣ)^-1 π + P'Ω^-1 Q],   Σ_BL = Σ + M
4. Tối ưu mean-variance: max w'μ_BL - δ/2 w'Σ_BL w  (long-only, 1'w = 1)

Do không có dữ liệu vốn hóa, w_mkt lấy theo tỷ lệ số mã của mỗi ngành trong VN30
(= danh mục equal-weight 30 mã gộp theo ngành). Views được sinh tự động từ momentum
(chỉ dùng dữ liệu quá khứ trong cửa sổ ước lượng -> không look-ahead).
"""
from __future__ import annotations

import numpy as np

from . import markowitz


def implied_returns(cov, w_mkt, delta=2.5):
    return delta * np.asarray(cov) @ np.asarray(w_mkt)


def posterior(cov, pi, P, Q, tau=0.05, omega=None):
    cov, P, Q = np.asarray(cov), np.atleast_2d(P), np.asarray(Q, dtype=float)
    if omega is None:
        omega = np.diag(np.diag(tau * P @ cov @ P.T))
    ts_inv = np.linalg.inv(tau * cov)
    om_inv = np.linalg.inv(omega)
    M = np.linalg.inv(ts_inv + P.T @ om_inv @ P)
    mu_bl = M @ (ts_inv @ pi + P.T @ om_inv @ Q)
    return mu_bl, cov + M


def momentum_views(returns_window, ppy, lookback=126, skip=21, n_pairs=2, scale=0.5):
    """
    Views tương đối: ngành momentum mạnh thứ k sẽ vượt ngành yếu thứ k một khoảng
    Q_k = scale × (chênh lệch momentum năm hóa). scale < 1 thể hiện sự hoài nghi về độ bền momentum.
    Momentum đo trên [t-lookback, t-skip] (bỏ tháng gần nhất để tránh hiệu ứng đảo chiều ngắn hạn).
    """
    R = np.asarray(returns_window)
    seg = R[-lookback:-skip] if skip > 0 else R[-lookback:]
    mom = np.prod(1 + seg, axis=0) ** (ppy / len(seg)) - 1
    rank = np.argsort(mom)[::-1]
    N = R.shape[1]
    P, Q, desc = [], [], []
    for k in range(n_pairs):
        top, bot = rank[k], rank[-1 - k]
        p = np.zeros(N); p[top], p[bot] = 1, -1
        P.append(p); Q.append(scale * (mom[top] - mom[bot])); desc.append((top, bot))
    return np.array(P), np.array(Q), desc


def black_litterman_weights(cov, w_mkt, P, Q, delta=2.5, tau=0.05, w_max=1.0, return_details=False):
    pi = implied_returns(cov, w_mkt, delta)
    mu_bl, cov_bl = posterior(cov, pi, P, Q, tau)
    w = markowitz.mean_variance_utility(mu_bl, cov_bl, delta, w_max)
    return (w, pi, mu_bl, cov_bl) if return_details else w
