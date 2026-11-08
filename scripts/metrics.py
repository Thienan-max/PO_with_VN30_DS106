"""Các thước đo hiệu quả danh mục, kiểm định Sharpe và hypervolume."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rf_per_period(rf_annual: float, ppy: float) -> float:
    return (1 + rf_annual) ** (1 / ppy) - 1


def cagr(r: pd.Series, ppy: float) -> float:
    r = np.asarray(r)
    return float(np.prod(1 + r) ** (ppy / len(r)) - 1)


def ann_vol(r, ppy: float) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(ppy))


def sharpe(r, rf_annual: float, ppy: float) -> float:
    ex = np.asarray(r) - rf_per_period(rf_annual, ppy)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(ppy))


def sortino(r, rf_annual: float, ppy: float) -> float:
    ex = np.asarray(r) - rf_per_period(rf_annual, ppy)
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    return float(ex.mean() / downside * np.sqrt(ppy))


def drawdown(r) -> np.ndarray:
    wealth = np.cumprod(1 + np.asarray(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return wealth / peak - 1


def max_drawdown(r) -> float:
    return float(-drawdown(r).min())


def var_cvar(r, alpha: float = 0.95):
    """VaR và CVaR lịch sử (số dương = mức lỗ) ở mức tin cậy alpha."""
    r = np.sort(np.asarray(r))
    k = int(np.floor((1 - alpha) * len(r)))
    k = max(k, 1)
    return float(-r[k - 1]), float(-r[:k].mean())


def enb(w) -> float:
    """Effective Number of Bets = 1 / sum(w^2) (1 = dồn 1 tài sản, N = chia đều)."""
    w = np.asarray(w)
    return float(1.0 / np.sum(w ** 2))


def summary(returns: pd.DataFrame, rf_annual: float, ppy: float, alpha: float = 0.95) -> pd.DataFrame:
    """Bảng chỉ số cho từng cột (mỗi cột là chuỗi return của 1 tài sản/chiến lược)."""
    rows = {}
    for c in returns.columns:
        r = returns[c].dropna().to_numpy()
        v, cv = var_cvar(r, alpha)
        mdd = max_drawdown(r)
        g = cagr(r, ppy)
        rows[c] = {
            "CAGR": g,
            "Volatility": ann_vol(r, ppy),
            "Sharpe": sharpe(r, rf_annual, ppy),
            "Sortino": sortino(r, rf_annual, ppy),
            "MaxDD": mdd,
            "Calmar": g / mdd if mdd > 0 else np.nan,
            f"VaR{int(alpha*100)}": v,
            f"CVaR{int(alpha*100)}": cv,
            "Skew": float(stats.skew(r)),
            "Kurtosis": float(stats.kurtosis(r)),  # excess kurtosis
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Kiểm định thống kê cho Sharpe ratio
# ---------------------------------------------------------------------------
def jobson_korkie_memmel(r1, r2, rf_annual: float, ppy: float):
    """
    Kiểm định H0: SR1 = SR2 (Jobson & Korkie 1981, hiệu chỉnh Memmel 2003).
    Trả về (z, p-value 2 phía).
    """
    rf = rf_per_period(rf_annual, ppy)
    a, b = np.asarray(r1) - rf, np.asarray(r2) - rf
    T = len(a)
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    s12 = np.cov(a, b, ddof=1)[0, 1]
    theta = (2 * s1**2 * s2**2 - 2 * s1 * s2 * s12 + 0.5 * m1**2 * s2**2
             + 0.5 * m2**2 * s1**2 - m1 * m2 / (s1 * s2) * s12**2) / T
    z = (s2 * m1 - s1 * m2) / np.sqrt(theta)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def block_bootstrap_sharpe(r, rf_annual: float, ppy: float, n_boot: int = 2000,
                           block: int = 21, seed: int = 0, ci: float = 0.95):
    """Khoảng tin cậy Sharpe bằng circular block bootstrap (giữ tự tương quan / cụm biến động)."""
    rng = np.random.default_rng(seed)
    ex = np.asarray(r) - rf_per_period(rf_annual, ppy)
    T = len(ex)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T] % T
    sample = ex[idx]
    sr = sample.mean(1) / sample.std(1, ddof=1) * np.sqrt(ppy)
    lo, hi = np.quantile(sr, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Hypervolume 2 mục tiêu (cả 2 đều minimize)
# ---------------------------------------------------------------------------
def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    F = np.asarray(F, dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    hv, best_f2 = 0.0, ref[1]
    for f1, f2 in F:
        if f2 < best_f2:
            hv += (ref[0] - f1) * (best_f2 - f2)
            best_f2 = f2
    return float(hv)
