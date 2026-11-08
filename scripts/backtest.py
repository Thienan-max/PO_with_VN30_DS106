"""
Backtest walk-forward (rolling window) - đánh giá out-of-sample, không look-ahead.

Tại mỗi ngày tái cân bằng t: ước lượng trên R[t-lookback : t] -> w_target, nắm giữ đến lần
tái cân bằng kế tiếp (tỷ trọng trôi theo giá - buy & hold giữa 2 lần rebalance).
Chi phí giao dịch = cost_bps × turnover một chiều, trừ vào phiên đầu tiên sau rebalance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series      # return ròng (sau chi phí)
    gross: pd.Series        # return trước chi phí
    weights: pd.DataFrame   # tỷ trọng mục tiêu tại mỗi ngày rebalance
    turnover: pd.Series     # turnover một chiều tại mỗi ngày rebalance
    runtime: float          # tổng thời gian tối ưu (giây)


def walk_forward(returns: pd.DataFrame, strategy, name: str, lookback: int = 484,
                 rebalance_every: int = 21, cost_bps: float = 20.0, start: int | None = None) -> BacktestResult:
    """start: chỉ số phiên bắt đầu out-of-sample (mặc định = lookback)."""
    R = returns.to_numpy()
    T, N = R.shape
    start = lookback if start is None else start
    assert start >= lookback
    rebal = list(range(start, T, rebalance_every))
    gross = np.zeros(T - start)
    net = np.zeros(T - start)
    w_drift = np.zeros(N)
    W_hist, TO, runtime = [], [], 0.0
    for k, t in enumerate(rebal):
        t_end = rebal[k + 1] if k + 1 < len(rebal) else T
        window = returns.iloc[t - lookback:t]
        t0 = time.perf_counter()
        w = np.asarray(strategy(window), dtype=float)
        runtime += time.perf_counter() - t0
        w = np.clip(w, 0, None); w /= w.sum()
        to = 0.5 * np.abs(w - w_drift).sum() if k > 0 else 0.5 * np.abs(w).sum()
        cost = to * cost_bps / 1e4
        seg = R[t:t_end]
        holdings = w * np.cumprod(1 + seg, axis=0)
        value = np.r_[1.0, holdings.sum(axis=1)]
        r_seg = value[1:] / value[:-1] - 1
        gross[t - start:t_end - start] = r_seg
        r_net = r_seg.copy()
        r_net[0] = (1 + r_net[0]) * (1 - cost) - 1
        net[t - start:t_end - start] = r_net
        w_drift = holdings[-1] / holdings[-1].sum()
        W_hist.append(w); TO.append(to)
    idx = returns.index[start:]
    rdates = returns.index[rebal]
    return BacktestResult(name, pd.Series(net, idx, name=name), pd.Series(gross, idx, name=name),
                          pd.DataFrame(W_hist, index=rdates, columns=returns.columns),
                          pd.Series(TO, rdates, name=name), runtime)


def run_all(returns, strategies: dict, verbose=True, **kw) -> dict[str, BacktestResult]:
    out = {}
    for name, fn in strategies.items():
        t0 = time.perf_counter()
        out[name] = walk_forward(returns, fn, name, **kw)
        if verbose:
            print(f"  {name:<18} xong sau {time.perf_counter() - t0:6.1f}s")
    return out


def apply_costs(res: BacktestResult, cost_bps: float) -> pd.Series:
    """Tính lại return ròng với mức phí khác từ return gộp + turnover (không cần chạy lại tối ưu)."""
    net = res.gross.copy()
    cost = res.turnover * cost_bps / 1e4
    net.loc[cost.index] = (1 + net.loc[cost.index]) * (1 - cost.values) - 1
    return net
