"""
MÔ HÌNH 1: TỐI ƯU HÓA DANH MỤC MARKOWITZ (MEAN-VARIANCE OPTIMIZATION)
Đồ án Tối ưu hóa & Ứng dụng - UIT

Mô hình sử dụng phương pháp Quy hoạch Phi tuyến (SLSQP) để tìm:
1. Danh mục Max Sharpe Ratio (Tối ưu hóa lợi nhuận / rủi ro).
2. Danh mục Minimum Volatility (Rủi ro thấp nhất).
3. Đường biên hiệu quả (Efficient Frontier).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Cấu hình UTF-8 để in tiếng Việt chuẩn trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def load_data(filepath=None):
    """Đọc dữ liệu tỷ suất sinh lợi từ file CSV."""
    if filepath is None:
        candidates = [
            os.path.join('data', 'VN30_Sectors_Returns_5Years.csv'),
            os.path.join('..', 'data', 'VN30_Sectors_Returns_5Years.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'VN30_Sectors_Returns_5Years.csv'),
            'VN30_Sectors_Returns_5Years.csv'
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break

    print(f"-> Đang tải dữ liệu từ: {filepath}")
    returns_df = pd.read_csv(filepath).set_index('Date')
    return returns_df


def portfolio_performance(weights, mean_returns, cov_matrix):
    """Tính toán rủi ro (độ lệch chuẩn p_std) và lợi nhuận kỳ vọng (p_ret) hàng năm."""
    p_ret = np.sum(mean_returns * weights)
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_std, p_ret


def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Tìm danh mục có Tỷ số Sharpe cao nhất (Max Sharpe Portfolio).
    Hàm mục tiêu: Min -Sharpe Ratio.
    Ràng buộc: Tổng trọng số = 1, mỗi trọng số w_i in [0, 1].
    """
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        if p_std == 0:
            return 1e6
        return -(p_ret - risk_free_rate) / p_std

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    opt_weights = result.x
    opt_std, opt_ret = portfolio_performance(opt_weights, mean_returns, cov_matrix)
    opt_sharpe = (opt_ret - risk_free_rate) / opt_std if opt_std > 0 else 0

    return opt_weights, opt_std, opt_ret, opt_sharpe


def optimize_min_volatility(mean_returns, cov_matrix):
    """
    Tìm danh mục có rủi ro thấp nhất (Minimum Variance / Volatility Portfolio).
    Hàm mục tiêu: Min Portfolio Volatility.
    """
    num_assets = len(mean_returns)

    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    result = minimize(
        portfolio_vol,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    min_weights = result.x
    min_std, min_ret = portfolio_performance(min_weights, mean_returns, cov_matrix)
    min_sharpe = min_ret / min_std if min_std > 0 else 0

    return min_weights, min_std, min_ret, min_sharpe


def compute_efficient_frontier(mean_returns, cov_matrix, num_points=50):
    """Tính toán các điểm trên đường biên hiệu quả (Efficient Frontier)."""
    _, min_std, min_ret, _ = optimize_min_volatility(mean_returns, cov_matrix)
    _, _, max_ret, _ = optimize_max_sharpe(mean_returns, cov_matrix)

    # Khoảng lợi nhuận từ min_ret đến max_ret (mở rộng thêm 20%)
    target_returns = np.linspace(min_ret * 0.9, max_ret * 1.1, num_points)
    frontier_risks = []
    frontier_returns = []

    num_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    for target_ret in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_ret}
        )
        res = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            np.array([1.0 / num_assets] * num_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if res.success:
            frontier_risks.append(res.fun)
            frontier_returns.append(target_ret)

    return np.array(frontier_risks), np.array(frontier_returns)


# =====================================================================
# CHẠY ĐỘC LẬP MÔ HÌNH MARKOWITZ
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("CHẠY MÔ HÌNH MARKOWITZ (MEAN-VARIANCE OPTIMIZATION)")
    print("=" * 65)

    # 1. Đọc dữ liệu
    returns_df = load_data()
    tickers = returns_df.columns.tolist()
    num_assets = len(tickers)

    # 2. Tính toán thống kê năm hóa (252 ngày giao dịch)
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    print(f"- Số lượng tài sản/ngành: {num_assets} ({', '.join(tickers)})")
    print(f"- Tổng số phiên giao dịch: {len(returns_df)}")

    # 3. Tối ưu hóa Max Sharpe
    print("\n[1] Đang giải bài toán Max Sharpe...")
    max_w, max_std, max_ret, max_sharpe = optimize_max_sharpe(mean_returns, cov_matrix)
    print(f"  • Lợi nhuận kỳ vọng (Return): {max_ret * 100:.2f}%")
    print(f"  • Rủi ro biến động (Risk):    {max_std * 100:.2f}%")
    print(f"  • Tỷ số Sharpe Ratio:         {max_sharpe:.4f}")

    # 4. Tối ưu hóa Min Volatility
    print("\n[2] Đang giải bài toán Min Volatility...")
    min_w, min_std, min_ret, min_sharpe = optimize_min_volatility(mean_returns, cov_matrix)
    print(f"  • Lợi nhuận kỳ vọng (Return): {min_ret * 100:.2f}%")
    print(f"  • Rủi ro biến động (Risk):    {min_std * 100:.2f}%")
    print(f"  • Tỷ số Sharpe Ratio:         {min_sharpe:.4f}")

    # 5. In bảng phân bổ tỷ trọng Max Sharpe
    print("\n[3] Tỷ trọng danh mục tối ưu Max Sharpe:")
    weight_series = pd.Series(max_w, index=tickers)
    for asset, w in weight_series.items():
        if w > 0.001:
            print(f"  • {asset:<20}: {w * 100:.2f}%")

    # 6. Trực quan hóa
    print("\n[4] Đang vẽ biểu đồ kết quả Markowitz...")
    f_risks, f_returns = compute_efficient_frontier(mean_returns, cov_matrix, num_points=60)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Biểu đồ 1: Efficient Frontier
    axes[0].plot(f_risks, f_returns, 'b--', linewidth=2, label='Đường biên hiệu quả (Efficient Frontier)')
    axes[0].scatter(max_std, max_ret, color='red', marker='*', s=350, edgecolor='black', zorder=5,
                    label=f'Max Sharpe ({max_sharpe:.2f})')
    axes[0].scatter(min_std, min_ret, color='green', marker='o', s=200, edgecolor='black', zorder=5,
                    label=f'Min Volatility ({min_std * 100:.1f}%)')

    # Vẽ các tài sản đơn lẻ
    single_vols = np.sqrt(np.diag(cov_matrix))
    axes[0].scatter(single_vols, mean_returns, color='gray', alpha=0.6, s=50, label='Các tài sản riêng lẻ')
    for i, txt in enumerate(tickers):
        axes[0].annotate(txt, (single_vols[i], mean_returns.iloc[i]), fontsize=8, alpha=0.8)

    axes[0].set_title('Mô hình Markowitz - Efficient Frontier', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Rủi ro năm - Volatility (σ)', fontsize=11)
    axes[0].set_ylabel('Lợi nhuận năm - Return (μ)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Biểu đồ 2: Tỷ trọng phân bổ
    significant_w = weight_series[weight_series > 0.005]
    significant_w.plot(kind='bar', ax=axes[1], color='#3b82f6', edgecolor='black', width=0.6)
    axes[1].set_title('Tỷ trọng phân bổ danh mục Max Sharpe', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Tỷ trọng', fontsize=11)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
