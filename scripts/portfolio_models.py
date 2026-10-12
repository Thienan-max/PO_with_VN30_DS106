"""
SO SÁNH CÁC MÔ HÌNH TỐI ƯU HÓA DANH MỤC ĐẦU TƯ
Đồ án Tối ưu hóa & Ứng dụng - UIT

File này đóng vai trò là Benchmark Runner để:
1. Gọi mô hình Markowitz từ `model_markowitz.py`.
2. Gọi mô hình NSGA-III từ `model_nsga3.py`.
3. So sánh hiệu năng (Lợi nhuận, Rủi ro, Tỷ số Sharpe, Cơ cấu tỷ trọng).
4. Trực quan hóa so sánh trực tiếp song song (Efficient Frontier vs Pareto Front).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cấu hình UTF-8 để in tiếng Việt chuẩn trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Thêm thư mục hiện tại của file vào sys.path để import an toàn
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import các mô hình riêng biệt
from model_markowitz import (
    load_data,
    optimize_max_sharpe,
    optimize_min_volatility,
    compute_efficient_frontier
)
from model_nsga3 import optimize_nsga3


def run_benchmark():
    print("=" * 70)
    print("CHƯƠNG TRÌNH SO SÁNH CÁC MÔ HÌNH TỐI ƯU HÓA DANH MỤC (PO BENCHMARK)")
    print("=" * 70)

    # 1. Đọc dữ liệu
    returns_df = load_data()
    tickers = returns_df.columns.tolist()
    num_assets = len(tickers)

    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    print(f"- Số nhóm tài sản/ngành: {num_assets} ({', '.join(tickers)})")
    print(f"- Số phiên giao dịch: {len(returns_df)}")

    # -------------------------------------------------------------
    # 2. CHẠY MÔ HÌNH MARKOWITZ
    # -------------------------------------------------------------
    print("\n[1] Đang thực thi mô hình Markowitz (SLSQP)...")
    mk_w, mk_std, mk_ret, mk_sharpe = optimize_max_sharpe(mean_returns, cov_matrix)
    print(f"  -> [Markowitz Max Sharpe] Return: {mk_ret * 100:.2f}% | Risk: {mk_std * 100:.2f}% | Sharpe: {mk_sharpe:.4f}")

    # -------------------------------------------------------------
    # 3. CHẠY MÔ HÌNH NSGA-III
    # -------------------------------------------------------------
    print("\n[2] Đang thực thi mô hình NSGA-III (Tiến hóa đa mục tiêu)...")
    res_nsga3, pareto_risks, pareto_returns, pareto_sharpes, nsga3_w, nsga3_std, nsga3_ret, nsga3_sharpe = optimize_nsga3(
        mean_returns, cov_matrix, pop_size=100, n_gen=300, seed=42
    )
    print(f"  -> [NSGA-III Best Sharpe] Return: {nsga3_ret * 100:.2f}% | Risk: {nsga3_std * 100:.2f}% | Sharpe: {nsga3_sharpe:.4f}")
    print(f"  -> Tìm thấy {len(pareto_risks)} danh mục Pareto tối ưu.")

    # -------------------------------------------------------------
    # 4. BẢNG SO SÁNH CHỈ SỐ VÀ TỶ TRỌNG
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP SO SÁNH HIỆU SUẤT:")
    print("=" * 70)

    comparison_metrics = pd.DataFrame({
        'Tiêu chí': ['Lợi nhuận năm (%)', 'Rủi ro biến động (%)', 'Tỷ số Sharpe (Annualized)'],
        'Markowitz': [f"{mk_ret * 100:.2f}%", f"{mk_std * 100:.2f}%", f"{mk_sharpe:.4f}"],
        'NSGA-III': [f"{nsga3_ret * 100:.2f}%", f"{nsga3_std * 100:.2f}%", f"{nsga3_sharpe:.4f}"]
    })
    print(comparison_metrics.to_string(index=False))

    print("\nSO SÁNH TỶ TRỌNG PHÂN BỔ (%):")
    weights_comp = pd.DataFrame({
        'Nhóm ngành': tickers,
        'Markowitz (%)': (mk_w * 100).round(2),
        'NSGA-III (%)': (nsga3_w * 100).round(2)
    }).set_index('Nhóm ngành')
    print(weights_comp)

    # -------------------------------------------------------------
    # 5. TRỰC QUAN HÓA SO SÁNH (VISUALIZATION)
    # -------------------------------------------------------------
    print("\n[3] Đang vẽ biểu đồ so sánh song song...")
    f_risks, f_returns = compute_efficient_frontier(mean_returns, cov_matrix, num_points=60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Biểu đồ 1: So sánh Efficient Frontier (Markowitz) và Pareto Front (NSGA-III)
    sc = axes[0].scatter(pareto_risks, pareto_returns, c=pareto_sharpes, cmap='viridis', marker='o', s=35,
                         label='NSGA-III Pareto Front', alpha=0.9)
    plt.colorbar(sc, ax=axes[0], label='Sharpe Ratio (NSGA-III)')

    axes[0].plot(f_risks, f_returns, 'r--', linewidth=2, label='Markowitz Efficient Frontier')
    axes[0].scatter(mk_std, mk_ret, color='red', marker='*', s=350, edgecolor='black', zorder=6,
                    label=f'Markowitz Max Sharpe ({mk_sharpe:.2f})')
    axes[0].scatter(nsga3_std, nsga3_ret, color='magenta', marker='X', s=220, edgecolor='black', zorder=5,
                    label=f'NSGA-III Best Sharpe ({nsga3_sharpe:.2f})')

    axes[0].set_title('So sánh Đường biên hiệu quả: Markowitz vs NSGA-III', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Rủi ro năm - Volatility (σ)', fontsize=11)
    axes[0].set_ylabel('Lợi nhuận năm - Return (μ)', fontsize=11)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Biểu đồ 2: So sánh cơ cấu tỷ trọng
    mk_series = pd.Series(mk_w, index=tickers)
    nsga3_series = pd.Series(nsga3_w, index=tickers)
    comp_df = pd.DataFrame({'Markowitz': mk_series, 'NSGA-III': nsga3_series})

    # Chỉ giữ các ngành có tỷ trọng > 0.5% ở ít nhất một mô hình
    comp_filtered = comp_df[(comp_df['Markowitz'] > 0.005) | (comp_df['NSGA-III'] > 0.005)]

    comp_filtered.plot(kind='bar', ax=axes[1], color=['#ef4444', '#10b981'], edgecolor='black', width=0.7)
    axes[1].set_title('So sánh Tỷ trọng phân bổ danh mục (>0.5%)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Tỷ trọng', fontsize=11)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_benchmark()
