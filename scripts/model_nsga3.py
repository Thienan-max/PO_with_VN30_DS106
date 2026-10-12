"""
MÔ HÌNH 2: TỐI ƯU HÓA DANH MỤC ĐA MỤC TIÊU NSGA-III (NON-DOMINATED SORTING GENETIC ALGORITHM III)
Đồ án Tối ưu hóa & Ứng dụng - UIT

Thuật toán tiến hóa đa mục tiêu tìm tập nghiệm Pareto tối ưu:
- Mục tiêu 1: Cực tiểu hóa Rủi ro (Minimizing Volatility / Risk)
- Mục tiêu 2: Cực đại hóa Lợi nhuận (Maximizing Expected Return -> Min -Return)
Sử dụng các hướng tham chiếu (Reference Directions) theo phương pháp Das-Dennis.
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

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.termination import get_termination
except ImportError:
    print("\n[LỖI] Thư viện 'pymoo' chưa được cài đặt.")
    print("Vui lòng cài đặt bằng lệnh: pip install pymoo")
    sys.exit(1)


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
    """Tính toán rủi ro và lợi nhuận kỳ vọng năm."""
    p_ret = np.sum(mean_returns * weights)
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_std, p_ret


class PortfolioProblem(ElementwiseProblem):
    """
    Định nghĩa bài toán Tối ưu hóa danh mục đa mục tiêu trong Pymoo.
    - Biến quyết định x: vector n_assets chiều, x_i in [0, 1].
    - Chuẩn hóa: w = x / sum(x) để đảm bảo tổng tỷ trọng = 1.
    - Mục tiêu F: [Risk, -Return] (cả 2 đều là bài toán Min).
    """
    def __init__(self, mean_returns, cov_matrix):
        self.mu = mean_returns.values if hasattr(mean_returns, 'values') else mean_returns
        self.cov = cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix
        n_assets = len(mean_returns)

        super().__init__(
            n_var=n_assets,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.zeros(n_assets),
            xu=np.ones(n_assets)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Chuẩn hóa simplex (tổng trọng số = 1)
        sum_x = np.sum(x)
        if sum_x == 0:
            w = np.ones(self.n_var) / self.n_var
        else:
            w = x / sum_x

        risk = np.sqrt(np.dot(w.T, np.dot(self.cov, w)))
        ret = np.sum(self.mu * w)

        out["F"] = [risk, -ret]


def optimize_nsga3(mean_returns, cov_matrix, pop_size=100, n_gen=300, seed=42):
    """
    Thực thi thuật toán NSGA-III giải bài toán tối ưu danh mục.
    Trả về:
    - res: đối tượng kết quả pymoo
    - pareto_risks, pareto_returns, pareto_sharpes: các giá trị trên tập Pareto
    - best_w: vector trọng số tối ưu Max Sharpe trích xuất từ tập Pareto
    """
    # Khởi tạo hướng tham chiếu Das-Dennis 2 mục tiêu
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)

    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    termination = get_termination("n_gen", n_gen)
    problem = PortfolioProblem(mean_returns, cov_matrix)

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False
    )

    pareto_risks = res.F[:, 0]
    pareto_returns = -res.F[:, 1]
    pareto_sharpes = pareto_returns / pareto_risks

    # Tìm danh mục có Sharpe Ratio cao nhất trên mặt Pareto
    best_idx = np.argmax(pareto_sharpes)
    best_w_raw = res.X[best_idx]
    best_w = best_w_raw / np.sum(best_w_raw)

    best_risk = pareto_risks[best_idx]
    best_ret = pareto_returns[best_idx]
    best_sharpe = pareto_sharpes[best_idx]

    return res, pareto_risks, pareto_returns, pareto_sharpes, best_w, best_risk, best_ret, best_sharpe


# =====================================================================
# CHẠY ĐỘC LẬP MÔ HÌNH NSGA-III
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("CHẠY MÔ HÌNH TIẾN HÓA ĐA MỤC TIÊU NSGA-III")
    print("=" * 65)

    # 1. Đọc dữ liệu
    returns_df = load_data()
    tickers = returns_df.columns.tolist()
    num_assets = len(tickers)

    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    print(f"- Số lượng tài sản/ngành: {num_assets} ({', '.join(tickers)})")
    print(f"- Tổng số phiên giao dịch: {len(returns_df)}")

    # 2. Chạy thuật toán NSGA-III
    POP_SIZE = 100
    N_GEN = 300
    print(f"\n[1] Đang tối ưu hóa NSGA-III (Pop: {POP_SIZE}, Generations: {N_GEN})...")
    res, p_risks, p_returns, p_sharpes, best_w, best_risk, best_ret, best_sharpe = optimize_nsga3(
        mean_returns, cov_matrix, pop_size=POP_SIZE, n_gen=N_GEN, seed=42
    )

    print(f"-> Thuật toán đã hội tụ và tìm thấy {len(res.F)} danh mục tối ưu trên tập Pareto.")
    print(f"\n[2] Danh mục Best Sharpe trích xuất từ tập Pareto NSGA-III:")
    print(f"  • Lợi nhuận kỳ vọng (Return): {best_ret * 100:.2f}%")
    print(f"  • Rủi ro biến động (Risk):    {best_risk * 100:.2f}%")
    print(f"  • Tỷ số Sharpe Ratio:         {best_sharpe:.4f}")

    # 3. In tỷ trọng danh mục Best Sharpe
    print("\n[3] Tỷ trọng phân bổ danh mục Best Sharpe:")
    weight_series = pd.Series(best_w, index=tickers)
    for asset, w in weight_series.items():
        if w > 0.001:
            print(f"  • {asset:<20}: {w * 100:.2f}%")

    # 4. Trực quan hóa
    print("\n[4] Đang vẽ biểu đồ kết quả NSGA-III...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Biểu đồ 1: Pareto Front
    sc = axes[0].scatter(p_risks, p_returns, c=p_sharpes, cmap='viridis', marker='o', s=35, label='Tập Pareto NSGA-III')
    cbar = plt.colorbar(sc, ax=axes[0])
    cbar.set_label('Tỷ số Sharpe (Sharpe Ratio)', fontsize=10)

    # Đánh dấu điểm Best Sharpe
    axes[0].scatter(best_risk, best_ret, color='magenta', marker='X', s=250, edgecolor='black', zorder=5,
                    label=f'Best Sharpe ({best_sharpe:.2f})')

    axes[0].set_title('Tập Pareto Frontier - Thuật toán NSGA-III', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Rủi ro năm - Volatility (σ)', fontsize=11)
    axes[0].set_ylabel('Lợi nhuận năm - Return (μ)', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Biểu đồ 2: Tỷ trọng phân bổ
    significant_w = weight_series[weight_series > 0.005]
    significant_w.plot(kind='bar', ax=axes[1], color='#10b981', edgecolor='black', width=0.6)
    axes[1].set_title('Tỷ trọng phân bổ danh mục Best Sharpe (NSGA-III)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Tỷ trọng', fontsize=11)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
