import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ===== CÀI ĐẶT THƯ VIỆN NẾU CẦN =====
# pip install scipy pandas numpy pymoo
# ===================================

print("Đang đọc dữ liệu...")
# 1. Đọc dữ liệu Returns (Bỏ qua cột Date để tính toán)
returns_df = pd.read_csv('VN30_Returns_2Years.csv').set_index('Date')

# 2. Tính toán Lợi suất kỳ vọng (Annualized) và Ma trận hiệp phương sai
mean_returns = returns_df.mean() * 252
cov_matrix = returns_df.cov() * 252
num_assets = len(returns_df.columns)
tickers = returns_df.columns.tolist()

# Hàm tính hiệu suất danh mục
def portfolio_performance(weights, mean_returns, cov_matrix):
    p_ret = np.sum(mean_returns * weights)
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_std, p_ret

# ==========================================
# 1. MÔ HÌNH MARKOWITZ (MAX SHARPE RATIO)
# ==========================================
print("\n[1] Đang chạy mô hình Markowitz (Tối ưu Max Sharpe)...")
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

# Ràng buộc tổng tỷ trọng = 1 và các tỷ trọng nằm trong khoảng [0, 1]
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = np.array([1. / num_assets] * num_assets)

# Giải bài toán tối ưu
result_markowitz = minimize(negative_sharpe, initial_weights, args=(mean_returns, cov_matrix),
                            method='SLSQP', bounds=bounds, constraints=constraints)

markowitz_weights = result_markowitz.x
markowitz_std, markowitz_ret = portfolio_performance(markowitz_weights, mean_returns, cov_matrix)

print(f"-> Markowitz Return (Lợi nhuận): {markowitz_ret*100:.2f}%")
print(f"-> Markowitz Risk (Rủi ro): {markowitz_std*100:.2f}%")
print(f"-> Markowitz Sharpe Ratio: {markowitz_ret/markowitz_std:.4f}")
print("-> Tỷ trọng Top 5 mã lớn nhất trong danh mục:")
weight_series = pd.Series(markowitz_weights, index=tickers)
print(weight_series.nlargest(5).apply(lambda x: f"{x*100:.2f}%"))


# ==========================================
# 2. MÔ HÌNH NSGA-III (ĐA MỤC TIÊU)
# ==========================================
print("\n[2] Đang chạy mô hình NSGA-III (Tìm tập Pareto)...")
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.termination import get_termination
    
    # Định nghĩa bài toán
    class PortfolioProblem(ElementwiseProblem):
        def __init__(self, mean_returns, cov_matrix):
            self.mu = mean_returns.values
            self.cov = cov_matrix.values
            n_assets = len(mean_returns)
            
            # 2 mục tiêu: Min Risk, Max Return (Min -Return)
            super().__init__(n_var=n_assets,
                             n_obj=2,
                             n_ieq_constr=0,
                             xl=np.zeros(n_assets),
                             xu=np.ones(n_assets))
                             
        def _evaluate(self, x, out, *args, **kwargs):
            # Normalization để tổng tỷ trọng = 1 (Giúp GA chạy tốt hơn thay vì ép constraint)
            if np.sum(x) == 0: 
                w = np.ones(self.n_var) / self.n_var
            else:
                w = x / np.sum(x)
                
            risk = np.sqrt(np.dot(w.T, np.dot(self.cov, w)))
            ret = np.sum(self.mu * w)
            
            out["F"] = [risk, -ret]

    # Khởi tạo hướng tham chiếu (Reference Directions) cho NSGA-III
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
    
    # Thiết lập thuật toán NSGA-III
    algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
    termination = get_termination("n_gen", 300)  # Lặp 300 thế hệ
    
    # Giải bài toán
    problem = PortfolioProblem(mean_returns, cov_matrix)
    res_nsga3 = pymoo_minimize(problem,
                               algorithm,
                               termination,
                               seed=42,
                               verbose=False)
                               
    print(f"-> NSGA-III đã tìm thấy {len(res_nsga3.F)} danh mục đầu tư Pareto tối ưu.")
    
    # Lấy danh mục có Tỷ số Sharpe cao nhất từ tập Pareto để so sánh với Markowitz
    pareto_risks = res_nsga3.F[:, 0]
    pareto_returns = -res_nsga3.F[:, 1]
    pareto_sharpes = pareto_returns / pareto_risks
    
    best_idx = np.argmax(pareto_sharpes)
    best_w_unnormalized = res_nsga3.X[best_idx]
    best_w = best_w_unnormalized / np.sum(best_w_unnormalized)
    
    print(f"-> [NSGA-III Best Sharpe] Return: {pareto_returns[best_idx]*100:.2f}% | Risk: {pareto_risks[best_idx]*100:.2f}% | Sharpe: {pareto_sharpes[best_idx]:.4f}")
    
    # ==========================================
    # 3. TRỰC QUAN HÓA (VISUALIZATION)
    # ==========================================
    print("\n[3] Đang vẽ biểu đồ trực quan hóa (Cửa sổ sẽ hiện lên, hãy kiểm tra taskbar nhé)...")
    
    # Biểu đồ 1: Efficient Frontier
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vẽ các điểm Pareto của NSGA-III
    sc = axes[0].scatter(pareto_risks, pareto_returns, c=pareto_sharpes, cmap='viridis', marker='o', s=30, label='NSGA-III Pareto Front')
    plt.colorbar(sc, ax=axes[0], label='Sharpe Ratio')
    
    # Đánh dấu các danh mục tối ưu nhất (Max Sharpe)
    axes[0].scatter(markowitz_std, markowitz_ret, color='red', marker='*', s=400, edgecolor='black', label='Markowitz (Max Sharpe)', zorder=5)
    axes[0].scatter(pareto_risks[best_idx], pareto_returns[best_idx], color='magenta', marker='X', s=200, edgecolor='black', label='NSGA-III (Max Sharpe)', zorder=4)
    
    axes[0].set_title('Đường biên hiệu quả (Efficient Frontier)', fontsize=14)
    axes[0].set_xlabel('Rủi ro (Volatility)', fontsize=12)
    axes[0].set_ylabel('Lợi nhuận kỳ vọng (Return)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Biểu đồ 2: So sánh tỷ trọng cổ phiếu
    mk_w = pd.Series(markowitz_weights, index=tickers)
    nsga3_w = pd.Series(best_w, index=tickers)
    
    # Chỉ giữ lại các mã có tỷ trọng > 1% để biểu đồ không bị rối
    mk_w = mk_w[mk_w > 0.01]
    nsga3_w = nsga3_w[nsga3_w > 0.01]
    
    # Gộp data để vẽ cột đứng song song
    comp_df = pd.DataFrame({'Markowitz': mk_w, 'NSGA-III': nsga3_w}).fillna(0)
    
    comp_df.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'], edgecolor='black', width=0.8)
    axes[1].set_title('Tỷ trọng cổ phiếu trong danh mục Max Sharpe (>1%)', fontsize=14)
    axes[1].set_ylabel('Tỷ trọng', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\n[LỖI] Thư viện 'pymoo' chưa được cài đặt.")
    print("Vui lòng chạy lệnh sau trong Terminal để cài đặt: pip install pymoo")
