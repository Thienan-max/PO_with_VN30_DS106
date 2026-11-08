# Tối ưu hóa danh mục VN30 — 7 nhóm ngành

Đồ án môn **Tối ưu hóa & Ứng dụng** (UIT). So sánh 7 mô hình phân bổ danh mục trên 7 chỉ số ngành equal-weighted được gom từ 30 mã VN30 (10/2021 – 10/2026).

## Cấu trúc

```
crawl_group_vn30.ipynb     Crawl dữ liệu (yfinance) và gom 30 mã thành 7 ngành
01_EDA.ipynb               EDA, kiểm tra chất lượng và làm sạch dữ liệu -> data/processed/
02_So_sanh_mo_hinh.ipynb   Triển khai và so sánh 7 mô hình (in-sample + walk-forward)
data/                      Dữ liệu gốc (không chỉnh sửa)
data/processed/            Dữ liệu đã làm sạch + nhật ký làm sạch (tạo bởi 01_EDA)
scripts/
  data_utils.py            Đọc, làm sạch dữ liệu, ước lượng μ, Σ
  metrics.py               Sharpe, Sortino, MDD, CVaR, kiểm định Jobson-Korkie-Memmel, bootstrap, hypervolume
  markowitz.py             Min-Variance, Max-Sharpe (QP lồi), đường biên hiệu quả
  nsga3.py                 NSGA-III 3 mục tiêu (return, volatility, max drawdown) - pymoo
  map_elites.py            MAP-Elites (Grid) và CVT-MAP-Elites, toán tử Iso+LineDD
  mean_cvar.py             Mean-CVaR (LP Rockafellar-Uryasev, Charnes-Cooper cho Max STARR) - HiGHS
  hrp.py                   Hierarchical Risk Parity
  black_litterman.py       Black-Litterman với views momentum
  backtest.py              Backtest walk-forward có phí giao dịch
  viz.py                   Style biểu đồ
```

## Chạy

```bash
pip install -r requirements.txt
# chạy lần lượt trong VS Code / Jupyter, từ thư mục gốc của đồ án:
#   01_EDA.ipynb  ->  02_So_sanh_mo_hinh.ipynb
```

`02_So_sanh_mo_hinh.ipynb` mất khoảng 10 phút trên CPU laptop (phần lớn là các thuật toán tiến hóa trong backtest walk-forward và bootstrap).
