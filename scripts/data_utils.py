"""
Đọc, kiểm tra chất lượng và làm sạch dữ liệu VN30.

File gốc trong data/ KHÔNG bị sửa. Dữ liệu sạch được ghi ra data/processed/.

Hai lỗi dữ liệu chính của nguồn yfinance cho thị trường VN:
1. Phiên nghỉ lễ (Tết, 30/4, 2/9...) bị forward-fill -> cả 30 mã có return = 0.
2. Giá chưa điều chỉnh đúng khi có chia tách / cổ tức cổ phiếu -> return vượt xa
   biên độ HOSE (+-7%), hoặc cặp "spike rồi đảo chiều" ở 2 phiên liên tiếp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

HOSE_BAND = 0.07  # biên độ dao động giá sàn HOSE

SECTOR_ORDER = [
    "NganHang", "BatDongSan", "TieuDung_BanLe", "NangLuong_TienIch",
    "Thep_VatLieu", "CongNghe", "TaiChinh_DichVu",
]
SECTOR_VI = {
    "NganHang": "Ngân hàng", "BatDongSan": "Bất động sản", "TieuDung_BanLe": "Tiêu dùng & Bán lẻ",
    "NangLuong_TienIch": "Năng lượng & Tiện ích", "Thep_VatLieu": "Thép & Vật liệu",
    "CongNghe": "Công nghệ", "TaiChinh_DichVu": "Dịch vụ tài chính",
}


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True).sort_index()


def load_raw() -> dict[str, pd.DataFrame]:
    """Đọc toàn bộ dữ liệu gốc."""
    mapping = pd.read_csv(DATA_DIR / "VN30_Sector_Mapping.csv")
    return {
        "prices": _read("VN30_Prices_5Years.csv"),
        "returns": _read("VN30_Returns_5Years.csv"),
        "sector_prices": _read("VN30_Sectors_Prices_5Years.csv"),
        "sector_returns": _read("VN30_Sectors_Returns_5Years.csv"),
        "mapping": mapping,
    }


def load_clean(level: str = "sector") -> pd.DataFrame:
    """Đọc dữ liệu đã làm sạch (chạy 01_EDA.ipynb trước). level = 'sector' | 'stock'."""
    name = "VN30_Sectors_Returns_clean.csv" if level == "sector" else "VN30_Returns_clean.csv"
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Hãy chạy notebook 01_EDA.ipynb trước.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def sector_counts(mapping: pd.DataFrame) -> pd.Series:
    """Số mã trong mỗi ngành (theo SECTOR_ORDER)."""
    return mapping["Sector"].value_counts().reindex(SECTOR_ORDER)


# ---------------------------------------------------------------------------
# Kiểm tra chất lượng
# ---------------------------------------------------------------------------
def holiday_rows(returns: pd.DataFrame) -> pd.Series:
    """Mask các phiên mà TẤT CẢ mã đều có return = 0 (ngày nghỉ bị forward-fill)."""
    return (returns == 0).all(axis=1)


def detect_anomalies(returns: pd.DataFrame, hard: float = 0.20, spike: float = 0.10) -> pd.DataFrame:
    """
    Phát hiện return bất thường (chạy SAU khi đã bỏ các phiên nghỉ lễ).

    - 'spike_reversal': |r_t| > spike, |r_{t+1}| > spike và trái dấu -> lỗi giá 1 phiên.
      Xử lý: gộp thành 1 return ròng ở phiên t+1, phiên t đặt = 0 (bảo toàn giá trị tích lũy).
    - 'corporate_action': |r_t| > hard (không thể đạt được kể cả 2-3 phiên trần/sàn liên tiếp
      sau khi đã bỏ phiên nghỉ) -> do chia tách/cổ tức cổ phiếu chưa điều chỉnh. Đặt = 0.
    """
    rows = []
    r = returns.to_numpy()
    idx = returns.index
    used = np.zeros_like(r, dtype=bool)
    for j, tic in enumerate(returns.columns):
        for t in range(len(r) - 1):
            a, b = r[t, j], r[t + 1, j]
            if abs(a) > spike and abs(b) > spike and np.sign(a) != np.sign(b) and not used[t, j]:
                rows.append(dict(Date=idx[t], Ticker=tic, Return=a, NextReturn=b,
                                 Type="spike_reversal", Fixed=(1 + a) * (1 + b) - 1))
                used[t, j] = used[t + 1, j] = True
        for t in range(len(r)):
            if abs(r[t, j]) > hard and not used[t, j]:
                rows.append(dict(Date=idx[t], Ticker=tic, Return=r[t, j],
                                 NextReturn=r[t + 1, j] if t + 1 < len(r) else np.nan,
                                 Type="corporate_action", Fixed=0.0))
                used[t, j] = True
    cols = ["Date", "Ticker", "Return", "NextReturn", "Type", "Fixed"]
    return pd.DataFrame(rows, columns=cols).sort_values("Date").reset_index(drop=True)


def clean_returns(returns: pd.DataFrame, hard: float = 0.20, spike: float = 0.10):
    """Trả về (returns_sạch, log_anomaly, số_phiên_nghỉ_bị_bỏ)."""
    hol = holiday_rows(returns)
    r = returns.loc[~hol].copy()
    log = detect_anomalies(r, hard=hard, spike=spike)
    for _, row in log.iterrows():
        t = r.index.get_loc(row.Date)
        if row.Type == "spike_reversal":
            r.iloc[t + 1, r.columns.get_loc(row.Ticker)] = row.Fixed
            r.iloc[t, r.columns.get_loc(row.Ticker)] = 0.0
        else:
            r.iloc[t, r.columns.get_loc(row.Ticker)] = 0.0
    return r, log, int(hol.sum())


def build_sector_returns(stock_returns: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Chỉ số ngành equal-weighted: R_k,t = mean_{i in S_k} R_i,t."""
    out = {s: stock_returns[mapping.loc[mapping.Sector == s, "Ticker"]].mean(axis=1)
           for s in SECTOR_ORDER}
    return pd.DataFrame(out, index=stock_returns.index)


def save_processed(stock_returns: pd.DataFrame, sector_returns: pd.DataFrame, log: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stock_returns.to_csv(PROCESSED_DIR / "VN30_Returns_clean.csv")
    sector_returns.to_csv(PROCESSED_DIR / "VN30_Sectors_Returns_clean.csv")
    log.to_csv(PROCESSED_DIR / "cleaning_log.csv", index=False)


# ---------------------------------------------------------------------------
# Ước lượng tham số
# ---------------------------------------------------------------------------
def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Số phiên/năm thực tế của mẫu (sau khi bỏ ngày nghỉ ~ 242, không phải 252)."""
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years


def estimate_moments(returns: pd.DataFrame, ppy: float, method: str = "sample"):
    """mu (năm, số học) và Sigma (năm). method: 'sample' | 'ledoit_wolf'."""
    mu = returns.mean().to_numpy() * ppy
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(returns.to_numpy()).covariance_ * ppy
    else:
        cov = returns.cov().to_numpy() * ppy
    return mu, cov
