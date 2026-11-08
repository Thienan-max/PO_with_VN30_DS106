"""Style biểu đồ thống nhất: bảng màu categorical 8 slot đã kiểm định CVD, thứ tự cố định."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

MODELS = ["Markowitz", "NSGA-III", "MAP-Elites", "CVT-MAP-Elites", "Mean-CVaR", "HRP", "Black-Litterman"]
BENCHMARKS = {"1/N (7 ngành)": INK2, "VN30-EW": MUTED}

# Màu bám theo thực thể (không theo thứ hạng)
MODEL_COLORS = {m: CATEGORICAL[i] for i, m in enumerate(MODELS)}
COLORS = {**MODEL_COLORS, **BENCHMARKS}
LINESTYLE = {"1/N (7 ngành)": "--", "VN30-EW": ":"}

DIVERGING = LinearSegmentedColormap.from_list("bl_gray_rd", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "figure.dpi": 110, "figure.autolayout": False,
    })


def color(name):
    return COLORS.get(name, MUTED)


def pct_axis(ax, axis="y", decimals=0):
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fmt(df, pct=(), dec=3, pct_dec=2):
    """Định dạng bảng để hiển thị: các cột trong `pct` dạng phần trăm, còn lại `dec` chữ số."""
    import pandas as pd
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            out[c] = df[c].map({True: "Có", False: "Không"})
        elif c in pct:
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{pct_dec}%}")
        elif pd.api.types.is_integer_dtype(df[c]):
            out[c] = df[c].map(lambda v: f"{v:,d}")
        elif pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{dec}f}")
        else:
            out[c] = df[c]
    return out
