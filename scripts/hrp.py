"""
Mô hình 6 - Hierarchical Risk Parity (López de Prado 2016).

Không cần nghịch đảo ma trận hiệp phương sai và không dùng μ (tránh sai số ước lượng lợi nhuận).
1. Tree clustering: khoảng cách d_ij = sqrt((1 - ρ_ij)/2), liên kết phân cấp.
2. Quasi-diagonalization: sắp xếp tài sản theo thứ tự lá của dendrogram.
3. Recursive bisection: chia đôi danh sách, phân bổ tỷ trọng tỉ lệ nghịch với phương sai cụm
   (phương sai cụm tính bằng danh mục inverse-variance bên trong cụm).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr):
    d = np.sqrt(np.clip((1 - np.asarray(corr)) / 2, 0, None))
    np.fill_diagonal(d, 0)
    return d


def _cluster_var(cov, idx):
    c = cov[np.ix_(idx, idx)]
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def hrp_weights(cov, corr=None, method="single", return_tree=False):
    cov = np.asarray(cov)
    if corr is None:
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
    link = linkage(squareform(correlation_distance(corr), checks=False), method=method)
    order = list(leaves_list(link))
    w = np.ones(len(cov))
    clusters = [order]
    while clusters:
        clusters = [c[s] for c in clusters for s in (slice(0, len(c) // 2), slice(len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            a = 1 - v_l / (v_l + v_r)
            w[left] *= a
            w[right] *= 1 - a
    w /= w.sum()
    return (w, link, order) if return_tree else w
