"""
Mô hình 3 & 4 - MAP-Elites (Mouret & Clune 2015) và CVT-MAP-Elites (Vassiliades et al. 2018).

Quality-Diversity: thay vì tìm 1 nghiệm tốt nhất, thuật toán điền một "bản đồ" các nghiệm
ưu tú (elites), mỗi ô ứng với một kiểu hành vi (behavior descriptor - BD) khác nhau.

- Fitness: Sharpe ratio (năm hóa).
- BD mặc định (2 chiều): độ biến động năm (volatility) và Effective Number of Bets (ENB = 1/Σw²).
- Grid  : chia đều từng chiều BD thành lưới -> số ô tăng theo cấp số mũ khi tăng số chiều BD.
- CVT   : Centroidal Voronoi Tessellation (k-means) -> số niche cố định k, không phụ thuộc số chiều.
          sampling='feasible' đặt centroid theo phân bố BD của danh mục khả thi (không phí niche vào
          vùng BD không thể đạt, vd. ENB=7 nhưng vol cao); 'uniform' là phiên bản gốc của bài báo.
- Biến dị: Iso+LineDD  w' = w1 + σ_iso·N(0,I) + σ_line·N(0,1)·(w2 - w1), chiếu lại lên simplex.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import markowitz
from .nsga3 import batch_cvar, batch_mdd


# ---------------------------------------------------------------------------
# Đánh giá danh mục
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, R, mu, cov, rf=0.0, descriptors=("volatility", "enb")):
        self.R, self.mu, self.cov, self.rf = np.asarray(R), mu, cov, rf
        self.descriptors = tuple(descriptors)
        self.n_eval = 0

    def __call__(self, W):
        self.n_eval += len(W)
        ret = W @ self.mu
        vol = np.sqrt(np.einsum("ij,jk,ik->i", W, self.cov, W))
        fit = (ret - self.rf) / vol
        bd = []
        for d in self.descriptors:
            if d == "volatility":
                bd.append(vol)
            elif d == "enb":
                bd.append(1.0 / (W ** 2).sum(axis=1))
            elif d == "mdd":
                bd.append(batch_mdd(self.R, W))
            elif d == "cvar":
                bd.append(batch_cvar(self.R, W))
            elif d == "max_weight":
                bd.append(W.max(axis=1))
            else:
                raise ValueError(d)
        return fit, np.column_stack(bd)


def random_portfolios(n, n_assets, rng):
    """Dirichlet với α ngẫu nhiên log-uniform [0.05, 5]: phủ cả danh mục tập trung lẫn đa dạng."""
    alpha = np.exp(rng.uniform(np.log(0.05), np.log(5), size=(n, 1)))
    G = rng.gamma(np.broadcast_to(alpha, (n, n_assets)))
    G = np.maximum(G, 1e-300)
    return G / G.sum(axis=1, keepdims=True)


def iso_line_dd(P1, P2, rng, sigma_iso=0.01, sigma_line=0.2):
    W = P1 + sigma_iso * rng.standard_normal(P1.shape) + sigma_line * rng.standard_normal((len(P1), 1)) * (P2 - P1)
    W = np.clip(W, 0, None)
    s = W.sum(axis=1, keepdims=True)
    return np.where(s > 0, W / np.where(s > 0, s, 1), 1 / W.shape[1])


def descriptor_bounds(evaluator: Evaluator, rng, n_sample=20000):
    """Biên của không gian BD, ước lượng từ danh mục ngẫu nhiên + các góc simplex + Min-Variance."""
    n = len(evaluator.mu)
    W = np.vstack([random_portfolios(n_sample, n, rng), np.eye(n),
                   markowitz.min_variance(evaluator.cov)[None, :], np.full((1, n), 1 / n)])
    _, bd = evaluator(W)
    evaluator.n_eval -= len(W)  # không tính vào ngân sách tối ưu
    lo, hi = bd.min(axis=0), bd.max(axis=0)
    for j, d in enumerate(evaluator.descriptors):
        if d == "enb":
            lo[j], hi[j] = 1.0, float(n)
    pad = 1e-9 + 0.01 * (hi - lo)
    return lo - pad, hi + pad, bd


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
class GridArchive:
    def __init__(self, lo, hi, bins_per_dim=20):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.bins = np.full(len(lo), bins_per_dim) if np.isscalar(bins_per_dim) else np.asarray(bins_per_dim)
        self.n_cells = int(np.prod(self.bins))

    def index(self, bd):
        z = (bd - self.lo) / (self.hi - self.lo)
        ij = np.clip((z * self.bins).astype(int), 0, self.bins - 1)
        return np.ravel_multi_index(ij.T, self.bins)

    def cell_centers(self):
        grids = [self.lo[d] + (np.arange(b) + 0.5) * (self.hi[d] - self.lo[d]) / b for d, b in enumerate(self.bins)]
        mesh = np.meshgrid(*grids, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])


class CVTArchive:
    def __init__(self, lo, hi, n_niches=400, samples=None, n_samples=50000, seed=0):
        from sklearn.cluster import KMeans
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        rng = np.random.default_rng(seed)
        if samples is None:  # CVT gốc: lấy mẫu đều trong hộp BD
            Z = rng.uniform(size=(n_samples, len(lo)))
        else:                # CVT theo vùng khả thi (lấy mẫu con để k-means nhanh)
            Z = self._norm(samples)
            if len(Z) > 5000:
                Z = Z[rng.choice(len(Z), 5000, replace=False)]
        km = KMeans(n_clusters=n_niches, n_init=1, random_state=seed).fit(Z)
        self.centroids = km.cluster_centers_
        self.n_cells = n_niches

    def _norm(self, bd):
        return (bd - self.lo) / (self.hi - self.lo)

    def index(self, bd):
        Z = self._norm(bd)
        d2 = (Z ** 2).sum(1)[:, None] - 2 * Z @ self.centroids.T + (self.centroids ** 2).sum(1)[None, :]
        return np.argmin(d2, axis=1)

    def cell_centers(self):
        return self.lo + self.centroids * (self.hi - self.lo)


# ---------------------------------------------------------------------------
# Vòng lặp MAP-Elites (dùng chung cho Grid và CVT)
# ---------------------------------------------------------------------------
@dataclass
class QDResult:
    W: np.ndarray          # (n_cells x N), NaN ở ô trống
    fitness: np.ndarray    # (n_cells,), -inf ở ô trống
    bd: np.ndarray         # (n_cells x D)
    archive: object
    n_eval: int
    history: dict = field(default_factory=dict)

    @property
    def filled(self):
        return np.isfinite(self.fitness)

    def coverage(self):
        return float(self.filled.mean())

    def qd_score(self, offset):
        return float((self.fitness[self.filled] - offset).sum())

    def elites(self):
        m = self.filled
        return self.W[m], self.fitness[m], self.bd[m]

    def select(self, min_enb=None, max_vol=None):
        """Elite có Sharpe cao nhất, có thể kèm ràng buộc hậu kiểm trên BD (lợi thế của QD)."""
        W, f, bd = self.elites()
        ok = np.ones(len(f), dtype=bool)
        if min_enb is not None:
            ok &= 1.0 / (W ** 2).sum(1) >= min_enb
        if max_vol is not None:
            ok &= bd[:, self.history["descriptors"].index("volatility")] <= max_vol
        if not ok.any():
            ok[:] = True
        return W[ok][int(np.argmax(f[ok]))]


def _insert(archive, Wa, Fa, Ba, W, f, bd):
    cells = archive.index(bd)
    order = np.lexsort((-f, cells))           # theo ô, fitness giảm dần
    cells, W, f, bd = cells[order], W[order], f[order], bd[order]
    first = np.r_[True, cells[1:] != cells[:-1]]
    cells, W, f, bd = cells[first], W[first], f[first], bd[first]
    better = f > Fa[cells]
    c = cells[better]
    Wa[c], Fa[c], Ba[c] = W[better], f[better], bd[better]
    return int(better.sum())


def run_map_elites(evaluator: Evaluator, variant="grid", bins_per_dim=20, n_niches=200,
                   cvt_sampling="feasible", n_init=1000, n_iter=200, batch=128,
                   sigma_iso=0.01, sigma_line=0.2, seed=0, log_every=10) -> QDResult:
    rng = np.random.default_rng(seed)
    n = len(evaluator.mu)
    lo, hi, bd_samples = descriptor_bounds(evaluator, rng)
    if variant == "grid":
        archive = GridArchive(lo, hi, bins_per_dim)
    elif variant == "cvt":
        archive = CVTArchive(lo, hi, n_niches, samples=bd_samples if cvt_sampling == "feasible" else None, seed=seed)
    else:
        raise ValueError(variant)

    Wa = np.full((archive.n_cells, n), np.nan)
    Fa = np.full(archive.n_cells, -np.inf)
    Ba = np.full((archive.n_cells, len(evaluator.descriptors)), np.nan)
    hist = {"evals": [], "coverage": [], "best": [], "qd_score": [], "descriptors": evaluator.descriptors}

    W0 = np.vstack([random_portfolios(n_init, n, rng), np.eye(n), np.full((1, n), 1 / n)])
    f0, b0 = evaluator(W0)
    _insert(archive, Wa, Fa, Ba, W0, f0, b0)
    offset = float(f0.min())
    hist["offset"] = offset

    for it in range(n_iter):
        filled = np.flatnonzero(np.isfinite(Fa))
        p1 = Wa[rng.choice(filled, batch)]
        p2 = Wa[rng.choice(filled, batch)]
        W = iso_line_dd(p1, p2, rng, sigma_iso, sigma_line)
        f, bd = evaluator(W)
        _insert(archive, Wa, Fa, Ba, W, f, bd)
        if it % log_every == 0 or it == n_iter - 1:
            m = np.isfinite(Fa)
            hist["evals"].append(evaluator.n_eval)
            hist["coverage"].append(m.mean())
            hist["best"].append(Fa[m].max())
            hist["qd_score"].append((Fa[m] - offset).sum())

    return QDResult(Wa, Fa, Ba, archive, evaluator.n_eval, hist)


def map_elites_weights(R, mu, cov, rf=0.0, variant="grid", min_enb=None, **kw) -> np.ndarray:
    ev = Evaluator(R, mu, cov, rf)
    return run_map_elites(ev, variant=variant, **kw).select(min_enb=min_enb)
