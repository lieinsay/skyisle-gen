"""岛群生成器的栅格工具：局部（非周期）分形噪声、连通分量、形态学、重采样、PNG 写出。全部 numpy，不引入 scipy。"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 噪声
def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


class LatticeNoise:
    """一层值噪声：格点随机值 + 平滑双线性插值，可在任意连续坐标（km）上取样（域扭曲需要）。"""

    def __init__(self, rng: np.random.Generator, x0: float, y0: float, x1: float, y1: float, cell_km: float):
        self.cell = float(cell_km)
        self.x0, self.y0 = float(x0) - 2 * self.cell, float(y0) - 2 * self.cell
        nx = int(np.ceil((x1 - self.x0) / self.cell)) + 3
        ny = int(np.ceil((y1 - self.y0) / self.cell)) + 3
        self.lat = rng.uniform(-1.0, 1.0, size=(ny, nx))

    def sample(self, x, y) -> np.ndarray:
        fx = (np.asarray(x, dtype=np.float64) - self.x0) / self.cell
        fy = (np.asarray(y, dtype=np.float64) - self.y0) / self.cell
        ny, nx = self.lat.shape
        i0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        j0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        ty = _smoothstep(np.clip(fy - i0, 0.0, 1.0))
        tx = _smoothstep(np.clip(fx - j0, 0.0, 1.0))
        v00 = self.lat[i0, j0]
        v01 = self.lat[i0, j0 + 1]
        v10 = self.lat[i0 + 1, j0]
        v11 = self.lat[i0 + 1, j0 + 1]
        return (v00 * (1 - ty) * (1 - tx) + v01 * (1 - ty) * tx
                + v10 * ty * (1 - tx) + v11 * ty * tx)


class FractalNoise:
    """多倍频值噪声，特征尺度以 km 给出（与栅格分辨率无关），归一到约 [-1, 1]。"""

    def __init__(self, rng, x0, y0, x1, y1, feature_km: float, octaves: int = 4,
                 persistence: float = 0.5, lacunarity: float = 2.0):
        self.layers: list[tuple[float, LatticeNoise]] = []
        amp, cell, total = 1.0, float(feature_km), 0.0
        for _ in range(max(1, int(octaves))):
            self.layers.append((amp, LatticeNoise(rng, x0, y0, x1, y1, cell)))
            total += amp
            amp *= persistence
            cell /= lacunarity
        self.total = total

    def sample(self, x, y) -> np.ndarray:
        out = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape, dtype=np.float64)
        for amp, layer in self.layers:
            out += amp * layer.sample(x, y)
        return out / self.total


# ---------------------------------------------------------------- 连通分量（行程 + 并查集，纯 numpy/Python，快）
def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """4/8 连通分量标号。返回 (labels[H,W]，0 = 背景，1..n)，n。"""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    if not mask.any():
        return labels, 0
    # 行程：每行的 [start, end) 与所属行
    pad = np.zeros((H, W + 1), dtype=np.int8)
    pad[:, :W] = mask
    d = np.diff(np.concatenate([np.zeros((H, 1), dtype=np.int8), pad], axis=1), axis=1)
    rows, starts = np.where(d == 1)
    _, ends = np.where(d == -1)
    n_run = rows.size
    parent = np.arange(n_run, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # 相邻两行的行程若重叠（4 连通：区间相交；8 连通：允许斜接）则合并
    slack = 1 if connectivity == 8 else 0
    row_start = np.searchsorted(rows, np.arange(H + 1))
    for r in range(1, H):
        a0, a1 = row_start[r - 1], row_start[r]
        b0, b1 = row_start[r], row_start[r + 1]
        if a1 == a0 or b1 == b0:
            continue
        ia, ib = a0, b0
        while ia < a1 and ib < b1:
            sa, ea = starts[ia], ends[ia]
            sb, eb = starts[ib], ends[ib]
            if sa < eb + slack and sb < ea + slack:
                ra, rb = find(ia), find(ib)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            if ea + slack <= eb + slack and ea < eb:
                ia += 1
            else:
                ib += 1
    roots = np.array([find(i) for i in range(n_run)], dtype=np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    run_label = (inv + 1).astype(np.int32)
    for k in range(n_run):
        labels[rows[k], starts[k]:ends[k]] = run_label[k]
    return labels, int(uniq.size)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = label_components(mask)
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


# ---------------------------------------------------------------- 形态学
def shift(a: np.ndarray, di: int, dj: int, fill):
    """整体平移（不回绕），空出的部分填 fill。"""
    out = np.full_like(a, fill)
    H, W = a.shape
    si0, si1 = max(0, di), min(H, H + di)
    sj0, sj1 = max(0, dj), min(W, W + dj)
    out[si0:si1, sj0:sj1] = a[si0 - di:si1 - di, sj0 - dj:sj1 - dj]
    return out


N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def binary_erode(mask: np.ndarray, iterations: int = 1, connectivity: int = 8) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    nb = N8 if connectivity == 8 else N4
    for _ in range(iterations):
        acc = m.copy()
        for di, dj in nb:
            acc &= shift(m, di, dj, False)
        m = acc
    return m


def binary_dilate(mask: np.ndarray, iterations: int = 1, connectivity: int = 8) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    nb = N8 if connectivity == 8 else N4
    for _ in range(iterations):
        acc = m.copy()
        for di, dj in nb:
            acc |= shift(m, di, dj, False)
        m = acc
    return m


def distance_bands(mask: np.ndarray, max_iter: int) -> np.ndarray:
    """到 mask 的近似距离（格数，八边形：4/8 邻域交替扩张，比切比雪夫圆得多），超过 max_iter 记 max_iter + 1。"""
    d = np.full(mask.shape, max_iter + 1, dtype=np.int32)
    cur = np.asarray(mask, dtype=bool)
    d[cur] = 0
    for k in range(1, max_iter + 1):
        nxt = binary_dilate(cur, 1, connectivity=4 if k % 2 else 8)
        d[nxt & ~cur] = k
        cur = nxt
        if cur.all():
            break
    return d


def nearest_propagate(seed: np.ndarray, max_iter: int, step_m: float = 1.0, within: np.ndarray | None = None):
    """从种子格向外扩张 max_iter 圈：返回 (到最近种子的倒角距离（step_m 为单位，直 1 / 斜 √2；超出记 inf），最近种子的扁平下标（无则 −1）)。
    `within` 限定只在其为真的格里扩张（例如只在本岛陆地上）。河谷剖面、河道加宽按它取最近河床的高程与河宽。"""
    H, W = seed.shape
    seed = np.asarray(seed, dtype=bool)
    dist = np.where(seed, 0.0, np.inf)
    src = np.where(seed, np.arange(H * W).reshape(H, W), -1)
    ok = np.ones((H, W), dtype=bool) if within is None else np.asarray(within, dtype=bool) | seed
    steps = [(di, dj, step_m * (math.sqrt(2.0) if di and dj else 1.0)) for di, dj in N8]
    for _ in range(int(max_iter)):
        changed = False
        for di, dj, s in steps:
            cand = shift(dist, di, dj, np.inf) + s
            better = ok & (cand < dist - 1e-9)
            if better.any():
                changed = True
                dist = np.where(better, cand, dist)
                src = np.where(better, shift(src, di, dj, -1), src)
        if not changed:
            break
    return dist, src


def window_extrema(a: np.ndarray, r: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(2r+1)² 方窗内的最大 / 最小值（只看 mask 内的格；可分离：先行后列）。用于局地起伏 = 最大 − 最小。"""
    hi = np.where(mask, a, -np.inf)
    lo = np.where(mask, a, np.inf)
    for axis in (0, 1):
        h2, l2 = hi.copy(), lo.copy()
        for k in range(1, int(r) + 1):
            for s in (k, -k):
                di, dj = (s, 0) if axis == 0 else (0, s)
                h2 = np.maximum(h2, shift(hi, di, dj, -np.inf))
                l2 = np.minimum(l2, shift(lo, di, dj, np.inf))
        hi, lo = h2, l2
    return hi, lo


# ---------------------------------------------------------------- 重采样
def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    H, W = a.shape
    Hp, Wp = (-H) % f, (-W) % f
    ap = np.pad(a, ((0, Hp), (0, Wp)), mode="edge")
    return ap.reshape(ap.shape[0] // f, f, ap.shape[1] // f, f).mean(axis=(1, 3))


def block_any(m: np.ndarray, f: int) -> np.ndarray:
    H, W = m.shape
    Hp, Wp = (-H) % f, (-W) % f
    mp = np.pad(m, ((0, Hp), (0, Wp)), mode="constant", constant_values=False)
    return mp.reshape(mp.shape[0] // f, f, mp.shape[1] // f, f).any(axis=(1, 3))


def upsample_bilinear(a: np.ndarray, f: int, H: int, W: int) -> np.ndarray:
    """粗网格（块均值，块中心为样点）→ 细网格双线性，裁到 (H, W)。"""
    h, w = a.shape
    fy = (np.arange(H) + 0.5) / f - 0.5
    fx = (np.arange(W) + 0.5) / f - 0.5
    i0 = np.clip(np.floor(fy).astype(np.int64), 0, h - 1)
    j0 = np.clip(np.floor(fx).astype(np.int64), 0, w - 1)
    i1 = np.clip(i0 + 1, 0, h - 1)
    j1 = np.clip(j0 + 1, 0, w - 1)
    ty = np.clip(fy - i0, 0.0, 1.0)[:, None]
    tx = np.clip(fx - j0, 0.0, 1.0)[None, :]
    return (a[np.ix_(i0, j0)] * (1 - ty) * (1 - tx) + a[np.ix_(i0, j1)] * (1 - ty) * tx
            + a[np.ix_(i1, j0)] * ty * (1 - tx) + a[np.ix_(i1, j1)] * ty * tx)


def laplacian(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内的 4 邻域拉普拉斯（掩膜外的邻居按本格值，即零通量边界）。"""
    out = np.zeros_like(a)
    for di, dj in N4:
        nb = shift(a, di, dj, np.nan)
        nbm = shift(mask, di, dj, False)
        out += np.where(nbm, nb - a, 0.0)
    return np.where(mask, out, 0.0)


def slope_deg(h: np.ndarray, mask: np.ndarray, res_m: float) -> np.ndarray:
    """坡度（度），中央差分，掩膜外邻居用本格值。"""
    hh = np.where(mask, h, np.nan)
    def g(di, dj):
        a = shift(hh, di, dj, np.nan)
        return np.where(np.isnan(a), hh, a)
    gx = (g(0, -1) - g(0, 1)) / (2.0 * res_m)
    gy = (g(-1, 0) - g(1, 0)) / (2.0 * res_m)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    return np.where(mask, np.nan_to_num(s), 0.0)


# ---------------------------------------------------------------- PNG 写出（无 PIL 依赖）
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    c = struct.pack(">I", len(data)) + tag + data
    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png16(path: Path, arr: np.ndarray) -> None:
    """16 位灰度 PNG。"""
    a = np.ascontiguousarray(np.clip(arr, 0, 65535).astype(">u2"))
    H, W = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(H))
    ihdr = struct.pack(">IIBBBBB", W, H, 16, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def write_png8(path: Path, arr: np.ndarray, palette: list[tuple[int, int, int]] | None = None) -> None:
    """8 位 PNG；给调色板则写索引色（类型 3），否则灰度（类型 0）。"""
    a = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    H, W = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(H))
    ctype = 3 if palette is not None else 0
    ihdr = struct.pack(">IIBBBBB", W, H, 8, ctype, 0, 0, 0)
    chunks = _png_chunk(b"IHDR", ihdr)
    if palette is not None:
        pal = bytes(sum(([int(r), int(g), int(b)] for r, g, b in palette), []))
        chunks += _png_chunk(b"PLTE", pal)
    chunks += _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunks)
