"""5.2 岛内地形：岛形（椭圆 + 域扭曲噪声 + 面积二分反解）→ 基形随岛龄 → 分形噪声 → 简化水流侵蚀 → 崖缘。

每座岛在自己的局部栅格上生成（分辨率 = 群栅格），再贴进群栅格。高程是「云带顶以上的绝对高度（m）」：
岛底 keel（≈ keel_clearance_m，薄片岛更低）→ 岸缘 rim（岸线上的地面高度，崖高 = rim − keel）→ 峰 peak。
侵蚀在 ≤ erosion_max_cells 的粗网格上做（纯 numpy + Python 循环的汇流），差值双线性回贴细网格。
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .grid import (FractalNoise, LatticeNoise, N8, binary_erode, block_any, block_mean, largest_component,
                   laplacian, shift, upsample_bilinear)

AGE_YOUNG, AGE_OLD = 0.3, 0.65


def age_class(age: float, c: dict | None = None) -> str:
    y = float(c["age_young"]) if c else AGE_YOUNG
    o = float(c["age_old"]) if c else AGE_OLD
    return "young" if age < y else ("mid" if age < o else "old")


# ---------------------------------------------------------------- 岛形
def island_shape(rng, area_km2: float, res_km: float, elong: float, theta: float, c: dict):
    """返回 (mask[H,W], inside[H,W] ∈ [0,1], X, Y)：inside 是岸线 0 → 内部 1 的平滑「向内程度」。面积误差 < 1%（离散化允许的范围内）。"""
    R = math.sqrt(area_km2 / math.pi)
    a, b = R * math.sqrt(elong), R / math.sqrt(elong)
    half = float(c["shape_pad"]) * a
    n = int(math.ceil(2 * half / res_km)) + 1
    n = max(n, 5)
    xs = (np.arange(n) - (n - 1) / 2.0) * res_km
    X, Y = np.meshgrid(xs, -xs)  # 行向下 = 南
    warp = FractalNoise(rng, -half, -half, half, half, feature_km=max(0.8 * R, 3 * res_km), octaves=2, persistence=0.5)
    warp2 = FractalNoise(rng, -half, -half, half, half, feature_km=max(0.8 * R, 3 * res_km), octaves=2, persistence=0.5)
    wamp = float(c["warp_amp"]) * R
    Xw = X + wamp * warp.sample(X, Y)
    Yw = Y + wamp * warp2.sample(X, Y)
    ct, st = math.cos(theta), math.sin(theta)
    u = (Xw * ct + Yw * st) / a
    v = (-Xw * st + Yw * ct) / b
    dome = 1.0 - (u * u + v * v)
    noise = FractalNoise(rng, -half, -half, half, half, feature_km=max(0.5 * R, 3 * res_km),
                         octaves=int(c["shape_octaves"]), persistence=0.55)
    f = dome + float(c["shape_noise_amp"]) * noise.sample(X, Y)
    target = area_km2 / (res_km * res_km)
    lo, hi = float(f.min()), float(f.max())
    best_mask, best_err = None, 1e18
    for _ in range(48):
        tau = 0.5 * (lo + hi)
        m = largest_component(f > tau)
        cnt = int(m.sum())
        err = abs(cnt - target)
        if err < best_err:
            best_mask, best_err = m, err
        if cnt > target:
            lo = tau
        else:
            hi = tau
        if err <= max(0.5, 0.002 * target):
            break
    mask = best_mask
    tau_eff = float(f[mask].min())
    inside = np.clip((f - tau_eff) / max(1e-9, float(f[mask].max()) - tau_eff), 0.0, 1.0)
    inside = np.where(mask, inside, 0.0)
    return mask, inside, X, Y


# ---------------------------------------------------------------- 基形
def _radial_gullies(rng, X, Y, cx, cy, n_lobes: int, amp: float):
    """围绕 (cx, cy) 的角向噪声 → 放射状沟谷因子 (1 − amp·g(θ))。"""
    th = np.arctan2(Y - cy, X - cx)
    lat = rng.uniform(0.0, 1.0, n_lobes)
    f = (th + np.pi) / (2 * np.pi) * n_lobes
    i0 = np.floor(f).astype(np.int64) % n_lobes
    t = f - np.floor(f)
    t = t * t * (3 - 2 * t)
    g = lat[i0] * (1 - t) + lat[(i0 + 1) % n_lobes] * t
    return 1.0 - amp * g


def base_form(rng, mask, inside, X, Y, age: float, area_km2: float, res_km: float, c: dict) -> tuple[np.ndarray, str]:
    """返回 shape ∈ [0,1]（岸缘 0 → 峰 1，未归一）与岛龄类别。"""
    R = math.sqrt(area_km2 / math.pi)
    half = float(np.abs(X).max())
    kind = age_class(age, c)
    ii, jj = np.where(mask)
    # 峰的位置：向内程度高的格里挑，偏离中心一点
    k_top = max(1, int(0.05 * ii.size))
    order = np.argsort(-inside[mask], kind="stable")[:k_top]
    pick = int(rng.integers(0, k_top))
    py, px = float(Y[ii[order[pick]], jj[order[pick]]]), float(X[ii[order[pick]], jj[order[pick]]])
    fn_ridge = FractalNoise(rng, -half, -half, half, half, feature_km=max(0.45 * R, 4 * res_km), octaves=4, persistence=0.5)
    n_oct = int(np.clip(round(math.log2(max(0.15 * R, 3 * res_km) / (2.5 * res_km))) + 1, 2, 7))
    fn_fine = FractalNoise(rng, -half, -half, half, half, feature_km=max(0.15 * R, 3 * res_km), octaves=n_oct, persistence=0.55)
    if kind == "young":
        n_cones = 2 if (area_km2 > 25.0 and rng.uniform() < 0.4) else 1
        Rc = float(c["cone_radius_rel"]) * R
        s = np.zeros_like(X)
        for k in range(n_cones):
            if k == 0:
                cx, cy = px, py
            else:
                ang = rng.uniform(-math.pi, math.pi)
                cx, cy = px + 0.45 * R * math.cos(ang), py + 0.45 * R * math.sin(ang)
            d = np.hypot(X - cx, Y - cy) / Rc
            cone = np.clip(1.0 - d, 0.0, 1.0) ** float(c["cone_profile_pow"])
            cone *= _radial_gullies(rng, X, Y, cx, cy, int(rng.integers(9, 16)), float(c["gully_amp"])) ** np.clip(d, 0, 1)
            s = np.maximum(s, cone * (1.0 if k == 0 else rng.uniform(0.55, 0.9)))
        shape = s * (0.7 + 0.3 * inside ** 0.5) + 0.06 * fn_fine.sample(X, Y)
    elif kind == "mid":
        ridged = 1.0 - np.abs(fn_ridge.sample(X, Y))
        # 主脊线：过峰、方向随机
        ang = rng.uniform(-math.pi, math.pi)
        dperp = np.abs(-(X - px) * math.sin(ang) + (Y - py) * math.cos(ang))
        ridge = np.exp(-(dperp / (0.35 * R)) ** 2)
        dist = np.hypot(X - px, Y - py) / (1.4 * R)
        core = np.clip(1.0 - dist, 0.0, 1.0)
        # 次级山峰：1–3 个高斯丘，让分水岭不止一条
        bumps = np.zeros_like(X)
        for _ in range(int(rng.integers(1, 4))):
            pk = int(rng.integers(0, ii.size))
            bx, by = float(X[ii[pk], jj[pk]]), float(Y[ii[pk], jj[pk]])
            bumps = np.maximum(bumps, rng.uniform(0.4, 0.8) * np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (0.25 * R) ** 2))
        shape = inside ** 0.35 * (0.2 + 0.45 * ridged + 0.35 * np.maximum(ridge, bumps)) * (0.55 + 0.45 * core)             + 0.05 * fn_fine.sample(X, Y)
    else:
        t = np.clip(inside / float(c["plateau_rise"]), 0.0, 1.0)
        plateau = t * t * (3 - 2 * t)
        valley = np.clip(fn_ridge.sample(X, Y), 0.0, 1.0)
        dist = np.hypot(X - px, Y - py) / (1.6 * R)
        tilt = np.clip(1.0 - dist, 0.0, 1.0)
        mounds = np.clip(fn_fine.sample(X, Y), 0.0, 1.0) * (1.0 - plateau)
        shape = plateau ** 0.5 * (0.75 + 0.25 * tilt) * (1.0 - float(c["plateau_valley_depth"]) * valley * inside) \
            + float(c["mound_amp"]) * mounds + 0.03 * fn_fine.sample(X, Y)
    shape = np.where(mask, shape, 0.0)
    return shape, kind


# ---------------------------------------------------------------- 水文核心（侵蚀与 5.3 共用）
def priority_fill(h: np.ndarray, mask: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """洼地填平（Barnes 优先泛洪）。掩膜外 = 虚空 = 出口；填后每格严格高于其下游。"""
    H, W = h.shape
    out = np.where(mask, h, -np.inf).astype(np.float64)
    closed = ~mask
    edge = mask & ~binary_erode(mask, 1)
    ii, jj = np.where(edge)
    heap = [(float(out[i, j]), int(i), int(j)) for i, j in zip(ii, jj)]
    heapq.heapify(heap)
    closed[edge] = True
    hl = out.tolist()
    cl = closed.tolist()
    res = [row[:] for row in hl]
    while heap:
        z, i, j = heapq.heappop(heap)
        for di, dj in N8:
            a, b = i + di, j + dj
            if a < 0 or b < 0 or a >= H or b >= W or cl[a][b]:
                continue
            cl[a][b] = True
            zn = hl[a][b]
            if zn < z + eps:
                zn = z + eps
            res[a][b] = zn
            heapq.heappush(heap, (zn, a, b))
    return np.where(mask, np.array(res), np.nan)


def d8(hf: np.ndarray, mask: np.ndarray, res_m: float):
    """D8 流向。返回 (recv_i, recv_j, slope)；出口格（流向虚空）recv = −1，slope 用 (h − keel 0) 的代理 = 本格向外的坡。"""
    H, W = hf.shape
    hh = np.where(mask, hf, -np.inf)
    best = np.full((H, W), -np.inf)
    ri = np.full((H, W), -1, dtype=np.int32)
    rj = np.full((H, W), -1, dtype=np.int32)
    to_void = np.zeros((H, W), dtype=bool)
    for di, dj in N8:
        nb = shift(hh, di, dj, -np.inf)
        dist = res_m * (math.sqrt(2.0) if di and dj else 1.0)
        with np.errstate(invalid="ignore"):
            drop = (hh - nb) / dist
        is_void = ~shift(mask, di, dj, False)
        # 虚空邻居：视作无限低的出口，但只在没有更低陆地邻居时才「掉出去」——这里让虚空的 drop 取本格相对岸缘的坡度代理
        drop = np.where(is_void, np.where(mask, 1e6, -np.inf), drop)
        better = drop > best
        best = np.where(better, drop, best)
        ri = np.where(better, np.clip(np.arange(H)[:, None] - di, 0, H - 1), ri)  # 注意 shift 语义：nb[i] = a[i − di]
        rj = np.where(better, np.clip(np.arange(W)[None, :] - dj, 0, W - 1), rj)
        to_void = np.where(better, is_void, to_void)
    ri = np.where(mask & ~to_void & (best > 0), ri, -1)
    rj = np.where(mask & ~to_void & (best > 0), rj, -1)
    slope = np.where(mask, np.where(to_void, np.nan, best), 0.0)
    # 出口格坡度：用最陡陆地邻居坡的中位数代替（只用于侵蚀量）
    med = float(np.nanmedian(slope[mask & ~to_void])) if (mask & ~to_void).any() else 0.05
    slope = np.where(np.isnan(slope), med, slope)
    return ri, rj, np.clip(slope, 0.0, None), to_void


def accumulate(hf: np.ndarray, mask: np.ndarray, ri: np.ndarray, rj: np.ndarray, weight=None) -> np.ndarray:
    """汇流量（格数或加权）。hf 严格递减到下游，按高度降序累加（Python 列表循环，30 万格约 0.1 s）。"""
    H, W = hf.shape
    flat = np.where(mask, hf, np.nan).ravel()
    idx = np.where(mask.ravel())[0]
    order = idx[np.argsort(-flat[idx], kind="stable")]
    recv = np.where(ri.ravel() >= 0, ri.ravel() * W + rj.ravel(), -1)
    A = (np.ones(H * W) if weight is None else np.asarray(weight, dtype=np.float64).ravel().copy())
    A[~mask.ravel()] = 0.0
    Al = A.tolist()
    rl = recv.tolist()
    for k in order.tolist():
        r = rl[k]
        if r >= 0:
            Al[r] += Al[k]
    return np.array(Al).reshape(H, W)


def erode(rng, h: np.ndarray, mask: np.ndarray, res_m: float, rounds: int, base_level: float, c: dict) -> np.ndarray:
    """简化侵蚀：河道冲刷（流量^m × 坡度）+ 热力坍塌 + 坡面扩散，每轮先把单格洼地抬到邻居之上。"""
    h = h.copy()
    kf = float(c["fluvial_k"])
    m_exp = float(c["fluvial_m"])
    talus = math.tan(math.radians(float(c["talus_deg"])))
    kd = float(c["diffusion_k"])
    cell_km2 = (res_m / 1000.0) ** 2
    relief = max(1.0, float(np.nanmax(np.where(mask, h, np.nan))) - base_level)
    area_km2 = float(mask.sum()) * cell_km2
    R_m = math.sqrt(area_km2 / math.pi) * 1000.0
    slope_ref = relief / max(R_m, res_m)          # 岛的平均坡度：起伏 / 等效半径
    a_ref = 0.1 * area_km2                        # 参考汇流面积：岛的十分之一
    for _ in range(rounds):
        # 单格洼地：抬到最低邻居之上
        hh = np.where(mask, h, np.inf)
        mn = np.full_like(hh, np.inf)
        for di, dj in N8:
            mn = np.minimum(mn, shift(hh, di, dj, np.inf))
        edge = mask & ~binary_erode(mask, 1)
        pit = mask & ~edge & (hh <= mn)
        h = np.where(pit, mn + 0.05, h)
        ri, rj, slope, _ = d8(h, mask, res_m)
        A = accumulate(h, mask, ri, rj)
        E = kf * relief * (A * cell_km2 / a_ref) ** m_exp * (slope / slope_ref)
        # 不切到下游以下（保持排水）
        recv_h = np.where(ri >= 0, h[np.clip(ri, 0, None), np.clip(rj, 0, None)], base_level)
        E = np.minimum(E, 0.6 * np.clip(h - recv_h, 0.0, None))
        h = np.where(mask, h - E, h)
        # 热力坍塌：坡度超过休止角，把超出的部分推给最低邻居
        hh = np.where(mask, h, np.nan)
        for di, dj in N8:
            nb = shift(hh, di, dj, np.nan)
            dist = res_m * (math.sqrt(2.0) if di and dj else 1.0)
            excess = (hh - nb) - talus * dist
            mv = np.where(np.isnan(excess), 0.0, np.clip(excess, 0.0, None)) * 0.25
            h = h - mv
            h = h + shift(mv, -di, -dj, 0.0)
        # 坡面扩散
        h = h + kd * laplacian(h, mask)
        h = np.where(mask, np.maximum(h, base_level), h)
    return h


def _coarse_factor(mask: np.ndarray, max_cells: int) -> int:
    return max(1, int(math.ceil(max(mask.shape) / max_cells)))


def sculpt_island(rng, mask, inside, X, Y, age: float, area_km2: float, res_km: float,
                  peak: float, rim: float, is_main: bool, c: dict) -> tuple[np.ndarray, str]:
    """一座岛的完整高程：基形 → 侵蚀（粗网格）→ 归一到 rim → peak（最高格 = peak）。返回 (h[H,W] 掩膜外 NaN, 岛龄类别)。"""
    shape, kind = base_form(rng, mask, inside, X, Y, age, area_km2, res_km, c)
    s = shape[mask]
    lo, hi = float(s.min()), float(s.max())
    shape = np.where(mask, (shape - lo) / max(1e-9, hi - lo), 0.0)
    h = rim + (peak - rim) * shape
    rounds = int(c["erosion_rounds_main"] if is_main else c["erosion_rounds_small"])
    rounds = int(round(rounds * {"young": 0.7, "mid": 1.0, "old": 1.3}[kind]))
    res_m = res_km * 1000.0
    if rounds > 0 and int(mask.sum()) >= 30:
        f = _coarse_factor(mask, int(c["erosion_max_cells"]))
        if f > 1:
            hc = block_mean(np.where(mask, h, rim), f)
            mc = block_any(mask, f)
            ec = erode(rng, hc, mc, res_m * f, rounds, rim, c)
            delta = np.where(mc, ec - hc, 0.0)
            h = h + upsample_bilinear(delta, f, mask.shape[0], mask.shape[1])
        else:
            h = erode(rng, h, mask, res_m, rounds, rim, c)
    # 岸缘不向海面收敛：只夹到 rim 以上；归一让最高格恰为 peak
    h = np.where(mask, np.maximum(h, rim), np.nan)
    hmax = float(np.nanmax(h))
    if hmax > rim + 1e-6:
        h = rim + (h - rim) * (peak - rim) / (hmax - rim)
    return h, kind
