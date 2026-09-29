"""5.2 岛内地形：岛形（椭圆 + 域扭曲噪声 + 面积二分反解）→ 基形随岛龄（新岛是拱、汇聚带的一部分大岛多核嵌合）→ 分形噪声 → 简化水流侵蚀（谷收拢）→ 崖缘。

每座岛在自己的局部栅格上生成（分辨率 = 群栅格），再贴进群栅格。高程是「云带顶以上的绝对高度（m）」：
岛底 keel（≈ keel_clearance_m，薄片岛更低）→ 岸缘 rim（岸线上的地面高度，崖高 = rim − keel）→ 峰 peak。
约束的是台面（陆地高程中位数 = ③ 的 height_m）与目标起伏（峰 − 岸缘，layout.relief_targets），岸缘与峰由拟合得出（四点十九）。
侵蚀在 ≤ erosion_max_cells 的粗网格上做（纯 numpy + Python 循环的汇流），差值双线性回贴细网格。
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .grid import (FractalNoise, LatticeNoise, N8, binary_erode, block_any, block_mean, largest_component,
                   laplacian, shift, smooth121, upsample_bilinear)

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
    if not mask.any():
        # 粗分辨率下比半格还小的礁：留最高的一格（默认 100 m 时最小的 0.3 km² 也有 30 格，碰不到这里）
        mask = np.zeros(f.shape, dtype=bool)
        mask[np.unravel_index(int(np.argmax(f)), f.shape)] = True
    tau_eff = float(f[mask].min())
    inside = np.clip((f - tau_eff) / max(1e-9, float(f[mask].max()) - tau_eff), 0.0, 1.0)
    inside = np.where(mask, inside, 0.0)
    return mask, inside, X, Y


# ---------------------------------------------------------------- 多核嵌合（P4，L18；DESIGN-NOTES 四点三十五）
def multicore_spec(rng, area_km2: float, age: float, kernel: float, btype: int, c: dict) -> dict | None:
    """这座岛是不是多核嵌合的（汇聚带的一部分大岛）：随机流 island:<节点>:cores:<岛号>（别的抽样次序不动）。
    不是返回 None；是返回 {"n": 2 或 3, "primary": 主核号, "strength": 各核强度（主核 1，小的、老的核低）, "rng": 接着抽缝的扭曲噪声}。"""
    frac = float(c.get("multicore_frac", 0.0))
    if not frac > 0.0:
        return None
    u = rng.uniform()
    min_km2 = float(c["multicore_min_km2"])
    if btype != 0 or area_km2 < min_km2 or age < float(c["age_young"]):
        return None
    p = frac * min(1.0, max(0.0, kernel / float(c["multicore_kernel_full"])))
    if not u < p:
        return None
    n = 2
    if area_km2 >= 2.0 * min_km2 and rng.uniform() < float(c["multicore_three_frac"]):
        n = 3
    primary = int(rng.integers(0, n))
    lo, hi = float(c["core_low_min"]), float(c["core_low_max"])
    strength = [1.0 if k == primary else float(rng.uniform(lo, hi)) for k in range(n)]
    return {"n": n, "primary": primary, "strength": strength, "rng": rng}


def _multicore_form(sp: dict, mask, inside, X, Y, R: float, res_km: float, half: float, px: float, py: float,
                    fn_ridge, fn_fine, c: dict) -> tuple[np.ndarray, dict]:
    """多核的造形：核摆在岛形里（离第一个挑的峰最远的一格、再离它最远的一格，三核再加离两者都远的一格），
    格归（域扭曲后）最近的核；主核是山（脊线垂直于两核连线：嵌合挤出的褶皱与缝平行），小的、老的核是朝远端缓降的平原、
    中间一道浅槽沿「主核 → 小核」的轴收水成干流；缝两边按到两核的距离差平滑过渡，缝上挤出一道脊；向岸缘收（inside^0.35）。
    返回 (shape, 核的摆放 {n, primary, sx, sy, strength, member})。C++ 版 terrain.cpp 的 multicore_form 同式。"""
    ii, jj = np.where(mask)
    Xc, Yc = X[ii, jj], Y[ii, jj]
    ins = inside[ii, jj]
    sel = ins >= float(c["core_inside_min"])
    if not sel.any():
        sel = np.ones(ii.size, dtype=bool)
    cx_, cy_ = Xc[sel], Yc[sel]

    def far_from(x0, y0):
        return int(np.argmax(np.hypot(cx_ - x0, cy_ - y0)))
    nc = int(sp["n"])
    sx, sy = [0.0] * nc, [0.0] * nc
    b = far_from(px, py)
    sx[0], sy[0] = cx_[b], cy_[b]
    b = far_from(sx[0], sy[0])
    sx[1], sy[1] = cx_[b], cy_[b]
    if nc == 3:
        b = int(np.argmax(np.minimum(np.hypot(cx_ - sx[0], cy_ - sy[0]), np.hypot(cx_ - sx[1], cy_ - sy[1]))))
        sx[2], sy[2] = cx_[b], cy_[b]
    P = int(sp["primary"])
    st = sp["strength"]
    rc = sp["rng"]
    wfeat = max(0.6 * R, 4 * res_km)
    fwx = FractalNoise(rc, -half, -half, half, half, feature_km=wfeat, octaves=3, persistence=0.5)
    fwy = FractalNoise(rc, -half, -half, half, half, feature_km=wfeat, octaves=3, persistence=0.5)
    wamp = float(c["core_seam_warp"]) * R
    crad, rw = float(c["core_radius_rel"]) * R, float(c["core_ridge_w"]) * R
    tw, tamp = float(c["core_trough_w"]), float(c["core_trough_amp"])
    blend, samp, sw = float(c["core_blend"]), float(c["core_seam_amp"]), float(c["core_seam_w"])
    j0 = 1 if P == 0 else 0
    dPx, dPy = sx[j0] - sx[P], sy[j0] - sy[P]
    lP = np.hypot(dPx, dPy)
    epx, epy = (dPx / lP, dPy / lP) if lP > 0 else (1.0, 0.0)
    ridged = 1.0 - np.abs(fn_ridge.sample(Xc, Yc))
    S = np.empty((nc, ii.size))
    for k in range(nc):
        if k == P:
            dperp = np.abs((Xc - sx[P]) * epx + (Yc - sy[P]) * epy)
            rq = dperp / rw
            ridge = np.exp(-(rq * rq))
            core = np.clip(1.0 - np.hypot(Xc - sx[P], Yc - sy[P]) / crad, 0.0, 1.0)
            S[k] = (0.2 + 0.45 * ridged + 0.35 * ridge) * (0.35 + 0.65 * core)
        else:
            ddx, ddy = sx[k] - sx[P], sy[k] - sy[P]
            ll = np.hypot(ddx, ddy)
            dl = ll if ll > 0 else 1.0
            ex, ey = (ddx / ll, ddy / ll) if ll > 0 else (1.0, 0.0)
            u = ((Xc - sx[P]) * ex + (Yc - sy[P]) * ey) / dl
            v = np.abs(-(Xc - sx[P]) * ey + (Yc - sy[P]) * ex) / R
            plain = np.clip(1.5 - u, 0.0, 1.0)
            vq = v / tw
            trough = np.exp(-(vq * vq)) * np.clip(2.0 * u - 1.0, 0.0, 1.0)
            S[k] = st[k] * (0.35 + 0.65 * plain) * (0.85 + 0.15 * ridged) - tamp * trough
    Xw = Xc + wamp * fwx.sample(Xc, Yc)
    Yw = Yc + wamp * fwy.sample(Xc, Yc)
    D = np.stack([np.hypot(Xw - sx[k], Yw - sy[k]) for k in range(nc)])
    k1 = np.argmin(D, axis=0)
    ar = np.arange(ii.size)
    d1 = D[k1, ar]
    D2 = D.copy()
    D2[k1, ar] = np.inf
    k2 = np.argmin(D2, axis=0)
    d2 = D[k2, ar]
    delta = (d2 - d1) / R
    t = np.clip(delta / blend, 0.0, 1.0)
    w = t * t * (3 - 2 * t)
    Sm = S[k1, ar] * (0.5 + 0.5 * w) + S[k2, ar] * (0.5 - 0.5 * w)
    stv = np.asarray(st, dtype=np.float64)
    dq = delta / sw
    seam = samp * (0.5 * (stv[k1] + stv[k2])) * np.exp(-(dq * dq))
    shape = np.zeros(mask.shape)
    shape[ii, jj] = ins ** 0.35 * np.maximum(Sm + seam, 0.0) + 0.05 * fn_fine.sample(Xc, Yc)
    member = np.full(mask.shape, -1, dtype=np.int8)
    member[ii, jj] = k1
    return shape, {"n": nc, "primary": P, "sx": sx, "sy": sy, "strength": list(st), "member": member}


def cores_json(cores: list[dict], gc: tuple[float, float], fl: float, res_km: float, c: dict) -> list[dict]:
    """island.json 的 islands[].cores（多核岛）：局部栅格 km → 群坐标（加局部栅格中心 gc）、浮高 fl 加进峰高，两个后端共用。
    根深 = core_root_ratio × 核上陆地高出岸缘的平均（山高根深：每个核的根在它的载荷中心正下方，浮力中心 = 重心，拼起来不歪）。"""
    ratio = float(c.get("core_root_ratio", 5.0))
    out = []
    for k, e in enumerate(cores):
        out.append({"id": k, "primary": bool(e["strength"] == 1.0), "strength": round(float(e["strength"]), 3),
                    "seed_km": [round(float(gc[0] + e["seed"][0]), 3), round(float(gc[1] + e["seed"][1]), 3)],
                    "center_km": [round(float(gc[0] + e["load_xy"][0]), 3), round(float(gc[1] + e["load_xy"][1]), 3)],
                    "area_km2": round(float(e["cells"]) * res_km * res_km, 3),
                    "peak_m": round(float(e["peak"]) + fl, 1),
                    "mean_above_rim_m": round(float(e["mean_above"]), 1),
                    "root_depth_m": round(ratio * float(e["mean_above"]), 1)})
    return out


# ---------------------------------------------------------------- 基形
def base_form(rng, mask, inside, X, Y, age: float, area_km2: float, res_km: float, c: dict, cores: dict | None = None,
              layout: dict | None = None) -> tuple[np.ndarray, str]:
    """返回 shape ∈ [0,1]（岸缘 0 → 峰 1，未归一）与岛龄类别。cores（multicore_spec 的结果）给了且不是新岛：多核的造形，
    核的摆放写进 layout（dict，就地填）。"""
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
    if cores is not None and kind != "young":
        shape, lay = _multicore_form(cores, mask, inside, X, Y, R, res_km, half, px, py, fn_ridge, fn_fine, c)
        if layout is not None:
            layout.update(lay)
    elif kind == "young":
        # 新岛成拱（P4，L17）：从海底挣脱出来的一整块，中间最厚、四周最薄——绕峰的抛物穹与岛形自己的「向内程度」（也是穹形，贴着轮廓）混合；
        # 表面是抬起来的海底岩层，平整，细噪声小；没有火山锥与放射状的熔岩沟（沟由下切自己切出来）
        Ra = float(c["arch_radius_rel"]) * R
        mix = float(c["arch_inside_mix"])
        d = np.hypot(X - px, Y - py) / Ra
        radial = np.clip(1.0 - d * d, 0.0, 1.0)
        shape = (1.0 - mix) * radial + mix * inside + float(c["arch_noise_amp"]) * fn_fine.sample(X, Y)
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


def fill_iter(h: np.ndarray, mask: np.ndarray, iters: int, eps: float = 0.01) -> np.ndarray:
    """Planchon–Darboux 迭代填洼（向量化，近似）：W 从岸缘向内收敛到 max(h, min 邻 W + eps)。iters 轮能填直径 ≤ iters 格的洼地。"""
    edge = mask & ~binary_erode(mask, 1)
    hh = np.where(mask, h, np.inf)
    Wf = np.where(edge, hh, np.inf)
    for _ in range(iters):
        mn = np.full_like(Wf, np.inf)
        for di, dj in N8:
            mn = np.minimum(mn, shift(Wf, di, dj, np.inf))
        new = np.where(edge, hh, np.maximum(hh, mn + eps))
        if np.array_equal(new, Wf):
            break
        Wf = new
    return np.where(mask & np.isfinite(Wf), Wf, np.where(mask, h, np.nan))


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


def d8_random(hf: np.ndarray, mask: np.ndarray, res_m: float, rng, p: float):
    """随机流向：每格在下坡邻格里按 坡降^p 加权抽一个作下游（p 大 → 接近最陡下降）。期望流向 = 真实梯度方向，
    不像 D8 那样在 30° 的坡上固定走「东、东北」交替的阶梯——多轮下切叠起来，沟谷不再横平竖直。
    返回 (recv_i, recv_j, to_void)，口径同 d8：邻接虚空的格是出口（recv = −1、to_void 真）；没有下坡邻格的也是 −1。"""
    H, W = hf.shape
    hh = np.where(mask, hf, np.inf)
    w = np.zeros((8, H, W))
    to_void = np.zeros((H, W), dtype=bool)
    for n, (di, dj) in enumerate(N8):
        nb = shift(hh, di, dj, np.inf)
        is_void = ~shift(mask, di, dj, False)
        to_void |= mask & is_void
        dist = res_m * (math.sqrt(2.0) if di and dj else 1.0)
        with np.errstate(invalid="ignore"):
            drop = (hh - nb) / dist
        w[n] = np.where(mask & ~is_void & (drop > 0), np.power(np.maximum(drop, 0.0), p), 0.0)
    S = w.sum(axis=0)
    u = rng.uniform(0.0, 1.0, (H, W)) * S
    k = np.minimum((np.cumsum(w, axis=0) <= u[None]).sum(axis=0), 7)   # 反查累积分布：第一个累积和 > u 的方向
    dis = np.array([d[0] for d in N8])[k]
    djs = np.array([d[1] for d in N8])[k]
    ok = mask & ~to_void & (S > 0)
    ri = np.where(ok, np.arange(H)[:, None] - dis, -1).astype(np.int32)   # shift 语义：nb[i] = a[i − di]
    rj = np.where(ok, np.arange(W)[None, :] - djs, -1).astype(np.int32)
    return ri, rj, to_void


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


def erode(rng, h: np.ndarray, mask: np.ndarray, res_m: float, rounds: int, base_level: float, c: dict,
          uplift: np.ndarray | None = None, jitter: np.ndarray | None = None) -> np.ndarray:
    """侵蚀：隐式河流功率下切（Braun & Willett 2013，n = 1）+ 抬升 + 热力坍塌 + 坡面扩散。
    先对整个网格精确填洼一次（基形里的封闭盆地否则永远排不出去，最后被 5.3 填成一块死平的台面）；
    每轮：抬升（uplift，m/轮；按基形分布，维持山体）→ 迭代填洼定流向（路由面 = 高程 + jitter）→ 随机流向 d8_random（carve_route_p = 0 时纯 D8）→ 汇流 A（km²）→
    按路由面升序（先下游后上游）隐式更新 h_i ← (h_i + F·h_r) / (1 + F)，F = carve_k · A^carve_m / 距离（km），无条件稳定；
    只有汇流 ≥ carve_a0_km2 的河道格下切，坡面靠热力坍塌跟着变陡；流向虚空的出口格以岸缘为下游（河从崖缘跌下）。
    结果最后整体仿射拟合到目标起伏（sculpt_island），所以这几个数只决定「切得多碎」，不决定绝对高度。"""
    h = np.where(mask, priority_fill(h, mask, eps=1e-3), h)
    K = float(c["carve_k"])
    m_exp = float(c["carve_m"])
    talus = math.tan(math.radians(float(c["talus_deg"])))
    kd = float(c["diffusion_k"])
    cell_km2 = (res_m / 1000.0) ** 2
    H, W = h.shape
    jit = np.zeros((H, W)) if jitter is None else jitter
    for _ in range(rounds):
        if uplift is not None:
            h = np.where(mask, h + uplift, h)
        # 保持排水：把洼地填到出口高度（粗网格上的近似填洼；水流沿填后的面走，下切作用在实际高程上）
        hf = fill_iter(h + jit, mask, int(c["fill_iters"]))
        h = np.where(mask, np.maximum(h, hf - jit - float(c["pit_keep_m"])), h)
        if rng is not None and float(c["carve_route_p"]) > 0:
            ri, rj, to_void = d8_random(hf, mask, res_m, rng, float(c["carve_route_p"]))
        else:
            ri, rj, _, to_void = d8(hf, mask, res_m)
        A = accumulate(hf, mask, ri, rj) * cell_km2
        diag = (ri >= 0) & (ri != np.arange(H)[:, None]) & (rj != np.arange(W)[None, :])
        dist_km = np.where(diag, math.sqrt(2.0), 1.0) * res_m / 1000.0
        # 只有汇流 ≥ carve_a0_km2 的格是河道、会下切；坡面只靠下面的热力坍塌跟着变陡——山脊保留基形高度，沟谷切进去
        F = np.where(A >= float(c["carve_a0_km2"]), K * np.power(np.maximum(A, cell_km2), m_exp) / dist_km, 0.0).ravel().tolist()
        recv = np.where(ri >= 0, ri * W + rj, np.where(mask & to_void, -2, -1)).ravel().tolist()
        idx = np.where(mask.ravel())[0]
        order = idx[np.argsort(np.where(mask, hf, np.inf).ravel()[idx], kind="stable")].tolist()
        hl = h.ravel().tolist()
        for k in order:
            r = recv[k]
            if r == -1:
                continue
            hr = base_level if r == -2 else hl[r]
            hk = hl[k]
            if hk > hr:
                f = F[k]
                hl[k] = (hk + f * hr) / (1.0 + f)
        h = np.array(hl).reshape(H, W)
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


def fit_rim(surface: float, relief: float, median_frac: float, rim_min: float) -> tuple[float, float]:
    """(岸缘, 起伏)：使 岸缘 + 起伏 × median_frac = 台面。岸缘低于 rim_min（岛底 + 最小崖高）时抬到 rim_min、压起伏，台面不动。"""
    if median_frac <= 1e-3:
        return max(surface, rim_min), relief
    rim = surface - median_frac * relief
    if rim < rim_min:
        rim = rim_min
        relief = max(1.0, (surface - rim_min) / median_frac)
    return rim, relief


def sculpt_island(rng, mask, inside, X, Y, age: float, area_km2: float, res_km: float,
                  surface: float, relief: float, rim_min: float, is_main: bool, c: dict, cores: dict | None = None,
                  core_out: list | None = None) -> tuple[np.ndarray, str, float, float]:
    """一座岛的完整高程：基形 → 测高曲线（幂次）→ 侵蚀（粗网格）→ 仿射拟合（陆地中位 = surface，峰 − 岸缘 = relief）。
    返回 (h[H,W] 掩膜外 NaN, 岛龄类别, 岸缘, 峰)。台面太低装不下 relief 时岸缘停在 rim_min、起伏按比例压小。
    cores（multicore_spec）给了：多核的造形；core_out（list）给了就填每个核拟合后的原始数（局部栅格 km，cores_json 换成 island.json 的形）。"""
    lay = {}
    shape, kind = base_form(rng, mask, inside, X, Y, age, area_km2, res_km, c, cores, lay)
    s = shape[mask]
    lo, hi = float(s.min()), float(s.max())
    shape = np.where(mask, (shape - lo) / max(1e-9, hi - lo), 0.0)
    # 测高曲线：u → u^γ 不改排序，中位分位 m → m^γ，解 γ 让它等于该岛龄的目标分位（大半是低地、山集中在中央）。
    # 台面离岛底太近、装不下目标起伏时，先把分位压低（最低 median_frac_min：一圈缓坡低地托一座陡峰），还不够才压起伏
    m_t = float(c["median_frac_arch" if kind == "young" else f"median_frac_{kind}"])
    # 新岛是拱：分位最低只压到 median_frac_arch_min（压成一圈低地托一座尖峰就不是拱了），装不下就压起伏——拱矮一些
    m_min = float(c["median_frac_arch_min"] if kind == "young" else c["median_frac_min"])
    m_t = min(m_t, max(m_min, (surface - rim_min) / max(1.0, relief)))
    m_raw = float(np.median(shape[mask]))
    if 1e-3 < m_raw < 0.999:
        shape = shape ** float(np.clip(math.log(m_t) / math.log(m_raw), 0.7, 3.5))
    rim, R = fit_rim(surface, relief, m_t, rim_min)
    h = rim + R * shape
    rounds = int(c["carve_rounds_main"] if is_main else c["carve_rounds_small"])
    rounds = int(round(rounds * {"young": 0.7, "mid": 1.0, "old": 1.3}[kind]))
    res_m = res_km * 1000.0
    up = float(c["uplift_rel"]) * R
    if rounds > 0 and int(mask.sum()) >= 30:
        f = _coarse_factor(mask, int(c["erosion_max_cells"]))
        half = float(np.abs(X).max())
        jit = float(c["carve_jitter_rel"]) * R * FractalNoise(rng, -half, -half, half, half, feature_km=3.0 * res_km * f,
                                                             octaves=2, persistence=0.5).sample(X, Y)
        if f > 1:
            hc = block_mean(np.where(mask, h, rim), f)
            mc = block_any(mask, f)
            ec = erode(rng, hc, mc, res_m * f, rounds, rim, c, uplift=up * block_mean(shape, f), jitter=block_mean(jit, f))
            # 下切量回贴：粗网格一格宽的沟双线性放大后是沿坐标轴的「方锥坑」——粗网格平滑 carve_smooth_coarse 遍、放大后细网格再平滑 carve_smooth_fine 遍
            delta = smooth121(np.where(mc, ec - hc, 0.0), mc, int(c["carve_smooth_coarse"]))
            up_d = upsample_bilinear(delta, f, mask.shape[0], mask.shape[1])
            h = h + smooth121(up_d, mask, int(c["carve_smooth_fine"]))
        else:
            h = erode(rng, h, mask, res_m, rounds, rim, c, uplift=up * shape, jitter=jit)
    # 岸缘不向海面收敛：只夹到 rim 以上；再仿射拟合：u = (h − rim)/(max − rim)，中位分位 m_u → 岸缘 = 台面 − m_u × 起伏
    h = np.where(mask, np.maximum(h, rim), np.nan)
    hmax = float(np.nanmax(h))
    u = np.clip((h - rim) / max(1e-6, hmax - rim), 0.0, 1.0)
    rim2, R2 = fit_rim(surface, relief, float(np.median(u[mask])), rim_min)
    h = np.where(mask, rim2 + R2 * u, np.nan)
    if core_out is not None and lay:
        # 多核：每个核的载荷（高出岸缘的量）与载荷中心——根就在它正下方
        for k in range(int(lay["n"])):
            sel = mask & (lay["member"] == k)
            a = h[sel] - rim2
            n_c = int(a.size)
            load = float(np.sum(a))
            ok = load > 0.0
            core_out.append({"seed": (float(lay["sx"][k]), float(lay["sy"][k])), "strength": float(lay["strength"][k]), "cells": n_c,
                             "peak": float(np.max(h[sel])) if n_c else float(rim2), "load": load,
                             "load_xy": ((float(np.sum(a * X[sel])) / load, float(np.sum(a * Y[sel])) / load) if ok
                                         else (float(lay["sx"][k]), float(lay["sy"][k]))),
                             "mean_above": load / n_c if n_c else 0.0})
    return h, kind, rim2, rim2 + R2
