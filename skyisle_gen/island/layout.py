"""5.1 群内布局：岛数、大小（Zipf）、位置（泊松盘 + 主岛引力 + 板块走向）、台面高度与目标起伏、索桥 / 短渡、导水槽网络。

只依赖行星产物里这个节点的标量（area_km2、main_area_km2、height_m、layered、age、板块边界核与类型），
随机数来自 rng.entity_rng(seed, ISLAND_STREAM, f"island:{node}:layout")。
位置用岛心的局部平面坐标（km，x 东 y 北），岛与岛之间的间距按各岛的角向半径剖面（由 5.2 的岛形给出）计算。
"""
from __future__ import annotations

import math

import numpy as np

CONVERGENT, DIVERGENT, TRANSFORM = 0, 1, 2


def island_count(rng, area_km2: float, area_median: float, c: dict) -> int:
    n = float(c["n0"]) * (area_km2 / max(area_median, 1e-9)) ** float(c["n_exp"])
    n *= math.exp(rng.normal(0.0, float(c["n_sigma"])))
    return int(np.clip(round(n), int(c["n_min"]), int(c["n_max"])))


def zipf_sizes(area_km2: float, main_km2: float, n: int, c: dict) -> np.ndarray:
    """主岛 = main_km2；其余 n−1 座按 Zipf 分 area − main。总和严格等于 area_km2。
    约束：其余最大者 ≤ 0.95 主岛（主岛必须是最大的岛），最小者 ≥ min_islet_km2（做不到就减岛数）。"""
    rest = max(0.0, area_km2 - main_km2)
    min_islet = float(c["min_islet_km2"])
    e = float(c["zipf_exp"])
    if rest < min_islet or n < 2:
        return np.array([area_km2], dtype=np.float64)
    while n > 2:
        ee = e
        sizes = None
        while ee >= 0.0:
            k = np.arange(1, n, dtype=np.float64)
            w = k ** (-ee)
            s = rest * w / w.sum()
            if s[0] <= 0.95 * main_km2:
                sizes = s
                break
            ee -= 0.1
        if sizes is None:
            n += 1  # 平均都比主岛大：只能再多切几座
            if n > 4 * int(c["n_max"]):
                break
            continue
        if sizes[-1] >= min_islet:
            return np.concatenate([[main_km2], sizes])
        n -= 1
    return np.array([main_km2, rest], dtype=np.float64)


def boundary_axis(plates: dict, lat: float, lon: float) -> tuple[float, float, int]:
    """板块边界在该点的走向（弧度，x 东为 0 逆时针）、边界核强度 [0,1]、边界类型。走向 = 核梯度的垂线。"""
    lats, lons = plates["lats"], plates["lons"]
    K = plates["boundary_kernel"].astype(np.float64)
    from ..sphere import grid_interp
    d = 0.75
    kx = (grid_interp(K, lats, lons, lat, lon + d) - grid_interp(K, lats, lons, lat, lon - d)) / (2 * d * max(math.cos(math.radians(lat)), 0.1))
    ky = (grid_interp(K, lats, lons, min(lat + d, 89.9), lon) - grid_interp(K, lats, lons, max(lat - d, -89.9), lon)) / (2 * d)
    k0 = float(grid_interp(K, lats, lons, lat, lon))
    ii = int(np.clip(np.searchsorted(lats, lat), 0, lats.size - 1))
    jj = int(np.clip(np.searchsorted(lons, ((lon + 180.0) % 360.0) - 180.0), 0, lons.size - 1))
    btype = int(plates["btype"][ii, jj])
    if math.hypot(kx, ky) < 1e-6:
        return 0.0, 0.0, btype
    return math.atan2(kx, -ky), float(np.clip(k0, 0.0, 1.0)), btype  # 垂直于梯度


def radial_profile(mask: np.ndarray, res_km: float, nbins: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """岛形的角向最大半径剖面 r(θ)（km，θ 以岛心为原点，x 东为 0）。返回 (center_xy_km 相对栅格中心, r[nbins])。"""
    ii, jj = np.where(mask)
    H, W = mask.shape
    cy, cx = ii.mean(), jj.mean()
    from .grid import binary_erode
    edge = mask & ~binary_erode(mask, 1)
    ei, ej = np.where(edge)
    dx = (ej - cx) * res_km
    dy = -(ei - cy) * res_km   # 行向下 = 南
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    b = ((th + np.pi) / (2 * np.pi) * nbins).astype(np.int64) % nbins
    prof = np.zeros(nbins)
    np.maximum.at(prof, b, r)
    # 空桶用邻桶补
    for _ in range(nbins):
        z = prof <= 0
        if not z.any():
            break
        prof = np.where(z, np.maximum(np.roll(prof, 1), np.roll(prof, -1)), prof)
    prof = np.maximum(prof, res_km)
    center = np.array([(cx - (W - 1) / 2.0) * res_km, -(cy - (H - 1) / 2.0) * res_km])
    return center, prof


def _r_at(prof: np.ndarray, theta: float) -> float:
    n = prof.size
    f = ((theta + math.pi) / (2 * math.pi) * n) % n
    i0 = int(math.floor(f)) % n
    t = f - math.floor(f)
    return float(prof[i0] * (1 - t) + prof[(i0 + 1) % n] * t)


def place_islands(rng, profiles: list[np.ndarray], sizes: np.ndarray, axis: float, kernel: float,
                  btype: int, c: dict) -> tuple[np.ndarray, dict]:
    """依次放置（主岛在原点）。返回 centers[n, 2]（km）。
    汇聚带：沿边界走向拉长（各向异性 a）；离散带：更散（间距放大）；走滑：略拉长。"""
    n = len(profiles)
    centers = np.zeros((n, 2))
    gap_lo, gap_hi = float(c["gap_min_km"]), float(c["gap_max_km"])
    if btype == CONVERGENT:
        aniso = 1.0 + (float(c["arc_elongation"]) - 1.0) * kernel
        spread = 1.0
    elif btype == DIVERGENT:
        aniso = 1.0
        spread = 1.0 + (float(c["divergent_spread"]) - 1.0) * kernel
    else:
        aniso = 1.0 + 0.5 * (float(c["arc_elongation"]) - 1.0) * kernel
        spread = 1.0
    ca, sa = math.cos(axis), math.sin(axis)
    weights = np.sqrt(sizes)
    weights[0] *= float(c["main_gravity"])

    def gap_ok(k, pos):
        worst = 1e9
        for j in range(k):
            d = pos - centers[j]
            dist = math.hypot(d[0], d[1])
            th = math.atan2(d[1], d[0])
            g = dist - _r_at(profiles[j], th) - _r_at(profiles[k], th + math.pi)
            worst = min(worst, g)
        return worst

    stats = {"gaps": []}
    for k in range(1, n):
        w = weights[:k] / weights[:k].sum()
        best, best_gap = None, -1e9
        for attempt in range(int(c["place_attempts"])):
            anchor = int(rng.choice(k, p=w))
            phi = rng.uniform(-math.pi, math.pi)
            vx, vy = math.cos(phi), math.sin(phi) / aniso
            nv = math.hypot(vx, vy)
            vx, vy = vx / nv, vy / nv
            dx, dy = vx * ca - vy * sa, vx * sa + vy * ca
            th = math.atan2(dy, dx)
            gap = (gap_lo + (gap_hi - gap_lo) * rng.beta(1.3, 2.2)) * spread
            gap = min(gap, gap_hi) if spread <= 1.0 else min(gap, gap_hi * spread)
            gap *= 1.0 + 0.15 * (attempt // 40)   # 放不下就慢慢拉开
            dist = _r_at(profiles[anchor], th) + gap + _r_at(profiles[k], th + math.pi)
            pos = centers[anchor] + np.array([dx, dy]) * dist
            g = gap_ok(k, pos)
            if g >= gap_lo:
                best, best_gap = pos, g
                break
            if g > best_gap:
                best, best_gap = pos, g
        centers[k] = best
        stats["gaps"].append(float(best_gap))
    return centers, stats


def surface_heights(rng, n: int, height_m: float, layered: bool, c: dict) -> np.ndarray:
    """各岛台面高度（陆地高程中位数，云带顶以上 m）：主岛 = height_m（③ 的口径，④ 的岛上气温就在这个高度），
    其余 = height_m × U(surface_lo, surface_hi)；叠层群再 ± U(0.5, 1) × layered_spread_m。抽样次序与旧 peak_heights 相同，布局不变。"""
    surf = np.empty(n)
    surf[0] = height_m
    if n > 1:
        surf[1:] = height_m * rng.uniform(float(c["surface_lo"]), float(c["surface_hi"]), n - 1)
        if layered:
            sgn = rng.choice([-1.0, 1.0], n - 1)
            surf[1:] += sgn * float(c["layered_spread_m"]) * rng.uniform(0.5, 1.0, n - 1)
    return np.maximum(surf, 50.0)


def relief_targets(rng, sizes: np.ndarray, ages: np.ndarray, c: dict) -> np.ndarray:
    """各岛目标起伏（岸缘 → 峰，m）：参考起伏按岛龄在 relief_young_m（岛龄 0）与 relief_old_m（岛龄 1）之间对数插值，
    × (面积 / 1000 km²)^relief_area_exp × 对数正态(relief_sigma)，夹 [relief_min_m, relief_max_m]。
    锚点是现实的岛：1000 km² 上下的新火山岛 2000 m 级（特内里费、济州、马德拉），中年岛 1000 m 级（瓦胡、罗得），老岛几百米（毛里求斯、巴巴多斯）。"""
    a = np.clip(np.asarray(ages, dtype=np.float64), 0.0, 1.0)
    ly, lo = math.log(float(c["relief_young_m"])), math.log(float(c["relief_old_m"]))
    ref = np.exp(ly + a * (lo - ly))
    R = ref * (np.asarray(sizes, dtype=np.float64) / 1000.0) ** float(c["relief_area_exp"])
    R *= np.exp(rng.normal(0.0, float(c["relief_sigma"]), R.size))
    return np.clip(R, float(c["relief_min_m"]), float(c["relief_max_m"]))


def shoreline_gaps(masks_pos: list[tuple[np.ndarray, int, int]], res_km: float, centers_cell: np.ndarray,
                   radii_km: np.ndarray, max_gap_km: float) -> dict[tuple[int, int], float]:
    """两岛岸线间的最短距离（km），只算圆盘距离可能 ≤ max_gap 的岛对。masks_pos: (mask, row0, col0) 在群栅格中的位置。"""
    from .grid import binary_erode
    edges = []
    for m, r0, c0 in masks_pos:
        e = m & ~binary_erode(m, 1)
        ii, jj = np.where(e)
        pts = np.stack([(jj + c0) * res_km, -(ii + r0) * res_km], axis=1)
        if pts.shape[0] > 1500:
            pts = pts[:: int(math.ceil(pts.shape[0] / 1500))]
        edges.append(pts)
    n = len(masks_pos)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(*(centers_cell[i] - centers_cell[j])) * res_km
            if d - radii_km[i] - radii_km[j] > max_gap_km:
                continue
            a, b = edges[i], edges[j]
            best = 1e9
            for s in range(0, a.shape[0], 512):
                blk = a[s:s + 512]
                dd = np.sqrt(((blk[:, None, :] - b[None, :, :]) ** 2).sum(-1))
                best = min(best, float(dd.min()))
            if best <= max_gap_km:
                out[(i, j)] = best
    return out


def links(gaps: dict[tuple[int, int], float], rims: np.ndarray, n: int, c: dict) -> tuple[list[dict], list[tuple[int, int]]]:
    """索桥：岸距 ≤ bridge_max_km 且岸缘高差 ≤ bridge_max_dh_m；否则短渡。保证连通（不够就补最近的短渡）。
    另返回导水槽网络：只走索桥边、以主岛为根的最小生成树。"""
    bmax, dh = float(c["bridge_max_km"]), float(c["bridge_max_dh_m"])
    out = []
    for (i, j), g in sorted(gaps.items()):
        kind = "bridge" if (g <= bmax and abs(rims[i] - rims[j]) <= dh) else "ferry"
        out.append({"a": i, "b": j, "gap_km": round(g, 3), "kind": kind, "dh_m": round(float(abs(rims[i] - rims[j])), 1)})
    # 连通性回退
    from ..graph import weak_components
    def comps():
        src = np.array([e["a"] for e in out], dtype=np.int64)
        dst = np.array([e["b"] for e in out], dtype=np.int64)
        return weak_components(n, src, dst)
    comp = comps()
    while comp.max() + 1 > 1 and gaps:
        # 在不同分量间选岸距最小的一对（gaps 里没有的对无法量岸距，退回圆心距）
        best = None
        for (i, j), g in sorted(gaps.items()):
            if comp[i] != comp[j] and (best is None or g < best[2]):
                best = (i, j, g)
        if best is None:
            break
        out.append({"a": best[0], "b": best[1], "gap_km": round(best[2], 3), "kind": "ferry", "dh_m": round(float(abs(rims[best[0]] - rims[best[1]])), 1), "fallback": True})
        comp = comps()
    # 导水槽：Prim 从主岛出发，只走索桥
    adj = {i: [] for i in range(n)}
    for e in out:
        if e["kind"] == "bridge":
            adj[e["a"]].append((e["gap_km"], e["b"]))
            adj[e["b"]].append((e["gap_km"], e["a"]))
    import heapq
    seen = {0}
    heap = [(g, 0, j) for g, j in adj[0]]
    heapq.heapify(heap)
    tree = []
    while heap:
        g, i, j = heapq.heappop(heap)
        if j in seen:
            continue
        seen.add(j)
        tree.append((i, j))
        for g2, k in adj[j]:
            if k not in seen:
                heapq.heappush(heap, (g2, j, k))
    return out, tree
