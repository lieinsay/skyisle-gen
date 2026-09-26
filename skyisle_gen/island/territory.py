"""势力范围：邻群的陆地不许叠在一起（Zhouzhu 行星计划 D12；DESIGN-NOTES 四点二十二）。

行星上相邻的两个节点 k、j（大圆距 d）之间画一条分界线：垂直于 k → j 的方位、离 k 的距离 t = d × r_k / (r_k + r_j)，
r = √(陆地 / π) 是等效半径——陆地多的群分得多。每群的陆地只许在自己那侧、离分界线至少 gap_km / 2，
所以任意两群的陆地至少隔 gap_km（两边各退半道缝，谁先生成都一样，按需生成也对得上）。

平面就是第三层栅格的平面：以节点为中心的方位等距投影（AEQD，离群心的大圆距离与方位不变，x 东 y 北，km）。
几百 km 以内它与球面上的分界相差 < 0.1 km，远小于缝。

用法（build_terrain）：布局先照旧摆，查出越界才带约束重来——主岛挪到势力范围里最近的地方（放不下就转走向），
其余的岛带着约束重新摆。所以没碰到约束的群，产物逐字节不变。
"""
from __future__ import annotations

import math

import numpy as np


def limits(ctx, node: int, inp: dict, c: dict) -> list[dict]:
    """该群的分界线：[{node, dist_km, u: (ux, uy), limit_km}]，陆地须满足 p·u ≤ limit（p 为群平面坐标，km）。
    只收可能碰得上的邻群：群心距 ≤ reach × (两群等效半径之和) + reach_km。"""
    isl = ctx.load_npz(3, "islands")
    R = float(inp["planet"]["radius_km"])
    lat = np.radians(isl["lat"].astype(np.float64))
    lon = np.radians(isl["lon"].astype(np.float64))
    area = isl["area_km2"].astype(np.float64)
    p0, l0 = lat[node], lon[node]
    cosang = np.sin(p0) * np.sin(lat) + np.cos(p0) * np.cos(lat) * np.cos(lon - l0)
    dist = R * np.arccos(np.clip(cosang, -1.0, 1.0))
    r = np.sqrt(area / math.pi)
    reach = float(c.get("reach", 3.0)) * (r[node] + r) + float(c.get("reach_km", 40.0))
    near = np.where((dist <= reach) & (np.arange(dist.size) != node))[0]
    gap = float(c.get("gap_km", 3.0))
    out = []
    for j in near[np.argsort(dist[near], kind="stable")]:
        dl = lon[j] - l0
        az = math.atan2(math.sin(dl) * math.cos(lat[j]),
                        math.cos(p0) * math.sin(lat[j]) - math.sin(p0) * math.cos(lat[j]) * math.cos(dl))
        d = float(dist[j])
        t = d * r[node] / (r[node] + r[j])
        out.append({"node": int(j), "dist_km": round(d, 3), "u": (math.sin(az), math.cos(az)), "limit_km": t - 0.5 * gap})
    return out


def _margin(u, res_km: float) -> float:
    """局部栅格的格心到群栅格里那一格最远角在 u 上的投影：贴进群栅格时格心偏半格、再按格取整偏至多半格，加上格子的半宽。"""
    return 1.5 * res_km * (abs(u[0]) + abs(u[1]))


def mask_support(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, lim: list[dict], res_km: float) -> np.ndarray:
    """岛的局部栅格（X, Y 相对局部栅格中心，km）上陆地在各分界线法向的最远投影 h[j]。"""
    xs, ys = X[mask], Y[mask]
    h = np.empty(len(lim))
    for k, L in enumerate(lim):
        ux, uy = L["u"]
        h[k] = float((xs * ux + ys * uy).max()) + _margin(L["u"], res_km) if xs.size else -1e9
    return h


def profile_support(prof: np.ndarray, lim: list[dict], res_km: float) -> np.ndarray:
    """角向半径剖面 r(θ)（相对岛的质心）在各分界线法向的最远投影。"""
    n = prof.size
    th = -math.pi + (np.arange(n) + 0.5) * (2 * math.pi / n)
    widen = 1.0 / math.cos(math.pi / n)   # 每桶只记了桶内最大半径，方向在桶里不定：放宽到桶边
    h = np.empty(len(lim))
    for k, L in enumerate(lim):
        ux, uy = L["u"]
        h[k] = float((prof * widen * (np.cos(th) * ux + np.sin(th) * uy)).max()) + _margin(L["u"], res_km)
    return h


def violation(pos: np.ndarray, support: np.ndarray, lim: list[dict]) -> float:
    """岛放在 pos（它支撑量的参考点，km）时越界多少（> 0 越界，≤ 0 是还剩的余量的相反数）。"""
    if not lim:
        return -1e9
    return max(float(pos[0] * L["u"][0] + pos[1] * L["u"][1] + h - L["limit_km"]) for L, h in zip(lim, support))


def nearest_fit(support: np.ndarray, lim: list[dict], iters: int = 400) -> tuple[np.ndarray, float]:
    """离原点最近、让支撑量为 support 的岛整个落在势力范围里的偏移（Dykstra 交替投影到各半平面）。
    返回 (偏移, 越界量)；越界量 > 0 说明这个岛在哪都放不下（返回的是越界最少的近似）。"""
    x = np.zeros(2)
    if not lim:
        return x, -1e9
    U = np.array([L["u"] for L in lim], dtype=np.float64)
    b = np.array([L["limit_km"] for L in lim], dtype=np.float64) - support
    P = np.zeros((len(lim), 2))
    for _ in range(iters):
        for i in range(len(lim)):
            y = x + P[i]
            s = float(y @ U[i]) - b[i]
            xn = y - max(0.0, s) * U[i]
            P[i] = y - xn
            x = xn
    return x, float((U @ x - b).max())


def raster_violation(height: np.ndarray, island_id: np.ndarray, raster: dict, lim: list[dict]) -> float:
    """群栅格上所有陆地格（按格子的角）对分界线的最大越界量（km；≤ 0 = 没越界，负值是最小余量）。"""
    if not lim:
        return -1e9
    res = float(raster["res_m"]) / 1000.0
    x0, y0 = raster["origin_km"]
    rr, cc = np.where(island_id >= 0)
    if rr.size == 0:
        return -1e9
    x = x0 + (cc + 0.5) * res
    y = y0 - (rr + 0.5) * res
    worst = -1e9
    for L in lim:
        ux, uy = L["u"]
        m = float((x * ux + y * uy).max()) + 0.5 * res * (abs(ux) + abs(uy)) - L["limit_km"]
        worst = max(worst, m)
    return worst
