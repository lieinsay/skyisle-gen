"""泊场、中转站、镇、航船、邑治（P7，Zhouzhu PLAN-LAND L24–L26；DISCUSS-SETTLE-ISLANDS 第〇、二、五、九节；DESIGN-NOTES 四点三十八）。settle.py 调用。

船寻常，但能控制的浮石船造起来有门槛，一般人家没有（L24）：一般人本岛走路，跨岛搭定班的航船。
- **泊场先于镇**（harbor_*）：能停很多船、能堆货的大块缓坡平地——坡 ≤ harbor_slope_max_deg、不是崖缘 / 水面 / 湿地 / 漫滩、没在种也没撂荒的格，
  开运算（先蚀一圈再胀一圈）去掉一两格宽的条子；每格数它 harbor_window_km 方窗里同岛的这种格（「能停多少船」按面积 × harbor_use_frac / ship_m2），
  够 harbor_min_km2 的格按（格数降序、格号）挑，彼此隔 harbor_sep_km。每个村照旧有自己的船台（村泊场），大泊场是另一层：镇、邑治挨着它。
- **中转站**（relays）：群内是群边高处的瞭望烽火（统一 P5 的烽火台）与进出本群的关卡；群与群之间是航线上的过夜、候风、避风、换船。
  行星层 ⑥ 的每条邻边（本群 ↔ 邻群，流量、有向成本、离开几天）按方位合成几个口子（流量 ≥ relay_flow_min，彼此差 ≥ relay_gate_sep_deg，至多 relay_max_gates），
  每个口子落在朝那个方向最靠外的一座岛上（主岛以外、方位在 ±relay_sector_deg 内、有能落船的平地、能存雨水；看得远的加分），实在没有就落在主岛朝外的崖边平地。
  住的是守关的、驿卒、开客栈的、修船的、仓夫，不种地、粮靠外运（和矿镇一样算常住的专业聚落），户从非农户里出（封顶 relay_cap_frac）；烽火台轮班守、不常住。
- **镇**（towns）：以周围户数为主——本岛走 walk_km 以内的村按走路算，别的岛（和本岛太远的）只算航船够得着的（按船程：直线距离 × 风的系数，顺风便宜、逆风贵）；
  挨着大泊场的加一成到三成；按「还没被服务的户」贪心挑（走路服务过的不再算，航船服务过的只让后来的镇按走路再算），镇距 ≥ town_spacing_km，不到 town_min_served_hh 就停。
- **航船**（boat_lines）：走不到镇的村搭航船去船程最近的镇；每个镇的航船村按方位扫一圈分线（每线至多 line_max_stops 个村、一趟船程 ≤ line_max_km），
  船从最远的村出发，挨个接人，开到镇上的泊场。
- **邑治**：从镇里挑航船汇得最多、靠大泊场的（(服务户 + seat_boat_weight × 航船送来的户) × (1 + seat_harbor_weight × min(1, 泊场的船 / harbor_ref_ships))），
  不按最大田块、不按好守。
人口：总户数照旧只读 ⑨；非农户先给专业聚落，再给中转站，余下按服务户数分给各镇（市户）。
纯函数、确定的次序（稳定排序、按格号 / 下标破平局、浮点按固定次序累加）；C++ 的 core/src/island/market.cpp 逐位同式。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import binary_dilate, binary_erode

LC_WET = 9
FUNC_ORDER = ("beacon", "gate", "inn", "wait", "shelter", "transship")
FUNC_ZH = {"beacon": "烽火", "gate": "关卡", "inn": "过夜", "wait": "候风", "shelter": "避风", "transship": "换船"}
INNER_FUNCS = ("beacon", "gate")                  # 群内：瞭望烽火、进出本群的关卡；其余是群与群之间的
ROLE_ORDER = ("gate", "post", "inn", "repair", "porter")
ROLE_ZH = {"gate": "守关", "post": "驿卒", "inn": "客栈", "repair": "修船", "porter": "仓夫"}
RELAY_OCC_ZH = {"resident": "常住", "rotation": "轮班"}
SCOPE_ZH = {"inner": "群内", "outer": "群间", "both": "群内、群间"}
MODE_ZH = {"walk": "步行", "boat": "航船"}
MARKET_NOTE = ("P7：船寻常，但能控制的浮石船造起来有门槛，一般人家没有——本岛走路赶集，跨岛搭定班的航船。harbors 是能停很多船的大块缓坡平地（先于镇）；"
               "镇以周围户数为主（本岛走路、跨岛只算航船够得着的），挨着大泊场的加分；邑治从镇里挑航船汇得最多、靠大泊场的。"
               "relays 是中转站：群内的瞭望烽火与关卡、群间的过夜 / 候风 / 避风 / 换船，住的人不种地、粮靠外运，户从非农户里出。")


def relay_note(funcs: list[str], island_main: bool) -> str:
    """中转站的说明（两个后端共用；funcs 是代码）。"""
    parts = []
    if "beacon" in funcs:
        parts.append("群边高处的瞭望与烽火，轮班守")
    if "gate" in funcs:
        parts.append("进出本群的船在这里停下验货交税")
    outer = [FUNC_ZH[f] for f in funcs if f not in INNER_FUNCS]
    if outer:
        parts.append("群间航线上的" + "、".join(outer))
    s = "；".join(parts)
    return s + ("（落在主岛朝外的崖边平地：朝这个方向没有合适的小岛）" if island_main else "")


def wind_factor(brg: float, wdir: float, steady: float, tail: float, head: float) -> float:
    """船程的风的系数：顺风 → tail，逆风 → head，按 cos(风去的方向 − 航向) × 风的稳定度插（与 ⑥ 的 dir_factor 同形）。"""
    c = math.cos(wdir - brg) * steady
    if c >= 0.0:
        return 1.0 / (1.0 + (1.0 / tail - 1.0) * c)
    return 1.0 + (head - 1.0) * (-c)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------- 泊场
def harbor_pad(g: dict, mc: dict) -> tuple[np.ndarray, np.ndarray]:
    """(能落船的平地 lflat, 大块泊场的格 pad)。lflat = 陆地、非崖、非河湖、坡 ≤ 阈值（与村的船台同口径）；
    pad = lflat 里再去掉溪涧、湿地、漫滩、在种与撂荒的田（林子算：镇会把它砍开），开运算去掉细条。"""
    iid = g["island_id"]
    land = iid >= 0
    smax = float(mc["harbor_slope_max_deg"])
    lflat = land & ~g["cliff"] & ~(g["river"] > 0) & ~g["lake"] & (g["slope_deg"] <= smax)
    lc = g["landcover"]
    flat = (lflat & ~(g["stream"] > 0) & (lc != LC_WET) & ~(g["floodplain"] > 0)
            & (g["cultivated"] == 0) & (g["fallow_years"] == 0))
    k = int(mc["harbor_open_cells"])
    pad = binary_dilate(binary_erode(flat, k), k) & flat if k > 0 else flat
    return lflat, pad


def island_cells(island_id: np.ndarray, n: int) -> list[np.ndarray]:
    """每座岛的格（扁平下标，升序）。"""
    flat = island_id.ravel()
    idx = np.flatnonzero(flat >= 0)
    o = idx[np.argsort(flat[idx], kind="stable")]
    b = np.searchsorted(flat[o], np.arange(n + 1))
    return [o[b[k]:b[k + 1]] for k in range(n)]


def harbor_count(pad: np.ndarray, iid: np.ndarray, cells: list[np.ndarray], r: int) -> np.ndarray:
    """每个陆地格：以它为心 (2r+1)² 方窗里同岛的 pad 格数（int32）。"""
    H, W = iid.shape
    cnt = np.zeros((H, W), dtype=np.int32)
    for k, c in enumerate(cells):
        if c.size == 0:
            continue
        ii, jj = c // W, c % W
        r0, r1, c0, c1 = int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1
        m = (pad[r0:r1, c0:c1] & (iid[r0:r1, c0:c1] == k)).astype(np.int64)
        S = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int64)
        S[1:, 1:] = m.cumsum(0).cumsum(1)
        a0 = np.clip(ii - r - r0, 0, r1 - r0)
        a1 = np.clip(ii + r + 1 - r0, 0, r1 - r0)
        b0 = np.clip(jj - r - c0, 0, c1 - c0)
        b1 = np.clip(jj + r + 1 - c0, 0, c1 - c0)
        cnt[ii, jj] = (S[a1, b1] - S[a0, b1] - S[a1, b0] + S[a0, b0]).astype(np.int32)
    return cnt


def ships_of(n_cells: int, cell_km2: float, mc: dict) -> int:
    return int(math.floor(float(n_cells) * cell_km2 * 1e6 * float(mc["harbor_use_frac"]) / float(mc["ship_m2"])))


def harbor_sites(g: dict, mc: dict, pad: np.ndarray, cnt: np.ndarray, km, res_km: float) -> list[dict]:
    """大泊场：pad 格里方窗数够 harbor_min_km2 的，按（格数降序、格号）挑，彼此隔 harbor_sep_km。"""
    H, W = pad.shape
    cell_km2 = res_km * res_km
    thr = float(mc["harbor_min_km2"]) / cell_km2
    cf = cnt.ravel()
    cand = np.flatnonzero(pad.ravel() & (cf >= thr))
    order = cand[np.lexsort((cand, -cf[cand].astype(np.int64)))]
    sep = float(mc["harbor_sep_km"]) / res_km
    R = int(math.ceil(sep))
    sep2 = sep * sep
    di, dj = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij")
    disc = (di * di + dj * dj).astype(np.float64) < sep2
    oi, oj = di[disc], dj[disc]
    blocked = np.zeros((H, W), dtype=bool)
    out = []
    iid = g["island_id"]
    for q in order.tolist():
        i, j = divmod(q, W)
        if blocked[i, j]:
            continue
        a, b = oi + i, oj + j
        ok = (a >= 0) & (a < H) & (b >= 0) & (b < W)
        blocked[a[ok], b[ok]] = True
        n = int(cf[q])
        out.append({"id": len(out) + 1, "island": int(iid[i, j]), "cell": [i, j], "km": km(i, j), "area_km2": round(n * cell_km2, 3),
                    "ships": ships_of(n, cell_km2, mc), "elev_m": round(float(g["height"][i, j]), 0), "town": None})
    return out


def landing_ships(lflat: np.ndarray, iid: np.ndarray, i: int, j: int, cell_km2: float, mc: dict) -> int:
    """村的船台 / 中转站能停几条船：3 × 3 里同岛能落船的格按面积折，封顶 landing_ships_max，至少 1。"""
    H, W = iid.shape
    k = iid[i, j]
    n = 0
    for a in range(max(0, i - 1), min(H, i + 2)):
        for b in range(max(0, j - 1), min(W, j + 2)):
            if lflat[a, b] and iid[a, b] == k:
                n += 1
    return max(1, min(int(mc["landing_ships_max"]), ships_of(n, cell_km2, mc)))


# ---------------------------------------------------------------- 中转站
def _angdiff(a: float, b: float) -> float:
    d = abs(a - b) % (2.0 * math.pi)
    return 2.0 * math.pi - d if d > math.pi else d


def build_relays(g: dict, mc: dict, sc: dict, routes: dict | None, villages: list[dict], hamlets: list[dict], specials: list[dict],
                 ruins: list[dict], lflat: np.ndarray, free: np.ndarray, cnt: np.ndarray, expo: np.ndarray, cells: list[np.ndarray], km,
                 res_km: float, nonfarm_left: int, storm: float) -> list[dict]:
    """中转站（群内的瞭望烽火与关卡、群间的过夜 / 候风 / 避风 / 换船）。返回记录（户数已按封顶折好）。
    lflat：能落船的平地（算船数）；free：站址能落的格（lflat 里没在种、没撂荒、没被村 / 散户 / 专业聚落 / 废村占的）。一座岛至多一处中转站。"""
    J = g["json"]
    I = J["islands"]
    n = len(I)
    iid = g["island_id"]
    H, W = iid.shape
    hflat = g["height"].ravel()
    x0, y0 = J["raster"]["origin_km"]
    cell_km2 = res_km * res_km
    cx0, cy0 = float(I[0]["center_km"][0]), float(I[0]["center_km"][1])
    P_mm = float(J["hydro"]["precip_mm"])
    wetisl = np.zeros(n, dtype=bool)
    wmask = ((g["river"] > 0) | g["lake"] | (g["stream"] > 0)) & (iid >= 0)
    wetisl[np.unique(iid[wmask])] = True
    rain_ok = P_mm >= float(sc["settle_rain_mm"])
    farm = np.zeros(n, dtype=bool)
    busy = np.zeros(n, dtype=bool)                    # 有村 / 散户 / 专业聚落 / 废村的岛（烽火台只挑没人住、没人用的岛）
    for v in villages + hamlets:
        farm[v["island"]] = True
        busy[v["island"]] = True
    for x in specials + ruins:
        busy[x["island"]] = True
    has_flat = np.zeros(n, dtype=bool)
    ff = free.ravel()
    for k in range(n):
        c = cells[k]
        has_flat[k] = bool(c.size) and bool(ff[c].any())
    relief_ref = float(mc["relay_relief_ref_m"])

    def island_score(k: int, ux: float, uy: float) -> float:
        dx = float(I[k]["center_km"][0]) - cx0
        dy = float(I[k]["center_km"][1]) - cy0
        rel = float(I[k]["peak_m"]) - float(I[k]["rim_m"])
        return (dx * ux + dy * uy) * (1.0 + min(1.0, max(0.0, rel) / relief_ref))

    def lookout(k: int) -> int:
        c = cells[k]
        return int(c[int(np.argmax(hflat[c]))])

    def station(k: int, ux: float, uy: float) -> int:
        """岛上的站址：能落船的平地里朝外（沿口子方向）最远、背风的格；平局按格号。"""
        c = cells[k]
        c = c[ff[c]]
        ii, jj = c // W, c % W
        best, bs = -1, 0.0
        lee_km = float(mc["relay_lee_km"])
        for q, i, j in zip(c.tolist(), ii.tolist(), jj.tolist()):
            x = x0 + (j + 0.5) * res_km
            y = y0 - (i + 0.5) * res_km
            s = (x - cx0) * ux + (y - cy0) * uy + lee_km * (1.0 - float(expo[i, j])) / 2.0
            if best < 0 or s > bs:
                best, bs = q, s
        return best

    relays = []                                       # 内部记录：{island, funcs[], edges[], station, lookout, ux, uy}
    by_island = {}
    # ---- 群间：⑥ 的邻边按方位合成口子
    edges = (routes or {}).get("edges") or []
    fmin = float(mc["relay_flow_min"])
    gw = [e for e in edges if float(e["flow_in"]) + float(e["flow_out"]) >= fmin]
    gw.sort(key=lambda e: (-(float(e["flow_in"]) + float(e["flow_out"])), int(e["node"])))
    gates = []                                        # [主边, [边…]]
    sep = math.radians(float(mc["relay_gate_sep_deg"]))
    for e in gw:
        best, bd = -1, 0.0
        for t, (e0, _) in enumerate(gates):
            d = _angdiff(float(e["bearing"]), float(e0["bearing"]))
            if best < 0 or d < bd:
                best, bd = t, d
        if best >= 0 and (bd < sep or len(gates) >= int(mc["relay_max_gates"])):
            gates[best][1].append(e)
        else:
            gates.append((e, [e]))
    sector = math.cos(math.radians(float(mc["relay_sector_deg"])))
    hub_gate = -1
    if routes and routes.get("hub") and gates:
        hub_gate = 0                                  # 流量最大的口子（gates 按主边流量降序建）
    for t, (e0, es) in enumerate(gates):
        th = float(e0["bearing"])
        ux, uy = math.sin(th), math.cos(th)
        best, bs = -1, 0.0
        for k in range(1, n):
            if not has_flat[k] or not (rain_ok or wetisl[k]):
                continue
            dx = float(I[k]["center_km"][0]) - cx0
            dy = float(I[k]["center_km"][1]) - cy0
            d = math.sqrt(dx * dx + dy * dy)
            if d <= 0.0 or (dx * ux + dy * uy) / d < sector:
                continue
            s = island_score(k, ux, uy)
            if best < 0 or s > bs:
                best, bs = k, s
        if best < 0:
            if not has_flat[0]:
                continue
            best = 0
        funcs = {"gate"}
        for e in es:
            if float(e["flow_in"]) >= fmin and float(e["cost_in"]) >= float(mc["relay_overnight_days"]):
                funcs.add("inn")
            if float(e["days"]) > 0.0 and float(e["cost_out"]) >= float(mc["relay_headwind_ratio"]) * float(e["days"]):
                funcs.add("wait")
        if storm >= float(mc["relay_storm_min"]):
            funcs.add("shelter")
        if t == hub_gate:
            funcs.add("transship")
        if best in by_island:
            r = relays[by_island[best]]
            r["funcs"] |= funcs
            r["edges"] += es
            continue
        by_island[best] = len(relays)
        relays.append({"island": best, "funcs": funcs, "edges": list(es), "ux": ux, "uy": uy})
    # ---- 群内：群边高处的瞭望烽火（统一 P5 的烽火台）：离主岛最远、看得远的几座没人住的岛，彼此方位差 ≥ beacon_sep_deg
    far = []
    for k in range(1, n):
        if not has_flat[k] or (busy[k] and k not in by_island) or farm[k]:
            continue
        dx = float(I[k]["center_km"][0]) - cx0
        dy = float(I[k]["center_km"][1]) - cy0
        d = math.sqrt(dx * dx + dy * dy)
        if d <= 0.0:
            continue
        far.append((-island_score(k, dx / d, dy / d), k, math.atan2(dy, dx)))
    far.sort(key=lambda x: (x[0], x[1]))
    bsep = math.radians(float(mc["beacon_sep_deg"]))
    angs = []
    for s, k, a in far:
        if len(angs) >= int(mc["beacon_max"]):
            break
        if any(_angdiff(a, b) < bsep for b in angs):
            continue
        angs.append(a)
        if k in by_island:
            relays[by_island[k]]["funcs"].add("beacon")
            continue
        dx = float(I[k]["center_km"][0]) - cx0
        dy = float(I[k]["center_km"][1]) - cy0
        d = math.sqrt(dx * dx + dy * dy)
        by_island[k] = len(relays)
        relays.append({"island": k, "funcs": {"beacon"}, "edges": [], "ux": dx / d, "uy": dy / d})
    # ---- 站址、户（守关、驿卒、客栈、修船、仓夫）
    fref = float(mc["relay_flow_ref"])
    raw = []
    for r in relays:
        flow = 0.0
        for e in r["edges"]:
            flow += float(e["flow_in"]) + float(e["flow_out"])
        s = min(2.0, max(0.5, math.sqrt(flow / fref))) if r["edges"] else 1.0
        f = r["funcs"]
        roles = {"gate": float(mc["relay_hh_gate"]) if "gate" in f else 0.0,
                 "post": float(mc["relay_hh_post"]) if "inn" in f else 0.0,
                 "inn": (float(mc["relay_hh_inn"]) * s if "inn" in f else 0.0) + (float(mc["relay_hh_wait"]) * s if "wait" in f else 0.0),
                 "repair": (float(mc["relay_hh_shelter"]) if "shelter" in f else 0.0) + (float(mc["relay_hh_repair"]) * s if "transship" in f else 0.0),
                 "porter": float(mc["relay_hh_porter"]) * s if "transship" in f else 0.0}
        r["flow"] = flow
        raw.append(roles)
    tot = 0.0
    for roles in raw:
        for k in ROLE_ORDER:
            tot += roles[k]
    cap = float(mc["relay_cap_frac"]) * float(nonfarm_left)
    fac = min(1.0, cap / max(1e-9, tot)) if tot > 0.0 else 0.0
    out = []
    for r, roles in zip(relays, raw):
        k = r["island"]
        st = station(k, r["ux"], r["uy"])
        si, sj = divmod(st, W)
        lk = lookout(k)
        li, lj = divmod(lk, W)
        hh_roles = {ROLE_ZH[q]: int(math.floor(roles[q] * fac)) for q in ROLE_ORDER if roles[q] > 0.0}
        hh = 0
        for v in hh_roles.values():
            hh += v
        funcs = [q for q in FUNC_ORDER if q in r["funcs"]]
        inner = any(q in INNER_FUNCS for q in funcs)
        outer = any(q not in INNER_FUNCS for q in funcs)
        rec = {"id": len(out) + 1, "island": k, "cell": [si, sj], "km": km(si, sj), "elev_m": round(float(g["height"][si, sj]), 0),
               "functions": [FUNC_ZH[q] for q in funcs], "scope": SCOPE_ZH["both" if inner and outer else ("inner" if inner else "outer")],
               "lookout_cell": [li, lj], "lookout_km": km(li, lj), "lookout_m": round(float(g["height"][li, lj]) - float(I[k]["rim_m"]), 0),
               "households": hh, "roles": {q: v for q, v in hh_roles.items() if v > 0}, "occupancy": RELAY_OCC_ZH["resident" if hh > 0 else "rotation"],
               "ships": max(landing_ships(lflat, iid, si, sj, cell_km2, mc), ships_of(int(cnt[si, sj]), cell_km2, mc)),
               "flow": round(r["flow"], 1),
               "routes": [{"node": int(e["node"]), "bearing_deg": round(math.degrees(float(e["bearing"])) % 360.0, 1), "days": round(float(e["days"]), 3),
                           "flow_in": round(float(e["flow_in"]), 1), "flow_out": round(float(e["flow_out"]), 1),
                           "cost_in": round(float(e["cost_in"]), 3), "cost_out": round(float(e["cost_out"]), 3), "hub": bool(e["hub"])} for e in r["edges"]],
               "note": relay_note(funcs, k == 0)}
        rec["name"] = f"中转站{rec['id']:02d}"
        out.append(rec)
    return out


# ---------------------------------------------------------------- 镇、航船、邑治
def build_towns(villages: list[dict], harbors: list[dict], mc: dict, rest_hh: int, wind: tuple[float, float]) -> dict:
    """镇、航船线、邑治。改写 villages 的 town / households_market / market_town / market_mode / market_km / boat_line / seat；
    被镇挨着的大泊场记 town。返回 {"towns", "lines", "seat"（villages 里的下标或 None）}。"""
    V = len(villages)
    if V == 0:
        return {"towns": [], "lines": [], "seat": None}
    u, v = float(wind[0]), float(wind[1])
    wdir = math.atan2(u, v)
    steady = min(1.0, math.sqrt(u * u + v * v) / float(mc["wind_ref_ms"]))
    tail, head = float(mc["tailwind_factor"]), float(mc["headwind_factor"])
    walk = float(mc["walk_km"])
    reach = float(mc["boat_reach_km"])
    X = [float(x["km"][0]) for x in villages]
    Y = [float(x["km"][1]) for x in villages]
    isl = [int(x["island"]) for x in villages]
    hh = np.array([int(x["households"]) for x in villages], dtype=np.int64)
    # 每个村当镇时挨着的大泊场：同岛、town_harbor_km 以内，船多的先（平局近的、编号小的）
    th_km = float(mc["town_harbor_km"])
    hb_of = [-1] * V
    for c in range(V):
        best, bk = -1, None
        for t, h in enumerate(harbors):
            if int(h["island"]) != isl[c]:
                continue
            d = _dist(X[c], Y[c], float(h["km"][0]), float(h["km"][1]))
            if d > th_km:
                continue
            key = (-int(h["ships"]), d, int(h["id"]))
            if best < 0 or key < bk:
                best, bk = t, key
        hb_of[c] = best
    TX = [float(harbors[hb_of[c]]["km"][0]) if hb_of[c] >= 0 else X[c] for c in range(V)]
    TY = [float(harbors[hb_of[c]]["km"][1]) if hb_of[c] >= 0 else Y[c] for c in range(V)]
    bonus = [float(mc["harbor_bonus"]) * min(1.0, float(harbors[hb_of[c]]["ships"]) / float(mc["harbor_ref_ships"])) if hb_of[c] >= 0 else 0.0
             for c in range(V)]

    def boat_cost(ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        d = math.sqrt(dx * dx + dy * dy)
        if d <= 0.0:
            return 0.0, 0.0
        return d, d * wind_factor(math.atan2(dx, dy), wdir, steady, tail, head)

    Wd = np.full((V, V), np.inf)                      # 走路（同岛直线）
    C = np.zeros((V, V))                              # 航船：村 → 候选镇的泊场（没有大泊场就是村自己）
    D = np.zeros((V, V))                              # 镇距（直线，不论岛）
    for a in range(V):
        for c in range(V):
            dd = _dist(X[a], Y[a], X[c], Y[c])
            D[a, c] = dd
            if isl[a] == isl[c]:
                Wd[a, c] = dd
            C[a, c] = boat_cost(X[a], Y[a], TX[c], TY[c])[1]
    walk_ok = Wd <= walk
    boat_ok = ~walk_ok & (C <= reach)
    wi = walk_ok.astype(np.int64)
    bi = boat_ok.astype(np.int64)
    walk_srv = np.zeros(V, dtype=bool)
    boat_srv = np.zeros(V, dtype=bool)
    small = hh < int(mc["town_min_village_hh"])       # P6b：几户人家的小圩村不当镇（镇从老村核心长出来）；全都小于门槛时照旧都能挑
    blocked = small.copy() if not small.all() else np.zeros(V, dtype=bool)
    chosen, scores = [], []
    spacing = float(mc["town_spacing_km"])
    min_srv = float(mc["town_min_served_hh"])
    bw = float(mc["boat_weight"])
    while not blocked.all():
        sw = (hh * ~walk_srv) @ wi                    # 还没走路赶上集的户
        sb = (hh * ~(walk_srv | boat_srv)) @ bi       # 还没有集可赶、航船够得着的户
        best, bs = -1, 0.0
        for c in range(V):
            if blocked[c]:
                continue
            s = (float(sw[c]) + bw * float(sb[c])) * (1.0 + bonus[c])
            if best < 0 or s > bs:
                best, bs = c, s
        if chosen and bs < min_srv:
            break
        chosen.append(best)
        scores.append(bs)
        blocked |= D[best] < spacing
        blocked[best] = True
        walk_srv |= walk_ok[:, best]
        boat_srv |= boat_ok[:, best]
    T = len(chosen)
    # 每个村归一个镇：本岛 walk_max_km 以内最近的镇（赶集当天往返）；走不到就搭航船去船程最近的镇
    tw = [-1] * V
    mode = [""] * V
    mkm = [0.0] * V
    walk_max = float(mc["walk_max_km"])
    for a in range(V):
        bt, bd = -1, 0.0
        for t, c in enumerate(chosen):
            if Wd[a, c] <= walk_max and (bt < 0 or Wd[a, c] < bd):
                bt, bd = t, float(Wd[a, c])
        if bt >= 0:
            tw[a], mode[a], mkm[a] = bt, "walk", bd
            continue
        for t, c in enumerate(chosen):
            if bt < 0 or C[a, c] < bd:
                bt, bd = t, float(C[a, c])
        tw[a], mode[a], mkm[a] = bt, "boat", bd
    # 航船：每个镇的航船村按方位扫一圈，分线
    maxs, maxk = int(mc["line_max_stops"]), float(mc["line_max_km"])

    def path(S, t):
        c = chosen[t]
        tx, ty = TX[c], TY[c]
        st = S[0]
        for a in S[1:]:
            if C[a, c] > C[st, c]:
                st = a
        order, rest = [st], [a for a in S if a != st]
        cur = st
        while rest:
            nb, nd = rest[0], _dist(X[cur], Y[cur], X[rest[0]], Y[rest[0]])
            for a in rest[1:]:
                d = _dist(X[cur], Y[cur], X[a], Y[a])
                if d < nd:
                    nb, nd = a, d
            order.append(nb)
            rest.remove(nb)
            cur = nb
        pts = [(X[a], Y[a]) for a in order] + [(tx, ty)]
        L = K = 0.0
        for p, q in zip(pts[:-1], pts[1:]):
            d, k = boat_cost(p[0], p[1], q[0], q[1])
            L += d
            K += k
        return order, pts, L, K

    lines_of = [[] for _ in range(T)]
    for t in range(T):
        c = chosen[t]
        mine = [a for a in range(V) if tw[a] == t and mode[a] == "boat"]
        if not mine:
            continue
        ang = [(math.atan2(X[a] - TX[c], Y[a] - TY[c]), a) for a in mine]
        ang.sort()
        m = len(ang)
        start = 0
        if m > 1:
            gbest = -1.0
            for q in range(m):
                gap = ang[(q + 1) % m][0] - ang[q][0] + (2.0 * math.pi if q == m - 1 else 0.0)
                if gap > gbest:
                    gbest, start = gap, (q + 1) % m
        seq = [ang[(start + q) % m][1] for q in range(m)]
        cur = []
        for a in seq:
            trial = cur + [a]
            if cur and (len(trial) > maxs or path(trial, t)[3] > maxk):
                lines_of[t].append(path(cur, t))
                cur = [a]
            else:
                cur = trial
        if cur:
            lines_of[t].append(path(cur, t))
    # 邑治：航船汇得最多、靠大泊场——(服务户 + seat_boat_weight × 航船送来的户) × (1 + seat_harbor_weight × min(1, 泊场的船 / harbor_ref_ships))，平局服务户多的、先挑的
    served = [0] * T
    boat_srv_hh = [0] * T
    for a in range(V):
        served[tw[a]] += int(hh[a])
        if mode[a] == "boat":
            boat_srv_hh[tw[a]] += int(hh[a])
    seat_t, sk = -1, None
    seat_sc = [0.0] * T
    for t in range(T):
        c = chosen[t]
        hq = min(1.0, float(harbors[hb_of[c]]["ships"]) / float(mc["harbor_ref_ships"])) if hb_of[c] >= 0 else 0.0
        s = (float(served[t]) + float(mc["seat_boat_weight"]) * float(boat_srv_hh[t])) * (1.0 + float(mc["seat_harbor_weight"]) * hq)
        seat_sc[t] = s
        key = (-s, -served[t], t)
        if seat_t < 0 or key < sk:
            seat_t, sk = t, key
    order_t = [seat_t] + [t for t in range(T) if t != seat_t]
    tid = {t: n + 1 for n, t in enumerate(order_t)}
    # 市户：非农户余量（专业聚落、中转站之后）按服务户数分（最大余数法）
    tot_srv = 0.0
    for t in order_t:
        tot_srv += float(served[t])
    raw = [float(rest_hh) * float(served[t]) / max(1e-9, tot_srv) for t in order_t]
    base = [int(math.floor(x)) for x in raw]
    left = rest_hh - sum(base)
    for q in sorted(range(T), key=lambda q: -(raw[q] - base[q]))[:max(0, left)]:
        base[q] += 1
    # 记录
    towns, lines = [], []
    for n, t in enumerate(order_t):
        c = chosen[t]
        vc = villages[c]
        vc["town"] = n + 1
        vc["households_market"] = int(base[n])
        walk_hh = boat_hh = other_hh = 0
        nv, mx = 0, 0.0
        for a in range(V):
            if tw[a] != t:
                continue
            nv += 1
            if mode[a] == "walk":
                walk_hh += int(hh[a])
                mx = max(mx, float(Wd[a, c]))
            else:
                boat_hh += int(hh[a])
            if isl[a] != isl[c]:
                other_hh += int(hh[a])
        ids = []
        for order, pts, L, K in lines_of[t]:
            lid = len(lines) + 1
            ids.append(lid)
            lhh = 0
            for a in order:
                lhh += int(hh[a])
                villages[a]["boat_line"] = lid
            lines.append({"id": lid, "name": f"航船{lid:02d}", "town": n + 1, "stops": [int(villages[a]["id"]) for a in order],
                          "islands": sorted({isl[a] for a in order}), "households": lhh,
                          "pts_km": [[round(p[0], 3), round(p[1], 3)] for p in pts], "length_km": round(L, 2), "cost_km": round(K, 2),
                          "hours": round(K / float(mc["boat_kmh"]), 2),
                          "interval_days": 1 if lhh >= int(mc["line_daily_hh"]) else int(mc["line_interval_days"])})
        h = harbors[hb_of[c]] if hb_of[c] >= 0 else None
        rec = {"id": n + 1, "name": "邑治" if t == seat_t else f"镇{n + 1:02d}", "village": int(vc["id"]), "island": isl[c], "cell": list(vc["cell"]),
               "km": list(vc["km"]), "households_farm": int(vc["households"]), "households_market": int(base[n]),
               "households": int(vc["households"]) + int(base[n]), "served_villages": nv, "served_households": served[t],
               "served_walk_households": walk_hh, "served_boat_households": boat_hh, "served_other_islands_households": other_hh,
               "max_served_km": round(mx, 2), "n_lines": len(ids), "lines": ids, "score": round(scores[t], 1), "seat_score": round(seat_sc[t], 1),
               "seat": t == seat_t, "harbor": int(h["id"]) if h else None, "harbor_ships": int(h["ships"]) if h else 0,
               "harbor_dist_km": round(_dist(X[c], Y[c], float(h["km"][0]), float(h["km"][1])), 2) if h else None, "street": None}
        if h:
            h["town"] = n + 1
            dx, dy = float(h["km"][0]) - X[c], float(h["km"][1]) - Y[c]
            rec["street"] = {"from_km": list(vc["km"]), "to_km": list(h["km"]), "length_km": round(math.sqrt(dx * dx + dy * dy), 2),
                             "bearing_deg": round(math.degrees(math.atan2(dx, dy)) % 360.0, 1)}
        towns.append(rec)
    for a in range(V):
        villages[a]["market_town"] = tid[tw[a]]
        villages[a]["market_mode"] = MODE_ZH[mode[a]]
        villages[a]["market_km"] = round(mkm[a], 2)
    villages[chosen[seat_t]]["seat"] = True
    return {"towns": towns, "lines": lines, "seat": chosen[seat_t]}


# ---------------------------------------------------------------- 行星层 ⑥ 的邻边（只读）
def node_routes(ctx, node: int) -> dict | None:
    """本群在 ⑥ 航线网里的邻边：[{node 邻群, bearing 方位（弧度，0 = 北、顺时针，sphere.initial_bearing）, days ③ 的离开几天,
    cost_out / cost_in ⑥ 的有向成本（天）, flow_out / flow_in ⑥ 的有向流量（routes.npz 存 float32）, hub 邻群是不是枢纽}]（按 ③ 的边号升序）
    与本群是不是 ⑥ 的枢纽（岛群级的中转岛）。没有 ⑥ 的产物给 None。C++ 的 planet::apply_routes 同式（_core.node_routes）。"""
    import json as _json
    p6 = ctx.stage_dir(6) / "routes.npz"
    ph = ctx.stage_dir(6) / "hubs.json"
    p3 = ctx.stage_dir(3) / "cand_edges.npz"
    if not (p6.exists() and ph.exists() and p3.exists()):
        return None
    isl = ctx.load_npz(3, "islands")
    with np.load(p3) as ce, np.load(p6) as z:
        src, dst, days = ce["src"], ce["dst"], ce["dist_days"]
        cost, flow = z["cost"], z["flow"]
        E = int(src.size)
        hubs = {int(h["node"]) for h in _json.loads(ph.read_text(encoding="utf-8"))["hubs"]}
        lat, lon = isl["lat"], isl["lon"]
        la1, lo1 = math.radians(float(lat[node])), math.radians(float(lon[node]))
        edges = []
        for e in np.flatnonzero((src == node) | (dst == node)).tolist():
            a, b = int(src[e]), int(dst[e])
            m = b if a == node else a
            o, i = (e, e + E) if a == node else (e + E, e)
            la2, lo2 = math.radians(float(lat[m])), math.radians(float(lon[m]))
            dl = lo2 - lo1
            y = math.sin(dl) * math.cos(la2)
            x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl)
            edges.append({"node": m, "bearing": math.atan2(y, x), "days": float(days[e]), "cost_out": float(cost[o]), "cost_in": float(cost[i]),
                          "flow_out": float(flow[o]), "flow_in": float(flow[i]), "hub": m in hubs})
    return {"hub": int(node) in hubs, "edges": edges}


# ---------------------------------------------------------------- 泊场记录、栅格、摘要
def add_landings(g: dict, lands: list[dict], towns: list[dict], harbors: list[dict], relays: list[dict], villages: list[dict], km) -> None:
    """镇 / 邑治的大泊场（邑治的是主泊场 = 仓场）、中转站的泊场接在村与专业聚落的船台后面；没有大泊场的镇用它那个村的船台（邑治的船台就是主泊场）。"""
    slope = g["slope_deg"]
    vid = {int(v["id"]): v for v in villages}
    for t in towns:
        if t["harbor"] is not None:
            h = harbors[int(t["harbor"]) - 1]
            i, j = h["cell"]
            L = {"id": len(lands) + 1, "kind": "主泊场（仓场）" if t["seat"] else "镇泊场", "of": t["name"], "island": int(t["island"]), "cell": [i, j],
                 "km": list(h["km"]), "slope_deg": round(float(slope[i, j]), 1), "flat": True, "dist_km": t["harbor_dist_km"], "main": bool(t["seat"]),
                 "ships": int(h["ships"]), "harbor": int(h["id"])}
            lands.append(L)
            t["landing"] = L["id"]
        else:
            lid = int(vid[int(t["village"])]["landing"])
            t["landing"] = lid
            if t["seat"]:
                lands[lid - 1]["kind"] = "主泊场（仓场）"
                lands[lid - 1]["main"] = True
    for r in relays:
        i, j = r["cell"]
        L = {"id": len(lands) + 1, "kind": "中转站泊场", "of": r["name"], "island": int(r["island"]), "cell": [i, j], "km": list(r["km"]),
             "slope_deg": round(float(slope[i, j]), 1), "flat": True, "dist_km": 0.0, "main": False, "ships": int(r["ships"])}
        lands.append(L)
        r["landing"] = L["id"]


def mark_raster(sraster: np.ndarray, towns: list[dict], harbors: list[dict], relays: list[dict]) -> None:
    """17 = 镇 / 邑治的大泊场、18 = 中转站（站址；有烽火的再加瞭望处）。只写还空着的格（水利、村、泊场已经在上面的不动）。"""
    for t in towns:
        if t["harbor"] is not None:
            i, j = harbors[int(t["harbor"]) - 1]["cell"]
            if sraster[i, j] == 0:
                sraster[i, j] = 17
    for r in relays:
        cs = [r["cell"]] + ([r["lookout_cell"]] if FUNC_ZH["beacon"] in r["functions"] else [])
        for i, j in cs:
            if sraster[i, j] == 0:
                sraster[i, j] = 18


def market_summary(harbors: list[dict], towns: list[dict], lines: list[dict], relays: list[dict], villages: list[dict], mc: dict) -> dict:
    """settlements.json 的 market：泊场、镇、航船、邑治、中转站的几个总数（两个后端共用）。"""
    area = ships = 0.0
    for h in harbors:
        area += float(h["area_km2"])
        ships += float(h["ships"])
    lkm = 0.0
    for ln in lines:
        lkm += float(ln["length_km"])
    boat = [v for v in villages if v.get("market_mode") == MODE_ZH["boat"]]
    seat = next((t for t in towns if t["seat"]), None)
    funcs = {}
    for r in relays:
        for f in r["functions"]:
            funcs[f] = funcs.get(f, 0) + 1
    return {"n_harbors": len(harbors), "harbor_km2": round(area, 3), "harbor_ships": int(ships),
            "n_towns": len(towns), "towns_with_harbor": sum(1 for t in towns if t["harbor"] is not None),
            "n_lines": len(lines), "line_km": round(lkm, 2), "boat_villages": len(boat), "walk_villages": len(villages) - len(boat),
            "boat_households": sum(int(v["households"]) for v in boat),
            "far_villages": sum(1 for v in boat if float(v["market_km"]) > float(mc["boat_reach_km"])),
            "seat_town": seat["id"] if seat else None, "seat_island": seat["island"] if seat else None,
            "n_relays": len(relays), "relay_functions": {FUNC_ZH[f]: funcs[FUNC_ZH[f]] for f in FUNC_ORDER if FUNC_ZH[f] in funcs},
            "relay_households": sum(int(r["households"]) for r in relays), "note": MARKET_NOTE}
