"""宜垦 / 已垦 / 撂荒、定居门槛、没人住 ≠ 没人用、荒地归谁（P5，Zhouzhu PLAN-LAND L7 / L10 / L21 / L29；DESIGN-NOTES 四点三十六）。

- **宜垦**（hydro 调 cultivable_land）：地本身能不能种——坡 < cultivable_slope_max_deg、最暖的月份（年均温 + 半个季节温差）≥ cultivable_summer_min_c、
  土层 ≥ cultivable_soil_min、够湿（湿度 ≥ cultivable_wet_min）或引得到水（离河湖 water_near_cells 内），不是崖缘、水面、湿地；
  坡 ≥ terrace_slope_deg 的要修梯田。→ g["cultivable"]（0 / 1 / 2）。
  行星层的可耕率照旧按适宜度取出「上等地」（hydro 的 g["arable"]，额度内最好的宜垦地）：资源层照旧把它当田土避开
  （岩类赋存不上这片地、林场不算这片，赋存场与点位逐字节不变），地表在这时照旧画成可耕地 / 梯田；
  g["cover_natural"] 留着它原本的地表，g["suit"] 留着适宜度给聚落层排先后。上等地一定宜垦（极冷、极陡的群也一样：行星层说这么多地在种）。
- **已垦**（settle 调 fill_cultivated）：人口从最好的宜垦地往外填（好地先占）——宜垦地按适宜度排先后（与 hydro 取上等地同一个数），
  填到行星层的额度为止（额度 = 上等地的格数，人口口径不变）。远近不排先后：群内的短渡飞船半小时内就到
  （试过「适宜度 × 跨岛 0.95 × exp(−离主岛的岸距 / 50 km)」：小岛的地本来就比主岛陡、冷、离河远，再一打折额度全落在主岛上，三群只剩主岛有村，太过）；
  **定居门槛**：宜垦地的连通片（不跨岛）要有水（settle_water_km 内有河 / 湖 / 溪涧 / 泉，或年降水 ≥ settle_rain_mm 能蓄雨）、
  够一个像样的村（片的面积 ≥ settle_min_hh 户的地）、能落船（landing_reach_cells 内有坡 ≤ landing_slope_max_deg 的平地）；
  头一轮填完分到的地不够 settle_min_hh 户的片整片让出来（不再按比例连续撒户），额度按次序补给留下的片——一轮就定（留下的片只会变多）。
  **大岛保底**（用户 09-29 定）：主岛以外 ≥ island_floor_km2（30 km²）的岛，有合门槛的宜垦片的，先在它最好的那片地上分一个村——
  从那片里适宜度最高的格起、按适宜度往外长成连成一块的 island_floor_hh（20）户的地（一块田配一个村），户从额度里出（从主岛匀过去）；其余照好地先占填。
- **撂荒**：留下的片里接着往外的一层（离在种的田 ≤ fallow_ring_cells 格）取已垦的 fallow_frac，每个连通块一个年头（1 … fallow_years_max）；
  **废村**：让出来的片里头一轮分得最多的几片（≥ ruin_min_hh 户，至多 ruin_max 个）——人撤走了，头一轮分到的那些地是撂荒的田，年头 = 撤空了几年。
  撂荒地的地表按年头：≤ fallow_grass_years 年草坡，≤ fallow_shrub_years 年灌丛（原本是林 / 灌丛的），再久回到原本的地表（小树林）。
- **没人住 ≠ 没人用**：光秃小岛（没有村和散户）上的浮石采石村、矿村、窑村改成「工棚」（白天来、晚上走，人算在最近的村）；
  烧炭营是季节住；矿镇、盐井村、温泉地照旧常住。没人常住的岛里按合理挑少数：放牧岛（只放牲口）/ 夏牧（夏天的牧棚，季节住）、
  群边高处的烽火台（轮班守）、邑治附近的庙、墓岛。
- **荒地归谁**（L29，用户定：荒地有主、领照开垦）：每岛记一个主——有农户（村或散户）的岛：未垦的宜垦地归就近的村（村里的大户 / 族产）；
  没人住但有人用的岛（工棚、季节住、放牧……）：归用它的那个村的大户；别的荒岛、废村的地：官荒，归邑（绝户田入官）。

纯函数、只用确定的次序（稳定排序、按格号 / 岛号 / 片号破平局），C++ 的 core/src/island/farmland.cpp 逐位同式。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import binary_dilate, distance_bands, label_by_island

LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET = 1, 2, 3, 4, 5, 6, 7, 8, 9
WORKCAMP_KINDS = ("浮石采石村", "矿村", "窑村")      # 光秃小岛上改成工棚的专业聚落
SEASONAL_KINDS = ("烧炭营",)
# cpp 后端的代码 → 中文（decode.py 用；两个后端的说明文字由下面几个函数拼，一处改）
SPECIAL_OCC_ZH = {"resident": "常住", "workcamp": "工棚", "seasonal": "季节住"}
USE_ZH = {"graze": "放牧", "shieling": "夏牧", "beacon": "烽火台", "shrine": "庙", "tomb": "墓岛"}
USE_OCC_ZH = {"livestock": "只放牲口", "seasonal": "季节住", "rotation": "轮班", "incense": "香火", "none": "无人"}
STATUS_ZH = {"resident": "常住", "seasonal": "季节住", "used": "有人用", "empty": "荒岛"}
OWNER_ZH = {"village": "村", "magnate": "大户", "crown": "官荒"}


def ruin_note(years: int, hh: int, km2: float) -> str:
    return f"撤空了 {years} 年；原来约 {hh} 户，旁边的田撂荒（{km2:.2f} km²），地收归官府（绝户田入官）"


def use_note(code: str, a: float, b: float | None = None) -> str:
    if code == "graze":
        return f"只放牲口，隔些日子来看一回（草场 {a:.1f} km²，离村 {b:.1f} km）"
    if code == "shieling":
        return f"夏天赶上来放牧，搭着牧棚住一季（草场 {a:.1f} km²，离村 {b:.1f} km）"
    if code == "beacon":
        return f"群边高处的瞭望与烽火，离主岛 {a:.1f} km；轮班守，不常住"
    if code == "shrine":
        return f"邑治附近的小岛上一座庙，离邑治 {a:.1f} km；逢年过节有人上岛"
    if code == "tomb":
        return f"葬地，离邑治 {a:.1f} km；清明上坟"
    raise ValueError(code)


# ---------------------------------------------------------------- 宜垦（hydro）
def cultivable_land(g: dict, lc: dict, cover: np.ndarray, arable: np.ndarray, suit: np.ndarray, slope: np.ndarray,
                    T: np.ndarray, soil: np.ndarray, wet: np.ndarray, near_water: np.ndarray) -> None:
    """hydro 在上等地（arable）取完、还没把它画进地表之前调：记下原本的地表、适宜度与宜垦。"""
    g["cover_natural"] = cover.copy()
    g["suit"] = suit
    half = 0.5 * float(g["inp"]["season_range"])        # 最暖的月份 ≈ 年均温 + 半个季节温差（④ 的岛上温差，大陆性强的群夏天热、冬天冷）
    ok = ((suit > 0.0) & (slope < float(lc["cultivable_slope_max_deg"])) & (T + half >= float(lc["cultivable_summer_min_c"]))
          & (soil >= float(lc["cultivable_soil_min"])) & ((wet >= float(lc["cultivable_wet_min"])) | (near_water > 0.0)))
    ok |= arable > 0
    out = np.zeros(ok.shape, dtype=np.uint8)
    out[ok] = 1
    out[ok & (slope >= float(lc["terrace_slope_deg"]))] = 2
    g["cultivable"] = out


# ---------------------------------------------------------------- 已垦、撂荒、废村（settle）
def fill_cultivated(g: dict, sc: dict, rng, water: np.ndarray, land_per_hh: float, n_quota: int, wc: dict | None = None) -> dict:
    """好地先占 + 定居门槛 → g["cultivated"]（0 / 1 田 / 2 梯田，在种）、g["fallow_years"]（撂荒了几年，0 = 不是撂荒地）；
    P6：wc（[island.works]）给了就在第一遍之后挑要排干的湿地（waterworks.polder_plan），第二遍圩田的格先占（额度之内）→ g["polder_id"]；
    改 g["landcover"]（上等地没人种的回原本的地表、已垦画成可耕地 / 梯田、撂荒按年头、圩田画成可耕地）与林场 / 芦苇荡的 patch_id。
    返回 {"ruins": [...（不含村址）], "summary": {...}, "polders": 圩田的安排}。"""
    island_id = g["island_id"]
    H, W = island_id.shape
    J = g["json"]
    res_km = float(g["res_km"])
    cell_km2 = res_km * res_km
    land = island_id >= 0
    wet = (g["landcover"] == LC_WET) & land          # 聚落层动手之前的湿地（P6 的圩田从这里挑）
    cult = g["cultivable"]
    suit = g["suit"]
    R = g.get("resources")
    # 采场（资源层挑的矿坑 / 硫磺坑 / 盐井）上、岩类赋存（矿化带、露石、硫磺）上不开田
    pit = np.zeros((H, W), dtype=bool)
    if R:
        for w in R["workings"]:
            pit[w["cell"][0], w["cell"][1]] = True
        if "res_field" in g:
            from .resources import FIELD_KINDS, ROCK_KINDS
            for k in ROCK_KINDS:
                pit |= g["res_field"][FIELD_KINDS.index(k)] > 0
    # ---- 宜垦的连通片（不跨岛）与定居门槛
    lab, n_lab = label_by_island(cult > 0, island_id, 8)
    labf = lab.ravel()
    n_cells = np.bincount(labf, minlength=n_lab + 1)
    src = water.copy()
    if R:
        for d in R["deposits"]:
            if d["kind"] == "spring":
                src[d["cell"][0], d["cell"][1]] = True
    cap_w = int(math.ceil(float(sc["settle_water_km"]) / res_km - 1e-9))
    dw = distance_bands(src, cap_w)
    near_w = np.zeros(n_lab + 1, dtype=bool)
    m = labf > 0
    near_w[np.unique(labf[m & (dw.ravel() <= cap_w)])] = True
    rain_ok = float(J["hydro"]["precip_mm"]) >= float(sc["settle_rain_mm"])
    reach = int(sc["landing_reach_cells"])
    flat = land & ~((g["river"] > 0) | g["lake"]) & ~g["cliff"] & (g["slope_deg"] <= float(sc["landing_slope_max_deg"]))
    near_f = np.zeros(n_lab + 1, dtype=bool)
    near_f[np.unique(labf[m & (distance_bands(flat, reach).ravel() <= reach)])] = True
    min_hh = float(sc["settle_min_hh"])
    big = n_cells.astype(np.float64) * cell_km2 >= min_hh * land_per_hh
    ok_t = (near_w | rain_ok) & near_f & big
    ok_t[0] = False
    # ---- 好地先占：宜垦格按适宜度降序（平局按格号）
    elig = np.flatnonzero((cult.ravel() > 0) & ~pit.ravel())
    order = elig[np.argsort(-suit.ravel()[elig], kind="stable")]
    t_ord = labf[order]
    # ---- 大岛保底：先给 ≥ island_floor_km2 的非主岛各分一块连成片的 island_floor_hh 户的地
    gcells, g_isl = floor_cells(g, sc, lab, n_lab, ok_t, order, t_ord, pit, land_per_hh, n_quota)
    n_g = int(gcells.size)
    reserved = np.zeros(H * W, dtype=bool)
    reserved[gcells] = True

    def fill_pass(pcells):
        """一遍好地先占：保底的格、圩田的格先占，其余按次序填留下的片；返回 (pick1, cnt1, hh1, kept, in_kept, take, forced)。"""
        n_r = n_g + int(pcells.size)
        ok1 = ok_t[t_ord] & ~reserved[order]
        pick1 = np.concatenate([gcells, pcells, order[ok1][: n_quota - n_r]])
        cnt1 = np.bincount(labf[pick1], minlength=n_lab + 1)
        hh1 = cnt1.astype(np.float64) * cell_km2 / land_per_hh
        kept = ok_t & (hh1 >= min_hh)
        kept[np.unique(labf[gcells])] = True
        kept[0] = False
        ok2 = kept[t_ord] & ~reserved[order]
        in_kept = order[ok2]
        take = np.concatenate([gcells, pcells, in_kept[: n_quota - n_r]])
        forced = 0
        if take.size < n_quota:                   # 留下的片不够额度：按次序补让出来的片，再补不合门槛的片
            rest = order[~kept[t_ord]]
            rt = labf[rest]
            extra = np.concatenate([rest[ok_t[rt]], rest[~ok_t[rt]]])[: n_quota - take.size]
            forced = int(extra.size)
            take = np.concatenate([take, extra])
        return pick1, cnt1, hh1, kept, in_kept, take, forced

    none = np.zeros(0, dtype=np.int64)
    pick1, cnt1, hh1, kept, in_kept, take, forced = fill_pass(none)
    # ---- 圩田（P6，waterworks.polder_plan）：第一遍填到了湿地边上（周围的平地种了过半）的湿地排干围圩，第二遍圩田的格先占（额度之内）
    from .waterworks import polder_plan
    take1 = np.zeros(H * W, dtype=bool)
    take1[take] = True
    PP = polder_plan(g, wc, wet, take1, n_quota - n_g) if wc is not None else \
        {"patches": [], "cells": none, "polder_id": np.zeros((H, W), dtype=np.int32), "wetland_cells": int(wet.sum())}
    pcells = PP["cells"]
    n_p = int(pcells.size)
    if n_p:
        pick1, cnt1, hh1, kept, in_kept, take, forced = fill_pass(pcells)
    g["polder_id"] = PP["polder_id"]
    cultivated = np.zeros(H * W, dtype=np.uint8)
    cultivated[take] = cult.ravel()[take]
    cultivated[pcells] = 1                        # 圩田：平地、水田
    cultivated = cultivated.reshape(H, W)
    has_cult = np.zeros(n_lab + 1, dtype=bool)
    has_cult[np.unique(labf[take])] = True
    has_cult[0] = False
    # ---- 撂荒：留下的片里接着往外的一层（离在种的田 ≤ fallow_ring_cells）
    ring_n = int(round(n_quota * float(sc["fallow_frac"])))
    near_c = binary_dilate(cultivated > 0, int(sc["fallow_ring_cells"])).ravel()
    after = in_kept[n_quota - n_g - n_p:]
    ring = after[near_c[after]][:ring_n]
    # ---- 废村：让出来的片里头一轮分得最多的几片
    cand = [t for t in range(1, n_lab + 1) if ok_t[t] and not kept[t] and not has_cult[t] and hh1[t] >= float(sc["ruin_min_hh"])]
    cand.sort(key=lambda t: (-int(cnt1[t]), t))
    chosen = cand[: int(sc["ruin_max"])]
    fallow = np.zeros(H * W, dtype=np.uint8)
    ylo, yhi = (int(x) for x in sc["ruin_years"])
    ruins = []
    for t in chosen:
        years = int(rng.integers(ylo, yhi + 1))
        cells = pick1[labf[pick1] == t]
        fallow[cells] = min(255, years)
        k_isl = int(island_id.ravel()[cells[0]])
        ruins.append({"tract": int(t), "island": k_isl, "abandoned_years": years, "households_before": int(round(float(hh1[t]))),
                      "cells": cells, "fallow_km2": round(float(cells.size) * cell_km2, 3)})
    ring_mask = np.zeros(H * W, dtype=bool)
    ring_mask[ring] = True
    ring_mask = ring_mask.reshape(H, W)
    rl, n_rl = label_by_island(ring_mask, island_id, 8)
    ymax = int(sc["fallow_years_max"])
    ys = [int(rng.integers(1, ymax + 1)) for _ in range(n_rl)]
    if n_rl:
        rlf = rl.ravel()
        yv = np.array([0] + ys, dtype=np.int64)
        fallow[ring] = np.minimum(255, yv[rlf[ring]]).astype(np.uint8)
    fallow = fallow.reshape(H, W)
    g["cultivated"] = cultivated
    g["fallow_years"] = fallow
    # ---- 地表：上等地没人种的回原本的地表；已垦画成可耕地 / 梯田；撂荒按年头
    cover = g["landcover"]
    nat = g["cover_natural"]
    prime = g["arable"] > 0
    f = fallow > 0
    back = prime & (cultivated == 0) & ~f
    cover[back] = nat[back]
    cover[f] = nat[f]
    woody = (nat == LC_FOREST) | (nat == LC_SHRUB)
    cover[f & (fallow > int(sc["fallow_grass_years"])) & (fallow <= int(sc["fallow_shrub_years"])) & woody] = LC_SHRUB
    cover[f & (fallow <= int(sc["fallow_grass_years"]))] = LC_GRASS
    cover[cultivated == 1] = LC_ARABLE
    cover[cultivated == 2] = LC_TERRACE
    if R and "patch_id" in g:                     # 开成田 / 撂荒还没长回林子的格：从林场划掉（面积由 sync_resources 重数）
        tids = [d["id"] for d in R["deposits"] if d["kind"] == "timber"]
        pid = g["patch_id"]
        pid[((cultivated > 0) | f) & (cover != LC_FOREST) & np.isin(pid, tids)] = -1
        if n_p:                                   # 排干围成圩田的湿地：从泥炭 / 芦苇荡的片里划掉
            rids = [d["id"] for d in R["deposits"] if d["kind"] == "peat"]
            pid[(PP["polder_id"] > 0) & np.isin(pid, rids)] = -1
    summary = {"quota_km2": round(float(n_quota) * cell_km2, 3),
               "cultivable_km2": round(float((cult > 0).sum()) * cell_km2, 3),
               "cultivated_km2": round(float((cultivated > 0).sum()) * cell_km2, 3),
               "fallow_km2": round(float(f.sum()) * cell_km2, 3),
               "fallow_ring_km2": round(float(ring.size) * cell_km2, 3),
               "ruin_fallow_km2": round(float(sum(r["cells"].size for r in ruins)) * cell_km2, 3),
               "forced_km2": round(float(forced) * cell_km2, 3),
               "n_tracts": int(n_lab), "n_tracts_ok": int(ok_t.sum()), "n_tracts_kept": int(kept.sum()),
               "n_tracts_dropped": int((ok_t & ~kept).sum()), "n_ruins": len(ruins),
               "settle_min_hh": int(min_hh), "rain_ok": rain_ok,
               "floor_islands": g_isl, "floor_km2": round(float(n_g) * cell_km2, 3),
               "wetland_km2": round(float(PP["wetland_cells"]) * cell_km2, 3), "polder_km2": round(float(n_p) * cell_km2, 3)}
    return {"ruins": ruins, "summary": summary, "polders": PP}


def floor_cells(g: dict, sc: dict, lab: np.ndarray, n_lab: int, ok_t: np.ndarray, order: np.ndarray, t_ord: np.ndarray, pit: np.ndarray,
                land_per_hh: float, n_quota: int) -> tuple[np.ndarray, list[int]]:
    """大岛保底（用户 09-29 定）：主岛以外 ≥ island_floor_km2 的岛，在它合门槛的宜垦片里挑最好的那片（片里最好的格排得最前），
    从片里最好的能开的格起按（适宜度降序、格号升序）往 8 邻域的能开的格长，长成连成一块的 need 格（= island_floor_hh 户的地）；
    长不到（能开的格被岩类赋存、采场隔成小块）就从片里下一个没走到的最好的格重长，整片都不行换下一片。
    各岛按「最好的格排在好地先占的第几」先后分，额度不够就停。返回（保底的格，按分的先后；分到的岛号）。"""
    import heapq
    I = g["json"]["islands"]
    island_id = g["island_id"]
    H, W = island_id.shape
    res_km = float(g["res_km"])
    cell_km2 = res_km * res_km
    need = int(math.ceil(float(sc["island_floor_hh"]) * land_per_hh / cell_km2 - 1e-9))
    floor_km2 = float(sc["island_floor_km2"])
    if need <= 0 or n_lab == 0:
        return np.zeros(0, dtype=np.int64), []
    labf = lab.ravel()
    ok_cell = (g["cultivable"].ravel() > 0) & ~pit.ravel()
    elig_cnt = np.bincount(t_ord, minlength=n_lab + 1)
    tracts, first = np.unique(t_ord, return_index=True)     # 每片在好地先占里最好的格的名次
    by_isl: dict[int, list[tuple[int, int]]] = {}
    iflat = island_id.ravel()
    for t, f in zip(tracts.tolist(), first.tolist()):
        if t == 0 or not ok_t[t] or elig_cnt[t] < need:
            continue
        k = int(iflat[int(order[f])])
        if k == 0 or float(I[k]["area_km2"]) < floor_km2:
            continue
        by_isl.setdefault(k, []).append((int(f), int(t)))
    plan = sorted((min(v)[0], k) for k, v in by_isl.items())
    out, isl, total = [], [], 0
    for _f, k in plan:
        if total + need > n_quota:
            break
        suit = g["suit"].ravel()
        done = False
        for f, t in sorted(by_isl[k]):
            seen = set()
            for s0 in order[t_ord == t].tolist():
                if s0 in seen:
                    continue
                got = []
                seen.add(s0)
                heap = [(-float(suit[s0]), s0)]
                while heap and len(got) < need:
                    _, q = heapq.heappop(heap)
                    got.append(q)
                    qi, qj = divmod(q, W)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            a, b = qi + di, qj + dj
                            if (di or dj) and 0 <= a < H and 0 <= b < W:
                                p = a * W + b
                                if p not in seen and labf[p] == t and ok_cell[p]:
                                    seen.add(p)
                                    heapq.heappush(heap, (-float(suit[p]), p))
                if len(got) == need:
                    out.extend(got)
                    isl.append(k)
                    total += need
                    done = True
                    break
            if done:
                break
    return np.array(out, dtype=np.int64), isl


def place_ruins(g: dict, sc: dict, ruins: list[dict], ok_site: np.ndarray, score_base: np.ndarray, taken: np.ndarray, km) -> None:
    """废村的村址：撂荒田 site_reach_cells 内同岛、能建村、没被占的格里按村址评分 + 近田取最高（平局按行列序）；没有就落在头一块田上。"""
    island_id = g["island_id"]
    H, W = island_id.shape
    reach = int(sc["site_reach_cells"])
    w_field = float(sc["site_weights"]["field"])
    for r in ruins:
        cells = r.pop("cells")
        ii, jj = cells // W, cells % W
        r0, r1 = max(0, int(ii.min()) - reach), min(H, int(ii.max()) + reach + 1)
        c0, c1 = max(0, int(jj.min()) - reach), min(W, int(jj.max()) + reach + 1)
        fm = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        fm[ii - r0, jj - c0] = True
        dfield = distance_bands(fm, reach)
        cand = ok_site[r0:r1, c0:c1] & (dfield <= reach) & (island_id[r0:r1, c0:c1] == r["island"]) & ~taken[r0:r1, c0:c1]
        if cand.any():
            s = np.where(cand, score_base[r0:r1, c0:c1] + w_field * (1.0 - dfield / (reach + 1.0)), -np.inf)
            p = int(np.argmax(s))
            gi, gj = r0 + p // (c1 - c0), c0 + p % (c1 - c0)
        else:
            gi, gj = int(ii[0]), int(jj[0])
        taken[gi, gj] = True
        r.update({"cell": [gi, gj], "km": km(gi, gj), "elev_m": round(float(g["height"][gi, gj]), 0)})
    ruins.sort(key=lambda r: (r["island"], r["cell"]))
    for k, r in enumerate(ruins):
        r["id"] = k + 1
        r["name"] = f"废村{k + 1:02d}"
        r["note"] = ruin_note(r["abandoned_years"], r["households_before"], r["fallow_km2"])


def special_occupancy(specials: list[dict], villages: list[dict], hamlets: list[dict]) -> None:
    """专业聚落的住法：烧炭营季节住；没有村和散户的岛上的浮石采石村 / 矿村 / 窑村是工棚（白天来、晚上走）；其余常住。
    工棚与季节住的人算在最近的村（home_village）：村记 households_workers / households_seasonal。"""
    farm_isl = {v["island"] for v in villages} | {h["island"] for h in hamlets}
    vc = np.array([v["cell"] for v in villages], dtype=np.int64).reshape(-1, 2)
    for s in specials:
        if s["kind"] in SEASONAL_KINDS:
            occ = "季节住"
        elif s["kind"] in WORKCAMP_KINDS and s["island"] not in farm_isl:
            occ = "工棚"
        else:
            occ = "常住"
        s["occupancy"] = occ
        if occ == "常住":
            continue
        s["home_village"] = None
        if vc.shape[0]:
            d = ((vc - np.array(s["cell"], dtype=np.int64)) ** 2).sum(1)
            v = villages[int(np.argmin(d))]
            s["home_village"] = v["id"]
            key = "households_workers" if occ == "工棚" else "households_seasonal"
            v[key] = v.get(key, 0) + int(s["households"])


def _island_cells(island_id: np.ndarray, n: int) -> list[np.ndarray]:
    flat = island_id.ravel()
    idx = np.flatnonzero(flat >= 0)
    o = idx[np.argsort(flat[idx], kind="stable")]
    b = np.searchsorted(flat[o], np.arange(n + 1))
    return [o[b[k]:b[k + 1]] for k in range(n)]


def island_uses(g: dict, sc: dict, rng, villages: list[dict], hamlets: list[dict], specials: list[dict], ruins: list[dict],
                seat: dict | None, km) -> tuple[list[dict], list[str]]:
    """没人常住的岛里挑少数有人用的：放牧 / 夏牧、烽火台、庙、墓岛。返回 (uses, 每岛的状态 常住 / 季节住 / 有人用 / 荒岛)。"""
    J = g["json"]
    I = J["islands"]
    n = len(I)
    island_id = g["island_id"]
    H, W = island_id.shape
    res_km = float(g["res_km"])
    cell_km2 = res_km * res_km
    resident = [False] * n
    seasonal = [False] * n
    used = [False] * n
    for v in villages + hamlets:
        resident[v["island"]] = True
    for s in specials:
        if s["occupancy"] == "常住":
            resident[s["island"]] = True
        elif s["occupancy"] == "季节住":
            seasonal[s["island"]] = True
        else:
            used[s["island"]] = True
    for r in ruins:
        used[r["island"]] = True
    r_shrine, r_tomb = float(rng.random()), float(rng.random())
    uses = []
    if not villages:
        return uses, _status(resident, seasonal, used)
    cells = _island_cells(island_id, n)
    cover = g["landcover"].ravel()
    hflat = g["height"].ravel()
    cliff = g["cliff"].ravel()
    free = [k for k in range(1, n) if not (resident[k] or seasonal[k] or used[k]) and cells[k].size]
    vkm = [(float(v["km"][0]), float(v["km"][1]), v["id"]) for v in villages]

    def nearest_village(k):
        cx, cy = float(I[k]["center_km"][0]), float(I[k]["center_km"][1])
        best, vid = math.inf, None
        for x, y, i in vkm:
            d = math.hypot(x - cx, y - cy)
            if d < best:
                best, vid = d, i
        return best, vid

    def centroid_cell(k, sel):
        c = cells[k][sel]
        ii, jj = c // W, c % W
        mi, mj = float(ii.sum()) / c.size, float(jj.sum()) / c.size
        di, dj = ii - mi, jj - mj
        d = di * di + dj * dj
        p = int(c[int(np.argmin(d))])
        return p // W, p % W

    def highest_cell(k):
        c = cells[k]
        p = int(c[int(np.argmax(hflat[c]))])
        return p // W, p % W

    def add(kind, k, cell, occ, vid, note, extra=None):
        i, j = int(cell[0]), int(cell[1])
        u = {"kind": kind, "island": k, "cell": [i, j], "km": km(i, j), "elev_m": round(float(g["height"][i, j]), 0),
             "occupancy": occ, "village": vid, "note": note}
        if extra:
            u.update(extra)
        uses.append(u)
        used[k] = True

    # 放牧 / 夏牧：草坡 / 高山草甸 / 灌丛够大、离最近的村不远；按离村近挑前 graze_max 个
    past = (cover == LC_GRASS) | (cover == LC_ALPINE) | (cover == LC_SHRUB)
    meadow = (cover == LC_GRASS) | (cover == LC_ALPINE)
    graze = []
    for k in free:
        pk = float(past[cells[k]].sum()) * cell_km2
        if pk < float(sc["graze_min_km2"]):
            continue
        d, vid = nearest_village(k)
        if d <= float(sc["graze_reach_km"]):
            graze.append((d, k, vid, pk, float(meadow[cells[k]].sum()) * cell_km2))
    graze.sort(key=lambda x: (x[0], x[1]))
    for d, k, vid, pk, mk in graze[: int(sc["graze_max"])]:
        shieling = mk >= float(sc["shieling_min_km2"])
        cell = centroid_cell(k, past[cells[k]])
        add("夏牧" if shieling else "放牧", k, cell, "季节住" if shieling else "只放牲口", vid, use_note("shieling" if shieling else "graze", pk, d),
            {"pasture_km2": round(pk, 3), "dist_km": round(d, 2)})
        if shieling:
            seasonal[k] = True
    # 烽火台：群边高处——离主岛最远的几座，彼此方位差 ≥ beacon_sep_deg
    c0 = I[0]["center_km"]
    sep = math.radians(float(sc["beacon_sep_deg"]))
    far = []
    for k in range(1, n):
        if used[k] or resident[k] or seasonal[k] or not cells[k].size:
            continue
        dx, dy = float(I[k]["center_km"][0]) - float(c0[0]), float(I[k]["center_km"][1]) - float(c0[1])
        far.append((-math.hypot(dx, dy), k, math.atan2(dy, dx)))
    far.sort(key=lambda x: (x[0], x[1]))
    angs = []
    for negd, k, a in far:
        if len(angs) >= int(sc["beacon_max"]):
            break
        ok = True
        for b in angs:
            dd = abs(a - b)
            if dd > math.pi:
                dd = 2.0 * math.pi - dd
            if dd < sep:
                ok = False
                break
        if not ok:
            continue
        angs.append(a)
        add("烽火台", k, highest_cell(k), "轮班", seat["id"] if seat else None, use_note("beacon", -negd))
    # 庙 / 墓岛：邑治附近没人住的岛（各按概率有没有）
    sx, sy = (float(seat["km"][0]), float(seat["km"][1])) if seat else (float(c0[0]), float(c0[1]))

    def nearest_free(max_km2=None):
        best = None
        for k in range(1, n):
            if used[k] or resident[k] or seasonal[k] or not cells[k].size:
                continue
            if max_km2 is not None and float(I[k]["area_km2"]) > max_km2:
                continue
            d = math.hypot(float(I[k]["center_km"][0]) - sx, float(I[k]["center_km"][1]) - sy)
            if best is None or d < best[0]:
                best = (d, k)
        return best

    if r_shrine < float(sc["shrine_p"]):
        b = nearest_free()
        if b:
            add("庙", b[1], highest_cell(b[1]), "香火", seat["id"] if seat else None, use_note("shrine", b[0]))
    if r_tomb < float(sc["tomb_p"]):
        b = nearest_free(float(sc["tomb_max_km2"]))
        if b:
            add("墓岛", b[1], highest_cell(b[1]), "无人", seat["id"] if seat else None, use_note("tomb", b[0]))
    for k, u in enumerate(uses):
        u["id"] = k + 1
    return uses, _status(resident, seasonal, used)


def _status(resident, seasonal, used) -> list[str]:
    return ["常住" if resident[k] else ("季节住" if seasonal[k] else ("有人用" if used[k] else "荒岛")) for k in range(len(resident))]


def land_tenure(g: dict, status: list[str], villages: list[dict], hamlets: list[dict], specials: list[dict], uses: list[dict],
                ruins: list[dict]) -> list[dict]:
    """荒地归谁（L29：地在册，不等于都种着）：每岛一条——有农户（村或散户）的岛：未垦的宜垦地归就近的村（大户 / 族产）；
    没有农户但有人用的岛（工棚、季节住、放牧 / 夏牧）：归用它的那个村（大户，领照开垦 / 租）；废村、只有常住专业聚落的岛与别的荒岛：官荒，归邑。"""
    island_id = g["island_id"]
    n = len(g["json"]["islands"])
    res_km = float(g["res_km"])
    cell_km2 = res_km * res_km
    flat = island_id.ravel()
    lf = flat >= 0
    cnt = lambda m: np.bincount(flat[lf & m.ravel()], minlength=n)
    cv, ct, fl = cnt(g["cultivable"] > 0), cnt(g["cultivated"] > 0), cnt(g["fallow_years"] > 0)
    user = {}
    for s in specials:
        if s["occupancy"] != "常住" and s.get("home_village") is not None:
            user.setdefault(s["island"], s["home_village"])
    for u in uses:
        if u["kind"] in ("放牧", "夏牧"):
            user.setdefault(u["island"], u["village"])
    ruin_isl = {r["island"] for r in ruins}
    farm_isl = {v["island"] for v in villages} | {h["island"] for h in hamlets}
    out = []
    for k in range(n):
        if k in farm_isl:
            owner, vid = "村", None
        elif status[k] != "荒岛" and k not in ruin_isl and k in user:
            owner, vid = "大户", user[k]
        else:
            owner, vid = "官荒", None
        out.append({"island": k, "status": status[k], "owner": owner, "village": vid,
                    "cultivable_km2": round(float(cv[k]) * cell_km2, 3), "cultivated_km2": round(float(ct[k]) * cell_km2, 3),
                    "fallow_km2": round(float(fl[k]) * cell_km2, 3)})
    return out
