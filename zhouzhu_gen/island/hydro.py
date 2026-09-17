"""5.3 水系、地表与可耕地：D8 流向 + 洼地填平 → 汇流 × 降水 = 径流 → 河（对齐 has_river / river_size）、溪涧、湖、集水盆地 →
地表分类（海拔温度、坡度、汇流、岛龄土层、迎风 / 背风）→ 可耕地按适宜度分位取到 arable_frac（误差 < 0.005）。

全群一张栅格；水文按岛分别做（每岛自成流域，虚空是出口）。行星层的相对降水按第八节 a 换算成 mm/年（只用于径流量与湿地判据）。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import N8, binary_dilate, binary_erode, distance_bands, label_components, shift, slope_deg
from .terrain import accumulate, d8, priority_fill

LC_VOID, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE = range(12)


def precip_mm(p_rel: float, c: dict) -> float:
    """第八节 a：0 → precip_mm_min，1 → precip_mm_max，中间按 p^exp。"""
    lo, hi, e = float(c["precip_mm_min"]), float(c["precip_mm_max"]), float(c["precip_mm_exp"])
    return lo + (hi - lo) * float(np.clip(p_rel, 0.0, 1.0)) ** e


def _wind_exposure(H: int, W: int, u: float, v: float, slope_dir_x, slope_dir_y):
    """迎风 = 坡面法线朝向来风（风向量 (u, v) 指向下游）。返回 [-1, 1]，正 = 迎风。"""
    sp = math.hypot(u, v)
    if sp < 1e-6:
        return np.zeros((H, W))
    ux, uy = -u / sp, -v / sp     # 指向上风的单位向量
    return np.clip(slope_dir_x * ux + slope_dir_y * uy, -1.0, 1.0)


def build_hydro(ctx, node: int, c: dict, g: dict, log=print) -> None:
    hc = c["hydro"]
    lc = c["landcover"]
    inp = g["inp"]
    res_km = g["res_km"]
    res_m = res_km * 1000.0
    cell_km2 = res_km * res_km
    height = g["height"]
    island_id = g["island_id"]
    H, W = height.shape
    land = island_id >= 0
    P_mm = precip_mm(inp["precip"], c["climate"])
    n_isl = len(g["json"]["islands"])

    # 风：④ 扰动后的风在群心处的值（迎风 / 背风打分）
    wl = ctx.load_npz(4, "wind_local")
    from ..sphere import grid_interp
    u = float(grid_interp(wl["u"], wl["lats"], wl["lons"], inp["lat"], inp["lon"]))
    v = float(grid_interp(wl["v"], wl["lats"], wl["lons"], inp["lat"], inp["lon"]))

    filled = np.full((H, W), np.nan)
    acc_km2 = np.zeros((H, W))
    lake = np.zeros((H, W), dtype=bool)
    river = np.zeros((H, W), dtype=np.uint8)      # 0 无 / 1 小河 / 2 中河 / 3 大河（常年）
    stream = np.zeros((H, W), dtype=np.uint8)     # 1 = 季节性溪涧
    basin_info = {}
    river_thr_km2 = None
    for k, J in enumerate(g["json"]["islands"]):
        m = island_id == k
        if not m.any():
            continue
        rr, cc = np.where(m)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        sl = (slice(r0, r1), slice(c0, c1))
        mk = m[sl]
        h = np.where(mk, height[sl], np.nan)
        hf = priority_fill(h, mk, eps=float(hc["fill_eps_m"]))
        # 湖：填平深度 ≥ lake_min_depth_m 且面积 ≥ lake_min_km2 的洼地
        depth = np.where(mk, hf - h, 0.0)
        pond = depth >= float(hc["lake_min_depth_m"])
        lk = np.zeros_like(mk)
        n_lakes = 0
        if pond.any():
            lab, nl = label_components(pond)
            if nl:
                cnt = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
                big = np.where(cnt * cell_km2 >= float(hc["lake_min_km2"]))[0] + 1
                lk = np.isin(lab, big)
                n_lakes = int(big.size)
        ri, rj, slope, _ = d8(hf, mk, res_m)
        A = accumulate(hf, mk, ri, rj)
        Akm = A * cell_km2
        filled[sl][mk] = hf[mk]
        acc_km2[sl][mk] = Akm[mk]
        lake[sl] |= lk
        # 湖面：高程抬到填平面（水面平），保持 height 的 NaN 语义
        height[sl][lk] = hf[lk]
        # 河的阈值：主岛按 has_river 定；小岛一律只有溪涧
        q = Akm * P_mm                          # km² × mm ≈ 千 m³/年 的量级（径流代理）
        thr_stream = float(hc["stream_min_km2"]) * P_mm
        if k == 0:
            if inp["has_river"]:
                # 常年河的阈值：默认 river_min_km2，但至少让主岛最大汇流的 river_reach_frac 成河（调阈值直到成立）
                thr_river = min(float(hc["river_min_km2"]), float(hc["river_reach_frac"]) * float(Akm.max()))
                river_thr_km2 = thr_river
                per = Akm >= thr_river
                # 分级：按 river_size（= 主岛面积 × 降水，行星层）定最大河的级别
                rs = inp["river_size"]
                top = 3 if rs >= float(hc["river_big_size"]) else (2 if rs >= float(hc["river_mid_size"]) else 1)
                amax = float(Akm.max())
                lvl = np.zeros_like(river[sl])
                lvl[per] = 1
                if top >= 2:
                    lvl[per & (Akm >= 0.3 * amax)] = 2
                if top >= 3:
                    lvl[per & (Akm >= 0.6 * amax)] = 3
                river[sl] = np.where(mk, lvl, river[sl]).astype(np.uint8)
                stream[sl] = np.where(mk & (Akm >= float(hc["stream_min_km2"])) & ~per, 1, stream[sl]).astype(np.uint8)
            else:
                stream[sl] = np.where(mk & (Akm >= float(hc["stream_min_km2"])), 1, stream[sl]).astype(np.uint8)
            # 集水盆地：主岛按出口分水岭；出口 = 流向虚空的格；沿岸线把出口聚成段（相邻 basin_merge_cells 内的出口算同一盆地）
            basin_info = _basins(hf, mk, ri, rj, Akm, cell_km2, hc)
        else:
            stream[sl] = np.where(mk & (Akm >= float(hc["stream_min_km2"])), 1, stream[sl]).astype(np.uint8)
        J["n_lakes"] = n_lakes
        J["lake_km2"] = round(float(lk.sum()) * cell_km2, 3)
        J["max_flowacc_km2"] = round(float(Akm.max()), 2)
        J["has_perennial_river"] = bool((river[sl][mk] > 0).any())
        J["has_stream"] = bool((stream[sl][mk] > 0).any())
    river[lake] = 0
    stream[lake] = 0

    # ---------- 地表分类 ----------
    slope = slope_deg(np.where(land, height, np.nan), land, res_m)
    T_sea = inp["temp_sea"]
    lapse = inp["lapse_c_per_km"]
    T = T_sea - lapse * np.where(land, height, 0.0) / 1000.0
    # 坡向 → 迎风性
    hh = np.where(land, height, np.nan)
    def nb(di, dj):
        a = shift(hh, di, dj, np.nan)
        return np.where(np.isnan(a), hh, a)
    gx = (nb(0, -1) - nb(0, 1)) / (2 * res_m)     # 东向坡降为正时地面向东下降 → 法线朝东
    gy = (nb(-1, 0) - nb(1, 0)) / (2 * res_m)     # 行向下 = 南：向北下降 → 法线朝北
    # 法线的水平方向 = 下坡方向 = -grad；下坡方向向量 (dx, dy) = (gx_eastdown, gy_northdown)
    gn = np.hypot(gx, gy)
    dx = np.where(gn > 1e-9, gx / np.maximum(gn, 1e-9), 0.0)
    dy = np.where(gn > 1e-9, gy / np.maximum(gn, 1e-9), 0.0)
    expo = _wind_exposure(H, W, u, v, dx, dy) * np.clip(slope / 15.0, 0.0, 1.0)
    # 土层厚度：老岛厚、平地厚、汇流处厚
    age_arr = np.zeros((H, W))
    for k, J in enumerate(g["json"]["islands"]):
        age_arr[island_id == k] = J["age"]
    soil = np.clip(0.35 + 0.5 * age_arr, 0, 1) * np.clip(1.0 - slope / 40.0, 0.0, 1.0) * (0.7 + 0.3 * np.clip(np.log1p(acc_km2) / 4.0, 0, 1))
    wet = np.clip(P_mm / 1500.0, 0.2, 2.0) * (1.0 + 0.4 * expo)
    dist_water = distance_bands((river > 0) | lake, int(lc["water_near_cells"]))
    near_water = np.clip(1.0 - dist_water / (int(lc["water_near_cells"]) + 1.0), 0.0, 1.0) ** 0.5

    cover = np.full((H, W), LC_VOID, dtype=np.uint8)
    cover[land] = LC_GRASS
    cover[land & (T < float(lc["alpine_temp_c"]))] = LC_ALPINE
    cover[land & (T < float(lc["rock_temp_c"]))] = LC_ROCK
    cover[land & (slope >= float(lc["rock_slope_deg"]))] = LC_ROCK
    # 林地要够湿、土够厚、不太冷；薄土（山脊、陡坡、新岛）与半干处成灌丛；干处 / 极薄土成草坡
    warm = land & (T >= float(lc["forest_temp_min_c"])) & (slope < float(lc["rock_slope_deg"]))
    forest = warm & (wet >= float(lc["forest_wet_min"])) & (soil >= float(lc["forest_soil_min"]))
    shrub = land & (T >= float(lc["alpine_temp_c"])) & (slope < float(lc["rock_slope_deg"])) & ~forest & (wet >= float(lc["shrub_wet_min"])) & (soil >= 0.5 * float(lc["forest_soil_min"]))
    cover[shrub] = LC_SHRUB
    cover[forest] = LC_FOREST
    wetland = land & (slope < float(lc["wet_slope_deg"])) & (acc_km2 >= float(lc["wet_acc_km2"])) & (P_mm >= float(lc["wet_precip_mm"]))
    cover[wetland] = LC_WET
    cover[g["cliff"]] = LC_CLIFF
    # ---------- 可耕地：适宜度分位 ----------
    # 适宜度只用于排名（可耕率是行星层给定的约束）：各项都是软打分、处处 > 0，寒冷 / 陡峭的群也能取到该有的比例
    suit = (0.02 + np.clip(1.0 - slope / float(lc["arable_slope_max_deg"]), 0.0, 1.0) ** 1.5
            * (0.1 + 0.9 * np.clip((T - float(lc["arable_temp_min_c"])) / 8.0, 0.0, 1.0))
            * (0.4 + 0.6 * soil) * (0.8 + 0.2 * near_water) * np.clip(wet / 0.6, 0.2, 1.2))
    suit = np.where(land & ~g["cliff"] & ~lake & (river == 0) & ~wetland, suit, -1.0)
    n_land = int(land.sum())
    n_arable = int(round(inp["arable_frac"] * n_land))
    arable = np.zeros((H, W), dtype=np.uint8)
    if n_arable > 0:
        flat = suit.ravel()
        order = np.argsort(-flat, kind="stable")[:n_arable]
        order = order[flat[order] > 0]
        a = np.zeros(H * W, dtype=bool)
        a[order] = True
        a = a.reshape(H, W)
        arable[a] = 1
        arable[a & (slope >= float(lc["terrace_slope_deg"]))] = 2
    cover[arable == 1] = LC_ARABLE
    cover[arable == 2] = LC_TERRACE
    cover[river > 0] = LC_RIVER
    cover[lake] = LC_LAKE

    g.update({"flowacc_km2": acc_km2.astype(np.float32), "river": river, "stream": stream, "lake": lake,
              "landcover": cover, "arable": arable, "slope_deg": slope.astype(np.float32), "filled": filled})
    from .output import LANDCOVER_CLASSES
    share = {LANDCOVER_CLASSES[i]: round(float(((cover == i) & land).sum()) / max(1, n_land), 4) for i in range(1, 12)}
    J = g["json"]
    J["landcover"] = {"classes": LANDCOVER_CLASSES, "share": share,
                      "note": "landcover.png 索引色；调色板见 palette", "palette": None}
    from .output import LANDCOVER_PALETTE
    J["landcover"]["palette"] = LANDCOVER_PALETTE
    J["hydro"] = {"precip_mm": round(P_mm, 0), "river_threshold_km2": None if river_thr_km2 is None else round(river_thr_km2, 2),
                  "main_max_flowacc_km2": J["islands"][0]["max_flowacc_km2"],
                  "n_lakes": int(sum(i["n_lakes"] for i in J["islands"])),
                  "lake_km2": round(float(lake.sum()) * cell_km2, 3),
                  "main_basins": basin_info,
                  "wind_ms": [round(u, 2), round(v, 2)],
                  "river_levels": {"1": "小河", "2": "中河", "3": "大河", "stream": "季节性溪涧（water.png 值 1）"}}
    J["constraints"]["arable_frac"]["actual"] = round(float((arable > 0).sum()) / max(1, n_land), 4)
    J["constraints"]["has_river"]["actual"] = bool(J["islands"][0]["has_perennial_river"])
    log(f"  水系：降水 {P_mm:.0f} mm，主岛河 {'有' if J['constraints']['has_river']['actual'] else '无'}（阈 {river_thr_km2}），湖 {J['hydro']['n_lakes']}，盆地 {basin_info.get('n_large', 0)} 大；"
        f"可耕 {J['constraints']['arable_frac']['actual']:.4f} / {inp['arable_frac']:.4f}")


def _basins(hf, mk, ri, rj, Akm, cell_km2, hc) -> dict:
    """主岛集水盆地：每格顺流到出口；只有汇流 ≥ basin_mouth_km2（或岛面积的 basin_mouth_frac）的出口算「河口」，
    其集水区为盆地；面积 ≥ basin_large_frac × 岛面积的为「大盆地」（R9 推论 2 的素材，≥ 2 个时标出）。"""
    H, W = hf.shape
    idx = np.where(mk.ravel())[0]
    recv = np.where(ri.ravel() >= 0, ri.ravel() * W + rj.ravel(), -1)
    order = idx[np.argsort(np.where(mk, hf, np.inf).ravel()[idx], kind="stable")]
    rl = recv.tolist()
    ol = [-1] * (H * W)
    for k in order.tolist():
        r = rl[k]
        ol[k] = k if r < 0 else ol[r]
    outlet = np.array(ol)
    total = float(mk.sum()) * cell_km2
    thr = max(float(hc["basin_mouth_km2"]), float(hc["basin_mouth_frac"]) * total)
    mouths = np.where(mk.ravel() & (Akm.ravel() >= thr) & (recv < 0))[0]
    sizes = np.array([float(Akm.ravel()[m]) for m in mouths])
    large = np.where(sizes >= float(hc["basin_large_frac"]) * total)[0]
    return {"n_basins": int(mouths.size), "n_large": int(large.size),
            "basin_km2": [round(float(x), 1) for x in sorted(sizes.tolist(), reverse=True)[:8]],
            "large_km2": [round(float(sizes[i]), 1) for i in sorted(large, key=lambda i: -sizes[i])],
            "largest_frac": round(float(sizes.max() / total), 3) if mouths.size else 0.0}
