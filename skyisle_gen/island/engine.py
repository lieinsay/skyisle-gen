"""后端开关与 C++ 桥（docs/PLAN-CORE.md 第六节；行星计划 P6a）。

`[engine] backend = "python" | "cpp"`：读法与 `[island]` 同（default.toml ← run 的快照 ← `--set engine.backend=cpp`，`--backend` 是简写），
不进任何阶段的缓存 key。分派点只有两处：`island.build_terrain` 与 `hydro.build_hydro` 的第一行。
cpp 后端调 `skyisle_gen._core`（core/，`python core/build.py` 编），这里把输入备好、把结果拼回与 Python 版同形的 g：
数组同 dtype，island.json 同键序、同 round 位数（C++ 只给原始的双精度数）。后面的步（资源、气候、天气、聚落、出图）照旧走 Python。
"""
from __future__ import annotations

import math

import numpy as np

AGE_ZH = {"young": "新岛", "mid": "中年", "old": "老岛"}
_PLANET_CACHE: dict = {}


def backend(ctx) -> str:
    b = str((ctx.cfg.get("engine") or {}).get("backend", "python")).lower()
    if b not in ("python", "cpp"):
        raise ValueError(f"[engine] backend 只能是 python 或 cpp，得到 {b!r}")
    return b


def threads(ctx) -> int:
    return max(1, int((ctx.cfg.get("engine") or {}).get("threads", 4)))


def core():
    try:
        from .. import _core
    except ImportError as e:            # 不静默退回 Python：免得以为跑的是 C++
        raise RuntimeError("[engine] backend = \"cpp\"，但 C++ 扩展 skyisle_gen._core 没编：在仓库根下运行 "
                           "`python core/build.py`（见 README「C++ 核心库」）") from e
    return _core


def flat_config(c: dict) -> dict:
    """[island] 段展平成 {"num": {"layout.n0": 30.0, …}, "vec": {"territory.stretch": [1.0, 1.6, 2.4], …}}（布尔 → 0 / 1，字符串跳过）。"""
    num, vec = {}, {}

    def walk(d, prefix):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(v, key + ".")
            elif isinstance(v, bool):
                num[key] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                num[key] = float(v)
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                vec[key] = [float(x) for x in v]
    walk(c, "")
    return {"num": num, "vec": vec}


def _grid_spec(lats: np.ndarray, lons: np.ndarray) -> dict:
    # 与 sphere.grid_interp 同：dlat = lats[1] − lats[0] 按原数组的 dtype 算
    return {"lat0": float(lats[0]), "dlat": float(lats[1] - lats[0]), "lon0": float(lons[0]), "dlon": float(lons[1] - lons[0]),
            "nlat": int(lats.size), "nlon": int(lons.size)}


def planet_view(ctx) -> dict:
    """行星层的网格与全体群：板块走向（boundary_axis）、势力范围（territory.limits）、局地风（hydro 的迎风 / 背风）、一年的秒数。按 run 缓存。"""
    key = str(ctx.out_dir.resolve())
    pv = _PLANET_CACHE.get(key)
    if pv is not None:
        return pv
    plates = ctx.load_npz(3, "plates")
    isl = ctx.load_npz(3, "islands")
    wl = ctx.load_npz(4, "wind_local")
    planet = ctx.load_json(1, "planet")
    cal = planet.get("calendar", {})
    pv = {
        "radius_km": float(planet["radius_km"]),
        "year_s": float(cal.get("year_days_solar", 336.0)) * float(cal.get("solar_day_hr", 24.0)) * 3600.0,
        "plates": {**_grid_spec(plates["lats"], plates["lons"]),
                   "K": np.ascontiguousarray(plates["boundary_kernel"], dtype=np.float64),
                   "btype": np.ascontiguousarray(plates["btype"], dtype=np.int32),
                   "lats": np.ascontiguousarray(plates["lats"], dtype=np.float64),
                   "lons": np.ascontiguousarray(plates["lons"], dtype=np.float64)},
        "islands": {"lat": np.ascontiguousarray(isl["lat"], dtype=np.float64), "lon": np.ascontiguousarray(isl["lon"], dtype=np.float64),
                    "area": np.ascontiguousarray(isl["area_km2"], dtype=np.float64)},
        "wind": {**_grid_spec(wl["lats"], wl["lons"]), "u": np.ascontiguousarray(wl["u"], dtype=np.float64),
                 "v": np.ascontiguousarray(wl["v"], dtype=np.float64)},
    }
    _PLANET_CACHE.clear()
    _PLANET_CACHE[key] = pv
    return pv


def inputs(ctx, node: int, inp: dict) -> dict:
    return {"node": int(node), "seed": int(ctx.seed), "lat": inp["lat"], "lon": inp["lon"], "area_km2": inp["area_km2"],
            "main_area_km2": inp["main_area_km2"], "height_m": inp["height_m"], "age": inp["age"], "layered": bool(inp["layered"]),
            "keel_clearance_m": inp["keel_clearance_m"], "area_median_km2": inp["area_median_km2"],
            "precip": inp["precip"], "temp_sea": inp["temp_sea"], "lapse_c_per_km": inp["lapse_c_per_km"],
            "arable_frac": inp["arable_frac"], "river_size": inp["river_size"], "has_river": bool(inp["has_river"])}


# ---------------------------------------------------------------- 第 1 步：布局 + 岛形 + 高程
def build_terrain_cpp(ctx, node: int, c: dict, inp: dict, res_m: float | None = None, log=print) -> dict:
    from ..stages.s03_islands import CLASS_NAMES, CLASS_ZH
    R = core().build_terrain(inputs(ctx, node, inp), planet_view(ctx), flat_config(c), float(res_m or 0.0), threads(ctx))
    res_km = float(R["res_km"])
    n = int(R["n"])
    height, island_id, cliff = R["height"], R["island_id"], R["cliff"]
    H, W = height.shape
    land = island_id >= 0
    x0, y0 = float(R["x0"]), float(R["y0"])
    res_m_eff = res_km * 1000.0
    islands_json = []
    for k, e in enumerate(R["islands"]):
        rim, peak, keel_k, surf = float(e["rim"]), float(e["peak"]), float(e["keel"]), float(e["surface"])
        r0, c0, m, _m = (int(v) for v in e["bbox"])
        islands_json.append({
            "id": k, "is_main": k == 0, "area_km2": round(float(e["area_cells"]) * res_km * res_km, 3),
            "area_target_km2": round(float(e["area_target"]), 3),
            "center_km": [round(float(e["cx"]), 3), round(float(e["cy"]), 3)],
            "surface_m": round(surf, 1), "relief_m": round(peak - rim, 1), "relief_target_m": round(float(e["relief_target"]), 1),
            "peak_m": round(peak, 1), "rim_m": round(rim, 1), "keel_m": round(keel_k, 1), "cliff_m": round(rim - keel_k, 1),
            "age": round(float(e["age"]), 3), "age_zh": AGE_ZH[e["kind"]],
            "bbox_cells": [r0, c0, m, m],
        })
    rims = np.asarray(R["rims"], dtype=np.float64)
    lk = []
    for e in R["links"]:
        d = {"a": int(e["a"]), "b": int(e["b"]), "gap_km": round(float(e["gap"]), 3), "kind": "bridge" if e["bridge"] else "ferry",
             "dh_m": round(float(e["dh"]), 1)}
        if e["fallback"]:
            d["fallback"] = True
        lk.append(d)
    tree = [(int(a), int(b)) for a, b in R["tree"]]
    i0 = islands_json[0]
    secs = R["seconds"]
    log(f"  地形 {n} 岛 {H}×{W} @ {res_m_eff:.0f} m，{secs[1]:.1f} s（C++，布局 {secs[0]:.1f} s）；主岛 岸缘 {i0['rim_m']:.0f} → 峰 {i0['peak_m']:.0f} m"
        f"（起伏 {i0['relief_m']:.0f} / 目标 {i0['relief_target_m']:.0f}）")
    btype = int(R["btype"])
    meta = {"node": node, "seed": ctx.seed, "run": ctx.out_dir.name, "lat": inp["lat"], "lon": inp["lon"],
            "cls": CLASS_NAMES[inp["cls"]], "cls_zh": CLASS_ZH[CLASS_NAMES[inp["cls"]]],
            "age": inp["age"], "age_zh": AGE_ZH[R["node_kind"]],
            "layered": inp["layered"], "plate": inp["plate"], "boundary_type": ["汇聚", "离散", "走滑"][btype],
            "boundary_kernel": round(float(R["kernel"]), 3), "boundary_axis_deg": round(math.degrees(float(R["axis"])), 1), "res_m": res_m_eff,
            "generator": "skyisle_gen.island", "layer": "第三层（按需生成，不回灌）", "engine": "cpp"}
    km_per_deg = 2 * math.pi * float(inp["planet"]["radius_km"]) / 360.0
    raster = {"res_m": res_m_eff, "rows": H, "cols": W, "origin_km": [round(x0, 3), round(y0, 3)],
              "origin_note": "origin_km = 栅格左上角相对群心（节点经纬度）的平面坐标（km，x 东 y 北）；行 r 列 c 的格心 = origin + ((c+0.5)·res, −(r+0.5)·res)",
              "km_per_deg_lat": round(km_per_deg, 3), "km_per_deg_lon": round(km_per_deg * math.cos(math.radians(inp["lat"])), 3),
              "height_note": "height.png 16 位灰度 = 云带顶以上高度 / height_png_scale_m_per_unit；0 = 虚空",
              "void_value": 0}
    T = R["territory"]
    terr = {"neighbours": int(T["neighbours"]), "gap_km": float(T["gap_km"]), "constrained": bool(T["constrained"])}
    if T["constrained"]:
        terr.update({"main_offset_km": [round(float(T["off_x"]), 3), round(float(T["off_y"]), 3)],
                     "main_turn_deg": round(float(T["turn_deg"]), 1), "main_stretch": float(T["stretch"]),
                     "before_km": round(float(T["before"]), 3), "after_km": round(float(T["after"]), 3)})
    if T["has_violation"]:
        terr["violation_km"] = round(float(T["violation"]), 3)
        terr["note"] = "势力范围（territory.py）：与每个邻群按等效半径分界、各退 gap/2；violation_km ≤ 0 = 没越界（负值是最小余量）"
    constraints = {
        "area_km2": {"target": round(inp["area_km2"], 2), "actual": round(float(land.sum()) * res_km * res_km, 2)},
        "main_area_km2": {"target": round(inp["main_area_km2"], 2), "actual": islands_json[0]["area_km2"]},
        "height_m": {"target": round(inp["height_m"], 1), "actual": round(float(np.nanmedian(height[island_id == 0])), 1),
                     "note": "③ 的 height_m = 主岛台面高度 = 陆地高程中位数（④ 的岛上气温在这个高度）；峰高见 peak_m"},
        "peak_m": {"actual": round(float(np.nanmax(height[island_id == 0])), 1), "relief_m": islands_json[0]["relief_m"],
                   "relief_target_m": islands_json[0]["relief_target_m"]},
        "arable_frac": {"target": round(inp["arable_frac"], 4)},
        "has_river": {"target": inp["has_river"]},
        "n_islands": n,
        "territory": terr,
    }
    J = {"meta": meta, "raster": raster, "constraints": constraints, "islands": islands_json, "links": lk,
         "channels": [[int(a), int(b)] for a, b in tree],
         "layout": {"n_bridges": sum(1 for e in lk if e["kind"] == "bridge"), "n_ferries": sum(1 for e in lk if e["kind"] == "ferry"),
                    "gap_median_km": round(float(np.median([e["gap_km"] for e in lk])), 2) if lk else None,
                    "channel_islands": len(tree) + 1}}
    masks_pos = [(mk, int(r0), int(c0)) for mk, r0, c0 in R["masks_pos"]]
    g = {"height": height, "island_id": island_id, "cliff": cliff, "json": J, "rims": rims, "res_km": res_km,
         "masks_pos": masks_pos, "inp": inp}
    return g


# ---------------------------------------------------------------- 第 2 步前半：水系、河道、地表、可耕地
def build_hydro_cpp(ctx, node: int, c: dict, g: dict, log=print) -> None:
    from .output import LANDCOVER_CLASSES, LANDCOVER_PALETTE
    hc = c["hydro"]
    inp = g["inp"]
    res_km = g["res_km"]
    cell_km2 = res_km * res_km
    J = g["json"]
    ox, oy = J["raster"]["origin_km"]
    state = {"inp": inputs(ctx, node, inp), "height": np.ascontiguousarray(g["height"], dtype=np.float64),
             "island_id": np.ascontiguousarray(g["island_id"], dtype=np.int16), "cliff": np.ascontiguousarray(g["cliff"], dtype=bool),
             "res_km": float(res_km), "origin_x": float(ox), "origin_y": float(oy),
             "islands": [{"rim_m": float(i["rim_m"]), "keel_m": float(i["keel_m"]), "age": float(i["age"])} for i in J["islands"]]}
    R = core().build_hydro(state, planet_view(ctx), flat_config(c), threads(ctx))
    island_id = g["island_id"]
    land = island_id >= 0
    height = R["height"]
    g["height"] = height
    for k, (Ji, e) in enumerate(zip(J["islands"], R["islands"])):
        if e is None:
            continue
        Ji["n_lakes"] = int(e["n_lakes"])
        Ji["lake_km2"] = round(float(e["lake_cells"]) * cell_km2, 3)
        Ji["max_flowacc_km2"] = round(float(e["max_flowacc"]), 2)
        Ji["has_perennial_river"] = bool(e["has_perennial"])
        Ji["has_stream"] = bool(e["has_stream"])
    dz = float(R["dz"])
    J0 = J["islands"][0]
    J0["rim_m"] = round(J0["rim_m"] + dz, 1)
    J0["peak_m"] = round(J0["peak_m"] + dz, 1)
    J0["cliff_m"] = round(J0["rim_m"] - J0["keel_m"], 1)
    # 主岛河口表（hydro.carve_channels 的 info["rivers"]）
    rivers_info = []
    for e in R["rivers"]:
        rivers_info.append({"mouth_cell": [int(e["mouth"][0]), int(e["mouth"][1])], "basin_km2": round(float(e["basin_km2"]), 1),
                            "length_km": round(float(e["length_km"]), 1), "discharge_m3s": round(float(e["discharge"]), 2),
                            "width_m": round(float(e["width"]), 1), "depth_m": round(float(e["depth"]), 2),
                            "level": int(e["level"]), "waterfall_m": round(float(e["waterfall"]), 0),
                            "incision_m": round(float(e["incision"]), 1)})
    rivers_info.sort(key=lambda r: -r["basin_km2"])
    # 主岛集水盆地（hydro._basins 同式）
    B = R["basins"]
    basin_info = {}
    if B["present"]:
        mouths = [(int(r), int(cc), float(a)) for r, cc, a in B["mouths"]]
        sizes = np.array([m_[2] for m_ in mouths], dtype=np.float64)
        total = float(B["total_km2"])
        large = np.where(sizes >= float(hc["basin_large_frac"]) * total)[0]
        order = np.argsort(-sizes, kind="stable") if len(mouths) else np.zeros(0, dtype=int)
        basin_info = {"n_basins": len(mouths), "n_large": int(large.size),
                      "mouths": [[mouths[i][0], mouths[i][1], round(float(sizes[i]), 1)] for i in order[:12]],
                      "basin_km2": [round(float(x), 1) for x in sorted(sizes.tolist(), reverse=True)[:8]],
                      "large_km2": [round(float(sizes[i]), 1) for i in sorted(large, key=lambda i: -sizes[i])],
                      "largest_frac": round(float(sizes.max() / total), 3) if len(mouths) else 0.0}
    river_lines = []
    for k, pts in R["lines"]:
        river_lines.append({"island": int(k), "pts": [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 1), int(p[3]),
                                                       round(float(p[4]), 2)] for p in pts.tolist()]})
    river, lake = R["river"], R["lake"]
    g["river_lines"] = river_lines
    g.update({"river_width_m": R["river_width_m"], "river_depth_m": R["river_depth_m"], "floodplain": R["floodplain"], "cut_m": R["cut_m"]})
    g.update({"flowacc_km2": R["flowacc_km2"], "river": river, "stream": R["stream"], "lake": lake,
              "landcover": R["landcover"], "arable": R["arable"], "slope_deg": R["slope_deg"], "filled": R["filled"],
              "recv_i": R["recv_i"], "recv_j": R["recv_j"], "route_h": R["route_h"]})
    cover, arable = R["landcover"], R["arable"]
    n_land = int(land.sum())
    share = {LANDCOVER_CLASSES[i]: round(float(((cover == i) & land).sum()) / max(1, n_land), 4) for i in range(1, 12)}
    J["landcover"] = {"classes": LANDCOVER_CLASSES, "share": share, "note": "landcover.png 索引色；调色板见 palette", "palette": None}
    J["landcover"]["palette"] = LANDCOVER_PALETTE
    P_mm = float(R["P_mm"])
    thr = float(R["river_thr"])
    river_thr_km2 = None if math.isnan(thr) else thr
    u, v = (float(x) for x in R["wind"])
    J["hydro"] = {"precip_mm": round(P_mm, 0), "river_threshold_km2": None if river_thr_km2 is None else round(river_thr_km2, 2),
                  "perennial_q_m3s": float(hc["river_min_q_m3s"]), "stream_min_km2": float(hc["stream_min_km2"]),
                  "runoff_coef": float(hc["runoff_coef"]),
                  "main_max_flowacc_km2": J["islands"][0]["max_flowacc_km2"],
                  "n_lakes": int(sum(i["n_lakes"] for i in J["islands"])),
                  "lake_km2": round(float(lake.sum()) * cell_km2, 3),
                  "main_basins": basin_info,
                  "rivers": rivers_info[:12],
                  "n_rivers": len(rivers_info),
                  "n_waterfalls": int(R["n_falls"]),
                  "river_km2": round(float((river > 0).sum()) * cell_km2, 3),
                  "floodplain_km2": round(float(g["floodplain"].sum()) * cell_km2, 3),
                  "max_incision_m": round(float(R["max_cut"]), 1),
                  "channel_note": "河宽 / 水深见 terrain.npz 的 river_width_m / river_depth_m（溪涧为湿季值）；height 在河道格是河床，水面 = 河床 + 水深；rivers[].waterfall_m = 河口跌下崖缘的落差",
                  "wind_ms": [round(u, 2), round(v, 2)],
                  "river_levels": {"1": "小河", "2": "中河", "3": "大河", "stream": "季节性溪涧（water.png 值 1）"}}
    J["constraints"]["arable_frac"]["actual"] = round(float((arable > 0).sum()) / max(1, n_land), 4)
    hm = height[island_id == 0]
    J["constraints"]["height_m"]["actual"] = round(float(np.nanmedian(hm)), 1)
    J["constraints"]["peak_m"]["actual"] = round(float(np.nanmax(hm)), 1)
    J["constraints"]["has_river"]["actual"] = bool(J["islands"][0]["has_perennial_river"])
    log(f"  水系（C++ {float(R['seconds']):.1f} s）：降水 {P_mm:.0f} mm，主岛河 {'有' if J['constraints']['has_river']['actual'] else '无'}"
        f"（阈 {'—' if river_thr_km2 is None else f'{river_thr_km2:.1f} km²'}），湖 {J['hydro']['n_lakes']}，盆地 {basin_info.get('n_large', 0)} 大；"
        f"可耕 {J['constraints']['arable_frac']['actual']:.4f} / {inp['arable_frac']:.4f}")
