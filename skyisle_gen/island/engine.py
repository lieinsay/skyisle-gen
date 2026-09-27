"""后端开关与 C++ 桥（docs/PLAN-CORE.md 第六节；行星计划 P6a / P6b）。

`[engine] backend = "python" | "cpp"`：读法与 `[island]` 同（default.toml ← run 的快照 ← `--set engine.backend=cpp`，`--backend` 是简写），
不进任何阶段的缓存 key。分派点：`island.generate`（P6b：整群进 C++）、`island.build_terrain` 与 `hydro.build_hydro`（粗版、对照、用时仍单独调）、
`weather.multi_year_stats`（IS-daily 的逐年模拟）、`lod.build_lod` 的块降采样、`climate.classify_all`（全量季型）。
cpp 后端调 `skyisle_gen._core`（core/，`python core/build.py` 编），这里把输入备好、把结果拼回与 Python 版同形的 g：
数组同 dtype，island.json 同键序、同 round 位数（C++ 只给原始的双精度数与 ASCII 代码，中文由 decode.py 译回）；写产物仍在 Python。
P6c：cpp 后端下行星层（PlanetView、本群的 NodeInputs）由 C++ 从 ①③④ 的产物对象直接给（_core.planet_view / node_inputs）：
同一进程里刚用 cpp 后端跑过 ①–④ 就直接用内存里的对象，否则从 npz 读回成 C++ 对象（skyisle_gen/engine.py 的 part）；
P6d：⑨ 的人口与邦都也由 C++ 从 ⑨ 的产物对象给（_core.node_polity：同进程跑过 cpp 的 ⑨ 就用内存里的，否则从 polity.npz 读回）；
旧 run 缺字段 / 没有 ⑨ 时退回按 npz 拼 dict 的 P6b 路径（值相同）。
"""
from __future__ import annotations

import math
import time

import numpy as np

AGE_ZH = {"young": "新岛", "mid": "中年", "old": "老岛"}
KEY_ASCII = {"汇聚": "convergent", "离散": "divergent", "走滑": "transform"}   # [island.resources] ore_gain 的键
_PLANET_CACHE: dict = {}
_PLANET_OBJ: dict = {}


def backend(ctx) -> str:
    from ..engine import backend as _backend
    return _backend(ctx.cfg)


def threads(ctx) -> int:
    return max(1, int((ctx.cfg.get("engine") or {}).get("threads", 4)))


def core():
    from ..engine import core as _core
    return _core()


def flat_config(c: dict) -> dict:
    """[island] 段展平成 {"num": {"layout.n0": 30.0, …}, "vec": {"territory.stretch": [1.0, 1.6, 2.4], …}}（布尔 → 0 / 1，字符串跳过）。
    中文键（resources.ore_gain 的板块边界类型）换成 ASCII：C++ 只用 ASCII 键。"""
    num, vec = {}, {}

    def walk(d, prefix):
        for k, v in d.items():
            key = f"{prefix}{KEY_ASCII.get(k, k)}"
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
    cg = ctx.load_npz(4, "climate_grid")
    bl = ctx.load_npz(4, "band_local")
    c4 = ctx.cfg["s04"]["climate"]
    keys = [str(k) for k in bl["keys"]]
    from .climate import calendar
    cal2 = calendar(planet)

    def f64(a):
        return np.ascontiguousarray(a, dtype=np.float64)
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
        # 5.4 四季：④ 的网格（grid_interp 同口径）、局部带界（local_edges 按 float64 插值）、倾角、热惯性常数、历法
        "climate": {**_grid_spec(cg["lats"], cg["lons"]), "precip": f64(cg["precip"]), "storm": f64(cg["storm"]), "window": f64(cg["window"]),
                    "continentality": f64(cg["continentality"]) if "continentality" in cg else None,
                    "band_lons": f64(bl["lons"]), "band_eq_n": f64(bl["edges"][keys.index("eq_n")]), "band_eq_s": f64(bl["edges"][keys.index("eq_s")]),
                    "tilt_deg": float(planet["axial_tilt_deg"]), "tau_land": float(c4["season_tau_land_days"]),
                    "tau_ocean": float(c4["season_tau_ocean_days"]), "alt_cont": float(c4.get("season_alt_continentality", 0.0)),
                    "calendar": {k: cal2[k] for k in ("seasons", "months_per_season", "days_per_month", "days_per_season", "year_days",
                                                      "day_offset_solstice_n")}},
    }
    _PLANET_CACHE.clear()
    _PLANET_CACHE[key] = pv
    return pv


def _run_key(ctx) -> tuple:
    from ..engine import _stage_key
    return (str(ctx.out_dir.resolve()),) + tuple(_stage_key(ctx, k) for k in (1, 2, 3, 4))


def _parts(ctx):
    """(① PlanetParams, ③ Islands, ④ Climate, 行星层配置对象) 或 None（扩展太旧 / 旧 run 缺字段）。按 run 与 ①–④ 的 key 缓存。"""
    if not hasattr(core(), "planet_view"):
        return None
    key = ("parts",) + _run_key(ctx)
    o = _PLANET_OBJ.get(key, False)
    if o is False:
        from .. import engine as E
        try:
            P, I, C = E.planet_parts(ctx)
            o = (P, I, C, core().make_config(E.planet_config(ctx.cfg)))
        except (KeyError, FileNotFoundError, ValueError, TypeError):
            o = None
        _PLANET_OBJ[key] = o
    return o


def planet_obj(ctx):
    """第三层的 PlanetView（C++ 对象，按 run 与 ①–④ 的 key 缓存）：P6c 起由 C++ 从 ①③④ 的产物对象直接给（_core.planet_view）；
    扩展太旧或旧 run 缺字段时退回按 npz 拼 dict（planet_view）再转。各步每次调用不再把行星层网格重新拷进 C++（全量季型要调 8000 次）。"""
    if not hasattr(core(), "make_planet"):        # 更旧的扩展：各步也收 dict
        return planet_view(ctx)
    key = ("view",) + _run_key(ctx)
    o = _PLANET_OBJ.get(key)
    if o is None:
        parts = _parts(ctx)
        if parts is not None:
            P, I, C, pc = parts
            o = core().planet_view(P, I, C, pc)
        else:
            o = core().make_planet(planet_view(ctx))
        for k in [k for k in _PLANET_OBJ if k[0] == "view"]:
            del _PLANET_OBJ[k]
        _PLANET_OBJ[key] = o
    return o


def _polity_part(ctx):
    """⑨ 的 C++ 对象（Polity）或 None（扩展太旧 / 这个 run 没有 ⑨）。按 run 与 ⑨ 的 key 缓存；同进程跑过 cpp 的 ⑨ 就是内存里那个。"""
    if not hasattr(core(), "node_polity"):
        return None
    from .. import engine as E
    key = ("polity",) + _run_key(ctx) + (E._stage_key(ctx, 9),)
    o = _PLANET_OBJ.get(key, False)
    if o is False:
        try:
            o = E.part(ctx, 9)
        except (KeyError, FileNotFoundError, ValueError, TypeError):
            o = None
        for k in [k for k in _PLANET_OBJ if k[0] == "polity"]:
            del _PLANET_OBJ[k]
        _PLANET_OBJ[key] = o
    return o


def inputs(ctx, node: int, inp: dict, full: bool = False) -> dict:
    """本群的 NodeInputs（dict）：P6c 起由 C++ 从 ③④ 的产物对象直接给（_core.node_inputs，与 _node_inputs 同值）；
    退回路径按 Python 的 inp 拼。full：再加 ⑨ 的人口与邦都（P6d 起由 C++ 从 ⑨ 的对象给：_core.node_polity，与 polity.npz 的读法同值）。"""
    parts = _parts(ctx)
    if parts is not None:
        _P, I, C, pc = parts
        d = core().node_inputs(I, C, int(node), int(ctx.seed), pc)
    else:
        d = inputs_py(ctx, node, inp)
    if full:
        # 聚落：人口只读 ⑨（没有 ⑨ 给 None，C++ 按可耕地 × 人口密度）；本群是不是某邦的都（settle._polity_role）
        pol = _polity_part(ctx) if parts is not None else None
        if pol is not None:
            d.update(core().node_polity(pol, int(node), parts[3]))
        else:
            from .settle import _polity_role
            d["pop"] = _polity_pop(ctx, node)
            d["people_per_arable_km2"] = float(ctx.cfg["shared"]["scale"]["people_per_arable_km2"])
            d["capital"] = _polity_role(ctx, node)
    return d


def inputs_polity_py(ctx, node: int) -> dict:
    """⑨ 的人口与邦都的 Python 读法（P6b 的路径；测试拿它与 _core.node_polity 对照）。"""
    from .settle import _polity_role
    return {"pop": _polity_pop(ctx, node), "people_per_arable_km2": float(ctx.cfg["shared"]["scale"]["people_per_arable_km2"]),
            "capital": _polity_role(ctx, node)}


def inputs_py(ctx, node: int, inp: dict) -> dict:
    """P6b 的拼法：从 Python 版 _node_inputs 的 inp 拼（退回路径；测试拿它与 C++ 的 node_inputs 对照）。"""
    d = {"node": int(node), "seed": int(ctx.seed), "lat": inp["lat"], "lon": inp["lon"], "area_km2": inp["area_km2"],
         "main_area_km2": inp["main_area_km2"], "height_m": inp["height_m"], "age": inp["age"], "layered": bool(inp["layered"]),
         "keel_clearance_m": inp["keel_clearance_m"], "area_median_km2": inp["area_median_km2"],
         "precip": inp["precip"], "temp_sea": inp["temp_sea"], "lapse_c_per_km": inp["lapse_c_per_km"],
         "arable_frac": inp["arable_frac"], "river_size": inp["river_size"], "has_river": bool(inp["has_river"])}
    for k in ("temp", "storm", "window", "season_range", "season_range_sea", "temp_winter", "temp_summer"):
        if k in inp:
            d[k] = float(inp[k])
    return d


def _polity_pop(ctx, node: int):
    p = ctx.stage_dir(9) / "polity.npz"
    if p.exists():
        with np.load(p) as z:
            if "pop" in z.files and node < z["pop"].size:
                return float(z["pop"][node])
    return None


# ---------------------------------------------------------------- 第 1 步：布局 + 岛形 + 高程
def build_terrain_cpp(ctx, node: int, c: dict, inp: dict, res_m: float | None = None, log=print) -> dict:
    R = core().build_terrain(inputs(ctx, node, inp), planet_obj(ctx), flat_config(c), float(res_m or 0.0), threads(ctx))
    return _terrain_from(ctx, node, inp, R, log)


def _terrain_from(ctx, node: int, inp: dict, R: dict, log=print) -> dict:
    from ..stages.s03_islands import CLASS_NAMES, CLASS_ZH
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
    inp = g["inp"]
    J = g["json"]
    ox, oy = J["raster"]["origin_km"]
    state = {"inp": inputs(ctx, node, inp), "height": np.ascontiguousarray(g["height"], dtype=np.float64),
             "island_id": np.ascontiguousarray(g["island_id"], dtype=np.int16), "cliff": np.ascontiguousarray(g["cliff"], dtype=bool),
             "res_km": float(g["res_km"]), "origin_x": float(ox), "origin_y": float(oy),
             "islands": [{"rim_m": float(i["rim_m"]), "keel_m": float(i["keel_m"]), "age": float(i["age"])} for i in J["islands"]]}
    R = core().build_hydro(state, planet_obj(ctx), flat_config(c), threads(ctx))
    _hydro_from(c, g, R, log)


def _hydro_from(c: dict, g: dict, R: dict, log=print) -> None:
    from .output import LANDCOVER_CLASSES, LANDCOVER_PALETTE
    hc = c["hydro"]
    inp = g["inp"]
    res_km = g["res_km"]
    cell_km2 = res_km * res_km
    J = g["json"]
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


# ---------------------------------------------------------------- P6b：整群 generate（地形 → 水系 → 资源 → 四季 → 天气 → 聚落）
def generate_cpp(ctx, node: int, c: dict, inp: dict, year: int = 0, res_m: float | None = None, steps: int = 9, log=print) -> dict:
    """整群在 C++ 里算完，拼回与 Python 版同形的 g（写产物照旧由 island.generate 做）。"""
    from . import decode
    from .climate import set_climate
    from .resources import resource_summary
    from .settle import set_settlements
    from .weather import set_weather
    t0 = time.perf_counter()
    R = core().generate(inputs(ctx, node, inp, full=steps >= 5), planet_obj(ctx), flat_config(c), int(year), int(steps),
                        float(res_m or 0.0), threads(ctx))
    t_core = time.perf_counter() - t0
    secs = R["seconds"]
    g = _terrain_from(ctx, node, inp, R["terrain"], log)
    g["timing"] = {"terrain": float(secs[0]), "core": t_core}
    J = g["json"]
    if steps >= 2:
        _hydro_from(c, g, R["hydro"], log)
        g["timing"]["hydro"] = float(secs[1])
        g["resources"] = decode.resources(g, node, c, R["resources"])
    if steps >= 3:
        set_climate(g, decode.climate(R["climate"], inp["planet"]), float(inp["season_range"]))
        D = R["daily"]
        g["daily"] = {"day": D["day"].astype(np.int64), "temp_c": D["temp_c"], "season": D["season"].astype(np.int64),
                      "precip_rel": D["precip_rel"], "precip_mm": D["precip_mm"], "storm": D["storm"], "window": D["window"],
                      "wind_u": D["wind_u"], "wind_v": D["wind_v"]}
        C = g["climate"]
        log(f"  气候（C++）：{C['season_type_zh']}（{'/'.join(C['season_names'])}）温 {[s['temp_c'] for s in C['seasons']]} "
            f"雨 {[int(s['precip_mm']) for s in C['seasons']]} mm")
    if steps >= 4:
        Y = R["weather"]
        y = {"day": Y["day"].astype(np.int64), "season": Y["season"].astype(np.int64), "month": Y["month"].astype(np.int64),
             "day_of_month": Y["day_of_month"].astype(np.int64), "type": Y["type"], "precip_mm": Y["precip_mm"], "temp_c": Y["temp_c"],
             "wind_from_deg": Y["wind_from_deg"], "wind_ms": Y["wind_ms"], "sailable": Y["sailable"], "storm_event": Y["storm_event"],
             "wet": Y["wet"], "temp_rim_c": Y["temp_rim_c"], "snow": Y["snow"]}
        set_weather(g, y, [dict(p) for p in Y["params"]], year, log=log)
    if steps >= 5:
        S = decode.settlements(R["settle"], g.get("climate"))
        g["settle_pop"] = float(R["settle_pop"])
        g["settle_raster"] = R["settle_raster"]
        g["settle_fields"] = R["settle_fields"]
        set_settlements(g, S)
        if "to_grass_km2" in S["clearing"]:
            J["landcover"]["note_clearing"] = "林地在村 / 镇 / 专业聚落半径内已开垦：内圈草坡（牧场草场）、外圈灌丛（薪炭林）"
        log(f"  聚落（C++）：人口 {S['population']:.0f} → {S['households']} 户；田块 {S['n_fields']}，村 {S['n_villages']}，散户 {S['n_hamlets']}，"
            f"镇 {len(S['towns'])}，专业聚落 {len(S['specials'])}，泊场 {len(S['landings'])}")
    if steps >= 2:
        resource_summary(g)
        Rr = g["resources"]
        log(f"  资源（C++）：点与片 {len(Rr['deposits'])} 处，赋存区 {len(Rr['occurrences'])}，采场 {len(Rr['workings'])}")
    g["timing"]["settle"] = float(secs[4])
    return g


def weather_years_cpp(ctx, node: int, c: dict, g: dict, years: int):
    """IS-daily 的多年逐日模拟（weather.multi_year_stats 的 cpp 分支）：返回 P[年, 季]（季降水和）与 F[年, 季]（季雨日比例）。"""
    P, F, _frd = core().weather_years(inputs(ctx, node, g["inp"]), planet_obj(ctx), flat_config(c), float(g["json"]["islands"][0]["rim_m"]),
                                      int(years))
    return P, F


def block_reduce_cpp(g: dict, f: int) -> dict:
    """粗版的块降采样（lod._block_reduce 的 cpp 分支）。"""
    return core().block_reduce(np.ascontiguousarray(g["island_id"], dtype=np.int16), np.ascontiguousarray(g["height"], dtype=np.float64),
                               np.ascontiguousarray(g["landcover"], dtype=np.uint8), np.ascontiguousarray(g["river"], dtype=np.uint8),
                               np.ascontiguousarray(g["lake"], dtype=bool), int(f))


def climate_only_cpp(ctx, node_inp: dict, c: dict, cfg_obj=None) -> dict:
    """只算四季（climate.classify_all 的 cpp 分支）。node_inp：classify_all 拼的 inp（lat / lon / height_m / 气候标量 / planet / keel）；
    cfg_obj：core().make_config(flat_config(c)) 预先转好的（逐群调用时省掉每次展平）。"""
    from . import decode
    d = {"node": 0, "seed": int(ctx.seed), "area_km2": 0.0, "main_area_km2": 0.0, "age": 0.0, "layered": False, "area_median_km2": 1.0,
         "lapse_c_per_km": 6.0, "arable_frac": 0.0, "river_size": 0.0, "has_river": False}
    for k in ("lat", "lon", "height_m", "keel_clearance_m", "precip", "temp", "temp_sea", "storm", "window", "season_range", "season_range_sea",
              "temp_winter", "temp_summer"):
        d[k] = float(node_inp[k])
    return decode.climate(core().climate_only(d, planet_obj(ctx), cfg_obj if cfg_obj is not None else flat_config(c)), node_inp["planet"])
