"""第三层的 C++ 桥（docs/PLAN-CORE.md 第六节；行星计划 P6a–P6d）。

岛群生成器的算法全在 `skyisle_gen._core`（core/，`python core/build.py` 编；Python 参考版删于 2026-09-30，tag python-reference-final）。
这里把输入备好、把结果拼回 g：数组定 dtype，island.json 定键序与 round 位数（C++ 只给原始的双精度数与 ASCII 代码，中文由 decode.py 译回）；
写产物在 Python。入口：`island.generate`（整群）、`island.build_terrain` 与 `hydro.build_hydro`（粗版、浮高统计只跑前两步）、
`weather.multi_year_stats`（IS-daily 的逐年模拟）、`lod.build_lod` 的块降采样与天气、`climate.classify_all`（全量季型）。
行星层（PlanetView、本群的 NodeInputs）由 C++ 从 ①③④ 的产物对象直接给（_core.planet_view / node_inputs：同一进程里刚跑过 ①–④ 就用内存里的对象，
否则从 npz 读回，skyisle_gen/engine.py 的 part）；⑨ 的人口与邦都（_core.node_polity）、⑥ 的邻边（_core.node_routes）同样，没有 ⑨ / ⑥ 的 run 给 None。
"""
from __future__ import annotations

import math
import time

import numpy as np

AGE_ZH = {"young": "新岛", "mid": "中年", "old": "老岛"}
KEY_ASCII = {"汇聚": "convergent", "离散": "divergent", "走滑": "transform"}   # [island.resources] ore_gain 的键
_PLANET_OBJ: dict = {}


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


def _run_key(ctx) -> tuple:
    from ..engine import _stage_key
    return (str(ctx.out_dir.resolve()),) + tuple(_stage_key(ctx, k) for k in (1, 2, 3, 4))


def _parts(ctx):
    """(① PlanetParams, ③ Islands, ④ Climate, 行星层配置对象)，按 run 与 ①–④ 的 key 缓存。"""
    key = ("parts",) + _run_key(ctx)
    o = _PLANET_OBJ.get(key)
    if o is None:
        from .. import engine as E
        P, I, C = E.planet_parts(ctx)
        o = (P, I, C, core().make_config(E.planet_config(ctx.cfg)))
        _PLANET_OBJ[key] = o
    return o


def planet_obj(ctx):
    """第三层的 PlanetView（C++ 对象，由 C++ 从 ①③④ 的产物对象直接给：_core.planet_view；按 run 与 ①–④ 的 key 缓存，
    全量季型要调 8000 次，不每次把行星层网格重新拷进 C++）。"""
    key = ("view",) + _run_key(ctx)
    o = _PLANET_OBJ.get(key)
    if o is None:
        P, I, C, pc = _parts(ctx)
        o = core().planet_view(P, I, C, pc)
        for k in [k for k in _PLANET_OBJ if k[0] == "view"]:
            del _PLANET_OBJ[k]
        _PLANET_OBJ[key] = o
    return o


def _stage_part(ctx, idx: int, tag: str):
    """第 idx 步的 C++ 对象，这个 run 没有那一步的产物（只跑到 ④ 的小世界）给 None。按 run 与该步的 key 缓存。"""
    from .. import engine as E
    key = (tag,) + _run_key(ctx) + (E._stage_key(ctx, idx),)
    o = _PLANET_OBJ.get(key, False)
    if o is False:
        try:
            o = E.part(ctx, idx)
        except FileNotFoundError:
            o = None
        for k in [k for k in _PLANET_OBJ if k[0] == tag]:
            del _PLANET_OBJ[k]
        _PLANET_OBJ[key] = o
    return o


def inputs(ctx, node: int, inp: dict, full: bool = False) -> dict:
    """本群的 NodeInputs（dict，由 C++ 从 ③④ 的产物对象给：_core.node_inputs，与 _node_inputs 同值）。
    full：再加 ⑨ 的人口与邦都（_core.node_polity；没有 ⑨ 给 None，C++ 按可耕地 × 人口密度）与 ⑥ 的邻边（P7 的中转站：_core.node_routes；没有 ⑥ 给 None）。"""
    _P, I, C, pc = _parts(ctx)
    d = core().node_inputs(I, C, int(node), int(ctx.seed), pc)
    if full:
        pol = _stage_part(ctx, 9, "polity")
        if pol is not None:
            d.update(core().node_polity(pol, int(node), pc))
        else:
            d.update({"pop": None, "people_per_arable_km2": float(ctx.cfg["shared"]["scale"]["people_per_arable_km2"]), "capital": None})
        r6 = _stage_part(ctx, 6, "routes")
        d["routes"] = core().node_routes(I, r6, int(node)) if r6 is not None else None
    return d


# ---------------------------------------------------------------- 第 1 步：布局 + 岛形 + 高程
def build_terrain_cpp(ctx, node: int, c: dict, inp: dict, res_m: float | None = None, log=print) -> dict:
    R = core().build_terrain(inputs(ctx, node, inp), planet_obj(ctx), flat_config(c), float(res_m or 0.0), threads(ctx))
    return _terrain_from(ctx, node, inp, R, log, c)


def _terrain_from(ctx, node: int, inp: dict, R: dict, log=print, c: dict | None = None) -> dict:
    from ..stages.s03_islands import CLASS_NAMES, CLASS_ZH
    from .terrain import cores_json
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
            "float_m": round(float(e["float"]), 1),
            "age": round(float(e["age"]), 3), "age_zh": AGE_ZH[e["kind"]],
            "bbox_cells": [r0, c0, m, m],
        })
        if e.get("cores"):
            gcx, gcy = e["gc"]
            islands_json[-1]["cores"] = cores_json(e["cores"], (float(gcx), float(gcy)), float(e["float"]), res_km,
                                                   (c or {}).get("terrain", {}))
        if e.get("strat"):          # B2 岩层：层厚按最终高程（拟合的比例换算过）
            st = e["strat"]
            sc_ = float(st["scale"])
            islands_json[-1]["strat"] = {"cap_m": round(float(st["t_cap"]) * sc_, 1), "sediment_m": round(float(st["t_sed"]) * sc_, 1),
                                         "gabbro_m": round(float(st["t_gab"]) * sc_, 1), "bed_lime_m": round(float(st["bed_lime"]) * sc_, 1),
                                         "bed_marl_m": round(float(st["bed_marl"]) * sc_, 1), "crown_exhumed_m": round(float(st["exhume"]) * sc_, 1)}
    rims = np.asarray(R["rims"], dtype=np.float64)
    lk = []
    for e in R["links"]:
        d = {"a": int(e["a"]), "b": int(e["b"]), "gap_km": round(float(e["gap"]), 3), "kind": "ferry",
             "dh_m": round(float(e["dh"]), 1)}
        if e["fallback"]:
            d["fallback"] = True
        lk.append(d)
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
              "height_note": "height.png 16 位灰度 = 零点（浮层基准，planet.json 的 vertical）以上的高度 / height_png_scale_m_per_unit；0 = 虚空",
              "void_value": 0}
    T = R["territory"]
    terr = {"neighbours": int(T["neighbours"]), "gap_km": float(T["gap_km"]), "constrained": bool(T["constrained"])}
    if T["constrained"]:
        terr.update({"main_offset_km": [round(float(T["off_x"]), 3), round(float(T["off_y"]), 3)],
                     "main_turn_deg": round(float(T["turn_deg"]), 1), "main_stretch": float(T["stretch"]),
                     "before_km": round(float(T["before"]), 3), "after_km": round(float(T["after"]), 3)})
    if T["has_violation"]:
        terr["violation_km"] = round(float(T["violation"]), 3)
        terr["note"] = "势力范围（C++ 的 territory.cpp）：与每个邻群按等效半径分界、各退 gap/2；violation_km ≤ 0 = 没越界（负值是最小余量）"
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
         "layout": {"n_ferries": sum(1 for e in lk if e["kind"] == "ferry"),
                    "gap_median_km": round(float(np.median([e["gap_km"] for e in lk])), 2) if lk else None}}
    masks_pos = [(mk, int(r0), int(c0)) for mk, r0, c0 in R["masks_pos"]]
    g = {"height": height, "island_id": island_id, "cliff": cliff, "json": J, "rims": rims, "res_km": res_km,
         "masks_pos": masks_pos, "inp": inp}
    g["core_layout_raw"] = [{"gc": e["gc"], "cores": e["cores"]} for e in R["islands"]]
    if "coast_dist_m" in R:          # B4 亚格岸距
        g["coast_dist_m"] = R["coast_dist_m"]
    if "strat_top" in R:             # B2 层面：地形与水系分两次调时原样交回 build_hydro（谷坡角、lith 读它）
        g["strat_top"], g["skel_top"] = R["strat_top"], R["skel_top"]
        g["strat_raw"] = [e.get("strat") for e in R["islands"]]
    if "landforms" in R:             # B3 地形阶段的地貌（资源之后另有识别型的，generate_cpp 换成全的）
        from .landforms import LANDFORM_NOTE, landforms_json
        J["landforms"] = landforms_json(R["landforms"])
        J["landforms_note"] = LANDFORM_NOTE
    return g


# ---------------------------------------------------------------- 第 2 步前半：水系、河道、地表、可耕地
def build_hydro_cpp(ctx, node: int, c: dict, g: dict, log=print) -> None:
    inp = g["inp"]
    J = g["json"]
    ox, oy = J["raster"]["origin_km"]
    state = {"inp": inputs(ctx, node, inp), "height": np.ascontiguousarray(g["height"], dtype=np.float64),
             "island_id": np.ascontiguousarray(g["island_id"], dtype=np.int16), "cliff": np.ascontiguousarray(g["cliff"], dtype=bool),
             "res_km": float(g["res_km"]), "origin_x": float(ox), "origin_y": float(oy),
             "islands": [{"rim_m": float(i["rim_m"]), "keel_m": float(i["keel_m"]), "peak_m": float(i["peak_m"]), "age": float(i["age"]), "young": i["age_zh"] == "新岛",
                          "multicore": bool(i.get("cores"))} for i in J["islands"]]}
    if "strat_top" in g:
        state["strat_top"] = np.ascontiguousarray(g["strat_top"], dtype=np.float64)
        state["skel_top"] = np.ascontiguousarray(g["skel_top"], dtype=np.float64)
        for e, st in zip(state["islands"], g["strat_raw"]):
            e["strat"] = st
    if "core_layout_raw" in g:
        for e, raw in zip(state["islands"], g["core_layout_raw"]):
            e.update(raw)
    elif c.get("water", {}).get("core_footprint_scale", 0) > 0:
        raise ValueError("Core footprint requires original core positions and mountain loads; regenerate terrain")
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
        if e.get("cap_ran"):
            Ji["captures"] = int(e["captures"])
    dz = float(R["dz"])
    J0 = J["islands"][0]
    J0["rim_m"] = round(J0["rim_m"] + dz, 1)
    J0["peak_m"] = round(J0["peak_m"] + dz, 1)
    J0["cliff_m"] = round(J0["rim_m"] - J0["keel_m"], 1)
    for co in J0.get("cores", []):
        co["peak_m"] = round(co["peak_m"] + dz, 1)
    # 主岛河口表（hydro.carve_channels 的 info["rivers"]）
    rivers_info = []
    for e in R["rivers"]:
        rivers_info.append({"mouth_cell": [int(e["mouth"][0]), int(e["mouth"][1])], "basin_km2": round(float(e["basin_km2"]), 1),
                            "length_km": round(float(e["length_km"]), 1), "discharge_m3s": round(float(e["discharge"]), 2),
                            # 河道（平岸）口径；逐点的平岸与年均口径在 rivers.json 的 segments
                            "w_ch_m": round(float(e["width"]), 1), "d_ch_m": round(float(e["depth"]), 2),
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
    g.update({"w_ch_m": R["w_ch_m"], "d_ch_m": R["d_ch_m"], "floodplain": R["floodplain"], "cut_m": R["cut_m"]})
    g.update({"flowacc_km2": R["flowacc_km2"], "river": river, "stream": R["stream"], "lake": lake,
              "landcover": R["landcover"], "arable": R["arable"], "slope_deg": R["slope_deg"], "filled": R["filled"],
              "recv_i": R["recv_i"], "recv_j": R["recv_j"], "route_h": R["route_h"], "rain_mm": R["rain_mm"],
              "runoff_mm": R["runoff_mm"]})
    if "cultivable" in R:                  # P5：宜垦（arable 是额度内的上等地）
        g["cultivable"] = R["cultivable"]
    if "lith" in R:                        # B2：露出的岩性
        from .landforms import lith_summary
        g["lith"] = R["lith"]
        J["lith"] = lith_summary(R["lith"], island_id)
    # C1 / C2：河床、谷底宽、限制度、记成水的河道格；C4 / C5：凝结水、云雾林、地下水、崖壁泉线
    for k in ("bed_m", "floor_w_m", "confine", "river_water", "condense_mm", "core_water_sources", "cloud_forest", "bfi", "recharge_mm", "recharge_acc", "runoff_acc",
              "wt", "wt_outlet"):
        if k in R:
            g[k] = R[k]
    if "springline" in R:
        g["springline"] = [dict(s) for s in R["springline"]]
    if "year_s" in R:
        g["year_s"] = float(R["year_s"])
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
                  "runoff_ratio": round(float(R["runoff_ratio"]), 3), "runoff_mm": round(P_mm * float(R["runoff_ratio"]), 0),
                  "budyko_w": float(hc["budyko_w"]),
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
                  "channel_height": "surface",
                  "channel_note": "河宽 / 水深见 terrain.npz 的 w_ch_m / d_ch_m（**河道 = 平岸**口径：年均流量下的水面宽深 × (bf_ratio_channel)^站内指数；溪涧按 stream_width_mult 缩）；逐点按真实平岸流量的宽深在 rivers.json 的 segments 里（w_bf_m / d_bf_m），年均流量下的水面宽是 w_mean_m；height 在河道格是平岸水面（= 滩面），"
                                  "河床 = height − 水深（C1；channel_height = surface，旧产物没有这个键时 height 是河床）；河宽 ≥ 一格的河道格地表记成河，"
                                  "更窄的在岸上（河槽由 rivers.json 的中心线 + 宽 + 深表达）；rivers[].waterfall_m = 河口跌下崖缘的落差",
                  "wind_ms": [round(u, 2), round(v, 2)],
                  "river_levels": {"1": "小河", "2": "中河", "3": "大河", "stream": "季节性溪涧（water.png 值 1）"}}
    from .hydro import local_precip_summary
    J["hydro"]["local_precip"] = local_precip_summary(g["rain_mm"], island_id, hc)
    from .groundwater import valley_summary, water_summary
    J["hydro"]["valley"] = valley_summary(g, cell_km2)
    if "condense_mm" in R:
        J["hydro"]["water"] = water_summary(g, R, island_id, cell_km2)
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
    """整群在 C++ 里算完，拼回前端要的 g（与删掉的 Python 参考版同形；写产物由 island.generate 做）。"""
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
    g = _terrain_from(ctx, node, inp, R["terrain"], log, c)
    g["timing"] = {"terrain": float(secs[0]), "core": t_core}
    J = g["json"]
    if steps >= 2:
        _hydro_from(c, g, R["hydro"], log)
        g["timing"]["hydro"] = float(secs[1])
        g["resources"] = decode.resources(g, node, c, R["resources"])
        RR = R["resources"]
        if "rockwall_m" in RR:            # B3 崖层与全部地貌
            g["rockwall_m"] = RR["rockwall_m"]
            g["rockwall_dir"] = RR["rockwall_dir"]
        if "landforms" in RR:
            from .landforms import LANDFORM_NOTE, landforms_json
            J["landforms"] = landforms_json(RR["landforms"])
            J["landforms_note"] = LANDFORM_NOTE
    if steps >= 3:
        set_climate(g, decode.climate(R["climate"], inp["planet"]), float(inp["season_range"]))
        g["daily"] = _daily_from(R["daily"])
        C = g["climate"]
        log(f"  气候（C++）：{C['season_type_zh']}（{'/'.join(C['season_names'])}）温 {[s['temp_c'] for s in C['seasons']]} "
            f"雨 {[int(s['precip_mm']) for s in C['seasons']]} mm")
    if steps >= 4:
        set_weather(g, _weather_from(R["weather"]), [dict(p) for p in R["weather"]["params"]], year, log=log)
    if steps >= 5:
        S = decode.settlements(R["settle"], g.get("climate"), c.get("market"))
        g["settle_pop"] = float(R["settle_pop"])
        g["settle_raster"] = R["settle_raster"]
        g["settle_fields"] = R["settle_fields"]
        g["cultivated"] = R["settle_cultivated"]          # P5：已垦（在种）与撂荒年头
        g["fallow_years"] = R["settle_fallow"]
        g["polder_id"] = R["settle_polder"]                # P6：圩号
        g["landcover_natural"] = R["settle_landcover_natural"]    # P6b：没有人以前的地表、人工改造
        g["landuse"] = R["settle_landuse"]
        set_settlements(g, S)
        if "to_grass_km2" in S["clearing"]:
            J["landcover"]["note_clearing"] = "林地在村 / 镇 / 专业聚落半径内已开垦：内圈草坡（牧场草场）、外圈灌丛（薪炭林）"
        log(f"  聚落（C++）：人口 {S['population']:.0f} → {S['households']} 户；田块 {S['n_fields']}，村 {S['n_villages']}，散户 {S['n_hamlets']}，"
            f"镇 {len(S['towns'])}，专业聚落 {len(S['specials'])}，泊场 {len(S['landings'])}")
        MS = S["market"]
        log(f"  镇与航船（C++）：大泊场 {MS['n_harbors']}（{MS['harbor_ships']} 条船），镇 {MS['n_towns']}（挨大泊场 {MS['towns_with_harbor']}），邑治在岛 {MS['seat_island']}；"
            f"航船 {MS['n_lines']} 线 {MS['line_km']:.0f} km、送 {MS['boat_villages']} 村；中转站 {MS['n_relays']}（常住 {MS['relay_households']} 户）")
        if "waterworks" in S:
            ws = S["waterworks"]["summary"]
            bw = ws.get("big") or {"n": 0}
            log(f"  水利（C++）：大堰 {bw['n']}" + (f"（灌区 {bw['planned_km2']:.0f} km²、在种 {bw['served_km2']:.0f}（已垦的 {bw['share']:.0%}），渠 {bw['canal_km']:.0f} km、"
                f"用水的村 {bw['villages']}）" if bw["n"] else "") + (f"；村的渠首 {ws['n_heads']}，渠 {ws['canal_km']:.0f} km、灌田 {ws['commanded_km2'] - bw.get('served_km2', 0.0):.0f} km²；"
                if ws.get("village_works", True) else "；村级的渠、塘不在岛群层出（归营建器）；") +
                f"塘 {ws['n_ponds']}，闸 {ws['n_sluices']}；圩田 {ws['polder_km2']:.1f} km²（湿地 {ws['wetland_km2']:.1f} 的 {ws['polder_share']:.0%}，{ws['n_polders']} 圩）")
    if steps >= 2:
        resource_summary(g)
        Rr = g["resources"]
        log(f"  资源（C++）：点与片 {len(Rr['deposits'])} 处，赋存区 {len(Rr['occurrences'])}，采场 {len(Rr['workings'])}")
    if "rivernet" in R:                   # C3：河的数据（天气之后算）
        from .rivernet import river_data_summary
        g["rivernet"] = R["rivernet"]
        J["hydro"]["river_data"] = rd = river_data_summary(g)
        log(f"  河的数据（C++）：{rd['segments']} 段，河 {rd['length_km']['河']:.0f} km、溪涧 {rd['length_km']['溪涧']:.0f} km；"
            f"瀑布 {rd['falls']}；流域 {rd['basins']}，平岸 / 年均 {rd.get('bf_ratio_range', '—')}")
    g["timing"]["settle"] = float(secs[4])
    return g


def _daily_from(D: dict) -> dict:
    """C++ 的逐日曲线 → g["daily"]（day / temp_c / season / 降水 / 风暴 / 窗口 / 风的逐日数组；总览图没有天气时画它）。"""
    return {"day": D["day"].astype(np.int64), "temp_c": D["temp_c"], "season": D["season"].astype(np.int64),
            "precip_rel": D["precip_rel"], "precip_mm": D["precip_mm"], "storm": D["storm"], "window": D["window"],
            "wind_u": D["wind_u"], "wind_v": D["wind_v"]}


def _weather_from(Y: dict) -> dict:
    """C++ 的一年逐日天气 → weather.set_weather 要的逐日数组 dict。"""
    return {"day": Y["day"].astype(np.int64), "season": Y["season"].astype(np.int64), "month": Y["month"].astype(np.int64),
            "day_of_month": Y["day_of_month"].astype(np.int64), "type": Y["type"], "precip_mm": Y["precip_mm"], "temp_c": Y["temp_c"],
            "wind_from_deg": Y["wind_from_deg"], "wind_ms": Y["wind_ms"], "sailable": Y["sailable"], "storm_event": Y["storm_event"],
            "wet": Y["wet"], "temp_rim_c": Y["temp_rim_c"], "snow": Y["snow"]}


def weather_year_cpp(ctx, node: int, c: dict, g: dict, year: int = 0, log=print) -> None:
    """地形 + 水系之后接着算四季 → 逐日曲线 → 一年天气（粗版 `island lod --weather` 用；与 generate 的第 3、4 步同式、同随机流）。
    岸缘取水系之后主岛的 rim_m（island.json 的值）。拼回 g["climate"] / g["daily"] / g["weather"]，与 generate 同形。"""
    from . import decode
    from .climate import set_climate
    from .weather import set_weather
    inp = g["inp"]
    R = core().weather_year(inputs(ctx, node, inp), planet_obj(ctx), flat_config(c), float(g["json"]["islands"][0]["rim_m"]), int(year))
    set_climate(g, decode.climate(R["climate"], inp["planet"]), float(inp["season_range"]))
    g["daily"] = _daily_from(R["daily"])
    set_weather(g, _weather_from(R["weather"]), [dict(p) for p in R["weather"]["params"]], year, log=log)


def weather_years_cpp(ctx, node: int, c: dict, g: dict, years: int):
    """IS-daily 的多年逐日模拟（weather.multi_year_stats 调）：返回 P[年, 季]（季降水和）与 F[年, 季]（季雨日比例）。"""
    P, F, _frd = core().weather_years(inputs(ctx, node, g["inp"]), planet_obj(ctx), flat_config(c), float(g["json"]["islands"][0]["rim_m"]),
                                      int(years))
    return P, F


def block_reduce_cpp(g: dict, f: int) -> dict:
    """粗版的块降采样（lod.build_lod 调；块怎么归并见 lod 的模块说明）。"""
    return core().block_reduce(np.ascontiguousarray(g["island_id"], dtype=np.int16), np.ascontiguousarray(g["height"], dtype=np.float64),
                               np.ascontiguousarray(g["landcover"], dtype=np.uint8), np.ascontiguousarray(g["river"], dtype=np.uint8),
                               np.ascontiguousarray(g["lake"], dtype=bool), int(f))


def climate_only_cpp(ctx, node_inp: dict, c: dict, cfg_obj=None) -> dict:
    """只算四季（climate.classify_all 调）。node_inp：classify_all 拼的 inp（lat / lon / height_m / 气候标量 / planet / keel）；
    cfg_obj：core().make_config(flat_config(c)) 预先转好的（逐群调用时省掉每次展平）。"""
    from . import decode
    d = {"node": 0, "seed": int(ctx.seed), "area_km2": 0.0, "main_area_km2": 0.0, "age": 0.0, "layered": False, "area_median_km2": 1.0,
         "lapse_c_per_km": 6.0, "arable_frac": 0.0, "river_size": 0.0, "has_river": False,
         "precip_mm_ref": float(ctx.cfg["s04"]["climate"].get("precip_mm_ref", 4000.0))}
    for k in ("lat", "lon", "height_m", "keel_clearance_m", "precip", "temp", "temp_sea", "storm", "window", "season_range", "season_range_sea",
              "temp_winter", "temp_summer"):
        d[k] = float(node_inp[k])
    if node_inp.get("precip_share") is not None:
        d["precip_share"] = [float(v) for v in node_inp["precip_share"]]
    return decode.climate(core().climate_only(d, planet_obj(ctx), cfg_obj if cfg_obj is not None else flat_config(c)), node_inp["planet"])
