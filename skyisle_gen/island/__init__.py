"""岛群生成器（第三层，docs/PLAN-ISLAND.md）：给一个节点号，按行星产物做约束，生成群内布局、岛内地形、水系、四季气候与逐日天气。

按需生成、不进十步管线、不回灌：stages/ 不得 import 本包（tests 有静态断言）。
随机数只从 rng.entity_rng(seed, ISLAND_STREAM, f"island:{node}:{部件}") 取；天气另加年份键。
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..config import CONFIG_DIR, _deep_merge, apply_sets
from ..rng import entity_rng
from ..stages.s03_islands import CLASS_NAMES, CLASS_ZH

ISLAND_STREAM = 21   # 与十步管线的流号 1–10 错开


def island_config(ctx, sets: list[str] | None = None) -> dict:
    """[island] 段：默认值 ← run 的 config.resolved.toml（若有）← --set。不进管线缓存 key。"""
    import tomllib
    with open(CONFIG_DIR / "default.toml", "rb") as fh:
        base = tomllib.load(fh).get("island", {})
    cfg = _deep_merge(base, ctx.cfg.get("island", {}))
    if sets:
        tmp = {"island": cfg}
        apply_sets(tmp, [s for s in sets if s.startswith("island.")])
        cfg = tmp["island"]
    ctx.cfg["island"] = cfg
    return cfg


def _rng(ctx, node: int, part: str):
    return entity_rng(ctx.seed, ISLAND_STREAM, f"island:{node}:{part}")


def _node_inputs(ctx, node: int) -> dict:
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    planet = ctx.load_json(1, "planet")
    n = isl["lat"].size
    if not (0 <= node < n):
        raise ValueError(f"节点号 {node} 超出范围 [0, {n})")
    d = {k: float(isl[k][node]) for k in ("lat", "lon", "area_km2", "main_area_km2", "height_m", "wall_m", "arable_frac", "age")}
    d["plate"] = int(isl["plate"][node])
    d["cls"] = int(isl["cls"][node])
    d["layered"] = bool(isl["layered"][node])
    for k in ("precip", "temp", "storm", "stability", "window", "river_size", "temp_sea", "season_range", "season_range_sea",
              "temp_winter", "temp_summer"):
        d[k] = float(cli[k][node])
    d["has_river"] = bool(cli["has_river"][node])
    d["area_median_km2"] = float(np.median(isl["area_km2"]))
    d["planet"] = planet
    d["keel_clearance_m"] = float(ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    d["lapse_c_per_km"] = float(ctx.cfg["s04"]["climate"]["lapse_c_per_km"])
    return d


def build_terrain(ctx, node: int, c: dict, inp: dict, res_m: float | None = None, log=print) -> dict:
    """第 1 步：布局 + 岛形 + 高程。返回群栅格字典 g（height / island_id / cliff / json / islands 列表）。"""
    from .layout import (boundary_axis, island_count, links, peak_heights, place_islands, radial_profile,
                         shoreline_gaps, zipf_sizes)
    from .terrain import age_class, island_shape, sculpt_island
    from .grid import binary_erode

    lay, ter = c["layout"], c["terrain"]
    plates = ctx.load_npz(3, "plates")
    axis, kernel, btype = boundary_axis(plates, inp["lat"], inp["lon"])
    rng_l = _rng(ctx, node, "layout")
    n = island_count(rng_l, inp["area_km2"], inp["area_median_km2"], lay)
    sizes = zipf_sizes(inp["area_km2"], inp["main_area_km2"], n, lay)
    n = sizes.size
    peaks = peak_heights(rng_l, n, inp["height_m"], inp["layered"], lay)
    ages = np.clip(inp["age"] + rng_l.normal(0.0, float(lay["age_jitter"]), n), 0.0, 1.0)
    ages[0] = inp["age"]
    elong = rng_l.uniform(1.0, float(ter["elongation_max"]), n)
    thetas = axis + rng_l.normal(0.0, 0.35 if kernel > 0.3 else 1.2, n)
    keel = inp["keel_clearance_m"]

    res0 = float(res_m or c["res_m"])
    grid_max = int(c["grid_max"])
    # 先用圆形剖面粗放一遍估算群外框，选定分辨率（超过 grid_max 就加倍），再按该分辨率生成岛形与正式布局
    r_eff = 1.2 * np.sqrt(sizes / math.pi)
    pre_prof = [np.full(72, r) for r in r_eff]
    pre_c, _ = place_islands(_rng(ctx, node, "place"), pre_prof, sizes, axis, kernel, btype, lay)
    ext_km = max(float((pre_c[:, 0] + r_eff).max() - (pre_c[:, 0] - r_eff).min()),
                 float((pre_c[:, 1] + r_eff).max() - (pre_c[:, 1] - r_eff).min())) + 2.0 * float(c["margin_km"])
    res_km = res0 / 1000.0
    while ext_km / res_km > 0.95 * grid_max:
        res_km *= 2.0
    if res_km * 1000.0 > res0:
        log(f"  群外框约 {ext_km:.0f} km，超过 {grid_max} 格上限，分辨率取 {res_km * 1000:.0f} m")
    shapes = []
    profiles, offsets = [], []
    for k in range(n):
        rng_s = _rng(ctx, node, f"shape:{k}")
        mask, inside, X, Y = island_shape(rng_s, float(sizes[k]), res_km, float(elong[k]), float(thetas[k]), ter)
        shapes.append((mask, inside, X, Y))
        ctr, prof = radial_profile(mask, res_km)
        profiles.append(prof)
        offsets.append(ctr)
    centers, pstats = place_islands(_rng(ctx, node, "place"), profiles, sizes, axis, kernel, btype, lay)
    halves = np.array([s[0].shape[0] * res_km / 2.0 for s in shapes])
    gc = centers - np.array(offsets)          # 各岛局部栅格中心（岛心 = 质心 + 偏移）
    xmin, xmax = float((gc[:, 0] - halves).min()), float((gc[:, 0] + halves).max())
    ymin, ymax = float((gc[:, 1] - halves).min()), float((gc[:, 1] + halves).max())
    res_m_eff = res_km * 1000.0

    # 贴图
    x0 = math.floor(xmin / res_km) * res_km
    y0 = math.ceil(ymax / res_km) * res_km
    W = int(math.ceil((xmax - x0) / res_km)) + 1
    H = int(math.ceil((y0 - ymin) / res_km)) + 1
    height = np.full((H, W), np.nan, dtype=np.float64)
    island_id = np.full((H, W), -1, dtype=np.int16)
    islands_json = []
    rims = np.zeros(n)
    masks_pos = []
    t0 = time.perf_counter()
    for k in range(n):
        mask, inside, X, Y = shapes[k]
        m = mask.shape[0]
        # 局部栅格中心 gc[k] 落到群栅格：左上角格
        c0 = int(round((gc[k, 0] - (m - 1) / 2.0 * res_km - x0) / res_km))
        r0 = int(round((y0 - (gc[k, 1] + (m - 1) / 2.0 * res_km)) / res_km))
        kind = age_class(float(ages[k]), ter)
        peak = float(peaks[k])
        keel_k = keel if peak > 2.0 * keel else 0.5 * peak
        rf_lo, rf_hi = {"young": (0.05, 0.15), "mid": (0.10, 0.30), "old": (0.30, 0.50)}[kind]
        rng_t = _rng(ctx, node, f"terrain:{k}")
        rim = keel_k + rng_t.uniform(rf_lo, rf_hi) * (peak - keel_k)
        rims[k] = rim
        h, kind = sculpt_island(rng_t, mask, inside, X, Y, float(ages[k]), float(sizes[k]), res_km, peak, rim, k == 0, ter)
        # 写入（不覆盖已有岛：布局保证不重叠）
        sl = (slice(r0, r0 + m), slice(c0, c0 + m))
        tgt = height[sl]
        put = mask & np.isnan(tgt)
        tgt[put] = h[put]
        island_id[sl][put] = k
        masks_pos.append((mask & put, r0, c0))
        ii, jj = np.where(island_id == k)
        islands_json.append({
            "id": k, "is_main": k == 0, "area_km2": round(float(put.sum()) * res_km * res_km, 3),
            "area_target_km2": round(float(sizes[k]), 3),
            "center_km": [round(float(centers[k, 0]), 3), round(float(centers[k, 1]), 3)],
            "peak_m": round(peak, 1), "rim_m": round(rim, 1), "keel_m": round(keel_k, 1), "cliff_m": round(rim - keel_k, 1),
            "age": round(float(ages[k]), 3), "age_zh": {"young": "新岛", "mid": "中年", "old": "老岛"}[kind],
            "bbox_cells": [r0, c0, m, m],
        })
    # 裁到陆地外框 + 边距（局部栅格有很大的空白外框）
    land = island_id >= 0
    rows, cols = np.where(land.any(axis=1))[0], np.where(land.any(axis=0))[0]
    mg = int(math.ceil(float(c["margin_km"]) / res_km))
    r_lo, r_hi = int(max(0, rows[0] - mg)), int(min(H, rows[-1] + mg + 1))
    c_lo, c_hi = int(max(0, cols[0] - mg)), int(min(W, cols[-1] + mg + 1))
    height = height[r_lo:r_hi, c_lo:c_hi]
    island_id = island_id[r_lo:r_hi, c_lo:c_hi]
    land = island_id >= 0
    H, W = height.shape
    x0 += c_lo * res_km
    y0 -= r_lo * res_km
    masks_pos = [(m, r0 - r_lo, c0 - c_lo) for m, r0, c0 in masks_pos]
    for i in islands_json:
        i["bbox_cells"][0] -= r_lo
        i["bbox_cells"][1] -= c_lo
    # 岸线间距 → 索桥 / 短渡 / 导水槽
    ctr_cells = np.array([[(i["center_km"][0] - x0) / res_km, -(i["center_km"][1] - y0) / res_km] for i in islands_json])
    radii = np.array([p.max() for p in profiles])
    gaps = shoreline_gaps(masks_pos, res_km, ctr_cells, radii, float(lay["ferry_max_km"]))
    lk, tree = links(gaps, rims, n, lay)
    cliff = land & ~binary_erode(land, int(ter["cliff_cells"]))
    log(f"  地形 {n} 岛 {H}×{W} @ {res_m_eff:.0f} m，{time.perf_counter() - t0:.1f} s")

    meta = {"node": node, "seed": ctx.seed, "run": ctx.out_dir.name, "lat": inp["lat"], "lon": inp["lon"],
            "cls": CLASS_NAMES[inp["cls"]], "cls_zh": CLASS_ZH[CLASS_NAMES[inp["cls"]]],
            "age": inp["age"], "age_zh": {"young": "新岛", "mid": "中年", "old": "老岛"}[age_class(inp["age"], ter)],
            "layered": inp["layered"], "plate": inp["plate"], "boundary_type": ["汇聚", "离散", "走滑"][btype],
            "boundary_kernel": round(kernel, 3), "boundary_axis_deg": round(math.degrees(axis), 1), "res_m": res_m_eff,
            "generator": "skyisle_gen.island", "layer": "第三层（按需生成，不回灌）"}
    km_per_deg = 2 * math.pi * float(inp["planet"]["radius_km"]) / 360.0
    raster = {"res_m": res_m_eff, "rows": H, "cols": W, "origin_km": [round(x0, 3), round(y0, 3)],
              "origin_note": "origin_km = 栅格左上角相对群心（节点经纬度）的平面坐标（km，x 东 y 北）；行 r 列 c 的格心 = origin + ((c+0.5)·res, −(r+0.5)·res)",
              "km_per_deg_lat": round(km_per_deg, 3), "km_per_deg_lon": round(km_per_deg * math.cos(math.radians(inp["lat"])), 3),
              "height_note": "height.png 16 位灰度 = 云带顶以上高度 / height_png_scale_m_per_unit；0 = 虚空",
              "void_value": 0}
    constraints = {
        "area_km2": {"target": round(inp["area_km2"], 2), "actual": round(float(land.sum()) * res_km * res_km, 2)},
        "main_area_km2": {"target": round(inp["main_area_km2"], 2), "actual": islands_json[0]["area_km2"]},
        "height_m": {"target": round(inp["height_m"], 1), "actual": round(float(np.nanmax(height[island_id == 0])), 1)},
        "arable_frac": {"target": round(inp["arable_frac"], 4)},
        "has_river": {"target": inp["has_river"]},
        "n_islands": n,
    }
    J = {"meta": meta, "raster": raster, "constraints": constraints, "islands": islands_json, "links": lk,
         "channels": [[int(a), int(b)] for a, b in tree],
         "layout": {"n_bridges": sum(1 for e in lk if e["kind"] == "bridge"), "n_ferries": sum(1 for e in lk if e["kind"] == "ferry"),
                    "gap_median_km": round(float(np.median([e["gap_km"] for e in lk])), 2) if lk else None,
                    "channel_islands": len(tree) + 1}}
    return {"height": height, "island_id": island_id, "cliff": cliff, "json": J, "rims": rims, "res_km": res_km,
            "masks_pos": masks_pos, "inp": inp}


def generate(ctx, node: int, year: int = 0, res_m: float | None = None, export: str | None = None,
             sets: list[str] | None = None, steps: int = 9, log=print, return_state: bool = False):
    """生成一个岛群的全部产物，写到 out/<run>/islands/<node>/。返回目录。"""
    from .output import write_preview, write_preview_main, write_terrain
    c = island_config(ctx, sets)
    inp = _node_inputs(ctx, node)
    t0 = time.perf_counter()
    log(f"[island {node}] 陆地 {inp['area_km2']:.0f} km²（主岛 {inp['main_area_km2']:.0f}）峰 {inp['height_m']:.0f} m 可耕 {inp['arable_frac']:.3f} "
        f"河 {'有' if inp['has_river'] else '无'} 岛龄 {inp['age']:.2f} 降水 {inp['precip']:.2f} 温差 {inp['season_range']:.1f} °C")
    g = build_terrain(ctx, node, c, inp, res_m=res_m, log=log)
    if steps >= 2:
        from .hydro import build_hydro
        build_hydro(ctx, node, c, g, log=log)
    if steps >= 3:
        from .climate import build_climate, daily_curves
        build_climate(ctx, node, c, g, log=log)
        g["daily"] = daily_curves(g["climate"], inp, ctx.cfg["s04"]["climate"])
    if steps >= 4:
        from .weather import build_weather
        build_weather(ctx, node, c, g, year=year, log=log)
    if steps >= 5:
        from .settle import build_settlements
        build_settlements(ctx, node, c, g, log=log)
    out = ctx.out_dir / "islands" / str(node)
    g["json"]["meta"]["seconds"] = round(time.perf_counter() - t0, 2)
    write_terrain(out, g)
    if "climate" in g:
        from .climate import write_climate
        write_climate(out, g, year)
    if "settle" in g:
        from .output import write_settlements
        write_settlements(out, g)
    p = write_preview(out, g)
    write_preview_main(out, g)
    log(f"[island {node}] 完成 {time.perf_counter() - t0:.1f} s → {out}")
    if export:
        import shutil
        dst = Path(export) / str(node)
        dst.mkdir(parents=True, exist_ok=True)
        for f in out.iterdir():
            shutil.copy2(f, dst / f.name)
        log(f"  已导出到 {dst}")
    return (out, g) if return_state else out
