"""聚落生成器（docs/PLAN-SETTLE.md）：给一个已生成的岛群，按人口与地形落下田块、村落与散户（第 1 步），
码头 / 桥头 / 水设施（第 2 步），主家候选 / 前哨 / 都与城（第 3 步）。

第三层，不回灌：人口只读 ⑨ 的 pop（没有 ⑨ 时按可耕地 × 100 人/km²）；不输出任何社会属性（原则乙、铁律五）。
随机数：entity_rng(seed, ISLAND_STREAM, f"settle:{node}:{部件}")。坐标：格 (row, col) 与 km（x 东 y 北，相对群心）。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import binary_dilate, distance_bands, label_components, shift

HOUSEHOLD = 5.0


def _pop_of(ctx, node: int, arable_km2: float, c: dict) -> tuple[float, str]:
    p = ctx.stage_dir(9) / "polity.npz"
    if p.exists():
        with np.load(p) as z:
            if "pop" in z.files and node < z["pop"].size:
                return float(z["pop"][node]), "⑨ polity.npz"
    return arable_km2 * float(ctx.cfg["shared"]["scale"]["people_per_arable_km2"]), "可耕地 × 人口密度（无 ⑨）"


def _kmeans_split(rng, ii: np.ndarray, jj: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """把一块田的格按坐标分成 k 份（Lloyd，种子确定）。返回每格的份号。"""
    n = ii.size
    if k <= 1 or n <= k:
        return np.zeros(n, dtype=np.int32)
    pts = np.stack([ii, jj], axis=1).astype(np.float64)
    ctr = pts[rng.choice(n, size=k, replace=False)]
    lab = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d = ((pts[:, None, :] - ctr[None, :, :]) ** 2).sum(-1)
        lab = np.argmin(d, axis=1)
        for m in range(k):
            sel = lab == m
            if sel.any():
                ctr[m] = pts[sel].mean(axis=0)
    return lab


def _wind_exposure(height: np.ndarray, land: np.ndarray, res_m: float, u: float, v: float) -> np.ndarray:
    hh = np.where(land, height, np.nan)
    def nb(di, dj):
        a = shift(hh, di, dj, np.nan)
        return np.where(np.isnan(a), hh, a)
    gx = (nb(0, -1) - nb(0, 1)) / (2 * res_m)
    gy = (nb(-1, 0) - nb(1, 0)) / (2 * res_m)
    gn = np.hypot(gx, gy)
    sp = math.hypot(u, v)
    if sp < 1e-6:
        return np.zeros_like(hh)
    dx = np.where(gn > 1e-9, gx / np.maximum(gn, 1e-9), 0.0)
    dy = np.where(gn > 1e-9, gy / np.maximum(gn, 1e-9), 0.0)
    ux, uy = -u / sp, -v / sp
    return np.nan_to_num(np.clip(dx * ux + dy * uy, -1.0, 1.0) * np.clip(gn * 1000.0 / 0.26, 0.0, 1.0))


def build_settlements(ctx, node: int, c: dict, g: dict, log=print) -> None:
    from . import _rng
    sc = c["settle"]
    J = g["json"]
    res_km = g["res_km"]
    res_m = res_km * 1000.0
    cell_km2 = res_km * res_km
    island_id = g["island_id"]
    land = island_id >= 0
    H, W = island_id.shape
    arable = g["arable"] > 0
    slope = g["slope_deg"].astype(np.float64)
    water = (g["river"] > 0) | g["lake"] | (g["stream"] > 0)
    # 漫滩：河两格内坡 < 3°（与调试台同口径）
    flood = binary_dilate(g["river"] > 0, 2) & (slope < 3.0) & ~(g["river"] > 0)
    dist_water = distance_bands(water, int(sc["water_near_cells"]))
    dist_shore = distance_bands(g["cliff"], int(sc["shore_near_cells"]))
    wl = ctx.load_npz(4, "wind_local")
    from ..sphere import grid_interp
    inp = g["inp"]
    u = float(grid_interp(wl["u"], wl["lats"], wl["lons"], inp["lat"], inp["lon"]))
    v = float(grid_interp(wl["v"], wl["lats"], wl["lons"], inp["lat"], inp["lon"]))
    expo = _wind_exposure(np.where(land, g["height"], 0.0), land, res_m, u, v)
    x0, y0 = J["raster"]["origin_km"]
    km = lambda i, j: [round(x0 + (j + 0.5) * res_km, 3), round(y0 - (i + 0.5) * res_km, 3)]

    # ---------- 人口 → 户 ----------
    arable_km2 = float(arable.sum()) * cell_km2
    pop, pop_src = _pop_of(ctx, node, arable_km2, c)
    hh_total = int(round(pop / float(sc["household_size"])))
    n_isl = len(J["islands"])
    ar_isl = np.array([float((arable & (island_id == k)).sum()) for k in range(n_isl)]) * cell_km2
    share = ar_isl / max(1e-9, ar_isl.sum()) if ar_isl.sum() > 0 else np.eye(1, n_isl)[0]
    hh_isl = np.floor(share * hh_total).astype(int)
    hh_isl[0] += hh_total - int(hh_isl.sum())          # 取整的零头归主岛
    land_per_hh = arable_km2 / max(1, hh_total)         # 户均地量 km²

    # ---------- 田块：可耕地 8 邻域连通块；小块并入散户田；大块按 80 户切分 ----------
    lab, n_lab = label_components(arable, 8)
    fields_raster = np.zeros((H, W), dtype=np.int32)
    fields = []
    rng = _rng(ctx, node, "settle:fields")
    cap_village = int(sc["village_max_hh"])
    cap_seat = int(sc["seat_max_hh"])
    fid = 0
    counts = np.bincount(lab.ravel(), minlength=n_lab + 1)
    # 每个连通块的格坐标一次取出（按标号排序），不逐块扫全图
    flat_l = lab.ravel()
    nz_l = np.nonzero(flat_l)[0]
    ord_l = nz_l[np.argsort(flat_l[nz_l], kind="stable")]
    bnd_l = np.searchsorted(flat_l[ord_l], np.arange(1, n_lab + 2))
    first_isl = island_id.ravel()[ord_l[bnd_l[:-1]]] if n_lab else np.zeros(0, dtype=int)
    # 主岛最大的田块可以撑到邑治规模
    main_biggest = -1
    if n_lab:
        main_lab = [(counts[L], L) for L in range(1, n_lab + 1) if first_isl[L - 1] == 0]
        if main_lab:
            main_biggest = max(main_lab)[1]
    for L in range(1, n_lab + 1):
        cl = ord_l[bnd_l[L - 1]:bnd_l[L]]
        ii, jj = cl // W, cl % W
        k_isl = int(first_isl[L - 1])
        area = ii.size * cell_km2
        hh_est = area / max(1e-9, land_per_hh)
        cap = cap_seat if L == main_biggest else cap_village
        k = max(1, int(math.ceil(hh_est / cap)))
        parts = _kmeans_split(rng, ii, jj, k)
        for m in range(k):
            sel = parts == m
            if not sel.any():
                continue
            fid += 1
            fields_raster[ii[sel], jj[sel]] = fid
            fields.append({"id": fid, "island": k_isl, "cells": int(sel.sum()), "area_km2": round(float(sel.sum()) * cell_km2, 3),
                           "terrace_frac": round(float((g["arable"][ii[sel], jj[sel]] == 2).mean()), 3),
                           "centroid_cell": [int(round(ii[sel].mean())), int(round(jj[sel].mean()))],
                           "centroid_km": km(ii[sel].mean(), jj[sel].mean()), "water_dist_km": round(float(dist_water[ii[sel], jj[sel]].min()) * res_km, 2)})
    # 户数按岛内田块面积分配（最大余数法，保证每岛之和 = 岛户数）
    for k_isl in range(n_isl):
        fs = [f for f in fields if f["island"] == k_isl]
        if not fs:
            continue
        tot = sum(f["area_km2"] for f in fs)
        raw = [hh_isl[k_isl] * f["area_km2"] / max(1e-9, tot) for f in fs]
        base = [int(math.floor(x)) for x in raw]
        rem = int(hh_isl[k_isl]) - sum(base)
        order = sorted(range(len(fs)), key=lambda i: -(raw[i] - base[i]))
        for i in order[:max(0, rem)]:
            base[i] += 1
        for f, b in zip(fs, base):
            f["households"] = int(b)

    # ---------- 村址评分 ----------
    ok_site = land & ~arable & ~g["cliff"] & ~water & ~flood & (slope < float(sc["site_slope_max_deg"]))
    w = sc["site_weights"]
    score_base = (float(w["water"]) * np.clip(1.0 - dist_water / (int(sc["water_near_cells"]) + 1.0), 0.0, 1.0)
                  + float(w["slope"]) * np.clip(1.0 - slope / float(sc["site_slope_max_deg"]), 0.0, 1.0)
                  + float(w["lee"]) * (1.0 - expo) / 2.0
                  + float(w["shore"]) * np.clip(1.0 - dist_shore / (int(sc["shore_near_cells"]) + 1.0), 0.0, 1.0))
    reach = int(sc["site_reach_cells"])
    min_sep = int(round(float(sc["village_min_sep_km"]) / res_km))
    villages, hamlets = [], []
    taken = np.zeros((H, W), dtype=bool)
    min_v = int(sc["village_min_hh"])
    # 每块田的格坐标一次取出（按田号排序），避免逐田扫全图
    flat_f = fields_raster.ravel()
    nz = np.nonzero(flat_f)[0]
    order_f = nz[np.argsort(flat_f[nz], kind="stable")]
    bounds = np.searchsorted(flat_f[order_f], np.arange(1, fid + 2))
    cells_of = {k: order_f[bounds[k - 1]:bounds[k]] for k in range(1, fid + 1)}
    for f in sorted(fields, key=lambda f: -f.get("households", 0)):
        hh = f.get("households", 0)
        if hh <= 0:
            continue
        cells = cells_of[f["id"]]
        ii, jj = cells // W, cells % W
        r0, r1 = max(0, ii.min() - reach), min(H, ii.max() + reach + 1)
        c0, c1 = max(0, jj.min() - reach), min(W, jj.max() + reach + 1)
        sl = (slice(r0, r1), slice(c0, c1))
        fm_local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        fm_local[ii - r0, jj - c0] = True
        dfield = distance_bands(fm_local, reach)
        cand = ok_site[sl] & (dfield <= reach) & (island_id[sl] == f["island"])
        if not cand.any():
            cand = (island_id[sl] == f["island"]) & land[sl] & ~water[sl] & (dfield <= reach)   # 退而求其次：允许落在田上
        if not cand.any():
            continue
        sc_map = np.where(cand, score_base[sl] + float(w["field"]) * (1.0 - dfield / (reach + 1.0)), -1e9)
        flat = np.argsort(-sc_map.ravel(), kind="stable")[:200]
        pick = None
        for p in flat:
            i, j = divmod(int(p), c1 - c0)
            if sc_map[i, j] <= -1e8:
                break
            gi, gj = i + r0, j + c0
            if hh >= min_v and taken[max(0, gi - min_sep):gi + min_sep + 1, max(0, gj - min_sep):gj + min_sep + 1].any():
                continue
            pick = (gi, gj)
            break
        if pick is None:
            # 退而求其次：放弃 1 km 间距，但绝不与已有村同格
            for p in np.argsort(-sc_map.ravel(), kind="stable"):
                i, j = divmod(int(p), c1 - c0)
                if sc_map[i, j] <= -1e8:
                    break
                if not taken[i + r0, j + c0]:
                    pick = (i + r0, j + c0)
                    break
        if pick is None:
            continue
        gi, gj = pick
        rec = {"id": 0, "island": f["island"], "cell": [int(gi), int(gj)], "km": km(gi, gj), "households": int(hh), "field": f["id"],
               "elev_m": round(float(g["height"][gi, gj]), 0), "water_dist_km": round(float(dist_water[gi, gj]) * res_km, 2),
               "shore_dist_km": round(float(dist_shore[gi, gj]) * res_km, 2), "on_arable": bool(arable[gi, gj])}
        f["village"] = None
        if hh >= min_v:
            taken[gi, gj] = True
            villages.append(rec)
        else:
            hamlets.append(rec)
    villages.sort(key=lambda r: (-r["households"], r["island"], r["cell"]))
    for k, r in enumerate(villages):
        r["id"] = k + 1
        r["name"] = f"村{k + 1:03d}"
    for k, r in enumerate(hamlets):
        r["id"] = k + 1
        r["name"] = f"散户{k + 1:03d}"
    vf = {r["field"]: r["id"] for r in villages}
    hf = {r["field"]: -r["id"] for r in hamlets}
    for f in fields:
        f["village"] = vf.get(f["id"], hf.get(f["id"]))
    seat = next((r for r in villages if r["island"] == 0), villages[0] if villages else None)
    if seat:
        seat["seat"] = True
    # 栅格：1 田 / 2 梯田 / 3 村 / 4 散户
    sraster = np.zeros((H, W), dtype=np.uint8)
    sraster[fields_raster > 0] = 1
    sraster[(fields_raster > 0) & (g["arable"] == 2)] = 2
    for r in villages:
        sraster[r["cell"][0], r["cell"][1]] = 3
    for r in hamlets:
        sraster[r["cell"][0], r["cell"][1]] = 4
    g["settle_raster"] = sraster
    g["settle_fields"] = fields_raster
    hh_v = sum(r["households"] for r in villages)
    hh_h = sum(r["households"] for r in hamlets)
    S = {"population": round(pop, 0), "population_source": pop_src, "household_size": float(sc["household_size"]), "households": hh_total,
         "households_by_island": hh_isl.tolist(), "land_per_household_km2": round(land_per_hh, 4),
         "n_fields": len(fields), "n_villages": len(villages), "n_hamlets": len(hamlets), "households_in_villages": hh_v, "households_in_hamlets": hh_h,
         "seat": seat["id"] if seat else None, "seat_households": seat["households"] if seat else 0,
         "village_hh_median": int(np.median([r["households"] for r in villages])) if villages else 0,
         "fields": fields, "villages": villages, "hamlets": hamlets,
         "raster_codes": {"1": "田块", "2": "梯田", "3": "村", "4": "散户"},
         "note": "第三层，人口只读 ⑨；村只有位置与户数，无等级（原则乙）。村名是 村NNN 占位。"}
    g["settle"] = S
    J["settlements"] = {k: S[k] for k in ("population", "households", "n_fields", "n_villages", "n_hamlets", "seat_households", "village_hh_median")}
    log(f"  聚落：人口 {pop:.0f}（{pop_src}）→ {hh_total} 户；田块 {len(fields)}，村 {len(villages)}（邑治 {S['seat_households']} 户，中位 {S['village_hh_median']}），散户 {len(hamlets)}；"
        f"村户 {hh_v} + 散户 {hh_h} = {hh_v + hh_h}")
