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
    g["settle_pop"] = pop
    hh_total = int(round(pop / float(sc["household_size"])))
    n_isl = len(J["islands"])
    ar_isl = np.array([float((arable & (island_id == k)).sum()) for k in range(n_isl)]) * cell_km2
    share = ar_isl / max(1e-9, ar_isl.sum()) if ar_isl.sum() > 0 else np.eye(1, n_isl)[0]
    hh_isl = np.floor(share * hh_total).astype(int)
    richest = int(np.argmax(ar_isl))                    # 可耕地最多的岛（通常是主岛，但主岛可能一块田都没有）
    hh_isl[richest] += hh_total - int(hh_isl.sum())     # 取整的零头
    for k in range(n_isl):                              # 没有田的岛不能有户：挪到可耕地最多的岛
        if ar_isl[k] <= 0 and hh_isl[k] > 0 and k != richest:
            hh_isl[richest] += hh_isl[k]
            hh_isl[k] = 0
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
            cand = fm_local.copy()                                                              # 再不行就落在田块自己的格上（户数不能丢）
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
            # 所有候选格都被占：落在田块里离已有村最远的格
            fi, fj = ii, jj
            tk = np.argwhere(taken)
            if tk.size:
                dd = np.sqrt(((np.stack([fi, fj], 1)[:, None, :] - tk[None, :, :]) ** 2).sum(-1)).min(axis=1)
                b = int(np.argmax(dd))
            else:
                b = 0
            pick = (int(fi[b]), int(fj[b]))
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
    # ---------- 第 2 步：码头 / 桥头 / 导水槽 / 水设施 ----------
    links_out = build_links_water(ctx, g, sc, villages, hamlets, sraster, km, res_km, dist_water)
    step3 = build_homes_city(ctx, node, g, sc, villages, hamlets, fields, links_out, ok_site, score_base, dist_water, km, res_km)
    g["settle_raster"] = sraster
    g["settle_fields"] = fields_raster
    hh_v = sum(r["households"] for r in villages)
    hh_h = sum(r["households"] for r in hamlets)
    S = {"population": round(pop, 0), "population_source": pop_src, "household_size": float(sc["household_size"]), "households": hh_total,
         "households_by_island": hh_isl.tolist(), "land_per_household_km2": round(land_per_hh, 4),
         "n_fields": len(fields), "n_villages": len(villages), "n_hamlets": len(hamlets), "households_in_villages": hh_v, "households_in_hamlets": hh_h,
         "seat": seat["id"] if seat else None, "seat_households": seat["households"] if seat else 0,
         "village_hh_median": int(np.median([r["households"] for r in villages])) if villages else 0,
         "fields": fields, "villages": villages, "hamlets": hamlets, **links_out, **step3,
         "raster_codes": {"1": "田块", "2": "梯田", "3": "村", "4": "散户", "5": "码头", "6": "桥头", "7": "蓄水池", "8": "取水点"},
         "note": "第三层，人口只读 ⑨；村只有位置与户数，无等级（原则乙）。村名是 村NNN 占位。"}
    g["settle"] = S
    J["settlements"] = {k: S[k] for k in ("population", "households", "n_fields", "n_villages", "n_hamlets", "seat_households", "village_hh_median")}
    J["settlements"].update({"n_docks": len(S["docks"]), "n_bridgeheads": len(S["bridgeheads"]), "n_cisterns": len(S["cisterns"]), "n_intakes": len(S["intakes"]),
                             "n_outposts": len(S["outposts"]), "home_candidates": [h["kind"] for h in S["home_candidates"]],
                             "city": ({k: S["city"][k] for k in ("role", "households", "population", "n_guo_islands")} if S.get("city") else None)})
    log(f"  聚落：人口 {pop:.0f}（{pop_src}）→ {hh_total} 户；田块 {len(fields)}，村 {len(villages)}（邑治 {S['seat_households']} 户，中位 {S['village_hh_median']}），散户 {len(hamlets)}；"
        f"村户 {hh_v} + 散户 {hh_h} = {hh_v + hh_h}；码头 {len(S['docks'])}，桥头 {len(S['bridgeheads'])}，蓄水池 {len(S['cisterns'])}，取水点 {len(S['intakes'])}，"
        f"村 1 km 内有水源 {S['water_ok_share']:.0%}；前哨 {len(S['outposts'])}，主家候选 {[h['kind'] for h in S['home_candidates']]}"
        + (f"；{S['city']['role']} {S['city']['households']} 户（城 {S['city']['inner_households']} + 郭 {S['city']['guo_households']}，{S['city']['n_guo_islands']} 岛）" if S.get("city") else ""))


# ---------------------------------------------------------------- 第 2 步：码头、桥头、导水槽、水设施
def _cliff_pts(g, k: int) -> np.ndarray:
    m = g["cliff"] & (g["island_id"] == k)
    ii, jj = np.where(m)
    return np.stack([ii, jj], axis=1)


def _nearest_pair(a: np.ndarray, b: np.ndarray, sub: int = 3) -> tuple[int, int, float]:
    """两组格点间的最近对（b 抽稀），返回 (a 的行号, b 的行号, 距离格数)。"""
    bb = b[::sub] if b.shape[0] > 600 else b
    best = (0, 0, 1e18)
    for s0 in range(0, a.shape[0], 1024):
        blk = a[s0:s0 + 1024]
        d = ((blk[:, None, :] - bb[None, :, :]) ** 2).sum(-1)
        f = int(np.argmin(d))
        i, j = divmod(f, bb.shape[0])
        if d[i, j] < best[2]:
            best = (s0 + i, j * (sub if bb is not b else 1), float(d[i, j]))
    return best[0], best[1], math.sqrt(best[2])


def build_links_water(ctx, g, sc, villages, hamlets, sraster, km, res_km, dist_water) -> dict:
    J = g["json"]
    island_id = g["island_id"]
    H, W = island_id.shape
    n_isl = len(J["islands"])
    cliffs = {k: _cliff_pts(g, k) for k in range(n_isl)}
    # 每岛最大的村（没有村就用散户、再没有就用岛心）
    big = {}
    for r in sorted(villages + hamlets, key=lambda r: -r["households"]):
        big.setdefault(r["island"], r["cell"])
    for k, isl in enumerate(J["islands"]):
        if k not in big:
            r0, c0, m, _ = isl["bbox_cells"]
            ii, jj = np.where(island_id[max(0, r0):r0 + m, max(0, c0):c0 + m] == k)
            big[k] = [int(ii.mean()) + max(0, r0), int(jj.mean()) + max(0, c0)] if ii.size else [0, 0]
    docks, bridgeheads = [], []
    w_v = float(sc["dock_village_weight"])
    merge_of = {k: max(float(sc["dock_merge_km"]), float(sc["dock_merge_rel"]) * math.sqrt(isl["area_km2"])) / res_km for k, isl in enumerate(J["islands"])}
    for e in J["links"]:
        a, b = int(e["a"]), int(e["b"])
        if cliffs[a].shape[0] == 0 or cliffs[b].shape[0] == 0:
            continue
        if e["kind"] == "bridge":
            ia, ib, d = _nearest_pair(cliffs[a], cliffs[b], sub=1)
            for k, idx in ((a, ia), (b, ib)):
                cell = [int(cliffs[k][idx][0]), int(cliffs[k][idx][1])]
                bridgeheads.append({"island": k, "to": b if k == a else a, "cell": cell, "km": km(*cell), "gap_km": e["gap_km"]})
            continue
        for k, other in ((a, b), (b, a)):
            # 岸缘格抽稀到 ≤ 600 / ≤ 300 个点：码头落点精度 1–2 格足够，全量算 99 条短渡要 4 s
            pts = cliffs[k][::max(1, cliffs[k].shape[0] // 600)]
            ob = cliffs[other][::max(1, cliffs[other].shape[0] // 300)]
            # 联合评分：离对岸近 + 离本岛最大村近
            d_other = np.sqrt(((pts[:, None, :] - ob[None, :, :]) ** 2).sum(-1)).min(axis=1)
            v = np.array(big[k], dtype=float)
            d_vill = np.sqrt(((pts - v) ** 2).sum(-1))
            score = d_other + w_v * d_vill
            idx = int(np.argmin(score))
            cell = [int(pts[idx][0]), int(pts[idx][1])]
            # 同岛 2 km 内已有码头就并入
            merged = False
            for dk in docks:
                if dk["island"] == k and math.hypot(dk["cell"][0] - cell[0], dk["cell"][1] - cell[1]) <= merge_of[k]:
                    dk["serves"].append(other)
                    merged = True
                    break
            if not merged:
                docks.append({"island": k, "cell": cell, "km": km(*cell), "serves": [other], "gap_km": e["gap_km"],
                              "village_dist_km": round(float(d_vill[idx]) * res_km, 2)})
    for i, d in enumerate(docks):
        d["id"] = i + 1
        d["serves"] = sorted(set(d["serves"]))
        d["main"] = False
    main_docks = [d for d in docks if d["island"] == 0]
    if main_docks:
        max(main_docks, key=lambda d: (len(d["serves"]), -d["village_dist_km"]))["main"] = True
    for i, bh in enumerate(bridgeheads):
        bh["id"] = i + 1
    channels = [{"a": int(a), "b": int(b), "a_km": J["islands"][a]["center_km"], "b_km": J["islands"][b]["center_km"]} for a, b in J["channels"]]

    # ---------- 水设施 ----------
    cisterns, intakes = [], []
    has_river = bool(J["constraints"]["has_river"].get("actual", False))
    fa = g["flowacc_km2"]
    cliff_dist = distance_bands(g["cliff"], 4)
    slope = g["slope_deg"]
    river = g["river"] > 0
    P_mm = float(J["hydro"]["precip_mm"])
    coef = float(sc["cistern_catch_coef"])

    def upstream(cell, steps=3):
        i, j = cell
        for _ in range(steps):
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < H and 0 <= b < W and island_id[a, b] == island_id[i, j] and fa[a, b] < fa[i, j] and (best is None or fa[a, b] > fa[best]):
                        best = (a, b)
            if best is None:
                break
            i, j = best
        return [int(i), int(j)]

    if not has_river:
        mouths = J["hydro"].get("main_basins", {}).get("mouths", [])
        main_area = J["islands"][0]["area_km2"]
        for mi, mj, area in mouths:
            if area < float(sc["cistern_basin_min_frac"]) * main_area:
                continue
            cell = upstream((mi, mj), 3)
            cisterns.append({"island": 0, "cell": cell, "km": km(*cell), "basin_km2": area, "capacity_1000m3": round(area * P_mm * coef, 0)})
    for k in range(1 if has_river else 0, n_isl):
        if has_river and k == 0:
            continue
        m = (island_id == k) & (cliff_dist >= 2)
        if not m.any():
            m = island_id == k
        fa_k = np.where(m, fa, -1.0)
        p = int(np.argmax(fa_k))
        cell = [p // W, p % W]
        if k == 0 and cisterns:
            continue
        area_k = J["islands"][k]["area_km2"]
        cisterns.append({"island": k, "cell": cell, "km": km(*cell), "basin_km2": round(float(fa[cell[0], cell[1]]), 2), "capacity_1000m3": round(area_k * P_mm * coef * 0.5, 0)})
    if has_river:
        rc = np.stack(np.where(river & (island_id == 0)), axis=1)
        for r in villages:
            if r["island"] != 0 or rc.shape[0] == 0:
                continue
            d = ((rc - np.array(r["cell"])) ** 2).sum(-1)
            p = int(np.argmin(d))
            cell = [int(rc[p][0]), int(rc[p][1])]
            intakes.append({"island": 0, "village": r["id"], "cell": cell, "km": km(*cell), "dist_km": round(math.sqrt(float(d[p])) * res_km, 2)})
    for i, x in enumerate(cisterns):
        x["id"] = i + 1
    for i, x in enumerate(intakes):
        x["id"] = i + 1
    # 村的水源：河 / 湖 / 溪涧（≤ water_near）、取水点、蓄水池
    wsrc = np.stack([np.array(x["cell"]) for x in cisterns], axis=0) if cisterns else np.zeros((0, 2))
    ok = 0
    lim = float(sc["village_water_km"]) / res_km
    for r in villages:
        dw = float(dist_water[r["cell"][0], r["cell"][1]])
        dc = float(np.sqrt(((wsrc - np.array(r["cell"])) ** 2).sum(-1)).min()) if wsrc.shape[0] else 1e9
        src, dd = ("河湖溪涧", dw) if dw <= dc else ("蓄水池", dc)
        if has_river and r["island"] == 0:
            it = next((x for x in intakes if x["village"] == r["id"]), None)
            if it and it["dist_km"] / res_km < dd:
                src, dd = "取水点", it["dist_km"] / res_km
        r["water"] = {"source": src, "dist_km": round(dd * res_km, 2)}
        if dd <= lim:
            ok += 1
    for d in docks:
        sraster[d["cell"][0], d["cell"][1]] = 5
    for bh in bridgeheads:
        sraster[bh["cell"][0], bh["cell"][1]] = 6
    for x in cisterns:
        sraster[x["cell"][0], x["cell"][1]] = 7
    for x in intakes:
        sraster[x["cell"][0], x["cell"][1]] = 8
    return {"docks": docks, "bridgeheads": bridgeheads, "channels": channels, "cisterns": cisterns, "intakes": intakes,
            "has_river": has_river, "water_ok_share": round(ok / max(1, len(villages)), 3)}


# ---------------------------------------------------------------- 第 3 步：前哨、主家候选、都与城
def _polity_role(ctx, node: int) -> dict | None:
    """本群是不是某邦的都；是则给邦人口与是否变法之国（只读 ⑨）。"""
    p = ctx.stage_dir(9) / "polity.npz"
    if not p.exists():
        return None
    import json as _json
    with np.load(p) as z:
        st = int(z["state"][node]) if "state" in z.files else -1
        if st < 0 or "capital" not in z.files or int(z["capital"][st]) != node:
            return None
        pop_state = float(z["pop"][z["state"] == st].sum())
        poly = int(z["polity"][node]) if "polity" in z.files else -1
    reformer = False
    pj = ctx.stage_dir(9) / "polities.json"
    if pj.exists():
        P = _json.loads(pj.read_text(encoding="utf-8"))
        reformer = (P.get("reformer") or {}).get("polity") == poly
    return {"state": st, "state_pop": pop_state, "reformer": reformer}


def build_homes_city(ctx, node, g, sc, villages, hamlets, fields, links, ok_site, score_base, dist_water, km, res_km) -> dict:
    J = g["json"]
    island_id = g["island_id"]
    H, W = island_id.shape
    height = np.where(island_id >= 0, g["height"], -1.0)
    C = g.get("climate", {})
    seat = next((v for v in villages if v.get("seat")), None)
    main_dock = next((d for d in links["docks"] if d.get("main")), None)
    docks_of = {}
    for d in links["docks"]:
        docks_of.setdefault(d["island"], d)
    biggest = {}
    for r in sorted(villages + hamlets, key=lambda r: -r["households"]):
        biggest.setdefault(r["island"], r)
    field_km2 = {}
    for f in fields:
        field_km2[f["island"]] = field_km2.get(f["island"], 0.0) + f["area_km2"]

    # ---- 前哨：有田有码头的小岛各一个，落在该岛最大的村 / 散户
    outposts = []
    for k in range(1, len(J["islands"])):
        if k in docks_of and field_km2.get(k, 0) > 0 and k in biggest:
            r = biggest[k]
            outposts.append({"id": len(outposts) + 1, "island": k, "cell": r["cell"], "km": r["km"], "households": r["households"],
                             "field_km2": round(field_km2[k], 3), "dock": docks_of[k]["id"], "settlement": r["name"]})

    def info(cell, kind, note):
        i, j = cell
        k = int(island_id[i, j])
        isl = J["islands"][k]
        d_seat = math.hypot(i - seat["cell"][0], j - seat["cell"][1]) * res_km if seat else None
        d_dock = math.hypot(i - main_dock["cell"][0], j - main_dock["cell"][1]) * res_km if main_dock else None
        wsum = C.get("weather", {})
        return {"kind": kind, "note": note, "island": k, "cell": [int(i), int(j)], "km": km(i, j), "elev_m": round(float(g["height"][i, j]), 0),
                "island_age": isl["age_zh"], "island_has_river": bool(isl.get("has_perennial_river", False)), "island_lakes": int(isl.get("n_lakes", 0)),
                "dist_seat_km": None if d_seat is None else round(d_seat, 2), "dist_main_dock_km": None if d_dock is None else round(d_dock, 2),
                "season_type": C.get("season_type_zh"), "season_names": C.get("season_names"),
                "snow_days": wsum.get("snow_days"), "storm_days": wsum.get("storm_days"), "sailable_days": wsum.get("sailable_days"),
                "landcover": int(g["landcover"][i, j]), "water_dist_km": round(float(dist_water[i, j]) * res_km, 2)}

    homes = []
    vcells = np.array([v["cell"] for v in villages], dtype=float) if villages else np.zeros((0, 2))
    def dist_to_villages(mask):
        """掩膜内每格到最近村的距离（格）：村数少，直接广播。"""
        ii, jj = np.where(mask)
        if vcells.shape[0] == 0 or ii.size == 0:
            return ii, jj, np.full(ii.size, 1e9)
        pts = np.stack([ii, jj], axis=1).astype(float)
        d = np.full(ii.size, 1e18)
        for s0 in range(0, ii.size, 20000):
            blk = pts[s0:s0 + 20000]
            d[s0:s0 + 20000] = np.sqrt(((blk[:, None, :] - vcells[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return ii, jj, d
    # 候选一：邑治旁（1–3 km 环内，近主码头）
    if seat:
        si, sj = seat["cell"]
        lo, hi = 1.0 / res_km, 3.0 / res_km
        ii, jj = np.where(ok_site & (island_id == seat["island"]))
        d = np.hypot(ii - si, jj - sj)
        sel = (d >= lo) & (d <= hi)
        if sel.any():
            ii, jj, d = ii[sel], jj[sel], d[sel]
            dd = np.hypot(ii - main_dock["cell"][0], jj - main_dock["cell"][1]) if main_dock else d
            sc_ = score_base[ii, jj] - 0.02 * dd
            b = int(np.argmax(sc_))
            homes.append(info((ii[b], jj[b]), "邑治旁", "人多、近航线：邑治与主码头之间的一块地"))
    # 候选二：河湖僻处（离任何村 ≥ 3 km、近水、坡缓）；没有就换「高台」
    far = 3.0 / res_km
    stride = np.zeros((H, W), dtype=bool); stride[::3, ::3] = True          # 候选格抽稀 3×3，够用且快
    ii, jj, dv = dist_to_villages(ok_site & (dist_water <= 5) & stride)
    sel = dv >= far
    if sel.any():
        ii, jj = ii[sel], jj[sel]
        sc_ = score_base[ii, jj] + 0.01 * np.minimum(dv[sel], 3 * far)
        b = int(np.argmax(sc_))
        homes.append(info((ii[b], jj[b]), "河湖僻处", "离村三公里外、临水的僻静处"))
    else:
        ii, jj = np.where(ok_site & (island_id == 0))
        if ii.size:
            b = int(np.argmax(height[ii, jj]))
            homes.append(info((ii[b], jj[b]), "高台", "主岛能落脚的最高处，俯瞰全群"))
    # 候选三：小岛前哨（离主岛最远的前哨）
    if outposts:
        c0 = np.array(J["islands"][0]["center_km"])
        o = max(outposts, key=lambda o: math.hypot(o["km"][0] - c0[0], o["km"][1] - c0[1]))
        h = info(o["cell"], "小岛前哨", f"离主岛最远的小岛，一片田（{o['field_km2']} km²）和一个码头")
        h["outpost"] = o["id"]
        homes.append(h)
    for i, h in enumerate(homes):
        h["id"] = i + 1

    # ---- 都与城（5.7）
    city = None
    role = _polity_role(ctx, node)
    if role and seat:
        pop_own = float(g["settle_pop"]) if "settle_pop" in g else 0.0
        rate = float(sc["city_gather_rate_reformer"] if role["reformer"] else sc["city_gather_rate"])
        city_pop = pop_own * float(sc["city_urban_rate"]) + role["state_pop"] * rate
        hh_size = float(sc["household_size"])
        city_hh = int(round(city_pop / hh_size))
        # 郭：索桥可达（导水槽树 = 索桥 MST）且离城 ≤ city_guo_km 的岛上的村
        reach = {0}
        adj = {}
        for a, b in J["channels"]:
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        stack = [0]
        while stack:
            a = stack.pop()
            for b in adj.get(a, []):
                if b not in reach:
                    reach.add(b); stack.append(b)
        lim = float(sc["city_guo_km"]) / res_km
        guo = [v for v in villages if v["island"] in reach and v["island"] != 0 and math.hypot(v["cell"][0] - seat["cell"][0], v["cell"][1] - seat["cell"][1]) <= lim]
        guo += [v for v in villages if v["island"] == 0 and v is not seat and math.hypot(v["cell"][0] - seat["cell"][0], v["cell"][1] - seat["cell"][1]) <= lim]
        guo_hh = sum(v["households"] for v in guo)
        inner_hh = max(seat["households"], city_hh - guo_hh)
        for v in guo:
            v["guo"] = True
        seat["city"] = True
        # 仓城 = 主码头；祭台：无河 → 最大蓄水池周围 5 格内最高处；有河 → 城 2 km 内能落脚的最高处
        altar = None
        if not links["has_river"] and links["cisterns"]:
            cz = max(links["cisterns"], key=lambda x: x["basin_km2"])
            i0, j0 = cz["cell"]
            r0, r1, c0, c1 = max(0, i0 - 5), min(H, i0 + 6), max(0, j0 - 5), min(W, j0 + 6)
            sub = np.where(island_id[r0:r1, c0:c1] == cz["island"], height[r0:r1, c0:c1], -1)
            b = int(np.argmax(sub)); altar = [r0 + b // (c1 - c0), c0 + b % (c1 - c0)]
        else:
            ii, jj = np.where(ok_site & (island_id == 0))
            d = np.hypot(ii - seat["cell"][0], jj - seat["cell"][1])
            sel = d <= 2.0 / res_km
            if sel.any():
                ii, jj = ii[sel], jj[sel]; b = int(np.argmax(height[ii, jj])); altar = [int(ii[b]), int(jj[b])]
        city = {"role": "变法之国的都" if role["reformer"] else "邦都", "state": role["state"], "state_pop": round(role["state_pop"], 0),
                "population": round(city_pop, 0), "households": city_hh, "inner_households": int(inner_hh), "guo_households": int(guo_hh),
                "guo_villages": [v["id"] for v in guo], "n_guo_islands": len({v["island"] for v in guo}),
                "seat_village": seat["id"], "granary_dock": main_dock["id"] if main_dock else None,
                "altar_cell": altar, "altar_km": km(*altar) if altar else None,
                "gather_rate": rate, "urban_rate": float(sc["city_urban_rate"]),
                "note": "城居人口 = 本邑人口 × 城居率 + 邦人口 × 集聚率，只在聚落层算，不进 ⑨；郭 = 索桥可达且离城 ≤ city_guo_km 的村；崖缘即城墙、桥头即城门"}
    return {"outposts": outposts, "home_candidates": homes, "city": city}
