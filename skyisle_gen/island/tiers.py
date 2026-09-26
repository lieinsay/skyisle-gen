"""聚落层级与交通（DESIGN-NOTES 四点十八）：专业聚落、集镇、飞船泊场、村周开垦。settle.py 调用。

世界设定：飞船取代了车船，能在任意平地停靠——没有码头，交通不绑岸线，集镇的服务范围按直线跨岛算。
人口口径：总户数只读 ⑨；其中 nonfarm_share 是非农户（前工业社会约一到两成），先给资源造出的专业聚落（矿镇、浮石采石村、窑村、
烧炭营、温泉地；封顶非农户的 special_cap_frac），余下按服务户数分给集镇（中心地：半径内户数最多的村升镇，镇距 ≥ town_spacing_km，邑治必为镇）。
农户照旧按田块分到村。Σ 村农户 + 散户 + 镇的非农户 + 专业聚落户 = 总户数（SET-pop）。

纯函数：输入数组与记录，返回新记录；开垦直接改 g["landcover"] / g["patch_id"]，开采（village_workings）往 g["resources"]["workings"] 里加采场，
两者之后 settle 调 resources.sync_resources 重数片的面积、主导栅格与地表占比。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import binary_dilate, nearest_propagate

LC_FOREST, LC_SHRUB, LC_GRASS = 4, 5, 6
GRADE_W = {"上": 1.5, "中": 1.0, "下": 0.5}


def _near_site(ok_site, fallback, island_id, cell, k, reach_cells):
    """cell 附近（≤ reach 格）同岛可落脚的格里离 cell 最近的；没有就退到 fallback（非水非崖的陆地）；再没有返回 None（不设）。"""
    H, W = island_id.shape
    i, j = cell
    r0, r1, c0, c1 = max(0, i - reach_cells), min(H, i + reach_cells + 1), max(0, j - reach_cells), min(W, j + reach_cells + 1)
    for m in (ok_site, fallback):
        win = m[r0:r1, c0:c1] & (island_id[r0:r1, c0:c1] == k)
        if win.any():
            ii, jj = np.where(win)
            t = int(np.argmin((ii + r0 - i) ** 2 + (jj + c0 - j) ** 2))
            return [int(ii[t] + r0), int(jj[t] + c0)]
    return None


def special_settlements(g, sc, villages, nonfarm_hh: int, ok_site, km, res_km) -> list[dict]:
    """资源造出的专业聚落。需求按资源量算，总户数封顶 special_cap_frac × 非农户（超了等比缩，缩完 < special_min_hh 的不设）。"""
    R = g.get("resources")
    if not R or nonfarm_hh <= 0:
        return []
    island_id = g["island_id"]
    dep = R["deposits"]
    occ = R["occurrences"]
    reach = int(round(1.0 / res_km))
    vcells = np.array([v["cell"] for v in villages], dtype=float) if villages else np.zeros((0, 2))
    near_forest = binary_dilate(g["landcover"] == LC_FOREST, max(1, int(round(float(sc["kiln_forest_km"]) / res_km))))
    want = []
    # 矿镇 / 矿村：每条矿化带（金属矿赋存区）一个，落在第一个矿坑旁；带里的矿坑记到它名下
    for o in occ:
        if o["kind"] == "ore":
            hh = float(sc["ore_hh_per_km2"]) * o["area_km2"] * GRADE_W.get(o["grade"], 1.0)
            pits = [w for w in R["workings"] if w["occurrence"] == o["id"]]
            cell = pits[0]["cell"] if pits else o["cell"]
            want.append({"kind": "矿镇" if hh >= float(sc["mine_town_hh"]) else "矿村", "resource": None, "occurrence": o["id"], "subtype": o["subtype"],
                         "island": o["island"], "at": cell, "hh": hh,
                         "note": f"{o['subtype']}矿化带 {o['area_km2']:.1f} km²（{o['grade']}品，{len(pits)} 坑），吃粮靠外运"})
    # 浮石采石村：每岛最大的几段露头
    by_isl: dict[int, list] = {}
    for d in dep:
        if d["kind"] == "floatstone" and d["area_km2"] >= float(sc["floatstone_site_min_km2"]):
            by_isl.setdefault(d["island"], []).append(d)
    for k, lst in sorted(by_isl.items()):
        lst.sort(key=lambda d: -d["area_km2"])
        for d in lst[: int(sc["floatstone_sites_main"]) if k == 0 else 1]:
            want.append({"kind": "浮石采石村", "resource": d["id"], "subtype": d["subtype"], "island": k, "at": d["cell"],
                         "hh": min(60.0, float(sc["floatstone_hh_per_km2"]) * d["area_km2"]), "note": f"{d['subtype']} {d['area_km2']:.2f} km²；飞船就地装运"})
    # 窑村：上等黏土区（品位高、面积大的先）+ 近林（燃料）
    n_kiln: dict[int, int] = {}
    for o in sorted((o for o in occ if o["kind"] == "clay"), key=lambda o: (-o["grade_peak"], -o["area_km2"], o["id"])):
        if o["grade"] == "上" and near_forest[o["cell"][0], o["cell"][1]]:
            k = o["island"]
            if n_kiln.get(k, 0) >= (2 if k == 0 else 1):
                continue
            n_kiln[k] = n_kiln.get(k, 0) + 1
            want.append({"kind": "窑村", "resource": None, "occurrence": o["id"], "subtype": o["subtype"], "island": k, "at": o["cell"], "hh": float(sc["kiln_hh"]),
                         "note": f"上等{o['subtype']}（{o['area_km2']:.2f} km²），附近有林可烧"})
    # 伐木烧炭营：离村远的大林场
    n_ch: dict[int, int] = {}
    far = float(sc["charcoal_far_km"]) / res_km
    for d in sorted(dep, key=lambda d: -d["area_km2"]):
        if d["kind"] != "timber" or d["area_km2"] < float(sc["charcoal_min_km2"]):
            continue
        dv = float(np.sqrt(((vcells - np.array(d["cell"])) ** 2).sum(-1)).min()) if vcells.shape[0] else 1e9
        k = d["island"]
        if dv <= far or n_ch.get(k, 0) >= 3:
            continue
        n_ch[k] = n_ch.get(k, 0) + 1
        want.append({"kind": "烧炭营", "resource": d["id"], "subtype": d["subtype"], "island": k, "at": d["cell"],
                     "hh": float(np.clip(d["area_km2"] * 0.5, 5, 30)), "note": f"{d['subtype']} {d['area_km2']:.0f} km²，离最近的村 {dv * res_km:.1f} km（季节性）"})
    for d in dep:
        if d["kind"] == "hotspring":
            want.append({"kind": "温泉地", "resource": d["id"], "subtype": None, "island": d["island"], "at": d["cell"], "hh": float(sc["hotspring_hh"]),
                         "note": "汤治 / 寺社"})
    if not want:
        return []
    fallback = (island_id >= 0) & ~g["cliff"] & ~g["lake"] & ~(g["river"] > 0)
    for w in want:
        w["cell"] = _near_site(ok_site, fallback, island_id, w["at"], w["island"], reach)
    want = [w for w in want if w["cell"] is not None]      # 岛上全是崖缘（极小的岛）：落不下，不设
    cap = float(sc["special_cap_frac"]) * nonfarm_hh
    tot = sum(w["hh"] for w in want)
    f = min(1.0, cap / max(1e-9, tot))
    out = []
    for w in want:
        hh = int(math.floor(w["hh"] * f))
        if hh < int(sc["special_min_hh"]):
            continue
        cell = w["cell"]
        out.append({"id": len(out) + 1, "kind": w["kind"], "resource": w["resource"], "occurrence": w.get("occurrence"), "subtype": w["subtype"],
                    "island": w["island"], "cell": cell, "km": km(*cell), "households": hh, "note": w["note"]})
    for s in out:
        s["name"] = f"{s['kind']}{s['id']:02d}"
        if s["kind"] in ("矿镇", "矿村"):
            for wk in R["workings"]:
                if wk["occurrence"] == s["occurrence"]:
                    wk["special"] = s["id"]
    return out


def market_towns(villages, seat, extra_hh: int, sc, res_km) -> list[dict]:
    """中心地：按 market_radius_km 内（直线跨岛）的户数贪心挑村升为镇，镇距 ≥ town_spacing_km，邑治先入；
    每个村归最近的镇；extra_hh（非农户余量）按各镇服务户数分（最大余数法）。改写 villages 的 town / market_town / households_market。"""
    if not villages:
        return []
    P = np.array([v["km"] for v in villages], dtype=float)
    hh = np.array([v["households"] for v in villages], dtype=float)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    cen = (np.where(D <= float(sc["market_radius_km"]), 1.0, 0.0) * hh[None, :]).sum(1)
    idx_seat = villages.index(seat) if seat in villages else int(np.argmax(cen))
    chosen = [idx_seat]
    for t in np.argsort(-cen, kind="stable").tolist():
        if t in chosen or cen[t] < float(sc["town_min_served_hh"]):
            continue
        if D[t, chosen].min() >= float(sc["town_spacing_km"]):
            chosen.append(int(t))
    near = np.array(chosen)[np.argmin(D[:, chosen], axis=1)]
    towns = []
    served = {c: float(hh[near == c].sum()) for c in chosen}
    raw = [extra_hh * served[c] / max(1e-9, sum(served.values())) for c in chosen]
    base = [int(math.floor(x)) for x in raw]
    for t in sorted(range(len(chosen)), key=lambda t: -(raw[t] - base[t]))[: max(0, extra_hh - sum(base))]:
        base[t] += 1
    for n, (c, add) in enumerate(zip(chosen, base)):
        v = villages[c]
        v["town"] = n + 1
        v["households_market"] = int(add)
        towns.append({"id": n + 1, "name": "邑治" if c == idx_seat else f"镇{n + 1:02d}", "village": v["id"], "island": v["island"],
                      "cell": v["cell"], "km": v["km"], "households_farm": v["households"], "households_market": int(add),
                      "households": v["households"] + int(add), "served_villages": int((near == c).sum()), "served_households": int(served[c]),
                      "max_served_km": round(float(D[near == c, c].max()), 2), "seat": c == idx_seat})
    for i, v in enumerate(villages):
        v["market_town"] = int(villages[int(near[i])]["town"])
    return towns


def landings(g, sc, settlements: list[tuple[str, dict]], km, res_km) -> list[dict]:
    """每个村 / 镇 / 专业聚落旁一块泊场：reach 格内同岛、坡 ≤ landing_slope_max_deg、非水非崖的格，非田非林优先、近者优先。"""
    island_id = g["island_id"]
    H, W = island_id.shape
    slope = g["slope_deg"]
    water = (g["river"] > 0) | g["lake"]
    flat = (island_id >= 0) & ~water & ~g["cliff"] & (slope <= float(sc["landing_slope_max_deg"]))
    open_ = ~(g["arable"] > 0) & (g["landcover"] != LC_FOREST)
    reach = int(sc["landing_reach_cells"])
    out = []
    for kind, s in settlements:
        i, j = s["cell"]
        k = s["island"]
        r0, r1, c0, c1 = max(0, i - reach), min(H, i + reach + 1), max(0, j - reach), min(W, j + reach + 1)
        ok = flat[r0:r1, c0:c1] & (island_id[r0:r1, c0:c1] == k)
        if ok.any():
            ii, jj = np.where(ok)
            d = np.hypot(ii + r0 - i, jj + c0 - j)
            key = d - 3.0 * open_[r0:r1, c0:c1][ii, jj] + 0.2 * slope[r0:r1, c0:c1][ii, jj]
            t = int(np.argmin(key))
            cell, flat_ok = [int(ii[t] + r0), int(jj[t] + c0)], True
        else:
            # 附近没有平地：取窗内同岛非水非崖最缓的格（陡坡上的吊篮泊位）
            ok2 = (island_id[r0:r1, c0:c1] == k) & ~water[r0:r1, c0:c1] & ~g["cliff"][r0:r1, c0:c1]
            if not ok2.any():
                ok2 = island_id[r0:r1, c0:c1] == k
            ii, jj = np.where(ok2)
            t = int(np.argmin(slope[r0:r1, c0:c1][ii, jj]))
            cell, flat_ok = [int(ii[t] + r0), int(jj[t] + c0)], False
        out.append({"id": len(out) + 1, "kind": kind, "of": s.get("name"), "island": k, "cell": cell, "km": km(*cell),
                    "slope_deg": round(float(slope[cell[0], cell[1]]), 1), "flat": flat_ok,
                    "dist_km": round(math.hypot(cell[0] - i, cell[1] - j) * res_km, 2), "main": kind == "主泊场（仓场）"})
    return out


def clear_forest(g, sc, villages, hamlets, specials, res_km) -> dict:
    """村周开垦：林地在聚落半径内改成草坡（内圈）/ 灌丛（外圈，薪炭林）；同步地表占比与林木资源。返回摘要。"""
    island_id = g["island_id"]
    land = island_id >= 0
    H, W = land.shape
    res_m = res_km * 1000.0
    cover = g["landcover"]
    seed = np.zeros((H, W), dtype=bool)
    rad = np.zeros((H, W))
    for s in villages + hamlets + specials:
        hh = s["households"] + s.get("households_market", 0)
        r = float(np.clip(float(sc["clear_base_km"]) * math.sqrt(max(hh, 1) / 40.0), float(sc["clear_min_km"]), float(sc["clear_max_km"])))
        if s.get("town"):
            r = min(float(sc["clear_max_km"]) * float(sc["town_clear_mult"]), r * float(sc["town_clear_mult"]))
        i, j = s["cell"]
        seed[i, j] = True
        rad[i, j] = max(rad[i, j], r * 1000.0)
    forest0 = (cover == LC_FOREST) & land
    if not seed.any() or not forest0.any():
        return {"cleared_km2": 0.0, "forest_share_before": 0.0, "forest_share_after": 0.0}
    max_cells = int(math.ceil(float(rad.max()) / res_m)) + 1
    dist, src = nearest_propagate(seed, max_cells, res_m, within=land)
    s_ = np.where(src >= 0, src, 0).ravel()
    r_at = rad.ravel()[s_].reshape(H, W)
    same = island_id.ravel()[s_].reshape(H, W) == island_id
    hit = forest0 & (src >= 0) & same & (dist <= r_at)
    inner = hit & (dist <= float(sc["clear_inner_frac"]) * r_at)
    cover[inner] = LC_GRASS
    cover[hit & ~inner] = LC_SHRUB
    cell_km2 = res_km * res_km
    n_land = max(1, int(land.sum()))
    J = g["json"]
    from .output import LANDCOVER_CLASSES
    J["landcover"]["share"] = {LANDCOVER_CLASSES[i]: round(float(((cover == i) & land).sum()) / n_land, 4) for i in range(1, 12)}
    J["landcover"]["note_clearing"] = "林地在村 / 镇 / 专业聚落半径内已开垦：内圈草坡（牧场草场）、外圈灌丛（薪炭林）"
    # 林木资源：林场的格（patch_id）里被开垦的划掉；面积、代表格、主导栅格由 settle 随后调的 sync_resources 按 patch_id 重数
    #（旧做法按林木栅格重新分块：资源层 4 连通、这里 8 连通，斜角相连的几片林场被并成一块、每片都记整块的面积，#1165 林木合计 9,334 km² > 陆地）
    if g.get("resources") and "patch_id" in g:
        pid = g["patch_id"]
        tids = [d["id"] for d in g["resources"]["deposits"] if d["kind"] == "timber"]
        for d in g["resources"]["deposits"]:
            if d["kind"] == "timber":
                d["area_before_clearing_km2"] = d["area_km2"]
        pid[np.isin(pid, tids) & hit] = -1
    return {"cleared_km2": round(float(hit.sum()) * cell_km2, 2), "to_grass_km2": round(float(inner.sum()) * cell_km2, 2),
            "to_shrub_km2": round(float((hit & ~inner).sum()) * cell_km2, 2),
            "forest_share_before": round(float(forest0.sum()) / n_land, 4), "forest_share_after": round(float(((cover == LC_FOREST) & land).sum()) / n_land, 4)}


def village_workings(g, sc, villages, specials, km, res_km) -> dict:
    """村的采场（开垦之后，DESIGN-NOTES 四点二十一）：石料 / 黏土 / 砂砾就近取，窑村在自己的黏土区里开土坑，砂金区各一处淘金点。
    往 g["resources"]["workings"] 里加采场、给村记 workings；返回摘要。每村（按户数降序）先合用 working_share_km 内已有的同类采场，
    否则在 *_reach_km 内按 品位 × exp(−距离 / working_decay_km) 挑格——挑中的格离已有采场也在 working_share_km 内就合用那一处，不另开。
    格必须在该类的赋存区里、非耕、非水非崖；沉积类（黏土 / 砂砾 / 砂金）不上林，石料可在林坡上（那格改裸岩）。跨岛也行（飞船）。"""
    from .resources import FIELD_KINDS, WORK_ZH, _group_cells, open_working
    R = g.get("resources")
    if not R or "occ_lab" not in g:
        return {}
    island_id = g["island_id"]
    land = island_id >= 0
    H, W = land.shape
    water = (g["river"] > 0) | g["lake"]
    base_ok = land & ~water & ~g["cliff"] & (g["arable"] == 0)
    RF = g["res_field"]
    occ = R["occurrences"]
    works = R["workings"]
    decay = float(sc["working_decay_km"])
    share = float(sc["working_share_km"])
    order = sorted(villages, key=lambda v: (-v["households"], v["id"]))

    def new_work(kind, a, b, fk, lab):
        w = {"id": len(works), "kind": kind, "kind_zh": WORK_ZH[kind], "occurrence": int(lab[a, b]), "island": int(island_id[a, b]),
             "cell": [a, b], "km": km(a, b), "grade": round(float(fk[a, b]), 3), "villages": [], "special": None}
        works.append(w)
        if kind == "stone":
            open_working(g, a, b)
        return w

    def nearest(mine, i, j):
        best, bd = None, share
        for (a, b, w) in mine:
            d = math.hypot(a - i, b - j) * res_km
            if d <= bd:
                best, bd = w, d
        return best

    out = {}
    for kind, reach_key in (("stone", "quarry_reach_km"), ("clay", "clay_reach_km"), ("gravel", "gravel_reach_km")):
        lab = g["occ_lab"][kind]
        fk = RF[FIELD_KINDS.index(kind)].astype(np.float64) / 255.0
        elig = base_ok & (lab >= 0)
        if kind != "stone":
            elig &= g["landcover"] != LC_FOREST
        reach = float(sc[reach_key])
        rc_ = int(math.ceil(reach / res_km))
        mine = [(w["cell"][0], w["cell"][1], w) for w in works if w["kind"] == kind]
        users = [(v, "village") for v in order]
        if kind == "clay":
            users += [(x, "special") for x in specials if x["kind"] == "窑村"]
        served = 0
        for v, role in users:
            i, j = v["cell"]
            w = nearest(mine, i, j) if role == "village" else None
            if w is None:
                r0, r1, c0, c1 = max(0, i - rc_), min(H, i + rc_ + 1), max(0, j - rc_), min(W, j + rc_ + 1)
                ok = elig[r0:r1, c0:c1]
                if role == "special":            # 窑村只在自己的黏土区里挖
                    ok = ok & (lab[r0:r1, c0:c1] == v["occurrence"])
                if ok.any():
                    ii, jj = np.nonzero(ok)
                    d = np.hypot(ii + r0 - i, jj + c0 - j) * res_km
                    score = np.where(d <= reach, fk[r0:r1, c0:c1][ii, jj] * np.exp(-d / decay), -1.0)
                    t = int(np.argmax(score))
                    if score[t] > 0:
                        a, b = int(ii[t] + r0), int(jj[t] + c0)
                        w = nearest(mine, a, b) if role == "village" else None
                        if w is None:
                            w = new_work(kind, a, b, fk, lab)
                            mine.append((a, b, w))
            if w is None:
                continue
            if role == "village":
                w["villages"].append(v["id"])
                v.setdefault("workings", []).append(w["id"])
                served += 1
            else:
                w["special"] = v["id"]
                v["working"] = w["id"]
        out[WORK_ZH[kind]] = {"n": sum(1 for x in works if x["kind"] == kind), "villages_served": served,
                              "villages_share": round(served / max(1, len(villages)), 3), "reach_km": reach}
    # 淘金点：每个砂金区在非耕非林的滩地上挑品位最高的一格，挂到 placer_link_km 内最近的村（没有就是季节性的外来淘金客）
    lab = g["occ_lab"]["placer"]
    fk = RF[FIELD_KINDS.index("placer")].astype(np.float64) / 255.0
    elig = base_ok & (g["landcover"] != LC_FOREST) & (lab >= 0)
    vc = np.array([v["cell"] for v in villages], dtype=float) if villages else np.zeros((0, 2))
    n_pl = 0
    for oid, (ii, jj) in sorted(_group_cells(lab, elig).items()):
        t = int(np.argmax(fk[ii, jj]))
        a, b = int(ii[t]), int(jj[t])
        w = new_work("placer", a, b, fk, lab)
        n_pl += 1
        if vc.shape[0]:
            dv = np.hypot(vc[:, 0] - a, vc[:, 1] - b) * res_km
            q = int(np.argmin(dv))
            if dv[q] <= float(sc["placer_link_km"]):
                w["villages"].append(villages[q]["id"])
                villages[q].setdefault("workings", []).append(w["id"])
                continue
        w["note"] = "附近没有村：季节性的外来淘金客"
    out[WORK_ZH["placer"]] = {"n": n_pl, "occurrences": sum(1 for o in occ if o["kind"] == "placer")}
    return out
