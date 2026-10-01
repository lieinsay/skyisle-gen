"""第六节：岛群生成器的一致性校验（`skyisle island check <节点>`）。

IS-area / IS-surface / IS-arable / IS-river / IS-channel / IS-water / IS-season / IS-float / IS-link / IS-terr / IS-det / IS-iso 为硬项，IS-daily、IS-terr-gap、IS-valley 为软项
（IS-water 水账闭合、IS-valley 谷底宽合理是 C6 加的，PLAN-NATURE 第七节）；
资源 RES-site / RES-occ / RES-work / RES-geo 为硬项，RES-quarry 为软项；
聚落 SET-pop / SET-field / SET-site / SET-land / SET-town / SET-home / SET-farm / SET-use / SET-works / SET-market / SET-nature 为硬项，SET-water 为软项（PLAN-SETTLE 第六节；SET-dock 随码头取消，四点十八；
SET-farm / SET-use 是 P5 加的：宜垦 / 已垦 / 撂荒与定居门槛、没人住的岛有人用；SET-works 是 P6 加的：渠、塘、闸、圩田（P6b 加管它的村走得到、废弃的水利）；
SET-market 是 P7 加的：大泊场、航船、中转站；SET-town 按 P7 改成本岛走路 / 跨岛航船、没有「靠别的岛上的人撑起来又没有航船」的镇；
SET-nature 是 P6b 加的：原始地貌（landcover_natural）与人工改造（landuse）对得上）。
退出码：2 = 硬项失败；1 = 软项失败；0 = 全过。批跑（batch.py）复用 evaluate()。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent


def hash_products(out: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
            if p.suffix in (".npz", ".png", ".csv", ".json") and p.name != "island.json"}


def static_isolation() -> tuple[bool, list[str]]:
    """IS-iso：stages/、check、ninegrid、polity、culture 不得 import skyisle_gen.island。"""
    bad = []
    files = sorted((PKG / "stages").glob("*.py")) + [PKG / n for n in ("check.py", "ninegrid.py", "polity.py", "culture.py", "pipeline.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+\.*\s*(skyisle_gen\.)?island\b", text, re.M):
            bad.append(f.name)
    return not bad, bad


def evaluate(g: dict, out: Path, ctx=None, node: int | None = None, c: dict | None = None,
             det_hashes: tuple[dict, dict] | None = None, daily_years: int = 600) -> list[dict]:
    J = g["json"]
    C = g.get("climate")
    cons = J["constraints"]
    items = []

    def add(id_, name, value, thr, ok, hard=True, note=None):
        items.append({"id": id_, "name": name, "value": value, "threshold": thr, "pass": bool(ok), "hard": hard, "note": note})

    a = cons["area_km2"]
    e_area = abs(a["actual"] - a["target"]) / max(1e-9, a["target"])
    m = cons["main_area_km2"]
    e_main = abs(m["actual"] - m["target"]) / max(1e-9, m["target"])
    add("IS-area", "各岛面积之和 = area_km2；主岛 = main_area_km2（相对误差）", {"sum": round(e_area, 5), "main": round(e_main, 5)}, "< 0.02",
        e_area < 0.02 and e_main < 0.02)
    h = cons["height_m"]
    e_h = abs(h["actual"] - h["target"])
    tol_h = max(10.0, 0.02 * h["target"])
    pk = cons.get("peak_m", {})
    add("IS-surface", "主岛台面（陆地高程中位数）= height_m（④ 的岛上气温在这个高度）；峰高 / 起伏只报告",
        {"err_m": round(e_h, 1), "tol_m": round(tol_h, 1), "peak_m": pk.get("actual"), "relief_m": pk.get("relief_m"),
         "relief_target_m": pk.get("relief_target_m")}, "≤ max(10 m, 2%)", e_h <= tol_h)
    ar = cons["arable_frac"]
    e_ar = abs(ar.get("actual", -1) - ar["target"])
    add("IS-arable", "已垦（在种）/ 陆地 = arable_frac（P5：行星层的可耕率是已垦的额度）", round(e_ar, 5), "< 0.005", e_ar < 0.005)
    rv = cons["has_river"]
    small_ok = all(not i.get("has_perennial_river", False) for i in J["islands"][1:])
    add("IS-river", "主岛有常年河 ⇔ has_river；小岛只有溪涧", {"main": rv.get("actual"), "target": rv["target"], "small_islands_ok": small_ok},
        "相等", rv.get("actual") == rv["target"] and small_ok)
    if "w_ch_m" in g:
        rvm = g["river"] > 0
        ok_wd = bool(((g["w_ch_m"][rvm] > 0) & (g["d_ch_m"][rvm] > 0)).all()) if rvm.any() else True
        # 河道下切：横断面上两岸都高于河道——四组对边邻格（南北 / 东西 / 两条对角）里至少一组两格都是岸且都不低于河道（容 0.5 m）。
        # 不用「≤ 相邻岸格均值 / 最低者」：陡的河段每格落差十几米，下游那侧的岸本来就比这格河床低（那样陡河段只有 63–89%）
        from .grid import shift
        hh = np.where((g["island_id"] >= 0) & ~rvm & ~g["lake"], g["height"], np.nan)
        inc = np.zeros(hh.shape, dtype=bool)
        anyp = np.zeros(hh.shape, dtype=bool)
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a_ = shift(hh, di, dj, np.nan)
            b_ = shift(hh, -di, -dj, np.nan)
            both = np.isfinite(a_) & np.isfinite(b_)
            anyp |= both
            inc |= both & (a_ >= g["height"] - 0.5) & (b_ >= g["height"] - 0.5)
        has_nb = rvm & anyp
        below = float(inc[has_nb].mean()) if has_nb.any() else 1.0
        # C1：河道格的 height 是平岸水面；记成水面的（river_water）恰是河宽够一格的河道格
        water_ok = True
        if "river_water" in g and c is not None:
            thr_w = float(c.get("hydro", {}).get("river_water_min_cells", 1.0)) * g["res_km"] * 1000.0
            water_ok = bool((g["river_water"].astype(bool) == (rvm & (g["w_ch_m"] >= thr_w))).all())
        add("IS-channel", "常年河每格有河宽 / 水深；河道切在两岸之下（横断面两侧都不低于河道）的占比；记成水面的河道格恰是河宽够一格的（C1）",
            {"width_depth_ok": ok_wd, "below_banks": round(below, 4), "river_water_ok": water_ok}, "全有 / ≥ 0.95 / 是", ok_wd and below >= 0.95 and water_ok)
    if "runoff_acc" in g:
        items.append(_water_budget(g))
        if "wt" in g:
            items.append(_wt_check(g, c))
    if "rivernet" in g:
        items.append(_valley_check(g, c))
    R = g.get("resources")
    if R is not None:
        from .grid import binary_dilate
        from .resources import FIELD_KINDS, LITH_LAYERS, ORE_ORIGIN, RES_INDEX, rock_site_mask
        bad_site, bad_geo = [], []
        ages = {i["id"]: i["age_zh"] for i in J["islands"]}
        core_s = (J.get("hydro", {}).get("water") or {}).get("core_strength") or []
        core_min = float((c or {}).get("water", {}).get("hotspring_core_min", 0.0))
        water = _water_surface(g)
        fs_ids = [d["id"] for d in R["deposits"] if d["kind"] == "floatstone"]
        near_fs = binary_dilate(np.isin(g["patch_id"], fs_ids), 1) if fs_ids else np.zeros(water.shape, dtype=bool)
        RFg = g["res_field"]
        fk = {k: i for i, k in enumerate(R["fields"]["kinds"])}
        occ_by_id = {o["id"]: o for o in R["occurrences"]}
        lab_stone = g["occ_lab"]["stone"] if "occ_lab" in g else None
        for d in R["deposits"]:
            if d.get("cleared"):
                continue                        # 整片被村周开垦掉的林场
            i, j_ = d["cell"]
            on_cliff_ok = d["kind"] in ("cave", "floatstone", "guano")
            marked = (g["patch_id"][i, j_] == d["id"]) if d["form"] == "patch" else (g["resource"][i, j_] == RES_INDEX[d["kind"]])
            if g["island_id"][i, j_] != d["island"] or water[i, j_] or (g["cliff"][i, j_] and not on_cliff_ok) or not marked:
                bad_site.append(d["id"])
        # P3（没有火山）：不许有熔岩管 / 火山口；骨架空洞开在浮石露头旁；温泉、硫磺只在新岛（C4 起集水核的核山温泉除外）；溶洞只在老岛；石料岩性是三层之一；
        # 金属矿是海底带上来的那几种；盐泉在岩盐赋存上；贝壳化石在海相石灰岩的石料赋存区里
        for d in R["deposits"] + R["occurrences"]:
            a = ages.get(d["island"])
            sub = d.get("subtype")
            i, j_ = d["cell"]
            text = f"{sub or ''}{d.get('note') or ''}"
            if (("熔岩" in text or "火山" in text)
                    or (sub == "骨架空洞" and not near_fs[i, j_])
                    or (sub in ("溶洞", "落水洞", "地下河") and a != "老岛")
                    or (d["kind"] in ("sulfur", "hotspring") and a != "新岛" and sub != "核山温泉")
                    or (sub == "核山温泉" and (d["island"] >= len(core_s) or core_s[d["island"]] < core_min - 1e-3))
                    or (d["kind"] == "stone" and sub not in LITH_LAYERS)
                    or (d["kind"] == "ore" and sub not in ORE_ORIGIN)
                    or (d["kind"] == "saltspring" and RFg[fk["salt"]][i, j_] < int(round(R["fields"]["thr"]["salt"] * 255.0)) - 1)
                    or (d["kind"] == "fossil" and (lab_stone is None or lab_stone[i, j_] < 0
                                                   or occ_by_id[int(lab_stone[i, j_])]["subtype"] != LITH_LAYERS[0]))):
                bad_geo.append(f"{d['kind']}:{d['id']}")
        add("RES-site", "点与片在所属岛的陆地上、不在水面；除崖洞 / 浮石 / 鸟粪外不在崖缘；片的代表格在自己的 patch_id 上、点在主导栅格上",
            {"bad": bad_site[:10], "n": len(R["deposits"])}, "bad = 0", not bad_site)
        # 赋存：岩类的场只在岩类可放区（山地 / 高山 / 丘陵陡坡 / 裸岩，非耕、非湿地、非漫滩、非平地林）；每个赋存区的峰值格在本类场的阈值以上
        RF, thr = g["res_field"], R["fields"]["thr"]
        rock = rock_site_mask(g, g["terrain_zone"], float(R["fields"]["rock_hill_slope_deg"]))
        outside = {k: int(((RF[FIELD_KINDS.index(k)] > 0) & ~rock).sum()) for k in R["fields"]["rock_kinds"]}
        bad_occ = []
        for o in R["occurrences"]:
            i, j_ = o["cell"]
            if (g["island_id"][i, j_] != o["island"] or water[i, j_]
                    or int(RF[FIELD_KINDS.index(o["kind"])][i, j_]) < int(round(thr[o["kind"]] * 255.0)) - 1):
                bad_occ.append(o["id"])
        add("RES-occ", "岩类（金属矿 / 石料 / 硫磺）的赋存场不出岩类可放区；赋存区的峰值格在所属岛陆地、不在水面、品位 ≥ 阈值",
            {"rock_outside_cells": outside, "bad_occ": bad_occ[:10], "n": len(R["occurrences"])}, "全 0", not bad_occ and not any(outside.values()))
        # 采场：不上田（在种与撂荒的都不上；P5 之前是按额度画死的可耕地）、不上林（林坡上的岩类采场已改裸岩）、不在水上崖上，在本类场里且挂着同类的赋存区
        farm = ((g["cultivated"] > 0) | (g["fallow_years"] > 0)) if "cultivated" in g else (g["arable"] > 0)
        bad_w = []
        for w in R["workings"]:
            i, j_ = w["cell"]
            o = R["occurrences"][w["occurrence"]] if 0 <= w["occurrence"] < len(R["occurrences"]) else None
            if (o is None or o["kind"] != w["kind"] or g["island_id"][i, j_] != w["island"] or water[i, j_] or g["cliff"][i, j_]
                    or farm[i, j_] or g["landcover"][i, j_] == 4 or RF[FIELD_KINDS.index(w["kind"])][i, j_] == 0):
                bad_w.append(w["id"])
        add("RES-work", "采场（矿坑 / 硫磺坑 / 淘金点 / 采石场 / 土坑 / 采砂场）不在耕地、林地、水面、崖缘上，落在本类的赋存场里、挂着同类赋存区",
            {"bad": bad_w[:10], "n": len(R["workings"])}, "bad = 0", not bad_w)
        add("RES-geo", "资源与地质背景一致（没有火山：无熔岩管 / 火山口，骨架空洞开在浮石露头旁；温泉、硫磺只在新岛（核山温泉只在大核山上），溶洞只在老岛；"
                       "石料岩性为海相石灰岩 / 辉长岩 / 蛇纹岩，金属矿是海底带上来的那几种；盐泉在岩盐赋存上，贝壳化石在海相石灰岩里）",
            {"bad": bad_geo[:10]}, "bad = 0", not bad_geo)
        nq = sum(1 for o in R["occurrences"] if o["kind"] == "stone" and o["island"] == 0)
        add("RES-quarry", "主岛至少一处石料赋存区（有石料盖城）", nq, "≥ 1", nq >= 1, hard=False)
    if C is not None:
        an, mc = C["annual"], C["means_check"]
        errs = {"precip": abs(mc["precip_rel"] - an["precip_rel"]) / max(1e-9, an["precip_rel"]),
                "storm": abs(mc["storm"] - an["storm"]) / max(0.05, an["storm"]),
                "window": abs(mc["window"] - an["window"]) / max(1e-9, an["window"]),
                "temp_c": abs(mc["temp_c"] - an["temp_c"])}
        add("IS-season", "四季降水 / 风暴 / 窗口 / 温度的平均 = 年均值", {k: round(v, 5) for k, v in errs.items()}, "< 0.01（温度 < 0.05 °C）",
            errs["precip"] < 0.01 and errs["storm"] < 0.01 and errs["window"] < 0.01 and errs["temp_c"] < 0.05)
        if ctx is not None and "daily" in g:
            from .weather import multi_year_stats
            st = multi_year_stats(ctx, node, c, g, years=daily_years)
            add("IS-daily", f"{daily_years} 年逐日降水的平均 = 气候值（相对误差 < 5% 或 |z| < 3，z = 偏差 / 年际标准误）；雨日比例落在设定 ±0.05",
                {"annual_rel_err": st["annual_rel_err"], "annual_z": st["annual_z"], "annual_se_rel": st["annual_se_rel"],
                 "season_rel_err": st["precip_rel_err"], "wet_frac_err": st["wet_frac_err"]},
                "< 0.05 或 z < 3 / ≤ 0.05", (st["annual_rel_err"] < 0.05 or st["annual_z"] < 3.0) and max(st["wet_frac_err"]) <= 0.05, hard=False)
    # IS-terr：势力范围（C++ 的 territory.cpp）——陆地不越过与邻群的分界线；只是离线不到半道缝的算软项
    t = cons.get("territory")
    if t is not None and "violation_km" in t:
        v, half = float(t["violation_km"]), 0.5 * float(t.get("gap_km", 3.0))
        add("IS-terr", "陆地不越过与邻群的分界线（按等效半径分界、各退半道缝）；越界量 ≤ 0 全过，≤ 半道缝只是缝窄（软）",
            {"violation_km": v, "constrained": t.get("constrained"), "neighbours": t.get("neighbours")}, "≤ 0", v <= half)
        if 0.0 < v <= half:
            add("IS-terr-gap", "离分界线不到半道缝（两群之间的缝比 gap_km 窄）", {"violation_km": v}, "≤ 0", False, hard=False)
    # IS-float：浮高（四点二十八）——主岛不动；其余岛 δ 在 [−down_max, +up_max] 里、平移后岸缘 ≥ rim_floor_m。
    # 全行星的分布是否落在标定区间（|δ| 中位 300–600 m、p90 约 1 km、七成往上、与 Δ岛龄的秩相关 ≤ −0.4）由 `island floats` 查
    fl = [i.get("float_m") for i in J["islands"]]
    if fl and all(x is not None for x in fl):
        fc = (c or {}).get("float") or {}
        up, dn, floor = float(fc.get("up_max_m", 1500.0)), float(fc.get("down_max_m", 500.0)), float(fc.get("rim_floor_m", 20.0))
        rest = J["islands"][1:]
        vals = [float(i["float_m"]) for i in rest]
        out_rng = [i["id"] for i in rest if not (-dn <= float(i["float_m"]) <= up)]
        low = [i["id"] for i in rest if float(i["rim_m"]) < floor]
        add("IS-float", "浮高：主岛 δ = 0，其余岛 δ ∈ [−down_max, +up_max]、平移后岸缘 ≥ rim_floor_m（全行星分布对标定区间用 island floats 查）",
            {"main_m": fl[0], "n_up": sum(v > 0 for v in vals), "n_down": sum(v < 0 for v in vals), "max_m": max(vals, default=0.0),
             "min_m": min(vals, default=0.0), "rim_min_m": min((float(i["rim_m"]) for i in rest), default=None),
             "out_of_range": out_rng[:10], "rim_below_floor": low[:10]},
            f"主岛 0；δ ∈ [−{dn:.0f}, +{up:.0f}] m；岸缘 ≥ {floor:.0f} m", fl[0] == 0.0 and not out_rng and not low)
    # IS-link：短渡连通（P5 起没有索桥）
    from ..graph import weak_components
    n = len(J["islands"])
    src = np.array([e["a"] for e in J["links"]], dtype=np.int64)
    dst = np.array([e["b"] for e in J["links"]], dtype=np.int64)
    ncomp = int(weak_components(n, src, dst).max()) + 1 if n > 1 else 1
    no_bridge = all(e.get("kind") == "ferry" for e in J["links"]) and "channels" not in J
    add("IS-link", "群内任意两岛经短渡连通；没有索桥与导水槽（P5）", {"components": ncomp, "islands": n, "no_bridge": no_bridge}, "1 个分量、无索桥",
        ncomp == 1 and no_bridge)
    # ---- 聚落（PLAN-SETTLE 第六节）
    S = g.get("settle")
    if S is not None:
        import numpy as _np
        relay_hh = S.get("households_in_relays", 0)
        parts = (S["households_in_villages"] + S["households_in_hamlets"] + S.get("households_in_towns_market", 0) + S.get("households_in_specials", 0)
                 + relay_hh)
        hh_ok = parts == S["households"] == round(S["population"] / S["household_size"])
        nf_ok = S.get("households_in_towns_market", 0) + S.get("households_in_specials", 0) + relay_hh == S.get("nonfarm_households", 0) or not S["villages"]
        add("SET-pop", "村农户 + 散户 + 镇非农户 + 专业聚落户 + 中转站户 = 总户数 = 群人口 / 户均（只读 ⑨）；镇 + 专业 + 中转站 = 非农户",
            {"households": S["households"], "parts": parts, "population": S["population"], "nonfarm": S.get("nonfarm_households")}, "相等", hh_ok and nf_ok)
        arable_km2 = float((g["cultivated"] > 0).sum()) * (J["raster"]["res_m"] / 1000.0) ** 2
        f_sum = sum(f["area_km2"] for f in S["fields"])
        # 旱地田块有村的都够 8 户；P6b 的圩田田块（polder）挂在村上或自成圩村，几户也算；每个村都有自己的田
        every = all(f.get("households", 0) >= 8 for f in S["fields"] if f.get("village") and f["village"] > 0 and not f.get("polder")) and \
            all(1 <= v["field"] <= len(S["fields"]) for v in S["villages"])
        add("SET-field", "每村有田；Σ 田块 = 已垦", {"fields_km2": round(f_sum, 3), "arable_km2": round(arable_km2, 3), "villages_have_field": every}, "< 1%",
            abs(f_sum - arable_km2) <= 0.01 * max(arable_km2, 1e-9) and every)
        wsurf = _water_surface(g)
        bad = 0
        for v in S["villages"] + S.get("specials", []):
            i, j_ = v["cell"]
            if g["cliff"][i, j_] or wsurf[i, j_] or g["island_id"][i, j_] != v["island"]:
                bad += 1
        cells = _np.array([v["cell"] for v in S["villages"]], dtype=float)
        sep_share = 1.0
        if cells.shape[0] > 1:
            d = _np.sqrt(((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1)) * J["raster"]["res_m"] / 1000.0
            _np.fill_diagonal(d, 9e9)
            sep_share = float((d.min(axis=1) >= 1.0 - 1e-9).mean())
            bad += int((d.min() <= 0.05))
        add("SET-site", "村不在崖缘 / 水面 / 别的岛上、不同格；1 km 间距（软，报告占比）", {"bad": bad, "sep_1km_share": round(sep_share, 3)}, "bad = 0", bad == 0)
        # SET-land：飞船随处可停——每个村 / 镇 / 专业聚落有一块泊场，在同岛、不在水面崖缘上（P5 起没有索桥，也没有桥头）
        lands = {L_["id"]: L_ for L_ in S.get("landings", [])}
        miss, bad_l, steep = 0, 0, 0
        for v in S["villages"] + S.get("specials", []) + S.get("towns", []) + S.get("relays", []):
            L_ = lands.get(v.get("landing"))
            if L_ is None:
                miss += 1
                continue
            i, j_ = L_["cell"]
            if g["island_id"][i, j_] != v["island"] or wsurf[i, j_] or g["cliff"][i, j_]:
                bad_l += 1
            steep += not L_["flat"]
        add("SET-land", "每个村 / 镇 / 专业聚落 / 中转站有泊场（同岛、非水非崖；坡 > 4° 的只报告）；没有桥头",
            {"missing": miss, "bad": bad_l, "not_flat": steep, "bridgeheads": len(S.get("bridgeheads", []))},
            "无缺", miss == 0 and bad_l == 0 and "bridgeheads" not in S)
        T_ = S.get("towns", [])
        vt = [v for v in S["villages"] if v.get("market_town")]
        seat_town = any(t_.get("seat") for t_ in T_)
        tby = {t_["id"]: t_ for t_ in T_}
        lby = {ln["id"]: ln for ln in S.get("boat_lines", [])}
        # P7：走路的村与它的镇同岛；航船村有一条通到它那个镇的航船；有别的岛上的人来赶集的镇都有航船（没有「靠别的岛上的人撑起来、又没有航船」的镇）
        bad_mode = [v["id"] for v in vt if "market_mode" in v and (
            (v["market_mode"] == "步行" and tby[v["market_town"]]["island"] != v["island"])
            or (v["market_mode"] == "航船" and (v.get("boat_line") not in lby or lby[v["boat_line"]]["town"] != v["market_town"]
                                              or v["id"] not in lby[v["boat_line"]]["stops"])))]
        no_line = [t_["id"] for t_ in T_ if t_.get("served_other_islands_households", 0) > 0 and not t_.get("n_lines", 0)]
        add("SET-town", "有村就有镇、邑治是镇、每个村归一个镇；走路赶集的村与镇同岛，搭航船的村有通到它的镇的航船；有别的岛上的人来赶集的镇都有航船",
            {"towns": len(T_), "villages": len(S["villages"]), "assigned": len(vt), "seat_is_town": seat_town, "bad_mode": bad_mode[:10],
             "towns_without_line": no_line[:10]},
            "全满足", not S["villages"] or (len(T_) >= 1 and seat_town and len(vt) == len(S["villages"]) and not bad_mode and not no_line))
        if "harbors" in S:
            items.append(_market_check(g, S, c))
        # 没有村的群（荒漠里只有散户与蓄水池，A3 起有真沙漠）不算：占比按村算，0 个村时没有意义
        add("SET-water", "村 1 km 内有水源的占比（没有村的群不查）", S["water_ok_share"] if S["villages"] else None, "≥ 0.8",
            not S["villages"] or S["water_ok_share"] >= 0.8, hard=False)
        if "farmland" in S:
            items.extend(_farm_checks(g, S, c))
        kinds = [h["kind"] for h in S["home_candidates"]]
        # 没有村的群没有邑治，「邑治旁」一类出不来（荒漠群只有散户，A3 起有真沙漠）：只要求 ≤ 3 个、类型各异
        add("SET-home", "主家候选 2–3 个、类型各异（没有村的群 0–3 个）", kinds, "2–3 个",
            (2 <= len(kinds) if S["villages"] else True) and len(kinds) <= 3 and len(kinds) == len(set(kinds)))
    if det_hashes is not None:
        h1, h2 = det_hashes
        diff = sorted(k for k in set(h1) | set(h2) if h1.get(k) != h2.get(k))
        add("IS-det", "同输入重跑两次，产物哈希一致", {"differ": diff}, "无差异", not diff)
    ok, bad = static_isolation()
    add("IS-iso", "stages/ 不 import skyisle_gen.island（第三层不回灌）", {"offenders": bad}, "无", ok)
    return items


def _farm_checks(g: dict, S: dict, c: dict | None) -> list[dict]:
    """SET-farm / SET-use（P5）。"""
    out = []
    J = g["json"]
    iid = g["island_id"]
    cv, ct, fy = g["cultivable"], g["cultivated"], g["fallow_years"]
    cell_km2 = (J["raster"]["res_m"] / 1000.0) ** 2
    sc = (c or {}).get("settle") or {}
    min_hh = float(S["farmland"].get("settle_min_hh", sc.get("settle_min_hh", 0)))
    lph = float(S["land_per_household_km2"])
    pol = g["polder_id"] > 0 if "polder_id" in g else np.zeros_like(ct, dtype=bool)      # P6：圩田是排干的湿地，不在宜垦里
    bad_sub = int(((ct > 0) & (cv == 0) & ~pol).sum())
    bad_ter = int(((ct > 0) & (ct != cv) & ~pol).sum())
    bad_pol = int((pol & ((ct != 1) | (cv > 0))).sum())
    bad_fal = int(((fy > 0) & ((cv == 0) | (ct > 0))).sum())
    n = len(J["islands"])
    ct_isl = np.bincount(iid[(iid >= 0) & (ct > 0)], minlength=n) * cell_km2
    farm_isl = {v["island"] for v in S["villages"]} | {h["island"] for h in S["hamlets"]}
    forced = float(S["farmland"].get("forced_km2", 0.0)) > 0
    small = [k for k in sorted(farm_isl) if ct_isl[k] < min_hh * lph * (1 - 1e-9) - 1e-9] if not forced else []
    # 强填（留下的宜垦片不够额度：干群、冷群里宜垦几乎只剩额度那点地，额度只好补进过不了门槛的碎片）时不查门槛与「有田就有人」：
    # 分到一小块田的岛按面积分户可能分不到一户（邻岛的人来种）
    no_field = [k for k in range(n) if ct_isl[k] > 0 and k not in farm_isl] if not forced else []
    vil_isl = {v["island"] for v in S["villages"]}
    no_floor = [k for k in S["farmland"].get("floor_islands", []) if k not in vil_isl]     # 大岛保底（用户 09-29 定）：保底的岛都有村
    bad_ruin = [r["id"] for r in S.get("ruins", []) if r["abandoned_years"] < 1 or iid[r["cell"][0], r["cell"][1]] != r["island"]
                or g["cliff"][r["cell"][0], r["cell"][1]] or _water_surface(g)[r["cell"][0], r["cell"][1]]]
    out.append({"id": "SET-farm", "name": "已垦 ⊂ 宜垦 ∪ 圩田（梯田标记一致；圩田是排干的湿地、记 1）、撂荒 ⊂ 宜垦且不与已垦重叠；有农户的岛已垦都够 settle_min_hh 户（定居门槛）、"
                "有已垦的岛都有村或散户（这两条强填时不查）；大岛保底的岛都有村；废村在自己的岛上、撤空了 ≥ 1 年",
                "value": {"not_cultivable": bad_sub, "terrace_mismatch": bad_ter, "bad_polder": bad_pol, "bad_fallow": bad_fal, "below_threshold": small[:10],
                          "fields_without_people": no_field[:10], "bad_ruins": bad_ruin[:10], "forced_km2": S["farmland"].get("forced_km2", 0.0),
                          "floor_without_village": no_floor[:10], "floor_islands": len(S["farmland"].get("floor_islands", [])),
                          "cultivable_km2": S["farmland"]["cultivable_km2"], "cultivated_km2": S["farmland"]["cultivated_km2"],
                          "polder_km2": S["farmland"].get("polder_km2", 0.0)},
                "threshold": "全 0", "pass": bool(not (bad_sub or bad_ter or bad_pol or bad_fal or small or no_field or bad_ruin or no_floor)), "hard": True,
                "note": None})
    if "waterworks" in S:
        out.append(_works_check(g, S))
    if "landuse" in g and "landcover_natural" in g:
        out.append(_nature_check(g, S))
    vids = {v["id"] for v in S["villages"]}
    bad_occ = [x["name"] for x in S["specials"] if x.get("occupancy") not in ("常住", "工棚", "季节住")
               or (x["occupancy"] != "常住" and S["villages"] and x.get("home_village") not in vids)
               or (x["occupancy"] == "工棚" and x["island"] in farm_isl)]
    res_isl = farm_isl | {x["island"] for x in S["specials"] if x.get("occupancy") == "常住"}
    bad_use = [u["id"] for u in S.get("uses", []) if u["island"] in res_isl or u["island"] == 0 or iid[u["cell"][0], u["cell"][1]] != u["island"]]
    T = S.get("land_tenure", [])
    bad_ten = [t["island"] for t in T if t["owner"] not in ("村", "大户", "官用", "官荒") or (t["owner"] == "村") != (t["island"] in farm_isl)]
    out.append({"id": "SET-use", "name": "专业聚落的住法是 常住 / 工棚 / 季节住，工棚与季节住有 home_village、工棚不在有村的岛上；有人用的岛（放牧 / 庙 / 墓岛；"
                "烽火台 P7 起归中转站）都没人常住；每岛一条荒地归属（有农户的岛归村，其余归大户、官用（只有中转站）或官荒）",
                "value": {"bad_occupancy": bad_occ[:10], "bad_uses": bad_use[:10], "bad_tenure": bad_ten[:10], "tenure_rows": len(T), "islands": n},
                "threshold": "全 0、每岛一条", "pass": bool(not (bad_occ or bad_use or bad_ten) and len(T) == n), "hard": True, "note": None})
    return out


def _market_check(g: dict, S: dict, c: dict | None) -> dict:
    """SET-market（P7）：大泊场在本岛的缓坡平地上（非水非崖）；镇挨着的大泊场同岛、在 town_harbor_km 内；航船线的每个村都是搭航船去这条线的镇的，
    每个搭航船的村恰在一条线上；中转站的站址与瞭望处在本岛陆地上、不上水面崖缘，户 = 各角色之和，常住的有户、轮班的没有，不在同一座岛上摆两处。"""
    iid, cliff, lake, river = g["island_id"], g["cliff"], g["lake"], _water_surface(g)   # river：记成水的河道格（C1）
    mc = (c or {}).get("market") or {}
    smax = float(mc.get("harbor_slope_max_deg", 4.0))
    th_km = float(mc.get("town_harbor_km", 2.0))

    def bad_cell(cell, k):
        i, j = cell
        return iid[i, j] != k or cliff[i, j] or lake[i, j] or river[i, j]
    H = S["harbors"]
    bad_h = [h["id"] for h in H if bad_cell(h["cell"], h["island"]) or not (g["slope_deg"][h["cell"][0], h["cell"][1]] <= np.float32(smax))]
    bad_t = []
    for t in S.get("towns", []):
        if t.get("harbor") is None:
            continue
        h = H[t["harbor"] - 1]
        d = float(np.hypot(h["km"][0] - t["km"][0], h["km"][1] - t["km"][1]))
        if h["island"] != t["island"] or d > th_km + 1e-6 or h.get("town") != t["id"]:
            bad_t.append(t["id"])
    vby = {v["id"]: v for v in S["villages"]}
    seen = {}
    bad_l = []
    for ln in S.get("boat_lines", []):
        for vid in ln["stops"]:
            v = vby.get(vid)
            seen[vid] = seen.get(vid, 0) + 1
            if v is None or v.get("market_mode") != "航船" or v.get("market_town") != ln["town"]:
                bad_l.append(ln["id"])
    boat = [v["id"] for v in S["villages"] if v.get("market_mode") == "航船"]
    bad_cover = [vid for vid in boat if seen.get(vid, 0) != 1]
    bad_r = []
    isl_seen = set()
    for r in S.get("relays", []):
        hh = sum(r["roles"].values())
        if (bad_cell(r["cell"], r["island"]) or iid[r["lookout_cell"][0], r["lookout_cell"][1]] != r["island"] or hh != r["households"]
                or (r["occupancy"] == "常住") != (r["households"] > 0) or r["island"] in isl_seen or not r["functions"]):
            bad_r.append(r["id"])
        isl_seen.add(r["island"])
    return {"id": "SET-market", "name": "大泊场在本岛缓坡平地上；镇挨着的大泊场同岛、够近；航船线只停搭航船去这个镇的村、每个航船村恰在一条线上；"
            "中转站在本岛陆地上、户 = 各角色之和、常住有户轮班没户、一岛至多一处（P7）",
            "value": {"harbors": len(H), "bad_harbors": bad_h[:10], "bad_town_harbor": bad_t[:10], "lines": len(S.get("boat_lines", [])),
                      "bad_lines": sorted(set(bad_l))[:10], "boat_villages": len(boat), "bad_cover": bad_cover[:10],
                      "relays": len(S.get("relays", [])), "bad_relays": bad_r[:10]},
            "threshold": "全 0", "pass": bool(not (bad_h or bad_t or bad_l or bad_cover or bad_r)), "hard": True, "note": None}


def _works_check(g: dict, S: dict) -> dict:
    """SET-works（P6 水利）：渠首在常年河 / 溪涧上；谷口的渠除渠首外都走在本岛陆地上（不上崖缘、湖、常年河）；塘在陆地上、不在水面崖缘上；
    闸在本岛陆地上；圩号栅格与圩的格数一致；一块田只归一处渠首；一个村至多一口塘。"""
    WK = S["waterworks"]
    iid, cliff, lake, river, stream = g["island_id"], g["cliff"], g["lake"], g["river"] > 0, g["stream"] > 0
    H, W = iid.shape
    bad_head = [h["id"] for h in WK["heads"] if not (river[h["cell"][0], h["cell"][1]] or stream[h["cell"][0], h["cell"][1]])
                or iid[h["cell"][0], h["cell"][1]] != h["island"]]
    bad_canal = []
    for c in WK["canals"]:
        if c["kind"] in ("干渠", "支渠"):
            for r, q in c["pts"][1:] if c["kind"] == "干渠" else c["pts"]:
                i, j = int(r), int(q)
                if iid[i, j] != c["island"] or cliff[i, j] or lake[i, j] or river[i, j]:
                    bad_canal.append(c["id"])
                    break
        elif not all(0 <= r <= H and 0 <= q <= W for r, q in c["pts"]):
            bad_canal.append(c["id"])
    bad_pond = [p["id"] for p in WK["ponds"] if iid[p["cell"][0], p["cell"][1]] != p["island"] or cliff[p["cell"][0], p["cell"][1]]
                or lake[p["cell"][0], p["cell"][1]] or river[p["cell"][0], p["cell"][1]]]
    bad_sluice = [x["id"] for x in WK["sluices"] if iid[x["cell"][0], x["cell"][1]] != x["island"]]
    pid = g["polder_id"] if "polder_id" in g else np.zeros((H, W), dtype=np.int32)
    cnt = np.bincount(pid.ravel(), minlength=len(WK["polders"]) + 1)
    bad_polder = [p["id"] for p in WK["polders"] if int(cnt[p["id"]]) != p["cells"]]
    if int((pid > 0).sum()) != sum(p["cells"] for p in WK["polders"]):
        bad_polder.append("raster")
    seen, dup = set(), []
    for h in WK["heads"]:
        for f in h["fields"]:
            if f in seen or not (1 <= f <= len(S["fields"])):
                dup.append(f)
            seen.add(f)
    vp = [p["village"] for p in WK["ponds"] if p.get("village") is not None and p["kind"] in ("村塘", "山塘") and not p.get("abandoned")]
    dup_pond = len(vp) - len(set(vp))
    far, unmanaged, bad_aband, bad_cmd = _works_manage(g, S, pid)
    # 四点四十：邑级大堰——堰在本岛的常年河上、邑管、灌区里在种的不超过规划的、用水的村都在、大堰的渠都记着堰号；村级的关了（village_works = false）就不该有村的渠首和塘
    vids = {v["id"] for v in S["villages"]}
    BW = WK.get("big_works", [])
    wids = {w["id"] for w in BW}
    bad_big = [w["id"] for w in BW if not river[w["cell"][0], w["cell"][1]] or iid[w["cell"][0], w["cell"][1]] != w["island"] or w["maintainer"] != "邑"
               or w["served_km2"] > w["planned_km2"] + 1e-9 or any(t["village"] not in vids for t in w["turnouts"])
               or [t["village"] for t in w["turnouts"]] != w["villages"]]
    bad_big += [f"渠{c['id']}" for c in WK["canals"] if "work" in c and (c["work"] not in wids or c.get("maintainer") != "邑")]
    vw_off = WK["summary"].get("village_works") is False and bool(WK["heads"] or WK["ponds"])
    ok = not (bad_head or bad_canal or bad_pond or bad_sluice or bad_polder or dup or dup_pond or far or unmanaged or bad_aband or bad_cmd or bad_big or vw_off)
    ws = WK["summary"]
    return {"id": "SET-works", "name": "水利（P6）：渠首在常年河 / 溪涧上；谷口的渠除渠首外走在本岛陆地上（不上崖缘、湖、常年河）；塘、闸在本岛陆地上；"
            "圩号栅格与圩的格数一致；一块田只归一处渠首；一个村至多一口村塘 / 山塘；P6b：每处都有管它的村（渠首、渠的每一点、塘、闸、圩的每一格（整格）"
            "都在它走得到的范围内），渠首只灌那个村的田；废弃的（abandoned）没有管它的村、记着废村与撤空的年头；四点四十：邑级大堰在本岛常年河上、邑管、"
            "在种的 ≤ 规划的灌区、分水口的村都在，大堰的渠记着堰号；村级水利关了就没有村的渠首和塘",
            "value": {"bad_heads": bad_head[:10], "bad_canals": bad_canal[:10], "bad_ponds": bad_pond[:10], "bad_sluices": bad_sluice[:10],
                      "bad_polders": bad_polder[:10], "field_in_two_heads": dup[:10], "village_two_ponds": dup_pond,
                      "too_far": far[:10], "unmanaged": unmanaged[:10], "bad_abandoned": bad_aband[:10], "head_field_not_village": bad_cmd[:10],
                      "bad_big_works": bad_big[:10], "village_works_left": vw_off, "big_works": len(BW),
                      "heads": ws["n_heads"], "canal_km": ws["canal_km"], "ponds": ws["n_ponds"], "sluices": ws["n_sluices"], "polder_km2": ws["polder_km2"],
                      "abandoned": ws.get("abandoned")},
            "threshold": "全 0", "pass": bool(ok), "hard": True, "note": None}


def _works_manage(g: dict, S: dict, pid: np.ndarray) -> tuple[list, list, list, list]:
    """P6b（L30）：每处水利管它的村在不在、够不够得着（走得到 = 村的格心到那一点 ≤ manage_walk_km；渠的点是格心或格边，圩量到每一格最远的那个角）；
    渠首灌的田是不是那个村的；废弃的有没有废村与年头。P6b 之前的产物（summary 没有 manage_walk_km）不查。"""
    WK = S["waterworks"]
    ws = WK["summary"]
    if "manage_walk_km" not in ws:
        return [], [], [], []
    res_km = float(g["res_km"])
    walk = float(ws["manage_walk_km"]) / res_km + 1e-9
    V = {v["id"]: v for v in S["villages"]}
    R = {r["id"]: r for r in S.get("ruins", [])}
    far, unmanaged, bad_aband, bad_cmd = [], [], [], []

    def dist(vid, pts):
        v = V[vid]
        P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return float(np.hypot(P[:, 0] - (v["cell"][0] + 0.5), P[:, 1] - (v["cell"][1] + 0.5)).max())

    def one(kind, x, pts):
        if x.get("abandoned"):
            if x.get("village") is not None or x.get("ruin") not in R or int(x.get("abandoned_years", 0)) < 1:
                bad_aband.append(f"{kind}{x['id']}")
            return
        vid = x.get("village")
        if vid not in V:
            unmanaged.append(f"{kind}{x['id']}")
        elif dist(vid, pts) > walk:
            far.append(f"{kind}{x['id']}")
    for x in WK["heads"]:
        one("渠首", x, [[x["cell"][0] + 0.5, x["cell"][1] + 0.5]])
        if not x.get("abandoned") and x.get("village") in V and x["fields"] != [V[x["village"]]["field"]]:
            bad_cmd.append(x["id"])
    for x in WK["canals"]:
        if "work" in x:                             # 四点四十：大堰的渠邑管（岁修按用水的村出工），不按村走得到查
            continue
        one("渠", x, x["pts"])
    for x in WK["ponds"]:
        one("塘", x, [[x["cell"][0] + 0.5, x["cell"][1] + 0.5]])
    for x in WK["sluices"]:
        one("闸", x, [[x["cell"][0] + 0.5, x["cell"][1] + 0.5]])
    if WK["polders"]:
        W = pid.shape[1]
        flat = pid.ravel()
        idx = np.flatnonzero(flat > 0)
        o = idx[np.argsort(flat[idx], kind="stable")]
        b = np.searchsorted(flat[o], np.arange(1, len(WK["polders"]) + 2))
        for p in WK["polders"]:
            cells = o[b[p["id"] - 1]:b[p["id"]]]
            ii, jj = cells // W, cells % W
            corners = np.concatenate([np.stack([ii + a, jj + c], axis=1) for a in (0, 1) for c in (0, 1)])
            one("圩", p, corners)
    return far, unmanaged, bad_aband, bad_cmd


def _water_budget(g: dict) -> dict:
    """IS-water（C6 水账闭合）：陆地的径流（雨 × Budyko 径流系数 + 凝结水）= 各出口（崖边、湖、平地的汇点）的径流累计；
    地下水补给 = 各出口的补给累计；每格凝结水 ≤ 局地雨（spec 13 第八节：最多 1 倍），雨产的径流 ≤ 局地雨；逐日径流指数年均 = 1。"""
    land = g["island_id"] >= 0
    cell = g["res_km"] ** 2
    outlet = land & (g["recv_i"] < 0)
    run = g["runoff_mm"].astype(np.float64)
    tot = float(run[land].sum()) * cell
    out_ = float(g["runoff_acc"][outlet].sum())
    e_run = abs(out_ - tot) / max(1e-9, tot)
    val = {"runoff_mm_km2": round(tot, 1), "outflow_mm_km2": round(out_, 1), "err": round(e_run, 7)}
    ok = e_run < 1e-4
    rain = g["rain_mm"].astype(np.float64)
    if "condense_mm" in g:
        cond = g["condense_mm"].astype(np.float64)
        over_c = int((land & (cond > rain + 1e-3)).sum())
        over_r = int((land & (run - cond > rain + 1e-3)).sum())
        val.update({"condense_over_rain": over_c, "rain_runoff_over_rain": over_r})
        ok = ok and over_c == 0 and over_r == 0
    if "recharge_acc" in g:
        rtot = float(g["recharge_mm"].astype(np.float64)[land].sum()) * cell
        # B 起补给按水位面的流向累计：补水口是水位面的出口（出岛 / 无下游），不是地表的 recv_i
        rec_outlet = land & (g["wt_outlet"] > 0) if "wt_outlet" in g else outlet
        e_rec = abs(float(g["recharge_acc"][rec_outlet].sum()) - rtot) / max(1e-9, rtot)
        val["recharge_err"] = round(e_rec, 7)
        val["recharge_outlets"] = int(rec_outlet.sum())
        ok = ok and e_rec < 1e-4
    if "rivernet" in g:
        e_idx = max((abs(float(np.mean(b["index"])) - 1.0) for b in g["rivernet"]["basins"] if len(b["index"])), default=0.0)
        val["index_mean_err"] = round(e_idx, 7)
        ok = ok and e_idx < 1e-6
    return {"id": "IS-water", "name": "水账闭合（C6）：陆地径流 = 出口的径流累计，补给 = 出口的补给累计；凝结水、雨产的径流都不超过局地雨；逐日径流指数年均 = 1",
            "value": val, "threshold": "相对误差 < 1e−4（径流权重取整到 1/16 mm、float32）、超出 0 格", "pass": bool(ok), "hard": True, "note": None}


def _wt_check(g: dict, c: dict | None) -> dict:
    """IS-wt（B+A，四点四十九）：水位面不高于地表（埋深 ≥ 0）；排水口（河道 / 溪涧 / 湖 / 岸缘）的水位贴在地表上；
    「泉 / 崖瀑」的段含水层够厚（≥ spring_min_aquifer_m）、出水量 ≥ 本岛的记录门槛。"""
    land = g["island_id"] >= 0
    res_m = g["res_km"] * 1000.0
    dep = g["height"].astype(np.float64) - g["wt"].astype(np.float64)
    neg = int((land & (dep < -0.05)).sum())
    drain = land & ((g["river"] > 0) | (g["stream"] > 0) | (g["lake"] > 0) | (g["coast_dist_m"] < res_m))
    dd = dep[drain]
    drain_p50 = float(np.median(dd)) if dd.size else 0.0
    drain_p90 = float(np.percentile(dd, 90)) if dd.size else 0.0
    amin = float((c or {}).get("water", {}).get("spring_min_aquifer_m", 20.0))
    qmin = float((c or {}).get("water", {}).get("springline_min_ls", 0.5))
    sl = g.get("springline", [])
    bad_aq = sum(1 for s in sl if int(s.get("kind", 0)) >= 1 and float(s.get("aquifer_m", 0.0)) + 1e-6 < amin)
    bad_q = sum(1 for s in sl if float(s["q_ls"]) + 1e-6 < qmin)
    kinds = {k: sum(1 for s in sl if int(s.get("kind", 0)) == k) for k in (0, 1, 2)}
    km = {k: round(sum(float(s["length_km"]) for s in sl if int(s.get("kind", 0)) == k), 1) for k in (0, 1, 2)}
    return {"id": "IS-wt", "name": "水位面（B）：不高于地表、排水口贴地表；泉 / 崖瀑的段含水层够厚、出水过门槛（A）",
            "value": {"depth_below_zero_cells": neg, "drain_depth_p50_m": round(drain_p50, 2), "drain_depth_p90_m": round(drain_p90, 2),
                      "springs_thin_aquifer": bad_aq, "springs_under_qmin": bad_q,
                      "kind_n": {"弥散渗出": kinds[0], "泉": kinds[1], "崖瀑": kinds[2]},
                      "kind_km": {"弥散渗出": km[0], "泉": km[1], "崖瀑": km[2]}},
            "threshold": "埋深 < −0.05 m 的格 = 0、排水口埋深中位 ≤ 0.5 m 且 p90 ≤ 2 m、含水层不够的泉段 = 0、出水不足的段 = 0",
            "pass": bool(neg == 0 and drain_p50 <= 0.5 and drain_p90 <= 2.0 and bad_aq == 0 and bad_q == 0), "hard": True, "note": None}


def _valley_check(g: dict, c: dict | None) -> dict:
    """IS-valley（C6 谷底宽合理，软）：常年河量出的左右谷底宽不超过 C2 的谷底宽；开阔段谷底宽 / 平岸河宽的中位在 10–200 倍
    （地球上不受限的河谷漫滩宽多为河宽的十几到上百倍）；谷底宽不超过上限。"""
    N = g["rivernet"]
    lvl = N["level"]
    rv = lvl >= 1
    res_m = g["res_km"] * 1000.0
    cells = N["cell"].astype(np.int64)
    W = g["height"].shape[1]
    fw = g["floor_w_m"].reshape(-1)[cells].astype(np.float64)
    w = N["w"].astype(np.float64)
    excess = int((rv & (N["fp_l"] + N["fp_r"] > np.maximum(0.0, fw - w) + res_m)).sum())
    vmax = float((c or {}).get("hydro", {}).get("valley_floor_max_m", 3000.0))
    over = int((rv & (fw > vmax + 1e-6)).sum())
    op = rv & (N["confine"] == 3)
    ratio = float(np.median(fw[op] / np.maximum(w[op], 0.1))) if op.any() else float("nan")
    ok = excess == 0 and over == 0 and (not op.any() or 10.0 <= ratio <= 200.0)
    return {"id": "IS-valley", "name": "谷底宽合理（C6）：量出的左右谷底宽不超过谷底宽，谷底宽不过上限，开阔段谷底宽 / 平岸河宽的中位在 10–200",
            "value": {"fp_excess": excess, "over_max": over, "open_ratio_p50": round(ratio, 1) if op.any() else None, "n_river_pts": int(rv.sum())},
            "threshold": "0 / 0 / 10–200", "pass": bool(ok), "hard": False, "note": None}


def _water_surface(g: dict) -> np.ndarray:
    """水面：湖与记成水的河道格（C1：河宽过一格的，地表 = 河）。更窄的河道格在岸上（河槽由中心线 + 宽表达），点、村、泊场可以在上面；
    旧产物（C1 之前）所有河道格的地表都是河，同一个式子还是旧口径。"""
    return g["lake"] | (g["landcover"] == 10)


def _nature_check(g: dict, S: dict) -> dict:
    """SET-nature（P6b，L31：原始地貌和人工地貌分开记）：landcover_natural 是没有人以前的地表、landuse 是人工改造——
    圩田那格原来是湿地；别的田原来不是湿地、河、湖、崖缘；地表变了的格都说得清（landuse > 0）；landuse 与已垦 / 圩田 / 撂荒对得上；
    没有农户（村、散户）的岛上没有田（强填的干群不查，同 SET-farm）、原始湿地 = 现状湿地（没人去排）。"""
    from .waterworks import LANDUSE_CLASSES
    iid = g["island_id"]
    land = iid >= 0
    nat, lu, lc = g["landcover_natural"], g["landuse"], g["landcover"]
    ct, fy = g["cultivated"], g["fallow_years"]
    pol = g["polder_id"] > 0 if "polder_id" in g else np.zeros_like(land)
    LC_WET_, LC_CLIFF, LC_RIVER, LC_LAKE = 9, 1, 10, 11
    bad_pol = int((pol & (nat != LC_WET_)).sum())
    bad_field = int(((ct > 0) & ~pol & np.isin(nat, [LC_WET_, LC_CLIFF, LC_RIVER, LC_LAKE])).sum())
    unexplained = int((land & (lc != nat) & (lu == 0)).sum())
    bad_code = int(((pol != (lu == 4)) | ((ct > 0) & ~pol & ~np.isin(lu, [1, 2, 3])) | ((fy > 0) & (lu != 5)) | ((lu == 3) & (ct == 0))
                    | (lu >= len(LANDUSE_CLASSES)) | (~land & (lu > 0))).sum())
    farm = {v["island"] for v in S["villages"]} | {h["island"] for h in S["hamlets"]}
    n = len(g["json"]["islands"])
    nob = np.array([k not in farm for k in range(n)], dtype=bool)
    nb = land & nob[np.where(land, iid, 0)]
    nb_field = int((nb & (ct > 0)).sum())
    nb_wet = int((nb & ((nat == LC_WET_) != (lc == LC_WET_))).sum())
    forced = float(S["farmland"].get("forced_km2", 0.0)) > 0      # 强填的干群：分到一小块田的岛可能分不到户（邻岛的人来种，同 SET-farm），田不查
    ok = not (bad_pol or bad_field or unexplained or bad_code or (nb_field and not forced) or nb_wet)
    cell_km2 = float(g["res_km"]) ** 2
    return {"id": "SET-nature", "name": "原始地貌与人工地貌（P6b）：圩田原来是湿地、别的田原来不是湿地 / 水 / 崖缘；地表变了的格都有人工改造码；"
            "改造码与已垦 / 圩田 / 撂荒对得上；没有农户的岛上没有田、湿地原样",
            "value": {"polder_not_wet": bad_pol, "field_bad_natural": bad_field, "unexplained_change": unexplained, "bad_landuse_code": bad_code,
                      "field_on_empty_island": nb_field, "forced_km2": S["farmland"].get("forced_km2", 0.0), "wet_changed_on_empty_island": nb_wet,
                      "wet_natural_km2": round(float((land & (nat == LC_WET_)).sum()) * cell_km2, 3),
                      "wet_now_km2": round(float((land & (lc == LC_WET_)).sum()) * cell_km2, 3),
                      "changed_km2": round(float((land & (lc != nat)).sum()) * cell_km2, 3)},
            "threshold": "全 0", "pass": bool(ok), "hard": True, "note": None}


def exit_code(items: list[dict]) -> int:
    if any(not i["pass"] and i["hard"] for i in items):
        return 2
    if any(not i["pass"] for i in items):
        return 1
    return 0


def print_report(items: list[dict], node: int) -> None:
    print(f"== 岛群 #{node} 一致性校验 ==")
    for i in items:
        flag = "✓" if i["pass"] else ("✗" if i["hard"] else "△")
        kind = "硬" if i["hard"] else "软"
        print(f"  {flag} {i['id']:<10} [{kind}] {i['name']}")
        print(f"      值 {json.dumps(i['value'], ensure_ascii=False)}  阈 {i['threshold']}")
    code = exit_code(items)
    print("结果：" + {0: "全过", 1: "软项失败", 2: "硬项失败"}[code])


def run_island_check(ctx, node: int | None, year: int = 0, sets: list[str] | None = None) -> int:
    from . import generate
    if node is None:
        print("用法：skyisle island check <节点>")
        return 2
    out, g = generate(ctx, node, year=year, sets=sets, return_state=True)
    h1 = hash_products(out)
    generate(ctx, node, year=year, sets=sets, log=lambda *a: None)
    h2 = hash_products(out)
    items = evaluate(g, out, ctx=ctx, node=node, c=ctx.cfg["island"], det_hashes=(h1, h2))
    print_report(items, node)
    (out / "check.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return exit_code(items)
