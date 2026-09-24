"""第六节：岛群生成器的一致性校验（`skyisle island check <节点>`）。

IS-area / IS-summit / IS-arable / IS-river / IS-channel / IS-season / IS-link / IS-det / IS-iso 为硬项，IS-daily 为软项；
资源 RES-site / RES-geo 为硬项，RES-quarry 为软项；
聚落 SET-pop / SET-field / SET-site / SET-land / SET-town / SET-home 为硬项，SET-water 为软项（PLAN-SETTLE 第六节；SET-dock 随码头取消，四点十八）。
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
             det_hashes: tuple[dict, dict] | None = None, daily_years: int = 60) -> list[dict]:
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
    e_h = abs(h["actual"] - h["target"]) / max(1e-9, h["target"])
    add("IS-summit", "主岛最高格 = height_m", round(e_h, 5), "< 0.01", e_h < 0.01)
    ar = cons["arable_frac"]
    e_ar = abs(ar.get("actual", -1) - ar["target"])
    add("IS-arable", "可耕地 / 陆地 = arable_frac", round(e_ar, 5), "< 0.005", e_ar < 0.005)
    rv = cons["has_river"]
    small_ok = all(not i.get("has_perennial_river", False) for i in J["islands"][1:])
    add("IS-river", "主岛有常年河 ⇔ has_river；小岛只有溪涧", {"main": rv.get("actual"), "target": rv["target"], "small_islands_ok": small_ok},
        "相等", rv.get("actual") == rv["target"] and small_ok)
    if "river_width_m" in g:
        rvm = g["river"] > 0
        ok_wd = bool(((g["river_width_m"][rvm] > 0) & (g["river_depth_m"][rvm] > 0)).all()) if rvm.any() else True
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
        add("IS-channel", "常年河每格有河宽 / 水深；河道切在两岸之下（横断面两侧都不低于河道）的占比", {"width_depth_ok": ok_wd, "below_banks": round(below, 4)},
            "全有 / ≥ 0.95", ok_wd and below >= 0.95)
    R = g.get("resources")
    if R is not None:
        bad_site, bad_geo = [], []
        ages = {i["id"]: i["age_zh"] for i in J["islands"]}
        lith = R["geology"]["old_island_lithology"]
        for d in R["deposits"]:
            if d.get("cleared"):
                continue                        # 整片被村周开垦掉的林场
            i, j_ = d["cell"]
            wet = bool(g["river"][i, j_] > 0 or g["lake"][i, j_])
            on_cliff_ok = d["kind"] in ("cave", "floatstone", "guano")
            if (g["island_id"][i, j_] != d["island"] or wet or (g["cliff"][i, j_] and not on_cliff_ok)
                    or g["resource"][i, j_] == 0):
                bad_site.append(d["id"])
            a = ages.get(d["island"])
            sub = d.get("subtype")
            if ((sub == "熔岩管" and a != "新岛") or (str(sub).startswith("熔岩管") and a == "老岛")
                    or (sub in ("溶洞", "落水洞", "地下河") and (a != "老岛" or lith != "石灰岩"))
                    or (d["kind"] == "sulfur" and a != "新岛") or (d["kind"] == "hotspring" and a == "老岛")
                    or (d["kind"] == "quarry" and sub != {"新岛": "玄武岩", "中年": "安山岩 / 凝灰岩", "老岛": lith}.get(a))):
                bad_geo.append(d["id"])
        add("RES-site", "矿点在所属岛的陆地上、不在水面；除崖洞 / 浮石 / 鸟粪外不在崖缘；资源栅格上有标记", {"bad": bad_site[:10], "n": len(R["deposits"])},
            "bad = 0", not bad_site)
        add("RES-geo", "矿点与地质背景一致（完整熔岩管 / 硫磺只在新岛、塌陷熔岩管不在老岛，溶洞只在石灰岩老岛，老岛无温泉，石料岩性随岛龄）", {"bad": bad_geo[:10]}, "bad = 0", not bad_geo)
        nq = sum(1 for d in R["deposits"] if d["kind"] == "quarry" and d["island"] == 0)
        add("RES-quarry", "主岛至少一处采石场（有石料盖城）", nq, "≥ 1", nq >= 1, hard=False)
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
    # IS-link：索桥 + 短渡连通
    from ..graph import weak_components
    n = len(J["islands"])
    src = np.array([e["a"] for e in J["links"]], dtype=np.int64)
    dst = np.array([e["b"] for e in J["links"]], dtype=np.int64)
    ncomp = int(weak_components(n, src, dst).max()) + 1 if n > 1 else 1
    add("IS-link", "群内任意两岛经索桥 + 短渡连通", {"components": ncomp, "islands": n}, "1 个分量", ncomp == 1)
    # ---- 聚落（PLAN-SETTLE 第六节）
    S = g.get("settle")
    if S is not None:
        import numpy as _np
        parts = S["households_in_villages"] + S["households_in_hamlets"] + S.get("households_in_towns_market", 0) + S.get("households_in_specials", 0)
        hh_ok = parts == S["households"] == round(S["population"] / S["household_size"])
        nf_ok = S.get("households_in_towns_market", 0) + S.get("households_in_specials", 0) == S.get("nonfarm_households", 0) or not S["villages"]
        add("SET-pop", "村农户 + 散户 + 镇非农户 + 专业聚落户 = 总户数 = 群人口 / 户均（只读 ⑨）；镇 + 专业 = 非农户",
            {"households": S["households"], "parts": parts, "population": S["population"], "nonfarm": S.get("nonfarm_households")}, "相等", hh_ok and nf_ok)
        arable_km2 = float((g["arable"] > 0).sum()) * (J["raster"]["res_m"] / 1000.0) ** 2
        f_sum = sum(f["area_km2"] for f in S["fields"])
        every = all(f.get("households", 0) >= 8 for f in S["fields"] if f.get("village") and f["village"] > 0)
        add("SET-field", "每村有田；Σ 田块 = 可耕地", {"fields_km2": round(f_sum, 3), "arable_km2": round(arable_km2, 3), "villages_have_field": every}, "< 1%",
            abs(f_sum - arable_km2) <= 0.01 * max(arable_km2, 1e-9) and every)
        bad = 0
        for v in S["villages"] + S.get("specials", []):
            i, j_ = v["cell"]
            if g["cliff"][i, j_] or g["lake"][i, j_] or g["river"][i, j_] or g["island_id"][i, j_] != v["island"]:
                bad += 1
        cells = _np.array([v["cell"] for v in S["villages"]], dtype=float)
        sep_share = 1.0
        if cells.shape[0] > 1:
            d = _np.sqrt(((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1)) * J["raster"]["res_m"] / 1000.0
            _np.fill_diagonal(d, 9e9)
            sep_share = float((d.min(axis=1) >= 1.0 - 1e-9).mean())
            bad += int((d.min() <= 0.05))
        add("SET-site", "村不在崖缘 / 水面 / 别的岛上、不同格；1 km 间距（软，报告占比）", {"bad": bad, "sep_1km_share": round(sep_share, 3)}, "bad = 0", bad == 0)
        # SET-land：飞船随处可停——每个村 / 镇 / 专业聚落有一块泊场，在同岛、不在水面崖缘上；索桥两端各一桥头
        lands = {L_["id"]: L_ for L_ in S.get("landings", [])}
        miss, bad_l, steep = 0, 0, 0
        for v in S["villages"] + S.get("specials", []):
            L_ = lands.get(v.get("landing"))
            if L_ is None:
                miss += 1
                continue
            i, j_ = L_["cell"]
            if g["island_id"][i, j_] != v["island"] or g["lake"][i, j_] or g["river"][i, j_] > 0 or g["cliff"][i, j_]:
                bad_l += 1
            steep += not L_["flat"]
        n_bridge = sum(1 for e in J["links"] if e["kind"] == "bridge")
        add("SET-land", "每个村 / 镇 / 专业聚落有泊场（同岛、非水非崖；坡 > 4° 的只报告）；索桥两端各一桥头",
            {"missing": miss, "bad": bad_l, "not_flat": steep, "bridgeheads": len(S["bridgeheads"]), "bridges": n_bridge},
            "无缺", miss == 0 and bad_l == 0 and len(S["bridgeheads"]) == 2 * n_bridge)
        T_ = S.get("towns", [])
        vt = [v for v in S["villages"] if v.get("market_town")]
        seat_town = any(t_.get("seat") for t_ in T_)
        add("SET-town", "有村就有镇、邑治是镇、每个村归一个镇", {"towns": len(T_), "villages": len(S["villages"]), "assigned": len(vt), "seat_is_town": seat_town},
            "全满足", not S["villages"] or (len(T_) >= 1 and seat_town and len(vt) == len(S["villages"])))
        add("SET-water", "村 1 km 内有水源的占比", S["water_ok_share"], "≥ 0.8", S["water_ok_share"] >= 0.8, hard=False)
        kinds = [h["kind"] for h in S["home_candidates"]]
        add("SET-home", "主家候选 2–3 个、类型各异", kinds, "2–3 个", 2 <= len(kinds) <= 3 and len(kinds) == len(set(kinds)))
    if det_hashes is not None:
        h1, h2 = det_hashes
        diff = sorted(k for k in set(h1) | set(h2) if h1.get(k) != h2.get(k))
        add("IS-det", "同输入重跑两次，产物哈希一致", {"differ": diff}, "无差异", not diff)
    ok, bad = static_isolation()
    add("IS-iso", "stages/ 不 import skyisle_gen.island（第三层不回灌）", {"offenders": bad}, "无", ok)
    return items


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
