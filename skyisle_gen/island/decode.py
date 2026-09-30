"""cpp 后端的记录 → 与 Python 版同形的 dict（行星计划 P6b，docs/PLAN-CORE.md 第五节）。

C++ 的记录里数已按 Python 的 round 舍好，字符串是 ASCII 代码（PLAN-CORE：C++ 只用 ASCII，中文在前端）；这里把代码译回中文、
把带数的备注按 Python 版的 f-string 拼出来。赋存区的长度与走向按 numpy 的 np.cov / np.linalg.eigh 重算（resources._shape：
OpenBLAS 的 syrk 与 LAPACK 的结果 C++ 追不上位，对称的小块协方差 ±1e−18 的符号就决定走向是 0° 还是 180°）。
"""
from __future__ import annotations

import numpy as np

from .resources import (NOTE_FOSSIL, NOTE_HOTSPRING, NOTE_SALT, NOTE_SALTSPRING, NOTE_SULFUR, NOTE_VOID, ORE_NOTE, RES_FORM, RES_INDEX, RES_NAMES,
                        WORK_ZH, ZONE_NAMES, _shape)

ZONE_KEYS = ["void", "alpine", "mountain", "hill", "plain", "valley", "cliff", "water"]
ZONE_ZH = dict(zip(ZONE_KEYS, ZONE_NAMES))
GRADE_ZH = {"hi": "上", "mid": "中", "lo": "下"}
AGE_ZH = {"young": "新岛", "mid": "中年", "old": "老岛"}
SUBTYPE_ZH = {
    "conifer": "针叶林", "mixed": "针阔混交林", "broadleaf": "阔叶林", "peat": "泥炭", "reed": "芦苇荡",
    "cliff_face": "崖面露头", "gorge": "峡谷露头",
    "skeleton_void": "骨架空洞", "gorge_cave": "峡谷岩洞", "fracture_cave": "裂隙洞",
    "karst_cave": "溶洞", "sinkhole": "落水洞", "underground_river": "地下河", "cliff_cave": "崖洞", "underside_cave": "底面洞",
    "waterfall_cave": "瀑布后洞",
    "goldsilver": "金银", "iron": "铁", "leadzinc": "铅锌", "copper": "铜", "tin": "锡", "chromium": "铬", "nickel": "镍", "manganese": "锰",
    "hydrothermal": "热泉硫磺", "marine_limestone": "海相石灰岩", "gabbro": "辉长岩", "serpentinite": "蛇纹岩",
    "lake_clay": "河湖黏土", "stream_clay": "溪边黏土", "river_gravel": "河滩砂砾", "placer_gold": "砂金", "salt_dome": "盐丘",
}
NOTE_ZH = {
    "floatstone_body": "岛体本身的浮石：可开采，采掉的量相对岛体微不足道，不影响浮空",
    "cliff_cave": "开在岸崖上：从云带上方悬索进出",
    "underside_cave": "入口在崖下的岛底，只能悬索或飞舟进出",
    "timber_cleared": "已开垦殆尽（村周草坡 / 薪炭林）",
    "ore_sulfide": ORE_NOTE["热泉硫化物"], "ore_serpentinite": ORE_NOTE["蛇纹岩"], "ore_nodule": ORE_NOTE["锰结核"],
    "placer_upstream": "上游有金银 / 铜矿化带（热泉硫化物）：河砂可淘金",
    "hotspring_young": NOTE_HOTSPRING, "sulfur_hydrothermal": NOTE_SULFUR, "skeleton_void": NOTE_VOID,
    "salt_sediment": NOTE_SALT, "salt_spring": NOTE_SALTSPRING, "fossil_shell": NOTE_FOSSIL,
    "placer_no_village": "附近没有村：季节性的外来淘金客",
}
# 四季
TYPE_ZH = {"four": "四季分明", "two": "冷暖两季", "rain": "雨旱季", "storm": "风暴季", "none_warm": "常夏", "none_cold": "常寒"}
_NUM = "一二三四五六"
SEASON_ZH = {"hot": "热季", "cold": "冷季", "warm": "暖季", "cool": "凉季", "wet": "雨季", "dry": "旱季", "turn": "转季",
             "stormy": "风暴季", "calm": "平静季", "transition": "过渡季", "season": "季"}
SEASON_ZH.update({f"warm{i + 1}": f"暖季{_NUM[i]}" for i in range(4)})
SEASON_ZH.update({f"cold{i + 1}": f"冷季{_NUM[i]}" for i in range(4)})
SEASON_ZH.update({f"s{i + 1}": f"第{_NUM[i]}季" for i in range(6)})
CLIMATE_NOTE = ("岛上气温（temp_c、逐日 temp_c）是台面 temp_ref_height_m（③ 的 height_m = 主岛陆地高程中位数）处的，某格 = temp + 直减率 × (台面 − 格高)；"
                "季名是软的（决定 5）；南北半球反相；每季数值为季中那一天的摆动取样再缩放到年均；precip_mm 是该季总量（四季之和 = 年降水），"
                "precip_mm_annual_rate 是折成年当量的强度")
# 聚落
SPECIAL_ZH = {"ore_town": "矿镇", "ore_village": "矿村", "floatstone_village": "浮石采石村", "kiln_village": "窑村",
              "charcoal_camp": "烧炭营", "hotspring": "温泉地", "salt_village": "盐井村"}
WATER_ZH = {"waterway": "河湖溪涧", "cistern": "蓄水池", "intake": "取水点"}
HOME_ZH = {"seat_side": "邑治旁", "riverside": "河湖僻处", "highland": "高台", "outpost_isle": "小岛前哨"}
HOME_NOTE = {"seat_side": "人多、近航线：邑治与主泊场（仓场）之间的一块地", "riverside": "离村三公里外、临水的僻静处",
             "highland": "主岛能落脚的最高处，俯瞰全群"}
WORKS_ZH = {"stone": WORK_ZH["stone"], "clay": WORK_ZH["clay"], "gravel": WORK_ZH["gravel"], "placer": WORK_ZH["placer"]}
POP_SRC_ZH = {"polity": "⑨ polity.npz", "arable": "可耕地 × 人口密度（无 ⑨）"}


def _sub(code):
    return None if code is None else SUBTYPE_ZH[code]


# ---------------------------------------------------------------- 资源
def deposit(d: dict) -> dict:
    kind = d["kind"]
    out = {"id": d["id"], "kind": kind, "kind_zh": RES_NAMES[RES_INDEX[kind]], "form": RES_FORM[kind], "subtype": _sub(d["subtype"]),
           "island": d["island"], "cell": d["cell"], "km": d["km"], "area_km2": d["area_km2"], "elev_m": d["elev_m"], "slope_deg": d["slope_deg"],
           "zone": ZONE_ZH[d["zone"]], "grade": GRADE_ZH[d["grade"]]}
    note = d.get("note")
    if note == "waterfall_cave":
        out["note"] = f"常年河（流域 {d['note_arg']:.0f} km²）跌下崖缘处的水帘后"
    elif note:
        out["note"] = NOTE_ZH[note]
    if d.get("cleared"):
        out["cleared"] = True
    if "area_before_clearing_km2" in d:
        out["area_before_clearing_km2"] = d["area_before_clearing_km2"]
    return out


def occurrence(o: dict, cells: np.ndarray, W: int, res_km: float) -> dict:
    kind = o["kind"]
    ii, jj = cells // W, cells % W
    length, ax = _shape(ii.astype(np.int64), jj.astype(np.int64), res_km)
    out = {"id": o["id"], "kind": kind, "kind_zh": RES_NAMES[RES_INDEX[kind]], "subtype": _sub(o["subtype"]), "island": o["island"],
           "cell": o["cell"], "km": o["km"], "area_km2": o["area_km2"], "length_km": length, "axis_deg": ax,
           "grade_peak": o["grade_peak"], "grade_mean": o["grade_mean"], "grade": GRADE_ZH[o["grade"]], "elev_m": o["elev_m"],
           "zone": ZONE_ZH[o["zone"]]}
    if o.get("note"):
        out["note"] = NOTE_ZH[o["note"]]
    out["n_workings"] = o["n_workings"]
    return out


def working(w: dict) -> dict:
    out = {"id": w["id"], "kind": w["kind"], "kind_zh": WORK_ZH[w["kind"]], "occurrence": w["occurrence"], "island": w["island"],
           "cell": w["cell"], "km": w["km"], "grade": w["grade"], "villages": w["villages"], "special": w["special"]}
    if w.get("note"):
        out["note"] = NOTE_ZH[w["note"]]
    return out


def resources(g: dict, node: int, c: dict, R: dict) -> dict:
    """C++ 的资源层 → g 的栅格与 g["resources"]（resources.json 的整体；计数由 resource_summary 补）。"""
    from .resources import FIELD_KINDS, resource_record
    rc = c["resources"]
    J = g["json"]
    W = g["island_id"].shape[1]
    land = g["island_id"] >= 0
    g.update({"terrain_zone": R["terrain_zone"], "patch_id": R["patch_id"], "resource": R["resource"], "res_field": R["res_field"],
              "occ_lab": {k: R["occ_lab"][i] for i, k in enumerate(FIELD_KINDS)}})
    meta = J["meta"]
    deps = [deposit(d) for d in R["deposits"]]
    occs = [occurrence(o, cl, W, g["res_km"]) for o, cl in zip(R["occurrences"], R["occ_cells"])]
    works = [working(w) for w in R["workings"]]
    thr = {k: float(rc["occ_thr"][k]) for k in FIELD_KINDS}
    return resource_record(node, R["terrain_zone"], land, rc, int(R["r_cells"]), thr, meta["boundary_type"], float(meta["boundary_kernel"]),
                           bool(meta["layered"]), float(R["geo_ore"]), float(R["fs_rate"]), deps, occs, works)


def resources_after_settle(Rg: dict, R: dict, W: int, res_km: float) -> None:
    """聚落之后：开垦改了林木的面积 / 代表格，村的采场加进了 workings、特殊聚落写了 special——按 C++ 的终值整体换掉三张表。"""
    Rg["deposits"] = [deposit(d) for d in R["deposits"]]
    Rg["occurrences"] = [occurrence(o, cl, W, res_km) for o, cl in zip(R["occurrences"], R["occ_cells"])]
    Rg["workings"] = [working(w) for w in R["workings"]]


# ---------------------------------------------------------------- 四季
def climate(Cj: dict, planet: dict) -> dict:
    from .climate import calendar
    out = dict(Cj)
    code = out.pop("season_type_code")
    out["calendar"] = calendar(planet)
    out["season_type_zh"] = TYPE_ZH[code]
    out["season_names"] = [SEASON_ZH[n] for n in Cj["season_names"]]
    seasons = []
    for s in Cj["seasons"]:
        s = dict(s)
        s["name"] = SEASON_ZH[s["name"]]
        seasons.append(s)
    out["seasons"] = seasons
    out["note"] = CLIMATE_NOTE
    return out


# ---------------------------------------------------------------- 聚落
def _name(code):
    """村 / 散户 / 专业聚落的名字代码 → 中文（village:12 → 村012）。"""
    if code is None:
        return None
    kind, _, rest = code.partition(":")
    if kind == "village":
        return f"村{int(rest):03d}"
    if kind == "hamlet":
        return f"散户{int(rest):03d}"
    if kind == "special":
        sk, _, sid = rest.partition(":")
        return f"{SPECIAL_ZH[sk]}{int(sid):02d}"
    if kind == "ruin":
        return f"废村{int(rest):02d}"
    if kind == "seat":
        return "邑治"
    if kind == "town":
        return f"镇{int(rest):02d}"
    if kind == "relay":
        return f"中转站{int(rest):02d}"
    if kind == "line":
        return f"航船{int(rest):02d}"
    raise ValueError(code)


def _village(v: dict) -> dict:
    from .market import MODE_ZH
    v = dict(v)
    v["name"] = _name(v["name"])
    if "market_mode" in v:
        v["market_mode"] = MODE_ZH[v["market_mode"]]
    if "water" in v:
        v["water"] = {"source": WATER_ZH[v["water"]["source"]], "dist_km": v["water"]["dist_km"]}
    return v


def _special_note(n: list) -> str:
    k = n[0]
    if k == "ore":
        return f"{SUBTYPE_ZH[n[1]]}矿化带 {n[2]:.1f} km²（{GRADE_ZH[n[3]]}品，{n[4]} 坑），吃粮靠外运"
    if k == "floatstone":
        return f"{SUBTYPE_ZH[n[1]]} {n[2]:.2f} km²；飞船就地装运"
    if k == "kiln":
        return f"上等{SUBTYPE_ZH[n[1]]}（{n[2]:.2f} km²），附近有林可烧"
    if k == "charcoal":
        return f"{SUBTYPE_ZH[n[1]]} {n[2]:.0f} km²，离最近的村 {n[3]:.1f} km（季节性）"
    if k == "hotspring":
        return "汤治 / 寺社"
    if k == "salt":
        return f"{SUBTYPE_ZH[n[1]]} {n[2]:.1f} km²（{GRADE_ZH[n[3]]}品），汲卤煮盐外运，吃粮靠外运"
    raise ValueError(k)


def _landing_kind(code: str) -> str:
    if code == "main":
        return "主泊场（仓场）"
    if code == "town":
        return "镇泊场"
    if code == "village":
        return "村泊场"
    if code == "relay":
        return "中转站泊场"
    return f"{SPECIAL_ZH[code.split(':', 1)[1]]}泊场"


def waterworks(Wj: dict) -> dict:
    """P6 水利：渠 / 塘 / 闸的代码、渠首的水源 → 中文，加说明（waterworks.py 的 build_waterworks 同形）。"""
    from .waterworks import BIG_ZH, CANAL_ZH, MAINT_ZH, POND_ZH, SLUICE_ZH, SOURCE_ZH, WORKS_NOTE
    W = dict(Wj)
    W["heads"] = [dict(h, source=SOURCE_ZH[h["source"]]) for h in Wj["heads"]]
    if "big_works" in Wj:                         # 四点四十：邑级大堰
        W["big_works"] = [dict(x, kind=BIG_ZH[x["kind"]], source=SOURCE_ZH[x["source"]], maintainer=MAINT_ZH[x["maintainer"]]) for x in Wj["big_works"]]
    W["canals"] = [dict(x, kind=CANAL_ZH[x["kind"]], **({"maintainer": MAINT_ZH[x["maintainer"]]} if "maintainer" in x else {})) for x in Wj["canals"]]
    W["ponds"] = [dict(x, kind=POND_ZH[x["kind"]]) for x in Wj["ponds"]]
    W["sluices"] = [dict(x, kind=SLUICE_ZH[x["kind"]]) for x in Wj["sluices"]]
    s = dict(Wj["summary"])
    s["ponds"] = {POND_ZH[k]: v for k, v in Wj["summary"]["ponds"].items()}
    s["sluices"] = {SLUICE_ZH[k]: v for k, v in Wj["summary"]["sluices"].items()}
    W["summary"] = s
    W["note"] = WORKS_NOTE
    return W


def settlements(Sj: dict, climate_zh: dict | None, mc: dict | None = None) -> dict:
    from . import market as MK
    from .farmland import OWNER_ZH, SPECIAL_OCC_ZH, STATUS_ZH, USE_OCC_ZH, USE_ZH, ruin_note, use_note
    from .settle import CITY_NOTE, RASTER_CODES, SETTLE_NOTE
    S = dict(Sj)
    S["population_source"] = POP_SRC_ZH[Sj["population_source"]]
    S["villages"] = [_village(v) for v in Sj["villages"]]
    S["hamlets"] = [_village(v) for v in Sj["hamlets"]]
    towns = []
    for t in Sj["towns"]:
        t = dict(t)
        t["name"] = _name(t["name"])
        towns.append(t)
    S["towns"] = towns
    # P7：航船线、大泊场、中转站（代码 → 中文，说明两个后端共用 market.relay_note）
    S["boat_lines"] = [dict(x, name=_name(x["name"])) for x in Sj.get("boat_lines", [])]
    S["harbors"] = [dict(x) for x in Sj.get("harbors", [])]
    relays = []
    for x in Sj.get("relays", []):
        x = dict(x)
        codes = list(x["functions"])
        x["functions"] = [MK.FUNC_ZH[f] for f in codes]
        x["scope"] = MK.SCOPE_ZH[x["scope"]]
        x["roles"] = {MK.ROLE_ZH[k]: v for k, v in x["roles"].items()}
        x["occupancy"] = MK.RELAY_OCC_ZH[x["occupancy"]]
        x["name"] = _name(x["name"])
        x["note"] = MK.relay_note(codes, x["island"] == 0)
        relays.append(x)
    S["relays"] = relays
    specials = []
    for x in Sj["specials"]:
        x = dict(x)
        x["kind"] = SPECIAL_ZH[x["kind"]]
        x["subtype"] = _sub(x["subtype"])
        x["note"] = _special_note(x["note"])
        x["name"] = _name(x["name"])
        x["occupancy"] = SPECIAL_OCC_ZH[x["occupancy"]]
        specials.append(x)
    S["specials"] = specials
    lands = []
    for L in Sj["landings"]:
        L = dict(L)
        L["kind"] = _landing_kind(L["kind"])
        L["of"] = _name(L["of"])
        lands.append(L)
    S["landings"] = lands
    S["outposts"] = [dict(o, settlement=_name(o["settlement"])) for o in Sj["outposts"]]
    homes = []
    for h in Sj["home_candidates"]:
        h = dict(h)
        code = h["kind"]
        h["kind"] = HOME_ZH[code]
        note = h["note"]
        h["note"] = f"离主岛最远的小岛，一片田（{note[1]} km²）和一块泊场" if isinstance(note, list) else HOME_NOTE[note]
        h["island_age"] = AGE_ZH[h["island_age"]]
        if climate_zh is not None:
            h["season_type"] = climate_zh["season_type_zh"]
            h["season_names"] = climate_zh["season_names"]
        homes.append(h)
    S["home_candidates"] = homes
    if Sj.get("city"):
        city = dict(Sj["city"])
        city["role"] = "变法之国的都" if city["role"] == "reformer_capital" else "邦都"
        city["note"] = CITY_NOTE
        S["city"] = city
    S["workings"] = {WORKS_ZH[k]: v for k, v in Sj["workings"].items()}
    # P5：废村、有人用的岛、各岛的住法与荒地归谁
    S["ruins"] = [dict(r, name=_name(r["name"]), note=ruin_note(r["note"][1], r["note"][2], r["note"][3])) for r in Sj["ruins"]]
    uses = []
    for u in Sj["uses"]:
        u = dict(u)
        n = u["note"]
        u["note"] = use_note(n[0], *n[1:])
        u["kind"] = USE_ZH[u["kind"]]
        u["occupancy"] = USE_OCC_ZH[u["occupancy"]]
        uses.append(u)
    S["uses"] = uses
    S["land_tenure"] = [dict(t, status=STATUS_ZH[t["status"]], owner=OWNER_ZH[t["owner"]]) for t in Sj["land_tenure"]]
    if "waterworks" in Sj:
        S["waterworks"] = waterworks(Sj["waterworks"])
    S["raster_codes"] = dict(RASTER_CODES)
    S["note"] = SETTLE_NOTE
    if mc is not None:
        S["market"] = MK.market_summary(S["harbors"], S["towns"], S["boat_lines"], S["relays"], S["villages"], mc)
    return S
