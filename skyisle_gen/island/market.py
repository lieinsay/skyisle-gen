"""泊场、中转站、镇、航船、邑治（P7，Zhouzhu PLAN-LAND L24–L26；DISCUSS-SETTLE-ISLANDS 第〇、二、五、九节；DESIGN-NOTES 四点三十八）。

算法在 C++ 核心里（core/src/island/market.cpp，Python 参考版删于 2026-09-30，tag python-reference-final）；
这里只剩代码 → 中文的表、中转站的备注与 settlements.json 的 market 摘要（decode.py 用）。下面是算法的说明：

船寻常，但能控制的浮石船造起来有门槛，一般人家没有（L24）：一般人本岛走路，跨岛搭定班的航船。
- **泊场先于镇**（harbor_*）：能停很多船、能堆货的大块缓坡平地——坡 ≤ harbor_slope_max_deg、不是崖缘 / 水面 / 湿地 / 漫滩、没在种也没撂荒的格，
  开运算（先蚀一圈再胀一圈）去掉一两格宽的条子；每格数它 harbor_window_km 方窗里同岛的这种格（「能停多少船」按面积 × harbor_use_frac / ship_m2），
  够 harbor_min_km2 的格按（格数降序、格号）挑，彼此隔 harbor_sep_km。每个村照旧有自己的船台（村泊场），大泊场是另一层：镇、邑治挨着它。
- **中转站**（relays）：群内是群边高处的瞭望烽火（统一 P5 的烽火台）与进出本群的关卡；群与群之间是航线上的过夜、候风、避风、换船。
  行星层 ⑥ 的每条邻边（本群 ↔ 邻群，流量、有向成本、离开几天）按方位合成几个口子（流量 ≥ relay_flow_min，彼此差 ≥ relay_gate_sep_deg，至多 relay_max_gates），
  每个口子落在朝那个方向最靠外的一座岛上（主岛以外、方位在 ±relay_sector_deg 内、有能落船的平地、能存雨水；看得远的加分），实在没有就落在主岛朝外的崖边平地。
  住的是守关的、驿卒、开客栈的、修船的、仓夫，不种地、粮靠外运（和矿镇一样算常住的专业聚落），户从非农户里出（封顶 relay_cap_frac）；烽火台轮班守、不常住。
- **镇**（towns）：以周围户数为主——本岛走 walk_km 以内的村按走路算，别的岛（和本岛太远的）只算航船够得着的（按船程：直线距离 × 风的系数，顺风便宜、逆风贵）；
  挨着大泊场的加一成到三成；按「还没被服务的户」贪心挑（走路服务过的不再算，航船服务过的只让后来的镇按走路再算），镇距 ≥ town_spacing_km，不到 town_min_served_hh 就停。
- **航船**（boat_lines）：走不到镇的村搭航船去船程最近的镇；每个镇的航船村按方位扫一圈分线（每线至多 line_max_stops 个村、一趟船程 ≤ line_max_km），
  船从最远的村出发，挨个接人，开到镇上的泊场。
- **邑治**：从镇里挑航船汇得最多、靠大泊场的（(服务户 + seat_boat_weight × 航船送来的户) × (1 + seat_harbor_weight × min(1, 泊场的船 / harbor_ref_ships))），
  不按最大田块、不按好守。
人口：总户数照旧只读 ⑨；非农户先给专业聚落，再给中转站，余下按服务户数分给各镇（市户）。
确定的次序（稳定排序、按格号 / 下标破平局、浮点按固定次序累加）。
"""
from __future__ import annotations

FUNC_ORDER = ("beacon", "gate", "inn", "wait", "shelter", "transship")
FUNC_ZH = {"beacon": "烽火", "gate": "关卡", "inn": "过夜", "wait": "候风", "shelter": "避风", "transship": "换船"}
INNER_FUNCS = ("beacon", "gate")                  # 群内：瞭望烽火、进出本群的关卡；其余是群与群之间的
ROLE_ORDER = ("gate", "post", "inn", "repair", "porter")
ROLE_ZH = {"gate": "守关", "post": "驿卒", "inn": "客栈", "repair": "修船", "porter": "仓夫"}
RELAY_OCC_ZH = {"resident": "常住", "rotation": "轮班"}
SCOPE_ZH = {"inner": "群内", "outer": "群间", "both": "群内、群间"}
MODE_ZH = {"walk": "步行", "boat": "航船"}
MARKET_NOTE = ("P7：船寻常，但能控制的浮石船造起来有门槛，一般人家没有——本岛走路赶集，跨岛搭定班的航船。harbors 是能停很多船的大块缓坡平地（先于镇）；"
               "镇以周围户数为主（本岛走路、跨岛只算航船够得着的），挨着大泊场的加分；邑治从镇里挑航船汇得最多、靠大泊场的。"
               "relays 是中转站：群内的瞭望烽火与关卡、群间的过夜 / 候风 / 避风 / 换船，住的人不种地、粮靠外运，户从非农户里出。")


def relay_note(funcs: list[str], island_main: bool) -> str:
    """中转站的说明（funcs 是代码）。"""
    parts = []
    if "beacon" in funcs:
        parts.append("群边高处的瞭望与烽火，轮班守")
    if "gate" in funcs:
        parts.append("进出本群的船在这里停下验货交税")
    outer = [FUNC_ZH[f] for f in funcs if f not in INNER_FUNCS]
    if outer:
        parts.append("群间航线上的" + "、".join(outer))
    s = "；".join(parts)
    return s + ("（落在主岛朝外的崖边平地：朝这个方向没有合适的小岛）" if island_main else "")


def market_summary(harbors: list[dict], towns: list[dict], lines: list[dict], relays: list[dict], villages: list[dict], mc: dict) -> dict:
    """settlements.json 的 market：泊场、镇、航船、邑治、中转站的几个总数。"""
    area = ships = 0.0
    for h in harbors:
        area += float(h["area_km2"])
        ships += float(h["ships"])
    lkm = 0.0
    for ln in lines:
        lkm += float(ln["length_km"])
    boat = [v for v in villages if v.get("market_mode") == MODE_ZH["boat"]]
    seat = next((t for t in towns if t["seat"]), None)
    funcs = {}
    for r in relays:
        for f in r["functions"]:
            funcs[f] = funcs.get(f, 0) + 1
    return {"n_harbors": len(harbors), "harbor_km2": round(area, 3), "harbor_ships": int(ships),
            "n_towns": len(towns), "towns_with_harbor": sum(1 for t in towns if t["harbor"] is not None),
            "n_lines": len(lines), "line_km": round(lkm, 2), "boat_villages": len(boat), "walk_villages": len(villages) - len(boat),
            "boat_households": sum(int(v["households"]) for v in boat),
            "far_villages": sum(1 for v in boat if float(v["market_km"]) > float(mc["boat_reach_km"])),
            "seat_town": seat["id"] if seat else None, "seat_island": seat["island"] if seat else None,
            "n_relays": len(relays), "relay_functions": {FUNC_ZH[f]: funcs[FUNC_ZH[f]] for f in FUNC_ORDER if FUNC_ZH[f] in funcs},
            "relay_households": sum(int(r["households"]) for r in relays), "note": MARKET_NOTE}
