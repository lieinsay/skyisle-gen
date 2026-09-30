"""聚落的前端（docs/PLAN-SETTLE.md）：聚落生成在 C++ 核心里（core/src/island/settle.cpp / farmland.cpp / waterworks.cpp / market.cpp，
Python 参考版删于 2026-09-30，tag python-reference-final）；这里只剩 settlements.png 的码表、备注，与 island.json 的 settlements 摘要、
地表的原始 / 人工改造摘要（decode.py 把 C++ 的记录译回中文之后拼）。

算法（C++ 同式）：给一个已生成的岛群，按人口与地形落下已垦的田（P5：好地先占 + 定居门槛）、
田块、村落与散户（P6b：圩田另成田块，挂到走得到的村上或自成圩村），专业聚落 / 集镇 / 飞船泊场（DESIGN-NOTES 四点十八）、水设施、
没人住的岛有人用（放牧 / 庙 / 墓岛）、主家候选 / 前哨 / 都与城，最后村周开垦与荒地归谁。
飞船取代车船、可在任意平地停靠：没有码头，交通不绑岸线；岛与岛之间没有索桥（P5，用户定），全靠船。
第三层，不回灌：人口只读 ⑨ 的 pop（没有 ⑨ 时按可耕地 × 100 人/km²）；不输出任何社会属性（原则乙、铁律五）。
"""
from __future__ import annotations

import numpy as np

RASTER_CODES = {"1": "田块", "2": "梯田", "3": "村", "4": "散户", "5": "泊场", "7": "蓄水池", "8": "取水点", "9": "镇", "10": "专业聚落",
                "11": "撂荒田", "12": "废村", "13": "工棚 / 季节住", "14": "有人用（放牧 / 夏牧 / 庙 / 墓岛）", "15": "塘", "16": "闸",
                "17": "大泊场（镇 / 邑治）", "18": "中转站（站址 / 烽火台的瞭望处）", "19": "废弃的水利（废村旁没人管的塘、渠首闸）"}
SETTLE_NOTE = ("第三层，人口只读 ⑨；村 / 镇 / 专业聚落只有位置与户数（原则乙）。名字是 村NNN / 镇NN 占位。"
               "飞船随处可停：没有码头，每个村 / 专业聚落旁一块泊场（船台），镇 / 邑治另挨着一处大泊场（harbors）；岛与岛之间没有索桥，全靠船——"
               "但能控制的浮石船造起来有门槛，一般人本岛走路赶集、跨岛搭航船（boat_lines）。households = 村农户 + 散户 + 镇的非农户 + 专业聚落户 + 中转站户"
               "（工棚、季节住的专业聚落的户也在里面，人住在 home_village 那个村；中转站的户住在中转站）。"
               "已垦 = 此刻有人种的田（terrain.npz 的 cultivated）；宜垦 = 地本身能种（cultivable）；撂荒 = fallow_years > 0。"
               "land_tenure：荒地有主、领照开垦（地在册，不等于都种着）——有村的岛归就近的村，没人住但有人用的归用它的村的大户，其余官荒归邑。"
               "P6b：圩田按 polder_village_blocks² 圩一组成田（fields[].polder），走得到现有的村就挂在那个村上（villages[].polder_fields），走不到就在圩上落圩村（villages[].polder）；"
               "水利每处都记管它的村（waterworks 的 village）。")


def set_settlements(g: dict, S: dict) -> None:
    """g["settle"] 与 island.json 的 settlements 摘要。"""
    g["settle"] = S
    J = g["json"]
    J["settlements"] = {k: S[k] for k in ("population", "households", "n_fields", "n_villages", "n_hamlets", "seat_households", "village_hh_median", "nonfarm_households")}
    J["settlements"].update({"n_towns": len(S["towns"]), "n_specials": len(S["specials"]), "specials": sorted({x["kind"] for x in S["specials"]}),
                             "n_landings": len(S["landings"]), "n_cisterns": len(S["cisterns"]), "n_intakes": len(S["intakes"]),
                             "n_harbors": len(S.get("harbors", [])), "n_boat_lines": len(S.get("boat_lines", [])), "n_relays": len(S.get("relays", [])),
                             "forest_share_after_clearing": S["clearing"]["forest_share_after"],
                             "workings": {k: {kk: vv for kk, vv in v.items() if kk in ("n", "villages_share")} for k, v in S["workings"].items()},
                             "n_outposts": len(S["outposts"]), "home_candidates": [h["kind"] for h in S["home_candidates"]],
                             "city": ({k: S["city"][k] for k in ("role", "households", "population", "n_guo_islands")} if S.get("city") else None)})
    # P5：已垦 / 宜垦 / 撂荒、废村、有人用的岛、各岛的住法与荒地归谁
    F = S["farmland"]
    st = [t["status"] for t in S["land_tenure"]]
    occ = [x["occupancy"] for x in S["specials"]]
    J["settlements"].update({
        "farmland_km2": {k: F[k] for k in ("cultivable_km2", "cultivated_km2", "fallow_km2")},
        "n_ruins": len(S["ruins"]), "uses": sorted({u["kind"] for u in S["uses"]}), "n_uses": len(S["uses"]),
        "specials_occupancy": {k: occ.count(k) for k in ("常住", "工棚", "季节住") if occ.count(k)},
        "island_status": {k: st.count(k) for k in ("常住", "季节住", "有人用", "荒岛") if st.count(k)}})
    if "waterworks" in S:                                # P6：水利
        ws = S["waterworks"]["summary"]
        J["settlements"]["waterworks"] = {k: ws[k] for k in ("n_heads", "canal_km", "polder_canal_km", "commanded_km2", "n_ponds", "n_sluices",
                                                             "wetland_km2", "polder_km2", "n_polders")}
        if "abandoned" in ws:                            # P6b：管水利的村、圩村、废弃的水利
            J["settlements"]["waterworks"].update({k: ws[k] for k in ("manage_walk_km", "n_polder_villages", "abandoned")})
    lf = J.get("landcover")
    if lf is not None:
        lf["note_farmland"] = "可耕地 / 梯田 = 已垦（此刻有人种）；宜垦见 terrain.npz 的 cultivable，撂荒见 fallow_years（地表按年头：草坡 → 灌丛 → 原本的地表）"
        if "landuse" in g:
            landuse_summary(g, lf)
    cons = J["constraints"]["arable_frac"]
    n_land = max(1, int((g["island_id"] >= 0).sum()))
    cons["actual"] = round(float((g["cultivated"] > 0).sum()) / n_land, 4)
    cons["note"] = "已垦（在种）/ 陆地；行星层的可耕率是已垦的额度（P5）；宜垦另见 cultivable_share"
    cons["cultivable_share"] = round(float((g["cultivable"] > 0).sum()) / n_land, 4)


def landuse_summary(g: dict, lf: dict) -> None:
    """P6b（L31）：island.json 的地表摘要加原始 / 现状的面积与人工改造各类的面积。
    natural_share / natural_km2：没有人以前的地表（landcover_natural）；current_km2：现状（landcover，share 是它的占比）；
    landuse：人工改造各类的面积（km2）与各类原来是什么地表（from_km2）。"""
    from .output import LANDCOVER_CLASSES
    from .waterworks import LANDUSE_CLASSES, LANDUSE_NOTE
    land = g["island_id"] >= 0
    n_land = max(1, int(land.sum()))
    cell_km2 = float(g["res_km"]) ** 2
    nat = g["landcover_natural"][land]
    cur = g["landcover"][land]
    lu = g["landuse"][land]
    cn = np.bincount(nat, minlength=12)
    cc = np.bincount(cur, minlength=12)
    lf["natural_share"] = {LANDCOVER_CLASSES[i]: round(float(cn[i]) / n_land, 4) for i in range(1, 12)}
    lf["natural_km2"] = {LANDCOVER_CLASSES[i]: round(float(cn[i]) * cell_km2, 3) for i in range(1, 12)}
    lf["current_km2"] = {LANDCOVER_CLASSES[i]: round(float(cc[i]) * cell_km2, 3) for i in range(1, 12)}
    x = np.bincount(lu.astype(np.int64) * 12 + nat.astype(np.int64), minlength=12 * len(LANDUSE_CLASSES)).reshape(len(LANDUSE_CLASSES), 12)
    lf["landuse"] = {"classes": LANDUSE_CLASSES,
                     "km2": {LANDUSE_CLASSES[u]: round(float(x[u].sum()) * cell_km2, 3) for u in range(1, len(LANDUSE_CLASSES))},
                     "from_km2": {LANDUSE_CLASSES[u]: {LANDCOVER_CLASSES[i]: round(float(x[u, i]) * cell_km2, 3) for i in range(1, 12) if x[u, i]}
                                  for u in range(1, len(LANDUSE_CLASSES))},
                     "changed_km2": round(float((nat != cur).sum()) * cell_km2, 3), "note": LANDUSE_NOTE}


CITY_NOTE = ("城居人口 = 本邑人口 × 城居率 + 邦人口 × 集聚率，只在聚落层算，不进 ⑨；郭 = 离城 ≤ city_guo_km、在城所在的岛上"
             "或与它岸距 ≤ city_guo_ferry_km（船程近）的岛上的村。有船的人哪里都飞得进来，崖缘挡不住：防守靠望楼、烽火与巡逻船")

