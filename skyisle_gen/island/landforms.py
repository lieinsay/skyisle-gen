"""岩层与特殊的山的前端（PLAN-NATURE B2 / B3，DESIGN-NOTES 四点四十六）：算法在 C++（core/src/island/strat.cpp、landforms.cpp）；
这里只有岩性与地貌的中文名、调色、island.json 的摘要与 lith.png。

- 岩性 lith（terrain.npz，uint8）：0 虚空 / 1 石灰岩 / 2 泥灰岩 / 3 辉长岩 / 4 蛇纹岩 / 5 浮石骨架——按地形的层面、水系之后的最终高程出。
- 地貌 landforms（island.json）：一处一条，kind 是代码、name 是中文，cell = 群栅格 [行, 列]，km = 相对群心（x 东 y 北），其余是各类的量。
- 崖层 rockwall_m / rockwall_dir（terrain.npz）：坡 ≥ [island.landform] rockwall_deg 的格所在岩壁的落差（m）与朝向（0–15，× 22.5°，0 = 朝北、顺时针；255 = 无）。
- 亚格岸距 coast_dist_m（terrain.npz，float32）：到岸线（岛形连续场的零等值线）的距离，陆地为正、虚空为负。
"""
from __future__ import annotations

import numpy as np

LITH_NAMES = ["虚空", "石灰岩", "泥灰岩", "辉长岩", "蛇纹岩", "浮石骨架"]
LITH_PALETTE = [(20, 24, 40), (225, 222, 205), (190, 170, 120), (60, 70, 70), (110, 150, 110), (190, 150, 230)]

LANDFORM_NAMES = {
    "fold_ridges": "平行岭谷",
    "tilted_block": "掀斜断块山",
    "tower_karst": "峰林",
    "cirque": "冰斗",
    "glacier": "冰川",
    "horn": "角峰",
    "collapse_scarp": "临空断山",
    "sky_mountain": "天上的山",
    "gabbro_crags": "辉长岩锯齿峰",
    "serpentine_barren": "蛇纹岩秃山",
    "mesa": "方山",
    "butte": "孤山",
    "natural_arch": "穿山天窗",
}

LANDFORM_NOTE = ("特殊的山（spec 13 第二节第 8、9 条）：按成因条件出，不设配额。cell = 群栅格 [行, 列]，km = 相对群心（x 东 y 北）。"
                 "公里级的形状已在高程里（平行岭谷、掀斜断块山、临空断山、冰斗、峰林的平地、方山）；100 m 以下的只记位置与尺寸，游戏建几何"
                 "（峰林的石柱：tower_h_m / density_per_km2；穿山天窗：opening_m、spur_strike_deg；岩壁看 terrain.npz 的 rockwall_m / rockwall_dir）")


def landforms_json(raw) -> list[dict]:
    """C++ 的记录（kind 代码）→ island.json 的 landforms（加中文名）。"""
    out = []
    for r in raw:
        d = dict(r)
        d["name"] = LANDFORM_NAMES.get(d["kind"], d["kind"])
        out.append(d)
    return out


def landform_counts(lf: list[dict]) -> dict:
    cnt: dict[str, int] = {}
    for r in lf:
        cnt[r["name"]] = cnt.get(r["name"], 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


def lith_summary(lith: np.ndarray, island_id: np.ndarray) -> dict:
    land = island_id >= 0
    n = max(1, int(land.sum()))
    share = {LITH_NAMES[k]: round(float(((lith == k) & land).sum()) / n, 4) for k in range(1, len(LITH_NAMES))}
    main = island_id == 0
    nm = max(1, int(main.sum()))
    share_main = {LITH_NAMES[k]: round(float(((lith == k) & main).sum()) / nm, 4) for k in range(1, len(LITH_NAMES))}
    return {"classes": LITH_NAMES, "palette": LITH_PALETTE, "share": share, "share_main": share_main,
            "note": "terrain.npz 的 lith（uint8）= 该格露出的岩性：按岛的层面（构造顶面、骨架顶面，[island.strat]）与水系之后的最终高程；"
                    "沉积盖层 = 石灰岩 / 泥灰岩互层（老岛顶上一层硬石灰岩盖），往下辉长岩、蛇纹岩，浮石骨架只在岸崖与贴着岸缘的深谷"}
