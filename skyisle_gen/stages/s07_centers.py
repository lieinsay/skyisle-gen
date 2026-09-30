"""⑦ 文明中心（骨架窗内涌现 + 固定 id）、史前扩散、⑦b 地区划分与中心间干线。

适宜度 = 降水 × 稳定气候 × 岛密度 × 岛群陆地^γ × 谷物门槛（docs/12 §五 ⑦ + docs/02 §六 集雨面）。
谷物门槛（骨架第二版 §4.2）= f(海面冬温)：冬天够冷才选得出一年生耐储谷物，太冷则生长季不够；
常夏之地与苦寒之地都起不了谷物农业。季节量取 ④ 的零点口径（原叫海面口径，键名 *_sea 沿用；区域陆地性，不含岛高，原则乙）。
不含高度（原则乙）；陆地不是海拔，是集雨面与人口容量。节点 = 岛群（R10）。
铁律自检：本阶段的社会推导不读 height_m。
由 C++ 核心（core/src/planet/stage7.cpp；行星计划 P6d，Python 参考版删于 2026-09-30，tag python-reference-final）算，centers.json / prehist.npz / regions.npz 与摘要照旧由这里写（_write）。
"""
from __future__ import annotations

import numpy as np

CENTER_IDS = ["north_west", "north_east", "south"]
CENTER_ZH = {"north_west": "北带西中心", "north_east": "北带东中心", "south": "南带中心"}


def run(ctx):
    from ..engine import core, part, planet_config, put_part
    cc = core()
    Ce = cc.planet_stage7(cc.make_config(planet_config(ctx.cfg)), part(ctx, 1), part(ctx, 3), part(ctx, 4), part(ctx, 5), part(ctx, 6))
    put_part(ctx, 7, Ce)
    return _write(ctx, cc.centers_arrays(Ce))


def _write(ctx, R: dict) -> dict:
    """写 prehist.npz / regions.npz / centers.json、出摘要（R 里的浮点是双精度原值，这里照旧转 float32）。"""
    isl = ctx.load_npz(3, "islands")
    lat, lon = isl["lat"], isl["lon"]
    suit = R["suit"]
    centers = {}
    for cid, node in zip(CENTER_IDS, R["nodes"]):
        node = int(node)
        centers[cid] = {"node": node, "lat": round(float(lat[node]), 3),
                        "lon": round(float(lon[node]), 3), "suitability": round(float(suit[node]), 4),
                        "zh": CENTER_ZH[cid]}
    chosen = [int(p) for p in R["secondary"]]
    seeds = [int(s) for s in R["region_seeds"]]
    trunks = {t["key"]: {"cost_days": round(float(t["cost_days"]), 1) if t["reachable"] else None, "path": [int(u) for u in t["path"]]}
              for t in R["trunks"]}
    ctx.save_npz(7, "prehist", dist_pre=R["dist_pre"], pred=R["pred"],
                 lineage=R["lineage"], arrival_yr=R["arrival_yr"].astype(np.float32))
    ctx.save_npz(7, "regions", region=R["region"], suitability=suit.astype(np.float32))
    ctx.save_json(7, "centers", {
        "centers": centers,
        "origin_node": int(R["origin_node"]),
        "secondary_peaks": [{"node": p, "lat": round(float(lat[p]), 3),
                             "lon": round(float(lon[p]), 3), "suitability": round(float(suit[p]), 4)}
                            for p in chosen],
        "region_seeds": seeds,
        "center_trunks": {k: {"cost_days": v["cost_days"], "n_nodes": len(v["path"]),
                              "path": v["path"]} for k, v in trunks.items()},
    })
    return {"centers": {cid: (centers[cid]["lat"], centers[cid]["lon"]) for cid in CENTER_IDS},
            "n_lineages": int(R["n_lineages"]), "n_regions": len(seeds),
            "trunk_nw_ne_days": trunks.get("north_east->north_west", trunks.get("north_west->north_east", {})).get("cost_days")}
