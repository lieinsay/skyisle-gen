"""⑥ 航线网络：有向成本（顺风廉价逆风昂贵）、分模式成本、抽样介数 → 干线与枢纽。

风是「轻度影响」（docs/02 §四）：只进成本，不进通过率。
干线/枢纽只由地理量（密度×集雨容量加权的抽样介数）决定，不读文明中心（⑦ 在后，不可倒序）。
由 C++ 核心（core/src/planet/stage6.cpp；介数按源分块并行，结果与线程数无关；行星计划 P6d，Python 参考版删于 2026-09-30，tag python-reference-final）算，routes.npz / hubs.json 与摘要照旧由这里写（_write）。
"""
from __future__ import annotations

import numpy as np

from .. import MODES


def run(ctx):
    from ..engine import core, part, planet_config, put_part
    cc = core()
    threads = max(1, int((ctx.cfg.get("engine") or {}).get("threads", 4)))
    R = cc.planet_stage6(cc.make_config(planet_config(ctx.cfg)), int(ctx.seed), part(ctx, 2), part(ctx, 3), part(ctx, 4), part(ctx, 5),
                         threads=threads)
    put_part(ctx, 6, R)
    return _write(ctx, cc.routes_arrays(R))


def _write(ctx, R: dict) -> dict:
    """写 routes.npz 与 hubs.json、出摘要（R 里的浮点是双精度原值，这里照旧转 float32）。"""
    isl = ctx.load_npz(3, "islands")
    lat, lon = isl["lat"], isl["lon"]
    cost_d, node_flow, near_g = R["cost"], R["node_flow"], R["near_g"]
    hubs_sorted = [int(i) for i in R["hubs"]]
    comp_report = R["components"]
    ctx.save_npz(6, "routes", cost=cost_d, cost_no_g=R["cost_no_g"], cost_m=R["cost_m"],
                 flow=R["flow"].astype(np.float32), node_flow=node_flow.astype(np.float32),
                 src_d=R["src_d"], dst_d=R["dst_d"], und_id=R["und_id"], betweenness_sources=R["sources"])
    ctx.save_json(6, "hubs", {
        "hubs": [{"node": int(i), "lat": round(float(lat[i]), 3), "lon": round(float(lon[i]), 3),
                  "flow": round(float(node_flow[i]), 1), "near_g": bool(near_g[i])}
                 for i in hubs_sorted],
        "n_sources": int(R["n_sources"]),
        "components": comp_report,
    })
    return {"n_hubs": len(hubs_sorted), "components": {m: comp_report[m]["n_components"] for m in MODES},
            "cost_median_days": round(float(np.median(cost_d)), 2)}
