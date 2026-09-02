"""⑨ 输出：world.json 汇总 + 全套可视化 + 九格表草稿 + 验收报告。"""
from __future__ import annotations

import numpy as np

from .s03_islands import CLASS_NAMES


def run(ctx):
    isl = ctx.load_npz(3, "islands")
    reg = ctx.load_npz(7, "regions")
    centers = ctx.load_json(7, "centers")
    barriers = ctx.load_json(5, "barriers")
    traits = ctx.load_json(8, "traits.resolved")
    planet = ctx.load_json(1, "planet")
    bands = ctx.load_json(2, "bands")

    N = isl["lat"].size
    world = {
        "planet": planet,
        "bands": bands,
        "barriers": barriers["barriers"],
        "centers": centers["centers"],
        "n_islands": N,
        "n_regions": int(reg["region"].max()) + 1,
        "n_traits": len(traits["traits"]),
        "slots": traits["slots"],
    }
    ctx.save_json(9, "world", world)

    islands_out = [{
        "id": i,
        "lat": round(float(isl["lat"][i]), 4), "lon": round(float(isl["lon"][i]), 4),
        "class": CLASS_NAMES[int(isl["cls"][i])], "layered": bool(isl["layered"][i]),
        "region": int(reg["region"][i]),
    } for i in range(N)]
    ctx.save_json(9, "islands", islands_out)

    summary = {"world_json": True}
    try:
        from ..viz import render_all
        n_fig = render_all(ctx)
        summary["figures"] = n_fig
    except ImportError:
        summary["figures"] = "viz 模块未就绪"
    try:
        from ..ninegrid import render_ninegrids
        n_regions = render_ninegrids(ctx)
        summary["ninegrids"] = n_regions
    except ImportError:
        summary["ninegrids"] = "ninegrid 模块未就绪"
    try:
        from ..check import run_check
        code = run_check(ctx)
        summary["check_exit"] = code
    except ImportError:
        summary["check_exit"] = "check 模块未就绪"
    return summary
