"""5.3 水系、地表与可耕地：地表分类码、局地雨的摘要、build_hydro 入口。

D8 流向、填洼、汇流 × 局地雨、河 / 溪涧 / 湖 / 盆地、地表分类、湿地、上等地与宜垦都在 C++ 核心里（core/src/island/hydro.cpp、river.cpp；
Python 参考版删于 2026-09-30，tag python-reference-final）。
"""
from __future__ import annotations

import numpy as np

LC_VOID, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE = range(12)


def build_hydro(ctx, node: int, c: dict, g: dict, log=print) -> None:
    """第 2 步（C++）：在 build_terrain 的 g 上加水系、地表、上等地、宜垦。只跑前两步的工具（粗版）用；整群生成走 island.generate。"""
    from .engine import build_hydro_cpp
    return build_hydro_cpp(ctx, node, c, g, log=log)


def local_precip_summary(rain_mm: np.ndarray, island_id: np.ndarray, hc: dict) -> dict:
    """island.json 的 hydro.local_precip：主岛局地雨的最少 / 最多 / 平均与分位（mm），全群平均（= 行星层的年降水）。"""
    on = float(hc.get("oro_rise_per_km", 0.0)) != 0.0 or float(hc.get("windward_gain", 0.0)) != 0.0
    m = rain_mm[island_id == 0].astype(np.float64)
    a = rain_mm[island_id >= 0].astype(np.float64)
    q = np.quantile(m, [0.1, 0.5, 0.9]) if m.size else [0.0, 0.0, 0.0]
    return {"on": on, "group_mean_mm": round(float(a.mean()), 0) if a.size else 0.0,
            "main_min_mm": round(float(m.min()), 0) if m.size else 0.0, "main_max_mm": round(float(m.max()), 0) if m.size else 0.0,
            "main_p10_p50_p90_mm": [round(float(x), 0) for x in q],
            "note": "局地年降水（terrain.npz 的 rain_mm）：海拔 × 山脉尺度的迎风坡，按全群陆地均值归一（群的雨总量是行星层给的）；河的流量、地表湿度、湿地按它算"}
