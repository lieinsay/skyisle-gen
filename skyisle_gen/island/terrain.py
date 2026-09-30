"""5.2 岛内地形：island.json 的多核岛记录（cores_json）。

岛形、基形（拱 / 脊 / 台地、多核嵌合）、分形噪声、侵蚀（谷收拢）、崖缘都在 C++ 核心里（core/src/island/terrain.cpp；
Python 参考版删于 2026-09-30，tag python-reference-final）。高程是零点（浮层基准）以上的绝对高度（m）：
岛底 keel（≈ keel_clearance_m，薄片岛更低）→ 岸缘 rim（岸线上的地面高度，崖高 = rim − keel）→ 峰 peak。
"""
from __future__ import annotations


def cores_json(cores: list[dict], gc: tuple[float, float], fl: float, res_km: float, c: dict) -> list[dict]:
    """island.json 的 islands[].cores（多核岛）：局部栅格 km → 群坐标（加局部栅格中心 gc）、浮高 fl 加进峰高。
    根深 = core_root_ratio × 核上陆地高出岸缘的平均（山高根深：每个核的根在它的载荷中心正下方，浮力中心 = 重心，拼起来不歪）。"""
    ratio = float(c.get("core_root_ratio", 5.0))
    out = []
    for k, e in enumerate(cores):
        out.append({"id": k, "primary": bool(e["strength"] == 1.0), "strength": round(float(e["strength"]), 3),
                    "seed_km": [round(float(gc[0] + e["seed"][0]), 3), round(float(gc[1] + e["seed"][1]), 3)],
                    "center_km": [round(float(gc[0] + e["load_xy"][0]), 3), round(float(gc[1] + e["load_xy"][1]), 3)],
                    "area_km2": round(float(e["cells"]) * res_km * res_km, 3),
                    "peak_m": round(float(e["peak"]) + fl, 1),
                    "mean_above_rim_m": round(float(e["mean_above"]), 1),
                    "root_depth_m": round(ratio * float(e["mean_above"]), 1)})
    return out
