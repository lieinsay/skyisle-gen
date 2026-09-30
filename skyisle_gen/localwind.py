"""②b 岛对风的扰动的共用定义（第三批第 3 步，docs/PLAN-BATCH3.md 5.3）。

障碍场、带界位移、摩擦 / 绕流 / 尾流都在 C++ 核心里算（core/src/planet/stage4.cpp；Python 参考版删于 2026-09-30，
tag python-reference-final）；这里只留 ④ 写 band_local.npz 用的带界键。
"""
from __future__ import annotations

# 带界键：北半球四条、南半球四条（签名纬度）
EDGE_KEYS = ["eq_n", "trades_n", "calm_n", "west_n", "eq_s", "trades_s", "calm_s", "west_s"]


def edge_lats(bands: dict) -> dict[str, float]:
    e, t, c, w = (float(bands["eq_storm_top_deg"]), float(bands["trades_top_deg"]),
                  float(bands["calm_top_deg"]), float(bands["westerlies_top_deg"]))
    return {"eq_n": e, "trades_n": t, "calm_n": c, "west_n": w,
            "eq_s": -e, "trades_s": -t, "calm_s": -c, "west_s": -w}

