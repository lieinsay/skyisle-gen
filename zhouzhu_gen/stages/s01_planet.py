"""① 行星参数 → 派生常量（带界缩放、距离换算）+ 历法 ↔ 轨道自洽（R1，almanac.py）。"""
from __future__ import annotations

import math

from ..almanac import derive as derive_calendar


def run(ctx):
    p = ctx.section(1)["planet"]
    w2 = ctx.cfg["s02"]["wind"]
    rot = float(p["rotation_period_hr"])
    scale = 1.0
    if p.get("held_hou_scaling"):
        # Held–Hou：哈德莱圈宽度 ∝ Ω^-1（小角近似取平方根缓和）
        scale = min(1.6, max(0.5, math.sqrt(rot / 24.0)))
    bands = {
        "eq_storm_top_deg": float(w2["eq_storm_top_deg"]) * scale,
        "trades_top_deg": float(w2["trades_top_deg"]) * scale,
        "calm_top_deg": float(w2["calm_top_deg"]) * scale,
        "westerlies_top_deg": float(w2["westerlies_top_deg"]) * scale,
    }
    if bands["westerlies_top_deg"] >= 85.0:
        raise ValueError("带界缩放后西风带顶超过 85°，自转太慢，请调 s01.planet")
    out = {
        "rotation_period_hr": rot,
        "axial_tilt_deg": float(p["axial_tilt_deg"]),
        "insolation_rel": float(p["insolation_rel"]),
        "radius_km": float(p["radius_km"]),
        "band_scale": scale,
        "bands": bands,
        "day_range_km": float(ctx.cfg["shared"]["day_range_km"]),
        "circumference_days": 2 * math.pi * float(p["radius_km"]) / float(ctx.cfg["shared"]["day_range_km"]),
    }
    cal = derive_calendar(ctx.cfg)
    out["calendar"] = cal
    ctx.save_json(1, "planet", out)
    summary = {"circumference_days": round(out["circumference_days"], 1), "band_scale": scale}
    if cal:
        summary.update({"year_days": round(cal["year_days_solar"], 2),
                        "star_msun": round(cal["star"]["mass_msun"], 3), "a_au": round(cal["semi_major_axis_au"], 3),
                        "tidal_lock_gyr": round(cal["tidal_lock_gyr"], 2)})
    return summary
