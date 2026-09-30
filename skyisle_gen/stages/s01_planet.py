"""① 行星参数 → 派生常量（带界缩放、距离换算）+ 历法 ↔ 轨道自洽（R1）。

由 C++ 核心算（core/src/planet/stage12.cpp，行星计划 P6c）；planet.json 由这里写。
Python 参考版删于 2026-09-30（git tag python-reference-final，DESIGN-NOTES 四点四十二）。
"""
from __future__ import annotations

from ..engine import core, planet_config, put_part


def run(ctx):
    c = core()
    P = c.planet_stage1(c.make_config(planet_config(ctx.cfg)))
    put_part(ctx, 1, P)
    return _write(ctx, c.planet_json(P))


def _write(ctx, out: dict) -> dict:
    """写 planet.json、出摘要。"""
    cal = out["calendar"]
    ctx.save_json(1, "planet", out)
    summary = {"circumference_days": round(out["circumference_days"], 1), "band_scale": out["band_scale"]}
    if cal:
        summary.update({"year_days": round(cal["year_days_solar"], 2),
                        "star_msun": round(cal["star"]["mass_msun"], 3), "a_au": round(cal["semi_major_axis_au"], 3),
                        "tidal_lock_gyr": round(cal["tidal_lock_gyr"], 2)})
    return summary
