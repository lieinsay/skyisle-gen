"""① 行星参数 → 派生常量（带界缩放、距离换算）+ 历法 ↔ 轨道自洽（R1）+ 零点与垂直结构（PLAN-NATURE A1，planet.json 的 vertical）。

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
    V = out.get("vertical") or {}
    if V:                                            # C++ 只给数（ASCII），说明文字在这里加
        V["pressure_formula"] = "p(z) = datum_pressure_atm · exp(−z / (1000 · scale_height_km))；z 相对零点（浮层基准）、m，往上为正"
        V["temperature"] = "零点以上按 [s04.climate] lapse_c_per_km 递减；零点以下（云带是逆温层）取零点处的气温。④ 的「海面口径」（temp_sea 等）说的就是零点处"
        V["note"] = ("高度一律相对零点：零点 = 浮层基准（气压约一个大气压的高度），不是地面、不是云带顶。海面、云带、岛底最低（keel_floor_m = 云带顶 + 间隙）都在零点以下；"
                     "岛面（台面）约 +0.3…+2.2 km。说明见 docs/DATA-GUIDE.md 第一节，设定见 spec 13 第九节第 2–4 条")
    ctx.save_json(1, "planet", out)
    summary = {"circumference_days": round(out["circumference_days"], 1), "band_scale": out["band_scale"]}
    if cal:
        summary.update({"year_days": round(cal["year_days_solar"], 2),
                        "star_msun": round(cal["star"]["mass_msun"], 3), "a_au": round(cal["semi_major_axis_au"], 3),
                        "tidal_lock_gyr": round(cal["tidal_lock_gyr"], 2)})
    if V:
        summary.update({"sea_pressure_atm": round(V["sea_pressure_atm"], 2), "sea_o2_atm": round(V["sea_o2_atm"], 2)})
    return summary
