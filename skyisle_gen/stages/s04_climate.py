"""④ 气候（第三批第 3、4 步，docs/PLAN-BATCH3.md 5.3 / 5.4）。

前半 = ②b「岛对风的扰动」（C++ 的 stage4.cpp）：③ 的岛群 → 障碍场 → 带界位移 Δφ(lon) → 按局部带界重新求值 ② 的
背景风系 + G 涡旋 → 摩擦 / 绕流 / 尾流 → 扰动后的 u, v；另给出「可靠局地风」v_local（密集群岛的热力日循环，⑥ 无风惩罚用）。
后半 = 水汽追踪降水（stage4.cpp）：E − P 收支沿扰动后的风推进到稳态，雨影与纬度雨带都是算出来的（决定 4）；
温度（纬度 + 直减率）、风暴（按局部带界的剪切带 + G，岛群略耗散）、季节窗口、集雨容量、河流（第三批 1）。
产物：wind_local.npz（⑤⑥⑦ 与操作台从这里读风，不再读 ②）、band_local.npz（局部带界）、climate_grid.npz、climate_islands.npz。
由 C++ 核心（core/src/planet/stage4.cpp；行星计划 P6c，Python 参考版删于 2026-09-30，tag python-reference-final）算，四个 npz 与摘要照旧由这里写（_write）。
"""
from __future__ import annotations

import numpy as np

from ..localwind import EDGE_KEYS, edge_lats
from ..skeleton import year_days


def run(ctx):
    from ..engine import core, part, planet_config, put_part
    cc = core()
    Cl = cc.planet_stage4(cc.make_config(planet_config(ctx.cfg)), int(ctx.seed), part(ctx, 1), part(ctx, 2), part(ctx, 3))
    put_part(ctx, 4, Cl)
    R = cc.climate_arrays(Cl)
    R["moisture"] = {"dt_s": round(R.pop("dt_s"), 1), "n_steps": int(R.pop("n_steps"))}
    return _write(ctx, R)


def _write(ctx, R: dict) -> dict:
    """写 wind_local / band_local / climate_grid / climate_islands 四个 npz、出摘要（R 里的浮点是双精度原值，这里照旧转 float32）。"""
    f32 = np.float32
    ctx.save_npz(4, "wind_local", lats=R["lats"], lons=R["lons"],
                 u=R["u"].astype(f32), v=R["v"].astype(f32),
                 u_bg=R["u_bg"].astype(f32), v_bg=R["v_bg"].astype(f32),
                 v_local=R["v_local"].astype(f32), obstacle=R["obstacle"].astype(f32),
                 land=R["land"].astype(f32), wake=R["wake"].astype(f32),
                 lat_eff=R["lat_eff"].astype(f32), band=R["band"].astype(np.int16))
    ctx.save_npz(4, "band_local", lons=R["lons"], edges=R["edges"].astype(f32),
                 keys=np.array(EDGE_KEYS), dphi=R["dphi"].astype(f32))
    ctx.save_npz(4, "climate_grid", lats=R["lats"], lons=R["lons"],
                 precip=R["precip"].astype(f32), temp=R["temp"].astype(f32),
                 storm=R["storm"].astype(f32), storm_no_g=R["storm_no_g"].astype(f32),
                 stability=R["stability"].astype(f32),
                 window=R["window"].astype(f32),
                 q=R["q"].astype(f32), uplift=R["uplift"].astype(f32),
                 conv=R["conv"].astype(f32), eps=R["eps"].astype(f32),
                 season_range=R["season_range"].astype(f32), continentality=R["continentality"].astype(f32),
                 lift=R["lift"].astype(f32), precip_share=R["precip_share"].astype(f32),
                 season_shift=R["season_shift"].astype(f32))
    I = R["islands"]
    ctx.save_npz(4, "climate_islands",
                 precip=I["precip"].astype(f32), temp=I["temp"].astype(f32),
                 storm=I["storm"].astype(f32), stability=I["stability"].astype(f32),
                 window=I["window"].astype(f32), catch=I["catch"].astype(f32),
                 has_river=I["has_river"], river_size=I["river_size"].astype(f32),
                 temp_sea=I["temp_sea"].astype(f32), season_range_sea=I["season_range_sea"].astype(f32),
                 season_range=I["season_range"].astype(f32),
                 temp_winter=I["temp_winter"].astype(f32),
                 temp_summer=I["temp_summer"].astype(f32),
                 precip_share=I["precip_share"].astype(f32),
                 precip_mm=I["precip_mm"].astype(f32), precip_eff_mm=I["precip_eff_mm"].astype(f32),
                 arable_frac_eff=I["arable_frac_eff"].astype(f32))
    planet = ctx.load_json(1, "planet")
    e0 = edge_lats(planet["bands"])
    precip, catch, has_river = R["precip"], I["catch"], I["has_river"]
    shift_amp = float(np.max(np.abs(R["edges"] - np.array([e0[k] for k in EDGE_KEYS])[:, None])))
    ck = ctx.cfg.get("check", {})
    isl = ctx.load_npz(3, "islands")
    area, ar = isl["area_km2"].astype(np.float64), isl["arable_frac"].astype(np.float64)
    mm = I["precip_mm"]
    extra = {}
    V = planet.get("vertical") or {}
    if V:                                            # 岛面（台面 height_m）的气压：p = p0 · exp(−z / H)（PLAN-NATURE A1，只进摘要）
        h = isl["height_m"].astype(np.float64)
        p = float(V["datum_pressure_atm"]) * np.exp(-h / (1000.0 * float(V["scale_height_km"])))
        extra["surface_pressure_atm"] = [round(float(p.min()), 3), round(float(np.median(p)), 3), round(float(p.max()), 3)]
    return extra | {"precip_range": [round(float(precip.min()), 2), round(float(precip.max()), 2)],
            "arid_island_share": round(float((mm < float(ck.get("arid_mm", 400.0))).mean()), 3),
            "precip_mm_median": round(float(np.median(mm)), 0),
            "land_below_mm": {str(t): round(float(area[mm < t].sum() / area.sum()), 3) for t in (200, 400)},
            "arable_eff_ratio": round(float((area * I["arable_frac_eff"]).sum() / (area * ar).sum()), 3),
            "storm_max": round(float(R["storm"].max()), 2),
            "band_shift_max_deg": round(shift_amp, 2),
            "season_shift_deg": [round(float(x), 2) for x in R["season_shift"]],
            "obstacle_max": round(float(R["obstacle"].max()), 2),
            "wind_speed_median": round(float(np.median(np.hypot(R["u"], R["v"]))), 2),
            "moisture": R["moisture"],
            "catch_median": round(float(np.median(catch)), 1),
            "river_share": round(float(has_river.mean()), 3),
            "year_days": year_days(planet),
            "season_range_island_median": round(float(np.median(I["season_range"])), 1)}
