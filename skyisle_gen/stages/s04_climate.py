"""④ 气候（第三批第 3、4 步，docs/PLAN-BATCH3.md 5.3 / 5.4）。

前半 = ②b「岛对风的扰动」（localwind.py）：③ 的岛群 → 障碍场 → 带界位移 Δφ(lon) → 按局部带界重新求值 ② 的
背景风系 + G 涡旋 → 摩擦 / 绕流 / 尾流 → 扰动后的 u, v；另给出「可靠局地风」v_local（密集群岛的热力日循环，⑥ 无风惩罚用）。
后半 = 水汽追踪降水（moisture.py）：E − P 收支沿扰动后的风推进到稳态，雨影与纬度雨带都是算出来的（决定 4）；
温度（纬度 + 直减率）、风暴（按局部带界的剪切带 + G，岛群略耗散）、季节窗口、集雨容量、河流（第三批 1）。
产物：wind_local.npz（⑤⑥⑦ 与操作台从这里读风，不再读 ②）、band_local.npz（局部带界）、climate_grid.npz、climate_islands.npz。
cpp 后端（行星计划 P6c）：由 C++ 核心（core/src/planet/stage4.cpp）算，四个 npz 与摘要照旧由这里写（_write，两个后端共用）。
"""
from __future__ import annotations

import numpy as np

from ..engine import backend
from ..localwind import EDGE_KEYS, band_displacement, edge_lats, obstacle_fields, perturb_wind
from ..moisture import run_on_coarse
from ..noise import fractal_noise
from ..rng import stage_rng
from ..skeleton import season_range, year_days
from ..sphere import angdist, grid_axes, grid_interp, latlon_to_xyz
from ..tectonics import _divergence
from .s02_wind import band_id_of, g_vortex, wind_profile


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def run(ctx):
    if backend(ctx.cfg) == "cpp":
        from ..engine import core, part, planet_config, put_part
        cc = core()
        Cl = cc.planet_stage4(cc.make_config(planet_config(ctx.cfg)), int(ctx.seed), part(ctx, 1), part(ctx, 2), part(ctx, 3))
        put_part(ctx, 4, Cl)
        R = cc.climate_arrays(Cl)
        R["moisture"] = {"dt_s": round(R.pop("dt_s"), 1), "n_steps": int(R.pop("n_steps"))}
        return _write(ctx, R)
    c = ctx.section(4)["climate"]
    lw = ctx.section(4)["localwind"]
    w2 = ctx.section(2)["wind"]
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    g_info = ctx.load_json(2, "bands")["G"]
    isl = ctx.load_npz(3, "islands")
    res = float(ctx.cfg["shared"]["grid_res_deg"])
    lats, lons = grid_axes(res)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    # ---------- ②b 岛对风的扰动 ----------
    O, L = obstacle_fields(isl, lats, lons, lw)
    dphi, D = band_displacement(O, lats, lons, bands, lw)
    lat_eff = LAT - D                                       # 局部带界坐标（R11：带界变成波状线）
    u_bg, v_bg = wind_profile(lat_eff, w2, bands)
    du, dv = g_vortex(LAT, LON, float(g_info["lat"]), float(g_info["lon"]),
                      float(g_info["radius_deg"]), float(w2["g_vortex_speed"]))
    u, v, wake = perturb_wind(u_bg + du, v_bg + dv, O, lats, lons, lw)
    v_local = float(lw["local_wind_max"]) * np.clip(L / float(lw["local_land_ref"]), 0.0, 1.0)
    e0 = edge_lats(bands)
    edges = np.stack([e0[k] + dphi[k] for k in EDGE_KEYS])   # [8, nlon] 签名纬度
    band_local = {"lons": lons, "edges": edges, "keys": np.array(EDGE_KEYS)}
    band_grid = band_id_of(LAT, LON, bands, band_local)

    # ---------- 温度（°C，海面）：日照驱动，按真实纬度 ----------
    a = np.abs(LAT)
    t = (float(c["temp_eq_c"]) - (float(c["temp_eq_c"]) - float(c["temp_pole_c"]))
         * np.sin(np.radians(a)) ** 2) * float(planet["insolation_rel"])

    # ---------- 风暴强度 0..1：赤道带 + 带间剪切 + 中纬风暴带（都按局部带界）+ G；岛群略耗散 ----------
    ae = np.abs(lat_eff)
    storm = float(c["storm_eq_amp"]) * _gauss(ae, 0.0, 0.8 * bands["eq_storm_top_deg"])
    shear_edges = (bands["eq_storm_top_deg"], bands["trades_top_deg"],
                   bands["calm_top_deg"], bands["westerlies_top_deg"])
    amps = c["storm_shear_amp"]
    amps = [float(amps)] * 4 if not isinstance(amps, (list, tuple)) else [float(x) for x in amps]
    for sl, amp in zip(shear_edges, amps):     # 骨架第二版：按带界分别给幅度（无风带顶的副热带急流穿过核心，压低）
        storm = storm + amp * _gauss(ae, sl, float(c["storm_shear_width_deg"]))
    mid_c = (float(c["storm_midlat_lat_deg"]) * float(planet.get("band_scale", 1.0))
             if "storm_midlat_lat_deg" in c else 0.5 * (bands["calm_top_deg"] + bands["westerlies_top_deg"]))
    storm = storm + float(c["storm_midlat_amp"]) * _gauss(ae, mid_c, float(c["storm_midlat_width_deg"]))
    storm = storm * (1.0 - float(lw["storm_k"]) * O)         # 二阶小量：永暴带不能被岛打散
    storm_no_g = np.clip(storm, 0.0, 1.0)                    # 反事实：从未有过 G 的风暴场（P6 用）
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    pts = latlon_to_xyz(LAT.ravel(), LON.ravel()).reshape(LAT.shape + (3,))
    dg = np.degrees(angdist(pts, g_xyz[None, None, :]))
    storm = storm + float(c["storm_g_amp"]) * _gauss(dg, 0.0, 1.2 * float(g_info["radius_deg"]))
    storm = np.clip(storm, 0.0, 1.0)
    stability = 1.0 - storm

    # ---------- 降水：上风水汽追踪 ----------
    E = np.exp(float(c["evap_temp_coeff"]) * (t - float(c["evap_temp_ref_c"])))
    div = _divergence(u, v, lats, lons)
    midrows = np.abs(LAT) < 80.0
    sd = float(div[midrows].std()) or 1.0
    conv = np.clip(-div / sd, -2.0, 2.0)                     # 标准化辐合（正 = 辐合 = 抬升）
    cosl = np.maximum(np.cos(np.radians(lats))[:, None], 0.1)
    dOdx = (np.roll(O, -1, axis=1) - np.roll(O, 1, axis=1)) / (2.0 * res) / cosl
    dOdy = np.gradient(O, res, axis=0)
    up_raw = np.maximum(0.0, u * dOdx + v * dOdy)             # 撞墙抬升（迎风坡）
    up_ref = float(np.quantile(up_raw[up_raw > 0], 0.98)) if (up_raw > 0).any() else 1.0
    uplift = np.clip(up_raw / max(up_ref, 1e-9), 0.0, 1.0)
    eps = (np.exp(float(c["precip_conv_k"]) * conv)
           * (1.0 + float(c["precip_uplift_k"]) * uplift + float(c["precip_storm_k"]) * storm))
    q, P, info = run_on_coarse(u, v, E, eps, lats, lons, c)
    P_ref = float(np.quantile(P, float(c["precip_norm_pct"]) / 100.0))
    precip = P / max(P_ref, 1e-12)
    if float(c.get("precip_noise_amp", 0.0)) > 0:
        rng = stage_rng(ctx.seed, 4)
        precip = precip + float(c["precip_noise_amp"]) * fractal_noise(rng, lats.size, lons.size, base_cells=6, octaves=3)
    precip = np.clip(precip, 0.02, 1.0)
    q_norm = np.clip(q / max(float(np.quantile(q, 0.98)), 1e-12), 0.0, 1.0)

    # ---------- 季节窗口（可航季节比例；由倾角驱动的带摆动近似）----------
    tilt = float(planet["axial_tilt_deg"])
    window = np.clip(1.0 - 0.85 * storm - 0.004 * tilt * _gauss(ae, mid_c, 12.0), 0.03, 1.0)

    # ---------- 季节强度（骨架第二版 §4.2）：全年温差 = 日照年变化 / λ × 热惯性振幅保留 ----------
    # 陆地性 = 区域陆地覆盖（L 去掉障碍增益，≈ 1° 格内陆地占比的抹开值）。⑦ 的谷物门槛只读这一份（不含岛高，原则乙）；
    # 岛群产物另给含高度修正的「岛上」全年温差（岛在云带之上，越高越脱离洋面调节），只供九格表与岛群生成器
    ydays = year_days(planet)
    tilt_deg = float(planet["axial_tilt_deg"])
    cont = np.clip(L / max(1e-9, float(lw.get("obstacle_gain", 1.0))), 0.0, 1.0)
    season = season_range(lats[:, None], cont, tilt_deg, ydays, c, float(planet["insolation_rel"]))

    lat_i, lon_i = isl["lat"], isl["lon"]
    cont_i = grid_interp(cont, lats, lons, lat_i, lon_i)
    season_sea_i = season_range(lat_i, cont_i, tilt_deg, ydays, c, float(planet["insolation_rel"]))
    keel = float(ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    cont_alt_i = np.clip(cont_i + float(c.get("season_alt_continentality", 0.0))
                         * np.clip((isl["height_m"].astype(np.float64) - keel) / 2000.0, 0.0, 1.0), 0.0, 1.0)
    season_i = season_range(lat_i, cont_alt_i, tilt_deg, ydays, c, float(planet["insolation_rel"]))
    temp_sea_i = grid_interp(t, lats, lons, lat_i, lon_i)
    precip_i = grid_interp(precip, lats, lons, lat_i, lon_i)
    storm_i = grid_interp(storm, lats, lons, lat_i, lon_i)
    stability_i = grid_interp(stability, lats, lons, lat_i, lon_i)
    window_i = grid_interp(window, lats, lons, lat_i, lon_i)
    temp_i = (grid_interp(t, lats, lons, lat_i, lon_i)
              - float(c["lapse_c_per_km"]) * isl["height_m"] / 1000.0)

    # ---- 集雨容量（docs/02 §六）：catch = 可用地率 × 群陆地 × 降水（只用陆地、可用地率与降水，不含高度）----
    catch = (isl["arable_frac"].astype(np.float64) * isl["area_km2"].astype(np.float64)
             * precip_i)
    # ---- 河流（第三批第 1 步，PLAN-BATCH3 5.5；决定 1）----
    main_area = isl["main_area_km2"].astype(np.float64)
    has_river = ((main_area >= float(c["river_main_area_km2"]))
                 & (isl["height_m"].astype(np.float64) >= float(c["river_height_m"]))
                 & (precip_i >= float(c["river_precip_min"])))
    river_size = np.where(has_river, main_area * precip_i, 0.0)
    bonus = float(c["river_capacity_bonus"])
    if bonus > 0 and has_river.any():
        catch = catch * (1.0 + bonus * river_size / float(np.median(river_size[has_river])))

    return _write(ctx, {
        "lats": lats, "lons": lons, "u": u, "v": v, "u_bg": u_bg + du, "v_bg": v_bg + dv, "v_local": v_local, "obstacle": O, "land": L,
        "wake": wake, "lat_eff": lat_eff, "band": band_grid, "edges": edges, "dphi": np.stack([dphi[k] for k in EDGE_KEYS]),
        "precip": precip, "temp": t, "storm": storm, "storm_no_g": storm_no_g, "stability": stability, "window": window, "q": q_norm,
        "uplift": uplift, "conv": conv, "eps": eps, "season_range": season, "continentality": cont,
        "islands": {"precip": precip_i, "temp": temp_i, "storm": storm_i, "stability": stability_i, "window": window_i, "catch": catch,
                    "has_river": has_river, "river_size": river_size, "temp_sea": temp_sea_i, "season_range_sea": season_sea_i,
                    "season_range": season_i, "temp_winter": temp_i - 0.5 * season_i, "temp_summer": temp_i + 0.5 * season_i},
        "moisture": info})


def _write(ctx, R: dict) -> dict:
    """写 wind_local / band_local / climate_grid / climate_islands 四个 npz、出摘要（两个后端共用；R 里的浮点是双精度原值，这里照旧转 float32）。"""
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
                 season_range=R["season_range"].astype(f32), continentality=R["continentality"].astype(f32))
    I = R["islands"]
    ctx.save_npz(4, "climate_islands",
                 precip=I["precip"].astype(f32), temp=I["temp"].astype(f32),
                 storm=I["storm"].astype(f32), stability=I["stability"].astype(f32),
                 window=I["window"].astype(f32), catch=I["catch"].astype(f32),
                 has_river=I["has_river"], river_size=I["river_size"].astype(f32),
                 temp_sea=I["temp_sea"].astype(f32), season_range_sea=I["season_range_sea"].astype(f32),
                 season_range=I["season_range"].astype(f32),
                 temp_winter=I["temp_winter"].astype(f32),
                 temp_summer=I["temp_summer"].astype(f32))
    planet = ctx.load_json(1, "planet")
    e0 = edge_lats(planet["bands"])
    precip, precip_i, catch, has_river = R["precip"], I["precip"], I["catch"], I["has_river"]
    shift_amp = float(np.max(np.abs(R["edges"] - np.array([e0[k] for k in EDGE_KEYS])[:, None])))
    return {"precip_range": [round(float(precip.min()), 2), round(float(precip.max()), 2)],
            "arid_island_share": round(float((precip_i < float(ctx.cfg.get("check", {}).get("arid_precip", 0.3))).mean()), 3),
            "storm_max": round(float(R["storm"].max()), 2),
            "band_shift_max_deg": round(shift_amp, 2),
            "obstacle_max": round(float(R["obstacle"].max()), 2),
            "wind_speed_median": round(float(np.median(np.hypot(R["u"], R["v"]))), 2),
            "moisture": R["moisture"],
            "catch_median": round(float(np.median(catch)), 1),
            "river_share": round(float(has_river.mean()), 3),
            "year_days": year_days(planet),
            "season_range_island_median": round(float(np.median(I["season_range"])), 1)}
