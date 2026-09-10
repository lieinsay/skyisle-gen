"""④ 气候：降水纬度分异、温度、风暴强度（含 G 尖峰）、稳定度、季节窗口、集雨容量。"""
from __future__ import annotations

import numpy as np

from ..noise import fractal_noise
from ..rng import stage_rng
from ..sphere import angdist, grid_axes, grid_interp, latlon_to_xyz


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def run(ctx):
    c = ctx.section(4)["climate"]
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    g_info = ctx.load_json(2, "bands")["G"]
    rng = stage_rng(ctx.seed, 4)
    res = float(ctx.cfg["shared"]["grid_res_deg"])
    lats, lons = grid_axes(res)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    a = np.abs(LAT)

    # ---- 降水（相对 0..1）：ITCZ 峰、副热带谷、中纬次峰、极地低 ----
    subtrop_c = 0.5 * (bands["trades_top_deg"] + bands["calm_top_deg"])
    precip = (float(c["precip_base"])
              + float(c["precip_itcz_amp"]) * _gauss(a, 0.0, float(c["precip_itcz_width_deg"]))
              - float(c["precip_subtrop_dip"]) * _gauss(a, subtrop_c, float(c["precip_subtrop_width_deg"]))
              + float(c["precip_midlat_amp"]) * _gauss(a, float(c["precip_midlat_center_deg"]),
                                                       float(c["precip_midlat_width_deg"]))
              - float(c["precip_polar_dip"]) * _gauss(a, 90.0, 15.0))
    noise = fractal_noise(rng, lats.size, lons.size, base_cells=6, octaves=3)
    precip = np.clip(precip + float(c["precip_noise_amp"]) * noise, 0.02, 1.0)

    # ---- 温度（°C，海面）----
    t = (float(c["temp_eq_c"]) - (float(c["temp_eq_c"]) - float(c["temp_pole_c"]))
         * np.sin(np.radians(a)) ** 2) * float(planet["insolation_rel"])

    # ---- 风暴强度 0..1：赤道带 + 带间剪切 + 中纬风暴带 + G ----
    storm = float(c["storm_eq_amp"]) * _gauss(a, 0.0, 0.8 * bands["eq_storm_top_deg"])
    for sl in (bands["eq_storm_top_deg"], bands["trades_top_deg"],
               bands["calm_top_deg"], bands["westerlies_top_deg"]):
        storm = storm + float(c["storm_shear_amp"]) * _gauss(a, sl, float(c["storm_shear_width_deg"]))
    mid_c = 0.5 * (bands["calm_top_deg"] + bands["westerlies_top_deg"])
    storm = storm + float(c["storm_midlat_amp"]) * _gauss(a, mid_c, float(c["storm_midlat_width_deg"]))
    storm_no_g = np.clip(storm, 0.0, 1.0)  # 反事实：从未有过 G 的风暴场（P6 用）
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    pts = latlon_to_xyz(LAT.ravel(), LON.ravel()).reshape(LAT.shape + (3,))
    dg = np.degrees(angdist(pts, g_xyz[None, None, :]))
    storm = storm + float(c["storm_g_amp"]) * _gauss(dg, 0.0, 1.2 * float(g_info["radius_deg"]))
    storm = np.clip(storm, 0.0, 1.0)
    stability = 1.0 - storm

    # ---- 季节窗口（可航季节比例；由倾角驱动的带摆动近似）----
    tilt = float(planet["axial_tilt_deg"])
    window = np.clip(1.0 - 0.85 * storm - 0.004 * tilt * _gauss(a, mid_c, 12.0), 0.03, 1.0)

    isl = ctx.load_npz(3, "islands")
    lat_i, lon_i = isl["lat"], isl["lon"]
    precip_i = grid_interp(precip, lats, lons, lat_i, lon_i)
    storm_i = grid_interp(storm, lats, lons, lat_i, lon_i)
    stability_i = grid_interp(stability, lats, lons, lat_i, lon_i)
    window_i = grid_interp(window, lats, lons, lat_i, lon_i)
    temp_i = (grid_interp(t, lats, lons, lat_i, lon_i)
              - float(c["lapse_c_per_km"]) * isl["height_m"] / 1000.0)

    ctx.save_npz(4, "climate_grid", lats=lats, lons=lons,
                 precip=precip.astype(np.float32), temp=t.astype(np.float32),
                 storm=storm.astype(np.float32), storm_no_g=storm_no_g.astype(np.float32),
                 stability=stability.astype(np.float32),
                 window=window.astype(np.float32))
    # ---- 集雨容量（docs/02 §六：岛上没有长期河流，淡水全靠集雨面）----
    # catch = 可用地率 × 群陆地 × 降水强度：一个岛群能养多少人的地理上限（R9：雨落在
    # 全部陆地上，但只有可耕地吃得下水与人；口径 P1 的人口密度按可耕地计）。
    # 是 ⑥ 介数源权重、⑦ 适宜度、⑨ 九格表 ⑤⑧ 的人口／政治体量代理。
    # 只用陆地、可用地率与降水，不含高度（原则乙）。
    catch = (isl["arable_frac"].astype(np.float64) * isl["area_km2"].astype(np.float64)
             * precip_i)
    # ---- 河流（第三批第 1 步，PLAN-BATCH3 5.5；决定 1：允许河，按地球常见程度）----
    # 主岛够大、够高、够湿 → 有常年河。高度在此只筛「有没有河」（同温度直减率，原则乙允许 ④ 用高度）。
    # river_size = 主岛面积 × 降水（集水面代理）。河默认只进文本：river_capacity_bonus = 0 时 catch 不变。
    main_area = isl["main_area_km2"].astype(np.float64)
    has_river = ((main_area >= float(c["river_main_area_km2"]))
                 & (isl["height_m"].astype(np.float64) >= float(c["river_height_m"]))
                 & (precip_i >= float(c["river_precip_min"])))
    river_size = np.where(has_river, main_area * precip_i, 0.0)
    bonus = float(c["river_capacity_bonus"])
    if bonus > 0 and has_river.any():
        catch = catch * (1.0 + bonus * river_size / float(np.median(river_size[has_river])))

    ctx.save_npz(4, "climate_islands",
                 precip=precip_i.astype(np.float32), temp=temp_i.astype(np.float32),
                 storm=storm_i.astype(np.float32), stability=stability_i.astype(np.float32),
                 window=window_i.astype(np.float32), catch=catch.astype(np.float32),
                 has_river=has_river, river_size=river_size.astype(np.float32))
    return {"precip_range": [round(float(precip.min()), 2), round(float(precip.max()), 2)],
            "storm_max": round(float(storm.max()), 2),
            "catch_median": round(float(np.median(catch)), 1),
            "catch_p95": round(float(np.quantile(catch, 0.95)), 1),
            "river_share": round(float(has_river.mean()), 3)}
