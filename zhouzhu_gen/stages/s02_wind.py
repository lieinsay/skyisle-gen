"""② 大气环流：纯行星风系（纬度函数）+ 定点永暴 G 的局部气旋。

全球皆海洋 → 无大陆热力差异 → 风系极规则（docs/11 第二节）。
带结构（北半球，南镜像）：赤道永暴 / 信风(东风) / 副热带无风 / 西风 / 极地东风。
"""
from __future__ import annotations

import numpy as np

from ..sphere import grid_axes, latlon_to_xyz, angdist

BAND_NAMES = [
    "equatorial_storm",              # 0
    "trades_n", "subtropical_calm_n", "westerlies_n", "polar_n",   # 1-4
    "trades_s", "subtropical_calm_s", "westerlies_s", "polar_s",   # 5-8
]


def band_id_of_lat(lat: np.ndarray, bands: dict) -> np.ndarray:
    a = np.abs(lat)
    out = np.zeros(lat.shape, dtype=np.int64)
    north = lat >= 0
    tier = np.select(
        [a < bands["eq_storm_top_deg"], a < bands["trades_top_deg"],
         a < bands["calm_top_deg"], a < bands["westerlies_top_deg"]],
        [0, 1, 2, 3], default=4)
    out = np.where(tier == 0, 0, np.where(north, tier, tier + 4))
    return out


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def run(ctx):
    w = ctx.section(2)["wind"]
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    sk = ctx.cfg["skeleton"]
    res = float(ctx.cfg["shared"]["grid_res_deg"])
    lats, lons = grid_axes(res)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    eq_top = bands["eq_storm_top_deg"]
    tr_top = bands["trades_top_deg"]
    calm_top = bands["calm_top_deg"]
    west_top = bands["westerlies_top_deg"]

    a = np.abs(LAT)
    sign = np.sign(LAT + 1e-12)  # 北 +1，南 −1
    tr_c = 0.5 * (eq_top + tr_top)
    tr_w = 0.35 * (tr_top - eq_top)
    we_c = 0.5 * (calm_top + west_top)
    we_w = 0.30 * (west_top - calm_top)
    po_c = 0.5 * (west_top + 90.0)
    po_w = 0.35 * (90.0 - west_top)

    # 纬向风 u（东正）：信风与极地东风为负，西风为正
    u = (-float(w["trade_speed"]) * _gauss(a, tr_c, tr_w)
         + float(w["westerly_speed"]) * _gauss(a, we_c, we_w)
         - float(w["polar_speed"]) * _gauss(a, po_c, po_w))
    # 经向风 v（北正）：信风带表层向赤道
    v = -sign * float(w["hadley_inflow_speed"]) * _gauss(a, tr_c, tr_w)

    # ---- 定点永暴 G：位置由剪切纬度派生（决策 3）----
    g_lat = tr_top - float(sk["g_delta_deg"])
    g_lon = 0.5 * (float(sk["d_lon_west"]) + float(sk["d_lon_east"]))
    g_r = np.radians(float(sk["g_radius_deg"]))
    g_xyz = latlon_to_xyz(np.array(g_lat), np.array(g_lon))
    pts = latlon_to_xyz(LAT.ravel(), LON.ravel()).reshape(LAT.shape + (3,))
    d = angdist(pts, g_xyz[None, None, :])
    # Rankine 型切向速度（北半球气旋逆时针）
    vmax = float(w["g_vortex_speed"])
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = np.where(d < g_r, vmax * d / g_r, vmax * np.exp(-(d - g_r) / (1.5 * g_r)))
    # 切向单位向量：绕 G 中心逆时针 = ĝ × r̂ 的切向分量，换算到东/北分量
    dlat = LAT - g_lat
    dlon = (LON - g_lon + 180.0) % 360.0 - 180.0
    theta = np.arctan2(np.radians(dlat), np.radians(dlon) * np.cos(np.radians(g_lat)))
    u = u + vt * (-np.sin(theta))
    v = v + vt * (np.cos(theta))

    band_grid = band_id_of_lat(LAT, bands)
    shear_lats = [eq_top, tr_top, calm_top, west_top,
                  -eq_top, -tr_top, -calm_top, -west_top]

    ctx.save_npz(2, "wind", lats=lats, lons=lons,
                 u=u.astype(np.float32), v=v.astype(np.float32),
                 band=band_grid.astype(np.int16))
    ctx.save_json(2, "bands", {
        "band_names": BAND_NAMES,
        "edges_deg": {k: bands[k] for k in bands},
        "shear_lats_deg": shear_lats,
        "G": {"lat": g_lat, "lon": g_lon, "radius_deg": float(sk["g_radius_deg"]),
              "derived_from": f"trades_top({tr_top:.1f}) - delta({sk['g_delta_deg']})"},
    })
    return {"G_lat": round(g_lat, 1), "G_lon": round(g_lon, 1),
            "u_range": [round(float(u.min()), 1), round(float(u.max()), 1)]}
