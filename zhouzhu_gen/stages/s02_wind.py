"""② 大气环流：纯行星风系（纬度函数）+ 定点永暴 G 的局部气旋。

全球皆海洋 → 无大陆热力差异 → 风系极规则（docs/11 第二节）。这是「未扰动基态」；
岛群对风的扰动与带界起伏在 ④ 前半（②b，localwind.py）里做，那里会用本模块的 wind_profile 按
局部带界坐标 lat_eff 重新求值。
带结构（北半球，南镜像）：赤道永暴 / 信风(东风) / 副热带无风 / 西风 / 极地东风。
经向分量：信风表层向赤道（哈德莱环流）、西风带表层向极（费雷尔环流）、极地东风向赤道（极地环流）
→ 副热带辐散（下沉、干）、中纬辐合（锋面、湿）、极区辐散（干）。这些是 ④ 水汽模型里雨带的来源。
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
    """全局带界（纯纬度）。"""
    a = np.abs(lat)
    north = lat >= 0
    tier = np.select(
        [a < bands["eq_storm_top_deg"], a < bands["trades_top_deg"],
         a < bands["calm_top_deg"], a < bands["westerlies_top_deg"]],
        [0, 1, 2, 3], default=4)
    return np.where(tier == 0, 0, np.where(north, tier, tier + 4))


def local_edges(band_local: dict, lon) -> dict[str, np.ndarray]:
    """④ 产物 band_local（lons、edges[8, nlon]、keys）按经度插值 → 每个点的八条局部带界（签名纬度）。"""
    lons = band_local["lons"].astype(np.float64)
    keys = [str(k) for k in band_local["keys"]]
    lon = np.asarray(lon, dtype=np.float64)
    res = float(lons[1] - lons[0])
    fj = (lon - lons[0]) / res
    j0 = np.floor(fj).astype(int)
    t = fj - j0
    n = lons.size
    j0 %= n
    j1 = (j0 + 1) % n
    out = {}
    for k, key in enumerate(keys):
        row = band_local["edges"][k].astype(np.float64)
        out[key] = row[j0] * (1 - t) + row[j1] * t
    return out


def band_id_of(lat: np.ndarray, lon: np.ndarray, bands: dict, band_local: dict | None = None) -> np.ndarray:
    """带号；给了 band_local 就用随经度起伏的局部带界（R11），否则退回纯纬度。"""
    if band_local is None:
        return band_id_of_lat(lat, bands)
    e = local_edges(band_local, lon)
    lat = np.asarray(lat, dtype=np.float64)
    north = lat >= 0
    eq = np.where(north, e["eq_n"], -e["eq_s"])
    tr = np.where(north, e["trades_n"], -e["trades_s"])
    ca = np.where(north, e["calm_n"], -e["calm_s"])
    we = np.where(north, e["west_n"], -e["west_s"])
    a = np.abs(lat)
    tier = np.select([a < eq, a < tr, a < ca, a < we], [0, 1, 2, 3], default=4)
    return np.where(tier == 0, 0, np.where(north, tier, tier + 4))


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def wind_profile(lat_eff: np.ndarray, w: dict, bands: dict) -> tuple[np.ndarray, np.ndarray]:
    """背景风系（不含 G）：按（可能已按局部带界修正的）纬度坐标求值。返回 (u 东正, v 北正)。"""
    eq_top = bands["eq_storm_top_deg"]
    tr_top = bands["trades_top_deg"]
    calm_top = bands["calm_top_deg"]
    west_top = bands["westerlies_top_deg"]
    a = np.abs(lat_eff)
    sign = np.sign(lat_eff + 1e-12)  # 北 +1，南 −1
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
    # 经向风 v（北正）：信风带表层向赤道；西风带表层向极；极地东风向赤道
    v = (-sign * float(w["hadley_inflow_speed"]) * _gauss(a, tr_c, tr_w)
         + sign * float(w.get("ferrel_outflow_speed", 0.0)) * _gauss(a, we_c, we_w)
         - sign * float(w.get("polar_outflow_speed", 0.0)) * _gauss(a, po_c, po_w))
    return u, v


def g_vortex(LAT: np.ndarray, LON: np.ndarray, g_lat: float, g_lon: float, g_r_deg: float,
             vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """定点永暴 G：Rankine 型切向速度（北半球气旋逆时针）。返回 (du, dv)。"""
    g_r = np.radians(g_r_deg)
    g_xyz = latlon_to_xyz(np.array(g_lat), np.array(g_lon))
    pts = latlon_to_xyz(LAT.ravel(), LON.ravel()).reshape(LAT.shape + (3,))
    d = angdist(pts, g_xyz[None, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = np.where(d < g_r, vmax * d / g_r, vmax * np.exp(-(d - g_r) / (1.5 * g_r)))
    dlat = LAT - g_lat
    dlon = (LON - g_lon + 180.0) % 360.0 - 180.0
    theta = np.arctan2(np.radians(dlat), np.radians(dlon) * np.cos(np.radians(g_lat)))
    return vt * (-np.sin(theta)), vt * np.cos(theta)


def run(ctx):
    w = ctx.section(2)["wind"]
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    sk = ctx.cfg["skeleton"]
    res = float(ctx.cfg["shared"]["grid_res_deg"])
    lats, lons = grid_axes(res)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    u, v = wind_profile(LAT, w, bands)

    # ---- 定点永暴 G：位置由剪切纬度派生（决策 3）----
    tr_top = bands["trades_top_deg"]
    g_lat = tr_top - float(sk["g_delta_deg"])
    g_lon = 0.5 * (float(sk["d_lon_west"]) + float(sk["d_lon_east"]))
    du, dv = g_vortex(LAT, LON, g_lat, g_lon, float(sk["g_radius_deg"]), float(w["g_vortex_speed"]))
    u = u + du
    v = v + dv

    band_grid = band_id_of_lat(LAT, bands)
    eq_top, calm_top, west_top = bands["eq_storm_top_deg"], bands["calm_top_deg"], bands["westerlies_top_deg"]
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
