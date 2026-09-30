"""② 大气环流：纯行星风系（纬度函数）+ 定点永暴 G 的局部气旋。

全球皆海洋 → 无大陆热力差异 → 风系极规则（docs/11 第二节）。这是「未扰动基态」；
岛群对风的扰动与带界起伏在 ④ 前半（②b，C++ 的 stage4.cpp）里做，那里会用本模块的 wind_profile 按
局部带界坐标 lat_eff 重新求值。
带结构（北半球，南镜像）：赤道永暴 / 信风(东风) / 副热带无风 / 西风 / 极地东风。
经向分量：信风表层向赤道（哈德莱环流）、西风带表层向极（费雷尔环流）、极地东风向赤道（极地环流）
→ 副热带辐散（下沉、干）、中纬辐合（锋面、湿）、极区辐散（干）。这些是 ④ 水汽模型里雨带的来源。
由 C++ 核心（core/src/planet/stage12.cpp；行星计划 P6c，Python 参考版删于 2026-09-30，tag python-reference-final）算，wind.npz / bands.json 照旧由这里写。
"""
from __future__ import annotations

import numpy as np

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


def run(ctx):
    from ..engine import core, part, planet_config, put_part
    c = core()
    W = c.planet_stage2(c.make_config(planet_config(ctx.cfg)), part(ctx, 1))
    put_part(ctx, 2, W)
    a = c.winds_arrays(W)
    return _write(ctx, a["lats"], a["lons"], a["u"], a["v"], a["band"], a["g_lat"], a["g_lon"], a["g_edge"])


def _write(ctx, lats, lons, u, v, band_grid, g_lat: float, g_lon: float, g_edge: str) -> dict:
    """写 wind.npz / bands.json、出摘要。"""
    bands = ctx.load_json(1, "planet")["bands"]
    sk = ctx.cfg["skeleton"]
    tr_top = bands["trades_top_deg"]
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
              "derived_from": f"{g_edge}({bands[g_edge]:.1f}) - delta({sk['g_delta_deg']})"},
    })
    return {"G_lat": round(g_lat, 1), "G_lon": round(g_lon, 1),
            "u_range": [round(float(u.min()), 1), round(float(u.max()), 1)]}
