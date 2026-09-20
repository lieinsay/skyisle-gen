"""②b 岛对风的扰动（第三批第 3 步，docs/PLAN-BATCH3.md 5.3）。由 ④ 调用，排在 ③ 撒岛之后。

岛群是一排悬空的山（决定 3：岛体从云带顶立到山顶，气流只能绕过或翻越，没有「穿下」）。
产出都在 1° 网格上：
  obstacle   障碍场 O ∈ [0,1]：墙高 ÷ 边界层厚度 × 实心率（陆地占比）× 势力范围对格点的覆盖，再高斯抹开
  land       陆地覆盖场 L（只按陆地占比，不看高度）：热力日循环「可靠局地风」的来源
  dphi       带界位移 Δφ(lon)：每条带界附近障碍的纬向异常低通后映射成 ±几度的起伏（R11「有变化但仍是条带」）
  D          位移场 D(lat, lon)：在各带界之间线性插值，lat_eff = lat − D 就是「局部带界」坐标
  wake       尾流场：障碍下风方向指数衰减的风速亏损
全部 numpy，无随机数。
"""
from __future__ import annotations

import numpy as np

from .sphere import grid_interp

# 带界键：北半球四条、南半球四条（签名纬度）
EDGE_KEYS = ["eq_n", "trades_n", "calm_n", "west_n", "eq_s", "trades_s", "calm_s", "west_s"]


def edge_lats(bands: dict) -> dict[str, float]:
    e, t, c, w = (float(bands["eq_storm_top_deg"]), float(bands["trades_top_deg"]),
                  float(bands["calm_top_deg"]), float(bands["westerlies_top_deg"]))
    return {"eq_n": e, "trades_n": t, "calm_n": c, "west_n": w,
            "eq_s": -e, "trades_s": -t, "calm_s": -c, "west_s": -w}


def gauss_smooth(field: np.ndarray, res_deg: float, sigma_deg: float) -> np.ndarray:
    """可分离高斯平滑：经度周期、纬度反射。"""
    if sigma_deg <= 0:
        return field.copy()
    n = int(np.ceil(3.0 * sigma_deg / res_deg))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * res_deg / sigma_deg) ** 2)
    k /= k.sum()
    out = np.zeros_like(field, dtype=np.float64)
    for i, kk in enumerate(k):
        out += kk * np.roll(field, i - n, axis=1)
    pad = np.pad(out, ((n, n), (0, 0)), mode="reflect")
    out2 = np.zeros_like(out)
    for i, kk in enumerate(k):
        out2 += kk * pad[i:i + field.shape[0]]
    return out2


def obstacle_fields(isl: dict, lats: np.ndarray, lons: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """岛群 → 障碍场 O 与陆地覆盖场 L（都 ∈ [0,1]）。"""
    res = float(lats[1] - lats[0])
    nlat, nlon = lats.size, lons.size
    lat, lon = isl["lat"].astype(np.float64), isl["lon"].astype(np.float64)
    i = np.clip(np.round((lat - lats[0]) / res).astype(int), 0, nlat - 1)
    j = np.round((lon - lons[0]) / res).astype(int) % nlon
    cell_km2 = (111.19 * res) ** 2 * np.maximum(np.cos(np.radians(lats[i])), 0.05)
    cover = isl["territory_km2"].astype(np.float64) / cell_km2
    wall_frac = np.minimum(1.0, isl["wall_m"].astype(np.float64) / float(p["boundary_layer_m"]))
    lf = isl["land_frac"].astype(np.float64)
    O = np.zeros((nlat, nlon)); L = np.zeros((nlat, nlon))
    np.add.at(O, (i, j), lf * wall_frac * cover)
    np.add.at(L, (i, j), lf * cover)
    gain = float(p.get("obstacle_gain", 1.0))   # 截面份额本身很小（陆地占行星 5%），乘增益后才是「一排悬空的山」的量级
    O = np.clip(gain * gauss_smooth(O, res, float(p["smooth_deg"])), 0.0, 1.0)
    L = np.clip(gain * gauss_smooth(L, res, float(p["smooth_deg"])), 0.0, 1.0)
    return O, L


def band_displacement(O: np.ndarray, lats: np.ndarray, lons: np.ndarray, bands: dict,
                      p: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """每条带界的位移 Δφ(lon)（签名纬度：正 = 向北）与全场位移 D(lat, lon)。
    带界附近 ±window 内障碍的纬向异常 → 低通到波数 ≤ kmax → 按标准差定标到 gain 度、夹到 ±max。
    符号：障碍多处带界向极侧推（正异常 → 远离赤道），南北镜像。"""
    e = edge_lats(bands)
    nlon = lons.size
    win = float(p["shift_window_deg"]); kmax = int(p["shift_wavenumber_max"])
    gain = float(p["shift_gain_deg"]); mx = float(p["shift_max_deg"])
    dphi: dict[str, np.ndarray] = {}
    for key in EDGE_KEYS:
        rows = np.abs(lats - e[key]) <= win
        a = O[rows].mean(axis=0) if rows.any() else np.zeros(nlon)
        a = a - a.mean()
        F = np.fft.rfft(a)
        F[kmax + 1:] = 0.0
        a_lp = np.fft.irfft(F, n=nlon)
        sd = float(a_lp.std())
        d = np.clip(gain * a_lp / sd, -mx, mx) if sd > 1e-12 else np.zeros(nlon)
        dphi[key] = d * (1.0 if e[key] > 0 else -1.0)
    # 位移场：结点放在「位移后的带界」上，值 = 位移，赤道与两极为 0 → lat_eff = lat − D 在位移后的带界处恰等于原带界
    order = ["west_s", "calm_s", "trades_s", "eq_s", "eq_n", "trades_n", "calm_n", "west_n"]
    D = np.zeros((lats.size, nlon))
    for j in range(nlon):
        kl = [-90.0] + [e[k] + dphi[k][j] for k in order] + [90.0]
        kv = [0.0] + [dphi[k][j] for k in order] + [0.0]
        # 结点必须单调（带界最窄 8°，位移夹到 ±max 保证不交叉）
        D[:, j] = np.interp(lats, kl, kv)
    return dphi, D


def wake_field(O: np.ndarray, u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray,
               p: dict) -> np.ndarray:
    """尾流：沿本地风向向上游回溯 K 步采样障碍，几何衰减累加（下风侧亏损）。"""
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    spd = np.hypot(u, v)
    ok = spd > 0.5
    uh = np.where(ok, u / np.maximum(spd, 1e-9), 0.0)
    vh = np.where(ok, v / np.maximum(spd, 1e-9), 0.0)
    cosl = np.maximum(np.cos(np.radians(LAT)), 0.1)
    step = float(p["wake_step_deg"]); lam = float(p["wake_decay"])
    W = np.zeros_like(O)
    for k in range(1, int(p["wake_steps"]) + 1):
        latq = np.clip(LAT - k * step * vh, lats[0], lats[-1])
        lonq = LON - k * step * uh / cosl
        W += lam ** k * grid_interp(O, lats, lons, latq, lonq)
    return np.clip(W, 0.0, 1.0)


def perturb_wind(u1: np.ndarray, v1: np.ndarray, O: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """摩擦 + 绕流偏转 + 尾流。返回 (u, v, wake)。"""
    res = float(lats[1] - lats[0])
    cosl = np.maximum(np.cos(np.radians(lats))[:, None], 0.1)
    dOdx = (np.roll(O, -1, axis=1) - np.roll(O, 1, axis=1)) / (2.0 * res) / cosl   # 每度
    dOdy = np.gradient(O, res, axis=0)
    spd = np.hypot(u1, v1)
    kb = float(p["deflect_k"]); kf = float(p["friction_k"])
    fric = 1.0 - kf * O
    u2 = (u1 - kb * spd * dOdx) * fric
    v2 = (v1 - kb * spd * dOdy) * fric
    W = wake_field(O, u2, v2, lats, lons, p)
    kw = float(p["wake_k"])
    return u2 * (1.0 - kw * W), v2 * (1.0 - kw * W), W
