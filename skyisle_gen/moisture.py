"""上风水汽追踪（第三批第 4 步，docs/PLAN-BATCH3.md 5.4；决定 2、4）。

在经纬网格上解二维稳态水汽收支：
    ∂q/∂t + ∇·(q u) = E − P,   P = q · ε / τ
  E  蒸发源：下方全球海洋经云带上涌，暖处多（Clausius–Clapeyron 式 6 %/°C），经度方向均匀
  ε  降水效率：辐合处大（抬升）、辐散处小（副热带下沉压制）、撞墙抬升与风暴锋面再加成
雨影、赤道湿 / 副热带干 / 中纬湿 / 极地干都是算出来的，不再手画高斯项。
通量形式迎风差分，显式时间推进到稳态；极区做纬向平均滤波（否则 CFL 被极点卡死）。
为了速度在粗网格（默认 2°）上解，再双线性插回 1°。全部 numpy，无随机数。
"""
from __future__ import annotations

import numpy as np

from .sphere import grid_axes, grid_interp

R_EARTH = 6.371e6


def coarsen(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, res_run: float):
    """把 1° 场按块平均到 res_run（须为整数倍）。"""
    f = int(round(res_run / float(lats[1] - lats[0])))
    if f <= 1:
        return field.copy()
    nlat, nlon = field.shape[0] // f * f, field.shape[1] // f * f
    a = field[:nlat, :nlon].reshape(nlat // f, f, nlon // f, f)
    return a.mean(axis=(1, 3))


def solve(u: np.ndarray, v: np.ndarray, E: np.ndarray, eps: np.ndarray,
          lats: np.ndarray, lons: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (q, P, info)，都在输入网格分辨率上（调用方负责先粗化再插回）。"""
    tau = float(p["moisture_tau_days"]) * 86400.0
    polar = float(p["moisture_polar_filter_lat"])
    days = float(p["moisture_days"])
    nlat, nlon = u.shape
    phi = np.radians(lats)
    dlam = np.radians(lons[1] - lons[0])
    dphi = np.radians(lats[1] - lats[0])
    cosphi = np.maximum(np.cos(phi), 0.02)[:, None]
    # CFL：只按 |lat| ≤ polar 的格点定步长；极区靠滤波
    mid = np.abs(lats) <= polar
    cmid = np.cos(np.radians(lats[mid]))[:, None]
    umax = float(np.max(np.abs(u[mid]) / (R_EARTH * cmid * dlam))) + 1e-12
    vmax = float(np.max(np.abs(v) / (R_EARTH * dphi))) + 1e-12
    dt = 0.45 / (umax + vmax)
    dt = min(dt, tau * 0.25)
    n_steps = int(np.ceil(days * 86400.0 / dt))
    # 面上的风：东面 / 北面
    ue = 0.5 * (u + np.roll(u, -1, axis=1))
    vn = 0.5 * (v[:-1] + v[1:])
    cosn = np.cos(0.5 * (phi[:-1] + phi[1:]))[:, None]
    vnc = vn * cosn
    inv_x = 1.0 / (R_EARTH * cosphi * dlam)
    inv_y = 1.0 / (R_EARTH * cosphi * dphi)
    rate = eps / tau
    q = E * tau / np.maximum(eps, 1e-6)           # 局地平衡作初值，收敛更快
    polar_rows = np.abs(lats) > polar
    Fy = np.zeros((nlat + 1, nlon))
    for _ in range(n_steps):
        q_e = np.where(ue > 0, q, np.roll(q, -1, axis=1))
        Fx = ue * q_e
        divx = (Fx - np.roll(Fx, 1, axis=1)) * inv_x
        q_n = np.where(vn > 0, q[:-1], q[1:])
        Fy[1:-1] = vnc * q_n
        divy = (Fy[1:] - Fy[:-1]) * inv_y
        q = q + dt * (E - q * rate - divx - divy)
        q = np.maximum(q, 0.0)
        if polar_rows.any():
            q[polar_rows] = q[polar_rows].mean(axis=1, keepdims=True)
    P = q * rate
    return q, P, {"dt_s": round(dt, 1), "n_steps": n_steps}


def run_on_coarse(u, v, E, eps, lats, lons, p):
    """在粗网格上解，再插回 1°。E/eps/u/v 都是 1° 场。"""
    res_run = float(p["moisture_res_deg"])
    lats_c, lons_c = grid_axes(res_run)
    uc, vc = coarsen(u, lats, lons, res_run), coarsen(v, lats, lons, res_run)
    Ec, ec = coarsen(E, lats, lons, res_run), coarsen(eps, lats, lons, res_run)
    qc, Pc, info = solve(uc, vc, Ec, ec, lats_c, lons_c, p)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    q = grid_interp(qc, lats_c, lons_c, LAT.ravel(), LON.ravel()).reshape(LAT.shape)
    P = grid_interp(Pc, lats_c, lons_c, LAT.ravel(), LON.ravel()).reshape(LAT.shape)
    return q, P, info
