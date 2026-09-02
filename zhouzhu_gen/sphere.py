"""单位球面几何。所有距离/近邻/方位一律经 xyz，避免经度回绕与极点问题。

约定：lat/lon 为度；lat ∈ [-90, 90]，lon ∈ [-180, 180)；东经为正。
"""
from __future__ import annotations

import numpy as np


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """[..., 3] 单位向量。x 轴指向 (0N, 0E)，z 轴指向北极。"""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def xyz_to_latlon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


def angdist(a_xyz: np.ndarray, b_xyz: np.ndarray) -> np.ndarray:
    """大圆角距（弧度）。数值稳定（用 arctan2 而非 arccos）。"""
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    cross = np.cross(a, b)
    sin_d = np.sqrt(np.sum(cross * cross, axis=-1))
    cos_d = np.sum(a * b, axis=-1)
    return np.arctan2(sin_d, cos_d)


def initial_bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """从点 1 到点 2 的初始方位角（弧度，0 = 正北，顺时针）。"""
    p1, l1 = np.radians(lat1), np.radians(lon1)
    p2, l2 = np.radians(lat2), np.radians(lon2)
    dl = l2 - l1
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.arctan2(y, x)


def slerp_points(a_xyz: np.ndarray, b_xyz: np.ndarray, n: int) -> np.ndarray:
    """大圆插值：a→b 之间 n 个采样点（含端点）。a, b: [E, 3] → [E, n, 3]。"""
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    omega = angdist(a, b)[..., None]  # [E, 1]
    t = np.linspace(0.0, 1.0, n)[None, :]  # [1, n]
    so = np.sin(omega)
    # 退化（同点）时线性
    with np.errstate(invalid="ignore", divide="ignore"):
        wa = np.where(so > 1e-12, np.sin((1 - t) * omega) / so, 1 - t)
        wb = np.where(so > 1e-12, np.sin(t * omega) / so, t)
    pts = wa[..., None] * a[:, None, :] + wb[..., None] * b[:, None, :]
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    return pts


def knn(xyz: np.ndarray, k: int, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """球面 kNN，分块暴力点积。返回 (idx[N,k], ang[N,k])，按 (角距, 索引) 排序（平局取小索引）。"""
    n = xyz.shape[0]
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    ang = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        dots = xyz[s:e] @ xyz.T  # [b, N]
        rows = np.arange(s, e)
        dots[np.arange(e - s), rows] = -2.0  # 排除自身
        # 取前 k：先 argpartition 再稳定排序（键 = (-dot, index)）
        part = np.argpartition(-dots, kth=k - 1, axis=1)[:, :k]
        pd = np.take_along_axis(dots, part, axis=1)
        order = np.lexsort((part, -pd), axis=1)
        top = np.take_along_axis(part, order, axis=1)
        idx[s:e] = top
        ang[s:e] = np.arccos(np.clip(np.take_along_axis(dots, top, axis=1), -1.0, 1.0))
    return idx, ang


def grid_axes(res_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """网格单元中心坐标。lats 升序（南→北），lons 升序（-180→180）。"""
    nlat = int(round(180.0 / res_deg))
    nlon = int(round(360.0 / res_deg))
    lats = -90.0 + (np.arange(nlat) + 0.5) * res_deg
    lons = -180.0 + (np.arange(nlon) + 0.5) * res_deg
    return lats, lons


def grid_interp(field: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                lat_q, lon_q) -> np.ndarray:
    """双线性插值，经度周期。field: [nlat, nlon]。"""
    lat_q = np.asarray(lat_q, dtype=np.float64)
    lon_q = np.asarray(lon_q, dtype=np.float64)
    nlat, nlon = field.shape
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    fi = (lat_q - lats[0]) / dlat
    fj = (lon_q - lons[0]) / dlon
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, nlat - 1)
    i1 = np.clip(i0 + 1, 0, nlat - 1)
    ti = np.clip(fi - i0, 0.0, 1.0)
    j0f = np.floor(fj).astype(np.int64)
    tj = fj - j0f
    j0 = j0f % nlon
    j1 = (j0f + 1) % nlon
    f00 = field[i0, j0]
    f01 = field[i0, j1]
    f10 = field[i1, j0]
    f11 = field[i1, j1]
    return (f00 * (1 - ti) * (1 - tj) + f01 * (1 - ti) * tj
            + f10 * ti * (1 - tj) + f11 * ti * tj)
