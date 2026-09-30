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


def angdist(a_xyz: np.ndarray, b_xyz: np.ndarray) -> np.ndarray:
    """大圆角距（弧度）。数值稳定（用 arctan2 而非 arccos）。"""
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    cross = np.cross(a, b)
    sin_d = np.sqrt(np.sum(cross * cross, axis=-1))
    cos_d = np.sum(a * b, axis=-1)
    return np.arctan2(sin_d, cos_d)


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
