"""多倍频值噪声（经度周期），numpy 自实现，确定性。"""
from __future__ import annotations

import numpy as np


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _value_noise_layer(rng: np.random.Generator, nlat: int, nlon: int,
                       cells_lat: int, cells_lon: int) -> np.ndarray:
    """一层值噪声，输出 [nlat, nlon]，经度周期，取值 [-1, 1]。"""
    lattice = rng.uniform(-1.0, 1.0, size=(cells_lat + 1, cells_lon))  # 经度周期：cells_lon 列
    fi = np.linspace(0.0, cells_lat, nlat, endpoint=False) + 0.5 * cells_lat / nlat
    fj = np.linspace(0.0, cells_lon, nlon, endpoint=False) + 0.5 * cells_lon / nlon
    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64)
    ti = _smoothstep(np.clip(fi - i0, 0.0, 1.0))[:, None]
    tj = _smoothstep(np.clip(fj - j0, 0.0, 1.0))[None, :]
    i0 = np.clip(i0, 0, cells_lat - 1)
    i1 = np.clip(i0 + 1, 0, cells_lat)
    j1 = (j0 + 1) % cells_lon
    v00 = lattice[np.ix_(i0, j0)]
    v01 = lattice[np.ix_(i0, j1)]
    v10 = lattice[np.ix_(i1, j0)]
    v11 = lattice[np.ix_(i1, j1)]
    return (v00 * (1 - ti) * (1 - tj) + v01 * (1 - ti) * tj
            + v10 * ti * (1 - tj) + v11 * ti * tj)


def fractal_noise(rng: np.random.Generator, nlat: int, nlon: int,
                  base_cells: int = 6, octaves: int = 5,
                  persistence: float = 0.55, lacunarity: float = 2.0) -> np.ndarray:
    """多倍频值噪声，[nlat, nlon]，归一化到 [-1, 1]。"""
    out = np.zeros((nlat, nlon), dtype=np.float64)
    amp, total = 1.0, 0.0
    c_lat, c_lon = base_cells, base_cells * 2
    for _ in range(octaves):
        out += amp * _value_noise_layer(rng, nlat, nlon, c_lat, c_lon)
        total += amp
        amp *= persistence
        c_lat = int(round(c_lat * lacunarity))
        c_lon = int(round(c_lon * lacunarity))
    return out / total
