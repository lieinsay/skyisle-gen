"""骨架几何的共享定义（骨架第二版，docs/PLAN-SKELETON2.md）。

以前 D 的纬度域、G 的锚定带界、文明中心窗、密度基线分别写死在 s03 / s05 / s07 / check / viz / 操作台里；
这里集中成一份，所有纬度都以 |纬度| 给出、南北对称，并随 ① 的 band_scale（Held–Hou 开关）等比缩放。
G 的锚定、密度剖面、季节强度的公式在 C++ 核心里（core/src/planet/，Python 参考版删于 2026-09-30，tag python-reference-final）；
这里只留 check、viz、操作台与各步 _write 要的几样。
"""
from __future__ import annotations

import numpy as np


def band_scale(planet: dict) -> float:
    return float(planet.get("band_scale", 1.0))


def core_lat_range(cfg: dict, planet: dict) -> tuple[float, float]:
    lo, hi = (float(x) for x in cfg["skeleton"]["core_lat_range"])
    s = band_scale(planet)
    return lo * s, hi * s


def d_lat_range(cfg: dict, planet: dict) -> tuple[float, float]:
    lo, hi = (float(x) for x in cfg["skeleton"]["d_lat_range"])
    s = band_scale(planet)
    return lo * s, hi * s


def in_core(cfg: dict, planet: dict, lat) -> np.ndarray:
    lo, hi = core_lat_range(cfg, planet)
    a = np.abs(np.asarray(lat, dtype=np.float64))
    return (a >= lo) & (a <= hi)


def year_days(planet: dict, default: float = 336.0) -> float:
    cal = planet.get("calendar") or {}
    return float(cal.get("year_days_solar", default))

