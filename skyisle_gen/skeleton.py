"""骨架几何与季节物理的共享定义（骨架第二版，docs/PLAN-SKELETON2.md）。

以前 D 的纬度域、G 的锚定带界、文明中心窗、密度基线分别写死在 s03 / s05 / s07 / check / viz / 操作台里；
这里集中成一份，所有纬度都以 |纬度| 给出、南北对称，并随 ① 的 band_scale（Held–Hou 开关）等比缩放。

季节强度（PLAN-SKELETON2 §4.2、PLAN-ISLAND 5.4）：
  日照的年变化一阶谐波 ΔQ(φ; 倾角) / λ × A，A = 1/√(1+(ωτ)²)，ω = 2π / 一年，
  τ = c·τ_land + (1 − c)·τ_ocean（c = 陆地性，热容按面积混合）。输出「全年温差」= 2 × 半振幅。
  按地球定标：郑州（34.7°，c≈0.6）算 24 °C / 实测 26，石家庄（38°，c≈0.7）28 / 29，香港（22°，c≈0.4）13 / 13。
"""
from __future__ import annotations

import math

import numpy as np

SOLAR_CONST = 1361.0


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


def g_latitude(cfg: dict, bands: dict) -> tuple[float, str]:
    sk = cfg["skeleton"]
    edge = str(sk.get("g_anchor_edge", "trades_top_deg"))
    return float(bands[edge]) - float(sk["g_delta_deg"]), edge


def lat_density(section: dict, planet: dict, lat) -> np.ndarray:
    """③ 的密度基线：|纬度| 分段线性（结点随 band_scale 缩放）。"""
    ld = section["lat_density"]
    knots = np.asarray(ld["lat"], dtype=np.float64) * band_scale(planet)
    vals = np.asarray(ld["density"], dtype=np.float64)
    if knots.size != vals.size or np.any(np.diff(knots) <= 0):
        raise ValueError("s03.islands.lat_density：lat 与 density 须等长，lat 严格递增")
    return np.interp(np.abs(np.asarray(lat, dtype=np.float64)), np.minimum(knots, 90.0), vals)


def insolation_first_harmonic(lat_deg, tilt_deg: float, n: int = 360) -> np.ndarray:
    """日均日照（W/m²）一年内的一阶谐波振幅（半振幅）。圆轨道，赤纬 δ = asin(sin ε · sin 2πt)。"""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    t = (np.arange(n) + 0.5) / n
    dec = np.arcsin(math.sin(math.radians(tilt_deg)) * np.sin(2 * np.pi * t))
    phi = lat.reshape(-1, 1)
    d = dec.reshape(1, -1)
    x = np.clip(-np.tan(phi) * np.tan(d), -1.0, 1.0)
    h0 = np.arccos(x)
    q = SOLAR_CONST / np.pi * (h0 * np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.sin(h0))
    c = np.exp(-2j * np.pi * t).reshape(1, -1)
    amp = 2.0 * np.abs((q * c).mean(axis=1))
    return amp.reshape(np.shape(lat_deg))


def amplitude_retained(year_days: float, tau_days) -> np.ndarray:
    w = 2.0 * math.pi / float(year_days)
    return 1.0 / np.sqrt(1.0 + (w * np.asarray(tau_days, dtype=np.float64)) ** 2)


def season_range(lat_deg, continentality, tilt_deg: float, year_days: float, c: dict,
                 insolation_rel: float = 1.0) -> np.ndarray:
    """全年温差（°C，= 2 × 半振幅）。continentality ∈ [0,1] 可为数组，与 lat_deg 可广播。"""
    cont = np.clip(np.asarray(continentality, dtype=np.float64), 0.0, 1.0)
    tau = cont * float(c["season_tau_land_days"]) + (1.0 - cont) * float(c["season_tau_ocean_days"])
    dq = insolation_first_harmonic(lat_deg, tilt_deg) * float(insolation_rel)
    return 2.0 * dq / float(c["season_lambda_w_m2_k"]) * amplitude_retained(year_days, tau)


def year_days(planet: dict, default: float = 336.0) -> float:
    cal = planet.get("calendar") or {}
    return float(cal.get("year_days_solar", default))


def smoothstep(x, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
