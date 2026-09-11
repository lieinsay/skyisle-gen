"""历法 ↔ 行星尺度的双向自洽（BACKLOG R1）。纯物理换算，不含随机数；① 调用，产物写进 planet.json 的 calendar 块。

约定（R1 未决问题的裁决，见 DESIGN-NOTES 四点十一）：
  a. 「一季 28 天」的天 = 太阳日；1 航行日 = 1 太阳日（day_range_km 是一个太阳日的航程）。
  b. 自由变量：历法（季数 × 季长）+ 自转周期 + 日照相对值 → 推出公转周期、恒星质量/光度/温度、轨道半径；
     行星半径独立（只经 days_per_rad 进图），密度决定重力（只报告）。
  c. 历法不反推带界：自转周期是输入不是输出，骨架带界（docs/11 定稿）不动；Held–Hou 仍是可选开关。
  d. 引入卫星：朔望月 = 一季（默认 28 日），一年 = 4 朔望月；由此推卫星轨道半径（行星质量由半径 × 密度给出）。

两个方向：
  calendar_to_orbit：给历法 → 公转周期 P = (季数 × 季长 + 1) 个自转周期（顺行：太阳日比恒星日多一天/年）
                     → 主序星标度 L = M^α（α = 3.5）+ 开普勒 a³ = P²M + 日照 I = L/a²  ⇒  M = (I · P^(4/3))^(1/(α − 2/3))
  orbit_to_calendar：给恒星质量 + 轨道半径 → 公转周期 → 一年的太阳日数 → 每季天数（与配置的季长比较，报残差）；
                     同时给出该轨道的日照，与 [s01.planet].insolation_rel 比较（只报告，④ 仍用配置值）。
潮汐锁定时标（Gladman 1996）：t = ω a⁶ I Q / (3 G M★² k₂ R⁵)，I = 0.4 m R²，Q、k₂ 可配。比太阳系年龄短就是历法留下的物理张力，
由 check 的 SK-cal 报警（推不出来的设定说明设定错了，不是推导错了——原则庚）；2026-09-11 用户拍板保留 4 × 28 并接受该张力
（`check.cal_accept_tidal_tension = true`）。
"""
from __future__ import annotations

import math

G_SI = 6.674e-11
M_SUN = 1.989e30
L_SUN = 3.828e26
R_SUN = 6.957e8
T_SUN = 5772.0
AU = 1.496e11
YEAR_S = 365.25 * 86400.0
M_EARTH = 5.972e24
R_EARTH = 6.371e6
G_EARTH = 9.81
MOON_A = 384400.0          # km
MOON_SIDEREAL = 27.3217    # d
ALPHA_ML = 3.5             # 主序星 L ∝ M^α（0.43 < M < 2 M☉）


def spectral_class(teff: float) -> str:
    for cls, lo in (("O", 30000), ("B", 10000), ("A", 7500), ("F", 6000), ("G", 5200), ("K", 3700)):
        if teff >= lo:
            return cls
    return "M"


def star_from_mass(m: float) -> dict:
    lum = m ** ALPHA_ML
    rad = m ** 0.8
    teff = T_SUN * (lum / rad ** 2) ** 0.25
    return {"mass_msun": m, "luminosity_lsun": lum, "radius_rsun": rad, "teff_k": teff,
            "spectral_class": spectral_class(teff)}


def tidal_lock_gyr(rot_hr: float, a_au: float, m_star: float, r_km: float, density_rel: float,
                   q: float, k2: float) -> float:
    r = r_km * 1e3
    m_p = M_EARTH * (r / R_EARTH) ** 3 * density_rel
    omega = 2.0 * math.pi / (rot_hr * 3600.0)
    inertia = 0.4 * m_p * r * r
    a = a_au * AU
    t = omega * a ** 6 * inertia * q / (3.0 * G_SI * (m_star * M_SUN) ** 2 * k2 * r ** 5)
    return t / (YEAR_S * 1e9)


def derive(cfg: dict) -> dict:
    p = cfg["s01"]["planet"]
    c = cfg["s01"].get("calendar", {})
    if not c:
        return {}
    mode = str(c.get("mode", "calendar_to_orbit"))
    seasons = int(c["seasons"])
    dps = float(c["days_per_season"])
    rot_hr = float(p["rotation_period_hr"])
    ins_cfg = float(p["insolation_rel"])
    r_km = float(p["radius_km"])
    dens = float(c.get("planet_density_rel", 1.0))
    q, k2 = float(c.get("tidal_q", 100.0)), float(c.get("tidal_k2", 0.3))
    prograde_extra = 1.0          # 顺行自转：一年的太阳日数 = 恒星日数 − 1

    out = {"mode": mode, "seasons": seasons, "days_per_season_config": dps,
           "solar_day_hr": rot_hr, "sailing_day_is_solar_day": True,
           "gravity_rel": (r_km / 6371.0) * dens, "planet_mass_rel_earth": (r_km / 6371.0) ** 3 * dens}

    if mode == "calendar_to_orbit":
        year_days = seasons * dps                                   # 太阳日
        p_orb_days = (year_days + prograde_extra) * rot_hr / 24.0   # 地球日
        p_yr = p_orb_days / 365.25
        m_star = (ins_cfg * p_yr ** (4.0 / 3.0)) ** (1.0 / (ALPHA_ML - 2.0 / 3.0))
        a_au = (p_yr ** 2 * m_star) ** (1.0 / 3.0)
        ins = m_star ** ALPHA_ML / a_au ** 2
        out.update({"year_days_solar": year_days, "days_per_season": dps,
                    "orbital_period_earth_days": p_orb_days, "orbital_period_years": p_yr,
                    "semi_major_axis_au": a_au, "insolation_derived": ins, "insolation_config": ins_cfg})
    elif mode == "orbit_to_calendar":
        m_star = float(c["stellar_mass_msun"])
        a_au = float(c["semi_major_axis_au"])
        p_yr = math.sqrt(a_au ** 3 / m_star)
        p_orb_days = p_yr * 365.25
        year_days = p_orb_days * 24.0 / rot_hr - prograde_extra
        ins = m_star ** ALPHA_ML / a_au ** 2
        out.update({"year_days_solar": year_days, "days_per_season": year_days / seasons,
                    "orbital_period_earth_days": p_orb_days, "orbital_period_years": p_yr,
                    "semi_major_axis_au": a_au, "insolation_derived": ins, "insolation_config": ins_cfg,
                    "days_per_season_residual": year_days / seasons - dps})
    else:
        raise ValueError(f"s01.calendar.mode 未知：{mode}")

    star = star_from_mass(m_star)
    star["apparent_size_rel_sun"] = star["radius_rsun"] / a_au      # 日轮视直径相对地球所见
    out["star"] = star
    out["tidal_lock_gyr"] = tidal_lock_gyr(rot_hr, a_au, m_star, r_km, dens, q, k2)
    # 若锁定时标短于太阳系年龄：报告至少要几季（同季长）才能把行星推远到安全距离（t ∝ a⁶/M★² ∝ P^4 M★^(-… )，数值扫描）
    age_gyr = float(c.get("system_age_gyr", 4.6))
    if out["tidal_lock_gyr"] < age_gyr and mode == "calendar_to_orbit":
        for n in range(seasons, 60):
            p_yr2 = ((n * dps + prograde_extra) * rot_hr / 24.0) / 365.25
            m2 = (ins_cfg * p_yr2 ** (4.0 / 3.0)) ** (1.0 / (ALPHA_ML - 2.0 / 3.0))
            a2 = (p_yr2 ** 2 * m2) ** (1.0 / 3.0)
            if tidal_lock_gyr(rot_hr, a2, m2, r_km, dens, q, k2) >= age_gyr:
                out["seasons_needed_for_no_lock"] = n
                break
    # 卫星：朔望月 = 一季 → 恒星月 1/(1/S + 1/Y) → 开普勒（行星质量 = 地球 × (R/R⊕)³ × 密度）
    if bool(c.get("moon", True)):
        syn = float(c.get("synodic_month_days", dps))
        year_days = out["year_days_solar"]
        sid = 1.0 / (1.0 / syn + 1.0 / year_days)
        m_p = M_EARTH * out["planet_mass_rel_earth"]
        t_s = sid * rot_hr * 3600.0
        a_m = (G_SI * m_p * t_s ** 2 / (4.0 * math.pi ** 2)) ** (1.0 / 3.0) / 1e3
        out["moon"] = {"synodic_month_days": syn, "sidereal_month_days": sid, "months_per_year": year_days / syn,
                       "distance_km": a_m, "distance_planet_radii": a_m / r_km,
                       "tide_rel_earth_same_mass": (MOON_A / a_m) ** 3}
    return out
