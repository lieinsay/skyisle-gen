"""5.5 逐日天气：一年（默认 336 天）按年份种子可复现。

晴雨：每季一个两状态马尔可夫链（干→湿、湿→湿），由该季雨量与雨日比例反解；雨量伽马分布，期望缩放到该季总量，
      单年围绕气候值波动、多年平均回到气候值（IS-daily：30 年样本 < 5%）。
风暴：按该季风暴强度生成持续 1–4 天的事件（泊松个数），风暴日强制大风、大雨、禁航。
风：围绕该季平均风向风速的 AR(1) 扰动；温度：季节曲线 + AR(1) 日际扰动，雨日 / 风暴日偏凉。
云海漫顶：低岛（峰高 < fog_peak_max_m）在静风、潮湿的日子被云海漫上岸缘 —— 本世界独有的天气类型。
随机数：rng.entity_rng(seed, ISLAND_STREAM, f"island:{node}:weather:{year}")，改年份不动地形与气候。
"""
from __future__ import annotations

import math

import numpy as np

TYPES = ["晴", "多云", "小雨", "大雨", "云海漫顶", "风暴", "小雪", "大雪", "暴风雪"]
TYPE_COLORS = {"晴": "#f7d76b", "多云": "#c8ccd2", "小雨": "#8fb8de", "大雨": "#3b6fb6", "云海漫顶": "#e6e1f2", "风暴": "#7a2d8c",
               "小雪": "#dfe8f5", "大雪": "#b7c6dc", "暴风雪": "#5a4a8c"}
SNOW_TYPES = {"小雪", "大雪", "暴风雪"}
STORM_TYPES = {"风暴", "暴风雪"}


def season_params(clim: dict, wc: dict) -> list[dict]:
    """每季的链参数：雨日比例 f_wet（随季雨量），p_ww / p_dw 使平稳分布 = f_wet，湿日均雨量 = 季雨量 / 雨日数。"""
    out = []
    dps = float(clim["calendar"]["days_per_season"])
    persist = float(wc["wet_persistence"])
    for s in clim["seasons"]:
        P = float(s["precip_mm"])
        f_wet = float(np.clip(float(wc["wet_frac_base"]) + float(wc["wet_frac_k"]) * (P / 1000.0) ** 0.7,
                              float(wc["wet_frac_min"]), float(wc["wet_frac_max"])))
        p_ww = f_wet + persist * (1.0 - f_wet)
        p_dw = f_wet * (1.0 - p_ww) / max(1e-9, 1.0 - f_wet)
        storm_frac = float(wc["storm_day_k"]) * float(s["storm"]) ** float(wc["storm_day_exp"])
        storm_frac = min(storm_frac, 0.6)
        # 雨量预算含风暴日：E[总量] = dps × [(1 − sf) f_wet + sf × mult] × 湿日均值 = 该季气候值
        mult = float(wc["storm_rain_mult"])
        mean_wet = P / max(1.0, dps * ((1.0 - storm_frac) * f_wet + storm_frac * mult))
        out.append({"f_wet": f_wet, "f_rain_days": (1.0 - storm_frac) * f_wet + storm_frac, "p_ww": p_ww, "p_dw": min(p_dw, 0.95), "mean_wet_mm": mean_wet,
                    "storm_frac": storm_frac, "p_calm": float(np.clip(float(s["window"]) / max(1e-6, 1.0 - storm_frac), 0.0, 1.0)),
                    "wind_u": float(s["wind"]["u"]), "wind_v": float(s["wind"]["v"]), "speed": float(s["wind"]["speed_ms"])})
    return out


def simulate_year(rng, clim: dict, daily: dict, params: list[dict], peak_m: float, wc: dict, rim_m: float | None = None,
                  lapse_c_per_km: float = 6.0) -> dict:
    cal = clim["calendar"]
    ydays = int(round(cal["year_days"]))
    dps = int(round(cal["days_per_season"]))
    dpm = int(round(cal["days_per_month"]))
    season = daily["season"]
    n = ydays
    wet = np.zeros(n, dtype=bool)
    # 马尔可夫链（起始按平稳分布）
    state = rng.uniform() < params[int(season[0])]["f_wet"]
    for d in range(n):
        p = params[int(season[d])]
        pr = p["p_ww"] if state else p["p_dw"]
        state = rng.uniform() < pr
        wet[d] = state
    # 雨量：伽马，湿日均值按季
    shape = float(wc["rain_gamma_shape"])
    mean_wet = np.array([params[int(s)]["mean_wet_mm"] for s in season])
    precip = np.where(wet, rng.gamma(shape, 1.0, n) * mean_wet / shape, 0.0)
    # 风暴事件
    storm_id = np.zeros(n, dtype=np.int32)
    eid = 0
    dur_lo, dur_hi = int(wc["storm_days_min"]), int(wc["storm_days_max"])
    mean_dur = 0.5 * (dur_lo + dur_hi)
    for s in range(int(cal["seasons"])):
        p = params[s]
        # 事件会重叠：泊松布尔模型的覆盖率 = 1 − exp(−λ·均长/季长)，反解 λ 使风暴日占比 = storm_frac
        lam = -math.log(max(1e-9, 1.0 - p["storm_frac"])) * dps / mean_dur if p["storm_frac"] > 0 else 0.0
        k = int(rng.poisson(lam)) if lam > 0 else 0
        starts = sorted(rng.integers(s * dps, (s + 1) * dps, k).tolist()) if k else []
        for st in starts:
            eid += 1
            dur = int(rng.integers(dur_lo, dur_hi + 1))
            for d in range(st, st + dur):
                storm_id[d % n] = eid           # 跨年回绕，不截断
    storm = storm_id > 0
    precip = np.where(storm, rng.gamma(shape, 1.0, n) * float(wc["storm_rain_mult"]) * mean_wet / shape, precip)
    wet = wet | storm
    # 风：AR(1) 围绕季曲线；风暴日加强
    phi = float(wc["wind_ar1"])
    mu_u, mu_v = daily["wind_u"], daily["wind_v"]
    sp_mu = np.hypot(mu_u, mu_v)
    sig = float(wc["wind_sigma_rel"]) * sp_mu + float(wc["wind_sigma_min"])
    eu = np.zeros(n)
    ev = np.zeros(n)
    z = rng.normal(0.0, 1.0, (n, 2))
    for d in range(1, n):
        eu[d] = phi * eu[d - 1] + math.sqrt(1 - phi * phi) * sig[d] * z[d, 0]
        ev[d] = phi * ev[d - 1] + math.sqrt(1 - phi * phi) * sig[d] * z[d, 1]
    u = mu_u + eu
    v = mu_v + ev
    speed = np.hypot(u, v)
    if storm.any():
        gust = float(wc["storm_wind_mult"]) * np.maximum(speed, 1.0) + float(wc["storm_wind_add"])
        speed = np.where(storm, gust, speed)
    wind_from = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    # 温度：季曲线 + AR(1)
    phi_t = float(wc["temp_ar1"])
    sig_t = float(wc["temp_sigma_c"])
    et = np.zeros(n)
    zt = rng.normal(0.0, 1.0, n)
    for d in range(1, n):
        et[d] = phi_t * et[d - 1] + math.sqrt(1 - phi_t * phi_t) * sig_t * zt[d]
    temp = daily["temp_c"] + et - float(wc["rain_cool_c"]) * wet - float(wc["storm_cool_c"]) * storm
    # 云海漫顶：低岛、静风、潮湿（今日或昨日有雨，或本季雨日多）、非风暴
    fog = np.zeros(n, dtype=bool)
    if peak_m < float(wc["fog_peak_max_m"]):
        low = 1.0 - min(1.0, max(0.0, (peak_m - 300.0) / max(1.0, float(wc["fog_peak_max_m"]) - 300.0)))
        humid = np.maximum(wet.astype(float), np.roll(wet, 1).astype(float) * 0.7) * 0.6 + np.array([params[int(s)]["f_wet"] for s in season]) * 0.4
        calm = np.clip(1.0 - speed / float(wc["fog_calm_ms"]), 0.0, 1.0)
        p_fog = float(wc["fog_p0"]) * (0.3 + 0.7 * low) * calm * humid
        fog = (rng.uniform(0.0, 1.0, n) < p_fog) & ~storm & (precip < float(wc["heavy_rain_mm"]))
    # 天气类型。雨 / 雪按岸缘气温分：逐日 temp 是主岛峰高处的气温，岸缘 = temp + 直减率 × (峰 − 岸缘)
    cloudy = ~wet & (rng.uniform(0.0, 1.0, n) < np.array([params[int(s)]["f_wet"] for s in season]) * float(wc["cloudy_k"]))
    rim = peak_m if rim_m is None else float(rim_m)
    t_rim = temp + lapse_c_per_km * max(0.0, peak_m - rim) / 1000.0
    snowy = t_rim <= float(wc["snow_temp_c"])
    heavy = precip >= float(wc["heavy_rain_mm"])
    t = np.full(n, 0, dtype=np.int8)                      # 晴
    t[cloudy] = 1
    t[wet & ~heavy] = 2
    t[wet & heavy] = 3
    t[fog] = 4
    t[storm] = 5
    t[wet & ~heavy & snowy & ~storm] = 6                  # 小雪
    t[wet & heavy & snowy & ~storm] = 7                   # 大雪
    t[storm & snowy] = 8                                  # 暴风雪
    t[fog & ~storm] = 4
    # 出航：非风暴、风 < sail_wind_max、按季窗口的平静概率
    calm_ok = rng.uniform(0.0, 1.0, n) < np.array([params[int(s)]["p_calm"] for s in season])
    sailable = ~storm & (speed < float(wc["sail_wind_max_ms"])) & calm_ok
    day = np.arange(n)
    return {"day": day, "season": season, "month": day // dpm, "day_of_month": day % dpm + 1, "type": t,
            "precip_mm": precip, "temp_c": temp, "wind_from_deg": wind_from, "wind_ms": speed, "sailable": sailable,
            "storm_event": storm_id, "wet": wet, "temp_rim_c": t_rim, "snow": snowy & wet}


def build_weather(ctx, node: int, c: dict, g: dict, year: int = 0, log=print) -> None:
    from . import _rng
    wc = c["weather"]
    clim = g["climate"]
    daily = g["daily"]
    params = season_params(clim, wc)
    peak = float(g["json"]["islands"][0]["peak_m"])
    rim = float(g["json"]["islands"][0]["rim_m"])
    rng = _rng(ctx, node, f"weather:{year}")
    y = simulate_year(rng, clim, daily, params, peak, wc, rim_m=rim, lapse_c_per_km=float(g["inp"]["lapse_c_per_km"]))
    names = clim["season_names"]
    n = y["day"].size
    days = []
    for d in range(n):
        s = int(y["season"][d])
        days.append({"day": int(d), "season": s, "season_name": names[s], "month": int(y["month"][d]) + 1, "day_of_month": int(y["day_of_month"][d]),
                     "type": TYPES[int(y["type"][d])], "precip_mm": round(float(y["precip_mm"][d]), 1), "temp_c": round(float(y["temp_c"][d]), 1),
                     "temp_rim_c": round(float(y["temp_rim_c"][d]), 1),
                     "wind_from_deg": int(round(float(y["wind_from_deg"][d]))), "wind_ms": round(float(y["wind_ms"][d]), 1),
                     "sailable": bool(y["sailable"][d]), "storm_event": int(y["storm_event"][d])})
    per_season = []
    for s in range(int(clim["calendar"]["seasons"])):
        m = y["season"] == s
        per_season.append({"season": s, "name": names[s], "rain_days": int((y["wet"][m] & ~y["snow"][m]).sum()), "snow_days": int(y["snow"][m].sum()),
                           "storm_days": int((y["storm_event"][m] > 0).sum()),
                           "storm_events": int(np.unique(y["storm_event"][m][y["storm_event"][m] > 0]).size),
                           "fog_days": int((y["type"][m] == 4).sum()), "sailable_days": int(y["sailable"][m].sum()),
                           "precip_mm": round(float(y["precip_mm"][m].sum()), 0), "precip_climate_mm": clim["seasons"][s]["precip_mm"],
                           "temp_mean_c": round(float(y["temp_c"][m].mean()), 1), "wet_frac_setting": round(params[s]["f_rain_days"], 3)})
    summary = {"year": year, "n_days": n, "types": {t: int((y["type"] == i).sum()) for i, t in enumerate(TYPES)},
               "snow_days": int(y["snow"].sum()), "storm_days": int((y["storm_event"] > 0).sum()),
               "precip_mm": round(float(y["precip_mm"].sum()), 0), "precip_climate_mm": clim["annual"]["precip_mm"],
               "storm_events": int(y["storm_event"].max()), "sailable_days": int(y["sailable"].sum()),
               "temp_min_c": round(float(y["temp_c"].min()), 1), "temp_max_c": round(float(y["temp_c"].max()), 1),
               "seasons": per_season, "params": [{k: round(v, 4) for k, v in p.items()} for p in params]}
    g["weather"] = {"json": summary, "days": days, "arrays": y}
    g["json"]["weather"] = {"year": year, "precip_mm": summary["precip_mm"], "types": summary["types"], "storm_events": summary["storm_events"],
                            "storm_days": summary["storm_days"], "snow_days": summary["snow_days"], "sailable_days": summary["sailable_days"]}
    log(f"  天气 y{year}：{summary['types']} 雨雪 {summary['precip_mm']:.0f} / 气候 {clim['annual']['precip_mm']:.0f} mm，雪日 {summary['snow_days']}，风暴 {summary['storm_events']} 场，可出航 {summary['sailable_days']} 天")


def multi_year_stats(ctx, node: int, c: dict, g: dict, years: int = 30) -> dict:
    """IS-daily：多年样本的季降水均值与雨日比例，对照气候值。"""
    from . import _rng
    wc = c["weather"]
    clim = g["climate"]
    params = season_params(clim, wc)
    peak = float(g["json"]["islands"][0]["peak_m"])
    n_s = int(clim["calendar"]["seasons"])
    P = np.zeros((years, n_s))
    F = np.zeros((years, n_s))
    for yv in range(years):
        y = simulate_year(_rng(ctx, node, f"weather:{yv}"), clim, g["daily"], params, peak, wc, rim_m=float(g["json"]["islands"][0].get("rim_m", peak)))
        for s in range(n_s):
            m = y["season"] == s
            P[yv, s] = y["precip_mm"][m].sum()
            F[yv, s] = y["wet"][m].mean()
    clim_p = np.array([s["precip_mm"] for s in clim["seasons"]])
    annual = P.sum(axis=1)
    se = float(annual.std(ddof=1) / math.sqrt(years)) if years > 1 else 1.0
    return {"years": years, "precip_mean_mm": P.mean(axis=0).round(1).tolist(), "precip_climate_mm": clim_p.tolist(),
            "precip_rel_err": (np.abs(P.mean(axis=0) - clim_p) / np.maximum(clim_p, 1.0)).round(4).tolist(),
            "annual_rel_err": round(float(abs(annual.mean() - clim_p.sum()) / max(1.0, clim_p.sum())), 4),
            "annual_z": round(float(abs(annual.mean() - clim_p.sum()) / max(se, 1e-9)), 2),
            "annual_se_rel": round(se / max(1.0, clim_p.sum()), 4),
            "wet_frac_mean": F.mean(axis=0).round(3).tolist(), "wet_frac_setting": [round(p["f_rain_days"], 3) for p in params],
            "wet_frac_err": np.abs(F.mean(axis=0) - np.array([p["f_rain_days"] for p in params])).round(4).tolist()}


def draw_weather_strip(ax, g: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    y = g["weather"]["arrays"]
    n = y["day"].size
    clim = g["climate"]
    cmap = ListedColormap([TYPE_COLORS[t] for t in TYPES])
    ax.imshow(y["type"][None, :], aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(TYPES) - 0.5, extent=[0, n, -1, 0], interpolation="nearest")
    ax.set_yticks([])
    ax2 = ax.twinx()
    ax2.bar(y["day"], y["precip_mm"], width=1.0, color="#3b6fb6", alpha=0.5)
    ax2.set_ylabel("mm/日")
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.045))
    ax3.plot(y["day"], y["temp_c"], color="#c44e52", lw=0.9)
    ax3.set_ylabel("°C", color="#c44e52")
    ax.set_xlim(0, n)
    ax.set_ylim(-1, 0)
    dps = int(round(clim["calendar"]["days_per_season"]))
    for s in range(int(clim["calendar"]["seasons"])):
        ax.axvline(s * dps, color="k", lw=0.5, alpha=0.5)
        ax.text((s + 0.5) * dps, -0.5, clim["season_names"][s], ha="center", va="center", fontsize=9, alpha=0.8)
    J = g["weather"]["json"]
    types = "  ".join(f"{k} {v}" for k, v in J["types"].items() if v)
    ax.set_title(f"逐日天气 y{J['year']}：{types}；雨 {J['precip_mm']:.0f} mm（气候 {J['precip_climate_mm']:.0f}）；风暴 {J['storm_events']} 场；可出航 {J['sailable_days']} 天", fontsize=9)
    ax.set_xlabel("日序（一年 336 日 = 4 季 × 3 月 × 28 日）")
