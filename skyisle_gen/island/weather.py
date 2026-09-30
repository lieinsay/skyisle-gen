"""5.5 逐日天气的前端：逐日模拟在 C++ 核心里（core/src/island/climate.cpp，Python 参考版删于 2026-09-30，tag python-reference-final）；
这里只剩天气类型表、每季的链参数（IS-daily 对照雨日比例用）、逐日表与季汇总的拼装（set_weather）、IS-daily 的多年统计与总览图的逐日条。

模拟的算法（C++ 同式）：一年（默认 336 天）按年份种子可复现。
晴雨：每季一个两状态马尔可夫链（干→湿、湿→湿），由该季雨量与雨日比例反解；雨量伽马分布，期望缩放到该季总量，
      单年围绕气候值波动、多年平均回到气候值（IS-daily：600 年样本 < 5%；A5 起由 60 年加长——干端的年雨偏态重，60 年的均值常低一成、样本标准误也偏小）。
风暴：按该季风暴强度生成持续 1–4 天的事件（泊松个数），风暴日强制大风、大雨、禁航。
风：围绕该季平均风向风速的 AR(1) 扰动；温度：季节曲线 + AR(1) 日际扰动，雨日 / 风暴日偏凉。
云海漫顶：低岛（台面 < fog_surface_max_m）在静风、潮湿的日子被云海漫上岸缘 —— 本世界独有的天气类型。
逐日气温是台面（③ 的 height_m = 主岛陆地高程中位数）处的气温，与 ④ 同口径；某格气温 = temp + 直减率 × (台面 − 格高)。
随机数：rng.entity_rng(seed, ISLAND_STREAM, f"island:{node}:weather:{year}")，改年份不动地形与气候。
"""
from __future__ import annotations

import math

import numpy as np

TYPES = ["晴", "多云", "小雨", "大雨", "云海漫顶", "风暴", "小雪", "大雪", "暴风雪"]
TYPE_COLORS = {"晴": "#f7d76b", "多云": "#c8ccd2", "小雨": "#8fb8de", "大雨": "#3b6fb6", "云海漫顶": "#e6e1f2", "风暴": "#7a2d8c",
               "小雪": "#dfe8f5", "大雪": "#b7c6dc", "暴风雪": "#5a4a8c"}


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


def set_weather(g: dict, y: dict, params: list[dict], year: int, log=print) -> None:
    """一年逐日数组 → 逐日表、季汇总、g["weather"] 与 island.json 的 weather 摘要（y 与 params 由 C++ 算）。"""
    clim = g["climate"]
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
    """IS-daily：多年样本的季降水均值与雨日比例，对照气候值（逐年模拟在 C++ 里，统计在这里）。"""
    from .engine import weather_years_cpp
    clim = g["climate"]
    params = season_params(clim, c["weather"])
    P, F = weather_years_cpp(ctx, node, c, g, years)
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
