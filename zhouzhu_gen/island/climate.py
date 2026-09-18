"""5.4 季节气候：行星层只有年均值，季节由「风带随太阳南北摆动」+ 热惯性得到。

季节强度不在这里重算：直接读 ④ 的 season_range（岛上口径，含高度陆地性）/ season_range_sea（海面口径），
相位滞后与带界摆动的振幅按 skeleton 同一套 τ / A 公式（amplitude_retained）算出。
历法：一年 = seasons × months_per_season × 28 日（默认 4 × 3 × 28 = 336）；季中 = 二分二至（季 1 的季中 = 北半球夏至）。
每季取季中那一天：带界向夏半球偏移 Δφ = k_shift × 倾角 × A_sea × cos(相位 − 滞后)，在 1° 网格上取「本岛纬度 − Δφ」处的
降水 / 风暴 / 航行窗口 / 风（沿本经度、按局部带界，赤道永暴带边界夹断），再整体缩放使四季平均 = 年均值。
季型（决定 5）：四季分明 / 冷暖两季 / 雨旱季 / 风暴季 / 常夏（常寒），按四季各量的年内差异比较判定，名字是软的。
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..skeleton import amplitude_retained, year_days
from ..sphere import grid_interp
from ..stages.s02_wind import local_edges

SEASON_NAMES = {
    "four": ["暖季", "热季", "凉季", "冷季"],
    "two": ["暖季", "冷季"],
    "rain": ["雨季", "旱季", "转季"],
    "storm": ["风暴季", "平静季", "过渡季"],
    "none": ["常季"],
}
TYPE_ZH = {"four": "四季分明", "two": "冷暖两季", "rain": "雨旱季", "storm": "风暴季", "none_warm": "常夏", "none_cold": "常寒"}


def calendar(planet: dict) -> dict:
    cal = planet.get("calendar") or {}
    n_seasons = int(cal.get("seasons", 4))
    mps = int(cal.get("months_per_season", 3))
    dpm = float(cal.get("moon", {}).get("synodic_month_days", 28.0)) if cal.get("moon") else 28.0
    dps = float(cal.get("days_per_season", mps * dpm))
    ydays = year_days(planet, n_seasons * dps)
    return {"seasons": n_seasons, "months_per_season": mps, "days_per_month": dpm, "days_per_season": dps,
            "year_days": ydays, "day_offset_solstice_n": dps * 1.5,
            "note": "日序 d ∈ [0, year_days)；季 s = d // days_per_season；季中 = 二分二至：季 0 春分、季 1 夏至（北）、季 2 秋分、季 3 冬至（北）"}


def phase_of_day(d, cal: dict) -> np.ndarray:
    """日照相位：0 = 北半球夏至。"""
    return 2.0 * math.pi * (np.asarray(d, dtype=np.float64) - cal["day_offset_solstice_n"]) / cal["year_days"]


def thermal(cont: float, c4: dict, ydays: float) -> tuple[float, float, float]:
    """(τ 日, 振幅保留 A, 相位滞后 rad)。"""
    tau = cont * float(c4["season_tau_land_days"]) + (1.0 - cont) * float(c4["season_tau_ocean_days"])
    w = 2.0 * math.pi / ydays
    return tau, float(amplitude_retained(ydays, tau)), math.atan(w * tau)


def continentality(ctx, inp: dict) -> tuple[float, float]:
    """(海面口径, 岛上口径) 的陆地性，与 ④ 同一公式。"""
    cg = ctx.load_npz(4, "climate_grid")
    c4 = ctx.cfg["s04"]["climate"]
    cont = float(grid_interp(cg["continentality"], cg["lats"], cg["lons"], inp["lat"], inp["lon"])) if "continentality" in cg else 0.1
    keel = inp["keel_clearance_m"]
    alt = float(np.clip(cont + float(c4.get("season_alt_continentality", 0.0)) * np.clip((inp["height_m"] - keel) / 2000.0, 0.0, 1.0), 0.0, 1.0))
    return float(np.clip(cont, 0.0, 1.0)), alt


def match_mean(raw, target: float, lo: float, hi: float, iters: int = 12) -> np.ndarray:
    """把四季原始值缩放到均值 = target 并夹在 [lo, hi]：先乘性，夹断后把剩余亏欠加性摊到未顶格的季。"""
    x = np.clip(np.asarray(raw, dtype=np.float64), lo, hi)
    if x.mean() > 1e-9:
        x = np.clip(x * (target / x.mean()), lo, hi)
    for _ in range(iters):
        d = target - float(x.mean())
        if abs(d) < 1e-7:
            break
        free = (x < hi - 1e-12) if d > 0 else (x > lo + 1e-12)
        if not free.any():
            break
        x = np.clip(x + np.where(free, d * x.size / free.sum(), 0.0), lo, hi)
    return x


def _sample_lat(lat: float, lon: float, dphi: float, band_local: dict) -> float:
    """摆动后的取样纬度：lat − Δφ（Δφ 为带系整体向北的偏移），不跨越赤道永暴带的局部边界。"""
    e = local_edges(band_local, np.array([lon]))
    eq = float(e["eq_n"][0]) if lat >= 0 else float(-e["eq_s"][0])
    s = lat - dphi
    if lat >= 0:
        return max(s, eq + 0.25)
    return min(s, -eq - 0.25)


def build_climate(ctx, node: int, c: dict, g: dict, log=print) -> None:
    cc = c["climate"]
    inp = g["inp"]
    planet = inp["planet"]
    c4 = ctx.cfg["s04"]["climate"]
    cal = calendar(planet)
    ydays = cal["year_days"]
    n_s = cal["seasons"]
    dps = cal["days_per_season"]
    tilt = float(planet["axial_tilt_deg"])
    lat, lon = inp["lat"], inp["lon"]
    south = lat < 0
    cont_sea, cont_isl = continentality(ctx, inp)
    tau_sea, A_sea, lag_sea = thermal(cont_sea, c4, ydays)
    tau_isl, A_isl, lag_isl = thermal(cont_isl, c4, ydays)

    cg = ctx.load_npz(4, "climate_grid")
    wl = ctx.load_npz(4, "wind_local")
    bl = ctx.load_npz(4, "band_local")
    band_local = {"lons": bl["lons"], "edges": bl["edges"], "keys": bl["keys"]}
    lats, lons = cg["lats"], cg["lons"]

    mids = np.array([(s + 0.5) * dps for s in range(n_s)])
    ph = phase_of_day(mids, cal)
    sgn = -1.0 if south else 1.0
    # 带界摆动（向北为正，夏半球向极）：全球一致，用海面口径的 A 与滞后
    dphi = float(cc["k_shift"]) * tilt * A_sea * np.cos(ph - lag_sea)
    raw = {"precip": [], "storm": [], "window": [], "u": [], "v": [], "lat_sample": []}
    for s in range(n_s):
        ls = _sample_lat(lat, lon, float(dphi[s]), band_local)
        raw["lat_sample"].append(ls)
        raw["precip"].append(float(grid_interp(cg["precip"], lats, lons, ls, lon)))
        raw["storm"].append(float(grid_interp(cg["storm"], lats, lons, ls, lon)))
        raw["window"].append(float(grid_interp(cg["window"], lats, lons, ls, lon)))
        raw["u"].append(float(grid_interp(wl["u"], wl["lats"], wl["lons"], ls, lon)))
        raw["v"].append(float(grid_interp(wl["v"], wl["lats"], wl["lons"], ls, lon)))
    raw = {k: np.array(v) for k, v in raw.items()}
    # 缩放到年均：降水 / 风暴乘性（风暴夹 [0,1] 后再校两轮），窗口加性
    precip = match_mean(raw["precip"], inp["precip"], 0.0, 1.0)
    storm = match_mean(raw["storm"], inp["storm"], 0.0, 1.0) if raw["storm"].mean() > 1e-9 else np.full(n_s, inp["storm"])
    window = match_mean(raw["window"], inp["window"], 0.03, 1.0)
    # 温度：年均 + 半振幅 × cos(相位 − 滞后)，南半球反相
    amp_isl = 0.5 * inp["season_range"]
    amp_sea = 0.5 * inp["season_range_sea"]
    t_isl = inp["temp"] + sgn * amp_isl * np.cos(ph - lag_isl)
    t_sea = inp["temp_sea"] + sgn * amp_sea * np.cos(ph - lag_sea)
    from .hydro import precip_mm
    p_rate = np.array([precip_mm(p, cc) for p in precip])     # 该季强度折成的年当量 mm/年
    p_mm = p_rate * dps / ydays                                # 该季实际降水 mm/季（四季之和 = 年降水）
    # 季型判定
    r_t = float(inp["season_range"])
    pr_ratio = float(precip.max() / max(1e-9, precip.min()))
    st_diff = float(storm.max() - storm.min())
    wn_diff = float(window.max() - window.min())
    scores = {"temp": r_t / float(cc["four_season_range_c"]),
              "rain": pr_ratio / float(cc["wet_dry_ratio"]),
              "storm": max(st_diff / float(cc["storm_season_diff"]), wn_diff / float(cc["window_season_diff"]))}
    if r_t >= float(cc["four_season_range_c"]):
        stype = "four"
    else:
        cand = {k: v for k, v in scores.items() if v >= 1.0 and not (k == "temp" and r_t < float(cc["two_season_range_c"]))}
        if cand:
            best = max(sorted(cand), key=lambda k: cand[k])
            stype = {"temp": "two", "rain": "rain", "storm": "storm"}[best]
        else:
            stype = "none"
    names = _season_names(stype, t_isl, precip, storm, window, n_s)
    type_zh = TYPE_ZH[stype] if stype != "none" else (TYPE_ZH["none_cold"] if inp["temp"] < float(cc["cold_mean_temp_c"]) else TYPE_ZH["none_warm"])

    seasons = []
    for s in range(n_s):
        sp = math.hypot(raw["u"][s], raw["v"][s])
        seasons.append({
            "index": s, "name": names[s], "days": [int(round(s * dps)), int(round((s + 1) * dps)) - 1], "mid_day": float(mids[s]),
            "months": [int(s * cal["months_per_season"]) + m for m in range(cal["months_per_season"])],
            "temp_c": round(float(t_isl[s]), 2), "temp_sea_c": round(float(t_sea[s]), 2),
            "precip_rel": round(float(precip[s]), 4), "precip_mm": round(float(p_mm[s]), 0), "precip_mm_annual_rate": round(float(p_rate[s]), 0),
            "storm": round(float(storm[s]), 4), "window": round(float(window[s]), 4),
            "wind": {"u": round(float(raw["u"][s]), 2), "v": round(float(raw["v"][s]), 2), "speed_ms": round(sp, 2),
                     "from_deg": round((math.degrees(math.atan2(-raw["u"][s], -raw["v"][s])) + 360.0) % 360.0, 0)},
            "band_shift_deg": round(float(dphi[s]), 2), "lat_sampled": round(float(raw["lat_sample"][s]), 2),
        })
    clim = {
        "calendar": cal,
        "season_type": stype, "season_type_zh": type_zh, "season_names": names,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "annual": {"temp_c": round(inp["temp"], 2), "temp_sea_c": round(inp["temp_sea"], 2), "precip_rel": round(inp["precip"], 4),
                   "precip_mm": round(precip_mm(inp["precip"], cc), 0), "storm": round(inp["storm"], 4), "window": round(inp["window"], 4),
                   "season_range_c": round(r_t, 2), "season_range_sea_c": round(float(inp["season_range_sea"]), 2),
                   "temp_winter_c": round(inp["temp_winter"], 2), "temp_summer_c": round(inp["temp_summer"], 2)},
        "thermal": {"continentality_sea": round(cont_sea, 4), "continentality_island": round(cont_isl, 4),
                    "tau_sea_days": round(tau_sea, 1), "tau_island_days": round(tau_isl, 1),
                    "amplitude_retained_sea": round(A_sea, 3), "amplitude_retained_island": round(A_isl, 3),
                    "lag_sea_days": round(lag_sea / (2 * math.pi) * ydays, 1), "lag_island_days": round(lag_isl / (2 * math.pi) * ydays, 1),
                    "k_shift": float(cc["k_shift"]), "band_shift_amp_deg": round(float(cc["k_shift"]) * tilt * A_sea, 2)},
        "seasons": seasons,
        "means_check": {"precip_rel": round(float(precip.mean()), 5), "storm": round(float(storm.mean()), 5),
                        "window": round(float(window.mean()), 5), "temp_c": round(float(t_isl.mean()), 4)},
        "note": "季名是软的（决定 5）；南北半球反相；每季数值为季中那一天的摆动取样再缩放到年均；precip_mm 是该季总量（四季之和 = 年降水），precip_mm_annual_rate 是折成年当量的强度",
    }
    g["climate"] = clim
    g["json"]["climate"] = {"season_type": stype, "season_type_zh": type_zh, "season_names": names,
                            "season_range_c": round(r_t, 1), "temps_c": [s["temp_c"] for s in seasons],
                            "precip_mm": [s["precip_mm"] for s in seasons]}
    log(f"  气候：{type_zh}（{'/'.join(names)}）温 {[round(x, 1) for x in t_isl.tolist()]} 雨 {[int(x) for x in p_mm.tolist()]} mm "
        f"风暴 {[round(x, 2) for x in storm.tolist()]} 窗 {[round(x, 2) for x in window.tolist()]} Δφ {[round(x, 1) for x in dphi.tolist()]}")


def _season_names(stype: str, t, precip, storm, window, n_s: int) -> list[str]:
    t = np.asarray(t)
    names = ["季"] * n_s
    if stype == "four":
        hot, cold = int(np.argmax(t)), int(np.argmin(t))
        for s in range(n_s):
            if s == hot:
                names[s] = "热季"
            elif s == cold:
                names[s] = "冷季"
            else:
                # 冷 → 热 之间是暖（回暖），热 → 冷 之间是凉
                d_from_cold = (s - cold) % n_s
                d_from_hot = (s - hot) % n_s
                names[s] = "暖季" if d_from_cold < d_from_hot else "凉季"
    elif stype == "two":
        order = np.argsort(-t, kind="stable")
        warm = set(order[: n_s // 2].tolist())
        wi = ci = 0
        for s in range(n_s):
            if s in warm:
                wi += 1
                names[s] = f"暖季{'一二三四'[wi - 1]}" if n_s > 2 else "暖季"
            else:
                ci += 1
                names[s] = f"冷季{'一二三四'[ci - 1]}" if n_s > 2 else "冷季"
    elif stype == "rain":
        wet, dry = int(np.argmax(precip)), int(np.argmin(precip))
        for s in range(n_s):
            names[s] = "雨季" if s == wet else ("旱季" if s == dry else "转季")
    elif stype == "storm":
        st = np.asarray(storm) - np.asarray(window)
        hi, lo = int(np.argmax(st)), int(np.argmin(st))
        for s in range(n_s):
            names[s] = "风暴季" if s == hi else ("平静季" if s == lo else "过渡季")
    else:
        names = [f"第{'一二三四五六'[s]}季" for s in range(n_s)]
    return names


# ---------------------------------------------------------------- 逐日曲线（供 5.5 与图）
def daily_curves(clim: dict, inp: dict, c4: dict) -> dict:
    """336 天的平滑季节曲线：温度按 cos(相位 − 滞后)；降水 / 风暴 / 窗口 / 风按季中值做周期余弦插值。"""
    cal = clim["calendar"]
    ydays = int(round(cal["year_days"]))
    d = np.arange(ydays)
    ph = phase_of_day(d, cal)
    sgn = -1.0 if inp["lat"] < 0 else 1.0
    lag = clim["thermal"]["lag_island_days"] / ydays * 2 * math.pi
    temp = inp["temp"] + sgn * 0.5 * inp["season_range"] * np.cos(ph - lag)
    n_s = cal["seasons"]
    mids = np.array([s["mid_day"] for s in clim["seasons"]])

    def interp_periodic(vals):
        vals = np.asarray(vals, dtype=np.float64)
        x = np.concatenate([mids - ydays, mids, mids + ydays])
        y = np.concatenate([vals, vals, vals])
        # 余弦平滑：先线性再做一次周期平滑（窗口 = 季长的 1/3）
        lin = np.interp(d, x, y)
        k = max(1, int(cal["days_per_season"] // 3))
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        ext = np.concatenate([lin[-k:], lin, lin[:k]])
        return np.convolve(ext, ker, mode="valid")

    out = {"day": d, "temp_c": temp, "season": (d // int(cal["days_per_season"])).astype(int)}
    for key in ("precip_rel", "precip_mm", "storm", "window"):
        out[key] = interp_periodic([s[key] for s in clim["seasons"]])
    out["wind_u"] = interp_periodic([s["wind"]["u"] for s in clim["seasons"]])
    out["wind_v"] = interp_periodic([s["wind"]["v"] for s in clim["seasons"]])
    return out


def write_climate(out: Path, g: dict, year: int) -> None:
    clim = dict(g["climate"])
    if "weather" in g:
        clim["weather"] = dict(g["weather"]["json"])
        days = g["weather"]["days"]
        clim["weather"]["days"] = days
        with open(out / f"weather_y{year}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["day", "season", "season_name", "month", "day_of_month", "type", "precip_mm", "temp_c", "wind_from_deg", "wind_ms", "sailable", "storm_event"])
            for row in days:
                w.writerow([row[k] for k in ("day", "season", "season_name", "month", "day_of_month", "type", "precip_mm", "temp_c", "wind_from_deg", "wind_ms", "sailable", "storm_event")])
    (out / "climate.json").write_text(json.dumps(clim, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def draw_climate_panels(fig, gs, g: dict) -> None:
    import matplotlib.pyplot as plt
    clim = g["climate"]
    S = clim["seasons"]
    names = [f"{s['name']}\n{s['days'][0]}–{s['days'][1]}日" for s in S]
    x = np.arange(len(S))
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, [s["precip_mm"] for s in S], color="#4c72b0", alpha=0.8, width=0.6)
    ax.set_ylabel("降水 mm/季")
    ax2 = ax.twinx()
    ax2.plot(x, [s["temp_c"] for s in S], color="#c44e52", marker="o")
    ax2.plot(x, [s["temp_sea_c"] for s in S], color="#c44e52", ls="--", alpha=0.6)
    ax2.set_ylabel("温度 °C（实线岛上，虚线海面）")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_title(f"四季：{clim['season_type_zh']}（全年温差 {clim['annual']['season_range_c']:.1f} °C）", fontsize=9)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.bar(x - 0.18, [s["storm"] for s in S], width=0.36, color="#8172b2", label="风暴强度")
    ax3.bar(x + 0.18, [s["window"] for s in S], width=0.36, color="#55a868", label="航行窗口")
    ax3.set_ylim(0, 1)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{s['name']}\n风 {s['wind']['speed_ms']:.0f} m/s 自 {s['wind']['from_deg']:.0f}°" for s in S], fontsize=7)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_title(f"风暴 / 窗口（带界摆幅 ±{clim['thermal']['band_shift_amp_deg']:.1f}°，滞后 {clim['thermal']['lag_island_days']:.0f} 日）", fontsize=9)
    ax4 = fig.add_subplot(gs[2, :])
    if "weather" in g:
        from .weather import draw_weather_strip
        draw_weather_strip(ax4, g)
    else:
        cur = g.get("daily")
        if cur is not None:
            ax4.plot(cur["day"], cur["temp_c"], color="#c44e52")
            ax4.set_ylabel("°C")
            ax4b = ax4.twinx()
            ax4b.fill_between(cur["day"], 0, cur["precip_mm"] / (cur["day"].size / 4), color="#4c72b0", alpha=0.3)
            ax4b.set_ylabel("mm/日")
            ax4.set_xlim(0, cur["day"].size)
            ax4.set_title("全年曲线（季节插值）", fontsize=9)
