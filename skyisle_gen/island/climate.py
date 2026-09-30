"""5.4 季节气候的前端：四季与逐日天气在 C++ 核心里算（core/src/island/climate.cpp，Python 参考版删于 2026-09-30，
tag python-reference-final）。这里只剩历法换算、island.json 的气候摘要、写 climate.json / weather_y<年>.csv、总览图的四季面板，
与全量季型统计（`skyisle island stats`）。

季节的算法（C++ 同式）：行星层只有年均值，季节由「风带随太阳南北摆动」+ 热惯性得到；季节强度读 ④ 的 season_range / season_range_sea，
相位滞后与带界摆动的振幅按 skeleton 的 τ / A 公式。历法：一年 = seasons × months_per_season × 28 日（默认 4 × 3 × 28 = 336）；
季中 = 二分二至（季 1 的季中 = 北半球夏至）。每季取季中那一天：带界向夏半球偏移 Δφ = k_shift × 倾角 × A_sea × cos(相位 − 滞后)，
在 1° 网格上取「本岛纬度 − Δφ」处的降水 / 风暴 / 航行窗口 / 风，再整体缩放使四季平均 = 年均值。
季型（决定 5）：四季分明 / 冷暖两季 / 雨旱季 / 风暴季 / 常夏（常寒），按四季各量的年内差异比较判定，名字是软的。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..skeleton import year_days

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


def set_climate(g: dict, clim: dict, r_t: float) -> None:
    """g["climate"] 与 island.json 的 climate 摘要。"""
    g["climate"] = clim
    g["json"]["climate"] = {"season_type": clim["season_type"], "season_type_zh": clim["season_type_zh"], "season_names": clim["season_names"],
                            "season_range_c": round(r_t, 1), "temps_c": [s["temp_c"] for s in clim["seasons"]],
                            "precip_mm": [s["precip_mm"] for s in clim["seasons"]]}


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


# ---------------------------------------------------------------- 全量季型统计（`skyisle island stats`）
TYPE_CODES = ["four", "two", "rain", "storm", "none_warm", "none_cold"]
TYPE_CODE_ZH = [TYPE_ZH[k] for k in TYPE_CODES]


def classify_all(ctx, c: dict, log=print) -> dict:
    """全世界每个岛群的季型（只算气候，不做地形；8000 群 4 s）。写 islands/season_stats.json，供操作台着色与文档统计。"""
    import time
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    planet = ctx.load_json(1, "planet")
    n = int(isl["lat"].size)
    keel = float(ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    codes = np.zeros(n, dtype=np.int8)
    ratio = np.zeros(n, dtype=np.float32)
    names_all = []
    from .engine import climate_only_cpp, core, flat_config
    cfg_obj = core().make_config(flat_config(c))
    t0 = time.perf_counter()
    for j in range(n):
        inp = {k: float(isl[k][j]) for k in ("lat", "lon", "height_m")}
        for k in ("precip", "temp", "storm", "window", "temp_sea", "season_range", "season_range_sea", "temp_winter", "temp_summer"):
            inp[k] = float(cli[k][j])
        inp["planet"] = planet
        inp["keel_clearance_m"] = keel
        C = climate_only_cpp(ctx, inp, c, cfg_obj)   # 四季在 C++ 里算（行星计划 P6b）
        st = C["season_type"]
        code = TYPE_CODES.index(st) if st != "none" else (5 if C["season_type_zh"] == TYPE_ZH["none_cold"] else 4)
        codes[j] = code
        ratio[j] = C["scores"]["rain"] * float(c["climate"]["wet_dry_ratio"])
        names_all.append("/".join(C["season_names"]))
        if j % 1000 == 999:
            log(f"  {j + 1}/{n} … {time.perf_counter() - t0:.0f} s")
    lat = np.abs(isl["lat"].astype(np.float64))
    bands = [(0, 8, "赤道永暴带"), (8, 28, "信风带"), (28, 36, "无风带"), (36, 62, "西风带"), (62, 90, "极地")]
    by_band = {}
    for lo, hi, name in bands:
        m = (lat >= lo) & (lat < hi)
        if m.any():
            cnt = np.bincount(codes[m], minlength=6)
            by_band[f"{name} {lo}–{hi}°"] = {"n": int(m.sum()), **{TYPE_CODE_ZH[k]: round(float(cnt[k] / m.sum()), 3) for k in range(6)}}
    cnt = np.bincount(codes, minlength=6)
    summary = {"run": ctx.out_dir.name, "seed": ctx.seed, "n": n, "seconds": round(time.perf_counter() - t0, 1),
               "share": {TYPE_CODE_ZH[k]: round(float(cnt[k] / n), 4) for k in range(6)},
               "count": {TYPE_CODE_ZH[k]: int(cnt[k]) for k in range(6)},
               "by_band": by_band,
               "season_range_c": {"median": round(float(np.median(cli["season_range"])), 1),
                                  "share_ge_20": round(float((cli["season_range"] >= 20).mean()), 4),
                                  "share_ge_16": round(float((cli["season_range"] >= 16).mean()), 4)},
               "wet_dry_ratio_median": round(float(np.median(ratio)), 2),
               "type_codes": TYPE_CODES, "type_zh": TYPE_CODE_ZH,
               "codes": codes.tolist(), "names": names_all}
    out = ctx.out_dir / "islands"
    out.mkdir(parents=True, exist_ok=True)
    (out / "season_stats.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return summary


def print_stats(st: dict) -> None:
    print(f"== 季型全量统计 {st['run']}（{st['n']} 群，{st['seconds']} s）==")
    print("  全球：" + "  ".join(f"{k} {v * 100:.1f}%" for k, v in st["share"].items() if v > 0))
    for band, row in st["by_band"].items():
        print(f"  {band:<16} n={row['n']:<5} " + "  ".join(f"{k} {v * 100:.0f}%" for k, v in row.items() if k != "n" and v > 0))
    sr = st["season_range_c"]
    print(f"  岛上全年温差中位 {sr['median']} °C，≥ 20 °C 占 {sr['share_ge_20'] * 100:.1f}%，≥ 16 °C 占 {sr['share_ge_16'] * 100:.1f}%；最湿/最干季比中位 {st['wet_dry_ratio_median']}")
