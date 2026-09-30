"""全行星岛群的柯本分类（PLAN-NATURE 阶段 A 的参照报告，DISCUSS-LANDFORM 第十六节第 12 条）：不写产物、不当验收指标。

每个岛群用岛群层的四季（各季台面高度的气温、各季降水，与 `island stats` 同一条 C++ 路 climate_only_cpp），按柯本分主类 / 细类，
按陆地面积（area_km2）加权；另报最冷月的分布与纬度带里的前几类。四季近似：季均换月极值按离年均的偏差 × 1.1，
夏半年 / 冬半年 = 最热 / 最冷季加相邻两季各一半，数差几个百分点，大局不变。地球的数（Peel 等 2007）一起打出来对照。

用法（仓库根下）：
    PYTHONUTF8=1 python docs/probes/koppen.py [--run out/seed42] [--json 结果.json]
seed42 全部 8000 群约十来秒（cpp）。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from skyisle_gen import island as isl_mod
from skyisle_gen.cli import _ctx_from_run
from skyisle_gen.island.engine import climate_only_cpp, core, flat_config

EARTH = {"A": 19.0, "B": 30.2, "C": 13.4, "D": 24.6, "E": 12.8}   # Peel, Finlayson & McMahon 2007，陆地面积占比 %


def koppen(T: np.ndarray, P: np.ndarray) -> str:
    """四季的气温（°C）与降水（mm）→ 柯本代码（主类 + 季节型，B 分 BW / BS）。"""
    Tann, Pann = T.mean(), P.sum()
    Tcold = Tann + 1.1 * (T.min() - Tann)
    Thot = Tann + 1.1 * (T.max() - Tann)
    ih, ic = int(T.argmax()), int(T.argmin())
    n4 = len(T)
    summer = P[ih] + 0.5 * (P[(ih - 1) % n4] + P[(ih + 1) % n4])
    fs = summer / max(Pann, 1e-9)
    if Thot < 10:
        return "E"
    pth = 20 * Tann + (280 if fs >= 0.7 else (0 if fs <= 0.3 else 140))
    if Pann < pth:
        return "BW" if Pann < 0.5 * pth else "BS"
    if Tcold >= 18:
        pdry = P.min() / 3.0
        if pdry >= 60:
            return "Af"
        return "Am" if pdry >= 100 - Pann / 25 else "Aw"
    # 季节型：夏最干的月 < 40 且 < 冬最湿月 / 3 → s；冬最干月 < 夏最湿月 / 10 → w（季里的月按 ± 两成摊）
    ps_dry, pw_wet = P[ih] / 3.0 * 0.8, P[ic] / 3.0 * 1.2
    pw_dry, ps_wet = P[ic] / 3.0 * 0.8, P[ih] / 3.0 * 1.2
    sub = "s" if (ps_dry < 40 and ps_dry < pw_wet / 3) else ("w" if pw_dry < ps_wet / 10 else "f")
    return ("C" if Tcold > 0 else "D") + sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/seed42")
    ap.add_argument("--json", default=None, help="每群的代码与汇总写到这里")
    a = ap.parse_args()

    ctx = _ctx_from_run(a.run)
    c = isl_mod.island_config(ctx)
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    planet = ctx.load_json(1, "planet")
    keel = float(ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    cfg_obj = core().make_config(flat_config(c))
    n = int(isl["lat"].size)
    rows = []
    t0 = time.perf_counter()
    for j in range(n):
        inp = {k: float(isl[k][j]) for k in ("lat", "lon", "height_m")}
        for k in ("precip", "temp", "storm", "window", "temp_sea", "season_range", "season_range_sea", "temp_winter", "temp_summer"):
            inp[k] = float(cli[k][j])
        inp["planet"] = planet
        inp["keel_clearance_m"] = keel
        ss = climate_only_cpp(ctx, inp, c, cfg_obj)["seasons"]
        rows.append((np.array([s["temp_c"] for s in ss]), np.array([s["precip_mm"] for s in ss])))
    print(f"{n} 群的四季：{time.perf_counter() - t0:.1f} s")

    area = isl["area_km2"].astype(np.float64)
    lat = isl["lat"].astype(np.float64)
    tot = area.sum()
    codes = [koppen(T, P) for T, P in rows]
    fine, main_ = defaultdict(float), defaultdict(float)
    for j, k in enumerate(codes):
        fine[k] += area[j]
        main_[k[0]] += area[j]
    print("主类（按陆地面积 %，括号里是地球）：" + "  ".join(f"{k} {100 * main_.get(k, 0) / tot:.1f}（{EARTH[k]}）" for k in "ABCDE"))
    print("细类：" + "  ".join(f"{k} {100 * v / tot:.1f}" for k, v in sorted(fine.items(), key=lambda x: -x[1])))

    Tc = np.array([T.mean() + 1.1 * (T.min() - T.mean()) for T, _ in rows])
    w = area / tot
    print("最冷月（按陆地面积）：" + "  ".join(f"[{lo},{hi}) {100 * w[(Tc >= lo) & (Tc < hi)].sum():.1f}%"
                                         for lo, hi in ((-99, -20), (-20, -10), (-10, -3), (-3, 0), (0, 5), (5, 18), (18, 99)))
          + f"；最低 {Tc.min():.1f} °C")
    alat = np.abs(lat)
    bands = {}
    for lo, hi in ((0, 8), (8, 28), (28, 36), (36, 48), (48, 62), (62, 90)):
        m = (alat >= lo) & (alat < hi)
        if not m.any():
            continue
        d = defaultdict(float)
        for j in np.where(m)[0]:
            d[codes[j]] += area[j]
        s = sum(d.values())
        top = sorted(d.items(), key=lambda x: -x[1])[:5]
        bands[f"{lo}-{hi}"] = {"land_pct": round(100 * s / tot, 1), "top": {k: round(100 * v / s, 1) for k, v in top}}
        print(f"  |纬度| {lo}–{hi}°：陆地 {100 * s / tot:.1f}%  " + "  ".join(f"{k} {100 * v / s:.0f}%" for k, v in top))

    if a.json:
        out = {"run": a.run, "main_pct": {k: round(100 * main_.get(k, 0) / tot, 2) for k in "ABCDE"},
               "fine_pct": {k: round(100 * v / tot, 2) for k, v in fine.items()}, "earth_main_pct": EARTH,
               "lat_bands": bands, "codes": codes}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"→ {a.json}")


if __name__ == "__main__":
    main()
