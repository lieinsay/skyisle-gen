"""第 5 步：抽样批跑（`zhouzhu island batch --run out/seed42 --sample 30`）：看分布与耗时，跑第六节的校验，写 islands/batch.json。

抽样按 run 的 seed 确定（entity_rng 键 island:batch），分层：主岛面积四分位各取四分之一，保证大岛小岛都覆盖。
前 3 个群另做 IS-det（重跑一次比哈希），其余只做单次生成的校验。
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from ..rng import entity_rng
from . import ISLAND_STREAM, generate
from .check import evaluate, exit_code, hash_products


def pick_nodes(ctx, sample: int) -> list[int]:
    isl = ctx.load_npz(3, "islands")
    ma = isl["main_area_km2"].astype(np.float64)
    rng = entity_rng(ctx.seed, ISLAND_STREAM, "island:batch")
    q = np.quantile(ma, [0.25, 0.5, 0.75])
    strata = [np.where(ma <= q[0])[0], np.where((ma > q[0]) & (ma <= q[1]))[0],
              np.where((ma > q[1]) & (ma <= q[2]))[0], np.where(ma > q[2])[0]]
    per = [sample // 4 + (1 if i < sample % 4 else 0) for i in range(4)]
    nodes = []
    for s, k in zip(strata, per):
        if s.size and k > 0:
            nodes += sorted(rng.choice(s, size=min(k, s.size), replace=False).tolist())
    return sorted(int(x) for x in nodes)


def run_batch(ctx, sample: int = 30, year: int = 0, sets: list[str] | None = None, det_first: int = 3) -> int:
    nodes = pick_nodes(ctx, sample)
    rows = []
    worst = 0
    t_all = time.perf_counter()
    for k, node in enumerate(nodes):
        t0 = time.perf_counter()
        out, g = generate(ctx, node, year=year, sets=sets, log=lambda *a: None, return_state=True)
        det = None
        if k < det_first:
            h1 = hash_products(out)
            generate(ctx, node, year=year, sets=sets, log=lambda *a: None)
            det = (h1, hash_products(out))
        items = evaluate(g, out, ctx=ctx, node=node, c=ctx.cfg["island"], det_hashes=det)
        code = exit_code(items)
        worst = max(worst, code)
        J, C = g["json"], g["climate"]
        fails = [i["id"] for i in items if not i["pass"]]
        row = {"node": node, "seconds": round(time.perf_counter() - t0, 1), "res_m": J["meta"]["res_m"], "rows": J["raster"]["rows"], "cols": J["raster"]["cols"],
               "area_km2": J["constraints"]["area_km2"]["target"], "main_km2": J["constraints"]["main_area_km2"]["target"],
               "n_islands": len(J["islands"]), "bridges": J["layout"]["n_bridges"], "ferries": J["layout"]["n_ferries"],
               "age_zh": J["meta"]["age_zh"], "lat": round(J["meta"]["lat"], 1), "season_type": C["season_type_zh"],
               "season_range_c": C["annual"]["season_range_c"], "precip_mm": C["annual"]["precip_mm"],
               "lakes": J["hydro"]["n_lakes"], "large_basins": J["hydro"]["main_basins"].get("n_large", 0),
               "forest": J["landcover"]["share"].get("林地", 0.0), "arable_err": round(abs(J["constraints"]["arable_frac"]["actual"] - J["constraints"]["arable_frac"]["target"]), 5),
               "storm_days": J["weather"]["types"].get("风暴", 0), "fog_days": J["weather"]["types"].get("云海漫顶", 0),
               "sailable_days": J["weather"]["sailable_days"], "code": code, "fails": fails}
        rows.append(row)
        print(f"  #{node:<5} {row['seconds']:5.1f}s  {row['res_m']:.0f} m  {row['rows']}×{row['cols']:<5} 陆地 {row['area_km2']:8.0f} 主岛 {row['main_km2']:7.0f} "
              f"{row['n_islands']:2d} 岛 {row['age_zh']} {row['lat']:6.1f}° {row['season_type']} 温差 {row['season_range_c']:.0f} 雨 {row['precip_mm']:.0f} "
              f"湖 {row['lakes']} 风暴 {row['storm_days']:3d} 雾 {row['fog_days']:3d} 航 {row['sailable_days']:3d}  " + ("全过" if code == 0 else f"失败 {fails}"))
    secs = np.array([r["seconds"] for r in rows])
    summary = {
        "run": ctx.out_dir.name, "seed": ctx.seed, "sample": len(rows), "year": year,
        "seconds": {"median": round(float(np.median(secs)), 1), "max": round(float(secs.max()), 1), "total": round(time.perf_counter() - t_all, 1)},
        "season_types": dict(Counter(r["season_type"] for r in rows)),
        "res_m": dict(Counter(str(int(r["res_m"])) for r in rows)),
        "n_islands": {"median": int(np.median([r["n_islands"] for r in rows])), "min": min(r["n_islands"] for r in rows), "max": max(r["n_islands"] for r in rows)},
        "lakes_share": round(float(np.mean([r["lakes"] > 0 for r in rows])), 3),
        "large_basins_share": round(float(np.mean([r["large_basins"] >= 2 for r in rows])), 3),
        "fog_days_median": int(np.median([r["fog_days"] for r in rows])),
        "storm_days_median": int(np.median([r["storm_days"] for r in rows])),
        "sailable_days_median": int(np.median([r["sailable_days"] for r in rows])),
        "worst_code": worst, "failed_nodes": [r["node"] for r in rows if r["code"]],
        "rows": rows,
    }
    p = ctx.out_dir / "islands" / "batch.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== 批跑 {len(rows)} 群：中位 {summary['seconds']['median']} s，最长 {summary['seconds']['max']} s，总 {summary['seconds']['total']} s；"
          f"季型 {summary['season_types']}；分辨率 {summary['res_m']}；岛数中位 {summary['n_islands']['median']}；"
          f"有湖 {summary['lakes_share']:.0%}，≥2 大盆地 {summary['large_basins_share']:.0%}；"
          f"雾日中位 {summary['fog_days_median']}，风暴日中位 {summary['storm_days_median']}，可出航中位 {summary['sailable_days_median']}")
    print("结果：" + ("全过" if worst == 0 else f"失败 {summary['failed_nodes']}") + f" → {p}")
    return worst
