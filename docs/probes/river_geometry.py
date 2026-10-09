"""独立核对平岸水力几何；条件对照，不替换生成器或写岛群产物。

Parker et al. (2007), DOI 10.1029/2006JF000549, 式 (13)。仅适用于可定义
平岸的单股冲积砾床河。D50 是独立假设，不能用本模型从水深反推的 D50 验证自己。
比较固定洪峰倍数 5 下的同一流量；不使用一年最大流量冒充平岸流量。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skyisle_gen import island
from skyisle_gen.cli import _ctx_from_run
from river_width import connected_widths, measure

SOURCE = "https://doi.org/10.1029/2006JF000549"


def parker_width(q_bf, d50_m):
    """式 (13)，SI 单位；零流量宽度取 0，负流量或非正粒径拒绝。"""
    q = np.asarray(q_bf, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q < 0) or not np.isfinite(d50_m) or d50_m <= 0:
        raise ValueError("Q 必须非负，D50 必须有限且为正")
    return 4.63 / 9.81 ** .2 * q ** .4 * (q / (9.81 ** .5 * d50_m ** 2.5)) ** .0667


def compare(g, node, ratio):
    baseline = measure(g, node)
    ncols = g["height"].shape[1]
    rows, cols = g["recv_i"].ravel(), g["recv_j"].ravel()
    down = rows * ncols + cols
    down[(rows < 0) | (cols < 0)] = -1
    area = g["flowacc_km2"].ravel()
    widths = g["w_ch_m"].ravel()
    main = g["island_id"].ravel() == 0
    cells = g["rivernet"]["cell"]
    q = g["runoff_acc"].ravel() * 1000 / g["year_s"]
    # 原模型固定假设的洪峰倍数；显式输入，不能偷换为 rivernet 的年最大值。
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("假设平岸流量倍数必须有限且为正")
    q_bf = q * ratio
    slope = np.full(len(q), np.nan)
    slope[cells] = g["rivernet"]["slope"]
    # 原论文样本数值范围；数值合格仍不代表河床确为砾床或单股冲积河。
    eligible = ((q_bf >= 2.7) & (q_bf <= 5440) & (slope >= .00034) &
                (slope <= .031) & (g["confine"].ravel() >= 2))
    edge = cells[(down[cells] >= 0)]
    membership = np.zeros(len(q), bool)
    membership[cells] = True
    edge = edge[membership[down[edge]] & main[edge] & main[down[edge]]]
    step = np.hypot(edge // ncols - down[edge] // ncols,
                    edge % ncols - down[edge] % ncols) * g["res_km"]
    both = eligible[edge] & eligible[down[edge]]
    valid_length = float(step[both].sum())
    scenarios = []
    for d50_mm in [30, 50, 100, 167.5]:
        predicted = parker_width(q_bf, d50_mm / 1000)
        conditional = widths.copy()
        conditional[eligible] = predicted[eligible]
        ratios = predicted[edge[both]] / widths[edge[both]]
        scenarios.append({"assumed_d50_mm": d50_mm,
                          "width_ratio_median": round(float(np.median(ratios)), 3) if len(ratios) else None,
                          "main": connected_widths(cells, down, area, conditional, ncols, g["res_km"], main),
                          "main_max_channel_m": round(float(conditional[main].max()), 3)})
    return {"node": node, "baseline": baseline, "assumed_bankfull_ratio": ratio,
            "numerically_eligible_main_length_km": round(valid_length, 3),
            "numerically_eligible_main_length_share": round(valid_length / step.sum(), 5) if step.sum() else 0,
            "scenarios": scenarios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="out/seed42")
    parser.add_argument("--nodes", default="6068,5758,6329")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--out", default="out/river-audit/geometry-comparison.json")
    args = parser.parse_args()
    results = []
    for node in map(int, args.nodes.split(",")):
        ctx = _ctx_from_run(args.run)
        g = island.generate(ctx, node, steps=4, sets=args.set,
                            write=False, log=lambda *a: None)
        row = compare(g, node, ctx.cfg["island"]["hydro"]["bf_ratio_channel"])
        results.append(row)
        print(json.dumps({"node": node, "eligible_length_km": row["numerically_eligible_main_length_km"],
                          "continuous_ge15_km": [row["baseline"]["main"]["continuous_ge_m_max_km"]["15"]] +
                          [x["main"]["continuous_ge_m_max_km"]["15"] for x in row["scenarios"]]}), flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": SOURCE, "run": args.run, "sets": args.set,
                               "note": "条件对照：仅数值范围内且非峡谷的点替换宽度，其他点保留原宽。粒径为独立假设，冲积/砾床资格尚未独立验证，结果不是建议默认值。地形、流量、雨量均未改变。",
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
