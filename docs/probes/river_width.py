"""按实际 D8 连通边量河宽、长度占比与连续宽河段；研究量具，不修改产物。

仓库根执行：python docs/probes/river_width.py --run out/seed42 --nodes 6068,5758,6329
--window 量 seed42 中 28–40°N、20–35°E、雨量 >=800 mm、主岛 >=300 km² 的全部节点。
--set island.xxx=value 为独立试算：每个节点重新读取配置，避免覆盖值串入下个节点。
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

BINS = [0, 1, 3, 5, 10, 15, 30, np.inf]
THRESHOLDS = [5, 10, 15, 20, 30]


def connected_widths(cells, down, area, width, columns, res_km, selected=None):
    """长度按 D8 直/斜边；含支流与干流间的接合边，不依赖 segments 的拆分方式。"""
    cells = np.asarray(cells, dtype=np.int64)
    member = np.zeros(len(down), dtype=bool)
    member[cells] = True
    safe_down = np.maximum(down[cells], 0)
    edges = cells[(down[cells] >= 0) & member[safe_down]]
    if selected is not None:
        edges = edges[selected[edges] & selected[down[edges]]]
    destinations = down[edges]
    lengths = np.hypot(edges // columns - destinations // columns,
                       edges % columns - destinations % columns) * res_km
    weights = np.histogram((width[edges] + width[destinations]) / 2, BINS, weights=lengths)[0]
    # 汇水面积沿流向严格增大（每格加本格面积）；上游先处理，汇合处取最长支路。
    ordered = edges[np.argsort(area[edges], kind="stable")]
    edge_length = dict(zip(edges.tolist(), lengths.tolist()))
    longest = {}
    totals = {}
    for threshold in THRESHOLDS:
        accumulated = {}
        total = maximum = 0.0
        for k in ordered:
            r = down[k]
            if width[k] < threshold or width[r] < threshold:
                continue
            distance = edge_length[k]
            total += distance
            value = accumulated.get(int(k), 0.0) + distance
            accumulated[int(r)] = max(accumulated.get(int(r), 0.0), value)
            maximum = max(maximum, value)
        longest[str(threshold)] = round(maximum, 3)
        totals[str(threshold)] = round(total, 3)
    return {"length_km": round(float(lengths.sum()), 3),
            "width_length_share": (weights / weights.sum()).round(6).tolist() if weights.sum() else [0.0] * 7,
            "continuous_ge_m_max_km": longest, "total_ge_m_km": totals}


def measure(g, node):
    columns = g["height"].shape[1]
    rows, cols = g["recv_i"].ravel(), g["recv_j"].ravel()
    down = rows.astype(np.int64) * columns + cols
    down[(rows < 0) | (cols < 0)] = -1
    ids = g["island_id"].ravel()
    area = g["flowacc_km2"].ravel()
    width = g["w_ch_m"].ravel()
    cells = g["rivernet"]["cell"]
    land, main = ids >= 0, ids == 0
    q = g["runoff_acc"].ravel() * 1000.0 / g["year_s"]
    outlets = np.flatnonzero(main & (down < 0))
    largest = int(outlets[np.argmax(area[outlets])])
    wettest = int(outlets[np.argmax(q[outlets])])
    per_island = []
    for k in range(len(g["json"]["islands"])):
        selected = ids == k
        mouths = selected & (down < 0)
        per_island.append({"island": k, "area_km2": round(float(selected.sum() * g["res_km"] ** 2), 2),
                           "max_basin_km2": round(float(area[mouths].max()), 2),
                           "max_channel_m": round(float(width[selected].max()), 3)})
    result = {"node": node, "islands": per_island,
              "main_rain_mm": round(float(g["rain_mm"].ravel()[main].mean()), 2),
              "main_runoff_mm": round(float(g["runoff_mm"].ravel()[main].mean()), 2),
              "largest_main_basin_km2": round(float(area[largest]), 2),
              "wettest_main_mouth": {"cell": [wettest // columns, wettest % columns],
                                     "basin_km2": round(float(area[wettest]), 2),
                                     "q_mean_m3s": round(float(q[wettest]), 4),
                                     "channel_m": round(float(width[wettest]), 3)},
              "all": connected_widths(cells, down, area, width, columns, g["res_km"]),
              "main": connected_widths(cells, down, area, width, columns, g["res_km"], main),
              "river": connected_widths(cells, down, area, width, columns, g["res_km"], g["river"].ravel() > 0),
              "runoff_budget_m3s": round(float(g["runoff_mm"].ravel()[land].sum() * g["res_km"] ** 2 * 1000 / g["year_s"]), 5),
              "outlet_budget_m3s": round(float(q[land & (down < 0)].sum()), 5)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="out/seed42")
    parser.add_argument("--nodes", default="6068,5758,6329")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--out", default="out/river-audit/connected-widths.json")
    args = parser.parse_args()
    ctx = _ctx_from_run(args.run)
    if args.window:
        layout = ctx.load_npz(3, "islands")
        climate = ctx.load_npz(4, "climate_islands")
        mask = ((layout["lat"] >= 28) & (layout["lat"] < 40) &
                (layout["lon"] >= 20) & (layout["lon"] <= 35) &
                (climate["precip_mm"] >= 800) & (layout["main_area_km2"] >= 300))
        nodes = np.flatnonzero(mask).tolist()
    else:
        nodes = [int(x) for x in args.nodes.split(",")]
    results = []
    for node in nodes:
        fresh = _ctx_from_run(args.run)
        g = island.generate(fresh, node, steps=4, sets=args.set, write=False, log=lambda *a: None)
        row = measure(g, node)
        results.append(row)
        print(json.dumps({"node": node, "largest_main_basin_km2": row["largest_main_basin_km2"],
                          "main_max_width_m": row["islands"][0]["max_channel_m"],
                          "main_continuous_ge15_km": row["main"]["continuous_ge_m_max_km"]["15"]}), flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": args.run, "island_version": island.ISLAND_VERSION,
                                "sets": args.set, "nodes": nodes, "width_bins_m": ["<1", "1–3", "3–5", "5–10", "10–15", "15–30", "≥30"],
                                "note": "量平岸河槽，不是当天水面；按 D8 连通边计长，含接合边。季节性细沟包含在 all；river 是年均流量门槛分类，不保证全年有水。",
                                "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
