"""取样：三 seed 各挑 4 个节点（按降水从湿到干），生成岛群产物，供日层收缩量的测量用。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skyisle_gen.cli import _ctx_from_run      # noqa: E402
from skyisle_gen import island as isl          # noqa: E402

for seed, run in ((42, "out/seed42"), (7, "out/seed7"), (2026, "out/seed2026")):
    ctx = _ctx_from_run(run)
    isl_npz = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    n = isl_npz["lat"].size
    # 只看有一定规模的群（小岛没有溪涧），按降水四分位取样
    ok = np.where(isl_npz["area_km2"] > 300)[0]
    order = ok[np.argsort(cli["precip"][ok])]
    picks = [int(order[int(q * (order.size - 1))]) for q in (0.05, 0.35, 0.65, 0.95)]
    print(f"seed {seed}: {[int(x) for x in picks]}  降水 {[round(float(cli['precip'][p]), 2) for p in picks]}", flush=True)
    for node in picks:
        out = isl.generate(ctx, node, year=0, log=lambda *a: None)
        print(f"  {node} → {out}", flush=True)
