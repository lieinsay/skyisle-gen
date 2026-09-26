"""两个后端的对照（`skyisle island compare --run out/seed42 --sample 30 [--jobs N]`；docs/PLAN-CORE.md 第八节，行星计划 P6a）。

抽样同 `island batch`（主岛面积四分位分层，按 run 的 seed 定）。每群 python、cpp 两个后端各跑一遍完整的 generate + island check
（前 det_first 群另做 IS-det 重跑比哈希），产物写 `islands_compare/<后端>/<节点>/`，不覆盖 `islands/`。逐群比：
  陆地、主岛按目标精确（两边 IS-area 都过）；岛数相同；峰高 ±10%；河长（主岛常年河中心线总长）±20%；两边 island check 全过。
另报可耕率、地表 12 类占比、湖数、河口数、最长河，与每群地形 / 水系两步的用时（--jobs 1 时群内 4 线程；多进程时每进程 1 线程）。
汇总写 islands/compare.json。
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

BACKENDS = ("python", "cpp")
_CTX = None


def _worker_init(run_dir: str):
    global _CTX
    from ..cli import _ctx_from_run
    _CTX = _ctx_from_run(run_dir)


def _river_len_km(g: dict) -> float:
    """主岛常年河（级别 > 0 的段）中心线总长（km）：rivers.json 的折线逐段量。"""
    res = float(g["res_km"])
    tot = 0.0
    for L in g.get("river_lines", []):
        if L["island"] != 0:
            continue
        p = L["pts"]
        for a, b in zip(p[:-1], p[1:]):
            if a[3] > 0 and b[3] > 0:
                tot += math.hypot(b[0] - a[0], b[1] - a[1]) * res
    return tot


def _stats(g: dict, items: list[dict], secs: float) -> dict:
    J = g["json"]
    C = J["constraints"]
    H = J.get("hydro", {})
    from .check import exit_code
    return {
        "land_km2": C["area_km2"]["actual"], "land_target": C["area_km2"]["target"],
        "main_km2": C["main_area_km2"]["actual"], "main_target": C["main_area_km2"]["target"],
        "n_islands": len(J["islands"]), "peak_m": C["peak_m"]["actual"], "surface_m": C["height_m"]["actual"],
        "relief_m": J["islands"][0]["relief_m"], "river_km": round(_river_len_km(g), 2),
        "longest_river_km": max([r["length_km"] for r in H.get("rivers", [])] or [0.0]),
        "n_rivers": H.get("n_rivers", 0), "n_lakes": H.get("n_lakes", 0),
        "arable": C["arable_frac"].get("actual"), "arable_target": C["arable_frac"]["target"],
        "landcover": J.get("landcover", {}).get("share", {}), "bridges": J["layout"]["n_bridges"],
        "constrained": bool(C.get("territory", {}).get("constrained")),
        "t_terrain": round(g["timing"]["terrain"], 3), "t_hydro": round(g["timing"].get("hydro", 0.0), 3), "t_generate": round(secs, 2),
        "check": exit_code(items), "fails": [i["id"] for i in items if not i["pass"]],
    }


def _one(ctx, node: int, det: bool, sets: list[str], year: int, threads: int) -> dict:
    from . import generate
    from .check import evaluate, hash_products
    row = {"node": node, "threads": threads}
    root = ctx.out_dir / "islands_compare"
    quiet = lambda *a, **k: None
    for b in BACKENDS:
        s = list(sets) + [f"engine.backend={b}", f"engine.threads={threads}"]
        t0 = time.perf_counter()
        out, g = generate(ctx, node, year=year, sets=s, log=quiet, return_state=True, out_root=root / b)
        secs = time.perf_counter() - t0
        dh = None
        if det:
            h1 = hash_products(out)
            generate(ctx, node, year=year, sets=s, log=quiet, out_root=root / b)
            dh = (h1, hash_products(out))
        items = evaluate(g, out, ctx=ctx, node=node, c=ctx.cfg["island"], det_hashes=dh)
        row[b] = _stats(g, items, secs)
    return row


def _worker(args):
    node, det, sets, year, threads = args
    return _one(_CTX, node, det, sets, year, threads)


def _judge(row: dict) -> dict:
    p, c = row["python"], row["cpp"]

    def rel(a, b):
        return abs(b - a) / max(abs(a), 1e-9)
    area_ok = all(rel(x["land_target"], x["land_km2"]) < 0.02 and rel(x["main_target"], x["main_km2"]) < 0.02 for x in (p, c))
    river_ok = (p["river_km"] == 0 and c["river_km"] == 0) or rel(p["river_km"], c["river_km"]) <= 0.20
    j = {"area": area_ok, "n_islands": p["n_islands"] == c["n_islands"], "peak": rel(p["peak_m"], c["peak_m"]) <= 0.10,
         "river": river_ok, "check": p["check"] == 0 and c["check"] == 0}
    j["identical"] = all(p[k] == c[k] for k in ("land_km2", "main_km2", "n_islands", "peak_m", "river_km", "arable", "landcover", "n_lakes", "n_rivers"))
    j["pass"] = all(j[k] for k in ("area", "n_islands", "peak", "river", "check"))
    return j


def run_compare(ctx, sample: int = 30, jobs: int = 1, sets: list[str] | None = None, nodes: list[int] | None = None,
                det_first: int = 3, year: int = 0) -> int:
    from .batch import pick_nodes
    from .engine import core
    core()                                   # 扩展没编就先报错
    sets = [s for s in (sets or []) if not s.startswith("engine.backend=")]
    nodes = nodes or pick_nodes(ctx, sample)
    threads = 4 if jobs <= 1 else 1
    print(f"[compare] {len(nodes)} 群，{jobs} 进程（群内 {threads} 线程）→ {ctx.out_dir / 'islands_compare'}")
    t_all = time.perf_counter()
    args = [(n, k < det_first, sets, year, threads) for k, n in enumerate(nodes)]
    rows = []
    if jobs <= 1:
        for a in args:
            rows.append(_one(ctx, *a))
            _print_row(rows[-1])
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs, initializer=_worker_init, initargs=(str(ctx.out_dir),)) as ex:
            for r in ex.map(_worker, args, chunksize=1):
                rows.append(r)
                _print_row(r)
    for r in rows:
        r["judge"] = _judge(r)
    summary = _summary(rows)
    summary.update({"run": ctx.out_dir.name, "seed": ctx.seed, "jobs": jobs, "threads": threads,
                    "seconds_total": round(time.perf_counter() - t_all, 1), "rows": rows})
    p = ctx.out_dir / "islands" / "compare.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    s = summary
    print(f"== 对照 {len(rows)} 群：全过 {s['pass']}/{len(rows)}（逐群完全相同 {s['identical']}）；"
          f"陆地主岛 {s['criteria']['area']}、岛数 {s['criteria']['n_islands']}、峰高 ±10% {s['criteria']['peak']}、河长 ±20% {s['criteria']['river']}、"
          f"check 两边全过 {s['criteria']['check']}")
    t = s["timing"]
    print(f"用时中位（地形 + 水系）：python {t['python']['terrain_hydro_median']} s（最长 {t['python']['terrain_hydro_max']}），"
          f"cpp {t['cpp']['terrain_hydro_median']} s（最长 {t['cpp']['terrain_hydro_max']}）；整群 generate 中位 python {t['python']['generate_median']} s / cpp {t['cpp']['generate_median']} s")
    print(f"→ {p}")
    return 0 if s["pass"] == len(rows) else 1


def run_timing(ctx, sample: int = 30, nodes: list[int] | None = None, sets: list[str] | None = None) -> int:
    """只量地形 + 水系两步的用时（`island compare --timing`）：顺序跑，python 一遍、cpp 群内 1 线程与 4 线程各一遍，不写产物。
    先各热身一次（读行星层 npz 不算进去）。汇总写 islands/timing.json。"""
    from . import _node_inputs, build_terrain, island_config
    from .batch import pick_nodes
    from .engine import core
    from .hydro import build_hydro
    core()
    sets = [s for s in (sets or []) if not s.startswith("engine.")]
    nodes = nodes or pick_nodes(ctx, sample)
    variants = [("python", 1), ("cpp", 1), ("cpp", 4)]
    quiet = lambda *a, **k: None

    def once(node, b, th):
        c = island_config(ctx, sets + [f"engine.backend={b}", f"engine.threads={th}"])
        inp = _node_inputs(ctx, node)
        t0 = time.perf_counter()
        g = build_terrain(ctx, node, c, inp, log=quiet)
        t1 = time.perf_counter()
        build_hydro(ctx, node, c, g, log=quiet)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, g["json"]["raster"]["rows"] * g["json"]["raster"]["cols"]
    for b, th in variants:
        once(nodes[0], b, th)
    rows = []
    print(f"[timing] {len(nodes)} 群：地形 + 水系（s）  python | cpp 1 线程 | cpp 4 线程")
    for node in nodes:
        r = {"node": node}
        for b, th in variants:
            tt, th_, cells = once(node, b, th)
            r[f"{b}{th}"] = [round(tt, 3), round(th_, 3)]
            r["cells"] = int(cells)
        rows.append(r)
        print(f"  #{node:<5} {r['cells'] / 1e6:5.2f} M 格  {sum(r['python1']):6.2f} | {sum(r['cpp1']):5.2f} | {sum(r['cpp4']):5.2f}")
    summ = {}
    for key in ("python1", "cpp1", "cpp4"):
        tot = np.array([sum(r[key]) for r in rows])
        summ[key] = {"terrain_median": round(float(np.median([r[key][0] for r in rows])), 3),
                     "hydro_median": round(float(np.median([r[key][1] for r in rows])), 3),
                     "total_median": round(float(np.median(tot)), 3), "total_max": round(float(tot.max()), 3),
                     "total_mean": round(float(tot.mean()), 3)}
    sp = [sum(r["python1"]) / max(1e-9, sum(r["cpp4"])) for r in rows]
    summ["speedup_cpp4_median"] = round(float(np.median(sp)), 1)
    out = {"run": ctx.out_dir.name, "seed": ctx.seed, "summary": summ, "rows": rows}
    p = ctx.out_dir / "islands" / "timing.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== 中位 地形 + 水系：python {summ['python1']['total_median']} s（最长 {summ['python1']['total_max']}），"
          f"cpp 1 线程 {summ['cpp1']['total_median']} s（最长 {summ['cpp1']['total_max']}），cpp 4 线程 {summ['cpp4']['total_median']} s"
          f"（最长 {summ['cpp4']['total_max']}）；加速中位 ×{summ['speedup_cpp4_median']} → {p}")
    return 0


def _print_row(r: dict) -> None:
    p, c = r["python"], r["cpp"]
    print(f"  #{r['node']:<5} 岛 {p['n_islands']:2d}/{c['n_islands']:<2d} 峰 {p['peak_m']:7.0f}/{c['peak_m']:<7.0f} 河 {p['river_km']:7.1f}/{c['river_km']:<7.1f} km "
          f"可耕 {p['arable']:.4f}/{c['arable']:.4f} 湖 {p['n_lakes']}/{c['n_lakes']}  地形+水系 {p['t_terrain'] + p['t_hydro']:5.2f}/{c['t_terrain'] + c['t_hydro']:5.2f} s  "
          f"check {p['check']}/{c['check']}" + (f"  失败 py{p['fails']} cpp{c['fails']}" if p["fails"] or c["fails"] else ""))


def _summary(rows: list[dict]) -> dict:
    crit = {k: sum(1 for r in rows if r["judge"][k]) for k in ("area", "n_islands", "peak", "river", "check")}
    timing = {}
    for b in BACKENDS:
        th = np.array([r[b]["t_terrain"] + r[b]["t_hydro"] for r in rows])
        tt = np.array([r[b]["t_terrain"] for r in rows])
        hy = np.array([r[b]["t_hydro"] for r in rows])
        gen = np.array([r[b]["t_generate"] for r in rows])
        timing[b] = {"terrain_median": round(float(np.median(tt)), 3), "hydro_median": round(float(np.median(hy)), 3),
                     "terrain_hydro_median": round(float(np.median(th)), 3), "terrain_hydro_max": round(float(th.max()), 3),
                     "terrain_hydro_mean": round(float(th.mean()), 3), "generate_median": round(float(np.median(gen)), 2)}
    rel = lambda k: [abs(r["cpp"][k] - r["python"][k]) / max(abs(r["python"][k]), 1e-9) for r in rows]
    return {"pass": sum(1 for r in rows if r["judge"]["pass"]), "identical": sum(1 for r in rows if r["judge"]["identical"]),
            "criteria": crit, "timing": timing,
            "max_rel_diff": {"peak_m": round(max(rel("peak_m")), 4), "river_km": round(max(rel("river_km")), 4)},
            "failed_nodes": [r["node"] for r in rows if not r["judge"]["pass"]]}
