"""两个后端的对照（`skyisle island compare --run out/seed42 --sample 30 [--jobs N]`；docs/PLAN-CORE.md 第八节，行星计划 P6a / P6b）。

抽样同 `island batch`（主岛面积四分位分层，按 run 的 seed 定）。每群 python、cpp 两个后端各跑一遍完整的 generate + island check
（前 det_first 群另做 IS-det 重跑比哈希），产物写 `islands_compare/<后端>/<节点>/`，不覆盖 `islands/`。逐群比：
  陆地、主岛按目标精确（两边 IS-area 都过）；岛数相同；峰高 ±10%；河长（主岛常年河中心线总长）±20%；
  P6b 起：村数 ±15%、户数相同（人口只读 ⑨）、资源处数（点 / 片 / 赋存区 / 采场）±20%、雨日比例相差 ≤ 0.05、季型相同；两边 island check 全过。
另报可耕率、地表 12 类占比、湖数、河口数、最长河、整套产物是否逐字节相同（island.json 去掉 meta.seconds / engine），
与每群用时（--jobs 1 时群内 4 线程；多进程时每进程 1 线程）。汇总写 islands/compare.json。
`--timing`：顺序跑、不写产物，量整群 generate（地形 → 聚落）的用时：python、cpp 1 线程、cpp 4 线程各一遍，先各热身一次。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

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


def _product_hash(out: Path) -> str:
    """整套产物的哈希（island.json 去掉每次都不同的 meta.seconds 与只在 cpp 后端有的 meta.engine）。"""
    h = hashlib.sha256()
    for p in sorted(out.iterdir()):
        if p.suffix not in (".npz", ".png", ".csv", ".json") or p.name == "check.json":
            continue
        if p.name == "island.json":
            J = json.loads(p.read_text(encoding="utf-8"))
            J["meta"].pop("seconds", None)
            J["meta"].pop("engine", None)
            data = json.dumps(J, sort_keys=True, ensure_ascii=False).encode()
        else:
            data = p.read_bytes()
        h.update(p.name.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def _stats(g: dict, items: list[dict], secs: float, out: Path) -> dict:
    J = g["json"]
    C = J["constraints"]
    H = J.get("hydro", {})
    from .check import exit_code
    R = g.get("resources") or {}
    S = g.get("settle") or {}
    W = (g.get("weather") or {}).get("arrays")
    deps = R.get("deposits", [])
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
        # P6b：聚落、资源、天气、季型
        "n_villages": S.get("n_villages", 0), "n_hamlets": S.get("n_hamlets", 0), "households": S.get("households", 0),
        "n_towns": len(S.get("towns", [])), "n_specials": len(S.get("specials", [])),
        "res_points": sum(1 for d in deps if d["form"] == "point"),
        "res_patches": sum(1 for d in deps if d["form"] == "patch" and not d.get("cleared")),
        "res_occurrences": len(R.get("occurrences", [])), "res_workings": len(R.get("workings", [])),
        "wet_frac": round(float(np.mean(W["wet"])), 4) if W is not None else None,
        "season_type": (g.get("climate") or {}).get("season_type"),
        "t_terrain": round(g["timing"]["terrain"], 3), "t_hydro": round(g["timing"].get("hydro", 0.0), 3),
        "t_gen": round(g["timing"].get("generate", 0.0), 3), "t_generate": round(secs, 2),
        "check": exit_code(items), "fails": [i["id"] for i in items if not i["pass"]],
        "products": _product_hash(out),
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
        row[b] = _stats(g, items, secs, out)
    return row


def _worker(args):
    node, det, sets, year, threads = args
    return _one(_CTX, node, det, sets, year, threads)


CRITERIA = ("area", "n_islands", "peak", "river", "villages", "households", "resources", "wet_frac", "season", "check")


def _judge(row: dict) -> dict:
    p, c = row["python"], row["cpp"]

    def rel(a, b):
        return abs(b - a) / max(abs(a), 1e-9)

    def within(k, tol):
        return (p[k] == 0 and c[k] == 0) or rel(p[k], c[k]) <= tol
    area_ok = all(rel(x["land_target"], x["land_km2"]) < 0.02 and rel(x["main_target"], x["main_km2"]) < 0.02 for x in (p, c))
    j = {"area": area_ok, "n_islands": p["n_islands"] == c["n_islands"], "peak": rel(p["peak_m"], c["peak_m"]) <= 0.10,
         "river": within("river_km", 0.20), "villages": within("n_villages", 0.15), "households": p["households"] == c["households"],
         "resources": all(within(k, 0.20) for k in ("res_points", "res_patches", "res_occurrences", "res_workings")),
         "wet_frac": p["wet_frac"] is None or abs(p["wet_frac"] - c["wet_frac"]) <= 0.05,
         "season": p["season_type"] == c["season_type"], "check": p["check"] == 0 and c["check"] == 0}
    j["identical"] = p["products"] == c["products"]
    j["pass"] = all(j[k] for k in CRITERIA)
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
    cr = s["criteria"]
    print(f"== 对照 {len(rows)} 群：全过 {s['pass']}/{len(rows)}（整套产物逐字节相同 {s['identical']}）；"
          f"陆地主岛 {cr['area']}、岛数 {cr['n_islands']}、峰高 ±10% {cr['peak']}、河长 ±20% {cr['river']}、村数 ±15% {cr['villages']}、"
          f"户数 {cr['households']}、资源处数 ±20% {cr['resources']}、雨日比例 {cr['wet_frac']}、季型 {cr['season']}、check 两边全过 {cr['check']}")
    t = s["timing"]
    print(f"用时中位（地形 + 水系）：python {t['python']['terrain_hydro_median']} s / cpp {t['cpp']['terrain_hydro_median']} s；"
          f"整群 generate（不写产物）中位 python {t['python']['gen_median']} s（最长 {t['python']['gen_max']}）/ cpp {t['cpp']['gen_median']} s"
          f"（最长 {t['cpp']['gen_max']}）；含写产物 python {t['python']['generate_median']} s / cpp {t['cpp']['generate_median']} s")
    print(f"→ {p}")
    return 0 if s["pass"] == len(rows) else 1


def run_timing(ctx, sample: int = 30, nodes: list[int] | None = None, sets: list[str] | None = None, python: bool = True) -> int:
    """整群 generate（地形 → 聚落，不写产物）的用时（`island compare --timing`）：顺序跑，python 一遍（python=False 跳过）、
    cpp 群内 1 线程与 4 线程各一遍；先各热身一次（读行星层 npz 不算进去）。另记地形 + 水系两步（P6a 的口径）。汇总写 islands/timing.json。"""
    from . import generate
    from .batch import pick_nodes
    from .engine import core
    core()
    sets = [s for s in (sets or []) if not s.startswith("engine.")]
    nodes = nodes or pick_nodes(ctx, sample)
    variants = ([("python", 1)] if python else []) + [("cpp", 1), ("cpp", 4)]
    quiet = lambda *a, **k: None

    def once(node, b, th):
        g = generate(ctx, node, sets=sets + [f"engine.backend={b}", f"engine.threads={th}"], log=quiet, write=False)
        T = g["timing"]
        return T["generate"], T["terrain"] + T.get("hydro", 0.0), g["json"]["raster"]["rows"] * g["json"]["raster"]["cols"]
    for b, th in variants:
        once(nodes[0], b, th)
    rows = []
    print(f"[timing] {len(nodes)} 群：整群 generate（s）  " + " | ".join(f"{b} {th} 线程" for b, th in variants))
    for node in nodes:
        r = {"node": node}
        for b, th in variants:
            tg, tth, cells = once(node, b, th)
            r[f"{b}{th}"] = [round(tg, 3), round(tth, 3)]
            r["cells"] = int(cells)
        rows.append(r)
        print(f"  #{node:<5} {r['cells'] / 1e6:5.2f} M 格  " + " | ".join(f"{r[f'{b}{th}'][0]:6.2f}" for b, th in variants), flush=True)
    summ = {}
    for b, th in variants:
        key = f"{b}{th}"
        tg = np.array([r[key][0] for r in rows])
        tth = np.array([r[key][1] for r in rows])
        summ[key] = {"generate_median": round(float(np.median(tg)), 3), "generate_max": round(float(tg.max()), 3),
                     "generate_mean": round(float(tg.mean()), 3), "terrain_hydro_median": round(float(np.median(tth)), 3),
                     "slowest_node": int(rows[int(np.argmax(tg))]["node"])}
    if python:
        sp = [r["python1"][0] / max(1e-9, r["cpp4"][0]) for r in rows]
        summ["speedup_cpp4_median"] = round(float(np.median(sp)), 1)
    out = {"run": ctx.out_dir.name, "seed": ctx.seed, "summary": summ, "rows": rows}
    p = ctx.out_dir / "islands" / "timing.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("== 整群 generate 中位 / 最长：" + "，".join(f"{k} {v['generate_median']} / {v['generate_max']} s" for k, v in summ.items() if isinstance(v, dict))
          + (f"；加速中位 ×{summ['speedup_cpp4_median']}" if python else "") + f" → {p}")
    return 0


def _print_row(r: dict) -> None:
    p, c = r["python"], r["cpp"]
    print(f"  #{r['node']:<5} 岛 {p['n_islands']:2d}/{c['n_islands']:<2d} 峰 {p['peak_m']:7.0f}/{c['peak_m']:<7.0f} 河 {p['river_km']:7.1f}/{c['river_km']:<7.1f} km "
          f"村 {p['n_villages']}/{c['n_villages']} 户 {p['households']}/{c['households']} 赋存区 {p['res_occurrences']}/{c['res_occurrences']} "
          f"采场 {p['res_workings']}/{c['res_workings']} 雨日 {p['wet_frac']}/{c['wet_frac']} "
          f"{'产物相同' if p['products'] == c['products'] else '产物不同'}  generate {p['t_gen']:5.2f}/{c['t_gen']:5.2f} s  "
          f"check {p['check']}/{c['check']}" + (f"  失败 py{p['fails']} cpp{c['fails']}" if p["fails"] or c["fails"] else ""), flush=True)


def _summary(rows: list[dict]) -> dict:
    crit = {k: sum(1 for r in rows if r["judge"][k]) for k in CRITERIA}
    timing = {}
    for b in BACKENDS:
        th = np.array([r[b]["t_terrain"] + r[b]["t_hydro"] for r in rows])
        gen = np.array([r[b]["t_generate"] for r in rows])
        tg = np.array([r[b]["t_gen"] for r in rows])
        timing[b] = {"terrain_hydro_median": round(float(np.median(th)), 3), "terrain_hydro_max": round(float(th.max()), 3),
                     "gen_median": round(float(np.median(tg)), 3), "gen_max": round(float(tg.max()), 3),
                     "generate_median": round(float(np.median(gen)), 2)}
    rel = lambda k: [abs(r["cpp"][k] - r["python"][k]) / max(abs(r["python"][k]), 1e-9) for r in rows]
    return {"pass": sum(1 for r in rows if r["judge"]["pass"]), "identical": sum(1 for r in rows if r["judge"]["identical"]),
            "criteria": crit, "timing": timing,
            "max_rel_diff": {k: round(max(rel(k)), 4) for k in ("peak_m", "river_km", "n_villages", "res_occurrences", "res_workings")},
            "failed_nodes": [r["node"] for r in rows if not r["judge"]["pass"]],
            "not_identical_nodes": [r["node"] for r in rows if not r["judge"]["identical"]]}
