"""浮高的全行星统计（`skyisle island floats --run out/seed42 [--jobs N]`；DESIGN-NOTES 四点二十八）。

每群只跑布局 + 地形（build_terrain：浮高在这一步定，往下的要等地形拟合出岸缘才按余量缩），收全部非主岛的
浮高 δ、岛龄差（岛龄 − 主岛岛龄）、面积、平移后的岸缘 / 台面（P5 起没有索桥与导水槽，原来每群的索桥数一栏删掉）。
写 islands/float_stats.json（摘要 + 与标定区间的对照）与 islands/float_stats.npz（逐岛数组）；不在标定区间 → 退出码 1。
cpp 后端 8000 群 30 进程约一两分钟；python 后端慢十来倍。
"""
from __future__ import annotations

import json
import time

import numpy as np

# 标定区间（浮高计划 G 期：seed 42 / 7 / 2026 的全部非主岛）；max 的上界取 up_max_m
CALIB = {"abs_median_m": (300.0, 600.0), "abs_p90_m": (800.0, 1200.0), "max_m": (1300.0, None), "up_share": (0.6, 0.8),
         "spearman_max": -0.4}

_CTX = None


def _worker_init(run_dir: str):
    global _CTX
    from ..cli import _ctx_from_run
    _CTX = _ctx_from_run(run_dir)


def _worker(args):
    node, sets = args
    from . import _node_inputs, build_terrain, island_config
    c = island_config(_CTX, sets)
    inp = _node_inputs(_CTX, node)
    g = build_terrain(_CTX, node, c, inp, log=lambda *a, **k: None)
    J = g["json"]
    I = J["islands"]
    a0 = float(I[0]["age"])
    rows = [(node, int(i["id"]), float(i["area_km2"]), float(i["age"]) - a0, float(i["float_m"]), float(i["rim_m"]), float(i["surface_m"]))
            for i in I[1:]]
    grp = (node, float(inp["height_m"]), len(I))
    return rows, grp


def rank_avg(x: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均，1 起）：斯皮尔曼秩相关用。"""
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    return float(np.corrcoef(rank_avg(x), rank_avg(y))[0, 1])


def summarize(d: np.ndarray, dage: np.ndarray, rim: np.ndarray, c: dict) -> dict:
    """非主岛的浮高分布摘要 + 与标定区间的对照（calib：各项是否在区间里）。"""
    ad = np.abs(d)
    up_max = float(c.get("up_max_m", 1500.0))
    s = {"n": int(d.size), "up_share": round(float((d > 0).mean()), 4), "down_share": round(float((d < 0).mean()), 4),
         "abs_median_m": round(float(np.median(ad)), 1), "abs_p90_m": round(float(np.percentile(ad, 90)), 1),
         "abs_p99_m": round(float(np.percentile(ad, 99)), 1), "max_m": round(float(d.max()), 1), "min_m": round(float(d.min()), 1),
         "up_median_m": round(float(np.median(d[d > 0])), 1) if (d > 0).any() else None,
         "down_median_m": round(float(np.median(d[d < 0])), 1) if (d < 0).any() else None,
         "spearman_age": round(spearman(d, dage), 4), "rim_min_m": round(float(rim.min()), 1),
         "rim_below_floor": int((rim < float(c.get("rim_floor_m", 20.0))).sum())}
    ok = {"abs_median_m": CALIB["abs_median_m"][0] <= s["abs_median_m"] <= CALIB["abs_median_m"][1],
          "abs_p90_m": CALIB["abs_p90_m"][0] <= s["abs_p90_m"] <= CALIB["abs_p90_m"][1],
          "max_m": CALIB["max_m"][0] <= s["max_m"] <= up_max,
          "up_share": CALIB["up_share"][0] <= s["up_share"] <= CALIB["up_share"][1],
          "spearman_age": s["spearman_age"] <= CALIB["spearman_max"],
          "rim_floor": s["rim_below_floor"] == 0}
    s["calib"] = ok
    s["calib_pass"] = all(ok.values())
    return s


def run_floats(ctx, jobs: int = 1, sets: list[str] | None = None, nodes: list[int] | None = None, log=print) -> int:
    from concurrent.futures import ProcessPoolExecutor
    from . import island_config
    c = island_config(ctx, sets)
    fc = c.get("float") or {}
    n_all = ctx.load_npz(3, "islands")["lat"].size
    nodes = list(range(n_all)) if nodes is None else nodes
    log(f"[floats] {len(nodes)} 群，{jobs} 进程，浮高 {'开' if fc.get('enabled') else '关'}")
    t0 = time.time()
    rows, grps = [], []
    with ProcessPoolExecutor(max(1, jobs), initializer=_worker_init, initargs=(str(ctx.out_dir),)) as ex:
        for i, (r, gp) in enumerate(ex.map(_worker, [(k, sets) for k in nodes], chunksize=4)):
            rows += r
            grps.append(gp)
            if (i + 1) % 1000 == 0 or i + 1 == len(nodes):
                el = time.time() - t0
                log(f"  {i + 1}/{len(nodes)}  {el / 60:.1f} min")
    A = np.array([r[:2] for r in rows], dtype=np.int32).reshape(-1, 2)
    F = np.array([r[2:] for r in rows], dtype=np.float64).reshape(-1, 5)
    G = np.array(grps, dtype=np.float64).reshape(-1, 3)
    area, dage, d, rim, surf = F.T
    summ = summarize(d, dage, rim, fc) if d.size else {"n": 0}
    # 按主岛台面分档：低台面的群往下挪得少
    bins = [0, 400, 800, 1600, 1e9]
    h_of = {int(g[0]): g[1] for g in G}
    hm = np.array([h_of[int(n)] for n in A[:, 0]]) if A.size else np.zeros(0)
    by_h = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (hm >= lo) & (hm < hi)
        if m.any():
            by_h.append({"main_surface_m": [lo, None if hi >= 1e9 else hi], "n": int(m.sum()), "up_share": round(float((d[m] > 0).mean()), 3),
                         "abs_median_m": round(float(np.median(np.abs(d[m]))), 1),
                         "down_median_m": round(float(np.median(d[m][d[m] < 0])), 1) if (d[m] < 0).any() else None})
    out = {"run": ctx.out_dir.name, "seed": ctx.seed, "groups": len(grps), "float": {k: v for k, v in fc.items()},
           "calib_targets": {k: list(v) if isinstance(v, tuple) else v for k, v in CALIB.items()},
           "non_main": summ, "by_main_surface": by_h,
           "islands": int(G[:, 2].sum()) if G.size else 0, "minutes": round((time.time() - t0) / 60.0, 2)}
    root = ctx.out_dir / "islands"
    root.mkdir(parents=True, exist_ok=True)
    tag = "" if fc.get("enabled") else "_off"
    (root / f"float_stats{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(root / f"float_stats{tag}.npz", node=A[:, 0], island=A[:, 1], area_km2=area, dage=dage, float_m=d, rim_m=rim,
                        surface_m=surf, group_node=G[:, 0].astype(np.int32), group_height_m=G[:, 1], group_islands=G[:, 2].astype(np.int32))
    s = summ
    if s.get("n"):
        log(f"[floats] 非主岛 {s['n']}：往上 {s['up_share']:.1%}，|δ| 中位 {s['abs_median_m']:.0f} m、p90 {s['abs_p90_m']:.0f}、p99 {s['abs_p99_m']:.0f}，"
            f"最高 {s['max_m']:+.0f}、最低 {s['min_m']:+.0f}；往上中位 {s['up_median_m']}、往下中位 {s['down_median_m']}；"
            f"δ 与 Δ岛龄秩相关 {s['spearman_age']:+.3f}；岸缘最低 {s['rim_min_m']:.1f} m")
        for b in by_h:
            log(f"  主岛台面 {b['main_surface_m']}：{b['n']} 岛，往上 {b['up_share']:.0%}，|δ| 中位 {b['abs_median_m']:.0f}，往下中位 {b['down_median_m']}")
    log(f"[floats] 共 {out['islands']} 岛；"
        f"标定 {'全在区间里' if s.get('calib_pass') else '有不在区间的：' + str([k for k, v in s.get('calib', {}).items() if not v])} → {root / f'float_stats{tag}.json'}")
    return 0 if s.get("calib_pass") or not fc.get("enabled") else 1
