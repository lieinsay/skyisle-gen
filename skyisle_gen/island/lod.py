"""岛群粗版（`skyisle island lod`）：给远处看的低分辨率版本，全行星批跑（Zhouzhu 行星计划 P0，DESIGN-NOTES 四点二十二）。

不在粗分辨率上重新生成：群内布局随分辨率而变（2051 在 1 km 下生成，岛心与 100 m 版差中位 24 km），远处的样子会和走近后对不上。
所以照原生分辨率跑布局 + 地形 + 水系（带势力范围约束，与 `skyisle island <节点>` 同一份），再按块降采样：
    land      块内陆地占比（0–255）
    height    块内陆地的平均高程（m，云带顶以上；没有陆地为 NaN）
    peak      块内陆地的最高点（m；远处的山的剪影靠它）
    island    块内陆地最多的岛号（−1 = 虚空）
    landcover 块内陆地最多的地表类（output.LANDCOVER_CLASSES 的下标）
    water     块内河道与湖的占比（0–255）
每群一个 `out/<run>/islands_lod/<分辨率>/<节点>.npz`（外加 meta：栅格头、各岛的岸缘 / 岛底 / 峰、势力范围记录），`index.json` 汇总。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

LOD_DIR = "islands_lod"


def _block_reduce(g: dict, f: int) -> dict:
    """按 f × f 块降采样（栅格右 / 下边补虚空到 f 的整数倍）。"""
    island = g["island_id"]
    H, W = island.shape
    Hb, Wb = -(-H // f), -(-W // f)

    def pad(a, fill):
        out = np.full((Hb * f, Wb * f), fill, dtype=a.dtype)
        out[:H, :W] = a
        return out.reshape(Hb, f, Wb, f).transpose(0, 2, 1, 3).reshape(Hb, Wb, f * f)

    isl = pad(island.astype(np.int16), np.int16(-1))
    land = isl >= 0
    n_land = land.sum(-1)
    h = pad(np.where(island >= 0, g["height"], np.nan).astype(np.float32), np.float32(np.nan))
    with np.errstate(invalid="ignore"):
        hmean = np.where(n_land > 0, np.nansum(h, -1) / np.maximum(n_land, 1), np.nan).astype(np.float32)
    hpeak = np.where(n_land > 0, np.nanmax(np.where(land, h, -np.inf), -1), np.nan).astype(np.float32)
    water = pad((((g["river"] > 0) | g["lake"]) & (island >= 0)).astype(np.uint8), np.uint8(0))

    def mode(vals, nclass, offset=0):
        """块内陆地格的众数（vals 已按块排好；非陆地不计）。"""
        out = np.full(vals.shape[:2], -1, dtype=np.int16)
        v = vals.astype(np.int32) + offset
        counts = np.zeros(vals.shape[:2] + (nclass,), dtype=np.int32)
        for k in range(nclass):
            counts[..., k] = ((v == k) & land).sum(-1)
        best = counts.argmax(-1)
        return np.where(n_land > 0, best - offset, out).astype(np.int16)

    n_isl = int(island.max()) + 1 if (island >= 0).any() else 0
    return {
        "land": np.round(255.0 * n_land / (f * f)).astype(np.uint8),
        "height": hmean,
        "peak": hpeak,
        "island": mode(isl, max(1, n_isl)),
        "landcover": mode(pad(g["landcover"].astype(np.uint8), np.uint8(0)), 12).astype(np.uint8),
        "water": np.round(255.0 * water.sum(-1) / (f * f)).astype(np.uint8),
    }


def build_lod(ctx, node: int, c: dict, res_list: list[float], native_res_m: float | None = None) -> dict:
    """原生分辨率跑布局 + 地形 + 水系，按 res_list 各降采样一份。返回 {res: (arrays, meta)}。
    native_res_m 只给测试用（小世界用粗的「原生」跑得快）；正式批跑一律用 island.res_m。"""
    from . import _node_inputs, build_terrain
    from .hydro import build_hydro
    t0 = time.perf_counter()
    inp = _node_inputs(ctx, node)
    quiet = lambda *a, **k: None
    g = build_terrain(ctx, node, c, inp, res_m=native_res_m, log=quiet)
    build_hydro(ctx, node, c, g, log=quiet)
    J = g["json"]
    native = float(J["raster"]["res_m"])
    out = {}
    for res in res_list:
        f = max(1, int(round(float(res) / native)))
        arr = _block_reduce(g, f)
        Hb, Wb = arr["land"].shape
        meta = {
            "node": node, "seed": ctx.seed, "lat": inp["lat"], "lon": inp["lon"],
            "raster": {"res_m": native * f, "rows": Hb, "cols": Wb, "origin_km": J["raster"]["origin_km"],
                       "native_res_m": native, "factor": f,
                       "note": "格 (r, c) 覆盖原生栅格的 [r·f, (r+1)·f) × [c·f, (c+1)·f)；格心 = origin + ((c+0.5)·res, −(r+0.5)·res)（km，x 东 y 北）"},
            "islands": [{k: i[k] for k in ("id", "is_main", "area_km2", "center_km", "surface_m", "rim_m", "keel_m", "peak_m", "cliff_m", "age_zh")}
                        for i in J["islands"]],
            "territory": {k: v for k, v in J["constraints"].get("territory", {}).items() if k != "note"},
            "seconds": round(time.perf_counter() - t0, 2),
        }
        out[float(res)] = (arr, meta)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_lod(root: Path, res: float, arr: dict, meta: dict) -> Path:
    d = root / f"{int(round(res))}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{meta['node']}.npz"
    np.savez_compressed(p, meta=np.array(json.dumps(_jsonable(meta), ensure_ascii=False)), **arr)
    return p


_CTX = None


def _worker_init(run_dir: str):
    global _CTX
    from ..cli import _ctx_from_run
    _CTX = _ctx_from_run(run_dir)


def _worker(args):
    node, res_list, sets = args
    from . import island_config
    c = island_config(_CTX, sets)
    root = _CTX.out_dir / LOD_DIR
    rows = []
    for res, (arr, meta) in build_lod(_CTX, node, c, res_list).items():
        write_lod(root, res, arr, meta)
        t = meta["territory"]
        rows.append({"node": node, "res": res, "rows": meta["raster"]["rows"], "cols": meta["raster"]["cols"],
                     "seconds": meta["seconds"], "constrained": bool(t.get("constrained")), "violation_km": t.get("violation_km")})
    return rows


def select_nodes(ctx, nodes: str | None, near: int | None, radius_km: float | None) -> list[int]:
    isl = ctx.load_npz(3, "islands")
    n = isl["lat"].size
    if nodes:
        return [int(x) for x in nodes.split(",") if x.strip()]
    if near is not None:
        R = float(ctx.load_json(1, "planet")["radius_km"])
        lat, lon = np.radians(isl["lat"].astype(float)), np.radians(isl["lon"].astype(float))
        p0, l0 = lat[near], lon[near]
        d = R * np.arccos(np.clip(np.sin(p0) * np.sin(lat) + np.cos(p0) * np.cos(lat) * np.cos(lon - l0), -1, 1))
        return [int(k) for k in np.where(d <= float(radius_km or 600.0))[0]]
    return list(range(n))


def run_lod(ctx, res_list: list[float], nodes: list[int], jobs: int, force: bool = False, sets: list[str] | None = None, log=print) -> int:
    """批跑粗版（多进程）。已有的跳过（--force 重跑）。写 index.json。返回退出码（有群越界 → 1）。"""
    from concurrent.futures import ProcessPoolExecutor
    root = ctx.out_dir / LOD_DIR
    todo = [k for k in nodes if force or not all((root / f"{int(round(r))}" / f"{k}.npz").exists() for r in res_list)]
    log(f"[lod] {len(nodes)} 群，要跑 {len(todo)}，分辨率 {res_list}，{jobs} 进程 → {root}")
    t0 = time.time()
    done = []
    if todo:
        with ProcessPoolExecutor(max(1, jobs), initializer=_worker_init, initargs=(str(ctx.out_dir),)) as ex:
            for i, rows in enumerate(ex.map(_worker, [(k, res_list, sets) for k in todo], chunksize=1)):
                done += rows
                if (i + 1) % 200 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    log(f"  {i + 1}/{len(todo)}  {el / 60:.1f} min，预计还要 {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min")
    # index.json：把目录里已有的全汇总（meta 从 npz 读）
    index = {}
    for r in res_list:
        d = root / f"{int(round(r))}"
        for p in sorted(d.glob("*.npz"), key=lambda q: int(q.stem)):
            with np.load(p) as z:
                m = json.loads(str(z["meta"]))
            t = m.get("territory", {})
            index.setdefault(str(int(round(r))), {})[p.stem] = {
                "rows": m["raster"]["rows"], "cols": m["raster"]["cols"], "res_m": m["raster"]["res_m"],
                "seconds": m["seconds"], "constrained": bool(t.get("constrained")), "violation_km": t.get("violation_km")}
    (root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")
    viol = [row for row in done if (row["violation_km"] or -1) > 0]
    log(f"[lod] 完成 {len(done)} 份，用时 {(time.time() - t0) / 60:.1f} min；越界的群 {len({r['node'] for r in viol})}")
    return 1 if viol else 0
