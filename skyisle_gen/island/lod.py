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

`--weather`（默认开，DESIGN-NOTES 四点二十七）：地形 + 水系之后接着算四季 → 逐日曲线 → 第 --year 年（默认 0）的逐日天气（与 `skyisle island <节点>` 同式、
同随机流、同一个主岛岸缘），同一个 npz 里多存 weather_<列>（与 weather_y0.csv 同列同值，外加 temp_rim_c）与 weather_meta（气候参数头，JSON）。
天气与分辨率无关，每个分辨率的 npz 各带一份（同值）；meta 与块降采样的数组不受影响（逐位同不带天气时）。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

LOD_DIR = "islands_lod"
# 粗版里的逐日天气列（weather_<列>）：与 weather_y<年>.csv 同列（season_name 换成 weather_meta 的 season_names 下标，type 换成 types 下标），外加 temp_rim_c
WEATHER_COLS = {"day": np.int16, "season": np.uint8, "month": np.uint8, "day_of_month": np.uint8, "type": np.uint8,
                "precip_mm": np.float64, "temp_c": np.float64, "temp_rim_c": np.float64, "wind_from_deg": np.int16, "wind_ms": np.float64,
                "sailable": np.bool_, "storm_event": np.int16}
WEATHER_NOTE = ("weather_<列>：第 year 年逐日一行，与 skyisle island <节点> 写的 weather_y<年>.csv 同列同值——day 从 0 起、month / day_of_month 从 1 起、"
                "season 是 season_names 的下标、type 是 types 的下标；precip_mm / temp_c / wind_ms 已按 csv 舍到 0.1（float64，str() 即 csv 里的写法）、"
                "wind_from_deg 取整、sailable 布尔（csv 写 True / False）、storm_event 是当年的风暴场次号（0 = 不是风暴日）；"
                "外加 temp_rim_c（主岛岸缘处的气温，climate.json 逐日表里的那列）。temp_c 是台面 climate.ref_m 处的气温，"
                "某高度 h 的气温 = temp_c + lapse_c_per_km × (ref_m − h) / 1000；岸缘气温 ≤ snow_temp_c 的降水是雪（type 已分好）。"
                "climate 各键与 Zhouzhu tools/export_skyisle.py 写 weather.tsv 头的同名（fog_rise_m 同式：120 + 0.25 × max(0, fog_surface_max_m − ref_m)，未舍）；"
                "precip_mm 是 island.json hydro.precip_mm（栅格年降水，weather.tsv 头的那个），summary 是 island.json 的 weather 摘要。")


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


def build_weather_year(ctx, node: int, c: dict, g: dict, year: int = 0) -> None:
    """地形 + 水系之后接着算四季 → 逐日曲线 → 一年的逐日天气（generate 的第 3、4 步；天气只依赖本群的行星层输入与主岛岸缘，
    资源不影响它）。结果进 g["climate"] / g["daily"] / g["weather"]，与 island generate 同形同值。"""
    from .engine import backend
    quiet = lambda *a, **k: None
    if backend(ctx) == "cpp":
        from .engine import weather_year_cpp
        weather_year_cpp(ctx, node, c, g, year=year, log=quiet)
    else:
        from .climate import build_climate, daily_curves
        from .weather import build_weather
        build_climate(ctx, node, c, g, log=quiet)
        g["daily"] = daily_curves(g["climate"], g["inp"], ctx.cfg["s04"]["climate"])
        build_weather(ctx, node, c, g, year=year, log=quiet)


def weather_arrays(g: dict, c: dict, year: int = 0) -> dict:
    """g["weather"]["days"]（weather_y<年>.csv 与 climate.json 逐日表写的同一份行，位数已舍好）→ 粗版 npz 的 weather_<列> 与 weather_meta（气候参数头）。"""
    from .weather import TYPES
    days = g["weather"]["days"]
    clim = g["climate"]
    names = clim["season_names"]
    out = {}
    for k, dt in WEATHER_COLS.items():
        if k == "type":
            vals = [TYPES.index(d["type"]) for d in days]
        else:
            vals = [d[k] for d in days]
        out[f"weather_{k}"] = np.array(vals, dtype=dt)
    cal = clim["calendar"]
    inp = g["inp"]
    J = g["json"]
    wc = c["weather"]
    ref = float(clim["annual"]["temp_ref_height_m"])
    fog_max = float(wc["fog_surface_max_m"])
    head = {
        "year": int(year), "n_days": len(days), "types": list(TYPES), "season_names": list(names),
        "season_type": clim["season_type"], "season_type_zh": clim["season_type_zh"],
        "calendar": {k: cal[k] for k in ("seasons", "months_per_season", "days_per_month", "days_per_season", "year_days", "day_offset_solstice_n")},
        "sun": {"lat": float(inp["lat"]), "lon": float(inp["lon"]), "tilt_deg": float(inp["planet"]["axial_tilt_deg"])},
        "climate": {"ref_m": ref, "lapse_c_per_km": float(inp["lapse_c_per_km"]), "rim_m": float(J["islands"][0]["rim_m"]),
                    "fog_surface_max_m": fog_max, "fog_rise_m": 120.0 + 0.25 * max(0.0, fog_max - ref), "snow_temp_c": float(wc["snow_temp_c"]),
                    "heavy_rain_mm": float(wc["heavy_rain_mm"]), "sail_wind_max_ms": float(wc["sail_wind_max_ms"])},
        "precip_mm": J["hydro"]["precip_mm"], "precip_climate_mm": clim["annual"]["precip_mm"],
        "summary": dict(J["weather"]),
        "note": WEATHER_NOTE,
    }
    out["weather_meta"] = np.array(json.dumps(_jsonable(head), ensure_ascii=False))
    return out


def build_lod(ctx, node: int, c: dict, res_list: list[float], native_res_m: float | None = None, weather: bool = True, year: int = 0) -> dict:
    """原生分辨率跑布局 + 地形 + 水系，按 res_list 各降采样一份。返回 {res: (arrays, meta)}。
    weather：再算第 year 年的逐日天气，arrays 里多 weather_<列> 与 weather_meta（各分辨率同一份）。
    native_res_m 只给测试用（小世界用粗的「原生」跑得快）；正式批跑一律用 island.res_m。"""
    from . import _node_inputs, build_terrain
    from .hydro import build_hydro
    t0 = time.perf_counter()
    inp = _node_inputs(ctx, node)
    quiet = lambda *a, **k: None
    g = build_terrain(ctx, node, c, inp, res_m=native_res_m, log=quiet)
    build_hydro(ctx, node, c, g, log=quiet)
    wx = {}
    if weather:
        build_weather_year(ctx, node, c, g, year=year)
        wx = weather_arrays(g, c, year)
    J = g["json"]
    native = float(J["raster"]["res_m"])
    out = {}
    from .engine import backend
    for res in res_list:
        f = max(1, int(round(float(res) / native)))
        if backend(ctx) == "cpp":              # 行星计划 P6b：块降采样也在 C++（同式，逐位相同）
            from .engine import block_reduce_cpp
            arr = block_reduce_cpp(g, f)
        else:
            arr = _block_reduce(g, f)
        arr.update(wx)
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
    node, res_list, sets, weather, year = args
    from . import island_config
    c = island_config(_CTX, sets)
    root = _CTX.out_dir / LOD_DIR
    rows = []
    for res, (arr, meta) in build_lod(_CTX, node, c, res_list, weather=weather, year=year).items():
        write_lod(root, res, arr, meta)
        t = meta["territory"]
        rows.append({"node": node, "res": res, "rows": meta["raster"]["rows"], "cols": meta["raster"]["cols"],
                     "seconds": meta["seconds"], "constrained": bool(t.get("constrained")), "violation_km": t.get("violation_km")})
    return rows


def _weather_year_of(p: Path) -> int | None:
    """这份粗版带的天气是第几年的（不带天气 / 读不了 → None）。"""
    try:
        with np.load(p) as z:
            return int(json.loads(str(z["weather_meta"]))["year"]) if "weather_meta" in z.files else None
    except (OSError, ValueError, KeyError):
        return None


def _done(p: Path, weather: bool, year: int = 0) -> bool:
    """这份粗版已有（要天气时还得带着同一年的天气：P0 时做的不带，要重跑）。"""
    return p.exists() and (not weather or _weather_year_of(p) == year)


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


def run_lod(ctx, res_list: list[float], nodes: list[int], jobs: int, force: bool = False, sets: list[str] | None = None, log=print,
            weather: bool = True, year: int = 0) -> int:
    """批跑粗版（多进程）。已有的跳过（--force 重跑；weather 时不带天气或天气不是第 year 年的也重跑）。写 index.json。返回退出码（有群越界 → 1）。"""
    from concurrent.futures import ProcessPoolExecutor
    root = ctx.out_dir / LOD_DIR
    todo = [k for k in nodes if force or not all(_done(root / f"{int(round(r))}" / f"{k}.npz", weather, year) for r in res_list)]
    log(f"[lod] {len(nodes)} 群，要跑 {len(todo)}，分辨率 {res_list}，{f'带第 {year} 年天气' if weather else '不带天气'}，{jobs} 进程 → {root}")
    t0 = time.time()
    done = []
    if todo:
        with ProcessPoolExecutor(max(1, jobs), initializer=_worker_init, initargs=(str(ctx.out_dir),)) as ex:
            for i, rows in enumerate(ex.map(_worker, [(k, res_list, sets, weather, year) for k in todo], chunksize=1)):
                done += rows
                if (i + 1) % 200 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    log(f"  {i + 1}/{len(todo)}  {el / 60:.1f} min，预计还要 {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min")
    # index.json：把这几个分辨率目录里已有的全汇总（meta 从 npz 读）；别的分辨率沿用旧 index 里的（原先整个重写，只剩最后一次跑的分辨率）
    index = {}
    if (root / "index.json").exists():
        try:
            index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        except ValueError:
            index = {}
    for r in res_list:
        index[str(int(round(r)))] = {}
        d = root / f"{int(round(r))}"
        for p in sorted(d.glob("*.npz"), key=lambda q: int(q.stem)):
            with np.load(p) as z:
                m = json.loads(str(z["meta"]))
                wx_year = int(json.loads(str(z["weather_meta"]))["year"]) if "weather_meta" in z.files else None
            t = m.get("territory", {})
            index.setdefault(str(int(round(r))), {})[p.stem] = {
                "rows": m["raster"]["rows"], "cols": m["raster"]["cols"], "res_m": m["raster"]["res_m"],
                "seconds": m["seconds"], "constrained": bool(t.get("constrained")), "violation_km": t.get("violation_km"),
                "weather_year": wx_year}
    (root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")
    viol = [row for row in done if (row["violation_km"] or -1) > 0]
    log(f"[lod] 完成 {len(done)} 份，用时 {(time.time() - t0) / 60:.1f} min；越界的群 {len({r['node'] for r in viol})}")
    return 1 if viol else 0
