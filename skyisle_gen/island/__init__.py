"""岛群生成器（第三层，docs/PLAN-ISLAND.md）：给一个节点号，按行星产物做约束，生成群内布局、岛内地形、水系、四季气候与逐日天气。

按需生成、不进十步管线、不回灌：stages/ 不得 import 本包（tests 有静态断言）。
算法全在 C++ 核心里（core/src/island/，行星计划 P6a / P6b；Python 参考版删于 2026-09-30，tag python-reference-final）：
generate 一次调 _core.generate 算完（island/engine.py 拼回 g 与 island.json），这里读输入、写产物。
随机数（C++ 的 rng.hpp，与 numpy 逐位一致）按 entity_rng(seed, ISLAND_STREAM, f"island:{node}:{部件}") 派生；天气另加年份键。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import CONFIG_DIR, _deep_merge, apply_sets

ISLAND_STREAM = 21   # 与十步管线的流号 1–10 错开


def island_config(ctx, sets: list[str] | None = None) -> dict:
    """[island] 段：默认值 ← run 的 config.resolved.toml（若有）← --set。不进管线缓存 key。"""
    import tomllib
    with open(CONFIG_DIR / "default.toml", "rb") as fh:
        full = tomllib.load(fh)
    base = full.get("island", {})
    cfg = _deep_merge(base, ctx.cfg.get("island", {}))
    if sets:
        tmp = {"island": cfg}
        apply_sets(tmp, [s for s in sets if s.startswith("island.")])
        cfg = tmp["island"]
    ctx.cfg["island"] = cfg
    # [engine]（C++ 的线程数）：同样是 default.toml ← run 的快照 ← --set engine.x=v，不进缓存 key
    eng = _deep_merge(full.get("engine", {}), ctx.cfg.get("engine", {}))
    if sets:
        tmp = {"engine": eng}
        apply_sets(tmp, [s for s in sets if s.startswith("engine.")])
        eng = tmp["engine"]
    ctx.cfg["engine"] = eng
    return cfg


def _node_inputs(ctx, node: int) -> dict:
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    planet = ctx.load_json(1, "planet")
    n = isl["lat"].size
    if not (0 <= node < n):
        raise ValueError(f"节点号 {node} 超出范围 [0, {n})")
    d = {k: float(isl[k][node]) for k in ("lat", "lon", "area_km2", "main_area_km2", "height_m", "wall_m", "arable_frac", "age")}
    d["plate"] = int(isl["plate"][node])
    d["cls"] = int(isl["cls"][node])
    d["layered"] = bool(isl["layered"][node])
    for k in ("precip", "temp", "storm", "stability", "window", "river_size", "temp_sea", "season_range", "season_range_sea",
              "temp_winter", "temp_summer"):
        d[k] = float(cli[k][node])
    d["has_river"] = bool(cli["has_river"][node])
    d["area_median_km2"] = float(np.median(isl["area_km2"]))
    d["planet"] = planet
    d["keel_clearance_m"] = float(ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    d["lapse_c_per_km"] = float(ctx.cfg["s04"]["climate"]["lapse_c_per_km"])
    return d


def build_terrain(ctx, node: int, c: dict, inp: dict, res_m: float | None = None, log=print) -> dict:
    """第 1 步：布局 + 岛形 + 高程（C++）。返回群栅格字典 g（height / island_id / cliff / json / islands 列表）。只跑地形的工具（floats、粗版）用。"""
    from .engine import build_terrain_cpp
    return build_terrain_cpp(ctx, node, c, inp, res_m=res_m, log=log)


def generate(ctx, node: int, year: int = 0, res_m: float | None = None, export: str | None = None,
             sets: list[str] | None = None, steps: int = 9, log=print, return_state: bool = False, out_root: Path | None = None,
             write: bool = True):
    """生成一个岛群的全部产物，写到 out/<run>/islands/<node>/（out_root 给了就写到 out_root/<node>/，对照工具用）。返回目录。
    g["timing"] 记地形、水系两步与整群算法部分（generate，不含写产物）的用时（不进产物）。write=False：只算不写，返回 g（用时对照用）。"""
    from .output import write_preview, write_preview_main, write_terrain
    c = island_config(ctx, sets)
    inp = _node_inputs(ctx, node)
    t0 = time.perf_counter()
    log(f"[island {node}] 陆地 {inp['area_km2']:.0f} km²（主岛 {inp['main_area_km2']:.0f}）台面 {inp['height_m']:.0f} m 可耕 {inp['arable_frac']:.3f} "
        f"河 {'有' if inp['has_river'] else '无'} 岛龄 {inp['age']:.2f} 降水 {inp['precip']:.2f} 温差 {inp['season_range']:.1f} °C")
    from .engine import generate_cpp            # 行星计划 P6b：整群在 C++ 里算，g 拼回同形；写产物在下面
    g = generate_cpp(ctx, node, c, inp, year=year, res_m=res_m, steps=steps, log=log)
    g["timing"]["generate"] = time.perf_counter() - t0      # 不含写产物（png / 预览图）
    if not write:
        return g
    out =(ctx.out_dir / "islands" if out_root is None else Path(out_root)) / str(node)
    g["json"]["meta"]["seconds"] = round(time.perf_counter() - t0, 2)
    write_terrain(out, g)
    if "climate" in g:
        from .climate import write_climate
        write_climate(out, g, year)
    if "settle" in g:
        from .output import write_settlements
        write_settlements(out, g)
    if "resources" in g:
        from .resources import write_preview_resources, write_resources
        write_resources(out, g)
        write_preview_resources(out, g)
    p = write_preview(out, g)
    write_preview_main(out, g)
    log(f"[island {node}] 完成 {time.perf_counter() - t0:.1f} s → {out}")
    if export:
        import shutil
        dst = Path(export) / str(node)
        dst.mkdir(parents=True, exist_ok=True)
        for f in out.iterdir():
            shutil.copy2(f, dst / f.name)
        log(f"  已导出到 {dst}")
    return (out, g) if return_state else out
