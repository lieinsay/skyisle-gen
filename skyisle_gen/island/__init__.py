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
ISLAND_VERSION = "3"  # 岛群层的实现版本：改了 core/src/island/（或这里的输入拼装、写产物）就 +1。
                     # 2 = 水位面（B+A，四点四十九）：流向按潜水面、泉线分弥散渗出 / 泉 / 崖瀑、terrain.npz 多 wt_m / wt_depth_m
                     # 3 = 河宽口径分开（四点五十）：rivers.json 的 w_bf_m / d_bf_m 改成真平岸（代 q_bf）、新增 w_mean_m / d_mean_m；
                     #     terrain.npz 的 river_width_m / river_depth_m 改名 w_mean_m / d_mean_m（年均口径）
                     # 写进 island.json 的 meta.stamp，控制台与 island check 拿它对盘上的产物判新旧


def island_stamp(ctx, c: dict, eng: dict) -> str:
    """产物的版本戳：岛群层实现版本 + [island] / [engine] 配置 + 上游 ①②③④ 的 stage key + seed。
    改了 core/src/island、改了 [island] 的默认值、或上游重算过，戳就变——盘上戳对不上的当没生成过。
    一次性的 --set 不进戳：否则按参数重生成一次，刷新页面就被判成旧的了。"""
    import hashlib
    import json
    from ..config import canonical
    keys = ""
    man = ctx.out_dir / "manifest.json"
    if man.exists():
        sk = json.loads(man.read_text(encoding="utf-8")).get("stage_keys", {})
        keys = "|".join(f"{k}={sk[k]}" for k in ("s01_planet", "s02_wind", "s03_islands", "s04_climate") if k in sk)
    payload = "\x1f".join([ISLAND_VERSION, canonical(c), canonical(eng), keys, str(ctx.seed)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def products_stale(ctx, node: int, year: int = 0) -> str | None:
    """盘上的岛群产物是不是当前的：None = 现成（直接读），否则返回该重生成的原因。
    判据只有三条：文件齐不齐、meta.stamp 与当前代码 / 配置 / 上游对不对得上、天气年份对不对。
    （2026-10-01 之前只查「文件在不在」——09-25 生成的产物会一直被当成命中。）"""
    import json
    island_config(ctx)          # 顺带把 ctx.island_stamp 算出来
    out = ctx.out_dir / "islands" / str(node)
    need = ("island.json", "climate.json", "terrain.npz", "rivers.json", "resources.json",
            "settlements.json", "preview.png", "preview_main.png", f"weather_y{year}.csv")
    miss = [f for f in need if not (out / f).exists()]
    if miss:
        return "缺产物：" + "、".join(miss)
    try:
        J = json.loads((out / "island.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"island.json 读不出来（{e}）"
    got = J.get("meta", {}).get("stamp")
    if got != ctx.island_stamp:
        return (f"版本戳不符（盘上 {got or '无'} ≠ 当前 {ctx.island_stamp}）："
                f"生成器版本、[island] 配置或上游 ①②③④ 变过")
    try:
        C = json.loads((out / "climate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"climate.json 读不出来（{e}）"
    w = C.get("weather", {})
    if w.get("year") != year or not isinstance(w.get("days"), list):
        return f"逐日天气不是第 {year} 年的"
    return None


def island_config(ctx, sets: list[str] | None = None) -> dict:
    """[island] 段：默认值 ← run 的 config.resolved.toml（若有）← --set。不进管线缓存 key。"""
    import tomllib
    with open(CONFIG_DIR / "default.toml", "rb") as fh:
        full = tomllib.load(fh)
    base = full.get("island", {})
    cfg = _deep_merge(base, ctx.cfg.get("island", {}))
    if not hasattr(ctx, "island_stamp"):     # 第一次调用时算（--set 之前），之后沿用：戳记的是配置本身
        ctx.island_stamp = island_stamp(ctx, cfg, _deep_merge(full.get("engine", {}), ctx.cfg.get("engine", {})))
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
    if "arable_frac_eff" in cli:          # A3 起：已垦额度 = 降水线之后的可耕率
        d["arable_frac"] = float(cli["arable_frac_eff"][node])
    if "precip_share" in cli:             # A2 起：四季降水按 ④ 的份额分（A5）
        d["precip_share"] = [float(v) for v in cli["precip_share"][:, node]]
    d["area_median_km2"] = float(np.median(isl["area_km2"]))
    d["planet"] = planet
    d["keel_clearance_m"] = float(ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    d["lapse_c_per_km"] = float(ctx.cfg["s04"]["climate"]["lapse_c_per_km"])
    d["precip_mm_ref"] = float(ctx.cfg["s04"]["climate"].get("precip_mm_ref", 4000.0))
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
    g["json"]["meta"]["island_version"] = ISLAND_VERSION
    g["json"]["meta"]["stamp"] = ctx.island_stamp
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
