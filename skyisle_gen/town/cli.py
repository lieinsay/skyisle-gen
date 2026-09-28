"""`skyisle town …`（PLAN-TOWN 7.11）。

skyisle town site 2051 --run out/seed42 --site 村037 [--scale 村] [--half 500] [--set town.键=值]   # 岛群里的一个聚落所在
skyisle town synth --terrain 河谷 [--scale 村] [--households 60] [--seed 1] [--lat 35]              # 合成地形
skyisle town synth --heightmap h.png --res 1 [--water w.png] [--height-scale 0.01]                  # 外部 16 位高程图
第一步只出地面（site.npz / plan.json / plan.png）；风格（--style）在第二步接上。
"""
from __future__ import annotations

import time
from pathlib import Path

from . import SCALE_ZH, scale_id, town_config
from .output import out_dir, write_plan_json, write_site
from .render import write_plan_png
from .site import site_from_group, site_heightmap, site_stats, site_synth


def add_parser(sub) -> None:
    p = sub.add_parser("town", help="聚落营建器（独立工具）：地形 + 规模 + 风格 → 建筑群（docs/PLAN-TOWN.md）")
    p.add_argument("what", choices=["site", "synth"], help="site = 岛群里的聚落；synth = 合成地形 / 外部高程图")
    p.add_argument("node", nargs="?", type=int, default=None, help="site：岛群节点号")
    p.add_argument("--run", default="out/seed42", help="site：行星产物目录")
    p.add_argument("--site", default=None, help="site：聚落名（村037 / 散户012 / 镇03 / 矿村01 / 城）")
    p.add_argument("--scale", default=None, help="宅院 / 小庄 / 村 / 集镇 / 城 / 专业聚落（site 默认按聚落的类，synth 默认村）")
    p.add_argument("--terrain", default=None, help="synth：平原 / 曲流平原 / 河谷 / 朝阳坡 / 山顶 / 湖岸 / 峡湾岸 / 黄土沟 / 两河交汇 / 岛缘崖台")
    p.add_argument("--heightmap", default=None, help="synth：16 位灰度高程图（代替 --terrain）")
    p.add_argument("--water", default=None, help="synth --heightmap：水面图（非零 = 水）")
    p.add_argument("--res", type=float, default=1.0, help="synth --heightmap：一个像素几米")
    p.add_argument("--height-scale", type=float, default=0.01, help="synth --heightmap：灰度 × 它 = 高程（m）")
    p.add_argument("--households", type=int, default=60, help="synth：户数")
    p.add_argument("--seed", type=int, default=1, help="synth：地形种子")
    p.add_argument("--lat", type=float, default=None, help="synth：纬度（正 = 北半球；默认 town.synth.lat_deg）")
    p.add_argument("--half", type=float, default=None, help="窗口半边长 m（默认按规模与户数）")
    p.add_argument("--style", default=None, help="风格（第二步起）")
    p.add_argument("--set", action="append", default=[], dest="sets", help="town.a.b=value 覆盖工具参数")
    p.add_argument("--out", default="out", help="synth：产物根目录")


def run_town(a) -> int:
    cfg = town_config(a.sets)
    t0 = time.perf_counter()
    if a.what == "site":
        if a.node is None or not a.site:
            raise SystemExit("town site 要节点号与 --site 聚落名，如：skyisle town site 2051 --run out/seed42 --site 村037")
        from ..cli import _ctx_from_run
        ctx = _ctx_from_run(a.run)
        sd, meta = site_from_group(ctx, a.node, a.site, scale_id(a.scale) if a.scale else None, cfg, a.half)
        base = ctx.out_dir
    else:
        scale = scale_id(a.scale or "村")
        if a.heightmap:
            sd, meta = site_heightmap(Path(a.heightmap), a.res, scale, a.households, cfg, water=Path(a.water) if a.water else None,
                                      lat_deg=a.lat, scale_m=a.height_scale)
        else:
            if not a.terrain:
                raise SystemExit("town synth 要 --terrain（或 --heightmap）")
            sd, meta = site_synth(a.terrain, scale, a.households, cfg, seed=a.seed, lat_deg=a.lat, half_m=a.half)
        base = Path(a.out)
    t1 = time.perf_counter()
    stats = site_stats(sd)
    out = out_dir(base, meta, a.style)
    write_site(out, sd)
    write_plan_json(out, sd, meta, stats, extra={"timing_s": {"site": round(t1 - t0, 3)}})
    png = write_plan_png(out, sd, meta, cfg, stats)
    print(f"地面 {stats['size_m'][0]:.0f} × {stats['size_m'][1]:.0f} m（{sd['res_m']:g} m 一格，{SCALE_ZH[meta['scale']]} {meta['households']} 户），"
          f"细化 {t1 - t0:.2f} s；田 {stats['farmland_share']:.0%}、虚空 {stats['sky_share']:.0%}、河 {len(stats['rivers'])} 条")
    print(f"产物：{out}（{png.name}）")
    return 0
