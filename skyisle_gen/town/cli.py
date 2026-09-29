"""`skyisle town …`（PLAN-TOWN 7.11）。

skyisle town site 2051 --run out/seed42 --site 村037 --style 华北集村 [--scale 村] [--half 500] [--set style.键=值]   # 岛群里的一个聚落
skyisle town synth --terrain 河谷 --style 华北集村 [--scale 村] [--households 60] [--seed 1] [--lat 35]              # 合成地形
skyisle town synth --heightmap h.png --res 1 [--water w.png] [--height-scale 0.01] --style 华北集村 --scale 宅院        # 外部 16 位高程图
skyisle town gallery --terrain 河谷[,平原,…] [--styles all | 华北集村,江南水乡] [--operators each] [--households 40]        # 画廊
skyisle town style list | show 华北集村                                                                           # 内置风格
不给 --style 只出地面（site.npz / plan.json / plan.png）。--operator 指定形态算子（默认按风格的权重挑，只在这块地能用的里挑）。
"""
from __future__ import annotations

import time
from pathlib import Path

from . import SCALE_ZH, scale_id, town_config
from .output import out_dir, plan_json, write_plan_json, write_site, write_style
from .render import write_plan_png, write_plan_svg
from .site import site_from_group, site_heightmap, site_stats, site_synth


def add_parser(sub) -> None:
    p = sub.add_parser("town", help="聚落营建器（独立工具）：地形 + 规模 + 风格 → 建筑群（docs/PLAN-TOWN.md）")
    p.add_argument("what", choices=["site", "synth", "gallery", "style"], help="site = 岛群里的聚落；synth = 合成地形 / 外部高程图；gallery = 画廊；style = 看风格")
    p.add_argument("node", nargs="?", default=None, help="site：岛群节点号；style：list / show")
    p.add_argument("name", nargs="?", default=None, help="style show：风格名")
    p.add_argument("--run", default="out/seed42", help="site：行星产物目录")
    p.add_argument("--site", default=None, help="site：聚落名（村037 / 散户012 / 镇03 / 矿村01 / 城）")
    p.add_argument("--scale", default=None, help="宅院 / 小庄 / 村 / 集镇 / 城 / 专业聚落（site 默认按聚落的类，synth 默认村）")
    p.add_argument("--terrain", default=None, help="synth / gallery（逗号分隔几种）：平原 / 曲流平原 / 河谷 / 朝阳坡 / 山顶 / 湖岸 / 峡湾岸 / 黄土沟 / 两河交汇 / 岛缘崖台")
    p.add_argument("--heightmap", default=None, help="synth：16 位灰度高程图（代替 --terrain）")
    p.add_argument("--water", default=None, help="synth --heightmap：水面图（非零 = 水）")
    p.add_argument("--res", type=float, default=1.0, help="synth --heightmap：一个像素几米")
    p.add_argument("--height-scale", type=float, default=0.01, help="synth --heightmap：灰度 × 它 = 高程（m）")
    p.add_argument("--households", type=int, default=60, help="synth / gallery：户数")
    p.add_argument("--seed", type=int, default=1, help="synth：地形与营建的种子")
    p.add_argument("--lat", type=float, default=None, help="synth：纬度（正 = 北半球；默认 town.synth.lat_deg）")
    p.add_argument("--half", type=float, default=None, help="窗口半边长 m（默认按规模与户数）")
    p.add_argument("--style", default=None, help="风格：id、中文名或 .toml 路径（skyisle town style list）；不给只出地面")
    p.add_argument("--operator", default=None, help="形态算子（fishbone / organic / street_village / hufen / waterfront / dispersed / green / comb / contour / enclosure），默认按风格的权重挑")
    p.add_argument("--styles", default=None, help="gallery：all（默认）或逗号分隔的风格名 / id")
    p.add_argument("--operators", default=None, choices=["default", "each"], help="gallery：default 一格一个风格（算子按权重挑）；each 每个能用的算子各一格")
    p.add_argument("--set", action="append", default=[], dest="sets", help="town.a.b=value 覆盖工具参数；style.a.b=value 覆盖风格")
    p.add_argument("--out", default="out", help="synth：产物根目录")


def _style_cmd(a) -> int:
    from .style import list_styles, load_style
    if a.node in (None, "list"):
        for s in list_styles():
            print(f"{s['name']}（{s['id']}）  {s['region']}")
        return 0
    if a.node == "show":
        if not a.name:
            raise SystemExit("town style show 要风格名")
        st = load_style(a.name, a.sets)
        from .style import dump_toml
        print(dump_toml(st))
        return 0
    raise SystemExit("town style list | show <风格>")


def run_town(a) -> int:
    if a.what == "style":
        return _style_cmd(a)
    cfg = town_config(a.sets)
    if a.what == "gallery":
        from .gallery import run_gallery
        return run_gallery(a, cfg)
    st = None
    if a.style:
        from .style import load_style
        st = load_style(a.style, a.sets)
    hf = float(st.get("site", {}).get("window_factor", 1.0)) if st else 1.0   # 风格要的窗口放大
    t0 = time.perf_counter()
    if a.what == "site":
        if a.node is None or not a.site:
            raise SystemExit("town site 要节点号与 --site 聚落名，如：skyisle town site 2051 --run out/seed42 --site 村037 --style 华北集村")
        from ..cli import _ctx_from_run
        ctx = _ctx_from_run(a.run)
        sd, meta = site_from_group(ctx, int(a.node), a.site, scale_id(a.scale) if a.scale else None, cfg, a.half, hf)
        base = ctx.out_dir
    else:
        scale = scale_id(a.scale or "村")
        if a.heightmap:
            sd, meta = site_heightmap(Path(a.heightmap), a.res, scale, a.households, cfg, water=Path(a.water) if a.water else None,
                                      lat_deg=a.lat, scale_m=a.height_scale)
        else:
            if not a.terrain:
                raise SystemExit("town synth 要 --terrain（或 --heightmap）")
            sd, meta = site_synth(a.terrain, scale, a.households, cfg, seed=a.seed, lat_deg=a.lat, half_m=a.half, half_factor=hf)
        base = Path(a.out)
    t1 = time.perf_counter()
    stats = site_stats(sd)
    P = None
    extra = {"timing_s": {"site": round(t1 - t0, 3)}}
    if st is not None:
        from .plan import hard_failures, make_plan
        from .style import style_hash
        P = make_plan(sd, meta, st, a.operator)
        meta["style_name"] = st["meta"]["name"]
        extra.update(plan_json(P, st, style_hash(st)))
        extra["timing_s"]["plan"] = round(time.perf_counter() - t1, 3)
    tag = (st["meta"]["name"] + (f"-{P['op_name']}" if a.operator else "")) if st else None   # 指定了算子：目录名带上，免得互相覆盖
    out = out_dir(base, meta, tag)
    write_site(out, sd, P["occ"] if P else None)
    write_plan_json(out, sd, meta, stats, extra=extra)
    t2 = time.perf_counter()
    png = write_plan_png(out, sd, meta, cfg, stats, P, st["meta"]["name"] if st else None)
    if P:
        write_plan_svg(out, sd, meta, P, st["meta"]["name"])
        write_style(out, st)
    print(f"地面 {stats['size_m'][0]:.0f} × {stats['size_m'][1]:.0f} m（{sd['res_m']:g} m 一格，{SCALE_ZH[meta['scale']]} {meta['households']} 户），"
          f"细化 {t1 - t0:.2f} s；田 {stats['farmland_share']:.0%}、虚空 {stats['sky_share']:.0%}、河 {len(stats['rivers'])} 条")
    if P:
        m = P["metrics"]
        op = P["op_name"]
        print(f"营建「{st['meta']['name']}」（{op}）{extra['timing_s']['plan']:.2f} s：宅院 {int(m.get('compounds', 0))}、房 {int(m.get('buildings', 0))}、"
              f"路 {len(P['roads'])} 条、桥 {len(P['bridges'])}、塘 / 场 / 井 / 树 {sum(f['kind'] == 'pond' for f in P['features'])} / "
              f"{sum(f['kind'] == 'threshing' for f in P['features'])} / {sum(f['kind'] == 'well' for f in P['features'])} / "
              f"{sum(f['kind'] == 'tree' for f in P['features'])}")
        for c in P["checks"]:
            print(f"  {'✓' if c['ok'] else ('✗' if c['hard'] else '△')} {c['id']}：{c['msg']}")
        bad = hard_failures(P)
        if bad:
            print(f"  硬项没过 {len(bad)} 条（产物照写，便于查看）")
    print(f"产物：{out}（{png.name}，出图 {time.perf_counter() - t2:.1f} s）")
    return 0
