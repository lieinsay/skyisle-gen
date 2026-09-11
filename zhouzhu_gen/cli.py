"""命令行入口。

zhouzhu run   --seed 42 [--config F]... [--set a.b.c=v]... [--upto 10] [--out out] [--explain]
zhouzhu stage K --seed 42        # 强制从第 K 阶段重算（之前阶段用缓存）
zhouzhu viz   <layer> --run out/seed42 [...]
zhouzhu probe <node|path|edge|trait> ...
zhouzhu check --run out/seed42 [--calibrate]
zhouzhu ninegrid --run out/seed42 [--region K]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Context, run as pipeline_run


def _add_common(p):
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", action="append", default=[], help="额外配置文件（可多次）")
    p.add_argument("--set", action="append", default=[], dest="sets", help="a.b.c=value 覆盖")
    p.add_argument("--out", default="out")


def _ctx_from_run(run_dir: str) -> Context:
    """从已有产物目录重建上下文（viz/probe/check/ninegrid 只读产物）。"""
    import tomllib
    p = Path(run_dir)
    with open(p / "config.resolved.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    import json
    seed = json.loads((p / "manifest.json").read_text(encoding="utf-8"))["seed"]
    return Context(cfg, seed, p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zhouzhu", description="行星地形与文明生成器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="执行十步管线（①–⑧ 地理与文化、⑨ 政治层、⑩ 输出）")
    _add_common(p_run)
    p_run.add_argument("--upto", type=int, default=10)
    p_run.add_argument("--explain", action="store_true")

    p_stage = sub.add_parser("stage", help="从第 K 阶段强制重算")
    p_stage.add_argument("k", type=int)
    _add_common(p_stage)
    p_stage.add_argument("--upto", type=int, default=10)

    p_viz = sub.add_parser("viz", help="可视化某一层")
    p_viz.add_argument("layer", help="wind|islands|scale|climate|barriers|perm|routes|centers|"
                                     "iso|polity|trait|slot|isogloss|distance|all|web（单文件操作台）")
    p_viz.add_argument("arg", nargs="?", default=None, help="trait id / slot id / node id / mode")
    p_viz.add_argument("--run", default="out/seed42")
    p_viz.add_argument("--mode", default=None)
    p_viz.add_argument("--show", action="store_true")

    p_probe = sub.add_parser("probe", help="探针（只读产物）")
    p_probe.add_argument("what", choices=["node", "path", "edge", "trait"])
    p_probe.add_argument("args", nargs="*")
    p_probe.add_argument("--run", default="out/seed42")
    p_probe.add_argument("--mode", default="trade")
    p_probe.add_argument("--node", type=int, default=None)

    p_check = sub.add_parser("check", help="八条验收现象 + 气候 + 铁律自检 + 骨架/历法校准")
    p_check.add_argument("--run", default="out/seed42")
    p_check.add_argument("--calibrate", action="store_true")

    p_nine = sub.add_parser("ninegrid", help="九格表草稿")
    p_nine.add_argument("--run", default="out/seed42")
    p_nine.add_argument("--region", type=int, default=None)

    p_pol = sub.add_parser("polity", help="政治层摘要：诸邦 / 变法之国 / 兼并史（只读 ⑨ 产物）")
    p_pol.add_argument("--run", default="out/seed42")
    p_pol.add_argument("--top", type=int, default=15)

    p_serve = sub.add_parser("serve", help="本地 3D 操作台（可改参数重跑）")
    p_serve.add_argument("--out", default="out")
    p_serve.add_argument("--port", type=int, default=8642)
    p_serve.add_argument("--no-open", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from .web.server import serve
        serve(Path(a.out), a.port, open_browser=not a.no_open)
        return 0

    if a.cmd in ("run", "stage"):
        cfg = load_config([Path(x) for x in a.config], a.sets)
        force_from = a.k if a.cmd == "stage" else None
        out = pipeline_run(cfg, a.seed, Path(a.out), upto=a.upto,
                           force_from=force_from, explain=getattr(a, "explain", False))
        print(f"产物目录：{out}")
        return 0

    ctx = _ctx_from_run(a.run)
    if a.cmd == "viz":
        if a.layer == "web":
            from .web.server import export_static
            out = export_static(ctx, ctx.stage_dir(10) / "viewer.html")
            print(f"已导出单文件操作台：{out}（双击打开，无需服务器；无重跑功能）")
            return 0
        from .viz import render
        render(ctx, a.layer, a.arg, mode=a.mode, show=a.show)
        return 0
    if a.cmd == "probe":
        from .probe import probe
        return probe(ctx, a.what, a.args, mode=a.mode, node=a.node)
    if a.cmd == "check":
        from .check import run_check
        code = run_check(ctx, calibrate=a.calibrate)
        return code
    if a.cmd == "ninegrid":
        from .ninegrid import render_ninegrids
        render_ninegrids(ctx, region=a.region)
        return 0
    if a.cmd == "polity":
        from .polity import print_summary
        print_summary(ctx, top=a.top)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
