"""第六节：岛群生成器的一致性校验（`zhouzhu island check <节点>`）。

IS-area / IS-summit / IS-arable / IS-river / IS-season / IS-link / IS-det / IS-iso 为硬项，IS-daily 为软项。
退出码：2 = 硬项失败；1 = 软项失败；0 = 全过。批跑（batch.py）复用 evaluate()。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent


def hash_products(out: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
            if p.suffix in (".npz", ".png", ".csv", ".json") and p.name != "island.json"}


def static_isolation() -> tuple[bool, list[str]]:
    """IS-iso：stages/、check、ninegrid、polity、culture 不得 import zhouzhu_gen.island。"""
    bad = []
    files = sorted((PKG / "stages").glob("*.py")) + [PKG / n for n in ("check.py", "ninegrid.py", "polity.py", "culture.py", "pipeline.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+\.*\s*(zhouzhu_gen\.)?island\b", text, re.M):
            bad.append(f.name)
    return not bad, bad


def evaluate(g: dict, out: Path, ctx=None, node: int | None = None, c: dict | None = None,
             det_hashes: tuple[dict, dict] | None = None, daily_years: int = 60) -> list[dict]:
    J = g["json"]
    C = g.get("climate")
    cons = J["constraints"]
    items = []

    def add(id_, name, value, thr, ok, hard=True, note=None):
        items.append({"id": id_, "name": name, "value": value, "threshold": thr, "pass": bool(ok), "hard": hard, "note": note})

    a = cons["area_km2"]
    e_area = abs(a["actual"] - a["target"]) / max(1e-9, a["target"])
    m = cons["main_area_km2"]
    e_main = abs(m["actual"] - m["target"]) / max(1e-9, m["target"])
    add("IS-area", "各岛面积之和 = area_km2；主岛 = main_area_km2（相对误差）", {"sum": round(e_area, 5), "main": round(e_main, 5)}, "< 0.02",
        e_area < 0.02 and e_main < 0.02)
    h = cons["height_m"]
    e_h = abs(h["actual"] - h["target"]) / max(1e-9, h["target"])
    add("IS-summit", "主岛最高格 = height_m", round(e_h, 5), "< 0.01", e_h < 0.01)
    ar = cons["arable_frac"]
    e_ar = abs(ar.get("actual", -1) - ar["target"])
    add("IS-arable", "可耕地 / 陆地 = arable_frac", round(e_ar, 5), "< 0.005", e_ar < 0.005)
    rv = cons["has_river"]
    small_ok = all(not i.get("has_perennial_river", False) for i in J["islands"][1:])
    add("IS-river", "主岛有常年河 ⇔ has_river；小岛只有溪涧", {"main": rv.get("actual"), "target": rv["target"], "small_islands_ok": small_ok},
        "相等", rv.get("actual") == rv["target"] and small_ok)
    if C is not None:
        an, mc = C["annual"], C["means_check"]
        errs = {"precip": abs(mc["precip_rel"] - an["precip_rel"]) / max(1e-9, an["precip_rel"]),
                "storm": abs(mc["storm"] - an["storm"]) / max(0.05, an["storm"]),
                "window": abs(mc["window"] - an["window"]) / max(1e-9, an["window"]),
                "temp_c": abs(mc["temp_c"] - an["temp_c"])}
        add("IS-season", "四季降水 / 风暴 / 窗口 / 温度的平均 = 年均值", {k: round(v, 5) for k, v in errs.items()}, "< 0.01（温度 < 0.05 °C）",
            errs["precip"] < 0.01 and errs["storm"] < 0.01 and errs["window"] < 0.01 and errs["temp_c"] < 0.05)
        if ctx is not None and "daily" in g:
            from .weather import multi_year_stats
            st = multi_year_stats(ctx, node, c, g, years=daily_years)
            add("IS-daily", f"{daily_years} 年逐日降水的平均 = 气候值（相对误差 < 5% 或 |z| < 3，z = 偏差 / 年际标准误）；雨日比例落在设定 ±0.05",
                {"annual_rel_err": st["annual_rel_err"], "annual_z": st["annual_z"], "annual_se_rel": st["annual_se_rel"],
                 "season_rel_err": st["precip_rel_err"], "wet_frac_err": st["wet_frac_err"]},
                "< 0.05 或 z < 3 / ≤ 0.05", (st["annual_rel_err"] < 0.05 or st["annual_z"] < 3.0) and max(st["wet_frac_err"]) <= 0.05, hard=False)
    # IS-link：索桥 + 短渡连通
    from ..graph import weak_components
    n = len(J["islands"])
    src = np.array([e["a"] for e in J["links"]], dtype=np.int64)
    dst = np.array([e["b"] for e in J["links"]], dtype=np.int64)
    ncomp = int(weak_components(n, src, dst).max()) + 1 if n > 1 else 1
    add("IS-link", "群内任意两岛经索桥 + 短渡连通", {"components": ncomp, "islands": n}, "1 个分量", ncomp == 1)
    if det_hashes is not None:
        h1, h2 = det_hashes
        diff = sorted(k for k in set(h1) | set(h2) if h1.get(k) != h2.get(k))
        add("IS-det", "同输入重跑两次，产物哈希一致", {"differ": diff}, "无差异", not diff)
    ok, bad = static_isolation()
    add("IS-iso", "stages/ 不 import zhouzhu_gen.island（第三层不回灌）", {"offenders": bad}, "无", ok)
    return items


def exit_code(items: list[dict]) -> int:
    if any(not i["pass"] and i["hard"] for i in items):
        return 2
    if any(not i["pass"] for i in items):
        return 1
    return 0


def print_report(items: list[dict], node: int) -> None:
    print(f"== 岛群 #{node} 一致性校验 ==")
    for i in items:
        flag = "✓" if i["pass"] else ("✗" if i["hard"] else "△")
        kind = "硬" if i["hard"] else "软"
        print(f"  {flag} {i['id']:<10} [{kind}] {i['name']}")
        print(f"      值 {json.dumps(i['value'], ensure_ascii=False)}  阈 {i['threshold']}")
    code = exit_code(items)
    print("结果：" + {0: "全过", 1: "软项失败", 2: "硬项失败"}[code])


def run_island_check(ctx, node: int | None, year: int = 0, sets: list[str] | None = None) -> int:
    from . import generate
    if node is None:
        print("用法：zhouzhu island check <节点>")
        return 2
    out, g = generate(ctx, node, year=year, sets=sets, return_state=True)
    h1 = hash_products(out)
    generate(ctx, node, year=year, sets=sets, log=lambda *a: None)
    h2 = hash_products(out)
    items = evaluate(g, out, ctx=ctx, node=node, c=ctx.cfg["island"], det_hashes=(h1, h2))
    print_report(items, node)
    (out / "check.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return exit_code(items)
