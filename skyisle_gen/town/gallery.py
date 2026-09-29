"""画廊（PLAN-TOWN 7.10 / 7.11）：同一块地 × 若干风格（每格一个方案），或几种地形 × 若干风格（一行一种地形），拼一张 gallery.png，
另写 gallery.json（每格的风格、算子、户、房、主要指标、没过的硬项与软项）。

skyisle town gallery --terrain 河谷 --styles all [--operators each] [--households 40] [--seed 1]
每一格与 `town synth --terrain 同一种 --style 同一种 [--operator 同一个]` 的方案逐字节相同（窗口按风格的放大系数；大小窗口中间逐格相同，
所以画的是同一块地），格子里看中哪一个，拿同样的参数跑 synth 就能出整套产物。
--operators each：每个风格把它的算子里这块地能用的都摆一格（滨水要河、等高线要坡）；默认一格一个风格，算子按风格的权重挑。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import SCALE_ZH, TERRAIN_ZH, scale_id, terrain_id  # noqa: E402
from .render import draw_plan, ground_rgb, plan_extent  # noqa: E402

TILE_IN = 4.2          # 每格的边长（英寸，dpi 100）
METRIC_KEYS = ("lambda", "shape_index", "coverage", "clark_evans", "orient_sun_share", "density_hh_per_ha", "face_street_share",
               "site_clark_evans", "site_nn_median_m", "water_front_share", "lane_spacing_cv", "hh_per_enclosure", "terraces",
               "buildings_per_compound_median", "well_dist_median_m")


def _styles(spec: str) -> list[str]:
    from .style import list_styles
    if spec in ("all", "全部"):
        return [s["id"] for s in list_styles()]
    return [s.strip() for s in spec.replace("，", ",").split(",") if s.strip()]


def gallery_tiles(terrains: list[str], styles: list[str], households: int, cfg: dict, seed: int = 1, lat_deg: float | None = None,
                  each: bool = False, sets: list[str] | None = None, scale: str = "village", log=print) -> tuple[list[dict], dict]:
    """算每一格：[{terrain, style, P, sd, meta, …}]。地面按 (地形, 窗口放大系数) 缓存。"""
    from .plan import make_plan
    from .site import site_synth
    from .style import load_style, operator_name
    grounds: dict[tuple[str, float], tuple[dict, dict]] = {}
    tiles = []
    for t in terrains:
        tid = terrain_id(t)
        for sname in styles:
            st = load_style(sname, sets)
            hf = float(st.get("site", {}).get("window_factor", 1.0))
            key = (tid, hf)
            if key not in grounds:
                grounds[key] = site_synth(tid, scale, households, cfg, seed=seed, lat_deg=lat_deg, half_factor=hf)
            sd, meta = grounds[key]
            t0 = time.perf_counter()
            P = make_plan(sd, meta, st, None)
            runs = [(None, P, time.perf_counter() - t0)]
            if each:
                for op in P.get("ops_fit", []):
                    if op == P["op"]:
                        continue
                    t1 = time.perf_counter()
                    runs.append((op, make_plan(sd, meta, st, op), time.perf_counter() - t1))
                runs.sort(key=lambda r: list(st["village"]["operators"]).index(r[1]["op"]) if r[1]["op"] in st["village"]["operators"] else 99)
            for forced, Q, dt in runs:
                bad = [c["id"] for c in Q["checks"] if c["hard"] and not c["ok"]]
                soft = [c["id"] for c in Q["checks"] if not c["hard"] and not c["ok"]]
                log(f"  {TERRAIN_ZH[tid]} · {st['meta']['name']} · {operator_name(st, Q['op'])}：{dt:.1f} s，"
                    f"{'硬项全过' if not bad else '没过 ' + '、'.join(bad)}" + (f"（软项 △ {'、'.join(soft)}）" if soft else ""))
                tiles.append({"terrain": tid, "style": st, "P": Q, "sd": sd, "meta": meta, "forced": forced, "plan_s": dt, "hard_fail": bad, "soft_miss": soft})
    return tiles, grounds


def _square_extent(P: dict, sd: dict, margin: float = 30.0) -> tuple[float, float, float, float]:
    ext = plan_extent(P, margin)
    if ext is None:
        x0, y1 = sd["x0"], sd["y0"]
        return x0, x0 + sd["W"] * sd["res_m"], y1 - sd["H"] * sd["res_m"], y1
    x0, x1, y0, y1 = ext
    cx, cy, h = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * max(x1 - x0, y1 - y0, 120.0)
    # 方框挪进窗口里（窗口比方框大时），不露窗外
    wx0, wx1 = sd["x0"], sd["x0"] + sd["W"] * sd["res_m"]
    wy0, wy1 = sd["y0"] - sd["H"] * sd["res_m"], sd["y0"]
    if wx1 - wx0 >= 2 * h:
        cx = min(max(cx, wx0 + h), wx1 - h)
    if wy1 - wy0 >= 2 * h:
        cy = min(max(cy, wy0 + h), wy1 - h)
    return cx - h, cx + h, cy - h, cy + h


def _draw_tile(ax, tile: dict, show_terrain: bool) -> None:
    sd, P, st = tile["sd"], tile["P"], tile["style"]
    x0, x1, y0, y1 = _square_extent(P, sd)
    res = sd["res_m"]
    j0, j1 = max(0, int(math.floor((x0 - sd["x0"]) / res))), min(sd["W"], int(math.ceil((x1 - sd["x0"]) / res)))
    i0, i1 = max(0, int(math.floor((sd["y0"] - y1) / res))), min(sd["H"], int(math.ceil((sd["y0"] - y0) / res)))
    stride = max(1, int(math.ceil(max(i1 - i0, j1 - j0) / 520.0)))
    img = ground_rgb(sd, stride, (i0, i1, j0, j1), fade=0.3)
    ext = [sd["x0"] + j0 * res, sd["x0"] + j1 * res, sd["y0"] - i1 * res, sd["y0"] - i0 * res]
    ax.set_facecolor("#12162a")
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.apply_aspect()
    bb = ax.get_window_extent()
    pt_per_m = bb.width / (x1 - x0) * 72.0 / ax.figure.dpi
    draw_plan(ax, P, pt_per_m, labels=False)
    span = x1 - x0
    bar = 10 ** math.floor(math.log10(span / 4))
    bar = bar * (5 if span / bar > 20 else 2 if span / bar > 8 else 1)
    bx, by = x0 + 0.05 * span, y0 + 0.05 * span
    ax.plot([bx, bx + bar], [by, by], color="w", lw=3, solid_capstyle="butt", zorder=20)
    ax.plot([bx, bx + bar], [by, by], color="k", lw=1.2, solid_capstyle="butt", zorder=21)
    ax.text(bx + bar / 2, by + 0.015 * span, f"{bar:g} m", ha="center", va="bottom", fontsize=7, zorder=21,
            bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 0.8})
    from .style import operator_name
    head = (TERRAIN_ZH[tile["terrain"]] + " · " if show_terrain else "") + st["meta"]["name"] + " · " + operator_name(st, P["op"])
    m = P["metrics"]
    parts = [f"{int(m.get('households_placed', 0))}/{int(m.get('households', 0))} 户", f"房 {int(m.get('buildings', 0))}"]
    if m.get("lambda") == m.get("lambda") and "lambda" in m:
        parts.append(f"λ {m['lambda']:.1f}")
    if P["op"] == "dispersed" and "site_clark_evans" in m:
        parts.append(f"簇 R {m['site_clark_evans']:.2f}")
    elif "coverage" in m and m["coverage"] == m["coverage"]:
        parts.append(f"覆盖 {m['coverage']:.0%}")
    ok = not tile["hard_fail"]
    status = "硬项全过" if ok else "× " + "、".join(tile["hard_fail"])
    if tile["soft_miss"]:
        status += f"（软项差 {len(tile['soft_miss'])} 条）"
    ax.set_title(head, fontsize=10, pad=14)
    ax.text(0.5, 1.012, " · ".join(parts) + "   " + status, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
            color="#2d6a2d" if ok else "#b3261e")


def write_gallery(out: Path, tiles: list[dict], title: str, rows_by_terrain: bool, ncols_max: int = 6) -> Path:
    n = len(tiles)
    if rows_by_terrain:
        order = list(dict.fromkeys(t["terrain"] for t in tiles))
        rows = [[t for t in tiles if t["terrain"] == k] for k in order]
        ncols = max(len(r) for r in rows)
    else:
        ncols = max(1, min(ncols_max, round(math.sqrt(n * 1.4)), n))
        rows = [tiles[i:i + ncols] for i in range(0, n, ncols)]
    nrows = len(rows)
    fig = plt.figure(figsize=(ncols * TILE_IN, nrows * (TILE_IN + 0.45) + 0.6), dpi=100)
    top = 1.0 - 0.6 / fig.get_figheight()
    fig.suptitle(title, fontsize=13, y=1.0 - 0.22 / fig.get_figheight())
    gh = top / nrows
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            ax = fig.add_axes([c / ncols + 0.008, top - (r + 1) * gh + 0.01, 1.0 / ncols - 0.016, gh - 0.45 / fig.get_figheight() - 0.01])
            _draw_tile(ax, tile, rows_by_terrain)
    p = out / "gallery.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def gallery_json(tiles: list[dict], req: dict) -> dict:
    from .style import operator_name

    def num(v):
        return round(float(v), 4) if isinstance(v, (int, float)) and v == v else None
    return {
        "tool": "skyisle town gallery", "request": req,
        "tiles": [{"terrain": TERRAIN_ZH[t["terrain"]], "style_id": t["style"]["meta"]["id"], "style": t["style"]["meta"]["name"],
                   "operator": t["P"]["op"], "operator_name": operator_name(t["style"], t["P"]["op"]), "operator_forced": t["forced"] is not None,
                   "operators_fit": list(t["P"].get("ops_fit", [])),
                   "households": int(t["P"]["metrics"].get("households", 0)), "households_placed": int(t["P"]["metrics"].get("households_placed", 0)),
                   "compounds": len(t["P"]["compounds"]), "buildings": len(t["P"]["buildings"]), "roads": len(t["P"]["roads"]),
                   "bridges": len(t["P"]["bridges"]), "window_half_m": round(t["meta"]["half_m"], 1),
                   "metrics": {k: num(t["P"]["metrics"][k]) for k in METRIC_KEYS if k in t["P"]["metrics"]},
                   "hard_fail": t["hard_fail"], "soft_miss": t["soft_miss"], "plan_s": round(t["plan_s"], 3)} for t in tiles],
    }


def run_gallery(a, cfg: dict) -> int:
    """命令入口：skyisle town gallery。"""
    if not a.terrain:
        raise SystemExit("town gallery 要 --terrain（逗号分隔，可几种），如：--terrain 河谷 或 --terrain 平原,河谷,朝阳坡")
    terrains = [terrain_id(t.strip()) for t in a.terrain.replace("，", ",").split(",") if t.strip()]
    styles = _styles(a.styles or "all")
    scale = scale_id(a.scale or "村")
    each = (a.operators or "default") == "each"
    t0 = time.perf_counter()
    print(f"画廊：{'、'.join(TERRAIN_ZH[t] for t in terrains)} × {len(styles)} 个风格{'（每个能用的算子各一格）' if each else ''}，"
          f"{SCALE_ZH[scale]} {a.households} 户，种子 {a.seed}")
    tiles, grounds = gallery_tiles(terrains, styles, a.households, cfg, seed=a.seed, lat_deg=a.lat, each=each, sets=a.sets, scale=scale)
    lat = next(iter(grounds.values()))[1]["lat_deg"]
    tag = "+".join(TERRAIN_ZH[t] for t in terrains) + f"-{SCALE_ZH[scale]}{a.households}户-s{a.seed}" + ("-各算子" if each else "")
    out = Path(a.out) / "town" / "gallery" / tag
    out.mkdir(parents=True, exist_ok=True)
    title = (f"画廊 · 合成地形「{'、'.join(TERRAIN_ZH[t] for t in terrains)}」· {SCALE_ZH[scale]} {a.households} 户 · 种子 {a.seed} · "
             f"纬度 {lat:.0f}°（{'北' if lat >= 0 else '南'}半球）")
    t1 = time.perf_counter()
    png = write_gallery(out, tiles, title, len(terrains) > 1)
    req = {"terrains": [TERRAIN_ZH[t] for t in terrains], "styles": styles, "scale": SCALE_ZH[scale], "households": a.households,
           "seed": a.seed, "lat_deg": lat, "operators": "each" if each else "default", "sets": list(a.sets)}
    J = gallery_json(tiles, req)
    J["timing_s"] = {"plans": round(t1 - t0, 2), "render": round(time.perf_counter() - t1, 2)}
    (out / "gallery.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    nbad = sum(1 for t in tiles if t["hard_fail"])
    print(f"{len(tiles)} 格，硬项没过 {nbad} 格；方案 {t1 - t0:.1f} s，出图 {time.perf_counter() - t1:.1f} s")
    print(f"产物：{out}（{png.name}、gallery.json）")
    return 0
