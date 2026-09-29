"""产物（PLAN-TOWN 7.10）：site.npz（细化后的地面 + 占地栅格）、plan.json（元数据、地面统计、路 / 桥 / 宅院 / 房 / 塘场井树 / 户 / 指标 / 校验）、
style.resolved.toml（这次实际用的风格参数）、plan.png / plan-detail.png / plan.svg。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SCALE_ZH, TERRAIN_ZH

VERSION = 2
SITE_ARRAYS = ("height", "water_level", "water", "sky", "edge", "farmland", "flood", "landcover", "island", "water_dist_m")


def out_dir(ctx_out: Path | None, meta: dict, style: str | None) -> Path:
    tag = style or "地面"
    for ch in '/\\:*?"<>|':   # 算子的叫法里可能有斜杠（「土楼 / 围龙屋」）：目录名里换掉
        tag = tag.replace(ch, "、" if ch == "/" else "-")
    tag = tag.replace(" 、 ", "、")
    if meta["source"] == "island":
        d = ctx_out / "islands" / str(meta["node"]) / "town" / f"{meta['site']}-{tag}"
    elif meta["source"] == "synth":
        base = Path(ctx_out) if ctx_out else Path("out")
        half = f"-半{meta['half_m']:g}m" if meta.get("half_given") else ""
        d = base / "town" / "synth" / f"{TERRAIN_ZH[meta['terrain']]}-{SCALE_ZH[meta['scale']]}{meta['households']}户-s{meta['seed']}{half}-{tag}"
    else:
        base = Path(ctx_out) if ctx_out else Path("out")
        d = base / "town" / "heightmap" / f"{Path(meta['file']).stem}-{SCALE_ZH[meta['scale']]}-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_site(out: Path, sd: dict, occ: np.ndarray | None = None) -> Path:
    arrs = {k: sd[k] for k in SITE_ARRAYS}
    if occ is not None:
        arrs["occ"] = occ   # 0 空 · 1 路 · 2 院 · 3 房 · 4 塘 · 5 场院 · 6 泊场 · 7 桥 · 8 公地 / 广场 / 林带 / 坑 · 9 環濠
    # 河：顶点拼成一张表，river_start[k] 是第 k 条的起点下标
    lines = sd["rivers"]
    if lines:
        arrs["river_xy"] = np.concatenate([r["line"] for r in lines]).astype(np.float64)
        arrs["river_width_m"] = np.concatenate([r["width_m"] for r in lines])
        arrs["river_depth_m"] = np.concatenate([r["depth_m"] for r in lines])
        arrs["river_surface_m"] = np.concatenate([r["surface_m"] for r in lines])
        arrs["river_start"] = np.cumsum([0] + [len(r["line"]) for r in lines[:-1]]).astype(np.int64)
        arrs["river_seasonal"] = np.array([r["seasonal"] for r in lines], dtype=bool)
    arrs["frame"] = np.array([sd["res_m"], sd["x0"], sd["y0"], sd["frame_x"], sd["frame_y"], sd["lat_deg"]], dtype=np.float64)
    p = out / "site.npz"
    np.savez_compressed(p, **arrs)
    return p


def write_plan_json(out: Path, sd: dict, meta: dict, stats: dict, extra: dict | None = None) -> Path:
    J = {
        "tool": "skyisle town", "version": VERSION,
        "note": "聚落营建器（docs/PLAN-TOWN.md）。平面坐标 x 向东、y 向北（m），原点是窗口中心；岛群来源时窗口中心在岛群平面坐标里是 center_km。",
        "meta": meta,
        "frame": {"res_m": sd["res_m"], "H": sd["H"], "W": sd["W"], "x0": sd["x0"], "y0": sd["y0"],
                  "frame_x_m": sd["frame_x"], "frame_y_m": sd["frame_y"],
                  "cell_note": "格 (i, j) 的中心 = (x0 + (j + 0.5)·res, y0 − (i + 0.5)·res)"},
        "site": stats,
    }
    if extra:
        J.update(extra)
    p = out / "plan.json"
    p.write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _r(x, nd=2):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, (list, tuple)):
        return [_r(e, nd) for e in x]
    if isinstance(x, np.ndarray):
        return np.round(x.astype(np.float64), nd).tolist()
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    return x


def plan_json(P: dict, st: dict, style_hash: str) -> dict:
    """方案 → plan.json 的各段（坐标保留两位小数）。"""
    from .plan import HH_KIND, ROAD_CLASSES, ROAD_ZH, SIDE_ZH
    hh = P["households"]
    return {
        "style": {"id": st["meta"]["id"], "name": st["meta"]["name"], "region": st["meta"].get("region", ""), "hash": style_hash},
        "request": P["request"],
        "plan": {"operator": P["op"], "operator_name": P.get("op_name", P["op"]), "operators_fit": list(P.get("ops_fit", [])), "center": _r(P["center"]), "facing_deg": round(P["facing_deg"] % 360.0, 2), "radius_m": round(P["radius_m"], 1)},
        "checks": P["checks"],
        "metrics": {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else None if isinstance(v, float) else v) for k, v in P["metrics"].items()},
        "timing_plan_s": _r(P["timing"], 4),
        "roads": [{"cls": ROAD_CLASSES[r["cls"]], "name": ROAD_ZH[ROAD_CLASSES[r["cls"]]], "width_m": round(r["width_m"], 2), "line": _r(r["line"])}
                  for r in P["roads"]],
        "bridges": [_r(b) for b in P["bridges"]],
        "compounds": [{**_r({k: v for k, v in c.items() if k not in ("access_side",)}), "access_side": SIDE_ZH[c["access_side"]]}
                      for c in P["compounds"]],
        "buildings": [_r(b) for b in P["buildings"]],
        "features": [_r(f) for f in P["features"]],
        "households": [{"id": int(i), "kind": HH_KIND[int(k)], "parent": int(p), "compound": int(c)}
                       for i, k, p, c in zip(hh["id"], hh["kind"], hh["parent"], hh["compound"])],
    }


def write_style(out: Path, st: dict) -> Path:
    from .style import dump_toml
    p = out / "style.resolved.toml"
    head = "# 这次实际用的风格参数（base ← 风格 ← --set 展开后；functions = 通用目录 ← 风格覆盖）\n"
    p.write_text(head + dump_toml(st), encoding="utf-8")
    return p
