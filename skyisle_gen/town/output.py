"""产物（PLAN-TOWN 7.10）：site.npz（细化后的地面）、plan.json（元数据、统计，之后加路网 / 地块 / 宅院 / 建筑）、plan.png。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SCALE_ZH, TERRAIN_ZH

VERSION = 1
SITE_ARRAYS = ("height", "water_level", "water", "sky", "edge", "farmland", "flood", "landcover", "island", "water_dist_m")


def out_dir(ctx_out: Path | None, meta: dict, style: str | None) -> Path:
    tag = style or "地面"
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


def write_site(out: Path, sd: dict) -> Path:
    arrs = {k: sd[k] for k in SITE_ARRAYS}
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
