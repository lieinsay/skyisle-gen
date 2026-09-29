"""场地地形的前端（PLAN-TOWN 7.1）：从岛群产物裁窗口、换坐标，交给 C++ 细化；合成地形与外部高程图同样落成 Site。

Site 在 Python 里是一个 dict：numpy 数组（height / water_level / water / sky / edge / farmland / flood / landcover / island / water_dist_m）、
元数据（res_m / H / W / x0 / y0 / frame_x / frame_y / lat_deg）与 rivers（每条 line [n, 2] + width_m / depth_m / surface_m / seasonal）。
平面坐标：x 向东、y 向北（m），原点是窗口中心；格 (i, j) 的中心 = (x0 + (j + 0.5)·res, y0 − (i + 0.5)·res)。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import core, flat, ground_seed, res_m, terrain_id, window_half_m

WATER_NAMES = {0: "无", 1: "河", 2: "季节性溪涧", 3: "湖", 4: "海"}


# ---------------------------------------------------------------- 岛群里的聚落
def _group_dir(ctx, node: int) -> Path:
    d = ctx.out_dir / "islands" / str(node)
    if not (d / "settlements.json").exists():
        raise FileNotFoundError(f"{d} 没有岛群产物：先跑 skyisle island {node} --run {ctx.out_dir}")
    return d


def find_site(S: dict, name: str) -> dict:
    """按名字在 settlements.json 里找聚落：村 / 散户 / 镇 / 专业聚落；「城」= 都城所在的邑治。
    返回 {kind, name, cell, km, households, households_farm, households_market, island, rec}。"""
    towns_by_village = {t["village"]: t for t in S.get("towns", []) if t.get("village") is not None}
    if name == "城":
        if not S.get("city"):
            raise ValueError("这个岛群不是都，没有城")
        v = next(x for x in S["villages"] if x.get("seat"))
        c = S["city"]
        return {"kind": "city", "name": "城", "cell": v["cell"], "km": v["km"], "households": int(c["households"]),
                "households_farm": int(v["households"]), "households_market": int(c["households"]) - int(v["households"]),
                "island": v["island"], "rec": {"village": v, "city": c}}
    for v in S.get("villages", []):
        if v["name"] == name:
            t = towns_by_village.get(v["id"])
            if t is not None:
                return {"kind": "town", "name": name, "cell": v["cell"], "km": v["km"],
                        "households": int(t["households_farm"]) + int(t["households_market"]),
                        "households_farm": int(t["households_farm"]), "households_market": int(t["households_market"]),
                        "island": v["island"], "rec": {"village": v, "town": t}}
            return {"kind": "village", "name": name, "cell": v["cell"], "km": v["km"], "households": int(v["households"]),
                    "households_farm": int(v["households"]), "households_market": 0, "island": v["island"], "rec": v}
    for t in S.get("towns", []):
        if t["name"] == name:
            v = next((x for x in S["villages"] if x["id"] == t.get("village")), None)
            return {"kind": "town", "name": name, "cell": t["cell"], "km": t["km"],
                    "households": int(t["households_farm"]) + int(t["households_market"]),
                    "households_farm": int(t["households_farm"]), "households_market": int(t["households_market"]),
                    "island": t["island"], "rec": {"village": v, "town": t}}
    for h in S.get("hamlets", []):
        if h["name"] == name:
            return {"kind": "hamlet", "name": name, "cell": h["cell"], "km": h["km"], "households": int(h["households"]),
                    "households_farm": int(h["households"]), "households_market": 0, "island": h["island"], "rec": h}
    for s in S.get("specials", []):
        if s["name"] == name:
            return {"kind": "special", "name": name, "cell": s["cell"], "km": s["km"], "households": int(s["households"]),
                    "households_farm": 0, "households_market": int(s["households"]), "island": s["island"], "rec": s}
    raise ValueError(f"岛群里没有叫「{name}」的聚落（村NNN / 散户NNN / 镇NN / 专业聚落名 / 城）")


def default_scale(kind: str) -> str:
    return {"village": "village", "hamlet": "hamlet", "town": "town", "city": "city", "special": "special"}[kind]


def _winter_wind(C: dict) -> dict:
    """最冷那一季的风（来向，度）与全年的雪日：朝向的「背风」规则与防风林用。"""
    seasons = C.get("seasons") or []
    if not seasons:
        return {}
    cold = min(seasons, key=lambda s: s.get("temp_c", 99.0))
    w = cold.get("wind") or {}
    out = {"season": cold.get("name"), "temp_c": cold.get("temp_c"), "from_deg": w.get("from_deg"), "speed_ms": w.get("speed_ms")}
    wx = C.get("weather") or {}
    out["snow_days"] = wx.get("snow_days")
    out["precip_mm"] = (C.get("annual") or {}).get("precip_mm")
    return out


def _anchors(S: dict, site: dict, frame_x: float, frame_y: float) -> dict:
    """上游锚点，换成窗口平面坐标（m）：本聚落的泊场（在前）与出村大路的去向（同岛的邻村、集镇 / 治所、主泊场、桥头）。"""
    def local(km):
        return [round(float(km[0]) * 1000.0 - frame_x, 2), round(float(km[1]) * 1000.0 - frame_y, 2)]

    rec = site["rec"]
    v = rec.get("village") if isinstance(rec, dict) and "village" in rec else rec
    isl = int(site["island"])
    landings = []
    lid = v.get("landing") if isinstance(v, dict) else None
    for L in S.get("landings", []):
        if lid is not None and L["id"] == lid:
            landings.insert(0, local(L["km"]))
    exits = []
    me = np.array([frame_x, frame_y]) / 1000.0

    def add(km, weight, kind, max_km):
        d = float(np.hypot(km[0] - me[0], km[1] - me[1]))
        if 0.15 < d <= max_km:
            exits.append({"bearing_deg": round(math.degrees(math.atan2(km[0] - me[0], km[1] - me[1])), 2), "dist_m": round(d * 1000.0, 1),
                          "weight": round(weight, 4), "kind": kind})

    mt = v.get("market_town") if isinstance(v, dict) else None
    for t in S.get("towns", []):
        if t.get("island") == isl and (t.get("id") == mt or t.get("seat")):
            add(t["km"], 3.0 if t.get("id") == mt else 2.5, "集镇" if not t.get("seat") else "治所", 12.0)
    near = sorted((x for x in S.get("villages", []) if x.get("island") == isl and x["name"] != site["name"]),
                  key=lambda x: math.hypot(x["km"][0] - me[0], x["km"][1] - me[1]))[:4]
    for x in near:
        d = math.hypot(x["km"][0] - me[0], x["km"][1] - me[1])
        add(x["km"], 1.5 / (1.0 + d / 2.0), "邻村", 6.0)
    for L in S.get("landings", []):
        if L.get("main") and L.get("island") == isl:
            add(L["km"], 2.0, "主泊场", 8.0)
    for bh in S.get("bridgeheads", []):
        if bh.get("island") == isl:
            add(bh["km"], 1.2, "桥头", 4.0)
    return {"landings": landings, "exits": exits}


def site_from_group(ctx, node: int, name: str, scale: str | None, cfg: dict, half_m: float | None = None,
                    half_factor: float = 1.0) -> tuple[dict, dict]:
    """岛群 node 里名叫 name 的聚落所在的一块地：返回 (Site, meta)。half_factor：风格要的窗口放大（散居要更大的地；地面只按坐标取，放大只是多裁一圈）。"""
    gdir = _group_dir(ctx, node)
    S = json.loads((gdir / "settlements.json").read_text(encoding="utf-8"))
    J = json.loads((gdir / "island.json").read_text(encoding="utf-8"))
    C = json.loads((gdir / "climate.json").read_text(encoding="utf-8")) if (gdir / "climate.json").exists() else {}
    T = np.load(gdir / "terrain.npz")
    if (gdir / "rivers.json").exists():
        R = json.loads((gdir / "rivers.json").read_text(encoding="utf-8"))
    elif (T["river"] > 0).any() or (T["stream"] > 0).any():
        # 河在细栅格上是按中心线重刻的（粗栅格上的河槽会先被填平），没有中心线河就没了——宁可报错也不静默丢河
        raise FileNotFoundError(f"{gdir} 有河道格却没有 rivers.json（河道成形之前的旧产物）：先重跑 skyisle island {node} --run {ctx.out_dir}")
    else:
        R = {"lines": []}
    site = find_site(S, name)
    scale = scale or default_scale(site["kind"])
    hh = int(site["households"])
    half = float(half_m) if half_m else window_half_m(scale, hh, cfg) * float(half_factor)
    res = res_m(scale, cfg)
    ras = J["raster"]
    cres = float(ras["res_m"])
    row, col = int(site["cell"][0]), int(site["cell"][1])
    Hg, Wg = T["height"].shape
    m = int(math.ceil(half / cres)) + 4
    r0, r1, c0, c1 = max(0, row - m), min(Hg, row + m + 1), max(0, col - m), min(Wg, col + m + 1)
    sl = (slice(r0, r1), slice(c0, c1))
    riv = (T["river"][sl] > 0) | (T["stream"][sl] > 0)
    rc, cc = row + 0.5, col + 0.5                      # 窗口中心（格心）在群栅格里的连续坐标
    frame_x = float(ras["origin_km"][0]) * 1000.0 + cc * cres
    frame_y = float(ras["origin_km"][1]) * 1000.0 - rc * cres
    rivers = []
    reach = half + 300.0
    for ln in R.get("lines", []):
        P = np.asarray(ln["pts"], dtype=np.float64)
        if P.shape[0] < 2:
            continue
        xy = np.stack([(P[:, 1] - cc) * cres, -(P[:, 0] - rc) * cres], axis=1)
        if np.abs(xy).max(axis=1).min() > reach:
            continue
        rivers.append({"line": np.ascontiguousarray(xy), "width_m": np.ascontiguousarray(P[:, 2]), "seasonal": bool(P[:, 3].max() == 0)})
    lat = float(J.get("meta", {}).get("lat", 0.0))
    inp = {
        "height": np.ascontiguousarray(T["height"][sl], dtype=np.float64),
        "island": np.ascontiguousarray(T["island_id"][sl], dtype=np.int16),
        "lake": np.ascontiguousarray(T["lake"][sl]),
        "floodplain": np.ascontiguousarray(T["floodplain"][sl]),
        "arable": np.ascontiguousarray(T["arable"][sl] >= 1),
        "terrace": np.ascontiguousarray(T["arable"][sl] == 2),
        "landcover": np.ascontiguousarray(T["landcover"][sl]),
        "river_depth": np.ascontiguousarray(np.where(riv, T["river_depth_m"][sl], 0.0), dtype=np.float64),
        "coarse_res_m": cres, "center_r": rc - r0, "center_c": cc - c0, "frame_x": frame_x, "frame_y": frame_y,
        "half_m": half, "res_m": res, "lat_deg": lat, "seed": ground_seed(ctx.seed, f"town:{node}:ground"), "rivers": rivers,
    }
    sd = core().town_site_window(inp, flat(cfg))
    meta = {
        "source": "island", "run": ctx.out_dir.name, "seed": int(ctx.seed), "node": int(node), "site": site["name"], "kind": site["kind"],
        "scale": scale, "households": hh, "households_farm": int(site["households_farm"]), "households_market": int(site["households_market"]),
        "island": int(site["island"]), "cell": [row, col], "center_km": [round(frame_x / 1000.0, 4), round(frame_y / 1000.0, 4)],
        "lat_deg": round(lat, 4), "winter": _winter_wind(C), "half_m": half,
        "upstream": {k: v for k, v in (site["rec"].items() if isinstance(site["rec"], dict) and "cell" in site["rec"] else [])
                     if k in ("elev_m", "field", "landing", "water", "shore_dist_km", "market_town", "seat", "subtype", "note")},
        "anchors": _anchors(S, site, frame_x, frame_y),
    }
    return sd, meta


# ---------------------------------------------------------------- 合成地形与外部高程图
def site_synth(terrain: str, scale: str, households: int, cfg: dict, seed: int = 1, lat_deg: float | None = None,
               half_m: float | None = None, half_factor: float = 1.0) -> tuple[dict, dict]:
    kind = terrain_id(terrain)
    half = float(half_m) if half_m else window_half_m(scale, households, cfg) * float(half_factor)
    res = res_m(scale, cfg)
    lat = float(cfg["synth"]["lat_deg"] if lat_deg is None else lat_deg)
    sd = core().town_site_synth(kind, half, res, lat, ground_seed(seed, f"town:synth:{kind}"), flat(cfg))
    meta = {"source": "synth", "terrain": kind, "seed": int(seed), "scale": scale, "households": int(households),
            "households_farm": int(households), "households_market": 0, "lat_deg": lat, "winter": {},
            "half_m": half, "half_given": half_m is not None}
    return sd, meta


def site_heightmap(path: Path, res: float, scale: str, households: int, cfg: dict, water: Path | None = None,
                   lat_deg: float | None = None, scale_m: float = 1.0) -> tuple[dict, dict]:
    """16 位灰度高程图：高程 = 灰度 × scale_m（m）；可选同尺寸的水面图（非零 = 水）。"""
    import matplotlib.image as mpimg
    h = mpimg.imread(str(path))
    if h.ndim == 3:
        h = h[..., 0]
    if h.dtype != np.uint16 and h.max() <= 1.0:
        h = h * 65535.0
    h = np.ascontiguousarray(h.astype(np.float64) * float(scale_m))
    w = None
    if water is not None:
        wm = mpimg.imread(str(water))
        w = np.ascontiguousarray((wm[..., 0] if wm.ndim == 3 else wm) > 0)
    lat = float(cfg["synth"]["lat_deg"] if lat_deg is None else lat_deg)
    sd = core().town_site_heightmap(h, w, float(res), lat, 0, flat(cfg))
    meta = {"source": "heightmap", "file": str(path), "scale": scale, "households": int(households), "households_farm": int(households),
            "households_market": 0, "lat_deg": lat, "winter": {}}
    return sd, meta


# ---------------------------------------------------------------- 统计
def site_stats(sd: dict) -> dict:
    """地面的摘要：面积、各类水 / 虚空 / 田 / 漫水 / 退让带的占比、坡度分位、高程范围、河的条数与长度。"""
    H, W, res = sd["H"], sd["W"], sd["res_m"]
    land = ~sd["sky"]
    n = max(1, int(land.sum()))
    h = sd["height"].astype(np.float64)
    gy, gx = np.gradient(np.where(land, h, np.nan), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    sl = slope[land & np.isfinite(slope) & (sd["water"] == 0)]
    water = sd["water"]
    out = {
        "size_m": [round(W * res, 1), round(H * res, 1)], "res_m": res, "cells": int(H * W),
        "sky_share": round(float(sd["sky"].mean()), 4),
        "water_share": {WATER_NAMES[k]: round(float(((water == k) & land).sum()) / n, 4) for k in (1, 2, 3, 4) if (water == k).any()},
        "farmland_share": round(float((sd["farmland"] & land).sum()) / n, 4),
        "flood_share": round(float((sd["flood"] & land).sum()) / n, 4),
        "edge_share": round(float((sd["edge"] & land).sum()) / n, 4),
        "height_m": [round(float(np.nanmin(h[land])), 1), round(float(np.nanmax(h[land])), 1)] if land.any() else None,
        "slope_deg": {"p50": round(float(np.percentile(sl, 50)), 2), "p90": round(float(np.percentile(sl, 90)), 2)} if sl.size else None,
        "rivers": [{"seasonal": r["seasonal"], "length_m": round(float(np.hypot(*np.diff(r["line"], axis=0).T).sum()), 1),
                    "width_m": [round(float(np.min(r["width_m"])), 1), round(float(np.max(r["width_m"])), 1)]} for r in sd["rivers"]],
    }
    return out
