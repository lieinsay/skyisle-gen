"""5.6 输出：island.json、height.png（16 位）、terrain.npz、preview.png（总览）。后续步骤再加 landcover / water / arable / climate。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

from .grid import write_png16, write_png8  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LANDCOVER_CLASSES = ["虚空", "崖缘", "裸岩", "高山草甸", "林地", "灌丛", "草坡", "可耕地", "梯田", "湿地", "河道", "湖"]
LANDCOVER_PALETTE = [(20, 24, 40), (90, 80, 75), (150, 150, 150), (170, 200, 120), (40, 110, 50), (120, 150, 70),
                     (190, 200, 110), (230, 200, 90), (210, 170, 60), (90, 160, 150), (40, 90, 200), (30, 60, 170)]


def write_terrain(out: Path, g: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h = g["height"]
    hv = np.where(np.isnan(h), 0.0, h)
    scale = 65535.0 / max(1.0, float(np.nanmax(h)) * 1.02)
    write_png16(out / "height.png", np.round(hv * scale))
    g["json"]["raster"]["height_png_scale_m_per_unit"] = round(1.0 / scale, 6)
    arrays = {"height": h.astype(np.float32), "island_id": g["island_id"].astype(np.int16), "cliff": g["cliff"]}
    for k in ("flowacc_km2", "river", "lake", "landcover", "arable", "slope_deg", "stream"):
        if k in g:
            arrays[k] = g[k]
    np.savez_compressed(out / "terrain.npz", **arrays)
    if "landcover" in g:
        write_png8(out / "landcover.png", g["landcover"], LANDCOVER_PALETTE)
        water = np.zeros_like(g["landcover"], dtype=np.uint8)
        water[g["stream"] > 0] = 1
        water[g["river"] > 0] = 1 + g["river"][g["river"] > 0]
        water[g["lake"]] = 5
        write_png8(out / "water.png", water, [(0, 0, 0), (120, 170, 255), (80, 130, 240), (50, 100, 220), (20, 70, 200), (30, 60, 170)])
        write_png8(out / "arable.png", g["arable"] * 120, None)
    (out / "island.json").write_text(json.dumps(g["json"], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def _auto_exag(h: np.ndarray, res_m: float) -> float:
    """竖直夸张：让坡度 P90 在图上约 30°（低缓的大岛也看得出沟谷）。"""
    hv = np.where(np.isnan(h), np.nan, h)
    gy, gx = np.gradient(hv, res_m)
    s = np.hypot(gx, gy)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return 1.0
    return float(np.clip(0.577 / max(np.quantile(s, 0.9), 1e-6), 1.0, 40.0))


def _hillshade_rgb(h: np.ndarray, res_m: float, vmin: float, vmax: float, cmap_name: str = "gist_earth"):
    hv = np.where(np.isnan(h), np.nanmin(h) if np.isfinite(np.nanmin(h)) else 0.0, h)
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.get_cmap(cmap_name)
    norm = (np.clip(hv, vmin, vmax) - vmin) / max(1e-9, vmax - vmin)
    rgb = cmap(norm)[..., :3]
    shaded = ls.shade_rgb(rgb, hv, vert_exag=_auto_exag(h, res_m), dx=res_m, dy=res_m, blend_mode="soft")
    shaded[np.isnan(h)] = (0.08, 0.09, 0.16)
    return shaded


def _lens_panel(ax, g: dict):
    """总览主图：晕渲 + 岛号 + 索桥 / 短渡 / 导水槽 + 水系 + 可耕地。"""
    J = g["json"]
    r = J["raster"]
    res_m = r["res_m"]
    H, W = g["height"].shape
    x0, y0 = r["origin_km"]
    extent = [x0, x0 + W * res_m / 1000.0, y0 - H * res_m / 1000.0, y0]
    h = g["height"]
    vmin, vmax = float(np.nanmin(h)), float(np.nanmax(h))
    ax.imshow(_hillshade_rgb(h, res_m, vmin, vmax), extent=extent, origin="upper", interpolation="nearest")
    if "arable" in g:
        ar = np.where(g["arable"] > 0, 1.0, np.nan)
        ax.imshow(ar, extent=extent, origin="upper", cmap="autumn", alpha=0.55, vmin=0, vmax=1, interpolation="nearest")
    if "river" in g:
        rv = np.where(g["river"] > 0, g["river"].astype(float), np.nan)
        ax.imshow(rv, extent=extent, origin="upper", cmap="Blues", alpha=0.95, vmin=-1, vmax=3, interpolation="nearest")
        lk = np.where(g["lake"], 1.0, np.nan)
        ax.imshow(lk, extent=extent, origin="upper", cmap="winter", alpha=0.9, vmin=0, vmax=1, interpolation="nearest")
    isl = J["islands"]
    cx = {i["id"]: i["center_km"] for i in isl}
    for e in J["links"]:
        a, b = cx[e["a"]], cx[e["b"]]
        if e["kind"] == "bridge":
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#ffdd55", lw=1.2, alpha=0.9)
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#ffffff", lw=0.7, ls="--", alpha=0.6)
    for a, b in J["channels"]:
        pa, pb = cx[a], cx[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#33ccff", lw=2.2, alpha=0.8)
    for i in isl:
        ax.text(i["center_km"][0], i["center_km"][1], str(i["id"]), color="white", fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.35, lw=0))
    ax.set_xlabel("km 东")
    ax.set_ylabel("km 北")
    ax.set_aspect("equal")


def write_preview(out: Path, g: dict) -> Path:
    J = g["json"]
    has_climate = "climate" in g
    fig = plt.figure(figsize=(17, 10) if has_climate else (12, 10))
    if has_climate:
        gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 1], height_ratios=[1, 1, 0.55], hspace=0.35, wspace=0.25)
        ax = fig.add_subplot(gs[:2, 0])
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], hspace=0.3, wspace=0.2)
        ax = fig.add_subplot(gs[:, 0])
    _lens_panel(ax, g)
    m = J["meta"]
    ax.set_title(f"岛群 #{m['node']}（{m['lat']:.1f}°, {m['lon']:.1f}°）{m['age_zh']} · {len(J['islands'])} 岛 · 陆地 {J['constraints']['area_km2']['actual']:.0f} km²",
                 fontsize=10)
    # 右上：岛的大小 / 峰高
    ax2 = fig.add_subplot(gs[0, 1])
    isl = J["islands"]
    ax2.scatter([i["area_km2"] for i in isl], [i["peak_m"] for i in isl], s=14, c=["#e6550d" if i["id"] == 0 else "#3182bd" for i in isl])
    ax2.set_xscale("log")
    ax2.set_xlabel("岛面积 km²")
    ax2.set_ylabel("峰高 m")
    ax2.set_title("各岛：面积 × 峰高", fontsize=9)
    ax2.grid(alpha=0.3)
    # 右中：主岛放大（晕渲 + 等高线）/ 地表占比
    ax3 = fig.add_subplot(gs[1, 1])
    res_m = J["raster"]["res_m"]
    if "landcover" not in J:
        r0, c0, mm, _ = J["islands"][0]["bbox_cells"]
        sub = g["height"][max(0, r0):r0 + mm, max(0, c0):c0 + mm]
        sub_id = g["island_id"][max(0, r0):r0 + mm, max(0, c0):c0 + mm]
        sub = np.where(sub_id == 0, sub, np.nan)
        rr, cc = np.where(~np.isnan(sub))
        sub = sub[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        ax3.imshow(_hillshade_rgb(sub, res_m, float(np.nanmin(sub)), float(np.nanmax(sub)), "terrain"), interpolation="nearest")
        ax3.contour(np.where(np.isnan(sub), np.nanmin(sub), sub), levels=10, colors="k", linewidths=0.3, alpha=0.5)
        ax3.set_title(f"主岛放大（{J['islands'][0]['age_zh']}，峰 {J['islands'][0]['peak_m']:.0f} m，岸缘 {J['islands'][0]['rim_m']:.0f} m）", fontsize=9)
        ax3.set_xticks([])
        ax3.set_yticks([])
    if "landcover" in J:
        lc = J["landcover"]["share"]
        names = [k for k in LANDCOVER_CLASSES[1:] if k in lc]
        vals = [lc[k] for k in names]
        cols = [tuple(c / 255 for c in LANDCOVER_PALETTE[LANDCOVER_CLASSES.index(k)]) for k in names]
        ax3.barh(names, vals, color=cols)
        ax3.set_title("地表占比（全群陆地）", fontsize=9)
        ax3.invert_yaxis()
    if has_climate:
        from .climate import draw_climate_panels
        draw_climate_panels(fig, gs, g)
    fig.suptitle(f"[seed {m['seed']} · run {m['run']} · 节点 {m['node']} · {m['res_m']:.0f} m/格]", fontsize=10)
    p = out / "preview.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def write_preview_main(out: Path, g: dict) -> Path:
    """主岛放大图：晕渲 + 等高线 + 河 / 湖 / 溪涧 + 可耕地 + 地表底色，给策划与场景美术看岛内细节。"""
    J = g["json"]
    res_m = J["raster"]["res_m"]
    r0, c0, mm, _ = J["islands"][0]["bbox_cells"]
    sl = (slice(max(0, r0), r0 + mm), slice(max(0, c0), c0 + mm))
    ids = g["island_id"][sl]
    m0 = ids == 0
    rr, cc = np.where(m0)
    sub = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    h = np.where(m0, g["height"][sl], np.nan)[sub]
    m = m0[sub]
    fig, ax = plt.subplots(figsize=(11, 9))
    shaded = _hillshade_rgb(h, res_m, float(np.nanmin(h)), float(np.nanmax(h)), "terrain")
    if "landcover" in g:
        lc = g["landcover"][sl][sub]
        pal = np.array(LANDCOVER_PALETTE, dtype=float) / 255.0
        col = pal[np.clip(lc, 0, len(pal) - 1)]
        shaded = np.where(m[..., None], 0.55 * shaded + 0.45 * col, shaded)
    ax.imshow(shaded, interpolation="nearest")
    ax.contour(np.where(np.isnan(h), np.nanmin(h), h), levels=12, colors="k", linewidths=0.35, alpha=0.6)
    if "river" in g:
        rv = g["river"][sl][sub].astype(float)
        st = g["stream"][sl][sub].astype(float)
        ax.imshow(np.where(st > 0, 1.0, np.nan), cmap="Blues", vmin=0, vmax=2, alpha=0.5, interpolation="nearest")
        ax.imshow(np.where(rv > 0, rv, np.nan), cmap="Blues", vmin=-1, vmax=3, alpha=1.0, interpolation="nearest")
        ax.imshow(np.where(g["lake"][sl][sub], 1.0, np.nan), cmap="winter", vmin=0, vmax=1, alpha=0.95, interpolation="nearest")
        ar = g["arable"][sl][sub]
        ax.contour(ar > 0, levels=[0.5], colors="#ffdd33", linewidths=0.6)
    km = 10.0 * 1000.0 / res_m
    ax.plot([10, 10 + km], [h.shape[0] - 10, h.shape[0] - 10], color="w", lw=3)
    ax.text(10 + km / 2, h.shape[0] - 16, "10 km", color="w", ha="center", fontsize=9)
    i0 = J["islands"][0]
    hy = J.get("hydro", {})
    ax.set_title(f"主岛（{i0['age_zh']}，{i0['area_km2']:.0f} km²，峰 {i0['peak_m']:.0f} m，岸缘 {i0['rim_m']:.0f} m，崖 {i0['cliff_m']:.0f} m）"
                 + (f" · 河阈 {hy.get('river_threshold_km2')} km² · 湖 {i0.get('n_lakes', 0)} · 盆地 {hy.get('main_basins', {}).get('n_basins', '-')}（大 {hy.get('main_basins', {}).get('n_large', '-')}）" if hy else ""),
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if "landcover" in g:
        from matplotlib.patches import Patch
        share = J["landcover"]["share"]
        handles = [Patch(color=tuple(c / 255 for c in LANDCOVER_PALETTE[k]), label=f"{LANDCOVER_CLASSES[k]} {share.get(LANDCOVER_CLASSES[k], 0) * 100:.0f}%")
                   for k in range(1, 12) if share.get(LANDCOVER_CLASSES[k], 0) >= 0.003]
        ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.7)
    p = out / "preview_main.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p
