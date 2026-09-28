"""出图（PLAN-TOWN 7.10）：plan.png。第一步只画地面——地表 / 田 / 水 / 虚空 + 晕渲 + 等高线 + 漫水与崖缘退让带；之后叠路网、地块、建筑。"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import numpy as np  # noqa: E402

from ..island.output import LANDCOVER_PALETTE  # noqa: E402
from . import SCALE_ZH, TERRAIN_ZH  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WATER_RGB = {1: (45, 95, 200), 2: (95, 150, 225), 3: (35, 70, 175), 4: (30, 55, 140)}
SKY_RGB = (18, 22, 38)
FARM_RGB = (226, 204, 120)


def _stride(sd: dict, max_px: int) -> int:
    return max(1, int(math.ceil(max(sd["H"], sd["W"]) / max_px)))


def _hillshade(h: np.ndarray, res: float, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    hh = np.where(np.isfinite(h), h, np.nanmin(h) if np.isfinite(h).any() else 0.0)
    gy, gx = np.gradient(hh, res)
    # 行向南：np.gradient 的第 0 轴是向南增加，北向的坡 = −gy
    nx, ny, nz = -gx, gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    return np.clip((nx * lx + ny * ly + nz * lz) / norm, 0.0, 1.0)


def ground_rgb(sd: dict, stride: int = 1) -> np.ndarray:
    """地面底图（uint8 RGB）：地表类别色 → 田 → 晕渲 → 漫水 → 水 → 崖缘退让带 → 虚空。"""
    s = (slice(None, None, stride), slice(None, None, stride))
    lc = sd["landcover"][s]
    pal = np.array(LANDCOVER_PALETTE, dtype=np.float64)
    rgb = pal[np.clip(lc, 0, len(pal) - 1)]
    farm = sd["farmland"][s]
    rgb[farm] = FARM_RGB
    shade = _hillshade(sd["height"][s].astype(np.float64), sd["res_m"] * stride)
    rgb *= (0.45 + 0.65 * shade)[..., None]
    flood = sd["flood"][s]
    rgb[flood] = rgb[flood] * 0.75 + np.array([120, 170, 230]) * 0.25
    water = sd["water"][s]
    for k, c in WATER_RGB.items():
        rgb[water == k] = c
    edge = sd["edge"][s]
    rgb[edge] = rgb[edge] * 0.55 + np.array([200, 60, 50]) * 0.45
    rgb[sd["sky"][s]] = SKY_RGB
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _contour_step(h: np.ndarray) -> float:
    fin = h[np.isfinite(h)]
    if fin.size == 0:
        return 2.0
    span = float(fin.max() - fin.min())
    return 2.0 if span < 60 else 5.0 if span < 200 else 10.0 if span < 500 else 20.0


def _title(meta: dict) -> str:
    scale = SCALE_ZH.get(meta.get("scale", ""), meta.get("scale", ""))
    hh = meta.get("households")
    if meta.get("source") == "island":
        head = f"{meta['run']} #{meta['node']} {meta['site']}（{scale}，{hh} 户）"
    elif meta.get("source") == "synth":
        head = f"合成地形「{TERRAIN_ZH.get(meta['terrain'], meta['terrain'])}」（{scale}，{hh} 户，种子 {meta['seed']}）"
    else:
        head = f"高程图 {Path(meta.get('file', '')).name}（{scale}，{hh} 户）"
    style = meta.get("style_name")
    return head + (f" · {style}" if style else " · 地面")


def write_plan_png(out: Path, sd: dict, meta: dict, cfg: dict, stats: dict | None = None) -> Path:
    rc = cfg["render"]
    st = _stride(sd, int(rc["max_px"]))
    img = ground_rgb(sd, st)
    res, H, W = sd["res_m"], sd["H"], sd["W"]
    ext = [sd["x0"], sd["x0"] + W * res, sd["y0"] - H * res, sd["y0"]]
    px = img.shape[1]
    fig = plt.figure(figsize=(px / 100.0 + 2.6, img.shape[0] / 100.0 + 0.9), dpi=100)
    ax = fig.add_axes([0.02, 0.04, px / (px + 260.0), 0.9])
    ax.imshow(img, extent=ext, origin="upper", interpolation="nearest")
    h = sd["height"][::st, ::st].astype(np.float64)
    h[sd["water"][::st, ::st] > 0] = np.nan          # 水下不画等高线
    step = max(_contour_step(h), float(rc["contour_m"]))
    fin = h[np.isfinite(h)]
    if fin.size and fin.max() - fin.min() > step:
        levels = np.arange(math.ceil(fin.min() / step) * step, fin.max(), step)
        ys = sd["y0"] - (np.arange(h.shape[0]) * st + 0.5 * st) * res
        xs = sd["x0"] + (np.arange(h.shape[1]) * st + 0.5 * st) * res
        ax.contour(xs, ys, np.ma.masked_invalid(h), levels=levels, colors="k", linewidths=0.35, alpha=0.35)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_title(_title(meta), fontsize=11)
    # 比例尺与指北
    span = W * res
    bar = 10 ** math.floor(math.log10(span / 4))
    bar = bar * (5 if span / bar > 20 else 2 if span / bar > 8 else 1)
    x0, y0 = ext[0] + 0.04 * span, ext[2] + 0.04 * span
    ax.plot([x0, x0 + bar], [y0, y0], color="w", lw=3, solid_capstyle="butt")
    ax.plot([x0, x0 + bar], [y0, y0], color="k", lw=1.2, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 + 0.012 * span, f"{bar:g} m", ha="center", va="bottom", fontsize=8,
            bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1})
    ax.annotate("北", xy=(ext[1] - 0.05 * span, ext[3] - 0.04 * span), xytext=(ext[1] - 0.05 * span, ext[3] - 0.12 * span),
                ha="center", fontsize=9, arrowprops={"arrowstyle": "-|>", "color": "k"}, bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1})
    handles = [Patch(color=np.array(FARM_RGB) / 255, label="田"), Patch(color=np.array(WATER_RGB[1]) / 255, label="河"),
               Patch(color=np.array(WATER_RGB[2]) / 255, label="季节性溪涧"), Patch(color=np.array(WATER_RGB[3]) / 255, label="湖 / 海"),
               Patch(color=(0.62, 0.72, 0.86), label="漫水"), Patch(color=(0.8, 0.3, 0.25), label="崖缘退让带"),
               Patch(color=np.array(SKY_RGB) / 255, label="虚空（岛外）")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    lines = [f"纬度 {meta.get('lat_deg', 0):.1f}°（{'北' if meta.get('lat_deg', 0) >= 0 else '南'}半球，朝阳 = 朝{'南' if meta.get('lat_deg', 0) >= 0 else '北'}）",
             f"等高线 {step:g} m · 格 {res:g} m"]
    if stats:
        if stats.get("height_m"):
            lines.append(f"高程 {stats['height_m'][0]:.0f}–{stats['height_m'][1]:.0f} m")
        if stats.get("slope_deg"):
            lines.append(f"坡度 中位 {stats['slope_deg']['p50']:.1f}° · P90 {stats['slope_deg']['p90']:.1f}°")
        lines.append(f"田 {stats['farmland_share']:.0%} · 漫水 {stats['flood_share']:.0%} · 虚空 {stats['sky_share']:.0%}")
    w = meta.get("winter") or {}
    if w.get("from_deg") is not None:
        lines.append(f"冬季风从 {w['from_deg']:.0f}° 来")
    fig.text(px / (px + 260.0) + 0.03, 0.08, "\n".join(lines), fontsize=8, va="bottom")
    p = out / "plan.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p
