"""出图（PLAN-TOWN 7.10）：plan.png（整个窗口）、plan-detail.png（建成区放大）、plan.svg（矢量，可无限放大）。
底图 = 地表 / 田 / 水 / 虚空 + 晕渲 + 等高线 + 漫水与崖缘退让带；有方案时叠路、桥、院、房、塘、场、泊场、井、树，公共建筑标名。
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon  # noqa: E402

import numpy as np  # noqa: E402

from ..island.output import LANDCOVER_PALETTE  # noqa: E402
from . import SCALE_ZH, TERRAIN_ZH  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WATER_RGB = {1: (45, 95, 200), 2: (95, 150, 225), 3: (35, 70, 175), 4: (30, 55, 140)}
SKY_RGB = (18, 22, 38)
FARM_RGB = (226, 204, 120)

ROAD_COLOR = {0: "#c9b68a", 1: "#d8c79b", 2: "#e0d0a8", 3: "#e8dcba", 4: "#cfc293"}
YARD_COLOR, WALL_COLOR = "#ece3cd", "#5a4d40"
BRIDGE_COLOR = "#7a6048"
POND_COLOR, POND_EDGE = "#5c8fc9", "#3d6ea8"
THRESH_COLOR, THRESH_EDGE = "#e6d49e", "#b09a5c"
LANDING_COLOR, LANDING_EDGE = "#cfc7b4", "#7d7563"
WELL_COLOR, TREE_COLOR = "#2f6db5", "#3f7d3a"


def building_color(b: dict) -> str:
    f, r = b.get("func", ""), b.get("role", "")
    if f in ("gate",) or r in ("gatehouse", "gate"):
        return "#2e241c"
    if f == "worship":
        return "#9c2f25"
    if f in ("shrine", "pond_temple"):
        return "#b8433a"
    if f == "landing_shed":
        return "#4f6277"
    if r == "main":
        return "#4a3b30"
    if f == "kitchen":
        return "#80695a"
    if f == "store" or r == "front":
        return "#8b7763"
    return "#6a5646"


def _stride(H: int, W: int, max_px: int) -> int:
    return max(1, int(math.ceil(max(H, W) / max_px)))


def _hillshade(h: np.ndarray, res: float, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    hh = np.where(np.isfinite(h), h, np.nanmin(h) if np.isfinite(h).any() else 0.0)
    gy, gx = np.gradient(hh, res)
    # 行向南：np.gradient 的第 0 轴是向南增加，北向的坡 = −gy
    nx, ny, nz = -gx, gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    return np.clip((nx * lx + ny * ly + nz * lz) / norm, 0.0, 1.0)


def ground_rgb(sd: dict, stride: int = 1, window: tuple[int, int, int, int] | None = None, fade: float = 0.0) -> np.ndarray:
    """地面底图（uint8 RGB）：地表类别色 → 田 → 晕渲 → 漫水 → 水 → 崖缘退让带 → 虚空。window = (i0, i1, j0, j1) 只画一块；fade 往白里褪。"""
    i0, i1, j0, j1 = window or (0, sd["H"], 0, sd["W"])
    s = (slice(i0, i1, stride), slice(j0, j1, stride))
    lc = sd["landcover"][s]
    pal = np.array(LANDCOVER_PALETTE, dtype=np.float64)
    rgb = pal[np.clip(lc, 0, len(pal) - 1)]
    farm = sd["farmland"][s]
    rgb[farm] = FARM_RGB
    shade = _hillshade(sd["height"][s].astype(np.float64), sd["res_m"] * stride)
    rgb *= (0.45 + 0.65 * shade)[..., None]
    flood = sd["flood"][s]
    rgb[flood] = rgb[flood] * 0.75 + np.array([120, 170, 230]) * 0.25
    if fade > 0:
        rgb = rgb * (1 - fade) + 255.0 * fade
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


def _title(meta: dict, P: dict | None, style_name: str | None) -> str:
    scale = SCALE_ZH.get(meta.get("scale", ""), meta.get("scale", ""))
    hh = meta.get("households")
    if meta.get("source") == "island":
        head = f"{meta['run']} #{meta['node']} {meta['site']}（{scale}，{hh} 户）"
    elif meta.get("source") == "synth":
        head = f"合成地形「{TERRAIN_ZH.get(meta['terrain'], meta['terrain'])}」（{scale}，{hh} 户，种子 {meta['seed']}）"
    else:
        head = f"高程图 {Path(meta.get('file', '')).name}（{scale}，{hh} 户）"
    if style_name:
        op = {"fishbone": "鱼骨街村", "organic": "团块生长", "single": "单个宅院"}.get(P.get("op", ""), P.get("op", "")) if P else ""
        return head + f" · {style_name}" + (f"（{op}）" if op else "")
    return head + " · 地面"


def plan_extent(P: dict, margin: float = 40.0) -> tuple[float, float, float, float] | None:
    pts = []
    for c in P["compounds"]:
        pts.append(c["plot"]["c"])
    for f in P["features"]:
        if f["kind"] in ("pond", "threshing", "landing"):
            pts.append(f["p"])
    if not pts:
        return None
    a = np.asarray(pts, dtype=np.float64)
    x0, y0 = a.min(axis=0) - margin
    x1, y1 = a.max(axis=0) + margin
    return float(x0), float(x1), float(y0), float(y1)


def _obb_poly(c, facing_deg, w, d) -> np.ndarray:
    f = math.radians(facing_deg)
    fv = np.array([math.sin(f), math.cos(f)]) * (0.5 * d)
    rv = np.array([math.cos(f), -math.sin(f)]) * (0.5 * w)
    c = np.asarray(c, dtype=np.float64)
    return np.array([c + fv + rv, c + fv - rv, c - fv - rv, c - fv + rv])


def draw_plan(ax, P: dict, pt_per_m: float, labels: bool = True, label_size: float = 7.0) -> None:
    """在已设好范围的坐标轴上画方案（平面坐标 m）。pt_per_m：一米合多少磅（路宽按真宽画）。"""
    feats = P["features"]
    # 场、泊场、塘在路下面
    patches, fc, ec = [], [], []
    for f in feats:
        if f["kind"] == "threshing":
            patches.append(Polygon(f["poly"], closed=True)), fc.append(THRESH_COLOR), ec.append(THRESH_EDGE)
        elif f["kind"] == "landing":
            patches.append(Polygon(f["poly"], closed=True)), fc.append(LANDING_COLOR), ec.append(LANDING_EDGE)
        elif f["kind"] == "pond":
            patches.append(Polygon(f["poly"], closed=True)), fc.append(POND_COLOR), ec.append(POND_EDGE)
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=fc, edgecolors=ec, linewidths=0.6, zorder=3))
    # 路：外框深一点再填色
    roads = sorted(P["roads"], key=lambda r: -r["cls"])
    for casing in (True, False):
        segs, lws, cols = [], [], []
        for r in roads:
            segs.append(np.asarray(r["line"]))
            w = r["width_m"] * pt_per_m
            lws.append(w + (0.8 if casing else 0.0))
            cols.append("#a8966c" if casing else ROAD_COLOR.get(r["cls"], "#e8dcba"))
        if segs:
            ax.add_collection(LineCollection(segs, linewidths=lws, colors=cols, capstyle="round", joinstyle="round", zorder=4 if casing else 5))
    if P["bridges"]:
        segs = [np.array([b["a"], b["b"]]) for b in P["bridges"]]
        ax.add_collection(LineCollection(segs, linewidths=[(b["width_m"] + 1.0) * pt_per_m for b in P["bridges"]], colors=BRIDGE_COLOR,
                                         capstyle="butt", zorder=6))
    # 院子与墙
    yards = [Polygon(_obb_poly(c["plot"]["c"], c["plot"]["facing_deg"], c["plot"]["w"], c["plot"]["d"]), closed=True) for c in P["compounds"]]
    if yards:
        ax.add_collection(PatchCollection(yards, facecolors=YARD_COLOR, edgecolors=[WALL_COLOR if c["walled"] else "#b7a98d" for c in P["compounds"]],
                                          linewidths=max(0.3, 0.4 * pt_per_m), zorder=7))
    # 房
    bl = [Polygon(_obb_poly([b["c"][0], b["c"][1]], b["facing_deg"], b["w"], b["d"]), closed=True) for b in P["buildings"]]
    if bl:
        ax.add_collection(PatchCollection(bl, facecolors=[building_color(b) for b in P["buildings"]], edgecolors="#1e1813",
                                          linewidths=max(0.15, 0.08 * pt_per_m), zorder=8))
        # 正面（檐口 / 门）一侧画一道浅线：看得出朝向
        segs = []
        for b in P["buildings"]:
            q = _obb_poly(b["c"], b["facing_deg"], b["w"], b["d"])
            segs.append(np.array([q[0], q[1]]))
        ax.add_collection(LineCollection(segs, linewidths=max(0.3, 0.35 * pt_per_m), colors="#e8b86a", zorder=9))
    # 树冠盖在路上、压在房子底下（村口的土地庙常在大树底下）；井在最上
    trees = [Circle(f["p"], f["r"]) for f in feats if f["kind"] == "tree"]
    if trees:
        ax.add_collection(PatchCollection(trees, facecolors=TREE_COLOR, edgecolors="white", linewidths=0.4, alpha=0.8, zorder=6.5))
    wells = [Circle(f["p"], max(1.2, f["r"])) for f in feats if f["kind"] == "well"]
    if wells:
        ax.add_collection(PatchCollection(wells, facecolors=WELL_COLOR, edgecolors="white", linewidths=0.4, alpha=0.9, zorder=10))
    if not labels:
        return
    for name, (x, y), up in _labels(P):
        ax.text(x, y + up, name, fontsize=label_size, ha="center", va="bottom" if up else "center", zorder=12,
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.8})


def _labels(P: dict) -> list:
    """要标名字的东西：[(名, 位置, 往上挪几米)]。路边小庙、货棚这样的小房标在房子上方，不然字把房子盖住。"""
    out = [(c["name"], c["plot"]["c"], 0.0) for c in P["compounds"] if c["kind"] == "public"]
    out += [(b["name"], b["c"], 0.5 * max(b["w"], b["d"]) + 1.0) for b in P["buildings"] if b["compound"] < 0]
    out += [(f["name"], f["p"], 0.0) for f in P["features"] if f["kind"] in ("landing", "tree", "pond")]
    return out


def _legend_handles(P: dict | None) -> list:
    h = [Patch(color=np.array(FARM_RGB) / 255, label="田"), Patch(color=np.array(WATER_RGB[1]) / 255, label="河"),
         Patch(color=np.array(WATER_RGB[2]) / 255, label="季节性溪涧"), Patch(color=np.array(WATER_RGB[3]) / 255, label="湖 / 海"),
         Patch(color=(0.62, 0.72, 0.86), label="漫水"), Patch(color=(0.8, 0.3, 0.25), label="崖缘退让带"),
         Patch(color=np.array(SKY_RGB) / 255, label="虚空（岛外）")]
    if P:
        h += [Patch(color="#4a3b30", label="正房"), Patch(color="#6a5646", label="厢房 / 住房"), Patch(color="#8b7763", label="倒座 / 杂屋"),
              Patch(color="#2e241c", label="门楼"), Patch(color="#9c2f25", label="庙"), Patch(color="#b8433a", label="小庙"),
              Patch(color="#4f6277", label="泊场货棚"), Patch(facecolor=YARD_COLOR, edgecolor=WALL_COLOR, label="院子与墙"),
              Patch(color=ROAD_COLOR[1], label="路"), Patch(color=BRIDGE_COLOR, label="桥"), Patch(color=POND_COLOR, label="塘"),
              Patch(color=THRESH_COLOR, label="场院"), Patch(color=LANDING_COLOR, label="泊场"), Patch(color=WELL_COLOR, label="井"),
              Patch(color=TREE_COLOR, label="树"), Patch(color="#e8b86a", label="房的正面")]
    return h


def _info_lines(meta: dict, stats: dict | None, P: dict | None, step: float, res: float) -> list[str]:
    lat = meta.get("lat_deg", 0)
    lines = [f"纬度 {lat:.1f}°（{'北' if lat >= 0 else '南'}半球，朝阳 = 朝{'南' if lat >= 0 else '北'}）", f"等高线 {step:g} m · 格 {res:g} m"]
    if stats:
        if stats.get("height_m"):
            lines.append(f"高程 {stats['height_m'][0]:.0f}–{stats['height_m'][1]:.0f} m")
        if stats.get("slope_deg"):
            lines.append(f"坡度 中位 {stats['slope_deg']['p50']:.1f}° · P90 {stats['slope_deg']['p90']:.1f}°")
        lines.append(f"田 {stats['farmland_share']:.0%} · 漫水 {stats['flood_share']:.0%} · 虚空 {stats['sky_share']:.0%}")
    w = meta.get("winter") or {}
    if w.get("from_deg") is not None:
        lines.append(f"冬季风从 {w['from_deg']:.0f}° 来")
    if P:
        m = P["metrics"]
        lines.append("")
        lines.append(f"宅院 {int(m.get('compounds', 0))} · 房 {int(m.get('buildings', 0))} · 户 {int(m.get('households_placed', 0))}/{int(m.get('households', 0))}")
        if "lambda" in m:
            lines.append(f"λ {m['lambda']:.2f} · S {m.get('shape_index', float('nan')):.2f} · 覆盖率 {m.get('coverage', float('nan')):.0%}")
        if "orient_sun_share" in m:
            lines.append(f"朝阳 {m['orient_sun_share']:.0%}（平均偏 {m.get('orient_dev_mean_deg', 0):.1f}°）")
        if "clark_evans" in m:
            lines.append(f"Clark–Evans R {m['clark_evans']:.2f} · 密度 {m.get('density_hh_per_ha', 0):.0f} 户/ha")
        lines.append(f"井 {int(m.get('wells', 0))} 口（中位 {m.get('well_dist_median_m', float('nan')):.0f} m）")
        lines.append("")
        for c in P["checks"]:
            mark = "○" if c["ok"] else ("×" if c["hard"] else "△")   # 雅黑里没有 ✓ ✗
            lines.append(f"{mark} {c['id']}")
    return lines


def _render(out_png: Path, sd: dict, meta: dict, cfg: dict, stats: dict | None, P: dict | None, style_name: str | None,
            extent: tuple[float, float, float, float] | None, max_px: int, detail: bool) -> Path:
    res, H, W = sd["res_m"], sd["H"], sd["W"]
    if extent is None:
        i0, i1, j0, j1 = 0, H, 0, W
    else:
        x0, x1, y0, y1 = extent
        j0 = max(0, int(math.floor((x0 - sd["x0"]) / res)))
        j1 = min(W, int(math.ceil((x1 - sd["x0"]) / res)))
        i0 = max(0, int(math.floor((sd["y0"] - y1) / res)))
        i1 = min(H, int(math.ceil((sd["y0"] - y0) / res)))
    st = _stride(i1 - i0, j1 - j0, max_px)
    img = ground_rgb(sd, st, (i0, i1, j0, j1), fade=0.25 if P else 0.0)
    ext = [sd["x0"] + j0 * res, sd["x0"] + j1 * res, sd["y0"] - i1 * res, sd["y0"] - i0 * res]
    # 画幅：图像按目标像素放大（细节图每米至少 3 像素）
    span_x, span_y = ext[1] - ext[0], ext[3] - ext[2]
    px = min(max_px, max(900, int(span_x * (3.0 if detail else 1.0) / 1.0)))
    py = int(px * span_y / span_x)
    fig = plt.figure(figsize=(px / 100.0 + 2.8, py / 100.0 + 0.9), dpi=100)
    axw = px / (px + 280.0)
    ax = fig.add_axes([0.02, 0.04, axw, 0.9])
    ax.imshow(img, extent=ext, origin="upper", interpolation="nearest" if not detail else "bilinear", zorder=0)
    h = sd["height"][i0:i1:st, j0:j1:st].astype(np.float64)
    h[sd["water"][i0:i1:st, j0:j1:st] > 0] = np.nan          # 水下不画等高线
    step = max(_contour_step(h), float(cfg["render"]["contour_m"]))
    fin = h[np.isfinite(h)]
    if fin.size and fin.max() - fin.min() > step:
        levels = np.arange(math.ceil(fin.min() / step) * step, fin.max(), step)
        ys = sd["y0"] - (i0 + np.arange(h.shape[0]) * st + 0.5 * st) * res
        xs = sd["x0"] + (j0 + np.arange(h.shape[1]) * st + 0.5 * st) * res
        ax.contour(xs, ys, np.ma.masked_invalid(h), levels=levels, colors="k", linewidths=0.35, alpha=0.3, zorder=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_title(_title(meta, P, style_name) + ("（局部）" if detail else ""), fontsize=11)
    if P:
        fig.canvas.draw()
        bb = ax.get_window_extent()
        pt_per_m = bb.width / span_x * 72.0 / fig.dpi
        draw_plan(ax, P, pt_per_m, labels=True, label_size=8 if detail else 6.5)
    # 比例尺与指北
    bar = 10 ** math.floor(math.log10(span_x / 4))
    bar = bar * (5 if span_x / bar > 20 else 2 if span_x / bar > 8 else 1)
    bx, by = ext[0] + 0.04 * span_x, ext[2] + 0.04 * span_y
    ax.plot([bx, bx + bar], [by, by], color="w", lw=3, solid_capstyle="butt", zorder=20)
    ax.plot([bx, bx + bar], [by, by], color="k", lw=1.2, solid_capstyle="butt", zorder=21)
    ax.text(bx + bar / 2, by + 0.012 * span_y, f"{bar:g} m", ha="center", va="bottom", fontsize=8, zorder=21,
            bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1})
    ax.annotate("北", xy=(ext[1] - 0.05 * span_x, ext[3] - 0.04 * span_y), xytext=(ext[1] - 0.05 * span_x, ext[3] - 0.12 * span_y),
                ha="center", fontsize=9, arrowprops={"arrowstyle": "-|>", "color": "k"}, bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1},
                zorder=21)
    ax.legend(handles=_legend_handles(P), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, frameon=False)
    fig.text(axw + 0.03, 0.05, "\n".join(_info_lines(meta, stats, P, step, res)), fontsize=7.5, va="bottom")
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def write_plan_png(out: Path, sd: dict, meta: dict, cfg: dict, stats: dict | None = None, P: dict | None = None,
                   style_name: str | None = None) -> Path:
    """plan.png：整个窗口；有方案时另出 plan-detail.png（建成区放大）。"""
    p = _render(out / "plan.png", sd, meta, cfg, stats, P, style_name, None, int(cfg["render"]["max_px"]), False)
    if P and P["compounds"]:
        ext = plan_extent(P)
        if ext:
            _render(out / "plan-detail.png", sd, meta, cfg, stats, P, style_name, ext, int(cfg["render"].get("detail_max_px", 2400)), True)
    return p


# ---------------------------------------------------------------- plan.svg
def write_plan_svg(out: Path, sd: dict, meta: dict, P: dict, style_name: str | None = None, margin: float = 80.0) -> Path:
    """矢量方案图：viewBox 按米，建成区外扩 margin；底图嵌一张地面 PNG；每栋房带 <title>（名字、角色、功能、尺寸、台基）。"""
    ext = plan_extent(P, margin) or (sd["x0"], sd["x0"] + sd["W"] * sd["res_m"], sd["y0"] - sd["H"] * sd["res_m"], sd["y0"])
    x0, x1, y0, y1 = ext
    res = sd["res_m"]
    j0, j1 = max(0, int((x0 - sd["x0"]) / res)), min(sd["W"], int(math.ceil((x1 - sd["x0"]) / res)))
    i0, i1 = max(0, int((sd["y0"] - y1) / res)), min(sd["H"], int(math.ceil((sd["y0"] - y0) / res)))
    x0, x1 = sd["x0"] + j0 * res, sd["x0"] + j1 * res
    y1, y0 = sd["y0"] - i0 * res, sd["y0"] - i1 * res
    img = ground_rgb(sd, 1, (i0, i1, j0, j1), fade=0.25)
    buf = io.BytesIO()
    plt.imsave(buf, img, format="png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    X = lambda x: x - x0  # noqa: E731
    Y = lambda y: y1 - y  # noqa: E731

    def pts(poly) -> str:
        return " ".join(f"{X(p[0]):.2f},{Y(p[1]):.2f}" for p in poly)

    def esc(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    Wm, Hm = x1 - x0, y1 - y0
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wm:.2f} {Hm:.2f}" width="{Wm * 4:.0f}" height="{Hm * 4:.0f}">',
         f"<title>{esc(_title(meta, P, style_name))}</title>",
         f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="{Wm:.2f}" height="{Hm:.2f}" preserveAspectRatio="none" style="image-rendering:pixelated"/>',
         '<g id="features">']
    for f in P["features"]:
        if f["kind"] in ("threshing", "landing", "pond"):
            fill, edge = {"threshing": (THRESH_COLOR, THRESH_EDGE), "landing": (LANDING_COLOR, LANDING_EDGE), "pond": (POND_COLOR, POND_EDGE)}[f["kind"]]
            o.append(f'<polygon points="{pts(f["poly"])}" fill="{fill}" stroke="{edge}" stroke-width="0.4"><title>{esc(f["name"])}</title></polygon>')
    o.append('</g><g id="roads" fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for r in sorted(P["roads"], key=lambda r: -r["cls"]):
        o.append(f'<polyline points="{pts(r["line"])}" stroke="#a8966c" stroke-width="{r["width_m"] + 0.4:.2f}"/>')
    for r in sorted(P["roads"], key=lambda r: -r["cls"]):
        o.append(f'<polyline points="{pts(r["line"])}" stroke="{ROAD_COLOR.get(r["cls"], "#e8dcba")}" stroke-width="{r["width_m"]:.2f}"/>')
    for b in P["bridges"]:
        o.append(f'<line x1="{X(b["a"][0]):.2f}" y1="{Y(b["a"][1]):.2f}" x2="{X(b["b"][0]):.2f}" y2="{Y(b["b"][1]):.2f}" '
                 f'stroke="{BRIDGE_COLOR}" stroke-width="{b["width_m"] + 1:.2f}"/>')
    o.append('</g><g id="compounds">')
    for c in P["compounds"]:
        q = _obb_poly(c["plot"]["c"], c["plot"]["facing_deg"], c["plot"]["w"], c["plot"]["d"])
        t = f'{c["template_name"]}' + (f'（{c["name"]}）' if c["name"] else "") + f'，住 {len(c["households"])} 户，台基 {c["base_m"]:.1f} m'
        o.append(f'<polygon points="{pts(q)}" fill="{YARD_COLOR}" stroke="{WALL_COLOR if c["walled"] else "#b7a98d"}" stroke-width="0.4">'
                 f"<title>{esc(t)}</title></polygon>")
    o.append('</g><g id="trees">')
    for f in P["features"]:
        if f["kind"] == "tree":
            o.append(f'<circle cx="{X(f["p"][0]):.2f}" cy="{Y(f["p"][1]):.2f}" r="{f["r"]:.2f}" fill="{TREE_COLOR}" fill-opacity="0.8"><title>{esc(f["name"])}</title></circle>')
    o.append('</g><g id="buildings" stroke="#1e1813" stroke-width="0.08">')
    for b in P["buildings"]:
        q = _obb_poly(b["c"], b["facing_deg"], b["w"], b["d"])
        t = f'{b["name"]}（{b["role"]} / {b["func"]}）{b["w"]:.1f} × {b["d"]:.1f} m，朝 {b["facing_deg"] % 360:.0f}°，{b["storeys"]} 层，屋顶 {b["roof"]}，台基 {b["base_m"]:.2f} m'
        o.append(f'<polygon points="{pts(q)}" fill="{building_color(b)}"><title>{esc(t)}</title></polygon>')
        o.append(f'<line x1="{X(q[0][0]):.2f}" y1="{Y(q[0][1]):.2f}" x2="{X(q[1][0]):.2f}" y2="{Y(q[1][1]):.2f}" stroke="#e8b86a" stroke-width="0.35"/>')
    o.append('</g><g id="points">')
    for f in P["features"]:
        if f["kind"] == "well":
            o.append(f'<circle cx="{X(f["p"][0]):.2f}" cy="{Y(f["p"][1]):.2f}" r="1.2" fill="{WELL_COLOR}" stroke="white" stroke-width="0.3"><title>{esc(f["name"])}</title></circle>')
    o.append('</g><g id="labels" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="4" text-anchor="middle">')
    for name, p, up in _labels(P):
        o.append(f'<text x="{X(p[0]):.2f}" y="{Y(p[1] + up) + (0.0 if up else 1.4):.2f}" paint-order="stroke" stroke="white" stroke-width="0.8">{esc(name)}</text>')
    o.append("</g></svg>")
    path = out / "plan.svg"
    path.write_text("\n".join(o), encoding="utf-8")
    return path
