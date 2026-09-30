"""5.6 输出：island.json、height.png（16 位）、terrain.npz、preview.png（总览）。后续步骤再加 landcover / water / farmland / climate。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap  # noqa: E402

from .grid import write_png16, write_png8  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans"]  # Windows 前两个，Linux 后三个
plt.rcParams["axes.unicode_minus"] = False

RIVER_CMAP = ListedColormap([(0.25, 0.5, 1.0), (0.1, 0.32, 0.9), (0.03, 0.15, 0.7)])   # 小 / 中 / 大河（出图用，深蓝，压得住可耕地的黄）
LANDCOVER_CLASSES = ["虚空", "崖缘", "裸岩", "高山草甸", "林地", "灌丛", "草坡", "可耕地", "梯田", "湿地", "河道", "湖"]
LANDCOVER_PALETTE = [(20, 24, 40), (90, 80, 75), (150, 150, 150), (170, 200, 120), (40, 110, 50), (120, 150, 70),
                     (190, 200, 110), (230, 200, 90), (210, 170, 60), (90, 160, 150), (40, 90, 200), (30, 60, 170)]


def _smooth_line(P: np.ndarray, n: int = 4) -> np.ndarray:
    """拉普拉斯松弛（首尾不动）+ 两次 Chaikin 切角：抹掉 D8 的 45° 台阶。P = [[行, 列, 宽(, 级别)], ...]。与调试台同口径：
    切角新点的级别取两端较大者（第 4 列插值后向上取整）。"""
    Q = P.astype(float).copy()
    for _ in range(n):
        if Q.shape[0] > 2:
            Q[1:-1, :2] = 0.25 * Q[:-2, :2] + 0.5 * Q[1:-1, :2] + 0.25 * Q[2:, :2]
    for _ in range(2):
        if Q.shape[0] < 3:
            break
        a, b = Q[:-1], Q[1:]
        mid = np.empty((2 * a.shape[0], Q.shape[1]))
        mid[0::2] = 0.75 * a + 0.25 * b
        mid[1::2] = 0.25 * a + 0.75 * b
        Q = np.vstack([Q[:1], mid, Q[-1:]])
    if Q.shape[1] > 3:
        Q[:, 3] = np.ceil(Q[:, 3] - 1e-9)
    return Q


def river_min_width(w_m: np.ndarray, lvl: np.ndarray) -> np.ndarray:
    """线宽下限（points）随河宽连续变化：2.5 m 的源流 0.3 → 百米宽的干流 1.3，中 / 大河再加 0.25 / 0.5——河从源头渐粗，不在 25 km² 阈值处钝头冒出。"""
    t = np.clip(np.log(np.maximum(w_m, 2.5) / 2.5) / math.log(40.0), 0.0, 1.0)
    return 0.3 + 1.0 * t + 0.25 * np.maximum(0, lvl - 1)


def _draw_rivers(ax, g: dict, to_xy, cell_px: float, sel_island: int | None = None, streams: bool = True) -> None:
    """河道矢量（rivers 中心线）：平滑折线，每段按自己的级别着色（干流线从源头的溪涧段起算，源流段是溪涧色），
    线宽 = 河宽 × 显示比例（points），下限随河宽渐变。to_xy(行, 列) → 数据坐标；cell_px = 一格在图上多少 points。"""
    from matplotlib.collections import LineCollection
    lines = g.get("river_lines")
    if not lines:
        return
    res_m = g["json"]["raster"]["res_m"]
    cols = np.array([(0.47, 0.67, 1.0, 0.55), (0.25, 0.5, 0.91, 1.0), (0.16, 0.39, 0.85, 1.0), (0.09, 0.28, 0.75, 1.0)])
    for want_stream in ((True, False) if streams else (False,)):
        segs, lws, cs = [], [], []
        for L in lines:
            if sel_island is not None and L["island"] != sel_island:
                continue
            P = np.array(L["pts"], dtype=float)
            if (int(P[:, 3].max()) == 0) != want_stream:
                continue
            Q = _smooth_line(P)
            x, y = to_xy(Q[:, 0], Q[:, 1])
            xy = np.stack([x, y], axis=1)
            w_m = 0.5 * (Q[:-1, 2] + Q[1:, 2])
            lv = Q[:-1, 3].astype(int)
            segs.extend(np.stack([xy[:-1], xy[1:]], axis=1))
            lws.extend(np.maximum(w_m / res_m * cell_px, river_min_width(w_m, lv)).tolist())
            cs.extend(cols[np.clip(lv, 0, 3)].tolist())
        if segs:
            ax.add_collection(LineCollection(segs, linewidths=lws, colors=cs, capstyle="round", joinstyle="round", zorder=4))


def _cell_points(ax, n_cols: float) -> float:
    """当前轴上一格多少 points（按轴宽 / 显示的列数）。"""
    fig = ax.figure
    w_in = ax.get_position().width * fig.get_figwidth()
    return w_in * 72.0 / max(1.0, n_cols)


def write_terrain(out: Path, g: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h = g["height"]
    hv = np.where(np.isnan(h), 0.0, h)
    scale = 65535.0 / max(1.0, float(np.nanmax(h)) * 1.02)
    write_png16(out / "height.png", np.round(hv * scale))
    g["json"]["raster"]["height_png_scale_m_per_unit"] = round(1.0 / scale, 6)
    arrays = {"height": h.astype(np.float32), "island_id": g["island_id"].astype(np.int16), "cliff": g["cliff"]}
    # P5：可耕地拆成 cultivable（宜垦 0 / 1 / 2 要修梯田）、cultivated（已垦、在种：0 / 1 田 / 2 梯田）、fallow_years（撂荒了几年，0 = 不是）；
    # 旧的 arable（按额度画死的可耕地）不再写（含义变了就改键名）；rain_mm = 局地年降水（P4）；runoff_mm = 年径流深（Budyko，A5）；polder_id = 圩号（P6，0 = 不是圩田；圩田也算已垦）；
    # P6b：landcover_natural = 没有人以前的地表（码同 landcover），landuse = 人工改造（waterworks.LANDUSE_CLASSES：0 没动过 … 7 采场）
    # B2–B4（DESIGN-NOTES 四点四十六）：lith = 露出的岩性（landforms.LITH_NAMES），rockwall_m / rockwall_dir = 崖层（落差、朝向 16 向，255 = 无），
    # coast_dist_m = 亚格岸距（陆地为正、虚空为负，岸线 = 岛形连续场的零等值线）
    # C1 / C2（DESIGN-NOTES 四点四十七）：河道格的 height 是平岸水面（= 谷底的滩面），bed_m = 河床（其余 NaN），river_water = 河宽够一格、记成水面的河道格；
    # floor_w_m / confine = 河道格上的谷底宽（m）与限制度（1 峡谷 / 2 半限制 / 3 开阔）；floodplain = 谷底里的岸上格；
    # C4 / C5：condense_mm = 集水核的凝结水（只进水账，runoff_mm 含它、rain_mm 不含），cloud_forest = 云雾林，recharge_mm = 地下水补给
    for k in ("flowacc_km2", "river", "lake", "landcover", "cultivable", "cultivated", "fallow_years", "slope_deg", "stream", "river_width_m",
              "river_depth_m", "floodplain", "terrain_zone", "resource", "res_field", "patch_id", "rain_mm", "runoff_mm", "polder_id", "landcover_natural", "landuse",
              "lith", "rockwall_m", "rockwall_dir", "coast_dist_m", "bed_m", "river_water", "floor_w_m", "confine", "condense_mm", "cloud_forest", "recharge_mm"):
        if k in g:
            arrays[k] = g[k]
    np.savez_compressed(out / "terrain.npz", **arrays)
    if "lith" in g:
        from .landforms import LITH_PALETTE
        write_png8(out / "lith.png", g["lith"], LITH_PALETTE)
    if "landcover" in g:
        write_png8(out / "landcover.png", g["landcover"], LANDCOVER_PALETTE)
        water = np.zeros_like(g["landcover"], dtype=np.uint8)
        water[g["stream"] > 0] = 1
        water[g["river"] > 0] = 1 + g["river"][g["river"] > 0]
        if "floodplain" in g:
            water[g["floodplain"]] = 6
        water[g["lake"]] = 5
        write_png8(out / "water.png", water, [(0, 0, 0), (120, 170, 255), (80, 130, 240), (50, 100, 220), (20, 70, 200), (30, 60, 170), (150, 200, 190)])
        if "cultivable" in g:
            write_png8(out / "farmland.png", farmland_codes(g), FARMLAND_PALETTE)
    if "river_lines" in g:
        # 河道中心线（矢量）：点 = [行, 列, 河宽 m, 级别 0 溪涧 / 1–3 小中大河, 汇流 km²]，群栅格坐标（格心 = 整数 + 0.5）
        doc = {"note": "河道中心线：每条从源头顺流到汇流点或出口；点 = [行, 列, 河宽 m, 级别（0 = 季节性溪涧）, 汇流 km²]，群栅格坐标",
               "res_m": g["json"]["raster"]["res_m"], "lines": g["river_lines"]}
        if "rivernet" in g:
            from .rivernet import rivers_doc
            doc.update(rivers_doc(g))
        if "springline" in g:
            doc["springline_note"] = ("崖壁泉线（C5）：没进河道、顺流向走到岸边的地下水，从崖壁上岩层与浮石的交界渗出；按段记：岛号、代表格 [行, 列]、"
                                      "出水 L/s、渗出处的高程（骨架顶面，m）、段长 km、是否成挂在崖壁上的泉瀑")
            doc["springline"] = [[int(s["island"]), [int(x) for x in s["cell"]], round(float(s["q_ls"]), 2), round(float(s["height_m"]), 0),
                                  round(float(s["length_km"]), 2), bool(s["fall"])] for s in g["springline"]]
        (out / "rivers.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "island.json").write_text(json.dumps(g["json"], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


FARMLAND_CLASSES = ["无", "宜垦（没开）", "已垦的田", "已垦的梯田", "撂荒", "已垦的圩田"]
FARMLAND_PALETTE = [(0, 0, 0), (120, 130, 80), (240, 205, 80), (210, 150, 40), (170, 110, 90), (110, 200, 170)]


def farmland_codes(g: dict) -> np.ndarray:
    """farmland.png 的索引（P5）：1 宜垦没开 / 2 已垦的田 / 3 已垦的梯田 / 4 撂荒；P6：5 已垦的圩田（排干的湿地）。"""
    out = np.where(g["cultivable"] > 0, 1, 0).astype(np.uint8)
    if "cultivated" in g:
        out[g["cultivated"] == 1] = 2
        out[g["cultivated"] == 2] = 3
        out[g["fallow_years"] > 0] = 4
    if "polder_id" in g:
        out[g["polder_id"] > 0] = 5
    return out


def _fields_mask(g: dict) -> np.ndarray | None:
    """出图画的田：已垦（在种）；还没跑聚落时是上等地（按额度取的，地表也照它画）。"""
    if "cultivated" in g:
        return g["cultivated"] > 0
    return g["arable"] > 0 if "arable" in g else None


def _auto_exag(h: np.ndarray, res_m: float) -> float:
    """竖直夸张：让坡度 P90 在图上约 30°（低缓的大岛也看得出沟谷）。"""
    hv = np.where(np.isnan(h), np.nan, h)
    gy, gx = np.gradient(hv, res_m)
    s = np.hypot(gx, gy)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return 1.0
    return float(np.clip(0.577 / max(np.quantile(s, 0.9), 1e-6), 1.0, 40.0))


def _h32(g: dict) -> np.ndarray:
    """出图用的高程：float32 口径（与 terrain.npz 同）。C++ 与当年的 Python 参考版、Windows 与 Linux 的 float64 高程末位（1e-12 m）有时不同，晕渲的 8 位颜色在渠、河的半透明叠色下偶尔差一个灰度（四点四十）。"""
    return g["height"].astype(np.float32).astype(np.float64)


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
    """总览主图：晕渲 + 岛号 + 短渡 + 水系 + 已垦的田 + 聚落（P7：航船线、大泊场、中转站、邑治）。"""
    J = g["json"]
    r = J["raster"]
    res_m = r["res_m"]
    H, W = g["height"].shape
    x0, y0 = r["origin_km"]
    extent = [x0, x0 + W * res_m / 1000.0, y0 - H * res_m / 1000.0, y0]
    h = _h32(g)
    vmin, vmax = float(np.nanmin(h)), float(np.nanmax(h))
    ax.imshow(_hillshade_rgb(h, res_m, vmin, vmax), extent=extent, origin="upper", interpolation="nearest")
    fm = _fields_mask(g)
    if fm is not None:
        ar = np.where(fm, 1.0, np.nan)
        ax.imshow(ar, extent=extent, origin="upper", cmap="autumn", alpha=0.55, vmin=0, vmax=1, interpolation="nearest")
    if "river" in g:
        if "river_lines" in g:
            rk = res_m / 1000.0
            _draw_rivers(ax, g, lambda i, j: (x0 + j * rk, y0 - i * rk), _cell_points(ax, W), streams=False)
        else:
            rv = np.where(g["river"] > 0, g["river"].astype(float), np.nan)
            ax.imshow(rv, extent=extent, origin="upper", cmap=RIVER_CMAP, alpha=0.95, vmin=0.5, vmax=3.5, interpolation="nearest")
        lk = np.where(g["lake"], 1.0, np.nan)
        ax.imshow(lk, extent=extent, origin="upper", cmap="winter", alpha=0.9, vmin=0, vmax=1, interpolation="nearest")
    if "settle" in g:
        S = g["settle"]
        vx = [r["km"][0] for r in S["villages"]]; vy = [r["km"][1] for r in S["villages"]]
        vs = [6 + 0.25 * r["households"] for r in S["villages"]]
        ax.scatter(vx, vy, s=vs, c="white", edgecolors="black", linewidths=0.5, zorder=5)
        ax.scatter([r["km"][0] for r in S["hamlets"]], [r["km"][1] for r in S["hamlets"]], s=5, c="#ffc8c8", edgecolors="black", linewidths=0.3, zorder=5)
        for ln in S.get("boat_lines", []):                  # P7：航船线（从最远的村开到镇上的泊场）
            P = np.array(ln["pts_km"], dtype=float)
            ax.plot(P[:, 0], P[:, 1], color="#40e0ff", lw=0.6, alpha=0.8, zorder=4)
        HB = S.get("harbors", [])
        if HB:
            ax.scatter([h["km"][0] for h in HB], [h["km"][1] for h in HB], s=7, marker="s", c=["#40e0ff" if h.get("town") else "#6a8aa0" for h in HB],
                       linewidths=0, zorder=5)
        T = S.get("towns", [])
        if T:
            ax.scatter([t["km"][0] for t in T], [t["km"][1] for t in T], s=[30 + 0.4 * t["households"] for t in T], facecolors="none",
                       edgecolors="#ff9628", linewidths=1.6, zorder=6)
            seat = [t for t in T if t.get("seat")]
            if seat:
                ax.scatter([seat[0]["km"][0]], [seat[0]["km"][1]], s=90, marker="*", c="#ffe040", edgecolors="black", linewidths=0.5, zorder=7)
        RL = S.get("relays", [])
        if RL:
            ax.scatter([r["km"][0] for r in RL], [r["km"][1] for r in RL], s=34, marker="^", c=["#ff4040" if r["households"] > 0 else "#ffa0a0" for r in RL],
                       edgecolors="black", linewidths=0.4, zorder=7)
        X = S.get("specials", [])
        if X:
            ax.scatter([x["km"][0] for x in X], [x["km"][1] for x in X], s=18, marker="D", c="#c85adc", edgecolors="black", linewidths=0.4, zorder=6)
    isl = J["islands"]
    cx = {i["id"]: i["center_km"] for i in isl}
    for e in J["links"]:
        a, b = cx[e["a"]], cx[e["b"]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#ffffff", lw=0.7, ls="--", alpha=0.6)
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
        sub = _h32(g)[max(0, r0):r0 + mm, max(0, c0):c0 + mm]
        sub_id = g["island_id"][max(0, r0):r0 + mm, max(0, c0):c0 + mm]
        sub = np.where(sub_id == 0, sub, np.nan)
        rr, cc = np.where(~np.isnan(sub))
        sub = sub[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        ax3.imshow(_hillshade_rgb(sub, res_m, float(np.nanmin(sub)), float(np.nanmax(sub)), "terrain"), interpolation="nearest")
        ax3.contour(np.where(np.isnan(sub), np.nanmin(sub), sub), levels=10, colors="k", linewidths=0.3, alpha=0.5)
        ax3.set_title(f"主岛放大（{J['islands'][0]['age_zh']}，台面 {J['constraints']['height_m']['actual']:.0f} m，峰 {J['islands'][0]['peak_m']:.0f} m，岸缘 {J['islands'][0]['rim_m']:.0f} m）", fontsize=9)
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
    """主岛放大图：晕渲 + 等高线 + 河 / 湖 / 溪涧 + 已垦的田 + 地表底色 + 水利（P6：渠、塘、圩田），给策划与场景美术看岛内细节。"""
    J = g["json"]
    res_m = J["raster"]["res_m"]
    r0, c0, mm, _ = J["islands"][0]["bbox_cells"]
    sl = (slice(max(0, r0), r0 + mm), slice(max(0, c0), c0 + mm))
    ids = g["island_id"][sl]
    m0 = ids == 0
    rr, cc = np.where(m0)
    sub = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    h = np.where(m0, _h32(g)[sl], np.nan)[sub]
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
        if "floodplain" in g:
            ax.imshow(np.where(g["floodplain"][sl][sub], 1.0, np.nan), cmap="GnBu", vmin=0, vmax=2, alpha=0.45, interpolation="nearest")
        if "river_lines" in g:
            # 子图像素坐标 = 群栅格 (行, 列) − 左上角偏移（imshow 的格心在整数处，格子占 [−0.5, +0.5]）
            oi, oj = sl[0].start + sub[0].start, sl[1].start + sub[1].start
            _draw_rivers(ax, g, lambda i, j: (j - oj - 0.5, i - oi - 0.5), _cell_points(ax, h.shape[1]), sel_island=0)
        else:
            ax.imshow(np.where(st > 0, 1.0, np.nan), cmap="Blues", vmin=0, vmax=2, alpha=0.5, interpolation="nearest")
            ax.imshow(np.where(rv > 0, rv, np.nan), cmap=RIVER_CMAP, vmin=0.5, vmax=3.5, alpha=1.0, interpolation="nearest")
        ax.imshow(np.where(g["lake"][sl][sub], 1.0, np.nan), cmap="winter", vmin=0, vmax=1, alpha=0.95, interpolation="nearest")
        fm = _fields_mask(g)
        if fm is not None:
            ax.contour(fm[sl][sub], levels=[0.5], colors="#ffdd33", linewidths=0.6)
        WK = (g.get("settle") or {}).get("waterworks")
        if WK:                                   # P6 水利：圩田（浅绿）、谷口的渠（青）、纵浦横塘与排水渠（蓝）、塘（蓝点）
            from matplotlib.collections import LineCollection
            oi, oj = sl[0].start + sub[0].start, sl[1].start + sub[1].start
            if "polder_id" in g:
                ax.imshow(np.where(g["polder_id"][sl][sub] > 0, 1.0, np.nan), cmap="summer", vmin=0, vmax=2, alpha=0.75, interpolation="nearest")
            for kinds, col, lw in ((("干渠",), "#0aa5a0", 0.9), (("支渠",), "#30c8c0", 0.45), (("纵浦", "横塘", "排水渠"), "#1f5fd0", 0.45)):
                segs = [np.array([[q[1] - oj - 0.5, q[0] - oi - 0.5] for q in c["pts"]]) for c in WK["canals"]
                        if c["island"] == 0 and c["kind"] in kinds and not c.get("abandoned")]
                if segs:
                    ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, zorder=5))
            segs = [np.array([[q[1] - oj - 0.5, q[0] - oi - 0.5] for q in c["pts"]]) for c in WK["canals"] if c["island"] == 0 and c.get("abandoned")]
            if segs:                             # P6b：废村旁没人管的废渠（灰虚线）
                ax.add_collection(LineCollection(segs, colors="#8a8580", linewidths=0.6, linestyles="dashed", zorder=5))
            for ab, col in ((False, "#2a7fff"), (True, "#8a8580")):
                P = [p["cell"] for p in WK["ponds"] if p["island"] == 0 and bool(p.get("abandoned")) == ab]
                if P:
                    P = np.array(P, dtype=float)
                    ax.scatter(P[:, 1] - oj, P[:, 0] - oi, s=2.5, c=col, linewidths=0, zorder=6)
    km = 10.0 * 1000.0 / res_m
    ax.plot([10, 10 + km], [h.shape[0] - 10, h.shape[0] - 10], color="w", lw=3)
    ax.text(10 + km / 2, h.shape[0] - 16, "10 km", color="w", ha="center", fontsize=9)
    i0 = J["islands"][0]
    hy = J.get("hydro", {})
    ax.set_title(f"主岛（{i0['age_zh']}，{i0['area_km2']:.0f} km²，台面 {J['constraints']['height_m']['actual']:.0f} m，峰 {i0['peak_m']:.0f} m，岸缘 {i0['rim_m']:.0f} m，崖 {i0['cliff_m']:.0f} m）"
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


SETTLE_PALETTE = [(0, 0, 0), (230, 200, 90), (210, 170, 60), (255, 255, 255), (255, 200, 200), (60, 200, 255), (0, 0, 0), (80, 120, 255), (120, 200, 255),
                  (255, 150, 40), (200, 90, 220), (170, 110, 90), (120, 90, 80), (150, 120, 200), (90, 200, 160), (40, 150, 230), (20, 40, 90),
                  (64, 224, 255), (255, 64, 64), (130, 120, 110)]
# 9 镇 / 10 专业聚落（常住）/ 11 撂荒田 / 12 废村 / 13 工棚、季节住 / 14 有人用（放牧、庙、墓岛）/ 15 塘 / 16 闸（P6）/ 17 大泊场（镇 / 邑治）/ 18 中转站（P7）；
# 19 废弃的水利（P6b：废村旁没人管的塘、渠首闸）；
# 6 原是桥头（P5 起没有索桥），空着


def write_settlements(out: Path, g: dict) -> None:
    """settlements.json + settlements.png（8 位索引：1 田块 / 2 梯田 / 3 村 / 4 散户 / 5 泊场 / 7 蓄水池 / 8 取水点 / 9 镇 / 10 专业聚落 /
    11 撂荒田 / 12 废村 / 13 工棚、季节住 / 14 有人用 / 15 塘 / 16 闸 / 17 大泊场 / 18 中转站）。渠、圩堤、航船线是线，在 settlements.json 的 waterworks / boat_lines 里。"""
    write_png8(out / "settlements.png", g["settle_raster"], SETTLE_PALETTE)
    (out / "settlements.json").write_text(json.dumps(g["settle"], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
