"""可视化：中间产物全部可单独成图（docs/12 §七：调试全靠看中间层）。

- 调试用等距圆柱投影，经度以 G 为中心（跨反经线的边拆两段）
- 四模式固定色；0–1 量 viridis；reach 用 log10 色标；公共底图（带界、A 阴影、G 圆、D 框）
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from . import MODES, MODE_ZH  # noqa: E402
from .culture import World  # noqa: E402
from .stages.s03_islands import CLASS_NAMES, CLASS_ZH  # noqa: E402

MODE_COLORS = {"daily": "#1f77b4", "trade": "#ff7f0e", "envoy": "#2ca02c", "migrate": "#d62728"}
CLASS_COLORS = {"dense": "#7b3294", "medium": "#008837", "sparse": "#d9a800", "isolated": "#c0392b"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _center_lon(ctx) -> float:
    cl = ctx.cfg.get("s10", {}).get("output", {}).get("center_lon", "auto")
    if cl == "auto":
        return float(ctx.load_json(2, "bands")["G"]["lon"])
    return float(cl)


def _recenter(lon, center):
    return ((np.asarray(lon, dtype=np.float64) - center + 180.0) % 360.0) - 180.0


def _edge_segments(x, y, src, dst, values=None):
    """把边变成线段列表，跨 ±180 的边拆两段。返回 (segments, values_expanded)。"""
    x1, y1, x2, y2 = x[src], y[src], x[dst], y[dst]
    dx = x2 - x1
    wrap = np.abs(dx) > 180.0
    segs, vals = [], []
    nw = ~wrap
    segs.append(np.stack([np.stack([x1[nw], y1[nw]], axis=1),
                          np.stack([x2[nw], y2[nw]], axis=1)], axis=1))
    if values is not None:
        vals.append(np.asarray(values)[nw])
    if wrap.any():
        s = np.sign(dx[wrap])
        for xa, xb in ((x1[wrap], x2[wrap] - 360.0 * s), (x1[wrap] + 360.0 * s, x2[wrap])):
            segs.append(np.stack([np.stack([xa, y1[wrap]], axis=1),
                                  np.stack([xb, y2[wrap]], axis=1)], axis=1))
            if values is not None:
                vals.append(np.asarray(values)[wrap])
    segments = np.concatenate(segs, axis=0)
    return segments, (np.concatenate(vals) if values is not None else None)


def _basemap(ax, ctx, center):
    bands = ctx.load_json(2, "bands")
    planet = ctx.load_json(1, "planet")["bands"]
    sk = ctx.cfg["skeleton"]
    for k in ("eq_storm_top_deg", "trades_top_deg", "calm_top_deg", "westerlies_top_deg"):
        for sgn in (1, -1):
            ax.axhline(sgn * planet[k], color="0.75", lw=0.5, ls="--", zorder=0)
    core = float(sk["eq_core_halfwidth_deg"])
    ax.axhspan(-core, core, color="#f2c9c9", alpha=0.5, zorder=0)
    g = bands["G"]
    gx = _recenter(g["lon"], center)
    th = np.linspace(0, 2 * np.pi, 64)
    r = g["radius_deg"]
    ax.plot(gx + r / np.cos(np.radians(g["lat"])) * np.cos(th),
            g["lat"] + r * np.sin(th), color="#8e2d2d", lw=1.2, zorder=5)
    lon_w = _recenter(sk["d_lon_west"], center)
    lon_e = _recenter(sk["d_lon_east"], center)
    from .skeleton import d_lat_range
    d_lo, d_hi = d_lat_range(ctx.cfg, ctx.load_json(1, "planet"))
    for yy in (d_lo, d_hi):
        ax.plot([lon_w, lon_e], [yy, yy], color="#c58bc5", lw=0.8, ls=":", zorder=1)
    for xx in (lon_w, lon_e):
        ax.plot([xx, xx], [d_lo, d_hi],
                color="#c58bc5", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("经度（以 G 为中心）")
    ax.set_ylabel("纬度")


def _stamp(fig, ctx, title):
    seed = ctx.seed
    fig.suptitle(f"{title}   [seed {seed} · run {ctx.out_dir.name}]", fontsize=11)


def _save(fig, ctx, name, show=False):
    d = ctx.stage_dir(10) / "fig"
    d.mkdir(parents=True, exist_ok=True)
    dpi = int(ctx.cfg.get("s10", {}).get("output", {}).get("dpi", 130))
    fig.savefig(d / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return d / f"{name}.png"


# ---------------------------------------------------------------- layers
def viz_wind(ctx, show=False):
    try:
        w = ctx.load_npz(4, "wind_local")     # 扰动后的风（第三批 3）；旧产物退回 ②
    except FileNotFoundError:
        w = ctx.load_npz(2, "wind")
    center = _center_lon(ctx)
    fig, ax = plt.subplots(figsize=(14, 7))
    lats, lons = w["lats"], w["lons"]
    xs = _recenter(lons, center)
    order = np.argsort(xs)
    speed = np.hypot(w["u"], w["v"])
    im = ax.pcolormesh(xs[order], lats, speed[:, order], cmap="Blues", shading="auto", zorder=0)
    step = max(1, lats.size // 30)
    LON, LAT = np.meshgrid(xs[order][::step * 2], lats[::step])
    ax.quiver(LON, LAT, w["u"][::step, order][:, ::step * 2], w["v"][::step, order][:, ::step * 2],
              color="0.25", width=0.0012, zorder=2)
    _basemap(ax, ctx, center)
    fig.colorbar(im, ax=ax, label="风速 m/s", shrink=0.8)
    _stamp(fig, ctx, "② 行星风系 + ②b 岛群扰动（波状带界、摩擦与尾流）+ 定点永暴 G")
    return _save(fig, ctx, "s02_wind", show)


def _area_marker(area, lo=1.0, hi=14.0):
    """点大小 ∝ √面积，在 p2/p98 处夹断（对数正态的尾巴会把 98% 的点压到最小端）。"""
    s = np.sqrt(np.maximum(np.asarray(area, dtype=np.float64), 1e-9))
    a, b = np.quantile(s, 0.02), np.quantile(s, 0.98)
    return lo + (hi - lo) * np.clip((s - a) / max(1e-9, b - a), 0.0, 1.0)


def viz_scale(ctx, show=False):
    """岛群规模：群陆地与集雨容量（docs/02 §六 一群 = 一水共同体 = 一个基本政治单位）。"""
    isl = ctx.load_npz(3, "islands")
    clim = ctx.load_npz(4, "climate_islands")
    center = _center_lon(ctx)
    x = _recenter(isl["lon"], center)
    y = isl["lat"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    for ax, val, title in (
            (axes[0], isl["area_km2"], "岛群陆地 km²（log10）"),
            (axes[1], clim["catch"], "集雨容量 = 可用地率 × 陆地 × 降水（log10）")):
        v = np.log10(np.maximum(np.asarray(val, dtype=np.float64), 1e-6))
        sc = ax.scatter(x, y, s=3, c=v, cmap="viridis", linewidths=0)
        _basemap(ax, ctx, center)
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.8)
    med = float(np.median(isl["area_km2"]))
    mx = float(np.max(isl["area_km2"]))
    _stamp(fig, ctx, f"③④ 岛群规模（群陆地中位 {med:.0f} km²，最大 {mx:.0f} km²）")
    return _save(fig, ctx, "s04_scale", show)


def viz_islands(ctx, show=False):
    isl = ctx.load_npz(3, "islands")
    dg = ctx.load_npz(3, "density_grid")
    center = _center_lon(ctx)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax = axes[0]
    x = _recenter(isl["lon"], center)
    msize = _area_marker(isl["area_km2"])
    for ci, cname in enumerate(CLASS_NAMES):
        m = isl["cls"] == ci
        ax.scatter(x[m], isl["lat"][m], s=msize[m], c=CLASS_COLORS[cname],
                   label=f"{CLASS_ZH[cname]} ({int(m.sum())})", linewidths=0)
    lay = isl["layered"]
    ax.scatter(x[lay], isl["lat"][lay], s=8, facecolors="none", edgecolors="k",
               linewidths=0.4, label=f"叠层 ({int(lay.sum())})")
    _basemap(ax, ctx, center)
    ax.legend(loc="lower left", fontsize=8, markerscale=2)
    ax.set_title("岛群分布（按地形类；点大小 ∝ √群陆地）")
    ax2 = axes[1]
    xs = _recenter(dg["lons"], center)
    order = np.argsort(xs)
    im = ax2.pcolormesh(xs[order], dg["lats"], np.log10(np.maximum(dg["density"][:, order], 1e-4)),
                        cmap="viridis", shading="auto")
    _basemap(ax2, ctx, center)
    fig.colorbar(im, ax=ax2, label="log10 密度", shrink=0.8)
    ax2.set_title("密度场（含 D 空域与绕道岛弧）")
    _stamp(fig, ctx, "③ 岛屿分布")
    return _save(fig, ctx, "s03_islands", show)


def viz_climate(ctx, show=False):
    c = ctx.load_npz(4, "climate_grid")
    center = _center_lon(ctx)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    xs = _recenter(c["lons"], center)
    order = np.argsort(xs)
    for ax, key, cmap, label in [
            (axes[0][0], "precip", "YlGnBu", "降水（相对）"),
            (axes[0][1], "temp", "coolwarm", "温度 °C"),
            (axes[1][0], "storm", "inferno", "风暴强度"),
            (axes[1][1], "stability", "viridis", "稳定度")]:
        im = ax.pcolormesh(xs[order], c["lats"], c[key][:, order], cmap=cmap, shading="auto")
        _basemap(ax, ctx, center)
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_title(label)
    _stamp(fig, ctx, "④ 气候")
    return _save(fig, ctx, "s04_climate", show)


def viz_barriers(ctx, show=False):
    w = World(ctx)
    isl, ce, pm = w.islands, w.cand_edges, w.perm
    center = _center_lon(ctx)
    x = _recenter(isl["lon"], center)
    y = isl["lat"]
    barrier_score = 1.0 - pm["perm"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(14, 7))
    segs, vals = _edge_segments(x, y, ce["src"], ce["dst"], barrier_score)
    lc = LineCollection(segs, cmap="inferno_r", norm=plt.Normalize(0, 1), linewidths=0.6)
    lc.set_array(vals)
    ax.add_collection(lc)
    gb = pm["g_blocked"]
    if gb.any():
        segs_g, _ = _edge_segments(x, y, ce["src"][gb], ce["dst"][gb])
        ax.add_collection(LineCollection(segs_g, colors="#8e2d2d", linewidths=1.2))
    ax.scatter(x, y, s=0.8, c="0.6", zorder=1, linewidths=0)
    _basemap(ax, ctx, center)
    fig.colorbar(lc, ax=ax, label="障碍分 1 − mean_m perm", shrink=0.8)
    _stamp(fig, ctx, "⑤ 障碍（边的平均阻断度；红 = G 阻断边）")
    return _save(fig, ctx, "s05_barriers", show)


def viz_perm(ctx, mode="daily", show=False):
    w = World(ctx)
    isl, ce, pm = w.islands, w.cand_edges, w.perm
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    mi = MODES.index(mode)
    p = pm["perm"][:, mi]
    fig, ax = plt.subplots(figsize=(14, 7))
    alive = p > 0
    segs, vals = _edge_segments(x, y, ce["src"][alive], ce["dst"][alive], p[alive])
    lc = LineCollection(segs, cmap="RdYlGn", norm=plt.Normalize(0, 1), linewidths=0.6)
    lc.set_array(vals)
    ax.add_collection(lc)
    dead = ~alive
    if dead.any():
        segs_d, _ = _edge_segments(x, y, ce["src"][dead], ce["dst"][dead])
        ax.add_collection(LineCollection(segs_d, colors="k", linewidths=0.5, alpha=0.5))
    _basemap(ax, ctx, center)
    fig.colorbar(lc, ax=ax, label="通过率", shrink=0.8)
    _stamp(fig, ctx, f"⑤ 通过率 · {MODE_ZH[mode]}（黑 = 该模式不可通）")
    return _save(fig, ctx, f"s05_perm_{mode}", show)


def viz_routes(ctx, show=False):
    w = World(ctx)
    isl, ce, rt = w.islands, w.cand_edges, w.routes
    centers = w.centers
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    E = ce["src"].size
    flow_und = rt["flow"][:E] + rt["flow"][E:]
    fig, ax = plt.subplots(figsize=(14, 7))
    segs, _ = _edge_segments(x, y, ce["src"], ce["dst"])
    ax.add_collection(LineCollection(segs, colors="0.85", linewidths=0.3))
    trunk = flow_und >= np.quantile(flow_und[flow_und > 0], 0.95) if (flow_und > 0).any() else flow_und > 0
    if trunk.any():
        segs_t, vals_t = _edge_segments(x, y, ce["src"][trunk], ce["dst"][trunk],
                                        np.log10(flow_und[trunk] + 1))
        lc = LineCollection(segs_t, cmap="plasma", linewidths=1.4)
        lc.set_array(vals_t)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="log10 流量（干线）", shrink=0.8)
    for h in w.hubs["hubs"]:
        ax.plot(_recenter(h["lon"], center), h["lat"], marker="*", ms=9,
                color="#d4af37" if not h["near_g"] else "#8e2d2d", mec="k", mew=0.3, zorder=6)
    for cid, c in centers["centers"].items():
        ax.plot(_recenter(c["lon"], center), c["lat"], marker="^", ms=12, color="#2050c0",
                mec="w", zorder=7)
        ax.annotate(c["zh"], (_recenter(c["lon"], center), c["lat"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    for key, tr in centers["center_trunks"].items():
        path = tr.get("path", [])
        if len(path) > 1:
            px, py = x[np.array(path)], y[np.array(path)]
            segs_p, _ = _edge_segments(px, py, np.arange(len(path) - 1), np.arange(1, len(path)))
            ax.add_collection(LineCollection(segs_p, colors="#2050c0", linewidths=1.0,
                                             linestyles="--", alpha=0.7))
    _basemap(ax, ctx, center)
    _stamp(fig, ctx, "⑥ 航线网络（灰=候选边；彩=干线；★=枢纽，红★=G 邻域中转岛；虚线=中心间干线）")
    return _save(fig, ctx, "s06_routes", show)


def viz_centers(ctx, show=False):
    w = World(ctx)
    isl = w.islands
    reg = w.regions
    pre = ctx.load_npz(7, "prehist")
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 17))
    ax = axes[0]
    sc = ax.scatter(x, y, s=3, c=reg["suitability"], cmap="viridis", linewidths=0)
    _basemap(ax, ctx, center)
    fig.colorbar(sc, ax=ax, label="适宜度 = 降水×稳定×密度", shrink=0.8)
    for cid, c in w.centers["centers"].items():
        ax.plot(_recenter(c["lon"], center), c["lat"], marker="^", ms=12, color="#e04040",
                mec="w", zorder=7)
        ax.annotate(c["zh"], (_recenter(c["lon"], center), c["lat"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9, color="#802020")
    ax.set_title("⑦ 适宜度与三个文明中心（骨架窗内涌现）")
    ax = axes[1]
    sc = ax.scatter(x, y, s=3, c=pre["arrival_yr"], cmap="magma_r", linewidths=0)
    o = int(w.centers["origin_node"])
    ax.plot(x[o], y[o], marker="P", ms=12, color="#00c0c0", mec="k", zorder=7)
    _basemap(ax, ctx, center)
    fig.colorbar(sc, ax=ax, label="史前到达（年）", shrink=0.8)
    ax.set_title("⑦ 史前扩散（抱石而渡，顺风单向；十字标 = 起源地）")
    ax = axes[2]
    rid = reg["region"]
    sc = ax.scatter(x, y, s=3, c=rid % 20, cmap="tab20", linewidths=0)
    _basemap(ax, ctx, center)
    ax.set_title(f"⑦b 地区划分（{int(rid.max()) + 1} 区，日常权重 Voronoi —— 仅输出用，非文化边界）")
    _stamp(fig, ctx, "⑦ 文明中心 · 史前扩散 · 地区")
    return _save(fig, ctx, "s07_centers", show)


def viz_polity(ctx, show=False):
    """⑨ 政治层：诸邦（按邦着色）、都城、变法之国的本朝、宗主。"""
    from .polity import Polity
    pol = Polity(ctx)
    if not pol.available:
        return None
    isl = ctx.load_npz(3, "islands")
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    arr, meta = pol.arr, pol.meta
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax = axes[0]
    pid = arr["polity"]
    kind = arr["kind"]
    col = np.where(kind == 0, pid % 20, -1)
    sc = ax.scatter(x[kind == 0], y[kind == 0], s=3, c=col[kind == 0], cmap="tab20", linewidths=0)
    ax.scatter(x[kind == 1], y[kind == 1], s=4, c="#888888", marker="x", linewidths=0.6, label="船团（不建国）")
    ax.scatter(x[kind == 2], y[kind == 2], s=8, c="#000000", marker="s", linewidths=0, label="部落")
    caps = arr["capital"]
    ax.scatter(x[caps], y[caps], s=10, c="k", marker=".", linewidths=0, zorder=5)
    for cid, s in meta["suzerain"].items():
        if s >= 0:
            c = int(caps[s])
            ax.plot(x[c], y[c], marker="*", ms=14, color="#ffd27a", mec="k", zorder=7)
    r = meta["reformer"]["polity"]
    if r >= 0:
        c = int(caps[r])
        ax.plot(x[c], y[c], marker="^", ms=13, color="#e04040", mec="k", zorder=7)
    _basemap(ax, ctx, center)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"⑨ 诸邦（{meta['n_states']} 邦 · 着色仅为区分；黑点 = 都城；★ 宗主；▲ 变法之国）")
    ax = axes[1]
    realm = arr["realm"]
    state = arr["state"]
    mine = (state >= 0) & (realm == r) if r >= 0 else np.zeros(x.size, bool)
    ax.scatter(x[state >= 0], y[state >= 0], s=2, c="#b0b8c8", linewidths=0)
    ax.scatter(x[mine & (state == r)], y[mine & (state == r)], s=5, c="#e04040", linewidths=0, label="变法之国本邦")
    ax.scatter(x[mine & (state != r)], y[mine & (state != r)], s=5, c="#f0a040", linewidths=0, label="已并之邦")
    fr = [f["polity"] for f in meta.get("fronts", [])]
    fm = np.isin(state, fr)
    ax.scatter(x[fm], y[fm], s=5, c="#4d9de0", linewidths=0, label="当前战线")
    sz = meta["suzerain"].get(meta["reformer"]["circle"], -1)
    if sz >= 0:
        ax.scatter(x[state == sz], y[state == sz], s=5, c="#ffd27a", linewidths=0, label="宗主（正统核心）")
    _basemap(ax, ctx, center)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"⑨ 兼并史：{meta['reformer']['n_annexed']} 邦已并，本朝约 {meta['reformer']['realm_pop'] / 1e4:.0f} 万口"
                 f"（变法距今 {meta['reformer']['reform_years_ago']:.0f} 年）")
    _stamp(fig, ctx, "⑨ 政治层：诸邦 · 宗主 · 变法之国与兼并")
    return _save(fig, ctx, "s09_polity", show)


def viz_iso(ctx, show=False):
    w = World(ctx)
    isl = w.islands
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for mi, m in enumerate(MODES):
        ax = axes[mi // 2][mi % 2]
        sc = ax.scatter(x, y, s=2.5, c=w.iso["iso"][mi], cmap="viridis",
                        norm=plt.Normalize(0, 1), linewidths=0)
        _basemap(ax, ctx, center)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f"隔离度 · {MODE_ZH[m]}")
    _stamp(fig, ctx, "⑧ 隔离度（反射型：越高本地自有越强）")
    return _save(fig, ctx, "s08_iso", show)


def viz_trait(ctx, trait_id: str, show=False):
    w = World(ctx)
    t = next((t for t in w.traits if t["id"] == trait_id), None)
    if t is None:
        raise SystemExit(f"未知特征 id：{trait_id}（见 s08_diffusion/traits.resolved.json）")
    ti = t["index"]
    isl = w.islands
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    f = w.fields
    fig, axes = plt.subplots(3, 1, figsize=(14, 17))
    reach = np.maximum(f["reach"][ti].astype(np.float64), 1e-6)
    panels = [(axes[0], np.log10(reach), "log10 reach（能不能传到）", "magma", (-4, 0)),
              (axes[1], f["adopt"][ti], "adopt（传到了接不接受）", "viridis", (0, 1)),
              (axes[2], f["strength"][ti], "strength = reach × adopt", "viridis", (0, 1))]
    for ax, val, label, cmap, (vmin, vmax) in panels:
        sc = ax.scatter(x, y, s=3, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        _basemap(ax, ctx, center)
        fig.colorbar(sc, ax=ax, label=label, shrink=0.8)
        ax.set_title(label)
    o = t["origin_node"]
    for ax, *_ in panels:
        ax.plot(x[o], y[o], marker="P", ms=11, color="#00c0c0", mec="k", zorder=7)
    _stamp(fig, ctx, f"⑧ 特征场 {trait_id}（{t['slot_zh']} · {MODE_ZH[t['mode']]} · "
                     f"阻力 {t['resistance']} · 半衰 {t['d_half_days']} 天）")
    return _save(fig, ctx, f"trait_{trait_id.replace(':', '_').replace('@', '_')}", show)


def viz_slot(ctx, slot_id: str, show=False):
    w = World(ctx)
    if slot_id not in w.slots:
        raise SystemExit(f"未知槽位：{slot_id}（可选：{', '.join(w.slots)}）")
    share, labels = w.shares(slot_id)
    isl = w.islands
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    V = share.shape[0]
    ncol = 2
    nrow = (V + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.2 * nrow), squeeze=False)
    for vi in range(nrow * ncol):
        ax = axes[vi // ncol][vi % ncol]
        if vi >= V:
            ax.axis("off")
            continue
        sc = ax.scatter(x, y, s=2.5, c=share[vi], cmap="viridis",
                        norm=plt.Normalize(0, 1), linewidths=0)
        _basemap(ax, ctx, center)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(labels[vi], fontsize=9)
    mode = w.slot_mode(slot_id)
    _stamp(fig, ctx, f"⑧ 槽位比例分布 · {slot_id}（{MODE_ZH[mode]}）—— 同槽位归一化，处处连续")
    return _save(fig, ctx, f"slot_{slot_id}", show)


def viz_isogloss(ctx, by_mode=False, show=False):
    w = World(ctx)
    isl, ce = w.islands, w.cand_edges
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    theta = float(ctx.cfg["check"]["iso_theta"])
    iso = w.isogloss_edges(theta)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    ax = axes[0]
    segs_all, _ = _edge_segments(x, y, ce["src"], ce["dst"])
    ax.add_collection(LineCollection(segs_all, colors="0.9", linewidths=0.25))
    rng_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for ti in range(iso.shape[0]):
        e = iso[ti]
        if not e.any():
            continue
        color = MODE_COLORS[w.traits[ti]["mode"]] if by_mode else rng_colors[ti % 20]
        mx = 0.5 * (x[ce["src"][e]] + x[ce["dst"][e]])
        ok = np.abs(x[ce["src"][e]] - x[ce["dst"][e]]) <= 180
        my = 0.5 * (y[ce["src"][e]] + y[ce["dst"][e]])
        ax.scatter(mx[ok], my[ok], s=4, color=color, linewidths=0, alpha=0.8)
    _basemap(ax, ctx, center)
    ax.set_title(f"同言线（share 跨 {theta} 的边中点）· " + ("按模式着色" if by_mode else "按特征着色"))
    if by_mode:
        for m in MODES:
            ax.scatter([], [], color=MODE_COLORS[m], label=MODE_ZH[m])
        ax.legend(loc="lower left", fontsize=8)
    ax2 = axes[1]
    bundle = iso.sum(axis=0)
    e = bundle > 0
    segs_b, vals_b = _edge_segments(x, y, ce["src"][e], ce["dst"][e], bundle[e])
    lc = LineCollection(segs_b, cmap="hot_r", norm=plt.Normalize(0, max(2, bundle.max())),
                        linewidths=1.0)
    lc.set_array(vals_b)
    ax2.add_collection(lc)
    _basemap(ax2, ctx, center)
    fig.colorbar(lc, ax=ax2, label="聚束数 b(e)", shrink=0.8)
    ax2.set_title("同言线聚束（聚束处 = 障碍所在，docs/02 §五 推论二）")
    _stamp(fig, ctx, "⑧ 同言线")
    return _save(fig, ctx, "s08_isogloss" + ("_by_mode" if by_mode else ""), show)


def viz_distance(ctx, node: int, show=False):
    w = World(ctx)
    isl = w.islands
    center = _center_lon(ctx)
    x, y = _recenter(isl["lon"], center), isl["lat"]
    N = isl["lat"].size
    all_nodes = np.arange(N)
    home = np.full(N, int(node))
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    d_all = w.tv_distance(home, all_nodes)
    ax = axes[0][0]
    sc = ax.scatter(x, y, s=2.5, c=d_all, cmap="viridis", norm=plt.Normalize(0, 1), linewidths=0)
    _basemap(ax, ctx, center)
    ax.plot(x[node], y[node], marker="P", ms=11, color="#00c0c0", mec="k", zorder=7)
    fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title("文化距离（全部槽位 TV 均值）")
    axes[0][1].axis("off")
    for k, m in enumerate(MODES):
        ax = axes[1 + k // 2][k % 2]
        dm = w.tv_distance(home, all_nodes, mode=m)
        sc = ax.scatter(x, y, s=2.5, c=dm, cmap="viridis", norm=plt.Normalize(0, 1), linewidths=0)
        _basemap(ax, ctx, center)
        ax.plot(x[node], y[node], marker="P", ms=11, color="#00c0c0", mec="k", zorder=7)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f"文化距离 · 仅 {MODE_ZH[m]} 槽位")
    _stamp(fig, ctx, f"⑨ 从节点 {node} 出发的文化距离（「文书通、口音不通」= envoy 图与 daily 图的反差）")
    return _save(fig, ctx, f"distance_{node}", show)


# ---------------------------------------------------------------- entry
def render(ctx, layer: str, arg=None, mode=None, show=False):
    if layer == "wind":
        p = viz_wind(ctx, show)
    elif layer == "islands":
        p = viz_islands(ctx, show)
    elif layer == "climate":
        p = viz_climate(ctx, show)
    elif layer == "barriers":
        p = viz_barriers(ctx, show)
    elif layer == "perm":
        p = viz_perm(ctx, mode or arg or "daily", show)
    elif layer == "routes":
        p = viz_routes(ctx, show)
    elif layer == "centers":
        p = viz_centers(ctx, show)
    elif layer == "iso":
        p = viz_iso(ctx, show)
    elif layer == "polity":
        p = viz_polity(ctx, show)
    elif layer == "scale":
        p = viz_scale(ctx, show)
    elif layer == "trait":
        p = viz_trait(ctx, arg, show)
    elif layer == "slot":
        p = viz_slot(ctx, arg, show)
    elif layer == "isogloss":
        p = viz_isogloss(ctx, by_mode=(arg == "mode" or mode == "mode"), show=show)
    elif layer == "distance":
        p = viz_distance(ctx, int(arg), show)
    elif layer == "all":
        return render_all(ctx)
    else:
        raise SystemExit(f"未知图层：{layer}")
    print(f"已输出：{p}")
    return p


def render_all(ctx) -> int:
    n = 0
    for fn in (viz_wind, viz_islands, viz_scale, viz_climate, viz_barriers, viz_routes,
               viz_centers, viz_iso, viz_polity):
        fn(ctx)
        n += 1
    for m in MODES:
        viz_perm(ctx, m)
        n += 1
    viz_isogloss(ctx, by_mode=False)
    viz_isogloss(ctx, by_mode=True)
    n += 2
    w = World(ctx)
    for slot in w.slots:
        viz_slot(ctx, slot)
        n += 1
    for cid, c in w.centers["centers"].items():
        viz_distance(ctx, int(c["node"]))
        n += 1
    return n
