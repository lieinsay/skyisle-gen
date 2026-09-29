"""镇与航船俯视图（P7）：直接读岛群产物，画整群（或一块窗口）的晕渲 + 田 + 大泊场（按船数）、镇（按户）、邑治、航船线、村（按赶集怎么去着色）、中转站（按功能）。

读 island.json、terrain.npz、settlements.json（P7 起有 harbors / boat_lines / relays）。
用法（仓库根下）：
    PYTHONUTF8=1 python docs/probes/market_view.py <岛群目录> [--rows r0,r1 --cols c0,c1 | --seat [--pad 120]] [--out 图.png]
    例：python docs/probes/market_view.py out/seed42/islands/6329 --out out/p7/market_6329.png
--seat：以邑治为心、--pad 格为半径开窗（看街朝泊场长、线汇到邑治）。不给窗口就画整群。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FUNC_COLOR = {"换船": "#ff2020", "过夜": "#ff7a00", "候风": "#ffb000", "避风": "#b060ff", "关卡": "#e04080", "烽火": "#ffffff"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--rows")
    ap.add_argument("--cols")
    ap.add_argument("--seat", action="store_true")
    ap.add_argument("--pad", type=int, default=120)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    J = json.loads((d / "island.json").read_text(encoding="utf-8"))
    S = json.loads((d / "settlements.json").read_text(encoding="utf-8"))
    Z = np.load(d / "terrain.npz")
    if "harbors" not in S:
        print("没有 harbors / boat_lines / relays（P7 之前的产物）")
        return 1
    res_m = float(J["raster"]["res_m"])
    res_km = res_m / 1000.0
    x0, y0 = J["raster"]["origin_km"]
    iid = Z["island_id"]
    H, W = iid.shape
    T = S["towns"]
    seat = next(t for t in T if t["seat"])
    if a.seat:
        r0, r1, c0, c1 = seat["cell"][0] - a.pad, seat["cell"][0] + a.pad, seat["cell"][1] - a.pad, seat["cell"][1] + a.pad
    elif a.rows:
        r0, r1 = (int(x) for x in a.rows.split(","))
        c0, c1 = (int(x) for x in a.cols.split(","))
    else:
        rr, cc = np.nonzero(iid >= 0)
        r0, r1, c0, c1 = int(rr.min()) - 10, int(rr.max()) + 10, int(cc.min()) - 10, int(cc.max()) + 10
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H, r1), min(W, c1)
    sl = (slice(r0, r1), slice(c0, c1))
    h = np.where(iid[sl] >= 0, Z["height"][sl], np.nan)
    land = iid[sl] >= 0
    ls = LightSource(azdeg=315, altdeg=45)
    hv = np.where(land, h, np.nanmin(h))
    rgb = ls.shade(hv, cmap=plt.get_cmap("gist_earth"), vert_exag=3, dx=res_m, dy=res_m, blend_mode="soft",
                   vmin=float(np.nanmin(h)), vmax=float(np.nanmax(h)))[..., :3]
    rgb[~land] = (0.08, 0.09, 0.16)
    cult = (Z["cultivated"][sl] > 0) & land
    rgb[cult] = 0.55 * rgb[cult] + 0.45 * np.array([0.95, 0.82, 0.35])
    ext = [x0 + c0 * res_km, x0 + c1 * res_km, y0 - r1 * res_km, y0 - r0 * res_km]
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    fig, ax = plt.subplots(figsize=(13, 13 * (ext[3] - ext[2]) / max(1e-9, ext[1] - ext[0]) + 0.6))
    ax.imshow(rgb, extent=ext, origin="upper", interpolation="nearest")
    inwin = lambda k: ext[0] <= k[0] <= ext[1] and ext[2] <= k[1] <= ext[3]
    # 航船线
    cmap = plt.get_cmap("tab20")
    for ln in S["boat_lines"]:
        P = np.array(ln["pts_km"], dtype=float)
        col = cmap((ln["town"] - 1) % 20)
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.4, alpha=0.95, zorder=4)
        ax.annotate("", xy=P[-1], xytext=P[-2], arrowprops=dict(arrowstyle="->", color=col, lw=1.2), zorder=4)
    # 村：走路赶集白、搭航船青；大小按户
    for v in S["villages"]:
        if not inwin(v["km"]):
            continue
        boat = v.get("market_mode") == "航船"
        ax.scatter([v["km"][0]], [v["km"][1]], s=4 + 0.12 * v["households"], c="#40e0ff" if boat else "white", edgecolors="black", linewidths=0.3, zorder=5)
    # 大泊场：方块，面积按船数；镇挨着的实心
    HB = S["harbors"]
    for hb in HB:
        if not inwin(hb["km"]):
            continue
        ax.scatter([hb["km"][0]], [hb["km"][1]], s=6 + 0.12 * hb["ships"], marker="s", facecolors="#40a0ff" if hb.get("town") else "none",
                   edgecolors="#40a0ff", linewidths=0.8, zorder=5, alpha=0.9)
    # 镇与街（老村核心 → 大泊场）
    for t in T:
        if not inwin(t["km"]):
            continue
        ax.scatter([t["km"][0]], [t["km"][1]], s=40 + 0.5 * t["households"], facecolors="none", edgecolors="#ff9628", linewidths=2.0, zorder=6)
        if t.get("street"):
            f, to = t["street"]["from_km"], t["street"]["to_km"]
            ax.plot([f[0], to[0]], [f[1], to[1]], "-", color="#ff9628", lw=2.5, zorder=6)
        ax.text(t["km"][0] + 0.01 * span, t["km"][1] + 0.01 * span, f"{t['name']}（{t['households']}，线 {t['n_lines']}）", color="#ffe0b0", fontsize=7, zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.35, lw=0))
    ax.scatter([seat["km"][0]], [seat["km"][1]], s=260, marker="*", c="#ffe040", edgecolors="black", linewidths=0.8, zorder=9)
    # 中转站：三角，按主要功能着色；烽火台连到瞭望处
    for r in S["relays"]:
        if not inwin(r["km"]):
            continue
        f = r["functions"]
        main = next((x for x in ("换船", "过夜", "候风", "避风", "关卡", "烽火") if x in f), "烽火")
        ax.scatter([r["km"][0]], [r["km"][1]], s=110, marker="^", c=FUNC_COLOR[main], edgecolors="black", linewidths=0.8, zorder=9)
        if "烽火" in f:
            ax.plot([r["km"][0], r["lookout_km"][0]], [r["km"][1], r["lookout_km"][1]], ":", color="white", lw=1.0, zorder=8)
            ax.scatter([r["lookout_km"][0]], [r["lookout_km"][1]], s=40, marker="x", c="white", zorder=9)
        for e in r["routes"][:1]:
            th = np.radians(e["bearing_deg"])
            L = 0.06 * span
            ax.annotate("", xy=(r["km"][0] + L * np.sin(th), r["km"][1] + L * np.cos(th)), xytext=(r["km"][0], r["km"][1]),
                        arrowprops=dict(arrowstyle="->", color=FUNC_COLOR[main], lw=1.5), zorder=8)
        ax.text(r["km"][0] + 0.012 * span, r["km"][1] - 0.02 * span, f"{r['name']}：{'、'.join(f)}（{r['households']} 户）", color="#ffc0c0", fontsize=7, zorder=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.35, lw=0))
    MS = S.get("market") or {}
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("km 东")
    ax.set_ylabel("km 北")
    u, v = J["hydro"].get("wind_ms", [0, 0])
    ax.set_title(f"#{J['meta']['node']} 镇、邑治、航船、中转站：大泊场 {len(HB)}、镇 {len(T)}、航船 {len(S['boat_lines'])} 线、中转站 {len(S['relays'])}；"
                 f"年均风 ({u:.1f}, {v:.1f}) m/s", fontsize=10)
    handles = [Line2D([], [], marker="*", ls="", color="#ffe040", markeredgecolor="black", markersize=12, label="邑治"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#ff9628", markersize=9, label="镇（橙线：朝大泊场的街）"),
               Line2D([], [], marker="s", ls="", color="#40a0ff", markersize=7, label="大泊场（实心 = 镇挨着的；大小按船数）"),
               Line2D([], [], marker="o", ls="", color="white", markeredgecolor="black", markersize=5, label="走路赶集的村"),
               Line2D([], [], marker="o", ls="", color="#40e0ff", markeredgecolor="black", markersize=5, label="搭航船的村"),
               Line2D([], [], color="#888", lw=1.4, label="航船线（颜色按镇，箭头到镇上的泊场）")]
    handles += [Line2D([], [], marker="^", ls="", color=c, markeredgecolor="black", markersize=9, label=f"中转站：{k}") for k, c in FUNC_COLOR.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.75)
    out = Path(a.out) if a.out else d / "market_view.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
