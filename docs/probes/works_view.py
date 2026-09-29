"""水利俯视图（P6）：直接读岛群产物，画一块地方的晕渲 + 田 / 湿地 / 河 + 渠（按宽）、塘、闸、圩（圩堤）。

读 island.json、terrain.npz、rivers.json、settlements.json（P6 起有 waterworks）。窗口给行列（群栅格），或 --head / --patch 按渠首号 / 圩田片号自动取。
用法（仓库根下）：
    PYTHONUTF8=1 python docs/probes/works_view.py <岛群目录> [--rows r0,r1 --cols c0,c1 | --head 3 | --patch 1 | --auto head|patch] [--pad 25] [--out 图.png]
    例：python docs/probes/works_view.py out/seed42/islands/6329 --auto patch --out out/p6/polder_6329.png
--auto head：灌田最多的渠首；--auto patch：最大的一片圩田。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--rows")
    ap.add_argument("--cols")
    ap.add_argument("--head", type=int)
    ap.add_argument("--patch", type=int)
    ap.add_argument("--auto", choices=["head", "patch"])
    ap.add_argument("--pad", type=int, default=25)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    J = json.loads((d / "island.json").read_text(encoding="utf-8"))
    S = json.loads((d / "settlements.json").read_text(encoding="utf-8"))
    Z = np.load(d / "terrain.npz")
    RV = json.loads((d / "rivers.json").read_text(encoding="utf-8"))
    WK = S.get("waterworks")
    if WK is None:
        print("没有 waterworks（P6 之前的产物）")
        return 1
    res_m = float(J["raster"]["res_m"])
    iid = Z["island_id"]
    H, W = iid.shape
    pad = a.pad
    if a.auto == "head" and WK["heads"]:
        a.head = max(WK["heads"], key=lambda x: (x["served_km2"], -x["id"]))["id"]
    if a.auto == "patch" and WK["polder_patches"]:
        a.patch = max(WK["polder_patches"], key=lambda x: (x["polder_km2"], -x["id"]))["id"]
    if a.head:
        pts = [p for c in WK["canals"] if c.get("head") == a.head for p in c["pts"]]
        P = np.array(pts)
        r0, r1, c0, c1 = int(P[:, 0].min()) - pad, int(P[:, 0].max()) + pad, int(P[:, 1].min()) - pad, int(P[:, 1].max()) + pad
        title = f"渠首 {a.head}"
    elif a.patch:
        pid = Z["polder_id"]
        ids = next(x for x in WK["polder_patches"] if x["id"] == a.patch)["polders"]
        ii, jj = np.nonzero(np.isin(pid, ids))
        r0, r1, c0, c1 = int(ii.min()) - pad, int(ii.max()) + pad, int(jj.min()) - pad, int(jj.max()) + pad
        title = f"圩田片 {a.patch}"
    else:
        r0, r1 = (int(x) for x in a.rows.split(","))
        c0, c1 = (int(x) for x in a.cols.split(","))
        title = f"行 {r0}–{r1} 列 {c0}–{c1}"
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H, r1), min(W, c1)
    sl = (slice(r0, r1), slice(c0, c1))
    land = iid[sl] >= 0
    h = np.where(land, Z["height"][sl].astype(np.float64), np.nan)
    hv = np.where(np.isnan(h), np.nanmin(h), h)
    shade = LightSource(315, 45).hillshade(hv, vert_exag=3, dx=res_m, dy=res_m)
    base = np.array([0.80, 0.78, 0.70])
    rgb = base[None, None, :] * (0.35 + 0.65 * shade[..., None])
    lc = Z["landcover"][sl]
    cult = Z["cultivated"][sl]
    pol = Z["polder_id"][sl] > 0
    wet = lc == 9
    forest = lc == 4
    rgb[forest] = rgb[forest] * 0.55 + np.array([0.25, 0.45, 0.22]) * 0.45
    rgb[cult > 0] = rgb[cult > 0] * 0.35 + np.array([0.95, 0.80, 0.35]) * 0.65
    rgb[pol] = rgb[pol] * 0.35 + np.array([0.55, 0.85, 0.55]) * 0.65
    rgb[wet] = rgb[wet] * 0.3 + np.array([0.45, 0.65, 0.55]) * 0.7
    rgb[Z["lake"][sl]] = (0.25, 0.45, 0.8)
    rgb[~land] = (0.08, 0.09, 0.16)
    fig_w = 12.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * (r1 - r0) / max(1, c1 - c0) + 0.8))
    ax.imshow(rgb, interpolation="nearest", extent=(c0, c1, r1, r0))
    ax.contour(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5, hv, levels=np.arange(0, 6000, 20), colors="k", linewidths=0.25, alpha=0.3)
    px_per_cell = fig_w * 72.0 / max(1, c1 - c0)
    to_pt = lambda w_m: max(0.6, w_m / res_m * px_per_cell)
    for ln in RV.get("lines", []):
        Pp = np.asarray(ln["pts"], dtype=float)
        if Pp.shape[0] < 2 or not ((Pp[:, 0] >= r0) & (Pp[:, 0] < r1) & (Pp[:, 1] >= c0) & (Pp[:, 1] < c1)).any():
            continue
        lv = int(Pp[:, 3].max())
        ax.plot(Pp[:, 1], Pp[:, 0], color=(0.2, 0.35, 0.85) if lv else (0.45, 0.6, 0.95), lw=to_pt(float(Pp[:, 2].mean())) if lv else 0.6, alpha=0.9, zorder=3)
    for c in WK["canals"]:
        Pp = np.asarray(c["pts"], dtype=float)
        if not ((Pp[:, 0] >= r0) & (Pp[:, 0] < r1) & (Pp[:, 1] >= c0) & (Pp[:, 1] < c1)).any():
            continue
        k = c["kind"]
        col = {"干渠": "#0a8a8a", "支渠": "#18b8b0", "排水渠": "#2050c0", "纵浦": "#1060d0", "横塘": "#1060d0"}[k]
        lw = max(1.4 if k == "干渠" else 0.9, to_pt(c.get("width_m", 8.0) * 3)) if k in ("干渠", "支渠") else 1.1
        ax.plot(Pp[:, 1], Pp[:, 0], color=col, lw=lw, solid_capstyle="round", zorder=5)
    for p in WK["polders"]:
        a0, b0, a1, b1 = p["cells_bbox"]
        if a1 < r0 or a0 > r1 or b1 < c0 or b0 > c1:
            continue
    # 圩堤：圩的边（圩号不同的格之间）
    pid = Z["polder_id"][sl]
    segs = []
    for i in range(pid.shape[0]):
        for j in range(pid.shape[1]):
            v = pid[i, j]
            if not v:
                continue
            if i == 0 or pid[i - 1, j] != v:
                segs.append(((c0 + j, r0 + i), (c0 + j + 1, r0 + i)))
            if i == pid.shape[0] - 1 or pid[i + 1, j] != v:
                segs.append(((c0 + j, r0 + i + 1), (c0 + j + 1, r0 + i + 1)))
            if j == 0 or pid[i, j - 1] != v:
                segs.append(((c0 + j, r0 + i), (c0 + j, r0 + i + 1)))
            if j == pid.shape[1] - 1 or pid[i, j + 1] != v:
                segs.append(((c0 + j + 1, r0 + i), (c0 + j + 1, r0 + i + 1)))
    if segs:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="#5a4020", linewidths=0.8, zorder=4, alpha=0.8))
    inside = lambda cell: r0 <= cell[0] < r1 and c0 <= cell[1] < c1
    for x in WK["ponds"]:
        if inside(x["cell"]):
            ax.scatter(x["cell"][1] + 0.5, x["cell"][0] + 0.5, s=8 + np.sqrt(x["area_m2"]) * 0.35, c={"村塘": "#40a0ff", "山塘": "#2070d0", "圩塘": "#60c0ff", "堰塘": "#1040a0"}[x["kind"]],
                       edgecolors="k", linewidths=0.4, zorder=7, marker="o")
    for x in WK["sluices"]:
        if inside(x["cell"]):
            ax.scatter(x["cell"][1] + 0.5, x["cell"][0] + 0.5, s=30 if x["kind"] == "渠首闸" else 14, c="#e03030" if x["kind"] == "渠首闸" else "#402020",
                       marker="s", zorder=8, edgecolors="w", linewidths=0.4)
    for v in S["villages"]:
        if inside(v["cell"]):
            ax.scatter(v["cell"][1] + 0.5, v["cell"][0] + 0.5, s=6 + v["households"] * 0.3, c="white", edgecolors="k", linewidths=0.5, zorder=6)
    for hd in WK["heads"]:
        if inside(hd["cell"]):
            ax.annotate(f"渠首{hd['id']}（{hd['served_km2']:.1f} km²）", (hd["cell"][1] + 0.5, hd["cell"][0] + 0.5), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="#a01010", zorder=9)
    km = 1000.0 / res_m
    ax.plot([c0 + 3, c0 + 3 + km], [r1 - 3, r1 - 3], color="w", lw=3, zorder=10)
    ax.text(c0 + 3 + km / 2, r1 - 5, "1 km", color="w", ha="center", fontsize=9, zorder=10)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_xticks([])
    ax.set_yticks([])
    ws = WK["summary"]
    ax.set_title(f"#{J['meta']['node']} {title}：渠 {ws['canal_km']:.0f} km / 塘 {ws['n_ponds']} / 闸 {ws['n_sluices']} / 圩田 {ws['polder_km2']:.1f} km²（全群）", fontsize=10)
    handles = [Patch(color=(0.95, 0.80, 0.35), label="已垦的田"), Patch(color=(0.55, 0.85, 0.55), label="圩田"), Patch(color=(0.45, 0.65, 0.55), label="湿地（芦苇荡）"),
               Line2D([], [], color="#0a8a8a", lw=2, label="干渠"), Line2D([], [], color="#18b8b0", lw=1.2, label="支渠"),
               Line2D([], [], color="#1060d0", lw=1.1, label="纵浦 / 横塘 / 排水渠"), Line2D([], [], color="#5a4020", lw=0.8, label="圩堤"),
               Line2D([], [], color=(0.2, 0.35, 0.85), lw=2, label="河"),
               Line2D([], [], marker="o", ls="", color="#40a0ff", label="塘"), Line2D([], [], marker="s", ls="", color="#e03030", label="渠首闸"),
               Line2D([], [], marker="s", ls="", color="#402020", label="圩闸 / 排水闸"), Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="村")]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)
    out = Path(a.out) if a.out else d / f"works_{r0}_{c0}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(out, (r1 - r0, c1 - c0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
