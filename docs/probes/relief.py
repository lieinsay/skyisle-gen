"""地形统计（PLAN-NATURE B 的参照）：山地 / 高山 / 丘陵的坡、陡坡占比、1 km 方窗高差、岩性与地貌，直接读岛群产物。

    python docs/probes/relief.py out/seed42/islands 6329,2051,1165 [--base 旧/islands]

每群一行：陆地 km²、各地形区的坡中位（°）、> 30° / > 40° 占全陆地的比例、山地（含高山）里 > 30° 的比例、
1 km 方窗（10 格）高差 p50 / p90（m）、主岛峰 − 岸缘（m）；有 lith 的产物再报各岩性的面积占比，有 island.json 的 landforms 再报各类地貌几处。
--base 给改前的目录时，每群多一行改前的数。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ZONES = {1: "高山", 2: "山地", 3: "丘陵", 4: "台地平原"}
LITH = ["—", "石灰岩", "泥灰岩", "辉长岩", "蛇纹岩", "浮石"]


def window_relief(h, land, cells):
    """cells × cells 方窗（步长 cells/2）里陆地的高差。"""
    H, W = h.shape
    step = max(1, cells // 2)
    out = []
    for i in range(0, H - cells + 1, step):
        for j in range(0, W - cells + 1, step):
            m = land[i:i + cells, j:j + cells]
            if m.mean() < 0.9:
                continue
            v = h[i:i + cells, j:j + cells][m]
            out.append(float(v.max() - v.min()))
    return np.array(out) if out else np.array([0.0])


def stats(d: Path):
    t = np.load(d / "terrain.npz")
    js = json.loads((d / "island.json").read_text(encoding="utf-8"))
    res_m = float(js["meta"]["res_m"]) if "res_m" in js.get("meta", {}) else float(js["grid"]["res_m"])
    h = t["height"].astype(float)
    land = t["island_id"] >= 0
    s = t["slope_deg"].astype(float)
    z = t["terrain_zone"]
    n = land.sum()
    row = {"km2": n * (res_m / 1000) ** 2, "res": res_m}
    for k, name in ZONES.items():
        m = land & (z == k)
        row[name] = float(np.median(s[m])) if m.any() else float("nan")
        row[name + "占"] = m.sum() / n
    row[">30"] = (land & (s > 30)).sum() / n
    row[">40"] = (land & (s > 40)).sum() / n
    mt = land & ((z == 1) | (z == 2))
    row["山>30"] = (mt & (s > 30)).sum() / max(1, mt.sum())
    cells = max(2, int(round(1000 / res_m)))
    wr = window_relief(h, land, cells)
    row["1km_p50"] = float(np.percentile(wr, 50))
    row["1km_p90"] = float(np.percentile(wr, 90))
    main = js["islands"][0]
    row["起伏"] = main["peak_m"] - main["rim_m"] if "peak_m" in main else float("nan")
    if "lith" in t:
        li = t["lith"]
        row["lith"] = {LITH[k]: (land & (li == k)).sum() / n for k in range(1, len(LITH)) if (land & (li == k)).any()}
    lf = js.get("landforms")
    if lf:
        cnt = {}
        for r in lf:
            cnt[r["kind"]] = cnt.get(r["kind"], 0) + 1
        row["landforms"] = cnt
    return row


def fmt(node, r, tag=""):
    zs = " ".join(f"{ZONES[k]} {r[ZONES[k]]:4.1f}°({r[ZONES[k] + '占'] * 100:3.0f}%)" for k in ZONES)
    s = (f"{tag}{node:>6} {r['km2']:7.0f} km² {r['res']:.0f} m | {zs} | >30° {r['>30'] * 100:4.1f}% >40° {r['>40'] * 100:4.1f}% 山>30° {r['山>30'] * 100:4.1f}% | "
         f"1km 高差 {r['1km_p50']:4.0f}/{r['1km_p90']:4.0f} m | 主岛起伏 {r['起伏']:5.0f}")
    if "lith" in r:
        s += " | " + " ".join(f"{k} {v * 100:.0f}%" for k, v in r["lith"].items())
    if "landforms" in r:
        s += " | " + " ".join(f"{k} {v}" for k, v in sorted(r["landforms"].items()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("nodes")
    ap.add_argument("--base")
    a = ap.parse_args()
    for node in a.nodes.split(","):
        d = Path(a.root) / node
        if not (d / "terrain.npz").exists():
            print(node, "没有产物", file=sys.stderr)
            continue
        if a.base and (Path(a.base) / node / "terrain.npz").exists():
            print(fmt(node, stats(Path(a.base) / node), "改前 "))
        print(fmt(node, stats(d), "     " if a.base else ""))


if __name__ == "__main__":
    main()
