"""岛群剖面体检：直接读生成器产物，出 Zhouzhu docs/PLAN-LAND.md 第二节那两张表的同口径数（P5 起，P6、P7 也用）。

读每群目录里的 island.json、terrain.npz、rivers.json、settlements.json（有就读 resources.json），不经 Zhouzhu 的导出：
  1. 地形 / 地表 / 河 / 湿地：每群一行（年降水、主岛面积与坡 < 2° 的占比、宽 ≥ 40 m 的河段长、最大出口的汇水与河宽、湿地面积）；
  2. 田：陆地、宜垦、已垦（在种）、撂荒；已垦离最近的村 / 聚落多远（中位 / P90 / 最大）；人口（⑨ 与聚落层 Σ 户 × 户均）；
  3. 小岛有没有人（主岛除外，按 < 10 / 10–30 / 30–100 / > 100 km² 分档）：有人住 = 有常住的村、散户或专业聚落；有村 = 有农村（村或散户）；
     只有专业聚落；有人用、没人住（工棚 / 季节住 / 放牧 / 烽火台 / 庙 / 墓岛 / 废村）；
  4. 有人用、没人住的岛，按用途列；索桥与桥头的个数。
P5 之前的产物（没有 cultivable / uses / ruins，专业聚落都算常住）也能量：宜垦记「—」，已垦按旧的 arable。

用法（仓库根下）：
    PYTHONUTF8=1 python docs/probes/transect.py <岛群目录的上级> [节点,…] [--run out/seed42] [--json 结果.json]
    例：python docs/probes/transect.py out/seed42/islands 6615,6610,6329,6073,6072,5766,5498,2051 --run out/seed42
--run 给了就读 ⑦ 的地区号与 ⑨ 的人口（没给只报聚落层的人口）。先生成岛群：skyisle island <节点> --run out/seed42（cpp 一群几秒）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NODES = [6615, 6610, 6329, 6073, 6072, 5766, 5498, 2051]
BINS = ((0, 10), (10, 30), (30, 100), (100, 1e9))
LC_WET = 9
NONRES = ("工棚", "季节住")          # 专业聚落的住法里不算常住的


def bin_label(lo, hi):
    return f"< {hi:g} km²" if lo == 0 else (f"> {lo:g} km²" if hi >= 1e9 else f"{lo:g}–{hi:g} km²")


def near_dist_km(cells: np.ndarray, pts: np.ndarray, res_km: float) -> np.ndarray:
    """每个格到最近的点（格坐标）的距离 km；分块广播。"""
    if pts.shape[0] == 0 or cells.shape[0] == 0:
        return np.full(cells.shape[0], np.inf)
    out = np.empty(cells.shape[0])
    P = pts.astype(np.float64)
    for s in range(0, cells.shape[0], 4096):
        blk = cells[s:s + 4096].astype(np.float64)
        d2 = ((blk[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        out[s:s + 4096] = np.sqrt(d2.min(axis=1))
    return out * res_km


def group(d: Path, node: int, run: Path | None) -> dict:
    J = json.loads((d / "island.json").read_text(encoding="utf-8"))
    S = json.loads((d / "settlements.json").read_text(encoding="utf-8")) if (d / "settlements.json").exists() else None
    Z = np.load(d / "terrain.npz")
    R = json.loads((d / "rivers.json").read_text(encoding="utf-8")) if (d / "rivers.json").exists() else {"lines": []}
    res_m = float(J["raster"]["res_m"])
    res_km = res_m / 1000.0
    cell_km2 = res_km * res_km
    iid = Z["island_id"]
    land = iid >= 0
    main = iid == 0
    slope = Z["slope_deg"].astype(np.float64)
    I = J["islands"]
    out = {"node": node, "lat": J["meta"]["lat"], "lon": J["meta"]["lon"], "precip_mm": J["hydro"]["precip_mm"], "n_islands": len(I),
           "land_km2": round(float(land.sum()) * cell_km2, 1), "main_km2": I[0]["area_km2"],
           "main_slope_lt2": round(float((slope[main] < 2.0).mean()), 3)}
    # 河：rivers.json 的中心线，段长按格心距 × 分辨率；宽取段两端的平均
    L = L40 = 0.0
    for ln in R.get("lines", []):
        P = np.asarray(ln["pts"], dtype=np.float64)
        if P.shape[0] < 2:
            continue
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])) * res_km
        w = 0.5 * (P[:-1, 2] + P[1:, 2])
        L += float(seg.sum())
        L40 += float(seg[w >= 40.0].sum())
    rv = J["hydro"].get("rivers") or []
    out.update({"river_km": round(L, 0), "river_w40_km": round(L40, 0),
                "outlet": [rv[0]["basin_km2"], rv[0]["width_m"]] if rv else None,
                "wetland_km2": round(float(((Z["landcover"] == LC_WET) & land).sum()) * cell_km2, 1)})
    if run is not None:
        try:
            out["region"] = int(np.load(run / "s07_centers" / "regions.npz")["region"][node])
            out["pop_polity"] = round(float(np.load(run / "s09_polity" / "polity.npz")["pop"][node]), 0)
        except (OSError, KeyError, IndexError):
            pass
    # 田
    new = "cultivated" in Z.files
    cult = (Z["cultivated"] > 0) if new else (Z["arable"] > 0 if "arable" in Z.files else np.zeros_like(land))
    fallow = (Z["fallow_years"] > 0) if "fallow_years" in Z.files else np.zeros_like(land)
    out.update({"format": "P5" if new else "P5 之前",
                "cultivable_km2": round(float((Z["cultivable"] > 0).sum()) * cell_km2, 1) if "cultivable" in Z.files else None,
                "cultivated_km2": round(float(cult.sum()) * cell_km2, 1), "fallow_km2": round(float(fallow.sum()) * cell_km2, 1)})
    if S is None:
        return out
    hh_size = float(S.get("household_size", 5.0))
    parts = S["households_in_villages"] + S["households_in_hamlets"] + S.get("households_in_towns_market", 0) + S.get("households_in_specials", 0)
    out.update({"pop_settle": round(parts * hh_size, 0), "households": S["households"], "population": S["population"],
                "n_villages": S["n_villages"], "n_hamlets": S["n_hamlets"], "n_specials": len(S.get("specials", [])),
                "n_bridges": sum(1 for e in J.get("links", []) if e.get("kind") == "bridge"), "n_bridgeheads": len(S.get("bridgeheads", []))})
    vc = np.array([v["cell"] for v in S["villages"]], dtype=np.int64).reshape(-1, 2)
    hc = np.array([v["cell"] for v in S["hamlets"]], dtype=np.int64).reshape(-1, 2)
    ii, jj = np.nonzero(cult)
    cells = np.stack([ii, jj], axis=1)
    dv = near_dist_km(cells, vc, res_km)
    dh = near_dist_km(cells, np.vstack([vc, hc]), res_km)
    q = lambda a, p: round(float(np.quantile(a, p)), 2) if a.size else None
    out["cult_to_village_km"] = {"median": q(dv, 0.5), "p90": q(dv, 0.9), "p99": q(dv, 0.99), "max": q(dv, 1.0),
                                 "share_le_1km": round(float((dv <= 1.0).mean()), 3) if dv.size else None,
                                 "share_le_2km": round(float((dv <= 2.0).mean()), 3) if dv.size else None}
    out["cult_to_settlement_km"] = {"median": q(dh, 0.5), "p90": q(dh, 0.9), "p99": q(dh, 0.99), "max": q(dh, 1.0),
                                    "share_le_1km": round(float((dh <= 1.0).mean()), 3) if dh.size else None}
    # 小岛有没有人
    farm, spec_res, spec_non, hamlet = Counter(), Counter(), Counter(), Counter()
    for v in S["villages"]:
        farm[v["island"]] += 1
    for v in S["hamlets"]:
        hamlet[v["island"]] += 1
    for x in S.get("specials", []):
        (spec_non if x.get("occupancy") in NONRES else spec_res)[x["island"]] += 1
    uses = defaultdict(set)                      # 岛 → 用途（有人用、没人住）
    for x in S.get("specials", []):
        if x.get("occupancy") in NONRES:
            uses[x["island"]].add(f"{x['occupancy']}（{x['kind']}）")
    for u in S.get("uses", []):
        uses[u["island"]].add(u["kind"])
    for r in S.get("ruins", []):
        uses[r["island"]].add("废村")
    rows = []
    for lo, hi in BINS:
        ids = [i["id"] for i in I if i["id"] != 0 and lo <= i["area_km2"] < hi]
        if not ids:
            rows.append({"bin": bin_label(lo, hi), "n": 0})
            continue
        lived = [i for i in ids if farm[i] or hamlet[i] or spec_res[i]]
        vil = [i for i in ids if farm[i] or hamlet[i]]
        only = [i for i in ids if spec_res[i] and not farm[i] and not hamlet[i]]
        used = [i for i in ids if i not in lived and uses.get(i)]
        rows.append({"bin": bin_label(lo, hi), "n": len(ids), "lived": len(lived), "village": len(vil), "only_special": len(only),
                     "used_not_lived": len(used)})
    out["small_islands"] = rows
    by_use = Counter()
    for k, us in uses.items():
        if k == 0 or farm[k] or hamlet[k] or spec_res[k]:
            continue
        for u in us:
            by_use[u] += 1
    out["used_not_lived_by_use"] = dict(sorted(by_use.items(), key=lambda kv: -kv[1]))
    out["used_not_lived_islands"] = {int(k): sorted(us) for k, us in sorted(uses.items())
                                     if k != 0 and not (farm[k] or hamlet[k] or spec_res[k])}
    fl = S.get("farmland")
    if fl:
        out["farmland"] = fl
    if S.get("ruins") is not None:
        out["ruins"] = [{"island": r["island"], "abandoned_years": r["abandoned_years"], "households_before": r.get("households_before"),
                         "fallow_km2": r.get("fallow_km2")} for r in S["ruins"]]
    if S.get("uses") is not None:
        out["uses"] = Counter(u["kind"] for u in S["uses"])
    return out


def pct(a, n):
    return f"{a / n:.0%}" if n else "—"


def print_tables(G: list[dict]) -> None:
    print("## 地形、地表、河、湿地（每群一行）\n")
    print("| 群 | 地区 | 北纬 / 东经 | 年降水 | 岛数 | 主岛 km² | 主岛坡 <2° | 河长 km | 宽 ≥40 m 的河段 km | 最大出口 汇水 km² / 宽 m | 湿地 km² |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in G:
        o = g["outlet"]
        print(f"| #{g['node']} | {g.get('region', '—')} | {g['lat']:.1f} / {g['lon']:.1f} | {g['precip_mm']:.0f} | {g['n_islands']} | {g['main_km2']:.0f} | "
              f"{g['main_slope_lt2']:.0%} | {g['river_km']:.0f} | {g['river_w40_km']:.0f} | {f'{o[0]:.0f} / {o[1]:.0f}' if o else '—'} | {g['wetland_km2']:.1f} |")
    print("\n## 田与人口\n")
    print("| 群 | 产物 | 陆地 km² | 宜垦 km² | 已垦（在种）km² | 撂荒 km² | 已垦离最近的村 km：中位 / P90 / P99 / 最大（≤1 km / ≤2 km） | 离最近的村或散户：中位 / P90 / P99 / 最大（≤1 km） | 村 / 散户 / 专业 | 人口 ⑨ | 聚落层 Σ户×5 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in G:
        c = g.get("cult_to_village_km", {})
        h = g.get("cult_to_settlement_km", {})
        ck = g["cultivable_km2"]
        print(f"| #{g['node']} | {g['format']} | {g['land_km2']:.0f} | {'—' if ck is None else f'{ck:.0f}'} | {g['cultivated_km2']:.0f} | {g['fallow_km2']:.1f} | "
              f"{c.get('median')} / {c.get('p90')} / {c.get('p99')} / {c.get('max')}（{pct(c.get('share_le_1km') or 0, 1)} / {pct(c.get('share_le_2km') or 0, 1)}） | "
              f"{h.get('median')} / {h.get('p90')} / {h.get('p99')} / {h.get('max')}（{pct(h.get('share_le_1km') or 0, 1)}） | "
              f"{g.get('n_villages')} / {g.get('n_hamlets')} / {g.get('n_specials')} | {g.get('pop_polity', '—'):.0f} | {g.get('pop_settle', '—'):.0f} |")
    print("\n## 小岛有没有人（主岛除外）：座数，有人住（常住：村、散户或常住的专业聚落）/ 有村（村或散户）/ 只有专业聚落 / 有人用、没人住\n")
    print("| 群 | " + " | ".join(bin_label(lo, hi) for lo, hi in BINS) + " |")
    print("|---|" + "---|" * len(BINS))
    for g in G:
        cells = []
        for r in g.get("small_islands", []):
            if not r["n"]:
                cells.append("—")
                continue
            n = r["n"]
            cells.append(f"{n} 座：住 {pct(r['lived'], n)}，村 {pct(r['village'], n)}，只专业 {pct(r['only_special'], n)}，用 {pct(r['used_not_lived'], n)}")
        print(f"| #{g['node']} | " + " | ".join(cells) + " |")
    print("\n## 有人用、没人住的岛（按用途；一岛多用途各记一次）；索桥 / 桥头\n")
    for g in G:
        u = g.get("used_not_lived_by_use") or {}
        print(f"- #{g['node']}：{'、'.join(f'{k} {v}' for k, v in u.items()) or '无'}；索桥 {g.get('n_bridges', 0)} / 桥头 {g.get('n_bridgeheads', 0)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="岛群目录的上级（里面是 <节点>/island.json …），如 out/seed42/islands")
    ap.add_argument("nodes", nargs="?", default=",".join(map(str, NODES)))
    ap.add_argument("--run", default=None, help="行星 run 目录（读 ⑦ 地区号、⑨ 人口），如 out/seed42")
    ap.add_argument("--json", default=None, help="把全部数写成 JSON")
    a = ap.parse_args(argv)
    root = Path(a.root)
    run = Path(a.run) if a.run else None
    G = [group(root / n, int(n), run) for n in a.nodes.split(",") if (root / n / "island.json").exists()]
    print_tables(G)
    if a.json:
        Path(a.json).write_text(json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
