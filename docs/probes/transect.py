"""岛群剖面体检：直接读生成器产物，出 Zhouzhu docs/PLAN-LAND.md 第二节那两张表的同口径数（P5 起，P6、P7 也用）。

读每群目录里的 island.json、terrain.npz、rivers.json、settlements.json（有就读 resources.json），不经 Zhouzhu 的导出：
  1. 地形 / 地表 / 河 / 湿地：每群一行（年降水、主岛面积与坡 < 2° 的占比、宽 ≥ 40 m 的河段长、最大出口的汇水与河宽、湿地面积）；
  2. 田：陆地、宜垦、已垦（在种）、撂荒；已垦离最近的村 / 聚落多远（中位 / P90 / 最大）；人口（⑨ 与聚落层 Σ 户 × 户均）；
  3. 小岛有没有人（主岛除外，按 < 10 / 10–30 / 30–100 / > 100 km² 分档）：有人住 = 有常住的村、散户或专业聚落；有村 = 有农村（村或散户）；
     只有专业聚落；有人用、没人住（工棚 / 季节住 / 放牧 / 烽火台 / 庙 / 墓岛 / 废村）；P7 起有常住户的中转站算常住（「只有专业聚落」那一格也算它）、烽火台（中转站）算有人用；
  4. 有人用、没人住的岛，按用途列；索桥与桥头的个数；
  5. 人住在哪：总户数里住在主岛、主岛以外 ≥ 30 km² 的岛、< 30 km² 的岛各几成（村与镇的户、散户、常住的专业聚落在本岛；
     工棚与季节住的专业聚落的人住在 home_village 那个村）——总人口不变，看的是住处；
  6. 水利（P6 起，settlements.json 的 waterworks）：渠首几处（季节性几处）、谷口的渠多少 km、灌多少田（占已垦几成）、
     塘几口（村塘 / 山塘 / 圩塘 / 堰塘）、闸几座（渠首闸 / 圩闸 / 排水闸）、湿地多少、圩田多少 km²（占湿地几成、几圩几片、水田几成）、纵浦横塘与圩堤多长；
  7. 镇、邑治、航船、中转站（P7，settlements.json 的 towns / harbors / boat_lines / relays）：镇几个、各离大泊场多远（同岛最近的一处）、几条航船线汇到镇、
     有别的岛上的人来赶集却没有航船的镇（P7 之前没有航船：赶集按直线跨岛，这一列就是靠别的岛上的人撑起来的镇）；邑治在哪（哪座岛、农户、离大泊场多远、汇了几条线）；
     航船几条 / 多长 / 连几个村；中转站几处（按功能）、住几户。P7 之前的产物没有大泊场：用 market.py 同一套算法在它的地形与田上现算（地形、田没变，算出来的与 P7 之后的相同）。
  8. 水利离村多远（P6b，Zhouzhu PLAN-LAND L30）：渠首、谷口的渠段、圩田的渠段、圩、塘、闸各离最近的村、离管它的村（village，P6b 起才有）多远，
     没有管它的村的、废弃的水利（abandoned）几处，村 / 圩村 / 挂着圩田的村、已垦 = 额度、人口 = ⑨；
  9. 原始地貌与人工地貌（P6b，L31）：原始湿地 / 现状湿地 / 圩田，没人常住的岛上的湿地与田，人工改造（landuse）各类的面积与开垦的田原来是什么。
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
    parts = (S["households_in_villages"] + S["households_in_hamlets"] + S.get("households_in_towns_market", 0) + S.get("households_in_specials", 0)
             + S.get("households_in_relays", 0))
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
    for x in S.get("relays", []):                 # P7：有常住户的中转站算常住（和常住的专业聚落一样），轮班的（烽火台）算有人用
        if x["households"] > 0:
            spec_res[x["island"]] += 1
    uses = defaultdict(set)                      # 岛 → 用途（有人用、没人住）
    for x in S.get("specials", []):
        if x.get("occupancy") in NONRES:
            uses[x["island"]].add(f"{x['occupancy']}（{x['kind']}）")
    for u in S.get("uses", []):
        uses[u["island"]].add(u["kind"])
    for r in S.get("ruins", []):
        uses[r["island"]].add("废村")
    for x in S.get("relays", []):
        if x["households"] == 0:
            uses[x["island"]].add("烽火台" if "烽火" in x["functions"] else "中转站（轮班）")
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
    # 人住在哪（户）：村（含镇的非农户）、散户、常住的专业聚落在本岛；工棚、季节住的住在 home_village 的岛
    vil_isl = {v["id"]: v["island"] for v in S["villages"]}
    res = Counter()
    for v in S["villages"]:
        res[v["island"]] += v["households"] + v.get("households_market", 0)
    for v in S["hamlets"]:
        res[v["island"]] += v["households"]
    for x in S.get("specials", []):
        k = x["island"]
        if x.get("occupancy") in NONRES and x.get("home_village") in vil_isl:
            k = vil_isl[x["home_village"]]
        res[k] += x["households"]
    for x in S.get("relays", []):
        res[x["island"]] += x["households"]
    area = {i["id"]: i["area_km2"] for i in I}
    out["residence_households"] = {"total": sum(res.values()), "main": res[0],
                                   "big": sum(v for k, v in res.items() if k != 0 and area[k] >= 30.0),
                                   "small": sum(v for k, v in res.items() if k != 0 and area[k] < 30.0)}
    by_use = Counter()
    for k, us in uses.items():
        if k == 0 or farm[k] or hamlet[k] or spec_res[k]:
            continue
        for u in us:
            by_use[u] += 1
    out["used_not_lived_by_use"] = dict(sorted(by_use.items(), key=lambda kv: (-kv[1], kv[0])))     # 同数按名字排（集合的次序每次跑不一样）
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
    out["market"] = market(J, S, Z, res_km)
    out["manage"] = works_manage(S, Z, res_km)
    out["nature"] = nature(J, S, Z, cell_km2)
    WK = S.get("waterworks")
    if WK is not None:
        ws = WK["summary"]
        main_km = sum(c["length_km"] for c in WK["canals"] if c["kind"] == "干渠")
        out["works"] = {**{k: ws[k] for k in ("n_heads", "n_heads_seasonal", "canal_km", "polder_canal_km", "drain_km", "commanded_km2", "commanded_share",
                                                "n_ponds", "ponds", "n_sluices", "sluices", "wetland_km2", "polder_km2", "polder_share", "n_polders",
                                                "n_polder_patches", "dike_km")},
                        "main_canal_km": round(main_km, 1), "paddy_share": round(sum(p["paddy"] for p in WK["polders"]) / max(1, len(WK["polders"])), 3),
                        "heads_main_island": sum(1 for h in WK["heads"] if h["island"] == 0),
                        "canal_km_per_100km2": round(100.0 * ws["canal_km"] / max(1e-9, out["land_km2"]), 2)}
    return out


WORKS_KINDS = ("渠首", "谷口的渠", "圩田的渠", "圩", "塘", "闸")


def _stats(a: list[float], walk: float) -> dict | None:
    if not a:
        return None
    x = np.asarray(a, dtype=np.float64)
    return {"n": int(x.size), "median": round(float(np.median(x)), 2), "p90": round(float(np.quantile(x, 0.9)), 2), "max": round(float(x.max()), 2),
            "over": round(float((x > walk + 1e-9).mean()), 3)}


def works_manage(S: dict, Z, res_km: float, walk_km: float = 2.0) -> dict | None:
    """P6b（L30）：水利离村多远——每处（渠首、谷口的渠段、圩田的渠段、圩、塘、闸）离最近的村、离管它的村（P6b 起记 village；之前没有）。
    点按格心（渠的 pts 已是格心 / 格边坐标，村 = 格 + 0.5）；渠段取离村最远的那个点；圩取圩的格心的均值（另记最远那格离管它的村）。
    废弃的（abandoned，没人管了）另数，不进离村的数。"""
    WK = S.get("waterworks")
    if WK is None or not S["villages"]:
        return None
    vc = np.array([[v["cell"][0] + 0.5, v["cell"][1] + 0.5] for v in S["villages"]], dtype=np.float64)
    vpos = {v["id"]: np.array([v["cell"][0] + 0.5, v["cell"][1] + 0.5]) for v in S["villages"]}

    def near(P):                                  # 每个点离最近的村（km）
        d2 = ((P[:, None, :] - vc[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2.min(axis=1)) * res_km

    def mgr(P, vid):
        if vid is None or vid not in vpos:
            return None
        return np.sqrt(((P - vpos[vid][None, :]) ** 2).sum(-1)) * res_km
    items = {k: [] for k in WORKS_KINDS}
    aband = Counter()
    pid = Z["polder_id"] if "polder_id" in Z.files else None

    def add(kind, P, vid, far=None):
        dn = float(near(P).max())
        dm = mgr(P, vid)
        items[kind].append((dn, None if dm is None else float(dm.max()), far))
    for x in WK["heads"]:
        if x.get("abandoned"):
            aband["渠首"] += 1
            continue
        add("渠首", np.array([[x["cell"][0] + 0.5, x["cell"][1] + 0.5]]), x.get("village"))
    for c in WK["canals"]:
        if c.get("abandoned"):
            aband["渠段"] += 1
            aband["渠 km"] += c["length_km"]
            continue
        add("谷口的渠" if c["kind"] in ("干渠", "支渠") else "圩田的渠", np.asarray(c["pts"], dtype=np.float64), c.get("village"))
    if pid is not None and WK["polders"]:
        flat = pid.ravel()
        idx = np.flatnonzero(flat > 0)
        o = idx[np.argsort(flat[idx], kind="stable")]
        b = np.searchsorted(flat[o], np.arange(1, len(WK["polders"]) + 2))
        W = pid.shape[1]
        for p in WK["polders"]:
            if p.get("abandoned"):
                aband["圩"] += 1
                continue
            cells = o[b[p["id"] - 1]:b[p["id"]]]
            P = np.stack([cells // W + 0.5, cells % W + 0.5], axis=1).astype(np.float64)
            far = mgr(P, p.get("village"))
            add("圩", P.mean(axis=0, keepdims=True), p.get("village"), None if far is None else float(far.max()))
    for x in WK["ponds"]:
        if x.get("abandoned"):
            aband["塘"] += 1
            continue
        add("塘", np.array([[x["cell"][0] + 0.5, x["cell"][1] + 0.5]]), x.get("village"))
    for x in WK["sluices"]:
        if x.get("abandoned"):
            aband["闸"] += 1
            continue
        add("闸", np.array([[x["cell"][0] + 0.5, x["cell"][1] + 0.5]]), x.get("village"))
    out = {"walk_km": walk_km, "near": {}, "mgr": {}, "unmanaged": {}, "abandoned": dict(aband)}
    for k, L in items.items():
        out["near"][k] = _stats([t[0] for t in L], walk_km)
        dm = [t[1] for t in L if t[1] is not None]
        out["mgr"][k] = _stats(dm, walk_km) if dm else None
        out["unmanaged"][k] = sum(1 for t in L if t[1] is None)
    fars = [t[2] for t in items["圩"] if t[2] is not None]
    out["polder_far_max_km"] = round(max(fars), 2) if fars else None
    out["n_polder_villages"] = sum(1 for v in S["villages"] if v.get("polder"))
    out["n_villages_with_polder"] = sum(1 for v in S["villages"] if v.get("polder_fields"))
    return out


LC_NAMES = ["虚空", "崖缘", "裸岩", "高山草甸", "林地", "灌丛", "草坡", "可耕地", "梯田", "湿地", "河道", "湖"]


def nature(J: dict, S: dict, Z, cell_km2: float) -> dict:
    """P6b（L31）：原始地貌与人工地貌——原始湿地 / 现状湿地 / 圩田；没人常住的岛上的湿地与田；人工改造各类的面积。
    P6b 之前的产物没有 landcover_natural：原始湿地按「现状湿地 + 圩田」（P6 的圩田是聚落层动湿地的唯一一处）。"""
    iid = Z["island_id"]
    land = iid >= 0
    lc = Z["landcover"]
    pol = (Z["polder_id"] > 0) if "polder_id" in Z.files else np.zeros_like(land)
    has_nat = "landcover_natural" in Z.files
    nat = Z["landcover_natural"] if has_nat else None
    wet_now = (lc == LC_WET) & land
    wet_nat = ((nat == LC_WET) & land) if has_nat else (wet_now | pol)
    n = len(J["islands"])
    status = {t["island"]: t["status"] for t in S.get("land_tenure", [])}
    farm = {v["island"] for v in S["villages"]} | {h["island"] for h in S["hamlets"]}
    nobody = np.zeros(n, dtype=bool)                  # 没人常住的岛（荒岛、有人用、季节住）
    for k in range(n):
        nobody[k] = status.get(k, "常住" if k in farm else "荒岛") != "常住"
    nb = land & nobody[np.where(land, iid, 0)]
    cult = Z["cultivated"] > 0 if "cultivated" in Z.files else np.zeros_like(land)
    km2 = lambda m: round(float(m.sum()) * cell_km2, 2)
    wet_isl = np.unique(iid[wet_nat])
    area = {i["id"]: i["area_km2"] for i in J["islands"]}
    out = {"has_natural": has_nat, "wet_natural_km2": km2(wet_nat), "wet_now_km2": km2(wet_now), "polder_km2": km2(pol),
           "wet_islands": int(wet_isl.size), "wet_island_min_km2": round(min(area[int(k)] for k in wet_isl), 1) if wet_isl.size else None,
           "nobody_islands": int(nobody.sum()), "nobody_land_km2": km2(nb), "nobody_wet_natural_km2": km2(wet_nat & nb),
           "nobody_wet_now_km2": km2(wet_now & nb), "nobody_cultivated_km2": km2(cult & nb),
           "nobody_wet_changed_km2": km2(wet_nat & nb & ~wet_now)}
    if has_nat:
        out["changed_km2"] = km2(land & (lc != nat))
        LU = J.get("landcover", {}).get("landuse") or {}
        out["landuse_km2"] = LU.get("km2")
        out["landuse_from_km2"] = LU.get("from_km2")
        lu = Z["landuse"] if "landuse" in Z.files else None
        if lu is not None:
            out["unexplained_km2"] = km2(land & (lc != nat) & (lu == 0))
    return out


def market(J: dict, S: dict, Z, res_km: float) -> dict:
    """P7 的镇、邑治、航船、中转站；P7 之前的产物（没有 harbors）按 market.py 现算大泊场。"""
    H = S.get("harbors")
    derived = H is None
    if derived:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import tomllib
        from skyisle_gen.island import market as MK
        mc = tomllib.load(open(Path(__file__).resolve().parents[2] / "config" / "default.toml", "rb"))["island"]["market"]
        g = {k: Z[k] for k in Z.files}
        g["json"] = J
        x0, y0 = J["raster"]["origin_km"]
        km = lambda i, j: [round(x0 + (j + 0.5) * res_km, 3), round(y0 - (i + 0.5) * res_km, 3)]
        _, pad = MK.harbor_pad(g, mc)
        cells = MK.island_cells(g["island_id"], len(J["islands"]))
        cnt = MK.harbor_count(pad, g["island_id"], cells, int(round(float(mc["harbor_window_km"]) / res_km)))
        H = MK.harbor_sites(g, mc, pad, cnt, km, res_km)
    V = {v["id"]: v for v in S["villages"]}
    T = S.get("towns", [])
    lines = S.get("boat_lines", [])
    by_isl = defaultdict(list)
    for h in H:
        by_isl[h["island"]].append(h)

    def near_harbor(t):
        best = None
        for h in by_isl.get(t["island"], []):
            d = math.hypot(h["km"][0] - t["km"][0], h["km"][1] - t["km"][1])
            if best is None or d < best[0]:
                best = (d, h)
        return best
    dh = [near_harbor(t) for t in T]
    dd = np.array([x[0] if x else np.inf for x in dh])
    other = []
    for t in T:
        if "served_other_islands_households" in t:
            other.append(t["served_other_islands_households"])
        else:                                           # P7 之前：按村的 market_town（直线跨岛最近的镇）数
            other.append(sum(v["households"] for v in S["villages"] if v.get("market_town") == t["id"] and v["island"] != t["island"]))
    nl_ = [t.get("n_lines", 0) for t in T]
    no_boat = [t["id"] for t, o, n_ in zip(T, other, nl_) if o > 0 and n_ == 0]
    propped = [t["id"] for t, o in zip(T, other) if t.get("served_households") and o >= 0.5 * t["served_households"] and not t.get("n_lines", 0)]
    seat = next((t for t in T if t.get("seat")), None)
    si = T.index(seat) if seat else None
    cross = sum(1 for v in S["villages"] if v.get("market_town") and v["island"] != next((t["island"] for t in T if t["id"] == v["market_town"]), v["island"]))
    R = S.get("relays")
    if R is None:                                       # P5 / P6：烽火台在 uses 里
        fn = Counter({"烽火": sum(1 for u in S.get("uses", []) if u["kind"] == "烽火台")})
        relay_hh = 0
    else:
        fn = Counter(f for r in R for f in r["functions"])
        relay_hh = sum(r["households"] for r in R)
    L = [ln["length_km"] for ln in lines]
    return {"derived_harbors": derived, "n_harbors": len(H), "harbor_ships": sum(h["ships"] for h in H),
            "harbor_main": sum(1 for h in H if h["island"] == 0),
            "n_towns": len(T), "town_harbor_km": {"median": round(float(np.median(dd)), 2) if dd.size else None,
                                                   "max": round(float(dd.max()), 2) if dd.size and np.isfinite(dd).all() else (None if not dd.size else "∞"),
                                                   "le2": int((dd <= 2.0).sum()), "none_on_island": int((~np.isfinite(dd)).sum())},
            "lines_per_town": {"median": float(np.median(nl_)) if nl_ else 0, "max": max(nl_, default=0), "with_line": sum(1 for x in nl_ if x)},
            "towns_other_island_no_boat": no_boat, "towns_propped_by_other_islands": propped, "villages_cross_island_market": cross,
            "seat": None if seat is None else {"island": seat["island"], "village": seat.get("village"), "farm_hh": seat.get("households_farm"),
                                               "harbor_km": None if not dh[si] else round(dh[si][0], 2), "lines": seat.get("n_lines", 0),
                                               "boat_hh": seat.get("served_boat_households"), "served": seat.get("served_households")},
            "n_lines": len(lines), "line_km": round(sum(L), 1), "line_km_median": round(float(np.median(L)), 1) if L else None,
            "line_km_max": round(max(L), 1) if L else None, "line_villages": sum(len(ln["stops"]) for ln in lines),
            "line_households": sum(ln["households"] for ln in lines), "relay_functions": dict(fn), "n_relays": len(R) if R is not None else sum(fn.values()),
            "relay_households": relay_hh}


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
    print("\n## 人住在哪（总户数里住在各处的占比；工棚、季节住的人算在他们住的村）\n")
    print("| 群 | 总户数 | 主岛 | 主岛以外 ≥ 30 km² 的岛 | < 30 km² 的岛 |")
    print("|---|---|---|---|---|")
    for g in G:
        r = g.get("residence_households")
        if r:
            t = max(1, r["total"])
            print(f"| #{g['node']} | {r['total']} | {r['main'] / t:.1%} | {r['big'] / t:.1%} | {r['small'] / t:.1%} |")
    print("\n## 有人用、没人住的岛（按用途；一岛多用途各记一次）；索桥 / 桥头\n")
    for g in G:
        u = g.get("used_not_lived_by_use") or {}
        print(f"- #{g['node']}：{'、'.join(f'{k} {v}' for k, v in u.items()) or '无'}；索桥 {g.get('n_bridges', 0)} / 桥头 {g.get('n_bridgeheads', 0)}")


def print_works(G: list[dict]) -> None:
    print("\n## 水利（P6：谷口的渠和塘、湿地排成圩田；settlements.json 的 waterworks）\n")
    print("| 群 | 渠首（季节性） | 谷口的渠 km（干渠）| 每 100 km² 陆地 | 灌田 km²（占已垦） | 塘：村塘 / 山塘 / 圩塘 / 堰塘 | 闸：渠首 / 圩 / 排水 | 湿地 km²（改前） | 圩田 km²（占湿地） | 圩 / 片（水田） | 纵浦横塘 / 排水渠 / 圩堤 km |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in G:
        w = g.get("works")
        if not w:
            print(f"| #{g['node']} | — | — | — | — | — | — | {g['wetland_km2']:.1f} | — | — | — |")
            continue
        p, sl = w["ponds"], w["sluices"]
        print(f"| #{g['node']} | {w['n_heads']}（{w['n_heads_seasonal']}） | {w['canal_km']:.0f}（{w['main_canal_km']:.0f}） | {w['canal_km_per_100km2']:.1f} | "
              f"{w['commanded_km2']:.0f}（{w['commanded_share']:.0%}） | {w['n_ponds']}：{p.get('村塘', 0)} / {p.get('山塘', 0)} / {p.get('圩塘', 0)} / {p.get('堰塘', 0)} | "
              f"{w['n_sluices']}：{sl.get('渠首闸', 0)} / {sl.get('圩闸', 0)} / {sl.get('排水闸', 0)} | {w['wetland_km2']:.1f} | "
              f"{w['polder_km2']:.1f}（{w['polder_share']:.0%}） | {w['n_polders']} / {w['n_polder_patches']}（{w['paddy_share']:.0%}） | "
              f"{w['polder_canal_km']:.0f} / {w['drain_km']:.1f} / {w['dike_km']:.0f} |")


def print_market(G: list[dict]) -> None:
    print("\n## 镇、邑治、航船、中转站（P7；P7 之前的大泊场按 market.py 在同一份地形与田上现算）\n")
    print("| 群 | 大泊场（主岛）/ 船 | 镇 | 镇离同岛最近的大泊场 km：中位 / 最大（≤ 2 km 的镇；岛上没有的） | 航船线汇到镇：中位 / 最多（有线的镇） | "
          "有别的岛上的人来赶集却没有航船的镇 | 别的岛上的人过半的镇 | 跨岛赶集的村 | 邑治：岛 / 农户 / 离大泊场 km / 汇几条线 | 航船：条 / 共 km（中位、最长）/ 连几村几户 | 中转站（按功能）/ 常住户 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in G:
        m = g.get("market")
        if not m:
            continue
        th, lp, st = m["town_harbor_km"], m["lines_per_town"], m["seat"] or {}
        fn = "、".join(f"{k} {v}" for k, v in m["relay_functions"].items() if v) or "—"
        print(f"| #{g['node']} | {m['n_harbors']}（{m['harbor_main']}）/ {m['harbor_ships']} | {m['n_towns']} | {th['median']} / {th['max']}（{th['le2']}；{th['none_on_island']}） | "
              f"{lp['median']:g} / {lp['max']}（{lp['with_line']}） | {len(m['towns_other_island_no_boat'])} | {len(m['towns_propped_by_other_islands'])} | "
              f"{m['villages_cross_island_market']} | {st.get('island')} / {st.get('farm_hh')} / {st.get('harbor_km')} / {st.get('lines')} | "
              f"{m['n_lines']} / {m['line_km']:g}（{m['line_km_median']}、{m['line_km_max']}）/ {m['line_villages']} 村 {m['line_households']} 户 | {fn} / {m['relay_households']} |")


def _cell(s) -> str:
    if not s:
        return "—"
    return f"{s['median']} / {s['p90']} / {s['max']}（{s['over']:.0%}，{s['n']}）"


def print_manage(G: list[dict]) -> None:
    print("\n## 水利离村多远（P6b，L30；km：中位 / P90 / 最大（超过 2 km 的几成，处数）；渠段取离村最远的点，圩取格心均值）\n")
    for key, title in (("near", "离最近的村"), ("mgr", "离管它的村（P6b 起记 village）")):
        print(f"**{title}**\n")
        print("| 群 | " + " | ".join(WORKS_KINDS) + " |")
        print("|---|" + "---|" * len(WORKS_KINDS))
        for g in G:
            m = g.get("manage")
            if not m:
                continue
            print(f"| #{g['node']} | " + " | ".join(_cell(m[key].get(k)) for k in WORKS_KINDS) + " |")
        print()
    print("| 群 | 没有管它的村（按类） | 圩最远那格离管它的村 km | 废弃的水利（没人管了） | 村 / 其中圩村 / 挂着圩田的村 | 散户 | 已垦 km² / 额度 | 人口 ⑨ / Σ户×5 |")
    print("|---|---|---|---|---|---|---|---|")
    for g in G:
        m = g.get("manage")
        if not m:
            continue
        un = "、".join(f"{k} {v}" for k, v in m["unmanaged"].items() if v) or "0"
        ab = "、".join(f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}" for k, v in m["abandoned"].items()) or "0"
        fl = g.get("farmland") or {}
        print(f"| #{g['node']} | {un} | {m['polder_far_max_km']} | {ab} | {g.get('n_villages')} / {m['n_polder_villages']} / {m['n_villages_with_polder']} | "
              f"{g.get('n_hamlets')} | {g['cultivated_km2']:.1f} / {fl.get('quota_km2', '—')} | {g.get('pop_polity', 0):.0f} / {g.get('pop_settle', 0):.0f} |")


def print_nature(G: list[dict]) -> None:
    print("\n## 原始地貌与人工地貌（P6b，L31；km²）\n")
    print("| 群 | 原始湿地 | 现状湿地 | 圩田 | 有原始湿地的岛：座数 / 最小的岛 km² | 没人常住的岛：座数 / 陆地 | 其上原始湿地 / 现状湿地 / 已垦 | 地表变了的 | 其中说不清的 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in G:
        x = g.get("nature")
        if not x:
            continue
        print(f"| #{g['node']} | {x['wet_natural_km2']:.1f}{'' if x['has_natural'] else '（推）'} | {x['wet_now_km2']:.1f} | {x['polder_km2']:.1f} | "
              f"{x['wet_islands']} / {x['wet_island_min_km2']} | "
              f"{x['nobody_islands']} / {x['nobody_land_km2']:.0f} | {x['nobody_wet_natural_km2']:.1f} / {x['nobody_wet_now_km2']:.1f} / {x['nobody_cultivated_km2']:.1f} | "
              f"{x.get('changed_km2', '—')} | {x.get('unexplained_km2', '—')} |")
    rows = [g for g in G if (g.get("nature") or {}).get("landuse_km2")]
    if rows:
        names = [k for k in rows[0]["nature"]["landuse_km2"]]
        print("\n人工改造（terrain.npz 的 landuse）：\n")
        print("| 群 | " + " | ".join(names) + " | 开垦的田（旱田 + 梯田 + 渠灌田）原为 |")
        print("|---|" + "---|" * (len(names) + 1))
        for g in rows:
            lu = g["nature"]["landuse_km2"]
            fr = g["nature"].get("landuse_from_km2") or {}
            src = Counter()
            for k in ("开垦的旱田", "梯田", "渠灌田"):
                for kk, v in (fr.get(k) or {}).items():
                    src[kk] += v
            tot = sum(src.values()) or 1.0
            print(f"| #{g['node']} | " + " | ".join(f"{lu[k]:.1f}" for k in names) + " | "
                  + "、".join(f"{k} {v / tot:.0%}" for k, v in sorted(src.items(), key=lambda kv: -kv[1]) if v / tot >= 0.005) + " |")


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
    print_works(G)
    print_market(G)
    print_manage(G)
    print_nature(G)
    if a.json:
        Path(a.json).write_text(json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
