"""⑦ 文明中心（骨架窗内涌现 + 固定 id）、史前扩散、⑦b 地区划分与中心间干线。

适宜度 = 降水 × 稳定气候 × 岛密度 × 岛群陆地^γ（docs/12 §五 ⑦ + docs/02 §六 集雨面）。
不含高度（原则乙）；陆地不是海拔，是集雨面与人口容量。节点 = 岛群（R10）。
铁律自检：本阶段的社会推导不读 height_m。
"""
from __future__ import annotations

import numpy as np

from .. import MODES
from ..graph import CSR, dijkstra
from ..sphere import angdist
from ..weights import lambda_ref, load_directed, mode_weight
from .s02_wind import band_id_of_lat

CENTER_IDS = ["north_west", "north_east", "south"]
CENTER_ZH = {"north_west": "北带西中心", "north_east": "北带东中心", "south": "南带中心"}


def _suitability(ctx, isl, clim, ce):
    dens = isl["density_at"].astype(np.float64)
    dn = np.log(np.maximum(dens, 1e-9))
    dn = (dn - dn.min()) / max(1e-9, dn.max() - dn.min())
    p = clim["precip"].astype(np.float64)
    pn = p / p.max()
    # 岛群规模：群陆地越大，越养得起一个中心（docs/02 §六「一群 = 一水共同体」、
    # §七「土地绝对有限」）。γ = 0 即退回 docs/12 §五 的原式，完全不看面积。
    gamma = float(ctx.section(7)["centers"]["area_exponent"])
    an = np.log(np.maximum(isl["area_km2"].astype(np.float64), 1e-9))
    an = (an - an.min()) / max(1e-9, an.max() - an.min())
    suit = pn * clim["stability"].astype(np.float64) * dn * an ** gamma
    # 候选边上 2 跳邻域平滑
    src, dst = ce["src"], ce["dst"]
    n = dens.size
    hops = int(ctx.section(7)["centers"]["smooth_hops"])
    for _ in range(hops):
        acc = suit.copy()
        cnt = np.ones(n)
        np.add.at(acc, src, suit[dst])
        np.add.at(acc, dst, suit[src])
        np.add.at(cnt, src, 1.0)
        np.add.at(cnt, dst, 1.0)
        suit = acc / cnt
    return suit


def _windows(cfg, bands, lat, lon):
    sk = cfg["skeleton"]
    w = float(sk["center_window_deg"])
    lon_w, lon_e = float(sk["d_lon_west"]), float(sk["d_lon_east"])
    band = band_id_of_lat(lat, bands)
    in_lon = lambda lo, hi: ((lon - lo) % 360.0) <= ((hi - lo) % 360.0)  # noqa: E731
    return {
        "north_west": (band == 1) & in_lon(lon_w - w, lon_w),
        "north_east": (band == 1) & in_lon(lon_e, lon_e + w),
        "south": band == 5,
    }


def run(ctx):
    c7 = ctx.section(7)["centers"]
    isl = ctx.load_npz(3, "islands")
    clim = ctx.load_npz(4, "climate_islands")
    ce = ctx.load_npz(3, "cand_edges")
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    lat, lon = isl["lat"], isl["lon"]
    N = lat.size
    days_per_rad = planet["radius_km"] / planet["day_range_km"]

    suit = _suitability(ctx, isl, clim, ce)

    # ---- 三个骨架窗内的涌现（docs/11 §五 定稿；窗内取 argmax）----
    wins = _windows(ctx.cfg, bands, lat, lon)
    global_max = float(suit.max())
    centers = {}
    for cid in CENTER_IDS:
        m = wins[cid]
        if not m.any():
            raise ValueError(f"文明中心窗 {cid} 内没有岛：请检查 skeleton 与密度配置")
        node = int(np.where(m)[0][np.argmax(suit[m])])
        val = float(suit[node])
        if val < float(c7["tau_c"]) * global_max:
            stats = {"precip_med": float(np.median(clim['precip'][m])),
                     "stability_med": float(np.median(clim['stability'][m])),
                     "n_in_window": int(m.sum())}
            raise ValueError(
                f"窗 {cid} 内最大适宜度 {val:.3f} < τ_c×全局最大 {global_max:.3f}（原则庚：推不出即错）。"
                f"窗内统计：{stats}。请调整降水/密度/骨架参数")
        centers[cid] = {"node": node, "lat": round(float(lat[node]), 3),
                        "lon": round(float(lon[node]), 3), "suitability": round(val, 4),
                        "zh": CENTER_ZH[cid]}

    # ---- 无约束次级极大（供 ⑧ 次级起源；诊断对照）----
    src, dst = ce["src"], ce["dst"]
    nb_max = suit.copy()
    np.maximum.at(nb_max, src, suit[dst])
    np.maximum.at(nb_max, dst, suit[src])
    is_peak = suit >= nb_max - 1e-15
    peak_nodes = np.where(is_peak)[0]
    peak_nodes = peak_nodes[np.argsort(-suit[peak_nodes], kind="stable")]
    xyz = isl["xyz"]
    min_sep = float(c7["secondary_min_sep_days"])
    chosen: list[int] = []
    main_nodes = [centers[cid]["node"] for cid in CENTER_IDS]
    for p in peak_nodes:
        if len(chosen) >= int(c7["n_secondary"]):
            break
        ok = all(angdist(xyz[p], xyz[q]) * days_per_rad >= min_sep for q in chosen + main_nodes)
        if ok:
            chosen.append(int(p))

    # ---- 史前扩散：抱石而渡，顺风单向（docs/11 §七）----
    g = load_directed(ctx)
    csr = CSR(N, g["src_d"], g["dst_d"])
    exp_w = float(c7["prehist_wind_exponent"])
    # 物理成本已含 w^1；再乘 w^(exp-1) 需要 w 本身 —— 用 cost/dist 近似风因子（storm、climb 也被强化，可接受）
    ce_dist = np.concatenate([ce["dist_days"], ce["dist_days"]])
    with np.errstate(divide="ignore", invalid="ignore"):
        wf = np.where(ce_dist > 0, g["cost"] / ce_dist, 1.0)
    w_pre = ce_dist * wf ** exp_w
    origin_cfg = c7["origin"]
    if origin_cfg == "auto":
        band_i = band_id_of_lat(lat, bands)
        lon_e = float(ctx.cfg["skeleton"]["d_lon_east"])
        m = (band_i == 1) & (((lon - lon_e) % 360.0) <= 90.0)
        origin_node = int(np.where(m)[0][np.argmax(suit[m])]) if m.any() else centers["north_east"]["node"]
    else:
        origin_node = centers[origin_cfg]["node"]
    dist_pre, pred_pre, pred_edge_pre = dijkstra(csr, w_pre, [origin_node])
    n_unreach = int((~np.isfinite(dist_pre)).sum())
    if n_unreach:
        raise ValueError(f"史前扩散有 {n_unreach} 个岛不可达 —— 违反原则己（处处有人）。图连通性有误")
    # 谱系：沿树在带界变化处切分
    band_i = band_id_of_lat(lat, bands)
    lineage = np.full(N, -1, dtype=np.int64)
    order = np.argsort(dist_pre, kind="stable")
    next_lineage = 0
    for u in order:
        p = pred_pre[u]
        if p < 0:
            lineage[u] = next_lineage
            next_lineage += 1
        elif band_i[u] != band_i[p]:
            lineage[u] = next_lineage
            next_lineage += 1
        else:
            lineage[u] = lineage[p]
    arrival_yr = dist_pre * float(c7["yr_per_day"])

    # ---- ⑦b 地区划分：种子 = 中心 ∪ 大枢纽 ∪ 最远点采样；日常模式权重（障碍自然成为边界）----
    lam = lambda_ref(ctx.cfg)
    w_daily = mode_weight(g["cost_m"], g["L"], "daily", lam["daily"])
    # 对称化（地区归属与方向无关）
    E2 = w_daily.size // 2
    w_sym_und = np.minimum(w_daily[:E2], w_daily[E2:])
    w_sym = np.concatenate([w_sym_und, w_sym_und])
    hubs = [h["node"] for h in ctx.load_json(6, "hubs")["hubs"]]
    n_regions = int(ctx.cfg["s07"]["regions"]["n_regions"])
    seeds = list(dict.fromkeys(main_nodes + hubs[: max(0, min(12, n_regions - 3))]))

    def multi_dist(seed_list):
        d, pn, _pe = dijkstra(csr, w_sym, seed_list)
        return d, pn

    d_min, _ = multi_dist(seeds)
    while len(seeds) < n_regions:
        cand = int(np.argmax(np.where(np.isfinite(d_min), d_min, np.inf)))
        if not np.isfinite(d_min[cand]):
            cand = int(np.argmax(~np.isfinite(d_min)))  # 不可达节点优先成为种子
        if cand in seeds:
            break
        seeds.append(cand)
        d_new, _, _ = dijkstra(csr, w_sym, [cand])
        d_min = np.minimum(d_min, d_new)
    def assign(seed_list):
        d, pn, _ = dijkstra(csr, w_sym, seed_list)
        lab = np.full(N, -1, dtype=np.int64)
        s_index = {s: i for i, s in enumerate(seed_list)}
        for u in np.argsort(d, kind="stable"):
            p = pn[u]
            lab[u] = s_index.get(int(u), lab[p] if p >= 0 else -1)
        return d, lab

    d_final, region = assign(seeds)
    # 超大地区分裂：信风带的地区不应吞掉上千岛（docs/11 分辨率：地区是第二层单位）
    max_isl = int(ctx.cfg["s07"]["regions"].get("max_region_islands", 250))
    max_regions = int(ctx.cfg["s07"]["regions"].get("max_regions", 150))
    while len(seeds) < max_regions:
        sizes = np.bincount(region[region >= 0], minlength=len(seeds))
        big = int(np.argmax(sizes))
        if sizes[big] <= max_isl:
            break
        members = np.where(region == big)[0]
        far = members[int(np.argmax(np.where(np.isfinite(d_final[members]),
                                             d_final[members], -1.0)))]
        if int(far) in seeds:
            break
        seeds.append(int(far))
        d_new, _, _ = dijkstra(csr, w_sym, [int(far)])
        take = d_new < d_final
        region[take] = len(seeds) - 1
        d_final = np.minimum(d_final, d_new)
    # 兜底：不可达节点（日常模式图上的孤立分量且未被选为种子）并入最近种子
    bad = np.where(region < 0)[0]
    for u in bad:
        dd = angdist(xyz[u], xyz[np.array(seeds)])
        region[u] = int(np.argmin(dd))

    # ---- 中心间干线（⑦b：仅展示与九格表用，不回写 ⑥）----
    envoy_i = MODES.index("envoy")
    w_env = np.where(g["perm_d"][:, envoy_i] > 0, g["cost"], np.inf)
    trunks = {}
    for a in CENTER_IDS:
        d, pn, _pe = dijkstra(csr, w_env, [centers[a]["node"]])
        for b in CENTER_IDS:
            if a >= b:
                continue
            tgt = centers[b]["node"]
            if not np.isfinite(d[tgt]):
                trunks[f"{a}->{b}"] = {"cost_days": None, "path": []}
                continue
            path = []
            u = tgt
            while u >= 0:
                path.append(int(u))
                u = int(pn[u])
            trunks[f"{a}->{b}"] = {"cost_days": round(float(d[tgt]), 1), "path": path[::-1]}

    ctx.save_npz(7, "prehist", dist_pre=dist_pre, pred=pred_pre,
                 lineage=lineage, arrival_yr=arrival_yr.astype(np.float32))
    ctx.save_npz(7, "regions", region=region, suitability=suit.astype(np.float32))
    ctx.save_json(7, "centers", {
        "centers": centers,
        "origin_node": origin_node,
        "secondary_peaks": [{"node": p, "lat": round(float(lat[p]), 3),
                             "lon": round(float(lon[p]), 3), "suitability": round(float(suit[p]), 4)}
                            for p in chosen],
        "region_seeds": [int(s) for s in seeds],
        "center_trunks": {k: {"cost_days": v["cost_days"], "n_nodes": len(v["path"]),
                              "path": v["path"]} for k, v in trunks.items()},
    })
    return {"centers": {cid: (centers[cid]["lat"], centers[cid]["lon"]) for cid in CENTER_IDS},
            "n_lineages": int(next_lineage), "n_regions": len(seeds),
            "trunk_nw_ne_days": trunks.get("north_east->north_west", trunks.get("north_west->north_east", {})).get("cost_days")}
