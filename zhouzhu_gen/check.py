"""验收（docs/12 第八节七条现象）+ 铁律自检（docs/01）+ 骨架一致性（docs/11，只报警）。

退出码：2 = 硬项（铁律）失败；1 = 软项（七现象）失败；0 = 全过。
每项附最差样例与复查命令。阈值全部在 config [check]。
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import MODES
from .culture import World
from .graph import CSR, dijkstra, betweenness_sampled
from .stages.s05_barriers import REGIONAL_ORDER
from .weights import lambda_ref, load_directed


def _spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


class Report:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, id_, name, value, threshold, ok, hard=False, worst=None, viz=None, note=None):
        self.items.append({"id": id_, "name": name, "value": value, "threshold": threshold,
                           "pass": bool(ok), "hard": hard, "worst": worst or [],
                           "viz_cmd": viz, "note": note})

    def exit_code(self):
        if any(not i["pass"] and i["hard"] for i in self.items):
            return 2
        if any(not i["pass"] and not i["hard"] for i in self.items):
            return 1
        return 0


# ---------------------------------------------------------------- P1
def check_p1(w: World, cfg, rep: Report):
    c = cfg["check"]
    ce = w.cand_edges
    src, dst = ce["src"], ce["dst"]
    free = w.free_edges(float(c["adj_cost_days"]), float(c["adj_perm_min"]))
    if free.any():
        d_free = w.tv_distance(src[free], dst[free])
        med, p95 = float(np.median(d_free)), float(np.quantile(d_free, 0.95))
    else:
        med = p95 = float("nan")
    rep.add("P1a", "相邻几乎相同（无障碍短边的文化距离）",
            {"median": round(med, 4), "p95": round(p95, 4), "n_edges": int(free.sum())},
            {"median<": c["adj_median_max"], "p95<": c["adj_p95_max"]},
            med < float(c["adj_median_max"]) and p95 < float(c["adj_p95_max"]),
            viz="zhouzhu viz distance <节点>")

    # P1b（硬项 = 铁律五的代码化）：任何边的差异必须由该边的成本与通过率解释
    E = src.size
    d_all = w.tv_distance(src, dst)
    cost = np.maximum(w.routes["cost"][:E], w.routes["cost"][E:])
    L = w.edge_max_L()
    L_cap = np.minimum(np.where(np.isfinite(L), L, 13.8), 13.8).max(axis=1)
    dh = cfg["s08"]["half_distance_days"]
    lam_max = np.log(2.0) / min(float(dh[m][0]) for m in MODES)
    bound = float(c["edge_bound_a"]) + float(c["edge_bound_b"]) * (lam_max * cost + L_cap)
    viol = d_all > bound
    worst_idx = np.argsort(-(d_all - bound), kind="stable")[:20]
    worst = [{"edge": int(i), "src": int(src[i]), "dst": int(dst[i]),
              "D": round(float(d_all[i]), 3), "bound": round(float(bound[i]), 3)}
             for i in worst_idx[:5]]
    rep.add("P1b", "连续场硬约束：D(e) ≤ a + b·(λ_max·cost + max_m L_m)【铁律五】",
            {"violations": int(viol.sum()), "max_excess": round(float((d_all - bound).max()), 4)},
            "违规数 = 0", int(viol.sum()) == 0, hard=True, worst=worst,
            viz="zhouzhu probe edge <src> <dst>")


# ---------------------------------------------------------------- P2
def check_p2(w: World, cfg, rep: Report, calibrate=False):
    c = cfg["check"]
    g = load_directed(w.ctx)
    N = w.islands["lat"].size
    csr = CSR(N, g["src_d"], g["dst_d"])
    trade_i = MODES.index("trade")
    wgt = np.where(g["perm_d"][:, trade_i] > 0, g["cost"], np.inf)
    ray_min = float(c["ray_min_days"])
    k_rays = int(c["rays_per_center"])
    spearmans, backsteps = [], []
    calib_rows = []
    for cid, cinfo in w.centers["centers"].items():
        o = int(cinfo["node"])
        dist, pred, _ = dijkstra(csr, wgt, [o])
        far = np.where(np.isfinite(dist) & (dist >= ray_min))[0]
        if far.size == 0:
            continue
        far = far[np.argsort(dist[far], kind="stable")]
        targets = far[np.linspace(0, far.size - 1, min(k_rays, far.size)).astype(int)]
        for t in targets:
            path = []
            u = int(t)
            while u >= 0:
                path.append(u)
                u = int(pred[u])
            path = path[::-1]
            nodes = np.array(path)
            dd = w.tv_distance(np.full(nodes.size, o), nodes)
            cc = dist[nodes]
            spearmans.append(_spearman(cc, dd))
            steps = np.diff(dd)
            backsteps.append(float((steps < -0.02).mean()) if steps.size else 0.0)
        if calibrate:
            for lo, hi in ((2, 4), (8, 12), (20, 30)):
                m = np.isfinite(dist) & (dist >= lo) & (dist <= hi)
                if m.any():
                    dm = w.tv_distance(np.full(int(m.sum()), o), np.where(m)[0])
                    calib_rows.append({"center": cid, "days": f"{lo}-{hi}",
                                       "D_median": round(float(np.median(dm)), 3)})
    sp = float(np.nanmean(spearmans)) if spearmans else float("nan")
    bs = float(np.mean(backsteps)) if backsteps else float("nan")
    rep.add("P2", "走远了才明显不同（沿航线单调累积）",
            {"spearman_mean": round(sp, 3), "backstep_frac": round(bs, 3),
             "n_rays": len(spearmans)},
            {"spearman>": c["spearman_min"], "backstep<": c["backstep_frac_max"]},
            sp > float(c["spearman_min"]) and bs < float(c["backstep_frac_max"]),
            note=("校准（对照 docs/04「走三天/走十天」）：" + json.dumps(calib_rows, ensure_ascii=False))
            if calibrate else None)


# ---------------------------------------------------------------- P3
def check_p3(w: World, cfg, rep: Report):
    c = cfg["check"]
    theta = float(c["iso_theta"])
    iso = w.isogloss_edges(theta)
    ce = w.cand_edges
    src, dst = ce["src"], ce["dst"]
    N = w.islands["lat"].size
    E = src.size
    traits = w.traits
    nonempty = [t["index"] for t in traits if iso[t["index"]].any()]
    ratio_thr = float(c["lambda_ratio_distinct"])

    incid = [[] for _ in range(N)]
    for e in range(E):
        incid[src[e]].append(e)
        incid[dst[e]].append(e)

    def expand(mask):
        nodes = np.unique(np.concatenate([src[mask], dst[mask]])) if mask.any() else np.array([], dtype=int)
        out = np.zeros(E, dtype=bool)
        for nd in nodes:
            out[incid[nd]] = True
        return out

    exp_cache = {ti: expand(iso[ti]) for ti in nonempty}
    free = w.free_edges(float(c["adj_cost_days"]), float(c["adj_perm_min"]))
    jac, jac_free = [], []
    pairs = [(a, b) for i, a in enumerate(nonempty) for b in nonempty[i + 1:]]
    stride = max(1, len(pairs) // 3000)
    for a, b in pairs[::stride]:
        ta, tb = traits[a], traits[b]
        distinct = (ta["mode"] != tb["mode"] or ta["origin_node"] != tb["origin_node"]
                    or max(ta["lambda"], tb["lambda"]) / max(1e-9, min(ta["lambda"], tb["lambda"])) > ratio_thr)
        if not distinct:
            continue
        A, B = iso[a], iso[b]
        union = (A | B).sum()
        if union == 0:
            continue
        inter = max((A & exp_cache[b]).sum(), (B & exp_cache[a]).sum())
        jac.append(inter / union)
        Af, Bf = A & free, B & free
        uf = (Af | Bf).sum()
        if uf > 0:
            jac_free.append(max((Af & exp_cache[b] & free).sum(), (Bf & exp_cache[a] & free).sum()) / uf)
    jm = float(np.mean(jac)) if jac else float("nan")
    jf = float(np.mean(jac_free)) if jac_free else float("nan")
    bundle = iso.sum(axis=0)
    barrier_score = 1.0 - w.perm["perm"].mean(axis=1)
    sp = _spearman(bundle, barrier_score)
    has_single = bool((bundle == 1).any())
    # 聚束（b ≥ 3）边的平均障碍分应显著高于全体（docs/02 §五 推论二）。
    # 全局秩相关会被自由区的「特征互斥型」同言线（docs/12 §二：合法来源二）稀释，只作参考。
    heavy = bundle >= 3
    ratio = (float(barrier_score[heavy].mean() / max(barrier_score.mean(), 1e-9))
             if heavy.any() else float("nan"))
    ok = (jm < float(c["jaccard_mean_max"]) and (np.isnan(jf) or jf < float(c["jaccard_free_mean_max"]))
          and ratio >= float(c["p3_bundle_ratio_min"]) and has_single)
    rep.add("P3", "同言线互不重合；聚束处 = 障碍所在",
            {"jaccard_mean": round(jm, 3), "jaccard_free_mean": round(jf, 3) if jac_free else None,
             "bundle_barrier_ratio": round(ratio, 2),
             "bundle_vs_barrier_spearman_ref": round(sp, 3), "has_b1_edges": has_single,
             "n_pairs": len(jac)},
            {"jaccard<": c["jaccard_mean_max"], "free<": c["jaccard_free_mean_max"],
             "bundle_ratio>=": c["p3_bundle_ratio_min"]},
            ok, viz="zhouzhu viz isogloss")


# ---------------------------------------------------------------- P4
def check_p4(w: World, cfg, rep: Report):
    c = cfg["check"]
    sk = w.ctx.cfg["skeleton"]
    isl = w.islands
    lat, lon = isl["lat"], isl["lon"]
    planet = w.ctx.load_json(1, "planet")["bands"]
    g_info = w.ctx.load_json(2, "bands")["G"]
    node_flow = w.routes["node_flow"]
    lon_w, lon_e = float(sk["d_lon_west"]), float(sk["d_lon_east"])
    in_band = (lat >= planet["eq_storm_top_deg"]) & (lat <= planet["trades_top_deg"])
    margin = 20.0
    west = in_band & (((lon_w - lon) % 360.0) <= margin)
    east = in_band & (((lon - lon_e) % 360.0) <= margin)
    from .sphere import latlon_to_xyz, angdist
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    near_g = np.degrees(angdist(isl["xyz"], g_xyz[None, :])) <= 2.5 * float(g_info["radius_deg"])
    west &= ~near_g
    east &= ~near_g

    def top_nodes(mask, k=40):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return idx
        return idx[np.argsort(-node_flow[idx], kind="stable")][:k]

    wn, en = top_nodes(west), top_nodes(east)
    if wn.size == 0 or en.size == 0:
        rep.add("P4", "文书通、口音不通", "无可用节点对", "-", False)
        return
    pairs = []
    for a in wn:
        b = en[np.argmin(np.abs(lat[en] - lat[a]))]
        pairs.append((int(a), int(b)))
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    d_daily = w.tv_distance(ii, jj, mode="daily")
    d_envoy = w.tv_distance(ii, jj, mode="envoy")
    envoy_slots = [s for s in w.slots if w.slot_mode(s) == "envoy"]
    share_min = float(c["p4_share_min"])
    shared_ok = np.zeros(len(pairs), dtype=bool)
    for s in envoy_slots:
        p, _ = w.shares(s)
        shared_ok |= ((p[:-1, ii] >= share_min) & (p[:-1, jj] >= share_min)).any(axis=0)
    guard = d_daily >= float(c["p4_daily_min"])
    cond = ((d_envoy <= float(c["p4_envoy_ratio"]) * d_daily)
            & (d_daily - d_envoy >= float(c["p4_gap_min"])) & shared_ok)
    frac = float(cond[guard].mean()) if guard.any() else float("nan")
    worst = [{"pair": pairs[k], "D_daily": round(float(d_daily[k]), 3),
              "D_envoy": round(float(d_envoy[k]), 3)}
             for k in np.argsort(~cond, kind="stable")[:3]]
    rep.add("P4", "文书通、口音不通（隔 D 而有官方航路的两地）",
            {"pass_frac": round(frac, 3), "n_pairs_guarded": int(guard.sum()),
             "D_daily_median": round(float(np.median(d_daily[guard])) if guard.any() else -1, 3),
             "D_envoy_median": round(float(np.median(d_envoy[guard])) if guard.any() else -1, 3)},
            {"pass_frac>": c["p4_pass_frac"]},
            (not np.isnan(frac)) and frac > float(c["p4_pass_frac"]),
            worst=worst, viz="zhouzhu viz distance <西侧节点>")


# ---------------------------------------------------------------- P5
def check_p5(w: World, cfg, rep: Report):
    c = cfg["check"]
    isl = w.islands
    lat, lon = isl["lat"], isl["lon"]
    wind = w.ctx.load_npz(2, "wind")
    from .sphere import grid_interp
    reach = w.fields["reach"].astype(np.float64)
    good = 0
    total = 0
    for t in w.traits:
        if t["mode"] not in ("daily", "trade") or t["kind"] == "local":
            continue
        o = t["origin_node"]
        u_o = float(grid_interp(wind["u"].astype(np.float64), wind["lats"], wind["lons"],
                                lat[o], lon[o]))
        if abs(u_o) < 1.0:
            continue
        r = reach[t["index"]]
        dlon = ((lon - lon[o] + 180.0) % 360.0) - 180.0
        if r.sum() <= 0:
            continue
        delta = float((r * dlon).sum() / r.sum())
        total += 1
        if delta * np.sign(u_o) > 0:
            good += 1
    frac = good / total if total else float("nan")

    # (b) 同带内上下风特征对：上风的传得过去，下风的传不回来
    ratios = []
    for m in ("daily", "trade"):
        ts = [t for t in w.traits if t["mode"] == m and t["kind"] in ("main", "sub")]
        for i, ta in enumerate(ts):
            for tb in ts[i + 1:]:
                oa, ob = ta["origin_node"], tb["origin_node"]
                if np.sign(lat[oa]) != np.sign(lat[ob]):
                    continue
                if not (5.0 <= abs(((lon[oa] - lon[ob] + 180) % 360) - 180) <= 60.0):
                    continue
                dlon_ab = ((lon[ob] - lon[oa] + 180) % 360) - 180
                # 信风带（东风）：下风 = 西。up = 东侧者
                up, dn = (ta, tb) if dlon_ab < 0 else (tb, ta)
                s_up_at_dn = w.fields["strength"][up["index"], dn["origin_node"]]
                s_dn_at_up = w.fields["strength"][dn["index"], up["origin_node"]]
                if s_dn_at_up > 1e-9:
                    ratios.append(float(s_up_at_dn / s_dn_at_up))
                elif s_up_at_dn > 1e-6:
                    ratios.append(float(c["p5_strength_ratio"]) * 10)
    med_ratio = float(np.median(ratios)) if ratios else float("nan")
    ok = (not np.isnan(frac)) and frac >= float(c["p5_centroid_frac"]) \
        and (not np.isnan(med_ratio)) and med_ratio > float(c["p5_strength_ratio"])
    rep.add("P5", "单向传播（顺风易、逆风难）",
            {"centroid_downwind_frac": round(frac, 3) if total else None, "n_traits": total,
             "updown_strength_ratio_median": round(med_ratio, 2) if ratios else None,
             "n_pairs": len(ratios)},
            {"frac>=": c["p5_centroid_frac"], "ratio>": c["p5_strength_ratio"]},
            ok)


# ---------------------------------------------------------------- P6
def check_p6(w: World, cfg, rep: Report):
    c = cfg["check"]
    hubs = [h for h in w.hubs["hubs"] if h["near_g"]]
    if not hubs:
        rep.add("P6", "改道产生贸易枢纽", "G 邻域无枢纽", "-", False,
                viz="zhouzhu viz routes")
        return
    H = np.array(sorted(h["node"] for h in hubs))
    # 起源圈份额：min/max 平衡度
    circle_of = {}
    for t in w.traits:
        org = t["origin"]
        if t["kind"] == "main":
            circle_of[t["index"]] = org
        elif t["kind"] == "sub":
            circle_of[t["index"]] = org.split(":")[1]
    N = w.islands["lat"].size
    bal_all = np.zeros(N)
    n_slots_used = np.zeros(N)
    # 混合度只看商旅与使节槽位：中转岛「两头通吃」的是货物与文书，市井口音仍是本地的
    # （撒马尔罕亦然）。daily 槽位在中转岛本就应保持本地化。
    mix_slots = [s for s in w.slots if w.slot_mode(s) in ("trade", "envoy")]
    for s in mix_slots:
        p, _ = w.shares(s)
        rows = w.slot_rows()[s]
        sh_nw = np.zeros(N)
        sh_ne = np.zeros(N)
        for k, ti in enumerate(rows):
            circ = circle_of.get(ti)
            if circ == "north_west":
                sh_nw += p[k]
            elif circ == "north_east":
                sh_ne += p[k]
        mx = np.maximum(sh_nw, sh_ne)
        mn = np.minimum(sh_nw, sh_ne)
        used = mx > 0.05
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where(used, mn / np.maximum(mx, 1e-12), 0.0)
        bal_all += b
        n_slots_used += used
    bal = np.where(n_slots_used > 0, bal_all / np.maximum(n_slots_used, 1), 0.0)
    mean_h = float(bal[H].mean())
    pct = float(np.percentile(bal, float(c["p6_pctile"])))

    # 反事实：从未有过 G 的世界（通过率与风暴成本都还原），H 的流量应大幅下降
    # —— 这正是「改道型」的机器判定：不阻断总量，只改变路径（docs/12 §四）
    g = load_directed(w.ctx)
    pm = w.perm
    perm_ng = np.concatenate([pm["perm_no_g"], pm["perm_no_g"]], axis=0)
    trade_i = MODES.index("trade")
    wgt = np.where(perm_ng[:, trade_i] > 0, w.routes["cost_no_g"], np.inf)
    csr = CSR(N, g["src_d"], g["dst_d"])
    sources = w.routes["betweenness_sources"]
    r6 = w.ctx.cfg["s06"]["routes"]
    flow_ng = betweenness_sampled(csr, wgt, g["cost"].size, sources,
                                  c_min=float(r6["betweenness_c_min_days"]))
    node_flow_ng = np.zeros(N)
    np.add.at(node_flow_ng, g["src_d"], flow_ng)
    np.add.at(node_flow_ng, g["dst_d"], flow_ng)
    node_flow_ng *= 0.5
    node_flow = w.routes["node_flow"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        drops = 1.0 - node_flow_ng[H] / np.maximum(node_flow[H], 1e-9)
    # 「它们的财富完全依赖于绕行本身」（docs/11 §六）——反事实中应有相当一部分
    # 中转枢纽坍缩。门户型节点（直线航路也要经过它）不坍缩是真实现象：
    # 绿洲门户挺过风暴消失，纯绕道驿站死掉——这正是丝路的结构。
    collapse_frac = float((drops >= float(c["p6_flow_drop"])).mean())
    # 门槛 = 绝对混合度 + 反事实坍缩。相对分位（枢纽 vs 全节点 P75）只报告不判：
    # 它对枢纽恰好落在环的哪一侧过于敏感，而 docs/12 #6 的断言是绝对的（「应是混合体」）
    ok = (mean_h >= float(c["p6_bal_min"])
          and collapse_frac >= float(c["p6_collapse_frac"]))
    rep.add("P6", "改道产生贸易枢纽（G 邻域 = 两文明圈的混合体，且流量依赖绕行）",
            {"mix_balance_mean": round(mean_h, 3), "all_pctile": round(pct, 3),
             "collapse_frac_without_G": round(collapse_frac, 3),
             "flow_drop_median_without_G": round(float(np.median(drops)), 3),
             "n_hub": int(H.size)},
            {"bal>=": c["p6_bal_min"], "collapse_frac>=": c["p6_collapse_frac"],
             "collapsed_hub_drop>=": c["p6_flow_drop"]},
            ok, viz="zhouzhu viz routes")


# ---------------------------------------------------------------- P7
def check_p7(w: World, cfg, rep: Report):
    c = cfg["check"]
    reach = w.fields["reach"]
    strength = w.fields["strength"]
    adopt = w.fields["adopt"]
    conflict_by = w.fields["conflict_by"]
    region = w.regions["region"]
    traits = w.traits
    # 「家乡的器物用得好好的，但家乡的规矩被明令不用」——同圈的低阻力器物作伴即可：
    # 高阻力特征含次级起源（京畿之地拒绝乡下变体、变体区拒绝京畿正统，都是这个现象）
    def circle(t):
        if t["kind"] == "main":
            return t["origin"]
        if t["kind"] == "sub":
            return t["origin"].split(":")[1]
        return None
    lows = [t for t in traits if t["resistance"] <= 0.2 and t["kind"] == "main"]
    highs = [t for t in traits if t["resistance"] >= 0.7 and circle(t) is not None]
    found_nodes = set()
    examples = []
    region_hit = {}
    for th in highs:
        for tl in lows:
            if circle(tl) != circle(th):
                continue
            m = ((strength[tl["index"]] >= float(c["p7_strength_low_min"]))
                 & (reach[th["index"]] >= float(c["p7_reach_high_min"]))
                 & (adopt[th["index"]] <= float(c["p7_adopt_max"])))
            # 守卫 1：挡人的不是同起源的特征
            blocker = conflict_by[th["index"]]
            same_origin = np.zeros_like(m)
            for bt in np.unique(blocker[m]):
                if bt >= 0 and traits[bt]["origin_node"] == th["origin_node"]:
                    same_origin |= (blocker == bt) & m
            m &= ~same_origin
            # 守卫 2：拒绝是地点性的（该特征另有采纳良好的地方）
            if not (adopt[th["index"]] > 0.9).any():
                continue
            nodes = np.where(m)[0]
            for nd in nodes:
                found_nodes.add(int(nd))
                region_hit.setdefault(int(region[nd]), set()).add(int(nd))
            if nodes.size and len(examples) < 10:
                nd = int(nodes[0])
                bt = int(conflict_by[th["index"], nd])
                examples.append({
                    "node": nd, "trait_rejected": th["id"],
                    "reach": round(float(reach[th["index"], nd]), 3),
                    "adopt": round(float(adopt[th["index"], nd]), 3),
                    "blocked_by": traits[bt]["id"] if bt >= 0 else "（本地自有）",
                    "companion_artifact": tl["id"],
                })
    region_sizes = np.bincount(region)
    region_cov = any(len(v) >= float(c["p7_region_frac"]) * region_sizes[k]
                     for k, v in region_hit.items() if region_sizes[k] > 0)
    ok = len(found_nodes) >= int(c["p7_min_nodes"]) and region_cov
    rep.add("P7", "传到了但不要（reach 高而 adopt 低）",
            {"n_nodes": len(found_nodes), "region_coverage": region_cov},
            {"n_nodes>=": c["p7_min_nodes"], "某地区覆盖>=": c["p7_region_frac"]},
            ok, worst=examples[:5], viz="zhouzhu probe trait <trait_id> --node <节点>")


# ---------------------------------------------------------------- 铁律
def check_ironlaws(w: World, cfg, rep: Report):
    ctx = w.ctx
    # 己：处处有人
    pre = ctx.load_npz(7, "prehist")
    all_reached = bool(np.isfinite(pre["dist_pre"]).all())
    share = w.fields["share"]
    sums_ok = True
    for s in w.slots:
        p, _ = w.shares(s)
        tot = p.sum(axis=0)
        if not np.allclose(tot, 1.0, atol=1e-4) or np.isnan(p).any():
            sums_ok = False
    region_ok = bool((w.regions["region"] >= 0).all())
    perm = w.perm["perm"]
    ce = w.cand_edges
    N = w.islands["lat"].size
    has_edge = np.zeros(N, dtype=bool)
    alive = perm.max(axis=1) > 0
    has_edge[ce["src"][alive]] = True
    has_edge[ce["dst"][alive]] = True
    rep.add("IL-ji", "原则己：世界的每个角落都有人（史前扩散全覆盖、槽位归一、地区归属、连通）",
            {"prehist_all_reached": all_reached, "share_sums_ok": sums_ok,
             "all_in_region": region_ok, "all_connected": bool(has_edge.all())},
            "全部为 true", all_reached and sums_ok and region_ok and bool(has_edge.all()),
            hard=True)

    # 乙：高度不派生社会（静态源码断言）
    import pathlib
    pkg = pathlib.Path(__file__).parent
    social_files = ["stages/s07_centers.py", "stages/s08_diffusion.py", "ninegrid.py"]
    height_hits = []
    for f in social_files:
        p = pkg / f
        if p.exists() and re.search(r"\[[\"']height_m[\"']\]", p.read_text(encoding="utf-8")):
            height_hits.append(f)
    rep.add("IL-yi", "原则乙：地理不决定贵贱（社会推导模块不读 height_m）",
            {"files_reading_height": height_hits}, "空列表", not height_hits, hard=True)

    # 铁律一/二/甲：浮石与地质不进任何计算（产物字段白名单）
    forbidden = re.compile(r"floatstone|浮石|geolog|uplift|mana|magic", re.IGNORECASE)
    bad_keys = []
    for idx, name in [(3, "islands"), (5, "perm"), (6, "routes"), (8, "fields")]:
        for k in ctx.load_npz(idx, name):
            if forbidden.search(k):
                bad_keys.append(f"s{idx:02d}/{name}:{k}")
    rep.add("IL-t12", "铁律一/二/原则甲：浮石与地质不参与任何计算",
            {"forbidden_keys": bad_keys}, "空列表", not bad_keys, hard=True)

    # 铁律五：文化产物中无离散标签（share 为浮点连续场）
    ok_types = w.fields["share"].dtype.kind == "f" and w.fields["strength"].dtype.kind == "f"
    rep.add("IL-t5", "铁律五：文化是连续场（无整数文化标签图层；硬边界由 P1b 逐边保证）",
            {"fields_are_float": bool(ok_types)}, "true", bool(ok_types), hard=True)


# ---------------------------------------------------------------- 骨架一致性（只报警）
def check_skeleton(w: World, cfg, rep: Report):
    tol = float(cfg["check"]["skeleton_tol"])
    ctx = w.ctx
    g = load_directed(ctx)
    N = w.islands["lat"].size
    csr = CSR(N, g["src_d"], g["dst_d"])
    lam = lambda_ref(ctx.cfg)
    phis = None
    # 每个区域障碍：两侧代表点之间最优穿越路径的 Π perm ↔ 配置矩阵
    from .stages.s05_barriers import node_phi
    planet = ctx.load_json(1, "planet")["bands"]
    isl = w.islands
    phis = node_phi(ctx.cfg, planet, isl["lat"], isl["lon"])
    barriers_cfg = ctx.cfg["s05"]["barriers"]
    rows = {}
    all_ok = True
    for bid in REGIONAL_ORDER:
        phi = phis[bid]
        side0 = np.where(np.nan_to_num(phi, nan=-1) == 0.0)[0]
        side1 = np.where(np.nan_to_num(phi, nan=-1) == 1.0)[0]
        if side0.size == 0 or side1.size == 0:
            rows[bid] = "无两侧节点"
            continue
        # 取贴着障碍两侧的最近节点对，隔离障碍本身的贡献（路途上的局部因子另算）
        xyz = w.islands["xyz"]
        a = xyz[side0]
        b = xyz[side1]
        best_pairs = []
        for i0 in range(0, side0.size, 512):
            dots = a[i0:i0 + 512] @ b.T
            k = np.argmax(dots, axis=1)
            v = dots[np.arange(dots.shape[0]), k]
            for j in range(v.size):
                best_pairs.append((float(-v[j]), int(side0[i0 + j]), int(side1[k[j]])))
        best_pairs.sort()
        s0 = [p[1] for p in best_pairs[:3]]
        s1 = [p[2] for p in best_pairs[:3]]
        row = {}
        for mi, m in enumerate(MODES):
            target_p = float(barriers_cfg[bid]["permeability"][m])
            wgt = 0.001 * g["cost"] + np.where(g["perm_d"][:, mi] > 0,
                                               -np.log(np.maximum(g["perm_d"][:, mi], 1e-300)),
                                               np.inf)
            d, _, _ = dijkstra(csr, wgt, [int(x) for x in s0])
            best = float(np.min(d[s1]))
            eff = float(np.exp(-best)) if np.isfinite(best) else 0.0
            row[m] = {"effective": round(eff, 3), "config": target_p,
                      "ok": abs(eff - target_p) <= max(tol, 0.03 if target_p <= 0.05 else tol)}
            if not row[m]["ok"]:
                all_ok = False
        rows[bid] = row
    rep.add("SK-perm", "骨架一致性：障碍聚合通过率 ↔ docs/11 §六 矩阵（只报警）",
            rows, f"逐格差 ≤ {tol}", all_ok, note="warn-only")

    # 南带谱系分化最深
    pre = ctx.load_npz(7, "prehist")
    south = w.islands["lat"] < 0
    div_ok = float(np.median(pre["dist_pre"][south])) > float(np.median(pre["dist_pre"][~south]))
    rep.add("SK-south", "南半球隔离最深（docs/11 §七）",
            {"south_median_days": round(float(np.median(pre["dist_pre"][south])), 1),
             "north_median_days": round(float(np.median(pre["dist_pre"][~south])), 1)},
            "south > north", div_ok, note="warn-only")


# ---------------------------------------------------------------- entry
def run_check(ctx, calibrate=False) -> int:
    w = World(ctx)
    rep = Report()
    check_ironlaws(w, ctx.cfg, rep)
    check_p1(w, ctx.cfg, rep)
    check_p2(w, ctx.cfg, rep, calibrate=calibrate)
    check_p3(w, ctx.cfg, rep)
    check_p4(w, ctx.cfg, rep)
    check_p5(w, ctx.cfg, rep)
    check_p6(w, ctx.cfg, rep)
    check_p7(w, ctx.cfg, rep)
    check_skeleton(w, ctx.cfg, rep)

    warn_only = {i["id"] for i in rep.items if (i.get("note") or "").startswith("warn")}
    hard_fail = [i for i in rep.items if not i["pass"] and i["hard"]]
    soft_fail = [i for i in rep.items if not i["pass"] and not i["hard"] and i["id"] not in warn_only]
    warns = [i for i in rep.items if not i["pass"] and i["id"] in warn_only]

    ctx.save_json(9, "check", {"items": rep.items,
                               "exit_code": 2 if hard_fail else (1 if soft_fail else 0)})
    lines = ["# 验收报告", ""]
    for i in rep.items:
        flag = "✅" if i["pass"] else ("⚠️" if i["id"] in warn_only else "❌")
        hard = "【硬】" if i["hard"] else ""
        lines.append(f"## {flag} {i['id']} {hard}{i['name']}")
        lines.append(f"- 值：`{json.dumps(i['value'], ensure_ascii=False)}`")
        lines.append(f"- 阈：`{json.dumps(i['threshold'], ensure_ascii=False)}`")
        if i["worst"]:
            lines.append(f"- 最差样例：`{json.dumps(i['worst'], ensure_ascii=False)}`")
        if i["viz_cmd"]:
            lines.append(f"- 复查：`{i['viz_cmd']}`")
        if i["note"]:
            lines.append(f"- 注：{i['note']}")
        lines.append("")
    (ctx.stage_dir(9) / "check.md").write_text("\n".join(lines), encoding="utf-8")

    for i in rep.items:
        flag = "PASS" if i["pass"] else ("WARN" if i["id"] in warn_only else "FAIL")
        print(f"[{flag}] {i['id']} {i['name']}")
    code = 2 if hard_fail else (1 if soft_fail else 0)
    print(f"check 完成：硬项失败 {len(hard_fail)}，软项失败 {len(soft_fail)}，报警 {len(warns)}"
          f" → exit {code}")
    return code
