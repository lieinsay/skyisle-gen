"""⑧ 特征场扩散（docs/12 §一–§三 的实现，本生成器的核心）。

strength(t, j) = reach(t, o→j) × adopt(t, j)
  reach  = exp(−最短路 Σ[λ_t·cost_mode + (−ln perm_mode)])   —— 最大 reach 路径
  adopt  = 1 − resistance(t) × conflict(t, j)                 —— 同槽位对称不动点
同槽位归一化后是当地的比例分布（含「本地自有」行，恒 > 0：原则己）。

铁律自检：本阶段不读 height_m；文化只以连续场（share ∈ [0,1]）形式存在，无离散标签。
"""
from __future__ import annotations

import math

import numpy as np

from .. import MODES
from ..graph import CSR, accumulate_along_tree, dijkstra, weak_components
from ..rng import entity_rng
from ..weights import lambda_ref, load_directed

STAGE = 8


# ---------------------------------------------------------------- traits
def build_traits(ctx, centers: dict, secondary: list[dict], circle_of_secondary: list[str],
                 iso_daily: np.ndarray, prehist_arrival: np.ndarray, cand_src, cand_dst) -> list[dict]:
    cfg8 = ctx.section(8)
    slots_cfg = ctx.cfg["slots"]
    cats = slots_cfg["categories"]
    slots = slots_cfg["slot"]
    rr = cfg8["resistance_range"]
    dh = cfg8["half_distance_days"]
    r_max = float(cfg8["resistance_max"])

    manual = ctx.cfg.get("traits_manual")
    traits: list[dict] = []

    def mk_trait(tid: str, slot: dict, origin_node: int, kind: str, origin_ref: str,
                 origin_time: float) -> dict:
        cat = cats[slot["category"]]
        mode = cat["mode"]
        rng = entity_rng(ctx.seed, STAGE, tid)
        lo, hi = float(dh[mode][0]), float(dh[mode][1])
        d_half = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        r_lo, r_hi = rr[cat["resistance"]]
        resistance = min(r_max, float(rng.uniform(float(r_lo), float(r_hi))))
        return {"id": tid, "slot": slot["id"], "slot_zh": slot["zh"], "phrase": slot["phrase"],
                "mode": mode, "resistance": round(resistance, 4),
                "d_half_days": round(d_half, 3), "lambda": round(math.log(2.0) / d_half, 6),
                "origin_node": int(origin_node), "origin": origin_ref, "kind": kind,
                "origin_time": float(origin_time),
                "resistance_tier": cat["resistance"]}

    if manual and manual.get("trait"):
        slot_by_id = {s["id"]: s for s in slots}
        for t in manual["trait"]:
            slot = slot_by_id[t["slot"]]
            origin = t["origin"]
            if isinstance(origin, str):
                origin_node = centers[origin]["node"]
            elif isinstance(origin, list):
                raise ValueError("traits.toml 的 origin 请给中心 id 或节点号")
            else:
                origin_node = int(origin)
            tr = mk_trait(t["id"], slot, origin_node, "manual", str(origin),
                          float(t.get("origin_time", 0.0)))
            for k in ("resistance", "d_half_days", "mode"):
                if k in t:
                    tr[k] = t[k]
            if "d_half_days" in t:
                tr["lambda"] = round(math.log(2.0) / float(t["d_half_days"]), 6)
            traits.append(tr)
        return traits

    center_ids = sorted(centers.keys())
    # 主起源：每槽位 × 每主中心 1 个值（docs/11 §五：三中心是特征的起源点）
    for slot in slots:
        for cid in center_ids:
            traits.append(mk_trait(f"{slot['id']}@{cid}", slot, centers[cid]["node"],
                                   "main", cid, 0.0))
    # 次级起源：中/高阻力槽位每文明圈 k_sub 个（圈内漂移的来源，docs/04 §一「各自漂移」）
    k_sub = int(cfg8["k_sub"])
    delay = float(cfg8["sub_origin_delay_days"])
    per_circle: dict[str, list[int]] = {cid: [] for cid in center_ids}
    for sec, circ in zip(secondary, circle_of_secondary):
        if len(per_circle[circ]) < k_sub:
            per_circle[circ].append(sec["node"])
    for slot in slots:
        if cats[slot["category"]]["resistance"] == "low":
            continue
        for cid in center_ids:
            for k, node in enumerate(per_circle[cid]):
                traits.append(mk_trait(f"{slot['id']}@sub:{cid}:{k}", slot, node,
                                       "sub", f"sub:{cid}:{k}", delay))
    # 本地起源：高隔离连通分量（反射型：文化在死胡同里积累 → 孤岛文化）
    thr = float(cfg8["iso_thr"])
    min_n = int(cfg8["iso_min_component"])
    n_local_slots = int(cfg8.get("local_slots_per_component", 3))
    hi_nodes = iso_daily >= thr
    mask_e = hi_nodes[cand_src] & hi_nodes[cand_dst]
    node_ids = np.where(hi_nodes)[0]
    reflect_components: list[list[int]] = []
    if node_ids.size:
        remap = -np.ones(iso_daily.size, dtype=np.int64)
        remap[node_ids] = np.arange(node_ids.size)
        comp = weak_components(node_ids.size, remap[cand_src[mask_e]], remap[cand_dst[mask_e]])
        for c in range(comp.max() + 1 if comp.size else 0):
            members = node_ids[comp == c]
            if members.size >= min_n:
                reflect_components.append([int(x) for x in members])
    mid_high = [s for s in slots if cats[s["category"]]["resistance"] != "low"]
    for members in reflect_components:
        compkey = f"comp{min(members)}"
        rng = entity_rng(ctx.seed, STAGE, f"local:{compkey}")
        picks = rng.choice(len(mid_high), size=min(n_local_slots, len(mid_high)), replace=False)
        origin_node = members[int(np.argmax(iso_daily[np.array(members)]))]
        for pi in sorted(picks.tolist()):
            slot = mid_high[pi]
            traits.append(mk_trait(f"{slot['id']}@local:{compkey}", slot, origin_node,
                                   "local", f"local:{compkey}",
                                   float(prehist_arrival[origin_node])))
    ctx.save_json(8, "reflect", {
        "note": "反射型障碍对象：高隔离（iso_daily ≥ 阈值）连通分量。文化积累不外流（docs/12 §四）",
        "iso_thr": thr,
        "components": [{"nodes": m, "n": len(m),
                        "max_iso": round(float(iso_daily[np.array(m)].max()), 3)}
                       for m in reflect_components],
    })
    return traits


# ---------------------------------------------------------------- adopt
def fixed_point_slot(R: np.ndarray, r: np.ndarray, tol: float, max_iter: int):
    """S_t = R_t·(1 − r_t·max_{u≠t} S_u) 的 Jacobi 不动点。
    R: [V, N]，r: [V]。收敛保证：r·R < 1（sup 范数 Lipschitz）。
    返回 (S, n_iter, converged)。
    """
    S = R.copy()
    V = R.shape[0]
    rows = np.arange(V)[:, None]
    for it in range(max_iter):
        idx1 = S.argmax(axis=0)
        M1 = np.take_along_axis(S, idx1[None, :], axis=0)[0]
        if V >= 2:
            M2 = np.partition(S, V - 2, axis=0)[V - 2]
        else:
            M2 = np.zeros_like(M1)
        other_max = np.where(rows == idx1[None, :], M2[None, :], M1[None, :])
        S_new = R * (1.0 - r[:, None] * other_max)
        d = float(np.abs(S_new - S).max())
        S = S_new
        if d < tol:
            return S, it + 1, True
    return S, max_iter, False


def conflict_argmax(S: np.ndarray) -> np.ndarray:
    """每行的「谁在挡」：同槽位其余行的 argmax。[V, N] int32。"""
    V, N = S.shape
    idx1 = S.argmax(axis=0)
    Sm = S.copy()
    Sm[idx1, np.arange(N)] = -np.inf
    idx2 = Sm.argmax(axis=0)
    rows = np.arange(V)[:, None]
    return np.where(rows == idx1[None, :], idx2[None, :], idx1[None, :]).astype(np.int32)


# ---------------------------------------------------------------- run
def run(ctx):
    cfg8 = ctx.section(8)
    if cfg8.get("engine", "field") == "mc":
        raise NotImplementedError(
            "Monte Carlo 引擎为保留接口（docs/12 §七：确定性场版本已足够）。请用 engine='field'")

    isl = ctx.load_npz(3, "islands")
    ce = ctx.load_npz(3, "cand_edges")
    centers_j = ctx.load_json(7, "centers")
    prehist = ctx.load_npz(7, "prehist")
    g = load_directed(ctx)
    N = isl["lat"].size
    csr = CSR(N, g["src_d"], g["dst_d"])
    lam_ref = lambda_ref(ctx.cfg)
    cost_m, L = g["cost_m"], g["L"]

    # ---- 隔离度（反射型的连续替代：到枢纽集的图距离，进出取大）----
    hubs = [h["node"] for h in ctx.load_json(6, "hubs")["hubs"]]
    H = sorted(set([c["node"] for c in centers_j["centers"].values()] + hubs))
    csr_rev = CSR(N, g["dst_d"], g["src_d"])
    d0 = float(cfg8["iso_d0"])
    iso = np.zeros((4, N))
    for mi, m in enumerate(MODES):
        w = lam_ref[m] * cost_m[:, mi] + L[:, mi]
        d_in, _, _ = dijkstra(csr, w, H)
        d_out, _, _ = dijkstra(csr_rev, w, H)
        d = np.maximum(d_in, d_out)
        iso[mi] = np.where(np.isfinite(d), 1.0 - np.exp(-d / d0), 1.0)

    # ---- 特征表 ----
    daily_i = MODES.index("daily")
    circle_of_secondary = []
    # 次级极大的圈归属：商旅权重下最近的主中心
    center_ids = sorted(centers_j["centers"].keys())
    trade_i = MODES.index("trade")
    w_tr = lam_ref["trade"] * cost_m[:, trade_i] + L[:, trade_i]
    c_nodes = [centers_j["centers"][cid]["node"] for cid in center_ids]
    dists_c = []
    for cn in c_nodes:
        d, _, _ = dijkstra(csr, w_tr, [cn])
        dists_c.append(d)
    dists_c = np.stack(dists_c)  # [3, N]
    for sec in centers_j["secondary_peaks"]:
        j = int(np.argmin(dists_c[:, sec["node"]]))
        circle_of_secondary.append(center_ids[j])
    traits = build_traits(ctx, centers_j["centers"], centers_j["secondary_peaks"],
                          circle_of_secondary, iso[daily_i], prehist["arrival_yr"],
                          ce["src"], ce["dst"])
    T = len(traits)

    # ---- reach：每特征一次最短路（决策 1 方案 A；fast = 方案 C）----
    reach = np.zeros((T, N), dtype=np.float64)
    C_arr = np.zeros((T, N), dtype=np.float32)
    L_arr = np.zeros((T, N), dtype=np.float32)
    tree_cache: dict[tuple[int, str], tuple] = {}
    fast = bool(cfg8.get("fast", False))
    for ti, t in enumerate(traits):
        mi = MODES.index(t["mode"])
        o = t["origin_node"]
        if fast:
            key = (o, t["mode"])
            if key not in tree_cache:
                w = lam_ref[t["mode"]] * cost_m[:, mi] + L[:, mi]
                dist_w, pn, pe = dijkstra(csr, w, [o])
                Cn, Ln = accumulate_along_tree(pn, pe, dist_w, cost_m[:, mi], L[:, mi])
                tree_cache[key] = (Cn, Ln)
            Cn, Ln = tree_cache[key]
            val = -(t["lambda"] * np.nan_to_num(Cn, nan=np.inf)
                    + np.nan_to_num(Ln, nan=np.inf))
            reach[ti] = np.exp(val)
            C_arr[ti], L_arr[ti] = np.nan_to_num(Cn, nan=np.inf), np.nan_to_num(Ln, nan=np.inf)
        else:
            w = t["lambda"] * cost_m[:, mi] + L[:, mi]
            dist_w, pn, pe = dijkstra(csr, w, [o])
            with np.errstate(over="ignore"):
                reach[ti] = np.where(np.isfinite(dist_w), np.exp(-dist_w), 0.0)
            Cn, Ln = accumulate_along_tree(pn, pe, dist_w, cost_m[:, mi], L[:, mi])
            C_arr[ti] = np.nan_to_num(Cn, nan=np.inf)
            L_arr[ti] = np.nan_to_num(Ln, nan=np.inf)

    # ---- adopt：同槽位对称不动点 + 本地行 ε（隔离度越高本地越强）----
    eps0, eps_max = float(cfg8["eps0"]), float(cfg8["eps_max"])
    slot_ids = sorted({t["slot"] for t in traits})
    slot_mode = {}
    for t in traits:
        slot_mode.setdefault(t["slot"], t["mode"])
    strength = np.zeros((T, N), dtype=np.float64)
    adopt = np.ones((T, N), dtype=np.float64)
    conflict_by = np.full((T, N), -1, dtype=np.int32)
    share = np.zeros((T, N), dtype=np.float64)
    local_share = np.zeros((len(slot_ids), N), dtype=np.float64)
    fp_report = {}
    for si, sid in enumerate(slot_ids):
        rows = [ti for ti, t in enumerate(traits) if t["slot"] == sid]
        mi = MODES.index(slot_mode[sid])
        eps = eps0 + (eps_max - eps0) * iso[mi]
        R = np.concatenate([reach[rows], eps[None, :]], axis=0)
        r = np.array([traits[ti]["resistance"] for ti in rows] + [0.0])
        S, n_iter, ok = fixed_point_slot(R, r, float(cfg8["fp_tol"]), int(cfg8["fp_max_iter"]))
        cb = conflict_argmax(S)
        total = S.sum(axis=0)
        sh = S / total[None, :]
        for k, ti in enumerate(rows):
            strength[ti] = S[k]
            with np.errstate(invalid="ignore", divide="ignore"):
                adopt[ti] = np.where(R[k] > 0, S[k] / R[k], 1.0)
            share[ti] = sh[k]
            # conflict_by 映射回全局特征号（本地行记 -2）
            row_map = np.array(rows + [-2], dtype=np.int32)
            conflict_by[ti] = row_map[cb[k]]
        local_share[si] = sh[-1]
        fp_report[sid] = {"n_iter": n_iter, "converged": bool(ok), "n_values": len(rows)}
        if not ok:
            print(f"  [s08] 警告：槽位 {sid} 不动点未收敛（{n_iter} 轮）")

    for ti, t in enumerate(traits):
        t["index"] = ti
    ctx.save_json(8, "traits.resolved", {"traits": traits, "slots": slot_ids,
                                         "slot_mode": slot_mode,
                                         "lambda_ref": {m: round(lam_ref[m], 6) for m in MODES},
                                         "fixed_point": fp_report})
    ctx.save_npz(8, "fields", reach=reach.astype(np.float32),
                 strength=strength.astype(np.float32), adopt=adopt.astype(np.float32),
                 share=share.astype(np.float32), conflict_by=conflict_by,
                 C=C_arr, L=L_arr)
    ctx.save_npz(8, "iso", iso=iso.astype(np.float32), local_share=local_share.astype(np.float32))
    kinds = {}
    for t in traits:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    reach_p50 = {m: round(float(np.median(reach[[t["index"] for t in traits if t["mode"] == m]])), 4)
                 for m in MODES if any(t["mode"] == m for t in traits)}
    return {"n_traits": T, "kinds": kinds, "n_slots": len(slot_ids),
            "reach_median_by_mode": reach_p50,
            "fp_all_converged": all(v["converged"] for v in fp_report.values())}
