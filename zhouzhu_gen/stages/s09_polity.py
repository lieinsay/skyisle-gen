"""⑨ 政治层（第四批 R7）：人口场 → 诸邦（宗法分封的邦国）→ 采邑树 → 名分 / 附庸 → 变法之国与兼并史。

依据：docs/02 §三（帝国只能长在密接群岛）、§六（一群 = 一邑）、§八（守方优势巨大，兼并极难）；
docs/04 §一（分封是运输成本的必然产物；五百年割据）、§二（兼并者在密接群岛边缘；编户齐民）、§四（渐进的占领）；
docs/11 §八（中心 ② 圈内一国完成变法、正吞并同圈诸邦；① 无大战事；③ 不展开；船团不建国、不被征服）。

模型（全部从 ①–⑧ 的产物推出；不读 height_m，不读浮石）：
  人口     pop_j = P1 × 可耕地 × 降水折减              —— 土地绝对有限，人口 = 容量（docs/02 §七）
  控制权重 w(e) = 商旅成本 × (索桥可达 ? 1 : ship_mult) + R0 × L_trade
                                                        —— 统治靠大宗后勤：能架桥短渡的才便宜（docs/02 §三）
  核心实力 S_c = Σ_j pop_j × control(c→j)              —— 一座都能「聚得拢」的人与粮（docs/02 §三 核心推论）
  控制力   control(c→j) = exp(−w_min(c→j) / R_c)，R_c = R0 × (S_c / S_ref)^β
                                                        —— 同一套统治技术，半径只随核心实力略放大
  建邦     按核心实力 S 依次择都（S 低于门槛者养不起一邦；已被圈进别邦者不再立都），立都时先圈 control ≥ θ 的未归属之邑；
           择都完毕后所有邑按「谁的控制力最大」归属（威慑距离内，不设下限：远处的采邑就是空心化的那一圈），
           无人能及之邑投附邻邑所属之邦，再无则独邑
  采邑     都城最短路树上 w > r_direct 的第一级子树 = 一个采邑（分封 = 运输成本的产物，docs/04 §一）
  名分     文明圈的宗主 = 中心节点所建之邦，五百年割据后权力空心化：控制半径 × suzerain_radius_mult，
           旧封国各自为邦，只剩名分（docs/04 §一）；附庸 = 接壤、强 ρ 倍、都在其威慑距离内
  变法     中心 ② 圈内、都在密接岛群、远离宗主、密接与中疏各半的最强邦 → 编户齐民，动员量级高一档
  兼并     只有变法之国能兼并（docs/02 §八「兼并极难」↔ 铁律三 的调和：正因兼并极难，只有先编户齐民者兼并得动）；
           先易后难逐邦吞并；占领后的消化阶段按 docs/04 §四 的时间表
铁律自检：本阶段不读 height_m；船团（稀疏岛链）不建国也不被征服；处处有人 → 每个节点都属于某个政体。
"""
from __future__ import annotations

import math

import numpy as np

from .. import MODES
from ..graph import CSR, dijkstra, weak_components
from ..weights import lambda_ref, load_directed
from .s03_islands import CLASS_NAMES
from .s05_barriers import REGIONAL_ORDER
from .s07_centers import CENTER_IDS, CENTER_ZH

STAGE = 9
KIND_STATE, KIND_FLEET, KIND_TRIBE = 0, 1, 2
KIND_ZH = {KIND_STATE: "邦", KIND_FLEET: "船团", KIND_TRIBE: "部落"}
# 政体类型（docs/02 §三 对照表 + docs/04 §二）
REGIME_ZH = {
    "reformed": "编户齐民（已变法）",
    "suzerain": "宗主（礼制中心，名分所出，实权只及本邦）",
    "centralizable": "密接之国：官僚集权可行，仍行宗法之制",
    "feudal": "宗法分封：公室与世卿分掌，权力出于世系与名分",
    "city": "独邑：一水共同体自治，掌水者即主事者",
    "fleet": "流动船团：无固定中心，首领以航技与战绩服众",
    "tribe": "部落：长老与掌水者议事",
}
# 渐进的占领（docs/04 §四）：占领后满多少年进入哪一阶段（阶段名, 内容）
STAGE_NAMES = ["military", "administrative", "economic", "cultural", "linguistic", "identity"]
STAGE_ZH = {
    "military": ("军事", "边邑易手、议和、驻军"),
    "administrative": ("行政", "户籍登记、度量重校、律法颁行，旧官吏留用或替换"),
    "economic": ("经济", "关卡撤除、道路改善、赋役摊派、旧契约重立"),
    "cultural": ("文化", "文字、历法、礼制、称谓的替换（一代人）"),
    "linguistic": ("语言", "官话渗透，方言退居家中（两至三代人）"),
    "identity": ("认同", "「我是某邦人」变成「我是某郡人」（三代以上）"),
}


# ---------------------------------------------------------------- 人口
def population(cfg: dict, isl: dict, clim: dict) -> np.ndarray:
    """pop_j = P1 × 可耕地 × clip(降水 / precip_full, floor, 1)。土地早已在册 → 人口 = 容量。"""
    p = cfg["s09"]["polity"]
    p1 = float(cfg["shared"]["scale"]["people_per_arable_km2"])
    arable = isl["area_km2"].astype(np.float64) * isl["arable_frac"].astype(np.float64)
    wet = np.clip(clim["precip"].astype(np.float64) / float(p["precip_full"]), float(p["precip_floor"]), 1.0)
    return p1 * arable * wet


def consolidation_stage(years: float, p: dict) -> str:
    th = p["stage_years"]   # 5 个门槛：军事 < th0 ≤ 行政 < th1 ≤ 经济 < th2 ≤ 文化 < th3 ≤ 语言 < th4 ≤ 认同
    for k, t in enumerate(th):
        if years < float(t):
            return STAGE_NAMES[k]
    return STAGE_NAMES[-1]


# ---------------------------------------------------------------- 主流程
def run(ctx):
    p = ctx.section(9)["polity"]
    isl = ctx.load_npz(3, "islands")
    ce = ctx.load_npz(3, "cand_edges")
    clim = ctx.load_npz(4, "climate_islands")
    pm = ctx.load_npz(5, "perm")
    centers = ctx.load_json(7, "centers")["centers"]
    g = load_directed(ctx)
    N = isl["lat"].size
    E = ce["src"].size
    cls = isl["cls"].astype(np.int64)
    layered = isl["layered"].astype(bool)
    trade_i = MODES.index("trade")

    pop = population(ctx.cfg, isl, clim)

    # ---- 控制权重（有向）：商旅成本 × 后勤倍率 + R0 × L_trade ----
    r0 = float(p["control_radius_days"])
    bridge_days = float(ctx.cfg["shared"]["ships"]["bridge_days"])
    ship_mult = float(p["ship_logistics_mult"])
    dist_und = np.concatenate([ce["dist_days"], ce["dist_days"]]).astype(np.float64)
    logistics = np.where(dist_und <= bridge_days, 1.0, ship_mult)
    w_pol = g["cost_m"][:, trade_i] * logistics + r0 * g["L"][:, trade_i]
    csr = CSR(N, g["src_d"], g["dst_d"])
    theta = float(p["control_min"])
    ln_theta = math.log(1.0 / theta)

    # ---- 核心实力 S：一座都在控制范围内「聚得拢」的人口 Σ pop × control（每邑一次有界搜索）----
    eligible = (cls == 0) | (cls == 1)          # 密接 / 中疏 才建邦；稀疏 = 船团，孤悬 = 部落
    S = np.zeros(N, dtype=np.float64)
    bound0 = r0 * ln_theta
    for c in np.where(eligible)[0].tolist():
        d, _pn, _pe = dijkstra(csr, w_pol, [c], max_dist=bound0)
        m = np.isfinite(d) & eligible
        S[c] = float((pop[m] * np.exp(-d[m] / r0)).sum())
    s_ref = float(np.median(S[eligible])) if eligible.any() else 1.0
    beta = float(p["radius_pop_exponent"])
    radius = r0 * np.clip((S / max(s_ref, 1e-9)) ** beta, float(p["radius_mult_min"]), float(p["radius_mult_max"]))
    s_min = float(p["capital_min_strength_frac"]) * s_ref

    # ---- 宗主先立：三个文明中心是最早的国家（docs/04 §一），但五百年后权力空心化 → 半径打折 ----
    main_nodes = [int(centers[cid]["node"]) for cid in CENTER_IDS]
    suz_mult = float(p["suzerain_radius_mult"])
    for cn in main_nodes:
        if eligible[cn]:
            radius[cn] = r0 * suz_mult      # 不随核心实力放大：中心处人口最稠，不打折它就是圈内最大的邦
    # ---- 建邦 第一遍：按核心实力依次择都（S ≥ 门槛、非叠层、尚无归属），圈 control ≥ θ 的未归属之邑 ----
    claimed = np.full(N, -1, dtype=np.int64)
    capitals: list[int] = []
    order = np.argsort(-S, kind="stable")
    first = [cn for cn in main_nodes if eligible[cn]]
    for c in first + order.tolist():
        if not eligible[c] or claimed[c] >= 0 or layered[c] or S[c] < s_min:
            continue
        sid = len(capitals)
        capitals.append(c)
        d, _pn, _pe = dijkstra(csr, w_pol, [c], max_dist=radius[c] * ln_theta)
        take = np.isfinite(d) & eligible & (claimed < 0)
        claimed[take] = sid
    n_cap1 = len(capitals)
    # 王畿：宗主在第一遍圈到的邑（半径已打折）锁定给宗主 —— 名分至少还护得住都城周围那一圈
    suz_sids = {sid for sid, c in enumerate(capitals) if c in first}
    locked = np.isin(claimed, list(suz_sids)) if suz_sids else np.zeros(N, dtype=bool)

    # ---- 第二遍：所有邑按「谁的控制力最大」归属（威慑距离内，不设下限）；记录都城间距离（附庸判定用）----
    vassal_reach = float(p["vassal_reach"])
    state = np.full(N, -1, dtype=np.int64)
    control = np.zeros(N, dtype=np.float64)
    d_cap = np.full(N, np.inf, dtype=np.float64)
    pred = np.full(N, -1, dtype=np.int64)
    cap_dist: list[dict[int, float]] = [dict() for _ in range(n_cap1)]
    cap_arr = np.array(capitals, dtype=np.int64)
    for sid, c in enumerate(capitals):
        bound_w = radius[c] * ln_theta
        d, pn, _pe = dijkstra(csr, w_pol, [c], max_dist=bound_w * vassal_reach)
        ctrl = np.exp(-d / radius[c])
        take = np.isfinite(d) & eligible & (ctrl > control + 1e-12)
        if sid not in suz_sids:
            take &= ~locked
        state[take] = sid
        control[take] = ctrl[take]
        d_cap[take] = d[take]
        pred[take] = pn[take]
        dc = d[cap_arr]
        for t in np.where(np.isfinite(dc))[0].tolist():
            if t != sid:
                cap_dist[sid][int(t)] = float(dc[t])
    # 威慑距离外无人能及之邑：投附邻邑（商旅可通的候选边）所属之邦，取控制力最高的邻邑；迭代到不再变化
    n_attached = 0
    trade_ok = pm["perm"][:, trade_i] > 0
    w_und = np.minimum(w_pol[:E], w_pol[E:])
    while True:
        loose = eligible & (state < 0)
        if not loose.any():
            break
        changed = False
        for j in np.where(loose)[0].tolist():
            m = ((ce["src"] == j) | (ce["dst"] == j)) & trade_ok
            if not m.any():
                continue
            other = np.where(ce["src"][m] == j, ce["dst"][m], ce["src"][m])
            ok = state[other] >= 0
            if not ok.any():
                continue
            k = int(np.argmax(np.where(ok, control[other] * np.exp(-w_und[m] / r0), -1.0)))
            o = int(other[k])
            state[j] = state[o]
            control[j] = float(control[o] * np.exp(-w_und[m][k] / radius[capitals[state[o]]]))
            d_cap[j] = float(d_cap[o] + w_und[m][k])
            pred[j] = o
            n_attached += 1
            changed = True
        if not changed:
            break
    # 再无人能及者自成独邑
    for c in np.where(eligible & (state < 0))[0].tolist():
        state[c] = len(capitals)
        capitals.append(c)
        cap_dist.append(dict())
        control[c] = 1.0
        d_cap[c] = 0.0
    n_states = len(capitals)
    cap_arr = np.array(capitals, dtype=np.int64)
    assert bool((state[eligible] >= 0).all()), "有可建邦之邑无归属"

    # ---- 采邑树：都城最短路树上 w > r_direct 的第一级子树 ----
    r_direct = float(p["direct_rule_days"])
    fief = np.full(N, -1, dtype=np.int64)     # -1 = 直辖；否则 = 采邑之主所在邑
    for j in np.argsort(d_cap, kind="stable").tolist():
        s = state[j]
        if s < 0 or not np.isfinite(d_cap[j]):
            continue
        if d_cap[j] <= r_direct:
            fief[j] = -1
            continue
        pj = int(pred[j])
        if pj < 0 or state[pj] != s or fief[pj] < 0:   # 父在直辖圈内（或树链离开本邦）→ 此邑即采邑之主
            fief[j] = j
        else:
            fief[j] = fief[pj]

    # ---- 船团与部落：稀疏岛链按连通分量成团；孤悬散岛各自为部落 ----
    kind = np.full(N, -1, dtype=np.int64)
    polity = np.full(N, -1, dtype=np.int64)
    kind[state >= 0] = KIND_STATE
    polity[state >= 0] = state[state >= 0]
    sparse = cls == 2
    n_pol = n_states
    fleets: list[list[int]] = []
    if sparse.any():
        ids = np.where(sparse)[0]
        remap = -np.ones(N, dtype=np.int64)
        remap[ids] = np.arange(ids.size)
        m = sparse[ce["src"]] & sparse[ce["dst"]]
        comp = weak_components(ids.size, remap[ce["src"][m]], remap[ce["dst"][m]])
        for k in range(int(comp.max()) + 1 if comp.size else 0):
            members = ids[comp == k]
            fleets.append([int(x) for x in members])
            polity[members] = n_pol
            kind[members] = KIND_FLEET
            n_pol += 1
    tribes: list[int] = []
    for j in np.where(kind < 0)[0].tolist():   # 孤悬（以及任何漏网者）
        tribes.append(j)
        polity[j] = n_pol
        kind[j] = KIND_TRIBE
        n_pol += 1
    assert bool((polity >= 0).all()), "原则己：每个节点都必须属于某个政体"

    # ---- 文明圈（与 ⑧ 同规则：商旅权重下最近的主中心）与宗主 ----
    lam = lambda_ref(ctx.cfg)
    w_tr = lam["trade"] * g["cost_m"][:, trade_i] + g["L"][:, trade_i]
    dists_c = np.stack([dijkstra(csr, w_tr, [cn])[0] for cn in main_nodes])
    circle = np.argmin(dists_c, axis=0).astype(np.int64)
    suzerain = {cid: (int(state[cn]) if state[cn] >= 0 else -1) for cid, cn in zip(CENTER_IDS, main_nodes)}
    suz_set = {v for v in suzerain.values() if v >= 0}

    # ---- 邦的属性 ----
    members_of = [np.where(state == s)[0] for s in range(n_states)]
    pop_state = np.array([float(pop[m].sum()) for m in members_of])
    n_nodes = np.array([m.size for m in members_of])
    circle_state = circle[cap_arr]
    dense_frac = np.array([float((cls[m] == 0).mean()) if m.size else 0.0 for m in members_of])

    # 邦际接壤（商旅可通的候选边）
    adj: list[dict[int, int]] = [dict() for _ in range(n_states)]
    ok_e = pm["perm"][:, trade_i] > 0
    sa, sb = state[ce["src"]], state[ce["dst"]]
    both = (sa >= 0) & (sb >= 0) & (sa != sb) & ok_e
    for a, b in zip(sa[both].tolist(), sb[both].tolist()):
        adj[a][b] = adj[a].get(b, 0) + 1
        adj[b][a] = adj[b].get(a, 0) + 1

    # ---- 附庸：接壤、人口 ≥ ρ 倍、都在对方威慑距离内；取威慑（人口 × 控制力）最大者 ----
    rho = float(p["vassal_pop_ratio"])
    overlord = np.full(n_states, -1, dtype=np.int64)
    for s in range(n_states):
        if s in suz_set:
            continue
        best, best_v = -1, 0.0
        for t in sorted(adj[s]):
            if pop_state[t] < rho * pop_state[s]:
                continue
            dts = cap_dist[t].get(capitals[s])
            if dts is None:
                continue
            v = pop_state[t] * math.exp(-dts / radius[capitals[t]])
            if v > best_v:
                best, best_v = t, v
        overlord[s] = best

    # ---- 变法之国（中心 ② 圈）----
    reform_circle = str(p["reform_circle"])
    rc = CENTER_IDS.index(reform_circle)
    center_node = main_nodes[rc]
    d_center_days, _, _ = dijkstra(csr, g["cost_m"][:, trade_i], [center_node])
    d_ref = float(p["reform_center_distance_days"])
    override = int(p.get("reformer_capital", -1))
    reformer = -1
    reformer_fallback = False
    if override >= 0 and state[override] >= 0:
        reformer = int(state[override])
    else:
        def score_of(s):
            c = capitals[s]
            edge = 4.0 * dense_frac[s] * (1.0 - dense_frac[s])      # 密接与中疏各半 = 「密接群岛的边缘」
            far = min(1.0, float(d_center_days[c]) / d_ref) if np.isfinite(d_center_days[c]) else 1.0
            return pop_state[s] * (0.25 + edge) * far
        cands = [s for s in range(n_states)
                 if circle_state[s] == rc and s not in suz_set and cls[capitals[s]] == 0 and n_nodes[s] >= 2]
        if not cands:
            reformer_fallback = True
            cands = [s for s in range(n_states) if circle_state[s] == rc and s not in suz_set and n_nodes[s] >= 3]
        if cands:
            reformer = max(cands, key=lambda s: (score_of(s), -s))

    # ---- 兼并史：只有变法之国能兼并；先易后难；消化阶段按 docs/04 §四 ----
    years_ago = float(p["reform_years_ago"])
    reform_dur = float(p["reform_duration_years"])
    mob = float(p["mobilization_mult"])
    defend = float(p["defender_advantage"])
    awe = float(p["suzerain_awe"])
    y_base = float(p["war_years_base"])
    y_scale = float(p["war_years_per_pop_ratio"])
    y_per_day = float(p["war_years_per_frontier_day"])
    n_fronts = int(p["active_fronts"])
    annexed_by = np.full(n_states, -1, dtype=np.int64)
    annexed_years = np.full(n_states, np.nan)
    history: list[dict] = []
    fronts: list[dict] = []
    realm_pop = 0.0
    if reformer >= 0:
        realm = {reformer}
        realm_pop = float(pop_state[reformer])
        elapsed = 0.0
        budget = max(0.0, years_ago - reform_dur)
        w_frontier = np.where(ok_e, np.minimum(w_pol[:E], w_pol[E:]), np.inf)

        def candidates():
            out = set()
            for s in realm:
                for t in adj[s]:
                    if t not in realm and circle_state[t] == rc:
                        out.add(t)
            # 前线成本：跨界最便宜的一条边（天）
            in_realm = np.isin(np.arange(n_states), list(realm))
            res = []
            for t in sorted(out):
                m = both & (((sa == t) & in_realm[np.maximum(sb, 0)]) | ((sb == t) & in_realm[np.maximum(sa, 0)]))
                fc = float(np.nanmin(w_frontier[m])) if m.any() else 1.0
                res.append((t, fc))
            return res

        while True:
            cands = candidates()
            scored = []
            for t, fc in cands:
                eff_def = pop_state[t] * defend * (awe if t in suz_set else 1.0)
                feasible = realm_pop * mob >= eff_def
                difficulty = eff_def * (1.0 + fc)
                scored.append((difficulty, t, fc, feasible))
            scored.sort()
            feas = [x for x in scored if x[3]]
            if not feas:
                fronts = [{"polity": int(t), "frontier_days": round(fc, 2), "feasible": False}
                          for _, t, fc, _ in scored[:n_fronts]]
                break
            difficulty, t, fc, _ = feas[0]
            ratio = pop_state[t] * (awe if t in suz_set else 1.0) / max(realm_pop, 1.0)
            war_years = y_base + y_scale * ratio + y_per_day * fc
            if elapsed + war_years > budget:
                fronts = [{"polity": int(tt), "frontier_days": round(fcc, 2), "feasible": True,
                           "war_years_needed": round(y_base + y_scale * pop_state[tt] / max(realm_pop, 1.0) + y_per_day * fcc, 1)}
                          for _, tt, fcc, _ in feas[:n_fronts]]
                break
            elapsed += war_years
            realm.add(int(t))
            realm_pop += float(pop_state[t])
            ya = years_ago - reform_dur - elapsed
            annexed_by[t] = reformer
            annexed_years[t] = ya
            history.append({"polity": int(t), "years_ago": round(ya, 1), "war_years": round(war_years, 1),
                            "pop": round(float(pop_state[t])), "n_nodes": int(n_nodes[t]),
                            "was_suzerain": bool(t in suz_set),
                            "stage": consolidation_stage(ya, p)})
            # 被并之邦的附庸改属兼并者
            for s in range(n_states):
                if overlord[s] == t:
                    overlord[s] = reformer
    realm_id = np.where(annexed_by >= 0, annexed_by, np.arange(n_states))
    realm_node = np.full(N, -1, dtype=np.int64)
    realm_node[state >= 0] = realm_id[state[state >= 0]]
    realm_node[state < 0] = polity[state < 0]

    # ---- 政体类型 ----
    regime = []
    for s in range(n_states):
        if s == reformer:
            regime.append("reformed")
        elif s in suz_set:
            regime.append("suzerain")
        elif n_nodes[s] == 1:
            regime.append("city")
        elif cls[capitals[s]] == 0:
            regime.append("centralizable")
        else:
            regime.append("feudal")

    # ---- 开局候选（docs/11 §九）----
    d_col = REGIONAL_ORDER.index("D")
    touch_d_e = pm["f_regional"][:, d_col] > 0.05
    touch_d = np.zeros(N, dtype=bool)
    touch_d[ce["src"][touch_d_e]] = True
    touch_d[ce["dst"][touch_d_e]] = True
    d_share = np.array([float(touch_d[m].mean()) if m.size else 0.0 for m in members_of])
    in_circle = np.array([circle_state[s] == rc for s in range(n_states)])
    free = in_circle & (annexed_by < 0) & (np.arange(n_states) != reformer)
    pool = np.where(in_circle & (d_share > 0) & (n_nodes >= 2))[0].tolist()
    a_c = [s for s in pool if free[s]]
    a_c.sort(key=lambda s: (pop_state[s], s))            # 边陲小邦：贴着 D、未被并、越小越像
    c_c = sorted(pool, key=lambda s: (-d_share[s], s))    # 过渡带：邑贴着 D 的比例最高者
    openings = {
        "A_frontier_small_state": a_c[:3],
        "B_orthodox_core": suzerain.get(reform_circle, -1),
        "C_transition_zone": c_c[:3],
        "D_reformer": int(reformer),
    }

    # ---- 产物 ----
    fief_seat_count = np.zeros(n_states, dtype=np.int64)
    direct_count = np.zeros(n_states, dtype=np.int64)
    for s in range(n_states):
        m = members_of[s]
        f = fief[m]
        direct_count[s] = int((f < 0).sum())
        fief_seat_count[s] = int(np.unique(f[f >= 0]).size)

    polities = []
    for s in range(n_states):
        c = capitals[s]
        polities.append({
            "id": s, "kind": "state", "name": f"邦{s:03d}", "capital": int(c),
            "capital_class": CLASS_NAMES[int(cls[c])], "circle": CENTER_IDS[int(circle_state[s])],
            "regime": regime[s], "n_nodes": int(n_nodes[s]), "pop": round(float(pop_state[s])),
            "dense_frac": round(float(dense_frac[s]), 3),
            "radius_days": round(float(radius[c]), 3),
            "n_direct": int(direct_count[s]), "n_fiefs": int(fief_seat_count[s]),
            "overlord": int(overlord[s]), "vassals": [int(t) for t in np.where(overlord == s)[0]],
            "neighbors": {str(t): int(n_e) for t, n_e in sorted(adj[s].items())},
            "annexed_by": int(annexed_by[s]),
            "annexed_years_ago": (None if np.isnan(annexed_years[s]) else round(float(annexed_years[s]), 1)),
            "stage": (consolidation_stage(float(annexed_years[s]), p) if not np.isnan(annexed_years[s]) else None),
            "is_suzerain": bool(s in suz_set),
        })
    for k, members in enumerate(fleets):
        polities.append({"id": n_states + k, "kind": "fleet", "name": f"船团{k:02d}", "capital": -1,
                         "n_nodes": len(members), "pop": round(float(pop[np.array(members)].sum())),
                         "regime": "fleet", "circle": CENTER_IDS[int(np.bincount(circle[np.array(members)]).argmax())]})
    for k, j in enumerate(tribes):
        polities.append({"id": n_states + len(fleets) + k, "kind": "tribe", "name": f"部落{k:02d}", "capital": int(j),
                         "n_nodes": 1, "pop": round(float(pop[j])), "regime": "tribe", "circle": CENTER_IDS[int(circle[j])]})

    ctx.save_npz(9, "polity",
                 pop=pop.astype(np.float32), state=state.astype(np.int32), polity=polity.astype(np.int32),
                 kind=kind.astype(np.int8), control=control.astype(np.float32), dist_cap=d_cap.astype(np.float32),
                 fief=fief.astype(np.int32), realm=realm_node.astype(np.int32), circle=circle.astype(np.int8),
                 capital=cap_arr.astype(np.int32), pop_state=pop_state.astype(np.float32))
    ctx.save_json(9, "polities", {
        "polities": polities,
        "n_states": n_states, "n_fleets": len(fleets), "n_tribes": len(tribes),
        "suzerain": suzerain,
        "reformer": {"polity": int(reformer), "fallback": reformer_fallback, "circle": reform_circle,
                     "reform_years_ago": years_ago, "reform_duration_years": reform_dur,
                     "realm_pop": round(realm_pop), "n_annexed": len(history)},
        "history": history, "fronts": fronts, "openings": openings,
        "stage_zh": {k: list(v) for k, v in STAGE_ZH.items()}, "regime_zh": REGIME_ZH,
        "center_zh": CENTER_ZH,
    })
    _write_history_md(ctx, polities, suzerain, reformer, reformer_fallback, history, fronts, openings, pop, p, n_states, fleets, tribes)

    by_cls = {}
    for cname, ci in (("dense", 0), ("medium", 1)):
        sel = [s for s in range(n_states) if cls[capitals[s]] == ci]
        by_cls[cname] = {"n": len(sel), "median_nodes": float(np.median(n_nodes[sel])) if sel else 0.0,
                         "max_nodes": int(n_nodes[sel].max()) if sel else 0}
    return {"pop_total_M": round(float(pop.sum()) / 1e6, 1), "n_states": n_states,
            "n_capitals_by_strength": n_cap1, "n_singletons": n_states - n_cap1, "n_attached_loose": n_attached,
            "n_fleets": len(fleets), "n_tribes": len(tribes),
            "median_state_nodes": float(np.median(n_nodes)), "max_state_nodes": int(n_nodes.max()),
            "states_by_capital_class": by_cls,
            "reformer": int(reformer), "reformer_fallback": reformer_fallback,
            "n_annexed": len(history), "realm_pop_M": round(realm_pop / 1e6, 2), "n_fronts": len(fronts)}


def _write_history_md(ctx, polities, suzerain, reformer, fallback, history, fronts, openings, pop, p,
                      n_states, fleets, tribes):
    P = {x["id"]: x for x in polities}

    def nm(s):
        if s is None or s < 0:
            return "（无）"
        x = P[s]
        return f"{x['name']}（都 #{x['capital']}，{x['n_nodes']} 邑，约 {x['pop'] / 1e4:.0f} 万口）" if x["kind"] == "state" else x["name"]

    big = sum(1 for x in polities if x["kind"] == "state" and x["n_nodes"] >= 50)
    mid = sum(1 for x in polities if x["kind"] == "state" and 10 <= x["n_nodes"] < 50)
    small = n_states - big - mid
    lines = [f"# 政治层：诸邦与兼并史（seed {ctx.seed}）", "",
             "> 自动生成。所有专名为结构性占位（邦 = 都城所在邑的编号）。名分、附庸、兼并全部由地理与航线推出。", "",
             "## 总览", "",
             f"全世界 {pop.size} 邑、约 {pop.sum() / 1e8:.2f} 亿口，分属 {n_states} 邦（大邦 ≥50 邑 {big}、中邦 10–49 邑 {mid}、小邦与独邑 {small}）、"
             f"{len(fleets)} 个船团（稀疏岛链，不建国、不被征服）、{len(tribes)} 个部落（孤悬散岛）。", "",
             "## 三圈的宗主（名分所出，实权只及本邦）", ""]
    for cid, s in suzerain.items():
        lines.append(f"- {CENTER_ZH[cid]}：{nm(s)}")
    lines += ["", "## 变法之国", ""]
    if reformer < 0:
        lines.append("（未能在中心 ② 圈内找到可变法之邦）")
    else:
        r = P[reformer]
        lines.append(f"{nm(reformer)}，都在{ '密接' if r['capital_class'] == 'dense' else '中疏' }岛群，"
                     f"邑中密接者占 {r['dense_frac'] * 100:.0f}%（密接群岛的边缘：有足够的连片耕地推行编户，又远离礼制腹地）。"
                     f"{'【注意：圈内无密接之邦，退而取中疏之邦】' if fallback else ''}")
        lines.append(f"约 {p['reform_years_ago']:.0f} 年前变法：编户齐民、军功爵、成文法、统一度量衡；用 {p['reform_duration_years']:.0f} 年完成，"
                     f"动员能力高一档（×{p['mobilization_mult']}）。此后开始兼并同圈诸邦——正因兼并极难（docs/02 §八），只有先编户齐民者兼并得动。")
        lines += ["", "## 兼并纪年（距今）", ""]
        if not history:
            lines.append("（尚未并得一邦）")
        for h in history:
            z = STAGE_ZH[h["stage"]]
            lines.append(f"- 距今 {h['years_ago']:.0f} 年：并 {nm(h['polity'])}{'——此即本圈宗主，名分自此归于新朝' if h['was_suzerain'] else ''}；"
                         f"战事 {h['war_years']:.0f} 年。今至「{z[0]}」阶段：{z[1]}。")
        lines += ["", "## 当前战事", ""]
        if not fronts:
            lines.append("（无：同圈已无接壤之邦）")
        for f in fronts:
            if f["feasible"]:
                lines.append(f"- 正攻伐 {nm(f['polity'])}：前线一跳 {f['frontier_days']:.1f} 日，尚需约 {f['war_years_needed']:.0f} 年。")
            else:
                lines.append(f"- 与 {nm(f['polity'])} 对峙：守方优势巨大，眼下动员不足以攻取。")
    lines += ["", "## 开局候选（docs/11 §九）", "",
              f"- A 边陲小邦（中心 ② 圈边缘、紧邻 D、未被兼并）：{'；'.join(nm(s) for s in openings['A_frontier_small_state']) or '（无）'}",
              f"- B 正统核心：{nm(openings['B_orthodox_core'])}",
              f"- C 过渡带（邑多半贴着 D）：{'；'.join(nm(s) for s in openings['C_transition_zone']) or '（无）'}",
              f"- D 兼并者：{nm(openings['D_reformer'])}", ""]
    (ctx.stage_dir(9) / "history.md").write_text("\n".join(lines), encoding="utf-8")
