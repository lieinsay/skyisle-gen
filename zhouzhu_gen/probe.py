"""探针：只读产物，逐节点/逐边/逐路径/逐特征分解。

zhouzhu probe node 6468            # 节点全景：属性、出边通过率、各槽位强度表
zhouzhu probe path 100 200 --mode daily
zhouzhu probe edge 100 105         # 因子 × 模式分解
zhouzhu probe trait calendar@north_east --node 6468   # 回溯：reach 在路上被谁砍掉
"""
from __future__ import annotations

import numpy as np

from . import MODES, MODE_ZH
from .culture import World
from .graph import CSR, dijkstra
from .stages.s03_islands import CLASS_NAMES, CLASS_ZH
from .stages.s05_barriers import REGIONAL_ORDER
from .weights import lambda_ref, load_directed

LOCAL_FACTORS = ["gap", "density_drop", "climb", "political"]


def _fmt_perm(p):
    return "/".join(f"{p[mi]:.2f}" for mi in range(4))


def probe_node(ctx, node: int):
    w = World(ctx)
    isl = w.islands
    n = int(node)
    cls = CLASS_NAMES[int(isl["cls"][n])]
    clim = ctx.load_npz(4, "climate_islands")
    pre = ctx.load_npz(7, "prehist")
    print(f"== 节点 {n} ==")
    print(f"位置 ({isl['lat'][n]:.2f}, {isl['lon'][n]:.2f})  地形 {CLASS_ZH[cls]}"
          f"{'（叠层）' if isl['layered'][n] else ''}  群陆地 {isl['area_km2'][n]:.0f} km²"
          f"（势力范围 {isl['territory_km2'][n]:.0f} km²，陆地占比 {isl['land_frac'][n]:.3f}，"
          f"可用地率 {isl['arable_frac'][n]:.2f}）")
    catch = clim["catch"]
    catch_q = float((catch <= catch[n]).mean())
    area_q = float((isl["area_km2"] <= isl["area_km2"][n]).mean())
    print(f"规模 陆地分位 {area_q:.2f}  集雨容量 {catch[n]:.1f}（分位 {catch_q:.2f}）"
          f"  —— 节点 = 岛群 = 一水共同体（docs/02 §六，R10）")
    print(f"地区 {int(w.regions['region'][n])}  降水 {clim['precip'][n]:.2f}"
          f"  稳定度 {clim['stability'][n]:.2f}  史前到达 {pre['arrival_yr'][n]:.0f} 年前"
          f"  谱系 L{int(pre['lineage'][n])}")
    iso = w.iso["iso"]
    print("隔离度 " + "  ".join(f"{MODE_ZH[m]} {iso[mi][n]:.2f}" for mi, m in enumerate(MODES)))
    from .polity import Polity
    pol = Polity(ctx)
    if pol.available:
        print(f"政体 约 {float(pol.pop[n]) / 1e4:.1f} 万口 · {pol.node_line(n)}")
    from .geology import Geology
    geo = Geology(ctx)
    if geo.available:
        print(f"地质 {geo.node_zh(n)}（背景，不涉玩法）")

    ce = w.cand_edges
    mask = (ce["src"] == n) | (ce["dst"] == n)
    idx = np.where(mask)[0]
    print(f"\n出边（{idx.size} 条）：对端  距离(日)  通过率 日常/商旅/使节/迁徙  限制因子")
    perm = w.perm["perm"]
    E = ce["src"].size
    cost = w.routes["cost"]
    for e in idx:
        other = int(ce["dst"][e]) if ce["src"][e] == n else int(ce["src"][e])
        lims = []
        f = w.perm["f_regional"][e]
        for bi, bid in enumerate(REGIONAL_ORDER):
            if f[bi] > 0.05:
                lims.append(f"{bid}(f={f[bi]:.2f})")
        for lf in LOCAL_FACTORS:
            if w.perm[lf][e].min() < 0.95:
                lims.append(lf)
        if w.perm["g_blocked"][e]:
            lims.append("G!")
        c_out = cost[e] if ce["src"][e] == n else cost[e + E]
        print(f"  → {other:5d}  {c_out:6.2f}  {_fmt_perm(perm[e])}  {' '.join(lims) or '—'}")

    print("\n各槽位（本地视角，share 为比例分布）：")
    fields = w.fields
    for slot in w.slots:
        p, labels = w.shares(slot)
        order = np.argsort(-p[:, n], kind="stable")[:3]
        tops = "，".join(f"{labels[k]} {p[k, n]:.2f}" for k in order if p[k, n] >= 0.02)
        print(f"  {slot}（{MODE_ZH[w.slot_mode(slot)]}）：{tops}")
        rows = w.slot_rows()[slot]
        for ti in rows:
            r_, a_, s_ = fields["reach"][ti, n], fields["adopt"][ti, n], fields["strength"][ti, n]
            if r_ >= 0.25 and a_ <= 0.5:
                cb = int(fields["conflict_by"][ti, n])
                blocker = w.traits[cb]["id"] if cb >= 0 else "（本地自有）"
                print(f"    ⚠ {w.traits[ti]['id']}：reach {r_:.2f} 但 adopt {a_:.2f}"
                      f" —— 传到了但不要，被 {blocker} 挡住")
    return 0


def probe_edge(ctx, a: int, b: int):
    w = World(ctx)
    ce = w.cand_edges
    hit = np.where(((ce["src"] == a) & (ce["dst"] == b)) | ((ce["src"] == b) & (ce["dst"] == a)))[0]
    if hit.size == 0:
        print(f"节点 {a}–{b} 之间没有候选边")
        return 1
    e = int(hit[0])
    E = ce["src"].size
    cost = w.routes["cost"]
    print(f"== 边 {int(ce['src'][e])} ↔ {int(ce['dst'][e])}（无向边 {e}）==")
    print(f"距离 {ce['dist_days'][e]:.2f} 日  成本 {cost[e]:.2f} / {cost[e + E]:.2f}（两向）"
          f"  种类 {['kNN', '远程', '远征', '回退'][int(ce['kind'][e])]}")
    print(f"{'因子':<14}{'日常':>8}{'商旅':>8}{'使节':>8}{'迁徙':>8}")
    f = w.perm["f_regional"][e]
    cfg_b = ctx.cfg["s05"]["barriers"]
    for bi, bid in enumerate(REGIONAL_ORDER):
        if f[bi] > 0.01:
            row = [float(cfg_b[bid]["permeability"][m]) ** float(f[bi]) for m in MODES]
            print(f"{bid + f'(f={f[bi]:.2f})':<14}" + "".join(f"{v:8.3f}" for v in row))
    for lf in LOCAL_FACTORS:
        v = w.perm[lf][e]
        if v.min() < 0.999:
            print(f"{lf:<14}" + "".join(f"{float(x):8.3f}" for x in v))
    if w.perm["g_blocked"][e]:
        print("G             —— 绝对阻断（改道型）")
    print(f"{'合计':<14}" + "".join(f"{float(x):8.3f}" for x in w.perm['perm'][e]))
    return 0


def probe_path(ctx, a: int, b: int, mode: str):
    w = World(ctx)
    g = load_directed(ctx)
    N = w.islands["lat"].size
    csr = CSR(N, g["src_d"], g["dst_d"])
    lam = lambda_ref(ctx.cfg)[mode]
    mi = MODES.index(mode)
    wgt = lam * g["cost_m"][:, mi] + g["L"][:, mi]
    dist, pn, pe = dijkstra(csr, wgt, [int(a)])
    if not np.isfinite(dist[b]):
        print(f"{MODE_ZH[mode]}模式下 {a} → {b} 不可达（沿途有通过率为 0 的障碍）")
        return 1
    path, edges = [], []
    u = int(b)
    while u >= 0:
        path.append(u)
        if pn[u] >= 0:
            edges.append(int(pe[u]))
        u = int(pn[u])
    path.reverse()
    edges.reverse()
    print(f"== {a} → {b}（{MODE_ZH[mode]}，λ_ref={lam:.4f}）reach = {np.exp(-dist[b]):.4f} ==")
    print(f"{'跳':<4}{'节点':<7}{'成本':>7}{'perm':>7}{'累计reach':>10}")
    acc = 0.0
    print(f"{0:<4}{path[0]:<7}{'':>7}{'':>7}{1.0:>10.3f}")
    for k, (u, e) in enumerate(zip(path[1:], edges)):
        acc += wgt[e]
        c = g["cost_m"][e, mi]
        p = g["perm_d"][e, mi]
        print(f"{k + 1:<4}{u:<7}{c:>7.2f}{p:>7.2f}{np.exp(-acc):>10.3f}")
    return 0


def probe_trait(ctx, trait_id: str, node: int | None):
    w = World(ctx)
    t = next((t for t in w.traits if t["id"] == trait_id), None)
    if t is None:
        print(f"未知特征：{trait_id}")
        return 1
    ti = t["index"]
    print(f"== {trait_id}  槽位 {t['slot']}（{t['slot_zh']}）  {MODE_ZH[t['mode']]}"
          f"  阻力 {t['resistance']}  半衰 {t['d_half_days']} 天  起源节点 {t['origin_node']} ==")
    if node is None:
        f = w.fields
        r = f["reach"][ti]
        print(f"reach：P50 {np.median(r):.3f}  >0.5 的节点 {(r > 0.5).sum()}"
              f"  >0.1 的节点 {(r > 0.1).sum()}")
        return 0
    n = int(node)
    g = load_directed(ctx)
    N = w.islands["lat"].size
    csr = CSR(N, g["src_d"], g["dst_d"])
    mi = MODES.index(t["mode"])
    wgt = t["lambda"] * g["cost_m"][:, mi] + g["L"][:, mi]
    dist, pn, pe = dijkstra(csr, wgt, [t["origin_node"]])
    f = w.fields
    print(f"在节点 {n}：reach {f['reach'][ti, n]:.3f}  adopt {f['adopt'][ti, n]:.3f}"
          f"  strength {f['strength'][ti, n]:.3f}"
          f"  距离杀/障碍杀 = {t['lambda'] * f['C'][ti, n]:.2f} / {f['L'][ti, n]:.2f}")
    if not np.isfinite(dist[n]):
        print("不可达。")
        return 0
    hops = []
    u = n
    while pn[u] >= 0:
        e = int(pe[u])
        hops.append((u, e, float(g["L"][e, mi]), float(t["lambda"] * g["cost_m"][e, mi])))
        u = int(pn[u])
    hops.reverse()
    worst = sorted(hops, key=lambda h: -(h[2] + h[3]))[:5]
    print("最伤 reach 的五跳（−ln perm + λ·cost）：")
    und = g["und_id"]
    for u, e, L_, lc in worst:
        lims = []
        fr = w.perm["f_regional"][und[e]]
        for bi, bid in enumerate(REGIONAL_ORDER):
            if fr[bi] > 0.05:
                lims.append(bid)
        print(f"  抵达 {u:5d}：障碍损失 {max(0.0, L_):.2f}  距离损失 {lc:.2f}  "
              f"{' '.join(lims) or '（纯距离）'}")
    return 0


def probe(ctx, what: str, args: list[str], mode: str = "trade", node: int | None = None) -> int:
    if what == "node":
        return probe_node(ctx, int(args[0]))
    if what == "edge":
        return probe_edge(ctx, int(args[0]), int(args[1]))
    if what == "path":
        return probe_path(ctx, int(args[0]), int(args[1]), mode)
    if what == "trait":
        return probe_trait(ctx, args[0], node)
    print("用法见 zhouzhu probe --help")
    return 1
