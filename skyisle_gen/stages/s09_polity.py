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
由 C++ 核心（core/src/planet/stage9.cpp；行星计划 P6d，Python 参考版删于 2026-09-30，tag python-reference-final）算，polity.npz / polities.json / history.md 与摘要照旧由这里写
（_write：邦名、中文、round 的位数都在这里）；第三层（岛群生成器）的人口与邦都直接从 C++ 的 ⑨ 对象取。
"""
from __future__ import annotations

import numpy as np

from .s03_islands import CLASS_NAMES
from .s07_centers import CENTER_IDS, CENTER_ZH

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


def consolidation_stage(years: float, p: dict) -> str:
    th = p["stage_years"]   # 5 个门槛：军事 < th0 ≤ 行政 < th1 ≤ 经济 < th2 ≤ 文化 < th3 ≤ 语言 < th4 ≤ 认同
    for k, t in enumerate(th):
        if years < float(t):
            return STAGE_NAMES[k]
    return STAGE_NAMES[-1]


# ---------------------------------------------------------------- 主流程
def run(ctx):
    from ..engine import core, part, planet_config, put_part
    cc = core()
    Pol = cc.planet_stage9(cc.make_config(planet_config(ctx.cfg)), part(ctx, 3), part(ctx, 4), part(ctx, 5), part(ctx, 6), part(ctx, 7))
    put_part(ctx, 9, Pol)
    return _write(ctx, cc.polity_arrays(Pol))


def _write(ctx, R: dict) -> dict:
    """写 polity.npz / polities.json / history.md、出摘要（邦名与中文、round 的位数都在这里；R 里的浮点是双精度原值，
    纪年的 years_ago / war_years 与战线的 war_years_needed 在当年的 Python 版里是 numpy 标量，按 np.float64 的 round 舍，照旧）。"""
    p = ctx.section(9)["polity"]
    cls = ctx.load_npz(3, "islands")["cls"].astype(np.int64)
    pop, circle = R["pop"], R["circle"]
    n_states = len(R["states"])
    fleets = [[int(x) for x in m] for m in R["fleets"]]
    tribes = [int(j) for j in R["tribes"]]
    reformer, reformer_fallback = int(R["reformer"]), bool(R["reformer_fallback"])
    reform_circle = str(p["reform_circle"])
    years_ago = float(p["reform_years_ago"])
    reform_dur = float(p["reform_duration_years"])
    realm_pop = float(R["realm_pop"])
    suzerain = {cid: int(v) for cid, v in zip(CENTER_IDS, R["suzerain"])}
    history = [{"polity": int(h["polity"]), "years_ago": round(np.float64(h["years_ago"]), 1),
                "war_years": round(np.float64(h["war_years"]), 1), "pop": round(float(h["pop"])), "n_nodes": int(h["n_nodes"]),
                "was_suzerain": bool(h["was_suzerain"]), "stage": h["stage"]} for h in R["history"]]
    fronts = []
    for f in R["fronts"]:
        x = {"polity": int(f["polity"]), "frontier_days": round(float(f["frontier_days"]), 2), "feasible": bool(f["feasible"])}
        if f["feasible"]:
            x["war_years_needed"] = round(np.float64(f["war_years_needed"]), 1)
        fronts.append(x)
    openings = {
        "A_frontier_small_state": [int(x) for x in R["open_a"]],
        "B_orthodox_core": int(R["open_b"]),
        "C_transition_zone": [int(x) for x in R["open_c"]],
        "D_reformer": int(R["open_d"]),
    }
    polities = []
    for s, x in enumerate(R["states"]):
        c = int(x["capital"])
        ay = float(x["annexed_years"])
        polities.append({
            "id": s, "kind": "state", "name": f"邦{s:03d}", "capital": c,
            "capital_class": CLASS_NAMES[int(cls[c])], "circle": CENTER_IDS[int(x["circle"])],
            "regime": x["regime"], "n_nodes": int(x["n_nodes"]), "pop": round(float(x["pop"])),
            "dense_frac": round(float(x["dense_frac"]), 3),
            "radius_days": round(float(x["radius"]), 3),
            "n_direct": int(x["n_direct"]), "n_fiefs": int(x["n_fiefs"]),
            "overlord": int(x["overlord"]), "vassals": [int(t) for t in x["vassals"]],
            "neighbors": {str(int(t)): int(n_e) for t, n_e in x["neighbors"]},
            "annexed_by": int(x["annexed_by"]),
            "annexed_years_ago": (None if np.isnan(ay) else round(ay, 1)),
            "stage": (consolidation_stage(ay, p) if not np.isnan(ay) else None),
            "is_suzerain": bool(x["is_suzerain"]),
        })
    for k, members in enumerate(fleets):
        polities.append({"id": n_states + k, "kind": "fleet", "name": f"船团{k:02d}", "capital": -1,
                         "n_nodes": len(members), "pop": round(float(R["fleet_pop"][k])),
                         "regime": "fleet", "circle": CENTER_IDS[int(R["fleet_circle"][k])]})
    for k, j in enumerate(tribes):
        polities.append({"id": n_states + len(fleets) + k, "kind": "tribe", "name": f"部落{k:02d}", "capital": int(j),
                         "n_nodes": 1, "pop": round(float(pop[j])), "regime": "tribe", "circle": CENTER_IDS[int(circle[j])]})

    ctx.save_npz(9, "polity",
                 pop=pop.astype(np.float32), state=R["state"].astype(np.int32), polity=R["polity"].astype(np.int32),
                 kind=R["kind"].astype(np.int8), control=R["control"].astype(np.float32), dist_cap=R["dist_cap"].astype(np.float32),
                 fief=R["fief"].astype(np.int32), realm=R["realm"].astype(np.int32), circle=circle.astype(np.int8),
                 capital=R["capital"].astype(np.int32), pop_state=R["pop_state"].astype(np.float32))
    ctx.save_json(9, "polities", {
        "polities": polities,
        "n_states": n_states, "n_fleets": len(fleets), "n_tribes": len(tribes),
        "suzerain": suzerain,
        "reformer": {"polity": reformer, "fallback": reformer_fallback, "circle": reform_circle,
                     "reform_years_ago": years_ago, "reform_duration_years": reform_dur,
                     "realm_pop": round(realm_pop), "n_annexed": len(history)},
        "history": history, "fronts": fronts, "openings": openings,
        "stage_zh": {k: list(v) for k, v in STAGE_ZH.items()}, "regime_zh": REGIME_ZH,
        "center_zh": CENTER_ZH,
    })
    _write_history_md(ctx, polities, suzerain, reformer, reformer_fallback, history, fronts, openings, pop, p, n_states, fleets, tribes)

    n_nodes = np.array([int(x["n_nodes"]) for x in R["states"]], dtype=np.int64)
    capitals = [int(x["capital"]) for x in R["states"]]
    by_cls = {}
    for cname, ci in (("dense", 0), ("medium", 1)):
        sel = [s for s in range(n_states) if cls[capitals[s]] == ci]
        by_cls[cname] = {"n": len(sel), "median_nodes": float(np.median(n_nodes[sel])) if sel else 0.0,
                         "max_nodes": int(n_nodes[sel].max()) if sel else 0}
    n_cap1 = int(R["n_cap1"])
    return {"pop_total_M": round(float(pop.sum()) / 1e6, 1), "n_states": n_states,
            "n_capitals_by_strength": n_cap1, "n_singletons": n_states - n_cap1, "n_attached_loose": int(R["n_attached"]),
            "n_fleets": len(fleets), "n_tribes": len(tribes),
            "median_state_nodes": float(np.median(n_nodes)), "max_state_nodes": int(n_nodes.max()),
            "states_by_capital_class": by_cls,
            "reformer": reformer, "reformer_fallback": reformer_fallback,
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
