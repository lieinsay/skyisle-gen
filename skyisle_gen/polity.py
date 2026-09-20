"""政治层（⑨）的公共读取与文本（ninegrid / probe / web / cli 共用，只读产物）。

铁律：本模块不读 height_m；政体是离散单位，但文化仍只以 ⑧ 的连续场存在（政体不回写文化）。
"""
from __future__ import annotations

import numpy as np

from .stages.s09_polity import KIND_ZH, REGIME_ZH, STAGE_ZH


class Polity:
    """一次 run 的政治层产物（旧 run 没有 ⑨ 时 available = False，各处按无政治层退化）。"""

    def __init__(self, ctx):
        self.ctx = ctx
        try:
            self.arr = ctx.load_npz(9, "polity")
            self.meta = ctx.load_json(9, "polities")
            self.available = True
        except FileNotFoundError:
            self.arr, self.meta, self.available = None, None, False
            return
        self.P = {x["id"]: x for x in self.meta["polities"]}
        self.state = self.arr["state"]
        self.polity_of = self.arr["polity"]
        self.pop = self.arr["pop"]
        self.reformer = int(self.meta["reformer"]["polity"])

    # ---- 文本 ----
    def name(self, pid: int) -> str:
        if pid is None or pid < 0:
            return "（无）"
        return self.P[int(pid)]["name"]

    def short(self, pid: int) -> str:
        """邦名 + 规模。"""
        if pid is None or pid < 0:
            return "（无）"
        x = self.P[int(pid)]
        if x["kind"] != "state":
            return f"{x['name']}（{KIND_ZH[0 if x['kind'] == 'state' else (1 if x['kind'] == 'fleet' else 2)]}，{x['n_nodes']} 邑）"
        return f"{x['name']}（都 #{x['capital']}，{x['n_nodes']} 邑，约 {x['pop'] / 1e4:.0f} 万口）"

    def regime_zh(self, pid: int) -> str:
        return REGIME_ZH.get(self.P[int(pid)]["regime"], "")

    def status_zh(self, pid: int) -> str:
        """兼并状态：已并（阶段）/ 正被攻伐 / 对峙 / 附庸 / 无。"""
        x = self.P[int(pid)]
        if x["kind"] != "state":
            return ""
        parts = []
        if x.get("annexed_by", -1) >= 0:
            z = STAGE_ZH[x["stage"]]
            parts.append(f"约 {x['annexed_years_ago']:.0f} 年前为{self.name(x['annexed_by'])}所并，今至「{z[0]}」阶段（{z[1]}）")
        for f in self.meta.get("fronts", []):
            if f["polity"] == x["id"]:
                parts.append("正被变法之国攻伐" if f["feasible"] else "与变法之国对峙")
        if x.get("overlord", -1) >= 0:
            parts.append(f"附庸于{self.name(x['overlord'])}")
        if x.get("vassals"):
            parts.append(f"有附庸 {len(x['vassals'])} 邦")
        if x.get("is_suzerain"):
            parts.append("本圈宗主")
        if x["id"] == self.reformer:
            parts.append("变法之国（兼并者）")
        return "；".join(parts)

    def node_line(self, j: int) -> str:
        """探针 / 九格表用的一行：此邑属于谁、直辖还是采邑、控制力。"""
        pid = int(self.polity_of[j])
        x = self.P[pid]
        if x["kind"] != "state":
            return f"{x['name']}（{REGIME_ZH[x['regime']]}；{x['n_nodes']} 邑）"
        f = int(self.arr["fief"][j])
        if j == x["capital"]:
            role = "都城"
        elif f < 0:
            role = "都城直辖"
        elif f == j:
            role = "采邑之主（封君所在）"
        else:
            role = f"采邑（主在 #{f}）"
        line = (f"{x['name']}（都 #{x['capital']}，{x['n_nodes']} 邑，约 {x['pop'] / 1e4:.0f} 万口）· {role}"
                f" · 控制力 {float(self.arr['control'][j]):.2f} · {REGIME_ZH[x['regime']]}")
        st = self.status_zh(pid)
        return line + (f" · {st}" if st else "")


def print_summary(ctx, top: int = 15) -> None:
    pol = Polity(ctx)
    if not pol.available:
        print("此 run 没有 ⑨ 政治层产物（旧版本）；请 `skyisle stage 9`。")
        return
    m = pol.meta
    states = [x for x in m["polities"] if x["kind"] == "state"]
    print(f"== 政治层（seed {ctx.seed}）==")
    print(f"邦 {m['n_states']}  船团 {m['n_fleets']}  部落 {m['n_tribes']}  人口 {float(pol.pop.sum()) / 1e8:.2f} 亿")
    print("宗主：" + "  ".join(f"{m['center_zh'][c]} → {pol.short(s)}" for c, s in m["suzerain"].items()))
    r = m["reformer"]
    print(f"变法之国：{pol.short(r['polity'])}  已并 {r['n_annexed']} 邦  本朝人口 {r['realm_pop'] / 1e4:.0f} 万"
          + ("  【退化：圈内无密接之邦】" if r["fallback"] else ""))
    for h in m["history"]:
        z = STAGE_ZH[h["stage"]]
        print(f"  距今 {h['years_ago']:5.0f} 年 并 {pol.short(h['polity'])}  → {z[0]}")
    for f in m["fronts"]:
        print(f"  {'正攻伐' if f['feasible'] else '对峙'} {pol.short(f['polity'])}")
    print(f"\n最大的 {top} 邦：")
    for x in sorted(states, key=lambda x: (-x["pop"], x["id"]))[:top]:
        print(f"  {x['name']} 都 #{x['capital']:5d} {x['capital_class']:7s} {x['circle']:10s} {x['n_nodes']:4d} 邑"
              f" {x['pop'] / 1e4:7.0f} 万  直辖 {x['n_direct']:3d} 采邑 {x['n_fiefs']:3d}  {REGIME_ZH[x['regime']][:10]}"
              f"  {pol.status_zh(x['id'])}")
    sizes = np.array([x["n_nodes"] for x in states])
    print(f"\n邦的规模：中位 {np.median(sizes):.0f} 邑，≥50 邑 {int((sizes >= 50).sum())}，10–49 邑 {int(((sizes >= 10) & (sizes < 50)).sum())}，"
          f"<10 邑 {int((sizes < 10).sum())}，独邑 {int((sizes == 1).sum())}")
    o = m["openings"]
    print("开局候选：A " + "；".join(pol.short(s) for s in o["A_frontier_small_state"]) + f"\n          B {pol.short(o['B_orthodox_core'])}"
          + "\n          C " + "；".join(pol.short(s) for s in o["C_transition_zone"]) + f"\n          D {pol.short(o['D_reformer'])}")
