"""九格表草稿生成（docs/08-地区设计规程）。

铁律遵循：
- ⑨ 写强度不写有无、写相对不写绝对；相邻地区必须几乎相同
- 错位必须能由 ②b（障碍 × 模式）解释；至少一条「这条还没漂移」
- 不读 height_m（原则乙）；推不出的格标「待填」
输出：ninegrid/region_NNN.md + region_NNN.sources.json + index.md
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import MODES, MODE_ZH
from .culture import World
from .geology import Geology
from .polity import Polity
from .stages.s03_islands import CLASS_NAMES, CLASS_ZH
from .stages.s05_barriers import REGIONAL_ORDER
from .stages.s09_polity import REGIME_ZH, STAGE_ZH

BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
BAND_ZH = {0: "赤道缘", 1: "北信风带", 2: "北无风带", 3: "北西风带", 4: "北极地",
           5: "南信风带", 6: "南无风带", 7: "南西风带", 8: "南极地"}
CLASS_SHORT = {"dense": "密接", "medium": "中疏", "sparse": "稀疏", "isolated": "孤悬"}

FORBIDDEN_TEXT = re.compile(r"(高|上层).{0,6}(贵|尊)|(低|下层).{0,6}(贱)")


def q_share(p: float) -> str:
    if p >= 0.9:
        return "近乎全部"
    if p >= 0.7:
        return "约七八成"
    if p >= 0.55:
        return "约六成"
    if p >= 0.45:
        return "约半"
    if p >= 0.3:
        return "约三四成"
    if p >= 0.15:
        return "约两成，多见于老辈人家"
    if p >= 0.05:
        return "零星可见"
    return "几乎无人"


def _wrap_dlon(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def _rank_q(v: np.ndarray) -> np.ndarray:
    """各地区在全体地区中的分位 [0,1]（写相对不写绝对）。"""
    n = v.size
    if n <= 1:
        return np.full(n, 0.5)
    order = np.argsort(v, kind="stable")
    q = np.empty(n)
    q[order] = np.arange(n) / (n - 1)
    return q


def _spacing_zh(cls: str) -> str:
    """群间间距的量词（节点 = 岛群，分类阈值量的是群与群之间；群内永远是密接）。"""
    return {"dense": "不过数刻航程，可架索桥短渡", "medium": "小船一日之内",
            "sparse": "需好船数日", "isolated": "大船亦难一跳而至"}.get(cls, "")


def _pct_zh(x: float) -> str:
    return f"{100.0 * x:.0f}%" if x >= 0.01 else f"{100.0 * x:.1f}%"


def _wan_km2(x: float) -> str:
    return f"{x / 1e4:.1f} 万 km²" if x >= 1e4 else f"{x:.0f} km²"


class RegionData:
    """一次性汇总各地区的推导输入。"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.w = World(ctx)
        w = self.w
        self.isl = w.islands
        self.region = w.regions["region"]
        self.n_regions = int(self.region.max()) + 1
        self.seeds = w.centers["region_seeds"]
        self.clim = ctx.load_npz(4, "climate_islands")
        self.pre = ctx.load_npz(7, "prehist")
        try:
            self.wind = ctx.load_npz(4, "wind_local")   # 扰动后的风（第三批 3）
        except FileNotFoundError:
            self.wind = ctx.load_npz(2, "wind")
        self.prod = ctx.cfg.get("production_templates", {})
        ce = w.cand_edges
        self.src, self.dst = ce["src"], ce["dst"]
        self.E = self.src.size
        self.perm = w.perm["perm"]
        self.f_reg = w.perm["f_regional"]
        cost = w.routes["cost"]
        self.cost_ab, self.cost_ba = cost[:self.E], cost[self.E:]
        self.flow_und = w.routes["flow"][:self.E] + w.routes["flow"][self.E:]

        # 地区聚合
        self.members = [np.where(self.region == r)[0] for r in range(self.n_regions)]
        self.cls_major = [int(np.bincount(self.isl["cls"][m], minlength=4).argmax())
                          if m.size else 1 for m in self.members]
        self.layered_share = [float(self.isl["layered"][m].mean()) if m.size else 0.0
                              for m in self.members]
        # 陆地与集雨容量（docs/02 §六/§七）：节点 = 岛群 = 一水共同体 = 一个基本政治单位（R10）
        area_i = self.isl["area_km2"]            # 群的总陆地
        arable_i = area_i * self.isl["arable_frac"]   # 可耕地（R9）
        catch_i = self.clim["catch"]
        self.area_med = np.array([float(np.median(area_i[m])) if m.size else 0.0
                                  for m in self.members])
        self.area_max = np.array([float(area_i[m].max()) if m.size else 0.0
                                  for m in self.members])
        self.area_sum = np.array([float(area_i[m].sum()) if m.size else 0.0
                                  for m in self.members])
        self.arable_sum = np.array([float(arable_i[m].sum()) if m.size else 0.0
                                    for m in self.members])
        self.land_frac_med = np.array([float(np.median(self.isl["land_frac"][m])) if m.size else 0.0
                                       for m in self.members])
        # 口径人口（shared.scale P1）：只是量词，不是模型量
        self.people_per_km2 = float(ctx.cfg["shared"]["scale"]["people_per_arable_km2"])
        self.catch_med = np.array([float(np.median(catch_i[m])) if m.size else 0.0
                                   for m in self.members])
        self.area_q = _rank_q(self.area_med)
        self.catch_q = _rank_q(self.catch_med)
        # 主岛与河流（第三批第 1 步）：只进文本，不进任何推导
        main_i = self.isl["main_area_km2"] if "main_area_km2" in self.isl else area_i * 0.5   # 旧产物兼容（dict 或 NpzFile 都支持 in）
        river_i = self.clim["has_river"] if "has_river" in self.clim else np.zeros(area_i.size, bool)
        self.main_med = np.array([float(np.median(main_i[m])) if m.size else 0.0 for m in self.members])
        self.river_share = np.array([float(river_i[m].mean()) if m.size else 0.0 for m in self.members])
        from .stages.s02_wind import band_id_of
        planet = ctx.load_json(1, "planet")["bands"]
        try:
            band_local = ctx.load_npz(4, "band_local")
        except FileNotFoundError:
            band_local = None
        self.band_of = band_id_of(self.isl["lat"], self.isl["lon"], planet, band_local)
        self.band_major = [int(np.bincount(self.band_of[m]).argmax()) if m.size else 0
                           for m in self.members]
        # 邻接与跨界边
        ra, rb = self.region[self.src], self.region[self.dst]
        crossing = ra != rb
        self.adj: dict[int, dict[int, np.ndarray]] = {}
        ce_idx = np.where(crossing)[0]
        for e in ce_idx:
            a, b = int(ra[e]), int(rb[e])
            self.adj.setdefault(a, {}).setdefault(b, []).append(int(e))
            self.adj.setdefault(b, {}).setdefault(a, []).append(int(e))
        for a in self.adj:
            for b in self.adj[a]:
                self.adj[a][b] = np.array(self.adj[a][b])
        pf = [float(self.flow_und[self.adj[a][b]].sum())
              for a in self.adj for b in self.adj[a] if a < b]
        pf.sort(reverse=True)
        self.trunk_flow_thr = pf[max(0, len(pf) // 10)] if pf else 0.0
        # 政治层（⑨，第四批 R7）：地区是展示分区，邦是政治单位；九格表 ⑤⑥⑧ 写邦级
        self.pol = Polity(ctx)
        # 地质表现层（R4）：只出文字，不进推导
        self.geo = Geology(ctx)

    # ---- 政治层：一个地区里有哪些邦 ----
    def region_polities(self, r: int) -> list[dict]:
        """按本区内邑数降序：[{pid, n_here, n_total, share_of_state}]。"""
        if not self.pol.available:
            return []
        m = self.members[r]
        pids, counts = np.unique(self.pol.polity_of[m], return_counts=True)
        out = []
        for pid, n_here in zip(pids.tolist(), counts.tolist()):
            x = self.pol.P[int(pid)]
            out.append({"pid": int(pid), "n_here": int(n_here), "n_total": int(x["n_nodes"]),
                        "share_of_state": n_here / max(1, x["n_nodes"]), "kind": x["kind"]})
        out.sort(key=lambda d: (-d["n_here"], d["pid"]))
        return out

    # ---- ②b：与邻区之间的通道聚合 ----
    def crossing_info(self, r: int, t: int) -> dict:
        edges = self.adj[r][t]
        best_perm = {m: float(self.perm[edges, mi].max()) for mi, m in enumerate(MODES)}
        out_is_ab = self.region[self.src[edges]] == r
        toward = np.where(out_is_ab, self.cost_ab[edges], self.cost_ba[edges])
        away = np.where(out_is_ab, self.cost_ba[edges], self.cost_ab[edges])
        # 单向性看同一条最佳边的双向比（docs/02 §五 第 2 类：逆风带的不对称）
        k = int(np.argmin(toward + away))
        c_out = max(0.1, float(toward[k]))
        c_in = max(0.1, float(away[k]))
        ratio = c_out / c_in
        one_way = None
        if ratio <= 0.67:
            one_way = "去易回难"
        elif ratio >= 1.5:
            one_way = "去难回易"
        barriers = []
        f = self.f_reg[edges]
        for bi, bid in enumerate(REGIONAL_ORDER):
            if float(f[:, bi].max()) > 0.05:
                barriers.append(bid)
        seasonal = any(self.ctx.cfg["s05"]["barriers"][b].get("seasonal") for b in barriers)
        return {"perm": best_perm, "cost_out": round(c_out, 2), "cost_in": round(c_in, 2),
                "barriers": barriers, "seasonal": seasonal, "one_way": one_way,
                "flow": float(self.flow_und[edges].sum())}

    def name(self, r: int) -> str:
        seed = self.seeds[r]
        lon = float(self.isl["lon"][seed])
        band = BAND_ZH[self.band_major[r]]
        sect = BRANCHES[int(((lon + 180.0) % 360.0) // 30.0)]
        cshort = CLASS_SHORT[CLASS_NAMES[self.cls_major[r]]]
        return f"{band}·{sect}段·{cshort}-{r:03d}"


def _moisture(p: float) -> str:
    return "humid" if p >= 0.55 else ("moderate" if p >= 0.3 else "arid")


MOIST_ZH = {"humid": "湿润", "moderate": "适中", "arid": "干旱"}


def build_region_md(rd: RegionData, r: int) -> tuple[str, dict]:
    w = rd.w
    ctx = rd.ctx
    seed = int(rd.seeds[r])
    members = rd.members[r]
    name = rd.name(r)
    cls = CLASS_NAMES[rd.cls_major[r]]
    sources: dict = {"region": r, "seed_node": seed, "lines": []}

    neighbors = sorted(rd.adj.get(r, {}).keys())
    cross = {t: rd.crossing_info(r, t) for t in neighbors}

    # 上/下风邻区（信风为东风 → 下风在西；按种子处纬向风符号判定）
    from .sphere import grid_interp
    u_seed = float(grid_interp(rd.wind["u"].astype(np.float64), rd.wind["lats"], rd.wind["lons"],
                               rd.isl["lat"][seed], rd.isl["lon"][seed]))
    def is_downwind(t):
        dl = _wrap_dlon(float(rd.isl["lon"][rd.seeds[t]]), float(rd.isl["lon"][seed]))
        return dl * np.sign(u_seed) > 0

    free_n = [t for t in neighbors
              if min(cross[t]["perm"].values()) >= 0.5 and not cross[t]["barriers"]]
    barrier_n = [t for t in neighbors if cross[t]["barriers"]
                 or min(cross[t]["perm"].values()) < 0.3]
    dn_free = next((t for t in free_n if is_downwind(t)), None)
    up_free = next((t for t in free_n if not is_downwind(t)), None)

    # ---------- ①
    d_n = min((cross[t]["cost_out"] for t in neighbors), default=float("nan"))
    lay = rd.layered_share[r]
    # 节点 = 岛群（R10）：面积是群的总陆地；群内数十小岛、半小时可达，属第三层
    pop_wan = rd.arable_sum[r] * rd.people_per_km2 / 1e4
    l1 = (f"{CLASS_ZH[cls]}。本区 {members.size} 个岛群，每群含数十岛，群内半小时可达；"
          f"群与群相隔{_spacing_zh(cls)}，与最近邻区相距约 {max(d_n, 0.1):.1f} 日。"
          f" 群陆地中位约 {rd.area_med[r]:.0f} km²，最大者 {rd.area_max[r]:.0f} km²，"
          f"陆地占势力范围约 {_pct_zh(rd.land_frac_med[r])}；"
          f"全区陆地约 {_wan_km2(rd.area_sum[r])}，可耕约 {_wan_km2(rd.arable_sum[r])}，"
          f"按口径折合约 {pop_wan:.0f} 万口（群外即虚空，无垦荒无拓边）。")
    if lay > 0.15:
        tenths = "一二三四五六七八九"[min(8, max(0, int(lay * 10) - 1))]
        l1 += f" 约{tenths}成岛群呈叠层堆叠。"
    rs = rd.river_share[r]
    l1 += f" 每群以一座主岛为主（主岛中位约 {rd.main_med[r]:.0f} km²）；"
    if rs >= 0.85:
        l1 += "主岛几乎皆有常年河流。"
    elif rs >= 0.15:
        tenths_r = "一二三四五六七八九"[min(8, max(0, int(rs * 10) - 1))]
        l1 += f"约{tenths_r}成的群主岛有常年河流，余者全赖集雨。"
    else:
        l1 += "主岛少有河流，用水几乎全赖集雨。"
    geo_zh = rd.geo.region_zh(members, rd.isl["lon"], rd.isl["lat"])
    if geo_zh:
        l1 += " " + geo_zh

    # ---------- ②
    band_zh = BAND_ZH[rd.band_major[r]]
    wind_word = "东风（下风在西）" if u_seed < -1 else ("西风（下风在东）" if u_seed > 1 else "风微弱")
    trunk_flow = max((cross[t]["flow"] for t in neighbors), default=0.0)
    on_trunk = trunk_flow >= rd.trunk_flow_thr > 0
    l2 = f"位于{band_zh}，盛行{wind_word}。" + ("有干线航路过境。" if on_trunk else "不在干线上。")

    # ---------- ②b
    l2b_lines = []
    for t in neighbors:
        ci = cross[t]
        pm = ci["perm"]
        seg = (f"对 {rd.name(t)}：{ci['cost_out']:.1f} 日"
               f"（日常 {pm['daily']:.2f} / 商旅 {pm['trade']:.2f} / 使节 {pm['envoy']:.2f}"
               f" / 迁徙 {pm['migrate']:.2f}）")
        if ci["barriers"]:
            seg += f"，隔【{'、'.join(ci['barriers'])}】"
        if ci["seasonal"]:
            seg += "，仅季节窗口"
        if ci["one_way"]:
            seg += f"，{ci['one_way']}"
        l2b_lines.append(seg)

    # ---------- ③⑦（史前）
    d_pre = rd.pre["dist_pre"]
    arr = rd.pre["arrival_yr"]
    lineage = rd.pre["lineage"]
    lin_major = int(np.bincount(lineage[members]).argmax())
    arr_med = float(np.median(arr[members]))
    arr_q = float((arr <= arr_med).mean())
    era = "最早一批" if arr_q < 0.25 else ("较早" if arr_q < 0.5 else ("较晚" if arr_q < 0.8 else "最晚近"))
    parent = int(rd.pre["pred"][seed])
    if parent >= 0:
        dlon_p = _wrap_dlon(float(rd.isl["lon"][seed]), float(rd.isl["lon"][parent]))
        dlat_p = float(rd.isl["lat"][seed] - rd.isl["lat"][parent])
        dir8 = ("东" if dlon_p > 0 else "西") if abs(dlon_p) > abs(dlat_p) else ("北" if dlat_p > 0 else "南")
        from_dir = f"自{dir8}侧漂来"
    else:
        from_dir = "即史前扩散的起源地"
    same_lin = [t for t in neighbors if int(np.bincount(lineage[rd.members[t]]).argmax()) == lin_major]
    l3 = (f"{from_dir}，属谱系 L{lin_major}，为{era}到达的一支（顺风抱石而渡）。")
    if same_lin:
        l3 += f" 与 {rd.name(same_lin[0])} 同源。"
    l7 = (f"谱系 L{lin_major}，隔离时长与扩散距离成正比（约 {arr_med:.0f} 年前到达）。"
          f" 体质细节【待填（03-生态与人）】。")

    # ---------- ④⑤⑥
    precip_med = float(np.median(rd.clim["precip"][members]))
    moist = _moisture(precip_med)
    prod_t = rd.prod.get("production", {}).get(cls, {}).get(moist, {}).get("text", "【待填】")
    l4 = f"（{MOIST_ZH[moist]}）{prod_t}"
    if lay > 0.15:
        l4 += rd.prod.get("layered_suffix", {}).get("text", "")
    l4 += " 特有物种与专项特产【待填（政治层/03）】。"
    # ⑤ 组织：邑级（水共同体）+ 邦级（⑨ 政治层，第四批 R7）
    scale_tier = "large" if rd.area_q[r] >= 0.66 else ("small" if rd.area_q[r] < 0.33 else "mid")
    wp = rd.prod.get("water_polity", {}).get(scale_tier, {}).get("text", "")
    l5 = ("邑：" + wp) if wp else ""
    if rs >= 0.5:
        l5 += " 有河之群，取水不必尽赖集雨，掌水之政稍轻，而河谷田畴之争代之。"
    pol = rd.pol
    rp = rd.region_polities(r)
    states_here = [d for d in rp if d["kind"] == "state"]
    dom = states_here[0] if states_here else None
    if pol.available and rp:
        n_states_here = len(states_here)
        head = f" 邦：本区 {members.size} 邑分属 {n_states_here} 邦" if n_states_here else " 邦：本区无邦"
        others = [d for d in rp if d["kind"] != "state"]
        if others:
            head += "，另有" + "、".join(f"{pol.name(d['pid'])}（{d['n_here']} 邑）" for d in others[:2])
        l5 += head + "。"
        if dom:
            x = pol.P[dom["pid"]]
            where = ("都在本区" if rd.region[x["capital"]] == r else f"都 #{x['capital']} 在 {rd.name(int(rd.region[x['capital']]))}")
            l5 += (f" 最大者 {pol.name(dom['pid'])}：{x['n_nodes']} 邑、约 {x['pop'] / 1e4:.0f} 万口，{where}，"
                   f"本区 {dom['n_here']} 邑归之（占其 {dom['share_of_state'] * 100:.0f}%）；直辖 {x['n_direct']} 邑、采邑 {x['n_fiefs']} 处。"
                   f" {REGIME_ZH[x['regime']]}。")
            suz = pol.meta["suzerain"].get(x["circle"], -1)
            if suz >= 0 and suz != x["id"]:
                l5 += f" 名分上奉{pol.name(suz)}为宗主（{pol.meta['center_zh'][x['circle']]}），实无贡赋。"
            st = pol.status_zh(dom["pid"])
            if st:
                l5 += f" {st}。"
            if n_states_here > 1:
                rest = states_here[1:4]
                l5 += " 其余：" + "、".join(f"{pol.name(d['pid'])}（本区 {d['n_here']} 邑 / 共 {d['n_total']} 邑）" for d in rest) + "。"
    else:
        l5 += " " + rd.prod.get("organization", {}).get(cls, {}).get("text", "【待填】")
    # ⑥ 军事：地形类的形态 + 邦级的强弱（谁打不过它、它打不过谁）
    l6 = rd.prod.get("military", {}).get(cls, {}).get("text", "【待填】")
    nb_dense = [t for t in neighbors if CLASS_NAMES[rd.cls_major[t]] == "dense"]
    nb_sparse = [t for t in neighbors if CLASS_NAMES[rd.cls_major[t]] == "sparse"]
    if cls in ("medium", "dense") and nb_sparse:
        l6 += f" 邻近的{rd.name(nb_sparse[0])}船团劫掠不绝，防之不胜防。"
    if pol.available and dom:
        x = pol.P[dom["pid"]]
        nbs = [(int(t), n_e) for t, n_e in x.get("neighbors", {}).items()]
        stronger = sorted((t for t, _ in nbs if pol.P[t]["pop"] >= 2 * x["pop"]), key=lambda t: -pol.P[t]["pop"])
        weaker = sorted((t for t, _ in nbs if pol.P[t]["pop"] * 2 <= x["pop"]), key=lambda t: pol.P[t]["pop"])
        if stronger:
            l6 += f" 打不过{pol.short(stronger[0])}；"
        if weaker:
            l6 += f" {pol.name(weaker[0])}等{len(weaker)}邦打不过它；"
        if not stronger and not weaker:
            l6 += " 与接壤诸邦势均力敌；"
        rf = pol.reformer
        if x["id"] == rf:
            l6 += " 此即变法之国：编户直接征发，动员量级高一档。"
        elif x.get("annexed_by", -1) >= 0:
            l6 += f" 已为{pol.name(x['annexed_by'])}所并，驻军与改制在进行中。"
        elif any(f["polity"] == x["id"] for f in pol.meta.get("fronts", [])):
            l6 += f" 变法之国{pol.name(rf)}的兵锋已至。"
        elif cls == "medium" and nb_dense:
            l6 += f" 若{rd.name(nb_dense[0])}方向的密接之国东出，此地挡不住。"
    elif cls == "medium" and nb_dense:
        l6 += f" 若{rd.name(nb_dense[0])}方向的密接之国东出，此地挡不住。"

    # ---------- ⑧
    if neighbors:
        dep = max(neighbors, key=lambda t: cross[t]["flow"])
        marry = max(neighbors, key=lambda t: cross[t]["perm"]["daily"])
        l8 = (f"贸易依赖 {rd.name(dep)}（跨界流量最大）；与 {rd.name(marry)} 日常往来最密"
              f"（通婚方向）。")
        if pol.available and dom:
            x = pol.P[dom["pid"]]
            nbs = [int(t) for t in x.get("neighbors", {})]
            # 世仇：接壤、势均力敌（人口比在 1/2–2 之间）、界边最多者 —— 五百年割据里打不完的邻居
            peers = [t for t in nbs if 0.5 <= pol.P[t]["pop"] / max(1, x["pop"]) <= 2.0]
            if peers:
                feud = max(peers, key=lambda t: (x["neighbors"][str(t)], -t))
                l8 += f" 世仇：{pol.short(feud)}（接壤 {x['neighbors'][str(feud)]} 边，势均力敌，五百年打不完）。"
            if x.get("overlord", -1) >= 0:
                l8 += f" 附庸于{pol.short(x['overlord'])}，岁有贡献。"
            if x.get("vassals"):
                l8 += f" 附庸之邦 {len(x['vassals'])}：" + "、".join(pol.name(v) for v in x["vassals"][:3]) + "。"
            if x.get("annexed_by", -1) >= 0:
                z = STAGE_ZH[x["stage"]]
                l8 += f" 约 {x['annexed_years_ago']:.0f} 年前为{pol.name(x['annexed_by'])}所并，今至「{z[0]}」阶段：{z[1]}。"
        else:
            l8 += " 世仇与盟约【待填（政治层）】。"
    else:
        l8 = "无邻区记录（孤悬）。外购几不可得。"
    press_tier = "high" if rd.catch_q[r] < 0.33 else ("low" if rd.catch_q[r] >= 0.66 else "mid")
    pp = rd.prod.get("population_pressure", {}).get(press_tier, {}).get("text", "")
    if pp:
        l8 += " " + pp
    sources["scale"] = {"n_clusters": int(members.size),
                        "area_median_km2": round(float(rd.area_med[r]), 1),
                        "area_quantile": round(float(rd.area_q[r]), 2),
                        "land_total_km2": round(float(rd.area_sum[r]), 0),
                        "arable_total_km2": round(float(rd.arable_sum[r]), 0),
                        "land_frac_median": round(float(rd.land_frac_med[r]), 4),
                        "population_by_quota": round(float(pop_wan * 1e4), 0),
                        "catch_median": round(float(rd.catch_med[r]), 1),
                        "catch_quantile": round(float(rd.catch_q[r]), 2),
                        "main_area_median_km2": round(float(rd.main_med[r]), 0),
                        "river_share": round(float(rs), 2),
                        "water_polity_tier": scale_tier, "pressure_tier": press_tier}
    if pol.available:
        sources["polity"] = {"states_here": [(d["pid"], d["n_here"]) for d in rp][:8],
                             "dominant": (dom["pid"] if dom else None)}

    # ---------- ⑨（核心：写强度、写相对、错位由 ②b 解释）----------
    l9_lines = []
    targets = []
    if up_free is not None:
        targets.append(("上风近邻", up_free))
    if dn_free is not None and dn_free != up_free:
        targets.append(("下风近邻", dn_free))
    for t in barrier_n[:2]:
        targets.append(("跨障碍", t))
    if not targets and neighbors:
        targets = [("近邻", neighbors[0])]

    seed_arr = np.array([seed])
    for role, t in targets:
        t_seed = np.array([int(rd.seeds[t])])
        per_slot = {}
        for s in w.slots:
            p, _ = w.shares(s)
            home_lead = int(p[:, seed].argmax())
            per_slot[s] = {
                "tv": float(0.5 * np.abs(p[:, seed] - p[:, t_seed[0]]).sum()),
                "home_lead": home_lead,
                "p_home": float(p[home_lead, seed]),
                "p_there": float(p[home_lead, t_seed[0]]),
            }
        ranked = sorted(per_slot.items(), key=lambda kv: -kv[1]["tv"])
        drifted = [kv for kv in ranked if kv[1]["tv"] >= 0.2][:2]
        stable = [kv for kv in reversed(ranked) if kv[1]["tv"] < 0.1]
        # 「还没漂移」优先挑与已漂移槽位模式不同者（同言线互不重合的直接呈现）
        drift_modes = {w.slot_mode(kv[0]) for kv in drifted}
        stable_pick = next((kv for kv in stable if w.slot_mode(kv[0]) not in drift_modes),
                           stable[0] if stable else None)
        ci = cross[t] if t in cross else None
        head = f"相对{role} {rd.name(t)}"
        if ci:
            head += f"（{ci['cost_out']:.1f} 日{'，隔' + '、'.join(ci['barriers']) if ci['barriers'] else ''}）"
        l9_lines.append(f"**{head}**：")
        if not drifted:
            l9_lines.append("  · 几乎一致：各槽位比例差均不足半成。差别只在细枝末节。")
        slots_cfg = {s2["id"]: s2 for s2 in ctx.cfg["slots"]["slot"]}
        for sid, info in drifted:
            zh = slots_cfg[sid]["zh"]
            phrase = slots_cfg[sid]["phrase"]
            mode = w.slot_mode(sid)
            attr = ""
            if ci:
                pm = ci["perm"]
                others = [m for m in MODES if pm[m] >= 2 * pm[mode] + 0.1]
                if pm[mode] < 0.5 and others:
                    attr = f"（{MODE_ZH[mode]}通过率仅 {pm[mode]:.2f}，而{MODE_ZH[others[0]]}有 {pm[others[0]]:.2f} —— 这条走不过去）"
                elif mode == "daily":
                    attr = "（日常往来衰减最快，纯距离所致）"
            l9_lines.append(
                f"  · {zh}：本区{phrase}者{q_share(info['p_home'])}；彼处{q_share(info['p_there'])}{attr}")
            sources["lines"].append({
                "target_region": t, "slot": sid, "mode": mode,
                "tv": round(info["tv"], 3),
                "share_home": round(info["p_home"], 3), "share_there": round(info["p_there"], 3),
                "barriers": ci["barriers"] if ci else [],
                "probe": f"zhouzhu probe node {int(rd.seeds[t])}",
            })
        if stable_pick is not None:
            sid, info = stable_pick
            zh = slots_cfg[sid]["zh"]
            mode = w.slot_mode(sid)
            note = ""
            if drifted and mode not in drift_modes:
                note = f"（此条走{MODE_ZH[mode]}，与上面漂移的不是同一张网 —— 同言线互不重合）"
            l9_lines.append(f"  · {zh}：**这条还没漂移**，两地几乎相同{note}")
            sources["lines"].append({"target_region": t, "slot": sid, "mode": mode,
                                     "tv": round(info["tv"], 3), "stable": True})

    # 「传到了但不要」行（若本区存在 reach 高 adopt 低的特征）
    reach_f, adopt_f = w.fields["reach"], w.fields["adopt"]
    for t_meta in w.traits:
        ti = t_meta["index"]
        if t_meta["resistance"] < 0.7:
            continue
        m = (reach_f[ti][members] >= 0.3) & (adopt_f[ti][members] <= 0.4)
        if m.mean() >= 0.3:
            cb = w.fields["conflict_by"][ti, members[m.argmax()]]
            blocker = w.traits[cb]["id"] if cb >= 0 else "本地自有之俗"
            l9_lines.append(
                f"  · 【明拒】{t_meta['slot_zh']}的外来做法（{t_meta['id']}）在此人人知道，"
                f"只是不用——并非没传到（reach {float(reach_f[ti][members].mean()):.2f}），"
                f"是被{blocker}挡在门外。")
            sources["lines"].append({"slot": t_meta["slot"], "rejected_trait": t_meta["id"],
                                     "blocked_by": blocker})
            break
    l9 = "\n".join(l9_lines) if l9_lines else "【待填】孤悬地区，比较对象不足。"

    md = f"""# 九格表草稿 · {name}

> 自动生成（seed {ctx.seed}）。所有专名为结构性占位；标【待填】的格子超出地理推导范围。
> 地区含 {members.size} 岛；种子节点 {seed}（lat {rd.isl['lat'][seed]:.1f}, lon {rd.isl['lon'][seed]:.1f}）。

| 格 | 内容 |
|---|---|
| ① 地形 | {l1} |
| ② 气流 | {l2} |
| ②b 障碍 | {'；'.join(l2b_lines) if l2b_lines else '无邻区'} |
| ③ 扩散史 | {l3} |
| ④ 生产 | {l4} |
| ⑤ 组织 | {l5} |
| ⑥ 军事 | {l6} |
| ⑦ 体质 | {l7} |
| ⑧ 对外 | {l8} |

## ⑨ 文化位置（相对邻区，写强度不写有无）

{l9}
"""
    return md, sources


def lint_md(md: str) -> list[str]:
    problems = []
    if FORBIDDEN_TEXT.search(md):
        problems.append("出现以高低论贵贱的表述（违反原则乙）")
    # 只有存在无障碍近邻时才要求「还没漂移」行——跨障碍邻区全槽位漂移是正确行为
    has_free_target = ("相对上风近邻" in md) or ("相对下风近邻" in md) or ("相对近邻" in md)
    if has_free_target and "还没漂移" not in md and "几乎一致" not in md:
        problems.append("⑨ 缺少「还没漂移/几乎一致」行（相邻地区必须几乎相同）")
    return problems


def render_ninegrids(ctx, region: int | None = None) -> int:
    rd = RegionData(ctx)
    out_dir = ctx.stage_dir(10) / "ninegrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [region] if region is not None else range(rd.n_regions)
    index = ["# 九格表索引", ""]
    n_problems = 0
    for r in targets:
        md, sources = build_region_md(rd, r)
        problems = lint_md(md)
        if problems:
            n_problems += 1
            md += "\n> ⚠️ lint：" + "；".join(problems) + "\n"
        (out_dir / f"region_{r:03d}.md").write_text(md, encoding="utf-8")
        (out_dir / f"region_{r:03d}.sources.json").write_text(
            json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
        index.append(f"- [{rd.name(r)}](region_{r:03d}.md) — {rd.members[r].size} 岛")
    if region is None:
        (out_dir / "index.md").write_text("\n".join(index), encoding="utf-8")
    print(f"九格表：{len(list(targets))} 份 → {out_dir}（lint 提示 {n_problems} 份）")
    return len(list(targets))
