"""5.3c 地形区与资源分布：山区 / 丘陵 / 台地 / 河谷的区划，和露天矿、采石场、洞穴、黏土、砂砾、泥炭、温泉、泉眼、林木等矿点。

全部从已生成的地形、水系、地表与节点的地质背景（岛龄、板块边界类型与远近、叠层）推出，不读人口、不回灌（第三层）。
随机数只走 _rng(node, "resources:…")。三种铺法（DESIGN-NOTES 四点十六 / 四点十七，按现实的量级定）：
- 稀有资源（露天矿、温泉、硫磺、泉眼、洞穴…）：每 100 km² 陆地的期望个数 × 地质倍率，泊松抽个数，在候选格里按「适宜度 × 斑块噪声」贪心取种子，长成斑块。
  露天矿是顺板块走向拉长的「矿化带」（1–5 km²），带内再点几个矿坑；板块内部的岛（边界核 ≈ 0）几乎不出金属矿（现实：夏威夷、冰岛没有可采金属矿，岛弧才多）。
- 散装建材（采石场、黏土坑）：按覆盖铺——把可居住的地切成 bulk_block_km 见方的块，每块里有合适的格就放一处小坑（前工业时代每个村附近都有自己的石坑、土坑）。
- 浮石露头：岛体本身就是浮石，所有崖面、深切的峡谷壁按露头率露出（汇聚带 / 叠层多、老岛盖得厚少）；可开采，采掉的量相对岛体微不足道，不影响浮空。
点状资源（洞穴、泉眼、温泉）只占一格。

产物：terrain_zone（uint8，见 ZONE_NAMES）与 resource（uint8，见 RES_NAMES，重叠时后画的优先）进 terrain.npz；
resources.json 列出每个矿点；resources.png 索引色；preview_resources.png 总览。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grid import FractalNoise, binary_dilate, label_components, window_extrema

ZONE_NAMES = ["虚空", "高山", "山地", "丘陵", "台地平原", "河谷", "崖缘", "水域"]
ZONE_PALETTE = [(20, 24, 40), (235, 235, 240), (150, 110, 80), (200, 170, 110), (170, 200, 120), (110, 190, 170), (90, 80, 75), (40, 90, 200)]

# 资源类：键 → (中文, 调色, 形态)。painting 顺序 = 本表顺序（后画覆盖先画）
RES_KINDS = [
    ("timber", "林木", (60, 130, 60), "patch"),
    ("spring", "泉眼", (120, 220, 255), "point"),
    ("clay", "黏土", (190, 120, 80), "patch"),
    ("peat", "泥炭 / 芦苇", (110, 90, 60), "patch"),
    ("gravel", "砂砾", (205, 195, 170), "patch"),
    ("placer", "砂金", (255, 215, 80), "patch"),
    ("quarry", "采石场", (170, 170, 185), "patch"),
    ("floatstone", "浮石", (150, 100, 230), "patch"),
    ("ore", "露天矿", (200, 60, 50), "patch"),
    ("hotspring", "温泉", (255, 140, 180), "point"),
    ("sulfur", "硫磺", (230, 230, 60), "patch"),
    ("cave", "洞穴", (40, 40, 40), "point"),
    ("guano", "鸟粪石", (240, 240, 210), "patch"),
]
RES_NAMES = ["无"] + [k[1] for k in RES_KINDS]
RES_PALETTE = [(0, 0, 0)] + [k[2] for k in RES_KINDS]
RES_INDEX = {k[0]: i + 1 for i, k in enumerate(RES_KINDS)}

# 露天矿的矿种权重（按最近的板块边界类型）；老岛加锡钨，新岛加硫化铜
ORE_WEIGHTS = {"汇聚": {"铜": 0.35, "铁": 0.25, "铅锌": 0.25, "金": 0.15},
               "离散": {"铁": 0.5, "铜": 0.3, "锰": 0.2},
               "走滑": {"铁": 0.4, "铜": 0.25, "锡": 0.2, "铅锌": 0.15}}


def _km(J, i, j):
    x0, y0 = J["raster"]["origin_km"]
    r = J["raster"]["res_m"] / 1000.0
    return [round(x0 + (j + 0.5) * r, 3), round(y0 - (i + 0.5) * r, 3)]


def _pick(rng, weights: dict) -> str:
    ks = sorted(weights)
    p = np.array([weights[k] for k in ks], dtype=float)
    return ks[int(rng.choice(len(ks), p=p / p.sum()))]


def _seeds(score: np.ndarray, n: int, min_sep: float) -> list[tuple[int, int]]:
    """贪心取种子：按分数降序，与已选的距离 ≥ min_sep 格。"""
    if n <= 0:
        return []
    flat = score.ravel()
    cand = np.where(flat > 0)[0]
    if cand.size == 0:
        return []
    order = cand[np.argsort(-flat[cand], kind="stable")][:40000]
    W = score.shape[1]
    out: list[tuple[int, int]] = []
    pts = np.zeros((0, 2))
    for k in order.tolist():
        i, j = divmod(k, W)
        if pts.shape[0] and (((pts - (i, j)) ** 2).sum(1) < min_sep * min_sep).any():
            continue
        out.append((i, j))
        pts = np.vstack([pts, (i, j)])
        if len(out) >= n:
            break
    return out


def _grow(seed, cand: np.ndarray, score: np.ndarray, n_cells: int, taken: np.ndarray,
          axis: float | None = None, elong: float = 1.0) -> np.ndarray:
    """从种子向外长斑块：窗内候选格按（距离 − 分数加权）排序取前 n_cells 个。返回 (行, 列) 下标。
    axis（弧度，x 东 y 北）+ elong > 1 时距离按椭圆量（沿 axis 拉长）：矿化带顺板块走向成条。"""
    H, W = cand.shape
    R = int(math.ceil(math.sqrt(max(1, n_cells) / math.pi) * 1.8 * math.sqrt(max(1.0, elong)))) + 1
    i, j = seed
    r0, r1, c0, c1 = max(0, i - R), min(H, i + R + 1), max(0, j - R), min(W, j + R + 1)
    ii, jj = np.mgrid[r0:r1, c0:c1]
    ok = cand[r0:r1, c0:c1] & ~taken[r0:r1, c0:c1]
    ok[i - r0, j - c0] = True
    if axis is not None and elong > 1.0:
        dx, dy = jj - j, -(ii - i)
        along = dx * math.cos(axis) + dy * math.sin(axis)
        across = -dx * math.sin(axis) + dy * math.cos(axis)
        d = np.hypot(along / elong, across) / max(1.0, R / math.sqrt(elong))
    else:
        d = np.hypot(ii - i, jj - j) / max(1.0, R)
    key = np.where(ok, d - 0.35 * score[r0:r1, c0:c1], np.inf).ravel()
    sel = np.argsort(key, kind="stable")[:max(1, n_cells)]
    sel = sel[np.isfinite(key[sel])]
    return ii.ravel()[sel], jj.ravel()[sel]


def build_resources(ctx, node: int, c: dict, g: dict, log=print) -> None:
    from . import _rng
    rc = c["resources"]
    J = g["json"]
    inp = g["inp"]
    res_km = g["res_km"]
    res_m = res_km * 1000.0
    cell_km2 = res_km * res_km
    island_id = g["island_id"]
    land = island_id >= 0
    H, W = land.shape
    h = np.where(land, g["height"], np.nan)
    hz = np.where(land, g["height"], 0.0)
    slope = g["slope_deg"].astype(np.float64)
    cover = g["landcover"]
    river = g["river"] > 0
    stream = g["stream"] > 0
    lake = g["lake"]
    water = river | lake
    cliff = g["cliff"]
    flood = g.get("floodplain", np.zeros((H, W), dtype=bool))
    acc = g["flowacc_km2"].astype(np.float64)
    T = inp["temp_sea"] - inp["lapse_c_per_km"] * hz / 1000.0
    meta = J["meta"]
    btype, kern, layered = meta["boundary_type"], float(meta["boundary_kernel"]), bool(meta["layered"])
    x0, y0 = J["raster"]["origin_km"]
    Xk = x0 + (np.arange(W) + 0.5) * res_km
    Yk = y0 - (np.arange(H) + 0.5) * res_km
    XX, YY = np.meshgrid(Xk, Yk)

    # ---------- 地形区 ----------
    r_cells = max(1, int(round(float(rc["relief_km"]) / res_km)))
    hi, lo = window_extrema(hz, r_cells, land)
    relief = np.where(land, hi - lo, 0.0)
    peak_rel = np.zeros((H, W))
    for k, isl in enumerate(J["islands"]):
        m = island_id == k
        if m.any():
            peak_rel[m] = (hz[m] - isl["rim_m"]) / max(1.0, isl["peak_m"] - isl["rim_m"])
    # 山地：起伏大或坡陡，或岛的高处（高出岸缘到峰高的 mountain_peak_frac 以上且起伏够丘陵）；丘陵同理
    mount = land & ((relief >= float(rc["mountain_relief_m"])) | (slope >= float(rc["mountain_slope_deg"]))
                    | ((peak_rel >= float(rc["mountain_peak_frac"])) & (relief >= 0.5 * float(rc["hill_relief_m"]))))
    hill = land & ~mount & ((relief >= float(rc["hill_relief_m"])) | (slope >= float(rc["hill_slope_deg"]))
                            | (peak_rel >= float(rc["hill_peak_frac"])))
    alpine = mount & ((T < float(c["landcover"]["alpine_temp_c"])) | (peak_rel >= float(rc["high_peak_frac"])))
    valley = land & (flood | (binary_dilate(river, 2) & (slope < float(rc["valley_slope_deg"]))))
    zone = np.zeros((H, W), dtype=np.uint8)
    zone[land] = 4
    zone[hill] = 3
    zone[mount] = 2
    zone[alpine] = 1
    zone[valley & ~mount] = 5
    zone[cliff] = 6
    zone[water] = 7

    # ---------- 资源 ----------
    res = np.zeros((H, W), dtype=np.uint8)
    taken = np.zeros((H, W), dtype=bool)
    deposits: list[dict] = []
    dens = rc["density_per_100km2"]
    noise = FractalNoise(_rng(ctx, node, "resources:noise"), Xk[0], Yk[-1], Xk[-1], Yk[0],
                         feature_km=float(rc["patch_km"]), octaves=3, persistence=0.5).sample(XX, YY)
    patchy = np.clip(0.5 + 0.5 * noise, 0.0, 1.0)
    soft = land & ~water & ~cliff
    # 金属矿只在板块边界附近：倍率 = max(下限, 边界类型增益 × 边界核)，叠层再乘；板块内部的岛（核 ≈ 0）几乎没有
    geo_ore = max(float(rc["ore_floor"]), float(rc["ore_gain"][btype]) * kern) * (float(rc["ore_layered_gain"]) if layered else 1.0)
    axis = math.radians(float(meta.get("boundary_axis_deg", 0.0)))
    cut = g.get("cut_m", np.zeros((H, W), dtype=np.float32)).astype(np.float64)
    # 浮石露头率：岛体就是浮石；汇聚带 / 叠层露得多，老岛风化层厚露得少
    fs_rate = float(rc["floatstone_expose"]) * (1.0 + float(rc["floatstone_convergent_gain"]) * kern * (btype == "汇聚")) * (float(rc["floatstone_layered_gain"]) if layered else 1.0)
    noise_fs = FractalNoise(_rng(ctx, node, "resources:floatstone"), Xk[0], Yk[-1], Xk[-1], Yk[0],
                            feature_km=float(rc["floatstone_patch_km"]), octaves=3, persistence=0.5).sample(XX, YY)

    # 群的老岛岩性：石灰岩（有溶洞）或砂岩；一群一个，确定性
    rng_lith = _rng(ctx, node, "resources:lith")
    old_limestone = bool(rng_lith.random() < float(rc["old_limestone_p"]))

    def lith(age_zh: str) -> str:
        return {"新岛": "玄武岩", "中年": "安山岩 / 凝灰岩", "老岛": "石灰岩" if old_limestone else "砂岩"}[age_zh]

    def add_deposit(kind, ii, jj, i, j, sub, q, note=None, extra=None):
        area = ii.size * cell_km2
        d = {"id": len(deposits), "kind": kind, "kind_zh": RES_NAMES[RES_INDEX[kind]], "subtype": sub,
             "island": int(island_id[i, j]), "cell": [int(i), int(j)], "km": _km(J, i, j), "area_km2": round(area, 3),
             "elev_m": round(float(hz[i, j]), 0), "slope_deg": round(float(slope[i, j]), 1),
             "zone": ZONE_NAMES[int(zone[i, j])], "grade": "上" if q >= 0.75 else ("中" if q >= 0.45 else "下")}
        if note:
            d["note"] = note
        if extra:
            d.update(extra)
        deposits.append(d)

    def place(kind: str, cand: np.ndarray, score: np.ndarray, n: int, area_med_km2: float, sep_km: float,
              sub_fn=None, note: str | None = None, island: int = -1, rng=None, seeds=None, elong: float = 1.0, pits: bool = False):
        if n <= 0 or not cand.any():
            return 0
        sc = np.where(cand, score, 0.0)
        if seeds is None:
            seeds = _seeds(sc, n, sep_km / res_km)
        _, _, _, form = RES_KINDS[RES_INDEX[kind] - 1]
        placed = 0
        for (i, j) in seeds:
            if form == "point":
                ii, jj = np.array([i]), np.array([j])
                area = cell_km2
            else:
                area = float(area_med_km2 * rng.lognormal(0.0, 0.6))
                ii, jj = _grow((i, j), cand, sc, max(1, int(round(area / cell_km2))), taken, axis=axis, elong=elong)
            res[ii, jj] = RES_INDEX[kind]
            taken[ii, jj] = True
            sub = sub_fn(i, j, rng) if sub_fn else None
            extra = None
            if pits and ii.size:
                # 带内的矿坑：品位最高的几格，彼此 ≥ 3 格
                n_p = int(min(int(rc["ore_pits_max"]), 1 + rng.poisson(ii.size * cell_km2)))
                o = np.argsort(-sc[ii, jj], kind="stable")
                chosen = []
                for t in o.tolist():
                    if all((ii[t] - a) ** 2 + (jj[t] - b) ** 2 >= 9 for a, b in chosen):
                        chosen.append((int(ii[t]), int(jj[t])))
                    if len(chosen) >= n_p:
                        break
                extra = {"pits": [[a, b] for a, b in chosen]}
            add_deposit(kind, ii, jj, i, j, sub, float(sc[i, j]), note, extra)
            placed += 1
        return placed

    def cover_place(kind: str, target: np.ndarray, cand: np.ndarray, fallback: np.ndarray | None, score: np.ndarray, block_km: float,
              area_med_km2: float, rng, sub_fn=None, note: str | None = None) -> int:
        """按覆盖铺：target（可居住的地）切成 block_km 见方的块，每块在 cand（没有就 fallback）里取分最高的格长一处小坑。"""
        b = max(1, int(round(block_km / res_km)))
        rr, cc = np.where(target)
        if rr.size == 0:
            return 0
        n = 0
        sc_c = np.where(cand, score, -1.0)
        sc_f = np.where(fallback, score, -1.0) if fallback is not None else None
        blocks = sorted(set(zip((rr // b).tolist(), (cc // b).tolist())))
        for bi, bj in blocks:
            sl = (slice(bi * b, (bi + 1) * b), slice(bj * b, (bj + 1) * b))
            for sc_ in (sc_c, sc_f):
                if sc_ is None:
                    continue
                win = sc_[sl]
                if win.size and win.max() > 0:
                    t = int(np.argmax(win))
                    i, j = bi * b + t // win.shape[1], bj * b + t % win.shape[1]
                    if taken[i, j]:
                        break
                    area = float(area_med_km2 * rng.lognormal(0.0, 0.5))
                    ii, jj = _grow((i, j), sc_ > 0, np.maximum(sc_, 0.0), max(1, int(round(area / cell_km2))), taken)
                    res[ii, jj] = RES_INDEX[kind]
                    taken[ii, jj] = True
                    add_deposit(kind, ii, jj, i, j, sub_fn(i, j, rng) if sub_fn else None, float(sc_[i, j]), note)
                    n += 1
                    break
        return n

    for k, isl in enumerate(J["islands"]):
        m = island_id == k
        if not m.any():
            continue
        A = float(m.sum()) * cell_km2
        age_zh = isl["age_zh"]
        rng = _rng(ctx, node, f"resources:{k}")
        lam = lambda key, mult=1.0: rng.poisson(max(0.0, float(dens[key]) * A / 100.0 * mult))
        mk = m & soft
        big = A >= float(rc["min_island_km2"])
        # 林木：大片林地（连通块 ≥ timber_min_km2），针叶 / 阔叶按年均温
        forest = m & (cover == 4)
        if forest.any():
            lab, nl = label_components(forest)
            if nl:
                cnt = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
                keep = [int(x) + 1 for x in np.argsort(-cnt, kind="stable")[:int(rc["timber_max_per_island"])] if cnt[x] * cell_km2 >= float(rc["timber_min_km2"])]
                for lb in keep:
                    cm = lab == lb
                    ii, jj = np.where(cm)
                    t_mean = float(T[cm].mean())
                    ci = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
                    res[cm & (res == 0)] = RES_INDEX["timber"]
                    deposits.append({"id": len(deposits), "kind": "timber", "kind_zh": "林木",
                                     "subtype": "针叶林" if t_mean < float(rc["conifer_temp_c"]) else ("针阔混交林" if t_mean < float(rc["conifer_temp_c"]) + 5 else "阔叶林"),
                                     "island": k, "cell": [int(ii[ci]), int(jj[ci])], "km": _km(J, int(ii[ci]), int(jj[ci])),
                                     "area_km2": round(float(cm.sum()) * cell_km2, 3), "elev_m": round(float(hz[cm].mean()), 0),
                                     "slope_deg": round(float(slope[cm].mean()), 1), "zone": ZONE_NAMES[int(zone[ii[ci], jj[ci]])],
                                     "grade": "上" if cm.sum() * cell_km2 >= 4 * float(rc["timber_min_km2"]) else "中"})
        # 泉眼：溪涧源头（本格有溪、上游八邻无溪），坡度转缓处优先
        if stream[m].any():
            st = m & stream
            nb = np.zeros((H, W), dtype=int)
            from .grid import shift, N8
            for di, dj in N8:
                nb += shift(st, di, dj, False)
            heads = st & (nb <= 1) & soft
            place("spring", heads, patchy * np.clip(1.2 - slope / 25.0, 0.05, 1.0), lam("spring"), 0, float(rc["spring_sep_km"]), rng=rng)
        # 黏土：漫滩 / 湖滨 / 湿地边，缓坡
        lake_edge = binary_dilate(lake, 2) & ~lake
        wet_edge = binary_dilate(cover == 9, 1)
        habitable = mk & (slope < float(rc["habitable_slope_deg"]))
        near_stream = binary_dilate(stream | river, int(rc["clay_stream_cells"]))
        cand = mk & (slope < float(rc["clay_slope_max_deg"])) & (flood | lake_edge | wet_edge)
        fb = mk & (slope < float(rc["clay_slope_max_deg"])) & near_stream
        cover_place("clay", habitable, cand, fb, 0.4 + 0.6 * patchy, float(rc["bulk_block_km"]), float(rc["clay_km2"]), rng,
              sub_fn=lambda i, j, r: "河湖黏土" if (flood[i, j] or lake_edge[i, j] or wet_edge[i, j]) else "溪边黏土")
        # 泥炭（凉湿）/ 芦苇荡（暖）：湿地
        cand = m & (cover == 9)
        place("peat", cand, 0.3 + 0.7 * patchy, lam("peat"), float(rc["peat_km2"]), float(rc["patch_sep_km"]),
              sub_fn=lambda i, j, r: "泥炭" if T[i, j] < float(rc["peat_temp_max_c"]) else "芦苇荡", rng=rng)
        # 砂砾（在露天矿之后抽：本岛有金 / 铜矿时一部分成砂金）：常年河 / 大溪边的缓坡滩地
        chan = m & (river | (stream & (acc >= float(rc["gravel_acc_km2"]))))
        # 露天矿：裸露基岩（裸岩 / 高山 / 灌丛 / 陡坡）且局地起伏大；不在崖缘（崖面开不了露天矿）
        exposed = (cover == 2) | (cover == 3) | (cover == 5) | (slope >= float(rc["ore_slope_min_deg"]))
        cand = mk & ((zone == 1) | (zone == 2) | (zone == 3)) & (slope <= float(rc["ore_slope_max_deg"]))
        w_ore = dict(ORE_WEIGHTS.get(btype, ORE_WEIGHTS["走滑"]))
        if age_zh == "老岛":
            w_ore["锡钨"] = 0.2
        if age_zh == "新岛":
            w_ore["铜"] = w_ore.get("铜", 0) + 0.15
        age_mult = {"新岛": 0.6, "中年": 1.0, "老岛": 1.2}[age_zh]
        if big:
            place("ore", cand, patchy * np.clip(relief / (2.0 * float(rc["mountain_relief_m"])), 0.1, 1.0) * np.where(exposed, 1.0, 0.6),
                  lam("ore", geo_ore * age_mult), float(rc["ore_km2"]), float(rc["ore_sep_km"]), sub_fn=lambda i, j, r: _pick(r, w_ore), rng=rng,
                  elong=float(rc["ore_elongation"]), pits=True, note="矿化带（顺板块走向）；pits = 带内矿坑")
        ore_kinds = {d["subtype"] for d in deposits if d["kind"] == "ore" and d["island"] == k}
        gold = bool(ore_kinds & {"金", "铜"})
        n_gr = lam("gravel")
        cand = mk & binary_dilate(chan, 1) & ~chan & (slope < float(rc["gravel_slope_max_deg"]))
        if gold and n_gr:
            n_pl = int(rng.binomial(n_gr, float(rc["placer_p"])))
            place("placer", cand, 0.3 + 0.7 * patchy, n_pl, float(rc["gravel_km2"]), float(rc["patch_sep_km"]),
                  note="本岛有金 / 铜矿化带：河砂可淘金", rng=rng)
            n_gr -= n_pl
        place("gravel", cand, 0.3 + 0.7 * patchy, n_gr, float(rc["gravel_km2"]), float(rc["patch_sep_km"]), rng=rng)
        # 采石场：中陡坡、薄土、非耕地；岩性按岛龄
        # 采石场：按覆盖铺（每块一处小石坑）；候选 = 中陡坡非耕地，块里没有就退到块里最陡的非耕地
        cand = mk & (slope >= float(rc["quarry_slope_min_deg"])) & (slope <= float(rc["quarry_slope_max_deg"])) & (g["arable"] == 0)
        fb = mk & (g["arable"] == 0)
        n_q0 = sum(1 for d in deposits if d["kind"] == "quarry")
        cover_place("quarry", habitable, cand, fb, np.clip(slope / 30.0, 0.05, 1.0) * (0.5 + 0.5 * patchy), float(rc["bulk_block_km"]),
              float(rc["quarry_km2"]), rng, sub_fn=lambda i, j, r: lith(age_zh))
        if k == 0 and sum(1 for d in deposits if d["kind"] == "quarry") == n_q0 and fb.any():   # 主岛至少一处（小岛全是缓坡时）
            place("quarry", fb, np.clip(slope / 30.0, 0.05, 1.0), 1, float(rc["quarry_km2"]), 1.0, sub_fn=lambda i, j, r: lith(age_zh), rng=rng)
        # 浮石露头：崖面 + 深切峡谷壁；露头率 × 岛龄（新岛 1.2 / 中年 1 / 老岛 0.6），按斑块噪声取格，连通段各算一处
        f_exp = float(np.clip(fs_rate * {"新岛": 1.2, "中年": 1.0, "老岛": 0.6}[age_zh], 0.02, 0.95))
        fs_cand_c = m & cliff & ~water
        fs_cand_g = mk & (cut >= float(rc["gorge_cut_m"])) & (slope >= float(rc["gorge_slope_deg"]))
        for fs_cand, sub_fs in ((fs_cand_c, "崖面露头"), (fs_cand_g, "峡谷露头")):
            if not fs_cand.any():
                continue
            thr = float(np.quantile(noise_fs[fs_cand], 1.0 - f_exp))
            ex = fs_cand & (noise_fs >= thr) & ~taken
            lab, nl = label_components(ex, connectivity=8)
            if not nl:
                continue
            cnt = np.bincount(lab.ravel(), minlength=nl + 1)
            for lb in range(1, nl + 1):
                if cnt[lb] < int(rc["floatstone_min_cells"]):
                    continue
                ii, jj = np.where(lab == lb)
                ci = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
                res[ii, jj] = RES_INDEX["floatstone"]
                taken[ii, jj] = True
                add_deposit("floatstone", ii, jj, int(ii[ci]), int(jj[ci]), sub_fs, float(np.clip(0.5 + 0.5 * noise_fs[ii, jj].mean(), 0, 1)),
                            note="岛体本身的浮石：可开采，采掉的量相对岛体微不足道，不影响浮空")
        # 温泉 / 硫磺：新岛（火山余热）；中年岛偶有温泉
        if age_zh != "老岛":
            hot = {"新岛": 1.0, "中年": float(rc["mid_hot_mult"])}[age_zh]
            cand = mk & (peak_rel >= 0.15) & (slope < 25)
            place("hotspring", cand & (stream | binary_dilate(stream, 1)), patchy + 0.3 * peak_rel, lam("hotspring", hot), 0, float(rc["spring_sep_km"]), rng=rng)
            if age_zh == "新岛" and big:
                cand = mk & (peak_rel >= float(rc["sulfur_peak_frac"]))
                place("sulfur", cand, patchy * peak_rel, lam("sulfur"), float(rc["sulfur_km2"]), float(rc["patch_sep_km"]), note="火山口 / 喷气孔", rng=rng)
        # 洞穴：熔岩管（新岛缓坡）/ 溶洞（石灰岩老岛的台地与落水洞）/ 崖洞（岸崖，所有岛）
        # 熔岩管：新岛多，中年岛仍在但多处塌成天窗（现实：几万年的岩流里还有十几 km 的管，上百万年才塌尽），老岛没有
        lt = {"新岛": 1.0, "中年": float(rc["lava_tube_mid_mult"]), "老岛": 0.0}[age_zh]
        if lt > 0:
            cand = mk & (slope >= 3) & (slope <= 18) & (peak_rel >= 0.25) & (peak_rel <= 0.8)
            place("cave", cand, patchy, lam("lava_tube", lt), 0, float(rc["cave_sep_km"]),
                  sub_fn=lambda i, j, r: "熔岩管" if age_zh == "新岛" else "熔岩管（塌陷天窗）", rng=rng)
        # 峡谷岩洞：下切的谷壁；裂隙洞：山地，近板块边界多
        place("cave", mk & (cut >= float(rc["gorge_cut_m"])) & (slope >= float(rc["gorge_slope_deg"])), patchy, lam("gorge_cave"), 0,
              float(rc["cave_sep_km"]), sub_fn=lambda i, j, r: "峡谷岩洞", rng=rng)
        place("cave", mk & ((zone == 1) | (zone == 2)) & (slope >= 15), patchy, lam("fracture_cave", 1.0 + 2.0 * kern), 0,
              float(rc["cave_sep_km"]), sub_fn=lambda i, j, r: "裂隙洞", rng=rng)
        if age_zh == "老岛" and old_limestone:
            cand = mk & (slope <= 15) & ((acc >= 0.3) | (zone == 2) | (zone == 3))
            place("cave", cand, patchy, lam("karst"), 0, float(rc["cave_sep_km"]), sub_fn=lambda i, j, r: r.choice(["溶洞", "落水洞", "地下河"]), rng=rng)
        rim_cells = m & cliff & ~water
        if rim_cells.any():
            n_c = rng.poisson(float(dens["cliff_cave"]) * float(rim_cells.sum()) * res_km / 10.0)
            place("cave", rim_cells, patchy * np.clip(isl["cliff_m"] / 150.0, 0.2, 1.0), int(n_c), 0, float(rc["cave_sep_km"]),
                  sub_fn=lambda i, j, r: "崖洞", note="开在岸崖上：从云带上方悬索进出", rng=rng)
            n_u = rng.poisson(float(dens["underside_cave"]) * float(rim_cells.sum()) * res_km / 10.0)
            place("cave", rim_cells, patchy, int(n_u), 0, float(rc["cave_sep_km"]), sub_fn=lambda i, j, r: "底面洞",
                  note="入口在崖下的岛底，只能悬索或飞舟进出", rng=rng)
            # 瀑布后洞：常年河跌下崖缘处，水帘后冲出的洞（河口旁 3 格内的崖缘格）
            if k == 0:
                for rv_ in J.get("hydro", {}).get("rivers", []):
                    if rng.random() >= float(rc["waterfall_cave_p"]):
                        continue
                    mi, mj = rv_["mouth_cell"]
                    r0_, c0_ = max(0, mi - 3), max(0, mj - 3)
                    win = rim_cells[r0_:mi + 4, c0_:mj + 4] & ~taken[r0_:mi + 4, c0_:mj + 4]
                    if win.any():
                        wi, wj = np.where(win)
                        t = int(np.argmin((wi + r0_ - mi) ** 2 + (wj + c0_ - mj) ** 2))
                        place("cave", rim_cells, np.ones((H, W)), 1, 0, 0.0, seeds=[(int(wi[t] + r0_), int(wj[t] + c0_))],
                              sub_fn=lambda i, j, r: "瀑布后洞", note=f"常年河（流域 {rv_['basin_km2']:.0f} km²）跌下崖缘处的水帘后", rng=rng)
            # 鸟粪石：小岛崖顶（海鸟 / 飞兽聚居）
            if A <= float(rc["guano_max_island_km2"]) and rng.random() < float(rc["guano_p"]):
                place("guano", m & binary_dilate(cliff, 2) & ~water, patchy + 0.2, 1, min(0.3 * A, float(rc["guano_km2"])), 1.0, rng=rng)

    counts: dict[str, int] = {}
    area: dict[str, float] = {}
    for d in deposits:
        counts[d["kind_zh"]] = counts.get(d["kind_zh"], 0) + 1
        area[d["kind_zh"]] = round(area.get(d["kind_zh"], 0.0) + d["area_km2"], 3)
    n_land = max(1, int(land.sum()))
    zshare = {ZONE_NAMES[i]: round(float(((zone == i) & land).sum()) / n_land, 4) for i in range(1, len(ZONE_NAMES))}
    R = {"node": node, "zones": {"classes": ZONE_NAMES, "palette": ZONE_PALETTE, "share": zshare,
                                 "rule": f"局地起伏 = {2 * r_cells + 1}×{2 * r_cells + 1} 格方窗内高差；山地 ≥ {rc['mountain_relief_m']} m 或坡 ≥ {rc['mountain_slope_deg']}° 或高于岸缘→峰的 {rc['mountain_peak_frac']}，"
                                         f"丘陵 ≥ {rc['hill_relief_m']} m 或坡 ≥ {rc['hill_slope_deg']}° 或高于 {rc['hill_peak_frac']}；高山 = 山地且（海拔温度 < 高山草甸线或近峰）；河谷 = 漫滩 + 河边缓坡"},
         "resources": {"classes": RES_NAMES, "palette": RES_PALETTE},
         "geology": {"boundary_type": btype, "boundary_kernel": kern, "layered": layered, "old_island_lithology": "石灰岩" if old_limestone else "砂岩",
                     "ore_multiplier": round(geo_ore, 3), "floatstone_expose_rate": round(fs_rate, 3),
                     "floatstone": "岛体本身就是浮石；可开采，采掉的量相对岛体微不足道，不影响浮空"},
         "counts": counts, "area_km2": area, "deposits": deposits,
         "note": "第三层叙事 / 场景素材，不进管线；cell = 群栅格 [行, 列]，km = 相对群心（x 东 y 北）；grade 是矿点在本类候选里的相对品位"}
    nt = (res > 0) & (res != RES_INDEX["timber"])
    R["non_timber_share"] = round(float(nt.sum()) / n_land, 4)
    g.update({"terrain_zone": zone, "resource": res, "resources": R})
    J["resources"] = {"zones_share": zshare, "counts": counts, "old_island_lithology": R["geology"]["old_island_lithology"]}
    log(f"  地形区：" + "，".join(f"{k} {v * 100:.0f}%" for k, v in zshare.items() if v >= 0.005)
        + f"；资源 {len(deposits)} 处（林木外占陆地 {R['non_timber_share'] * 100:.1f}%）：" + "，".join(f"{k} {v}" for k, v in counts.items()))


def write_resources(out: Path, g: dict) -> None:
    from .grid import write_png8
    write_png8(out / "resources.png", g["resource"], RES_PALETTE)
    write_png8(out / "terrain_zone.png", g["terrain_zone"], ZONE_PALETTE)
    (out / "resources.json").write_text(json.dumps(g["resources"], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def write_preview_resources(out: Path, g: dict) -> Path:
    """资源总览：左 = 地形区（晕渲叠色），右 = 资源斑块与点位（主岛放大）。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from .output import _hillshade_rgb
    J = g["json"]
    R = g["resources"]
    res_m = J["raster"]["res_m"]
    H, W = g["height"].shape
    x0, y0 = J["raster"]["origin_km"]
    ext = [x0, x0 + W * res_m / 1000.0, y0 - H * res_m / 1000.0, y0]
    h = g["height"]
    shade = _hillshade_rgb(h, res_m, float(np.nanmin(h)), float(np.nanmax(h)), "gray")
    land = g["island_id"] >= 0
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    zp = np.array(ZONE_PALETTE, dtype=float) / 255.0
    zc = zp[g["terrain_zone"]]
    axes[0].imshow(np.where(land[..., None], 0.5 * shade + 0.5 * zc, shade), extent=ext, origin="upper", interpolation="nearest")
    zs = R["zones"]["share"]
    axes[0].legend(handles=[Patch(color=tuple(zp[i]), label=f"{ZONE_NAMES[i]} {zs.get(ZONE_NAMES[i], 0) * 100:.0f}%") for i in range(1, len(ZONE_NAMES))],
                   loc="upper left", fontsize=8, framealpha=0.75)
    axes[0].set_title("地形区（山区 / 丘陵 / 台地 / 河谷）", fontsize=10)
    # 右：主岛外框
    r0, c0, mm, _ = J["islands"][0]["bbox_cells"]
    ii, jj = np.where(g["island_id"] == 0)
    pad = 5
    a0, a1, b0, b1 = max(0, ii.min() - pad), min(H, ii.max() + pad + 1), max(0, jj.min() - pad), min(W, jj.max() + pad + 1)
    sub = (slice(a0, a1), slice(b0, b1))
    rp = np.array(RES_PALETTE, dtype=float) / 255.0
    rr = g["resource"][sub]
    base = shade[sub].copy()
    patch = (rr > 0) & ~np.isin(rr, [RES_INDEX["spring"], RES_INDEX["cave"], RES_INDEX["hotspring"], RES_INDEX["timber"]])
    tim = rr == RES_INDEX["timber"]
    base[tim] = 0.8 * base[tim] + 0.2 * rp[RES_INDEX["timber"]]          # 林木只淡淡一层，不盖住矿点
    base[patch] = 0.25 * base[patch] + 0.75 * rp[rr[patch]]
    base[g["river"][sub] > 0] = (0.15, 0.35, 0.85)
    base[g["lake"][sub]] = (0.12, 0.25, 0.7)
    sext = [x0 + b0 * res_m / 1000.0, x0 + b1 * res_m / 1000.0, y0 - a1 * res_m / 1000.0, y0 - a0 * res_m / 1000.0]
    axes[1].imshow(base, extent=sext, origin="upper", interpolation="nearest")
    marks = {"spring": ("o", 14), "cave": ("^", 34), "hotspring": ("*", 60), "ore": ("s", 26), "quarry": ("D", 20), "placer": ("P", 30),
             "sulfur": ("X", 30), "floatstone": ("h", 40)}
    pits = [_km(J, a, b) for d in R["deposits"] if d["kind"] == "ore" and d["island"] == 0 for a, b in d.get("pits", [])]
    if pits:
        p_ = np.array(pits)
        axes[1].scatter(p_[:, 0], p_[:, 1], marker="s", s=22, c=[tuple(rp[RES_INDEX["ore"]])], edgecolors="black", linewidths=0.5, zorder=6)
    for key, (mk_, sz) in marks.items():
        if key in ("ore", "quarry", "floatstone"):
            continue
        pts = [d["km"] for d in R["deposits"] if d["kind"] == key and d["island"] == 0]
        if pts:
            p = np.array(pts)
            axes[1].scatter(p[:, 0], p[:, 1], marker=mk_, s=sz, c=[tuple(rp[RES_INDEX[key]])], edgecolors="black", linewidths=0.5, zorder=5)
    handles = [Patch(color=tuple(rp[RES_INDEX[k[0]]]), label=f"{k[1]} {R['counts'].get(k[1], 0)}") for k in RES_KINDS if k[3] == "patch" and R["counts"].get(k[1])]
    handles += [Line2D([], [], marker=marks[k][0], ls="", color=tuple(rp[RES_INDEX[k]]), markeredgecolor="black", label=f"{RES_NAMES[RES_INDEX[k]]} {R['counts'].get(RES_NAMES[RES_INDEX[k]], 0)}")
                for k in ("spring", "cave", "hotspring") if R["counts"].get(RES_NAMES[RES_INDEX[k]])]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.75)
    geo = R["geology"]
    axes[1].set_title(f"主岛资源（图例为全群计数）· 最近板块边界：{geo['boundary_type']}（核 {geo['boundary_kernel']:.2f}）{' · 叠层' if geo['layered'] else ''} · 老岛岩性 {geo['old_island_lithology']}", fontsize=10)
    for ax in axes:
        ax.set_xlabel("km 东")
        ax.set_aspect("equal")
    axes[0].set_ylabel("km 北")
    m = J["meta"]
    fig.suptitle(f"岛群 #{m['node']} 地形区与资源 [seed {m['seed']} · {m['res_m']:.0f} m/格]", fontsize=11)
    p = out / "preview_resources.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return p
