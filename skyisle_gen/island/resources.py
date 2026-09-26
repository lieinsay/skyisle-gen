"""5.3c 地形区与资源分布：山区 / 丘陵 / 台地 / 河谷的区划，和三种形态的资源（DESIGN-NOTES 四点十六 / 四点十七 / 四点二十一）。

全部从已生成的地形、水系、地表与节点的地质背景（岛龄、板块边界类型与远近、叠层）推出，不读人口、不回灌（第三层）。
随机数只走 _rng(node, "resources:…")。资源按形态分三种说法：
- 点（泉眼、温泉、洞穴）：位置就是资源，只占一格；记在 deposits。
- 片（林木、泥炭 / 芦苇、浮石露头、鸟粪石）：边界清楚的一片地就是资源；记在 deposits，占的格写进 patch_id（片与片、片与点互斥）。
- 散（金属矿、石料、黏土、砂砾、砂金、硫磺）：分三层——
  赋存场（res_field：每类每格 0–1 品位，各类可叠在同一格）→ 赋存区（occurrences：品位 ≥ occ_thr 的连通块，可命名、可叙述）→
  采场（workings：人在哪挖）。稀缺的人就矿：矿坑、硫磺坑在这里按品位挑；常用的就近取：采石场、土坑、采砂场、淘金点在聚落之后按村挑（tiers.py）。
  岩类（金属矿、石料、硫磺）只在「岩类可放区」rock_site：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ rock_hill_slope_deg；
  林坡算（采场那格改裸岩），平地的林、耕地、湿地、漫滩不算。沉积类（黏土、砂砾、砂金）的赋存可以压在田下，坑不上田不上林（2026-09-26 拍板）。

产物：terrain_zone（uint8，见 ZONE_NAMES）、res_field（uint8 [6, H, W]，品位 × 255，层序 FIELD_KINDS）、patch_id（int32，−1 = 无）、
resource（uint8，见 RES_NAMES：显示用的「主导」类，按 DOMINANT_ORDER 后画盖先画）进 terrain.npz；
resources.json 列出点与片（deposits）、赋存区（occurrences）、采场（workings）；resources.png 主导类索引色；preview_resources.png 总览。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grid import N8, FractalNoise, binary_dilate, label_by_island, label_components, shift, window_extrema
from .hydro import LC_ALPINE, LC_FOREST, LC_ROCK, LC_WET

ZONE_NAMES = ["虚空", "高山", "山地", "丘陵", "台地平原", "河谷", "崖缘", "水域"]
ZONE_PALETTE = [(20, 24, 40), (235, 235, 240), (150, 110, 80), (200, 170, 110), (170, 200, 120), (110, 190, 170), (90, 80, 75), (40, 90, 200)]

# 资源类：键 → (中文, 调色, 形态)。形态：point 点 / patch 片 / field 散（赋存场）。编码 = 本表序号 + 1
RES_KINDS = [
    ("timber", "林木", (60, 130, 60), "patch"),
    ("spring", "泉眼", (120, 220, 255), "point"),
    ("clay", "黏土", (190, 120, 80), "field"),
    ("peat", "泥炭 / 芦苇", (110, 90, 60), "patch"),
    ("gravel", "砂砾", (205, 195, 170), "field"),
    ("placer", "砂金", (255, 215, 80), "field"),
    ("stone", "石料", (225, 222, 210), "field"),         # 浅石色：旧的 (170, 170, 185) 和晕渲的灰分不开，看上去全岛都是石料
    ("floatstone", "浮石", (150, 100, 230), "patch"),
    ("ore", "金属矿", (200, 60, 50), "field"),
    ("hotspring", "温泉", (255, 140, 180), "point"),
    ("sulfur", "硫磺", (230, 230, 60), "field"),
    ("cave", "洞穴", (40, 40, 40), "point"),
    ("guano", "鸟粪石", (240, 240, 210), "patch"),
]
RES_NAMES = ["无"] + [k[1] for k in RES_KINDS]
RES_PALETTE = [(0, 0, 0)] + [k[2] for k in RES_KINDS]
RES_INDEX = {k[0]: i + 1 for i, k in enumerate(RES_KINDS)}
RES_FORM = {k[0]: k[3] for k in RES_KINDS}
FIELD_KINDS = ["ore", "sulfur", "placer", "clay", "gravel", "stone"]       # res_field 的层序
ROCK_KINDS = ("ore", "sulfur", "stone")                                      # 岩类：只在 rock_site 里
WORK_ZH = {"ore": "矿坑", "sulfur": "硫磺坑", "placer": "淘金点", "stone": "采石场", "clay": "土坑", "gravel": "采砂场"}
# 主导栅格（显示用）的画法顺序：后画的盖先画的——稀的盖常的，点最后
DOMINANT_ORDER = ["timber", "stone", "gravel", "clay", "peat", "guano", "floatstone", "placer", "sulfur", "ore", "spring", "hotspring", "cave"]

# 金属矿的矿种权重（按最近的板块边界类型）；老岛加锡钨，新岛加硫化铜
ORE_WEIGHTS = {"汇聚": {"铜": 0.35, "铁": 0.25, "铅锌": 0.25, "金": 0.15},
               "离散": {"铁": 0.5, "铜": 0.3, "锰": 0.2},
               "走滑": {"铁": 0.4, "铜": 0.25, "锡": 0.2, "铅锌": 0.15}}
PLACER_METALS = ("金", "铜")        # 这两种矿化带会往下游冲出砂金（斑岩铜常伴金）


def _km(J, i, j):
    x0, y0 = J["raster"]["origin_km"]
    r = J["raster"]["res_m"] / 1000.0
    return [round(x0 + (j + 0.5) * r, 3), round(y0 - (i + 0.5) * r, 3)]


def _pick(rng, weights: dict) -> str:
    ks = sorted(weights)
    p = np.array([weights[k] for k in ks], dtype=float)
    return ks[int(rng.choice(len(ks), p=p / p.sum()))]


def _grade_zh(q: float) -> str:
    return "上" if q >= 0.75 else ("中" if q >= 0.45 else "下")


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


def _grow(seed, cand: np.ndarray, score: np.ndarray, n_cells: int, taken: np.ndarray) -> np.ndarray:
    """从种子向外长斑块：窗内候选格按（距离 − 分数加权）排序取前 n_cells 个。返回 (行, 列) 下标。"""
    H, W = cand.shape
    R = int(math.ceil(math.sqrt(max(1, n_cells) / math.pi) * 1.8)) + 1
    i, j = seed
    r0, r1, c0, c1 = max(0, i - R), min(H, i + R + 1), max(0, j - R), min(W, j + R + 1)
    ii, jj = np.mgrid[r0:r1, c0:c1]
    ok = cand[r0:r1, c0:c1] & ~taken[r0:r1, c0:c1]
    ok[i - r0, j - c0] = True
    d = np.hypot(ii - i, jj - j) / max(1.0, R)
    key = np.where(ok, d - 0.35 * score[r0:r1, c0:c1], np.inf).ravel()
    sel = np.argsort(key, kind="stable")[:max(1, n_cells)]
    sel = sel[np.isfinite(key[sel])]
    return ii.ravel()[sel], jj.ravel()[sel]


def _group_cells(lab: np.ndarray, valid: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """标号栅格 → {标号: (行, 列)}，只取 valid 为真的格；一次排序分组，不逐标号扫全图。"""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return {}
    v = lab.ravel()[idx]
    o = np.argsort(v, kind="stable")
    idx, v = idx[o], v[o]
    cut = np.flatnonzero(np.diff(v)) + 1
    W = lab.shape[1]
    out = {}
    for a, b in zip(np.r_[0, cut].tolist(), np.r_[cut, idx.size].tolist()):
        seg = idx[a:b]
        out[int(v[a])] = (seg // W, seg % W)
    return out


def _spread_max(a: np.ndarray, n: int) -> np.ndarray:
    """最大值向八邻域扩 n 圈（河道格的量带给两岸滩地）。"""
    for _ in range(n):
        b = a.copy()
        for di, dj in N8:
            np.maximum(b, shift(a, di, dj, 0.0), out=b)
        a = b
    return a


def _shape(ii: np.ndarray, jj: np.ndarray, res_km: float) -> tuple[float, float | None]:
    """赋存区的长度（km，按主轴方差：均匀椭圆的全长 = 4σ）与走向（自东逆时针 0–180°）。"""
    if ii.size < 3:
        return round(max(1.0, math.sqrt(ii.size)) * res_km, 2), None
    C = np.cov(np.vstack([jj * res_km, -ii * res_km]))
    w, V = np.linalg.eigh(C)
    length = max(4.0 * math.sqrt(max(float(w[-1]), 0.0)), res_km)
    return round(length, 2), round(math.degrees(math.atan2(V[1, -1], V[0, -1])) % 180.0, 0)


def rock_site_mask(g: dict, zone: np.ndarray, rock_hill_slope_deg: float) -> np.ndarray:
    """岩类（金属矿 / 石料 / 硫磺）可放区：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ rock_hill_slope_deg；
    不含耕地、湿地、漫滩、水、崖缘。check 的 RES-occ 也按它验。"""
    land = g["island_id"] >= 0
    cover = g["landcover"]
    slope = g["slope_deg"]
    water = (g["river"] > 0) | g["lake"]
    flood = g.get("floodplain", np.zeros(land.shape, dtype=bool))
    terrain = np.isin(zone, [1, 2]) | ((zone == 3) & (slope >= rock_hill_slope_deg)) | np.isin(cover, [LC_ROCK, LC_ALPINE])
    return land & ~water & ~g["cliff"] & (g["arable"] == 0) & (cover != LC_WET) & ~flood & terrain


def open_working(g: dict, i: int, j: int) -> None:
    """岩类采场开在林坡 / 灌丛上：那一格改成裸岩，从林场里划掉（面积由 sync_resources 重数）。"""
    g["landcover"][i, j] = LC_ROCK
    pid = g["patch_id"]
    k = int(pid[i, j])
    if k >= 0 and g["resources"]["deposits"][k]["kind"] == "timber":
        pid[i, j] = -1


def build_resources(ctx, node: int, c: dict, g: dict, log=print) -> None:
    from . import _rng
    rc = c["resources"]
    J = g["json"]
    inp = g["inp"]
    res_km = g["res_km"]
    cell_km2 = res_km * res_km
    island_id = g["island_id"]
    land = island_id >= 0
    H, W = land.shape
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
    g["terrain_zone"] = zone

    # ---------- 资源的共用量 ----------
    patch_id = np.full((H, W), -1, dtype=np.int32)
    taken = np.zeros((H, W), dtype=bool)          # 点与片互斥（散不占格）
    deposits: list[dict] = []
    occurrences: list[dict] = []
    workings: list[dict] = []
    F = {k: np.zeros((H, W)) for k in FIELD_KINDS}
    occ_lab = {k: np.full((H, W), -1, dtype=np.int32) for k in FIELD_KINDS}
    thr = {k: float(rc["occ_thr"][k]) for k in FIELD_KINDS}
    min_cells = int(rc["occ_min_cells"])
    dens = rc["density_per_100km2"]
    noise = FractalNoise(_rng(ctx, node, "resources:noise"), Xk[0], Yk[-1], Xk[-1], Yk[0],
                         feature_km=float(rc["patch_km"]), octaves=3, persistence=0.5).sample(XX, YY)
    patchy = np.clip(0.5 + 0.5 * noise, 0.0, 1.0)
    soft = land & ~water & ~cliff
    arable = g["arable"] > 0
    rock_site = rock_site_mask(g, zone, float(rc["rock_hill_slope_deg"]))
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
    ages = [isl["age_zh"] for isl in J["islands"]]

    def lith(age_zh: str) -> str:
        return {"新岛": "玄武岩", "中年": "安山岩 / 凝灰岩", "老岛": "石灰岩" if old_limestone else "砂岩"}[age_zh]

    def add_deposit(kind, ii, jj, i, j, sub, q, note=None, extra=None):
        area = ii.size * cell_km2
        d = {"id": len(deposits), "kind": kind, "kind_zh": RES_NAMES[RES_INDEX[kind]], "form": RES_FORM[kind], "subtype": sub,
             "island": int(island_id[i, j]), "cell": [int(i), int(j)], "km": _km(J, i, j), "area_km2": round(area, 3),
             "elev_m": round(float(hz[i, j]), 0), "slope_deg": round(float(slope[i, j]), 1),
             "zone": ZONE_NAMES[int(zone[i, j])], "grade": _grade_zh(q)}
        if note:
            d["note"] = note
        if extra:
            d.update(extra)
        deposits.append(d)
        if RES_FORM[kind] == "patch":
            patch_id[ii, jj] = d["id"]
        return d

    def place(kind: str, cand: np.ndarray, score: np.ndarray, n: int, area_med_km2: float, sep_km: float,
              sub_fn=None, note: str | None = None, rng=None, seeds=None):
        """点 / 片：贪心取种子（同类最小间距），点只占一格，片按「距离 − 0.35 × 适宜度」长成斑块；与已有的点 / 片互斥。"""
        if n <= 0 or not cand.any():
            return 0
        sc = np.where(cand, score, 0.0)
        if seeds is None:
            seeds = _seeds(np.where(taken, 0.0, sc), n, sep_km / res_km)
        placed = 0
        for (i, j) in seeds:
            if RES_FORM[kind] == "point":
                ii, jj = np.array([i]), np.array([j])
            else:
                area = float(area_med_km2 * rng.lognormal(0.0, 0.6))
                ii, jj = _grow((i, j), cand, sc, max(1, int(round(area / cell_km2))), taken)
            taken[ii, jj] = True
            add_deposit(kind, ii, jj, i, j, sub_fn(i, j, rng) if sub_fn else None, float(sc[i, j]), note)
            placed += 1
        return placed

    def add_occ(kind, ii, jj, gv, sub, note=None, extra=None) -> dict:
        """赋存区记录：峰值格、面积、长度与走向、峰值 / 均值品位、高程范围。区号写进 occ_lab（同类重叠时品位高的占格）。"""
        pk = int(np.argmax(gv))
        i, j = int(ii[pk]), int(jj[pk])
        length, ax = _shape(ii, jj, res_km)
        o = {"id": len(occurrences), "kind": kind, "kind_zh": RES_NAMES[RES_INDEX[kind]], "subtype": sub,
             "island": int(island_id[i, j]), "cell": [i, j], "km": _km(J, i, j), "area_km2": round(ii.size * cell_km2, 3),
             "length_km": length, "axis_deg": ax, "grade_peak": round(float(gv.max()), 3), "grade_mean": round(float(gv.mean()), 3),
             "grade": _grade_zh(float(gv.max())), "elev_m": [round(float(hz[ii, jj].min()), 0), round(float(hz[ii, jj].max()), 0)],
             "zone": ZONE_NAMES[int(zone[i, j])]}
        if note:
            o["note"] = note
        if extra:
            o.update(extra)
        occurrences.append(o)
        lab = occ_lab[kind]
        own = gv >= F[kind][ii, jj] - 1e-12
        lab[ii[own], jj[own]] = o["id"]
        return o

    def add_work(kind, o, i, j, note=None) -> dict:
        w = {"id": len(workings), "kind": kind, "kind_zh": WORK_ZH[kind], "occurrence": o["id"], "island": int(island_id[i, j]),
             "cell": [int(i), int(j)], "km": _km(J, i, j), "grade": round(float(F[kind][i, j]), 3), "villages": [], "special": None}
        if note:
            w["note"] = note
        workings.append(w)
        return w

    def field_occurrences(kind, sub_fn, note=None):
        """散的赋存区（石料 / 黏土 / 砂砾 / 砂金）：品位 ≥ 阈值的格，隔 occ_merge_cells 格以内的碎块算一处（坡度、噪声把一片切成的碎块），
        不跨岛；小于 occ_min_km2（且不少于 occ_min_cells 格）的不成区。"""
        m = F[kind] >= thr[kind]
        if not m.any():
            return
        lab, _ = label_by_island(binary_dilate(m, int(rc["occ_merge_cells"])) & land, island_id, 8)
        n_min = max(min_cells, int(math.ceil(float(rc["occ_min_km2"]) / cell_km2 - 1e-9)))
        for _, (ii, jj) in sorted(_group_cells(lab, m & (lab > 0)).items()):
            if ii.size < n_min:
                continue
            add_occ(kind, ii, jj, F[kind][ii, jj], sub_fn(ii, jj), note)

    # ---------- 散：石料 / 黏土 / 砂砾（全群一次算，赋存与植被无关） ----------
    # 石料：按露头算——岩类可放区只说「可以在哪」，长着林子的缓坡底下虽是基岩，却不是能开的石头（第一版把 8° 林坡都算进来，#1165 石料区占陆地 31%）。
    # 品位 = 坡从 stone_slope_lo_deg 起算、到 stone_slope_hi_deg 满；裸岩 / 高山草甸 / 峡谷壁至少 stone_exposed_grade；林坡再乘 stone_forest_mult
    exposed = np.isin(cover, [LC_ROCK, LC_ALPINE]) | ((cut >= float(rc["gorge_cut_m"])) & (slope >= float(rc["gorge_slope_deg"])))
    s_lo, s_hi = float(rc["stone_slope_lo_deg"]), float(rc["stone_slope_hi_deg"])
    g_st = np.maximum(np.clip((slope - s_lo) / max(1e-6, s_hi - s_lo), 0.0, 1.0), np.where(exposed, float(rc["stone_exposed_grade"]), 0.0))
    F["stone"] = np.where(rock_site & (slope <= float(rc["stone_slope_max_deg"])),
                          g_st * (0.75 + 0.25 * patchy) * np.where(cover == LC_FOREST, float(rc["stone_forest_mult"]), 1.0), 0.0)
    # 黏土：漫滩 / 湖滨 / 湿地边（河湖黏土）高、溪边缓坡（溪边黏土）约一半；可压在田下
    lake_edge = binary_dilate(lake, 2) & ~lake
    wet = cover == LC_WET
    strong = flood | lake_edge | binary_dilate(wet, 1)
    near_stream = binary_dilate(stream | river, int(rc["clay_stream_cells"]))
    clay_ok = soft & ~wet & (slope < float(rc["clay_slope_max_deg"]))
    F["clay"] = np.where(clay_ok & strong, 0.7 + 0.3 * patchy, np.where(clay_ok & near_stream, 0.35 + 0.25 * patchy, 0.0))
    # 砂砾：常年河 / 大溪边的缓坡滩地，品位随河的大小（汇流 10 km² → 0.6，1000 km² → 1）
    chan = river | (stream & (acc >= float(rc["gravel_acc_km2"])))
    bank = int(rc["gravel_bank_cells"])
    bars = binary_dilate(chan, bank) & ~chan & soft & ~wet & (slope < float(rc["gravel_slope_max_deg"]))
    acc_bar = _spread_max(np.where(chan, acc, 0.0), bank)
    F["gravel"] = np.where(bars, (0.4 + 0.6 * np.clip(np.log10(np.maximum(acc_bar, 1.0)) / 3.0, 0.0, 1.0)) * (0.6 + 0.4 * patchy), 0.0)
    gold_src = np.zeros((H, W))                   # 金 / 铜矿化带的品位（砂金的上游来源）

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
        # 林木（片）：大片林地（连通块 ≥ timber_min_km2），针叶 / 阔叶按年均温
        forest = m & (cover == LC_FOREST)
        if forest.any():
            lab, nl = label_components(forest, connectivity=8)
            if nl:
                cnt = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
                keep = [int(x) + 1 for x in np.argsort(-cnt, kind="stable")[:int(rc["timber_max_per_island"])] if cnt[x] * cell_km2 >= float(rc["timber_min_km2"])]
                cells = _group_cells(lab, np.isin(lab, keep)) if keep else {}
                for lb in keep:
                    ii, jj = cells[lb]
                    t_mean = float(T[ii, jj].mean())
                    ci = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
                    d = add_deposit("timber", ii, jj, int(ii[ci]), int(jj[ci]),
                                    "针叶林" if t_mean < float(rc["conifer_temp_c"]) else ("针阔混交林" if t_mean < float(rc["conifer_temp_c"]) + 5 else "阔叶林"),
                                    1.0 if ii.size * cell_km2 >= 4 * float(rc["timber_min_km2"]) else 0.5)
                    d["elev_m"], d["slope_deg"] = round(float(hz[ii, jj].mean()), 0), round(float(slope[ii, jj].mean()), 1)
        # 泉眼（点）：溪涧源头（本格有溪、上游八邻无溪），坡度转缓处优先
        if stream[m].any():
            st = m & stream
            nb = np.zeros((H, W), dtype=int)
            for di, dj in N8:
                nb += shift(st, di, dj, False)
            heads = st & (nb <= 1) & soft
            place("spring", heads, patchy * np.clip(1.2 - slope / 25.0, 0.05, 1.0), lam("spring"), 0, float(rc["spring_sep_km"]), rng=rng)
        # 泥炭（凉湿）/ 芦苇荡（暖）（片）：湿地
        place("peat", m & wet, 0.3 + 0.7 * patchy, lam("peat"), float(rc["peat_km2"]), float(rc["patch_sep_km"]),
              sub_fn=lambda i, j, r: "泥炭" if T[i, j] < float(rc["peat_temp_max_c"]) else "芦苇荡", rng=rng)
        # 金属矿（散）：顺板块走向拉长的矿化带 = 椭圆核 × 斑块噪声，裁到岩类可放区；带内按品位点矿坑
        cand = m & rock_site & (slope <= float(rc["ore_slope_max_deg"]))
        w_ore = dict(ORE_WEIGHTS.get(btype, ORE_WEIGHTS["走滑"]))
        if age_zh == "老岛":
            w_ore["锡钨"] = 0.2
        if age_zh == "新岛":
            w_ore["铜"] = w_ore.get("铜", 0) + 0.15
        age_mult = {"新岛": 0.6, "中年": 1.0, "老岛": 1.2}[age_zh]
        if big and cand.any():
            score = np.where(cand, patchy * np.clip(relief / (2.0 * float(rc["mountain_relief_m"])), 0.1, 1.0), 0.0)
            elong = float(rc["ore_elongation"])
            for (i, j) in _seeds(score, lam("ore", geo_ore * age_mult), float(rc["ore_sep_km"]) / res_km):
                area = float(rc["ore_km2"]) * float(rng.lognormal(0.0, 0.6))
                b_km = math.sqrt(area / (math.pi * elong))
                a_km = elong * b_km
                R = int(math.ceil(a_km / res_km)) + 1
                r0, r1, c0, c1 = max(0, i - R), min(H, i + R + 1), max(0, j - R), min(W, j + R + 1)
                wi, wj = np.mgrid[r0:r1, c0:c1]
                dx, dy = (wj - j) * res_km, -(wi - i) * res_km
                along = dx * math.cos(axis) + dy * math.sin(axis)
                across = -dx * math.sin(axis) + dy * math.cos(axis)
                gw = np.clip(1.0 - (along / a_km) ** 2 - (across / b_km) ** 2, 0.0, 1.0) * (0.65 + 0.35 * patchy[r0:r1, c0:c1])
                gw = np.where(cand[r0:r1, c0:c1], gw, 0.0)
                belt = gw >= thr["ore"]
                if belt.sum() < min_cells:
                    continue
                sub = _pick(rng, w_ore)
                bi, bj = wi[belt], wj[belt]
                o = add_occ("ore", bi, bj, gw[belt], sub, note="矿化带（顺板块走向）；只画在岩类可放区里，林下 / 田下的不算（前工业时代找不到、也开不了）")
                F["ore"][r0:r1, c0:c1] = np.maximum(F["ore"][r0:r1, c0:c1], gw)
                if sub in PLACER_METALS:
                    gold_src[r0:r1, c0:c1] = np.maximum(gold_src[r0:r1, c0:c1], gw)
                # 矿坑：带里品位最高的几格，彼此 ≥ 3 格
                n_p = int(min(int(rc["ore_pits_max"]), 1 + rng.poisson(o["area_km2"])))
                gv = gw[belt]
                chosen = []
                for t in np.argsort(-gv, kind="stable").tolist():
                    if all((bi[t] - a) ** 2 + (bj[t] - b) ** 2 >= 9 for a, b in chosen):
                        chosen.append((int(bi[t]), int(bj[t])))
                    if len(chosen) >= n_p:
                        break
                for (a, b) in chosen:
                    add_work("ore", o, a, b)
        # 浮石露头（片）：崖面 + 深切峡谷壁；露头率 × 岛龄（新岛 1.2 / 中年 1 / 老岛 0.6），按斑块噪声取格，连通段各算一处
        f_exp = float(np.clip(fs_rate * {"新岛": 1.2, "中年": 1.0, "老岛": 0.6}[age_zh], 0.02, 0.95))
        fs_cand_c = m & cliff & ~water
        fs_cand_g = mk & (cut >= float(rc["gorge_cut_m"])) & (slope >= float(rc["gorge_slope_deg"]))
        for fs_cand, sub_fs in ((fs_cand_c, "崖面露头"), (fs_cand_g, "峡谷露头")):
            if not fs_cand.any():
                continue
            q_ = float(np.quantile(noise_fs[fs_cand], 1.0 - f_exp))
            ex = fs_cand & (noise_fs >= q_) & ~taken
            lab, nl = label_components(ex, connectivity=8)
            if not nl:
                continue
            for lb, (ii, jj) in sorted(_group_cells(lab, lab > 0).items()):
                if ii.size < int(rc["floatstone_min_cells"]):
                    continue
                ci = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
                taken[ii, jj] = True
                add_deposit("floatstone", ii, jj, int(ii[ci]), int(jj[ci]), sub_fs, float(np.clip(0.5 + 0.5 * noise_fs[ii, jj].mean(), 0, 1)),
                            note="岛体本身的浮石：可开采，采掉的量相对岛体微不足道，不影响浮空")
        # 温泉（点）/ 硫磺（散）：新岛（火山余热）；中年岛偶有温泉
        if age_zh != "老岛":
            hot = {"新岛": 1.0, "中年": float(rc["mid_hot_mult"])}[age_zh]
            cand = mk & (peak_rel >= 0.15) & (slope < 25)
            place("hotspring", cand & (stream | binary_dilate(stream, 1)), patchy + 0.3 * peak_rel, lam("hotspring", hot), 0, float(rc["spring_sep_km"]), rng=rng)
            if age_zh == "新岛" and big:
                cand = m & rock_site & (peak_rel >= float(rc["sulfur_peak_frac"]))
                for (i, j) in _seeds(np.where(cand, patchy * peak_rel, 0.0), lam("sulfur"), float(rc["patch_sep_km"]) / res_km):
                    r_km = math.sqrt(float(rc["sulfur_km2"]) * float(rng.lognormal(0.0, 0.6)) / math.pi)
                    R = int(math.ceil(r_km / res_km)) + 1
                    r0, r1, c0, c1 = max(0, i - R), min(H, i + R + 1), max(0, j - R), min(W, j + R + 1)
                    wi, wj = np.mgrid[r0:r1, c0:c1]
                    gw = np.clip(1.0 - (np.hypot(wi - i, wj - j) * res_km / r_km) ** 2, 0.0, 1.0) * (0.7 + 0.3 * patchy[r0:r1, c0:c1])
                    gw = np.where(cand[r0:r1, c0:c1], gw, 0.0)
                    zm = gw >= thr["sulfur"]
                    if zm.sum() < min_cells:
                        continue
                    o = add_occ("sulfur", wi[zm], wj[zm], gw[zm], "火山口 / 喷气孔")
                    F["sulfur"][r0:r1, c0:c1] = np.maximum(F["sulfur"][r0:r1, c0:c1], gw)
                    add_work("sulfur", o, *o["cell"])
        # 洞穴（点）：熔岩管（新岛缓坡）/ 溶洞（石灰岩老岛的台地与落水洞）/ 崖洞（岸崖，所有岛）
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
            # 鸟粪石（片）：小岛崖顶（海鸟 / 飞兽聚居）
            if A <= float(rc["guano_max_island_km2"]) and rng.random() < float(rc["guano_p"]):
                place("guano", m & binary_dilate(cliff, 2) & ~water, patchy + 0.2, 1, min(0.3 * A, float(rc["guano_km2"])), 1.0, rng=rng)

    # ---------- 散：砂金（上游金 / 铜矿化带的平均品位，带给河边滩地）、石料 / 黏土 / 砂砾的赋存区 ----------
    if gold_src.any():
        ok = land & np.isfinite(g["route_h"])
        from .terrain import accumulate
        up = accumulate(g["route_h"], ok, g["recv_i"], g["recv_j"], weight=gold_src) * cell_km2
        mean_up = np.where(chan, up / np.maximum(acc, cell_km2), 0.0)
        F["placer"] = F["gravel"] * np.clip(float(rc["placer_gain"]) * _spread_max(mean_up, bank), 0.0, 1.0)
    field_occurrences("stone", lambda ii, jj: lith(ages[int(island_id[ii[0], jj[0]])]))
    field_occurrences("clay", lambda ii, jj: "河湖黏土" if strong[ii, jj].mean() >= 0.5 else "溪边黏土")
    field_occurrences("gravel", lambda ii, jj: "河滩砂砾")
    field_occurrences("placer", lambda ii, jj: "砂金", note="上游有金 / 铜矿化带：河砂可淘金")
    g.update({"patch_id": patch_id, "occ_lab": occ_lab,
              "res_field": np.stack([np.round(np.clip(F[k], 0.0, 1.0) * 255.0).astype(np.uint8) for k in FIELD_KINDS])})
    R = {"node": node, "zones": {"classes": ZONE_NAMES, "palette": ZONE_PALETTE,
                                 "share": {ZONE_NAMES[i]: round(float(((zone == i) & land).sum()) / max(1, int(land.sum())), 4) for i in range(1, len(ZONE_NAMES))},
                                 "rule": f"局地起伏 = {2 * r_cells + 1}×{2 * r_cells + 1} 格方窗内高差；山地 ≥ {rc['mountain_relief_m']} m 或坡 ≥ {rc['mountain_slope_deg']}° 或高于岸缘→峰的 {rc['mountain_peak_frac']}，"
                                         f"丘陵 ≥ {rc['hill_relief_m']} m 或坡 ≥ {rc['hill_slope_deg']}° 或高于 {rc['hill_peak_frac']}；高山 = 山地且（海拔温度 < 高山草甸线或近峰）；河谷 = 漫滩 + 河边缓坡"},
         "resources": {"classes": RES_NAMES, "palette": RES_PALETTE, "forms": {k[1]: k[3] for k in RES_KINDS},
                       "dominant_order": [RES_NAMES[RES_INDEX[k]] for k in DOMINANT_ORDER]},
         "fields": {"kinds": FIELD_KINDS, "names": [RES_NAMES[RES_INDEX[k]] for k in FIELD_KINDS], "thr": thr,
                    "rock_kinds": list(ROCK_KINDS), "rock_hill_slope_deg": float(rc["rock_hill_slope_deg"]),
                    "rule": f"res_field = 品位 × 255。岩类（金属矿 / 石料 / 硫磺）只在岩类可放区：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ {rc['rock_hill_slope_deg']}°"
                            "（林坡算，平地林、耕地、湿地、漫滩不算）；沉积类（黏土 / 砂砾 / 砂金）可压在田下。赋存区 = 品位 ≥ thr 的连通块（矿化带 / 硫磺按各自的核）"},
         "geology": {"boundary_type": btype, "boundary_kernel": kern, "layered": layered, "old_island_lithology": "石灰岩" if old_limestone else "砂岩",
                     "ore_multiplier": round(geo_ore, 3), "floatstone_expose_rate": round(fs_rate, 3),
                     "floatstone": "岛体本身就是浮石；可开采，采掉的量相对岛体微不足道，不影响浮空"},
         "deposits": deposits, "occurrences": occurrences, "workings": workings,
         "note": "第三层叙事 / 场景素材，不进管线；cell = 群栅格 [行, 列]，km = 相对群心（x 东 y 北）。deposits = 点与片，occurrences = 散的赋存区（cell = 品位峰值格，"
                 "axis_deg = 走向，自东逆时针），workings = 采场（villages / special = 用它的村 / 专业聚落）；grade 是本类里的相对品位"}
    g["resources"] = R
    for w in workings:                            # 岩类采场开在林坡 / 灌丛上：那格改裸岩
        if w["kind"] in ROCK_KINDS:
            open_working(g, *w["cell"])
    sync_resources(g)
    log(f"  地形区：" + "，".join(f"{k} {v * 100:.0f}%" for k, v in R["zones"]["share"].items() if v >= 0.005)
        + f"；点与片 {len(deposits)} 处，赋存区 {len(occurrences)}，采场 {len(workings)}（林木外占陆地 {R['non_timber_share'] * 100:.1f}%）：" + "，".join(f"{k} {v}" for k, v in R["counts"].items()))


def sync_resources(g: dict) -> None:
    """片的面积（按 patch_id 重数：开垦、开采都会划掉格）、代表格、主导栅格、计数与地表占比。资源层末尾与聚落开垦 / 开采之后各调一次。"""
    from .output import LANDCOVER_CLASSES
    R = g["resources"]
    J = g["json"]
    pid = g["patch_id"]
    cell_km2 = g["res_km"] ** 2
    land = g["island_id"] >= 0
    deps = R["deposits"]
    cnt = np.bincount(pid[pid >= 0].ravel(), minlength=len(deps)) if deps else np.zeros(0, dtype=int)
    for d in deps:
        if d["form"] != "patch":
            continue
        n = int(cnt[d["id"]])
        d["area_km2"] = round(n * cell_km2, 3)
        if n == 0:
            d["cleared"] = True                  # 整片开垦 / 开采掉了：留在表里（编号不变——烧炭营、采石村按编号引用），不再计数
            if d["kind"] == "timber":
                d["note"] = "已开垦殆尽（村周草坡 / 薪炭林）"
            continue
        i, j = d["cell"]
        if pid[i, j] != d["id"]:
            ii, jj = np.nonzero(pid == d["id"])
            t = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
            d["cell"] = [int(ii[t]), int(jj[t])]
            d["km"] = _km(J, int(ii[t]), int(jj[t]))
    # 主导栅格（显示用）：按 DOMINANT_ORDER 后画盖先画
    code_of = np.zeros(len(deps) + 1, dtype=np.uint8)
    for d in deps:
        if d["form"] == "patch":
            code_of[d["id"]] = RES_INDEX[d["kind"]]
    patch_code = np.where(pid >= 0, code_of[np.maximum(pid, 0)], 0)
    RF = g["res_field"]
    thr = R["fields"]["thr"]
    res = np.zeros(land.shape, dtype=np.uint8)
    for key in DOMINANT_ORDER:
        code = RES_INDEX[key]
        if RES_FORM[key] == "field":
            res[RF[FIELD_KINDS.index(key)] >= int(round(thr[key] * 255.0))] = code
        elif RES_FORM[key] == "patch":
            res[patch_code == code] = code
        else:
            for d in deps:
                if d["kind"] == key:
                    res[d["cell"][0], d["cell"][1]] = code
    g["resource"] = res
    counts: dict[str, int] = {}
    area: dict[str, float] = {}
    for d in deps + R["occurrences"]:
        if d.get("cleared"):
            continue
        counts[d["kind_zh"]] = counts.get(d["kind_zh"], 0) + 1
        area[d["kind_zh"]] = round(area.get(d["kind_zh"], 0.0) + d["area_km2"], 3)
    n_work = np.bincount([w["occurrence"] for w in R["workings"]], minlength=len(R["occurrences"])) if R["workings"] else np.zeros(len(R["occurrences"]), dtype=int)
    for o in R["occurrences"]:
        o["n_workings"] = int(n_work[o["id"]])
    wc: dict[str, int] = {}
    for w in R["workings"]:
        wc[w["kind_zh"]] = wc.get(w["kind_zh"], 0) + 1
    R["counts"], R["area_km2"], R["workings_counts"] = counts, area, wc
    n_land = max(1, int(land.sum()))
    R["non_timber_share"] = round(float(((res > 0) & (res != RES_INDEX["timber"])).sum()) / n_land, 4)
    cover = g["landcover"]
    J["landcover"]["share"] = {LANDCOVER_CLASSES[i]: round(float(((cover == i) & land).sum()) / n_land, 4) for i in range(1, 12)}
    J["resources"] = {"zones_share": R["zones"]["share"], "counts": counts, "workings": wc, "old_island_lithology": R["geology"]["old_island_lithology"]}


def write_resources(out: Path, g: dict) -> None:
    from .grid import write_png8
    write_png8(out / "resources.png", g["resource"], RES_PALETTE)
    write_png8(out / "terrain_zone.png", g["terrain_zone"], ZONE_PALETTE)
    (out / "resources.json").write_text(json.dumps(g["resources"], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def write_preview_resources(out: Path, g: dict) -> Path:
    """资源总览：左 = 地形区（晕渲叠色），右 = 主导资源与采场 / 点位（主岛放大）。"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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
    ii, jj = np.where(g["island_id"] == 0)
    pad = 5
    a0, a1, b0, b1 = max(0, ii.min() - pad), min(H, ii.max() + pad + 1), max(0, jj.min() - pad), min(W, jj.max() + pad + 1)
    sub = (slice(a0, a1), slice(b0, b1))
    rp = np.array(RES_PALETTE, dtype=float) / 255.0
    rr = g["resource"][sub]
    base = shade[sub].copy()
    pts_codes = [RES_INDEX[k] for k in ("spring", "cave", "hotspring")]
    faint = np.isin(rr, [RES_INDEX["timber"], RES_INDEX["stone"]])        # 林木、石料铺得最广：淡淡一层，不盖住别的
    patch = (rr > 0) & ~np.isin(rr, pts_codes) & ~faint
    base[faint] = 0.7 * base[faint] + 0.3 * rp[rr[faint]]
    base[patch] = 0.25 * base[patch] + 0.75 * rp[rr[patch]]
    base[g["river"][sub] > 0] = (0.15, 0.35, 0.85)
    base[g["lake"][sub]] = (0.12, 0.25, 0.7)
    sext = [x0 + b0 * res_m / 1000.0, x0 + b1 * res_m / 1000.0, y0 - a1 * res_m / 1000.0, y0 - a0 * res_m / 1000.0]
    axes[1].imshow(base, extent=sext, origin="upper", interpolation="nearest")
    wmarks = {"ore": ("s", 26), "sulfur": ("X", 30), "placer": ("P", 30), "stone": ("D", 14), "clay": (".", 20), "gravel": (".", 20)}
    for key, (mk_, sz) in wmarks.items():
        pts = [w["km"] for w in R["workings"] if w["kind"] == key and w["island"] == 0]
        if pts:
            p = np.array(pts)
            axes[1].scatter(p[:, 0], p[:, 1], marker=mk_, s=sz, c=[tuple(rp[RES_INDEX[key]])], edgecolors="black", linewidths=0.4, zorder=6)
    pmarks = {"spring": ("o", 14), "cave": ("^", 34), "hotspring": ("*", 60)}
    for key, (mk_, sz) in pmarks.items():
        pts = [d["km"] for d in R["deposits"] if d["kind"] == key and d["island"] == 0]
        if pts:
            p = np.array(pts)
            axes[1].scatter(p[:, 0], p[:, 1], marker=mk_, s=sz, c=[tuple(rp[RES_INDEX[key]])], edgecolors="black", linewidths=0.5, zorder=5)
    handles = [Patch(color=tuple(rp[RES_INDEX[k[0]]]), label=f"{k[1]} {R['counts'].get(k[1], 0)}") for k in RES_KINDS if k[3] != "point" and R["counts"].get(k[1])]
    handles += [Line2D([], [], marker=wmarks[k][0], ls="", color=tuple(rp[RES_INDEX[k]]), markeredgecolor="black", label=f"{WORK_ZH[k]} {R['workings_counts'].get(WORK_ZH[k], 0)}")
                for k in wmarks if R["workings_counts"].get(WORK_ZH[k])]
    handles += [Line2D([], [], marker=pmarks[k][0], ls="", color=tuple(rp[RES_INDEX[k]]), markeredgecolor="black", label=f"{RES_NAMES[RES_INDEX[k]]} {R['counts'].get(RES_NAMES[RES_INDEX[k]], 0)}")
                for k in pmarks if R["counts"].get(RES_NAMES[RES_INDEX[k]])]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.75)
    geo = R["geology"]
    axes[1].set_title(f"主岛资源（底色 = 主导类，符号 = 采场与点；图例为全群计数）· 最近板块边界：{geo['boundary_type']}（核 {geo['boundary_kernel']:.2f}）"
                      f"{' · 叠层' if geo['layered'] else ''} · 老岛岩性 {geo['old_island_lithology']}", fontsize=10)
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
