"""③ 岛群分布：密度场 → 面积加权采样 → kNN → 陆地（势力范围 × 陆地占比）→ 分类 → 纯几何候选边集。

节点 = 岛群（R10）：一个节点是一个岛群 = 一个「邑」= 一个水共同体；群内数十小岛彼此 5–15 km，
属第三层、不进管线。产物字段沿用 islands/n_islands 等旧名，语义均为「群」。
密度 = 带基线 × exp(γ·噪声) × 骨架修饰（D 空域、绕道岛弧、赤道无岛核心）。
陆地 area_km2 = 势力范围 T × 陆地占比 f（R8）；T 依赖 kNN，故 kNN 必须先于陆地计算。
候选边 = kNN 并集 + 远程边（≤ 大船航程）+ 跨赤道远征边 + 连通性回退。
"""
from __future__ import annotations

import numpy as np

from ..noise import fractal_noise
from ..rng import stage_rng
from ..sphere import angdist, grid_axes, grid_interp, knn, latlon_to_xyz
from .s02_wind import band_id_of_lat

CLASS_NAMES = ["dense", "medium", "sparse", "isolated"]
CLASS_ZH = {"dense": "密接群岛", "medium": "中疏诸岛", "sparse": "稀疏岛链", "isolated": "孤悬散岛"}
KIND_KNN, KIND_FAR, KIND_EXPEDITION, KIND_FALLBACK = 0, 1, 2, 3


def _density_grid(ctx, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = ctx.section(3)["islands"]
    sk = ctx.cfg["skeleton"]
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    g_info = ctx.load_json(2, "bands")["G"]
    res = float(ctx.cfg["shared"]["grid_res_deg"])
    lats, lons = grid_axes(res)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    bd = s["band_density"]
    band = band_id_of_lat(LAT, bands)
    base = np.select(
        [band == 0,
         np.isin(band, [1, 5]), np.isin(band, [2, 6]),
         np.isin(band, [3, 7]), np.isin(band, [4, 8])],
        [float(bd["equatorial_margin"]), float(bd["trades"]), float(bd["subtropical_calm"]),
         float(bd["westerlies"]), float(bd["polar"])])

    noise = fractal_noise(rng, lats.size, lons.size,
                          base_cells=int(s["noise_base_cells"]), octaves=int(s["noise_octaves"]),
                          persistence=float(s["noise_persistence"]))
    dens = base * np.exp(float(s["noise_gamma"]) * noise)

    # 赤道无岛核心（障碍 A 的几何：全球皆海洋 → 永暴带内无立足处）
    core = float(sk["eq_core_halfwidth_deg"])
    dens[np.abs(LAT) < core] = 0.0

    # 中央宽空域 D：从赤道无岛核心边缘到无风带顶，经度 [west, east]
    # （下界必须是核心边缘而非风系带界，否则赤道缘岛条带成为绕过 D 的走廊）
    lon_w, lon_e = float(sk["d_lon_west"]), float(sk["d_lon_east"])
    d_lat_mask = (LAT >= float(sk["eq_core_halfwidth_deg"])) & (LAT <= bands["calm_top_deg"])
    in_d_lon = ((LON - lon_w) % 360.0) <= ((lon_e - lon_w) % 360.0)
    d_mask = d_lat_mask & in_d_lon
    dens[d_mask] *= float(sk["d_density_mult"])

    # 绕道岛弧：G 南北两段（仍远低于带基线）→ 中转岛的物理基础
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    pts = latlon_to_xyz(LAT.ravel(), LON.ravel()).reshape(LAT.shape + (3,))
    dg = np.degrees(angdist(pts, g_xyz[None, None, :]))
    arc_r = float(g_info["radius_deg"]) + float(sk["detour_arc_gap_deg"])
    arc = (np.abs(dg - arc_r) < float(sk["detour_arc_halfwidth_deg"])) & d_mask
    dens[arc] *= float(sk["detour_arc_boost"])
    # G 盘内无岛：靠得太近就是死（docs/11 §六）。否则盘内岛全边被阻断，成为孤岛
    dens[dg < float(g_info["radius_deg"])] = 0.0

    return lats, lons, dens


def _sample_islands(rng, lats, lons, dens, n_target: int, g_xyz, g_r_deg, core_deg):
    """拒绝采样。G 盘与赤道核心用解析排除（网格插值会在边界泄漏出岛）。"""
    dmax = float(dens.max())
    got_lat, got_lon = [], []
    n = 0
    while n < n_target:
        m = max(4096, 2 * (n_target - n))
        z = rng.uniform(-1.0, 1.0, m)
        lat = np.degrees(np.arcsin(z))
        lon = rng.uniform(-180.0, 180.0, m)
        u = rng.uniform(0.0, 1.0, m)
        d = grid_interp(dens, lats, lons, lat, lon)
        ok = u < d / dmax
        ok &= np.abs(lat) >= core_deg
        ok &= np.degrees(angdist(latlon_to_xyz(lat, lon), g_xyz[None, :])) >= g_r_deg
        got_lat.append(lat[ok])
        got_lon.append(lon[ok])
        n += int(ok.sum())
    lat = np.concatenate(got_lat)[:n_target]
    lon = np.concatenate(got_lon)[:n_target]
    return lat, lon


HEX_FACTOR = 0.866  # 六边形填充：势力范围 = (√3/2) × 间距²


def _land(rng, s, scale, spacing_km, density_at):
    """陆地 = 势力范围 × 陆地占比（R8）。

    T_j = 0.866 × spacing_j²；f_j = min(f_cap, f0 · (ρ_j/ρ_med)^α · exp(N(0, σ)))；area_j = T_j f_j。
    f0 由 Σ area_j = total_land_km2 二分反解（封顶后的和对 f0 单调不减，可二分）。
    返回 (territory, land_frac, area, f0)。
    """
    alpha = float(s["land_frac_alpha"])
    cap = float(s["land_frac_cap"])
    sigma = float(s["land_frac_sigma"])
    target = float(scale["total_land_km2"])
    territory = HEX_FACTOR * spacing_km.astype(np.float64) ** 2
    rho = np.maximum(density_at.astype(np.float64), 1e-9)
    shape = (rho / float(np.median(rho))) ** alpha
    if sigma > 0:
        shape = shape * np.exp(rng.normal(0.0, sigma, rho.size))
    if cap * territory.sum() < target:
        raise ValueError(
            f"[s03] 陆地目标 {target:.0f} km² 超过几何上限 f_cap × Σ势力范围 = "
            f"{cap * territory.sum():.0f} km²：调低 shared.scale.total_land_km2 或提高 land_frac_cap")
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        f0 = float(np.sqrt(lo * hi))
        f = np.minimum(cap, f0 * shape)
        if float((territory * f).sum()) < target:
            lo = f0
        else:
            hi = f0
        if hi / lo < 1.0 + 1e-10:
            break
    f0 = float(np.sqrt(lo * hi))
    land_frac = np.minimum(cap, f0 * shape)
    area = territory * land_frac
    return territory, land_frac, area, f0


def run(ctx):
    s = ctx.section(3)["islands"]
    ships = ctx.cfg["shared"]["ships"]
    planet = ctx.load_json(1, "planet")
    rng = stage_rng(ctx.seed, 3)
    km_per_rad = planet["radius_km"]
    days_per_rad = km_per_rad / planet["day_range_km"]

    lats_g, lons_g, dens = _density_grid(ctx, rng)
    n_target = int(s["n_islands"])
    g_info0 = ctx.load_json(2, "bands")["G"]
    g_xyz0 = latlon_to_xyz(np.array(g_info0["lat"]), np.array(g_info0["lon"]))
    lat, lon = _sample_islands(rng, lats_g, lons_g, dens, n_target,
                               g_xyz0, float(g_info0["radius_deg"]),
                               float(ctx.cfg["skeleton"]["eq_core_halfwidth_deg"]))

    # 稳定编号：按 (2° 纬度桶, 经度) 排序（docs/12 §七 确定性）
    order = np.lexsort((lon, np.round(lat / 2.0)))
    lat, lon = lat[order], lon[order]
    xyz = latlon_to_xyz(lat, lon)

    density_at = grid_interp(dens, lats_g, lons_g, lat, lon)

    hfield = fractal_noise(rng, lats_g.size, lons_g.size,
                           base_cells=int(s["height_noise_cells"]), octaves=3)
    height = (0.5 + 0.5 * grid_interp(hfield, lats_g, lons_g, lat, lon)) * float(s["height_scale_m"])
    height = height + rng.normal(0.0, float(s["height_jitter_m"]), n_target)
    # 叠层岛区：堆叠噪声区内高度呈双层分布（岛在不同高度堆叠，docs/02 §三）
    stack_field = fractal_noise(rng, lats_g.size, lons_g.size, base_cells=12, octaves=2)
    in_stack = grid_interp(stack_field, lats_g, lons_g, lat, lon) > float(s["stack_zone_threshold"])
    height = np.where(in_stack & (rng.uniform(0, 1, n_target) < 0.5),
                      height + float(s["stack_range_m"]), height)
    height = np.clip(height, 50.0, None)

    # ---- kNN 与候选边 ----
    k = int(s["knn_k"])
    n_far = int(s["n_far"])
    idx, ang = knn(xyz, k + n_far)
    dist_days_nn = ang * days_per_rad
    big = float(ships["big_days"])
    mean_nn = dist_days_nn[:, :k].mean(axis=1)   # 群间平均间距（天）；分类与势力范围共用

    # ---- 陆地（R8/R10）：area_km2 = 群的总陆地 = 集雨面 = 人口容量 = 政治体量
    # （docs/02 §六「一群 = 一水共同体 = 一个基本政治单位」、§七「土地绝对有限」）。
    # 不是装饰字段：进 ④ 集雨容量、⑥ 介数源权重、⑦ 适宜度、⑨ 九格表 ①⑤⑧。
    # 建模为「势力范围 × 陆地占比」：几何自洽由构造保证，f 与密度正相关（现实群岛如此）。
    scale = ctx.cfg["shared"]["scale"]
    territory, land_frac, area, f0 = _land(rng, s, scale, mean_nn * planet["day_range_km"],
                                           density_at)
    # ---- 可用地率（R9）：只是一个标量，不生成岛内地形；与高度无关（原则乙的卫生习惯）
    lo_a, hi_a = (float(x) for x in s["arable_frac_range"])
    sig_a = float(s["arable_frac_sigma"])
    arable_frac = (float(scale["arable_frac_mean"])
                   * np.exp(rng.normal(-0.5 * sig_a * sig_a, sig_a, n_target)))
    arable_frac = np.clip(arable_frac, lo_a, hi_a)

    edges: dict[tuple[int, int], tuple[float, int]] = {}

    def add_edge(a: int, b: int, d: float, kind: int):
        key = (a, b) if a < b else (b, a)
        cur = edges.get(key)
        if cur is None or d < cur[0] - 1e-12 or (abs(d - cur[0]) <= 1e-12 and kind < cur[1]):
            edges[key] = (d, kind)

    for i in range(n_target):
        for j in range(k + n_far):
            d = dist_days_nn[i, j]
            if d <= big:
                add_edge(i, int(idx[i, j]), float(d), KIND_KNN if j < k else KIND_FAR)

    # G 邻域几何弦：环上两侧若在大船航程内，直线航路本应存在——被 G 阻断后
    # 才有「所有往来必须绕行」（docs/11 §六）。kNN 只连同侧近邻，须显式补候选。
    g_info = ctx.load_json(2, "bands")["G"]
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    sk = cfg_sk = ctx.cfg["skeleton"]
    reach_deg = (float(g_info["radius_deg"]) + float(sk["detour_arc_gap_deg"])
                 + float(sk["detour_arc_halfwidth_deg"]) + 1.0)
    near_g = np.where(np.degrees(angdist(xyz, g_xyz[None, :])) <= reach_deg)[0]
    n_chord = 0
    if near_g.size >= 2:
        dd = angdist(xyz[near_g][:, None, :], xyz[near_g][None, :, :]) * days_per_rad
        for ii in range(near_g.size):
            for jj in range(ii + 1, near_g.size):
                if dd[ii, jj] <= big:
                    add_edge(int(near_g[ii]), int(near_g[jj]), float(dd[ii, jj]), KIND_FAR)
                    n_chord += 1

    # 跨赤道远征边（障碍 A 的唯一穿越通道：几代人一次的国家远征）
    eq_top = planet["bands"]["eq_storm_top_deg"]
    north_m = (lat > 0) & (lat < eq_top + 2.0)
    south_m = (lat < 0) & (lat > -eq_top - 2.0)
    n_idx, s_idx = np.where(north_m)[0], np.where(south_m)[0]
    n_exp = 0
    if n_idx.size and s_idx.size:
        dd = angdist(xyz[n_idx][:, None, :], xyz[s_idx][None, :, :]) * days_per_rad
        flat = np.argsort(dd, axis=None, kind="stable")
        chosen_lons: list[float] = []
        min_sep = float(s["expedition_min_lon_sep_deg"])
        for f in flat:
            if n_exp >= int(s["expedition_pairs"]):
                break
            a = n_idx[f // s_idx.size]
            b = s_idx[f % s_idx.size]
            lon_mid = float(lon[a])
            if all(abs((lon_mid - c + 180.0) % 360.0 - 180.0) >= min_sep for c in chosen_lons):
                add_edge(int(a), int(b), float(dd[f // s_idx.size, f % s_idx.size]), KIND_EXPEDITION)
                chosen_lons.append(lon_mid)
                n_exp += 1

    # 连通性回退：任何岛都不允许与世隔绝到图之外（原则己）
    src = np.array([a for a, _ in edges], dtype=np.int64)
    dst = np.array([b for _, b in edges], dtype=np.int64)
    from ..graph import weak_components
    n_fallback = 0
    while True:
        comp = weak_components(n_target, src, dst)
        n_comp = comp.max() + 1
        if n_comp == 1:
            break
        sizes = np.bincount(comp)
        c_small = int(np.argmin(sizes))
        inside = np.where(comp == c_small)[0]
        outside = np.where(comp != c_small)[0]
        dd = angdist(xyz[inside][:, None, :], xyz[outside][None, :, :])
        f = int(np.argmin(dd))
        a = int(inside[f // outside.size])
        b = int(outside[f % outside.size])
        add_edge(a, b, float(dd.flat[f] * days_per_rad), KIND_FALLBACK)
        n_fallback += 1
        src = np.array([x for x, _ in edges], dtype=np.int64)
        dst = np.array([y for _, y in edges], dtype=np.int64)

    keys = sorted(edges)
    e_src = np.array([a for a, _ in keys], dtype=np.int64)
    e_dst = np.array([b for _, b in keys], dtype=np.int64)
    e_dist = np.array([edges[k_][0] for k_ in keys])
    e_kind = np.array([edges[k_][1] for k_ in keys], dtype=np.int8)

    # ---- 分类：能力阈值（docs/02 §三），与船只参数共用一组 ----
    cls = np.select(
        [mean_nn < float(ships["bridge_days"]), mean_nn < float(ships["small_days"]),
         mean_nn < big],
        [0, 1, 2], default=3).astype(np.int8)
    nb_h = height[idx[:, :k]]
    layered = (np.std(np.concatenate([nb_h, height[:, None]], axis=1), axis=1)
               > float(s["layered_height_std_m"]))

    # ---- 主岛与岛体（第三批第 1 步，PLAN-BATCH3 5.1）----
    # 主岛 = 群内最大的一座岛，占群陆地 main_frac；河（④）与挡风（②b）都看它。
    # 随机数放在本阶段所有抽样之后，不改变既有产物的随机序列。
    sig_m = float(s["main_frac_sigma"])
    lo_m, hi_m = (float(x) for x in s["main_frac_range"])
    main_frac = float(s["main_frac_mean"]) * np.exp(rng.normal(-0.5 * sig_m * sig_m, sig_m, n_target))
    main_frac = np.clip(main_frac, lo_m, hi_m)
    main_area = area * main_frac
    # 岛体：云带顶面 = 高度零点，岛底离云带顶 keel_clearance_m，岛体实心（决定 3）。
    # 墙高 = 迎风截面的高度；低于间隙的岛当薄片（墙高 0）。只作几何量，不进社会推导（原则乙同高度）。
    wall = np.maximum(0.0, height - float(s["keel_clearance_m"]))

    ctx.save_npz(3, "islands", lat=lat, lon=lon, xyz=xyz,
                 area_km2=area.astype(np.float32), height_m=height.astype(np.float32),
                 territory_km2=territory.astype(np.float32),
                 land_frac=land_frac.astype(np.float32),
                 arable_frac=arable_frac.astype(np.float32),
                 main_frac=main_frac.astype(np.float32),
                 main_area_km2=main_area.astype(np.float32),
                 wall_m=wall.astype(np.float32),
                 cls=cls, layered=layered, density_at=density_at.astype(np.float32),
                 mean_nn_days=mean_nn.astype(np.float32))
    ctx.save_npz(3, "cand_edges", src=e_src, dst=e_dst,
                 dist_days=e_dist, kind=e_kind)
    ctx.save_npz(3, "density_grid", lats=lats_g, lons=lons_g, density=dens.astype(np.float32))

    share = {CLASS_NAMES[i]: round(float((cls == i).mean()), 3) for i in range(4)}
    area_by_cls = {CLASS_NAMES[i]: round(float(np.median(area[cls == i])), 1)
                   for i in range(4) if (cls == i).any()}
    f_by_cls = {CLASS_NAMES[i]: round(float(np.median(land_frac[cls == i])), 4)
                for i in range(4) if (cls == i).any()}
    arable_km2 = area * arable_frac
    # 口径自检（shared.scale）：25M km² × 0.10 × 100 人/km² ≈ 2.5 亿人。只是摘要，不进模型
    pop = float(arable_km2.sum() * float(scale["people_per_arable_km2"]))
    return {"n_islands": n_target, "n_edges": len(keys), "n_expedition": n_exp,
            "n_fallback": n_fallback, "n_g_chords": n_chord, "class_share": share,
            "layered_share": round(float(layered.mean()), 3),
            "main_area_median_km2": round(float(np.median(main_area)), 0),
            "wall_median_m": round(float(np.median(wall)), 0),
            "land_total_km2": round(float(area.sum()), 0),
            "land_target_km2": float(scale["total_land_km2"]),
            "territory_total_km2": round(float(territory.sum()), 0),
            "land_frac_f0": round(f0, 5),
            "land_frac_mean": round(float(area.sum() / territory.sum()), 4),
            "land_frac_capped_share": round(
                float((land_frac >= float(s["land_frac_cap"]) - 1e-6).mean()), 3),
            "land_frac_median_by_class": f_by_cls,
            "area_median_km2": round(float(np.median(area)), 1),
            "area_p95_km2": round(float(np.quantile(area, 0.95)), 1),
            "area_max_km2": round(float(area.max()), 1),
            "area_median_by_class": area_by_cls,
            "arable_frac_mean": round(float(arable_frac.mean()), 4),
            "arable_total_km2": round(float(arable_km2.sum()), 0),
            "implied_population_M": round(pop / 1e6, 1)}
