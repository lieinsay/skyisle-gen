"""浮石板块（第三批第 2 步，docs/PLAN-BATCH3.md 5.2）：给 ③ 的密度场与高度场提供「板块逻辑」。

世界观来源（docs/02 §九）：老岛下沉，浮石骨架漂浮，随洋流汇聚嵌合，再极缓慢升起。
→ 浮石以「板块」漂移：汇聚边界堆出密度脊与叠层（碰撞带），离散边界拉出空域，走滑边界错断岛链；
  热点链是固定点上拖出的线性岛链（年龄单调）；② 的风场辐合再给一层纬向纹理。
这里只产出静态格局（原则甲：没有任何机制或玩法依赖「过程」）。全部 numpy，确定性（随机数只来自传入的 rng）。

产出（都在 1° 网格上）：
  factor       密度乘子 = 板块丰度 × 边界核 × 辐合纹理 × 热点链
  conv_kernel  汇聚边界核 ∈ [0,1]（③ 用它决定抬升与叠层区）
  age          岛龄 ∈ [0,1]（0 新 1 老；离散边界处新、远离处老；热点链沿链单调）
  plate_id     所属板块；btype  最近边界的类型（0 汇聚 / 1 离散 / 2 走滑）
"""
from __future__ import annotations

import numpy as np

from .sphere import latlon_to_xyz

CONVERGENT, DIVERGENT, TRANSFORM = 0, 1, 2


def _random_points(rng, n: int, lat_max: float = 90.0) -> np.ndarray:
    z = rng.uniform(-np.sin(np.radians(lat_max)), np.sin(np.radians(lat_max)), n)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(z))
    return latlon_to_xyz(lat, lon)


def _divergence(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """球面水平散度（单位 1/m，只用相对值）：∂u/∂x + ∂(v cosφ)/∂y / cosφ；经度周期，纬度单侧差分。"""
    R = 6.371e6
    phi = np.radians(lats)[:, None]
    cosphi = np.maximum(np.cos(phi), 0.05)
    dlam = np.radians(lons[1] - lons[0])
    dphi = np.radians(lats[1] - lats[0])
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dlam * R * cosphi)
    vc = v * np.cos(phi)
    dv_dy = np.gradient(vc, dphi, axis=0) / (R * cosphi)
    return du_dx + dv_dy


def plate_fields(rng, lats: np.ndarray, lons: np.ndarray, p: dict,
                 wind_u: np.ndarray | None = None, wind_v: np.ndarray | None = None) -> dict:
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    pts = latlon_to_xyz(LAT.ravel(), LON.ravel())          # [M, 3]
    n_pl = int(p["n_plates"])
    seeds = _random_points(rng, n_pl)                       # [P, 3]
    plate_mult = np.exp(rng.normal(0.0, float(p["plate_sigma"]), n_pl))
    # 无序板块对 → 边界类型（用一张上三角随机表，确定性）
    pair_u = rng.uniform(0.0, 1.0, (n_pl, n_pl))
    pair_u = np.triu(pair_u, 1)
    pair_u = pair_u + pair_u.T
    p_c, p_d = float(p["p_convergent"]), float(p["p_divergent"])
    btype_tab = np.where(pair_u < p_c, CONVERGENT, np.where(pair_u < p_c + p_d, DIVERGENT, TRANSFORM))

    dots = np.clip(pts @ seeds.T, -1.0, 1.0)                # [M, P]
    order = np.argsort(-dots, axis=1)[:, :2]
    i1, i2 = order[:, 0], order[:, 1]
    d1 = np.arccos(dots[np.arange(pts.shape[0]), i1])
    d2 = np.arccos(dots[np.arange(pts.shape[0]), i2])
    dist_deg = np.degrees(0.5 * (d2 - d1))                  # 到 Voronoi 边界的近似角距
    w = float(p["boundary_width_deg"])
    kern = np.exp(-0.5 * (dist_deg / w) ** 2)
    btype = btype_tab[i1, i2]

    # 走滑：沿边界方向的错断调制（沿边界坐标 = 点在两板块种子法向上的投影角）
    nvec = np.cross(seeds[i1], seeds[i2])
    nvec /= np.maximum(np.linalg.norm(nvec, axis=1, keepdims=True), 1e-12)
    s_along = np.degrees(np.arcsin(np.clip(np.sum(pts * nvec, axis=1), -1.0, 1.0)))
    seg = float(p["transform_segment_deg"])
    trans_mod = 0.5 + 0.5 * np.sin(2.0 * np.pi * s_along / seg)

    bfac = np.where(btype == CONVERGENT, 1.0 + float(p["convergent_boost"]) * kern,
                    np.where(btype == DIVERGENT, 1.0 - float(p["divergent_cut"]) * kern,
                             1.0 + float(p["transform_boost"]) * kern * trans_mod))
    factor = plate_mult[i1] * bfac
    conv_kernel = np.where(btype == CONVERGENT, kern, 0.0)

    # 岛龄：任何边界附近都较新（板块活动带），板块内部老；离散边界（扩张脊）处最新
    age_scale = float(p["age_scale_deg"])
    age = 0.3 + 0.7 * np.clip(dist_deg / age_scale, 0.0, 1.0)
    age = np.where(btype == DIVERGENT, 0.5 * age, age)

    # 热点链：固定点 + 漂移方向 → 线性岛链，年龄沿链单调（t=0 最新端）
    n_hot = int(p["n_hotspots"])
    hot = np.zeros(pts.shape[0])
    if n_hot > 0:
        starts = _random_points(rng, n_hot, lat_max=70.0)
        bearings = rng.uniform(0.0, 2.0 * np.pi, n_hot)
        length = np.radians(float(p["hotspot_length_deg"]))
        spacing = np.radians(float(p["hotspot_spacing_deg"]))
        r_h = np.radians(float(p["hotspot_radius_deg"]))
        n_pts = max(2, int(round(length / spacing)) + 1)
        for k in range(n_hot):
            c = starts[k]
            # 局部东/北基向量，按方位角合成切向，沿大圆前进
            east = np.array([-c[1], c[0], 0.0]); east /= max(np.linalg.norm(east), 1e-12)
            north = np.cross(c, east)
            tdir = np.cos(bearings[k]) * north + np.sin(bearings[k]) * east
            for m in range(n_pts):
                t = m / (n_pts - 1)
                ang = t * length
                q = np.cos(ang) * c + np.sin(ang) * tdir
                dq = np.arccos(np.clip(pts @ q, -1.0, 1.0))
                bump = np.exp(-0.5 * (dq / r_h) ** 2) * (1.0 - 0.6 * t)
                hot = np.maximum(hot, bump)
                # 链上年龄：靠近这个点的格点取 t（新端 0 → 老端 1）
                near = dq < 1.5 * r_h
                age[near] = np.where(bump[near] > 0.3, t, age[near])
    factor = factor * (1.0 + float(p["hotspot_boost"]) * hot)

    # ② 风场辐合纹理：辐合线密度增强、辐散线减弱（docs/02 §九「随洋流汇聚嵌合」）
    if wind_u is not None and float(p["convergence_kappa"]) != 0.0:
        conv = -_divergence(wind_u.astype(np.float64), wind_v.astype(np.float64), lats, lons)
        mid = np.abs(LAT) < 80.0
        sd = float(conv[mid].std()) or 1.0
        z = np.clip(conv / sd, -2.0, 2.0)
        factor = factor * np.exp(float(p["convergence_kappa"]) * z.ravel())

    shape = LAT.shape
    return {
        "factor": factor.reshape(shape),
        "conv_kernel": conv_kernel.reshape(shape),
        "age": np.clip(age, 0.0, 1.0).reshape(shape),
        "plate_id": i1.astype(np.int16).reshape(shape),
        "btype": btype.astype(np.int8).reshape(shape),
        "boundary_kernel": kern.reshape(shape),
        "seeds_xyz": seeds,
    }
