"""⑤ 障碍识别：区域障碍（规则生成命名 A/B/C/D/F/G）+ 边局部因子 → 每边四模式通过率。

关键：区域障碍用穿越坐标 Φ_b(j) ∈ [0,1]，边指数 f_b(e) = |Φ_b(v) − Φ_b(u)|，
perm_m(e) = Π_b P_b[m]^f_b × Π 局部因子。
任一单调穿越 Σf = 1 → 乘积恰为矩阵值，与跳数、岛数、seed 无关（docs/02 §五 数据模型）。
障碍是选择性过滤器，不是墙（D34）。
"""
from __future__ import annotations

import numpy as np

from .. import MODES
from ..skeleton import d_lat_range
from ..sphere import latlon_to_xyz, angdist

REGIONAL_ORDER = ["A", "B", "C", "D", "F_N", "F_S"]  # G 单独处理（删边）


def _band_range(band: str, bands: dict, band_local: dict | None = None, lon=None):
    """带障碍的 (lo, hi)。给了 band_local（④ 产物）就按每个点的经度取局部带界（第三批 3：带界是波状线），
    Φ = (lat − lo(lon)) / (hi(lon) − lo(lon))，「任一单调穿越 Σf = 1」的性质不变。"""
    if band_local is None:
        t = {"subtropical_calm_n": (bands["trades_top_deg"], bands["calm_top_deg"]),
             "subtropical_calm_s": (-bands["calm_top_deg"], -bands["trades_top_deg"]),
             "westerlies_n": (bands["calm_top_deg"], bands["westerlies_top_deg"]),
             "westerlies_s": (-bands["westerlies_top_deg"], -bands["calm_top_deg"])}
        return t[band]
    from .s02_wind import local_edges
    e = local_edges(band_local, lon)
    t = {"subtropical_calm_n": (e["trades_n"], e["calm_n"]),
         "subtropical_calm_s": (e["calm_s"], e["trades_s"]),
         "westerlies_n": (e["calm_n"], e["west_n"]),
         "westerlies_s": (e["west_s"], e["calm_s"])}
    return t[band]


def node_phi(cfg: dict, planet_bands: dict, lat: np.ndarray, lon: np.ndarray,
             band_local: dict | None = None, band_scale: float = 1.0) -> dict[str, np.ndarray]:
    """每个区域障碍的穿越坐标 Φ_b（NaN = 该节点不在障碍定义域内，f 记 0）。band_local：局部带界（④）。
    kind：eq_core（赤道核心）/ band（整条风带，随局部带界起伏）/ lat_band（固定纬度区间，骨架第二版）/ void（D）。"""
    sk = cfg["skeleton"]
    phis: dict[str, np.ndarray] = {}
    barriers = cfg["s05"]["barriers"]
    for bid in REGIONAL_ORDER:
        b = barriers[bid]
        if b["kind"] == "eq_core":
            core = float(sk["eq_core_halfwidth_deg"])
            phis[bid] = np.clip((lat + core) / (2 * core), 0.0, 1.0)
        elif b["kind"] == "band":
            lo, hi = _band_range(b["band"], planet_bands, band_local, lon)
            phis[bid] = np.clip((lat - lo) / (hi - lo), 0.0, 1.0)
        elif b["kind"] == "lat_band":
            lo, hi = (float(x) * band_scale for x in b["lat_range"])
            phis[bid] = np.clip((lat - lo) / (hi - lo), 0.0, 1.0)
        elif b["kind"] == "void":
            lon_w, lon_e = float(sk["d_lon_west"]), float(sk["d_lon_east"])
            width = (lon_e - lon_w) % 360.0
            delta = (lon - lon_w) % 360.0
            inside = delta <= width
            # 空域外：两侧是 0/1 台地（按更近的一侧），保证进入空域的第一跳也计入份额
            east_side = (delta - width) <= (360.0 - delta)
            phi = np.where(inside, delta / width, np.where(east_side, 1.0, 0.0))
            # 纬度域 = skeleton.d_lat_range（与 s03 的密度修饰同一份定义；骨架第二版随核心北移）
            d_lo, d_hi = (float(x) * band_scale for x in sk["d_lat_range"])
            in_lat = (lat >= d_lo) & (lat <= d_hi)
            phis[bid] = np.where(in_lat, phi, np.nan)
        else:
            raise ValueError(f"未知区域障碍 kind: {b['kind']}")
    return phis


def _g_blocked(xyz_a, xyz_b, g_xyz, g_r_rad) -> np.ndarray:
    """边的大圆弧段到 G 中心的最小角距 < R_G → 阻断（端点判定会漏掉穿越长边）。"""
    da = angdist(xyz_a, g_xyz[None, :])
    db = angdist(xyz_b, g_xyz[None, :])
    n = np.cross(xyz_a, xyz_b)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_hat = np.where(norm > 1e-12, n / norm, n)
    cross_track = np.abs(np.arcsin(np.clip(np.sum(n_hat * g_xyz[None, :], axis=-1), -1, 1)))
    # 垂足是否落在弧段内：垂足 = 归一化(g − (g·n̂)n̂)
    dot_gn = np.sum(g_xyz[None, :] * n_hat, axis=-1, keepdims=True)
    proj = g_xyz[None, :] - dot_gn * n_hat
    pn = np.linalg.norm(proj, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = np.where(pn > 1e-12, proj / pn, xyz_a)
    seg = angdist(xyz_a, xyz_b)
    within = (angdist(xyz_a, proj) + angdist(proj, xyz_b)) <= seg + 1e-9
    min_d = np.where(within, cross_track, np.minimum(da, db))
    return min_d < g_r_rad


def run(ctx):
    cfg = ctx.cfg
    s5 = ctx.section(5)
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    g_info = ctx.load_json(2, "bands")["G"]
    isl = ctx.load_npz(3, "islands")
    ce = ctx.load_npz(3, "cand_edges")
    lat, lon = isl["lat"], isl["lon"]
    xyz = isl["xyz"]
    src, dst = ce["src"], ce["dst"]
    dist_days = ce["dist_days"]
    E = src.size
    perm_min = float(s5["perm_min"])
    ships = cfg["shared"]["ships"]

    # ---- 区域障碍 Φ 与边指数 f ----
    band_local = ctx.load_npz(4, "band_local")     # 局部带界（第三批 3）
    phis = node_phi(cfg, bands, lat, lon, band_local, float(planet.get("band_scale", 1.0)))
    f = np.zeros((E, len(REGIONAL_ORDER)), dtype=np.float64)
    for bi, bid in enumerate(REGIONAL_ORDER):
        phi = phis[bid]
        pu, pv = phi[src], phi[dst]
        fb = np.abs(pv - pu)
        fb = np.where(np.isnan(fb), 0.0, fb)
        f[:, bi] = fb

    perm = np.ones((E, 4), dtype=np.float64)
    for bi, bid in enumerate(REGIONAL_ORDER):
        P = np.array([float(s5["barriers"][bid]["permeability"][m]) for m in MODES])
        with np.errstate(divide="ignore", invalid="ignore"):
            logP = np.where(P > 0, np.log(P), -np.inf)
            contrib = f[:, bi:bi + 1] * logP[None, :]
        contrib = np.where(f[:, bi:bi + 1] > 0, contrib, 0.0)  # 0×(−inf) → 0
        perm *= np.exp(contrib)

    crosses_regional = f.sum(axis=1) > 0.05

    # ---- 边局部因子（只作用于不跨区域障碍的边，避免双计）----
    local_perm = {}
    # 宽阔无岛空域：P^((span − small)/ref)
    gp = s5["local"]["gap"]
    span_excess = np.maximum(0.0, dist_days - float(ships["small_days"])) / float(gp["ref_days"])
    gap_pm = np.ones((E, 4))
    for mi, m in enumerate(MODES):
        gap_pm[:, mi] = float(gp["permeability"][m]) ** span_excess
    gap_pm[crosses_regional] = 1.0
    local_perm["gap"] = gap_pm
    # 岛密度骤降：P^(|ln ρu/ρv| / ln ratio_ref)
    dp = s5["local"]["density_drop"]
    rho = np.maximum(isl["density_at"].astype(np.float64), 1e-9)
    ratio = np.abs(np.log(rho[src] / rho[dst])) / np.log(float(dp["ratio_ref"]))
    dens_pm = np.ones((E, 4))
    for mi, m in enumerate(MODES):
        dens_pm[:, mi] = float(dp["permeability"][m]) ** ratio
    dens_pm[crosses_regional] = 1.0
    local_perm["density_drop"] = dens_pm
    # 高度落差：筛「谁付得起」（原则乙：不筛贵贱）
    cp = s5["local"]["climb"]
    dh = np.abs(isl["height_m"][src].astype(np.float64) - isl["height_m"][dst].astype(np.float64))
    climb_pm = np.ones((E, 4))
    hit = dh > float(cp["h_thr_m"])
    for mi, m in enumerate(MODES):
        climb_pm[hit, mi] = float(cp["permeability"][m])
    local_perm["climb"] = climb_pm
    # 政治性障碍（人为、可变；默认空）
    pol_pm = np.ones((E, 4))
    mid = xyz[src] + xyz[dst]
    mid /= np.linalg.norm(mid, axis=-1, keepdims=True)
    from ..sphere import xyz_to_latlon
    mlat, mlon = xyz_to_latlon(mid)
    for ov in s5.get("political", {}).get("overrides", []):
        la, lb = ov["lat_range"]
        lo_a, lo_b = ov["lon_range"]
        hitp = (mlat >= la) & (mlat <= lb) & (((mlon - lo_a) % 360.0) <= ((lo_b - lo_a) % 360.0))
        for mi, m in enumerate(MODES):
            pol_pm[hitp, mi] *= float(ov["permeability"][m])
    local_perm["political"] = pol_pm

    for lp in local_perm.values():
        perm *= lp

    # ---- G：绝对阻断，但只阻断一个点（改道型）----
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    g_blocked = _g_blocked(xyz[src], xyz[dst], g_xyz, np.radians(g_info["radius_deg"]))
    perm_no_g = np.where(perm < perm_min, 0.0, np.clip(perm, 0.0, 1.0))  # 反事实：没有 G 的世界
    perm[g_blocked] = 0.0

    perm = np.where(perm < perm_min, 0.0, np.clip(perm, 0.0, 1.0))

    # ---- 障碍对象表 ----
    barriers_out = {}
    for bi, bid in enumerate(REGIONAL_ORDER):
        b = s5["barriers"][bid]
        geom = {}
        if b["kind"] == "band":
            lo, hi = _band_range(b["band"], bands)
            geom = {"lat_range": [lo, hi]}
        elif b["kind"] == "lat_band":
            geom = {"lat_range": [float(x) * float(planet.get("band_scale", 1.0)) for x in b["lat_range"]]}
        elif b["kind"] == "eq_core":
            core = float(cfg["skeleton"]["eq_core_halfwidth_deg"])
            geom = {"lat_range": [-core, core], "wraps_globe": True}
        elif b["kind"] == "void":
            geom = {"lon_range": [cfg["skeleton"]["d_lon_west"], cfg["skeleton"]["d_lon_east"]],
                    "lat_range": list(d_lat_range(cfg, planet))}
        barriers_out[bid] = {
            "kind": b["kind"], "type": b["type"], "geometry": geom,
            "seasonal": bool(b.get("seasonal", False)),
            "permeability": {m: float(b["permeability"][m]) for m in MODES},
            "n_edges_crossing": int((f[:, bi] > 0.01).sum()),
        }
    barriers_out["G"] = {
        "kind": "disc", "type": "deflecting",
        "geometry": {"lat": g_info["lat"], "lon": g_info["lon"], "radius_deg": g_info["radius_deg"]},
        "seasonal": False,
        "permeability": {m: 0.0 for m in MODES},
        "n_edges_blocked": int(g_blocked.sum()),
        "note": "绝对阻断，但只阻断一个点；直线航路被堵死，所有往来必须绕行（docs/11 §六）",
    }

    ctx.save_npz(5, "perm", perm=perm.astype(np.float64), perm_no_g=perm_no_g.astype(np.float64),
                 f_regional=f.astype(np.float32),
                 g_blocked=g_blocked,
                 gap=local_perm["gap"].astype(np.float32),
                 density_drop=local_perm["density_drop"].astype(np.float32),
                 climb=local_perm["climb"].astype(np.float32),
                 political=local_perm["political"].astype(np.float32))
    ctx.save_json(5, "barriers", {"regional_order": REGIONAL_ORDER, "barriers": barriers_out,
                                  "modes": list(MODES)})
    removed = {m: int((perm[:, mi] == 0).sum()) for mi, m in enumerate(MODES)}
    return {"n_edges": E, "g_blocked": int(g_blocked.sum()), "edges_removed_per_mode": removed}
