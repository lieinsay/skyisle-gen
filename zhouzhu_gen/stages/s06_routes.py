"""⑥ 航线网络：有向成本（顺风廉价逆风昂贵）、分模式成本、抽样介数 → 干线与枢纽。

风是「轻度影响」（docs/02 §四）：只进成本，不进通过率。
干线/枢纽只由地理量（密度×面积加权的抽样介数）决定，不读文明中心（⑦ 在后，不可倒序）。
"""
from __future__ import annotations

import numpy as np

from .. import MODES
from ..graph import CSR, betweenness_sampled, weak_components
from ..rng import stage_rng
from ..sphere import grid_interp, initial_bearing, slerp_points, xyz_to_latlon, angdist, latlon_to_xyz


def build_directed(ctx):
    """把无向候选边展开为有向边。返回 (src_d, dst_d, und_id)；前 E 条 a→b，后 E 条 b→a。"""
    ce = ctx.load_npz(3, "cand_edges")
    src, dst = ce["src"], ce["dst"]
    src_d = np.concatenate([src, dst])
    dst_d = np.concatenate([dst, src])
    und_id = np.concatenate([np.arange(src.size), np.arange(src.size)])
    return src_d, dst_d, und_id


def run(ctx):
    r = ctx.section(6)["routes"]
    isl = ctx.load_npz(3, "islands")
    ce = ctx.load_npz(3, "cand_edges")
    wind = ctx.load_npz(2, "wind")
    clim = ctx.load_npz(4, "climate_grid")
    pm = ctx.load_npz(5, "perm")
    g_info = ctx.load_json(2, "bands")["G"]

    xyz = isl["xyz"]
    lat, lon = isl["lat"], isl["lon"]
    src, dst = ce["src"], ce["dst"]
    dist_days = ce["dist_days"]
    E = src.size
    N = lat.size

    # ---- 沿边采样：风（中点）、风暴（最大）----
    n_samp = int(r["edge_samples"])
    pts = slerp_points(xyz[src], xyz[dst], n_samp)  # [E, S, 3]
    plat, plon = xyz_to_latlon(pts.reshape(-1, 3))
    storm_s = grid_interp(clim["storm"].astype(np.float64), clim["lats"], clim["lons"],
                          plat, plon).reshape(E, n_samp)
    storm_max = storm_s.max(axis=1)
    storm_ng = grid_interp(clim["storm_no_g"].astype(np.float64), clim["lats"], clim["lons"],
                           plat, plon).reshape(E, n_samp).max(axis=1)
    mid = pts[:, n_samp // 2, :]
    mlat, mlon = xyz_to_latlon(mid)
    u = grid_interp(wind["u"].astype(np.float64), wind["lats"], wind["lons"], mlat, mlon)
    v = grid_interp(wind["v"].astype(np.float64), wind["lats"], wind["lons"], mlat, mlon)
    v_wind = np.hypot(u, v)
    wind_dir = np.arctan2(u, v)  # 方位角约定：0 = 北，顺时针

    # ---- 有向成本 ----
    tail, head = float(r["tailwind_factor"]), float(r["headwind_factor"])
    calm = np.minimum(float(r["calm_max"]),
                      1.0 + float(r["calm_kappa"]) * np.maximum(0.0, 1.0 - v_wind / float(r["calm_v_ref"])))
    storm_f = 1.0 + float(r["storm_kappa"]) * storm_max
    storm_f_ng = 1.0 + float(r["storm_kappa"]) * storm_ng
    h = isl["height_m"].astype(np.float64)

    def dir_factor(c):
        return np.where(c >= 0, 1.0 / (1.0 + (1.0 / tail - 1.0) * c), 1.0 + (head - 1.0) * (-c))

    def cost_one_dir(a_idx, b_idx):
        brg = initial_bearing(mlat, mlon, lat[b_idx], lon[b_idx])
        c = np.cos(wind_dir - brg)
        w = dir_factor(c) * calm
        climb = 1.0 + float(r["climb_kappa"]) * np.maximum(0.0, h[b_idx] - h[a_idx]) / 1000.0
        base = dist_days * storm_f * climb
        cost_phys = base * w
        cost_modes = np.empty((E, 4))
        for mi, m in enumerate(MODES):
            cost_modes[:, mi] = base * w ** float(r["alpha"][m])
        return cost_phys, cost_modes

    cost_ab, cm_ab = cost_one_dir(src, dst)
    cost_ba, cm_ba = cost_one_dir(dst, src)
    cost_d = np.concatenate([cost_ab, cost_ba])            # [2E] 物理成本
    cost_m = np.concatenate([cm_ab, cm_ba], axis=0)        # [2E, 4]
    ng_ratio = np.concatenate([storm_f_ng / storm_f, storm_f_ng / storm_f])
    cost_no_g = cost_d * ng_ratio                          # 反事实：从未有过 G（P6 用）
    perm_d = np.concatenate([pm["perm"], pm["perm"]], axis=0)  # 通过率对称

    src_d, dst_d, und_id = build_directed(ctx)
    csr = CSR(N, src_d, dst_d)

    # ---- 抽样介数（商旅可通的物理成本图）----
    rng = stage_rng(ctx.seed, 6)
    w_src = isl["density_at"].astype(np.float64) * isl["area_km2"].astype(np.float64)
    w_src /= w_src.sum()
    n_s = min(int(r["betweenness_sources"]), N)
    sources = np.sort(rng.choice(N, size=n_s, replace=False, p=w_src))
    trade_i = MODES.index("trade")
    w_bt = np.where(perm_d[:, trade_i] > 0, cost_d, np.inf)
    flow = betweenness_sampled(csr, w_bt, 2 * E, sources, c_min=float(r["betweenness_c_min_days"]))

    node_flow = np.zeros(N)
    np.add.at(node_flow, src_d, flow)
    np.add.at(node_flow, dst_d, flow)
    node_flow *= 0.5

    # 枢纽：全局 top 分位 ∪ G 邻域内 top 分位（改道逼出的中转岛候选）
    q_global = np.quantile(node_flow, 1.0 - float(r["hub_top_frac"]))
    hubs = set(np.where(node_flow >= q_global)[0].tolist())
    g_xyz = latlon_to_xyz(np.array(g_info["lat"]), np.array(g_info["lon"]))
    d_to_g = np.degrees(angdist(xyz, g_xyz[None, :]))
    near_g = d_to_g <= float(r["g_neighborhood_radii"]) * float(g_info["radius_deg"])
    if near_g.any():
        qg = np.quantile(node_flow[near_g], 1.0 - float(r["hub_g_top_frac"]))
        hubs |= set(np.where(near_g & (node_flow >= qg) & (node_flow > 0))[0].tolist())
    hubs_sorted = sorted(hubs, key=lambda i: (-node_flow[i], i))

    # ---- 连通分量报告（每模式）----
    comp_report = {}
    for mi, m in enumerate(MODES):
        keep = pm["perm"][:, mi] > 0
        comp = weak_components(N, src[keep], dst[keep])
        sizes = np.bincount(comp)
        comp_report[m] = {"n_components": int(sizes.size), "largest": int(sizes.max())}

    ctx.save_npz(6, "routes", cost=cost_d, cost_no_g=cost_no_g, cost_m=cost_m,
                 flow=flow.astype(np.float32), node_flow=node_flow.astype(np.float32),
                 src_d=src_d, dst_d=dst_d, und_id=und_id, betweenness_sources=sources)
    ctx.save_json(6, "hubs", {
        "hubs": [{"node": int(i), "lat": round(float(lat[i]), 3), "lon": round(float(lon[i]), 3),
                  "flow": round(float(node_flow[i]), 1), "near_g": bool(near_g[i])}
                 for i in hubs_sorted],
        "n_sources": n_s,
        "components": comp_report,
    })
    return {"n_hubs": len(hubs_sorted), "components": {m: comp_report[m]["n_components"] for m in MODES},
            "cost_median_days": round(float(np.median(cost_d)), 2)}
