"""⑤ 障碍识别：区域障碍（规则生成命名 A/B/C/D/F/G）+ 边局部因子 → 每边四模式通过率。

关键：区域障碍用穿越坐标 Φ_b(j) ∈ [0,1]，边指数 f_b(e) = |Φ_b(v) − Φ_b(u)|，
perm_m(e) = Π_b P_b[m]^f_b × Π 局部因子。
任一单调穿越 Σf = 1 → 乘积恰为矩阵值，与跳数、岛数、seed 无关（docs/02 §五 数据模型）。
障碍是选择性过滤器，不是墙（D34）。
由 C++ 核心（core/src/planet/stage5.cpp；行星计划 P6d，Python 参考版删于 2026-09-30，tag python-reference-final）算，perm.npz / barriers.json 与摘要照旧由这里写（_write）。
"""
from __future__ import annotations

import numpy as np

from .. import MODES
from ..skeleton import d_lat_range

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


def run(ctx):
    from ..engine import core, part, planet_config, put_part
    cc = core()
    B = cc.planet_stage5(cc.make_config(planet_config(ctx.cfg)), part(ctx, 1), part(ctx, 2), part(ctx, 3), part(ctx, 4))
    put_part(ctx, 5, B)
    return _write(ctx, cc.barriers_arrays(B))


def _write(ctx, R: dict) -> dict:
    """写 perm.npz 与 barriers.json、出摘要（R 里的浮点是双精度原值，这里照旧转 float32）。"""
    cfg = ctx.cfg
    s5 = ctx.section(5)
    planet = ctx.load_json(1, "planet")
    bands = planet["bands"]
    g_info = ctx.load_json(2, "bands")["G"]
    perm, f, g_blocked = R["perm"], R["f"], R["g_blocked"]
    E = perm.shape[0]

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

    ctx.save_npz(5, "perm", perm=perm.astype(np.float64), perm_no_g=R["perm_no_g"].astype(np.float64),
                 f_regional=f.astype(np.float32),
                 g_blocked=g_blocked,
                 gap=R["gap"].astype(np.float32),
                 density_drop=R["density_drop"].astype(np.float32),
                 climb=R["climb"].astype(np.float32),
                 political=R["political"].astype(np.float32))
    ctx.save_json(5, "barriers", {"regional_order": REGIONAL_ORDER, "barriers": barriers_out,
                                  "modes": list(MODES)})
    removed = {m: int((perm[:, mi] == 0).sum()) for mi, m in enumerate(MODES)}
    return {"n_edges": E, "g_blocked": int(g_blocked.sum()), "edges_removed_per_mode": removed}
