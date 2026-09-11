"""把一次 run 的产物打包成浏览器可用的紧凑 JSON（大数组走 base64 定型数组）。"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

from .. import MODES, MODE_ZH
from ..culture import World
from ..polity import Polity
from ..stages.s03_islands import CLASS_NAMES, CLASS_ZH
from ..stages.s05_barriers import REGIONAL_ORDER


def _arr(a: np.ndarray, dtype: str) -> dict:
    a = np.ascontiguousarray(np.asarray(a).astype(dtype))
    return {"dtype": dtype, "shape": list(a.shape),
            "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def _q8(a: np.ndarray) -> dict:
    return _arr(np.clip(np.round(np.asarray(a, dtype=np.float64) * 255.0), 0, 255), "uint8")


def _qmm(a: np.ndarray, bits: int = 8) -> dict:
    """带 min/max 元数据的线性量化（u/v/temp 有负值，_q8 的 [0,1] 映射不适用）。
    前端还原：x = min + q/qmax · (max − min)。8 位：温度 ≈ 0.18 °C；
    u/v 用 16 位（≈ 0.0006 m/s）——航线顺逆风的方向因子不随风速缩放，无风带里 8 位的方向噪声会把 dir(c) 算错。"""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        hi = lo + 1.0
    qmax = (1 << bits) - 1
    d = _arr(np.clip(np.round((a - lo) / (hi - lo) * qmax), 0, qmax), "uint16" if bits > 8 else "uint8")
    d["min"], d["max"] = lo, hi
    return d


# 传给操作台的 1° 网格场（R2）。8 位场 64,800 点 → base64 ≈ 86 KB，u/v 16 位各 ≈ 173 KB；
# 六场合计 ≈ 690 KB，相对单文件导出的 12 MB（globe.gl 1.9 MB + 特征场）可接受。
# 体积紧张时在 [web].grid_fields 只留 u/v/temp。
GRID_FIELDS_DEFAULT = ["u", "v", "temp", "precip", "storm", "band",
                       "q", "uplift", "obstacle", "wake", "plate", "age"]   # 第三批：水汽 / 抬升 / 障碍 / 尾流 / 板块 / 岛龄
GRID_BITS = {"u": 16, "v": 16}


def build_grid(ctx) -> dict:
    """② wind.npz 与 ④ climate_grid.npz 的网格场（量化 + base64），供流线积分、标量填色与悬停读数。
    不重算任何东西；只读现有产物。风速在前端由 u/v 算出（不单独传）。"""
    try:
        wind = ctx.load_npz(4, "wind_local")   # ②b 扰动后的风与局部带号（第三批 3）
    except FileNotFoundError:
        wind = ctx.load_npz(2, "wind")
    clim = ctx.load_npz(4, "climate_grid")
    bands = ctx.load_json(2, "bands")
    lats, lons = wind["lats"], wind["lons"]
    src = {"u": wind["u"], "v": wind["v"], "band": wind["band"],
           "temp": clim["temp"], "precip": clim["precip"], "storm": clim["storm"],
           "storm_no_g": clim["storm_no_g"], "stability": clim["stability"], "window": clim["window"]}
    for k in ("q", "uplift", "conv", "eps"):
        if k in clim:
            src[k] = clim[k]
    for k in ("obstacle", "wake", "v_local", "u_bg", "v_bg", "land"):
        if k in wind:
            src[k] = wind[k]
    try:
        pl = ctx.load_npz(3, "plates")
        src["plate"] = pl["plate_id"]
        src["age"] = pl["age"]
        src["btype"] = pl["btype"]
        src["boundary_kernel"] = pl["boundary_kernel"]
    except FileNotFoundError:
        pass
    want = list(ctx.cfg.get("web", {}).get("grid_fields", GRID_FIELDS_DEFAULT))
    fields = {}
    for k in want:
        if k not in src:
            continue
        fields[k] = _arr(src[k], "uint8") if k in ("band", "plate", "btype") else _qmm(src[k], GRID_BITS.get(k, 8))
    band_local = None
    try:
        bl = ctx.load_npz(4, "band_local")
        band_local = {"keys": [str(x) for x in bl["keys"]], "lons": [round(float(x), 2) for x in bl["lons"]],
                      "edges": [[round(float(x), 2) for x in row] for row in bl["edges"]]}
    except FileNotFoundError:
        pass
    return {
        "nlat": int(lats.size), "nlon": int(lons.size),
        "lat0": float(lats[0]), "lon0": float(lons[0]),
        "res": float(lats[1] - lats[0]) if lats.size > 1 else 1.0,
        "band_names": bands["band_names"],
        "band_local": band_local,
        "fields": fields,
    }


def build_world(ctx) -> dict:
    w = World(ctx)
    isl, ce = w.islands, w.cand_edges
    pm, rt = w.perm, w.routes
    E = ce["src"].size
    clim = ctx.load_npz(4, "climate_islands")
    pre = ctx.load_npz(7, "prehist")
    planet = ctx.load_json(1, "planet")
    bands = ctx.load_json(2, "bands")
    centers = w.centers
    from ..ninegrid import RegionData
    rd = RegionData(ctx)
    region_names = [rd.name(r) for r in range(rd.n_regions)]
    f_reg = pm["f_regional"]
    f_max_idx = np.where(f_reg.max(axis=1) > 0.05, f_reg.argmax(axis=1), -1)
    traits = w.traits
    # ⑨ 政治层（第四批 R7）：旧 run 没有时前端按无政治层退化
    pol = Polity(ctx)
    if pol.available:
        m = pol.meta
        keep = ("id", "kind", "name", "capital", "capital_class", "circle", "regime", "n_nodes", "pop",
                "n_direct", "n_fiefs", "overlord", "vassals", "annexed_by", "annexed_years_ago", "stage", "is_suzerain")
        polity_meta = {"polities": [{k: x[k] for k in keep if k in x} for x in m["polities"]],
                       "n_states": m["n_states"], "n_fleets": m["n_fleets"], "n_tribes": m["n_tribes"],
                       "suzerain": m["suzerain"], "reformer": m["reformer"], "history": m["history"],
                       "fronts": m["fronts"], "openings": m["openings"], "stage_zh": m["stage_zh"],
                       "regime_zh": m["regime_zh"], "center_zh": m["center_zh"]}
        pa = pol.arr
        polity_arrays = {"pop": _arr(pa["pop"], "float32"), "state": _arr(pa["state"], "int32"),
                         "polity": _arr(pa["polity"], "int32"), "pkind": _arr(pa["kind"], "int8"),
                         "control": _q8(np.clip(pa["control"], 0, 1)), "realm": _arr(pa["realm"], "int32"),
                         "fief": _arr(pa["fief"], "int32")}
    else:
        polity_meta = None
        polity_arrays = {}
    return {
        "run_id": ctx.out_dir.name, "seed": ctx.seed,
        "modes": list(MODES), "mode_zh": MODE_ZH,
        "class_names": CLASS_NAMES, "class_zh": CLASS_ZH,
        "planet": planet, "bands": bands,
        "skeleton": ctx.cfg["skeleton"],
        "barriers": w.barriers["barriers"], "regional_order": REGIONAL_ORDER,
        "centers": centers["centers"], "origin_node": centers["origin_node"],
        "center_trunks": {k: v["path"] for k, v in centers["center_trunks"].items()},
        "hubs": w.hubs["hubs"],
        "region_names": region_names,
        "region_seeds": centers["region_seeds"],
        "polity": polity_meta,
        "slots": w.slots,
        "slot_meta": {s["id"]: {"zh": s["zh"], "phrase": s["phrase"], "category": s["category"]}
                      for s in ctx.cfg["slots"]["slot"]},
        "slot_mode": w.traits_meta["slot_mode"],
        "traits": [{"id": t["id"], "slot": t["slot"], "mode": t["mode"], "kind": t["kind"],
                    "resistance": t["resistance"], "d_half": t["d_half_days"],
                    "origin_node": t["origin_node"], "origin": t["origin"], "index": t["index"]}
                   for t in traits],
        "n_islands": int(isl["lat"].size), "n_edges": int(E),
        "arrays": {
            "lat": _arr(isl["lat"], "float32"), "lon": _arr(isl["lon"], "float32"),
            "cls": _arr(isl["cls"], "uint8"), "layered": _arr(isl["layered"], "uint8"),
            "area": _arr(isl["area_km2"], "float32"),
            "land_frac": _arr(isl["land_frac"], "float32"),
            "arable_frac": _arr(isl["arable_frac"], "float32"),
            "main_area": _arr(isl["main_area_km2"], "float32"),
            "age": _arr(isl["age"], "float32") if "age" in isl else _arr(np.zeros(isl["lat"].size), "float32"),
            "plate": _arr(isl["plate"], "int16") if "plate" in isl else _arr(np.zeros(isl["lat"].size), "int16"),
            "wall_m": _arr(isl["wall_m"], "float32"),
            "catch": _arr(clim["catch"], "float32"),
            "has_river": _arr(clim["has_river"], "uint8"),
            "river_size": _arr(clim["river_size"], "float32"),
            "region": _arr(w.regions["region"], "int32"),
            "precip": _q8(clim["precip"]), "stability": _q8(clim["stability"]),
            "arrival_yr": _arr(pre["arrival_yr"], "float32"),
            "lineage": _arr(pre["lineage"], "int32"),
            "suitability": _q8(w.regions["suitability"] / max(1e-9, float(w.regions["suitability"].max()))),
            "node_flow": _arr(rt["node_flow"], "float32"),
            "iso": _q8(w.iso["iso"]),
            "e_src": _arr(ce["src"], "int32"), "e_dst": _arr(ce["dst"], "int32"),
            "e_perm": _q8(pm["perm"]),
            "e_cost_ab": _arr(rt["cost"][:E], "float32"), "e_cost_ba": _arr(rt["cost"][E:], "float32"),
            "e_kind": _arr(ce["kind"], "uint8"),
            "e_gblock": _arr(pm["g_blocked"], "uint8"),
            "e_barrier": _arr(f_max_idx, "int8"),
            "e_flow": _arr(rt["flow"][:E] + rt["flow"][E:], "float32"),
            **polity_arrays,
        },
    }


def build_fields(ctx) -> dict:
    w = World(ctx)
    f = w.fields
    return {
        "arrays": {
            "share": _q8(f["share"]), "reach": _q8(f["reach"]),
            "adopt": _q8(f["adopt"]), "strength": _q8(f["strength"]),
            "conflict_by": _arr(f["conflict_by"], "int16"),
            "local_share": _q8(w.iso["local_share"]),
        },
        "slot_rows": w.slot_rows(),
    }


def build_texture(ctx, width: int = 2048) -> bytes:
    """行星表面贴图（等距圆柱 PNG）：木星式条带 + 风暴暗纹 + G 漩涡 + 密度微纹。"""
    import matplotlib
    matplotlib.use("Agg")
    try:
        wind = ctx.load_npz(4, "wind_local")   # 扰动后的风与局部带号（第三批 3）
    except FileNotFoundError:
        wind = ctx.load_npz(2, "wind")
    clim = ctx.load_npz(4, "climate_grid")
    dens = ctx.load_npz(3, "density_grid")["density"].astype(np.float64)
    from ..sphere import grid_axes
    lats, lons = grid_axes(float(ctx.cfg["shared"]["grid_res_deg"]))
    band = wind["band"].astype(np.int64)
    storm = clim["storm"].astype(np.float64)
    precip = clim["precip"].astype(np.float64)
    u = wind["u"].astype(np.float64)
    palette = {
        0: (0.72, 0.62, 0.55), 1: (0.86, 0.74, 0.55), 2: (0.93, 0.88, 0.78),
        3: (0.62, 0.66, 0.72), 4: (0.86, 0.88, 0.90),
        5: (0.86, 0.74, 0.55), 6: (0.93, 0.88, 0.78), 7: (0.62, 0.66, 0.72), 8: (0.86, 0.88, 0.90),
    }
    rgb = np.zeros(band.shape + (3,))
    for k, col in palette.items():
        rgb[band == k] = col
    # 条带内按风速做细纹
    streak = 0.06 * np.sin(np.radians(lats)[:, None] * 40.0) * (np.abs(u) / max(1e-9, np.abs(u).max()))
    rgb += streak[..., None]
    # 降水偏蓝绿、风暴变暗
    rgb[..., 2] += 0.12 * (precip - 0.5)
    rgb[..., 0] -= 0.05 * (precip - 0.5)
    rgb *= (1.0 - 0.55 * storm)[..., None]
    # G 漩涡：大红斑式的旋臂（纯视觉，用 G 位置与半径画）
    g = ctx.load_json(2, "bands")["G"]
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    dlat = LAT - g["lat"]
    dlon = ((LON - g["lon"] + 180.0) % 360.0) - 180.0
    dx = dlon * np.cos(np.radians(g["lat"]))
    r = np.hypot(dx, dlat) / float(g["radius_deg"])
    th = np.arctan2(dlat, dx)
    swirl = np.exp(-0.5 * (r / 1.6) ** 2) * (0.55 + 0.45 * np.cos(2.0 * th - 3.0 * r))
    eye = np.exp(-0.5 * (r / 0.45) ** 2)
    tint = np.array([0.72, 0.36, 0.30])
    rgb = rgb * (1 - 0.85 * swirl[..., None]) + tint[None, None, :] * (0.85 * swirl)[..., None]
    rgb = rgb * (1 - 0.6 * eye[..., None]) + np.array([0.95, 0.90, 0.80])[None, None, :] * (0.6 * eye)[..., None]
    # 密度微纹（岛群隐约可见）
    dn = np.log10(np.maximum(dens, 1e-4))
    dn = (dn - dn.min()) / max(1e-9, dn.max() - dn.min())
    rgb += 0.05 * (dn - 0.5)[..., None]
    rgb = np.clip(rgb, 0, 1)
    # 上采样到 width（经度）× width/2（纬度），行序：北在上
    img = (rgb[::-1] * 255).astype(np.uint8)
    from matplotlib.image import imsave
    buf = io.BytesIO()
    h = width // 2
    # 最近邻上采样
    yi = (np.arange(h) * img.shape[0] / h).astype(int)
    xi = (np.arange(width) * img.shape[1] / width).astype(int)
    imsave(buf, img[yi][:, xi], format="png")
    return buf.getvalue()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
