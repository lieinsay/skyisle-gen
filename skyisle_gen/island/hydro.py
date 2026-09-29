"""5.3 水系、地表与可耕地：D8 流向 + 洼地填平 → 汇流 × 局地雨 = 径流 → 河（对齐 has_river / river_size）、溪涧、湖、集水盆地 →
地表分类（海拔温度、坡度、汇流、岛龄土层、迎风 / 背风、局地雨）→ 湿地（坡缓、离崖缘远、排不走，再乘雨）→ 可耕地按适宜度分位取到 arable_frac（误差 < 0.005）。

全群一张栅格；水文按岛分别做（每岛自成流域，虚空是出口）。行星层的相对降水按第八节 a 换算成 mm/年，
再按海拔与山脉尺度的迎风坡在群里重新分（局地雨，P4；群的总量不变），用于径流量、地表湿度与湿地。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import N8, binary_dilate, binary_erode, distance_bands, label_components, shift, slope_deg
from .terrain import accumulate, d8, priority_fill

LC_VOID, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE = range(12)


def precip_mm(p_rel: float, c: dict) -> float:
    """第八节 a：0 → precip_mm_min，1 → precip_mm_max，中间按 p^exp。"""
    lo, hi, e = float(c["precip_mm_min"]), float(c["precip_mm_max"]), float(c["precip_mm_exp"])
    return lo + (hi - lo) * float(np.clip(p_rel, 0.0, 1.0)) ** e


def river_area_for_q(q_m3s: float, P_mm: float, runoff: float, year_s: float) -> float:
    """年均流量 q（m³/s）对应的汇流面积 km²：A = q × 一年秒数 / (年降水 m × 径流系数)。river.discharge_m3s 的反函数。"""
    return q_m3s * year_s / max(1e-9, P_mm / 1000.0 * runoff) / 1e6


RAIN_Q = 16.0   # 雨量加权汇流的量子（1/16 mm）：权重取整后求和与次序无关，两个后端逐位相同


def local_rain(height: np.ndarray, land: np.ndarray, P_mm: float, rim_cell: np.ndarray, u: float, v: float, res_km: float, hc: dict) -> np.ndarray:
    """局地年降水（mm，P4，L11 的生成器那半）：本岛的起伏 × 山脉尺度的迎风坡，再按全群陆地的均值归一——群的雨总量是行星层给的，山只把它重新分。
    起伏 = 高出本岛岸缘多少（rim_cell：每格所在那座岛的岸缘；用户 09-29 定：地形雨是山逼着气流抬升才多下的，整座岛浮得高并不逼气流上升，
    只让它冷（气温照旧随高度降），不该让它更湿；旧口径「高程 − 主岛台面」让浮得高的小岛整座多雨）。
    迎风 G = 顺风方向的起伏升降（m/km），在 windward_smooth_km 见方的块均值上量（虚空按最近的陆地的起伏填：岛浮在空中、风从底下绕过去，
    岸崖不算迎风的山），双线性放大回来。两个系数都 0：处处 P_mm。C++ 版 hydro.cpp 同式。"""
    oro, wwg = float(hc.get("oro_rise_per_km", 0.0)), float(hc.get("windward_gain", 0.0))
    rain = np.where(land, P_mm, 0.0)
    if oro == 0.0 and wwg == 0.0:
        return rain
    from .grid import block_mean, upsample_bilinear
    H, W = height.shape
    f = max(1, int(round(float(hc["windward_smooth_km"]) / res_km)))
    # 迎风坡也按本岛的起伏（高出本岛岸缘多少）量：浮得高的岛不算山。虚空按最近的陆地填（3 块以内，取那座岛的起伏；再远按 0 = 岸缘）：
    # 浮在空中的岛下面有风绕过去，岸崖不算迎风的「山」
    from .grid import nearest_propagate
    _nd, ns = nearest_propagate(land, 3 * f, 1.0)
    rel = np.where(land, height - rim_cell, 0.0)
    hfill = np.where(land, rel, np.where(ns >= 0, rel.ravel()[np.maximum(ns, 0)], 0.0))
    hc_ = block_mean(hfill, f)
    Hc, Wc = hc_.shape
    step = f * res_km
    jp, jm = np.minimum(np.arange(Wc) + 1, Wc - 1), np.maximum(np.arange(Wc) - 1, 0)
    ip, im = np.minimum(np.arange(Hc) + 1, Hc - 1), np.maximum(np.arange(Hc) - 1, 0)
    gx = (hc_[:, jp] - hc_[:, jm]) / (np.maximum(jp - jm, 1) * step)[None, :]     # 向东升高为正（m/km）
    gy = (hc_[im, :] - hc_[ip, :]) / (np.maximum(ip - im, 1) * step)[:, None]     # 向北升高为正（行向下 = 南）
    sp = math.hypot(u, v)
    Gc = np.zeros((Hc, Wc)) if sp < 1e-6 else (gx * u + gy * v) / sp               # 顺风方向地势升高 = 迎风坡
    G = upsample_bilinear(Gc, f, H, W)
    r = (1.0 + oro * np.clip((height - rim_cell) / 1000.0, 0.0, float(hc["oro_rise_max_km"]))) \
        * (1.0 + wwg * np.clip(G / float(hc["windward_ref_m_per_km"]), -1.0, 1.0))
    rmean = np.mean(r[land])
    return np.where(land, P_mm * r / rmean, 0.0)


def local_precip_summary(rain_mm: np.ndarray, island_id: np.ndarray, hc: dict) -> dict:
    """island.json 的 hydro.local_precip（两个后端共用）：主岛局地雨的最少 / 最多 / 平均与分位（mm），全群平均（= 行星层的年降水）。"""
    on = float(hc.get("oro_rise_per_km", 0.0)) != 0.0 or float(hc.get("windward_gain", 0.0)) != 0.0
    m = rain_mm[island_id == 0].astype(np.float64)
    a = rain_mm[island_id >= 0].astype(np.float64)
    q = np.quantile(m, [0.1, 0.5, 0.9]) if m.size else [0.0, 0.0, 0.0]
    return {"on": on, "group_mean_mm": round(float(a.mean()), 0) if a.size else 0.0,
            "main_min_mm": round(float(m.min()), 0) if m.size else 0.0, "main_max_mm": round(float(m.max()), 0) if m.size else 0.0,
            "main_p10_p50_p90_mm": [round(float(x), 0) for x in q],
            "note": "局地年降水（terrain.npz 的 rain_mm）：海拔 × 山脉尺度的迎风坡，按全群陆地均值归一（群的雨总量是行星层给的）；河的流量、地表湿度、湿地按它算"}


def capture_rivers(h: np.ndarray, mk: np.ndarray, noise: np.ndarray, res_m: float, cell_km2: float, reach_km: float, hc: dict) -> tuple[np.ndarray, list]:
    """谷收拢（P4，L15；DESIGN-NOTES 四点三十五）：山在中间的岛，平行的小沟里有的切得快、往上游啃，把旁边沟的上游抢过来（河流袭夺）。
    一座岛（局部切片）先按现在的地形走一遍水（填洼、路由面 = 填平面 + noise、D8、汇流），按出口分盆地；
    汇流最大的 n 个出口是「大谷」（n = 面积 / capture_km2_per_valley，夹 [capture_min_n, capture_max_n]，出口汇流 ≥ capture_win_min_km2）。
    其余出口汇流 ≥ capture_loser_min_km2 的小沟（从大到小）：沿干流从河口往上游找（只看干流上游 capture_s_min 以上那段），
    头一个 reach_km 以内有大谷河道格（汇流 ≥ capture_chan_km2）比它低得多（路由面低于 z1 − capture_margin_m，
    z1 = 本格 − 2 × 到干流下一格的落差 − 1 m）的格就是袭夺点：从它到那个大谷格切一道直沟（地面压到从 z1 线性降到大谷格的路由面，
    只压低），小沟的上半截改流进大谷、下半截只剩崖边一小段。大谷收了之后不超过岛面积的 capture_max_frac（别太过）。
    返回 (新地形, 袭夺记录 [(小沟出口, 袭夺点, 大谷格, 抢来的汇流 km²)])。C++ 版 hydro.cpp 的 capture_rivers 同式。"""
    from .terrain import accumulate, d8, priority_fill
    Hh, Ww = mk.shape
    eps = float(hc["fill_eps_m"])
    hf = priority_fill(h, mk, eps=eps)
    hr = priority_fill(np.where(mk, hf + noise, np.nan), mk, eps=eps)
    ri, rj, _, _ = d8(hr, mk, res_m)
    A = accumulate(hr, mk, ri, rj) * cell_km2
    recv = np.where(ri >= 0, ri * Ww + rj, -1).ravel()
    hrf = hr.ravel()
    Af = A.ravel()
    idx = np.where(mk.ravel())[0]
    order = idx[np.argsort(hrf[idx], kind="stable")]
    rl = recv.tolist()
    lab = [-1] * (Hh * Ww)
    for k in order.tolist():
        r = rl[k]
        lab[k] = k if r < 0 else lab[r]
    total = float(mk.sum()) * cell_km2
    outs = idx[recv[idx] < 0]
    outs = outs[np.argsort(-Af[outs], kind="stable")].tolist()
    n_v = int(np.clip(round(total / float(hc["capture_km2_per_valley"])), int(hc["capture_min_n"]), int(hc["capture_max_n"])))
    wins = [o for o in outs[:n_v] if Af[o] >= float(hc["capture_win_min_km2"])]
    if not wins:
        return h, []
    budget = {o: float(hc["capture_max_frac"]) * total - float(Af[o]) for o in wins}
    is_win = np.zeros(Hh * Ww, dtype=bool)
    is_win[wins] = True
    labf = np.array(lab)
    okw = mk.ravel() & is_win[np.maximum(labf, 0)] & (labf >= 0) & (Af >= float(hc["capture_chan_km2"]))
    donors = [[] for _ in range(Hh * Ww)]
    for k in idx.tolist():
        r = rl[k]
        if r >= 0:
            donors[r].append(k)
    Al = Af.tolist()
    a_head = float(hc["capture_head_km2"])
    s_min = float(hc["capture_s_min"])
    margin = float(hc["capture_margin_m"])
    Rc = int(math.floor(reach_km * 1000.0 / res_m))
    R2 = Rc * Rc
    h = h.copy()
    mkf = mk.ravel()
    nz = noise.ravel()
    log = []
    for o in outs:
        if is_win[o] or Al[o] < float(hc["capture_loser_min_km2"]):
            continue
        path = [o]
        k = o
        while donors[k]:
            best = donors[k][0]
            for d in donors[k][1:]:
                if Al[d] > Al[best]:
                    best = d
            if Al[best] < a_head:
                break
            path.append(best)
            k = best
        L = len(path) - 1
        if L < 1:
            continue
        done = False
        for t in range(1, L + 1):
            if t / L < s_min:
                continue
            b = path[t]
            bi, bj = divmod(b, Ww)
            z1 = hrf[b] - 2.0 * (hrf[b] - hrf[path[t - 1]]) - 1.0
            i0, i1 = max(0, bi - Rc), min(Hh, bi + Rc + 1)
            j0, j1 = max(0, bj - Rc), min(Ww, bj + Rc + 1)
            ii, jj = np.mgrid[i0:i1, j0:j1]
            q = (ii * Ww + jj).ravel()
            d2 = ((ii - bi) ** 2 + (jj - bj) ** 2).ravel()
            cand = okw[q] & (d2 <= R2) & (hrf[q] <= z1 - margin)
            if not cand.any():
                continue
            cq, cd = q[cand], d2[cand]
            sel = np.lexsort((cq, hrf[cq], cd))
            for w in cq[sel].tolist():
                ow = lab[w]
                if budget[ow] < Al[b]:
                    continue
                wi, wj = divmod(w, Ww)
                n = max(abs(wi - bi), abs(wj - bj))
                cells = []
                ok = True
                for s in range(1, n):
                    ci = bi + int(np.rint(s * (wi - bi) / n))
                    cj = bj + int(np.rint(s * (wj - bj) / n))
                    c_ = ci * Ww + cj
                    if not mkf[c_]:
                        ok = False
                        break
                    cells.append(c_)
                if not ok:
                    continue
                zw = hrf[w]
                hv = h.ravel()
                for s, c_ in enumerate(cells, start=1):
                    z = z1 + (zw - z1) * (s - 1) / max(1, n - 1)
                    zz = z - nz[c_]
                    if zz < hv[c_]:
                        hv[c_] = zz
                budget[ow] -= Al[b]
                log.append((int(o), int(b), int(w), round(float(Al[b]), 2)))
                done = True
                break
            if done:
                break
    return h, log


def _wind_exposure(H: int, W: int, u: float, v: float, slope_dir_x, slope_dir_y):
    """迎风 = 坡面法线朝向来风（风向量 (u, v) 指向下游）。返回 [-1, 1]，正 = 迎风。"""
    sp = math.hypot(u, v)
    if sp < 1e-6:
        return np.zeros((H, W))
    ux, uy = -u / sp, -v / sp     # 指向上风的单位向量
    return np.clip(slope_dir_x * ux + slope_dir_y * uy, -1.0, 1.0)


def build_hydro(ctx, node: int, c: dict, g: dict, log=print) -> None:
    from .engine import backend
    if backend(ctx) == "cpp":
        from .engine import build_hydro_cpp
        return build_hydro_cpp(ctx, node, c, g, log=log)
    hc = c["hydro"]
    lc = c["landcover"]
    inp = g["inp"]
    res_km = g["res_km"]
    res_m = res_km * 1000.0
    cell_km2 = res_km * res_km
    height = g["height"]
    island_id = g["island_id"]
    H, W = height.shape
    land = island_id >= 0
    P_mm = precip_mm(inp["precip"], c["climate"])
    n_isl = len(g["json"]["islands"])

    # 风：④ 扰动后的风在群心处的值（迎风 / 背风打分）
    wl = ctx.load_npz(4, "wind_local")
    from ..sphere import grid_interp
    u = float(grid_interp(wl["u"], wl["lats"], wl["lons"], inp["lat"], inp["lon"]))
    v = float(grid_interp(wl["v"], wl["lats"], wl["lons"], inp["lat"], inp["lon"]))

    filled = np.full((H, W), np.nan)
    acc_km2 = np.zeros((H, W))
    lake = np.zeros((H, W), dtype=bool)
    river = np.zeros((H, W), dtype=np.uint8)      # 0 无 / 1 小河 / 2 中河 / 3 大河（常年）
    stream = np.zeros((H, W), dtype=np.uint8)     # 1 = 季节性溪涧
    width_m = np.zeros((H, W), dtype=np.float32)   # 河宽 / 水深（m）：常年河与溪涧（溪涧只在湿季有水）
    depth_m = np.zeros((H, W), dtype=np.float32)
    floodplain = np.zeros((H, W), dtype=bool)
    cut_m = np.zeros((H, W), dtype=np.float32)       # 河道 / 河谷下切了多少米（资源层找峡谷壁用）
    recv_i = np.full((H, W), -1, dtype=np.int32)      # 群栅格上的 D8 下游格（−1 = 出口或无）与路由面：资源层按上游累积（砂金）用
    recv_j = np.full((H, W), -1, dtype=np.int32)
    route_h = np.full((H, W), np.nan)
    basin_info = {}
    rivers_info = []
    river_lines = []                                 # 河道中心线折线（矢量渲染用，rivers.json）
    n_falls = 0
    max_cut = 0.0
    river_thr_km2 = None
    cal = inp["planet"].get("calendar", {})
    year_s = float(cal.get("year_days_solar", 336.0)) * float(cal.get("solar_day_hr", 24.0)) * 3600.0
    from .river import carve_channels
    # 汇流路由面：填平面 + 弯曲噪声 + 朝岸缘的微倾（只用来定流向；湖与抬洼仍按原填平面）。
    # 否则 D8 在平缓面上走成网格直线，岸缘那圈被夹平的台面上河会贴着崖边平行跑
    from . import _rng
    from .grid import FractalNoise
    # 局地雨（P4）：水系改高程之前按拟合出来的地形算（C++ 同）
    rim_cell = np.zeros((H, W))
    for k_, J_ in enumerate(g["json"]["islands"]):
        rim_cell[island_id == k_] = float(J_["rim_m"])
    rain = local_rain(height, land, P_mm, rim_cell, u, v, res_km, hc)
    local_on = float(hc.get("oro_rise_per_km", 0.0)) != 0.0 or float(hc.get("windward_gain", 0.0)) != 0.0
    pit = np.zeros((H, W))
    x0_, y0_ = g["json"]["raster"]["origin_km"]
    Xk_ = x0_ + (np.arange(W) + 0.5) * res_km
    Yk_ = y0_ - (np.arange(H) + 0.5) * res_km
    meander = FractalNoise(_rng(ctx, node, "meander"), Xk_[0], Yk_[-1], Xk_[-1], Yk_[0], feature_km=float(hc["meander_km"]),
                           octaves=3, persistence=0.5).sample(*np.meshgrid(Xk_, Yk_)) * float(hc["meander_m"])
    edge_km = distance_bands(~land, int(math.ceil(float(hc["edge_tilt_km"]) / res_km))).astype(np.float64) * res_km
    tilt = float(hc["edge_tilt_m_per_km"]) * np.minimum(edge_km, float(hc["edge_tilt_km"]))
    for k, J in enumerate(g["json"]["islands"]):
        m = island_id == k
        if not m.any():
            continue
        rr, cc = np.where(m)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        sl = (slice(r0, r1), slice(c0, c1))
        mk = m[sl]
        h = np.where(mk, height[sl], np.nan)
        # 谷收拢（P4）：山在中间的岛（多核岛有自己的干流，不收），新岛 × capture_young
        reach = float(hc.get("capture_reach_km", 0.0)) * (float(hc.get("capture_young", 0.0)) if J["age_zh"] == "新岛" else 1.0)
        if reach > 0.0 and not J.get("cores") and float(mk.sum()) * cell_km2 >= float(hc["capture_min_island_km2"]):
            h, cap_log = capture_rivers(h, mk, np.where(mk, meander[sl] + tilt[sl], 0.0), res_m, cell_km2, reach, hc)
            height[sl] = np.where(mk, h, height[sl])
            J["captures"] = len(cap_log)
        hf = priority_fill(h, mk, eps=float(hc["fill_eps_m"]))
        # 湖：填平深度 ≥ lake_min_depth_m 且面积 ≥ lake_min_km2 的洼地里最大的几个（每岛 ≤ lake_max_per_island）；
        # 其余洼地按填平面抬起（最多低于填平面 pit_keep_m），不留一地小坑。湖默认少见（大岛偶有）
        depth = np.where(mk, hf - h, 0.0)
        pond = depth >= float(hc["lake_min_depth_m"])
        lk = np.zeros_like(mk)
        n_lakes = 0
        if pond.any():
            lab, nl = label_components(pond)
            if nl:
                cnt = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
                big = [int(i) + 1 for i in np.argsort(-cnt, kind="stable") if cnt[i] * cell_km2 >= float(hc["lake_min_km2"])]
                big = big[: int(hc["lake_max_per_island"] if k == 0 else hc["lake_max_per_islet"])]
                lk = np.isin(lab, big) if big else lk
                n_lakes = len(big)
        raised = mk & ~lk & (hf - h > float(hc["pit_keep_m"]))
        h = np.where(raised, hf - float(hc["pit_keep_m"]), h)
        height[sl][raised] = h[raised]
        # 洼（抬过之后还剩的、不是湖的）：湿地的「排不走」（P4，L14）
        pit[sl] = np.where(mk & ~lk, np.maximum(hf - h, 0.0), pit[sl])
        hr = priority_fill(np.where(mk, hf + meander[sl] + tilt[sl], np.nan), mk, eps=float(hc["fill_eps_m"]))
        ri, rj, slope, _ = d8(hr, mk, res_m)
        A = accumulate(hr, mk, ri, rj)
        ok_r = mk & (ri >= 0)
        recv_i[sl] = np.where(ok_r, ri + int(r0), recv_i[sl])
        recv_j[sl] = np.where(ok_r, rj + int(c0), recv_j[sl])
        route_h[sl] = np.where(mk, hr, route_h[sl])
        Akm = A * cell_km2
        # 局地雨：年均流量按上游各格的雨（取整到 1/RAIN_Q mm 的权重求和，次序无关）
        Qloc = None
        if local_on:
            Aw = accumulate(hr, mk, ri, rj, weight=np.where(mk, np.rint(rain[sl] * RAIN_Q), 0.0))
            Qloc = np.where(mk, Aw * (cell_km2 / RAIN_Q) * 1000.0 * float(hc["runoff_coef"]) / year_s, 0.0)
        filled[sl][mk] = hf[mk]
        acc_km2[sl][mk] = Akm[mk]
        lake[sl] |= lk
        # 湖面：高程抬到填平面（水面平），保持 height 的 NaN 语义
        height[sl][lk] = hf[lk]
        # 河的阈值：主岛按 has_river 定；小岛一律只有溪涧
        q = Akm * P_mm                          # km² × mm ≈ 千 m³/年 的量级（径流代理）
        thr_stream = float(hc["stream_min_km2"]) * P_mm
        if k == 0:
            if inp["has_river"]:
                # 常年河的阈值按流量：年均流量 ≥ river_min_q_m3s 才算河（够宽够深、旱季不断流），换算成汇流面积——
                # 干岛要大得多的集雨面，湿岛小流域就够；但至少让主岛最大汇流的 river_reach_frac 成河（has_river 是行星层给的）
                thr_river = min(river_area_for_q(float(hc["river_min_q_m3s"]), P_mm, float(hc["runoff_coef"]), year_s),
                                float(hc["river_reach_frac"]) * float(Akm.max()))
                river_thr_km2 = thr_river
                per = Akm >= thr_river
                if Qloc is not None:
                    # 局地雨：常年河直接按流量判（≥ river_min_q_m3s，或主岛最大流量的 river_reach_frac）
                    qthr = min(float(hc["river_min_q_m3s"]), float(hc["river_reach_frac"]) * float(Qloc[mk].max()))
                    per = mk & (Qloc >= qthr)
                # 分级：按 river_size（= 主岛面积 × 降水，行星层）定最大河的级别
                rs = inp["river_size"]
                top = 3 if rs >= float(hc["river_big_size"]) else (2 if rs >= float(hc["river_mid_size"]) else 1)
                amax = float(Akm.max())
                lvl = np.zeros_like(river[sl])
                lvl[per] = 1
                if top >= 2:
                    lvl[per & (Akm >= 0.3 * amax)] = 2
                if top >= 3:
                    lvl[per & (Akm >= 0.6 * amax)] = 3
                river[sl] = np.where(mk, lvl, river[sl]).astype(np.uint8)
                stream[sl] = np.where(mk & (Akm >= float(hc["stream_min_km2"])) & ~per, 1, stream[sl]).astype(np.uint8)
            else:
                stream[sl] = np.where(mk & (Akm >= float(hc["stream_min_km2"])), 1, stream[sl]).astype(np.uint8)
            # 集水盆地：主岛按出口分水岭；出口 = 流向虚空的格；沿岸线把出口聚成段（相邻 basin_merge_cells 内的出口算同一盆地）
            basin_info = _basins(hr, mk, ri, rj, Akm, cell_km2, hc)
            basin_info["mouths"] = [[m_[0] + int(r0), m_[1] + int(c0), m_[2]] for m_ in basin_info.get("mouths", [])]   # 转成群栅格坐标
        else:
            stream[sl] = np.where(mk & (Akm >= float(hc["stream_min_km2"])), 1, stream[sl]).astype(np.uint8)
        # 河道成形（river.py）：河宽 / 水深、下切的河床、河谷与漫滩、河口瀑布
        h_now = np.where(mk, height[sl], np.nan)
        h_new, lvl_w, wd, dp, fp, rinfo = carve_channels(
            h_now, hr, mk, lk, ri, rj, Akm, np.where(mk, river[sl], 0).astype(np.uint8), np.where(mk, stream[sl], 0),
            P_mm, float(J["rim_m"]), float(J["keel_m"]), res_m, year_s, hc, k == 0, Qloc)
        cut_m[sl] = np.where(mk, np.nan_to_num(h_now - h_new), cut_m[sl])
        height[sl] = np.where(mk, h_new, height[sl])
        river[sl] = np.where(mk, lvl_w, river[sl]).astype(np.uint8)
        stream[sl] = np.where(mk & (lvl_w > 0), 0, stream[sl]).astype(np.uint8)
        width_m[sl] = np.where(mk, wd, width_m[sl])
        depth_m[sl] = np.where(mk, dp, depth_m[sl])
        floodplain[sl] |= fp & mk
        n_falls += rinfo["n_stream_falls"] + len(rinfo["rivers"])
        max_cut = max(max_cut, rinfo.get("max_cut_m", 0.0))
        for L in rinfo.get("lines", []):
            river_lines.append({"island": k, "pts": [[round(q[0] + r0, 2), round(q[1] + c0, 2), q[2], q[3], q[4]] for q in L]})
        if k == 0:
            rivers_info = rinfo["rivers"]
            for rv_ in rivers_info:        # 切片坐标 → 群栅格坐标（调试台「飞到河口」、资源层的瀑布后洞都按群栅格读）
                rv_["mouth_cell"] = [rv_["mouth_cell"][0] + int(r0), rv_["mouth_cell"][1] + int(c0)]
        J["n_lakes"] = n_lakes
        J["lake_km2"] = round(float(lk.sum()) * cell_km2, 3)
        J["max_flowacc_km2"] = round(float(Akm.max()), 2)
        J["has_perennial_river"] = bool((river[sl][mk] > 0).any())
        J["has_stream"] = bool((stream[sl][mk] > 0).any())
    river[lake] = 0
    stream[lake] = 0
    # 台面校正：河道下切 / 河谷压低了主岛的一圈格子，陆地中位比 height_m 低几米（100 m 栅格约 4 m、300 m 约 10 m）——主岛整体抬回。
    # 平移不改坡度、河床单调与湖面；岸缘 / 峰 / 崖高同步（后面的地表、资源、天气都读它们）
    m0 = island_id == 0
    dz = float(inp["height_m"]) - float(np.nanmedian(height[m0]))
    height[m0] += dz
    filled[m0] += dz
    J0 = g["json"]["islands"][0]
    J0["rim_m"] = round(J0["rim_m"] + dz, 1)
    J0["peak_m"] = round(J0["peak_m"] + dz, 1)
    J0["cliff_m"] = round(J0["rim_m"] - J0["keel_m"], 1)
    for co in J0.get("cores", []):
        co["peak_m"] = round(co["peak_m"] + dz, 1)

    # ---------- 地表分类 ----------
    slope = slope_deg(np.where(land, height, np.nan), land, res_m)
    T_sea = inp["temp_sea"]
    lapse = inp["lapse_c_per_km"]
    T = T_sea - lapse * np.where(land, height, 0.0) / 1000.0
    # 坡向 → 迎风性
    hh = np.where(land, height, np.nan)
    def nb(di, dj):
        a = shift(hh, di, dj, np.nan)
        return np.where(np.isnan(a), hh, a)
    gx = (nb(0, -1) - nb(0, 1)) / (2 * res_m)     # 东向坡降为正时地面向东下降 → 法线朝东
    gy = (nb(-1, 0) - nb(1, 0)) / (2 * res_m)     # 行向下 = 南：向北下降 → 法线朝北
    # 法线的水平方向 = 下坡方向 = -grad；下坡方向向量 (dx, dy) = (gx_eastdown, gy_northdown)
    gn = np.hypot(gx, gy)
    dx = np.where(gn > 1e-9, gx / np.maximum(gn, 1e-9), 0.0)
    dy = np.where(gn > 1e-9, gy / np.maximum(gn, 1e-9), 0.0)
    expo = _wind_exposure(H, W, u, v, dx, dy) * np.clip(slope / 15.0, 0.0, 1.0)
    # 土层厚度：老岛厚、平地厚、汇流处厚
    age_arr = np.zeros((H, W))
    for k, J in enumerate(g["json"]["islands"]):
        age_arr[island_id == k] = J["age"]
    soil = np.clip(0.35 + 0.5 * age_arr, 0, 1) * np.clip(1.0 - slope / float(lc["soil_slope_zero_deg"]), 0.0, 1.0) * (0.7 + 0.3 * np.clip(np.log1p(acc_km2) / 4.0, 0, 1))
    wet = np.clip(rain / 1500.0, 0.2, 2.0) * (1.0 + float(lc["aspect_wet_gain"]) * expo)   # 局地雨（关掉时 = P_mm）
    dist_water = distance_bands((river > 0) | lake, int(lc["water_near_cells"]))
    near_water = np.clip(1.0 - dist_water / (int(lc["water_near_cells"]) + 1.0), 0.0, 1.0) ** 0.5

    cover = np.full((H, W), LC_VOID, dtype=np.uint8)
    cover[land] = LC_GRASS
    cover[land & (T < float(lc["alpine_temp_c"]))] = LC_ALPINE
    cover[land & (T < float(lc["rock_temp_c"]))] = LC_ROCK
    cover[land & (slope >= float(lc["rock_slope_deg"]))] = LC_ROCK
    # 林地要够湿、土够厚、不太冷；薄土（山脊、陡坡、新岛）与半干处成灌丛；干处 / 极薄土成草坡
    warm = land & (T >= float(lc["forest_temp_min_c"])) & (slope < float(lc["rock_slope_deg"]))
    forest = warm & (wet >= float(lc["forest_wet_min"])) & (soil >= float(lc["forest_soil_min"]))
    shrub = land & (T >= float(lc["alpine_temp_c"])) & (slope < float(lc["rock_slope_deg"])) & ~forest & (wet >= float(lc["shrub_wet_min"])) & (soil >= 0.5 * float(lc["forest_soil_min"]))
    cover[shrub] = LC_SHRUB
    cover[forest] = LC_FOREST
    # 湿地（P4，L14）：坡缓、离崖缘远、汇来的水排不走（台面上的洼、平地），再乘局地雨——不拿年降水一刀切。C++ 版同式
    from .grid import window_extrema
    e1 = float(lc["wet_edge_far_km"])
    e0 = float(lc["wet_edge_near_km"])
    edge = np.clip((distance_bands(~land, int(math.ceil(e1 / res_km))) * res_km - e0) / (e1 - e0), 0.0, 1.0)
    snb, _slo = window_extrema(slope, int(lc["wet_flat_cells"]), land)        # 周围最陡的坡：成片的平地，不是一条条沟底
    tanb = np.tan(np.maximum(snb, float(lc["wet_slope_floor_deg"])) * (math.pi / 180.0))
    ahi, _alo = window_extrema(acc_km2, int(lc["wet_spread_cells"]), land)
    with np.errstate(invalid="ignore"):          # 虚空格 ahi = −inf、离崖缘 0：不用
        wet_idx = ahi * 1e6 / res_m * np.power(rain / float(lc["wet_ref_mm"]), float(lc["wet_rain_exp"])) * edge / tanb \
            * (1.0 + float(lc["wet_pit_gain"]) * np.clip(pit / float(lc["wet_pit_ref_m"]), 0.0, 1.0))
    wetland = land & (snb < float(lc["wet_slope_max_deg"])) & (wet_idx >= float(lc["wet_index_min"]))
    cover[wetland] = LC_WET
    cover[g["cliff"]] = LC_CLIFF
    # ---------- 可耕地：适宜度分位 ----------
    # 适宜度只用于排名（可耕率是行星层给定的约束）：各项都是软打分、处处 > 0，寒冷 / 陡峭的群也能取到该有的比例。
    # 再乘一层斑块噪声（特征 arable_patch_km），田块成团而不是沿等值线切出的带与方块
    from . import _rng
    from .grid import FractalNoise
    x0, y0 = g["json"]["raster"]["origin_km"]
    Xk = x0 + (np.arange(W) + 0.5) * res_km
    Yk = y0 - (np.arange(H) + 0.5) * res_km
    XX, YY = np.meshgrid(Xk, Yk)
    patch = FractalNoise(_rng(ctx, node, "arable"), Xk[0], Yk[-1], Xk[-1], Yk[0], feature_km=float(lc["arable_patch_km"]), octaves=3, persistence=0.5).sample(XX, YY)
    patch = 1.0 + float(lc["arable_patch_amp"]) * patch
    suit = patch * (0.02 + np.clip(1.0 - slope / float(lc["arable_slope_max_deg"]), 0.0, 1.0) ** 1.5
            * (0.1 + 0.9 * np.clip((T - float(lc["arable_temp_min_c"])) / 8.0, 0.0, 1.0))
            * (0.4 + 0.6 * soil) * (0.8 + 0.2 * near_water) * np.clip(wet / 0.6, 0.2, 1.2))
    suit = np.where(land & ~g["cliff"] & ~lake & (river == 0) & ~wetland, suit, -1.0)
    n_land = int(land.sum())
    n_arable = int(round(inp["arable_frac"] * n_land))
    arable = np.zeros((H, W), dtype=np.uint8)
    if n_arable > 0:
        flat = suit.ravel()
        order = np.argsort(-flat, kind="stable")[:n_arable]
        order = order[flat[order] > 0]
        a = np.zeros(H * W, dtype=bool)
        a[order] = True
        a = a.reshape(H, W)
        arable[a] = 1
        arable[a & (slope >= float(lc["terrace_slope_deg"]))] = 2
    # P5：上面按行星层额度取的是「上等地」（资源层照旧避开它）；宜垦另记，已垦在聚落层定（farmland.py）
    from .farmland import cultivable_land
    cultivable_land(g, lc, cover, arable, suit, slope, T, soil, wet, near_water)
    cover[arable == 1] = LC_ARABLE
    cover[arable == 2] = LC_TERRACE
    cover[river > 0] = LC_RIVER
    cover[lake] = LC_LAKE

    g["river_lines"] = river_lines
    g.update({"river_width_m": np.where(land, width_m, 0).astype(np.float32), "river_depth_m": np.where(land, depth_m, 0).astype(np.float32),
              "floodplain": floodplain & land & (river == 0) & ~lake, "cut_m": np.where(land, np.maximum(cut_m, 0), 0).astype(np.float32)})
    g.update({"flowacc_km2": acc_km2.astype(np.float32), "river": river, "stream": stream, "lake": lake,
              "landcover": cover, "arable": arable, "slope_deg": slope.astype(np.float32), "filled": filled,
              "recv_i": recv_i, "recv_j": recv_j, "route_h": route_h, "rain_mm": rain.astype(np.float32)})
    from .output import LANDCOVER_CLASSES
    share = {LANDCOVER_CLASSES[i]: round(float(((cover == i) & land).sum()) / max(1, n_land), 4) for i in range(1, 12)}
    J = g["json"]
    J["landcover"] = {"classes": LANDCOVER_CLASSES, "share": share,
                      "note": "landcover.png 索引色；调色板见 palette", "palette": None}
    from .output import LANDCOVER_PALETTE
    J["landcover"]["palette"] = LANDCOVER_PALETTE
    J["hydro"] = {"precip_mm": round(P_mm, 0), "river_threshold_km2": None if river_thr_km2 is None else round(river_thr_km2, 2),
                  "perennial_q_m3s": float(hc["river_min_q_m3s"]), "stream_min_km2": float(hc["stream_min_km2"]),
                  "runoff_coef": float(hc["runoff_coef"]),
                  "main_max_flowacc_km2": J["islands"][0]["max_flowacc_km2"],
                  "n_lakes": int(sum(i["n_lakes"] for i in J["islands"])),
                  "lake_km2": round(float(lake.sum()) * cell_km2, 3),
                  "main_basins": basin_info,
                  "rivers": rivers_info[:12],
                  "n_rivers": len(rivers_info),
                  "n_waterfalls": n_falls,
                  "river_km2": round(float((river > 0).sum()) * cell_km2, 3),
                  "floodplain_km2": round(float(g["floodplain"].sum()) * cell_km2, 3),
                  "max_incision_m": round(max_cut, 1),
                  "channel_note": "河宽 / 水深见 terrain.npz 的 river_width_m / river_depth_m（溪涧为湿季值）；height 在河道格是河床，水面 = 河床 + 水深；rivers[].waterfall_m = 河口跌下崖缘的落差",
                  "wind_ms": [round(u, 2), round(v, 2)],
                  "river_levels": {"1": "小河", "2": "中河", "3": "大河", "stream": "季节性溪涧（water.png 值 1）"}}
    J["hydro"]["local_precip"] = local_precip_summary(g["rain_mm"], island_id, hc)
    J["constraints"]["arable_frac"]["actual"] = round(float((arable > 0).sum()) / max(1, n_land), 4)
    # 河道下切 / 抬洼 / 湖面改过高程：台面（主岛陆地中位）与峰按最终高程重记（IS-surface 校的是这个）
    hm = height[island_id == 0]
    J["constraints"]["height_m"]["actual"] = round(float(np.nanmedian(hm)), 1)
    J["constraints"]["peak_m"]["actual"] = round(float(np.nanmax(hm)), 1)
    J["constraints"]["has_river"]["actual"] = bool(J["islands"][0]["has_perennial_river"])
    log(f"  水系：降水 {P_mm:.0f} mm，主岛河 {'有' if J['constraints']['has_river']['actual'] else '无'}（阈 {'—' if river_thr_km2 is None else f'{river_thr_km2:.1f} km²'}），湖 {J['hydro']['n_lakes']}，盆地 {basin_info.get('n_large', 0)} 大；"
        f"可耕 {J['constraints']['arable_frac']['actual']:.4f} / {inp['arable_frac']:.4f}")


def _basins(hf, mk, ri, rj, Akm, cell_km2, hc) -> dict:
    """主岛集水盆地：每格顺流到出口；只有汇流 ≥ basin_mouth_km2（或岛面积的 basin_mouth_frac）的出口算「河口」，
    其集水区为盆地；面积 ≥ basin_large_frac × 岛面积的为「大盆地」（R9 推论 2 的素材，≥ 2 个时标出）。"""
    H, W = hf.shape
    idx = np.where(mk.ravel())[0]
    recv = np.where(ri.ravel() >= 0, ri.ravel() * W + rj.ravel(), -1)
    order = idx[np.argsort(np.where(mk, hf, np.inf).ravel()[idx], kind="stable")]
    rl = recv.tolist()
    ol = [-1] * (H * W)
    for k in order.tolist():
        r = rl[k]
        ol[k] = k if r < 0 else ol[r]
    outlet = np.array(ol)
    total = float(mk.sum()) * cell_km2
    thr = max(float(hc["basin_mouth_km2"]), float(hc["basin_mouth_frac"]) * total)
    mouths = np.where(mk.ravel() & (Akm.ravel() >= thr) & (recv < 0))[0]
    sizes = np.array([float(Akm.ravel()[m]) for m in mouths])
    large = np.where(sizes >= float(hc["basin_large_frac"]) * total)[0]
    order = np.argsort(-sizes, kind="stable") if mouths.size else np.zeros(0, dtype=int)
    return {"n_basins": int(mouths.size), "n_large": int(large.size),
            "mouths": [[int(mouths[i] // W), int(mouths[i] % W), round(float(sizes[i]), 1)] for i in order[:12]],
            "basin_km2": [round(float(x), 1) for x in sorted(sizes.tolist(), reverse=True)[:8]],
            "large_km2": [round(float(sizes[i]), 1) for i in sorted(large, key=lambda i: -sizes[i])],
            "largest_frac": round(float(sizes.max() / total), 3) if mouths.size else 0.0}
