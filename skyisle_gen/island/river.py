"""5.3b 河道成形：给 D8 汇流线一个真正的河道——宽、深、下切的河床、两岸的河谷与漫滩、河口跌下崖缘的瀑布。

水力几何（Leopold–Maddock）：年均流量 Q = 汇流 × 年降水 × 径流系数 / 一年秒数，河宽 w = a·Q^b、水深 d = c·Q^f，
再乘 width_scale / depth_scale（空岛的河按「看得出是河」来夸张；设 1 即回到真实比例——100 m 格上一条 10 m³/s 的河只有 16 m 宽）。

河床：中心线格 = min(实际高程, 填平面) − 水深 − 下切量（大河下切 incise_m × (汇流/主岛最大汇流)^0.5），按高度降序传给下游保证单调；
河口最多切进岸缘 notch_m（豁口）。河宽超过两格的，把格心离中心线 < w/2 的格并进河道。
河谷：离最近河床格距离 x 处的地面压到 河床 + 水深 + max(0, x − 漫滩半宽) × tan(谷坡)（只压低、不抬高）：
漫滩半宽 = floodplain_mult × w / 2，谷坡从上游小流量的 gorge_deg（峡谷）按 log Q 过渡到下游的 wide_deg（宽谷）。
季节性溪涧也浅切一道沟（stream_incise_m，谷坡按峡谷算，影响范围 stream_valley_cells 格），河床从汇入干流处按 stream_grade_max 往上爬（不悬在干流河谷上）。

只压低陆地、不改 island_id；湖面不动；每岛最高格不动（IS-summit）。纯函数：输入数组，返回新数组与摘要。
"""
from __future__ import annotations

import math

import numpy as np

from .grid import nearest_propagate


def discharge_m3s(acc_km2, precip_mm: float, runoff: float, year_s: float):
    """年均流量（m³/s）：km² × mm → m³ 再除以一年的秒数。"""
    return np.asarray(acc_km2, dtype=np.float64) * 1e6 * (precip_mm / 1000.0) * runoff / year_s


def hydraulic_geometry(Q, hc: dict):
    """河宽 / 水深（m）。Q ≤ 0 处为 0。"""
    Q = np.maximum(np.asarray(Q, dtype=np.float64), 0.0)
    w = float(hc["width_a"]) * Q ** float(hc["width_b"]) * float(hc["width_scale"])
    d = float(hc["depth_c"]) * Q ** float(hc["depth_f"]) * float(hc["depth_scale"])
    return w, d


def carve_channels(h, hf, mk, lake, ri, rj, Akm, river_lvl, stream, P_mm: float, rim: float, keel: float,
                   res_m: float, year_s: float, hc: dict, is_main: bool):
    """一座岛（局部切片）的河道下切。返回 (h_new, river_lvl_wide, width_m, depth_m, floodplain, info)。"""
    H, W = h.shape
    work = mk & ~lake
    center_r = work & (river_lvl > 0)
    center_s = work & (stream > 0) & ~center_r
    seed = center_r | center_s
    width = np.zeros((H, W))
    depth = np.zeros((H, W))
    info = {"rivers": [], "n_stream_falls": 0}
    if not seed.any():
        return h, river_lvl, width, depth, np.zeros((H, W), dtype=bool), info
    runoff = float(hc["runoff_coef"])
    Q = discharge_m3s(Akm, P_mm, runoff, year_s)
    w, d = hydraulic_geometry(Q, hc)
    sm = float(hc["stream_width_mult"])
    width = np.where(center_r, w, np.where(center_s, sm * w, 0.0))
    depth = np.where(center_r, d, np.where(center_s, sm * d, 0.0))
    amax = float(Akm[center_r].max()) if center_r.any() else 1.0
    incise = np.where(center_r, float(hc["incise_m"]) * np.sqrt(np.clip(Akm / amax, 0.0, 1.0)),
                      np.where(center_s, float(hc["stream_incise_m"]), 0.0))
    base = np.minimum(np.where(mk, h, np.inf), np.where(mk, hf, np.inf))
    bed = np.where(seed, base - depth - incise, np.nan)
    # 河床向下游单调：按填平面降序，把上游河床（减一个极小落差）传给下游
    floor = rim - float(hc["notch_m"])
    flat_hf = np.where(seed, hf, np.nan).ravel()
    idx = np.where(seed.ravel())[0]
    order = idx[np.argsort(-flat_hf[idx], kind="stable")]
    recv = np.where(ri.ravel() >= 0, ri.ravel() * W + rj.ravel(), -1)
    bl = np.maximum(np.nan_to_num(bed.ravel(), nan=np.inf), floor).tolist()
    rl = recv.tolist()
    sd = seed.ravel().tolist()
    step = res_m / 1000.0
    Lkm = [0.0] * (H * W)           # 到最远源头的河长（km），只沿常年河累计
    cr = center_r.ravel().tolist()
    for k in order.tolist():
        r = rl[k]
        if r >= 0 and sd[r]:
            if bl[r] > bl[k] - 0.01:
                bl[r] = max(floor, bl[k] - 0.01)
            if cr[k] and cr[r]:
                Lkm[r] = max(Lkm[r], Lkm[k] + step)
    # 支流接干流：溪涧河床从汇入处往上游按最大比降 stream_grade_max 爬升（下游先算），不在干流河谷里留一道悬着的沟脊
    rise = res_m * float(hc["stream_grade_max"])
    cs = center_s.ravel().tolist()
    for k in order[::-1].tolist():
        r = rl[k]
        if cs[k] and r >= 0 and sd[r] and bl[k] > bl[r] + rise:
            bl[k] = bl[r] + rise
    bed = np.where(seed, np.array(bl).reshape(H, W), np.nan)

    # 河谷剖面：最近河床格的 河床 / 水深 / 漫滩半宽 / 谷坡
    t = np.clip(np.log10(np.maximum(Q, 1e-6) / float(hc["q_gorge"])) / math.log10(float(hc["q_wide"]) / float(hc["q_gorge"])), 0.0, 1.0)
    side = np.where(center_r, (1 - t) * float(hc["gorge_deg"]) + t * float(hc["wide_deg"]), float(hc["gorge_deg"]))
    tan_side = np.tan(np.radians(side))
    fp_half = np.where(center_r, np.minimum(float(hc["floodplain_mult"]) * width / 2.0, float(hc["floodplain_max_m"]) / 2.0), 0.0)
    reach_cells = int(math.ceil(float(hc["valley_max_km"]) * 1000.0 / res_m)) if center_r.any() else int(hc["stream_valley_cells"])
    dist, src = nearest_propagate(seed, reach_cells, res_m, within=work)
    has = work & (src >= 0)
    s = np.where(has, src, 0).ravel()
    bed_s = bed.ravel()[s].reshape(H, W)
    dep_s = depth.ravel()[s].reshape(H, W)
    fp_s = fp_half.ravel()[s].reshape(H, W)
    tan_s = tan_side.ravel()[s].reshape(H, W)
    w_s = width.ravel()[s].reshape(H, W)
    isr_s = center_r.ravel()[s].reshape(H, W)
    lvl_s = river_lvl.ravel()[s].reshape(H, W)
    # 溪涧的影响只到 stream_valley_cells 格
    lim = np.where(isr_s, np.inf, int(hc["stream_valley_cells"]) * res_m)
    has &= dist <= lim
    valley = bed_s + dep_s + np.maximum(0.0, dist - fp_s) * tan_s
    h_new = np.where(has, np.minimum(h, valley), h)
    # 河道加宽：格心离中心线 < w/2 的格并入河道，高程取河床（w ≥ 2 格时河道占 3 格宽）
    wide = has & isr_s & ~center_r & (dist < w_s / 2.0)
    chan = center_r | wide
    h_new = np.where(chan, np.minimum(h_new, bed_s), h_new)
    h_new = np.where(seed, bed, h_new)
    lvl_out = np.where(wide, lvl_s, river_lvl).astype(river_lvl.dtype)
    width_out = np.where(chan, np.where(center_r, width, w_s), np.where(center_s, width, 0.0))
    depth_out = np.where(chan, np.where(center_r, depth, dep_s), np.where(center_s, depth, 0.0))
    floodplain = has & isr_s & ~chan & (dist <= fp_s + 0.5 * res_m) & (h_new <= bed_s + dep_s + 0.5)
    # 最高格不动（IS-summit）
    hm = np.where(mk, h, -np.inf)
    top = np.unravel_index(int(np.argmax(hm)), hm.shape)
    h_new[top] = h[top]
    h_new = np.where(mk, h_new, h)

    # 摘要：每个河口（常年河流向虚空处）一条河
    if center_r.any():
        mouths = np.where(center_r.ravel() & (recv < 0))[0]
        rivers = []
        for m_ in mouths.tolist():
            i, j = divmod(m_, W)
            rivers.append({"mouth_cell": [int(i), int(j)], "basin_km2": round(float(Akm[i, j]), 1),
                           "length_km": round(Lkm[m_] + step, 1), "discharge_m3s": round(float(Q[i, j]), 2),
                           "width_m": round(float(width[i, j]), 1), "depth_m": round(float(depth[i, j]), 2),
                           "level": int(river_lvl[i, j]), "waterfall_m": round(float(bed[i, j] - keel), 0),
                           "incision_m": round(float(base[i, j] - bed[i, j]), 1)})
        rivers.sort(key=lambda r: -r["basin_km2"])
        info["rivers"] = rivers
    info["n_stream_falls"] = int((center_s.ravel() & (recv < 0)).sum())
    info["max_cut_m"] = round(float(np.max(np.where(mk, h - h_new, 0.0))), 1)
    info["lines"] = trace_lines(seed, mk, recv, width, np.where(center_r, river_lvl, 0), H, W, Akm)
    return h_new, lvl_out, width_out, depth_out, floodplain, info


def trace_lines(seed, mk, recv, width, lvl, H: int, W: int, acc=None) -> list[list[list[float]]]:
    """河道中心线折线（给矢量渲染）：从每个源头顺流走到汇入已走过的格（汇流点）或出口为止。
    点 = [行 + 0.5, 列 + 0.5, 河宽 m, 级别（0 = 溪涧）, 汇流 km²]，局部切片坐标；出口多补一个点落在崖缘外半格（河跌下崖缘处）。
    汇流给调试台的水情用：干旱时溪涧从源头往下游一段段断流（汇流小于当日门槛的段），不是整条忽有忽无。"""
    sd = seed.ravel()
    rl = recv.tolist()
    idx = np.where(sd)[0]
    indeg = np.zeros(H * W, dtype=np.int32)
    tgt = recv[idx]
    ok = (tgt >= 0) & sd[np.maximum(tgt, 0)]
    np.add.at(indeg, tgt[ok], 1)
    wf, lf = width.ravel(), lvl.ravel()
    af = np.zeros(H * W) if acc is None else np.asarray(acc, dtype=np.float64).ravel()
    visited = np.zeros(H * W, dtype=bool)
    mkf = mk.ravel()
    lines = []
    for s in idx[indeg[idx] == 0].tolist():
        pts, k = [], s
        while True:
            i, j = divmod(k, W)
            pts.append([i + 0.5, j + 0.5, round(float(wf[k]), 1), int(lf[k]), round(float(af[k]), 2)])
            if visited[k] and len(pts) > 1:
                break
            visited[k] = True
            r = rl[k]
            if r < 0:
                # 出口：朝一个虚空邻格补半格
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    a, b = i + di, j + dj
                    if not (0 <= a < H and 0 <= b < W) or not mkf[a * W + b]:
                        pts.append([i + 0.5 + 0.5 * di, j + 0.5 + 0.5 * dj, pts[-1][2], pts[-1][3], pts[-1][4]])
                        break
                break
            if not sd[r]:
                break
            k = r
        if len(pts) >= 2:
            lines.append(pts)
    return lines
