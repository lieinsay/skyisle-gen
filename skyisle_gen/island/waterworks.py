"""水利（P6，Zhouzhu PLAN-LAND L13；DESIGN-NOTES 四点三十七）：谷口的渠和塘、湿地排成圩田。水网是人挖的，放在聚落层。

- **圩田**（polder_plan，farmland.fill_cultivated 在好地先占的第一遍之后调）：好地占完了才去开湿地——
  湿地片（地表 = 湿地，8 连通、不跨岛，≥ polder_patch_min_km2）周围 polder_ring_km 内的平地（宜垦、坡 < polder_flat_deg）
  在第一遍里已经种了 ≥ polder_pressure_min 成（人口压力到了这片湿地边上），才排干围成圩田；没到的照旧是芦苇荡。
  格子按空岛湿地的大小缩：纵浦横塘 polder_spacing_km 一格（江南宋以后分圩，一圩几百亩到一两千亩；这里 600 m 一格 ≈ 540 亩），
  网格锚在湿地的出水口（片里汇流最大的格），纵浦南北、横塘东西；一格里的湿地够 polder_block_min_km2 才围成一圩，边角留作荡。
  圩田算已垦、**在额度之内**：第二遍好地先占时圩田的格先占（在大岛保底之后），额度照旧 = 行星层的可耕率、人口照旧 = ⑨，
  挤掉的是第一遍最外一层的地（开圩是因为没有更好的地了）。
- **谷口的渠**（build_waterworks）：渠首设在常年河（和汇水 ≥ works_stream_min_km2 的大溪涧：湿季引水，渠首配堰塘）上汇水 ≥ works_src_min_km2 的格，
  从高往低挑（水排到崖边就掉下去了：在出山口、平原上游就截住分走）。从渠首按「最省工的路」走出去（Dijkstra：一步的工 = 步长 × (1 + canal_slope_cost × 坡²)，
  横穿坡地费工 → 渠顺等高线走），渠水面按 canal_grad_m_per_km 往下降，地面最多高出渠水面 canal_cut_m（挖得穿），走出 canal_reach_km 为止；
  不过常年河、湖、崖缘、湿地，不跨岛。一块田有 canal_cmd_frac 以上的格在渠水面以下、走得到，就算这处渠首的灌区；新灌的田够 canal_min_cmd_km2 才修。
  渠 = 渠首到各块田的入口、再按 canal_lateral_km 的格点（以渠首为原点）铺进田里的路合成的树（最省工的路树，从渠首散开成扇）：
  从渠首出来的一段是干渠，分出去的是支渠；宽按所灌的田（每 km² canal_q_per_km2 m³/s，宽 5·Q^0.5 m）。
  渠首之间隔 ≥ canal_head_sep_km，试过的候选 canal_try_sep_km 内不再试。
- **塘**：每个村一口——灌区里的村（有渠或圩田）是村塘（接渠水），不在灌区、村在高山 / 山地 / 丘陵的是山塘（陂塘：在村上坡的沟里筑坝蓄雨水，
  按所灌的田定大小），其余是村塘（接雨水）；圩里各留一口圩塘（圩的 polder_pond_frac）；季节性的渠首配一口堰塘（蓄湿季的水）。
- **闸**：渠首闸（每处渠首）、圩闸（每圩一座，在圩堤上离出水口最近的地方）、排水闸（每片圩田的出水口，接排水渠到河）。

纯函数、只用确定的次序（稳定排序、按格号 / 田号破平局；浮点累加一律顺序加），C++ 的 core/src/island/waterworks.cpp 逐位同式。
"""
from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from .grid import distance_bands, label_by_island

LC_WET = 9
SQRT2 = math.sqrt(2.0)
STEPS = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2))
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
# 代码 → 中文（cpp 后端的 decode 与 python 后端共用）
CANAL_ZH = {"main": "干渠", "branch": "支渠", "drain": "排水渠", "ns": "纵浦", "ew": "横塘"}
POND_ZH = {"weir": "堰塘", "village": "村塘", "hill": "山塘", "polder": "圩塘"}
SLUICE_ZH = {"head": "渠首闸", "polder": "圩闸", "outlet": "排水闸"}
SOURCE_ZH = {"river": "常年河", "stream": "季节性溪涧"}
WORKS_NOTE = ("水利（P6）：渠首（heads）→ 渠（canals：干渠 / 支渠从渠首出来，纵浦 / 横塘是圩田的格子，排水渠从圩田的出水口接到河；"
              "pts = [行, 列] 群栅格坐标，格心 = 整数 + 0.5，圩田的纵浦横塘走在格边上 = 整数）；塘（ponds：村塘 / 山塘 / 圩塘 / 堰塘，area_m2 水面）；"
              "闸（sluices：渠首闸 / 圩闸 / 排水闸）；圩（polders：一格一圩，terrain.npz 的 polder_id 是圩号；dike_km = 圩堤长）与圩田片（polder_patches）。"
              "圩田算已垦、在额度之内（人口照旧 = ⑨）；没排干的湿地照旧是芦苇荡。")


def _cells_min(n_km2: float, cell_km2: float) -> int:
    return int(math.ceil(n_km2 / cell_km2 - 1e-9))


# ---------------------------------------------------------------- 圩田：哪几片湿地排干、怎么分圩（farmland 调）
def polder_plan(g: dict, wc: dict, wet: np.ndarray, take1: np.ndarray, n_avail: int) -> dict:
    """wet：聚落层动手之前的湿地（布尔 [H, W]）；take1：第一遍好地先占的已垦（扁平布尔）；n_avail：额度里还能给圩田的格数。
    返回 {"patches": [...], "cells": 圩田的格（扁平下标：按片、按圩、格号升序）, "polder_id": 圩号栅格, "wetland_cells": 湿地格数}。"""
    iid = g["island_id"]
    H, W = iid.shape
    res_km = float(g["res_km"])
    cell_km2 = res_km * res_km
    out = {"patches": [], "cells": np.zeros(0, dtype=np.int64), "polder_id": np.zeros((H, W), dtype=np.int32), "wetland_cells": int(wet.sum())}
    lab, n = label_by_island(wet, iid, 8)
    if n == 0:
        return out
    labf = lab.ravel()
    idx = np.flatnonzero(labf)
    idx = idx[np.argsort(labf[idx], kind="stable")]
    bnd = np.searchsorted(labf[idx], np.arange(1, n + 2))
    R = int(math.ceil(float(wc["polder_ring_km"]) / res_km - 1e-9))
    min_patch = _cells_min(float(wc["polder_patch_min_km2"]), cell_km2)
    min_block = _cells_min(float(wc["polder_block_min_km2"]), cell_km2)
    s = max(1, int(round(float(wc["polder_spacing_km"]) / res_km)))
    p_min = float(wc["polder_pressure_min"])
    flat_ok = (g["cultivable"] > 0) & (g["slope_deg"] < np.float32(wc["polder_flat_deg"]))
    tk = take1.reshape(H, W)
    fa = g["flowacc_km2"].ravel()
    cand = []
    for L in range(1, n + 1):
        cells = idx[bnd[L - 1]:bnd[L]]
        if cells.size < min_patch:
            continue
        ii, jj = cells // W, cells % W
        k = int(iid.ravel()[cells[0]])
        r0, r1 = max(0, int(ii.min()) - R - 1), min(H, int(ii.max()) + R + 2)
        c0, c1 = max(0, int(jj.min()) - R - 1), min(W, int(jj.max()) + R + 2)
        pm = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        pm[ii - r0, jj - c0] = True
        d = distance_bands(pm, R)
        F = (d >= 1) & (d <= R) & (iid[r0:r1, c0:c1] == k) & flat_ok[r0:r1, c0:c1]
        nF = int(F.sum())
        if nF == 0:
            continue
        p = int((F & tk[r0:r1, c0:c1]).sum()) / nF
        if p >= p_min:
            cand.append((p, L, cells))
    cand.sort(key=lambda x: (-x[0], x[1]))
    pid = out["polder_id"].ravel()
    used, npol, chunks = 0, 0, []
    for p, L, cells in cand:
        ii, jj = cells // W, cells % W
        o = int(cells[int(np.argmax(fa[cells]))])                  # 出水口：片里汇流最大的格（平局取格号小的）
        io, jo = divmod(o, W)
        bi, bj = (ii - io) // s, (jj - jo) // s
        srt = np.lexsort((cells, bj, bi))
        cs, bis, bjs = cells[srt], bi[srt], bj[srt]
        brk = np.flatnonzero((np.diff(bis) != 0) | (np.diff(bjs) != 0)) + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [cs.size]])
        blocks = [(int(bis[a]), int(bjs[a]), cs[a:b]) for a, b in zip(starts.tolist(), ends.tolist()) if b - a >= min_block]
        tot = sum(int(b[2].size) for b in blocks)
        if not blocks or used + tot > n_avail:
            continue
        used += tot
        ids = []
        for b in blocks:
            npol += 1
            pid[b[2]] = npol
            ids.append(npol)
            chunks.append(b[2])
        out["patches"].append({"label": L, "island": int(iid.ravel()[cells[0]]), "outlet": o, "anchor": (io, jo), "spacing": s,
                               "pressure": p, "wet_cells": int(cells.size), "cells": cells, "blocks": blocks, "polder_ids": ids})
    if chunks:
        out["cells"] = np.concatenate(chunks).astype(np.int64)
    return out


# ---------------------------------------------------------------- 渠、塘、闸（settle 在水设施之后调）
def build_waterworks(g: dict, wc: dict, fields: list[dict], fields_raster: np.ndarray, villages: list[dict], sraster: np.ndarray,
                     plan: dict, km) -> dict:
    iid = g["island_id"]
    H, W = iid.shape
    res_km = float(g["res_km"])
    res_m = res_km * 1000.0
    cell_km2 = res_km * res_km
    land = iid >= 0
    h = g["height"]
    hf = h.ravel()
    river = g["river"] > 0
    stream = g["stream"] > 0
    lake = g["lake"]
    cover = g["landcover"]
    acc = g["flowacc_km2"]
    accf = acc.ravel()
    iidf = iid.ravel()
    fidf = fields_raster.ravel()
    nf = len(fields)
    f_cells = [0] + [int(f["cells"]) for f in fields]
    polder_id = plan["polder_id"]
    pol_field = np.zeros(nf + 1, dtype=bool)
    pf = np.unique(fidf[polder_id.ravel() > 0])
    pol_field[pf[pf > 0]] = True
    zlev = np.where(land, h, 0.0) + g["river_depth_m"].astype(np.float64)
    zf = zlev.ravel()

    # ---------- 渠首与渠
    src = land & ~lake & (acc >= np.float32(wc["works_src_min_km2"])) & (river | (stream & (acc >= np.float32(wc["works_stream_min_km2"]))))
    cand = np.flatnonzero(src.ravel())
    cand = cand[np.argsort(-zf[cand], kind="stable")].tolist()
    passable = land & ~g["cliff"] & ~lake & ~river & (cover != LC_WET)
    grad_c = float(wc["canal_grad_m_per_km"]) * res_km
    cut = float(wc["canal_cut_m"])
    Rc = float(wc["canal_reach_km"]) / res_km
    Rw = int(math.ceil(Rc - 1e-9))
    a_cost = float(wc["canal_slope_cost"])
    t = float(wc["canal_head_sep_km"]) / res_km
    sep_h2 = t * t
    t = float(wc["canal_try_sep_km"]) / res_km
    sep_t2 = t * t
    min_cmd = _cells_min(float(wc["canal_min_cmd_km2"]), cell_km2)
    frac = float(wc["canal_cmd_frac"])
    qk = float(wc["canal_q_per_km2"])
    wmin = float(wc["canal_width_min_m"])
    s_lat = max(1, int(round(float(wc["canal_lateral_km"]) / res_km)))
    b0 = float(wc["canal_base_km"])
    bk = float(wc["canal_km_per_km2"])
    cmd = [False] * (nf + 1)
    cmd_np = np.zeros(nf + 1, dtype=bool)
    heads, head_cells, tried, canals = [], [], [], []
    n1_all = n2_all = 0
    for c in cand:
        ci, cj = divmod(c, W)
        if any((ci - a) * (ci - a) + (cj - b) * (cj - b) < sep_h2 for a, b in head_cells):
            continue
        if any((ci - a) * (ci - a) + (cj - b) * (cj - b) < sep_t2 for a, b in tried):
            continue
        k = int(iidf[c])
        z0 = float(zf[c])
        r0, r1 = max(0, ci - Rw), min(H, ci + Rw + 1)
        c0, c1 = max(0, cj - Rw), min(W, cj + Rw + 1)
        sub_f = fields_raster[r0:r1, c0:c1]
        di = (np.arange(r0, r1) - ci)[:, None]
        dj = (np.arange(c0, c1) - cj)[None, :]
        dist = np.sqrt((di * di + dj * dj).astype(np.float64))
        m = (sub_f > 0) & ~cmd_np[sub_f] & (iid[r0:r1, c0:c1] == k) & (dist <= Rc) & (h[r0:r1, c0:c1] <= z0 - grad_c * dist)
        if int(m.sum()) < frac * min_cmd:                        # 走得到的田格的上界都不够：不必试
            continue
        tried.append((ci, cj))
        ww = c1 - c0
        hw = h[r0:r1, c0:c1].ravel().tolist()
        pw = (passable[r0:r1, c0:c1] & (iid[r0:r1, c0:c1] == k)).ravel().tolist()
        fw = sub_f.ravel().tolist()
        nwin = (r1 - r0) * ww
        cost = [math.inf] * nwin
        ln = [0.0] * nwin
        par = [-1] * nwin
        s0 = (ci - r0) * ww + (cj - c0)
        cost[s0] = 0.0
        heap = [(0.0, c)]
        cnt, entry, ecost, rc = {}, {}, {}, []
        while heap:
            d, q = heapq.heappop(heap)
            qi, qj = divmod(q, W)
            lq = (qi - r0) * ww + (qj - c0)
            if d > cost[lq]:
                continue
            if d > Rc:
                break
            hq = z0
            if lq != s0:
                hq = hw[lq]
                F = fw[lq]
                if F > 0 and not cmd[F] and hq <= z0 - grad_c * ln[lq]:
                    rc.append((q, F))
                    if F in cnt:
                        cnt[F] += 1
                    else:
                        cnt[F] = 1
                        entry[F] = q
                        ecost[F] = d
            for sdi, sdj, L in STEPS:
                a, b = qi + sdi, qj + sdj
                if a < r0 or a >= r1 or b < c0 or b >= c1:
                    continue
                lp = (a - r0) * ww + (b - c0)
                if not pw[lp]:
                    continue
                nl = ln[lq] + L
                hp = hw[lp]
                if hp > z0 - grad_c * nl + cut:
                    continue
                sl = abs(hp - hq) / (L * res_m)
                nd = d + L * (1.0 + a_cost * sl * sl)
                if nd < cost[lp]:
                    cost[lp] = nd
                    ln[lp] = nl
                    par[lp] = q
                    heapq.heappush(heap, (nd, a * W + b))
        cand_f = [F for F in sorted(cnt) if cnt[F] >= frac * f_cells[F]]
        # 渠通到每块田的入口，再按 canal_lateral_km 的格点铺进田里（格点以渠首为原点）：渠网从渠首散开成扇
        lat = {F: [] for F in cand_f}
        for q, F in rc:
            if F in lat and q != entry[F]:
                qi, qj = divmod(q, W)
                if (qi - ci) % s_lat == 0 and (qj - cj) % s_lat == 0:
                    lat[F].append(q)
        # 修渠划不划算：按入口的工从近到远，一块田要从已经挖好的渠上新接出去的渠 ≤ canal_base_km + canal_km_per_km2 × 田的面积 才修
        intree = {c}
        newly = []
        for F in sorted(cand_f, key=lambda x: (ecost[x], x)):
            q = entry[F]
            n1 = n2 = 0
            while q not in intree:
                qi, qj = divmod(q, W)
                p = par[(qi - r0) * ww + (qj - c0)]
                if p // W != qi and p % W != qj:
                    n2 += 1
                else:
                    n1 += 1
                q = p
            if (n1 + n2 * SQRT2) * res_km > b0 + bk * (f_cells[F] * cell_km2):
                continue
            newly.append(F)
            for t in [entry[F]] + sorted(lat[F]):
                q = t
                while q not in intree:
                    intree.add(q)
                    qi, qj = divmod(q, W)
                    q = par[(qi - r0) * ww + (qj - c0)]
        newly.sort()
        tot = 0
        for F in newly:
            tot += f_cells[F]
        if tot < min_cmd:
            continue
        hid = len(heads) + 1
        head_cells.append((ci, cj))
        served, children = {}, {}
        for F in newly:
            cmd[F] = True
            cmd_np[F] = True
            tl = [entry[F]] + sorted(lat[F])
            base, rem = divmod(f_cells[F], len(tl))
            for ti, q in enumerate(tl):
                share = base + (1 if ti < rem else 0)
                prev = -1
                while True:
                    served[q] = served.get(q, 0) + share
                    if prev >= 0:
                        children.setdefault(q, set()).add(prev)
                    if q == c:
                        break
                    prev = q
                    qi, qj = divmod(q, W)
                    q = par[(qi - r0) * ww + (qj - c0)]
        n1_h = n2_h = 0
        n_seg = 0
        dq = deque([c])
        while dq:
            s_ = dq.popleft()
            for ch in sorted(children.get(s_, ())):
                pts = [s_, ch]
                cur = ch
                while len(children.get(cur, ())) == 1:
                    cur = next(iter(children[cur]))
                    pts.append(cur)
                n1 = n2 = 0
                for u, v in zip(pts[:-1], pts[1:]):
                    ui, uj = divmod(u, W)
                    vi, vj = divmod(v, W)
                    if ui != vi and uj != vj:
                        n2 += 1
                    else:
                        n1 += 1
                n1_h += n1
                n2_h += n2
                sv = served[pts[1]]
                q_m3s = qk * (sv * cell_km2)
                canals.append({"kind": CANAL_ZH["main" if s_ == c else "branch"], "head": hid, "island": k,
                               "pts": [[float(p // W) + 0.5, float(p % W) + 0.5] for p in pts],
                               "length_km": round((n1 + n2 * SQRT2) * res_km, 3), "served_km2": round(sv * cell_km2, 3),
                               "width_m": round(max(wmin, 5.0 * math.sqrt(q_m3s)), 1)})
                n_seg += 1
                if children.get(cur):
                    dq.append(cur)
        n1_all += n1_h
        n2_all += n2_h
        seasonal = not bool(river.ravel()[c])
        heads.append({"id": hid, "island": k, "cell": [ci, cj], "km": km(ci, cj), "source": SOURCE_ZH["stream" if seasonal else "river"],
                      "seasonal": seasonal, "level_m": round(z0, 1), "basin_km2": round(float(accf[c]), 1), "fields": newly,
                      "served_km2": round(tot * cell_km2, 3), "canal_km": round((n1_h + n2_h * SQRT2) * res_km, 3), "n_canals": n_seg})

    # ---------- 圩田：纵浦横塘、排水渠、圩堤、圩塘与闸
    polders, patches, pol_canals, sluices_p, sluices_o, ponds_p = [], [], [], [], [], []
    pidf = polder_id.ravel()
    recv_i, recv_j = g["recv_i"], g["recv_j"]
    drain_max = int(wc["polder_drain_max_cells"])
    pond_frac = float(wc["polder_pond_frac"])
    paddy_mm = np.float32(wc["polder_paddy_mm"])
    rain = g["rain_mm"]
    ns_cells = ew_cells = drain_n1 = drain_n2 = dike_edges = 0
    for P in plan["patches"]:
        io, jo = P["anchor"]
        s = int(P["spacing"])
        o = int(P["outlet"])
        oi, oj = divmod(o, W)
        k = int(P["island"])
        drained = 0
        for b_i, b_j, bc in P["blocks"]:
            pid_ = int(pidf[int(bc[0])])
            n = int(bc.size)
            drained += n
            ii, jj = bc // W, bc % W
            edges = 0
            best_s, best_d = -1, 0
            for q in bc.tolist():
                qi, qj = divmod(q, W)
                bnd = False
                for sdi, sdj in N4:
                    a, b = qi + sdi, qj + sdj
                    if a < 0 or a >= H or b < 0 or b >= W or pidf[a * W + b] != pid_:
                        edges += 1
                        bnd = True
                if bnd:
                    dd = (qi - oi) * (qi - oi) + (qj - oj) * (qj - oj)
                    if best_s < 0 or dd < best_d:
                        best_s, best_d = q, dd
            dike_edges += edges
            pond = int(bc[int(np.argmax(accf[bc]))])               # 圩塘：圩里最低洼、汇水最多的一格
            pi_, pj_ = divmod(pond, W)
            si_, sj_ = divmod(best_s, W)
            ponds_p.append({"kind": POND_ZH["polder"], "island": k, "cell": [pi_, pj_], "km": km(pi_, pj_), "polder": pid_,
                            "area_m2": int(round(n * cell_km2 * 1e6 * pond_frac)), "elev_m": round(float(hf[pond]), 0)})
            sluices_p.append({"kind": SLUICE_ZH["polder"], "island": k, "cell": [si_, sj_], "km": km(si_, sj_), "polder": pid_})
            r0_, r1_ = int(ii.min()), int(ii.max()) + 1
            c0_, c1_ = int(jj.min()), int(jj.max()) + 1
            x0, y0 = km(r0_ - 0.5, c0_ - 0.5)
            x1, y1 = km(r1_ - 0.5, c1_ - 0.5)
            polders.append({"id": pid_, "patch": len(patches) + 1, "island": k, "block": [b_i, b_j], "cells": n, "area_km2": round(n * cell_km2, 3),
                            "cells_bbox": [r0_, c0_, r1_, c1_], "km_bbox": [x0, y1, x1, y0], "dike_km": round(edges * res_km, 3),
                            "paddy": bool(rain[pi_, pj_] >= paddy_mm)})
        # 排水渠：出水口顺 D8 往下游接到河 / 溪涧 / 湖（至多 polder_drain_max_cells 格）
        pts = [o]
        q = o
        for _ in range(drain_max):
            qi, qj = divmod(q, W)
            a, b = int(recv_i[qi, qj]), int(recv_j[qi, qj])
            if a < 0 or b < 0 or (a == qi and b == qj):
                break
            q = a * W + b
            pts.append(q)
            if iidf[q] != k or river.ravel()[q] or stream.ravel()[q] or lake.ravel()[q]:
                break
        n1 = n2 = 0
        for u, v in zip(pts[:-1], pts[1:]):
            if u // W != v // W and u % W != v % W:
                n2 += 1
            else:
                n1 += 1
        drain_n1 += n1
        drain_n2 += n2
        if len(pts) >= 2:
            pol_canals.append({"kind": CANAL_ZH["drain"], "patch": len(patches) + 1, "island": k,
                               "pts": [[float(p // W) + 0.5, float(p % W) + 0.5] for p in pts], "length_km": round((n1 + n2 * SQRT2) * res_km, 3)})
        sluices_o.append({"kind": SLUICE_ZH["outlet"], "island": k, "cell": [oi, oj], "km": km(oi, oj), "patch": len(patches) + 1})
        # 纵浦横塘：格线（行边 io + s·m、列边 jo + s·m）穿过这片圩田的段
        dm = np.zeros(H * W, dtype=bool)
        for _b_i, _b_j, bc in P["blocks"]:
            dm[bc] = True
        dm = dm.reshape(H, W)
        ci_, cj_ = np.nonzero(dm)
        rmin, rmax, cmin, cmax = int(ci_.min()), int(ci_.max()), int(cj_.min()), int(cj_.max())
        e = io + s * ((rmin - io + s - 1) // s)
        while e <= rmax + 1:
            above = dm[e - 1, cmin:cmax + 1] if e - 1 >= 0 else np.zeros(cmax - cmin + 1, dtype=bool)
            below = dm[e, cmin:cmax + 1] if e < H else np.zeros(cmax - cmin + 1, dtype=bool)
            for a0, a1 in _runs((above | below).tolist()):
                ew_cells += a1 - a0
                pol_canals.append({"kind": CANAL_ZH["ew"], "patch": len(patches) + 1, "island": k,
                                   "pts": [[float(e), float(cmin + a0)], [float(e), float(cmin + a1)]], "length_km": round((a1 - a0) * res_km, 3)})
            e += s
        e = jo + s * ((cmin - jo + s - 1) // s)
        while e <= cmax + 1:
            left = dm[rmin:rmax + 1, e - 1] if e - 1 >= 0 else np.zeros(rmax - rmin + 1, dtype=bool)
            right = dm[rmin:rmax + 1, e] if e < W else np.zeros(rmax - rmin + 1, dtype=bool)
            for a0, a1 in _runs((left | right).tolist()):
                ns_cells += a1 - a0
                pol_canals.append({"kind": CANAL_ZH["ns"], "patch": len(patches) + 1, "island": k,
                                   "pts": [[float(rmin + a0), float(e)], [float(rmin + a1), float(e)]], "length_km": round((a1 - a0) * res_km, 3)})
            e += s
        patches.append({"id": len(patches) + 1, "island": k, "outlet_cell": [oi, oj], "km": km(oi, oj), "pressure": round(float(P["pressure"]), 3),
                        "wetland_km2": round(P["wet_cells"] * cell_km2, 3), "polder_km2": round(drained * cell_km2, 3),
                        "polders": list(P["polder_ids"])})

    # ---------- 塘：堰塘（季节性渠首）、村塘 / 山塘（每村一口）、圩塘
    free = land & ~g["cliff"] & ~lake & ~river & (cover != LC_WET) & (g["cultivated"] == 0) & (g["fallow_years"] == 0) & (sraster == 0)
    reach = int(wc["pond_reach_cells"])
    near = int(wc["pond_near_cells"])
    zone = g["terrain_zone"]
    hill_ratio = float(wc["hill_pond_ratio"])
    hill_min = float(wc["hill_pond_min_m2"])
    m2_hh = float(wc["pond_m2_per_hh"])
    v_min = float(wc["pond_min_m2"])

    def pick(ci, cj, r, hmin):
        """(ci, cj) 周围 r 格（方窗）里能挖塘的格：汇流最大，平局离得近、格号小；hmin 给了就只要不低于它的格。"""
        best, bacc, bd = -1, 0.0, 0
        for a in range(max(0, ci - r), min(H, ci + r + 1)):
            for b in range(max(0, cj - r), min(W, cj + r + 1)):
                if not free[a, b] or iid[a, b] != iid[ci, cj]:
                    continue
                if hmin is not None and not (h[a, b] >= hmin):
                    continue
                av = float(acc[a, b])
                dd = (a - ci) * (a - ci) + (b - cj) * (b - cj)
                if best < 0 or av > bacc or (av == bacc and dd < bd):
                    best, bacc, bd = a * W + b, av, dd
        return best

    ponds = []

    def add_pond(q, kind, area, **ref):
        qi, qj = divmod(q, W)
        free[qi, qj] = False
        ponds.append({"kind": POND_ZH[kind], "island": int(iidf[q]), "cell": [qi, qj], "km": km(qi, qj), "area_m2": int(round(area)),
                      "elev_m": round(float(hf[q]), 0), **ref})

    for hd in heads:
        if not hd["seasonal"]:
            continue
        ci, cj = hd["cell"]
        q = pick(ci, cj, near, None)
        if q >= 0:
            add_pond(q, "weir", max(hill_min, hd["served_km2"] * 1e6 * hill_ratio), head=hd["id"])
    for v in villages:
        F = int(v["field"])
        f = fields[F - 1]
        vi, vj = v["cell"]
        if cmd[F] or pol_field[F]:
            kind = "village"
        elif 1 <= int(zone[vi, vj]) <= 3:              # 高山 / 山地 / 丘陵：陂塘
            kind = "hill"
        else:
            kind = "village"
        if kind == "hill":
            q = pick(vi, vj, reach, float(h[vi, vj]))
            if q < 0:
                q = pick(vi, vj, reach, None)
            area = max(hill_min, float(f["area_km2"]) * 1e6 * hill_ratio)
        else:
            q = pick(vi, vj, near, None)
            if q < 0:
                q = pick(vi, vj, reach, None)
            area = max(v_min, m2_hh * (int(v["households"]) + int(v.get("households_market", 0))))
        if q >= 0:
            add_pond(q, kind, area, village=v["id"])
    ponds.extend(ponds_p)
    for x in ponds:
        sraster[x["cell"][0], x["cell"][1]] = 15
    for i, x in enumerate(ponds):
        x["id"] = i + 1

    # ---------- 闸
    sluices = [{"kind": SLUICE_ZH["head"], "island": hd["island"], "cell": list(hd["cell"]), "km": hd["km"], "head": hd["id"]} for hd in heads]
    sluices += sluices_p + sluices_o
    for i, x in enumerate(sluices):
        x["id"] = i + 1
        a, b = x["cell"]
        if sraster[a, b] in (0, 1, 2):
            sraster[a, b] = 16
    canals += pol_canals
    for i, x in enumerate(canals):
        x["id"] = i + 1

    cmd_cells = 0
    for F in range(1, nf + 1):
        if cmd[F]:
            cmd_cells += f_cells[F]
    n_cult = int((g["cultivated"] > 0).sum())
    wet_cells = int(plan["wetland_cells"])
    pol_cells = int((polder_id > 0).sum())
    pk = {kk: 0 for kk in POND_ZH.values()}
    for x in ponds:
        pk[x["kind"]] += 1
    sk = {kk: 0 for kk in SLUICE_ZH.values()}
    for x in sluices:
        sk[x["kind"]] += 1
    summary = {"n_heads": len(heads), "n_heads_seasonal": sum(1 for x in heads if x["seasonal"]),
               "canal_km": round((n1_all + n2_all * SQRT2) * res_km, 3), "polder_canal_km": round((ns_cells + ew_cells) * res_km, 3),
               "drain_km": round((drain_n1 + drain_n2 * SQRT2) * res_km, 3),
               "commanded_km2": round(cmd_cells * cell_km2, 3), "commanded_share": round(cmd_cells / max(1, n_cult), 3),
               "n_ponds": len(ponds), "ponds": pk, "n_sluices": len(sluices), "sluices": sk,
               "wetland_km2": round(wet_cells * cell_km2, 3), "polder_km2": round(pol_cells * cell_km2, 3),
               "polder_share": round(pol_cells / wet_cells, 3) if wet_cells else 0.0,
               "n_polders": len(polders), "n_polder_patches": len(patches), "dike_km": round(dike_edges * res_km, 3)}
    return {"heads": heads, "canals": canals, "ponds": ponds, "sluices": sluices, "polders": polders, "polder_patches": patches,
            "summary": summary, "note": WORKS_NOTE}


def _runs(flags: list) -> list[tuple[int, int]]:
    """连续为真的段 [(起, 止)]（止不含）。"""
    out, a = [], -1
    for i, f in enumerate(flags):
        if f and a < 0:
            a = i
        elif not f and a >= 0:
            out.append((a, i))
            a = -1
    if a >= 0:
        out.append((a, len(flags)))
    return out
