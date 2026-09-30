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
  不过常年河、湖、崖缘、湿地，不跨岛。P6 第一版：一块田有五成以上的格在渠水面以下、走得到，就整块算这处渠首的灌区；
  P6b 起按格算（管它的那个村的田里渠水面以下、走得到的格），够 canal_min_cmd_km2 才修。
  渠 = 渠首到各块田的入口、再按 canal_lateral_km 的格点（以渠首为原点）铺进田里的路合成的树（最省工的路树，从渠首散开成扇）：
  从渠首出来的一段是干渠，分出去的是支渠；宽按所灌的田（每 km² canal_q_per_km2 m³/s，宽 5·Q^0.5 m）。
  渠首之间隔 ≥ village_head_sep_km（P6b 起一村一堰；P6 第一版一处渠首灌几个村，隔 3 km），试过的候选 canal_try_sep_km 内不再试。
- **塘**：每个村一口——灌区里的村（有渠或圩田）是村塘（接渠水），不在灌区、村在高山 / 山地 / 丘陵的是山塘（陂塘：在村上坡的沟里筑坝蓄雨水，
  按所灌的田定大小），其余是村塘（接雨水）；圩里各留一口圩塘（圩的 polder_pond_frac）；季节性的渠首配一口堰塘（蓄湿季的水）。
- **闸**：渠首闸（每处渠首）、圩闸（每圩一座，在圩堤上离出水口最近的地方）、排水闸（每片圩田的出水口，接排水渠到河）。
- **有水利就有人维护**（P6b，Zhouzhu PLAN-LAND L30，DESIGN-NOTES 四点三十九）：每处渠首、渠、塘、闸、圩都记管它的村（village），
  都在那个村走得到的范围（manage_walk_km，直线）内。次序是「田（含圩田）→ 村址（圩田按 polder_village_blocks² 圩一组成田，走不到现有的村就在圩上落圩村，
  settle.polder_villages）→ 水利」：渠首一村一堰——只灌管它的村自己的旱地田，Dijkstra 只走那个村走得到的格（渠走不出去 = 截短），
  渠首之间隔 village_head_sep_km；圩、圩塘、圩闸归种那组圩田的村，纵浦横塘按两旁的圩分段归各自的村，排水闸与排水渠归出水口最近、走得到的村
  （排水渠走出去就截短，没有走得到的村就不修）。废村（P5 的 ruins）旁另挑没人管的废塘（每个废村一口）与废渠首、废渠（灌过它的撂荒田，一个废村至多一处）：
  abandoned = True、abandoned_years = 撤空了几年、ruin = 废村号、village = None。
- **原始地表与人工改造**（P6b，L31，landuse_layers）：landcover_natural = 没有人以前的地表（hydro 画田之前的 cover_natural，圩田那格原是湿地）；
  landuse = 人工改造码（LANDUSE_CLASSES）。

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
BIG_ZH = {"weir": "大堰"}                       # 四点四十：邑级大堰（都江堰级）
MAINT_ZH = {"yi": "邑"}                         # 管它的：邑（官府设堰官、岁修按用水的村出工）
WORKS_NOTE = ("水利（P6）：渠首（heads）→ 渠（canals：干渠 / 支渠从渠首出来，纵浦 / 横塘是圩田的格子，排水渠从圩田的出水口接到河；"
              "pts = [行, 列] 群栅格坐标，格心 = 整数 + 0.5，圩田的纵浦横塘走在格边上 = 整数）；塘（ponds：村塘 / 山塘 / 圩塘 / 堰塘，area_m2 水面）；"
              "闸（sluices：渠首闸 / 圩闸 / 排水闸）；圩（polders：一格一圩，terrain.npz 的 polder_id 是圩号；dike_km = 圩堤长）与圩田片（polder_patches）。"
              "圩田算已垦、在额度之内（人口照旧 = ⑨）；没排干的湿地照旧是芦苇荡。"
              "P6b：每处都记管它的村（village，都在 summary.manage_walk_km 以内；渠首一村一堰、只灌那个村的田）；"
              "abandoned = true 的是废村旁没人管的废渠首 / 废渠 / 废塘 / 废渠首闸（abandoned_years = 撤空了几年，ruin = 废村号，village = null），不进 summary 的数（另见 summary.abandoned）。"
              "水利分级（四点四十，用户 09-30 定）：岛群层只出邑级的——big_works 是大堰（都江堰级：常年河上的堰，干渠 / 支渠带 work = 堰号，"
              "maintainer = 邑，turnouts 是每个用水的村的分水口，served_km2 = 灌区里在种的地）与圩区；村级的（村的渠首与渠、村塘 / 山塘 / 堰塘、圩塘、废塘 / 废渠）"
              "归营建器按风格修，summary.village_works = false 时这里是空的。")


def _cells_min(n_km2: float, cell_km2: float) -> int:
    return int(math.ceil(n_km2 / cell_km2 - 1e-9))


# ---------------------------------------------------------------- 圩田：哪几片湿地排干、怎么分圩（farmland 调）
def polder_plan(g: dict, wc: dict, wet: np.ndarray, take1: np.ndarray, n_avail: int, pit: np.ndarray | None = None) -> dict:
    """wet：聚落层动手之前的湿地（布尔 [H, W]）；take1：第一遍好地先占的已垦（扁平布尔）；n_avail：额度里还能给圩田的格数；
    pit：不开田的格（扁平布尔：资源层的采场——湿地里的盐井——与岩类赋存；P6b 起不围进圩里，湿地片与出水口照旧按整片算）。
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
        o = int(cells[int(np.argmax(fa[cells]))])                  # 出水口：片里汇流最大的格（平局取格号小的）
        io, jo = divmod(o, W)
        cb = cells[~pit[cells]] if pit is not None else cells       # 采场那格不围进圩
        ii, jj = cb // W, cb % W
        bi, bj = (ii - io) // s, (jj - jo) // s
        srt = np.lexsort((cb, bj, bi))
        cs, bis, bjs = cb[srt], bi[srt], bj[srt]
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


# ---------------------------------------------------------------- 邑级大堰（四点四十：farmland.fill_cultivated 在好地先占之前调）
def big_plan(g: dict, wc: dict, pit: np.ndarray, n_quota: int) -> list[dict]:
    """水利分级（用户 09-30 定，Zhouzhu PLAN-LAND L32–L34）：邑级的（都江堰级）是因，在田和村之前定——
    常年河上汇水 ≥ big_src_min_km2 的格（按汇水降序、隔 big_try_sep_km）为候选渠首，渠水面 = 河床 + 水深 + big_weir_m；
    灌区的上界 = 离渠首 ≤ big_reach_km、同岛、在渠水面线以下（直线距离算比降）的目标格（坡 < big_cmd_slope_deg、宜垦、不是采场 / 岩类赋存），
    与水量上限（汇水 × 年雨 × 径流系数 / big_duty_mm）取小；按上界降序试前 big_try 个，各按最省工的路走一遍（与村的渠同式，不限村走得到），
    灌区按出堆的次序到水量上限就停，取灌得最多的；够 big_min_cmd_km2 且 ≥ 额度的 big_min_quota_frac 才修。一个水系（顺流而下到同一处）至多一处，至多 big_max 处。
    pit：不开田的格（[H, W] 布尔）。返回 [{island, head, outlet, z0, acc, water_km2, win, par, rc}]。C++ 的 waterworks.cpp big_plan 逐位同式。"""
    iid = g["island_id"]
    H, W = iid.shape
    N = H * W
    res_km = float(g["res_km"])
    res_m = res_km * 1000.0
    cell_km2 = res_km * res_km
    big_max = int(wc["big_max"])
    if big_max <= 0:
        return []
    J = g["json"]
    land = iid >= 0
    h = g["height"]
    acc = g["flowacc_km2"]
    accf = acc.ravel()
    depf = g["river_depth_m"].ravel()
    weir = float(wc["big_weir_m"])
    grad_c = float(wc["big_grad_m_per_km"]) * res_km
    cut = float(wc["canal_cut_m"])
    a_cost = float(wc["canal_slope_cost"])
    Rc = float(wc["big_reach_km"]) / res_km
    Rw = int(math.ceil(Rc - 1e-9))
    P = float(J["hydro"]["precip_mm"]) / 1000.0
    runoff = float(J["hydro"]["runoff_coef"])
    duty = float(wc["big_duty_mm"]) / 1000.0
    n_try = int(wc["big_try"])
    t = float(wc["big_try_sep_km"]) / res_km
    sep2 = t * t
    need = max(_cells_min(float(wc["big_min_cmd_km2"]), cell_km2), int(math.ceil(float(wc["big_min_quota_frac"]) * n_quota - 1e-9)))
    # 干渠走得过的格（不过崖缘、湖、常年河、湿地，不跨岛）；灌区的目标格
    base = land & ~g["cliff"] & ~g["lake"] & ~(g["river"] > 0) & (g["landcover"] != LC_WET)
    tgt = base & (g["cultivable"] > 0) & ~pit & (g["slope_deg"] < np.float32(wc["big_cmd_slope_deg"]))
    tgtf = tgt.reshape(-1)
    passl = base.ravel().tolist()
    tgtl = tgtf.tolist()
    hl = h.ravel().tolist()
    iidl = iid.ravel().tolist()
    # 渠首候选：按汇水降序（平局格号小），彼此隔 big_try_sep_km
    cand = np.flatnonzero((land & ~g["lake"] & (g["river"] > 0) & (acc >= np.float32(wc["big_src_min_km2"]))).ravel())
    cand = cand[np.argsort(-accf[cand], kind="stable")].tolist()
    kept, kp = [], []
    for q in cand:
        qi, qj = divmod(q, W)
        if any((qi - a) * (qi - a) + (qj - b) * (qj - b) < sep2 for a, b in kp):
            continue
        kept.append(q)
        kp.append((qi, qj))
    # 顺流而下到哪（D8 接收格走到头）：同一水系只修一处
    ri, rj = g["recv_i"].ravel().tolist(), g["recv_j"].ravel().tolist()
    outs = []
    for q in kept:
        for _ in range(N):
            qi, qj = divmod(q, W)
            a, b = ri[q], rj[q]
            if a < 0 or b < 0 or (a == qi and b == qj):
                break
            q = a * W + b
        outs.append(q)

    def dig(cc, z0, cap_n):
        """从渠首按最省工的路走出去：目标格里在渠水面以下的就是灌区，按出堆的次序到 cap_n 就停。"""
        ci, cj = divmod(cc, W)
        k = iidl[cc]
        r0, r1, c0, c1 = max(0, ci - Rw), min(H, ci + Rw + 1), max(0, cj - Rw), min(W, cj + Rw + 1)
        ww = c1 - c0
        nwin = (r1 - r0) * ww
        cost = [math.inf] * nwin
        ln = [0.0] * nwin
        par = [-1] * nwin
        s0 = (ci - r0) * ww + (cj - c0)
        cost[s0] = 0.0
        heap = [(0.0, cc)]
        rc = []
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
                hq = hl[q]
                if tgtl[q] and hq <= z0 - grad_c * ln[lq]:
                    rc.append(q)
                    if len(rc) >= cap_n:
                        break
            for sdi, sdj, L in STEPS:
                a, b = qi + sdi, qj + sdj
                if a < r0 or a >= r1 or b < c0 or b >= c1:
                    continue
                gp = a * W + b
                if not passl[gp] or iidl[gp] != k:
                    continue
                lp = (a - r0) * ww + (b - c0)
                nl = ln[lq] + L
                hp = hl[gp]
                if hp > z0 - grad_c * nl + cut:
                    continue
                sl = abs(hp - hq) / (L * res_m)
                nd = d + L * (1.0 + a_cost * sl * sl)
                if nd < cost[lp]:
                    cost[lp] = nd
                    ln[lp] = nl
                    par[lp] = q
                    heapq.heappush(heap, (nd, gp))
        return {"island": k, "head": cc, "z0": z0, "acc": float(accf[cc]), "win": (r0, r1, c0, c1), "par": par, "rc": rc}

    out, used_out = [], []
    for _w in range(big_max):
        ups = []
        for i, cc in enumerate(kept):
            if outs[i] in used_out:
                continue
            ci, cj = divmod(cc, W)
            k = iidl[cc]
            z0 = hl[cc] + float(depf[cc]) + weir
            water = float(accf[cc]) * P * runoff / duty
            cap_n = int(math.floor(water / cell_km2))
            r0, r1, c0, c1 = max(0, ci - Rw), min(H, ci + Rw + 1), max(0, cj - Rw), min(W, cj + Rw + 1)
            di = (np.arange(r0, r1) - ci)[:, None]
            dj = (np.arange(c0, c1) - cj)[None, :]
            dist = np.sqrt((di * di + dj * dj).astype(np.float64))
            m = int((tgt[r0:r1, c0:c1] & (iid[r0:r1, c0:c1] == k) & (dist <= Rc) & (h[r0:r1, c0:c1] <= z0 - grad_c * dist)).sum())
            ups.append((float(min(m, cap_n)), cc, water, z0, cap_n))
        ups.sort(key=lambda u: -u[0])
        best, best_n = None, 0
        for up, cc, water, z0, cap_n in ups[:max(0, n_try)]:
            if not (up > best_n):
                break
            B = dig(cc, z0, cap_n)
            if len(B["rc"]) > best_n:
                best_n = len(B["rc"])
                B["water_km2"] = water
                best = B
        if best_n == 0 or best_n < need:
            break
        best["outlet"] = outs[kept.index(best["head"])]
        used_out.append(best["outlet"])
        idx = np.asarray(best["rc"], dtype=np.int64)
        tgtf[idx] = False                               # 下一处不再灌这一处的地
        for q in best["rc"]:
            tgtl[q] = False
        out.append(best)
    return out


# ---------------------------------------------------------------- 渠、塘、闸（settle 在水设施之后调）
def build_waterworks(g: dict, wc: dict, fields: list[dict], fields_raster: np.ndarray, villages: list[dict], sraster: np.ndarray,
                     plan: dict, km, ruins: list[dict] | tuple = (), ruin_cells: dict | None = None, big: list | tuple = ()) -> dict:
    """P6b（L30）：每处水利都有管它的村（village），在那个村走得到的范围（manage_walk_km）内——
    渠首一村一堰：只灌管它的那个村自己的（旱地）田，渠只走在那个村走得到的地方；圩、圩塘、圩闸归种那组圩田的村，
    纵浦横塘按两旁的圩分给各自的村，排水闸 / 排水渠归出水口最近、走得到的村（排水渠走出去就截短）；
    废村旁（ruins）另挑没人管的废渠首、废渠（灌过它的撂荒田）与废塘（abandoned = True、abandoned_years = 撤空了几年）。"""
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
    # 管水利的村：田 → 种它的村（挂在村上的圩田也算）；走得到 = 离村格心的整数平方 ≤ W2
    t = float(wc["manage_walk_km"]) / res_km
    W2 = t * t
    wr = int(math.ceil(t - 1e-9))
    fvill = [0] * (nf + 1)
    for v in villages:
        fvill[v["field"]] = v["id"]
        for x in v.get("polder_fields", []):
            fvill[x] = v["id"]

    # ---------- 渠首与渠（一村一堰）
    src = land & ~lake & (acc >= np.float32(wc["works_src_min_km2"])) & (river | (stream & (acc >= np.float32(wc["works_stream_min_km2"]))))
    cand = np.flatnonzero(src.ravel())
    cand = cand[np.argsort(-zf[cand], kind="stable")].tolist()
    passable = land & ~g["cliff"] & ~lake & ~river & (cover != LC_WET)
    grad_c = float(wc["canal_grad_m_per_km"]) * res_km
    cut = float(wc["canal_cut_m"])
    Rc = float(wc["canal_reach_km"]) / res_km
    Rw = int(math.ceil(Rc - 1e-9))
    a_cost = float(wc["canal_slope_cost"])
    t = float(wc["village_head_sep_km"]) / res_km
    sep_h2 = t * t
    t = float(wc["canal_try_sep_km"]) / res_km
    sep_t2 = t * t
    min_cmd = _cells_min(float(wc["canal_min_cmd_km2"]), cell_km2)
    qk = float(wc["canal_q_per_km2"])
    wmin = float(wc["canal_width_min_m"])
    s_lat = max(1, int(round(float(wc["canal_lateral_km"]) / res_km)))
    b0 = float(wc["canal_base_km"])
    bk = float(wc["canal_km_per_km2"])

    def window(ci, cj, vi, vj):
        """Dijkstra 的窗：渠首 Rc 的方窗与管它的村 walk 的方窗之交。"""
        return max(0, ci - Rw, vi - wr), min(H, ci + Rw + 1, vi + wr + 1), max(0, cj - Rw, vj - wr), min(W, cj + Rw + 1, vj + wr + 1)

    def upper(ci, cj, z0, vi, vj, tgt):
        """灌得到的目标格的上界：离渠首 ≤ Rc、离管它的村走得到、低于渠水面线（直线距离算比降）。tgt(r0, r1, c0, c1) → 窗里的目标格。"""
        r0, r1, c0, c1 = window(ci, cj, vi, vj)
        if r0 >= r1 or c0 >= c1:
            return 0
        di = (np.arange(r0, r1) - ci)[:, None]
        dj = (np.arange(c0, c1) - cj)[None, :]
        dist = np.sqrt((di * di + dj * dj).astype(np.float64))
        ei = (np.arange(r0, r1) - vi)[:, None]
        ej = (np.arange(c0, c1) - vj)[None, :]
        m = tgt(r0, r1, c0, c1) & (dist <= Rc) & ((ei * ei + ej * ej) <= W2) & (h[r0:r1, c0:c1] <= z0 - grad_c * dist)
        return int(m.sum())

    def dig(c, ci, cj, z0, k, vi, vj, tgt):
        """从渠首按最省工的路走出去（只走管它的村走得到的格）：目标田里走得到、在渠水面以下的格就是灌区（按格算），
        够 canal_min_cmd_km2、新接到入口的渠 ≤ canal_base_km + canal_km_per_km2 × 灌区的面积（划得来）才修。
        返回 (入口, 格点, 父指针, 灌区的格（按出堆的次序）) 或 None。"""
        r0, r1, c0, c1 = window(ci, cj, vi, vj)
        ww = c1 - c0
        hw = h[r0:r1, c0:c1].ravel().tolist()
        ei = (np.arange(r0, r1) - vi)[:, None]
        ej = (np.arange(c0, c1) - vj)[None, :]
        pw = (passable[r0:r1, c0:c1] & (iid[r0:r1, c0:c1] == k) & ((ei * ei + ej * ej) <= W2)).ravel().tolist()
        tw = tgt(r0, r1, c0, c1).ravel().tolist()
        nwin = (r1 - r0) * ww
        cost = [math.inf] * nwin
        ln = [0.0] * nwin
        par = [-1] * nwin
        s0 = (ci - r0) * ww + (cj - c0)
        cost[s0] = 0.0
        heap = [(0.0, c)]
        cnt, entry, rc = 0, -1, []
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
                if tw[lq] and hq <= z0 - grad_c * ln[lq]:
                    rc.append(q)
                    if cnt == 0:
                        entry = q
                    cnt += 1
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
        if cnt < min_cmd:
            return None

        def up(q):
            return par[(q // W - r0) * ww + (q % W - c0)]
        q, n1, n2 = entry, 0, 0
        while q != c:
            p = up(q)
            if p // W != q // W and p % W != q % W:
                n2 += 1
            else:
                n1 += 1
            q = p
        if (n1 + n2 * SQRT2) * res_km > b0 + bk * (cnt * cell_km2):
            return None
        lat = sorted(q for q in rc if q != entry and (q // W - ci) % s_lat == 0 and (q % W - cj) % s_lat == 0)
        return entry, lat, up, rc

    def emit(c, k, entry, lat, up, n_cmd, hid, extra, idkey="head"):
        """渠 = 渠首到入口、再到各格点的路合成的树：从渠首出来的一段是干渠，分出去的是支渠；灌区的格数按格点均分给各段算渠宽。返回 (段, 直步, 斜步)。
        idkey：村的渠记渠首号（head），大堰的渠记堰号（work）。"""
        tl = [entry] + lat
        base, rem = divmod(n_cmd, len(tl))
        served, children = {}, {}
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
                q = up(q)
        segs, n1_h, n2_h = [], 0, 0
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
                    if u // W != v // W and u % W != v % W:
                        n2 += 1
                    else:
                        n1 += 1
                n1_h += n1
                n2_h += n2
                sv = served[pts[1]]
                q_m3s = qk * (sv * cell_km2)
                segs.append({"kind": CANAL_ZH["main" if s_ == c else "branch"], idkey: hid, "island": k,
                             "pts": [[float(p // W) + 0.5, float(p % W) + 0.5] for p in pts],
                             "length_km": round((n1 + n2 * SQRT2) * res_km, 3), "served_km2": round(sv * cell_km2, 3),
                             "width_m": round(max(wmin, 5.0 * math.sqrt(q_m3s)), 1), **extra})
                if children.get(cur):
                    dq.append(cur)
        return segs, n1_h, n2_h

    def head_rec(hid, k, c, ci, cj, z0, flist, n_cmd, n1_h, n2_h, n_seg, extra):
        seasonal = not bool(river.ravel()[c])
        return {"id": hid, "island": k, "cell": [ci, cj], "km": km(ci, cj), "source": SOURCE_ZH["stream" if seasonal else "river"],
                "seasonal": seasonal, "level_m": round(z0, 1), "basin_km2": round(float(accf[c]), 1), "fields": flist,
                "served_km2": round(n_cmd * cell_km2, 3), "canal_km": round((n1_h + n2_h * SQRT2) * res_km, 3), "n_canals": n_seg, **extra}

    near = lambda ci, cj, pts, r2: any((ci - a) * (ci - a) + (cj - b) * (cj - b) < r2 for a, b in pts)
    cmd = [False] * (nf + 1)
    cvill = [v for v in villages if not pol_field[v["field"]] and f_cells[v["field"]] >= min_cmd]     # 一村一堰：灌自己的旱地田
    heads, head_cells, tried, canals, cmd_cells = [], [], [], [], []
    n1_all = n2_all = 0

    # ---------- 邑级大堰（四点四十）：渠网按 big_plan 的那棵最省工的路树修到灌区里在种的格；每个用水的村一个分水口（渠上离村最近的格）；邑管
    vw = bool(wc["village_works"])
    s_big = max(1, int(round(float(wc["big_lateral_km"]) / res_km)))
    vby = {v["id"]: v for v in villages}
    bigserved = np.zeros((H, W), dtype=bool)
    cultf = g["cultivated"].ravel()
    big_works = []
    big_n1 = big_n2 = big_sv = big_pl = 0
    big_vs = set()
    for wid, B in enumerate(big, start=1):
        cc = B["head"]
        ci, cj = divmod(cc, W)
        k = B["island"]
        sv = [q for q in B["rc"] if cultf[q] > 0]          # 灌区里在种的格（按出堆的次序）
        big_pl += len(B["rc"])
        n1_h = n2_h = n_seg = 0
        nodes = []                                          # 渠经过的格（不含渠首），升序
        if sv:
            r0, r1, c0, c1 = B["win"]
            ww = c1 - c0
            par = B["par"]
            up = lambda q, r0=r0, c0=c0, ww=ww, par=par: par[(q // W - r0) * ww + (q % W - c0)]
            entry = sv[0]
            lat = sorted(q for q in sv if q != entry and (q // W - ci) % s_big == 0 and (q % W - cj) % s_big == 0)
            ns = set()
            for q in [entry] + lat:
                x = q
                while x != cc and x not in ns:
                    ns.add(x)
                    x = up(x)
            nodes = sorted(ns)
            segs, n1_h, n2_h = emit(cc, k, entry, lat, up, len(sv), wid, {"village": None, "maintainer": MAINT_ZH["yi"]}, "work")
            canals.extend(segs)
            n_seg = len(segs)
        big_n1 += n1_h
        big_n2 += n2_h
        big_sv += len(sv)
        if sv:
            bigserved.reshape(-1)[np.asarray(sv, dtype=np.int64)] = True
        cmd_cells.extend(sv)
        vcnt = {}                                           # 用水的村：灌区里在种的格属哪个村的田（挂在村上的圩田也算；散户的田不算）
        for q in sv:
            f = int(fidf[q])
            if f > 0 and fvill[f]:
                vcnt[fvill[f]] = vcnt.get(fvill[f], 0) + 1
        turnouts, vids = [], []
        for vid in sorted(vcnt):
            vi, vj = vby[vid]["cell"]
            bq, bd = -1, 0
            for q in nodes:
                d2 = (q // W - vi) * (q // W - vi) + (q % W - vj) * (q % W - vj)
                if bq < 0 or d2 < bd:
                    bq, bd = q, d2
            if bq < 0:
                continue
            turnouts.append({"village": vid, "cell": [bq // W, bq % W], "km": km(bq // W, bq % W), "served_km2": round(vcnt[vid] * cell_km2, 3),
                             "dist_km": round(math.sqrt(bd) * res_km, 2)})
            vids.append(vid)
            big_vs.add(vid)
        big_works.append({"id": wid, "kind": BIG_ZH["weir"], "island": k, "cell": [ci, cj], "km": km(ci, cj), "source": SOURCE_ZH["river"],
                          "level_m": round(B["z0"], 1), "basin_km2": round(B["acc"], 1), "water_km2": round(B["water_km2"], 1),
                          "planned_km2": round(len(B["rc"]) * cell_km2, 3), "served_km2": round(len(sv) * cell_km2, 3),
                          "canal_km": round((n1_h + n2_h * SQRT2) * res_km, 3), "n_canals": n_seg, "maintainer": MAINT_ZH["yi"],
                          "villages": vids, "turnouts": turnouts})
    if not vw:
        cand = []                                           # 村级的渠首、渠（含废渠）不在岛群层出
    for c in cand:
        ci, cj = divmod(c, W)
        if near(ci, cj, head_cells, sep_h2) or near(ci, cj, tried, sep_t2):
            continue
        k = int(iidf[c])
        z0 = float(zf[c])
        vs = []
        for v in cvill:
            vi, vj = v["cell"]
            d2 = (vi - ci) * (vi - ci) + (vj - cj) * (vj - cj)
            if v["island"] == k and not cmd[v["field"]] and d2 <= W2:
                vs.append((d2, v["id"], v))
        vs.sort(key=lambda x: (x[0], x[1]))
        ok_v = []
        for _d2, _id, v in vs:
            F = v["field"]
            if upper(ci, cj, z0, v["cell"][0], v["cell"][1], lambda r0, r1, c0, c1, F=F: (fields_raster[r0:r1, c0:c1] == F) & ~bigserved[r0:r1, c0:c1]) >= min_cmd:
                ok_v.append(v)
        if not ok_v:
            continue
        tried.append((ci, cj))
        for v in ok_v:
            F = v["field"]
            got = dig(c, ci, cj, z0, k, v["cell"][0], v["cell"][1], lambda r0, r1, c0, c1, F=F: (fields_raster[r0:r1, c0:c1] == F) & ~bigserved[r0:r1, c0:c1])
            if got is None:
                continue
            hid = len(heads) + 1
            head_cells.append((ci, cj))
            cmd[F] = True
            n_cmd = len(got[3])
            cmd_cells.extend(got[3])
            segs, n1_h, n2_h = emit(c, k, got[0], got[1], got[2], n_cmd, hid, {"village": v["id"]})
            canals.extend(segs)
            n1_all += n1_h
            n2_all += n2_h
            heads.append(head_rec(hid, k, c, ci, cj, z0, [F], n_cmd, n1_h, n2_h, len(segs), {"village": v["id"]}))
            break
    # 废村旁没人管的废渠首、废渠：灌过它的撂荒田（田照死了的村算）；一个废村至多一处
    ab_heads, ab_canals = [], []
    ab_n1 = ab_n2 = 0
    for r in sorted(ruins, key=lambda r: r["id"]):
        rcells = (ruin_cells or {}).get(r["tract"])
        if rcells is None or int(rcells.size) < min_cmd:
            continue
        tm = np.zeros(H * W, dtype=bool)
        tm[rcells] = True
        tm = tm.reshape(H, W)
        vi, vj = r["cell"]
        k = int(r["island"])
        rtried = []
        for c in cand:
            ci, cj = divmod(c, W)
            if int(iidf[c]) != k or (ci - vi) * (ci - vi) + (cj - vj) * (cj - vj) > W2:
                continue
            if near(ci, cj, head_cells, sep_h2) or near(ci, cj, rtried, sep_t2):
                continue
            z0 = float(zf[c])
            tgt = lambda r0, r1, c0, c1, tm=tm: tm[r0:r1, c0:c1]
            if upper(ci, cj, z0, vi, vj, tgt) < min_cmd:
                continue
            rtried.append((ci, cj))
            got = dig(c, ci, cj, z0, k, vi, vj, tgt)
            if got is None:
                continue
            hid = len(heads) + len(ab_heads) + 1
            head_cells.append((ci, cj))
            extra = {"village": None, "ruin": r["id"], "abandoned": True, "abandoned_years": int(r["abandoned_years"])}
            n_cmd = len(got[3])
            segs, n1_h, n2_h = emit(c, k, got[0], got[1], got[2], n_cmd, hid, extra)
            ab_canals.extend(segs)
            ab_n1 += n1_h
            ab_n2 += n2_h
            ab_heads.append(head_rec(hid, k, c, ci, cj, z0, [], n_cmd, n1_h, n2_h, len(segs), extra))
            break

    # ---------- 圩田：纵浦横塘、排水渠、圩堤、圩塘与闸（各归种那组圩田的村）
    polders, patches, pol_canals, sluices_p, sluices_o, ponds_p = [], [], [], [], [], []
    pidf = polder_id.ravel()
    recv_i, recv_j = g["recv_i"], g["recv_j"]
    drain_max = int(wc["polder_drain_max_cells"])
    pond_frac = float(wc["polder_pond_frac"])
    paddy_mm = np.float32(wc["polder_paddy_mm"])
    rain = g["rain_mm"]
    ns_cells = ew_cells = drain_n1 = drain_n2 = dike_edges = 0
    n_out_unmanaged = 0
    for P in plan["patches"]:
        io, jo = P["anchor"]
        s = int(P["spacing"])
        o = int(P["outlet"])
        oi, oj = divmod(o, W)
        k = int(P["island"])
        drained = 0
        pv = set()
        for b_i, b_j, bc in P["blocks"]:
            pid_ = int(pidf[int(bc[0])])
            vid = fvill[int(fidf[int(bc[0])])] or None
            if vid is not None:
                pv.add(vid)
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
            if vw:                                          # 圩塘是村级的，归营建器
                ponds_p.append({"kind": POND_ZH["polder"], "island": k, "cell": [pi_, pj_], "km": km(pi_, pj_), "polder": pid_,
                                "area_m2": int(round(n * cell_km2 * 1e6 * pond_frac)), "elev_m": round(float(hf[pond]), 0), "village": vid})
            sluices_p.append({"kind": SLUICE_ZH["polder"], "island": k, "cell": [si_, sj_], "km": km(si_, sj_), "polder": pid_, "village": vid})
            r0_, r1_ = int(ii.min()), int(ii.max()) + 1
            c0_, c1_ = int(jj.min()), int(jj.max()) + 1
            x0, y0 = km(r0_ - 0.5, c0_ - 0.5)
            x1, y1 = km(r1_ - 0.5, c1_ - 0.5)
            polders.append({"id": pid_, "patch": len(patches) + 1, "island": k, "block": [b_i, b_j], "cells": n, "area_km2": round(n * cell_km2, 3),
                            "cells_bbox": [r0_, c0_, r1_, c1_], "km_bbox": [x0, y1, x1, y0], "dike_km": round(edges * res_km, 3),
                            "paddy": bool(rain[pi_, pj_] >= paddy_mm), "village": vid})
        # 排水闸与排水渠：出水口最近、走得到的村管（同岛，平局村号小）；没有就不修；排水渠顺 D8 往下游接到河 / 溪涧 / 湖，走出那个村走得到的地方就截短
        ov = None
        for v in villages:
            if v["island"] != k:
                continue
            d2 = (v["cell"][0] - oi) * (v["cell"][0] - oi) + (v["cell"][1] - oj) * (v["cell"][1] - oj)
            if d2 <= W2 and (ov is None or d2 < ov[0]):
                ov = (d2, v)
        if ov is None:
            n_out_unmanaged += 1
        else:
            ovid = ov[1]["id"]
            vi, vj = ov[1]["cell"]
            pts = [o]
            q = o
            for _ in range(drain_max):
                qi, qj = divmod(q, W)
                a, b = int(recv_i[qi, qj]), int(recv_j[qi, qj])
                if a < 0 or b < 0 or (a == qi and b == qj) or (a - vi) * (a - vi) + (b - vj) * (b - vj) > W2:
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
                                   "pts": [[float(p // W) + 0.5, float(p % W) + 0.5] for p in pts], "length_km": round((n1 + n2 * SQRT2) * res_km, 3),
                                   "village": ovid})
            sluices_o.append({"kind": SLUICE_ZH["outlet"], "island": k, "cell": [oi, oj], "km": km(oi, oj), "patch": len(patches) + 1, "village": ovid})
        # 纵浦横塘：格线（行边 io + s·m、列边 jo + s·m）穿过这片圩田的段，按两旁的圩分给各自的村（上 / 左那格优先）
        dm = np.zeros(H * W, dtype=bool)
        for _b_i, _b_j, bc in P["blocks"]:
            dm[bc] = True
        dm = dm.reshape(H, W)
        ci_, cj_ = np.nonzero(dm)
        rmin, rmax, cmin, cmax = int(ci_.min()), int(ci_.max()), int(cj_.min()), int(cj_.max())
        own = lambda a, b: fvill[int(fidf[a * W + b])] if 0 <= a < H and 0 <= b < W and dm[a, b] else -1
        e = io + s * ((rmin - io + s - 1) // s)
        while e <= rmax + 1:
            ow = []
            for b in range(cmin, cmax + 1):
                x = own(e - 1, b)
                ow.append(x if x >= 0 else own(e, b))
            for a0, a1, vo in _runs_owner(ow):
                ew_cells += a1 - a0
                pol_canals.append({"kind": CANAL_ZH["ew"], "patch": len(patches) + 1, "island": k,
                                   "pts": [[float(e), float(cmin + a0)], [float(e), float(cmin + a1)]], "length_km": round((a1 - a0) * res_km, 3),
                                   "village": vo or None})
            e += s
        e = jo + s * ((cmin - jo + s - 1) // s)
        while e <= cmax + 1:
            ow = []
            for a in range(rmin, rmax + 1):
                x = own(a, e - 1)
                ow.append(x if x >= 0 else own(a, e))
            for a0, a1, vo in _runs_owner(ow):
                ns_cells += a1 - a0
                pol_canals.append({"kind": CANAL_ZH["ns"], "patch": len(patches) + 1, "island": k,
                                   "pts": [[float(rmin + a0), float(e)], [float(rmin + a1), float(e)]], "length_km": round((a1 - a0) * res_km, 3),
                                   "village": vo or None})
            e += s
        patches.append({"id": len(patches) + 1, "island": k, "outlet_cell": [oi, oj], "km": km(oi, oj), "pressure": round(float(P["pressure"]), 3),
                        "wetland_km2": round(P["wet_cells"] * cell_km2, 3), "polder_km2": round(drained * cell_km2, 3),
                        "polders": list(P["polder_ids"]), "villages": sorted(pv)})

    # ---------- 塘：堰塘（季节性渠首）、村塘 / 山塘（每村一口）、圩塘；废村旁的废塘
    free = land & ~g["cliff"] & ~lake & ~river & (cover != LC_WET) & (g["cultivated"] == 0) & (g["fallow_years"] == 0) & (sraster == 0)
    reach = int(wc["pond_reach_cells"])
    near_c = int(wc["pond_near_cells"])
    zone = g["terrain_zone"]
    hill_ratio = float(wc["hill_pond_ratio"])
    hill_min = float(wc["hill_pond_min_m2"])
    m2_hh = float(wc["pond_m2_per_hh"])
    v_min = float(wc["pond_min_m2"])

    def pick(ci, cj, r, hmin, walk=None):
        """(ci, cj) 周围 r 格（方窗）里能挖塘的格：汇流最大，平局离得近、格号小；hmin 给了就只要不低于它的格；
        walk = (行, 列) 给了就只要那个村走得到的格（堰塘：管渠首的村）。"""
        best, bacc, bd = -1, 0.0, 0
        for a in range(max(0, ci - r), min(H, ci + r + 1)):
            for b in range(max(0, cj - r), min(W, cj + r + 1)):
                if not free[a, b] or iid[a, b] != iid[ci, cj]:
                    continue
                if hmin is not None and not (h[a, b] >= hmin):
                    continue
                if walk is not None and (a - walk[0]) * (a - walk[0]) + (b - walk[1]) * (b - walk[1]) > W2:
                    continue
                av = float(acc[a, b])
                dd = (a - ci) * (a - ci) + (b - cj) * (b - cj)
                if best < 0 or av > bacc or (av == bacc and dd < bd):
                    best, bacc, bd = a * W + b, av, dd
        return best

    def mk_pond(q, kind, area, **ref):
        qi, qj = divmod(q, W)
        free[qi, qj] = False
        return {"kind": POND_ZH[kind], "island": int(iidf[q]), "cell": [qi, qj], "km": km(qi, qj), "area_m2": int(round(area)),
                "elev_m": round(float(hf[q]), 0), **ref}

    def village_pond(ci, cj, kind, area_hill, area_village, walk=None):
        """山塘在村上坡（不低于村）的格里挑，没有就放宽；村塘先在近处挑，没有就放宽；walk 给了就只挑那个村走得到的格。"""
        if kind == "hill":
            q = pick(ci, cj, reach, float(h[ci, cj]), walk)
            if q < 0:
                q = pick(ci, cj, reach, None, walk)
            return q, area_hill
        q = pick(ci, cj, near_c, None, walk)
        if q < 0:
            q = pick(ci, cj, reach, None, walk)
        return q, area_village

    ponds = []
    vcell = {v["id"]: v["cell"] for v in villages}
    for hd in heads:
        if not hd["seasonal"]:
            continue
        ci, cj = hd["cell"]
        q = pick(ci, cj, near_c, None, vcell[hd["village"]])
        if q >= 0:
            ponds.append(mk_pond(q, "weir", max(hill_min, hd["served_km2"] * 1e6 * hill_ratio), head=hd["id"], village=hd["village"]))
    for v in (villages if vw else ()):                   # 村塘 / 山塘归营建器
        F = int(v["field"])
        f = fields[F - 1]
        vi, vj = v["cell"]
        if cmd[F] or pol_field[F] or v.get("polder_fields") or v["id"] in big_vs:          # 灌区里的村（有渠或圩田）：村塘接渠水
            kind = "village"
        elif 1 <= int(zone[vi, vj]) <= 3:              # 高山 / 山地 / 丘陵：陂塘
            kind = "hill"
        else:
            kind = "village"
        q, area = village_pond(vi, vj, kind, max(hill_min, float(f["area_km2"]) * 1e6 * hill_ratio),
                               max(v_min, m2_hh * (int(v["households"]) + int(v.get("households_market", 0)))), (vi, vj))
        if q >= 0:
            ponds.append(mk_pond(q, kind, area, village=v["id"]))
    ponds_ab = []
    for r in (sorted(ruins, key=lambda r: r["id"]) if vw else ()):     # 废塘：死了的村的村塘 / 山塘，淤了、没人管
        vi, vj = r["cell"]
        kind = "hill" if 1 <= int(zone[vi, vj]) <= 3 else "village"
        q, area = village_pond(vi, vj, kind, max(hill_min, float(r["fallow_km2"]) * 1e6 * hill_ratio),
                               max(v_min, m2_hh * int(r["households_before"])))
        if q >= 0:
            ponds_ab.append(mk_pond(q, kind, area, village=None, ruin=r["id"], abandoned=True, abandoned_years=int(r["abandoned_years"])))
    ponds.extend(ponds_p)
    n_live_ponds = len(ponds)
    ponds.extend(ponds_ab)
    for i, x in enumerate(ponds):
        sraster[x["cell"][0], x["cell"][1]] = 19 if x.get("abandoned") else 15
        x["id"] = i + 1

    # ---------- 闸
    sluices = [{"kind": SLUICE_ZH["head"], "island": hd["island"], "cell": list(hd["cell"]), "km": hd["km"], "head": hd["id"], "village": hd["village"]}
               for hd in heads]
    sluices += sluices_p + sluices_o
    n_live_sl = len(sluices)
    sluices += [{"kind": SLUICE_ZH["head"], "island": hd["island"], "cell": list(hd["cell"]), "km": hd["km"], "head": hd["id"], "village": None,
                 "ruin": hd["ruin"], "abandoned": True, "abandoned_years": hd["abandoned_years"]} for hd in ab_heads]
    for i, x in enumerate(sluices):
        x["id"] = i + 1
        a, b = x["cell"]
        if x.get("abandoned"):
            if sraster[a, b] in (0, 1, 2, 11):
                sraster[a, b] = 19
        elif sraster[a, b] in (0, 1, 2):
            sraster[a, b] = 16
    canals += pol_canals
    canals += ab_canals
    for i, x in enumerate(canals):
        x["id"] = i + 1
    heads += ab_heads

    n_cmd_all = len(cmd_cells)
    n_cult = int((g["cultivated"] > 0).sum())
    wet_cells = int(plan["wetland_cells"])
    pol_cells = int((polder_id > 0).sum())
    pk = {kk: 0 for kk in POND_ZH.values()}
    for x in ponds[:n_live_ponds]:
        pk[x["kind"]] += 1
    sk = {kk: 0 for kk in SLUICE_ZH.values()}
    for x in sluices[:n_live_sl]:
        sk[x["kind"]] += 1
    summary = {"n_heads": len(heads) - len(ab_heads), "n_heads_seasonal": sum(1 for x in heads if x["seasonal"] and not x.get("abandoned")),
               "canal_km": round((n1_all + n2_all * SQRT2) * res_km, 3), "polder_canal_km": round((ns_cells + ew_cells) * res_km, 3),
               "drain_km": round((drain_n1 + drain_n2 * SQRT2) * res_km, 3),
               "commanded_km2": round(n_cmd_all * cell_km2, 3), "commanded_share": round(n_cmd_all / max(1, n_cult), 3),
               "n_ponds": n_live_ponds, "ponds": pk, "n_sluices": n_live_sl, "sluices": sk,
               "wetland_km2": round(wet_cells * cell_km2, 3), "polder_km2": round(pol_cells * cell_km2, 3),
               "polder_share": round(pol_cells / wet_cells, 3) if wet_cells else 0.0,
               "n_polders": len(polders), "n_polder_patches": len(patches), "dike_km": round(dike_edges * res_km, 3),
               "manage_walk_km": float(wc["manage_walk_km"]), "n_outlets_unmanaged": n_out_unmanaged,
               "n_polder_villages": sum(1 for v in villages if v.get("polder")),
               "n_villages_with_polders": sum(1 for v in villages if v.get("polder_fields")),
               "abandoned": {"heads": len(ab_heads), "canal_km": round((ab_n1 + ab_n2 * SQRT2) * res_km, 3), "ponds": len(ponds_ab),
                             "sluices": len(ab_heads)},
               "village_works": vw,
               "big": {"n": len(big), "planned_km2": round(big_pl * cell_km2, 3), "served_km2": round(big_sv * cell_km2, 3),
                       "canal_km": round((big_n1 + big_n2 * SQRT2) * res_km, 3), "share": round(big_sv / max(1, n_cult), 3), "villages": len(big_vs)}}
    return {"heads": heads, "big_works": big_works, "canals": canals, "ponds": ponds, "sluices": sluices, "polders": polders, "polder_patches": patches,
            "summary": summary, "note": WORKS_NOTE, "_commanded_cells": sorted(cmd_cells)}


def _runs_owner(owners: list) -> list[tuple[int, int, int]]:
    """格线上连续有圩、管它的村不变的段 [(起, 止, 村号)]（止不含）；owners 里 −1 = 两旁都不是这片的圩，0 = 圩没有管它的村。"""
    out, a = [], -1
    for i, x in enumerate(owners):
        if a >= 0 and x != owners[a]:
            out.append((a, i, owners[a]))
            a = -1
        if x >= 0 and a < 0:
            a = i
    if a >= 0:
        out.append((a, len(owners), owners[a]))
    return out


# ---------------------------------------------------------------- 原始地表与人工改造（P6b，Zhouzhu PLAN-LAND L31）
LC_ROCK, LC_FOREST, LC_SHRUB, LC_GRASS, LC_RIVER, LC_LAKE = 2, 4, 5, 6, 10, 11
LANDUSE_CLASSES = ["没动过（原始地貌）", "开垦的田", "梯田", "渠灌田", "圩田（原为湿地）", "撂荒（在往回长）", "樵牧（村周砍林）", "采场（挖开的地）"]
LANDUSE_NOTE = ("人工改造（terrain.npz 的 landuse，P6b）：0 没动过；1 开垦的田（在种的旱田，原来是什么见 landcover_natural）；2 梯田（坡地修成台阶）；"
                "3 渠灌田（邑级大堰的渠灌得到的在种的地；village_works = true 时也含村的渠灌的田）；4 圩田（原为湿地，排干围圩）；5 撂荒（开过又撂下、按年头在往回长，含废村的田）；"
                "6 樵牧（村周的林子砍成草场 / 薪炭林）；7 采场（挖开的坑、采石场）。landcover_natural = 没有人以前的地表（上等地画田之前的地表，河、湖照旧）")


def landuse_layers(g: dict, cmd_cells: list[int]) -> None:
    """g["landcover_natural"]（没有人以前的地表：hydro 在上等地画进地表之前记的 cover_natural，河道 / 湖照现状）与 g["landuse"]（人工改造码，LANDUSE_CLASSES）。
    后写的盖先写的：采场 → 樵牧 → 撂荒 → 开垦的田 / 梯田 → 渠灌田 → 圩田。C++ 的 settle.cpp 同式。"""
    nat = g["cover_natural"].copy()
    nat[g["river"] > 0] = LC_RIVER
    nat[g["lake"]] = LC_LAKE
    lc = g["landcover"]
    land = g["island_id"] >= 0
    chg = land & (lc != nat)
    lu = np.zeros(lc.shape, dtype=np.uint8)
    lu[chg & (lc == LC_ROCK)] = 7
    lu[chg & ((lc == LC_GRASS) | (lc == LC_SHRUB)) & (nat == LC_FOREST)] = 6
    lu[g["fallow_years"] > 0] = 5
    cult = g["cultivated"]
    lu[cult == 1] = 1
    lu[cult == 2] = 2
    if cmd_cells:                                   # 渠灌田：谷口的渠灌得到的格（走得到、在渠水面以下）
        lu.ravel()[np.asarray(cmd_cells, dtype=np.int64)] = 3
    lu[g["polder_id"] > 0] = 4
    g["landcover_natural"] = nat
    g["landuse"] = lu
