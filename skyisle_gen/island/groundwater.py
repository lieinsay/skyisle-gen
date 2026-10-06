"""谷底（C2）、地下水（C5）的前端（PLAN-NATURE C，DESIGN-NOTES 四点四十七）：算法在 C++（core/src/island/river.cpp、groundwater.cpp）；
这里只有 island.json 的摘要、中文名与说明。

- 谷底：terrain.npz 的 floodplain = 谷底里的岸上格（滩面 = 平岸水面，涨水即漫）；河道格的 floor_w_m / confine 在 rivers.json 的沿程点上（C3）。
- 地下水：terrain.npz 的 recharge_mm（补给 = 雨的径流 × 基流比例）、**wt_m / wt_depth_m（水位面与埋深，B，四点四十九）**；崖壁泉线在 rivers.json 的 springline（分弥散渗出 / 泉 / 崖瀑）。
"""
from __future__ import annotations

import numpy as np

CONFINE_ZH = {1: "峡谷", 2: "半限制", 3: "开阔"}
SPRING_ZH = {"karst_spring": "岩溶泉", "contact_spring": "接触泉", "fissure_spring": "裂隙泉", "seep_spring": "渗水泉"}
NOTE_SPRING = "出水约 {:.1f} L/s（该处顺流向累计的地下水补给）"
WATER_NOTE = ("普通地下水：补给 = 降雨产出的径流 × 基流比例（按出露岩性）；进了河道是河的基流，"
              "其余流向岸边形成崖壁泉线。不含集水核额外凝结水。")
WT_NOTE = ("水位面（B，四点四十九）：稳态潜水面 ∇·(T∇h) = −R——排水口（河道 / 溪涧 / 湖 / 岸缘）固定水头 = 地表，水位高过地表就钉回地表（渗出面），"
           "不低于骨架顶面 + 最小含水厚；流向按水位面的梯度（不再按地表）。terrain.npz 的 wt_m = 水位（零点口径，m）、wt_depth_m = 埋深（地表 − 水位，m）——"
           "井打多深、挖到哪层见水读它。崖壁泉线按段分三等：弥散渗出 / 泉 / 崖瀑（出水 ≥ 本岛出口段分位且含水层够厚才算泉，A）")
KIND_ZH = {0: "弥散渗出", 1: "泉", 2: "崖瀑"}


def valley_summary(g: dict, cell_km2: float) -> dict:
    """谷底（C2）：漫滩面积、常年河河道格的谷底宽分位与限制度的占比。"""
    out = {"floodplain_km2": round(float(g["floodplain"].sum()) * cell_km2, 3)}
    if "floor_w_m" in g:
        rv = g["river"] > 0
        fw = g["floor_w_m"][rv].astype(np.float64)
        cf = g["confine"][rv]
        if fw.size:
            out["river_floor_w_m"] = {"p10": round(float(np.percentile(fw, 10)), 0), "p50": round(float(np.percentile(fw, 50)), 0),
                                      "p90": round(float(np.percentile(fw, 90)), 0), "max": round(float(fw.max()), 0)}
            out["river_confinement"] = {CONFINE_ZH[k]: round(float((cf == k).mean()), 3) for k in (1, 2, 3)}
    return out


def water_summary(g: dict, R: dict, island_id: np.ndarray, cell_km2: float) -> dict:
    """地下水（C5）的 island.json 摘要。"""
    land = island_id >= 0
    n = max(1, int(land.sum()))
    run = R["runoff_mm"].astype(np.float64)
    tot_run = float(run[land].sum())
    out = {"note": WATER_NOTE}
    if "wt" in R:
        # B（四点四十九）：水位面 / 埋深——井打多深、挖到哪层见水、泉在哪，都读它
        wt = R["wt"].astype(np.float64)
        hgt = R["height"].astype(np.float64)
        d = np.where(land, hgt - wt, np.nan)[land]
        sl = R.get("springline", [])
        n_k = [sum(1 for s in sl if int(s.get("kind", 0)) == k) for k in range(3)]
        km_k = [round(sum(float(s["length_km"]) for s in sl if int(s.get("kind", 0)) == k), 1) for k in range(3)]
        ls_k = [round(sum(float(s["q_ls"]) for s in sl if int(s.get("kind", 0)) == k), 1) for k in range(3)]
        out.update({"wt_note": WT_NOTE,
                    "wt_depth_m": {"p10": round(float(np.nanpercentile(d, 10)), 1), "p50": round(float(np.nanpercentile(d, 50)), 1),
                                   "p90": round(float(np.nanpercentile(d, 90)), 1), "max": round(float(np.nanmax(d)), 1)},
                    "wt_shallow_share": round(float((d <= 5.0).mean()), 3),
                    "spring_kind": {KIND_ZH[k]: {"n": n_k[k], "km": km_k[k], "ls": ls_k[k]} for k in range(3)}})
    if "recharge_mm" in R:
        rec = R["recharge_mm"].astype(np.float64)
        sl = R.get("springline", [])
        q = [float(s["q_ls"]) for s in sl]
        out.update({"recharge_mm": round(float(rec[land].sum()) / n, 1),
                    "baseflow_share": round(float(rec[land].sum()) / max(1e-9, tot_run), 3),
                    "springline_segments": len(sl), "springline_ls": round(sum(q), 1),
                    "springline_falls": int(sum(1 for s in sl if s["fall"])),
                    "springline_km": round(float(sum(float(s["length_km"]) for s in sl)), 1)})
    return out
