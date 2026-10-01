"""谷底（C2）、集水核（C4）与地下水（C5）的前端（PLAN-NATURE C，DESIGN-NOTES 四点四十七）：算法在 C++（core/src/island/river.cpp、groundwater.cpp）；
这里只有 island.json 的摘要、中文名与说明。

- 谷底：terrain.npz 的 floodplain = 谷底里的岸上格（滩面 = 平岸水面，涨水即漫）；河道格的 floor_w_m / confine 在 rivers.json 的沿程点上（C3）。
- 集水核：terrain.npz 的 condense_mm（凝结水，mm/年；只进水账——runoff_mm 含它，rain_mm 不含）、cloud_forest（云雾林）。
- 地下水：terrain.npz 的 recharge_mm（补给 = 雨的径流 × 基流比例 + 凝结水）、**wt_m / wt_depth_m（水位面与埋深，B，四点四十九）**；崖壁泉线在 rivers.json 的 springline（分弥散渗出 / 泉 / 崖瀑）。
"""
from __future__ import annotations

import numpy as np

CONFINE_ZH = {1: "峡谷", 2: "半限制", 3: "开阔"}
SPRING_ZH = {"karst_spring": "岩溶泉", "contact_spring": "接触泉", "fissure_spring": "裂隙泉", "seep_spring": "渗水泉", "core_hotspring": "核山温泉"}
NOTE_SPRING = "出水约 {:.1f} L/s（该处顺流向累计的地下水补给）"
NOTE_HOTSPRING_CORE = ("集水核凝结的热约千分之一进岩体（本岛约 {:.1f} MW）：大核山下深循环的水出成温泉，"
                       "沿谷底、离核山近（spec 13 第八节第 7 条）")
WATER_NOTE = ("集水核（C4，spec 13 第八节）：凝结水 condense_mm 只进水账（渗进岩层 → 基流与泉，runoff_mm 含它），不进局地雨 rain_mm、不进天气；"
              "最多到局地雨的 1 倍、跟着湿度走。地下水（C5）：补给 = 雨的径流 × 基流比例（按出露岩性）+ 凝结水；进了河道是河的基流，"
              "没进河道走到岸边的从崖壁上岩层与浮石的交界渗出（崖壁泉线，rivers.json 的 springline）")
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
    """集水核（C4）与地下水（C5）的 island.json 摘要。"""
    land = island_id >= 0
    n = max(1, int(land.sum()))
    cond = R["condense_mm"].astype(np.float64)
    rain = R["rain_mm"].astype(np.float64)
    run = R["runoff_mm"].astype(np.float64)
    tot_run = float(run[land].sum())
    out = {"note": WATER_NOTE,
           "condense_mm": round(float(cond[land].sum()) / n, 1),
           "condense_share_of_runoff": round(float(cond[land].sum()) / max(1e-9, tot_run), 3),
           "condense_max_ratio": round(float(np.max(np.where(land & (rain > 0), cond / np.maximum(rain, 1e-9), 0.0))), 3),
           "core_strength": [round(float(x), 3) for x in R["core_s"]],
           "cloud_forest_km2": round(float(R["cloud_forest"].sum()) * cell_km2, 3) if "cloud_forest" in R else 0.0}
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
