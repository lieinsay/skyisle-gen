"""河的数据（PLAN-NATURE C3，DESIGN-NOTES 四点四十七）的前端：算法在 C++（core/src/island/rivernet.cpp）；
这里只有中文名、rivers.json 的段 / 瀑布 / 流域与 island.json 的摘要。只给驱动量：流速、当天水位与水面宽、河段型、冰情、水色由游戏推。
"""
from __future__ import annotations

import numpy as np

D50_ZH = ["泥", "砂", "砾", "卵石", "漂砾", "基岩"]
PLANFORM_ZH = ["受限（随谷）", "低能顺直", "曲流", "曲流带串沟心滩", "辫状"]
CONFINE_ZH = ["", "峡谷", "半限制", "开阔"]
EXIT_ZH = ["汇入", "崖边", "湖", "没入地里"]
FALL_ZH = ["崖边瀑布", "瀑布", "跌水"]
COLS = ["row", "col", "level", "acc_km2", "q_mean", "q_bf", "w_bf_m", "d_bf_m", "w_mean_m", "d_mean_m", "surf_m", "bed_m", "slope", "d50_mm", "d50", "planform",
        "ssc_mgl", "fp_left_m", "fp_right_m", "confine"]
NOTE = ("河网（C3）：segments 每段从上游到下游，干流按汇水最大的一支往上追，down = 汇入的段号（−1 = 出口）、join = 汇入处在下游段里的点号；"
        "exit：汇入 / 崖边 / 湖 / 没入地里；basin = 所在流域（basins 的下标，−1 = 其余）。pts 的列见 cols：行列是群栅格的格（格心 = 整数 + 0.5），"
        "level 0 = 季节性溪涧、1–3 小中大河；q_mean 年均流量（m³/s，按格的 Budyko 径流累计，含集水核的凝结水）、q_bf 平岸流量（= 年均 × 所在流域逐日径流指数的年最大）；"
        "**w_bf_m / d_bf_m 平岸宽与深**（= 年均口径 × (q_bf/q_mean)^站内指数，L&M 1964：宽 0.26、深 0.40；比 w_mean_m 宽 (平岸/年均)^0.26 ≈ 1.5–2 倍）、"
        "**w_mean_m / d_mean_m 年均流量下的水面宽深**（w = 5·Q^0.5 直接算；屏幕上那条河画的是栅格 terrain.npz 的 w_ch_m / d_ch_m = **河道**，两者差一个假设的 bf_ratio_channel）；surf_m 平岸水面（= 该格 height）、bed_m 河床；slope 河床比降（顺流 500 m）；"
        "d50_mm 河床质中值粒径（平岸 Shields 数：砾床 0.05、砂床 1；乘上游岩性的粗细）与档 d50（d50_classes）；"
        "planform 平面型（Kleinhans & van den Berg 2011，planform_classes）；ssc_mgl 年均悬沙浓度（BQART，Syvitski & Milliman 2007，按上游的林与已垦）；"
        "fp_left_m / fp_right_m 顺流向左右的谷底宽（到谷坡脚）；confine 限制度（confine_classes）")
INDEX_NOTE = ("逐日径流指数（第 year 年）：每条出崖边的常年河一个流域，最后一个（rest）是其余的小流域与直接出崖边的坡面共用；某点某天的流量 = 年均 × 所在流域当天的指数。"
              "由逐日天气推：按高程分带积雪（日均温 ≤ 0.5 °C）、度日融雪，年径流按当天到地面的水分到各日，快流（时间常数随流域面积）与基流"
              "（按岩性的基流比例与退水常数，加凝结水）两个线性水库，同一年跑两遍取第二遍")
FALLS_NOTE = ("瀑布与跌水：崖边瀑布 = 河从崖边跌下（落差 = 河面 − 岛底）；岛内相邻两点河床陡落（≥ 3 m 且比降 ≥ 0.15）连着的几级合起来 ≥ 5 m 的记下，"
              "合起来 ≥ 20 m 且平均比降 ≥ 0.25、或落在崖层上的叫瀑布（格上看不出单级的竖直跌落，河穿过硬岩崖带多半是一串瀑布与深潭）、其余叫跌水；seg / idx 指 segments 里的点")


def rivers_doc(g: dict) -> dict:
    """rivers.json 里 C3 的部分（段、瀑布、流域）。"""
    N = g["rivernet"]
    W = g["height"].shape[1]
    cell = N["cell"].astype(np.int64)
    num = {"row": cell // W, "col": cell % W, "level": N["level"].astype(np.int64),
           "acc_km2": np.round(N["acc"].astype(np.float64), 2), "q_mean": np.round(N["q_mean"].astype(np.float64), 4),
           "q_bf": np.round(N["q_bf"].astype(np.float64), 4), "w_bf_m": np.round(N["w"].astype(np.float64), 2),
           "d_bf_m": np.round(N["d"].astype(np.float64), 2),
           "w_mean_m": np.round(N["w_mean"].astype(np.float64), 2), "d_mean_m": np.round(N["d_mean"].astype(np.float64), 2), "surf_m": np.round(N["surf"].astype(np.float64), 1),
           "bed_m": np.round(N["bed"].astype(np.float64), 1), "slope": np.round(N["slope"].astype(np.float64), 5),
           "d50_mm": np.round(N["d50_mm"].astype(np.float64), 3), "d50": N["d50c"].astype(np.int64), "planform": N["planform"].astype(np.int64),
           "ssc_mgl": np.round(N["ssc"].astype(np.float64), 0), "fp_left_m": np.round(N["fp_l"].astype(np.float64), 0),
           "fp_right_m": np.round(N["fp_r"].astype(np.float64), 0), "confine": N["confine"].astype(np.int64)}
    cols = [num[k].tolist() for k in COLS]
    rows = [list(r) for r in zip(*cols)]
    segs = []
    for i, s in enumerate(N["segs"]):
        a, n = int(s["start"]), int(s["n"])
        segs.append({"id": i, "island": int(s["island"]), "down": int(s["down"]), "join": int(s["join"]), "exit": EXIT_ZH[int(s["exit"])],
                     "level": int(s["level"]), "basin": int(s["basin"]), "length_km": round(float(s["length_km"]), 2), "pts": rows[a:a + n]})
    falls = [{"seg": int(f["seg"]), "idx": int(f["idx"]), "cell": [int(x) for x in f["cell"]], "kind": FALL_ZH[int(f["kind"])], "level": int(f["level"]),
              "drop_m": round(float(f["drop_m"]), 1), "length_m": round(float(f["length_m"]), 0), "q_mean": round(float(f["q_mean"]), 4)}
             for f in N["falls"]]
    basins = []
    nb = len(N["basins"]) - 1
    for i, b in enumerate(N["basins"]):
        e = {"id": i, "area_km2": round(float(b["area_km2"]), 2), "q_mean": round(float(b["q_mean"]), 4), "bfi": round(float(b["bfi"]), 3),
             "recession_days": round(float(b["recession_days"]), 1), "quick_days": round(float(b["quick_days"]), 2),
             "snow_frac": round(float(b["snow_frac"]), 3), "cond_frac": round(float(b["cond_frac"]), 3), "bf_ratio": round(float(b["bf_ratio"]), 2),
             "index": np.round(np.asarray(b["index"], dtype=np.float64), 3).tolist()}
        if i < nb:
            e.update({"seg": int(b["seg"]), "island": int(b["island"]), "mouth": [int(x) for x in b["mouth"]]})
        else:
            e["rest"] = True
        basins.append(e)
    return {"network_note": NOTE, "cols": COLS, "d50_classes": D50_ZH, "planform_classes": PLANFORM_ZH, "confine_classes": CONFINE_ZH,
            "segments": segs, "falls_note": FALLS_NOTE, "falls": falls, "index_note": INDEX_NOTE, "year": int(g.get("year", 0)), "basins": basins}


def river_data_summary(g: dict) -> dict:
    """island.json 的 hydro.river_data：段数、各级长度、河（level ≥ 1）按长度的河床质 / 平面型 / 限制度占比、瀑布、流域的平岸流量倍数与积雪。"""
    N = g["rivernet"]
    lvl = N["level"].astype(np.int64)
    segs = N["segs"]
    length = {"溪涧": 0.0, "河": 0.0}
    for s in segs:
        length["河" if int(s["level"]) >= 1 else "溪涧"] += float(s["length_km"])
    rv = lvl >= 1
    out = {"segments": len(segs), "points": int(lvl.size), "length_km": {k: round(v, 1) for k, v in length.items()}}
    if rv.any():
        out["river_d50"] = {D50_ZH[k]: round(float((N["d50c"][rv] == k).mean()), 3) for k in range(len(D50_ZH)) if (N["d50c"][rv] == k).any()}
        out["river_planform"] = {PLANFORM_ZH[k]: round(float((N["planform"][rv] == k).mean()), 3) for k in range(len(PLANFORM_ZH))
                                 if (N["planform"][rv] == k).any()}
        out["river_confine"] = {CONFINE_ZH[k]: round(float((N["confine"][rv] == k).mean()), 3) for k in (1, 2, 3)}
        out["river_ssc_mgl_p50"] = round(float(np.median(N["ssc"][rv])), 0)
    kinds = [FALL_ZH[int(f["kind"])] for f in N["falls"]]
    inner = [float(f["drop_m"]) for f in N["falls"] if int(f["kind"]) > 0]
    out["falls"] = {k: kinds.count(k) for k in FALL_ZH}
    if inner:
        out["falls"]["inner_max_drop_m"] = round(max(inner), 1)
    B = N["basins"][:-1]
    out["basins"] = len(B)
    if B:
        big = max(B, key=lambda b: float(b["area_km2"]))
        out["largest_basin"] = {"area_km2": round(float(big["area_km2"]), 1), "q_mean": round(float(big["q_mean"]), 3), "bf_ratio": round(float(big["bf_ratio"]), 2),
                                "bfi": round(float(big["bfi"]), 3), "snow_frac": round(float(big["snow_frac"]), 3)}
        out["bf_ratio_range"] = [round(min(float(b["bf_ratio"]) for b in B), 2), round(max(float(b["bf_ratio"]) for b in B), 2)]
    out["rest_bf_ratio"] = round(float(N["basins"][-1]["bf_ratio"]), 2)
    return out
