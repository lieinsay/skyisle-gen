"""量「季相与水情」日层的溪涧干季收缩：把 island.html 的日层模型（两个线性水库 + aMin）在 Python 里复现，逐节点算一年里
「有水」的溪涧格占比，用来看 A_REF（island.html 里的常数，= 年中位门槛相对溪涧遮罩阈值的倍数）该取多少。

日层模型（island.html buildDayTab；四点二十）：
    S ← S·kb + 液体降水（τ 30 天，两遍跑到年初平衡）；aMin = clip(A_REF × S_REF / S, 溪涧阈 aStream, 常年河阈 aPer)；
    某格有水 ⇔ 该格汇流 ≥ aMin（±25% 内按比例半干半湿，这里只数「有水」）。S_REF = (1000/n)/(1−kb)：年 1000 mm 均匀下时基流库的稳态。

用法：python docs/probes/daywater.py [run/id ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAU_B, MELT, A_REF = 30.0, 3.0, 2.0
KB = float(np.exp(-1.0 / TAU_B))
AREFS = (1.0, 1.4, 2.0, 3.0)


def a_min_series(days: list, n: int, a_stream: float, a_per: float, aref: float) -> np.ndarray:
    """基流水库 → 每日的溪涧门槛 aMin（km²）。与 island.html 的 buildDayTab 同式（含积雪 / 度日融雪那一步——
    冷天的降水存进雪包，暖天按 melt = 3 mm/°C·日 放回来；不这样 S 会偏低、干季偏深）。"""
    S_REF = (1000.0 / n) / (1.0 - KB)
    out = np.zeros(n)
    S = spk = 0.0
    for p in range(2):                      # 两遍连着跑：第一遍把水库与雪包转到年初的平衡态，第二遍记录
        for d in range(n):
            T = float(days[d]["temp_c"])
            P = float(days[d]["precip_mm"])
            liquid = 0.0
            if T <= 0.5:
                spk += P
            else:
                liquid += P
            melt = min(spk, MELT * T) if T > 0 else 0.0
            spk -= melt
            liquid += melt
            S = S * KB + liquid
            if p:
                out[d] = min(a_per, max(a_stream, aref * S_REF / max(S, 1e-6)))
    return out


def wet_frac(d: Path, aref: float) -> dict:
    J = json.loads((d / "island.json").read_text(encoding="utf-8"))
    C = json.loads((d / "climate.json").read_text(encoding="utf-8"))
    days = C["weather"]["days"]
    n = len(days)
    a_stream = float(J["hydro"].get("stream_min_km2") or 1.0)
    a_per = float(J["hydro"].get("river_threshold_km2") or 1e9)
    with np.load(d / "terrain.npz") as z:
        st = z["stream"] > 0
        acc = z["flowacc_km2"].astype(np.float64)   # 汇流 km²（栅格里就是它；JS 画的时候才从量化的 log u8 还原）
    acc = acc[st]
    amin = a_min_series(days, n, a_stream, a_per, aref)
    frac = np.array([float((acc >= amin[d] * 0.75).mean()) for d in range(n)])   # ±25% 内按比例半干半湿，这里数「有水」
    return {"n_cells": int(acc.size), "a_stream": a_stream, "a_per": a_per,
            "aMin_med": float(np.median(amin)), "aMin_max": float(amin.max()),
            "wet_min": float(frac.min()), "wet_p10": float(np.percentile(frac, 10)), "wet_med": float(np.median(frac)),
            "days_below_half": int((frac < 0.5).sum()), "days_below_quarter": int((frac < 0.25).sum())}


def main(argv: list[str]) -> int:
    dirs = []
    for run in (argv or ["out/seed42", "out/seed7", "out/seed2026"]):
        for p in sorted((ROOT / run / "islands").iterdir()):
            if (p / "island.json").exists():
                dirs.append(p)
    print(f"{len(dirs)} 个节点；A_REF = {AREFS}（现值 {A_REF}）\n")
    print(f"{'节点':<16}{'溪涧格':>7}{'aStream':>9}{'aMin 中位':>11}" + "".join(f"{' 湿' + str(a):>9}" for a in AREFS) + f"{'  <50% 天':>10}{'  <25% 天':>10}")
    for d in dirs:
        r = wet_frac(d, A_REF)
        line = f"{d.parent.parent.name}/{d.name:<6}{r['n_cells']:>8}{r['a_stream']:>9.2f}{r['aMin_med']:>11.2f}"
        for a in AREFS:
            line += f"{wet_frac(d, a)['wet_min'] * 100:>8.0f}%"
        print(line + f"{r['days_below_half']:>10}{r['days_below_quarter']:>10}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
