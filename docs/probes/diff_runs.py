"""两份产物逐项对照（改动前后的回归；Python 参考后端删掉以后顶 `island compare` 的位置，DESIGN-NOTES 四点四十二）。

    python docs/probes/diff_runs.py out/seed42 out/verify42                 # 行星层：①–⑨ 的 npz 逐数组逐位、json 逐值、md 逐字
    python docs/probes/diff_runs.py 旧/islands/2051 out/seed42/islands/2051  # 岛群产物：npz 逐数组、json 逐值、png / csv / svg 逐字节
    python docs/probes/diff_runs.py 旧/islands_lod out/seed42/islands_lod    # 别的目录：所有文件，npz / json 按值（npz 里的 JSON 串去掉用时再比），其余逐字节
    python docs/probes/diff_runs.py A B --tol 1e-9                          # 浮点在相对 1e-9 内算相同（跨平台 libm 只差末位时用）

行星层跳过 ⑩（图的标题里有 run 名）、islands* 与 _meta.json、config.resolved.toml；json 里去掉用时、run 名、阶段 key 这类键
（SKIP_KEYS）；md 里把两边的 run 目录名换成同一个词再比。岛群产物跳过 town/（营建器的产物）。
浮点数组先比字节（±0、NaN 的位也算），不同再报个数、最大绝对差与相对差。退出码：相同 0，有不同 1。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SKIP_KEYS = {"elapsed_s", "seconds", "time_s", "timing", "run_id", "run", "engine", "key", "cache_key", "stage_key", "created", "backend"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in SKIP_KEYS}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff_npz(fa: Path, fb: Path, rel: str, tol: float) -> list[str]:
    bad = []
    za, zb = np.load(fa, allow_pickle=False), np.load(fb, allow_pickle=False)
    if sorted(za.files) != sorted(zb.files):
        bad.append(f"{rel} 键不同 {sorted(set(za.files) ^ set(zb.files))}")
    for k in za.files:
        if k not in zb.files:
            continue
        x, y = za[k], zb[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(f"{rel}:{k} 形 {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        elif x.dtype.kind in "fc":
            if np.array_equal(x.view(np.uint8), y.view(np.uint8)):
                continue
            xf, yf = x.astype(np.complex128 if x.dtype.kind == "c" else np.float64), y.astype(np.complex128 if y.dtype.kind == "c" else np.float64)
            with np.errstate(invalid="ignore"):
                d = np.abs(xf - yf)                  # inf − inf 是 NaN，下面按有限值算
            fin = np.isfinite(d)
            nan_mismatch = int((np.isnan(xf) != np.isnan(yf)).sum())
            dmax = float(d[fin].max()) if fin.any() else 0.0
            rel_d = d[fin] / np.maximum(np.abs(xf[fin]), 1e-300)
            rmax = float(rel_d.max()) if fin.any() else 0.0
            if tol > 0 and nan_mismatch == 0 and rmax <= tol:
                continue
            bad.append(f"{rel}:{k} 差 {int((x != y).sum())} / {x.size} 个，最大绝对差 {dmax:.3g}、相对差 {rmax:.3g}"
                       + (f"，NaN 位置不同 {nan_mismatch}" if nan_mismatch else ""))
        elif x.dtype.kind == "U" and x.ndim == 0 and str(x)[:1] in "{[":
            try:                                     # 粗版的 meta / weather_meta：JSON 串，按值比、去掉用时
                same = strip(json.loads(str(x))) == strip(json.loads(str(y)))
            except ValueError:
                same = str(x) == str(y)
            if not same:
                bad.append(f"{rel}:{k} JSON 不同")
        elif not np.array_equal(x, y):
            bad.append(f"{rel}:{k} 差 {int((x != y).sum())} / {x.size} 个")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="两份产物逐项对照（行星层 run 目录，或一个岛群的产物目录）")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.0, help="浮点数组的相对差在这以内算相同（默认 0 = 逐位）")
    ap.add_argument("--show", type=int, default=40, help="最多列几处不同")
    a_ = ap.parse_args(argv)
    a, b = Path(a_.a), Path(a_.b)
    island = (a / "island.json").exists()
    planet = not island and any(d.is_dir() and d.name.startswith("s0") for d in a.iterdir())
    bad, n = [], 0
    for fa in sorted(a.rglob("*")):
        if not fa.is_file():
            continue
        rel = fa.relative_to(a)
        top = rel.parts[0]
        if island:
            if top == "town":
                continue
        elif planet and ((not top.startswith("s0")) or fa.name == "_meta.json" or fa.suffix not in (".npz", ".json", ".md")):
            continue
        fb = b / rel
        if not fb.exists():
            bad.append(f"缺 {rel}")
            continue
        n += 1
        if fa.suffix == ".npz":
            bad += diff_npz(fa, fb, str(rel), a_.tol)
        elif fa.suffix == ".json":
            if strip(json.loads(fa.read_text("utf-8"))) != strip(json.loads(fb.read_text("utf-8"))):
                bad.append(f"{rel} json 不同")
        elif fa.suffix == ".md":
            ta = fa.read_text("utf-8").replace(a.name, "RUN")
            tb = fb.read_text("utf-8").replace(b.name, "RUN")
            if ta != tb:
                bad.append(f"{rel} 文本不同")
        elif fa.read_bytes() != fb.read_bytes():
            bad.append(f"{rel} 字节不同")
    def wanted(r: Path) -> bool:
        if island:
            return r.parts[0] != "town"
        if planet:
            return r.parts[0].startswith("s0") and r.suffix in (".npz", ".json", ".md") and r.name != "_meta.json"
        return True
    extra = [p.relative_to(b) for p in sorted(b.rglob("*")) if p.is_file() and not (a / p.relative_to(b)).exists() and wanted(p.relative_to(b))]
    bad += [f"多 {r}" for r in extra]
    print(f"{a} vs {b}（{'岛群产物' if island else '行星层 ①–⑨' if planet else '全部文件'}）：比了 {n} 个文件，{len(bad)} 处不同")
    for x in bad[:a_.show]:
        print("  ", x)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
