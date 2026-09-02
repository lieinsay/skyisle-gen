"""阶段框架：注册、缓存 key 链、产物 IO、run/stage 执行。

缓存规则：key_k = sha256(key_{k-1} ‖ config[s0k] ‖ config[shared] ‖ config[skeleton] ‖ seed ‖ STAGE_VERSION)。
命中（_meta.json 的 stage_key 相同）则跳过；任一 miss，其后全部重算。
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import __version__
from .config import canonical, dump_toml, section_hash

STAGES = [
    (1, "s01_planet"),
    (2, "s02_wind"),
    (3, "s03_islands"),
    (4, "s04_climate"),
    (5, "s05_barriers"),
    (6, "s06_routes"),
    (7, "s07_centers"),
    (8, "s08_diffusion"),
    (9, "s09_output"),
]

# 改动会使缓存失效的实现版本号（每阶段独立）
STAGE_VERSIONS = {i: "1" for i, _ in STAGES}
STAGE_VERSIONS[3] = "5"  # D 域下界=赤道核心边缘；G 邻域几何弦；G 盘/赤道核心解析排除
STAGE_VERSIONS[4] = "2"  # storm_no_g
STAGE_VERSIONS[5] = "3"  # perm_no_g；D 的 Φ 域与 s03 对齐
STAGE_VERSIONS[6] = "3"  # betweenness_sources；cost_no_g


class Context:
    """传给每个阶段的上下文：配置、seed、产物读写。"""

    def __init__(self, cfg: dict, seed: int, out_dir: Path):
        self.cfg = cfg
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.summaries: dict[str, dict] = {}

    def stage_dir(self, idx: int) -> Path:
        name = dict(STAGES)[idx]
        d = self.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- 产物 IO ----
    def save_npz(self, idx: int, name: str, **arrays) -> None:
        np.savez_compressed(self.stage_dir(idx) / f"{name}.npz", **arrays)

    def load_npz(self, idx: int, name: str) -> dict:
        with np.load(self.stage_dir(idx) / f"{name}.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save_json(self, idx: int, name: str, obj) -> None:
        p = self.stage_dir(idx) / f"{name}.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1),
                     encoding="utf-8")

    def load_json(self, idx: int, name: str):
        p = self.stage_dir(idx) / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def section(self, idx: int) -> dict:
        return self.cfg.get(f"s{idx:02d}", {})


def _stage_key_chain(cfg: dict, seed: int) -> list[str]:
    keys = []
    prev = "root"
    for idx, _name in STAGES:
        prev = section_hash(prev, cfg, f"s{idx:02d}", seed, STAGE_VERSIONS[idx])
        keys.append(prev)
    return keys


def run(cfg: dict, seed: int, out_root: Path, upto: int = 9,
        force_from: int | None = None, explain: bool = False, log=print) -> Path:
    run_id = str(cfg.get("run", {}).get("id") or f"seed{seed}")
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.toml").write_text(dump_toml(cfg), encoding="utf-8")

    ctx = Context(cfg, seed, out_dir)
    keys = _stage_key_chain(cfg, seed)
    from .stages import (s01_planet, s02_wind, s03_islands, s04_climate, s05_barriers,
                         s06_routes, s07_centers, s08_diffusion, s09_output)
    impls = {1: s01_planet, 2: s02_wind, 3: s03_islands, 4: s04_climate, 5: s05_barriers,
             6: s06_routes, 7: s07_centers, 8: s08_diffusion, 9: s09_output}

    chain_broken = False
    for idx, name in STAGES:
        if idx > upto:
            break
        sd = ctx.stage_dir(idx)
        meta_p = sd / "_meta.json"
        key = keys[idx - 1]
        if force_from is not None and idx >= force_from:
            chain_broken = True
        hit = False
        if meta_p.exists() and not chain_broken:
            hit = json.loads(meta_p.read_text(encoding="utf-8")).get("stage_key") == key
        if hit:
            if explain:
                log(f"[{name}] cache hit ({key[:8]})")
            continue
        chain_broken = True
        t0 = time.perf_counter()
        log(f"[{name}] running …")
        summary = impls[idx].run(ctx) or {}
        dt = time.perf_counter() - t0
        meta = {
            "stage_key": key,
            "upstream_key": keys[idx - 2] if idx > 1 else "root",
            "stage_version": STAGE_VERSIONS[idx],
            "generator_version": __version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "seed": seed,
            "seconds": round(dt, 3),
            "summary": summary,
        }
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1),
                          encoding="utf-8")
        line = "; ".join(f"{k}={v}" for k, v in summary.items() if not isinstance(v, (dict, list)))
        log(f"[{name}] done in {dt:.1f}s  {line}")

    manifest = {
        "run_id": run_id,
        "seed": seed,
        "generator_version": __version__,
        "numpy_version": np.__version__,
        "stage_keys": {name: keys[i - 1] for i, name in STAGES if i <= upto},
        "config_hash": section_hash("root", cfg, "__all__", seed, canonical(sorted(cfg.keys()))),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    return out_dir
