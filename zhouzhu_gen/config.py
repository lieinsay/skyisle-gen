"""配置：TOML 加载、深合并、--set 覆盖、校验、canonical 序列化、resolved 写出。"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from . import MODES

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(s: str) -> Any:
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def apply_sets(cfg: dict, sets: list[str]) -> dict:
    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set 需要 a.b.c=value 形式，得到：{item}")
        path, _, raw = item.partition("=")
        keys = path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set 路径 {path} 中 {k} 不是表")
        node[keys[-1]] = _parse_value(raw.strip())
    return cfg


def load_config(paths: list[Path] | None = None, sets: list[str] | None = None) -> dict:
    files = [CONFIG_DIR / "default.toml"]
    if paths:
        files += [Path(p) for p in paths]
    cfg: dict = {}
    for f in files:
        with open(f, "rb") as fh:
            cfg = _deep_merge(cfg, tomllib.load(fh))
    # slots / production templates 独立文件
    for name in ("slots", "production_templates"):
        f = CONFIG_DIR / f"{name}.toml"
        if f.exists():
            with open(f, "rb") as fh:
                cfg.setdefault(name, {})
                cfg[name] = _deep_merge(cfg[name], tomllib.load(fh))
    traits_f = CONFIG_DIR / "traits.toml"
    if traits_f.exists():
        with open(traits_f, "rb") as fh:
            cfg["traits_manual"] = tomllib.load(fh)
    if sets:
        apply_sets(cfg, sets)
    validate(cfg)
    return cfg


def _check_perm_table(name: str, tab: dict) -> None:
    for m in MODES:
        if m not in tab:
            raise ValueError(f"{name} 缺少模式 {m} 的通过率")
        v = tab[m]
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{name}.{m} = {v} 不在 [0,1]")


def validate(cfg: dict) -> None:
    for bid, b in cfg.get("s05", {}).get("barriers", {}).items():
        if "permeability" in b:
            _check_perm_table(f"s05.barriers.{bid}", b["permeability"])
    for fid, f in cfg.get("s05", {}).get("local", {}).items():
        if isinstance(f, dict) and "permeability" in f:
            _check_perm_table(f"s05.local.{fid}", f["permeability"])
    d8 = cfg.get("s08", {})
    r_max = float(d8.get("resistance_max", 0.98))
    for cat, rng in d8.get("resistance_range", {}).items():
        lo, hi = float(rng[0]), float(rng[1])
        if not (0.0 <= lo <= hi <= r_max):
            raise ValueError(f"s08.resistance_range.{cat} = {rng} 超出 [0, {r_max}]")
    dh = d8.get("half_distance_days", {})
    for m in MODES:
        if m in dh:
            lo, hi = float(dh[m][0]), float(dh[m][1])
            if not (0.0 < lo <= hi):
                raise ValueError(f"s08.half_distance_days.{m} 非法：{dh[m]}")
    # λ 排序约束：λ = ln2/d_half，即半衰距离 envoy ≥ migrate ≥ trade ≥ daily（几何均值比较）
    if all(m in dh for m in MODES):
        import math
        gm = {m: math.sqrt(float(dh[m][0]) * float(dh[m][1])) for m in MODES}
        if not (gm["daily"] <= gm["trade"] <= gm["migrate"] <= gm["envoy"]):
            raise ValueError(
                f"半衰距离几何均值须满足 daily ≤ trade ≤ migrate ≤ envoy，得到 {gm}")
    if "eps0" in d8 and float(d8["eps0"]) <= 0:
        raise ValueError("s08.eps0 必须 > 0（每槽位本地行保证处处有文化，原则己）")
    # ---- 行星尺度与岛屿规模 ----
    p1 = cfg.get("s01", {}).get("planet", {})
    if "radius_km" in p1 and float(p1["radius_km"]) <= 0:
        raise ValueError("s01.planet.radius_km 必须 > 0（默认 6371 = 地球大小）")
    if float(cfg.get("shared", {}).get("day_range_km", 1.0)) <= 0:
        raise ValueError("shared.day_range_km 必须 > 0")
    s3 = cfg.get("s03", {}).get("islands", {})
    if "area_lognorm_sigma" in s3 and float(s3["area_lognorm_sigma"]) <= 0:
        raise ValueError("s03.islands.area_lognorm_sigma 必须 > 0")
    if "area_density_beta" in s3 and float(s3["area_density_beta"]) < 0:
        raise ValueError(
            "s03.islands.area_density_beta 必须 ≥ 0（负值会让密接区长出巨岛，与 docs/02 §七 相悖）")
    c7 = cfg.get("s07", {}).get("centers", {})
    if "area_exponent" in c7 and float(c7["area_exponent"]) < 0:
        raise ValueError("s07.centers.area_exponent 必须 ≥ 0")


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def section_hash(prev_key: str, cfg: dict, section: str, seed: int, stage_version: str,
                 extra_sections: tuple[str, ...] = ()) -> str:
    """阶段缓存 key。extra_sections 用于 default.toml 之外的独立配置文件
    （slots.toml / traits.toml / production_templates.toml）——它们只影响读它们的阶段，
    不放进所有阶段的 key，避免过度失效。"""
    payload = "\x1f".join([
        prev_key,
        canonical(cfg.get(section, {})),
        canonical(cfg.get("shared", {})),
        canonical(cfg.get("skeleton", {})),
        *[canonical(cfg.get(s, {})) for s in extra_sections],
        str(seed),
        stage_version,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_toml(obj: dict, indent_path: str = "") -> str:
    """极简 TOML 写出（str/int/float/bool/list/dict）。仅用于 config.resolved.toml。"""
    lines: list[str] = []
    scalars = {k: v for k, v in obj.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in obj.items() if isinstance(v, dict)}

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{k} = {fmt(x)}" for k, x in v.items()) + " }"
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return json.dumps(v, ensure_ascii=False)

    for k, v in scalars.items():
        key = k if k.replace("_", "").replace("-", "").isalnum() else json.dumps(k, ensure_ascii=False)
        lines.append(f"{key} = {fmt(v)}")
    for k, v in tables.items():
        path = f"{indent_path}.{k}" if indent_path else k
        lines.append("")
        lines.append(f"[{path}]")
        lines.append(dump_toml(v, path))
    return "\n".join(lines)
