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
    # ---- 行星尺度、岛群陆地与尺度口径 ----
    p1 = cfg.get("s01", {}).get("planet", {})
    if "radius_km" in p1 and float(p1["radius_km"]) <= 0:
        raise ValueError("s01.planet.radius_km 必须 > 0（默认 6371 = 地球大小）")
    if float(cfg.get("shared", {}).get("day_range_km", 1.0)) <= 0:
        raise ValueError("shared.day_range_km 必须 > 0")
    s3 = cfg.get("s03", {}).get("islands", {})
    if "land_frac_alpha" in s3 and float(s3["land_frac_alpha"]) < 0:
        raise ValueError(
            "s03.islands.land_frac_alpha 必须 ≥ 0（负值 = 陆地占比随密度下降，与现实群岛相悖）")
    if "land_frac_cap" in s3 and not (0.0 < float(s3["land_frac_cap"]) <= 1.0):
        raise ValueError("s03.islands.land_frac_cap 必须在 (0, 1]（陆地不能超过势力范围）")
    if "land_frac_sigma" in s3 and float(s3["land_frac_sigma"]) < 0:
        raise ValueError("s03.islands.land_frac_sigma 必须 ≥ 0")
    if "arable_frac_sigma" in s3 and float(s3["arable_frac_sigma"]) < 0:
        raise ValueError("s03.islands.arable_frac_sigma 必须 ≥ 0")
    if "arable_frac_range" in s3:
        lo, hi = (float(x) for x in s3["arable_frac_range"])
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError("s03.islands.arable_frac_range 必须满足 0 < lo ≤ hi ≤ 1")
    sc = cfg.get("shared", {}).get("scale", {})
    if "total_land_km2" in sc and float(sc["total_land_km2"]) <= 0:
        raise ValueError("shared.scale.total_land_km2 必须 > 0")
    if "arable_frac_mean" in sc and not (0.0 < float(sc["arable_frac_mean"]) <= 1.0):
        raise ValueError("shared.scale.arable_frac_mean 必须在 (0, 1]")
    if "people_per_arable_km2" in sc and float(sc["people_per_arable_km2"]) <= 0:
        raise ValueError("shared.scale.people_per_arable_km2 必须 > 0")
    c7 = cfg.get("s07", {}).get("centers", {})
    if "area_exponent" in c7 and float(c7["area_exponent"]) < 0:
        raise ValueError("s07.centers.area_exponent 必须 ≥ 0")
    # ---- 主岛 / 岛体 / 河流（第三批第 1 步）----
    if "main_frac_range" in s3:
        lo, hi = (float(x) for x in s3["main_frac_range"])
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError("s03.islands.main_frac_range 必须满足 0 < lo ≤ hi ≤ 1（主岛不能大于群）")
    if "main_frac_sigma" in s3 and float(s3["main_frac_sigma"]) < 0:
        raise ValueError("s03.islands.main_frac_sigma 必须 ≥ 0")
    if "keel_clearance_m" in s3 and float(s3["keel_clearance_m"]) < 0:
        raise ValueError("s03.islands.keel_clearance_m 必须 ≥ 0")
    pl = cfg.get("s03", {}).get("plates", {})
    if "n_plates" in pl and int(pl["n_plates"]) < 4:
        raise ValueError("s03.plates.n_plates 必须 ≥ 4")
    if "p_convergent" in pl and "p_divergent" in pl:
        if not (0.0 <= float(pl["p_convergent"]) and 0.0 <= float(pl["p_divergent"])
                and float(pl["p_convergent"]) + float(pl["p_divergent"]) <= 1.0):
            raise ValueError("s03.plates.p_convergent + p_divergent 必须在 [0, 1]")
    if "divergent_cut" in pl and not (0.0 <= float(pl["divergent_cut"]) <= 1.0):
        raise ValueError("s03.plates.divergent_cut 必须在 [0, 1]")
    if "height_age_decay" in pl and not (0.0 <= float(pl["height_age_decay"]) < 1.0):
        raise ValueError("s03.plates.height_age_decay 必须在 [0, 1)")
    c4 = cfg.get("s04", {}).get("climate", {})
    for k in ("river_main_area_km2", "river_height_m", "river_capacity_bonus"):
        if k in c4 and float(c4[k]) < 0:
            raise ValueError(f"s04.climate.{k} 必须 ≥ 0")
    if "river_precip_min" in c4 and not (0.0 <= float(c4["river_precip_min"]) <= 1.0):
        raise ValueError("s04.climate.river_precip_min 必须在 [0, 1]")
    # ---- 水汽模型与岛对风的扰动（第三批 3、4）----
    if "moisture_tau_days" in c4 and float(c4["moisture_tau_days"]) <= 0:
        raise ValueError("s04.climate.moisture_tau_days 必须 > 0")
    if "moisture_res_deg" in c4:
        g = float(cfg.get("shared", {}).get("grid_res_deg", 1.0))
        r = float(c4["moisture_res_deg"])
        if r < g or abs(r / g - round(r / g)) > 1e-9:
            raise ValueError("s04.climate.moisture_res_deg 必须是 shared.grid_res_deg 的整数倍")
    if "moisture_polar_filter_lat" in c4 and not (0.0 < float(c4["moisture_polar_filter_lat"]) < 90.0):
        raise ValueError("s04.climate.moisture_polar_filter_lat 必须在 (0, 90)")
    lwc = cfg.get("s04", {}).get("localwind", {})
    for k in ("friction_k", "wake_k", "storm_k"):
        if k in lwc and not (0.0 <= float(lwc[k]) < 1.0):
            raise ValueError(f"s04.localwind.{k} 必须在 [0, 1)（风速与风暴不能被岛减到负）")
    if "shift_max_deg" in lwc and not (0.0 <= float(lwc["shift_max_deg"]) < 4.0):
        raise ValueError("s04.localwind.shift_max_deg 必须在 [0, 4)（无风带 8° 宽，两侧带界不能交叉）")


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
