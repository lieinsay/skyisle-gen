"""聚落营建器（docs/PLAN-TOWN.md）：给一块地形、一个规模、一种风格，营建出一个建筑群。

独立工具：只读岛群生成器的产物（地形地貌、水系、聚落点位、气候），不回写；stages/、island/ 等不得 import 本包（tests 有静态断言）。
算法在 C++ 核心库（core/include/skyisle/town/），这里只做前端：配置、裁窗口、调 _core、写产物、出图。
随机数：地面的噪声种子 = entity_rng(run seed, TOWN_STREAM, "town:{节点}:ground")（不含聚落名、不含风格：同一个地方在任何窗口、任何风格下都是同一块地）；
合成地形 = entity_rng(seed, TOWN_STREAM, "town:synth:{地形}")。
"""
from __future__ import annotations

import math
import tomllib

from ..config import CONFIG_DIR, apply_sets

TOWN_STREAM = 22   # 十步管线 1–10、岛群 21 之后
TOWN_DIR = CONFIG_DIR / "town"

# 命令行用中文名，C++ 用 ASCII 名
SCALES = {"宅院": "compound", "小庄": "hamlet", "村": "village", "集镇": "town", "城": "city", "专业聚落": "special"}
TERRAINS = {"平原": "plain", "曲流平原": "meander", "河谷": "valley", "朝阳坡": "sunslope", "山顶": "hilltop", "湖岸": "lakeshore",
            "峡湾岸": "fjord", "黄土沟": "gully", "两河交汇": "confluence", "岛缘崖台": "rim"}
SCALE_ZH = {v: k for k, v in SCALES.items()}
TERRAIN_ZH = {v: k for k, v in TERRAINS.items()}


def town_config(sets: list[str] | None = None) -> dict:
    """工具参数：config/town/town.toml ← --set town.键=值（与风格无关）。"""
    with open(TOWN_DIR / "town.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    if sets:
        tmp = {"town": cfg}
        apply_sets(tmp, [s for s in sets if s.startswith("town.")])
        cfg = tmp["town"]
    return cfg


def flat(cfg: dict) -> dict:
    """展平成 C++ 的 Config（engine.flatten：{"num": {"site.res_village_m": 1.0, …}, "vec": {…}, "str": {…}}，布尔 → 0 / 1）。"""
    from ..engine import flatten
    return flatten(cfg)


def scale_id(name: str) -> str:
    if name in SCALES.values():
        return name
    if name not in SCALES:
        raise ValueError(f"不认识的规模「{name}」：可选 {'、'.join(SCALES)}")
    return SCALES[name]


def terrain_id(name: str) -> str:
    if name in TERRAINS.values():
        return name
    if name not in TERRAINS:
        raise ValueError(f"不认识的合成地形「{name}」：可选 {'、'.join(TERRAINS)}")
    return TERRAINS[name]


def window_half_m(scale: str, households: int, cfg: dict) -> float:
    """窗口半边长（PLAN-TOWN 7.1）：spread × √(户 × 毛用地 / π) + margin，不小于该规模的下限。"""
    w = cfg["window"]
    if scale == "compound":
        return float(w["compound_half_m"])
    if scale == "hamlet":
        return float(w["hamlet_half_m"])
    key = {"village": "village", "special": "village", "town": "town", "city": "city"}[scale]
    m2, lo = float(w[f"{key}_m2_per_household"]), float(w[f"{key}_min_half_m"])
    return max(lo, float(w["spread"]) * math.sqrt(max(1, households) * m2 / math.pi) + float(w["margin_m"]))


def res_m(scale: str, cfg: dict) -> float:
    return float(cfg["site"]["res_town_m" if scale in ("town", "city") else "res_village_m"])


def core():
    from ..engine import core as _core
    return _core()


def ground_seed(seed: int, key: str) -> int:
    """地面噪声的种子：entity_rng(seed, TOWN_STREAM, key) 的第一个 64 位数。"""
    return int(core().rng_raw(int(seed), TOWN_STREAM, key, 1)[0])
