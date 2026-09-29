"""营建方案的前端（PLAN-TOWN 第七节）：地面 + 风格 + 上游锚点 → _core.town_plan → 方案 dict。

随机：方案的种子 = entity_rng(seed, TOWN_STREAM, "town:{节点}:{聚落}:{风格}:plan")（合成地形 "town:synth:{地形}:{风格}:plan"）——
与地面的种子分开，换风格地面不变；同一个地方同一种风格重跑逐字节相同。
"""
from __future__ import annotations

from pathlib import Path

from . import TOWN_STREAM, core
from .style import operator_name, style_flat

ROAD_CLASSES = ["trunk", "main", "street", "lane", "path"]
ROAD_ZH = {"trunk": "出村大路", "main": "主街", "street": "街", "lane": "巷", "path": "田间道"}
SIDE_ZH = ["前", "后", "左", "右"]
HH_KIND = ["农户", "市户", "专业户"]
PLAN_SCALES = ("compound", "hamlet", "village")   # 集镇 / 城在第四步，专业聚落在第六步


def plan_seed(meta: dict, style_id: str) -> int:
    if meta["source"] == "island":
        seed, key = meta["seed"], f"town:{meta['node']}:{meta['site']}:{style_id}:plan"
    elif meta["source"] == "synth":
        seed, key = meta["seed"], f"town:synth:{meta['terrain']}:{style_id}:plan"
    else:
        seed, key = 0, f"town:heightmap:{Path(meta['file']).name}:{style_id}:plan"
    return int(core().rng_raw(int(seed), TOWN_STREAM, key, 1)[0])


def plan_request(meta: dict, st: dict, operator: str | None = None) -> dict:
    anchors = meta.get("anchors") or {}
    wind = (meta.get("winter") or {}).get("from_deg")
    return {
        "scale": meta["scale"], "hh_farm": int(meta["households_farm"]), "hh_market": int(meta["households_market"]), "hh_special": 0,
        "seed": plan_seed(meta, st["meta"]["id"]), "anchor": [0.0, 0.0],
        "landings": anchors.get("landings", []), "exits": anchors.get("exits", []),
        "wind_from_deg": float(wind) if wind is not None else None, "want_landing": True, "operator": operator or "",
    }


def make_plan(sd: dict, meta: dict, st: dict, operator: str | None = None) -> dict:
    if meta["scale"] not in PLAN_SCALES:
        from . import SCALE_ZH
        raise SystemExit(f"规模「{SCALE_ZH.get(meta['scale'], meta['scale'])}」的营建在 PLAN-TOWN 第四 / 六步才做；现在能营建：宅院、小庄、村")
    if operator and operator not in st["village"]["operators"]:
        raise SystemExit(f"风格 {st['meta']['name']} 没有形态算子 {operator}（有：{'、'.join(st['village']['operators'])}）")
    req = plan_request(meta, st, operator)
    P = core().town_plan(sd, req, style_flat(st))
    P["op_name"] = "单个宅院" if P["op"] == "single" else operator_name(st, P["op"])
    P["request"] = {k: v for k, v in req.items() if k != "seed"} | {"seed": str(req["seed"])}
    return P


def hard_failures(P: dict) -> list[dict]:
    return [c for c in P["checks"] if c["hard"] and not c["ok"]]
