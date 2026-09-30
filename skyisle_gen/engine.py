"""行星层的 C++ 桥（docs/PLAN-CORE.md 第六节；行星计划 P6c / P6d）。

①–⑨ 与第三层的算法只在 C++ 核心（skyisle_gen._core）里；Python 参考后端 2026-09-30 删了（git tag python-reference-final，
DESIGN-NOTES 四点四十二）。各步的 run(ctx) 调 C++、再由共用的 _write 写 npz / json 与摘要；⑩ 输出（九格表、出图）只在 Python。

各步的产物是 C++ 的不透明对象（_core.PlanetParams / Winds / Islands / Climate / Barriers / Routes / Centers / Diffusion / Polity），
按（run 目录, 阶段, 阶段 key）缓存在进程内：
下一步直接吃上一步的对象（不经 npz）；缓存里没有（上游命中了磁盘缓存、或换了进程）就从该步的 npz / json 读回（_core.*_from）。
阶段 key 覆盖了配置、seed 与上游，key 相同的对象与磁盘上的产物是同一份，不会拿到旧的。
第三层（岛群生成器）同样从这里取行星层（planet_parts → _core.planet_view / node_inputs；⑨ 的人口与邦都 → _core.node_polity），
不再在 Python 里拼网格、也不再读 polity.npz。

本模块不 import island（stages/ 会 import 它；第三层不回灌的静态断言照旧）。
"""
from __future__ import annotations

import json
from pathlib import Path

CPP_STAGES = (1, 2, 3, 4, 5, 6, 7, 8, 9)   # 有 C++ 实现的管线阶段（P6c：①–④，P6d：⑤–⑨）；⑩ 输出（九格表、出图）只在 Python
PLANET_SECTIONS = ("shared", "skeleton", "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09",
                   "slots", "traits_manual")   # ⑧ 的槽位表（slots.toml）与手工特征表（traits.toml，可缺）也进 C++

_PARTS: dict = {}                  # (run 目录, 阶段, 阶段 key) → C++ 对象


def core():
    try:
        from . import _core
    except ImportError as e:            # 算法只在 C++ 里（Python 参考后端已删），没编就跑不了
        raise RuntimeError("C++ 扩展 skyisle_gen._core 没编：在仓库根下运行 "
                           "`python core/build.py`（见 README「C++ 核心库」）") from e
    return _core


def flatten(d: dict, prefix: str = "", key_map: dict | None = None) -> dict:
    """嵌套配置展平成 {"num": {"a.b": 1.0}, "vec": {"a.c": [..]}, "str": {"a.d": "x"}}（布尔 → 0 / 1；字符串列表跳过；
    表的数组展成 "a.n" 与 "a.<i>.<子键>"）。"""
    num, vec, strs = {}, {}, {}

    def walk(x, pre):
        for k, v in x.items():
            key = f"{pre}{(key_map or {}).get(k, k)}"
            if isinstance(v, dict):
                walk(v, key + ".")
            elif isinstance(v, bool):
                num[key] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                num[key] = float(v)
            elif isinstance(v, str):
                strs[key] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v):
                vec[key] = [float(e) for e in v]
            elif isinstance(v, (list, tuple)) and v and all(isinstance(e, dict) for e in v):
                # 表的数组（slots.slot、s05.political.overrides、traits_manual.trait）：key.n = 个数，各项展成 key.<i>.<子键>
                num[key + ".n"] = float(len(v))
                for i, e in enumerate(v):
                    walk(e, f"{key}.{i}.")
    walk(d, prefix)
    return {"num": num, "vec": vec, "str": strs}


def planet_config(cfg: dict) -> dict:
    """行星层 ①–⑨ 读的配置段（shared / skeleton / s01–s09、槽位表、手工特征表）展平；C++ 按 "s03.islands.n_islands" 这样的全路径取。"""
    return flatten({k: cfg[k] for k in PLANET_SECTIONS if k in cfg})


def cpp_key_suffix(cfg: dict, idx: int) -> str:
    """阶段缓存 key 的后端分量：①–⑨ 固定加 "+cpp"（沿用 P6c / P6d 的 cpp 后端 key，删 Python 参考后端前的 cpp run 照旧命中）。"""
    return "+cpp" if idx in CPP_STAGES else ""


# ---------------------------------------------------------------- 各步产物的 C++ 对象：进程内缓存 + 从磁盘读回
def _stage_key(ctx, idx: int) -> str | None:
    keys = getattr(ctx, "stage_keys", None)
    if keys:
        return keys[idx - 1]
    p = ctx.stage_dir(idx) / "_meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("stage_key")
    return None


def _slot(ctx, idx: int):
    return (str(Path(ctx.out_dir).resolve()), idx, _stage_key(ctx, idx))


def put_part(ctx, idx: int, obj) -> None:
    run = str(Path(ctx.out_dir).resolve())
    for k in [k for k in _PARTS if k[0] == run and k[1] == idx]:
        del _PARTS[k]
    _PARTS[_slot(ctx, idx)] = obj


def _load_part(ctx, idx: int):
    c = core()
    if idx == 1:
        return c.planet_from_json(ctx.load_json(1, "planet"))
    if idx == 2:
        return c.winds_from(ctx.load_npz(2, "wind"), ctx.load_json(2, "bands"))
    if idx == 3:
        return c.islands_from(ctx.load_npz(3, "islands"), ctx.load_npz(3, "plates"), ctx.load_npz(3, "cand_edges"),
                              ctx.load_npz(3, "density_grid"))
    if idx == 4:
        return c.climate_from(ctx.load_npz(4, "wind_local"), ctx.load_npz(4, "band_local"), ctx.load_npz(4, "climate_grid"),
                              ctx.load_npz(4, "climate_islands"))
    if idx == 5:
        return c.barriers_from(ctx.load_npz(5, "perm"))
    if idx == 6:
        return c.routes_from(ctx.load_npz(6, "routes"), ctx.load_json(6, "hubs"))
    if idx == 7:
        return c.centers_from(ctx.load_json(7, "centers"), ctx.load_npz(7, "prehist"), ctx.load_npz(7, "regions"))
    if idx == 9:
        return c.polity_from(ctx.load_npz(9, "polity"), ctx.load_json(9, "polities"))
    # ⑧ 不从产物读回：管线里没有下游读它（⑨ 与第三层都不读 ⑧），⑩ 与 check 读的是 npz
    raise ValueError(f"第 {idx} 步的 C++ 对象不从产物读回")


def part(ctx, idx: int):
    """第 idx 步产物的 C++ 对象：进程内有同 key 的就直接给，否则从该步的 npz / json 读回（并缓存）。"""
    slot = _slot(ctx, idx)
    obj = _PARTS.get(slot)
    if obj is None:
        obj = _load_part(ctx, idx)
        put_part(ctx, idx, obj)
    return obj


def planet_parts(ctx):
    """第三层要的行星层：(① PlanetParams, ③ Islands, ④ Climate)。"""
    return part(ctx, 1), part(ctx, 3), part(ctx, 4)


def clear_cache() -> None:
    _PARTS.clear()
