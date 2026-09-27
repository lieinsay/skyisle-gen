"""生成器后端开关与行星层的 C++ 桥（docs/PLAN-CORE.md 第六节；行星计划 P6c）。

`[engine] backend = "python" | "cpp"`（默认 python）。管线里已移到 C++ 的阶段（P6c：①–④）在 run(ctx) 第一行按它分派；
第三层（island/engine.py）的分派也读这里的 backend()。

cpp 后端下各步的产物是 C++ 的不透明对象（_core.PlanetParams / Winds / Islands / Climate），按（run 目录, 阶段, 阶段 key）缓存在进程内：
下一步直接吃上一步的对象（不经 npz）；缓存里没有（上游命中了磁盘缓存、或换了进程）就从该步的 npz / json 读回（_core.*_from）。
阶段 key 覆盖了配置、seed、上游与后端，key 相同的对象与磁盘上的产物是同一份，不会拿到旧的。
第三层（岛群生成器）在 cpp 后端下同样从这里取行星层（planet_parts → _core.planet_view / node_inputs），不再在 Python 里拼网格。

本模块不 import island（stages/ 会 import 它；第三层不回灌的静态断言照旧）。
"""
from __future__ import annotations

import json
from pathlib import Path

BACKENDS = ("python", "cpp")
CPP_STAGES = (1, 2, 3, 4)          # 已有 C++ 实现的管线阶段（P6c）；P6d 接着移 ⑤–⑨
PLANET_SECTIONS = ("shared", "skeleton", "s01", "s02", "s03", "s04")

_PARTS: dict = {}                  # (run 目录, 阶段, 阶段 key) → C++ 对象


def backend(cfg: dict) -> str:
    b = str((cfg.get("engine") or {}).get("backend", "python")).lower()
    if b not in BACKENDS:
        raise ValueError(f"[engine] backend 只能是 python 或 cpp，得到 {b!r}")
    return b


def core():
    try:
        from . import _core
    except ImportError as e:            # 不静默退回 Python：免得以为跑的是 C++
        raise RuntimeError("[engine] backend = \"cpp\"，但 C++ 扩展 skyisle_gen._core 没编：在仓库根下运行 "
                           "`python core/build.py`（见 README「C++ 核心库」）") from e
    return _core


def flatten(d: dict, prefix: str = "", key_map: dict | None = None) -> dict:
    """嵌套配置展平成 {"num": {"a.b": 1.0}, "vec": {"a.c": [..]}, "str": {"a.d": "x"}}（布尔 → 0 / 1；字符串列表跳过）。"""
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
    walk(d, prefix)
    return {"num": num, "vec": vec, "str": strs}


def planet_config(cfg: dict) -> dict:
    """行星层 ①–④ 读的配置段（shared / skeleton / s01–s04）展平；C++ 按 "s03.islands.n_islands" 这样的全路径取。"""
    return flatten({k: cfg[k] for k in PLANET_SECTIONS if k in cfg})


def cpp_key_suffix(cfg: dict, idx: int) -> str:
    """阶段缓存 key 的后端分量：python 后端为空（key 与 P6c 之前一字不差，旧 run 的缓存照旧命中）；cpp 后端下有 C++ 实现的阶段加 "+cpp"。"""
    return "+cpp" if idx in CPP_STAGES and backend(cfg) == "cpp" else ""


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
    raise ValueError(idx)


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
