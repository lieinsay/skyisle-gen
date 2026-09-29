"""风格（PLAN-TOWN 第六节）：base ← 内置风格 ← --set style.键=值，深合并；模板与功能整理成数组后展平给 C++（键前缀 style.）。

找风格：id（huabei）、中文名（华北集村）或一个 .toml 路径（自定义风格的 extends 链往后放，6.7）。
功能表：config/town/functions.toml 是通用目录，风格的 [functions.<id>] 逐键覆盖；只把 enabled 的交给 C++。
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from ..config import _deep_merge, apply_sets
from . import TOWN_DIR

STYLE_DIR = TOWN_DIR / "styles"
# 形态算子（PLAN-TOWN 6.2）与默认的中文名；风格可在 [village.operator_names] 里改叫法（子预设）
OPERATORS = {"fishbone": "鱼骨街村", "organic": "团块生长", "street_village": "街村", "hufen": "林地排村", "waterfront": "滨水",
             "dispersed": "散居", "green": "围绿", "comb": "梳式", "contour": "等高线", "enclosure": "围合单体"}
FUNCTIONS_FILE = TOWN_DIR / "functions.toml"


def _load(p: Path) -> dict:
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def list_styles() -> list[dict]:
    """内置风格（base 除外）：[{id, name, region, file}]，按文件名排。"""
    out = []
    for p in sorted(STYLE_DIR.glob("*.toml")):
        if p.stem == "base":
            continue
        m = _load(p).get("meta", {})
        out.append({"id": m.get("id", p.stem), "name": m.get("name", p.stem), "region": m.get("region", ""), "file": p})
    return out


def style_file(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".toml" and p.exists():
        return p
    for s in list_styles():
        if name in (s["id"], s["name"]):
            return s["file"]
    raise ValueError(f"不认识的风格「{name}」：内置的有 {'、'.join(s['name'] + '（' + s['id'] + '）' for s in list_styles())}；"
                     f"自定义风格给 .toml 路径")


def load_style(name: str, sets: list[str] | None = None) -> dict:
    """解析后的风格表（base ← 风格 ← --set style.*），外加 functions（通用目录 ← 风格覆盖）。"""
    f = style_file(name)
    st = _deep_merge(_load(STYLE_DIR / "base.toml"), _load(f))
    if sets:
        tmp = {"style": st}
        apply_sets(tmp, [s for s in sets if s.startswith("style.")])
        st = tmp["style"]
    funcs = _load(FUNCTIONS_FILE)
    st["functions"] = {k: _deep_merge(v, st.get("functions", {}).get(k, {})) for k, v in funcs.items()}
    for k, v in st.get("functions", {}).items():
        if k not in funcs:
            raise ValueError(f"风格 {f.name} 的 [functions.{k}] 不在通用功能目录（config/town/functions.toml）里")
    _check(st, f)
    return st


def _check(st: dict, f: Path) -> None:
    """最基本的自检（style check 的完整版往后放）：模板、功能引用的模板、区间不倒置、形态算子认识。"""
    tpls = st.get("compound", {}).get("templates", {})
    if not any(t.get("weight", 0) > 0 for t in tpls.values()):
        raise ValueError(f"风格 {f.name} 没有住宅模板（[compound.templates.<id>] 且 weight > 0）")
    for k, fn in st["functions"].items():
        if fn.get("enabled") and fn.get("mode") == "compound" and fn.get("template") not in tpls:
            raise ValueError(f"功能 {k} 引用的模板 {fn.get('template')} 不存在")
    ops = st.get("village", {}).get("operators", {})
    bad = [k for k, w in ops.items() if w > 0 and k not in OPERATORS]
    if bad:
        raise ValueError(f"风格 {f.name} 的形态算子 {bad} 不认识（有：{'、'.join(OPERATORS)}）")

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list) and len(x) == 2 and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in x):
            if x[0] > x[1]:
                raise ValueError(f"风格 {f.name}：区间 {path} = {x} 倒置")
    walk(st, "")


def style_names(st: dict) -> dict:
    """前端出图、写 plan.json 用的显示名：模板、功能。"""
    return {"templates": {k: v.get("name", k) for k, v in st["compound"]["templates"].items()},
            "functions": {k: v.get("name", k) for k, v in st["functions"].items()}}


def _join_ops(d: dict) -> dict:
    """字符串列表（ops = ["contour"]）展平时会被跳过：拼成逗号串交给 C++。"""
    if isinstance(d.get("ops"), list):
        d = {**d, "ops": ",".join(d["ops"])}
    return d


def operator_name(st: dict, op: str) -> str:
    """形态算子在这个风格里的叫法（子预设），没写就用默认中文名。"""
    return (st.get("village", {}).get("operator_names", {}) or {}).get(op) or OPERATORS.get(op, op)


def style_flat(st: dict) -> dict:
    """展平给 C++：模板字典 → compound.template 数组（带 id），启用的功能 → func 数组（带 id）；前缀 style.。"""
    from ..engine import flatten
    s = {k: v for k, v in st.items() if k not in ("functions", "meta")}
    comp = dict(s.get("compound", {}))
    tpls = comp.pop("templates", {})
    comp["template"] = [_join_ops({"id": k, **v}) for k, v in tpls.items()]
    s["compound"] = comp
    s["func"] = [_join_ops({"id": k, **v}) for k, v in st["functions"].items() if v.get("enabled")]
    s["meta"] = {"id": st["meta"]["id"]}
    return flatten(s, prefix="style.")


def style_hash(st: dict) -> str:
    return hashlib.sha1(json.dumps(st, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- style.resolved.toml
def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(e) for e in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_toml_value(e)}" for k, e in v.items()) + " }"
    raise TypeError(type(v))


def _key(k: str) -> str:
    return k if k.replace("_", "").replace("-", "").isascii() and k.replace("_", "").replace("-", "").isalnum() else json.dumps(k, ensure_ascii=False)


def dump_toml(d: dict) -> str:
    """够用的 TOML 输出：表、表的数组、数 / 串 / 布尔 / 列表；只含标量的小表写成行内表。"""
    lines: list[str] = []

    def is_aot(v):
        return isinstance(v, list) and v and all(isinstance(e, dict) for e in v)

    def emit(t: dict, path: list[str]):
        scal = {k: v for k, v in t.items() if not (isinstance(v, dict) and _has_table(v)) and not is_aot(v)}
        for k, v in scal.items():
            lines.append(f"{_key(k)} = {_toml_value(v)}")
        for k, v in t.items():
            if isinstance(v, dict) and _has_table(v):
                lines.append("")
                lines.append("[" + ".".join(_key(p) for p in path + [k]) + "]")
                emit(v, path + [k])
            elif is_aot(v):
                for e in v:
                    lines.append("")
                    lines.append("[[" + ".".join(_key(p) for p in path + [k]) + "]]")
                    emit(e, path + [k])

    def _has_table(v: dict) -> bool:
        return len(v) > 6 or any(isinstance(e, dict) or is_aot(e) for e in v.values())

    emit(d, [])
    return "\n".join(lines).lstrip("\n") + "\n"
