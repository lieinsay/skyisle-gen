"""岛群生成器（第三层，docs/PLAN-ISLAND.md）：静态隔离断言、确定性、约束一致性。小世界只跑到 ④。"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zhouzhu_gen.config import load_config
from zhouzhu_gen.pipeline import Context, run

PKG = Path(__file__).resolve().parent.parent / "zhouzhu_gen"
SMALL = ["s03.islands.n_islands=1600"]
STEPS = 2   # 开发中：已实现到第几步


# ---------------- IS-iso：管线不得读岛群生成器（第三层不回灌） ----------------
def test_stages_do_not_import_island():
    for f in sorted((PKG / "stages").glob("*.py")) + [PKG / "check.py", PKG / "ninegrid.py", PKG / "polity.py", PKG / "culture.py"]:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\.*\s*(zhouzhu_gen\.)?island\b", text, re.M), f"{f.name} import 了岛群生成器"
        assert "island_config" not in text and "build_terrain" not in text, f"{f.name} 用了岛群生成器"


def test_island_config_section_present():
    cfg = load_config()
    assert "island" in cfg and "layout" in cfg["island"] and "terrain" in cfg["island"]


@pytest.fixture(scope="module")
def small_ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("isl")
    cfg = load_config(sets=SMALL + ["run.id=isl"])
    out = run(cfg, 7, root, upto=4)
    return Context(cfg, 7, out)


def _pick_node(ctx):
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    # 有河、主岛不太大（测试快）
    ok = np.where(cli["has_river"] & (isl["main_area_km2"] < 1500) & (isl["main_area_km2"] > 300))[0]
    return int(ok[0]) if ok.size else 0


def _hash_dir(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.suffix in (".npz", ".png", ".csv")}


def test_island_deterministic_and_consistent(small_ctx):
    from zhouzhu_gen import island as isl
    node = _pick_node(small_ctx)
    out = isl.generate(small_ctx, node, res_m=300.0, steps=STEPS, log=lambda *a: None)
    h1 = _hash_dir(out)
    J1 = json.loads((out / "island.json").read_text(encoding="utf-8"))
    out2 = isl.generate(small_ctx, node, res_m=300.0, steps=STEPS, log=lambda *a: None)
    h2 = _hash_dir(out2)
    assert h1 == h2, "同输入重跑两次，产物不一致（IS-det）"
    c = J1["constraints"]
    assert abs(c["area_km2"]["actual"] - c["area_km2"]["target"]) <= 0.02 * c["area_km2"]["target"]
    assert abs(c["main_area_km2"]["actual"] - c["main_area_km2"]["target"]) <= 0.02 * c["main_area_km2"]["target"]
    assert abs(c["height_m"]["actual"] - c["height_m"]["target"]) <= 0.01 * c["height_m"]["target"]
    if STEPS >= 2:
        assert abs(c["arable_frac"]["actual"] - c["arable_frac"]["target"]) < 0.005      # IS-arable
        assert c["has_river"]["actual"] == c["has_river"]["target"]                       # IS-river
        assert all(not i["has_perennial_river"] for i in J1["islands"][1:])               # 小岛只有溪涧
        z = np.load(out / "terrain.npz")
        land = z["island_id"] >= 0
        assert np.isnan(z["height"][~land]).all() and not np.isnan(z["height"][land]).any()
        assert (z["landcover"][land] > 0).all() and (z["landcover"][~land] == 0).all()
    # 岛数与大小：主岛最大，最小岛 ≥ 0.3 km²（离散化允许一格误差），总和 = area
    areas = [i["area_target_km2"] for i in J1["islands"]]
    assert areas[0] == max(areas)
    assert min(areas) >= 0.3 - 1e-6
    assert abs(sum(areas) - c["area_km2"]["target"]) < 1e-3
    # 连通：索桥 + 短渡把所有岛连成一片（IS-link）
    from zhouzhu_gen.graph import weak_components
    n = len(J1["islands"])
    src = np.array([e["a"] for e in J1["links"]], dtype=np.int64)
    dst = np.array([e["b"] for e in J1["links"]], dtype=np.int64)
    assert weak_components(n, src, dst).max() == 0


def test_shape_area_and_single_component():
    from zhouzhu_gen.island.terrain import island_shape
    from zhouzhu_gen.island.grid import label_components
    cfg = load_config()["island"]["terrain"]
    rng = np.random.default_rng(3)
    for area in (0.5, 12.0, 400.0):
        mask, inside, X, Y = island_shape(rng, area, 0.1, 1.6, 0.4, cfg)
        cells = mask.sum() * 0.01
        assert abs(cells - area) <= max(0.01, 0.01 * area) + 0.01
        _, n = label_components(mask)
        assert n == 1


def test_label_components_runs():
    from zhouzhu_gen.island.grid import label_components
    m = np.zeros((6, 6), dtype=bool)
    m[0, 0:3] = True
    m[1, 2] = True
    m[3, 4] = True
    m[4, 5] = True
    lab4, n4 = label_components(m, 4)
    lab8, n8 = label_components(m, 8)
    assert n4 == 3 and n8 == 2
    assert lab4[0, 0] == lab4[1, 2]
    assert lab8[3, 4] == lab8[4, 5] and lab4[3, 4] != lab4[4, 5]
