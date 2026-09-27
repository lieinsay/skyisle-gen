"""浮高（DESIGN-NOTES 四点二十八，Zhouzhu 浮高计划 G 期）：三个 seed 的小世界里两个后端逐字节对照、平移的语义、关掉时不浮。

扩展没编（python core/build.py）时整个文件跳过。小世界（1600 群）三个 seed 各跑到 ④，每个 seed 挑两三群：
  两个后端整群 generate 的整套产物逐字节相同（island.json 只差 meta.seconds / engine），且真的有岛浮了；
  浮高 = 整座平移：地形那一步开关浮高，岛号栅格不变、每座非主岛的高程整体差 δ、主岛不动，岸缘 / 峰 / 台面 / 岛底跟着平移、崖高不变；
  岸缘 + δ ≥ rim_floor_m、δ ∈ [−down_max, +up_max]、IS-float 过；村与资源点的海拔读平移后的高程；关掉（float.enabled = false）时 float_m 全 0。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")

SMALL = ["s03.islands.n_islands=1600"]
SEEDS = (42, 7, 2026)
QUIET = lambda *a, **k: None   # noqa: E731


@pytest.fixture(scope="module", params=SEEDS)
def world(request, tmp_path_factory):
    from skyisle_gen.config import load_config
    from skyisle_gen.pipeline import Context, run
    seed = request.param
    root = tmp_path_factory.mktemp(f"float{seed}")
    cfg = load_config(sets=SMALL + [f"run.id=float{seed}"])
    out = run(cfg, seed, root, upto=4)
    return Context(cfg, seed, out)


def _nodes(ctx, k=2):
    """有河、主岛不太大的几群 + 一个无河的（测试快）。"""
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    ma = isl["main_area_km2"]
    ok = np.where(cli["has_river"] & (ma < 2500) & (ma > 200))[0]
    dry = np.where(~cli["has_river"] & (ma < 2500) & (ma > 200))[0]
    return [int(x) for x in ok[:k]] + [int(x) for x in dry[:1]]


def _products(out: Path) -> dict:
    res = {}
    for p in sorted(out.iterdir()):
        if p.name == "island.json":
            J = json.loads(p.read_text(encoding="utf-8"))
            J["meta"].pop("seconds", None)
            J["meta"].pop("engine", None)
            res[p.name] = json.dumps(J, sort_keys=True, ensure_ascii=False)
        elif p.suffix in (".npz", ".png", ".csv", ".json"):
            res[p.name] = p.read_bytes()
    return res


def _gen(ctx, node, backend, root, sets=()):
    from skyisle_gen import island as isl
    out, g = isl.generate(ctx, node, res_m=300.0, sets=[f"engine.backend={backend}"] + list(sets), log=QUIET, return_state=True, out_root=root)
    ctx.cfg["engine"]["backend"] = "python"
    return out, g


def _terrain(ctx, node, backend, sets=()):
    from skyisle_gen import island as isl
    c = isl.island_config(ctx, [f"engine.backend={backend}"] + list(sets))
    g = isl.build_terrain(ctx, node, c, isl._node_inputs(ctx, node), res_m=300.0, log=QUIET)
    ctx.cfg["engine"]["backend"] = "python"
    return c, g


def test_float_products_identical_two_backends(world, tmp_path):
    """整群 generate（地形 → 资源 → 四季 → 天气 → 聚落）：两个后端的整套产物逐字节相同，且群里真的有岛往上、往下浮了。"""
    moved = 0
    for node in _nodes(world):
        a, ga = _gen(world, node, "python", tmp_path / "py")
        b, gb = _gen(world, node, "cpp", tmp_path / "cpp")
        pa, pb = _products(a), _products(b)
        assert sorted(pa) == sorted(pb), node
        bad = [k for k in pa if pa[k] != pb[k]]
        assert not bad, (world.seed, node, bad)
        assert ga["settle"] == gb["settle"] and ga["resources"] == gb["resources"]
        fl = [i["float_m"] for i in gb["json"]["islands"]]
        assert fl[0] == 0.0
        moved += sum(1 for x in fl[1:] if x != 0.0)
    assert moved > 0


def test_float_is_a_translation(world):
    """地形那一步开关浮高：岛号栅格不变；每座非主岛整体平移 δ（高程、岸缘、峰、台面、岛底），崖高与起伏不变；主岛一格不动。"""
    node = _nodes(world, 1)[0]
    for backend in ("python", "cpp"):
        _, g_on = _terrain(world, node, backend)
        _, g_off = _terrain(world, node, backend, ["island.float.enabled=false"])
        assert np.array_equal(g_on["island_id"], g_off["island_id"])
        iid = g_on["island_id"]
        I_on, I_off = g_on["json"]["islands"], g_off["json"]["islands"]
        assert all(i["float_m"] == 0.0 for i in I_off)
        m0 = iid == 0
        assert np.array_equal(g_on["height"][m0], g_off["height"][m0])
        for a, b in zip(I_on, I_off):
            k = a["id"]
            d = g_on["height"][iid == k] - g_off["height"][iid == k]
            if not d.size:
                continue
            assert np.allclose(d, d[0], atol=1e-6), (backend, k)
            assert abs(float(d[0]) - a["float_m"]) <= 0.051, (backend, k, float(d[0]), a["float_m"])
            for key in ("rim_m", "peak_m", "surface_m", "keel_m"):
                assert abs(a[key] - b[key] - a["float_m"]) <= 0.15, (backend, k, key)
            assert abs(a["cliff_m"] - b["cliff_m"]) <= 0.1 and abs(a["relief_m"] - b["relief_m"]) <= 0.1
        # 索桥按平移后的岸缘高差判
        dh = {(e["a"], e["b"]): e["dh_m"] for e in g_on["json"]["links"]}
        rims = {i["id"]: i["rim_m"] for i in I_on}
        for (x, y), v in dh.items():
            assert abs(v - abs(rims[x] - rims[y])) <= 0.15


def test_float_bounds_and_is_float(world, tmp_path):
    """δ ∈ [−down_max, +up_max]、岸缘 ≥ rim_floor_m；IS-float 过，改坏了会报；村与资源点的海拔是平移后的高程。"""
    from skyisle_gen.island.check import evaluate
    node = _nodes(world, 1)[0]
    out, g = _gen(world, node, "cpp", tmp_path / "c")
    c = g["json"]
    from skyisle_gen import island as isl
    fc = isl.island_config(world)["float"]
    for i in c["islands"][1:]:
        assert -fc["down_max_m"] <= i["float_m"] <= fc["up_max_m"]
        assert i["rim_m"] >= fc["rim_floor_m"]
    items = {x["id"]: x for x in evaluate(g, out, c=isl.island_config(world))}
    assert items["IS-float"]["pass"], items["IS-float"]
    g2 = {**g, "json": json.loads(json.dumps(g["json"]))}
    if len(g2["json"]["islands"]) > 1:
        g2["json"]["islands"][1]["rim_m"] = fc["rim_floor_m"] - 1.0
        items2 = {x["id"]: x for x in evaluate(g2, out, c=isl.island_config(world))}
        assert not items2["IS-float"]["pass"]
    h = g["height"]
    for v in g["settle"]["villages"]:
        i, j = v["cell"]
        assert v["elev_m"] == round(float(h[i, j]), 0)
    for d in g["resources"]["deposits"]:
        if d.get("form") == "point":
            i, j = d["cell"]
            assert abs(d["elev_m"] - round(float(h[i, j]), 0)) <= 1.0, d


def test_float_off_means_no_float(world):
    """float.enabled = false：island.json 的 float_m 全 0，两个后端相同。"""
    node = _nodes(world, 1)[0]
    _, ga = _terrain(world, node, "python", ["island.float.enabled=false"])
    _, gb = _terrain(world, node, "cpp", ["island.float.enabled=false"])
    assert np.array_equal(ga["height"], gb["height"], equal_nan=True)
    assert [i["float_m"] for i in ga["json"]["islands"]] == [0.0] * len(ga["json"]["islands"])
    assert ga["json"]["islands"] == gb["json"]["islands"]


def test_float_stats_summary():
    """island floats 的摘要：秩相关、标定对照的口径。"""
    from skyisle_gen.island.floats import rank_avg, spearman, summarize
    assert np.array_equal(rank_avg(np.array([3.0, 1.0, 1.0, 2.0])), np.array([4.0, 1.5, 1.5, 3.0]))
    x = np.arange(10.0)
    assert spearman(x, -x ** 3) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    dage = rng.normal(0, 1, 5000)
    d = np.where(-dage + 0.5 + rng.normal(0, 1, 5000) > 0, 450.0, -250.0) * rng.uniform(0.3, 2.0, 5000)
    s = summarize(np.clip(d, -500, 1450), dage, np.full(5000, 100.0), {"up_max_m": 1500.0, "rim_floor_m": 20.0})
    assert s["spearman_age"] < -0.4 and 0.6 < s["up_share"] < 0.8 and s["calib"]["rim_floor"]
