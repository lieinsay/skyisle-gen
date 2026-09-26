"""C++ 核心（core/，skyisle_gen._core）与 numpy 版的同输入对照（docs/PLAN-CORE.md 第三、八节）。

扩展没编（python core/build.py）时整个文件跳过。能逐位的逐位比：随机流、噪声、连通分量、形态学、填洼、D8、汇流；
小世界上的 build_terrain / build_hydro 两个后端的对照、cpp 后端的确定性与线程数无关在文件后半。
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")

from skyisle_gen.island import grid as G          # noqa: E402
from skyisle_gen.island import terrain as T        # noqa: E402
from skyisle_gen.rng import entity_rng             # noqa: E402

KEYS = ["island:2051:layout", "island:1165:place", "island:7:shape:3", "island:0:terrain:12", "x"]


def _rand_mask(rng, H, W, p=0.55, blobs=True):
    m = rng.random((H, W)) < p
    if blobs:
        m = G.binary_dilate(G.binary_erode(m, 1, 4), 1, 4)
    return m


# ---------------------------------------------------------------- 随机数：与 numpy 逐位一致
@pytest.mark.parametrize("key", KEYS)
def test_rng_raw_matches_numpy(key):
    for seed in (42, 7, 2026, 0):
        ref = np.random.PCG64(np.random.SeedSequence([seed, 21, zlib.crc32(key.encode())])).random_raw(64)
        assert np.array_equal(core.rng_raw(seed, 21, key, 64), ref)
    assert core.crc32(key) == zlib.crc32(key.encode())


@pytest.mark.parametrize("key", KEYS[:3])
def test_rng_distributions_match_numpy(key):
    r = entity_rng(42, 21, key)
    assert np.array_equal(core.rng_draw(42, 21, key, "uniform", 5000, -1.0, 1.0), r.uniform(-1.0, 1.0, 5000))
    r = entity_rng(42, 21, key)
    assert np.array_equal(core.rng_draw(42, 21, key, "normal", 200000, 0.0, 0.25), r.normal(0.0, 0.25, 200000))
    r = entity_rng(42, 21, key)
    assert np.array_equal(core.rng_draw(42, 21, key, "beta", 20000, 1.3, 2.2), r.beta(1.3, 2.2, 20000))
    for lo, hi in ((0, 2), (9, 16), (0, 7), (0, 100003), (1, 4)):
        r = entity_rng(42, 21, key)
        ref = np.array([r.integers(lo, hi) for _ in range(3000)], dtype=np.float64)
        assert np.array_equal(core.rng_draw(42, 21, key, "integers", 3000, lo, hi), ref), (lo, hi)
    r = entity_rng(42, 21, key)
    ref = r.choice([-1.0, 1.0], 777)
    got = core.rng_draw(42, 21, key, "integers", 777, 0, 2)
    assert np.array_equal(np.array([-1.0, 1.0])[got.astype(int)], ref)
    w = np.sqrt(np.array([500.0, 30.0, 12.0, 3.0, 1.0]))
    w[0] *= 3.0
    p = w / w.sum()
    r = entity_rng(42, 21, key)
    ref = [int(r.choice(5, p=p)) for _ in range(4000)]
    assert core.rng_choice_p(42, 21, key, p.tolist(), 4000).tolist() == ref


def test_math_matches_numpy_and_python():
    import math
    rng = np.random.default_rng(9)
    x = rng.random(100000) * 10 ** rng.uniform(-4, 4, 100000)
    y = (rng.random(100000) - 0.5) * 10 ** rng.uniform(-4, 4, 100000)
    for e in (2.0, 0.5, 1.5, 0.35, -1.0, 1.0):
        a, b, c, d = core.math_fns(x, np.full(100000, e))
        assert np.array_equal(a, x ** e), e                      # numpy 数组的幂（含快路径）
        assert b.tolist() == [v ** e for v in x.tolist()], e     # Python 浮点的幂 = C pow
    a, b, c, d = core.math_fns(x, y)
    assert np.array_equal(c, np.hypot(x, y))
    assert d.tolist() == [math.hypot(p, q) for p, q in zip(x.tolist(), y.tolist())]


def test_np_sum_and_pyround():
    rng = np.random.default_rng(5)
    for n in (1, 7, 8, 9, 100, 128, 129, 1000, 4097):
        a = rng.random(n) * 10 ** rng.uniform(-3, 3, n)
        assert core.np_sum(a) == float(a.sum())
    for x in (2.675, 0.125, 0.375, 1.0005, -3.14159, 123456.78949, 1e-7):
        for nd in (0, 1, 2, 3):
            assert core.pyround(x, nd) == round(x, nd)


# ---------------------------------------------------------------- 噪声与栅格
def test_fractal_noise_matches():
    xs = np.linspace(-37.0, 41.0, 90)
    X, Y = np.meshgrid(xs, -xs)
    fn = G.FractalNoise(entity_rng(42, 21, "island:5:shape:0"), -40, -40, 42, 42, feature_km=9.3, octaves=5, persistence=0.55)
    got = core.fractal_noise(42, 21, "island:5:shape:0", -40, -40, 42, 42, 9.3, 5, 0.55, X, Y)
    assert np.array_equal(got, fn.sample(X, Y))


def test_label_components_and_morphology_match():
    rng = np.random.default_rng(1)
    for t in range(6):
        H, W = int(rng.integers(5, 90)), int(rng.integers(5, 90))
        m = _rand_mask(rng, H, W, p=float(rng.uniform(0.3, 0.8)), blobs=bool(t % 2))
        for conn in (4, 8):
            lab, n = G.label_components(m, conn)
            lab2, n2 = core.label_components(m, conn)
            assert n == n2 and np.array_equal(lab, lab2)
        assert np.array_equal(core.largest_component(m), G.largest_component(m))
        for conn in (4, 8):
            assert np.array_equal(core.binary_erode(m, 2, conn), G.binary_erode(m, 2, conn))
            assert np.array_equal(core.binary_dilate(m, 2, conn), G.binary_dilate(m, 2, conn))
        assert np.array_equal(core.distance_bands(m, 7), G.distance_bands(m, 7))
        seed = rng.random((H, W)) < 0.03
        within = _rand_mask(rng, H, W, 0.7)
        for w in (None, within):
            d, s = G.nearest_propagate(seed, 9, 100.0, within=w)
            d2, s2 = core.nearest_propagate(seed, 9, 100.0, w)
            assert np.array_equal(d, d2) and np.array_equal(s, s2)


def test_resample_and_smooth_match():
    rng = np.random.default_rng(2)
    a = rng.random((37, 23)) * 900.0
    m = _rand_mask(rng, 37, 23, 0.7)
    for f in (2, 3, 5, 8):
        assert np.array_equal(core.block_mean(a, f), G.block_mean(a, f))
        assert np.array_equal(core.block_any(m, f), G.block_any(m, f))
        c = G.block_mean(a, f)
        assert np.array_equal(core.upsample_bilinear(c, f, 37, 23), G.upsample_bilinear(c, f, 37, 23))
    assert np.array_equal(core.smooth121(a, m, 2), G.smooth121(a, m, 2))
    assert np.array_equal(core.laplacian(a, m), G.laplacian(a, m))
    hh = np.where(m, a, np.nan)
    assert np.allclose(core.slope_deg(hh, m, 100.0), G.slope_deg(hh, m, 100.0), rtol=0, atol=1e-12)


# ---------------------------------------------------------------- 水文核心
def _terrain(rng, H, W):
    m = G.largest_component(_rand_mask(rng, H, W, 0.8))
    xs = np.linspace(-1, 1, W)
    ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(xs, ys)
    h = 800.0 * (1 - X ** 2 - Y ** 2) + 60.0 * rng.random((H, W))
    return np.where(m, h, 0.0), m


def test_fill_d8_accumulate_match():
    rng = np.random.default_rng(3)
    for _ in range(4):
        H, W = int(rng.integers(20, 80)), int(rng.integers(20, 80))
        h, m = _terrain(rng, H, W)
        pf = T.priority_fill(h, m, eps=1e-3)
        assert np.array_equal(core.priority_fill(h, m, 1e-3), pf, equal_nan=True)
        fi = T.fill_iter(h, m, 12)
        assert np.array_equal(core.fill_iter(h, m, 12, 0.01), fi, equal_nan=True)
        ri, rj, _s, tv = T.d8(pf, m, 100.0)
        ri2, rj2, tv2 = core.d8(np.nan_to_num(pf), m, 100.0)
        assert np.array_equal(ri, ri2) and np.array_equal(rj, rj2) and np.array_equal(tv, tv2)
        A = T.accumulate(pf, m, ri, rj)
        assert np.array_equal(core.accumulate(m, ri, rj), A)
        ri, rj, tv = T.d8_random(pf, m, 100.0, entity_rng(42, 21, "island:1:terrain:0"), 1.5)
        ri2, rj2, tv2 = core.d8_random(np.nan_to_num(pf), m, 100.0, 42, 21, "island:1:terrain:0", 1.5)
        assert np.array_equal(ri, ri2) and np.array_equal(rj, rj2) and np.array_equal(tv, tv2)


# ---------------------------------------------------------------- 岛形、地形、势力范围：同输入逐位对照
def _island_cfg(**over):
    import tomllib
    with open(Path(__file__).resolve().parent.parent / "config" / "default.toml", "rb") as fh:
        c = tomllib.load(fh)["island"]
    for k, v in over.items():
        sec, key = k.split("__")
        c[sec] = dict(c[sec], **{key: v})
    return c


def test_island_shape_and_profile_match():
    from skyisle_gen.island.engine import flat_config
    from skyisle_gen.island.layout import radial_profile
    c = _island_cfg()
    for key, area, res, el, th in (("island:5:shape:0", 60.0, 0.2, 1.7, 0.4), ("island:9:shape:3", 3.1, 0.1, 1.0, -2.0),
                                   ("island:1:shape:7", 0.3, 0.4, 2.1, 1.0)):      # 最后一个比半格还小：留一格
        mask, inside, X, Y = T.island_shape(entity_rng(42, 21, key), area, res, el, th, c["terrain"])
        m2, in2, xs = core.island_shape(42, key, area, res, el, th, flat_config(c))
        assert np.array_equal(m2, mask) and np.array_equal(in2, inside) and np.array_equal(xs, X[0])
        ctr, prof = radial_profile(mask, res)
        ctr2, prof2 = core.radial_profile(mask, res)
        assert np.array_equal(prof2, prof) and tuple(ctr2) == tuple(ctr)


@pytest.mark.parametrize("age", [0.1, 0.5, 0.9])
def test_sculpt_island_matches(age):
    """三种岛龄基形 + 侵蚀（粗网格 f > 1 与 f = 1 两条路）+ 仿射拟合，与 numpy 版逐位相同。"""
    from skyisle_gen.island.engine import flat_config
    for emc in (40, 320):
        c = _island_cfg(terrain__erosion_max_cells=emc)
        area, res = 40.0, 0.2
        mask, inside, X, Y = T.island_shape(entity_rng(7, 21, "island:3:shape:1"), area, res, 1.4, 0.3, c["terrain"])
        h, kind, rim, peak = T.sculpt_island(entity_rng(7, 21, "island:3:terrain:1"), mask, inside, X, Y, age, area, res,
                                             600.0, 900.0, 180.0, False, c["terrain"])
        h2, kind2, rim2, peak2 = core.sculpt_island(7, "island:3:shape:1", "island:3:terrain:1", area, res, 1.4, 0.3, age,
                                                    600.0, 900.0, 180.0, False, flat_config(c))
        assert kind2 == kind and rim2 == rim and peak2 == peak
        assert np.array_equal(h2, h, equal_nan=True), (age, emc, np.nanmax(np.abs(h2 - h)))


def test_territory_limits_match():
    from skyisle_gen.island.territory import limits
    rng = np.random.default_rng(4)
    n = 400
    lat, lon = rng.uniform(-60, 60, n), rng.uniform(-180, 180, n)
    area = rng.uniform(50, 5000, n).astype(np.float32)

    class Ctx:
        def load_npz(self, k, name):
            return {"lat": lat, "lon": lon, "area_km2": area}
    planet = {"radius_km": 6371.0, "year_s": 1.0, "islands": {"lat": lat, "lon": lon, "area": area.astype(np.float64)}}
    for node in (0, 17, 399):
        ref = limits(Ctx(), node, {"planet": {"radius_km": 6371.0}}, {"gap_km": 3.0, "reach": 30.0, "reach_km": 2000.0})
        got = core.territory_limits(planet, node, 3.0, 30.0, 2000.0)
        assert [L["node"] for L in ref] == [L["node"] for L in got]
        for a, b in zip(ref, got):
            assert a["dist_km"] == round(b["dist_km"], 3) and tuple(a["u"]) == tuple(b["u"]) and a["limit_km"] == b["limit_km"]


# ---------------------------------------------------------------- 小世界：两个后端的 build_terrain / build_hydro
SMALL = ["s03.islands.n_islands=1600"]


@pytest.fixture(scope="module")
def small_ctx(tmp_path_factory):
    from skyisle_gen.config import load_config
    from skyisle_gen.pipeline import Context, run
    root = tmp_path_factory.mktemp("core")
    cfg = load_config(sets=SMALL + ["run.id=core"])
    out = run(cfg, 7, root, upto=4)
    return Context(cfg, 7, out)


def _nodes(ctx, k=3):
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    ok = np.where(cli["has_river"] & (isl["main_area_km2"] < 2500) & (isl["main_area_km2"] > 200))[0]
    return [int(x) for x in ok[:k]]


def _build(ctx, node, backend, res_m=300.0, threads=4):
    from skyisle_gen import island as isl
    from skyisle_gen.island.hydro import build_hydro
    c = isl.island_config(ctx, [f"engine.backend={backend}", f"engine.threads={threads}"])
    inp = isl._node_inputs(ctx, node)
    g = isl.build_terrain(ctx, node, c, inp, res_m=res_m, log=lambda *a: None)
    build_hydro(ctx, node, c, g, log=lambda *a: None)
    return g


GRIDS = ("island_id", "cliff", "height", "river", "stream", "lake", "landcover", "arable", "floodplain", "flowacc_km2",
         "river_width_m", "river_depth_m", "cut_m", "slope_deg", "filled", "recv_i", "recv_j", "route_h")


def test_small_world_backends_agree(small_ctx):
    """同一个群两个后端：栅格同形同 dtype；陆地 / 主岛按目标、岛数、可耕率一致（这几个群实测逐位相同）。"""
    for node in _nodes(small_ctx):
        a, b = _build(small_ctx, node, "python"), _build(small_ctx, node, "cpp")
        Ja, Jb = a["json"], b["json"]
        assert Jb["meta"]["engine"] == "cpp" and "engine" not in Ja["meta"]
        for k in GRIDS:
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        ca, cb = Ja["constraints"], Jb["constraints"]
        for k in ("area_km2", "main_area_km2"):
            assert abs(cb[k]["actual"] - cb[k]["target"]) <= 0.02 * cb[k]["target"]
        assert len(Ja["islands"]) == len(Jb["islands"])
        assert abs(cb["arable_frac"]["actual"] - cb["arable_frac"]["target"]) < 0.005
        assert cb["has_river"]["actual"] == cb["has_river"]["target"]
        assert abs(cb["peak_m"]["actual"] - ca["peak_m"]["actual"]) <= 0.1 * ca["peak_m"]["actual"]
        same = all(np.array_equal(a[k], b[k], equal_nan=True) for k in GRIDS)
        Ja["meta"].pop("engine", None), Jb["meta"].pop("engine", None)
        assert same == (Ja == Jb), node                       # 栅格全同则 island.json 也全同（拼装与 Python 版同式）


def test_cpp_deterministic_and_thread_independent(small_ctx):
    node = _nodes(small_ctx, 1)[0]
    g1 = _build(small_ctx, node, "cpp", threads=1)
    g4 = _build(small_ctx, node, "cpp", threads=4)
    g4b = _build(small_ctx, node, "cpp", threads=4)
    for k in GRIDS:
        assert np.array_equal(g1[k], g4[k], equal_nan=True) and np.array_equal(g4[k], g4b[k], equal_nan=True), k
    assert g1["json"] == g4["json"] and g1["river_lines"] == g4["river_lines"]


def test_cpp_generate_passes_checks(small_ctx):
    """cpp 后端下整条 generate（资源、气候、天气、聚落照旧 Python）的产物照常写出、硬项全过、重跑哈希一致。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.check import evaluate, hash_products
    node = _nodes(small_ctx, 1)[0]
    s = ["engine.backend=cpp"]
    out, g = isl.generate(small_ctx, node, res_m=300.0, sets=s, log=lambda *a: None, return_state=True)
    h1 = hash_products(out)
    isl.generate(small_ctx, node, res_m=300.0, sets=s, log=lambda *a: None)
    items = evaluate(g, out, ctx=small_ctx, node=node, c=small_ctx.cfg["island"], det_hashes=(h1, hash_products(out)), daily_years=10)
    bad = [i["id"] for i in items if not i["pass"] and i["hard"]]
    assert not bad, bad
    small_ctx.cfg["engine"]["backend"] = "python"


def test_nearest_fit_matches_numpy():
    """Dykstra 的越界量在贴线时是 ±1e−15 量级：乘加次序要与 numpy（BLAS 的 ddot / dgemv）一样，否则 v ≤ 0 的判断会翻。"""
    from skyisle_gen.island.territory import nearest_fit
    rng = np.random.default_rng(6)
    for t in range(300):
        m = int(rng.integers(1, 12))
        az = rng.uniform(-np.pi, np.pi, m)
        lim = [{"u": (float(np.sin(a)), float(np.cos(a))), "limit_km": float(rng.uniform(5, 60))} for a in az]
        sup = rng.uniform(0, 70, m)
        o, v = nearest_fit(sup, lim)
        ox, oy, v2 = core.nearest_fit(sup.tolist(), [[L["u"][0], L["u"][1], L["limit_km"]] for L in lim])
        assert (ox, oy, v2) == (float(o[0]), float(o[1]), float(v)), t
