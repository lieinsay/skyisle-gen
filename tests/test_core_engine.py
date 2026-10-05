"""C++ 核心（core/，skyisle_gen._core）的公共件与第三层的地形、水系（docs/PLAN-CORE.md 第三、八节）。

扩展没编（python core/build.py）时整个文件跳过。随机流、幂 / hypot、求和与 round 仍与 numpy / Python 逐位比（那是 numpy 自己，不是删掉的参考后端）；
噪声、连通分量、形态学、重采样、填洼、D8、汇流、岛形、造形、势力范围按性质验（Python 参考后端 2026-09-30 删了，git tag python-reference-final）；
小世界上的 build_terrain / build_hydro 验约束、确定性与线程数无关，整群 generate 验 check 的硬项。
"""
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")

from skyisle_gen.rng import entity_rng             # noqa: E402

KEYS = ["island:2051:layout", "island:1165:place", "island:7:shape:3", "island:0:terrain:12", "x"]


def _rand_mask(rng, H, W, p=0.55, blobs=True):
    m = rng.random((H, W)) < p
    if blobs:
        m = core.binary_dilate(core.binary_erode(m, 1, 4), 1, 4)
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


# ---------------------------------------------------------------- 噪声与栅格：按性质验
def test_fractal_noise_deterministic_and_keyed():
    xs = np.linspace(-37.0, 41.0, 90)
    X, Y = np.meshgrid(xs, -xs)
    a = core.fractal_noise(42, 21, "island:5:shape:0", -40, -40, 42, 42, 9.3, 5, 0.55, X, Y)
    b = core.fractal_noise(42, 21, "island:5:shape:0", -40, -40, 42, 42, 9.3, 5, 0.55, X, Y)
    c = core.fractal_noise(42, 21, "island:5:shape:1", -40, -40, 42, 42, 9.3, 5, 0.55, X, Y)
    assert np.array_equal(a, b) and np.isfinite(a).all()
    assert np.abs(a).max() <= 1.0 and a.std() > 0.05
    assert np.abs(a - c).max() > 0.1                             # 随机流按 key 分开
    assert np.abs(np.diff(a, axis=1)).mean() < a.std()           # 相邻格相关（特征尺度 9.3 km ≫ 格距 0.9 km）


def test_label_components_and_morphology():
    rng = np.random.default_rng(1)
    for t in range(6):
        H, W = int(rng.integers(5, 90)), int(rng.integers(5, 90))
        m = _rand_mask(rng, H, W, p=float(rng.uniform(0.3, 0.8)), blobs=bool(t % 2))
        n_by = {}
        for conn in (4, 8):
            lab, n = core.label_components(m, conn)
            n_by[conn] = n
            assert (lab[~m] == 0).all() and set(np.unique(lab[m]).tolist()) == set(range(1, n + 1))
            if n:                                                  # 每个标号自己是一个连通块
                one = lab == 1
                assert core.label_components(one, conn)[1] == 1
        assert n_by[8] <= n_by[4]
        big = core.largest_component(m)
        if m.any():
            lab, n = core.label_components(m, 4)
            cnt = np.bincount(lab[m])
            assert big.sum() == cnt.max() and not (big & ~m).any() and core.label_components(big, 4)[1] == 1
        for conn in (4, 8):
            er, di = core.binary_erode(m, 2, conn), core.binary_dilate(m, 2, conn)
            assert not (er & ~m).any() and not (m & ~di).any()
            assert not (core.binary_dilate(er, 2, conn) & ~m).any()   # 开运算 ⊆ 原集
        db = core.distance_bands(m, 7)
        assert (db[m] == 0).all() and (db[~m] >= 1).all() and db.max() <= 8
        seed = rng.random((H, W)) < 0.03
        if seed.any():
            d, src = core.nearest_propagate(seed, 9, 100.0)
            assert (d[seed] == 0).all() and (src[seed] == np.flatnonzero(seed.ravel())).all()
            got = src >= 0
            assert seed.ravel()[src[got]].all() and (d[got & ~seed] > 0).all()   # 每格的来源是个种子


def test_resample_and_smooth():
    rng = np.random.default_rng(2)
    H, W = 37, 23
    const = np.full((H, W), 7.5)
    m = _rand_mask(rng, H, W, 0.7)
    for f in (2, 3, 5, 8):
        assert np.array_equal(core.block_mean(const, f), np.full((-(-H // f), -(-W // f)), 7.5))
        pt = np.zeros((H, W), bool)
        pt[H - 1, W - 1] = True
        bb = core.block_any(pt, f)
        assert bb.sum() == 1 and bb[-1, -1]
        up = core.upsample_bilinear(np.full((-(-H // f), -(-W // f)), 3.0), f, H, W)
        assert up.shape == (H, W) and np.allclose(up, 3.0)
    assert np.allclose(core.smooth121(const, m, 2)[m], 7.5)
    J, I = np.meshgrid(np.arange(W), np.arange(H))
    full = np.ones((H, W), bool)
    lap = core.laplacian(2.0 * J + 3.0 * I, full)
    assert np.abs(lap[1:-1, 1:-1]).max() < 1e-9                  # 平面的拉普拉斯为 0
    sl = core.slope_deg(10.0 * J.astype(float), full, 100.0)      # 每格高 10 m、格距 100 m
    assert np.allclose(sl[1:-1, 1:-1], math.degrees(math.atan(0.1)))


# ---------------------------------------------------------------- 水文核心：按性质验
def _terrain(rng, H, W):
    m = core.largest_component(_rand_mask(rng, H, W, 0.8))
    xs = np.linspace(-1, 1, W)
    ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(xs, ys)
    h = 800.0 * (1 - X ** 2 - Y ** 2) + 60.0 * rng.random((H, W))
    return np.where(m, h, 0.0), m


def test_fill_d8_accumulate():
    """填洼只抬不降；D8 的下游是八邻域里填平面更低的格（没有下游的是出口）；汇流守恒：出口的汇流之和 = 陆地格数。"""
    rng = np.random.default_rng(3)
    for _ in range(4):
        H, W = int(rng.integers(20, 80)), int(rng.integers(20, 80))
        h, m = _terrain(rng, H, W)
        pf = core.priority_fill(h, m, 1e-3)
        assert np.isnan(pf[~m]).all() and (pf[m] >= h[m]).all()
        fi = core.fill_iter(h, m, 12, 0.01)
        assert (fi[m] >= h[m] - 1e-9).all()
        for ri, rj, _tv in (core.d8(np.nan_to_num(pf), m, 100.0),
                            core.d8_random(np.nan_to_num(pf), m, 100.0, 42, 21, "island:1:terrain:0", 1.5)):
            ii, jj = np.nonzero(m & (ri >= 0))
            r2, c2 = ri[ii, jj], rj[ii, jj]
            assert (np.maximum(np.abs(r2 - ii), np.abs(c2 - jj)) == 1).all()
            assert m[r2, c2].all() and (pf[r2, c2] < pf[ii, jj]).all()
            A = core.accumulate(m, ri, rj)
            out = m & (ri < 0)
            assert out.any() and (A[m] >= 1.0).all() and A[out].sum() == pytest.approx(float(m.sum()))
        r1 = core.d8_random(np.nan_to_num(pf), m, 100.0, 42, 21, "island:1:terrain:0", 1.5)
        r2 = core.d8_random(np.nan_to_num(pf), m, 100.0, 42, 21, "island:1:terrain:0", 1.5)
        assert all(np.array_equal(a, b) for a, b in zip(r1, r2))


# ---------------------------------------------------------------- 岛形、造形、势力范围：按性质验
def _island_cfg(**over):
    import tomllib
    with open(Path(__file__).resolve().parent.parent / "config" / "default.toml", "rb") as fh:
        c = tomllib.load(fh)["island"]
    for k, v in over.items():
        sec, key = k.split("__")
        c[sec] = dict(c[sec], **{key: v})
    return c


def test_island_shape_and_profile():
    """面积二分反解到目标（比半格还小的留一格）、一个连通块；角向半径剖面 72 个方向、都在岛内。"""
    from skyisle_gen.island.engine import flat_config
    c = flat_config(_island_cfg())
    for key, area, res, el, th in (("island:5:shape:0", 60.0, 0.2, 1.7, 0.4), ("island:9:shape:3", 3.1, 0.1, 1.0, -2.0),
                                   ("island:1:shape:7", 0.3, 0.4, 2.1, 1.0)):
        mask, inside, xs = core.island_shape(42, key, area, res, el, th, c)
        assert mask.sum() >= 1 and abs(mask.sum() * res * res - area) <= max(res * res, 0.01 * area)
        assert core.label_components(mask, 4)[1] == 1
        assert inside.shape == mask.shape and xs.shape == (mask.shape[1],)
        m2, in2, xs2 = core.island_shape(42, key, area, res, el, th, c)
        assert np.array_equal(m2, mask) and np.array_equal(in2, inside) and np.array_equal(xs2, xs)
        ctr, prof = core.radial_profile(mask, res)
        assert prof.shape == (72,) and (prof > 0).all() and prof.max() <= math.hypot(*mask.shape) * res


@pytest.mark.parametrize("age", [0.1, 0.5, 0.9])
def test_sculpt_island(age):
    """三种岛龄基形 + 侵蚀（粗网格 f > 1 与 f = 1 两条路）+ 仿射拟合：陆地中位 = 台面、峰 − 岸缘 = 目标起伏，岛外是 NaN，重跑相同。"""
    from skyisle_gen.island.engine import flat_config
    kinds = {0.1: "young", 0.5: "mid", 0.9: "old"}
    for emc in (40, 320):
        c = flat_config(_island_cfg(terrain__erosion_max_cells=emc))
        mask, _, _ = core.island_shape(7, "island:3:shape:1", 40.0, 0.2, 1.4, 0.3, c)
        args = (7, "island:3:shape:1", "island:3:terrain:1", 40.0, 0.2, 1.4, 0.3, age, 600.0, 900.0, 180.0, False, c)
        h, kind, rim, peak = core.sculpt_island(*args)
        assert kind == kinds[age]
        assert np.isfinite(h[mask]).all() and np.isnan(h[~mask]).all()
        assert float(np.median(h[mask])) == pytest.approx(600.0, abs=1.0)
        assert peak - rim == pytest.approx(900.0, abs=1.0) and peak == pytest.approx(float(np.nanmax(h)), abs=1e-6)
        h2, *_ = core.sculpt_island(*args)
        assert np.array_equal(h2, h, equal_nan=True)


@pytest.mark.parametrize("n_seed", [0, 1, 2])
def test_sculpt_island_multicore(n_seed):
    """多核嵌合（P4）：两三个核、主核强度 1、各核的格数加起来 = 岛；不在汇聚带、新岛不是多核（粗网格 f > 1 与 f = 1 两条路）。"""
    from skyisle_gen.island.engine import flat_config
    for emc in (40, 320):
        c = flat_config(_island_cfg(terrain__erosion_max_cells=emc, terrain__multicore_frac=1.0, terrain__multicore_three_frac=0.5))
        key = f"island:{n_seed}:cores:0"
        mask, _, _ = core.island_shape(7, "island:3:shape:0", 700.0, 0.8, 1.6, 0.3, c)
        h, kind, rim, peak, cores = core.sculpt_island_cores(7, "island:3:shape:0", "island:3:terrain:0", key, 700.0, 0.8, 1.6, 0.3, 0.45,
                                                             900.0, 1500.0, 320.0, True, 0.9, 0, c)
        assert 2 <= len(cores) <= 3 and max(x[2] for x in cores) == 1.0
        assert sum(x[3] for x in cores) == int(mask.sum())
        assert peak - rim == pytest.approx(1500.0, abs=1.0) and np.isfinite(h[mask]).all()
    c = flat_config(_island_cfg(terrain__multicore_frac=1.0))
    base = (7, "island:3:shape:0", "island:3:terrain:0", "k", 700.0, 0.8, 1.6, 0.3)
    assert not core.sculpt_island_cores(*base, 0.45, 900.0, 1500.0, 320.0, True, 0.9, 1, c)[4]   # 不在汇聚带
    assert not core.sculpt_island_cores(*base, 0.1, 900.0, 1500.0, 320.0, True, 0.9, 0, c)[4]    # 新岛


def test_territory_limits_split():
    """势力范围：两群各自画的分界线是同一条（t_k + t_j + 缝 = 群心距）、法向相反。"""
    rng = np.random.default_rng(4)
    n = 400
    lat, lon = rng.uniform(-60, 60, n), rng.uniform(-180, 180, n)
    area = rng.uniform(50, 5000, n)
    planet = {"radius_km": 6371.0, "year_s": 1.0, "islands": {"lat": lat, "lon": lon, "area": area}}
    for node in (0, 17, 399):
        lk = core.territory_limits(planet, node, 3.0, 30.0, 2000.0)
        assert lk and all(L["dist_km"] > 0 and L["node"] != node for L in lk)
        L0 = lk[0]
        lj = {L["node"]: L for L in core.territory_limits(planet, L0["node"], 3.0, 30.0, 2000.0)}
        assert node in lj
        assert L0["limit_km"] + lj[node]["limit_km"] + 3.0 == pytest.approx(L0["dist_km"], abs=1e-6)
        u1, u2 = np.array(L0["u"]), np.array(lj[node]["u"])
        assert np.hypot(*u1) == pytest.approx(1.0) and u1 @ -u2 > 0.9


def test_nearest_fit():
    """Dykstra：放得下时越界量 ≤ 0 且偏移后每条线都守住；放不下时越界量 > 0。"""
    rng = np.random.default_rng(6)
    n_fit = 0
    for t in range(300):
        m = int(rng.integers(1, 12))
        az = rng.uniform(-np.pi, np.pi, m)
        lim = [[float(np.sin(a)), float(np.cos(a)), float(rng.uniform(5, 60))] for a in az]
        sup = rng.uniform(0, 70, m).tolist()
        ox, oy, v = core.nearest_fit(sup, lim)
        worst = max(s + ux * ox + uy * oy - L for s, (ux, uy, L) in zip(sup, lim))
        if v <= 1e-9:
            n_fit += 1
            assert worst <= 1e-6, t
        assert (ox, oy, v) == core.nearest_fit(sup, lim)
    assert n_fit > 0


# ---------------------------------------------------------------- 小世界：build_terrain / build_hydro
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


def _build(ctx, node, res_m=300.0, threads=4, sets=()):
    from skyisle_gen import island as isl
    from skyisle_gen.island.hydro import build_hydro
    c = isl.island_config(ctx, list(sets) + [f"engine.threads={threads}"])
    inp = isl._node_inputs(ctx, node)
    g = isl.build_terrain(ctx, node, c, inp, res_m=res_m, log=lambda *a: None)
    build_hydro(ctx, node, c, g, log=lambda *a: None)
    return g


GRIDS = ("island_id", "cliff", "height", "river", "stream", "lake", "landcover", "arable", "cultivable", "floodplain", "flowacc_km2",
         "w_ch_m", "d_ch_m", "cut_m", "slope_deg", "filled", "recv_i", "recv_j", "route_h", "lith", "coast_dist_m")


def test_small_world_constraints(small_ctx):
    """地形 + 水系：陆地 / 主岛按目标、可耕率、主岛有没有河与行星层对得上；栅格齐全同形。"""
    for node in _nodes(small_ctx):
        g = _build(small_ctx, node)
        J = g["json"]
        assert J["meta"]["engine"] == "cpp"
        shape = g["island_id"].shape
        for k in GRIDS:
            assert g[k].shape == shape, k
        cb = J["constraints"]
        for k in ("area_km2", "main_area_km2"):
            assert abs(cb[k]["actual"] - cb[k]["target"]) <= 0.02 * cb[k]["target"]
        assert abs(cb["arable_frac"]["actual"] - cb["arable_frac"]["target"]) < 0.005
        assert cb["has_river"]["actual"] == cb["has_river"]["target"]
        assert cb["peak_m"]["actual"] > cb["height_m"]["actual"]


def test_p4_terrain_hydro(small_ctx):
    """地貌 P4（新岛成拱、多核嵌合、谷收拢、局地雨、湿地）：强开多核之后汇聚带上有多核岛、有谷收拢；局地雨的全群均值 = 行星层的年降水。"""
    from skyisle_gen import island as isl
    keys = ("multicore_frac", "multicore_kernel_full", "multicore_min_km2")
    orig = {k: isl.island_config(small_ctx)["terrain"][k] for k in keys}
    sets = ["island.terrain.multicore_frac=1.0", "island.terrain.multicore_kernel_full=0.01", "island.terrain.multicore_min_km2=30.0"]
    isl3, plates = small_ctx.load_npz(3, "islands"), small_ctx.load_npz(3, "plates")

    def btype(n):                               # 板块边界类型（0 = 汇聚），取最近的板块网格格
        lat, lon = float(isl3["lat"][n]), float(isl3["lon"][n])
        ii = int(np.clip(np.searchsorted(plates["lats"], lat), 0, plates["lats"].size - 1))
        jj = int(np.clip(np.searchsorted(plates["lons"], ((lon + 180.0) % 360.0) - 180.0), 0, plates["lons"].size - 1))
        return int(plates["btype"][ii, jj])
    conv = [n for n in _nodes(small_ctx, 40) if btype(n) == 0]
    seen_cores = seen_cap = False
    for node in _nodes(small_ctx, 2) + conv[:2]:
        g = _build(small_ctx, node, sets=sets)
        J = g["json"]
        seen_cores |= any(i.get("cores") for i in J["islands"])
        seen_cap |= sum(i.get("captures", 0) for i in J["islands"]) > 0
        land = g["island_id"] >= 0
        assert abs(float(g["rain_mm"][land].astype(np.float64).mean()) - J["hydro"]["precip_mm"]) <= 1.0
    isl.island_config(small_ctx, [f"island.terrain.{k}={v}" for k, v in orig.items()])   # --set 会留在 ctx 上：改回默认
    assert seen_cores and seen_cap


def test_cpp_deterministic_and_thread_independent(small_ctx):
    node = _nodes(small_ctx, 1)[0]
    g1 = _build(small_ctx, node, threads=1)
    g4 = _build(small_ctx, node, threads=4)
    g4b = _build(small_ctx, node, threads=4)
    for k in GRIDS:
        assert np.array_equal(g1[k], g4[k], equal_nan=True) and np.array_equal(g4[k], g4b[k], equal_nan=True), k
    assert g1["json"] == g4["json"] and g1["river_lines"] == g4["river_lines"]


def test_two_calls_match_generate(small_ctx):
    """地形与水系分两次调（粗版 lod、浮高统计走这条）与整群 generate 一次调：地形、水系、岩性逐位相同、主岛岸缘相同
    （B2 的层面与每岛的层序要原样交回 build_hydro——漏了，谷坡角退回常数，下切、dz、粗版天气的岸缘都差一点）。"""
    from skyisle_gen import island as isl
    node = _nodes(small_ctx, 1)[0]
    g2 = _build(small_ctx, node)
    assert "strat_top" in g2
    g1 = isl.generate(small_ctx, node, write=False, res_m=300.0, steps=2, log=lambda *a: None)
    for k in ("height", "lith", "river", "stream", "flowacc_km2", "slope_deg", "w_ch_m", "coast_dist_m"):
        assert np.array_equal(g1[k], g2[k], equal_nan=True), k
    assert g1["json"]["islands"][0]["rim_m"] == g2["json"]["islands"][0]["rim_m"]


def test_cpp_generate_passes_checks(small_ctx):
    """整条 generate（地形 → 资源 → 四季 → 天气 → 聚落）的产物照常写出、硬项全过、重跑哈希一致。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.check import evaluate, hash_products
    node = _nodes(small_ctx, 1)[0]
    out, g = isl.generate(small_ctx, node, res_m=300.0, log=lambda *a: None, return_state=True)
    h1 = hash_products(out)
    isl.generate(small_ctx, node, res_m=300.0, log=lambda *a: None)
    items = evaluate(g, out, ctx=small_ctx, node=node, c=small_ctx.cfg["island"], det_hashes=(h1, hash_products(out)), daily_years=10)
    bad = [i["id"] for i in items if not i["pass"] and i["hard"]]
    assert not bad, bad
