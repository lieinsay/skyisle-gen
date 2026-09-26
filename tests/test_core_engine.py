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
