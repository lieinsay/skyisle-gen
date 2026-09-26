"""行星计划 P6b：第三层其余部分（资源、聚落与层级、四季、逐日天气、粗版降采样、整群 generate）的 C++ 核心与 numpy 版对照。

扩展没编（python core/build.py）时整个文件跳过。公共件逐位比：指数 ziggurat / 伽马（形状 < 1）/ 对数正态 / 泊松 / 不放回抽签、
np.quantile / np.interp / np.convolve（BLAS ddot）/ float32 成对求和、label_by_island / window_extrema、田块的 k-means；
整群的两个后端对照（产物逐字节、island check）在文件后半。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")

from skyisle_gen.island import grid as G          # noqa: E402
from skyisle_gen.rng import entity_rng             # noqa: E402

KEY = "island:2051:weather:0"


# ---------------------------------------------------------------- 随机流：P6b 新用到的分布
def test_exponential_gamma_lognormal_match():
    r = entity_rng(42, 21, KEY)
    assert np.array_equal(core.rng_draw(42, 21, KEY, "exponential", 100000), r.standard_exponential(100000))
    r = entity_rng(42, 21, KEY)
    assert np.array_equal(core.rng_draw(42, 21, KEY, "gamma2", 100000, 0.8, 1.0), r.gamma(0.8, 1.0, 100000))   # 形状 < 1：指数 ziggurat
    r = entity_rng(42, 21, KEY)
    assert np.array_equal(core.rng_draw(42, 21, KEY, "lognormal", 50000, 0.0, 0.6), r.lognormal(0.0, 0.6, 50000))


@pytest.mark.parametrize("lam", [0.0, 0.3, 2.5, 9.99, 10.0, 30.8, 123.4, 2000.0])
def test_poisson_matches(lam):
    r = entity_rng(42, 21, KEY)
    ref = np.array([r.poisson(lam) for _ in range(20000)], dtype=np.float64)
    assert np.array_equal(core.rng_draw(42, 21, KEY, "poisson", 20000, lam), ref)


@pytest.mark.parametrize("pop,size", [(5, 2), (10, 10), (100, 7), (3000, 40), (20000, 30), (20000, 500), (50000, 3000)])
def test_choice_without_replacement_matches(pop, size):
    """Floyd + 洗牌；总体 > 10000 且 size > pop // 50 时尾部洗牌。"""
    r = entity_rng(42, 21, KEY)
    ref = np.concatenate([r.choice(pop, size=size, replace=False) for _ in range(20)])
    assert np.array_equal(core.rng_choice_noreplace(42, 21, KEY, pop, size, 20), ref)


# ---------------------------------------------------------------- numpy 同式的小工具
def test_quantile_interp_convolve_sum32():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        n = int(rng.integers(1, 300))
        a = rng.normal(0, 1, n)
        q = float(rng.uniform(0, 1))
        assert core.np_quantile(a, q) == float(np.quantile(a, q))
    d = np.arange(336)
    mids = np.array([42.0, 126.0, 210.0, 294.0])
    x = np.concatenate([mids - 336, mids, mids + 336])
    y = np.tile(rng.normal(0, 3, 4), 3)
    assert np.array_equal(core.np_interp(d.astype(float), x, y), np.interp(d, x, y))
    for n in (5, 17, 57, 100, 131):
        ext = rng.normal(0, 3, 336 + n - 1)
        ker = np.ones(n) / n
        assert np.array_equal(core.np_convolve_valid(ext, ker), np.convolve(ext, ker, mode="valid")), n   # BLAS ddot 的乘加次序
    for n in (3, 8, 100, 129, 400, 1600):
        f32 = rng.normal(500, 300, n).astype(np.float32)
        assert core.np_sum_f32(f32) == float(np.sum(f32))


def test_label_by_island_and_window_extrema():
    rng = np.random.default_rng(4)
    for _ in range(20):
        H, W = 60, 80
        iid = np.full((H, W), -1, np.int16)
        iid[5:30, 5:40] = 0
        iid[20:55, 35:75] = 1
        iid[40:58, 2:30] = 2
        m = (rng.random((H, W)) < 0.6) & (iid >= 0)
        a1, n1 = G.label_by_island(m, iid, 8)
        a2, n2 = core.label_by_island(m, iid, 8)
        assert n1 == n2 and np.array_equal(a1, a2)
        h = rng.normal(0, 100, (H, W))
        hi1, lo1 = G.window_extrema(h, 4, iid >= 0)
        hi2, lo2 = core.window_extrema(h, 4, iid >= 0)
        assert np.array_equal(hi1, hi2) and np.array_equal(lo1, lo2)
