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


def test_kmeans_split_matches():
    """田块切分（settle._kmeans_split）：不放回抽初值 + 12 轮 Lloyd，均值沿 axis 0 顺序加。"""
    from skyisle_gen.island.settle import _kmeans_split
    rng = np.random.default_rng(5)
    for t in range(40):
        n = int(rng.integers(5, 3000))
        ii, jj = rng.integers(0, 200, n), rng.integers(0, 200, n)
        k = int(rng.integers(1, 12))
        key = f"island:{t}:settle:fields"
        a = _kmeans_split(entity_rng(7, 21, key), ii, jj, k)
        b = core.kmeans_split(7, key, ii.astype(np.int32).tolist(), jj.astype(np.int32).tolist(), k)
        assert np.array_equal(np.asarray(a), b), t


# ---------------------------------------------------------------- 小世界：整群 generate 两个后端逐字节对照
SMALL = ["s03.islands.n_islands=1600"]


@pytest.fixture(scope="module")
def small_ctx(tmp_path_factory):
    from skyisle_gen.config import load_config
    from skyisle_gen.pipeline import Context, run
    root = tmp_path_factory.mktemp("p6b")
    cfg = load_config(sets=SMALL + ["run.id=p6b"])
    out = run(cfg, 7, root, upto=4)
    return Context(cfg, 7, out)


def _nodes(ctx, k=3):
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    ok = np.where(cli["has_river"] & (isl["main_area_km2"] < 2500) & (isl["main_area_km2"] > 200))[0]
    dry = np.where(~cli["has_river"] & (isl["main_area_km2"] < 2500) & (isl["main_area_km2"] > 200))[0]
    return [int(x) for x in ok[:k]] + [int(x) for x in dry[:1]]      # 有河的几群 + 一个无河的（蓄水池、祭台的另一条路）


def _products(out):
    import json
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


def _gen(ctx, node, backend, root, threads=4, **kw):
    from skyisle_gen import island as isl
    out, g = isl.generate(ctx, node, res_m=300.0, sets=[f"engine.backend={backend}", f"engine.threads={threads}"], log=lambda *a: None,
                          return_state=True, out_root=root, **kw)
    ctx.cfg["engine"]["backend"] = "python"
    return out, g


def test_generate_products_identical(small_ctx, tmp_path):
    """整群 generate（地形 → 资源 → 四季 → 天气 → 聚落）：两个后端的整套产物逐字节相同（island.json 只差 meta.seconds / engine）。"""
    for node in _nodes(small_ctx):
        a, ga = _gen(small_ctx, node, "python", tmp_path / "py")
        b, gb = _gen(small_ctx, node, "cpp", tmp_path / "cpp")
        pa, pb = _products(a), _products(b)
        assert sorted(pa) == sorted(pb), node
        bad = [k for k in pa if pa[k] != pb[k]]
        assert not bad, (node, bad)
        assert gb["json"]["meta"]["engine"] == "cpp"
        for k in ("terrain_zone", "patch_id", "resource", "res_field", "settle_raster", "settle_fields", "landcover"):
            assert ga[k].dtype == gb[k].dtype and np.array_equal(ga[k], gb[k]), (node, k)
        assert ga["settle"] == gb["settle"] and ga["resources"] == gb["resources"] and ga["climate"] == gb["climate"]
        for k in ("day", "season", "temp_c", "wind_u", "wind_v", "precip_mm"):
            assert np.array_equal(ga["daily"][k], gb["daily"][k]), (node, k)


def test_generate_p3_resources_identical(small_ctx, tmp_path):
    """P3（没有火山）的新资源（骨架空洞、只在新岛的温泉、热泉硫磺、按剥蚀深浅的石料岩性、岩盐 / 盐泉 / 盐井 / 盐井村、贝壳化石）：
    把新岛门槛、盐丘与化石的密度调高让每样都出得来，两个后端的整套产物仍逐字节相同。"""
    from skyisle_gen import island as isl
    extra = ["island.terrain.age_young=0.6", "island.resources.density_per_100km2.salt=3.0", "island.resources.fossil_per_km2=1.0"]
    node = _nodes(small_ctx, 1)[0]
    outs = {}
    for b in ("python", "cpp"):
        out, g = isl.generate(small_ctx, node, res_m=300.0, sets=[f"engine.backend={b}", "engine.threads=4"] + extra, log=lambda *a: None,
                              return_state=True, out_root=tmp_path / b)
        small_ctx.cfg["engine"]["backend"] = "python"
        outs[b] = (_products(out), g)
    (pa, ga), (pb, gb) = outs["python"], outs["cpp"]
    kinds = {d["kind"] for d in ga["resources"]["deposits"]} | {o["kind"] for o in ga["resources"]["occurrences"]}
    assert {"salt", "saltspring", "fossil", "hotspring"} <= kinds, kinds
    assert sorted(pa) == sorted(pb)
    assert not [k for k in pa if pa[k] != pb[k]]
    assert ga["resources"] == gb["resources"] and ga["settle"] == gb["settle"]


def test_generate_p5_farmland_identical(small_ctx, tmp_path):
    """P5（宜垦 / 已垦 / 撂荒、定居门槛、没人住 ≠ 没人用、荒地归谁）：小世界里没人住的岛多半已有工棚，特殊用途分两遍走——
    一遍放宽放牧（夏牧、烽火台、废村、工棚），一遍关掉放牧与烽火台、庙与墓岛必有；每条代码路径走到，两个后端的整套产物仍逐字节相同。"""
    from skyisle_gen import island as isl
    common = ["island.settle.ruin_min_hh=4", "island.settle.graze_min_km2=0.05", "island.settle.graze_reach_km=1e3", "island.settle.shieling_min_km2=0.5"]
    passes = [common + ["island.settle.graze_max=1"], common + ["island.settle.graze_max=0", "island.settle.beacon_max=0", "island.settle.shrine_p=1.0", "island.settle.tomb_p=1.0",
                                "island.settle.tomb_max_km2=1e6"]]
    base = ["island.settle.ruin_min_hh=8", "island.settle.graze_min_km2=0.5", "island.settle.graze_reach_km=10.0", "island.settle.shieling_min_km2=3.0",
            "island.settle.graze_max=4", "island.settle.beacon_max=2", "island.settle.shrine_p=0.5", "island.settle.tomb_p=0.35", "island.settle.tomb_max_km2=5.0"]
    seen = set()
    for extra in passes:
        for node in _nodes(small_ctx, 3)[1:3]:
            outs = {}
            for b in ("python", "cpp"):
                out, g = isl.generate(small_ctx, node, res_m=300.0, sets=[f"engine.backend={b}", "engine.threads=4"] + extra, log=lambda *a: None,
                                      return_state=True, out_root=tmp_path / b)
                small_ctx.cfg["engine"]["backend"] = "python"
                outs[b] = (_products(out), g)
            (pa, ga), (pb, gb) = outs["python"], outs["cpp"]
            assert sorted(pa) == sorted(pb)
            assert not [k for k in pa if pa[k] != pb[k]], node
            assert ga["settle"] == gb["settle"]
            S = gb["settle"]
            seen |= {u["kind"] for u in S["uses"]} | {x["occupancy"] for x in S["specials"]} | ({"废村"} if S["ruins"] else set())
            seen |= {"保底"} if S["farmland"]["floor_islands"] else set()
    isl.island_config(small_ctx, base)                      # --set 会留在 ctx 上：改回默认
    assert {"烽火台", "庙", "墓岛", "废村", "工棚", "保底"} <= seen and seen & {"放牧", "夏牧"}, seen


def test_generate_cpp_thread_independent(small_ctx, tmp_path):
    node = _nodes(small_ctx, 1)[0]
    a, _ = _gen(small_ctx, node, "cpp", tmp_path / "t1", threads=1)
    b, _ = _gen(small_ctx, node, "cpp", tmp_path / "t4", threads=4)
    assert _products(a) == _products(b)


def test_steps_partial_match(small_ctx, tmp_path):
    """只算到资源 / 四季 / 天气（steps 2–4）时两个后端也相同。"""
    node = _nodes(small_ctx, 1)[0]
    for steps in (2, 3, 4):
        a, _ = _gen(small_ctx, node, "python", tmp_path / f"p{steps}", steps=steps)
        b, _ = _gen(small_ctx, node, "cpp", tmp_path / f"c{steps}", steps=steps)
        assert _products(a) == _products(b), steps


def test_lod_block_reduce_matches(small_ctx):
    from skyisle_gen import island as isl
    from skyisle_gen.island.lod import build_lod
    node = _nodes(small_ctx, 1)[0]
    outs = {}
    for b in ("python", "cpp"):
        c = isl.island_config(small_ctx, [f"engine.backend={b}"])
        outs[b] = build_lod(small_ctx, node, c, [1600.0, 3200.0], native_res_m=400.0)
    small_ctx.cfg["engine"]["backend"] = "python"
    for res in outs["python"]:
        A, B = outs["python"][res][0], outs["cpp"][res][0]
        assert sorted(A) == sorted(B) and "weather_type" in A and "weather_meta" in A     # 默认带天气（C++ 的 weather_year）
        for k in A:
            assert A[k].dtype == B[k].dtype and np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f"), (res, k)


def test_is_daily_and_climate_only_match(small_ctx, tmp_path):
    """IS-daily 的多年逐日（C++ 的 weather_years）与全量季型（climate_only）与 Python 版同值。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.climate import build_climate
    from skyisle_gen.island.engine import climate_only_cpp
    from skyisle_gen.island.weather import multi_year_stats
    node = _nodes(small_ctx, 1)[0]
    _, g = _gen(small_ctx, node, "python", tmp_path / "d")
    c = isl.island_config(small_ctx)
    st_py = multi_year_stats(small_ctx, node, c, g, years=10)
    c = isl.island_config(small_ctx, ["engine.backend=cpp"])
    st_cpp = multi_year_stats(small_ctx, node, c, g, years=10)
    small_ctx.cfg["engine"]["backend"] = "python"
    assert st_py == st_cpp
    isl_npz = small_ctx.load_npz(3, "islands")
    cli = small_ctx.load_npz(4, "climate_islands")
    planet = small_ctx.load_json(1, "planet")
    for j in range(0, isl_npz["lat"].size, 97):
        inp = {k: float(isl_npz[k][j]) for k in ("lat", "lon", "height_m")}
        for k in ("precip", "temp", "storm", "window", "temp_sea", "season_range", "season_range_sea", "temp_winter", "temp_summer"):
            inp[k] = float(cli[k][j])
        inp["planet"] = planet
        inp["keel_clearance_m"] = float(small_ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
        g2 = {"inp": inp, "json": {}}
        build_climate(small_ctx, j, c, g2, log=lambda *a: None)
        assert climate_only_cpp(small_ctx, inp, c) == g2["climate"], j
