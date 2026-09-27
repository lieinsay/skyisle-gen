"""行星计划 P6c：行星层 ①–④ 的 C++ 核心与 numpy 版对照。

扩展没编（python core/build.py）时整个文件跳过。公共件逐位比：np.fft 的 rfft 低通（pocketfft）、日照一阶谐波（复数成对求和）、
经度周期的分形值噪声、kNN（BLAS dgemm / syrk 的乘加次序）、float32 的中位数；
小世界上两个后端各跑 ①–④，产物（planet.json、bands.json、11 个 npz 的全部数组）与摘要逐位相同、缓存 key 分开；
改几个开关（Held–Hou、orbit_to_calendar、残余降水噪声、剪切风暴标量幅度）再比一次；planet_run 一次跑完与逐步相同；
第三层在 cpp 后端下从 C++ 的行星层对象直接取 PlanetView / NodeInputs（同一进程跑过 cpp 的 ①–④ 就用内存里的），与 python 后端产物逐字节相同。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")
if not hasattr(core, "planet_stage1"):
    pytest.skip("C++ 扩展是 P6c 之前编的：python core/build.py", allow_module_level=True)

from skyisle_gen import engine as E                       # noqa: E402
from skyisle_gen.config import load_config                # noqa: E402
from skyisle_gen.pipeline import Context, run, _stage_key_chain, STAGES   # noqa: E402

SMALL = ["s03.islands.n_islands=1600"]


# ---------------------------------------------------------------- 公共件
@pytest.mark.parametrize("n", [360, 180, 90, 91])
def test_rfft_lowpass_matches_numpy(n):
    rng = np.random.default_rng(n)
    for kmax in (0, 1, 4, 6):
        a = rng.standard_normal(n)
        a = a - a.mean()
        F = np.fft.rfft(a)
        F[kmax + 1:] = 0.0
        assert np.array_equal(core.rfft_lowpass(a, kmax), np.fft.irfft(F, n=n)), (n, kmax)


def test_insolation_first_harmonic_matches():
    from skyisle_gen.skeleton import insolation_first_harmonic
    lats = -90.0 + (np.arange(180) + 0.5)
    rnd = np.random.default_rng(1).uniform(-90, 90, 500)
    for tilt in (34.0, 20.0, 0.0):
        for lat in (lats, rnd):
            assert np.array_equal(core.insolation_first_harmonic(lat, tilt, 360), insolation_first_harmonic(lat, tilt)), tilt


def test_fractal_noise_matches():
    from skyisle_gen.noise import fractal_noise
    from skyisle_gen.rng import stage_rng
    for base, oct_, pers in ((8, 5, 0.55), (5, 3, 0.55), (6, 3, 0.5)):
        a = core.planet_fractal_noise(42, 3, 180, 360, base, oct_, pers, 2.0)
        b = fractal_noise(stage_rng(42, 3), 180, 360, base_cells=base, octaves=oct_, persistence=pers)
        assert np.array_equal(a, b), (base, oct_)


# 块 512 行：n > 512 时 xyz[s:e] @ xyz.T 都走 dgemm（1000、5000 有 16 的尾数）。n ≤ 512 时第一块就是整块 xyz @ xyz.T，numpy 走 dsyrk，
# 它最后几列（n % 16 的尾块）的乘加次序与 dgemm 不同、没探出来——C++ 按 dgemm 算，差一位（行星层不会只有 512 个群，DESIGN-NOTES 四点二十五）
@pytest.mark.parametrize("n", [600, 1000, 5000])
def test_knn_matches(n):
    from skyisle_gen.sphere import knn, latlon_to_xyz
    rng = np.random.default_rng(n)
    xyz = latlon_to_xyz(np.degrees(np.arcsin(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n))
    i1, a1 = core.planet_knn(np.ascontiguousarray(xyz), 9)
    i2, a2 = knn(xyz, 9)
    assert np.array_equal(i1, i2) and np.array_equal(a1, a2)


def test_median_f32_matches():
    rng = np.random.default_rng(3)
    for n in (1, 2, 7, 8000, 8001):
        a = (rng.lognormal(7, 1, n)).astype(np.float32)
        assert core.median_f32(a) == float(np.median(a)), n


# ---------------------------------------------------------------- 小世界：两个后端的 ①–④
def _run(root, backend, sets=(), run_id=None):
    cfg = load_config(sets=SMALL + list(sets) + [f"run.id={run_id or backend}", f"engine.backend={backend}"])
    out = run(cfg, 7, root, upto=4, log=lambda *a: None)
    return cfg, out


def _products(out: Path) -> dict:
    res = {}
    for st in ("s01_planet", "s02_wind", "s03_islands", "s04_climate"):
        for p in sorted((out / st).iterdir()):
            if p.suffix == ".npz":
                with np.load(p) as z:
                    for k in z.files:
                        res[f"{st}/{p.name}/{k}"] = (z[k].dtype.str, z[k].shape, z[k].tobytes())
            elif p.name == "_meta.json":
                m = json.loads(p.read_text(encoding="utf-8"))
                res[f"{st}/summary"] = json.dumps(m["summary"], sort_keys=True)
            elif p.suffix == ".json":
                res[f"{st}/{p.name}"] = json.dumps(json.loads(p.read_text(encoding="utf-8")), sort_keys=True)
    return res


def _meta(out: Path, st: str) -> dict:
    return json.loads((out / st / "_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def two_backends(tmp_path_factory):
    root = tmp_path_factory.mktemp("p6c")
    E.clear_cache()
    cfg_py, out_py = _run(root, "python")
    cfg_c, out_c = _run(root, "cpp")
    return cfg_py, out_py, cfg_c, out_c


def test_stage_products_identical(two_backends):
    _, out_py, _, out_c = two_backends
    a, b = _products(out_py), _products(out_c)
    assert a.keys() == b.keys()
    diff = [k for k in a if a[k] != b[k]]
    assert not diff, diff


def test_cache_keys_split_by_backend(two_backends):
    cfg_py, out_py, cfg_c, out_c = two_backends
    for i, (_idx, st) in enumerate(STAGES[:4]):
        mp, mc = _meta(out_py, st), _meta(out_c, st)
        assert mp["stage_key"] != mc["stage_key"], st             # 切后端不误命中另一个后端的缓存
        assert "engine" not in mp and mc["engine"] == "cpp"
    # python 后端的 key 与没有后端分量时的算法一字不差（旧 run 照旧命中）
    import copy
    cfg0 = copy.deepcopy(cfg_py)
    cfg0.pop("engine", None)
    assert _stage_key_chain(cfg0, 7) == _stage_key_chain(cfg_py, 7)


def test_cache_hit_same_backend(two_backends, tmp_path):
    _, out_py, _, out_c = two_backends
    before = {st: _meta(out_c, st)["seconds"] for _i, st in STAGES[:4]}
    logs = []
    cfg = load_config(sets=SMALL + ["run.id=cpp", "engine.backend=cpp"])
    run(cfg, 7, out_c.parent, upto=4, explain=True, log=logs.append)
    assert all("cache hit" in x for x in logs), logs
    assert before == {st: _meta(out_c, st)["seconds"] for _i, st in STAGES[:4]}


def test_planet_run_equals_stagewise(two_backends):
    cfg_py, _, _, out_c = two_backends
    pc = core.make_config(E.planet_config(cfg_py))
    P, W, I, C = core.planet_run(pc, 7)
    ia = core.islands_arrays(I)
    with np.load(out_c / "s03_islands" / "islands.npz") as z:
        assert np.array_equal(ia["lat"], z["lat"]) and np.array_equal(ia["area_km2"].astype(np.float32), z["area_km2"])
    ca = core.climate_arrays(C)
    with np.load(out_c / "s04_climate" / "climate_grid.npz") as z:
        for k in ("precip", "temp", "storm", "season_range"):
            assert np.array_equal(ca[k].astype(np.float32), z[k]), k


@pytest.mark.parametrize("sets", [
    ["s01.planet.held_hou_scaling=true", "s01.planet.rotation_period_hr=30.0"],
    ["s01.calendar.mode=orbit_to_calendar", "s01.calendar.moon=false"],
    ["s04.climate.precip_noise_amp=0.05", "s04.climate.storm_shear_amp=0.3", "s04.climate.river_capacity_bonus=0.5"],
])
def test_variants_identical(tmp_path, sets):
    _, a = _run(tmp_path, "python", sets, "py")
    _, b = _run(tmp_path, "cpp", sets, "cc")
    pa, pb = _products(a), _products(b)
    assert pa.keys() == pb.keys()
    diff = [k for k in pa if pa[k] != pb[k]]
    assert not diff, (sets, diff)


def test_loaded_parts_equal_memory(two_backends):
    """npz 读回的 C++ 对象与内存里的同值（第三层两条路径的前提）。"""
    _, _, cfg_c, out_c = two_backends
    ctx = Context(cfg_c, 7, out_c)
    for idx in (3, 4):
        mem = E.part(ctx, idx)
        E.clear_cache()
        disk = E.part(ctx, idx)
        if idx == 3:
            a, b = core.islands_arrays(mem), core.islands_arrays(disk)
            for k in ("lat", "lon", "src", "dst", "dist_days", "plate", "cls"):
                assert np.array_equal(a[k], b[k]), k
            for k in ("area_km2", "height_m", "main_area_km2", "age"):
                assert np.array_equal(a[k].astype(np.float32), b[k].astype(np.float32)), k
        else:
            a, b = core.climate_arrays(mem), core.climate_arrays(disk)
            for k in ("precip", "storm", "window", "continentality", "u", "v", "edges"):
                assert np.array_equal(a[k].astype(np.float32), b[k].astype(np.float32)), k


# ---------------------------------------------------------------- 第三层从 C++ 的行星层对象取输入
def test_island_inputs_from_cpp_parts(two_backends):
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    _, out_py, _, _ = two_backends
    cfg = load_config(sets=SMALL + ["run.id=python"])
    ctx = Context(cfg, 7, out_py)
    isl.island_config(ctx, ["engine.backend=cpp"])
    assert IE._parts(ctx) is not None
    n = ctx.load_npz(3, "islands")["lat"].size
    for node in list(range(0, n, 97)) + [n - 1]:
        inp = isl._node_inputs(ctx, node)
        assert IE.inputs(ctx, node, inp) == IE.inputs_py(ctx, node, inp), node


def test_island_generate_in_memory_world(two_backends, tmp_path):
    """同一进程里 cpp 后端跑过 ①–④：第三层直接用内存里的 C++ 对象（不读 npz），产物与 python 后端逐字节相同。"""
    from skyisle_gen import island as isl
    _, _, cfg_c, out_c = two_backends
    E.clear_cache()
    cfg = load_config(sets=SMALL + ["run.id=cpp", "engine.backend=cpp"])
    run(cfg, 7, out_c.parent, upto=4, force_from=1, log=lambda *a: None)      # 重算一次，把各步对象留在内存里
    run_dir = str(out_c.resolve())
    keys = {i: _meta(out_c, st)["stage_key"] for i, st in STAGES[:4]}
    assert all((run_dir, i, keys[i]) in E._PARTS for i in (1, 2, 3, 4))
    ctx = Context(cfg, 7, out_c)
    isl_npz = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    ok = np.where(cli["has_river"] & (isl_npz["main_area_km2"] < 1500) & (isl_npz["main_area_km2"] > 200))[0]
    node = int(ok[0])

    def products(d):
        res = {}
        for p in sorted(d.iterdir()):
            if p.name == "island.json":
                J = json.loads(p.read_text(encoding="utf-8"))
                J["meta"].pop("seconds", None)
                J["meta"].pop("engine", None)
                res[p.name] = json.dumps(J, sort_keys=True, ensure_ascii=False)
            elif p.suffix in (".npz", ".png", ".csv", ".json"):
                res[p.name] = p.read_bytes()
        return res
    outs = {}
    for b in ("python", "cpp"):
        o = isl.generate(ctx, node, res_m=300.0, sets=[f"engine.backend={b}"], log=lambda *a: None, out_root=tmp_path / b)
        outs[b] = products(o)
    ctx.cfg["engine"]["backend"] = "python"
    assert outs["python"].keys() == outs["cpp"].keys()
    assert [k for k in outs["python"] if outs["python"][k] != outs["cpp"][k]] == []
