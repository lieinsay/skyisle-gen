"""行星计划 P6c：行星层 ①–④ 的 C++ 核心。

扩展没编（python core/build.py）时整个文件跳过。与 numpy 逐位比的公共件：np.fft 的 rfft 低通（pocketfft）、float32 的中位数；
日照一阶谐波、经度周期的分形值噪声、kNN 按性质验（kNN 对暴力的角距排序）；
小世界上跑 ①–④：缓存 key（固定混入 "+cpp"、与 [engine] 段无关）、同 key 命中、planet_run 一次跑完与逐步相同、改几个开关（Held–Hou、
orbit_to_calendar、残余降水噪声、剪切风暴标量幅度）照样跑通且只动该动的产物、npz 读回的对象与内存里的同值；
第三层从 C++ 的行星层对象取 NodeInputs（与 npz 同值），内存里的对象与 npz 读回的对象生成的岛群逐字节相同。
Python 参考后端 2026-09-30 删了（git tag python-reference-final），两个后端逐位的对照随之删掉。
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")

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


def test_insolation_first_harmonic():
    """日照一年的一阶谐波振幅：南北对称；赤道与倾角 0 时为 0（赤道一年两个峰）；中纬度随倾角与纬度变大。"""
    lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    a34 = core.insolation_first_harmonic(lats, 34.0, 360)
    assert np.allclose(a34, a34[::-1]) and abs(a34[2]) < 1e-9
    assert 0 < a34[3] < a34[4]
    assert np.abs(core.insolation_first_harmonic(lats, 0.0, 360)).max() < 1e-9
    a20 = core.insolation_first_harmonic(lats, 20.0, 360)
    assert (a20[3:] < a34[3:]).all()


def test_planet_fractal_noise():
    """经度周期的分形值噪声：同种子相同、换流号不同、东西两边接得上（首末列之差不比相邻列大）。"""
    for base, oct_, pers in ((8, 5, 0.55), (5, 3, 0.55), (6, 3, 0.5)):
        a = core.planet_fractal_noise(42, 3, 180, 360, base, oct_, pers, 2.0)
        assert a.shape == (180, 360) and np.isfinite(a).all()
        assert np.array_equal(a, core.planet_fractal_noise(42, 3, 180, 360, base, oct_, pers, 2.0))
        assert np.abs(a - core.planet_fractal_noise(42, 4, 180, 360, base, oct_, pers, 2.0)).max() > 0.1
        assert np.abs(a[:, 0] - a[:, -1]).mean() < 2.0 * np.abs(np.diff(a, axis=1)).mean(), base


@pytest.mark.parametrize("n", [600, 1000, 5000])
def test_knn(n):
    """kNN（③）：每点的 k 个近邻与暴力的角距排序一样、角距升序。"""
    from skyisle_gen.sphere import latlon_to_xyz
    rng = np.random.default_rng(n)
    xyz = np.ascontiguousarray(latlon_to_xyz(np.degrees(np.arcsin(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n)))
    idx, ang = core.planet_knn(xyz, 9)
    assert idx.shape == ang.shape == (n, 9)
    assert (np.diff(ang, axis=1) >= 0).all()
    for i in range(0, n, max(1, n // 60)):
        d = np.arccos(np.clip(xyz @ xyz[i], -1.0, 1.0))
        d[i] = np.inf
        ref = np.argsort(d, kind="stable")[:9]
        assert idx[i].tolist() == ref.tolist(), i
        assert np.allclose(ang[i], d[ref], atol=1e-9), i


def test_median_f32_matches():
    rng = np.random.default_rng(3)
    for n in (1, 2, 7, 8000, 8001):
        a = (rng.lognormal(7, 1, n)).astype(np.float32)
        assert core.median_f32(a) == float(np.median(a)), n


# ---------------------------------------------------------------- 小世界：①–④
def _run(root, sets=(), run_id="cpp"):
    cfg = load_config(sets=SMALL + list(sets) + [f"run.id={run_id}"])
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
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p6c")
    E.clear_cache()
    return _run(root)


def test_cache_keys(world):
    """①–④ 的 key 固定混入 "+cpp"（删 Python 参考后端前的 cpp run 照旧命中），与 [engine] 段（线程数、旧的 backend 键）无关。"""
    cfg, out = world
    for _i, st in STAGES[:4]:
        assert _meta(out, st)["engine"] == "cpp", st
    keys = _stage_key_chain(cfg, 7)
    cfg0 = copy.deepcopy(cfg)
    cfg0.pop("engine", None)
    assert _stage_key_chain(cfg0, 7) == keys
    cfg1 = copy.deepcopy(cfg)
    cfg1["engine"] = {"backend": "cpp", "threads": 1}
    assert _stage_key_chain(cfg1, 7) == keys
    assert [_meta(out, st)["stage_key"] for _i, st in STAGES[:4]] == keys[:4]


def test_cache_hit(world):
    _, out = world
    before = {st: _meta(out, st)["seconds"] for _i, st in STAGES[:4]}
    logs = []
    cfg = load_config(sets=SMALL + ["run.id=cpp"])
    run(cfg, 7, out.parent, upto=4, explain=True, log=logs.append)
    assert all("cache hit" in x for x in logs), logs
    assert before == {st: _meta(out, st)["seconds"] for _i, st in STAGES[:4]}


def test_planet_run_equals_stagewise(world):
    cfg, out = world
    pc = core.make_config(E.planet_config(cfg))
    P, W, I, C = core.planet_run(pc, 7)
    ia = core.islands_arrays(I)
    with np.load(out / "s03_islands" / "islands.npz") as z:
        assert np.array_equal(ia["lat"], z["lat"]) and np.array_equal(ia["area_km2"].astype(np.float32), z["area_km2"])
    ca = core.climate_arrays(C)
    with np.load(out / "s04_climate" / "climate_grid.npz") as z:
        for k in ("precip", "temp", "storm", "season_range"):
            assert np.array_equal(ca[k].astype(np.float32), z[k]), k


@pytest.mark.parametrize("sets,first", [
    (["s01.planet.held_hou_scaling=true", "s01.planet.rotation_period_hr=30.0"], "s01_planet"),
    (["s01.calendar.mode=orbit_to_calendar", "s01.calendar.moon=false"], "s01_planet"),
    (["s04.climate.precip_noise_amp=0.05", "s04.climate.storm_shear_amp=0.3", "s04.climate.river_capacity_bonus=0.5"], "s04_climate"),
])
def test_variants(world, tmp_path, sets, first):
    """改开关：照样跑通，改动从该改的那一步起才出现（前面的步产物不变）。"""
    _, base = world
    _, out = _run(tmp_path, sets, "var")
    pa, pb = _products(base), _products(out)
    assert pa.keys() == pb.keys()
    diff = sorted({k.split("/")[0] for k in pa if pa[k] != pb[k]})
    assert diff and diff[0] == first, (sets, diff)
    if "orbit_to_calendar" in " ".join(sets):
        planet = json.loads((out / "s01_planet" / "planet.json").read_text(encoding="utf-8"))
        assert planet["calendar"]["mode"] == "orbit_to_calendar"


def test_loaded_parts_equal_memory(world):
    """npz 读回的 C++ 对象与内存里的同值（第三层两条路径的前提）。"""
    cfg, out = world
    ctx = Context(cfg, 7, out)
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
def test_island_inputs_from_cpp_parts(world):
    """C++ 给的 NodeInputs 与前端从 ③④ 的 npz 读的（island._node_inputs）同值。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    cfg, out = world
    ctx = Context(copy.deepcopy(cfg), 7, out)
    isl.island_config(ctx)
    assert IE._parts(ctx) is not None
    n = ctx.load_npz(3, "islands")["lat"].size
    for node in list(range(0, n, 97)) + [n - 1]:
        inp = isl._node_inputs(ctx, node)
        d = IE.inputs(ctx, node, inp)
        ref = {"node": int(node), "seed": 7, "lat": inp["lat"], "lon": inp["lon"], "area_km2": inp["area_km2"],
               "main_area_km2": inp["main_area_km2"], "height_m": inp["height_m"], "age": inp["age"], "layered": bool(inp["layered"]),
               "keel_clearance_m": inp["keel_clearance_m"], "area_median_km2": inp["area_median_km2"],
               "precip": inp["precip"], "temp_sea": inp["temp_sea"], "lapse_c_per_km": inp["lapse_c_per_km"],
               "arable_frac": inp["arable_frac"], "river_size": inp["river_size"], "has_river": bool(inp["has_river"])}
        for k in ("temp", "storm", "window", "season_range", "season_range_sea", "temp_winter", "temp_summer"):
            if k in inp:
                ref[k] = float(inp[k])
        assert {k: d[k] for k in ref} == ref, node


def _island_products(d: Path) -> dict:
    res = {}
    for p in sorted(d.iterdir()):
        if p.name == "island.json":
            J = json.loads(p.read_text(encoding="utf-8"))
            J["meta"].pop("seconds", None)
            res[p.name] = json.dumps(J, sort_keys=True, ensure_ascii=False)
        elif p.suffix in (".npz", ".png", ".csv", ".json"):
            res[p.name] = p.read_bytes()
    return res


def test_island_generate_in_memory_world(world, tmp_path):
    """同一进程里跑过 ①–④：第三层直接用内存里的 C++ 对象（不读 npz）；清掉进程内缓存后从 npz 读回，生成的岛群逐字节相同。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    _, out = world
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    cfg = load_config(sets=SMALL + ["run.id=cpp"])
    run(cfg, 7, out.parent, upto=4, force_from=1, log=lambda *a: None)      # 重算一次，把各步对象留在内存里
    run_dir = str(out.resolve())
    keys = {i: _meta(out, st)["stage_key"] for i, st in STAGES[:4]}
    assert all((run_dir, i, keys[i]) in E._PARTS for i in (1, 2, 3, 4))
    ctx = Context(cfg, 7, out)
    isl_npz = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    ok = np.where(cli["has_river"] & (isl_npz["main_area_km2"] < 1500) & (isl_npz["main_area_km2"] > 200))[0]
    node = int(ok[0])
    a = _island_products(isl.generate(ctx, node, res_m=300.0, log=lambda *a: None, out_root=tmp_path / "mem"))
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    ctx2 = Context(load_config(sets=SMALL + ["run.id=cpp"]), 7, out)
    b = _island_products(isl.generate(ctx2, node, res_m=300.0, log=lambda *a: None, out_root=tmp_path / "disk"))
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
