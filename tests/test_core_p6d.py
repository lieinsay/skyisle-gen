"""行星计划 P6d：行星层 ⑤–⑨ 的 C++ 核心与 numpy 版对照。

扩展没编（python core/build.py）时整个文件跳过。公共件逐位比：Dijkstra（多源、有界、带 inf 的边、平局）、沿树累加、
Brandes 抽样介数（线程数无关）、弱连通分量、choice(n, size, replace=False, p)；
小世界上两个后端各跑 ①–⑨，⑤–⑨ 的产物（npz 的全部数组、json、history.md、摘要）逐位相同、缓存 key 分开；
改几组开关（fast 参考树、lat_band → band 障碍 + 政治性障碍、起源指定中心 + 指定变法之国、手工特征表）再比一次；
planet_run(upto=9) 一次跑完与逐步相同；从产物读回的 ⑤⑥⑦ 对象接着算 ⑧⑨ 与内存里的同值；
第三层的人口与邦都由 C++ 从 ⑨ 的对象给，与读 polity.npz 的 Python 版同值，内存里的 ①–⑨ 生成岛群与 python 后端产物逐字节相同。
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")
if not hasattr(core, "planet_stage9"):
    pytest.skip("C++ 扩展是 P6d 之前编的：python core/build.py", allow_module_level=True)

from skyisle_gen import engine as E                       # noqa: E402
from skyisle_gen import graph as G                        # noqa: E402
from skyisle_gen.config import load_config                # noqa: E402
from skyisle_gen.pipeline import Context, run, _stage_key_chain, STAGES   # noqa: E402

# 同 test_pipeline：1600 岛的小世界骨架窗内岛少，τ_c 放宽
SMALL = ["s03.islands.n_islands=1600", "s07.regions.n_regions=12", "s07.regions.max_regions=20",
         "s06.routes.betweenness_sources=48", "s08.k_sub=1", "s07.centers.tau_c=0.3"]
CIV = ("s05_barriers", "s06_routes", "s07_centers", "s08_diffusion", "s09_polity")


# ---------------------------------------------------------------- 公共件
def _graph(seed, n=300, m=1500, inf_frac=0.05, ties=False):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = rng.integers(1, 4, src.size).astype(np.float64) * 0.5 if ties else rng.uniform(0.1, 3.0, src.size)
    w[rng.random(src.size) < inf_frac] = np.inf
    return n, src.astype(np.int64), dst.astype(np.int64), w


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("ties", [False, True])
def test_dijkstra_matches(seed, ties):
    n, src, dst, w = _graph(seed, ties=ties)
    csr = G.CSR(n, src, dst)
    for sources, bound in (([0], None), ([5, 17, 5, 200], None), ([3], 2.0), ([1, 2], 0.7)):
        d1, p1, e1 = core.graph_dijkstra(n, src, dst, w, sources, bound)
        d2, p2, e2 = G.dijkstra(csr, w, sources, max_dist=bound)
        assert np.array_equal(d1, d2) and np.array_equal(p1, p2) and np.array_equal(e1, e2), (seed, ties, sources, bound)


@pytest.mark.parametrize("ties", [False, True])
def test_betweenness_matches(ties):
    n, src, dst, w = _graph(9, n=400, m=2400, ties=ties)
    csr = G.CSR(n, src, dst)
    sources = np.sort(np.random.default_rng(1).choice(n, 40, replace=False))
    ref = G.betweenness_sampled(csr, w, src.size, sources, c_min=1.0)
    for threads in (1, 4, 7):
        assert np.array_equal(core.graph_betweenness(n, src, dst, w, sources.tolist(), 1.0, threads), ref), threads


def test_weak_components_matches():
    n, src, dst, _w = _graph(4, n=500, m=300)
    assert np.array_equal(core.graph_weak_components(n, src, dst), G.weak_components(n, src, dst))


def test_choice_noreplace_p_matches():
    from skyisle_gen.rng import stage_rng
    for seed in range(8):
        r = np.random.default_rng(seed)
        n = int(r.integers(20, 3000))
        p = r.random(n) ** 3
        p[r.random(n) < 0.2] = 0.0
        p /= p.sum()
        size = int(r.integers(1, min(int((p > 0).sum()), 300)))
        a = core.rng_choice_noreplace_p(seed, 6, p, size)
        b = stage_rng(seed, 6).choice(n, size=size, replace=False, p=p)
        assert np.array_equal(a, b), seed


# ---------------------------------------------------------------- 小世界：两个后端的 ①–⑨
def _cfg(backend, sets=(), run_id=None, mutate=None):
    cfg = load_config(sets=SMALL + list(sets) + [f"run.id={run_id or backend}", f"engine.backend={backend}"])
    if mutate:
        mutate(cfg)
    return cfg


def _run(root, backend, sets=(), run_id=None, mutate=None, upto=9):
    cfg = _cfg(backend, sets, run_id, mutate)
    return cfg, run(cfg, 7, root, upto=upto, log=lambda *a: None)


def _products(out: Path, stages=CIV) -> dict:
    res = {}
    for st in stages:
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
            elif p.suffix == ".md":
                res[f"{st}/{p.name}"] = p.read_text(encoding="utf-8").replace(out.name, "<run>")
    return res


def _meta(out: Path, st: str) -> dict:
    return json.loads((out / st / "_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def two_backends(tmp_path_factory):
    root = tmp_path_factory.mktemp("p6d")
    E.clear_cache()
    cfg_py, out_py = _run(root, "python")
    cfg_c, out_c = _run(root, "cpp")          # 各步的 C++ 对象留在内存里（engine._PARTS）
    return cfg_py, out_py, cfg_c, out_c


def test_stage_products_identical(two_backends):
    _, out_py, _, out_c = two_backends
    a, b = _products(out_py), _products(out_c)
    assert a.keys() == b.keys() and len(a) > 30
    diff = [k for k in a if a[k] != b[k]]
    assert not diff, diff


def test_vassals_found(two_backends):
    """附庸判定按邦号查都城间距离（原先拿都城节点号查，附庸几乎为零）；overlord 与 vassals 互相对得上。"""
    _, out_py, _, out_c = two_backends
    for out in (out_py, out_c):
        pol = json.loads((out / "s09_polity" / "polities.json").read_text(encoding="utf-8"))["polities"]
        vassal = {x["id"]: x["overlord"] for x in pol if x.get("overlord", -1) >= 0}
        assert len(vassal) >= 3, len(vassal)
        by_id = {x["id"]: x for x in pol}
        for s, t in vassal.items():
            assert s in by_id[t]["vassals"]


def test_cache_keys_split_by_backend(two_backends):
    cfg_py, out_py, _, out_c = two_backends
    for _idx, st in STAGES[:9]:
        mp, mc = _meta(out_py, st), _meta(out_c, st)
        assert mp["stage_key"] != mc["stage_key"], st
        assert "engine" not in mp and mc["engine"] == "cpp"
    cfg0 = copy.deepcopy(cfg_py)
    cfg0.pop("engine", None)                   # 没有 [engine] 段的旧配置 = python：key 一字不差
    assert _stage_key_chain(cfg0, 7) == _stage_key_chain(cfg_py, 7)


def test_planet_run_upto9_equals_stagewise(two_backends):
    cfg_py, _, _, out_c = two_backends
    pc = core.make_config(E.planet_config(cfg_py))
    objs = core.planet_run(pc, 7, upto=9, threads=3)
    assert len(objs) == 9
    B, R, Ce, D, Pol = objs[4:]
    with np.load(out_c / "s05_barriers" / "perm.npz") as z:
        assert np.array_equal(core.barriers_arrays(B)["perm"], z["perm"])
    with np.load(out_c / "s06_routes" / "routes.npz") as z:
        ra = core.routes_arrays(R)
        assert np.array_equal(ra["cost_m"], z["cost_m"]) and np.array_equal(ra["flow"].astype(np.float32), z["flow"])
    with np.load(out_c / "s07_centers" / "prehist.npz") as z:
        assert np.array_equal(core.centers_arrays(Ce)["dist_pre"], z["dist_pre"])
    with np.load(out_c / "s08_diffusion" / "fields.npz") as z:
        assert np.array_equal(core.diffusion_arrays(D)["share"].astype(np.float32), z["share"])
    with np.load(out_c / "s09_polity" / "polity.npz") as z:
        pa = core.polity_arrays(Pol)
        for k in ("state", "polity", "fief", "realm", "capital"):
            assert np.array_equal(pa[k], z[k]), k
        assert np.array_equal(pa["pop"].astype(np.float32), z["pop"])
    # ⑧ 可跳过：⑨ 照样算（游戏只要人口与邦时）
    objs2 = core.planet_run(pc, 7, upto=9, skip_diffusion=True)
    assert np.array_equal(core.polity_arrays(objs2[8])["state"], pa["state"])
    assert core.planet_run(pc, 7)[3] is not None and len(core.planet_run(pc, 7)) == 4   # 默认仍是 P6c 的 ①–④


def test_loaded_parts_equal_memory(two_backends):
    """⑤⑥⑦ 从 npz / json 读回的 C++ 对象接着算 ⑧⑨，与一路在内存里算的同值（缓存命中后换进程的路径）。"""
    _, _, cfg_c, out_c = two_backends
    ctx = Context(cfg_c, 7, out_c)
    pc = core.make_config(E.planet_config(cfg_c))
    E.clear_cache()
    P, I, C = E.part(ctx, 1), E.part(ctx, 3), E.part(ctx, 4)
    B, R, Ce = E.part(ctx, 5), E.part(ctx, 6), E.part(ctx, 7)
    d = core.diffusion_arrays(core.planet_stage8(pc, 7, I, B, R, Ce))
    with np.load(out_c / "s08_diffusion" / "fields.npz") as z:
        for k in ("reach", "share", "strength"):
            assert np.array_equal(d[k].astype(np.float32), z[k]), k
    pa = core.polity_arrays(core.planet_stage9(pc, I, C, B, R, Ce))
    with np.load(out_c / "s09_polity" / "polity.npz") as z:
        for k in ("state", "polity", "kind", "fief", "realm", "circle", "capital"):
            assert np.array_equal(pa[k], z[k]), k
        for k in ("pop", "control", "dist_cap", "pop_state"):
            assert np.array_equal(pa[k].astype(np.float32), z[k]), k
    Pol = E.part(ctx, 9)                       # ⑨ 读回（第三层用）
    assert np.array_equal(core.polity_arrays(Pol)["capital"], pa["capital"])
    with pytest.raises(ValueError):
        E._load_part(ctx, 8)


def _band_and_political(cfg):
    cfg["s05"]["barriers"]["B"] = {"kind": "band", "band": "subtropical_calm_n", "type": "selective",
                                   "permeability": {"daily": 0.1, "trade": 0.5, "envoy": 0.8, "migrate": 0.1}}
    cfg["s05"]["barriers"]["C"] = {"kind": "band", "band": "westerlies_s", "type": "selective",
                                   "permeability": {"daily": 0.3, "trade": 0.6, "envoy": 0.9, "migrate": 0.2}}
    cfg["s05"]["political"] = {"overrides": [
        {"lat_range": [20.0, 45.0], "lon_range": [150.0, -170.0], "permeability": {"daily": 0.5, "trade": 0.2, "envoy": 0.9, "migrate": 0.4}},
        {"lat_range": [-50.0, -10.0], "lon_range": [0.0, 60.0], "permeability": {"daily": 0.7, "trade": 0.7, "envoy": 1.0, "migrate": 0.7}}]}


def _manual_traits(cfg):
    cfg["traits_manual"] = {"trait": [
        {"id": "m1", "slot": "calendar", "origin": "north_east"},
        {"id": "m2", "slot": "calendar", "origin": 7, "resistance": 0.5, "origin_time": 12.0},
        {"id": "m3", "slot": "coinage", "origin": "south", "d_half_days": 11, "mode": "envoy"},
        {"id": "m4", "slot": "white_hemp", "origin": 100, "d_half_days": 7.5, "resistance": 0.3}]}


@pytest.mark.parametrize("name,sets,mutate", [
    ("fast", ["s08.fast=true"], None),
    ("band_political", [], _band_and_political),
    ("origin_reformer", ["s07.centers.origin=north_west", "s09.polity.reformer_capital=5", "s09.polity.active_fronts=3"], None),
    ("manual_traits", [], _manual_traits),
])
def test_variants_identical(tmp_path, name, sets, mutate):
    _, a = _run(tmp_path, "python", sets, "py", mutate)
    _, b = _run(tmp_path, "cpp", sets, "cc", mutate)
    pa, pb = _products(a), _products(b)
    assert pa.keys() == pb.keys()
    diff = [k for k in pa if pa[k] != pb[k]]
    assert not diff, (name, diff)
    if name == "manual_traits":
        assert not (a / "s08_diffusion" / "reflect.json").exists() and not (b / "s08_diffusion" / "reflect.json").exists()


# ---------------------------------------------------------------- 第三层：人口与邦都从 C++ 的 ⑨ 给
def test_island_polity_inputs_from_cpp(two_backends):
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    _, out_py, _, _ = two_backends
    cfg = _cfg("python")
    ctx = Context(cfg, 7, out_py)
    isl.island_config(ctx, ["engine.backend=cpp"])
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    assert IE._polity_part(ctx) is not None
    with np.load(out_py / "s09_polity" / "polity.npz") as z:
        caps, n = z["capital"], z["pop"].size
    nodes = sorted(set(caps.tolist()) | set(range(0, n, 53)) | {n - 1})
    for node in nodes:
        inp = isl._node_inputs(ctx, node)
        a = IE.inputs(ctx, node, inp, full=True)
        b = IE.inputs_polity_py(ctx, node)
        assert {k: a[k] for k in b} == b, node


def test_island_generate_in_memory_world(two_backends, tmp_path):
    """同一进程里 cpp 后端跑过 ①–⑨：第三层直接用内存里的 C++ 对象（含 ⑨ 的人口与邦都），整群产物与 python 后端逐字节相同。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    _, _, cfg_c, out_c = two_backends
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    cfg = _cfg("cpp")
    run(cfg, 7, out_c.parent, upto=9, force_from=1, log=lambda *a: None)
    run_dir = str(out_c.resolve())
    keys = {i: _meta(out_c, st)["stage_key"] for i, st in STAGES[:9]}
    assert all((run_dir, i, keys[i]) in E._PARTS for i in range(1, 10))
    ctx = Context(cfg, 7, out_c)
    with np.load(out_c / "s09_polity" / "polity.npz") as z:
        caps = z["capital"]
    area = ctx.load_npz(3, "islands")["area_km2"]
    node = int(caps[np.argsort(area[caps])[len(caps) // 2]])      # 中等大小的邦都（有城、主家候选）

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
    assert outs["python"].keys() == outs["cpp"].keys()
    assert [k for k in outs["python"] if outs["python"][k] != outs["cpp"][k]] == []
    J = json.loads(outs["cpp"]["island.json"])
    assert J["settlements"]["city"] is not None          # 邦都：人口与邦都确实从 ⑨ 来了（有都与城）
