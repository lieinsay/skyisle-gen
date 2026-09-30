"""行星计划 P6d：行星层 ⑤–⑨ 的 C++ 核心。

扩展没编（python core/build.py）时整个文件跳过。公共件：Dijkstra（多源、有界、带 inf 的边、平局）、Brandes 抽样介数（线程数无关）、
弱连通分量与前端的 graph.py（探针的路径还用它）逐位比，choice(n, size, replace=False, p) 与 numpy 逐位比；
小世界上跑 ①–⑨：缓存 key、附庸、planet_run(upto=9) 一次跑完与逐步相同、从产物读回的 ⑤⑥⑦ 对象接着算 ⑧⑨ 与内存里的同值；
改几组开关（fast 参考树、lat_band → band 障碍 + 政治性障碍、起源指定中心 + 指定变法之国、手工特征表）照样跑通且只动 ⑤ 以后；
第三层的人口与邦都、⑥ 的邻边由 C++ 从 ⑨ / ⑥ 的对象给（与 npz 对得上），中转站各功能走到，内存里的 ①–⑨ 与 npz 读回的生成岛群逐字节相同。
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
from skyisle_gen import graph as G                        # noqa: E402
from skyisle_gen.config import load_config                # noqa: E402
from skyisle_gen.pipeline import Context, run, _stage_key_chain, STAGES   # noqa: E402

# 同 test_pipeline：1600 岛的小世界骨架窗内岛少，τ_c 放宽
SMALL = ["s03.islands.n_islands=1600", "s07.regions.n_regions=12", "s07.regions.max_regions=20",
         "s06.routes.betweenness_sources=48", "s08.k_sub=1", "s07.centers.tau_c=0.3"]
CIV = ("s05_barriers", "s06_routes", "s07_centers", "s08_diffusion", "s09_polity")


# ---------------------------------------------------------------- 公共件（C++ 的 ⑥ 与前端 graph.py 同值）
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
    """与 numpy 的 Generator.choice(n, size, replace=False, p) 逐位相同（阶段随机流 = SeedSequence([seed, 阶段])）。"""
    for seed in range(8):
        r = np.random.default_rng(seed)
        n = int(r.integers(20, 3000))
        p = r.random(n) ** 3
        p[r.random(n) < 0.2] = 0.0
        p /= p.sum()
        size = int(r.integers(1, min(int((p > 0).sum()), 300)))
        a = core.rng_choice_noreplace_p(seed, 6, p, size)
        b = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 6]))).choice(n, size=size, replace=False, p=p)
        assert np.array_equal(a, b), seed


# ---------------------------------------------------------------- 小世界：①–⑨
def _cfg(sets=(), run_id="cpp", mutate=None):
    cfg = load_config(sets=SMALL + list(sets) + [f"run.id={run_id}"])
    if mutate:
        mutate(cfg)
    return cfg


def _run(root, sets=(), run_id="cpp", mutate=None, upto=9):
    cfg = _cfg(sets, run_id, mutate)
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
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p6d")
    E.clear_cache()
    return _run(root)                  # 各步的 C++ 对象留在内存里（engine._PARTS）


def test_products_present(world):
    _, out = world
    a = _products(out)
    assert len(a) > 30
    for f in ("s09_polity/polity.npz/state", "s09_polity/polities.json", "s09_polity/history.md", "s08_diffusion/fields.npz/share"):
        assert f in a, f


def test_vassals_found(world):
    """附庸判定按邦号查都城间距离（原先拿都城节点号查，附庸几乎为零）；overlord 与 vassals 互相对得上。"""
    _, out = world
    pol = json.loads((out / "s09_polity" / "polities.json").read_text(encoding="utf-8"))["polities"]
    vassal = {x["id"]: x["overlord"] for x in pol if x.get("overlord", -1) >= 0}
    assert len(vassal) >= 3, len(vassal)
    by_id = {x["id"]: x for x in pol}
    for s, t in vassal.items():
        assert s in by_id[t]["vassals"]


def test_cache_keys(world):
    """①–⑨ 的 key 固定混入 "+cpp"、与 [engine] 段无关（没有 [engine] 段、旧的 backend = "cpp" 都一样）。"""
    cfg, out = world
    for _idx, st in STAGES[:9]:
        assert _meta(out, st)["engine"] == "cpp", st
    keys = _stage_key_chain(cfg, 7)
    cfg0 = copy.deepcopy(cfg)
    cfg0.pop("engine", None)
    assert _stage_key_chain(cfg0, 7) == keys
    assert [_meta(out, st)["stage_key"] for _i, st in STAGES[:9]] == keys[:9]


def test_planet_run_upto9_equals_stagewise(world):
    cfg, out = world
    pc = core.make_config(E.planet_config(cfg))
    objs = core.planet_run(pc, 7, upto=9, threads=3)
    assert len(objs) == 9
    B, R, Ce, D, Pol = objs[4:]
    with np.load(out / "s05_barriers" / "perm.npz") as z:
        assert np.array_equal(core.barriers_arrays(B)["perm"], z["perm"])
    with np.load(out / "s06_routes" / "routes.npz") as z:
        ra = core.routes_arrays(R)
        assert np.array_equal(ra["cost_m"], z["cost_m"]) and np.array_equal(ra["flow"].astype(np.float32), z["flow"])
    with np.load(out / "s07_centers" / "prehist.npz") as z:
        assert np.array_equal(core.centers_arrays(Ce)["dist_pre"], z["dist_pre"])
    with np.load(out / "s08_diffusion" / "fields.npz") as z:
        assert np.array_equal(core.diffusion_arrays(D)["share"].astype(np.float32), z["share"])
    with np.load(out / "s09_polity" / "polity.npz") as z:
        pa = core.polity_arrays(Pol)
        for k in ("state", "polity", "fief", "realm", "capital"):
            assert np.array_equal(pa[k], z[k]), k
        assert np.array_equal(pa["pop"].astype(np.float32), z["pop"])
    # ⑧ 可跳过：⑨ 照样算（游戏只要人口与邦时）
    objs2 = core.planet_run(pc, 7, upto=9, skip_diffusion=True)
    assert np.array_equal(core.polity_arrays(objs2[8])["state"], pa["state"])
    assert core.planet_run(pc, 7)[3] is not None and len(core.planet_run(pc, 7)) == 4   # 默认仍是 P6c 的 ①–④


def test_loaded_parts_equal_memory(world):
    """⑤⑥⑦ 从 npz / json 读回的 C++ 对象接着算 ⑧⑨，与一路在内存里算的同值（缓存命中后换进程的路径）。"""
    cfg, out = world
    ctx = Context(cfg, 7, out)
    pc = core.make_config(E.planet_config(cfg))
    E.clear_cache()
    P, I, C = E.part(ctx, 1), E.part(ctx, 3), E.part(ctx, 4)
    B, R, Ce = E.part(ctx, 5), E.part(ctx, 6), E.part(ctx, 7)
    d = core.diffusion_arrays(core.planet_stage8(pc, 7, I, B, R, Ce))
    with np.load(out / "s08_diffusion" / "fields.npz") as z:
        for k in ("reach", "share", "strength"):
            assert np.array_equal(d[k].astype(np.float32), z[k]), k
    pa = core.polity_arrays(core.planet_stage9(pc, I, C, B, R, Ce))
    with np.load(out / "s09_polity" / "polity.npz") as z:
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


@pytest.mark.parametrize("name,sets,mutate,first", [
    ("fast", ["s08.fast=true"], None, "s08_diffusion"),
    ("band_political", [], _band_and_political, "s05_barriers"),
    ("origin_reformer", ["s07.centers.origin=north_west", "s09.polity.reformer_capital=5", "s09.polity.active_fronts=3"], None, "s07_centers"),
    ("manual_traits", [], _manual_traits, "s08_diffusion"),
])
def test_variants(world, tmp_path, name, sets, mutate, first):
    """改开关：照样跑通到 ⑨，改动从该改的那一步起才出现（前面的步产物不变）。"""
    _, base = world
    _, out = _run(tmp_path, sets, "var", mutate)
    pa, pb = _products(base), _products(out)
    assert pa.keys() - pb.keys() <= {"s08_diffusion/reflect.json"} and pb.keys() <= pa.keys()
    diff = sorted({k.split("/")[0] for k in pa if pa[k] != pb.get(k)})
    assert diff and diff[0] == first, (name, diff)
    if name == "manual_traits":
        assert not (out / "s08_diffusion" / "reflect.json").exists()
        tr = json.loads((out / "s08_diffusion" / "traits.resolved.json").read_text(encoding="utf-8"))
        ids = [t["id"] for t in (tr["traits"] if isinstance(tr, dict) else tr)]
        assert ids == ["m1", "m2", "m3", "m4"], ids


# ---------------------------------------------------------------- 第三层：人口与邦都、⑥ 的邻边从 C++ 的对象给
def test_island_polity_inputs_from_cpp(world):
    """⑨ 的人口与邦都（_core.node_polity）与 polity.npz 对得上：人口 = pop[节点]；是邦都才有 capital，邦人口 = 本邦各节点之和。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    cfg, out = world
    ctx = Context(copy.deepcopy(cfg), 7, out)
    isl.island_config(ctx)
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    with np.load(out / "s09_polity" / "polity.npz") as z:
        caps, pop, state = z["capital"], z["pop"].astype(np.float64), z["state"]
    reformer = (json.loads((out / "s09_polity" / "polities.json").read_text(encoding="utf-8")).get("reformer") or {})
    n = pop.size
    nodes = sorted(set(caps.tolist()) | set(range(0, n, 53)) | {n - 1})
    seen_cap = 0
    for node in nodes:
        d = IE.inputs(ctx, node, isl._node_inputs(ctx, node), full=True)
        assert d["pop"] == pytest.approx(float(pop[node]), rel=1e-6), node
        st = int(state[node])
        is_cap = st >= 0 and int(caps[st]) == node
        assert (d["capital"] is not None) == is_cap, node
        if is_cap:
            seen_cap += 1
            assert d["capital"]["state"] == st
            assert d["capital"]["state_pop"] == pytest.approx(float(pop[state == st].sum()), rel=1e-5)
            assert isinstance(d["capital"]["reformer"], bool)
    assert seen_cap >= 3
    assert any(IE.inputs(ctx, int(c), isl._node_inputs(ctx, int(c)), full=True)["capital"]["reformer"] for c in caps) == bool(reformer)


def test_island_routes_inputs_from_cpp(world):
    """P7：第三层的 ⑥ 邻边（中转站读它，_core.node_routes）：邻群 = ③ 的候选边上的邻居，枢纽的标记与 hubs.json 对得上，成本、流量为正。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    cfg, out = world
    ctx = Context(copy.deepcopy(cfg), 7, out)
    isl.island_config(ctx)
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    hubs = {h["node"] for h in json.loads((out / "s06_routes" / "hubs.json").read_text(encoding="utf-8"))["hubs"]}
    with np.load(out / "s03_islands" / "cand_edges.npz") as ce:
        src, dst = ce["src"], ce["dst"]
    n = ctx.load_npz(3, "islands")["lat"].size
    nodes = sorted(set(sorted(hubs)[:10]) | set(range(0, n, 97)) | {n - 1})
    seen_hub = False
    for node in nodes:
        r = IE.inputs(ctx, node, isl._node_inputs(ctx, node), full=True)["routes"]
        nb = sorted(int(b) if a == node else int(a) for a, b in zip(src.tolist(), dst.tolist()) if node in (a, b))
        assert sorted(e["node"] for e in r["edges"]) == nb, node
        assert r["hub"] == (node in hubs) and all(e["hub"] == (e["node"] in hubs) for e in r["edges"])
        seen_hub |= r["hub"]
        assert r["edges"] and all(e["flow_in"] >= 0 and e["cost_out"] > 0 and -np.pi <= e["bearing"] <= np.pi for e in r["edges"])
    assert seen_hub


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


def test_island_generate_relays(world, tmp_path):
    """P7：⑥ 的枢纽群（群间的换船）与邻边的口子（关卡、过夜、候风、避风）：放低门槛让各种功能都出来，
    中转站的户从非农户里出（户数各项加起来 = 总户数）。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    cfg, out = world
    ctx = Context(copy.deepcopy(cfg), 7, out)
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    hubs = [h["node"] for h in json.loads((out / "s06_routes" / "hubs.json").read_text(encoding="utf-8"))["hubs"]]
    area = ctx.load_npz(3, "islands")["area_km2"]
    ok = [h for h in hubs if 300.0 < float(area[h]) < 4000.0]
    node = ok[len(ok) // 2] if ok else hubs[0]
    keys = ("relay_flow_min", "relay_storm_min", "relay_overnight_days", "relay_headwind_ratio")
    orig = {k: isl.island_config(ctx)["market"][k] for k in keys}
    sets = ["island.market.relay_flow_min=1.0", "island.market.relay_storm_min=0.0", "island.market.relay_overnight_days=0.0",
            "island.market.relay_headwind_ratio=0.5"]
    P = _island_products(isl.generate(ctx, node, res_m=300.0, sets=sets, log=lambda *a: None, out_root=tmp_path / "a"))
    isl.island_config(ctx, [f"island.market.{k}={v}" for k, v in orig.items()])   # --set 会留在 ctx 上：改回默认
    S = json.loads(P["settlements.json"])
    funcs = {f for r in S["relays"] for f in r["functions"]}
    assert {"关卡", "过夜", "候风", "避风", "换船"} <= funcs, funcs
    assert S["households_in_relays"] == sum(r["households"] for r in S["relays"]) > 0
    assert (S["households_in_villages"] + S["households_in_hamlets"] + S["households_in_towns_market"] + S["households_in_specials"]
            + S["households_in_relays"] == S["households"])


def test_island_generate_in_memory_world(world, tmp_path):
    """同一进程里跑过 ①–⑨：第三层直接用内存里的 C++ 对象（含 ⑨ 的人口与邦都）；清掉进程内缓存后从 npz 读回，生成的岛群逐字节相同。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine as IE
    _, out = world
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    cfg = _cfg()
    run(cfg, 7, out.parent, upto=9, force_from=1, log=lambda *a: None)
    run_dir = str(out.resolve())
    keys = {i: _meta(out, st)["stage_key"] for i, st in STAGES[:9]}
    assert all((run_dir, i, keys[i]) in E._PARTS for i in range(1, 10))
    ctx = Context(cfg, 7, out)
    with np.load(out / "s09_polity" / "polity.npz") as z:
        caps = z["capital"]
    area = ctx.load_npz(3, "islands")["area_km2"]
    node = int(caps[np.argsort(area[caps])[len(caps) // 2]])      # 中等大小的邦都（有城、主家候选）
    a = _island_products(isl.generate(ctx, node, res_m=300.0, log=lambda *a: None, out_root=tmp_path / "mem"))
    E.clear_cache()
    IE._PLANET_OBJ.clear()
    b = _island_products(isl.generate(Context(_cfg(), 7, out), node, res_m=300.0, log=lambda *a: None, out_root=tmp_path / "disk"))
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
    J = json.loads(a["island.json"])
    assert J["settlements"]["city"] is not None          # 邦都：人口与邦都确实从 ⑨ 来了（有都与城）
