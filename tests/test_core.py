"""单元测试：公式层与守则（不跑完整管线）。

Python 参考后端删掉以后（2026-09-30），⑧ 的 adopt 不动点、④ 的水汽推进与带界位移、③ 的陆地模型、① 的历法换算、季节强度与纬度密度剖面
没有单独的 C++ 绑定，这里不再单测：它们由三 seed 的 check（P1–P8、C1–C5、SK-*）与 core/tests/selftest.cpp 看着。
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyisle_gen.graph import CSR, dijkstra, weak_components
from skyisle_gen.sphere import latlon_to_xyz, grid_axes, grid_interp
from skyisle_gen.config import load_config

try:                                   # 算法只在 C++ 核心里：没编（python core/build.py）时调 C++ 的用例跳过，静态断言与配置校验照跑
    import skyisle_gen._core as core
except ImportError:
    core = None
needs_core = pytest.mark.skipif(core is None, reason="C++ 扩展没编：python core/build.py")
PKG = Path(__file__).resolve().parent.parent / "skyisle_gen"


# ---------------- 图 ----------------
def _line_graph(costs):
    n = len(costs) + 1
    src = np.array(list(range(n - 1)) + list(range(1, n)))
    dst = np.array(list(range(1, n)) + list(range(n - 1)))
    w = np.array(list(costs) + list(costs), dtype=float)
    return n, src, dst, w


def test_dijkstra_line():
    """graph.py（前端：探针的路径）的 Dijkstra。"""
    n, src, dst, w = _line_graph([1.0, 2.0, 3.0])
    csr = CSR(n, src, dst)
    dist, pn, pe = dijkstra(csr, w, [0])
    assert dist.tolist() == [0.0, 1.0, 3.0, 6.0]
    assert pn.tolist() == [-1, 0, 1, 2]


@needs_core
def test_dijkstra_line_cpp():
    n, src, dst, w = _line_graph([1.0, 2.0, 3.0])
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    dist, pn, pe = core.graph_dijkstra(n, src, dst, w, [0])
    assert dist.tolist() == [0.0, 1.0, 3.0, 6.0]
    assert pn.tolist() == [-1, 0, 1, 2] and pe.tolist()[1:] == [0, 1, 2]
    dist, _, _ = core.graph_dijkstra(n, src, dst, w, [0], 2.0)          # 有界：超过 2 天的不走
    assert dist[1] == 1.0 and not np.isfinite(dist[2]) and not np.isfinite(dist[3])


def test_dijkstra_inf_edges_removed():
    n, src, dst, w = _line_graph([1.0, np.inf, 1.0])
    csr = CSR(n, src, dst)
    dist, _, _ = dijkstra(csr, w, [0])
    assert np.isfinite(dist[1]) and not np.isfinite(dist[2])


@needs_core
def test_dijkstra_inf_edges_removed_cpp():
    n, src, dst, w = _line_graph([1.0, np.inf, 1.0])
    dist, _, _ = core.graph_dijkstra(n, src.astype(np.int64), dst.astype(np.int64), w, [0])
    assert np.isfinite(dist[1]) and not np.isfinite(dist[2])


def test_weak_components():
    comp = weak_components(4, np.array([0, 2]), np.array([1, 3]))
    assert comp[0] == comp[1] and comp[2] == comp[3] and comp[0] != comp[2]


@needs_core
def test_weak_components_cpp():
    comp = core.graph_weak_components(5, np.array([0, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64))
    assert comp[0] == comp[1] and comp[2] == comp[3] and comp[0] != comp[2] and comp[4] not in (comp[0], comp[2])


@needs_core
def test_betweenness_thread_independent_cpp():
    """⑥ 的抽样介数按源并行：结果与线程数无关。"""
    rng = np.random.default_rng(9)
    n, m = 300, 1500
    src, dst = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = src != dst
    src, dst = src[keep].astype(np.int64), dst[keep].astype(np.int64)
    w = rng.uniform(0.1, 3.0, src.size)
    sources = sorted(rng.choice(n, 40, replace=False).tolist())
    ref = core.graph_betweenness(n, src, dst, w, sources, 1.0, 1)
    for threads in (4, 7):
        assert np.array_equal(core.graph_betweenness(n, src, dst, w, sources, 1.0, threads), ref), threads
    assert ref.shape == (src.size,) and (ref >= 0).all() and ref.sum() > 0


# ---------------- Φ 穿越归一化 ----------------
def test_phi_crossing_hop_invariance():
    """跨越同一障碍，2 跳与 4 跳的通过率乘积应相同（评审修订清单 P0-2）。"""
    P = 0.1
    for hops in (2, 4, 8):
        phi = np.linspace(0.0, 1.0, hops + 1)
        f = np.abs(np.diff(phi))
        total = float(np.prod(P ** f))
        assert total == pytest.approx(P, rel=1e-9)


# ---------------- 球面 ----------------
@needs_core
def test_antimeridian_knn():
    """经度 ±179 的点必须互为近邻（东西向世界不允许 ±180 假障碍）；③ 的 kNN 在 C++ 里。"""
    lat = np.array([10.0, 10.0, 10.0, 40.0])
    lon = np.array([179.5, -179.5, 100.0, 179.5])
    xyz = np.ascontiguousarray(latlon_to_xyz(lat, lon))
    idx, ang = core.planet_knn(xyz, 1)
    assert idx[0, 0] == 1 and idx[1, 0] == 0
    assert ang[0, 0] == pytest.approx(math.radians(1.0) * math.cos(math.radians(10.0)), rel=1e-3)


def test_grid_interp_periodic():
    lats, lons = grid_axes(1.0)
    field = np.tile(np.sin(np.radians(lons))[None, :], (lats.size, 1))
    v = grid_interp(field, lats, lons, np.array([0.0]), np.array([179.9]))
    assert abs(v[0] - math.sin(math.radians(179.9))) < 0.01


@needs_core
def test_grid_interp_periodic_cpp():
    lats, lons = grid_axes(1.0)
    field = np.ascontiguousarray(np.tile(np.sin(np.radians(lons))[None, :], (lats.size, 1)))
    for lon in (179.9, -179.9, 0.3):
        v = core.grid_interp(field, float(lats[0]), float(lats[1] - lats[0]), float(lons[0]), float(lons[1] - lons[0]), 0.0, lon)
        assert abs(v - math.sin(math.radians(lon))) < 0.01, lon
        assert v == pytest.approx(float(grid_interp(field, lats, lons, np.array([0.0]), np.array([lon]))[0]), abs=1e-12), lon


# ---------------- 配置 ----------------
def test_band_id_local_reduces_to_global_when_flat():
    import numpy as np
    from skyisle_gen.stages.s02_wind import band_id_of, band_id_of_lat
    from skyisle_gen.localwind import edge_lats, EDGE_KEYS
    bands = {"eq_storm_top_deg": 8.0, "trades_top_deg": 28.0, "calm_top_deg": 36.0, "westerlies_top_deg": 62.0}
    lons = np.arange(-179.5, 180.0, 1.0)
    e = edge_lats(bands)
    bl = {"lons": lons, "edges": np.stack([np.full(lons.size, e[k]) for k in EDGE_KEYS]), "keys": np.array(EDGE_KEYS)}
    rng = np.random.default_rng(1)
    lat = rng.uniform(-89, 89, 2000); lon = rng.uniform(-180, 180, 2000)
    assert np.array_equal(band_id_of(lat, lon, bands, bl), band_id_of_lat(lat, bands))


def test_default_config_valid():
    cfg = load_config()
    assert cfg["s05"]["barriers"]["A"]["permeability"]["envoy"] == 0.03
    assert cfg["s08"]["eps0"] > 0


def test_resolved_toml_roundtrip():
    """config.resolved.toml 必须读得回来：[island.resources] 的 ore_gain 用中文键（TOML 裸键只许 ASCII）。"""
    import tomllib
    from skyisle_gen.config import dump_toml
    cfg = load_config()
    assert tomllib.loads(dump_toml(cfg)) == cfg


def test_lambda_order_enforced():
    with pytest.raises(ValueError):
        load_config(sets=["s08.half_distance_days.daily=[100.0,200.0]"])


def test_resistance_clamped():
    with pytest.raises(ValueError):
        load_config(sets=["s08.resistance_range.high=[0.7,1.0]"])


def test_python_backend_rejected():
    """Python 参考后端 2026-09-30 删了：engine.backend 只能是 cpp（或不写）。"""
    load_config(sets=["engine.backend=cpp"])
    with pytest.raises(ValueError):
        load_config(sets=["engine.backend=python"])


# ---------------- 铁律（静态） ----------------
def test_no_height_in_social_modules():
    """原则乙：s07/s08/s09/ninegrid/polity 不得读取 height_m；C++ 版的 ⑦⑧⑨（core/src/planet/stage7–9.cpp）不得读 Islands 的 height。"""
    import re
    from skyisle_gen.check import CPP_SOCIAL_FILES
    root = Path(__file__).resolve().parent.parent
    pkg = root / "skyisle_gen"
    for f in ["stages/s07_centers.py", "stages/s08_diffusion.py", "stages/s09_polity.py", "ninegrid.py", "polity.py"]:
        text = (pkg / f).read_text(encoding="utf-8")
        assert not re.search(r"\[[\"']height_m[\"']\]", text), f"{f} 读取了 height_m"
    for f in CPP_SOCIAL_FILES:
        text = (root / f).read_text(encoding="utf-8")
        assert not re.search(r"\.height\b", text), f"{f} 读取了 height"


def test_no_discrete_culture_assignment():
    """铁律五：不得从 share 的 argmax 派生地区/文化标签（文化是连续场）。"""
    import re
    pkg = Path(__file__).resolve().parent.parent / "skyisle_gen"
    for f in ["stages/s08_diffusion.py", "stages/s07_centers.py", "../core/src/planet/stage7.cpp", "../core/src/planet/stage8.cpp"]:
        text = (pkg / f).read_text(encoding="utf-8")
        assert "flood" not in text.lower()
        assert not re.search(r"culture_id|culture_label", text)


# ---------------- IS-iso：管线不得读岛群生成器（第三层不回灌）；聚落营建器是独立工具 ----------------
def test_stages_do_not_import_island():
    for f in sorted((PKG / "stages").glob("*.py")) + [PKG / "check.py", PKG / "ninegrid.py", PKG / "polity.py", PKG / "culture.py"]:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\.*\s*(skyisle_gen\.)?island\b", text, re.M), f"{f.name} import 了岛群生成器"
        assert "island_config" not in text and "build_terrain" not in text, f"{f.name} 用了岛群生成器"


def test_nothing_imports_town():
    files = sorted((PKG / "stages").glob("*.py")) + sorted((PKG / "island").glob("*.py"))
    files += [PKG / n for n in ("check.py", "ninegrid.py", "polity.py", "culture.py", "pipeline.py", "engine.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\.*\s*(skyisle_gen\.)?town\b", text, re.M), f"{f.name} import 了聚落营建器"


def test_island_config_section_present():
    cfg = load_config()
    assert "island" in cfg and "layout" in cfg["island"] and "terrain" in cfg["island"]


# ---------------- 岛群规模（陆地 = 势力范围 × 陆地占比，docs/02 §六/§七，R8/R9/R10） ----------------
def test_area_config_validation():
    """陆地与尺度口径参数的合法性校验。"""
    with pytest.raises(ValueError):
        load_config(sets=["s03.islands.land_frac_alpha=-1"])
    with pytest.raises(ValueError):
        load_config(sets=["s03.islands.land_frac_cap=1.5"])
    with pytest.raises(ValueError):
        load_config(sets=["s03.islands.land_frac_sigma=-0.1"])
    with pytest.raises(ValueError):
        load_config(sets=["shared.scale.total_land_km2=0"])
    with pytest.raises(ValueError):
        load_config(sets=["shared.scale.arable_frac_mean=1.5"])
    with pytest.raises(ValueError):
        load_config(sets=["s07.centers.area_exponent=-0.5"])
    with pytest.raises(ValueError):
        load_config(sets=["s01.planet.radius_km=0"])


def test_scale_quota_is_self_consistent():
    """三个口径自洽：陆地 × 可用地率 × 人口密度 = 2.5 亿（公元 1 年全球人口）。"""
    sc = load_config()["shared"]["scale"]
    pop = float(sc["total_land_km2"]) * float(sc["arable_frac_mean"]) * float(sc["people_per_arable_km2"])
    assert pop == pytest.approx(2.5e8, rel=0.02)


def test_planet_default_is_earth_sized():
    """默认行星 = 地球大小（半径 6371 km）。"""
    cfg = load_config()
    assert float(cfg["s01"]["planet"]["radius_km"]) == pytest.approx(6371.0)


def test_area_exponent_zero_recovers_old_suitability():
    """γ=0 时适宜度退回 docs/12 §五 原式（完全不看面积）。"""
    pn = np.array([0.8, 0.5, 0.2])
    stab = np.array([0.9, 0.7, 0.6])
    dn = np.array([1.0, 0.5, 0.25])
    an = np.array([0.1, 0.9, 0.4])
    base = pn * stab * dn
    assert np.allclose(base * an ** 0.0, base)
    assert not np.allclose(base * an ** 1.0, base)


# ---------------- 缓存 key 链 ----------------
def test_template_change_invalidates_only_ninegrid_stage():
    """改生产模板只该让 ⑩ 输出失效，①–⑨（含政治层）必须继续命中缓存。"""
    import copy
    from skyisle_gen.pipeline import _stage_key_chain
    cfg = load_config()
    base = _stage_key_chain(cfg, 42)
    c2 = copy.deepcopy(cfg)
    c2["production_templates"]["organization"]["dense"]["text"] = "改了"
    k2 = _stage_key_chain(c2, 42)
    assert base[:9] == k2[:9], "改生产模板不该让 ①–⑨ 失效"
    assert base[9] != k2[9], "改生产模板必须让 ⑩ 失效"


def test_polity_param_change_invalidates_only_polity_and_output():
    """改 [s09.polity] 只该让 ⑨⑩ 失效，①–⑧ 继续命中（政治层不回写地理与文化）。"""
    import copy
    from skyisle_gen.pipeline import _stage_key_chain
    cfg = load_config()
    base = _stage_key_chain(cfg, 42)
    c2 = copy.deepcopy(cfg)
    c2["s09"]["polity"]["reform_years_ago"] = 120.0
    k2 = _stage_key_chain(c2, 42)
    assert base[:8] == k2[:8] and base[8] != k2[8] and base[9] != k2[9]


def test_slots_change_invalidates_diffusion_stage():
    """改槽位表必须让 ⑧⑨ 失效（否则旧特征场会被当成命中）。"""
    import copy
    from skyisle_gen.pipeline import _stage_key_chain
    cfg = load_config()
    base = _stage_key_chain(cfg, 42)
    c2 = copy.deepcopy(cfg)
    c2["slots"]["slot"][0]["zh"] = "改了"
    k2 = _stage_key_chain(c2, 42)
    assert base[:7] == k2[:7], "改槽位表不该让 ①–⑦ 失效"
    assert base[7] != k2[7] and base[9] != k2[9], "改槽位表必须让 ⑧⑩ 失效"
