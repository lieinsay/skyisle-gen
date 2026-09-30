"""行星计划 P6b：第三层其余部分（资源、聚落与层级、四季、逐日天气、粗版降采样、整群 generate）的 C++ 核心。

扩展没编（python core/build.py）时整个文件跳过。与 numpy 逐位比的公共件：指数 ziggurat / 伽马（形状 < 1）/ 对数正态 / 泊松 / 不放回抽签、
np.quantile / np.interp / np.convolve（BLAS ddot）/ float32 成对求和；label_by_island / window_extrema / 田块的 k-means 按性质验；
文件后半是整群 generate 的各条代码路径（资源 P3、已垦 P5、水利 P6 / P6b / 分级、镇与航船 P7）：每条都走到、check 的相应项过、与线程数无关。
Python 参考后端 2026-09-30 删了（git tag python-reference-final），两个后端逐字节的对照随之删掉。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

core = pytest.importorskip("skyisle_gen._core", reason="C++ 扩展没编：python core/build.py")

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
    """label_by_island：8 连通块按岛拆开，每个标号只在一座岛上；window_extrema = 方窗里陆地格的最大 / 最小（暴力对照）。"""
    rng = np.random.default_rng(4)
    for t in range(20):
        H, W = 60, 80
        iid = np.full((H, W), -1, np.int16)
        iid[5:30, 5:40] = 0
        iid[20:55, 35:75] = 1
        iid[40:58, 2:30] = 2
        m = (rng.random((H, W)) < 0.6) & (iid >= 0)
        lab, n = core.label_by_island(m, iid, 8)
        assert (lab[~m] == 0).all() and set(np.unique(lab[m]).tolist()) == set(range(1, n + 1))
        lab8, _ = core.label_components(m, 8)                 # 标号 = (8 连通块, 岛号) 的一对：跨岛的块按岛拆开
        pairs = {(int(x), int(y)) for x, y in zip(lab8[m], iid[m])}
        assert n == len(pairs) >= core.label_components(m, 8)[1]
        for v in range(1, n + 1):
            cells = lab == v
            assert np.unique(iid[cells]).size == 1 and np.unique(lab8[cells]).size == 1
        h = rng.normal(0, 100, (H, W))
        land = iid >= 0
        hi, lo = core.window_extrema(h, 4, land)
        if t < 3:                                  # 暴力对照（慢，挑几轮）
            for i in range(H):
                for j in range(W):
                    w = land[max(0, i - 4):i + 5, max(0, j - 4):j + 5]
                    if w.any():
                        v = h[max(0, i - 4):i + 5, max(0, j - 4):j + 5][w]
                        assert hi[i, j] == v.max() and lo[i, j] == v.min(), (i, j)
        assert (hi[land] >= h[land]).all() and (lo[land] <= h[land]).all()


def test_kmeans_split():
    """田块切分（C++ 的 kmeans_split：不放回抽初值 + 12 轮 Lloyd）：标号在 [0, k)、不多于点数、同 key 重跑相同。"""
    rng = np.random.default_rng(5)
    for t in range(40):
        n = int(rng.integers(5, 3000))
        ii, jj = rng.integers(0, 200, n), rng.integers(0, 200, n)
        k = int(rng.integers(1, 12))
        key = f"island:{t}:settle:fields"
        args = (7, key, ii.astype(np.int32).tolist(), jj.astype(np.int32).tolist(), k)
        a = np.asarray(core.kmeans_split(*args))
        assert a.shape == (n,) and a.min() >= 0 and a.max() < k and np.unique(a).size <= min(k, n), t
        assert np.array_equal(a, core.kmeans_split(*args)), t
        if k > 1 and n >= 50:
            assert np.unique(a).size > 1, t


# ---------------------------------------------------------------- 小世界：整群 generate 的各条代码路径
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


# B0 起河宽不夸张（× 1），漫滩 = 10 × 真实河宽不到一格；B1 起谷是真切出来的 V 形——C2 的谷底宽之前，小世界里没有平的谷底：
# 湿地（周围最陡的坡 < 2.5°）、圩田、大堰（干渠从渠首第一格就被谷坡挡住）都出不来。水利那几条测的是算法的每条路径，
# 照旧用夸张的河宽（漫滩跟着宽）把谷底铺出来（DESIGN-NOTES 四点四十六）
OLD_FLOOD = ["island.hydro.width_scale=8.0", "island.hydro.depth_scale=3.0"]


def _gen(ctx, node, root, threads=4, sets=(), **kw):
    from skyisle_gen import island as isl
    return isl.generate(ctx, node, res_m=300.0, sets=[f"engine.threads={threads}"] + list(sets), log=lambda *a: None,
                        return_state=True, out_root=root, **kw)


def _hard_fails(ctx, node, g, out):
    from skyisle_gen.island.check import evaluate
    return [i["id"] for i in evaluate(g, out, node=node, c=ctx.cfg["island"]) if not i["pass"] and i["hard"]]   # 不给 ctx：IS-daily（软项、慢）不跑


def test_generate_products(small_ctx, tmp_path):
    """整群 generate（地形 → 资源 → 四季 → 天气 → 聚落）：产物齐全、栅格同形、check 的硬项全过；有河、无河的群都走到。"""
    for node in _nodes(small_ctx):
        b, gb = _gen(small_ctx, node, tmp_path / "a")
        assert gb["json"]["meta"]["engine"] == "cpp"
        shape = gb["island_id"].shape
        for k in ("terrain_zone", "patch_id", "resource", "settle_raster", "settle_fields", "landcover"):
            assert gb[k].shape == shape, (node, k)
        assert gb["res_field"].shape == (7,) + shape
        for f in ("island.json", "terrain.npz", "climate.json", "resources.json", "settlements.json", "weather_y0.csv", "preview.png"):
            assert (b / f).exists(), (node, f)
        n_days = len(gb["weather"]["days"])
        assert all(gb["daily"][k].shape == (n_days,) for k in ("day", "season", "temp_c", "wind_u", "wind_v", "precip_mm"))
        assert not _hard_fails(small_ctx, node, gb, b), node


def test_generate_p3_resources(small_ctx, tmp_path):
    """P3（没有火山）的新资源（骨架空洞、只在新岛的温泉、热泉硫磺、按剥蚀深浅的石料岩性、岩盐 / 盐泉 / 盐井 / 盐井村、贝壳化石）：
    把新岛门槛、盐丘与化石的密度调高让每样都出得来，RES-* 与其余硬项过。"""
    from skyisle_gen import island as isl
    extra = ["island.terrain.age_young=0.6", "island.resources.density_per_100km2.salt=3.0", "island.resources.fossil_per_km2=1.0"]
    node = _nodes(small_ctx, 1)[0]
    out, ga = _gen(small_ctx, node, tmp_path / "a", sets=extra)
    kinds = {d["kind"] for d in ga["resources"]["deposits"]} | {o["kind"] for o in ga["resources"]["occurrences"]}
    assert {"salt", "saltspring", "fossil", "hotspring"} <= kinds, kinds
    assert not _hard_fails(small_ctx, node, ga, out)
    isl.island_config(small_ctx, ["island.terrain.age_young=0.3", "island.resources.density_per_100km2.salt=0.05",
                                  "island.resources.fossil_per_km2=0.1"])   # --set 会留在 ctx 上：改回默认（之前漏了，后面的测试都跑在 age_young 0.6 上）


def test_generate_p5_farmland(small_ctx, tmp_path):
    """P5（宜垦 / 已垦 / 撂荒、定居门槛、没人住 ≠ 没人用、荒地归谁）：小世界里没人住的岛多半已有工棚，特殊用途分两遍走——
    一遍放宽放牧（夏牧、烽火台、废村、工棚），一遍关掉放牧与烽火台、庙与墓岛必有；每条代码路径走到、check 的硬项过。"""
    from skyisle_gen import island as isl
    common = ["island.settle.ruin_min_hh=4", "island.settle.graze_min_km2=0.05", "island.settle.graze_reach_km=1e3", "island.settle.shieling_min_km2=0.5"]
    passes = [common + ["island.settle.graze_max=1"], common + ["island.settle.graze_max=0", "island.market.beacon_max=0", "island.settle.shrine_p=1.0", "island.settle.tomb_p=1.0",
                                "island.settle.tomb_max_km2=1e6"]]
    base = ["island.settle.ruin_min_hh=8", "island.settle.graze_min_km2=0.5", "island.settle.graze_reach_km=10.0", "island.settle.shieling_min_km2=3.0",
            "island.settle.graze_max=4", "island.market.beacon_max=2", "island.settle.shrine_p=0.5", "island.settle.tomb_p=0.35", "island.settle.tomb_max_km2=5.0"]
    seen = set()
    for extra in passes:
        # 第 3、4 个有河的群（A5 起 river_min_mm 400，#1 也有河了，排在前面；要的是有没人住的小岛可放庙、墓岛的那两群）
        for node in _nodes(small_ctx, 4)[2:4]:
            out, gb = _gen(small_ctx, node, tmp_path / "a", sets=extra)
            assert not _hard_fails(small_ctx, node, gb, out), node
            S = gb["settle"]
            seen |= {u["kind"] for u in S["uses"]} | {x["occupancy"] for x in S["specials"]} | ({"废村"} if S["ruins"] else set())
            seen |= {"烽火台"} if any("烽火" in r["functions"] for r in S["relays"]) else set()      # P7：烽火台归中转站
            seen |= {"保底"} if S["farmland"]["floor_islands"] else set()
    isl.island_config(small_ctx, base)                      # --set 会留在 ctx 上：改回默认
    assert {"烽火台", "庙", "墓岛", "废村", "工棚", "保底"} <= seen and seen & {"放牧", "夏牧"}, seen


def test_generate_p6_waterworks(small_ctx, tmp_path):
    """P6（水利：谷口的渠、村塘 / 山塘 / 堰塘、圩田的纵浦横塘与圩塘、闸）：默认门槛一遍（季节性渠首）；再强开两遍圩田
    （压力门槛 0、片与圩的下限放小：整片湿地排干；一遍水田线照旧、一遍抬高出泽田），check 的硬项过，圩田在额度之内（已垦 = 额度），
    各种渠、塘、闸与水田 / 泽田都走到。B 起默认门槛下小世界的湿地排不成圩（V 形谷、没有平的谷底，见 OLD_FLOOD），圩田靠强开那两遍。"""
    from skyisle_gen import island as isl
    vw = ["island.works.village_works=true", "island.works.big_max=0"]          # 四点四十起村级默认不出、大堰默认修：这里测留着的村级算法
    # 强开的第二遍把水田线抬到 5000 mm：A3 起毫米换算是线性的，小世界的圩区都过 800 mm、全是水田，泽田要这样才走得到
    forced = ["island.works.polder_pressure_min=0.0", "island.works.polder_patch_min_km2=0.05", "island.works.polder_block_min_km2=0.05"]
    zetian = ["island.works.polder_paddy_mm=5000"]
    base = ["island.hydro.width_scale=1.0", "island.hydro.depth_scale=1.0", "island.works.polder_pressure_min=0.5", "island.works.polder_patch_min_km2=0.3", "island.works.polder_block_min_km2=0.1",
            "island.works.village_works=false", "island.works.big_max=2", "island.works.polder_paddy_mm=800"]
    seen = set()
    for extra in (OLD_FLOOD + vw, OLD_FLOOD + vw + forced, OLD_FLOOD + vw + forced + zetian):
        for node in _nodes(small_ctx, 2):
            out, gb = _gen(small_ctx, node, tmp_path / "a", sets=extra)
            assert not _hard_fails(small_ctx, node, gb, out), node
            S = gb["settle"]
            W = S["waterworks"]
            assert S["farmland"]["cultivated_km2"] == S["farmland"]["quota_km2"]
            seen |= {p["kind"] for p in W["ponds"]} | {x["kind"] for x in W["sluices"]} | {c["kind"] for c in W["canals"]}
            seen |= {"季节性渠首"} if W["summary"]["n_heads_seasonal"] else set()
            seen |= {"水田" if p["paddy"] else "泽田" for p in W["polders"]}
    isl.island_config(small_ctx, base)                      # --set 会留在 ctx 上：改回默认
    assert {"村塘", "山塘", "圩塘", "堰塘", "渠首闸", "圩闸", "排水闸", "干渠", "支渠", "纵浦", "横塘", "排水渠", "季节性渠首", "水田", "泽田"} <= seen, seen


def test_generate_p6b_manage(small_ctx, tmp_path):
    """P6b（有水利就有人维护、原始地貌与人工地貌分开记）：默认一遍（废村旁的废塘）；再强开一遍——
    一圩一组、走得到放到 3 km、废村放宽、渠首的汇水门槛与灌区下限放低（废村旁出废渠首、废渠、废渠首闸）、圩田强开（B 起默认门槛下
    小世界的湿地排不成圩：圩村、挂在村上的圩田靠这一遍），check 的硬项过；
    每处水利都有管它的村、在它走得到的范围内（check 的 SET-works），原始地表与人工改造对得上（SET-nature）。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.check import _nature_check, _works_manage
    vw = ["island.works.village_works=true", "island.works.big_max=0"]          # 四点四十起村级默认不出、大堰默认修：这里测留着的村级算法
    forced = ["island.works.polder_village_blocks=1", "island.works.manage_walk_km=3.0", "island.settle.ruin_min_hh=2",
              "island.works.works_src_min_km2=1.0", "island.works.works_stream_min_km2=2.0", "island.works.canal_min_cmd_km2=0.05",
              "island.works.polder_pressure_min=0.0", "island.works.polder_patch_min_km2=0.05", "island.works.polder_block_min_km2=0.05"]
    base = ["island.hydro.width_scale=1.0", "island.hydro.depth_scale=1.0", "island.works.polder_pressure_min=0.5", "island.works.polder_patch_min_km2=0.3",
            "island.works.polder_block_min_km2=0.1", "island.works.polder_village_blocks=3", "island.works.manage_walk_km=2.0", "island.settle.ruin_min_hh=8",
            "island.works.works_src_min_km2=10.0", "island.works.works_stream_min_km2=30.0", "island.works.canal_min_cmd_km2=0.5",
            "island.works.village_works=false", "island.works.big_max=2"]
    seen = set()
    for extra in (OLD_FLOOD + vw, OLD_FLOOD + vw + forced):
        for node in _nodes(small_ctx, 2)[:2]:
            out, gb = _gen(small_ctx, node, tmp_path / "a", sets=extra)
            assert not _hard_fails(small_ctx, node, gb, out), node
            for k in ("landcover_natural", "landuse"):
                assert gb[k].dtype == np.uint8 and gb[k].shape == gb["island_id"].shape, (node, k)
            S = gb["settle"]
            W = S["waterworks"]
            assert _works_manage(gb, S, gb["polder_id"]) == ([], [], [], [])
            assert _nature_check(gb, S)["pass"], _nature_check(gb, S)["value"]
            seen |= {"圩村"} if any(v.get("polder") for v in S["villages"]) else set()
            seen |= {"挂圩田"} if any(v.get("polder_fields") for v in S["villages"]) else set()
            seen |= {"废塘"} if any(x.get("abandoned") for x in W["ponds"]) else set()
            seen |= {"废渠首"} if any(x.get("abandoned") for x in W["heads"]) and any(x.get("abandoned") for x in W["sluices"]) else set()
            seen |= {"废渠"} if any(x.get("abandoned") for x in W["canals"]) else set()
            seen |= {f"改造{int(u)}" for u in np.unique(gb["landuse"])}
    isl.island_config(small_ctx, base)                      # --set 会留在 ctx 上：改回默认
    assert {"圩村", "挂圩田", "废塘", "废渠首", "废渠", "改造1", "改造3", "改造4", "改造5"} <= seen, seen


def test_generate_bigworks(small_ctx, tmp_path):
    """水利分级（四点四十，用户 09-30 定）：默认一遍（村级的渠、塘不在岛群层出）；再强开一遍邑级大堰（小世界的河小：汇水、灌区的门槛放低、
    不看额度占比），check 的硬项过；大堰在本岛常年河上、邑管，灌区里在种的地 = landuse 的渠灌田、都是已垦，
    每个用水的村一个分水口，SET-works / SET-nature 过，已垦 = 额度、人口不变。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.check import _nature_check, _works_check
    forced = ["island.works.big_src_min_km2=1.0", "island.works.big_min_cmd_km2=0.2", "island.works.big_min_quota_frac=0.0", "island.works.big_try=4"]
    base = ["island.hydro.width_scale=1.0", "island.hydro.depth_scale=1.0", "island.works.big_src_min_km2=30.0", "island.works.big_min_cmd_km2=20.0", "island.works.big_min_quota_frac=0.1", "island.works.big_try=10"]
    seen = set()
    for extra in (OLD_FLOOD, OLD_FLOOD + forced):
        for node in _nodes(small_ctx, 3):
            out, gb = _gen(small_ctx, node, tmp_path / "a", sets=extra)
            assert not _hard_fails(small_ctx, node, gb, out), node
            S = gb["settle"]
            W = S["waterworks"]
            assert W["summary"]["village_works"] is False and not W["heads"] and not W["ponds"]
            assert S["farmland"]["cultivated_km2"] == S["farmland"]["quota_km2"]
            assert _works_check(gb, S)["pass"], _works_check(gb, S)["value"]
            assert _nature_check(gb, S)["pass"], _nature_check(gb, S)["value"]
            V = {v["id"] for v in S["villages"]}
            ck = (gb["json"]["raster"]["res_m"] / 1000.0) ** 2
            served = 0.0
            for bw in W["big_works"]:
                i, j = bw["cell"]
                assert gb["river"][i, j] > 0 and gb["island_id"][i, j] == bw["island"] and bw["maintainer"] == "邑"
                assert bw["served_km2"] <= bw["planned_km2"] <= bw["water_km2"] + 1e-9
                assert [t["village"] for t in bw["turnouts"]] == bw["villages"] and set(bw["villages"]) <= V
                served += bw["served_km2"]
                seen.add("大堰")
                seen |= {"用水的村"} if bw["villages"] else set()
            lu3 = gb["landuse"] == 3
            assert abs(float(lu3.sum()) * ck - served) < 1e-2 and (gb["cultivated"][lu3] > 0).all()
            seen |= {c["kind"] for c in W["canals"] if "work" in c}
    isl.island_config(small_ctx, base)                      # --set 会留在 ctx 上：改回默认
    assert {"大堰", "用水的村", "干渠", "支渠"} <= seen, seen


def test_generate_p7_market(small_ctx, tmp_path):
    """P7（大泊场、镇、航船、邑治、群内的烽火台）：小世界到 ④、没有 ⑥ 的航线，只有群内的中转站。默认一遍，再一遍把走路赶集收到 2 km、
    镇距放到 4 km（多数村搭航船、线多、邑治挑得开），check 的硬项过；大泊场、镇挨着的泊场、航船线、搭航船的村、烽火台都走到。"""
    from skyisle_gen import island as isl
    forced = ["island.market.walk_km=2.0", "island.market.walk_max_km=2.0", "island.market.town_spacing_km=4.0", "island.market.line_max_stops=3",
              "island.market.harbor_min_km2=0.2", "island.market.tailwind_factor=0.5", "island.market.headwind_factor=2.5", "island.market.wind_ref_ms=0.5"]
    base = ["island.market.walk_km=6.0", "island.market.walk_max_km=10.0", "island.market.town_spacing_km=10.0", "island.market.line_max_stops=6",
            "island.market.harbor_min_km2=0.5", "island.market.tailwind_factor=0.7", "island.market.headwind_factor=1.5", "island.market.wind_ref_ms=5.0"]
    seen = set()
    for extra in ([], forced):
        for node in _nodes(small_ctx, 3):                      # 第三群才有烽火台（群边高处的没人住的岛）
            out, gb = _gen(small_ctx, node, tmp_path / "a", sets=extra)
            assert not _hard_fails(small_ctx, node, gb, out), node
            S = gb["settle"]
            seen |= {"大泊场"} if S["harbors"] else set()
            seen |= {"镇挨着大泊场"} if any(t["harbor"] is not None for t in S["towns"]) else set()
            seen |= {"航船"} if S["boat_lines"] else set()
            seen |= {v["market_mode"] for v in S["villages"]}
            seen |= {f for r in S["relays"] for f in r["functions"]}
            seen |= {"邑治不在主岛"} if any(t["seat"] and t["island"] != 0 for t in S["towns"]) else set()
    isl.island_config(small_ctx, base)                      # --set 会留在 ctx 上：改回默认
    assert {"大泊场", "镇挨着大泊场", "航船", "步行", "烽火"} <= seen, seen


def test_generate_cpp_thread_independent(small_ctx, tmp_path):
    node = _nodes(small_ctx, 1)[0]
    a, _ = _gen(small_ctx, node, tmp_path / "t1", threads=1)
    b, _ = _gen(small_ctx, node, tmp_path / "t4", threads=4)
    assert _products(a) == _products(b)


def test_steps_partial(small_ctx, tmp_path):
    """只算到资源 / 四季 / 天气（steps 2–4）：该有的有、后面的没有；已算的部分与整群 generate 的一样。"""
    node = _nodes(small_ctx, 1)[0]
    _, gf = _gen(small_ctx, node, tmp_path / "full")
    for steps in (2, 3, 4):
        _, g = _gen(small_ctx, node, tmp_path / f"s{steps}", steps=steps)
        assert ("resources" in g, "climate" in g, "weather" in g, "settle" in g) == (True, steps >= 3, steps >= 4, False), steps
        assert np.array_equal(g["height"], gf["height"], equal_nan=True) and np.array_equal(g["island_id"], gf["island_id"])
        if steps >= 3:
            assert g["climate"] == gf["climate"], steps
        if steps >= 4:
            assert g["weather"]["json"] == gf["weather"]["json"], steps


def test_lod_block_reduce(small_ctx):
    """粗版：两个分辨率各一份，默认带天气（C++ 的 weather_year），块降采样的数组同形、陆地占比按格加总与原生一样。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.lod import build_lod
    node = _nodes(small_ctx, 1)[0]
    c = isl.island_config(small_ctx)
    out = build_lod(small_ctx, node, c, [1600.0, 3200.0], native_res_m=400.0)
    assert sorted(out) == [1600.0, 3200.0]
    for res, (A, meta) in out.items():
        assert "weather_type" in A and "weather_meta" in A, res
        shape = A["land"].shape
        for k in ("height", "peak", "island", "landcover", "water"):
            assert A[k].shape == shape, (res, k)
        assert meta["raster"]["factor"] == round(res / 400.0)
    a, b = out[1600.0][0], out[3200.0][0]
    assert abs(a["land"].astype(np.float64).sum() * 1.6 ** 2 - b["land"].astype(np.float64).sum() * 3.2 ** 2) <= 0.01 * a["land"].astype(np.float64).sum() * 1.6 ** 2
    for k in a:
        if k.startswith("weather_"):
            assert np.array_equal(a[k], b[k]), k                  # 天气与分辨率无关


def test_is_daily_and_climate_only(small_ctx, tmp_path):
    """IS-daily 的多年逐日（C++ 的 weather_years）：多年平均回到气候值、同输入重跑相同；全量季型（climate_only）与整群 generate 的四季同值。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.engine import climate_only_cpp
    from skyisle_gen.island.weather import multi_year_stats
    node = _nodes(small_ctx, 1)[0]
    _, g = _gen(small_ctx, node, tmp_path / "d")
    c = isl.island_config(small_ctx)
    st = multi_year_stats(small_ctx, node, c, g, years=10)
    assert st == multi_year_stats(small_ctx, node, c, g, years=10)
    assert st["years"] == 10 and len(st["precip_mean_mm"]) == len(g["climate"]["seasons"])
    assert st["annual_rel_err"] < 0.1 or st["annual_z"] < 3.0, st
    isl_npz = small_ctx.load_npz(3, "islands")
    cli = small_ctx.load_npz(4, "climate_islands")
    planet = small_ctx.load_json(1, "planet")
    inp = {k: float(isl_npz[k][node]) for k in ("lat", "lon", "height_m")}
    for k in ("precip", "temp", "storm", "window", "temp_sea", "season_range", "season_range_sea", "temp_winter", "temp_summer"):
        inp[k] = float(cli[k][node])
    inp["precip_share"] = cli["precip_share"][:, node]          # A5：四季降水按 ④ 的份额（classify_all 同样传）
    inp["planet"] = planet
    inp["keel_clearance_m"] = float(small_ctx.cfg["s03"]["islands"].get("keel_clearance_m", 300.0))
    assert climate_only_cpp(small_ctx, inp, c) == g["climate"]
