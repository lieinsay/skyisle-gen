"""聚落营建器（docs/PLAN-TOWN.md）：静态隔离、合成地形、岛群窗口细化。依赖 C++ 扩展（没编就跳过）。"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PKG = Path(__file__).resolve().parent.parent / "skyisle_gen"


# ---------------- 独立工具：管线与岛群生成器都不得读它 ----------------
def test_nothing_imports_town():
    files = sorted((PKG / "stages").glob("*.py")) + sorted((PKG / "island").glob("*.py"))
    files += [PKG / n for n in ("check.py", "ninegrid.py", "polity.py", "culture.py", "pipeline.py", "engine.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\.*\s*(skyisle_gen\.)?town\b", text, re.M), f"{f.name} import 了聚落营建器"


_core = pytest.importorskip("skyisle_gen._core")

from skyisle_gen.town import TERRAINS, town_config, window_half_m  # noqa: E402
from skyisle_gen.town.site import site_stats, site_synth  # noqa: E402

ARRAYS = ("height", "water_level", "water", "sky", "edge", "farmland", "flood", "landcover", "island", "water_dist_m")
WET = {"曲流平原", "河谷", "朝阳坡", "湖岸", "峡湾岸", "黄土沟", "两河交汇", "岛缘崖台"}


def _check_site(sd):
    land = ~sd["sky"]
    assert np.isfinite(sd["height"][land]).all() and np.isnan(sd["height"][sd["sky"]]).all()
    wet = sd["water"] > 0
    assert np.isfinite(sd["water_level"][wet]).all() and np.isnan(sd["water_level"][~wet]).all()
    assert (sd["height"][wet] <= sd["water_level"][wet] + 1e-3).all(), "水下的地面（河床 / 湖底）不能高过水面"
    assert not (sd["farmland"] & (wet | sd["sky"] | sd["edge"])).any(), "田不能在水里、虚空里、崖缘退让带上"
    assert not (sd["flood"] & (wet | sd["sky"])).any()
    assert not (sd["edge"] & sd["sky"]).any()
    assert (sd["island"][sd["sky"]] == -1).all()


@pytest.mark.parametrize("terrain", list(TERRAINS))
def test_synth_terrains(terrain):
    cfg = town_config()
    sd, meta = site_synth(terrain, "village", 40, cfg, seed=3, half_m=340.0)   # 朝阳坡的溪在 315 m 外
    assert sd["H"] == sd["W"] == 680 and sd["height"].shape == (680, 680)
    _check_site(sd)
    wet = (sd["water"] > 0).any()
    assert wet == (terrain in WET), f"{terrain}：有没有水不对"
    assert sd["sky"].any() == (terrain == "岛缘崖台")
    if terrain == "岛缘崖台":
        assert sd["edge"].any()
    st = site_stats(sd)
    assert 0.0 <= st["farmland_share"] <= 1.0 and st["size_m"] == [680.0, 680.0]
    sd2, _ = site_synth(terrain, "village", 40, cfg, seed=3, half_m=340.0)
    for k in ARRAYS:
        assert np.array_equal(sd[k], sd2[k], equal_nan=True), f"{terrain}：{k} 重跑不一致"
    sd3, _ = site_synth(terrain, "village", 40, cfg, seed=4, half_m=340.0)
    assert not np.array_equal(sd["height"], sd3["height"], equal_nan=True), "换种子地面应当不同"


@pytest.mark.parametrize("terrain", ["河谷", "黄土沟", "湖岸"])
def test_synth_window_only_crops(terrain):
    """窗口只决定裁多大：大窗口的中间一块与小窗口逐格相同（地貌按绝对米数摆、噪声按坐标取）。"""
    cfg = town_config()
    a, _ = site_synth(terrain, "village", 40, cfg, seed=5, half_m=200.0)
    b, _ = site_synth(terrain, "village", 40, cfg, seed=5, half_m=320.0)
    off = (b["H"] - a["H"]) // 2
    m = 60   # 离窗口边 60 m 以内，最近的河 / 沟可能在小窗口外，距离场会不同
    ia = (slice(m, a["H"] - m), slice(m, a["W"] - m))
    ib = (slice(off + m, off + a["H"] - m), slice(off + m, off + a["W"] - m))
    for k in ("height", "water", "farmland", "flood"):
        assert np.array_equal(a[k][ia], b[k][ib], equal_nan=True), f"{terrain}：{k} 随窗口变了"


def test_southern_hemisphere_flips_sun_side():
    """朝阳坡在北半球朝南（地面向北升高）、在南半球朝北。"""
    cfg = town_config()
    n, _ = site_synth("朝阳坡", "village", 40, cfg, seed=2, half_m=200.0, lat_deg=35.0)
    s, _ = site_synth("朝阳坡", "village", 40, cfg, seed=2, half_m=200.0, lat_deg=-35.0)
    c = n["W"] // 2
    assert n["height"][50, c] > n["height"][-150, c]      # 行 50 在北边：北半球北边高
    assert s["height"][50, c] < s["height"][-150, c]


def test_window_half_grows_with_households():
    cfg = town_config()
    assert window_half_m("village", 30, cfg) == pytest.approx(cfg["window"]["village_min_half_m"])
    assert window_half_m("village", 300, cfg) > window_half_m("village", 60, cfg)
    assert window_half_m("city", 15000, cfg) > 2000.0


# ---------------- 岛群窗口：小世界里生成一个岛群，再细化一个村 ----------------
@pytest.fixture(scope="module")
def group(tmp_path_factory):
    from skyisle_gen import island as isl
    from skyisle_gen.config import load_config
    from skyisle_gen.pipeline import Context, run
    root = tmp_path_factory.mktemp("town")
    cfg = load_config(sets=["s03.islands.n_islands=1600", "run.id=town"])
    out = run(cfg, 7, root, upto=4)
    ctx = Context(cfg, 7, out)
    I = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    ok = np.where(cli["has_river"] & (I["main_area_km2"] < 1500) & (I["main_area_km2"] > 300))[0]
    node = int(ok[0]) if ok.size else 0
    isl.generate(ctx, node, res_m=300.0, log=lambda *a: None)
    import json
    S = json.loads((out / "islands" / str(node) / "settlements.json").read_text(encoding="utf-8"))
    # 户数最多的普通村：不是邑治、也不是镇所在的村（那是「镇」，营建器不营建）
    town_v = {t["village"] for t in S.get("towns", []) if t.get("village") is not None}
    v = max(S["villages"], key=lambda x: x["households"] if not x.get("seat") and x["id"] not in town_v else -1)
    return ctx, node, v


def test_site_window(group):
    from skyisle_gen.town.site import site_from_group
    ctx, node, v = group
    cfg = town_config()
    sd, meta = site_from_group(ctx, node, v["name"], None, cfg, half_m=300.0)
    assert meta["scale"] == "village" and meta["households"] == v["households"]
    assert sd["res_m"] == cfg["site"]["res_village_m"] and sd["H"] == 600
    _check_site(sd)
    land = ~sd["sky"]
    # 窗口中心就是村的格心：地面高程与上游的村址高程同量级（100 m 格的插值 + 细节噪声）
    c = sd["H"] // 2
    assert abs(float(sd["height"][c, c]) - float(v["elev_m"])) < 25.0
    assert land.mean() > 0.3
    # 重跑逐格相同；大窗口的中间一块与小窗口相同
    sd2, _ = site_from_group(ctx, node, v["name"], None, cfg, half_m=300.0)
    for k in ARRAYS:
        assert np.array_equal(sd[k], sd2[k], equal_nan=True), f"{k} 重跑不一致"
    big, _ = site_from_group(ctx, node, v["name"], None, cfg, half_m=450.0)
    off, m = (big["H"] - sd["H"]) // 2, 80
    ia = (slice(m, sd["H"] - m), slice(m, sd["W"] - m))
    ib = (slice(off + m, off + sd["H"] - m), slice(off + m, off + sd["W"] - m))
    for k in ("height", "sky", "farmland", "landcover", "water"):
        assert np.array_equal(sd[k][ia], big[k][ib], equal_nan=True), f"{k} 随窗口变了"


def test_town_anchors_for_step4(group):
    """P7：镇 / 城的窗口带着营建器第四步「集镇与城」要读的锚点——镇挨着的大泊场（在 landings 最前面）与老村核心朝它长的街；村没有这两样。"""
    import json
    from skyisle_gen.town.site import site_from_group
    ctx, node, v = group
    cfg = town_config()
    S = json.loads((ctx.out_dir / "islands" / str(node) / "settlements.json").read_text(encoding="utf-8"))
    _, meta_v = site_from_group(ctx, node, v["name"], None, cfg, half_m=250.0)
    assert "harbor" not in meta_v["anchors"] and "street" not in meta_v["anchors"]
    towns = [t for t in S["towns"] if t["harbor"] is not None]
    if not towns:
        pytest.skip("这个小岛群里没有挨着大泊场的镇")
    t = towns[0]
    tv = next(x for x in S["villages"] if x["id"] == t["village"])
    _, meta = site_from_group(ctx, node, tv["name"], None, cfg, half_m=400.0)
    A = meta["anchors"]
    assert meta["kind"] == "town" and A["harbor"]["id"] == t["harbor"] and A["landings"][0] == A["harbor"]["xy"]
    assert A["street"]["to"] == A["harbor"]["xy"] and abs(A["street"]["length_m"] - t["street"]["length_km"] * 1000.0) < 1e-6
    assert abs(A["street"]["from"][0]) < 200.0 and abs(A["street"]["from"][1]) < 200.0      # 街从老村核心（窗口中心那格）出发


def test_site_command_writes_products(group, tmp_path):
    from skyisle_gen.town.output import out_dir, write_plan_json, write_site
    from skyisle_gen.town.render import write_plan_png
    from skyisle_gen.town.site import site_from_group
    ctx, node, v = group
    cfg = town_config()
    sd, meta = site_from_group(ctx, node, v["name"], None, cfg, half_m=250.0)
    out = out_dir(ctx.out_dir, meta, None)
    st = site_stats(sd)
    write_site(out, sd)
    write_plan_json(out, sd, meta, st)
    write_plan_png(out, sd, meta, cfg, st)
    for f in ("site.npz", "plan.json", "plan.png"):
        assert (out / f).stat().st_size > 0
    z = np.load(out / "site.npz")
    assert z["height"].shape == (sd["H"], sd["W"]) and z["frame"][0] == sd["res_m"]


def test_heightmap_source(tmp_path):
    """外部高程图：16 位灰度 × scale_m = 高程；水面图非零处是水、地面压到水面以下。"""
    from PIL import Image
    from skyisle_gen.town.site import site_heightmap
    yy, xx = np.mgrid[0:200, 0:240]
    h = (1000 + 8 * xx + 3 * yy).astype(np.uint16)             # 0.1 m 一级：100–164 m 的斜面
    Image.fromarray(h).save(tmp_path / "h.png")
    wm = np.zeros((200, 240), np.uint8)
    wm[90:110, :] = 255
    Image.fromarray(wm).save(tmp_path / "w.png")
    cfg = town_config()
    sd, meta = site_heightmap(tmp_path / "h.png", 2.0, "village", 40, cfg, water=tmp_path / "w.png", scale_m=0.1)
    assert meta["source"] == "heightmap" and (sd["H"], sd["W"]) == (200, 240) and sd["res_m"] == 2.0
    _check_site(sd)
    assert (sd["water"][95:105, 10:230] > 0).all() and not (sd["water"][:80] > 0).any()
    dry = ~(sd["water"] > 0) & ~sd["flood"]
    assert np.allclose(sd["height"][dry], h[dry] * 0.1, atol=1e-3)


# ---------------- 营建（第二步：华北集村）：风格、方案、硬校验、朝向、产物 ----------------
import json  # noqa: E402
import math  # noqa: E402

from skyisle_gen.town.output import plan_json  # noqa: E402
from skyisle_gen.town.plan import hard_failures, make_plan, plan_seed  # noqa: E402
from skyisle_gen.town.style import list_styles, load_style, style_flat, style_hash  # noqa: E402


def test_style_load_and_override(tmp_path):
    assert "华北集村" in {s["name"] for s in list_styles()}
    st = load_style("华北集村")
    assert st == load_style("huabei"), "中文名与 id 是同一个风格"
    assert st["ground"]["max_slope_deg"] == 12 and st["street"]["plot_gap_m"] is not None   # 风格盖 base、base 补齐
    assert st["functions"]["temple"]["enabled"] and st["functions"]["temple"]["name"] == "关帝庙"
    assert st["functions"]["well"]["mode"] == "well", "通用目录里的字段合进来"
    assert load_style("华北集村", ["style.ground.max_slope_deg=8"])["ground"]["max_slope_deg"] == 8
    flat = style_flat(st)   # 给 C++ 的 {num, vec, str}：模板与启用的功能展成数组
    assert flat["num"]["style.compound.template.n"] == len(st["compound"]["templates"])
    assert flat["num"]["style.func.n"] == sum(bool(v.get("enabled")) for v in st["functions"].values())
    with pytest.raises(ValueError):
        load_style("没有这个风格")
    bad = tmp_path / "bad.toml"
    bad.write_text('[meta]\nid = "bad"\nname = "坏"\n[functions.nope]\nenabled = true\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_style(str(bad))


def _plan(terrain, op, lat=None, seed=1, hh=60):
    cfg = town_config()
    sd, meta = site_synth(terrain, "village", hh, cfg, seed=seed, lat_deg=lat)
    st = load_style("华北集村")
    return sd, meta, st, make_plan(sd, meta, st, operator=op)


def _local(c, p):
    o = c["plot"]
    f = math.radians(o["facing_deg"])
    d = np.asarray(p, float) - np.asarray(o["c"], float)
    return d @ np.array([math.cos(f), -math.sin(f)]), d @ np.array([math.sin(f), math.cos(f)])   # 右、前


@pytest.mark.parametrize("terrain,op", [("平原", "fishbone"), ("平原", "organic"), ("朝阳坡", "fishbone"), ("河谷", "organic"),
                                        ("山顶", "organic"), ("两河交汇", "fishbone"), ("岛缘崖台", "organic")])
def test_plan_hard_checks(terrain, op):
    sd, meta, st, P = _plan(terrain, op)
    assert P["op"] == op
    assert not hard_failures(P), [c["id"] + "：" + c["msg"] for c in hard_failures(P)]
    assert (P["households"]["compound"] >= 0).all(), "每户都住下"
    assert any(c["func"] == "temple" for c in P["compounds"]) and any(f["kind"] == "well" for f in P["features"])
    assert sum(b["func"] == "landing" or b["name"] == "货棚" for b in P["buildings"]) >= 1 or any(f["kind"] == "landing" for f in P["features"])


def test_plan_deterministic_and_seeded():
    sd, meta, st, P = _plan("河谷", "organic")
    before = {k: sd[k].copy() for k in ARRAYS}
    P2 = make_plan(sd, meta, st, operator="organic")
    for k in ARRAYS:
        assert np.array_equal(before[k], sd[k], equal_nan=True), f"营建改了地面的 {k}"
    h = style_hash(st)

    def dump(Q):   # 用时每次不同，不算
        return json.dumps({k: v for k, v in plan_json(Q, st, h).items() if not k.startswith("timing")}, sort_keys=True)

    a = dump(P)
    assert a == dump(P2) and np.array_equal(P["occ"], P2["occ"]), "同一地面同一风格重跑逐字节相同"
    _, _, _, P3 = _plan("河谷", "organic", seed=2)
    assert dump(P3) != a, "换种子方案不同"
    # 方案的种子跟风格走、地面的种子不跟：换风格地面不变
    assert plan_seed(meta, "huabei") != plan_seed(meta, "other")


@pytest.mark.parametrize("lat,sun", [(35.0, 180.0), (-35.0, 0.0)])
@pytest.mark.parametrize("op", ["fishbone", "organic"])
def test_courtyards_face_the_equator(lat, sun, op):
    """北半球院子朝南、南半球朝北；正房在院子后头，门按「前沿开左、后沿开右、侧边开前头」。"""
    _, _, _, P = _plan("平原", op, lat=lat)
    houses = [c for c in P["compounds"] if c["kind"] == "house"]
    dev = np.array([abs((c["plot"]["facing_deg"] - sun + 180.0) % 360.0 - 180.0) for c in houses])
    assert (dev <= 15.0).mean() >= 0.9, f"朝阳的只有 {(dev <= 15.0).mean():.0%}"
    for c in houses:
        o = c["plot"]
        for b in c["buildings"]:
            if P["buildings"][b]["role"] == "main":
                assert _local(c, P["buildings"][b]["c"])[1] < 0, "正房在院子后半"
        x, y = _local(c, c["gate"])
        side = c["access_side"]
        if side == 0:
            assert abs(y - o["d"] / 2) < 0.8 and x < 0, "前沿开门在左手（北半球即东南角）"
        elif side == 1:
            assert abs(y + o["d"] / 2) < 0.8 and x > 0
        else:
            assert abs(abs(x) - o["w"] / 2) < 0.8 and y > 0


def test_plan_cli_products(tmp_path):
    from skyisle_gen.cli import main
    assert main(["town", "synth", "--terrain", "平原", "--households", "30", "--style", "华北集村", "--out", str(tmp_path)]) == 0
    out = next((tmp_path / "town" / "synth").iterdir())
    for f in ("site.npz", "plan.json", "plan.png", "plan-detail.png", "plan.svg", "style.resolved.toml"):
        assert (out / f).stat().st_size > 0, f
    J = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert J["style"]["id"] == "huabei" and J["compounds"] and J["checks"]
    assert all(c["ok"] for c in J["checks"] if c["hard"])
    assert load_style(str(out / "style.resolved.toml")) is not None, "解析后的风格能原样再读回来"


# ---------------------------------------------------------------- 第三步：其余村级算子与十一个风格、画廊
from skyisle_gen.town.style import OPERATORS  # noqa: E402

ALL_STYLES = [x["id"] for x in list_styles()]
_GROUNDS: dict = {}


def _ground(terrain, hh=40, hf=1.0):
    key = (terrain, hh, hf)
    if key not in _GROUNDS:
        _GROUNDS[key] = site_synth(terrain, "village", hh, town_config(), seed=1, half_factor=hf)
    return _GROUNDS[key]


def _style_plan(style, terrain, op=None, hh=40):
    """与 town synth --terrain 同一种 --style 同一种 一样：窗口按风格的放大系数。"""
    st = load_style(style)
    sd, meta = _ground(terrain, hh, float(st.get("site", {}).get("window_factor", 1.0)))
    return st, make_plan(sd, meta, st, op)


def test_all_styles_load():
    assert set(ALL_STYLES) == {"huabei", "jiangnan", "huizhou", "linpan", "lingnan", "yaodong", "hakka", "nordic", "central_eu", "england",
                               "med_hill", "japan"}
    for sid in ALL_STYLES:
        st = load_style(sid)
        ops = {k: v for k, v in st["village"]["operators"].items() if v > 0}
        assert ops and set(ops) <= set(OPERATORS), sid
        assert style_flat(st)["str"]["style.meta.id"] == sid


@pytest.mark.parametrize("style", ALL_STYLES)
def test_every_style_on_the_same_valley(style):
    """画廊的验收（PLAN-TOWN 第三步）：同一块河谷地 × 十二个风格，按风格的权重挑算子——硬项全过、户都住下。"""
    st, P = _style_plan(style, "河谷")
    assert not hard_failures(P), [c["id"] + "：" + c["msg"] for c in hard_failures(P)]
    assert P["op"] in st["village"]["operators"] and P["op"] in P["ops_fit"]
    assert (P["households"]["compound"] >= 0).all()


# 每个算子在合它的地上（风格 × 算子 × 地形）；后面是这个算子的特征指标
OP_CASES = [("central_eu", "hufen", "平原"), ("central_eu", "street_village", "河谷"), ("central_eu", "green", "平原"),
            ("central_eu", "organic", "朝阳坡"), ("england", "street_village", "平原"), ("england", "green", "河谷"),
            ("england", "organic", "平原"), ("hakka", "enclosure", "朝阳坡"), ("hakka", "enclosure", "平原"), ("huizhou", "organic", "河谷"),
            ("japan", "dispersed", "平原"), ("japan", "street_village", "河谷"), ("japan", "organic", "平原"),
            ("jiangnan", "waterfront", "曲流平原"), ("jiangnan", "waterfront", "两河交汇"), ("jiangnan", "street_village", "平原"),
            ("lingnan", "comb", "平原"), ("linpan", "dispersed", "曲流平原"), ("med_hill", "organic", "山顶"),
            ("nordic", "dispersed", "朝阳坡"), ("nordic", "street_village", "河谷"), ("yaodong", "contour", "黄土沟"),
            ("yaodong", "organic", "平原")]
SIGNATURE = {"dispersed": ("site_clark_evans", "sites", "site_nn_median_m"), "waterfront": ("water_front_share",),
             "comb": ("lane_spacing_cv", "lane_spacing_m"), "enclosure": ("hh_per_enclosure", "enclosures"), "contour": ("terraces", "terrace_step_m"),
             "hufen": ("frontage_cv",), "street_village": ("face_street_share",)}


@pytest.mark.parametrize("style,op,terrain", OP_CASES)
def test_operator_on_fitting_ground(style, op, terrain):
    st, P = _style_plan(style, terrain, op)
    assert P["op"] == op
    assert not hard_failures(P), [c["id"] + "：" + c["msg"] for c in hard_failures(P)]
    for k in SIGNATURE.get(op, ()):
        assert k in P["metrics"], f"{op} 没出指标 {k}"
    kinds = {f["kind"] for f in P["features"]}
    M = P["metrics"]
    if op == "dispersed":
        assert M["sites"] >= 5 and M["site_nn_median_m"] > 40.0, "散居：一处处隔开"
        if style == "japan":
            assert M["site_clark_evans"] >= 1.0, "砺波散居村 R ≥ 1（匀散）"
    if op == "waterfront":
        assert M["water_front_share"] >= 0.3 and "steps" in kinds, "前街后河：房临水、有河埠头"
    if op == "enclosure":
        assert any(b["role"] == "ring" for b in P["buildings"]) and M["hh_per_enclosure"] >= 4, "土楼 / 围龙屋：一楼住多户"
    if op == "contour":
        assert M["terraces"] >= 2, "沿沟台：至少两层台"
    if op == "green":
        assert "green" in kinds, "围绿：有村绿"
    if op == "comb":
        assert M["lane_spacing_cv"] <= 0.3, "梳式：巷距匀"
    if style == "england" and op == "street_village":
        assert "garden" in kinds and "furlong" in kinds, "toft 后是 croft，村外是敞田的条田"
    if style == "japan" and op == "organic":
        assert "moat" in kinds, "環濠集落有濠"


@pytest.mark.parametrize("style,terrain", [("huizhou", "山顶"), ("med_hill", "岛缘崖台"), ("huabei", "河谷")])
def test_more_households_spread_out(style, terrain):
    """户少就摊开、户多就往外长（用户定的规矩：合理比凑密度要紧）：每户都住下，村子的外包随户数长，覆盖率由风格的排法定、不随户数挤。
    山城在岛缘崖台 30 户那一例曾是教堂的地块擦进主街（公共建筑挑地没按几何查路）。"""
    cov, ext = [], []
    for hh in (30, 120):
        _, P = _style_plan(style, terrain, hh=hh)
        assert not hard_failures(P), [c["id"] + "：" + c["msg"] for c in hard_failures(P)]
        cov.append(P["metrics"]["coverage"])
        ext.append(P["metrics"]["extent_long_m"])
    assert ext[1] > 1.4 * ext[0], f"120 户的村没往外长：{ext}"
    assert abs(cov[1] - cov[0]) < 0.06, f"覆盖率随户数变了：{cov}"


def test_operator_fit_rules():
    """挑算子只在这块地能用的里挑：滨水要河、等高线要坡、地坑院 / 散居村 / 環濠集落 / 绿地村要平地。"""
    _, P = _style_plan("yaodong", "平原")
    assert "contour" not in P["ops_fit"] and P["op"] == "organic", "平地上没有沿沟台"
    _, P = _style_plan("yaodong", "峡湾岸")
    assert "organic" not in P["ops_fit"] and P["op"] == "contour", "陡岸上没有地坑院"
    _, P = _style_plan("jiangnan", "平原")
    assert "waterfront" not in P["ops_fit"], "平原没有河：不做前街后河"
    _, P = _style_plan("japan", "朝阳坡")
    assert P["ops_fit"] == ["street_village"] and P["op"] == "street_village"


def _check(P, cid):
    return next((c for c in P["checks"] if c["id"] == cid), None)


def test_terrain_first_relaxes_ground_limits():
    """优先次序（用户定）：地形 > 合理 > 风格。水乡的坡度上限 8°，朝阳坡上处处更陡——照风格一户都住不下，
    风格让路：地面上限放宽一档（坡 14°、挖填 1.5 m、台地 2.4 m）再排，全村住下，TP-terrain 说放宽了多少。"""
    _, P = _style_plan("jiangnan", "朝阳坡")
    assert not hard_failures(P), [c["id"] + "：" + c["msg"] for c in hard_failures(P)]
    assert (P["households"]["compound"] >= 0).all()
    c = _check(P, "TP-terrain")
    assert c and not c["hard"] and c["ok"] and "地形优先" in c["msg"] and "坡 8°" in c["msg"] and "只住下 0 / 40 户" in c["msg"]
    M = P["metrics"]
    assert M["terrain_relax_level"] == 1 and M["max_slope_deg_used"] == pytest.approx(14.0) and M["max_cut_m_used"] == pytest.approx(1.5)
    assert M["buildable_share_near"] > 0.5   # 按放宽后的上限量


def test_style_that_fits_the_ground_is_not_relaxed():
    """照风格住得下的一次排完：没有 TP-terrain、没有放宽的指标，产物与不放宽时一样。"""
    _, P = _style_plan("huabei", "河谷", hh=60)
    assert _check(P, "TP-terrain") is None and "terrain_relax_level" not in P["metrics"]
    assert _check(P, "TP-hh")["ok"]


def test_terrain_first_smaller_yards_when_plots_do_not_fit():
    """绿地村的宅地（toft & croft）进深 60–100 m，峡湾岸海边那条平地摆不下：风格让路，余下的户换风格里最小的院子（小农舍）沿路排开。"""
    _, P = _style_plan("england", "峡湾岸", op="green")
    assert not hard_failures(P), [c["id"] + "：" + c["msg"] for c in hard_failures(P)]
    assert (P["households"]["compound"] >= 0).all()
    names = {c["template_name"] for c in P["compounds"] if c["kind"] == "house"}
    assert "toft & croft" in names and "小农舍（cottage）" in names


def test_style_miss_caused_by_terrain_is_excused():
    """TP-style 里地形逼出来的偏离单列「地形所致、不算」：华北团村上山顶，只能顺山脊成带（村心摆一块 λ 1.6 的，
    一半落在坡上）；朝阳坡上鱼骨街村拉成 λ 4 是排法自己拉长的（摆得下团村），照实算没过。"""
    _, P = _style_plan("huabei", "山顶", op="organic", hh=60)
    c = _check(P, "TP-style")
    assert "地形所致、不算：λ" in c["msg"] and "lambda =" not in c["msg"]
    assert P["metrics"]["lambda"] > 1.6 and P["metrics"]["lambda_room_share"] < 0.85
    _, P = _style_plan("huabei", "朝阳坡", op="fishbone", hh=60)
    c = _check(P, "TP-style")
    assert not c["ok"] and "lambda =" in c["msg"] and "地形所致" not in c["msg"]
    assert P["metrics"]["lambda_room_share"] >= 0.85


def test_gallery_command(tmp_path):
    from skyisle_gen.cli import main
    assert main(["town", "gallery", "--terrain", "河谷", "--styles", "华北集村,jiangnan", "--households", "30", "--out", str(tmp_path)]) == 0
    out = tmp_path / "town" / "gallery" / "河谷-村30户-s1"
    assert (out / "gallery.png").stat().st_size > 0
    J = json.loads((out / "gallery.json").read_text(encoding="utf-8"))
    assert [t["style_id"] for t in J["tiles"]] == ["huabei", "jiangnan"] and J["request"]["operators"] == "default"
    # 每一格与 town synth 同参数的方案逐一相同
    for t in J["tiles"]:
        st, P = _style_plan(t["style_id"], "河谷", hh=30)
        assert t["operator"] == P["op"] and t["buildings"] == len(P["buildings"]) and t["compounds"] == len(P["compounds"])
        assert not t["hard_fail"]
    # 每个能用的算子各一格；两种地形一行一种
    assert main(["town", "gallery", "--terrain", "黄土沟,平原", "--styles", "yaodong", "--operators", "each", "--households", "30",
                 "--out", str(tmp_path)]) == 0
    J = json.loads((tmp_path / "town" / "gallery" / "黄土沟+平原-村30户-s1-各算子" / "gallery.json").read_text(encoding="utf-8"))
    by = {}
    for t in J["tiles"]:
        by.setdefault(t["terrain"], []).append(t["operator"])
        assert t["operator"] in t["operators_fit"]
    assert sorted(by["黄土沟"]) == sorted(J["tiles"][0]["operators_fit"]) and by["平原"] == ["organic"]


# ---------------------------------------------------------------- 第五步：营建调试台 /town.html
def test_town_console_api(tmp_path):
    import base64
    import threading
    import urllib.parse
    import urllib.request
    from functools import partial
    from http.server import ThreadingHTTPServer

    from skyisle_gen.web.server import App, Handler
    from skyisle_gen.web.town_api import town_dumps
    app = App(tmp_path)
    api = app.town
    S = api.styles()
    assert len(S["styles"]) == 12 and "河谷" in S["terrains"]
    assert {o["id"] for o in next(x for x in S["styles"] if x["id"] == "japan")["operators"]} == {"dispersed", "street_village", "organic"}
    req = {"source": "synth", "terrain": "平原", "households": 20, "style": "huabei", "operator": "organic"}
    J = api.plan(req)
    assert J["op_name"] == "团块生长" and J["plan"]["plan"]["operator"] == "organic" and J["saved"] is None
    assert all(c["ok"] for c in J["plan"]["checks"] if c["hard"])
    g = J["ground"]
    assert len(base64.b64decode(g["height_u16"])) == 2 * g["rows"] * g["cols"] and len(base64.b64decode(g["occ_u8"])) == g["rows"] * g["cols"]
    assert base64.b64decode(g["png"])[:4] == b"\x89PNG"
    json.loads(town_dumps(J))
    # 同一块地换一种窗口系数相同的风格：地面走缓存；方案与命令行的一样
    J2 = api.plan({**req, "style": "lingnan", "operator": ""})
    assert len(api.grounds) == 1 and J2["plan"]["style"]["id"] == "lingnan"
    st = load_style("lingnan")
    sd, meta = _ground("平原", 20)
    assert len(J2["plan"]["buildings"]) == len(make_plan(sd, meta, st)["buildings"])
    # 参数覆盖：只收 style.* / town.*；覆盖进了风格哈希
    with pytest.raises(ValueError):
        api.plan({**req, "sets": ["bogus=1"]})
    J3 = api.plan({**req, "sets": ["style.street.lane_width_m=[2.5,3]"], "save": True})
    assert J3["plan"]["style"]["hash"] != J["plan"]["style"]["hash"]
    out = Path(J3["saved"])
    for fn in ("plan.json", "site.npz", "plan.png", "plan.svg", "style.resolved.toml"):
        assert (out / fn).stat().st_size > 0, fn
    assert "lane_width_m = [2.5, 3]" in api.style_text("huabei", ["style.street.lane_width_m=[2.5,3]"])
    # 走一遍 HTTP：页面、清单、营建、参数
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, app=app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert b"/api/town/plan" in op.open(base + "/town.html").read()
        assert len(json.loads(op.open(base + "/api/town/styles").read())["styles"]) == 12
        body = json.dumps({**req, "style": "英格兰集村", "operator": "street_village"}, ensure_ascii=False).encode("utf-8")
        Jh = json.loads(op.open(urllib.request.Request(base + "/api/town/plan", data=body, method="POST")).read())
        assert Jh["plan"]["style"]["id"] == "england" and Jh["op_name"].startswith("规划型")
        assert 'id = "huabei"' in op.open(base + "/api/town/style?name=" + urllib.parse.quote("华北集村")).read().decode("utf-8")
    finally:
        srv.shutdown()


def test_town_console_sites(group):
    """岛群里的聚落清单与「营建此聚落」：村能营建、镇与专业聚落标出来（第四 / 六步）。"""
    from skyisle_gen.web.town_api import TownApi
    ctx, node, v = group
    api = TownApi(ctx.out_dir.parent, lambda run: ctx)
    S = api.sites("town", node)
    names = {r["name"]: r for r in S["sites"]}
    assert names[v["name"]]["plannable"] and names[v["name"]]["households"] == v["households"]
    assert all(not r["plannable"] for r in S["sites"] if r["kind"] == "镇")
    J = api.plan({"source": "site", "run": "town", "node": node, "site": v["name"], "style": "huabei", "half": 300})
    assert J["meta"]["site"] == v["name"] and J["plan"]["households"]


def test_island_console_links_to_town():
    static = PKG / "web" / "static"
    i = (static / "island.html").read_text(encoding="utf-8")
    assert "/town.html?run=" in i and "营建此聚落" in i and "dblclick" in i
    t = (static / "town.html").read_text(encoding="utf-8")
    for api in ("/api/town/styles", "/api/town/sites", "/api/town/plan", "/api/town/style"):
        assert api in t
