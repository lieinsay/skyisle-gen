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
    v = max(S["villages"], key=lambda x: x["households"] if not x.get("seat") else -1)
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
