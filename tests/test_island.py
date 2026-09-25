"""岛群生成器（第三层，docs/PLAN-ISLAND.md）：静态隔离断言、确定性、约束一致性。小世界只跑到 ④。"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyisle_gen.config import load_config
from skyisle_gen.pipeline import Context, run

PKG = Path(__file__).resolve().parent.parent / "skyisle_gen"
SMALL = ["s03.islands.n_islands=1600"]
STEPS = 5   # 开发中：已实现到第几步


# ---------------- IS-iso：管线不得读岛群生成器（第三层不回灌） ----------------
def test_stages_do_not_import_island():
    for f in sorted((PKG / "stages").glob("*.py")) + [PKG / "check.py", PKG / "ninegrid.py", PKG / "polity.py", PKG / "culture.py"]:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+\.*\s*(skyisle_gen\.)?island\b", text, re.M), f"{f.name} import 了岛群生成器"
        assert "island_config" not in text and "build_terrain" not in text, f"{f.name} 用了岛群生成器"


def test_island_config_section_present():
    cfg = load_config()
    assert "island" in cfg and "layout" in cfg["island"] and "terrain" in cfg["island"]


@pytest.fixture(scope="module")
def small_ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("isl")
    cfg = load_config(sets=SMALL + ["run.id=isl"])
    out = run(cfg, 7, root, upto=4)
    return Context(cfg, 7, out)


def _pick_node(ctx):
    isl = ctx.load_npz(3, "islands")
    cli = ctx.load_npz(4, "climate_islands")
    # 有河、主岛不太大（测试快）
    ok = np.where(cli["has_river"] & (isl["main_area_km2"] < 1500) & (isl["main_area_km2"] > 300))[0]
    return int(ok[0]) if ok.size else 0


def _hash_dir(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.suffix in (".npz", ".png", ".csv")}


def test_island_deterministic_and_consistent(small_ctx):
    from skyisle_gen import island as isl
    node = _pick_node(small_ctx)
    out = isl.generate(small_ctx, node, res_m=300.0, steps=STEPS, log=lambda *a: None)
    h1 = _hash_dir(out)
    J1 = json.loads((out / "island.json").read_text(encoding="utf-8"))
    out2 = isl.generate(small_ctx, node, res_m=300.0, steps=STEPS, log=lambda *a: None)
    h2 = _hash_dir(out2)
    assert h1 == h2, "同输入重跑两次，产物不一致（IS-det）"
    c = J1["constraints"]
    assert abs(c["area_km2"]["actual"] - c["area_km2"]["target"]) <= 0.02 * c["area_km2"]["target"]
    assert abs(c["main_area_km2"]["actual"] - c["main_area_km2"]["target"]) <= 0.02 * c["main_area_km2"]["target"]
    # IS-surface：height_m 是主岛台面（陆地高程中位数），峰在它之上按岛龄 × 面积长出来
    assert abs(c["height_m"]["actual"] - c["height_m"]["target"]) <= max(10.0, 0.02 * c["height_m"]["target"])
    assert c["peak_m"]["actual"] > c["height_m"]["actual"]
    if STEPS >= 2:
        assert abs(c["arable_frac"]["actual"] - c["arable_frac"]["target"]) < 0.005      # IS-arable
        assert c["has_river"]["actual"] == c["has_river"]["target"]                       # IS-river
        assert all(not i["has_perennial_river"] for i in J1["islands"][1:])               # 小岛只有溪涧
        z = np.load(out / "terrain.npz")
        land = z["island_id"] >= 0
        assert np.isnan(z["height"][~land]).all() and not np.isnan(z["height"][land]).any()
        assert (z["landcover"][land] > 0).all() and (z["landcover"][~land] == 0).all()
        # 河道成形：常年河 / 溪涧每格有河宽水深；河道下切（主岛有河时至少一条入虚空、带瀑布落差）
        rv = z["river"] > 0
        assert (z["river_width_m"][rv] > 0).all() and (z["river_depth_m"][rv] > 0).all()
        assert (z["river_width_m"][~land] == 0).all() and not (z["floodplain"] & (rv | z["lake"])).any()
        if c["has_river"]["actual"]:
            r0 = J1["hydro"]["rivers"][0]
            assert J1["hydro"]["n_rivers"] >= 1 and r0["width_m"] > 0 and r0["depth_m"] > 0 and r0["waterfall_m"] > 0
        # 河道中心线（矢量渲染）：每条 ≥ 2 点、落在栅格内；常年河的线河宽 > 0；主岛有河时至少一条常年河线
        RV = json.loads((out / "rivers.json").read_text(encoding="utf-8"))
        Hh, Ww = z["height"].shape
        for L in RV["lines"]:
            P = np.array(L["pts"])
            assert P.shape[0] >= 2 and (P[:, 0] >= -1).all() and (P[:, 0] <= Hh + 1).all() and (P[:, 1] >= -1).all() and (P[:, 1] <= Ww + 1).all()
            assert (P[P[:, 3] > 0, 2] > 0).all()
        if c["has_river"]["actual"]:
            assert any(max(q[3] for q in L["pts"]) > 0 and L["island"] == 0 for L in RV["lines"])
        # 资源：地形区覆盖全部陆地；矿点在所属岛的陆地、不在水面，资源栅格上有标记
        Rj = json.loads((out / "resources.json").read_text(encoding="utf-8"))
        assert (z["terrain_zone"][land] > 0).all() and (z["terrain_zone"][~land] == 0).all()
        assert abs(sum(Rj["zones"]["share"].values()) - 1.0) < 1e-3
        for d in Rj["deposits"]:
            if d.get("cleared"):
                continue
            i, j = d["cell"]
            assert z["island_id"][i, j] == d["island"] and z["river"][i, j] == 0 and not z["lake"][i, j] and z["resource"][i, j] > 0
        assert any(d["kind"] == "quarry" and d["island"] == 0 for d in Rj["deposits"])
    if STEPS >= 3:
        C = json.loads((out / "climate.json").read_text(encoding="utf-8"))
        a, m = C["annual"], C["means_check"]                                                 # IS-season
        assert abs(m["precip_rel"] - a["precip_rel"]) <= 0.01 * a["precip_rel"] + 1e-6
        assert abs(m["storm"] - a["storm"]) <= 0.01 * max(a["storm"], 0.05) + 1e-6
        assert abs(m["window"] - a["window"]) <= 0.01 * a["window"] + 1e-6
        assert abs(m["temp_c"] - a["temp_c"]) <= 0.05
        assert C["season_type"] in ("four", "two", "rain", "storm", "none")
        assert len(C["seasons"]) == C["calendar"]["seasons"] and C["calendar"]["year_days"] == 336.0
        assert abs(sum(s["precip_mm"] for s in C["seasons"]) - a["precip_mm"]) <= 0.01 * a["precip_mm"] + 2
    if STEPS >= 4:
        W = C["weather"]
        assert W["n_days"] == 336 and len(W["days"]) == 336 and sum(W["types"].values()) == 336
        assert (out / "weather_y0.csv").exists()
        # 改年份不动地形与气候：只有天气产物变
        out3 = isl.generate(small_ctx, node, res_m=300.0, steps=STEPS, year=1, log=lambda *a: None)
        h3 = _hash_dir(out3)
        assert h3["terrain.npz"] == h1["terrain.npz"] and h3["height.png"] == h1["height.png"]
        assert (out3 / "weather_y1.csv").read_bytes() != (out / "weather_y0.csv").read_bytes()
    if STEPS >= 5:
        S = json.loads((out / "settlements.json").read_text(encoding="utf-8"))
        z = np.load(out / "terrain.npz")
        # SET-pop：村农户 + 散户 + 镇非农户 + 专业聚落户 = 人口 / 户均，镇 + 专业 = 非农户；SET-field：田块面积之和 = 可耕地；SET-site：村不在崖缘 / 水面 / 漫滩，且村之间 ≥ 1 km
        parts = S["households_in_villages"] + S["households_in_hamlets"] + S["households_in_towns_market"] + S["households_in_specials"]
        assert parts == S["households"] == round(S["population"] / S["household_size"])
        if S["villages"]:
            assert S["households_in_towns_market"] + S["households_in_specials"] == S["nonfarm_households"] == round(S["households"] * S["nonfarm_share"])
        assert abs(sum(f["area_km2"] for f in S["fields"]) - float((z["arable"] > 0).sum()) * (J1["raster"]["res_m"] / 1000) ** 2) < 1e-3
        assert all(f["households"] >= 8 for f in S["fields"] if f["village"] and f["village"] > 0)
        for v in S["villages"] + S["specials"]:
            i, j = v["cell"]
            assert not z["cliff"][i, j] and not z["lake"][i, j] and z["river"][i, j] == 0 and z["island_id"][i, j] == v["island"]
        cells = np.array([v["cell"] for v in S["villages"]], dtype=float)
        if cells.shape[0] > 1:
            d = np.sqrt(((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1)) * J1["raster"]["res_m"] / 1000
            np.fill_diagonal(d, 9e9)
            assert d.min() > 0.05, d.min()                       # 不同村不同格（1 km 间距是软项：放不下时退而求其次）
            assert (d.min(axis=1) >= 1.0 - 1e-9).mean() >= 0.8    # 八成以上的村满足 1 km 间距
        assert any(v.get("seat") for v in S["villages"]) and S["villages"][0]["island"] == 0 or S["n_villages"] == 0
        # SET-land：飞船随处可停——没有码头；每个村 / 专业聚落一块同岛、非水非崖的泊场；每条索桥两端各一桥头
        assert "docks" not in S
        lands = {L["id"]: L for L in S["landings"]}
        assert len(lands) == len(S["villages"]) + len(S["specials"])
        for v in S["villages"] + S["specials"]:
            L = lands[v["landing"]]
            i, j = L["cell"]
            assert z["island_id"][i, j] == v["island"] and not z["cliff"][i, j] and not z["lake"][i, j] and z["river"][i, j] == 0
        n_bridge = sum(1 for e in J1["links"] if e["kind"] == "bridge")
        assert len(S["bridgeheads"]) == 2 * n_bridge
        for d in S["bridgeheads"]:
            assert z["cliff"][d["cell"][0], d["cell"][1]] and z["island_id"][d["cell"][0], d["cell"][1]] == d["island"]
        # SET-town：邑治是镇，每个村归一个镇；开垦只减林地（林地占比不升）
        if S["villages"]:
            assert any(t["seat"] for t in S["towns"]) and all(v.get("market_town") for v in S["villages"])
        assert S["clearing"]["forest_share_after"] <= S["clearing"]["forest_share_before"]
        assert S["water_ok_share"] >= 0.8, S["water_ok_share"]
        if S["has_river"]:
            assert len(S["intakes"]) == sum(1 for v in S["villages"] if v["island"] == 0)
        else:
            assert len(S["cisterns"]) >= 1
        # SET-home：三个主家候选类型各不同，都落在陆地上；前哨都在有泊场（有村）的小岛
        kinds = [h["kind"] for h in S["home_candidates"]]
        assert len(kinds) == len(set(kinds)) and 2 <= len(kinds) <= 3, kinds
        for h in S["home_candidates"]:
            assert z["island_id"][h["cell"][0], h["cell"][1]] == h["island"]
        land_isl = {L["island"] for L in S["landings"]}
        assert all(o["island"] in land_isl and o["island"] != 0 for o in S["outposts"])
    # 岛数与大小：主岛最大，最小岛 ≥ 0.3 km²（离散化允许一格误差），总和 = area
    areas = [i["area_target_km2"] for i in J1["islands"]]
    assert areas[0] == max(areas)
    assert min(areas) >= 0.3 - 1e-6
    assert abs(sum(areas) - c["area_km2"]["target"]) < 1e-3
    # 连通：索桥 + 短渡把所有岛连成一片（IS-link）
    from skyisle_gen.graph import weak_components
    n = len(J1["islands"])
    src = np.array([e["a"] for e in J1["links"]], dtype=np.int64)
    dst = np.array([e["b"] for e in J1["links"]], dtype=np.int64)
    assert weak_components(n, src, dst).max() == 0


def test_shape_area_and_single_component():
    from skyisle_gen.island.terrain import island_shape
    from skyisle_gen.island.grid import label_components
    cfg = load_config()["island"]["terrain"]
    rng = np.random.default_rng(3)
    for area in (0.5, 12.0, 400.0):
        mask, inside, X, Y = island_shape(rng, area, 0.1, 1.6, 0.4, cfg)
        cells = mask.sum() * 0.01
        assert abs(cells - area) <= max(0.01, 0.01 * area) + 0.01
        _, n = label_components(mask)
        assert n == 1


def test_label_components_runs():
    from skyisle_gen.island.grid import label_components
    m = np.zeros((6, 6), dtype=bool)
    m[0, 0:3] = True
    m[1, 2] = True
    m[3, 4] = True
    m[4, 5] = True
    lab4, n4 = label_components(m, 4)
    lab8, n8 = label_components(m, 8)
    assert n4 == 3 and n8 == 2
    assert lab4[0, 0] == lab4[1, 2]
    assert lab8[3, 4] == lab8[4, 5] and lab4[3, 4] != lab4[4, 5]


def test_label_by_island_does_not_cross_islands():
    """两岛斜对角贴着（布局允许一格的岸距）：8 邻域的田块 / 林场不能并到别的岛上（seed 2026 #5246、seed 7 #418）。"""
    from skyisle_gen.island.grid import label_by_island, label_components
    iid = np.full((4, 4), -1, dtype=np.int16)
    iid[0:2, 0:2] = 0
    iid[2:4, 2:4] = 1
    m = iid >= 0
    _, n_plain = label_components(m, 8)
    lab, n = label_by_island(m, iid, 8)
    assert n_plain == 1 and n == 2
    assert lab[0, 0] == lab[1, 1] != lab[2, 2] == lab[3, 3]


def test_season_type_table():
    """5.4 的季型表：温差 ≥ 20 → 四季分明；雨季 ≥ 旱季 × 2.5 → 雨旱季；都不达标 → 常夏。"""
    from skyisle_gen.island.climate import _season_names
    n = _season_names("four", [5, 20, 25, 10], [1, 1, 1, 1], [0] * 4, [1] * 4, 4)
    assert n == ["冷季", "暖季", "热季", "凉季"]
    n = _season_names("rain", [20] * 4, [0.1, 0.5, 0.2, 0.15], [0] * 4, [1] * 4, 4)
    assert n == ["旱季", "雨季", "转季", "转季"]
    n = _season_names("storm", [20] * 4, [1] * 4, [0.9, 0.2, 0.1, 0.3], [0.2, 0.8, 0.9, 0.7], 4)
    assert n[0] == "风暴季" and n[2] == "平静季"


def test_daily_weather_returns_to_climate(small_ctx):
    """IS-daily：60 年逐日降水的平均回到气候值（< 5%），雨日比例落在设定 ±0.05（30 年时单季标准误约 5%，见 DESIGN-NOTES）。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.weather import multi_year_stats
    node = _pick_node(small_ctx)
    c = isl.island_config(small_ctx)
    inp = isl._node_inputs(small_ctx, node)
    g = isl.build_terrain(small_ctx, node, c, inp, res_m=400.0, log=lambda *a: None)
    from skyisle_gen.island.climate import build_climate, daily_curves
    build_climate(small_ctx, node, c, g, log=lambda *a: None)
    g["daily"] = daily_curves(g["climate"], inp, small_ctx.cfg["s04"]["climate"])
    st = multi_year_stats(small_ctx, node, c, g, years=60)
    assert st["annual_rel_err"] < 0.05 or st["annual_z"] < 3.0, st
    assert max(st["wet_frac_err"]) <= 0.05, st


def test_classify_all_small_world(small_ctx):
    """全量季型统计：每群有一个类别，类别名与代码对应；西风带（36–62°）应几乎全是四季分明（骨架第二版 C5 的口径）。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.climate import classify_all
    st = classify_all(small_ctx, isl.island_config(small_ctx), log=lambda *a: None)
    assert st["n"] == len(st["codes"]) == len(st["names"])
    assert abs(sum(st["share"].values()) - 1.0) < 1e-6
    west = [v for k, v in st["by_band"].items() if k.startswith("西风带")]
    assert west and west[0]["四季分明"] >= 0.9
