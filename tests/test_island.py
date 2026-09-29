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
        # 资源：地形区覆盖全部陆地；点与片在所属岛的陆地、不在水面，主导栅格上有标记，片的代表格在自己的 patch_id 上
        Rj = json.loads((out / "resources.json").read_text(encoding="utf-8"))
        assert (z["terrain_zone"][land] > 0).all() and (z["terrain_zone"][~land] == 0).all()
        assert abs(sum(Rj["zones"]["share"].values()) - 1.0) < 1e-3
        for d in Rj["deposits"]:
            if d.get("cleared"):
                continue
            i, j = d["cell"]
            assert z["island_id"][i, j] == d["island"] and z["river"][i, j] == 0 and not z["lake"][i, j] and z["resource"][i, j] > 0
            assert d["form"] != "patch" or z["patch_id"][i, j] == d["id"]
        # 片的面积 = patch_id 的格数（林场不再重复计数），林木合计不超过林地
        ck = (J1["raster"]["res_m"] / 1000.0) ** 2
        tim = [d for d in Rj["deposits"] if d["kind"] == "timber"]
        assert abs(sum(d["area_km2"] for d in tim) - np.isin(z["patch_id"], [d["id"] for d in tim]).sum() * ck) < 0.01 * max(1, len(tim))
        # 散：赋存场 [7, H, W]；岩类（金属矿 / 石料 / 硫磺）的场不上田（在种与撂荒的；P5 起已垦避开它）、不上湿地；赋存区与采场的格有效
        RF = z["res_field"]
        farm = (z["cultivated"] > 0) | (z["fallow_years"] > 0) if STEPS >= 5 else np.zeros(land.shape, dtype=bool)
        assert RF.shape == (len(Rj["fields"]["kinds"]),) + z["height"].shape and (RF[:, ~land] == 0).all()
        for k in Rj["fields"]["rock_kinds"]:
            f = RF[Rj["fields"]["kinds"].index(k)] > 0
            assert not (f & farm).any() and not (f & (z["landcover"] == 9)).any()
        assert Rj["occurrences"], "至少有石料 / 黏土 / 砂砾之一的赋存区"
        for o in Rj["occurrences"]:
            i, j = o["cell"]
            assert z["island_id"][i, j] == o["island"] and RF[Rj["fields"]["kinds"].index(o["kind"]), i, j] > 0
        for w in Rj["workings"]:
            i, j = w["cell"]
            assert Rj["occurrences"][w["occurrence"]]["kind"] == w["kind"]
            assert not farm[i, j] and z["landcover"][i, j] != 4 and z["river"][i, j] == 0 and not z["lake"][i, j] and not z["cliff"][i, j]
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
        parts = (S["households_in_villages"] + S["households_in_hamlets"] + S["households_in_towns_market"] + S["households_in_specials"]
                 + S["households_in_relays"])
        assert parts == S["households"] == round(S["population"] / S["household_size"])
        if S["villages"]:
            assert (S["households_in_towns_market"] + S["households_in_specials"] + S["households_in_relays"] == S["nonfarm_households"]
                    == round(S["households"] * S["nonfarm_share"]))
        assert abs(sum(f["area_km2"] for f in S["fields"]) - float((z["cultivated"] > 0).sum()) * (J1["raster"]["res_m"] / 1000) ** 2) < 1e-3
        assert all(f["households"] >= 8 for f in S["fields"] if f["village"] and f["village"] > 0 and not f.get("polder"))   # P6b：圩田的田块几户也算
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
        # SET-land：飞船随处可停——没有码头；每个村 / 专业聚落一块同岛、非水非崖的泊场；P5 起没有索桥、桥头与导水槽
        assert "docks" not in S
        lands = {L["id"]: L for L in S["landings"]}
        assert len(lands) == len(S["villages"]) + len(S["specials"]) + sum(t["harbor"] is not None for t in S["towns"]) + len(S["relays"])
        for v in S["villages"] + S["specials"] + S["towns"] + S["relays"]:
            L = lands[v["landing"]]
            i, j = L["cell"]
            assert z["island_id"][i, j] == v["island"] and not z["cliff"][i, j] and not z["lake"][i, j] and z["river"][i, j] == 0
        assert all(e["kind"] == "ferry" for e in J1["links"]) and "channels" not in J1 and "n_bridges" not in J1["layout"]
        assert "bridgeheads" not in S and "channels" not in S
        # P5：宜垦 / 已垦 / 撂荒——已垦 ⊂ 宜垦（梯田标记一致）、撂荒 ⊂ 宜垦且不与已垦重叠；宜垦 > 已垦 = 行星层额度；旧的 arable 不再写
        assert "arable" not in z.files and (out / "farmland.png").exists() and not (out / "arable.png").exists()
        cv, ct, fy = z["cultivable"], z["cultivated"], z["fallow_years"]
        pol = z["polder_id"] > 0                                  # P6：圩田是排干的湿地，不在宜垦里、记 1
        assert not ((ct > 0) & (ct != cv) & ~pol).any() and not ((fy > 0) & ((cv == 0) | (ct > 0))).any()
        assert (ct[pol] == 1).all() and (cv[pol] == 0).all()
        assert (cv > 0).sum() > (ct > 0).sum() > 0
        F = S["farmland"]
        assert F["cultivated_km2"] == F["quota_km2"] and F["cultivable_km2"] > F["cultivated_km2"]
        assert abs(J1["constraints"]["arable_frac"]["actual"] - J1["constraints"]["arable_frac"]["target"]) < 0.005
        # 定居门槛：有农户（村或散户）的岛，已垦都够 settle_min_hh 户的地；专业聚落的住法、工棚不在有农户的岛上、工棚 / 季节住记着住哪个村
        ck = (J1["raster"]["res_m"] / 1000) ** 2
        farm_isl = {v["island"] for v in S["villages"]} | {h["island"] for h in S["hamlets"]}
        for k in farm_isl:
            assert float(((z["island_id"] == k) & (ct > 0)).sum()) * ck >= F["settle_min_hh"] * S["land_per_household_km2"] - 1e-6 or F["forced_km2"] > 0
        # 大岛保底（用户 09-29 定）：主岛以外 ≥ island_floor_km2 且保底的岛都有村，保底的地连成一块
        for k in F["floor_islands"]:
            assert k != 0 and J1["islands"][k]["area_km2"] >= 30.0 and any(v["island"] == k for v in S["villages"])
        vids = {v["id"] for v in S["villages"]}
        for x in S["specials"]:
            assert x["occupancy"] in ("常住", "工棚", "季节住")
            assert x["occupancy"] != "工棚" or x["island"] not in farm_isl
            assert x["occupancy"] == "常住" or not S["villages"] or x["home_village"] in vids
        # 没人住 ≠ 没人用：有人用的岛都没人常住；废村在自己的岛上、撤空了几年、旁边的田撂荒同样的年头；每岛一条荒地归属
        res_isl = farm_isl | {x["island"] for x in S["specials"] if x["occupancy"] == "常住"} | {x["island"] for x in S["relays"] if x["households"] > 0}
        for u in S["uses"]:
            assert u["island"] != 0 and u["island"] not in res_isl and z["island_id"][u["cell"][0], u["cell"][1]] == u["island"]
        for r in S["ruins"]:
            assert r["abandoned_years"] >= 1 and z["island_id"][r["cell"][0], r["cell"][1]] == r["island"] and r["name"].startswith("废村")
            assert (fy == min(255, r["abandoned_years"])).any()
        assert [t["island"] for t in S["land_tenure"]] == list(range(len(J1["islands"])))
        for t in S["land_tenure"]:
            assert t["owner"] in ("村", "大户", "官用", "官荒") and (t["owner"] == "村") == (t["island"] in farm_isl)
            assert t["status"] in ("常住", "季节住", "有人用", "荒岛") and (t["status"] == "常住") == (t["island"] in res_isl)
        # SET-town：邑治是镇，每个村归一个镇；开垦只减林地（林地占比不升）
        if S["villages"]:
            assert any(t["seat"] for t in S["towns"]) and all(v.get("market_town") for v in S["villages"])
        # P7：本岛走路、跨岛只搭航船——走路赶集的村与镇同岛，每个搭航船的村恰在一条通到它的镇的航船线上，有别的岛上的人来赶集的镇都有航船；
        # 大泊场在本岛缓坡平地上、镇挨着的在 town_harbor_km 内；中转站的户 = 各角色之和、一岛至多一处；邑治的村带 seat
        tby = {t["id"]: t for t in S["towns"]}
        cover = {}
        for ln in S["boat_lines"]:
            for vid in ln["stops"]:
                cover[vid] = cover.get(vid, 0) + 1
        for v in S["villages"]:
            assert v["market_mode"] in ("步行", "航船")
            if v["market_mode"] == "步行":
                assert tby[v["market_town"]]["island"] == v["island"] and "boat_line" not in v
            else:
                assert cover.get(v["id"]) == 1 and S["boat_lines"][v["boat_line"] - 1]["town"] == v["market_town"]
        assert all(t["n_lines"] > 0 for t in S["towns"] if t["served_other_islands_households"] > 0)
        assert sum(t["seat"] for t in S["towns"]) == (1 if S["villages"] else 0)
        seat_t = next((t for t in S["towns"] if t["seat"]), None)
        assert seat_t is None or next(v for v in S["villages"] if v["id"] == seat_t["village"]).get("seat")
        for h in S["harbors"]:
            i, j = h["cell"]
            assert z["island_id"][i, j] == h["island"] and z["slope_deg"][i, j] <= np.float32(4.0) and not z["cliff"][i, j] and h["ships"] > 0
        for t in S["towns"]:
            if t["harbor"] is not None:
                h = S["harbors"][t["harbor"] - 1]
                assert h["island"] == t["island"] and h["town"] == t["id"] and t["harbor_dist_km"] <= 2.0 + 1e-9 and t["street"]["to_km"] == h["km"]
        assert len({r["island"] for r in S["relays"]}) == len(S["relays"])
        for r in S["relays"]:
            assert sum(r["roles"].values()) == r["households"] and (r["occupancy"] == "常住") == (r["households"] > 0) and r["functions"]
            assert z["island_id"][r["cell"][0], r["cell"][1]] == r["island"] == z["island_id"][r["lookout_cell"][0], r["lookout_cell"][1]]
        assert not any(u["kind"] == "烽火台" for u in S["uses"])          # 烽火台 P7 起归中转站
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
        # P6 水利：渠首在常年河 / 溪涧上，谷口的渠除渠首外走在本岛陆地上；一块田只归一处渠首；每村至多一口塘；塘、闸在本岛陆地上；
        # 圩号栅格与圩的格数一致，圩田在额度之内（上面已垦 = 额度照旧成立）
        W = S["waterworks"]
        wet = z["landcover"] == 9
        for h in W["heads"]:
            i, j = h["cell"]
            assert (z["river"][i, j] > 0 or z["stream"][i, j] > 0) and z["island_id"][i, j] == h["island"] and h["served_km2"] > 0
        for cn in W["canals"]:
            if cn["kind"] in ("干渠", "支渠"):
                for r_, q_ in cn["pts"][1:]:
                    i, j = int(r_), int(q_)
                    assert z["island_id"][i, j] == cn["island"] and not z["cliff"][i, j] and not z["lake"][i, j] and z["river"][i, j] == 0 and not wet[i, j]
        cmd = [f for h in W["heads"] for f in h["fields"]]
        assert len(cmd) == len(set(cmd))
        vp = [p["village"] for p in W["ponds"] if p["kind"] in ("村塘", "山塘") and not p.get("abandoned")]
        assert len(vp) == len(set(vp)) and set(vp) <= {v["id"] for v in S["villages"]}
        for x in W["ponds"] + W["sluices"]:
            i, j = x["cell"]
            assert z["island_id"][i, j] == x["island"] and not z["cliff"][i, j] and not z["lake"][i, j]
        cnt = np.bincount(z["polder_id"].ravel(), minlength=len(W["polders"]) + 1)
        assert all(int(cnt[p["id"]]) == p["cells"] for p in W["polders"]) and int(pol.sum()) == sum(p["cells"] for p in W["polders"])
        assert abs(W["summary"]["polder_km2"] - S["farmland"]["polder_km2"]) < 1e-9 and W["summary"]["polder_km2"] <= W["summary"]["wetland_km2"]
        # P6b（L30）：每处水利都有管它的村、在它走得到的范围内（圩量到每格最远的角），渠首只灌那个村的田；废弃的没有村、记着废村与年头
        res = J1["raster"]["res_m"] / 1000
        walk = W["summary"]["manage_walk_km"] / res + 1e-9
        V = {v["id"]: v for v in S["villages"]}
        rid = {r["id"] for r in S["ruins"]}
        def far(vid, pts):
            P = np.asarray(pts, dtype=float).reshape(-1, 2)
            return float(np.hypot(P[:, 0] - V[vid]["cell"][0] - 0.5, P[:, 1] - V[vid]["cell"][1] - 0.5).max())
        for kind, xs in (("heads", W["heads"]), ("canals", W["canals"]), ("ponds", W["ponds"]), ("sluices", W["sluices"])):
            for x in xs:
                if x.get("abandoned"):
                    assert x["village"] is None and x["ruin"] in rid and x["abandoned_years"] >= 1
                    continue
                pts = x["pts"] if kind == "canals" else [[x["cell"][0] + 0.5, x["cell"][1] + 0.5]]
                assert x["village"] in V and far(x["village"], pts) <= walk, (kind, x["id"])
        for h in W["heads"]:
            assert h.get("abandoned") or h["fields"] == [V[h["village"]]["field"]]
        W_ = z["polder_id"].shape[1]
        for pr in W["polders"]:
            ii, jj = np.nonzero(z["polder_id"] == pr["id"])
            corners = [[ii + a, jj + b] for a in (0, 1) for b in (0, 1)]
            assert pr["village"] in V and max(far(pr["village"], np.stack(cc, 1)) for cc in corners) <= walk
        pf_ids = {f["id"] for f in S["fields"] if f.get("polder")}
        assert all(v["field"] in pf_ids for v in S["villages"] if v.get("polder"))                  # 圩村落在自己那组圩田上
        assert all(set(v.get("polder_fields", [])) <= pf_ids for v in S["villages"])
        # P6b（L31）：原始地表（没有人以前）与人工改造——圩田原是湿地、地表变了的都说得清、没农户的岛湿地原样；island.json 给原始 / 现状 / 各类改造的面积
        nat, lu = z["landcover_natural"], z["landuse"]
        land = z["island_id"] >= 0
        assert (nat[pol] == 9).all() and (lu[pol] == 4).all() and not (land & (z["landcover"] != nat) & (lu == 0)).any()
        assert (lu[(ct > 0) & ~pol] != 0).all() and (lu[fy > 0] == 5).all() and not (~land & (lu > 0)).any()
        LC = J1["landcover"]
        assert set(LC["natural_km2"]) == set(LC["current_km2"]) and abs(sum(LC["landuse"]["km2"].values()) - float((land & (lu > 0)).sum()) * ck) < 1e-2
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


P3_SETS = ["island.terrain.age_young=0.6", "island.resources.density_per_100km2.salt=3.0", "island.resources.fossil_per_km2=1.0"]


def test_resources_from_the_seafloor(small_ctx, tmp_path):
    """P3（没有火山，DESIGN-NOTES 四点三十四）：没有熔岩管 / 火山口（文字里也没有）；骨架空洞开在浮石露头旁；温泉只在新岛；
    石料岩性是三层之一、金属矿是海底带上来的那几种；岩盐是赋存场第 7 层，盐泉在岩盐上，盐井村挂在有盐井的岩盐区上；贝壳化石在海相石灰岩里。
    小世界里新岛、盐丘、化石都少：把新岛门槛、盐丘与化石的密度调高，让每样都出得来（RES-geo 逐条查）。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.check import evaluate
    from skyisle_gen.island.resources import LITH_LAYERS, ORE_ORIGIN
    node = _pick_node(small_ctx)
    out, g = isl.generate(small_ctx, node, res_m=300.0, steps=STEPS, sets=P3_SETS, log=lambda *a: None, return_state=True, out_root=tmp_path)
    items = {i["id"]: i for i in evaluate(g, out)}
    for k in ("RES-site", "RES-occ", "RES-work", "RES-geo"):
        assert items[k]["pass"], items[k]
    R, J, S = g["resources"], g["json"], g["settle"]
    text = json.dumps(R, ensure_ascii=False) + json.dumps(S, ensure_ascii=False) + json.dumps(J["resources"], ensure_ascii=False)
    assert "火山" not in text and "熔岩" not in text
    assert R["fields"]["kinds"][-1] == "salt" and g["res_field"].shape[0] == 7 and "old_island_lithology" not in R["geology"]
    kinds = {d["kind"] for d in R["deposits"]} | {o["kind"] for o in R["occurrences"]}
    assert {"salt", "saltspring", "fossil"} <= kinds, kinds
    assert any(d.get("subtype") == "骨架空洞" for d in R["deposits"])
    ages = {i["id"]: i["age_zh"] for i in J["islands"]}
    assert all(ages[d["island"]] == "新岛" for d in R["deposits"] if d["kind"] == "hotspring")
    assert {o["subtype"] for o in R["occurrences"] if o["kind"] == "stone"} <= set(LITH_LAYERS)
    assert {o["subtype"] for o in R["occurrences"] if o["kind"] == "ore"} <= set(ORE_ORIGIN)
    wells = {w["occurrence"]: w for w in R["workings"] if w["kind"] == "salt"}
    assert wells and all(R["occurrences"][o]["kind"] == "salt" for o in wells)
    for s in S["specials"]:
        if s["kind"] == "盐井村":
            assert R["occurrences"][s["occurrence"]]["kind"] == "salt" and wells[s["occurrence"]]["special"] == s["id"]


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


# ---------------- 势力范围（territory.py，DESIGN-NOTES 四点二十二）与粗版（lod.py） ----------------
def test_territory_split_is_consistent(small_ctx):
    """邻群两边各画的分界线是同一条（k 离线 t_k、j 离线 t_j，t_k + t_j = 群心距），各退半道缝后两群之间正好隔 gap_km。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.territory import limits
    c = isl.island_config(small_ctx)
    tc = dict(c["territory"], reach=50.0, reach_km=5000.0)   # 小世界稀：放宽只为找到一对邻群
    k = _pick_node(small_ctx)
    lk = limits(small_ctx, k, isl._node_inputs(small_ctx, k), tc)
    assert lk
    j = lk[0]["node"]
    lj = {L["node"]: L for L in limits(small_ctx, j, isl._node_inputs(small_ctx, j), tc)}
    assert k in lj
    gap = float(tc["gap_km"])
    assert abs(lk[0]["limit_km"] + lj[k]["limit_km"] + gap - lk[0]["dist_km"]) < 1e-3      # dist_km 记到米
    u1, u2 = np.array(lk[0]["u"]), np.array(lj[k]["u"])
    assert np.hypot(*u1) == pytest.approx(1.0) and u1 @ -u2 > 0.99     # 两边的法向相反（几百 km 内近似平面）


def test_territory_nearest_fit():
    """Dykstra：放得下时给离原点最近的偏移，放不下时越界量 > 0。"""
    from skyisle_gen.island.territory import nearest_fit, violation
    lim = [{"u": (1.0, 0.0), "limit_km": 10.0}, {"u": (0.0, 1.0), "limit_km": 10.0}, {"u": (-1.0, 0.0), "limit_km": 10.0}]
    o, v = nearest_fit(np.array([15.0, 4.0, 4.0]), lim)          # 向东伸 15 km：要往西挪 5 km
    assert v <= 1e-6 and o[0] == pytest.approx(-5.0, abs=1e-3) and abs(o[1]) < 1e-3
    assert violation(o, np.array([15.0, 4.0, 4.0]), lim) <= 1e-6
    o, v = nearest_fit(np.array([15.0, 4.0, 15.0]), lim)         # 东西都伸 15，只有 20 km 宽：放不下
    assert v > 4.0


def test_territory_off_leaves_unconstrained_groups_identical(small_ctx):
    """没碰到约束的群，开不开势力范围产物逐字节一样（布局照旧摆，越界了才重摆）。"""
    from skyisle_gen import island as isl
    node = _pick_node(small_ctx)
    c = isl.island_config(small_ctx)
    inp = isl._node_inputs(small_ctx, node)
    g1 = isl.build_terrain(small_ctx, node, c, inp, res_m=400.0, log=lambda *a: None)
    t = g1["json"]["constraints"]["territory"]
    if t.get("constrained"):
        pytest.skip("这个节点碰到了约束")
    c2 = dict(c, territory=dict(c["territory"], enabled=False))
    g2 = isl.build_terrain(small_ctx, node, c2, inp, res_m=400.0, log=lambda *a: None)
    assert np.array_equal(g1["island_id"], g2["island_id"])
    assert np.array_equal(np.nan_to_num(g1["height"], nan=-1.0), np.nan_to_num(g2["height"], nan=-1.0))
    assert t.get("violation_km", -1.0) <= 0.0


def test_lod_block_reduce():
    """粗版降采样：陆地占比、块内平均高 / 最高、岛号与地表取众数。"""
    from skyisle_gen.island.lod import _block_reduce
    iid = np.full((4, 4), -1, dtype=np.int16)
    iid[0:2, 0:2] = 0
    iid[0, 2] = 1
    h = np.where(iid >= 0, 100.0, np.nan)
    h[0, 0] = 300.0
    lc = np.where(iid >= 0, 4, 0).astype(np.uint8)
    lc[1, 1] = 6
    g = {"island_id": iid, "height": h, "landcover": lc, "river": np.zeros((4, 4), np.uint8), "lake": np.zeros((4, 4), bool)}
    r = _block_reduce(g, 2)
    assert r["land"][0, 0] == 255 and r["land"][0, 1] == 64 and r["land"][1, 1] == 0
    assert r["height"][0, 0] == pytest.approx(150.0) and r["peak"][0, 0] == pytest.approx(300.0)
    assert np.isnan(r["height"][1, 1])
    assert r["island"][0, 0] == 0 and r["island"][0, 1] == 1 and r["island"][1, 1] == -1
    assert r["landcover"][0, 0] == 4


def test_lod_keeps_land_and_layout(small_ctx):
    """粗版 = 原生分辨率生成再降采样：陆地面积与原生一样（占比按格加总），岛数、各岛的岸缘 / 峰原样带过去。"""
    from skyisle_gen import island as isl
    from skyisle_gen.island.lod import build_lod
    node = _pick_node(small_ctx)
    c = isl.island_config(small_ctx)
    out = build_lod(small_ctx, node, c, [1600.0, 3200.0], native_res_m=400.0)
    inp = isl._node_inputs(small_ctx, node)
    g = isl.build_terrain(small_ctx, node, c, inp, res_m=400.0, log=lambda *a: None)
    native_km2 = float((g["island_id"] >= 0).sum()) * 0.16
    for res, (arr, meta) in out.items():
        land_km2 = float(arr["land"].astype(np.float64).sum()) / 255.0 * (meta["raster"]["res_m"] / 1000.0) ** 2
        assert meta["raster"]["factor"] == round(res / 400.0)
        assert abs(land_km2 - native_km2) <= 0.01 * native_km2, (res, land_km2, native_km2)
        assert len(meta["islands"]) == len(g["json"]["islands"])
        assert [i["center_km"] for i in meta["islands"]] == [i["center_km"] for i in g["json"]["islands"]]   # 岸缘等水系之后会改，岛心不会
        assert np.nanmax(arr["peak"]) == pytest.approx(float(np.nanmax(g["height"])), abs=5.0)   # 水系（填洼、河道）会动几米


def test_lod_weather_matches_generate(small_ctx, tmp_path):
    """粗版带的天气（island lod --weather，DESIGN-NOTES 四点二十七）：与 generate 写的 weather_y0.csv 逐行相同、temp_rim_c 与 climate.json 的逐日表相同；
    带不带天气，块降采样的数组与 meta 一样；气候参数头与 island.json / climate.json 对得上；不带天气（或年份不对）的已有粗版在 --weather 时要重跑。"""
    import csv
    from skyisle_gen import island as isl
    from skyisle_gen.island.lod import _done, build_lod, write_lod
    node = _pick_node(small_ctx)
    c = isl.island_config(small_ctx)
    arr, meta = build_lod(small_ctx, node, c, [1600.0], native_res_m=400.0)[1600.0]
    arr0, meta0 = build_lod(small_ctx, node, c, [1600.0], native_res_m=400.0, weather=False)[1600.0]
    assert sorted(arr0) == sorted(k for k in arr if not k.startswith("weather_"))
    for k in arr0:
        assert arr0[k].dtype == arr[k].dtype and np.array_equal(arr0[k], arr[k], equal_nan=arr[k].dtype.kind == "f"), k
    assert {k: v for k, v in meta.items() if k != "seconds"} == {k: v for k, v in meta0.items() if k != "seconds"}

    out = isl.generate(small_ctx, node, res_m=400.0, steps=4, log=lambda *a: None, out_root=tmp_path / "isl")
    head = json.loads(str(arr["weather_meta"]))
    with open(out / "weather_y0.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == head["n_days"] == arr["weather_day"].size
    for d, r in enumerate(rows):
        mine = {"day": str(int(arr["weather_day"][d])), "season": str(int(arr["weather_season"][d])),
                "season_name": head["season_names"][int(arr["weather_season"][d])], "month": str(int(arr["weather_month"][d])),
                "day_of_month": str(int(arr["weather_day_of_month"][d])), "type": head["types"][int(arr["weather_type"][d])],
                "precip_mm": str(float(arr["weather_precip_mm"][d])), "temp_c": str(float(arr["weather_temp_c"][d])),
                "wind_from_deg": str(int(arr["weather_wind_from_deg"][d])), "wind_ms": str(float(arr["weather_wind_ms"][d])),
                "sailable": str(bool(arr["weather_sailable"][d])), "storm_event": str(int(arr["weather_storm_event"][d]))}
        assert mine == r, d
    clim = json.loads((out / "climate.json").read_text(encoding="utf-8"))
    J = json.loads((out / "island.json").read_text(encoding="utf-8"))
    assert arr["weather_temp_rim_c"].tolist() == [x["temp_rim_c"] for x in clim["weather"]["days"]]
    assert head["climate"]["ref_m"] == clim["annual"]["temp_ref_height_m"] and head["climate"]["rim_m"] == J["islands"][0]["rim_m"]
    assert head["climate"]["rim_m"] == meta["islands"][0]["rim_m"]
    assert head["precip_mm"] == J["hydro"]["precip_mm"] and head["summary"] == J["weather"]
    assert head["season_names"] == clim["season_names"] and head["calendar"]["year_days"] == clim["calendar"]["year_days"]

    p = write_lod(tmp_path / "lod", 1600.0, arr0, meta0)
    assert _done(p, weather=False) and not _done(p, weather=True)
    p = write_lod(tmp_path / "lod", 1600.0, arr, meta)
    assert _done(p, weather=True, year=0) and not _done(p, weather=True, year=1)
    with np.load(p) as z:
        assert np.array_equal(z["weather_precip_mm"], arr["weather_precip_mm"]) and str(z["weather_meta"]) == str(arr["weather_meta"])
