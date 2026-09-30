"""5.3c 地形区与资源分布的前端（DESIGN-NOTES 四点十六 / 四点十七 / 四点二十一 / 四点三十四）：算法在 C++ 核心里
（core/src/island/resources.cpp，Python 参考版删于 2026-09-30，tag python-reference-final）；这里只剩类表与中文名、备注、
岩类可放区（check 的 RES-occ 用）、resources.json 的整体（resource_record，decode.py 译回之后拼）与摘要、写 png / json 与总览图。

全部从已生成的地形、水系、地表与节点的地质背景（岛龄、板块边界类型与远近、叠层）推出，不读人口、不回灌（第三层）。
随机数走 island:<节点>:resources:…。资源按形态分三种说法：
- 点（泉眼、温泉、洞穴、盐泉、贝壳化石）：位置就是资源，只占一格；记在 deposits。
- 片（林木、泥炭 / 芦苇、浮石露头、鸟粪石）：边界清楚的一片地就是资源；记在 deposits，占的格写进 patch_id（片与片、片与点互斥）。
- 散（金属矿、石料、黏土、砂砾、砂金、硫磺、岩盐）：分三层——
  赋存场（res_field：每类每格 0–1 品位，各类可叠在同一格）→ 赋存区（occurrences：品位 ≥ occ_thr 的连通块，可命名、可叙述）→
  采场（workings：人在哪挖）。稀缺的人就矿：矿坑、硫磺坑、盐井在这里按品位挑；常用的就近取：采石场、土坑、采砂场、淘金点在聚落之后按村挑（C++ 的 tiers）。
  岩类（金属矿、石料、硫磺）只在「岩类可放区」rock_site：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ rock_hill_slope_deg；
  林坡算（采场那格改裸岩），平地的林、耕地、湿地、漫滩不算。沉积类（黏土、砂砾、砂金、岩盐）的赋存可以压在田下，坑不上田不上林（2026-09-26 拍板）。

没有火山（P3，Zhouzhu docs/PLAN-LAND.md L16 / L17）：岩浆在海下，岛是从海底挣脱出来的拱，顶上带着海底的岩层一起升上来。
于是矿是海底带上来的（热泉硫化物：铜、铅锌、金银、铁帽、少量锡；蛇纹岩：铬、镍；锰结核），硫磺是海底热泉沉积的，温泉只在刚出海的新岛（余热），
熔岩管换成浮石骨架里的空洞（开在浮石露头旁），火山口删掉；海底的沉积层带来海相石灰岩（石料的一种岩性）、贝壳化石、岩盐与盐泉。
岩层（B2 起，DESIGN-NOTES 四点四十六）按地形的层面（terrain.npz 的 lith，[island.strat]）：沉积盖层（石灰岩 / 泥灰岩互层）、辉长岩、蛇纹岩、浮石骨架。

产物：terrain_zone（uint8，见 ZONE_NAMES）、res_field（uint8 [7, H, W]，品位 × 255，层序 FIELD_KINDS）、patch_id（int32，−1 = 无）、
resource（uint8，见 RES_NAMES：显示用的「主导」类，按 DOMINANT_ORDER 后画盖先画）进 terrain.npz；
resources.json 列出点与片（deposits）、赋存区（occurrences）、采场（workings）；resources.png 主导类索引色；preview_resources.png 总览。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .hydro import LC_ALPINE, LC_ROCK, LC_WET

ZONE_NAMES = ["虚空", "高山", "山地", "丘陵", "台地平原", "河谷", "崖缘", "水域"]
ZONE_PALETTE = [(20, 24, 40), (235, 235, 240), (150, 110, 80), (200, 170, 110), (170, 200, 120), (110, 190, 170), (90, 80, 75), (40, 90, 200)]

# 资源类：键 → (中文, 调色, 形态)。形态：point 点 / patch 片 / field 散（赋存场）。编码 = 本表序号 + 1
RES_KINDS = [
    ("timber", "林木", (60, 130, 60), "patch"),
    ("spring", "泉眼", (120, 220, 255), "point"),
    ("clay", "黏土", (190, 120, 80), "field"),
    ("peat", "泥炭 / 芦苇", (110, 90, 60), "patch"),
    ("gravel", "砂砾", (205, 195, 170), "field"),
    ("placer", "砂金", (255, 215, 80), "field"),
    ("stone", "石料", (225, 222, 210), "field"),         # 浅石色：旧的 (170, 170, 185) 和晕渲的灰分不开，看上去全岛都是石料
    ("floatstone", "浮石", (150, 100, 230), "patch"),
    ("ore", "金属矿", (200, 60, 50), "field"),
    ("hotspring", "温泉", (255, 140, 180), "point"),
    ("sulfur", "硫磺", (230, 230, 60), "field"),
    ("cave", "洞穴", (40, 40, 40), "point"),
    ("guano", "鸟粪石", (240, 240, 210), "patch"),
    # P3（没有火山）：海底的沉积层跟着岛升上来——追加在末尾，前 13 类的编码不变
    ("salt", "岩盐", (215, 235, 245), "field"),
    ("saltspring", "盐泉", (70, 200, 190), "point"),
    ("fossil", "贝壳化石", (235, 180, 120), "point"),
]
RES_NAMES = ["无"] + [k[1] for k in RES_KINDS]
RES_PALETTE = [(0, 0, 0)] + [k[2] for k in RES_KINDS]
RES_INDEX = {k[0]: i + 1 for i, k in enumerate(RES_KINDS)}
RES_FORM = {k[0]: k[3] for k in RES_KINDS}
FIELD_KINDS = ["ore", "sulfur", "placer", "clay", "gravel", "stone", "salt"]       # res_field 的层序（岩盐是 P3 加的第 7 层）
ROCK_KINDS = ("ore", "sulfur", "stone")                                      # 岩类：只在 rock_site 里
WORK_ZH = {"ore": "矿坑", "sulfur": "硫磺坑", "placer": "淘金点", "stone": "采石场", "clay": "土坑", "gravel": "采砂场", "salt": "盐井"}
# 主导栅格（显示用）的画法顺序：后画的盖先画的——稀的盖常的，点最后
DOMINANT_ORDER = ["timber", "stone", "gravel", "clay", "peat", "guano", "floatstone", "placer", "salt", "sulfur", "ore",
                  "spring", "hotspring", "saltspring", "fossil", "cave"]

ORE_ORIGIN = {"铜": "热泉硫化物", "铅锌": "热泉硫化物", "金银": "热泉硫化物", "铁": "热泉硫化物", "锡": "热泉硫化物",
              "铬": "蛇纹岩", "镍": "蛇纹岩", "锰": "锰结核"}
ORE_NOTE = {"热泉硫化物": "热泉硫化物矿化带：海底热泉（黑烟囱）一边冒一边沉积的块状硫化物，随岛从海底带上来（顺板块走向）；"
                     "只画在岩类可放区里，林下 / 田下的不算（前工业时代找不到、也开不了）",
            "蛇纹岩": "蛇纹岩带：海底下的地幔岩随岛带上来、蚀变成蛇纹岩，带着铬铁矿与镍（顺板块走向）；只画在岩类可放区里，林下 / 田下的不算",
            "锰结核": "锰结核层：深海底铺着的一层结核，随岛带上天（顺板块走向）；只画在岩类可放区里，林下 / 田下的不算"}
# 石料的岩性（按剥蚀指数从浅到深）：沉积盖层 / 海底地壳 / 地幔岩
LITH_LAYERS = ("海相石灰岩", "辉长岩", "蛇纹岩")
# 备注（C++ 记录里的代码见 decode.NOTE_ZH，两边文字一样）
NOTE_HOTSPRING = "刚出海的新岛从海底带上来的余热：几万年就凉了，老一点的岛上没有"
NOTE_SULFUR = "海底热泉沉积的自然硫，随岛带上来；岛刚出海、还没风化掉，老一点的岛上早已氧化流失"
NOTE_VOID = "浮石骨架里天然的空洞：开在浮石露头旁，从这里钻进岛体"
NOTE_SALT = "海底沉积层里夹的盐层跟着岛升上来，拱成盐丘"
NOTE_SALTSPRING = "溪水流过地下的盐层冒出的咸泉：熬它就出盐"
NOTE_FOSSIL = "海相石灰岩里的贝壳化石层：海底的沉积跟着岛一起升上天"


def _shape(ii: np.ndarray, jj: np.ndarray, res_km: float) -> tuple[float, float | None]:
    """赋存区的长度（km，按主轴方差：均匀椭圆的全长 = 4σ）与走向（自东逆时针 0–180°）。"""
    if ii.size < 3:
        return round(max(1.0, math.sqrt(ii.size)) * res_km, 2), None
    C = np.cov(np.vstack([jj * res_km, -ii * res_km]))
    w, V = np.linalg.eigh(C)
    length = max(4.0 * math.sqrt(max(float(w[-1]), 0.0)), res_km)
    return round(length, 2), round(math.degrees(math.atan2(V[1, -1], V[0, -1])) % 180.0, 0)


def rock_site_mask(g: dict, zone: np.ndarray, rock_hill_slope_deg: float) -> np.ndarray:
    """岩类（金属矿 / 石料 / 硫磺）可放区：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ rock_hill_slope_deg；
    不含耕地、湿地、漫滩、水、崖缘。check 的 RES-occ 也按它验。"""
    land = g["island_id"] >= 0
    cover = g["landcover"]
    slope = g["slope_deg"]
    water = (g["river"] > 0) | g["lake"]
    flood = g.get("floodplain", np.zeros(land.shape, dtype=bool))
    terrain = np.isin(zone, [1, 2]) | ((zone == 3) & (slope >= rock_hill_slope_deg)) | np.isin(cover, [LC_ROCK, LC_ALPINE])
    return land & ~water & ~g["cliff"] & (g["arable"] == 0) & (cover != LC_WET) & ~flood & terrain


def resource_record(node, zone, land, rc, r_cells, thr, btype, kern, layered, geo_ore, fs_rate, deposits, occurrences, workings) -> dict:
    """resources.json 的整体（除 resource_summary 补的计数）；C++ 的记录由 decode.py 译回同形。"""
    return {"node": node, "zones": {"classes": ZONE_NAMES, "palette": ZONE_PALETTE,
                                    "share": {ZONE_NAMES[i]: round(float(((zone == i) & land).sum()) / max(1, int(land.sum())), 4) for i in range(1, len(ZONE_NAMES))},
                                    "rule": f"局地起伏 = {2 * r_cells + 1}×{2 * r_cells + 1} 格方窗内高差；山地 ≥ {rc['mountain_relief_m']} m 或坡 ≥ {rc['mountain_slope_deg']}° 或高于岸缘→峰的 {rc['mountain_peak_frac']}，"
                                            f"丘陵 ≥ {rc['hill_relief_m']} m 或坡 ≥ {rc['hill_slope_deg']}° 或高于 {rc['hill_peak_frac']}；高山 = 山地且（海拔温度 < 高山草甸线或近峰）；河谷 = 漫滩 + 河边缓坡"},
            "resources": {"classes": RES_NAMES, "palette": RES_PALETTE, "forms": {k[1]: k[3] for k in RES_KINDS},
                          "dominant_order": [RES_NAMES[RES_INDEX[k]] for k in DOMINANT_ORDER]},
            "fields": {"kinds": FIELD_KINDS, "names": [RES_NAMES[RES_INDEX[k]] for k in FIELD_KINDS], "thr": thr,
                       "rock_kinds": list(ROCK_KINDS), "rock_hill_slope_deg": float(rc["rock_hill_slope_deg"]),
                       "rule": f"res_field = 品位 × 255。岩类（金属矿 / 石料 / 硫磺）只在岩类可放区：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ {rc['rock_hill_slope_deg']}°"
                               "（林坡算，平地林、耕地、湿地、漫滩不算）；沉积类（黏土 / 砂砾 / 砂金 / 岩盐）可压在田下。赋存区 = 品位 ≥ thr 的连通块（矿化带 / 硫磺 / 盐丘按各自的核）"},
            "geology": {"boundary_type": btype, "boundary_kernel": kern, "layered": layered,
                        "ore_multiplier": round(geo_ore, 3), "floatstone_expose_rate": round(fs_rate, 3),
                        "floatstone": "岛体本身就是浮石；可开采，采掉的量相对岛体微不足道，不影响浮空",
                        "origin": "岩浆只在海下、岛上不喷发：岛是从海底挣脱出来的拱，顶上带着海底的岩层一起升上来——矿、硫磺、盐、石灰岩与贝壳化石都是海底带上来的，温泉只在刚出海的新岛",
                        "layers": {"names": list(LITH_LAYERS),
                                   "rule": "按地形的层面（B2，terrain.npz 的 lith、[island.strat]）：露出哪层 = 该格在构造顶面下多深——沉积盖层（石灰岩 / 泥灰岩互层："
                                           "海相石灰岩、岩盐、盐泉、溶洞）、辉长岩、蛇纹岩（铬、镍只在这里）、浮石骨架（岸崖、贴着岸缘的深谷）；石料赋存区取区内露出最多的一层"}},
            "deposits": deposits, "occurrences": occurrences, "workings": workings,
            "note": "第三层叙事 / 场景素材，不进管线；cell = 群栅格 [行, 列]，km = 相对群心（x 东 y 北）。deposits = 点与片，occurrences = 散的赋存区（cell = 品位峰值格，"
                    "axis_deg = 走向，自东逆时针），workings = 采场（villages / special = 用它的村 / 专业聚落）；grade 是本类里的相对品位"}


def resource_summary(g: dict) -> None:
    """计数、面积、采场数、林木外占陆地、地表占比与 island.json 的 resources 摘要（片的面积、主导栅格在 C++ 里已同步好）。"""
    from .output import LANDCOVER_CLASSES
    R = g["resources"]
    J = g["json"]
    res = g["resource"]
    land = g["island_id"] >= 0
    deps = R["deposits"]
    counts: dict[str, int] = {}
    area: dict[str, float] = {}
    for d in deps + R["occurrences"]:
        if d.get("cleared"):
            continue
        counts[d["kind_zh"]] = counts.get(d["kind_zh"], 0) + 1
        area[d["kind_zh"]] = round(area.get(d["kind_zh"], 0.0) + d["area_km2"], 3)
    n_work = np.bincount([w["occurrence"] for w in R["workings"]], minlength=len(R["occurrences"])) if R["workings"] else np.zeros(len(R["occurrences"]), dtype=int)
    for o in R["occurrences"]:
        o["n_workings"] = int(n_work[o["id"]])
    wc: dict[str, int] = {}
    for w in R["workings"]:
        wc[w["kind_zh"]] = wc.get(w["kind_zh"], 0) + 1
    R["counts"], R["area_km2"], R["workings_counts"] = counts, area, wc
    n_land = max(1, int(land.sum()))
    R["non_timber_share"] = round(float(((res > 0) & (res != RES_INDEX["timber"])).sum()) / n_land, 4)
    cover = g["landcover"]
    J["landcover"]["share"] = {LANDCOVER_CLASSES[i]: round(float(((cover == i) & land).sum()) / n_land, 4) for i in range(1, 12)}
    J["resources"] = {"zones_share": R["zones"]["share"], "counts": counts, "workings": wc}


def write_resources(out: Path, g: dict) -> None:
    from .grid import write_png8
    write_png8(out / "resources.png", g["resource"], RES_PALETTE)
    write_png8(out / "terrain_zone.png", g["terrain_zone"], ZONE_PALETTE)
    (out / "resources.json").write_text(json.dumps(g["resources"], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def write_preview_resources(out: Path, g: dict) -> Path:
    """资源总览：左 = 地形区（晕渲叠色），右 = 主导资源与采场 / 点位（主岛放大）。"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from .output import _hillshade_rgb
    J = g["json"]
    R = g["resources"]
    res_m = J["raster"]["res_m"]
    H, W = g["height"].shape
    x0, y0 = J["raster"]["origin_km"]
    ext = [x0, x0 + W * res_m / 1000.0, y0 - H * res_m / 1000.0, y0]
    h = g["height"]
    shade = _hillshade_rgb(h, res_m, float(np.nanmin(h)), float(np.nanmax(h)), "gray")
    land = g["island_id"] >= 0
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    zp = np.array(ZONE_PALETTE, dtype=float) / 255.0
    zc = zp[g["terrain_zone"]]
    axes[0].imshow(np.where(land[..., None], 0.5 * shade + 0.5 * zc, shade), extent=ext, origin="upper", interpolation="nearest")
    zs = R["zones"]["share"]
    axes[0].legend(handles=[Patch(color=tuple(zp[i]), label=f"{ZONE_NAMES[i]} {zs.get(ZONE_NAMES[i], 0) * 100:.0f}%") for i in range(1, len(ZONE_NAMES))],
                   loc="upper left", fontsize=8, framealpha=0.75)
    axes[0].set_title("地形区（山区 / 丘陵 / 台地 / 河谷）", fontsize=10)
    # 右：主岛外框
    ii, jj = np.where(g["island_id"] == 0)
    pad = 5
    a0, a1, b0, b1 = max(0, ii.min() - pad), min(H, ii.max() + pad + 1), max(0, jj.min() - pad), min(W, jj.max() + pad + 1)
    sub = (slice(a0, a1), slice(b0, b1))
    rp = np.array(RES_PALETTE, dtype=float) / 255.0
    rr = g["resource"][sub]
    base = shade[sub].copy()
    pts_codes = [RES_INDEX[k] for k in ("spring", "cave", "hotspring", "saltspring", "fossil")]
    faint = np.isin(rr, [RES_INDEX["timber"], RES_INDEX["stone"]])        # 林木、石料铺得最广：淡淡一层，不盖住别的
    patch = (rr > 0) & ~np.isin(rr, pts_codes) & ~faint
    base[faint] = 0.7 * base[faint] + 0.3 * rp[rr[faint]]
    base[patch] = 0.25 * base[patch] + 0.75 * rp[rr[patch]]
    base[g["river"][sub] > 0] = (0.15, 0.35, 0.85)
    base[g["lake"][sub]] = (0.12, 0.25, 0.7)
    sext = [x0 + b0 * res_m / 1000.0, x0 + b1 * res_m / 1000.0, y0 - a1 * res_m / 1000.0, y0 - a0 * res_m / 1000.0]
    axes[1].imshow(base, extent=sext, origin="upper", interpolation="nearest")
    wmarks = {"ore": ("s", 26), "sulfur": ("X", 30), "salt": ("H", 34), "placer": ("P", 30), "stone": ("D", 14), "clay": (".", 20), "gravel": (".", 20)}
    for key, (mk_, sz) in wmarks.items():
        pts = [w["km"] for w in R["workings"] if w["kind"] == key and w["island"] == 0]
        if pts:
            p = np.array(pts)
            axes[1].scatter(p[:, 0], p[:, 1], marker=mk_, s=sz, c=[tuple(rp[RES_INDEX[key]])], edgecolors="black", linewidths=0.4, zorder=6)
    pmarks = {"spring": ("o", 14), "cave": ("^", 34), "hotspring": ("*", 60), "saltspring": ("p", 40), "fossil": ("v", 24)}
    for key, (mk_, sz) in pmarks.items():
        pts = [d["km"] for d in R["deposits"] if d["kind"] == key and d["island"] == 0]
        if pts:
            p = np.array(pts)
            axes[1].scatter(p[:, 0], p[:, 1], marker=mk_, s=sz, c=[tuple(rp[RES_INDEX[key]])], edgecolors="black", linewidths=0.5, zorder=5)
    handles = [Patch(color=tuple(rp[RES_INDEX[k[0]]]), label=f"{k[1]} {R['counts'].get(k[1], 0)}") for k in RES_KINDS if k[3] != "point" and R["counts"].get(k[1])]
    handles += [Line2D([], [], marker=wmarks[k][0], ls="", color=tuple(rp[RES_INDEX[k]]), markeredgecolor="black", label=f"{WORK_ZH[k]} {R['workings_counts'].get(WORK_ZH[k], 0)}")
                for k in wmarks if R["workings_counts"].get(WORK_ZH[k])]
    handles += [Line2D([], [], marker=pmarks[k][0], ls="", color=tuple(rp[RES_INDEX[k]]), markeredgecolor="black", label=f"{RES_NAMES[RES_INDEX[k]]} {R['counts'].get(RES_NAMES[RES_INDEX[k]], 0)}")
                for k in pmarks if R["counts"].get(RES_NAMES[RES_INDEX[k]])]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.75)
    geo = R["geology"]
    axes[1].set_title(f"主岛资源（底色 = 主导类，符号 = 采场与点；图例为全群计数）· 最近板块边界：{geo['boundary_type']}（核 {geo['boundary_kernel']:.2f}）"
                      f"{' · 叠层' if geo['layered'] else ''} · 石料岩性按剥蚀深浅：{' / '.join(geo['layers']['names'])}", fontsize=10)
    for ax in axes:
        ax.set_xlabel("km 东")
        ax.set_aspect("equal")
    axes[0].set_ylabel("km 北")
    m = J["meta"]
    fig.suptitle(f"岛群 #{m['node']} 地形区与资源 [seed {m['seed']} · {m['res_m']:.0f} m/格]", fontsize=11)
    p = out / "preview_resources.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return p
