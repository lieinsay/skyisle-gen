"""水利（P6，Zhouzhu PLAN-LAND L13；DESIGN-NOTES 四点三十七 / 四点三十九 / 四点四十）：谷口的渠和塘、湿地排成圩田。水网是人挖的，放在聚落层。

算法在 C++ 核心里（core/src/island/waterworks.cpp，Python 参考版删于 2026-09-30，tag python-reference-final）；
这里只剩代码 → 中文的表、水利的备注与人工改造的类表（decode.py、settle.landuse_summary、check 用）。下面是算法的说明
（四点四十起分级：邑级大堰由 big_plan 定灌区；village_works = false（默认）时村级的渠首 / 渠 / 塘 / 废水利不出，圩区照旧）：

- **圩田**（polder_plan，farmland.fill_cultivated 在好地先占的第一遍之后调）：好地占完了才去开湿地——
  湿地片（地表 = 湿地，8 连通、不跨岛，≥ polder_patch_min_km2）周围 polder_ring_km 内的平地（宜垦、坡 < polder_flat_deg）
  在第一遍里已经种了 ≥ polder_pressure_min 成（人口压力到了这片湿地边上），才排干围成圩田；没到的照旧是芦苇荡。
  格子按空岛湿地的大小缩：纵浦横塘 polder_spacing_km 一格（江南宋以后分圩，一圩几百亩到一两千亩；这里 600 m 一格 ≈ 540 亩），
  网格锚在湿地的出水口（片里汇流最大的格），纵浦南北、横塘东西；一格里的湿地够 polder_block_min_km2 才围成一圩，边角留作荡。
  圩田算已垦、**在额度之内**：第二遍好地先占时圩田的格先占（在大岛保底之后），额度照旧 = 行星层的可耕率、人口照旧 = ⑨，
  挤掉的是第一遍最外一层的地（开圩是因为没有更好的地了）。
- **谷口的渠**（build_waterworks）：渠首设在常年河（和汇水 ≥ works_stream_min_km2 的大溪涧：湿季引水，渠首配堰塘）上汇水 ≥ works_src_min_km2 的格，
  从高往低挑（水排到崖边就掉下去了：在出山口、平原上游就截住分走）。从渠首按「最省工的路」走出去（Dijkstra：一步的工 = 步长 × (1 + canal_slope_cost × 坡²)，
  横穿坡地费工 → 渠顺等高线走），渠水面按 canal_grad_m_per_km 往下降，地面最多高出渠水面 canal_cut_m（挖得穿），走出 canal_reach_km 为止；
  不过常年河、湖、崖缘、湿地，不跨岛。P6 第一版：一块田有五成以上的格在渠水面以下、走得到，就整块算这处渠首的灌区；
  P6b 起按格算（管它的那个村的田里渠水面以下、走得到的格），够 canal_min_cmd_km2 才修。
  渠 = 渠首到各块田的入口、再按 canal_lateral_km 的格点（以渠首为原点）铺进田里的路合成的树（最省工的路树，从渠首散开成扇）：
  从渠首出来的一段是干渠，分出去的是支渠；宽按所灌的田（每 km² canal_q_per_km2 m³/s，宽 5·Q^0.5 m）。
  渠首之间隔 ≥ village_head_sep_km（P6b 起一村一堰；P6 第一版一处渠首灌几个村，隔 3 km），试过的候选 canal_try_sep_km 内不再试。
- **塘**：每个村一口——灌区里的村（有渠或圩田）是村塘（接渠水），不在灌区、村在高山 / 山地 / 丘陵的是山塘（陂塘：在村上坡的沟里筑坝蓄雨水，
  按所灌的田定大小），其余是村塘（接雨水）；圩里各留一口圩塘（圩的 polder_pond_frac）；季节性的渠首配一口堰塘（蓄湿季的水）。
- **闸**：渠首闸（每处渠首）、圩闸（每圩一座，在圩堤上离出水口最近的地方）、排水闸（每片圩田的出水口，接排水渠到河）。
- **有水利就有人维护**（P6b，Zhouzhu PLAN-LAND L30，DESIGN-NOTES 四点三十九）：每处渠首、渠、塘、闸、圩都记管它的村（village），
  都在那个村走得到的范围（manage_walk_km，直线）内。次序是「田（含圩田）→ 村址（圩田按 polder_village_blocks² 圩一组成田，走不到现有的村就在圩上落圩村，
  settle.polder_villages）→ 水利」：渠首一村一堰——只灌管它的村自己的旱地田，Dijkstra 只走那个村走得到的格（渠走不出去 = 截短），
  渠首之间隔 village_head_sep_km；圩、圩塘、圩闸归种那组圩田的村，纵浦横塘按两旁的圩分段归各自的村，排水闸与排水渠归出水口最近、走得到的村
  （排水渠走出去就截短，没有走得到的村就不修）。废村（P5 的 ruins）旁另挑没人管的废塘（每个废村一口）与废渠首、废渠（灌过它的撂荒田，一个废村至多一处）：
  abandoned = True、abandoned_years = 撤空了几年、ruin = 废村号、village = None。
- **原始地表与人工改造**（P6b，L31，landuse_layers）：landcover_natural = 没有人以前的地表（hydro 画田之前的 cover_natural，圩田那格原是湿地）；
  landuse = 人工改造码（LANDUSE_CLASSES）。

只用确定的次序（稳定排序、按格号 / 田号破平局；浮点累加一律顺序加）。
"""
from __future__ import annotations

# 代码 → 中文（decode.py 用）
CANAL_ZH = {"main": "干渠", "branch": "支渠", "drain": "排水渠", "ns": "纵浦", "ew": "横塘"}
POND_ZH = {"weir": "堰塘", "village": "村塘", "hill": "山塘", "polder": "圩塘"}
SLUICE_ZH = {"head": "渠首闸", "polder": "圩闸", "outlet": "排水闸"}
SOURCE_ZH = {"river": "常年河", "stream": "季节性溪涧"}
BIG_ZH = {"weir": "大堰"}                       # 四点四十：邑级大堰（都江堰级）
MAINT_ZH = {"yi": "邑"}                         # 管它的：邑（官府设堰官、岁修按用水的村出工）
WORKS_NOTE = ("水利（P6）：渠首（heads）→ 渠（canals：干渠 / 支渠从渠首出来，纵浦 / 横塘是圩田的格子，排水渠从圩田的出水口接到河；"
              "pts = [行, 列] 群栅格坐标，格心 = 整数 + 0.5，圩田的纵浦横塘走在格边上 = 整数）；塘（ponds：村塘 / 山塘 / 圩塘 / 堰塘，area_m2 水面）；"
              "闸（sluices：渠首闸 / 圩闸 / 排水闸）；圩（polders：一格一圩，terrain.npz 的 polder_id 是圩号；dike_km = 圩堤长）与圩田片（polder_patches）。"
              "圩田算已垦、在额度之内（人口照旧 = ⑨）；没排干的湿地照旧是芦苇荡。"
              "P6b：每处都记管它的村（village，都在 summary.manage_walk_km 以内；渠首一村一堰、只灌那个村的田）；"
              "abandoned = true 的是废村旁没人管的废渠首 / 废渠 / 废塘 / 废渠首闸（abandoned_years = 撤空了几年，ruin = 废村号，village = null），不进 summary 的数（另见 summary.abandoned）。"
              "水利分级（四点四十，用户 09-30 定）：岛群层只出邑级的——big_works 是大堰（都江堰级：常年河上的堰，干渠 / 支渠带 work = 堰号，"
              "maintainer = 邑，turnouts 是每个用水的村的分水口，served_km2 = 灌区里在种的地）与圩区；村级的（村的渠首与渠、村塘 / 山塘 / 堰塘、圩塘、废塘 / 废渠）"
              "归营建器按风格修，summary.village_works = false 时这里是空的。")


# ---------------------------------------------------------------- 原始地表与人工改造（P6b，Zhouzhu PLAN-LAND L31）
LANDUSE_CLASSES = ["没动过（原始地貌）", "开垦的田", "梯田", "渠灌田", "圩田（原为湿地）", "撂荒（在往回长）", "樵牧（村周砍林）", "采场（挖开的地）"]
LANDUSE_NOTE = ("人工改造（terrain.npz 的 landuse，P6b）：0 没动过；1 开垦的田（在种的旱田，原来是什么见 landcover_natural）；2 梯田（坡地修成台阶）；"
                "3 渠灌田（邑级大堰的渠灌得到的在种的地；village_works = true 时也含村的渠灌的田）；4 圩田（原为湿地，排干围圩）；5 撂荒（开过又撂下、按年头在往回长，含废村的田）；"
                "6 樵牧（村周的林子砍成草场 / 薪炭林）；7 采场（挖开的坑、采石场）。landcover_natural = 没有人以前的地表（上等地画田之前的地表，河、湖照旧）")

