"""宜垦 / 已垦 / 撂荒、定居门槛、没人住 ≠ 没人用、荒地归谁（P5，Zhouzhu PLAN-LAND L7 / L10 / L21 / L29；DESIGN-NOTES 四点三十六）。

算法在 C++ 核心里（core/src/island/farmland.cpp 与 settle.cpp，Python 参考版删于 2026-09-30，tag python-reference-final）；
这里只剩代码 → 中文的表与备注的拼法（decode.py 用）。下面是算法的说明：

- **宜垦**（水系里算）：地本身能不能种——坡 < cultivable_slope_max_deg、最暖的月份（年均温 + 半个季节温差）≥ cultivable_summer_min_c、
  土层 ≥ cultivable_soil_min、够湿（湿度 ≥ cultivable_wet_min）或引得到水（离河湖 water_near_cells 内），不是崖缘、水面、湿地；
  坡 ≥ terrace_slope_deg 的要修梯田。→ g["cultivable"]（0 / 1 / 2）。
  行星层的可耕率照旧按适宜度取出「上等地」（hydro 的 g["arable"]，额度内最好的宜垦地）：资源层照旧把它当田土避开
  （岩类赋存不上这片地、林场不算这片，赋存场与点位逐字节不变），地表在这时照旧画成可耕地 / 梯田；
  g["cover_natural"] 留着它原本的地表，g["suit"] 留着适宜度给聚落层排先后。上等地一定宜垦（极冷、极陡的群也一样：行星层说这么多地在种）。
- **已垦**（聚落里算）：人口从最好的宜垦地往外填（好地先占）——宜垦地按适宜度排先后（与 hydro 取上等地同一个数），
  填到行星层的额度为止（额度 = 上等地的格数，人口口径不变）。远近不排先后：群内的短渡飞船半小时内就到
  （试过「适宜度 × 跨岛 0.95 × exp(−离主岛的岸距 / 50 km)」：小岛的地本来就比主岛陡、冷、离河远，再一打折额度全落在主岛上，三群只剩主岛有村，太过）；
  **定居门槛**：宜垦地的连通片（不跨岛）要有水（settle_water_km 内有河 / 湖 / 溪涧 / 泉，或年降水 ≥ settle_rain_mm 能蓄雨）、
  够一个像样的村（片的面积 ≥ settle_min_hh 户的地）、能落船（landing_reach_cells 内有坡 ≤ landing_slope_max_deg 的平地）；
  头一轮填完分到的地不够 settle_min_hh 户的片整片让出来（不再按比例连续撒户），额度按次序补给留下的片——一轮就定（留下的片只会变多）。
  **大岛保底**（用户 09-29 定）：主岛以外 ≥ island_floor_km2（30 km²）的岛，有合门槛的宜垦片的，先在它最好的那片地上分一个村——
  从那片里适宜度最高的格起、按适宜度往外长成连成一块的 island_floor_hh（20）户的地（一块田配一个村），户从额度里出（从主岛匀过去）；其余照好地先占填。
- **撂荒**：留下的片里接着往外的一层（离在种的田 ≤ fallow_ring_cells 格）取已垦的 fallow_frac，每个连通块一个年头（1 … fallow_years_max）；
  **废村**：让出来的片里头一轮分得最多的几片（≥ ruin_min_hh 户，至多 ruin_max 个）——人撤走了，头一轮分到的那些地是撂荒的田，年头 = 撤空了几年。
  撂荒地的地表按年头：≤ fallow_grass_years 年草坡，≤ fallow_shrub_years 年灌丛（原本是林 / 灌丛的），再久回到原本的地表（小树林）。
- **没人住 ≠ 没人用**：光秃小岛（没有村和散户）上的浮石采石村、矿村、窑村改成「工棚」（白天来、晚上走，人算在最近的村）；
  烧炭营是季节住；矿镇、盐井村、温泉地照旧常住。没人常住的岛里按合理挑少数：放牧岛（只放牲口）/ 夏牧（夏天的牧棚，季节住）、
  邑治附近的庙、墓岛。群边高处的烽火台 P7 起归中转站（market：瞭望烽火与关卡、过夜 / 候风 / 避风 / 换船一起挑，统一、不重复）。
- **荒地归谁**（L29，用户定：荒地有主、领照开垦）：每岛记一个主——有农户（村或散户）的岛：未垦的宜垦地归就近的村（村里的大户 / 族产）；
  没人住但有人用的岛（工棚、季节住、放牧……）：归用它的那个村的大户；只有中转站（驿、关、烽）的岛：官用；别的荒岛、废村的地：官荒，归邑（绝户田入官）。

只用确定的次序（稳定排序、按格号 / 岛号 / 片号破平局）。
"""
from __future__ import annotations

# C++ 的代码 → 中文（decode.py 用；说明文字由下面几个函数拼）
SPECIAL_OCC_ZH = {"resident": "常住", "workcamp": "工棚", "seasonal": "季节住"}
USE_ZH = {"graze": "放牧", "shieling": "夏牧", "beacon": "烽火台", "shrine": "庙", "tomb": "墓岛"}
USE_OCC_ZH = {"livestock": "只放牲口", "seasonal": "季节住", "rotation": "轮班", "incense": "香火", "none": "无人"}
STATUS_ZH = {"resident": "常住", "seasonal": "季节住", "used": "有人用", "empty": "荒岛"}
OWNER_ZH = {"village": "村", "magnate": "大户", "office": "官用", "crown": "官荒"}


def ruin_note(years: int, hh: int, km2: float) -> str:
    return f"撤空了 {years} 年；原来约 {hh} 户，旁边的田撂荒（{km2:.2f} km²），地收归官府（绝户田入官）"


def use_note(code: str, a: float, b: float | None = None) -> str:
    if code == "graze":
        return f"只放牲口，隔些日子来看一回（草场 {a:.1f} km²，离村 {b:.1f} km）"
    if code == "shieling":
        return f"夏天赶上来放牧，搭着牧棚住一季（草场 {a:.1f} km²，离村 {b:.1f} km）"
    if code == "beacon":
        return f"群边高处的瞭望与烽火，离主岛 {a:.1f} km；轮班守，不常住"
    if code == "shrine":
        return f"邑治附近的小岛上一座庙，离邑治 {a:.1f} km；逢年过节有人上岛"
    if code == "tomb":
        return f"葬地，离邑治 {a:.1f} km；清明上坟"
    raise ValueError(code)

