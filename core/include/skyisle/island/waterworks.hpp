// 水利（P6，Zhouzhu PLAN-LAND L13；skyisle_gen/island/waterworks.py 同式，DESIGN-NOTES 四点三十七）：谷口的渠和塘、湿地排成圩田。
// polder_plan：好地先占第一遍之后，挑人口压力到了的湿地排干、分成一格一圩（farmland.cpp 调，第二遍圩田的格先占、在额度之内）；
// build_waterworks：渠首 → 渠（Dijkstra 最省工的路合成的树）、村塘 / 山塘 / 圩塘 / 堰塘、渠首闸 / 圩闸 / 排水闸、纵浦横塘与排水渠（settle.cpp 在水设施之后调）。
// 字符串是代码（main / branch / drain / ns / ew，weir / village / hill / polder，head / polder / outlet，river / stream），中文在前端 decode.py 映射。
#pragma once

#include <cstdint>
#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

struct PolderBlock {
    int bi = 0, bj = 0;                 // 格子号（相对锚点，向下取整）
    std::vector<int32_t> cells;         // 扁平下标，升序
};

struct PolderPatch {                    // 一片排干的湿地
    int label = 0, island = 0;
    int32_t outlet = 0;                 // 出水口：片里汇流最大的格（平局格号小）
    int io = 0, jo = 0, spacing = 1;    // 网格锚点与间距（格）
    double pressure = 0;                // 周围平地第一遍种了几成
    int64_t wet_cells = 0;
    std::vector<PolderBlock> blocks;    // 围成圩的格子（按 (bi, bj)）
    std::vector<int> ids;               // 各圩的号
};

struct PolderPlan {
    std::vector<PolderPatch> patches;
    std::vector<int32_t> cells;         // 圩田的格：按片、按圩、格号升序
    GridI polder_id;                    // 圩号（0 = 不是）
    int64_t wetland_cells = 0;
};

// 邑级大堰（都江堰级；用户 09-30 定，Zhouzhu PLAN-LAND L32–L34，DESIGN-NOTES 四点四十）：水利先后看谁修——邑级的是因，在田和村之前；
// 村级的是果，在村之后、归营建器按风格修。big_plan 在好地先占之前（fill_cultivated 调）：常年河上挑渠首，按最省工的路走出干渠，
// 灌区 = 坡缓、宜垦、在渠水面以下、走得到的格，按水量截；够大（big_min_cmd_km2 且 ≥ 额度 big_min_quota_frac）才修，一个水系至多一处。
// 灌区的适宜度加成，好地先占先占它 → 人往灌区聚。渠网、分水口、用水的村在村落好以后（build_waterworks）按同一棵最省工的路树修到在种的格
struct BigWork {
    int island = 0;
    int32_t head = 0;                   // 堰（渠首）的格：常年河上
    int32_t outlet = 0;                 // 这条河顺流而下到哪（同一水系只修一处）
    double z0 = 0;                      // 渠水面 = 河床 + 水深 + 堰抬的
    double acc = 0;                     // 渠首的汇水 km²（float32 的值）
    double water_km2 = 0;               // 水够灌的地 = 汇水 × 年雨 × 径流系数 / 一年灌一遍要的水
    int r0 = 0, r1 = 0, c0 = 0, c1 = 0; // Dijkstra 的窗
    std::vector<int32_t> par;           // 窗内每格的父格（全局格号，−1 = 没走到）
    std::vector<int32_t> rc;            // 灌区的格（按出堆的次序；到水量上限就停）
};
struct BigPlan {
    std::vector<BigWork> works;
};
// pit：不开田的格（采场、岩类赋存）；n_quota：已垦的额度（格）
BigPlan big_plan(const Group& g, const Config& c, const std::vector<uint8_t>& pit, int64_t n_quota);

// wet：聚落层动手之前的湿地；take1：第一遍好地先占的已垦（扁平）；n_avail：额度里还能给圩田的格数；
// pit：不开田的格（扁平：资源层的采场——湿地里的盐井——与岩类赋存；P6b 起不围进圩里，湿地片与出水口照旧按整片算）
PolderPlan polder_plan(const Group& g, const Config& c, const Mask& wet, const std::vector<uint8_t>& take1, int64_t n_avail,
                       const std::vector<uint8_t>& pit);

struct WorksField {                     // 田块（settle 的 fields，id = 下标 + 1）
    int64_t cells = 0;
    double area_km2 = 0;                // 已按 round 3 舍好的
};
struct WorksVillage {                   // 村（不含散户），按村号
    int id = 0, island = 0, ci = 0, cj = 0, field = 0;
    int64_t households = 0;             // 农户 + 镇的非农户
    std::vector<int> polder_fields;     // P6b：挂在这个村上的圩田（田号，按挂上的先后：户多的组先）
    bool polder = false;                // P6b：圩村（落在圩田上）
};
struct WorksRuin {                      // P6b：废村（按废村号）与它的撂荒田（废渠灌过的地）
    int id = 0, island = 0, ci = 0, cj = 0, years = 0;
    int64_t households_before = 0;
    double fallow_km2 = 0;              // 已按 round 3 舍好的
    std::vector<int32_t> cells;         // 撂荒田的格（扁平下标）
};

// 渠、塘、闸、圩的记录（settlements.json 的 waterworks，代码形）；塘写 15、闸写 16、废塘与废渠首闸写 19 进 sraster。
// P6b：每处记管它的村（village），都在 works.manage_walk_km 以内；渠首一村一堰、只灌那个村的旱地田（按格算灌区）；
// 废村旁另挑废渠首 / 废渠 / 废塘（abandoned）。cmd_cells 回传渠灌得到的在种的格（landuse 的渠灌田，升序）。
// 水利分级（四点四十）：big 的大堰先修（邑管；渠网只修到在种的格、每个用水的村一个分水口）；works.village_works = 0 时村级的
// （村的渠首与渠、村塘 / 山塘 / 堰塘、圩塘、废塘 / 废渠）不在岛群层出，归营建器
Json build_waterworks(const Group& g, const Config& c, const std::vector<WorksField>& fields, const GridI& fields_raster,
                      const std::vector<WorksVillage>& villages, Grid<uint8_t>& sraster, const PolderPlan& plan,
                      const std::vector<WorksRuin>& ruins, const BigPlan& big, std::vector<int32_t>& cmd_cells);

// P6b（L31）：没有人以前的地表（cover_natural，河道 / 湖照现状）与人工改造码（waterworks.py 的 LANDUSE_CLASSES：0 没动过 1 开垦的田 2 梯田 3 渠灌田
// 4 圩田 5 撂荒 6 樵牧 7 采场）→ g.landcover_natural / g.landuse
void landuse_layers(Group& g, const std::vector<int32_t>& cmd_cells);

}  // namespace skyisle::island
