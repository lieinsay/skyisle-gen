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

// wet：聚落层动手之前的湿地；take1：第一遍好地先占的已垦（扁平）；n_avail：额度里还能给圩田的格数
PolderPlan polder_plan(const Group& g, const Config& c, const Mask& wet, const std::vector<uint8_t>& take1, int64_t n_avail);

struct WorksField {                     // 田块（settle 的 fields，id = 下标 + 1）
    int64_t cells = 0;
    double area_km2 = 0;                // 已按 round 3 舍好的
};
struct WorksVillage {                   // 村（不含散户），按村号
    int id = 0, ci = 0, cj = 0, field = 0;
    int64_t households = 0;             // 农户 + 镇的非农户
};

// 渠、塘、闸、圩的记录（settlements.json 的 waterworks，代码形）；塘写 15、闸写 16 进 sraster
Json build_waterworks(const Group& g, const Config& c, const std::vector<WorksField>& fields, const GridI& fields_raster,
                      const std::vector<WorksVillage>& villages, Grid<uint8_t>& sraster, const PolderPlan& plan);

}  // namespace skyisle::island
