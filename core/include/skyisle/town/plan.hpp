// 营建方案（docs/PLAN-TOWN.md 第五、七节）：给一块 Site、一种风格、一个请求（规模、户数、上游锚点），营建出路、桥、宅院、每栋房、塘 / 场 / 井 / 树、户，
// 外加 TP-* 校验与形态指标。plan_site 是编排层：唯一持有「正在长的方案」，下面各层按值拿参数。
#pragma once

#include <map>
#include <string>
#include <vector>

#include "skyisle/grid.hpp"
#include "skyisle/town/geom.hpp"
#include "skyisle/town/site.hpp"
#include "skyisle/town/style.hpp"

namespace skyisle::town {

// 占地栅格的码（plan.occ）
// OCC_OPEN：公地、广场、林带、坑——不盖房、路能过；OCC_MOAT：環濠的水，只在桥上过
enum Occ : uint8_t { OCC_FREE = 0, OCC_ROAD = 1, OCC_PLOT = 2, OCC_BUILDING = 3, OCC_POND = 4, OCC_THRESH = 5, OCC_LANDING = 6, OCC_BRIDGE = 7,
                     OCC_OPEN = 8, OCC_MOAT = 9 };
enum HouseholdKind : int { HH_FARM = 0, HH_MARKET = 1, HH_SPECIAL = 2 };

struct Road {
    std::vector<V2> line;
    int cls = 3;             // RoadClass
    double width_m = 3.0;
};

struct Bridge {
    V2 a, b;                 // 两头（岸上）
    double width_m = 3.0;
    int road = -1;
};

struct Building {
    std::string func, role, name, roof, material;
    int compound = -1;       // −1 = 不在宅院里（路边小庙、货棚）
    Obb box;                 // facing = 正面（门、檐口）朝的方向
    int storeys = 1;
    double eave_m = 3.0, pitch_deg = 30.0;
    double base_m = 0.0;     // 台基顶面高程（= 脚下地面的中位 + 台高）
    double cut_m3 = 0.0, fill_m3 = 0.0, max_cut_m = 0.0;
};

struct Compound {
    std::string kind;        // house / public
    std::string tmpl, tmpl_name, func, name;
    Obb plot;                // 地块（facing = 宅院朝向）
    int access_side = 0;     // 路在地块的哪一边（Side）
    V2 access;               // 接路的点（路中线上）
    V2 gate;                 // 门的位置（地块边上）
    double gate_bearing = 0.0;   // 门朝外的方向
    bool walled = true;
    double base_m = 0.0, max_cut_m = 0.0;   // 地块中位高程、地块里离它最远的高差（> 风格的 max_cut_m 即台地）
    std::vector<int> households, buildings;
};

struct Feature {
    // pond 塘 / threshing 场 / well 井 / tree 树 / landing 泊场 / grove 林带（林盘竹林、屋敷林、水口林）/ garden 园（croft）/ strip 条地（林地排村）/
    // green 公地 / square 广场 / pit 地坑院的坑 / steps 河埠头 / moat 環濠（poly = 闭合中线，r = 宽）/ channel 水圳（poly = 折线，r = 宽）/
    // arch 牌坊（p、facing、r = 跨度）/ furlong 条田块（poly、facing = 条的走向、r = 条宽）
    std::string kind;
    std::string func, name;
    std::vector<V2> poly;    // 外形（面）或中线（moat、channel）
    V2 p;
    double r = 0.0;          // 井、树的半径；moat / channel 的宽；arch 的跨度；furlong 的条宽
    double facing = 0.0;
    double z = 0.0;          // 塘的水面 / 场的地面
    int compound = -1;       // 属于哪个宅院（园、坑）
};

struct Household {
    int id = 0, kind = HH_FARM, parent = -1, compound = -1;
};

struct Check {
    std::string id;
    bool hard = true, ok = true;
    std::string msg;
};

// 出村大路的去向（相邻聚落、治所、泊场、桥头……）
struct ExitTarget {
    double bearing = 0.0;    // 从上游点位看的方位角
    double dist_m = 0.0;
    double weight = 1.0;
    std::string kind;
};

struct PlanRequest {
    std::string scale = "village";   // compound / hamlet / village（town / city / special 在第四、六步）
    int hh_farm = 0, hh_market = 0, hh_special = 0;
    uint64_t seed = 0;
    V2 anchor;                       // 上游点位（平面坐标）
    std::vector<V2> landings;        // 上游的泊场点（属于本聚落的在前）
    std::vector<ExitTarget> exits;
    double wind_from = NaN;          // 冬季风从哪来（方位角，弧度）
    bool want_landing = true;
    std::string force_operator;      // 非空时不按权重挑
};

struct Plan {
    std::string op;                  // 用了哪个形态算子
    std::vector<std::string> ops_fit;   // 风格的算子里这块地能用的（滨水要河、等高线要坡……）
    V2 center;
    double facing = 0.0, radius = 0.0;
    std::vector<Road> roads;
    std::vector<Bridge> bridges;
    std::vector<Compound> compounds;
    std::vector<Building> buildings;
    std::vector<Feature> features;
    std::vector<Household> households;
    std::map<std::string, double> metrics;
    std::vector<Check> checks;
    Grid<uint8_t> occ;               // Occ 码
    std::map<std::string, double> timing;
};

Plan plan_site(const Site& s, const Style& st, const PlanRequest& req);

}  // namespace skyisle::town
