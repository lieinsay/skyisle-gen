// 第三层（岛群生成器）的输入与群状态（skyisle_gen/island 的 inp / g 的 C++ 形）。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/grid.hpp"
#include "skyisle/island/strat.hpp"
#include "skyisle/json.hpp"

namespace skyisle::island {

constexpr uint64_t ISLAND_STREAM = 21;   // 与十步管线的流号 1–10 错开
enum AgeKind { YOUNG = 0, MID = 1, OLD = 2 };
const char* age_name(AgeKind k);          // "young" / "mid" / "old"

// 行星层 ⑥ 的一条邻边（本群 ↔ 邻群；market.node_routes / planet::apply_routes，P7 的中转站读它）
struct RouteEdge {
    int64_t node = 0;                     // 邻群
    double bearing = 0;                   // 方位（弧度，0 = 北、顺时针；sphere.initial_bearing）
    double days = 0;                      // ③ 的离开几天
    double cost_out = 0, cost_in = 0;     // ⑥ 的有向成本（天）：本群 → 邻群 / 邻群 → 本群
    double flow_out = 0, flow_in = 0;     // ⑥ 的有向流量（routes.npz 存 float32，这里也是 float32 的值）
    bool hub = false;                     // 邻群是不是 ⑥ 的枢纽
};

// 本群的行星层标量（_node_inputs）
struct NodeInputs {
    int64_t node = 0;
    uint64_t seed = 0;
    double lat = 0, lon = 0, area_km2 = 0, main_area_km2 = 0, height_m = 0, age = 0;
    bool layered = false;
    double keel_clearance_m = 300.0, area_median_km2 = 1.0;
    // 水系用
    double precip = 0, temp_sea = 0, lapse_c_per_km = 6.0, arable_frac = 0, river_size = 0;
    double precip_mm_ref = 4000.0;   // 相对降水 1 的毫米数（④ 的 precip_mm_ref，A3 起线性换算）；arable_frac 是降水线之后的（④ 的 arable_frac_eff）
    std::vector<double> precip_share;   // ④ 各季降水占全年的份额（A2 起；空 = 旧产物，四季降水按带界摆动取样）
    bool has_river = false;
    // 气候用（④ 的岛上年均值）
    double temp = 0, storm = 0, window = 0, season_range = 0, season_range_sea = 0, temp_winter = 0, temp_summer = 0;
    // 聚落用：人口（⑨ 的 pop；没有 ⑨ 时 NaN → 可耕地 × people_per_arable_km2）与本群是不是某邦的都
    double pop = NaN, people_per_arable_km2 = 100.0;
    bool is_capital = false, reformer = false;
    int64_t state = -1;
    double state_pop = 0;
    // P7：⑥ 的邻边（按 ③ 的边号升序）与本群是不是 ⑥ 的枢纽（岛群级的中转岛）；没有 ⑥ 时空着（只挑群内的烽火台）
    bool hub = false;
    std::vector<RouteEdge> routes;
};

// 历法（climate.calendar）
struct Calendar {
    int seasons = 4, months_per_season = 3;
    double days_per_month = 28.0, days_per_season = 84.0, year_days = 336.0, offset = 126.0;   // offset = day_offset_solstice_n
};

// 行星层的网格与全体群（板块走向、势力范围、局地风）
struct PlanetView {
    double radius_km = 6371.0;
    double year_s = 336.0 * 24.0 * 3600.0;
    LatLonGrid plate_grid;
    std::vector<double> plate_K;        // boundary_kernel（float64）
    std::vector<int32_t> plate_btype;
    std::vector<double> plate_lats, plate_lons;
    std::vector<double> isl_lat, isl_lon, isl_area;   // 全体群：势力范围
    LatLonGrid wind_grid;
    std::vector<double> wind_u, wind_v;
    // ④ 的气候网格与带界（5.4 四季）、行星常数
    LatLonGrid cg_grid;
    std::vector<double> cg_precip, cg_storm, cg_window, cg_cont;   // cg_cont 空 = 没有陆地性场（取 0.1）
    std::vector<double> band_lons, band_eq_n, band_eq_s;           // band_local 的经度与 eq_n / eq_s 两行
    double tilt_deg = 34.0, tau_land = 8.0, tau_ocean = 110.0, alt_cont = 0.0;
    std::vector<double> season_shift;   // ④ 每季的带界位移（°，全球一个数，A2 起）；空 = 旧产物，按 band_shift_k 现算
    double band_shift_k = 0.35;         // [s04.climate] season_band_shift_k
    Calendar cal;
};

struct Limit {
    int64_t node;
    double dist_km, ux, uy, limit_km;
};

// 一座岛的局部栅格（island_shape）：X[i, j] = xs[j]，Y[i, j] = −xs[i]
struct Shape {
    Mask mask;
    GridD inside;
    GridD phi;                           // 岸线的连续场 f − τ（B4：零等值线就是岸线；与 mask 同号——掩膜内 > 0、外 < 0）
    std::vector<double> xs;
    double X(int, int j) const { return xs[j]; }
    double Y(int i, int) const { return -xs[i]; }
};

// 多核岛拟合后的一个核（P4；island.json 的 islands[].cores，局部栅格 km——前端加上局部栅格中心 gcx / gcy 换成群坐标）
struct CoreRec {
    double seed_x = 0, seed_y = 0, strength = 0;
    int64_t cells = 0;
    double peak = 0, load = 0, load_x = 0, load_y = 0, mean_above = 0;
};

struct IslandRec {
    int id = 0;
    std::vector<CoreRec> cores;          // 多核嵌合的岛才有（P4）
    double gcx = 0, gcy = 0;             // 局部栅格中心（群坐标 km）
    int64_t area_cells = 0;
    // surface / rim / peak / keel 都已含浮高 fl（整座平移的 δ，m；主岛 0，DESIGN-NOTES 四点二十八）
    double area_target = 0, cx = 0, cy = 0, surface = 0, relief_target = 0, rim = 0, peak = 0, keel = 0, age = 0, fl = 0;
    AgeKind kind = MID;
    StratRec strat;                      // 岩层（B2）：层厚与拟合的换算；层面在群栅格 strat_top / skel_top
    int r0 = 0, c0 = 0, m = 0;          // 局部栅格左上角在群栅格里的行列、边长（裁切后）
    // island.json 里的四舍五入值（Python 版 hydro 读的是它们）
    double rim_j = 0, keel_j = 0, age_j = 0;
    // 水系之后的 island.json 值（主岛台面校正过；资源、天气、聚落读它们）：finalize_islands 填
    double peak_j = 0, cliff_j = 0, area_j = 0, cx_j = 0, cy_j = 0;
    // 水系（build_hydro 填）
    bool hydro = false;
    int n_lakes = 0;
    int64_t lake_cells = 0;
    double max_flowacc = 0;
    bool has_perennial = false, has_stream = false;
    bool cap_ran = false;               // 谷收拢跑过（P4）
    int captures = 0;                   // 袭夺了几条小沟
};

struct Link {                         // 短渡（飞船航线）；P5 起没有索桥
    int a = 0, b = 0;
    double gap = 0, dh = 0;
    bool fallback = false;
};

struct TerritoryRec {
    int neighbours = 0;
    double gap_km = 3.0;
    bool constrained = false;
    double off_x = 0, off_y = 0, turn_deg = 0, stretch = 1.0, before = 0, after = 0;
    bool has_violation = false;
    double violation = 0;
};

struct MaskPos {
    Mask mask;
    int r0 = 0, c0 = 0;
};

// 河口表的一条（carve_channels 的 info["rivers"]，局部切片坐标已换成群栅格）
struct RiverRec {
    int mouth_r = 0, mouth_c = 0;
    double basin_km2 = 0, length_km = 0, discharge = 0, width = 0, depth = 0, waterfall = 0, incision = 0;
    int level = 0;
};

struct LinePt {
    double r, c, w;
    int lvl;
    double acc;
};
struct RiverLine {
    int island = 0;
    std::vector<LinePt> pts;   // 群栅格坐标（行 + 0.5、列 + 0.5）
};

// 地貌（B3，landforms.cpp）：一处特殊的山 / 地貌。kind 是 ASCII 代码（前端映射中文）；r、c 是群栅格的行列（格心 = 整数 + 0.5，同 LinePt），
// 地形阶段先按局部栅格记、贴图时换成群栅格；attr 是各类的量（尺寸、成因条件）
struct LandformRec {
    std::string kind;
    int island = 0;
    double r = 0, c = 0;
    Json attr = Json::obj();
};

// 主岛集水盆地（hydro._basins 的原始数）：河口 = 汇流 ≥ thr 的出口格，按扁平下标升序；排序、取前几个、四舍五入在前端
struct Basins {
    bool present = false;
    double total_km2 = 0, thr_km2 = 0;
    std::vector<std::array<double, 3>> mouths;   // 群栅格 [行, 列, 汇流 km²]
};

// ---------------------------------------------------------------- 资源的记录（resources.hpp）
struct Deposit {                      // 点与片（resources.json 的 deposits）
    int id = 0, kind = 0;
    std::string subtype;              // "" = None
    int island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0;            // km（round 3）
    double area_km2 = 0, elev_m = 0, slope_deg = 0;
    int zone = 0;
    std::string grade;
    std::string note;                 // 代码；"" = 无
    double note_arg = 0;
    bool cleared = false;
    bool has_area_before = false;     // 开垦前面积（clear_forest 记，只林木）
    double area_before = 0;
};

struct Occurrence {                   // 散的赋存区（occurrences）
    int id = 0, kind = 0;
    std::string subtype, note;
    int island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0, area_km2 = 0;
    double length_km = 0;             // C++ 的闭式主轴；前端按 numpy 的 cov / eigh 重算（格子在 cells）
    bool has_axis = false;
    double axis_deg = 0;
    double grade_peak = 0, grade_mean = 0;
    std::string grade;
    double elev_lo = 0, elev_hi = 0;
    int zone = 0;
    int n_workings = 0;
    std::vector<int32_t> cells;       // 群栅格扁平下标（add_occ 的 ii、jj 次序）
};

struct Working {                      // 采场（workings）
    int id = 0, kind = 0, occurrence = 0, island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0, grade = 0;
    std::vector<int> villages;
    int special = -1;                 // −1 = None
    std::string note;
};

struct Resources {
    std::vector<Deposit> deposits;
    std::vector<Occurrence> occ;
    std::vector<Working> works;
    double geo_ore = 0, fs_rate = 0, kernel_j = 0;
    int r_cells = 1;
    std::array<double, 7> thr{};   // 各类散的赋存区阈值（层序同 res_field；P3 起 7 层，第 7 层岩盐）
};

// ---------------------------------------------------------------- 四季与逐日天气的记录（climate.hpp）
struct SeasonRec {
    int index = 0;
    std::string name;                 // 季名代码（前端映射成中文）
    int d0 = 0, d1 = 0;
    double mid_day = 0;
    std::vector<int> months;
    double temp_c = 0, temp_sea_c = 0, precip_rel = 0, precip_mm = 0, precip_rate = 0, storm = 0, window = 0;
    double wu = 0, wv = 0, speed = 0, from_deg = 0, band_shift = 0, lat_sampled = 0;   // 均已按 Python 的 round 舍好
};

struct Climate {
    Calendar cal;
    std::string stype;                // four / two / rain / storm / none
    std::string type_code;            // four / two / rain / storm / none_warm / none_cold（前端映射季型中文）
    std::vector<std::string> names;
    double score_temp = 0, score_rain = 0, score_storm = 0;
    double a_temp = 0, a_temp_sea = 0, a_precip_rel = 0, a_precip_mm = 0, a_storm = 0, a_window = 0, a_range = 0, a_range_sea = 0,
           a_winter = 0, a_summer = 0, a_ref_h = 0;
    double cont_sea = 0, cont_isl = 0, tau_sea = 0, tau_isl = 0, A_sea = 0, A_isl = 0, lag_sea_days = 0, lag_isl_days = 0, k_shift = 0,
           band_amp = 0;
    std::vector<SeasonRec> seasons;
    double m_precip = 0, m_storm = 0, m_window = 0, m_temp = 0;
    double season_range_1 = 0;        // round(r_t, 1)：island.json 的 climate 摘要
};

struct Daily {                        // daily_curves：一年逐日的季节曲线
    std::vector<int32_t> day, season;
    std::vector<double> temp_c, precip_rel, precip_mm, storm, window, wind_u, wind_v;
};

struct SeasonParams {                 // weather.season_params
    double f_wet = 0, f_rain_days = 0, p_ww = 0, p_dw = 0, mean_wet_mm = 0, storm_frac = 0, p_calm = 0, wind_u = 0, wind_v = 0, speed = 0;
};

struct WeatherYear {                  // weather.simulate_year 的数组
    std::vector<int32_t> day, season, month, day_of_month, storm_event;
    std::vector<int8_t> type;
    std::vector<double> precip_mm, temp_c, wind_from_deg, wind_ms, temp_rim_c;
    std::vector<uint8_t> sailable, wet, snow;
};

// 群状态（Python 版的 g）
struct Group {
    NodeInputs inp;
    int H = 0, W = 0;
    double res_km = 0.1, x0 = 0, y0 = 0;    // x0 / y0：栅格左上角（km，未四舍五入）
    double origin_x = 0, origin_y = 0;       // island.json 的 origin_km（四舍五入到 3 位，水系按它取噪声坐标）
    GridD height;                            // 虚空 NaN
    Grid<int16_t> island_id;                 // −1 虚空
    Mask cliff;
    // B2 岩层：构造顶面、骨架顶面（最终高程口径，虚空 NaN）；lith 在水系之后按最终高程出（strat.hpp 的 Lith）
    GridD strat_top, skel_top;
    Grid<uint8_t> lith;
    // B3 地貌：记录；崖层 rockwall_m（该格所在岩壁的落差，0 = 不是岩壁）与朝向 rockwall_dir（0–15，×22.5°、0 = 朝北、顺时针；255 = 无）
    std::vector<LandformRec> landforms;
    GridD rockwall_m;
    Grid<uint8_t> rockwall_dir;
    // B4 亚格岸距（m，陆地为正、虚空为负，岸线 = 岛形连续场的零等值线）
    GridD coast_dist;
    std::vector<double> rims;
    std::vector<IslandRec> islands;
    std::vector<Link> links;
    std::vector<MaskPos> masks_pos;
    std::vector<Limit> lim;
    TerritoryRec territory;
    double axis = 0, kernel = 0;
    int btype = 0;
    AgeKind node_kind = MID;
    int n = 0;
    double sec_layout = 0, sec_paste = 0, sec_total = 0;

    // ---- 水系（build_hydro）
    bool has_hydro = false;
    GridD filled, route_h;
    GridD acc_km2;
    Grid<uint8_t> river, stream, lake, floodplain, landcover, arable;   // arable：行星层额度内的上等地（P5 起不是已垦，资源层避开它）
    // P5（farmland.cpp）：宜垦（0 / 1 / 2 要修梯田）、上等地画进地表之前原本的地表、适宜度（聚落层排先后）
    Grid<uint8_t> cultivable, cover_natural;
    std::vector<double> suit;
    GridD width_m, depth_m, cut_m, slope;
    GridD rain;                     // 局地年降水（mm，P4；陆地格，关掉局地雨时 = P_mm）
    GridD runoff;                   // 年径流深（mm，A5：局地雨 × Budyko 径流系数；陆地格）
    GridD runoff_acc;               // 上游的径流累计（mm·km²，= 年径流量 / 1000 m³）：大堰算水够灌多少地
    double runoff_ratio = 0;        // 全群陆地的径流 / 降水
    GridI recv_i, recv_j;
    std::vector<RiverLine> lines;
    std::vector<RiverRec> rivers;   // 主岛
    Basins basins;
    double P_mm = 0, river_thr = NaN, dz = 0, wind_u = 0, wind_v = 0, max_cut = 0;
    int n_falls = 0;
    double sec_hydro = 0;

    // ---- 资源（build_resources）
    bool has_resources = false;
    Grid<uint8_t> zone;                              // terrain_zone
    GridI patch_id;                                  // −1 = 无
    std::array<GridI, 7> occ_lab;                    // 各类散的赋存区号（−1 = 无），层序同 res_field
    std::array<Grid<uint8_t>, 7> res_field;          // 品位 × 255（金属矿、硫磺、砂金、黏土、砂砾、石料、岩盐）
    Grid<uint8_t> resource;                          // 主导类（显示用）
    Resources res;
    double sec_resources = 0;

    // ---- 四季、逐日曲线、天气（build_climate / daily_curves / build_weather）
    bool has_climate = false, has_weather = false;
    Climate clim;
    Daily daily;
    std::vector<SeasonParams> wparams;
    WeatherYear weather;
    int year = 0;
    double sec_climate = 0;

    // ---- 聚落（build_settlements）：记录直接成 JSON 的形（代码），栅格另存
    bool has_settle = false;
    Json settle;
    Grid<uint8_t> settle_raster;     // 1 田 / 2 梯田 / 3 村 / 4 散户 / 5 泊场 / 7 蓄水池 / 8 取水点 / 9 镇 / 10 专业聚落 / 11 撂荒田 / 12 废村 / 13 工棚、季节住 / 14 有人用
                                     // 15 塘 / 16 闸（P6）/ 17 大泊场（镇 / 邑治）/ 18 中转站（P7）/ 19 废塘、废渠首闸（P6b）
    GridI settle_fields;             // 田块号（0 = 无）
    Grid<uint8_t> cultivated, fallow_years;   // P5：已垦（在种，0 / 1 田 / 2 梯田）、撂荒了几年（0 = 不是撂荒地）
    GridI polder_id;                 // P6：圩号（0 = 不是圩田；圩田也算已垦）
    Grid<uint8_t> landcover_natural, landuse;   // P6b：没有人以前的地表（码同 landcover）、人工改造（0 没动过 … 7 采场，waterworks.hpp）
    double settle_pop = 0;
    double sec_settle = 0;
};

// 水系之后把 island.json 的值备齐（rim / peak / cliff 按 Python 的 round，主岛再加台面校正 dz）
void finalize_islands(Group& g);
// entity_rng(seed, 21, "island:{node}:{部件}")
Rng part_rng(const NodeInputs& inp, const std::string& part);

}  // namespace skyisle::island
