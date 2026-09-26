// 第三层（岛群生成器）的输入与群状态（skyisle_gen/island 的 inp / g 的 C++ 形）。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/grid.hpp"

namespace skyisle::island {

constexpr uint64_t ISLAND_STREAM = 21;   // 与十步管线的流号 1–10 错开
enum AgeKind { YOUNG = 0, MID = 1, OLD = 2 };
const char* age_name(AgeKind k);          // "young" / "mid" / "old"

// 本群的行星层标量（_node_inputs）
struct NodeInputs {
    int64_t node = 0;
    uint64_t seed = 0;
    double lat = 0, lon = 0, area_km2 = 0, main_area_km2 = 0, height_m = 0, age = 0;
    bool layered = false;
    double keel_clearance_m = 300.0, area_median_km2 = 1.0;
    // 水系用
    double precip = 0, temp_sea = 0, lapse_c_per_km = 6.0, arable_frac = 0, river_size = 0;
    bool has_river = false;
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
};

struct Limit {
    int64_t node;
    double dist_km, ux, uy, limit_km;
};

// 一座岛的局部栅格（island_shape）：X[i, j] = xs[j]，Y[i, j] = −xs[i]
struct Shape {
    Mask mask;
    GridD inside;
    std::vector<double> xs;
    double X(int, int j) const { return xs[j]; }
    double Y(int i, int) const { return -xs[i]; }
};

struct IslandRec {
    int id = 0;
    int64_t area_cells = 0;
    double area_target = 0, cx = 0, cy = 0, surface = 0, relief_target = 0, rim = 0, peak = 0, keel = 0, age = 0;
    AgeKind kind = MID;
    int r0 = 0, c0 = 0, m = 0;          // 局部栅格左上角在群栅格里的行列、边长（裁切后）
    // island.json 里的四舍五入值（Python 版 hydro 读的是它们）
    double rim_j = 0, keel_j = 0, age_j = 0;
    // 水系（build_hydro 填）
    bool hydro = false;
    int n_lakes = 0;
    int64_t lake_cells = 0;
    double max_flowacc = 0;
    bool has_perennial = false, has_stream = false;
};

struct Link {
    int a = 0, b = 0;
    double gap = 0, dh = 0;
    bool bridge = false, fallback = false;
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

// 主岛集水盆地（hydro._basins 的原始数）：河口 = 汇流 ≥ thr 的出口格，按扁平下标升序；排序、取前几个、四舍五入在前端
struct Basins {
    bool present = false;
    double total_km2 = 0, thr_km2 = 0;
    std::vector<std::array<double, 3>> mouths;   // 群栅格 [行, 列, 汇流 km²]
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
    std::vector<double> rims;
    std::vector<IslandRec> islands;
    std::vector<Link> links;
    std::vector<std::pair<int, int>> tree;
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
    Grid<uint8_t> river, stream, lake, floodplain, landcover, arable;
    GridD width_m, depth_m, cut_m, slope;
    GridI recv_i, recv_j;
    std::vector<RiverLine> lines;
    std::vector<RiverRec> rivers;   // 主岛
    Basins basins;
    double P_mm = 0, river_thr = NaN, dz = 0, wind_u = 0, wind_v = 0, max_cut = 0;
    int n_falls = 0;
    double sec_hydro = 0;
};

}  // namespace skyisle::island
