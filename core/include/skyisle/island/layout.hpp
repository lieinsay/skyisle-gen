// 5.1 群内布局（skyisle_gen/island/layout.py）：岛数、Zipf 大小、板块走向、角向半径剖面、放置、台面与起伏、岸距、短渡（P5 起没有索桥与导水槽）。
#pragma once

#include <map>
#include <utility>
#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

constexpr int CONVERGENT = 0, DIVERGENT = 1, TRANSFORM = 2;

int island_count(Rng& rng, double area_km2, double area_median, const Config& c);
std::vector<double> zipf_sizes(double area_km2, double main_km2, int n, const Config& c);
// 板块边界走向（弧度）、边界核强度、边界类型
void boundary_axis(const PlanetView& pv, double lat, double lon, double& axis, double& kernel, int& btype);
// 角向最大半径剖面（72 桶）；center = 岛心相对局部栅格中心（km）
void radial_profile(const Mask& mask, double res_km, std::array<double, 2>& center, std::vector<double>& prof, int nbins = 72);

struct TerritoryPlace {
    const std::vector<Limit>* lim = nullptr;
    const std::vector<std::vector<double>>* support = nullptr;
    double main_x = 0, main_y = 0;
    double gap_min_km = 0.3;
};
std::vector<std::array<double, 2>> place_islands(Rng& rng, const std::vector<std::vector<double>>& profiles, const std::vector<double>& sizes,
                                                 double axis, double kernel, int btype, const Config& c,
                                                 const TerritoryPlace* territory = nullptr);
std::vector<double> surface_heights(Rng& rng, int n, double height_m, bool layered, const Config& c);
std::vector<double> relief_targets(Rng& rng, const std::vector<double>& sizes, const std::vector<double>& ages, const Config& c);
// 浮高的原始值（主岛 0；往下的等地形拟合出岸缘后由 build_terrain 按离下限的余量缩）：参数读 float.*（DESIGN-NOTES 四点二十八）
std::vector<double> float_offsets(Rng& rng, const std::vector<double>& ages, const Config& c);
std::map<std::pair<int, int>, double> shoreline_gaps(const std::vector<MaskPos>& masks_pos, double res_km,
                                                     const std::vector<std::array<double, 2>>& centers_cell,
                                                     const std::vector<double>& radii_km, double max_gap_km);
void links(const std::map<std::pair<int, int>, double>& gaps, const std::vector<double>& rims, int n, std::vector<Link>& out);

}  // namespace skyisle::island
