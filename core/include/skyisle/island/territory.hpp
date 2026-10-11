// 势力范围（territory.py）：邻群的陆地不许叠——与每个邻群按等效半径分界、各退 gap / 2。
#pragma once

#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

std::vector<Limit> limits(const PlanetView& pv, int64_t node, double gap_km, double reach, double reach_km);
// 仅重建已有几何候选；必须再经过新容量/原生mask认证，不能直接作为生成结果。
std::vector<Limit> historical_limits(const PlanetView& pv,int64_t node,double gap_km,double reach,double reach_km);
struct CapacityModel {
    std::vector<std::vector<Limit>> boundaries;
    std::vector<double> pressure_km, usable_area_km2, radius_bound_km;
    int sweeps=0, certificates=0, added_pairs=0;
    double worst_relative=0, max_pressure_km=0;
};
const CapacityModel& capacity_model(const PlanetView& pv,double gap_km,double reach,double reach_km);
// 岛的局部栅格（X − ox, Y − oy：相对岛的质心）上陆地在各分界线法向的最远投影
std::vector<double> mask_support(const Shape& s, double ox, double oy, const std::vector<Limit>& lim, double res_km);
std::vector<double> profile_support(const std::vector<double>& prof, const std::vector<Limit>& lim, double res_km);
double violation(double px, double py, const std::vector<double>& support, const std::vector<Limit>& lim);
// Dykstra 交替投影：离原点最近、整个落在势力范围里的偏移；返回越界量（> 0 放不下）
double nearest_fit(const std::vector<double>& support, const std::vector<Limit>& lim, double& ox, double& oy, int iters = 400);
// 把指定候选投影到同一可行域；不把候选所在的方向或各约束分开兜底。
double project_fit(const std::vector<double>& support, const std::vector<Limit>& lim, double& x, double& y, int iters = 400);
// 有界平移可行域的顶点；只用于寻找候选，最终仍用 violation 验收。
std::vector<std::array<double,2>> translation_vertices(const std::vector<double>& support,const std::vector<Limit>& lim);
double raster_violation(const Group& g, const std::vector<Limit>& lim);

}  // namespace skyisle::island
