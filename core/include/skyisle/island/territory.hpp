// 势力范围（territory.py）：邻群的陆地不许叠——与每个邻群按等效半径分界、各退 gap / 2。
#pragma once

#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

std::vector<Limit> limits(const PlanetView& pv, int64_t node, double gap_km, double reach, double reach_km);
// 岛的局部栅格（X − ox, Y − oy：相对岛的质心）上陆地在各分界线法向的最远投影
std::vector<double> mask_support(const Shape& s, double ox, double oy, const std::vector<Limit>& lim, double res_km);
std::vector<double> profile_support(const std::vector<double>& prof, const std::vector<Limit>& lim, double res_km);
double violation(double px, double py, const std::vector<double>& support, const std::vector<Limit>& lim);
// Dykstra 交替投影：离原点最近、整个落在势力范围里的偏移；返回越界量（> 0 放不下）
double nearest_fit(const std::vector<double>& support, const std::vector<Limit>& lim, double& ox, double& oy, int iters = 400);
double raster_violation(const Group& g, const std::vector<Limit>& lim);

}  // namespace skyisle::island
