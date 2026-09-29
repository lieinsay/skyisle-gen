// 等值线（marching squares）：栅格场 g 上 = level 的线，连成折线（平面坐标）。等高线算子取台线、環濠取外廓的等距线都用它。
#pragma once

#include <vector>

#include "skyisle/grid.hpp"
#include "skyisle/town/geom.hpp"
#include "skyisle/town/site.hpp"

namespace skyisle::town {

// 每 stride 格取一个点；只看离 c 在 radius 以内的格；NaN 格断开。闭合的线首尾相同（最后一点 = 第一点）。
// 鞍点按四角均值分：均值在 level 以上时连成一片。
std::vector<std::vector<V2>> iso_lines(const Site& s, const GridF& g, double level, int stride, V2 c, double radius);

}  // namespace skyisle::town
