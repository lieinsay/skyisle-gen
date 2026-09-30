// 亚格岸距（PLAN-NATURE B4，DESIGN-NOTES 四点四十六）：岸线是岛形连续场 f = τ 的等值线，不是格子的边。
#pragma once

#include "skyisle/grid.hpp"

namespace skyisle::island {

// phi：群栅格上的岸线场（陆地格 > 0、虚空 < 0，零等值线 = 岸线）。按格心做 marching squares 出折线，
// 每格到折线的欧氏距离（先算折线旁两格，再按邻格的最近线段传两遍），陆地为正、虚空为负，单位 m
GridD coast_distance(const GridD& phi, double res_m);

}  // namespace skyisle::island
