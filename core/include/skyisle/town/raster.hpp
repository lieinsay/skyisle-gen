// 聚落营建器的细栅格件：按绝对坐标取值的哈希噪声（换窗口、换风格不变）、粗栅格上的双三次 / 双线性采样、精确欧氏距离变换。
#pragma once

#include <cstdint>

#include "skyisle/grid.hpp"

namespace skyisle::town {

uint64_t mix64(uint64_t x);                                   // splitmix64 的混合函数
double hash_unit(uint64_t seed, int64_t i, int64_t j);        // 格点 (i, j) 的哈希值，[−1, 1)
double value_noise(uint64_t seed, double x, double y);        // 格距 1 的值噪声（五次缓动），[−1, 1]
// 分形值噪声：x、y 与 wavelength 同单位（m）；各层振幅 persistence^k、频率 2^k，按总振幅归一，约 [−1, 1]
double fbm(uint64_t seed, double x, double y, double wavelength, int octaves, double persistence = 0.5);

// 粗栅格采样：连续坐标 (r, c) 以格角为原点（格 (i, j) 的值在 (i + 0.5, j + 0.5)），出界夹到边上。NaN 格要先填好。
double sample_cubic(const GridD& g, double r, double c);      // Catmull–Rom（过格心，不越出相邻值太多）
double sample_linear(const GridD& g, double r, double c);
double sample_linear(const Mask& m, double r, double c);      // 0 / 1 掩码的双线性（软边界）

// 到最近种子格的精确欧氏距离（单位 = 格；两遍一维平方距离变换，O(格数)）；src 记最近种子的扁平下标（无则 −1）。
// 不用 8 邻域倒角：它的梯度方向被量化成八向，陡坡上的晕渲会出竖条纹
void edt(const Mask& seed, GridF& dist, Grid<int32_t>* src);

}  // namespace skyisle::town
