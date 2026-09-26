// 水文核心（skyisle_gen/island/terrain.py 的 priority_fill / fill_iter / d8 / d8_random / accumulate）。
// 流向记成 recv（扁平下标）：≥ 0 下游格；−1 无（出口或平地）。to_void = 邻接虚空的出口格。
#pragma once

#include <vector>

#include "skyisle/grid.hpp"

namespace skyisle {

// Barnes 优先泛洪填洼（堆按 (z, i, j) 出，同 heapq）。掩膜外 = 虚空 = 出口；返回掩膜外 NaN。
GridD priority_fill(const GridD& h, const Mask& mask, double eps = 1e-3);
// Planchon–Darboux 迭代填洼（Jacobi，iters 轮）；只重算邻居变过的格，结果与整体迭代相同。
GridD fill_iter(const GridD& h, const Mask& mask, int iters, double eps = 0.01);

struct FlowDir {
    std::vector<int32_t> ri, rj;   // 下游格的行 / 列（−1 = 无）
    std::vector<uint8_t> to_void;  // 出口格（邻接虚空）
};
// D8：最陡下降（N8 次序、严格大于）；邻接虚空的格一律是出口。
FlowDir d8(const GridD& hf, const Mask& mask, double res_m);
// 随机流向：下坡邻格按 坡降^p 加权抽一个（抽 H × W 个均匀数，光栅次序）。
FlowDir d8_random(const GridD& hf, const Mask& mask, double res_m, Rng& rng, double p);

// 按流向的拓扑序：先下游后上游（recv < 0 的根在前，按下标升序起步）。只含掩膜内的格。
std::vector<int32_t> downstream_first(const std::vector<int64_t>& recv, const Mask& mask);
std::vector<int64_t> recv_flat(const FlowDir& f, int W);
// 汇流（格数或加权），掩膜外 0。
GridD accumulate(const Mask& mask, const FlowDir& f, const GridD* weight = nullptr);

}  // namespace skyisle
