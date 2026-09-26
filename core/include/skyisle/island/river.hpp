// 5.3b 河道成形（river.py）：水力几何的河宽 / 水深、下切并向下游单调的河床、河口豁口、河谷与漫滩、溪涧接干流、河道中心线。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

struct Channels {
    GridD h_new, width, depth;
    Grid<uint8_t> lvl, floodplain;
    std::vector<RiverRec> rivers;          // 局部切片坐标，按河口扁平下标升序（前端按汇流排序）
    int n_stream_falls = 0;
    bool has_cut = false;
    double max_cut_m = 0;                  // 已按 Python 的 round(·, 1)
    std::vector<std::vector<LinePt>> lines;   // 局部切片坐标
};

// 一座岛（局部切片）的河道下切。h：掩膜外 NaN；hf：路由面；recv：D8 下游（扁平下标，−1 无）。
Channels carve_channels(const GridD& h, const GridD& hf, const Mask& mk, const Mask& lake, const std::vector<int64_t>& recv,
                        const GridD& Akm, const Grid<uint8_t>& river_lvl, const Grid<uint8_t>& stream, double P_mm, double rim,
                        double keel, double res_m, double year_s, const Config& c, bool is_main);

std::vector<std::vector<LinePt>> trace_lines(const Mask& seed, const Mask& mk, const std::vector<int64_t>& recv, const GridD& width,
                                             const Grid<uint8_t>& lvl, const GridD& acc);

}  // namespace skyisle::island
