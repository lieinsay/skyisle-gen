// 5.3b 河道成形（river.py）：水力几何的河宽 / 水深、下切并向下游单调的河床、河口豁口、河谷与漫滩、溪涧接干流、河道中心线。
// C1（PLAN-NATURE，DESIGN-NOTES 四点四十七）：河宽、水深是真实比例；河道格的高程记平岸水面（= 滩面，河床 = 高程 − 水深）——窄河那一格的平均就是岸，
// 河槽由中心线 + 宽 + 深表达（游戏切）。C2：谷底宽 W = K·A^0.4 × 岩性 × 岛龄，谷坡按限制度（峡谷 → 开阔）。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

struct Channels {
    GridD h_new, width, depth;
    GridD bed;                             // 河床高程（河道格；其余 NaN）
    GridD floor_w;                         // 谷底宽（m，河道格：这一段的目标谷底全宽 W）
    Grid<uint8_t> confine;                 // 限制度（河道格）：1 峡谷 / 2 半限制 / 3 开阔；0 = 不是河道
    GridD slope;                           // 河道比降（中心线格：顺流向 valley_slope_len_m 内的河床落差 / 流程；其余 NaN）
    Grid<uint8_t> lvl, floodplain;
    std::vector<RiverRec> rivers;          // 局部切片坐标，按河口扁平下标升序（前端按汇流排序）
    int n_stream_falls = 0;
    bool has_cut = false;
    double max_cut_m = 0;                  // 已按 Python 的 round(·, 1)
    std::vector<std::vector<LinePt>> lines;   // 局部切片坐标
};

// 一座岛（局部切片）的河道下切。h：掩膜外 NaN；hf：路由面；recv：D8 下游（扁平下标，−1 无）。
Channels carve_channels(const GridD& h, const GridD& hf, const Mask& mk, const Mask& lake, const std::vector<int64_t>& recv,
                        const GridD& Akm, const Grid<uint8_t>& river_lvl, const Grid<uint8_t>& stream, double P_mm, double runoff, double rim,
                        double keel, double res_m, double year_s, const Config& c, bool is_main,
                        const GridD* Qin = nullptr,    // Qin：局地雨算出的年均流量（m³/s，P4）；空 = 汇流 × P_mm 的旧式
                        const GridD* wall_deg = nullptr,    // 谷壁坡（°，B2：河床那格露出的岩性的坍塌角）；空 = [island.hydro] gorge_deg
                        const GridD* floor_lith = nullptr,  // 谷底宽的岩性系数（C2，河床那格露出的岩性）；空 = 1
                        double age = 0.5);                  // 岛龄（C2：谷底宽的岛龄系数）

std::vector<std::vector<LinePt>> trace_lines(const Mask& seed, const Mask& mk, const std::vector<int64_t>& recv, const GridD& width,
                                             const Grid<uint8_t>& lvl, const GridD& acc);

}  // namespace skyisle::island
