// 场地地形 Site（docs/PLAN-TOWN.md 7.1）：一块细栅格（1–2 m）的地面、水、虚空、崖缘退让带、田、漫滩、地表，外加河的矢量。
// 三个来源——岛群窗口细化、合成地形、外部高程图——都落成同一个 Site；之后各层只认 Site。
// 地面的随机只按来源坐标取（哈希噪声），不含风格：同一块地换风格，地面逐字节不变。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/grid.hpp"
#include "skyisle/town/geom.hpp"

namespace skyisle::town {

// 水格的种类
enum : uint8_t { WATER_NONE = 0, WATER_RIVER = 1, WATER_STREAM = 2, WATER_LAKE = 3, WATER_SEA = 4 };
// 地表类别（与岛群生成器 hydro.py 同码）
enum : uint8_t { LC_VOID = 0, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE };

struct River {
    std::vector<V2> line;                 // 平面坐标（m），从上游到下游
    std::vector<double> width_m;          // 每个顶点的河宽
    std::vector<double> depth_m;          // 每个顶点的水深（河心）
    std::vector<double> surface_m;        // 每个顶点的水面高程（顺流单调不增）
    bool seasonal = false;                // 季节性溪涧（岛群的级别 0）
};

struct Site {
    double res_m = 1.0;
    int H = 0, W = 0;
    double x0 = 0.0, y0 = 0.0;            // 左上格角的平面坐标：格 (i, j) 的中心 = (x0 + (j + 0.5)·res, y0 − (i + 0.5)·res)
    double frame_x = 0.0, frame_y = 0.0;  // 平面原点在来源坐标系里的位置（岛群平面 m；合成地形 0）
    double lat_deg = 0.0;                 // 所在纬度：定朝阳方向（南半球朝北）
    GridF height;                         // 地面（水下是河床 / 湖底）
    GridF water_level;                    // 水面（非水格 NaN）
    Grid<uint8_t> water;                  // WATER_*
    Mask sky;                             // 岛外虚空
    Mask edge;                            // 崖缘退让带（离虚空 < edge_setback_m）
    Mask farmland;                        // 田（上游的可耕地；合成地形按平、近水自动划）
    Mask flood;                           // 漫水（上游漫滩，或离水面高差小且离水近）
    Grid<uint8_t> landcover;              // LC_*
    Grid<int16_t> island;                 // 岛号（−1 = 虚空；合成地形全 0）
    GridF water_dist_m;                   // 到最近水格的距离
    std::vector<River> rivers;

    V2 center(int i, int j) const { return {x0 + (j + 0.5) * res_m, y0 - (i + 0.5) * res_m}; }
    bool cell_of(V2 p, int& i, int& j) const;
};

// 岛群窗口的输入：前端从 terrain.npz / rivers.json 裁出、换好坐标
struct WindowIn {
    GridD height;                         // 粗栅格高程（NaN = 虚空）
    Grid<int16_t> island;
    Mask lake, floodplain, arable, terrace;
    Grid<uint8_t> landcover;
    GridD river_depth;                    // 河道 / 溪涧格的水深
    double coarse_res_m = 100.0;
    double center_r = 0.0, center_c = 0.0;   // 窗口中心在粗栅格里的连续坐标（格角为原点，行向南、列向东）
    double frame_x = 0.0, frame_y = 0.0;     // 窗口中心在岛群平面坐标里的位置（m）
    std::vector<River> rivers;            // 以窗口中心为原点的平面坐标；只需 line 与 width_m，水深与水面在这里按粗栅格补
    double half_m = 500.0, res_m = 1.0, lat_deg = 0.0;
    uint64_t seed = 0;
};
Site build_site_window(const WindowIn& in, const Config& c);

// 合成地形：kind ∈ plain / meander / valley / sunslope / hilltop / lakeshore / fjord / gully / confluence / rim
Site build_site_synth(const std::string& kind, double half_m, double res_m, double lat_deg, uint64_t seed, const Config& c);
bool synth_kind_known(const std::string& kind);

// 外部高程图（m，已按 res 排好，行向南）；water 可空（非空时 1 = 水面，水位取该格高程）
Site build_site_heightmap(const GridD& height, const Mask* water, double res_m, double lat_deg, uint64_t seed, const Config& c);

// 公共收尾：按 rivers 刻河（河床、河岸、水面），算 water_dist、漫水、崖缘退让带，田去掉水 / 虚空 / 退让带。
// 调用前 height / sky / island / landcover / farmland / flood（上游给的部分）与湖（water = WATER_LAKE、water_level）要先填好。
void finish_site(Site& s, const Config& c);

}  // namespace skyisle::town
