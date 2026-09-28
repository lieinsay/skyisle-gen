// 路网与寻路（docs/PLAN-TOWN.md 7.3）：路网图（节点 + 折线边，等级、宽度、每米代价）、图上的最短路、
// 细栅格上的 16 邻域 A*（坡度代价、纵坡上限、只在窄处过水、已有路打折）、折线 / 有向矩形的光栅化。
#pragma once

#include <functional>
#include <vector>

#include "skyisle/grid.hpp"
#include "skyisle/town/geom.hpp"
#include "skyisle/town/site.hpp"

namespace skyisle::town {

enum RoadClass : int { RC_TRUNK = 0, RC_MAIN = 1, RC_STREET = 2, RC_LANE = 3, RC_PATH = 4 };

struct NetEdge {
    int a = -1, b = -1;
    std::vector<V2> line;          // 从 a 到 b
    int cls = RC_LANE;
    double width = 3.0, len = 0.0, cost = 1.0;   // cost：每米的「路程」（主街打折、过桥加价已折进来）
    int street = -1;               // 属于哪条街（输出时把同一条街的边连成一条折线）
    double used_a = 0.0, used_b = 0.0;   // 从 a 端 / b 端起用到多长（≥ len 即整条）
    int bridges = 0;
};

struct NetNode {
    V2 p;
    std::vector<int> edges;
};

struct Network {
    std::vector<NetNode> nodes;
    std::vector<NetEdge> edges;
    int add_node(V2 p);
    int add_edge(int a, int b, std::vector<V2> line, int cls, double width, double cost, int street);
    // 从 src 出发的最短路（按 len × cost）；parent 记到达各节点的边（−1 = 源或到不了）
    std::vector<double> dijkstra(int src, std::vector<int>* parent) const;
    V2 point_at(int e, double s, V2* tangent = nullptr) const;   // 边上弧长 s 处的点与切向
    bool used(int e) const { return edges[e].used_a > 0.0 || edges[e].used_b > 0.0; }
    bool full(int e) const { return edges[e].used_a + edges[e].used_b >= edges[e].len - 1e-6; }
    int degree_used(int n) const;   // 用到的边里有几条接在这个节点上
};

// 细栅格寻路的参数
struct PathParams {
    double max_grade = 0.12;       // 纵坡上限（更陡的一步不可走）
    double slope_k = 3.0;          // 代价 × (1 + slope_k × (纵坡 / 上限)²)
    double bridge_max_m = 24.0;    // 水面比它宽的地方不过
    double water_cost = 8.0;       // 水上每米的代价（桥贵，所以垂直过窄处）
    double farm_cost = 0.25;       // 压田加价
    double road_factor = 0.35;     // 已有路上的代价倍数
    double flood_cost = 0.3;
    int bbox_margin = -1;          // > 0：只在起点与目标外包框外扩这么多格的范围里找（团块生长里的短巷）
    double clearance_m = 0.0;      // 路中线离障碍至少这么远（起点那一格不算；调用方从间距以外起步）：路不蹭着院墙走
    bool allow_squeeze = true;     // 严的找不到时，许不许接入点不管间距（团块生长的巷不许：宁可换个位置）
};

// 细栅格上的路网掩码与障碍：blocked 格不可走，road 格打折；从 from 走到任一 goal 格。成功时 out 是格心折线（含起点、终点），已简化平滑。
// fit_width（可空）：这条路离障碍最近处（起点 1.5 m 以外）容得下的路宽——严的没找到、放宽了接入点时，调用方把路收窄到它（巷子在两堵墙之间变窄）
bool find_path(const Site& s, const Mask& blocked, const Mask& road, const Mask& goal, const PathParams& pp, V2 from,
               std::vector<V2>& out, double* fit_width = nullptr);

// 光栅化：折线（宽 width）盖到的格、有向矩形（内缩 shrink）盖到的格
void raster_line(const Site& s, const std::vector<V2>& line, double width, const std::function<void(int, int)>& f);
void raster_obb(const Site& s, const Obb& o, double shrink, const std::function<void(int, int)>& f);
void raster_polygon(const Site& s, const std::vector<V2>& poly, const std::function<void(int, int)>& f);

// 折线上过河 / 溪的段（按河、溪格连续的一串；湖海不算——路不进湖海）：每段起止弧长
std::vector<std::pair<double, double>> water_runs(const Site& s, const std::vector<V2>& line);
std::vector<V2> subline(const std::vector<V2>& line, double s0, double s1);   // 弧长 [s0, s1] 的一段

}  // namespace skyisle::town
