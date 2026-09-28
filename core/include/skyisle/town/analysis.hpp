// 场地分析（docs/PLAN-TOWN.md 7.2）：Site 上的一组栅格场，纯函数；住宅兴趣图 = Σ 风格权重 × 归一化的场。
#pragma once

#include "skyisle/grid.hpp"
#include "skyisle/town/site.hpp"
#include "skyisle/town/style.hpp"

namespace skyisle::town {

struct Fields {
    GridF slope_deg;        // 虚空 NaN
    GridF downslope;        // 下坡方向的方位角（弧度；平地 NaN）
    GridF sun;              // 冬至正午日照强度 / 平地的（0 = 背阴，1 = 平地，> 1 = 朝阳坡）
    GridF shelter;          // 背风：冬季风上风向 30–100 m 处比这里高多少（坡度量纲；没风向时 0）
    GridF tpi_s, tpi_l;     // 地势：离 25 m / 100 m 邻域平均高多少（m）
    GridF dist_farm_m, dist_sky_m;
    Mask buildable;         // 非虚空、非崖缘、非水、坡 ≤ 上限、非漫水（风格允许时除外）
    double sun_bearing = 0.0;   // 朝阳的方位角（朝赤道 + 风格偏角）
    double sun_alt = 0.0;       // 冬至正午太阳高度角
    double wind_from = NaN;     // 冬季风从哪来（方位角；NaN = 不知道）
};

Fields analyze(const Site& s, const Style& st, double wind_from);
// 住宅兴趣图（0–约 1 的量级；不可建格 NaN）
GridF interest_dwelling(const Site& s, const Fields& f, const Style& st);

// 小工具：格上的值（出界 / NaN 返回 fb）；按平面坐标双线性取
float field_at(const Site& s, const GridF& g, V2 p, float fb);

}  // namespace skyisle::town
