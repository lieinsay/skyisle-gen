// 朝向规则链（docs/PLAN-TOWN.md 6.3）：每条规则给「想朝哪」与「多想」，在候选角度上取 Σ 权重 × 满足度最大的。
// 满足度：容差内 1，容差外 cos(3 × (偏差 − 容差))（超出 30° 到 0）；另加 0.01 × cos(偏差) 让解靠近目标而不是贴着容差边。
#pragma once

#include <vector>

#include "skyisle/grid.hpp"
#include "skyisle/town/style.hpp"

namespace skyisle::town {

// 某处的朝向语境：各目标方向（方位角，弧度）；NaN = 这里没有这一项，相应规则跳过
struct OrientCtx {
    double sun = NaN;            // 朝阳（朝赤道）
    double wind_from = NaN;      // 冬季风从哪来（背风 = 朝它的反方向）
    double street = NaN;         // 朝街：从这里指向最近的街
    double street_dir = NaN;     // 街的走向（切线）
    double water = NaN;          // 指向最近的水
    double downslope = NaN;
    double yard = NaN;           // 指向共同的院子
    double baseline = NaN;       // 梳式的前沿
};

double solve_facing(const std::vector<OrientRule>& rules, const OrientCtx& ctx, double snap, double fallback);
// 一个朝向对某条规则的偏差（弧度；这条规则在语境里没有目标时返回 NaN）
double rule_deviation(const OrientRule& r, const OrientCtx& ctx, double facing);

}  // namespace skyisle::town
