// 5.2 岛内地形（terrain.py）：岛形（椭圆 + 域扭曲 + 面积二分）→ 岛龄基形 → 测高曲线 → 隐式河流功率下切（粗网格）→ 仿射拟合。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

AgeKind age_class(double age, const Config& c);
Shape island_shape(Rng& rng, double area_km2, double res_km, double elong, double theta, const Config& c);
GridD base_form(Rng& rng, const Shape& s, double age, double area_km2, double res_km, const Config& c, AgeKind& kind);
GridD erode(Rng* rng, GridD h, const Mask& mask, double res_m, int rounds, double base_level, const Config& c,
            const GridD* uplift, const GridD* jitter);
void fit_rim(double surface, double relief, double median_frac, double rim_min, double& rim, double& R);

struct Sculpt {
    GridD h;   // 掩膜外 NaN
    AgeKind kind = MID;
    double rim = 0, peak = 0;
};
Sculpt sculpt_island(Rng& rng, const Shape& s, double age, double area_km2, double res_km, double surface, double relief,
                     double rim_min, bool is_main, const Config& c);

}  // namespace skyisle::island
