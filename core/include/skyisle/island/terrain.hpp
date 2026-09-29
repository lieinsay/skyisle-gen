// 5.2 岛内地形（terrain.py）：岛形（椭圆 + 域扭曲 + 面积二分）→ 岛龄基形 → 测高曲线 → 隐式河流功率下切（粗网格）→ 仿射拟合。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

AgeKind age_class(double age, const Config& c);
Shape island_shape(Rng& rng, double area_km2, double res_km, double elong, double theta, const Config& c);

// 多核嵌合（P4，DESIGN-NOTES 四点三十五）：汇聚带的一部分大岛由两三个核嵌成。
// multicore_spec 按岛定（随机流 island:<节点>:cores:<岛号>，别的抽样次序不动）；base_form 按岛形摆核、算造形并记下每格归哪个核；
// sculpt_island 拟合完量每个核的载荷与根（每个核的根在它的载荷正下方：浮力中心 = 重心，岛不歪）。
struct CoreSpec {
    bool on = false;
    int n = 0, primary = 0;
    std::vector<double> strength;       // 各核相对强度（主核 1；小的、老的核低）
    std::vector<Rng> rng;               // 接着抽缝的扭曲噪声（空 = 没有）
};
CoreSpec multicore_spec(Rng& rng, double area_km2, double age, double kernel, int btype, const Config& c);
struct CoreLayout {                     // base_form 摆出的核（局部栅格 km）
    int n = 0, primary = 0;
    std::vector<double> sx, sy, strength;
    Grid<int8_t> member;                // 每格归哪个核（掩膜外 −1）
};
GridD base_form(Rng& rng, const Shape& s, double age, double area_km2, double res_km, const Config& c, AgeKind& kind,
                CoreSpec* cores = nullptr, CoreLayout* layout = nullptr);
GridD erode(Rng* rng, GridD h, const Mask& mask, double res_m, int rounds, double base_level, const Config& c,
            const GridD* uplift, const GridD* jitter);
void fit_rim(double surface, double relief, double median_frac, double rim_min, double& rim, double& R);

struct Sculpt {
    GridD h;   // 掩膜外 NaN
    AgeKind kind = MID;
    double rim = 0, peak = 0;
    std::vector<CoreRec> cores;         // 多核岛才有
};
Sculpt sculpt_island(Rng& rng, const Shape& s, double age, double area_km2, double res_km, double surface, double relief,
                     double rim_min, bool is_main, const Config& c, CoreSpec* cores = nullptr);

}  // namespace skyisle::island
