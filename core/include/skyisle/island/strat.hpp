// 岩层（PLAN-NATURE B2，DESIGN-NOTES 四点四十六；spec 13 第二节第 6 条）：岛是两层——上面是从海底带上来的岩层
// （由浅到深：沉积盖层 = 硬的海相石灰岩与软的泥灰岩互层、辉长岩、蛇纹岩），下面是浮石骨架。
// 每座岛一套层面：构造顶面 top（岩层的顶，按基形的平滑穹拱起、冠顶已剥去一截）与骨架顶面 skel（岸崖上居中、往山下拱起）；
// 某格露出哪层 = 该格高程在层面下多深（lith_at）。侵蚀每轮按露出的岩性取可蚀性、坍塌角、坡面扩散（硬层成崖、软层成坡），
// 水系之后按最终高程出 lith 栅格，地表（蛇纹岩秃山）、资源（石料岩性、盐、溶洞）、地貌都读它。
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "skyisle/config.hpp"

namespace skyisle::island {

enum Lith : uint8_t { LI_VOID = 0, LI_LIME = 1, LI_MARL = 2, LI_GABBRO = 3, LI_SERP = 4, LI_PUMICE = 5, LI_COUNT = 6 };
const char* lith_name(int li);   // "limestone" / "marl" / "gabbro" / "serpentinite" / "pumice"

// 一座岛的层序（厚度是拟合前的米；scale 把它换到最终高程：拟合是仿射的，层面跟着同一个仿射变）
struct StratRec {
    bool on = false;
    double t_cap = 0;          // 老岛顶上的硬石灰岩盖（0 = 没有）
    double t_sed = 0;          // 沉积盖层总厚（含盖）
    double t_gab = 0;          // 辉长岩厚
    double bed_lime = 0, bed_marl = 0, bed_phase = 0;   // 盖以下石灰岩 / 泥灰岩的单层厚、从哪一层起
    double exhume = 0;         // 冠顶已剥去的深度（m，拟合前）
    double scale = 1.0;        // 拟合后 1 m 厚 = scale m 高程
};

// 各岩性的侵蚀参数（[island.strat] 的 erod / talus_deg / diffuse，层序同 Lith 去掉 VOID）
struct LithTable {
    std::array<double, LI_COUNT> k{}, talus_tan{}, diff{}, talus_deg{};
};
LithTable lith_table(const Config& c);

// 高程 h 处露出哪层：深度 = (top − h) / scale；h ≤ skel 是浮石骨架
inline uint8_t lith_at(double h, double top, double skel, const StratRec& s) {
    if (!s.on) return LI_GABBRO;
    if (h <= skel) return LI_PUMICE;
    const double d = (top - h) / s.scale;
    if (d < s.t_cap) return LI_LIME;
    if (d < s.t_sed) {
        const double per = s.bed_lime + s.bed_marl;
        double x = std::fmod(d - s.t_cap + s.bed_phase, per);
        if (x < 0) x += per;
        return x < s.bed_lime ? LI_LIME : LI_MARL;
    }
    if (d < s.t_sed + s.t_gab) return LI_GABBRO;
    return LI_SERP;
}
inline bool lith_sediment(uint8_t li) { return li == LI_LIME || li == LI_MARL; }

struct StratLayer {
    double bottom_m = 0, top_m = 0;
    uint8_t lith = LI_VOID;
};

// Exact surviving intervals, bottom to surface, using the same contacts as
// lith_at. No invented weathering thickness or permeability assignment.
std::vector<StratLayer> strat_column(double surface, double structural_top,
                                     double skeleton_top, const StratRec& strata);

}  // namespace skyisle::island
