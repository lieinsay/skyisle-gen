// 岩层（PLAN-NATURE B2，DESIGN-NOTES 四点四十六）：层面与岩性表。
#include "skyisle/island/strat.hpp"

#include <algorithm>
#include <cmath>

#include "skyisle/island/terrain.hpp"

namespace skyisle::island {

const char* lith_name(int li) {
    static const char* N[LI_COUNT] = {"void", "limestone", "marl", "gabbro", "serpentinite", "pumice"};
    return (li >= 0 && li < LI_COUNT) ? N[li] : "void";
}

LithTable lith_table(const Config& c) {
    LithTable t;
    const std::vector<double> k = c.list("strat.erod", {0.5, 2.0, 0.6, 1.4, 0.7});
    const std::vector<double> a = c.list("strat.talus_deg", {55.0, 26.0, 45.0, 30.0, 60.0});
    const std::vector<double> d = c.list("strat.diffuse", {0.4, 1.6, 0.4, 1.3, 0.3});
    t.k.fill(1.0);
    t.diff.fill(1.0);
    t.talus_deg.fill(c.get("terrain.talus_deg"));
    for (int q = 1; q < LI_COUNT; ++q) {
        if (q - 1 < static_cast<int>(k.size())) t.k[q] = k[q - 1];
        if (q - 1 < static_cast<int>(a.size())) t.talus_deg[q] = a[q - 1];
        if (q - 1 < static_cast<int>(d.size())) t.diff[q] = d[q - 1];
    }
    for (int q = 0; q < LI_COUNT; ++q) t.talus_tan[q] = std::tan(t.talus_deg[q] * (PI / 180.0));
    return t;
}

StratField build_strat(Rng& rng, const Shape& s, const GridD& shape, AgeKind kind, double age, double rim, double R, double keel, double res_km,
                       double area_km2, const Config& c) {
    StratField F;
    const int n = s.mask.H;
    const size_t N = s.mask.size();
    StratRec& r = F.rec;
    r.on = true;
    // 层厚（一岛一组，随机流 island:<节点>:strat:<岛号>）
    const double sed = c.get("strat.sed_thick_m") * rng.lognormal(0.0, c.get("strat.sed_thick_sigma"));
    r.t_gab = c.get("strat.gabbro_thick_m") * rng.lognormal(0.0, c.get("strat.gabbro_thick_sigma"));
    const std::vector<double> bl = c.list("strat.bed_lime_m", {60.0, 160.0}), bm = c.list("strat.bed_marl_m", {40.0, 120.0});
    r.bed_lime = rng.uniform(bl[0], bl[1]);
    r.bed_marl = rng.uniform(bm[0], bm[1]);
    r.bed_phase = rng.uniform(0.0, r.bed_lime + r.bed_marl);
    const std::vector<double> cap = c.list("strat.cap_old_m", {80.0, 200.0});
    const double tc = rng.uniform(cap[0], cap[1]);
    r.t_cap = kind == OLD ? tc : 0.0;
    r.t_sed = r.t_cap + sed;
    // 冠顶已剥去的深度（m）按本岛的岛龄插值：削了多久就剥了多深，与现在剩下的起伏无关（老岛起伏小，却是剥得最深的）
    r.exhume = np_interp({age}, c.list("strat.exhume_age", {0.0, 0.3, 0.65, 1.0}), c.list("strat.exhume_m", {0.0, 400.0, 1400.0, 2500.0}))[0];
    const double p = c.get("strat.exhume_exp");
    // 构造面：基形按 struct_smooth_rel × 等效半径抹开（块均值 + [1,2,1] + 双线性放大）；s = 它归一到 [0, 1]（穹）
    const double Reff = std::sqrt(area_km2 / PI);
    const int f = std::max(1, static_cast<int>(std::nearbyint(c.get("strat.struct_smooth_rel") * Reff / res_km / 2.0)));
    GridD sh0(n, n, 0.0);
    for (size_t k = 0; k < N; ++k) sh0.v[k] = s.mask.v[k] ? shape.v[k] : 0.0;
    GridD m;
    if (f > 1) {
        GridD mc = block_mean(sh0, f);
        mc = smooth121(mc, Mask(mc.H, mc.W, 1), 2);
        m = upsample_bilinear(mc, f, n, n);
    } else {
        m = smooth121(sh0, Mask(n, n, 1), 4);
    }
    double lo = INF, hi = -INF;
    for (size_t k = 0; k < N; ++k)
        if (s.mask.v[k]) {
            lo = std::min(lo, m.v[k]);
            hi = std::max(hi, m.v[k]);
        }
    const double den = std::max(1e-9, hi - lo);
    const double ske = c.get("strat.skel_edge_frac"), skr = c.get("strat.skel_rise"), rmin = c.get("strat.rock_min_m");
    const double below = c.get("strat.skel_below_rim_m");
    F.top = GridD(n, n, NaN);
    F.skel = GridD(n, n, NaN);
    F.dome = GridD(n, n, 0.0);
    const double z_edge = keel + ske * (rim - keel);
    for (size_t k = 0; k < N; ++k) {
        if (!s.mask.v[k]) continue;
        const double sv = clip((m.v[k] - lo) / den, 0.0, 1.0);
        F.dome.v[k] = sv;
        F.top.v[k] = rim + R * m.v[k] + r.exhume * np_pow(sv, p);
        // 骨架顶面：岸崖上居中、往山下拱起，但不高过岸缘以下 skel_below_rim_m（流到岸缘的谷切不到它：浮石只在岸崖、河口豁口露）；
        // 岛心上面至少压着 rock_min_m 的岩层（按抹开的基形量）
        F.skel.v[k] = std::min({z_edge + skr * R * sv, rim + R * m.v[k] - rmin * sv, rim - below});
    }
    return F;
}

}  // namespace skyisle::island
