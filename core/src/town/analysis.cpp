// 场地分析（PLAN-TOWN 7.2）：坡度、下坡方向、冬至日照、背风、两个尺度的地势（TPI）、离田 / 离虚空、可建、住宅兴趣图。
#include "skyisle/town/analysis.hpp"

#include <algorithm>
#include <cmath>

#include "skyisle/town/raster.hpp"

namespace skyisle::town {

namespace {

constexpr double DEG = PI / 180.0;

// 高程的格值，虚空 / 出界用最近的有效值代替（只给差分用）
double hval(const Site& s, int i, int j, double fb) {
    i = std::clamp(i, 0, s.H - 1), j = std::clamp(j, 0, s.W - 1);
    const double v = s.height(i, j);
    return std::isfinite(v) ? v : fb;
}

// 盒式邻域平均（积分图；只数有效格）
GridF box_mean(const Site& s, int r) {
    const int H = s.H, W = s.W;
    std::vector<double> S(static_cast<size_t>(H + 1) * (W + 1), 0.0), N(S.size(), 0.0);
    auto at = [&](std::vector<double>& a, int i, int j) -> double& { return a[static_cast<size_t>(i) * (W + 1) + j]; };
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const double v = s.height(i, j);
            const bool ok = std::isfinite(v);
            at(S, i + 1, j + 1) = at(S, i, j + 1) + at(S, i + 1, j) - at(S, i, j) + (ok ? v : 0.0);
            at(N, i + 1, j + 1) = at(N, i, j + 1) + at(N, i + 1, j) - at(N, i, j) + (ok ? 1.0 : 0.0);
        }
    GridF out(H, W, static_cast<float>(NaN));
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const int a = std::max(0, i - r), b = std::min(H, i + r + 1), c = std::max(0, j - r), d = std::min(W, j + r + 1);
            const double n = at(N, b, d) - at(N, a, d) - at(N, b, c) + at(N, a, c);
            if (n > 0) out(i, j) = static_cast<float>((at(S, b, d) - at(S, a, d) - at(S, b, c) + at(S, a, c)) / n);
        }
    return out;
}

GridF dist_m(const Mask& seed, double res) {
    GridF d;
    bool any = false;
    for (uint8_t v : seed.v) any = any || v;
    if (!any) return GridF(seed.H, seed.W, 1e9f);
    edt(seed, d, nullptr);
    for (float& x : d.v) x = static_cast<float>(x * res);
    return d;
}

}  // namespace

float field_at(const Site& s, const GridF& g, V2 p, float fb) {
    const double c = (p.x - s.x0) / s.res_m - 0.5, r = (s.y0 - p.y) / s.res_m - 0.5;
    const int i0 = static_cast<int>(std::floor(r)), j0 = static_cast<int>(std::floor(c));
    if (i0 < 0 || j0 < 0 || i0 + 1 >= s.H || j0 + 1 >= s.W) {
        int i, j;
        if (!s.cell_of(p, i, j)) return fb;
        const float v = g(i, j);
        return std::isfinite(v) ? v : fb;
    }
    const double tr = r - i0, tc = c - j0;
    const float a = g(i0, j0), b = g(i0, j0 + 1), d = g(i0 + 1, j0), e = g(i0 + 1, j0 + 1);
    if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(d) && std::isfinite(e))) {
        int i, j;
        s.cell_of(p, i, j);
        const float v = g(i, j);
        return std::isfinite(v) ? v : fb;
    }
    return static_cast<float>((a * (1 - tc) + b * tc) * (1 - tr) + (d * (1 - tc) + e * tc) * tr);
}

Fields analyze(const Site& s, const Style& st, double wind_from) {
    const int H = s.H, W = s.W;
    const double res = s.res_m;
    Fields f;
    f.slope_deg = GridF(H, W, static_cast<float>(NaN));
    f.downslope = GridF(H, W, static_cast<float>(NaN));
    f.sun = GridF(H, W, 1.0f);
    f.shelter = GridF(H, W, 0.0f);
    f.buildable = Mask(H, W, 0);
    f.wind_from = wind_from;
    // 朝阳 = 朝赤道；冬至正午太阳高度 = 90° − |纬度| − 23.44°（夹到 ≥ 5°）
    f.sun_bearing = wrap_pi((s.lat_deg >= 0.0 ? PI : 0.0) + st.sun_offset);
    f.sun_alt = std::max(5.0, 90.0 - std::fabs(s.lat_deg) - 23.44) * DEG;
    const V2 sh = bearing_vec(f.sun_bearing);
    const double sx = sh.x * std::cos(f.sun_alt), sy = sh.y * std::cos(f.sun_alt), sz = std::sin(f.sun_alt);
    const bool has_wind = std::isfinite(wind_from);
    const V2 up = has_wind ? bearing_vec(wind_from) : V2{0, 0};

    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const double h = s.height(i, j);
            if (!std::isfinite(h)) continue;
            const double dzdx = (hval(s, i, j + 1, h) - hval(s, i, j - 1, h)) / (2.0 * res);   // 向东
            const double dzdy = (hval(s, i - 1, j, h) - hval(s, i + 1, j, h)) / (2.0 * res);   // 向北（行向南）
            const double g = std::hypot(dzdx, dzdy);
            f.slope_deg(i, j) = static_cast<float>(std::atan(g) / DEG);
            if (g > 1e-4) f.downslope(i, j) = static_cast<float>(bearing_of({-dzdx, -dzdy}));
            const double nz = 1.0 / std::sqrt(1.0 + g * g);
            const double cosi = (-dzdx * sx - dzdy * sy + sz) * nz;
            f.sun(i, j) = static_cast<float>(std::max(0.0, cosi) / sz);
            if (has_wind) {
                double best = -1e9;
                for (double d : {30.0, 60.0, 100.0}) {
                    int ii, jj;
                    if (!s.cell_of(s.center(i, j) + up * d, ii, jj)) continue;
                    const double hu = s.height(ii, jj);
                    if (std::isfinite(hu)) best = std::max(best, (hu - h) / d);
                }
                f.shelter(i, j) = static_cast<float>(best > -1e8 ? std::clamp(best, -0.3, 0.6) : 0.0);
            }
        }
    const GridF ms = box_mean(s, std::max(1, static_cast<int>(std::lround(25.0 / res))));
    const GridF ml = box_mean(s, std::max(2, static_cast<int>(std::lround(100.0 / res))));
    f.tpi_s = GridF(H, W, 0.0f);
    f.tpi_l = GridF(H, W, 0.0f);
    for (size_t k = 0; k < s.height.size(); ++k) {
        if (!std::isfinite(s.height.v[k])) continue;
        f.tpi_s.v[k] = s.height.v[k] - ms.v[k];
        f.tpi_l.v[k] = s.height.v[k] - ml.v[k];
    }
    f.dist_farm_m = dist_m(s.farmland, res);
    f.dist_sky_m = dist_m(s.sky, res);
    for (size_t k = 0; k < s.height.size(); ++k) {
        const bool ok = !s.sky.v[k] && !s.edge.v[k] && s.water.v[k] == WATER_NONE && (st.allow_flood || !s.flood.v[k]) &&
                        std::isfinite(f.slope_deg.v[k]) && f.slope_deg.v[k] <= st.max_slope_deg;
        f.buildable.v[k] = ok ? 1 : 0;
    }
    return f;
}

GridF interest_dwelling(const Site& s, const Fields& f, const Style& st) {
    GridF out(s.H, s.W, static_cast<float>(NaN));
    for (size_t k = 0; k < out.size(); ++k) {
        if (!f.buildable.v[k]) continue;
        const double flat = std::clamp(1.0 - f.slope_deg.v[k] / std::max(1.0, st.max_slope_deg), 0.0, 1.0);
        const double water = std::exp(-s.water_dist_m.v[k] / st.water_scale_m);
        const double dry = std::clamp(0.5 + f.tpi_l.v[k] / 3.0, 0.0, 1.0);
        const double sun = std::clamp(static_cast<double>(f.sun.v[k]) / 1.5, 0.0, 1.0);
        const double farm = s.farmland.v[k] ? 1.0 : 0.0;
        const double edge = std::exp(-f.dist_sky_m.v[k] / st.edge_scale_m);
        const double shel = std::clamp(f.shelter.v[k] / 0.2, 0.0, 1.0);
        out.v[k] = static_cast<float>(st.w_flat * flat + st.w_water * water + st.w_dry * dry + st.w_sun * sun + st.w_farmland * farm +
                                      st.w_edge * edge + st.w_shelter * shel);
    }
    // 归一到 [0, 1]：各村之间只比相对高低
    float lo = 1e30f, hi = -1e30f;
    for (float v : out.v)
        if (std::isfinite(v)) lo = std::min(lo, v), hi = std::max(hi, v);
    if (hi > lo)
        for (float& v : out.v)
            if (std::isfinite(v)) v = (v - lo) / (hi - lo);
    return out;
}

}  // namespace skyisle::town
