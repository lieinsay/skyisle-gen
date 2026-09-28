// Site 的公共收尾（刻河、水距、漫水、崖缘退让带）与外部高程图。
#include "skyisle/town/site.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "skyisle/town/raster.hpp"

namespace skyisle::town {

bool Site::cell_of(V2 p, int& i, int& j) const {
    j = static_cast<int>(std::floor((p.x - x0) / res_m));
    i = static_cast<int>(std::floor((y0 - p.y) / res_m));
    return i >= 0 && j >= 0 && i < H && j < W;
}

namespace {

struct Best {
    GridF d, surf, depth, hw;
    Grid<uint8_t> seasonal;
};

void carve_rivers(Site& s, const Config& c, GridF& frise, GridF& freach) {
    if (s.rivers.empty()) return;
    const float inf = std::numeric_limits<float>::infinity();
    const double bank = c.get("site.bank_m");
    const double freeboard = c.get("site.bank_freeboard_m");
    const double bank_slope = c.get("site.bank_slope");
    const double rise0 = c.get("site.flood_rise_m"), reach0 = c.get("site.flood_reach_m");
    const double ref_depth = c.get("site.flood_ref_depth_m"), reach_w = c.get("site.flood_reach_widths");
    Best b{GridF(s.H, s.W, inf), GridF(s.H, s.W, 0.0f), GridF(s.H, s.W, 0.0f), GridF(s.H, s.W, 0.0f), Grid<uint8_t>(s.H, s.W, 0)};
    for (const River& r : s.rivers) {
        for (size_t k = 0; k + 1 < r.line.size(); ++k) {
            const V2 a = r.line[k], e = r.line[k + 1];
            const double reach = 0.5 * std::max(r.width_m[k], r.width_m[k + 1]) + bank + 2.0 * s.res_m;
            int i0, j0, i1, j1;
            s.cell_of({std::min(a.x, e.x) - reach, std::max(a.y, e.y) + reach}, i0, j0);
            s.cell_of({std::max(a.x, e.x) + reach, std::min(a.y, e.y) - reach}, i1, j1);
            i0 = std::max(i0, 0), j0 = std::max(j0, 0), i1 = std::min(i1, s.H - 1), j1 = std::min(j1, s.W - 1);
            for (int i = i0; i <= i1; ++i)
                for (int j = j0; j <= j1; ++j) {
                    double t = 0.0;
                    const double d = dist_point_segment(s.center(i, j), a, e, &t);
                    if (d >= reach || d >= b.d(i, j)) continue;
                    b.d(i, j) = static_cast<float>(d);
                    b.hw(i, j) = static_cast<float>(0.5 * (r.width_m[k] + (r.width_m[k + 1] - r.width_m[k]) * t));
                    b.depth(i, j) = static_cast<float>(r.depth_m[k] + (r.depth_m[k + 1] - r.depth_m[k]) * t);
                    b.surf(i, j) = static_cast<float>(r.surface_m[k] + (r.surface_m[k + 1] - r.surface_m[k]) * t);
                    b.seasonal(i, j) = r.seasonal ? 1 : 0;
                }
        }
    }
    for (int i = 0; i < s.H; ++i)
        for (int j = 0; j < s.W; ++j) {
            const float d = b.d(i, j);
            if (!(d < inf) || s.sky(i, j) || s.water(i, j) == WATER_LAKE || s.water(i, j) == WATER_SEA) continue;
            const double hw = std::max(0.5 * s.res_m, static_cast<double>(b.hw(i, j)));
            const double surf = b.surf(i, j);
            if (d < hw) {
                const double q = d / hw;
                const double bed = surf - b.depth(i, j) * (1.0 - q * q);
                s.height(i, j) = static_cast<float>(std::min(static_cast<double>(s.height(i, j)), bed));
                s.water(i, j) = b.seasonal(i, j) ? WATER_STREAM : WATER_RIVER;
                s.water_level(i, j) = static_cast<float>(surf);
                // 漫水按河的大小：小溪涨不了多高、也漫不了多远
                frise(i, j) = static_cast<float>(rise0 * clip(b.depth(i, j) / ref_depth, 0.3, 1.0));
                freach(i, j) = static_cast<float>(std::min(reach0, reach_w * 2.0 * hw));
            } else {
                const double top = surf + freeboard + (d - hw) * bank_slope;
                s.height(i, j) = static_cast<float>(std::min(static_cast<double>(s.height(i, j)), top));
            }
        }
}

}  // namespace

void finish_site(Site& s, const Config& c) {
    // 各水格能漫多高、多远：湖海按 flood_lake_rise_m 与 flood_reach_m，河在刻河时按河的大小填
    GridF frise(s.H, s.W, static_cast<float>(c.get("site.flood_lake_rise_m")));
    GridF freach(s.H, s.W, static_cast<float>(c.get("site.flood_reach_m")));
    carve_rivers(s, c, frise, freach);
    // 水距与最近水格的水面
    Mask wet(s.H, s.W, 0);
    for (size_t k = 0; k < wet.size(); ++k) wet[k] = s.water[k] != WATER_NONE ? 1 : 0;
    Grid<int32_t> src;
    edt(wet, s.water_dist_m, &src);
    for (auto& d : s.water_dist_m.v) d = d * static_cast<float>(s.res_m);
    // 只有平地会漫水：陡坡上挨着溪的格与溪面同高，但水顺坡就走了
    const double flat = std::tan(c.get("site.flood_max_slope_deg") * PI / 180.0);
    for (int i = 0; i < s.H; ++i)
        for (int j = 0; j < s.W; ++j) {
            const size_t k = static_cast<size_t>(i) * s.W + j;
            if (s.sky[k] || wet[k]) continue;
            const int q = src[k];
            if (q < 0) continue;
            const size_t w = static_cast<size_t>(q);
            if (!(s.water_dist_m[k] < freach[w] && s.height[k] - s.water_level[w] < frise[w])) continue;
            const int j0 = std::max(0, j - 1), j1 = std::min(s.W - 1, j + 1), i0 = std::max(0, i - 1), i1 = std::min(s.H - 1, i + 1);
            const double gx = (s.height(i, j1) - s.height(i, j0)) / ((j1 - j0) * s.res_m);
            const double gy = (s.height(i0, j) - s.height(i1, j)) / ((i1 - i0) * s.res_m);
            if (!(std::hypot(gx, gy) < flat)) continue;   // NaN（挨着虚空）也不算
            s.flood[k] = 1;
        }
    // 崖缘退让带
    GridF dsky;
    edt(s.sky, dsky, nullptr);
    const double setback = c.get("site.edge_setback_m");
    s.edge = Mask(s.H, s.W, 0);
    for (size_t k = 0; k < dsky.size(); ++k) s.edge[k] = (!s.sky[k] && dsky[k] * s.res_m < setback) ? 1 : 0;
    for (size_t k = 0; k < wet.size(); ++k) {
        if (wet[k] || s.sky[k] || s.edge[k]) s.farmland[k] = 0;
        if (s.sky[k]) {
            s.landcover[k] = LC_VOID;
            s.height[k] = std::numeric_limits<float>::quiet_NaN();
        } else if (wet[k]) {
            s.landcover[k] = s.water[k] == WATER_LAKE ? LC_LAKE : LC_RIVER;
        } else if (s.farmland[k] && s.landcover[k] != LC_TERRACE) {
            s.landcover[k] = LC_ARABLE;
        }
        if (!wet[k]) s.water_level[k] = std::numeric_limits<float>::quiet_NaN();
    }
}

Site build_site_heightmap(const GridD& height, const Mask* water, double res_m, double lat_deg, uint64_t seed, const Config& c) {
    (void)seed;
    Site s;
    s.res_m = res_m, s.H = height.H, s.W = height.W, s.lat_deg = lat_deg;
    s.x0 = -0.5 * s.W * res_m, s.y0 = 0.5 * s.H * res_m;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    s.height = GridF(s.H, s.W, 0.0f);
    s.water_level = GridF(s.H, s.W, nan);
    s.water = Grid<uint8_t>(s.H, s.W, WATER_NONE);
    s.sky = Mask(s.H, s.W, 0);
    s.farmland = Mask(s.H, s.W, 0);
    s.flood = Mask(s.H, s.W, 0);
    s.landcover = Grid<uint8_t>(s.H, s.W, LC_GRASS);
    s.island = Grid<int16_t>(s.H, s.W, 0);
    const double farm_slope = std::tan(c.get("synth.farm_max_slope_deg") * 3.141592653589793 / 180.0);
    for (int i = 0; i < s.H; ++i)
        for (int j = 0; j < s.W; ++j) {
            const double h = height(i, j);
            if (std::isnan(h)) {
                s.sky(i, j) = 1;
                s.island(i, j) = -1;
                continue;
            }
            s.height(i, j) = static_cast<float>(h);
            if (water && (*water)(i, j)) {
                s.water(i, j) = WATER_LAKE;
                s.water_level(i, j) = static_cast<float>(h);
            }
        }
    for (int i = 1; i + 1 < s.H; ++i)
        for (int j = 1; j + 1 < s.W; ++j) {
            const double gx = (s.height(i, j + 1) - s.height(i, j - 1)) / (2.0 * res_m);
            const double gy = (s.height(i - 1, j) - s.height(i + 1, j)) / (2.0 * res_m);
            if (std::hypot(gx, gy) < farm_slope && !s.sky(i, j)) s.farmland(i, j) = 1;
        }
    finish_site(s, c);
    return s;
}

}  // namespace skyisle::town
