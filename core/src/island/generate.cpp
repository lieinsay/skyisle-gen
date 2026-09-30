// 整群生成的编排（island.generate 的算法部分）与粗版降采样（lod._block_reduce 同式）。
#include "skyisle/island/generate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "skyisle/island/build.hpp"
#include "skyisle/island/climate.hpp"
#include "skyisle/island/landforms.hpp"
#include "skyisle/island/resources.hpp"
#include "skyisle/island/rivernet.hpp"
#include "skyisle/island/settle.hpp"

namespace skyisle::island {

namespace {
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

Group generate(const NodeInputs& inp, const PlanetView& pv, const Config& c, int year, int steps, double res_m, int threads) {
    Group g = build_terrain(inp, pv, c, res_m, threads);
    finalize_islands(g);
    if (steps >= 2) {
        build_hydro(g, pv, c, threads);
        finalize_islands(g);
        build_resources(g, c);
        detect_landforms(g, c);
    }
    if (steps >= 3) {
        const double t0 = now_s();
        g.clim = build_climate(g.inp, pv, c);
        g.daily = daily_curves(g.clim, g.inp);
        g.has_climate = true;
        g.sec_climate = now_s() - t0;
    }
    if (steps >= 4) {
        const double t0 = now_s();
        g.wparams = season_params(g.clim, c);
        Rng r = part_rng(g.inp, "weather:" + std::to_string(year));
        g.weather = simulate_year(r, g.clim, g.daily, g.wparams, g.inp.height_m, c, g.islands[0].rim_j, g.inp.lapse_c_per_km);
        g.year = year;
        g.has_weather = true;
        g.sec_climate += now_s() - t0;
    }
    if (steps >= 5) build_settlements(g, pv, c);
    if (g.has_weather) build_rivernet(g, c);   // C3：河的数据（逐日径流指数读天气；有聚落时悬沙读已垦）
    return g;
}

LodBlock block_reduce(const Group& g, int f) {
    LodBlock B;
    const int H = g.H, W = g.W;
    const int Hb = (H + f - 1) / f, Wb = (W + f - 1) / f;
    B.H = Hb;
    B.W = Wb;
    B.f = f;
    const size_t nb = static_cast<size_t>(Hb) * Wb, ff = static_cast<size_t>(f) * f;
    B.land.assign(nb, 0);
    B.water.assign(nb, 0);
    B.landcover.assign(nb, 0);
    B.height.assign(nb, 0.0f);
    B.peak.assign(nb, 0.0f);
    B.island.assign(nb, -1);
    int n_isl = 0;
    for (size_t k = 0; k < g.island_id.v.size(); ++k) n_isl = std::max(n_isl, g.island_id.v[k] + 1);
    const int ncls = std::max(1, n_isl);
    std::vector<float> hv(ff);
    std::vector<int> cnt_i(static_cast<size_t>(ncls)), cnt_c(12);
    for (int bi = 0; bi < Hb; ++bi)
        for (int bj = 0; bj < Wb; ++bj) {
            int64_t n_land = 0, n_water = 0;
            float peak = -std::numeric_limits<float>::infinity();
            std::fill(cnt_i.begin(), cnt_i.end(), 0);
            std::fill(cnt_c.begin(), cnt_c.end(), 0);
            size_t q = 0;
            for (int p = 0; p < f; ++p)
                for (int r = 0; r < f; ++r, ++q) {
                    const int i = bi * f + p, j = bj * f + r;
                    const bool in = i < H && j < W;
                    const int id = in ? g.island_id(i, j) : -1;
                    const bool land = id >= 0;
                    // nansum：NaN（非陆地、补的边）按 0 加
                    const float h32 = land ? static_cast<float>(g.height(i, j)) : std::numeric_limits<float>::quiet_NaN();
                    hv[q] = std::isnan(h32) ? 0.0f : h32;
                    if (land) {
                        ++n_land;
                        peak = std::max(peak, h32);
                        if (id < ncls) cnt_i[id]++;
                        const uint8_t lc = g.landcover(i, j);
                        if (lc < 12) cnt_c[lc]++;
                        if (lc == 10 || g.lake(i, j)) ++n_water;   // 地表是河（C1：宽过一格的河道格）或湖；窄河在岸上，不算水面
                    }
                }
            const size_t o = static_cast<size_t>(bi) * Wb + bj;
            B.land[o] = static_cast<uint8_t>(std::nearbyint(255.0 * static_cast<double>(n_land) / static_cast<double>(ff)));
            B.water[o] = static_cast<uint8_t>(std::nearbyint(255.0 * static_cast<double>(n_water) / static_cast<double>(ff)));
            if (n_land > 0) {
                const float s = np_sum_f32(hv.data(), ff);
                B.height[o] = static_cast<float>(static_cast<double>(s) / static_cast<double>(std::max<int64_t>(n_land, 1)));
                B.peak[o] = peak;
                int bk = 0;
                for (int k = 1; k < ncls; ++k)
                    if (cnt_i[k] > cnt_i[bk]) bk = k;
                B.island[o] = static_cast<int16_t>(bk);
                int bc = 0;
                for (int k = 1; k < 12; ++k)
                    if (cnt_c[k] > cnt_c[bc]) bc = k;
                B.landcover[o] = static_cast<uint8_t>(bc);
            } else {
                B.height[o] = std::numeric_limits<float>::quiet_NaN();
                B.peak[o] = std::numeric_limits<float>::quiet_NaN();
                B.island[o] = -1;
                B.landcover[o] = 255;      // Python 版 mode(...) 的 −1 转 uint8
            }
        }
    return B;
}

}  // namespace skyisle::island
