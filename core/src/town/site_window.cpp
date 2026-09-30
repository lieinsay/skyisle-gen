// 岛群窗口细化（PLAN-TOWN 7.1）：100 m 粗栅格 → 1–2 m 细栅格。
// 高程双三次 + 按坡度定振幅的细节噪声（离河近处衰减）；岸线、田、地表的类别边界加扰动去方格；河按中心线与河宽重刻。
// 噪声一律按岛群平面坐标取：同一个地方在任何窗口、任何风格下都是同一块地。
#include <algorithm>
#include <cmath>
#include <limits>

#include "skyisle/town/raster.hpp"
#include "skyisle/town/site.hpp"

namespace skyisle::town {

namespace {

// 虚空格（NaN）填最近陆地格的值；src 同时给出岛号的最近填充
void fill_from_nearest(const GridD& h, const Grid<int16_t>& isl, GridD& hf, Grid<int16_t>& idf) {
    Mask land(h.H, h.W, 0);
    for (size_t k = 0; k < land.size(); ++k) land[k] = std::isnan(h[k]) ? 0 : 1;
    GridF d;
    Grid<int32_t> src;
    edt(land, d, &src);
    hf = GridD(h.H, h.W, 0.0);
    idf = Grid<int16_t>(h.H, h.W, -1);
    for (size_t k = 0; k < land.size(); ++k) {
        const int q = src[k];
        if (q < 0) continue;
        hf[k] = h[static_cast<size_t>(q)];
        idf[k] = isl[static_cast<size_t>(q)];
    }
}

int nearest_cell(double v, int n) { return std::clamp(static_cast<int>(std::floor(v)), 0, n - 1); }

// 地表类别的软投票：四个双线性角按权重投给各自的类别，每类再加一份按绝对坐标取的噪声，取最大——类别边界是弯的，不是 100 m 的方块。
// 虚空、河、湖的角按草坡算（水与岸线另有来源）
uint8_t soft_class(const Grid<uint8_t>& lc, double r, double c, uint64_t seed, double gx, double gy, double noise) {
    const double y = r - 0.5, x = c - 0.5;
    const int i = static_cast<int>(std::floor(y)), j = static_cast<int>(std::floor(x));
    const double ty = y - i, tx = x - j;
    double w[12] = {0};
    const int di[4] = {0, 0, 1, 1}, dj[4] = {0, 1, 0, 1};
    const double wt[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
    for (int k = 0; k < 4; ++k) {
        uint8_t cls = lc(std::clamp(i + di[k], 0, lc.H - 1), std::clamp(j + dj[k], 0, lc.W - 1));
        if (cls == LC_VOID || cls == LC_RIVER || cls == LC_LAKE || cls >= 12) cls = LC_GRASS;
        w[cls] += wt[k];
    }
    uint8_t best = LC_GRASS;
    double bv = -1e9;
    for (int k = 1; k < 12; ++k) {
        if (w[k] <= 0.0) continue;
        const double v = w[k] + noise * fbm(mix64(seed ^ (0x5100 + k)), gx, gy, 60.0, 2);
        if (v > bv) bv = v, best = static_cast<uint8_t>(k);
    }
    return best;
}

}  // namespace

Site build_site_window(const WindowIn& in, const Config& c) {
    Site s;
    s.res_m = in.res_m;
    s.W = s.H = std::max(8, static_cast<int>(std::ceil(2.0 * in.half_m / in.res_m)));
    s.x0 = -0.5 * s.W * s.res_m, s.y0 = 0.5 * s.H * s.res_m;
    s.frame_x = in.frame_x, s.frame_y = in.frame_y;
    s.lat_deg = in.lat_deg;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    s.height = GridF(s.H, s.W, 0.0f);
    s.water_level = GridF(s.H, s.W, nan);
    s.water = Grid<uint8_t>(s.H, s.W, WATER_NONE);
    s.sky = Mask(s.H, s.W, 0);
    s.farmland = Mask(s.H, s.W, 0);
    s.flood = Mask(s.H, s.W, 0);
    s.landcover = Grid<uint8_t>(s.H, s.W, LC_GRASS);
    s.island = Grid<int16_t>(s.H, s.W, -1);

    const GridD& hc = in.height;
    const int H0 = hc.H, W0 = hc.W;
    const double cres = in.coarse_res_m;
    GridD hf;
    Grid<int16_t> idf;
    fill_from_nearest(hc, in.island, hf, idf);
    Mask land0(H0, W0, 0), all0(H0, W0, 1);
    for (size_t k = 0; k < land0.size(); ++k) land0[k] = std::isnan(hc[k]) ? 0 : 1;

    // 细节振幅（粗栅格上按坡度定，离河近处衰减）
    const GridD slope = slope_deg(hf, all0, cres);
    const double amin = c.get("site.detail_min_m"), amax = c.get("site.detail_max_m"), again = c.get("site.detail_slope_gain");
    Mask riv0(H0, W0, 0);
    for (size_t k = 0; k < riv0.size(); ++k) riv0[k] = in.river_depth[k] > 0.0 ? 1 : 0;
    GridF driv;
    edt(riv0, driv, nullptr);
    // 地面不带河槽：粗栅格上河道 / 溪涧格存的是河床（溪涧浅切约 3 m、干流更深），一格宽的槽经双三次会摊成两三百米宽的浅洼。
    // 河道格先取最近的非河道陆地格的高程（岸上的地面），河槽之后在细栅格上按真实河宽另刻（finish_site）。
    GridD hg = hf;
    {
        Mask bank(H0, W0, 0);
        for (size_t k = 0; k < bank.size(); ++k) bank[k] = (land0[k] && !riv0[k]) ? 1 : 0;
        GridF d;
        Grid<int32_t> src;
        edt(bank, d, &src);
        for (size_t k = 0; k < hg.size(); ++k)
            if (riv0[k] && src[k] >= 0) hg[k] = hf[static_cast<size_t>(src[k])];
    }
    const double fade = c.get("site.detail_river_fade_m");
    GridD amp(H0, W0, 0.0);
    for (size_t k = 0; k < amp.size(); ++k) {
        const double a = clip(amin + again * std::tan(slope[k] * PI / 180.0), amin, amax);
        const double near = clip(driv[k] * cres / fade, 0.0, 1.0);
        amp[k] = a * smoothstep(near);
    }
    // 湖面：每个湖格的高程就是湖面（填平），非湖格取最近湖格的值
    GridD lakelev(H0, W0, 0.0);
    {
        GridF d;
        Grid<int32_t> src;
        edt(in.lake, d, &src);
        for (size_t k = 0; k < lakelev.size(); ++k)
            if (src[k] >= 0) lakelev[k] = hf[static_cast<size_t>(src[k])];
    }

    const uint64_t sd = in.seed;
    const double wl = c.get("site.detail_wavelength_m");
    const int oct = c.geti("site.detail_octaves");
    const double coast = c.get("site.coast_noise");
    const double jit = c.get("site.class_jitter_m") / cres;
    const double lake_depth = c.get("site.lake_depth_m");
    const double vote_noise = c.get("site.class_vote_noise");
    const double farm_noise = c.get("site.farm_edge_noise");
    for (int i = 0; i < s.H; ++i)
        for (int j = 0; j < s.W; ++j) {
            const V2 p = s.center(i, j);
            const double r = in.center_r - p.y / cres, cc = in.center_c + p.x / cres;
            const double gx = in.frame_x + p.x, gy = in.frame_y + p.y;
            const double landv = sample_linear(land0, r, cc) + coast * fbm(mix64(sd ^ 0x11), gx, gy, 90.0, 3);
            if (landv < 0.5) {
                s.sky(i, j) = 1;
                s.height(i, j) = nan;
                continue;
            }
            const double h = sample_cubic(hg, r, cc) + sample_linear(amp, r, cc) * fbm(mix64(sd ^ 0x22), gx, gy, wl, oct);
            s.height(i, j) = static_cast<float>(h);
            const double jr = jit * fbm(mix64(sd ^ 0x33), gx, gy, 70.0, 2), jc = jit * fbm(mix64(sd ^ 0x44), gx, gy, 70.0, 2);
            const int ni = nearest_cell(r + jr, H0), nj = nearest_cell(cc + jc, W0);
            s.island(i, j) = idf(nearest_cell(r, H0), nearest_cell(cc, W0));
            s.landcover(i, j) = soft_class(in.landcover, r, cc, sd, gx, gy, vote_noise);
            if (sample_linear(in.arable, r + jr, cc + jc) + farm_noise * fbm(mix64(sd ^ 0x55), gx, gy, 90.0, 3) > 0.5) {
                s.farmland(i, j) = 1;
                if (in.terrace(ni, nj)) s.landcover(i, j) = LC_TERRACE;
            }
            if (sample_linear(in.floodplain, r, cc) + farm_noise * fbm(mix64(sd ^ 0x66), gx, gy, 90.0, 3) > 0.5) s.flood(i, j) = 1;
            if (sample_linear(in.lake, r + 0.5 * jr, cc + 0.5 * jc) > 0.5) {
                const double lev = sample_linear(lakelev, r, cc);
                s.water(i, j) = WATER_LAKE;
                s.water_level(i, j) = static_cast<float>(lev);
                s.height(i, j) = static_cast<float>(std::min(h, lev - lake_depth));
            }
        }

    // 河：顶点处的水面 = 粗格高程（C1 起河道格存的是平岸水面；旧产物存的是河床，水面 = 河床 + 水深），顺流单调不增；再平滑
    const int smooth = c.geti("site.river_smooth");
    for (const River& r0 : in.rivers) {
        if (r0.line.size() < 2) continue;
        River r = r0;
        r.depth_m.assign(r.line.size(), 0.0);
        r.surface_m.assign(r.line.size(), 0.0);
        double run = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < r.line.size(); ++k) {
            const double rr = in.center_r - r.line[k].y / cres, cc = in.center_c + r.line[k].x / cres;
            const int ni = nearest_cell(rr, H0), nj = nearest_cell(cc, W0);
            const double depth = std::max(0.1, in.river_depth(ni, nj));
            const double surf = in.height_is_surface ? hf(ni, nj) : hf(ni, nj) + depth;
            run = std::min(run, surf);
            r.depth_m[k] = depth;
            r.surface_m[k] = run;
        }
        std::vector<std::vector<double>> attrs(r.line.size());
        for (size_t k = 0; k < r.line.size(); ++k) attrs[k] = {r.width_m[k], r.depth_m[k], r.surface_m[k]};
        chaikin(r.line, &attrs, smooth);
        r.width_m.resize(r.line.size());
        r.depth_m.resize(r.line.size());
        r.surface_m.resize(r.line.size());
        for (size_t k = 0; k < r.line.size(); ++k) r.width_m[k] = attrs[k][0], r.depth_m[k] = attrs[k][1], r.surface_m[k] = attrs[k][2];
        s.rivers.push_back(std::move(r));
    }
    finish_site(s, c);
    return s;
}

}  // namespace skyisle::town
