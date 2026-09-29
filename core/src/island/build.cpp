// build_terrain（skyisle_gen/island/__init__.py 的 build_terrain + _fit_territory 同式）。
#include "skyisle/island/build.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "skyisle/island/layout.hpp"
#include "skyisle/island/terrain.hpp"
#include "skyisle/island/territory.hpp"

namespace skyisle::island {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct ShapeSet {
    std::vector<Shape> shapes;
    std::vector<std::vector<double>> profiles;
    std::vector<std::array<double, 2>> offsets;
};

// 势力范围：照旧摆好的布局若越过与邻群的分界线，就把主岛挪进来（放不下就转走向、拉长）、其余岛带约束重摆。
// 没越界时什么都不动。返回是否重摆；重摆时改 centers 与 ss 的主岛那一项。
void fit_territory(const NodeInputs& inp, const PlanetView& pv, const Config& c, ShapeSet& ss, std::vector<std::array<double, 2>>& centers,
                   const std::vector<double>& sizes, const std::vector<double>& elong, const std::vector<double>& thetas, double res_km,
                   double axis, double kernel, int btype, Group& g) {
    TerritoryRec& rec = g.territory;
    const bool enabled = c.get("territory.enabled", 1.0) != 0.0;
    rec.gap_km = c.get("territory.gap_km", 3.0);
    if (enabled) g.lim = limits(pv, inp.node, rec.gap_km, c.get("territory.reach", 3.0), c.get("territory.reach_km", 40.0));
    const std::vector<Limit>& lim = g.lim;
    rec.neighbours = static_cast<int>(lim.size());
    rec.constrained = false;
    if (lim.empty()) return;
    const int n = static_cast<int>(ss.shapes.size());
    std::vector<std::vector<double>> sup(n);
    for (int k = 0; k < n; ++k) sup[k] = mask_support(ss.shapes[k], ss.offsets[k][0], ss.offsets[k][1], lim, res_km);
    double before = -INF;
    for (int k = 0; k < n; ++k) before = std::max(before, violation(centers[k][0], centers[k][1], sup[k], lim));
    if (before <= 0.0) return;
    const int turns = std::max(1, static_cast<int>(c.get("territory.turns", 8.0)));
    const std::vector<double> stretches = c.list("territory.stretch", {1.0, 1.6, 2.4});
    struct Cand {
        double f;
        int m;
        Shape shp;
        std::array<double, 2> off;
        std::vector<double> prof, s0;
        double ox, oy, v;
    };
    std::vector<Cand> cands;
    int chosen = -1;
    for (double f : stretches) {
        const size_t tier0 = cands.size();
        for (int m = 0; m < turns; ++m) {
            Cand cd;
            cd.f = f;
            cd.m = m;
            if (f == 1.0 && m == 0) {
                cd.shp = ss.shapes[0];
                cd.off = ss.offsets[0];
                cd.prof = ss.profiles[0];
            } else {
                const double th = thetas[0] + m * PI / turns;
                Rng r = part_rng(inp, "shape:0");
                cd.shp = island_shape(r, sizes[0], res_km, elong[0] * f, th, c);
                radial_profile(cd.shp.mask, res_km, cd.off, cd.prof);
            }
            cd.s0 = mask_support(cd.shp, cd.off[0], cd.off[1], lim, res_km);
            cd.v = nearest_fit(cd.s0, lim, cd.ox, cd.oy);
            const bool stop = f == 1.0 && m == 0 && cd.v <= 0.0;
            cands.push_back(std::move(cd));
            if (stop) break;
        }
        double best = INF;
        for (size_t q = tier0; q < cands.size(); ++q)
            if (cands[q].v <= 0.0) {
                const double hyp = np_hypot(cands[q].ox, cands[q].oy);
                if (hyp < best) {
                    best = hyp;
                    chosen = static_cast<int>(q);
                }
            }
        if (chosen >= 0) break;
    }
    if (chosen < 0) {
        double best = INF;
        for (size_t q = 0; q < cands.size(); ++q)
            if (cands[q].v < best) {
                best = cands[q].v;
                chosen = static_cast<int>(q);
            }
    }
    Cand& ch = cands[chosen];
    ss.shapes[0] = ch.shp;
    ss.offsets[0] = ch.off;
    ss.profiles[0] = ch.prof;
    sup[0] = ch.s0;
    std::vector<std::vector<double>> psup(n);
    for (int k = 0; k < n; ++k) psup[k] = profile_support(ss.profiles[k], lim, res_km);
    TerritoryPlace tp;
    tp.lim = &lim;
    tp.support = &psup;
    tp.main_x = ch.ox;
    tp.main_y = ch.oy;
    tp.gap_min_km = c.get("territory.inner_gap_min_km", 0.3);
    Rng rp = part_rng(inp, "place");
    centers = place_islands(rp, ss.profiles, sizes, axis, kernel, btype, c, &tp);
    double after = -INF;
    for (int k = 0; k < n; ++k) after = std::max(after, violation(centers[k][0], centers[k][1], sup[k], lim));
    rec.constrained = true;
    rec.off_x = ch.ox;
    rec.off_y = ch.oy;
    rec.turn_deg = ch.m * 180.0 / turns;
    rec.stretch = ch.f;
    rec.before = before;
    rec.after = after;
}

}  // namespace

Group build_terrain(const NodeInputs& inp, const PlanetView& pv, const Config& c, double res_m, int threads) {
    const double t_start = now_s();
    Group g;
    g.inp = inp;
    boundary_axis(pv, inp.lat, inp.lon, g.axis, g.kernel, g.btype);
    const double axis = g.axis, kernel = g.kernel;
    const int btype = g.btype;
    Rng rng_l = part_rng(inp, "layout");
    int n = island_count(rng_l, inp.area_km2, inp.area_median_km2, c);
    const std::vector<double> sizes = zipf_sizes(inp.area_km2, inp.main_area_km2, n, c);
    n = static_cast<int>(sizes.size());
    g.n = n;
    const std::vector<double> surfs = surface_heights(rng_l, n, inp.height_m, inp.layered, c);
    std::vector<double> ages(n), elong(n), thetas(n);
    const double jit = c.get("layout.age_jitter");
    for (int k = 0; k < n; ++k) ages[k] = clip(inp.age + rng_l.normal(0.0, jit), 0.0, 1.0);
    ages[0] = inp.age;
    const double emax = c.get("terrain.elongation_max");
    for (int k = 0; k < n; ++k) elong[k] = rng_l.uniform(1.0, emax);
    const double tsig = kernel > 0.3 ? 0.35 : 1.2;
    for (int k = 0; k < n; ++k) thetas[k] = axis + rng_l.normal(0.0, tsig);
    Rng rng_r = part_rng(inp, "relief");
    const std::vector<double> reliefs = relief_targets(rng_r, sizes, ages, c);
    // 浮高（四点二十八）：其余岛整座上下平移 δ（主岛 0），另一条随机流；岸缘下限等地形拟合完再夹。
    // 配置里没有 float 段（旧的展平配置）= 不浮，与改前逐位相同
    const bool float_on = c.get("float.enabled", 0.0) != 0.0;
    std::vector<double> floats(n, 0.0);
    if (float_on) {
        Rng rng_f = part_rng(inp, "float");
        floats = float_offsets(rng_f, ages, c);
    }
    const double keel = inp.keel_clearance_m;

    const double res0 = res_m > 0 ? res_m : c.get("res_m");
    const double grid_max = c.get("grid_max");
    // 先用圆形剖面粗放一遍估算群外框，选定分辨率（超过 grid_max 就加倍）
    std::vector<double> r_eff(n);
    for (int k = 0; k < n; ++k) r_eff[k] = 1.2 * std::sqrt(sizes[k] / PI);
    std::vector<std::vector<double>> pre_prof(n);
    for (int k = 0; k < n; ++k) pre_prof[k].assign(72, r_eff[k]);
    std::vector<std::array<double, 2>> pre_c;
    {
        Rng rp = part_rng(inp, "place");
        pre_c = place_islands(rp, pre_prof, sizes, axis, kernel, btype, c);
    }
    double ax_hi = -INF, ax_lo = INF, ay_hi = -INF, ay_lo = INF;
    for (int k = 0; k < n; ++k) {
        ax_hi = std::max(ax_hi, pre_c[k][0] + r_eff[k]);
        ax_lo = std::min(ax_lo, pre_c[k][0] - r_eff[k]);
        ay_hi = std::max(ay_hi, pre_c[k][1] + r_eff[k]);
        ay_lo = std::min(ay_lo, pre_c[k][1] - r_eff[k]);
    }
    const double ext_km = std::max(ax_hi - ax_lo, ay_hi - ay_lo) + 2.0 * c.get("margin_km");
    double res_km = res0 / 1000.0;
    while (ext_km / res_km > 0.95 * grid_max) res_km *= 2.0;
    g.res_km = res_km;

    ShapeSet ss;
    ss.shapes.resize(n);
    ss.profiles.resize(n);
    ss.offsets.resize(n);
    parallel_for(n, threads, [&](int k) {
        Rng rs = part_rng(inp, "shape:" + std::to_string(k));
        ss.shapes[k] = island_shape(rs, sizes[k], res_km, elong[k], thetas[k], c);
        radial_profile(ss.shapes[k].mask, res_km, ss.offsets[k], ss.profiles[k]);
    });
    std::vector<std::array<double, 2>> centers;
    {
        Rng rp = part_rng(inp, "place");
        centers = place_islands(rp, ss.profiles, sizes, axis, kernel, btype, c);
    }
    fit_territory(inp, pv, c, ss, centers, sizes, elong, thetas, res_km, axis, kernel, btype, g);
    g.sec_layout = now_s() - t_start;

    // 贴图
    std::vector<double> halves(n), gcx(n), gcy(n);
    double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;
    for (int k = 0; k < n; ++k) {
        halves[k] = ss.shapes[k].mask.H * res_km / 2.0;
        gcx[k] = centers[k][0] - ss.offsets[k][0];
        gcy[k] = centers[k][1] - ss.offsets[k][1];
        xmin = std::min(xmin, gcx[k] - halves[k]);
        xmax = std::max(xmax, gcx[k] + halves[k]);
        ymin = std::min(ymin, gcy[k] - halves[k]);
        ymax = std::max(ymax, gcy[k] + halves[k]);
    }
    double x0 = std::floor(xmin / res_km) * res_km;
    double y0 = std::ceil(ymax / res_km) * res_km;
    int W = static_cast<int>(std::ceil((xmax - x0) / res_km)) + 1;
    int H = static_cast<int>(std::ceil((y0 - ymin) / res_km)) + 1;
    GridD height(H, W, NaN);
    Grid<int16_t> island_id(H, W, -1);
    const double t_paste = now_s();
    std::vector<Sculpt> sc(n);
    std::vector<double> keels(n);
    const double ksf = c.get("terrain.keel_surface_frac"), cmin = c.get("terrain.cliff_min_m");
    for (int k = 0; k < n; ++k) keels[k] = std::min(keel, ksf * surfs[k]);
    // 多核嵌合（P4）：每座岛一条随机流 island:<节点>:cores:<岛号>，按面积、岛龄、板块边界定（别的抽样次序不动）
    std::vector<CoreSpec> specs(n);
    for (int k = 0; k < n; ++k) {
        Rng rc = part_rng(inp, "cores:" + std::to_string(k));
        specs[k] = multicore_spec(rc, sizes[k], ages[k], kernel, btype, c);
    }
    parallel_for(n, threads, [&](int k) {
        Rng rt = part_rng(inp, "terrain:" + std::to_string(k));
        sc[k] = sculpt_island(rt, ss.shapes[k], ages[k], sizes[k], res_km, surfs[k], reliefs[k], keels[k] + cmin, k == 0, c,
                              specs[k].on ? &specs[k] : nullptr);
    });
    g.rims.assign(n, 0.0);
    g.islands.resize(n);
    g.masks_pos.resize(n);
    const double rim_floor = float_on ? c.get("float.rim_floor_m") : 0.0, down_max = float_on ? c.get("float.down_max_m") : 1.0;
    for (int k = 0; k < n; ++k) {
        const Shape& s = ss.shapes[k];
        const int m = s.mask.H;
        const int c0 = static_cast<int>(std::nearbyint((gcx[k] - (m - 1) / 2.0 * res_km - x0) / res_km));
        const int r0 = static_cast<int>(std::nearbyint((y0 - (gcy[k] + (m - 1) / 2.0 * res_km)) / res_km));
        // 浮高：整座平移（高程、岸缘、峰、台面、岛底一起），在水系、地表、资源、气温、聚落之前
        double fl = 0.0;
        if (float_on && k > 0) {
            fl = floats[k];
            // 往下的按岸缘离下限的余量缩（余量 ≥ down_max_m 不缩）：低台面的群往下挪得少，岸缘不到下限、也不在下限上堆一摞
            if (fl < 0.0) fl *= std::min(1.0, std::max(0.0, (sc[k].rim - rim_floor) / down_max));
            fl = std::max(fl, rim_floor - sc[k].rim);              // 岸缘 + δ ≥ rim_floor_m（兜底）
            for (double& v : sc[k].h.v) v += fl;                    // 掩膜外是 NaN，加了还是 NaN
            sc[k].rim += fl;
            sc[k].peak += fl;
        }
        g.rims[k] = sc[k].rim;
        MaskPos mp;
        mp.mask = Mask(m, m, 0);
        mp.r0 = r0;
        mp.c0 = c0;
        int64_t put_n = 0;
        for (int i = 0; i < m; ++i) {
            const int gi = r0 + i;
            if (gi < 0 || gi >= H) continue;
            for (int j = 0; j < m; ++j) {
                const int gj = c0 + j;
                if (gj < 0 || gj >= W || !s.mask(i, j)) continue;
                if (!std::isnan(height(gi, gj))) continue;
                height(gi, gj) = sc[k].h(i, j);
                island_id(gi, gj) = static_cast<int16_t>(k);
                mp.mask(i, j) = 1;
                ++put_n;
            }
        }
        g.masks_pos[k] = std::move(mp);
        IslandRec& rec = g.islands[k];
        rec.id = k;
        rec.area_cells = put_n;
        rec.area_target = sizes[k];
        rec.cx = centers[k][0];
        rec.cy = centers[k][1];
        rec.surface = surfs[k] + fl;
        rec.relief_target = reliefs[k];
        rec.rim = sc[k].rim;
        rec.peak = sc[k].peak;
        rec.keel = keels[k] + fl;
        rec.fl = fl;
        rec.age = ages[k];
        rec.kind = sc[k].kind;
        rec.cores = sc[k].cores;
        rec.gcx = gcx[k];
        rec.gcy = gcy[k];
        rec.r0 = r0;
        rec.c0 = c0;
        rec.m = m;
        rec.rim_j = pyround(rec.rim, 1);
        rec.keel_j = pyround(rec.keel, 1);
        rec.age_j = pyround(rec.age, 3);
    }
    // 裁到陆地外框 + 边距
    int rlo_l = H, rhi_l = -1, clo_l = W, chi_l = -1;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            if (island_id(i, j) >= 0) {
                rlo_l = std::min(rlo_l, i);
                rhi_l = std::max(rhi_l, i);
                clo_l = std::min(clo_l, j);
                chi_l = std::max(chi_l, j);
            }
    const int mg = static_cast<int>(std::ceil(c.get("margin_km") / res_km));
    const int r_lo = std::max(0, rlo_l - mg), r_hi = std::min(H, rhi_l + mg + 1);
    const int c_lo = std::max(0, clo_l - mg), c_hi = std::min(W, chi_l + mg + 1);
    g.H = r_hi - r_lo;
    g.W = c_hi - c_lo;
    g.height = GridD(g.H, g.W);
    g.island_id = Grid<int16_t>(g.H, g.W);
    for (int i = 0; i < g.H; ++i)
        for (int j = 0; j < g.W; ++j) {
            g.height(i, j) = height(i + r_lo, j + c_lo);
            g.island_id(i, j) = island_id(i + r_lo, j + c_lo);
        }
    x0 += c_lo * res_km;
    y0 -= r_lo * res_km;
    g.x0 = x0;
    g.y0 = y0;
    g.origin_x = pyround(x0, 3);
    g.origin_y = pyround(y0, 3);
    for (int k = 0; k < n; ++k) {
        g.masks_pos[k].r0 -= r_lo;
        g.masks_pos[k].c0 -= c_lo;
        g.islands[k].r0 -= r_lo;
        g.islands[k].c0 -= c_lo;
    }
    // 岸线间距 → 短渡（岛心按 island.json 里四舍五入的 center_km，同 Python 版；P5 起没有索桥与导水槽）
    std::vector<std::array<double, 2>> ctr_cells(n);
    std::vector<double> radii(n);
    for (int k = 0; k < n; ++k) {
        ctr_cells[k] = {(pyround(centers[k][0], 3) - x0) / res_km, -(pyround(centers[k][1], 3) - y0) / res_km};
        radii[k] = *std::max_element(ss.profiles[k].begin(), ss.profiles[k].end());
    }
    const auto gaps = shoreline_gaps(g.masks_pos, res_km, ctr_cells, radii, c.get("layout.ferry_max_km"));
    links(gaps, g.rims, n, g.links);
    Mask land(g.H, g.W, 0);
    for (size_t k = 0; k < land.size(); ++k) land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
    const Mask er = binary_erode(land, c.geti("terrain.cliff_cells"));
    g.cliff = Mask(g.H, g.W, 0);
    for (size_t k = 0; k < land.size(); ++k) g.cliff.v[k] = (land.v[k] && !er.v[k]) ? 1 : 0;
    g.node_kind = age_class(inp.age, c);
    if (!g.lim.empty()) {
        g.territory.has_violation = true;
        g.territory.violation = raster_violation(g, g.lim);
    }
    g.sec_paste = now_s() - t_paste;
    g.sec_total = now_s() - t_start;
    return g;
}

}  // namespace skyisle::island
