// 特殊的山与地貌（PLAN-NATURE B3，DESIGN-NOTES 四点四十六）：按成因条件出，不设配额。见 landforms.hpp。
#include "skyisle/island/landforms.hpp"

#include <algorithm>
#include <cmath>

#include "skyisle/flow.hpp"
#include "skyisle/island/climate.hpp"
#include "skyisle/island/resources.hpp"

namespace skyisle::island {

namespace {

double bearing_deg(double dx, double dy) {   // 罗盘方位（0 = 北、顺时针）；dx 向东、dy 向北
    double b = std::atan2(dx, dy) * 180.0 / PI;
    return b < 0 ? b + 360.0 : b;
}

bool edge_of(const Mask& m, int i, int j) {
    for (int d = 0; d < 4; ++d) {
        const int a = i + N4[d][0], b = j + N4[d][1];
        if (a < 0 || b < 0 || a >= m.H || b >= m.W || !m(a, b)) return true;
    }
    return false;
}

double median_of(std::vector<double> v) { return v.empty() ? 0.0 : np_median(v); }

// ---------------------------------------------------------------- 峰林
// 老岛、沉积盖层出露、暖（该格年均 ≥ karst_temp_c）湿（群的年雨 ≥ karst_rain_mm）：海相石灰岩在湿热处溶蚀，平地上立起石柱。
// 生成器压低峰间的平地（方窗 karst_window_km 里的最低 + karst_keep × 高出的量），石柱一两百米、比格子小，只记范围、密度、高（游戏建）
void karst(const LandformEnv& e, const Config& c, double res_km, Shape& s, Sculpt& sc) {
    if (sc.kind != OLD || !sc.strat.on || e.P_mm < c.get("landform.karst_rain_mm")) return;
    const int n = s.mask.H;
    const size_t N = s.mask.size();
    const double tmin = c.get("landform.karst_temp_c");
    const double hlow = sc.rim + c.get("landform.karst_low_frac") * (sc.peak - sc.rim);   // 峰间平地在低处：高出岸缘不到起伏的 karst_low_frac
    Mask cand(n, n, 0);
    bool any = false;
    for (size_t q = 0; q < N; ++q) {
        if (!s.mask.v[q]) continue;
        const double T = e.t_sea - e.lapse * std::max(sc.h.v[q], 0.0) / 1000.0;
        const uint8_t li = lith_at(sc.h.v[q], sc.top.v[q], sc.skel.v[q], sc.strat);
        cand.v[q] = (lith_sediment(li) && T >= tmin && sc.h.v[q] <= hlow) ? 1 : 0;
        any = any || cand.v[q];
    }
    if (!any) return;
    // 隔 karst_merge_km 以内的碎片算一片（按膨胀后的连通块分组，压平只压候选格）
    GridI lab;
    const Mask grown = binary_dilate(cand, std::max(1, static_cast<int>(std::nearbyint(c.get("landform.karst_merge_km") / res_km))));
    const int nl = label_components(grown, 8, lab);
    std::vector<int64_t> cnt(nl + 1, 0);
    for (size_t q = 0; q < N; ++q)
        if (cand.v[q]) cnt[lab.v[q]]++;   // 面积只数候选格
    const double cell_km2 = res_km * res_km, amin = c.get("landform.karst_min_km2");
    const int r = std::max(1, static_cast<int>(std::nearbyint(c.get("landform.karst_window_km") / res_km)));
    GridD hin(n, n, NaN);
    for (size_t q = 0; q < N; ++q)
        if (cand.v[q]) hin.v[q] = sc.h.v[q];
    GridD hi, lo;
    window_extrema(hin, r, cand, hi, lo);
    const double keep = c.get("landform.karst_keep");
    const std::vector<double> th = c.list("landform.karst_tower_m", {50.0, 250.0});
    std::vector<std::vector<int32_t>> cells(nl + 1);
    for (size_t q = 0; q < N; ++q)
        if (cand.v[q] && cnt[lab.v[q]] * cell_km2 >= amin) cells[lab.v[q]].push_back(static_cast<int32_t>(q));
    for (int L = 1; L <= nl; ++L) {
        if (cells[L].empty()) continue;
        std::vector<double> rel;
        double si = 0, sj = 0;
        for (int32_t q : cells[L]) {
            const double d = sc.h.v[q] - lo.v[q];
            if (d > 10.0) rel.push_back(d);
            si += q / n;
            sj += q % n;
        }
        for (int32_t q : cells[L]) sc.h.v[q] = std::max(sc.rim, lo.v[q] + keep * (sc.h.v[q] - lo.v[q]));
        LandformRec R;
        R.kind = "tower_karst";
        R.r = si / cells[L].size() + 0.5;
        R.c = sj / cells[L].size() + 0.5;
        R.attr.set("area_km2", pyround(static_cast<double>(cells[L].size()) * cell_km2, 2));
        R.attr.set("tower_h_m", pyround(clip(median_of(rel), th[0], th[1]), 0));
        R.attr.set("density_per_km2", pyround(c.get("landform.karst_density_km2") * clip(e.P_mm / 1500.0, 0.5, 1.5), 1));
        R.attr.set("rain_mm", pyround(e.P_mm, 0));
        sc.lf.push_back(std::move(R));
    }
}

// ---------------------------------------------------------------- 冰斗、冰川、角峰
// 峰高于雪线（平衡线：台面处最暖一季的气温按直减率降到 glacier_summer_c 的高度，湿的地方高一些的气温也积得住雪）：
// 雪线附近背阳（朝极）的沟头刨出圈谷（碗底 = 沟头 − cirque_depth_m，碗壁按 (d / r)³ 升到沟头 + 一个深），
// 三面以上被圈谷围住的峰是角峰；雪线以上是冰川（只记范围）
void glaciers(const LandformEnv& e, const Config& c, double res_km, double pole_y, Shape& s, Sculpt& sc) {
    const double min_above = c.get("landform.cirque_min_above_m");
    if (!(sc.peak > e.ela + min_above)) return;
    const int n = s.mask.H;
    const size_t N = s.mask.size();
    const double res_m = res_km * 1000.0, cell_km2 = res_km * res_km;
    // 一大半的地在雪线以上是冰盖（冰越过岸缘断进云海），没有山岳冰川的冰斗、角峰：只记冰川
    int64_t n_mask = 0, n_above = 0;
    for (size_t q = 0; q < N; ++q)
        if (s.mask.v[q]) {
            ++n_mask;
            n_above += sc.h.v[q] >= e.ela ? 1 : 0;
        }
    const bool ice_cap = static_cast<double>(n_above) > c.get("landform.ice_cap_frac") * static_cast<double>(n_mask);
    const GridD hf = priority_fill(sc.h, s.mask, 1e-3);
    const FlowDir fd = d8(hf, s.mask, res_m);
    const GridD A = accumulate(s.mask, fd);
    const GridD sl = slope_deg(sc.h, s.mask, res_m);
    const double below = c.get("landform.cirque_band_below_m"), above = c.get("landform.cirque_band_above_m");
    const double amin = c.get("landform.cirque_acc_min_km2"), amax = c.get("landform.cirque_acc_max_km2"), smin = c.get("landform.cirque_slope_min_deg");
    std::vector<int32_t> cand;
    for (size_t q = 0; q < N && !ice_cap; ++q) {
        if (!s.mask.v[q] || fd.ri[q] < 0) continue;
        const double h = sc.h.v[q], a = A.v[q] * cell_km2;
        if (h < e.ela - below || h > e.ela + above || a < amin || a > amax || sl.v[q] < smin || h - c.get("landform.cirque_depth_m") < sc.rim + 50.0) continue;
        const int i = static_cast<int>(q / n), j = static_cast<int>(q % n);
        const double dx = fd.rj[q] - j, dy = -(fd.ri[q] - i);   // 下坡方向
        if ((dy * pole_y) / std::max(1e-9, np_hypot(dx, dy)) < -0.3) continue;   // 朝阳的坡不留雪
        cand.push_back(static_cast<int32_t>(q));
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return sc.h.v[a] > sc.h.v[b]; });
    const double sep = c.get("landform.cirque_sep_km") / res_km, rad = c.get("landform.cirque_radius_km") / res_km;
    const double depth = c.get("landform.cirque_depth_m");
    std::vector<std::array<double, 2>> centers;
    const int rr = static_cast<int>(std::ceil(rad));
    for (int32_t q : cand) {
        const int i = static_cast<int>(q / n), j = static_cast<int>(q % n);
        bool ok = true;
        for (const auto& ct : centers)
            if (np_hypot(ct[0] - i, ct[1] - j) < sep) {
                ok = false;
                break;
            }
        if (!ok) continue;
        const double hc = sc.h.v[q], zf = hc - depth;
        double head = hc;
        // 围椅形：只刨沟头上坡的那半边（顺下坡方向 > 0.3 r 的不动），碗朝下坡开口
        const double ddi = fd.ri[q] - i, ddj = fd.rj[q] - j, dl = std::max(1e-9, np_hypot(ddi, ddj));
        for (int a = std::max(0, i - rr); a <= std::min(n - 1, i + rr); ++a)
            for (int b = std::max(0, j - rr); b <= std::min(n - 1, j + rr); ++b) {
                const size_t p = static_cast<size_t>(a) * n + b;
                if (!s.mask.v[p]) continue;
                const double d = np_hypot(a - i, b - j) / rad;
                if (d > 1.0 || ((a - i) * ddi + (b - j) * ddj) / dl > 0.3 * rad) continue;
                head = std::max(head, sc.h.v[p]);
                sc.h.v[p] = std::max(sc.rim, std::min(sc.h.v[p], zf + 2.0 * depth * d * d * d));
            }
        centers.push_back({static_cast<double>(i), static_cast<double>(j)});
        LandformRec R;
        R.kind = "cirque";
        R.r = i + 0.5;
        R.c = j + 0.5;
        R.attr.set("radius_km", pyround(rad * res_km, 2));
        R.attr.set("floor_m", pyround(zf, 0));
        R.attr.set("headwall_m", pyround(head - zf, 0));
        R.attr.set("aspect_deg", pyround(bearing_deg(fd.rj[q] - j, -(fd.ri[q] - i)), 0));
        R.attr.set("ela_m", pyround(e.ela, 0));
        sc.lf.push_back(std::move(R));
    }
    // 冰川：雪线以上的格（一岛记一处）
    int64_t ng = 0;
    size_t top = 0;
    double best = -INF;
    for (size_t q = 0; q < N; ++q)
        if (s.mask.v[q] && sc.h.v[q] >= e.ela) {
            ++ng;
            if (sc.h.v[q] > best) {
                best = sc.h.v[q];
                top = q;
            }
        }
    if (ng > 0) {
        LandformRec R;
        R.kind = "glacier";
        R.r = static_cast<double>(top / n) + 0.5;
        R.c = static_cast<double>(top % n) + 0.5;
        R.attr.set("area_km2", pyround(static_cast<double>(ng) * cell_km2, 2));
        R.attr.set("ela_m", pyround(e.ela, 0));
        R.attr.set("peak_m", pyround(best, 0));
        R.attr.set("n_cirques", static_cast<int64_t>(centers.size()));
        int64_t nm = 0;
        for (uint8_t x : s.mask.v) nm += x;
        (void)nm;
        R.attr.set("ice_cap", static_cast<int64_t>(ice_cap ? 1 : 0));   // 一大半的地在雪线以上：冰盖（冰越过岸缘断进云海）
        sc.lf.push_back(std::move(R));
    }
    // 角峰：雪线以上、horn_radius_km 内最高的峰，周围 horn_radius_km 内有 ≥ horn_min_cirques 个圈谷
    const double hr = c.get("landform.horn_radius_km") / res_km;
    const int need = c.geti("landform.horn_min_cirques");
    if (static_cast<int>(centers.size()) < need) return;
    std::vector<int32_t> peaks;
    for (size_t q = 0; q < N; ++q)
        if (s.mask.v[q] && sc.h.v[q] >= e.ela) peaks.push_back(static_cast<int32_t>(q));
    std::stable_sort(peaks.begin(), peaks.end(), [&](int32_t a, int32_t b) { return sc.h.v[a] > sc.h.v[b]; });
    std::vector<std::array<double, 2>> horns;
    for (int32_t q : peaks) {
        const int i = static_cast<int>(q / n), j = static_cast<int>(q % n);
        bool near = false;
        for (const auto& hp : horns)
            if (np_hypot(hp[0] - i, hp[1] - j) < hr) near = true;
        if (near) continue;
        int k = 0, quad = 0;
        for (const auto& ct : centers)
            if (np_hypot(ct[0] - i, ct[1] - j) <= hr) {
                ++k;
                const double ang = std::atan2(-(ct[0] - i), ct[1] - j);   // 峰 → 冰斗的方向
                quad |= 1 << (static_cast<int>(std::floor((ang + PI) / (PI / 2.0))) & 3);
            }
        horns.push_back({static_cast<double>(i), static_cast<double>(j)});   // 最高的峰占住一圈，别的峰不再算
        const int sides = ((quad >> 0) & 1) + ((quad >> 1) & 1) + ((quad >> 2) & 1) + ((quad >> 3) & 1);
        if (k < need || sides < need || sc.h.v[q] < e.ela + c.get("landform.horn_above_ela_m")) continue;
        LandformRec R;
        R.kind = "horn";
        R.r = i + 0.5;
        R.c = j + 0.5;
        R.attr.set("peak_m", pyround(sc.h.v[q], 0));
        R.attr.set("n_cirques", static_cast<int64_t>(k));
        sc.lf.push_back(std::move(R));
    }
}

// ---------------------------------------------------------------- 临空断山
// 岛的边缘像伸出去的悬臂，太厚太重就崩（第二节第 9 条）：岸边格离它 collapse_radius 内的最高点高出它越多越容易崩，
// 概率 = collapse_p × clip((高差 − collapse_relief_m[0]) / (… [1] − … [0]))，一岛至多一处；崩掉以岸边格为心、半径 r 的半圆（r 让丢掉的面积 ≤ collapse_max_loss），
// 留下围椅形的断壁直落云海（新岸缘就是断壁）
void collapse(Rng& rng, const Config& c, double area_km2, double res_km, Shape& s, Sculpt& sc) {
    const double u = rng.uniform(0.0, 1.0);
    if (area_km2 < c.get("landform.collapse_min_km2")) return;
    const double rc_km = std::min(c.get("landform.collapse_radius_max_km"), std::sqrt(c.get("landform.collapse_max_loss") * 2.0 * area_km2 / PI));
    if (rc_km < c.get("landform.collapse_radius_min_km")) return;
    const int n = s.mask.H;
    const size_t N = s.mask.size();
    const double rc = rc_km / res_km;
    GridD hi, lo;
    window_extrema(sc.h, static_cast<int>(std::ceil(rc)), s.mask, hi, lo);
    double best = -INF;
    int bi = -1, bj = -1;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const size_t q = static_cast<size_t>(i) * n + j;
            if (!s.mask.v[q] || !edge_of(s.mask, i, j)) continue;
            const double rel = hi.v[q] - sc.h.v[q];
            if (rel > best) {
                best = rel;
                bi = i;
                bj = j;
            }
        }
    if (bi < 0) return;
    const std::vector<double> rl = c.list("landform.collapse_relief_m", {600.0, 1500.0});
    const double p = c.get("landform.collapse_p") * clip((best - rl[0]) / (rl[1] - rl[0]), 0.0, 1.0);
    if (!(u < p)) return;
    const double Reff = std::sqrt(area_km2 / PI);
    const double gs = 2.0 / Reff * res_km;   // 岸线场每格的斜率量级（穹 1 − u²，岸边 ≈ 2 / R 每 km）
    int64_t lost = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const size_t q = static_cast<size_t>(i) * n + j;
            const double d = np_hypot(i - bi, j - bj);
            s.phi.v[q] = std::min(s.phi.v[q], (d - rc) * gs);
            if (d <= rc && s.mask.v[q]) {
                s.mask.v[q] = 0;
                ++lost;
            }
        }
    int cnt = 0;
    const Mask keep = largest_component(s.mask, &cnt);
    for (size_t q = 0; q < N; ++q) {
        if (s.mask.v[q] && !keep.v[q]) ++lost;
        s.mask.v[q] = keep.v[q];
        if (!s.mask.v[q]) {
            s.phi.v[q] = std::min(s.phi.v[q], -1e-6);
            sc.h.v[q] = NaN;
            if (!sc.top.v.empty()) {
                sc.top.v[q] = NaN;
                sc.skel.v[q] = NaN;
            }
        } else {
            s.phi.v[q] = std::max(s.phi.v[q], 1e-6);
        }
    }
    // 断壁：新岸缘上离崩塌中心 rc + 2 格以内最高的一格
    double hw = -INF;
    int wi = bi, wj = bj;
    for (int i = std::max(0, bi - static_cast<int>(rc) - 2); i <= std::min(n - 1, bi + static_cast<int>(rc) + 2); ++i)
        for (int j = std::max(0, bj - static_cast<int>(rc) - 2); j <= std::min(n - 1, bj + static_cast<int>(rc) + 2); ++j) {
            const size_t q = static_cast<size_t>(i) * n + j;
            if (!s.mask.v[q] || np_hypot(i - bi, j - bj) > rc + 2.0 || !edge_of(s.mask, i, j)) continue;
            if (sc.h.v[q] > hw) {
                hw = sc.h.v[q];
                wi = i;
                wj = j;
            }
        }
    LandformRec R;
    R.kind = "collapse_scarp";
    R.r = wi + 0.5;
    R.c = wj + 0.5;
    R.attr.set("radius_km", pyround(rc_km, 2));
    R.attr.set("headwall_m", pyround(hw - sc.rim, 0));
    R.attr.set("lost_km2", pyround(static_cast<double>(lost) * res_km * res_km, 2));
    R.attr.set("edge_relief_m", pyround(best, 0));
    R.attr.set("p", pyround(p, 3));
    sc.lf.push_back(std::move(R));
}

}  // namespace

LandformEnv landform_env(const NodeInputs& inp, const PlanetView& pv, const Config& c) {
    LandformEnv e;
    e.t_sea = inp.temp_sea;
    e.lapse = inp.lapse_c_per_km;
    e.P_mm = inp.precip_mm_ref * clip(inp.precip, 0.0, 1.0);
    const SeasonThermal th = season_thermal(inp, pv);
    e.t_warm = th.t_ref.empty() ? inp.temp : *std::max_element(th.t_ref.begin(), th.t_ref.end());
    e.h_ref = inp.height_m;
    e.pole_y = inp.lat < 0 ? -1.0 : 1.0;
    // 平衡线：最暖一季的气温降到 t_ela 的高度；湿的地方雪多，t_ela 高一些（年雨每翻一倍 + glacier_wet_c，夹 [−1, 1.5] 倍）
    const double t_ela = c.get("landform.glacier_summer_c", 1.0) +
                         c.get("landform.glacier_wet_c", 0.0) * clip(std::log2(std::max(e.P_mm, 1.0) / 1000.0), -1.0, 1.5);
    e.ela = e.h_ref + (e.t_warm - t_ela) / std::max(0.1, e.lapse) * 1000.0;
    return e;
}

void landform_pass(Rng& rng, const LandformEnv& e, const Config& c, double area_km2, double res_km, Shape& s, Sculpt& sc) {
    if (c.get("landform.enabled", 0.0) == 0.0) return;
    const bool strat = sc.strat.on;
    if (strat) karst(e, c, res_km, s, sc);
    glaciers(e, c, res_km, e.pole_y, s, sc);
    collapse(rng, c, area_km2, res_km, s, sc);
}

void compute_lith(Group& g) {
    const size_t N = static_cast<size_t>(g.H) * g.W;
    g.lith = Grid<uint8_t>(g.H, g.W, LI_VOID);
    if (g.strat_top.v.empty()) return;
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        g.lith.v[k] = lith_at(g.height.v[k], g.strat_top.v[k], g.skel_top.v[k], g.islands[id].strat);
    }
}

void detect_landforms(Group& g, const Config& c) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, res_m = res_km * 1000.0, cell_km2 = res_km * res_km;
    // 崖层：坡 ≥ rockwall_deg 的格，落差 = 本格 − 8 邻里最低，朝向 = 往那格的方位（16 向）
    g.rockwall_m = GridD(H, W, 0.0);
    g.rockwall_dir = Grid<uint8_t>(H, W, 255);
    const double wall_deg = c.get("landform.rockwall_deg", 40.0);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            if (g.island_id.v[k] < 0 || g.slope.v[k] < wall_deg) continue;
            double lo = g.height.v[k];
            int li = 0, lj = 0;
            for (int d = 0; d < 8; ++d) {
                const int a = i + N8[d][0], b = j + N8[d][1];
                if (a < 0 || b < 0 || a >= H || b >= W || g.island_id(a, b) < 0) continue;
                if (g.height(a, b) < lo) {
                    lo = g.height(a, b);
                    li = N8[d][0];
                    lj = N8[d][1];
                }
            }
            if (li == 0 && lj == 0) continue;
            g.rockwall_m.v[k] = g.height.v[k] - lo;
            const double b = bearing_deg(lj, -li);
            g.rockwall_dir.v[k] = static_cast<uint8_t>(static_cast<int>(std::nearbyint(b / 22.5)) % 16);
        }
    if (c.get("landform.enabled", 0.0) == 0.0 || g.lith.v.empty() || !g.has_resources) return;
    Mask water(H, W, 0);
    for (size_t k = 0; k < N; ++k) water.v[k] = (g.river.v[k] > 0 || g.lake.v[k]) ? 1 : 0;
    auto peak_rel = [&](size_t k) {
        const IslandRec& J = g.islands[g.island_id.v[k]];
        return (g.height.v[k] - J.rim_j) / std::max(1.0, J.peak_j - J.rim_j);
    };
    auto comps = [&](const Mask& m, std::vector<std::vector<int32_t>>& out) {
        GridI lab;
        const int nl = label_components(m, 8, lab);
        out.assign(nl + 1, {});
        for (size_t k = 0; k < N; ++k)
            if (lab.v[k] > 0) out[lab.v[k]].push_back(static_cast<int32_t>(k));
    };
    auto highest = [&](const std::vector<int32_t>& cs) {
        int32_t b = cs[0];
        for (int32_t q : cs)
            if (g.height.v[q] > g.height.v[b]) b = q;
        return b;
    };
    auto rec = [&](const char* kind, int32_t at, const std::vector<int32_t>& cs) -> LandformRec& {
        LandformRec R;
        R.kind = kind;
        R.island = g.island_id.v[at];
        R.r = static_cast<double>(at / W) + 0.5;
        R.c = static_cast<double>(at % W) + 0.5;
        R.attr.set("area_km2", pyround(static_cast<double>(cs.size()) * cell_km2, 2));
        g.landforms.push_back(std::move(R));
        return g.landforms.back();
    };
    // 辉长岩锯齿峰：辉长岩露在山地 / 高山的高处（高出岸缘到峰高的 crag_peak_frac 以上），成片 ≥ crag_min_km2 且坡中位 ≥ crag_slope_deg
    {
        Mask m(H, W, 0);
        const double pf = c.get("landform.crag_peak_frac");
        for (size_t k = 0; k < N; ++k)
            m.v[k] = (g.island_id.v[k] >= 0 && g.lith.v[k] == LI_GABBRO && (g.zone.v[k] == 1 || g.zone.v[k] == 2) && !water.v[k] && peak_rel(k) >= pf) ? 1 : 0;
        std::vector<std::vector<int32_t>> cs;
        comps(m, cs);
        for (size_t L = 1; L < cs.size(); ++L) {
            if (static_cast<double>(cs[L].size()) * cell_km2 < c.get("landform.crag_min_km2")) continue;
            std::vector<double> sl;
            for (int32_t q : cs[L]) sl.push_back(g.slope.v[q]);
            const double med = median_of(sl);
            if (med < c.get("landform.crag_slope_deg")) continue;
            const int32_t top = highest(cs[L]);
            LandformRec& R = rec("gabbro_crags", top, cs[L]);
            R.attr.set("peak_m", pyround(g.height.v[top], 0));
            R.attr.set("slope_med_deg", pyround(med, 1));
        }
    }
    // 蛇纹岩秃山：蛇纹岩露出的地成片 ≥ serp_min_km2（地表在水系里已改成稀草 / 裸岩；铬、镍就在这里）
    {
        Mask m(H, W, 0);
        for (size_t k = 0; k < N; ++k) m.v[k] = (g.island_id.v[k] >= 0 && g.lith.v[k] == LI_SERP && !water.v[k]) ? 1 : 0;
        std::vector<std::vector<int32_t>> cs;
        comps(m, cs);
        for (size_t L = 1; L < cs.size(); ++L) {
            if (static_cast<double>(cs[L].size()) * cell_km2 < c.get("landform.serp_min_km2")) continue;
            int64_t bare = 0;
            double lo = INF, hi = -INF;
            for (int32_t q : cs[L]) {
                bare += g.landcover.v[q] == 2 ? 1 : 0;
                lo = std::min(lo, g.height.v[q]);
                hi = std::max(hi, g.height.v[q]);
            }
            const int32_t top = highest(cs[L]);
            LandformRec& R = rec("serpentine_barren", top, cs[L]);
            R.attr.set("bare_frac", pyround(static_cast<double>(bare) / static_cast<double>(cs[L].size()), 3));
            R.attr.set("elev_lo_m", pyround(lo, 0));
            R.attr.set("elev_hi_m", pyround(hi, 0));
        }
    }
    // 方山 / 孤山：老岛顶上那层硬石灰岩盖（层面下不到 t_cap）的残块、平顶（坡 < mesa_top_slope_deg），四周多半是陡边（一格内落差 ≥ mesa_drop_m），
    // 整块高出周围（边上 3 格内的落差中位 ≥ mesa_rise_m）、不是贴着岛缘的台地本身；≥ mesa_big_km2 叫方山，小的叫孤山
    {
        Mask m(H, W, 0);
        const double ts = c.get("landform.mesa_top_slope_deg");
        for (size_t k = 0; k < N; ++k) {
            const int id = g.island_id.v[k];
            if (id < 0 || g.strat_top.v.empty()) continue;
            const StratRec& st = g.islands[id].strat;
            const bool cap = st.on && st.t_cap > 0 && (g.strat_top.v[k] - g.height.v[k]) / st.scale < st.t_cap;
            m.v[k] = (g.islands[id].kind == OLD && cap && g.lith.v[k] == LI_LIME && g.slope.v[k] < ts && !water.v[k]) ? 1 : 0;
        }
        std::vector<std::vector<int32_t>> cs;
        comps(m, cs);
        GridI lab;
        label_components(m, 8, lab);
        const double drop_m = c.get("landform.mesa_drop_m"), rise_m = c.get("landform.mesa_rise_m");
        for (size_t L = 1; L < cs.size(); ++L) {
            const double area = static_cast<double>(cs[L].size()) * cell_km2;
            if (area < c.get("landform.mesa_min_km2") || area > c.get("landform.mesa_max_km2")) continue;
            int64_t nb = 0, steep = 0, at_edge = 0;
            std::vector<double> rise;
            const int lb = lab.v[cs[L][0]];
            for (int32_t q : cs[L]) {
                const int i = q / W, j = q % W;
                bool bnd = false, void_nb = false;
                double lo1 = g.height.v[q], lo3 = g.height.v[q];
                for (int d = 0; d < 8; ++d) {
                    const int a = i + N8[d][0], b = j + N8[d][1];
                    if (a < 0 || b < 0 || a >= H || b >= W || g.island_id(a, b) < 0) {
                        void_nb = true;
                        continue;
                    }
                    if (lab(a, b) != lb) {
                        bnd = true;
                        lo1 = std::min(lo1, g.height(a, b));
                    }
                }
                if (!bnd && !void_nb) continue;
                ++nb;
                if (void_nb) {
                    ++at_edge;
                    continue;
                }
                for (int a = std::max(0, i - 3); a <= std::min(H - 1, i + 3); ++a)
                    for (int b = std::max(0, j - 3); b <= std::min(W - 1, j + 3); ++b)
                        if (g.island_id(a, b) >= 0) lo3 = std::min(lo3, g.height(a, b));
                if (g.height.v[q] - lo1 >= drop_m) ++steep;
                rise.push_back(g.height.v[q] - lo3);
            }
            if (nb == 0 || static_cast<double>(at_edge) / nb > 0.3) continue;
            const double steep_frac = static_cast<double>(steep) / static_cast<double>(nb - at_edge);
            const double rmed = median_of(rise);
            if (steep_frac < c.get("landform.mesa_steep_frac") || rmed < rise_m) continue;
            std::vector<double> hs;
            for (int32_t q : cs[L]) hs.push_back(g.height.v[q]);
            const int32_t top = highest(cs[L]);
            LandformRec& R = rec(area >= c.get("landform.mesa_big_km2") ? "mesa" : "butte", top, cs[L]);
            R.attr.set("top_m", pyround(median_of(hs), 0));
            R.attr.set("rise_m", pyround(rmed, 0));
            R.attr.set("steep_frac", pyround(steep_frac, 3));
        }
    }
    // 穿山天窗 / 天生桥：浮石骨架里的空洞被削穿。骨架只在岸崖上露（岩层几百米厚），所以天窗开在岸缘上被削薄的岬角里——
    // 骨架空洞（资源的点）arch_reach_km 内、沿某个方向两侧 arch_spur_cells 格内都是虚空的陆地格（宽 ≤ 2 × arch_spur_cells + 1 格的岬角），
    // 取其中最高的一格：从洞里看得见下面的云。只记点（位置、岬角的走向、洞口大小），游戏建
    {
        const int reach = std::max(1, static_cast<int>(std::nearbyint(c.get("landform.arch_reach_km") / res_km)));
        const int spur = std::max(1, c.geti("landform.arch_spur_cells"));
        const double fin_m = c.get("landform.arch_fin_m");
        static const int AX[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
        auto is_void = [&](int a, int b) { return a < 0 || b < 0 || a >= H || b >= W || g.island_id(a, b) < 0; };
        std::vector<std::array<int, 2>> used;
        const double dsep = c.get("landform.arch_sep_km") / res_km;
        for (const Deposit& d : g.res.deposits) {
            if (d.kind != RK_CAVE || d.subtype != "skeleton_void") continue;
            double best = -INF;
            int bi = -1, bj = -1, bax = 0;
            for (int a = std::max(0, d.ci - reach); a <= std::min(H - 1, d.ci + reach); ++a)
                for (int b = std::max(0, d.cj - reach); b <= std::min(W - 1, d.cj + reach); ++b) {
                    if (g.island_id(a, b) != d.island || water(a, b)) continue;
                    if (g.height(a, b) - g.islands[d.island].rim_j < fin_m) continue;
                    // 岬角：沿轴 x 两侧 spur 格内都是虚空，且沿岬角的走向（垂直于 x）前后至少还有一格也是这样（至少两格长的岩鳍）
                    auto thin = [&](int p, int q, int x) {
                        if (p < 0 || q < 0 || p >= H || q >= W || g.island_id(p, q) < 0) return false;
                        bool v1 = false, v2 = false;
                        for (int s = 1; s <= spur; ++s) {
                            v1 = v1 || is_void(p + s * AX[x][0], q + s * AX[x][1]);
                            v2 = v2 || is_void(p - s * AX[x][0], q - s * AX[x][1]);
                        }
                        return v1 && v2;
                    };
                    for (int x = 0; x < 4; ++x) {
                        const int px = -AX[x][1], py = AX[x][0];   // 垂直于 x 的方向
                        if (thin(a, b, x) && (thin(a + py, b + px, x) || thin(a - py, b - px, x)) && g.height(a, b) > best) {
                            best = g.height(a, b);
                            bi = a;
                            bj = b;
                            bax = x;
                        }
                    }
                }
            if (bi < 0) continue;
            bool dup = false;
            for (const auto& u : used) dup = dup || np_hypot(u[0] - bi, u[1] - bj) < dsep;
            if (dup) continue;
            used.push_back({bi, bj});
            const IslandRec& J = g.islands[d.island];
            const double base = g.skel_top.v.empty() ? J.keel_j : g.skel_top(bi, bj);
            LandformRec R;
            R.kind = "natural_arch";
            R.island = d.island;
            R.r = bi + 0.5;
            R.c = bj + 0.5;
            // 岬角的走向垂直于两侧虚空的方向
            R.attr.set("spur_strike_deg", pyround(std::fmod(bearing_deg(AX[bax][1], -AX[bax][0]) + 90.0, 180.0), 0));
            R.attr.set("opening_m", pyround(clip(0.3 * (best - base), c.get("landform.arch_open_min_m"), c.get("landform.arch_open_max_m")), 0));
            R.attr.set("spur_top_m", pyround(best, 0));
            R.attr.set("sees_clouds", static_cast<int64_t>(1));
            R.attr.set("void_id", static_cast<int64_t>(d.id));
            g.landforms.push_back(std::move(R));
        }
    }
}

Json landform_json(const Group& g, const LandformRec& r, int id) {
    Json j = Json::obj();
    j.set("id", static_cast<int64_t>(id));
    j.set("kind", r.kind);
    j.set("island", static_cast<int64_t>(r.island));
    j.set("cell", Json::ipair(static_cast<int64_t>(std::floor(r.r)), static_cast<int64_t>(std::floor(r.c))));
    j.set("km", Json::pair(pyround(g.origin_x + r.c * g.res_km, 3), pyround(g.origin_y - r.r * g.res_km, 3)));
    for (const auto& kv : r.attr.fields()) j.set(kv.first, kv.second);
    return j;
}

}  // namespace skyisle::island
