// 集水核（C4）与地下水（C5）：见 groundwater.hpp。
#include "skyisle/island/groundwater.hpp"

#include "skyisle/flow.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace skyisle::island {

namespace {

double wc(const Config& c, const std::string& k) { return c.get("water." + k); }

// 凝结 1 m/年放出的潜热（W/m²）：2.45 MJ/kg × 1000 kg/m³ / 一年的秒数（一年 336 天约 84 W/m²，spec 13 第八节第 7 条）
double latent_w_per_m(double year_s) { return 2.45e6 * 1000.0 / std::max(1.0, year_s); }

// 一格的含水层底：骨架顶面 + 最小含水厚（没有岩层时退回「岛底 + 0.4 × 崖高」，与泉线高程的口径同）
inline double aquifer_floor(const Group& g, const IslandRec& J, size_t k, double min_thick) {
    const double s = (!g.skel_top.v.empty() && !std::isnan(g.skel_top.v[k])) ? g.skel_top.v[k]
                                                                          : J.keel_j + 0.4 * (J.rim_j - J.keel_j);
    return s + min_thick;
}

// 稳态地下水位（B，四点四十九）：∇·(T∇h) = −R 的离散形式——水位 = 「8 邻水位的均值 + 源项」，
// 排水口（河道 / 溪涧 / 湖 / 岸缘）钉在地表，水位高过地表就钉回地表（渗出面），不低于含水层底。
// 在 1/gw_coarse 的粗格上做 SOR（水位面是光滑场、粗格够用；逐岛解，岛之间隔着虚空互不影响），
// 再双线性插值回原分辨率并按每格夹一次（≤ 地表、≥ 含水层底）
GridD water_table(const Group& g, const Config& c, const Mask& land, double min_thick) {
    const int H = g.H, W = g.W;
    GridD wt(H, W, NaN);
    const int step = std::max(1, static_cast<int>(std::lround(wc(c, "gw_coarse"))));
    const double dx = g.res_km * 1000.0 * step;
    const double src = wc(c, "gw_r_over_t") * dx * dx / 4.0;
    const int iters = std::max(1, static_cast<int>(std::lround(wc(c, "gw_iters"))));
    const double omega = wc(c, "gw_sor_omega"), tol = std::max(1e-6, wc(c, "gw_tol_m"));
    // 岸缘 = 邻接虚空的陆地格（也是排水口）
    Mask rim(H, W, 0);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            if (!land.v[k]) continue;
            bool v = false;
            for (int n = 0; n < 8 && !v; ++n) {
                const int a = i - N8[n][0], b = j - N8[n][1];
                if (a < 0 || b < 0 || a >= H || b >= W || !land.v[static_cast<size_t>(a) * W + b]) v = true;
            }
            rim.v[k] = v ? 1 : 0;
        }
    const int n = static_cast<int>(g.islands.size());
    // 每岛的窗口：从 island_id 自己算外接框（地形 + 水系分两次调时，IslandRec 里没有窗口）
    std::vector<int> r0v(n, H), r1v(n, -1), c0v(n, W), c1v(n, -1);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const int id = g.island_id.v[static_cast<size_t>(i) * W + j];
            if (id < 0) continue;
            if (i < r0v[id]) r0v[id] = i;
            if (i > r1v[id]) r1v[id] = i;
            if (j < c0v[id]) c0v[id] = j;
            if (j > c1v[id]) c1v[id] = j;
        }
    std::vector<double> cb, cfl, ch;
    std::vector<uint8_t> cl, cdr;
    for (int id = 0; id < n; ++id) {
        const IslandRec& J = g.islands[id];
        if (r1v[id] < r0v[id] || c1v[id] < c0v[id]) continue;      // 这个岛没有陆地
        const int r0 = r0v[id], c0 = c0v[id], mh = r1v[id] - r0v[id] + 1, mw = c1v[id] - c0v[id] + 1;
        const int mch = (mh + step - 1) / step, mcw = (mw + step - 1) / step;
        const size_t CN = static_cast<size_t>(mch) * mcw;
        cb.assign(CN, NaN);
        cfl.assign(CN, NaN);
        ch.assign(CN, NaN);
        cl.assign(CN, 0);
        cdr.assign(CN, 0);
        for (int bi = 0; bi < mch; ++bi)
            for (int bj = 0; bj < mcw; ++bj) {
                double sb = 0.0, sf = 0.0, sdr = INF, cnt = 0.0;
                bool dr = false;
                for (int qi = 0; qi < step; ++qi)
                    for (int qj = 0; qj < step; ++qj) {
                        const int i = r0 + bi * step + qi, j = c0 + bj * step + qj;
                        if (i < 0 || j < 0 || i >= H || j >= W) continue;
                        const size_t k = static_cast<size_t>(i) * W + j;
                        if (g.island_id.v[k] != id) continue;
                        cnt += 1.0;
                        sb += g.height.v[k];
                        sf += aquifer_floor(g, J, k, min_thick);
                        if (g.river.v[k] > 0 || g.stream.v[k] > 0 || g.lake.v[k] || rim.v[k]) {
                            dr = true;
                            sdr = std::min(sdr, g.height.v[k]);   // 排水口的水头 = 那一格的地表（河口 / 岸缘最低的那格）
                        }
                    }
                if (cnt <= 0.0) continue;
                const size_t q = static_cast<size_t>(bi) * mcw + bj;
                cl[q] = 1;
                cb[q] = dr ? std::min(sdr, sb / cnt) : sb / cnt;
                cfl[q] = sf / cnt;
                cdr[q] = dr ? 1 : 0;
                ch[q] = std::max(cb[q], cfl[q]);
            }
        for (int it = 0; it < iters; ++it) {
            double mx = 0.0;
            for (int bi = 0; bi < mch; ++bi)
                for (int bj = 0; bj < mcw; ++bj) {
                    const size_t q = static_cast<size_t>(bi) * mcw + bj;
                    if (!cl[q]) continue;
                    if (cdr[q]) { ch[q] = cb[q]; continue; }
                    double s = 0.0;
                    int cnb = 0;
                    for (int di = -1; di <= 1; ++di)
                        for (int dj = -1; dj <= 1; ++dj) {
                            if (!di && !dj) continue;
                            const int i = bi + di, j = bj + dj;
                            if (i < 0 || j < 0 || i >= mch || j >= mcw) continue;
                            const size_t p = static_cast<size_t>(i) * mcw + j;
                            if (!cl[p]) continue;
                            s += ch[p];
                            ++cnb;
                        }
                    if (!cnb) continue;
                    double v = s / cnb + src;
                    if (v > cb[q]) v = cb[q];
                    if (v < cfl[q]) v = cfl[q];
                    const double nv = ch[q] + omega * (v - ch[q]);
                    mx = std::max(mx, std::fabs(nv - ch[q]));
                    ch[q] = nv;
                }
            if (mx < tol) break;
        }
        for (int i = std::max(0, r0); i < std::min(H, r0 + mh); ++i)
            for (int j = std::max(0, c0); j < std::min(W, c0 + mw); ++j) {
                const size_t k = static_cast<size_t>(i) * W + j;
                if (g.island_id.v[k] != id) continue;
                const double u = (static_cast<double>(i - r0) + 0.5) / step - 0.5;
                const double v = (static_cast<double>(j - c0) + 0.5) / step - 0.5;
                const double fu = clip(u - std::floor(u), 0.0, 1.0), fv = clip(v - std::floor(v), 0.0, 1.0);
                const int b0i = std::min(mch - 1, std::max(0, static_cast<int>(std::floor(u))));
                const int b0j = std::min(mcw - 1, std::max(0, static_cast<int>(std::floor(v))));
                const int b1i = std::min(mch - 1, b0i + 1), b1j = std::min(mcw - 1, b0j + 1);
                const size_t qs[4] = {static_cast<size_t>(b0i) * mcw + b0j, static_cast<size_t>(b0i) * mcw + b1j,
                                      static_cast<size_t>(b1i) * mcw + b0j, static_cast<size_t>(b1i) * mcw + b1j};
                const double wsz[4] = {(1 - fu) * (1 - fv), (1 - fu) * fv, fu * (1 - fv), fu * fv};
                double acc = 0.0, wsum = 0.0;
                for (int t = 0; t < 4; ++t) {
                    if (!cl[qs[t]]) continue;
                    acc += ch[qs[t]] * wsz[t];
                    wsum += wsz[t];
                }
                double hh = wsum > 0 ? acc / wsum : g.height.v[k];
                hh = std::max(hh, aquifer_floor(g, J, k, min_thick));   // 不穿底（含水层底）
                // 排水口（河道 / 溪涧 / 湖 / 岸缘）：水位就是它自己的地表——河因此是「得水河」（粗格解只能给到粗格的水头）
                if (g.river.v[k] > 0 || g.stream.v[k] > 0 || g.lake.v[k] || rim.v[k]) hh = g.height.v[k];
                hh = std::min(hh, g.height.v[k]);                       // 渗出面：水位不高于地表（后夹，保证不高于地表）
                wt.v[k] = hh;
            }
    }
    return wt;
}

}  // namespace

GridD condensation(const Group& g, const GridD& Gw, const Config& c, std::vector<double>& core_s,
                   std::vector<CoreWaterSource>* sources) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const int n = static_cast<int>(g.islands.size());
    GridD out(H, W, 0.0);
    core_s.assign(n, 0.0);
    if (sources) sources->clear();
    // Opt-in research model, no production default. The time includes unresolved
    // vertical contact / transport; it is a world-setting input, NOT a measured
    // dehumidifier efficiency or a value inferred from desired river widths.
    const double exchange_days = c.get("water.core_exchange_days", 0.0);
    const double footprint_scale = c.get("water.core_footprint_scale", 0.0);
    const bool mountain_domain = c.get("water.core_mountain_domain", 0.0) != 0;
    const double activity = c.get("water.core_activity_per_km_day", NaN);
    if (mountain_domain && (!(std::isfinite(activity) && activity >= 0) || exchange_days != 0 || footprint_scale != 0))
        throw std::invalid_argument("mountain core requires explicit nonnegative activity; do not mix old exchange/footprint modes");
    if (!std::isfinite(footprint_scale) || footprint_scale < 0 || (footprint_scale > 0 && !(exchange_days > 0)))
        throw std::invalid_argument("core footprint requires a positive exchange time and finite nonnegative scale");
    if (!std::isfinite(exchange_days) || exchange_days < 0.0)
        throw std::invalid_argument("water.core_exchange_days must be finite and nonnegative");
    double column_mm = 0.0;
    if (exchange_days > 0.0 || mountain_domain) {
        if (g.inp.water_column_mm.empty())
            throw std::invalid_argument("balanced core requires physical water_column_mm; old climate cache has no column budget");
        for (double q : g.inp.water_column_mm) {
            if (!std::isfinite(q) || q < 0.0)
                throw std::invalid_argument("water_column_mm must be finite and nonnegative");
            column_mm += q;
        }
        column_mm /= static_cast<double>(g.inp.water_column_mm.size());
    }
    const double gain = wc(c, "core_gain");
    if (!(gain > 0.0) && !(exchange_days > 0.0) && !mountain_domain) return out;
    const double cell_km2 = g.res_km * g.res_km;
    // 核的强度跟山走：高出岸缘的山体（km³）的立方根 / core_len_km——大山根深、核大；碎的小岛核弱（spec 13 第八节第 6 条）
    std::vector<double> vol(n, 0.0);
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        vol[id] += std::max(0.0, g.height.v[k] - g.islands[id].rim_j) * cell_km2 / 1000.0;
    }
    const double lref = wc(c, "core_len_km"), smax = wc(c, "core_s_max");
    for (int k = 0; k < n; ++k) core_s[k] = clip(std::cbrt(vol[k]) / lref, 0.0, smax);
    if (footprint_scale > 0 || mountain_domain) {
        // User-confirmed setting: source location/size/range, no mountain-top
        // elevation gate and no rainfall ceiling. Kernel shape, range scale,
        // mountain-load proxy and exchange time are explicit model assumptions.
        // A compact smooth footprint avoids infinite weak supply everywhere.
        if (!(std::isfinite(lref) && lref > 0 && std::isfinite(smax) && smax > 0))
            throw std::invalid_argument("invalid core structural strength scale");
        std::vector<CoreWaterSource> all;
        std::vector<std::vector<size_t>> by_island(n);
        std::vector<double> vx(n, 0.0), vy(n, 0.0);
        for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
            const int id = g.island_id(i, j);
            if (id < 0) continue;
            const double v = std::max(0.0, g.height(i, j)-g.islands[id].rim_j)*cell_km2/1000.0;
            vx[id] += v*(g.origin_x+(j+0.5)*g.res_km);
            vy[id] += v*(g.origin_y-(i+0.5)*g.res_km);
        }
        auto add = [&](int id, double x, double y, double volume) {
            if (!(volume > 0) && !mountain_domain) return;
            const double length = std::cbrt(volume);
            CoreWaterSource s;
            s.island = id; s.x_km = x; s.y_km = y; s.mountain_volume_km3 = volume;
            s.core_index = static_cast<int>(by_island[id].size());
            s.radius_km = footprint_scale*length;
            s.strength = std::min(smax, length/lref);
            by_island[id].push_back(all.size());
            all.push_back(s);
        };
        for (int id = 0; id < n; ++id) {
            if (!(vol[id] > 0)) continue;
            const auto& J = g.islands[id];
            if (J.cores.empty()) add(id, vx[id]/vol[id], vy[id]/vol[id], vol[id]);
            else if (mountain_domain) {
                for (const auto& core : J.cores) add(id, J.gcx+core.load_x, J.gcy+core.load_y, 0);
            } else {
                double total = 0;
                for (const auto& core : J.cores) total += std::max(0.0, core.load);
                if (!(total > 0)) throw std::invalid_argument("multicore supply lacks positive mountain loads");
                for (const auto& core : J.cores)
                    add(id, J.gcx+core.load_x, J.gcy+core.load_y, vol[id]*std::max(0.0, core.load)/total);
            }
        }
        if (mountain_domain) {
            if (g.core_member.size() != N || g.core_neighbor.size() != N || g.core_member_weight.size() != N)
                throw std::invalid_argument("mountain core supply requires the terrain's original ownership weights");
            for (auto& source : all) source.mountain_volume_km3 = 0;
            auto each_share = [&](size_t k, const auto& use) {
                const int id = g.island_id.v[k];
                if (id < 0 || by_island[id].empty()) return;
                const int primary = g.core_member.v[k], secondary = g.core_neighbor.v[k];
                const double weight = g.core_member_weight.v[k];
                if (!(std::isfinite(weight) && weight >= 0 && weight <= 1) || primary < 0 ||
                    primary >= static_cast<int>(by_island[id].size()) ||
                    (weight < 1 && (secondary < 0 || secondary >= static_cast<int>(by_island[id].size()))))
                    throw std::invalid_argument("invalid terrain core ownership");
                use(all[by_island[id][primary]], weight);
                if (weight < 1) use(all[by_island[id][secondary]], 1-weight);
            };
            // Recompute each load from the actually pasted terrain and its
            // existing blend weights. A clipped-away mountain cannot retain
            // a budget, nor can two overlapping cores count the same load twice.
            for (size_t k = 0; k < N; ++k) each_share(k, [&](CoreWaterSource& source, double weight) {
                source.domain_area_km2 += weight*cell_km2;
                source.mountain_volume_km3 += weight*cell_km2*
                    std::max(0.0, g.height.v[k]-g.islands[source.island].rim_j)/1000.0;
            });
            for (auto& source : all) {
                source.strength = std::min(smax, std::cbrt(source.mountain_volume_km3)/lref);
                source.activity_per_km_day = activity;
                // mm * km^3 * (km day)^-1 -> mm km^2/day -> m^3/s.
                // This is an explicit world ability per supported mountain
                // volume, NOT an Earth measurement or a radius multiplier.
                source.capacity_m3s = column_mm*source.mountain_volume_km3*activity*1000.0/86400.0;
            }
            for (size_t k = 0; k < N; ++k) each_share(k, [&](CoreWaterSource& source, double weight) {
                if (!(source.domain_area_km2 > 0)) return;
                const double mm = source.capacity_m3s*g.year_s/(source.domain_area_km2*1000.0)*weight;
                out.v[k] += mm;
                source.condense_m3s += mm*cell_km2*1000.0/g.year_s;
            });
            if (sources) *sources = std::move(all);
            return out;
        }
        const double annual_mm = column_mm/exchange_days*(g.year_s/86400.0);
        for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
            const int id = g.island_id(i, j);
            if (id < 0) continue;
            const double x = g.origin_x+(j+0.5)*g.res_km, y = g.origin_y-(i+0.5)*g.res_km;
            for (size_t s : by_island[id]) {
                auto& source = all[s];
                const double dx = x-source.x_km, dy = y-source.y_km;
                const double u = (dx*dx+dy*dy)/(source.radius_km*source.radius_km);
                if (u >= 1) continue;
                const double mm = annual_mm*source.strength*(1-u)*(1-u);
                out(i, j) += mm;
                source.condense_m3s += mm*cell_km2*1000.0/g.year_s;
            }
        }
        if (sources) *sources = std::move(all);
        return out;
    }
    // 落在核山的迎风高处（凝结是气流被山逼着抬升、贴着林子与岩面过去时截下来的）；跟着湿度走：像除湿器，空气干就凝得少
    const double eexp = wc(c, "core_elev_exp"), wg = wc(c, "core_windward_gain"), hfull = wc(c, "core_hum_full_mm"), hexp = wc(c, "core_hum_exp");
    const double gref = c.get("hydro.windward_ref_m_per_km");
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        const IslandRec& J = g.islands[id];
        const double e = clip((g.height.v[k] - J.rim_j) / std::max(1.0, J.peak_j - J.rim_j), 0.0, 1.0);
        if (exchange_days > 0.0) {
            // Prescribed stable background maintained by matching ocean
            // evaporation (user setting, 2026-10-05). Equal-duration seasons.
            // Keep the existing mountain footprint and annual rain ceiling for
            // this isolated comparison; neither is a physical law. No fog gate,
            // rainfall-as-humidity factor, windward multiplier or legacy gain.
            const double annual = column_mm / exchange_days * (g.year_s / 86400.0)
                                * core_s[id] * np_pow(e, eexp);
            out.v[k] = std::min(g.rain.v[k], annual);
            continue;
        }
        const double wf = Gw.v.empty() ? 1.0 : clip(1.0 + wg * clip(Gw.v[k] / gref, -1.0, 1.0), 0.0, 2.0);
        const double P = g.rain.v[k];
        const double hum = np_pow(clip(P / hfull, 0.0, 1.0), hexp);
        out.v[k] = P * std::min(1.0, gain * core_s[id] * np_pow(e, eexp) * wf * hum);   // 最多到局地雨的 1 倍（spec 13 第八节第 4 条）
    }
    return out;
}

void aquifer(Group& g, const Config& c) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double cell_km2 = g.res_km * g.res_km;
    const bool has_lith = !g.lith.v.empty() && !g.strat_top.v.empty();
    const std::vector<double> bl = c.list("water.bfi", {0.55, 0.25, 0.4, 0.2, 0.0});
    const double bdef = wc(c, "bfi_default");
    // 基流比例按出露岩性（石灰岩岩溶、辉长岩裂隙、泥灰岩与蛇纹岩差、浮石闭孔不透水）；补给 = 雨的径流 × 基流比例 + 凝结水（凝结水全渗进岩层）
    g.bfi = GridD(H, W, 0.0);
    g.recharge = GridD(H, W, 0.0);
    for (size_t k = 0; k < N; ++k) {
        if (g.island_id.v[k] < 0) continue;
        double b = bdef;
        if (has_lith) {
            const uint8_t li = g.lith.v[k];
            b = (li >= 1 && li - 1 < static_cast<int>(bl.size())) ? bl[li - 1] : bdef;
        }
        const double cond = g.condense.v.empty() ? 0.0 : g.condense.v[k];
        const double rr = std::max(0.0, g.runoff.v[k] - cond);
        g.bfi.v[k] = b;
        g.recharge.v[k] = rr * b + cond;
    }
    // ---------- 水位面（B，DESIGN-NOTES 四点四十九）----------
    // 稳态地下水位 ∇·(T∇h) = −R：排水口 = 河道 / 溪涧 / 湖 / 岸缘（固定水头 = 地表），渗出面 = 水位高过地表的格，
    // 底 = 骨架顶面 + 最小含水厚。在 1/gw_coarse 的粗格上解（水位面是光滑场），双线性插值回原分辨率，再按每格夹一次。
    Mask land(H, W, 0);
    for (size_t k = 0; k < N; ++k) land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
    g.wt = water_table(g, c, land, wc(c, "gw_min_thick_m"));
    // 顺流向累计（mm·km²）：**按水位面的梯度**（不是地表）——水沿着潜水面往低处汇；
    // 排水口（河道 / 湖 / 岸缘）上的累计 = 河的基流 / 崖壁泉。拓扑序（上游先加），与线程数无关
    GridD wr(H, W, NaN);
    {
        const bool has_cd = !g.coast_dist.v.empty();
        const double tie = wc(c, "gw_tie_m_per_km") / 1000.0, tcap = wc(c, "gw_tie_km") * 1000.0;
        for (size_t k = 0; k < N; ++k) {
            if (!land.v[k]) continue;
            double t = 0.0;
            if (has_cd) t = tie * std::min(std::max(g.coast_dist.v[k], 0.0), tcap);   // 只为打破平局（水位面是光滑场，平局多）
            wr.v[k] = g.wt.v[k] + t;
        }
    }
    const FlowDir wflow = d8(wr, land, g.res_km * 1000.0);
    g.wt_outlet = Grid<uint8_t>(H, W, 0);
    for (size_t k = 0; k < N; ++k)
        g.wt_outlet.v[k] = (land.v[k] && (wflow.to_void[k] || wflow.ri[k] < 0)) ? 1 : 0;   // 无下游的格 = 水位面上的汇（水位出露的地方）
    GridD wcell(H, W, 0.0);
    for (size_t k = 0; k < N; ++k) wcell.v[k] = g.recharge.v[k] * cell_km2;
    g.recharge_acc = accumulate(land, wflow, &wcell);
    // 崖壁泉线：流向（在水位面上）走到岸边、一路没进河道（河、溪涧）也没进湖的出口格——这片坡的地下水没被河截走，
    // 顺着骨架顶面走到崖边，从崖壁上岩层与浮石的交界渗出来。出口格按岛、按 8 连通成串，长串按绕岛心的方位切成 springline_seg_km 一段
    g.springline.clear();
    Mask ex(H, W, 0);
    for (size_t k = 0; k < N; ++k)
        ex.v[k] = (land.v[k] && wflow.to_void[k] && g.river.v[k] == 0 && g.stream.v[k] == 0 && !g.lake.v[k] &&
                   g.recharge_acc.v[k] > 0.0) ? 1 : 0;
    GridI lab;
    const int nl = label_components(ex, 8, lab);
    std::vector<std::vector<int32_t>> comp(static_cast<size_t>(nl) + 1);
    for (size_t k = 0; k < N; ++k)
        if (lab.v[k] > 0) comp[lab.v[k]].push_back(static_cast<int32_t>(k));
    const int n = static_cast<int>(g.islands.size());
    std::vector<double> ci(n, 0.0), cj(n, 0.0), cn(n, 0.0);
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        ci[id] += static_cast<double>(k / W);
        cj[id] += static_cast<double>(k % W);
        cn[id] += 1.0;
    }
    const int seg = std::max(1, static_cast<int>(std::nearbyint(wc(c, "springline_seg_km") / g.res_km)));
    const double to_ls = 1e6 / std::max(1.0, g.year_s);   // mm·km² / 年 → L/s（1 mm·km² = 1000 m³）
    const double qmin = wc(c, "springline_min_ls");
    const double min_aq = wc(c, "spring_min_aquifer_m");
    const double pct = clip(wc(c, "spring_pct"), 0.0, 1.0), fpct = clip(wc(c, "spring_fall_pct"), 0.0, 1.0);
    const double ske = c.get("strat.skel_edge_frac", 0.4);
    std::vector<SpringSeg> all;                 // 先全收（弥散渗出），再按本岛分位与含水层厚度分等
    for (int L = 1; L <= nl; ++L) {
        std::vector<int32_t>& cs = comp[L];
        const int id = g.island_id.v[cs[0]];
        const IslandRec& J = g.islands[id];
        const double oi = ci[id] / std::max(1.0, cn[id]), oj = cj[id] / std::max(1.0, cn[id]);
        std::vector<double> ang(cs.size());
        for (size_t q = 0; q < cs.size(); ++q) ang[q] = std::atan2(-(static_cast<double>(cs[q] / W) - oi), static_cast<double>(cs[q] % W) - oj);
        std::vector<size_t> ord(cs.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return ang[a] < ang[b] || (ang[a] == ang[b] && cs[a] < cs[b]); });
        for (size_t s0 = 0; s0 < ord.size(); s0 += static_cast<size_t>(seg)) {
            const size_t s1 = std::min(ord.size(), s0 + static_cast<size_t>(seg));
            SpringSeg S;
            S.island = id;
            double q = 0.0, hs = 0.0, aq = 0.0;
            for (size_t t = s0; t < s1; ++t) {
                const int32_t k = cs[ord[t]];
                S.cells.push_back(k);
                q += g.recharge_acc.v[k];
                const double skel = (!g.skel_top.v.empty() && !std::isnan(g.skel_top.v[k])) ? g.skel_top.v[k]
                                                                                           : J.keel_j + ske * (J.rim_j - J.keel_j);
                hs += skel;
                aq += std::max(0.0, g.wt.v[k] - skel);   // 含水层厚度（水位 − 骨架顶面）
            }
            S.q_ls = q * to_ls;
            if (S.q_ls < qmin) continue;
            const int32_t mid = cs[ord[(s0 + s1 - 1) / 2]];
            S.ci = mid / W;
            S.cj = mid % W;
            S.height_m = hs / static_cast<double>(s1 - s0);
            S.length_km = static_cast<double>(s1 - s0) * g.res_km;
            S.aquifer_m = aq / static_cast<double>(s1 - s0);
            all.push_back(std::move(S));
        }
    }
    // A（四点四十九）：段的出水 ≥ 本岛分位（spring_pct / spring_fall_pct）**且**含水层够厚（≥ spring_min_aquifer_m）才叫泉 / 崖瀑，
    // 其余是弥散渗出——这样干湿岛的「有泉比例」一致，崖壁那条圈也断成有限几处
    const int n_isl = static_cast<int>(g.islands.size());
    std::vector<std::vector<double>> byq(static_cast<size_t>(n_isl));
    for (const SpringSeg& s : all) byq[static_cast<size_t>(s.island)].push_back(s.q_ls);
    for (auto& v : byq) std::sort(v.begin(), v.end());
    for (SpringSeg& S : all) {
        const std::vector<double>& v = byq[static_cast<size_t>(S.island)];
        if (v.size() >= 4 && S.aquifer_m >= min_aq) {
            const size_t i85 = static_cast<size_t>(std::llround(pct * static_cast<double>(v.size() - 1)));
            const size_t i97 = static_cast<size_t>(std::llround(fpct * static_cast<double>(v.size() - 1)));
            if (S.q_ls >= v[std::min(i97, v.size() - 1)]) S.kind = 2;
            else if (S.q_ls >= v[std::min(i85, v.size() - 1)]) S.kind = 1;
        }
        S.fall = (S.kind == 2);
        g.springline.push_back(std::move(S));
    }
}

std::vector<int64_t> group_recv(const Group& g) {
    const size_t N = static_cast<size_t>(g.H) * g.W;
    std::vector<int64_t> recv(N, -1);
    for (size_t k = 0; k < N; ++k) {
        const int ri = g.recv_i.v[k], rj = g.recv_j.v[k];
        if (g.island_id.v[k] >= 0 && ri >= 0 && rj >= 0) {
            const int64_t r = static_cast<int64_t>(ri) * g.W + rj;
            if (g.island_id.v[r] >= 0) recv[k] = r;
        }
    }
    return recv;
}

double core_heat_mw(const Group& g, int island, const Config& c) {
    if (g.condense.v.empty()) return 0.0;
    const size_t N = static_cast<size_t>(g.H) * g.W;
    const double cell_m2 = g.res_km * g.res_km * 1e6;
    double s = 0.0;
    for (size_t k = 0; k < N; ++k)
        if (g.island_id.v[k] == island) s += g.condense.v[k] / 1000.0;
    return s * cell_m2 * latent_w_per_m(g.year_s) * wc(c, "heat_share") / 1e6;
}

}  // namespace skyisle::island
