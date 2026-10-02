// 河的数据（C3）：见 rivernet.hpp。
#include "skyisle/island/rivernet.hpp"

#include <algorithm>
#include <cmath>
#include <deque>

#include "skyisle/flow.hpp"
#include "skyisle/island/groundwater.hpp"

namespace skyisle::island {

namespace {

constexpr int LC_FOREST = 4;
constexpr double RHO_G = 1000.0 * 9.81;

double rc(const Config& c, const std::string& k) { return c.get("rivers." + k); }

double lith_at(const std::vector<double>& tab, uint8_t li, double dflt) {
    return (li >= 1 && li - 1 < static_cast<int>(tab.size())) ? tab[li - 1] : dflt;
}

int sgn(int x) { return (x > 0) - (x < 0); }

}  // namespace

std::vector<double> event_runoff_index(const std::vector<double>& liquid, double quick_days) {
    const int n = static_cast<int>(liquid.size());
    std::vector<double> out(n, 0.0);
    if (!n) return out;
    // 有限传播时间，不用无限指数尾巴把雨水沟变成泉水。窗宽为三倍快流时间常数。
    const int window = std::min(n, std::max(1, static_cast<int>(std::ceil(3.0 * quick_days))));
    double total = 0.0;
    for (int d = 0; d < n; ++d) {
        for (int lag = 0; lag < window; ++lag)
            out[d] += std::max(0.0, liquid[(d - lag + n) % n]) * (window - lag);
        total += out[d];
    }
    if (total > 0.0) for (double& x : out) x *= n / total;
    return out;
}

void build_rivernet(Group& g, const Config& c) {
    RiverNet& R = g.rnet;
    R = RiverNet();
    g.has_rnet = true;
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_m = g.res_km * 1000.0, cell_km2 = g.res_km * g.res_km;
    const double year_s = std::max(1.0, g.year_s);
    const std::vector<int64_t> recv = group_recv(g);
    Mask land(H, W, 0), seed(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
        seed.v[k] = (land.v[k] && !g.lake.v[k] && (g.river.v[k] > 0 || g.stream.v[k] > 0)) ? 1 : 0;
    }
    auto void_side = [&](size_t k) {
        const int i = static_cast<int>(k / W), j = static_cast<int>(k % W);
        for (int a = -1; a <= 1; ++a)
            for (int b = -1; b <= 1; ++b) {
                const int ii = i + a, jj = j + b;
                if (ii < 0 || ii >= H || jj < 0 || jj >= W || g.island_id(ii, jj) < 0) return true;
            }
        return false;
    };
    auto step_len = [&](int64_t a, int64_t b) { return res_m * ((a / W != b / W && a % W != b % W) ? SQRT2 : 1.0); };

    // ---- 干支分段：出口按汇水降序（平局格号小）；每段从出口往上追汇水最大的一支，别的支各起一段（先进先出，段号即次序）
    std::vector<int64_t> down(N, -1);
    std::vector<int32_t> cnt(N + 1, 0);
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k] && recv[k] >= 0 && seed.v[recv[k]]) {
            down[k] = recv[k];
            cnt[recv[k] + 1]++;
        }
    for (size_t k = 0; k < N; ++k) cnt[k + 1] += cnt[k];
    std::vector<int32_t> don(cnt[N]);
    {
        std::vector<int32_t> pos(cnt.begin(), cnt.end() - 1);
        for (size_t k = 0; k < N; ++k)
            if (down[k] >= 0) don[pos[down[k]]++] = static_cast<int32_t>(k);
    }
    auto by_acc = [&](int32_t a, int32_t b) { return g.acc_km2.v[a] > g.acc_km2.v[b] || (g.acc_km2.v[a] == g.acc_km2.v[b] && a < b); };
    for (size_t k = 0; k < N; ++k)
        if (cnt[k + 1] - cnt[k] > 1) std::sort(don.begin() + cnt[k], don.begin() + cnt[k + 1], by_acc);
    std::vector<int32_t> outs;
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k] && down[k] < 0) outs.push_back(static_cast<int32_t>(k));
    std::sort(outs.begin(), outs.end(), by_acc);
    std::vector<int32_t> pos_in(N, -1);
    struct Job {
        int32_t start;
        int down_seg;
        int32_t join_cell;
    };
    for (int32_t o : outs) {
        std::deque<Job> q{{o, -1, -1}};
        while (!q.empty()) {
            const Job jb = q.front();
            q.pop_front();
            const int sid = static_cast<int>(R.segs.size());
            std::vector<int32_t> rev{jb.start};
            for (int32_t cur = jb.start;;) {
                const int32_t b = cnt[cur], e = cnt[cur + 1];
                if (b == e) break;
                for (int32_t t = b + 1; t < e; ++t) q.push_back({don[t], sid, cur});
                cur = don[b];
                rev.push_back(cur);
            }
            RiverNet::Seg s;
            s.island = g.island_id.v[jb.start];
            s.down = jb.down_seg;
            s.start = static_cast<int>(R.cell.size());
            s.n = static_cast<int>(rev.size());
            for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
                pos_in[*it] = static_cast<int32_t>(R.cell.size()) - s.start;
                R.cell.push_back(*it);
                s.level = std::max(s.level, static_cast<int>(g.river.v[*it]));
            }
            for (int t = s.start + 1; t < s.start + s.n; ++t) s.length_km += step_len(R.cell[t - 1], R.cell[t]) / 1000.0;
            if (jb.down_seg >= 0) {
                s.join = pos_in[jb.join_cell];
                s.exit = 0;
            } else {
                const int64_t r = recv[jb.start];
                s.exit = (r < 0 && void_side(static_cast<size_t>(jb.start))) ? 1 : ((r >= 0 && g.lake.v[r]) ? 2 : 3);
            }
            R.segs.push_back(s);
        }
    }

    // ---- 上游累计（按流向的拓扑序，上游先加）：格数、高程、BQART 的岩性项、河床质的粗细、已垦、林地、最高点
    const std::vector<double> Lt = c.list("rivers.bqart_lith", {1.0, 1.75, 0.6, 1.5, 1.0});
    const std::vector<double> Ct = c.list("rivers.d50_lith", {1.0, 0.4, 1.3, 0.6, 0.3});
    const bool has_lith = !g.lith.v.empty();
    const bool has_farm = !g.cultivated.v.empty();
    const std::vector<int32_t> order = downstream_first(recv, land);
    std::vector<float> a_n(N, 0.f), a_z(N, 0.f), a_L(N, 0.f), a_c(N, 0.f), a_farm(N, 0.f), a_for(N, 0.f), a_zmax(N, 0.f);
    std::vector<double> rain_acc(N, 0.0);
    for (int32_t k : order) {
        const uint8_t li = has_lith ? g.lith.v[k] : 0;
        a_n[k] = 1.f;
        rain_acc[k] = std::max(0.0, g.runoff.v[k] - (g.condense.v.empty() ? 0.0 : g.condense.v[k])) * cell_km2;
        a_z[k] = static_cast<float>(g.height.v[k]);
        a_zmax[k] = static_cast<float>(g.height.v[k]);
        a_L[k] = static_cast<float>(lith_at(Lt, li, 1.0));
        a_c[k] = static_cast<float>(lith_at(Ct, li, 1.0));
        a_farm[k] = (has_farm && g.cultivated.v[k] > 0) ? 1.f : 0.f;
        a_for[k] = g.landcover.v[k] == LC_FOREST ? 1.f : 0.f;
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t r = recv[*it];
        if (r < 0) continue;
        const int32_t k = *it;
        a_n[r] += a_n[k];
        rain_acc[r] += rain_acc[k];
        a_z[r] += a_z[k];
        a_L[r] += a_L[k];
        a_c[r] += a_c[k];
        a_farm[r] += a_farm[k];
        a_for[r] += a_for[k];
        a_zmax[r] = std::max(a_zmax[r], a_zmax[k]);
    }

    // ---- 每个河网出口（含临时流路、入湖）单独汇水，不能把无关的小流域混成一个供水源。
    std::vector<int> basin_of(N, -1);
    for (size_t s = 0; s < R.segs.size(); ++s) {
        RiverNet::Seg& sg = R.segs[s];
        if (sg.down < 0) {
            RiverNet::Basin B;
            B.seg = static_cast<int>(s);
            B.island = sg.island;
            const int32_t k = R.cell[sg.start + sg.n - 1];
            B.ci = k / W;
            B.cj = k % W;
            basin_of[k] = static_cast<int>(R.basins.size());
            R.basins.push_back(B);
        }
    }
    for (int32_t k : order) {
        const int64_t r = recv[k];
        if (r >= 0 && basin_of[k] < 0) basin_of[k] = basin_of[r];
    }
    const int nb = static_cast<int>(R.basins.size());
    R.basins.push_back(RiverNet::Basin());   // 其余无河网坡面，仅用于水账
    for (RiverNet::Seg& sg : R.segs) {
        const int b = basin_of[R.cell[sg.start + sg.n - 1]];
        sg.basin = b;
    }

    // ---- 逐日径流指数
    const std::vector<double> rec_t = c.list("water.recession_days", {60.0, 15.0, 30.0, 15.0, 30.0});
    const double bfi_def = c.get("water.bfi_default", 0.4);
    const double snow_t = rc(c, "snow_t_c"), melt_t = rc(c, "melt_t_c"), ddf = rc(c, "ddf_mm");
    const int nbands = std::max(1, c.geti("rivers.bands"));
    const double qk = rc(c, "quick_k_days"), qe = rc(c, "quick_exp"), qmin = rc(c, "quick_min_days"), qmax = rc(c, "quick_max_days");
    const double lapse = g.inp.lapse_c_per_km, h_ref = g.inp.height_m;
    const std::vector<double>& Pd = g.weather.precip_mm;
    const std::vector<double>& Td = g.weather.temp_c;
    const int nd = static_cast<int>(Pd.size());
    double Ptot = 0.0;
    for (double p : Pd) Ptot += p;
    std::vector<std::vector<int32_t>> members(static_cast<size_t>(nb) + 1);
    for (size_t k = 0; k < N; ++k)
        if (land.v[k]) members[basin_of[k] >= 0 ? basin_of[k] : nb].push_back(static_cast<int32_t>(k));
    for (int b = 0; b <= nb; ++b) {
        RiverNet::Basin& B = R.basins[b];
        const std::vector<int32_t>& cells = members[b];
        B.area_km2 = static_cast<double>(cells.size()) * cell_km2;
        if (cells.empty() || nd == 0) continue;
        double Wsum = 0.0, Csum = 0.0, wb = 0.0, wbr = 0.0;
        std::vector<double> w(cells.size());
        for (size_t q = 0; q < cells.size(); ++q) {
            const int32_t k = cells[q];
            const double cond = g.condense.v.empty() ? 0.0 : g.condense.v[k];
            w[q] = std::max(0.0, g.runoff.v[k] - cond) * cell_km2;
            const double bf = g.bfi.v.empty() ? bfi_def : g.bfi.v[k];
            Wsum += w[q];
            Csum += cond * cell_km2;
            wb += w[q] * bf;
            wbr += w[q] * bf * lith_at(rec_t, has_lith ? g.lith.v[k] : 0, 30.0);
        }
        B.q_mean = b < nb ? g.runoff_acc.v[static_cast<size_t>(B.ci) * W + B.cj] * 1000.0 / year_s : (Wsum + Csum) * 1000.0 / year_s;
        B.bfi = Wsum > 0.0 ? wb / Wsum : bfi_def;
        B.recession_days = wb > 0.0 ? wbr / wb : 30.0;
        B.cond_frac = (Wsum + Csum) > 0.0 ? Csum / (Wsum + Csum) : 0.0;
        B.quick_days = b < nb ? clip(qk * np_pow(B.area_km2, qe), qmin, qmax) : qmin;
        // 按高程分带（等径流量），带内按径流量加权的高程与局地雨
        std::vector<size_t> ix(cells.size());
        for (size_t q = 0; q < ix.size(); ++q) ix[q] = q;
        std::stable_sort(ix.begin(), ix.end(), [&](size_t a, size_t b2) { return g.height.v[cells[a]] < g.height.v[cells[b2]]; });
        std::vector<double> bz(nbands, 0.0), bp(nbands, 0.0), bw(nbands, 0.0);
        double cum = 0.0;
        for (size_t q : ix) {
            const int band = Wsum > 0.0 ? std::min(nbands - 1, static_cast<int>(cum / Wsum * nbands)) : 0;
            cum += w[q];
            bw[band] += w[q];
            bz[band] += w[q] * g.height.v[cells[q]];
            bp[band] += w[q] * g.rain.v[cells[q]];
        }
        for (int t = 0; t < nbands; ++t)
            if (bw[t] > 0.0) {
                bz[t] /= bw[t];
                bp[t] /= bw[t];
            }
        const double aq = 1.0 - std::exp(-1.0 / B.quick_days), as = 1.0 - std::exp(-1.0 / std::max(1.0, B.recession_days));
        std::vector<double> swe(nbands, 0.0);
        double Sq = 0.0, Ss = 0.0, tot = 0.0, from_snow = 0.0, gen = 0.0;
        B.index.assign(nd, 0.0);
        std::vector<double> event_liquid(nd, 0.0);
        for (int pass = 0; pass < 2; ++pass)
            for (int d = 0; d < nd; ++d) {
                double quick = 0.0, slow = Csum / nd;
                for (int t = 0; t < nbands; ++t) {
                    if (!(bw[t] > 0.0) || !(bp[t] > 0.0)) continue;
                    const double T = Td[d] + lapse * (h_ref - bz[t]) / 1000.0;
                    const double p = Ptot > 0.0 ? bp[t] * Pd[d] / Ptot : bp[t] / nd;
                    double liq = 0.0;
                    if (T <= snow_t) swe[t] += p;
                    else liq = p;
                    const double m = std::min(swe[t], ddf * std::max(0.0, T - melt_t));
                    swe[t] -= m;
                    liq += m;
                    const double r = bw[t] * liq / bp[t];   // 这一天产的径流（mm·km²）：年径流按当天到地面的水（雨 + 融雪）分
                    quick += (1.0 - B.bfi) * r;
                    slow += B.bfi * r;
                    if (pass == 1) {
                        gen += r;
                        event_liquid[d] += r;
                        from_snow += bw[t] * m / bp[t];
                    }
                }
                Sq += quick;
                Ss += slow;
                const double Qq = aq * Sq, Qs = as * Ss;
                Sq -= Qq;
                Ss -= Qs;
                if (pass == 1) {
                    B.index[d] = Qq + Qs;
                    tot += Qq + Qs;
                }
            }
        const double mean = tot / nd;
        double mx = 0.0;
        for (double& x : B.index) {
            x = mean > 0.0 ? x / mean : 1.0;
            mx = std::max(mx, x);
        }
        B.bf_ratio = mx;
        B.snow_frac = gen > 0.0 ? from_snow / gen : 0.0;
        B.event_index = event_runoff_index(event_liquid, B.quick_days);
        for (double x : B.event_index) B.event_bf_ratio = std::max(B.event_bf_ratio, x);
    }

    // ---- 沿程每点
    const double tg = rc(c, "shields_gravel"), ts = rc(c, "shields_sand"), low_w = rc(c, "low_energy_w_m2");
    // 平岸宽深 = 年均宽深 × (平岸 / 年均)^站内指数（L&M 1964：w ∝ Q^0.26、d ∝ Q^0.40）——别用下游的 0.5（四点五十一）
    const double wbexp = c.get("hydro.at_station_width_b", 0.26), dfexp = c.get("hydro.at_station_depth_f", 0.40);
    // 年均流量下的水面宽深（下游关系，L&M 1953）：w = width_a·Q^width_b、d = depth_c·Q^depth_f
    const double wa = c.get("hydro.width_a", 5.0), wb = c.get("hydro.width_b", 0.5);
    const double dc0 = c.get("hydro.depth_c", 0.35), df0 = c.get("hydro.depth_f", 0.4);
    const double bf_farm = rc(c, "bqart_farm"), bf_for = rc(c, "bqart_forest"), tol = rc(c, "floor_tol_m");
    const double a_temp = g.clim.a_temp, a_ref = g.clim.a_ref_h;
    const size_t M = R.cell.size();
    R.level.resize(M);
    R.flow_regime.resize(M);
    R.d50c.resize(M);
    R.planform.resize(M);
    R.confine.resize(M);
    for (auto* v : {&R.acc, &R.q_mean, &R.q_bf, &R.w, &R.d, &R.w_mean, &R.d_mean, &R.surf, &R.bed, &R.slope, &R.d50_mm, &R.ssc, &R.fp_l, &R.fp_r}) v->resize(M);
    auto fp_walk = [&](int i, int j, int a, int b, double surf, double nominal, double w) {
        if (!(nominal > 0.0) || (a == 0 && b == 0)) return 0.0;
        const double step = res_m * ((a != 0 && b != 0) ? SQRT2 : 1.0);
        const int maxs = static_cast<int>(std::ceil((nominal + w / 2.0) / step)) + 1;
        int n = 0;
        for (int s = 1; s <= maxs; ++s) {
            const int ii = i + s * a, jj = j + s * b;
            if (ii < 0 || ii >= H || jj < 0 || jj >= W || g.island_id(ii, jj) < 0) break;
            const double hq = g.height(ii, jj);
            if (!(hq <= surf + tol)) break;
            ++n;
        }
        return clip(std::min((n + 0.5) * step - w / 2.0, nominal), 0.0, nominal);
    };
    for (const RiverNet::Seg& sg : R.segs) {
        const auto& basin = R.basins[sg.basin >= 0 ? sg.basin : nb];
        for (int t = 0; t < sg.n; ++t) {
            const size_t m = static_cast<size_t>(sg.start + t);
            const int32_t k = R.cell[m];
            const int i = k / W, j = k % W;
            const bool perennial = g.river.v[k] > 0;
            R.flow_regime[m] = perennial ? 1 : 0;
            const double ratio = perennial ? basin.bf_ratio : basin.event_bf_ratio;
            const double rq = ratio > 0.0 ? ratio : 1.0;
            // 临时流路无已建模的稳定出露水源。凝结补给留在地下，遇常驻河后才计入河水。
            const double q = (perennial ? g.runoff_acc.v[k] : rain_acc[k]) * 1000.0 / year_s;
            const double qbf = q * (ratio > 0.0 ? ratio : 1.0);
            const double w = g.width_m.v[k], d = g.depth_m.v[k], surf = g.height.v[k];
            const double bed = std::isnan(g.bed_m.v[k]) ? surf - d : g.bed_m.v[k];
            const double S = (g.chan_slope.v.empty() || std::isnan(g.chan_slope.v[k])) ? 0.0 : g.chan_slope.v[k];
            const double n_up = std::max(1.0f, a_n[k]);
            // 河床质：平岸 Shields 数，砾床算出来 < 2 mm 的落到砂床（砾—砂的突变）；乘上游岩性的粗细
            const double coarse = a_c[k] / n_up;
            double D = d * S / (1.65 * tg) * coarse;
            if (D < 0.002) D = d * S / (1.65 * ts) * coarse;
            const uint8_t dc = D < 6.25e-5 ? 0 : (D < 0.002 ? 1 : (D < 0.064 ? 2 : (D < 0.256 ? 3 : (D < 1.0 ? 4 : 5))));
            // 平面型
            const uint8_t cf = g.confine.v[k];
            uint8_t pf = 2;
            if (cf == 1) pf = 0;
            else {
                const double om = qbf > 0.0 ? RHO_G * qbf * S / (4.7 * std::sqrt(qbf)) : 0.0;
                const double Dm = np_pow(std::max(D, 1e-5), 0.42);
                pf = om > 900.0 * Dm ? 4 : (om > 90.0 * Dm ? 3 : ((D < 0.002 && om < low_w) ? 1 : 2));
            }
            // 悬沙（BQART）：Qs = 0.02 kg/s × B × Q^0.31（km³/年）× A^0.5 × R（km）× T（< 2 °C 按 2）
            const double zmean = a_z[k] / n_up;
            const double T = a_temp + lapse * (a_ref - zmean) / 1000.0;
            const double Rkm = std::max(0.0, static_cast<double>(a_zmax[k]) - bed) / 1000.0;
            const double E = (1.0 + bf_farm * a_farm[k] / n_up) * (1.0 - bf_for * a_for[k] / n_up);
            const double Qs = 0.02 * (a_L[k] / n_up) * E * np_pow(std::max(0.0, q * year_s / 1e9), 0.31) * std::sqrt(std::max(0.0, g.acc_km2.v[k])) * Rkm *
                              std::max(T, 2.0);
            // 左右谷底宽：沿垂直流向走到谷坡脚，夹 C2 的谷底宽
            int64_t nx = down[k] >= 0 ? down[k] : recv[k];
            int di = 0, dj = 0;
            if (nx >= 0) {
                di = sgn(static_cast<int>(nx / W) - i);
                dj = sgn(static_cast<int>(nx % W) - j);
            } else if (t > 0) {
                const int32_t pv = R.cell[m - 1];
                di = sgn(i - pv / W);
                dj = sgn(j - pv % W);
            }
            const double nominal = std::max(0.0, (g.floor_w.v[k] - w) / 2.0);
            R.level[m] = g.river.v[k];
            R.acc[m] = static_cast<float>(g.acc_km2.v[k]);
            R.q_mean[m] = static_cast<float>(q);
            R.q_bf[m] = static_cast<float>(qbf);
            // 三个口径各算各的（别叠乘）：年均流量下的水面 = 直接按式子（不依赖栅格，栅格里的已经是河道）；
            // 真平岸 = 年均 × (q_bf/q_mean)^站内指数；栅格 w_ch_m / d_ch_m = 河道 = 年均 × (假设比值)^站内指数（river.cpp 切的）
            const double w_mean = wa * np_pow(q, wb), d_mean = dc0 * np_pow(q, df0);
            R.w_mean[m] = static_cast<float>(w_mean);
            R.d_mean[m] = static_cast<float>(d_mean);
            // 平岸宽与深：年均口径 × (平岸 / 年均)^站内指数（L&M 1964：宽 0.26、深 0.40）——q_bf = 年均 × 所在流域的 bf_ratio
            R.w[m] = static_cast<float>(w_mean * np_pow(rq, wbexp));
            R.d[m] = static_cast<float>(d_mean * np_pow(rq, dfexp));
            R.surf[m] = static_cast<float>(surf);
            R.bed[m] = static_cast<float>(bed);
            R.slope[m] = static_cast<float>(S);
            R.d50_mm[m] = static_cast<float>(D * 1000.0);
            R.d50c[m] = dc;
            R.planform[m] = pf;
            R.ssc[m] = static_cast<float>(q > 0.0 ? Qs / q * 1000.0 : 0.0);
            R.fp_l[m] = static_cast<float>(fp_walk(i, j, -dj, di, surf, nominal, w));
            R.fp_r[m] = static_cast<float>(fp_walk(i, j, dj, -di, surf, nominal, w));
            R.confine[m] = cf;
        }
    }

    // ---- 瀑布与跌水
    const double f_step = rc(c, "fall_step_m"), f_grad = rc(c, "fall_grad"), f_min = rc(c, "fall_min_m");
    const double f_wall = rc(c, "fall_wall_grad"), f_wall_m = rc(c, "fall_wall_m");
    const bool has_wall = !g.rockwall_m.v.empty();
    for (size_t s = 0; s < R.segs.size(); ++s) {
        const RiverNet::Seg& sg = R.segs[s];
        int t0 = -1;
        double tot = 0.0, len = 0.0;
        bool wall = false;
        auto flush = [&]() {
            if (t0 >= 0 && tot >= f_min) {
                RiverNet::Fall F;
                F.seg = static_cast<int>(s);
                F.idx = t0;
                const int32_t k = R.cell[sg.start + t0];
                F.ci = k / W;
                F.cj = k % W;
                F.level = R.level[sg.start + t0];
                F.kind = (wall || (tot >= f_wall_m && tot / std::max(1.0, len) >= f_wall)) ? 1 : 2;
                F.drop_m = tot;
                F.length_m = len;
                F.q_mean = R.q_mean[sg.start + t0];
                R.falls.push_back(F);
            }
            t0 = -1;
        };
        for (int t = 0; t + 1 < sg.n; ++t) {
            const size_t m = static_cast<size_t>(sg.start + t);
            const double drop = static_cast<double>(R.bed[m]) - R.bed[m + 1];
            const double dist = step_len(R.cell[m], R.cell[m + 1]);
            if (drop >= f_step && drop / dist >= f_grad) {
                if (t0 < 0) {
                    t0 = t;
                    tot = len = 0.0;
                    wall = false;
                }
                tot += drop;
                len += dist;
                wall = wall || (has_wall && (g.rockwall_m.v[R.cell[m]] > 0.0 || g.rockwall_m.v[R.cell[m + 1]] > 0.0));
            } else {
                flush();
            }
        }
        flush();
        if (sg.exit == 1) {
            const size_t m = static_cast<size_t>(sg.start + sg.n - 1);
            const int32_t k = R.cell[m];
            RiverNet::Fall F;
            F.seg = static_cast<int>(s);
            F.idx = sg.n - 1;
            F.ci = k / W;
            F.cj = k % W;
            F.level = R.level[m];
            F.kind = 0;
            F.drop_m = R.surf[m] - g.islands[sg.island].keel_j;
            F.q_mean = R.q_mean[m];
            R.falls.push_back(F);
        }
    }
}

}  // namespace skyisle::island
