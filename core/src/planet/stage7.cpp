// ⑦ 文明中心（skyisle_gen/stages/s07_centers.py）：适宜度（降水 × 稳定 × 岛密度 × 陆地^γ × 谷物门槛，候选边上 2 跳平滑）
// → 三个骨架窗内取 argmax（窗空了按 1.5 倍逐步放宽）→ 无约束次级极大（每圈保底）→ 史前扩散（抱石而渡、顺风单向）与谱系
// → ⑦b 地区（种子 = 中心 ∪ 大枢纽 ∪ 最远点采样，超大地区分裂）→ 中心间干线。不读 height_m（原则乙）。
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "skyisle/planet/civ.hpp"

namespace skyisle::planet {

namespace {

inline double smoothstep_r(double x, double lo, double hi) {   // skeleton.smoothstep
    const double t = clip((x - lo) / std::max(1e-9, hi - lo), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// s02_wind.band_id_of（给了 band_local：局部带界按经度插值，edges 按 float32 读回）
std::vector<int> band_ids(const Islands& isl, const Climate& c) {
    const std::vector<double> edges = f32v(c.edges);
    const std::vector<double>& lons = c.ax.lons;
    const int W = c.ax.nlon;
    const double res = lons[1] - lons[0];
    std::vector<int> out(isl.n());
    for (size_t q = 0; q < isl.n(); ++q) {
        const double fj = (isl.lon[q] - lons[0]) / res;
        const int64_t j0f = static_cast<int64_t>(std::floor(fj));
        const double t = fj - static_cast<double>(j0f);
        const int64_t j0 = ((j0f % W) + W) % W, j1 = (j0 + 1) % W;
        double e[8];
        for (int k = 0; k < 8; ++k) e[k] = edges[static_cast<size_t>(k) * W + j0] * (1 - t) + edges[static_cast<size_t>(k) * W + j1] * t;
        const double la = isl.lat[q];
        const bool north = la >= 0;
        const double eq = north ? e[0] : -e[4], tr = north ? e[1] : -e[5], ca = north ? e[2] : -e[6], we = north ? e[3] : -e[7];
        const double a = std::fabs(la);
        const int tier = a < eq ? 0 : a < tr ? 1 : a < ca ? 2 : a < we ? 3 : 4;
        out[q] = tier == 0 ? 0 : (north ? tier : tier + 4);
    }
    return out;
}

inline Vec3 xyz_of(const Islands& isl, int64_t q) { return {isl.xyz[3 * q], isl.xyz[3 * q + 1], isl.xyz[3 * q + 2]}; }

// 窗内 argmax（第一个最大者）；窗空 → −1
int64_t argmax_in(const std::vector<double>& v, const std::vector<uint8_t>& m) {
    int64_t best = -1;
    for (size_t q = 0; q < v.size(); ++q)
        if (m[q] && (best < 0 || v[q] > v[best])) best = static_cast<int64_t>(q);
    return best;
}

}  // namespace

Centers stage7(const Config& cfg, const Planet& p, const Islands& isl, const Climate& c, const Barriers& b, const Routes& r) {
    const std::string C7 = "s07.centers.";
    const int64_t N = static_cast<int64_t>(isl.n());
    Centers out;

    // ---- 适宜度（_suitability）
    std::vector<double> suit(N);
    {
        std::vector<double> dn(N), an(N);
        for (int64_t q = 0; q < N; ++q) {
            dn[q] = std::log(np_maximum(f32(isl.density_at[q]), 1e-9));
            an[q] = std::log(np_maximum(f32(isl.area[q]), 1e-9));
        }
        auto normalize = [](std::vector<double>& x) {
            const double mn = *std::min_element(x.begin(), x.end()), mx = *std::max_element(x.begin(), x.end());
            const double den = std::max(1e-9, mx - mn);
            for (double& v : x) v = (v - mn) / den;
        };
        normalize(dn);
        normalize(an);
        const double gamma = cfg.get(C7 + "area_exponent");
        // 降水项是按毫米的驼峰（PLAN-NATURE A4）：年雨在 precip_hump_mm 之间 = 1，往干、往湿按对数距离的高斯降（靠天种粟麦的半湿润最好，旧的「越湿越好」作废）
        const std::vector<double>& hump = cfg.list(C7 + "precip_hump_mm");
        const double hw = cfg.get(C7 + "precip_hump_width");
        for (int64_t q = 0; q < N; ++q) {
            const double mm = np_maximum(precip_mm(f32(c.i_precip[q]), cfg), 1.0);
            const double d = mm < hump[0] ? std::log(hump[0] / mm) : (mm > hump[1] ? std::log(mm / hump[1]) : 0.0);
            const double pn = std::exp(-0.5 * (d / hw) * (d / hw));
            suit[q] = pn * f32(c.i_stability[q]) * dn[q] * np_pow(an[q], gamma);
        }
        if (cfg.has_list(C7 + "season_winter_warm_c") && !c.i_season_range_sea.empty()) {
            const std::vector<double>& wr = cfg.list(C7 + "season_winter_warm_c");
            const std::vector<double>& cr = cfg.list(C7 + "season_winter_cold_c");
            for (int64_t q = 0; q < N; ++q) {
                const double winter = f32(c.i_temp_sea[q]) - 0.5 * f32(c.i_season_range_sea[q]);
                const double grain = (1.0 - smoothstep_r(winter, wr[0], wr[1])) * smoothstep_r(winter, cr[0], cr[1]);
                suit[q] = suit[q] * grain;
            }
        }
        // 候选边上 hops 跳邻域平滑（np.add.at 按下标次序：先全部 src 再全部 dst）
        const int hops = static_cast<int>(cfg.get(C7 + "smooth_hops"));
        const size_t E = isl.src.size();
        for (int h = 0; h < hops; ++h) {
            std::vector<double> acc = suit, cnt(N, 1.0);
            for (size_t e = 0; e < E; ++e) acc[isl.src[e]] += suit[isl.dst[e]];
            for (size_t e = 0; e < E; ++e) acc[isl.dst[e]] += suit[isl.src[e]];
            for (size_t e = 0; e < E; ++e) cnt[isl.src[e]] += 1.0;
            for (size_t e = 0; e < E; ++e) cnt[isl.dst[e]] += 1.0;
            for (int64_t q = 0; q < N; ++q) suit[q] = acc[q] / cnt[q];
        }
    }
    out.suit = suit;

    // ---- 三个骨架窗内的涌现（窗内取 argmax）
    const double lon_w = cfg.get("skeleton.d_lon_west"), lon_e = cfg.get("skeleton.d_lon_east");
    const std::vector<double>& clr = cfg.list("skeleton.core_lat_range");
    const double core_lo = clr[0] * p.band_scale, core_hi = clr[1] * p.band_scale;
    auto window = [&](int cid, double w) {
        std::vector<uint8_t> m(N);
        auto in_lon = [](double lon, double lo, double hi) { return pymod(lon - lo, 360.0) <= pymod(hi - lo, 360.0); };
        for (int64_t q = 0; q < N; ++q) {
            const double la = isl.lat[q], lo = isl.lon[q], a = std::fabs(la);
            const bool core = (a >= core_lo) && (a <= core_hi);
            bool v;
            if (cid == 0) v = core && la > 0 && in_lon(lo, lon_w - w, lon_w);
            else if (cid == 1) v = core && la > 0 && in_lon(lo, lon_e, lon_e + w);
            else v = core && la < 0;
            m[q] = v;
        }
        return m;
    };
    double global_max = -INF;
    for (double v : suit) global_max = std::max(global_max, v);
    const double tau_c = cfg.get(C7 + "tau_c");
    const double base_w = cfg.get("skeleton.center_window_deg");
    for (int cid = 0; cid < 3; ++cid) {
        std::vector<uint8_t> m = window(cid, base_w);
        auto any = [](const std::vector<uint8_t>& x) { return std::any_of(x.begin(), x.end(), [](uint8_t v) { return v != 0; }); };
        if (!any(m)) {   // 板块把窗内岛群抽空：按 1.5 倍逐步放宽窗宽，最多到 4 倍
            double mult = 1.5;
            while (!any(m) && mult <= 4.0) {
                m = window(cid, base_w * mult);
                mult *= 1.5;
            }
        }
        const int64_t node = argmax_in(suit, m);
        if (node < 0) throw std::invalid_argument(std::string("center window ") + CENTER_KEYS[cid] + " has no island");
        if (suit[node] < tau_c * global_max)
            throw std::invalid_argument(std::string("center window ") + CENTER_KEYS[cid] + ": max suitability < tau_c * global max (principle geng)");
        out.node[cid] = node;
    }

    // ---- 无约束次级极大（供 ⑧ 次级起源）
    const size_t E = isl.src.size();
    std::vector<double> nb_max = suit;
    for (size_t e = 0; e < E; ++e) nb_max[isl.src[e]] = np_maximum(nb_max[isl.src[e]], suit[isl.dst[e]]);
    for (size_t e = 0; e < E; ++e) nb_max[isl.dst[e]] = np_maximum(nb_max[isl.dst[e]], suit[isl.src[e]]);
    std::vector<int64_t> peaks;
    for (int64_t q = 0; q < N; ++q)
        if (suit[q] >= nb_max[q] - 1e-15) peaks.push_back(q);
    std::stable_sort(peaks.begin(), peaks.end(), [&](int64_t a, int64_t bq) { return -suit[a] < -suit[bq]; });
    const double days_per_rad = p.radius_km / p.day_range_km;
    const double min_sep = cfg.get(C7 + "secondary_min_sep_days");
    const int64_t n_sec = static_cast<int64_t>(cfg.get(C7 + "n_secondary"));
    std::vector<int64_t>& chosen = out.secondary;
    const std::vector<int64_t> main_nodes(out.node.begin(), out.node.end());
    auto far_enough = [&](int64_t pk, double sep) {
        const Vec3 xp = xyz_of(isl, pk);
        for (int64_t q : chosen)
            if (!(angdist(xp, xyz_of(isl, q)) * days_per_rad >= sep)) return false;
        for (int64_t q : main_nodes)
            if (!(angdist(xp, xyz_of(isl, q)) * days_per_rad >= sep)) return false;
        return true;
    };
    for (int64_t pk : peaks) {
        if (static_cast<int64_t>(chosen.size()) >= n_sec) break;
        if (far_enough(pk, min_sep)) chosen.push_back(pk);
    }
    const Directed g = directed(isl, b, r);
    // 每圈保底：圈归属与 ⑧ 同规则（商旅权重下离哪个主中心最近）
    const int per = static_cast<int>(cfg.get(C7 + "secondary_per_circle", 0.0));
    const double min_sep_q = cfg.get(C7 + "secondary_per_circle_min_sep_days", min_sep);
    if (per > 0) {
        const std::array<double, N_MODES> lam = lambda_ref(cfg);
        const std::vector<double> w_tr = mode_weight(r, g, M_TRADE, lam[M_TRADE]);
        std::vector<std::vector<double>> dc;
        for (int64_t cn : main_nodes) dc.push_back(dijkstra(g.csr, w_tr, {cn}).dist);
        std::vector<int> circle_of(N);
        for (int64_t q = 0; q < N; ++q) {
            int bi = 0;
            for (int k = 1; k < 3; ++k)
                if (dc[k][q] < dc[bi][q]) bi = k;
            circle_of[q] = bi;
        }
        int counts[3] = {0, 0, 0};
        for (int64_t pk : chosen) ++counts[circle_of[pk]];
        for (int64_t pk : peaks) {
            const int ci = circle_of[pk];
            if (counts[ci] >= per || std::find(chosen.begin(), chosen.end(), pk) != chosen.end()) continue;
            if (far_enough(pk, min_sep_q)) {
                chosen.push_back(pk);
                ++counts[ci];
            }
        }
    }

    // ---- 史前扩散：抱石而渡，顺风单向（物理成本已含 w¹，再乘 w^(exp−1)：用 cost / dist 近似风因子）
    const int64_t E2 = 2 * static_cast<int64_t>(E);
    const double exp_w = cfg.get(C7 + "prehist_wind_exponent");
    std::vector<double> w_pre(E2);
    for (int64_t k = 0; k < E2; ++k) {
        const double cd = isl.dist_days[k < static_cast<int64_t>(E) ? k : k - static_cast<int64_t>(E)];
        const double wf = cd > 0 ? r.cost[k] / cd : 1.0;
        w_pre[k] = cd * np_pow(wf, exp_w);
    }
    const std::vector<int> band_i = band_ids(isl, c);
    const std::string origin_cfg = cfg.gets(C7 + "origin", "auto");
    if (origin_cfg == "auto") {
        std::vector<uint8_t> m(N);
        for (int64_t q = 0; q < N; ++q) m[q] = band_i[q] == 1 && pymod(isl.lon[q] - lon_e, 360.0) <= 90.0;
        const int64_t o = argmax_in(suit, m);
        out.origin_node = o >= 0 ? o : out.node[1];
    } else {
        int k = -1;
        for (int cid = 0; cid < 3; ++cid)
            if (origin_cfg == CENTER_KEYS[cid]) k = cid;
        if (k < 0) throw std::invalid_argument("s07.centers.origin: unknown center id " + origin_cfg);
        out.origin_node = out.node[k];
    }
    const Paths pre = dijkstra(g.csr, w_pre, {out.origin_node});
    for (double d : pre.dist)
        if (!std::isfinite(d)) throw std::invalid_argument("prehistoric diffusion: unreachable islands (principle ji)");
    out.dist_pre = pre.dist;
    out.pred_pre = pre.pred_node;
    out.lineage.assign(N, -1);
    int64_t next_lineage = 0;
    for (int64_t u : argsort_stable(pre.dist)) {   // 谱系：沿树在带界变化处切分
        const int64_t pp = pre.pred_node[u];
        if (pp < 0 || band_i[u] != band_i[pp]) out.lineage[u] = next_lineage++;
        else out.lineage[u] = out.lineage[pp];
    }
    out.n_lineages = next_lineage;
    const double yr = cfg.get(C7 + "yr_per_day");
    out.arrival_yr.resize(N);
    for (int64_t q = 0; q < N; ++q) out.arrival_yr[q] = pre.dist[q] * yr;

    // ---- ⑦b 地区划分：种子 = 中心 ∪ 大枢纽 ∪ 最远点采样；日常模式权重（对称化）
    const std::array<double, N_MODES> lam = lambda_ref(cfg);
    const std::vector<double> w_daily = mode_weight(r, g, M_DAILY, lam[M_DAILY]);
    std::vector<double> w_sym(E2);
    for (size_t e = 0; e < E; ++e) w_sym[e] = w_sym[E + e] = np_minimum(w_daily[e], w_daily[E + e]);
    const int64_t n_regions = static_cast<int64_t>(cfg.get("s07.regions.n_regions"));
    std::vector<int64_t>& seeds = out.region_seeds;
    auto add_seed = [&](int64_t s) {
        if (std::find(seeds.begin(), seeds.end(), s) == seeds.end()) seeds.push_back(s);
    };
    for (int64_t cn : main_nodes) add_seed(cn);
    const int64_t n_hub = std::max<int64_t>(0, std::min<int64_t>(12, n_regions - 3));
    for (int64_t k = 0; k < n_hub && k < static_cast<int64_t>(r.hubs.size()); ++k) add_seed(r.hubs[k]);
    std::vector<double> d_min = dijkstra(g.csr, w_sym, seeds).dist;
    auto in_seeds = [&](int64_t s) { return std::find(seeds.begin(), seeds.end(), s) != seeds.end(); };
    while (static_cast<int64_t>(seeds.size()) < n_regions) {
        int64_t cand = 0;
        for (int64_t q = 1; q < N; ++q)
            if (d_min[q] > d_min[cand]) cand = q;   // argmax（inf 先到者）
        if (in_seeds(cand)) break;
        seeds.push_back(cand);
        const std::vector<double> dn = dijkstra(g.csr, w_sym, {cand}).dist;
        for (int64_t q = 0; q < N; ++q) d_min[q] = np_minimum(d_min[q], dn[q]);
    }
    std::vector<double> d_final;
    out.region.assign(N, -1);
    {
        const Paths a = dijkstra(g.csr, w_sym, seeds);
        d_final = a.dist;
        for (int64_t u : argsort_stable(a.dist)) {
            const auto it = std::find(seeds.begin(), seeds.end(), u);
            if (it != seeds.end()) out.region[u] = it - seeds.begin();
            else out.region[u] = a.pred_node[u] >= 0 ? out.region[a.pred_node[u]] : -1;
        }
    }
    // 超大地区分裂（地区是第二层单位，不应吞掉上千岛）
    const int64_t max_isl = static_cast<int64_t>(cfg.get("s07.regions.max_region_islands", 250.0));
    const int64_t max_regions = static_cast<int64_t>(cfg.get("s07.regions.max_regions", 150.0));
    while (static_cast<int64_t>(seeds.size()) < max_regions) {
        std::vector<int64_t> sizes(seeds.size(), 0);
        for (int64_t v : out.region)
            if (v >= 0) ++sizes[v];
        int64_t big = 0;
        for (size_t k = 1; k < sizes.size(); ++k)
            if (sizes[k] > sizes[big]) big = static_cast<int64_t>(k);
        if (sizes[big] <= max_isl) break;
        int64_t far = -1;
        double best = 0;
        for (int64_t q = 0; q < N; ++q) {
            if (out.region[q] != big) continue;
            const double v = std::isfinite(d_final[q]) ? d_final[q] : -1.0;
            if (far < 0 || v > best) far = q, best = v;
        }
        if (in_seeds(far)) break;
        seeds.push_back(far);
        const std::vector<double> dn = dijkstra(g.csr, w_sym, {far}).dist;
        const int64_t lab = static_cast<int64_t>(seeds.size()) - 1;
        for (int64_t q = 0; q < N; ++q) {
            if (dn[q] < d_final[q]) out.region[q] = lab;
            d_final[q] = np_minimum(d_final[q], dn[q]);
        }
    }
    // 兜底：日常模式图上不可达、又没被选为种子的节点并入角距最近的种子
    for (int64_t u = 0; u < N; ++u) {
        if (out.region[u] >= 0) continue;
        const Vec3 xu = xyz_of(isl, u);
        int64_t bi = 0;
        double bd = INF;
        for (size_t k = 0; k < seeds.size(); ++k) {
            const double dd = angdist(xu, xyz_of(isl, seeds[k]));
            if (k == 0 || dd < bd) bi = static_cast<int64_t>(k), bd = dd;
        }
        out.region[u] = bi;
    }

    // ---- 中心间干线（⑦b：仅展示与九格表用）：使节可通的物理成本
    std::vector<double> w_env(E2);
    for (int64_t k = 0; k < E2; ++k) w_env[k] = g.perm_d[static_cast<size_t>(k) * N_MODES + M_ENVOY] > 0 ? r.cost[k] : INF;
    for (int a = 0; a < 3; ++a) {
        std::vector<int> bs;
        for (int bq = 0; bq < 3; ++bq)
            if (std::string(CENTER_KEYS[a]) < std::string(CENTER_KEYS[bq])) bs.push_back(bq);
        if (bs.empty()) continue;
        const Paths t = dijkstra(g.csr, w_env, {out.node[a]});
        for (int bq : bs) {
            Trunk tr;
            tr.key = std::string(CENTER_KEYS[a]) + "->" + CENTER_KEYS[bq];
            const int64_t tgt = out.node[bq];
            if (std::isfinite(t.dist[tgt])) {
                tr.reachable = true;
                tr.cost_days = t.dist[tgt];
                for (int64_t u = tgt; u >= 0; u = t.pred_node[u]) tr.path.push_back(u);
                std::reverse(tr.path.begin(), tr.path.end());
            }
            out.trunks.push_back(std::move(tr));
        }
    }
    return out;
}

}  // namespace skyisle::planet
