// ⑥ 航路（skyisle_gen/stages/s06_routes.py）：沿边的风 / 风暴取样 → 有向成本（顺风廉价逆风昂贵、无风惩罚、爬升）与分模式成本
// → 按岛密度 × 集雨容量加权抽源的 Brandes 抽样介数 → 枢纽（全局与 G 邻域的流量分位）→ 每模式的弱连通分量。
// 另：⑥ 之后各步共用的有向图（weights.load_directed）与 λ_ref。
#include <algorithm>
#include <cmath>
#include <numeric>

#include "skyisle/planet/civ.hpp"

namespace skyisle::planet {

namespace {

inline double sum3(double a, double b, double c) { return ((0.0 + a) + b) + c; }

inline void xyz_to_latlon(double x, double y, double z, double& lat, double& lon) {
    lat = rad2deg(std::asin(clip(z, -1.0, 1.0)));
    lon = rad2deg(std::atan2(y, x));
}

// sphere.initial_bearing（弧度，0 = 正北，顺时针）
inline double initial_bearing(double lat1, double lon1, double lat2, double lon2) {
    const double p1 = deg2rad(lat1), l1 = deg2rad(lon1), p2 = deg2rad(lat2), l2 = deg2rad(lon2);
    const double dl = l2 - l1;
    const double y = std::sin(dl) * std::cos(p2);
    const double x = std::cos(p1) * std::sin(p2) - std::sin(p1) * std::cos(p2) * std::cos(dl);
    return std::atan2(y, x);
}

}  // namespace

std::array<double, N_MODES> lambda_ref(const Config& cfg) {
    std::array<double, N_MODES> lam{};
    for (int m = 0; m < N_MODES; ++m) {
        const std::vector<double>& dh = cfg.list(std::string("s08.half_distance_days.") + MODE_KEYS[m]);
        lam[m] = std::log(2.0) / std::sqrt(dh[0] * dh[1]);
    }
    return lam;
}

Directed directed(const Islands& isl, const Barriers& b, const Routes& r) {
    Directed g;
    g.N = static_cast<int64_t>(isl.n());
    g.E = b.E;
    g.csr = make_csr(g.N, r.src_d, r.dst_d);
    const size_t EM = static_cast<size_t>(b.E) * N_MODES;
    g.perm_d.resize(2 * EM);
    std::copy(b.perm.begin(), b.perm.end(), g.perm_d.begin());
    std::copy(b.perm.begin(), b.perm.end(), g.perm_d.begin() + EM);
    g.L.resize(2 * EM);
    for (size_t k = 0; k < 2 * EM; ++k) g.L[k] = g.perm_d[k] > 0 ? -std::log(np_maximum(g.perm_d[k], 1e-300)) : INF;
    return g;
}

std::vector<double> mode_weight(const Routes& r, const Directed& g, int mi, double lam) {
    const size_t n = r.cost.size();
    std::vector<double> w(n);
    for (size_t k = 0; k < n; ++k) w[k] = lam * r.cost_m[k * N_MODES + mi] + g.L[k * N_MODES + mi];
    return w;
}

Routes stage6(const Config& cfg, uint64_t seed, const Winds& wd, const Islands& isl, const Climate& c, const Barriers& b, int threads) {
    const std::string R = "s06.routes.";
    const int64_t N = static_cast<int64_t>(isl.n());
    const int64_t E = static_cast<int64_t>(isl.src.size());
    Routes out;

    // ---- 沿边采样：风（第 ⌊n/2⌋ 个采样点）、风暴（最大）
    const int ns = static_cast<int>(cfg.get(R + "edge_samples"));
    std::vector<double> tt(ns);
    {
        const int div = ns - 1;   // np.linspace(0, 1, n)：arange · (1 / (n−1))，末项 = 1
        const double step = div > 0 ? 1.0 / static_cast<double>(div) : 0.0;
        for (int k = 0; k < ns; ++k) tt[k] = div > 0 ? static_cast<double>(k) * step + 0.0 : 0.0;
        if (ns > 1) tt[ns - 1] = 1.0;
    }
    const LatLonGrid G = llg(c.ax);
    const std::vector<double> storm = f32v(c.storm), storm_ng_f = f32v(c.storm_no_g), wu = f32v(c.u), wv = f32v(c.v),
                              vloc_f = f32v(c.v_local);
    std::vector<double> storm_max(E), storm_ng(E), mlat(E), mlon(E), wind_dir(E), calm(E), storm_f(E), storm_f_ng(E);
    const double calm_max = cfg.get(R + "calm_max"), calm_kappa = cfg.get(R + "calm_kappa"), calm_v_ref = cfg.get(R + "calm_v_ref");
    const double storm_kappa = cfg.get(R + "storm_kappa");
    for (int64_t e = 0; e < E; ++e) {
        const double* a = &isl.xyz[static_cast<size_t>(isl.src[e]) * 3];
        const double* bb = &isl.xyz[static_cast<size_t>(isl.dst[e]) * 3];
        const Vec3 va{a[0], a[1], a[2]}, vb{bb[0], bb[1], bb[2]};
        const double omega = angdist(va, vb);
        const double so = std::sin(omega);
        double smax = -INF, sngmax = -INF;
        for (int k = 0; k < ns; ++k) {
            const double t = tt[k];
            const double wa = so > 1e-12 ? std::sin((1 - t) * omega) / so : 1 - t;
            const double wb = so > 1e-12 ? std::sin(t * omega) / so : t;
            double px = wa * a[0] + wb * bb[0], py = wa * a[1] + wb * bb[1], pz = wa * a[2] + wb * bb[2];
            const double nm = std::sqrt(sum3(px * px, py * py, pz * pz));
            px /= nm;
            py /= nm;
            pz /= nm;
            double la, lo;
            xyz_to_latlon(px, py, pz, la, lo);
            const double s1 = grid_interp(storm, G, la, lo), s2 = grid_interp(storm_ng_f, G, la, lo);
            smax = k == 0 ? s1 : std::max(smax, s1);
            sngmax = k == 0 ? s2 : std::max(sngmax, s2);
            if (k == ns / 2) {
                mlat[e] = la;
                mlon[e] = lo;
            }
        }
        storm_max[e] = smax;
        storm_ng[e] = sngmax;
        const double u = grid_interp(wu, G, mlat[e], mlon[e]), v = grid_interp(wv, G, mlat[e], mlon[e]);
        const double v_wind = np_hypot(u, v);
        wind_dir[e] = std::atan2(u, v);   // 方位角：0 = 北，顺时针
        const double v_eff = np_maximum(v_wind, grid_interp(vloc_f, G, mlat[e], mlon[e]));
        calm[e] = np_minimum(calm_max, 1.0 + calm_kappa * np_maximum(0.0, 1.0 - v_eff / calm_v_ref));
        storm_f[e] = 1.0 + storm_kappa * storm_max[e];
        storm_f_ng[e] = 1.0 + storm_kappa * storm_ng[e];
    }

    // ---- 有向成本
    const double tail = cfg.get(R + "tailwind_factor"), head = cfg.get(R + "headwind_factor");
    const double k_tail = 1.0 / tail - 1.0, k_head = head - 1.0;
    const double climb_kappa = cfg.get(R + "climb_kappa");
    double alpha[N_MODES];
    for (int m = 0; m < N_MODES; ++m) alpha[m] = cfg.get(R + "alpha." + MODE_KEYS[m]);
    out.cost.resize(2 * E);
    out.cost_m.resize(static_cast<size_t>(2 * E) * N_MODES);
    for (int dir = 0; dir < 2; ++dir) {
        for (int64_t e = 0; e < E; ++e) {
            const int64_t ai = dir == 0 ? isl.src[e] : isl.dst[e], bi = dir == 0 ? isl.dst[e] : isl.src[e];
            const double brg = initial_bearing(mlat[e], mlon[e], isl.lat[bi], isl.lon[bi]);
            const double cc = std::cos(wind_dir[e] - brg);
            const double df = cc >= 0 ? 1.0 / (1.0 + k_tail * cc) : 1.0 + k_head * (-cc);
            const double w = df * calm[e];
            const double climb = 1.0 + climb_kappa * np_maximum(0.0, f32(isl.height[bi]) - f32(isl.height[ai])) / 1000.0;
            const double base = isl.dist_days[e] * storm_f[e] * climb;
            const size_t k = static_cast<size_t>(dir * E + e);
            out.cost[k] = base * w;
            for (int m = 0; m < N_MODES; ++m) out.cost_m[k * N_MODES + m] = base * np_pow(w, alpha[m]);
        }
    }
    out.cost_no_g.resize(2 * E);
    for (int dir = 0; dir < 2; ++dir)
        for (int64_t e = 0; e < E; ++e) {
            const size_t k = static_cast<size_t>(dir * E + e);
            out.cost_no_g[k] = out.cost[k] * (storm_f_ng[e] / storm_f[e]);   // 反事实：从未有过 G（P6 用）
        }
    out.src_d.resize(2 * E);
    out.dst_d.resize(2 * E);
    out.und_id.resize(2 * E);
    for (int64_t e = 0; e < E; ++e) {
        out.src_d[e] = isl.src[e];
        out.src_d[E + e] = isl.dst[e];
        out.dst_d[e] = isl.dst[e];
        out.dst_d[E + e] = isl.src[e];
        out.und_id[e] = out.und_id[E + e] = e;
    }
    const CSR csr = make_csr(N, out.src_d, out.dst_d);

    // ---- 抽样介数（商旅可通的物理成本图）：源权重 = 岛密度 × 集雨容量（纯地理量，不读文明中心）
    Rng rng = stage_rng(seed, 6);
    std::vector<double> w_src(N);
    for (int64_t q = 0; q < N; ++q) w_src[q] = f32(isl.density_at[q]) * f32(c.i_catch[q]);
    const double wsum = np_sum(w_src.data(), w_src.size());
    for (double& x : w_src) x /= wsum;
    out.n_sources = std::min<int64_t>(static_cast<int64_t>(cfg.get(R + "betweenness_sources")), N);
    out.sources = rng.choice_noreplace_p(w_src, out.n_sources);
    std::sort(out.sources.begin(), out.sources.end());
    std::vector<double> w_bt(2 * E);
    for (int64_t k = 0; k < 2 * E; ++k) {
        const int64_t e = k < E ? k : k - E;
        w_bt[k] = b.perm[static_cast<size_t>(e) * N_MODES + M_TRADE] > 0 ? out.cost[k] : INF;
    }
    out.flow = betweenness_sampled(csr, w_bt, 2 * E, out.sources, cfg.get(R + "betweenness_c_min_days"), 1e-9, threads);
    out.node_flow.assign(N, 0.0);
    for (int64_t k = 0; k < 2 * E; ++k) out.node_flow[out.src_d[k]] += out.flow[k];   // np.add.at：按下标次序
    for (int64_t k = 0; k < 2 * E; ++k) out.node_flow[out.dst_d[k]] += out.flow[k];
    for (double& x : out.node_flow) x *= 0.5;

    // ---- 枢纽：全局 top 分位 ∪ G 邻域内 top 分位
    const double q_global = np_quantile(out.node_flow, 1.0 - cfg.get(R + "hub_top_frac"));
    std::vector<uint8_t> is_hub(N, 0);
    for (int64_t q = 0; q < N; ++q) is_hub[q] = out.node_flow[q] >= q_global;
    const Vec3 gx = latlon_to_xyz(wd.g.lat, wd.g.lon);
    const double reach = cfg.get(R + "g_neighborhood_radii") * wd.g.radius_deg;
    out.near_g.assign(N, 0);
    std::vector<double> near_flow;
    for (int64_t q = 0; q < N; ++q) {
        const Vec3 p{isl.xyz[3 * q], isl.xyz[3 * q + 1], isl.xyz[3 * q + 2]};
        out.near_g[q] = rad2deg(angdist(p, gx)) <= reach;
        if (out.near_g[q]) near_flow.push_back(out.node_flow[q]);
    }
    if (!near_flow.empty()) {
        const double qg = np_quantile(near_flow, 1.0 - cfg.get(R + "hub_g_top_frac"));
        for (int64_t q = 0; q < N; ++q)
            if (out.near_g[q] && out.node_flow[q] >= qg && out.node_flow[q] > 0) is_hub[q] = 1;
    }
    for (int64_t q = 0; q < N; ++q)
        if (is_hub[q]) out.hubs.push_back(q);
    std::stable_sort(out.hubs.begin(), out.hubs.end(), [&](int64_t a, int64_t bq) {
        return out.node_flow[a] != out.node_flow[bq] ? -out.node_flow[a] < -out.node_flow[bq] : a < bq;
    });

    // ---- 连通分量（每模式）
    for (int m = 0; m < N_MODES; ++m) {
        std::vector<int64_t> s, d;
        for (int64_t e = 0; e < E; ++e)
            if (b.perm[static_cast<size_t>(e) * N_MODES + m] > 0) {
                s.push_back(isl.src[e]);
                d.push_back(isl.dst[e]);
            }
        const std::vector<int64_t> comp = weak_components(N, s, d);
        int64_t nc = 0;
        for (int64_t x : comp) nc = std::max(nc, x + 1);
        std::vector<int64_t> sizes(nc, 0);
        for (int64_t x : comp) ++sizes[x];
        out.n_components[m] = nc;
        out.largest[m] = nc ? *std::max_element(sizes.begin(), sizes.end()) : 0;
    }
    return out;
}

}  // namespace skyisle::planet
