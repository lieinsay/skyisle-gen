// ⑤ 障碍（skyisle_gen/stages/s05_barriers.py）：区域障碍的穿越坐标 Φ → 边指数 f → 四模式通过率；边局部因子；G 的大圆弧段阻断。
// 运算次序照 numpy：Python 浮点 ** 数组 = 逐个 C 的 pow（c_pow）；3 维的 np.sum / np.linalg.norm = ((0 + x₀) + x₁) + x₂；
// np.maximum / minimum 相等取第二个；f.sum(axis=1) 的 6 列顺序相加。
#include <cmath>
#include <stdexcept>

#include "skyisle/planet/civ.hpp"

namespace skyisle::planet {

const char* const MODE_KEYS[N_MODES] = {"daily", "trade", "envoy", "migrate"};
const char* const REGIONAL_ORDER[N_REGIONAL] = {"A", "B", "C", "D", "F_N", "F_S"};
const char* const CENTER_KEYS[3] = {"north_west", "north_east", "south"};

int mode_index(const std::string& m) {
    for (int k = 0; k < N_MODES; ++k)
        if (m == MODE_KEYS[k]) return k;
    throw std::invalid_argument("unknown mode: " + m);
}

namespace {

inline double sum3(double a, double b, double c) { return ((0.0 + a) + b) + c; }
inline double norm3(const Vec3& v) { return std::sqrt(sum3(v.x * v.x, v.y * v.y, v.z * v.z)); }

// sphere.xyz_to_latlon
inline void xyz_to_latlon(const Vec3& p, double& lat, double& lon) {
    lat = rad2deg(std::asin(clip(p.z, -1.0, 1.0)));
    lon = rad2deg(std::atan2(p.y, p.x));
}

// s02_wind.local_edges：局部带界按经度线性插值（edges 从 npz 读，按 float32）
struct LocalEdges {
    std::vector<double> lons, edges;   // edges [8, W]（已 f32）
    int W = 0;
    double at(int key, double lon) const {
        const double res = lons[1] - lons[0];
        const double fj = (lon - lons[0]) / res;
        const int64_t j0f = static_cast<int64_t>(std::floor(fj));
        const double t = fj - static_cast<double>(j0f);
        const int64_t j0 = ((j0f % W) + W) % W, j1 = (j0 + 1) % W;
        const double* row = &edges[static_cast<size_t>(key) * W];
        return row[j0] * (1 - t) + row[j1] * t;
    }
};

}  // namespace

Barriers stage5(const Config& cfg, const Planet& p, const Winds& w, const Islands& isl, const Climate& c) {
    const size_t N = isl.n();
    const int64_t E = static_cast<int64_t>(isl.src.size());
    Barriers B;
    B.E = E;
    const double band_scale = p.band_scale;
    const double perm_min = cfg.get("s05.perm_min");
    LocalEdges le;
    le.lons = c.ax.lons;
    le.edges = f32v(c.edges);
    le.W = c.ax.nlon;

    // ---- 区域障碍 Φ（node_phi）与边指数 f
    B.f.assign(static_cast<size_t>(E) * N_REGIONAL, 0.0);
    std::vector<double> phi(N);
    for (int bi = 0; bi < N_REGIONAL; ++bi) {
        const std::string pre = std::string("s05.barriers.") + REGIONAL_ORDER[bi] + ".";
        const std::string kind = cfg.gets(pre + "kind");
        if (kind == "eq_core") {
            const double core = cfg.get("skeleton.eq_core_halfwidth_deg");
            for (size_t q = 0; q < N; ++q) phi[q] = clip((isl.lat[q] + core) / (2 * core), 0.0, 1.0);
        } else if (kind == "band") {
            const std::string band = cfg.gets(pre + "band");
            int klo, khi;   // EDGE_KEYS：eq_n trades_n calm_n west_n eq_s trades_s calm_s west_s
            if (band == "subtropical_calm_n") klo = 1, khi = 2;
            else if (band == "subtropical_calm_s") klo = 6, khi = 5;
            else if (band == "westerlies_n") klo = 2, khi = 3;
            else if (band == "westerlies_s") klo = 7, khi = 6;
            else throw std::invalid_argument("unknown band: " + band);
            for (size_t q = 0; q < N; ++q) {
                const double lo = le.at(klo, isl.lon[q]), hi = le.at(khi, isl.lon[q]);
                phi[q] = clip((isl.lat[q] - lo) / (hi - lo), 0.0, 1.0);
            }
        } else if (kind == "lat_band") {
            const std::vector<double>& lr = cfg.list(pre + "lat_range");
            const double lo = lr[0] * band_scale, hi = lr[1] * band_scale;
            for (size_t q = 0; q < N; ++q) phi[q] = clip((isl.lat[q] - lo) / (hi - lo), 0.0, 1.0);
        } else if (kind == "void") {
            const double lon_w = cfg.get("skeleton.d_lon_west"), lon_e = cfg.get("skeleton.d_lon_east");
            const double width = pymod(lon_e - lon_w, 360.0);
            const std::vector<double>& dr = cfg.list("skeleton.d_lat_range");
            const double d_lo = dr[0] * band_scale, d_hi = dr[1] * band_scale;
            for (size_t q = 0; q < N; ++q) {
                const double delta = pymod(isl.lon[q] - lon_w, 360.0);
                const bool inside = delta <= width;
                const bool east_side = (delta - width) <= (360.0 - delta);
                const double ph = inside ? delta / width : (east_side ? 1.0 : 0.0);
                const bool in_lat = (isl.lat[q] >= d_lo) && (isl.lat[q] <= d_hi);
                phi[q] = in_lat ? ph : NaN;
            }
        } else {
            throw std::invalid_argument("unknown regional barrier kind: " + kind);
        }
        for (int64_t e = 0; e < E; ++e) {
            const double fb = std::fabs(phi[isl.dst[e]] - phi[isl.src[e]]);
            B.f[static_cast<size_t>(e) * N_REGIONAL + bi] = std::isnan(fb) ? 0.0 : fb;
        }
    }

    B.perm.assign(static_cast<size_t>(E) * N_MODES, 1.0);
    for (int bi = 0; bi < N_REGIONAL; ++bi) {
        const std::string pre = std::string("s05.barriers.") + REGIONAL_ORDER[bi] + ".permeability.";
        double logP[N_MODES];
        for (int m = 0; m < N_MODES; ++m) {
            const double P = cfg.get(pre + MODE_KEYS[m]);
            logP[m] = P > 0 ? std::log(P) : -INF;
        }
        for (int64_t e = 0; e < E; ++e) {
            const double fe = B.f[static_cast<size_t>(e) * N_REGIONAL + bi];
            for (int m = 0; m < N_MODES; ++m) {
                const double contrib = fe > 0 ? fe * logP[m] : 0.0;   // 0 × (−inf) → 0
                B.perm[static_cast<size_t>(e) * N_MODES + m] *= std::exp(contrib);
            }
        }
    }
    std::vector<uint8_t> crosses(E);
    for (int64_t e = 0; e < E; ++e) {
        const double* fe = &B.f[static_cast<size_t>(e) * N_REGIONAL];
        double s = 0.0;
        for (int k = 0; k < N_REGIONAL; ++k) s += fe[k];
        crosses[e] = s > 0.05;
    }

    // ---- 边局部因子（只作用于不跨区域障碍的边）
    const size_t EM = static_cast<size_t>(E) * N_MODES;
    const double small_days = cfg.get("shared.ships.small_days");
    // 宽阔无岛空域：P^((span − small)/ref)
    B.gap.assign(EM, 1.0);
    {
        const double ref = cfg.get("s05.local.gap.ref_days");
        for (int m = 0; m < N_MODES; ++m) {
            const double P = cfg.get(std::string("s05.local.gap.permeability.") + MODE_KEYS[m]);
            for (int64_t e = 0; e < E; ++e) {
                const double span = np_maximum(0.0, isl.dist_days[e] - small_days) / ref;
                B.gap[static_cast<size_t>(e) * N_MODES + m] = crosses[e] ? 1.0 : c_pow(P, span);
            }
        }
    }
    // 岛密度骤降：P^(|ln ρu/ρv| / ln ratio_ref)
    B.density_drop.assign(EM, 1.0);
    {
        const double lnref = std::log(cfg.get("s05.local.density_drop.ratio_ref"));
        std::vector<double> ratio(E);
        for (int64_t e = 0; e < E; ++e) {
            const double ru = np_maximum(f32(isl.density_at[isl.src[e]]), 1e-9), rv = np_maximum(f32(isl.density_at[isl.dst[e]]), 1e-9);
            ratio[e] = std::fabs(std::log(ru / rv)) / lnref;
        }
        for (int m = 0; m < N_MODES; ++m) {
            const double P = cfg.get(std::string("s05.local.density_drop.permeability.") + MODE_KEYS[m]);
            for (int64_t e = 0; e < E; ++e) B.density_drop[static_cast<size_t>(e) * N_MODES + m] = crosses[e] ? 1.0 : c_pow(P, ratio[e]);
        }
    }
    // 高度落差：筛「谁付得起」（原则乙：不筛贵贱）
    B.climb.assign(EM, 1.0);
    {
        const double thr = cfg.get("s05.local.climb.h_thr_m");
        for (int64_t e = 0; e < E; ++e) {
            const double dh = std::fabs(f32(isl.height[isl.src[e]]) - f32(isl.height[isl.dst[e]]));
            if (dh > thr)
                for (int m = 0; m < N_MODES; ++m)
                    B.climb[static_cast<size_t>(e) * N_MODES + m] = cfg.get(std::string("s05.local.climb.permeability.") + MODE_KEYS[m]);
        }
    }
    // 政治性障碍（人为、可变；默认空）：边中点落在经纬矩形里
    B.political.assign(EM, 1.0);
    {
        const int n_ov = static_cast<int>(cfg.get("s05.political.overrides.n", 0.0));
        if (n_ov > 0) {
            std::vector<double> mlat(E), mlon(E);
            for (int64_t e = 0; e < E; ++e) {
                const double* a = &isl.xyz[static_cast<size_t>(isl.src[e]) * 3];
                const double* b = &isl.xyz[static_cast<size_t>(isl.dst[e]) * 3];
                Vec3 mid{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
                const double nm = norm3(mid);
                mid = {mid.x / nm, mid.y / nm, mid.z / nm};
                xyz_to_latlon(mid, mlat[e], mlon[e]);
            }
            for (int o = 0; o < n_ov; ++o) {
                const std::string pre = "s05.political.overrides." + std::to_string(o) + ".";
                const std::vector<double>& la = cfg.list(pre + "lat_range");
                const std::vector<double>& lo = cfg.list(pre + "lon_range");
                const double span = pymod(lo[1] - lo[0], 360.0);
                for (int64_t e = 0; e < E; ++e) {
                    const bool hit = (mlat[e] >= la[0]) && (mlat[e] <= la[1]) && (pymod(mlon[e] - lo[0], 360.0) <= span);
                    if (!hit) continue;
                    for (int m = 0; m < N_MODES; ++m) B.political[static_cast<size_t>(e) * N_MODES + m] *= cfg.get(pre + "permeability." + MODE_KEYS[m]);
                }
            }
        }
    }
    for (const std::vector<double>* lp : {&B.gap, &B.density_drop, &B.climb, &B.political})
        for (size_t k = 0; k < EM; ++k) B.perm[k] *= (*lp)[k];

    // ---- G：绝对阻断，但只阻断一个点（改道型）：边的大圆弧段到 G 心的最小角距 < R_G
    const Vec3 g = latlon_to_xyz(w.g.lat, w.g.lon);
    const double g_r = deg2rad(w.g.radius_deg);
    B.g_blocked.assign(E, 0);
    for (int64_t e = 0; e < E; ++e) {
        const double* pa = &isl.xyz[static_cast<size_t>(isl.src[e]) * 3];
        const double* pb = &isl.xyz[static_cast<size_t>(isl.dst[e]) * 3];
        const Vec3 a{pa[0], pa[1], pa[2]}, b{pb[0], pb[1], pb[2]};
        const double da = angdist(a, g), db = angdist(b, g);
        const Vec3 n = cross(a, b);
        const double nn = norm3(n);
        const Vec3 nh = nn > 1e-12 ? Vec3{n.x / nn, n.y / nn, n.z / nn} : n;
        const double dot = sum3(nh.x * g.x, nh.y * g.y, nh.z * g.z);
        const double ct = std::fabs(std::asin(clip(dot, -1.0, 1.0)));
        Vec3 proj{g.x - dot * nh.x, g.y - dot * nh.y, g.z - dot * nh.z};
        const double pn = norm3(proj);
        proj = pn > 1e-12 ? Vec3{proj.x / pn, proj.y / pn, proj.z / pn} : a;
        const double seg = angdist(a, b);
        const bool within = (angdist(a, proj) + angdist(proj, b)) <= seg + 1e-9;
        const double min_d = within ? ct : np_minimum(da, db);
        B.g_blocked[e] = min_d < g_r;
    }
    B.perm_no_g.resize(EM);
    for (size_t k = 0; k < EM; ++k) B.perm_no_g[k] = B.perm[k] < perm_min ? 0.0 : clip(B.perm[k], 0.0, 1.0);   // 反事实：没有 G 的世界
    for (int64_t e = 0; e < E; ++e)
        if (B.g_blocked[e])
            for (int m = 0; m < N_MODES; ++m) B.perm[static_cast<size_t>(e) * N_MODES + m] = 0.0;
    for (size_t k = 0; k < EM; ++k) B.perm[k] = B.perm[k] < perm_min ? 0.0 : clip(B.perm[k], 0.0, 1.0);
    return B;
}

}  // namespace skyisle::planet
