// ③ 岛群分布（s03_islands.py）与浮石板块（tectonics.py）。
// 随机数只从 stage_rng(seed, 3) 取，抽取次序与 Python 版相同：板块（种子、丰度、边界类型表、热点）→ 残余噪声 → 撒点 → 高度噪声 → 高度抖动
// → 叠层 → 陆地占比抖动 → 可用地率 → 主岛占比。
#include "skyisle/planet/planet.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace skyisle::planet {

namespace {

constexpr int CONVERGENT = 0, DIVERGENT = 1, TRANSFORM = 2;
constexpr int KIND_KNN = 0, KIND_FAR = 1, KIND_EXPEDITION = 2, KIND_FALLBACK = 3;
constexpr double HEX_FACTOR = 0.866;

std::vector<Vec3> random_points(Rng& rng, int n, double lat_max = 90.0) {
    const double s = std::sin(deg2rad(lat_max));
    std::vector<double> z(n), lon(n);
    for (int k = 0; k < n; ++k) z[k] = rng.uniform(-s, s);
    for (int k = 0; k < n; ++k) lon[k] = rng.uniform(-180.0, 180.0);
    std::vector<Vec3> out(n);
    for (int k = 0; k < n; ++k) out[k] = latlon_to_xyz(rad2deg(std::asin(z[k])), lon[k]);
    return out;
}

// numpy 的 1 维 norm（x.dot(x) 走 BLAS ddot：按 k 顺序 FMA）
double norm_ddot(const Vec3& a) { return std::sqrt(dot_gemm(a, a)); }

}  // namespace

Tectonics plate_fields(Rng& rng, const Axes& ax, const Config& cfg, const std::vector<double>* wind_u, const std::vector<double>* wind_v) {
    const size_t M = ax.size();
    std::vector<Vec3> pts(M);
    for (int i = 0; i < ax.nlat; ++i)
        for (int j = 0; j < ax.nlon; ++j) pts[static_cast<size_t>(i) * ax.nlon + j] = latlon_to_xyz(ax.lats[i], ax.lons[j]);
    const int n_pl = cfg.geti("s03.plates.n_plates");
    const std::vector<Vec3> seeds = random_points(rng, n_pl);
    std::vector<double> plate_mult(n_pl);
    const double psig = cfg.get("s03.plates.plate_sigma");
    for (int k = 0; k < n_pl; ++k) plate_mult[k] = std::exp(rng.normal(0.0, psig));
    std::vector<double> pair_u(static_cast<size_t>(n_pl) * n_pl);
    rng.uniform_fill(0.0, 1.0, pair_u.data(), pair_u.size());
    const double p_c = cfg.get("s03.plates.p_convergent"), p_d = cfg.get("s03.plates.p_divergent");
    const double p_cd = p_c + p_d;
    std::vector<int8_t> btab(static_cast<size_t>(n_pl) * n_pl);
    for (int a = 0; a < n_pl; ++a)
        for (int b = 0; b < n_pl; ++b) {
            // triu(U, 1) + triu(U, 1).T：上三角原值、下三角取对称、对角 0
            const double u = a < b ? pair_u[static_cast<size_t>(a) * n_pl + b] + 0.0
                                   : (a > b ? 0.0 + pair_u[static_cast<size_t>(b) * n_pl + a] : 0.0);
            btab[static_cast<size_t>(a) * n_pl + b] = static_cast<int8_t>(u < p_c ? CONVERGENT : (u < p_cd ? DIVERGENT : TRANSFORM));
        }

    Tectonics T;
    T.seeds_xyz.resize(static_cast<size_t>(n_pl) * 3);
    for (int k = 0; k < n_pl; ++k) {
        T.seeds_xyz[3 * k] = seeds[k].x;
        T.seeds_xyz[3 * k + 1] = seeds[k].y;
        T.seeds_xyz[3 * k + 2] = seeds[k].z;
    }
    const double w = cfg.get("s03.plates.boundary_width_deg");
    const double seg = cfg.get("s03.plates.transform_segment_deg");
    const double cb = cfg.get("s03.plates.convergent_boost"), dc = cfg.get("s03.plates.divergent_cut"),
                 tb = cfg.get("s03.plates.transform_boost");
    const double age_scale = cfg.get("s03.plates.age_scale_deg");
    T.factor.resize(M);
    T.conv_kernel.resize(M);
    T.age.resize(M);
    T.boundary_kernel.resize(M);
    T.plate_id.resize(M);
    T.btype.resize(M);
    for (size_t m = 0; m < M; ++m) {
        // 前两名（argsort(−dots)[:, :2]；并列时按下标）
        int i1 = -1, i2 = -1;
        double d1v = 0, d2v = 0;
        for (int k = 0; k < n_pl; ++k) {
            const double d = clip(dot_gemm(pts[m], seeds[k]), -1.0, 1.0);   // pts @ seeds.T（dgemm）
            if (i1 < 0 || d > d1v) {
                i2 = i1;
                d2v = d1v;
                i1 = k;
                d1v = d;
            } else if (i2 < 0 || d > d2v) {
                i2 = k;
                d2v = d;
            }
        }
        const double d1 = std::acos(d1v), d2 = std::acos(d2v);
        const double dist_deg = rad2deg(0.5 * (d2 - d1));
        const double z = dist_deg / w;
        const double kern = std::exp(-0.5 * (z * z));
        const int bt = btab[static_cast<size_t>(i1) * n_pl + i2];
        Vec3 nv = cross(seeds[i1], seeds[i2]);
        const double nn = np_maximum(std::sqrt(((0.0 + nv.x * nv.x) + nv.y * nv.y) + nv.z * nv.z), 1e-12);
        nv = {nv.x / nn, nv.y / nn, nv.z / nn};
        const double sa = rad2deg(std::asin(clip(((0.0 + pts[m].x * nv.x) + pts[m].y * nv.y) + pts[m].z * nv.z, -1.0, 1.0)));
        const double trans_mod = 0.5 + 0.5 * std::sin(2.0 * PI * sa / seg);
        const double bfac = bt == CONVERGENT ? 1.0 + cb * kern : (bt == DIVERGENT ? 1.0 - dc * kern : 1.0 + tb * kern * trans_mod);
        T.factor[m] = plate_mult[i1] * bfac;
        T.conv_kernel[m] = bt == CONVERGENT ? kern : 0.0;
        double age = 0.3 + 0.7 * clip(dist_deg / age_scale, 0.0, 1.0);
        if (bt == DIVERGENT) age = 0.5 * age;
        T.age[m] = age;
        T.boundary_kernel[m] = kern;
        T.plate_id[m] = static_cast<int16_t>(i1);
        T.btype[m] = static_cast<int8_t>(bt);
    }

    // 热点链：固定点 + 漂移方向 → 线性岛链，年龄沿链单调（t = 0 最新端）
    const int n_hot = cfg.geti("s03.plates.n_hotspots");
    std::vector<double> hot(M, 0.0);
    if (n_hot > 0) {
        const std::vector<Vec3> starts = random_points(rng, n_hot, 70.0);
        std::vector<double> bearings(n_hot);
        for (int k = 0; k < n_hot; ++k) bearings[k] = rng.uniform(0.0, 2.0 * PI);
        const double length = deg2rad(cfg.get("s03.plates.hotspot_length_deg"));
        const double spacing = deg2rad(cfg.get("s03.plates.hotspot_spacing_deg"));
        const double r_h = deg2rad(cfg.get("s03.plates.hotspot_radius_deg"));
        const int n_pts = std::max(2, static_cast<int>(std::nearbyint(length / spacing)) + 1);
        for (int k = 0; k < n_hot; ++k) {
            const Vec3 c = starts[k];
            Vec3 east{-c.y, c.x, 0.0};
            const double en = std::max(norm_ddot(east), 1e-12);
            east = {east.x / en, east.y / en, east.z / en};
            const Vec3 north = cross(c, east);
            const double cbk = std::cos(bearings[k]), sbk = std::sin(bearings[k]);
            const Vec3 tdir{cbk * north.x + sbk * east.x, cbk * north.y + sbk * east.y, cbk * north.z + sbk * east.z};
            for (int m = 0; m < n_pts; ++m) {
                const double t = static_cast<double>(m) / static_cast<double>(n_pts - 1);
                const double ang = t * length;
                const double ca = std::cos(ang), sa = std::sin(ang);
                const Vec3 q{ca * c.x + sa * tdir.x, ca * c.y + sa * tdir.y, ca * c.z + sa * tdir.z};
                const double fall = 1.0 - 0.6 * t;
                const double near_r = 1.5 * r_h;
                for (size_t p = 0; p < M; ++p) {
                    const double dq = std::acos(clip(dot_gemv(pts[p], q), -1.0, 1.0));   // pts @ q（dgemv）
                    const double zz = dq / r_h;
                    const double bump = std::exp(-0.5 * (zz * zz)) * fall;
                    hot[p] = np_maximum(hot[p], bump);
                    if (dq < near_r && bump > 0.3) T.age[p] = t;
                }
            }
        }
    }
    const double hb = cfg.get("s03.plates.hotspot_boost");
    for (size_t m = 0; m < M; ++m) T.factor[m] = T.factor[m] * (1.0 + hb * hot[m]);

    // ② 风场辐合纹理
    const double kappa = cfg.get("s03.plates.convergence_kappa");
    if (wind_u && wind_v && kappa != 0.0) {
        std::vector<double> conv = divergence(*wind_u, *wind_v, ax);
        for (double& x : conv) x = -x;
        std::vector<double> mid;
        for (int i = 0; i < ax.nlat; ++i)
            if (std::fabs(ax.lats[i]) < 80.0)
                for (int j = 0; j < ax.nlon; ++j) mid.push_back(conv[static_cast<size_t>(i) * ax.nlon + j]);
        double sd = np_std(mid);
        if (sd == 0.0) sd = 1.0;
        for (size_t m = 0; m < M; ++m) T.factor[m] = T.factor[m] * std::exp(kappa * clip(conv[m] / sd, -2.0, 2.0));
    }
    for (double& a : T.age) a = clip(a, 0.0, 1.0);
    return T;
}

namespace {

struct EdgeRec {
    double d;
    int kind;
};

// 无向（弱）连通分量：分量号按最小成员排（graph.weak_components 的并查集总把大根挂到小根下，根 = 最小成员，与边的次序无关）
std::vector<int64_t> weak_components(int64_t n, const std::map<std::pair<int64_t, int64_t>, EdgeRec>& edges, int64_t& n_comp) {
    std::vector<int64_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int64_t x) {
        int64_t r = x;
        while (parent[r] != r) r = parent[r];
        while (parent[x] != r) {
            const int64_t nx = parent[x];
            parent[x] = r;
            x = nx;
        }
        return r;
    };
    for (const auto& kv : edges) {
        const int64_t ra = find(kv.first.first), rb = find(kv.first.second);
        if (ra != rb) {
            if (ra < rb) parent[rb] = ra;
            else parent[ra] = rb;
        }
    }
    std::vector<int64_t> label(n, -1), comp(n);
    n_comp = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = find(i);
        if (label[r] < 0) label[r] = n_comp++;   // 根 = 最小成员，按 i 升序首次遇到即 np.unique 的次序
        comp[i] = label[r];
    }
    return comp;
}

}  // namespace

Islands stage3(const Config& cfg, uint64_t seed, const Planet& p, const Winds& w) {
    Rng rng = stage_rng(seed, 3);
    const double km_per_rad = p.radius_km;
    const double days_per_rad = km_per_rad / p.day_range_km;
    const GInfo& gi = w.g;

    // ---------------- 密度场（_density_grid）
    const Axes ax = grid_axes(cfg.get("shared.grid_res_deg"));
    const size_t M = ax.size();
    const int H = ax.nlat, W = ax.nlon;
    std::vector<double> base(M);
    {
        std::vector<double> knots = cfg.list("s03.islands.lat_density.lat");
        const std::vector<double>& vals = cfg.list("s03.islands.lat_density.density");
        for (double& x : knots) x = x * p.band_scale;
        if (knots.size() != vals.size()) throw std::invalid_argument("s03.islands.lat_density: lat and density lengths differ");
        for (size_t k = 1; k < knots.size(); ++k)
            if (knots[k] - knots[k - 1] <= 0) throw std::invalid_argument("s03.islands.lat_density: lat must increase strictly");
        for (double& x : knots) x = np_minimum(x, 90.0);
        std::vector<double> a(M);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) a[static_cast<size_t>(i) * W + j] = std::fabs(ax.lats[i]);
        base = np_interp(a, knots, vals);
    }
    const double lon_w = cfg.get("skeleton.d_lon_west"), lon_e = cfg.get("skeleton.d_lon_east");
    const std::vector<double>& dlr = cfg.list("skeleton.d_lat_range");
    const double d_lo = dlr[0] * p.band_scale, d_hi = dlr[1] * p.band_scale;
    const double lon_span = pymod(lon_e - lon_w, 360.0);
    const Vec3 g_xyz = latlon_to_xyz(gi.lat, gi.lon);
    std::vector<uint8_t> d_mask(M), skel(M);
    std::vector<double> dg(M);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            const double la = ax.lats[i], lo = ax.lons[j];
            d_mask[k] = (la >= d_lo && la <= d_hi) && (pymod(lo - lon_w, 360.0) <= lon_span);
            dg[k] = rad2deg(angdist(latlon_to_xyz(la, lo), g_xyz));
            skel[k] = d_mask[k] || (dg[k] < 3.0 * gi.radius_deg);
        }
    // 板块逻辑：② 的风（npz 存 float32）
    const std::vector<double> wu = f32v(w.u), wv = f32v(w.v);
    Tectonics tect = plate_fields(rng, ax, cfg, &wu, &wv);
    {
        std::vector<double> dom;
        for (size_t k = 0; k < M; ++k)
            if (!skel[k] && base[k] > 0) dom.push_back(tect.factor[k]);
        const double fm = std::max(1e-9, np_sum(dom.data(), dom.size()) / static_cast<double>(dom.size()));
        for (size_t k = 0; k < M; ++k) tect.factor[k] = skel[k] ? 1.0 : tect.factor[k] / fm;
    }
    const std::vector<double> noise = fractal_noise(rng, H, W, cfg.geti("s03.islands.noise_base_cells"), cfg.geti("s03.islands.noise_octaves"),
                                                    cfg.get("s03.islands.noise_persistence"));
    const double gamma = cfg.get("s03.islands.noise_gamma");
    const double core = cfg.get("skeleton.eq_core_halfwidth_deg");
    const double d_mult = cfg.get("skeleton.d_density_mult");
    const double arc_r = gi.radius_deg + cfg.get("skeleton.detour_arc_gap_deg");
    const double arc_hw = cfg.get("skeleton.detour_arc_halfwidth_deg"), arc_boost = cfg.get("skeleton.detour_arc_boost");
    std::vector<double> dens(M);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            double d = base[k] * tect.factor[k] * std::exp(gamma * noise[k]);
            if (std::fabs(ax.lats[i]) < core) d = 0.0;
            if (d_mask[k]) d *= d_mult;
            if (std::fabs(dg[k] - arc_r) < arc_hw && d_mask[k]) d *= arc_boost;
            if (dg[k] < gi.radius_deg) d = 0.0;
            dens[k] = d;
        }
    const LatLonGrid G = llg(ax);

    // ---------------- 撒点（_sample_islands）
    const int n_target = cfg.geti("s03.islands.n_islands");
    std::vector<double> lat, lon;
    {
        const double dmax = *std::max_element(dens.begin(), dens.end());
        int64_t n = 0;
        while (n < n_target) {
            const int64_t m = std::max<int64_t>(4096, 2 * (n_target - n));
            std::vector<double> z(m), lo(m), u(m);
            for (auto& x : z) x = rng.uniform(-1.0, 1.0);
            for (auto& x : lo) x = rng.uniform(-180.0, 180.0);
            for (auto& x : u) x = rng.uniform(0.0, 1.0);
            for (int64_t q = 0; q < m; ++q) {
                const double la = rad2deg(std::asin(z[q]));
                const double dd = grid_interp(dens, G, la, lo[q]);
                bool ok = u[q] < dd / dmax;
                ok = ok && std::fabs(la) >= core;
                ok = ok && rad2deg(angdist(latlon_to_xyz(la, lo[q]), g_xyz)) >= gi.radius_deg;
                if (ok) {
                    lat.push_back(la);
                    lon.push_back(lo[q]);
                    ++n;
                }
            }
        }
        lat.resize(n_target);
        lon.resize(n_target);
    }
    // 稳定编号：按 (2° 纬度桶, 经度) 排序（np.lexsort 稳定）
    {
        std::vector<int64_t> ord(n_target);
        std::iota(ord.begin(), ord.end(), 0);
        std::vector<double> key(n_target);
        for (int q = 0; q < n_target; ++q) key[q] = std::nearbyint(lat[q] / 2.0);
        std::stable_sort(ord.begin(), ord.end(), [&](int64_t a, int64_t b) {
            if (key[a] != key[b]) return key[a] < key[b];
            return lon[a] < lon[b];
        });
        std::vector<double> la2(n_target), lo2(n_target);
        for (int q = 0; q < n_target; ++q) {
            la2[q] = lat[ord[q]];
            lo2[q] = lon[ord[q]];
        }
        lat.swap(la2);
        lon.swap(lo2);
    }
    const size_t N = static_cast<size_t>(n_target);
    std::vector<Vec3> xyz(N);
    for (size_t q = 0; q < N; ++q) xyz[q] = latlon_to_xyz(lat[q], lon[q]);

    Islands I;
    I.ax = ax;
    I.lat = lat;
    I.lon = lon;
    I.xyz.resize(3 * N);
    for (size_t q = 0; q < N; ++q) {
        I.xyz[3 * q] = xyz[q].x;
        I.xyz[3 * q + 1] = xyz[q].y;
        I.xyz[3 * q + 2] = xyz[q].z;
    }
    I.density_at.resize(N);
    for (size_t q = 0; q < N; ++q) I.density_at[q] = grid_interp(dens, G, lat[q], lon[q]);

    // ---------------- 高度（台面口径）与板块的作用
    const std::vector<double> hfield = fractal_noise(rng, H, W, cfg.geti("s03.islands.height_noise_cells"), 3);
    const double hscale = cfg.get("s03.islands.height_scale_m");
    std::vector<double> height(N);
    for (size_t q = 0; q < N; ++q) height[q] = (0.5 + 0.5 * grid_interp(hfield, G, lat[q], lon[q])) * hscale;
    const double hj = cfg.get("s03.islands.height_jitter_m");
    for (size_t q = 0; q < N; ++q) height[q] = height[q] + rng.normal(0.0, hj);
    const double hcb = cfg.get("s03.plates.height_convergent_boost"), had = cfg.get("s03.plates.height_age_decay");
    std::vector<double> conv_at(N), age(N);
    for (size_t q = 0; q < N; ++q) {
        conv_at[q] = grid_interp(tect.conv_kernel, G, lat[q], lon[q]);
        age[q] = clip(grid_interp(tect.age, G, lat[q], lon[q]), 0.0, 1.0);
        height[q] = height[q] * (1.0 + hcb * conv_at[q]) * (1.0 - had * age[q]);
    }
    const double stack_thr = cfg.get("s03.islands.stack_zone_threshold"), stack_range = cfg.get("s03.islands.stack_range_m");
    I.in_stack.resize(N);
    for (size_t q = 0; q < N; ++q) {
        I.in_stack[q] = conv_at[q] > stack_thr;
        const double u = rng.uniform(0.0, 1.0);
        if (I.in_stack[q] && u < 0.5) height[q] = height[q] + stack_range;
        height[q] = np_maximum(height[q], 50.0);   // np.clip(height, 50, None) = np.maximum
    }
    I.plate.resize(N);
    for (size_t q = 0; q < N; ++q) {
        const int64_t ii = std::min<int64_t>(std::max<int64_t>(static_cast<int64_t>(std::nearbyint((lat[q] - ax.lats[0]) / (ax.lats[1] - ax.lats[0]))), 0), H - 1);
        int64_t jj = static_cast<int64_t>(std::nearbyint((lon[q] - ax.lons[0]) / (ax.lons[1] - ax.lons[0]))) % W;
        if (jj < 0) jj += W;
        I.plate[q] = tect.plate_id[static_cast<size_t>(ii) * W + jj];
    }

    // ---------------- kNN 与候选边
    const int k = cfg.geti("s03.islands.knn_k");
    const int n_far = cfg.geti("s03.islands.n_far");
    std::vector<int64_t> idx;
    std::vector<double> ang;
    knn(xyz, k + n_far, idx, ang);
    const int kk = static_cast<int>(std::min<int64_t>(k + n_far, static_cast<int64_t>(N) - 1));
    const double big = cfg.get("shared.ships.big_days");
    std::vector<double> dnn(ang.size());
    for (size_t q = 0; q < ang.size(); ++q) dnn[q] = ang[q] * days_per_rad;
    I.mean_nn.resize(N);
    for (size_t q = 0; q < N; ++q) {
        I.mean_nn[q] = np_sum(&dnn[q * kk], static_cast<size_t>(k)) / static_cast<double>(k);   // mean(axis=1)：行内成对求和
    }

    // ---------------- 陆地（_land）：势力范围 × 陆地占比，f0 二分反解
    {
        const double alpha = cfg.get("s03.islands.land_frac_alpha"), cap = cfg.get("s03.islands.land_frac_cap"),
                     sigma = cfg.get("s03.islands.land_frac_sigma"), target = cfg.get("shared.scale.total_land_km2");
        I.territory.resize(N);
        std::vector<double> rho(N), shape(N);
        for (size_t q = 0; q < N; ++q) {
            const double sp = I.mean_nn[q] * p.day_range_km;
            I.territory[q] = HEX_FACTOR * (sp * sp);
            rho[q] = np_maximum(I.density_at[q], 1e-9);
        }
        const double med = np_median(rho);
        for (size_t q = 0; q < N; ++q) shape[q] = np_pow(rho[q] / med, alpha);
        if (sigma > 0)
            for (size_t q = 0; q < N; ++q) shape[q] = shape[q] * std::exp(rng.normal(0.0, sigma));
        if (cap * np_sum(I.territory.data(), N) < target)
            throw std::invalid_argument("s03: land target exceeds land_frac_cap x sum(territory)");
        double lo = 1e-9, hi = 1e6;
        std::vector<double> tf(N);
        for (int it = 0; it < 200; ++it) {
            const double f0 = std::sqrt(lo * hi);
            for (size_t q = 0; q < N; ++q) tf[q] = I.territory[q] * np_minimum(cap, f0 * shape[q]);
            if (np_sum(tf.data(), N) < target) lo = f0;
            else hi = f0;
            if (hi / lo < 1.0 + 1e-10) break;
        }
        I.f0 = std::sqrt(lo * hi);
        I.land_frac.resize(N);
        I.area.resize(N);
        for (size_t q = 0; q < N; ++q) {
            I.land_frac[q] = np_minimum(cap, I.f0 * shape[q]);
            I.area[q] = I.territory[q] * I.land_frac[q];
        }
    }
    // ---------------- 可用地率（R9）
    {
        const std::vector<double>& ar = cfg.list("s03.islands.arable_frac_range");
        const double sig = cfg.get("s03.islands.arable_frac_sigma"), mean = cfg.get("shared.scale.arable_frac_mean");
        I.arable_frac.resize(N);
        for (size_t q = 0; q < N; ++q) I.arable_frac[q] = clip(mean * std::exp(rng.normal(-0.5 * sig * sig, sig)), ar[0], ar[1]);
    }

    // ---------------- 候选边：kNN ∪ 远程（≤ 大船航程）
    std::map<std::pair<int64_t, int64_t>, EdgeRec> edges;
    auto add_edge = [&](int64_t a, int64_t b, double d, int kind) {
        const auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        auto it = edges.find(key);
        if (it == edges.end()) {
            edges.emplace(key, EdgeRec{d, kind});
        } else if (d < it->second.d - 1e-12 || (std::fabs(d - it->second.d) <= 1e-12 && kind < it->second.kind)) {
            it->second = EdgeRec{d, kind};
        }
    };
    for (size_t q = 0; q < N; ++q)
        for (int m = 0; m < kk; ++m) {
            const double d = dnn[q * kk + m];
            if (d <= big) add_edge(static_cast<int64_t>(q), idx[q * kk + m], d, m < k ? KIND_KNN : KIND_FAR);
        }
    // G 邻域几何弦
    {
        const double reach = gi.radius_deg + cfg.get("skeleton.detour_arc_gap_deg") + cfg.get("skeleton.detour_arc_halfwidth_deg") + 1.0;
        std::vector<int64_t> near_g;
        for (size_t q = 0; q < N; ++q)
            if (rad2deg(angdist(xyz[q], g_xyz)) <= reach) near_g.push_back(static_cast<int64_t>(q));
        if (near_g.size() >= 2)
            for (size_t a = 0; a < near_g.size(); ++a)
                for (size_t b = a + 1; b < near_g.size(); ++b) {
                    const double d = angdist(xyz[near_g[a]], xyz[near_g[b]]) * days_per_rad;
                    if (d <= big) {
                        add_edge(near_g[a], near_g[b], d, KIND_FAR);
                        ++I.n_chord;
                    }
                }
    }
    // 跨赤道远征边
    {
        const double eq_top = p.eq_storm_top;
        std::vector<int64_t> ni, si;
        for (size_t q = 0; q < N; ++q) {
            if (lat[q] > 0 && lat[q] < eq_top + 2.0) ni.push_back(static_cast<int64_t>(q));
            if (lat[q] < 0 && lat[q] > -eq_top - 2.0) si.push_back(static_cast<int64_t>(q));
        }
        if (!ni.empty() && !si.empty()) {
            const size_t ns = si.size();
            std::vector<double> dd(ni.size() * ns);
            for (size_t a = 0; a < ni.size(); ++a)
                for (size_t b = 0; b < ns; ++b) dd[a * ns + b] = angdist(xyz[ni[a]], xyz[si[b]]) * days_per_rad;
            std::vector<size_t> flat(dd.size());
            std::iota(flat.begin(), flat.end(), 0);
            std::stable_sort(flat.begin(), flat.end(), [&](size_t x, size_t y) { return dd[x] < dd[y]; });
            std::vector<double> chosen;
            const double min_sep = cfg.get("s03.islands.expedition_min_lon_sep_deg");
            const int pairs = cfg.geti("s03.islands.expedition_pairs");
            for (size_t f : flat) {
                if (I.n_exp >= pairs) break;
                const int64_t a = ni[f / ns], b = si[f % ns];
                const double lon_mid = lon[a];
                bool ok = true;
                for (double c : chosen)
                    if (!(std::fabs(pymod(lon_mid - c + 180.0, 360.0) - 180.0) >= min_sep)) {
                        ok = false;
                        break;
                    }
                if (ok) {
                    add_edge(a, b, dd[f], KIND_EXPEDITION);
                    chosen.push_back(lon_mid);
                    ++I.n_exp;
                }
            }
        }
    }
    // 连通性回退（原则己）
    while (true) {
        int64_t n_comp = 0;
        const std::vector<int64_t> comp = weak_components(static_cast<int64_t>(N), edges, n_comp);
        if (n_comp == 1) break;
        std::vector<int64_t> sizes(n_comp, 0);
        for (int64_t c : comp) ++sizes[c];
        const int64_t c_small = std::min_element(sizes.begin(), sizes.end()) - sizes.begin();
        std::vector<int64_t> inside, outside;
        for (size_t q = 0; q < N; ++q) (comp[q] == c_small ? inside : outside).push_back(static_cast<int64_t>(q));
        double best = INF;
        int64_t ba = -1, bb = -1;
        for (int64_t a : inside)
            for (int64_t b : outside) {
                const double d = angdist(xyz[a], xyz[b]);
                if (d < best) {
                    best = d;
                    ba = a;
                    bb = b;
                }
            }
        add_edge(ba, bb, best * days_per_rad, KIND_FALLBACK);
        ++I.n_fallback;
    }
    for (const auto& kv : edges) {
        I.src.push_back(kv.first.first);
        I.dst.push_back(kv.first.second);
        I.dist_days.push_back(kv.second.d);
        I.kind.push_back(static_cast<int8_t>(kv.second.kind));
    }

    // ---------------- 分类与叠层
    const double bridge = cfg.get("shared.ships.bridge_days"), small = cfg.get("shared.ships.small_days");
    I.cls.resize(N);
    I.layered.resize(N);
    const double lay_thr = cfg.get("s03.islands.layered_height_std_m");
    std::vector<double> row(k + 1);
    for (size_t q = 0; q < N; ++q) {
        const double mn = I.mean_nn[q];
        I.cls[q] = static_cast<int8_t>(mn < bridge ? 0 : mn < small ? 1 : mn < big ? 2 : 3);
        for (int m = 0; m < k; ++m) row[m] = height[idx[q * kk + m]];
        row[k] = height[q];
        const double mean = np_sum(row.data(), row.size()) / static_cast<double>(k + 1);   // np.std(axis=1)
        for (double& x : row) {
            const double d = x - mean;
            x = d * d;
        }
        I.layered[q] = std::sqrt(np_sum(row.data(), row.size()) / static_cast<double>(k + 1)) > lay_thr;
    }

    // ---------------- 主岛与岛体
    {
        const double sig = cfg.get("s03.islands.main_frac_sigma"), mean = cfg.get("s03.islands.main_frac_mean");
        const std::vector<double>& mr = cfg.list("s03.islands.main_frac_range");
        const double keel = cfg.get("s03.islands.keel_clearance_m");
        I.main_frac.resize(N);
        I.main_area.resize(N);
        I.wall.resize(N);
        for (size_t q = 0; q < N; ++q) {
            I.main_frac[q] = clip(mean * std::exp(rng.normal(-0.5 * sig * sig, sig)), mr[0], mr[1]);
            I.main_area[q] = I.area[q] * I.main_frac[q];
            I.wall[q] = np_maximum(0.0, height[q] - keel);
        }
    }
    I.height = height;
    I.age = age;
    I.plate_id = tect.plate_id;
    I.btype = tect.btype;
    I.boundary_kernel = tect.boundary_kernel;
    I.conv_kernel = tect.conv_kernel;
    I.plate_age = tect.age;
    I.factor = tect.factor;
    I.seeds_xyz = tect.seeds_xyz;
    I.density = dens;
    return I;
}

}  // namespace skyisle::planet
