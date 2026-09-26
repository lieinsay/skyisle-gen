// 5.2 岛内地形（terrain.py 同式）。
#include "skyisle/island/terrain.hpp"

#include <algorithm>
#include <cmath>

#include "skyisle/flow.hpp"

namespace skyisle::island {

AgeKind age_class(double age, const Config& c) {
    const double y = c.get("terrain.age_young"), o = c.get("terrain.age_old");
    return age < y ? YOUNG : (age < o ? MID : OLD);
}

// ---------------------------------------------------------------- 岛形
Shape island_shape(Rng& rng, double area_km2, double res_km, double elong, double theta, const Config& c) {
    const double R = std::sqrt(area_km2 / PI);
    const double a = R * std::sqrt(elong), b = R / std::sqrt(elong);
    const double half = c.get("terrain.shape_pad") * a;
    int n = static_cast<int>(std::ceil(2 * half / res_km)) + 1;
    n = std::max(n, 5);
    Shape s;
    s.xs.resize(n);
    for (int k = 0; k < n; ++k) s.xs[k] = (k - (n - 1) / 2.0) * res_km;
    FractalNoise warp(rng, -half, -half, half, half, std::max(0.8 * R, 3 * res_km), 2, 0.5);
    FractalNoise warp2(rng, -half, -half, half, half, std::max(0.8 * R, 3 * res_km), 2, 0.5);
    const double wamp = c.get("terrain.warp_amp") * R;
    const double ct = std::cos(theta), st = std::sin(theta);
    FractalNoise noise(rng, -half, -half, half, half, std::max(0.5 * R, 3 * res_km), c.geti("terrain.shape_octaves"), 0.55);
    const double namp = c.get("terrain.shape_noise_amp");
    GridD f(n, n);
    double fmin = INF, fmax = -INF;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double X = s.xs[j], Y = -s.xs[i];
            const double Xw = X + wamp * warp.sample(X, Y);
            const double Yw = Y + wamp * warp2.sample(X, Y);
            const double u = (Xw * ct + Yw * st) / a;
            const double v = (-Xw * st + Yw * ct) / b;
            const double dome = 1.0 - (u * u + v * v);
            const double val = dome + namp * noise.sample(X, Y);
            f(i, j) = val;
            fmin = std::min(fmin, val);
            fmax = std::max(fmax, val);
        }
    const double target = area_km2 / (res_km * res_km);
    double lo = fmin, hi = fmax, best_err = 1e18, best_tau = 0;
    bool have = false;
    Mask thr(n, n, 0);
    for (int it = 0; it < 48; ++it) {
        const double tau = 0.5 * (lo + hi);
        for (size_t k = 0; k < f.size(); ++k) thr.v[k] = f.v[k] > tau ? 1 : 0;
        int cnt = 0;
        largest_component(thr, &cnt);
        const double err = std::fabs(cnt - target);
        if (err < best_err) {
            best_err = err;
            best_tau = tau;
            have = true;
        }
        if (cnt > target) lo = tau;
        else hi = tau;
        if (err <= std::max(0.5, 0.002 * target)) break;
    }
    Mask mask(n, n, 0);
    if (have) {
        for (size_t k = 0; k < f.size(); ++k) thr.v[k] = f.v[k] > best_tau ? 1 : 0;
        mask = largest_component(thr);
    }
    bool any = false;
    for (uint8_t x : mask.v) any = any || x;
    if (!any) {
        // 粗分辨率下比半格还小的礁：留最高的一格
        size_t am = 0;
        for (size_t k = 1; k < f.size(); ++k)
            if (f.v[k] > f.v[am]) am = k;
        mask.v[am] = 1;
    }
    double tmin = INF, tmax = -INF;
    for (size_t k = 0; k < f.size(); ++k)
        if (mask.v[k]) {
            tmin = std::min(tmin, f.v[k]);
            tmax = std::max(tmax, f.v[k]);
        }
    s.inside = GridD(n, n, 0.0);
    const double den = std::max(1e-9, tmax - tmin);
    for (size_t k = 0; k < f.size(); ++k)
        if (mask.v[k]) s.inside.v[k] = clip((f.v[k] - tmin) / den, 0.0, 1.0);
    s.mask = std::move(mask);
    return s;
}

// ---------------------------------------------------------------- 基形
namespace {

struct Gullies {
    std::vector<double> lat;
    double cx, cy, amp;
    int n;
    double at(double X, double Y) const {
        const double th = std::atan2(Y - cy, X - cx);
        const double f = (th + PI) / (2 * PI) * n;
        const double fl = std::floor(f);
        const int64_t i0 = ((static_cast<int64_t>(fl) % n) + n) % n;
        double t = f - fl;
        t = t * t * (3 - 2 * t);
        const double g = lat[i0] * (1 - t) + lat[(i0 + 1) % n] * t;
        return 1.0 - amp * g;
    }
};

Gullies radial_gullies(Rng& rng, double cx, double cy, int n_lobes, double amp) {
    Gullies g;
    g.lat.resize(n_lobes);
    rng.uniform_fill(0.0, 1.0, g.lat.data(), n_lobes);
    g.cx = cx;
    g.cy = cy;
    g.amp = amp;
    g.n = n_lobes;
    return g;
}

}  // namespace

GridD base_form(Rng& rng, const Shape& s, double age, double area_km2, double res_km, const Config& c, AgeKind& kind) {
    const double R = std::sqrt(area_km2 / PI);
    const int n = s.mask.H;
    double half = 0;
    for (double x : s.xs) half = std::max(half, std::fabs(x));
    kind = age_class(age, c);
    std::vector<int32_t> cells;
    for (int k = 0; k < n * n; ++k)
        if (s.mask.v[k]) cells.push_back(k);
    const int64_t k_top = std::max<int64_t>(1, static_cast<int64_t>(0.05 * static_cast<double>(cells.size())));
    const int64_t pick = rng.integers(0, k_top);
    // 向内程度降序（稳定）排第 pick 的格
    std::vector<int32_t> ord(cells.size());
    for (size_t q = 0; q < ord.size(); ++q) ord[q] = static_cast<int32_t>(q);
    std::nth_element(ord.begin(), ord.begin() + pick, ord.end(), [&](int32_t a, int32_t b) {
        const double va = s.inside.v[cells[a]], vb = s.inside.v[cells[b]];
        return va > vb || (va == vb && a < b);
    });
    const int32_t pc = cells[ord[pick]];
    const double py = s.Y(pc / n, 0), px = s.X(0, pc % n);
    FractalNoise fn_ridge(rng, -half, -half, half, half, std::max(0.45 * R, 4 * res_km), 4, 0.5);
    const int n_oct = static_cast<int>(clip(std::nearbyint(std::log2(std::max(0.15 * R, 3 * res_km) / (2.5 * res_km))) + 1, 2, 7));
    FractalNoise fn_fine(rng, -half, -half, half, half, std::max(0.15 * R, 3 * res_km), n_oct, 0.55);
    GridD shape(n, n, 0.0);
    if (kind == YOUNG) {
        int n_cones = 1;
        if (area_km2 > 25.0 && rng.uniform(0.0, 1.0) < 0.4) n_cones = 2;
        const double Rc = c.get("terrain.cone_radius_rel") * R;
        const double cpow = c.get("terrain.cone_profile_pow"), gamp = c.get("terrain.gully_amp");
        GridD sc(n, n, 0.0);
        for (int k = 0; k < n_cones; ++k) {
            double cx = px, cy = py;
            if (k > 0) {
                const double ang = rng.uniform(-PI, PI);
                cx = px + 0.45 * R * std::cos(ang);
                cy = py + 0.45 * R * std::sin(ang);
            }
            const int nl = static_cast<int>(rng.integers(9, 16));
            const Gullies gul = radial_gullies(rng, cx, cy, nl, gamp);
            const double mult = k == 0 ? 1.0 : rng.uniform(0.55, 0.9);
            for (int32_t q : cells) {
                const double X = s.X(0, q % n), Y = s.Y(q / n, 0);
                const double d = np_hypot(X - cx, Y - cy) / Rc;
                double cone = np_pow(clip(1.0 - d, 0.0, 1.0), cpow);
                cone *= c_pow(gul.at(X, Y), clip(d, 0.0, 1.0));
                sc.v[q] = std::max(sc.v[q], cone * mult);
            }
        }
        for (int32_t q : cells) {
            const double X = s.X(0, q % n), Y = s.Y(q / n, 0);
            shape.v[q] = sc.v[q] * (0.7 + 0.3 * std::sqrt(s.inside.v[q])) + 0.06 * fn_fine.sample(X, Y);
        }
    } else if (kind == MID) {
        const double ang = rng.uniform(-PI, PI);
        const double sa = std::sin(ang), ca = std::cos(ang);
        const int nb = static_cast<int>(rng.integers(1, 4));
        std::vector<double> bx(nb), by(nb), bm(nb);
        for (int t = 0; t < nb; ++t) {
            const int64_t pk = rng.integers(0, static_cast<int64_t>(cells.size()));
            bx[t] = s.X(0, cells[pk] % n);
            by[t] = s.Y(cells[pk] / n, 0);
            bm[t] = rng.uniform(0.4, 0.8);
        }
        const double qd = c_pow(0.25 * R, 2);
        for (int32_t q : cells) {
            const double X = s.X(0, q % n), Y = s.Y(q / n, 0);
            const double ridged = 1.0 - std::fabs(fn_ridge.sample(X, Y));
            const double dperp = std::fabs(-(X - px) * sa + (Y - py) * ca);
            const double rq = dperp / (0.35 * R);
            const double ridge = std::exp(-(rq * rq));
            const double dist = np_hypot(X - px, Y - py) / (1.4 * R);
            const double core = clip(1.0 - dist, 0.0, 1.0);
            double bumps = 0.0;
            for (int t = 0; t < nb; ++t) {
                const double dx = X - bx[t], dy = Y - by[t];
                bumps = std::max(bumps, bm[t] * std::exp(-(dx * dx + dy * dy) / qd));
            }
            shape.v[q] = np_pow(s.inside.v[q], 0.35) * (0.2 + 0.45 * ridged + 0.35 * std::max(ridge, bumps)) * (0.55 + 0.45 * core) +
                         0.05 * fn_fine.sample(X, Y);
        }
    } else {
        const double rise = c.get("terrain.plateau_rise"), vdepth = c.get("terrain.plateau_valley_depth"), mamp = c.get("terrain.mound_amp");
        for (int32_t q : cells) {
            const double X = s.X(0, q % n), Y = s.Y(q / n, 0);
            const double t = clip(s.inside.v[q] / rise, 0.0, 1.0);
            const double plateau = t * t * (3 - 2 * t);
            const double valley = clip(fn_ridge.sample(X, Y), 0.0, 1.0);
            const double dist = np_hypot(X - px, Y - py) / (1.6 * R);
            const double tilt = clip(1.0 - dist, 0.0, 1.0);
            const double fine = fn_fine.sample(X, Y);
            const double mounds = clip(fine, 0.0, 1.0) * (1.0 - plateau);
            shape.v[q] = std::sqrt(plateau) * (0.75 + 0.25 * tilt) * (1.0 - vdepth * valley * s.inside.v[q]) + mamp * mounds + 0.03 * fine;
        }
    }
    return shape;
}

// ---------------------------------------------------------------- 侵蚀
GridD erode(Rng* rng, GridD h, const Mask& mask, double res_m, int rounds, double base_level, const Config& c,
            const GridD* uplift, const GridD* jitter) {
    const int H = h.H, W = h.W;
    const size_t N = h.size();
    {
        GridD pf = priority_fill(h, mask, 1e-3);
        for (size_t k = 0; k < N; ++k)
            if (mask.v[k]) h.v[k] = pf.v[k];
    }
    const double K = c.get("terrain.carve_k"), m_exp = c.get("terrain.carve_m");
    const double talus = std::tan(c.get("terrain.talus_deg") * (PI / 180.0));
    const double kd = c.get("terrain.diffusion_k");
    const double cell_km2 = c_pow(res_m / 1000.0, 2);
    const double a0 = c.get("terrain.carve_a0_km2"), pit_keep = c.get("terrain.pit_keep_m"), route_p = c.get("terrain.carve_route_p");
    const int fill_iters = c.geti("terrain.fill_iters");
    GridD zero(H, W, 0.0);
    const GridD& jit = jitter ? *jitter : zero;
    GridD hj(H, W);
    std::vector<int64_t> recv(N);
    std::vector<double> F(N);
    for (int rd = 0; rd < rounds; ++rd) {
        if (uplift)
            for (size_t k = 0; k < N; ++k)
                if (mask.v[k]) h.v[k] = h.v[k] + uplift->v[k];
        for (size_t k = 0; k < N; ++k) hj.v[k] = h.v[k] + jit.v[k];
        const GridD hf = fill_iter(hj, mask, fill_iters);
        for (size_t k = 0; k < N; ++k)
            if (mask.v[k]) h.v[k] = std::max(h.v[k], hf.v[k] - jit.v[k] - pit_keep);
        FlowDir fd = (rng && route_p > 0) ? d8_random(hf, mask, res_m, *rng, route_p) : d8(hf, mask, res_m);
        GridD A = accumulate(mask, fd);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                const size_t k = static_cast<size_t>(i) * W + j;
                const double Ak = A.v[k] * cell_km2;
                const int ri = fd.ri[k], rj = fd.rj[k];
                const bool diag = ri >= 0 && ri != i && rj != j;
                const double dist_km = (diag ? SQRT2 : 1.0) * res_m / 1000.0;
                F[k] = Ak >= a0 ? K * np_pow(std::max(Ak, cell_km2), m_exp) / dist_km : 0.0;
                recv[k] = ri >= 0 ? static_cast<int64_t>(ri) * W + rj : ((mask.v[k] && fd.to_void[k]) ? -2 : -1);
            }
        // 隐式下切：先下游后上游（每格只依赖其下游的终值，与按路由面升序同）
        std::vector<int64_t> rpos(N);
        for (size_t k = 0; k < N; ++k) rpos[k] = recv[k] >= 0 ? recv[k] : -1;
        const std::vector<int32_t> order = downstream_first(rpos, mask);
        for (int32_t k : order) {
            const int64_t r = recv[k];
            if (r == -1) continue;
            const double hr = r == -2 ? base_level : h.v[r];
            const double hk = h.v[k];
            if (hk > hr) h.v[k] = (hk + F[k] * hr) / (1.0 + F[k]);
        }
        // 热力坍塌：坡度超过休止角，把超出的部分推给邻居（hh 取坍塌前）
        const GridD hh = h;
        GridD mv(H, W, 0.0);
        for (int n8 = 0; n8 < 8; ++n8) {
            const int di = N8[n8][0], dj = N8[n8][1];
            const double dist = res_m * ((di && dj) ? SQRT2 : 1.0);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    const size_t k = static_cast<size_t>(i) * W + j;
                    const int a = i - di, b = j - dj;
                    double m = 0.0;
                    if (mask.v[k] && a >= 0 && b >= 0 && a < H && b < W && mask(a, b)) {
                        const double ex = (hh.v[k] - hh(a, b)) - talus * dist;
                        m = (ex > 0.0 ? ex : 0.0) * 0.25;
                    }
                    mv.v[k] = m;
                }
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    const size_t k = static_cast<size_t>(i) * W + j;
                    h.v[k] = h.v[k] - mv.v[k];
                    const int a = i + di, b = j + dj;
                    h.v[k] = h.v[k] + ((a >= 0 && b >= 0 && a < H && b < W) ? mv(a, b) : 0.0);
                }
        }
        // 坡面扩散
        const GridD lap = laplacian(h, mask);
        for (size_t k = 0; k < N; ++k) h.v[k] = h.v[k] + kd * lap.v[k];
        for (size_t k = 0; k < N; ++k)
            if (mask.v[k]) h.v[k] = std::max(h.v[k], base_level);
    }
    return h;
}

void fit_rim(double surface, double relief, double median_frac, double rim_min, double& rim, double& R) {
    if (median_frac <= 1e-3) {
        rim = std::max(surface, rim_min);
        R = relief;
        return;
    }
    rim = surface - median_frac * relief;
    R = relief;
    if (rim < rim_min) {
        rim = rim_min;
        R = std::max(1.0, (surface - rim_min) / median_frac);
    }
}

Sculpt sculpt_island(Rng& rng, const Shape& s, double age, double area_km2, double res_km, double surface, double relief,
                     double rim_min, bool is_main, const Config& c) {
    Sculpt out;
    const int n = s.mask.H;
    const size_t N = s.mask.size();
    GridD shape = base_form(rng, s, age, area_km2, res_km, c, out.kind);
    double lo = INF, hi = -INF;
    std::vector<double> sm;
    for (size_t k = 0; k < N; ++k)
        if (s.mask.v[k]) {
            lo = std::min(lo, shape.v[k]);
            hi = std::max(hi, shape.v[k]);
        }
    const double den = std::max(1e-9, hi - lo);
    for (size_t k = 0; k < N; ++k) shape.v[k] = s.mask.v[k] ? (shape.v[k] - lo) / den : 0.0;
    const char* kn = out.kind == YOUNG ? "terrain.median_frac_young" : (out.kind == MID ? "terrain.median_frac_mid" : "terrain.median_frac_old");
    double m_t = c.get(kn);
    m_t = std::min(m_t, std::max(c.get("terrain.median_frac_min"), (surface - rim_min) / std::max(1.0, relief)));
    for (size_t k = 0; k < N; ++k)
        if (s.mask.v[k]) sm.push_back(shape.v[k]);
    const double m_raw = np_median(sm);
    if (1e-3 < m_raw && m_raw < 0.999) {
        const double gamma = clip(std::log(m_t) / std::log(m_raw), 0.7, 3.5);
        for (size_t k = 0; k < N; ++k) shape.v[k] = np_pow(shape.v[k], gamma);
    }
    double rim, R;
    fit_rim(surface, relief, m_t, rim_min, rim, R);
    GridD h(n, n);
    for (size_t k = 0; k < N; ++k) h.v[k] = rim + R * shape.v[k];
    int rounds = c.geti(is_main ? "terrain.carve_rounds_main" : "terrain.carve_rounds_small");
    const double fac = out.kind == YOUNG ? 0.7 : (out.kind == MID ? 1.0 : 1.3);
    rounds = static_cast<int>(std::nearbyint(rounds * fac));
    const double res_m = res_km * 1000.0;
    const double up = c.get("terrain.uplift_rel") * R;
    int64_t msum = 0;
    for (uint8_t x : s.mask.v) msum += x;
    if (rounds > 0 && msum >= 30) {
        const int f = std::max(1, static_cast<int>(std::ceil(static_cast<double>(n) / c.get("terrain.erosion_max_cells"))));
        double half = 0;
        for (double x : s.xs) half = std::max(half, std::fabs(x));
        const double jr = c.get("terrain.carve_jitter_rel") * R;
        FractalNoise jn(rng, -half, -half, half, half, 3.0 * res_km * f, 2, 0.5);
        GridD jit(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) jit(i, j) = jr * jn.sample(s.xs[j], -s.xs[i]);
        if (f > 1) {
            GridD hin(n, n);
            for (size_t k = 0; k < N; ++k) hin.v[k] = s.mask.v[k] ? h.v[k] : rim;
            const GridD hc = block_mean(hin, f);
            const Mask mc = block_any(s.mask, f);
            GridD upc = block_mean(shape, f);
            for (double& v : upc.v) v = up * v;
            const GridD jc = block_mean(jit, f);
            const GridD ec = erode(&rng, hc, mc, res_m * f, rounds, rim, c, &upc, &jc);
            GridD d(hc.H, hc.W, 0.0);
            for (size_t k = 0; k < d.size(); ++k) d.v[k] = mc.v[k] ? ec.v[k] - hc.v[k] : 0.0;
            const GridD delta = smooth121(d, mc, c.geti("terrain.carve_smooth_coarse"));
            const GridD up_d = upsample_bilinear(delta, f, n, n);
            const GridD sm2 = smooth121(up_d, s.mask, c.geti("terrain.carve_smooth_fine"));
            for (size_t k = 0; k < N; ++k) h.v[k] = h.v[k] + sm2.v[k];
        } else {
            GridD upf(n, n);
            for (size_t k = 0; k < N; ++k) upf.v[k] = up * shape.v[k];
            h = erode(&rng, h, s.mask, res_m, rounds, rim, c, &upf, &jit);
        }
    }
    double hmax = -INF;
    for (size_t k = 0; k < N; ++k)
        if (s.mask.v[k]) {
            h.v[k] = std::max(h.v[k], rim);
            hmax = std::max(hmax, h.v[k]);
        }
    const double den2 = std::max(1e-6, hmax - rim);
    std::vector<double> um;
    GridD u(n, n, 0.0);
    for (size_t k = 0; k < N; ++k)
        if (s.mask.v[k]) {
            u.v[k] = clip((h.v[k] - rim) / den2, 0.0, 1.0);
            um.push_back(u.v[k]);
        }
    double rim2, R2;
    fit_rim(surface, relief, np_median(um), rim_min, rim2, R2);
    out.h = GridD(n, n, NaN);
    for (size_t k = 0; k < N; ++k)
        if (s.mask.v[k]) out.h.v[k] = rim2 + R2 * u.v[k];
    out.rim = rim2;
    out.peak = rim2 + R2;
    return out;
}

}  // namespace skyisle::island
