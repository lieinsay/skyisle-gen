// ④ 局地风与气候（s04_climate.py）：岛对风的扰动（localwind.py）、温度、风暴、上风水汽追踪降水（moisture.py）、季节强度（skeleton.season_range）、
// 各群的气候标量与河流。无随机数（precip_noise_amp > 0 时才从 stage_rng(seed, 4) 取残余噪声）。
#include "skyisle/planet/planet.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace skyisle::planet {

const char* const EDGE_KEYS[8] = {"eq_n", "trades_n", "calm_n", "west_n", "eq_s", "trades_s", "calm_s", "west_s"};

namespace {

constexpr double R_EARTH_M = 6.371e6;
constexpr double SOLAR_CONST = 1361.0;

inline double gauss(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return std::exp(-0.5 * (z * z));
}

// numpy 2.x 复数的 np.abs（loops_unary_complex 的 SIMD 版，连续数组连尾巴都走它）：不是 C 的 hypot，
// 而是 larger · sqrt(fma(r, r, 1))，r = smaller / larger（larger = 0 或 smaller = inf 时 r = 0）
double np_cabs(double re, double im) {
    re = std::fabs(re);
    im = std::fabs(im);
    const double inf = INF;
    if (re == inf) im = inf;
    if (im == inf) re = inf;
    if (std::isnan(re)) im = NaN;
    if (std::isnan(im)) re = NaN;
    const double larger = re > im ? re : im;     // _mm_max_pd(re, im)
    const double smaller = im < re ? im : re;    // _mm_min_pd(im, re)
    const double ratio = (larger == 0.0 || smaller == inf) ? 0.0 : smaller / larger;
    return std::sqrt(std::fma(ratio, ratio, 1.0)) * larger;
}

std::vector<double> edge_values(const Planet& p) {
    return {p.eq_storm_top, p.trades_top, p.calm_top, p.westerlies_top, -p.eq_storm_top, -p.trades_top, -p.calm_top, -p.westerlies_top};
}

// localwind.gauss_smooth：可分离高斯平滑，经度周期、纬度反射（np.pad mode="reflect"）
std::vector<double> gauss_smooth(const std::vector<double>& field, int H, int W, double res_deg, double sigma_deg) {
    if (sigma_deg <= 0) return field;
    const int n = static_cast<int>(std::ceil(3.0 * sigma_deg / res_deg));
    std::vector<double> k(2 * n + 1);
    for (int i = 0; i < 2 * n + 1; ++i) {
        const double z = static_cast<double>(i - n) * res_deg / sigma_deg;
        k[i] = std::exp(-0.5 * (z * z));
    }
    const double ks = np_sum(k.data(), k.size());
    for (double& x : k) x /= ks;
    std::vector<double> out(field.size(), 0.0);
    for (int t = 0; t < 2 * n + 1; ++t) {
        const int shift = t - n;   // np.roll(field, shift, axis=1)[:, j] = field[:, j − shift]
        for (int i = 0; i < H; ++i) {
            const double* f = &field[static_cast<size_t>(i) * W];
            double* o = &out[static_cast<size_t>(i) * W];
            for (int j = 0; j < W; ++j) o[j] += k[t] * f[((j - shift) % W + W) % W];
        }
    }
    auto refl = [H](int x) { return x < 0 ? -x : (x > H - 1 ? 2 * (H - 1) - x : x); };
    std::vector<double> out2(field.size(), 0.0);
    for (int t = 0; t < 2 * n + 1; ++t)
        for (int i = 0; i < H; ++i) {
            const double* s = &out[static_cast<size_t>(refl(i + t - n)) * W];
            double* o = &out2[static_cast<size_t>(i) * W];
            for (int j = 0; j < W; ++j) o[j] += k[t] * s[j];
        }
    return out2;
}

// localwind.obstacle_fields：岛群 → 障碍场 O 与陆地覆盖场 L
void obstacle_fields(const Islands& isl, const Axes& ax, const Config& cfg, std::vector<double>& O, std::vector<double>& L) {
    const double res = ax.lats[1] - ax.lats[0];
    const int H = ax.nlat, W = ax.nlon;
    O.assign(ax.size(), 0.0);
    L.assign(ax.size(), 0.0);
    const double blm = cfg.get("s04.localwind.boundary_layer_m");
    const double cell0 = c_pow(111.19 * res, 2);   // Python 浮点的 ** = C 的 pow
    for (size_t q = 0; q < isl.n(); ++q) {
        const int64_t i = std::min<int64_t>(std::max<int64_t>(static_cast<int64_t>(std::nearbyint((isl.lat[q] - ax.lats[0]) / res)), 0), H - 1);
        int64_t j = static_cast<int64_t>(std::nearbyint((isl.lon[q] - ax.lons[0]) / res)) % W;
        if (j < 0) j += W;
        const double cell_km2 = cell0 * np_maximum(std::cos(deg2rad(ax.lats[i])), 0.05);
        const double cover = f32(isl.territory[q]) / cell_km2;
        const double wall_frac = np_minimum(1.0, f32(isl.wall[q]) / blm);
        const double lf = f32(isl.land_frac[q]);
        O[static_cast<size_t>(i) * W + j] += lf * wall_frac * cover;   // np.add.at：按下标次序累加
        L[static_cast<size_t>(i) * W + j] += lf * cover;
    }
    const double gain = cfg.get("s04.localwind.obstacle_gain", 1.0);
    const double sm = cfg.get("s04.localwind.smooth_deg");
    O = gauss_smooth(O, H, W, res, sm);
    L = gauss_smooth(L, H, W, res, sm);
    for (double& x : O) x = clip(gain * x, 0.0, 1.0);
    for (double& x : L) x = clip(gain * x, 0.0, 1.0);
}

// localwind.band_displacement：每条带界的位移 Δφ(lon)（行序 EDGE_KEYS）与全场位移 D(lat, lon)
void band_displacement(const std::vector<double>& O, const Axes& ax, const Planet& p, const Config& cfg, std::vector<double>& dphi,
                       std::vector<double>& D) {
    const int H = ax.nlat, W = ax.nlon;
    const std::vector<double> e = edge_values(p);
    const double win = cfg.get("s04.localwind.shift_window_deg");
    const int kmax = cfg.geti("s04.localwind.shift_wavenumber_max");
    const double gain = cfg.get("s04.localwind.shift_gain_deg"), mx = cfg.get("s04.localwind.shift_max_deg");
    dphi.assign(static_cast<size_t>(8) * W, 0.0);
    for (int key = 0; key < 8; ++key) {
        std::vector<double> a(W, 0.0);
        int cnt = 0;
        for (int i = 0; i < H; ++i)
            if (std::fabs(ax.lats[i] - e[key]) <= win) {
                for (int j = 0; j < W; ++j) a[j] += O[static_cast<size_t>(i) * W + j];   // O[rows].mean(axis=0)：逐行顺序加
                ++cnt;
            }
        if (cnt)
            for (double& x : a) x /= static_cast<double>(cnt);
        const double am = np_sum(a.data(), a.size()) / static_cast<double>(W);
        for (double& x : a) x = x - am;
        const std::vector<double> lp = rfft_lowpass(a, kmax);
        const double sd = np_std(lp);
        const double sgn = e[key] > 0 ? 1.0 : -1.0;
        for (int j = 0; j < W; ++j) dphi[static_cast<size_t>(key) * W + j] = (sd > 1e-12 ? clip(gain * lp[j] / sd, -mx, mx) : 0.0) * sgn;
    }
    // 结点放在位移后的带界上（南 → 北），值 = 位移，两极为 0
    const int order[8] = {7, 6, 5, 4, 0, 1, 2, 3};   // west_s calm_s trades_s eq_s eq_n trades_n calm_n west_n
    D.assign(ax.size(), 0.0);
    std::vector<double> kl(10), kv(10);
    for (int j = 0; j < W; ++j) {
        kl[0] = -90.0;
        kv[0] = 0.0;
        for (int m = 0; m < 8; ++m) {
            const double d = dphi[static_cast<size_t>(order[m]) * W + j];
            kl[m + 1] = e[order[m]] + d;
            kv[m + 1] = d;
        }
        kl[9] = 90.0;
        kv[9] = 0.0;
        const std::vector<double> col = np_interp(ax.lats, kl, kv);
        for (int i = 0; i < H; ++i) D[static_cast<size_t>(i) * W + j] = col[i];
    }
}

// localwind.wake_field：沿本地风向向上游回溯 K 步采样障碍，几何衰减累加
std::vector<double> wake_field(const std::vector<double>& O, const std::vector<double>& u, const std::vector<double>& v, const Axes& ax,
                               const Config& cfg) {
    const int H = ax.nlat, W = ax.nlon;
    const LatLonGrid G = llg(ax);
    const double step = cfg.get("s04.localwind.wake_step_deg"), lam = cfg.get("s04.localwind.wake_decay");
    const int steps = cfg.geti("s04.localwind.wake_steps");
    std::vector<double> Wk(O.size(), 0.0);
    for (int i = 0; i < H; ++i) {
        const double cosl = np_maximum(std::cos(deg2rad(ax.lats[i])), 0.1);
        for (int j = 0; j < W; ++j) {
            const size_t c = static_cast<size_t>(i) * W + j;
            const double spd = np_hypot(u[c], v[c]);
            const bool ok = spd > 0.5;
            const double uh = ok ? u[c] / np_maximum(spd, 1e-9) : 0.0;
            const double vh = ok ? v[c] / np_maximum(spd, 1e-9) : 0.0;
            double w = 0.0;
            for (int k = 1; k <= steps; ++k) {
                const double ks = static_cast<double>(k) * step;
                const double latq = clip(ax.lats[i] - ks * vh, ax.lats[0], ax.lats[H - 1]);
                const double lonq = ax.lons[j] - ks * uh / cosl;
                w += c_pow(lam, static_cast<double>(k)) * grid_interp(O, G, latq, lonq);
            }
            Wk[c] = clip(w, 0.0, 1.0);
        }
    }
    return Wk;
}

// moisture.solve：二维稳态水汽收支，通量形式迎风差分，显式推进；极区纬向平均
void moisture_solve(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& E, const std::vector<double>& eps,
                    const Axes& ax, const Config& cfg, std::vector<double>& q, std::vector<double>& P, double& dt_out, int64_t& n_steps) {
    const int H = ax.nlat, W = ax.nlon;
    const double tau = cfg.get("s04.climate.moisture_tau_days") * 86400.0;
    const double polar = cfg.get("s04.climate.moisture_polar_filter_lat");
    const double days = cfg.get("s04.climate.moisture_days");
    const double dlam = deg2rad(ax.lons[1] - ax.lons[0]);
    const double dphi = deg2rad(ax.lats[1] - ax.lats[0]);
    std::vector<double> phi(H), cosphi(H);
    for (int i = 0; i < H; ++i) {
        phi[i] = deg2rad(ax.lats[i]);
        cosphi[i] = np_maximum(std::cos(phi[i]), 0.02);
    }
    double umax = -INF, vmax = -INF;
    for (int i = 0; i < H; ++i) {
        if (std::fabs(ax.lats[i]) <= polar) {
            const double den = R_EARTH_M * std::cos(deg2rad(ax.lats[i])) * dlam;
            for (int j = 0; j < W; ++j) umax = std::max(umax, std::fabs(u[static_cast<size_t>(i) * W + j]) / den);
        }
        for (int j = 0; j < W; ++j) vmax = std::max(vmax, std::fabs(v[static_cast<size_t>(i) * W + j]) / (R_EARTH_M * dphi));
    }
    umax += 1e-12;
    vmax += 1e-12;
    double dt = 0.45 / (umax + vmax);
    dt = std::min(dt, tau * 0.25);
    n_steps = static_cast<int64_t>(std::ceil(days * 86400.0 / dt));
    dt_out = dt;
    std::vector<double> ue(u.size()), vnc(static_cast<size_t>(H - 1) * W), vn(static_cast<size_t>(H - 1) * W);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) ue[static_cast<size_t>(i) * W + j] = 0.5 * (u[static_cast<size_t>(i) * W + j] + u[static_cast<size_t>(i) * W + (j + 1) % W]);
    for (int i = 0; i + 1 < H; ++i) {
        const double cosn = std::cos(0.5 * (phi[i] + phi[i + 1]));
        for (int j = 0; j < W; ++j) {
            const size_t c = static_cast<size_t>(i) * W + j;
            vn[c] = 0.5 * (v[c] + v[c + W]);
            vnc[c] = vn[c] * cosn;
        }
    }
    std::vector<double> inv_x(H), inv_y(H);
    for (int i = 0; i < H; ++i) {
        inv_x[i] = 1.0 / (R_EARTH_M * cosphi[i] * dlam);
        inv_y[i] = 1.0 / (R_EARTH_M * cosphi[i] * dphi);
    }
    std::vector<double> rate(eps.size());
    q.resize(eps.size());
    for (size_t c = 0; c < eps.size(); ++c) {
        rate[c] = eps[c] / tau;
        q[c] = E[c] * tau / np_maximum(eps[c], 1e-6);   // 局地平衡作初值
    }
    std::vector<uint8_t> polar_rows(H);
    for (int i = 0; i < H; ++i) polar_rows[i] = std::fabs(ax.lats[i]) > polar;
    std::vector<double> Fx(u.size()), Fy(static_cast<size_t>(H + 1) * W, 0.0), qn(q.size());
    for (int64_t s = 0; s < n_steps; ++s) {
        for (int i = 0; i < H; ++i) {
            const double* qr = &q[static_cast<size_t>(i) * W];
            for (int j = 0; j < W; ++j) {
                const size_t c = static_cast<size_t>(i) * W + j;
                Fx[c] = ue[c] * (ue[c] > 0 ? qr[j] : qr[(j + 1) % W]);
            }
        }
        for (int i = 0; i + 1 < H; ++i)
            for (int j = 0; j < W; ++j) {
                const size_t c = static_cast<size_t>(i) * W + j;
                Fy[c + W] = vnc[c] * (vn[c] > 0 ? q[c] : q[c + W]);
            }
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                const size_t c = static_cast<size_t>(i) * W + j;
                const double divx = (Fx[c] - Fx[static_cast<size_t>(i) * W + (j - 1 + W) % W]) * inv_x[i];
                const double divy = (Fy[c + W] - Fy[c]) * inv_y[i];
                const double qq = q[c] + dt * (E[c] - q[c] * rate[c] - divx - divy);
                qn[c] = np_maximum(qq, 0.0);
            }
        q.swap(qn);
        for (int i = 0; i < H; ++i)
            if (polar_rows[i]) {
                double* r = &q[static_cast<size_t>(i) * W];
                const double m = np_sum(r, W) / static_cast<double>(W);
                for (int j = 0; j < W; ++j) r[j] = m;
            }
    }
    P.resize(q.size());
    for (size_t c = 0; c < q.size(); ++c) P[c] = q[c] * rate[c];
}

GridD to_grid(const std::vector<double>& v, int H, int W) {
    GridD g(H, W);
    g.v = v;
    return g;
}

// moisture.coarsen：块平均到 res_run（整数倍；mean(axis=(1, 3))）
std::vector<double> coarsen(const std::vector<double>& f, const Axes& ax, double res_run, int& Hc, int& Wc) {
    const int fct = static_cast<int>(std::nearbyint(res_run / (ax.lats[1] - ax.lats[0])));
    if (fct <= 1) {
        Hc = ax.nlat;
        Wc = ax.nlon;
        return f;
    }
    const int H = ax.nlat / fct * fct, W = ax.nlon / fct * fct;
    GridD g(H, W);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) g(i, j) = f[static_cast<size_t>(i) * ax.nlon + j];
    GridD b = block_mean(g, fct);
    Hc = b.H;
    Wc = b.W;
    return b.v;
}

}  // namespace

std::vector<double> insolation_first_harmonic(const std::vector<double>& lat_deg, double tilt_deg, int n) {
    std::vector<double> t(n), sd(n), cd(n), td(n);
    std::vector<std::complex<double>> c(n);
    const double st = std::sin(deg2rad(tilt_deg));
    for (int k = 0; k < n; ++k) {
        t[k] = (static_cast<double>(k) + 0.5) / static_cast<double>(n);
        const double dec = std::asin(st * std::sin(2 * PI * t[k]));
        sd[k] = std::sin(dec);
        cd[k] = std::cos(dec);
        td[k] = std::tan(dec);
        const double ang = -(2 * PI * t[k]);   // exp(−2jπt)：实部 0，虚部 −(2π·t)
        c[k] = {std::cos(ang), std::sin(ang)};
    }
    std::vector<double> out(lat_deg.size());
    std::vector<double> re(n), im(n);
    for (size_t m = 0; m < lat_deg.size(); ++m) {
        const double phi = deg2rad(lat_deg[m]);
        const double sp = std::sin(phi), cp = std::cos(phi), tp = -std::tan(phi);
        for (int k = 0; k < n; ++k) {
            const double x = clip(tp * td[k], -1.0, 1.0);
            const double h0 = std::acos(x);
            const double q = SOLAR_CONST / PI * (h0 * sp * sd[k] + cp * cd[k] * std::sin(h0));
            re[k] = q * c[k].real();
            im[k] = q * c[k].imag();
        }
        // 复数的成对求和（numpy 的 CDOUBLE pairwise：按双精度个数 2n 分块，4 个复数累加器）
        struct PW {
            static void sum(const double* r, const double* i, size_t D, double& rr, double& ri) {
                if (D < 8) {
                    rr = 0.0;
                    ri = 0.0;
                    for (size_t e = 0; e < D / 2; ++e) {
                        rr += r[e];
                        ri += i[e];
                    }
                } else if (D <= 128) {
                    double ar[4] = {r[0], r[1], r[2], r[3]}, ai[4] = {i[0], i[1], i[2], i[3]};
                    size_t k = 8;
                    for (; k < D - (D % 8); k += 8)
                        for (int a = 0; a < 4; ++a) {
                            ar[a] += r[k / 2 + a];
                            ai[a] += i[k / 2 + a];
                        }
                    rr = (ar[0] + ar[1]) + (ar[2] + ar[3]);
                    ri = (ai[0] + ai[1]) + (ai[2] + ai[3]);
                    for (; k < D; k += 2) {
                        rr += r[k / 2];
                        ri += i[k / 2];
                    }
                } else {
                    size_t n2 = D / 2;
                    n2 -= n2 % 8;
                    double r1, i1, r2, i2;
                    sum(r, i, n2, r1, i1);
                    sum(r + n2 / 2, i + n2 / 2, D - n2, r2, i2);
                    rr = r1 + r2;
                    ri = i1 + i2;
                }
            }
        };
        double sr, si;
        PW::sum(re.data(), im.data(), static_cast<size_t>(2 * n), sr, si);
        sr = 0.0 + sr;
        si = 0.0 + si;
        // 除以个数：numpy 的复数除法（Smith）对 (n + 0j) 是乘以 1/n
        const double scl = 1.0 / (static_cast<double>(n) + 0.0 * 0.0);
        const double mr = (sr + si * 0.0) * scl, mi = (si - sr * 0.0) * scl;
        out[m] = 2.0 * np_cabs(mr, mi);
    }
    return out;
}

std::vector<double> season_range(const std::vector<double>& lat_deg, const std::vector<double>& cont, double tilt_deg, double year_days,
                                 const Config& cfg, double insolation_rel) {
    const double tl = cfg.get("s04.climate.season_tau_land_days"), to = cfg.get("s04.climate.season_tau_ocean_days");
    const double lam = cfg.get("s04.climate.season_lambda_w_m2_k");
    const std::vector<double> dq = insolation_first_harmonic(lat_deg, tilt_deg);
    const double w = 2.0 * PI / year_days;
    std::vector<double> out(lat_deg.size());
    for (size_t m = 0; m < lat_deg.size(); ++m) {
        const double c = clip(cont[m], 0.0, 1.0);
        const double tau = c * tl + (1.0 - c) * to;
        const double wt = w * tau;
        const double A = 1.0 / std::sqrt(1.0 + wt * wt);
        out[m] = 2.0 * (dq[m] * insolation_rel) / lam * A;
    }
    return out;
}

Climate stage4(const Config& cfg, uint64_t seed, const Planet& p, const Winds& w, const Islands& isl) {
    Climate C;
    const double res = cfg.get("shared.grid_res_deg");
    C.ax = grid_axes(res);
    const Axes& ax = C.ax;
    const int H = ax.nlat, W = ax.nlon;
    const size_t M = ax.size();
    const LatLonGrid G = llg(ax);
    const GInfo& gi = w.g;

    // ---------- ②b 岛对风的扰动
    std::vector<double> O, L;
    obstacle_fields(isl, ax, cfg, O, L);
    std::vector<double> D;
    band_displacement(O, ax, p, cfg, C.dphi, D);
    C.lat_eff.resize(M);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) C.lat_eff[static_cast<size_t>(i) * W + j] = ax.lats[i] - D[static_cast<size_t>(i) * W + j];
    std::vector<double> u_bg, v_bg, du, dv;
    wind_profile(C.lat_eff, cfg, p, u_bg, v_bg);
    g_vortex(ax, gi.lat, gi.lon, gi.radius_deg, cfg.get("s02.wind.g_vortex_speed"), du, dv);
    C.u_bg.resize(M);
    C.v_bg.resize(M);
    for (size_t c = 0; c < M; ++c) {
        C.u_bg[c] = u_bg[c] + du[c];
        C.v_bg[c] = v_bg[c] + dv[c];
    }
    {   // perturb_wind：摩擦 + 绕流偏转 + 尾流
        const double r = ax.lats[1] - ax.lats[0];
        const std::vector<double> dOdy = gradient_rows(O, H, W, r);
        const double kb = cfg.get("s04.localwind.deflect_k"), kf = cfg.get("s04.localwind.friction_k");
        std::vector<double> u2(M), v2(M);
        for (int i = 0; i < H; ++i) {
            const double cosl = np_maximum(std::cos(deg2rad(ax.lats[i])), 0.1);
            for (int j = 0; j < W; ++j) {
                const size_t c = static_cast<size_t>(i) * W + j;
                const double dOdx = (O[static_cast<size_t>(i) * W + (j + 1) % W] - O[static_cast<size_t>(i) * W + (j - 1 + W) % W]) / (2.0 * r) / cosl;
                const double spd = np_hypot(C.u_bg[c], C.v_bg[c]);
                const double fric = 1.0 - kf * O[c];
                u2[c] = (C.u_bg[c] - kb * spd * dOdx) * fric;
                v2[c] = (C.v_bg[c] - kb * spd * dOdy[c]) * fric;
            }
        }
        C.wake = wake_field(O, u2, v2, ax, cfg);
        const double kw = cfg.get("s04.localwind.wake_k");
        C.u.resize(M);
        C.v.resize(M);
        for (size_t c = 0; c < M; ++c) {
            C.u[c] = u2[c] * (1.0 - kw * C.wake[c]);
            C.v[c] = v2[c] * (1.0 - kw * C.wake[c]);
        }
    }
    const double vlm = cfg.get("s04.localwind.local_wind_max"), llr = cfg.get("s04.localwind.local_land_ref");
    C.v_local.resize(M);
    for (size_t c = 0; c < M; ++c) C.v_local[c] = vlm * clip(L[c] / llr, 0.0, 1.0);
    const std::vector<double> e0 = edge_values(p);
    C.edges.resize(static_cast<size_t>(8) * W);
    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < W; ++j) C.edges[static_cast<size_t>(k) * W + j] = e0[k] + C.dphi[static_cast<size_t>(k) * W + j];
    // 局部带号（band_id_of + local_edges：经度线性插值，edges 按 float64）
    C.band.resize(M);
    {
        const double lres = ax.lons[1] - ax.lons[0];
        for (int j = 0; j < W; ++j) {
            const double fj = (ax.lons[j] - ax.lons[0]) / lres;
            const int64_t j0f = static_cast<int64_t>(std::floor(fj));
            const double t = fj - static_cast<double>(j0f);
            const int64_t j0 = ((j0f % W) + W) % W, j1 = (j0 + 1) % W;
            double e[8];
            for (int k = 0; k < 8; ++k) e[k] = C.edges[static_cast<size_t>(k) * W + j0] * (1 - t) + C.edges[static_cast<size_t>(k) * W + j1] * t;
            for (int i = 0; i < H; ++i) {
                const double la = ax.lats[i];
                const bool north = la >= 0;
                const double eq = north ? e[0] : -e[4], tr = north ? e[1] : -e[5], ca = north ? e[2] : -e[6], we = north ? e[3] : -e[7];
                const double a = std::fabs(la);
                const int tier = a < eq ? 0 : a < tr ? 1 : a < ca ? 2 : a < we ? 3 : 4;
                C.band[static_cast<size_t>(i) * W + j] = static_cast<int16_t>(tier == 0 ? 0 : (north ? tier : tier + 4));
            }
        }
    }
    C.obstacle = O;
    C.land = L;

    // ---------- 温度（°C，海面）
    const double teq = cfg.get("s04.climate.temp_eq_c"), tpole = cfg.get("s04.climate.temp_pole_c");
    C.temp.resize(M);
    for (int i = 0; i < H; ++i) {
        const double s = std::sin(deg2rad(std::fabs(ax.lats[i])));
        const double t = (teq - (teq - tpole) * (s * s)) * p.insolation_rel;
        for (int j = 0; j < W; ++j) C.temp[static_cast<size_t>(i) * W + j] = t;
    }
    // ---------- 风暴
    const double storm_k = cfg.get("s04.localwind.storm_k");
    const double mid_c = cfg.has("s04.climate.storm_midlat_lat_deg") ? cfg.get("s04.climate.storm_midlat_lat_deg") * p.band_scale
                                                                     : 0.5 * (p.calm_top + p.westerlies_top);
    std::vector<double> amps;
    if (cfg.has_list("s04.climate.storm_shear_amp")) amps = cfg.list("s04.climate.storm_shear_amp");
    else amps.assign(4, cfg.get("s04.climate.storm_shear_amp"));
    const double shear_edges[4] = {p.eq_storm_top, p.trades_top, p.calm_top, p.westerlies_top};
    const double sw = cfg.get("s04.climate.storm_shear_width_deg");
    const double seq = cfg.get("s04.climate.storm_eq_amp"), sma = cfg.get("s04.climate.storm_midlat_amp"),
                 smw = cfg.get("s04.climate.storm_midlat_width_deg"), sga = cfg.get("s04.climate.storm_g_amp");
    const Vec3 g_xyz = latlon_to_xyz(gi.lat, gi.lon);
    C.storm.resize(M);
    C.storm_no_g.resize(M);
    C.stability.resize(M);
    std::vector<double> ae(M);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t c = static_cast<size_t>(i) * W + j;
            ae[c] = std::fabs(C.lat_eff[c]);
            double s = seq * gauss(ae[c], 0.0, 0.8 * p.eq_storm_top);
            for (size_t k = 0; k < 4 && k < amps.size(); ++k) s = s + amps[k] * gauss(ae[c], shear_edges[k], sw);
            s = s + sma * gauss(ae[c], mid_c, smw);
            s = s * (1.0 - storm_k * O[c]);
            C.storm_no_g[c] = clip(s, 0.0, 1.0);
            const double dg = rad2deg(angdist(latlon_to_xyz(ax.lats[i], ax.lons[j]), g_xyz));
            s = s + sga * gauss(dg, 0.0, 1.2 * gi.radius_deg);
            C.storm[c] = clip(s, 0.0, 1.0);
            C.stability[c] = 1.0 - C.storm[c];
        }

    // ---------- 降水：上风水汽追踪
    const double ec = cfg.get("s04.climate.evap_temp_coeff"), er = cfg.get("s04.climate.evap_temp_ref_c");
    std::vector<double> E(M);
    for (size_t c = 0; c < M; ++c) E[c] = std::exp(ec * (C.temp[c] - er));
    {
        const std::vector<double> div = divergence(C.u, C.v, ax);
        std::vector<double> mid;
        for (int i = 0; i < H; ++i)
            if (std::fabs(ax.lats[i]) < 80.0)
                for (int j = 0; j < W; ++j) mid.push_back(div[static_cast<size_t>(i) * W + j]);
        double sd = np_std(mid);
        if (sd == 0.0) sd = 1.0;
        C.conv.resize(M);
        for (size_t c = 0; c < M; ++c) C.conv[c] = clip(-div[c] / sd, -2.0, 2.0);
    }
    {
        const std::vector<double> dOdy = gradient_rows(O, H, W, res);
        std::vector<double> up(M), pos;
        for (int i = 0; i < H; ++i) {
            const double cosl = np_maximum(std::cos(deg2rad(ax.lats[i])), 0.1);
            for (int j = 0; j < W; ++j) {
                const size_t c = static_cast<size_t>(i) * W + j;
                const double dOdx = (O[static_cast<size_t>(i) * W + (j + 1) % W] - O[static_cast<size_t>(i) * W + (j - 1 + W) % W]) / (2.0 * res) / cosl;
                up[c] = np_maximum(0.0, C.u[c] * dOdx + C.v[c] * dOdy[c]);
                if (up[c] > 0) pos.push_back(up[c]);
            }
        }
        const double up_ref = pos.empty() ? 1.0 : np_quantile(pos, 0.98);
        C.uplift.resize(M);
        for (size_t c = 0; c < M; ++c) C.uplift[c] = clip(up[c] / std::max(up_ref, 1e-9), 0.0, 1.0);
    }
    const double pck = cfg.get("s04.climate.precip_conv_k"), puk = cfg.get("s04.climate.precip_uplift_k"), psk = cfg.get("s04.climate.precip_storm_k");
    C.eps.resize(M);
    for (size_t c = 0; c < M; ++c) C.eps[c] = std::exp(pck * C.conv[c]) * (1.0 + puk * C.uplift[c] + psk * C.storm[c]);
    std::vector<double> qf, Pf;
    {   // run_on_coarse：粗网格上解，再双线性插回
        const double res_run = cfg.get("s04.climate.moisture_res_deg");
        const Axes axc = grid_axes(res_run);
        int Hc = 0, Wc = 0;
        const std::vector<double> uc = coarsen(C.u, ax, res_run, Hc, Wc), vc = coarsen(C.v, ax, res_run, Hc, Wc),
                                  Ec = coarsen(E, ax, res_run, Hc, Wc), epc = coarsen(C.eps, ax, res_run, Hc, Wc);
        if (Hc != axc.nlat || Wc != axc.nlon) throw std::invalid_argument("s04.climate.moisture_res_deg must be a multiple of shared.grid_res_deg");
        std::vector<double> qc, Pc;
        moisture_solve(uc, vc, Ec, epc, axc, cfg, qc, Pc, C.dt_s, C.n_steps);
        const LatLonGrid Gc = llg(axc);
        qf.resize(M);
        Pf.resize(M);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                qf[static_cast<size_t>(i) * W + j] = grid_interp(qc, Gc, ax.lats[i], ax.lons[j]);
                Pf[static_cast<size_t>(i) * W + j] = grid_interp(Pc, Gc, ax.lats[i], ax.lons[j]);
            }
    }
    {
        const double P_ref = np_quantile(Pf, cfg.get("s04.climate.precip_norm_pct") / 100.0);
        C.precip.resize(M);
        for (size_t c = 0; c < M; ++c) C.precip[c] = Pf[c] / std::max(P_ref, 1e-12);
        const double amp = cfg.get("s04.climate.precip_noise_amp", 0.0);
        if (amp > 0) {
            Rng rng = stage_rng(seed, 4);
            const std::vector<double> nz = fractal_noise(rng, H, W, 6, 3);
            for (size_t c = 0; c < M; ++c) C.precip[c] = C.precip[c] + amp * nz[c];
        }
        for (double& x : C.precip) x = clip(x, 0.02, 1.0);
        const double q98 = std::max(np_quantile(qf, 0.98), 1e-12);
        C.q.resize(M);
        for (size_t c = 0; c < M; ++c) C.q[c] = clip(qf[c] / q98, 0.0, 1.0);
    }
    // ---------- 季节窗口
    const double tilt = p.axial_tilt_deg;
    C.window.resize(M);
    for (size_t c = 0; c < M; ++c) C.window[c] = clip(1.0 - 0.85 * C.storm[c] - 0.004 * tilt * gauss(ae[c], mid_c, 12.0), 0.03, 1.0);
    // ---------- 季节强度：全年温差 = 日照年变化 / λ × 热惯性振幅保留（区域陆地性 = L / obstacle_gain）
    const double ydays = p.year_days();
    const double gain = cfg.get("s04.localwind.obstacle_gain", 1.0);
    C.continentality.resize(M);
    for (size_t c = 0; c < M; ++c) C.continentality[c] = clip(L[c] / std::max(1e-9, gain), 0.0, 1.0);
    {
        const std::vector<double> amp = insolation_first_harmonic(ax.lats, tilt);
        const double tl = cfg.get("s04.climate.season_tau_land_days"), to = cfg.get("s04.climate.season_tau_ocean_days");
        const double lam = cfg.get("s04.climate.season_lambda_w_m2_k");
        const double wy = 2.0 * PI / ydays;
        C.season_range.resize(M);
        for (int i = 0; i < H; ++i) {
            const double dq2 = 2.0 * (amp[i] * p.insolation_rel) / lam;
            for (int j = 0; j < W; ++j) {
                const size_t c = static_cast<size_t>(i) * W + j;
                const double cc = clip(C.continentality[c], 0.0, 1.0);
                const double tau = cc * tl + (1.0 - cc) * to;
                const double wt = wy * tau;
                C.season_range[c] = dq2 * (1.0 / std::sqrt(1.0 + wt * wt));
            }
        }
    }

    // ---------- 各群
    const size_t N = isl.n();
    std::vector<double> cont_i(N), cont_alt(N);
    const double keel = cfg.get("s03.islands.keel_clearance_m", 300.0);
    const double alt_k = cfg.get("s04.climate.season_alt_continentality", 0.0);
    for (size_t q = 0; q < N; ++q) {
        cont_i[q] = grid_interp(C.continentality, G, isl.lat[q], isl.lon[q]);
        cont_alt[q] = clip(cont_i[q] + alt_k * clip((f32(isl.height[q]) - keel) / 2000.0, 0.0, 1.0), 0.0, 1.0);
    }
    C.i_season_range_sea = season_range(isl.lat, cont_i, tilt, ydays, cfg, p.insolation_rel);
    C.i_season_range = season_range(isl.lat, cont_alt, tilt, ydays, cfg, p.insolation_rel);
    const float lapse32 = static_cast<float>(cfg.get("s04.climate.lapse_c_per_km"));
    C.i_temp_sea.resize(N);
    C.i_precip.resize(N);
    C.i_storm.resize(N);
    C.i_stability.resize(N);
    C.i_window.resize(N);
    C.i_temp.resize(N);
    C.i_catch.resize(N);
    C.i_has_river.resize(N);
    C.i_river_size.resize(N);
    C.i_temp_winter.resize(N);
    C.i_temp_summer.resize(N);
    const double rma = cfg.get("s04.climate.river_main_area_km2"), rh = cfg.get("s04.climate.river_height_m"),
                 rpm = cfg.get("s04.climate.river_precip_min");
    for (size_t q = 0; q < N; ++q) {
        const double la = isl.lat[q], lo = isl.lon[q];
        C.i_temp_sea[q] = grid_interp(C.temp, G, la, lo);
        C.i_precip[q] = grid_interp(C.precip, G, la, lo);
        C.i_storm[q] = grid_interp(C.storm, G, la, lo);
        C.i_stability[q] = grid_interp(C.stability, G, la, lo);
        C.i_window[q] = grid_interp(C.window, G, la, lo);
        // float(lapse) × height_m（float32 数组）/ 1000：NEP 50 下按 float32 算
        const float h32 = static_cast<float>(isl.height[q]);
        const float lapse_h = static_cast<float>(lapse32 * h32) / 1000.0f;
        C.i_temp[q] = grid_interp(C.temp, G, la, lo) - static_cast<double>(lapse_h);
        C.i_catch[q] = f32(isl.arable_frac[q]) * f32(isl.area[q]) * C.i_precip[q];
        const double ma = f32(isl.main_area[q]);
        C.i_has_river[q] = (ma >= rma) && (f32(isl.height[q]) >= rh) && (C.i_precip[q] >= rpm);
        C.i_river_size[q] = C.i_has_river[q] ? ma * C.i_precip[q] : 0.0;
    }
    const double bonus = cfg.get("s04.climate.river_capacity_bonus");
    if (bonus > 0) {
        std::vector<double> rs;
        for (size_t q = 0; q < N; ++q)
            if (C.i_has_river[q]) rs.push_back(C.i_river_size[q]);
        if (!rs.empty()) {
            const double med = np_median(rs);
            for (size_t q = 0; q < N; ++q) C.i_catch[q] = C.i_catch[q] * (1.0 + bonus * C.i_river_size[q] / med);
        }
    }
    for (size_t q = 0; q < N; ++q) {
        C.i_temp_winter[q] = C.i_temp[q] - 0.5 * C.i_season_range[q];
        C.i_temp_summer[q] = C.i_temp[q] + 0.5 * C.i_season_range[q];
    }
    return C;
}

World run(const Config& cfg, uint64_t seed) {
    World w;
    w.planet = stage1(cfg);
    w.winds = stage2(cfg, w.planet);
    w.islands = stage3(cfg, seed, w.planet, w.winds);
    w.climate = stage4(cfg, seed, w.planet, w.winds, w.islands);
    return w;
}

}  // namespace skyisle::planet
