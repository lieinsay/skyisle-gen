// 势力范围（territory.py 同式）。
#include "skyisle/island/territory.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace skyisle::island {

std::vector<Limit> limits(const PlanetView& pv, int64_t node, double gap_km, double reach, double reach_km) {
    const double R = pv.radius_km;
    const size_t n = pv.isl_lat.size();
    const double d2r = PI / 180.0;
    std::vector<double> lat(n), lon(n), dist(n), r(n);
    for (size_t k = 0; k < n; ++k) {
        lat[k] = pv.isl_lat[k] * d2r;
        lon[k] = pv.isl_lon[k] * d2r;
    }
    const double p0 = lat[node], l0 = lon[node];
    const double sp0 = std::sin(p0), cp0 = std::cos(p0);
    for (size_t k = 0; k < n; ++k) {
        const double cosang = sp0 * std::sin(lat[k]) + cp0 * std::cos(lat[k]) * std::cos(lon[k] - l0);
        dist[k] = R * std::acos(clip(cosang, -1.0, 1.0));
        r[k] = std::sqrt(pv.isl_area[k] / PI);
    }
    std::vector<int64_t> near;
    for (size_t k = 0; k < n; ++k) {
        const double rc = reach * (r[node] + r[k]) + reach_km;
        if (dist[k] <= rc && static_cast<int64_t>(k) != node) near.push_back(static_cast<int64_t>(k));
    }
    std::stable_sort(near.begin(), near.end(), [&](int64_t a, int64_t b) { return dist[a] < dist[b]; });
    std::vector<Limit> out;
    for (int64_t j : near) {
        const double dl = lon[j] - l0;
        const double az = std::atan2(std::sin(dl) * std::cos(lat[j]), std::cos(p0) * std::sin(lat[j]) - std::sin(p0) * std::cos(lat[j]) * std::cos(dl));
        const double d = dist[j];
        const double t = d * r[node] / (r[node] + r[j]);
        out.push_back({j, d, std::sin(az), std::cos(az), t - 0.5 * gap_km});
    }
    return out;
}

namespace {
inline double margin(const Limit& L, double res_km) { return 1.5 * res_km * (std::fabs(L.ux) + std::fabs(L.uy)); }
}  // namespace

std::vector<double> mask_support(const Shape& s, double ox, double oy, const std::vector<Limit>& lim, double res_km) {
    std::vector<double> h(lim.size(), -1e9);
    const int H = s.mask.H, W = s.mask.W;
    for (size_t k = 0; k < lim.size(); ++k) {
        const double ux = lim[k].ux, uy = lim[k].uy;
        double best = -INF;
        bool any = false;
        for (int i = 0; i < H; ++i) {
            const double y = s.Y(i, 0) - oy;
            for (int j = 0; j < W; ++j) {
                if (!s.mask(i, j)) continue;
                const double v = (s.X(i, j) - ox) * ux + y * uy;
                if (v > best) best = v;
                any = true;
            }
        }
        if (any) h[k] = best + margin(lim[k], res_km);
    }
    return h;
}

std::vector<double> profile_support(const std::vector<double>& prof, const std::vector<Limit>& lim, double res_km) {
    const int n = static_cast<int>(prof.size());
    const double widen = 1.0 / std::cos(PI / n);
    std::vector<double> ct(n), st(n);
    for (int b = 0; b < n; ++b) {
        const double th = -PI + (b + 0.5) * (2 * PI / n);
        ct[b] = std::cos(th);
        st[b] = std::sin(th);
    }
    std::vector<double> h(lim.size());
    for (size_t k = 0; k < lim.size(); ++k) {
        double best = -INF;
        for (int b = 0; b < n; ++b) best = std::max(best, prof[b] * widen * (ct[b] * lim[k].ux + st[b] * lim[k].uy));
        h[k] = best + margin(lim[k], res_km);
    }
    return h;
}

double violation(double px, double py, const std::vector<double>& support, const std::vector<Limit>& lim) {
    if (lim.empty()) return -1e9;
    double worst = -INF;
    for (size_t k = 0; k < lim.size(); ++k) worst = std::max(worst, px * lim[k].ux + py * lim[k].uy + support[k] - lim[k].limit_km);
    return worst;
}

double nearest_fit(const std::vector<double>& support, const std::vector<Limit>& lim, double& ox, double& oy, int iters) {
    ox = 0.0;
    oy = 0.0;
    if (lim.empty()) return -1e9;
    const size_t m = lim.size();
    std::vector<double> b(m), P0(m, 0.0), P1(m, 0.0);
    for (size_t i = 0; i < m; ++i) b[i] = lim[i].limit_km - support[i];
    double x0 = 0.0, x1 = 0.0;
    // numpy 的 y @ U[i]（长 2 的点积走 BLAS ddot）= fma(y1, u1, y0·u0)；U @ x（dgemv）行数 ≥ 2 时 = fma(u0, x0, u1·x1)、1 行时同点积。
    // 越界量 v 在「恰好贴着分界线」时是 ±1e−15 量级，v ≤ 0 的判断差一位就翻（主岛挑哪个走向），所以照 numpy 的乘加次序算。
    for (int it = 0; it < iters; ++it)
        for (size_t i = 0; i < m; ++i) {
            const double y0 = x0 + P0[i], y1 = x1 + P1[i];
            const double s = std::fma(y1, lim[i].uy, y0 * lim[i].ux) - b[i];
            const double f = std::max(0.0, s);
            const double n0 = y0 - f * lim[i].ux, n1 = y1 - f * lim[i].uy;
            P0[i] = y0 - n0;
            P1[i] = y1 - n1;
            x0 = n0;
            x1 = n1;
        }
    ox = x0;
    oy = x1;
    double worst = -INF;
    for (size_t i = 0; i < m; ++i) {
        const double ux = m >= 2 ? std::fma(lim[i].ux, x0, lim[i].uy * x1) : std::fma(x1, lim[i].uy, x0 * lim[i].ux);
        worst = std::max(worst, ux - b[i]);
    }
    return worst;
}

double raster_violation(const Group& g, const std::vector<Limit>& lim) {
    if (lim.empty()) return -1e9;
    const double res = (g.res_km * 1000.0) / 1000.0;
    bool any = false;
    double worst = -1e9;
    for (const Limit& L : lim) {
        double best = -INF;
        for (int r = 0; r < g.H; ++r)
            for (int c = 0; c < g.W; ++c) {
                if (g.island_id(r, c) < 0) continue;
                any = true;
                const double x = g.origin_x + (c + 0.5) * res;
                const double y = g.origin_y - (r + 0.5) * res;
                best = std::max(best, x * L.ux + y * L.uy);
            }
        if (!any) return -1e9;
        worst = std::max(worst, best + 0.5 * res * (std::fabs(L.ux) + std::fabs(L.uy)) - L.limit_km);
    }
    return worst;
}

}  // namespace skyisle::island
