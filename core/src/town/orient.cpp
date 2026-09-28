#include "skyisle/town/orient.hpp"

#include <cmath>

#include "skyisle/town/geom.hpp"

namespace skyisle::town {

namespace {

// 目标方向与对称性：1 = 有向，2 = 模 π（山墙朝街：屋脊顺街），4 = 模 π/2（顺街成排）
bool target_of(const OrientRule& r, const OrientCtx& c, double& t, int& sym) {
    sym = 1;
    if (r.kind == "sun") t = c.sun;
    else if (r.kind == "wind") t = std::isfinite(c.wind_from) ? wrap_pi(c.wind_from + PI) : NaN;
    else if (r.kind == "street") t = c.street;
    else if (r.kind == "gable_street") t = c.street_dir, sym = 2;
    else if (r.kind == "align_street") t = c.street_dir, sym = 4;
    else if (r.kind == "water") t = c.water;
    else if (r.kind == "downslope") t = c.downslope;
    else if (r.kind == "yard") t = c.yard;
    else if (r.kind == "baseline") t = c.baseline;
    else if (r.kind == "fixed") t = r.angle;
    else return false;
    return std::isfinite(t);
}

double dev(double facing, double t, int sym) {
    double d = angle_diff(facing, t);
    if (sym == 2) d = std::min(d, PI - d);
    if (sym == 4) {
        d = std::fmod(d, 0.5 * PI);
        d = std::min(d, 0.5 * PI - d);
    }
    return d;
}

}  // namespace

double rule_deviation(const OrientRule& r, const OrientCtx& ctx, double facing) {
    double t;
    int sym;
    if (!target_of(r, ctx, t, sym)) return NaN;
    return dev(facing, t, sym);
}

double solve_facing(const std::vector<OrientRule>& rules, const OrientCtx& ctx, double snap, double fallback) {
    std::vector<double> cand;
    for (int k = 0; k < 360; ++k) cand.push_back(wrap_pi(k * PI / 180.0));
    bool any = false;
    for (const OrientRule& r : rules) {
        double t;
        int sym;
        if (!target_of(r, ctx, t, sym)) continue;
        any = true;
        for (int q = 0; q < sym; ++q) {
            const double base = t + q * 2.0 * PI / sym;
            for (double o : {0.0, r.tol, -r.tol}) cand.push_back(wrap_pi(base + o));
        }
    }
    if (!any) return fallback;
    double best = -1e18, arg = fallback;
    for (double th : cand) {
        double sc = 0.0;
        for (const OrientRule& r : rules) {
            double t;
            int sym;
            if (!target_of(r, ctx, t, sym)) continue;
            const double d = dev(th, t, sym);
            const double sat = d <= r.tol + 1e-9 ? 1.0 : std::max(0.0, std::cos(std::min(0.5 * PI, 3.0 * (d - r.tol))));
            sc += r.weight * (sat + 0.01 * std::cos(d));
        }
        if (sc > best + 1e-12) best = sc, arg = th;
    }
    if (snap > 0.0 && std::isfinite(ctx.sun)) arg = wrap_pi(ctx.sun + std::round(wrap_pi(arg - ctx.sun) / snap) * snap);
    return arg;
}

}  // namespace skyisle::town
