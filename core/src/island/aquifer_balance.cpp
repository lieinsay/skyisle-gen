#include "skyisle/island/aquifer_balance.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace skyisle::island {
namespace {
constexpr int DI[4] = {-1, 1, 0, 0}, DJ[4] = {0, 0, -1, 1};
double potential(double head, double face_bottom) {
    const double thickness = std::max(0.0, head - face_bottom);
    return 0.5 * thickness * thickness;
}
AquiferBalance solve_balance(const GridD& top, const GridD& base,
    const GridD& km, const GridD& recharge, const Mask& land,
    double dx, int max_iterations, double rtol,
    const GridD* previous = nullptr, const GridD* sy = nullptr, double dt = 0) {
    if (!(std::isfinite(dx) && dx > 0 && max_iterations > 0 &&
          std::isfinite(rtol) && rtol > 0 && rtol < 1))
        throw std::invalid_argument("invalid aquifer solver controls");
    const int H = top.H, W = top.W;
    for (const GridD* a : {&base, &km, &recharge})
        if (a->H != H || a->W != W) throw std::invalid_argument("aquifer grid shape mismatch");
    if (previous) {
        if (!sy || !std::isfinite(dt) || dt <= 0)
            throw std::invalid_argument("invalid aquifer time step");
        for (const GridD* a : {previous, sy})
            if (a->H != H || a->W != W) throw std::invalid_argument("aquifer storage grid shape mismatch");
    }
    if (land.H != H || land.W != W) throw std::invalid_argument("aquifer mask shape mismatch");
    AquiferBalance out;
    out.head = GridD(H, W, NaN);
    out.surface_m3s = out.coast_m3s = out.residual_m3s = GridD(H, W, 0.0);
    out.storage_change_m3 = GridD(H, W, 0.0);
    struct Cell {
        size_t k;
        int count = 0;
        size_t other[4]{};
        double conductance[4]{}, face_base[4]{};
        double coast = 0, source = 0, storage_rate = 0, previous = 0;
    };
    std::vector<Cell> cells;
    for (size_t k = 0; k < top.size(); ++k) {
        if (!land.v[k]) continue;
        if (!(std::isfinite(top.v[k]) && std::isfinite(base.v[k]) && top.v[k] >= base.v[k] &&
              std::isfinite(km.v[k]) && km.v[k] >= 0 && std::isfinite(recharge.v[k]) && recharge.v[k] >= 0))
            throw std::invalid_argument("invalid aquifer surface, bed, conductivity or recharge");
        if (previous && !(std::isfinite(previous->v[k]) && previous->v[k] >= base.v[k] &&
            previous->v[k] <= top.v[k] && std::isfinite(sy->v[k]) && sy->v[k] > 0 && sy->v[k] <= 1))
            throw std::invalid_argument("invalid aquifer previous head or specific yield");
        out.head.v[k] = previous ? previous->v[k] : base.v[k];
    }
    for (size_t k = 0; k < top.size(); ++k) {
        if (!land.v[k]) continue;
        Cell a;
        a.k = k;
        a.source = recharge.v[k] * dx * dx;
        if (previous) {
            a.storage_rate = sy->v[k] * dx * dx / dt;
            a.previous = previous->v[k];
        }
        out.recharge_m3s += a.source;
        const int i = static_cast<int>(k / W), j = static_cast<int>(k % W);
        for (int d = 0; d < 4; ++d) {
            const int ni = i+DI[d], nj = j+DJ[d];
            if (ni < 0 || ni >= H || nj < 0 || nj >= W || !land(ni, nj)) {
                // Half-cell distance to a free coastal face; head outside at
                // aquifer base, never an unlimited reservoir supplying water.
                a.coast += 2.0 * km.v[k];
            } else {
                const size_t n = static_cast<size_t>(ni)*W+nj;
                const double sum = km.v[k] + km.v[n];
                const int t = a.count++;
                a.other[t] = n;
                a.conductance[t] = sum > 0 ? 2.0*km.v[k]*km.v[n]/sum : 0.0;
                a.face_base[t] = std::max(base.v[k], base.v[n]);
            }
        }
        cells.push_back(a);
    }
    auto flux = [&](const Cell& a, double h) {
        double sum = a.coast * potential(h, base.v[a.k]) + a.storage_rate*(h-a.previous);
        for (int t = 0; t < a.count; ++t)
            sum += a.conductance[t] * (potential(h, a.face_base[t]) -
                                      potential(out.head.v[a.other[t]], a.face_base[t]));
        return sum;
    };
    auto derivative = [&](const Cell& a, double h) {
        double d = a.coast * std::max(0.0, h-base.v[a.k]) + a.storage_rate;
        for (int t = 0; t < a.count; ++t)
            d += a.conductance[t] * std::max(0.0, h-a.face_base[t]);
        return d;
    };
    // Exact zero-source solution; numerical dry-cell floors must not emit water.
    if (!previous && out.recharge_m3s == 0) { out.converged = true; return out; }
    for (int it = 0; it < max_iterations; ++it) {
        for (const Cell& a : cells) {
            const size_t k = a.k;
            if (flux(a, top.v[k]) <= a.source) {
                out.head.v[k] = top.v[k];
                continue;
            }
            double lo = base.v[k], hi = top.v[k];
            double h = std::clamp(out.head.v[k], lo, hi);
            for (int n = 0; n < 40; ++n) {
                const double error = flux(a, h)-a.source;
                if (std::abs(error) <= std::max(1e-15, a.source*1e-12)) break;
                if (error > 0) hi = h; else lo = h;
                const double grad = derivative(a, h);
                double next = grad > 0 ? h-error/grad : (lo+hi)/2;
                if (!(next > lo && next < hi)) next = (lo+hi)/2;
                h = next;
            }
            // Relax the integrated thickness potential, not the water source.
            // SOR accelerates long low-gradient flow paths; acceptance still
            // requires the original unmodified finite-volume mass residual.
            const double old = out.head.v[k]-base.v[k], next = h-base.v[k];
            const double upper = top.v[k]-base.v[k];
            // An isolated cell has no slow spatial mode: keep its direct root.
            // Transient storage is linear in head and damps spatial modes.
            const double omega = a.count && !previous ? 1.7 : 1.0;
            const double square = std::clamp(old*old+omega*(next*next-old*old), 0.0, upper*upper);
            out.head.v[k] = base.v[k]+std::sqrt(square);
        }
        out.iterations = it+1;
        out.discharge_m3s = out.max_residual_m3s = 0;
        out.total_storage_change_m3 = 0;
        double absolute_residual = 0;
        for (const Cell& a : cells) {
            const size_t k = a.k;
            const double h = out.head.v[k];
            const double residual = a.source-flux(a, h);
            const double seep = h == top.v[k] ? std::max(0.0, residual) : 0.0;
            out.surface_m3s.v[k] = seep;
            out.coast_m3s.v[k] = a.coast*potential(h, base.v[k]);
            out.residual_m3s.v[k] = residual-seep;
            out.storage_change_m3.v[k] = a.storage_rate*(h-a.previous)*dt;
            out.total_storage_change_m3 += out.storage_change_m3.v[k];
            absolute_residual += std::abs(residual-seep);
            out.max_residual_m3s = std::max(out.max_residual_m3s, std::abs(residual-seep));
            out.discharge_m3s += seep+out.coast_m3s.v[k];
        }
        // Sum of absolute cell residuals prevents errors cancelling globally.
        // No renormalization of heads or discharges to manufacture closure.
        const double scale = previous ? std::max(out.recharge_m3s, out.discharge_m3s) : out.recharge_m3s;
        const double tolerance = std::max(1e-12, scale * rtol);
        if (absolute_residual <= tolerance) { out.converged = true; break; }
    }
    return out;
}
} // namespace

AquiferBalance solve_aquifer_balance(const GridD& top, const GridD& base,
    const GridD& km, const GridD& recharge, const Mask& land,
    double dx, int max_iterations, double rtol) {
    return solve_balance(top, base, km, recharge, land, dx, max_iterations, rtol);
}

AquiferBalance step_aquifer_balance(const GridD& top, const GridD& base,
    const GridD& km, const GridD& recharge, const Mask& land,
    const GridD& previous, const GridD& sy, double dx, double dt,
    int max_iterations, double rtol) {
    return solve_balance(top, base, km, recharge, land, dx, max_iterations, rtol, &previous, &sy, dt);
}
} // namespace skyisle::island
