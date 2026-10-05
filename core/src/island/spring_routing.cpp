#include "skyisle/island/spring_routing.hpp"
#include "skyisle/flow.hpp"

#include <cmath>
#include <stdexcept>

namespace skyisle::island {
SpringRoutingStep step_spring_routing(const GridD& top, const GridD& bottom,
    const GridD& km, const GridD& rain, const GridD& core, const GridD& fraction,
    const Mask& land, const std::vector<int64_t>& recv, const GridD& previous,
    const GridD& sy, double dx, double dt, int max_iterations, double rtol) {
    const size_t N = top.size();
    if (recv.size() != N || land.H != top.H || land.W != top.W)
        throw std::invalid_argument("spring routing shape mismatch");
    for (const auto* g : {&rain, &core, &fraction})
        if (g->H != top.H || g->W != top.W)
            throw std::invalid_argument("spring source shape mismatch");
    size_t count = 0;
    SpringRoutingStep out;
    GridD recharge(top.H, top.W, 0.0);
    out.local_surface_m3s = GridD(top.H, top.W, 0.0);
    for (size_t k = 0; k < N; ++k) {
        if (!land.v[k]) continue;
        ++count;
        if (!(std::isfinite(rain.v[k]) && rain.v[k] >= 0 &&
            std::isfinite(core.v[k]) && core.v[k] >= 0 &&
            std::isfinite(fraction.v[k]) && fraction.v[k] >= 0 && fraction.v[k] <= 1))
            throw std::invalid_argument("invalid spring source or recharge fraction");
        const int64_t r = recv[k];
        if (r < -1 || r >= static_cast<int64_t>(N) || (r >= 0 && !land.v[r]))
            throw std::invalid_argument("invalid spring routing receiver");
        if (r >= 0 && (std::abs(static_cast<int64_t>(k/top.W)-r/top.W) > 1 ||
                       std::abs(static_cast<int64_t>(k%top.W)-r%top.W) > 1))
            throw std::invalid_argument("spring receiver is not a D8 neighbor");
        recharge.v[k] = rain.v[k]*fraction.v[k]+core.v[k];
        out.local_surface_m3s.v[k] = rain.v[k]*(1.0-fraction.v[k])*dx*dx;
        out.rain_input_m3s += rain.v[k]*dx*dx;
        out.core_input_m3s += core.v[k]*dx*dx;
    }
    const auto order = downstream_first(recv, land);
    if (order.size() != count) throw std::invalid_argument("cyclic spring surface routing");
    out.aquifer = step_aquifer_balance(top, bottom, km, recharge, land, previous,
                                    sy, dx, dt, max_iterations, rtol);
    if (!out.aquifer.converged)
        throw std::runtime_error("aquifer did not converge; surface routing was not accepted");
    for (size_t k = 0; k < N; ++k)
        out.local_surface_m3s.v[k] += out.aquifer.surface_m3s.v[k];
    out.river_m3s = out.local_surface_m3s;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t r = recv[*it];
        if (r >= 0) out.river_m3s.v[r] += out.river_m3s.v[*it];
        else out.surface_outlet_m3s += out.river_m3s.v[*it];
    }
    return out;
}
} // namespace skyisle::island
