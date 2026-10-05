#include "skyisle/island/spring_routing.hpp"
#include "skyisle/flow.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace skyisle::island {
namespace {
std::vector<int32_t> surface_order(const Mask& land, const std::vector<int64_t>& recv) {
    if (recv.size() != land.size()) throw std::invalid_argument("spring routing shape mismatch");
    size_t count = 0;
    for (size_t k = 0; k < land.size(); ++k) {
        if (!land.v[k]) continue;
        ++count;
        const int64_t r = recv[k];
        if (r < -1 || r >= static_cast<int64_t>(land.size()) || (r >= 0 && !land.v[r]))
            throw std::invalid_argument("invalid spring routing receiver");
        if (r >= 0 && (std::abs(static_cast<int64_t>(k/land.W)-r/land.W) > 1 ||
                       std::abs(static_cast<int64_t>(k%land.W)-r%land.W) > 1))
            throw std::invalid_argument("spring receiver is not a D8 neighbor");
    }
    auto order = downstream_first(recv, land);
    if (order.size() != count) throw std::invalid_argument("cyclic spring surface routing");
    return order;
}
SurfaceRoutingResult route_surface(const GridD& local, const Mask& land,
    const std::vector<int64_t>& recv, const std::vector<int32_t>& order) {
    SurfaceRoutingResult out;
    out.river_m3s = GridD(land.H, land.W, 0.0);
    for (size_t k = 0; k < land.size(); ++k) if (land.v[k]) {
        if (!(std::isfinite(local.v[k]) && local.v[k] >= 0))
            throw std::invalid_argument("invalid local surface supply");
        out.river_m3s.v[k] = local.v[k]; out.input_m3s += local.v[k];
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t r = recv[*it];
        if (r >= 0) out.river_m3s.v[r] += out.river_m3s.v[*it];
        else out.outlet_m3s += out.river_m3s.v[*it];
    }
    return out;
}
}
SurfaceRoutingResult route_surface_water(const GridD& local, const Mask& land,
                                         const std::vector<int64_t>& recv) {
    if (local.H != land.H || local.W != land.W) throw std::invalid_argument("surface supply grid mismatch");
    return route_surface(local, land, recv, surface_order(land, recv));
}
SpringRainStep spring_rain_step(const GridD& pscale, const GridD& fraction,
    const GridD& toffset, const GridD& snow, const Mask& land, double p, double temp,
    double dx, double dt, double snow_t, double melt_t, double ddf) {
    if (!(std::isfinite(p) && p >= 0 && std::isfinite(temp) && std::isfinite(dx) && dx > 0 &&
          std::isfinite(dt) && dt > 0 && std::isfinite(snow_t) && std::isfinite(melt_t) &&
          std::isfinite(ddf) && ddf >= 0)) throw std::invalid_argument("invalid spring weather input");
    for (const auto* grid : {&fraction, &toffset, &snow})
        if (grid->H != land.H || grid->W != land.W) throw std::invalid_argument("spring weather grid mismatch");
    if (pscale.H != land.H || pscale.W != land.W) throw std::invalid_argument("spring precipitation grid mismatch");
    SpringRainStep out;
    out.snowpack_mm = out.liquid_runoff_ms = GridD(land.H, land.W, 0.0);
    const double volume_per_mm = dx*dx/1000.0;
    for (size_t k = 0; k < land.size(); ++k) {
        if (!land.v[k]) continue;
        if (!(std::isfinite(pscale.v[k]) && pscale.v[k] >= 0 && std::isfinite(fraction.v[k]) &&
              fraction.v[k] >= 0 && fraction.v[k] <= 1 && std::isfinite(toffset.v[k]) &&
              std::isfinite(snow.v[k]) && snow.v[k] >= 0))
            throw std::invalid_argument("invalid spring local weather or runoff fraction");
        const double local_p = p*pscale.v[k], t = temp+toffset.v[k];
        double stored = snow.v[k], liquid = 0;
        if (t <= snow_t) stored += local_p;
        else liquid = local_p;
        const double melt = std::min(stored, ddf*std::max(0.0, t-melt_t)*dt/86400.0);
        stored -= melt;
        liquid += melt;
        const double runoff = liquid*fraction.v[k];
        out.snowpack_mm.v[k] = stored;
        out.liquid_runoff_ms.v[k] = runoff/(1000.0*dt);
        out.precipitation_m3 += local_p*volume_per_mm;
        out.snow_storage_change_m3 += (stored-snow.v[k])*volume_per_mm;
        out.runoff_m3 += runoff*volume_per_mm;
        out.nonrunoff_m3 += (liquid-runoff)*volume_per_mm;
    }
    return out;
}

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
    const auto order = surface_order(land, recv);
    SpringRoutingStep out;
    GridD recharge(top.H, top.W, 0.0);
    out.local_surface_m3s = GridD(top.H, top.W, 0.0);
    for (size_t k = 0; k < N; ++k) {
        if (!land.v[k]) continue;
        if (!(std::isfinite(rain.v[k]) && rain.v[k] >= 0 &&
            std::isfinite(core.v[k]) && core.v[k] >= 0 &&
            std::isfinite(fraction.v[k]) && fraction.v[k] >= 0 && fraction.v[k] <= 1))
            throw std::invalid_argument("invalid spring source or recharge fraction");
        recharge.v[k] = rain.v[k]*fraction.v[k]+core.v[k];
        out.local_surface_m3s.v[k] = rain.v[k]*(1.0-fraction.v[k])*dx*dx;
        out.rain_input_m3s += rain.v[k]*dx*dx;
        out.core_input_m3s += core.v[k]*dx*dx;
    }
    out.aquifer = step_aquifer_balance(top, bottom, km, recharge, land, previous,
                                    sy, dx, dt, max_iterations, rtol);
    if (!out.aquifer.converged)
        throw std::runtime_error("aquifer did not converge; surface routing was not accepted");
    for (size_t k = 0; k < N; ++k)
        out.local_surface_m3s.v[k] += out.aquifer.surface_m3s.v[k];
    auto routed = route_surface(out.local_surface_m3s, land, recv, order);
    out.river_m3s = std::move(routed.river_m3s);
    out.surface_outlet_m3s = routed.outlet_m3s;
    return out;
}
} // namespace skyisle::island
