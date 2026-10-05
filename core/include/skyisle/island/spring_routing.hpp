#pragma once

#include "skyisle/island/aquifer_balance.hpp"

namespace skyisle::island {

struct SpringRoutingStep {
    AquiferBalance aquifer;
    GridD local_surface_m3s, river_m3s;
    double rain_input_m3s = 0, core_input_m3s = 0, surface_outlet_m3s = 0;
};

// Rain runoff is the net liquid yield after evapotranspiration/snow accounting.
// The recharge fraction partitions it; core water enters the aquifer only.
// Aquifer surface emergence joins D8 flow once; coastal groundwater leaves the
// island directly. No perennial mask or target river width enters this solver.
SpringRoutingStep step_spring_routing(const GridD& surface, const GridD& bottom,
    const GridD& conductivity_ms, const GridD& rain_runoff_ms,
    const GridD& core_ms, const GridD& recharge_fraction, const Mask& land,
    const std::vector<int64_t>& receiver, const GridD& previous_head,
    const GridD& specific_yield, double cell_m, double dt_s,
    int max_iterations = 5000, double relative_tolerance = 1e-7);

} // namespace skyisle::island
