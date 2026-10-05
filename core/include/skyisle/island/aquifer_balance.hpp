#pragma once

#include "skyisle/grid.hpp"

namespace skyisle::island {

// Steady, depth-integrated unconfined aquifer with impermeable bed and exposed
// coastal faces. Inputs are independent of river width and preclassified drains.
// A seepage face is active only when head reaches surface and has excess inflow.
// This Dupuit approximation cannot resolve perched layers or individual karst
// conduits. Material conductivity is a caller-supplied model input, not inferred
// by fitting discharge or channel dimensions.
struct AquiferBalance {
    GridD head, surface_m3s, coast_m3s, residual_m3s, storage_change_m3;
    bool converged = false;
    int iterations = 0;
    double recharge_m3s = 0, discharge_m3s = 0, max_residual_m3s = 0;
    double total_storage_change_m3 = 0;
};

AquiferBalance solve_aquifer_balance(const GridD& surface, const GridD& bottom,
    const GridD& conductivity_ms, const GridD& recharge_ms, const Mask& land,
    double cell_m, int max_iterations = 5000, double relative_tolerance = 1e-7);

// Backward-Euler water-table storage: Sy * cell_area * (head - previous_head).
// Recharge is input at the water table, after any unsaturated-zone accounting.
// No yearly rescaling: recharge * dt = discharge * dt + storage change + error.
AquiferBalance step_aquifer_balance(const GridD& surface, const GridD& bottom,
    const GridD& conductivity_ms, const GridD& recharge_ms, const Mask& land,
    const GridD& previous_head, const GridD& specific_yield, double cell_m,
    double dt_s, int max_iterations = 5000, double relative_tolerance = 1e-7);

} // namespace skyisle::island
