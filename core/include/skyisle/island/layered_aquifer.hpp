#pragma once

#include "skyisle/grid.hpp"
#include "skyisle/island/strat.hpp"
#include <array>

namespace skyisle::island {

struct HydraulicMaterial {
    double horizontal_ms = 0, vertical_ms = 0, specific_yield = 0, specific_storage_per_m = 0;
};

struct LayeredAquiferResult {
    std::vector<double> head_m, residual_m3s;
    GridD surface_m3s, coast_m3s, storage_change_m3;
    bool converged = false;
    int iterations = 0;
    double input_m3s = 0, output_m3s = 0, total_storage_change_m3 = 0, absolute_residual_m3s = 0;
};

// Layer-resolved finite volumes, hydrostatic horizontal pressure integration,
// series vertical resistance and a disclosed perched-head correction. This is
// not a Richards/karst simulation or an implementation of the full MODFLOW model.
// All material values are explicit SI inputs; no fitting to desired river size.
class LayeredAquifer {
public:
    LayeredAquifer(const GridD& surface, const GridD& structural_top, const GridD& skeleton_top,
                   const Mask& land, const StratRec& strata,
                   const std::array<HydraulicMaterial, LI_COUNT>& materials, double cell_m,
                   double shallow_depth_m = 0,
                   const std::array<HydraulicMaterial, LI_COUNT>& shallow_materials = {});
    LayeredAquiferResult solve(const GridD& recharge_ms, int max_iterations = 10000, double rtol = 1e-7) const;
    LayeredAquiferResult step(const GridD& recharge_ms, const std::vector<double>& previous_head,
                             double dt_s, int max_iterations = 10000, double rtol = 1e-7) const;
    std::vector<int64_t> node_cells() const;
    std::vector<double> node_bottoms() const;
    std::vector<double> node_tops() const;
    std::vector<int> node_lithologies() const;
private:
    struct Node {
        size_t cell = 0;
        double bottom = 0, top = 0, maximum_head = 0, coast = 0;
        HydraulicMaterial material;
        uint8_t lith = LI_VOID;
        bool surface = false;
        std::vector<size_t> edges;
        struct SlopeFace { size_t receiver; double bottom, top; };
        std::vector<SlopeFace> slope_faces;
    };
    struct Edge {
        size_t a = 0, b = 0;
        double conductance = 0, bottom = 0, top = 0;
        bool vertical = false;  // a is the upper node when vertical
    };
    int height_ = 0, width_ = 0;
    double area_ = 0;
    Mask land_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<size_t> offsets_;
    void connect(Edge edge);
    LayeredAquiferResult advance(const GridD& recharge, const std::vector<double>* previous,
                                 double dt, int max_iterations, double rtol) const;
};
} // namespace skyisle::island
