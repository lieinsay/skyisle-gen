#pragma once

#include <cmath>
#include <stdexcept>
#include "skyisle/config.hpp"

namespace skyisle::island {

// Hydraulic-geometry depth is A/W, not the depth at the thalweg.
// For z/D = |2x/W|^(f/b), integration gives A = W D f/(b+f).
// The coefficients remain a regional approximation, not a universal river law.
struct ChannelGeometry {
    double q_reference = 0.0;
    double width = 0.0;
    double max_depth = 0.0;
    double width_at_mean_flow = 0.0;
    double max_depth_at_mean_flow = 0.0;
};

inline double channel_area_factor(const Config& c) {
    const double b = c.get("hydro.at_station_width_b", 0.26);
    const double f = c.get("hydro.at_station_depth_f", 0.40);
    if (!(b > 0.0 && f > 0.0)) throw std::invalid_argument("channel exponents must be positive");
    return f / (b + f);
}

inline ChannelGeometry channel_geometry(double q, const Config& c) {
    ChannelGeometry out;
    if (!(q > 0.0)) return out;
    const double ratio = c.get("hydro.bf_ratio_channel", 5.0);
    if (!(ratio > 0.0)) throw std::invalid_argument("channel reference discharge must be positive");
    out.q_reference = q * ratio;
    out.width_at_mean_flow = c.get("hydro.width_a", 5.0) * std::pow(q, c.get("hydro.width_b", 0.5));
    out.max_depth_at_mean_flow = c.get("hydro.depth_c", 0.35) * std::pow(q, c.get("hydro.depth_f", 0.4)) / channel_area_factor(c);
    out.width = out.width_at_mean_flow * std::pow(ratio, c.get("hydro.at_station_width_b", 0.26));
    out.max_depth = out.max_depth_at_mean_flow * std::pow(ratio, c.get("hydro.at_station_depth_f", 0.40));
    return out;
}

} // namespace skyisle::island
