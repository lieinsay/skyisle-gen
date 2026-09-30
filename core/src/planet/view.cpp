// 行星层 → 第三层的输入（view.hpp）。
#include "skyisle/planet/view.hpp"

#include <algorithm>
#include <stdexcept>

namespace skyisle::planet {

double median_f32(const std::vector<double>& v_in) {
    std::vector<float> v(v_in.size());
    for (size_t k = 0; k < v.size(); ++k) v[k] = static_cast<float>(v_in[k]);
    if (v.empty()) return NaN;
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const float hi = v[h];
    if (n % 2) return hi;
    const float lo = *std::max_element(v.begin(), v.begin() + h);
    const float s = (0.0f + lo) + hi;                                   // mean 的 float32 累加
    return static_cast<double>(static_cast<float>(static_cast<double>(s) / 2.0));
}

island::Calendar island_calendar(const Planet& p) {
    island::Calendar c;
    if (p.cal.present) {
        c.seasons = p.cal.seasons;
        c.months_per_season = p.cal.months_per_season;
        c.days_per_month = p.cal.has_moon ? p.cal.moon.synodic_month_days : 28.0;
        c.days_per_season = p.cal.days_per_season;
        c.year_days = p.cal.year_days_solar;
    } else {
        c.seasons = 4;
        c.months_per_season = 3;
        c.days_per_month = 28.0;
        c.days_per_season = static_cast<double>(c.months_per_season) * c.days_per_month;
        c.year_days = static_cast<double>(c.seasons) * c.days_per_season;
    }
    c.offset = c.days_per_season * 1.5;
    return c;
}

island::PlanetView planet_view(const Planet& p, const Islands& isl, const Climate& c, const Config& cfg) {
    island::PlanetView v;
    v.radius_km = p.radius_km;
    const double yd = p.cal.present ? p.cal.year_days_solar : 336.0;
    const double hr = p.cal.present ? p.cal.solar_day_hr : 24.0;
    v.year_s = yd * hr * 3600.0;
    v.plate_grid = llg(isl.ax);
    v.plate_K = f32v(isl.boundary_kernel);
    v.plate_btype.assign(isl.btype.begin(), isl.btype.end());
    v.plate_lats = isl.ax.lats;
    v.plate_lons = isl.ax.lons;
    v.isl_lat = isl.lat;
    v.isl_lon = isl.lon;
    v.isl_area = f32v(isl.area);
    v.wind_grid = llg(c.ax);
    v.wind_u = f32v(c.u);
    v.wind_v = f32v(c.v);
    v.cg_grid = llg(c.ax);
    v.cg_precip = f32v(c.precip);
    v.cg_storm = f32v(c.storm);
    v.cg_window = f32v(c.window);
    v.cg_cont = f32v(c.continentality);
    const size_t W = c.ax.lons.size();
    v.band_lons = c.ax.lons;
    v.band_eq_n.assign(W, 0.0);
    v.band_eq_s.assign(W, 0.0);
    for (size_t j = 0; j < W; ++j) {
        v.band_eq_n[j] = f32(c.edges[0 * W + j]);   // EDGE_KEYS[0] = eq_n
        v.band_eq_s[j] = f32(c.edges[4 * W + j]);   // EDGE_KEYS[4] = eq_s
    }
    v.tilt_deg = p.axial_tilt_deg;
    v.tau_land = cfg.get("s04.climate.season_tau_land_days");
    v.tau_ocean = cfg.get("s04.climate.season_tau_ocean_days");
    v.alt_cont = cfg.get("s04.climate.season_alt_continentality", 0.0);
    v.cal = island_calendar(p);
    return v;
}

island::NodeInputs node_inputs(const Islands& isl, const Climate& c, int64_t node, uint64_t seed, const Config& cfg) {
    if (node < 0 || static_cast<size_t>(node) >= isl.n()) throw std::out_of_range("node out of range");
    const size_t q = static_cast<size_t>(node);
    island::NodeInputs x;
    x.node = node;
    x.seed = seed;
    x.lat = isl.lat[q];
    x.lon = isl.lon[q];
    x.area_km2 = f32(isl.area[q]);
    x.main_area_km2 = f32(isl.main_area[q]);
    x.height_m = f32(isl.height[q]);
    x.age = f32(isl.age[q]);
    x.layered = isl.layered[q] != 0;
    x.keel_clearance_m = cfg.get("s03.islands.keel_clearance_m", 300.0);
    x.area_median_km2 = median_f32(isl.area);
    x.precip = f32(c.i_precip[q]);
    x.temp_sea = f32(c.i_temp_sea[q]);
    x.lapse_c_per_km = cfg.get("s04.climate.lapse_c_per_km");
    x.arable_frac = c.i_arable_frac_eff.empty() ? f32(isl.arable_frac[q]) : f32(c.i_arable_frac_eff[q]);   // A3 起是降水线之后的
    x.precip_mm_ref = cfg.get("s04.climate.precip_mm_ref", 4000.0);
    x.river_size = f32(c.i_river_size[q]);
    x.has_river = c.i_has_river[q] != 0;
    x.temp = f32(c.i_temp[q]);
    x.storm = f32(c.i_storm[q]);
    x.window = f32(c.i_window[q]);
    x.season_range = f32(c.i_season_range[q]);
    x.season_range_sea = f32(c.i_season_range_sea[q]);
    x.temp_winter = f32(c.i_temp_winter[q]);
    x.temp_summer = f32(c.i_temp_summer[q]);
    return x;
}

}  // namespace skyisle::planet
