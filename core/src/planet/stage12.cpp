// ① 行星参数与历法（s01_planet.py、almanac.py）、② 风带与 G（s02_wind.py）。
// Python 浮点的 ** 一律 C 的 pow（c_pow）；numpy 数组的 ** 2 是平方（x·x）。
#include "skyisle/planet/planet.hpp"

#include <cmath>
#include <stdexcept>

namespace skyisle::planet {

namespace {

constexpr double G_SI = 6.674e-11;
constexpr double M_SUN = 1.989e30;
constexpr double T_SUN = 5772.0;
constexpr double AU = 1.496e11;
constexpr double YEAR_S = 365.25 * 86400.0;
constexpr double M_EARTH = 5.972e24;
constexpr double R_EARTH = 6.371e6;
constexpr double MOON_A = 384400.0;
constexpr double ALPHA_ML = 3.5;

std::string spectral_class(double teff) {
    const struct { const char* c; double lo; } tab[] = {{"O", 30000}, {"B", 10000}, {"A", 7500}, {"F", 6000}, {"G", 5200}, {"K", 3700}};
    for (const auto& t : tab)
        if (teff >= t.lo) return t.c;
    return "M";
}

Star star_from_mass(double m) {
    Star s;
    s.mass_msun = m;
    s.luminosity_lsun = c_pow(m, ALPHA_ML);
    s.radius_rsun = c_pow(m, 0.8);
    s.teff_k = T_SUN * c_pow(s.luminosity_lsun / c_pow(s.radius_rsun, 2), 0.25);
    s.spectral_class = spectral_class(s.teff_k);
    return s;
}

double tidal_lock_gyr(double rot_hr, double a_au, double m_star, double r_km, double density_rel, double q, double k2) {
    const double r = r_km * 1e3;
    const double m_p = M_EARTH * c_pow(r / R_EARTH, 3) * density_rel;
    const double omega = 2.0 * PI / (rot_hr * 3600.0);
    const double inertia = 0.4 * m_p * r * r;
    const double a = a_au * AU;
    const double t = omega * c_pow(a, 6) * inertia * q / (3.0 * G_SI * c_pow(m_star * M_SUN, 2) * k2 * c_pow(r, 5));
    return t / (YEAR_S * 1e9);
}

void orbit_from_period(double p_yr, double ins_cfg, double& m_star, double& a_au) {
    m_star = c_pow(ins_cfg * c_pow(p_yr, 4.0 / 3.0), 1.0 / (ALPHA_ML - 2.0 / 3.0));
    a_au = c_pow(c_pow(p_yr, 2) * m_star, 1.0 / 3.0);
}

}  // namespace

Almanac derive_calendar(const Config& cfg) {
    Almanac out;
    if (!cfg.has_prefix("s01.calendar.")) return out;   // `if not c: return {}`
    out.present = true;
    out.mode = cfg.gets("s01.calendar.mode", "calendar_to_orbit");
    const int seasons = cfg.geti("s01.calendar.seasons");
    const double dps = cfg.get("s01.calendar.days_per_season");
    const double rot_hr = cfg.get("s01.planet.rotation_period_hr");
    const double ins_cfg = cfg.get("s01.planet.insolation_rel");
    const double r_km = cfg.get("s01.planet.radius_km");
    const double dens = cfg.get("s01.calendar.planet_density_rel", 1.0);
    const double q = cfg.get("s01.calendar.tidal_q", 100.0), k2 = cfg.get("s01.calendar.tidal_k2", 0.3);
    const double prograde_extra = 1.0;
    out.seasons = seasons;
    out.days_per_season_config = dps;
    out.months_per_season = static_cast<int>(cfg.get("s01.calendar.months_per_season", 1.0));
    out.solar_day_hr = rot_hr;
    out.gravity_rel = (r_km / 6371.0) * dens;
    out.planet_mass_rel_earth = c_pow(r_km / 6371.0, 3) * dens;

    double m_star = 0, a_au = 0;
    if (out.mode == "calendar_to_orbit") {
        const double year_days = static_cast<double>(seasons) * dps;
        const double p_orb_days = (year_days + prograde_extra) * rot_hr / 24.0;
        const double p_yr = p_orb_days / 365.25;
        orbit_from_period(p_yr, ins_cfg, m_star, a_au);
        out.year_days_solar = year_days;
        out.days_per_season = dps;
        out.orbital_period_earth_days = p_orb_days;
        out.orbital_period_years = p_yr;
        out.semi_major_axis_au = a_au;
        out.insolation_derived = c_pow(m_star, ALPHA_ML) / c_pow(a_au, 2);
        out.insolation_config = ins_cfg;
    } else if (out.mode == "orbit_to_calendar") {
        m_star = cfg.get("s01.calendar.stellar_mass_msun");
        a_au = cfg.get("s01.calendar.semi_major_axis_au");
        const double p_yr = std::sqrt(c_pow(a_au, 3) / m_star);
        const double p_orb_days = p_yr * 365.25;
        const double year_days = p_orb_days * 24.0 / rot_hr - prograde_extra;
        out.year_days_solar = year_days;
        out.days_per_season = year_days / static_cast<double>(seasons);
        out.orbital_period_earth_days = p_orb_days;
        out.orbital_period_years = p_yr;
        out.semi_major_axis_au = a_au;
        out.insolation_derived = c_pow(m_star, ALPHA_ML) / c_pow(a_au, 2);
        out.insolation_config = ins_cfg;
        out.has_residual = true;
        out.days_per_season_residual = year_days / static_cast<double>(seasons) - dps;
    } else {
        throw std::invalid_argument("s01.calendar.mode unknown: " + out.mode);
    }
    out.star = star_from_mass(m_star);
    out.star.apparent_size_rel_sun = out.star.radius_rsun / a_au;
    out.tidal_lock_gyr = tidal_lock_gyr(rot_hr, a_au, m_star, r_km, dens, q, k2);
    const double age_gyr = cfg.get("s01.calendar.system_age_gyr", 4.6);
    if (out.tidal_lock_gyr < age_gyr && out.mode == "calendar_to_orbit") {
        for (int n = seasons; n < 60; ++n) {
            const double p_yr2 = ((static_cast<double>(n) * dps + prograde_extra) * rot_hr / 24.0) / 365.25;
            double m2 = 0, a2 = 0;
            orbit_from_period(p_yr2, ins_cfg, m2, a2);
            if (tidal_lock_gyr(rot_hr, a2, m2, r_km, dens, q, k2) >= age_gyr) {
                out.seasons_needed_for_no_lock = n;
                break;
            }
        }
    }
    if (cfg.get("s01.calendar.moon", 1.0) != 0.0) {
        out.has_moon = true;
        const double syn = cfg.get("s01.calendar.synodic_month_days", dps);
        const double year_days = out.year_days_solar;
        const double sid = 1.0 / (1.0 / syn + 1.0 / year_days);
        const double m_p = M_EARTH * out.planet_mass_rel_earth;
        const double t_s = sid * rot_hr * 3600.0;
        const double a_m = c_pow(G_SI * m_p * c_pow(t_s, 2) / (4.0 * c_pow(PI, 2)), 1.0 / 3.0) / 1e3;
        out.moon = {syn, sid, year_days / syn, a_m, a_m / r_km, c_pow(MOON_A / a_m, 3)};
    }
    return out;
}

double Planet::band(const std::string& key) const {
    if (key == "eq_storm_top_deg") return eq_storm_top;
    if (key == "trades_top_deg") return trades_top;
    if (key == "calm_top_deg") return calm_top;
    if (key == "westerlies_top_deg") return westerlies_top;
    throw std::out_of_range("planet: unknown band edge " + key);
}

Planet stage1(const Config& cfg) {
    Planet p;
    p.rotation_period_hr = cfg.get("s01.planet.rotation_period_hr");
    double scale = 1.0;
    if (cfg.get("s01.planet.held_hou_scaling", 0.0) != 0.0)   // Held–Hou：哈德莱圈宽度 ∝ Ω^-1（取平方根缓和）
        scale = std::min(1.6, std::max(0.5, std::sqrt(p.rotation_period_hr / 24.0)));
    p.band_scale = scale;
    p.eq_storm_top = cfg.get("s02.wind.eq_storm_top_deg") * scale;
    p.trades_top = cfg.get("s02.wind.trades_top_deg") * scale;
    p.calm_top = cfg.get("s02.wind.calm_top_deg") * scale;
    p.westerlies_top = cfg.get("s02.wind.westerlies_top_deg") * scale;
    if (p.westerlies_top >= 85.0) throw std::invalid_argument("westerlies top beyond 85 deg after band scaling (rotation too slow)");
    p.axial_tilt_deg = cfg.get("s01.planet.axial_tilt_deg");
    p.insolation_rel = cfg.get("s01.planet.insolation_rel");
    p.radius_km = cfg.get("s01.planet.radius_km");
    p.day_range_km = cfg.get("shared.day_range_km");
    p.circumference_days = 2 * PI * p.radius_km / p.day_range_km;
    p.cal = derive_calendar(cfg);
    return p;
}

// ---------------------------------------------------------------- ②
namespace {

inline double gauss(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return std::exp(-0.5 * (z * z));
}

int16_t band_id_lat(double lat, const Planet& p) {
    const double a = std::fabs(lat);
    int tier = a < p.eq_storm_top ? 0 : a < p.trades_top ? 1 : a < p.calm_top ? 2 : a < p.westerlies_top ? 3 : 4;
    if (tier == 0) return 0;
    return static_cast<int16_t>(lat >= 0 ? tier : tier + 4);
}

}  // namespace

void wind_profile(const std::vector<double>& lat_eff, const Config& cfg, const Planet& p, std::vector<double>& u, std::vector<double>& v) {
    const double eq_top = p.eq_storm_top, tr_top = p.trades_top, calm_top = p.calm_top, west_top = p.westerlies_top;
    const double tr_c = 0.5 * (eq_top + tr_top), tr_w = 0.35 * (tr_top - eq_top);
    const double we_c = 0.5 * (calm_top + west_top), we_w = 0.30 * (west_top - calm_top);
    const double po_c = 0.5 * (west_top + 90.0), po_w = 0.35 * (90.0 - west_top);
    const double trade = cfg.get("s02.wind.trade_speed"), westerly = cfg.get("s02.wind.westerly_speed"),
                 polar = cfg.get("s02.wind.polar_speed"), hadley = cfg.get("s02.wind.hadley_inflow_speed"),
                 ferrel = cfg.get("s02.wind.ferrel_outflow_speed", 0.0), polar_out = cfg.get("s02.wind.polar_outflow_speed", 0.0);
    u.resize(lat_eff.size());
    v.resize(lat_eff.size());
    for (size_t k = 0; k < lat_eff.size(); ++k) {
        const double a = std::fabs(lat_eff[k]);
        const double s = lat_eff[k] + 1e-12;
        const double sign = s > 0 ? 1.0 : (s < 0 ? -1.0 : 0.0);   // np.sign
        const double g1 = gauss(a, tr_c, tr_w), g2 = gauss(a, we_c, we_w), g3 = gauss(a, po_c, po_w);
        u[k] = (-trade) * g1 + westerly * g2 - polar * g3;
        v[k] = (-sign) * hadley * g1 + sign * ferrel * g2 - sign * polar_out * g3;
    }
}

void g_vortex(const Axes& ax, double g_lat, double g_lon, double g_r_deg, double vmax, std::vector<double>& du, std::vector<double>& dv) {
    const double g_r = deg2rad(g_r_deg);
    const Vec3 g = latlon_to_xyz(g_lat, g_lon);
    const double cg = std::cos(deg2rad(g_lat));
    du.resize(ax.size());
    dv.resize(ax.size());
    for (int i = 0; i < ax.nlat; ++i)
        for (int j = 0; j < ax.nlon; ++j) {
            const size_t k = static_cast<size_t>(i) * ax.nlon + j;
            const double la = ax.lats[i], lo = ax.lons[j];
            const double d = angdist(latlon_to_xyz(la, lo), g);
            const double vt = d < g_r ? vmax * d / g_r : vmax * std::exp(-(d - g_r) / (1.5 * g_r));
            const double dlat = la - g_lat;
            const double dlon = pymod(lo - g_lon + 180.0, 360.0) - 180.0;
            const double th = std::atan2(deg2rad(dlat), deg2rad(dlon) * cg);
            du[k] = vt * (-std::sin(th));
            dv[k] = vt * std::cos(th);
        }
}

Winds stage2(const Config& cfg, const Planet& p) {
    Winds w;
    w.ax = grid_axes(cfg.get("shared.grid_res_deg"));
    const Axes& ax = w.ax;
    std::vector<double> lat(ax.size());
    for (int i = 0; i < ax.nlat; ++i)
        for (int j = 0; j < ax.nlon; ++j) lat[static_cast<size_t>(i) * ax.nlon + j] = ax.lats[i];
    wind_profile(lat, cfg, p, w.u, w.v);
    // 定点永暴 G：纬度 = 锚定带界 − δ（skeleton.g_latitude），经度 = D 的中央
    w.g.edge = cfg.gets("skeleton.g_anchor_edge", "trades_top_deg");
    w.g.lat = p.band(w.g.edge) - cfg.get("skeleton.g_delta_deg");
    w.g.lon = 0.5 * (cfg.get("skeleton.d_lon_west") + cfg.get("skeleton.d_lon_east"));
    w.g.radius_deg = cfg.get("skeleton.g_radius_deg");
    std::vector<double> du, dv;
    g_vortex(ax, w.g.lat, w.g.lon, w.g.radius_deg, cfg.get("s02.wind.g_vortex_speed"), du, dv);
    w.band.resize(ax.size());
    for (size_t k = 0; k < ax.size(); ++k) {
        w.u[k] = w.u[k] + du[k];
        w.v[k] = w.v[k] + dv[k];
        w.band[k] = band_id_lat(lat[k], p);
    }
    return w;
}

}  // namespace skyisle::planet
