// 行星层 ①–④ 的绑定（行星计划 P6c）：各步（planet_stage1–4 / planet_run）、产物的数组形（前端写 npz / json）、从产物读回（npz / json → C++）、
// 行星层 → 第三层（planet_view / node_inputs），以及给 pytest 同输入对照的公共件。
// 各步的产物是不透明对象（PlanetParams / Winds / Islands / Climate），前端按 run 与阶段 key 缓存在内存里，下一步直接吃；
// 缓存没命中的上游从 npz / json 读回（*_from）。数组一律双精度的原值，写 npz 时由前端照 Python 版转 float32。
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <stdexcept>

#include "bind_util.hpp"
#include "skyisle/planet/civ.hpp"
#include "skyisle/planet/planet.hpp"
#include "skyisle/planet/view.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace skyisle;
using namespace skyisle::planet;

namespace {

Axes axes_from(const std::vector<double>& lats, const std::vector<double>& lons) {
    Axes ax;
    ax.lats = lats;
    ax.lons = lons;
    ax.nlat = static_cast<int>(lats.size());
    ax.nlon = static_cast<int>(lons.size());
    ax.res = lats.size() > 1 ? lats[1] - lats[0] : 1.0;
    return ax;
}


// ---------------------------------------------------------------- ① planet.json 的形
nb::dict calendar_py(const Almanac& c) {
    nb::dict d;
    if (!c.present) return d;
    d["mode"] = c.mode;
    d["seasons"] = c.seasons;
    d["days_per_season_config"] = c.days_per_season_config;
    d["months_per_season"] = c.months_per_season;
    d["solar_day_hr"] = c.solar_day_hr;
    d["sailing_day_is_solar_day"] = true;
    d["gravity_rel"] = c.gravity_rel;
    d["planet_mass_rel_earth"] = c.planet_mass_rel_earth;
    d["year_days_solar"] = c.year_days_solar;
    d["days_per_season"] = c.days_per_season;
    d["orbital_period_earth_days"] = c.orbital_period_earth_days;
    d["orbital_period_years"] = c.orbital_period_years;
    d["semi_major_axis_au"] = c.semi_major_axis_au;
    d["insolation_derived"] = c.insolation_derived;
    d["insolation_config"] = c.insolation_config;
    if (c.has_residual) d["days_per_season_residual"] = c.days_per_season_residual;
    nb::dict s;
    s["mass_msun"] = c.star.mass_msun;
    s["luminosity_lsun"] = c.star.luminosity_lsun;
    s["radius_rsun"] = c.star.radius_rsun;
    s["teff_k"] = c.star.teff_k;
    s["spectral_class"] = c.star.spectral_class;
    s["apparent_size_rel_sun"] = c.star.apparent_size_rel_sun;
    d["star"] = s;
    d["tidal_lock_gyr"] = c.tidal_lock_gyr;
    if (c.seasons_needed_for_no_lock >= 0) d["seasons_needed_for_no_lock"] = c.seasons_needed_for_no_lock;
    if (c.has_moon) {
        nb::dict m;
        m["synodic_month_days"] = c.moon.synodic_month_days;
        m["sidereal_month_days"] = c.moon.sidereal_month_days;
        m["months_per_year"] = c.moon.months_per_year;
        m["distance_km"] = c.moon.distance_km;
        m["distance_planet_radii"] = c.moon.distance_planet_radii;
        m["tide_rel_earth_same_mass"] = c.moon.tide_rel_earth_same_mass;
        d["moon"] = m;
    }
    return d;
}

nb::dict planet_py(const Planet& p) {
    nb::dict d;
    d["rotation_period_hr"] = p.rotation_period_hr;
    d["axial_tilt_deg"] = p.axial_tilt_deg;
    d["insolation_rel"] = p.insolation_rel;
    d["radius_km"] = p.radius_km;
    d["band_scale"] = p.band_scale;
    nb::dict b;
    b["eq_storm_top_deg"] = p.eq_storm_top;
    b["trades_top_deg"] = p.trades_top;
    b["calm_top_deg"] = p.calm_top;
    b["westerlies_top_deg"] = p.westerlies_top;
    d["bands"] = b;
    d["day_range_km"] = p.day_range_km;
    d["circumference_days"] = p.circumference_days;
    d["calendar"] = calendar_py(p.cal);
    return d;
}

Planet planet_from_py(const nb::dict& d) {
    Planet p;
    p.rotation_period_hr = num(d, "rotation_period_hr");
    p.axial_tilt_deg = num(d, "axial_tilt_deg");
    p.insolation_rel = num(d, "insolation_rel");
    p.radius_km = num(d, "radius_km");
    p.band_scale = d.contains("band_scale") ? num(d, "band_scale") : 1.0;
    nb::dict b = nb::cast<nb::dict>(d["bands"]);
    p.eq_storm_top = num(b, "eq_storm_top_deg");
    p.trades_top = num(b, "trades_top_deg");
    p.calm_top = num(b, "calm_top_deg");
    p.westerlies_top = num(b, "westerlies_top_deg");
    p.day_range_km = num(d, "day_range_km");
    p.circumference_days = num(d, "circumference_days");
    if (d.contains("calendar")) {
        nb::dict c = nb::cast<nb::dict>(d["calendar"]);
        if (nb::len(c) > 0) {
            Almanac& a = p.cal;
            a.present = true;
            a.mode = nb::cast<std::string>(c["mode"]);
            a.seasons = nb::cast<int>(c["seasons"]);
            a.days_per_season_config = num(c, "days_per_season_config");
            a.months_per_season = nb::cast<int>(c["months_per_season"]);
            a.solar_day_hr = num(c, "solar_day_hr");
            a.gravity_rel = num(c, "gravity_rel");
            a.planet_mass_rel_earth = num(c, "planet_mass_rel_earth");
            a.year_days_solar = num(c, "year_days_solar");
            a.days_per_season = num(c, "days_per_season");
            a.orbital_period_earth_days = num(c, "orbital_period_earth_days");
            a.orbital_period_years = num(c, "orbital_period_years");
            a.semi_major_axis_au = num(c, "semi_major_axis_au");
            a.insolation_derived = num(c, "insolation_derived");
            a.insolation_config = num(c, "insolation_config");
            if (c.contains("days_per_season_residual")) {
                a.has_residual = true;
                a.days_per_season_residual = num(c, "days_per_season_residual");
            }
            nb::dict s = nb::cast<nb::dict>(c["star"]);
            a.star.mass_msun = num(s, "mass_msun");
            a.star.luminosity_lsun = num(s, "luminosity_lsun");
            a.star.radius_rsun = num(s, "radius_rsun");
            a.star.teff_k = num(s, "teff_k");
            a.star.spectral_class = nb::cast<std::string>(s["spectral_class"]);
            a.star.apparent_size_rel_sun = num(s, "apparent_size_rel_sun");
            a.tidal_lock_gyr = num(c, "tidal_lock_gyr");
            if (c.contains("seasons_needed_for_no_lock")) a.seasons_needed_for_no_lock = nb::cast<int>(c["seasons_needed_for_no_lock"]);
            if (c.contains("moon")) {
                nb::dict m = nb::cast<nb::dict>(c["moon"]);
                a.has_moon = true;
                a.moon = {num(m, "synodic_month_days"), num(m, "sidereal_month_days"), num(m, "months_per_year"), num(m, "distance_km"),
                          num(m, "distance_planet_radii"), num(m, "tide_rel_earth_same_mass")};
            }
        }
    }
    return p;
}

}  // namespace

void bind_planet(nb::module_& m) {
    nb::class_<Planet>(m, "PlanetParams", "① 行星参数与历法（planet.json）");
    nb::class_<Winds>(m, "Winds", "② 风带与 G（wind.npz + bands.json）");
    nb::class_<Islands>(m, "Islands", "③ 岛群、板块、候选边（islands / plates / cand_edges / density_grid .npz）");
    nb::class_<Climate>(m, "Climate", "④ 局地风与气候（wind_local / band_local / climate_grid / climate_islands .npz）");

    // ---------------------------------------------------------------- 各步
    m.def("planet_stage1", [](nb::handle cfg) {
        Config tmp;
        return stage1(cfg_of(cfg, tmp));
    });
    m.def("planet_stage2", [](nb::handle cfg, const Planet& p) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage2(c, p);
    });
    m.def("planet_stage3", [](nb::handle cfg, uint64_t seed, const Planet& p, const Winds& w) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage3(c, seed, p, w);
    });
    m.def("planet_stage4", [](nb::handle cfg, uint64_t seed, const Planet& p, const Winds& w, const Islands& isl) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage4(c, seed, p, w, isl);
    });
    // upto = 4（默认，P6c 的形）：(P, W, I, C)；5–9（P6d）：再接 (B, R, Ce, D, Pol) 到第 upto 步。skip_diffusion：⑧ 不算（D 为空对象，⑨ 照算）
    m.def(
        "planet_run",
        [](nb::handle cfg, uint64_t seed, int upto, int threads, bool skip_diffusion) {
            Config tmp;
            const Config& c = cfg_of(cfg, tmp);
            World wd;
            Society so;
            {
                nb::gil_scoped_release rel;
                wd = run(c, seed);
                if (upto >= 5) so = run_society(c, seed, wd, upto, skip_diffusion, threads);
            }
            nb::list out;
            out.append(nb::cast(std::move(wd.planet), nb::rv_policy::move));
            out.append(nb::cast(std::move(wd.winds), nb::rv_policy::move));
            out.append(nb::cast(std::move(wd.islands), nb::rv_policy::move));
            out.append(nb::cast(std::move(wd.climate), nb::rv_policy::move));
            if (upto >= 5) out.append(nb::cast(std::move(so.barriers), nb::rv_policy::move));
            if (upto >= 6) out.append(nb::cast(std::move(so.routes), nb::rv_policy::move));
            if (upto >= 7) out.append(nb::cast(std::move(so.centers), nb::rv_policy::move));
            if (upto >= 8) out.append(nb::cast(std::move(so.diffusion), nb::rv_policy::move));
            if (upto >= 9) out.append(nb::cast(std::move(so.polity), nb::rv_policy::move));
            return nb::tuple(out);
        },
        "cfg"_a, "seed"_a, "upto"_a = 4, "threads"_a = 1, "skip_diffusion"_a = false);

    // ---------------------------------------------------------------- 产物的形（前端写 npz / json）
    m.def("planet_json", [](const Planet& p) { return planet_py(p); });
    m.def("winds_arrays", [](const Winds& w) {
        nb::dict d;
        const size_t H = w.ax.nlat, W = w.ax.nlon;
        d["lats"] = arr(w.ax.lats);
        d["lons"] = arr(w.ax.lons);
        d["u"] = arr2(w.u, H, W);
        d["v"] = arr2(w.v, H, W);
        d["band"] = arr2(w.band, H, W);
        d["g_lat"] = w.g.lat;
        d["g_lon"] = w.g.lon;
        d["g_radius_deg"] = w.g.radius_deg;
        d["g_edge"] = w.g.edge;
        return d;
    });
    m.def("islands_arrays", [](const Islands& I) {
        nb::dict d;
        const size_t N = I.n(), H = I.ax.nlat, W = I.ax.nlon;
        d["lat"] = arr(I.lat);
        d["lon"] = arr(I.lon);
        d["xyz"] = arr2(I.xyz, N, 3);
        for (auto [k, v] : std::initializer_list<std::pair<const char*, const std::vector<double>*>>{
                 {"area_km2", &I.area}, {"height_m", &I.height}, {"territory_km2", &I.territory}, {"land_frac", &I.land_frac},
                 {"arable_frac", &I.arable_frac}, {"main_frac", &I.main_frac}, {"main_area_km2", &I.main_area}, {"wall_m", &I.wall},
                 {"age", &I.age}, {"density_at", &I.density_at}, {"mean_nn_days", &I.mean_nn}})
            d[k] = arr(*v);
        d["plate"] = arr(I.plate);
        d["cls"] = arr(I.cls);
        d["layered"] = barr(I.layered);
        d["in_stack"] = barr(I.in_stack);
        d["plates_lats"] = arr(I.ax.lats);
        d["plates_lons"] = arr(I.ax.lons);
        d["plate_id"] = arr2(I.plate_id, H, W);
        d["btype"] = arr2(I.btype, H, W);
        d["boundary_kernel"] = arr2(I.boundary_kernel, H, W);
        d["conv_kernel"] = arr2(I.conv_kernel, H, W);
        d["plate_age"] = arr2(I.plate_age, H, W);
        d["factor"] = arr2(I.factor, H, W);
        d["seeds_xyz"] = arr2(I.seeds_xyz, I.seeds_xyz.size() / 3, 3);
        d["src"] = arr(I.src);
        d["dst"] = arr(I.dst);
        d["dist_days"] = arr(I.dist_days);
        d["kind"] = arr(I.kind);
        d["density"] = arr2(I.density, H, W);
        d["n_exp"] = I.n_exp;
        d["n_fallback"] = I.n_fallback;
        d["n_chord"] = I.n_chord;
        d["f0"] = I.f0;
        return d;
    });
    m.def("climate_arrays", [](const Climate& C) {
        nb::dict d;
        const size_t H = C.ax.nlat, W = C.ax.nlon;
        d["lats"] = arr(C.ax.lats);
        d["lons"] = arr(C.ax.lons);
        for (auto [k, v] : std::initializer_list<std::pair<const char*, const std::vector<double>*>>{
                 {"u", &C.u}, {"v", &C.v}, {"u_bg", &C.u_bg}, {"v_bg", &C.v_bg}, {"v_local", &C.v_local}, {"obstacle", &C.obstacle},
                 {"land", &C.land}, {"wake", &C.wake}, {"lat_eff", &C.lat_eff}, {"precip", &C.precip}, {"temp", &C.temp},
                 {"storm", &C.storm}, {"storm_no_g", &C.storm_no_g}, {"stability", &C.stability}, {"window", &C.window}, {"q", &C.q},
                 {"uplift", &C.uplift}, {"conv", &C.conv}, {"eps", &C.eps}, {"season_range", &C.season_range},
                 {"continentality", &C.continentality}})
            d[k] = arr2(*v, H, W);
        d["band"] = arr2(C.band, H, W);
        d["edges"] = arr2(C.edges, 8, W);
        d["dphi"] = arr2(C.dphi, 8, W);
        nb::dict i;
        for (auto [k, v] : std::initializer_list<std::pair<const char*, const std::vector<double>*>>{
                 {"precip", &C.i_precip}, {"temp", &C.i_temp}, {"storm", &C.i_storm}, {"stability", &C.i_stability},
                 {"window", &C.i_window}, {"catch", &C.i_catch}, {"river_size", &C.i_river_size}, {"temp_sea", &C.i_temp_sea},
                 {"season_range_sea", &C.i_season_range_sea}, {"season_range", &C.i_season_range}, {"temp_winter", &C.i_temp_winter},
                 {"temp_summer", &C.i_temp_summer}})
            i[k] = arr(*v);
        i["has_river"] = barr(C.i_has_river);
        d["islands"] = i;
        d["dt_s"] = C.dt_s;
        d["n_steps"] = C.n_steps;
        return d;
    });

    // ---------------------------------------------------------------- 从产物读回（npz 的数组 dict / json 的 dict）
    m.def("planet_from_json", [](nb::dict planet) { return planet_from_py(planet); });
    m.def("winds_from", [](nb::dict wind, nb::dict bands) {
        Winds w;
        w.ax = axes_from(dv(wind, "lats"), dv(wind, "lons"));
        w.u = dv(wind, "u");
        w.v = dv(wind, "v");
        w.band = anyvec<int16_t>(wind["band"]);
        nb::dict g = nb::cast<nb::dict>(bands["G"]);
        w.g.lat = num(g, "lat");
        w.g.lon = num(g, "lon");
        w.g.radius_deg = num(g, "radius_deg");
        return w;
    });
    m.def("islands_from", [](nb::dict isl, nb::dict plates, nb::dict cand, nb::dict dens) {
        Islands I;
        I.lat = dv(isl, "lat");
        I.lon = dv(isl, "lon");
        I.xyz = dv(isl, "xyz");
        I.area = dv(isl, "area_km2");
        I.height = dv(isl, "height_m");
        I.territory = dv(isl, "territory_km2");
        I.land_frac = dv(isl, "land_frac");
        I.arable_frac = dv(isl, "arable_frac");
        I.main_frac = dv(isl, "main_frac");
        I.main_area = dv(isl, "main_area_km2");
        I.wall = dv(isl, "wall_m");
        I.age = dv(isl, "age");
        I.density_at = dv(isl, "density_at");
        I.mean_nn = dv(isl, "mean_nn_days");
        I.plate = anyvec<int16_t>(isl["plate"]);
        I.cls = anyvec<int8_t>(isl["cls"]);
        I.layered = anyvec<uint8_t>(isl["layered"]);
        I.ax = axes_from(dv(plates, "lats"), dv(plates, "lons"));
        I.plate_id = anyvec<int16_t>(plates["plate_id"]);
        I.btype = anyvec<int8_t>(plates["btype"]);
        I.boundary_kernel = dv(plates, "boundary_kernel");
        I.conv_kernel = dv(plates, "conv_kernel");
        I.plate_age = dv(plates, "age");
        I.factor = dv(plates, "factor");
        I.seeds_xyz = dv(plates, "seeds_xyz");
        I.src = anyvec<int64_t>(cand["src"]);
        I.dst = anyvec<int64_t>(cand["dst"]);
        I.dist_days = dv(cand, "dist_days");
        I.kind = anyvec<int8_t>(cand["kind"]);
        I.density = dv(dens, "density");
        return I;
    });
    m.def("climate_from", [](nb::dict wl, nb::dict bl, nb::dict cg, nb::dict ci) {
        Climate C;
        C.ax = axes_from(dv(wl, "lats"), dv(wl, "lons"));
        C.u = dv(wl, "u");
        C.v = dv(wl, "v");
        C.u_bg = dv(wl, "u_bg");
        C.v_bg = dv(wl, "v_bg");
        C.v_local = dv(wl, "v_local");
        C.obstacle = dv(wl, "obstacle");
        C.land = dv(wl, "land");
        C.wake = dv(wl, "wake");
        C.lat_eff = dv(wl, "lat_eff");
        C.band = anyvec<int16_t>(wl["band"]);
        // band_local 的行按 keys 排回 EDGE_KEYS 的次序
        const std::vector<std::string> keys = nb::cast<std::vector<std::string>>(bl["keys"]);
        const std::vector<double> e = dv(bl, "edges"), dp = dv(bl, "dphi");
        const size_t W = C.ax.nlon;
        C.edges.assign(8 * W, 0.0);
        C.dphi.assign(8 * W, 0.0);
        for (int k = 0; k < 8; ++k) {
            size_t r = keys.size();
            for (size_t t = 0; t < keys.size(); ++t)
                if (keys[t] == EDGE_KEYS[k]) r = t;
            if (r == keys.size()) throw std::invalid_argument("band_local: missing edge key");
            std::memcpy(&C.edges[k * W], &e[r * W], W * sizeof(double));
            std::memcpy(&C.dphi[k * W], &dp[r * W], W * sizeof(double));
        }
        for (auto [k, v] : std::initializer_list<std::pair<const char*, std::vector<double>*>>{
                 {"precip", &C.precip}, {"temp", &C.temp}, {"storm", &C.storm}, {"storm_no_g", &C.storm_no_g}, {"stability", &C.stability},
                 {"window", &C.window}, {"q", &C.q}, {"uplift", &C.uplift}, {"conv", &C.conv}, {"eps", &C.eps},
                 {"season_range", &C.season_range}})
            *v = dv(cg, k);
        if (cg.contains("continentality")) C.continentality = dv(cg, "continentality");
        for (auto [k, v] : std::initializer_list<std::pair<const char*, std::vector<double>*>>{
                 {"precip", &C.i_precip}, {"temp", &C.i_temp}, {"storm", &C.i_storm}, {"stability", &C.i_stability},
                 {"window", &C.i_window}, {"catch", &C.i_catch}, {"river_size", &C.i_river_size}, {"temp_sea", &C.i_temp_sea},
                 {"season_range_sea", &C.i_season_range_sea}, {"season_range", &C.i_season_range}, {"temp_winter", &C.i_temp_winter},
                 {"temp_summer", &C.i_temp_summer}})
            *v = dv(ci, k);
        C.i_has_river = anyvec<uint8_t>(ci["has_river"]);
        return C;
    });

    // ---------------------------------------------------------------- 行星层 → 第三层（不经 npz）
    m.def("planet_view", [](const Planet& p, const Islands& isl, const Climate& c, nb::handle cfg) {
        Config tmp;
        return planet_view(p, isl, c, cfg_of(cfg, tmp));
    });
    m.def("node_inputs", [](const Islands& isl, const Climate& c, int64_t node, uint64_t seed, nb::handle cfg) {
        Config tmp;
        const island::NodeInputs x = node_inputs(isl, c, node, seed, cfg_of(cfg, tmp));
        nb::dict d;
        d["node"] = x.node;
        d["seed"] = x.seed;
        d["lat"] = x.lat;
        d["lon"] = x.lon;
        d["area_km2"] = x.area_km2;
        d["main_area_km2"] = x.main_area_km2;
        d["height_m"] = x.height_m;
        d["age"] = x.age;
        d["layered"] = x.layered;
        d["keel_clearance_m"] = x.keel_clearance_m;
        d["area_median_km2"] = x.area_median_km2;
        d["precip"] = x.precip;
        d["temp_sea"] = x.temp_sea;
        d["lapse_c_per_km"] = x.lapse_c_per_km;
        d["arable_frac"] = x.arable_frac;
        d["river_size"] = x.river_size;
        d["has_river"] = x.has_river;
        d["temp"] = x.temp;
        d["storm"] = x.storm;
        d["window"] = x.window;
        d["season_range"] = x.season_range;
        d["season_range_sea"] = x.season_range_sea;
        d["temp_winter"] = x.temp_winter;
        d["temp_summer"] = x.temp_summer;
        return d;
    });

    // ---------------------------------------------------------------- 公共件（pytest 同输入对照）
    m.def("rfft_lowpass", [](ArrD1 a, int kmax) {
        std::vector<double> v(a.data(), a.data() + a.size());
        std::vector<double> r = rfft_lowpass(v, kmax);
        const size_t n = r.size();
        return to_np(std::move(r), {n});
    });
    m.def("insolation_first_harmonic", [](ArrD1 lat, double tilt, int n) {
        std::vector<double> v(lat.data(), lat.data() + lat.size());
        std::vector<double> r = insolation_first_harmonic(v, tilt, n);
        const size_t k = r.size();
        return to_np(std::move(r), {k});
    });
    m.def("planet_fractal_noise", [](uint64_t seed, uint64_t stream, int nlat, int nlon, int base, int octaves, double persistence,
                                     double lacunarity) {
        Rng r = stage_rng(seed, stream);
        std::vector<double> v = fractal_noise(r, nlat, nlon, base, octaves, persistence, lacunarity);
        return to_np(std::move(v), {static_cast<size_t>(nlat), static_cast<size_t>(nlon)});
    });
    m.def("planet_knn", [](ArrD2 xyz, int k) {
        std::vector<Vec3> p(xyz.shape(0));
        const double* x = xyz.data();
        for (size_t i = 0; i < p.size(); ++i) p[i] = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
        std::vector<int64_t> idx;
        std::vector<double> ang;
        knn(p, k, idx, ang);
        const size_t n = p.size(), kk = n ? idx.size() / n : 0;
        return nb::make_tuple(to_np(std::move(idx), {n, kk}), to_np(std::move(ang), {n, kk}));
    });
    m.def("median_f32", [](nb::handle a) { return median_f32(anyvec<double>(a)); });
}
