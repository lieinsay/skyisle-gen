// 第三层（岛群生成器）的绑定：build_terrain / build_hydro 与岛形、剖面、势力范围等公共件。
// 输入：inp（本群标量 dict）、planet（行星层网格 dict）、cfg（{"num": {键: 数}, "vec": {键: [数]}}，前端把 [island] 段展平）。
// 输出：原始的双精度数与数组；island.json 的拼装（键序、round 位数）在前端 skyisle_gen/island/engine.py。
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bind_util.hpp"
#include "skyisle/island/build.hpp"
#include "skyisle/island/layout.hpp"
#include "skyisle/island/terrain.hpp"
#include "skyisle/island/territory.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace skyisle;
using namespace skyisle::island;

namespace {

using ArrAny = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using ArrAnyI = nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu>;

double dget(const nb::dict& d, const char* k) { return nb::cast<double>(d[k]); }
std::vector<double> vget(const nb::dict& d, const char* k) {
    ArrAny a = nb::cast<ArrAny>(d[k]);
    return std::vector<double>(a.data(), a.data() + a.size());
}
std::vector<int32_t> iget(const nb::dict& d, const char* k) {
    ArrAnyI a = nb::cast<ArrAnyI>(d[k]);
    return std::vector<int32_t>(a.data(), a.data() + a.size());
}

Config cfg_from(const nb::dict& d) {
    Config c;
    c.num = nb::cast<std::map<std::string, double>>(d["num"]);
    c.vec = nb::cast<std::map<std::string, std::vector<double>>>(d["vec"]);
    return c;
}

NodeInputs inp_from(const nb::dict& d) {
    NodeInputs x;
    x.node = nb::cast<int64_t>(d["node"]);
    x.seed = nb::cast<uint64_t>(d["seed"]);
    x.lat = dget(d, "lat");
    x.lon = dget(d, "lon");
    x.area_km2 = dget(d, "area_km2");
    x.main_area_km2 = dget(d, "main_area_km2");
    x.height_m = dget(d, "height_m");
    x.age = dget(d, "age");
    x.layered = nb::cast<bool>(d["layered"]);
    x.keel_clearance_m = dget(d, "keel_clearance_m");
    x.area_median_km2 = dget(d, "area_median_km2");
    x.precip = dget(d, "precip");
    x.temp_sea = dget(d, "temp_sea");
    x.lapse_c_per_km = dget(d, "lapse_c_per_km");
    x.arable_frac = dget(d, "arable_frac");
    x.river_size = dget(d, "river_size");
    x.has_river = nb::cast<bool>(d["has_river"]);
    return x;
}

LatLonGrid llg_from(const nb::dict& d) {
    LatLonGrid g;
    g.lat0 = dget(d, "lat0");
    g.dlat = dget(d, "dlat");
    g.lon0 = dget(d, "lon0");
    g.dlon = dget(d, "dlon");
    g.nlat = nb::cast<int>(d["nlat"]);
    g.nlon = nb::cast<int>(d["nlon"]);
    return g;
}

PlanetView planet_from(const nb::dict& d) {
    PlanetView p;
    p.radius_km = dget(d, "radius_km");
    p.year_s = dget(d, "year_s");
    if (d.contains("plates")) {
        nb::dict q = nb::cast<nb::dict>(d["plates"]);
        p.plate_grid = llg_from(q);
        p.plate_K = vget(q, "K");
        p.plate_btype = iget(q, "btype");
        p.plate_lats = vget(q, "lats");
        p.plate_lons = vget(q, "lons");
    }
    if (d.contains("islands")) {
        nb::dict q = nb::cast<nb::dict>(d["islands"]);
        p.isl_lat = vget(q, "lat");
        p.isl_lon = vget(q, "lon");
        p.isl_area = vget(q, "area");
    }
    if (d.contains("wind")) {
        nb::dict q = nb::cast<nb::dict>(d["wind"]);
        p.wind_grid = llg_from(q);
        p.wind_u = vget(q, "u");
        p.wind_v = vget(q, "v");
    }
    return p;
}

template <class T>
nb::ndarray<nb::numpy, float> f32_np(const Grid<T>& g) {
    std::vector<float> v(g.v.size());
    for (size_t k = 0; k < v.size(); ++k) v[k] = static_cast<float>(g.v[k]);
    return to_np(std::move(v), {static_cast<size_t>(g.H), static_cast<size_t>(g.W)});
}

nb::dict terrain_dict(Group& g) {
    nb::dict d;
    d["H"] = g.H;
    d["W"] = g.W;
    d["n"] = g.n;
    d["res_km"] = g.res_km;
    d["x0"] = g.x0;
    d["y0"] = g.y0;
    d["origin_x"] = g.origin_x;
    d["origin_y"] = g.origin_y;
    d["axis"] = g.axis;
    d["kernel"] = g.kernel;
    d["btype"] = g.btype;
    d["node_kind"] = age_name(g.node_kind);
    d["height"] = grid_np(GridD(g.height));
    d["island_id"] = grid_np(Grid<int16_t>(g.island_id));
    d["cliff"] = mask_np(Mask(g.cliff));
    d["rims"] = to_np(std::vector<double>(g.rims), {g.rims.size()});
    nb::list isl;
    for (const IslandRec& r : g.islands) {
        nb::dict e;
        e["id"] = r.id;
        e["area_cells"] = r.area_cells;
        e["area_target"] = r.area_target;
        e["cx"] = r.cx;
        e["cy"] = r.cy;
        e["surface"] = r.surface;
        e["relief_target"] = r.relief_target;
        e["rim"] = r.rim;
        e["peak"] = r.peak;
        e["keel"] = r.keel;
        e["age"] = r.age;
        e["kind"] = age_name(r.kind);
        e["bbox"] = nb::make_tuple(r.r0, r.c0, r.m, r.m);
        isl.append(e);
    }
    d["islands"] = isl;
    nb::list lk;
    for (const Link& e : g.links) {
        nb::dict x;
        x["a"] = e.a;
        x["b"] = e.b;
        x["gap"] = e.gap;
        x["dh"] = e.dh;
        x["bridge"] = e.bridge;
        x["fallback"] = e.fallback;
        lk.append(x);
    }
    d["links"] = lk;
    nb::list tr;
    for (const auto& p : g.tree) tr.append(nb::make_tuple(p.first, p.second));
    d["tree"] = tr;
    nb::list mp;
    for (MaskPos& m : g.masks_pos) mp.append(nb::make_tuple(mask_np(Mask(m.mask)), m.r0, m.c0));
    d["masks_pos"] = mp;
    nb::dict t;
    const TerritoryRec& T = g.territory;
    t["neighbours"] = T.neighbours;
    t["gap_km"] = T.gap_km;
    t["constrained"] = T.constrained;
    t["off_x"] = T.off_x;
    t["off_y"] = T.off_y;
    t["turn_deg"] = T.turn_deg;
    t["stretch"] = T.stretch;
    t["before"] = T.before;
    t["after"] = T.after;
    t["has_violation"] = T.has_violation;
    t["violation"] = T.violation;
    d["territory"] = t;
    d["seconds"] = nb::make_tuple(g.sec_layout, g.sec_paste, g.sec_total);
    return d;
}

nb::dict hydro_dict(Group& g) {
    nb::dict d;
    d["height"] = grid_np(GridD(g.height));
    d["filled"] = grid_np(GridD(g.filled));
    d["route_h"] = grid_np(GridD(g.route_h));
    d["flowacc_km2"] = f32_np(g.acc_km2);
    d["river"] = grid_np(Grid<uint8_t>(g.river));
    d["stream"] = grid_np(Grid<uint8_t>(g.stream));
    d["lake"] = mask_np(Mask(g.lake));
    d["floodplain"] = mask_np(Mask(g.floodplain));
    d["landcover"] = grid_np(Grid<uint8_t>(g.landcover));
    d["arable"] = grid_np(Grid<uint8_t>(g.arable));
    d["river_width_m"] = f32_np(g.width_m);
    d["river_depth_m"] = f32_np(g.depth_m);
    d["cut_m"] = f32_np(g.cut_m);
    d["slope_deg"] = f32_np(g.slope);
    d["recv_i"] = grid_np(GridI(g.recv_i));
    d["recv_j"] = grid_np(GridI(g.recv_j));
    d["P_mm"] = g.P_mm;
    d["river_thr"] = g.river_thr;
    d["dz"] = g.dz;
    d["wind"] = nb::make_tuple(g.wind_u, g.wind_v);
    d["n_falls"] = g.n_falls;
    d["max_cut"] = g.max_cut;
    d["seconds"] = g.sec_hydro;
    nb::list isl;
    for (const IslandRec& r : g.islands) {
        if (!r.hydro) {
            isl.append(nb::none());
            continue;
        }
        nb::dict e;
        e["n_lakes"] = r.n_lakes;
        e["lake_cells"] = r.lake_cells;
        e["max_flowacc"] = r.max_flowacc;
        e["has_perennial"] = r.has_perennial;
        e["has_stream"] = r.has_stream;
        isl.append(e);
    }
    d["islands"] = isl;
    nb::list rv;
    for (const RiverRec& r : g.rivers) {
        nb::dict e;
        e["mouth"] = nb::make_tuple(r.mouth_r, r.mouth_c);
        e["basin_km2"] = r.basin_km2;
        e["length_km"] = r.length_km;
        e["discharge"] = r.discharge;
        e["width"] = r.width;
        e["depth"] = r.depth;
        e["level"] = r.level;
        e["waterfall"] = r.waterfall;
        e["incision"] = r.incision;
        rv.append(e);
    }
    d["rivers"] = rv;
    nb::dict b;
    b["present"] = g.basins.present;
    b["total_km2"] = g.basins.total_km2;
    b["thr_km2"] = g.basins.thr_km2;
    nb::list bm;
    for (const auto& m : g.basins.mouths) bm.append(nb::make_tuple(static_cast<int>(m[0]), static_cast<int>(m[1]), m[2]));
    b["mouths"] = bm;
    d["basins"] = b;
    nb::list ln;
    for (const RiverLine& L : g.lines) {
        std::vector<double> pts(L.pts.size() * 5);
        for (size_t q = 0; q < L.pts.size(); ++q) {
            pts[5 * q] = L.pts[q].r;
            pts[5 * q + 1] = L.pts[q].c;
            pts[5 * q + 2] = L.pts[q].w;
            pts[5 * q + 3] = L.pts[q].lvl;
            pts[5 * q + 4] = L.pts[q].acc;
        }
        ln.append(nb::make_tuple(L.island, to_np(std::move(pts), {L.pts.size(), static_cast<size_t>(5)})));
    }
    d["lines"] = ln;
    return d;
}

}  // namespace

void bind_island(nb::module_& m) {
    m.def("build_terrain", [](nb::dict inp, nb::dict planet, nb::dict cfg, double res_m, int threads) {
        const NodeInputs ni = inp_from(inp);
        const PlanetView pv = planet_from(planet);
        const Config c = cfg_from(cfg);
        Group g;
        {
            nb::gil_scoped_release rel;
            g = build_terrain(ni, pv, c, res_m, threads);
        }
        return terrain_dict(g);
    }, "inp"_a, "planet"_a, "cfg"_a, "res_m"_a = 0.0, "threads"_a = 1);

    m.def("build_hydro", [](nb::dict state, nb::dict planet, nb::dict cfg, int threads) {
        Group g;
        g.inp = inp_from(nb::cast<nb::dict>(state["inp"]));
        const ArrD2 h = nb::cast<ArrD2>(state["height"]);
        g.H = static_cast<int>(h.shape(0));
        g.W = static_cast<int>(h.shape(1));
        g.height = grid_from(h);
        const ArrS2 ii = nb::cast<ArrS2>(state["island_id"]);
        g.island_id = Grid<int16_t>(g.H, g.W);
        std::memcpy(g.island_id.v.data(), ii.data(), g.island_id.v.size() * sizeof(int16_t));
        g.cliff = mask_from(nb::cast<ArrB2>(state["cliff"]));
        g.res_km = dget(state, "res_km");
        g.origin_x = dget(state, "origin_x");
        g.origin_y = dget(state, "origin_y");
        nb::list isl = nb::cast<nb::list>(state["islands"]);
        for (size_t k = 0; k < isl.size(); ++k) {
            nb::dict e = nb::cast<nb::dict>(isl[k]);
            IslandRec r;
            r.id = static_cast<int>(k);
            r.rim_j = dget(e, "rim_m");
            r.keel_j = dget(e, "keel_m");
            r.age_j = dget(e, "age");
            g.islands.push_back(r);
        }
        const PlanetView pv = planet_from(planet);
        const Config c = cfg_from(cfg);
        {
            nb::gil_scoped_release rel;
            build_hydro(g, pv, c, threads);
        }
        return hydro_dict(g);
    }, "state"_a, "planet"_a, "cfg"_a, "threads"_a = 1);

    // ---------------------------------------------------------------- 公共件（测试对照用）
    m.def("island_shape", [](uint64_t seed, const std::string& key, double area, double res_km, double elong, double theta, nb::dict cfg) {
        Rng r = entity_rng(seed, ISLAND_STREAM, key);
        Shape s = island_shape(r, area, res_km, elong, theta, cfg_from(cfg));
        const size_t n = s.xs.size();
        return nb::make_tuple(mask_np(std::move(s.mask)), grid_np(std::move(s.inside)), to_np(std::move(s.xs), {n}));
    });
    m.def("radial_profile", [](ArrB2 mask, double res_km) {
        std::array<double, 2> ctr;
        std::vector<double> prof;
        radial_profile(mask_from(mask), res_km, ctr, prof);
        const size_t n = prof.size();
        return nb::make_tuple(nb::make_tuple(ctr[0], ctr[1]), to_np(std::move(prof), {n}));
    });
    m.def("sculpt_island", [](uint64_t seed, const std::string& shape_key, const std::string& key, double area, double res_km, double elong,
                              double theta, double age, double surface, double relief, double rim_min, bool is_main, nb::dict cfg) {
        const Config c = cfg_from(cfg);
        Rng rs = entity_rng(seed, ISLAND_STREAM, shape_key);
        Shape s = island_shape(rs, area, res_km, elong, theta, c);
        Rng rt = entity_rng(seed, ISLAND_STREAM, key);
        Sculpt sc = sculpt_island(rt, s, age, area, res_km, surface, relief, rim_min, is_main, c);
        return nb::make_tuple(grid_np(std::move(sc.h)), age_name(sc.kind), sc.rim, sc.peak);
    });
    m.def("territory_limits", [](nb::dict planet, int64_t node, double gap_km, double reach, double reach_km) {
        const PlanetView pv = planet_from(planet);
        nb::list out;
        for (const Limit& L : limits(pv, node, gap_km, reach, reach_km)) {
            nb::dict e;
            e["node"] = L.node;
            e["dist_km"] = L.dist_km;
            e["u"] = nb::make_tuple(L.ux, L.uy);
            e["limit_km"] = L.limit_km;
            out.append(e);
        }
        return out;
    });
    m.def("nearest_fit", [](std::vector<double> support, std::vector<std::vector<double>> lims) {   // lims：[(ux, uy, limit_km)]
        std::vector<Limit> lim;
        for (const auto& L : lims) lim.push_back({0, 0.0, L[0], L[1], L[2]});
        double ox, oy;
        const double v = nearest_fit(support, lim, ox, oy);
        return nb::make_tuple(ox, oy, v);
    });
    m.def("boundary_axis", [](nb::dict planet, double lat, double lon) {
        const PlanetView pv = planet_from(planet);
        double axis, kernel;
        int btype;
        boundary_axis(pv, lat, lon, axis, kernel, btype);
        return nb::make_tuple(axis, kernel, btype);
    });
}
