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
#include "skyisle/island/climate.hpp"
#include "skyisle/island/generate.hpp"
#include "skyisle/island/landforms.hpp"
#include "skyisle/island/resources.hpp"
#include "skyisle/island/settle.hpp"
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

}  // namespace

Config cfg_from(const nb::dict& d) {
    Config c;
    c.num = nb::cast<std::map<std::string, double>>(d["num"]);
    c.vec = nb::cast<std::map<std::string, std::vector<double>>>(d["vec"]);
    if (d.contains("str")) c.str = nb::cast<std::map<std::string, std::string>>(d["str"]);   // 行星层（P6c）有字符串键
    return c;
}

namespace {

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
    // P6b：气候与聚落用（旧调用方没给就用缺省）
    auto opt = [&](const char* k, double& dst) {
        if (d.contains(k)) dst = dget(d, k);
    };
    opt("temp", x.temp);
    opt("storm", x.storm);
    opt("window", x.window);
    opt("season_range", x.season_range);
    opt("season_range_sea", x.season_range_sea);
    opt("temp_winter", x.temp_winter);
    opt("temp_summer", x.temp_summer);
    opt("people_per_arable_km2", x.people_per_arable_km2);
    opt("precip_mm_ref", x.precip_mm_ref);
    if (d.contains("precip_share") && !d["precip_share"].is_none()) x.precip_share = nb::cast<std::vector<double>>(d["precip_share"]);
    if (d.contains("pop") && !d["pop"].is_none()) x.pop = dget(d, "pop");
    if (d.contains("capital") && !d["capital"].is_none()) {
        nb::dict cap = nb::cast<nb::dict>(d["capital"]);
        x.is_capital = true;
        x.state = nb::cast<int64_t>(cap["state"]);
        x.state_pop = dget(cap, "state_pop");
        x.reformer = nb::cast<bool>(cap["reformer"]);
    }
    // P7：⑥ 的邻边与本群是不是枢纽（market.node_routes / _core.node_routes 的形；没有 ⑥ 给 None）
    if (d.contains("routes") && !d["routes"].is_none()) {
        nb::dict r = nb::cast<nb::dict>(d["routes"]);
        x.hub = nb::cast<bool>(r["hub"]);
        for (nb::handle h : nb::cast<nb::list>(r["edges"])) {
            nb::dict e = nb::cast<nb::dict>(h);
            RouteEdge q;
            q.node = nb::cast<int64_t>(e["node"]);
            q.bearing = dget(e, "bearing");
            q.days = dget(e, "days");
            q.cost_out = dget(e, "cost_out");
            q.cost_in = dget(e, "cost_in");
            q.flow_out = dget(e, "flow_out");
            q.flow_in = dget(e, "flow_in");
            q.hub = nb::cast<bool>(e["hub"]);
            x.routes.push_back(q);
        }
    }
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
    if (d.contains("climate")) {   // ④ 的气候网格、局部带界、行星常数与历法（5.4 四季）
        nb::dict q = nb::cast<nb::dict>(d["climate"]);
        p.cg_grid = llg_from(q);
        p.cg_precip = vget(q, "precip");
        p.cg_storm = vget(q, "storm");
        p.cg_window = vget(q, "window");
        if (q.contains("continentality") && !q["continentality"].is_none()) p.cg_cont = vget(q, "continentality");
        p.band_lons = vget(q, "band_lons");
        p.band_eq_n = vget(q, "band_eq_n");
        p.band_eq_s = vget(q, "band_eq_s");
        p.tilt_deg = dget(q, "tilt_deg");
        p.tau_land = dget(q, "tau_land");
        p.tau_ocean = dget(q, "tau_ocean");
        p.alt_cont = dget(q, "alt_cont");
        if (q.contains("season_shift") && !q["season_shift"].is_none()) p.season_shift = nb::cast<std::vector<double>>(q["season_shift"]);
        if (q.contains("band_shift_k")) p.band_shift_k = dget(q, "band_shift_k");
        nb::dict cal = nb::cast<nb::dict>(q["calendar"]);
        p.cal.seasons = nb::cast<int>(cal["seasons"]);
        p.cal.months_per_season = nb::cast<int>(cal["months_per_season"]);
        p.cal.days_per_month = dget(cal, "days_per_month");
        p.cal.days_per_season = dget(cal, "days_per_season");
        p.cal.year_days = dget(cal, "year_days");
        p.cal.offset = dget(cal, "day_offset_solstice_n");
    }
    return p;
}

// JSON 值 → Python 对象（dict / list / str / int / float / bool / None）
nb::object json_py(const Json& j) {
    switch (j.type()) {
        case Json::NUL: return nb::none();
        case Json::BOOL: return nb::bool_(j.as_bool());
        case Json::INT: return nb::int_(j.as_int());
        case Json::NUM: return nb::float_(j.as_num());
        case Json::STR: return nb::str(j.as_str().c_str());
        case Json::ARR: {
            nb::list l;
            for (const Json& x : j.items()) l.append(json_py(x));
            return l;
        }
        case Json::OBJ: {
            nb::dict d;
            for (const auto& kv : j.fields()) d[kv.first.c_str()] = json_py(kv.second);
            return d;
        }
    }
    return nb::none();
}

template <class T>
nb::ndarray<nb::numpy, T> vec_np(const std::vector<T>& v) {
    return to_np(std::vector<T>(v), {v.size()});
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
        e["float"] = r.fl;
        e["age"] = r.age;
        e["kind"] = age_name(r.kind);
        e["bbox"] = nb::make_tuple(r.r0, r.c0, r.m, r.m);
        // 多核嵌合（P4）：局部栅格 km 的原始数，前端（terrain.cores_json）加 gc 换成群坐标、四舍五入
        nb::list cl;
        for (const CoreRec& q : r.cores) {
            nb::dict x;
            x["seed"] = nb::make_tuple(q.seed_x, q.seed_y);
            x["strength"] = q.strength;
            x["cells"] = q.cells;
            x["peak"] = q.peak;
            x["load"] = q.load;
            x["load_xy"] = nb::make_tuple(q.load_x, q.load_y);
            x["mean_above"] = q.mean_above;
            cl.append(x);
        }
        e["cores"] = cl;
        e["gc"] = nb::make_tuple(r.gcx, r.gcy);
        if (r.strat.on) {   // B2 岩层（拟合前的米；scale 换成最终高程）
            nb::dict st;
            st["t_cap"] = r.strat.t_cap;
            st["t_sed"] = r.strat.t_sed;
            st["t_gab"] = r.strat.t_gab;
            st["bed_lime"] = r.strat.bed_lime;
            st["bed_marl"] = r.strat.bed_marl;
            st["bed_phase"] = r.strat.bed_phase;
            st["exhume"] = r.strat.exhume;
            st["scale"] = r.strat.scale;
            e["strat"] = st;
        }
        isl.append(e);
    }
    d["islands"] = isl;
    if (!g.coast_dist.v.empty()) d["coast_dist_m"] = f32_np(g.coast_dist);
    // B2 层面（地形与水系分两次调时，build_hydro 的 state 原样带回去：谷坡角、lith 都读它）
    if (!g.strat_top.v.empty()) {
        d["strat_top"] = grid_np(GridD(g.strat_top));
        d["skel_top"] = grid_np(GridD(g.skel_top));
    }
    nb::list lfl;
    for (size_t q = 0; q < g.landforms.size(); ++q) lfl.append(json_py(landform_json(g, g.landforms[q], static_cast<int>(q))));
    d["landforms"] = lfl;
    nb::list lk;
    for (const Link& e : g.links) {
        nb::dict x;
        x["a"] = e.a;
        x["b"] = e.b;
        x["gap"] = e.gap;
        x["dh"] = e.dh;
        x["fallback"] = e.fallback;
        lk.append(x);
    }
    d["links"] = lk;
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
    if (!g.cultivable.v.empty()) d["cultivable"] = grid_np(Grid<uint8_t>(g.cultivable));
    d["river_width_m"] = f32_np(g.width_m);
    d["river_depth_m"] = f32_np(g.depth_m);
    d["cut_m"] = f32_np(g.cut_m);
    d["slope_deg"] = f32_np(g.slope);
    d["rain_mm"] = f32_np(g.rain);
    d["runoff_mm"] = f32_np(g.runoff);
    d["runoff_ratio"] = g.runoff_ratio;
    if (!g.lith.v.empty() && !g.strat_top.v.empty()) d["lith"] = grid_np(Grid<uint8_t>(g.lith));
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
        e["cap_ran"] = r.cap_ran;
        e["captures"] = r.captures;
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

nb::dict resources_dict(Group& g) {
    nb::dict d;
    const size_t H = g.H, W = g.W;
    if (!g.rockwall_m.v.empty()) {   // B3 崖层
        d["rockwall_m"] = f32_np(g.rockwall_m);
        d["rockwall_dir"] = grid_np(Grid<uint8_t>(g.rockwall_dir));
    }
    nb::list lfl;
    for (size_t q = 0; q < g.landforms.size(); ++q) lfl.append(json_py(landform_json(g, g.landforms[q], static_cast<int>(q))));
    d["landforms"] = lfl;
    d["terrain_zone"] = grid_np(Grid<uint8_t>(g.zone));
    d["patch_id"] = grid_np(GridI(g.patch_id));
    d["resource"] = grid_np(Grid<uint8_t>(g.resource));
    std::vector<uint8_t> rf(static_cast<size_t>(FK_COUNT) * H * W);
    for (int f = 0; f < FK_COUNT; ++f) std::copy(g.res_field[f].v.begin(), g.res_field[f].v.end(), rf.begin() + static_cast<size_t>(f) * H * W);
    d["res_field"] = to_np(std::move(rf), {static_cast<size_t>(FK_COUNT), H, W});
    nb::list occ_lab;
    for (int f = 0; f < FK_COUNT; ++f) occ_lab.append(grid_np(GridI(g.occ_lab[f])));
    d["occ_lab"] = occ_lab;
    nb::list deps, occs, works, cells;
    for (const Deposit& x : g.res.deposits) deps.append(json_py(deposit_json(x)));
    for (const Occurrence& x : g.res.occ) {
        occs.append(json_py(occurrence_json(x)));
        cells.append(vec_np(x.cells));
    }
    for (const Working& x : g.res.works) works.append(json_py(working_json(x)));
    d["deposits"] = deps;
    d["occurrences"] = occs;
    d["workings"] = works;
    d["occ_cells"] = cells;
    d["geo_ore"] = g.res.geo_ore;
    d["fs_rate"] = g.res.fs_rate;
    d["kernel"] = g.res.kernel_j;
    d["r_cells"] = g.res.r_cells;
    nb::list thr;
    for (double t : g.res.thr) thr.append(t);
    d["thr"] = thr;
    d["seconds"] = g.sec_resources;
    return d;
}

nb::dict daily_dict(const Daily& D) {
    nb::dict d;
    d["day"] = vec_np(D.day);
    d["temp_c"] = vec_np(D.temp_c);
    d["season"] = vec_np(D.season);
    d["precip_rel"] = vec_np(D.precip_rel);
    d["precip_mm"] = vec_np(D.precip_mm);
    d["storm"] = vec_np(D.storm);
    d["window"] = vec_np(D.window);
    d["wind_u"] = vec_np(D.wind_u);
    d["wind_v"] = vec_np(D.wind_v);
    return d;
}

nb::dict weather_dict(const WeatherYear& Y, const std::vector<SeasonParams>& P) {
    nb::dict d;
    d["day"] = vec_np(Y.day);
    d["season"] = vec_np(Y.season);
    d["month"] = vec_np(Y.month);
    d["day_of_month"] = vec_np(Y.day_of_month);
    d["type"] = vec_np(Y.type);
    d["precip_mm"] = vec_np(Y.precip_mm);
    d["temp_c"] = vec_np(Y.temp_c);
    d["wind_from_deg"] = vec_np(Y.wind_from_deg);
    d["wind_ms"] = vec_np(Y.wind_ms);
    d["sailable"] = bool_np(std::vector<uint8_t>(Y.sailable), {Y.sailable.size()});
    d["storm_event"] = vec_np(Y.storm_event);
    d["wet"] = bool_np(std::vector<uint8_t>(Y.wet), {Y.wet.size()});
    d["temp_rim_c"] = vec_np(Y.temp_rim_c);
    d["snow"] = bool_np(std::vector<uint8_t>(Y.snow), {Y.snow.size()});
    nb::list pl;
    for (const SeasonParams& p : P) {
        nb::dict e;
        e["f_wet"] = p.f_wet;
        e["f_rain_days"] = p.f_rain_days;
        e["p_ww"] = p.p_ww;
        e["p_dw"] = p.p_dw;
        e["mean_wet_mm"] = p.mean_wet_mm;
        e["storm_frac"] = p.storm_frac;
        e["p_calm"] = p.p_calm;
        e["wind_u"] = p.wind_u;
        e["wind_v"] = p.wind_v;
        e["speed"] = p.speed;
        pl.append(e);
    }
    d["params"] = pl;
    return d;
}

nb::dict lod_dict(LodBlock&& B) {
    nb::dict d;
    const size_t H = B.H, W = B.W;
    d["land"] = to_np(std::move(B.land), {H, W});
    d["height"] = to_np(std::move(B.height), {H, W});
    d["peak"] = to_np(std::move(B.peak), {H, W});
    d["island"] = to_np(std::move(B.island), {H, W});
    d["landcover"] = to_np(std::move(B.landcover), {H, W});
    d["water"] = to_np(std::move(B.water), {H, W});
    return d;
}

using ArrU2 = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// planet / cfg 参数：make_planet / make_config 转好的对象直接用，dict 就当场转
const PlanetView& pv_of(nb::handle h, PlanetView& tmp) {
    if (nb::isinstance<PlanetView>(h)) return nb::cast<const PlanetView&>(h);
    tmp = planet_from(nb::cast<nb::dict>(h));
    return tmp;
}
}  // namespace

const Config& cfg_of(nb::handle h, Config& tmp) {
    if (nb::isinstance<Config>(h)) return nb::cast<const Config&>(h);
    tmp = cfg_from(nb::cast<nb::dict>(h));
    return tmp;
}

void bind_island(nb::module_& m) {
    nb::class_<PlanetView>(m, "Planet", "行星层的网格与全体群（make_planet 转好，按 run 缓存）");
    nb::class_<Config>(m, "Config", "展平的配置：[island] 段或行星层的 shared / skeleton / s01–s04（make_config 转好）");
    m.def("make_planet", [](nb::dict planet) { return planet_from(planet); });
    m.def("make_config", [](nb::dict cfg) { return cfg_from(cfg); });

    // ---------------------------------------------------------------- 整群生成（P6b）：地形 → 水系 → 资源 → 四季 → 天气 → 聚落
    m.def("generate", [](nb::dict inp, nb::handle planet, nb::handle cfg, int year, int steps, double res_m, int threads) {
        const NodeInputs ni = inp_from(inp);
        PlanetView pv_tmp;
        Config c_tmp;
        const PlanetView& pv = pv_of(planet, pv_tmp);
        const Config& c = cfg_of(cfg, c_tmp);
        Group g;
        {
            nb::gil_scoped_release rel;
            g = generate(ni, pv, c, year, steps, res_m, threads);
        }
        nb::dict d;
        d["terrain"] = terrain_dict(g);
        if (g.has_hydro) d["hydro"] = hydro_dict(g);
        if (g.has_resources) d["resources"] = resources_dict(g);
        if (g.has_climate) {
            d["climate"] = json_py(climate_json(g.clim));
            d["daily"] = daily_dict(g.daily);
        }
        if (g.has_weather) d["weather"] = weather_dict(g.weather, g.wparams);
        if (g.has_settle) {
            d["settle"] = json_py(g.settle);
            d["settle_raster"] = grid_np(Grid<uint8_t>(g.settle_raster));
            d["settle_fields"] = grid_np(GridI(g.settle_fields));
            d["settle_cultivated"] = grid_np(Grid<uint8_t>(g.cultivated));
            d["settle_fallow"] = grid_np(Grid<uint8_t>(g.fallow_years));
            d["settle_polder"] = grid_np(GridI(g.polder_id));
            d["settle_landcover_natural"] = grid_np(Grid<uint8_t>(g.landcover_natural));   // P6b：没有人以前的地表、人工改造
            d["settle_landuse"] = grid_np(Grid<uint8_t>(g.landuse));
            d["settle_pop"] = g.settle_pop;
        }
        d["seconds"] = nb::make_tuple(g.sec_total, g.sec_hydro, g.sec_resources, g.sec_climate, g.sec_settle);
        return d;
    }, "inp"_a, "planet"_a, "cfg"_a, "year"_a = 0, "steps"_a = 5, "res_m"_a = 0.0, "threads"_a = 1);

    // 粗版的块降采样（lod._block_reduce）：输入群栅格的 island_id / height / landcover / river / lake
    m.def("block_reduce", [](ArrS2 island_id, ArrD2 height, ArrU2 landcover, ArrU2 river, ArrB2 lake, int f) {
        Group g;
        g.H = static_cast<int>(island_id.shape(0));
        g.W = static_cast<int>(island_id.shape(1));
        g.island_id = Grid<int16_t>(g.H, g.W);
        std::memcpy(g.island_id.v.data(), island_id.data(), g.island_id.v.size() * sizeof(int16_t));
        g.height = grid_from(height);
        g.landcover = Grid<uint8_t>(g.H, g.W);
        std::memcpy(g.landcover.v.data(), landcover.data(), g.landcover.v.size());
        g.river = Grid<uint8_t>(g.H, g.W);
        std::memcpy(g.river.v.data(), river.data(), g.river.v.size());
        g.lake = mask_from(lake);
        LodBlock B;
        {
            nb::gil_scoped_release rel;
            B = block_reduce(g, f);
        }
        return lod_dict(std::move(B));
    });

    // 只算四季（island stats 全量季型用）与多年逐日天气（IS-daily）
    m.def("climate_only", [](nb::dict inp, nb::handle planet, nb::handle cfg) {
        const NodeInputs ni = inp_from(inp);
        PlanetView pv_tmp;
        Config c_tmp;
        const PlanetView& pv = pv_of(planet, pv_tmp);
        const Config& c = cfg_of(cfg, c_tmp);
        return json_py(climate_json(build_climate(ni, pv, c)));
    });
    m.def("weather_years", [](nb::dict inp, nb::handle planet, nb::handle cfg, double rim_m, int years) {
        const NodeInputs ni = inp_from(inp);
        PlanetView pv_tmp;
        Config c_tmp;
        const PlanetView& pv = pv_of(planet, pv_tmp);
        const Config& c = cfg_of(cfg, c_tmp);
        std::vector<double> P, F;
        std::vector<SeasonParams> par;
        {
            nb::gil_scoped_release rel;
            const Climate clim = build_climate(ni, pv, c);
            const Daily daily = daily_curves(clim, ni);
            par = season_params(clim, c);
            multi_year(ni, clim, daily, par, rim_m, c, years, P, F);
        }
        const size_t ns = par.size();
        std::vector<double> frd;
        for (const SeasonParams& p : par) frd.push_back(p.f_rain_days);
        return nb::make_tuple(to_np(std::move(P), {static_cast<size_t>(years), ns}), to_np(std::move(F), {static_cast<size_t>(years), ns}),
                              to_np(std::move(frd), {ns}));
    });
    // 一年的四季 + 逐日曲线 + 逐日天气（island lod --weather：粗版顺带出天气）：与 generate 的第 3、4 步同式、同随机流（weather:{year}）；
    // rim_m 是水系之后主岛的岸缘（island.json 的 rim_m，= generate 里的 islands[0].rim_j）
    m.def("weather_year", [](nb::dict inp, nb::handle planet, nb::handle cfg, double rim_m, int year) {
        const NodeInputs ni = inp_from(inp);
        PlanetView pv_tmp;
        Config c_tmp;
        const PlanetView& pv = pv_of(planet, pv_tmp);
        const Config& c = cfg_of(cfg, c_tmp);
        Climate clim;
        Daily daily;
        std::vector<SeasonParams> par;
        WeatherYear Y;
        {
            nb::gil_scoped_release rel;
            clim = build_climate(ni, pv, c);
            daily = daily_curves(clim, ni);
            par = season_params(clim, c);
            Rng r = part_rng(ni, "weather:" + std::to_string(year));
            Y = simulate_year(r, clim, daily, par, ni.height_m, c, rim_m, ni.lapse_c_per_km);
        }
        nb::dict d;
        d["climate"] = json_py(climate_json(clim));
        d["daily"] = daily_dict(daily);
        d["weather"] = weather_dict(Y, par);
        return d;
    }, "inp"_a, "planet"_a, "cfg"_a, "rim_m"_a, "year"_a = 0);
    m.def("kmeans_split", [](uint64_t seed, const std::string& key, std::vector<int32_t> ii, std::vector<int32_t> jj, int k) {
        Rng r = entity_rng(seed, ISLAND_STREAM, key);
        return vec_np(kmeans_split(r, ii, jj, k));
    });

    m.def("build_terrain", [](nb::dict inp, nb::handle planet, nb::handle cfg, double res_m, int threads) {
        const NodeInputs ni = inp_from(inp);
        PlanetView pv_tmp;
        Config c_tmp;
        const PlanetView& pv = pv_of(planet, pv_tmp);
        const Config& c = cfg_of(cfg, c_tmp);
        Group g;
        {
            nb::gil_scoped_release rel;
            g = build_terrain(ni, pv, c, res_m, threads);
        }
        return terrain_dict(g);
    }, "inp"_a, "planet"_a, "cfg"_a, "res_m"_a = 0.0, "threads"_a = 1);

    m.def("build_hydro", [](nb::dict state, nb::handle planet, nb::handle cfg, int threads) {
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
            // 谷收拢（P4）看岛龄档与是不是多核岛：新岛 × capture_young、多核岛不收（多核的核在这里只要「有没有」）
            if (e.contains("young") && nb::cast<bool>(e["young"])) r.kind = YOUNG;
            if (e.contains("multicore") && nb::cast<bool>(e["multicore"])) r.cores.resize(1);
            if (e.contains("strat") && !e["strat"].is_none()) {   // B2 岩层（build_terrain 交出来的原数）
                nb::dict st = nb::cast<nb::dict>(e["strat"]);
                r.strat.on = true;
                r.strat.t_cap = dget(st, "t_cap");
                r.strat.t_sed = dget(st, "t_sed");
                r.strat.t_gab = dget(st, "t_gab");
                r.strat.bed_lime = dget(st, "bed_lime");
                r.strat.bed_marl = dget(st, "bed_marl");
                r.strat.bed_phase = dget(st, "bed_phase");
                r.strat.exhume = dget(st, "exhume");
                r.strat.scale = dget(st, "scale");
            }
            g.islands.push_back(r);
        }
        if (state.contains("strat_top") && !state["strat_top"].is_none()) {
            g.strat_top = grid_from(nb::cast<ArrD2>(state["strat_top"]));
            g.skel_top = grid_from(nb::cast<ArrD2>(state["skel_top"]));
        }
        PlanetView pv_tmp;
        Config c_tmp;
        const PlanetView& pv = pv_of(planet, pv_tmp);
        const Config& c = cfg_of(cfg, c_tmp);
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
    // 多核岛（P4）：cores_key 是这座岛的核随机流（island:<节点>:cores:<岛号>），kernel / btype 是板块边界；返回同 sculpt_island 外加各核的原始数
    m.def("sculpt_island_cores", [](uint64_t seed, const std::string& shape_key, const std::string& key, const std::string& cores_key, double area,
                                    double res_km, double elong, double theta, double age, double surface, double relief, double rim_min, bool is_main,
                                    double kernel, int btype, nb::dict cfg) {
        const Config c = cfg_from(cfg);
        Rng rs = entity_rng(seed, ISLAND_STREAM, shape_key);
        Shape s = island_shape(rs, area, res_km, elong, theta, c);
        Rng rc = entity_rng(seed, ISLAND_STREAM, cores_key);
        CoreSpec sp = multicore_spec(rc, area, age, kernel, btype, c);
        Rng rt = entity_rng(seed, ISLAND_STREAM, key);
        Sculpt sc = sculpt_island(rt, s, age, area, res_km, surface, relief, rim_min, is_main, c, sp.on ? &sp : nullptr);
        nb::list cl;
        for (const CoreRec& q : sc.cores)
            cl.append(nb::make_tuple(q.seed_x, q.seed_y, q.strength, q.cells, q.peak, q.load, q.load_x, q.load_y, q.mean_above));
        return nb::make_tuple(grid_np(std::move(sc.h)), age_name(sc.kind), sc.rim, sc.peak, cl);
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
