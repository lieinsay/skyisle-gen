// 聚落营建器（docs/PLAN-TOWN.md）的绑定：场地地形（岛群窗口细化 / 合成地形 / 外部高程图）与营建方案（town_plan）。
// 数组进来一律复制进 Grid；返回的 Site / Plan 拆成 numpy 数组与 dict，前端拼产物。
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bind_util.hpp"
#include "skyisle/town/plan.hpp"
#include "skyisle/town/site.hpp"
#include "skyisle/town/style.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace skyisle;
using namespace skyisle::town;

namespace {

template <class T>
Grid<T> grid_any(nb::handle h) {
    ArrAnyC a = nb::cast<ArrAnyC>(h);
    if (a.ndim() != 2) throw std::invalid_argument("expected a 2-d array");
    Grid<T> g(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)));
    g.v = anyvec<T>(h);
    return g;
}

Mask mask_any(nb::handle h) {
    Grid<uint8_t> g = grid_any<uint8_t>(h);
    for (auto& x : g.v) x = x ? 1 : 0;
    return g;
}

std::vector<V2> line_of(nb::handle h) {
    const std::vector<double> v = anyvec<double>(h);
    std::vector<V2> out(v.size() / 2);
    for (size_t k = 0; k < out.size(); ++k) out[k] = {v[2 * k], v[2 * k + 1]};
    return out;
}

nb::dict site_dict(Site&& s) {
    nb::dict d;
    d["res_m"] = s.res_m;
    d["H"] = s.H;
    d["W"] = s.W;
    d["x0"] = s.x0;
    d["y0"] = s.y0;
    d["frame_x"] = s.frame_x;
    d["frame_y"] = s.frame_y;
    d["lat_deg"] = s.lat_deg;
    d["height"] = grid_np(std::move(s.height));
    d["water_level"] = grid_np(std::move(s.water_level));
    d["water"] = grid_np(std::move(s.water));
    d["sky"] = mask_np(std::move(s.sky));
    d["edge"] = mask_np(std::move(s.edge));
    d["farmland"] = mask_np(std::move(s.farmland));
    d["flood"] = mask_np(std::move(s.flood));
    d["landcover"] = grid_np(std::move(s.landcover));
    d["island"] = grid_np(std::move(s.island));
    d["water_dist_m"] = grid_np(std::move(s.water_dist_m));
    nb::list rivers;
    for (River& r : s.rivers) {
        nb::dict q;
        std::vector<double> xy(2 * r.line.size());
        for (size_t k = 0; k < r.line.size(); ++k) xy[2 * k] = r.line[k].x, xy[2 * k + 1] = r.line[k].y;
        q["line"] = to_np(std::move(xy), {r.line.size(), 2});
        q["width_m"] = arr(r.width_m);
        q["depth_m"] = arr(r.depth_m);
        q["surface_m"] = arr(r.surface_m);
        q["seasonal"] = r.seasonal;
        rivers.append(q);
    }
    d["rivers"] = rivers;
    return d;
}

// site_dict 的逆：前端手里的地面 dict → Site
Site site_from(nb::dict d) {
    Site s;
    s.res_m = num(d, "res_m");
    s.x0 = num(d, "x0");
    s.y0 = num(d, "y0");
    s.frame_x = num(d, "frame_x");
    s.frame_y = num(d, "frame_y");
    s.lat_deg = num(d, "lat_deg");
    s.height = grid_any<float>(d["height"]);
    s.H = s.height.H, s.W = s.height.W;
    s.water_level = grid_any<float>(d["water_level"]);
    s.water = grid_any<uint8_t>(d["water"]);
    s.sky = mask_any(d["sky"]);
    s.edge = mask_any(d["edge"]);
    s.farmland = mask_any(d["farmland"]);
    s.flood = mask_any(d["flood"]);
    s.landcover = grid_any<uint8_t>(d["landcover"]);
    s.island = grid_any<int16_t>(d["island"]);
    s.water_dist_m = grid_any<float>(d["water_dist_m"]);
    for (nb::handle h : nb::cast<nb::list>(d["rivers"])) {
        nb::dict q = nb::cast<nb::dict>(h);
        River r;
        r.line = line_of(q["line"]);
        r.width_m = anyvec<double>(q["width_m"]);
        r.depth_m = anyvec<double>(q["depth_m"]);
        r.surface_m = anyvec<double>(q["surface_m"]);
        r.seasonal = nb::cast<bool>(q["seasonal"]);
        s.rivers.push_back(std::move(r));
    }
    return s;
}

constexpr double DEG = 3.141592653589793 / 180.0;

nb::list xy(const V2& p) {
    nb::list l;
    l.append(p.x);
    l.append(p.y);
    return l;
}

nb::ndarray<nb::numpy, double> line_np(const std::vector<V2>& line) {
    std::vector<double> v(2 * line.size());
    for (size_t k = 0; k < line.size(); ++k) v[2 * k] = line[k].x, v[2 * k + 1] = line[k].y;
    return to_np(std::move(v), {line.size(), 2});
}

nb::dict obb_dict(const Obb& o) {
    nb::dict d;
    d["c"] = xy(o.c);
    d["facing_deg"] = o.facing / DEG;
    d["w"] = 2.0 * o.hw;
    d["d"] = 2.0 * o.hd;
    return d;
}

nb::dict plan_dict(Plan&& P, const Style& st) {
    nb::dict d;
    d["op"] = P.op;
    d["center"] = xy(P.center);
    d["facing_deg"] = P.facing / DEG;
    d["radius_m"] = P.radius;
    nb::list roads;
    for (const Road& r : P.roads) {
        nb::dict q;
        q["line"] = line_np(r.line);
        q["cls"] = r.cls;
        q["width_m"] = r.width_m;
        roads.append(q);
    }
    d["roads"] = roads;
    nb::list bridges;
    for (const Bridge& b : P.bridges) {
        nb::dict q;
        q["a"] = xy(b.a);
        q["b"] = xy(b.b);
        q["width_m"] = b.width_m;
        q["road"] = b.road;
        bridges.append(q);
    }
    d["bridges"] = bridges;
    nb::list comps;
    for (const Compound& c : P.compounds) {
        nb::dict q;
        q["kind"] = c.kind;
        q["template"] = c.tmpl;
        q["template_name"] = c.tmpl_name;
        q["func"] = c.func;
        q["name"] = c.name;
        q["plot"] = obb_dict(c.plot);
        q["access_side"] = c.access_side;
        q["access"] = xy(c.access);
        q["gate"] = xy(c.gate);
        q["gate_bearing_deg"] = c.gate_bearing / DEG;
        q["walled"] = c.walled;
        q["base_m"] = c.base_m;
        q["max_cut_m"] = c.max_cut_m;
        q["terraced"] = c.max_cut_m > st.max_cut_m;
        q["households"] = c.households;
        q["buildings"] = c.buildings;
        comps.append(q);
    }
    d["compounds"] = comps;
    nb::list bl;
    for (const Building& b : P.buildings) {
        nb::dict q = obb_dict(b.box);
        q["func"] = b.func;
        q["role"] = b.role;
        q["name"] = b.name;
        q["roof"] = b.roof;
        q["material"] = b.material;
        q["compound"] = b.compound;
        q["storeys"] = b.storeys;
        q["eave_m"] = b.eave_m;
        q["pitch_deg"] = b.pitch_deg;
        q["base_m"] = b.base_m;
        q["cut_m3"] = b.cut_m3;
        q["fill_m3"] = b.fill_m3;
        q["max_cut_m"] = b.max_cut_m;
        bl.append(q);
    }
    d["buildings"] = bl;
    nb::list fl;
    for (const Feature& f : P.features) {
        nb::dict q;
        q["kind"] = f.kind;
        q["func"] = f.func;
        q["name"] = f.name;
        q["poly"] = line_np(f.poly);
        q["p"] = xy(f.p);
        q["r"] = f.r;
        q["facing_deg"] = f.facing / DEG;
        q["z"] = f.z;
        fl.append(q);
    }
    d["features"] = fl;
    std::vector<int32_t> hid, hk, hp, hc;
    for (const Household& h : P.households) hid.push_back(h.id), hk.push_back(h.kind), hp.push_back(h.parent), hc.push_back(h.compound);
    nb::dict hh;
    hh["id"] = arr(hid);
    hh["kind"] = arr(hk);
    hh["parent"] = arr(hp);
    hh["compound"] = arr(hc);
    d["households"] = hh;
    nb::dict mt;
    for (auto& [k, v] : P.metrics) mt[k.c_str()] = v;
    d["metrics"] = mt;
    nb::list ck;
    for (const Check& c : P.checks) {
        nb::dict q;
        q["id"] = c.id;
        q["hard"] = c.hard;
        q["ok"] = c.ok;
        q["msg"] = c.msg;
        ck.append(q);
    }
    d["checks"] = ck;
    nb::dict tm;
    for (auto& [k, v] : P.timing) tm[k.c_str()] = v;
    d["timing"] = tm;
    d["occ"] = grid_np(std::move(P.occ));
    return d;
}

}  // namespace

void bind_town(nb::module_& m) {
    m.def("town_plan", [](nb::dict site, nb::dict req, nb::handle cfg) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        const Site s = site_from(site);
        const Style st = parse_style(c);
        PlanRequest r;
        r.scale = nb::cast<std::string>(req["scale"]);
        r.hh_farm = nb::cast<int>(req["hh_farm"]);
        r.hh_market = nb::cast<int>(req["hh_market"]);
        r.hh_special = nb::cast<int>(req["hh_special"]);
        r.seed = nb::cast<uint64_t>(req["seed"]);
        const std::vector<double> an = nb::cast<std::vector<double>>(req["anchor"]);
        r.anchor = {an.at(0), an.at(1)};
        for (nb::handle h : nb::cast<nb::list>(req["landings"])) {
            const std::vector<double> q = nb::cast<std::vector<double>>(h);
            r.landings.push_back({q.at(0), q.at(1)});
        }
        for (nb::handle h : nb::cast<nb::list>(req["exits"])) {
            nb::dict q = nb::cast<nb::dict>(h);
            r.exits.push_back({num(q, "bearing_deg") * DEG, num(q, "dist_m"), num(q, "weight"), nb::cast<std::string>(q["kind"])});
        }
        r.wind_from = req["wind_from_deg"].is_none() ? NaN : num(req, "wind_from_deg") * DEG;
        r.want_landing = nb::cast<bool>(req["want_landing"]);
        r.force_operator = nb::cast<std::string>(req["operator"]);
        Plan P;
        {
            nb::gil_scoped_release nogil;
            P = plan_site(s, st, r);
        }
        return plan_dict(std::move(P), st);
    }, "site"_a, "req"_a, "cfg"_a);

    m.def("town_site_window", [](nb::dict inp, nb::handle cfg) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        WindowIn in;
        in.height = grid_any<double>(inp["height"]);
        in.island = grid_any<int16_t>(inp["island"]);
        in.lake = mask_any(inp["lake"]);
        in.floodplain = mask_any(inp["floodplain"]);
        in.arable = mask_any(inp["arable"]);
        in.terrace = mask_any(inp["terrace"]);
        in.landcover = grid_any<uint8_t>(inp["landcover"]);
        in.river_depth = grid_any<double>(inp["river_depth"]);
        in.coarse_res_m = num(inp, "coarse_res_m");
        in.center_r = num(inp, "center_r");
        in.center_c = num(inp, "center_c");
        in.frame_x = num(inp, "frame_x");
        in.frame_y = num(inp, "frame_y");
        in.half_m = num(inp, "half_m");
        in.res_m = num(inp, "res_m");
        in.lat_deg = num(inp, "lat_deg");
        in.seed = nb::cast<uint64_t>(inp["seed"]);
        for (nb::handle h : nb::cast<nb::list>(inp["rivers"])) {
            nb::dict q = nb::cast<nb::dict>(h);
            River r;
            r.line = line_of(q["line"]);
            r.width_m = anyvec<double>(q["width_m"]);
            r.seasonal = nb::cast<bool>(q["seasonal"]);
            in.rivers.push_back(std::move(r));
        }
        Site s;
        {
            nb::gil_scoped_release nogil;
            s = build_site_window(in, c);
        }
        return site_dict(std::move(s));
    }, "inp"_a, "cfg"_a);

    m.def("town_site_synth", [](const std::string& kind, double half_m, double res_m, double lat_deg, uint64_t seed, nb::handle cfg) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        if (!synth_kind_known(kind)) throw std::invalid_argument("unknown synth terrain: " + kind);
        Site s;
        {
            nb::gil_scoped_release nogil;
            s = build_site_synth(kind, half_m, res_m, lat_deg, seed, c);
        }
        return site_dict(std::move(s));
    }, "kind"_a, "half_m"_a, "res_m"_a, "lat_deg"_a, "seed"_a, "cfg"_a);

    m.def("town_site_heightmap", [](nb::handle height, nb::handle water, double res_m, double lat_deg, uint64_t seed, nb::handle cfg) {
        Config tmp;
        const Config& c = cfg_of(cfg, tmp);
        GridD h = grid_any<double>(height);
        Mask w;
        const bool has_w = !water.is_none();
        if (has_w) w = mask_any(water);
        Site s;
        {
            nb::gil_scoped_release nogil;
            s = build_site_heightmap(h, has_w ? &w : nullptr, res_m, lat_deg, seed, c);
        }
        return site_dict(std::move(s));
    }, "height"_a, "water"_a, "res_m"_a, "lat_deg"_a, "seed"_a, "cfg"_a);
}
