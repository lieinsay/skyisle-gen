// 聚落营建器（docs/PLAN-TOWN.md）的绑定：场地地形（岛群窗口细化 / 合成地形 / 外部高程图）。
// 数组进来一律复制进 Grid；返回的 Site 拆成 numpy 数组与元数据的 dict，前端拼产物。
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bind_util.hpp"
#include "skyisle/town/site.hpp"

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

}  // namespace

void bind_town(nb::module_& m) {
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
