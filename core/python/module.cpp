// nanobind 绑定：skyisle_gen._core（docs/PLAN-CORE.md 第五节）。
// 两层：「步」（build_terrain / build_hydro，前端正式调用）与「公共件」（给 pytest 做同输入对照）。
// 数组一律 C 连续；传入时复制进 Grid，返回的数组由 C++ 分配、numpy 持有。
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>

#include <cstring>
#include <optional>

#include "skyisle/flow.hpp"
#include "skyisle/grid.hpp"
#include "skyisle/rng.hpp"
#include "bind_util.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace skyisle;

void bind_island(nb::module_& m);
void bind_planet(nb::module_& m);
void bind_civ(nb::module_& m);
void bind_town(nb::module_& m);

namespace {

Rng rng_of(uint64_t seed, uint64_t stream, const std::string& key) {
    return key.empty() ? stage_rng(seed, stream) : entity_rng(seed, stream, key);
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "skyisle-gen 的 C++ 核心（docs/PLAN-CORE.md）";
    m.def("version", []() { return std::string("p6c-1"); });

    // ---------------------------------------------------------------- 随机数
    m.def("rng_raw", [](uint64_t seed, uint64_t stream, const std::string& key, size_t n) {
        Rng r = rng_of(seed, stream, key);
        std::vector<uint64_t> out(n);
        for (auto& x : out) x = r.raw();
        return to_np(std::move(out), {n});
    }, "seed"_a, "stream"_a, "key"_a, "n"_a);
    m.def("rng_draw", [](uint64_t seed, uint64_t stream, const std::string& key, const std::string& kind, size_t n, double a, double b) {
        Rng r = rng_of(seed, stream, key);
        std::vector<double> out(n);
        for (auto& x : out) {
            if (kind == "random") x = r.random();
            else if (kind == "uniform") x = r.uniform(a, b);
            else if (kind == "normal") x = r.normal(a, b);
            else if (kind == "gamma") x = r.standard_gamma(a);
            else if (kind == "gamma2") x = r.gamma(a, b);
            else if (kind == "exponential") x = r.standard_exponential();
            else if (kind == "lognormal") x = r.lognormal(a, b);
            else if (kind == "poisson") x = static_cast<double>(r.poisson(a));
            else if (kind == "beta") x = r.beta(a, b);
            else if (kind == "integers") x = static_cast<double>(r.integers(static_cast<int64_t>(a), static_cast<int64_t>(b)));
            else throw std::invalid_argument("rng_draw: unknown kind " + kind);
        }
        return to_np(std::move(out), {n});
    }, "seed"_a, "stream"_a, "key"_a, "kind"_a, "n"_a, "a"_a = 0.0, "b"_a = 1.0);
    m.def("rng_choice_p", [](uint64_t seed, uint64_t stream, const std::string& key, std::vector<double> p, size_t n) {
        Rng r = rng_of(seed, stream, key);
        std::vector<int64_t> out(n);
        for (auto& x : out) x = r.choice_p(p);
        return to_np(std::move(out), {n});
    }, "seed"_a, "stream"_a, "key"_a, "p"_a, "n"_a);
    m.def("rng_choice_noreplace", [](uint64_t seed, uint64_t stream, const std::string& key, int64_t pop, int64_t size, int reps) {
        Rng r = rng_of(seed, stream, key);
        std::vector<int64_t> out;
        for (int t = 0; t < reps; ++t) {
            const std::vector<int64_t> v = r.choice_noreplace(pop, size);
            out.insert(out.end(), v.begin(), v.end());
        }
        const size_t n = out.size();
        return to_np(std::move(out), {n});
    });
    m.def("crc32", [](const std::string& s) { return crc32(s); });
    m.def("np_quantile", [](ArrD1 a, double q) { return np_quantile(std::vector<double>(a.data(), a.data() + a.shape(0)), q); });
    m.def("np_interp", [](ArrD1 x, ArrD1 xp, ArrD1 fp) {
        std::vector<double> r = np_interp(std::vector<double>(x.data(), x.data() + x.shape(0)), std::vector<double>(xp.data(), xp.data() + xp.shape(0)),
                                          std::vector<double>(fp.data(), fp.data() + fp.shape(0)));
        const size_t n = r.size();
        return to_np(std::move(r), {n});
    });
    m.def("np_convolve_valid", [](ArrD1 a, ArrD1 v) {
        std::vector<double> r = np_convolve_valid(std::vector<double>(a.data(), a.data() + a.shape(0)), std::vector<double>(v.data(), v.data() + v.shape(0)));
        const size_t n = r.size();
        return to_np(std::move(r), {n});
    });
    m.def("blas_ddot", [](ArrD1 a, ArrD1 b) { return blas_ddot(a.data(), b.data(), a.shape(0)); });
    m.def("np_sum_f32", [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) { return np_sum_f32(a.data(), a.shape(0)); });

    // ---------------------------------------------------------------- 小工具
    m.def("np_sum", [](ArrD1 a) { return np_sum(a.data(), a.shape(0)); });
    m.def("pyround", [](double x, int nd) { return pyround(x, nd); });
    m.def("math_fns", [](ArrD1 x, ArrD1 y) {   // 测试用：np_pow(x, y[0]) / c_pow / np_hypot / py_hypot
        const size_t n = x.shape(0);
        std::vector<double> a(n), b(n), c(n), d(n);
        for (size_t k = 0; k < n; ++k) {
            a[k] = np_pow(x.data()[k], y.data()[0]);
            b[k] = c_pow(x.data()[k], y.data()[0]);
            c[k] = np_hypot(x.data()[k], y.data()[k]);
            d[k] = py_hypot(x.data()[k], y.data()[k]);
        }
        return nb::make_tuple(to_np(std::move(a), {n}), to_np(std::move(b), {n}), to_np(std::move(c), {n}), to_np(std::move(d), {n}));
    });
    m.def("grid_interp", [](ArrD2 field, double lat0, double dlat, double lon0, double dlon, double lat, double lon) {
        LatLonGrid g{lat0, dlat, lon0, dlon, static_cast<int>(field.shape(0)), static_cast<int>(field.shape(1))};
        std::vector<double> f(field.data(), field.data() + field.size());
        return grid_interp(f, g, lat, lon);
    });

    // ---------------------------------------------------------------- 噪声
    m.def("fractal_noise", [](uint64_t seed, uint64_t stream, const std::string& key, double x0, double y0, double x1, double y1,
                              double feature_km, int octaves, double persistence, ArrD2 X, ArrD2 Y) {
        Rng r = rng_of(seed, stream, key);
        FractalNoise fn(r, x0, y0, x1, y1, feature_km, octaves, persistence);
        const size_t H = X.shape(0), W = X.shape(1);
        std::vector<double> out(H * W);
        for (size_t k = 0; k < out.size(); ++k) out[k] = fn.sample(X.data()[k], Y.data()[k]);
        return to_np(std::move(out), {H, W});
    });

    // ---------------------------------------------------------------- 栅格
    m.def("label_components", [](ArrB2 mask, int connectivity) {
        GridI lab;
        int n = label_components(mask_from(mask), connectivity, lab);
        return nb::make_tuple(grid_np(std::move(lab)), n);
    }, "mask"_a, "connectivity"_a = 4);
    m.def("largest_component", [](ArrB2 mask) { return mask_np(largest_component(mask_from(mask))); });
    m.def("binary_erode", [](ArrB2 mask, int it, int conn) { return mask_np(binary_erode(mask_from(mask), it, conn)); },
          "mask"_a, "iterations"_a = 1, "connectivity"_a = 8);
    m.def("binary_dilate", [](ArrB2 mask, int it, int conn) { return mask_np(binary_dilate(mask_from(mask), it, conn)); },
          "mask"_a, "iterations"_a = 1, "connectivity"_a = 8);
    m.def("distance_bands", [](ArrB2 mask, int max_iter) { return grid_np(distance_bands(mask_from(mask), max_iter)); });
    m.def("label_by_island", [](ArrB2 mask, ArrS2 island_id, int conn) {
        Grid<int16_t> ii(static_cast<int>(island_id.shape(0)), static_cast<int>(island_id.shape(1)));
        std::memcpy(ii.v.data(), island_id.data(), ii.v.size() * sizeof(int16_t));
        GridI lab;
        const int n = label_by_island(mask_from(mask), ii, conn, lab);
        return nb::make_tuple(grid_np(std::move(lab)), n);
    });
    m.def("window_extrema", [](ArrD2 a, int r, ArrB2 mask) {
        GridD hi, lo;
        window_extrema(grid_from(a), r, mask_from(mask), hi, lo);
        return nb::make_tuple(grid_np(std::move(hi)), grid_np(std::move(lo)));
    });
    m.def("nearest_propagate", [](ArrB2 seed, int max_iter, double step_m, std::optional<ArrB2> within) {
        GridD dist;
        Grid<int64_t> src;
        Mask w;
        if (within) w = mask_from(*within);
        nearest_propagate(mask_from(seed), max_iter, step_m, within ? &w : nullptr, dist, src);
        return nb::make_tuple(grid_np(std::move(dist)), grid_np(std::move(src)));
    }, "seed"_a, "max_iter"_a, "step_m"_a = 1.0, "within"_a = nb::none());
    m.def("block_mean", [](ArrD2 a, int f) { return grid_np(block_mean(grid_from(a), f)); });
    m.def("block_any", [](ArrB2 a, int f) { return mask_np(block_any(mask_from(a), f)); });
    m.def("upsample_bilinear", [](ArrD2 a, int f, int H, int W) { return grid_np(upsample_bilinear(grid_from(a), f, H, W)); });
    m.def("smooth121", [](ArrD2 a, ArrB2 mask, int passes) { return grid_np(smooth121(grid_from(a), mask_from(mask), passes)); });
    m.def("laplacian", [](ArrD2 a, ArrB2 mask) { return grid_np(laplacian(grid_from(a), mask_from(mask))); });
    m.def("slope_deg", [](ArrD2 h, ArrB2 mask, double res_m) { return grid_np(slope_deg(grid_from(h), mask_from(mask), res_m)); });

    // ---------------------------------------------------------------- 水文核心
    m.def("priority_fill", [](ArrD2 h, ArrB2 mask, double eps) { return grid_np(priority_fill(grid_from(h), mask_from(mask), eps)); },
          "h"_a, "mask"_a, "eps"_a = 1e-3);
    m.def("fill_iter", [](ArrD2 h, ArrB2 mask, int iters, double eps) { return grid_np(fill_iter(grid_from(h), mask_from(mask), iters, eps)); },
          "h"_a, "mask"_a, "iters"_a, "eps"_a = 0.01);
    m.def("d8", [](ArrD2 hf, ArrB2 mask, double res_m) {
        FlowDir f = d8(grid_from(hf), mask_from(mask), res_m);
        return flow_np(std::move(f), hf.shape(0), hf.shape(1));
    });
    m.def("d8_random", [](ArrD2 hf, ArrB2 mask, double res_m, uint64_t seed, uint64_t stream, const std::string& key, double p) {
        Rng r = rng_of(seed, stream, key);
        FlowDir f = d8_random(grid_from(hf), mask_from(mask), res_m, r, p);
        return flow_np(std::move(f), hf.shape(0), hf.shape(1));
    });
    m.def("accumulate", [](ArrB2 mask, ArrI2 ri, ArrI2 rj) {
        FlowDir f;
        f.ri.assign(ri.data(), ri.data() + ri.size());
        f.rj.assign(rj.data(), rj.data() + rj.size());
        f.to_void.assign(ri.size(), 0);
        return grid_np(accumulate(mask_from(mask), f));
    });

    bind_island(m);
    bind_planet(m);
    bind_civ(m);
    bind_town(m);
}
