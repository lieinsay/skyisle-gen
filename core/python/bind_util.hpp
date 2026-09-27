// 绑定用的数组转换：numpy ↔ Grid / std::vector。
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <initializer_list>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/flow.hpp"
#include "skyisle/grid.hpp"

namespace nb = nanobind;

using ArrD1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrD2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ArrB2 = nb::ndarray<const bool, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ArrI2 = nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ArrS2 = nb::ndarray<const int16_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> to_np(std::vector<T>&& v, std::initializer_list<size_t> shape) {
    auto* p = new std::vector<T>(std::move(v));
    nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
    return nb::ndarray<nb::numpy, T>(p->data(), shape, owner);
}

inline nb::ndarray<nb::numpy, bool> bool_np(std::vector<uint8_t>&& v, std::initializer_list<size_t> shape) {
    auto* p = new std::vector<uint8_t>(std::move(v));
    nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<uint8_t>*>(q); });
    return nb::ndarray<nb::numpy, bool>(reinterpret_cast<bool*>(p->data()), shape, owner);
}

template <class T>
nb::ndarray<nb::numpy, T> grid_np(skyisle::Grid<T>&& g) {
    const size_t H = g.H, W = g.W;
    return to_np(std::move(g.v), {H, W});
}

inline nb::ndarray<nb::numpy, bool> mask_np(skyisle::Mask&& g) {
    const size_t H = g.H, W = g.W;
    return bool_np(std::move(g.v), {H, W});
}

inline skyisle::GridD grid_from(const ArrD2& a) {
    skyisle::GridD g(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)));
    std::memcpy(g.v.data(), a.data(), g.v.size() * sizeof(double));
    return g;
}

inline skyisle::Mask mask_from(const ArrB2& a) {
    skyisle::Mask g(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)));
    const bool* p = a.data();
    for (size_t k = 0; k < g.v.size(); ++k) g.v[k] = p[k] ? 1 : 0;
    return g;
}

inline nb::tuple flow_np(skyisle::FlowDir&& f, size_t H, size_t W) {
    return nb::make_tuple(to_np(std::move(f.ri), {H, W}), to_np(std::move(f.rj), {H, W}), bool_np(std::move(f.to_void), {H, W}));
}

// 展平的配置：{"num": {键: 数}, "vec": {键: [数]}, "str": {键: 串}（可缺）}；Config 对象（make_config 转好的）直接用
skyisle::Config cfg_from(const nb::dict& d);
const skyisle::Config& cfg_of(nb::handle h, skyisle::Config& tmp);

