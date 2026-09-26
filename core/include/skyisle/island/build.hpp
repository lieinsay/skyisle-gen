// 第三层的「步」：build_terrain（布局 → 岛形 → 势力范围 → 贴栅格 → 岸距与连接 → 崖缘）与 build_hydro（水系、河道、地表、可耕地）。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

// res_m ≤ 0 = 用配置的 res_m；threads ≤ 0 = 1。结果与线程数无关。
Group build_terrain(const NodeInputs& inp, const PlanetView& pv, const Config& c, double res_m = 0.0, int threads = 1);
void build_hydro(Group& g, const PlanetView& pv, const Config& c, int threads = 1);

// 并行地对 [0, n) 做 fn(k)（threads ≤ 1 时顺序做）
template <class Fn>
void parallel_for(int n, int threads, Fn&& fn);

}  // namespace skyisle::island

#include <atomic>
#include <thread>
#include <vector>

template <class Fn>
void skyisle::island::parallel_for(int n, int threads, Fn&& fn) {
    if (threads <= 1 || n <= 1) {
        for (int k = 0; k < n; ++k) fn(k);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    std::exception_ptr err = nullptr;
    std::atomic<bool> failed{false};
    const int t = std::min(threads, n);
    for (int w = 0; w < t; ++w)
        pool.emplace_back([&]() {
            for (;;) {
                const int k = next.fetch_add(1);
                if (k >= n || failed.load()) break;
                try {
                    fn(k);
                } catch (...) {
                    if (!failed.exchange(true)) err = std::current_exception();
                }
            }
        });
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
}
