// 水文核心：优先泛洪、迭代填洼、D8 / 随机流向、汇流、拓扑序。
#include "skyisle/flow.hpp"

#include <algorithm>
#include <queue>
#include <tuple>

namespace skyisle {

GridD priority_fill(const GridD& h, const Mask& mask, double eps) {
    const int H = h.H, W = h.W;
    const size_t N = h.size();
    std::vector<double> hl(N);
    for (size_t k = 0; k < N; ++k) hl[k] = mask.v[k] ? h.v[k] : -INF;
    std::vector<uint8_t> closed(N);
    for (size_t k = 0; k < N; ++k) closed[k] = mask.v[k] ? 0 : 1;
    Mask er = binary_erode(mask, 1);
    using E = std::tuple<double, int, int>;
    std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            if (mask.v[k] && !er.v[k]) {
                heap.emplace(hl[k], i, j);
                closed[k] = 1;
            }
        }
    std::vector<double> res = hl;
    while (!heap.empty()) {
        const auto [z, i, j] = heap.top();
        heap.pop();
        for (int n = 0; n < 8; ++n) {
            const int a = i + N8[n][0], b = j + N8[n][1];
            if (a < 0 || b < 0 || a >= H || b >= W) continue;
            const size_t k = static_cast<size_t>(a) * W + b;
            if (closed[k]) continue;
            closed[k] = 1;
            double zn = hl[k];
            if (zn < z + eps) zn = z + eps;
            res[k] = zn;
            heap.emplace(zn, a, b);
        }
    }
    GridD out(H, W, NaN);
    for (size_t k = 0; k < N; ++k)
        if (mask.v[k]) out.v[k] = res[k];
    return out;
}

GridD fill_iter(const GridD& h, const Mask& mask, int iters, double eps) {
    const int H = h.H, W = h.W;
    const size_t N = h.size();
    Mask er = binary_erode(mask, 1);
    std::vector<double> hh(N), Wf(N, INF);
    std::vector<uint8_t> edge(N, 0);
    std::vector<int32_t> changed;
    for (size_t k = 0; k < N; ++k) {
        hh[k] = mask.v[k] ? h.v[k] : INF;
        if (mask.v[k] && !er.v[k]) {
            edge[k] = 1;
            Wf[k] = hh[k];
            changed.push_back(static_cast<int32_t>(k));
        }
    }
    std::vector<int32_t> stamp(N, -1), cand;
    std::vector<std::pair<int32_t, double>> upd;
    for (int it = 0; it < iters; ++it) {
        cand.clear();
        for (int32_t c : changed) {
            const int i = c / W, j = c % W;
            for (int n = 0; n < 8; ++n) {
                const int a = i + N8[n][0], b = j + N8[n][1];
                if (a < 0 || b < 0 || a >= H || b >= W) continue;
                const int32_t t = a * W + b;
                if (!mask.v[t] || edge[t] || stamp[t] == it) continue;
                stamp[t] = it;
                cand.push_back(t);
            }
        }
        upd.clear();
        for (int32_t t : cand) {
            const int i = t / W, j = t % W;
            double mn = INF;
            for (int n = 0; n < 8; ++n) {
                const int a = i - N8[n][0], b = j - N8[n][1];
                const double v = (a < 0 || b < 0 || a >= H || b >= W) ? INF : Wf[static_cast<size_t>(a) * W + b];
                mn = std::min(mn, v);
            }
            const double nv = std::max(hh[t], mn + eps);
            if (nv != Wf[t]) upd.emplace_back(t, nv);
        }
        if (upd.empty()) break;
        changed.clear();
        for (const auto& u : upd) {
            Wf[u.first] = u.second;
            changed.push_back(u.first);
        }
    }
    GridD out(H, W, NaN);
    for (size_t k = 0; k < N; ++k)
        if (mask.v[k]) out.v[k] = std::isfinite(Wf[k]) ? Wf[k] : h.v[k];
    return out;
}

FlowDir d8(const GridD& hf, const Mask& mask, double res_m) {
    const int H = hf.H, W = hf.W;
    const size_t N = hf.size();
    FlowDir f;
    f.ri.assign(N, -1);
    f.rj.assign(N, -1);
    f.to_void.assign(N, 0);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            const double hh = mask.v[k] ? hf.v[k] : -INF;
            double best = -INF;
            int bi = -1, bj = -1;
            bool tv = false;
            for (int n = 0; n < 8; ++n) {
                const int di = N8[n][0], dj = N8[n][1];
                const int a = i - di, b = j - dj;
                const bool inb = a >= 0 && b >= 0 && a < H && b < W;
                const bool is_void = !inb || !mask(a, b);
                double drop;
                if (is_void) {
                    drop = mask.v[k] ? 1e6 : -INF;
                } else {
                    const double nb = hf(a, b);   // 邻格在掩膜内
                    const double dist = res_m * ((di && dj) ? SQRT2 : 1.0);
                    drop = (hh - nb) / dist;
                }
                if (drop > best) {
                    best = drop;
                    bi = std::min(std::max(i - di, 0), H - 1);
                    bj = std::min(std::max(j - dj, 0), W - 1);
                    tv = is_void;
                }
            }
            f.to_void[k] = tv ? 1 : 0;
            if (mask.v[k] && !tv && best > 0) {
                f.ri[k] = bi;
                f.rj[k] = bj;
            }
        }
    return f;
}

FlowDir d8_random(const GridD& hf, const Mask& mask, double res_m, Rng& rng, double p) {
    const int H = hf.H, W = hf.W;
    const size_t N = hf.size();
    FlowDir f;
    f.ri.assign(N, -1);
    f.rj.assign(N, -1);
    f.to_void.assign(N, 0);
    std::vector<double> u(N);
    rng.uniform_fill(0.0, 1.0, u.data(), N);
    double w[8];
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            const bool m = mask.v[k] != 0;
            const double hh = m ? hf.v[k] : INF;
            bool tv = false;
            for (int n = 0; n < 8; ++n) {
                const int di = N8[n][0], dj = N8[n][1];
                const int a = i - di, b = j - dj;
                const bool inb = a >= 0 && b >= 0 && a < H && b < W;
                const bool is_void = !inb || !mask(a, b);
                if (m && is_void) tv = true;
                w[n] = 0.0;
                if (m && !is_void) {
                    const double dist = res_m * ((di && dj) ? SQRT2 : 1.0);
                    const double drop = (hh - hf(a, b)) / dist;
                    if (drop > 0) w[n] = std::pow(std::max(drop, 0.0), p);
                }
            }
            f.to_void[k] = tv ? 1 : 0;
            double S = w[0];
            for (int n = 1; n < 8; ++n) S = S + w[n];
            const double uu = u[k] * S;
            int cnt = 0;
            double cs = 0.0;
            for (int n = 0; n < 8; ++n) {
                cs = n == 0 ? w[0] : cs + w[n];
                if (cs <= uu) ++cnt;
            }
            const int kk = std::min(cnt, 7);
            if (m && !tv && S > 0) {
                f.ri[k] = i - N8[kk][0];
                f.rj[k] = j - N8[kk][1];
            }
        }
    return f;
}

std::vector<int64_t> recv_flat(const FlowDir& f, int W) {
    std::vector<int64_t> r(f.ri.size(), -1);
    for (size_t k = 0; k < r.size(); ++k)
        if (f.ri[k] >= 0) r[k] = static_cast<int64_t>(f.ri[k]) * W + f.rj[k];
    return r;
}

std::vector<int32_t> downstream_first(const std::vector<int64_t>& recv, const Mask& mask) {
    const size_t N = recv.size();
    std::vector<int32_t> cnt(N + 1, 0);
    for (size_t k = 0; k < N; ++k)
        if (mask.v[k] && recv[k] >= 0 && mask.v[recv[k]]) cnt[recv[k] + 1]++;
    for (size_t k = 0; k < N; ++k) cnt[k + 1] += cnt[k];
    std::vector<int32_t> donors(cnt[N]);
    std::vector<int32_t> pos(cnt.begin(), cnt.end() - 1);
    for (size_t k = 0; k < N; ++k)
        if (mask.v[k] && recv[k] >= 0 && mask.v[recv[k]]) donors[pos[recv[k]]++] = static_cast<int32_t>(k);
    std::vector<int32_t> order;
    order.reserve(N);
    for (size_t k = 0; k < N; ++k)
        if (mask.v[k] && !(recv[k] >= 0 && mask.v[recv[k]])) order.push_back(static_cast<int32_t>(k));
    for (size_t q = 0; q < order.size(); ++q) {
        const int32_t r = order[q];
        for (int32_t d = cnt[r]; d < cnt[r + 1]; ++d) order.push_back(donors[d]);
    }
    return order;
}

GridD accumulate(const Mask& mask, const FlowDir& f, const GridD* weight) {
    const int H = mask.H, W = mask.W;
    const size_t N = mask.size();
    GridD A(H, W, 0.0);
    for (size_t k = 0; k < N; ++k)
        if (mask.v[k]) A.v[k] = weight ? weight->v[k] : 1.0;
    const std::vector<int64_t> recv = recv_flat(f, W);
    const std::vector<int32_t> order = downstream_first(recv, mask);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {   // 上游先加
        const int64_t r = recv[*it];
        if (r >= 0) A.v[r] += A.v[*it];
    }
    return A;
}

}  // namespace skyisle
