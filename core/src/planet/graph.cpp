// 航线网络的图算法（graph.hpp；skyisle_gen/graph.py 的 C++ 版）。
#include "skyisle/planet/graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <thread>
#include <utility>

namespace skyisle::planet {

namespace {
using Item = std::pair<double, int64_t>;   // heapq 的 (dist, node)：先比距离、再比节点号
using MinHeap = std::priority_queue<Item, std::vector<Item>, std::greater<Item>>;
}  // namespace

std::vector<int64_t> argsort_stable(const std::vector<double>& a) {
    std::vector<int64_t> idx(a.size());
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t x, int64_t y) { return a[x] < a[y]; });
    return idx;
}

CSR make_csr(int64_t n, const std::vector<int64_t>& src, const std::vector<int64_t>& dst) {
    CSR g;
    g.n = n;
    const size_t m = src.size();
    std::vector<int64_t> order(m);
    std::iota(order.begin(), order.end(), int64_t{0});
    // np.lexsort((dst, src))：主键 src、次键 dst，稳定
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return src[a] != src[b] ? src[a] < src[b] : dst[a] < dst[b];
    });
    g.dst.resize(m);
    g.edge_id = order;
    g.indptr.assign(static_cast<size_t>(n) + 1, 0);
    for (size_t k = 0; k < m; ++k) {
        g.dst[k] = dst[order[k]];
        ++g.indptr[static_cast<size_t>(src[order[k]]) + 1];
    }
    for (int64_t i = 0; i < n; ++i) g.indptr[i + 1] += g.indptr[i];
    return g;
}

Paths dijkstra(const CSR& g, const std::vector<double>& w, const std::vector<int64_t>& sources, double max_dist) {
    const int64_t n = g.n;
    Paths r;
    r.dist.assign(n, INF);
    r.pred_node.assign(n, -1);
    r.pred_edge.assign(n, -1);
    std::vector<uint8_t> done(n, 0);
    MinHeap heap;
    for (int64_t s : sources) {
        if (0.0 < r.dist[s]) {
            r.dist[s] = 0.0;
            heap.push({0.0, s});
        }
    }
    while (!heap.empty()) {
        const Item it = heap.top();
        heap.pop();
        const double d = it.first;
        const int64_t u = it.second;
        if (d > max_dist) break;
        if (done[u]) continue;
        done[u] = 1;
        for (int64_t k = g.indptr[u]; k < g.indptr[u + 1]; ++k) {
            const int64_t e = g.edge_id[k];
            const double we = w[e];
            if (!std::isfinite(we)) continue;
            const int64_t v = g.dst[k];
            const double nd = d + we;
            if (nd < r.dist[v]) {
                r.dist[v] = nd;
                r.pred_node[v] = u;
                r.pred_edge[v] = e;
                heap.push({nd, v});
            }
        }
    }
    if (max_dist < INF) {
        for (int64_t i = 0; i < n; ++i)
            if (r.dist[i] > max_dist) {
                r.dist[i] = INF;
                r.pred_node[i] = -1;
                r.pred_edge[i] = -1;
            }
    }
    return r;
}

std::vector<double> accumulate_along_tree(const Paths& t, const std::vector<double>& ev) {
    const size_t n = t.dist.size();
    std::vector<double> o(n, 0.0);
    for (int64_t u : argsort_stable(t.dist)) {
        if (!std::isfinite(t.dist[u])) {
            o[u] = NaN;
            continue;
        }
        const int64_t p = t.pred_node[u], e = t.pred_edge[u];
        if (p < 0) continue;
        o[u] = o[p] + ev[e];
    }
    return o;
}

namespace {

// 一个源的 Brandes：把本源对各条边的贡献写进 contrib（每条边至多一次；未碰到的边保持 0）
void brandes_one(const CSR& g, const std::vector<double>& w, int64_t s, double c_min, double rel_tol, std::vector<double>& contrib,
                 std::vector<double>& dist, std::vector<double>& sigma, std::vector<double>& delta, std::vector<uint8_t>& done,
                 std::vector<int64_t>& order, std::vector<int64_t>& touched) {
    const int64_t n = g.n;
    std::fill(dist.begin(), dist.end(), INF);
    std::fill(sigma.begin(), sigma.end(), 0.0);
    std::fill(done.begin(), done.end(), 0);
    order.clear();
    dist[s] = 0.0;
    sigma[s] = 1.0;
    MinHeap heap;
    heap.push({0.0, s});
    while (!heap.empty()) {
        const Item it = heap.top();
        heap.pop();
        const double d = it.first;
        const int64_t u = it.second;
        if (done[u]) continue;
        done[u] = 1;
        order.push_back(u);
        for (int64_t k = g.indptr[u]; k < g.indptr[u + 1]; ++k) {
            const double we = w[g.edge_id[k]];
            if (!std::isfinite(we)) continue;
            const int64_t v = g.dst[k];
            const double nd = d + we;
            const double eps = rel_tol * std::max(1.0, std::fabs(nd));
            if (nd < dist[v] - eps) {
                dist[v] = nd;
                sigma[v] = sigma[u];
                heap.push({nd, v});
            } else if (std::isfinite(dist[v]) && std::fabs(nd - dist[v]) <= eps) {
                sigma[v] += sigma[u];
            }
        }
    }
    std::fill(delta.begin(), delta.end(), 0.0);
    auto tw = [&](int64_t v) { return (v != s && dist[v] >= c_min && std::isfinite(dist[v])) ? 1.0 : 0.0; };
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t u = *it;
        const double du = dist[u];
        for (int64_t k = g.indptr[u]; k < g.indptr[u + 1]; ++k) {
            const int64_t e = g.edge_id[k];
            const double we = w[e];
            if (!std::isfinite(we)) continue;
            const int64_t v = g.dst[k];
            if (!std::isfinite(dist[v])) continue;
            if (std::fabs(du + we - dist[v]) <= rel_tol * std::max(1.0, std::fabs(du + we)) && sigma[v] > 0) {
                const double c = sigma[u] / sigma[v] * (tw(v) + delta[v]);
                delta[u] += c;
                contrib[e] += c;
                touched.push_back(e);
            }
        }
    }
    (void)n;
}

}  // namespace

std::vector<double> betweenness_sampled(const CSR& g, const std::vector<double>& w, int64_t n_edges, const std::vector<int64_t>& sources,
                                        double c_min, double rel_tol, int threads) {
    std::vector<double> flow(n_edges, 0.0);
    const int64_t ns = static_cast<int64_t>(sources.size());
    const int T = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(threads, ns)));
    struct Work {
        std::vector<double> contrib, dist, sigma, delta;
        std::vector<uint8_t> done;
        std::vector<int64_t> order, touched;
    };
    std::vector<Work> ws(T);
    for (auto& x : ws) {
        x.contrib.assign(n_edges, 0.0);
        x.dist.resize(g.n);
        x.sigma.resize(g.n);
        x.delta.resize(g.n);
        x.done.resize(g.n);
    }
    // 源按块（每块 T 个）并行；块内按源的次序把贡献加进 flow（与逐源顺序相加逐位相同：每个源对每条边至多加一次）
    for (int64_t b0 = 0; b0 < ns; b0 += T) {
        const int nb = static_cast<int>(std::min<int64_t>(T, ns - b0));
        auto job = [&](int t) {
            Work& x = ws[t];
            brandes_one(g, w, sources[b0 + t], c_min, rel_tol, x.contrib, x.dist, x.sigma, x.delta, x.done, x.order, x.touched);
        };
        if (nb == 1) {
            job(0);
        } else {
            std::vector<std::thread> th;
            for (int t = 0; t < nb; ++t) th.emplace_back(job, t);
            for (auto& h : th) h.join();
        }
        for (int t = 0; t < nb; ++t) {
            Work& x = ws[t];
            for (int64_t e : x.touched) {   // 一条边在一个源里只会碰到一次
                flow[e] += x.contrib[e];
                x.contrib[e] = 0.0;
            }
            x.touched.clear();
        }
    }
    return flow;
}

std::vector<int64_t> weak_components(int64_t n, const std::vector<int64_t>& src, const std::vector<int64_t>& dst) {
    std::vector<int64_t> parent(n);
    std::iota(parent.begin(), parent.end(), int64_t{0});
    auto find = [&](int64_t x) {
        int64_t root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != root) {
            const int64_t nx = parent[x];
            parent[x] = root;
            x = nx;
        }
        return root;
    };
    for (size_t k = 0; k < src.size(); ++k) {
        const int64_t ra = find(src[k]), rb = find(dst[k]);
        if (ra != rb) {
            if (ra < rb) parent[rb] = ra;
            else parent[ra] = rb;
        }
    }
    // 分量号 = 根（= 最小成员）在所有根里的名次（np.unique(roots, return_inverse=True)）
    std::vector<int64_t> roots(n), rank(n, -1), comp(n);
    for (int64_t i = 0; i < n; ++i) roots[i] = find(i);
    int64_t c = 0;
    for (int64_t i = 0; i < n; ++i)
        if (roots[i] == i) rank[i] = c++;
    for (int64_t i = 0; i < n; ++i) comp[i] = rank[roots[i]];
    return comp;
}

}  // namespace skyisle::planet
