// 航线网络的图算法（行星计划 P6d）：skyisle_gen/graph.py 的 C++ 版——CSR、Dijkstra（单源 / 多源 / 有界）、沿最短路树累加、
// Brandes 抽样介数、弱连通分量。
//
// 与 Python 版逐位一致的要点：
//   - CSR 按 (src, dst) 稳定排序（np.lexsort((dst, src))），edge_id 指回外部边号；
//   - 堆的次序同 heapq 的 (dist, node) 元组：距离相同按节点号，同值的重复项先弹哪个都一样（DESIGN-NOTES 四点二十六）；
//   - 松弛只在严格更小时替换前驱；权为 inf 的边视为不存在；有界搜索弹出 > 界即停、再把界外的有限值清回 inf；
//   - 介数的 σ 与 δ 按弹出次序 / 倒序累加，flow[e] 按源的次序相加（每个源对每条边至多加一次）。
#pragma once

#include <cstdint>
#include <vector>

#include "skyisle/grid.hpp"

namespace skyisle::planet {

struct CSR {                      // graph.CSR：有向图的压缩邻接
    int64_t n = 0;
    std::vector<int64_t> indptr, dst, edge_id;
};
CSR make_csr(int64_t n, const std::vector<int64_t>& src, const std::vector<int64_t>& dst);

struct Paths {                    // dijkstra 的 (dist, pred_node, pred_edge)
    std::vector<double> dist;
    std::vector<int64_t> pred_node, pred_edge;
};
// weights 按外部边号；sources 各自距离 0（重复的源只取第一次）；max_dist = INF 即不设界
Paths dijkstra(const CSR& g, const std::vector<double>& w, const std::vector<int64_t>& sources, double max_dist = INF);
// graph.accumulate_along_tree：按 dist 稳定升序，父先于子，o[u] = o[p] + ev[e]；不可达记 NaN
std::vector<double> accumulate_along_tree(const Paths& t, const std::vector<double>& ev);
// graph.betweenness_sampled：有向、加权的 Brandes 抽样介数；只累计 dist ≥ c_min 的 OD。threads > 1 时按源分块并行、按源的次序归约（结果与线程数无关）
std::vector<double> betweenness_sampled(const CSR& g, const std::vector<double>& w, int64_t n_edges, const std::vector<int64_t>& sources,
                                        double c_min, double rel_tol = 1e-9, int threads = 1);
// graph.weak_components：无向（弱）连通分量，分量号按最小成员排
std::vector<int64_t> weak_components(int64_t n, const std::vector<int64_t>& src, const std::vector<int64_t>& dst);

// np.argsort(a, kind="stable")（a 可含 inf）
std::vector<int64_t> argsort_stable(const std::vector<double>& a);

}  // namespace skyisle::planet
