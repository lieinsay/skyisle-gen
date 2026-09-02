"""航线网络图算法：CSR、Dijkstra（单源/多源）、抽样介数（Brandes）、连通分量。

确定性守则：堆键 (dist, node_id)；邻接按 (src, dst) 稳定排序；
相等距离的松弛不替换前驱（首达者留，处理顺序确定）。
"""
from __future__ import annotations

import heapq

import numpy as np

INF = np.inf


class CSR:
    """有向图的压缩邻接。edge_id 指回外部边数组（可与权重数组对齐）。"""

    __slots__ = ("n", "indptr", "dst", "edge_id")

    def __init__(self, n: int, src: np.ndarray, dst: np.ndarray):
        self.n = n
        order = np.lexsort((dst, src))  # 主键 src、次键 dst，稳定
        self.dst = dst[order].astype(np.int64)
        self.edge_id = order.astype(np.int64)
        counts = np.bincount(src[order], minlength=n)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def dijkstra(csr: CSR, weights: np.ndarray, sources, source_dist=None):
    """weights 与 csr.edge_id 对齐（weights[csr.edge_id[k]] 是 CSR 第 k 条边的权）。
    返回 (dist[n], pred_node[n], pred_edge[n])；pred_edge 为外部边 id，-1 表示无。
    不可达为 inf。权重含 inf 的边视为不存在。
    """
    n = csr.n
    dist = np.full(n, INF, dtype=np.float64)
    pred_node = np.full(n, -1, dtype=np.int64)
    pred_edge = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    heap: list[tuple[float, int]] = []
    if source_dist is None:
        source_dist = [0.0] * len(sources)
    for s, d0 in zip(sources, source_dist):
        s = int(s)
        if d0 < dist[s]:
            dist[s] = d0
            heapq.heappush(heap, (float(d0), s))
    indptr, dsts, eids = csr.indptr, csr.dst, csr.edge_id
    w = weights
    push = heapq.heappush
    pop = heapq.heappop
    while heap:
        d, u = pop(heap)
        if done[u]:
            continue
        done[u] = True
        for k in range(indptr[u], indptr[u + 1]):
            we = w[eids[k]]
            if not np.isfinite(we):
                continue
            v = dsts[k]
            nd = d + we
            if nd < dist[v]:
                dist[v] = nd
                pred_node[v] = u
                pred_edge[v] = eids[k]
                push(heap, (nd, int(v)))
    return dist, pred_node, pred_edge


def accumulate_along_tree(pred_node: np.ndarray, pred_edge: np.ndarray, dist: np.ndarray,
                          *edge_values: np.ndarray):
    """沿最短路树把每条边上的量累加到节点（如 C = Σcost、L = Σ(-ln perm)）。
    按 dist 升序处理，parent 必先于 child。返回与 edge_values 数目相同的节点数组。
    """
    n = pred_node.shape[0]
    outs = [np.zeros(n, dtype=np.float64) for _ in edge_values]
    order = np.argsort(dist, kind="stable")
    for u in order:
        if not np.isfinite(dist[u]):
            for o in outs:
                o[u] = np.nan
            continue
        p, e = pred_node[u], pred_edge[u]
        if p < 0:
            continue
        for o, ev in zip(outs, edge_values):
            o[u] = o[p] + ev[e]
    return outs


def betweenness_sampled(csr: CSR, weights: np.ndarray, n_edges: int,
                        sources: np.ndarray, c_min: float = 0.0,
                        rel_tol: float = 1e-9):
    """Brandes 抽样介数（有向、加权）。只累计距离 ≥ c_min 的 OD（排除近距噪声）。
    返回 flow[n_edges]（外部边 id 索引）。
    """
    n = csr.n
    flow = np.zeros(n_edges, dtype=np.float64)
    indptr, dsts, eids = csr.indptr, csr.dst, csr.edge_id
    for s in sources:
        s = int(s)
        dist = np.full(n, INF)
        sigma = np.zeros(n)
        dist[s] = 0.0
        sigma[s] = 1.0
        done = np.zeros(n, dtype=bool)
        heap = [(0.0, s)]
        order: list[int] = []
        while heap:
            d, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            order.append(u)
            for k in range(indptr[u], indptr[u + 1]):
                we = weights[eids[k]]
                if not np.isfinite(we):
                    continue
                v = dsts[k]
                nd = d + we
                eps = rel_tol * max(1.0, abs(nd))
                if nd < dist[v] - eps:
                    dist[v] = nd
                    sigma[v] = sigma[u]
                    heapq.heappush(heap, (nd, int(v)))
                elif np.isfinite(dist[v]) and abs(nd - dist[v]) <= eps:
                    sigma[v] += sigma[u]
        tw = ((dist >= c_min) & np.isfinite(dist)).astype(np.float64)
        tw[s] = 0.0
        delta = np.zeros(n)
        for u in reversed(order):
            du = dist[u]
            for k in range(indptr[u], indptr[u + 1]):
                we = weights[eids[k]]
                if not np.isfinite(we):
                    continue
                v = dsts[k]
                if not np.isfinite(dist[v]):
                    continue
                if abs(du + we - dist[v]) <= rel_tol * max(1.0, abs(du + we)) and sigma[v] > 0:
                    c = sigma[u] / sigma[v] * (tw[v] + delta[v])
                    delta[u] += c
                    flow[eids[k]] += c
    return flow


def weak_components(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """无向（弱）连通分量。返回 comp[n]，分量 id 按最小成员节点编号排序重标。"""
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    uniq, comp = np.unique(roots, return_inverse=True)
    return comp
