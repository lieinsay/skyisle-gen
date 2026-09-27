// ⑧ 特征扩散（skyisle_gen/stages/s08_diffusion.py）：隔离度 → 特征表（主起源 / 次级起源 / 高隔离分量的本地起源，或手工特征表）
// → reach = exp(−最短路)（决策 1 方案 A；fast = 方案 C 的参考 λ 树）→ 同槽位对称不动点（含本地行 ε）→ share / strength / adopt。
// 特征表的随机数：entity_rng(seed, 8, 特征 id) 各一条流（log 均匀的半衰距离、阻力档内均匀），本地起源的槽位 entity_rng(seed, 8, "local:compN")。
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <stdexcept>

#include "skyisle/planet/civ.hpp"

namespace skyisle::planet {

namespace {

struct Slot {
    std::string id, category, mode, tier;
};

std::vector<Slot> read_slots(const Config& cfg) {
    std::vector<Slot> out;
    const int n = static_cast<int>(cfg.get("slots.slot.n", 0.0));
    for (int i = 0; i < n; ++i) {
        const std::string pre = "slots.slot." + std::to_string(i) + ".";
        Slot s;
        s.id = cfg.gets(pre + "id");
        s.category = cfg.gets(pre + "category");
        s.mode = cfg.gets("slots.categories." + s.category + ".mode");
        s.tier = cfg.gets("slots.categories." + s.category + ".resistance");
        out.push_back(s);
    }
    if (out.empty()) throw std::invalid_argument("s08: no slots (slots.toml)");
    return out;
}

// np.nan_to_num(x, nan=inf)：NaN → inf；原本的 ±inf → ±最大有限值（掩码在替换前算好）
inline double nan_to_inf(double x) {
    if (std::isnan(x)) return INF;
    if (x == INF) return DBL_MAX;
    if (x == -INF) return -DBL_MAX;
    return x;
}

// fixed_point_slot：S_t = R_t·(1 − r_t·max_{u≠t} S_u) 的 Jacobi 不动点。R [V, N]
FixedPoint fixed_point_slot(const std::vector<double>& R, const std::vector<double>& r, int V, int64_t N, double tol, int max_iter,
                            std::vector<double>& S) {
    S = R;
    std::vector<double> Sn(R.size());
    for (int it = 0; it < max_iter; ++it) {
        double dmax = 0.0;
        for (int64_t j = 0; j < N; ++j) {
            int i1 = 0;
            for (int i = 1; i < V; ++i)
                if (S[static_cast<size_t>(i) * N + j] > S[static_cast<size_t>(i1) * N + j]) i1 = i;   // argmax：第一个最大者
            const double m1 = S[static_cast<size_t>(i1) * N + j];
            double m2 = 0.0;   // np.partition(S, V−2, axis=0)[V−2]：第二大的值（并列最大时 = 最大）
            if (V >= 2) {
                bool have = false;
                for (int i = 0; i < V; ++i) {
                    if (i == i1) continue;
                    const double x = S[static_cast<size_t>(i) * N + j];
                    if (!have || x > m2) m2 = x, have = true;
                }
            }
            for (int i = 0; i < V; ++i) {
                const size_t k = static_cast<size_t>(i) * N + j;
                const double other = i == i1 ? m2 : m1;
                Sn[k] = R[k] * (1.0 - r[i] * other);
                dmax = std::max(dmax, std::fabs(Sn[k] - S[k]));
            }
        }
        S.swap(Sn);
        if (dmax < tol) return {it + 1, true, 0};
    }
    return {max_iter, false, 0};
}

}  // namespace

Diffusion stage8(const Config& cfg, uint64_t seed, const Islands& isl, const Barriers& b, const Routes& r, const Centers& ce) {
    if (cfg.gets("s08.engine", "field") == "mc") throw std::invalid_argument("s08.engine = mc is a reserved interface; use field");
    const int64_t N = static_cast<int64_t>(isl.n());
    const Directed g = directed(isl, b, r);
    const std::array<double, N_MODES> lam_ref = lambda_ref(cfg);
    Diffusion D;
    D.N = N;
    D.lambda_ref = lam_ref;

    // ---- 隔离度（反射型的连续替代：到枢纽集的图距离，进出取大）
    std::vector<int64_t> H(r.hubs.begin(), r.hubs.end());
    for (int64_t cn : ce.node) H.push_back(cn);
    std::sort(H.begin(), H.end());
    H.erase(std::unique(H.begin(), H.end()), H.end());
    const CSR csr_rev = make_csr(N, r.dst_d, r.src_d);
    const double d0 = cfg.get("s08.iso_d0");
    D.iso.assign(static_cast<size_t>(N_MODES) * N, 0.0);
    for (int m = 0; m < N_MODES; ++m) {
        const std::vector<double> w = mode_weight(r, g, m, lam_ref[m]);
        const std::vector<double> din = dijkstra(g.csr, w, H).dist, dout = dijkstra(csr_rev, w, H).dist;
        for (int64_t q = 0; q < N; ++q) {
            const double d = np_maximum(din[q], dout[q]);
            D.iso[static_cast<size_t>(m) * N + q] = std::isfinite(d) ? 1.0 - std::exp(-d / d0) : 1.0;
        }
    }
    const double* iso_daily = &D.iso[static_cast<size_t>(M_DAILY) * N];

    // ---- 次级极大的圈归属：商旅权重下最近的主中心（中心 id 按字母序：north_east, north_west, south）
    const int sorted_c[3] = {1, 0, 2};   // CENTER_KEYS 下标，按 id 排序后
    std::vector<int> circle_of_secondary;
    {
        const std::vector<double> w_tr = mode_weight(r, g, M_TRADE, lam_ref[M_TRADE]);
        std::vector<std::vector<double>> dc;
        for (int k = 0; k < 3; ++k) dc.push_back(dijkstra(g.csr, w_tr, {ce.node[sorted_c[k]]}).dist);
        for (int64_t s : ce.secondary) {
            int j = 0;
            for (int k = 1; k < 3; ++k)
                if (dc[k][s] < dc[j][s]) j = k;
            circle_of_secondary.push_back(j);   // 排序后的中心序号
        }
    }

    // ---- 特征表（build_traits）
    const std::vector<Slot> slots = read_slots(cfg);
    const double r_max = cfg.get("s08.resistance_max");
    auto mk_trait = [&](const std::string& tid, int si, int64_t origin_node, const std::string& kind, const std::string& origin_ref,
                        double origin_time) {
        const Slot& sl = slots[si];
        Rng rng = entity_rng(seed, 8, tid);
        const std::vector<double>& dh = cfg.list("s08.half_distance_days." + sl.mode);
        const double d_half = std::exp(rng.uniform(std::log(dh[0]), std::log(dh[1])));
        const std::vector<double>& rr = cfg.list("s08.resistance_range." + sl.tier);
        const double x = rng.uniform(rr[0], rr[1]);
        const double resistance = x < r_max ? x : r_max;   // min(r_max, x)
        Trait t;
        t.id = tid;
        t.slot = sl.id;
        t.slot_index = si;
        t.mode = sl.mode;
        t.resistance = pyround(resistance, 4);
        t.d_half_days = pyround(d_half, 3);
        t.lambda = pyround(std::log(2.0) / d_half, 6);
        t.origin_node = origin_node;
        t.origin = origin_ref;
        t.kind = kind;
        t.origin_time = origin_time;
        t.tier = sl.tier;
        return t;
    };
    std::vector<Trait>& traits = D.traits;
    const int n_manual = static_cast<int>(cfg.get("traits_manual.trait.n", 0.0));
    if (n_manual > 0) {
        D.has_reflect = false;
        for (int i = 0; i < n_manual; ++i) {
            const std::string pre = "traits_manual.trait." + std::to_string(i) + ".";
            const std::string sid = cfg.gets(pre + "slot");
            int si = -1;
            for (size_t k = 0; k < slots.size(); ++k)
                if (slots[k].id == sid) si = static_cast<int>(k);
            if (si < 0) throw std::invalid_argument("traits.toml: unknown slot " + sid);
            int64_t origin_node;
            std::string origin_ref;
            if (cfg.str.count(pre + "origin")) {
                origin_ref = cfg.gets(pre + "origin");
                int k = -1;
                for (int c = 0; c < 3; ++c)
                    if (origin_ref == CENTER_KEYS[c]) k = c;
                if (k < 0) throw std::invalid_argument("traits.toml: unknown center " + origin_ref);
                origin_node = ce.node[k];
            } else {
                origin_node = static_cast<int64_t>(cfg.get(pre + "origin"));
                origin_ref = std::to_string(origin_node);
            }
            Trait t = mk_trait(cfg.gets(pre + "id"), si, origin_node, "manual", origin_ref, cfg.get(pre + "origin_time", 0.0));
            if (cfg.has(pre + "resistance")) t.resistance = cfg.get(pre + "resistance");
            if (cfg.has(pre + "d_half_days")) {
                t.d_half_days = cfg.get(pre + "d_half_days");
                t.lambda = pyround(std::log(2.0) / t.d_half_days, 6);
            }
            if (cfg.str.count(pre + "mode")) t.mode = cfg.gets(pre + "mode");
            traits.push_back(t);
        }
    } else {
        // 主起源：每槽位 × 每主中心 1 个值
        for (size_t si = 0; si < slots.size(); ++si)
            for (int k = 0; k < 3; ++k) {
                const int c = sorted_c[k];
                traits.push_back(mk_trait(slots[si].id + "@" + CENTER_KEYS[c], static_cast<int>(si), ce.node[c], "main", CENTER_KEYS[c], 0.0));
            }
        // 次级起源：中 / 高阻力槽位每文明圈 k_sub 个
        const size_t k_sub = static_cast<size_t>(cfg.get("s08.k_sub"));
        const double delay = cfg.get("s08.sub_origin_delay_days");
        std::vector<std::vector<int64_t>> per_circle(3);
        for (size_t s = 0; s < ce.secondary.size(); ++s)
            if (per_circle[circle_of_secondary[s]].size() < k_sub) per_circle[circle_of_secondary[s]].push_back(ce.secondary[s]);
        for (size_t si = 0; si < slots.size(); ++si) {
            if (slots[si].tier == "low") continue;
            for (int k = 0; k < 3; ++k) {
                const std::string cid = CENTER_KEYS[sorted_c[k]];
                for (size_t j = 0; j < per_circle[k].size(); ++j) {
                    const std::string ref = "sub:" + cid + ":" + std::to_string(j);
                    traits.push_back(mk_trait(slots[si].id + "@" + ref, static_cast<int>(si), per_circle[k][j], "sub", ref, delay));
                }
            }
        }
        // 本地起源：高隔离连通分量（反射型：文化在死胡同里积累 → 孤岛文化）
        D.iso_thr = cfg.get("s08.iso_thr");
        const size_t min_n = static_cast<size_t>(cfg.get("s08.iso_min_component"));
        const int64_t n_local = static_cast<int64_t>(cfg.get("s08.local_slots_per_component", 3.0));
        std::vector<int64_t> node_ids, remap(N, -1);
        for (int64_t q = 0; q < N; ++q)
            if (iso_daily[q] >= D.iso_thr) {
                remap[q] = static_cast<int64_t>(node_ids.size());
                node_ids.push_back(q);
            }
        if (!node_ids.empty()) {
            std::vector<int64_t> es, ed;
            for (size_t e = 0; e < isl.src.size(); ++e)
                if (remap[isl.src[e]] >= 0 && remap[isl.dst[e]] >= 0) {
                    es.push_back(remap[isl.src[e]]);
                    ed.push_back(remap[isl.dst[e]]);
                }
            const std::vector<int64_t> comp = weak_components(static_cast<int64_t>(node_ids.size()), es, ed);
            int64_t nc = 0;
            for (int64_t x : comp) nc = std::max(nc, x + 1);
            std::vector<std::vector<int64_t>> members(nc);
            for (size_t k = 0; k < comp.size(); ++k) members[comp[k]].push_back(node_ids[k]);
            for (auto& m : members)
                if (m.size() >= min_n) D.reflect.push_back(m);
        }
        std::vector<int> mid_high;
        for (size_t si = 0; si < slots.size(); ++si)
            if (slots[si].tier != "low") mid_high.push_back(static_cast<int>(si));
        for (const auto& members : D.reflect) {
            const std::string compkey = "comp" + std::to_string(*std::min_element(members.begin(), members.end()));
            Rng rng = entity_rng(seed, 8, "local:" + compkey);
            std::vector<int64_t> picks =
                rng.choice_noreplace(static_cast<int64_t>(mid_high.size()), std::min<int64_t>(n_local, static_cast<int64_t>(mid_high.size())));
            int64_t origin_node = members[0];
            for (int64_t q : members)
                if (iso_daily[q] > iso_daily[origin_node]) origin_node = q;
            std::sort(picks.begin(), picks.end());
            for (int64_t pi : picks) {
                const int si = mid_high[pi];
                traits.push_back(mk_trait(slots[si].id + "@local:" + compkey, si, origin_node, "local", "local:" + compkey,
                                          f32(ce.arrival_yr[origin_node])));
            }
        }
    }
    const int64_t T = static_cast<int64_t>(traits.size());

    // ---- reach：每特征一次最短路（决策 1 方案 A；fast = 方案 C：每 (起源, 模式) 一棵参考 λ 树）
    const size_t TN = static_cast<size_t>(T) * N;
    D.reach.assign(TN, 0.0);
    D.C.assign(TN, 0.0);
    D.L.assign(TN, 0.0);
    const bool fast = cfg.get("s08.fast", 0.0) != 0.0;
    std::map<std::pair<int64_t, int>, std::pair<std::vector<double>, std::vector<double>>> tree_cache;
    const size_t E2 = r.cost.size();
    std::vector<std::vector<double>> cost_col(N_MODES, std::vector<double>(E2)), L_col(N_MODES, std::vector<double>(E2));
    for (int m = 0; m < N_MODES; ++m)
        for (size_t k = 0; k < E2; ++k) {
            cost_col[m][k] = r.cost_m[k * N_MODES + m];
            L_col[m][k] = g.L[k * N_MODES + m];
        }
    for (int64_t ti = 0; ti < T; ++ti) {
        const Trait& t = traits[ti];
        const int mi = mode_index(t.mode);
        double* reach = &D.reach[static_cast<size_t>(ti) * N];
        double* Ca = &D.C[static_cast<size_t>(ti) * N];
        double* La = &D.L[static_cast<size_t>(ti) * N];
        if (fast) {
            const auto key = std::make_pair(t.origin_node, mi);
            auto it = tree_cache.find(key);
            if (it == tree_cache.end()) {
                const std::vector<double> w = mode_weight(r, g, mi, lam_ref[mi]);
                const Paths pt = dijkstra(g.csr, w, {t.origin_node});
                it = tree_cache.emplace(key, std::make_pair(accumulate_along_tree(pt, cost_col[mi]), accumulate_along_tree(pt, L_col[mi]))).first;
            }
            const std::vector<double>&Cn = it->second.first, &Ln = it->second.second;
            for (int64_t q = 0; q < N; ++q) {
                const double c = nan_to_inf(Cn[q]), l = nan_to_inf(Ln[q]);
                reach[q] = std::exp(-(t.lambda * c + l));
                Ca[q] = c;
                La[q] = l;
            }
        } else {
            const std::vector<double> w = mode_weight(r, g, mi, t.lambda);
            const Paths pt = dijkstra(g.csr, w, {t.origin_node});
            const std::vector<double> Cn = accumulate_along_tree(pt, cost_col[mi]), Ln = accumulate_along_tree(pt, L_col[mi]);
            for (int64_t q = 0; q < N; ++q) {
                reach[q] = std::isfinite(pt.dist[q]) ? std::exp(-pt.dist[q]) : 0.0;
                Ca[q] = nan_to_inf(Cn[q]);
                La[q] = nan_to_inf(Ln[q]);
            }
        }
    }

    // ---- adopt：同槽位对称不动点 + 本地行 ε（隔离度越高本地越强）
    const double eps0 = cfg.get("s08.eps0"), eps_max = cfg.get("s08.eps_max");
    for (const Trait& t : traits)
        if (std::find(D.slots.begin(), D.slots.end(), t.slot) == D.slots.end()) D.slots.push_back(t.slot);
    std::sort(D.slots.begin(), D.slots.end());
    for (const std::string& s : D.slots)
        for (const Trait& t : traits)
            if (t.slot == s) {
                D.slot_mode.push_back(t.mode);   // setdefault：该槽位第一条特征的模式
                break;
            }
    D.strength.assign(TN, 0.0);
    D.adopt.assign(TN, 1.0);
    D.share.assign(TN, 0.0);
    D.conflict_by.assign(TN, -1);
    D.local_share.assign(D.slots.size() * static_cast<size_t>(N), 0.0);
    const double tol = cfg.get("s08.fp_tol");
    const int max_iter = static_cast<int>(cfg.get("s08.fp_max_iter"));
    for (size_t si = 0; si < D.slots.size(); ++si) {
        std::vector<int64_t> rows;
        for (int64_t ti = 0; ti < T; ++ti)
            if (traits[ti].slot == D.slots[si]) rows.push_back(ti);
        const int mi = mode_index(D.slot_mode[si]);
        const int V = static_cast<int>(rows.size()) + 1;
        std::vector<double> R(static_cast<size_t>(V) * N), rv(V);
        for (int k = 0; k + 1 < V; ++k) {
            std::copy(&D.reach[static_cast<size_t>(rows[k]) * N], &D.reach[static_cast<size_t>(rows[k]) * N] + N, &R[static_cast<size_t>(k) * N]);
            rv[k] = traits[rows[k]].resistance;
        }
        for (int64_t q = 0; q < N; ++q)
            R[static_cast<size_t>(V - 1) * N + q] = eps0 + (eps_max - eps0) * D.iso[static_cast<size_t>(mi) * N + q];
        rv[V - 1] = 0.0;
        std::vector<double> S;
        FixedPoint fp = fixed_point_slot(R, rv, V, N, tol, max_iter, S);
        fp.n_values = V - 1;
        D.fixed_point.push_back(fp);
        // conflict_argmax：同槽位其余行的 argmax
        std::vector<int> cb(static_cast<size_t>(V) * N);
        std::vector<double> total(N, 0.0);
        for (int64_t j = 0; j < N; ++j) {
            int i1 = 0;
            for (int i = 1; i < V; ++i)
                if (S[static_cast<size_t>(i) * N + j] > S[static_cast<size_t>(i1) * N + j]) i1 = i;
            int i2 = -1;   // 把 i1 那一行记 −inf 再取 argmax
            double b2 = 0.0;
            for (int i = 0; i < V; ++i) {
                const double x = i == i1 ? -INF : S[static_cast<size_t>(i) * N + j];
                if (i2 < 0 || x > b2) i2 = i, b2 = x;
            }
            for (int i = 0; i < V; ++i) cb[static_cast<size_t>(i) * N + j] = i == i1 ? i2 : i1;
            for (int i = 0; i < V; ++i) total[j] += S[static_cast<size_t>(i) * N + j];   // S.sum(axis=0)：逐行顺序加
        }
        for (int k = 0; k < V; ++k) {
            const bool local = k == V - 1;
            for (int64_t j = 0; j < N; ++j) {
                const size_t sk = static_cast<size_t>(k) * N + j;
                const double sh = S[sk] / total[j];
                if (local) {
                    D.local_share[si * N + j] = sh;
                    continue;
                }
                const size_t o = static_cast<size_t>(rows[k]) * N + j;
                D.strength[o] = S[sk];
                D.adopt[o] = R[sk] > 0 ? S[sk] / R[sk] : 1.0;
                D.share[o] = sh;
                const int c = cb[sk];
                D.conflict_by[o] = c == V - 1 ? -2 : static_cast<int32_t>(rows[c]);   // 本地行记 −2
            }
        }
    }
    return D;
}

}  // namespace skyisle::planet
