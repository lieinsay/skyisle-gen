// ⑨ 政治层（skyisle_gen/stages/s09_polity.py）：人口 → 核心实力 → 建邦（宗主先立、按实力择都、圈 control ≥ θ 的邑）→ 所有邑按控制力归属、
// 威慑距离外投附邻邑、再无则独邑 → 采邑树 → 船团与部落 → 文明圈与宗主 → 邦际接壤 → 附庸 → 变法之国 → 兼并史 → 政体类型 → 开局候选。
// 不读 height_m（原则乙）；每个节点都属于某个政体（原则己）。
//
// 照抄 Python 版的一处怪处：附庸判定里 `cap_dist[t].get(capitals[s])` 拿「都城的节点号」去查以「邦号」为键的表（DESIGN-NOTES 四点二十六）。
// 这是参考实现的行为，改它会让 seed42 的附庸变——先照抄，修不修由以后定。
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

#include "skyisle/planet/civ.hpp"

namespace skyisle::planet {

const char* const REGIME_KEYS[5] = {"reformed", "suzerain", "centralizable", "feudal", "city"};
const char* const STAGE_KEYS[6] = {"military", "administrative", "economic", "cultural", "linguistic", "identity"};

std::vector<double> population(const Config& cfg, const Islands& isl, const Climate& c) {
    const double p1 = cfg.get("shared.scale.people_per_arable_km2");
    const double full = cfg.get("s09.polity.precip_full"), floor_ = cfg.get("s09.polity.precip_floor");
    std::vector<double> pop(isl.n());
    for (size_t q = 0; q < pop.size(); ++q) {
        const double arable = f32(isl.area[q]) * f32(isl.arable_frac[q]);
        const double wet = clip(f32(c.i_precip[q]) / full, floor_, 1.0);
        pop[q] = p1 * arable * wet;
    }
    return pop;
}

namespace {

int consolidation_stage(double years, const std::vector<double>& th) {
    for (size_t k = 0; k < th.size(); ++k)
        if (years < th[k]) return static_cast<int>(k);
    return 5;
}

}  // namespace

Polity stage9(const Config& cfg, const Islands& isl, const Climate& c, const Barriers& b, const Routes& r, const Centers& ce) {
    const std::string P = "s09.polity.";
    const int64_t N = static_cast<int64_t>(isl.n());
    const int64_t E = static_cast<int64_t>(isl.src.size());
    const Directed g = directed(isl, b, r);
    Polity out;
    std::vector<int64_t> cls(N);
    std::vector<uint8_t> layered(N), eligible(N);
    for (int64_t q = 0; q < N; ++q) {
        cls[q] = isl.cls[q];
        layered[q] = isl.layered[q] != 0;
        eligible[q] = cls[q] == 0 || cls[q] == 1;   // 密接 / 中疏 才建邦；稀疏 = 船团，孤悬 = 部落
    }
    const std::vector<double> pop = population(cfg, isl, c);
    out.pop = pop;

    // ---- 控制权重（有向）：商旅成本 × 后勤倍率 + R0 × L_trade
    const double r0 = cfg.get(P + "control_radius_days");
    const double bridge_days = cfg.get("shared.ships.bridge_days");
    const double ship_mult = cfg.get(P + "ship_logistics_mult");
    std::vector<double> w_pol(2 * E);
    for (int64_t k = 0; k < 2 * E; ++k) {
        const double du = isl.dist_days[k < E ? k : k - E];
        const double logistics = du <= bridge_days ? 1.0 : ship_mult;
        w_pol[k] = r.cost_m[static_cast<size_t>(k) * N_MODES + M_TRADE] * logistics + r0 * g.L[static_cast<size_t>(k) * N_MODES + M_TRADE];
    }
    const CSR& csr = g.csr;
    const double theta = cfg.get(P + "control_min");
    const double ln_theta = std::log(1.0 / theta);

    // ---- 核心实力 S：控制范围内 Σ 人口 × 控制力（每邑一次有界搜索）
    std::vector<double> S(N, 0.0);
    const double bound0 = r0 * ln_theta;
    std::vector<double> buf;
    for (int64_t cn = 0; cn < N; ++cn) {
        if (!eligible[cn]) continue;
        const Paths pt = dijkstra(csr, w_pol, {cn}, bound0);
        buf.clear();
        for (int64_t q = 0; q < N; ++q)
            if (std::isfinite(pt.dist[q]) && eligible[q]) buf.push_back(pop[q] * std::exp(-pt.dist[q] / r0));
        S[cn] = np_sum(buf.data(), buf.size());
    }
    double s_ref = 1.0;
    {
        std::vector<double> se;
        for (int64_t q = 0; q < N; ++q)
            if (eligible[q]) se.push_back(S[q]);
        if (!se.empty()) s_ref = np_median(se);
    }
    const double beta = cfg.get(P + "radius_pop_exponent");
    const double rmin = cfg.get(P + "radius_mult_min"), rmax = cfg.get(P + "radius_mult_max");
    std::vector<double> radius(N);
    const double sden = std::max(s_ref, 1e-9);
    for (int64_t q = 0; q < N; ++q) radius[q] = r0 * clip(np_pow(S[q] / sden, beta), rmin, rmax);
    const double s_min = cfg.get(P + "capital_min_strength_frac") * s_ref;

    // ---- 宗主先立：三个文明中心是最早的国家，五百年后权力空心化 → 半径打折（不随核心实力放大）
    const std::vector<int64_t> main_nodes(ce.node.begin(), ce.node.end());
    const double suz_mult = cfg.get(P + "suzerain_radius_mult");
    for (int64_t cn : main_nodes)
        if (eligible[cn]) radius[cn] = r0 * suz_mult;
    // ---- 建邦 第一遍：按核心实力依次择都，圈 control ≥ θ 的未归属之邑
    std::vector<int64_t> claimed(N, -1);
    std::vector<int64_t> capitals;
    std::vector<int64_t> first;
    for (int64_t cn : main_nodes)
        if (eligible[cn]) first.push_back(cn);
    std::vector<int64_t> order(N);
    for (int64_t q = 0; q < N; ++q) order[q] = q;
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t bq) { return -S[a] < -S[bq]; });
    std::vector<int64_t> cand_list = first;
    cand_list.insert(cand_list.end(), order.begin(), order.end());
    for (int64_t cn : cand_list) {
        if (!eligible[cn] || claimed[cn] >= 0 || layered[cn] || S[cn] < s_min) continue;
        const int64_t sid = static_cast<int64_t>(capitals.size());
        capitals.push_back(cn);
        const Paths pt = dijkstra(csr, w_pol, {cn}, radius[cn] * ln_theta);
        for (int64_t q = 0; q < N; ++q)
            if (std::isfinite(pt.dist[q]) && eligible[q] && claimed[q] < 0) claimed[q] = sid;
    }
    const int64_t n_cap1 = static_cast<int64_t>(capitals.size());
    out.n_cap1 = n_cap1;
    std::set<int64_t> suz_sids;
    for (int64_t sid = 0; sid < n_cap1; ++sid)
        if (std::find(first.begin(), first.end(), capitals[sid]) != first.end()) suz_sids.insert(sid);
    std::vector<uint8_t> locked(N, 0);   // 王畿：宗主第一遍圈到的邑锁定给宗主
    for (int64_t q = 0; q < N; ++q) locked[q] = suz_sids.count(claimed[q]) > 0;

    // ---- 第二遍：所有邑按「谁的控制力最大」归属（威慑距离内）；记录都城间距离（附庸判定用）
    const double vassal_reach = cfg.get(P + "vassal_reach");
    std::vector<int64_t> state(N, -1), pred(N, -1);
    std::vector<double> control(N, 0.0), d_cap(N, INF);
    std::vector<std::map<int64_t, double>> cap_dist(n_cap1);   // 键 = 邦号（t 是 cap_arr 的下标）
    for (int64_t sid = 0; sid < n_cap1; ++sid) {
        const int64_t cn = capitals[sid];
        const double bound_w = radius[cn] * ln_theta;
        const Paths pt = dijkstra(csr, w_pol, {cn}, bound_w * vassal_reach);
        const bool suz = suz_sids.count(sid) > 0;
        for (int64_t q = 0; q < N; ++q) {
            const double d = pt.dist[q];
            const double ctrl = std::exp(-d / radius[cn]);
            if (!(std::isfinite(d) && eligible[q] && ctrl > control[q] + 1e-12)) continue;
            if (!suz && locked[q]) continue;
            state[q] = sid;
            control[q] = ctrl;
            d_cap[q] = d;
            pred[q] = pt.pred_node[q];
        }
        for (int64_t t = 0; t < n_cap1; ++t)
            if (std::isfinite(pt.dist[capitals[t]]) && t != sid) cap_dist[sid][t] = pt.dist[capitals[t]];
    }
    // 威慑距离外无人能及之邑：投附邻邑（商旅可通的候选边）所属之邦，取控制力最高的邻邑；迭代到不再变化
    std::vector<std::vector<int64_t>> inc(N);   // 每个节点的关联边（按边号）
    for (int64_t e = 0; e < E; ++e) {
        inc[isl.src[e]].push_back(e);
        if (isl.dst[e] != isl.src[e]) inc[isl.dst[e]].push_back(e);
    }
    std::vector<uint8_t> trade_ok(E);
    for (int64_t e = 0; e < E; ++e) trade_ok[e] = b.perm[static_cast<size_t>(e) * N_MODES + M_TRADE] > 0;
    std::vector<double> w_und(E);
    for (int64_t e = 0; e < E; ++e) w_und[e] = np_minimum(w_pol[e], w_pol[E + e]);
    int64_t n_attached = 0;
    while (true) {
        std::vector<int64_t> loose;
        for (int64_t q = 0; q < N; ++q)
            if (eligible[q] && state[q] < 0) loose.push_back(q);
        if (loose.empty()) break;
        bool changed = false;
        for (int64_t j : loose) {
            int64_t best_k = -1, best_o = -1;
            double best_v = 0.0;
            bool any_edge = false;
            for (int64_t e : inc[j]) {
                if (!trade_ok[e]) continue;
                any_edge = true;
                const int64_t o = isl.src[e] == j ? isl.dst[e] : isl.src[e];
                const double v = state[o] >= 0 ? control[o] * std::exp(-w_und[e] / r0) : -1.0;
                if (best_k < 0 || v > best_v) best_k = e, best_o = o, best_v = v;   // argmax：第一个最大者
            }
            if (!any_edge || state[best_o] < 0) continue;   // 没有商旅边，或邻邑都还没归属（最大者是 −1）
            const int64_t o = best_o;
            state[j] = state[o];
            control[j] = control[o] * std::exp(-w_und[best_k] / radius[capitals[state[o]]]);
            d_cap[j] = d_cap[o] + w_und[best_k];
            pred[j] = o;
            ++n_attached;
            changed = true;
        }
        if (!changed) break;
    }
    out.n_attached = n_attached;
    // 再无人能及者自成独邑
    for (int64_t q = 0; q < N; ++q) {
        if (!(eligible[q] && state[q] < 0)) continue;
        state[q] = static_cast<int64_t>(capitals.size());
        capitals.push_back(q);
        cap_dist.emplace_back();
        control[q] = 1.0;
        d_cap[q] = 0.0;
    }
    const int64_t n_states = static_cast<int64_t>(capitals.size());

    // ---- 采邑树：都城最短路树上 w > r_direct 的第一级子树
    const double r_direct = cfg.get(P + "direct_rule_days");
    std::vector<int64_t> fief(N, -1);
    for (int64_t j : argsort_stable(d_cap)) {
        const int64_t s = state[j];
        if (s < 0 || !std::isfinite(d_cap[j])) continue;
        if (d_cap[j] <= r_direct) {
            fief[j] = -1;
            continue;
        }
        const int64_t pj = pred[j];
        if (pj < 0 || state[pj] != s || fief[pj] < 0) fief[j] = j;
        else fief[j] = fief[pj];
    }

    // ---- 船团与部落：稀疏岛链按连通分量成团；孤悬散岛各自为部落
    std::vector<int64_t> kind(N, -1), polity(N, -1);
    for (int64_t q = 0; q < N; ++q)
        if (state[q] >= 0) kind[q] = KIND_STATE, polity[q] = state[q];
    int64_t n_pol = n_states;
    {
        std::vector<int64_t> ids, remap(N, -1);
        for (int64_t q = 0; q < N; ++q)
            if (cls[q] == 2) remap[q] = static_cast<int64_t>(ids.size()), ids.push_back(q);
        if (!ids.empty()) {
            std::vector<int64_t> es, ed;
            for (int64_t e = 0; e < E; ++e)
                if (remap[isl.src[e]] >= 0 && remap[isl.dst[e]] >= 0) es.push_back(remap[isl.src[e]]), ed.push_back(remap[isl.dst[e]]);
            const std::vector<int64_t> comp = weak_components(static_cast<int64_t>(ids.size()), es, ed);
            int64_t nc = 0;
            for (int64_t x : comp) nc = std::max(nc, x + 1);
            out.fleets.assign(nc, {});
            for (size_t k = 0; k < comp.size(); ++k) out.fleets[comp[k]].push_back(ids[k]);
            for (auto& m : out.fleets) {
                for (int64_t q : m) polity[q] = n_pol, kind[q] = KIND_FLEET;
                ++n_pol;
            }
        }
    }
    for (int64_t q = 0; q < N; ++q)
        if (kind[q] < 0) {
            out.tribes.push_back(q);
            polity[q] = n_pol++;
            kind[q] = KIND_TRIBE;
        }

    // ---- 文明圈（与 ⑧ 同规则：商旅权重下最近的主中心）与宗主
    const std::array<double, N_MODES> lam = lambda_ref(cfg);
    {
        const std::vector<double> w_tr = mode_weight(r, g, M_TRADE, lam[M_TRADE]);
        std::vector<std::vector<double>> dc;
        for (int64_t cn : main_nodes) dc.push_back(dijkstra(csr, w_tr, {cn}).dist);
        out.circle.resize(N);
        for (int64_t q = 0; q < N; ++q) {
            int bi = 0;
            for (int k = 1; k < 3; ++k)
                if (dc[k][q] < dc[bi][q]) bi = k;
            out.circle[q] = static_cast<int8_t>(bi);
        }
    }
    std::set<int64_t> suz_set;
    for (int k = 0; k < 3; ++k) {
        out.suzerain[k] = state[main_nodes[k]] >= 0 ? state[main_nodes[k]] : -1;
        if (out.suzerain[k] >= 0) suz_set.insert(out.suzerain[k]);
    }

    // ---- 邦的属性
    std::vector<std::vector<int64_t>> members_of(n_states);
    for (int64_t q = 0; q < N; ++q)
        if (state[q] >= 0) members_of[state[q]].push_back(q);
    std::vector<double> pop_state(n_states), dense_frac(n_states);
    std::vector<int64_t> n_nodes(n_states), circle_state(n_states);
    for (int64_t s = 0; s < n_states; ++s) {
        const auto& m = members_of[s];
        buf.clear();
        int64_t nd = 0;
        for (int64_t q : m) buf.push_back(pop[q]), nd += cls[q] == 0;
        pop_state[s] = np_sum(buf.data(), buf.size());
        n_nodes[s] = static_cast<int64_t>(m.size());
        circle_state[s] = out.circle[capitals[s]];
        dense_frac[s] = m.empty() ? 0.0 : static_cast<double>(nd) / static_cast<double>(m.size());
    }
    // 邦际接壤（商旅可通的候选边）
    std::vector<std::map<int64_t, int64_t>> adj(n_states);
    std::vector<uint8_t> both(E);
    std::vector<int64_t> sa(E), sb(E);
    for (int64_t e = 0; e < E; ++e) {
        sa[e] = state[isl.src[e]];
        sb[e] = state[isl.dst[e]];
        both[e] = sa[e] >= 0 && sb[e] >= 0 && sa[e] != sb[e] && trade_ok[e];
        if (both[e]) {
            ++adj[sa[e]][sb[e]];
            ++adj[sb[e]][sa[e]];
        }
    }
    // ---- 附庸：接壤、人口 ≥ ρ 倍、都在对方威慑距离内；取威慑（人口 × 控制力）最大者
    const double rho = cfg.get(P + "vassal_pop_ratio");
    std::vector<int64_t> overlord(n_states, -1);
    for (int64_t s = 0; s < n_states; ++s) {
        if (suz_set.count(s)) continue;
        int64_t best = -1;
        double best_v = 0.0;
        for (const auto& kv : adj[s]) {
            const int64_t t = kv.first;
            if (pop_state[t] < rho * pop_state[s]) continue;
            const auto it = cap_dist[t].find(capitals[s]);   // 照抄 Python 版：拿节点号查以邦号为键的表
            if (it == cap_dist[t].end()) continue;
            const double v = pop_state[t] * std::exp(-it->second / radius[capitals[t]]);
            if (v > best_v) best = t, best_v = v;
        }
        overlord[s] = best;
    }

    // ---- 变法之国（中心 ② 圈）
    out.reform_circle = cfg.gets(P + "reform_circle");
    int rc = -1;
    for (int k = 0; k < 3; ++k)
        if (out.reform_circle == CENTER_KEYS[k]) rc = k;
    if (rc < 0) throw std::invalid_argument("s09.polity.reform_circle: unknown center id " + out.reform_circle);
    std::vector<double> cost_tr(2 * E);
    for (int64_t k = 0; k < 2 * E; ++k) cost_tr[k] = r.cost_m[static_cast<size_t>(k) * N_MODES + M_TRADE];
    const std::vector<double> d_center = dijkstra(csr, cost_tr, {main_nodes[rc]}).dist;
    const double d_ref = cfg.get(P + "reform_center_distance_days");
    const int64_t override_cap = static_cast<int64_t>(cfg.get(P + "reformer_capital", -1.0));
    int64_t reformer = -1;
    if (override_cap >= 0 && state[override_cap] >= 0) {
        reformer = state[override_cap];
    } else {
        auto score_of = [&](int64_t s) {
            const int64_t cn = capitals[s];
            const double edge = 4.0 * dense_frac[s] * (1.0 - dense_frac[s]);   // 密接与中疏各半 = 「密接群岛的边缘」
            double far = 1.0;
            if (std::isfinite(d_center[cn])) {
                const double x = d_center[cn] / d_ref;
                far = x < 1.0 ? x : 1.0;   // min(1.0, x)
            }
            return pop_state[s] * (0.25 + edge) * far;
        };
        std::vector<int64_t> cands;
        for (int64_t s = 0; s < n_states; ++s)
            if (circle_state[s] == rc && !suz_set.count(s) && cls[capitals[s]] == 0 && n_nodes[s] >= 2) cands.push_back(s);
        if (cands.empty()) {
            out.reformer_fallback = true;
            for (int64_t s = 0; s < n_states; ++s)
                if (circle_state[s] == rc && !suz_set.count(s) && n_nodes[s] >= 3) cands.push_back(s);
        }
        double best = 0.0;
        for (int64_t s : cands) {   // max(key = (score, −s))：分数高者，同分取号小者
            const double v = score_of(s);
            if (reformer < 0 || v > best || (v == best && s < reformer)) reformer = s, best = v;
        }
    }
    out.reformer = reformer;

    // ---- 兼并史：只有变法之国能兼并；先易后难；消化阶段按 docs/04 §四
    const double years_ago = cfg.get(P + "reform_years_ago"), reform_dur = cfg.get(P + "reform_duration_years");
    const double mob = cfg.get(P + "mobilization_mult"), defend = cfg.get(P + "defender_advantage"), awe = cfg.get(P + "suzerain_awe");
    const double y_base = cfg.get(P + "war_years_base"), y_scale = cfg.get(P + "war_years_per_pop_ratio");
    const double y_per_day = cfg.get(P + "war_years_per_frontier_day");
    const size_t n_fronts = static_cast<size_t>(cfg.get(P + "active_fronts"));
    const std::vector<double>& stage_years = cfg.list(P + "stage_years");
    std::vector<int64_t> annexed_by(n_states, -1);
    std::vector<double> annexed_years(n_states, NaN);
    double realm_pop = 0.0;
    if (reformer >= 0) {
        std::set<int64_t> realm{reformer};
        realm_pop = pop_state[reformer];
        double elapsed = 0.0;
        const double budget = std::max(0.0, years_ago - reform_dur);
        std::vector<double> w_frontier(E);
        for (int64_t e = 0; e < E; ++e) w_frontier[e] = trade_ok[e] ? np_minimum(w_pol[e], w_pol[E + e]) : INF;
        struct Scored {
            double difficulty;
            int64_t t;
            double fc;
            bool feasible;
        };
        while (true) {
            std::set<int64_t> cand;
            for (int64_t s : realm)
                for (const auto& kv : adj[s])
                    if (!realm.count(kv.first) && circle_state[kv.first] == rc) cand.insert(kv.first);
            std::vector<Scored> scored;
            for (int64_t t : cand) {   // 前线成本：跨界最便宜的一条边（天）
                double fc = INF;
                bool any = false;
                for (int64_t e = 0; e < E; ++e) {
                    if (!both[e]) continue;
                    if ((sa[e] == t && realm.count(sb[e])) || (sb[e] == t && realm.count(sa[e]))) {
                        any = true;
                        fc = std::min(fc, w_frontier[e]);
                    }
                }
                if (!any) fc = 1.0;
                const double eff_def = pop_state[t] * defend * (suz_set.count(t) ? awe : 1.0);
                const bool feasible = realm_pop * mob >= eff_def;
                scored.push_back({eff_def * (1.0 + fc), t, fc, feasible});
            }
            std::sort(scored.begin(), scored.end(), [](const Scored& x, const Scored& y) {
                return x.difficulty != y.difficulty ? x.difficulty < y.difficulty : x.t < y.t;
            });
            std::vector<Scored> feas;
            for (const auto& x : scored)
                if (x.feasible) feas.push_back(x);
            if (feas.empty()) {
                for (size_t k = 0; k < scored.size() && k < n_fronts; ++k) out.fronts.push_back({scored[k].t, scored[k].fc, false, 0.0});
                break;
            }
            const int64_t t = feas[0].t;
            const double fc = feas[0].fc;
            const double ratio = pop_state[t] * (suz_set.count(t) ? awe : 1.0) / std::max(realm_pop, 1.0);
            const double war_years = y_base + y_scale * ratio + y_per_day * fc;
            if (elapsed + war_years > budget) {
                for (size_t k = 0; k < feas.size() && k < n_fronts; ++k) {
                    const int64_t tt = feas[k].t;
                    const double need = y_base + y_scale * pop_state[tt] / std::max(realm_pop, 1.0) + y_per_day * feas[k].fc;
                    out.fronts.push_back({tt, feas[k].fc, true, need});
                }
                break;
            }
            elapsed += war_years;
            realm.insert(t);
            realm_pop += pop_state[t];
            const double ya = years_ago - reform_dur - elapsed;
            annexed_by[t] = reformer;
            annexed_years[t] = ya;
            out.history.push_back({t, ya, war_years, pop_state[t], n_nodes[t], suz_set.count(t) > 0, consolidation_stage(ya, stage_years)});
            for (int64_t s = 0; s < n_states; ++s)   // 被并之邦的附庸改属兼并者
                if (overlord[s] == t) overlord[s] = reformer;
        }
    }
    out.realm_pop = realm_pop;

    // ---- 开局候选（docs/11 §九）：边陲小邦 / 正统核心 / 过渡带 / 兼并者
    {
        std::vector<uint8_t> touch_d(N, 0);
        const float thr = static_cast<float>(0.05);   // f_regional 从 npz 读（float32），与 Python 浮点比较按 float32（NEP 50）
        for (int64_t e = 0; e < E; ++e)
            if (static_cast<float>(b.f[static_cast<size_t>(e) * N_REGIONAL + 3]) > thr) touch_d[isl.src[e]] = touch_d[isl.dst[e]] = 1;
        std::vector<double> d_share(n_states, 0.0);
        for (int64_t s = 0; s < n_states; ++s) {
            const auto& m = members_of[s];
            int64_t k = 0;
            for (int64_t q : m) k += touch_d[q];
            d_share[s] = m.empty() ? 0.0 : static_cast<double>(k) / static_cast<double>(m.size());
        }
        std::vector<int64_t> pool, a_c;
        for (int64_t s = 0; s < n_states; ++s) {
            const bool in_circle = circle_state[s] == rc;
            if (in_circle && d_share[s] > 0 && n_nodes[s] >= 2) pool.push_back(s);
        }
        for (int64_t s : pool)
            if (annexed_by[s] < 0 && s != reformer) a_c.push_back(s);   // pool 已在圈内
        std::sort(a_c.begin(), a_c.end(), [&](int64_t x, int64_t y) { return pop_state[x] != pop_state[y] ? pop_state[x] < pop_state[y] : x < y; });
        std::vector<int64_t> c_c = pool;
        std::sort(c_c.begin(), c_c.end(), [&](int64_t x, int64_t y) { return d_share[x] != d_share[y] ? -d_share[x] < -d_share[y] : x < y; });
        out.open_a.assign(a_c.begin(), a_c.begin() + std::min<size_t>(3, a_c.size()));
        out.open_c.assign(c_c.begin(), c_c.begin() + std::min<size_t>(3, c_c.size()));
        out.open_b = out.suzerain[rc];
        out.open_d = reformer;
    }

    // ---- 产物
    out.states.resize(n_states);
    for (int64_t s = 0; s < n_states; ++s) {
        StateRec& x = out.states[s];
        const int64_t cn = capitals[s];
        x.capital = cn;
        x.circle = static_cast<int>(circle_state[s]);
        x.regime = s == reformer ? REG_REFORMED : suz_set.count(s) ? REG_SUZERAIN : n_nodes[s] == 1 ? REG_CITY : cls[cn] == 0 ? REG_CENTRALIZABLE : REG_FEUDAL;
        x.n_nodes = n_nodes[s];
        x.pop = pop_state[s];
        x.dense_frac = dense_frac[s];
        x.radius = radius[cn];
        std::set<int64_t> seats;
        for (int64_t q : members_of[s]) {
            if (fief[q] < 0) ++x.n_direct;
            else seats.insert(fief[q]);
        }
        x.n_fiefs = static_cast<int64_t>(seats.size());
        x.overlord = overlord[s];
        for (int64_t t = 0; t < n_states; ++t)
            if (overlord[t] == s) x.vassals.push_back(t);
        for (const auto& kv : adj[s]) x.neighbors.emplace_back(kv.first, kv.second);
        x.annexed_by = annexed_by[s];
        x.annexed_years = annexed_years[s];
        x.is_suzerain = suz_set.count(s) > 0;
    }
    for (const auto& m : out.fleets) {
        buf.clear();
        std::vector<int64_t> cnt(3, 0);
        for (int64_t q : m) buf.push_back(pop[q]), ++cnt[out.circle[q]];
        out.fleet_pop.push_back(np_sum(buf.data(), buf.size()));
        out.fleet_circle.push_back(static_cast<int>(std::max_element(cnt.begin(), cnt.end()) - cnt.begin()));   // bincount().argmax()
    }
    out.state.resize(N);
    out.polity.resize(N);
    out.fief.resize(N);
    out.realm.resize(N);
    out.kind.resize(N);
    out.control = control;
    out.dist_cap = d_cap;
    for (int64_t q = 0; q < N; ++q) {
        out.state[q] = static_cast<int32_t>(state[q]);
        out.polity[q] = static_cast<int32_t>(polity[q]);
        out.fief[q] = static_cast<int32_t>(fief[q]);
        out.kind[q] = static_cast<int8_t>(kind[q]);
        // 兼并后的本朝：邦 → 兼并者（没被并就是自己）；船团 / 部落 → 自己的政体号
        out.realm[q] = static_cast<int32_t>(state[q] >= 0 ? (annexed_by[state[q]] >= 0 ? annexed_by[state[q]] : state[q]) : polity[q]);
    }
    out.capital.assign(capitals.begin(), capitals.end());
    out.pop_state = pop_state;
    return out;
}

void apply_polity(const Polity& pol, int64_t node, const Config& cfg, island::NodeInputs& inp) {
    inp.pop = f32(pol.pop[node]);                                   // polity.npz 的 pop（float32）
    inp.people_per_arable_km2 = cfg.get("shared.scale.people_per_arable_km2");
    inp.is_capital = false;
    const int64_t st = pol.state[node];
    if (st < 0 || pol.capital[st] != node) return;
    std::vector<float> v;                                           // pop[state == st].sum()：float32 成对求和
    for (size_t q = 0; q < pol.pop.size(); ++q)
        if (pol.state[q] == st) v.push_back(static_cast<float>(pol.pop[q]));
    inp.is_capital = true;
    inp.state = st;
    inp.state_pop = static_cast<double>(np_sum_f32(v.data(), v.size()));
    inp.reformer = pol.reformer == pol.polity[node];
}

Society run_society(const Config& cfg, uint64_t seed, const World& w, int upto, bool skip_diffusion, int threads) {
    Society s;
    if (upto < 5) return s;
    s.barriers = stage5(cfg, w.planet, w.winds, w.islands, w.climate);
    if (upto < 6) return s;
    s.routes = stage6(cfg, seed, w.winds, w.islands, w.climate, s.barriers, threads);
    if (upto < 7) return s;
    s.centers = stage7(cfg, w.planet, w.islands, w.climate, s.barriers, s.routes);
    if (upto >= 8 && !skip_diffusion) s.diffusion = stage8(cfg, seed, w.islands, s.barriers, s.routes, s.centers);
    if (upto < 9) return s;
    s.polity = stage9(cfg, w.islands, w.climate, s.barriers, s.routes, s.centers);
    return s;
}

}  // namespace skyisle::planet
