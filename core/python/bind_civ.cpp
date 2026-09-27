// 行星层 ⑤–⑨ 的绑定（行星计划 P6d）：各步（planet_stage5–9）、产物的数组形（前端写 npz / json：*_arrays）、从产物读回（*_from）、
// 第三层的 NodeInputs 补上 ⑨（node_polity），以及给 pytest 同输入对照的图算法公共件。
// 各步的产物是不透明对象（Barriers / Routes / Centers / Diffusion / Polity），前端按 run 与阶段 key 缓存在内存里，下一步直接吃；
// 缓存没命中的上游从 npz / json 读回。数组一律双精度原值，写 npz 时前端照 Python 版转 float32。
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <stdexcept>

#include "bind_util.hpp"
#include "skyisle/planet/civ.hpp"
#include "skyisle/planet/view.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace skyisle;
using namespace skyisle::planet;

namespace {

nb::list ilist(const std::vector<int64_t>& v) {
    nb::list l;
    for (int64_t x : v) l.append(x);
    return l;
}
std::vector<int64_t> ivec(nb::handle h) {
    std::vector<int64_t> out;
    for (nb::handle x : h) out.push_back(nb::cast<int64_t>(x));
    return out;
}

nb::dict routes_py(const Routes& R) {
    nb::dict d;
    const size_t E2 = R.cost.size();
    d["cost"] = arr(R.cost);
    d["cost_no_g"] = arr(R.cost_no_g);
    d["cost_m"] = arr2(R.cost_m, E2, N_MODES);
    d["flow"] = arr(R.flow);
    d["node_flow"] = arr(R.node_flow);
    d["src_d"] = arr(R.src_d);
    d["dst_d"] = arr(R.dst_d);
    d["und_id"] = arr(R.und_id);
    d["sources"] = arr(R.sources);
    d["hubs"] = arr(R.hubs);
    d["near_g"] = barr(R.near_g);
    d["n_sources"] = R.n_sources;
    nb::dict comp;
    for (int m = 0; m < N_MODES; ++m) {
        nb::dict c;
        c["n_components"] = R.n_components[m];
        c["largest"] = R.largest[m];
        comp[MODE_KEYS[m]] = c;
    }
    d["components"] = comp;
    return d;
}

}  // namespace

void bind_civ(nb::module_& m) {
    nb::class_<Barriers>(m, "Barriers", "⑤ 障碍（perm.npz）");
    nb::class_<Routes>(m, "Routes", "⑥ 航路（routes.npz + hubs.json）");
    nb::class_<Centers>(m, "Centers", "⑦ 文明中心、史前扩散、地区（centers.json + prehist.npz + regions.npz）");
    nb::class_<Diffusion>(m, "Diffusion", "⑧ 特征扩散（traits.resolved.json + fields.npz + iso.npz）");
    nb::class_<Polity>(m, "Polity", "⑨ 政治层（polity.npz + polities.json）");

    // ---------------------------------------------------------------- 各步
    m.def("planet_stage5", [](nb::handle cfg, const Planet& p, const Winds& w, const Islands& isl, const Climate& c) {
        Config tmp;
        const Config& cc = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage5(cc, p, w, isl, c);
    });
    m.def(
        "planet_stage6",
        [](nb::handle cfg, uint64_t seed, const Winds& w, const Islands& isl, const Climate& c, const Barriers& b, int threads) {
            Config tmp;
            const Config& cc = cfg_of(cfg, tmp);
            nb::gil_scoped_release rel;
            return stage6(cc, seed, w, isl, c, b, threads);
        },
        "cfg"_a, "seed"_a, "winds"_a, "islands"_a, "climate"_a, "barriers"_a, "threads"_a = 1);
    m.def("planet_stage7", [](nb::handle cfg, const Planet& p, const Islands& isl, const Climate& c, const Barriers& b, const Routes& r) {
        Config tmp;
        const Config& cc = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage7(cc, p, isl, c, b, r);
    });
    m.def("planet_stage8", [](nb::handle cfg, uint64_t seed, const Islands& isl, const Barriers& b, const Routes& r, const Centers& ce) {
        Config tmp;
        const Config& cc = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage8(cc, seed, isl, b, r, ce);
    });
    m.def("planet_stage9", [](nb::handle cfg, const Islands& isl, const Climate& c, const Barriers& b, const Routes& r, const Centers& ce) {
        Config tmp;
        const Config& cc = cfg_of(cfg, tmp);
        nb::gil_scoped_release rel;
        return stage9(cc, isl, c, b, r, ce);
    });

    // ---------------------------------------------------------------- 产物的形（前端写 npz / json）
    m.def("barriers_arrays", [](const Barriers& B) {
        nb::dict d;
        const size_t E = static_cast<size_t>(B.E);
        d["perm"] = arr2(B.perm, E, N_MODES);
        d["perm_no_g"] = arr2(B.perm_no_g, E, N_MODES);
        d["f"] = arr2(B.f, E, N_REGIONAL);
        d["g_blocked"] = barr(B.g_blocked);
        d["gap"] = arr2(B.gap, E, N_MODES);
        d["density_drop"] = arr2(B.density_drop, E, N_MODES);
        d["climb"] = arr2(B.climb, E, N_MODES);
        d["political"] = arr2(B.political, E, N_MODES);
        return d;
    });
    m.def("routes_arrays", [](const Routes& R) { return routes_py(R); });
    m.def("centers_arrays", [](const Centers& C) {
        nb::dict d;
        d["nodes"] = ilist(std::vector<int64_t>(C.node.begin(), C.node.end()));
        d["origin_node"] = C.origin_node;
        d["secondary"] = ilist(C.secondary);
        d["region_seeds"] = ilist(C.region_seeds);
        nb::list tr;
        for (const Trunk& t : C.trunks) {
            nb::dict x;
            x["key"] = t.key;
            x["reachable"] = t.reachable;
            x["cost_days"] = t.cost_days;
            x["path"] = ilist(t.path);
            tr.append(x);
        }
        d["trunks"] = tr;
        d["suit"] = arr(C.suit);
        d["dist_pre"] = arr(C.dist_pre);
        d["pred"] = arr(C.pred_pre);
        d["lineage"] = arr(C.lineage);
        d["arrival_yr"] = arr(C.arrival_yr);
        d["region"] = arr(C.region);
        d["n_lineages"] = C.n_lineages;
        return d;
    });
    m.def("diffusion_arrays", [](const Diffusion& D) {
        nb::dict d;
        nb::list traits;
        for (const Trait& t : D.traits) {
            nb::dict x;
            x["id"] = t.id;
            x["slot"] = t.slot;
            x["slot_index"] = t.slot_index;
            x["mode"] = t.mode;
            x["resistance"] = t.resistance;
            x["d_half_days"] = t.d_half_days;
            x["lambda"] = t.lambda;
            x["origin_node"] = t.origin_node;
            x["origin"] = t.origin;
            x["kind"] = t.kind;
            x["origin_time"] = t.origin_time;
            x["resistance_tier"] = t.tier;
            traits.append(x);
        }
        d["traits"] = traits;
        d["slots"] = D.slots;
        d["slot_mode"] = D.slot_mode;
        d["lambda_ref"] = std::vector<double>(D.lambda_ref.begin(), D.lambda_ref.end());
        nb::list fp;
        for (const FixedPoint& f : D.fixed_point) {
            nb::dict x;
            x["n_iter"] = f.n_iter;
            x["converged"] = f.converged;
            x["n_values"] = f.n_values;
            fp.append(x);
        }
        d["fixed_point"] = fp;
        d["has_reflect"] = D.has_reflect;
        d["iso_thr"] = D.iso_thr;
        nb::list rf;
        for (const auto& c : D.reflect) rf.append(ilist(c));
        d["reflect"] = rf;
        const size_t T = D.traits.size(), N = static_cast<size_t>(D.N);
        d["reach"] = arr2(D.reach, T, N);
        d["strength"] = arr2(D.strength, T, N);
        d["adopt"] = arr2(D.adopt, T, N);
        d["share"] = arr2(D.share, T, N);
        d["C"] = arr2(D.C, T, N);
        d["L"] = arr2(D.L, T, N);
        d["conflict_by"] = arr2(D.conflict_by, T, N);
        d["iso"] = arr2(D.iso, N_MODES, N);
        d["local_share"] = arr2(D.local_share, D.slots.size(), N);
        return d;
    });
    m.def("polity_arrays", [](const Polity& P) {
        nb::dict d;
        d["pop"] = arr(P.pop);
        d["state"] = arr(P.state);
        d["polity"] = arr(P.polity);
        d["kind"] = arr(P.kind);
        d["control"] = arr(P.control);
        d["dist_cap"] = arr(P.dist_cap);
        d["fief"] = arr(P.fief);
        d["realm"] = arr(P.realm);
        d["circle"] = arr(P.circle);
        d["capital"] = arr(P.capital);
        d["pop_state"] = arr(P.pop_state);
        nb::list states;
        for (const StateRec& s : P.states) {
            nb::dict x;
            x["capital"] = s.capital;
            x["circle"] = s.circle;
            x["regime"] = REGIME_KEYS[s.regime];
            x["n_nodes"] = s.n_nodes;
            x["pop"] = s.pop;
            x["dense_frac"] = s.dense_frac;
            x["radius"] = s.radius;
            x["n_direct"] = s.n_direct;
            x["n_fiefs"] = s.n_fiefs;
            x["overlord"] = s.overlord;
            x["vassals"] = ilist(s.vassals);
            nb::list nbs;
            for (const auto& kv : s.neighbors) nbs.append(nb::make_tuple(kv.first, kv.second));
            x["neighbors"] = nbs;
            x["annexed_by"] = s.annexed_by;
            x["annexed_years"] = s.annexed_years;
            x["is_suzerain"] = s.is_suzerain;
            states.append(x);
        }
        d["states"] = states;
        nb::list fl;
        for (const auto& f : P.fleets) fl.append(ilist(f));
        d["fleets"] = fl;
        d["fleet_pop"] = P.fleet_pop;
        d["fleet_circle"] = P.fleet_circle;
        d["tribes"] = ilist(P.tribes);
        d["suzerain"] = ilist(std::vector<int64_t>(P.suzerain.begin(), P.suzerain.end()));
        d["reformer"] = P.reformer;
        d["reformer_fallback"] = P.reformer_fallback;
        d["realm_pop"] = P.realm_pop;
        nb::list hist;
        for (const Annex& h : P.history) {
            nb::dict x;
            x["polity"] = h.polity;
            x["years_ago"] = h.years_ago;
            x["war_years"] = h.war_years;
            x["pop"] = h.pop;
            x["n_nodes"] = h.n_nodes;
            x["was_suzerain"] = h.was_suzerain;
            x["stage"] = STAGE_KEYS[h.stage];
            hist.append(x);
        }
        d["history"] = hist;
        nb::list fr;
        for (const Front& f : P.fronts) {
            nb::dict x;
            x["polity"] = f.polity;
            x["frontier_days"] = f.frontier_days;
            x["feasible"] = f.feasible;
            if (f.feasible) x["war_years_needed"] = f.war_years_needed;
            fr.append(x);
        }
        d["fronts"] = fr;
        d["open_a"] = ilist(P.open_a);
        d["open_b"] = P.open_b;
        d["open_c"] = ilist(P.open_c);
        d["open_d"] = P.open_d;
        d["n_cap1"] = P.n_cap1;
        d["n_attached"] = P.n_attached;
        return d;
    });

    // ---------------------------------------------------------------- 从产物读回（npz 的数组 dict / json 的 dict）
    m.def("barriers_from", [](nb::dict perm) {
        Barriers B;
        B.perm = dv(perm, "perm");
        B.E = static_cast<int64_t>(B.perm.size() / N_MODES);
        B.perm_no_g = dv(perm, "perm_no_g");
        B.f = dv(perm, "f_regional");
        B.g_blocked = anyvec<uint8_t>(perm["g_blocked"]);
        B.gap = dv(perm, "gap");
        B.density_drop = dv(perm, "density_drop");
        B.climb = dv(perm, "climb");
        B.political = dv(perm, "political");
        return B;
    });
    m.def("routes_from", [](nb::dict routes, nb::dict hubs) {
        Routes R;
        R.cost = dv(routes, "cost");
        R.cost_no_g = dv(routes, "cost_no_g");
        R.cost_m = dv(routes, "cost_m");
        R.flow = dv(routes, "flow");
        R.node_flow = dv(routes, "node_flow");
        R.src_d = anyvec<int64_t>(routes["src_d"]);
        R.dst_d = anyvec<int64_t>(routes["dst_d"]);
        R.und_id = anyvec<int64_t>(routes["und_id"]);
        R.sources = anyvec<int64_t>(routes["betweenness_sources"]);
        R.near_g.assign(R.node_flow.size(), 0);
        for (nb::handle h : nb::cast<nb::list>(hubs["hubs"])) {
            nb::dict x = nb::cast<nb::dict>(h);
            const int64_t q = nb::cast<int64_t>(x["node"]);
            R.hubs.push_back(q);
            if (nb::cast<bool>(x["near_g"])) R.near_g[q] = 1;
        }
        R.n_sources = nb::cast<int64_t>(hubs["n_sources"]);
        nb::dict comp = nb::cast<nb::dict>(hubs["components"]);
        for (int m = 0; m < N_MODES; ++m) {
            nb::dict c = nb::cast<nb::dict>(comp[MODE_KEYS[m]]);
            R.n_components[m] = nb::cast<int64_t>(c["n_components"]);
            R.largest[m] = nb::cast<int64_t>(c["largest"]);
        }
        return R;
    });
    m.def("centers_from", [](nb::dict centers, nb::dict prehist, nb::dict regions) {
        Centers C;
        nb::dict cs = nb::cast<nb::dict>(centers["centers"]);
        for (int k = 0; k < 3; ++k) C.node[k] = nb::cast<int64_t>(nb::cast<nb::dict>(cs[CENTER_KEYS[k]])["node"]);
        C.origin_node = nb::cast<int64_t>(centers["origin_node"]);
        for (nb::handle h : nb::cast<nb::list>(centers["secondary_peaks"])) C.secondary.push_back(nb::cast<int64_t>(nb::cast<nb::dict>(h)["node"]));
        C.region_seeds = ivec(centers["region_seeds"]);
        nb::dict tr = nb::cast<nb::dict>(centers["center_trunks"]);
        for (auto kv : tr) {
            Trunk t;
            t.key = nb::cast<std::string>(kv.first);
            nb::dict x = nb::cast<nb::dict>(kv.second);
            t.reachable = !x["cost_days"].is_none();
            t.cost_days = t.reachable ? nb::cast<double>(x["cost_days"]) : 0.0;
            t.path = ivec(x["path"]);
            C.trunks.push_back(std::move(t));
        }
        C.suit = dv(regions, "suitability");
        C.region = anyvec<int64_t>(regions["region"]);
        C.dist_pre = dv(prehist, "dist_pre");
        C.pred_pre = anyvec<int64_t>(prehist["pred"]);
        C.lineage = anyvec<int64_t>(prehist["lineage"]);
        C.arrival_yr = dv(prehist, "arrival_yr");
        int64_t nl = 0;
        for (int64_t x : C.lineage) nl = std::max(nl, x + 1);
        C.n_lineages = nl;
        return C;
    });
    m.def("polity_from", [](nb::dict pz, nb::dict polities) {
        Polity P;
        P.pop = dv(pz, "pop");
        P.state = anyvec<int32_t>(pz["state"]);
        P.polity = anyvec<int32_t>(pz["polity"]);
        P.kind = anyvec<int8_t>(pz["kind"]);
        P.control = dv(pz, "control");
        P.dist_cap = dv(pz, "dist_cap");
        P.fief = anyvec<int32_t>(pz["fief"]);
        P.realm = anyvec<int32_t>(pz["realm"]);
        P.circle = anyvec<int8_t>(pz["circle"]);
        P.capital = anyvec<int32_t>(pz["capital"]);
        P.pop_state = dv(pz, "pop_state");
        if (polities.contains("reformer")) {
            nb::dict r = nb::cast<nb::dict>(polities["reformer"]);
            P.reformer = nb::cast<int64_t>(r["polity"]);
            P.reformer_fallback = nb::cast<bool>(r["fallback"]);
        }
        return P;
    });

    // ---------------------------------------------------------------- 第三层：NodeInputs 补上 ⑨（人口、是不是都、是不是变法之国）
    m.def("node_polity", [](const Polity& P, int64_t node, nb::handle cfg) {
        Config tmp;
        const Config& cc = cfg_of(cfg, tmp);
        island::NodeInputs x;
        apply_polity(P, node, cc, x);
        nb::dict d;
        d["pop"] = x.pop;
        d["people_per_arable_km2"] = x.people_per_arable_km2;
        if (x.is_capital) {
            nb::dict c;
            c["state"] = x.state;
            c["state_pop"] = x.state_pop;
            c["reformer"] = x.reformer;
            d["capital"] = c;
        } else {
            d["capital"] = nb::none();
        }
        return d;
    });

    // ---------------------------------------------------------------- 公共件（pytest 同输入对照）
    m.def("graph_dijkstra", [](int64_t n, nb::handle src, nb::handle dst, ArrD1 w, nb::handle sources, nb::handle max_dist) {
        const CSR g = make_csr(n, anyvec<int64_t>(src), anyvec<int64_t>(dst));
        const std::vector<double> wv(w.data(), w.data() + w.size());
        const Paths p = dijkstra(g, wv, ivec(sources), max_dist.is_none() ? INF : nb::cast<double>(max_dist));
        return nb::make_tuple(arr(p.dist), arr(p.pred_node), arr(p.pred_edge));
    });
    m.def("graph_betweenness", [](int64_t n, nb::handle src, nb::handle dst, ArrD1 w, nb::handle sources, double c_min, int threads) {
        const CSR g = make_csr(n, anyvec<int64_t>(src), anyvec<int64_t>(dst));
        const std::vector<double> wv(w.data(), w.data() + w.size());
        return arr(betweenness_sampled(g, wv, static_cast<int64_t>(wv.size()), ivec(sources), c_min, 1e-9, threads));
    });
    m.def("graph_weak_components", [](int64_t n, nb::handle src, nb::handle dst) {
        return arr(weak_components(n, anyvec<int64_t>(src), anyvec<int64_t>(dst)));
    });
    m.def("rng_choice_noreplace_p", [](uint64_t seed, uint64_t stream, ArrD1 p, int64_t size) {
        Rng r = stage_rng(seed, stream);
        return arr(r.choice_noreplace_p(std::vector<double>(p.data(), p.data() + p.size()), size));
    });
}
