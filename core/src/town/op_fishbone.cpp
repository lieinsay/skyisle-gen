// 形态算子「鱼骨 / 井字街村」（PLAN-TOWN 6.2；华北）：
//   1. 候选街网：过村心、顺整村朝向的横街（主街 + 每隔一个街坊一条平行的街，缓弯），每个街坊各自抽间距的竖巷（不对齐 → 丁字口），过村心的一条竖巷是主街（十字街）；
//      过崖缘 / 虚空、过宽水、纵坡太陡的边不要。
//   2. 沿每条边两侧切地块：模板按权重挑、尺寸按模板抽，朝向按朝向链（朝阳 + 顺街），贴街摆；地块不压任何候选路。
//   3. 公共宅院（庙）先占位；然后一户一户落位：沿路离村心近、直线近、兴趣高、离父户近、与已有宅院贴着的先。
//   4. 用到的路：每块落了宅院的地块，沿最短路回到村心的边都用上；两端都已通的巷也用上（村里的巷是通的）。
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "plan_work.hpp"
#include "skyisle/town/raster.hpp"

namespace skyisle::town {

namespace {

V2 left_of(V2 t) { return {-t.y, t.x}; }

struct Hash {
    double cell;
    std::unordered_map<int64_t, std::vector<int>> m;
    int64_t key(int a, int b) const { return (static_cast<int64_t>(a) << 32) ^ static_cast<uint32_t>(b); }
    void add(V2 p, int id) { m[key(static_cast<int>(std::floor(p.x / cell)), static_cast<int>(std::floor(p.y / cell)))].push_back(id); }
    template <class F>
    void near(V2 p, double r, F f) const {
        const int a0 = static_cast<int>(std::floor((p.x - r) / cell)), a1 = static_cast<int>(std::floor((p.x + r) / cell));
        const int b0 = static_cast<int>(std::floor((p.y - r) / cell)), b1 = static_cast<int>(std::floor((p.y + r) / cell));
        for (int a = a0; a <= a1; ++a)
            for (int b = b0; b <= b1; ++b) {
                auto it = m.find(key(a, b));
                if (it != m.end())
                    for (int id : it->second) f(id);
            }
    }
};

// 一条边在地面上走得通吗：不出窗口、不过虚空 / 崖缘、过水的每一段 ≤ 最长桥、纵坡（4 m 一量）不超过上限的 1.3 倍
bool edge_ok(const Work& w, const std::vector<V2>& line, int& bridges) {
    const Site& s = w.s;
    bridges = 0;
    const double L = polyline_length(line);
    const double step = std::max(0.5 * s.res_m, 0.5);
    double prev_h = NaN, prev_at = 0.0;
    for (double at = 0.0; at <= L + 1e-9; at += step) {
        std::vector<V2> q = subline(line, at, std::min(L, at + 1e-6));
        const V2 p = q.empty() ? line.back() : q.front();
        int i, j;
        if (!s.cell_of(p, i, j) || s.sky(i, j) || s.edge(i, j) || s.water(i, j) == WATER_LAKE || s.water(i, j) == WATER_SEA) return false;
        double h = s.height(i, j);
        if (s.water(i, j) != WATER_NONE) h = std::isfinite(s.water_level(i, j)) ? std::max(h, static_cast<double>(s.water_level(i, j)) + 0.8) : h;
        if (at - prev_at >= 4.0 || !std::isfinite(prev_h)) {
            if (std::isfinite(prev_h) && std::fabs(h - prev_h) / (at - prev_at) > 1.3 * w.st.max_grade) return false;
            prev_h = h, prev_at = at;
        }
    }
    for (auto [a, b] : water_runs(s, line)) {
        if (b - a > w.st.bridge_max_m) return false;
        ++bridges;
    }
    return true;
}

// 从路边沿 n_out 往外走，到别的候选路（不是自己这条）为止的空地进深；走满 max 返回 max
double free_depth(const Site& s, const Mask& cand_road, V2 p, V2 n_out, double start, double max) {
    const double step = 0.5 * s.res_m;
    bool left_own = false;
    for (double a = start; a <= max; a += step) {
        int i, j;
        if (!s.cell_of(p + n_out * a, i, j)) return a;
        const bool road = cand_road(i, j) != 0;
        if (!road) left_own = true;
        else if (left_own) return a - start;
    }
    return max;
}

struct Street {
    double v;             // 横街：离村心的前后偏移；竖巷：所在的横向位置 u
    int cls;
    double width;
    int id;
};

}  // namespace

void op_fishbone(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("fishbone");
    const V2 C = w.center;
    const V2 nv = bearing_vec(w.facing);     // v 轴：宅院的前沿方向
    const V2 tu = bearing_right(w.facing);   // u 轴：横街走向
    const double U = st.fb_extent * w.R + 40.0, V = U;
    const uint64_t wseed = w.req.seed ^ 0x9e3779b97f4a7c15ULL;

    // ---- 横街的前后位置
    std::vector<double> vs = {0.0};
    for (double v = 0.0; v < V;) vs.push_back(v += st.fb_street_spacing.sample(r));
    for (double v = 0.0; v > -V;) vs.push_back(v -= st.fb_street_spacing.sample(r));
    std::sort(vs.begin(), vs.end());
    const int K = static_cast<int>(vs.size());
    const int k0 = static_cast<int>(std::find(vs.begin(), vs.end(), 0.0) - vs.begin());
    int sid = 0;
    std::vector<Street> hs(K);
    for (int k = 0; k < K; ++k) {
        const int cls = k == k0 ? RC_MAIN : RC_STREET;
        hs[k] = {vs[k], cls, road_width(w, cls, r), sid++};
    }
    auto street_pt = [&](int k, double u) {
        const double dv = st.fb_warp_m * fbm(wseed + static_cast<uint64_t>(k) * 7919ULL, u, 1000.0 * k, st.fb_warp_wl, 2);
        return C + tu * u + nv * (vs[k] + dv);
    };
    // ---- 每个街坊（横街 k 与 k + 1 之间）的竖巷
    struct Lane {
        int k;          // 在街坊 k 里（连横街 k 与 k + 1）
        double u;
        int street;     // 同一条（主街的竖巷跨各街坊用同一个 id）
        int cls;
        double width;
    };
    std::vector<Lane> lanes;
    const int main_cross_id = sid++;
    const double main_cross_w = road_width(w, RC_MAIN, r);
    for (int k = 0; k + 1 < K; ++k) {
        if (st.fb_main_cross) lanes.push_back({k, 0.0, main_cross_id, RC_MAIN, main_cross_w});
        const double q0 = st.fb_lane_spacing.sample(r);
        for (double u = -U + r.uniform(0.0, q0); u < U; u += st.fb_lane_spacing.sample(r)) {
            if (st.fb_main_cross && std::fabs(u) < 0.5 * st.fb_lane_spacing.lo) continue;
            lanes.push_back({k, u, sid++, RC_LANE, road_width(w, RC_LANE, r)});
        }
    }
    // ---- 路网：横街按路口切段，竖巷一段一条
    Network& net = w.net;
    std::vector<std::vector<std::pair<double, int>>> cuts(K);   // 每条横街上的（u，节点）
    for (int k = 0; k < K; ++k) {
        std::vector<double> us = {-U, U};
        for (const Lane& l : lanes)
            if (l.k == k || l.k + 1 == k) us.push_back(l.u);
        std::sort(us.begin(), us.end());
        for (double u : us)
            if (cuts[k].empty() || u - cuts[k].back().first > 1.0) cuts[k].push_back({u, net.add_node(street_pt(k, u))});
    }
    auto node_at = [&](int k, double u) {
        for (auto& [uu, n] : cuts[k])
            if (std::fabs(uu - u) <= 1.0) return n;
        return -1;
    };
    // 边先收集，再逐条查地面
    struct Cand {
        int a, b, cls, street;
        double width;
        std::vector<V2> line;
    };
    std::vector<Cand> cand;
    for (int k = 0; k < K; ++k)
        for (size_t q = 0; q + 1 < cuts[k].size(); ++q) {
            const double ua = cuts[k][q].first, ub = cuts[k][q + 1].first;
            std::vector<V2> line;
            const int n = std::max(1, static_cast<int>(std::ceil((ub - ua) / 6.0)));
            for (int m = 0; m <= n; ++m) line.push_back(street_pt(k, ua + (ub - ua) * m / n));
            cand.push_back({cuts[k][q].second, cuts[k][q + 1].second, hs[k].cls, hs[k].id, hs[k].width, line});
        }
    std::sort(lanes.begin(), lanes.end(), [](const Lane& a, const Lane& b) { return a.street != b.street ? a.street < b.street : a.k < b.k; });
    for (const Lane& l : lanes) {
        const int a = node_at(l.k, l.u), b = node_at(l.k + 1, l.u);
        if (a < 0 || b < 0) continue;
        const V2 pa = net.nodes[a].p, pb = net.nodes[b].p;
        const V2 mid = lerp(pa, pb, 0.5) + tu * (1.2 * fbm(wseed + 17, l.u, 50.0 * l.k, 60.0, 1));
        cand.push_back({a, b, l.cls, l.street, l.width, {pa, mid, pb}});
    }
    for (Cand& c : cand) {
        int nb = 0;
        if (!edge_ok(w, c.line, nb)) continue;
        double cost = c.cls == RC_MAIN ? st.g_main_discount : c.cls == RC_STREET ? 1.0 : 1.1;
        const double L = polyline_length(c.line);
        if (nb > 0 && L > 0) cost += 15.0 * nb / L;
        const int e = net.add_edge(c.a, c.b, c.line, c.cls, c.width, cost, c.street);
        net.edges[e].bridges = nb;
    }
    // 村心的节点：离村心最近、接着边的
    double bd = INF;
    for (int n = 0; n < static_cast<int>(net.nodes.size()); ++n)
        if (!net.nodes[n].edges.empty() && len(net.nodes[n].p - C) < bd) bd = len(net.nodes[n].p - C), w.root = n;
    if (w.root < 0) {
        organic_fill(w, 0);
        return;
    }
    std::vector<int> parent;
    const std::vector<double> dn = net.dijkstra(w.root, &parent);

    // ---- 候选路的光栅（地块一律不压）
    Mask cand_road(s.H, s.W, 0);
    for (const NetEdge& e : net.edges)
        if (std::isfinite(dn[e.a]) || std::isfinite(dn[e.b])) raster_line(s, e.line, e.width, [&](int i, int j) { cand_road(i, j) = 1; });

    // ---- 切地块
    auto node_hw = [&](int n, int except) {
        double h = 0.0;
        for (int e : net.nodes[n].edges)
            if (e != except) h = std::max(h, 0.5 * net.edges[e].width);
        return h;
    };
    std::vector<Slot>& slots = w.slots;
    for (int e = 0; e < static_cast<int>(net.edges.size()); ++e) {
        const NetEdge& E = net.edges[e];
        if (!std::isfinite(dn[E.a]) && !std::isfinite(dn[E.b])) continue;
        for (int sgn : {1, -1}) {
            double sp = node_hw(E.a, e) + st.setback.sample(r) + 0.3;
            const double s_end = E.len - node_hw(E.b, e) - 0.3;
            for (int guard = 0; guard < 400 && sp < s_end; ++guard) {
                const int tmpl = pick_template(w, r);
                const TemplateSpec& T = st.templates[tmpl];
                double Wd = T.plot_w.sample(r), Dp = T.plot_d.sample(r);
                const double sb = st.setback.sample(r);
                V2 tan;
                V2 p = net.point_at(e, sp, &tan);
                Slot probe = slot_along(w, p, tan, left_of(tan) * sgn, 0.5 * E.width, tmpl, Wd, Dp, sb);
                // 街坊里前后两排各占一半进深：离对面那条路的空地进深的一半就是这一排最深能到哪
                {
                    V2 tm;
                    const V2 pm = net.point_at(e, sp + probe.half_t, &tm);
                    const double half = 0.5 * (free_depth(s, cand_road, pm, left_of(tm) * sgn, 0.5 * E.width + 0.3, 90.0) - 0.6);
                    const V2 fv = bearing_vec(probe.box.facing);
                    const bool deep_along_n = std::fabs(dot(fv, left_of(tm))) > 0.7;   // 进深方向顺着 n（前 / 后临街）
                    const double lo = 0.85 * (deep_along_n ? T.plot_d.lo : T.plot_w.lo);
                    double& dim = deep_along_n ? Dp : Wd;
                    if (dim > half) {
                        if (half < lo) {
                            sp += 3.0;
                            continue;
                        }
                        dim = half;
                        probe = slot_along(w, p, tan, left_of(tan) * sgn, 0.5 * E.width, tmpl, Wd, Dp, sb);
                    }
                }
                if (sp + 2.0 * probe.half_t > s_end) break;
                p = net.point_at(e, sp + probe.half_t, &tan);
                Slot sl = slot_along(w, p, tan, left_of(tan) * sgn, 0.5 * E.width, tmpl, Wd, Dp, sb);
                sl.edge = e;
                sl.cls = E.cls;
                sl.s0 = sp, sl.s1 = sp + 2.0 * sl.half_t, sl.s_acc = sp + sl.half_t;
                slot_ground(w, sl, &cand_road);
                sl.vacant = r.random() < st.fb_vacant_p;
                sl.noise = r.uniform(-1.0, 1.0);
                sl.d_net = std::min(dn[E.a] + sl.s_acc * E.cost, dn[E.b] + (E.len - sl.s_acc) * E.cost);
                slots.push_back(sl);
                sp = sl.s1 + st.plot_gap.sample(r);
            }
        }
    }
    // 冲突（相交）与贴着（外扩 1 m 相交但本身不交）
    Hash hash{32.0, {}};
    for (int k = 0; k < static_cast<int>(slots.size()); ++k) hash.add(slots[k].box.c, k);
    for (int k = 0; k < static_cast<int>(slots.size()); ++k) {
        Slot& a = slots[k];
        if (!a.valid) continue;
        hash.near(a.box.c, 45.0, [&](int q) {
            if (q == k || !slots[q].valid) return;
            if (overlap(a.box, slots[q].box, -0.1)) a.conflicts.push_back(q);
            else if (overlap(a.box, slots[q].box, 1.2)) a.touch.push_back(q);
        });
    }
    auto take = [&](int k, int comp) {
        slots[k].taken = true;
        slots[k].comp = comp;
        for (int q : slots[k].conflicts) slots[q].taken = true;
    };

    // ---- 公共宅院（庙）：候选 = 各边两端（靠路口）的一块，按功能的模板定尺寸
    {
        std::vector<Slot> pc;
        for (const FuncSpec& f : st.funcs) {
            if (f.mode != "compound") continue;
            const int tmpl = st.template_index(f.tmpl);
            const TemplateSpec& T = st.templates[tmpl];
            for (int e = 0; e < static_cast<int>(net.edges.size()); ++e) {
                const NetEdge& E = net.edges[e];
                if (!std::isfinite(dn[E.a]) || len(lerp(net.nodes[E.a].p, net.nodes[E.b].p, 0.5) - C) > 1.5 * w.R) continue;
                for (int sgn : {1, -1})
                    for (int end = 0; end < 2; ++end) {
                        const double Wd = T.plot_w.sample(r), Dp = T.plot_d.sample(r);
                        V2 tan;
                        const double sp0 = end == 0 ? node_hw(E.a, e) + 0.3 : 0.0;
                        Slot probe = slot_along(w, net.point_at(e, std::max(0.0, sp0), &tan), tan, left_of(tan) * sgn, 0.5 * E.width, tmpl, Wd, Dp, 0.0);
                        const double sp = end == 0 ? sp0 : E.len - node_hw(E.b, e) - 0.3 - 2.0 * probe.half_t;
                        if (sp < 0.0 || sp + 2.0 * probe.half_t > E.len) continue;
                        const V2 p = net.point_at(e, sp + probe.half_t, &tan);
                        Slot sl = slot_along(w, p, tan, left_of(tan) * sgn, 0.5 * E.width, tmpl, Wd, Dp, 0.0);
                        sl.edge = e;
                        sl.cls = E.cls;
                        sl.s0 = sp, sl.s1 = sp + 2.0 * sl.half_t, sl.s_acc = sp + sl.half_t;
                        if (!slot_ground(w, sl, &cand_road)) continue;
                        sl.d_net = std::min(dn[E.a] + sl.s_acc * E.cost, dn[E.b] + (E.len - sl.s_acc) * E.cost);
                        pc.push_back(sl);
                    }
            }
        }
        const size_t before = w.plan.compounds.size();
        place_public_compounds(w, pc);
        for (size_t ci = before; ci < w.plan.compounds.size(); ++ci) {
            const Obb& b = w.plan.compounds[ci].plot;
            hash.near(b.c, 45.0, [&](int q) {
                if (overlap(b, slots[q].box, -0.1)) slots[q].taken = true;
            });
        }
        // 公共宅院的地块也按 slot 记进来，好让路接到它
        for (size_t ci = before; ci < w.plan.compounds.size(); ++ci) {
            for (Slot& sl : pc)
                if (len(sl.box.c - w.plan.compounds[ci].plot.c) < 1e-6) {
                    sl.taken = true;
                    sl.comp = static_cast<int>(ci);
                    slots.push_back(sl);
                    break;
                }
        }
    }

    // ---- 一户一户落位
    const double R = w.R;
    size_t g = 0;
    for (; g < w.hh_groups.size(); ++g) {
        const int h0 = w.hh_groups[g][0];
        const int par = w.plan.households[h0].parent;
        const int pc = par >= 0 ? w.plan.households[par].compound : -1;
        const V2 pp = pc >= 0 ? w.plan.compounds[pc].plot.c : C;
        double best = -INF;
        int arg = -1;
        for (int k = 0; k < static_cast<int>(slots.size()); ++k) {
            const Slot& sl = slots[k];
            if (!sl.valid || sl.vacant || sl.taken || !std::isfinite(sl.d_net)) continue;
            int contact = 0;
            for (int q : sl.touch) contact += slots[q].comp >= 0 && w.plan.compounds[slots[q].comp].kind == "house" ? 1 : 0;
            double sc = -(st.g_net * sl.d_net + st.g_dist * len(sl.box.c - C)) / R + st.g_interest * sl.interest +
                        st.g_contact * std::min(contact, 3) + st.g_noise * sl.noise;
            if (pc >= 0) sc -= st.g_kin * len(sl.box.c - pp) / R;
            if (sc > best) best = sc, arg = k;
        }
        if (arg < 0) break;
        const int ci = commit_compound(w, slots[arg], "house", "dwelling", "", w.hh_groups[g]);
        take(arg, ci);
    }

    // ---- 用到的路
    for (const Slot& sl : slots) {
        if (sl.comp < 0) continue;
        NetEdge& E = net.edges[sl.edge];
        int from;
        if (dn[E.a] + sl.s_acc * E.cost <= dn[E.b] + (E.len - sl.s_acc) * E.cost) {
            E.used_a = std::max(E.used_a, std::min(E.len, sl.s1));
            from = E.a;
        } else {
            E.used_b = std::max(E.used_b, std::min(E.len, E.len - sl.s0));
            from = E.b;
        }
        for (int n = from; parent[n] >= 0;) {
            NetEdge& P = net.edges[parent[n]];
            P.used_a = P.len;
            n = P.a == n ? P.b : P.a;
        }
    }
    std::vector<uint8_t> reached(net.nodes.size(), 0);
    auto mark_reached = [&]() {
        std::fill(reached.begin(), reached.end(), 0);
        for (const NetEdge& E : net.edges) {
            const bool full = E.used_a + E.used_b >= E.len - 1e-6;
            if (E.used_a > 0.0 || full) reached[E.a] = 1;
            if (E.used_b > 0.0 || full) reached[E.b] = 1;
        }
    };
    mark_reached();
    for (NetEdge& E : net.edges)
        if (E.cls != RC_STREET && E.used_a + E.used_b < E.len - 1e-6 && reached[E.a] && reached[E.b]) {
            // 竖巷两头都通了：整条用上（村里的巷是通的）；两边都没有宅院的不要
            bool flank = false;
            for (const Slot& sl : slots)
                if (sl.comp >= 0 && len(sl.box.c - lerp(net.nodes[E.a].p, net.nodes[E.b].p, 0.5)) < 0.5 * E.len + 25.0) {
                    flank = true;
                    break;
                }
            if (flank) E.used_a = E.len;
        }

    // ---- 连成折线进 plan.roads：同一条街的边按次序接
    std::vector<std::vector<int>> by_street(static_cast<size_t>(sid));
    for (int e = 0; e < static_cast<int>(net.edges.size()); ++e) by_street[net.edges[e].street].push_back(e);
    for (auto& es : by_street) {
        std::vector<V2> cur;
        int cls = RC_LANE;
        double width = 3.0;
        auto flush = [&]() {
            if (cur.size() >= 2) add_road(w, cur, cls, width);
            cur.clear();
        };
        for (int e : es) {
            const NetEdge& E = net.edges[e];
            cls = E.cls, width = E.width;
            const bool full = E.used_a + E.used_b >= E.len - 1e-6;
            if (full) {
                for (const V2& p : E.line)
                    if (cur.empty() || len(cur.back() - p) > 1e-6) cur.push_back(p);
                continue;
            }
            if (E.used_a > 0.0) {
                for (const V2& p : subline(E.line, 0.0, E.used_a))
                    if (cur.empty() || len(cur.back() - p) > 1e-6) cur.push_back(p);
            }
            flush();
            if (E.used_b > 0.0) cur = subline(E.line, E.len - E.used_b, E.len);
        }
        flush();
    }

    // ---- 外排的巷伸进田里成田间道
    {
        Rng rx = w.rng("fishbone.fieldpaths");
        for (int e = 0; e < static_cast<int>(net.edges.size()); ++e) {
            const NetEdge& E = net.edges[e];
            if (E.cls == RC_STREET || !(E.used_a + E.used_b >= E.len - 1e-6)) continue;
            for (int end : {E.a, E.b}) {
                // 这个节点外面（沿巷的方向）没有用到的边：伸出去
                bool outward_used = false;
                for (int e2 : net.nodes[end].edges)
                    if (e2 != e && net.edges[e2].street == E.street && net.used(e2)) outward_used = true;
                if (outward_used || rx.random() > st.fb_lane_extend_p) continue;
                const V2 p = net.nodes[end].p, q = net.nodes[end == E.a ? E.b : E.a].p;
                const V2 dir = (p - q) * (1.0 / std::max(1e-9, len(p - q)));
                const double L = st.fb_lane_extend.sample(rx);
                std::vector<V2> line = {p};
                for (double a = 6.0; a <= L; a += 6.0) {
                    const V2 x = p + dir * a + left_of(dir) * (2.0 * fbm(wseed + 5, a, static_cast<double>(e), 80.0, 1));
                    int i, j;
                    if (!s.cell_of(x, i, j) || s.sky(i, j) || s.edge(i, j) || s.water(i, j) != WATER_NONE || w.blocked(i, j)) break;
                    line.push_back(x);
                }
                // 按几何量宽：伸出去的田间道可能擦着别家院角（栅格上只挡 blocked 格，不到一格的看不出），比最窄的路还窄就不伸
                const double pw = road_width(w, RC_PATH, rx);
                if (line.size() >= 4) {
                    const double fit = road_fit(w, line);
                    if (fit >= std::max(0.8, 0.5 * pw)) add_road(w, line, RC_PATH, std::min(pw, fit));
                }
            }
        }
    }
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
