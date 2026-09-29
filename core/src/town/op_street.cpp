// 形态算子「街村」与「林地排村」（PLAN-TOWN 6.2；中欧街村、宿場町、瑞典列村、英格兰 toft & croft；林地排村 Waldhufendorf）：
//   1. 主街：过村心、顺街的方向（有坡顺等高线、有河顺河、否则朝最要紧的出村方向）按坡度代价修，长度按要排的面宽定；
//   2. 沿街两侧切条地（面宽沿街，地块一律朝街——山墙还是檐口朝街由模板里房子的摆法定）；
//   3. 公共宅院先占，然后从村心沿街往两头一户户落；排满了在主街上分支街再排；还不够交给团块生长；
//   4. 林地排村：每户地块后面一条与面宽同宽的长条地（条身是自家田），一直伸到窗口边、虚空、水或陡坡；
//   5. 英格兰：地块后面各一条背巷。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

V2 left_of(V2 t) { return {-t.y, t.x}; }

// 主街的方向
double street_axis(const Work& w, const std::string& mode) {
    const Site& s = w.s;
    int ci, cj;
    s.cell_of(w.center, ci, cj);
    auto contour = [&](double& b) {
        const float d = w.f.downslope(ci, cj);
        if (!std::isfinite(d) || !(w.f.slope_deg(ci, cj) > 2.0f)) return false;
        b = wrap_pi(d + 0.5 * PI);
        return true;
    };
    auto river = [&](double& b) {
        double bd = INF;
        for (const River& rv : s.rivers) {
            int seg;
            double t;
            const double d = dist_point_polyline(w.center, rv.line, &seg, &t);
            if (d < bd && d < w.R + 150.0 && seg + 1 < static_cast<int>(rv.line.size())) bd = d, b = bearing_of(rv.line[seg + 1] - rv.line[seg]);
        }
        return std::isfinite(bd);
    };
    auto exit = [&](double& b) {
        if (w.req.exits.empty()) return false;
        const ExitTarget* e = &w.req.exits[0];
        for (const ExitTarget& x : w.req.exits)
            if (x.weight > e->weight) e = &x;
        b = e->bearing;
        return true;
    };
    double b = wrap_pi(w.facing + 0.5 * PI);
    if (mode == "contour") contour(b);
    else if (mode == "river") river(b) || contour(b);
    else if (mode == "exit") exit(b);
    else if (mode == "across") b = wrap_pi(w.facing + 0.5 * PI);
    else if (!(river(b) || contour(b) || exit(b))) b = wrap_pi(w.facing + 0.5 * PI);   // auto
    return b;
}

// 折线上离 p 最近处的弧长
double arc_of(const std::vector<V2>& line, V2 p) {
    int seg;
    double t;
    dist_point_polyline(p, line, &seg, &t);
    double a = 0.0;
    for (int k = 0; k < seg; ++k) a += len(line[k + 1] - line[k]);
    return a + t * len(line[seg + 1] - line[seg]);
}

}  // namespace

void op_street(Work& w, bool hufen) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng(hufen ? "hufen" : "street_village");
    const std::string sec = hufen ? "hufen." : "street_village.";
    const int n = static_cast<int>(w.hh_groups.size());
    const int sides = hufen ? 2 : std::clamp(static_cast<int>(st.pn("street_village.sides")), 1, 2);
    // 平均面宽（沿街）：给了就用给的，否则按住宅模板的面宽
    double tw = 0.0, ws = 0.0;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op)) tw += st.templates[k].weight * st.templates[k].plot_w.mid(), ws += st.templates[k].weight;
    Range front = hufen ? st.pr("hufen.frontage_m") : st.pr("street_village.front_w_m");
    const double f_mean = front.hi > 0 ? front.mid() : (ws > 0 ? tw / ws : 15.0);
    const double factor = hufen ? 1.05 : st.pn("street_village.length_factor");
    const double half_max = 0.5 * std::min(s.W, s.H) * s.res_m - 30.0;
    double half = std::clamp(0.5 * n * (f_mean + st.plot_gap.mid()) / sides * factor + 20.0, 60.0, half_max);
    const double axis = street_axis(w, st.ps(sec + "axis", "auto"));
    const std::vector<V2> main = through_road(w, w.center, axis, half);
    if (main.size() < 2 || polyline_length(main) < 40.0) {
        w.plan.op = "organic";   // 退回：标成实际用的算子
        op_organic(w);
        return;
    }
    const double main_w = road_width(w, RC_MAIN, r);
    add_road(w, main, RC_MAIN, main_w);
    {
        const int a = w.net.add_node(main.front()), b = w.net.add_node(main.back());
        const int e = w.net.add_edge(a, b, main, RC_MAIN, main_w, 1.0, 0);
        w.net.edges[e].used_a = w.net.edges[e].len;
    }
    const double L = polyline_length(main);
    // 林地排村：街不够长时把面宽压窄（不窄于 min_frontage_m）
    if (hufen) {
        const double fit = L * sides / std::max(1, n) - st.plot_gap.mid();
        const double f = std::clamp(fit, st.pn("hufen.min_frontage_m"), front.hi);
        front = {0.9 * f, std::min(front.hi, 1.1 * f)};
    }
    // 公共宅院（教堂、庙、本陣……）：主街靠村心的一段
    {
        std::vector<Slot> pc = public_candidates(w, w.center, 0.8 * w.R + 30.0, RC_MAIN, r);
        place_public_compounds(w, pc);
    }
    // 沿街两侧切条地，按离村心（沿街）由近到远
    SliceOpt o;
    o.face_road = hufen || st.pn("street_village.face_street") != 0.0;
    o.front_w = front;
    if (!hufen) o.depth = st.pr("street_village.depth_m");
    const double sc = arc_of(main, w.center);
    std::vector<Slot> slots;
    for (int sd : {1, -1}) {
        if (sides == 1 && sd == -1) break;
        o.side = sd;
        std::vector<Slot> ss = slice_along(w, main, 0.5 * main_w, o, r);
        for (Slot& sl : ss) sl.edge = sd;   // 记在哪一侧（背巷用）
        slots.insert(slots.end(), ss.begin(), ss.end());
    }
    std::stable_sort(slots.begin(), slots.end(), [&](const Slot& a, const Slot& b) {
        return std::fabs(a.s_acc - sc) + 0.3 * a.noise < std::fabs(b.s_acc - sc) + 0.3 * b.noise;
    });
    const size_t first_comp = w.plan.compounds.size();
    size_t g = fill_slots(w, slots, 0);
    // 排满了：主街上分支街（从村心往两头，左右交替）
    if (g < w.hh_groups.size()) {
        const Range bsp = st.pr("street_village.branch_spacing_m"), blen = st.pr("street_village.branch_len_m");
        std::vector<double> at;
        for (double d = 0.5 * bsp.sample(r); d < 0.5 * L; d += bsp.sample(r)) at.push_back(sc + d), at.push_back(sc - d);
        int flip = 1;
        for (double a : at) {
            if (g >= w.hh_groups.size()) break;
            if (a < 15.0 || a > L - 15.0) continue;
            const std::vector<V2> seg = subline(main, a - 0.5, a + 0.5);
            if (seg.size() < 2) continue;
            V2 t = seg.back() - seg.front();
            t = t * (1.0 / std::max(1e-9, len(t)));
            const V2 p = lerp(seg.front(), seg.back(), 0.5);
            flip = -flip;
            for (int sd : {flip, -flip}) {
                const V2 dir = left_of(t) * static_cast<double>(sd);
                const double bl = blen.sample(r);
                V2 tip = p;
                for (double d = bl; d > 25.0; d -= 5.0) {
                    const V2 q = p + dir * d;
                    int i, j;
                    if (s.cell_of(q, i, j) && !s.sky(i, j) && !s.edge(i, j) && s.water(i, j) == WATER_NONE && !w.blocked(i, j)) {
                        tip = q;
                        break;
                    }
                }
                if (len(tip - p) < 25.0) continue;
                PathParams pp = w.pp;
                pp.bbox_margin = static_cast<int>(std::lround((bl + 40.0) / s.res_m));
                const double bw = road_width(w, RC_STREET, r);
                pp.clearance_m = 0.5 * bw + 0.3;
                std::vector<V2> path;
                double fit = INF;
                if (!find_path(s, w.blocked, w.road_mask, w.road_core, pp, tip, path, &fit) || path.size() < 2) continue;
                // 格上的间距在平滑、拐角处会被吃掉：按几何再量一次，挤得比半条街还窄（蹭着人家的房）就不修
                fit = std::min(fit, road_fit(w, path));
                if (fit < std::max(0.8, 0.5 * bw)) continue;
                std::reverse(path.begin(), path.end());
                if (add_road(w, path, RC_STREET, std::min(bw, fit)) < 0) continue;
                std::vector<Slot> bs;
                for (int s2 : {1, -1}) {
                    o.side = s2;
                    std::vector<Slot> ss = slice_along(w, path, 0.5 * std::min(bw, fit), o, r);
                    for (Slot& sl : ss) sl.edge = 0;
                    bs.insert(bs.end(), ss.begin(), ss.end());
                }
                std::stable_sort(bs.begin(), bs.end(), [](const Slot& a, const Slot& b) { return a.s_acc < b.s_acc; });
                g = fill_slots(w, bs, g);
                if (g >= w.hh_groups.size()) break;
            }
        }
    }
    const size_t last_comp = w.plan.compounds.size();
    // 林地排村的条地：地块后面、与面宽同宽，伸到走不下去为止
    if (hufen) {
        const Range sl = st.pr("hufen.strip_len_m");
        for (size_t ci = first_comp; ci < last_comp; ++ci) {
            const Compound& C = w.plan.compounds[ci];
            if (C.kind != "house") continue;
            const V2 out = side_normal(C.plot, C.access_side) * -1.0;   // 离街而去
            const bool ew = C.access_side == SIDE_FRONT || C.access_side == SIDE_BACK;
            const double half_w = ew ? C.plot.hw : C.plot.hd, half_d = ew ? C.plot.hd : C.plot.hw;
            const V2 across = V2{-out.y, out.x};
            const V2 base = C.plot.c + out * half_d;
            const double want = sl.sample(r);
            double got = 0.0;
            for (double d = 5.0; d <= want; d += 5.0) {
                bool bad = false;
                for (double u : {-0.9, 0.0, 0.9}) {
                    int i, j;
                    const V2 q = base + out * d + across * (u * half_w);
                    if (!s.cell_of(q, i, j) || s.sky(i, j) || s.edge(i, j) || s.water(i, j) != WATER_NONE || w.f.slope_deg(i, j) > 25.0f ||
                        w.plan.occ(i, j) == OCC_PLOT || w.plan.occ(i, j) == OCC_BUILDING) {
                        bad = true;
                        break;
                    }
                }
                if (bad) break;
                got = d;
            }
            if (got < 30.0) continue;
            Feature ft;
            ft.kind = "strip";
            ft.func = "hufen";
            ft.name = st.ps("hufen.strip_name", "");
            ft.compound = static_cast<int>(ci);
            ft.poly = {base - across * half_w, base + across * half_w, base + out * got + across * half_w, base + out * got - across * half_w};
            ft.p = base + out * (0.5 * got);
            ft.facing = bearing_of(out);
            ft.r = 2.0 * half_w;
            w.plan.features.push_back(ft);
        }
    }
    // 背巷（英格兰）：每一侧地块的后沿外 gap 处连成一条小路，两头接回主街
    if (!hufen && st.pn("street_village.back_lane") != 0.0) {
        const double gap = st.pn("street_village.back_lane_gap_m");
        for (int sd : {1, -1}) {
            std::vector<std::pair<double, V2>> pts;
            for (const Slot& sl : slots) {
                if (sl.comp < 0 || sl.edge != sd) continue;
                const Compound& C = w.plan.compounds[sl.comp];
                const V2 out = side_normal(C.plot, C.access_side) * -1.0;
                const double half_d = (C.access_side == SIDE_FRONT || C.access_side == SIDE_BACK) ? C.plot.hd : C.plot.hw;
                pts.push_back({sl.s_acc, C.plot.c + out * (half_d + gap + 0.5 * st.path_w.mid())});
            }
            if (pts.size() < 4) continue;
            std::sort(pts.begin(), pts.end(), [](auto& a, auto& b) { return a.first < b.first; });
            std::vector<V2> lane;
            for (auto& [a, p] : pts) {
                int i, j;
                if (s.cell_of(p, i, j) && w.plan.occ(i, j) == OCC_FREE && s.water(i, j) == WATER_NONE && !s.sky(i, j)) lane.push_back(p);
            }
            if (lane.size() < 4) continue;
            chaikin(lane, nullptr, 2);
            const double pw = road_width(w, RC_PATH, r);
            if (road_fit(w, lane) < 1.0) continue;
            // 两头接回主街（目标只是主街的中线）；一头都接不上就不修这条背巷（孤零零一条路不连路网）
            Mask goal(s.H, s.W, 0);
            raster_line(s, main, std::max(s.res_m, main_w - s.res_m), [&](int i, int j) { goal(i, j) = 1; });
            std::vector<std::pair<std::vector<V2>, double>> links;
            for (const V2& end : {lane.front(), lane.back()}) {
                std::vector<V2> path;
                double fit = INF;
                if (plan_link(w, end, pw, 200.0, &goal, path, fit)) links.push_back({path, std::min(pw, fit)});
            }
            if (links.empty()) continue;
            if (add_road(w, lane, RC_PATH, pw) < 0) continue;
            for (auto& [path, wd] : links) add_road(w, path, RC_PATH, wd);
        }
    }
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
