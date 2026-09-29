// 形态算子「围绿」（PLAN-TOWN 6.2；中欧绿地村 Angerdorf、英格兰 village green；小村成环村 Rundling）：
//   绿地村：村心一块眼形（两头尖）的公地，长轴顺村路的方向（有河顺河、有坡顺等高线、否则朝出村方向）；
//     村路到公地一头分成两股绕着公地走、另一头合拢再出村；房屋沿两股路的外侧排，一律朝公地；
//     公地上可以有教堂（公地够宽时）、池塘；公地不压房、不修路（OCC_OPEN）。
//   环村：户数 ≤ rundling_max_hh 且挑到 rundling 时，一块圆形公地、一圈路，宅地呈扇形朝圆心，只留一个口出村。
//   地面不合适（公地压到水、虚空、陡坡）就换个方向、挪一挪；实在不行退回街村。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

V2 left_of(V2 t) { return {-t.y, t.x}; }

// 公地的外形能不能落：平、不压水、不出窗口、不压已有的东西
bool green_ok(const Work& w, const std::vector<V2>& poly, double max_slope) {
    bool ok = true;
    int n = 0;
    raster_polygon(w.s, poly, [&](int i, int j) {
        ++n;
        if (w.s.sky(i, j) || w.s.edge(i, j) || w.s.water(i, j) != WATER_NONE || w.f.slope_deg(i, j) > max_slope || w.plan.occ(i, j) != OCC_FREE) ok = false;
    });
    for (const V2& p : poly) {
        int i, j;
        ok = ok && w.s.cell_of(p, i, j);
    }
    return ok && n > 0;
}

// 眼形：长 L、宽 Wd，中心 c，长轴方位 ax；半边 = 上沿 / 下沿（从一头尖到另一头尖）
std::vector<V2> lens_edge(V2 c, double ax, double L, double Wd, int sgn, int n) {
    const V2 u = bearing_vec(ax), v = left_of(u);
    std::vector<V2> out;
    for (int k = 0; k <= n; ++k) {
        const double x = -0.5 * L + L * k / n, t = 2.0 * x / L;
        out.push_back(c + u * x + v * (sgn * 0.5 * Wd * (1.0 - t * t)));
    }
    return out;
}

double axis_of(const Work& w) {
    const Site& s = w.s;
    int ci, cj;
    s.cell_of(w.center, ci, cj);
    double bd = INF, b = wrap_pi(w.facing + 0.5 * PI);
    for (const River& rv : s.rivers) {
        int seg;
        double t;
        const double d = dist_point_polyline(w.center, rv.line, &seg, &t);
        if (d < bd && d < w.R + 150.0 && seg + 1 < static_cast<int>(rv.line.size())) bd = d, b = bearing_of(rv.line[seg + 1] - rv.line[seg]);
    }
    if (std::isfinite(bd)) return b;
    const float d = w.f.downslope(ci, cj);
    if (std::isfinite(d) && w.f.slope_deg(ci, cj) > 2.0f) return wrap_pi(d + 0.5 * PI);
    if (!w.req.exits.empty()) {
        const ExitTarget* e = &w.req.exits[0];
        for (const ExitTarget& x : w.req.exits)
            if (x.weight > e->weight) e = &x;
        return e->bearing;
    }
    return b;
}

void mark_green(Work& w, const std::vector<V2>& poly) {
    raster_polygon(w.s, poly, [&](int i, int j) {
        if (w.plan.occ(i, j) == OCC_FREE) w.plan.occ(i, j) = OCC_OPEN;
    });
}

Feature green_feature(const Work& w, const std::vector<V2>& poly, V2 c, double facing, double r) {
    Feature ft;
    ft.kind = "green";
    ft.func = "green";
    ft.name = w.st.ps("green.green_name", "");
    ft.poly = poly;
    ft.p = c;
    ft.facing = facing;
    ft.r = r;
    ft.z = field_at(w.s, w.s.height, c, 0.0f);
    return ft;
}

// 公地上挖一个塘（靠一头）
void pond_on(Work& w, V2 c, double r, Rng& rng) {
    std::vector<V2> poly;
    const double ph = rng.uniform(0, 2 * PI);
    for (int k = 0; k < 18; ++k) {
        const double th = 2 * PI * k / 18;
        poly.push_back(c + bearing_vec(th) * (r * (1.0 + 0.12 * std::sin(2 * th + ph))));
    }
    double zmin = INF;
    bool ok = true;
    raster_polygon(w.s, poly, [&](int i, int j) {
        if (w.plan.occ(i, j) != OCC_OPEN) ok = false;
        if (std::isfinite(w.s.height(i, j))) zmin = std::min(zmin, static_cast<double>(w.s.height(i, j)));
    });
    if (!ok || !std::isfinite(zmin)) return;
    Feature ft;
    ft.kind = "pond";
    ft.func = "pond";
    ft.name = w.st.ps("green.pond_name", "");
    ft.poly = poly;
    ft.p = c;
    ft.r = r;
    ft.z = zmin - 0.4;
    raster_polygon(w.s, poly, [&](int i, int j) { w.plan.occ(i, j) = OCC_POND, w.blocked(i, j) = 1; });
    w.plan.features.push_back(ft);
}

// 绿地上的教堂：公地中段、长轴顺公地，占下来（地块不进 OCC_OPEN 前先落）
int church_on(Work& w, V2 c, double ax, double room_w, Rng& r) {
    for (const FuncSpec& f : w.st.funcs) {
        if (f.id != "church" || f.mode != "compound" || !f.allows(w.plan.op) || f.count(static_cast<int>(w.plan.households.size())) < 1) continue;
        const int tmpl = w.st.template_index(f.tmpl);
        const TemplateSpec& T = w.st.templates[tmpl];
        const double Wd = T.plot_w.sample(r), Dp = T.plot_d.sample(r);
        if (Wd + 12.0 > room_w) return -1;
        for (double sh : {0.0, 12.0, -12.0, 24.0, -24.0}) {
            Slot sl;
            sl.tmpl = tmpl;
            sl.box = {c + bearing_vec(ax) * sh, ax, 0.5 * Wd, 0.5 * Dp};   // 塔在前，朝长轴一头
            sl.access_side = SIDE_FRONT;
            sl.access = sl.box.c + bearing_vec(ax) * (0.5 * Dp + 2.0);
            if (!plot_fits(w, sl, nullptr)) continue;
            return commit_compound(w, sl, "public", f.id, f.name, {});
        }
    }
    return -1;
}

}  // namespace

void op_green(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("green");
    const int n = static_cast<int>(w.hh_groups.size());
    double tw = 0.0, ws = 0.0;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op)) tw += st.templates[k].weight * st.templates[k].plot_w.mid(), ws += st.templates[k].weight;
    const double f_mean = (ws > 0 ? tw / ws : 20.0) + st.plot_gap.mid();
    // 挑 anger / rundling
    std::string mode = "anger";
    {
        std::vector<std::string> ids;
        std::vector<double> p;
        for (const std::string& k : {std::string("anger"), std::string("rundling")}) {
            const double v = st.pn("green.modes." + k);
            if (v <= 0) continue;
            if (k == "rundling" && n > static_cast<int>(st.pn("green.rundling_max_hh"))) continue;
            ids.push_back(k), p.push_back(v);
        }
        if (!ids.empty()) mode = ids[static_cast<size_t>(r.choice_p(p))];
    }
    const double max_slope = std::min(8.0, st.max_slope_deg);
    const double rw = road_width(w, RC_MAIN, r);
    if (mode == "rundling") {
        // 环村：圆形公地，一圈路；宅地扇形（前窄后宽，这里用前沿压窄的矩形近似）朝圆心
        const double front = std::max(8.0, 0.55 * f_mean);
        const double Rg = std::max(15.0, n * front / (2 * PI * 0.85) - 0.5 * rw - 2.0);
        const double Rr = Rg + 0.5 * rw + 1.0;
        V2 best_c{};
        bool found = false;
        for (double rad = 0.0; rad <= 60.0 && !found; rad += 10.0)
            for (int k = 0; k < (rad == 0.0 ? 1 : 8) && !found; ++k) {
                const V2 c = w.center + bearing_vec(2 * PI * k / 8) * rad;
                std::vector<V2> circ;
                for (int q = 0; q < 32; ++q) circ.push_back(c + bearing_vec(2 * PI * q / 32) * (Rr + 0.5 * rw));
                if (green_ok(w, circ, max_slope)) best_c = c, found = true;
            }
        if (found) {
            const V2 c = best_c;
            // 出口：朝最要紧的出村方向
            const double ex = axis_of(w);
            std::vector<V2> ring;
            const int nseg = 48;
            for (int q = 0; q <= nseg; ++q) ring.push_back(c + bearing_vec(ex + 2 * PI * q / nseg) * Rr);
            const int rid = add_road(w, ring, RC_MAIN, rw);
            // 出村的一段（从环上出口往外）
            const V2 gate = c + bearing_vec(ex) * Rr;
            std::vector<V2> out_road{gate, gate + bearing_vec(ex) * 30.0};
            add_road(w, out_road, RC_STREET, road_width(w, RC_STREET, r));
            std::vector<V2> gpoly;
            for (int q = 0; q < 32; ++q) gpoly.push_back(c + bearing_vec(2 * PI * q / 32) * (Rr - 0.5 * rw - 0.5));
            const Feature gf = green_feature(w, gpoly, c, ex, Rg);
            mark_green(w, gpoly);
            if (st.pn("green.pond_on_green") != 0.0) pond_on(w, c, 0.3 * Rg, r);
            w.plan.features.push_back(gf);
            // 宅地：环路外侧，朝圆心（方位角增大 = 顺时针走，圆心在右手，外侧是左手）
            SliceOpt o;
            o.face_road = true;
            o.front_w = {0.9 * front, 1.1 * front};
            {
                const std::vector<V2>& ln = w.plan.roads[rid].line;
                const V2 t = ln[1] - ln[0];
                o.side = dot(left_of(t), ln[0] - c) > 0 ? 1 : -1;
            }
            std::vector<Slot> slots = slice_along(w, w.plan.roads[rid].line, 0.5 * w.plan.roads[rid].width_m, o, r);
            std::stable_sort(slots.begin(), slots.end(), [&](const Slot& a, const Slot& b) { return a.noise < b.noise; });
            size_t g = fill_slots(w, slots, 0);
            w.plan.metrics["green_area_m2"] = PI * Rg * Rg;
            if (g < w.hh_groups.size()) organic_fill(w, g);
            return;
        }
        mode = "anger";
    }
    // 绿地村：眼形公地
    const double half_max = 0.5 * std::min(s.W, s.H) * s.res_m - 40.0;
    const Range Lr = st.pr("green.length_m"), Wr = st.pr("green.width_m");
    // 两股路外侧各排一排：公地长 ≈ 户数 × 面宽 / 2 × 0.8（余下的排在两头出村的路边）
    double L = std::clamp(0.5 * n * f_mean * 0.8, Lr.lo, std::min(Lr.hi, 2.0 * half_max - 80.0));
    double Wd = std::clamp(Wr.sample(r), Wr.lo, Wr.hi);
    const double ax0 = axis_of(w);
    std::vector<V2> up, lo;
    V2 c{};
    double ax = ax0;
    bool found = false;
    // 两股路外侧要有地排农庄：沿两条边往外 rw/2 + 半个地块进深处取点，可建的占比取两侧的小者（河谷里公地别把谷底占满、一侧只剩河）
    double dmean = 0.0, ws2 = 0.0;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op)) dmean += st.templates[k].weight * st.templates[k].plot_d.mid(), ws2 += st.templates[k].weight;
    dmean = ws2 > 0 ? dmean / ws2 : 40.0;
    auto side_room = [&](const std::vector<V2>& edge, V2 cc, double a) {
        const V2 v = left_of(bearing_vec(a));
        int n_ok = 0, n_all = 0;
        for (size_t q = 2; q + 2 < edge.size(); ++q) {
            const double sg = dot(edge[q] - cc, v) >= 0 ? 1.0 : -1.0;
            for (double d : {0.5 * rw + 0.3 * dmean, 0.5 * rw + 0.7 * dmean}) {
                int i, j;
                ++n_all;
                if (s.cell_of(edge[q] + v * (sg * d), i, j) && w.f.buildable(i, j) && w.plan.occ(i, j) == OCC_FREE) ++n_ok;
            }
        }
        return n_all ? static_cast<double>(n_ok) / n_all : 0.0;
    };
    for (double shrink : {1.0, 0.8, 0.65, 0.5}) {
        const double Ls = std::max(Lr.lo * 0.6, L * shrink), Ws = std::max(Wr.lo, Wd * std::sqrt(shrink));
        double best = -INF;
        for (double da : {0.0, 0.2, -0.2, 0.4, -0.4, 0.7, -0.7})
            for (double rad = 0.0; rad <= 90.0; rad += 15.0)
                for (int k = 0; k < (rad == 0.0 ? 1 : 8); ++k) {
                    const V2 cc = w.center + bearing_vec(2 * PI * k / 8) * rad;
                    const double a = wrap_pi(ax0 + da);
                    std::vector<V2> u = lens_edge(cc, a, Ls, Ws + rw + 2.0, 1, 24), l = lens_edge(cc, a, Ls, Ws + rw + 2.0, -1, 24);
                    std::vector<V2> poly = u;
                    for (auto it = l.rbegin() + 1; it + 1 != l.rend(); ++it) poly.push_back(*it);
                    if (!green_ok(w, poly, max_slope)) continue;
                    const double room = std::min(side_room(u, cc, a), side_room(l, cc, a));
                    const double sc = room - 0.15 * std::fabs(da) - 0.1 * rad / 90.0;
                    if (room < 0.45 || sc <= best) continue;
                    best = sc;
                    up = lens_edge(cc, a, Ls, Ws + rw, 1, 24), lo = lens_edge(cc, a, Ls, Ws + rw, -1, 24);
                    c = cc, ax = a, found = true;
                    L = Ls;
                }
        if (found) {
            Wd = Ws;
            break;
        }
    }
    if (!found) {
        w.plan.op = "street_village";   // 退回：标成实际用的算子
        op_street(w, false);
        return;
    }
    // 公地的外形（路内侧）
    std::vector<V2> gpoly = lens_edge(c, ax, L - 2.0 * rw, std::max(4.0, Wd - 1.0), 1, 24);
    {
        const std::vector<V2> l = lens_edge(c, ax, L - 2.0 * rw, std::max(4.0, Wd - 1.0), -1, 24);
        for (auto it = l.rbegin() + 1; it + 1 != l.rend(); ++it) gpoly.push_back(*it);
    }
    // 教堂先落在公地上（够宽时）
    const int church = st.pn("green.church_on_green") != 0.0 ? church_on(w, c, ax, Wd, r) : -1;
    // 两股路（两头尖处合拢），再从两头往外出村
    const int ru = add_road(w, up, RC_MAIN, rw), rl = add_road(w, lo, RC_MAIN, rw);
    std::vector<int> ext;
    for (int sg : {1, -1}) {
        const V2 tip = c + bearing_vec(ax) * (sg * 0.5 * L);
        V2 far = tip;
        for (double d = 160.0; d > 20.0; d -= 5.0) {
            const V2 q = tip + bearing_vec(ax) * (sg * d);
            int i, j;
            if (s.cell_of(q, i, j) && !s.sky(i, j) && !s.edge(i, j) && s.water(i, j) == WATER_NONE && !w.blocked(i, j)) {
                far = q;
                break;
            }
        }
        if (len(far - tip) > 20.0) {
            Mask goal(s.H, s.W, 0);
            int ti, tj;
            if (s.cell_of(tip, ti, tj)) goal(ti, tj) = 1;
            PathParams pp = w.pp;
            pp.bbox_margin = static_cast<int>(std::lround(80.0 / s.res_m));
            std::vector<V2> path;
            if (find_path(s, w.blocked, w.road_mask, goal, pp, far, path) && path.size() >= 2) {
                std::reverse(path.begin(), path.end());   // 从公地的尖往外
                ext.push_back(add_road(w, path, RC_MAIN, rw));
            }
        }
    }
    mark_green(w, gpoly);
    if (church >= 0) lane_to_gate(w, church, st.lane_w.mid(), 0.5 * Wd + 20.0);   // 公地上的小路通到教堂门口
    if (st.pn("green.pond_on_green") != 0.0) pond_on(w, c - bearing_vec(ax) * (0.22 * L), std::min(0.22 * Wd, 14.0), r);
    w.plan.features.push_back(green_feature(w, gpoly, c, ax, 0.5 * Wd));
    w.plan.metrics["green_area_m2"] = std::fabs(polygon_area(gpoly));
    // 其余公共宅院：沿两股路
    {
        std::vector<Slot> pc = public_candidates(w, c, 0.5 * L + 40.0, RC_MAIN, r);
        std::vector<Slot> keep;
        for (const Slot& sl : pc) {
            bool dup = false;
            for (const Compound& C : w.plan.compounds) dup = dup || (C.kind == "public" && C.tmpl == st.templates[sl.tmpl].id);
            if (!dup) keep.push_back(sl);
        }
        place_public_compounds(w, keep);
    }
    // 宅地：两股路的外侧，朝公地；从中段往两头
    std::vector<Slot> slots;
    for (int id : {ru, rl}) {
        if (id < 0) continue;
        const std::vector<V2>& line = w.plan.roads[id].line;
        for (int sd : {1, -1}) {
            SliceOpt o;
            o.face_road = true;
            o.side = sd;
            std::vector<Slot> ss = slice_along(w, line, 0.5 * w.plan.roads[id].width_m, o, r);
            // 只要外侧（离公地中心远的一侧）
            std::vector<Slot> out;
            for (const Slot& sl : ss) {
                int i, j;
                const V2 in = sl.box.c;
                if (s.cell_of(in, i, j) && !point_in_polygon(gpoly, in)) out.push_back(sl);
            }
            if (!out.empty()) {
                // 外侧：地块中心离长轴比路远
                const V2 v = left_of(bearing_vec(ax));
                const double dp = std::fabs(dot(out[out.size() / 2].box.c - c, v));
                const double dr = std::fabs(dot(out[out.size() / 2].access - c, v));
                if (dp > dr) slots.insert(slots.end(), out.begin(), out.end());
            }
        }
    }
    std::stable_sort(slots.begin(), slots.end(), [&](const Slot& a, const Slot& b) {
        return std::fabs(dot(a.box.c - c, bearing_vec(ax))) + 0.2 * f_mean * a.noise < std::fabs(dot(b.box.c - c, bearing_vec(ax))) + 0.2 * f_mean * b.noise;
    });
    // 两头出村的路两侧也排（离公地近的先）
    std::vector<Slot> more;
    for (int id : ext) {
        if (id < 0) continue;
        for (int sd : {1, -1}) {
            SliceOpt o;
            o.face_road = true;
            o.side = sd;
            std::vector<Slot> ss = slice_along(w, w.plan.roads[id].line, 0.5 * w.plan.roads[id].width_m, o, r);
            more.insert(more.end(), ss.begin(), ss.end());
        }
    }
    std::stable_sort(more.begin(), more.end(), [&](const Slot& a, const Slot& b) { return len(a.access - c) < len(b.access - c); });
    slots.insert(slots.end(), more.begin(), more.end());
    size_t g = fill_slots(w, slots, 0);
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
