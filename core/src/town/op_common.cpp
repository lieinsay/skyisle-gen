// 各形态算子共用的件（PLAN-TOWN 6.2）：沿路切宅基、按次序落户、过村心的一条路、门到路网的巷、公共宅院的候选、算子的适用条件。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

V2 left_of(V2 t) { return {-t.y, t.x}; }

// 折线上弧长 a 处的点与单位切向
V2 point_on(const std::vector<V2>& line, double a, V2& t) {
    const double L = polyline_length(line);
    const std::vector<V2> q = subline(line, std::clamp(a - 0.5, 0.0, L), std::clamp(a + 0.5, 0.0, L));
    if (q.size() < 2) {
        t = {1.0, 0.0};
        return line.empty() ? V2{} : line.front();
    }
    t = q.back() - q.front();
    const double l = len(t);
    t = l > 1e-9 ? t * (1.0 / l) : V2{1.0, 0.0};
    return lerp(q.front(), q.back(), 0.5);
}

}  // namespace

V2 side_normal(const Obb& o, int side) {
    const V2 f = bearing_vec(o.facing), r = bearing_right(o.facing);
    return side == SIDE_FRONT ? f : side == SIDE_BACK ? f * -1.0 : side == SIDE_LEFT ? r * -1.0 : r;
}

V2 gate_point_of(const Work& w, const Obb& plot, int side, const TemplateSpec& T) {
    const double W = 2 * plot.hw, D = 2 * plot.hd, t = T.wall ? w.st.wall_thickness_m : 0.0;
    const double inset = t + 0.5 * w.st.gate_w.mid() + 0.3;
    std::string rule = side == SIDE_FRONT ? w.st.gate_front : side == SIDE_BACK ? w.st.gate_back : w.st.gate_side;
    if (!T.gate.empty()) rule = T.gate;
    double x, y;
    if (side == SIDE_FRONT || side == SIDE_BACK) {
        x = rule == "left" ? -0.5 * W + inset : rule == "right" ? 0.5 * W - inset : 0.0;
        y = side == SIDE_FRONT ? 0.5 * D : -0.5 * D;
    } else {
        y = rule == "back" ? -0.5 * D + inset : rule == "center" ? 0.0 : 0.5 * D - inset;
        x = side == SIDE_LEFT ? -0.5 * W : 0.5 * W;
    }
    return plot.c + bearing_right(plot.facing) * x + bearing_vec(plot.facing) * y;
}

std::vector<Slot> slice_along(Work& w, const std::vector<V2>& line, double road_hw, const SliceOpt& o, Rng& r) {
    std::vector<Slot> out;
    if (line.size() < 2) return out;
    const double L = polyline_length(line);
    const double s_end = std::min(L, o.s1);
    double sp = std::max(0.0, o.s0);
    for (int guard = 0; guard < 2000 && sp < s_end; ++guard) {
        const int tmpl = o.tmpl >= 0 ? o.tmpl : pick_template(w, r);
        const TemplateSpec& T = w.st.templates[tmpl];
        const double Wd = o.front_w.hi > 0 ? o.front_w.sample(r) : T.plot_w.sample(r);
        const double Dp = o.depth.hi > 0 ? o.depth.sample(r) : T.plot_d.sample(r);
        const double sb = o.setback >= 0 ? o.setback : w.st.setback.sample(r);
        V2 t;
        V2 p = point_on(line, sp, t);
        V2 n_out = left_of(t) * static_cast<double>(o.side);
        double force = o.face_road ? bearing_of(n_out * -1.0) : o.face_away ? bearing_of(n_out) : o.force_facing;
        const Slot probe = slot_along(w, p, t, n_out, road_hw, tmpl, Wd, Dp, sb, force);
        if (sp + 2.0 * probe.half_t > s_end) break;
        p = point_on(line, sp + probe.half_t, t);
        n_out = left_of(t) * static_cast<double>(o.side);
        if (o.face_road) force = bearing_of(n_out * -1.0);
        else if (o.face_away) force = bearing_of(n_out);
        // 弯街的内侧地块角会蹭着路：往后退一点再试；坡上院子深了高差太大：进深依次缩到 ¾、½ 再试（不浅于 18 m 或原进深）
        bool placed = false;
        for (int q = 0; q < 9 && !placed; ++q) {
            const double k = q < 3 ? 1.0 : q < 6 ? 0.75 : 0.55, extra = 0.6 * (q % 3);
            if (k < 1.0 && Dp * k < std::min(18.0, Dp)) break;
            Slot sl = slot_along(w, p, t, n_out, road_hw, tmpl, Wd, Dp * k, sb + extra, force);
            sl.s0 = sp, sl.s1 = sp + 2.0 * sl.half_t, sl.s_acc = sp + sl.half_t;
            sl.noise = r.uniform(-1.0, 1.0);
            // 弯道内侧相邻两块的后角会互压：与这一溜已切下的再按矩形判一次
            bool clash = false;
            for (auto it = out.rbegin(); it != out.rend() && it - out.rbegin() < 4 && !clash; ++it) clash = overlap(it->box, sl.box, -0.05);
            if (!clash && plot_fits(w, sl, o.extra)) {
                out.push_back(sl);
                sp = sl.s1 + (o.gap.hi > 0 ? o.gap.sample(r) : w.st.plot_gap.sample(r));
                placed = true;
            }
        }
        if (!placed) sp += 2.0;
    }
    return out;
}

size_t fill_slots(Work& w, std::vector<Slot>& slots, size_t g0) {
    size_t g = g0;
    for (Slot& sl : slots) {
        if (g >= w.hh_groups.size()) break;
        if (sl.taken || sl.vacant) continue;
        Slot t = sl;
        sl.taken = true;
        if (!plot_fits(w, t, nullptr)) continue;
        sl.comp = commit_compound(w, t, "house", "dwelling", "", w.hh_groups[g]);
        ++g;
    }
    return g;
}

std::vector<V2> through_road(Work& w, V2 c, double bearing, double half_len) {
    const Site& s = w.s;
    const V2 tu = bearing_vec(bearing);
    int ci, cj;
    if (!s.cell_of(c, ci, cj)) return {};
    Mask goal(s.H, s.W, 0);
    goal(ci, cj) = 1;
    std::vector<V2> halves[2];
    for (int k = 0; k < 2; ++k) {
        // 端点：顺方向往外，退到走得通的格
        V2 e = c;
        for (double a = half_len; a > 10.0; a -= 5.0) {
            const V2 p = c + tu * (k == 0 ? a : -a);
            int i, j;
            if (s.cell_of(p, i, j) && !s.sky(i, j) && !s.edge(i, j) && s.water(i, j) == WATER_NONE && !w.blocked(i, j)) {
                e = p;
                break;
            }
        }
        if (len(e - c) < 10.0) continue;
        find_path(s, w.blocked, w.road_mask, goal, w.pp, e, halves[k]);
    }
    std::vector<V2> line(halves[0].begin(), halves[0].end());
    if (!halves[1].empty()) {
        if (!line.empty()) line.pop_back();
        for (auto it = halves[1].rbegin(); it != halves[1].rend(); ++it) line.push_back(*it);
    }
    return line;
}

void withdraw_last_compound(Work& w) {
    Compound& C = w.plan.compounds.back();
    for (int h : C.households) w.plan.households[h].compound = -1;
    raster_obb(w.s, C.plot, 0.0, [&](int i, int j) {
        if (w.plan.occ(i, j) == OCC_PLOT) w.plan.occ(i, j) = OCC_FREE, w.blocked(i, j) = 0;
    });
    w.plan.compounds.pop_back();
}

bool lane_to_gate(Work& w, int ci, double lane_w, double reach_m) {
    const Site& s = w.s;
    Compound& C = w.plan.compounds[ci];
    const TemplateSpec& T = w.st.templates[w.st.template_index(C.tmpl)];
    const V2 gate = gate_point_of(w, C.plot, C.access_side, T);
    const V2 from = gate + side_normal(C.plot, C.access_side) * (0.5 * lane_w + 0.8);
    PathParams pp = w.pp;
    pp.bbox_margin = static_cast<int>(std::lround((reach_m + 25.0) / s.res_m));
    pp.clearance_m = 0.5 * lane_w + 0.5;
    pp.allow_squeeze = false;
    int fi, fj;
    if (!s.cell_of(from, fi, fj) || !(w.plan.occ(fi, fj) == OCC_FREE || w.plan.occ(fi, fj) == OCC_ROAD || w.plan.occ(fi, fj) == OCC_OPEN)) return false;
    if (w.road_core(fi, fj)) {
        C.access = from;
        return true;
    }
    std::vector<V2> path;
    double fit = INF;
    // 格上的间距在平滑、拐角处会被吃掉：先按宽一点的间距找，再按常规的；按几何再量一次，挤得比最窄的巷一半还窄（或 < 0.8 m）就不修
    bool ok = false;
    for (int pass = 0; pass < 2 && !ok; ++pass) {
        pp.clearance_m = 0.5 * lane_w + (pass == 0 ? 1.2 : 0.5);
        fit = INF;
        ok = find_path(s, w.blocked, w.road_mask, w.road_core, pp, from, path, &fit) && path.size() >= 2;
        if (ok) fit = std::min(fit, road_fit(w, path)), ok = fit >= std::max(0.8, 0.5 * w.st.lane_w.lo);
    }
    if (!ok) return false;
    std::reverse(path.begin(), path.end());
    C.access = path.back();
    add_road(w, path, RC_LANE, std::min(lane_w, fit));
    return true;
}

std::vector<Slot> public_candidates(Work& w, V2 c, double reach, int max_cls, Rng& r) {
    std::vector<Slot> pc;
    for (const FuncSpec& f : w.st.funcs) {
        if (f.mode != "compound" || !f.allows(w.plan.op)) continue;
        const int tmpl = w.st.template_index(f.tmpl);
        const TemplateSpec& T = w.st.templates[tmpl];
        for (const Road& rd : w.plan.roads) {
            if (rd.cls > max_cls) continue;
            const double L = polyline_length(rd.line);
            for (double a = 5.0; a < L - 5.0; a += 6.0) {
                V2 t;
                const V2 p = point_on(rd.line, a, t);
                if (len(p - c) > reach) continue;
                for (double sg : {1.0, -1.0}) {
                    Slot sl = slot_along(w, p, t, left_of(t) * sg, 0.5 * rd.width_m, tmpl, T.plot_w.sample(r), T.plot_d.sample(r), 0.0);
                    if (plot_fits(w, sl, nullptr)) pc.push_back(sl);
                }
            }
        }
    }
    return pc;
}

std::vector<V2> resample(const std::vector<V2>& line, double step) {
    std::vector<V2> out;
    if (line.size() < 2) return line;
    const double L = polyline_length(line);
    const int n = std::max(1, static_cast<int>(std::ceil(L / step)));
    V2 t;
    out.push_back(line.front());
    for (int k = 1; k < n; ++k) out.push_back(point_on(line, L * k / n, t));
    out.push_back(line.back());
    return out;
}

std::vector<std::vector<V2>> split_runs(const Work& w, const std::vector<V2>& pts, const std::function<bool(int, int)>& ok, double min_len) {
    std::vector<std::vector<V2>> out;
    std::vector<V2> cur;
    auto flush = [&] {
        if (cur.size() >= 2 && polyline_length(cur) >= min_len) out.push_back(cur);
        cur.clear();
    };
    for (const V2& p : pts) {
        int i, j;
        if (w.s.cell_of(p, i, j) && ok(i, j)) cur.push_back(p);
        else flush();
    }
    flush();
    return out;
}

bool plan_link(const Work& w, V2 from, double width, double reach_m, const Mask* goal, std::vector<V2>& path, double& fit) {
    const Site& s = w.s;
    PathParams pp = w.pp;
    pp.bbox_margin = static_cast<int>(std::lround((reach_m + 30.0) / s.res_m));
    // 格上的路线平滑后会切院角：先按宽一点的间距找（平滑了也蹭不着墙），再按常规间距、最后许接入处挤一点（挤得比最窄的路还窄就不修）
    const Mask& g = goal ? *goal : w.road_core;
    for (int pass = 0; pass < 3; ++pass) {
        pp.clearance_m = 0.5 * width + (pass == 0 ? 1.5 : 0.3);
        pp.allow_squeeze = pass == 2;
        fit = INF;
        if (find_path(s, w.blocked, w.road_mask, g, pp, from, path, &fit) && path.size() >= 2) {
            fit = std::min(fit, road_fit(w, path));
            if (fit >= 0.8) {
                std::reverse(path.begin(), path.end());   // 从路网到 from
                return true;
            }
        }
    }
    return false;
}

int link_road(Work& w, V2 from, int cls, double width, double reach_m, const Mask* goal) {
    std::vector<V2> path;
    double fit = INF;
    if (!plan_link(w, from, width, reach_m, goal, path, fit)) return -1;
    return add_road(w, path, cls, std::min(width, fit));
}

int commit_public(Work& w, const std::string& func_id, Slot sl) {
    for (const FuncSpec& f : w.st.funcs) {
        if (f.id != func_id || f.mode != "compound" || !f.allows(w.plan.op)) continue;
        sl.tmpl = w.st.template_index(f.tmpl);
        return commit_compound(w, sl, "public", f.id, f.name, {});
    }
    return -1;
}

void place_missing_public(Work& w) {
    const int N = static_cast<int>(w.plan.households.size());
    for (const FuncSpec& f : w.st.funcs) {
        if (f.mode != "compound" || !f.required || !f.allows(w.plan.op) || f.count(N) < 1) continue;
        bool have = false;
        for (const Compound& c : w.plan.compounds) have = have || (c.kind == "public" && c.func == f.id);
        if (have) continue;
        const int tmpl = w.st.template_index(f.tmpl);
        const TemplateSpec& T = w.st.templates[tmpl];
        const double Wd = T.plot_w.lo, Dp = T.plot_d.lo;
        struct C {
            double sc;
            Slot sl;
        };
        // 先在村子一带（1.5 R + 80 m）找；陡坡上村子占了能用的地（山城在峡湾岸上）就到村外远一点（3 R + 200 m）的平处去，修条路接回来——
        // 村外高处、坡嘴上的教堂、庙本来就常见
        bool placed = false;
        for (int pass = 0; pass < 2 && !placed; ++pass) {
            const double rmax = pass == 0 ? 1.5 * w.R + 80.0 : 3.0 * w.R + 200.0, step = pass == 0 ? 6.0 : 10.0;
            std::vector<C> cands;
            for (double rad = pass == 0 ? 0.0 : 1.5 * w.R + 80.0; rad <= rmax; rad += step)
                for (int k = 0; k < (rad == 0.0 ? 1 : std::max(8, static_cast<int>(2 * PI * rad / 12.0))); ++k) {
                    const V2 p = w.center + bearing_vec(2 * PI * k / std::max(8.0, std::floor(2 * PI * rad / 12.0))) * rad;
                    const OrientCtx c = orient_ctx(w, p);
                    std::vector<double> facs{solve_facing(w.st.rules, c, w.st.snap, c.sun)};
                    if (std::isfinite(c.downslope)) facs.push_back(wrap_pi(c.downslope + 0.5 * PI)), facs.push_back(wrap_pi(c.downslope - 0.5 * PI)), facs.push_back(c.downslope);
                    for (double fac : facs) {
                        Slot sl;
                        sl.tmpl = tmpl;
                        sl.box = {p, fac, 0.5 * Wd, 0.5 * Dp};
                        sl.access_side = SIDE_FRONT;
                        sl.access = p + bearing_vec(fac) * (0.5 * Dp + 2.0);
                        if (!plot_fits(w, sl, nullptr)) continue;
                        cands.push_back({site_score(w, f, p) - rad / (w.R + 80.0) - 0.2 * sl.max_cut, sl});
                    }
                }
            std::sort(cands.begin(), cands.end(), [](const C& a, const C& b) { return a.sc > b.sc; });
            int tries = 0;
            for (C& c : cands) {
                if (++tries > 30) break;
                if (!plot_fits(w, c.sl, nullptr)) continue;
                const int ci = commit_compound(w, c.sl, "public", f.id, f.name, {});
                if (lane_to_gate(w, ci, w.st.lane_w.mid(), pass == 0 ? 80.0 : 250.0)) {
                    placed = true;
                    break;
                }
                withdraw_last_compound(w);
            }
        }
    }
}

bool op_fits(const Work& w, const std::string& op) {
    auto it = w.st.op_max_hh.find(op);
    const int n = static_cast<int>(w.plan.households.size());
    if (it != w.st.op_max_hh.end() && it->second > 0 && n > it->second) return false;
    {
        auto fd = w.st.op_flat_deg.find(op), fs = w.st.op_flat_share.find(op);
        if (fd != w.st.op_flat_deg.end() && fs != w.st.op_flat_share.end() &&
            1.0 - total_slope_share(w, w.center, 1.2 * w.R + 40.0, fd->second) < fs->second)
            return false;
    }
    if (op == "waterfront") {
        const double reach = w.R + w.st.pn("waterfront.reach_m");
        for (const River& rv : w.s.rivers) {
            double wmax = 0.0;
            for (double x : rv.width_m) wmax = std::max(wmax, x);
            if (wmax >= w.st.pn("waterfront.min_width_m") && dist_point_polyline(w.center, rv.line) <= reach) return true;
        }
        return false;
    }
    if (op == "contour")
        return total_slope_share(w, w.center, 1.2 * w.R + 40.0, w.st.pn("contour.min_slope_deg")) >= w.st.pn("contour.min_share");
    return true;
}

}  // namespace skyisle::town
