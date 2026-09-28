// 形态算子「团块生长」（PLAN-TOWN 6.2；华北、徽州、中欧团村、地中海、英格兰外围）与「单个宅院」。
// 团块：先按坡度代价修一条过村心的主街；之后一户一户落位——在父户（同一房支）或一户邻居周围试位置：临现有路的一块，或贴着邻户的左右 / 前后；
// 评分 = 离村心近 + 兴趣高 + 离父户近 + 贴着邻户 − 要新修的巷长；落下后门到最近的路之间用 A* 修一条巷（路随门长出来）。
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

// 路上的采样点（每 4 m），带一个粗网格索引：最近路点查询
struct RoadIndex {
    struct Pt {
        V2 p, t;
        double hw;
    };
    std::vector<Pt> pts;
    std::unordered_map<int64_t, std::vector<int>> grid;
    double cell = 16.0;
    size_t roads_seen = 0;
    int64_t key(int a, int b) const { return (static_cast<int64_t>(a) << 32) ^ static_cast<uint32_t>(b); }
    void sync(const Work& w) {
        for (; roads_seen < w.plan.roads.size(); ++roads_seen) {
            const Road& rd = w.plan.roads[roads_seen];
            if (rd.cls > RC_LANE) continue;
            const double L = polyline_length(rd.line);
            for (double a = 1.0; a < L - 0.5; a += 4.0) {
                const std::vector<V2> q = subline(rd.line, std::max(0.0, a - 0.5), std::min(L, a + 0.5));
                if (q.size() < 2) continue;
                V2 t = q.back() - q.front();
                const double l = len(t);
                if (l < 1e-9) continue;
                pts.push_back({lerp(q.front(), q.back(), 0.5), t * (1.0 / l), 0.5 * rd.width_m});
                const V2 p = pts.back().p;
                grid[key(static_cast<int>(std::floor(p.x / cell)), static_cast<int>(std::floor(p.y / cell)))].push_back(static_cast<int>(pts.size()) - 1);
            }
        }
    }
    template <class F>
    void near(V2 p, double r, F f) const {
        const int a0 = static_cast<int>(std::floor((p.x - r) / cell)), a1 = static_cast<int>(std::floor((p.x + r) / cell));
        const int b0 = static_cast<int>(std::floor((p.y - r) / cell)), b1 = static_cast<int>(std::floor((p.y + r) / cell));
        for (int a = a0; a <= a1; ++a)
            for (int b = b0; b <= b1; ++b) {
                auto it = grid.find(key(a, b));
                if (it == grid.end()) continue;
                for (int k : it->second)
                    if (len(pts[k].p - p) <= r) f(k);
            }
    }
    double nearest(V2 p, double r, int* arg) const {
        double d = INF;
        near(p, r, [&](int k) {
            const double x = len(pts[k].p - p) - pts[k].hw;
            if (x < d) d = x, *arg = k;
        });
        return d;
    }
};

// 地块压不压任何东西（路、别的地块、房、塘……）且地面可建、挖填在上限内
bool fits(Work& w, Slot& sl) {
    bool clear = true;
    raster_obb(w.s, sl.box, 0.0, [&](int i, int j) {
        if (w.plan.occ(i, j) != OCC_FREE) clear = false;
    });
    if (!clear) return false;
    // 栅格看不出不到一格的相交：与附近的地块再按矩形判一次，与路按「路中线离地块 ≥ 半宽」判
    const double reach = sl.box.hw + sl.box.hd + 40.0;
    for (const Compound& c : w.plan.compounds)
        if (len(c.plot.c - sl.box.c) < reach && overlap(c.plot, sl.box, -0.05)) return false;
    if (road_hits_obb(w, sl.box, 0.05)) return false;
    return slot_ground(w, sl, nullptr);
}

// 门在地块哪条边的哪个位置（与宅院成形同一规则，门宽取中值）
V2 gate_point(const Work& w, const Obb& plot, int side, const TemplateSpec& T) {
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

// 朝向别漂出朝阳规则的容差（邻户一户户偏下去会越偏越远）：夹到容差的 0.7 以内
double clamp_facing(const Work& w, double fac) {
    for (const OrientRule& r : w.st.rules)
        if (r.kind == "sun") {
            const double m = 0.7 * r.tol, d = wrap_pi(fac - w.f.sun_bearing);
            return wrap_pi(w.f.sun_bearing + std::clamp(d, -m, m));
        }
    return fac;
}

V2 side_normal(const Obb& o, int side) {
    const V2 f = bearing_vec(o.facing), r = bearing_right(o.facing);
    return side == SIDE_FRONT ? f : side == SIDE_BACK ? f * -1.0 : side == SIDE_LEFT ? r * -1.0 : r;
}

}  // namespace

void organic_fill(Work& w, size_t first_group) {
    const Style& st = w.st;
    const Site& s = w.s;
    Rng r = w.rng("organic");
    RoadIndex ri;
    ri.sync(w);
    const double R = w.R;
    PathParams pp = w.pp;
    pp.bbox_margin = static_cast<int>(std::lround((st.og_lane_reach_m + 25.0) / s.res_m));
    pp.clearance_m = 0.5 * st.lane_w.lo + 0.2;
    pp.allow_squeeze = false;
    for (size_t g = first_group; g < w.hh_groups.size(); ++g) {
        const int h0 = w.hh_groups[g][0];
        const int par = w.plan.households[h0].parent;
        const int pc = par >= 0 ? w.plan.households[par].compound : -1;
        bool placed = false;
        int small = -1;
        {
            double a = INF;
            for (int k : st.house_templates())
                if (st.templates[k].plot_w.lo * st.templates[k].plot_d.lo < a) a = st.templates[k].plot_w.lo * st.templates[k].plot_d.lo, small = k;
        }
        // 退一步时的朝向：进深顺等高线（下坡方向 ± 90°，挑离朝阳近的）；平地就朝阳
        auto contour_facing = [&](V2 p) {
            const OrientCtx c = orient_ctx(w, p);
            if (!std::isfinite(c.downslope)) return w.f.sun_bearing;
            const double a = wrap_pi(c.downslope + 0.5 * PI), b = wrap_pi(c.downslope - 0.5 * PI);
            return angle_diff(a, w.f.sun_bearing) < angle_diff(b, w.f.sun_bearing) ? a : b;
        };
        for (int attempt = 0; attempt < 16 && !placed; ++attempt) {
            const bool relax = attempt >= 10;
            // 参照：父户；否则按离村心的远近随机挑一户已有的（没有就村心）
            V2 ref = w.center;
            std::vector<int> houses;
            for (int c = 0; c < static_cast<int>(w.plan.compounds.size()); ++c)
                if (w.plan.compounds[c].kind == "house") houses.push_back(c);
            if (pc >= 0 && attempt < 3) ref = w.plan.compounds[pc].plot.c;
            else if (!houses.empty()) {
                std::vector<double> p;
                for (int c : houses) p.push_back(1.0 / (1.0 + std::pow(len(w.plan.compounds[c].plot.c - w.center) / R, 2.0)));
                ref = w.plan.compounds[houses[static_cast<size_t>(r.choice_p(p))]].plot.c;
            }
            const double reach = 45.0 + 25.0 * std::min(attempt, 9);
            struct Cand {
                Slot sl;
                double lane;   // 要新修的巷长
                V2 gate;
            };
            std::vector<Cand> cands;
            // (a) 临现有路
            std::vector<int> rps;
            ri.near(ref, reach, [&](int k) { rps.push_back(k); });
            for (int q = 0; q < st.og_candidates / 2 && !rps.empty(); ++q) {
                const auto& rp = ri.pts[rps[static_cast<size_t>(r.integers(0, static_cast<int64_t>(rps.size())))]];
                const int tmpl = relax ? small : pick_template(w, r);
                const TemplateSpec& T = st.templates[tmpl];
                const double sg = r.random() < 0.5 ? 1.0 : -1.0;
                Slot sl = slot_along(w, rp.p, rp.t, V2{-rp.t.y, rp.t.x} * sg, rp.hw, tmpl, relax ? T.plot_w.lo : T.plot_w.sample(r),
                                     relax ? T.plot_d.lo : T.plot_d.sample(r), st.setback.sample(r), relax ? contour_facing(rp.p) : NaN);
                if (!fits(w, sl)) continue;
                cands.push_back({sl, 0.0, gate_point(w, sl.box, sl.access_side, T)});
            }
            // (b) 贴着邻户
            std::vector<int> nb;
            for (int c : houses)
                if (len(w.plan.compounds[c].plot.c - ref) < reach) nb.push_back(c);
            for (int q = 0; q < st.og_candidates - st.og_candidates / 2 && !nb.empty(); ++q) {
                const Compound& N = w.plan.compounds[nb[static_cast<size_t>(r.integers(0, static_cast<int64_t>(nb.size())))]];
                const int tmpl = relax ? small : pick_template(w, r);
                const TemplateSpec& T = st.templates[tmpl];
                const double Wd = relax ? T.plot_w.lo : T.plot_w.sample(r), Dp = relax ? T.plot_d.lo : T.plot_d.sample(r);
                const double fac = relax ? contour_facing(N.plot.c) : clamp_facing(w, wrap_pi(N.plot.facing + r.uniform(-st.og_jitter, st.og_jitter)));
                const V2 fv = bearing_vec(N.plot.facing), rv = bearing_right(N.plot.facing);
                const int dir = static_cast<int>(r.integers(0, 4));
                const double gap = st.og_gap.sample(r), stg = r.uniform(-st.og_stagger_m, st.og_stagger_m);
                V2 c;
                // 左右邻：前沿对齐（再错开一点）；前后邻：隔一条巷，侧边对齐
                if (dir == 0) c = N.plot.c + rv * (N.plot.hw + 0.5 * Wd + gap) + fv * (N.plot.hd - 0.5 * Dp + stg);
                else if (dir == 1) c = N.plot.c - rv * (N.plot.hw + 0.5 * Wd + gap) + fv * (N.plot.hd - 0.5 * Dp + stg);
                else {
                    const double lane = st.lane_w.hi + 0.8;   // 前后之间留一条巷（按最宽的巷留）
                    const double off = N.plot.hd + 0.5 * Dp + lane;
                    const double al = (r.random() < 0.5 ? 1.0 : -1.0) * (N.plot.hw - 0.5 * Wd);
                    c = N.plot.c + fv * (dir == 2 ? off : -off) + rv * (al + stg);
                }
                Slot sl;
                sl.tmpl = tmpl;
                sl.box = {c, fac, 0.5 * Wd, 0.5 * Dp};
                if (!fits(w, sl)) continue;
                // 门开在离路最近的那一边（前沿优先一点）
                double best = INF;
                int side = SIDE_FRONT;
                V2 gate{};
                for (int sd = 0; sd < 4; ++sd) {
                    const V2 gp = gate_point(w, sl.box, sd, T);
                    int k = -1;
                    const double d = ri.nearest(gp + side_normal(sl.box, sd), st.og_lane_reach_m + 10.0, &k) + (sd == SIDE_FRONT ? 0.0 : sd == SIDE_BACK ? 3.0 : 5.0);
                    if (k >= 0 && d < best) best = d, side = sd, gate = gp;
                }
                if (!(best <= st.og_lane_reach_m)) continue;
                sl.access_side = side;
                cands.push_back({sl, std::max(0.0, best), gate});
            }
            // 评分
            double bs = -INF;
            int arg = -1;
            for (int k = 0; k < static_cast<int>(cands.size()); ++k) {
                const Slot& sl = cands[k].sl;
                int touch = 0;
                for (int c : houses) {
                    const Obb& o = w.plan.compounds[c].plot;
                    if (len(o.c - sl.box.c) < 45.0 && overlap(o, sl.box, 1.5)) ++touch;
                }
                double sc = -(st.g_net * (len(sl.box.c - w.center) + cands[k].lane) + st.g_dist * len(sl.box.c - w.center)) / R +
                            st.g_interest * sl.interest + st.g_contact * std::min(touch, 3) + st.g_noise * r.uniform(-1.0, 1.0) - 0.03 * cands[k].lane;
                if (pc >= 0) sc -= st.g_kin * len(sl.box.c - w.plan.compounds[pc].plot.c) / R;
                if (sc > bs) bs = sc, arg = k;
            }
            if (arg < 0) continue;
            Cand& cd = cands[arg];
            Slot sl = cd.sl;
            if (cd.lane > 0.5) {
                const double lane_w = Range{st.lane_w.lo, st.lane_w.mid()}.sample(r);
                // 巷的尽头离门外 0.5 × 巷宽 + 0.8 m：起点本身就在间距以外，巷不会拐着蹭自家的墙角
                const V2 from = cd.gate + side_normal(sl.box, sl.access_side) * (0.5 * lane_w + 0.8);
                pp.clearance_m = 0.5 * lane_w + 0.5;
                std::vector<V2> path;
                // 先占住地块再找路，免得巷从自家院里穿
                Slot tmp = sl;
                const int ci = commit_compound(w, tmp, "house", "dwelling", "", w.hh_groups[g]);
                double fit = INF;
                int fi, fj;
                const bool from_ok = s.cell_of(from, fi, fj) && (w.plan.occ(fi, fj) == OCC_FREE || w.plan.occ(fi, fj) == OCC_ROAD);
                bool ok = from_ok && find_path(s, w.blocked, w.road_mask, w.road_core, pp, from, path, &fit) && path.size() >= 2;
                // 格上的间距在平滑、拐角处会被吃掉：按几何再量一次，挤得比最窄的巷一半还窄（蹭着人家的墙、门楼）就不修
                if (ok) fit = std::min(fit, road_fit(w, path)), ok = fit >= 0.5 * st.lane_w.lo;
                if (!ok) {
                    // 修不出巷：撤回这块
                    w.plan.compounds.pop_back();
                    for (int h : w.hh_groups[g]) w.plan.households[h].compound = -1;
                    raster_obb(s, sl.box, 0.0, [&](int i, int j) {
                        if (w.plan.occ(i, j) == OCC_PLOT) w.plan.occ(i, j) = OCC_FREE, w.blocked(i, j) = 0;
                    });
                    continue;
                }
                std::vector<V2> lane = path;
                std::reverse(lane.begin(), lane.end());   // 从路到门外
                w.plan.compounds[ci].access = lane.back();
                add_road(w, lane, RC_LANE, std::min(lane_w, fit));
                ri.sync(w);
            } else {
                commit_compound(w, sl, "house", "dwelling", "", w.hh_groups[g]);
            }
            placed = true;
        }
        if (!placed && small >= 0) {
            // 随机试不出来：沿所有路点两侧挨个试最小的宅院（进深顺等高线），取离村心近、兴趣高的一块——坡上、山顶的村就顺着出村的路往下长
            const TemplateSpec& T = st.templates[small];
            double bs = -INF;
            Slot best{};
            for (const auto& rp : ri.pts)
                for (double sg : {1.0, -1.0}) {
                    Slot sl = slot_along(w, rp.p, rp.t, V2{-rp.t.y, rp.t.x} * sg, rp.hw, small, T.plot_w.lo, T.plot_d.lo, st.setback.lo,
                                         contour_facing(rp.p));
                    if (!fits(w, sl)) continue;
                    const double sc = -len(sl.box.c - w.center) / R + st.g_interest * sl.interest;
                    if (sc > bs) bs = sc, best = sl;
                }
            if (std::isfinite(bs)) commit_compound(w, best, "house", "dwelling", "", w.hh_groups[g]), placed = true;
        }
        if (!placed) break;
    }
}

void op_organic(Work& w) {
    const Site& s = w.s;
    Rng r = w.rng("organic.main");
    if (w.st.og_main_street) {
        const V2 tu = bearing_right(w.facing);
        const double L = 1.3 * w.R + 30.0;
        int ci, cj;
        s.cell_of(w.center, ci, cj);
        Mask goal(s.H, s.W, 0);
        goal(ci, cj) = 1;
        std::vector<V2> halves[2];
        for (int k = 0; k < 2; ++k) {
            // 端点：沿主街方向往外，退到走得通的格
            V2 e = w.center;
            for (double a = L; a > 10.0; a -= 5.0) {
                const V2 p = w.center + tu * (k == 0 ? a : -a);
                int i, j;
                if (s.cell_of(p, i, j) && !s.sky(i, j) && !s.edge(i, j) && s.water(i, j) == WATER_NONE) {
                    e = p;
                    break;
                }
            }
            if (len(e - w.center) < 10.0) continue;
            find_path(s, w.blocked, w.road_mask, goal, w.pp, e, halves[k]);
        }
        std::vector<V2> main;
        for (auto it = halves[0].begin(); it != halves[0].end(); ++it) main.push_back(*it);
        if (!halves[1].empty()) {
            if (!main.empty()) main.pop_back();
            for (auto it = halves[1].rbegin(); it != halves[1].rend(); ++it) main.push_back(*it);
        }
        if (main.size() >= 2) {
            const double width = road_width(w, RC_MAIN, r);
            add_road(w, main, RC_MAIN, width);
            const int a = w.net.add_node(main.front()), b = w.net.add_node(main.back());
            const int e = w.net.add_edge(a, b, main, RC_MAIN, width, 1.0, 0);
            w.net.edges[e].used_a = w.net.edges[e].len;
        }
    }
    // 公共宅院（庙）：临主街、靠村心的几块候选
    {
        std::vector<Slot> pc;
        for (const FuncSpec& f : w.st.funcs) {
            if (f.mode != "compound") continue;
            const int tmpl = w.st.template_index(f.tmpl);
            const TemplateSpec& T = w.st.templates[tmpl];
            for (const Road& rd : w.plan.roads) {
                const double L = polyline_length(rd.line);
                for (double a = 5.0; a < L - 5.0; a += 6.0) {
                    const std::vector<V2> q = subline(rd.line, a - 0.5, a + 0.5);
                    if (q.size() < 2 || len(q.front() - w.center) > 1.2 * w.R) continue;
                    V2 t = q.back() - q.front();
                    t = t * (1.0 / std::max(1e-9, len(t)));
                    for (double sg : {1.0, -1.0}) {
                        Slot sl = slot_along(w, q.front(), t, V2{-t.y, t.x} * sg, 0.5 * rd.width_m, tmpl, T.plot_w.sample(r), T.plot_d.sample(r), 0.0);
                        bool clear = true;
                        raster_obb(s, sl.box, 0.3 * s.res_m, [&](int i, int j) { clear = clear && w.plan.occ(i, j) == OCC_FREE; });
                        if (clear && slot_ground(w, sl, nullptr)) pc.push_back(sl);
                    }
                }
            }
        }
        place_public_compounds(w, pc);
    }
    organic_fill(w, 0);
}

void op_single(Work& w) {
    Rng r = w.rng("single");
    const int tmpl = pick_template(w, r);
    const TemplateSpec& T = w.st.templates[tmpl];
    const double Wd = T.plot_w.sample(r), Dp = T.plot_d.sample(r);
    double best = -INF;
    Slot arg;
    for (double rad = 0.0; rad <= 50.0; rad += 3.0)
        for (int k = 0; k < (rad == 0.0 ? 1 : 16); ++k) {
            const V2 p = w.center + bearing_vec(2 * PI * k / 16) * rad;
            const OrientCtx c = orient_ctx(w, p);
            Slot sl;
            sl.tmpl = tmpl;
            sl.box = {p, solve_facing(w.st.rules, c, w.st.snap, c.sun), 0.5 * Wd, 0.5 * Dp};
            sl.access_side = SIDE_FRONT;
            sl.access = p + bearing_vec(sl.box.facing) * (0.5 * Dp + 2.0);
            if (!slot_ground(w, sl, nullptr)) continue;
            const double sc = sl.interest - rad / 100.0;
            if (sc > best) best = sc, arg = sl;
        }
    if (!std::isfinite(best)) return;
    std::vector<int> all;
    for (const auto& g : w.hh_groups) all.insert(all.end(), g.begin(), g.end());
    commit_compound(w, arg, "house", "dwelling", "", all);
}

}  // namespace skyisle::town
