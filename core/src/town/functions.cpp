// 功能分配（PLAN-TOWN 6.4、7.5、7.7）：选址谓词；公共宅院（庙）占位；出村大路；泊场；塘；路边的庙与树；场院；井。
// 各功能的数量、尺寸、谓词权重都来自功能表（functions.toml ← 风格覆盖）；这里只有各 mode 的摆法。
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "plan_work.hpp"
#include "skyisle/town/raster.hpp"

namespace skyisle::town {

namespace {

double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }

float at(const Work& w, const GridF& g, V2 p, float fb) {
    int i, j;
    if (!w.s.cell_of(p, i, j)) return fb;
    const float v = g(i, j);
    return std::isfinite(v) ? v : fb;
}

double dist_built(const Work& w, V2 p) {
    double d = INF;
    for (const Compound& c : w.plan.compounds) d = std::min(d, dist_obb(c.plot, p));
    return d;
}

int total_households(const Work& w) { return static_cast<int>(w.plan.households.size()); }

// 建成区 = 宅院（民宅与庙这样的公共宅院）的地块；路边小庙、泊场货棚不算（不然场院会跟着土地庙跑到村外几十米）
Mask compound_mask(const Work& w, bool& any) {
    Mask m(w.s.H, w.s.W, 0);
    any = false;
    for (const Compound& c : w.plan.compounds) raster_obb(w.s, c.plot, 0.0, [&](int i, int j) { m(i, j) = 1, any = true; });
    return m;
}

// 离建成区（地块）的距离场（m）
GridF built_distance(const Work& w) {
    bool any = false;
    const Mask m = compound_mask(w, any);
    GridF d;
    if (!any) return GridF(w.s.H, w.s.W, 1e9f);
    edt(m, d, nullptr);
    for (float& x : d.v) x = static_cast<float>(x * w.s.res_m);
    return d;
}

// 建成区的外廓：地块外扩 15 m 再内缩 15 m（30 m 虚边界，与形态指标同口径）；村边的设施落在它外面
Mask built_outline(const Work& w) {
    bool any = false;
    const Mask plot = compound_mask(w, any);
    if (!any) return plot;
    GridF d1, d2;
    edt(plot, d1, nullptr);
    const float r = static_cast<float>(15.0 / w.s.res_m);
    Mask outside(w.s.H, w.s.W, 0);
    for (size_t k = 0; k < plot.size(); ++k) outside.v[k] = d1.v[k] > r;
    edt(outside, d2, nullptr);
    Mask body(w.s.H, w.s.W, 0);
    for (size_t k = 0; k < plot.size(); ++k) body.v[k] = d2.v[k] > r || plot.v[k];
    return body;
}

bool outside_outline(const Work& w, const Mask& body, const Obb& o) {
    bool ok = true;
    raster_obb(w.s, o, 0.0, [&](int i, int j) { ok = ok && !body(i, j); });
    return ok;
}

// 沿用到的路（等级 ≤ max_cls）每 step 米一个点：点、切向、路宽
struct RoadPt {
    V2 p, t;
    double width;
    int road, cls;
};
std::vector<RoadPt> road_points(const Work& w, int max_cls, double step) {
    std::vector<RoadPt> out;
    for (int k = 0; k < static_cast<int>(w.plan.roads.size()); ++k) {
        const Road& rd = w.plan.roads[k];
        if (rd.cls > max_cls) continue;
        const double L = polyline_length(rd.line);
        for (double a = 0.5 * step; a < L; a += step) {
            const std::vector<V2> seg = subline(rd.line, std::max(0.0, a - 0.5), std::min(L, a + 0.5));
            if (seg.size() < 2) continue;
            V2 t = seg.back() - seg.front();
            const double l = len(t);
            if (l < 1e-9) continue;
            out.push_back({lerp(seg.front(), seg.back(), 0.5), t * (1.0 / l), rd.width_m, k, rd.cls});
        }
    }
    return out;
}

double ground_max_cut(const Work& w, const Obb& o, double* base) {
    std::vector<double> hs;
    raster_obb(w.s, o, 0.0, [&](int i, int j) {
        const double h = w.s.height(i, j);
        if (std::isfinite(h)) hs.push_back(h);
    });
    if (hs.empty()) return INF;
    std::vector<double> t = hs;
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    const double b = t[t.size() / 2];
    if (base) *base = b;
    double m = 0.0;
    for (double h : hs) m = std::max(m, std::fabs(h - b));
    return m;
}

Building small_building(const Work& w, const FuncSpec& f, const Obb& box, const std::string& role) {
    Building b;
    b.func = f.id;
    b.role = role;
    b.name = f.name;
    b.roof = f.gets("roof", "gable");
    b.box = box;
    b.storeys = static_cast<int>(f.getn("storeys", 1.0));
    b.eave_m = f.getn("eave_m", 2.5);
    b.pitch_deg = f.getn("pitch_deg", 28.0);
    double base = 0.0;
    ground_max_cut(w, box, &base);
    b.base_m = base + f.getn("plinth_m", 0.3);
    raster_obb(w.s, box, 0.0, [&](int i, int j) {
        const double h = w.s.height(i, j);
        if (!std::isfinite(h)) return;
        const double a = w.s.res_m * w.s.res_m;
        if (h > base) b.cut_m3 += (h - base) * a;
        else b.fill_m3 += (base - h) * a;
    });
    return b;
}

void add_building(Work& w, const Building& b) {
    w.plan.buildings.push_back(b);
    mark_obb(w, b.box, OCC_BUILDING, 0.0);
}

std::vector<V2> ellipse_poly(V2 c, double a, double b, double rot, Rng& r, double wobble) {
    std::vector<V2> out;
    const int n = 28;
    const V2 u = bearing_vec(rot), v = bearing_right(rot);
    const double ph1 = r.uniform(0, 2 * PI), ph2 = r.uniform(0, 2 * PI);
    const double a1 = r.uniform(0.3, 1.0) * wobble, a2 = r.uniform(0.2, 0.7) * wobble;
    for (int k = 0; k < n; ++k) {
        const double th = 2 * PI * k / n;
        const double m = 1.0 + a1 * std::sin(2 * th + ph1) + a2 * std::sin(3 * th + ph2);
        out.push_back(c + u * (a * m * std::cos(th)) + v * (b * m * std::sin(th)));
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- 选址谓词
double predicate(const Work& w, const std::string& name, V2 p) {
    const double R = std::max(20.0, w.R);
    if (name == "center") return 1.0 - clamp01(len(p - w.center) / R);
    if (name == "crossroad") {
        bool any_used = false;
        for (int e = 0; e < static_cast<int>(w.net.edges.size()) && !any_used; ++e) any_used = w.net.used(e);
        double d = INF;
        for (int n = 0; n < static_cast<int>(w.net.nodes.size()); ++n) {
            const int deg = any_used ? w.net.degree_used(n) : static_cast<int>(w.net.nodes[n].edges.size());
            if (deg >= 3) d = std::min(d, len(w.net.nodes[n].p - p));
        }
        return std::isfinite(d) ? std::exp(-d / 15.0) : 0.0;
    }
    if (name == "main") {
        double d = INF;
        if (!w.plan.roads.empty()) {
            for (const Road& rd : w.plan.roads)
                if (rd.cls <= RC_MAIN) d = std::min(d, dist_point_polyline(p, rd.line));
        } else {
            for (const NetEdge& e : w.net.edges)
                if (e.cls <= RC_MAIN) d = std::min(d, dist_point_polyline(p, e.line));
        }
        return std::isfinite(d) ? std::exp(-d / 12.0) : 0.0;
    }
    if (name == "entrance") return w.has_entrance ? std::exp(-len(p - w.entrance) / 20.0) : 0.0;
    if (name == "edge") return clamp01(len(p - w.center) / (1.2 * R));
    if (name == "low") return clamp01(0.5 - at(w, w.f.tpi_s, p, 0.0f) / 1.5);
    if (name == "high") return clamp01(0.5 + at(w, w.f.tpi_l, p, 0.0f) / 3.0);
    if (name == "water") return std::exp(-at(w, w.s.water_dist_m, p, 1e6f) / 80.0);
    if (name == "farmland") return std::exp(-at(w, w.f.dist_farm_m, p, 1e6f) / 30.0);
    if (name == "flat") return clamp01(1.0 - at(w, w.f.slope_deg, p, 90.0f) / 5.0);
    if (name == "sun") return clamp01(at(w, w.f.sun, p, 0.0f) / 1.3);
    if (name == "pond") {
        double d = INF;
        for (const Feature& f : w.plan.features)
            if (f.kind == "pond") d = std::min(d, std::max(0.0, len(p - f.p) - f.r));
        return std::isfinite(d) ? std::exp(-d / 15.0) : 0.0;
    }
    if (name == "landing") {
        const V2 q = w.req.landings.empty() ? w.req.anchor : w.req.landings[0];
        return std::exp(-len(p - q) / 100.0);
    }
    throw std::invalid_argument("town: unknown site predicate " + name);
}

double site_score(const Work& w, const FuncSpec& f, V2 p) {
    double s = 0.0;
    for (auto& [k, wt] : f.site) s += wt * predicate(w, k, p);
    return s;
}

// ---------------------------------------------------------------- 公共宅院
int place_public_compounds(Work& w, std::vector<Slot>& cands) {
    int placed = 0;
    const int N = total_households(w);
    for (const FuncSpec& f : w.st.funcs) {
        if (f.mode != "compound") continue;
        const int tmpl = w.st.template_index(f.tmpl);
        const int n = f.count(N);
        for (int q = 0; q < n; ++q) {
            double best = -INF;
            int arg = -1;
            for (int k = 0; k < static_cast<int>(cands.size()); ++k) {
                const Slot& sl = cands[k];
                if (sl.taken || sl.tmpl != tmpl || !sl.valid) continue;
                bool clash = false;
                for (const Compound& c : w.plan.compounds) clash = clash || overlap(c.plot, sl.box, -0.1);
                if (clash) continue;
                const double sc = site_score(w, f, sl.box.c) + 0.3 * sl.interest + (sl.access_side == SIDE_FRONT ? 0.3 : 0.0);
                if (sc > best) best = sc, arg = k;
            }
            if (arg < 0) break;
            commit_compound(w, cands[arg], "public", f.id, f.name, {});
            cands[arg].taken = true;
            for (Slot& sl : cands)
                if (!sl.taken && overlap(sl.box, cands[arg].box, -0.1)) sl.taken = true;
            ++placed;
        }
    }
    return placed;
}

// ---------------------------------------------------------------- 出村大路
void place_exits(Work& w) {
    const Site& s = w.s;
    Rng r = w.rng("exits");
    std::vector<ExitTarget> T = w.req.exits;
    if (T.empty()) {
        const double side = w.facing + 0.5 * PI;
        T = {{wrap_pi(side), 1e4, 1.0, "road"}, {wrap_pi(side + PI), 1e4, 0.9, "road"}, {wrap_pi(w.facing), 1e4, 0.5, "fields"}};
    }
    std::stable_sort(T.begin(), T.end(), [](const ExitTarget& a, const ExitTarget& b) { return a.weight > b.weight; });
    bool any_road = false;
    for (uint8_t v : w.road_mask.v) any_road = any_road || v;
    Mask goal = w.road_core;
    if (!any_road) {
        int ci, cj;
        if (!s.cell_of(w.center, ci, cj)) return;
        goal(ci, cj) = 1;
    }
    const double xmin = s.x0 + 3.0, xmax = s.x0 + s.W * s.res_m - 3.0, ymax = s.y0 - 3.0, ymin = s.y0 - s.H * s.res_m + 3.0;
    int made = 0;
    std::vector<double> used_bearings;
    for (const ExitTarget& e : T) {
        if (made >= w.st.exits) break;
        bool close = false;
        for (double b : used_bearings) close = close || angle_diff(b, e.bearing) < 35.0 * PI / 180.0;
        if (close) continue;
        // 从村心沿方位角走到窗口边（或目标处，若在窗口里）；中途遇虚空就不修这条
        const V2 d = bearing_vec(e.bearing);
        V2 end = w.center;
        bool void_hit = false;
        for (double a = 0.0;; a += s.res_m) {
            const V2 p = w.center + d * a;
            if (p.x < xmin || p.x > xmax || p.y < ymin || p.y > ymax || a > len(w.center - w.req.anchor) + e.dist_m) break;
            int i, j;
            s.cell_of(p, i, j);
            if (s.sky(i, j)) {
                void_hit = true;
                break;
            }
            if (s.water(i, j) == WATER_NONE && !s.edge(i, j) && !w.blocked(i, j)) end = p;
        }
        if (void_hit || len(end - w.center) < 0.6 * w.R) continue;
        std::vector<V2> path;
        const double width = road_width(w, RC_TRUNK, r);
        PathParams pp = w.pp;
        pp.clearance_m = 0.5 * width + 0.3;
        double fit = INF;
        if (!find_path(s, w.blocked, w.road_mask, goal, pp, end, path, &fit)) continue;
        std::reverse(path.begin(), path.end());
        const int rid = add_road(w, path, RC_TRUNK, std::min(width, fit));
        if (rid < 0) continue;
        used_bearings.push_back(e.bearing);
        ++made;
        if (!w.has_entrance) {
            // 村口：沿路从村里往外，第一个离所有宅院 > 12 m 的点
            const double L = polyline_length(path);
            for (double a = 0.0; a <= L; a += 2.0) {
                const std::vector<V2> q = subline(path, a, std::min(L, a + 0.01));
                if (q.empty()) break;
                if (dist_built(w, q.front()) > 12.0) {
                    w.entrance = q.front();
                    w.has_entrance = true;
                    w.entrance_road = rid;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 泊场（空岛特有，照上游）
void place_landing(Work& w) {
    const FuncSpec* f = w.st.func("landing");
    if (!f) return;
    const Site& s = w.s;
    Rng r = w.rng("landing");
    const double pw = f->getr("pad_w_m", {30, 40}).sample(r), pd = f->getr("pad_d_m", {20, 26}).sample(r);
    const double sw = f->getr("shed_w_m", {10, 14}).sample(r), sd = f->getr("shed_d_m", {5, 7}).sample(r);
    const Range ring = f->getr("ring_m", {10, 120});
    const double search = f->getn("search_m", 250.0), ms = f->getn("max_slope_deg", 3.0);
    const V2 L0 = w.req.landings.empty() ? w.req.anchor : w.req.landings[0];
    // 长边顺冬季风（空艇迎风起降）；不知道风向时顺村子
    const double fac = std::isfinite(w.f.wind_from) ? w.f.wind_from : w.facing + 0.5 * PI;
    const GridF db = built_distance(w);
    const Mask body = built_outline(w);
    const int stride = std::max(1, static_cast<int>(std::lround(4.0 / s.res_m)));
    // 先找平地（坡 ≤ max_slope_deg、挖填 ≤ 0.8 m）；坡地上找不到就退一步：垫一块台地（坡 ≤ 14°、挖填 ≤ platform_max_m）
    const double pmax = f->getn("platform_max_m", 3.5);
    struct C {
        double sc;
        Obb pad;
    };
    std::vector<C> cands;
    for (int pass = 0; pass < 2 && cands.empty(); ++pass) {
        const double slope_lim = pass == 0 ? ms : 14.0, cut_lim = pass == 0 ? 0.8 : pmax;
        for (int i = 0; i < s.H; i += stride)
            for (int j = 0; j < s.W; j += stride) {
                const V2 p = s.center(i, j);
                if (len(p - L0) > search * (pass == 0 ? 1.0 : 1.5) || db(i, j) < ring.lo || db(i, j) > ring.hi || body(i, j)) continue;
                if (w.plan.occ(i, j) != OCC_FREE || !(w.f.slope_deg(i, j) <= slope_lim)) continue;
                for (double rot : {0.0, 0.5 * PI}) {
                    const Obb pad{p, wrap_pi(fac + rot), 0.5 * pd, 0.5 * pw};
                    if (!obb_free(w, pad, 0.0, false, slope_lim) || !outside_outline(w, body, pad)) continue;
                    const double cut = ground_max_cut(w, pad, nullptr);
                    if (cut > cut_lim) continue;
                    cands.push_back({site_score(w, *f, p) - 0.5 * len(p - L0) / search - (rot > 0 ? 0.2 : 0.0) - 0.3 * cut, pad});
                }
            }
    }
    std::sort(cands.begin(), cands.end(), [](const C& a, const C& b) { return a.sc > b.sc; });
    bool any_road = false;
    for (uint8_t v : w.road_mask.v) any_road = any_road || v;
    const double width = road_width(w, RC_STREET, r);
    PathParams pp = w.pp;
    pp.clearance_m = 0.5 * width + 0.3;
    Obb arg{};
    std::vector<V2> road;
    int road_side = -1;
    double road_fit = INF;
    bool found = false;
    std::vector<Obb> tried;
    for (const C& c : cands) {
        if (tried.size() >= 24) break;
        bool near = false;
        for (const Obb& t : tried) near = near || len(t.c - c.pad.c) < 25.0;   // 挨着试过的不再试
        if (near) continue;
        tried.push_back(c.pad);
        const auto cs = corners(c.pad);
        int side = 0;
        double bd = INF;
        for (int k = 0; k < 4; ++k) {
            const V2 m = lerp(cs[k], cs[(k + 1) % 4], 0.5);
            if (len(m - w.center) < bd) bd = len(m - w.center), side = k;
        }
        if (!any_road) {
            arg = c.pad, road_side = side, found = true;
            break;
        }
        const V2 m = lerp(cs[side], cs[(side + 1) % 4], 0.5);
        const V2 from = m + (m - c.pad.c) * ((pp.clearance_m + 0.5) / std::max(1e-6, len(m - c.pad.c)));
        Mask blk = w.blocked;
        raster_obb(s, c.pad, 0.0, [&](int i, int j) { blk(i, j) = 1; });
        std::vector<V2> path;
        double fit = INF;
        if (find_path(s, blk, w.road_mask, w.road_core, pp, from, path, &fit)) {
            std::reverse(path.begin(), path.end());
            path.push_back(m);   // 一直通到场边
            arg = c.pad, road = path, road_side = side, found = true, road_fit = fit;
            break;
        }
    }
    if (!found) return;
    Feature ft;
    ft.kind = "landing";
    ft.func = "landing";
    ft.name = f->name;
    ft.p = arg.c;
    ft.facing = arg.facing;
    const auto cs2 = corners(arg);
    ft.poly.assign(cs2.begin(), cs2.end());
    ground_max_cut(w, arg, &ft.z);
    w.plan.features.push_back(ft);
    mark_obb(w, arg, OCC_LANDING, 0.0);
    if (road.size() >= 2) add_road(w, road, RC_STREET, std::min(width, road_fit));
    // 货棚：贴着场的一条边（不是进场的那条），棚口朝场；先挑长边、离村近的
    {
        Obb best_o{};
        double bs = INF;
        for (int k = 0; k < 4; ++k) {
            if (k == road_side) continue;
            const V2 a = cs2[k], b = cs2[(k + 1) % 4], m = lerp(a, b, 0.5);
            const V2 out = (m - arg.c) * (1.0 / std::max(1e-9, len(m - arg.c)));
            const double edge_len = len(b - a);
            if (edge_len < sw + 1.0) continue;
            const Obb o{m + out * (0.5 * sd + 3.0), bearing_of(out * -1.0), 0.5 * sw, 0.5 * sd};
            if (!obb_free(w, o, 0.0, false, 8.0) || road_hits_obb(w, o, -0.3)) continue;
            const double sc = len(o.c - w.center) - 0.5 * edge_len;
            if (sc < bs) bs = sc, best_o = o;
        }
        if (std::isfinite(bs)) {
            FuncSpec g = *f;
            g.name = f->gets("shed_name", "shed");
            g.str["roof"] = f->gets("shed_roof", "shed");
            g.num["eave_m"] = f->getn("shed_eave_m", 3.2);
            add_building(w, small_building(w, g, best_o, "shed"));
            w.plan.buildings.back().func = "landing_shed";
        }
    }
}

// ---------------------------------------------------------------- 塘
void place_ponds(Work& w) {
    const FuncSpec* f = w.st.func("pond");
    if (!f) return;
    const Site& s = w.s;
    const int N = total_households(w);
    const int n = f->count(N);
    if (n <= 0) return;
    Rng r = w.rng("ponds");
    const Range ar = f->getr("area_m2", {300, 3000}), ring = f->getr("ring_m", {6, 60}), asp = f->getr("aspect", {1, 1.8});
    const double each = std::clamp(N * f->getn("area_per_household_m2", 15.0) / n, ar.lo, ar.hi);
    const GridF db = built_distance(w);
    const Mask body = built_outline(w);
    const int stride = std::max(1, static_cast<int>(std::lround(3.0 / s.res_m)));
    for (int q = 0; q < n; ++q) {
        const double A = each * r.uniform(0.7, 1.3), ap = asp.sample(r);
        const double a = std::sqrt(A * ap / PI), b = A / (PI * a);
        struct C {
            double sc;
            V2 p;
        };
        std::vector<C> cs;
        for (int i = 0; i < s.H; i += stride)
            for (int j = 0; j < s.W; j += stride) {
                if (db(i, j) < ring.lo + b || db(i, j) > ring.hi + a || body(i, j)) continue;
                if (w.plan.occ(i, j) != OCC_FREE || s.water(i, j) != WATER_NONE || s.sky(i, j) || s.edge(i, j) || w.f.slope_deg(i, j) > 4.0f) continue;
                const V2 p = s.center(i, j);
                bool near = false;
                for (const Feature& ft : w.plan.features)
                    if (ft.kind == "pond" && len(ft.p - p) < 60.0) near = true;
                if (near) continue;
                cs.push_back({site_score(w, *f, p) + 0.1 * r.random(), p});
            }
        std::sort(cs.begin(), cs.end(), [](const C& x, const C& y) { return x.sc > y.sc; });
        bool done = false;
        for (size_t k = 0; k < cs.size() && k < 200 && !done; ++k) {
            const double rot = r.uniform(0, PI);
            for (double scale = 1.0; scale > 0.5 && !done; scale *= 0.85) {
                const std::vector<V2> poly = ellipse_poly(cs[k].p, a * scale, b * scale, rot, r, 0.12);
                int bad = 0, tot = 0;
                double zmin = INF;
                raster_polygon(s, poly, [&](int i, int j) {
                    ++tot;
                    if (w.plan.occ(i, j) != OCC_FREE || s.water(i, j) != WATER_NONE || s.sky(i, j) || s.edge(i, j)) ++bad;
                    if (std::isfinite(s.height(i, j))) zmin = std::min(zmin, static_cast<double>(s.height(i, j)));
                });
                if (tot == 0 || bad > 0) continue;
                Feature ft;
                ft.kind = "pond";
                ft.func = f->id;
                ft.name = f->name;
                ft.poly = poly;
                ft.p = polygon_centroid(poly);
                ft.r = std::sqrt(a * b) * scale;
                ft.facing = rot;
                ft.z = zmin - 0.5;   // 水面比塘边最低处低半米
                raster_polygon(s, poly, [&](int i, int j) {
                    w.plan.occ(i, j) = OCC_POND;
                    w.blocked(i, j) = 1;
                });
                w.plan.features.push_back(ft);
                done = true;
            }
        }
    }
}

// ---------------------------------------------------------------- 路边的小庙、村口大树
void place_roadside(Work& w) {
    const int N = total_households(w);
    Rng r = w.rng("roadside");
    const std::vector<RoadPt> rp = road_points(w, RC_LANE, 3.0);
    const GridF db = built_distance(w);
    bool any_pond = false;
    for (const Feature& ft : w.plan.features) any_pond = any_pond || ft.kind == "pond";
    for (const FuncSpec& f : w.st.funcs) {
        if (f.mode != "roadside" && f.mode != "tree") continue;
        // 塘边庙要先有塘：没挖塘的村（坡上、山顶）就不修
        bool wants_pond = f.gets("face", "road") == "pond";
        for (auto& [k, wt] : f.site) wants_pond = wants_pond || (k == "pond" && wt > 0.0);
        if (wants_pond && !any_pond) continue;
        const double reach = f.getn("reach_m", 40.0);   // 离建成区边最远多远（出村大路一直通到图边，别修到几百米外）
        const int n = f.count(N);
        for (int q = 0; q < n; ++q) {
            const bool tree = f.mode == "tree";
            const double bw = tree ? 0.0 : f.getr("size_w_m", {2, 3}).sample(r), bd = tree ? 0.0 : f.getr("size_d_m", {2, 2.6}).sample(r);
            const double tr = tree ? f.getr("radius_m", {4, 7}).sample(r) : 0.0;
            const std::string face = f.gets("face", "road");
            double best = -INF;
            Obb arg{};
            // 候选：路两边，再加塘的四周
            struct P {
                V2 p, t;
                double hw;
            };
            std::vector<P> cand;
            for (const RoadPt& x : rp) cand.push_back({x.p, x.t, 0.5 * x.width});
            for (const Feature& ft : w.plan.features)
                if (ft.kind == "pond")
                    for (int k = 0; k < 16; ++k) {
                        const V2 d = bearing_vec(2 * PI * k / 16);
                        cand.push_back({ft.p + d * (ft.r * 1.25), {d.y, -d.x}, 1.0});
                    }
            for (const P& c : cand)
                for (double sg : {1.0, -1.0}) {
                    const V2 nrm = V2{-c.t.y, c.t.x} * sg;
                    Obb o;
                    if (tree) o = {c.p + nrm * (c.hw + 0.8 + 0.5 * tr), 0.0, 0.5 * tr, 0.5 * tr};
                    else {
                        const V2 ctr = c.p + nrm * (c.hw + 1.0 + 0.5 * bd);
                        double fac;
                        if (face == "road") fac = bearing_of(nrm * -1.0);
                        else if (face == "pond") {
                            double dm = INF;
                            fac = w.f.sun_bearing;
                            for (const Feature& ft : w.plan.features)
                                if (ft.kind == "pond" && len(ft.p - ctr) < dm) dm = len(ft.p - ctr), fac = bearing_of(ft.p - ctr);
                        } else if (face == "water") {
                            fac = orient_ctx(w, ctr).water;
                            if (!std::isfinite(fac)) fac = w.f.sun_bearing;
                        } else fac = w.f.sun_bearing;
                        o = {ctr, fac, 0.5 * bw, 0.5 * bd};
                    }
                    if (wants_pond) {
                        double dp = INF;
                        for (const Feature& ft : w.plan.features)
                            if (ft.kind == "pond") dp = std::min(dp, len(ft.p - o.c) - ft.r);
                        if (dp > f.getn("pond_reach_m", 20.0)) continue;
                    } else if (at(w, db, o.c, 1e9f) > reach) continue;
                    if (!obb_free(w, o, 0.0, false, 12.0) || road_hits_obb(w, o, -0.3)) continue;   // 离路边至少 0.3 m
                    const double sc = site_score(w, f, o.c) + 0.05 * r.random();
                    if (sc > best) best = sc, arg = o;
                }
            if (!std::isfinite(best)) break;
            if (tree) {
                Feature ft;
                ft.kind = "tree";
                ft.func = f.id;
                ft.name = f.name;
                ft.p = arg.c;
                ft.r = 2.0 * arg.hw;
                w.plan.features.push_back(ft);
                mark_obb(w, {arg.c, 0.0, 0.35 * ft.r, 0.35 * ft.r}, OCC_BUILDING, 0.0);   // 树干那一圈不让别的占
            } else add_building(w, small_building(w, f, arg, "hall"));
        }
    }
}

// ---------------------------------------------------------------- 场院
void place_threshing(Work& w) {
    const FuncSpec* f = w.st.func("threshing");
    if (!f) return;
    const Site& s = w.s;
    const int N = total_households(w);
    Rng r = w.rng("threshing");
    const double per = f->getr("households_per", {6, 12}).sample(r);
    const int n = N > 0 ? static_cast<int>(std::ceil(N / per)) : 0;
    const Range ar = f->getr("area_m2", {500, 2000}), ring = f->getr("ring_m", {4, 70}), asp = f->getr("aspect", {1, 1.6});
    const double spacing = f->getn("spacing_m", 45.0), ms = f->getn("max_slope_deg", 2.5);
    const GridF db = built_distance(w);
    const Mask body = built_outline(w);
    const int stride = std::max(1, static_cast<int>(std::lround(4.0 / s.res_m)));
    std::vector<V2> placed;
    for (int q = 0; q < n; ++q) {
        const double A = ar.sample(r), ap = asp.sample(r);
        const double a = std::sqrt(A * ap), b = A / a;
        const double rot = w.facing + (r.random() < 0.5 ? 0.0 : 0.5 * PI);
        double best = -INF;
        Obb arg{};
        const double pmax = f->getn("platform_max_m", 1.5);
        for (int pass = 0; pass < 2 && !std::isfinite(best); ++pass) {
            const double slope_lim = pass == 0 ? ms : 12.0, cut_lim = pass == 0 ? 0.5 : pmax;
            for (int i = 0; i < s.H; i += stride)
                for (int j = 0; j < s.W; j += stride) {
                    if (db(i, j) < ring.lo + 0.5 * b || db(i, j) > ring.hi + 0.5 * a) continue;
                    if (w.plan.occ(i, j) != OCC_FREE || !(w.f.slope_deg(i, j) <= slope_lim)) continue;
                    const V2 p = s.center(i, j);
                    bool near = false;
                    for (const V2& x : placed) near = near || len(x - p) < spacing + 0.5 * a;
                    if (near) continue;
                    const Obb o{p, rot, 0.5 * a, 0.5 * b};
                    if (body(i, j) || !obb_free(w, o, 0.0, false, slope_lim) || !outside_outline(w, body, o)) continue;
                    const double cut = ground_max_cut(w, o, nullptr);
                    if (cut > cut_lim) continue;
                    const double sc = site_score(w, *f, p) + 0.05 * r.random() - 0.3 * cut;
                    if (sc > best) best = sc, arg = o;
                }
        }
        if (!std::isfinite(best)) break;
        Feature ft;
        ft.kind = "threshing";
        ft.func = f->id;
        ft.name = f->name;
        ft.p = arg.c;
        ft.facing = arg.facing;
        const auto cs = corners(arg);
        ft.poly.assign(cs.begin(), cs.end());
        ground_max_cut(w, arg, &ft.z);
        w.plan.features.push_back(ft);
        mark_obb(w, arg, OCC_THRESH, 0.0);
        placed.push_back(arg.c);
        // 一条小路接到最近的路
        const V2 dir = w.center - arg.c;
        PathParams pp = w.pp;
        const double width = road_width(w, RC_PATH, r);
        pp.bbox_margin = static_cast<int>(std::lround(150.0 / s.res_m));
        pp.clearance_m = 0.5 * width + 0.2;
        // 场边离村心最近的那一侧中点，再往外 clearance + 0.5
        V2 from = arg.c;
        {
            double bd = INF;
            const auto cs = corners(arg);
            for (int k = 0; k < 4; ++k) {
                const V2 m = lerp(cs[k], cs[(k + 1) % 4], 0.5);
                if (len(m - w.center) < bd) bd = len(m - w.center), from = m;
            }
            from = from + (from - arg.c) * ((pp.clearance_m + 0.5) / std::max(1e-6, len(from - arg.c)));
        }
        (void)dir;
        std::vector<V2> path;
        double fit = INF;
        if (find_path(s, w.blocked, w.road_mask, w.road_core, pp, from, path, &fit) && path.size() >= 2) {
            std::reverse(path.begin(), path.end());
            if (road_fit(w, path) >= 1.0) add_road(w, path, RC_PATH, std::min(width, fit));
        }
    }
}

// ---------------------------------------------------------------- 井：按覆盖半径贪心
void place_wells(Work& w) {
    const FuncSpec* f = w.st.func("well");
    if (!f) return;
    Rng r = w.rng("wells");
    const double R = f->getr("radius_m", {70, 100}).sample(r);
    std::vector<V2> gates;
    for (const Compound& c : w.plan.compounds)
        if (c.kind == "house") gates.push_back(c.gate);
    if (gates.empty()) return;
    struct Cand {
        V2 p;
        double pref;
    };
    std::vector<Cand> cand;
    for (const RoadPt& x : road_points(w, RC_LANE, 5.0)) {
        const V2 p = x.p + V2{-x.t.y, x.t.x} * std::max(0.0, 0.5 * x.width - 0.9);
        int i, j;
        if (!w.s.cell_of(p, i, j) || w.s.water(i, j) != WATER_NONE || w.plan.occ(i, j) == OCC_BRIDGE) continue;
        cand.push_back({p, site_score(w, *f, p)});
    }
    std::vector<uint8_t> covered(gates.size(), 0);
    size_t left = gates.size();
    while (left > 0 && !cand.empty()) {
        double best = -INF;
        int arg = -1;
        for (int k = 0; k < static_cast<int>(cand.size()); ++k) {
            int cnt = 0;
            for (size_t g = 0; g < gates.size(); ++g) cnt += !covered[g] && len(gates[g] - cand[k].p) <= R;
            const double sc = cnt + 0.3 * cand[k].pref;
            if (cnt > 0 && sc > best) best = sc, arg = k;
        }
        if (arg < 0) break;
        Feature ft;
        ft.kind = "well";
        ft.func = f->id;
        ft.name = f->name;
        ft.p = cand[arg].p;
        ft.r = 0.8;
        ft.z = at(w, w.s.height, ft.p, 0.0f);
        w.plan.features.push_back(ft);
        for (size_t g = 0; g < gates.size(); ++g)
            if (!covered[g] && len(gates[g] - ft.p) <= R) covered[g] = 1, --left;
        cand.erase(cand.begin() + arg);
    }
    w.plan.metrics["well_radius_m"] = R;
}

}  // namespace skyisle::town
