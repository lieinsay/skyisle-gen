// 校验与形态指标（PLAN-TOWN 第八节）：TP-hh / overlap / ground / access / anchor / func 硬项，TP-shift / style 软项；
// 指标：外包长宽比 λ、形状指数 S（30 m 虚边界）、覆盖率、朝阳比例、Clark–Evans R、丁字 / 十字路口、离井 / 离田中位数、密度、路长、土方。
#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "plan_work.hpp"
#include "skyisle/town/raster.hpp"

namespace skyisle::town {

namespace {

std::vector<V2> convex_hull(std::vector<V2> p) {
    std::sort(p.begin(), p.end(), [](V2 a, V2 b) { return a.x != b.x ? a.x < b.x : a.y < b.y; });
    if (p.size() < 3) return p;
    std::vector<V2> h(2 * p.size());
    size_t k = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        while (k >= 2 && cross(h[k - 1] - h[k - 2], p[i] - h[k - 2]) <= 0) --k;
        h[k++] = p[i];
    }
    for (size_t i = p.size() - 1, t = k + 1; i > 0; --i) {
        while (k >= t && cross(h[k - 1] - h[k - 2], p[i - 1] - h[k - 2]) <= 0) --k;
        h[k++] = p[i - 1];
    }
    h.resize(k - 1);
    return h;
}

// 最小面积外接矩形的长、短边（旋转卡壳的朴素版：每条凸包边试一次）
void min_rect(const std::vector<V2>& hull, double& L, double& S) {
    L = S = 0.0;
    double best = INF;
    for (size_t i = 0; i < hull.size(); ++i) {
        const V2 e = hull[(i + 1) % hull.size()] - hull[i];
        const double l = len(e);
        if (l < 1e-9) continue;
        const V2 u = e * (1.0 / l), v{-u.y, u.x};
        double a0 = INF, a1 = -INF, b0 = INF, b1 = -INF;
        for (const V2& q : hull) a0 = std::min(a0, dot(q, u)), a1 = std::max(a1, dot(q, u)), b0 = std::min(b0, dot(q, v)), b1 = std::max(b1, dot(q, v));
        const double area = (a1 - a0) * (b1 - b0);
        if (area < best) best = area, L = std::max(a1 - a0, b1 - b0), S = std::min(a1 - a0, b1 - b0);
    }
}

double median(std::vector<double> v) {
    if (v.empty()) return NaN;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

void check(Work& w, const std::string& id, bool hard, bool ok, const std::string& msg) { w.plan.checks.push_back({id, hard, ok, msg}); }

bool seg_cross(V2 a, V2 b, V2 c, V2 d) {
    const double d1 = cross(b - a, c - a), d2 = cross(b - a, d - a), d3 = cross(d - c, a - c), d4 = cross(d - c, b - c);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0)) && d1 != 0 && d2 != 0 && d3 != 0 && d4 != 0;
}

// 两条路相交或相接（一条的某个顶点落在另一条上——路网图里共用的节点就是这样）
bool lines_cross(const std::vector<V2>& A, const std::vector<V2>& B) {
    for (size_t i = 0; i + 1 < A.size(); ++i)
        for (size_t j = 0; j + 1 < B.size(); ++j)
            if (seg_cross(A[i], A[i + 1], B[j], B[j + 1])) return true;
    for (const V2& p : A)
        if (dist_point_polyline(p, B) < 0.5) return true;
    for (const V2& p : B)
        if (dist_point_polyline(p, A) < 0.5) return true;
    return false;
}

// 以 g 为心、长轴朝 ang 的 L × S 矩形里能盖房的地占几成（2 m 取一个样；虚空不算进分母）
double room_in_rect(const Work& w, V2 g, double ang, double L, double S) {
    const V2 u = bearing_vec(ang), v = bearing_right(ang);
    int n = 0, k = 0;
    for (double a = -0.5 * L; a <= 0.5 * L; a += 2.0)
        for (double b = -0.5 * S; b <= 0.5 * S; b += 2.0) {
            int i, j;
            if (!w.s.cell_of(g + u * a + v * b, i, j) || w.s.sky(i, j)) continue;
            ++n;
            k += w.f.buildable(i, j) ? 1 : 0;
        }
    return n ? static_cast<double>(k) / n : 0.0;
}

std::string fmt(double x, int prec = 2) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(prec);
    o << x;
    return o.str();
}

}  // namespace

void verify(Work& w) {
    Plan& P = w.plan;
    const Site& s = w.s;
    const Style& st = w.st;
    const double cell = s.res_m * s.res_m;
    std::map<std::string, double>& M = P.metrics;

    // ---- TP-hh
    {
        int placed = 0, want = static_cast<int>(P.households.size());
        int kinds[3] = {0, 0, 0}, kinds_placed[3] = {0, 0, 0};
        for (const Household& h : P.households) {
            ++kinds[h.kind];
            if (h.compound >= 0) ++placed, ++kinds_placed[h.kind];
        }
        const bool ok = placed == want && want == w.req.hh_farm + w.req.hh_market + w.req.hh_special;
        // 住不下时说一句地的情形：村心一带（1.5 R + 60 m）能盖房的地（坡 ≤ 风格上限、不是水、崖缘、漫水）占几成
        std::string why;
        {
            const double rad = 1.5 * w.R + 60.0;
            int ci, cj, n = 0, k = 0;
            if (s.cell_of(w.center, ci, cj)) {
                const int rc = static_cast<int>(std::ceil(rad / s.res_m));
                for (int i = std::max(0, ci - rc); i <= std::min(s.H - 1, ci + rc); i += 2)
                    for (int j = std::max(0, cj - rc); j <= std::min(s.W - 1, cj + rc); j += 2) {
                        if (len(s.center(i, j) - w.center) > rad || s.sky(i, j)) continue;
                        ++n;
                        k += w.f.buildable(i, j) ? 1 : 0;
                    }
            }
            const double share = n ? static_cast<double>(k) / n : 0.0;
            M["buildable_share_near"] = share;
            if (!ok) {
                std::ostringstream o;
                o << "；村心一带能盖房的地（坡 ≤ " << std::lround(w.st.max_slope_deg) << "°、不是水 / 崖缘" << (w.st.allow_flood ? "" : " / 漫水") << "）只占 "
                  << std::lround(100.0 * share) << "%";
                // 算子自己要平地（地坑院、散居村、環濠集落、绿地村）而这里平处不够：多半是 --operator 硬点了不合这块地的算子
                auto fd = w.st.op_flat_deg.find(P.op), fs = w.st.op_flat_share.find(P.op);
                if (fd != w.st.op_flat_deg.end() && fs != w.st.op_flat_share.end()) {
                    const double flat = 1.0 - total_slope_share(w, w.center, 1.2 * w.R + 40.0, fd->second);
                    if (flat < fs->second)
                        o << "；这个算子要坡 ≤ " << std::lround(fd->second) << "° 的平地占到 " << std::lround(100.0 * fs->second) << "%，这里只有 "
                          << std::lround(100.0 * flat) << "%（不合这块地）";
                }
                why = o.str();
            }
        }
        check(w, "TP-hh", true, ok, "住下 " + std::to_string(placed) + " / " + std::to_string(want) + " 户（农 " + std::to_string(kinds_placed[0]) + "、市 " +
                                       std::to_string(kinds_placed[1]) + "、专业 " + std::to_string(kinds_placed[2]) + "）" + why);
        M["households"] = want;
        M["households_placed"] = placed;
    }
    // ---- TP-overlap：房两两不交；房不压路、不压水；地块两两不交
    {
        std::unordered_map<int64_t, std::vector<int>> grid;
        auto key = [](int a, int b) { return (static_cast<int64_t>(a) << 32) ^ static_cast<uint32_t>(b); };
        const double cs = 24.0;
        for (int k = 0; k < static_cast<int>(P.buildings.size()); ++k) {
            const V2 c = P.buildings[k].box.c;
            grid[key(static_cast<int>(std::floor(c.x / cs)), static_cast<int>(std::floor(c.y / cs)))].push_back(k);
        }
        int bb = 0;
        for (int k = 0; k < static_cast<int>(P.buildings.size()); ++k) {
            const V2 c = P.buildings[k].box.c;
            const int a = static_cast<int>(std::floor(c.x / cs)), b = static_cast<int>(std::floor(c.y / cs));
            for (int da = -1; da <= 1; ++da)
                for (int db = -1; db <= 1; ++db) {
                    auto it = grid.find(key(a + da, b + db));
                    if (it == grid.end()) continue;
                    for (int q : it->second)
                        if (q > k && overlap(P.buildings[k].box, P.buildings[q].box, -0.05)) ++bb;
                }
        }
        int pp = 0;
        for (size_t a = 0; a < P.compounds.size(); ++a)
            for (size_t b = a + 1; b < P.compounds.size(); ++b)
                if (len(P.compounds[a].plot.c - P.compounds[b].plot.c) < 80.0 && overlap(P.compounds[a].plot, P.compounds[b].plot, -0.1)) ++pp;
        int on_road = 0, on_water = 0;
        for (const Building& b : P.buildings) {
            bool wt = false;
            raster_obb(s, b.box, 0.5 * s.res_m, [&](int i, int j) { wt = wt || s.water(i, j) != WATER_NONE; });
            on_road += road_hits_obb(w, b.box, 0.1), on_water += wt;
        }
        check(w, "TP-overlap", true, bb == 0 && pp == 0 && on_road == 0 && on_water == 0,
              "房相交 " + std::to_string(bb) + " 对、地块相交 " + std::to_string(pp) + " 对、房压路 " + std::to_string(on_road) + " 栋、房压水 " + std::to_string(on_water) + " 栋");
    }
    // ---- TP-ground
    {
        int bad = 0, cut = 0;
        for (const Building& b : P.buildings) {
            bool x = false;
            raster_obb(s, b.box, 0.5 * s.res_m, [&](int i, int j) {
                x = x || s.sky(i, j) || s.edge(i, j) || (!st.allow_flood && s.flood(i, j));
            });
            bad += x;
        }
        int terr = 0, deep = 0;
        int dug = 0;
        for (const Compound& c : P.compounds) {
            // 靠崖窑的院子挖进崖里、围龙屋依山分台：地块高差不按台地查（各栋自己的挖填照查）
            if (st.templates[st.template_index(c.tmpl)].dug_in || st.templates[st.template_index(c.tmpl)].shape == "weilong") {
                ++dug;
                continue;
            }
            cut += c.max_cut_m > st.max_terrace_m + 1e-6, terr += c.max_cut_m > st.max_cut_m + 1e-6;
        }
        M["dug_in_compounds"] = dug;
        for (const Building& b : P.buildings) deep += b.max_cut_m > st.max_cut_m + 0.3;   // 半格的余量：台基按格心的中位
        check(w, "TP-ground", true, bad == 0 && cut == 0 && deep == 0,
              "房在虚空 / 崖缘 / 漫水上 " + std::to_string(bad) + " 栋、地块高差超 " + fmt(st.max_terrace_m, 1) + " m 的宅院 " + std::to_string(cut) +
                  " 个、台基挖填超 " + fmt(st.max_cut_m, 1) + " m 的房 " + std::to_string(deep) + " 栋（台地院 " + std::to_string(terr) + " 个）");
        M["terraced_compounds"] = terr;
    }
    // ---- 路的连通（按端点贴着别的路算相连）
    const int nr = static_cast<int>(P.roads.size());
    std::vector<int> comp(nr);
    std::iota(comp.begin(), comp.end(), 0);
    std::function<int(int)> find = [&](int x) { return comp[x] == x ? x : comp[x] = find(comp[x]); };
    int n_t = 0, n_x = 0;
    {
        std::vector<std::pair<V2, int>> ends;   // 路口：端点落在别的路上
        std::vector<std::array<double, 4>> bb(nr);
        for (int a = 0; a < nr; ++a) {
            bb[a] = {INF, -INF, INF, -INF};
            for (const V2& q : P.roads[a].line) bb[a] = {std::min(bb[a][0], q.x), std::max(bb[a][1], q.x), std::min(bb[a][2], q.y), std::max(bb[a][3], q.y)};
        }
        for (int a = 0; a < nr; ++a)
            for (int b = a + 1; b < nr; ++b)
                if (bb[a][0] <= bb[b][1] && bb[b][0] <= bb[a][1] && bb[a][2] <= bb[b][3] && bb[b][2] <= bb[a][3] &&
                    lines_cross(P.roads[a].line, P.roads[b].line))
                    comp[find(a)] = find(b);
        for (int a = 0; a < nr; ++a)
            for (const V2& e : {P.roads[a].line.front(), P.roads[a].line.back()})
                for (int b = 0; b < nr; ++b) {
                    if (a == b) continue;
                    int seg;
                    double t;
                    const double d = dist_point_polyline(e, P.roads[b].line, &seg, &t);
                    if (d <= 0.5 * P.roads[b].width_m + 1.5) {
                        comp[find(a)] = find(b);
                        ends.push_back({e, b});
                    }
                }
        // 丁字 / 十字：鱼骨有路网图就按节点度数；否则按「端点落在别的路中段」数
        bool graph = false;
        for (int e = 0; e < static_cast<int>(w.net.edges.size()); ++e) graph = graph || (w.net.used(e) && w.net.edges[e].street > 0);
        if (graph) {
            for (int n = 0; n < static_cast<int>(w.net.nodes.size()); ++n) {
                const int d = w.net.degree_used(n);
                n_t += d == 3, n_x += d >= 4;
            }
        } else {
            std::vector<V2> js;
            for (auto& [p, b] : ends) {
                const V2 f = P.roads[b].line.front(), l = P.roads[b].line.back();
                if (len(p - f) < 3.0 || len(p - l) < 3.0) continue;
                bool dup = false;
                for (const V2& q : js) dup = dup || len(q - p) < 4.0;
                if (!dup) js.push_back(p);
            }
            n_t = static_cast<int>(js.size());
        }
    }
    int main_comp = -1;
    {
        std::map<int, double> L;
        for (int a = 0; a < nr; ++a) L[find(a)] += polyline_length(P.roads[a].line);
        double bl = -1;
        for (auto& [c, l] : L)
            if (l > bl) bl = l, main_comp = c;
    }
    auto road_near = [&](V2 p, double extra) {
        for (int a = 0; a < nr; ++a)
            if (find(a) == main_comp && dist_point_polyline(p, P.roads[a].line) <= 0.5 * P.roads[a].width_m + extra) return true;
        return false;
    };
    // ---- TP-access
    if (w.req.scale != "compound") {
        int no_access = 0;
        for (const Compound& c : P.compounds) no_access += !road_near(c.access, 1.5) && !road_near(c.gate, 2.5);
        int split = 0;
        for (int a = 0; a < nr; ++a) split += find(a) != main_comp;
        bool landing_ok = true;
        for (const Feature& f : P.features)
            if (f.kind == "landing") {
                bool any = false;
                for (const V2& q : f.poly) any = any || road_near(q, 6.0);
                for (size_t k = 0; k < f.poly.size(); ++k) any = any || road_near(lerp(f.poly[k], f.poly[(k + 1) % f.poly.size()], 0.5), 6.0);
                landing_ok = any;
            }
        check(w, "TP-access", true, no_access == 0 && split == 0 && landing_ok,
              "门不接路的宅院 " + std::to_string(no_access) + " 个、不连通的路 " + std::to_string(split) + " 条" + (landing_ok ? "" : "、泊场没接上路"));
    }
    // ---- TP-anchor
    if (w.req.scale != "compound" && w.req.want_landing && st.func("landing")) {
        int n = 0;
        for (const Feature& f : P.features) n += f.kind == "landing";
        check(w, "TP-anchor", true, n > 0, n > 0 ? "泊场在" : "上游有泊场，这里没摆下");
    }
    // ---- TP-func
    const int N = static_cast<int>(P.households.size());
    std::vector<V2> wells;
    for (const Feature& f : P.features)
        if (f.kind == "well") wells.push_back(f.p);
    std::vector<double> dwell;
    for (const Compound& c : P.compounds) {
        if (c.kind != "house") continue;
        double d = INF;
        for (const V2& q : wells) d = std::min(d, len(q - c.gate));
        dwell.push_back(d);
    }
    if (w.req.scale != "compound") {
        std::string missing;
        for (const FuncSpec& f : st.funcs) {
            if (!f.required || !f.allows(P.op)) continue;
            if ((f.mode == "compound" || f.mode == "roadside") && f.count(N) <= 0) continue;
            if (f.mode == "landing" && !w.req.want_landing) continue;
            bool have = false;
            if (f.mode == "compound")
                for (const Compound& c : P.compounds) have = have || c.func == f.id;
            else if (f.mode == "roadside")
                for (const Building& b : P.buildings) have = have || b.func == f.id;
            else
                for (const Feature& ft : P.features) have = have || ft.func == f.id;
            if (!have) missing += (missing.empty() ? "" : "、") + f.name;
        }
        const double R = M.count("well_radius_m") ? M["well_radius_m"] : 0.0;
        int far = 0;
        for (double d : dwell) far += d > R + 1e-6;
        check(w, "TP-func", true, missing.empty() && far == 0,
              (missing.empty() ? std::string("要求的设施都在") : "缺：" + missing) + "；离井超过 " + fmt(R, 0) + " m 的宅院 " + std::to_string(far) + " 个");
    }
    // ---- TP-shift（软）
    const double shift = len(w.center - w.req.anchor);
    check(w, "TP-shift", false, shift <= st.center_search_m + 1e-6, "村心离上游点位 " + fmt(shift, 0) + " m");
    M["center_shift_m"] = shift;

    // ---- 形态指标
    std::vector<V2> pts, ctrs;
    double footprint = 0.0;
    for (const Compound& c : P.compounds) {
        const auto cs = corners(c.plot);
        pts.insert(pts.end(), cs.begin(), cs.end());
        if (c.kind == "house") ctrs.push_back(c.plot.c);
    }
    for (const Building& b : P.buildings)
        if (b.compound >= 0) footprint += 4.0 * b.box.hw * b.box.hd;
    M["compounds"] = static_cast<double>(P.compounds.size());
    M["buildings"] = static_cast<double>(P.buildings.size());
    M["footprint_m2"] = footprint;
    if (pts.size() >= 3) {
        double L, S;
        min_rect(convex_hull(pts), L, S);
        M["lambda"] = S > 0 ? L / S : NaN;
        M["extent_long_m"] = L;
        M["extent_short_m"] = S;
    }
    // 30 m 虚边界的外廓（闭运算：离地块 ≤ 15 m 的区域，再内缩 15 m）
    {
        Mask plot(s.H, s.W, 0);
        bool any = false;
        for (size_t k = 0; k < plot.size(); ++k) plot.v[k] = P.occ.v[k] == OCC_PLOT || P.occ.v[k] == OCC_BUILDING, any = any || plot.v[k];
        if (any) {
            GridF d1;
            edt(plot, d1, nullptr);
            const float r = static_cast<float>(15.0 / s.res_m);
            Mask outside(s.H, s.W, 0);
            for (size_t k = 0; k < plot.size(); ++k) outside.v[k] = d1.v[k] > r;
            GridF d2;
            edt(outside, d2, nullptr);
            Mask body(s.H, s.W, 0);
            double A = 0.0, per = 0.0;
            for (size_t k = 0; k < plot.size(); ++k) body.v[k] = d2.v[k] > r || plot.v[k];
            for (int i = 0; i < s.H; ++i)
                for (int j = 0; j < s.W; ++j) {
                    if (!body(i, j)) continue;
                    A += cell;
                    for (auto& o : N4) {
                        const int a = i + o[0], b = j + o[1];
                        if (!body.in(a, b) || !body(a, b)) per += s.res_m;
                    }
                }
            M["outline_area_m2"] = A;
            M["shape_index"] = A > 0 ? per / (2.0 * std::sqrt(PI * A)) : NaN;
            M["coverage"] = A > 0 ? footprint / A : NaN;
            M["density_hh_per_ha"] = A > 0 ? N / (A / 1e4) : NaN;
            if (ctrs.size() >= 2) {
                double sum = 0.0;
                for (size_t a = 0; a < ctrs.size(); ++a) {
                    double d = INF;
                    for (size_t b = 0; b < ctrs.size(); ++b)
                        if (a != b) d = std::min(d, len(ctrs[a] - ctrs[b]));
                    sum += d;
                }
                const double robs = sum / ctrs.size(), rexp = 0.5 / std::sqrt(ctrs.size() / A);
                M["clark_evans"] = robs / rexp;
            }
        }
    }
    // 朝向：宅院朝向与朝阳的偏差（容差取风格 sun 规则的，没有就 15°）
    {
        double tol = 15.0 * PI / 180.0;
        for (const OrientRule& r : st.rules)
            if (r.kind == "sun") tol = r.tol;
        int ok = 0, okt = 0, n = 0;
        double dev = 0.0;
        for (const Compound& c : P.compounds) {
            if (c.kind != "house") continue;
            const double d = angle_diff(c.plot.facing, w.f.sun_bearing);
            const bool sun_ok = d <= tol + 1e-6;
            ok += sun_ok, ++n, dev += d;
            // 地形优先：坡 ≥ 4° 处朝下坡、或进深顺等高线（朝下坡 ± 90°）的，是顺着地形摆的，也合理
            bool terrain_ok = false;
            int ci, cj;
            if (!sun_ok && s.cell_of(c.plot.c, ci, cj) && w.f.slope_deg(ci, cj) >= 4.0f && std::isfinite(w.f.downslope(ci, cj)))
                for (double off : {0.0, 0.5 * PI, -0.5 * PI})
                    terrain_ok = terrain_ok || angle_diff(c.plot.facing, wrap_pi(w.f.downslope(ci, cj) + off)) <= tol + 1e-6;
            okt += sun_ok || terrain_ok;
        }
        if (n) {
            M["orient_sun_share"] = static_cast<double>(ok) / n;
            M["orient_terrain_share"] = static_cast<double>(okt) / n;
            M["orient_dev_mean_deg"] = dev / n * 180.0 / PI;
        }
    }
    M["junction_t"] = n_t;
    M["junction_x"] = n_x;
    M["t_x_ratio"] = n_x > 0 ? static_cast<double>(n_t) / n_x : NaN;
    M["well_dist_median_m"] = median(dwell);
    M["wells"] = static_cast<double>(wells.size());
    {
        Mask farm(s.H, s.W, 0);
        bool any = false;
        for (size_t k = 0; k < farm.size(); ++k) farm.v[k] = s.farmland.v[k] && P.occ.v[k] == OCC_FREE, any = any || farm.v[k];
        if (any) {
            GridF d;
            edt(farm, d, nullptr);
            std::vector<double> v;
            for (const Compound& c : P.compounds) {
                int i, j;
                if (c.kind == "house" && s.cell_of(c.gate, i, j)) v.push_back(d(i, j) * s.res_m);
            }
            M["field_dist_median_m"] = median(v);
        }
    }
    const char* cls_name[5] = {"trunk", "main", "street", "lane", "path"};
    for (const Road& r : P.roads) M[std::string("road_m_") + cls_name[std::clamp(r.cls, 0, 4)]] += polyline_length(r.line);
    M["bridges"] = static_cast<double>(P.bridges.size());
    double cut = 0.0, fill = 0.0;
    for (const Building& b : P.buildings) cut += b.cut_m3, fill += b.fill_m3;
    M["cut_m3"] = cut;
    M["fill_m3"] = fill;

    // ---- 第三步加的指标：面街、山墙朝街、贴线、面宽变异、朝向离散、顺等高线、每院栋数、院外单栋离院
    {
        std::vector<double> front, face, gable, along, dirs_x, dirs_y, contour, nb;
        int n_face = 0, n_gable = 0, n_along = 0, n_cont = 0, n_cont_ok = 0;
        for (const Compound& c : P.compounds) {
            if (c.kind != "house") continue;
            nb.push_back(static_cast<double>(c.buildings.size()));
            front.push_back(c.access_side == SIDE_FRONT || c.access_side == SIDE_BACK ? 2.0 * c.plot.hw : 2.0 * c.plot.hd);
            dirs_x.push_back(std::sin(c.plot.facing)), dirs_y.push_back(std::cos(c.plot.facing));
            // 最近的路（巷及以上）
            double bd = INF;
            V2 q{}, t{};
            double hw = 0.0;
            for (const Road& rd : P.roads) {
                if (rd.cls > RC_LANE) continue;
                int seg;
                double u;
                const double d = dist_point_polyline(c.plot.c, rd.line, &seg, &u);
                if (d < bd) {
                    bd = d, q = lerp(rd.line[seg], rd.line[seg + 1], u), hw = 0.5 * rd.width_m;
                    t = rd.line[seg + 1] - rd.line[seg];
                    t = t * (1.0 / std::max(1e-9, len(t)));
                }
            }
            if (std::isfinite(bd)) {
                ++n_face;
                if (angle_diff(c.plot.facing, bearing_of(q - c.plot.c)) <= PI / 6) face.push_back(1.0);
                // 贴线：临路那条边的中点离路边 ≤ 1.5 m
                const V2 edge = c.plot.c + [&] {
                    const V2 f = bearing_vec(c.plot.facing), rr = bearing_right(c.plot.facing);
                    return c.access_side == SIDE_FRONT ? f * c.plot.hd : c.access_side == SIDE_BACK ? f * -c.plot.hd : c.access_side == SIDE_LEFT ? rr * -c.plot.hw : rr * c.plot.hw;
                }();
                double de = INF;
                for (const Road& rd : P.roads)
                    if (rd.cls <= RC_LANE) de = std::min(de, dist_point_polyline(edge, rd.line) - 0.5 * rd.width_m);
                ++n_along;
                if (de <= 1.5) along.push_back(1.0);
                // 山墙朝街：主屋的屋脊（面阔方向）垂直于街
                for (int bi : c.buildings) {
                    const Building& b = P.buildings[bi];
                    if (b.role != "main" && b.func != "dwelling") continue;
                    ++n_gable;
                    if (std::fabs(dot(bearing_right(b.box.facing), t)) < 0.5) gable.push_back(1.0);
                    break;
                }
            }
            int i, j;
            if (s.cell_of(c.plot.c, i, j) && w.f.slope_deg(i, j) > 3.0f && std::isfinite(w.f.downslope(i, j))) {
                ++n_cont;
                n_cont_ok += angle_diff(c.plot.facing, w.f.downslope(i, j)) <= 25.0 * PI / 180.0;
            }
        }
        if (n_face) M["face_street_share"] = static_cast<double>(face.size()) / n_face;
        if (n_along) M["along_street_share"] = static_cast<double>(along.size()) / n_along;
        if (n_gable) M["gable_street_share"] = static_cast<double>(gable.size()) / n_gable;
        if (front.size() >= 3) {
            const double m = std::accumulate(front.begin(), front.end(), 0.0) / front.size();
            double v = 0.0;
            for (double x : front) v += (x - m) * (x - m);
            M["frontage_cv"] = m > 0 ? std::sqrt(v / front.size()) / m : NaN;
            M["frontage_median_m"] = median(front);
        }
        if (dirs_x.size() >= 2) {
            const double mx = std::accumulate(dirs_x.begin(), dirs_x.end(), 0.0) / dirs_x.size(), my = std::accumulate(dirs_y.begin(), dirs_y.end(), 0.0) / dirs_y.size();
            const double Rm = std::min(1.0, std::hypot(mx, my));
            M["orient_dispersion_deg"] = Rm > 0 ? std::sqrt(std::max(0.0, -2.0 * std::log(Rm))) * 180.0 / PI : 180.0;
        }
        if (n_cont >= 5) M["contour_share"] = static_cast<double>(n_cont_ok) / n_cont;
        // 临水：地块的角与边中点离水 ≤ 10 m 的宅院占比（有水时才算）
        {
            bool any_water = false;
            for (uint8_t v : s.water.v) any_water = any_water || v != WATER_NONE;
            int nh = 0, nw = 0;
            if (any_water)
                for (const Compound& c : P.compounds) {
                    if (c.kind != "house") continue;
                    ++nh;
                    const auto cs = corners(c.plot);
                    double d = INF;
                    for (int k = 0; k < 4; ++k)
                        for (V2 p : {cs[k], lerp(cs[k], cs[(k + 1) % 4], 0.5)}) d = std::min(d, static_cast<double>(field_at(s, s.water_dist_m, p, 1e9f)));
                    nw += d <= 10.0;
                }
            if (nh) M["water_front_share"] = static_cast<double>(nw) / nh;
        }
        if (!nb.empty()) M["buildings_per_compound_median"] = median(nb);
        // 院外单栋（role = detached）离自家院子多远：取最近的一栋
        double dmin = INF;
        for (const Building& b : P.buildings)
            if (b.role == "detached" && b.compound >= 0) dmin = std::min(dmin, dist_obb(P.compounds[b.compound].plot, b.box.c) - std::hypot(b.box.hw, b.box.hd));
        if (std::isfinite(dmin)) M["detached_min_m"] = dmin;
        // 巷宽中位（徽州：≤ 2.4 m）
        std::vector<double> lw;
        for (const Road& rd : P.roads)
            if (rd.cls == RC_LANE) lw.push_back(rd.width_m);
        if (!lw.empty()) M["lane_width_median_m"] = median(lw);
    }

    // ---- TP-style（软）：风格的目标区间（算子自己的目标盖过通用的）
    std::string miss;
    int nt = 0;
    std::map<std::string, Range> targets = st.targets;
    {
        auto it = st.op_targets.find(P.op);
        if (it != st.op_targets.end())
            for (auto& [k, rg] : it->second) targets[k] = rg;
    }
    // 优先次序（用户定）：地形 > 合理 > 风格。地形逼出来的偏离不算风格没做到，单列「地形所致」
    std::string excused;
    for (auto& [k, rg] : targets) {
        auto it = M.find(k);
        if (it == M.end() || !std::isfinite(it->second)) continue;
        ++nt;
        const double v = it->second;
        if (v >= rg.lo - 1e-9 && v <= rg.hi + 1e-9) continue;
        // λ 不在区间里是不是地形逼的：村心（宅院的重心）上摆一块同样大小（外包长 × 宽）、长宽比到区间那一头的块（太长了摆 λ = 上限的，太团了摆 λ = 下限的），
        // 12 个方向里最顺的那个也有一成半以上盖不了房——这块地容不下风格要的样子（河谷、海边窄条只能成带；山顶、圆丘拉不成带）
        if (k == "lambda" && M.count("extent_long_m") && !ctrs.empty()) {
            V2 g{0.0, 0.0};
            for (const V2& c : ctrs) g = g + c;
            g = g * (1.0 / static_cast<double>(ctrs.size()));
            const double lam = v > rg.hi ? std::max(1.0, rg.hi) : rg.lo;
            const double A = M["extent_long_m"] * M["extent_short_m"], L = std::sqrt(A * lam), S = std::sqrt(A / lam);
            double best = 0.0;
            for (int a = 0; a < 12; ++a) best = std::max(best, room_in_rect(w, g, a * PI / 12.0, L, S));
            M["lambda_room_share"] = best;
            if (best < 0.85) {
                excused += (excused.empty() ? "" : "、") + std::string("λ ") + fmt(v) + "（村心摆一块 λ " + fmt(lam, 1) + " 的，最顺的方向上能盖房的地也只占 " +
                           std::to_string(std::lround(100.0 * best)) + "%）";
                continue;
            }
        }
        if (k == "orient_sun_share" && v < rg.lo && M.count("orient_terrain_share") && M["orient_terrain_share"] >= rg.lo - 1e-9) {
            excused += (excused.empty() ? "" : "、") + std::string("朝阳 ") + fmt(v) + "（坡上顺等高线的算上是 " + fmt(M["orient_terrain_share"]) + "）";
            continue;
        }
        miss += (miss.empty() ? "" : "；") + k + " = " + fmt(v) + " 不在 [" + fmt(rg.lo) + ", " + fmt(rg.hi) + "]";
    }
    if (nt) {
        std::string msg = miss.empty() ? "形态指标都在风格的目标区间里（" + std::to_string(nt) + " 项）" : miss;
        if (!excused.empty()) msg += "；地形所致、不算：" + excused;
        check(w, "TP-style", false, miss.empty(), msg);
    }
}

}  // namespace skyisle::town
