// 路网图、细栅格 A*、光栅化（PLAN-TOWN 7.3）。
#include "skyisle/town/network.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

#include "skyisle/town/raster.hpp"

namespace skyisle::town {

int Network::add_node(V2 p) {
    nodes.push_back({p, {}});
    return static_cast<int>(nodes.size()) - 1;
}

int Network::add_edge(int a, int b, std::vector<V2> line, int cls, double width, double cost, int street) {
    NetEdge e;
    e.a = a, e.b = b, e.cls = cls, e.width = width, e.cost = cost, e.street = street;
    if (line.empty()) line = {nodes[a].p, nodes[b].p};
    e.line = std::move(line);
    e.len = polyline_length(e.line);
    edges.push_back(std::move(e));
    const int id = static_cast<int>(edges.size()) - 1;
    nodes[a].edges.push_back(id);
    if (b != a) nodes[b].edges.push_back(id);
    return id;
}

std::vector<double> Network::dijkstra(int src, std::vector<int>* parent) const {
    std::vector<double> d(nodes.size(), INF);
    if (parent) parent->assign(nodes.size(), -1);
    if (src < 0) return d;
    using Q = std::pair<double, int>;
    std::priority_queue<Q, std::vector<Q>, std::greater<Q>> pq;
    d[src] = 0.0;
    pq.push({0.0, src});
    while (!pq.empty()) {
        auto [dd, n] = pq.top();
        pq.pop();
        if (dd > d[n]) continue;
        for (int e : nodes[n].edges) {
            const NetEdge& E = edges[e];
            const int m = E.a == n ? E.b : E.a;
            const double nd = dd + E.len * E.cost;
            if (nd < d[m]) {
                d[m] = nd;
                if (parent) (*parent)[m] = e;
                pq.push({nd, m});
            }
        }
    }
    return d;
}

V2 Network::point_at(int e, double s, V2* tangent) const {
    const std::vector<V2>& L = edges[e].line;
    for (size_t k = 0; k + 1 < L.size(); ++k) {
        const double l = len(L[k + 1] - L[k]);
        if (s <= l || k + 2 == L.size()) {
            const double t = l > 0 ? std::clamp(s / l, 0.0, 1.0) : 0.0;
            if (tangent) *tangent = l > 0 ? (L[k + 1] - L[k]) * (1.0 / l) : V2{1, 0};
            return lerp(L[k], L[k + 1], t);
        }
        s -= l;
    }
    if (tangent) *tangent = {1, 0};
    return L.back();
}

int Network::degree_used(int n) const {
    int k = 0;
    for (int e : nodes[n].edges) {
        const NetEdge& E = edges[e];
        const bool reach = (E.a == n && E.used_a > 0.0) || (E.b == n && E.used_b > 0.0) || full(e);
        k += reach ? 1 : 0;
    }
    return k;
}

// ---------------------------------------------------------------- 细栅格 A*
namespace {

struct Step {
    int di, dj;
    double len;
    int ci0, cj0, ci1, cj1;   // 马步经过的两个格（相对）；直走 / 斜走时同终点
};

const std::vector<Step>& steps16() {
    static const std::vector<Step> S = [] {
        std::vector<Step> v;
        for (int di = -2; di <= 2; ++di)
            for (int dj = -2; dj <= 2; ++dj) {
                const int a = std::abs(di), b = std::abs(dj);
                if ((a == 0 && b == 0) || (a == 2 && b != 1) || (b == 2 && a != 1) || (a == 2 && b == 2)) continue;
                Step s{di, dj, std::hypot(di, dj), di, dj, di, dj};
                if (a == 2) s.ci0 = di / 2, s.cj0 = 0, s.ci1 = di / 2, s.cj1 = dj;
                if (b == 2) s.ci0 = 0, s.cj0 = dj / 2, s.ci1 = di, s.cj1 = dj / 2;
                v.push_back(s);
            }
        return v;
    }();
    return S;
}

}  // namespace

namespace {
bool find_path_impl(const Site& s, const Mask& blocked, const Mask& road, const Mask& goal, const PathParams& pp, V2 from,
                    std::vector<V2>& out, bool goal_exempt);
}

bool find_path(const Site& s, const Mask& blocked, const Mask& road, const Mask& goal, const PathParams& pp, V2 from,
               std::vector<V2>& out, double* fit_width) {
    // 先严：接入点（已有路上）也要离墙 clearance；接不上再放宽到接入点不管间距
    bool ok = find_path_impl(s, blocked, road, goal, pp, from, out, false);
    if (!ok && pp.clearance_m > 0.0 && pp.allow_squeeze) ok = find_path_impl(s, blocked, road, goal, pp, from, out, true);
    if (ok && fit_width) {
        // 沿路每 0.5 m 看离最近障碍格（格心）多远：路宽 = 2 × (距离 − 半格 − 0.1)
        double dmin = INF;
        const double L = polyline_length(out);
        const int r = static_cast<int>(std::ceil(4.0 / s.res_m));
        for (double a = 1.5; a <= L; a += 0.5) {
            const std::vector<V2> q = subline(out, a, std::min(L, a + 1e-3));
            if (q.empty()) break;
            int ci, cj;
            if (!s.cell_of(q.front(), ci, cj)) continue;
            for (int i = std::max(0, ci - r); i <= std::min(s.H - 1, ci + r); ++i)
                for (int j = std::max(0, cj - r); j <= std::min(s.W - 1, cj + r); ++j)
                    if (blocked(i, j)) dmin = std::min(dmin, len(s.center(i, j) - q.front()) - 0.5 * s.res_m);
        }
        *fit_width = std::isfinite(dmin) ? std::max(1.0, 2.0 * (dmin - 0.1)) : INF;
    }
    return ok;
}

namespace {

bool find_path_impl(const Site& s, const Mask& blocked, const Mask& road, const Mask& goal, const PathParams& pp, V2 from,
                    std::vector<V2>& out, bool goal_exempt) {
    out.clear();
    int si, sj;
    if (!s.cell_of(from, si, sj)) return false;
    // 搜索范围
    int i0 = 0, j0 = 0, i1 = s.H - 1, j1 = s.W - 1;
    if (pp.bbox_margin > 0) {
        i0 = std::max(0, si - pp.bbox_margin), i1 = std::min(s.H - 1, si + pp.bbox_margin);
        j0 = std::max(0, sj - pp.bbox_margin), j1 = std::min(s.W - 1, sj + pp.bbox_margin);
    }
    const int h = i1 - i0 + 1, w = j1 - j0 + 1;
    auto lid = [&](int i, int j) { return static_cast<size_t>(i - i0) * w + (j - j0); };
    const double res = s.res_m;
    // 水面宽：水格到最近岸的距离 × 2（只算一次，缓存在静态里不安全——每次按范围算）
    Mask wet(h, w, 0);
    bool any_wet = false;
    for (int i = i0; i <= i1; ++i)
        for (int j = j0; j <= j1; ++j)
            if (s.water(i, j) != WATER_NONE) wet(i - i0, j - j0) = 1, any_wet = true;
    GridF wd;
    if (any_wet) {
        Mask dry(h, w, 0);
        for (size_t k = 0; k < dry.size(); ++k) dry.v[k] = wet.v[k] ? 0 : 1;
        edt(dry, wd, nullptr);
    }
    // 启发：到目标的直线距离（整窗时用欧氏距离变换；小范围直接 Dijkstra）
    GridF hd;
    bool use_h = false;
    if (pp.bbox_margin <= 0) {
        Mask g(h, w, 0);
        bool any = false;
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j)
                if (goal(i, j)) g(i - i0, j - j0) = 1, any = true;
        if (!any) return false;
        edt(g, hd, nullptr);
        use_h = true;
    }
    const double hfac = res * std::min(1.0, pp.road_factor);
    // 离障碍的距离（格）：路中线要离院墙 clearance 以上
    GridF bd;
    const bool use_clear = pp.clearance_m > 0.0;
    if (use_clear) {
        Mask bl(h, w, 0);
        bool any = false;
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j)
                if (blocked(i, j)) bl(i - i0, j - j0) = 1, any = true;
        if (any) edt(bl, bd, nullptr);
    }
    const double exempt2 = 0.0;   // 调用方都从间距以外起步：起点那一格以外一律要间距
    auto he = [&](int i, int j) {
        const double z = s.height(i, j);
        if (s.water(i, j) != WATER_NONE && std::isfinite(s.water_level(i, j))) return std::max(z, static_cast<double>(s.water_level(i, j)) + 0.8);
        return z;
    };
    auto passable = [&](int i, int j) {
        if (i < i0 || j < j0 || i > i1 || j > j1) return false;
        if (s.sky(i, j) || blocked(i, j) || !std::isfinite(s.height(i, j))) return false;
        if (any_wet && wet(i - i0, j - j0) && 2.0 * wd(i - i0, j - j0) * res > pp.bridge_max_m) return false;
        if (s.water(i, j) == WATER_LAKE || s.water(i, j) == WATER_SEA) return false;   // 湖海不架桥
        if (use_clear && !bd.v.empty() && !(goal_exempt && goal(i, j)) && bd(i - i0, j - j0) * res < pp.clearance_m &&
            static_cast<double>((i - si) * (i - si) + (j - sj) * (j - sj)) > exempt2)
            return false;
        return true;
    };
    std::vector<float> g(static_cast<size_t>(h) * w, std::numeric_limits<float>::infinity());
    std::vector<int32_t> par(g.size(), -1);
    std::vector<uint8_t> closed(g.size(), 0);
    using Q = std::pair<double, int32_t>;
    std::priority_queue<Q, std::vector<Q>, std::greater<Q>> pq;
    const size_t s0 = lid(si, sj);
    g[s0] = 0.0f;
    pq.push({use_h ? hd.v[s0] * hfac : 0.0, static_cast<int32_t>(s0)});
    int32_t found = -1;
    while (!pq.empty()) {
        const int32_t cur = pq.top().second;
        pq.pop();
        if (closed[cur]) continue;
        closed[cur] = 1;
        const int ci = cur / w + i0, cj = cur % w + j0;
        if (goal(ci, cj) && static_cast<size_t>(cur) != s0 && s.water(ci, cj) == WATER_NONE) {   // 只在岸上接进路网：不在桥当中岔出去
            found = cur;
            break;
        }
        if (goal(ci, cj) && static_cast<size_t>(cur) == s0) {
            found = cur;
            break;
        }
        const double hc = he(ci, cj);
        for (const Step& st : steps16()) {
            const int ni = ci + st.di, nj = cj + st.dj;
            if (!passable(ni, nj)) continue;
            if ((st.ci0 != st.di || st.cj0 != st.dj) && (!passable(ci + st.ci0, cj + st.cj0) || !passable(ci + st.ci1, cj + st.cj1))) continue;
            const size_t nid = lid(ni, nj);
            if (closed[nid]) continue;
            const double L = st.len * res;
            const double grade = std::fabs(he(ni, nj) - hc) / L;
            const bool on_road = road(ni, nj) != 0;
            // 上下桥的那一步不看纵坡：桥面搭在两岸上，岸比水面高多少都能架（宽窄由 bridge_max_m 管）
            const bool bridge_step = s.water(ni, nj) != WATER_NONE || s.water(ci, cj) != WATER_NONE;
            if (grade > pp.max_grade && !on_road && !bridge_step) continue;
            // 水上：不从已有的桥当中拐下去另架一座，也不从半河里并上已有的桥（桥是两岸之间一整座）
            if (s.water(ni, nj) != WATER_NONE && s.water(ci, cj) != WATER_NONE && (road(ci, cj) != 0) != on_road) continue;
            double f = 1.0 + pp.slope_k * (grade / pp.max_grade) * (grade / pp.max_grade);
            if (s.water(ni, nj) != WATER_NONE) f = pp.water_cost;
            else if (on_road) f *= pp.road_factor;
            else {
                if (s.farmland(ni, nj)) f += pp.farm_cost;
                if (s.flood(ni, nj)) f += pp.flood_cost;
            }
            const double ng = g[cur] + L * f;
            if (ng < g[nid]) {
                g[nid] = static_cast<float>(ng);
                par[nid] = cur;
                pq.push({ng + (use_h ? hd.v[nid] * hfac : 0.0), static_cast<int32_t>(nid)});
            }
        }
    }
    if (found < 0) return false;
    std::vector<V2> pts;
    for (int32_t c = found; c >= 0; c = par[c]) pts.push_back(s.center(c / w + i0, c % w + j0));
    std::reverse(pts.begin(), pts.end());
    pts.front() = from;
    if (pts.size() == 1) pts.push_back(from);
    pts = douglas_peucker(pts, 0.6 * res);
    chaikin(pts, nullptr, 2);
    out = std::move(pts);
    return true;
}

}  // namespace

// ---------------------------------------------------------------- 光栅化
void raster_line(const Site& s, const std::vector<V2>& line, double width, const std::function<void(int, int)>& f) {
    const double hw = 0.5 * width;
    for (size_t k = 0; k + 1 < line.size(); ++k) {
        const V2 a = line[k], b = line[k + 1];
        int ia, ja, ib, jb;
        s.cell_of({std::min(a.x, b.x) - hw - s.res_m, std::max(a.y, b.y) + hw + s.res_m}, ia, ja);
        s.cell_of({std::max(a.x, b.x) + hw + s.res_m, std::min(a.y, b.y) - hw - s.res_m}, ib, jb);
        ia = std::max(ia, 0), ja = std::max(ja, 0), ib = std::min(ib, s.H - 1), jb = std::min(jb, s.W - 1);
        for (int i = ia; i <= ib; ++i)
            for (int j = ja; j <= jb; ++j)
                if (dist_point_segment(s.center(i, j), a, b) <= hw) f(i, j);
    }
    if (line.size() == 1) {
        int i, j;
        if (s.cell_of(line[0], i, j)) f(i, j);
    }
}

void raster_obb(const Site& s, const Obb& o, double shrink, const std::function<void(int, int)>& f) {
    const auto c = corners(o);
    double xmin = c[0].x, xmax = c[0].x, ymin = c[0].y, ymax = c[0].y;
    for (const V2& p : c) xmin = std::min(xmin, p.x), xmax = std::max(xmax, p.x), ymin = std::min(ymin, p.y), ymax = std::max(ymax, p.y);
    int ia, ja, ib, jb;
    s.cell_of({xmin, ymax}, ia, ja);
    s.cell_of({xmax, ymin}, ib, jb);
    ia = std::max(ia, 0), ja = std::max(ja, 0), ib = std::min(ib, s.H - 1), jb = std::min(jb, s.W - 1);
    const V2 fv = bearing_vec(o.facing), rv = bearing_right(o.facing);
    const double hd = o.hd - shrink, hw = o.hw - shrink;
    if (hd <= 0 || hw <= 0) return;
    for (int i = ia; i <= ib; ++i)
        for (int j = ja; j <= jb; ++j) {
            const V2 d = s.center(i, j) - o.c;
            if (std::fabs(dot(d, fv)) <= hd && std::fabs(dot(d, rv)) <= hw) f(i, j);
        }
}

void raster_polygon(const Site& s, const std::vector<V2>& poly, const std::function<void(int, int)>& f) {
    if (poly.size() < 3) return;
    double xmin = poly[0].x, xmax = xmin, ymin = poly[0].y, ymax = ymin;
    for (const V2& p : poly) xmin = std::min(xmin, p.x), xmax = std::max(xmax, p.x), ymin = std::min(ymin, p.y), ymax = std::max(ymax, p.y);
    int ia, ja, ib, jb;
    s.cell_of({xmin, ymax}, ia, ja);
    s.cell_of({xmax, ymin}, ib, jb);
    ia = std::max(ia, 0), ja = std::max(ja, 0), ib = std::min(ib, s.H - 1), jb = std::min(jb, s.W - 1);
    for (int i = ia; i <= ib; ++i)
        for (int j = ja; j <= jb; ++j)
            if (point_in_polygon(poly, s.center(i, j))) f(i, j);
}

std::vector<std::pair<double, double>> water_runs(const Site& s, const std::vector<V2>& line) {
    std::vector<std::pair<double, double>> out;
    const double step = 0.5 * s.res_m;
    double acc = 0.0, start = -1.0;
    for (size_t k = 0; k + 1 < line.size(); ++k) {
        const double l = len(line[k + 1] - line[k]);
        const int n = std::max(1, static_cast<int>(std::ceil(l / step)));
        for (int q = 0; q < n; ++q) {
            const double t = static_cast<double>(q) / n;
            int i, j;
            const bool wet = s.cell_of(lerp(line[k], line[k + 1], t), i, j) && (s.water(i, j) == WATER_RIVER || s.water(i, j) == WATER_STREAM);
            const double at = acc + t * l;
            if (wet && start < 0) {
                // 岸边几格干的（沙嘴、交汇处的尖）不把一座桥断成几座：离上一段不到 3 m 就接上
                if (!out.empty() && at - out.back().second < 3.0) start = out.back().first, out.pop_back();
                else start = at;
            }
            if (!wet && start >= 0) out.emplace_back(start, at), start = -1.0;
        }
        acc += l;
    }
    if (start >= 0) out.emplace_back(start, acc);
    return out;
}

std::vector<V2> subline(const std::vector<V2>& line, double s0, double s1) {
    std::vector<V2> out;
    if (line.size() < 2 || s1 <= s0) return out;
    double acc = 0.0;
    for (size_t k = 0; k + 1 < line.size(); ++k) {
        const double l = len(line[k + 1] - line[k]);
        const double a = acc, b = acc + l;
        if (b >= s0 && a <= s1 && l > 0) {
            const double t0 = std::clamp((s0 - a) / l, 0.0, 1.0), t1 = std::clamp((s1 - a) / l, 0.0, 1.0);
            const V2 p0 = lerp(line[k], line[k + 1], t0), p1 = lerp(line[k], line[k + 1], t1);
            if (out.empty() || len(out.back() - p0) > 1e-9) out.push_back(p0);
            out.push_back(p1);
        }
        acc = b;
    }
    return out;
}

}  // namespace skyisle::town
