// 编排（PLAN-TOWN 第五节）：场地分析 → 户与亲缘 → 村心与朝向 → 形态算子（街网、地块、落位）→ 出村大路 → 泊场 → 塘 → 路边的庙与树 →
// 场院 → 井 → 宅院成形 → 校验与指标。各步的随机流按名字分开（Work::rng）。
#include "skyisle/town/plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "plan_work.hpp"
#include "skyisle/town/raster.hpp"

namespace skyisle::town {

Rng Work::rng(const std::string& tag) const { return Rng(SeedSequence({req.seed, static_cast<uint64_t>(crc32(tag))})); }

// ---------------------------------------------------------------- 共用小件
double sample_interest(const Work& w, const Obb& o) {
    double s = 0.0;
    int n = 0;
    raster_obb(w.s, o, 0.0, [&](int i, int j) {
        const float v = w.interest(i, j);
        s += std::isfinite(v) ? v : 0.0;
        ++n;
    });
    return n ? s / n : 0.0;
}

bool slot_ground(const Work& w, Slot& sl, const Mask* extra) {
    std::vector<double> hs;
    bool ok = true;
    double isum = 0.0;
    // 左右三分之一、前后三分之一的平均高程：估地块在面阔方向、进深方向的坡
    double sL = 0, sR = 0, sB = 0, sF = 0;
    int nL = 0, nR = 0, nB = 0, nF = 0;
    const V2 fv = bearing_vec(sl.box.facing), rv = bearing_right(sl.box.facing);
    // 靠崖窑（dug_in）：院子是在坡上削出来的台，窑挖进后面的崖——整块地都是挖出来的，只要不是水、虚空、崖缘、不陡过 60°
    const bool dug = sl.tmpl >= 0 && w.st.templates[sl.tmpl].dug_in;
    raster_obb(w.s, sl.box, 0.0, [&](int i, int j) {
        bool can = w.f.buildable(i, j);
        if (!can && dug)
            can = !w.s.sky(i, j) && !w.s.edge(i, j) && w.s.water(i, j) == WATER_NONE && (w.st.allow_flood || !w.s.flood(i, j)) &&
                  w.f.slope_deg(i, j) <= 60.0f;
        if (!can || (extra && (*extra)(i, j))) ok = false;
        const double h = w.s.height(i, j);
        if (std::isfinite(h)) {
            hs.push_back(h);
            const V2 d = w.s.center(i, j) - sl.box.c;
            const double x = dot(d, rv), y = dot(d, fv);
            if (x < -sl.box.hw / 3) sL += h, ++nL;
            if (x > sl.box.hw / 3) sR += h, ++nR;
            if (y < -sl.box.hd / 3) sB += h, ++nB;
            if (y > sl.box.hd / 3) sF += h, ++nF;
        }
        const float v = w.interest(i, j);
        isum += std::isfinite(v) ? v : 0.0;
    });
    if (hs.empty()) {
        sl.valid = false;
        return false;
    }
    sl.interest = isum / static_cast<double>(hs.size());
    std::vector<double> t = hs;
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    sl.base_m = t[t.size() / 2];
    double mc = 0.0;
    for (double h : hs) mc = std::max(mc, std::fabs(h - sl.base_m));
    sl.max_cut = mc;
    // 靠崖窑：地块里的高差不按台地查（挖方记在窑上）
    // 围龙屋依山分台（前低后高、化胎在后）：地块高差放到台地上限的 2.5 倍，各栋自己查挖填
    const bool terraced = sl.tmpl >= 0 && w.st.templates[sl.tmpl].shape == "weilong";
    sl.valid = ok && (dug || mc <= (terraced ? 2.5 : 1.0) * w.st.max_terrace_m);
    // 每栋房两头的高差（面阔方向坡 × 半长 + 进深方向坡 × 半深）也得在 max_cut_m 以内：前后两边的房顺面阔、左右两边的顺进深；
    // 窑（挖进去的）与院外单栋不算
    if (nL && nR && nB && nF && sl.tmpl >= 0 && !dug) {
        const double gr = std::fabs(sR / nR - sL / nL) / (4.0 / 3.0 * sl.box.hw), gf = std::fabs(sF / nF - sB / nB) / (4.0 / 3.0 * sl.box.hd);
        const TemplateSpec& T = w.st.templates[sl.tmpl];
        double worst = 0.0;
        for (const BuildingSpec& S : T.b) {
            if (S.roof == "cave" || S.detached_m.hi > 0) continue;
            const bool along_w = S.side == SIDE_FRONT || S.side == SIDE_BACK;
            const double span = along_w ? 2.0 * sl.box.hw : 2.0 * sl.box.hd;
            const double L = S.bays.hi > 0 ? std::min(span, S.bays.hi * T.bay_m.hi) : S.len_frac.hi * span;
            const double d = S.depth_m.mid();
            worst = std::max(worst, along_w ? gr * 0.5 * L + gf * 0.5 * d : gf * 0.5 * L + gr * 0.5 * d);
        }
        if (T.b.empty() || T.shape != "yard") worst = gr * sl.box.hw + gf * 3.5;
        if (T.shape == "weilong") worst = gr * 0.5 * T.ring_room_m.hi * 4.0 + gf * 0.5 * T.ring_depth_m.hi;   // 围龙屋依山：各栋分台（前低后高），只看一段房两头的高差
        if (worst > w.st.max_cut_m + 0.2) sl.valid = false;
    }
    // 要做台地的院子（高差超过一栋房的挖填上限）排后一点
    if (mc > w.st.max_cut_m) sl.interest -= 0.5 * (mc - w.st.max_cut_m) / std::max(0.1, w.st.max_terrace_m - w.st.max_cut_m);
    return sl.valid;
}

OrientCtx orient_ctx(const Work& w, V2 p) {
    OrientCtx c;
    c.sun = w.f.sun_bearing;
    c.wind_from = w.f.wind_from;
    int i, j;
    if (w.s.cell_of(p, i, j)) {
        const float d = w.f.downslope(i, j);
        if (std::isfinite(d) && w.f.slope_deg(i, j) > 1.0f) c.downslope = d;
        if (!w.water_src.v.empty() && w.water_src(i, j) >= 0) {
            const int k = w.water_src(i, j);
            const V2 q = w.s.center(k / w.s.W, k % w.s.W);
            if (len(q - p) > 1.0) c.water = bearing_of(q - p);
        }
    }
    return c;
}

int pick_template(const Work& w, Rng& r) {
    // 围合单体（圆楼、方楼、围龙屋）只由围合算子按户数定尺寸落，这里只挑院落
    std::vector<int> ids, yards;
    for (int k : w.st.house_templates())
        if (w.st.templates[k].shape == "yard") {
            yards.push_back(k);
            if (w.st.templates[k].allows(w.plan.op)) ids.push_back(k);
        }
    if (ids.empty()) ids = yards;   // 风格没给这个算子的模板：院落都行
    if (ids.empty()) ids = w.st.house_templates();
    std::vector<double> p;
    for (int k : ids) p.push_back(w.st.templates[k].weight);
    return ids[static_cast<size_t>(r.choice_p(p))];
}

int access_side_of(const Obb& box, V2 to_road) {
    const V2 f = bearing_vec(box.facing), rr = bearing_right(box.facing);
    const double sc[4] = {dot(f, to_road), -dot(f, to_road), -dot(rr, to_road), dot(rr, to_road)};   // 前、后、左、右
    int best = 0;
    for (int k = 1; k < 4; ++k)
        if (sc[k] > sc[best]) best = k;
    return best;   // SIDE_FRONT / BACK / LEFT / RIGHT 的次序与 sc 一致
}

Slot slot_along(const Work& w, V2 p, V2 t, V2 n_out, double road_hw, int tmpl, double W, double D, double setback, double force_facing) {
    OrientCtx c = orient_ctx(w, p);
    c.street = bearing_of(n_out * -1.0);
    c.street_dir = bearing_of(t);
    const double facing = std::isfinite(force_facing) ? force_facing : solve_facing(w.st.rules, c, w.st.snap, c.sun);
    const V2 f = bearing_vec(facing), r = bearing_right(facing);
    const double ht = std::fabs(0.5 * W * dot(r, t)) + std::fabs(0.5 * D * dot(f, t));
    const double hn = std::fabs(0.5 * W * dot(r, n_out)) + std::fabs(0.5 * D * dot(f, n_out));
    Slot sl;
    sl.tmpl = tmpl;
    sl.box = {p + n_out * (road_hw + setback + hn), facing, 0.5 * W, 0.5 * D};
    sl.half_t = ht;
    sl.access = p;
    sl.access_side = access_side_of(sl.box, n_out * -1.0);
    return sl;
}

double dist_obb(const Obb& o, V2 p) {
    const V2 d = p - o.c;
    const double x = std::max(0.0, std::fabs(dot(d, bearing_right(o.facing))) - o.hw);
    const double y = std::max(0.0, std::fabs(dot(d, bearing_vec(o.facing))) - o.hd);
    return std::hypot(x, y);
}

bool road_hits_obb(const Work& w, const Obb& o, double tol) {
    const double r = o.hw + o.hd;
    for (const Road& rd : w.plan.roads) {
        const double hw = 0.5 * rd.width_m;
        for (size_t k = 0; k + 1 < rd.line.size(); ++k) {
            const V2 a = rd.line[k], b = rd.line[k + 1];
            if (dist_point_segment(o.c, a, b) > r + hw) continue;
            const double L = len(b - a);
            const int n = std::max(1, static_cast<int>(std::ceil(L / 0.4)));
            for (int q = 0; q <= n; ++q)
                if (dist_obb(o, lerp(a, b, static_cast<double>(q) / n)) < hw - tol) return true;
        }
    }
    return false;
}

// 几何地查一个框会不会碰到已有的房或宅院地块（小房子落在格上可能一格都不占，光看占用格会漏）
bool hits_built(const Work& w, const Obb& o, double clearance) {
    const double r = std::hypot(o.hw, o.hd) + clearance;
    for (const Building& b : w.plan.buildings)
        if (len(b.box.c - o.c) < r + std::hypot(b.box.hw, b.box.hd) && overlap(o, b.box, clearance)) return true;
    for (const Compound& c : w.plan.compounds)
        if (len(c.plot.c - o.c) < r + std::hypot(c.plot.hw, c.plot.hd) && overlap(o, c.plot, clearance)) return true;
    return false;
}

void mark_obb(Work& w, const Obb& o, uint8_t code, double shrink) {
    raster_obb(w.s, o, shrink, [&](int i, int j) {
        if (w.plan.occ(i, j) == OCC_FREE || w.plan.occ(i, j) == OCC_PLOT || code == OCC_BUILDING) w.plan.occ(i, j) = code;
        w.blocked(i, j) = 1;
    });
}

bool obb_free(const Work& w, const Obb& o, double shrink, bool need_buildable, double max_slope_deg) {
    bool ok = true, any = false;
    raster_obb(w.s, o, shrink, [&](int i, int j) {
        any = true;
        if (!ok) return;
        if (w.plan.occ(i, j) != OCC_FREE || w.s.sky(i, j) || w.s.edge(i, j) || w.s.water(i, j) != WATER_NONE) ok = false;
        else if (!w.st.allow_flood && w.s.flood(i, j)) ok = false;
        else if (need_buildable && !w.f.buildable(i, j)) ok = false;
        else if (!(w.f.slope_deg(i, j) <= max_slope_deg)) ok = false;
    });
    return ok && any;
}

bool plot_fits(Work& w, Slot& sl, const Mask* extra) {
    bool clear = true;
    raster_obb(w.s, sl.box, 0.0, [&](int i, int j) {
        if (w.plan.occ(i, j) != OCC_FREE) clear = false;
    });
    if (!clear) return false;
    // 栅格看不出不到一格的相交：与附近的地块再按矩形判一次，与路按「路中线离地块 ≥ 半宽」判
    const double reach = sl.box.hw + sl.box.hd + 60.0;
    for (const Compound& c : w.plan.compounds)
        if (len(c.plot.c - sl.box.c) < reach + c.plot.hw + c.plot.hd && overlap(c.plot, sl.box, -0.05)) return false;
    if (road_hits_obb(w, sl.box, 0.05)) return false;
    return slot_ground(w, sl, extra);
}

double total_slope_share(const Work& w, V2 c, double radius, double min_deg) {
    const Site& s = w.s;
    int ci, cj;
    if (!s.cell_of(c, ci, cj)) return 0.0;
    const int rc = static_cast<int>(std::ceil(radius / s.res_m));
    int n = 0, k = 0;
    for (int i = std::max(0, ci - rc); i <= std::min(s.H - 1, ci + rc); i += 2)
        for (int j = std::max(0, cj - rc); j <= std::min(s.W - 1, cj + rc); j += 2) {
            if (len(s.center(i, j) - c) > radius || s.sky(i, j) || s.water(i, j) != WATER_NONE) continue;
            ++n;
            k += w.f.slope_deg(i, j) >= min_deg;
        }
    return n ? static_cast<double>(k) / n : 0.0;
}

int commit_compound(Work& w, const Slot& sl, const std::string& kind, const std::string& func, const std::string& name,
                    const std::vector<int>& households) {
    Compound c;
    const TemplateSpec& T = w.st.templates[sl.tmpl];
    c.kind = kind;
    c.tmpl = T.id;
    c.tmpl_name = T.name;
    c.func = func;
    c.name = name;
    c.plot = sl.box;
    c.access_side = sl.access_side;
    c.access = sl.access;
    c.walled = T.wall;
    c.base_m = sl.base_m;
    c.max_cut_m = sl.max_cut;
    c.households = households;
    w.plan.compounds.push_back(c);
    const int ci = static_cast<int>(w.plan.compounds.size()) - 1;
    for (int h : households) w.plan.households[h].compound = ci;
    mark_obb(w, sl.box, OCC_PLOT, 0.0);
    return ci;
}

double road_width(const Work& w, int cls, Rng& r) {
    switch (cls) {
        case RC_TRUNK: return w.st.trunk_w.sample(r);
        case RC_MAIN: return w.st.main_w.sample(r);
        case RC_STREET: return w.st.street_w.sample(r);
        case RC_LANE: return w.st.lane_w.sample(r);
        default: return w.st.path_w.sample(r);
    }
}

// 按几何量这条路离地块、院外单栋房最近多远：路宽不超过 2 × 最近距离 − 0.1（栅格上的间距只准到一格，这里补上）
double road_fit(const Work& w, const std::vector<V2>& line) {
    double xmin = INF, xmax = -INF, ymin = INF, ymax = -INF;
    for (const V2& p : line) xmin = std::min(xmin, p.x), xmax = std::max(xmax, p.x), ymin = std::min(ymin, p.y), ymax = std::max(ymax, p.y);
    std::vector<const Obb*> near;
    auto consider = [&](const Obb& o) {
        const double r = o.hw + o.hd + 8.0;
        if (o.c.x + r >= xmin && o.c.x - r <= xmax && o.c.y + r >= ymin && o.c.y - r <= ymax) near.push_back(&o);
    };
    for (const Compound& c : w.plan.compounds) consider(c.plot);
    for (const Building& b : w.plan.buildings)
        if (b.compound < 0) consider(b.box);
    if (near.empty()) return INF;
    double dmin = INF;
    for (size_t k = 0; k + 1 < line.size(); ++k) {
        const V2 a = line[k], b = line[k + 1];
        const int n = std::max(1, static_cast<int>(std::ceil(len(b - a) / 0.4)));
        for (const Obb* o : near) {
            if (dist_point_segment(o->c, a, b) > o->hw + o->hd + 8.0) continue;
            for (int q = 0; q <= n; ++q) dmin = std::min(dmin, dist_obb(*o, lerp(a, b, static_cast<double>(q) / n)));
        }
    }
    return std::isfinite(dmin) ? 2.0 * dmin - 0.1 : INF;
}

int add_road(Work& w, const std::vector<V2>& line, int cls, double width) {
    if (line.size() < 2 || polyline_length(line) < 0.5) return -1;
    width = std::max(0.8, std::min(width, road_fit(w, line)));   // 挤在两堵墙之间的一段路收窄（整条按最窄处，最窄 0.8 m）
    w.plan.roads.push_back({line, cls, width});
    const int id = static_cast<int>(w.plan.roads.size()) - 1;
    raster_line(w.s, line, width, [&](int i, int j) {
        uint8_t& o = w.plan.occ(i, j);
        if (o == OCC_FREE) o = w.s.water(i, j) != WATER_NONE ? OCC_BRIDGE : OCC_ROAD;
        w.road_mask(i, j) = 1;
    });
    raster_line(w.s, line, std::max(w.s.res_m, width - w.s.res_m), [&](int i, int j) { w.road_core(i, j) = 1; });
    for (auto [a, b] : water_runs(w.s, line)) {
        const std::vector<V2> seg = subline(line, std::max(0.0, a - 1.5), std::min(polyline_length(line), b + 1.5));
        if (seg.size() >= 2) w.plan.bridges.push_back({seg.front(), seg.back(), width, id});
    }
    return id;
}

// ---------------------------------------------------------------- 户与亲缘（7.8）
namespace {

void make_households(Work& w) {
    Rng r = w.rng("households");
    const int n = w.req.hh_farm + w.req.hh_market + w.req.hh_special;
    const int founders = std::clamp(w.st.founders.sample_int(r), 1, std::max(1, n));
    for (int i = 0; i < n; ++i) {
        Household h;
        h.id = i;
        h.kind = i < w.req.hh_farm ? HH_FARM : i < w.req.hh_farm + w.req.hh_market ? HH_MARKET : HH_SPECIAL;
        h.parent = i < founders ? -1 : (r.random() < w.st.kin_p ? static_cast<int>(r.integers(0, i)) : -1);
        w.plan.households.push_back(h);
    }
    for (int i = 0; i < n;) {
        const int k = std::clamp(w.st.hh_per_compound.sample_int(r), 1, n - i);
        std::vector<int> g;
        for (int q = 0; q < k; ++q) g.push_back(i + q);
        w.hh_groups.push_back(g);
        i += k;
    }
    w.n_compounds = static_cast<int>(w.hh_groups.size());
}

// 预计半径：宅院数 × 平均地块面积 / 地块占建成区的比例（0.55，估）
void estimate_radius(Work& w) {
    double a = 0.0, ws = 0.0;
    for (int k : w.st.house_templates()) {
        const TemplateSpec& T = w.st.templates[k];
        a += T.weight * T.plot_w.mid() * T.plot_d.mid();
        ws += T.weight;
    }
    w.plot_area = ws > 0 ? a / ws : 400.0;
    w.R = std::max(20.0, std::sqrt(std::max(1, w.n_compounds) * w.plot_area / (PI * 0.55)));
}

// 村心：上游点位 center_search_m 以内，半径 R/2 的邻域里兴趣图均值（不可建算 0）最高处；离点位远一点扣一点分
void choose_center(Work& w) {
    const Site& s = w.s;
    const int H = s.H, W = s.W;
    std::vector<double> S(static_cast<size_t>(H + 1) * (W + 1), 0.0);
    auto at = [&](int i, int j) -> double& { return S[static_cast<size_t>(i) * (W + 1) + j]; };
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const float v = w.interest(i, j);
            at(i + 1, j + 1) = at(i, j + 1) + at(i + 1, j) - at(i, j) + (std::isfinite(v) ? v : 0.0);
        }
    const int rb = std::max(3, static_cast<int>(std::lround(std::max(10.0, 0.5 * w.R) / s.res_m)));
    const int stride = std::max(1, static_cast<int>(std::lround(4.0 / s.res_m)));
    const double rs = w.st.center_search_m;
    double best = -1e18;
    V2 arg = w.req.anchor;
    for (int i = 0; i < H; i += stride)
        for (int j = 0; j < W; j += stride) {
            const V2 p = s.center(i, j);
            const double d = len(p - w.req.anchor);
            if (d > rs || !w.f.buildable(i, j)) continue;
            const int a = std::max(0, i - rb), b = std::min(H, i + rb + 1), c = std::max(0, j - rb), e = std::min(W, j + rb + 1);
            const double area = static_cast<double>(2 * rb + 1) * (2 * rb + 1);   // 出窗口的部分也算进分母：别贴着窗口边
            const double m = (at(b, e) - at(a, e) - at(b, c) + at(a, c)) / area;
            const double sc = m - 0.4 * (d / rs) * (d / rs);   // 小挪便宜、大挪贵
            if (sc > best) best = sc, arg = p;
        }
    w.center = arg;
}

// 整村的朝向：朝向链在村心的语境里解（没有街）；村心一带有坡时另加一条弱的「顺坡」，让街顺等高线
void choose_facing(Work& w) {
    const Site& s = w.s;
    double gx = 0.0, gy = 0.0;
    int n = 0;
    const int rc = std::max(2, static_cast<int>(std::lround(w.R / s.res_m)));
    int ci, cj;
    s.cell_of(w.center, ci, cj);
    for (int i = std::max(1, ci - rc); i < std::min(s.H - 1, ci + rc); i += 2)
        for (int j = std::max(1, cj - rc); j < std::min(s.W - 1, cj + rc); j += 2) {
            const double a = s.height(i, j + 1), b = s.height(i, j - 1), c = s.height(i - 1, j), d = s.height(i + 1, j);
            if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c) && std::isfinite(d))) continue;
            gx += (a - b) / (2 * s.res_m), gy += (c - d) / (2 * s.res_m);
            ++n;
        }
    OrientCtx c;
    c.sun = w.f.sun_bearing;
    c.wind_from = w.f.wind_from;
    std::vector<OrientRule> rules = w.st.rules;
    if (n > 0) {
        gx /= n, gy /= n;
        const double slope = std::atan(std::hypot(gx, gy)) * 180.0 / PI;
        if (slope > 0.5) {
            c.downslope = bearing_of({-gx, -gy});
            rules.push_back({"downslope", 0.5 * std::min(1.0, slope / 4.0), 0.0, 0.0});
        }
    }
    const OrientCtx ctx = orient_ctx(w, w.center);
    c.water = ctx.water;
    w.facing = solve_facing(rules, c, w.st.snap, c.sun);
}

// 按风格的权重挑，只在这块地能用的算子里挑（滨水要河、等高线要坡、地坑院要平塬、小村的算子看户数）；
// 一个都不能用就用风格里权重最大的那个（风格不合这块地：照风格硬做，校验会说地不够），风格一个算子都没写才团块生长
std::string pick_operator(Work& w) {
    if (!w.req.force_operator.empty()) return w.req.force_operator;
    Rng r = w.rng("operator");
    std::vector<double> p;
    std::vector<std::string> ids;
    for (auto& kv : w.st.operators)
        if (kv.second > 0.0 && op_fits(w, kv.first)) ids.push_back(kv.first), p.push_back(kv.second);
    if (ids.empty()) {
        std::string best = "organic";
        double bw = 0.0;
        for (auto& kv : w.st.operators)
            if (kv.second > bw) bw = kv.second, best = kv.first;
        return best;
    }
    return ids[static_cast<size_t>(r.choice_p(p))];
}

}  // namespace

Plan plan_site(const Site& s, const Style& st, const PlanRequest& req) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto lap = [&](Work& w, const char* k) {
        const auto t1 = clk::now();
        w.plan.timing[k] = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    };
    if (req.scale != "compound" && req.scale != "hamlet" && req.scale != "village")
        throw std::invalid_argument("town plan: scale " + req.scale + " is not implemented yet (PLAN-TOWN step 4/6)");
    Work w(s, st, req);
    w.plan.occ = Grid<uint8_t>(s.H, s.W, OCC_FREE);
    w.road_mask = Mask(s.H, s.W, 0);
    w.road_core = Mask(s.H, s.W, 0);
    w.blocked = Mask(s.H, s.W, 0);
    w.f = analyze(s, st, req.wind_from);
    w.interest = interest_dwelling(s, w.f, st);
    {
        Mask wet(s.H, s.W, 0);
        bool any = false;
        for (size_t k = 0; k < wet.size(); ++k) wet.v[k] = s.water.v[k] != WATER_NONE, any = any || wet.v[k];
        if (any) {
            GridF d;
            edt(wet, d, &w.water_src);
        }
    }
    w.pp.max_grade = st.max_grade;
    w.pp.bridge_max_m = st.bridge_max_m;
    lap(w, "analysis");

    make_households(w);
    estimate_radius(w);
    choose_center(w);
    choose_facing(w);
    w.plan.center = w.center;
    w.plan.facing = w.facing;
    w.plan.radius = w.R;
    lap(w, "center");

    if (req.scale == "compound") {
        w.plan.op = "single";
        op_single(w);
    } else {
        for (auto& kv : w.st.operators)
            if (kv.second > 0.0 && op_fits(w, kv.first)) w.plan.ops_fit.push_back(kv.first);
        w.plan.op = pick_operator(w);
        const std::string& op = w.plan.op;
        if (op == "fishbone") op_fishbone(w);
        else if (op == "organic") op_organic(w);
        else if (op == "street_village") op_street(w, false);
        else if (op == "hufen") op_street(w, true);
        else if (op == "waterfront") op_waterfront(w);
        else if (op == "dispersed") op_dispersed(w);
        else if (op == "green") op_green(w);
        else if (op == "comb") op_comb(w);
        else if (op == "contour") op_contour(w);
        else if (op == "enclosure") op_enclosure(w);
        else throw std::invalid_argument("town plan: unknown operator " + op);
        place_missing_public(w);
    }
    lap(w, "layout");
    {
        Rng r = w.rng("compounds");
        for (int ci = 0; ci < static_cast<int>(w.plan.compounds.size()); ++ci) {
            const TemplateSpec& T = w.st.templates[w.st.template_index(w.plan.compounds[ci].tmpl)];
            if (T.shape == "yard") instantiate_compound(w, ci, r);
            else instantiate_enclosure(w, ci, r);
        }
    }
    lap(w, "compounds");
    if (req.scale != "compound") {
        place_exits(w);
        lap(w, "exits");
        place_moat(w);
        if (req.want_landing) place_landing(w);
        place_ponds(w);
        place_roadside(w);
        place_threshing(w);
        place_wells(w);
        place_extras(w);
        lap(w, "functions");
    }
    verify(w);
    lap(w, "verify");
    return std::move(w.plan);
}

}  // namespace skyisle::town
