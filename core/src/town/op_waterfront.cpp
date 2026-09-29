// 形态算子「滨水」（PLAN-TOWN 6.2；江南水乡）：以河为骨。
//   1. 挑离村心最近、够宽的一条河，按户数定沿河的一段（村心投影到河上的那一点往两头）；
//   2. 两岸各挑一种排法：前街后河（街在陆上，房在街与河之间，后门临水）/ 前河后街（街沿水，房在街的陆侧）；
//   3. 街按河的中线往外偏（偏距 = 半河宽 + 河埠平台 + 一排房的进深 + 半街宽），格上走不通处断开；
//   4. 桥按桥距横跨两岸的街；前街后河的一岸每几户留一条水弄通到水边，尽头是河埠头（石阶）；
//   5. 公共宅院先占，临水那排先落、街的另一侧后落，按离村心（沿河）由近到远；还不够交给团块生长。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

V2 left_of(V2 t) { return {-t.y, t.x}; }

double arc_at(const std::vector<V2>& line, V2 p) {
    int seg;
    double t;
    dist_point_polyline(p, line, &seg, &t);
    double a = 0.0;
    for (int k = 0; k < seg; ++k) a += len(line[k + 1] - line[k]);
    return a + t * len(line[seg + 1] - line[seg]);
}

}  // namespace

void op_waterfront(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("waterfront");
    // 1. 河
    const River* rv = nullptr;
    double bd = INF;
    for (const River& x : s.rivers) {
        double wmax = 0.0;
        for (double v : x.width_m) wmax = std::max(wmax, v);
        if (wmax < st.pn("waterfront.min_width_m") || x.line.size() < 2) continue;
        const double d = dist_point_polyline(w.center, x.line);
        if (d < bd) bd = d, rv = &x;
    }
    if (!rv || bd > w.R + st.pn("waterfront.reach_m")) {
        w.plan.op = "street_village";   // 退回：标成实际用的算子
        op_street(w, false);
        return;
    }
    auto width_at = [&](V2 p) {
        int seg;
        double t;
        dist_point_polyline(p, rv->line, &seg, &t);
        const size_t a = static_cast<size_t>(seg), b = std::min(rv->width_m.size() - 1, a + 1);
        return rv->width_m.empty() ? 6.0 : rv->width_m[a] + (rv->width_m[b] - rv->width_m[a]) * t;
    };
    // 沿河的一段：按要排的面宽定长（两岸、每岸临水一排加街对面一排）
    double tw = 0.0, ws = 0.0;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op)) tw += st.templates[k].weight * st.templates[k].plot_w.mid(), ws += st.templates[k].weight;
    const double f_mean = ws > 0 ? tw / ws : 8.0;
    const int n = static_cast<int>(w.hh_groups.size());
    const double L_all = polyline_length(rv->line);
    const double sc0 = arc_at(rv->line, w.center);
    const double half = std::clamp(0.5 * n * (f_mean + st.plot_gap.mid()) / 2.0 * st.pn("waterfront.length_factor") + 30.0, 80.0, 0.5 * std::min(s.W, s.H) * s.res_m);
    const double a0 = std::max(0.0, sc0 - half), a1 = std::min(L_all, sc0 + half);
    const std::vector<V2> seg = resample(subline(rv->line, a0, a1), 3.0);
    if (seg.size() < 8) {
        w.plan.op = "street_village";   // 退回：标成实际用的算子
        op_street(w, false);
        return;
    }
    const double sc = sc0 - a0;
    // 平滑的法向（前后各 12 m 的弦）
    const int m = 4, N = static_cast<int>(seg.size());
    std::vector<V2> nrm(N);
    for (int k = 0; k < N; ++k) {
        V2 t = seg[std::min(N - 1, k + m)] - seg[std::max(0, k - m)];
        t = t * (1.0 / std::max(1e-9, len(t)));
        nrm[k] = left_of(t);
    }
    // 水边：沿法向从中线往两边走，到第一个不是水的格（河的记录宽与栅格上的水面不一定一样宽）
    std::vector<double> ew[2];
    for (int q = 0; q < 2; ++q) {
        ew[q].resize(N);
        const double sd = q == 0 ? 1.0 : -1.0;
        for (int k = 0; k < N; ++k) {
            const double h0 = 0.5 * width_at(seg[k]);
            double edge = h0;
            bool seen = false;
            for (double t = 0.0; t <= 3.0 * h0 + 40.0; t += 0.5) {
                int i, j;
                if (!s.cell_of(seg[k] + nrm[k] * (sd * t), i, j)) break;
                if (s.water(i, j) != WATER_NONE) seen = true, edge = t + 0.5 * s.res_m;
                else if (seen) break;
            }
            ew[q][k] = edge;
        }
    }
    auto hw = [&](int sd, int k) { return ew[sd > 0 ? 0 : 1][k]; };
    // 2–3. 两岸的街
    const double bank = st.pn("waterfront.bank_m");
    const Range setback = st.setback;
    std::vector<double> modes_p;
    std::vector<std::string> modes;
    for (const std::string& k : {std::string("qianjie_houhe"), std::string("qianhe_houjie")}) {
        const double p = st.pn("waterfront.modes." + k);
        if (p > 0) modes.push_back(k), modes_p.push_back(p);
    }
    if (modes.empty()) modes = {"qianjie_houhe"}, modes_p = {1.0};
    struct Bank {
        int sd;
        bool houhe;             // 前街后河
        double D;               // 临水一排的进深
        double sw;              // 街宽
        std::vector<double> off;
        std::vector<std::vector<V2>> runs;   // 街（沿河走向）
        std::vector<int> roads;
    };
    std::vector<Bank> banks;
    auto street_ok = [&](int i, int j) {
        return !s.sky(i, j) && !s.edge(i, j) && s.water(i, j) == WATER_NONE && w.f.slope_deg(i, j) <= 15.0f && w.plan.occ(i, j) == OCC_FREE;
    };
    int main_done = 0;
    // 这一段河上有能架桥的地方吗（河宽 ≤ 最长的桥）：没有就只排村心那一岸
    bool bridgeable = false;
    for (int k = 0; k < N; ++k) bridgeable = bridgeable || hw(1, k) + hw(-1, k) <= st.bridge_max_m;
    const int k_c = std::clamp(static_cast<int>(std::lround(sc / 3.0)), 0, N - 1);
    const int center_side = dot(w.center - seg[k_c], nrm[k_c]) >= 0 ? 1 : -1;
    for (int sd : {1, -1}) {
        if (!bridgeable && sd != center_side) {
            banks.push_back(Bank{sd, true, 0.0, 0.0, std::vector<double>(N, 0.0), {}, {}});
            continue;
        }
        Bank b;
        b.sd = sd;
        b.houhe = modes[static_cast<size_t>(r.choice_p(modes_p))] == "qianjie_houhe";
        b.D = st.pr("waterfront.row_depth_m").sample(r);
        b.sw = road_width(w, main_done ? RC_STREET : RC_MAIN, r);
        const double sb = setback.sample(r);
        std::vector<V2> pts(N);
        b.off.resize(N);
        for (int k = 0; k < N; ++k) {
            b.off[k] = b.houhe ? hw(sd, k) + bank + b.D + sb + 0.5 * b.sw : hw(sd, k) + 0.5 * b.sw + 0.6;
            pts[k] = seg[k] + nrm[k] * (sd * b.off[k]);
        }
        b.runs = split_runs(w, pts, street_ok, 40.0);
        for (auto& run : b.runs) {
            chaikin(run, nullptr, 1);
            const int id = add_road(w, run, main_done ? RC_STREET : RC_MAIN, b.sw);
            b.roads.push_back(id);
            if (id >= 0) ++main_done;
        }
        banks.push_back(b);
    }
    if (!main_done) {
        w.plan.op = "street_village";   // 退回：标成实际用的算子
        op_street(w, false);
        return;
    }
    // 4a. 桥：两岸都有街的地方，按桥距；一座都没架成就从村心往两头找第一处能架的
    {
        auto try_bridge = [&](int k) {
            if (hw(1, k) + hw(-1, k) > st.bridge_max_m) return false;
            // 两岸各取街上离这一点最近处（街平滑过，不一定正好在偏距上）
            V2 ends[2];
            for (int q = 0; q < 2; ++q) {
                double dmin = INF;
                for (int id : banks[q].roads) {
                    if (id < 0) continue;
                    int sg;
                    double t;
                    const std::vector<V2>& ln = w.plan.roads[id].line;
                    const double d = dist_point_polyline(seg[k], ln, &sg, &t);
                    if (d < dmin) dmin = d, ends[q] = lerp(ln[sg], ln[sg + 1], t);
                }
                if (!(dmin < 1.5 * banks[q].off[k] + 4.0)) return false;
            }
            return add_road(w, {ends[0], seg[k], ends[1]}, RC_STREET, road_width(w, RC_STREET, r)) >= 0;
        };
        if (!banks[0].roads.empty() && !banks[1].roads.empty()) {
            const Range bsp = st.pr("waterfront.bridge_spacing_m");
            std::vector<double> at{sc};
            for (double d = bsp.sample(r); d < half; d += bsp.sample(r)) at.push_back(sc + d), at.push_back(sc - d);
            int made = 0;
            for (double a : at) made += try_bridge(std::clamp(static_cast<int>(std::lround(a / 3.0)), 0, N - 1)) ? 1 : 0;
            for (int d = 1; made == 0 && d < N; ++d)
                for (int k : {k_c + d, k_c - d})
                    if (made == 0 && k >= 0 && k < N && try_bridge(k)) ++made;
        }
    }
    // 4b. 水弄与河埠头（前街后河的一岸：每几户一条，通到水边）；前河后街的一岸河埠头直接在街边
    {
        const Range every = st.pr("waterfront.steps_every");
        for (Bank& b : banks) {
            const double sp = std::max(8.0, every.sample(r) * f_mean);
            for (double a = std::fmod(sc, sp) + 0.5 * sp; a < (N - 1) * 3.0; a += every.sample(r) * f_mean) {
                const int k = std::clamp(static_cast<int>(std::lround(a / 3.0)), 0, N - 1);
                const V2 on_street = seg[k] + nrm[k] * (b.sd * b.off[k]);
                int i, j;
                if (!s.cell_of(on_street, i, j) || !w.road_core(i, j)) continue;
                const V2 edge = seg[k] + nrm[k] * (b.sd * (hw(b.sd, k) + 0.4));
                if (b.houhe) {
                    const double lw = st.pr("waterfront.water_lane_m").sample(r);
                    if (add_road(w, {on_street, edge}, RC_PATH, lw) < 0) continue;
                }
                Feature ft;
                ft.kind = "steps";
                ft.func = "steps";
                ft.name = st.ps("waterfront.steps_name", "");
                ft.p = seg[k] + nrm[k] * (b.sd * (hw(b.sd, k) + 0.2));
                ft.facing = bearing_of(nrm[k] * -static_cast<double>(b.sd));   // 朝水
                ft.r = 2.5;
                const V2 u = nrm[k] * static_cast<double>(b.sd), v = left_of(u);
                ft.poly = {ft.p + v * 1.25, ft.p + v * 1.25 + u * 1.2, ft.p - v * 1.25 + u * 1.2, ft.p - v * 1.25};
                ft.z = std::isfinite(s.height(i, j)) ? s.height(i, j) : 0.0;
                w.plan.features.push_back(ft);
            }
        }
    }
    // 5. 公共宅院 → 临水一排 → 街对面一排
    {
        std::vector<Slot> pc = public_candidates(w, w.center, 0.8 * w.R + 30.0, RC_STREET, r);
        place_public_compounds(w, pc);
    }
    std::vector<Slot> first, second;
    for (Bank& b : banks)
        for (size_t q = 0; q < b.runs.size(); ++q) {
            if (b.roads[q] < 0) continue;
            const std::vector<V2>& run = w.plan.roads[b.roads[q]].line;
            const double rhw = 0.5 * w.plan.roads[b.roads[q]].width_m;
            // 街的走向与河同向：左岸（sd = 1）河在街的右手
            SliceOpt o;
            o.face_road = true;
            o.side = -b.sd;
            if (b.houhe) {
                o.depth = {0.85 * b.D, b.D};
                o.setback = 0.0;
                auto ss = slice_along(w, run, rhw, o, r);
                first.insert(first.end(), ss.begin(), ss.end());
            }
            o = SliceOpt{};
            o.face_road = true;
            o.side = b.sd;
            auto ss = slice_along(w, run, rhw, o, r);
            (b.houhe ? second : first).insert((b.houhe ? second : first).end(), ss.begin(), ss.end());
        }
    auto key = [&](const Slot& sl) { return std::fabs(arc_at(seg, sl.box.c) - sc) + 0.3 * f_mean * sl.noise; };
    std::stable_sort(first.begin(), first.end(), [&](const Slot& a, const Slot& b) { return key(a) < key(b); });
    std::stable_sort(second.begin(), second.end(), [&](const Slot& a, const Slot& b) { return key(a) < key(b); });
    // 两排的地块在街的两侧，互不压（切的时候前一排还没落，这里按几何再查）
    size_t g = fill_slots(w, first, 0);
    g = fill_slots(w, second, g);
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
