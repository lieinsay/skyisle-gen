// 形态算子「等高线」（PLAN-TOWN 6.2；黄土高原靠崖窑、山地村）：
//   1. 层差：资料给的层差与「一层院子的进深在坡上的高差」取大的（上一层的台路不能压下一层的院子）；
//   2. 村心一带每一层取等高线（marching squares），坡太缓、太陡、压水 / 虚空处断开，太短的不要：每段是一层的台路（平的）；
//   3. 台路之间用之字路接起来：从离村心最近的一层起，像 Prim 一样一层层按坡度代价接到已接上的台路（纵坡上限让路自己拐之字）；
//   4. 院子排在台路的上坡一侧，背靠坡、朝下坡（窑挖进后面的崖里）；落不下的交给团块生长。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

V2 left_of(V2 t) { return {-t.y, t.x}; }

}  // namespace

void op_contour(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("contour");
    const double radius = 1.3 * w.R + 60.0;
    const double smin = st.pn("contour.min_slope_deg"), smax = st.pn("contour.max_slope_deg");
    // 1. 层差
    std::vector<double> sl;
    double hmin = INF, hmax = -INF;
    {
        int ci, cj;
        s.cell_of(w.center, ci, cj);
        const int rc = static_cast<int>(std::ceil(radius / s.res_m));
        for (int i = std::max(0, ci - rc); i < std::min(s.H, ci + rc); i += 2)
            for (int j = std::max(0, cj - rc); j < std::min(s.W, cj + rc); j += 2) {
                if (len(s.center(i, j) - w.center) > radius || s.sky(i, j) || s.water(i, j) != WATER_NONE) continue;
                const double h = s.height(i, j);
                if (!std::isfinite(h)) continue;
                hmin = std::min(hmin, h), hmax = std::max(hmax, h);
                if (w.f.slope_deg(i, j) >= smin) sl.push_back(w.f.slope_deg(i, j));
            }
    }
    if (sl.size() < 20 || !std::isfinite(hmin)) {
        w.plan.op = "organic";   // 退回：标成实际用的算子
        op_organic(w);
        return;
    }
    std::nth_element(sl.begin(), sl.begin() + sl.size() / 2, sl.end());
    const double med = sl[sl.size() / 2];
    double dmean = 0.0, ws = 0.0;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op)) dmean += st.templates[k].weight * st.templates[k].plot_d.mid(), ws += st.templates[k].weight;
    dmean = ws > 0 ? dmean / ws : 15.0;
    const double pw = road_width(w, RC_STREET, r);
    const double dh = std::max(st.pr("contour.layer_m").sample(r), std::tan(med * PI / 180.0) * (dmean + pw + 3.0));
    // 2. 各层的台路
    const double h0 = field_at(s, s.height, w.center, static_cast<float>(0.5 * (hmin + hmax)));
    struct Terrace {
        double level, dc;
        std::vector<V2> line;
        int road = -1;
    };
    std::vector<Terrace> ts;
    auto ok = [&](int i, int j) {
        const float g = w.f.slope_deg(i, j);
        return !s.sky(i, j) && !s.edge(i, j) && s.water(i, j) == WATER_NONE && w.plan.occ(i, j) == OCC_FREE && g <= smax && g >= 0.5 * smin;
    };
    const int kmin = static_cast<int>(std::floor((hmin - h0) / dh)), kmax = static_cast<int>(std::ceil((hmax - h0) / dh));
    for (int k = kmin; k <= kmax; ++k) {
        const double level = h0 + k * dh + 0.01;
        for (const auto& line : iso_lines(s, s.height, level, 2, w.center, radius))
            for (auto& run : split_runs(w, resample(line, 2.0), ok, st.pn("contour.min_line_m"))) {
                double dc = INF;
                for (const V2& p : run) dc = std::min(dc, len(p - w.center));
                ts.push_back({level, dc, run, -1});
            }
    }
    if (ts.empty()) {
        w.plan.op = "organic";   // 退回：标成实际用的算子
        op_organic(w);
        return;
    }
    std::sort(ts.begin(), ts.end(), [](const Terrace& a, const Terrace& b) { return a.dc < b.dc; });
    for (Terrace& t : ts) {
        chaikin(t.line, nullptr, 1);
        t.road = add_road(w, t.line, &t == &ts[0] ? RC_STREET : RC_LANE, &t == &ts[0] ? pw : st.lane_w.mid());
    }
    // 3. 之字路：一层层接到已接上的台路
    {
        Mask goal(s.H, s.W, 0);
        auto add_goal = [&](int road) {
            if (road < 0) return;
            const Road& rd = w.plan.roads[road];
            raster_line(s, rd.line, std::max(s.res_m, rd.width_m - s.res_m), [&](int i, int j) { goal(i, j) = 1; });
        };
        add_goal(ts[0].road);
        std::vector<bool> linked(ts.size(), false);
        linked[0] = true;
        for (int round = 0; round < 3; ++round)
            for (size_t k = 1; k < ts.size(); ++k) {
                if (linked[k] || ts[k].road < 0) continue;
                const std::vector<V2>& line = w.plan.roads[ts[k].road].line;
                const double L = polyline_length(line);
                // 从离已接上的最近处起：试中点、两头
                for (double a : {0.5 * L, 0.0, L}) {
                    const std::vector<V2> q = subline(line, std::max(0.0, a - 0.5), std::min(L, a + 0.5));
                    if (q.empty()) continue;
                    const V2 p = q.front();
                    const int id = link_road(w, p, RC_PATH, road_width(w, RC_PATH, r), 2.5 * dh / std::max(0.02, w.pp.max_grade) + 80.0, &goal);
                    if (id >= 0) {
                        linked[k] = true;
                        add_goal(ts[k].road);
                        add_goal(id);
                        break;
                    }
                }
            }
    }
    // 4. 院子：台路的上坡一侧，朝台路（下坡）
    {
        std::vector<Slot> pc = public_candidates(w, w.center, 0.8 * w.R + 30.0, RC_LANE, r);
        place_public_compounds(w, pc);
    }
    std::vector<Slot> slots;
    for (const Terrace& t : ts) {
        if (t.road < 0) continue;
        const Road& rd = w.plan.roads[t.road];
        if (rd.line.size() < 2) continue;
        // 上坡在哪一侧：中点往左 5 m 的地面比台路高吗
        const V2 m = rd.line[rd.line.size() / 2];
        const V2 tv = rd.line[std::min(rd.line.size() - 1, rd.line.size() / 2 + 1)] - rd.line[rd.line.size() / 2 - (rd.line.size() > 2 ? 1 : 0)];
        const V2 lft = left_of(tv * (1.0 / std::max(1e-9, len(tv))));
        const double hl = field_at(s, s.height, m + lft * 5.0, static_cast<float>(t.level));
        SliceOpt o;
        o.face_road = true;
        o.side = hl > t.level ? 1 : -1;
        o.setback = 0.5;
        std::vector<Slot> ss = slice_along(w, rd.line, 0.5 * rd.width_m, o, r);
        slots.insert(slots.end(), ss.begin(), ss.end());
    }
    std::stable_sort(slots.begin(), slots.end(), [&](const Slot& a, const Slot& b) {
        return len(a.box.c - w.center) + 15.0 * a.noise < len(b.box.c - w.center) + 15.0 * b.noise;
    });
    w.plan.metrics["terraces"] = static_cast<double>(ts.size());
    w.plan.metrics["terrace_step_m"] = dh;
    size_t g = fill_slots(w, slots, 0);
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
