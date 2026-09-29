// 形态算子「梳式」（PLAN-TOWN 6.2；岭南广府村落，如大旗头村）：
//   整村朝前（朝阳 / 朝水），前沿一条横街；横街外是晒坪、再外是风水塘；
//   垂直于前沿开一条条平行的巷（巷距 12–14 m），两巷之间一列房，一进一栋前后排，房都朝前、门开在巷里（侧门）；
//   祠堂在前排、朝塘，占两列；后沿一条横巷把各巷连起来。
//   地面不平处的格子空着；落不下的交给团块生长。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

void op_comb(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("comb");
    const int n = static_cast<int>(w.hh_groups.size());
    // 房的尺寸：按住宅模板（梳式一律一个模板为主）
    int tmpl = -1;
    double wmax = -1;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op) && st.templates[k].weight > wmax) wmax = st.templates[k].weight, tmpl = k;
    if (tmpl < 0) {
        w.plan.op = "organic";   // 退回：标成实际用的算子
        op_organic(w);
        return;
    }
    const TemplateSpec& T = st.templates[tmpl];
    const double lane_w = st.lane_w.mid();
    const double Ls = st.pr("comb.lane_spacing_m").sample(r);
    const double pw = std::max(6.0, Ls - lane_w - 0.4);           // 一列房的面宽
    const double Dh = T.plot_d.mid(), gap = st.pr("comb.row_gap_m").sample(r);
    const Range rows_r = st.pr("comb.rows");
    const int halls = std::min(static_cast<int>(st.pn("comb.halls_max")), static_cast<int>(std::ceil(n / std::max(1.0, st.pn("comb.hall_every_hh")))));
    // 行数与列数：大致方整，行数夹在 rows 里（祠堂占前排两列 × 两行）
    const int need = n + 4 * halls;
    int rows = std::clamp(static_cast<int>(std::lround(std::sqrt(need * Ls / (Dh + gap)))), static_cast<int>(rows_r.lo), static_cast<int>(rows_r.hi));
    int cols = static_cast<int>(std::ceil(static_cast<double>(need) / rows));
    const double td = st.pr("comb.threshing_depth_m").sample(r), pd = st.pr("comb.pond_depth_m").sample(r);
    const double fw = road_width(w, RC_MAIN, r);
    // 在村心附近挑原点与朝向：能落下的格子最多
    struct Frame {
        V2 o;            // 前沿横街中线的中点
        double fac;
        int valid = -1;
    };
    Frame best;
    const int extra_cols = 2;   // 多备两列，坏格子多时顶上
    auto cell_slot = [&](const Frame& F, int col, int row, int span_c, int span_r) {
        const V2 fv = bearing_vec(F.fac), rv = bearing_right(F.fac);
        const double x = (col - 0.5 * (cols + extra_cols)) * Ls + 0.5 * span_c * Ls;
        const double y = -(0.5 * fw + 0.5 + row * (Dh + gap) + 0.5 * (span_r * (Dh + gap) - gap));
        Slot sl;
        sl.tmpl = tmpl;
        sl.box = {F.o + rv * x + fv * y, F.fac, 0.5 * (span_c * Ls - lane_w - 0.4), 0.5 * (span_r * (Dh + gap) - gap)};
        return sl;
    };
    {
        const double base_fac = std::isfinite(w.facing) ? w.facing : w.f.sun_bearing;
        for (double da : {0.0, 0.14, -0.14})
            for (double rad = 0.0; rad <= 40.0; rad += 10.0)
                for (int k = 0; k < (rad == 0.0 ? 1 : 8); ++k) {
                    Frame F;
                    F.fac = wrap_pi(base_fac + da);
                    // 村心在整片的中间：前沿在村心前方半个进深处
                    F.o = w.center + bearing_vec(2 * PI * k / 8) * rad + bearing_vec(F.fac) * (0.5 * rows * (Dh + gap));
                    int ok = 0;
                    for (int c = 0; c < cols + extra_cols; ++c)
                        for (int q = 0; q < rows; ++q) {
                            Slot sl = cell_slot(F, c, q, 1, 1);
                            ok += slot_ground(w, sl, nullptr) ? 1 : 0;
                        }
                    // 前面的晒坪与塘也要平、不压水
                    const V2 fv = bearing_vec(F.fac);
                    const Obb front{F.o + fv * (0.5 * fw + 0.5 * (td + pd)), F.fac, 0.5 * (cols + extra_cols) * Ls, 0.5 * (td + pd)};
                    if (!obb_free(w, front, 0.0, false, 6.0)) ok -= cols;
                    F.valid = ok - static_cast<int>(rad / 10.0);
                    if (F.valid > best.valid) best = F;
                }
    }
    if (best.valid < std::max(4, n / 3)) {
        w.plan.op = "organic";   // 退回：标成实际用的算子
        op_organic(w);
        return;
    }
    const Frame F = best;
    const V2 fv = bearing_vec(F.fac), rv = bearing_right(F.fac);
    const int C = cols + extra_cols;
    const double x0 = -0.5 * C * Ls, x1 = 0.5 * C * Ls;
    const double depth_all = 0.5 * fw + 0.5 + rows * (Dh + gap);
    // 晒坪、风水塘（前沿横街之外）
    {
        const Obb thr{F.o + fv * (0.5 * fw + 0.5 * td), F.fac, 0.5 * C * Ls, 0.5 * td};
        if (obb_free(w, thr, 0.0, false, 6.0)) {
            Feature ft;
            ft.kind = "threshing";
            ft.func = "threshing";
            ft.name = st.ps("comb.threshing_name", "");
            ft.p = thr.c;
            ft.facing = F.fac;
            ft.r = thr.hw;
            const auto cs = corners(thr);
            ft.poly.assign(cs.begin(), cs.end());
            ft.z = field_at(s, s.height, thr.c, 0.0f);
            w.plan.features.push_back(ft);
            mark_obb(w, thr, OCC_THRESH, 0.0);
            // 晒坪让路走
            raster_obb(s, thr, 0.0, [&](int i, int j) { w.blocked(i, j) = 0; });
        }
        // 塘：前沿一长条，两头圆
        const V2 pc = F.o + fv * (0.5 * fw + td + 0.5 * pd + 2.0);
        std::vector<V2> poly;
        const double a = 0.45 * C * Ls, b = 0.45 * pd;
        for (int k = 0; k < 28; ++k) {
            const double th = 2 * PI * k / 28;
            const double cx = std::cos(th), sy = std::sin(th);
            const double e = std::pow(std::fabs(cx), 0.35) * (cx < 0 ? -1.0 : 1.0);
            poly.push_back(pc + rv * (a * e) + fv * (b * sy));
        }
        bool ok = true;
        double zmin = INF;
        raster_polygon(s, poly, [&](int i, int j) {
            if (w.plan.occ(i, j) != OCC_FREE || s.sky(i, j) || s.edge(i, j)) ok = false;
            if (std::isfinite(s.height(i, j))) zmin = std::min(zmin, static_cast<double>(s.height(i, j)));
        });
        if (ok && std::isfinite(zmin)) {
            Feature ft;
            ft.kind = "pond";
            ft.func = "pond";
            ft.name = st.ps("comb.pond_name", "");
            ft.poly = poly;
            ft.p = pc;
            ft.r = b;
            ft.facing = F.fac;
            ft.z = zmin - 0.5;
            raster_polygon(s, poly, [&](int i, int j) {
                if (s.water(i, j) == WATER_NONE) w.plan.occ(i, j) = OCC_POND;
                w.blocked(i, j) = 1;
            });
            w.plan.features.push_back(ft);
        }
    }
    // 祠堂的列：前排中间起，左右交替，每座占两列两行
    std::vector<int> hall_cols;
    for (int k = 0; k < halls; ++k) {
        const int c = C / 2 - 1 + (k % 2 ? 1 : -1) * 3 * ((k + 1) / 2);
        if (c >= 0 && c + 1 < C) hall_cols.push_back(c);
    }
    auto in_hall = [&](int c, int q) {
        for (int h : hall_cols)
            if ((c == h || c == h + 1) && q < 2) return true;
        return false;
    };
    // 路：前沿横街、后沿横巷、各巷（祠堂两列之间的那条巷从祠堂后面起）
    std::vector<V2> front{F.o + rv * (x0 - 4.0), F.o + rv * (x1 + 4.0)};
    std::vector<V2> back{F.o + rv * (x0 - 2.0) - fv * (depth_all + 0.5 * lane_w + 0.3), F.o + rv * (x1 + 2.0) - fv * (depth_all + 0.5 * lane_w + 0.3)};
    auto clipped = [&](const std::vector<V2>& line, double min_len) {
        return split_runs(w, resample(line, 2.0), [&](int i, int j) {
            return !s.sky(i, j) && s.water(i, j) == WATER_NONE && (w.plan.occ(i, j) == OCC_FREE || w.plan.occ(i, j) == OCC_ROAD);
        }, min_len);
    };
    for (auto& run : clipped(front, 20.0)) add_road(w, run, RC_MAIN, fw);
    for (auto& run : clipped(back, 20.0)) add_road(w, run, RC_LANE, lane_w);
    std::vector<double> lane_x;
    for (int k = 0; k <= C; ++k) {
        const double x = x0 + k * Ls;
        bool mid_hall = false;
        for (int h : hall_cols) mid_hall = mid_hall || k == h + 1;
        const double y0 = mid_hall ? 0.5 * fw + 0.5 + 2 * (Dh + gap) + 0.5 * lane_w + 0.3 : 0.0;   // 祠堂后面起，别压着祠堂的后墙
        std::vector<V2> lane{F.o + rv * x - fv * y0, F.o + rv * x - fv * (depth_all + 0.5 * lane_w + 0.3)};
        for (auto& run : clipped(lane, 8.0))
            if (add_road(w, run, RC_LANE, lane_w) >= 0) lane_x.push_back(x);
    }
    // 巷距的变异系数（梳式：巷距几乎一样）
    if (lane_x.size() >= 3) {
        std::sort(lane_x.begin(), lane_x.end());
        std::vector<double> d;
        for (size_t k = 1; k < lane_x.size(); ++k) d.push_back(lane_x[k] - lane_x[k - 1]);
        double m = 0.0, v = 0.0;
        for (double x : d) m += x;
        m /= d.size();
        for (double x : d) v += (x - m) * (x - m);
        w.plan.metrics["lane_spacing_cv"] = m > 0 ? std::sqrt(v / d.size()) / m : NaN;
        w.plan.metrics["lane_spacing_m"] = m;
    }
    // 祠堂（公共宅院）
    for (int h : hall_cols) {
        Slot sl = cell_slot(F, h, 0, 2, 2);
        sl.access_side = SIDE_FRONT;
        sl.access = F.o + rv * dot(sl.box.c - F.o, rv);
        if (plot_fits(w, sl, nullptr)) commit_public(w, "ancestral", sl);
    }
    // 房：从前排中间往后、往两边；门开在离村心近的那条巷一侧
    struct Cell {
        int c, q;
        double key;
    };
    std::vector<Cell> cells;
    for (int c = 0; c < C; ++c)
        for (int q = 0; q < rows; ++q)
            if (!in_hall(c, q)) cells.push_back({c, q, q * 1.0 + std::fabs(c + 0.5 - 0.5 * C) * 0.6 + 0.3 * r.random()});
    std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b) { return a.key < b.key; });
    std::vector<Slot> slots;
    for (const Cell& cl : cells) {
        Slot sl = cell_slot(F, cl.c, cl.q, 1, 1);
        const bool left = (cl.c + 0.5) < 0.5 * C;   // 左半边门朝右（朝中轴），右半边朝左
        sl.access_side = left ? SIDE_RIGHT : SIDE_LEFT;
        const double lx = x0 + (cl.c + (left ? 1 : 0)) * Ls;
        sl.access = F.o + rv * lx + fv * dot(sl.box.c - F.o, fv);
        slots.push_back(sl);
    }
    size_t g = fill_slots(w, slots, 0);
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
