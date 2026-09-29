// 形态算子「围合单体」（PLAN-TOWN 6.2；客家土楼、围龙屋）：一座楼住一个宗族的许多户，户 = 竖向一列房间。
//   1. 楼数：一座楼一圈 ≤ max_rooms 间，每户 hh_columns 列 → 能住几户；按户数分到各楼（尽量平均）；
//   2. 每座楼按间数定尺寸（圆楼：周长 = 间数 × 每间弧长；方楼：四边；围龙屋：后半圈 + 两横），挑模板（圆 / 方 / 围龙按权重）；
//   3. 楼址：第一座在村心附近兴趣最高处，朝向按朝向链（背山面水、朝阳）；以后各座离已有的至少 gap_factor × 平均直径，
//      挑平的地（整座楼一个台基）；围龙屋门前要留出禾坪与半月池的地；
//   4. 禾坪（门前晒坪）、半月池（围龙屋）；路：第一座门前的禾坪边起一段路，其余各座门口按坡度代价接上；
//   5. 楼放不下的户交给团块生长（楼外的小屋）。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

void op_enclosure(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("enclosure");
    std::vector<int> tmpls;
    std::vector<double> wts;
    for (int k : st.house_templates())
        if (st.templates[k].allows(w.plan.op) && st.templates[k].shape != "yard") tmpls.push_back(k), wts.push_back(st.templates[k].weight);
    if (tmpls.empty()) {
        w.plan.op = "organic";   // 退回：标成实际用的算子
        op_organic(w);
        return;
    }
    const int ng = static_cast<int>(w.hh_groups.size());
    const Range cols_r = st.pr("enclosure.hh_columns");
    const int max_rooms = static_cast<int>(st.pn("enclosure.max_rooms")), min_rooms = static_cast<int>(st.pn("enclosure.min_rooms"));
    const double cols_mean = std::max(1.0, cols_r.mid());
    const int per = std::max(1, static_cast<int>(std::floor(max_rooms / cols_mean)) - 2);   // 门、祖堂占掉的
    const int nb = std::max(1, static_cast<int>(std::ceil(static_cast<double>(ng) / per)));
    const bool pond = st.pn("enclosure.front_pond") != 0.0;
    struct Bld {
        int tmpl;
        double W, D;
        std::vector<int> groups;
        double front;   // 门前要留的进深（禾坪 + 半月池）
    };
    std::vector<Bld> blds;
    // 按间数定尺寸（换模板时重算）
    auto size_bld = [&](Bld& B, int rooms) {
        const TemplateSpec& T = st.templates[B.tmpl];
        const double room = T.ring_room_m.mid(), depth = T.ring_depth_m.mid();
        if (T.shape == "ring") {
            const double R_mid = rooms * room / (2 * PI);
            B.W = B.D = 2.0 * (R_mid + 0.5 * depth) + 0.6;
        } else if (T.shape == "square_ring") {
            const double side = (rooms * room + 4.0 * depth + 1.5 * room) / 4.0;
            B.W = B.D = side + 0.6;
        } else {   // weilong：后半圈 π·R 的弧 + 两横各 ≈ 0.9·W
            const double Rw = rooms * room / (PI + 3.6);
            B.W = 2.0 * (Rw + 0.5 * depth) + 0.6;
            B.D = 1.35 * B.W;
        }
        B.front = st.pr("comb.threshing_depth_m").mid() * 0.6 + (pond && T.shape == "weilong" ? 0.5 * B.W : 0.0);
    };
    std::vector<int> rooms_of;
    int weilong = -1;
    for (int k : tmpls)
        if (st.templates[k].shape == "weilong") weilong = k;
    for (int b = 0, g = 0; b < nb; ++b) {
        const int m = (ng - g) / (nb - b);
        Bld B;
        B.tmpl = tmpls[static_cast<size_t>(r.choice_p(wts))];
        for (int q = 0; q < m; ++q) B.groups.push_back(g++);
        int cols = 0;
        for (int q = 0; q < m; ++q) cols += std::max(1, cols_r.sample_int(r));
        const int rooms = std::clamp(cols + 2, min_rooms, max_rooms);
        size_bld(B, rooms);
        blds.push_back(B);
        rooms_of.push_back(rooms);
    }
    // 楼址
    std::vector<int> placed;   // 宅院号
    double dmean = 0.0;
    for (const Bld& B : blds) dmean += std::max(B.W, B.D) / blds.size();
    const double gapf = st.pn("enclosure.gap_factor");
    const double reach = 2.5 * w.R + 60.0;
    for (size_t bi = 0; bi < blds.size(); ++bi) {
        Bld& B = blds[bi];
        double best = -INF;
        Slot arg;
        // 圆楼、方楼整座一个台基，要平地；坡上找不到就改成依山分台的围龙屋再找
        for (int round = 0; round < 2 && !std::isfinite(best); ++round) {
        if (round == 1) {
            if (weilong < 0 || B.tmpl == weilong) break;
            B.tmpl = weilong;
            size_bld(B, rooms_of[bi]);
        }
        const bool terr = st.templates[B.tmpl].shape == "weilong";
        for (double rad = 0.0; rad <= reach; rad += 8.0)
            for (int k = 0; k < (rad == 0.0 ? 1 : std::max(8, static_cast<int>(2 * PI * rad / 16.0))); ++k) {
                const V2 p = w.center + bearing_vec(2 * PI * k / std::max(8.0, std::floor(2 * PI * rad / 16.0)) + 0.3 * rad) * rad;
                bool far = true;
                for (int ci : placed) far = far && len(w.plan.compounds[ci].plot.c - p) >= gapf * dmean;
                if (!far) continue;
                const OrientCtx c = orient_ctx(w, p);
                Slot sl;
                sl.tmpl = B.tmpl;
                sl.box = {p, solve_facing(st.rules, c, st.snap, c.sun), 0.5 * B.W, 0.5 * B.D};
                sl.access_side = SIDE_FRONT;
                sl.access = p + bearing_vec(sl.box.facing) * (0.5 * B.D + B.front + 2.0);
                if (!plot_fits(w, sl, nullptr)) continue;
                // 整座楼一个台基：挖填不超过一栋房的上限（围龙屋分台，不查）
                if (!terr && sl.max_cut > st.max_cut_m + 0.5) continue;
                const Obb front{p + bearing_vec(sl.box.facing) * (0.5 * B.D + 0.5 * B.front + 0.5), sl.box.facing, 0.5 * B.W, 0.5 * B.front};
                if (B.front > 1.0 && !obb_free(w, front, 0.0, false, 10.0)) continue;
                const double sc = sl.interest - len(p - w.center) / (w.R + 50.0) - 0.3 * sl.max_cut / std::max(0.1, st.max_cut_m);
                if (sc > best) best = sc, arg = sl;
            }
        }
        if (!std::isfinite(best)) continue;
        const TemplateSpec& T = st.templates[B.tmpl];
        std::vector<int> hh;
        for (int g : B.groups) hh.insert(hh.end(), w.hh_groups[g].begin(), w.hh_groups[g].end());
        const int ci = commit_compound(w, arg, "house", "dwelling", "", hh);
        placed.push_back(ci);
        // 禾坪、半月池
        const V2 fv = bearing_vec(arg.box.facing), rv = bearing_right(arg.box.facing);
        const double td = st.pr("comb.threshing_depth_m").mid() * 0.6;
        const Obb thr{arg.box.c + fv * (arg.box.hd + 0.5 * td + 0.3), arg.box.facing, arg.box.hw, 0.5 * td};
        if (obb_free(w, thr, 0.0, false, 10.0)) {
            Feature ft;
            ft.kind = "threshing";
            ft.func = "threshing";
            ft.name = st.ps("enclosure.threshing_name", "");
            ft.p = thr.c;
            ft.facing = thr.facing;
            ft.r = thr.hw;
            ft.compound = ci;
            const auto cs = corners(thr);
            ft.poly.assign(cs.begin(), cs.end());
            ft.z = arg.base_m;
            w.plan.features.push_back(ft);
            raster_obb(s, thr, 0.0, [&](int i, int j) {
                if (w.plan.occ(i, j) == OCC_FREE) w.plan.occ(i, j) = OCC_THRESH;
            });
        }
        if (pond && T.shape == "weilong") {
            const V2 pc = arg.box.c + fv * (arg.box.hd + td + 0.6);   // 半圆的直边贴着禾坪
            const double pr = arg.box.hw * 0.9;
            std::vector<V2> poly;
            for (int q = 0; q <= 20; ++q) {
                const double th = -0.5 * PI + PI * q / 20;
                poly.push_back(pc + rv * (pr * std::sin(th)) + fv * (0.55 * pr * std::cos(th)));
            }
            bool ok = true;
            double zmin = INF;
            raster_polygon(s, poly, [&](int i, int j) {
                if (w.plan.occ(i, j) != OCC_FREE || s.sky(i, j) || s.edge(i, j) || s.water(i, j) != WATER_NONE) ok = false;
                if (std::isfinite(s.height(i, j))) zmin = std::min(zmin, static_cast<double>(s.height(i, j)));
            });
            if (ok && std::isfinite(zmin)) {
                Feature ft;
                ft.kind = "pond";
                ft.func = "pond";
                ft.name = st.ps("enclosure.pond_name", "");
                ft.poly = poly;
                ft.p = pc;
                ft.r = pr;
                ft.facing = arg.box.facing;
                ft.z = zmin - 0.5;
                ft.compound = ci;
                raster_polygon(s, poly, [&](int i, int j) { w.plan.occ(i, j) = OCC_POND, w.blocked(i, j) = 1; });
                w.plan.features.push_back(ft);
            }
        }
    }
    if (placed.empty()) {
        w.plan.op = "organic";   // 一座楼都落不下：退回团块生长（楼外小屋）
        op_organic(w);
        return;
    }
    // 路：第一座门前禾坪的一侧起一段，其余各座门口接上
    const double sw = road_width(w, RC_STREET, r);
    for (size_t k = 0; k < placed.size(); ++k) {
        const Compound& C = w.plan.compounds[placed[k]];
        const V2 fv = bearing_vec(C.plot.facing), rv = bearing_right(C.plot.facing);
        const double td = st.pr("comb.threshing_depth_m").mid() * 0.6;
        const V2 door = C.plot.c + fv * (C.plot.hd + 0.5 * sw + 0.8);
        if (k == 0) {
            // 门 → 禾坪 → 禾坪的一侧往外
            double side = 1.0;
            {
                const V2 a = C.plot.c + fv * (C.plot.hd + 0.5 * td) + rv * (C.plot.hw + 20.0), b = C.plot.c + fv * (C.plot.hd + 0.5 * td) - rv * (C.plot.hw + 20.0);
                side = field_at(s, w.interest, a, -1.0f) >= field_at(s, w.interest, b, -1.0f) ? 1.0 : -1.0;
            }
            const V2 mid = C.plot.c + fv * (C.plot.hd + 0.5 * td + 0.3);
            const V2 out = mid + rv * (side * (C.plot.hw + 25.0));
            std::vector<V2> line{door, mid, out};
            auto runs = split_runs(w, resample(line, 1.5), [&](int i, int j) {
                return !s.sky(i, j) && s.water(i, j) == WATER_NONE && w.plan.occ(i, j) != OCC_POND && w.plan.occ(i, j) != OCC_BUILDING && w.plan.occ(i, j) != OCC_PLOT;
            }, 5.0);
            if (!runs.empty()) add_road(w, runs.front(), RC_STREET, sw);
        } else {
            link_road(w, door, RC_STREET, sw, reach + 60.0);
        }
    }
    // 门口没接上的（门前的路被截断了）：门修巷接到路网
    for (int ci : placed) lane_to_gate(w, ci, sw, reach);
    // 围合单体的指标：每座楼住几户
    if (!placed.empty()) {
        double hh = 0.0;
        for (int ci : placed) hh += w.plan.compounds[ci].households.size();
        w.plan.metrics["hh_per_enclosure"] = hh / placed.size();
        w.plan.metrics["enclosures"] = static_cast<double>(placed.size());
    }
    // 楼放不下的户
    size_t g = 0;
    std::vector<bool> housed(w.hh_groups.size(), false);
    for (size_t q = 0; q < w.hh_groups.size(); ++q) housed[q] = w.plan.households[w.hh_groups[q][0]].compound >= 0;
    // organic_fill 从某个下标起按次序补：把没住下的挪到后面
    std::vector<std::vector<int>> order;
    for (size_t q = 0; q < w.hh_groups.size(); ++q)
        if (housed[q]) order.push_back(w.hh_groups[q]);
    g = order.size();
    for (size_t q = 0; q < w.hh_groups.size(); ++q)
        if (!housed[q]) order.push_back(w.hh_groups[q]);
    w.hh_groups = order;
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
