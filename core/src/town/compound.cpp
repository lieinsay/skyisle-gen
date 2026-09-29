// 宅院成形（PLAN-TOWN 6.5、7.6）：模板在地块的局部坐标里摆各栋——先贴后沿 / 前沿的，再在两排之间摆两厢；
// 门按路在哪一边与风格的门规则定在某个角，门洞里的房截短（截下的一间做门楼，或墙上开小门楼）；台基 = 地块中位高程 + 各栋台高，记挖填方。
// 局部坐标：x 向右（站在院里面朝前沿时的右手），y 向前；地块 [−W/2, W/2] × [−D/2, D/2]。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

namespace {

struct Local {
    double x0, x1, y0, y1;   // 占的局部矩形
    double facing;           // 相对地块朝向的转角（0 = 朝前）
    int spec;
    bool gatehouse = false;
};

double along_len(const Local& b) {
    // 面阔方向：朝前 / 朝后的房沿 x，朝左 / 朝右的沿 y
    const double c = std::fabs(std::cos(b.facing));
    return c > 0.5 ? b.x1 - b.x0 : b.y1 - b.y0;
}

}  // namespace

void instantiate_compound(Work& w, int ci, Rng& r) {
    Compound& C = w.plan.compounds[ci];
    const TemplateSpec& T = w.st.templates[w.st.template_index(C.tmpl)];
    const double W = 2.0 * C.plot.hw, D = 2.0 * C.plot.hd;
    const double t = C.walled ? w.st.wall_thickness_m : 0.0;
    const double bay = T.bay_m.sample(r);
    const double pit_depth = 6.5;   // 地坑院的坑深 6–7 m（TOWN-SOURCES §6）
    // 地块后部做园（croft）时，房只摆在前面那一截：「后沿」是园的前沿
    const double y_back = -0.5 * D + t + T.garden_frac * D, Deff = (1.0 - T.garden_frac) * D;
    std::vector<Local> bs;
    double back_d = 0.0, front_d = 0.0;
    int last_back = -1, last_front = -1;   // 同一条边上前一栋（耳房贴它）
    auto length_of = [&](const BuildingSpec& b, double avail) {
        double L;
        if (b.bays.hi > 0) {
            int n = b.bays.sample_int(r, b.odd);
            while (n > 1 && n * bay > avail) n -= b.odd ? 2 : 1;
            n = std::max(1, n);
            L = std::min(avail, n * bay);
        } else L = avail * std::clamp(b.len_frac.sample(r), 0.05, 1.0);
        return L;
    };
    // 1. 后沿与前沿
    for (int k = 0; k < static_cast<int>(T.b.size()); ++k) {
        const BuildingSpec& b = T.b[k];
        if (b.side != SIDE_BACK && b.side != SIDE_FRONT) continue;
        if (b.detached_m.hi > 0) continue;
        if (r.random() >= b.prob) continue;
        const double avail = W - 2.0 * t;
        if (avail < 2.0) continue;
        Local q;
        double d = std::min(b.depth_m.sample(r), 0.45 * Deff);
        if (!b.attach.empty()) {
            const int base = b.side == SIDE_BACK ? last_back : last_front;
            if (base < 0) continue;
            const Local& B = bs[base];
            const double room = b.attach == "left" ? B.x0 - (-0.5 * W + t) : (0.5 * W - t) - B.x1;
            if (room < 2.4) continue;
            const double L = length_of(b, room);
            if (L < 2.4) continue;
            if (b.attach == "left") q.x1 = B.x0, q.x0 = B.x0 - L;
            else q.x0 = B.x1, q.x1 = B.x1 + L;
            d = std::min(d, b.side == SIDE_BACK ? B.y1 - B.y0 : B.y1 - B.y0);
        } else {
            const double L = length_of(b, avail);
            const double xc = -0.5 * W + t + 0.5 * L + b.pos * (avail - L);
            q.x0 = xc - 0.5 * L, q.x1 = xc + 0.5 * L;
        }
        if (b.side == SIDE_BACK) {
            q.y0 = y_back, q.y1 = q.y0 + d;
            q.facing = b.face == FACE_OUT ? PI : 0.0;
            back_d = std::max(back_d, d);
        } else {
            q.y1 = 0.5 * D - t, q.y0 = q.y1 - d;
            q.facing = b.face == FACE_IN ? PI : 0.0;
            front_d = std::max(front_d, d);
        }
        q.spec = k;
        bs.push_back(q);
        if (b.attach.empty()) (b.side == SIDE_BACK ? last_back : last_front) = static_cast<int>(bs.size()) - 1;
    }
    // 2. 两厢：在两排之间
    const double yb = y_back + back_d + (back_d > 0 ? 0.8 : 0.0);
    const double yf = 0.5 * D - t - front_d - (front_d > 0 ? 0.8 : 2.0);
    double left_d = 0.0;
    for (int k = 0; k < static_cast<int>(T.b.size()); ++k) {
        const BuildingSpec& b = T.b[k];
        if (b.side != SIDE_LEFT && b.side != SIDE_RIGHT) continue;
        if (b.detached_m.hi > 0) continue;
        if (r.random() >= b.prob) continue;
        const double avail = yf - yb;
        if (avail < 2.5) continue;
        const double L = length_of(b, avail), d = b.depth_m.sample(r);
        // 院心要留够：两厢之间 ≥ yard_min
        const double other = b.side == SIDE_RIGHT ? left_d : 0.0;
        if (W - 2.0 * t - d - other < w.st.yard_min_m) continue;
        const double yc = yf - 0.5 * L - b.pos * (avail - L);
        Local q;
        q.y0 = yc - 0.5 * L, q.y1 = yc + 0.5 * L;
        if (b.side == SIDE_LEFT) {
            q.x0 = -0.5 * W + t, q.x1 = q.x0 + d;
            q.facing = b.face == FACE_IN ? 0.5 * PI : b.face == FACE_OUT ? -0.5 * PI : 0.0;
            left_d = d;
        } else {
            q.x1 = 0.5 * W - t, q.x0 = q.x1 - d;
            q.facing = b.face == FACE_IN ? -0.5 * PI : b.face == FACE_OUT ? 0.5 * PI : 0.0;
        }
        q.spec = k;
        bs.push_back(q);
    }
    // 3. 门：路在哪一边 → 这边的哪个角（风格规则，模板可覆盖）
    const double gw = w.st.gate_w.sample(r);
    const int side = C.access_side;
    std::string rule = side == SIDE_FRONT ? w.st.gate_front : side == SIDE_BACK ? w.st.gate_back : w.st.gate_side;
    if (!T.gate.empty()) rule = T.gate;
    const double inset = t + 0.5 * gw + 0.3;
    double gx = 0.0, gy = 0.0;
    double pos;   // 门在这条边上的坐标（前后边是 x，左右边是 y）
    if (side == SIDE_FRONT || side == SIDE_BACK) {
        // 「左 / 右」是站在院里面朝前沿时的左右，前后两边都一样
        pos = rule == "left" ? -0.5 * W + inset : rule == "right" ? 0.5 * W - inset : 0.0;
        gx = pos, gy = side == SIDE_FRONT ? 0.5 * D : -0.5 * D;
    } else {
        pos = rule == "back" ? -0.5 * D + inset : rule == "center" ? 0.0 : 0.5 * D - inset;   // 侧门默认靠前
        gy = pos, gx = side == SIDE_LEFT ? -0.5 * W : 0.5 * W;
    }
    // 门洞（进门的过道）：沿门这条边宽 gw、往里深 passage 的一条。压着它的房：
    //   顺着这条边的（面阔沿边）——先试整栋挪开；挪不开（或是这条边上的一整排、模板要门楼）就在门宽处截断，截在边上的一排里补一间门楼；
    //   垂直于这条边的——把靠边那头缩回到过道以里；它本来贴着这条边时，缩掉的那头做门楼（侧门开在倒座 / 正房的尽间）。
    const double passage = 3.5;
    const bool edge_x = side == SIDE_FRONT || side == SIDE_BACK;   // 门这条边沿局部 x
    double hx0, hx1, hy0, hy1;
    if (edge_x) {
        hx0 = pos - 0.5 * gw, hx1 = pos + 0.5 * gw;
        hy0 = side == SIDE_FRONT ? 0.5 * D - passage : -0.5 * D, hy1 = side == SIDE_FRONT ? 0.5 * D : -0.5 * D + passage;
    } else {
        hy0 = pos - 0.5 * gw, hy1 = pos + 0.5 * gw;
        hx0 = side == SIDE_LEFT ? -0.5 * W : 0.5 * W - passage, hx1 = side == SIDE_LEFT ? -0.5 * W + passage : 0.5 * W;
    }
    auto touches_edge = [&](const Local& q) {
        return side == SIDE_FRONT ? q.y1 >= 0.5 * D - t - 1e-6 : side == SIDE_BACK ? q.y0 <= -0.5 * D + t + 1e-6
             : side == SIDE_LEFT ? q.x0 <= -0.5 * W + t + 1e-6 : q.x1 >= 0.5 * W - t - 1e-6;
    };
    std::vector<Local> out;
    std::vector<Local> gatehouses;
    for (const Local& q : bs) {
        if (!(q.x0 < hx1 && q.x1 > hx0 && q.y0 < hy1 && q.y1 > hy0)) {
            out.push_back(q);
            continue;
        }
        const bool len_x = std::fabs(std::cos(q.facing)) > 0.5;
        const bool parallel = len_x == edge_x;
        if (parallel) {
            const double a0 = edge_x ? q.x0 : q.y0, a1 = edge_x ? q.x1 : q.y1;
            const double h0 = edge_x ? hx0 : hy0, h1 = edge_x ? hx1 : hy1;
            const bool wing = !edge_x;   // 厢房沿 y 挪时不出两排之间
            const double lo = edge_x ? -0.5 * W + t : (wing ? yb : -0.5 * D + t), hi = edge_x ? 0.5 * W - t : (wing ? yf : 0.5 * D - t);
            const double Lq = a1 - a0, mid = 0.5 * (a0 + a1);
            const bool row = touches_edge(q) && Lq > 0.8 * (hi - lo);
            double shift = NaN;
            for (auto [s0, s1] : {std::pair<double, double>{lo, h0 - 0.2}, std::pair<double, double>{h1 + 0.2, hi}}) {
                if (s1 - s0 < Lq) continue;
                const double m = std::clamp(mid, s0 + 0.5 * Lq, s1 - 0.5 * Lq);
                if (!std::isfinite(shift) || std::fabs(m - mid) < std::fabs(shift)) shift = m - mid;
            }
            if (std::isfinite(shift) && !(row && T.gate_house)) {
                Local p = q;
                (edge_x ? p.x0 : p.y0) += shift;
                (edge_x ? p.x1 : p.y1) += shift;
                out.push_back(p);
                continue;
            }
            for (auto [s0, s1] : {std::pair<double, double>{a0, h0 - 0.2}, std::pair<double, double>{h1 + 0.2, a1}}) {
                if (s1 - s0 < 2.4) continue;
                Local p = q;
                (edge_x ? p.x0 : p.y0) = s0;
                (edge_x ? p.x1 : p.y1) = s1;
                out.push_back(p);
            }
            if (touches_edge(q) && T.gate_house) {
                Local g = q;
                (edge_x ? g.x0 : g.y0) = std::max(a0, h0);
                (edge_x ? g.x1 : g.y1) = std::min(a1, h1);
                gatehouses.push_back(g);
            }
        } else {
            // 垂直于门这条边：靠边那头缩回到过道以里
            Local p = q, g = q;
            bool keep = true;
            if (side == SIDE_FRONT) p.y1 = std::min(q.y1, hy0 - 0.2), g.y0 = hy0, keep = p.y1 - p.y0 >= 2.4;
            else if (side == SIDE_BACK) p.y0 = std::max(q.y0, hy1 + 0.2), g.y1 = hy1, keep = p.y1 - p.y0 >= 2.4;
            else if (side == SIDE_LEFT) p.x0 = std::max(q.x0, hx1 + 0.2), g.x1 = hx1, keep = p.x1 - p.x0 >= 2.4;
            else p.x1 = std::min(q.x1, hx0 - 0.2), g.x0 = hx0, keep = p.x1 - p.x0 >= 2.4;
            if (keep) out.push_back(p);
            if (touches_edge(q) && T.gate_house) gatehouses.push_back(g);
        }
    }
    const double gface = side == SIDE_FRONT ? 0.0 : side == SIDE_BACK ? PI : side == SIDE_LEFT ? -0.5 * PI : 0.5 * PI;
    if (!gatehouses.empty()) {
        Local g = gatehouses.front();
        g.gatehouse = true;
        g.spec = -1;
        g.facing = gface;
        out.push_back(g);
    } else if (C.walled) {
        // 墙上的小门楼：跨在墙线上，宽 gw、深 1.2 m
        Local g{};
        if (edge_x) {
            g.x0 = hx0, g.x1 = hx1;
            const double yy = side == SIDE_FRONT ? 0.5 * D - 0.6 : -0.5 * D + 0.6;
            g.y0 = yy - 0.6, g.y1 = yy + 0.6;
        } else {
            g.y0 = hy0, g.y1 = hy1;
            const double xx = side == SIDE_LEFT ? -0.5 * W + 0.6 : 0.5 * W - 0.6;
            g.x0 = xx - 0.6, g.x1 = xx + 0.6;
        }
        g.gatehouse = true;
        g.spec = -2;
        g.facing = gface;
        out.push_back(g);
    }
    // 门处理完后同一院里不许相交：门楼优先，其余按模板次序（正房在前），后来的压着前面的就不要
    {
        std::vector<Local> keep;
        auto hit = [](const Local& a, const Local& b) {
            return a.x0 < b.x1 - 0.05 && b.x0 < a.x1 - 0.05 && a.y0 < b.y1 - 0.05 && b.y0 < a.y1 - 0.05;
        };
        std::stable_sort(out.begin(), out.end(), [](const Local& a, const Local& b) {
            const int ka = a.gatehouse ? -1 : a.spec, kb = b.gatehouse ? -1 : b.spec;
            return ka < kb;
        });
        for (const Local& q : out) {
            bool clash = false;
            for (const Local& k : keep) clash = clash || hit(q, k);
            if (!clash) keep.push_back(q);
        }
        out.swap(keep);
    }
    // 4. 落到平面坐标，台基与土方
    const V2 fv = bearing_vec(C.plot.facing), rv = bearing_right(C.plot.facing);
    auto world = [&](double x, double y) { return C.plot.c + rv * x + fv * y; };
    C.gate = world(gx, gy);
    C.gate_bearing = wrap_pi(C.plot.facing + (side == SIDE_FRONT ? 0.0 : side == SIDE_BACK ? PI : side == SIDE_LEFT ? -0.5 * PI : 0.5 * PI));
    const double cell = w.s.res_m * w.s.res_m;
    for (const Local& q : out) {
        Building b;
        b.compound = ci;
        const double facing = wrap_pi(C.plot.facing + q.facing);
        const bool xs = std::fabs(std::cos(q.facing)) > 0.5;   // 面阔沿局部 x
        const double L = along_len(q), d = xs ? q.y1 - q.y0 : q.x1 - q.x0;
        b.box = {world(0.5 * (q.x0 + q.x1), 0.5 * (q.y0 + q.y1)), facing, 0.5 * L, 0.5 * d};
        // 台基：这一栋脚下地面的中位（坡上的院子后排高、前排低，院子成台地）
        std::vector<double> hs;
        raster_obb(w.s, b.box, 0.0, [&](int i, int j) {
            const double h = w.s.height(i, j);
            if (std::isfinite(h)) hs.push_back(h);
        });
        double ground = C.base_m;
        if (!hs.empty()) {
            std::vector<double> tmp = hs;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            ground = tmp[tmp.size() / 2];
        }
        double plinth = 0.15;
        if (q.spec >= 0) {
            const BuildingSpec& S = T.b[q.spec];
            b.func = S.func, b.role = S.role, b.name = S.name, b.roof = S.roof, b.material = S.material;
            b.storeys = S.storeys, b.eave_m = S.eave_m, b.pitch_deg = S.pitch_deg;
            plinth = S.plinth_m;
        } else {
            b.func = "gate", b.role = q.spec == -1 ? "gatehouse" : "gate", b.name = q.spec == -1 ? "gatehouse" : "gate";
            b.roof = "gable", b.storeys = 1, b.eave_m = q.spec == -1 ? 3.0 : 2.6, b.pitch_deg = 30.0;
        }
        const bool cave = b.roof == "cave" || T.dug_in;   // 靠崖窑院里的房也落在削出来的院子上（院子整块是挖的）
        if (cave) {
            // 窑是挖出来的：靠崖窑的地面 = 院子，地坑院的 = 坑底；挖方记下，不算台基的挖填（TP-ground 不查）
            ground = T.sunken ? C.base_m - pit_depth : C.base_m;
            b.base_m = ground;
            for (double h : hs)
                if (h > ground) b.cut_m3 += (h - ground) * cell;
        } else {
            b.base_m = ground + plinth;
            for (double h : hs) {
                if (h > ground) b.cut_m3 += (h - ground) * cell;
                else b.fill_m3 += (ground - h) * cell;
                b.max_cut_m = std::max(b.max_cut_m, std::fabs(h - ground));
            }
        }
        raster_obb(w.s, b.box, 0.0, [&](int i, int j) { w.plan.occ(i, j) = OCC_BUILDING; });
        C.buildings.push_back(static_cast<int>(w.plan.buildings.size()));
        w.plan.buildings.push_back(b);
    }
    // 5. 园（地块后部）、地坑院的坑
    if (T.garden_frac > 0.0) {
        Feature g;
        g.kind = "garden";
        g.compound = ci;
        const double y1 = -0.5 * D + T.garden_frac * D;
        g.poly = {world(-0.5 * W + t, -0.5 * D + t), world(0.5 * W - t, -0.5 * D + t), world(0.5 * W - t, y1), world(-0.5 * W + t, y1)};
        g.p = world(0.0, 0.5 * (-0.5 * D + y1));
        g.facing = C.plot.facing;
        w.plan.features.push_back(g);
    }
    if (T.sunken) {
        double x0 = -0.5 * W + t, x1 = 0.5 * W - t, y0 = -0.5 * D + t, y1 = 0.5 * D - t;
        for (const Local& q : out) {
            if (q.gatehouse) continue;
            const double cx = 0.5 * (q.x0 + q.x1), cy = 0.5 * (q.y0 + q.y1);
            if (std::fabs(cx) * D > std::fabs(cy) * W) {
                if (cx < 0) x0 = std::max(x0, q.x1);
                else x1 = std::min(x1, q.x0);
            } else {
                if (cy < 0) y0 = std::max(y0, q.y1);
                else y1 = std::min(y1, q.y0);
            }
        }
        if (x1 - x0 > 3.0 && y1 - y0 > 3.0) {
            Feature g;
            g.kind = "pit";
            g.compound = ci;
            g.poly = {world(x0, y0), world(x1, y0), world(x1, y1), world(x0, y1)};
            g.p = world(0.5 * (x0 + x1), 0.5 * (y0 + y1));
            g.z = C.base_m - pit_depth;
            g.facing = C.plot.facing;
            w.plan.features.push_back(g);
        }
    }
    // 6. 院外单栋（北欧：浴房近水、铁匠房离主院 ≥ 20 m）
    for (const BuildingSpec& S : T.b) {
        if (S.detached_m.hi <= 0 || r.random() >= S.prob) continue;
        const double L = S.bays.hi > 0 ? S.bays.sample_int(r) * bay : S.depth_m.sample(r) * 1.3, d = S.depth_m.sample(r);
        const double r0 = std::hypot(C.plot.hw, C.plot.hd);
        double best = -INF;
        Obb arg{};
        for (int k = 0; k < 24; ++k) {
            const double th = C.plot.facing + 2.0 * PI * k / 24.0;
            for (double dd = S.detached_m.lo; dd <= S.detached_m.hi + 1e-9; dd += 4.0) {
                const V2 p = C.plot.c + bearing_vec(th) * (r0 + dd);
                const Obb o{p, wrap_pi(th + PI), 0.5 * L, 0.5 * d};
                if (!obb_free(w, o, 0.0, false, 15.0) || road_hits_obb(w, o, -0.5) || hits_built(w, o, 1.5)) continue;
                bool clash = false;
                for (const Compound& o2 : w.plan.compounds) clash = clash || (len(o2.plot.c - p) < 120.0 && overlap(o2.plot, o, 2.0));
                if (clash) continue;
                double sc = -dd / 50.0 + 0.05 * r.random();
                if (S.near == "water") {
                    int i, j;
                    const double dw = w.s.cell_of(p, i, j) ? w.s.water_dist_m(i, j) : 1e9;
                    if (dw > 100.0) continue;   // 桑拿在湖岸 20–100 m（估）
                    sc -= dw / 30.0;
                }
                if (sc > best) best = sc, arg = o;
            }
        }
        if (!std::isfinite(best)) continue;
        Building b;
        b.compound = ci;
        b.box = arg;
        b.func = S.func, b.role = S.role, b.name = S.name, b.roof = S.roof, b.material = S.material;
        b.storeys = S.storeys, b.eave_m = S.eave_m, b.pitch_deg = S.pitch_deg;
        std::vector<double> hs;
        raster_obb(w.s, arg, 0.0, [&](int i, int j) {
            const double h = w.s.height(i, j);
            if (std::isfinite(h)) hs.push_back(h);
        });
        double ground = C.base_m;
        if (!hs.empty()) {
            std::vector<double> tmp = hs;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            ground = tmp[tmp.size() / 2];
        }
        b.base_m = ground + S.plinth_m;
        for (double h : hs) {
            if (h > ground) b.cut_m3 += (h - ground) * cell;
            else b.fill_m3 += (ground - h) * cell;
            b.max_cut_m = std::max(b.max_cut_m, std::fabs(h - ground));
        }
        mark_obb(w, arg, OCC_BUILDING, 0.0);
        C.buildings.push_back(static_cast<int>(w.plan.buildings.size()));
        w.plan.buildings.push_back(b);
    }
}

// ---------------------------------------------------------------- 围合单体：圆楼 / 方楼 / 围龙屋
// 局部坐标同上（x 右、y 前）；环上的房间朝院心，一段几间合成一栋（段宽按内圈算，段与段在外圈留楔形缝，不相交）；
// 门在前方正中（占一段），中间祖堂；户 = 竖向一列房间，列数由算子按户数定、这里按地块尺寸排段。
void instantiate_enclosure(Work& w, int ci, Rng& r) {
    Compound& C = w.plan.compounds[ci];
    const TemplateSpec& T = w.st.templates[w.st.template_index(C.tmpl)];
    const double W = 2.0 * C.plot.hw, D = 2.0 * C.plot.hd;
    const double depth = T.ring_depth_m.sample(r), room = T.ring_room_m.sample(r);
    const int storeys = std::max(1, T.ring_storeys.sample_int(r));
    const BuildingSpec *ring = nullptr, *hall = nullptr, *gate = nullptr, *side = nullptr;
    for (const BuildingSpec& S : T.b) {
        if (S.role == "ring" && !ring) ring = &S;
        else if (S.role == "hall" && !hall) hall = &S;
        else if (S.role == "gate" && !gate) gate = &S;
        else if (S.role == "side_row" && !side) side = &S;
    }
    const V2 fv = bearing_vec(C.plot.facing), rv = bearing_right(C.plot.facing);
    auto world = [&](double x, double y) { return C.plot.c + rv * x + fv * y; };
    struct Piece {
        V2 c;              // 局部
        double facing;     // 相对地块朝向
        double L, d;
        const BuildingSpec* spec;
        int storeys;
        std::string role;
    };
    std::vector<Piece> ps;
    // o = 圆心（局部）；角从地块前方（+y）起、顺时针为正；[th0, th1] 这一段弧；gate_front：正前方那一段做门
    auto ring_arc = [&](V2 o, double R_out, double th0, double th1, bool gate_front) {
        const double R_in = R_out - depth, R_mid = 0.5 * (R_out + R_in);
        const double arc = (th1 - th0) * R_mid;
        const int n_rooms = std::max(4, static_cast<int>(std::floor(arc / room)));
        const int nseg = std::max(3, static_cast<int>(std::lround(n_rooms / 2.0)));   // 两间一段：外圈看着是圆的
        const double dth = (th1 - th0) / nseg;
        const double width = std::max(1.5, 2.0 * R_in * std::tan(0.5 * dth) - 0.1);
        int gate_k = -1;
        if (gate_front) {
            double bd = INF;
            for (int k = 0; k < nseg; ++k) {
                const double d = std::fabs(wrap_pi(th0 + (k + 0.5) * dth));
                if (d < bd) bd = d, gate_k = k;
            }
        }
        for (int k = 0; k < nseg; ++k) {
            const double th = th0 + (k + 0.5) * dth;
            const V2 c = o + V2{std::sin(th), std::cos(th)} * R_mid;
            const bool is_gate = k == gate_k;
            ps.push_back({c, wrap_pi(th + PI), width, depth, is_gate && gate ? gate : ring, is_gate ? 1 : storeys, is_gate ? "gate" : "ring"});
        }
        return n_rooms;
    };
    int rooms = 0;
    double gate_y = 0.5 * D;
    if (T.shape == "ring") {
        const double R_out = 0.5 * std::min(W, D) - 0.3;
        rooms = ring_arc({0.0, 0.0}, R_out, -PI, PI, true);
        gate_y = R_out;
        const double R_in = R_out - depth;
        if (hall && R_in > 7.0) ps.push_back({{0.0, 0.0}, 0.0, std::min(12.0, 0.8 * R_in), std::min(8.0, 0.6 * R_in), hall, 1, "hall"});
    } else if (T.shape == "square_ring") {
        const double x0 = -0.5 * W + 0.3, x1 = 0.5 * W - 0.3, y0 = -0.5 * D + 0.3, y1 = 0.5 * D - 0.3;
        const double gw = std::max(4.0, 1.5 * room);
        ps.push_back({{0.0, y0 + 0.5 * depth}, 0.0, x1 - x0, depth, ring, storeys, "ring"});                            // 后排
        ps.push_back({{x0 + 0.5 * depth, 0.0}, 0.5 * PI, y1 - y0 - 2.0 * depth - 0.2, depth, ring, storeys, "ring"});   // 左排（朝院心）
        ps.push_back({{x1 - 0.5 * depth, 0.0}, -0.5 * PI, y1 - y0 - 2.0 * depth - 0.2, depth, ring, storeys, "ring"});  // 右排
        const double half = 0.5 * (x1 - x0 - gw) - 0.1;
        ps.push_back({{x0 + 0.5 * half, y1 - 0.5 * depth}, PI, half, depth, ring, storeys, "ring"});                     // 前排左半
        ps.push_back({{x1 - 0.5 * half, y1 - 0.5 * depth}, PI, half, depth, ring, storeys, "ring"});                     // 前排右半
        ps.push_back({{0.0, y1 - 0.5 * depth}, 0.0, gw, depth, gate ? gate : ring, 1, "gate"});
        rooms = static_cast<int>(std::floor((2.0 * (x1 - x0) + 2.0 * (y1 - y0) - 4.0 * depth - gw) / room));
        gate_y = y1;
        const double inner = std::min(x1 - x0, y1 - y0) - 2.0 * depth;
        if (hall && inner > 12.0) ps.push_back({{0.0, 0.0}, 0.0, std::min(12.0, 0.5 * inner), std::min(8.0, 0.4 * inner), hall, 1, "hall"});
    } else {   // weilong：两堂两横一围龙（后面半圈），前面下堂开门
        const double Rw = 0.5 * W - 0.3;                    // 围龙半径
        const double yc = -0.5 * D + 0.3 + Rw;              // 围龙圆心（局部 y）
        rooms = ring_arc({0.0, yc}, Rw, 0.5 * PI, 1.5 * PI, false);   // 后半圈：从右（90°）经正后方到左（270°）
        const double row_d = side ? side->depth_m.mid() : depth;
        const double ys0 = yc + 0.3, ys1 = 0.5 * D - 0.3;
        const double xs = 0.5 * W - 0.3 - 0.5 * row_d;
        if (ys1 - ys0 > 6.0) {
            // 横屋朝轴线；依山前低后高，按每四间一段分开（各段分台）
            const BuildingSpec* sr = side ? side : ring;
            const int nsg = std::max(1, static_cast<int>(std::ceil((ys1 - ys0) / (4.0 * room))));
            const double sl = (ys1 - ys0) / nsg;
            for (int k = 0; k < nsg; ++k) {
                const double yc = ys0 + (k + 0.5) * sl;
                ps.push_back({{-xs, yc}, 0.5 * PI, sl - 0.3, row_d, sr, 1, "side_row"});
                ps.push_back({{xs, yc}, -0.5 * PI, sl - 0.3, row_d, sr, 1, "side_row"});
            }
            rooms += 2 * static_cast<int>(std::floor((ys1 - ys0) / room));
        }
        const double inner_w = W - 2.0 * (row_d + 0.3) - 4.0;
        const double hw = std::min(inner_w, 18.0);
        if (hw > 8.0) {
            const double hd = std::min(9.0, 0.25 * (ys1 - ys0));
            ps.push_back({{0.0, ys1 - 0.5 * hd}, 0.0, hw, hd, gate ? gate : hall, 1, "gate"});                 // 下堂（门）
            if (hall) ps.push_back({{0.0, yc + 0.4 * (ys1 - yc)}, 0.0, hw, hd, hall, 1, "hall"});           // 上堂（祖堂）
        }
        gate_y = ys1;
    }
    // 落到平面：整座楼一个台基（地块中位 + 台高）
    const double cell = w.s.res_m * w.s.res_m;
    for (const Piece& q : ps) {
        if (!q.spec) continue;
        Building b;
        b.compound = ci;
        b.box = {world(q.c.x, q.c.y), wrap_pi(C.plot.facing + q.facing), 0.5 * q.L, 0.5 * q.d};
        b.func = q.spec->func, b.role = q.role, b.name = q.spec->name, b.roof = q.spec->roof, b.material = q.spec->material;
        b.storeys = q.storeys;
        b.eave_m = q.role == "ring" ? q.spec->eave_m * q.storeys : q.spec->eave_m;
        b.pitch_deg = q.spec->pitch_deg;
        // 台基：圆楼、方楼整座一个；围龙屋依山，各栋按自己脚下分台（前低后高、化胎在后）
        double ground = C.base_m;
        if (T.shape == "weilong") {
            std::vector<double> hs;
            raster_obb(w.s, b.box, 0.0, [&](int i, int j) {
                const double h = w.s.height(i, j);
                if (std::isfinite(h)) hs.push_back(h);
            });
            if (!hs.empty()) {
                std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
                ground = hs[hs.size() / 2];
            }
        }
        b.base_m = ground + q.spec->plinth_m;
        raster_obb(w.s, b.box, 0.0, [&](int i, int j) {
            const double h = w.s.height(i, j);
            if (std::isfinite(h)) {
                if (h > ground) b.cut_m3 += (h - ground) * cell;
                else b.fill_m3 += (ground - h) * cell;
                b.max_cut_m = std::max(b.max_cut_m, std::fabs(h - ground));
            }
            w.plan.occ(i, j) = OCC_BUILDING;
        });
        C.buildings.push_back(static_cast<int>(w.plan.buildings.size()));
        w.plan.buildings.push_back(b);
    }
    C.access_side = SIDE_FRONT;
    C.gate = world(0.0, gate_y);
    C.gate_bearing = C.plot.facing;
    w.plan.metrics["enclosure_rooms_sum"] += rooms;
}


}  // namespace skyisle::town
