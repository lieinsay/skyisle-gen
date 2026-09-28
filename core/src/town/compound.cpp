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
        if (r.random() >= b.prob) continue;
        const double avail = W - 2.0 * t;
        if (avail < 2.0) continue;
        Local q;
        double d = std::min(b.depth_m.sample(r), 0.45 * D);
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
            q.y0 = -0.5 * D + t, q.y1 = q.y0 + d;
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
    const double yb = -0.5 * D + t + back_d + (back_d > 0 ? 0.8 : 0.0);
    const double yf = 0.5 * D - t - front_d - (front_d > 0 ? 0.8 : 2.0);
    double left_d = 0.0;
    for (int k = 0; k < static_cast<int>(T.b.size()); ++k) {
        const BuildingSpec& b = T.b[k];
        if (b.side != SIDE_LEFT && b.side != SIDE_RIGHT) continue;
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
        b.base_m = ground + plinth;
        for (double h : hs) {
            if (h > ground) b.cut_m3 += (h - ground) * cell;
            else b.fill_m3 += (ground - h) * cell;
            b.max_cut_m = std::max(b.max_cut_m, std::fabs(h - ground));
        }
        raster_obb(w.s, b.box, 0.0, [&](int i, int j) { w.plan.occ(i, j) = OCC_BUILDING; });
        C.buildings.push_back(static_cast<int>(w.plan.buildings.size()));
        w.plan.buildings.push_back(b);
    }
}

}  // namespace skyisle::town
