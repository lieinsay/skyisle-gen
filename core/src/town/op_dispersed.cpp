// 形态算子「散居」（PLAN-TOWN 6.2；川西林盘、砺波散居村、北欧农庄 klyngetun）：
//   1. 按户数定点数（每点一户或一簇），在整个窗口里按加权泊松盘撒点：评分 = 兴趣 + 在田中间 + 离村心近一点，点距 ≥ spacing
//      （撒不够就把点距一轮轮放小，最小 40 m）；
//   2. 每点落一簇：第一户在点上，其余在点周围 cluster_radius_m 以内，朝向按朝向链；
//   3. 路：从离村心最近的一簇起，像 Prim 一样一簇簇按坡度代价接到已有的路（田间道），簇里各户的门再修巷接上；
//   4. 林：ring 环林（林盘的竹林把一簇围起来）/ windward 冬季风上风侧的屋敷林（砺波カイニョ）；
//   5. 指标：点的最近邻距离中位、点的 Clark–Evans。
#include <algorithm>
#include <cmath>

#include "plan_work.hpp"

namespace skyisle::town {

void op_dispersed(Work& w) {
    const Site& s = w.s;
    const Style& st = w.st;
    Rng r = w.rng("dispersed");
    const int ng = static_cast<int>(w.hh_groups.size());
    // 1. 每簇几户
    std::vector<int> sizes;
    for (int k = 0; k < ng;) {
        const int c = std::clamp(st.pr("dispersed.cluster_hh").sample_int(r), 1, ng - k);
        sizes.push_back(c);
        k += c;
    }
    const int np = static_cast<int>(sizes.size());
    // 撒点的评分：兴趣 + 田中间（60 m 内田的占比）+ 离村心近一点
    const int stride = std::max(1, static_cast<int>(std::lround(6.0 / s.res_m)));
    const double win = 0.5 * std::min(s.W, s.H) * s.res_m;
    struct P {
        V2 p;
        double sc;
    };
    std::vector<P> cand;
    const double wi = st.pn("dispersed.w_interest"), wf = st.pn("dispersed.w_farmland");
    const int rf = std::max(1, static_cast<int>(std::lround(60.0 / s.res_m)));
    // 离窗口边留出一簇加一条路的地（贴着窗口边的簇常常接不出路）
    const int mb = std::max(stride, static_cast<int>(std::lround((st.pn("dispersed.cluster_radius_m") + 50.0) / s.res_m)));
    for (int i = mb; i < s.H - mb; i += stride)
        for (int j = mb; j < s.W - mb; j += stride) {
            if (!w.f.buildable(i, j)) continue;
            const float v = w.interest(i, j);
            if (!std::isfinite(v)) continue;
            int nf = 0, nt = 0;
            for (int a = std::max(0, i - rf); a < std::min(s.H, i + rf); a += 4 * stride)
                for (int b = std::max(0, j - rf); b < std::min(s.W, j + rf); b += 4 * stride) ++nt, nf += s.farmland(a, b) != 0;
            const V2 p = s.center(i, j);
            cand.push_back({p, wi * v + wf * (nt ? static_cast<double>(nf) / nt : 0.0) - st.pn("dispersed.w_center") * len(p - w.center) / win + 0.15 * r.random()});
        }
    std::sort(cand.begin(), cand.end(), [](const P& a, const P& b) { return a.sc > b.sc; });
    const Range spacing = st.pr("dispersed.spacing_m");
    std::vector<V2> pts;
    // 第一点：村心
    pts.push_back(w.center);
    // 多撒一半当备用：接不上路（隔着过不去的河、陡崖）的簇撤回后用备用点补
    const int npts = np + np / 2 + 3;
    for (double k = 1.0; static_cast<int>(pts.size()) < npts && k > 0.05; k *= 0.8) {
        const double dmin = std::max(40.0, spacing.lo * k);
        for (const P& c : cand) {
            if (static_cast<int>(pts.size()) >= npts) break;
            bool ok = true;
            for (const V2& q : pts) ok = ok && len(q - c.p) >= dmin * (1.0 + 0.4 * (spacing.hi / std::max(1.0, spacing.lo) - 1.0) * r.random());
            if (ok) pts.push_back(c.p);
        }
        if (dmin <= 40.0) break;
    }
    // 2–3. 从离村心最近的点起，每点落一簇，落下就接路（第一簇门口起一段路当根，以后各簇按坡度代价接到已有的路）；
    //      接不上（隔着过不去的河、陡崖）就撤回这一簇，户留给后面的点或最后的团块生长
    auto by_center = [&](const V2& a, const V2& b) { return len(a - w.center) < len(b - w.center); };
    const auto main_end = pts.begin() + std::min(static_cast<int>(pts.size()), np);
    std::sort(pts.begin() + 1, main_end, by_center);   // 正选的点按离村心排，备用的排在后面
    std::sort(main_end, pts.end(), by_center);
    const double crad = st.pn("dispersed.cluster_radius_m");
    std::vector<std::vector<int>> clusters;   // 各簇的宅院号
    std::vector<V2> centers;
    const double pw = road_width(w, RC_PATH, r), lw = st.lane_w.mid();
    bool any = false;
    size_t g = 0;
    int si = 0;
    for (int q = 0; q < static_cast<int>(pts.size()) && g < w.hh_groups.size(); ++q) {
        std::vector<int> cl;
        const size_t g0 = g;
        const int want = sizes[std::min(static_cast<size_t>(si), sizes.size() - 1)];
        for (int h = 0; h < want && g < w.hh_groups.size(); ++h) {
            double best = -INF;
            Slot arg;
            const int tmpl = pick_template(w, r);
            const TemplateSpec& T = st.templates[tmpl];
            const double Wd = T.plot_w.sample(r), Dp = T.plot_d.sample(r);
            for (double rad = 0.0; rad <= crad + (h ? 0.5 * (Wd + Dp) : 0.0); rad += 4.0)
                for (int k = 0; k < (rad == 0.0 ? 1 : 12); ++k) {
                    const V2 p = pts[q] + bearing_vec(2 * PI * (k + 0.5 * (rad / 4.0)) / 12) * rad;
                    const OrientCtx c = orient_ctx(w, p);
                    Slot sl;
                    sl.tmpl = tmpl;
                    sl.box = {p, solve_facing(st.rules, c, st.snap, c.sun), 0.5 * Wd, 0.5 * Dp};
                    sl.access_side = SIDE_FRONT;
                    sl.access = p + bearing_vec(sl.box.facing) * (0.5 * Dp + 3.0);
                    if (!plot_fits(w, sl, nullptr)) continue;
                    // 同一簇挨着一点、别挤死（留出巷）
                    double dn = 0.0;
                    for (int ci : cl) dn = std::max(dn, 1.0 / (1.0 + len(w.plan.compounds[ci].plot.c - p)));
                    bool tight = false;
                    for (int ci : cl) tight = tight || overlap(w.plan.compounds[ci].plot, sl.box, 2.0 * st.lane_w.lo + 1.0);
                    if (tight) continue;
                    const double sc = sl.interest - rad / std::max(10.0, crad) + 3.0 * dn;
                    if (sc > best) best = sc, arg = sl;
                }
            if (!std::isfinite(best)) break;
            cl.push_back(commit_compound(w, arg, "house", "dwelling", "", w.hh_groups[g]));
            ++g;
        }
        if (cl.empty()) continue;
        V2 cc{};
        for (int ci : cl) cc = cc + w.plan.compounds[ci].plot.c;
        cc = cc * (1.0 / cl.size());
        // 接路：离已接上的最近那一簇多远，就在多大范围里找
        double reach = 0.0;
        if (!centers.empty()) {
            reach = INF;
            for (const V2& c : centers) reach = std::min(reach, len(c - cc));
        }
        reach = std::min(reach, 2.5 * spacing.hi) + 80.0;
        bool linked = false;
        for (int ci : cl) {
            if (linked) break;
            const Compound& C = w.plan.compounds[ci];
            const TemplateSpec& T = st.templates[st.template_index(C.tmpl)];
            const V2 nv = side_normal(C.plot, C.access_side);
            const V2 door = gate_point_of(w, C.plot, C.access_side, T) + nv * (0.5 * pw + 1.5);
            int di, dj;
            if (!s.cell_of(door, di, dj) || w.blocked(di, dj) || w.plan.occ(di, dj) != OCC_FREE) continue;
            if (!any) {
                // 第一簇：门口往外一段路，当作路网的根
                const V2 far = door + nv * 25.0;
                int fi, fj;
                if (s.cell_of(far, fi, fj) && !w.blocked(fi, fj) && s.water(fi, fj) == WATER_NONE && !s.sky(fi, fj) &&
                    road_fit(w, {door, far}) >= pw && add_road(w, {door, far}, RC_PATH, pw) >= 0)
                    any = linked = true;
            } else if (link_road(w, door, RC_PATH, pw, reach) >= 0) {
                linked = true;
            }
        }
        if (!linked) {
            // 撤回这一簇（它们是最后落的几块），户留给后面
            for (size_t k = 0; k < cl.size(); ++k) withdraw_last_compound(w);
            g = g0;
            continue;
        }
        ++si;
        for (int ci : cl) {
            if (lane_to_gate(w, ci, lw, crad + 40.0)) continue;
            // 簇里的巷修不出来（被邻户挡着）：门口直接按坡度代价接到最近的路，范围放大
            Compound& C = w.plan.compounds[ci];
            const TemplateSpec& T = st.templates[st.template_index(C.tmpl)];
            for (int sd : std::initializer_list<int>{C.access_side, SIDE_FRONT, SIDE_LEFT, SIDE_RIGHT, SIDE_BACK}) {
                const V2 door = gate_point_of(w, C.plot, sd, T) + side_normal(C.plot, sd) * (0.5 * lw + 1.2);
                int di, dj;
                if (!s.cell_of(door, di, dj) || w.blocked(di, dj) || w.plan.occ(di, dj) != OCC_FREE) continue;
                const int id = link_road(w, door, RC_LANE, lw, crad + 200.0);
                if (id < 0) continue;
                C.access_side = sd;
                C.access = w.plan.roads[id].line.back();
                break;
            }
        }
        centers.push_back(cc);
        clusters.push_back(cl);
    }
    // 4. 林
    const std::string grove = st.ps("dispersed.grove", "none");
    if (grove != "none") {
        const Range gw = st.pr("dispersed.grove_width_m");
        for (size_t k = 0; k < clusters.size(); ++k) {
            double rin = 0.0;
            for (int ci : clusters[k]) {
                const Obb& o = w.plan.compounds[ci].plot;
                rin = std::max(rin, len(o.c - centers[k]) + std::hypot(o.hw, o.hd));
            }
            rin += 3.0;
            const double wd = gw.sample(r);
            Feature ft;
            ft.kind = "grove";
            ft.func = grove;
            ft.name = st.ps("dispersed.grove_name", "");
            ft.p = centers[k];
            ft.r = rin;
            ft.compound = clusters[k][0];
            const int nseg = 36;
            const double ph = r.uniform(0, 2 * PI);
            if (grove == "ring") {
                for (int q = 0; q < nseg; ++q) {
                    const double th = 2 * PI * q / nseg;
                    ft.poly.push_back(centers[k] + bearing_vec(th) * ((rin + wd) * (1.0 + 0.08 * std::sin(3 * th + ph))));
                }
                ft.facing = 0.0;
            } else {   // windward：冬季风上风侧 ±70° 的一段弧带
                const double wf = std::isfinite(w.f.wind_from) ? w.f.wind_from : wrap_pi(w.f.sun_bearing + PI);
                ft.facing = wf;
                for (int q = 0; q <= nseg / 2; ++q) {
                    const double th = wf - 1.22 + 2.44 * q / (nseg / 2);
                    ft.poly.push_back(centers[k] + bearing_vec(th) * (rin + wd));
                }
                for (int q = nseg / 2; q >= 0; --q) {
                    const double th = wf - 1.22 + 2.44 * q / (nseg / 2);
                    ft.poly.push_back(centers[k] + bearing_vec(th) * rin);
                }
            }
            // 林占住空地（别的东西不往里放；路照走）
            raster_polygon(s, ft.poly, [&](int i, int j) {
                if (len(s.center(i, j) - centers[k]) < rin && grove == "ring") return;
                if (w.plan.occ(i, j) == OCC_FREE && s.water(i, j) == WATER_NONE && !s.sky(i, j)) w.plan.occ(i, j) = OCC_OPEN;
            });
            w.plan.features.push_back(ft);
        }
    }
    // 5. 指标：点（簇心）的最近邻
    if (centers.size() >= 2) {
        std::vector<double> nn;
        for (size_t a = 0; a < centers.size(); ++a) {
            double d = INF;
            for (size_t b = 0; b < centers.size(); ++b)
                if (a != b) d = std::min(d, len(centers[a] - centers[b]));
            nn.push_back(d);
        }
        std::vector<double> t = nn;
        std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
        w.plan.metrics["site_nn_median_m"] = t[t.size() / 2];
        // Clark–Evans：观测的平均最近邻 / 随机分布的期望（0.5 / √密度）；面积用窗口里可建的地
        double mean = 0.0;
        for (double d : nn) mean += d;
        mean /= nn.size();
        size_t nb = 0;
        for (uint8_t v : w.f.buildable.v) nb += v != 0;
        const double area = std::max(1.0, nb * s.res_m * s.res_m);
        w.plan.metrics["site_clark_evans"] = mean / (0.5 / std::sqrt(centers.size() / area));
        w.plan.metrics["sites"] = static_cast<double>(centers.size());
    }
    if (g < w.hh_groups.size()) organic_fill(w, g);
}

}  // namespace skyisle::town
