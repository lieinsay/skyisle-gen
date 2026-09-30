// 5.3b 河道成形（river.py 同式）。
#include "skyisle/island/river.hpp"

#include <algorithm>
#include <cmath>

namespace skyisle::island {

Channels carve_channels(const GridD& h, const GridD& hf, const Mask& mk, const Mask& lake, const std::vector<int64_t>& recv,
                        const GridD& Akm, const Grid<uint8_t>& river_lvl, const Grid<uint8_t>& stream, double P_mm, double runoff, double rim,
                        double keel, double res_m, double year_s, const Config& c, bool /*is_main*/, const GridD* Qin,
                        const GridD* wall_deg) {
    const int H = h.H, W = h.W;
    const size_t N = h.size();
    Channels out;
    Mask work(H, W, 0), center_r(H, W, 0), center_s(H, W, 0), seed(H, W, 0);
    bool any_seed = false, any_r = false;
    for (size_t k = 0; k < N; ++k) {
        work.v[k] = (mk.v[k] && !lake.v[k]) ? 1 : 0;
        center_r.v[k] = (work.v[k] && river_lvl.v[k] > 0) ? 1 : 0;
        center_s.v[k] = (work.v[k] && stream.v[k] > 0 && !center_r.v[k]) ? 1 : 0;
        seed.v[k] = (center_r.v[k] || center_s.v[k]) ? 1 : 0;
        any_seed = any_seed || seed.v[k];
        any_r = any_r || center_r.v[k];
    }
    out.width = GridD(H, W, 0.0);
    out.depth = GridD(H, W, 0.0);
    if (!any_seed) {
        out.h_new = h;
        out.lvl = river_lvl;
        out.floodplain = Grid<uint8_t>(H, W, 0);
        return out;
    }
    const double wa = c.get("hydro.width_a"), wb = c.get("hydro.width_b"), wsc = c.get("hydro.width_scale");
    const double dc = c.get("hydro.depth_c"), df = c.get("hydro.depth_f"), dsc = c.get("hydro.depth_scale");
    const double sm = c.get("hydro.stream_width_mult");
    std::vector<double> Q(N), width(N), depth(N), incise(N), base(N), bed(N, NaN);
    double amax = -INF;
    for (size_t k = 0; k < N; ++k)
        if (center_r.v[k]) amax = std::max(amax, Akm.v[k]);
    if (!any_r) amax = 1.0;
    const double inc_m = c.get("hydro.incise_m"), s_inc = c.get("hydro.stream_incise_m");
    for (size_t k = 0; k < N; ++k) {
        Q[k] = Qin ? Qin->v[k] : Akm.v[k] * 1e6 * (P_mm / 1000.0) * runoff / year_s;
        const double q = std::max(Q[k], 0.0);
        const double w = wa * np_pow(q, wb) * wsc;
        const double d = dc * np_pow(q, df) * dsc;
        width[k] = center_r.v[k] ? w : (center_s.v[k] ? sm * w : 0.0);
        depth[k] = center_r.v[k] ? d : (center_s.v[k] ? sm * d : 0.0);
        incise[k] = center_r.v[k] ? inc_m * std::sqrt(clip(Akm.v[k] / amax, 0.0, 1.0)) : (center_s.v[k] ? s_inc : 0.0);
        base[k] = std::min(mk.v[k] ? h.v[k] : INF, mk.v[k] ? hf.v[k] : INF);
        if (seed.v[k]) bed[k] = base[k] - depth[k] - incise[k];
    }
    // 河床向下游单调：按路由面降序，把上游河床（减一个极小落差）传给下游
    const double floor = rim - c.get("hydro.notch_m");
    std::vector<int32_t> order;
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k]) order.push_back(static_cast<int32_t>(k));
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return hf.v[a] > hf.v[b]; });
    std::vector<double> bl(N), Lkm(N, 0.0);
    for (size_t k = 0; k < N; ++k) bl[k] = std::max(std::isnan(bed[k]) ? INF : bed[k], floor);
    const double step = res_m / 1000.0;
    for (int32_t k : order) {
        const int64_t r = recv[k];
        if (r >= 0 && seed.v[r]) {
            if (bl[r] > bl[k] - 0.01) bl[r] = std::max(floor, bl[k] - 0.01);
            if (center_r.v[k] && center_r.v[r]) Lkm[r] = std::max(Lkm[r], Lkm[k] + step);
        }
    }
    // 支流接干流：溪涧河床从汇入处往上游按最大比降爬升（下游先算）
    const double rise = res_m * c.get("hydro.stream_grade_max");
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int32_t k = *it;
        const int64_t r = recv[k];
        if (center_s.v[k] && r >= 0 && seed.v[r] && bl[k] > bl[r] + rise) bl[k] = bl[r] + rise;
    }
    for (size_t k = 0; k < N; ++k) bed[k] = seed.v[k] ? bl[k] : NaN;

    // 河谷剖面：最近河床格的 河床 / 水深 / 漫滩半宽 / 谷坡。谷壁（小流量）的坡按河床那格露出的岩性的坍塌角（B2；旧的一律 gorge_deg 32°）
    const double qg = c.get("hydro.q_gorge"), qw = c.get("hydro.q_wide");
    const double gorge0 = c.get("hydro.gorge_deg", 32.0), wide_deg = c.get("hydro.wide_deg");
    const double fpm = c.get("hydro.floodplain_mult"), fpmax = c.get("hydro.floodplain_max_m");
    std::vector<double> tan_side(N), fp_half(N);
    const double lq = std::log10(qw / qg);
    for (size_t k = 0; k < N; ++k) {
        const double t = clip(std::log10(std::max(Q[k], 1e-6) / qg) / lq, 0.0, 1.0);
        const double gorge = wall_deg ? wall_deg->v[k] : gorge0;
        const double side = center_r.v[k] ? (1 - t) * gorge + t * std::min(wide_deg, gorge) : gorge;
        tan_side[k] = std::tan(side * (PI / 180.0));
        fp_half[k] = center_r.v[k] ? std::min(fpm * width[k] / 2.0, fpmax / 2.0) : 0.0;
    }
    const int svc = c.geti("hydro.stream_valley_cells");
    const int reach_cells = any_r ? static_cast<int>(std::ceil(c.get("hydro.valley_max_km") * 1000.0 / res_m)) : svc;
    GridD dist;
    Grid<int64_t> src;
    nearest_propagate(seed, reach_cells, res_m, &work, dist, src);
    out.h_new = GridD(H, W);
    out.lvl = Grid<uint8_t>(H, W, 0);
    out.floodplain = Grid<uint8_t>(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        bool has = work.v[k] && src.v[k] >= 0;
        const size_t s = has ? static_cast<size_t>(src.v[k]) : 0;
        const double bed_s = bed[s], dep_s = depth[s], fp_s = fp_half[s], tan_s = tan_side[s], w_s = width[s];
        const bool isr_s = center_r.v[s] != 0;
        const uint8_t lvl_s = river_lvl.v[s];
        const double lim = isr_s ? INF : svc * res_m;
        has = has && dist.v[k] <= lim;
        const double valley = bed_s + dep_s + std::max(0.0, dist.v[k] - fp_s) * tan_s;
        double hn = has ? std::min(h.v[k], valley) : h.v[k];
        if (has && std::isnan(h.v[k])) hn = NaN;
        const bool wide = has && isr_s && !center_r.v[k] && dist.v[k] < w_s / 2.0;
        const bool chan = center_r.v[k] || wide;
        if (chan) hn = std::isnan(hn) || std::isnan(bed_s) ? NaN : std::min(hn, bed_s);
        if (seed.v[k]) hn = bed[k];
        out.h_new.v[k] = hn;
        out.lvl.v[k] = wide ? lvl_s : river_lvl.v[k];
        out.width.v[k] = chan ? (center_r.v[k] ? width[k] : w_s) : (center_s.v[k] ? width[k] : 0.0);
        out.depth.v[k] = chan ? (center_r.v[k] ? depth[k] : dep_s) : (center_s.v[k] ? depth[k] : 0.0);
        out.floodplain.v[k] = (has && isr_s && !chan && dist.v[k] <= fp_s + 0.5 * res_m && hn <= bed_s + dep_s + 0.5) ? 1 : 0;
    }
    // 最高格不动（IS-summit）
    size_t top = 0;
    double best = -INF;
    for (size_t k = 0; k < N; ++k) {
        const double v = mk.v[k] ? h.v[k] : -INF;
        if (v > best) {
            best = v;
            top = k;
        }
    }
    out.h_new.v[top] = h.v[top];
    for (size_t k = 0; k < N; ++k)
        if (!mk.v[k]) out.h_new.v[k] = h.v[k];

    // 摘要：每个河口（常年河流向虚空处）一条河
    if (any_r)
        for (size_t k = 0; k < N; ++k)
            if (center_r.v[k] && recv[k] < 0) {
                RiverRec rv;
                rv.mouth_r = static_cast<int>(k / W);
                rv.mouth_c = static_cast<int>(k % W);
                rv.basin_km2 = Akm.v[k];
                rv.length_km = Lkm[k] + step;
                rv.discharge = Q[k];
                rv.width = width[k];
                rv.depth = depth[k];
                rv.level = river_lvl.v[k];
                rv.waterfall = bed[k] - keel;
                rv.incision = base[k] - bed[k];
                out.rivers.push_back(rv);
            }
    for (size_t k = 0; k < N; ++k)
        if (center_s.v[k] && recv[k] < 0) ++out.n_stream_falls;
    double mc = -INF;
    for (size_t k = 0; k < N; ++k) mc = std::max(mc, mk.v[k] ? h.v[k] - out.h_new.v[k] : 0.0);
    out.has_cut = true;
    out.max_cut_m = pyround(mc, 1);
    GridD wgrid(H, W);
    wgrid.v = width;
    Grid<uint8_t> lv(H, W, 0);
    for (size_t k = 0; k < N; ++k) lv.v[k] = center_r.v[k] ? river_lvl.v[k] : 0;
    out.lines = trace_lines(seed, mk, recv, wgrid, lv, Akm);
    return out;
}

std::vector<std::vector<LinePt>> trace_lines(const Mask& seed, const Mask& mk, const std::vector<int64_t>& recv, const GridD& width,
                                             const Grid<uint8_t>& lvl, const GridD& acc) {
    const int H = seed.H, W = seed.W;
    const size_t N = seed.size();
    std::vector<int32_t> indeg(N, 0);
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k]) {
            const int64_t t = recv[k];
            if (t >= 0 && seed.v[t]) indeg[t]++;
        }
    std::vector<uint8_t> visited(N, 0);
    std::vector<std::vector<LinePt>> lines;
    static const int D8L[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (size_t s = 0; s < N; ++s) {
        if (!seed.v[s] || indeg[s] != 0) continue;
        std::vector<LinePt> pts;
        int64_t k = static_cast<int64_t>(s);
        for (;;) {
            const int i = static_cast<int>(k / W), j = static_cast<int>(k % W);
            pts.push_back({i + 0.5, j + 0.5, width.v[k], static_cast<int>(lvl.v[k]), acc.v[k]});
            if (visited[k] && pts.size() > 1) break;
            visited[k] = 1;
            const int64_t r = recv[k];
            if (r < 0) {
                for (const auto& d : D8L) {
                    const int a = i + d[0], b = j + d[1];
                    if (!(a >= 0 && a < H && b >= 0 && b < W) || !mk.v[static_cast<size_t>(a) * W + b]) {
                        const LinePt last = pts.back();
                        pts.push_back({i + 0.5 + 0.5 * d[0], j + 0.5 + 0.5 * d[1], last.w, last.lvl, last.acc});
                        break;
                    }
                }
                break;
            }
            if (!seed.v[r]) break;
            k = r;
        }
        if (pts.size() >= 2) lines.push_back(std::move(pts));
    }
    return lines;
}

}  // namespace skyisle::island
