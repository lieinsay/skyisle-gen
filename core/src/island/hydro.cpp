// 5.3 水系、地表与可耕地（hydro.py 的 build_hydro 同式）。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "skyisle/flow.hpp"
#include "skyisle/island/build.hpp"
#include "skyisle/island/farmland.hpp"
#include "skyisle/island/river.hpp"

namespace skyisle::island {

namespace {

enum { LC_VOID, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE };

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// 一座岛的水系结果（切片坐标），合并时按岛号次序写回群栅格
struct IslandHydro {
    bool present = false;
    int r0 = 0, c0 = 0, h = 0, w = 0;
    Mask mk;
    GridD h_final, hf, hr, Akm, cut, width, depth, pit, Q;
    Grid<uint8_t> river, stream, lake, fp;
    std::vector<int64_t> recv;
    int n_lakes = 0;
    int64_t lake_cells = 0;
    double max_acc = 0, river_thr = NaN;
    Channels ch;
    Basins basins;
    bool cap_ran = false;
    int captures = 0;
};

// 谷收拢（P4，L15；hydro.py 的 capture_rivers 同式）：先按现在的地形走一遍水、按出口分盆地；汇流最大的 n 个出口是大谷，
// 其余小沟沿干流从河口往上游找，头一个 reach 以内有大谷河道格比它低得多的格就是袭夺点，从它到那格切一道直沟（只压低地面）。
// h：局部切片（掩膜外 NaN），就地改；noise = 弯曲噪声 + 朝岸缘微倾（路由面用）。返回袭夺了几条
int capture_rivers(GridD& h, const Mask& mk, const GridD& noise, double res_m, double cell_km2, double reach_km, const Config& c) {
    const int Hh = h.H, Ww = h.W;
    const size_t N = h.size();
    const double eps = c.get("hydro.fill_eps_m");
    const GridD hf = priority_fill(h, mk, eps);
    GridD rin(Hh, Ww, NaN);
    for (size_t k = 0; k < N; ++k)
        if (mk.v[k]) rin.v[k] = hf.v[k] + noise.v[k];
    const GridD hr = priority_fill(rin, mk, eps);
    const FlowDir fd = d8(hr, mk, res_m);
    const std::vector<int64_t> recv = recv_flat(fd, Ww);
    GridD A = accumulate(mk, fd);
    for (double& v : A.v) v = v * cell_km2;
    std::vector<int64_t> lab(N, -1);
    for (int32_t k : downstream_first(recv, mk)) lab[k] = recv[k] < 0 ? k : lab[recv[k]];
    int64_t cnt = 0;
    std::vector<int32_t> outs;
    for (size_t k = 0; k < N; ++k)
        if (mk.v[k]) {
            ++cnt;
            if (recv[k] < 0) outs.push_back(static_cast<int32_t>(k));
        }
    const double total = static_cast<double>(cnt) * cell_km2;
    std::stable_sort(outs.begin(), outs.end(), [&](int32_t a, int32_t b) { return A.v[a] > A.v[b]; });
    const int n_v = static_cast<int>(clip(std::nearbyint(total / c.get("hydro.capture_km2_per_valley")), c.get("hydro.capture_min_n"), c.get("hydro.capture_max_n")));
    std::vector<uint8_t> is_win(N, 0);
    std::vector<double> budget(N, 0.0);
    bool any = false;
    const double win_min = c.get("hydro.capture_win_min_km2"), max_frac = c.get("hydro.capture_max_frac");
    for (int q = 0; q < n_v && q < static_cast<int>(outs.size()); ++q) {
        const int32_t o = outs[q];
        if (A.v[o] >= win_min) {
            is_win[o] = 1;
            budget[o] = max_frac * total - A.v[o];
            any = true;
        }
    }
    if (!any) return 0;
    const double chan = c.get("hydro.capture_chan_km2");
    std::vector<uint8_t> okw(N, 0);
    for (size_t k = 0; k < N; ++k) okw[k] = (mk.v[k] && lab[k] >= 0 && is_win[lab[k]] && A.v[k] >= chan) ? 1 : 0;
    std::vector<std::vector<int32_t>> donors(N);
    for (size_t k = 0; k < N; ++k)
        if (mk.v[k] && recv[k] >= 0) donors[recv[k]].push_back(static_cast<int32_t>(k));
    const double a_head = c.get("hydro.capture_head_km2"), s_min = c.get("hydro.capture_s_min"), margin = c.get("hydro.capture_margin_m");
    const double loser_min = c.get("hydro.capture_loser_min_km2");
    const int Rc = static_cast<int>(std::floor(reach_km * 1000.0 / res_m));
    const int64_t R2 = static_cast<int64_t>(Rc) * Rc;
    int n_cap = 0;
    struct Cand {
        int64_t d2;
        double hr;
        int32_t q;
    };
    std::vector<Cand> cand;
    for (int32_t o : outs) {
        if (is_win[o] || A.v[o] < loser_min) continue;
        std::vector<int32_t> path{o};
        int32_t k = o;
        while (!donors[k].empty()) {
            int32_t best = donors[k][0];
            for (size_t d = 1; d < donors[k].size(); ++d)
                if (A.v[donors[k][d]] > A.v[best]) best = donors[k][d];
            if (A.v[best] < a_head) break;
            path.push_back(best);
            k = best;
        }
        const int L = static_cast<int>(path.size()) - 1;
        if (L < 1) continue;
        bool done = false;
        for (int t = 1; t <= L && !done; ++t) {
            if (static_cast<double>(t) / L < s_min) continue;
            const int32_t b = path[t];
            const int bi = b / Ww, bj = b % Ww;
            const double z1 = hr.v[b] - 2.0 * (hr.v[b] - hr.v[path[t - 1]]) - 1.0;
            cand.clear();
            for (int i = std::max(0, bi - Rc); i < std::min(Hh, bi + Rc + 1); ++i)
                for (int j = std::max(0, bj - Rc); j < std::min(Ww, bj + Rc + 1); ++j) {
                    const int32_t q = i * Ww + j;
                    const int64_t d2 = static_cast<int64_t>(i - bi) * (i - bi) + static_cast<int64_t>(j - bj) * (j - bj);
                    if (okw[q] && d2 <= R2 && hr.v[q] <= z1 - margin) cand.push_back({d2, hr.v[q], q});
                }
            if (cand.empty()) continue;
            std::sort(cand.begin(), cand.end(), [](const Cand& x, const Cand& y) {
                if (x.d2 != y.d2) return x.d2 < y.d2;
                if (x.hr != y.hr) return x.hr < y.hr;
                return x.q < y.q;
            });
            for (const Cand& cd : cand) {
                const int32_t w = cd.q;
                const int64_t ow = lab[w];
                if (budget[ow] < A.v[b]) continue;
                const int wi = w / Ww, wj = w % Ww;
                const int n = std::max(std::abs(wi - bi), std::abs(wj - bj));
                std::vector<int32_t> cells;
                bool ok = true;
                for (int s = 1; s < n; ++s) {
                    const int ci = bi + static_cast<int>(std::nearbyint(static_cast<double>(s * (wi - bi)) / n));
                    const int cj = bj + static_cast<int>(std::nearbyint(static_cast<double>(s * (wj - bj)) / n));
                    const int32_t cc = ci * Ww + cj;
                    if (!mk.v[cc]) {
                        ok = false;
                        break;
                    }
                    cells.push_back(cc);
                }
                if (!ok) continue;
                const double zw = hr.v[w];
                for (int s = 1; s <= static_cast<int>(cells.size()); ++s) {
                    const int32_t cc = cells[s - 1];
                    const double z = z1 + (zw - z1) * (s - 1) / std::max(1, n - 1);
                    const double zz = z - noise.v[cc];
                    if (zz < h.v[cc]) h.v[cc] = zz;
                }
                budget[ow] -= A.v[b];
                ++n_cap;
                done = true;
                break;
            }
        }
    }
    return n_cap;
}

}  // namespace

void build_hydro(Group& g, const PlanetView& pv, const Config& c, int threads) {
    const double t0 = now_s();
    const NodeInputs& inp = g.inp;
    const double res_km = g.res_km, res_m = res_km * 1000.0, cell_km2 = res_km * res_km;
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    Mask land(H, W, 0);
    for (size_t k = 0; k < N; ++k) land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
    {
        const double lo = c.get("climate.precip_mm_min"), hi = c.get("climate.precip_mm_max"), e = c.get("climate.precip_mm_exp");
        g.P_mm = lo + (hi - lo) * c_pow(clip(inp.precip, 0.0, 1.0), e);
    }
    const double P_mm = g.P_mm;
    g.wind_u = grid_interp(pv.wind_u, pv.wind_grid, inp.lat, inp.lon);
    g.wind_v = grid_interp(pv.wind_v, pv.wind_grid, inp.lat, inp.lon);
    const double year_s = pv.year_s;

    // 汇流路由面：填平面 + 弯曲噪声 + 朝岸缘的微倾（只定流向）
    const double ox = g.origin_x, oy = g.origin_y;
    std::vector<double> Xk(W), Yk(H);
    for (int j = 0; j < W; ++j) Xk[j] = ox + (j + 0.5) * res_km;
    for (int i = 0; i < H; ++i) Yk[i] = oy - (i + 0.5) * res_km;
    GridD meander(H, W, 0.0), tilt(H, W, 0.0);
    {
        Rng rm = entity_rng(inp.seed, ISLAND_STREAM, "island:" + std::to_string(inp.node) + ":meander");
        FractalNoise fn(rm, Xk[0], Yk[H - 1], Xk[W - 1], Yk[0], c.get("hydro.meander_km"), 3, 0.5);
        const double mm = c.get("hydro.meander_m");
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) meander(i, j) = fn.sample(Xk[j], Yk[i]) * mm;
        Mask notland(H, W, 0);
        for (size_t k = 0; k < N; ++k) notland.v[k] = land.v[k] ? 0 : 1;
        const double tkm = c.get("hydro.edge_tilt_km"), tmk = c.get("hydro.edge_tilt_m_per_km");
        const GridI eb = distance_bands(notland, static_cast<int>(std::ceil(tkm / res_km)));
        for (size_t k = 0; k < N; ++k) tilt.v[k] = tmk * std::min(static_cast<double>(eb.v[k]) * res_km, tkm);
    }
    // 局地雨（P4，L11 的生成器那半；hydro.py 的 local_rain 同式）：本岛的起伏（高出本岛岸缘多少，用户 09-29 定：整座岛浮得高不逼气流抬升，
    // 只让它冷、不让它湿）× 山脉尺度的迎风坡，再按陆地均值归一——群的雨总量是行星层给的，山只把它重新分；关掉（两个系数都 0）时 rain ≡ P_mm，河宽按旧式
    const double oro = c.get("hydro.oro_rise_per_km", 0.0), wwgain = c.get("hydro.windward_gain", 0.0);
    const bool local_rain = oro != 0.0 || wwgain != 0.0;
    g.rain = GridD(H, W, 0.0);
    {
        for (size_t k = 0; k < N; ++k)
            if (land.v[k]) g.rain.v[k] = P_mm;
        if (local_rain) {
            const int f = std::max(1, static_cast<int>(std::nearbyint(c.get("hydro.windward_smooth_km") / res_km)));
            // 迎风坡也按本岛的起伏（高出本岛岸缘多少）量：浮得高的岛不算山。虚空按最近的陆地填（3 块以内，取那座岛的起伏；再远按 0 = 岸缘）：
            // 浮在空中的岛下面有风绕过去，岸崖不算迎风的「山」
            GridD nd;
            Grid<int64_t> ns;
            nearest_propagate(land, 3 * f, 1.0, nullptr, nd, ns);
            auto rel = [&](size_t q) { return g.height.v[q] - g.islands[g.island_id.v[q]].rim_j; };
            GridD hfill(H, W);
            for (size_t k = 0; k < N; ++k) hfill.v[k] = land.v[k] ? rel(k) : (ns.v[k] >= 0 ? rel(static_cast<size_t>(ns.v[k])) : 0.0);
            const GridD hc = block_mean(hfill, f);
            const int Hc = hc.H, Wc = hc.W;
            const double sp = py_hypot(g.wind_u, g.wind_v);
            GridD Gc(Hc, Wc, 0.0);
            const double step = f * res_km;
            for (int i = 0; i < Hc; ++i)
                for (int j = 0; j < Wc; ++j) {
                    const int jp = std::min(j + 1, Wc - 1), jm = std::max(j - 1, 0), ip = std::min(i + 1, Hc - 1), im = std::max(i - 1, 0);
                    const double gx = (hc(i, jp) - hc(i, jm)) / (std::max(jp - jm, 1) * step);   // 向东升高为正（m/km）
                    const double gy = (hc(im, j) - hc(ip, j)) / (std::max(ip - im, 1) * step);   // 向北升高为正（行向下 = 南）
                    Gc(i, j) = sp < 1e-6 ? 0.0 : (gx * g.wind_u + gy * g.wind_v) / sp;          // 顺风方向地势升高 = 迎风坡
                }
            const GridD G = upsample_bilinear(Gc, f, H, W);
            const double hi = c.get("hydro.oro_rise_max_km"), gref = c.get("hydro.windward_ref_m_per_km");
            std::vector<double> rl;
            for (size_t k = 0; k < N; ++k)
                if (land.v[k]) {
                    const double rim = g.islands[g.island_id.v[k]].rim_j;   // 本岛岸缘（island.json 的 rim_m）
                    const double r = (1.0 + oro * clip((g.height.v[k] - rim) / 1000.0, 0.0, hi)) * (1.0 + wwgain * clip(G.v[k] / gref, -1.0, 1.0));
                    g.rain.v[k] = r;
                    rl.push_back(r);
                }
            const double rmean = np_mean(rl);
            for (size_t k = 0; k < N; ++k)
                if (land.v[k]) g.rain.v[k] = P_mm * g.rain.v[k] / rmean;
        }
    }
    const double rq = 16.0;   // 雨量加权汇流的量子（1/16 mm）：权重取整后求和与次序无关，两个后端逐位相同
    const int n_isl = static_cast<int>(g.islands.size());
    // 各岛的外框
    std::vector<int> rlo(n_isl, H), rhi(n_isl, -1), clo(n_isl, W), chi(n_isl, -1);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const int k = g.island_id(i, j);
            if (k < 0) continue;
            rlo[k] = std::min(rlo[k], i);
            rhi[k] = std::max(rhi[k], i);
            clo[k] = std::min(clo[k], j);
            chi[k] = std::max(chi[k], j);
        }
    const double eps = c.get("hydro.fill_eps_m"), lake_depth = c.get("hydro.lake_min_depth_m"), lake_km2 = c.get("hydro.lake_min_km2");
    const double pit_keep = c.get("hydro.pit_keep_m"), stream_min = c.get("hydro.stream_min_km2");
    std::vector<IslandHydro> res(n_isl);
    parallel_for(n_isl, threads, [&](int k) {
        if (rhi[k] < 0) return;
        IslandHydro& R = res[k];
        R.present = true;
        R.r0 = rlo[k];
        R.c0 = clo[k];
        R.h = rhi[k] + 1 - rlo[k];
        R.w = chi[k] + 1 - clo[k];
        const int h = R.h, w = R.w;
        const size_t n = static_cast<size_t>(h) * w;
        R.mk = Mask(h, w, 0);
        GridD hh(h, w, NaN);
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                if (g.island_id(i + R.r0, j + R.c0) == k) {
                    R.mk(i, j) = 1;
                    hh(i, j) = g.height(i + R.r0, j + R.c0);
                }
        const Mask& mk = R.mk;
        // 谷收拢（P4）：山在中间的岛（多核岛有自己的干流，不收），新岛 × capture_young
        {
            const IslandRec& Ji = g.islands[k];
            const double reach = c.get("hydro.capture_reach_km", 0.0) * (Ji.kind == YOUNG ? c.get("hydro.capture_young", 0.0) : 1.0);
            int64_t mcount = 0;
            for (uint8_t x : mk.v) mcount += x;
            if (reach > 0.0 && Ji.cores.empty() && static_cast<double>(mcount) * cell_km2 >= c.get("hydro.capture_min_island_km2")) {
                GridD nz(h, w, 0.0);
                for (int i = 0; i < h; ++i)
                    for (int j = 0; j < w; ++j)
                        if (mk(i, j)) nz(i, j) = meander(i + R.r0, j + R.c0) + tilt(i + R.r0, j + R.c0);
                R.captures = capture_rivers(hh, mk, nz, res_m, cell_km2, reach, c);
                R.cap_ran = true;
            }
        }
        R.hf = priority_fill(hh, mk, eps);
        // 湖：填平深 ≥ lake_min_depth 且面积 ≥ lake_min_km2 的洼地里最大的几个；其余洼地抬到填平面以下 pit_keep
        Mask pond(h, w, 0);
        bool any_pond = false;
        for (size_t q = 0; q < n; ++q) {
            const double depth = mk.v[q] ? R.hf.v[q] - hh.v[q] : 0.0;
            pond.v[q] = depth >= lake_depth ? 1 : 0;
            any_pond = any_pond || pond.v[q];
        }
        R.lake = Grid<uint8_t>(h, w, 0);
        if (any_pond) {
            GridI lab;
            const int nl = label_components(pond, 4, lab);
            if (nl) {
                std::vector<int64_t> cnt(nl + 1, 0);
                for (int32_t x : lab.v) cnt[x]++;
                std::vector<int> ids(nl);
                for (int q = 0; q < nl; ++q) ids[q] = q;
                std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return cnt[a + 1] > cnt[b + 1]; });
                std::vector<int> big;
                for (int q : ids)
                    if (cnt[q + 1] * cell_km2 >= lake_km2) big.push_back(q + 1);
                const size_t cap = static_cast<size_t>(c.get(k == 0 ? "hydro.lake_max_per_island" : "hydro.lake_max_per_islet"));
                if (big.size() > cap) big.resize(cap);
                std::vector<uint8_t> isbig(nl + 1, 0);
                for (int b : big) isbig[b] = 1;
                for (size_t q = 0; q < n; ++q) R.lake.v[q] = isbig[lab.v[q]];
                R.n_lakes = static_cast<int>(big.size());
            }
        }
        for (size_t q = 0; q < n; ++q)
            if (mk.v[q] && !R.lake.v[q] && R.hf.v[q] - hh.v[q] > pit_keep) hh.v[q] = R.hf.v[q] - pit_keep;
        // 洼（抬过之后还剩的、不是湖的）：湿地的「排不走」（P4，L14）
        R.pit = GridD(h, w, 0.0);
        for (size_t q = 0; q < n; ++q)
            if (mk.v[q] && !R.lake.v[q]) R.pit.v[q] = std::max(R.hf.v[q] - hh.v[q], 0.0);
        GridD rin(h, w, NaN);
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                if (mk(i, j)) rin(i, j) = R.hf(i, j) + meander(i + R.r0, j + R.c0) + tilt(i + R.r0, j + R.c0);
        R.hr = priority_fill(rin, mk, eps);
        const FlowDir fd = d8(R.hr, mk, res_m);
        R.recv = recv_flat(fd, w);
        const GridD A = accumulate(mk, fd);
        R.Akm = GridD(h, w, 0.0);
        double amax = -INF;
        for (size_t q = 0; q < n; ++q) {
            R.Akm.v[q] = A.v[q] * cell_km2;
            if (mk.v[q]) amax = std::max(amax, R.Akm.v[q]);
        }
        R.max_acc = amax;
        // 局地雨：年均流量按上游各格的雨（取整到 1/rq mm 的权重求和，次序无关）
        if (local_rain) {
            GridD wt(h, w, 0.0);
            for (int i = 0; i < h; ++i)
                for (int j = 0; j < w; ++j)
                    if (mk(i, j)) wt(i, j) = std::nearbyint(g.rain(i + R.r0, j + R.c0) * rq);
            const GridD Aw = accumulate(mk, fd, &wt);
            const double runoff = c.get("hydro.runoff_coef");
            R.Q = GridD(h, w, 0.0);
            for (size_t q = 0; q < n; ++q)
                if (mk.v[q]) R.Q.v[q] = Aw.v[q] * (cell_km2 / rq) * 1000.0 * runoff / year_s;
        }
        // 湖面：高程抬到填平面
        for (size_t q = 0; q < n; ++q) {
            if (R.lake.v[q]) {
                hh.v[q] = R.hf.v[q];
                R.lake_cells++;
            }
        }
        R.river = Grid<uint8_t>(h, w, 0);
        R.stream = Grid<uint8_t>(h, w, 0);
        if (k == 0) {
            if (inp.has_river) {
                const double runoff = c.get("hydro.runoff_coef");
                const double q_area = c.get("hydro.river_min_q_m3s") * year_s / std::max(1e-9, P_mm / 1000.0 * runoff) / 1e6;
                double amx = -INF;
                for (size_t q = 0; q < n; ++q) amx = std::max(amx, R.Akm.v[q]);
                const double thr = std::min(q_area, c.get("hydro.river_reach_frac") * amx);
                R.river_thr = thr;
                // 局地雨：常年河直接按流量判（≥ river_min_q_m3s，或主岛最大流量的 river_reach_frac）
                double qthr = 0.0;
                if (local_rain) {
                    double qmx = -INF;
                    for (size_t q = 0; q < n; ++q)
                        if (mk.v[q]) qmx = std::max(qmx, R.Q.v[q]);
                    qthr = std::min(c.get("hydro.river_min_q_m3s"), c.get("hydro.river_reach_frac") * qmx);
                }
                const double rs = inp.river_size;
                const int top = rs >= c.get("hydro.river_big_size") ? 3 : (rs >= c.get("hydro.river_mid_size") ? 2 : 1);
                for (size_t q = 0; q < n; ++q) {
                    const bool per = local_rain ? (mk.v[q] && R.Q.v[q] >= qthr) : R.Akm.v[q] >= thr;
                    uint8_t lvl = per ? 1 : 0;
                    if (top >= 2 && per && R.Akm.v[q] >= 0.3 * amx) lvl = 2;
                    if (top >= 3 && per && R.Akm.v[q] >= 0.6 * amx) lvl = 3;
                    if (mk.v[q]) R.river.v[q] = lvl;
                    if (mk.v[q] && R.Akm.v[q] >= stream_min && !per) R.stream.v[q] = 1;
                }
            } else {
                for (size_t q = 0; q < n; ++q)
                    if (mk.v[q] && R.Akm.v[q] >= stream_min) R.stream.v[q] = 1;
            }
            // 集水盆地（原始数；排序与取前几个在前端）
            double total = 0;
            int64_t cells = 0;
            for (size_t q = 0; q < n; ++q) cells += mk.v[q];
            total = static_cast<double>(cells) * cell_km2;
            const double thr = std::max(c.get("hydro.basin_mouth_km2"), c.get("hydro.basin_mouth_frac") * total);
            R.basins.present = true;
            R.basins.total_km2 = total;
            R.basins.thr_km2 = thr;
            for (size_t q = 0; q < n; ++q)
                if (mk.v[q] && R.Akm.v[q] >= thr && R.recv[q] < 0)
                    R.basins.mouths.push_back({static_cast<double>(q / w + R.r0), static_cast<double>(q % w + R.c0), R.Akm.v[q]});
        } else {
            for (size_t q = 0; q < n; ++q)
                if (mk.v[q] && R.Akm.v[q] >= stream_min) R.stream.v[q] = 1;
        }
        // 河道成形
        const IslandRec& J = g.islands[k];
        R.ch = carve_channels(hh, R.hr, mk, R.lake, R.recv, R.Akm, R.river, R.stream, P_mm, J.rim_j, J.keel_j, res_m, year_s, c, k == 0,
                              local_rain ? &R.Q : nullptr);
        R.cut = GridD(h, w, 0.0);
        R.h_final = GridD(h, w, NaN);
        for (size_t q = 0; q < n; ++q) {
            if (!mk.v[q]) continue;
            const double d = hh.v[q] - R.ch.h_new.v[q];
            R.cut.v[q] = std::isnan(d) ? 0.0 : d;
            R.h_final.v[q] = R.ch.h_new.v[q];
            R.river.v[q] = R.ch.lvl.v[q];
            if (R.ch.lvl.v[q] > 0) R.stream.v[q] = 0;
        }
    });

    // 写回群栅格
    g.filled = GridD(H, W, NaN);
    g.acc_km2 = GridD(H, W, 0.0);
    g.lake = Grid<uint8_t>(H, W, 0);
    g.river = Grid<uint8_t>(H, W, 0);
    g.stream = Grid<uint8_t>(H, W, 0);
    g.width_m = GridD(H, W, 0.0);
    g.depth_m = GridD(H, W, 0.0);
    g.floodplain = Grid<uint8_t>(H, W, 0);
    g.cut_m = GridD(H, W, 0.0);
    g.recv_i = GridI(H, W, -1);
    g.recv_j = GridI(H, W, -1);
    g.route_h = GridD(H, W, NaN);
    GridD pit(H, W, 0.0);
    g.lines.clear();
    g.rivers.clear();
    g.n_falls = 0;
    g.max_cut = 0.0;
    g.river_thr = NaN;
    for (int k = 0; k < n_isl; ++k) {
        IslandHydro& R = res[k];
        if (!R.present) continue;
        for (int i = 0; i < R.h; ++i)
            for (int j = 0; j < R.w; ++j) {
                if (!R.mk(i, j)) continue;
                const int gi = i + R.r0, gj = j + R.c0;
                const size_t q = static_cast<size_t>(i) * R.w + j;
                const int64_t r = R.recv[q];
                if (r >= 0) {
                    g.recv_i(gi, gj) = static_cast<int32_t>(r / R.w + R.r0);
                    g.recv_j(gi, gj) = static_cast<int32_t>(r % R.w + R.c0);
                }
                g.route_h(gi, gj) = R.hr.v[q];
                pit(gi, gj) = R.pit.v[q];
                g.filled(gi, gj) = R.hf.v[q];
                g.acc_km2(gi, gj) = R.Akm.v[q];
                if (R.lake.v[q]) g.lake(gi, gj) = 1;
                g.cut_m(gi, gj) = R.cut.v[q];
                g.height(gi, gj) = R.h_final.v[q];
                g.river(gi, gj) = R.river.v[q];
                g.stream(gi, gj) = R.stream.v[q];
                g.width_m(gi, gj) = R.ch.width.v[q];
                g.depth_m(gi, gj) = R.ch.depth.v[q];
                if (R.ch.floodplain.v[q]) g.floodplain(gi, gj) = 1;
            }
        g.n_falls += R.ch.n_stream_falls + static_cast<int>(R.ch.rivers.size());
        if (R.ch.has_cut) g.max_cut = std::max(g.max_cut, R.ch.max_cut_m);
        for (const auto& L : R.ch.lines) {
            RiverLine rl;
            rl.island = k;
            for (const LinePt& p : L) rl.pts.push_back({p.r + R.r0, p.c + R.c0, p.w, p.lvl, p.acc});
            g.lines.push_back(std::move(rl));
        }
        if (k == 0) {
            g.rivers = R.ch.rivers;
            for (RiverRec& rv : g.rivers) {
                rv.mouth_r += R.r0;
                rv.mouth_c += R.c0;
            }
            g.basins = R.basins;
            g.river_thr = R.river_thr;
        }
        IslandRec& J = g.islands[k];
        J.hydro = true;
        J.n_lakes = R.n_lakes;
        J.lake_cells = R.lake_cells;
        J.max_flowacc = R.max_acc;
        J.has_perennial = false;
        J.has_stream = false;
        J.cap_ran = R.cap_ran;
        J.captures = R.captures;
        for (size_t q = 0; q < R.mk.size(); ++q)
            if (R.mk.v[q]) {
                J.has_perennial = J.has_perennial || R.river.v[q] > 0;
                J.has_stream = J.has_stream || R.stream.v[q] > 0;
            }
    }
    for (size_t k = 0; k < N; ++k)
        if (g.lake.v[k]) {
            g.river.v[k] = 0;
            g.stream.v[k] = 0;
        }
    // 台面校正：河道下切 / 河谷压低了主岛的一圈格子，主岛整体抬回（平移不改坡度、河床单调与湖面）
    {
        std::vector<double> hm;
        for (size_t k = 0; k < N; ++k)
            if (g.island_id.v[k] == 0) hm.push_back(g.height.v[k]);
        g.dz = inp.height_m - np_median(hm);
        for (size_t k = 0; k < N; ++k)
            if (g.island_id.v[k] == 0) {
                g.height.v[k] += g.dz;
                g.filled.v[k] += g.dz;
            }
    }

    // ---------- 地表分类 ----------
    GridD hl(H, W, NaN);
    for (size_t k = 0; k < N; ++k)
        if (land.v[k]) hl.v[k] = g.height.v[k];
    g.slope = slope_deg(hl, land, res_m);
    const GridD& slope = g.slope;
    const double T_sea = inp.temp_sea, lapse = inp.lapse_c_per_km;
    std::vector<double> T(N), expo(N, 0.0), soil(N), wet(N), near_water(N);
    const double sp = py_hypot(g.wind_u, g.wind_v);
    const double wux = sp < 1e-6 ? 0.0 : -g.wind_u / sp, wuy = sp < 1e-6 ? 0.0 : -g.wind_v / sp;
    std::vector<double> age_arr(N, 0.0);
    for (size_t k = 0; k < N; ++k)
        if (g.island_id.v[k] >= 0) age_arr[k] = g.islands[g.island_id.v[k]].age_j;
    const double szero = c.get("landcover.soil_slope_zero_deg"), wgain = c.get("landcover.aspect_wet_gain");
    const int near_cells = c.geti("landcover.water_near_cells");
    Mask water(H, W, 0);
    for (size_t k = 0; k < N; ++k) water.v[k] = (g.river.v[k] > 0 || g.lake.v[k]) ? 1 : 0;
    const GridI dw = distance_bands(water, near_cells);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            T[k] = T_sea - lapse * (land.v[k] ? g.height.v[k] : 0.0) / 1000.0;
            if (sp >= 1e-6) {
                auto nb = [&](int di, int dj) {
                    const double self = hl.v[k];
                    const int a = i - di, b = j - dj;
                    if (a < 0 || b < 0 || a >= H || b >= W) return self;
                    const double v = hl(a, b);
                    return std::isnan(v) ? self : v;
                };
                const double gx = (nb(0, -1) - nb(0, 1)) / (2 * res_m);
                const double gy = (nb(-1, 0) - nb(1, 0)) / (2 * res_m);
                const double gn = np_hypot(gx, gy);
                const double dx = gn > 1e-9 ? gx / std::max(gn, 1e-9) : 0.0;
                const double dy = gn > 1e-9 ? gy / std::max(gn, 1e-9) : 0.0;
                const double ex = clip(dx * wux + dy * wuy, -1.0, 1.0);
                expo[k] = ex * clip(slope.v[k] / 15.0, 0.0, 1.0);
            }
            soil[k] = clip(0.35 + 0.5 * age_arr[k], 0, 1) * clip(1.0 - slope.v[k] / szero, 0.0, 1.0) *
                      (0.7 + 0.3 * clip(std::log1p(g.acc_km2.v[k]) / 4.0, 0, 1));
            wet[k] = clip(g.rain.v[k] / 1500.0, 0.2, 2.0) * (1.0 + wgain * expo[k]);   // 局地雨（关掉时 = P_mm）
            near_water[k] = std::sqrt(clip(1.0 - dw.v[k] / (near_cells + 1.0), 0.0, 1.0));
        }
    const double alpine = c.get("landcover.alpine_temp_c"), rock_t = c.get("landcover.rock_temp_c"), rock_s = c.get("landcover.rock_slope_deg");
    const double f_tmin = c.get("landcover.forest_temp_min_c"), f_wet = c.get("landcover.forest_wet_min"), f_soil = c.get("landcover.forest_soil_min");
    const double s_wet = c.get("landcover.shrub_wet_min");
    // 湿地（P4，L14）：坡缓、离崖缘远、汇来的水排不走（台面上的洼、平地），再乘局地雨——不拿年降水一刀切。
    // 指数 = 汇流（方窗内最大，平地上的水摊开）/ 格宽 × (雨 / wet_ref_mm)^wet_rain_exp × 离崖缘的斜坡 / tan(周围最陡的坡) × (1 + 洼的加成)，
    // ≥ wet_index_min 且周围最陡的坡（wet_flat_cells 格方窗）< wet_slope_max_deg 为湿地：成片的平地，不是一条条沟底
    const double w_smax = c.get("landcover.wet_slope_max_deg"), w_sfloor = c.get("landcover.wet_slope_floor_deg");
    const double w_e0 = c.get("landcover.wet_edge_near_km"), w_e1 = c.get("landcover.wet_edge_far_km"), w_ref = c.get("landcover.wet_ref_mm");
    const double w_pg = c.get("landcover.wet_pit_gain"), w_pref = c.get("landcover.wet_pit_ref_m"), w_imin = c.get("landcover.wet_index_min");
    const double w_rexp = c.get("landcover.wet_rain_exp");
    std::vector<double> wet_idx(N, 0.0);
    GridD snb, slo;
    window_extrema(slope, c.geti("landcover.wet_flat_cells"), land, snb, slo);
    {
        Mask notland(H, W, 0);
        for (size_t k = 0; k < N; ++k) notland.v[k] = land.v[k] ? 0 : 1;
        const GridI eb = distance_bands(notland, static_cast<int>(std::ceil(w_e1 / res_km)));
        GridD ahi, alo;
        window_extrema(g.acc_km2, c.geti("landcover.wet_spread_cells"), land, ahi, alo);
        for (size_t k = 0; k < N; ++k) {
            if (!land.v[k]) continue;
            const double edge = clip((static_cast<double>(eb.v[k]) * res_km - w_e0) / (w_e1 - w_e0), 0.0, 1.0);
            const double tanb = std::tan(std::max(snb.v[k], w_sfloor) * (PI / 180.0));
            wet_idx[k] = ahi.v[k] * 1e6 / res_m * np_pow(g.rain.v[k] / w_ref, w_rexp) * edge / tanb * (1.0 + w_pg * clip(pit.v[k] / w_pref, 0.0, 1.0));
        }
    }
    g.landcover = Grid<uint8_t>(H, W, LC_VOID);
    std::vector<uint8_t> wetland(N, 0);
    for (size_t k = 0; k < N; ++k) {
        if (!land.v[k]) continue;
        uint8_t cv = LC_GRASS;
        if (T[k] < alpine) cv = LC_ALPINE;
        if (T[k] < rock_t) cv = LC_ROCK;
        if (slope.v[k] >= rock_s) cv = LC_ROCK;
        const bool warm = T[k] >= f_tmin && slope.v[k] < rock_s;
        const bool forest = warm && wet[k] >= f_wet && soil[k] >= f_soil;
        const bool shrub = T[k] >= alpine && slope.v[k] < rock_s && !forest && wet[k] >= s_wet && soil[k] >= 0.5 * f_soil;
        if (shrub) cv = LC_SHRUB;
        if (forest) cv = LC_FOREST;
        wetland[k] = (snb.v[k] < w_smax && wet_idx[k] >= w_imin) ? 1 : 0;
        if (wetland[k]) cv = LC_WET;
        if (g.cliff.v[k]) cv = LC_CLIFF;
        g.landcover.v[k] = cv;
    }
    // ---------- 可耕地：适宜度分位 ----------
    std::vector<double> suit(N, -1.0);
    {
        Rng ra = entity_rng(inp.seed, ISLAND_STREAM, "island:" + std::to_string(inp.node) + ":arable");
        FractalNoise fn(ra, Xk[0], Yk[H - 1], Xk[W - 1], Yk[0], c.get("landcover.arable_patch_km"), 3, 0.5);
        const double pamp = c.get("landcover.arable_patch_amp"), smax = c.get("landcover.arable_slope_max_deg"), tmin = c.get("landcover.arable_temp_min_c");
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                const size_t k = static_cast<size_t>(i) * W + j;
                const double patch = 1.0 + pamp * fn.sample(Xk[j], Yk[i]);
                const double s = patch * (0.02 + np_pow(clip(1.0 - slope.v[k] / smax, 0.0, 1.0), 1.5) * (0.1 + 0.9 * clip((T[k] - tmin) / 8.0, 0.0, 1.0)) *
                                                     (0.4 + 0.6 * soil[k]) * (0.8 + 0.2 * near_water[k]) * clip(wet[k] / 0.6, 0.2, 1.2));
                if (land.v[k] && !g.cliff.v[k] && !g.lake.v[k] && g.river.v[k] == 0 && !wetland[k]) suit[k] = s;
            }
    }
    int64_t n_land = 0;
    for (uint8_t x : land.v) n_land += x;
    const int64_t n_arable = static_cast<int64_t>(std::nearbyint(inp.arable_frac * static_cast<double>(n_land)));
    g.arable = Grid<uint8_t>(H, W, 0);
    if (n_arable > 0) {
        std::vector<int32_t> idx(N);
        for (size_t k = 0; k < N; ++k) idx[k] = static_cast<int32_t>(k);
        const size_t take = std::min<size_t>(static_cast<size_t>(n_arable), N);
        auto cmp = [&](int32_t a, int32_t b) { return suit[a] > suit[b] || (suit[a] == suit[b] && a < b); };
        std::nth_element(idx.begin(), idx.begin() + (take - 1), idx.end(), cmp);
        const double terr = c.get("landcover.terrace_slope_deg");
        // nth_element 之后前 take 个就是降序的前 take 名（集合相同，次序无关）
        for (size_t q = 0; q < take; ++q) {
            const int32_t k = idx[q];
            if (suit[k] > 0) g.arable.v[k] = slope.v[k] >= terr ? 2 : 1;
        }
    }
    // P5：上面按行星层额度取的是「上等地」（资源层照旧避开它）；宜垦另记，已垦在聚落层定（farmland.cpp）
    cultivable_land(g, c, suit, T, soil, wet, near_water);
    for (size_t k = 0; k < N; ++k) {
        if (g.arable.v[k] == 1) g.landcover.v[k] = LC_ARABLE;
        if (g.arable.v[k] == 2) g.landcover.v[k] = LC_TERRACE;
        if (g.river.v[k] > 0) g.landcover.v[k] = LC_RIVER;
        if (g.lake.v[k]) g.landcover.v[k] = LC_LAKE;
    }
    // 出图 / 下游用的最终形：河宽 / 水深只留陆地，漫滩去掉河湖，下切量不为负
    for (size_t k = 0; k < N; ++k) {
        if (!land.v[k]) {
            g.width_m.v[k] = 0;
            g.depth_m.v[k] = 0;
            g.cut_m.v[k] = 0;
        } else {
            g.cut_m.v[k] = std::max(g.cut_m.v[k], 0.0);
        }
        g.floodplain.v[k] = (g.floodplain.v[k] && land.v[k] && g.river.v[k] == 0 && !g.lake.v[k]) ? 1 : 0;
    }
    g.has_hydro = true;
    g.sec_hydro = now_s() - t0;
}

}  // namespace skyisle::island
