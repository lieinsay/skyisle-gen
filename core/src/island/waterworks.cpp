// 水利（P6；skyisle_gen/island/waterworks.py 的 polder_plan / build_waterworks 逐位同式）。
#include "skyisle/island/waterworks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <string>

namespace skyisle::island {

namespace {
constexpr int LC_WET = 9;
struct Step {
    int di, dj;
    double L;
};
const Step STEPS[8] = {{-1, 0, 1.0}, {1, 0, 1.0}, {0, -1, 1.0}, {0, 1, 1.0}, {-1, -1, SQRT2}, {-1, 1, SQRT2}, {1, -1, SQRT2}, {1, 1, SQRT2}};
const int N4W[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

int64_t cells_min(double km2, double cell_km2) { return static_cast<int64_t>(std::ceil(km2 / cell_km2 - 1e-9)); }
int floordiv(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
inline float f32(double x) { return static_cast<float>(x); }

// 格线上连续有圩、管它的村不变的段 [起, 止, 村号]（止不含）；owners 里 −1 = 两旁都不是这片的圩，0 = 圩没有管它的村（waterworks._runs_owner）
std::vector<std::array<int, 3>> runs_owner(const std::vector<int>& ow) {
    std::vector<std::array<int, 3>> out;
    int a = -1;
    for (int i = 0; i < static_cast<int>(ow.size()); ++i) {
        if (a >= 0 && ow[i] != ow[a]) {
            out.push_back({a, i, ow[a]});
            a = -1;
        }
        if (ow[i] >= 0 && a < 0) a = i;
    }
    if (a >= 0) out.push_back({a, static_cast<int>(ow.size()), ow[a]});
    return out;
}
}  // namespace

PolderPlan polder_plan(const Group& g, const Config& c, const Mask& wet, const std::vector<uint8_t>& take1, int64_t n_avail,
                       const std::vector<uint8_t>& pit) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, cell_km2 = res_km * res_km;
    auto wc = [&](const std::string& k) { return c.get("works." + k); };
    PolderPlan out;
    out.polder_id = GridI(H, W, 0);
    for (size_t k = 0; k < N; ++k) out.wetland_cells += wet.v[k] ? 1 : 0;
    GridI lab;
    const int n = label_by_island(wet, g.island_id, 8, lab);
    if (n == 0) return out;
    std::vector<std::vector<int32_t>> cells_of(static_cast<size_t>(n) + 1);
    for (size_t k = 0; k < N; ++k)
        if (lab.v[k]) cells_of[lab.v[k]].push_back(static_cast<int32_t>(k));
    const int R = static_cast<int>(std::ceil(wc("polder_ring_km") / res_km - 1e-9));
    const int64_t min_patch = cells_min(wc("polder_patch_min_km2"), cell_km2);
    const int64_t min_block = cells_min(wc("polder_block_min_km2"), cell_km2);
    const int s = std::max(1, static_cast<int>(std::nearbyint(wc("polder_spacing_km") / res_km)));
    const double p_min = wc("polder_pressure_min");
    const float flat_deg = f32(wc("polder_flat_deg"));
    struct Cand {
        double p;
        int L;
    };
    std::vector<Cand> cand;
    for (int L = 1; L <= n; ++L) {
        const auto& cells = cells_of[L];
        if (static_cast<int64_t>(cells.size()) < min_patch) continue;
        int imin = H, imax = -1, jmin = W, jmax = -1;
        for (int32_t q : cells) {
            imin = std::min(imin, q / W);
            imax = std::max(imax, q / W);
            jmin = std::min(jmin, q % W);
            jmax = std::max(jmax, q % W);
        }
        const int k = g.island_id.v[cells[0]];
        const int r0 = std::max(0, imin - R - 1), r1 = std::min(H, imax + R + 2);
        const int c0 = std::max(0, jmin - R - 1), c1 = std::min(W, jmax + R + 2);
        const int h = r1 - r0, w = c1 - c0;
        Mask pm(h, w, 0);
        for (int32_t q : cells) pm(q / W - r0, q % W - c0) = 1;
        const GridI d = distance_bands(pm, R);
        int64_t nF = 0, nT = 0;
        for (int a = 0; a < h; ++a)
            for (int b = 0; b < w; ++b) {
                const int dv = d(a, b);
                if (dv < 1 || dv > R) continue;
                const size_t gk = static_cast<size_t>(a + r0) * W + (b + c0);
                if (g.island_id.v[gk] != k || !(g.cultivable.v[gk] > 0 && f32(g.slope.v[gk]) < flat_deg)) continue;
                ++nF;
                nT += take1[gk] ? 1 : 0;
            }
        if (nF == 0) continue;
        const double p = static_cast<double>(nT) / static_cast<double>(nF);
        if (p >= p_min) cand.push_back({p, L});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
        if (a.p != b.p) return a.p > b.p;
        return a.L < b.L;
    });
    int64_t used = 0;
    int npol = 0;
    for (const Cand& cd : cand) {
        const auto& cells = cells_of[cd.L];
        int32_t o = cells[0];
        float best = f32(g.acc_km2.v[o]);
        for (int32_t q : cells) {
            const float a = f32(g.acc_km2.v[q]);
            if (a > best) {
                best = a;
                o = q;
            }
        }
        const int io = o / W, jo = o % W;
        struct BC {
            int bi, bj;
            int32_t q;
        };
        std::vector<BC> bcs;
        bcs.reserve(cells.size());
        for (int32_t q : cells)
            if (!pit[q]) bcs.push_back({floordiv(q / W - io, s), floordiv(q % W - jo, s), q});      // 采场那格不围进圩
        std::sort(bcs.begin(), bcs.end(), [](const BC& a, const BC& b) {
            if (a.bi != b.bi) return a.bi < b.bi;
            if (a.bj != b.bj) return a.bj < b.bj;
            return a.q < b.q;
        });
        PolderPatch P;
        P.label = cd.L;
        P.island = g.island_id.v[cells[0]];
        P.outlet = o;
        P.io = io;
        P.jo = jo;
        P.spacing = s;
        P.pressure = cd.p;
        P.wet_cells = static_cast<int64_t>(cells.size());
        int64_t tot = 0;
        for (size_t a = 0; a < bcs.size();) {
            size_t b = a;
            while (b < bcs.size() && bcs[b].bi == bcs[a].bi && bcs[b].bj == bcs[a].bj) ++b;
            if (static_cast<int64_t>(b - a) >= min_block) {
                PolderBlock B;
                B.bi = bcs[a].bi;
                B.bj = bcs[a].bj;
                for (size_t t = a; t < b; ++t) B.cells.push_back(bcs[t].q);
                tot += static_cast<int64_t>(B.cells.size());
                P.blocks.push_back(std::move(B));
            }
            a = b;
        }
        if (P.blocks.empty() || used + tot > n_avail) continue;
        used += tot;
        for (const PolderBlock& B : P.blocks) {
            ++npol;
            for (int32_t q : B.cells) {
                out.polder_id.v[q] = npol;
                out.cells.push_back(q);
            }
            P.ids.push_back(npol);
        }
        out.patches.push_back(std::move(P));
    }
    return out;
}

Json build_waterworks(const Group& g, const Config& c, const std::vector<WorksField>& fields, const GridI& fields_raster,
                      const std::vector<WorksVillage>& villages, Grid<uint8_t>& sraster, const PolderPlan& plan,
                      const std::vector<WorksRuin>& ruins, std::vector<int32_t>& cmd_cells) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, res_m = res_km * 1000.0, cell_km2 = res_km * res_km;
    auto wc = [&](const std::string& k) { return c.get("works." + k); };
    const double x0 = g.origin_x, y0 = g.origin_y;
    auto km = [&](double i, double j) { return Json::pair(pyround(x0 + (j + 0.5) * res_km, 3), pyround(y0 - (i + 0.5) * res_km, 3)); };
    auto cellpt = [&](int32_t q) { return Json::pair(static_cast<double>(q / W) + 0.5, static_cast<double>(q % W) + 0.5); };
    auto land = [&](size_t k) { return g.island_id.v[k] >= 0; };
    const int nf = static_cast<int>(fields.size());
    std::vector<int64_t> f_cells(static_cast<size_t>(nf) + 1, 0);
    for (int f = 0; f < nf; ++f) f_cells[f + 1] = fields[f].cells;
    std::vector<uint8_t> pol_field(static_cast<size_t>(nf) + 1, 0);
    for (size_t k = 0; k < N; ++k)
        if (plan.polder_id.v[k] > 0 && fields_raster.v[k] > 0) pol_field[fields_raster.v[k]] = 1;
    std::vector<double> zf(N, 0.0);
    for (size_t k = 0; k < N; ++k) zf[k] = (land(k) ? g.height.v[k] : 0.0) + (land(k) ? static_cast<double>(f32(g.depth_m.v[k])) : 0.0);
    // 管水利的村：田 → 种它的村（挂在村上的圩田也算）；走得到 = 离村格心的整数平方 ≤ W2
    double t = wc("manage_walk_km") / res_km;
    const double W2 = t * t;
    const int wr = static_cast<int>(std::ceil(t - 1e-9));
    std::vector<int> fvill(static_cast<size_t>(nf) + 1, 0);
    for (const WorksVillage& v : villages) {
        fvill[v.field] = v.id;
        for (int x : v.polder_fields) fvill[x] = v.id;
    }
    auto vjson = [](int vid) { return vid ? Json(vid) : Json(); };

    // ---------- 渠首与渠（一村一堰）
    const float src_min = f32(wc("works_src_min_km2")), st_min = f32(wc("works_stream_min_km2"));
    std::vector<int32_t> cand;
    for (size_t k = 0; k < N; ++k) {
        if (!land(k) || g.lake.v[k]) continue;
        const float a = f32(g.acc_km2.v[k]);
        if (!(a >= src_min)) continue;
        if (g.river.v[k] > 0 || (g.stream.v[k] > 0 && a >= st_min)) cand.push_back(static_cast<int32_t>(k));
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return zf[a] > zf[b]; });
    std::vector<uint8_t> passable(N, 0);
    for (size_t k = 0; k < N; ++k)
        passable[k] = (land(k) && !g.cliff.v[k] && !g.lake.v[k] && !(g.river.v[k] > 0) && g.landcover.v[k] != LC_WET) ? 1 : 0;
    const double grad_c = wc("canal_grad_m_per_km") * res_km;
    const double cut = wc("canal_cut_m");
    const double Rc = wc("canal_reach_km") / res_km;
    const int Rw = static_cast<int>(std::ceil(Rc - 1e-9));
    const double a_cost = wc("canal_slope_cost");
    t = wc("village_head_sep_km") / res_km;
    const double sep_h2 = t * t;
    t = wc("canal_try_sep_km") / res_km;
    const double sep_t2 = t * t;
    const int64_t min_cmd = cells_min(wc("canal_min_cmd_km2"), cell_km2);
    const double qk = wc("canal_q_per_km2");
    const double wmin = wc("canal_width_min_m");
    const int s_lat = std::max(1, static_cast<int>(std::nearbyint(wc("canal_lateral_km") / res_km)));
    const double b0 = wc("canal_base_km"), bk = wc("canal_km_per_km2");
    struct Win {
        int r0, r1, c0, c1;
    };
    auto window = [&](int ci, int cj, int vi, int vj) {
        return Win{std::max({0, ci - Rw, vi - wr}), std::min({H, ci + Rw + 1, vi + wr + 1}), std::max({0, cj - Rw, vj - wr}),
                   std::min({W, cj + Rw + 1, vj + wr + 1})};
    };
    auto walk_ok = [&](int a, int b, int vi, int vj) {
        const int64_t di = a - vi, dj = b - vj;
        return static_cast<double>(di * di + dj * dj) <= W2;
    };
    // 灌得到的目标格的上界：离渠首 ≤ Rc、离管它的村走得到、低于渠水面线（直线距离算比降）
    auto upper = [&](int ci, int cj, double z0, int vi, int vj, const std::function<bool(size_t)>& tgt) {
        const Win w = window(ci, cj, vi, vj);
        int64_t m = 0;
        for (int a = w.r0; a < w.r1; ++a)
            for (int b = w.c0; b < w.c1; ++b) {
                const size_t gk = static_cast<size_t>(a) * W + b;
                if (!tgt(gk)) continue;
                const int di = a - ci, dj = b - cj;
                const double dist = std::sqrt(static_cast<double>(di * di + dj * dj));
                if (!(dist <= Rc) || !walk_ok(a, b, vi, vj)) continue;
                if (g.height.v[gk] <= z0 - grad_c * dist) ++m;
            }
        return m;
    };
    struct Dug {
        bool ok = false;
        int32_t entry = -1;
        std::vector<int32_t> lat, rc;
        Win w{0, 0, 0, 0};
        std::vector<int32_t> par;
    };
    // 从渠首按最省工的路走出去（只走管它的村走得到的格）：目标田里走得到、在渠水面以下的格就是灌区（按格算），够 min_cmd、划得来才修
    auto dig = [&](int32_t cc, int ci, int cj, double z0, int k, int vi, int vj, const std::function<bool(size_t)>& tgt) {
        Dug D;
        D.w = window(ci, cj, vi, vj);
        const int r0 = D.w.r0, r1 = D.w.r1, c0 = D.w.c0, c1 = D.w.c1;
        const int ww = c1 - c0;
        const size_t nwin = static_cast<size_t>(r1 - r0) * ww;
        std::vector<double> cost(nwin, INF), ln(nwin, 0.0);
        D.par.assign(nwin, -1);
        const size_t s0 = static_cast<size_t>(ci - r0) * ww + (cj - c0);
        cost[s0] = 0.0;
        using E = std::pair<double, int64_t>;
        std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
        heap.push({0.0, static_cast<int64_t>(cc)});
        while (!heap.empty()) {
            const double d = heap.top().first;
            const int32_t q = static_cast<int32_t>(heap.top().second);
            heap.pop();
            const int qi = q / W, qj = q % W;
            const size_t lq = static_cast<size_t>(qi - r0) * ww + (qj - c0);
            if (d > cost[lq]) continue;
            if (d > Rc) break;
            double hq = z0;
            if (lq != s0) {
                hq = g.height.v[q];
                if (tgt(static_cast<size_t>(q)) && hq <= z0 - grad_c * ln[lq]) D.rc.push_back(q);
            }
            for (const Step& st : STEPS) {
                const int a = qi + st.di, b = qj + st.dj;
                if (a < r0 || a >= r1 || b < c0 || b >= c1) continue;
                const size_t gp = static_cast<size_t>(a) * W + b;
                if (!passable[gp] || g.island_id.v[gp] != k || !walk_ok(a, b, vi, vj)) continue;
                const size_t lp = static_cast<size_t>(a - r0) * ww + (b - c0);
                const double nl = ln[lq] + st.L;
                const double hp = g.height.v[gp];
                if (hp > z0 - grad_c * nl + cut) continue;
                const double sl = std::abs(hp - hq) / (st.L * res_m);
                const double nd = d + st.L * (1.0 + a_cost * sl * sl);
                if (nd < cost[lp]) {
                    cost[lp] = nd;
                    ln[lp] = nl;
                    D.par[lp] = q;
                    heap.push({nd, static_cast<int64_t>(gp)});
                }
            }
        }
        const int64_t cnt = static_cast<int64_t>(D.rc.size());
        if (cnt < min_cmd) return D;
        D.entry = D.rc[0];
        auto up = [&](int32_t q) { return D.par[static_cast<size_t>(q / W - r0) * ww + (q % W - c0)]; };
        int64_t n1 = 0, n2 = 0;
        for (int32_t q = D.entry; q != cc;) {
            const int32_t p = up(q);
            if (p / W != q / W && p % W != q % W) ++n2;
            else ++n1;
            q = p;
        }
        if ((static_cast<double>(n1) + static_cast<double>(n2) * SQRT2) * res_km > b0 + bk * (static_cast<double>(cnt) * cell_km2)) return D;
        for (int32_t q : D.rc)
            if (q != D.entry && (q / W - ci) % s_lat == 0 && (q % W - cj) % s_lat == 0) D.lat.push_back(q);
        std::sort(D.lat.begin(), D.lat.end());
        D.ok = true;
        return D;
    };
    // 渠 = 渠首到入口、再到各格点的路合成的树：从渠首出来的一段是干渠，分出去的是支渠；灌区的格数按格点均分给各段算渠宽
    auto emit = [&](int32_t cc, int k, const Dug& D, int64_t n_cmd, int hid, const std::function<void(Json&)>& extra, Json& out_arr, int64_t& n1_h,
                    int64_t& n2_h) {
        const int ww = D.w.c1 - D.w.c0;
        auto up = [&](int32_t q) { return D.par[static_cast<size_t>(q / W - D.w.r0) * ww + (q % W - D.w.c0)]; };
        std::vector<int32_t> tl{D.entry};
        tl.insert(tl.end(), D.lat.begin(), D.lat.end());
        const int64_t nt = static_cast<int64_t>(tl.size());
        const int64_t base = n_cmd / nt, rem = n_cmd % nt;
        std::map<int32_t, int64_t> served;
        std::map<int32_t, std::set<int32_t>> children;
        for (int64_t ti = 0; ti < nt; ++ti) {
            const int64_t share = base + (ti < rem ? 1 : 0);
            int32_t q = tl[ti];
            int32_t prev = -1;
            while (true) {
                served[q] += share;
                if (prev >= 0) children[q].insert(prev);
                if (q == cc) break;
                prev = q;
                q = up(q);
            }
        }
        n1_h = n2_h = 0;
        int n_seg = 0;
        std::deque<int32_t> dq{cc};
        auto kids = [&](int32_t q) -> const std::set<int32_t>* {
            auto it = children.find(q);
            return it == children.end() ? nullptr : &it->second;
        };
        while (!dq.empty()) {
            const int32_t s_ = dq.front();
            dq.pop_front();
            const auto* ks = kids(s_);
            if (!ks) continue;
            for (int32_t ch : *ks) {
                std::vector<int32_t> pts{s_, ch};
                int32_t cur = ch;
                while (true) {
                    const auto* kc = kids(cur);
                    if (!kc || kc->size() != 1) break;
                    cur = *kc->begin();
                    pts.push_back(cur);
                }
                int64_t n1 = 0, n2 = 0;
                for (size_t u = 0; u + 1 < pts.size(); ++u) {
                    if (pts[u] / W != pts[u + 1] / W && pts[u] % W != pts[u + 1] % W) ++n2;
                    else ++n1;
                }
                n1_h += n1;
                n2_h += n2;
                const int64_t sv = served[pts[1]];
                const double q_m3s = qk * (static_cast<double>(sv) * cell_km2);
                Json cj_ = Json::obj();
                cj_.set("kind", s_ == cc ? "main" : "branch");
                cj_.set("head", hid);
                cj_.set("island", k);
                Json pa = Json::arr();
                for (int32_t p : pts) pa.push(cellpt(p));
                cj_.set("pts", pa);
                cj_.set("length_km", pyround((static_cast<double>(n1) + static_cast<double>(n2) * SQRT2) * res_km, 3));
                cj_.set("served_km2", pyround(static_cast<double>(sv) * cell_km2, 3));
                cj_.set("width_m", pyround(std::max(wmin, 5.0 * std::sqrt(q_m3s)), 1));
                extra(cj_);
                out_arr.push(std::move(cj_));
                ++n_seg;
                const auto* kc = kids(cur);
                if (kc && !kc->empty()) dq.push_back(cur);
            }
        }
        return n_seg;
    };
    struct HeadRec {
        int id, island, ci, cj;
        bool seasonal;
        double level, basin;
        std::vector<int> fields;
        int64_t tot;
        int64_t n1, n2;
        int n_seg;
        int village = 0, ruin = 0, years = 0;
        bool abandoned = false;
    };
    auto near = [](int ci, int cj, const std::vector<std::array<int, 2>>& pts, double r2) {
        for (const auto& hc : pts) {
            const double d2 = static_cast<double>((ci - hc[0]) * (ci - hc[0]) + (cj - hc[1]) * (cj - hc[1]));
            if (d2 < r2) return true;
        }
        return false;
    };
    std::vector<uint8_t> cmd(static_cast<size_t>(nf) + 1, 0);
    std::vector<const WorksVillage*> cvill;      // 一村一堰：灌自己的旱地田
    for (const WorksVillage& v : villages)
        if (!pol_field[v.field] && f_cells[v.field] >= min_cmd) cvill.push_back(&v);
    std::vector<std::array<int, 2>> head_cells, tried;
    std::vector<HeadRec> heads;
    Json canals = Json::arr();
    int64_t n1_all = 0, n2_all = 0;
    for (int32_t cc : cand) {
        const int ci = cc / W, cj = cc % W;
        if (near(ci, cj, head_cells, sep_h2) || near(ci, cj, tried, sep_t2)) continue;
        const int k = g.island_id.v[cc];
        const double z0 = zf[cc];
        struct VS {
            int64_t d2;
            int id;
            const WorksVillage* v;
        };
        std::vector<VS> vs;
        for (const WorksVillage* v : cvill) {
            const int64_t d2 = static_cast<int64_t>(v->ci - ci) * (v->ci - ci) + static_cast<int64_t>(v->cj - cj) * (v->cj - cj);
            if (v->island == k && !cmd[v->field] && static_cast<double>(d2) <= W2) vs.push_back({d2, v->id, v});
        }
        std::stable_sort(vs.begin(), vs.end(), [](const VS& a, const VS& b) {
            if (a.d2 != b.d2) return a.d2 < b.d2;
            return a.id < b.id;
        });
        std::vector<const WorksVillage*> ok_v;
        for (const VS& x : vs) {
            const int F = x.v->field;
            if (upper(ci, cj, z0, x.v->ci, x.v->cj, [&](size_t q) { return fields_raster.v[q] == F; }) >= min_cmd) ok_v.push_back(x.v);
        }
        if (ok_v.empty()) continue;
        tried.push_back({ci, cj});
        for (const WorksVillage* v : ok_v) {
            const int F = v->field;
            const Dug D = dig(cc, ci, cj, z0, k, v->ci, v->cj, [&](size_t q) { return fields_raster.v[q] == F; });
            if (!D.ok) continue;
            const int hid = static_cast<int>(heads.size()) + 1;
            head_cells.push_back({ci, cj});
            cmd[F] = 1;
            const int64_t n_cmd = static_cast<int64_t>(D.rc.size());
            cmd_cells.insert(cmd_cells.end(), D.rc.begin(), D.rc.end());
            int64_t n1_h = 0, n2_h = 0;
            const int vid = v->id;
            const int n_seg = emit(cc, k, D, n_cmd, hid, [&](Json& j) { j.set("village", vid); }, canals, n1_h, n2_h);
            n1_all += n1_h;
            n2_all += n2_h;
            HeadRec hr{hid, k, ci, cj, !(g.river.v[cc] > 0), z0, static_cast<double>(f32(g.acc_km2.v[cc])), {F}, n_cmd, n1_h, n2_h, n_seg};
            hr.village = vid;
            heads.push_back(std::move(hr));
            break;
        }
    }
    std::sort(cmd_cells.begin(), cmd_cells.end());
    // 废村旁没人管的废渠首、废渠：灌过它的撂荒田；一个废村至多一处
    std::vector<HeadRec> ab_heads;
    Json ab_canals = Json::arr();
    int64_t ab_n1 = 0, ab_n2 = 0;
    for (const WorksRuin& r : ruins) {
        if (static_cast<int64_t>(r.cells.size()) < min_cmd) continue;
        std::vector<uint8_t> tm(N, 0);
        for (int32_t q : r.cells) tm[q] = 1;
        const int vi = r.ci, vj = r.cj, k = r.island;
        std::vector<std::array<int, 2>> rtried;
        for (int32_t cc : cand) {
            const int ci = cc / W, cj = cc % W;
            if (g.island_id.v[cc] != k || !walk_ok(ci, cj, vi, vj)) continue;
            if (near(ci, cj, head_cells, sep_h2) || near(ci, cj, rtried, sep_t2)) continue;
            const double z0 = zf[cc];
            auto tgt = [&](size_t q) { return tm[q] != 0; };
            if (upper(ci, cj, z0, vi, vj, tgt) < min_cmd) continue;
            rtried.push_back({ci, cj});
            const Dug D = dig(cc, ci, cj, z0, k, vi, vj, tgt);
            if (!D.ok) continue;
            const int hid = static_cast<int>(heads.size() + ab_heads.size()) + 1;
            head_cells.push_back({ci, cj});
            const int64_t n_cmd = static_cast<int64_t>(D.rc.size());
            int64_t n1_h = 0, n2_h = 0;
            const int rid = r.id, years = r.years;
            const int n_seg = emit(cc, k, D, n_cmd, hid, [&](Json& j) {
                j.set("village", Json());
                j.set("ruin", rid);
                j.set("abandoned", true);
                j.set("abandoned_years", years);
            }, ab_canals, n1_h, n2_h);
            ab_n1 += n1_h;
            ab_n2 += n2_h;
            HeadRec hr{hid, k, ci, cj, !(g.river.v[cc] > 0), z0, static_cast<double>(f32(g.acc_km2.v[cc])), {}, n_cmd, n1_h, n2_h, n_seg};
            hr.ruin = rid;
            hr.years = years;
            hr.abandoned = true;
            ab_heads.push_back(std::move(hr));
            break;
        }
    }

    // ---------- 圩田：纵浦横塘、排水渠、圩堤、圩塘与闸（各归种那组圩田的村）
    Json polders = Json::arr(), patches = Json::arr(), pol_canals = Json::arr();
    std::vector<Json> sluices_p, sluices_o, ponds_p;
    const int drain_max = static_cast<int>(wc("polder_drain_max_cells"));
    const double pond_frac = wc("polder_pond_frac");
    const float paddy_mm = f32(wc("polder_paddy_mm"));
    int64_t ns_cells = 0, ew_cells = 0, drain_n1 = 0, drain_n2 = 0, dike_edges = 0, n_out_unmanaged = 0;
    int patch_no = 0;
    for (const PolderPatch& P : plan.patches) {
        ++patch_no;
        const int io = P.io, jo = P.jo, s = P.spacing;
        const int32_t o = P.outlet;
        const int oi = o / W, oj = o % W;
        const int k = P.island;
        int64_t drained = 0;
        std::set<int> pv;
        for (const PolderBlock& B : P.blocks) {
            const int pid = plan.polder_id.v[B.cells[0]];
            const int vid = fvill[fields_raster.v[B.cells[0]]];
            if (vid) pv.insert(vid);
            const int64_t n = static_cast<int64_t>(B.cells.size());
            drained += n;
            int64_t edges = 0;
            int32_t best_s = -1;
            int64_t best_d = 0;
            int imin = H, imax = -1, jmin = W, jmax = -1;
            int32_t pond = B.cells[0];
            float pacc = f32(g.acc_km2.v[pond]);
            for (int32_t q : B.cells) {
                const int qi = q / W, qj = q % W;
                imin = std::min(imin, qi);
                imax = std::max(imax, qi);
                jmin = std::min(jmin, qj);
                jmax = std::max(jmax, qj);
                bool bnd = false;
                for (const auto& d4 : N4W) {
                    const int a = qi + d4[0], b = qj + d4[1];
                    if (a < 0 || a >= H || b < 0 || b >= W || plan.polder_id(a, b) != pid) {
                        ++edges;
                        bnd = true;
                    }
                }
                if (bnd) {
                    const int64_t dd = static_cast<int64_t>(qi - oi) * (qi - oi) + static_cast<int64_t>(qj - oj) * (qj - oj);
                    if (best_s < 0 || dd < best_d) {
                        best_s = q;
                        best_d = dd;
                    }
                }
                const float a = f32(g.acc_km2.v[q]);
                if (a > pacc) {
                    pacc = a;
                    pond = q;
                }
            }
            dike_edges += edges;
            const int pi = pond / W, pj = pond % W, si = best_s / W, sj = best_s % W;
            Json pj_ = Json::obj();
            pj_.set("kind", "polder");
            pj_.set("island", k);
            pj_.set("cell", Json::ipair(pi, pj));
            pj_.set("km", km(pi, pj));
            pj_.set("polder", pid);
            pj_.set("area_m2", static_cast<int64_t>(std::nearbyint(static_cast<double>(n) * cell_km2 * 1e6 * pond_frac)));
            pj_.set("elev_m", pyround(g.height.v[pond], 0));
            pj_.set("village", vjson(vid));
            ponds_p.push_back(std::move(pj_));
            Json sj_ = Json::obj();
            sj_.set("kind", "polder");
            sj_.set("island", k);
            sj_.set("cell", Json::ipair(si, sj));
            sj_.set("km", km(si, sj));
            sj_.set("polder", pid);
            sj_.set("village", vjson(vid));
            sluices_p.push_back(std::move(sj_));
            const int r0_ = imin, r1_ = imax + 1, c0_ = jmin, c1_ = jmax + 1;
            const double bx0 = pyround(x0 + (static_cast<double>(c0_) - 0.5 + 0.5) * res_km, 3), by0 = pyround(y0 - (static_cast<double>(r0_) - 0.5 + 0.5) * res_km, 3);
            const double bx1 = pyround(x0 + (static_cast<double>(c1_) - 0.5 + 0.5) * res_km, 3), by1 = pyround(y0 - (static_cast<double>(r1_) - 0.5 + 0.5) * res_km, 3);
            Json pr = Json::obj();
            pr.set("id", pid);
            pr.set("patch", patch_no);
            pr.set("island", k);
            pr.set("block", Json::ipair(B.bi, B.bj));
            pr.set("cells", n);
            pr.set("area_km2", pyround(static_cast<double>(n) * cell_km2, 3));
            Json bb = Json::arr();
            bb.push(r0_);
            bb.push(c0_);
            bb.push(r1_);
            bb.push(c1_);
            pr.set("cells_bbox", bb);
            Json kb = Json::arr();
            kb.push(bx0);
            kb.push(by1);
            kb.push(bx1);
            kb.push(by0);
            pr.set("km_bbox", kb);
            pr.set("dike_km", pyround(static_cast<double>(edges) * res_km, 3));
            pr.set("paddy", f32(g.rain.v[pond]) >= paddy_mm);
            pr.set("village", vjson(vid));
            polders.push(std::move(pr));
        }
        // 排水闸与排水渠：出水口最近、走得到的村管（同岛，平局村号小）；没有就不修；排水渠走出那个村走得到的地方就截短
        const WorksVillage* ov = nullptr;
        int64_t od = 0;
        for (const WorksVillage& v : villages) {
            if (v.island != k) continue;
            const int64_t d2 = static_cast<int64_t>(v.ci - oi) * (v.ci - oi) + static_cast<int64_t>(v.cj - oj) * (v.cj - oj);
            if (static_cast<double>(d2) <= W2 && (!ov || d2 < od)) {
                ov = &v;
                od = d2;
            }
        }
        if (!ov) {
            ++n_out_unmanaged;
        } else {
            std::vector<int32_t> pts{o};
            int32_t q = o;
            for (int st = 0; st < drain_max; ++st) {
                const int qi = q / W, qj = q % W;
                const int a = g.recv_i(qi, qj), b = g.recv_j(qi, qj);
                if (a < 0 || b < 0 || (a == qi && b == qj) || !walk_ok(a, b, ov->ci, ov->cj)) break;
                q = a * W + b;
                pts.push_back(q);
                if (g.island_id.v[q] != k || g.river.v[q] > 0 || g.stream.v[q] > 0 || g.lake.v[q]) break;
            }
            int64_t n1 = 0, n2 = 0;
            for (size_t u = 0; u + 1 < pts.size(); ++u) {
                if (pts[u] / W != pts[u + 1] / W && pts[u] % W != pts[u + 1] % W) ++n2;
                else ++n1;
            }
            drain_n1 += n1;
            drain_n2 += n2;
            if (pts.size() >= 2) {
                Json cj_ = Json::obj();
                cj_.set("kind", "drain");
                cj_.set("patch", patch_no);
                cj_.set("island", k);
                Json pa = Json::arr();
                for (int32_t p : pts) pa.push(cellpt(p));
                cj_.set("pts", pa);
                cj_.set("length_km", pyround((static_cast<double>(n1) + static_cast<double>(n2) * SQRT2) * res_km, 3));
                cj_.set("village", ov->id);
                pol_canals.push(std::move(cj_));
            }
            Json so = Json::obj();
            so.set("kind", "outlet");
            so.set("island", k);
            so.set("cell", Json::ipair(oi, oj));
            so.set("km", km(oi, oj));
            so.set("patch", patch_no);
            so.set("village", ov->id);
            sluices_o.push_back(std::move(so));
        }
        // 纵浦横塘：格线穿过这片圩田的段，按两旁的圩分给各自的村（上 / 左那格优先）
        std::vector<uint8_t> dm(N, 0);
        int rmin = H, rmax = -1, cmin = W, cmax = -1;
        for (const PolderBlock& B : P.blocks)
            for (int32_t cq : B.cells) {
                dm[cq] = 1;
                rmin = std::min(rmin, cq / W);
                rmax = std::max(rmax, cq / W);
                cmin = std::min(cmin, cq % W);
                cmax = std::max(cmax, cq % W);
            }
        auto own = [&](int a, int b) -> int {
            if (a < 0 || a >= H || b < 0 || b >= W) return -1;
            const size_t q = static_cast<size_t>(a) * W + b;
            return dm[q] ? fvill[fields_raster.v[q]] : -1;
        };
        for (int e = io + s * floordiv(rmin - io + s - 1, s); e <= rmax + 1; e += s) {
            std::vector<int> ow;
            for (int b = cmin; b <= cmax; ++b) {
                const int x = own(e - 1, b);
                ow.push_back(x >= 0 ? x : own(e, b));
            }
            for (const auto& r : runs_owner(ow)) {
                ew_cells += r[1] - r[0];
                Json cj_ = Json::obj();
                cj_.set("kind", "ew");
                cj_.set("patch", patch_no);
                cj_.set("island", k);
                Json pa = Json::arr();
                pa.push(Json::pair(static_cast<double>(e), static_cast<double>(cmin + r[0])));
                pa.push(Json::pair(static_cast<double>(e), static_cast<double>(cmin + r[1])));
                cj_.set("pts", pa);
                cj_.set("length_km", pyround(static_cast<double>(r[1] - r[0]) * res_km, 3));
                cj_.set("village", vjson(r[2]));
                pol_canals.push(std::move(cj_));
            }
        }
        for (int e = jo + s * floordiv(cmin - jo + s - 1, s); e <= cmax + 1; e += s) {
            std::vector<int> ow;
            for (int a = rmin; a <= rmax; ++a) {
                const int x = own(a, e - 1);
                ow.push_back(x >= 0 ? x : own(a, e));
            }
            for (const auto& r : runs_owner(ow)) {
                ns_cells += r[1] - r[0];
                Json cj_ = Json::obj();
                cj_.set("kind", "ns");
                cj_.set("patch", patch_no);
                cj_.set("island", k);
                Json pa = Json::arr();
                pa.push(Json::pair(static_cast<double>(rmin + r[0]), static_cast<double>(e)));
                pa.push(Json::pair(static_cast<double>(rmin + r[1]), static_cast<double>(e)));
                cj_.set("pts", pa);
                cj_.set("length_km", pyround(static_cast<double>(r[1] - r[0]) * res_km, 3));
                cj_.set("village", vjson(r[2]));
                pol_canals.push(std::move(cj_));
            }
        }
        Json pt = Json::obj();
        pt.set("id", patch_no);
        pt.set("island", k);
        pt.set("outlet_cell", Json::ipair(oi, oj));
        pt.set("km", km(oi, oj));
        pt.set("pressure", pyround(P.pressure, 3));
        pt.set("wetland_km2", pyround(static_cast<double>(P.wet_cells) * cell_km2, 3));
        pt.set("polder_km2", pyround(static_cast<double>(drained) * cell_km2, 3));
        pt.set("polders", Json::arr_of(P.ids));
        pt.set("villages", Json::arr_of(std::vector<int>(pv.begin(), pv.end())));
        patches.push(std::move(pt));
    }

    // ---------- 塘：堰塘（季节性渠首）、村塘 / 山塘（每村一口）、圩塘；废村旁的废塘
    std::vector<uint8_t> freec(N, 0);
    for (size_t k = 0; k < N; ++k)
        freec[k] = (land(k) && !g.cliff.v[k] && !g.lake.v[k] && !(g.river.v[k] > 0) && g.landcover.v[k] != LC_WET && g.cultivated.v[k] == 0 &&
                    g.fallow_years.v[k] == 0 && sraster.v[k] == 0) ? 1 : 0;
    const int reach = static_cast<int>(wc("pond_reach_cells")), nearc = static_cast<int>(wc("pond_near_cells"));
    const double hill_ratio = wc("hill_pond_ratio"), hill_min = wc("hill_pond_min_m2"), m2_hh = wc("pond_m2_per_hh"), v_min = wc("pond_min_m2");
    // walk_v 给了（≥ 0）就只要那个村（格 walk_i, walk_j）走得到的格（堰塘：管渠首的村）
    auto pick = [&](int ci, int cj, int r, bool use_h, double hmin, int walk_i = -1, int walk_j = -1) {
        int32_t best = -1;
        double bacc = 0.0;
        int64_t bd = 0;
        const int k0 = g.island_id(ci, cj);
        for (int a = std::max(0, ci - r); a < std::min(H, ci + r + 1); ++a)
            for (int b = std::max(0, cj - r); b < std::min(W, cj + r + 1); ++b) {
                const size_t gk = static_cast<size_t>(a) * W + b;
                if (!freec[gk] || g.island_id.v[gk] != k0) continue;
                if (use_h && !(g.height.v[gk] >= hmin)) continue;
                if (walk_i >= 0 && !walk_ok(a, b, walk_i, walk_j)) continue;
                const double av = static_cast<double>(f32(g.acc_km2.v[gk]));
                const int64_t dd = static_cast<int64_t>(a - ci) * (a - ci) + static_cast<int64_t>(b - cj) * (b - cj);
                if (best < 0 || av > bacc || (av == bacc && dd < bd)) {
                    best = static_cast<int32_t>(gk);
                    bacc = av;
                    bd = dd;
                }
            }
        return best;
    };
    auto mk_pond = [&](int32_t q, const char* kind, double area) {
        freec[q] = 0;
        Json p = Json::obj();
        p.set("kind", kind);
        p.set("island", static_cast<int>(g.island_id.v[q]));
        p.set("cell", Json::ipair(q / W, q % W));
        p.set("km", km(q / W, q % W));
        p.set("area_m2", static_cast<int64_t>(std::nearbyint(area)));
        p.set("elev_m", pyround(g.height.v[q], 0));
        return p;
    };
    // 村塘 / 山塘的挑法：山塘在村上坡（不低于村）的格里挑，没有就放宽；村塘先在近处挑，没有就放宽
    // walk 为真就只挑这个村（ci, cj）走得到的格
    auto village_pond = [&](int ci, int cj, bool hill, bool walk) {
        const int wi = walk ? ci : -1, wj = walk ? cj : -1;
        if (hill) {
            int32_t q = pick(ci, cj, reach, true, g.height(ci, cj), wi, wj);
            if (q < 0) q = pick(ci, cj, reach, false, 0.0, wi, wj);
            return q;
        }
        int32_t q = pick(ci, cj, nearc, false, 0.0, wi, wj);
        if (q < 0) q = pick(ci, cj, reach, false, 0.0, wi, wj);
        return q;
    };
    std::vector<Json> ponds;
    std::map<int, std::array<int, 2>> vcell;
    for (const WorksVillage& v : villages) vcell[v.id] = {v.ci, v.cj};
    for (const HeadRec& hd : heads) {
        if (!hd.seasonal) continue;
        const auto& vc = vcell.at(hd.village);
        const int32_t q = pick(hd.ci, hd.cj, nearc, false, 0.0, vc[0], vc[1]);
        const double served_km2 = pyround(static_cast<double>(hd.tot) * cell_km2, 3);
        if (q >= 0) {
            Json p = mk_pond(q, "weir", std::max(hill_min, served_km2 * 1e6 * hill_ratio));
            p.set("head", hd.id);
            p.set("village", vjson(hd.village));
            ponds.push_back(std::move(p));
        }
    }
    for (const WorksVillage& v : villages) {
        const WorksField& f = fields[v.field - 1];
        bool hill = false;
        if (cmd[v.field] || pol_field[v.field] || !v.polder_fields.empty()) hill = false;   // 灌区里的村（有渠或圩田）：村塘接渠水
        else if (g.zone(v.ci, v.cj) >= 1 && g.zone(v.ci, v.cj) <= 3) hill = true;         // 高山 / 山地 / 丘陵：陂塘
        const int32_t q = village_pond(v.ci, v.cj, hill, true);
        const double area = hill ? std::max(hill_min, f.area_km2 * 1e6 * hill_ratio) : std::max(v_min, m2_hh * static_cast<double>(v.households));
        if (q >= 0) {
            Json p = mk_pond(q, hill ? "hill" : "village", area);
            p.set("village", v.id);
            ponds.push_back(std::move(p));
        }
    }
    std::vector<Json> ponds_ab;
    for (const WorksRuin& r : ruins) {                     // 废塘：死了的村的村塘 / 山塘，淤了、没人管
        const bool hill = g.zone(r.ci, r.cj) >= 1 && g.zone(r.ci, r.cj) <= 3;
        const int32_t q = village_pond(r.ci, r.cj, hill, false);
        const double area = hill ? std::max(hill_min, r.fallow_km2 * 1e6 * hill_ratio) : std::max(v_min, m2_hh * static_cast<double>(r.households_before));
        if (q >= 0) {
            Json p = mk_pond(q, hill ? "hill" : "village", area);
            p.set("village", Json());
            p.set("ruin", r.id);
            p.set("abandoned", true);
            p.set("abandoned_years", r.years);
            ponds_ab.push_back(std::move(p));
        }
    }
    for (Json& p : ponds_p) ponds.push_back(std::move(p));
    const size_t n_live_ponds = ponds.size();
    for (Json& p : ponds_ab) ponds.push_back(std::move(p));
    std::map<std::string, int64_t> pk{{"weir", 0}, {"village", 0}, {"hill", 0}, {"polder", 0}};
    Json pj = Json::arr();
    for (size_t i = 0; i < ponds.size(); ++i) {
        Json& p = ponds[i];
        const auto& cell = p.at("cell").items();
        sraster(static_cast<int>(cell[0].as_int()), static_cast<int>(cell[1].as_int())) = i < n_live_ponds ? 15 : 19;
        p.set("id", static_cast<int64_t>(i) + 1);
        if (i < n_live_ponds) pk[p.at("kind").as_str()]++;
        pj.push(std::move(p));
    }

    // ---------- 闸
    std::vector<Json> sl;
    auto head_sluice = [&](const HeadRec& hd) {
        Json x = Json::obj();
        x.set("kind", "head");
        x.set("island", hd.island);
        x.set("cell", Json::ipair(hd.ci, hd.cj));
        x.set("km", km(hd.ci, hd.cj));
        x.set("head", hd.id);
        x.set("village", vjson(hd.village));
        if (hd.abandoned) {
            x.set("ruin", hd.ruin);
            x.set("abandoned", true);
            x.set("abandoned_years", hd.years);
        }
        return x;
    };
    for (const HeadRec& hd : heads) sl.push_back(head_sluice(hd));
    for (Json& x : sluices_p) sl.push_back(std::move(x));
    for (Json& x : sluices_o) sl.push_back(std::move(x));
    const size_t n_live_sl = sl.size();
    for (const HeadRec& hd : ab_heads) sl.push_back(head_sluice(hd));
    std::map<std::string, int64_t> sk{{"head", 0}, {"polder", 0}, {"outlet", 0}};
    Json sj = Json::arr();
    for (size_t i = 0; i < sl.size(); ++i) {
        Json& x = sl[i];
        x.set("id", static_cast<int64_t>(i) + 1);
        const auto& cell = x.at("cell").items();
        uint8_t& sv = sraster(static_cast<int>(cell[0].as_int()), static_cast<int>(cell[1].as_int()));
        if (i >= n_live_sl) {
            if (sv == 0 || sv == 1 || sv == 2 || sv == 11) sv = 19;
        } else {
            if (sv == 0 || sv == 1 || sv == 2) sv = 16;
            sk[x.at("kind").as_str()]++;
        }
        sj.push(std::move(x));
    }
    for (Json& x : pol_canals.items()) canals.push(std::move(x));
    for (Json& x : ab_canals.items()) canals.push(std::move(x));
    for (size_t i = 0; i < canals.items().size(); ++i) canals.items()[i].set("id", static_cast<int64_t>(i) + 1);

    Json hj = Json::arr();
    int64_t n_seas = 0;
    auto head_json = [&](const HeadRec& hd) {
        Json x = Json::obj();
        x.set("id", hd.id);
        x.set("island", hd.island);
        x.set("cell", Json::ipair(hd.ci, hd.cj));
        x.set("km", km(hd.ci, hd.cj));
        x.set("source", hd.seasonal ? "stream" : "river");
        x.set("seasonal", hd.seasonal);
        x.set("level_m", pyround(hd.level, 1));
        x.set("basin_km2", pyround(hd.basin, 1));
        x.set("fields", Json::arr_of(hd.fields));
        x.set("served_km2", pyround(static_cast<double>(hd.tot) * cell_km2, 3));
        x.set("canal_km", pyround((static_cast<double>(hd.n1) + static_cast<double>(hd.n2) * SQRT2) * res_km, 3));
        x.set("n_canals", hd.n_seg);
        x.set("village", vjson(hd.village));
        if (hd.abandoned) {
            x.set("ruin", hd.ruin);
            x.set("abandoned", true);
            x.set("abandoned_years", hd.years);
        }
        return x;
    };
    for (const HeadRec& hd : heads) {
        hj.push(head_json(hd));
        n_seas += hd.seasonal ? 1 : 0;
    }
    for (const HeadRec& hd : ab_heads) hj.push(head_json(hd));
    int64_t n_cult = 0, pol_cells = 0;
    for (size_t k = 0; k < N; ++k) {
        n_cult += g.cultivated.v[k] > 0 ? 1 : 0;
        pol_cells += plan.polder_id.v[k] > 0 ? 1 : 0;
    }
    const int64_t n_cmd_all = static_cast<int64_t>(cmd_cells.size());
    const int64_t wet_cells = plan.wetland_cells;
    Json S = Json::obj();
    S.set("n_heads", static_cast<int64_t>(heads.size()));
    S.set("n_heads_seasonal", n_seas);
    S.set("canal_km", pyround((static_cast<double>(n1_all) + static_cast<double>(n2_all) * SQRT2) * res_km, 3));
    S.set("polder_canal_km", pyround(static_cast<double>(ns_cells + ew_cells) * res_km, 3));
    S.set("drain_km", pyround((static_cast<double>(drain_n1) + static_cast<double>(drain_n2) * SQRT2) * res_km, 3));
    S.set("commanded_km2", pyround(static_cast<double>(n_cmd_all) * cell_km2, 3));
    S.set("commanded_share", pyround(static_cast<double>(n_cmd_all) / static_cast<double>(std::max<int64_t>(1, n_cult)), 3));
    S.set("n_ponds", static_cast<int64_t>(n_live_ponds));
    Json pko = Json::obj();
    for (const auto& kv : pk) pko.set(kv.first, kv.second);
    S.set("ponds", pko);
    S.set("n_sluices", static_cast<int64_t>(n_live_sl));
    Json sko = Json::obj();
    for (const auto& kv : sk) sko.set(kv.first, kv.second);
    S.set("sluices", sko);
    S.set("wetland_km2", pyround(static_cast<double>(wet_cells) * cell_km2, 3));
    S.set("polder_km2", pyround(static_cast<double>(pol_cells) * cell_km2, 3));
    S.set("polder_share", wet_cells ? pyround(static_cast<double>(pol_cells) / static_cast<double>(wet_cells), 3) : 0.0);
    S.set("n_polders", static_cast<int64_t>(polders.size()));
    S.set("n_polder_patches", static_cast<int64_t>(patches.size()));
    S.set("dike_km", pyround(static_cast<double>(dike_edges) * res_km, 3));
    S.set("manage_walk_km", wc("manage_walk_km"));
    S.set("n_outlets_unmanaged", n_out_unmanaged);
    int64_t n_pv = 0, n_vp = 0;
    for (const WorksVillage& v : villages) {
        n_pv += v.polder ? 1 : 0;
        n_vp += v.polder_fields.empty() ? 0 : 1;
    }
    S.set("n_polder_villages", n_pv);
    S.set("n_villages_with_polders", n_vp);
    Json ab = Json::obj();
    ab.set("heads", static_cast<int64_t>(ab_heads.size()));
    ab.set("canal_km", pyround((static_cast<double>(ab_n1) + static_cast<double>(ab_n2) * SQRT2) * res_km, 3));
    ab.set("ponds", static_cast<int64_t>(ponds_ab.size()));
    ab.set("sluices", static_cast<int64_t>(ab_heads.size()));
    S.set("abandoned", ab);
    Json out = Json::obj();
    out.set("heads", hj);
    out.set("canals", canals);
    out.set("ponds", pj);
    out.set("sluices", sj);
    out.set("polders", polders);
    out.set("polder_patches", patches);
    out.set("summary", S);
    return out;
}

void landuse_layers(Group& g, const std::vector<int32_t>& cmd_cells) {
    enum { LU_FIELD = 1, LU_TERRACE = 2, LU_CANAL = 3, LU_POLDER = 4, LU_FALLOW = 5, LU_PASTURE = 6, LU_PIT = 7 };
    constexpr uint8_t ROCK = 2, FOREST = 4, SHRUB = 5, GRASS = 6, RIVER = 10, LAKE = 11;
    const size_t N = static_cast<size_t>(g.H) * g.W;
    g.landcover_natural = g.cover_natural;
    g.landuse = Grid<uint8_t>(g.H, g.W, 0);
    for (size_t k = 0; k < N; ++k) {
        uint8_t& nat = g.landcover_natural.v[k];
        if (g.river.v[k] > 0) nat = RIVER;
        if (g.lake.v[k]) nat = LAKE;
        const uint8_t lc = g.landcover.v[k];
        const bool chg = g.island_id.v[k] >= 0 && lc != nat;
        uint8_t u = 0;
        if (chg && lc == ROCK) u = LU_PIT;
        if (chg && (lc == GRASS || lc == SHRUB) && nat == FOREST) u = LU_PASTURE;
        if (g.fallow_years.v[k] > 0) u = LU_FALLOW;
        if (g.cultivated.v[k] == 1) u = LU_FIELD;
        if (g.cultivated.v[k] == 2) u = LU_TERRACE;
        g.landuse.v[k] = u;
    }
    for (int32_t q : cmd_cells) g.landuse.v[q] = LU_CANAL;           // 渠灌田：谷口的渠灌得到的格
    for (size_t k = 0; k < N; ++k)
        if (g.polder_id.v[k] > 0) g.landuse.v[k] = LU_POLDER;
}

}  // namespace skyisle::island
