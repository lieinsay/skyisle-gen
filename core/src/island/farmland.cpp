// 宜垦 / 已垦 / 撂荒（P5；skyisle_gen/island/farmland.py 的 cultivable_land / fill_cultivated 逐位同式）。
#include "skyisle/island/farmland.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>

#include "skyisle/island/resources.hpp"

namespace skyisle::island {

namespace {
enum { LC_FOREST = 4, LC_SHRUB = 5, LC_GRASS = 6, LC_ARABLE = 7, LC_TERRACE = 8, LC_WET = 9 };

// 大岛保底（用户 09-29 定；farmland.floor_cells 同式）：主岛以外 ≥ island_floor_km2 的岛，在它合门槛的宜垦片里挑最好的那片，
// 从片里最好的能开的格起按（适宜度降序、格号升序）往 8 邻域的能开的格长成连成一块的 need 格；长不到就从片里下一个没走到的最好的格重长，
// 整片都不行换下一片。各岛按「最好的格排在好地先占的第几」先后分，额度不够就停
void floor_cells(const Group& g, const Config& c, const GridI& lab, int n_lab, const std::vector<uint8_t>& ok_t, const std::vector<int32_t>& order,
                 const std::vector<uint8_t>& pit, double land_per_hh, int64_t n_quota, std::vector<int32_t>& out, std::vector<int>& isl) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double cell_km2 = g.res_km * g.res_km;
    const int64_t need = static_cast<int64_t>(std::ceil(c.get("settle.island_floor_hh") * land_per_hh / cell_km2 - 1e-9));
    const double floor_km2 = c.get("settle.island_floor_km2");
    if (need <= 0 || n_lab == 0) return;
    std::vector<int64_t> elig_cnt(static_cast<size_t>(n_lab) + 1, 0), first(static_cast<size_t>(n_lab) + 1, -1);
    for (size_t f = 0; f < order.size(); ++f) {
        const int t = lab.v[order[f]];
        if (first[t] < 0) first[t] = static_cast<int64_t>(f);
        elig_cnt[t]++;
    }
    std::map<int, std::vector<std::pair<int64_t, int>>> by_isl;
    for (int t = 1; t <= n_lab; ++t) {
        if (first[t] < 0 || !ok_t[t] || elig_cnt[t] < need) continue;
        const int k = g.island_id.v[order[first[t]]];
        if (k == 0 || g.islands[k].area_j < floor_km2) continue;
        by_isl[k].push_back({first[t], t});
    }
    std::vector<std::pair<int64_t, int>> plan;
    for (auto& kv : by_isl) {
        std::sort(kv.second.begin(), kv.second.end());
        plan.push_back({kv.second.front().first, kv.first});
    }
    std::sort(plan.begin(), plan.end());
    std::map<int, std::vector<int32_t>> tcells;            // 候选片的格（按好地先占的次序）
    for (const auto& kv : by_isl)
        for (const auto& ft : kv.second) tcells[ft.second];
    for (int32_t q : order) {
        auto it = tcells.find(lab.v[q]);
        if (it != tcells.end()) it->second.push_back(q);
    }
    std::vector<int> stamp(N, -1);
    int attempt = 0;
    int64_t total = 0;
    for (const auto& pk : plan) {
        const int k = pk.second;
        if (total + need > n_quota) break;
        bool done = false;
        for (const auto& ft : by_isl[k]) {
            const int t = ft.second;
            ++attempt;                                          // 同一片里各次重长共用一张「走到过」
            for (int32_t s0 : tcells[t]) {
                if (stamp[s0] == attempt) continue;
                std::vector<int32_t> got;
                using E = std::pair<double, int32_t>;
                std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
                stamp[s0] = attempt;
                heap.push({-g.suit[s0], s0});
                while (!heap.empty() && static_cast<int64_t>(got.size()) < need) {
                    const int32_t q = heap.top().second;
                    heap.pop();
                    got.push_back(q);
                    const int qi = q / W, qj = q % W;
                    for (int di = -1; di <= 1; ++di)
                        for (int dj = -1; dj <= 1; ++dj) {
                            const int a = qi + di, b = qj + dj;
                            if (!(di || dj) || a < 0 || a >= H || b < 0 || b >= W) continue;
                            const int32_t p = a * W + b;
                            if (stamp[p] != attempt && lab.v[p] == t && g.cultivable.v[p] > 0 && !pit[p]) {
                                stamp[p] = attempt;
                                heap.push({-g.suit[p], p});
                            }
                        }
                }
                if (static_cast<int64_t>(got.size()) == need) {
                    out.insert(out.end(), got.begin(), got.end());
                    isl.push_back(k);
                    total += need;
                    done = true;
                    break;
                }
            }
            if (done) break;
        }
    }
}
}  // namespace

void cultivable_land(Group& g, const Config& c, const std::vector<double>& suit, const std::vector<double>& T,
                     const std::vector<double>& soil, const std::vector<double>& wet, const std::vector<double>& near_water) {
    const size_t N = static_cast<size_t>(g.H) * g.W;
    g.cover_natural = g.landcover;
    g.suit = suit;
    const double smax = c.get("landcover.cultivable_slope_max_deg"), tmin = c.get("landcover.cultivable_summer_min_c");
    const double half = 0.5 * g.inp.season_range;      // 最暖的月份 ≈ 年均温 + 半个季节温差
    const double soil_min = c.get("landcover.cultivable_soil_min"), wet_min = c.get("landcover.cultivable_wet_min");
    const double terr = c.get("landcover.terrace_slope_deg");
    g.cultivable = Grid<uint8_t>(g.H, g.W, 0);
    for (size_t k = 0; k < N; ++k) {
        const double s = g.slope.v[k];
        bool ok = suit[k] > 0.0 && s < smax && T[k] + half >= tmin && soil[k] >= soil_min && (wet[k] >= wet_min || near_water[k] > 0.0);
        ok = ok || g.arable.v[k] > 0;
        if (ok) g.cultivable.v[k] = s >= terr ? 2 : 1;
    }
}

FillResult fill_cultivated(Group& g, const Config& c, Rng& rng, const Mask& water, double land_per_hh, int64_t n_quota) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, cell_km2 = res_km * res_km;
    auto sc = [&](const std::string& k) { return c.get("settle." + k); };
    Resources& R = g.res;
    FillResult out;
    // 采场（资源层挑的矿坑 / 硫磺坑 / 盐井）上、岩类赋存（矿化带、露石、硫磺）上不开田
    std::vector<uint8_t> pit(N, 0);
    if (g.has_resources) {
        for (const Working& w : R.works) pit[static_cast<size_t>(w.ci) * W + w.cj] = 1;
        for (size_t k = 0; k < N; ++k)
            if (g.res_field[FK_ORE].v[k] > 0 || g.res_field[FK_SULFUR].v[k] > 0 || g.res_field[FK_STONE].v[k] > 0) pit[k] = 1;
    }
    Mask wet(H, W, 0);                                    // 聚落层动手之前的湿地（P6 的圩田从这里挑）
    for (size_t k = 0; k < N; ++k) wet.v[k] = (g.island_id.v[k] >= 0 && g.landcover.v[k] == LC_WET) ? 1 : 0;
    // ---- 宜垦的连通片（不跨岛）与定居门槛
    Mask cm(H, W, 0);
    for (size_t k = 0; k < N; ++k) cm.v[k] = g.cultivable.v[k] > 0 ? 1 : 0;
    GridI lab;
    const int n_lab = label_by_island(cm, g.island_id, 8, lab);
    std::vector<int64_t> n_cells(static_cast<size_t>(n_lab) + 1, 0);
    for (size_t k = 0; k < N; ++k) n_cells[lab.v[k]]++;
    Mask src = water;
    if (g.has_resources)
        for (const Deposit& d : R.deposits)
            if (d.kind == RK_SPRING) src(d.ci, d.cj) = 1;
    const int cap_w = static_cast<int>(std::ceil(sc("settle_water_km") / res_km - 1e-9));
    const GridI dw = distance_bands(src, cap_w);
    const bool rain_ok = pyround(g.P_mm, 0) >= sc("settle_rain_mm");
    const int reach = static_cast<int>(sc("landing_reach_cells"));
    const float lmax = static_cast<float>(sc("landing_slope_max_deg"));
    Mask flat(H, W, 0);
    for (size_t k = 0; k < N; ++k)
        flat.v[k] = (g.island_id.v[k] >= 0 && !(g.river.v[k] > 0 || g.lake.v[k]) && !g.cliff.v[k] && static_cast<float>(g.slope.v[k]) <= lmax) ? 1 : 0;
    const GridI df = distance_bands(flat, reach);
    std::vector<uint8_t> near_w(static_cast<size_t>(n_lab) + 1, 0), near_f(static_cast<size_t>(n_lab) + 1, 0);
    for (size_t k = 0; k < N; ++k) {
        const int t = lab.v[k];
        if (t <= 0) continue;
        if (dw.v[k] <= cap_w) near_w[t] = 1;
        if (df.v[k] <= reach) near_f[t] = 1;
    }
    const double min_hh = sc("settle_min_hh");
    std::vector<uint8_t> ok_t(static_cast<size_t>(n_lab) + 1, 0);
    for (int t = 1; t <= n_lab; ++t) {
        const bool big = static_cast<double>(n_cells[t]) * cell_km2 >= min_hh * land_per_hh;
        ok_t[t] = ((near_w[t] || rain_ok) && near_f[t] && big) ? 1 : 0;
    }
    // ---- 好地先占：宜垦格按适宜度降序（平局按格号）
    std::vector<int32_t> order;
    for (size_t k = 0; k < N; ++k)
        if (g.cultivable.v[k] > 0 && !pit[k]) order.push_back(static_cast<int32_t>(k));
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return g.suit[a] > g.suit[b]; });
    // ---- 大岛保底：先给 ≥ island_floor_km2 的非主岛各分一块连成片的 island_floor_hh 户的地
    std::vector<int32_t> gcells;
    std::vector<int> g_isl;
    floor_cells(g, c, lab, n_lab, ok_t, order, pit, land_per_hh, n_quota, gcells, g_isl);
    const int64_t n_g = static_cast<int64_t>(gcells.size());
    std::vector<uint8_t> reserved(N, 0);
    for (int32_t k : gcells) reserved[k] = 1;
    // 一遍好地先占：保底的格、圩田的格先占，其余按次序填留下的片（farmland.fill_cultivated 的 fill_pass）
    struct Pass {
        std::vector<int32_t> pick1, in_kept, take;
        std::vector<int64_t> cnt1;
        std::vector<double> hh1;
        std::vector<uint8_t> kept;
        int64_t forced = 0;
    };
    auto fill_pass = [&](const std::vector<int32_t>& pcells) {
        Pass r;
        const int64_t n_r = n_g + static_cast<int64_t>(pcells.size());
        r.pick1 = gcells;
        r.pick1.insert(r.pick1.end(), pcells.begin(), pcells.end());
        for (int32_t k : order) {
            if (static_cast<int64_t>(r.pick1.size()) >= std::max<int64_t>(n_quota, n_r)) break;
            if (ok_t[lab.v[k]] && !reserved[k]) r.pick1.push_back(k);
        }
        r.cnt1.assign(static_cast<size_t>(n_lab) + 1, 0);
        for (int32_t k : r.pick1) r.cnt1[lab.v[k]]++;
        r.hh1.assign(static_cast<size_t>(n_lab) + 1, 0.0);
        r.kept.assign(static_cast<size_t>(n_lab) + 1, 0);
        for (int t = 0; t <= n_lab; ++t) {
            r.hh1[t] = static_cast<double>(r.cnt1[t]) * cell_km2 / land_per_hh;
            r.kept[t] = (ok_t[t] && r.hh1[t] >= min_hh) ? 1 : 0;
        }
        for (int32_t k : gcells) r.kept[lab.v[k]] = 1;
        r.kept[0] = 0;
        std::vector<int32_t> rest_ok, rest_bad;
        for (int32_t k : order) {
            const int t = lab.v[k];
            if (r.kept[t]) {
                if (!reserved[k]) r.in_kept.push_back(k);
            } else if (ok_t[t]) rest_ok.push_back(k);
            else rest_bad.push_back(k);
        }
        r.take = gcells;
        r.take.insert(r.take.end(), pcells.begin(), pcells.end());
        r.take.insert(r.take.end(), r.in_kept.begin(),
                      r.in_kept.begin() + std::min<size_t>(r.in_kept.size(), static_cast<size_t>(std::max<int64_t>(0, n_quota - n_r))));
        if (static_cast<int64_t>(r.take.size()) < n_quota) {    // 留下的片不够额度：按次序补让出来的片，再补不合门槛的片
            for (const auto* v : {&rest_ok, &rest_bad})
                for (int32_t k : *v) {
                    if (static_cast<int64_t>(r.take.size()) >= n_quota) break;
                    r.take.push_back(k);
                    ++r.forced;
                }
        }
        return r;
    };
    Pass ps = fill_pass({});
    // ---- 圩田（P6，polder_plan）：第一遍填到了湿地边上（周围的平地种了过半）的湿地排干围圩，第二遍圩田的格先占（额度之内）
    {
        std::vector<uint8_t> take1(N, 0);
        for (int32_t k : ps.take) take1[k] = 1;
        out.polders = polder_plan(g, c, wet, take1, n_quota - n_g, pit);
    }
    const std::vector<int32_t>& pcells = out.polders.cells;
    const int64_t n_p = static_cast<int64_t>(pcells.size());
    if (n_p) ps = fill_pass(pcells);
    g.polder_id = out.polders.polder_id;
    const std::vector<int32_t>& pick1 = ps.pick1;
    const std::vector<int32_t>& in_kept = ps.in_kept;
    const std::vector<int32_t>& take = ps.take;
    const std::vector<int64_t>& cnt1 = ps.cnt1;
    const std::vector<double>& hh1 = ps.hh1;
    const std::vector<uint8_t>& kept = ps.kept;
    const int64_t forced = ps.forced;
    g.cultivated = Grid<uint8_t>(H, W, 0);
    std::vector<uint8_t> has_cult(static_cast<size_t>(n_lab) + 1, 0);
    for (int32_t k : take) {
        g.cultivated.v[k] = g.cultivable.v[k];
        has_cult[lab.v[k]] = 1;
    }
    for (int32_t k : pcells) g.cultivated.v[k] = 1;       // 圩田：平地、水田
    has_cult[0] = 0;
    // ---- 撂荒：留下的片里接着往外的一层（离在种的田 ≤ fallow_ring_cells）
    const int64_t ring_n = static_cast<int64_t>(std::nearbyint(static_cast<double>(n_quota) * sc("fallow_frac")));
    Mask cmask(H, W, 0);
    for (size_t k = 0; k < N; ++k) cmask.v[k] = g.cultivated.v[k] > 0 ? 1 : 0;
    const Mask near_c = binary_dilate(cmask, static_cast<int>(sc("fallow_ring_cells")));
    std::vector<int32_t> ring;
    for (size_t q = static_cast<size_t>(std::max<int64_t>(0, n_quota - n_g - n_p)); q < in_kept.size(); ++q) {
        if (static_cast<int64_t>(ring.size()) >= ring_n) break;
        if (near_c.v[in_kept[q]]) ring.push_back(in_kept[q]);
    }
    // ---- 废村：让出来的片里头一轮分得最多的几片
    std::vector<int> cand;
    for (int t = 1; t <= n_lab; ++t)
        if (ok_t[t] && !kept[t] && !has_cult[t] && hh1[t] >= sc("ruin_min_hh")) cand.push_back(t);
    std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) {
        if (cnt1[a] != cnt1[b]) return cnt1[a] > cnt1[b];
        return a < b;
    });
    if (cand.size() > static_cast<size_t>(sc("ruin_max"))) cand.resize(static_cast<size_t>(sc("ruin_max")));
    g.fallow_years = Grid<uint8_t>(H, W, 0);
    const std::vector<double>& ry = c.list("settle.ruin_years");
    const int64_t ylo = static_cast<int64_t>(ry[0]), yhi = static_cast<int64_t>(ry[1]);
    int64_t ruin_cells = 0;
    for (int t : cand) {
        RuinTract r;
        r.tract = t;
        r.years = static_cast<int>(rng.integers(ylo, yhi + 1));
        for (int32_t k : pick1)
            if (lab.v[k] == t) r.cells.push_back(k);
        for (int32_t k : r.cells) g.fallow_years.v[k] = static_cast<uint8_t>(std::min(255, r.years));
        r.island = g.island_id.v[r.cells[0]];
        r.households_before = static_cast<int64_t>(std::nearbyint(hh1[t]));
        r.fallow_km2 = pyround(static_cast<double>(r.cells.size()) * cell_km2, 3);
        ruin_cells += static_cast<int64_t>(r.cells.size());
        out.ruins.push_back(std::move(r));
    }
    Mask rmask(H, W, 0);
    for (int32_t k : ring) rmask.v[k] = 1;
    GridI rl;
    const int n_rl = label_by_island(rmask, g.island_id, 8, rl);
    const int64_t ymax = static_cast<int64_t>(sc("fallow_years_max"));
    std::vector<int64_t> ys(static_cast<size_t>(n_rl) + 1, 0);
    for (int q = 1; q <= n_rl; ++q) ys[q] = rng.integers(1, ymax + 1);
    for (int32_t k : ring) g.fallow_years.v[k] = static_cast<uint8_t>(std::min<int64_t>(255, ys[rl.v[k]]));
    // ---- 地表：上等地没人种的回原本的地表；已垦画成可耕地 / 梯田；撂荒按年头
    const int gy = static_cast<int>(sc("fallow_grass_years")), sy = static_cast<int>(sc("fallow_shrub_years"));
    std::vector<int> tids, rids;
    if (g.has_resources)
        for (const Deposit& d : R.deposits) {
            if (d.kind == RK_TIMBER) tids.push_back(d.id);
            if (d.kind == RK_PEAT) rids.push_back(d.id);
        }
    int64_t n_cult = 0, n_fal = 0, n_cv = 0;
    for (size_t k = 0; k < N; ++k) {
        const uint8_t nat = g.cover_natural.v[k];
        const int fy = g.fallow_years.v[k];
        const bool f = fy > 0;
        uint8_t& cv = g.landcover.v[k];
        if (g.arable.v[k] > 0 && g.cultivated.v[k] == 0 && !f) cv = nat;
        if (f) {
            cv = nat;
            if (fy > gy && fy <= sy && (nat == LC_FOREST || nat == LC_SHRUB)) cv = LC_SHRUB;
            if (fy <= gy) cv = LC_GRASS;
        }
        if (g.cultivated.v[k] == 1) cv = LC_ARABLE;
        if (g.cultivated.v[k] == 2) cv = LC_TERRACE;
        if (g.has_resources && (g.cultivated.v[k] > 0 || f) && cv != LC_FOREST) {
            const int32_t p = g.patch_id.v[k];
            if (p >= 0 && std::find(tids.begin(), tids.end(), p) != tids.end()) g.patch_id.v[k] = -1;
        }
        if (g.has_resources && n_p && g.polder_id.v[k] > 0) {         // 排干围成圩田的湿地：从泥炭 / 芦苇荡的片里划掉
            const int32_t p = g.patch_id.v[k];
            if (p >= 0 && std::find(rids.begin(), rids.end(), p) != rids.end()) g.patch_id.v[k] = -1;
        }
        n_cult += g.cultivated.v[k] > 0 ? 1 : 0;
        n_fal += f ? 1 : 0;
        n_cv += g.cultivable.v[k] > 0 ? 1 : 0;
    }
    int64_t n_ok = 0, n_kept = 0, n_drop = 0;
    for (int t = 1; t <= n_lab; ++t) {
        n_ok += ok_t[t];
        n_kept += kept[t];
        n_drop += (ok_t[t] && !kept[t]) ? 1 : 0;
    }
    Json s = Json::obj();
    s.set("quota_km2", pyround(static_cast<double>(n_quota) * cell_km2, 3));
    s.set("cultivable_km2", pyround(static_cast<double>(n_cv) * cell_km2, 3));
    s.set("cultivated_km2", pyround(static_cast<double>(n_cult) * cell_km2, 3));
    s.set("fallow_km2", pyround(static_cast<double>(n_fal) * cell_km2, 3));
    s.set("fallow_ring_km2", pyround(static_cast<double>(ring.size()) * cell_km2, 3));
    s.set("ruin_fallow_km2", pyround(static_cast<double>(ruin_cells) * cell_km2, 3));
    s.set("forced_km2", pyround(static_cast<double>(forced) * cell_km2, 3));
    s.set("n_tracts", static_cast<int64_t>(n_lab));
    s.set("n_tracts_ok", n_ok);
    s.set("n_tracts_kept", n_kept);
    s.set("n_tracts_dropped", n_drop);
    s.set("n_ruins", static_cast<int64_t>(out.ruins.size()));
    s.set("settle_min_hh", static_cast<int64_t>(min_hh));
    s.set("rain_ok", rain_ok);
    s.set("floor_islands", Json::arr_of(g_isl));
    s.set("floor_km2", pyround(static_cast<double>(n_g) * cell_km2, 3));
    s.set("wetland_km2", pyround(static_cast<double>(out.polders.wetland_cells) * cell_km2, 3));
    s.set("polder_km2", pyround(static_cast<double>(n_p) * cell_km2, 3));
    out.summary = std::move(s);
    return out;
}

}  // namespace skyisle::island
