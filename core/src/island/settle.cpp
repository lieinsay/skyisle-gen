// 聚落（settle.py 的 build_settlements / build_links_water / build_homes_city）与层级（tiers.py 的 special_settlements /
// market_towns / landings / clear_forest / village_workings）。运算次序、排序的稳定性、round 的位数照抄 Python 版。
#include "skyisle/island/settle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <set>

#include "skyisle/island/resources.hpp"

namespace skyisle::island {

namespace {

enum { LC_VOID, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE };

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Field {
    int id = 0, island = 0, cells = 0;
    double area_km2 = 0, terrace_frac = 0;
    int cci = 0, ccj = 0;
    double ckx = 0, cky = 0, water_dist_km = 0;
    int64_t households = 0;
    bool has_hh = false;
    int village = 0;                 // >0 村号，<0 −散户号，0 = None
    bool has_village_key = false;
};

struct Village {                     // 村与散户
    bool hamlet = false;
    int id = 0, island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0;
    int64_t households = 0;
    int field = 0;
    double elev_m = 0, water_dist_km = 0, shore_dist_km = 0;
    bool on_arable = false;
    bool seat = false, guo = false, city = false;
    int town = 0;                    // 0 = 不是镇
    bool has_market = false;
    int64_t households_market = 0;
    int market_town = 0;
    bool has_market_town = false;
    int landing = 0;
    bool has_landing = false;
    std::string water_src;
    double water_dist = 0;
    bool has_water = false;
    std::vector<int> workings;
    std::string name() const { return (hamlet ? "hamlet:" : "village:") + std::to_string(id); }
};

struct Special {
    int id = 0;
    std::string kind;                // ore_town / ore_village / floatstone_village / kiln_village / charcoal_camp / hotspring / salt_village
    int resource = -1, occurrence = -1;
    std::string subtype;
    int island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0;
    int64_t households = 0;
    Json note;                       // [代码, 参数…]
    int landing = 0;
    bool has_landing = false;
    int working = -1;
    std::string name() const { return "special:" + kind + ":" + std::to_string(id); }
};

struct Want {
    std::string kind;
    int resource = -1, occurrence = -1;
    std::string subtype;
    int island = 0, ai = 0, aj = 0;
    double hh = 0;
    Json note;
    bool has_cell = false;
    int ci = 0, cj = 0;
};

Json cell_json(int i, int j) { return Json::ipair(i, j); }

}  // namespace

std::vector<int32_t> kmeans_split(Rng& rng, const std::vector<int32_t>& ii, const std::vector<int32_t>& jj, int k, int iters) {
    const size_t n = ii.size();
    std::vector<int32_t> lab(n, 0);
    if (k <= 1 || static_cast<int64_t>(n) <= k) return lab;
    std::vector<double> px(n), py(n);
    for (size_t q = 0; q < n; ++q) {
        px[q] = ii[q];
        py[q] = jj[q];
    }
    const std::vector<int64_t> pick = rng.choice_noreplace(static_cast<int64_t>(n), k);
    std::vector<double> cx(k), cy(k);
    for (int m = 0; m < k; ++m) {
        cx[m] = px[pick[m]];
        cy[m] = py[pick[m]];
    }
    for (int it = 0; it < iters; ++it) {
        for (size_t q = 0; q < n; ++q) {
            double best = INF;
            int bm = 0;
            for (int m = 0; m < k; ++m) {
                const double dx = px[q] - cx[m], dy = py[q] - cy[m];
                const double d = dx * dx + dy * dy;
                if (d < best) {
                    best = d;
                    bm = m;
                }
            }
            lab[q] = bm;
        }
        // ctr[m] = pts[sel].mean(axis=0)：numpy 沿 axis 0 的和是逐行顺序加
        std::vector<double> sx(k, 0.0), sy(k, 0.0);
        std::vector<int64_t> cnt(k, 0);
        std::vector<uint8_t> first(k, 1);
        for (size_t q = 0; q < n; ++q) {
            const int m = lab[q];
            if (first[m]) {
                sx[m] = px[q];
                sy[m] = py[q];
                first[m] = 0;
            } else {
                sx[m] += px[q];
                sy[m] += py[q];
            }
            cnt[m]++;
        }
        for (int m = 0; m < k; ++m)
            if (cnt[m]) {
                cx[m] = sx[m] / static_cast<double>(cnt[m]);
                cy[m] = sy[m] / static_cast<double>(cnt[m]);
            }
    }
    return lab;
}

void build_settlements(Group& g, const PlanetView& pv, const Config& c) {
    const double t0 = now_s();
    const NodeInputs& inp = g.inp;
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, res_m = res_km * 1000.0, cell_km2 = res_km * res_km;
    const int n_isl = static_cast<int>(g.islands.size());
    const std::string P = "settle.";
    auto sc = [&](const std::string& k) { return c.get(P + k); };
    const double x0 = g.origin_x, y0 = g.origin_y;
    auto kmx = [&](double j) { return pyround(x0 + (j + 0.5) * res_km, 3); };
    auto kmy = [&](double i) { return pyround(y0 - (i + 0.5) * res_km, 3); };
    // 被舍的数是 numpy 标量时（numpy 数组的均值 / 元素、numpy 整数参与的算式）round 是 rint(x·1000)/1000
    auto kmx_np = [&](double j, bool np_) { return np_ ? npround(x0 + (j + 0.5) * res_km, 3) : kmx(j); };
    auto kmy_np = [&](double i, bool np_) { return np_ ? npround(y0 - (i + 0.5) * res_km, 3) : kmy(i); };
    auto kmj = [&](double i, double j) { return Json::pair(kmx(j), kmy(i)); };
    Resources& R = g.res;

    Mask land(H, W, 0), arable(H, W, 0), water(H, W, 0), river(H, W, 0);
    std::vector<double> slope(N);
    for (size_t k = 0; k < N; ++k) {
        land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
        arable.v[k] = g.arable.v[k] > 0 ? 1 : 0;
        river.v[k] = g.river.v[k] > 0 ? 1 : 0;
        water.v[k] = (river.v[k] || g.lake.v[k] || g.stream.v[k] > 0) ? 1 : 0;
        slope[k] = static_cast<double>(static_cast<float>(g.slope.v[k]));
    }
    // 漫滩：河两格内坡 < 3°
    Mask flood(H, W, 0);
    {
        const Mask r2 = binary_dilate(river, 2);
        for (size_t k = 0; k < N; ++k) flood.v[k] = (r2.v[k] && slope[k] < 3.0 && !river.v[k]) ? 1 : 0;
    }
    const int wnear = static_cast<int>(sc("water_near_cells")), snear = static_cast<int>(sc("shore_near_cells"));
    const GridI dist_water = distance_bands(water, wnear);
    const GridI dist_shore = distance_bands(g.cliff, snear);
    // 迎风性（settle._wind_exposure）
    std::vector<double> expo(N, 0.0);
    {
        const double u = g.wind_u, v = g.wind_v;
        const double sp = py_hypot(u, v);
        if (sp >= 1e-6) {
            const double ux = -u / sp, uy = -v / sp;
            auto hh = [&](int i, int j) { return land(i, j) ? g.height(i, j) : NaN; };
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    const double self = hh(i, j);
                    auto nb = [&](int di, int dj) {
                        const int a = i - di, b = j - dj;
                        if (a < 0 || b < 0 || a >= H || b >= W) return self;
                        const double x = hh(a, b);
                        return std::isnan(x) ? self : x;
                    };
                    const double gx = (nb(0, -1) - nb(0, 1)) / (2 * res_m);
                    const double gy = (nb(-1, 0) - nb(1, 0)) / (2 * res_m);
                    const double gn = np_hypot(gx, gy);
                    const double dx = gn > 1e-9 ? gx / std::max(gn, 1e-9) : 0.0;
                    const double dy = gn > 1e-9 ? gy / std::max(gn, 1e-9) : 0.0;
                    const double e = clip(dx * ux + dy * uy, -1.0, 1.0) * clip(gn * 1000.0 / 0.26, 0.0, 1.0);
                    expo[static_cast<size_t>(i) * W + j] = std::isnan(e) ? 0.0 : e;
                }
        }
    }

    // ---------- 人口 → 户 ----------
    int64_t n_arable = 0;
    for (uint8_t x : arable.v) n_arable += x;
    const double arable_km2 = static_cast<double>(n_arable) * cell_km2;
    const bool pop_polity = !std::isnan(inp.pop);
    const double pop = pop_polity ? inp.pop : arable_km2 * inp.people_per_arable_km2;
    g.settle_pop = pop;
    const double hh_size = sc("household_size");
    const int64_t hh_total = static_cast<int64_t>(std::nearbyint(pop / hh_size));
    std::vector<double> ar_isl(n_isl, 0.0);
    {
        std::vector<int64_t> cnt(n_isl, 0);
        for (size_t k = 0; k < N; ++k)
            if (arable.v[k] && g.island_id.v[k] >= 0) cnt[g.island_id.v[k]]++;
        for (int k = 0; k < n_isl; ++k) ar_isl[k] = static_cast<double>(cnt[k]) * cell_km2;
    }
    const double ar_sum = np_sum(ar_isl.data(), ar_isl.size());
    const int64_t nonfarm_hh = ar_sum > 0 ? static_cast<int64_t>(std::nearbyint(static_cast<double>(hh_total) * sc("nonfarm_share"))) : 0;
    const int64_t farm_total = hh_total - nonfarm_hh;
    std::vector<int64_t> hh_isl(n_isl, 0);
    for (int k = 0; k < n_isl; ++k) {
        const double share = ar_sum > 0 ? ar_isl[k] / std::max(1e-9, ar_sum) : (k == 0 ? 1.0 : 0.0);
        hh_isl[k] = static_cast<int64_t>(std::floor(share * static_cast<double>(farm_total)));
    }
    int richest = 0;
    for (int k = 1; k < n_isl; ++k)
        if (ar_isl[k] > ar_isl[richest]) richest = k;
    {
        int64_t s = 0;
        for (int64_t x : hh_isl) s += x;
        hh_isl[richest] += farm_total - s;
    }
    for (int k = 0; k < n_isl; ++k)
        if (ar_isl[k] <= 0 && hh_isl[k] > 0 && k != richest) {
            hh_isl[richest] += hh_isl[k];
            hh_isl[k] = 0;
        }
    const double land_per_hh = arable_km2 / static_cast<double>(std::max<int64_t>(1, farm_total));

    // ---------- 田块：可耕地 8 邻域连通块（不跨岛）；大块按 80 户（邑治 300）k-means 切分 ----------
    GridI lab;
    const int n_lab = label_by_island(arable, g.island_id, 8, lab);
    GridI fields_raster(H, W, 0);
    std::vector<Field> fields;
    {
        Rng rng = part_rng(inp, "settle:fields");
        const int cap_village = static_cast<int>(sc("village_max_hh")), cap_seat = static_cast<int>(sc("seat_max_hh"));
        std::vector<std::vector<int32_t>> cl(static_cast<size_t>(n_lab) + 1);
        for (size_t k = 0; k < N; ++k)
            if (lab.v[k]) cl[lab.v[k]].push_back(static_cast<int32_t>(k));
        int main_biggest = -1;
        size_t best_cnt = 0;
        for (int L = 1; L <= n_lab; ++L) {
            if (g.island_id.v[cl[L][0]] != 0) continue;
            if (main_biggest < 0 || cl[L].size() > best_cnt || (cl[L].size() == best_cnt && L > main_biggest)) {
                best_cnt = cl[L].size();
                main_biggest = L;
            }
        }
        int fid = 0;
        for (int L = 1; L <= n_lab; ++L) {
            const auto& cells = cl[L];
            std::vector<int32_t> ii(cells.size()), jj(cells.size());
            for (size_t q = 0; q < cells.size(); ++q) {
                ii[q] = cells[q] / W;
                jj[q] = cells[q] % W;
            }
            const int k_isl = g.island_id.v[cells[0]];
            const double area = static_cast<double>(cells.size()) * cell_km2;
            const double hh_est = area / std::max(1e-9, land_per_hh);
            const int cap = L == main_biggest ? cap_seat : cap_village;
            const int k = std::max(1, static_cast<int>(std::ceil(hh_est / cap)));
            const std::vector<int32_t> parts = kmeans_split(rng, ii, jj, k);
            for (int m = 0; m < k; ++m) {
                std::vector<int32_t> sel;
                for (size_t q = 0; q < cells.size(); ++q)
                    if (parts[q] == m) sel.push_back(static_cast<int32_t>(q));
                if (sel.empty()) continue;
                ++fid;
                int64_t si = 0, sj = 0, terr = 0;
                int32_t dmin = INT32_MAX;
                for (int32_t q : sel) {
                    fields_raster.v[cells[q]] = fid;
                    si += ii[q];
                    sj += jj[q];
                    terr += g.arable.v[cells[q]] == 2 ? 1 : 0;
                    dmin = std::min(dmin, dist_water.v[cells[q]]);
                }
                const double n = static_cast<double>(sel.size());
                const double mi = static_cast<double>(si) / n, mj = static_cast<double>(sj) / n;
                Field f;
                f.id = fid;
                f.island = k_isl;
                f.cells = static_cast<int>(sel.size());
                f.area_km2 = pyround(n * cell_km2, 3);
                f.terrace_frac = pyround(static_cast<double>(terr) / n, 3);
                f.cci = static_cast<int>(std::nearbyint(mi));
                f.ccj = static_cast<int>(std::nearbyint(mj));
                f.ckx = kmx_np(mj, true);      // km(ii[sel].mean(), jj[sel].mean())：numpy 标量
                f.cky = kmy_np(mi, true);
                f.water_dist_km = pyround(static_cast<double>(dmin) * res_km, 2);
                fields.push_back(f);
            }
        }
    }
    // 户数按岛内田块面积分配（最大余数法）
    for (int k = 0; k < n_isl; ++k) {
        std::vector<Field*> fs;
        for (Field& f : fields)
            if (f.island == k) fs.push_back(&f);
        if (fs.empty()) continue;
        std::vector<double> areas;
        for (Field* f : fs) areas.push_back(f->area_km2);
        const double tot = py_sum(areas);          // Python 的 sum()：Neumaier 补偿求和
        std::vector<double> raw(fs.size());
        std::vector<int64_t> base(fs.size());
        int64_t sb = 0;
        for (size_t q = 0; q < fs.size(); ++q) {
            raw[q] = static_cast<double>(hh_isl[k]) * fs[q]->area_km2 / std::max(1e-9, tot);
            base[q] = static_cast<int64_t>(std::floor(raw[q]));
            sb += base[q];
        }
        const int64_t rem = hh_isl[k] - sb;
        std::vector<size_t> order(fs.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return -(raw[a] - base[a]) < -(raw[b] - base[b]); });
        for (int64_t q = 0; q < std::max<int64_t>(0, rem) && q < static_cast<int64_t>(order.size()); ++q) base[order[q]] += 1;
        for (size_t q = 0; q < fs.size(); ++q) {
            fs[q]->households = base[q];
            fs[q]->has_hh = true;
        }
    }

    // ---------- 村址评分 ----------
    const double smax = sc("site_slope_max_deg");
    Mask ok_site(H, W, 0);
    std::vector<double> score_base(N);
    {
        const double ww = sc("site_weights.water"), wsl = sc("site_weights.slope"), wl = sc("site_weights.lee"), wsh = sc("site_weights.shore");
        for (size_t k = 0; k < N; ++k) {
            ok_site.v[k] = (land.v[k] && !arable.v[k] && !g.cliff.v[k] && !water.v[k] && !flood.v[k] && slope[k] < smax) ? 1 : 0;
            score_base[k] = ww * clip(1.0 - dist_water.v[k] / (wnear + 1.0), 0.0, 1.0) + wsl * clip(1.0 - slope[k] / smax, 0.0, 1.0) +
                            wl * (1.0 - expo[k]) / 2.0 + wsh * clip(1.0 - dist_shore.v[k] / (snear + 1.0), 0.0, 1.0);
        }
    }
    const int reach = static_cast<int>(sc("site_reach_cells"));
    const int min_sep = static_cast<int>(std::nearbyint(sc("village_min_sep_km") / res_km));
    const double w_field = sc("site_weights.field");
    std::vector<Village> villages, hamlets;
    Mask taken(H, W, 0);
    const int64_t min_v = static_cast<int64_t>(sc("village_min_hh"));
    {
        std::vector<std::vector<int32_t>> cells_of(fields.size() + 1);
        for (size_t k = 0; k < N; ++k)
            if (fields_raster.v[k]) cells_of[fields_raster.v[k]].push_back(static_cast<int32_t>(k));
        std::vector<size_t> ford(fields.size());
        std::iota(ford.begin(), ford.end(), 0);
        std::stable_sort(ford.begin(), ford.end(), [&](size_t a, size_t b) { return -fields[a].households < -fields[b].households; });
        for (size_t fi : ford) {
            Field& f = fields[fi];
            const int64_t hh = f.households;
            if (hh <= 0) continue;
            const auto& cells = cells_of[f.id];
            int imin = H, imax = -1, jmin = W, jmax = -1;
            for (int32_t q : cells) {
                imin = std::min(imin, q / W);
                imax = std::max(imax, q / W);
                jmin = std::min(jmin, q % W);
                jmax = std::max(jmax, q % W);
            }
            const int r0 = std::max(0, imin - reach), r1 = std::min(H, imax + reach + 1);
            const int c0 = std::max(0, jmin - reach), c1 = std::min(W, jmax + reach + 1);
            const int h = r1 - r0, w = c1 - c0;
            Mask fm(h, w, 0);
            for (int32_t q : cells) fm(q / W - r0, q % W - c0) = 1;
            const GridI dfield = distance_bands(fm, reach);
            std::vector<uint8_t> cand(static_cast<size_t>(h) * w, 0);
            bool any = false;
            for (int a = 0; a < h; ++a)
                for (int b = 0; b < w; ++b) {
                    const size_t gk = static_cast<size_t>(a + r0) * W + (b + c0);
                    const uint8_t v = (ok_site.v[gk] && dfield(a, b) <= reach && g.island_id.v[gk] == f.island) ? 1 : 0;
                    cand[static_cast<size_t>(a) * w + b] = v;
                    any = any || v;
                }
            if (!any)
                for (int a = 0; a < h; ++a)
                    for (int b = 0; b < w; ++b) {
                        const size_t gk = static_cast<size_t>(a + r0) * W + (b + c0);
                        const uint8_t v = (g.island_id.v[gk] == f.island && land.v[gk] && !water.v[gk] && dfield(a, b) <= reach) ? 1 : 0;
                        cand[static_cast<size_t>(a) * w + b] = v;
                        any = any || v;
                    }
            if (!any)
                for (size_t q = 0; q < cand.size(); ++q) cand[q] = fm.v[q];
            std::vector<double> scm(cand.size());
            for (int a = 0; a < h; ++a)
                for (int b = 0; b < w; ++b) {
                    const size_t q = static_cast<size_t>(a) * w + b;
                    const size_t gk = static_cast<size_t>(a + r0) * W + (b + c0);
                    scm[q] = cand[q] ? score_base[gk] + w_field * (1.0 - dfield(a, b) / (reach + 1.0)) : -1e9;
                }
            std::vector<int32_t> ord(scm.size());
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int32_t x, int32_t y) { return -scm[x] < -scm[y]; });
            // Python 版 r0 = max(0, ii.min() − reach) 在 > 0 时是 numpy 整数、否则是 Python 的 0（r1 / c0 / c1 同理）；
            // divmod(p, c1 − c0) 与 gi = i + r0 随之成 numpy 整数，km 的 round 就按 numpy 的来
            const bool r0_np = imin - reach > 0, c0_np = jmin - reach > 0, c1_np = jmax + reach + 1 < W;
            const bool w_np = c0_np || c1_np;
            bool gi_np = false, gj_np = false;
            int pi = -1, pj = -1;
            for (size_t t = 0; t < ord.size() && t < 200; ++t) {
                const int p = ord[t];
                const int a = p / w, b = p % w;
                if (scm[p] <= -1e8) break;
                const int gi = a + r0, gj = b + c0;
                if (hh >= min_v) {
                    bool hit = false;
                    for (int x = std::max(0, gi - min_sep); x < std::min(H, gi + min_sep + 1) && !hit; ++x)
                        for (int y = std::max(0, gj - min_sep); y < std::min(W, gj + min_sep + 1); ++y)
                            if (taken(x, y)) {
                                hit = true;
                                break;
                            }
                    if (hit) continue;
                }
                pi = gi;
                pj = gj;
                gi_np = w_np || r0_np;
                gj_np = w_np || c0_np;
                break;
            }
            if (pi < 0) {
                for (int p : ord) {
                    const int a = p / w, b = p % w;
                    if (scm[p] <= -1e8) break;
                    if (!taken(a + r0, b + c0)) {
                        pi = a + r0;
                        pj = b + c0;
                        gi_np = w_np || r0_np;
                        gj_np = w_np || c0_np;
                        break;
                    }
                }
            }
            if (pi < 0) {
                // 所有候选格都被占：落在田块里离已有村最远的格
                std::vector<int32_t> tk;
                for (size_t k = 0; k < N; ++k)
                    if (taken.v[k]) tk.push_back(static_cast<int32_t>(k));
                size_t b = 0;
                if (!tk.empty()) {
                    double best = -INF;
                    for (size_t q = 0; q < cells.size(); ++q) {
                        double dmin = INF;
                        const double fi2 = cells[q] / W, fj2 = cells[q] % W;
                        for (int32_t t : tk) {
                            const double di = fi2 - t / W, dj = fj2 - t % W;
                            dmin = std::min(dmin, std::sqrt(di * di + dj * dj));
                        }
                        if (dmin > best) {
                            best = dmin;
                            b = q;
                        }
                    }
                }
                pi = cells[b] / W;
                pj = cells[b] % W;
            }
            Village r;
            r.island = f.island;
            r.ci = pi;
            r.cj = pj;
            r.kx = kmx_np(pj, gj_np);
            r.ky = kmy_np(pi, gi_np);
            r.households = hh;
            r.field = f.id;
            r.elev_m = pyround(g.height(pi, pj), 0);
            r.water_dist_km = pyround(static_cast<double>(dist_water(pi, pj)) * res_km, 2);
            r.shore_dist_km = pyround(static_cast<double>(dist_shore(pi, pj)) * res_km, 2);
            r.on_arable = arable(pi, pj) != 0;
            f.village = 0;
            f.has_village_key = true;
            if (hh >= min_v) {
                taken(pi, pj) = 1;
                villages.push_back(r);
            } else {
                r.hamlet = true;
                hamlets.push_back(r);
            }
        }
    }
    std::stable_sort(villages.begin(), villages.end(), [](const Village& a, const Village& b) {
        if (a.households != b.households) return a.households > b.households;
        if (a.island != b.island) return a.island < b.island;
        if (a.ci != b.ci) return a.ci < b.ci;
        return a.cj < b.cj;
    });
    for (size_t k = 0; k < villages.size(); ++k) villages[k].id = static_cast<int>(k) + 1;
    for (size_t k = 0; k < hamlets.size(); ++k) hamlets[k].id = static_cast<int>(k) + 1;
    {
        std::map<int, int> vf, hf;
        for (const Village& r : villages) vf[r.field] = r.id;
        for (const Village& r : hamlets) hf[r.field] = -r.id;
        for (Field& f : fields) {
            f.has_village_key = true;
            auto it = vf.find(f.id);
            if (it != vf.end()) f.village = it->second;
            else {
                auto jt = hf.find(f.id);
                f.village = jt != hf.end() ? jt->second : 0;
            }
        }
    }
    int seat = -1;
    for (size_t k = 0; k < villages.size(); ++k)
        if (villages[k].island == 0) {
            seat = static_cast<int>(k);
            break;
        }
    if (seat < 0 && !villages.empty()) seat = 0;
    if (seat >= 0) villages[seat].seat = true;

    // ---------- 聚落层级（tiers.py）：专业聚落 → 集镇 → 泊场 ----------
    std::vector<Special> specials;
    if (g.has_resources && nonfarm_hh > 0) {
        const int sreach = static_cast<int>(std::nearbyint(1.0 / res_km));
        Mask forest(H, W, 0);
        for (size_t k = 0; k < N; ++k) forest.v[k] = g.landcover.v[k] == LC_FOREST ? 1 : 0;
        const Mask near_forest = binary_dilate(forest, std::max(1, static_cast<int>(std::nearbyint(sc("kiln_forest_km") / res_km))));
        std::vector<Want> want;
        auto gw = [](const std::string& gr) { return gr == "hi" ? 1.5 : (gr == "mid" ? 1.0 : (gr == "lo" ? 0.5 : 1.0)); };
        for (const Occurrence& o : R.occ) {
            if (o.kind != RK_ORE) continue;
            const double hh = sc("ore_hh_per_km2") * o.area_km2 * gw(o.grade);
            std::vector<const Working*> pits;
            for (const Working& wk : R.works)
                if (wk.occurrence == o.id) pits.push_back(&wk);
            Want w;
            w.kind = hh >= sc("mine_town_hh") ? "ore_town" : "ore_village";
            w.occurrence = o.id;
            w.subtype = o.subtype;
            w.island = o.island;
            w.ai = pits.empty() ? o.ci : pits[0]->ci;
            w.aj = pits.empty() ? o.cj : pits[0]->cj;
            w.hh = hh;
            w.note = Json::arr();
            w.note.push("ore");
            w.note.push(o.subtype);
            w.note.push(o.area_km2);
            w.note.push(o.grade);
            w.note.push(static_cast<int64_t>(pits.size()));
            want.push_back(w);
        }
        {
            std::map<int, std::vector<const Deposit*>> by_isl;
            for (const Deposit& d : R.deposits)
                if (d.kind == RK_FLOATSTONE && d.area_km2 >= sc("floatstone_site_min_km2")) by_isl[d.island].push_back(&d);
            for (auto& kv : by_isl) {
                auto& lst = kv.second;
                std::stable_sort(lst.begin(), lst.end(), [](const Deposit* a, const Deposit* b) { return -a->area_km2 < -b->area_km2; });
                const size_t lim = kv.first == 0 ? static_cast<size_t>(sc("floatstone_sites_main")) : 1;
                for (size_t q = 0; q < lst.size() && q < lim; ++q) {
                    const Deposit* d = lst[q];
                    Want w;
                    w.kind = "floatstone_village";
                    w.resource = d->id;
                    w.subtype = d->subtype;
                    w.island = kv.first;
                    w.ai = d->ci;
                    w.aj = d->cj;
                    w.hh = std::min(60.0, sc("floatstone_hh_per_km2") * d->area_km2);
                    w.note = Json::arr();
                    w.note.push("floatstone");
                    w.note.push(d->subtype);
                    w.note.push(d->area_km2);
                    want.push_back(w);
                }
            }
        }
        {
            std::vector<const Occurrence*> clays;
            for (const Occurrence& o : R.occ)
                if (o.kind == RK_CLAY) clays.push_back(&o);
            std::stable_sort(clays.begin(), clays.end(), [](const Occurrence* a, const Occurrence* b) {
                if (a->grade_peak != b->grade_peak) return -a->grade_peak < -b->grade_peak;
                if (a->area_km2 != b->area_km2) return -a->area_km2 < -b->area_km2;
                return a->id < b->id;
            });
            std::map<int, int> n_kiln;
            for (const Occurrence* o : clays) {
                if (!(o->grade == "hi" && near_forest(o->ci, o->cj))) continue;
                const int k = o->island;
                if (n_kiln[k] >= (k == 0 ? 2 : 1)) continue;
                n_kiln[k] += 1;
                Want w;
                w.kind = "kiln_village";
                w.occurrence = o->id;
                w.subtype = o->subtype;
                w.island = k;
                w.ai = o->ci;
                w.aj = o->cj;
                w.hh = sc("kiln_hh");
                w.note = Json::arr();
                w.note.push("kiln");
                w.note.push(o->subtype);
                w.note.push(o->area_km2);
                want.push_back(w);
            }
        }
        {
            std::map<int, int> n_ch;
            const double far = sc("charcoal_far_km") / res_km;
            std::vector<const Deposit*> deps;
            for (const Deposit& d : R.deposits) deps.push_back(&d);
            std::stable_sort(deps.begin(), deps.end(), [](const Deposit* a, const Deposit* b) { return -a->area_km2 < -b->area_km2; });
            for (const Deposit* d : deps) {
                if (d->kind != RK_TIMBER || d->area_km2 < sc("charcoal_min_km2")) continue;
                double dv = 1e9;
                if (!villages.empty()) {
                    double m = INF;
                    for (const Village& v : villages) {
                        const double di = static_cast<double>(v.ci) - d->ci, dj = static_cast<double>(v.cj) - d->cj;
                        m = std::min(m, std::sqrt(di * di + dj * dj));
                    }
                    dv = m;
                }
                const int k = d->island;
                if (dv <= far || n_ch[k] >= 3) continue;
                n_ch[k] += 1;
                Want w;
                w.kind = "charcoal_camp";
                w.resource = d->id;
                w.subtype = d->subtype;
                w.island = k;
                w.ai = d->ci;
                w.aj = d->cj;
                w.hh = clip(d->area_km2 * 0.5, 5, 30);
                w.note = Json::arr();
                w.note.push("charcoal");
                w.note.push(d->subtype);
                w.note.push(d->area_km2);
                w.note.push(dv * res_km);
                want.push_back(w);
            }
        }
        for (const Deposit& d : R.deposits)
            if (d.kind == RK_HOTSPRING) {
                Want w;
                w.kind = "hotspring";
                w.resource = d.id;
                w.island = d.island;
                w.ai = d.ci;
                w.aj = d.cj;
                w.hh = sc("hotspring_hh");
                w.note = Json::arr();
                w.note.push("hotspring");
                want.push_back(w);
            }
        // 盐井村（P3）：每个开了盐井的岩盐区一个，落在盐井旁——汲卤煮盐要常年的人手和柴，盐是头等的货、外运，吃粮靠外运
        for (const Occurrence& o : R.occ) {
            if (o.kind != RK_SALT) continue;
            const Working* well = nullptr;
            for (const Working& wk : R.works)
                if (wk.occurrence == o.id) {
                    well = &wk;
                    break;
                }
            if (!well) continue;
            Want w;
            w.kind = "salt_village";
            w.occurrence = o.id;
            w.subtype = o.subtype;
            w.island = o.island;
            w.ai = well->ci;
            w.aj = well->cj;
            w.hh = std::min(sc("salt_hh_max"), sc("salt_hh_per_km2") * o.area_km2 * gw(o.grade));
            w.note = Json::arr();
            w.note.push("salt");
            w.note.push(o.subtype);
            w.note.push(o.area_km2);
            w.note.push(o.grade);
            want.push_back(w);
        }
        if (!want.empty()) {
            // _near_site：cell 附近同岛可落脚的格里最近的；没有就退到非水非崖的陆地；再没有就不设
            for (Want& w : want) {
                const int i = w.ai, j = w.aj, k = w.island;
                const int r0 = std::max(0, i - sreach), r1 = std::min(H, i + sreach + 1), c0 = std::max(0, j - sreach), c1 = std::min(W, j + sreach + 1);
                for (int pass = 0; pass < 2 && !w.has_cell; ++pass) {
                    double best = INF;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            const bool m = pass == 0 ? ok_site.v[q] != 0
                                                     : (land.v[q] && !g.cliff.v[q] && !g.lake.v[q] && !(g.river.v[q] > 0));
                            if (!(m && g.island_id.v[q] == k)) continue;
                            const double d = static_cast<double>((a - i) * (a - i) + (b - j) * (b - j));
                            if (d < best) {
                                best = d;
                                w.ci = a;
                                w.cj = b;
                                w.has_cell = true;
                            }
                        }
                }
            }
            std::vector<Want> kept;
            for (Want& w : want)
                if (w.has_cell) kept.push_back(w);
            const double cap = sc("special_cap_frac") * static_cast<double>(nonfarm_hh);
            std::vector<double> hhs;
            for (const Want& w : kept) hhs.push_back(w.hh);
            const double tot = py_sum(hhs);
            const double f = std::min(1.0, cap / std::max(1e-9, tot));
            for (const Want& w : kept) {
                const int64_t hh = static_cast<int64_t>(std::floor(w.hh * f));
                if (hh < static_cast<int64_t>(sc("special_min_hh"))) continue;
                Special s;
                s.id = static_cast<int>(specials.size()) + 1;
                s.kind = w.kind;
                s.resource = w.resource;
                s.occurrence = w.occurrence;
                s.subtype = w.subtype;
                s.island = w.island;
                s.ci = w.ci;
                s.cj = w.cj;
                s.kx = kmx(w.cj);
                s.ky = kmy(w.ci);
                s.households = hh;
                s.note = w.note;
                specials.push_back(s);
            }
            for (const Special& s : specials)
                if (s.kind == "ore_town" || s.kind == "ore_village" || s.kind == "salt_village")
                    for (Working& wk : R.works)
                        if (wk.occurrence == s.occurrence) wk.special = s.id;
        }
    }
    int64_t rest = nonfarm_hh;
    for (const Special& s : specials) rest -= s.households;
    // 集镇（market_towns）：按服务半径内户数贪心挑村升镇，镇距 ≥ town_spacing_km，邑治先入；非农户余量按服务户数分
    std::vector<Json> towns_json;
    if (!villages.empty()) {
        const size_t nv = villages.size();
        std::vector<double> D(nv * nv);
        for (size_t a = 0; a < nv; ++a)
            for (size_t b = 0; b < nv; ++b) {
                const double dx = villages[a].kx - villages[b].kx, dy = villages[a].ky - villages[b].ky;
                D[a * nv + b] = std::sqrt(dx * dx + dy * dy);
            }
        const double rad = sc("market_radius_km");
        std::vector<double> cen(nv, 0.0);
        for (size_t a = 0; a < nv; ++a) {
            double s = 0;
            for (size_t b = 0; b < nv; ++b) s += D[a * nv + b] <= rad ? static_cast<double>(villages[b].households) : 0.0;
            cen[a] = s;
        }
        const int idx_seat = seat >= 0 ? seat : static_cast<int>(std::max_element(cen.begin(), cen.end()) - cen.begin());
        std::vector<int> chosen{idx_seat};
        std::vector<int> ord(nv);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return -cen[a] < -cen[b]; });
        const double min_srv = sc("town_min_served_hh"), spacing = sc("town_spacing_km");
        for (int t : ord) {
            if (std::find(chosen.begin(), chosen.end(), t) != chosen.end() || cen[t] < min_srv) continue;
            double dm = INF;
            for (int ch : chosen) dm = std::min(dm, D[static_cast<size_t>(t) * nv + ch]);
            if (dm >= spacing) chosen.push_back(t);
        }
        std::vector<int> near(nv);
        for (size_t a = 0; a < nv; ++a) {
            size_t bq = 0;
            for (size_t q = 1; q < chosen.size(); ++q)
                if (D[a * nv + chosen[q]] < D[a * nv + chosen[bq]]) bq = q;
            near[a] = chosen[bq];
        }
        std::vector<double> served(chosen.size(), 0.0);
        for (size_t q = 0; q < chosen.size(); ++q)
            for (size_t a = 0; a < nv; ++a)
                if (near[a] == chosen[q]) served[q] += static_cast<double>(villages[a].households);
        const double ssum = py_sum(served);
        std::vector<double> raw(chosen.size());
        std::vector<int64_t> base(chosen.size());
        int64_t sb = 0;
        for (size_t q = 0; q < chosen.size(); ++q) {
            raw[q] = static_cast<double>(rest) * served[q] / std::max(1e-9, ssum);
            base[q] = static_cast<int64_t>(std::floor(raw[q]));
            sb += base[q];
        }
        std::vector<size_t> ro(chosen.size());
        std::iota(ro.begin(), ro.end(), 0);
        std::stable_sort(ro.begin(), ro.end(), [&](size_t a, size_t b) { return -(raw[a] - base[a]) < -(raw[b] - base[b]); });
        for (int64_t q = 0; q < std::max<int64_t>(0, rest - sb) && q < static_cast<int64_t>(ro.size()); ++q) base[ro[q]] += 1;
        for (size_t n = 0; n < chosen.size(); ++n) {
            Village& v = villages[chosen[n]];
            v.town = static_cast<int>(n) + 1;
            v.has_market = true;
            v.households_market = base[n];
            int sv = 0;
            double mx = -INF;
            for (size_t a = 0; a < nv; ++a)
                if (near[a] == chosen[n]) {
                    ++sv;
                    mx = std::max(mx, D[a * nv + chosen[n]]);
                }
            Json t = Json::obj();
            t.set("id", static_cast<int64_t>(n) + 1);
            t.set("name", chosen[n] == idx_seat ? std::string("seat") : "town:" + std::to_string(n + 1));
            t.set("village", v.id);
            t.set("island", v.island);
            t.set("cell", cell_json(v.ci, v.cj));
            t.set("km", Json::pair(v.kx, v.ky));
            t.set("households_farm", v.households);
            t.set("households_market", base[n]);
            t.set("households", v.households + base[n]);
            t.set("served_villages", sv);
            t.set("served_households", static_cast<int64_t>(served[n]));
            t.set("max_served_km", pyround(mx, 2));
            t.set("seat", chosen[n] == idx_seat);
            towns_json.push_back(t);
        }
        for (size_t a = 0; a < nv; ++a) {
            villages[a].market_town = villages[near[a]].town;
            villages[a].has_market_town = true;
        }
    }
    if (villages.empty() && rest > 0) {
        if (!hamlets.empty()) {
            size_t b = 0;
            for (size_t q = 1; q < hamlets.size(); ++q)
                if (hamlets[q].households > hamlets[b].households) b = q;
            hamlets[b].households += rest;
        }
        rest = 0;
    }
    // 泊场（landings）：reach 格内同岛、坡 ≤ landing_slope_max_deg、非水非崖的格，非田非林优先、近者优先
    struct LandTarget {
        std::string kind, of;
        int island, ci, cj;
        Village* v;
        Special* s;
    };
    std::vector<LandTarget> lst;
    for (Village& v : villages) lst.push_back({v.seat ? "main" : (v.town ? "town" : "village"), v.name(), v.island, v.ci, v.cj, &v, nullptr});
    for (Special& s : specials) lst.push_back({"special:" + s.kind, s.name(), s.island, s.ci, s.cj, nullptr, &s});
    std::vector<Json> lands;
    std::vector<std::array<int, 3>> land_cells;   // [岛, 行, 列]
    int main_dock = -1;
    std::map<int, int> docks_of;                  // 岛 → 该岛第一块泊场（下标）
    {
        const float lmax = static_cast<float>(sc("landing_slope_max_deg"));
        const int lreach = static_cast<int>(sc("landing_reach_cells"));
        for (const LandTarget& t : lst) {
            const int i = t.ci, j = t.cj, k = t.island;
            const int r0 = std::max(0, i - lreach), r1 = std::min(H, i + lreach + 1), c0 = std::max(0, j - lreach), c1 = std::min(W, j + lreach + 1);
            int bi = -1, bj = -1;
            bool flat_ok = false;
            double best = INF;
            for (int a = r0; a < r1; ++a)
                for (int b = c0; b < c1; ++b) {
                    const size_t q = static_cast<size_t>(a) * W + b;
                    const bool wtr = g.river.v[q] > 0 || g.lake.v[q];
                    const bool fl = land.v[q] && !wtr && !g.cliff.v[q] && static_cast<float>(g.slope.v[q]) <= lmax;
                    if (!(fl && g.island_id.v[q] == k)) continue;
                    const bool open = !(g.arable.v[q] > 0) && g.landcover.v[q] != LC_FOREST;
                    const double d = np_hypot(static_cast<double>(a - i), static_cast<double>(b - j));
                    const float s32 = static_cast<float>(g.slope.v[q]);
                    const float p32 = 0.2f * s32;
                    const double key = (d - 3.0 * (open ? 1.0 : 0.0)) + static_cast<double>(p32);
                    if (key < best) {
                        best = key;
                        bi = a;
                        bj = b;
                    }
                }
            if (bi >= 0) {
                flat_ok = true;
            } else {
                for (int pass = 0; pass < 2 && bi < 0; ++pass) {
                    float bs = 0;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            const bool wtr = g.river.v[q] > 0 || g.lake.v[q];
                            const bool ok2 = pass == 0 ? (g.island_id.v[q] == k && !wtr && !g.cliff.v[q]) : g.island_id.v[q] == k;
                            if (!ok2) continue;
                            const float s32 = static_cast<float>(g.slope.v[q]);
                            if (bi < 0 || s32 < bs) {
                                bs = s32;
                                bi = a;
                                bj = b;
                            }
                        }
                }
            }
            const int id = static_cast<int>(lands.size()) + 1;
            Json L = Json::obj();
            L.set("id", id);
            L.set("kind", t.kind);
            L.set("of", t.of);
            L.set("island", k);
            L.set("cell", cell_json(bi, bj));
            L.set("km", Json::pair(kmx(bj), kmy(bi)));
            L.set("slope_deg", pyround(static_cast<double>(static_cast<float>(g.slope(bi, bj))), 1));
            L.set("flat", flat_ok);
            L.set("dist_km", pyround(py_hypot(static_cast<double>(bi - i), static_cast<double>(bj - j)) * res_km, 2));
            L.set("main", t.kind == "main");
            lands.push_back(L);
            land_cells.push_back({k, bi, bj});
            if (t.kind == "main" && main_dock < 0) main_dock = id - 1;
            if (!docks_of.count(k)) docks_of[k] = id - 1;
            if (t.v) {
                t.v->landing = id;
                t.v->has_landing = true;
            } else {
                t.s->landing = id;
                t.s->has_landing = true;
            }
        }
    }
    // 栅格：1 田 / 2 梯田 / 3 村 / 4 散户 / 5 泊场 / 6 桥头 / 7 蓄水池 / 8 取水点 / 9 镇 / 10 专业聚落
    Grid<uint8_t> sr(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        if (fields_raster.v[k] > 0) sr.v[k] = 1;
        if (fields_raster.v[k] > 0 && g.arable.v[k] == 2) sr.v[k] = 2;
    }
    for (const auto& lc : land_cells) sr(lc[1], lc[2]) = 5;
    for (const Village& v : villages) sr(v.ci, v.cj) = v.town ? 9 : 3;
    for (const Village& v : hamlets) sr(v.ci, v.cj) = 4;
    for (const Special& s : specials) sr(s.ci, s.cj) = 10;

    // ---------- 第 2 步：桥头 / 导水槽 / 水设施（build_links_water） ----------
    std::vector<Json> bridgeheads, channels, cisterns, intakes;
    std::vector<std::array<double, 3>> cistern_info;   // [岛, 行, 列] 与 basin
    std::vector<double> cistern_basin;
    const bool has_river = !g.islands.empty() && g.islands[0].has_perennial;
    double water_ok_share = 0;
    {
        std::vector<std::vector<std::array<int, 2>>> cliffs(n_isl);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                if (g.cliff(i, j) && g.island_id(i, j) >= 0) cliffs[g.island_id(i, j)].push_back({i, j});
        for (const Link& e : g.links) {
            const int a = e.a, b = e.b;
            if (!e.bridge || cliffs[a].empty() || cliffs[b].empty()) continue;
            // _nearest_pair（sub = 1）：全体点对的最小距离，同值取 a 的次序、再取 b 的次序最先者。b 按行分桶、按当前最优半径剪枝
            const auto& A = cliffs[a];
            const auto& B = cliffs[b];
            std::map<int, std::vector<std::pair<int, int>>> rows;    // 行 → (列, b 下标)，列升序
            for (size_t q = 0; q < B.size(); ++q) rows[B[q][0]].push_back({B[q][1], static_cast<int>(q)});
            int64_t best = INT64_MAX;
            size_t ia = 0, ib = 0;
            for (size_t p = 0; p < A.size(); ++p) {
                const int ai = A[p][0], aj = A[p][1];
                int64_t mine = INT64_MAX;
                int mb = -1;
                const int64_t lim = std::min<int64_t>(best, INT64_MAX / 4);
                const int span = lim >= (1LL << 40) ? H + W : static_cast<int>(std::ceil(std::sqrt(static_cast<double>(lim)))) + 1;
                auto lo = rows.lower_bound(ai - span);
                for (auto it = lo; it != rows.end() && it->first <= ai + span; ++it) {
                    const int64_t dr = it->first - ai;
                    for (const auto& cb : it->second) {
                        const int64_t dc = cb.first - aj;
                        const int64_t d = dr * dr + dc * dc;
                        if (d < mine || (d == mine && cb.second < mb)) {
                            mine = d;
                            mb = cb.second;
                        }
                    }
                }
                if (mb >= 0 && mine < best) {
                    best = mine;
                    ia = p;
                    ib = static_cast<size_t>(mb);
                }
            }
            for (int s = 0; s < 2; ++s) {
                const int k = s == 0 ? a : b;
                const auto& cell = s == 0 ? A[ia] : B[ib];
                Json bh = Json::obj();
                bh.set("island", k);
                bh.set("to", k == a ? b : a);
                bh.set("cell", cell_json(cell[0], cell[1]));
                bh.set("km", Json::pair(kmx(cell[1]), kmy(cell[0])));
                bh.set("gap_km", pyround(e.gap, 3));
                bridgeheads.push_back(bh);
            }
        }
        for (size_t q = 0; q < bridgeheads.size(); ++q) bridgeheads[q].set("id", static_cast<int64_t>(q) + 1);
        for (const auto& t : g.tree) {
            Json ch = Json::obj();
            ch.set("a", t.first);
            ch.set("b", t.second);
            ch.set("a_km", Json::pair(g.islands[t.first].cx_j, g.islands[t.first].cy_j));
            ch.set("b_km", Json::pair(g.islands[t.second].cx_j, g.islands[t.second].cy_j));
            channels.push_back(ch);
        }
        // 水设施
        std::vector<float> fa(N);
        for (size_t k = 0; k < N; ++k) fa[k] = static_cast<float>(g.acc_km2.v[k]);
        const GridI cliff_dist = distance_bands(g.cliff, 4);
        const double P_mm = pyround(g.P_mm, 0);
        const double coef = sc("cistern_catch_coef");
        auto upstream = [&](int i, int j, int steps) {
            for (int s = 0; s < steps; ++s) {
                int bi = -1, bj = -1;
                for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 1; ++dj) {
                        const int a = i + di, b = j + dj;
                        if ((di || dj) && a >= 0 && a < H && b >= 0 && b < W && g.island_id(a, b) == g.island_id(i, j) &&
                            fa[static_cast<size_t>(a) * W + b] < fa[static_cast<size_t>(i) * W + j] &&
                            (bi < 0 || fa[static_cast<size_t>(a) * W + b] > fa[static_cast<size_t>(bi) * W + bj])) {
                            bi = a;
                            bj = b;
                        }
                    }
                if (bi < 0) break;
                i = bi;
                j = bj;
            }
            return std::array<int, 2>{i, j};
        };
        std::vector<std::array<int, 3>> cis_cells;
        if (!has_river && g.basins.present) {
            // J["hydro"]["main_basins"]["mouths"]：按汇流降序的前 12 个，面积 round 1
            std::vector<std::array<double, 3>> ms = g.basins.mouths;
            std::stable_sort(ms.begin(), ms.end(), [](const auto& x, const auto& y) { return -x[2] < -y[2]; });
            if (ms.size() > 12) ms.resize(12);
            const double main_area = g.islands[0].area_j;
            for (const auto& m : ms) {
                const double area = pyround(m[2], 1);
                if (area < sc("cistern_basin_min_frac") * main_area) continue;
                const auto cell = upstream(static_cast<int>(m[0]), static_cast<int>(m[1]), 3);
                Json x = Json::obj();
                x.set("island", 0);
                x.set("cell", cell_json(cell[0], cell[1]));
                x.set("km", Json::pair(kmx(cell[1]), kmy(cell[0])));
                x.set("basin_km2", area);
                x.set("capacity_1000m3", pyround(area * P_mm * coef, 0));
                cisterns.push_back(x);
                cis_cells.push_back({0, cell[0], cell[1]});
                cistern_basin.push_back(area);
            }
        }
        for (int k = has_river ? 1 : 0; k < n_isl; ++k) {
            // m = 本岛且离崖 ≥ 2 格（没有就整岛）里汇流最大的格
            int bi = -1, bj = -1;
            float bv = -1.0f;
            bool any2 = false;
            for (int pass = 0; pass < 2 && !any2; ++pass)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        if (g.island_id.v[q] != k || (pass == 0 && cliff_dist.v[q] < 2)) continue;
                        any2 = true;
                        if (bi < 0 || fa[q] > bv) {
                            bv = fa[q];
                            bi = i;
                            bj = j;
                        }
                    }
            if (bi < 0) {
                bi = 0;
                bj = 0;
            }
            if (k == 0 && !cisterns.empty()) continue;
            const double area_k = g.islands[k].area_j;
            Json x = Json::obj();
            x.set("island", k);
            x.set("cell", cell_json(bi, bj));
            x.set("km", Json::pair(kmx(bj), kmy(bi)));
            x.set("basin_km2", pyround(static_cast<double>(fa[static_cast<size_t>(bi) * W + bj]), 2));
            x.set("capacity_1000m3", pyround(area_k * P_mm * coef * 0.5, 0));
            cisterns.push_back(x);
            cis_cells.push_back({k, bi, bj});
            cistern_basin.push_back(pyround(static_cast<double>(fa[static_cast<size_t>(bi) * W + bj]), 2));
        }
        std::vector<std::array<int, 3>> intake_rec;   // [村号, 行, 列]
        std::vector<double> intake_dist;
        if (has_river) {
            std::vector<std::array<int, 2>> rc;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    if (g.river(i, j) > 0 && g.island_id(i, j) == 0) rc.push_back({i, j});
            for (const Village& r : villages) {
                if (r.island != 0 || rc.empty()) continue;
                int64_t best = INT64_MAX;
                size_t p = 0;
                for (size_t q = 0; q < rc.size(); ++q) {
                    const int64_t di = rc[q][0] - r.ci, dj = rc[q][1] - r.cj;
                    const int64_t d = di * di + dj * dj;
                    if (d < best) {
                        best = d;
                        p = q;
                    }
                }
                Json x = Json::obj();
                x.set("island", 0);
                x.set("village", r.id);
                x.set("cell", cell_json(rc[p][0], rc[p][1]));
                x.set("km", Json::pair(kmx(rc[p][1]), kmy(rc[p][0])));
                const double dk = pyround(std::sqrt(static_cast<double>(best)) * res_km, 2);
                x.set("dist_km", dk);
                intakes.push_back(x);
                intake_rec.push_back({r.id, rc[p][0], rc[p][1]});
                intake_dist.push_back(dk);
            }
        }
        for (size_t q = 0; q < cisterns.size(); ++q) cisterns[q].set("id", static_cast<int64_t>(q) + 1);
        for (size_t q = 0; q < intakes.size(); ++q) intakes[q].set("id", static_cast<int64_t>(q) + 1);
        int ok = 0;
        const double lim = sc("village_water_km") / res_km;
        for (Village& r : villages) {
            const double dw = static_cast<double>(dist_water(r.ci, r.cj));
            double dc = 1e9;
            if (!cis_cells.empty()) {
                double m = INF;
                for (const auto& cc : cis_cells) {
                    const double di = static_cast<double>(cc[1]) - r.ci, dj = static_cast<double>(cc[2]) - r.cj;
                    m = std::min(m, std::sqrt(di * di + dj * dj));
                }
                dc = m;
            }
            std::string src = dw <= dc ? "waterway" : "cistern";
            double dd = dw <= dc ? dw : dc;
            if (has_river && r.island == 0) {
                for (size_t q = 0; q < intake_rec.size(); ++q)
                    if (intake_rec[q][0] == r.id) {
                        if (intake_dist[q] / res_km < dd) {
                            src = "intake";
                            dd = intake_dist[q] / res_km;
                        }
                        break;
                    }
            }
            r.water_src = src;
            r.water_dist = pyround(dd * res_km, 2);
            r.has_water = true;
            if (dd <= lim) ++ok;
        }
        water_ok_share = pyround(static_cast<double>(ok) / static_cast<double>(std::max<size_t>(1, villages.size())), 3);
        for (const Json& bh : bridgeheads) sr(static_cast<int>(bh.at("cell").items()[0].as_int()), static_cast<int>(bh.at("cell").items()[1].as_int())) = 6;
        for (const auto& cc : cis_cells) sr(cc[1], cc[2]) = 7;
        for (const auto& it : intake_rec) sr(it[1], it[2]) = 8;
    }

    // ---------- 第 3 步：前哨、主家候选、都与城（build_homes_city） ----------
    std::vector<Json> outposts, homes;
    Json city;
    {
        std::vector<double> height(N);
        for (size_t k = 0; k < N; ++k) height[k] = g.island_id.v[k] >= 0 ? g.height.v[k] : -1.0;
        const Village* sv = seat >= 0 ? &villages[seat] : nullptr;
        std::map<int, const Village*> biggest;
        {
            std::vector<const Village*> all;
            for (const Village& v : villages) all.push_back(&v);
            for (const Village& v : hamlets) all.push_back(&v);
            std::stable_sort(all.begin(), all.end(), [](const Village* a, const Village* b) { return -a->households < -b->households; });
            for (const Village* v : all)
                if (!biggest.count(v->island)) biggest[v->island] = v;
        }
        std::map<int, double> field_km2;
        for (const Field& f : fields) field_km2[f.island] += f.area_km2;
        struct Outpost {
            int id, island, ci, cj;
            double kx, ky, field_km2;
        };
        std::vector<Outpost> ops;
        for (int k = 1; k < n_isl; ++k) {
            const auto fk = field_km2.find(k);
            if (docks_of.count(k) && fk != field_km2.end() && fk->second > 0 && biggest.count(k)) {
                const Village* r = biggest[k];
                Json o = Json::obj();
                const int id = static_cast<int>(outposts.size()) + 1;
                o.set("id", id);
                o.set("island", k);
                o.set("cell", cell_json(r->ci, r->cj));
                o.set("km", Json::pair(r->kx, r->ky));
                o.set("households", r->households);
                o.set("field_km2", pyround(fk->second, 3));
                o.set("landing", docks_of[k] + 1);
                o.set("settlement", r->name());
                outposts.push_back(o);
                ops.push_back({id, k, r->ci, r->cj, r->kx, r->ky, pyround(fk->second, 3)});
            }
        }
        const std::array<int, 3>* md = main_dock >= 0 ? &land_cells[main_dock] : nullptr;
        auto info = [&](int i, int j, const char* kind, Json note, bool np_ = true) {   // np_：格来自 np.where（numpy 整数）
            const int k = g.island_id(i, j);
            const IslandRec& isl = g.islands[k];
            Json h = Json::obj();
            h.set("kind", kind);
            h.set("note", std::move(note));
            h.set("island", k);
            h.set("cell", cell_json(i, j));
            h.set("km", Json::pair(kmx_np(j, np_), kmy_np(i, np_)));
            h.set("elev_m", pyround(g.height(i, j), 0));
            h.set("island_age", age_name(isl.kind));
            h.set("island_has_river", isl.hydro ? isl.has_perennial : false);
            h.set("island_lakes", isl.hydro ? isl.n_lakes : 0);
            h.set("dist_seat_km", sv ? Json(pyround(py_hypot(static_cast<double>(i - sv->ci), static_cast<double>(j - sv->cj)) * res_km, 2)) : Json());
            h.set("dist_main_landing_km",
                  md ? Json(pyround(py_hypot(static_cast<double>(i - (*md)[1]), static_cast<double>(j - (*md)[2])) * res_km, 2)) : Json());
            h.set("season_type", g.has_climate ? Json(g.clim.type_code) : Json());
            h.set("season_names", g.has_climate ? Json::arr_of(g.clim.names) : Json());
            h.set("snow_days", Json());
            h.set("storm_days", Json());
            h.set("sailable_days", Json());
            h.set("landcover", static_cast<int>(g.landcover(i, j)));
            h.set("water_dist_km", pyround(static_cast<double>(dist_water(i, j)) * res_km, 2));
            return h;
        };
        // 候选一：邑治旁（1–3 km 环内，近主泊场）
        if (sv) {
            const int si = sv->ci, sj = sv->cj;
            const double lo = 1.0 / res_km, hi = 3.0 / res_km;
            double best = -INF;
            int bi = -1, bj = -1;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    const size_t q = static_cast<size_t>(i) * W + j;
                    if (!(ok_site.v[q] && g.island_id.v[q] == sv->island)) continue;
                    const double d = np_hypot(static_cast<double>(i - si), static_cast<double>(j - sj));
                    if (!(d >= lo && d <= hi)) continue;
                    const double dd = md ? np_hypot(static_cast<double>(i - (*md)[1]), static_cast<double>(j - (*md)[2])) : d;
                    const double s = score_base[q] - 0.02 * dd;
                    if (bi < 0 || s > best) {
                        best = s;
                        bi = i;
                        bj = j;
                    }
                }
            if (bi >= 0) homes.push_back(info(bi, bj, "seat_side", Json("seat_side")));
        }
        // 候选二：河湖僻处（离任何村 ≥ 3 km、近水、坡缓；候选格抽稀 3×3）；没有就换「高台」
        {
            const double far = 3.0 / res_km;
            double best = -INF;
            int bi = -1, bj = -1;
            for (int i = 0; i < H; i += 3)
                for (int j = 0; j < W; j += 3) {
                    const size_t q = static_cast<size_t>(i) * W + j;
                    if (!(ok_site.v[q] && dist_water.v[q] <= 5)) continue;
                    double dv = 1e9;
                    if (!villages.empty()) {
                        double m = 1e18;
                        for (const Village& v : villages) {
                            const double di = static_cast<double>(i) - v.ci, dj = static_cast<double>(j) - v.cj;
                            m = std::min(m, std::sqrt(di * di + dj * dj));
                        }
                        dv = m;
                    }
                    if (!(dv >= far)) continue;
                    const double s = score_base[q] + 0.01 * std::min(dv, 3 * far);
                    if (bi < 0 || s > best) {
                        best = s;
                        bi = i;
                        bj = j;
                    }
                }
            if (bi >= 0) {
                homes.push_back(info(bi, bj, "riverside", Json("riverside")));
            } else {
                double bh = -INF;
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        if (!(ok_site.v[q] && g.island_id.v[q] == 0)) continue;
                        if (bi < 0 || height[q] > bh) {
                            bh = height[q];
                            bi = i;
                            bj = j;
                        }
                    }
                if (bi >= 0) homes.push_back(info(bi, bj, "highland", Json("highland")));
            }
        }
        // 候选三：小岛前哨（离主岛最远的前哨）
        if (!ops.empty()) {
            const double c0x = g.islands[0].cx_j, c0y = g.islands[0].cy_j;
            size_t b = 0;
            double best = -INF;
            for (size_t q = 0; q < ops.size(); ++q) {
                const double d = py_hypot(ops[q].kx - c0x, ops[q].ky - c0y);
                if (q == 0 || d > best) {
                    best = d;
                    b = q;
                }
            }
            Json note = Json::arr();
            note.push("outpost_isle");
            note.push(ops[b].field_km2);
            Json h = info(ops[b].ci, ops[b].cj, "outpost_isle", note, false);
            h.set("outpost", ops[b].id);
            homes.push_back(h);
        }
        for (size_t q = 0; q < homes.size(); ++q) homes[q].set("id", static_cast<int64_t>(q) + 1);
        // 都与城（5.7）
        if (inp.is_capital && sv) {
            Village& seatv = villages[seat];
            const double pop_own = g.settle_pop;
            const double rate = inp.reformer ? sc("city_gather_rate_reformer") : sc("city_gather_rate");
            const double city_pop = pop_own * sc("city_urban_rate") + inp.state_pop * rate;
            const int64_t city_hh = static_cast<int64_t>(std::nearbyint(city_pop / hh_size));
            std::set<int> reach_set{0};
            std::map<int, std::vector<int>> adj;
            for (const auto& t : g.tree) {
                adj[t.first].push_back(t.second);
                adj[t.second].push_back(t.first);
            }
            std::vector<int> stack{0};
            while (!stack.empty()) {
                const int a = stack.back();
                stack.pop_back();
                for (int b : adj[a])
                    if (!reach_set.count(b)) {
                        reach_set.insert(b);
                        stack.push_back(b);
                    }
            }
            const double glim = sc("city_guo_km") / res_km;
            std::vector<Village*> guo;
            for (Village& v : villages)
                if (reach_set.count(v.island) && v.island != 0 &&
                    py_hypot(static_cast<double>(v.ci - seatv.ci), static_cast<double>(v.cj - seatv.cj)) <= glim)
                    guo.push_back(&v);
            for (Village& v : villages)
                if (v.island == 0 && &v != &seatv && py_hypot(static_cast<double>(v.ci - seatv.ci), static_cast<double>(v.cj - seatv.cj)) <= glim)
                    guo.push_back(&v);
            int64_t guo_hh = 0;
            for (Village* v : guo) guo_hh += v->households;
            const int64_t inner_hh = std::max(seatv.households + (seatv.has_market ? seatv.households_market : 0), city_hh - guo_hh);
            for (Village* v : guo) v->guo = true;
            seatv.city = true;
            Json altar;
            if (!has_river && !cistern_basin.empty()) {
                size_t b = 0;
                for (size_t q = 1; q < cistern_basin.size(); ++q)
                    if (cistern_basin[q] > cistern_basin[b]) b = q;
                const int i0 = static_cast<int>(cisterns[b].at("cell").items()[0].as_int()), j0 = static_cast<int>(cisterns[b].at("cell").items()[1].as_int());
                const int isl_c = static_cast<int>(cisterns[b].at("island").as_int());
                const int r0 = std::max(0, i0 - 5), r1 = std::min(H, i0 + 6), c0 = std::max(0, j0 - 5), c1 = std::min(W, j0 + 6);
                double bv = -INF;
                int ai = r0, aj = c0;
                for (int a = r0; a < r1; ++a)
                    for (int bb = c0; bb < c1; ++bb) {
                        const double v = g.island_id(a, bb) == isl_c ? height[static_cast<size_t>(a) * W + bb] : -1.0;
                        if (v > bv) {
                            bv = v;
                            ai = a;
                            aj = bb;
                        }
                    }
                altar = cell_json(ai, aj);
            } else {
                double bh = -INF;
                int ai = -1, aj = -1;
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        if (!(ok_site.v[q] && g.island_id.v[q] == 0)) continue;
                        const double d = np_hypot(static_cast<double>(i - seatv.ci), static_cast<double>(j - seatv.cj));
                        if (!(d <= 2.0 / res_km)) continue;
                        if (ai < 0 || height[q] > bh) {
                            bh = height[q];
                            ai = i;
                            aj = j;
                        }
                    }
                if (ai >= 0) altar = cell_json(ai, aj);
            }
            city = Json::obj();
            city.set("role", inp.reformer ? "reformer_capital" : "capital");
            city.set("state", inp.state);
            city.set("state_pop", pyround(inp.state_pop, 0));
            city.set("population", pyround(city_pop, 0));
            city.set("households", city_hh);
            city.set("inner_households", inner_hh);
            city.set("guo_households", guo_hh);
            Json gv = Json::arr();
            std::set<int> gi;
            for (Village* v : guo) {
                gv.push(v->id);
                gi.insert(v->island);
            }
            city.set("guo_villages", gv);
            city.set("n_guo_islands", static_cast<int64_t>(gi.size()));
            city.set("seat_village", seatv.id);
            city.set("granary_landing", main_dock >= 0 ? Json(main_dock + 1) : Json());
            city.set("altar_cell", altar);
            if (altar.is_null()) city.set("altar_km", Json());
            else city.set("altar_km", Json::pair(kmx(static_cast<double>(altar.items()[1].as_int())), kmy(static_cast<double>(altar.items()[0].as_int()))));
            city.set("gather_rate", rate);
            city.set("urban_rate", sc("city_urban_rate"));
        }
    }

    // ---------- 村周开垦（clear_forest）：林地在聚落半径内改草坡（内圈）/ 灌丛（外圈） ----------
    Json clearing = Json::obj();
    {
        Mask seedm(H, W, 0);
        std::vector<double> rad(N, 0.0);
        auto add_seed = [&](int i, int j, int64_t hh, bool town) {
            double r = clip(sc("clear_base_km") * std::sqrt(static_cast<double>(std::max<int64_t>(hh, 1)) / 40.0), sc("clear_min_km"), sc("clear_max_km"));
            if (town) r = std::min(sc("clear_max_km") * sc("town_clear_mult"), r * sc("town_clear_mult"));
            seedm(i, j) = 1;
            const size_t q = static_cast<size_t>(i) * W + j;
            rad[q] = std::max(rad[q], r * 1000.0);
        };
        for (const Village& v : villages) add_seed(v.ci, v.cj, v.households + (v.has_market ? v.households_market : 0), v.town != 0);
        for (const Village& v : hamlets) add_seed(v.ci, v.cj, v.households, false);
        for (const Special& s : specials) add_seed(s.ci, s.cj, s.households, false);
        int64_t n_land = 0, nf0 = 0;
        bool any_seed = false;
        for (size_t k = 0; k < N; ++k) {
            n_land += land.v[k];
            nf0 += (g.landcover.v[k] == LC_FOREST && land.v[k]) ? 1 : 0;
            any_seed = any_seed || seedm.v[k];
        }
        n_land = std::max<int64_t>(1, n_land);
        if (!any_seed || nf0 == 0) {
            clearing.set("cleared_km2", 0.0);
            clearing.set("forest_share_before", 0.0);
            clearing.set("forest_share_after", 0.0);
        } else {
            const double rmax = *std::max_element(rad.begin(), rad.end());
            const int max_cells = static_cast<int>(std::ceil(rmax / res_m)) + 1;
            GridD dist;
            Grid<int64_t> src;
            nearest_propagate(seedm, max_cells, res_m, &land, dist, src);
            const double inner_frac = sc("clear_inner_frac");
            int64_t n_hit = 0, n_inner = 0;
            std::vector<uint8_t> hit(N, 0);
            for (size_t k = 0; k < N; ++k) {
                const bool f0 = g.landcover.v[k] == LC_FOREST && land.v[k];
                const int64_t s = src.v[k] >= 0 ? src.v[k] : 0;
                const double r_at = rad[s];
                const bool same = g.island_id.v[s] == g.island_id.v[k];
                if (f0 && src.v[k] >= 0 && same && dist.v[k] <= r_at) {
                    hit[k] = 1;
                    ++n_hit;
                    if (dist.v[k] <= inner_frac * r_at) {
                        g.landcover.v[k] = LC_GRASS;
                        ++n_inner;
                    } else {
                        g.landcover.v[k] = LC_SHRUB;
                    }
                }
            }
            if (g.has_resources) {
                for (Deposit& d : R.deposits)
                    if (d.kind == RK_TIMBER) {
                        d.has_area_before = true;
                        d.area_before = d.area_km2;
                    }
                for (size_t k = 0; k < N; ++k) {
                    const int32_t p = g.patch_id.v[k];
                    if (hit[k] && p >= 0 && R.deposits[p].kind == RK_TIMBER) g.patch_id.v[k] = -1;
                }
            }
            int64_t nf1 = 0;
            for (size_t k = 0; k < N; ++k) nf1 += (g.landcover.v[k] == LC_FOREST && land.v[k]) ? 1 : 0;
            clearing.set("cleared_km2", pyround(static_cast<double>(n_hit) * cell_km2, 2));
            clearing.set("to_grass_km2", pyround(static_cast<double>(n_inner) * cell_km2, 2));
            clearing.set("to_shrub_km2", pyround(static_cast<double>(n_hit - n_inner) * cell_km2, 2));
            clearing.set("forest_share_before", pyround(static_cast<double>(nf0) / static_cast<double>(n_land), 4));
            clearing.set("forest_share_after", pyround(static_cast<double>(nf1) / static_cast<double>(n_land), 4));
        }
    }

    // ---------- 村的采场（village_workings）：开垦之后，石料 / 黏土 / 砂砾就近取，窑村开土坑，砂金区各一处淘金点 ----------
    Json wsum = Json::obj();
    if (g.has_resources) {
        std::vector<uint8_t> base_ok(N);
        for (size_t k = 0; k < N; ++k)
            base_ok[k] = (land.v[k] && !(g.river.v[k] > 0 || g.lake.v[k]) && !g.cliff.v[k] && g.arable.v[k] == 0) ? 1 : 0;
        const double decay = sc("working_decay_km"), share = sc("working_share_km");
        std::vector<Village*> order;
        for (Village& v : villages) order.push_back(&v);
        std::stable_sort(order.begin(), order.end(), [](const Village* a, const Village* b) {
            if (a->households != b->households) return a->households > b->households;
            return a->id < b->id;
        });
        auto new_work = [&](int kind, int a, int b, const std::vector<double>& fk, const GridI& lab) -> int {
            Working w;
            w.id = static_cast<int>(R.works.size());
            w.kind = kind;
            w.occurrence = lab(a, b);
            w.island = g.island_id(a, b);
            w.ci = a;
            w.cj = b;
            w.kx = kmx(b);
            w.ky = kmy(a);
            w.grade = pyround(fk[static_cast<size_t>(a) * W + b], 3);
            R.works.push_back(w);
            if (kind == RK_STONE) open_working(g, a, b);
            return w.id;
        };
        struct Mine {
            int a, b, w;
        };
        auto nearest = [&](const std::vector<Mine>& mine, int i, int j) {
            int best = -1;
            double bd = share;
            for (const Mine& m : mine) {
                const double d = py_hypot(static_cast<double>(m.a - i), static_cast<double>(m.b - j)) * res_km;
                if (d <= bd) {
                    best = m.w;
                    bd = d;
                }
            }
            return best;
        };
        const std::pair<int, const char*> kinds[3] = {{RK_STONE, "quarry_reach_km"}, {RK_CLAY, "clay_reach_km"}, {RK_GRAVEL, "gravel_reach_km"}};
        for (const auto& kr : kinds) {
            const int kind = kr.first;
            const int fki = field_of(kind);
            const GridI& lab = g.occ_lab[fki];
            std::vector<double> fk(N);
            for (size_t k = 0; k < N; ++k) fk[k] = static_cast<double>(g.res_field[fki].v[k]) / 255.0;
            std::vector<uint8_t> elig(N);
            for (size_t k = 0; k < N; ++k) {
                elig[k] = (base_ok[k] && lab.v[k] >= 0) ? 1 : 0;
                if (kind != RK_STONE && g.landcover.v[k] == LC_FOREST) elig[k] = 0;
            }
            const double reachkm = sc(kr.second);
            const int rc_ = static_cast<int>(std::ceil(reachkm / res_km));
            std::vector<Mine> mine;
            for (const Working& w : R.works)
                if (w.kind == kind) mine.push_back({w.ci, w.cj, w.id});
            struct User {
                Village* v;
                Special* s;
            };
            std::vector<User> users;
            for (Village* v : order) users.push_back({v, nullptr});
            if (kind == RK_CLAY)
                for (Special& s : specials)
                    if (s.kind == "kiln_village") users.push_back({nullptr, &s});
            int served = 0;
            for (const User& u : users) {
                const int i = u.v ? u.v->ci : u.s->ci, j = u.v ? u.v->cj : u.s->cj;
                int w = u.v ? nearest(mine, i, j) : -1;
                if (w < 0) {
                    const int r0 = std::max(0, i - rc_), r1 = std::min(H, i + rc_ + 1), c0 = std::max(0, j - rc_), c1 = std::min(W, j + rc_ + 1);
                    double best = 0;
                    int bi = -1, bj = -1;
                    bool first = true;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            if (!elig[q]) continue;
                            if (u.s && lab.v[q] != u.s->occurrence) continue;
                            const double d = np_hypot(static_cast<double>(a - i), static_cast<double>(b - j)) * res_km;
                            const double s = d <= reachkm ? fk[q] * std::exp(-d / decay) : -1.0;
                            if (first || s > best) {
                                best = s;
                                bi = a;
                                bj = b;
                                first = false;
                            }
                        }
                    if (!first && best > 0) {
                        w = u.v ? nearest(mine, bi, bj) : -1;
                        if (w < 0) {
                            w = new_work(kind, bi, bj, fk, lab);
                            mine.push_back({bi, bj, w});
                        }
                    }
                }
                if (w < 0) continue;
                if (u.v) {
                    R.works[w].villages.push_back(u.v->id);
                    u.v->workings.push_back(w);
                    ++served;
                } else {
                    R.works[w].special = u.s->id;
                    u.s->working = w;
                }
            }
            int64_t nk = 0;
            for (const Working& x : R.works) nk += x.kind == kind ? 1 : 0;
            Json e = Json::obj();
            e.set("n", nk);
            e.set("villages_served", served);
            e.set("villages_share", pyround(static_cast<double>(served) / static_cast<double>(std::max<size_t>(1, villages.size())), 3));
            e.set("reach_km", reachkm);
            wsum.set(res_key(kind), e);
        }
        // 淘金点：每个砂金区在非耕非林的滩地上挑品位最高的一格，挂到 placer_link_km 内最近的村
        {
            const int fki = FK_PLACER;
            const GridI& lab = g.occ_lab[fki];
            std::vector<double> fk(N);
            for (size_t k = 0; k < N; ++k) fk[k] = static_cast<double>(g.res_field[fki].v[k]) / 255.0;
            std::map<int, std::vector<int32_t>> groups;
            for (size_t k = 0; k < N; ++k)
                if (base_ok[k] && g.landcover.v[k] != LC_FOREST && lab.v[k] >= 0) groups[lab.v[k]].push_back(static_cast<int32_t>(k));
            int64_t n_pl = 0;
            for (auto& kv : groups) {
                const auto& cells = kv.second;
                size_t t = 0;
                for (size_t q = 1; q < cells.size(); ++q)
                    if (fk[cells[q]] > fk[cells[t]]) t = q;
                const int a = cells[t] / W, b = cells[t] % W;
                const int w = new_work(RK_PLACER, a, b, fk, lab);
                ++n_pl;
                if (!villages.empty()) {
                    size_t qb = 0;
                    double bd = INF;
                    for (size_t q = 0; q < villages.size(); ++q) {
                        const double d = np_hypot(static_cast<double>(villages[q].ci) - a, static_cast<double>(villages[q].cj) - b) * res_km;
                        if (d < bd) {
                            bd = d;
                            qb = q;
                        }
                    }
                    if (bd <= sc("placer_link_km")) {
                        R.works[w].villages.push_back(villages[qb].id);
                        villages[qb].workings.push_back(w);
                        continue;
                    }
                }
                R.works[w].note = "placer_no_village";
            }
            int64_t n_occ = 0;
            for (const Occurrence& o : R.occ) n_occ += o.kind == RK_PLACER ? 1 : 0;
            Json e = Json::obj();
            e.set("n", n_pl);
            e.set("occurrences", n_occ);
            wsum.set("placer", e);
        }
        sync_resources(g);
    }

    // ---------- 汇总 ----------
    auto village_json = [&](const Village& r) {
        Json j = Json::obj();
        j.set("id", r.id);
        j.set("island", r.island);
        j.set("cell", cell_json(r.ci, r.cj));
        j.set("km", Json::pair(r.kx, r.ky));
        j.set("households", r.households);
        j.set("field", r.field);
        j.set("elev_m", r.elev_m);
        j.set("water_dist_km", r.water_dist_km);
        j.set("shore_dist_km", r.shore_dist_km);
        j.set("on_arable", r.on_arable);
        j.set("name", r.name());
        if (r.seat) j.set("seat", true);
        if (r.town) j.set("town", r.town);
        if (r.has_market) j.set("households_market", r.households_market);
        if (r.has_market_town) j.set("market_town", r.market_town);
        if (r.has_landing) j.set("landing", r.landing);
        if (r.has_water) {
            Json w = Json::obj();
            w.set("source", r.water_src);
            w.set("dist_km", r.water_dist);
            j.set("water", w);
        }
        if (r.guo) j.set("guo", true);
        if (r.city) j.set("city", true);
        if (!r.workings.empty()) j.set("workings", Json::arr_of(r.workings));
        return j;
    };
    Json S = Json::obj();
    S.set("population", pyround(pop, 0));
    S.set("population_source", pop_polity ? "polity" : "arable");
    S.set("household_size", hh_size);
    S.set("households", hh_total);
    S.set("households_by_island", Json::arr_of(hh_isl));
    S.set("land_per_household_km2", pyround(land_per_hh, 4));
    S.set("nonfarm_share", sc("nonfarm_share"));
    S.set("nonfarm_households", nonfarm_hh);
    Json fj = Json::arr();
    for (const Field& f : fields) {
        Json j = Json::obj();
        j.set("id", f.id);
        j.set("island", f.island);
        j.set("cells", f.cells);
        j.set("area_km2", f.area_km2);
        j.set("terrace_frac", f.terrace_frac);
        j.set("centroid_cell", cell_json(f.cci, f.ccj));
        j.set("centroid_km", Json::pair(f.ckx, f.cky));
        j.set("water_dist_km", f.water_dist_km);
        if (f.has_hh) j.set("households", f.households);
        j.set("village", f.village == 0 ? Json() : Json(f.village));
        fj.push(j);
    }
    S.set("fields", fj);
    Json vj = Json::arr(), hj = Json::arr(), sj = Json::arr();
    int64_t hh_v = 0, hh_h = 0, hh_m = 0, hh_s = 0;
    std::vector<double> vh;
    for (const Village& v : villages) {
        vj.push(village_json(v));
        hh_v += v.households;
        hh_m += v.has_market ? v.households_market : 0;
        vh.push_back(static_cast<double>(v.households));
    }
    for (const Village& v : hamlets) {
        hj.push(village_json(v));
        hh_h += v.households;
    }
    for (const Special& s : specials) {
        Json j = Json::obj();
        j.set("id", s.id);
        j.set("kind", s.kind);
        j.set("resource", s.resource < 0 ? Json() : Json(s.resource));
        j.set("occurrence", s.occurrence < 0 ? Json() : Json(s.occurrence));
        j.set("subtype", s.subtype.empty() ? Json() : Json(s.subtype));
        j.set("island", s.island);
        j.set("cell", cell_json(s.ci, s.cj));
        j.set("km", Json::pair(s.kx, s.ky));
        j.set("households", s.households);
        j.set("note", s.note);
        j.set("name", s.name());
        if (s.has_landing) j.set("landing", s.landing);
        if (s.working >= 0) j.set("working", s.working);
        sj.push(j);
        hh_s += s.households;
    }
    S.set("villages", vj);
    S.set("hamlets", hj);
    S.set("n_fields", static_cast<int64_t>(fields.size()));
    S.set("n_villages", static_cast<int64_t>(villages.size()));
    S.set("n_hamlets", static_cast<int64_t>(hamlets.size()));
    S.set("households_in_villages", hh_v);
    S.set("households_in_hamlets", hh_h);
    S.set("households_in_towns_market", hh_m);
    S.set("households_in_specials", hh_s);
    S.set("seat", seat >= 0 ? Json(villages[seat].id) : Json());
    S.set("seat_households", seat >= 0 ? villages[seat].households : 0);
    S.set("village_hh_median", villages.empty() ? 0 : static_cast<int64_t>(np_median(vh)));
    Json tj = Json::arr();
    for (Json& t : towns_json) tj.push(std::move(t));
    S.set("towns", tj);
    S.set("specials", sj);
    auto arr_of_json = [](std::vector<Json>& v) {
        Json a = Json::arr();
        for (Json& x : v) a.push(std::move(x));
        return a;
    };
    S.set("bridgeheads", arr_of_json(bridgeheads));
    S.set("channels", arr_of_json(channels));
    S.set("cisterns", arr_of_json(cisterns));
    S.set("intakes", arr_of_json(intakes));
    S.set("has_river", has_river);
    S.set("water_ok_share", water_ok_share);
    S.set("landings", arr_of_json(lands));
    S.set("outposts", arr_of_json(outposts));
    S.set("home_candidates", arr_of_json(homes));
    S.set("city", city);
    S.set("clearing", clearing);
    S.set("workings", wsum);
    g.settle = std::move(S);
    g.settle_raster = std::move(sr);
    g.settle_fields = std::move(fields_raster);
    g.has_settle = true;
    g.sec_settle = now_s() - t0;
}

}  // namespace skyisle::island
