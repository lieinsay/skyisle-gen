// 5.3c 地形区与资源（resources.py 的 build_resources / sync_resources / rock_site_mask / open_working 同式）。
// 随机流与 Python 版逐次相同：resources:noise / floatstone / lith / {岛号}；每岛的逐项抽样次序照抄（泊松、对数正态、抽签的先后）。
// 每岛只在岛的外框里做（Python 版每次整图运算，但掩膜都含 m = 本岛）：光栅次序、连通分量的编号次序与整图一致。
#include "skyisle/island/resources.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>

#include "skyisle/flow.hpp"

namespace skyisle::island {

namespace {

enum { LC_VOID, LC_CLIFF, LC_ROCK, LC_ALPINE, LC_FOREST, LC_SHRUB, LC_GRASS, LC_ARABLE, LC_TERRACE, LC_WET, LC_RIVER, LC_LAKE };

const char* RES_KEYS[RK_COUNT] = {"none", "timber", "spring", "clay", "peat", "gravel", "placer", "stone", "floatstone", "ore",
                                  "hotspring", "sulfur", "cave", "guano"};
const ResForm RES_FORMS[RK_COUNT] = {FORM_POINT, FORM_PATCH, FORM_POINT, FORM_FIELD, FORM_PATCH, FORM_FIELD, FORM_FIELD, FORM_FIELD,
                                     FORM_PATCH, FORM_FIELD, FORM_POINT, FORM_FIELD, FORM_POINT, FORM_PATCH};
const int FIELD_RES[FK_COUNT] = {RK_ORE, RK_SULFUR, RK_PLACER, RK_CLAY, RK_GRAVEL, RK_STONE};
const char* ZONE_KEYS[8] = {"void", "alpine", "mountain", "hill", "plain", "valley", "cliff", "water"};
// 主导栅格的画法顺序：后画的盖先画的（稀的盖常的，点最后）
const int DOMINANT_ORDER[13] = {RK_TIMBER, RK_STONE, RK_GRAVEL, RK_CLAY, RK_PEAT, RK_GUANO, RK_FLOATSTONE, RK_PLACER, RK_SULFUR, RK_ORE,
                                RK_SPRING, RK_HOTSPRING, RK_CAVE};

// 金属矿的矿种权重：Python 版按中文键 sorted() 取签，码位序 = 金 < 铁 < 铅锌 < 铜 < 锡 < 锡钨 < 锰
enum Metal { M_GOLD, M_IRON, M_LEADZINC, M_COPPER, M_TIN, M_TINTUNGSTEN, M_MANGANESE, M_COUNT };
const char* METAL_KEYS[M_COUNT] = {"gold", "iron", "leadzinc", "copper", "tin", "tintungsten", "manganese"};

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline double f32(double x) { return static_cast<double>(static_cast<float>(x)); }

struct Box {
    int r0 = 0, r1 = 0, c0 = 0, c1 = 0;   // [r0, r1) × [c0, c1)
    bool empty() const { return r1 <= r0; }
};

// 贪心取种子（_seeds）：按分数降序（同分按扁平下标），与已选的距离 ≥ min_sep 格；最多看前 40000 个候选。
// score 是岛外框里的局部栅格（外框外的分数在 Python 版里都是 0）：局部光栅次序与整图的扁平下标次序一致
std::vector<std::pair<int, int>> seeds_of(const std::vector<double>& score, const Box& bx, int64_t n, double min_sep) {
    std::vector<std::pair<int, int>> out;
    if (n <= 0) return out;
    const int bw = bx.c1 - bx.c0;
    std::vector<std::pair<double, int32_t>> cand;
    for (size_t q = 0; q < score.size(); ++q)
        if (score[q] > 0) cand.emplace_back(score[q], static_cast<int32_t>(q));
    if (cand.empty()) return out;
    auto cmp = [](const std::pair<double, int32_t>& a, const std::pair<double, int32_t>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    const size_t take = std::min<size_t>(cand.size(), 40000);
    std::partial_sort(cand.begin(), cand.begin() + take, cand.end(), cmp);
    const double ms2 = min_sep * min_sep;
    std::vector<std::pair<double, double>> pts;
    for (size_t q = 0; q < take; ++q) {
        const int i = bx.r0 + cand[q].second / bw, j = bx.c0 + cand[q].second % bw;
        bool bad = false;
        for (const auto& p : pts) {
            const double di = p.first - i, dj = p.second - j;
            if (di * di + dj * dj < ms2) {
                bad = true;
                break;
            }
        }
        if (bad) continue;
        out.emplace_back(i, j);
        pts.emplace_back(static_cast<double>(i), static_cast<double>(j));
        if (static_cast<int64_t>(out.size()) >= n) break;
    }
    return out;
}

double centroid_argmin_cells(const std::vector<int32_t>& cells, int W, int& ci, int& cj) {
    // ci = argmin((ii − ii.mean())² + (jj − jj.mean())²)；整数下标的和精确，均值 = 和 / n
    double si = 0, sj = 0;
    for (int32_t k : cells) {
        si += k / W;
        sj += k % W;
    }
    const double mi = si / static_cast<double>(cells.size()), mj = sj / static_cast<double>(cells.size());
    double best = INF;
    size_t bt = 0;
    for (size_t q = 0; q < cells.size(); ++q) {
        const double di = (cells[q] / W) - mi, dj = (cells[q] % W) - mj;
        const double d = di * di + dj * dj;
        if (d < best) {
            best = d;
            bt = q;
        }
    }
    ci = cells[bt] / W;
    cj = cells[bt] % W;
    return best;
}

}  // namespace

const char* res_key(int kind) { return RES_KEYS[kind]; }
ResForm res_form(int kind) { return RES_FORMS[kind]; }
int field_of(int kind) {
    for (int f = 0; f < FK_COUNT; ++f)
        if (FIELD_RES[f] == kind) return f;
    return -1;
}
int field_res(int fk) { return FIELD_RES[fk]; }
const char* zone_key(int zone) { return ZONE_KEYS[zone]; }
const char* grade_key(double q) { return q >= 0.75 ? "hi" : (q >= 0.45 ? "mid" : "lo"); }

Rng part_rng(const NodeInputs& inp, const std::string& part) {
    return entity_rng(inp.seed, ISLAND_STREAM, "island:" + std::to_string(inp.node) + ":" + part);
}

void finalize_islands(Group& g) {
    for (size_t k = 0; k < g.islands.size(); ++k) {
        IslandRec& r = g.islands[k];
        r.rim_j = pyround(r.rim, 1);
        r.keel_j = pyround(r.keel, 1);
        r.peak_j = pyround(r.peak, 1);
        r.cliff_j = pyround(r.rim - r.keel, 1);
        r.area_j = pyround(static_cast<double>(r.area_cells) * g.res_km * g.res_km, 3);   // (格数 × res) × res，与 island.json 同式
        r.cx_j = pyround(r.cx, 3);
        r.cy_j = pyround(r.cy, 3);
    }
    if (!g.islands.empty() && g.has_hydro) {
        IslandRec& r = g.islands[0];
        r.rim_j = pyround(r.rim_j + g.dz, 1);
        r.peak_j = pyround(r.peak_j + g.dz, 1);
        r.cliff_j = pyround(r.rim_j - r.keel_j, 1);
    }
}

Mask rock_site_mask(const Group& g, const Grid<uint8_t>& zone, double rock_hill_slope_deg) {
    // slope_deg 是 float32：与 Python 浮点比较时 numpy 把阈值也降成 float32
    const float thr = static_cast<float>(rock_hill_slope_deg);
    Mask out(g.H, g.W, 0);
    for (size_t k = 0; k < out.v.size(); ++k) {
        if (g.island_id.v[k] < 0) continue;
        const uint8_t cv = g.landcover.v[k];
        const bool water = g.river.v[k] > 0 || g.lake.v[k];
        const bool flood = !g.floodplain.v.empty() && g.floodplain.v[k];
        const uint8_t z = zone.v[k];
        const bool terrain = z == 1 || z == 2 || (z == 3 && static_cast<float>(g.slope.v[k]) >= thr) || cv == LC_ROCK || cv == LC_ALPINE;
        out.v[k] = (!water && !g.cliff.v[k] && g.arable.v[k] == 0 && cv != LC_WET && !flood && terrain) ? 1 : 0;
    }
    return out;
}

void open_working(Group& g, int i, int j) {
    g.landcover(i, j) = LC_ROCK;
    const int32_t k = g.patch_id(i, j);
    if (k >= 0 && g.res.deposits[k].kind == RK_TIMBER) g.patch_id(i, j) = -1;
}

void build_resources(Group& g, const Config& c) {
    const double t0 = now_s();
    const NodeInputs& inp = g.inp;
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, cell_km2 = res_km * res_km;
    const double rkm = (res_km * 1000.0) / 1000.0;      // _km 用 J["raster"]["res_m"] / 1000
    const int n_isl = static_cast<int>(g.islands.size());
    const double x0 = g.origin_x, y0 = g.origin_y;
    auto kmx = [&](int j) { return pyround(x0 + (j + 0.5) * rkm, 3); };
    auto kmy = [&](int i) { return pyround(y0 - (i + 0.5) * rkm, 3); };
    const std::string P = "resources.";
    auto cf = [&](const std::string& k) { return c.get(P + k); };

    Mask land(H, W, 0);
    for (size_t k = 0; k < N; ++k) land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
    std::vector<double> hz(N), slope(N), acc(N), cut(N), T(N);
    for (size_t k = 0; k < N; ++k) {
        hz[k] = land.v[k] ? g.height.v[k] : 0.0;
        slope[k] = f32(g.slope.v[k]);           // slope_deg / flowacc_km2 / cut_m 在 g 里是 float32
        acc[k] = f32(g.acc_km2.v[k]);
        cut[k] = f32(g.cut_m.v[k]);
        T[k] = inp.temp_sea - inp.lapse_c_per_km * hz[k] / 1000.0;
    }
    Mask river(H, W, 0), stream(H, W, 0), lake(H, W, 0), water(H, W, 0), flood(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        river.v[k] = g.river.v[k] > 0 ? 1 : 0;
        stream.v[k] = g.stream.v[k] > 0 ? 1 : 0;
        lake.v[k] = g.lake.v[k] ? 1 : 0;
        water.v[k] = (river.v[k] || lake.v[k]) ? 1 : 0;
        flood.v[k] = g.floodplain.v.empty() ? 0 : g.floodplain.v[k];
    }
    const Grid<uint8_t>& cover = g.landcover;
    // meta：boundary_type / boundary_kernel（round 3）/ boundary_axis_deg（round 1）/ layered —— Python 版读的是 island.json 里舍过的值
    const int btype = g.btype;
    const double kern = pyround(g.kernel, 3);
    const bool layered = inp.layered;
    const double axis_deg = pyround(g.axis * (180.0 / PI), 1);
    const double axis = axis_deg * (PI / 180.0);
    std::vector<double> Xk(W), Yk(H);
    for (int j = 0; j < W; ++j) Xk[j] = x0 + (j + 0.5) * res_km;
    for (int i = 0; i < H; ++i) Yk[i] = y0 - (i + 0.5) * res_km;

    // ---------- 地形区 ----------
    Resources& R = g.res;
    R = Resources();
    R.kernel_j = kern;
    const int r_cells = std::max(1, static_cast<int>(std::nearbyint(cf("relief_km") / res_km)));
    R.r_cells = r_cells;
    GridD hiw, low;
    {
        GridD hzg(H, W);
        hzg.v = hz;
        window_extrema(hzg, r_cells, land, hiw, low);
    }
    std::vector<double> relief(N, 0.0), peak_rel(N, 0.0);
    for (size_t k = 0; k < N; ++k) {
        if (land.v[k]) relief[k] = hiw.v[k] - low.v[k];
        const int id = g.island_id.v[k];
        if (id >= 0) {
            const IslandRec& isl = g.islands[id];
            peak_rel[k] = (hz[k] - isl.rim_j) / std::max(1.0, isl.peak_j - isl.rim_j);
        }
    }
    Mask river2 = binary_dilate(river, 2);
    const double mrel = cf("mountain_relief_m"), mslope = cf("mountain_slope_deg"), mpeak = cf("mountain_peak_frac");
    const double hrel = cf("hill_relief_m"), hslope = cf("hill_slope_deg"), hpeak = cf("hill_peak_frac");
    const double alpine_t = c.get("landcover.alpine_temp_c"), hipeak = cf("high_peak_frac"), vslope = cf("valley_slope_deg");
    g.zone = Grid<uint8_t>(H, W, 0);
    Grid<uint8_t>& zone = g.zone;
    for (size_t k = 0; k < N; ++k) {
        if (!land.v[k]) continue;
        const bool mount = relief[k] >= mrel || slope[k] >= mslope || (peak_rel[k] >= mpeak && relief[k] >= 0.5 * hrel);
        const bool hill = !mount && (relief[k] >= hrel || slope[k] >= hslope || peak_rel[k] >= hpeak);
        const bool alpine = mount && (T[k] < alpine_t || peak_rel[k] >= hipeak);
        const bool valley = flood.v[k] || (river2.v[k] && slope[k] < vslope);
        uint8_t z = Z_PLAIN;
        if (hill) z = Z_HILL;
        if (mount) z = Z_MOUNTAIN;
        if (alpine) z = Z_ALPINE;
        if (valley && !mount) z = Z_VALLEY;
        zone.v[k] = z;
    }
    for (size_t k = 0; k < N; ++k) {
        if (g.cliff.v[k]) zone.v[k] = Z_CLIFF;
        if (water.v[k]) zone.v[k] = Z_WATER;
    }

    // ---------- 共用量 ----------
    g.patch_id = GridI(H, W, -1);
    GridI& patch_id = g.patch_id;
    Mask taken(H, W, 0);
    std::array<std::vector<double>, FK_COUNT> F;
    for (auto& f : F) f.assign(N, 0.0);
    for (int f = 0; f < FK_COUNT; ++f) g.occ_lab[f] = GridI(H, W, -1);
    static const char* FK_KEYS[FK_COUNT] = {"ore", "sulfur", "placer", "clay", "gravel", "stone"};
    for (int f = 0; f < FK_COUNT; ++f) R.thr[f] = cf(std::string("occ_thr.") + FK_KEYS[f]);
    const auto& thr = R.thr;
    const int min_cells = static_cast<int>(cf("occ_min_cells"));
    std::vector<double> patchy(N), noise_fs(N);
    {
        Rng rn = part_rng(inp, "resources:noise");
        FractalNoise fn(rn, Xk[0], Yk[H - 1], Xk[W - 1], Yk[0], cf("patch_km"), 3, 0.5);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) patchy[static_cast<size_t>(i) * W + j] = clip(0.5 + 0.5 * fn.sample(Xk[j], Yk[i]), 0.0, 1.0);
    }
    Mask soft(H, W, 0), arable(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        soft.v[k] = (land.v[k] && !water.v[k] && !g.cliff.v[k]) ? 1 : 0;
        arable.v[k] = g.arable.v[k] > 0 ? 1 : 0;
    }
    const Mask rock_site = rock_site_mask(g, zone, cf("rock_hill_slope_deg"));
    static const char* BT_KEYS[3] = {"convergent", "divergent", "transform"};
    const double ore_gain = cf(std::string("ore_gain.") + BT_KEYS[btype]);
    R.geo_ore = std::max(cf("ore_floor"), ore_gain * kern) * (layered ? cf("ore_layered_gain") : 1.0);
    const double geo_ore = R.geo_ore;
    R.fs_rate = cf("floatstone_expose") * (1.0 + cf("floatstone_convergent_gain") * kern * (btype == 0 ? 1.0 : 0.0)) *
                (layered ? cf("floatstone_layered_gain") : 1.0);
    {
        Rng rf = part_rng(inp, "resources:floatstone");
        FractalNoise fn(rf, Xk[0], Yk[H - 1], Xk[W - 1], Yk[0], cf("floatstone_patch_km"), 3, 0.5);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) noise_fs[static_cast<size_t>(i) * W + j] = fn.sample(Xk[j], Yk[i]);
    }
    {
        Rng rl = part_rng(inp, "resources:lith");
        R.old_limestone = rl.random() < cf("old_limestone_p");
    }
    const bool old_limestone = R.old_limestone;

    auto add_deposit = [&](int kind, const std::vector<int32_t>& cells, int i, int j, const std::string& sub, double q,
                           const std::string& note, double note_arg) -> Deposit& {
        Deposit d;
        d.id = static_cast<int>(R.deposits.size());
        d.kind = kind;
        d.subtype = sub;
        d.island = g.island_id(i, j);
        d.ci = i;
        d.cj = j;
        d.kx = kmx(j);
        d.ky = kmy(i);
        d.area_km2 = pyround(static_cast<double>(cells.size()) * cell_km2, 3);
        d.elev_m = pyround(hz[static_cast<size_t>(i) * W + j], 0);
        d.slope_deg = pyround(slope[static_cast<size_t>(i) * W + j], 1);
        d.zone = zone(i, j);
        d.grade = grade_key(q);
        d.note = note;
        d.note_arg = note_arg;
        if (RES_FORMS[kind] == FORM_PATCH)
            for (int32_t k : cells) patch_id.v[k] = d.id;
        R.deposits.push_back(std::move(d));
        return R.deposits.back();
    };

    // 点 / 片：贪心取种子（同类最小间距），点只占一格，片按「距离 − 0.35 × 适宜度」长成斑块；与已有的点 / 片互斥
    auto place = [&](int kind, const Box& bx, const std::function<bool(size_t)>& cand, const std::function<double(size_t)>& score,
                     int64_t n, double area_med, double sep_km, const std::function<std::string(int, int, Rng&)>& sub_fn,
                     const std::string& note, double note_arg, Rng& rng, const std::vector<std::pair<int, int>>* seeds_in) -> int {
        if (n <= 0) return 0;
        bool any = false;
        for (int i = bx.r0; i < bx.r1 && !any; ++i)
            for (int j = bx.c0; j < bx.c1; ++j)
                if (cand(static_cast<size_t>(i) * W + j)) {
                    any = true;
                    break;
                }
        if (!any) return 0;
        auto sc_at = [&](size_t k) { return cand(k) ? score(k) : 0.0; };
        std::vector<std::pair<int, int>> seeds;
        if (seeds_in) {
            seeds = *seeds_in;
        } else {
            const int bw = bx.c1 - bx.c0;
            std::vector<double> s0(static_cast<size_t>(bx.r1 - bx.r0) * bw, 0.0);
            for (int i = bx.r0; i < bx.r1; ++i)
                for (int j = bx.c0; j < bx.c1; ++j) {
                    const size_t k = static_cast<size_t>(i) * W + j;
                    s0[static_cast<size_t>(i - bx.r0) * bw + (j - bx.c0)] = taken.v[k] ? 0.0 : sc_at(k);
                }
            seeds = seeds_of(s0, bx, n, sep_km / res_km);
        }
        int placed = 0;
        for (const auto& sd : seeds) {
            const int i = sd.first, j = sd.second;
            std::vector<int32_t> cells;
            if (RES_FORMS[kind] == FORM_POINT) {
                cells.push_back(i * W + j);
            } else {
                const double area = area_med * rng.lognormal(0.0, 0.6);
                const int64_t n_cells = std::max<int64_t>(1, static_cast<int64_t>(std::nearbyint(area / cell_km2)));
                // _grow：窗内候选格按（距离 − 0.35 × 适宜度）排序取前 n_cells 个
                const int Rw = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(std::max<int64_t>(1, n_cells)) / PI) * 1.8)) + 1;
                const int r0 = std::max(0, i - Rw), r1 = std::min(H, i + Rw + 1), c0 = std::max(0, j - Rw), c1 = std::min(W, j + Rw + 1);
                std::vector<std::pair<double, int32_t>> key;
                const double Rd = std::max(1.0, static_cast<double>(Rw));
                int local = 0;
                for (int a = r0; a < r1; ++a)
                    for (int b = c0; b < c1; ++b, ++local) {
                        const size_t k = static_cast<size_t>(a) * W + b;
                        const bool ok = (a == i && b == j) || (cand(k) && !taken.v[k]);
                        const double d = np_hypot(static_cast<double>(a - i), static_cast<double>(b - j)) / Rd;
                        key.emplace_back(ok ? d - 0.35 * sc_at(k) : INF, local);
                    }
                std::stable_sort(key.begin(), key.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
                const int w = c1 - c0;
                for (size_t q = 0; q < key.size() && static_cast<int64_t>(q) < n_cells; ++q) {
                    if (!std::isfinite(key[q].first)) continue;
                    cells.push_back((r0 + key[q].second / w) * W + (c0 + key[q].second % w));
                }
            }
            for (int32_t k : cells) taken.v[k] = 1;
            const std::string sub = sub_fn ? sub_fn(i, j, rng) : std::string();
            add_deposit(kind, cells, i, j, sub, sc_at(static_cast<size_t>(i) * W + j), note, note_arg);
            ++placed;
        }
        return placed;
    };

    // 赋存区记录：峰值格、面积、长度与走向、峰值 / 均值品位、高程范围；区号写进 occ_lab（同类重叠时品位高的占格）
    auto add_occ = [&](int kind, const std::vector<int32_t>& cells, const std::vector<double>& gv, const std::string& sub,
                       const std::string& note) -> Occurrence& {
        const int fk = field_of(kind);
        size_t pk = 0;
        for (size_t q = 1; q < gv.size(); ++q)
            if (gv[q] > gv[pk]) pk = q;
        const int i = cells[pk] / W, j = cells[pk] % W;
        Occurrence o;
        o.id = static_cast<int>(R.occ.size());
        o.kind = kind;
        o.subtype = sub;
        o.note = note;
        o.island = g.island_id(i, j);
        o.ci = i;
        o.cj = j;
        o.kx = kmx(j);
        o.ky = kmy(i);
        o.area_km2 = pyround(static_cast<double>(cells.size()) * cell_km2, 3);
        // 长度与走向（_shape）：< 3 格按格数；否则 C++ 用闭式的 2×2 协方差主轴（前端另按 numpy 重算以逐位对上）
        const size_t n = cells.size();
        if (n < 3) {
            o.length_km = pyround(std::max(1.0, std::sqrt(static_cast<double>(n))) * res_km, 2);
            o.has_axis = false;
        } else {
            double mx = 0, my = 0;
            for (int32_t k : cells) {
                mx += (k % W) * res_km;
                my += -(k / W) * res_km;
            }
            mx /= static_cast<double>(n);
            my /= static_cast<double>(n);
            double sxx = 0, syy = 0, sxy = 0;
            for (int32_t k : cells) {
                const double dx = (k % W) * res_km - mx, dy = -(k / W) * res_km - my;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
            sxx /= static_cast<double>(n - 1);
            syy /= static_cast<double>(n - 1);
            sxy /= static_cast<double>(n - 1);
            const double tr = 0.5 * (sxx + syy), det = std::sqrt(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy);
            const double lam = tr + det;
            o.length_km = pyround(std::max(4.0 * std::sqrt(std::max(lam, 0.0)), res_km), 2);
            double vx = sxy, vy = lam - sxx;
            if (std::fabs(vx) + std::fabs(vy) < 1e-300) {
                vx = lam - syy;
                vy = sxy;
            }
            if (std::fabs(vx) + std::fabs(vy) < 1e-300) {
                vx = 1.0;
                vy = 0.0;
            }
            o.has_axis = true;
            o.axis_deg = pyround(pymod(std::atan2(vy, vx) * (180.0 / PI), 180.0), 0);
        }
        double gmax = -INF, gmin_e = INF, gmax_e = -INF;
        for (size_t q = 0; q < n; ++q) {
            gmax = std::max(gmax, gv[q]);
            gmin_e = std::min(gmin_e, hz[cells[q]]);
            gmax_e = std::max(gmax_e, hz[cells[q]]);
        }
        o.grade_peak = pyround(gmax, 3);
        o.grade_mean = pyround(np_sum(gv.data(), n) / static_cast<double>(n), 3);
        o.grade = grade_key(gmax);
        o.elev_lo = pyround(gmin_e, 0);
        o.elev_hi = pyround(gmax_e, 0);
        o.zone = zone(i, j);
        o.cells = cells;
        GridI& lab = g.occ_lab[fk];
        for (size_t q = 0; q < n; ++q)
            if (gv[q] >= F[fk][cells[q]] - 1e-12) lab.v[cells[q]] = o.id;
        R.occ.push_back(std::move(o));
        return R.occ.back();
    };

    auto add_work = [&](int kind, const Occurrence& o, int i, int j, const std::string& note) {
        Working w;
        w.id = static_cast<int>(R.works.size());
        w.kind = kind;
        w.occurrence = o.id;
        w.island = g.island_id(i, j);
        w.ci = i;
        w.cj = j;
        w.kx = kmx(j);
        w.ky = kmy(i);
        w.grade = pyround(F[field_of(kind)][static_cast<size_t>(i) * W + j], 3);
        w.note = note;
        R.works.push_back(std::move(w));
    };

    // 散的赋存区（石料 / 黏土 / 砂砾 / 砂金）：品位 ≥ 阈值的格，隔 occ_merge_cells 格以内的碎块算一处，不跨岛；小于 occ_min_km2 的不成区
    auto field_occurrences = [&](int fk, const std::function<std::string(const std::vector<int32_t>&)>& sub_fn, const std::string& note) {
        Mask m(H, W, 0);
        bool any = false;
        for (size_t k = 0; k < N; ++k) {
            m.v[k] = F[fk][k] >= thr[fk] ? 1 : 0;
            any = any || m.v[k];
        }
        if (!any) return;
        Mask md = binary_dilate(m, static_cast<int>(cf("occ_merge_cells")));
        for (size_t k = 0; k < N; ++k) md.v[k] = (md.v[k] && land.v[k]) ? 1 : 0;
        GridI lab;
        const int nl = label_by_island(md, g.island_id, 8, lab);
        const int n_min = std::max(min_cells, static_cast<int>(std::ceil(cf("occ_min_km2") / cell_km2 - 1e-9)));
        std::vector<std::vector<int32_t>> groups(static_cast<size_t>(nl) + 1);
        for (size_t k = 0; k < N; ++k)
            if (m.v[k] && lab.v[k] > 0) groups[lab.v[k]].push_back(static_cast<int32_t>(k));
        for (int L = 1; L <= nl; ++L) {
            const auto& cells = groups[L];
            if (cells.empty() || static_cast<int>(cells.size()) < n_min) continue;
            std::vector<double> gv(cells.size());
            for (size_t q = 0; q < cells.size(); ++q) gv[q] = F[fk][cells[q]];
            add_occ(field_res(fk), cells, gv, sub_fn(cells), note);
        }
    };

    // ---------- 散：石料 / 黏土 / 砂砾（全群一次算） ----------
    std::vector<uint8_t> exposed(N), strong(N), chan(N);
    {
        const double gcut = cf("gorge_cut_m"), gsl = cf("gorge_slope_deg");
        const double s_lo = cf("stone_slope_lo_deg"), s_hi = cf("stone_slope_hi_deg"), sexp = cf("stone_exposed_grade");
        const double smax = cf("stone_slope_max_deg"), sfm = cf("stone_forest_mult");
        for (size_t k = 0; k < N; ++k) {
            exposed[k] = (cover.v[k] == LC_ROCK || cover.v[k] == LC_ALPINE || (cut[k] >= gcut && slope[k] >= gsl)) ? 1 : 0;
            const double g_st = std::max(clip((slope[k] - s_lo) / std::max(1e-6, s_hi - s_lo), 0.0, 1.0), exposed[k] ? sexp : 0.0);
            F[FK_STONE][k] = (rock_site.v[k] && slope[k] <= smax) ? g_st * (0.75 + 0.25 * patchy[k]) * (cover.v[k] == LC_FOREST ? sfm : 1.0) : 0.0;
        }
        Mask lake2 = binary_dilate(lake, 2);
        Mask wet(H, W, 0);
        for (size_t k = 0; k < N; ++k) wet.v[k] = cover.v[k] == LC_WET ? 1 : 0;
        Mask wet1 = binary_dilate(wet, 1);
        Mask sr(H, W, 0);
        for (size_t k = 0; k < N; ++k) sr.v[k] = (stream.v[k] || river.v[k]) ? 1 : 0;
        Mask near_stream = binary_dilate(sr, static_cast<int>(cf("clay_stream_cells")));
        const double cmax = cf("clay_slope_max_deg");
        for (size_t k = 0; k < N; ++k) {
            const bool lake_edge = lake2.v[k] && !lake.v[k];
            strong[k] = (flood.v[k] || lake_edge || wet1.v[k]) ? 1 : 0;
            const bool clay_ok = soft.v[k] && !wet.v[k] && slope[k] < cmax;
            F[FK_CLAY][k] = (clay_ok && strong[k]) ? 0.7 + 0.3 * patchy[k] : ((clay_ok && near_stream.v[k]) ? 0.35 + 0.25 * patchy[k] : 0.0);
        }
        const double gacc = cf("gravel_acc_km2"), gmax = cf("gravel_slope_max_deg");
        const int bank = static_cast<int>(cf("gravel_bank_cells"));
        Mask chm(H, W, 0);
        for (size_t k = 0; k < N; ++k) {
            chan[k] = (river.v[k] || (stream.v[k] && acc[k] >= gacc)) ? 1 : 0;
            chm.v[k] = chan[k];
        }
        Mask chd = binary_dilate(chm, bank);
        // _spread_max：河道格的汇流向八邻域扩 bank 圈
        std::vector<double> acc_bar(N);
        for (size_t k = 0; k < N; ++k) acc_bar[k] = chan[k] ? acc[k] : 0.0;
        for (int it = 0; it < bank; ++it) {
            std::vector<double> b = acc_bar;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    double v = b[static_cast<size_t>(i) * W + j];
                    for (int d = 0; d < 8; ++d) {
                        const int a = i - N8[d][0], bb = j - N8[d][1];
                        const double s = (a >= 0 && bb >= 0 && a < H && bb < W) ? acc_bar[static_cast<size_t>(a) * W + bb] : 0.0;
                        v = std::max(v, s);
                    }
                    b[static_cast<size_t>(i) * W + j] = v;
                }
            acc_bar.swap(b);
        }
        for (size_t k = 0; k < N; ++k) {
            const bool bars = chd.v[k] && !chan[k] && soft.v[k] && !wet.v[k] && slope[k] < gmax;
            F[FK_GRAVEL][k] = bars ? (0.4 + 0.6 * clip(std::log10(std::max(acc_bar[k], 1.0)) / 3.0, 0.0, 1.0)) * (0.6 + 0.4 * patchy[k]) : 0.0;
        }
    }
    std::vector<double> gold_src(N, 0.0);
    bool any_gold = false;

    // 每岛外框
    std::vector<Box> box(n_isl, Box{H, -1, W, -1});
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const int k = g.island_id(i, j);
            if (k < 0) continue;
            Box& b = box[k];
            b.r0 = std::min(b.r0, i);
            b.r1 = std::max(b.r1, i + 1);
            b.c0 = std::min(b.c0, j);
            b.c1 = std::max(b.c1, j + 1);
        }
    Mask stream1 = binary_dilate(stream, 1);
    Mask cliff2 = binary_dilate(g.cliff, 2);
    const auto& dens_key = [&](const char* key) { return cf(std::string("density_per_100km2.") + key); };
    // 主岛的河口（J["hydro"]["rivers"]：按流域面积（round 1）降序的前 12 条）
    std::vector<std::pair<std::array<int, 2>, double>> mouths;
    {
        std::vector<RiverRec> rv = g.rivers;
        std::stable_sort(rv.begin(), rv.end(), [](const RiverRec& a, const RiverRec& b) { return pyround(a.basin_km2, 1) > pyround(b.basin_km2, 1); });
        for (size_t q = 0; q < rv.size() && q < 12; ++q) mouths.push_back({{rv[q].mouth_r, rv[q].mouth_c}, pyround(rv[q].basin_km2, 1)});
    }

    for (int k = 0; k < n_isl; ++k) {
        const Box bx = box[k];
        if (bx.empty()) continue;
        const IslandRec& isl = g.islands[k];
        auto in_m = [&](size_t q) { return g.island_id.v[q] == k; };
        int64_t m_cnt = 0;
        for (int i = bx.r0; i < bx.r1; ++i)
            for (int j = bx.c0; j < bx.c1; ++j) m_cnt += g.island_id(i, j) == k;
        const double A = static_cast<double>(m_cnt) * cell_km2;
        const AgeKind age = isl.kind;
        Rng rng = part_rng(inp, "resources:" + std::to_string(k));
        auto lam = [&](const char* key, double mult) { return rng.poisson(std::max(0.0, dens_key(key) * A / 100.0 * mult)); };
        const bool big = A >= cf("min_island_km2");
        auto in_mk = [&](size_t q) { return in_m(q) && soft.v[q]; };
        const int bw = bx.c1 - bx.c0;
        // 林木（片）：大片林地（连通块 ≥ timber_min_km2），针叶 / 阔叶按年均温
        {
            Mask forest(bx.r1 - bx.r0, bw, 0);
            bool any = false;
            for (int i = bx.r0; i < bx.r1; ++i)
                for (int j = bx.c0; j < bx.c1; ++j) {
                    const size_t q = static_cast<size_t>(i) * W + j;
                    const uint8_t f = (in_m(q) && cover.v[q] == LC_FOREST) ? 1 : 0;
                    forest(i - bx.r0, j - bx.c0) = f;
                    any = any || f;
                }
            if (any) {
                GridI lab;
                const int nl = label_components(forest, 8, lab);
                if (nl) {
                    std::vector<int64_t> cnt(static_cast<size_t>(nl) + 1, 0);
                    for (int32_t x : lab.v) cnt[x]++;
                    std::vector<int> ids(nl);
                    std::iota(ids.begin(), ids.end(), 0);
                    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return cnt[a + 1] > cnt[b + 1]; });
                    const size_t cap = static_cast<size_t>(cf("timber_max_per_island"));
                    std::vector<int> keep;
                    for (size_t q = 0; q < ids.size() && q < cap; ++q)
                        if (static_cast<double>(cnt[ids[q] + 1]) * cell_km2 >= cf("timber_min_km2")) keep.push_back(ids[q] + 1);
                    std::vector<int> slot(static_cast<size_t>(nl) + 1, -1);
                    for (size_t q = 0; q < keep.size(); ++q) slot[keep[q]] = static_cast<int>(q);
                    std::vector<std::vector<int32_t>> cells(keep.size());
                    for (int a = 0; a < lab.H; ++a)
                        for (int b = 0; b < lab.W; ++b) {
                            const int L = lab(a, b);
                            if (L && slot[L] >= 0) cells[slot[L]].push_back((a + bx.r0) * W + (b + bx.c0));
                        }
                    const double conifer = cf("conifer_temp_c"), tmin = cf("timber_min_km2");
                    for (size_t q = 0; q < keep.size(); ++q) {
                        const auto& cl = cells[q];
                        std::vector<double> tv(cl.size()), hv(cl.size()), sv(cl.size());
                        for (size_t p = 0; p < cl.size(); ++p) {
                            tv[p] = T[cl[p]];
                            hv[p] = hz[cl[p]];
                            sv[p] = slope[cl[p]];
                        }
                        const double t_mean = np_mean(tv);
                        int ci, cj;
                        centroid_argmin_cells(cl, W, ci, cj);
                        const std::string sub = t_mean < conifer ? "conifer" : (t_mean < conifer + 5 ? "mixed" : "broadleaf");
                        Deposit& d = add_deposit(RK_TIMBER, cl, ci, cj, sub, static_cast<double>(cl.size()) * cell_km2 >= 4 * tmin ? 1.0 : 0.5, "", 0);
                        d.elev_m = pyround(np_mean(hv), 0);
                        d.slope_deg = pyround(np_mean(sv), 1);
                    }
                }
            }
        }
        // 泉眼（点）：溪涧源头（本格有溪、上游八邻无溪），坡度转缓处优先
        {
            bool any_st = false;
            for (int i = bx.r0; i < bx.r1 && !any_st; ++i)
                for (int j = bx.c0; j < bx.c1; ++j)
                    if (g.island_id(i, j) == k && stream(i, j)) {
                        any_st = true;
                        break;
                    }
            if (any_st) {
                auto st = [&](int a, int b) { return a >= 0 && b >= 0 && a < H && b < W && g.island_id(a, b) == k && stream(a, b); };
                std::vector<uint8_t> heads(static_cast<size_t>(bx.r1 - bx.r0) * bw, 0);
                for (int i = bx.r0; i < bx.r1; ++i)
                    for (int j = bx.c0; j < bx.c1; ++j) {
                        if (!st(i, j)) continue;
                        int nb = 0;
                        for (int d = 0; d < 8; ++d) nb += st(i - N8[d][0], j - N8[d][1]) ? 1 : 0;
                        const size_t q = static_cast<size_t>(i) * W + j;
                        heads[static_cast<size_t>(i - bx.r0) * bw + (j - bx.c0)] = (nb <= 1 && soft.v[q]) ? 1 : 0;
                    }
                const int64_t n = lam("spring", 1.0);
                auto head_at = [&](size_t q) {
                    const int i = static_cast<int>(q / W), j = static_cast<int>(q % W);
                    return i >= bx.r0 && i < bx.r1 && j >= bx.c0 && j < bx.c1 && heads[static_cast<size_t>(i - bx.r0) * bw + (j - bx.c0)] != 0;
                };
                place(RK_SPRING, bx, head_at, [&](size_t q) { return patchy[q] * clip(1.2 - slope[q] / 25.0, 0.05, 1.0); },
                      n, 0, cf("spring_sep_km"), nullptr, "", 0, rng, nullptr);
            }
        }
        // 泥炭（凉湿）/ 芦苇荡（暖）（片）：湿地
        {
            const int64_t n = lam("peat", 1.0);
            const double tmax = cf("peat_temp_max_c");
            place(RK_PEAT, bx, [&](size_t q) { return in_m(q) && cover.v[q] == LC_WET; }, [&](size_t q) { return 0.3 + 0.7 * patchy[q]; }, n,
                  cf("peat_km2"), cf("patch_sep_km"), [&](int i, int j, Rng&) { return std::string(T[static_cast<size_t>(i) * W + j] < tmax ? "peat" : "reed"); },
                  "", 0, rng, nullptr);
        }
        // 金属矿（散）：顺板块走向拉长的矿化带 = 椭圆核 × 斑块噪声，裁到岩类可放区；带内按品位点矿坑
        {
            const double omax = cf("ore_slope_max_deg");
            auto cand = [&](size_t q) { return in_m(q) && rock_site.v[q] && slope[q] <= omax; };
            // 矿种权重（按 Python 的 sorted(中文键) 次序排）
            double w_ore[M_COUNT];
            bool has[M_COUNT] = {};
            for (double& x : w_ore) x = 0.0;
            auto setw = [&](int m, double v) { w_ore[m] = v; has[m] = true; };
            if (btype == 0) { setw(M_COPPER, 0.35); setw(M_IRON, 0.25); setw(M_LEADZINC, 0.25); setw(M_GOLD, 0.15); }
            else if (btype == 1) { setw(M_IRON, 0.5); setw(M_COPPER, 0.3); setw(M_MANGANESE, 0.2); }
            else { setw(M_IRON, 0.4); setw(M_COPPER, 0.25); setw(M_TIN, 0.2); setw(M_LEADZINC, 0.15); }
            if (age == OLD) setw(M_TINTUNGSTEN, 0.2);
            if (age == YOUNG) setw(M_COPPER, (has[M_COPPER] ? w_ore[M_COPPER] : 0.0) + 0.15);
            const double age_mult = age == YOUNG ? 0.6 : (age == MID ? 1.0 : 1.2);
            bool any = false;
            for (int i = bx.r0; i < bx.r1 && !any; ++i)
                for (int j = bx.c0; j < bx.c1; ++j)
                    if (cand(static_cast<size_t>(i) * W + j)) {
                        any = true;
                        break;
                    }
            if (big && any) {
                const double mrel2 = 2.0 * mrel;
                std::vector<double> score(static_cast<size_t>(bx.r1 - bx.r0) * bw, 0.0);
                for (int i = bx.r0; i < bx.r1; ++i)
                    for (int j = bx.c0; j < bx.c1; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        if (cand(q)) score[static_cast<size_t>(i - bx.r0) * bw + (j - bx.c0)] = patchy[q] * clip(relief[q] / mrel2, 0.1, 1.0);
                    }
                const int64_t n = lam("ore", geo_ore * age_mult);
                const double elong = cf("ore_elongation");
                const double ca = std::cos(axis), sa = std::sin(axis);
                for (const auto& sd : seeds_of(score, bx, n, cf("ore_sep_km") / res_km)) {
                    const int i = sd.first, j = sd.second;
                    const double area = cf("ore_km2") * rng.lognormal(0.0, 0.6);
                    const double b_km = std::sqrt(area / (PI * elong));
                    const double a_km = elong * b_km;
                    const int Rw = static_cast<int>(std::ceil(a_km / res_km)) + 1;
                    const int r0 = std::max(0, i - Rw), r1 = std::min(H, i + Rw + 1), c0 = std::max(0, j - Rw), c1 = std::min(W, j + Rw + 1);
                    std::vector<double> gw(static_cast<size_t>(r1 - r0) * (c1 - c0), 0.0);
                    std::vector<int32_t> belt;
                    std::vector<double> gvb;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            const double dx = (b - j) * res_km, dy = -(a - i) * res_km;
                            const double along = dx * ca + dy * sa;
                            const double across = -dx * sa + dy * ca;
                            const double pa = along / a_km, pb = across / b_km;
                            double v = clip(1.0 - pa * pa - pb * pb, 0.0, 1.0) * (0.65 + 0.35 * patchy[q]);
                            if (!cand(q)) v = 0.0;
                            gw[static_cast<size_t>(a - r0) * (c1 - c0) + (b - c0)] = v;
                            if (v >= thr[FK_ORE]) {
                                belt.push_back(static_cast<int32_t>(q));
                                gvb.push_back(v);
                            }
                        }
                    if (static_cast<int>(belt.size()) < min_cells) continue;
                    // _pick：Python 按 sorted(中文键) 的次序抽，p = 权重 / 和
                    std::vector<int> ks;
                    std::vector<double> p;
                    for (int m = 0; m < M_COUNT; ++m)
                        if (has[m]) {
                            ks.push_back(m);
                            p.push_back(w_ore[m]);
                        }
                    const double ps = np_sum(p.data(), p.size());
                    for (double& x : p) x /= ps;
                    const int sub = ks[static_cast<size_t>(rng.choice_p(p))];
                    Occurrence& o = add_occ(RK_ORE, belt, gvb, METAL_KEYS[sub], "ore_belt");
                    const int oid = o.id;
                    const double o_area = o.area_km2;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            const double v = gw[static_cast<size_t>(a - r0) * (c1 - c0) + (b - c0)];
                            F[FK_ORE][q] = std::max(F[FK_ORE][q], v);
                            if (sub == M_GOLD || sub == M_COPPER) {
                                gold_src[q] = std::max(gold_src[q], v);
                                any_gold = any_gold || gold_src[q] != 0.0;
                            }
                        }
                    // 矿坑：带里品位最高的几格，彼此 ≥ 3 格
                    const int64_t n_p = std::min<int64_t>(static_cast<int64_t>(cf("ore_pits_max")), 1 + rng.poisson(o_area));
                    std::vector<size_t> ord(belt.size());
                    std::iota(ord.begin(), ord.end(), 0);
                    std::stable_sort(ord.begin(), ord.end(), [&](size_t x, size_t y) { return gvb[x] > gvb[y]; });
                    std::vector<std::pair<int, int>> chosen;
                    for (size_t t : ord) {
                        const int bi = belt[t] / W, bj = belt[t] % W;
                        bool ok = true;
                        for (const auto& ch : chosen)
                            if ((bi - ch.first) * (bi - ch.first) + (bj - ch.second) * (bj - ch.second) < 9) {
                                ok = false;
                                break;
                            }
                        if (ok) chosen.emplace_back(bi, bj);
                        if (static_cast<int64_t>(chosen.size()) >= n_p) break;
                    }
                    for (const auto& ch : chosen) add_work(RK_ORE, R.occ[oid], ch.first, ch.second, "");
                }
            }
        }
        // 浮石露头（片）：崖面 + 深切峡谷壁；露头率 × 岛龄，按斑块噪声取格，连通段各算一处
        {
            const double f_exp = clip(R.fs_rate * (age == YOUNG ? 1.2 : (age == MID ? 1.0 : 0.6)), 0.02, 0.95);
            const double gcut = cf("gorge_cut_m"), gsl = cf("gorge_slope_deg");
            for (int pass = 0; pass < 2; ++pass) {
                auto fs_cand = [&](size_t q) {
                    return pass == 0 ? (in_m(q) && g.cliff.v[q] && !water.v[q]) : (in_mk(q) && cut[q] >= gcut && slope[q] >= gsl);
                };
                std::vector<double> vals;
                for (int i = bx.r0; i < bx.r1; ++i)
                    for (int j = bx.c0; j < bx.c1; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        if (fs_cand(q)) vals.push_back(noise_fs[q]);
                    }
                if (vals.empty()) continue;
                const double q_ = np_quantile(std::move(vals), 1.0 - f_exp);
                Mask ex(bx.r1 - bx.r0, bw, 0);
                for (int i = bx.r0; i < bx.r1; ++i)
                    for (int j = bx.c0; j < bx.c1; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        ex(i - bx.r0, j - bx.c0) = (fs_cand(q) && noise_fs[q] >= q_ && !taken.v[q]) ? 1 : 0;
                    }
                GridI lab;
                const int nl = label_components(ex, 8, lab);
                if (!nl) continue;
                std::vector<std::vector<int32_t>> groups(static_cast<size_t>(nl) + 1);
                for (int a = 0; a < lab.H; ++a)
                    for (int b = 0; b < lab.W; ++b)
                        if (lab(a, b)) groups[lab(a, b)].push_back((a + bx.r0) * W + (b + bx.c0));
                for (int L = 1; L <= nl; ++L) {
                    const auto& cl = groups[L];
                    if (static_cast<int>(cl.size()) < static_cast<int>(cf("floatstone_min_cells"))) continue;
                    int ci, cj;
                    centroid_argmin_cells(cl, W, ci, cj);
                    for (int32_t q : cl) taken.v[q] = 1;
                    std::vector<double> nv(cl.size());
                    for (size_t p = 0; p < cl.size(); ++p) nv[p] = noise_fs[cl[p]];
                    add_deposit(RK_FLOATSTONE, cl, ci, cj, pass == 0 ? "cliff_face" : "gorge", clip(0.5 + 0.5 * np_mean(nv), 0, 1), "floatstone_body", 0);
                }
            }
        }
        // 温泉（点）/ 硫磺（散）：新岛（火山余热）；中年岛偶有温泉
        if (age != OLD) {
            const double hot = age == YOUNG ? 1.0 : cf("mid_hot_mult");
            auto cand_h = [&](size_t q) { return in_mk(q) && peak_rel[q] >= 0.15 && slope[q] < 25 && (stream.v[q] || stream1.v[q]); };
            const int64_t n = lam("hotspring", hot);
            place(RK_HOTSPRING, bx, cand_h, [&](size_t q) { return patchy[q] + 0.3 * peak_rel[q]; }, n, 0, cf("spring_sep_km"), nullptr, "", 0, rng, nullptr);
            if (age == YOUNG && big) {
                const double spf = cf("sulfur_peak_frac");
                auto cand = [&](size_t q) { return in_m(q) && rock_site.v[q] && peak_rel[q] >= spf; };
                std::vector<double> score(static_cast<size_t>(bx.r1 - bx.r0) * bw, 0.0);
                for (int i = bx.r0; i < bx.r1; ++i)
                    for (int j = bx.c0; j < bx.c1; ++j) {
                        const size_t q = static_cast<size_t>(i) * W + j;
                        if (cand(q)) score[static_cast<size_t>(i - bx.r0) * bw + (j - bx.c0)] = patchy[q] * peak_rel[q];
                    }
                const int64_t ns = lam("sulfur", 1.0);
                for (const auto& sd : seeds_of(score, bx, ns, cf("patch_sep_km") / res_km)) {
                    const int i = sd.first, j = sd.second;
                    const double r_km = std::sqrt(cf("sulfur_km2") * rng.lognormal(0.0, 0.6) / PI);
                    const int Rw = static_cast<int>(std::ceil(r_km / res_km)) + 1;
                    const int r0 = std::max(0, i - Rw), r1 = std::min(H, i + Rw + 1), c0 = std::max(0, j - Rw), c1 = std::min(W, j + Rw + 1);
                    std::vector<double> gw(static_cast<size_t>(r1 - r0) * (c1 - c0), 0.0);
                    std::vector<int32_t> zm;
                    std::vector<double> gvz;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            const double rr = np_hypot(static_cast<double>(a - i), static_cast<double>(b - j)) * res_km / r_km;
                            double v = clip(1.0 - rr * rr, 0.0, 1.0) * (0.7 + 0.3 * patchy[q]);
                            if (!cand(q)) v = 0.0;
                            gw[static_cast<size_t>(a - r0) * (c1 - c0) + (b - c0)] = v;
                            if (v >= thr[FK_SULFUR]) {
                                zm.push_back(static_cast<int32_t>(q));
                                gvz.push_back(v);
                            }
                        }
                    if (static_cast<int>(zm.size()) < min_cells) continue;
                    Occurrence& o = add_occ(RK_SULFUR, zm, gvz, "vent", "");
                    const int oid = o.id;
                    for (int a = r0; a < r1; ++a)
                        for (int b = c0; b < c1; ++b) {
                            const size_t q = static_cast<size_t>(a) * W + b;
                            F[FK_SULFUR][q] = std::max(F[FK_SULFUR][q], gw[static_cast<size_t>(a - r0) * (c1 - c0) + (b - c0)]);
                        }
                    add_work(RK_SULFUR, R.occ[oid], R.occ[oid].ci, R.occ[oid].cj, "");
                }
            }
        }
        // 洞穴（点）：熔岩管 / 峡谷岩洞 / 裂隙洞 / 溶洞 / 崖洞 / 底面洞 / 瀑布后洞
        const double cave_sep = cf("cave_sep_km");
        auto patchy_score = [&](size_t q) { return patchy[q]; };
        {
            const double lt = age == YOUNG ? 1.0 : (age == MID ? cf("lava_tube_mid_mult") : 0.0);
            if (lt > 0) {
                auto cand = [&](size_t q) { return in_mk(q) && slope[q] >= 3 && slope[q] <= 18 && peak_rel[q] >= 0.25 && peak_rel[q] <= 0.8; };
                const int64_t n = lam("lava_tube", lt);
                const std::string sub = age == YOUNG ? "lava_tube" : "lava_tube_collapsed";
                place(RK_CAVE, bx, cand, patchy_score, n, 0, cave_sep, [&](int, int, Rng&) { return sub; }, "", 0, rng, nullptr);
            }
            const double gcut = cf("gorge_cut_m"), gsl = cf("gorge_slope_deg");
            {
                const int64_t n = lam("gorge_cave", 1.0);
                place(RK_CAVE, bx, [&](size_t q) { return in_mk(q) && cut[q] >= gcut && slope[q] >= gsl; }, patchy_score, n, 0, cave_sep,
                      [&](int, int, Rng&) { return std::string("gorge_cave"); }, "", 0, rng, nullptr);
            }
            {
                const int64_t n = lam("fracture_cave", 1.0 + 2.0 * kern);
                place(RK_CAVE, bx, [&](size_t q) { return in_mk(q) && (zone.v[q] == 1 || zone.v[q] == 2) && slope[q] >= 15; }, patchy_score, n, 0,
                      cave_sep, [&](int, int, Rng&) { return std::string("fracture_cave"); }, "", 0, rng, nullptr);
            }
            if (age == OLD && old_limestone) {
                const int64_t n = lam("karst", 1.0);
                place(RK_CAVE, bx, [&](size_t q) { return in_mk(q) && slope[q] <= 15 && (acc[q] >= 0.3 || zone.v[q] == 2 || zone.v[q] == 3); },
                      patchy_score, n, 0, cave_sep, [&](int, int, Rng& r) {
                          static const char* K[3] = {"karst_cave", "sinkhole", "underground_river"};
                          return std::string(K[r.integers(0, 3)]);
                      }, "", 0, rng, nullptr);
            }
        }
        {
            int64_t rim_n = 0;
            for (int i = bx.r0; i < bx.r1; ++i)
                for (int j = bx.c0; j < bx.c1; ++j) {
                    const size_t q = static_cast<size_t>(i) * W + j;
                    rim_n += (in_m(q) && g.cliff.v[q] && !water.v[q]) ? 1 : 0;
                }
            if (rim_n > 0) {
                auto rim_cells = [&](size_t q) { return in_m(q) && g.cliff.v[q] && !water.v[q]; };
                const int64_t n_c = rng.poisson(dens_key("cliff_cave") * static_cast<double>(rim_n) * res_km / 10.0);
                const double cmul = clip(isl.cliff_j / 150.0, 0.2, 1.0);
                place(RK_CAVE, bx, rim_cells, [&](size_t q) { return patchy[q] * cmul; }, n_c, 0, cave_sep,
                      [&](int, int, Rng&) { return std::string("cliff_cave"); }, "cliff_cave", 0, rng, nullptr);
                const int64_t n_u = rng.poisson(dens_key("underside_cave") * static_cast<double>(rim_n) * res_km / 10.0);
                place(RK_CAVE, bx, rim_cells, patchy_score, n_u, 0, cave_sep, [&](int, int, Rng&) { return std::string("underside_cave"); },
                      "underside_cave", 0, rng, nullptr);
                // 瀑布后洞：常年河跌下崖缘处（河口旁 3 格内的崖缘格）
                if (k == 0) {
                    for (const auto& mv : mouths) {
                        if (rng.random() >= cf("waterfall_cave_p")) continue;
                        const int mi = mv.first[0], mj = mv.first[1];
                        const int r0 = std::max(0, mi - 3), c0 = std::max(0, mj - 3);
                        const int r1 = std::min(H, mi + 4), c1 = std::min(W, mj + 4);
                        int bi = -1, bj = -1;
                        double best = INF;
                        for (int a = r0; a < r1; ++a)
                            for (int b = c0; b < c1; ++b) {
                                const size_t q = static_cast<size_t>(a) * W + b;
                                if (!(rim_cells(q) && !taken.v[q])) continue;
                                const double d = static_cast<double>((a - mi) * (a - mi) + (b - mj) * (b - mj));
                                if (d < best) {
                                    best = d;
                                    bi = a;
                                    bj = b;
                                }
                            }
                        if (bi >= 0) {
                            std::vector<std::pair<int, int>> sd{{bi, bj}};
                            place(RK_CAVE, bx, rim_cells, [](size_t) { return 1.0; }, 1, 0, 0.0,
                                  [&](int, int, Rng&) { return std::string("waterfall_cave"); }, "waterfall_cave", mv.second, rng, &sd);
                        }
                    }
                }
                // 鸟粪石（片）：小岛崖顶（海鸟 / 飞兽聚居）
                if (A <= cf("guano_max_island_km2") && rng.random() < cf("guano_p")) {
                    place(RK_GUANO, bx, [&](size_t q) { return in_m(q) && cliff2.v[q] && !water.v[q]; }, [&](size_t q) { return patchy[q] + 0.2; }, 1,
                          std::min(0.3 * A, cf("guano_km2")), 1.0, nullptr, "", 0, rng, nullptr);
                }
            }
        }
    }

    // ---------- 散：砂金（上游金 / 铜矿化带的平均品位，带给河边滩地）、石料 / 黏土 / 砂砾的赋存区 ----------
    if (any_gold) {
        // accumulate(route_h, ok, recv, weight=gold_src)：Python 按路由面降序逐格 Al[r] += Al[k]。
        // 路由面严格向下游递减，每格的终值只依赖上游，所以按拓扑序算、每个下游格按（路由面降序, 下标升序）依次加上各上游格即逐位相同
        std::vector<double> Al(N, 0.0);
        std::vector<int64_t> recv(N, -1);
        std::vector<int32_t> indeg(N, 0);
        std::vector<uint8_t> ok(N, 0);
        for (size_t q = 0; q < N; ++q) {
            ok[q] = (land.v[q] && std::isfinite(g.route_h.v[q])) ? 1 : 0;
            if (!ok[q]) continue;
            Al[q] = gold_src[q];
            if (g.recv_i.v[q] >= 0) {
                recv[q] = static_cast<int64_t>(g.recv_i.v[q]) * W + g.recv_j.v[q];
                indeg[recv[q]]++;
            }
        }
        std::vector<int32_t> start(N + 1, 0);
        for (size_t q = 0; q < N; ++q)
            if (ok[q] && recv[q] >= 0) start[recv[q] + 1]++;
        for (size_t q = 0; q < N; ++q) start[q + 1] += start[q];
        std::vector<int32_t> donors(start[N]);
        {
            std::vector<int32_t> fill(start.begin(), start.end() - 1);
            for (size_t q = 0; q < N; ++q)
                if (ok[q] && recv[q] >= 0) donors[fill[recv[q]]++] = static_cast<int32_t>(q);
        }
        std::vector<int32_t> stack;
        for (size_t q = 0; q < N; ++q)
            if (ok[q] && indeg[q] == 0) stack.push_back(static_cast<int32_t>(q));
        std::vector<int32_t> ds;
        while (!stack.empty()) {
            const int32_t q = stack.back();
            stack.pop_back();
            // 上游格都已终值：按（路由面降序, 下标升序）加
            ds.assign(donors.begin() + start[q], donors.begin() + start[q + 1]);
            std::sort(ds.begin(), ds.end(), [&](int32_t a, int32_t b) {
                return g.route_h.v[a] > g.route_h.v[b] || (g.route_h.v[a] == g.route_h.v[b] && a < b);
            });
            double s = gold_src[q];
            for (int32_t d : ds) s += Al[d];
            Al[q] = s;
            if (recv[q] >= 0 && ok[recv[q]] && --indeg[recv[q]] == 0) stack.push_back(static_cast<int32_t>(recv[q]));
        }
        const int bank = static_cast<int>(cf("gravel_bank_cells"));
        std::vector<double> mean_up(N, 0.0);
        for (size_t q = 0; q < N; ++q)
            if (chan[q]) mean_up[q] = (Al[q] * cell_km2) / std::max(acc[q], cell_km2);
        for (int it = 0; it < bank; ++it) {
            std::vector<double> b = mean_up;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    double v = b[static_cast<size_t>(i) * W + j];
                    for (int d = 0; d < 8; ++d) {
                        const int a = i - N8[d][0], bb = j - N8[d][1];
                        const double s = (a >= 0 && bb >= 0 && a < H && bb < W) ? mean_up[static_cast<size_t>(a) * W + bb] : 0.0;
                        v = std::max(v, s);
                    }
                    b[static_cast<size_t>(i) * W + j] = v;
                }
            mean_up.swap(b);
        }
        const double gain = cf("placer_gain");
        for (size_t q = 0; q < N; ++q) F[FK_PLACER][q] = F[FK_GRAVEL][q] * clip(gain * mean_up[q], 0.0, 1.0);
    }
    field_occurrences(FK_STONE, [&](const std::vector<int32_t>& cells) {
        const AgeKind a = g.islands[g.island_id.v[cells[0]]].kind;
        return std::string(a == YOUNG ? "basalt" : (a == MID ? "andesite" : (old_limestone ? "limestone" : "sandstone")));
    }, "");
    field_occurrences(FK_CLAY, [&](const std::vector<int32_t>& cells) {
        int64_t s = 0;
        for (int32_t q : cells) s += strong[q];
        return std::string(static_cast<double>(s) / static_cast<double>(cells.size()) >= 0.5 ? "lake_clay" : "stream_clay");
    }, "");
    field_occurrences(FK_GRAVEL, [&](const std::vector<int32_t>&) { return std::string("river_gravel"); }, "");
    field_occurrences(FK_PLACER, [&](const std::vector<int32_t>&) { return std::string("placer_gold"); }, "placer_upstream");
    for (int f = 0; f < FK_COUNT; ++f) {
        g.res_field[f] = Grid<uint8_t>(H, W, 0);
        for (size_t q = 0; q < N; ++q) g.res_field[f].v[q] = static_cast<uint8_t>(std::nearbyint(clip(F[f][q], 0.0, 1.0) * 255.0));
    }
    for (const Working& w : R.works)
        if (w.kind == RK_ORE || w.kind == RK_SULFUR || w.kind == RK_STONE) open_working(g, w.ci, w.cj);
    g.has_resources = true;
    sync_resources(g);
    g.sec_resources = now_s() - t0;
}

void sync_resources(Group& g) {
    Resources& R = g.res;
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    // sync_resources 的 cell_km2 = g["res_km"] ** 2（Python 浮点的幂 = C 的 pow，不是 res·res）
    const double res_km = g.res_km, cell_km2 = c_pow(res_km, 2.0), rkm = (res_km * 1000.0) / 1000.0;
    const size_t nd = R.deposits.size();
    std::vector<std::vector<int32_t>> cells(nd);
    for (size_t q = 0; q < N; ++q) {
        const int32_t p = g.patch_id.v[q];
        if (p >= 0) cells[p].push_back(static_cast<int32_t>(q));
    }
    for (Deposit& d : R.deposits) {
        if (res_form(d.kind) != FORM_PATCH) continue;
        const auto& cl = cells[d.id];
        d.area_km2 = pyround(static_cast<double>(cl.size()) * cell_km2, 3);
        if (cl.empty()) {
            d.cleared = true;
            if (d.kind == RK_TIMBER) d.note = "timber_cleared";
            continue;
        }
        if (g.patch_id(d.ci, d.cj) != d.id) {
            int ci, cj;
            centroid_argmin_cells(cl, W, ci, cj);
            d.ci = ci;
            d.cj = cj;
            d.kx = pyround(g.origin_x + (cj + 0.5) * rkm, 3);
            d.ky = pyround(g.origin_y - (ci + 0.5) * rkm, 3);
        }
    }
    // 主导栅格（显示用）：按 DOMINANT_ORDER 后画盖先画
    std::vector<uint8_t> code_of(nd, 0);
    for (const Deposit& d : R.deposits)
        if (res_form(d.kind) == FORM_PATCH) code_of[d.id] = static_cast<uint8_t>(d.kind);
    g.resource = Grid<uint8_t>(H, W, 0);
    for (int key : DOMINANT_ORDER) {
        const ResForm f = res_form(key);
        if (f == FORM_FIELD) {
            const int fk = field_of(key);
            const int cut = static_cast<int>(std::nearbyint(R.thr[fk] * 255.0));
            for (size_t q = 0; q < N; ++q)
                if (g.res_field[fk].v[q] >= cut) g.resource.v[q] = static_cast<uint8_t>(key);
        } else if (f == FORM_PATCH) {
            for (size_t q = 0; q < N; ++q) {
                const int32_t p = g.patch_id.v[q];
                if (p >= 0 && code_of[p] == key) g.resource.v[q] = static_cast<uint8_t>(key);
            }
        } else {
            for (const Deposit& d : R.deposits)
                if (d.kind == key) g.resource(d.ci, d.cj) = static_cast<uint8_t>(key);
        }
    }
    for (Occurrence& o : R.occ) o.n_workings = 0;
    for (const Working& w : R.works) R.occ[w.occurrence].n_workings++;
}

Json deposit_json(const Deposit& d) {
    Json j = Json::obj();
    j.set("id", d.id);
    j.set("kind", res_key(d.kind));
    j.set("subtype", d.subtype.empty() ? Json() : Json(d.subtype));
    j.set("island", d.island);
    j.set("cell", Json::ipair(d.ci, d.cj));
    j.set("km", Json::pair(d.kx, d.ky));
    j.set("area_km2", d.area_km2);
    j.set("elev_m", d.elev_m);
    j.set("slope_deg", d.slope_deg);
    j.set("zone", zone_key(d.zone));
    j.set("grade", d.grade);
    if (!d.note.empty()) {
        j.set("note", d.note);
        if (d.note == "waterfall_cave") j.set("note_arg", d.note_arg);
    }
    if (d.cleared) j.set("cleared", true);
    if (d.has_area_before) j.set("area_before_clearing_km2", d.area_before);
    return j;
}

Json occurrence_json(const Occurrence& o) {
    Json j = Json::obj();
    j.set("id", o.id);
    j.set("kind", res_key(o.kind));
    j.set("subtype", o.subtype.empty() ? Json() : Json(o.subtype));
    j.set("island", o.island);
    j.set("cell", Json::ipair(o.ci, o.cj));
    j.set("km", Json::pair(o.kx, o.ky));
    j.set("area_km2", o.area_km2);
    j.set("length_km", o.length_km);
    j.set("axis_deg", o.has_axis ? Json(o.axis_deg) : Json());
    j.set("grade_peak", o.grade_peak);
    j.set("grade_mean", o.grade_mean);
    j.set("grade", o.grade);
    j.set("elev_m", Json::pair(o.elev_lo, o.elev_hi));
    j.set("zone", zone_key(o.zone));
    if (!o.note.empty()) j.set("note", o.note);
    j.set("n_workings", o.n_workings);
    return j;
}

Json working_json(const Working& w) {
    Json j = Json::obj();
    j.set("id", w.id);
    j.set("kind", res_key(w.kind));
    j.set("occurrence", w.occurrence);
    j.set("island", w.island);
    j.set("cell", Json::ipair(w.ci, w.cj));
    j.set("km", Json::pair(w.kx, w.ky));
    j.set("grade", w.grade);
    j.set("villages", Json::arr_of(w.villages));
    j.set("special", w.special < 0 ? Json() : Json(w.special));
    if (!w.note.empty()) j.set("note", w.note);
    return j;
}

}  // namespace skyisle::island
