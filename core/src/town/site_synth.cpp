// 合成地形（PLAN-TOWN 7.1）：十种预设，调风格、出画廊用。平面坐标以窗口中心为原点；「向阳」一侧 = 赤道一侧（北半球是南）。
// 各预设的尺寸都是绝对米数（河宽、谷宽、坡度），窗口只是裁多大一块。
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "skyisle/town/raster.hpp"
#include "skyisle/town/site.hpp"

namespace skyisle::town {

namespace {

const char* KINDS[] = {"plain", "meander", "valley", "sunslope", "hilltop", "lakeshore", "fjord", "gully", "confluence", "rim"};

Site blank(double half_m, double res_m, double lat_deg) {
    Site s;
    s.res_m = res_m;
    s.W = s.H = std::max(8, static_cast<int>(std::ceil(2.0 * half_m / res_m)));
    s.x0 = -0.5 * s.W * res_m, s.y0 = 0.5 * s.H * res_m;
    s.lat_deg = lat_deg;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    s.height = GridF(s.H, s.W, 0.0f);
    s.water_level = GridF(s.H, s.W, nan);
    s.water = Grid<uint8_t>(s.H, s.W, WATER_NONE);
    s.sky = Mask(s.H, s.W, 0);
    s.farmland = Mask(s.H, s.W, 0);
    s.flood = Mask(s.H, s.W, 0);
    s.landcover = Grid<uint8_t>(s.H, s.W, LC_GRASS);
    s.island = Grid<int16_t>(s.H, s.W, 0);
    return s;
}

// a → b 的直线加一条横向摆动（两端收到 0），每 step 米一个顶点
std::vector<V2> wavy(V2 a, V2 b, double amp, double wavelength, uint64_t seed, double step) {
    const double L = len(b - a);
    const int n = std::max(2, static_cast<int>(std::ceil(L / step)) + 1);
    const V2 dir = (b - a) * (1.0 / L);
    const V2 nrm{-dir.y, dir.x};
    const double ph = 3.141592653589793 * hash_unit(seed, 1, 2);
    std::vector<V2> out;
    for (int k = 0; k < n; ++k) {
        const double t = static_cast<double>(k) / (n - 1), s = t * L;
        const double taper = std::sin(3.141592653589793 * t);
        const double off = amp * taper * (std::sin(2.0 * 3.141592653589793 * s / wavelength + ph) + 0.35 * fbm(seed, s, 0.0, 0.45 * wavelength, 2));
        out.push_back(a + dir * s + nrm * off);
    }
    return out;
}

// 若干折线的距离场与「最近那条线上的值」（线先光栅化成种子格，再做欧氏距离变换）
struct LineField {
    GridF dist;     // m
    GridF val;      // 最近种子格的值（顶点值沿线线性插值）
};
LineField line_field(const Site& s, const std::vector<std::vector<V2>>& lines, const std::vector<std::vector<double>>& vals) {
    Mask seed(s.H, s.W, 0);
    GridF sv(s.H, s.W, 0.0f);
    for (size_t l = 0; l < lines.size(); ++l) {
        const auto& L = lines[l];
        for (size_t k = 0; k + 1 < L.size(); ++k) {
            const double seg = len(L[k + 1] - L[k]);
            const int n = std::max(1, static_cast<int>(std::ceil(seg / (0.5 * s.res_m))));
            for (int q = 0; q <= n; ++q) {
                const double t = static_cast<double>(q) / n;
                int i, j;
                if (!s.cell_of(lerp(L[k], L[k + 1], t), i, j)) continue;
                seed(i, j) = 1;
                sv(i, j) = static_cast<float>(vals[l][k] + (vals[l][k + 1] - vals[l][k]) * t);
            }
        }
    }
    LineField f;
    Grid<int32_t> src;
    edt(seed, f.dist, &src);
    f.val = GridF(s.H, s.W, 0.0f);
    for (size_t k = 0; k < src.size(); ++k) {
        f.dist[k] *= static_cast<float>(s.res_m);
        if (src[k] >= 0) f.val[k] = sv[static_cast<size_t>(src[k])];
        else f.dist[k] = std::numeric_limits<float>::infinity();
    }
    return f;
}

// 沿线的水面：start 起按坡降下降
std::vector<double> falling(const std::vector<V2>& line, double start, double gradient) {
    std::vector<double> v(line.size(), start);
    for (size_t k = 1; k < line.size(); ++k) v[k] = v[k - 1] - gradient * len(line[k] - line[k - 1]);
    return v;
}

River make_river(std::vector<V2> line, std::vector<double> surface, double width, double depth, bool seasonal) {
    River r;
    r.line = std::move(line);
    r.surface_m = std::move(surface);
    for (size_t k = 1; k < r.surface_m.size(); ++k) r.surface_m[k] = std::min(r.surface_m[k], r.surface_m[k - 1]);
    r.width_m.assign(r.line.size(), width);
    r.depth_m.assign(r.line.size(), depth);
    r.seasonal = seasonal;
    return r;
}

// 顺地形走的溪：水面 = 地面 − 深度 − 0.3（沿线单调不增）
River terrain_stream(const Site& s, std::vector<V2> line, double width, double depth, bool seasonal,
                     const std::function<double(V2)>& ground) {
    std::vector<double> surf(line.size());
    for (size_t k = 0; k < line.size(); ++k) surf[k] = ground(line[k]) - depth - 0.3;
    (void)s;
    return make_river(std::move(line), std::move(surf), width, depth, seasonal);
}

// 按函数填高程；返回的 bool 为 true 的格记为虚空
void fill(Site& s, const std::function<double(V2)>& h, const std::function<bool(V2)>& is_sky = nullptr) {
    for (int i = 0; i < s.H; ++i)
        for (int j = 0; j < s.W; ++j) {
            const V2 p = s.center(i, j);
            if (is_sky && is_sky(p)) {
                s.sky(i, j) = 1;
                s.island(i, j) = -1;
                continue;
            }
            s.height(i, j) = static_cast<float>(h(p));
        }
}

// 自动划田：坡缓的是平田、再陡一些的开梯田，非水非虚空，按噪声留出一些不种的地块
void auto_farmland(Site& s, const Config& c, uint64_t seed) {
    const double tmax = std::tan(c.get("synth.farm_max_slope_deg") * 3.141592653589793 / 180.0);
    const double tter = std::tan(c.get("synth.terrace_max_slope_deg") * 3.141592653589793 / 180.0);
    const double cut = c.get("synth.farm_noise_cut");
    for (int i = 1; i + 1 < s.H; ++i)
        for (int j = 1; j + 1 < s.W; ++j) {
            if (s.sky(i, j) || s.water(i, j) || s.sky(i, j + 1) || s.sky(i, j - 1) || s.sky(i - 1, j) || s.sky(i + 1, j)) continue;
            const double gx = (s.height(i, j + 1) - s.height(i, j - 1)) / (2.0 * s.res_m);
            const double gy = (s.height(i - 1, j) - s.height(i + 1, j)) / (2.0 * s.res_m);
            const V2 p = s.center(i, j);
            const double g = std::hypot(gx, gy);
            if (g >= tter || fbm(mix64(seed ^ 0x77), p.x, p.y, 220.0, 3) <= cut) continue;
            s.farmland(i, j) = 1;
            if (g >= tmax) s.landcover(i, j) = LC_TERRACE;
        }
}

// 湖 / 海：in(p) 为真的格是水，水面 level，水底按离岸距离加深
void fill_water(Site& s, const std::function<bool(V2)>& in, double level, double depth, uint8_t kind) {
    Mask w(s.H, s.W, 0);
    for (int i = 0; i < s.H; ++i)
        for (int j = 0; j < s.W; ++j)
            if (!s.sky(i, j) && in(s.center(i, j))) w(i, j) = 1;
    Mask dry(s.H, s.W, 0);
    for (size_t k = 0; k < w.size(); ++k) dry[k] = w[k] ? 0 : 1;
    GridF d;
    edt(dry, d, nullptr);
    for (size_t k = 0; k < w.size(); ++k) {
        if (!w[k]) continue;
        s.water[k] = kind;
        s.water_level[k] = static_cast<float>(level);
        const double off = d[k] * s.res_m;
        s.height[k] = static_cast<float>(level - depth * std::min(1.0, 0.2 + off / 60.0));
    }
}

}  // namespace

bool synth_kind_known(const std::string& kind) {
    for (const char* k : KINDS)
        if (kind == k) return true;
    return false;
}

Site build_site_synth(const std::string& kind, double half_m, double res_m, double lat_deg, uint64_t seed, const Config& c) {
    Site s = blank(half_m, res_m, lat_deg);
    const double base = c.get("synth.base_m");
    const double sg = lat_deg >= 0.0 ? 1.0 : -1.0;   // 向极一侧的 y 符号：v = sg·y > 0 是背阳（向极），< 0 是向阳（向赤道）
    const uint64_t sd = seed;
    // 地貌的尺寸与窗口无关：特征按 scale_m 摆（岸线离中心多远、支沟多长……），线一律画到 ±extent_m——窗口大小只决定裁多大一块
    const double Lm = c.get("synth.scale_m");
    const double H = c.get("synth.extent_m");
    auto K = [&](const char* key) { return c.get(std::string("synth.") + kind + "." + key); };
    auto undul = [&](V2 p, double relief) {
        return relief * fbm(mix64(sd ^ 0x101), p.x, p.y, 400.0, 3) + 0.25 * fbm(mix64(sd ^ 0x202), p.x, p.y, 45.0, 2);
    };

    if (kind == "plain") {
        const double relief = K("relief_m");
        fill(s, [&](V2 p) { return base + undul(p, relief); });
    } else if (kind == "meander" || kind == "valley") {
        const double w = K("river_width_m"), dep = K("river_depth_m"), g = K("gradient");
        const std::vector<V2> line = wavy({-H, 0.0}, {H, 0.0}, K("amplitude_m"), K("wavelength_m"), mix64(sd ^ 0x301), 8.0);
        const std::vector<double> surf = falling(line, base, g);
        const LineField f = line_field(s, {line}, {surf});
        if (kind == "meander") {
            const double relief = K("relief_m"), terrace = K("floodplain_rise");
            fill(s, [&](V2 p) {
                int i, j;
                s.cell_of(p, i, j);
                const double d = f.dist(i, j);
                return f.val(i, j) + 0.8 + terrace * std::min(d, 600.0) + undul(p, relief) * std::min(1.0, d / 80.0);
            });
        } else {
            const double floor = K("floor_width_m") * 0.5, wall_h = K("wall_height_m");
            const double wall_t = std::tan(K("wall_slope_deg") * 3.141592653589793 / 180.0);
            fill(s, [&](V2 p) {
                int i, j;
                s.cell_of(p, i, j);
                const double d = f.dist(i, j), ex = std::max(0.0, d - floor);
                const double wall = wall_h * (1.0 - std::exp(-ex * wall_t / wall_h));
                return f.val(i, j) + 0.8 + K("floor_rise") * std::min(d, floor) + wall + undul(p, 1.0 + 0.03 * wall) * std::min(1.0, d / 60.0);
            });
        }
        s.rivers.push_back(make_river(line, surf, w, dep, false));
    } else if (kind == "sunslope") {
        const double t = std::tan(K("slope_deg") * 3.141592653589793 / 180.0);
        auto ground = [&](V2 p) { return base + t * (sg * p.y + Lm) + undul(p, 1.5); };
        fill(s, ground);
        const double yv = -K("stream_offset") * Lm * sg;   // 坡脚（向阳一侧）的溪
        s.rivers.push_back(terrain_stream(s, wavy({-H, yv}, {H, yv}, 25.0, 500.0, mix64(sd ^ 0x401), 6.0), K("stream_width_m"), K("stream_depth_m"),
                                          false, ground));
    } else if (kind == "hilltop") {
        const double hh = K("height_m"), R = K("radius_m");
        const double spur = 2.0 * 3.141592653589793 * (0.5 + 0.5 * hash_unit(sd, 7, 7));
        const V2 sdir = bearing_vec(spur);
        fill(s, [&](V2 p) {
            const double r2 = dot(p, p) / (R * R);
            const double along = dot(p, sdir), across = cross(sdir, p);
            const double sp = 0.45 * hh * std::exp(-std::pow((along - 1.2 * R) / (1.1 * R), 2) - std::pow(across / (0.45 * R), 2));
            return base + hh * std::exp(-r2) + sp * smoothstep(clip(along / R + 0.5, 0.0, 1.0)) + undul(p, 2.0);
        });
    } else if (kind == "lakeshore") {
        const double off = K("shore_offset") * Lm, rise = K("rise"), wave = K("wave_m");
        auto shore_v = [&](double x) { return -off + wave * fbm(mix64(sd ^ 0x501), x, 0.0, 350.0, 3); };
        auto ground = [&](V2 p) { return base + 1.0 + rise * std::max(0.0, sg * p.y - shore_v(p.x)) + undul(p, 1.2); };
        fill(s, ground);
        fill_water(s, [&](V2 p) { return sg * p.y < shore_v(p.x); }, base, K("lake_depth_m"), WATER_LAKE);
        const double xs = 0.25 * Lm;
        s.rivers.push_back(terrain_stream(s, wavy({xs, sg * H}, {xs + 30.0, sg * (shore_v(xs) - 20.0)}, 30.0, 400.0, mix64(sd ^ 0x502), 6.0),
                                          K("stream_width_m"), K("stream_depth_m"), false, ground));
    } else if (kind == "fjord") {
        const double off = K("water_offset") * Lm, strip = K("strip_m"), wall_h = K("wall_height_m");
        const double wall_t = std::tan(K("wall_slope_deg") * 3.141592653589793 / 180.0);
        auto shore_v = [&](double x) { return -off + K("wave_m") * fbm(mix64(sd ^ 0x601), x, 0.0, 500.0, 3); };
        auto ground = [&](V2 p) {
            const double v = sg * p.y - shore_v(p.x);
            const double ex = std::max(0.0, v - strip);
            const double wall = wall_h * (1.0 - std::exp(-ex * wall_t / wall_h));
            return base + 1.5 + 0.03 * std::min(std::max(v, 0.0), strip) + wall + undul(p, 1.0 + 0.04 * wall);
        };
        fill(s, ground);
        fill_water(s, [&](V2 p) { return sg * p.y < shore_v(p.x); }, base, K("water_depth_m"), WATER_SEA);
        const double xs = -0.3 * Lm;
        s.rivers.push_back(terrain_stream(s, wavy({xs, sg * H}, {xs - 20.0, sg * (shore_v(xs) - 15.0)}, 15.0, 300.0, mix64(sd ^ 0x602), 5.0),
                                          K("stream_width_m"), K("stream_depth_m"), false, ground));
    } else if (kind == "gully") {
        const double top = base + K("table_m"), dep = K("depth_m"), fhw = 0.5 * K("floor_width_m"), wall_w = K("wall_width_m");
        const double g = K("gradient");
        std::vector<std::vector<V2>> lines;
        std::vector<std::vector<double>> floors;
        const std::vector<V2> main = wavy({-H, 0.0}, {H, 0.0}, 70.0, 700.0, mix64(sd ^ 0x701), 6.0);
        lines.push_back(main);
        floors.push_back(falling(main, top - dep, g));
        const int nb = c.geti("synth.gully.branches");
        for (int b = 0; b < nb; ++b) {
            // 支沟在中心 ±1.2 scale_m 内均匀排开，交替朝两侧
            const double xb = (-1.0 + 2.0 * (b + 0.5) / nb) * 1.2 * Lm;
            size_t k = 0;
            for (size_t q = 1; q < main.size(); ++q)
                if (std::fabs(main[q].x - xb) < std::fabs(main[k].x - xb)) k = q;
            const double side = (b % 2 == 0) ? 1.0 : -1.0;
            const double L = (0.45 + 0.35 * (0.5 + 0.5 * hash_unit(sd, b, 11))) * Lm;
            const V2 a = main[k];
            const V2 e = a + V2{(0.4 * hash_unit(sd, b, 12)) * L, side * L};
            const std::vector<V2> br = wavy(e, a, 25.0, 260.0, mix64(sd ^ (0x710 + b)), 6.0);   // 从沟头流向主沟
            const double f0 = floors[0][k];
            std::vector<double> fl(br.size());
            for (size_t q = 0; q < br.size(); ++q) {
                const double u = static_cast<double>(q) / (br.size() - 1);   // 0 = 沟头，1 = 汇口
                fl[q] = top - (dep * 0.25 + dep * 0.75 * u) + (f0 - (top - dep)) * u;
            }
            lines.push_back(br);
            floors.push_back(fl);
        }
        std::vector<LineField> fs;
        for (size_t l = 0; l < lines.size(); ++l) fs.push_back(line_field(s, {lines[l]}, {floors[l]}));
        fill(s, [&](V2 p) {
            int i, j;
            s.cell_of(p, i, j);
            double h = top + undul(p, 1.2);
            for (const LineField& f : fs) {
                const double d = f.dist(i, j);
                if (!(d < fhw + wall_w)) continue;
                const double q = clip((d - fhw) / wall_w, 0.0, 1.0);
                const double fl = f.val(i, j);
                h = std::min(h, fl + (top - fl) * smoothstep(q) + 0.3 * fbm(mix64(sd ^ 0x7ff), p.x, p.y, 20.0, 2));
            }
            return h;
        });
        s.rivers.push_back(make_river(main, floors[0], K("stream_width_m"), 0.3, true));
    } else if (kind == "confluence") {
        const double w = K("river_width_m"), dep = K("river_depth_m"), g = K("gradient");
        const V2 o{0.0, 0.0};
        const std::vector<V2> a = wavy({-H, sg * 0.55 * H}, o, 60.0, 600.0, mix64(sd ^ 0x901), 8.0);
        const std::vector<V2> b = wavy({H, sg * 0.45 * H}, o, 50.0, 550.0, mix64(sd ^ 0x902), 8.0);
        const std::vector<V2> cc = wavy(o, {0.12 * H, -sg * H}, 60.0, 700.0, mix64(sd ^ 0x903), 8.0);
        const double la = polyline_length(a), lb = polyline_length(b);
        const std::vector<double> sa = falling(a, base + g * la, g), sb = falling(b, base + g * lb, g), scc = falling(cc, base, g);
        const LineField f = line_field(s, {a, b, cc}, {sa, sb, scc});
        const double relief = K("relief_m"), terrace = K("floodplain_rise");
        fill(s, [&](V2 p) {
            int i, j;
            s.cell_of(p, i, j);
            const double d = f.dist(i, j);
            return f.val(i, j) + 0.8 + terrace * std::min(d, 600.0) + undul(p, relief) * std::min(1.0, d / 80.0);
        });
        s.rivers.push_back(make_river(a, sa, w, dep, false));
        s.rivers.push_back(make_river(b, sb, w, dep, false));
        s.rivers.push_back(make_river(cc, scc, K("main_width_m"), dep * 1.2, false));
    } else if (kind == "rim") {
        const double off = K("edge_offset") * Lm, slope = K("plateau_slope");
        auto edge_v = [&](double x) { return -off + K("wave_m") * fbm(mix64(sd ^ 0xa01), x, 0.0, 300.0, 3); };
        auto ground = [&](V2 p) { return base + slope * (sg * p.y - edge_v(p.x)) + undul(p, 1.5); };
        fill(s, ground, [&](V2 p) { return sg * p.y < edge_v(p.x); });
        const double xs = 0.2 * Lm;
        s.rivers.push_back(terrain_stream(s, wavy({xs, sg * H}, {xs + 40.0, sg * (edge_v(xs + 40.0) - 30.0)}, 30.0, 400.0, mix64(sd ^ 0xa02), 6.0),
                                          K("stream_width_m"), K("stream_depth_m"), false, ground));
    } else {
        throw std::invalid_argument("unknown synth terrain: " + kind);
    }
    auto_farmland(s, c, sd);
    finish_site(s, c);
    return s;
}

}  // namespace skyisle::town
