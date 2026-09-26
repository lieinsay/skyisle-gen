// 5.1 群内布局（layout.py 同式）。
#include "skyisle/island/layout.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <tuple>

#include "skyisle/island/territory.hpp"

namespace skyisle::island {

const char* age_name(AgeKind k) { return k == YOUNG ? "young" : (k == MID ? "mid" : "old"); }

int island_count(Rng& rng, double area_km2, double area_median, const Config& c) {
    double n = c.get("layout.n0") * c_pow(area_km2 / std::max(area_median, 1e-9), c.get("layout.n_exp"));
    n *= std::exp(rng.normal(0.0, c.get("layout.n_sigma")));
    const double r = std::nearbyint(n);
    return static_cast<int>(clip(r, c.get("layout.n_min"), c.get("layout.n_max")));
}

std::vector<double> zipf_sizes(double area_km2, double main_km2, int n, const Config& c) {
    const double rest = std::max(0.0, area_km2 - main_km2);
    const double min_islet = c.get("layout.min_islet_km2");
    const double e = c.get("layout.zipf_exp");
    const int n_max = c.geti("layout.n_max");
    if (rest < min_islet || n < 2) return {area_km2};
    while (n > 2) {
        double ee = e;
        std::vector<double> sizes;
        bool found = false;
        while (ee >= 0.0) {
            std::vector<double> w(n - 1), s(n - 1);
            for (int k = 1; k < n; ++k) w[k - 1] = np_pow(static_cast<double>(k), -ee);
            const double sum = np_sum(w.data(), w.size());
            for (int k = 0; k < n - 1; ++k) s[k] = rest * w[k] / sum;
            if (s[0] <= 0.95 * main_km2) {
                sizes = std::move(s);
                found = true;
                break;
            }
            ee -= 0.1;
        }
        if (!found) {
            n += 1;   // 平均都比主岛大：只能再多切几座
            if (n > 4 * n_max) break;
            continue;
        }
        if (sizes.back() >= min_islet) {
            std::vector<double> out{main_km2};
            out.insert(out.end(), sizes.begin(), sizes.end());
            return out;
        }
        n -= 1;
    }
    return {main_km2, rest};
}

void boundary_axis(const PlanetView& pv, double lat, double lon, double& axis, double& kernel, int& btype) {
    const double d = 0.75;
    auto gi = [&](double la, double lo) { return grid_interp(pv.plate_K, pv.plate_grid, la, lo); };
    const double deg2rad = PI / 180.0;
    const double kx = (gi(lat, lon + d) - gi(lat, lon - d)) / (2 * d * std::max(std::cos(lat * deg2rad), 0.1));
    const double ky = (gi(std::min(lat + d, 89.9), lon) - gi(std::max(lat - d, -89.9), lon)) / (2 * d);
    const double k0 = gi(lat, lon);
    auto ss = [](const std::vector<double>& a, double v) {
        return static_cast<int64_t>(std::lower_bound(a.begin(), a.end(), v) - a.begin());
    };
    const int64_t nlat = static_cast<int64_t>(pv.plate_lats.size()), nlon = static_cast<int64_t>(pv.plate_lons.size());
    const int64_t ii = std::min<int64_t>(std::max<int64_t>(ss(pv.plate_lats, lat), 0), nlat - 1);
    const int64_t jj = std::min<int64_t>(std::max<int64_t>(ss(pv.plate_lons, pymod(lon + 180.0, 360.0) - 180.0), 0), nlon - 1);
    btype = pv.plate_btype[ii * nlon + jj];
    if (py_hypot(kx, ky) < 1e-6) {
        axis = 0.0;
        kernel = 0.0;
        return;
    }
    axis = std::atan2(kx, -ky);   // 垂直于梯度
    kernel = clip(k0, 0.0, 1.0);
}

void radial_profile(const Mask& mask, double res_km, std::array<double, 2>& center, std::vector<double>& prof, int nbins) {
    const int H = mask.H, W = mask.W;
    int64_t si = 0, sj = 0, cnt = 0;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            if (mask(i, j)) {
                si += i;
                sj += j;
                ++cnt;
            }
    const double cy = static_cast<double>(si) / static_cast<double>(cnt);
    const double cx = static_cast<double>(sj) / static_cast<double>(cnt);
    const Mask er = binary_erode(mask, 1);
    prof.assign(nbins, 0.0);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            if (!mask(i, j) || er(i, j)) continue;
            const double dx = (j - cx) * res_km;
            const double dy = -(i - cy) * res_km;
            const double r = np_hypot(dx, dy);
            const double th = std::atan2(dy, dx);
            const int64_t b = static_cast<int64_t>((th + PI) / (2 * PI) * nbins) % nbins;
            if (r > prof[b]) prof[b] = r;
        }
    // 空桶用邻桶补（整体更新，同 np.roll 的写法）
    for (int it = 0; it < nbins; ++it) {
        bool any = false;
        for (double v : prof)
            if (v <= 0) any = true;
        if (!any) break;
        std::vector<double> nxt = prof;
        for (int b = 0; b < nbins; ++b)
            if (prof[b] <= 0) nxt[b] = std::max(prof[(b - 1 + nbins) % nbins], prof[(b + 1) % nbins]);
        prof.swap(nxt);
    }
    for (double& v : prof) v = std::max(v, res_km);
    center = {(cx - (W - 1) / 2.0) * res_km, -(cy - (H - 1) / 2.0) * res_km};
}

namespace {

double r_at(const std::vector<double>& prof, double theta) {
    const int n = static_cast<int>(prof.size());
    const double f = pymod((theta + PI) / (2 * PI) * n, static_cast<double>(n));
    const double fl = std::floor(f);
    const int i0 = static_cast<int>(((static_cast<int64_t>(fl) % n) + n) % n);
    const double t = f - fl;
    return prof[i0] * (1 - t) + prof[(i0 + 1) % n] * t;
}

double r_at_max(const std::vector<double>& prof, double theta) {
    const int n = static_cast<int>(prof.size());
    const int i0 = static_cast<int>(std::floor(pymod((theta + PI) / (2 * PI) * n, static_cast<double>(n))));
    auto P = [&](int k) { return prof[((k % n) + n) % n]; };
    return std::max(std::max(P(i0 - 1), P(i0)), std::max(P(i0 + 1), P(i0 + 2)));
}

}  // namespace

std::vector<std::array<double, 2>> place_islands(Rng& rng, const std::vector<std::vector<double>>& profiles, const std::vector<double>& sizes,
                                                 double axis, double kernel, int btype, const Config& c, const TerritoryPlace* territory) {
    const int n = static_cast<int>(profiles.size());
    std::vector<std::array<double, 2>> centers(n, {0.0, 0.0});
    if (territory) centers[0] = {territory->main_x, territory->main_y};
    double gap_lo = c.get("layout.gap_min_km");
    const double gap_hi = c.get("layout.gap_max_km");
    if (territory) gap_lo = std::min(gap_lo, territory->gap_min_km);
    double aniso, spread;
    const double arc = c.get("layout.arc_elongation");
    if (btype == CONVERGENT) {
        aniso = 1.0 + (arc - 1.0) * kernel;
        spread = 1.0;
    } else if (btype == DIVERGENT) {
        aniso = 1.0;
        spread = 1.0 + (c.get("layout.divergent_spread") - 1.0) * kernel;
    } else {
        aniso = 1.0 + 0.5 * (arc - 1.0) * kernel;
        spread = 1.0;
    }
    const double ca = std::cos(axis), sa = std::sin(axis);
    std::vector<double> weights(n);
    for (int k = 0; k < n; ++k) weights[k] = std::sqrt(sizes[k]);
    weights[0] *= c.get("layout.main_gravity");
    auto rat = [&](const std::vector<double>& p, double th) { return territory ? r_at_max(p, th) : r_at(p, th); };

    auto gap_ok = [&](int k, double px, double py) {
        double worst = 1e9;
        for (int j = 0; j < k; ++j) {
            const double dx = px - centers[j][0], dy = py - centers[j][1];
            const double dist = py_hypot(dx, dy);
            const double th = std::atan2(dy, dx);
            const double g = dist - rat(profiles[j], th) - rat(profiles[k], th + PI);
            worst = std::min(worst, g);
        }
        return worst;
    };
    const int attempts = c.geti("layout.place_attempts") * (territory ? 2 : 1);
    for (int k = 1; k < n; ++k) {
        const double wsum = np_sum(weights.data(), k);
        std::vector<double> w(k);
        for (int j = 0; j < k; ++j) w[j] = weights[j] / wsum;
        double bx = 0, by = 0, best_gap = -1e9;
        bool have = false;
        for (int attempt = 0; attempt < attempts; ++attempt) {
            const int anchor = static_cast<int>(rng.choice_p(w));
            const double phi = rng.uniform(-PI, PI);
            double vx = std::cos(phi), vy = std::sin(phi) / aniso;
            const double nv = py_hypot(vx, vy);
            vx = vx / nv;
            vy = vy / nv;
            const double dx = vx * ca - vy * sa, dy = vx * sa + vy * ca;
            const double th = std::atan2(dy, dx);
            double gap = (gap_lo + (gap_hi - gap_lo) * rng.beta(1.3, 2.2)) * spread;
            gap = spread <= 1.0 ? std::min(gap, gap_hi) : std::min(gap, gap_hi * spread);
            if (!territory) gap *= 1.0 + 0.15 * static_cast<double>(attempt / 40);
            else gap *= std::max(0.3, 1.0 - 0.12 * static_cast<double>(attempt / 40));
            const double dist = rat(profiles[anchor], th) + gap + rat(profiles[k], th + PI);
            const double px = centers[anchor][0] + dx * dist, py = centers[anchor][1] + dy * dist;
            double g = gap_ok(k, px, py);
            if (territory) {
                const double score = std::min(g - gap_lo, -violation(px, py, (*territory->support)[k], *territory->lim));
                g = g >= gap_lo ? score + gap_lo : g - 1e6;
            }
            if (g >= gap_lo) {
                bx = px;
                by = py;
                best_gap = g;
                have = true;
                break;
            }
            if (g > best_gap) {
                bx = px;
                by = py;
                best_gap = g;
                have = true;
            }
        }
        centers[k] = {bx, by};
    }
    return centers;
}

std::vector<double> surface_heights(Rng& rng, int n, double height_m, bool layered, const Config& c) {
    std::vector<double> surf(n);
    surf[0] = height_m;
    if (n > 1) {
        const double lo = c.get("layout.surface_lo"), hi = c.get("layout.surface_hi");
        for (int k = 1; k < n; ++k) surf[k] = height_m * rng.uniform(lo, hi);
        if (layered) {
            std::vector<double> sgn(n - 1);
            for (int k = 0; k < n - 1; ++k) sgn[k] = rng.integers(0, 2) == 0 ? -1.0 : 1.0;
            const double spread = c.get("layout.layered_spread_m");
            for (int k = 1; k < n; ++k) surf[k] += sgn[k - 1] * spread * rng.uniform(0.5, 1.0);
        }
    }
    for (double& s : surf) s = std::max(s, 50.0);
    return surf;
}

std::vector<double> relief_targets(Rng& rng, const std::vector<double>& sizes, const std::vector<double>& ages, const Config& c) {
    const size_t n = sizes.size();
    const double ly = std::log(c.get("terrain.relief_young_m")), lo = std::log(c.get("terrain.relief_old_m"));
    const double aexp = c.get("terrain.relief_area_exp"), sigma = c.get("terrain.relief_sigma");
    std::vector<double> R(n);
    for (size_t k = 0; k < n; ++k) {
        const double a = clip(ages[k], 0.0, 1.0);
        const double ref = std::exp(ly + a * (lo - ly));
        R[k] = ref * np_pow(sizes[k] / 1000.0, aexp);
    }
    for (size_t k = 0; k < n; ++k) R[k] *= std::exp(rng.normal(0.0, sigma));
    for (double& r : R) r = clip(r, c.get("terrain.relief_min_m"), c.get("terrain.relief_max_m"));
    return R;
}

std::map<std::pair<int, int>, double> shoreline_gaps(const std::vector<MaskPos>& masks_pos, double res_km,
                                                     const std::vector<std::array<double, 2>>& centers_cell,
                                                     const std::vector<double>& radii_km, double max_gap_km) {
    std::vector<std::vector<std::array<double, 2>>> edges;
    for (const MaskPos& mp : masks_pos) {
        const Mask er = binary_erode(mp.mask, 1);
        std::vector<std::array<double, 2>> pts;
        for (int i = 0; i < mp.mask.H; ++i)
            for (int j = 0; j < mp.mask.W; ++j)
                if (mp.mask(i, j) && !er(i, j))
                    pts.push_back({static_cast<double>(j + mp.c0) * res_km, -static_cast<double>(i + mp.r0) * res_km});
        if (pts.size() > 1500) {
            const size_t step = static_cast<size_t>(std::ceil(static_cast<double>(pts.size()) / 1500.0));
            std::vector<std::array<double, 2>> sub;
            for (size_t k = 0; k < pts.size(); k += step) sub.push_back(pts[k]);
            pts.swap(sub);
        }
        edges.push_back(std::move(pts));
    }
    const int n = static_cast<int>(masks_pos.size());
    std::map<std::pair<int, int>, double> out;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double d = np_hypot(centers_cell[i][0] - centers_cell[j][0], centers_cell[i][1] - centers_cell[j][1]) * res_km;
            if (d - radii_km[i] - radii_km[j] > max_gap_km) continue;
            const auto& a = edges[i];
            const auto& b = edges[j];
            if (a.empty() || b.empty()) continue;   // 贴进群栅格时被别的岛全压掉了（粗分辨率下的小礁）
            double best2 = INF;
            for (const auto& p : a)
                for (const auto& q : b) {
                    const double dx = p[0] - q[0], dy = p[1] - q[1];
                    const double s = dx * dx + dy * dy;
                    if (s < best2) best2 = s;
                }
            const double best = std::sqrt(best2);
            if (best <= max_gap_km) out[{i, j}] = best;
        }
    return out;
}

namespace {
int n_components(int n, const std::vector<Link>& out, std::vector<int>& comp) {
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (const Link& e : out) {
        const int ra = find(e.a), rb = find(e.b);
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
    }
    comp.assign(n, 0);
    std::set<int> roots;
    for (int i = 0; i < n; ++i) {
        comp[i] = find(i);
        roots.insert(comp[i]);
    }
    return static_cast<int>(roots.size());
}
}  // namespace

void links(const std::map<std::pair<int, int>, double>& gaps, const std::vector<double>& rims, int n, const Config& c,
           std::vector<Link>& out, std::vector<std::pair<int, int>>& tree) {
    const double bmax = c.get("layout.bridge_max_km"), dhmax = c.get("layout.bridge_max_dh_m");
    out.clear();
    tree.clear();
    for (const auto& kv : gaps) {
        const int i = kv.first.first, j = kv.first.second;
        const double g = kv.second;
        Link e;
        e.a = i;
        e.b = j;
        e.gap = g;
        e.dh = std::fabs(rims[i] - rims[j]);
        e.bridge = g <= bmax && std::fabs(rims[i] - rims[j]) <= dhmax;
        out.push_back(e);
    }
    std::vector<int> comp;
    int nc = n_components(n, out, comp);
    while (nc > 1 && !gaps.empty()) {
        bool have = false;
        int bi = 0, bj = 0;
        double bg = 0;
        for (const auto& kv : gaps) {
            const int i = kv.first.first, j = kv.first.second;
            if (comp[i] != comp[j] && (!have || kv.second < bg)) {
                have = true;
                bi = i;
                bj = j;
                bg = kv.second;
            }
        }
        if (!have) break;
        Link e;
        e.a = bi;
        e.b = bj;
        e.gap = bg;
        e.dh = std::fabs(rims[bi] - rims[bj]);
        e.bridge = false;
        e.fallback = true;
        out.push_back(e);
        nc = n_components(n, out, comp);
    }
    // 导水槽：Prim 从主岛出发，只走索桥（堆里比的是 island.json 里四舍五入到 3 位的岸距，同 Python 版）
    std::vector<std::vector<std::pair<double, int>>> adj(n);
    for (const Link& e : out)
        if (e.bridge) {
            const double g3 = pyround(e.gap, 3);
            adj[e.a].push_back({g3, e.b});
            adj[e.b].push_back({g3, e.a});
        }
    using E = std::tuple<double, int, int>;
    std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
    std::vector<uint8_t> seen(n, 0);
    seen[0] = 1;
    for (const auto& gj : adj[0]) heap.emplace(gj.first, 0, gj.second);
    while (!heap.empty()) {
        const auto [g, i, j] = heap.top();
        heap.pop();
        if (seen[j]) continue;
        seen[j] = 1;
        tree.emplace_back(i, j);
        for (const auto& gk : adj[j])
            if (!seen[gk.second]) heap.emplace(gk.first, j, gk.second);
    }
}

}  // namespace skyisle::island
