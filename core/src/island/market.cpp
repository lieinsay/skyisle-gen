// 泊场、中转站、镇、航船、邑治（P7；skyisle_gen/island/market.py 的 harbor_pad / harbor_count / harbor_sites / landing_ships /
// build_relays / build_towns 逐位同式）。运算次序、排序的稳定性、round 的位数照抄 Python 版。
#include "skyisle/island/market.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace skyisle::island {

namespace {

constexpr int LC_WET = 9;
const char* const FUNC_ORDER[6] = {"beacon", "gate", "inn", "wait", "shelter", "transship"};
enum { F_BEACON = 1, F_GATE = 2, F_INN = 4, F_WAIT = 8, F_SHELTER = 16, F_TRANSSHIP = 32 };
const char* const ROLE_ORDER[5] = {"gate", "post", "inn", "repair", "porter"};

inline double deg2rad(double x) { return x * (PI / 180.0); }   // math.radians
inline double rad2deg(double x) { return x * (180.0 / PI); }   // math.degrees
inline double dist(double ax, double ay, double bx, double by) {
    const double dx = bx - ax, dy = by - ay;
    return std::sqrt(dx * dx + dy * dy);
}
// market._angdiff：abs(a − b) % 2π，大于 π 取补
inline double angdiff(double a, double b) {
    const double d = std::fmod(std::fabs(a - b), 2.0 * PI);
    return d > PI ? 2.0 * PI - d : d;
}
double wind_factor(double brg, double wdir, double steady, double tail, double head) {
    const double c = std::cos(wdir - brg) * steady;
    if (c >= 0.0) return 1.0 / (1.0 + (1.0 / tail - 1.0) * c);
    return 1.0 + (head - 1.0) * (-c);
}

}  // namespace

void harbor_pad(const Group& g, const Config& c, Mask& lflat, Mask& pad) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const float smax = static_cast<float>(c.get("market.harbor_slope_max_deg"));   // slope_deg 是 float32：阈值也降成 float32
    lflat = Mask(H, W, 0);
    Mask flat(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        const bool lf = g.island_id.v[k] >= 0 && !g.cliff.v[k] && !(g.river.v[k] > 0) && !g.lake.v[k] && static_cast<float>(g.slope.v[k]) <= smax;
        lflat.v[k] = lf ? 1 : 0;
        const bool fp = !g.floodplain.v.empty() && g.floodplain.v[k];
        flat.v[k] = (lf && !(g.stream.v[k] > 0) && g.landcover.v[k] != LC_WET && !fp && g.cultivated.v[k] == 0 && g.fallow_years.v[k] == 0) ? 1 : 0;
    }
    const int kk = static_cast<int>(c.get("market.harbor_open_cells"));
    if (kk > 0) {
        const Mask o = binary_dilate(binary_erode(flat, kk), kk);
        pad = Mask(H, W, 0);
        for (size_t k = 0; k < N; ++k) pad.v[k] = (o.v[k] && flat.v[k]) ? 1 : 0;
    } else {
        pad = flat;
    }
}

std::vector<std::vector<int32_t>> island_cells(const Group& g) {
    std::vector<std::vector<int32_t>> cells(g.islands.size());
    for (size_t k = 0; k < g.island_id.v.size(); ++k) {
        const int i = g.island_id.v[k];
        if (i >= 0 && i < static_cast<int>(cells.size())) cells[i].push_back(static_cast<int32_t>(k));
    }
    return cells;
}

GridI harbor_count(const Mask& pad, const Group& g, const std::vector<std::vector<int32_t>>& cells, int r) {
    const int H = g.H, W = g.W;
    GridI cnt(H, W, 0);
    for (size_t k = 0; k < cells.size(); ++k) {
        const auto& cl = cells[k];
        if (cl.empty()) continue;
        int r0 = H, r1 = 0, c0 = W, c1 = 0;
        for (int32_t q : cl) {
            r0 = std::min(r0, q / W);
            r1 = std::max(r1, q / W + 1);
            c0 = std::min(c0, q % W);
            c1 = std::max(c1, q % W + 1);
        }
        const int h = r1 - r0, w = c1 - c0;
        std::vector<int64_t> S(static_cast<size_t>(h + 1) * (w + 1), 0);
        for (int a = 0; a < h; ++a) {
            int64_t row = 0;
            for (int b = 0; b < w; ++b) {
                const size_t q = static_cast<size_t>(a + r0) * W + (b + c0);
                row += (pad.v[q] && g.island_id.v[q] == static_cast<int>(k)) ? 1 : 0;
                S[static_cast<size_t>(a + 1) * (w + 1) + (b + 1)] = S[static_cast<size_t>(a) * (w + 1) + (b + 1)] + row;
            }
        }
        auto at = [&](int a, int b) { return S[static_cast<size_t>(a) * (w + 1) + b]; };
        for (int32_t q : cl) {
            const int i = q / W, j = q % W;
            const int a0 = std::clamp(i - r - r0, 0, h), a1 = std::clamp(i + r + 1 - r0, 0, h);
            const int b0 = std::clamp(j - r - c0, 0, w), b1 = std::clamp(j + r + 1 - c0, 0, w);
            cnt.v[q] = static_cast<int32_t>(at(a1, b1) - at(a0, b1) - at(a1, b0) + at(a0, b0));
        }
    }
    return cnt;
}

int64_t harbor_ships(int64_t n_cells, double cell_km2, const Config& c) {
    return static_cast<int64_t>(std::floor(static_cast<double>(n_cells) * cell_km2 * 1e6 * c.get("market.harbor_use_frac") / c.get("market.ship_m2")));
}

std::vector<Harbor> harbor_sites(const Group& g, const Config& c, const Mask& pad, const GridI& cnt) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double res_km = g.res_km, cell_km2 = res_km * res_km;
    const double thr = c.get("market.harbor_min_km2") / cell_km2;
    std::vector<int32_t> order;
    for (size_t q = 0; q < N; ++q)
        if (pad.v[q] && static_cast<double>(cnt.v[q]) >= thr) order.push_back(static_cast<int32_t>(q));
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return cnt.v[a] > cnt.v[b]; });
    const double sep = c.get("market.harbor_sep_km") / res_km;
    const int R = static_cast<int>(std::ceil(sep));
    const double sep2 = sep * sep;
    std::vector<std::array<int, 2>> disc;
    for (int di = -R; di <= R; ++di)
        for (int dj = -R; dj <= R; ++dj)
            if (static_cast<double>(di * di + dj * dj) < sep2) disc.push_back({di, dj});
    Mask blocked(H, W, 0);
    std::vector<Harbor> out;
    for (int32_t q : order) {
        const int i = q / W, j = q % W;
        if (blocked.v[q]) continue;
        for (const auto& d : disc) {
            const int a = i + d[0], b = j + d[1];
            if (a >= 0 && a < H && b >= 0 && b < W) blocked(a, b) = 1;
        }
        const int64_t n = cnt.v[q];
        Harbor h;
        h.id = static_cast<int>(out.size()) + 1;
        h.island = g.island_id.v[q];
        h.ci = i;
        h.cj = j;
        h.kx = pyround(g.origin_x + (j + 0.5) * res_km, 3);
        h.ky = pyround(g.origin_y - (i + 0.5) * res_km, 3);
        h.area_km2 = pyround(static_cast<double>(n) * cell_km2, 3);
        h.ships = harbor_ships(n, cell_km2, c);
        h.elev_m = pyround(g.height.v[q], 0);
        out.push_back(h);
    }
    return out;
}

int64_t landing_ships(const Mask& lflat, const Group& g, int i, int j, const Config& c) {
    const int H = g.H, W = g.W;
    const int k = g.island_id(i, j);
    int64_t n = 0;
    for (int a = std::max(0, i - 1); a < std::min(H, i + 2); ++a)
        for (int b = std::max(0, j - 1); b < std::min(W, j + 2); ++b)
            if (lflat(a, b) && g.island_id(a, b) == k) ++n;
    const int64_t s = harbor_ships(n, g.res_km * g.res_km, c);
    return std::max<int64_t>(1, std::min(static_cast<int64_t>(c.get("market.landing_ships_max")), s));
}

// ---------------------------------------------------------------- 中转站
std::vector<Relay> build_relays(const Group& g, const Config& c, const std::vector<uint8_t>& farm, const std::vector<uint8_t>& busy,
                                const Mask& lflat, const Mask& free, const GridI& cnt, const std::vector<double>& expo,
                                const std::vector<std::vector<int32_t>>& cells, int64_t nonfarm_left) {
    auto mc = [&](const char* k) { return c.get(std::string("market.") + k); };
    const int W = g.W;
    const int n = static_cast<int>(g.islands.size());
    const double res_km = g.res_km;
    const double x0 = g.origin_x, y0 = g.origin_y;
    const double cx0 = g.islands[0].cx_j, cy0 = g.islands[0].cy_j;
    const double P_mm = pyround(g.P_mm, 0);
    std::vector<uint8_t> wetisl(n, 0), has_flat(n, 0);
    for (size_t k = 0; k < g.island_id.v.size(); ++k) {
        const int i = g.island_id.v[k];
        if (i >= 0 && (g.river.v[k] > 0 || g.lake.v[k] || g.stream.v[k] > 0)) wetisl[i] = 1;
    }
    const bool rain_ok = P_mm >= c.get("settle.settle_rain_mm");
    for (int k = 0; k < n; ++k)
        for (int32_t q : cells[k])
            if (free.v[q]) {
                has_flat[k] = 1;
                break;
            }
    const double relief_ref = mc("relay_relief_ref_m");
    auto island_score = [&](int k, double ux, double uy) {
        const double dx = g.islands[k].cx_j - cx0, dy = g.islands[k].cy_j - cy0;
        const double rel = g.islands[k].peak_j - g.islands[k].rim_j;
        return (dx * ux + dy * uy) * (1.0 + std::min(1.0, std::max(0.0, rel) / relief_ref));
    };
    auto lookout = [&](int k) {
        int32_t bq = -1;
        double bh = 0;
        for (int32_t q : cells[k])
            if (bq < 0 || g.height.v[q] > bh) {
                bh = g.height.v[q];
                bq = q;
            }
        return bq;
    };
    const double lee_km = mc("relay_lee_km");
    auto station = [&](int k, double ux, double uy) {
        int32_t best = -1;
        double bs = 0;
        for (int32_t q : cells[k]) {
            if (!free.v[q]) continue;
            const int i = q / W, j = q % W;
            const double x = x0 + (j + 0.5) * res_km, y = y0 - (i + 0.5) * res_km;
            const double s = (x - cx0) * ux + (y - cy0) * uy + lee_km * (1.0 - expo[q]) / 2.0;
            if (best < 0 || s > bs) {
                best = q;
                bs = s;
            }
        }
        return best;
    };
    struct R0 {
        int island;
        int funcs;
        std::vector<RouteEdge> edges;
        double ux, uy, flow = 0;
    };
    std::vector<R0> relays;
    std::map<int, int> by_island;
    // ---- 群间：⑥ 的邻边按方位合成口子
    const double fmin = mc("relay_flow_min");
    std::vector<RouteEdge> gw;
    for (const RouteEdge& e : g.inp.routes)
        if (e.flow_in + e.flow_out >= fmin) gw.push_back(e);
    std::stable_sort(gw.begin(), gw.end(), [](const RouteEdge& a, const RouteEdge& b) {
        const double fa = a.flow_in + a.flow_out, fb = b.flow_in + b.flow_out;
        if (fa != fb) return -fa < -fb;
        return a.node < b.node;
    });
    std::vector<std::vector<RouteEdge>> gates;   // [0] 是主边
    const double sep = deg2rad(mc("relay_gate_sep_deg"));
    const size_t max_gates = static_cast<size_t>(mc("relay_max_gates"));
    for (const RouteEdge& e : gw) {
        int best = -1;
        double bd = 0;
        for (size_t t = 0; t < gates.size(); ++t) {
            const double d = angdiff(e.bearing, gates[t][0].bearing);
            if (best < 0 || d < bd) {
                best = static_cast<int>(t);
                bd = d;
            }
        }
        if (best >= 0 && (bd < sep || gates.size() >= max_gates)) gates[best].push_back(e);
        else gates.push_back({e});
    }
    const double sector = std::cos(deg2rad(mc("relay_sector_deg")));
    const int hub_gate = (g.inp.hub && !gates.empty()) ? 0 : -1;
    for (size_t t = 0; t < gates.size(); ++t) {
        const double th = gates[t][0].bearing;
        const double ux = std::sin(th), uy = std::cos(th);
        int best = -1;
        double bs = 0;
        for (int k = 1; k < n; ++k) {
            if (!has_flat[k] || !(rain_ok || wetisl[k])) continue;
            const double dx = g.islands[k].cx_j - cx0, dy = g.islands[k].cy_j - cy0;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d <= 0.0 || (dx * ux + dy * uy) / d < sector) continue;
            const double s = island_score(k, ux, uy);
            if (best < 0 || s > bs) {
                best = k;
                bs = s;
            }
        }
        if (best < 0) {
            if (!has_flat[0]) continue;
            best = 0;
        }
        int funcs = F_GATE;
        for (const RouteEdge& e : gates[t]) {
            if (e.flow_in >= fmin && e.cost_in >= mc("relay_overnight_days")) funcs |= F_INN;
            if (e.days > 0.0 && e.cost_out >= mc("relay_headwind_ratio") * e.days) funcs |= F_WAIT;
        }
        if (g.inp.storm >= mc("relay_storm_min")) funcs |= F_SHELTER;
        if (static_cast<int>(t) == hub_gate) funcs |= F_TRANSSHIP;
        auto it = by_island.find(best);
        if (it != by_island.end()) {
            R0& r = relays[it->second];
            r.funcs |= funcs;
            r.edges.insert(r.edges.end(), gates[t].begin(), gates[t].end());
            continue;
        }
        by_island[best] = static_cast<int>(relays.size());
        relays.push_back({best, funcs, gates[t], ux, uy});
    }
    // ---- 群内：群边高处的瞭望烽火（统一 P5 的烽火台）
    struct Far {
        double s;
        int k;
        double a;
    };
    std::vector<Far> far;
    for (int k = 1; k < n; ++k) {
        if (!has_flat[k] || (busy[k] && !by_island.count(k)) || farm[k]) continue;
        const double dx = g.islands[k].cx_j - cx0, dy = g.islands[k].cy_j - cy0;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d <= 0.0) continue;
        far.push_back({-island_score(k, dx / d, dy / d), k, std::atan2(dy, dx)});
    }
    std::stable_sort(far.begin(), far.end(), [](const Far& a, const Far& b) {
        if (a.s != b.s) return a.s < b.s;
        return a.k < b.k;
    });
    const double bsep = deg2rad(mc("beacon_sep_deg"));
    const size_t bmax = static_cast<size_t>(mc("beacon_max"));
    std::vector<double> angs;
    for (const Far& f : far) {
        if (angs.size() >= bmax) break;
        bool bad = false;
        for (double b : angs)
            if (angdiff(f.a, b) < bsep) {
                bad = true;
                break;
            }
        if (bad) continue;
        angs.push_back(f.a);
        auto it = by_island.find(f.k);
        if (it != by_island.end()) {
            relays[it->second].funcs |= F_BEACON;
            continue;
        }
        const double dx = g.islands[f.k].cx_j - cx0, dy = g.islands[f.k].cy_j - cy0;
        const double d = std::sqrt(dx * dx + dy * dy);
        by_island[f.k] = static_cast<int>(relays.size());
        relays.push_back({f.k, F_BEACON, {}, dx / d, dy / d});
    }
    // ---- 站址、户
    const double fref = mc("relay_flow_ref");
    std::vector<std::array<double, 5>> raw;
    for (R0& r : relays) {
        double flow = 0.0;
        for (const RouteEdge& e : r.edges) flow += e.flow_in + e.flow_out;
        const double s = r.edges.empty() ? 1.0 : std::min(2.0, std::max(0.5, std::sqrt(flow / fref)));
        const int f = r.funcs;
        std::array<double, 5> roles{};
        roles[0] = (f & F_GATE) ? mc("relay_hh_gate") : 0.0;
        roles[1] = (f & F_INN) ? mc("relay_hh_post") : 0.0;
        roles[2] = ((f & F_INN) ? mc("relay_hh_inn") * s : 0.0) + ((f & F_WAIT) ? mc("relay_hh_wait") * s : 0.0);
        roles[3] = ((f & F_SHELTER) ? mc("relay_hh_shelter") : 0.0) + ((f & F_TRANSSHIP) ? mc("relay_hh_repair") * s : 0.0);
        roles[4] = (f & F_TRANSSHIP) ? mc("relay_hh_porter") * s : 0.0;
        r.flow = flow;
        raw.push_back(roles);
    }
    double tot = 0.0;
    for (const auto& roles : raw)
        for (double x : roles) tot += x;
    const double cap = mc("relay_cap_frac") * static_cast<double>(nonfarm_left);
    const double fac = tot > 0.0 ? std::min(1.0, cap / std::max(1e-9, tot)) : 0.0;
    std::vector<Relay> out;
    for (size_t q = 0; q < relays.size(); ++q) {
        const R0& r = relays[q];
        const int k = r.island;
        const int32_t st = station(k, r.ux, r.uy);
        const int32_t lk = lookout(k);
        Relay x;
        x.id = static_cast<int>(out.size()) + 1;
        x.island = k;
        x.ci = st / W;
        x.cj = st % W;
        x.kx = pyround(x0 + (x.cj + 0.5) * res_km, 3);
        x.ky = pyround(y0 - (x.ci + 0.5) * res_km, 3);
        x.elev_m = pyround(g.height.v[st], 0);
        x.li = lk / W;
        x.lj = lk % W;
        x.lkx = pyround(x0 + (x.lj + 0.5) * res_km, 3);
        x.lky = pyround(y0 - (x.li + 0.5) * res_km, 3);
        x.lookout_m = pyround(g.height.v[lk] - g.islands[k].rim_j, 0);
        for (int b = 0; b < 6; ++b)
            if (r.funcs & (1 << b)) x.funcs.push_back(FUNC_ORDER[b]);
        int64_t hh = 0;
        for (int b = 0; b < 5; ++b) {
            if (!(raw[q][b] > 0.0)) continue;
            const int64_t v = static_cast<int64_t>(std::floor(raw[q][b] * fac));
            hh += v;
            if (v > 0) x.roles.push_back({ROLE_ORDER[b], v});
        }
        x.households = hh;
        x.ships = std::max(landing_ships(lflat, g, x.ci, x.cj, c), harbor_ships(cnt.v[st], res_km * res_km, c));
        x.flow = pyround(r.flow, 1);
        x.edges = r.edges;
        out.push_back(x);
    }
    return out;
}

// ---------------------------------------------------------------- 镇、航船、邑治
TownsResult build_towns(const std::vector<MarketVillage>& villages, std::vector<Harbor>& harbors, const Config& c, int64_t rest_hh,
                        double wind_u, double wind_v) {
    auto mc = [&](const char* k) { return c.get(std::string("market.") + k); };
    TownsResult out;
    const int V = static_cast<int>(villages.size());
    if (V == 0) return out;
    const double wdir = std::atan2(wind_u, wind_v);
    const double steady = std::min(1.0, std::sqrt(wind_u * wind_u + wind_v * wind_v) / mc("wind_ref_ms"));
    const double tail = mc("tailwind_factor"), head = mc("headwind_factor");
    const double walk = mc("walk_km"), reach = mc("boat_reach_km");
    std::vector<double> X(V), Y(V);
    std::vector<int> isl(V);
    std::vector<int64_t> hh(V);
    for (int a = 0; a < V; ++a) {
        X[a] = villages[a].kx;
        Y[a] = villages[a].ky;
        isl[a] = villages[a].island;
        hh[a] = villages[a].households;
    }
    const double th_km = mc("town_harbor_km");
    std::vector<int> hb_of(V, -1);
    for (int cc = 0; cc < V; ++cc) {
        int best = -1;
        int64_t bships = 0;
        double bd = 0;
        int bid = 0;
        for (size_t t = 0; t < harbors.size(); ++t) {
            const Harbor& h = harbors[t];
            if (h.island != isl[cc]) continue;
            const double d = dist(X[cc], Y[cc], h.kx, h.ky);
            if (d > th_km) continue;
            // key = (−ships, d, id)
            bool better = best < 0;
            if (!better) {
                if (-h.ships != -bships) better = -h.ships < -bships;
                else if (d != bd) better = d < bd;
                else better = h.id < bid;
            }
            if (better) {
                best = static_cast<int>(t);
                bships = h.ships;
                bd = d;
                bid = h.id;
            }
        }
        hb_of[cc] = best;
    }
    std::vector<double> TX(V), TY(V), bonus(V, 0.0);
    for (int cc = 0; cc < V; ++cc) {
        TX[cc] = hb_of[cc] >= 0 ? harbors[hb_of[cc]].kx : X[cc];
        TY[cc] = hb_of[cc] >= 0 ? harbors[hb_of[cc]].ky : Y[cc];
        if (hb_of[cc] >= 0)
            bonus[cc] = mc("harbor_bonus") * std::min(1.0, static_cast<double>(harbors[hb_of[cc]].ships) / mc("harbor_ref_ships"));
    }
    auto boat_cost = [&](double ax, double ay, double bx, double by, double& d, double& k) {
        const double dx = bx - ax, dy = by - ay;
        d = std::sqrt(dx * dx + dy * dy);
        if (d <= 0.0) {
            d = 0.0;
            k = 0.0;
            return;
        }
        k = d * wind_factor(std::atan2(dx, dy), wdir, steady, tail, head);
    };
    const size_t VV = static_cast<size_t>(V) * V;
    std::vector<double> Wd(VV, INF), C(VV, 0.0), D(VV, 0.0);
    for (int a = 0; a < V; ++a)
        for (int cc = 0; cc < V; ++cc) {
            const size_t q = static_cast<size_t>(a) * V + cc;
            const double dd = dist(X[a], Y[a], X[cc], Y[cc]);
            D[q] = dd;
            if (isl[a] == isl[cc]) Wd[q] = dd;
            double d0, k0;
            boat_cost(X[a], Y[a], TX[cc], TY[cc], d0, k0);
            C[q] = k0;
        }
    std::vector<uint8_t> walk_ok(VV), boat_ok(VV);
    for (size_t q = 0; q < VV; ++q) {
        walk_ok[q] = Wd[q] <= walk ? 1 : 0;
        boat_ok[q] = (!walk_ok[q] && C[q] <= reach) ? 1 : 0;
    }
    std::vector<uint8_t> walk_srv(V, 0), boat_srv(V, 0), blocked(V, 0);
    std::vector<int> chosen;
    std::vector<double> scores;
    const double spacing = mc("town_spacing_km"), min_srv = mc("town_min_served_hh"), bw = mc("boat_weight");
    auto all_blocked = [&]() {
        for (uint8_t b : blocked)
            if (!b) return false;
        return true;
    };
    while (!all_blocked()) {
        std::vector<int64_t> sw(V, 0), sb(V, 0);
        for (int a = 0; a < V; ++a) {
            const int64_t hw = walk_srv[a] ? 0 : hh[a];
            const int64_t hb = (walk_srv[a] || boat_srv[a]) ? 0 : hh[a];
            for (int cc = 0; cc < V; ++cc) {
                const size_t q = static_cast<size_t>(a) * V + cc;
                if (walk_ok[q]) sw[cc] += hw;
                if (boat_ok[q]) sb[cc] += hb;
            }
        }
        int best = -1;
        double bs = 0;
        for (int cc = 0; cc < V; ++cc) {
            if (blocked[cc]) continue;
            const double s = (static_cast<double>(sw[cc]) + bw * static_cast<double>(sb[cc])) * (1.0 + bonus[cc]);
            if (best < 0 || s > bs) {
                best = cc;
                bs = s;
            }
        }
        if (!chosen.empty() && bs < min_srv) break;
        chosen.push_back(best);
        scores.push_back(bs);
        for (int cc = 0; cc < V; ++cc)
            if (D[static_cast<size_t>(best) * V + cc] < spacing) blocked[cc] = 1;
        blocked[best] = 1;
        for (int a = 0; a < V; ++a) {
            const size_t q = static_cast<size_t>(a) * V + best;
            if (walk_ok[q]) walk_srv[a] = 1;
            if (boat_ok[q]) boat_srv[a] = 1;
        }
    }
    const int T = static_cast<int>(chosen.size());
    std::vector<int> tw(V, -1), mode(V, 0);
    std::vector<double> mkm(V, 0.0);
    const double walk_max = mc("walk_max_km");
    for (int a = 0; a < V; ++a) {
        int bt = -1;
        double bd = 0;
        for (int t = 0; t < T; ++t) {
            const double w = Wd[static_cast<size_t>(a) * V + chosen[t]];
            if (w <= walk_max && (bt < 0 || w < bd)) {
                bt = t;
                bd = w;
            }
        }
        if (bt >= 0) {
            tw[a] = bt;
            mode[a] = 0;
            mkm[a] = bd;
            continue;
        }
        for (int t = 0; t < T; ++t) {
            const double k = C[static_cast<size_t>(a) * V + chosen[t]];
            if (bt < 0 || k < bd) {
                bt = t;
                bd = k;
            }
        }
        tw[a] = bt;
        mode[a] = 1;
        mkm[a] = bd;
    }
    // 航船：每个镇的航船村按方位扫一圈，分线
    const size_t maxs = static_cast<size_t>(mc("line_max_stops"));
    const double maxk = mc("line_max_km");
    struct Path {
        std::vector<int> order;
        std::vector<std::array<double, 2>> pts;
        double L = 0, K = 0;
    };
    auto path = [&](const std::vector<int>& S, int t) {
        const int cc = chosen[t];
        Path p;
        int st = S[0];
        for (size_t q = 1; q < S.size(); ++q)
            if (C[static_cast<size_t>(S[q]) * V + cc] > C[static_cast<size_t>(st) * V + cc]) st = S[q];
        p.order.push_back(st);
        std::vector<int> rest;
        for (int a : S)
            if (a != st) rest.push_back(a);
        int cur = st;
        while (!rest.empty()) {
            size_t nb = 0;
            double nd = dist(X[cur], Y[cur], X[rest[0]], Y[rest[0]]);
            for (size_t q = 1; q < rest.size(); ++q) {
                const double d = dist(X[cur], Y[cur], X[rest[q]], Y[rest[q]]);
                if (d < nd) {
                    nb = q;
                    nd = d;
                }
            }
            cur = rest[nb];
            p.order.push_back(cur);
            rest.erase(rest.begin() + static_cast<long>(nb));
        }
        for (int a : p.order) p.pts.push_back({X[a], Y[a]});
        p.pts.push_back({TX[cc], TY[cc]});
        for (size_t q = 0; q + 1 < p.pts.size(); ++q) {
            double d, k;
            boat_cost(p.pts[q][0], p.pts[q][1], p.pts[q + 1][0], p.pts[q + 1][1], d, k);
            p.L += d;
            p.K += k;
        }
        return p;
    };
    std::vector<std::vector<Path>> lines_of(T);
    for (int t = 0; t < T; ++t) {
        const int cc = chosen[t];
        std::vector<std::pair<double, int>> ang;
        for (int a = 0; a < V; ++a)
            if (tw[a] == t && mode[a] == 1) ang.push_back({std::atan2(X[a] - TX[cc], Y[a] - TY[cc]), a});
        if (ang.empty()) continue;
        std::sort(ang.begin(), ang.end());
        const size_t m = ang.size();
        size_t start = 0;
        if (m > 1) {
            double gbest = -1.0;
            for (size_t q = 0; q < m; ++q) {
                const double gap = ang[(q + 1) % m].first - ang[q].first + (q == m - 1 ? 2.0 * PI : 0.0);
                if (gap > gbest) {
                    gbest = gap;
                    start = (q + 1) % m;
                }
            }
        }
        std::vector<int> cur;
        for (size_t q = 0; q < m; ++q) {
            const int a = ang[(start + q) % m].second;
            std::vector<int> trial = cur;
            trial.push_back(a);
            if (!cur.empty() && (trial.size() > maxs || path(trial, t).K > maxk)) {
                lines_of[t].push_back(path(cur, t));
                cur = {a};
            } else {
                cur = trial;
            }
        }
        if (!cur.empty()) lines_of[t].push_back(path(cur, t));
    }
    // 邑治：(服务户 + seat_boat_weight × 航船送来的户) × (1 + seat_harbor_weight × min(1, 泊场的船 / harbor_ref_ships))，平局服务户多的、先挑的
    std::vector<int64_t> served(T, 0), boat_hh(T, 0);
    for (int a = 0; a < V; ++a) {
        served[tw[a]] += hh[a];
        if (mode[a] == 1) boat_hh[tw[a]] += hh[a];
    }
    int seat_t = -1;
    double sk_s = 0;
    int64_t sk_srv = 0;
    std::vector<double> seat_sc(T, 0.0);
    for (int t = 0; t < T; ++t) {
        const int cc = chosen[t];
        const double hq = hb_of[cc] >= 0 ? std::min(1.0, static_cast<double>(harbors[hb_of[cc]].ships) / mc("harbor_ref_ships")) : 0.0;
        const double s = (static_cast<double>(served[t]) + mc("seat_boat_weight") * static_cast<double>(boat_hh[t])) * (1.0 + mc("seat_harbor_weight") * hq);
        seat_sc[t] = s;
        // key = (−s, −served, t)
        bool better = seat_t < 0;
        if (!better) {
            if (-s != -sk_s) better = -s < -sk_s;
            else better = -served[t] < -sk_srv;
        }
        if (better) {
            seat_t = t;
            sk_s = s;
            sk_srv = served[t];
        }
    }
    std::vector<int> order_t{seat_t};
    for (int t = 0; t < T; ++t)
        if (t != seat_t) order_t.push_back(t);
    std::vector<int> tid(T);
    for (int q = 0; q < T; ++q) tid[order_t[q]] = q + 1;
    // 市户：非农户余量按服务户数分（最大余数法）
    double tot_srv = 0.0;
    for (int t : order_t) tot_srv += static_cast<double>(served[t]);
    std::vector<double> raw(T);
    std::vector<int64_t> base(T);
    int64_t sb_ = 0;
    for (int q = 0; q < T; ++q) {
        raw[q] = static_cast<double>(rest_hh) * static_cast<double>(served[order_t[q]]) / std::max(1e-9, tot_srv);
        base[q] = static_cast<int64_t>(std::floor(raw[q]));
        sb_ += base[q];
    }
    std::vector<int> ro(T);
    for (int q = 0; q < T; ++q) ro[q] = q;
    std::stable_sort(ro.begin(), ro.end(), [&](int a, int b) { return -(raw[a] - static_cast<double>(base[a])) < -(raw[b] - static_cast<double>(base[b])); });
    for (int64_t q = 0; q < std::max<int64_t>(0, rest_hh - sb_) && q < T; ++q) base[ro[q]] += 1;
    // 记录
    out.market_town.assign(V, 0);
    out.mode.assign(V, 0);
    out.boat_line.assign(V, 0);
    out.town.assign(V, 0);
    out.market_km.assign(V, 0.0);
    out.households_market.assign(V, 0);
    for (int q = 0; q < T; ++q) {
        const int t = order_t[q];
        const int cc = chosen[t];
        const MarketVillage& vc = villages[cc];
        Town tr;
        tr.id = q + 1;
        tr.village = vc.id;
        tr.island = isl[cc];
        tr.ci = vc.ci;
        tr.cj = vc.cj;
        tr.kx = vc.kx;
        tr.ky = vc.ky;
        tr.households_farm = vc.households;
        tr.households_market = base[q];
        out.town[cc] = q + 1;
        out.households_market[cc] = base[q];
        double mx = 0.0;
        for (int a = 0; a < V; ++a) {
            if (tw[a] != t) continue;
            tr.served_villages += 1;
            if (mode[a] == 0) {
                tr.served_walk += hh[a];
                mx = std::max(mx, Wd[static_cast<size_t>(a) * V + cc]);
            } else {
                tr.served_boat += hh[a];
            }
            if (isl[a] != isl[cc]) tr.served_other += hh[a];
        }
        tr.served = served[t];
        tr.max_served_km = pyround(mx, 2);
        for (const Path& p : lines_of[t]) {
            BoatLine ln;
            ln.id = static_cast<int>(out.lines.size()) + 1;
            ln.town = q + 1;
            tr.lines.push_back(ln.id);
            std::set<int> is;
            for (int a : p.order) {
                ln.households += hh[a];
                out.boat_line[a] = ln.id;
                ln.stops.push_back(villages[a].id);
                is.insert(isl[a]);
            }
            ln.islands.assign(is.begin(), is.end());
            for (const auto& pt : p.pts) ln.pts.push_back({pyround(pt[0], 3), pyround(pt[1], 3)});
            ln.length_km = pyround(p.L, 2);
            ln.cost_km = pyround(p.K, 2);
            ln.hours = pyround(p.K / mc("boat_kmh"), 2);
            ln.interval_days = ln.households >= static_cast<int64_t>(mc("line_daily_hh")) ? 1 : static_cast<int>(mc("line_interval_days"));
            out.lines.push_back(ln);
        }
        tr.score = pyround(scores[t], 1);
        tr.seat_score = pyround(seat_sc[t], 1);
        tr.seat = t == seat_t;
        if (hb_of[cc] >= 0) {
            Harbor& h = harbors[hb_of[cc]];
            tr.harbor = h.id;
            tr.harbor_ships = h.ships;
            tr.harbor_dist_km = pyround(dist(X[cc], Y[cc], h.kx, h.ky), 2);
            h.town = q + 1;
            const double dx = h.kx - X[cc], dy = h.ky - Y[cc];
            tr.street_len_km = pyround(std::sqrt(dx * dx + dy * dy), 2);
            tr.street_bearing_deg = pyround(pymod(rad2deg(std::atan2(dx, dy)), 360.0), 1);
        }
        out.towns.push_back(tr);
    }
    for (int a = 0; a < V; ++a) {
        out.market_town[a] = tid[tw[a]];
        out.mode[a] = mode[a];
        out.market_km[a] = pyround(mkm[a], 2);
    }
    out.seat = chosen[seat_t];
    return out;
}

}  // namespace skyisle::island
