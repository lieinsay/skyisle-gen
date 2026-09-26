// 栅格公共件（skyisle_gen/island/grid.py 与 sphere.grid_interp 的 C++ 版）。
#include "skyisle/grid.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace skyisle {

// ---------------------------------------------------------------- 小工具
double np_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

double np_median(std::vector<double> v) {
    if (v.empty()) return NaN;
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const double hi = v[h];
    if (n % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + h);
    return (lo + hi) / 2.0;
}

double pyround(double x, int ndigits) {
    if (!std::isfinite(x)) return x;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*f", ndigits, x);
    return std::strtod(buf, nullptr);
}

double np_pow(double x, double e) {
    if (e == 2.0) return x * x;
    if (e == 0.5) return std::sqrt(x);
    if (e == 1.0) return x;
    if (e == 0.0) return 1.0;
    if (e == -1.0) return 1.0 / x;
    return c_pow(x, e);
}

double c_pow(double x, double e) {
    volatile double ev = e;   // 不让编译器把常数指数的 pow 化成乘法 / 开方
    return std::pow(x, static_cast<double>(ev));
}

double np_hypot(double x, double y) { return std::hypot(x, y); }

double py_hypot(double x, double y) {
    // CPython 3.12 mathmodule.c 的 vector_norm（n = 2）：无损缩放 + 补偿求和 + 一步微分修正
    double v[2] = {std::fabs(x), std::fabs(y)};
    const bool found_nan = std::isnan(v[0]) || std::isnan(v[1]);
    double max = 0.0;
    for (double a : v)
        if (a > max) max = a;
    if (std::isinf(max)) return max;
    if (found_nan) return NaN;
    if (max == 0.0) return max;
    int max_e;
    std::frexp(max, &max_e);
    if (max_e < -1023) {
        const double dmin = 2.2250738585072014e-308;
        return dmin * py_hypot(v[0] / dmin, v[1] / dmin);
    }
    const double scale = std::ldexp(1.0, -max_e);
    double csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    for (double a : v) {
        const double xs = a * scale;
        const double hi = xs * xs;
        const double lo = std::fma(xs, xs, -hi);
        const double s = csum + hi;
        const double sl = (csum - s) + hi;
        csum = s;
        frac1 += lo;
        frac2 += sl;
    }
    double h = std::sqrt(csum - 1.0 + (frac1 + frac2));
    const double phi = -h * h;
    const double plo = std::fma(-h, h, -phi);
    const double s = csum + phi;
    const double sl = (csum - s) + phi;
    csum = s;
    frac1 += plo;
    frac2 += sl;
    const double xx = csum - 1.0 + (frac1 + frac2);
    h += xx / (2.0 * h);
    return h / scale;
}

double pymod(double x, double m) {
    double mod = std::fmod(x, m);
    if (mod != 0.0) {
        if ((m < 0) != (mod < 0)) mod += m;
    } else {
        mod = std::copysign(0.0, m);
    }
    return mod;
}

// ---------------------------------------------------------------- 噪声
LatticeNoise::LatticeNoise(Rng& rng, double x0, double y0, double x1, double y1, double cell_km) {
    cell_ = cell_km;
    x0_ = x0 - 2 * cell_;
    y0_ = y0 - 2 * cell_;
    nx_ = static_cast<int>(std::ceil((x1 - x0_) / cell_)) + 3;
    ny_ = static_cast<int>(std::ceil((y1 - y0_) / cell_)) + 3;
    lat_.resize(static_cast<size_t>(nx_) * ny_);
    rng.uniform_fill(-1.0, 1.0, lat_.data(), lat_.size());
}

double LatticeNoise::sample(double x, double y) const {
    const double fx = (x - x0_) / cell_;
    const double fy = (y - y0_) / cell_;
    int64_t i0 = static_cast<int64_t>(std::floor(fy));
    int64_t j0 = static_cast<int64_t>(std::floor(fx));
    i0 = std::min<int64_t>(std::max<int64_t>(i0, 0), ny_ - 2);
    j0 = std::min<int64_t>(std::max<int64_t>(j0, 0), nx_ - 2);
    const double ty = smoothstep(clip(fy - static_cast<double>(i0), 0.0, 1.0));
    const double tx = smoothstep(clip(fx - static_cast<double>(j0), 0.0, 1.0));
    const double* r0 = &lat_[static_cast<size_t>(i0) * nx_ + j0];
    const double* r1 = r0 + nx_;
    const double v00 = r0[0], v01 = r0[1], v10 = r1[0], v11 = r1[1];
    return v00 * (1 - ty) * (1 - tx) + v01 * (1 - ty) * tx + v10 * ty * (1 - tx) + v11 * ty * tx;
}

FractalNoise::FractalNoise(Rng& rng, double x0, double y0, double x1, double y1, double feature_km, int octaves,
                           double persistence, double lacunarity) {
    double amp = 1.0, cell = feature_km, total = 0.0;
    for (int k = 0; k < std::max(1, octaves); ++k) {
        layers_.emplace_back(amp, LatticeNoise(rng, x0, y0, x1, y1, cell));
        total += amp;
        amp *= persistence;
        cell /= lacunarity;
    }
    total_ = total;
}

double FractalNoise::sample(double x, double y) const {
    double out = 0.0;
    for (const auto& L : layers_) out += L.first * L.second.sample(x, y);
    return out / total_;
}

// ---------------------------------------------------------------- 连通分量
namespace {
inline int32_t uf_find(std::vector<int32_t>& p, int32_t a) {
    while (p[a] != a) {
        p[a] = p[p[a]];
        a = p[a];
    }
    return a;
}
inline void uf_union(std::vector<int32_t>& p, int32_t a, int32_t b) {
    a = uf_find(p, a);
    b = uf_find(p, b);
    if (a == b) return;
    if (a < b) p[b] = a;
    else p[a] = b;
}
}  // namespace

int label_components(const Mask& mask, int connectivity, GridI& labels) {
    const int H = mask.H, W = mask.W;
    labels = GridI(H, W, 0);
    std::vector<int32_t> parent;
    parent.reserve(1024);
    parent.push_back(0);   // 0 = 背景
    for (int i = 0; i < H; ++i) {
        const uint8_t* m = &mask.v[static_cast<size_t>(i) * W];
        int32_t* L = &labels.v[static_cast<size_t>(i) * W];
        const int32_t* U = i > 0 ? &labels.v[static_cast<size_t>(i - 1) * W] : nullptr;
        for (int j = 0; j < W; ++j) {
            if (!m[j]) continue;
            int32_t best = 0;
            auto take = [&](int32_t lab) {
                if (lab <= 0) return;
                if (best == 0) best = lab;
                else if (lab != best) uf_union(parent, best, lab);
            };
            if (j > 0) take(L[j - 1]);
            if (U) {
                take(U[j]);
                if (connectivity == 8) {
                    if (j > 0) take(U[j - 1]);
                    if (j + 1 < W) take(U[j + 1]);
                }
            }
            if (best == 0) {
                best = static_cast<int32_t>(parent.size());
                parent.push_back(best);
            }
            L[j] = best;
        }
    }
    // 根 = 集合里最小的临时号 = 该分量光栅扫描首次出现的格；按根的次序重新编号
    std::vector<int32_t> final_label(parent.size(), 0);
    int n = 0;
    for (size_t k = 1; k < parent.size(); ++k) {
        int32_t r = uf_find(parent, static_cast<int32_t>(k));
        if (r == static_cast<int32_t>(k)) final_label[k] = ++n;
    }
    for (size_t k = 1; k < parent.size(); ++k) final_label[k] = final_label[uf_find(parent, static_cast<int32_t>(k))];
    for (auto& x : labels.v)
        if (x) x = final_label[x];
    return n;
}

Mask largest_component(const Mask& mask, int* count) {
    GridI lab;
    const int n = label_components(mask, 4, lab);
    Mask out(mask.H, mask.W, 0);
    if (n == 0) {
        if (count) *count = 0;
        return out;
    }
    std::vector<int64_t> cnt(n + 1, 0);
    for (int32_t x : lab.v) cnt[x]++;
    int best = 1;
    for (int k = 2; k <= n; ++k)
        if (cnt[k] > cnt[best]) best = k;
    for (size_t k = 0; k < lab.v.size(); ++k) out.v[k] = lab.v[k] == best ? 1 : 0;
    if (count) *count = static_cast<int>(cnt[best]);
    return out;
}

// ---------------------------------------------------------------- 形态学
Mask binary_erode(const Mask& mask, int iterations, int connectivity) {
    Mask m = mask;
    const int H = m.H, W = m.W;
    const int nn = connectivity == 8 ? 8 : 4;
    for (int it = 0; it < iterations; ++it) {
        Mask acc(H, W, 0);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                if (!m(i, j)) continue;
                bool ok = true;
                for (int d = 0; d < nn && ok; ++d) {
                    const int a = i - N8[d][0], b = j - N8[d][1];
                    if (!m.in(a, b) || !m(a, b)) ok = false;
                }
                acc(i, j) = ok ? 1 : 0;
            }
        m = std::move(acc);
    }
    return m;
}

Mask binary_dilate(const Mask& mask, int iterations, int connectivity) {
    Mask m = mask;
    const int H = m.H, W = m.W;
    const int nn = connectivity == 8 ? 8 : 4;
    for (int it = 0; it < iterations; ++it) {
        Mask acc = m;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                if (!m(i, j)) continue;
                for (int d = 0; d < nn; ++d) {
                    const int a = i + N8[d][0], b = j + N8[d][1];
                    if (m.in(a, b)) acc(a, b) = 1;
                }
            }
        m = std::move(acc);
    }
    return m;
}

GridI distance_bands(const Mask& mask, int max_iter) {
    const int H = mask.H, W = mask.W;
    GridI d(H, W, max_iter + 1);
    Mask cur = mask;
    std::vector<int32_t> active;
    auto has_free_nb = [&](int k) {
        const int i = k / W, j = k % W;
        for (int n = 0; n < 8; ++n) {
            const int a = i + N8[n][0], b = j + N8[n][1];
            if (cur.in(a, b) && !cur(a, b)) return true;
        }
        return false;
    };
    for (int k = 0; k < H * W; ++k)
        if (cur.v[k]) {
            d.v[k] = 0;
            if (has_free_nb(k)) active.push_back(k);
        }
    for (int step = 1; step <= max_iter && !active.empty(); ++step) {
        const int nn = (step % 2) ? 4 : 8;
        std::vector<int32_t> fresh;
        for (int k : active) {
            const int i = k / W, j = k % W;
            for (int n = 0; n < nn; ++n) {
                const int a = i + N8[n][0], b = j + N8[n][1];
                if (cur.in(a, b) && !cur(a, b)) {
                    cur(a, b) = 1;
                    d(a, b) = step;
                    fresh.push_back(a * W + b);
                }
            }
        }
        std::vector<int32_t> next;
        next.reserve(active.size() + fresh.size());
        for (int k : active)
            if (has_free_nb(k)) next.push_back(k);
        for (int k : fresh)
            if (has_free_nb(k)) next.push_back(k);
        active.swap(next);
    }
    return d;
}

void nearest_propagate(const Mask& seed, int max_iter, double step_m, const Mask* within, GridD& dist, Grid<int64_t>& src) {
    const int H = seed.H, W = seed.W;
    const size_t N = static_cast<size_t>(H) * W;
    dist = GridD(H, W, INF);
    src = Grid<int64_t>(H, W, -1);
    std::vector<uint8_t> ok(N, 1);
    if (within)
        for (size_t k = 0; k < N; ++k) ok[k] = (within->v[k] || seed.v[k]) ? 1 : 0;
    double steps[8];
    for (int n = 0; n < 8; ++n) steps[n] = step_m * ((N8[n][0] && N8[n][1]) ? SQRT2 : 1.0);
    // 与 numpy 版同：每轮按 N8 的次序做 8 次整体平移比较（同一次平移内用平移前的值），轮间无变化即停。
    // 只有「上次同方向平移之后变过」的格才可能让邻格变好，所以只看最近 8 次平移里变过的格。
    std::vector<std::vector<int32_t>> changes(9);   // changes[q % 9]：第 q 次平移变过的格；q = −1 记种子（放在第 8 槽）
    auto slot = [](long q) { return static_cast<size_t>(q < 0 ? 8 : q % 8); };
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k]) {
            dist.v[k] = 0.0;
            src.v[k] = static_cast<int64_t>(k);
            changes[8].push_back(static_cast<int32_t>(k));
        }
    std::vector<int32_t> stamp(N, -1000);
    struct Upd {
        int32_t t;
        double d;
        int64_t s;
    };
    std::vector<Upd> upd;
    long p = 0;
    for (int it = 0; it < max_iter; ++it) {
        bool changed = false;
        for (int n = 0; n < 8; ++n, ++p) {
            const int di = N8[n][0], dj = N8[n][1];
            upd.clear();
            // 源：第 p−8 … p−1 次平移（及种子，若 p < 8）里变过的格
            for (long q = std::max<long>(-1, p - 8); q <= p - 1; ++q) {
                const auto& lst = changes[slot(q)];
                for (int32_t s : lst) {
                    if (stamp[s] == static_cast<int32_t>(p)) continue;
                    stamp[s] = static_cast<int32_t>(p);
                    const int si = s / W, sj = s % W;
                    const int ti = si + di, tj = sj + dj;
                    if (ti < 0 || tj < 0 || ti >= H || tj >= W) continue;
                    const int32_t t = ti * W + tj;
                    if (!ok[t]) continue;
                    const double cand = dist.v[s] + steps[n];
                    if (cand < dist.v[t] - 1e-9) upd.push_back({t, cand, src.v[s]});
                }
            }
            // 第 p 次平移变过的格写进槽 p % 8（那里原是第 p − 8 次的，刚刚已经用过）
            auto& out = changes[slot(p)];
            out.clear();
            for (const Upd& u : upd) {
                dist.v[u.t] = u.d;
                src.v[u.t] = u.s;
                out.push_back(u.t);
            }
            if (!upd.empty()) changed = true;
            if (p == 7) changes[8].clear();   // 种子只在第一轮当源
        }
        if (!changed) break;
    }
}

// ---------------------------------------------------------------- 重采样与平滑
GridD block_mean(const GridD& a, int f) {
    const int H = a.H, W = a.W;
    const int Hb = (H + f - 1) / f, Wb = (W + f - 1) / f;
    GridD out(Hb, Wb, 0.0);
    std::vector<double> row(f);
    for (int bi = 0; bi < Hb; ++bi)
        for (int bj = 0; bj < Wb; ++bj) {
            double s = 0.0;
            for (int p = 0; p < f; ++p) {
                const int i = std::min(bi * f + p, H - 1);   // np.pad mode="edge"
                for (int q = 0; q < f; ++q) row[q] = a(i, std::min(bj * f + q, W - 1));
                s += np_sum(row.data(), f);
            }
            out(bi, bj) = s / static_cast<double>(f * f);
        }
    return out;
}

Mask block_any(const Mask& m, int f) {
    const int H = m.H, W = m.W;
    const int Hb = (H + f - 1) / f, Wb = (W + f - 1) / f;
    Mask out(Hb, Wb, 0);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            if (m(i, j)) out(i / f, j / f) = 1;
    return out;
}

GridD upsample_bilinear(const GridD& a, int f, int H, int W) {
    const int h = a.H, w = a.W;
    std::vector<int64_t> i0(H), i1(H), j0(W), j1(W);
    std::vector<double> ty(H), tx(W);
    for (int r = 0; r < H; ++r) {
        const double fy = (static_cast<double>(r) + 0.5) / f - 0.5;
        int64_t k = static_cast<int64_t>(std::floor(fy));
        k = std::min<int64_t>(std::max<int64_t>(k, 0), h - 1);
        i0[r] = k;
        i1[r] = std::min<int64_t>(std::max<int64_t>(k + 1, 0), h - 1);
        ty[r] = clip(fy - static_cast<double>(k), 0.0, 1.0);
    }
    for (int c = 0; c < W; ++c) {
        const double fx = (static_cast<double>(c) + 0.5) / f - 0.5;
        int64_t k = static_cast<int64_t>(std::floor(fx));
        k = std::min<int64_t>(std::max<int64_t>(k, 0), w - 1);
        j0[c] = k;
        j1[c] = std::min<int64_t>(std::max<int64_t>(k + 1, 0), w - 1);
        tx[c] = clip(fx - static_cast<double>(k), 0.0, 1.0);
    }
    GridD out(H, W, 0.0);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            const double y = ty[r], x = tx[c];
            out(r, c) = a(i0[r], j0[c]) * (1 - y) * (1 - x) + a(i0[r], j1[c]) * (1 - y) * x + a(i1[r], j0[c]) * y * (1 - x) +
                        a(i1[r], j1[c]) * y * x;
        }
    return out;
}

GridD smooth121(const GridD& a_in, const Mask& mask, int passes) {
    const int H = a_in.H, W = a_in.W;
    GridD m(H, W, 0.0), a(H, W, 0.0);
    for (size_t k = 0; k < a.size(); ++k) {
        m.v[k] = mask.v[k] ? 1.0 : 0.0;
        a.v[k] = mask.v[k] ? a_in.v[k] : 0.0;
    }
    for (int p = 0; p < std::max(0, passes); ++p) {
        for (int ax = 0; ax < 2; ++ax) {
            const int di = ax == 0 ? 1 : 0, dj = ax == 0 ? 0 : 1;
            GridD am(H, W, 0.0), out(H, W, 0.0);
            for (size_t k = 0; k < a.size(); ++k) am.v[k] = a.v[k] * m.v[k];
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    if (!mask(i, j)) continue;
                    const int ia = i - di, ja = j - dj, ib = i + di, jb = j + dj;
                    const double sa = a.in(ia, ja) ? am(ia, ja) : 0.0, sb = a.in(ib, jb) ? am(ib, jb) : 0.0;
                    const double ma = a.in(ia, ja) ? m(ia, ja) : 0.0, mb = a.in(ib, jb) ? m(ib, jb) : 0.0;
                    const double num = 2.0 * a(i, j) * m(i, j) + sa + sb;
                    const double den = 2.0 * m(i, j) + ma + mb;
                    out(i, j) = num / std::max(den, 1e-9);
                }
            a = std::move(out);
        }
    }
    return a;
}

GridD laplacian(const GridD& a, const Mask& mask) {
    const int H = a.H, W = a.W;
    GridD out(H, W, 0.0);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            if (!mask(i, j)) continue;
            double s = 0.0;
            for (int d = 0; d < 4; ++d) {
                const int p = i - N4[d][0], q = j - N4[d][1];
                s += (a.in(p, q) && mask(p, q)) ? a(p, q) - a(i, j) : 0.0;
            }
            out(i, j) = s;
        }
    return out;
}

GridD slope_deg(const GridD& h, const Mask& mask, double res_m) {
    const int H = h.H, W = h.W;
    GridD out(H, W, 0.0);
    auto g = [&](int i, int j, int di, int dj) {
        const double self = h(i, j);
        const int p = i - di, q = j - dj;
        if (!h.in(p, q) || !mask(p, q)) return self;
        const double v = h(p, q);
        return std::isnan(v) ? self : v;
    };
    const double den = 2.0 * res_m;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            if (!mask(i, j)) continue;
            const double gx = (g(i, j, 0, -1) - g(i, j, 0, 1)) / den;
            const double gy = (g(i, j, -1, 0) - g(i, j, 1, 0)) / den;
            const double s = std::atan(std::hypot(gx, gy)) * (180.0 / PI);
            out(i, j) = std::isnan(s) ? 0.0 : s;
        }
    return out;
}

double grid_interp(const std::vector<double>& field, const LatLonGrid& g, double lat, double lon) {
    const double fi = (lat - g.lat0) / g.dlat;
    const double fj = (lon - g.lon0) / g.dlon;
    int64_t i0 = static_cast<int64_t>(std::floor(fi));
    i0 = std::min<int64_t>(std::max<int64_t>(i0, 0), g.nlat - 1);
    const int64_t i1 = std::min<int64_t>(std::max<int64_t>(i0 + 1, 0), g.nlat - 1);
    const double ti = clip(fi - static_cast<double>(i0), 0.0, 1.0);
    const int64_t j0f = static_cast<int64_t>(std::floor(fj));
    const double tj = fj - static_cast<double>(j0f);
    auto pm = [&](int64_t a) { int64_t r = a % g.nlon; return r < 0 ? r + g.nlon : r; };
    const int64_t j0 = pm(j0f), j1 = pm(j0f + 1);
    const double f00 = field[i0 * g.nlon + j0], f01 = field[i0 * g.nlon + j1];
    const double f10 = field[i1 * g.nlon + j0], f11 = field[i1 * g.nlon + j1];
    return f00 * (1 - ti) * (1 - tj) + f01 * (1 - ti) * tj + f10 * ti * (1 - tj) + f11 * ti * tj;
}

}  // namespace skyisle
