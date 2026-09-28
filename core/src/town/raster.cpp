#include "skyisle/town/raster.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace skyisle::town {

uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

double hash_unit(uint64_t seed, int64_t i, int64_t j) {
    const uint64_t h = mix64(seed ^ mix64(static_cast<uint64_t>(i) * 0x632BE59BD9B4E019ULL + static_cast<uint64_t>(j) * 0x85157AF5ULL));
    return static_cast<double>(h >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

namespace {
inline double fade(double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
}

double value_noise(uint64_t seed, double x, double y) {
    const double fx = std::floor(x), fy = std::floor(y);
    const int64_t i = static_cast<int64_t>(fx), j = static_cast<int64_t>(fy);
    const double u = fade(x - fx), v = fade(y - fy);
    const double a = hash_unit(seed, i, j), b = hash_unit(seed, i + 1, j);
    const double c = hash_unit(seed, i, j + 1), d = hash_unit(seed, i + 1, j + 1);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}

double fbm(uint64_t seed, double x, double y, double wavelength, int octaves, double persistence) {
    double amp = 1.0, freq = 1.0 / wavelength, sum = 0.0, tot = 0.0;
    for (int k = 0; k < octaves; ++k) {
        sum += amp * value_noise(mix64(seed + static_cast<uint64_t>(k) * 0x9E37ULL), x * freq, y * freq);
        tot += amp;
        amp *= persistence;
        freq *= 2.0;
    }
    return tot > 0.0 ? sum / tot : 0.0;
}

namespace {
inline double at(const GridD& g, int i, int j) {
    i = std::clamp(i, 0, g.H - 1);
    j = std::clamp(j, 0, g.W - 1);
    return g(i, j);
}
inline double cr(double p0, double p1, double p2, double p3, double t) {
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)));
}
}  // namespace

double sample_cubic(const GridD& g, double r, double c) {
    const double y = r - 0.5, x = c - 0.5;
    const int i = static_cast<int>(std::floor(y)), j = static_cast<int>(std::floor(x));
    const double ty = y - i, tx = x - j;
    double col[4];
    for (int k = -1; k <= 2; ++k)
        col[k + 1] = cr(at(g, i + k, j - 1), at(g, i + k, j), at(g, i + k, j + 1), at(g, i + k, j + 2), tx);
    return cr(col[0], col[1], col[2], col[3], ty);
}

double sample_linear(const GridD& g, double r, double c) {
    const double y = r - 0.5, x = c - 0.5;
    const int i = static_cast<int>(std::floor(y)), j = static_cast<int>(std::floor(x));
    const double ty = y - i, tx = x - j;
    const double a = at(g, i, j), b = at(g, i, j + 1), d = at(g, i + 1, j), e = at(g, i + 1, j + 1);
    return (a + (b - a) * tx) * (1.0 - ty) + (d + (e - d) * tx) * ty;
}

double sample_linear(const Mask& m, double r, double c) {
    const double y = r - 0.5, x = c - 0.5;
    const int i = static_cast<int>(std::floor(y)), j = static_cast<int>(std::floor(x));
    const double ty = y - i, tx = x - j;
    auto mv = [&](int a, int b) {
        a = std::clamp(a, 0, m.H - 1);
        b = std::clamp(b, 0, m.W - 1);
        return m(a, b) ? 1.0 : 0.0;
    };
    const double a = mv(i, j), b = mv(i, j + 1), d = mv(i + 1, j), e = mv(i + 1, j + 1);
    return (a + (b - a) * tx) * (1.0 - ty) + (d + (e - d) * tx) * ty;
}

namespace {
// 一维平方距离变换（Felzenszwalb & Huttenlocher 2012）：f[q] 是 q 处的代价（种子 0、其余 inf 或上一轮的平方距离），
// 输出 d[q] = min_p (q − p)² + f[p] 与取到最小的 p
void dt1d(const std::vector<double>& f, int n, std::vector<double>& d, std::vector<int>& arg, std::vector<int>& v, std::vector<double>& z) {
    const double inf = std::numeric_limits<double>::infinity();
    int k = -1;
    for (int q = 0; q < n; ++q) {
        if (!(f[q] < inf)) continue;
        if (k < 0) {
            k = 0, v[0] = q, z[0] = -inf, z[1] = inf;
            continue;
        }
        double s = 0.0;
        while (true) {
            const int p = v[k];
            s = ((f[q] + static_cast<double>(q) * q) - (f[p] + static_cast<double>(p) * p)) / (2.0 * (q - p));
            if (s <= z[k] && k > 0) {
                --k;
                continue;
            }
            break;
        }
        if (s <= z[k]) {   // k == 0 且新抛物线整条更低
            v[0] = q, z[0] = -inf, z[1] = inf;
            continue;
        }
        ++k;
        v[k] = q, z[k] = s, z[k + 1] = inf;
    }
    if (k < 0) {
        for (int q = 0; q < n; ++q) d[q] = inf, arg[q] = -1;
        return;
    }
    int t = 0;
    for (int q = 0; q < n; ++q) {
        while (z[t + 1] < q) ++t;
        const int p = v[t];
        d[q] = static_cast<double>(q - p) * (q - p) + f[p];
        arg[q] = p;
    }
}
}  // namespace

void edt(const Mask& seed, GridF& dist, Grid<int32_t>* src) {
    // 精确欧氏距离：先按列、再按行做一维平方距离变换，列那一遍记下最近种子的行号
    const int H = seed.H, W = seed.W;
    const double inf = std::numeric_limits<double>::infinity();
    const int n = std::max(H, W);
    std::vector<double> f(n), d(n), z(n + 1);
    std::vector<int> arg(n), v(n);
    std::vector<double> col_d(static_cast<size_t>(H) * W);
    std::vector<int32_t> col_row(static_cast<size_t>(H) * W);
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) f[i] = seed(i, j) ? 0.0 : inf;
        dt1d(f, H, d, arg, v, z);
        for (int i = 0; i < H; ++i) col_d[static_cast<size_t>(i) * W + j] = d[i], col_row[static_cast<size_t>(i) * W + j] = arg[i];
    }
    dist = GridF(H, W, std::numeric_limits<float>::infinity());
    if (src) *src = Grid<int32_t>(H, W, -1);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) f[j] = col_d[static_cast<size_t>(i) * W + j];
        dt1d(f, W, d, arg, v, z);
        for (int j = 0; j < W; ++j) {
            if (!(d[j] < inf)) continue;
            dist(i, j) = static_cast<float>(std::sqrt(d[j]));
            if (src) {
                const int jj = arg[j];
                const int ii = col_row[static_cast<size_t>(i) * W + jj];
                (*src)(i, j) = ii * W + jj;
            }
        }
    }
}

}  // namespace skyisle::town
