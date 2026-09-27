// 行星层的公共件：经纬网格、球面几何、kNN、经度周期的值噪声、梯度与散度、标准差、rfft 低通。
// 与 skyisle_gen/sphere.py、noise.py、tectonics._divergence、localwind.band_displacement 的 np.fft 部分同式（运算次序照 numpy）。
#include "skyisle/planet/planet.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

#define POCKETFFT_NO_MULTITHREADING   // 与 numpy/fft/_pocketfft_umath.cpp 同
#include "pocketfft/pocketfft_hdronly.h"

namespace skyisle::planet {

std::vector<double> f32v(const std::vector<double>& v) {
    std::vector<double> out(v.size());
    for (size_t k = 0; k < v.size(); ++k) out[k] = f32(v[k]);
    return out;
}

Axes grid_axes(double res_deg) {
    Axes ax;
    ax.res = res_deg;
    ax.nlat = static_cast<int>(std::nearbyint(180.0 / res_deg));   // int(round(180 / res))
    ax.nlon = static_cast<int>(std::nearbyint(360.0 / res_deg));
    ax.lats.resize(ax.nlat);
    ax.lons.resize(ax.nlon);
    for (int i = 0; i < ax.nlat; ++i) ax.lats[i] = -90.0 + (static_cast<double>(i) + 0.5) * res_deg;
    for (int j = 0; j < ax.nlon; ++j) ax.lons[j] = -180.0 + (static_cast<double>(j) + 0.5) * res_deg;
    return ax;
}

LatLonGrid llg(const Axes& ax) {
    LatLonGrid g;
    g.lat0 = ax.lats[0];
    g.dlat = ax.lats[1] - ax.lats[0];
    g.lon0 = ax.lons[0];
    g.dlon = ax.lons[1] - ax.lons[0];
    g.nlat = ax.nlat;
    g.nlon = ax.nlon;
    return g;
}

Vec3 latlon_to_xyz(double lat_deg, double lon_deg) {
    const double lat = deg2rad(lat_deg), lon = deg2rad(lon_deg);
    const double cl = std::cos(lat);
    return {cl * std::cos(lon), cl * std::sin(lon), std::sin(lat)};
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double angdist(const Vec3& a, const Vec3& b) {
    const Vec3 c = cross(a, b);
    const double s2 = ((0.0 + c.x * c.x) + c.y * c.y) + c.z * c.z;   // np.sum(…, axis=-1)：3 个顺序相加
    const double cd = ((0.0 + a.x * b.x) + a.y * b.y) + a.z * b.z;
    return std::atan2(std::sqrt(s2), cd);
}

void knn(const std::vector<Vec3>& xyz, int k_in, std::vector<int64_t>& idx, std::vector<double>& ang) {
    const int64_t n = static_cast<int64_t>(xyz.size());
    const int k = static_cast<int>(std::min<int64_t>(k_in, n - 1));
    idx.assign(static_cast<size_t>(n) * k, 0);
    ang.assign(static_cast<size_t>(n) * k, 0.0);
    std::vector<double> best_d(k + 1);
    std::vector<int64_t> best_j(k + 1);
    for (int64_t i = 0; i < n; ++i) {
        // 前 k：键 = (−点积, 下标) 升序（与 argpartition + lexsort 同；只有第 k 名并列时 numpy 的挑法未定义）。
        // 点积 = xyz[s:e] @ xyz.T（dgemm / syrk 同为按 k 顺序 FMA），自身记 −2
        int cnt = 0;
        for (int64_t j = 0; j < n; ++j) {
            const double d = j == i ? -2.0 : dot_gemm(xyz[i], xyz[j]);
            if (cnt == k && !(d > best_d[k - 1])) continue;   // j 递增：点积相同时先来的下标小，排前
            int pos = cnt < k ? cnt++ : k - 1;
            while (pos > 0 && d > best_d[pos - 1]) {
                best_d[pos] = best_d[pos - 1];
                best_j[pos] = best_j[pos - 1];
                --pos;
            }
            best_d[pos] = d;
            best_j[pos] = j;
        }
        for (int m = 0; m < k; ++m) {
            idx[static_cast<size_t>(i) * k + m] = best_j[m];
            ang[static_cast<size_t>(i) * k + m] = std::acos(clip(best_d[m], -1.0, 1.0));
        }
    }
}

// ---------------------------------------------------------------- noise.py
namespace {

std::vector<double> value_noise_layer(Rng& rng, int nlat, int nlon, int cells_lat, int cells_lon) {
    std::vector<double> lattice(static_cast<size_t>(cells_lat + 1) * cells_lon);
    rng.uniform_fill(-1.0, 1.0, lattice.data(), lattice.size());
    // np.linspace(0, cells, n, endpoint=False) = arange(n) · (cells / n)，再 + 0.5 · cells / n
    auto axis = [](int cells, int n, std::vector<int64_t>& i0, std::vector<int64_t>& i1, std::vector<double>& t, bool wrap) {
        const double step = static_cast<double>(cells) / static_cast<double>(n);
        const double off = 0.5 * static_cast<double>(cells) / static_cast<double>(n);
        i0.resize(n);
        i1.resize(n);
        t.resize(n);
        for (int k = 0; k < n; ++k) {
            const double f = static_cast<double>(k) * step + off;
            const int64_t a = static_cast<int64_t>(std::floor(f));
            t[k] = smoothstep(clip(f - static_cast<double>(a), 0.0, 1.0));
            if (wrap) {
                i0[k] = a;
                i1[k] = (a + 1) % cells;
            } else {
                i0[k] = std::min<int64_t>(std::max<int64_t>(a, 0), cells - 1);
                i1[k] = std::min<int64_t>(std::max<int64_t>(i0[k] + 1, 0), cells);
            }
        }
    };
    std::vector<int64_t> i0, i1, j0, j1;
    std::vector<double> ti, tj;
    axis(cells_lat, nlat, i0, i1, ti, false);
    axis(cells_lon, nlon, j0, j1, tj, true);
    std::vector<double> out(static_cast<size_t>(nlat) * nlon);
    for (int r = 0; r < nlat; ++r) {
        const double* L0 = &lattice[static_cast<size_t>(i0[r]) * cells_lon];
        const double* L1 = &lattice[static_cast<size_t>(i1[r]) * cells_lon];
        const double a = 1 - ti[r], b = ti[r];
        for (int c = 0; c < nlon; ++c) {
            const double v00 = L0[j0[c]], v01 = L0[j1[c]], v10 = L1[j0[c]], v11 = L1[j1[c]];
            out[static_cast<size_t>(r) * nlon + c] =
                v00 * a * (1 - tj[c]) + v01 * a * tj[c] + v10 * b * (1 - tj[c]) + v11 * b * tj[c];
        }
    }
    return out;
}

}  // namespace

std::vector<double> fractal_noise(Rng& rng, int nlat, int nlon, int base_cells, int octaves, double persistence, double lacunarity) {
    std::vector<double> out(static_cast<size_t>(nlat) * nlon, 0.0);
    double amp = 1.0, total = 0.0;
    int c_lat = base_cells, c_lon = base_cells * 2;
    for (int o = 0; o < octaves; ++o) {
        const std::vector<double> layer = value_noise_layer(rng, nlat, nlon, c_lat, c_lon);
        for (size_t k = 0; k < out.size(); ++k) out[k] += amp * layer[k];
        total += amp;
        amp *= persistence;
        c_lat = static_cast<int>(std::nearbyint(static_cast<double>(c_lat) * lacunarity));   // int(round(…))：逢半取偶
        c_lon = static_cast<int>(std::nearbyint(static_cast<double>(c_lon) * lacunarity));
    }
    for (double& x : out) x /= total;
    return out;
}

// ---------------------------------------------------------------- 梯度、散度、标准差
std::vector<double> gradient_rows(const std::vector<double>& f, int nlat, int nlon, double dx) {
    std::vector<double> out(f.size());
    const double dx2 = 2. * dx;
    for (int i = 1; i + 1 < nlat; ++i)
        for (int j = 0; j < nlon; ++j)
            out[static_cast<size_t>(i) * nlon + j] = (f[static_cast<size_t>(i + 1) * nlon + j] - f[static_cast<size_t>(i - 1) * nlon + j]) / dx2;
    for (int j = 0; j < nlon; ++j) {
        out[j] = (f[static_cast<size_t>(nlon) + j] - f[j]) / dx;
        const size_t a = static_cast<size_t>(nlat - 1) * nlon + j, b = static_cast<size_t>(nlat - 2) * nlon + j;
        out[a] = (f[a] - f[b]) / dx;
    }
    return out;
}

std::vector<double> divergence(const std::vector<double>& u, const std::vector<double>& v, const Axes& ax) {
    const double R = 6.371e6;
    const int H = ax.nlat, W = ax.nlon;
    const double dlam = deg2rad(ax.lons[1] - ax.lons[0]);
    const double dphi = deg2rad(ax.lats[1] - ax.lats[0]);
    std::vector<double> cosp(H), cosphi(H);
    for (int i = 0; i < H; ++i) {
        cosp[i] = std::cos(deg2rad(ax.lats[i]));
        cosphi[i] = np_maximum(cosp[i], 0.05);
    }
    std::vector<double> vc(u.size());
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) vc[static_cast<size_t>(i) * W + j] = v[static_cast<size_t>(i) * W + j] * cosp[i];
    const std::vector<double> dv = gradient_rows(vc, H, W, dphi);
    std::vector<double> out(u.size());
    const double c2 = 2 * dlam * R;
    for (int i = 0; i < H; ++i) {
        const double den_x = c2 * cosphi[i], den_y = R * cosphi[i];
        const double* ur = &u[static_cast<size_t>(i) * W];
        for (int j = 0; j < W; ++j) {
            const double du_dx = (ur[(j + 1) % W] - ur[(j - 1 + W) % W]) / den_x;
            out[static_cast<size_t>(i) * W + j] = du_dx + dv[static_cast<size_t>(i) * W + j] / den_y;
        }
    }
    return out;
}

double np_std(const std::vector<double>& a) {
    const size_t n = a.size();
    const double m = np_sum(a.data(), n) / static_cast<double>(n);
    std::vector<double> x(n);
    for (size_t k = 0; k < n; ++k) {
        const double d = a[k] - m;
        x[k] = d * d;
    }
    return std::sqrt(np_sum(x.data(), n) / static_cast<double>(n));
}

// ---------------------------------------------------------------- np.fft.rfft / irfft（numpy/fft/_pocketfft_umath.cpp 的 rfft_impl / irfft_loop）
std::vector<double> rfft_lowpass(const std::vector<double>& a, int kmax) {
    const size_t n = a.size();
    const size_t nout = n / 2 + 1;
    pocketfft::detail::pocketfft_r<double> plan(n);
    // rfft：实数放在缓冲区的第 1 格起，执行后把 I0 挪成 R0（fct = 1）
    std::vector<double> buf(2 * nout, 0.0);
    std::copy(a.begin(), a.end(), buf.begin() + 1);
    plan.exec(buf.data() + 1, 1.0, true);
    buf[0] = buf[1];
    buf[1] = 0.0;
    // F[kmax+1:] = 0
    for (size_t k = static_cast<size_t>(kmax) + 1; k < nout; ++k) buf[2 * k] = buf[2 * k + 1] = 0.0;
    // irfft(F, n)：R0, R1, I1, …（偶数 n 最后一格是 Rn/2），fct = 1 / n
    std::vector<double> c(n);
    c[0] = buf[0];
    for (size_t k = 1; k <= (n - 1) / 2; ++k) {
        c[2 * k - 1] = buf[2 * k];
        c[2 * k] = buf[2 * k + 1];
    }
    if (n % 2 == 0) c[n - 1] = buf[2 * (n / 2)];
    plan.exec(c.data(), 1.0 / static_cast<double>(n), false);
    return c;
}

}  // namespace skyisle::planet
