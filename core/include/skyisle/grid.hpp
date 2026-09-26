// 栅格公共件：Grid<T>（行主序）、值噪声 / 分形噪声、连通分量、形态学、倒角传播、重采样、平滑、坡度。
// 与 skyisle_gen/island/grid.py 同式（numpy 的 shift(a, di, dj)[i, j] = a[i − di, j − dj]，出界填 fill）。
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "skyisle/rng.hpp"

namespace skyisle {

template <class T>
struct Grid {
    int H = 0, W = 0;
    std::vector<T> v;
    Grid() = default;
    Grid(int h, int w, T fill = T()) : H(h), W(w), v(static_cast<size_t>(h) * static_cast<size_t>(w), fill) {}
    T& operator()(int i, int j) { return v[static_cast<size_t>(i) * W + j]; }
    const T& operator()(int i, int j) const { return v[static_cast<size_t>(i) * W + j]; }
    T& operator[](size_t k) { return v[k]; }
    const T& operator[](size_t k) const { return v[k]; }
    size_t size() const { return v.size(); }
    bool in(int i, int j) const { return i >= 0 && j >= 0 && i < H && j < W; }
};

using GridD = Grid<double>;
using GridF = Grid<float>;
using GridI = Grid<int32_t>;
using Mask = Grid<uint8_t>;   // 0 / 1

constexpr int N4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
constexpr int N8[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double SQRT2 = 1.4142135623730951;   // math.sqrt(2.0)
constexpr double PI = 3.141592653589793;

// ---------------------------------------------------------------- 与 numpy / Python 同式的小工具
double np_sum(const double* a, size_t n);          // numpy 的成对求和（np.sum / ndarray.sum 的 1 维）
double np_median(std::vector<double> v);           // np.median（偶数个取中间两个的平均）
double pyround(double x, int ndigits);             // Python 的 round(x, n)：二进制值的精确十进制舍入、逢半取偶
double pymod(double x, double m);                  // Python 的浮点 %（结果与除数同号）
inline double smoothstep(double t) { return t * t * (3.0 - 2.0 * t); }
inline double clip(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---------------------------------------------------------------- 噪声（grid.py 的 LatticeNoise / FractalNoise）
class LatticeNoise {
public:
    LatticeNoise(Rng& rng, double x0, double y0, double x1, double y1, double cell_km);
    double sample(double x, double y) const;

private:
    double cell_, x0_, y0_;
    int ny_, nx_;
    std::vector<double> lat_;
};

class FractalNoise {
public:
    FractalNoise(Rng& rng, double x0, double y0, double x1, double y1, double feature_km, int octaves = 4,
                 double persistence = 0.5, double lacunarity = 2.0);
    double sample(double x, double y) const;

private:
    std::vector<std::pair<double, LatticeNoise>> layers_;
    double total_ = 0.0;
};

// ---------------------------------------------------------------- 连通分量
// 4 / 8 连通分量标号：labels 0 = 背景，1..n 按光栅扫描首次出现的次序（与 grid.label_components 同）。返回 n。
int label_components(const Mask& mask, int connectivity, GridI& labels);
Mask largest_component(const Mask& mask, int* count = nullptr);

// ---------------------------------------------------------------- 形态学
Mask binary_erode(const Mask& m, int iterations = 1, int connectivity = 8);
Mask binary_dilate(const Mask& m, int iterations = 1, int connectivity = 8);
GridI distance_bands(const Mask& mask, int max_iter);
// 倒角距离传播（grid.nearest_propagate）：dist（step_m 为单位，出界 inf）与最近种子的扁平下标 src（无则 −1）
void nearest_propagate(const Mask& seed, int max_iter, double step_m, const Mask* within, GridD& dist, Grid<int64_t>& src);

// ---------------------------------------------------------------- 重采样与平滑
GridD block_mean(const GridD& a, int f);
Mask block_any(const Mask& m, int f);
GridD upsample_bilinear(const GridD& a, int f, int H, int W);
GridD smooth121(const GridD& a, const Mask& mask, int passes);
GridD laplacian(const GridD& a, const Mask& mask);
GridD slope_deg(const GridD& h, const Mask& mask, double res_m);

// ---------------------------------------------------------------- 经纬网格上的双线性插值（sphere.grid_interp，经度周期）
struct LatLonGrid {
    double lat0 = 0, dlat = 1, lon0 = 0, dlon = 1;   // lats[0]、lats[1] − lats[0]（按原数组的 dtype 算好传进来）
    int nlat = 0, nlon = 0;
};
double grid_interp(const std::vector<double>& field, const LatLonGrid& g, double lat, double lon);

}  // namespace skyisle
