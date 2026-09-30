// 行星层 ①–④（行星计划 P6c）：skyisle_gen/stages/s01–s04 与 almanac / sphere / noise / tectonics / localwind / moisture / skeleton 的 C++ 版。
//
// 每步一个函数：stage1（行星与历法）→ stage2（风带与 G）→ stage3（岛群分布、板块、候选边）→ stage4（岛对风的扰动、温度、风暴、水汽追踪降水、
// 季节强度、各群的气候标量）。产物结构与 npz / json 同名同形；浮点一律存双精度的「原值」（Python 在内存里的数），
// npz 里存 float32 的字段由**读它的一方**先按 float32 舍一次（f32()）——与 Python 版下游读 npz 的口径相同，产物从 npz 读回来也不变（舍两次同一次）。
// 运算次序照 numpy：成对求和、BLAS 的乘加次序（dgemm / ddot = 按 k 顺序 FMA，dgemv [M,3]·[3] = fma(a₂, x₂, fma(a₀, x₀, a₁·x₁))）、
// np.fft（pocketfft，第三方头文件）、numpy 标量的 ** = C 的 pow（DESIGN-NOTES 四点二十五）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/grid.hpp"

namespace skyisle::planet {

inline double f32(double x) { return static_cast<double>(static_cast<float>(x)); }
// np.maximum / np.minimum：相等时取第二个（与 x86 的 maxpd / minpd 同），±0 的符号跟着走；Python 内置 max / min 与 std::max / min 同（相等取第一个）
inline double np_maximum(double a, double b) { return a > b ? a : b; }
inline double np_minimum(double a, double b) { return a < b ? a : b; }
std::vector<double> f32v(const std::vector<double>& v);

// ---------------------------------------------------------------- 球面与经纬网格（sphere.py）
struct Axes {                     // grid_axes：格心，lats 南 → 北、lons −180 → 180
    int nlat = 0, nlon = 0;
    double res = 1.0;
    std::vector<double> lats, lons;
    size_t size() const { return static_cast<size_t>(nlat) * nlon; }
};
Axes grid_axes(double res_deg);
LatLonGrid llg(const Axes& ax);   // grid_interp 用：dlat = lats[1] − lats[0]

inline double deg2rad(double x) { return x * (PI / 180.0); }   // np.radians / math.radians
inline double rad2deg(double x) { return x * (180.0 / PI); }   // np.degrees
struct Vec3 {
    double x = 0, y = 0, z = 0;
};
Vec3 latlon_to_xyz(double lat_deg, double lon_deg);
Vec3 cross(const Vec3& a, const Vec3& b);                       // np.cross：a₁b₂ − a₂b₁ …
double angdist(const Vec3& a, const Vec3& b);                   // arctan2(|a×b|, a·b)，各分量顺序相加
// kNN（sphere.knn）：点积 = BLAS dgemm（按 k 顺序 FMA），按 (−点积, 下标) 取前 k；返回 idx[N, k] 与 arccos 角距
void knn(const std::vector<Vec3>& xyz, int k, std::vector<int64_t>& idx, std::vector<double>& ang);
inline double dot_gemm(const Vec3& a, const Vec3& b) { return std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x)); }
inline double dot_gemv(const Vec3& a, const Vec3& b) { return std::fma(a.z, b.z, std::fma(a.x, b.x, a.y * b.y)); }

// 经度周期的多倍频值噪声（noise.fractal_noise，[nlat, nlon]，归一到 [−1, 1]）
std::vector<double> fractal_noise(Rng& rng, int nlat, int nlon, int base_cells, int octaves, double persistence = 0.55,
                                  double lacunarity = 2.0);
// np.gradient(f, dx, axis=0)（等距，边上一阶）
std::vector<double> gradient_rows(const std::vector<double>& f, int nlat, int nlon, double dx);
// tectonics._divergence：球面水平散度（单位 1/m，只用相对值）
std::vector<double> divergence(const std::vector<double>& u, const std::vector<double>& v, const Axes& ax);
double np_std(const std::vector<double>& a);                    // ndarray.std()（1 维，成对求和）
// np.fft.rfft → F[kmax+1:] = 0 → np.fft.irfft(F, n)（pocketfft，与 numpy 2.x 同一份实现）
std::vector<double> rfft_lowpass(const std::vector<double>& a, int kmax);

// ---------------------------------------------------------------- ① 行星与历法（s01_planet.py、almanac.py）
struct Star {
    double mass_msun = 0, luminosity_lsun = 0, radius_rsun = 0, teff_k = 0, apparent_size_rel_sun = 0;
    std::string spectral_class;
};
struct Moon {
    double synodic_month_days = 0, sidereal_month_days = 0, months_per_year = 0, distance_km = 0, distance_planet_radii = 0,
           tide_rel_earth_same_mass = 0;
};
struct Almanac {                  // planet.json 的 calendar（almanac.derive；没有 [s01.calendar] 时 present = false）
    bool present = false;
    std::string mode;
    int seasons = 4, months_per_season = 1;
    double days_per_season_config = 0, solar_day_hr = 24, gravity_rel = 1, planet_mass_rel_earth = 1;
    double year_days_solar = 0, days_per_season = 0, orbital_period_earth_days = 0, orbital_period_years = 0,
           semi_major_axis_au = 0, insolation_derived = 0, insolation_config = 0;
    bool has_residual = false;
    double days_per_season_residual = 0;
    Star star;
    double tidal_lock_gyr = 0;
    int seasons_needed_for_no_lock = -1;   // −1 = 没有这一项
    bool has_moon = false;
    Moon moon;
};
// 零点与垂直结构（PLAN-NATURE A1，spec 13 第九节第 2–4 条）：planet.json 的 vertical；没有 [s01.atmosphere] 时 present = false。
// 高度一律相对零点（浮层基准：气压约一个大气压的高度），往上为正、m。纯换算，不改任何场。
struct Vertical {
    bool present = false;
    double datum_pressure_atm = 1, scale_height_km = 8, o2_fraction = 0.2095;
    double sea_level_m = -6800, cloud_base_m = -6200, cloud_top_m = -5000, keel_gap_m = 1000;
    // 派生
    double keel_floor_m = 0;       // 岛底最低 = 云带顶 + 间隙
    double sea_pressure_atm = 0, sea_o2_atm = 0;
    double pressure_atm(double z_m) const;   // p(z) = p0 · exp(−z / H)
};
Vertical derive_vertical(const Config& cfg);

struct Planet {                   // planet.json
    double rotation_period_hr = 24, axial_tilt_deg = 0, insolation_rel = 1, radius_km = 6371, band_scale = 1;
    double eq_storm_top = 0, trades_top = 0, calm_top = 0, westerlies_top = 0;   // bands（*_deg）
    double day_range_km = 500, circumference_days = 0;
    Almanac cal;
    Vertical vert;
    double band(const std::string& key) const;                     // "eq_storm_top_deg" / "trades_top_deg" / "calm_top_deg" / "westerlies_top_deg"
    double year_days(double fallback = 336.0) const { return cal.present ? cal.year_days_solar : fallback; }
};
Almanac derive_calendar(const Config& cfg);
Planet stage1(const Config& cfg);

// ---------------------------------------------------------------- ② 风带（s02_wind.py）
struct GInfo {                    // bands.json 的 G
    double lat = 0, lon = 0, radius_deg = 0;
    std::string edge;             // 锚定的带界键（skeleton.g_anchor_edge）
};
struct Winds {                    // wind.npz（u / v 存 float32）+ bands.json
    Axes ax;
    std::vector<double> u, v;
    std::vector<int16_t> band;
    GInfo g;
};
// 背景风系（wind_profile，不含 G）与 G 涡旋（g_vortex），④ 按局部带界坐标复用
void wind_profile(const std::vector<double>& lat_eff, const Config& cfg, const Planet& p, std::vector<double>& u,
                  std::vector<double>& v);
void g_vortex(const Axes& ax, double g_lat, double g_lon, double g_r_deg, double vmax, std::vector<double>& du,
              std::vector<double>& dv);
Winds stage2(const Config& cfg, const Planet& p);

// ---------------------------------------------------------------- ③ 岛群分布、板块、候选边（s03_islands.py、tectonics.py）
struct Islands {
    // islands.npz：lat / lon / xyz 存 float64，其余浮点存 float32
    std::vector<double> lat, lon, xyz, area, height, territory, land_frac, arable_frac, main_frac, main_area, wall, age, density_at,
        mean_nn;
    std::vector<int16_t> plate;
    std::vector<int8_t> cls;
    std::vector<uint8_t> layered;
    // plates.npz（boundary_kernel / conv_kernel / age / factor 存 float32）
    Axes ax;
    std::vector<int16_t> plate_id;
    std::vector<int8_t> btype;
    std::vector<double> boundary_kernel, conv_kernel, plate_age, factor, seeds_xyz;
    // cand_edges.npz
    std::vector<int64_t> src, dst;
    std::vector<double> dist_days;
    std::vector<int8_t> kind;
    // density_grid.npz（float32）
    std::vector<double> density;
    // 摘要用（不存盘）
    std::vector<uint8_t> in_stack;
    int n_exp = 0, n_fallback = 0, n_chord = 0;
    double f0 = 0;
    size_t n() const { return lat.size(); }
};
struct Tectonics {                // tectonics.plate_fields
    std::vector<double> factor, conv_kernel, age, boundary_kernel, seeds_xyz;
    std::vector<int16_t> plate_id;
    std::vector<int8_t> btype;
};
Tectonics plate_fields(Rng& rng, const Axes& ax, const Config& cfg, const std::vector<double>* wind_u, const std::vector<double>* wind_v);
Islands stage3(const Config& cfg, uint64_t seed, const Planet& p, const Winds& w);

// ---------------------------------------------------------------- ④ 局地风与气候（s04_climate.py、localwind.py、moisture.py、skeleton.season_range）
extern const char* const EDGE_KEYS[8];   // eq_n trades_n calm_n west_n eq_s trades_s calm_s west_s
struct Climate {
    Axes ax;
    // wind_local.npz（浮点存 float32）
    std::vector<double> u, v, u_bg, v_bg, v_local, obstacle, land, wake, lat_eff;
    std::vector<int16_t> band;
    // band_local.npz：edges / dphi [8, nlon]（float32），行序 EDGE_KEYS
    std::vector<double> edges, dphi;
    // climate_grid.npz（float32）
    std::vector<double> precip, temp, storm, storm_no_g, stability, window, q, uplift, conv, eps, season_range, continentality;
    // climate_islands.npz（float32；has_river 布尔）
    std::vector<double> i_precip, i_temp, i_storm, i_stability, i_window, i_catch, i_river_size, i_temp_sea, i_season_range_sea,
        i_season_range, i_temp_winter, i_temp_summer;
    std::vector<uint8_t> i_has_river;
    // 摘要（水汽模型）
    double dt_s = 0;
    int64_t n_steps = 0;
};
// skeleton.season_range：全年温差 = 2 × 日照年变化一阶谐波 / λ × 振幅保留（lat 与 cont 等长）
std::vector<double> season_range(const std::vector<double>& lat_deg, const std::vector<double>& cont, double tilt_deg, double year_days,
                                 const Config& cfg, double insolation_rel);
std::vector<double> insolation_first_harmonic(const std::vector<double>& lat_deg, double tilt_deg, int n = 360);
Climate stage4(const Config& cfg, uint64_t seed, const Planet& p, const Winds& w, const Islands& isl);

// ---------------------------------------------------------------- 整个行星层（游戏新建世界时一次跑完，不经 npz）
struct World {
    Planet planet;
    Winds winds;
    Islands islands;
    Climate climate;
};
World run(const Config& cfg, uint64_t seed);

}  // namespace skyisle::planet
