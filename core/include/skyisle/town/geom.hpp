// 聚落营建器的几何件（docs/PLAN-TOWN.md 第五节「基础件」）：平面向量、有向矩形、多边形、折线。纯函数。
// 平面坐标：x 向东、y 向北，单位 m。朝向一律用罗盘方位角（0 = 北、90° = 东，顺时针），弧度存、度数出。
#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace skyisle::town {

struct V2 {
    double x = 0.0, y = 0.0;
};
inline V2 operator+(V2 a, V2 b) { return {a.x + b.x, a.y + b.y}; }
inline V2 operator-(V2 a, V2 b) { return {a.x - b.x, a.y - b.y}; }
inline V2 operator*(V2 a, double s) { return {a.x * s, a.y * s}; }
inline V2 operator*(double s, V2 a) { return {a.x * s, a.y * s}; }
inline double dot(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
inline double cross(V2 a, V2 b) { return a.x * b.y - a.y * b.x; }
inline double len(V2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }
inline V2 lerp(V2 a, V2 b, double t) { return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t}; }
// 方位角 b（弧度）的单位向量：(sin b, cos b)；右手方向 = (cos b, −sin b)
inline V2 bearing_vec(double b) { return {std::sin(b), std::cos(b)}; }
inline V2 bearing_right(double b) { return {std::cos(b), -std::sin(b)}; }
inline double bearing_of(V2 v) { return std::atan2(v.x, v.y); }   // (−π, π]
double wrap_pi(double a);                                          // 折到 (−π, π]
double angle_diff(double a, double b);                             // |a − b| 折到 [0, π]

// 有向矩形：中心 c、朝向 facing（方位角，门 / 正面朝的方向）、半宽 hw（沿右手方向，面阔的一半）、半深 hd（沿朝向，进深的一半）
struct Obb {
    V2 c;
    double facing = 0.0, hw = 0.0, hd = 0.0;
};
std::array<V2, 4> corners(const Obb& o);   // 右前、左前、左后、右后（逆时针）
// 两个有向矩形是否相交（分离轴；只贴边不算）；clearance > 0 时各自外扩 clearance / 2 再判，< 0 时各自内缩
bool overlap(const Obb& a, const Obb& b, double clearance = 0.0);
bool contains(const Obb& o, V2 p);

// 多边形（简单多边形，顶点次序任意）
double polygon_area(const std::vector<V2>& p);     // 有符号：逆时针为正
V2 polygon_centroid(const std::vector<V2>& p);
bool point_in_polygon(const std::vector<V2>& p, V2 q);

// 点到线段 ab 的距离；t 返回投影参数（夹到 [0, 1]）
double dist_point_segment(V2 p, V2 a, V2 b, double* t = nullptr);
// 点到折线的距离；seg / t 返回最近的段号与段内参数
double dist_point_polyline(V2 p, const std::vector<V2>& line, int* seg = nullptr, double* t = nullptr);
double polyline_length(const std::vector<V2>& line);

// 折线平滑与简化。Chaikin：每轮把每段切成 1/4、3/4 两点，首尾不动（开折线）。attrs 与顶点一一对应、随之线性插值（可为空）
void chaikin(std::vector<V2>& line, std::vector<std::vector<double>>* attrs, int iterations);
std::vector<V2> douglas_peucker(const std::vector<V2>& line, double eps);

}  // namespace skyisle::town
