// 特殊的山与地貌（PLAN-NATURE B3，DESIGN-NOTES 四点四十六；spec 13 第二节第 8、9 条）：按成因条件出，不设配额。
// 地形阶段（sculpt_island 里）：平行岭谷（接缝带的褶皱）、掀斜断块山（老核）；
// 贴图之前（landform_pass，浮高之后、按最终高程）：峰林（老岛、沉积盖层、暖湿——压低峰间的平地，石柱只记范围 / 密度 / 高）、
//   冰斗与角峰（峰在雪线以上）、临空断山（峰贴着岸缘、按条件概率整片崩掉）；天上的山（浮高，build 里标）；
// 资源之后（detect_landforms）：辉长岩锯齿峰、蛇纹岩秃山、方山 / 孤山（老岛的石灰岩硬盖）、穿山天窗（薄山脊碰上骨架空洞），与崖层 rockwall。
#pragma once

#include "skyisle/island/terrain.hpp"
#include "skyisle/island/types.hpp"

namespace skyisle::island {

// 贴图前那一步要的气候：零点口径的年均气温与直减率、群的年雨（mm）、台面处最暖一季的气温
struct LandformEnv {
    double t_sea = 0, lapse = 6.0, P_mm = 0, t_warm = 0, h_ref = 0;
    double ela = INF;                 // 雪线（平衡线，m）
    double pole_y = 1.0;              // 朝极的方向（北半球 +1 = 北、南半球 −1）：背阳的坡留雪
};
LandformEnv landform_env(const NodeInputs& inp, const PlanetView& pv, const Config& c);

// 一座岛：就地改 s.mask / s.phi（崩掉的那片）与 sc.h / top / skel，地貌记进 sc.lf（局部栅格行列）
void landform_pass(Rng& rng, const LandformEnv& e, const Config& c, double area_km2, double res_km, Shape& s, Sculpt& sc);

// 水系之后按最终高程出 lith（hydro 调）
void compute_lith(Group& g);
// 资源之后：识别型的地貌与崖层
void detect_landforms(Group& g, const Config& c);
// island.json 的 landforms（群栅格行列 → km）
Json landform_json(const Group& g, const LandformRec& r, int id);

}  // namespace skyisle::island
