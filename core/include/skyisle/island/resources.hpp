// 5.3c 地形区与资源（resources.py 同式）：点 / 片 / 散（赋存场 → 赋存区 → 采场），岩类只在岩类可放区。
// 记录里的数与 Python 版 dict 里的数一一对应（该四舍五入的已按 pyround 舍好）；字符串是 ASCII 代码，中文在前端（skyisle_gen/island/decode.py）。
#pragma once

#include <array>
#include <string>
#include <vector>

#include "skyisle/json.hpp"
#include "skyisle/island/types.hpp"

namespace skyisle::island {

// 资源类：编码 = RES_KINDS 的序号 + 1（resource 栅格的值）。P3（没有火山，2026-09-29）在末尾加了岩盐、盐泉、贝壳化石，前 13 类的编码不变
enum ResKind { RK_NONE = 0, RK_TIMBER, RK_SPRING, RK_CLAY, RK_PEAT, RK_GRAVEL, RK_PLACER, RK_STONE, RK_FLOATSTONE, RK_ORE,
               RK_HOTSPRING, RK_SULFUR, RK_CAVE, RK_GUANO, RK_SALT, RK_SALTSPRING, RK_FOSSIL, RK_COUNT };
enum ResForm { FORM_POINT, FORM_PATCH, FORM_FIELD };
// res_field 的层序（FIELD_KINDS）；岩盐是 P3 加的第 7 层
enum FieldKind { FK_ORE = 0, FK_SULFUR, FK_PLACER, FK_CLAY, FK_GRAVEL, FK_STONE, FK_SALT, FK_COUNT };
enum Zone { Z_VOID = 0, Z_ALPINE, Z_MOUNTAIN, Z_HILL, Z_PLAIN, Z_VALLEY, Z_CLIFF, Z_WATER };

const char* res_key(int kind);        // "timber" …
ResForm res_form(int kind);
int field_of(int kind);               // 散的层号；不是散 → −1
int field_res(int fk);                // 层号 → 资源类
const char* zone_key(int zone);       // "alpine" …
const char* grade_key(double q);      // "hi"（上，≥ 0.75）/ "mid"（中，≥ 0.45）/ "lo"（下）

// 岩类可放区（rock_site_mask）：山地、高山、裸岩 / 高山草甸，或丘陵且坡 ≥ rock_hill_slope_deg；不含耕地、湿地、漫滩、水、崖缘
Mask rock_site_mask(const Group& g, const Grid<uint8_t>& zone, double rock_hill_slope_deg);
// 地形区 + 点 / 片 / 散；需 build_hydro 之后（g.islands 的 rim_j / peak_j / cliff_j 已是 island.json 的值）
void build_resources(Group& g, const Config& c);
// 岩类采场开在林坡 / 灌丛上：那格改裸岩、从林场划掉
void open_working(Group& g, int i, int j);
// 片的面积 / 代表格（按 patch_id 重数）、主导栅格、赋存区的采场数（sync_resources 的栅格与记录部分）
void sync_resources(Group& g);

Json deposit_json(const Deposit& d);
Json occurrence_json(const Occurrence& o);
Json working_json(const Working& w);

}  // namespace skyisle::island
