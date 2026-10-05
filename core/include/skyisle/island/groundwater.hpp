// 集水核（PLAN-NATURE C4，spec 13 第八节）与地下水（C5，spec 13 第一节第 7 条、第二节第 6 条）；DESIGN-NOTES 四点四十七。
// - 集水核长在山根的浮石骨架里，大小跟山走；让流过山的空气里的水汽凝在林子、岩面上（云雾截留，放大）：凝结水只进水账（渗进岩层 → 基流与泉），
//   不进局地雨、不进天气；最多到当地局地雨的 1 倍，跟着湿度走（干的群照样干）。凝结的热千分之一进岩体 → 大核山旁有温泉。
// - 岛是两层：上面的岩层是含水层，闭孔的浮石骨架是地下水的底。各岩性的基流比例与退水不同（石灰岩岩溶、辉长岩裂隙、泥灰岩与蛇纹岩差）；
//   地下水顺流向走：进了河道就是河的基流，没进河道就走到崖边，从崖壁上岩层与浮石的交界渗出来——崖壁泉线。
#pragma once

#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

// C4：各格一年的凝结水（mm，陆地格；其余 0）。Gw：顺风方向的地势升降（m/km，局地雨算的山脉尺度迎风坡；空 = 不分迎背风）。
// core_s（出）：各岛集水核的强度（跟山走：高出岸缘的山体体积的立方根 / core_len_km，夹 core_s_max）
// Research opt-in: water.core_exchange_days > 0 uses inp.water_column_mm and a
// prescribed balanced ocean source; requires explicit physical inputs. Existing
// default remains unchanged. See DESIGN-NOTES 2026-10-05 for unvalidated terms.
GridD condensation(const Group& g, const GridD& Gw, const Config& c, std::vector<double>& core_s);

// C5：含水层。水系、岩性之后调：各格的基流比例（按出露岩性）、补给（= 雨的径流 × 基流比例 + 凝结水）、
// 顺流向累计的补给（河道格 = 河的基流）、崖壁泉线（没进河道、从岸边走出的地下水，按段合起来）
void aquifer(Group& g, const Config& c);

// 群栅格的 D8 下游（扁平下标，−1 = 出口或虚空）：recv_i / recv_j 拼成
std::vector<int64_t> group_recv(const Group& g);

// C4：一座岛的集水核放进岩体的热（MW，= Σ 凝结 × 潜热 × heat_share；温泉按它的多少给）
double core_heat_mw(const Group& g, int island, const Config& c);

}  // namespace skyisle::island
