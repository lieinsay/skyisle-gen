// - 岛是两层：上面的岩层是含水层，闭孔的浮石骨架是地下水的底。各岩性的基流比例与退水不同（石灰岩岩溶、辉长岩裂隙、泥灰岩与蛇纹岩差）；
//   地下水顺流向走：进了河道就是河的基流，没进河道就走到崖边，从崖壁上岩层与浮石的交界渗出来——崖壁泉线。
#pragma once

#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

// C5：含水层。水系、岩性之后调：各格的基流比例（按出露岩性）、补给（= 雨的径流 × 基流比例）、
// 顺流向累计的补给（河道格 = 河的基流）、崖壁泉线（没进河道、从岸边走出的地下水，按段合起来）
void aquifer(Group& g, const Config& c);

// 群栅格的 D8 下游（扁平下标，−1 = 出口或虚空）：recv_i / recv_j 拼成
std::vector<int64_t> group_recv(const Group& g);

}  // namespace skyisle::island
