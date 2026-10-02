// 河的数据（PLAN-NATURE C3，DESIGN-NOTES 四点四十七）：只给驱动量，流速、当天水位与水面宽、河段型、冰情、水色由游戏推。
// - 干支分段：从每个出口（崖边、湖、没入地里）往上追，汇水最大的一支接着当本段（干流），别的支各起一段，记下游段与汇入点；
// - 沿程每点：年均流量（按格的 Budyko 径流累计）、平岸流量（= 年均 × 所在流域逐日径流指数的年最大）、平岸宽与深（C1 的河道）、河面与河床、
//   比降、河床质 D50（平岸 Shields 数：砾床 0.05、算出来 < 2 mm 的落到砂床 1.0；乘上游岩性的粗细）、平面型（Kleinhans & van den Berg 2011：
//   潜在比河流功率 ω = ρg·Q_bf·S / (4.7·√Q_bf) 对 900 / 90 × D50^0.42）、年均悬沙（Syvitski & Milliman 2007 的 BQART，再按上游的林与已垦）、
//   左右谷底宽（沿垂直流向量到谷坡脚，夹 C2 的谷底宽）、限制度；
// - 瀑布与跌水：崖边的出口一律记（落差 = 河面 − 岛底）；岛内相邻两点河床陡落、连着几级合起来够落差的记；
// - 逐日径流指数：每个河网出口独立流域（无河网坡面仅入水账）：第 year 年的逐日天气 → 按高程分带积雪、度日融雪 →
//   按 Budyko 的年径流分到各日 → 快流（地表，时间常数随流域面积）与基流（按岩性的基流比例与退水常数，加凝结水）两个线性水库，同一年跑两遍取第二遍。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

// 天气之后调（有聚落就在聚落之后：悬沙的人的项读已垦）；结果放进 g.rnet
void build_rivernet(Group& g, const Config& c);

// 有限汇流窗的事件径流指数。周期年边界；无雨雪输入输出全零，非零输入的指数年均为 1。
std::vector<double> event_runoff_index(const std::vector<double>& liquid, double quick_days);

}  // namespace skyisle::island
