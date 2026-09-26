// 聚落（settle.py）与聚落层级（tiers.py）同式：户 → 田块 → 村址 → 专业聚落 / 集镇 / 泊场 → 桥头 / 水设施 → 前哨 / 主家候选 / 都与城 → 村周开垦 → 村的采场。
// 结果放进 g.settle（settlements.json 的形，字符串是代码，中文在前端 decode.py 映射）与两张栅格；开垦改 g.landcover、g.patch_id，采场加进 g.res.works。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

void build_settlements(Group& g, const PlanetView& pv, const Config& c);

// _kmeans_split：把一块田的格按坐标分成 k 份（Lloyd 12 轮，初值 = choice(n, k, replace=False)）；返回每格的份号（测试对照用）
std::vector<int32_t> kmeans_split(Rng& rng, const std::vector<int32_t>& ii, const std::vector<int32_t>& jj, int k, int iters = 12);

}  // namespace skyisle::island
