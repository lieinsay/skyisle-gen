// 整群生成（island.generate 的算法部分）与粗版降采样（lod._block_reduce）。写产物（npz / json / png / 预览图）在前端。
#pragma once

#include "skyisle/island/types.hpp"

namespace skyisle::island {

// steps：1 地形；2 + 水系与资源；3 + 四季与逐日曲线；4 + 逐日天气（year 年）；5 + 聚落（与 Python 版 generate 的 steps 同义）
Group generate(const NodeInputs& inp, const PlanetView& pv, const Config& c, int year = 0, int steps = 5, double res_m = 0.0, int threads = 1);

// 粗版：按 f × f 块降采样（右 / 下边补虚空到 f 的整数倍）
struct LodBlock {
    int H = 0, W = 0, f = 1;
    std::vector<uint8_t> land, landcover, water;   // 陆地占比 0–255、众数地表类、河湖占比 0–255
    std::vector<float> height, peak;               // 块内陆地平均高 / 最高（无陆地 NaN）
    std::vector<int16_t> island;                   // 块内陆地最多的岛号（−1 = 虚空）
};
LodBlock block_reduce(const Group& g, int f);

}  // namespace skyisle::island
