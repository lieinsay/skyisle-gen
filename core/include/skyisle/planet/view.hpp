// 行星层 → 第三层（P6c）：岛群生成器要读的行星层数据（PlanetView、NodeInputs）直接从 ①–④ 的产物结构给，不经 npz。
// 与 Python 前端 island/engine.planet_view、island._node_inputs + engine.inputs 同值（npz 里 float32 的字段先按 float32 舍）。
// 人口与邦都（⑨）不在这里：P6d 之前仍由前端从 polity.npz 填（NodeInputs.pop / is_capital …）。
#pragma once

#include "skyisle/island/types.hpp"
#include "skyisle/planet/planet.hpp"

namespace skyisle::planet {

// cfg：展平的行星层配置（s04.climate.season_tau_land_days 等）
island::PlanetView planet_view(const Planet& p, const Islands& isl, const Climate& c, const Config& cfg);
island::NodeInputs node_inputs(const Islands& isl, const Climate& c, int64_t node, uint64_t seed, const Config& cfg);
island::Calendar island_calendar(const Planet& p);   // island/climate.calendar(planet)
double median_f32(const std::vector<double>& v);     // np.median(float32 数组)：偶数个时 float32 相加再 / 2

}  // namespace skyisle::planet
