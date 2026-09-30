// 5.4 四季（climate.py）与 5.5 逐日天气（weather.py）同式。
#pragma once

#include "skyisle/json.hpp"
#include "skyisle/island/types.hpp"

namespace skyisle::island {

// 四季：降水按 ④ 的份额、风暴 / 窗口 / 风按每季带界位移取样 → 缩放到年均；温度 = 年均 + 半振幅 × cos(相位 − 滞后)；季型分类（只用行星层标量与网格，不看地形）
Climate build_climate(const NodeInputs& inp, const PlanetView& pv, const Config& c);
// 台面（inp.height_m）处各季季中的气温（build_climate 的 t_isl 同式）与季中的昼长（小时，按赤纬 = 倾角 × cos(季相)）：Budyko 的潜在蒸散用
struct SeasonThermal {
    std::vector<double> t_ref, day_hr;
    double days_per_season = 84.0;
    int months_per_season = 3;
};
SeasonThermal season_thermal(const NodeInputs& inp, const PlanetView& pv);
// 一年逐日的季节曲线（温度按余弦；降水 / 风暴 / 窗口 / 风按季中值做周期插值 + 滑动平均）
Daily daily_curves(const Climate& clim, const NodeInputs& inp);
std::vector<SeasonParams> season_params(const Climate& clim, const Config& c);
WeatherYear simulate_year(Rng& rng, const Climate& clim, const Daily& daily, const std::vector<SeasonParams>& params, double surface_m,
                          const Config& c, double rim_m, double lapse_c_per_km);
// 多年样本（IS-daily）：P[年, 季] = 季降水和，F[年, 季] = 季雨日比例；年份 y 用 weather:{y} 流
void multi_year(const NodeInputs& inp, const Climate& clim, const Daily& daily, const std::vector<SeasonParams>& params, double rim_m,
                const Config& c, int years, std::vector<double>& P, std::vector<double>& F);

Json climate_json(const Climate& clim);

}  // namespace skyisle::island
