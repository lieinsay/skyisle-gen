// 宜垦 / 已垦 / 撂荒（P5，Zhouzhu PLAN-LAND L7 / L10 / L21；skyisle_gen/island/farmland.py 同式，DESIGN-NOTES 四点三十六）。
// hydro 在上等地（g.arable，行星层额度内最好的宜垦地）取完、还没画进地表之前调 cultivable_land；聚落层调 fill_cultivated：
// 好地先占 + 定居门槛 → g.cultivated / g.fallow_years，改 g.landcover 与林场的 patch_id。废村的村址、专业聚落的住法、有人用的岛、
// 荒地归谁要用聚落层的村与专业聚落，在 settle.cpp 里。
#pragma once

#include <cstdint>
#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

// 宜垦：suit > 0、坡 < cultivable_slope_max_deg、年均温 + 半个季节温差 ≥ cultivable_summer_min_c、土层 ≥ cultivable_soil_min、
// 湿度 ≥ cultivable_wet_min 或近水；上等地一定宜垦；坡 ≥ terrace_slope_deg 记 2。另存原本的地表与适宜度
void cultivable_land(Group& g, const Config& c, const std::vector<double>& suit, const std::vector<double>& T,
                     const std::vector<double>& soil, const std::vector<double>& wet, const std::vector<double>& near_water);

struct RuinTract {                    // 废村：让出来的宜垦片（村址在 settle 里挑）
    int tract = 0, island = 0, years = 0;
    int64_t households_before = 0;
    std::vector<int32_t> cells;       // 头一轮分到的格（扁平下标，按好地先占的次序）
    double fallow_km2 = 0;
};

struct FillResult {
    std::vector<RuinTract> ruins;
    Json summary;
};

// water：河 / 湖 / 溪涧（settle 的 water）；land_per_hh：户均地量 km²；n_quota：已垦的额度（上等地的格数）。随机流 settle:fallow
FillResult fill_cultivated(Group& g, const Config& c, Rng& rng, const Mask& water, double land_per_hh, int64_t n_quota);

}  // namespace skyisle::island
