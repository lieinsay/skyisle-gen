// 泊场、中转站、镇、航船、邑治（P7，Zhouzhu PLAN-LAND L24–L26；skyisle_gen/island/market.py 逐位同式，DESIGN-NOTES 四点三十八）。
// settle.cpp 在专业聚落之后调：harbor_pad / harbor_count / harbor_sites（泊场先于镇）→ build_relays（中转站，户从非农户里出）→
// build_towns（镇：本岛走路 + 跨岛只算航船；航船线；邑治：航船汇得最多、靠大泊场）。
// 记录里的字符串是代码（功能 beacon / gate / inn / wait / shelter / transship，住法 resident / rotation，范围 inner / outer / both，
// 角色 gate / post / inn / repair / porter，赶集 walk / boat），中文在前端 decode.py 映射。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "skyisle/island/types.hpp"

namespace skyisle::island {

struct MarketVillage {                  // 村（settle 的 villages，按村号次序）；km 是 JSON 里那个（round 3）
    int id = 0, island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0;
    int64_t households = 0;
};

struct Harbor {                         // 大泊场（harbors）
    int id = 0, island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0, area_km2 = 0, elev_m = 0;
    int64_t ships = 0;
    int town = 0;                       // 挨着它的镇号（0 = 没有）
};

struct Relay {                          // 中转站（relays）
    int id = 0, island = 0, ci = 0, cj = 0, li = 0, lj = 0;
    double kx = 0, ky = 0, elev_m = 0, lkx = 0, lky = 0, lookout_m = 0, flow = 0;
    std::vector<std::string> funcs;     // 代码，按 beacon / gate / inn / wait / shelter / transship
    std::vector<std::pair<std::string, int64_t>> roles;   // 角色代码 → 户（> 0 的，按 gate / post / inn / repair / porter）
    int64_t households = 0, ships = 0;
    std::vector<RouteEdge> edges;
    int landing = 0;
};

struct BoatLine {
    int id = 0, town = 0;
    std::vector<int> stops;             // 村号，船走的次序
    std::vector<int> islands;           // 升序去重
    int64_t households = 0;
    std::vector<std::array<double, 2>> pts;   // km（已 round 3），最后一点是镇的泊场
    double length_km = 0, cost_km = 0, hours = 0;
    int interval_days = 1;
};

struct Town {
    int id = 0, village = 0, island = 0, ci = 0, cj = 0;
    double kx = 0, ky = 0;
    int64_t households_farm = 0, households_market = 0;
    int64_t served_villages = 0, served = 0, served_walk = 0, served_boat = 0, served_other = 0;
    double max_served_km = 0, score = 0, seat_score = 0;
    std::vector<int> lines;
    bool seat = false;
    int harbor = 0;                     // 大泊场号（0 = 没有）
    int64_t harbor_ships = 0;
    double harbor_dist_km = 0, street_len_km = 0, street_bearing_deg = 0;
    int landing = 0;
};

struct TownsResult {
    std::vector<Town> towns;            // 邑治在前（id 1），其余按挑的先后
    std::vector<BoatLine> lines;
    int seat = -1;                      // villages 的下标
    // 每个村：归哪个镇（镇号）、怎么去（0 走路 / 1 航船）、路程（km，航船按风折过）、搭哪条航船（0 = 走路）、是哪个镇（0 = 不是）与它的市户
    std::vector<int> market_town, mode, boat_line, town;
    std::vector<double> market_km;
    std::vector<int64_t> households_market;
};

// lflat：能落船的平地（与村的船台同口径）；pad：大块泊场的格（lflat 再去掉溪涧、湿地、漫滩、在种与撂荒的田，开运算）
void harbor_pad(const Group& g, const Config& c, Mask& lflat, Mask& pad);
// 每座岛的格（扁平下标，升序）
std::vector<std::vector<int32_t>> island_cells(const Group& g);
// 每个陆地格：以它为心 (2r+1)² 方窗里同岛的 pad 格数
GridI harbor_count(const Mask& pad, const Group& g, const std::vector<std::vector<int32_t>>& cells, int r);
std::vector<Harbor> harbor_sites(const Group& g, const Config& c, const Mask& pad, const GridI& cnt);
int64_t harbor_ships(int64_t n_cells, double cell_km2, const Config& c);
int64_t landing_ships(const Mask& lflat, const Group& g, int i, int j, const Config& c);
// 中转站：farm / busy 是每座岛有没有村或散户 / 有没有村、散户、专业聚落、废村；free 是站址能落的格；expo 是迎风性（settle 的村址评分用的那个）
std::vector<Relay> build_relays(const Group& g, const Config& c, const std::vector<uint8_t>& farm, const std::vector<uint8_t>& busy,
                                const Mask& lflat, const Mask& free, const GridI& cnt, const std::vector<double>& expo,
                                const std::vector<std::vector<int32_t>>& cells, int64_t nonfarm_left);
TownsResult build_towns(const std::vector<MarketVillage>& villages, std::vector<Harbor>& harbors, const Config& c, int64_t rest_hh,
                        double wind_u, double wind_v);

}  // namespace skyisle::island
