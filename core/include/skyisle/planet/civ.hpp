// 行星层 ⑤–⑨（行星计划 P6d）：skyisle_gen/stages/s05–s09 与 graph / weights 的 C++ 版。
//
// ⑤ 障碍（Φ 穿越归一化、边局部因子、G 的大圆弧段阻断）→ ⑥ 航路（有向成本、按集雨容量加权的抽样介数、枢纽、连通分量）
// → ⑦ 文明中心（适宜度、骨架窗内涌现、次级极大、史前扩散与谱系、⑦b 地区、中心间干线）
// → ⑧ 特征扩散（特征表、隔离度、reach、同槽位对称不动点）→ ⑨ 政治层（人口、诸邦、采邑、船团与部落、宗主、附庸、变法与兼并史、开局候选）。
//
// 口径同 planet.hpp（P6c）：产物结构与 npz / json 同名同形，浮点存双精度原值；npz 里存 float32 的字段由读它的一方先 f32()。
// json 里的记录（名字、中文、round 的位数）由 Python 前端拼（各步的 `_write` 两个后端共用），这里只给拼它要的原始数；
// 但凡 Python 版在后面的计算里读的是**已舍过的**数（⑧ 特征表的 resistance / lambda），这里也按 Python 的 round 舍好再用。
// 字符串只用 ASCII（槽位 id、模式、中心 id 都是 ASCII）。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/grid.hpp"
#include "skyisle/island/types.hpp"
#include "skyisle/planet/graph.hpp"
#include "skyisle/planet/planet.hpp"

namespace skyisle::planet {

constexpr int N_MODES = 4;
extern const char* const MODE_KEYS[N_MODES];          // daily trade envoy migrate（skyisle_gen.MODES）
constexpr int M_DAILY = 0, M_TRADE = 1, M_ENVOY = 2, M_MIGRATE = 3;
constexpr int N_REGIONAL = 6;
extern const char* const REGIONAL_ORDER[N_REGIONAL];   // A B C D F_N F_S（G 单独处理）
extern const char* const CENTER_KEYS[3];               // north_west north_east south（s07.CENTER_IDS）
int mode_index(const std::string& m);

// ---------------------------------------------------------------- ⑤ 障碍（s05_barriers.py）
struct Barriers {                 // perm.npz（perm / perm_no_g 存 float64，其余浮点 float32）
    int64_t E = 0;
    std::vector<double> perm, perm_no_g;                       // [E, 4]
    std::vector<double> f;                                     // f_regional [E, 6]
    std::vector<uint8_t> g_blocked;                            // [E]
    std::vector<double> gap, density_drop, climb, political;   // [E, 4]
};
Barriers stage5(const Config& cfg, const Planet& p, const Winds& w, const Islands& isl, const Climate& c);

// ---------------------------------------------------------------- ⑥ 航路（s06_routes.py）
struct Routes {                   // routes.npz（flow / node_flow 存 float32）+ hubs.json
    std::vector<double> cost, cost_no_g;                       // [2E]：前 E 条 a→b，后 E 条 b→a
    std::vector<double> cost_m;                                // [2E, 4]
    std::vector<double> flow, node_flow;                       // [2E]、[N]
    std::vector<int64_t> src_d, dst_d, und_id, sources;
    std::vector<int64_t> hubs;                                 // 按 (−node_flow, 节点号) 排（⑦ 的地区种子取前 12 个）
    std::vector<uint8_t> near_g;                               // [N]：G 邻域
    int64_t n_sources = 0;
    std::array<int64_t, N_MODES> n_components{}, largest{};    // 每模式的弱连通分量
};
Routes stage6(const Config& cfg, uint64_t seed, const Winds& w, const Islands& isl, const Climate& c, const Barriers& b, int threads = 1);

// ⑥ 之后各步共用的有向图（weights.load_directed）：perm_d = 通过率两份拼起、L = −ln perm（perm = 0 → inf）
struct Directed {
    int64_t N = 0, E = 0;
    CSR csr;
    std::vector<double> perm_d, L;                             // [2E, 4]
};
Directed directed(const Islands& isl, const Barriers& b, const Routes& r);
std::array<double, N_MODES> lambda_ref(const Config& cfg);     // weights.lambda_ref：ln2 / √(d_lo · d_hi)
std::vector<double> mode_weight(const Routes& r, const Directed& g, int mi, double lam);   // λ·cost_m[:, m] + L[:, m]

// ---------------------------------------------------------------- ⑦ 文明中心（s07_centers.py）
struct Trunk {                    // 中心间干线（centers.json 的 center_trunks）
    std::string key;              // "north_west->south" …
    bool reachable = false;
    double cost_days = 0;
    std::vector<int64_t> path;
};
struct Centers {                  // centers.json + prehist.npz（arrival_yr float32）+ regions.npz（suitability float32）
    std::array<int64_t, 3> node{};                             // CENTER_KEYS 的次序
    int64_t origin_node = -1;
    std::vector<int64_t> secondary;                            // 次级极大（选中的次序）
    std::vector<int64_t> region_seeds;
    std::vector<Trunk> trunks;
    std::vector<double> suit;                                  // 适宜度（双精度原值；centers.json 按它 round）
    std::vector<double> dist_pre, arrival_yr;
    std::vector<int64_t> pred_pre, lineage, region;
    int64_t n_lineages = 0;
};
Centers stage7(const Config& cfg, const Planet& p, const Islands& isl, const Climate& c, const Barriers& b, const Routes& r);

// ---------------------------------------------------------------- ⑧ 特征扩散（s08_diffusion.py）
struct Trait {                    // traits.resolved.json 的一条（数已按 Python 的 round 舍好：后面的计算读的就是舍过的）
    std::string id, slot, mode, origin, kind, tier;            // kind：main / sub / local / manual
    int slot_index = 0;                                        // 在配置 slots.slot 里的序号
    double resistance = 0, d_half_days = 0, lambda = 0, origin_time = 0;
    int64_t origin_node = 0;
};
struct FixedPoint {
    int n_iter = 0;
    bool converged = false;
    int n_values = 0;
};
struct Diffusion {                // traits.resolved.json、fields.npz、iso.npz（浮点存 float32）、reflect.json
    std::vector<Trait> traits;
    std::vector<std::string> slots;                            // 有特征的槽位 id，排序后
    std::vector<std::string> slot_mode;                        // 与 slots 对齐
    std::array<double, N_MODES> lambda_ref{};
    std::vector<FixedPoint> fixed_point;                       // 与 slots 对齐
    bool has_reflect = true;                                   // 手工特征表（traits.toml）时不写 reflect.json
    double iso_thr = 0;
    std::vector<std::vector<int64_t>> reflect;                 // 高隔离连通分量（反射型）
    int64_t N = 0;
    std::vector<double> reach, strength, adopt, share, C, L;   // [T, N]
    std::vector<int32_t> conflict_by;                          // [T, N]
    std::vector<double> iso;                                   // [4, N]
    std::vector<double> local_share;                           // [S, N]
};
Diffusion stage8(const Config& cfg, uint64_t seed, const Islands& isl, const Barriers& b, const Routes& r, const Centers& ce);

// ---------------------------------------------------------------- ⑨ 政治层（s09_polity.py）
constexpr int KIND_STATE = 0, KIND_FLEET = 1, KIND_TRIBE = 2;
enum Regime : int { REG_REFORMED = 0, REG_SUZERAIN, REG_CENTRALIZABLE, REG_FEUDAL, REG_CITY };
extern const char* const REGIME_KEYS[5];                       // reformed suzerain centralizable feudal city
extern const char* const STAGE_KEYS[6];                        // military administrative economic cultural linguistic identity
struct StateRec {                 // polities.json 里 kind = state 的一条（名字、中文、round 由前端拼）
    int64_t capital = -1;
    int circle = 0;
    int regime = REG_FEUDAL;
    int64_t n_nodes = 0, n_direct = 0, n_fiefs = 0;
    double pop = 0, dense_frac = 0, radius = 0;
    int64_t overlord = -1, annexed_by = -1;
    double annexed_years = NaN;                                // NaN = 未被兼并
    bool is_suzerain = false;
    std::vector<int64_t> vassals;
    std::vector<std::pair<int64_t, int64_t>> neighbors;        // (邻邦, 界边数)，按邻邦号排
};
struct Annex {                    // 兼并纪年一条（years_ago / war_years 是 numpy 标量：前端按 np.float64 舍）
    int64_t polity = -1;
    double years_ago = 0, war_years = 0, pop = 0;
    int64_t n_nodes = 0;
    bool was_suzerain = false;
    int stage = 0;
};
struct Front {
    int64_t polity = -1;
    double frontier_days = 0;
    bool feasible = false;
    double war_years_needed = 0;                               // feasible 时才有（numpy 标量）
};
struct Polity {                   // polity.npz（pop / control / dist_cap / pop_state 存 float32）+ polities.json + history.md 要的数
    std::vector<double> pop, control, dist_cap, pop_state;
    std::vector<int32_t> state, polity, fief, realm, capital;
    std::vector<int8_t> kind, circle;
    std::vector<StateRec> states;
    std::vector<std::vector<int64_t>> fleets;
    std::vector<double> fleet_pop;
    std::vector<int> fleet_circle;
    std::vector<int64_t> tribes;
    std::array<int64_t, 3> suzerain{};                         // CENTER_KEYS 的次序；−1 = 中心不在邦里
    int64_t reformer = -1;
    bool reformer_fallback = false;
    std::string reform_circle;
    double realm_pop = 0;
    std::vector<Annex> history;
    std::vector<Front> fronts;
    std::vector<int64_t> open_a, open_c;
    int64_t open_b = -1, open_d = -1;
    int64_t n_cap1 = 0, n_attached = 0;
};
std::vector<double> population(const Config& cfg, const Islands& isl, const Climate& c);   // pop = P1 × 可耕地 × 降水折减
Polity stage9(const Config& cfg, const Islands& isl, const Climate& c, const Barriers& b, const Routes& r, const Centers& ce);
// 第三层的 NodeInputs 补上 ⑨：本群人口（float32 的值）、是不是某邦的都（邦人口 = float32 成对求和）、是不是变法之国
void apply_polity(const Polity& pol, int64_t node, const Config& cfg, island::NodeInputs& inp);
// 第三层的 NodeInputs 补上 ⑥（P7 的中转站）：本群的邻边（按 ③ 的边号升序：邻群、方位、离开几天、有向成本与流量（按 float32 存盘的值）、邻群是不是枢纽）
// 与本群是不是 ⑥ 的枢纽。与 skyisle_gen/island/market.node_routes（从 npz 读）同值
void apply_routes(const Islands& isl, const Routes& r, int64_t node, island::NodeInputs& inp);

// ---------------------------------------------------------------- ①–⑨ 一次跑完（游戏新建世界：planet::run 之后接着算）
struct Society {
    Barriers barriers;
    Routes routes;
    Centers centers;
    Diffusion diffusion;
    Polity polity;
};
// upto：5–9（⑧ 可跳过：skip_diffusion = true 时 ⑨ 照样算，⑧ 留空——游戏只要人口与邦时省下 ⑧）
Society run_society(const Config& cfg, uint64_t seed, const World& w, int upto = 9, bool skip_diffusion = false, int threads = 1);

}  // namespace skyisle::planet
