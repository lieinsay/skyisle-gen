// 编排层内部（不对外）：plan_site 持有的「正在长的方案」与各步共用的小件。
// 形态算子、功能、宅院、校验都拿 Work 的引用干活——状态只有这一份，都在编排层里。
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "skyisle/rng.hpp"
#include "skyisle/town/analysis.hpp"
#include "skyisle/town/contour.hpp"
#include "skyisle/town/network.hpp"
#include "skyisle/town/orient.hpp"
#include "skyisle/town/plan.hpp"

namespace skyisle::town {

// 候选地块：临一条路（或一条待修的巷）的一块宅基
struct Slot {
    Obb box;
    int tmpl = -1;
    int edge = -1;              // 临的路网边（团块生长：−1）
    double s0 = 0.0, s1 = 0.0;  // 在边上占的弧长区间
    double s_acc = 0.0;         // 接路点的弧长
    int access_side = SIDE_FRONT;
    V2 access;                  // 接路点（路中线上）
    double interest = 0.0, base_m = 0.0, max_cut = 0.0;
    bool valid = false, vacant = false, taken = false;
    int comp = -1;              // 落了哪个宅院（成排的「贴着」按它算）
    int cls = RC_LANE;          // 临的路的等级
    double noise = 0.0, d_net = INF;
    double half_t = 0.0;        // 沿路方向的半长（摆下一块用）
    std::vector<int> conflicts, touch;
};

struct Work {
    const Site& s;
    const Style& st;
    const PlanRequest& req;
    Fields f;
    GridF interest;
    Plan plan;
    Network net;
    Mask road_mask;     // 细栅格上的路（寻路打折、地块避让）
    Mask road_core;     // 路中线那一两格：新路接到这里（接在路边上会贴着墙）
    Mask blocked;       // 寻路不可走：地块、房、塘、场、泊场
    Grid<int32_t> water_src;   // 每格最近的水格（扁平下标；没有水 −1）：朝水的朝向用
    std::vector<Slot> slots;
    V2 center;
    double facing = 0.0, R = 100.0, plot_area = 400.0;
    int n_compounds = 0;
    std::vector<int> hh_order;          // 落位次序（户号）
    std::vector<std::vector<int>> hh_groups;   // 每个宅院住哪几户（按次序）
    PathParams pp;
    int root = -1;                      // 路网里村心的节点
    int entrance_road = -1;             // 最要紧的出村大路（plan.roads 的下标）
    V2 entrance;                        // 它离开建成区的点
    bool has_entrance = false;

    Work(const Site& s_, const Style& st_, const PlanRequest& r_) : s(s_), st(st_), req(r_) {}
    Rng rng(const std::string& tag) const;   // 各步独立的随机流：改一步不牵动别的
};

// ---------------------------------------------------------------- 共用小件（plan.cpp）
double sample_interest(const Work& w, const Obb& o);           // 地块里兴趣图的均值（不可建格算 0）
// 地块的地面：台基（中位高程）、最大挖填深、全可建？（extra 非空时这些格也不行）
bool slot_ground(const Work& w, Slot& sl, const Mask* extra);
OrientCtx orient_ctx(const Work& w, V2 p);                     // 某处的语境（朝阳、风、下坡、最近的水）
int pick_template(const Work& w, Rng& r);                      // 按权重挑住宅模板（只挑本村形态算子能用的）
// 临路摆一块地：路上一点 p、切向 t、朝外的法向 n_out、路半宽；按朝向链定朝向，贴路摆，给出接路边
Slot slot_along(const Work& w, V2 p, V2 t, V2 n_out, double road_hw, int tmpl, double W, double D, double setback, double force_facing = NaN);
int access_side_of(const Obb& box, V2 to_road);                // 地块哪一边朝着路
int commit_compound(Work& w, const Slot& sl, const std::string& kind, const std::string& func, const std::string& name,
                    const std::vector<int>& households);
int add_road(Work& w, const std::vector<V2>& line, int cls, double width);   // 进 plan.roads、盖占地、记桥
double road_width(const Work& w, int cls, Rng& r);
bool hits_built(const Work& w, const Obb& o, double clearance);
void mark_obb(Work& w, const Obb& o, uint8_t code, double shrink);
double dist_obb(const Obb& o, V2 p);                              // 点到有向矩形的距离（里面为 0）
bool road_hits_obb(const Work& w, const Obb& o, double tol);      // 有没有哪条路（按真宽）压到这个矩形（tol < 0：还要离路边 −tol）
double road_fit(const Work& w, const std::vector<V2>& line);       // 这条线离地块、院外单栋房最近处容得下的路宽
bool obb_free(const Work& w, const Obb& o, double shrink, bool need_buildable, double max_slope_deg);
// 地块能落吗：占地全空（extra 非空时这些格也不行）、与附近地块按几何不交、不压路、地面可建且挖填在上限内（会写 sl 的台基与兴趣）
bool plot_fits(Work& w, Slot& sl, const Mask* extra = nullptr);
double total_slope_share(const Work& w, V2 c, double radius, double min_deg);   // c 周围 radius 内坡 ≥ min_deg 的格占多少

// ---------------------------------------------------------------- 各算子共用（op_common.cpp）
// 沿一条路的一侧切宅基：从弧长 s0 到 s1，每块按模板抽面宽进深（front_w / depth 给了就用它们），朝向按朝向链（force_facing 给了就用它），
// 地块之间留 gap；返回按弧长排好的、能落的宅基（s_acc = 接路点的弧长）。road_hw = 路半宽
struct SliceOpt {
    double s0 = 0.0, s1 = INF;
    int side = 1;                  // 1 = 路的左边，−1 = 右边（相对折线走向）
    Range front_w, depth;          // hi = 0：按模板
    Range gap;                     // hi = 0：按风格 plot_gap
    double setback = -1.0;         // < 0：按风格
    double force_facing = NaN;
    bool face_road = false;        // 地块一律朝路（街村、滨水）：不看朝向链
    bool face_away = false;        // 地块一律背着这条线朝外（等高线：背靠台线、朝下坡）
    int tmpl = -1;                 // −1：按权重挑
    const Mask* extra = nullptr;   // 这些格也不能压
};
std::vector<Slot> slice_along(Work& w, const std::vector<V2>& line, double road_hw, const SliceOpt& o, Rng& r);
// 按给定次序把一户户落进宅基（公共宅院先由调用方占好）：从第 g0 个宅院组起，落下的宅基标 taken；返回落到第几组
size_t fill_slots(Work& w, std::vector<Slot>& slots, size_t g0);
// 过 c、顺方位角 bearing 的一条路：两头各往外 half_len（退到走得通的地方），按坡度代价 A* 到 c；返回折线（从一头到另一头）
std::vector<V2> through_road(Work& w, V2 c, double bearing, double half_len);
// 宅院的门到路网修一条巷（门外起步、只在路中线上接、按几何量挤就不修）；成功时 access 改成巷的尽头
bool lane_to_gate(Work& w, int ci, double lane_w, double reach_m);
void withdraw_last_compound(Work& w);                          // 撤回最后落的一块（户回到没住处、地块让出来）
V2 gate_point_of(const Work& w, const Obb& plot, int side, const TemplateSpec& T);   // 门在地块哪条边的哪个位置（与宅院成形同一规则）
V2 side_normal(const Obb& o, int side);
// 公共宅院的候选：沿已有路（等级 ≤ max_cls、离 c 在 reach 以内）两侧，按功能的模板定尺寸
std::vector<Slot> public_candidates(Work& w, V2 c, double reach, int max_cls, Rng& r);
// 折线按弧长每 step 取一点（首尾都在）
std::vector<V2> resample(const std::vector<V2>& line, double step);
// 折线上的点按格上的条件切成连续的几段（每段 ≥ min_len）
std::vector<std::vector<V2>> split_runs(const Work& w, const std::vector<V2>& pts, const std::function<bool(int, int)>& ok, double min_len);
// 从 from 按坡度代价修一条路接到路网（goal 为空时接 road_core）；返回 plan.roads 的下标（−1 = 没修成）
int link_road(Work& w, V2 from, int cls, double width, double reach_m, const Mask* goal = nullptr);
// 同上但只找不修：path 从路网到 from，fit 是容得下的路宽
bool plan_link(const Work& w, V2 from, double width, double reach_m, const Mask* goal, std::vector<V2>& path, double& fit);
// 公共宅院直接落在给定的地块上（绿地上的教堂）：功能 id 找不到或不许这个算子用时返回 −1
int commit_public(Work& w, const std::string& func_id, Slot sl);
// 风格要求必须有、形态算子却没落下的公共宅院（坡上的祠堂、教堂）：在村心附近按几个朝向找一块能落的地，门修巷接到路网
void place_missing_public(Work& w);

// ---------------------------------------------------------------- 各步
void op_fishbone(Work& w);        // op_fishbone.cpp：候选街网 + 地块 → 公共宅院占位 → 一户一户落位 → 用到的路
void op_organic(Work& w);         // op_organic.cpp：主街 → 一户一户挨着长、巷随门长
void op_single(Work& w);          // 宅院规模：兴趣最高处一个宅院
void op_street(Work& w, bool hufen);   // op_street.cpp：街村 / 林地排村
void op_waterfront(Work& w);      // op_waterfront.cpp：滨水（前街后河）
void op_dispersed(Work& w);       // op_dispersed.cpp：散居（林盘、北欧农庄、散居村）
void op_green(Work& w);           // op_green.cpp：围绿（绿地村、环村）
void op_comb(Work& w);            // op_comb.cpp：梳式
void op_contour(Work& w);         // op_contour.cpp：等高线（靠崖窑）
void op_enclosure(Work& w);       // op_enclosure.cpp：围合单体（土楼、围龙屋）
bool op_fits(const Work& w, const std::string& op);   // 这块地能用这个算子吗（滨水要河、等高线要坡、小村的算子看户数）
// functions.cpp：选址谓词（某处对某项的 0–1 分）与按功能表挑位
double predicate(const Work& w, const std::string& name, V2 p);
double site_score(const Work& w, const FuncSpec& f, V2 p);
// 公共宅院（庙）：在候选地块里按谓词挑，占住并让冲突的住宅地块作废；返回占了几块
int place_public_compounds(Work& w, std::vector<Slot>& cands);
void organic_fill(Work& w, size_t first_group);   // op_organic.cpp：从第 first_group 个宅院起按团块生长补齐
void place_exits(Work& w);
void place_landing(Work& w);
void place_ponds(Work& w);
void place_roadside(Work& w);     // 土地庙、塘边庙、村口大树
void place_threshing(Work& w);
void place_wells(Work& w);
void place_moat(Work& w);         // 環濠：建成区外廓外一圈水，路过处是桥
void place_extras(Work& w);       // 其余设施：水圳、牌坊、水口、条田、院外单栋以外的杂项
void instantiate_compound(Work& w, int ci, Rng& r);   // compound.cpp：模板 → 各栋、门、台基与土方；院外单栋、园、坑
void instantiate_enclosure(Work& w, int ci, Rng& r);  // compound.cpp：圆楼 / 方楼 / 围龙屋
void verify(Work& w);             // verify.cpp：TP-* 与形态指标

}  // namespace skyisle::town
