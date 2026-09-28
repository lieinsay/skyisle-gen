// 编排层内部（不对外）：plan_site 持有的「正在长的方案」与各步共用的小件。
// 形态算子、功能、宅院、校验都拿 Work 的引用干活——状态只有这一份，都在编排层里。
#pragma once

#include <string>
#include <vector>

#include "skyisle/rng.hpp"
#include "skyisle/town/analysis.hpp"
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
int pick_template(const Work& w, Rng& r);                      // 按权重挑住宅模板
// 临路摆一块地：路上一点 p、切向 t、朝外的法向 n_out、路半宽；按朝向链定朝向，贴路摆，给出接路边
Slot slot_along(const Work& w, V2 p, V2 t, V2 n_out, double road_hw, int tmpl, double W, double D, double setback, double force_facing = NaN);
int access_side_of(const Obb& box, V2 to_road);                // 地块哪一边朝着路
int commit_compound(Work& w, const Slot& sl, const std::string& kind, const std::string& func, const std::string& name,
                    const std::vector<int>& households);
int add_road(Work& w, const std::vector<V2>& line, int cls, double width);   // 进 plan.roads、盖占地、记桥
double road_width(const Work& w, int cls, Rng& r);
void mark_obb(Work& w, const Obb& o, uint8_t code, double shrink);
double dist_obb(const Obb& o, V2 p);                              // 点到有向矩形的距离（里面为 0）
bool road_hits_obb(const Work& w, const Obb& o, double tol);      // 有没有哪条路（按真宽）压到这个矩形（tol < 0：还要离路边 −tol）
double road_fit(const Work& w, const std::vector<V2>& line);       // 这条线离地块、院外单栋房最近处容得下的路宽
bool obb_free(const Work& w, const Obb& o, double shrink, bool need_buildable, double max_slope_deg);

// ---------------------------------------------------------------- 各步
void op_fishbone(Work& w);        // op_fishbone.cpp：候选街网 + 地块 → 公共宅院占位 → 一户一户落位 → 用到的路
void op_organic(Work& w);         // op_organic.cpp：主街 → 一户一户挨着长、巷随门长
void op_single(Work& w);          // 宅院规模：兴趣最高处一个宅院
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
void instantiate_compound(Work& w, int ci, Rng& r);   // compound.cpp：模板 → 各栋、门、台基与土方
void verify(Work& w);             // verify.cpp：TP-* 与形态指标

}  // namespace skyisle::town
