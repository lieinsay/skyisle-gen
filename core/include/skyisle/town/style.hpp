// 风格（docs/PLAN-TOWN.md 第六节）：前端把 base ← 风格 ← --set 展平成 "style.*" 键，这里一次读成带类型的结构，各层只认 Style。
// 模板（compound.template.<i>）与功能（func.<i>）由前端整理成数组；名字（中文）原样带着，C++ 源码里不写中文。
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "skyisle/config.hpp"
#include "skyisle/rng.hpp"

namespace skyisle::town {

struct Range {
    double lo = 0.0, hi = 0.0;
    double mid() const { return 0.5 * (lo + hi); }
    double sample(Rng& r) const { return hi > lo ? r.uniform(lo, hi) : lo; }
    int sample_int(Rng& r, bool odd = false) const;   // [lo, hi] 里的整数（odd：只取奇数，没有就取最近的奇数）
};

// 地块的边 / 一栋房贴着哪条边
enum Side : int { SIDE_FRONT = 0, SIDE_BACK = 1, SIDE_LEFT = 2, SIDE_RIGHT = 3, SIDE_NONE = -1 };
enum Face : int { FACE_FRONT = 0, FACE_IN = 1, FACE_OUT = 2 };

struct OrientRule {
    std::string kind;          // sun / wind / street / gable_street / align_street / water / downslope / fixed
    double weight = 1.0, tol = 0.0, angle = 0.0;   // tol、angle（fixed 的方位角）弧度
};

struct BuildingSpec {
    std::string role, name, func, roof, material;
    std::string attach;        // left / right：贴在同一边前一栋的左 / 右头
    int side = SIDE_BACK;
    Range bays;                // bays.hi > 0：面阔 = 间数 × 开间
    bool odd = false;
    Range len_frac;            // 否则占这条边空余长度的比例
    Range depth_m;
    double pos = 0.5;          // 0 = 左 / 前端，1 = 右 / 后端
    int face = FACE_IN;
    double prob = 1.0;
    int storeys = 1;
    double eave_m = 3.0, pitch_deg = 30.0, plinth_m = 0.3;
    Range detached_m;          // hi > 0：院外单栋，离地块边这么远（北欧的浴房、铁匠房按防火规矩离开主院）
    std::string near;          // 院外单栋挑哪里：water 近水（桑拿在湖岸）/ 空 = 离院近
};

struct TemplateSpec {
    std::string id, name, kind;   // kind：house / public
    double weight = 0.0;
    Range plot_w, plot_d, bay_m;
    bool wall = true, gate_house = false;
    std::string gate;             // 非空时覆盖风格的门规则（left / right / center）
    std::vector<BuildingSpec> b;
    std::vector<std::string> ops; // 只给这些形态算子用（空 = 都行）：靠崖窑只在等高线上、町家只在街村
    // 形状：yard 院落（各栋贴边）/ ring 圆楼 / square_ring 方楼 / weilong 围龙屋（围合单体，户 = 竖向一列房间）
    std::string shape = "yard";
    bool dug_in = false;          // 后排挖进坡里（靠崖窑）：不查后排的挖填，台基按院子
    bool sunken = false;          // 下沉式（地坑院）：院心是坑，四壁是窑
    double garden_frac = 0.0;     // 地块后部这么多是园（英格兰 croft、菜园）
    Range ring_room_m, ring_depth_m, ring_storeys;   // 围合单体：每间弧长、房进深、层数
    bool allows(const std::string& op) const;
};

// 功能（functions.toml ← 风格覆盖）：通用字段 + 各 mode 自己的数、区间、串
struct FuncSpec {
    std::string id, name, mode, tmpl;
    int count_base = 0, count_min_hh = 0, count_max = 1;
    double count_per_hh = 0.0;
    bool required = false;
    std::vector<std::pair<std::string, double>> site;   // 选址谓词权重
    std::map<std::string, double> num;
    std::map<std::string, Range> range;
    std::map<std::string, std::string> str;
    std::vector<std::string> ops;   // 只在这些形态算子下修（空 = 都修）：環濠只围環濠集落
    bool allows(const std::string& op) const;
    double getn(const std::string& k, double fb) const;
    Range getr(const std::string& k, Range fb) const;
    std::string gets(const std::string& k, const std::string& fb) const;
    int count(int households) const;
};

struct Style {
    std::string id;
    // 选址
    double center_search_m = 300.0, water_scale_m = 200.0, edge_scale_m = 80.0;
    double w_flat = 1, w_water = 1, w_dry = 1, w_sun = 0.5, w_farmland = -0.3, w_edge = -1, w_shelter = 0;
    // 地面
    double max_slope_deg = 10.0, max_cut_m = 1.5, max_terrace_m = 2.5;
    bool allow_flood = false;
    // 村
    std::vector<std::pair<std::string, double>> operators;
    Range hh_per_compound, founders;
    double kin_p = 0.6;
    // 路
    Range trunk_w, main_w, street_w, lane_w, path_w, plot_gap, setback;
    double max_grade = 0.12, bridge_max_m = 24.0;
    int exits = 3;
    // 鱼骨
    Range fb_street_spacing, fb_lane_spacing, fb_lane_extend;
    double fb_warp_m = 3, fb_warp_wl = 300, fb_extent = 1.8, fb_vacant_p = 0.05, fb_lane_extend_p = 0.4;
    bool fb_main_cross = true;
    // 团块
    int og_candidates = 24;
    Range og_gap;
    double og_jitter = 0.07, og_stagger_m = 1.5, og_lane_reach_m = 40.0;
    bool og_main_street = true;
    // 生长评分
    double g_dist = 1, g_net = 1, g_interest = 0.6, g_kin = 0.8, g_contact = 0.3, g_noise = 0.15, g_main_discount = 0.7;
    // 朝向与门
    std::vector<OrientRule> rules;
    double snap = 0.0, sun_offset = 0.0;
    std::string gate_front = "left", gate_back = "right", gate_side = "front";
    Range gate_w;
    // 宅院
    double wall_thickness_m = 0.4, yard_min_m = 4.0;
    std::vector<TemplateSpec> templates;
    std::vector<FuncSpec> funcs;
    // 目标（TP-style）
    std::map<std::string, Range> targets;
    std::map<std::string, std::map<std::string, Range>> op_targets;   // [targets.<算子>]：子预设各自的目标（盖过通用的）
    // 各形态算子自己的参数（[street_village] [hufen] [waterfront] [dispersed] [green] [comb] [contour] [enclosure] 与 [organic] 的其余键），
    // 按 "green.width_m" 这样的键取；base.toml 给全了默认值，取不到就抛异常
    std::map<std::string, double> op_num;
    std::map<std::string, Range> op_range;
    std::map<std::string, std::string> op_str;
    std::map<std::string, int> op_max_hh;              // 某个算子只在户数 ≤ 它时用（列村、环村是小村）
    std::map<std::string, double> op_flat_deg, op_flat_share;   // 某个算子要村心一带坡 ≤ flat_deg 的地占到 flat_share 以上（地坑院要平塬）
    double pn(const std::string& k) const;
    Range pr(const std::string& k) const;
    std::string ps(const std::string& k, const std::string& fb = "") const;

    int template_index(const std::string& id) const;   // 没有返回 −1
    const FuncSpec* func(const std::string& id) const; // 没启用返回 nullptr
    std::vector<int> house_templates() const;
};

Style parse_style(const Config& c);   // 读 "style.*"；缺键抛异常（base.toml 给全了默认值）

}  // namespace skyisle::town
