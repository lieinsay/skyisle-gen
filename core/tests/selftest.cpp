// C++ 自检（ctest）：随机流对 numpy 的参考值、栅格公共件的小例子。不依赖 Python。
// 参考值来自 numpy 2.x：np.random.PCG64(np.random.SeedSequence([42, 21, zlib.crc32(b"island:2051:layout")])).random_raw(3) 等。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "skyisle/flow.hpp"
#include "skyisle/grid.hpp"
#include "skyisle/island/coast.hpp"
#include "skyisle/rng.hpp"
#include "skyisle/town/contour.hpp"
#include "skyisle/town/geom.hpp"
#include "skyisle/town/network.hpp"
#include "skyisle/town/orient.hpp"
#include "skyisle/town/raster.hpp"
#include "skyisle/town/site.hpp"

using namespace skyisle;

#define REF_RAW0 4869714485165228175ULL
#define REF_RAW1 11149298506873232026ULL
#define REF_RAW2 14528220131426250776ULL
#define REF_NORMAL0 0x1.7787311499fd8p-3
#define REF_INT1 6
#define REF_BETA2 0x1.250d91b12be7bp-2

static int fails = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

int main() {
    CHECK(crc32("island:2051:layout") == 4019611777u);
    {
        Rng r = entity_rng(42, 21, "island:2051:layout");
        const uint64_t ref[3] = {REF_RAW0, REF_RAW1, REF_RAW2};
        for (uint64_t x : ref) CHECK(r.raw() == x);
    }
    {
        Rng r = entity_rng(42, 21, "island:2051:layout");
        CHECK(r.standard_normal() == REF_NORMAL0);
        CHECK(r.integers(0, 10) == REF_INT1);
        CHECK(r.beta(1.3, 2.2) == REF_BETA2);
    }
    {
        // 一个 5×5 的小盆：优先泛洪后中心被填到边缘 + eps 以上，D8 每格有下游或是出口
        GridD h(5, 5, 10.0);
        Mask m(5, 5, 1);
        h(2, 2) = 1.0;
        GridD f = priority_fill(h, m, 1e-3);
        CHECK(f(2, 2) > 10.0 && f(2, 2) < 10.01);
        FlowDir d = d8(f, m, 100.0);
        int outlets = 0;
        for (size_t k = 0; k < d.to_void.size(); ++k) outlets += d.to_void[k];
        CHECK(outlets == 16);
        CHECK(d.ri[2 * 5 + 2] >= 0);
        GridD A = accumulate(m, d);
        double tot = 0;
        for (int k = 0; k < 25; ++k)
            if (d.ri[k] < 0) tot += A.v[k];
        CHECK(tot == 25.0);
    }
    CHECK(pyround(2.675, 2) == 2.67);   // 二进制 2.675 = 2.67499999…
    CHECK(pyround(0.125, 2) == 0.12);   // 恰为一半：逢半取偶
    {
        // 聚落营建器（PLAN-TOWN）的基础件
        using namespace skyisle::town;
        // 有向矩形：正北朝向、面阔 10 进深 6 的两块，中心相距 11 m 不相交，加 2 m 间距就相交；转 45° 仍包含中心
        const Obb a{{0.0, 0.0}, 0.0, 5.0, 3.0}, b{{11.0, 0.0}, 0.0, 5.0, 3.0};
        CHECK(!overlap(a, b));
        CHECK(overlap(a, b, 2.0));
        CHECK(contains(Obb{{0.0, 0.0}, PI / 4, 5.0, 3.0}, {0.0, 0.0}));
        CHECK(std::fabs(bearing_of(bearing_vec(1.0)) - 1.0) < 1e-12);
        CHECK(std::fabs(angle_diff(0.1, 2 * PI - 0.1) - 0.2) < 1e-12);
        // 精确欧氏距离与最近种子：与暴力法逐格相同
        Mask seed(9, 11, 0);
        seed(2, 3) = 1, seed(7, 9) = 1, seed(0, 10) = 1;
        GridF d;
        Grid<int32_t> src;
        edt(seed, d, &src);
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 11; ++j) {
                double best = 1e9;
                for (int p = 0; p < 9; ++p)
                    for (int q = 0; q < 11; ++q)
                        if (seed(p, q)) best = std::min(best, std::hypot(double(i - p), double(j - q)));
                CHECK(std::fabs(d(i, j) - best) < 1e-5);
                const int k = src(i, j), p = k / 11, q = k % 11;
                CHECK(seed(p, q) && std::fabs(std::hypot(double(i - p), double(j - q)) - best) < 1e-5);
            }
        // 哈希噪声只看坐标与种子
        CHECK(value_noise(7, 12.3, -4.5) == value_noise(7, 12.3, -4.5));
        CHECK(fbm(7, 1000.0, 2000.0, 40.0, 3) != fbm(8, 1000.0, 2000.0, 40.0, 3));
        CHECK(std::fabs(value_noise(7, 3.0, 4.0) - hash_unit(7, 3, 4)) < 1e-12);   // 格点上取格点值
        // Chaikin 保首尾、属性随之插值；折线距离
        std::vector<V2> L = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}};
        std::vector<std::vector<double>> at = {{0.0}, {1.0}, {2.0}};
        chaikin(L, &at, 2);
        CHECK(L.front().x == 0.0 && L.back().y == 10.0 && at.back()[0] == 2.0 && L.size() == at.size());
        CHECK(std::fabs(dist_point_polyline({5.0, 3.0}, {{0.0, 0.0}, {10.0, 0.0}}) - 3.0) < 1e-12);
    }
    {
        // 朝向规则链：朝阳（权 3，容差 15°）为主、顺街成排（权 1，容差 4°，模 90°）为辅
        using namespace skyisle::town;
        const double D = PI / 180.0;
        const std::vector<OrientRule> rules = {{"sun", 3.0, 15 * D, 0.0}, {"align_street", 1.0, 4 * D, 0.0}};
        OrientCtx c;
        c.sun = PI;
        c.street_dir = PI / 2 + 10 * D;   // 街偏 10°：朝阳容差里就能顺街，两条都满足
        double f = solve_facing(rules, c, 0.0, NaN);
        CHECK(angle_diff(f, PI) <= 15 * D + 1e-9);
        CHECK(rule_deviation(rules[1], c, f) <= 4 * D + 1e-6);
        c.street_dir = PI / 2 + 40 * D;   // 街偏 40°：顺不了街；容差外是软的，朝阳让出几度但不丢
        f = solve_facing(rules, c, 0.0, NaN);
        CHECK(angle_diff(f, PI) < 22 * D && rule_deviation(rules[1], c, f) > 4 * D);
        c.sun = 0.0, c.street_dir = NaN;  // 南半球朝北；没有街的语境，顺街那条跳过
        CHECK(angle_diff(solve_facing(rules, c, 0.0, NaN), 0.0) < 1e-9);
        c.street_dir = 0.3;               // 山墙朝街模 180°：反过来也算顺
        CHECK(std::fabs(rule_deviation(OrientRule{"gable_street", 1.0, 0.0, 0.0}, c, 0.3 + PI)) < 1e-9);
        CHECK(solve_facing({OrientRule{"water", 1.0, 0.0, 0.0}}, OrientCtx{}, 0.0, 1.25) == 1.25);   // 规则都没目标：回退
    }
    {
        // 寻路：60 × 40 格平地，x = 30 一堵南北墙只在行 17–22 留 6 m 的口；路从口里过、离墙 ≥ 间距；间距比口的一半还大就不过；湖不架桥、窄河架桥
        using namespace skyisle::town;
        const int H = 40, W = 60;
        Site s;
        s.res_m = 1.0, s.H = H, s.W = W, s.x0 = 0.0, s.y0 = 40.0;
        s.height = GridF(H, W, 100.0f), s.water_level = GridF(H, W, std::nanf("")), s.water = Grid<uint8_t>(H, W, WATER_NONE);
        s.sky = Mask(H, W, 0), s.edge = Mask(H, W, 0), s.farmland = Mask(H, W, 0), s.flood = Mask(H, W, 0);
        s.landcover = Grid<uint8_t>(H, W, 0), s.island = Grid<int16_t>(H, W, 0), s.water_dist_m = GridF(H, W, 1e6f);
        Mask blocked(H, W, 0), road(H, W, 0), goal(H, W, 0);
        for (int i = 0; i < H; ++i) {
            if (i < 17 || i > 22) blocked(i, 30) = 1;
            goal(i, 55) = 1;
        }
        PathParams pp;
        pp.clearance_m = 1.5;
        std::vector<V2> out;
        CHECK(find_path(s, blocked, road, goal, pp, s.center(20, 5), out));
        bool through = false;
        double dmin = 1e9;
        for (size_t k = 0; k + 1 < out.size(); ++k) {
            if ((out[k].x - 30.5) * (out[k + 1].x - 30.5) <= 0.0) through = through || (out[k].y > 17.0 && out[k].y < 23.0);
            for (int i = 0; i < H; ++i)
                if (blocked(i, 30)) dmin = std::min(dmin, dist_point_segment(s.center(i, 30), out[k], out[k + 1]));
        }
        CHECK(through && out.back().x > 54.0 && out.back().x < 57.0);
        CHECK(dmin >= 1.5 - 0.9);   // 格心量的间距，平滑后差不过一格
        pp.clearance_m = 3.0;       // 口的中线离两头的墙格 3 格：刚好过
        CHECK(find_path(s, blocked, road, goal, pp, s.center(20, 5), out));
        pp.clearance_m = 3.5;
        CHECK(!find_path(s, blocked, road, goal, pp, s.center(20, 5), out));
        pp.clearance_m = 0.0;
        for (int i = 0; i < H; ++i)
            for (int j = 40; j < 43; ++j) s.water(i, j) = WATER_RIVER, s.water_level(i, j) = 98.0f, s.height(i, j) = 96.5f;
        CHECK(find_path(s, blocked, road, goal, pp, s.center(20, 5), out));   // 3 m 宽、岸高出水面 2 m 的河：架桥
        pp.bridge_max_m = 2.0;
        CHECK(!find_path(s, blocked, road, goal, pp, s.center(20, 5), out));  // 比能架的宽：不过
        pp.bridge_max_m = 24.0;
        for (int i = 0; i < H; ++i)
            for (int j = 40; j < 43; ++j) s.water(i, j) = WATER_LAKE;
        CHECK(!find_path(s, blocked, road, goal, pp, s.center(20, 5), out));  // 湖：不过
    }
    {
        // 等值线（等高线算子的台线、環濠的外廓用它）：锥面 g = 到 (30, 20) 的距离，= 10 的线是闭合的圆；斜面 g = x 的线是直的、不闭合；NaN 格断开
        using namespace skyisle::town;
        const int H = 40, W = 60;
        Site s;
        s.res_m = 1.0, s.H = H, s.W = W, s.x0 = 0.0, s.y0 = 40.0;
        GridF cone(H, W, 0.0f), ramp(H, W, 0.0f);
        const V2 c0{30.0, 20.0};
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) cone(i, j) = static_cast<float>(len(s.center(i, j) - c0)), ramp(i, j) = static_cast<float>(s.center(i, j).x);
        auto L = iso_lines(s, cone, 10.0, 1, c0, 100.0);
        CHECK(L.size() == 1 && L[0].size() > 20);
        if (L.size() == 1) {
            CHECK(len(L[0].front() - L[0].back()) < 1e-9);   // 闭合
            double e = 0.0;
            for (const V2& p : L[0]) e = std::max(e, std::fabs(len(p - c0) - 10.0));
            CHECK(e < 0.3);
        }
        L = iso_lines(s, ramp, 20.5, 1, c0, 100.0);
        CHECK(L.size() == 1 && L[0].size() >= 30 && len(L[0].front() - L[0].back()) > 30.0);
        if (!L.empty()) {
            double e = 0.0;
            for (const V2& p : L[0]) e = std::max(e, std::fabs(p.x - 20.5));
            CHECK(e < 1e-6);
        }
        for (int j = 0; j < W; ++j) ramp(20, j) = std::nanf("");   // 横着一行 NaN：竖线断成两段
        CHECK(iso_lines(s, ramp, 20.5, 1, c0, 100.0).size() == 2);
        CHECK(iso_lines(s, cone, 10.0, 1, c0, 5.0).empty());      // 半径外不看
    }
    {
        // 非正方形格网与偏离对角线的岸线，防止行列互换被对称图形掩盖。
        GridD phi(5, 9, 0.0);
        for (int axis = 0; axis < 2; ++axis) {
            for (int i = 0; i < phi.H; ++i)
                for (int j = 0; j < phi.W; ++j)
                    phi(i, j) = (axis == 0 ? j - 5.25 : i - 1.75);
            const auto distance = island::coast_distance(phi, 20.0);
            for (size_t k = 0; k < phi.size(); ++k)
                CHECK(std::fabs(distance.v[k] - phi.v[k] * 20.0) < 1e-9);
        }
    }
    if (fails) {
        std::printf("%d 项失败\n", fails);
        return 1;
    }
    std::printf("core_selftest 全过\n");
    return 0;
}
