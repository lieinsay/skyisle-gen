// 亚格岸距（PLAN-NATURE B4，DESIGN-NOTES 四点四十六）。见 coast.hpp。
#include "skyisle/island/coast.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace skyisle::island {

namespace {

struct Seg {
    double ax, ay, bx, by;   // x = 列、y = 行（格心 = 整数）
};

double seg_dist(const Seg& s, double y, double x) {
    const double dx = s.bx - s.ax, dy = s.by - s.ay;
    const double L2 = dx * dx + dy * dy;
    double t = L2 > 0 ? ((y - s.ay) * dy + (x - s.ax) * dx) / L2 : 0.0;
    t = std::min(1.0, std::max(0.0, t));
    const double py = s.ay + t * dy, px = s.ax + t * dx;
    return std::sqrt((y - py) * (y - py) + (x - px) * (x - px));
}

}  // namespace

GridD coast_distance(const GridD& phi, double res_m) {
    const int H = phi.H, W = phi.W;
    const size_t N = phi.size();
    GridD out(H, W, NaN);
    if (H < 2 || W < 2) return out;
    // marching squares：块 (i, j) 的四角是格心 (i, j) (i, j+1) (i+1, j+1) (i+1, j)
    std::vector<Seg> segs;
    std::vector<int32_t> head(static_cast<size_t>(H) * W, -1), next;
    auto add = [&](int bi, int bj, double ay, double ax, double by, double bx) {
        segs.push_back({ax, ay, bx, by});
        next.push_back(head[static_cast<size_t>(bi) * W + bj]);
        head[static_cast<size_t>(bi) * W + bj] = static_cast<int32_t>(segs.size() - 1);
    };
    for (int i = 0; i + 1 < H; ++i)
        for (int j = 0; j + 1 < W; ++j) {
            const double v[4] = {phi(i, j), phi(i, j + 1), phi(i + 1, j + 1), phi(i + 1, j)};
            const double py[4] = {0, 0, 1, 1}, px[4] = {0, 1, 1, 0};
            int code = 0;
            for (int q = 0; q < 4; ++q) code |= (v[q] > 0 ? 1 : 0) << q;
            if (code == 0 || code == 15) continue;
            // 各边（q → q+1）的交点
            std::array<double, 4> ey{}, ex{};
            std::array<bool, 4> has{};
            for (int q = 0; q < 4; ++q) {
                const int r = (q + 1) % 4;
                if ((v[q] > 0) != (v[r] > 0)) {
                    const double t = v[q] / (v[q] - v[r]);
                    ey[q] = i + py[q] + t * (py[r] - py[q]);
                    ex[q] = j + px[q] + t * (px[r] - px[q]);
                    has[q] = true;
                }
            }
            std::vector<int> e;
            for (int q = 0; q < 4; ++q)
                if (has[q]) e.push_back(q);
            if (e.size() == 2) {
                add(i, j, ey[e[0]], ex[e[0]], ey[e[1]], ex[e[1]]);
            } else if (e.size() == 4) {
                // 鞍点：按块心的均值定连法（块心是陆地就把两块陆地连起来）
                const double cv = 0.25 * (v[0] + v[1] + v[2] + v[3]);
                const bool c0 = v[0] > 0;
                if ((cv > 0) == c0) {   // 角 0 与块心同号：切掉角 1、角 3
                    add(i, j, ey[0], ex[0], ey[1], ex[1]);
                    add(i, j, ey[2], ex[2], ey[3], ex[3]);
                } else {                 // 切掉角 0、角 2
                    add(i, j, ey[3], ex[3], ey[0], ex[0]);
                    add(i, j, ey[1], ex[1], ey[2], ex[2]);
                }
            }
        }
    if (segs.empty()) return out;
    std::vector<int32_t> best(N, -1);
    std::vector<double> dist(N, INF);
    // 折线旁：格心周围 5 × 5 块里的线段
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            const size_t k = static_cast<size_t>(i) * W + j;
            for (int a = std::max(0, i - 2); a <= std::min(H - 2, i + 1); ++a)
                for (int b = std::max(0, j - 2); b <= std::min(W - 2, j + 1); ++b)
                    for (int32_t s = head[static_cast<size_t>(a) * W + b]; s >= 0; s = next[s]) {
                        const double d = seg_dist(segs[s], i, j);
                        if (d < dist[k]) {
                            dist[k] = d;
                            best[k] = s;
                        }
                    }
        }
    // 往外传：邻格的最近线段也试一下（前后各两遍）
    auto relax = [&](int i, int j, int a, int b) {
        if (a < 0 || b < 0 || a >= H || b >= W) return;
        const int32_t s = best[static_cast<size_t>(a) * W + b];
        if (s < 0) return;
        const size_t k = static_cast<size_t>(i) * W + j;
        if (best[k] == s) return;
        const double d = seg_dist(segs[s], i, j);
        if (d < dist[k]) {
            dist[k] = d;
            best[k] = s;
        }
    };
    for (int rep = 0; rep < 2; ++rep) {
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                relax(i, j, i - 1, j - 1);
                relax(i, j, i - 1, j);
                relax(i, j, i - 1, j + 1);
                relax(i, j, i, j - 1);
            }
        for (int i = H - 1; i >= 0; --i)
            for (int j = W - 1; j >= 0; --j) {
                relax(i, j, i + 1, j + 1);
                relax(i, j, i + 1, j);
                relax(i, j, i + 1, j - 1);
                relax(i, j, i, j + 1);
            }
    }
    for (size_t k = 0; k < N; ++k)
        if (best[k] >= 0) out.v[k] = (phi.v[k] > 0 ? 1.0 : -1.0) * dist[k] * res_m;
    return out;
}

}  // namespace skyisle::island
