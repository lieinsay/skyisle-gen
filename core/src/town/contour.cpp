#include "skyisle/town/contour.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace skyisle::town {

std::vector<std::vector<V2>> iso_lines(const Site& s, const GridF& g, double level, int stride, V2 c, double radius) {
    std::vector<std::vector<V2>> out;
    stride = std::max(1, stride);
    int ci = 0, cj = 0;
    int i0 = 0, i1 = s.H - 1, j0 = 0, j1 = s.W - 1;
    if (std::isfinite(radius) && radius < 1e8 && s.cell_of(c, ci, cj)) {
        const int rc = static_cast<int>(std::ceil(radius / s.res_m)) + stride;
        i0 = std::max(0, ci - rc), i1 = std::min(s.H - 1, ci + rc), j0 = std::max(0, cj - rc), j1 = std::min(s.W - 1, cj + rc);
    }
    const int ni = (i1 - i0) / stride + 1, nj = (j1 - j0) / stride + 1;
    if (ni < 2 || nj < 2) return out;
    auto val = [&](int a, int b) {
        const float v = g(i0 + a * stride, j0 + b * stride);
        return std::isfinite(v) ? static_cast<double>(v) - level : NaN;
    };
    auto pos = [&](int a, int b) { return s.center(i0 + a * stride, j0 + b * stride); };
    // 交点落在采样网格的边上：横边 (a, b)–(a, b + 1) 的键 2 × (a·nj + b)，竖边 (a, b)–(a + 1, b) 的键 2 × (a·nj + b) + 1
    std::unordered_map<int64_t, V2> pt;
    std::vector<std::pair<int64_t, int64_t>> segs;
    auto hkey = [&](int a, int b) { return 2 * (static_cast<int64_t>(a) * nj + b); };
    auto vkey = [&](int a, int b) { return 2 * (static_cast<int64_t>(a) * nj + b) + 1; };
    auto cross_pt = [&](int64_t key, V2 p, V2 q, double vp, double vq) {
        if (!pt.count(key)) pt[key] = lerp(p, q, vp / (vp - vq));
        return key;
    };
    for (int a = 0; a + 1 < ni; ++a)
        for (int b = 0; b + 1 < nj; ++b) {
            const double v00 = val(a, b), v01 = val(a, b + 1), v10 = val(a + 1, b), v11 = val(a + 1, b + 1);
            if (!(std::isfinite(v00) && std::isfinite(v01) && std::isfinite(v10) && std::isfinite(v11))) continue;
            if (std::isfinite(radius) && radius < 1e8 && len(lerp(pos(a, b), pos(a + 1, b + 1), 0.5) - c) > radius) continue;
            const bool p00 = v00 >= 0, p01 = v01 >= 0, p10 = v10 >= 0, p11 = v11 >= 0;
            std::vector<int64_t> e;   // 有交点的边，按 上、右、下、左 的次序
            if (p00 != p01) e.push_back(cross_pt(hkey(a, b), pos(a, b), pos(a, b + 1), v00, v01));
            if (p01 != p11) e.push_back(cross_pt(vkey(a, b + 1), pos(a, b + 1), pos(a + 1, b + 1), v01, v11));
            if (p10 != p11) e.push_back(cross_pt(hkey(a + 1, b), pos(a + 1, b), pos(a + 1, b + 1), v10, v11));
            if (p00 != p10) e.push_back(cross_pt(vkey(a, b), pos(a, b), pos(a + 1, b), v00, v10));
            if (e.size() == 2) segs.push_back({e[0], e[1]});
            else if (e.size() == 4) {
                // 鞍点：四角均值在 level 以上 → 高的两角连成一片
                const bool mid_pos = (v00 + v01 + v10 + v11) >= 0;
                if (mid_pos == p00) segs.push_back({e[0], e[1]}), segs.push_back({e[2], e[3]});
                else segs.push_back({e[0], e[3]}), segs.push_back({e[1], e[2]});
            }
        }
    // 连成折线
    std::unordered_map<int64_t, std::vector<int>> at;
    for (int k = 0; k < static_cast<int>(segs.size()); ++k) at[segs[k].first].push_back(k), at[segs[k].second].push_back(k);
    std::vector<uint8_t> used(segs.size(), 0);
    for (int k0 = 0; k0 < static_cast<int>(segs.size()); ++k0) {
        if (used[k0]) continue;
        used[k0] = 1;
        std::vector<int64_t> chain = {segs[k0].first, segs[k0].second};
        for (int dir = 0; dir < 2; ++dir) {
            for (;;) {
                const int64_t tip = chain.back();
                int nxt = -1;
                for (int k : at[tip])
                    if (!used[k]) {
                        nxt = k;
                        break;
                    }
                if (nxt < 0) break;
                used[nxt] = 1;
                chain.push_back(segs[nxt].first == tip ? segs[nxt].second : segs[nxt].first);
            }
            std::reverse(chain.begin(), chain.end());
        }
        std::vector<V2> line;
        for (int64_t key : chain) line.push_back(pt[key]);
        if (line.size() >= 2) out.push_back(std::move(line));
    }
    return out;
}

}  // namespace skyisle::town
