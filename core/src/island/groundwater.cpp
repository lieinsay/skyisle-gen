// 集水核（C4）与地下水（C5）：见 groundwater.hpp。
#include "skyisle/island/groundwater.hpp"

#include "skyisle/flow.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace skyisle::island {

namespace {

double wc(const Config& c, const std::string& k) { return c.get("water." + k); }

// 凝结 1 m/年放出的潜热（W/m²）：2.45 MJ/kg × 1000 kg/m³ / 一年的秒数（一年 336 天约 84 W/m²，spec 13 第八节第 7 条）
double latent_w_per_m(double year_s) { return 2.45e6 * 1000.0 / std::max(1.0, year_s); }

}  // namespace

GridD condensation(const Group& g, const GridD& Gw, const Config& c, std::vector<double>& core_s) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const int n = static_cast<int>(g.islands.size());
    GridD out(H, W, 0.0);
    core_s.assign(n, 0.0);
    const double gain = wc(c, "core_gain");
    if (!(gain > 0.0)) return out;
    const double cell_km2 = g.res_km * g.res_km;
    // 核的强度跟山走：高出岸缘的山体（km³）的立方根 / core_len_km——大山根深、核大；碎的小岛核弱（spec 13 第八节第 6 条）
    std::vector<double> vol(n, 0.0);
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        vol[id] += std::max(0.0, g.height.v[k] - g.islands[id].rim_j) * cell_km2 / 1000.0;
    }
    const double lref = wc(c, "core_len_km"), smax = wc(c, "core_s_max");
    for (int k = 0; k < n; ++k) core_s[k] = clip(std::cbrt(vol[k]) / lref, 0.0, smax);
    // 落在核山的迎风高处（凝结是气流被山逼着抬升、贴着林子与岩面过去时截下来的）；跟着湿度走：像除湿器，空气干就凝得少
    const double eexp = wc(c, "core_elev_exp"), wg = wc(c, "core_windward_gain"), hfull = wc(c, "core_hum_full_mm"), hexp = wc(c, "core_hum_exp");
    const double gref = c.get("hydro.windward_ref_m_per_km");
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        const IslandRec& J = g.islands[id];
        const double e = clip((g.height.v[k] - J.rim_j) / std::max(1.0, J.peak_j - J.rim_j), 0.0, 1.0);
        const double wf = Gw.v.empty() ? 1.0 : clip(1.0 + wg * clip(Gw.v[k] / gref, -1.0, 1.0), 0.0, 2.0);
        const double P = g.rain.v[k];
        const double hum = np_pow(clip(P / hfull, 0.0, 1.0), hexp);
        out.v[k] = P * std::min(1.0, gain * core_s[id] * np_pow(e, eexp) * wf * hum);   // 最多到局地雨的 1 倍（spec 13 第八节第 4 条）
    }
    return out;
}

void aquifer(Group& g, const Config& c) {
    const int H = g.H, W = g.W;
    const size_t N = static_cast<size_t>(H) * W;
    const double cell_km2 = g.res_km * g.res_km;
    const bool has_lith = !g.lith.v.empty() && !g.strat_top.v.empty();
    const std::vector<double> bl = c.list("water.bfi", {0.55, 0.25, 0.4, 0.2, 0.0});
    const double bdef = wc(c, "bfi_default");
    // 基流比例按出露岩性（石灰岩岩溶、辉长岩裂隙、泥灰岩与蛇纹岩差、浮石闭孔不透水）；补给 = 雨的径流 × 基流比例 + 凝结水（凝结水全渗进岩层）
    g.bfi = GridD(H, W, 0.0);
    g.recharge = GridD(H, W, 0.0);
    for (size_t k = 0; k < N; ++k) {
        if (g.island_id.v[k] < 0) continue;
        double b = bdef;
        if (has_lith) {
            const uint8_t li = g.lith.v[k];
            b = (li >= 1 && li - 1 < static_cast<int>(bl.size())) ? bl[li - 1] : bdef;
        }
        const double cond = g.condense.v.empty() ? 0.0 : g.condense.v[k];
        const double rr = std::max(0.0, g.runoff.v[k] - cond);
        g.bfi.v[k] = b;
        g.recharge.v[k] = rr * b + cond;
    }
    // 顺流向累计（mm·km²）：地下水跟着地表流向走（岩层里的水顺着骨架顶面往外、往低处去，与地表的大方向一致）；
    // 河道格上的累计 = 河的基流。按流向的拓扑序（上游先加），与线程数无关
    const std::vector<int64_t> recv = group_recv(g);
    Mask land(H, W, 0);
    for (size_t k = 0; k < N; ++k) land.v[k] = g.island_id.v[k] >= 0 ? 1 : 0;
    const std::vector<int32_t> order = downstream_first(recv, land);
    g.recharge_acc = GridD(H, W, 0.0);
    for (size_t k = 0; k < N; ++k) g.recharge_acc.v[k] = g.recharge.v[k] * cell_km2;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t r = recv[*it];
        if (r >= 0) g.recharge_acc.v[r] += g.recharge_acc.v[*it];
    }
    // 崖壁泉线：流向走到岸边、一路没进河道（河、溪涧）也没进湖的出口格——这片坡的地下水没被河截走，顺着骨架顶面走到崖边，
    // 从崖壁上岩层与浮石的交界渗出来。出口格按岛、按 8 连通成串，长串按绕岛心的方位切成 springline_seg_km 一段
    g.springline.clear();
    Mask ex(H, W, 0);
    for (size_t k = 0; k < N; ++k)
        ex.v[k] = (g.island_id.v[k] >= 0 && g.recv_i.v[k] < 0 && g.river.v[k] == 0 && g.stream.v[k] == 0 && !g.lake.v[k] &&
                   g.recharge_acc.v[k] > 0.0) ? 1 : 0;
    GridI lab;
    const int nl = label_components(ex, 8, lab);
    std::vector<std::vector<int32_t>> comp(static_cast<size_t>(nl) + 1);
    for (size_t k = 0; k < N; ++k)
        if (lab.v[k] > 0) comp[lab.v[k]].push_back(static_cast<int32_t>(k));
    const int n = static_cast<int>(g.islands.size());
    std::vector<double> ci(n, 0.0), cj(n, 0.0), cn(n, 0.0);
    for (size_t k = 0; k < N; ++k) {
        const int id = g.island_id.v[k];
        if (id < 0) continue;
        ci[id] += static_cast<double>(k / W);
        cj[id] += static_cast<double>(k % W);
        cn[id] += 1.0;
    }
    const int seg = std::max(1, static_cast<int>(std::nearbyint(wc(c, "springline_seg_km") / g.res_km)));
    const double to_ls = 1e6 / std::max(1.0, g.year_s);   // mm·km² / 年 → L/s（1 mm·km² = 1000 m³）
    const double qmin = wc(c, "springline_min_ls"), qfall = wc(c, "springline_fall_ls");
    const double ske = c.get("strat.skel_edge_frac", 0.4);
    for (int L = 1; L <= nl; ++L) {
        std::vector<int32_t>& cs = comp[L];
        const int id = g.island_id.v[cs[0]];
        const double oi = ci[id] / std::max(1.0, cn[id]), oj = cj[id] / std::max(1.0, cn[id]);
        std::vector<double> ang(cs.size());
        for (size_t q = 0; q < cs.size(); ++q) ang[q] = std::atan2(-(static_cast<double>(cs[q] / W) - oi), static_cast<double>(cs[q] % W) - oj);
        std::vector<size_t> ord(cs.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return ang[a] < ang[b] || (ang[a] == ang[b] && cs[a] < cs[b]); });
        for (size_t s0 = 0; s0 < ord.size(); s0 += static_cast<size_t>(seg)) {
            const size_t s1 = std::min(ord.size(), s0 + static_cast<size_t>(seg));
            SpringSeg S;
            S.island = id;
            double q = 0.0, hs = 0.0;
            for (size_t t = s0; t < s1; ++t) {
                const int32_t k = cs[ord[t]];
                S.cells.push_back(k);
                q += g.recharge_acc.v[k];
                const IslandRec& J = g.islands[id];
                hs += (!g.skel_top.v.empty() && !std::isnan(g.skel_top.v[k])) ? g.skel_top.v[k] : J.keel_j + ske * (J.rim_j - J.keel_j);
            }
            S.q_ls = q * to_ls;
            if (S.q_ls < qmin) continue;
            const int32_t mid = cs[ord[(s0 + s1 - 1) / 2]];
            S.ci = mid / W;
            S.cj = mid % W;
            S.height_m = hs / static_cast<double>(s1 - s0);
            S.length_km = static_cast<double>(s1 - s0) * g.res_km;
            S.fall = S.q_ls >= qfall;
            g.springline.push_back(std::move(S));
        }
    }
}

std::vector<int64_t> group_recv(const Group& g) {
    const size_t N = static_cast<size_t>(g.H) * g.W;
    std::vector<int64_t> recv(N, -1);
    for (size_t k = 0; k < N; ++k) {
        const int ri = g.recv_i.v[k], rj = g.recv_j.v[k];
        if (g.island_id.v[k] >= 0 && ri >= 0 && rj >= 0) {
            const int64_t r = static_cast<int64_t>(ri) * g.W + rj;
            if (g.island_id.v[r] >= 0) recv[k] = r;
        }
    }
    return recv;
}

double core_heat_mw(const Group& g, int island, const Config& c) {
    if (g.condense.v.empty()) return 0.0;
    const size_t N = static_cast<size_t>(g.H) * g.W;
    const double cell_m2 = g.res_km * g.res_km * 1e6;
    double s = 0.0;
    for (size_t k = 0; k < N; ++k)
        if (g.island_id.v[k] == island) s += g.condense.v[k] / 1000.0;
    return s * cell_m2 * latent_w_per_m(g.year_s) * wc(c, "heat_share") / 1e6;
}

}  // namespace skyisle::island
