// 5.3b 河道成形（river.py 同式）。
#include "skyisle/island/river.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "skyisle/flow.hpp"

namespace skyisle::island {

namespace {
double hydraulic_depth(double q, double w, double s, double n) {
    if (!(q > 0 && w > 0)) return 0;
    auto capacity = [&](double d) {
        const double a = w * d, r = a / (w + 2 * d);
        return a * std::pow(r, 2.0 / 3.0) * std::sqrt(s) / n;
    };
    double lo = 0, hi = 1;
    while (capacity(hi) < q) hi *= 2;
    for (int k = 0; k < 48; ++k) {
        const double mid = (lo + hi) / 2;
        if (capacity(mid) < q) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}
double ramp(double x, double lo, double hi) {
    const double t = clip((x - lo) / (hi - lo), 0.0, 1.0);
    return t * t * (3 - 2 * t);
}
}

ReachGeometry reach_geometry(const GridD& h, const Mask& land, const Mask& lake,
                             const std::vector<int64_t>& recv, const GridD& q,
                             const Grid<uint8_t>& lith, const Channels& ref,
                             double res_m, double year_s, const Config& c) {
    const int H = h.H, W = h.W;
    const size_t N = h.size();
    ReachGeometry out{ref.width, ref.depth, GridD(H,W,0), GridD(H,W,0), GridD(H,W,0), GridD(H,W,0)};
    const double yield = c.get("hydro.reach_source_mm_y", .05);
    const double abrasion = c.get("hydro.reach_abrasion_km", 20);
    const double days = c.get("hydro.reach_transport_days", 30);
    const double roughness = c.get("hydro.reach_manning_n", .035);
    const double trap = c.get("hydro.reach_lake_trap", .8);
    const double ratio = c.get("hydro.bf_ratio_channel", 5);
    if (!(std::isfinite(yield) && std::isfinite(abrasion) && std::isfinite(days) && std::isfinite(roughness) && std::isfinite(ratio) &&
          yield >= 0 && abrasion > 0 && days > 0 && days <= year_s/86400 && roughness > 0 && trap >= 0 && trap <= 1 && ratio > 0))
        throw std::invalid_argument("河段水沙参数超出有效范围");
    const auto grains = c.list("hydro.reach_lith_grain_mm", {40,30,80,65,30});
    const auto coarse = c.list("hydro.reach_lith_coarse", {.5,.3,.7,.6,.3});
    if (grains.size()!=5 || coarse.size()!=5) throw std::invalid_argument("河段岩性表须为五项");
    for (size_t k=0;k<5;++k)
        if (!(grains[k]>=27 && grains[k]<=167.5 && coarse[k]>=0 && coarse[k]<=1))
            throw std::invalid_argument("河段粒径或粗粒比例无效");
    std::vector<double> fine(N,0), gravel(N,0), moment(N,0);
    std::vector<double> local_q=q.v;
    for (size_t k=0;k<N;++k) if (land.v[k] && recv[k]>=0) local_q[recv[k]]-=q.v[k];
    auto order = downstream_first(recv, land);
    for (auto it=order.rbegin(); it!=order.rend(); ++it) {
        const int32_t k=*it;
        const int64_t r=recv[k];
        const double step = r>=0 ? res_m*((r/W!=k/W && r%W!=k%W)?SQRT2:1) : res_m;
        // 独立的坡面来源代理（kg/年），不是凭空增加水，也不声称等于地形演化移除的质量。
        const int li = lith.v.empty()?2:std::max(0,std::min(4,static_cast<int>(lith.v[k])-1));
        const double slope = r>=0 ? std::max(0.0,(h.v[k]-h.v[r])/step):0;
        const double runoff_mm=std::max(0.0,local_q[k])*year_s/(res_m*res_m)*1000;
        const double mass = lake.v[k]?0:yield/1000 * res_m*res_m * 2650 *
            std::sqrt(clip(slope/.1,0.0,10.0))*clip(runoff_mm/500,0.0,5.0);
        out.source.v[k]=mass;
        fine[k]+=mass*(1-coarse[li]); gravel[k]+=mass*coarse[li];
        moment[k]+=mass*coarse[li]*std::log(grains[li]/1000);
        double d = gravel[k]>0 ? std::exp(moment[k]/gravel[k]):grains[li]/1000;
        if (ref.width.v[k]>0 && !lake.v[k]) {
            const double S = std::max(c.get("hydro.min_grade",.001),ref.slope.v[k]);
            const double qb = q.v[k]*ratio;
            // 不将低流量、陡峡谷或非砾床条件直接套进经验样本范围；过渡权重避免硬门槛生成截头。
            const double grain_fit = ramp(d,.02,.027);
            const double confinement = ref.floor_w.v[k]/std::max(ref.width.v[k],.1);
            const double alpha = gravel[k]>0 ? ramp(qb,2.7,5.4)*(1-ramp(S,.015,.031))*ramp(confinement,3,15)*grain_fit : 0;
            const double D = clip(d,.027,.1675);
            const double target = 4.63/std::pow(9.81,.2)*std::pow(qb,.4)*
                                  std::pow(qb/(std::sqrt(9.81)*std::pow(D,2.5)),.0667);
            const double width = (1-alpha)*ref.width.v[k]+alpha*target;
            // 过渡区连续混合阻力，而不是直接混合两种水深（后者不满足输水方程）。
            const double a0=ref.width.v[k]*ref.depth.v[k];
            const double radius0=a0/(ref.width.v[k]+2*ref.depth.v[k]);
            const double n0=qb>0 ? a0*std::pow(radius0,2.0/3.0)*std::sqrt(S)/qb : roughness;
            const double depth = hydraulic_depth(qb,width,S,(1-alpha)*n0+alpha*roughness);
            out.width.v[k]=width;
            out.depth.v[k]=alpha>0?depth:ref.depth.v[k];
            // MPM 简化容量，流动天数是独立待校准假设；超额粗沙存为沉积，不丢失。
            const double radius=width*depth/(width+2*depth);
            const double theta=radius*S/(1.65*std::max(d,1e-5));
            const double capacity=8*std::sqrt(1.65*9.81*std::pow(d,3))*std::pow(std::max(theta-.047,0.0),1.5)*width*2650*days*86400;
            const double retained = std::max(0.0,gravel[k]-capacity);
            const double frac=gravel[k]>0 ? (gravel[k]-retained)/gravel[k]:0;
            out.deposit.v[k]+=retained; gravel[k]*=frac; moment[k]*=frac;
        }
        if (lake.v[k]) {
            out.deposit.v[k]+=trap*(fine[k]+gravel[k]);
            fine[k]*=1-trap;gravel[k]*=1-trap;moment[k]*=1-trap;
        }
        out.grain.v[k]=d;
        out.flux.v[k]=fine[k]+gravel[k];
        if (r>=0) {
            // 磨细转成细沙，质量守恒；幸存粗粒逐步变细，汇合按质量混合。
            const double survive=std::exp(-step/(abrasion*1000));
            fine[r]+=fine[k]+gravel[k]*(1-survive);
            gravel[r]+=gravel[k]*survive;
            moment[r]+=gravel[k]*survive*(gravel[k]>0?moment[k]/gravel[k]+std::log(survive):0);
        }
    }
    return out;
}

Channels carve_channels(const GridD& h, const GridD& hf, const Mask& mk, const Mask& lake, const std::vector<int64_t>& recv,
                        const GridD& Akm, const Grid<uint8_t>& river_lvl, const Grid<uint8_t>& stream, double P_mm, double runoff, double rim,
                        double keel, double res_m, double year_s, const Config& c, bool /*is_main*/, const GridD* Qin,
                        const GridD* wall_deg, const GridD* floor_lith, double age, const ReachGeometry* geometry) {
    const int H = h.H, W = h.W;
    const size_t N = h.size();
    Channels out;
    Mask work(H, W, 0), center_r(H, W, 0), center_s(H, W, 0), seed(H, W, 0);
    bool any_seed = false, any_r = false;
    for (size_t k = 0; k < N; ++k) {
        work.v[k] = (mk.v[k] && !lake.v[k]) ? 1 : 0;
        center_r.v[k] = (work.v[k] && river_lvl.v[k] > 0) ? 1 : 0;
        center_s.v[k] = (work.v[k] && stream.v[k] > 0 && !center_r.v[k]) ? 1 : 0;
        seed.v[k] = (center_r.v[k] || center_s.v[k]) ? 1 : 0;
        any_seed = any_seed || seed.v[k];
        any_r = any_r || center_r.v[k];
    }
    out.width = GridD(H, W, 0.0);
    out.depth = GridD(H, W, 0.0);
    out.bed = GridD(H, W, NaN);
    out.floor_w = GridD(H, W, 0.0);
    out.confine = Grid<uint8_t>(H, W, 0);
    out.slope = GridD(H, W, NaN);
    if (!any_seed) {
        out.h_new = h;
        out.lvl = river_lvl;
        out.floodplain = Grid<uint8_t>(H, W, 0);
        return out;
    }
    // 本模型的年均水面幂律：w = 5·Q^0.5、d = 0.35·Q^0.4 m。
    // 幂律形式参考水力几何，系数尚未按小型岛河独立标定；不把少数大河示例当作通用校准。
    // **平岸**宽深在 rivernet 里按站内指数换算（w ∝ Q^0.26、d ∝ Q^0.40，四点五十一）。
    // C1 起没有夸张（旧 width_scale / depth_scale 删了）
    const double wa = c.get("hydro.width_a"), wb = c.get("hydro.width_b");
    const double dc = c.get("hydro.depth_c"), df = c.get("hydro.depth_f");
    // 切出来的河道 = **平岸河道**：年均流量下的水面宽深 × (平岸 / 年均)^站内指数（地形阶段只有假设的比值 bf_ratio_channel，
    // rivers.json 另记逐日年最大比值，但它不是已验证的平岸比；两者都不能取代独立标定）
    const double bfr = c.get("hydro.bf_ratio_channel", 5.0);
    const double aw = c.get("hydro.at_station_width_b", 0.26), ad = c.get("hydro.at_station_depth_f", 0.40);
    const double ch_w = np_pow(bfr, aw), ch_d = np_pow(bfr, ad);
    std::vector<double> Q(N), width(N), depth(N), incise(N), base(N), bed(N, NaN);
    double amax = -INF;
    for (size_t k = 0; k < N; ++k)
        if (center_r.v[k]) amax = std::max(amax, Akm.v[k]);
    if (!any_r) amax = 1.0;
    const double inc_m = c.get("hydro.incise_m"), s_inc = c.get("hydro.stream_incise_m");
    for (size_t k = 0; k < N; ++k) {
        Q[k] = Qin ? Qin->v[k] : Akm.v[k] * 1e6 * (P_mm / 1000.0) * runoff / year_s;
        const double q = std::max(Q[k], 0.0);
        const double w = geometry ? geometry->width.v[k] : wa * np_pow(q, wb) * ch_w;
        const double d = geometry ? geometry->depth.v[k] : dc * np_pow(q, df) * ch_d;
        width[k] = seed.v[k] ? w : 0.0; // 同一流量的河槽不因河 / 溪涧标签而突变。
        depth[k] = seed.v[k] ? d : 0.0;
        incise[k] = center_r.v[k] ? inc_m * std::sqrt(clip(Akm.v[k] / amax, 0.0, 1.0)) : (center_s.v[k] ? s_inc : 0.0);
        base[k] = std::min(mk.v[k] ? h.v[k] : INF, mk.v[k] ? hf.v[k] : INF);
        if (seed.v[k]) bed[k] = base[k] - depth[k] - incise[k];
    }
    // 河床向下游单调：按路由面降序，把上游河床（减一个极小落差）传给下游
    const double floor = rim - c.get("hydro.notch_m");
    std::vector<int32_t> order;
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k]) order.push_back(static_cast<int32_t>(k));
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return hf.v[a] > hf.v[b]; });
    std::vector<double> bl(N), Lkm(N, 0.0);
    for (size_t k = 0; k < N; ++k) bl[k] = std::max(std::isnan(bed[k]) ? INF : bed[k], floor);
    const double step = res_m / 1000.0;
    for (int32_t k : order) {
        const int64_t r = recv[k];
        if (r >= 0 && seed.v[r]) {
            if (bl[r] > bl[k] - 0.01) bl[r] = std::max(floor, bl[k] - 0.01);
            if (center_r.v[k] && center_r.v[r]) Lkm[r] = std::max(Lkm[r], Lkm[k] + step);
        }
    }
    // 下游段不压平（C3）：到河口的河道长 L 上，河床至少高出豁口 min_grade × L（分级河流搬得动泥沙的最小比降；
    // 旧的下切到豁口以下的河段一路按每格 1 cm 压在豁口上，两三成的河道比降 ≈ 0）。抬高不超过本格地面 − 水深，之后再把单调补一遍
    const double smin = c.get("hydro.min_grade");
    if (smin > 0.0) {
        std::vector<double> Ld(N, 0.0);
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const int32_t k = *it;
            const int64_t r = recv[k];
            if (r >= 0 && seed.v[r]) Ld[k] = Ld[r] + res_m * ((r / W != k / W && r % W != k % W) ? SQRT2 : 1.0);
        }
        for (int32_t k : order) {
            const double lift = std::min(floor + smin * Ld[k], base[k] - depth[k]);
            if (lift > bl[k]) bl[k] = lift;
        }
        for (int32_t k : order) {
            const int64_t r = recv[k];
            if (r >= 0 && seed.v[r] && bl[r] > bl[k] - 0.01) bl[r] = std::max(floor, bl[k] - 0.01);
        }
    }
    // 支流接干流：溪涧河床从汇入处往上游按最大比降爬升（下游先算）
    const double rise = res_m * c.get("hydro.stream_grade_max");
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int32_t k = *it;
        const int64_t r = recv[k];
        if (center_s.v[k] && r >= 0 && seed.v[r] && bl[k] > bl[r] + rise) bl[k] = bl[r] + rise;
    }
    for (size_t k = 0; k < N; ++k) bed[k] = seed.v[k] ? bl[k] : NaN;

    // 河道比降：顺流向 valley_slope_len_m 以内的河床落差 / 流程；走不出一步的（河口、流进湖的）沿用上游的
    const int nstep = std::max(1, static_cast<int>(std::nearbyint(c.get("hydro.valley_slope_len_m") / res_m)));
    std::vector<double> Sch(N, NaN);
    for (int32_t k : order) {
        double d = 0.0;
        int64_t cur = k;
        for (int s = 0; s < nstep; ++s) {
            const int64_t r = recv[cur];
            if (r < 0 || !seed.v[r]) break;
            d += res_m * ((r / W != cur / W && r % W != cur % W) ? SQRT2 : 1.0);
            cur = r;
        }
        if (d > 0.0) Sch[k] = std::max(0.0, (bed[k] - bed[cur]) / d);
    }
    for (int32_t k : order) {
        const int64_t r = recv[k];
        if (r >= 0 && seed.v[r] && std::isnan(Sch[r])) Sch[r] = Sch[k];
    }
    // 河谷剖面（C2，谷底宽照地球）：最近河道格的 河床 / 水深 / 谷底半宽 / 谷坡。
    // 谷底全宽 W = 河宽 + valley_floor_k_m × A^valley_floor_exp × 岩性系数 × 岛龄系数 × 比降系数（夹 valley_floor_max_m）：侧蚀把谷底展平，软岩、老岛宽，硬岩、新岛窄；
    // 比降系数 1 / (1 + (S / valley_slope_ref)^valley_slope_exp)：陡处下切快过侧蚀，谷底只剩河道（Montgomery & Buffington 1997 的河型：
    // < 1.5% 滩槽型不受限、1–3% 平床型、3–10% 阶梯深潭多半受限、> 10% 跌水）。
    // 谷底 = 平岸水面（滩面，涨水即漫）。限制度按 W / 河宽：≤ confine_gorge_ratio 峡谷、≥ confine_open_ratio 开阔、中间半限制；
    // 谷坡按限制度在该处岩性的坍塌角（峡谷，B2）与 min(坍塌角, valley_side_open_deg)（开阔：谷坡有坡积裙、缓）之间按 log 比值过渡
    // （旧的 10 × 河宽的漫滩与按流量 32° → 9° 的谷坡作废）。溪涧同式（它的 A 小，谷底多半不到一格）
    const double gorge0 = c.get("hydro.gorge_deg", 32.0), open_deg = c.get("hydro.valley_side_open_deg");
    const double vk = c.get("hydro.valley_floor_k_m"), ve = c.get("hydro.valley_floor_exp"), vmax = c.get("hydro.valley_floor_max_m");
    const double rg = c.get("hydro.confine_gorge_ratio"), ro = c.get("hydro.confine_open_ratio");
    const double age_mult = np_interp({age}, c.list("hydro.valley_age", {0.0, 0.3, 0.65, 1.0}), c.list("hydro.valley_age_mult", {0.5, 0.8, 1.2, 1.6}))[0];
    std::vector<double> tan_side(N), fp_half(N);
    const double lr = std::log(ro / rg);
    const double sref = c.get("hydro.valley_slope_ref"), sexp = c.get("hydro.valley_slope_exp");
    for (size_t k = 0; k < N; ++k) {
        const double gorge = wall_deg ? wall_deg->v[k] : gorge0;
        if (!seed.v[k]) {
            tan_side[k] = std::tan(gorge * (PI / 180.0));
            fp_half[k] = 0.0;
            continue;
        }
        const double fl = floor_lith ? floor_lith->v[k] : 1.0;
        const double sf = std::isnan(Sch[k]) ? 1.0 : 1.0 / (1.0 + np_pow(Sch[k] / sref, sexp));
        const double Wf = std::min(vmax, width[k] + vk * np_pow(std::max(Akm.v[k], 0.0), ve) * fl * age_mult * sf);
        const double ratio = Wf / std::max(width[k], 0.1);
        const double t = clip(std::log(std::max(ratio, 1e-9) / rg) / lr, 0.0, 1.0);
        const double side = (1 - t) * gorge + t * std::min(open_deg, gorge);
        tan_side[k] = std::tan(side * (PI / 180.0));
        fp_half[k] = Wf / 2.0;
        out.floor_w.v[k] = Wf;
        out.slope.v[k] = std::isnan(Sch[k]) ? 0.0 : Sch[k];
        out.confine.v[k] = static_cast<uint8_t>(ratio <= rg ? 1 : (ratio >= ro ? 3 : 2));
    }
    const int svc = c.geti("hydro.stream_valley_cells");
    const int reach_cells = any_r ? static_cast<int>(std::ceil(c.get("hydro.valley_max_km") * 1000.0 / res_m)) : svc;
    GridD dist;
    Grid<int64_t> src;
    nearest_propagate(seed, reach_cells, res_m, &work, dist, src);
    out.h_new = GridD(H, W);
    out.lvl = Grid<uint8_t>(H, W, 0);
    out.floodplain = Grid<uint8_t>(H, W, 0);
    for (size_t k = 0; k < N; ++k) {
        bool has = work.v[k] && src.v[k] >= 0;
        const size_t s = has ? static_cast<size_t>(src.v[k]) : 0;
        const double bed_s = bed[s], dep_s = depth[s], fp_s = fp_half[s], tan_s = tan_side[s], w_s = width[s];
        const bool isr_s = center_r.v[s] != 0;
        const uint8_t lvl_s = river_lvl.v[s];
        const double lim = isr_s ? INF : svc * res_m;
        has = has && dist.v[k] <= lim;
        const double valley = bed_s + dep_s + std::max(0.0, dist.v[k] - fp_s) * tan_s;
        double hn = has ? std::min(h.v[k], valley) : h.v[k];
        if (has && std::isnan(h.v[k])) hn = NaN;
        const bool wide = has && isr_s && !center_r.v[k] && dist.v[k] < w_s / 2.0;
        const bool chan = center_r.v[k] || wide;
        // 河道格记平岸水面（C1：= 滩面 = 河床 + 水深；河床另存 bed）
        if (chan) hn = std::isnan(hn) || std::isnan(bed_s) ? NaN : std::min(hn, bed_s + dep_s);
        if (seed.v[k]) hn = bed[k] + depth[k];
        if (chan || seed.v[k]) out.bed.v[k] = seed.v[k] ? bed[k] : bed_s;
        out.h_new.v[k] = hn;
        out.lvl.v[k] = wide ? lvl_s : river_lvl.v[k];
        out.width.v[k] = chan ? (center_r.v[k] ? width[k] : w_s) : (center_s.v[k] ? width[k] : 0.0);
        out.depth.v[k] = chan ? (center_r.v[k] ? depth[k] : dep_s) : (center_s.v[k] ? depth[k] : 0.0);
        // 漫滩 = 谷底里的岸上格（河与溪涧都算：溪涧的谷底多半不到一格）
        out.floodplain.v[k] = (has && !chan && !seed.v[k] && dist.v[k] <= fp_s + 0.5 * res_m && hn <= bed_s + dep_s + 0.5) ? 1 : 0;
    }
    // 最高格不动（IS-summit）
    size_t top = 0;
    double best = -INF;
    for (size_t k = 0; k < N; ++k) {
        const double v = mk.v[k] ? h.v[k] : -INF;
        if (v > best) {
            best = v;
            top = k;
        }
    }
    out.h_new.v[top] = h.v[top];
    for (size_t k = 0; k < N; ++k)
        if (!mk.v[k]) out.h_new.v[k] = h.v[k];

    // 摘要：每个河口（常年河流向虚空处）一条河
    if (any_r)
        for (size_t k = 0; k < N; ++k)
            if (center_r.v[k] && recv[k] < 0) {
                RiverRec rv;
                rv.mouth_r = static_cast<int>(k / W);
                rv.mouth_c = static_cast<int>(k % W);
                rv.basin_km2 = Akm.v[k];
                rv.length_km = Lkm[k] + step;
                rv.discharge = Q[k];
                rv.width = width[k];
                rv.depth = depth[k];
                rv.level = river_lvl.v[k];
                rv.waterfall = bed[k] - keel;
                rv.incision = base[k] - bed[k];
                out.rivers.push_back(rv);
            }
    for (size_t k = 0; k < N; ++k)
        if (center_s.v[k] && recv[k] < 0) ++out.n_stream_falls;
    double mc = -INF;
    for (size_t k = 0; k < N; ++k) mc = std::max(mc, mk.v[k] ? h.v[k] - out.h_new.v[k] : 0.0);
    out.has_cut = true;
    out.max_cut_m = pyround(mc, 1);
    GridD wgrid(H, W);
    wgrid.v = width;
    Grid<uint8_t> lv(H, W, 0);
    for (size_t k = 0; k < N; ++k) lv.v[k] = center_r.v[k] ? river_lvl.v[k] : 0;
    out.lines = trace_lines(seed, mk, recv, wgrid, lv, Akm);
    return out;
}

std::vector<std::vector<LinePt>> trace_lines(const Mask& seed, const Mask& mk, const std::vector<int64_t>& recv, const GridD& width,
                                             const Grid<uint8_t>& lvl, const GridD& acc) {
    const int H = seed.H, W = seed.W;
    const size_t N = seed.size();
    std::vector<int32_t> indeg(N, 0);
    for (size_t k = 0; k < N; ++k)
        if (seed.v[k]) {
            const int64_t t = recv[k];
            if (t >= 0 && seed.v[t]) indeg[t]++;
        }
    std::vector<uint8_t> visited(N, 0);
    std::vector<std::vector<LinePt>> lines;
    static const int D8L[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (size_t s = 0; s < N; ++s) {
        if (!seed.v[s] || indeg[s] != 0) continue;
        std::vector<LinePt> pts;
        int64_t k = static_cast<int64_t>(s);
        for (;;) {
            const int i = static_cast<int>(k / W), j = static_cast<int>(k % W);
            pts.push_back({i + 0.5, j + 0.5, width.v[k], static_cast<int>(lvl.v[k]), acc.v[k]});
            if (visited[k] && pts.size() > 1) break;
            visited[k] = 1;
            const int64_t r = recv[k];
            if (r < 0) {
                for (const auto& d : D8L) {
                    const int a = i + d[0], b = j + d[1];
                    if (!(a >= 0 && a < H && b >= 0 && b < W) || !mk.v[static_cast<size_t>(a) * W + b]) {
                        const LinePt last = pts.back();
                        pts.push_back({i + 0.5 + 0.5 * d[0], j + 0.5 + 0.5 * d[1], last.w, last.lvl, last.acc});
                        break;
                    }
                }
                break;
            }
            if (!seed.v[r]) break;
            k = r;
        }
        if (pts.size() >= 2) lines.push_back(std::move(pts));
    }
    return lines;
}

}  // namespace skyisle::island
