// 5.4 四季（climate.py 的 build_climate / daily_curves）与 5.5 逐日天气（weather.py 的 season_params / simulate_year / multi_year_stats）。
// 数都照 Python 版的运算次序：numpy 数组的 cos / hypot / arctan2 = C 库同名函数（本机实测逐位相同），Python 浮点的 ** = C 的 pow，
// 小数组的 mean = 成对求和 / n，np.interp 与 np.convolve（BLAS ddot）见 grid.hpp。
#include "skyisle/island/climate.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace skyisle::island {

namespace {

double mean4(const std::vector<double>& v) { return np_sum(v.data(), v.size()) / static_cast<double>(v.size()); }

// match_mean：把四季原始值缩放到均值 = target 并夹在 [lo, hi]：先乘性，夹断后把剩余亏欠加性摊到未顶格的季
std::vector<double> match_mean(const std::vector<double>& raw, double target, double lo, double hi, int iters = 12) {
    std::vector<double> x(raw.size());
    for (size_t k = 0; k < x.size(); ++k) x[k] = clip(raw[k], lo, hi);
    const double m0 = mean4(x);
    if (m0 > 1e-9) {
        const double f = target / m0;
        for (double& v : x) v = clip(v * f, lo, hi);
    }
    for (int it = 0; it < iters; ++it) {
        const double d = target - mean4(x);
        if (std::fabs(d) < 1e-7) break;
        std::vector<uint8_t> free(x.size());
        int64_t nf = 0;
        for (size_t k = 0; k < x.size(); ++k) {
            free[k] = d > 0 ? (x[k] < hi - 1e-12) : (x[k] > lo + 1e-12);
            nf += free[k];
        }
        if (!nf) break;
        const double add = d * static_cast<double>(x.size()) / static_cast<double>(nf);
        for (size_t k = 0; k < x.size(); ++k) x[k] = clip(x[k] + (free[k] ? add : 0.0), lo, hi);
    }
    return x;
}

// 相对降水 → 毫米：与行星层同一条（④ 的 precip_mm_ref × p，PLAN-NATURE A3；旧的 150 + 3850 × p^1.3 作废）。
// 不夹上限：按份额分的雨季折成年当量可以过 1
double precip_mm_of(double p_rel, const NodeInputs& inp) { return inp.precip_mm_ref * std::max(p_rel, 0.0); }

size_t argmax_first(const std::vector<double>& v) {
    size_t b = 0;
    for (size_t k = 1; k < v.size(); ++k)
        if (v[k] > v[b]) b = k;
    return b;
}
size_t argmin_first(const std::vector<double>& v) {
    size_t b = 0;
    for (size_t k = 1; k < v.size(); ++k)
        if (v[k] < v[b]) b = k;
    return b;
}

std::vector<std::string> season_names(const std::string& stype, const std::vector<double>& t, const std::vector<double>& precip,
                                      const std::vector<double>& storm, const std::vector<double>& window, int n_s) {
    std::vector<std::string> names(n_s, "season");
    if (stype == "four") {
        const int hot = static_cast<int>(argmax_first(t)), cold = static_cast<int>(argmin_first(t));
        for (int s = 0; s < n_s; ++s) {
            if (s == hot) names[s] = "hot";
            else if (s == cold) names[s] = "cold";
            else {
                const int dc = ((s - cold) % n_s + n_s) % n_s, dh = ((s - hot) % n_s + n_s) % n_s;
                names[s] = dc < dh ? "warm" : "cool";
            }
        }
    } else if (stype == "two") {
        std::vector<int> order(n_s);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -t[a] < -t[b]; });
        std::vector<uint8_t> warm(n_s, 0);
        for (int q = 0; q < n_s / 2; ++q) warm[order[q]] = 1;
        int wi = 0, ci = 0;
        for (int s = 0; s < n_s; ++s) {
            if (warm[s]) {
                ++wi;
                names[s] = n_s > 2 ? "warm" + std::to_string(wi) : "warm";
            } else {
                ++ci;
                names[s] = n_s > 2 ? "cold" + std::to_string(ci) : "cold";
            }
        }
    } else if (stype == "rain") {
        const int wet = static_cast<int>(argmax_first(precip)), dry = static_cast<int>(argmin_first(precip));
        for (int s = 0; s < n_s; ++s) names[s] = s == wet ? "wet" : (s == dry ? "dry" : "turn");
    } else if (stype == "storm") {
        std::vector<double> st(n_s);
        for (int s = 0; s < n_s; ++s) st[s] = storm[s] - window[s];
        const int hi = static_cast<int>(argmax_first(st)), lo = static_cast<int>(argmin_first(st));
        for (int s = 0; s < n_s; ++s) names[s] = s == hi ? "stormy" : (s == lo ? "calm" : "transition");
    } else {
        for (int s = 0; s < n_s; ++s) names[s] = "s" + std::to_string(s + 1);
    }
    return names;
}

}  // namespace

namespace {

// 岛上口径的陆地性（④ 同式）→ 热惯性的滞后（弧度）
double island_lag(const NodeInputs& inp, const PlanetView& pv) {
    const double cont = pv.cg_cont.empty() ? 0.1 : grid_interp(pv.cg_cont, pv.cg_grid, inp.lat, inp.lon);
    const double ct = clip(cont + pv.alt_cont * clip((inp.height_m - inp.keel_clearance_m) / 2000.0, 0.0, 1.0), 0.0, 1.0);
    const double tau = ct * pv.tau_land + (1.0 - ct) * pv.tau_ocean;
    return std::atan(2.0 * PI / pv.cal.year_days * tau);
}

}  // namespace

SeasonThermal season_thermal(const NodeInputs& inp, const PlanetView& pv) {
    const Calendar& cal = pv.cal;
    const int n_s = cal.seasons;
    SeasonThermal th;
    th.days_per_season = cal.days_per_season;
    th.months_per_season = cal.months_per_season;
    th.t_ref.resize(n_s);
    th.day_hr.resize(n_s);
    const double lag = island_lag(inp, pv), sgn = inp.lat < 0 ? -1.0 : 1.0, amp = 0.5 * inp.season_range;
    const double tl = std::tan(inp.lat * (PI / 180.0));
    for (int s = 0; s < n_s; ++s) {
        const double ph = 2.0 * PI * ((s + 0.5) * cal.days_per_season - cal.offset) / cal.year_days;
        th.t_ref[s] = inp.temp + sgn * amp * std::cos(ph - lag);
        const double dec = pv.tilt_deg * std::cos(ph) * (PI / 180.0);
        th.day_hr[s] = 24.0 / PI * std::acos(clip(-tl * std::tan(dec), -1.0, 1.0));
    }
    return th;
}

Climate build_climate(const NodeInputs& inp, const PlanetView& pv, const Config& c) {
    Climate C;
    C.cal = pv.cal;
    const Calendar& cal = pv.cal;
    const double ydays = cal.year_days;
    const int n_s = cal.seasons;
    const double dps = cal.days_per_season;
    const double tilt = pv.tilt_deg;
    const double lat = inp.lat, lon = inp.lon;
    const bool south = lat < 0;
    // 陆地性：零点口径（原叫海面口径，_sea）/ 岛上口径（④ 同式）
    const double cont = pv.cg_cont.empty() ? 0.1 : grid_interp(pv.cg_cont, pv.cg_grid, lat, lon);
    const double alt = clip(cont + pv.alt_cont * clip((inp.height_m - inp.keel_clearance_m) / 2000.0, 0.0, 1.0), 0.0, 1.0);
    const double cont_sea = clip(cont, 0.0, 1.0), cont_isl = alt;
    const double w = 2.0 * PI / ydays;
    auto thermal = [&](double ct, double& tau, double& A, double& lag) {
        tau = ct * pv.tau_land + (1.0 - ct) * pv.tau_ocean;
        const double wt = w * tau;
        A = 1.0 / std::sqrt(1.0 + wt * wt);
        lag = std::atan(w * tau);
    };
    double tau_sea, A_sea, lag_sea, tau_isl, A_isl, lag_isl;
    thermal(cont_sea, tau_sea, A_sea, lag_sea);
    thermal(cont_isl, tau_isl, A_isl, lag_isl);
    // 每季的带界位移读 ④ 的（全球一个数，PLAN-NATURE A5；旧的 [island.climate] k_shift × 本地 A_sea 作废）；旧产物没有时按同式现算
    std::vector<double> mids(n_s), ph(n_s), dphi(n_s);
    const bool have_shift = static_cast<int>(pv.season_shift.size()) == n_s;
    const double wt_o = w * pv.tau_ocean, A_o = 1.0 / std::sqrt(1.0 + wt_o * wt_o), lag_o = std::atan(wt_o);
    for (int s = 0; s < n_s; ++s) {
        mids[s] = (s + 0.5) * dps;
        ph[s] = 2.0 * PI * (mids[s] - cal.offset) / ydays;
        dphi[s] = have_shift ? pv.season_shift[s] : pv.band_shift_k * tilt * A_o * std::cos(ph[s] - lag_o);
    }
    // 局部带界（local_edges）：按经度线性插值 eq_n / eq_s 两行
    double eqn, eqs;
    {
        const auto& L = pv.band_lons;
        const size_t n = L.size();
        const double res = L[1] - L[0];
        const double fj = (lon - L[0]) / res;
        const double j0f = std::floor(fj);
        const double t = fj - j0f;
        int64_t j0 = static_cast<int64_t>(j0f) % static_cast<int64_t>(n);
        if (j0 < 0) j0 += static_cast<int64_t>(n);
        const int64_t j1 = (j0 + 1) % static_cast<int64_t>(n);
        eqn = pv.band_eq_n[j0] * (1 - t) + pv.band_eq_n[j1] * t;
        eqs = pv.band_eq_s[j0] * (1 - t) + pv.band_eq_s[j1] * t;
    }
    std::vector<double> rp(n_s), rs(n_s), rw(n_s), ru(n_s), rv(n_s), ls(n_s);
    for (int s = 0; s < n_s; ++s) {
        const double eq = lat >= 0 ? eqn : -eqs;
        const double sm = lat - dphi[s];
        double v;
        if (lat >= 0) {
            const double b = eq + 0.25;
            v = b > sm ? b : sm;
        } else {
            const double b = -eq - 0.25;
            v = b < sm ? b : sm;
        }
        ls[s] = v;
        rp[s] = grid_interp(pv.cg_precip, pv.cg_grid, v, lon);
        rs[s] = grid_interp(pv.cg_storm, pv.cg_grid, v, lon);
        rw[s] = grid_interp(pv.cg_window, pv.cg_grid, v, lon);
        ru[s] = grid_interp(pv.wind_u, pv.wind_grid, v, lon);
        rv[s] = grid_interp(pv.wind_v, pv.wind_grid, v, lon);
    }
    // 四季降水：④ 的份额（四季各解一遍水汽，A2）× 季数 × 年均；旧产物没有份额时照旧取样再缩放到年均
    std::vector<double> precip(n_s);
    if (static_cast<int>(inp.precip_share.size()) == n_s)
        for (int s = 0; s < n_s; ++s) precip[s] = inp.precip * n_s * inp.precip_share[s];
    else
        precip = match_mean(rp, inp.precip, 0.0, 1.0);
    const std::vector<double> storm = mean4(rs) > 1e-9 ? match_mean(rs, inp.storm, 0.0, 1.0) : std::vector<double>(n_s, inp.storm);
    const std::vector<double> window = match_mean(rw, inp.window, 0.03, 1.0);
    const double sgn = south ? -1.0 : 1.0;
    const double amp_isl = 0.5 * inp.season_range, amp_sea = 0.5 * inp.season_range_sea;
    std::vector<double> t_isl(n_s), t_sea(n_s), p_rate(n_s), p_mm(n_s);
    for (int s = 0; s < n_s; ++s) {
        t_isl[s] = inp.temp + sgn * amp_isl * std::cos(ph[s] - lag_isl);
        t_sea[s] = inp.temp_sea + sgn * amp_sea * std::cos(ph[s] - lag_sea);
        p_rate[s] = precip_mm_of(precip[s], inp);
        p_mm[s] = p_rate[s] * dps / ydays;
    }
    // 季型判定：分数 = 各项差异 / 该项门槛（温度按冷暖两季的门槛）
    const double r_t = inp.season_range;
    const double pmax = *std::max_element(precip.begin(), precip.end()), pmin = *std::min_element(precip.begin(), precip.end());
    const double pr_ratio = pmax / std::max(1e-9, pmin);
    const double st_diff = *std::max_element(storm.begin(), storm.end()) - *std::min_element(storm.begin(), storm.end());
    const double wn_diff = *std::max_element(window.begin(), window.end()) - *std::min_element(window.begin(), window.end());
    const double sc_temp = r_t / c.get("climate.two_season_range_c");
    const double sc_rain = pr_ratio / c.get("climate.wet_dry_ratio");
    const double sc_storm = std::max(st_diff / c.get("climate.storm_season_diff"), wn_diff / c.get("climate.window_season_diff"));
    std::string stype;
    if (r_t >= c.get("climate.four_season_range_c")) {
        stype = "four";
    } else {
        // max(sorted(cand), key=分数)：键按字母序（rain < storm < temp），同分取先者
        std::string best;
        double bv = -INF;
        const std::pair<const char*, double> cands[3] = {{"rain", sc_rain}, {"storm", sc_storm}, {"temp", sc_temp}};
        for (const auto& kv : cands)
            if (kv.second >= 1.0 && kv.second > bv) {
                bv = kv.second;
                best = kv.first;
            }
        if (best.empty()) stype = "none";
        else stype = best == "temp" ? "two" : (best == "rain" ? "rain" : "storm");
    }
    C.stype = stype;
    C.names = season_names(stype, t_isl, precip, storm, window, n_s);
    C.type_code = stype != "none" ? stype : (inp.temp < c.get("climate.cold_mean_temp_c") ? "none_cold" : "none_warm");
    C.score_temp = pyround(sc_temp, 3);
    C.score_rain = pyround(sc_rain, 3);
    C.score_storm = pyround(sc_storm, 3);
    for (int s = 0; s < n_s; ++s) {
        SeasonRec R;
        R.index = s;
        R.name = C.names[s];
        R.d0 = static_cast<int>(std::nearbyint(s * dps));
        R.d1 = static_cast<int>(std::nearbyint((s + 1) * dps)) - 1;
        R.mid_day = mids[s];
        for (int m = 0; m < cal.months_per_season; ++m) R.months.push_back(s * cal.months_per_season + m);
        R.temp_c = pyround(t_isl[s], 2);
        R.temp_sea_c = pyround(t_sea[s], 2);
        R.precip_rel = pyround(precip[s], 4);
        R.precip_mm = pyround(p_mm[s], 0);
        R.precip_rate = pyround(p_rate[s], 0);
        R.storm = pyround(storm[s], 4);
        R.window = pyround(window[s], 4);
        R.wu = pyround(ru[s], 2);
        R.wv = pyround(rv[s], 2);
        R.speed = pyround(py_hypot(ru[s], rv[s]), 2);
        R.from_deg = pyround(pymod(std::atan2(-ru[s], -rv[s]) * (180.0 / PI) + 360.0, 360.0), 0);
        R.band_shift = pyround(dphi[s], 2);
        R.lat_sampled = pyround(ls[s], 2);
        C.seasons.push_back(R);
    }
    C.a_temp = pyround(inp.temp, 2);
    C.a_temp_sea = pyround(inp.temp_sea, 2);
    C.a_precip_rel = pyround(inp.precip, 4);
    C.a_precip_mm = pyround(precip_mm_of(inp.precip, inp), 0);
    C.a_storm = pyround(inp.storm, 4);
    C.a_window = pyround(inp.window, 4);
    C.a_range = pyround(r_t, 2);
    C.a_range_sea = pyround(inp.season_range_sea, 2);
    C.a_winter = pyround(inp.temp_winter, 2);
    C.a_summer = pyround(inp.temp_summer, 2);
    C.a_ref_h = pyround(inp.height_m, 1);
    C.cont_sea = pyround(cont_sea, 4);
    C.cont_isl = pyround(cont_isl, 4);
    C.tau_sea = pyround(tau_sea, 1);
    C.tau_isl = pyround(tau_isl, 1);
    C.A_sea = pyround(A_sea, 3);
    C.A_isl = pyround(A_isl, 3);
    C.lag_sea_days = pyround(lag_sea / (2 * PI) * ydays, 1);
    C.lag_isl_days = pyround(lag_isl / (2 * PI) * ydays, 1);
    C.k_shift = pv.band_shift_k;
    double amp = 0.0;
    for (int s = 0; s < n_s; ++s) amp = std::max(amp, std::fabs(dphi[s]));
    C.band_amp = pyround(amp, 2);
    C.m_precip = pyround(mean4(precip), 5);
    C.m_storm = pyround(mean4(storm), 5);
    C.m_window = pyround(mean4(window), 5);
    C.m_temp = pyround(mean4(t_isl), 4);
    C.season_range_1 = pyround(r_t, 1);
    return C;
}

Daily daily_curves(const Climate& clim, const NodeInputs& inp) {
    const Calendar& cal = clim.cal;
    const int ydays = static_cast<int>(std::nearbyint(cal.year_days));
    Daily D;
    D.day.resize(ydays);
    std::vector<double> d(ydays);
    for (int k = 0; k < ydays; ++k) {
        D.day[k] = k;
        d[k] = k;
    }
    const double sgn = inp.lat < 0 ? -1.0 : 1.0;
    const double lag = clim.lag_isl_days / ydays * 2 * PI;
    D.temp_c.resize(ydays);
    for (int k = 0; k < ydays; ++k) {
        const double ph = 2.0 * PI * (d[k] - cal.offset) / cal.year_days;
        D.temp_c[k] = inp.temp + sgn * 0.5 * inp.season_range * std::cos(ph - lag);
    }
    const int n_s = static_cast<int>(clim.seasons.size());
    std::vector<double> x(3 * n_s);
    for (int s = 0; s < n_s; ++s) {
        x[s] = clim.seasons[s].mid_day - ydays;
        x[n_s + s] = clim.seasons[s].mid_day;
        x[2 * n_s + s] = clim.seasons[s].mid_day + ydays;
    }
    const int k = std::max(1, static_cast<int>(std::floor(cal.days_per_season / 3)));
    std::vector<double> ker(2 * k + 1, 1.0 / static_cast<double>(2 * k + 1));
    auto interp_periodic = [&](const std::vector<double>& vals) {
        std::vector<double> y(3 * n_s);
        for (int s = 0; s < n_s; ++s) y[s] = y[n_s + s] = y[2 * n_s + s] = vals[s];
        const std::vector<double> lin = np_interp(d, x, y);
        std::vector<double> ext;
        ext.reserve(lin.size() + 2 * k);
        ext.insert(ext.end(), lin.end() - k, lin.end());
        ext.insert(ext.end(), lin.begin(), lin.end());
        ext.insert(ext.end(), lin.begin(), lin.begin() + k);
        return np_convolve_valid(ext, ker);
    };
    D.season.resize(ydays);
    const int dps_i = static_cast<int>(cal.days_per_season);
    for (int q = 0; q < ydays; ++q) D.season[q] = q / dps_i;
    std::vector<double> v(n_s);
    for (int s = 0; s < n_s; ++s) v[s] = clim.seasons[s].precip_rel;
    D.precip_rel = interp_periodic(v);
    for (int s = 0; s < n_s; ++s) v[s] = clim.seasons[s].precip_mm;
    D.precip_mm = interp_periodic(v);
    for (int s = 0; s < n_s; ++s) v[s] = clim.seasons[s].storm;
    D.storm = interp_periodic(v);
    for (int s = 0; s < n_s; ++s) v[s] = clim.seasons[s].window;
    D.window = interp_periodic(v);
    for (int s = 0; s < n_s; ++s) v[s] = clim.seasons[s].wu;
    D.wind_u = interp_periodic(v);
    for (int s = 0; s < n_s; ++s) v[s] = clim.seasons[s].wv;
    D.wind_v = interp_periodic(v);
    return D;
}

std::vector<SeasonParams> season_params(const Climate& clim, const Config& c) {
    std::vector<SeasonParams> out;
    const double dps = clim.cal.days_per_season;
    const double persist = c.get("weather.wet_persistence");
    for (const SeasonRec& s : clim.seasons) {
        SeasonParams p;
        const double P = s.precip_mm;
        p.f_wet = clip(c.get("weather.wet_frac_base") + c.get("weather.wet_frac_k") * c_pow(P / 1000.0, 0.7), c.get("weather.wet_frac_min"),
                       c.get("weather.wet_frac_max"));
        p.p_ww = p.f_wet + persist * (1.0 - p.f_wet);
        double p_dw = p.f_wet * (1.0 - p.p_ww) / std::max(1e-9, 1.0 - p.f_wet);
        double sf = c.get("weather.storm_day_k") * c_pow(s.storm, c.get("weather.storm_day_exp"));
        sf = std::min(sf, 0.6);
        const double mult = c.get("weather.storm_rain_mult");
        p.mean_wet_mm = P / std::max(1.0, dps * ((1.0 - sf) * p.f_wet + sf * mult));
        p.f_rain_days = (1.0 - sf) * p.f_wet + sf;
        p.p_dw = std::min(p_dw, 0.95);
        p.storm_frac = sf;
        p.p_calm = clip(s.window / std::max(1e-6, 1.0 - sf), 0.0, 1.0);
        p.wind_u = s.wu;
        p.wind_v = s.wv;
        p.speed = s.speed;
        out.push_back(p);
    }
    return out;
}

WeatherYear simulate_year(Rng& rng, const Climate& clim, const Daily& daily, const std::vector<SeasonParams>& params, double surface_m,
                          const Config& c, double rim_m, double lapse_c_per_km) {
    const Calendar& cal = clim.cal;
    const int n = static_cast<int>(std::nearbyint(cal.year_days));
    const int dps = static_cast<int>(std::nearbyint(cal.days_per_season));
    const int dpm = static_cast<int>(std::nearbyint(cal.days_per_month));
    const std::vector<int32_t>& season = daily.season;
    WeatherYear Y;
    std::vector<uint8_t> wet(n, 0);
    bool state = rng.uniform(0.0, 1.0) < params[season[0]].f_wet;
    for (int d = 0; d < n; ++d) {
        const SeasonParams& p = params[season[d]];
        const double pr = state ? p.p_ww : p.p_dw;
        state = rng.uniform(0.0, 1.0) < pr;
        wet[d] = state ? 1 : 0;
    }
    const double shape = c.get("weather.rain_gamma_shape");
    std::vector<double> mean_wet(n), precip(n);
    for (int d = 0; d < n; ++d) mean_wet[d] = params[season[d]].mean_wet_mm;
    for (int d = 0; d < n; ++d) {
        const double gv = rng.gamma(shape, 1.0);
        precip[d] = wet[d] ? gv * mean_wet[d] / shape : 0.0;
    }
    // 风暴事件：泊松布尔模型的覆盖率反解事件数，起日均匀，持续 storm_days_min–max
    std::vector<int32_t> storm_id(n, 0);
    int eid = 0;
    const int dlo = static_cast<int>(c.get("weather.storm_days_min")), dhi = static_cast<int>(c.get("weather.storm_days_max"));
    const double mean_dur = 0.5 * (dlo + dhi);
    for (int s = 0; s < cal.seasons; ++s) {
        const SeasonParams& p = params[s];
        const double lam = p.storm_frac > 0 ? -std::log(std::max(1e-9, 1.0 - p.storm_frac)) * dps / mean_dur : 0.0;
        const int64_t k = lam > 0 ? rng.poisson(lam) : 0;
        std::vector<int64_t> starts;
        for (int64_t q = 0; q < k; ++q) starts.push_back(rng.integers(static_cast<int64_t>(s) * dps, static_cast<int64_t>(s + 1) * dps));
        std::sort(starts.begin(), starts.end());
        for (int64_t st : starts) {
            ++eid;
            const int dur = static_cast<int>(rng.integers(dlo, dhi + 1));
            for (int64_t d = st; d < st + dur; ++d) storm_id[static_cast<size_t>(d % n)] = eid;
        }
    }
    std::vector<uint8_t> storm(n);
    bool any_storm = false;
    for (int d = 0; d < n; ++d) {
        storm[d] = storm_id[d] > 0 ? 1 : 0;
        any_storm = any_storm || storm[d];
    }
    const double smult = c.get("weather.storm_rain_mult");
    for (int d = 0; d < n; ++d) {
        const double gv = rng.gamma(shape, 1.0);
        if (storm[d]) precip[d] = gv * smult * mean_wet[d] / shape;
    }
    for (int d = 0; d < n; ++d) wet[d] = (wet[d] || storm[d]) ? 1 : 0;
    // 风：AR(1) 围绕季曲线；风暴日加强
    const double phi = c.get("weather.wind_ar1");
    std::vector<double> sig(n), eu(n, 0.0), ev(n, 0.0), z(2 * static_cast<size_t>(n));
    for (int d = 0; d < n; ++d) sig[d] = c.get("weather.wind_sigma_rel") * np_hypot(daily.wind_u[d], daily.wind_v[d]) + c.get("weather.wind_sigma_min");
    for (auto& x : z) x = rng.normal(0.0, 1.0);
    const double sq = std::sqrt(1 - phi * phi);
    for (int d = 1; d < n; ++d) {
        eu[d] = phi * eu[d - 1] + sq * sig[d] * z[2 * static_cast<size_t>(d)];
        ev[d] = phi * ev[d - 1] + sq * sig[d] * z[2 * static_cast<size_t>(d) + 1];
    }
    std::vector<double> u(n), v(n), speed(n), wind_from(n);
    for (int d = 0; d < n; ++d) {
        u[d] = daily.wind_u[d] + eu[d];
        v[d] = daily.wind_v[d] + ev[d];
        speed[d] = np_hypot(u[d], v[d]);
    }
    if (any_storm) {
        const double gm = c.get("weather.storm_wind_mult"), ga = c.get("weather.storm_wind_add");
        for (int d = 0; d < n; ++d)
            if (storm[d]) speed[d] = gm * std::max(speed[d], 1.0) + ga;
    }
    for (int d = 0; d < n; ++d) wind_from[d] = pymod(std::atan2(-u[d], -v[d]) * (180.0 / PI) + 360.0, 360.0);
    // 温度：季曲线 + AR(1)；雨日 / 风暴日偏凉
    const double phi_t = c.get("weather.temp_ar1"), sig_t = c.get("weather.temp_sigma_c");
    std::vector<double> et(n, 0.0), zt(n);
    for (auto& x : zt) x = rng.normal(0.0, 1.0);
    const double sqt = std::sqrt(1 - phi_t * phi_t);
    for (int d = 1; d < n; ++d) et[d] = phi_t * et[d - 1] + sqt * sig_t * zt[d];
    std::vector<double> temp(n);
    const double rc = c.get("weather.rain_cool_c"), sc = c.get("weather.storm_cool_c");
    for (int d = 0; d < n; ++d) temp[d] = daily.temp_c[d] + et[d] - rc * wet[d] - sc * storm[d];
    // 云海漫顶：低岛、静风、潮湿、非风暴
    std::vector<uint8_t> fog(n, 0);
    const double heavy_mm = c.get("weather.heavy_rain_mm");
    const double fmax = c.get("weather.fog_surface_max_m");
    if (surface_m < fmax) {
        const double low = 1.0 - std::min(1.0, std::max(0.0, (surface_m - 300.0) / std::max(1.0, fmax - 300.0)));
        const double p0 = c.get("weather.fog_p0"), fcalm = c.get("weather.fog_calm_ms");
        std::vector<double> p_fog(n);
        for (int d = 0; d < n; ++d) {
            const double prev = wet[(d + n - 1) % n] ? 1.0 : 0.0;
            const double humid = std::max(static_cast<double>(wet[d]), prev * 0.7) * 0.6 + params[season[d]].f_wet * 0.4;
            const double calm = clip(1.0 - speed[d] / fcalm, 0.0, 1.0);
            p_fog[d] = p0 * (0.3 + 0.7 * low) * calm * humid;
        }
        for (int d = 0; d < n; ++d) {
            const double uu = rng.uniform(0.0, 1.0);
            fog[d] = (uu < p_fog[d] && !storm[d] && precip[d] < heavy_mm) ? 1 : 0;
        }
    }
    std::vector<uint8_t> cloudy(n);
    const double ck = c.get("weather.cloudy_k");
    for (int d = 0; d < n; ++d) {
        const double uu = rng.uniform(0.0, 1.0);
        cloudy[d] = (!wet[d] && uu < params[season[d]].f_wet * ck) ? 1 : 0;
    }
    const double tr_off = lapse_c_per_km * (surface_m - rim_m) / 1000.0;
    const double snow_t = c.get("weather.snow_temp_c");
    Y.type.assign(n, 0);
    Y.temp_rim_c.resize(n);
    Y.snow.resize(n);
    for (int d = 0; d < n; ++d) {
        Y.temp_rim_c[d] = temp[d] + tr_off;
        const bool snowy = Y.temp_rim_c[d] <= snow_t;
        const bool heavy = precip[d] >= heavy_mm;
        int8_t t = 0;
        if (cloudy[d]) t = 1;
        if (wet[d] && !heavy) t = 2;
        if (wet[d] && heavy) t = 3;
        if (fog[d]) t = 4;
        if (storm[d]) t = 5;
        if (wet[d] && !heavy && snowy && !storm[d]) t = 6;
        if (wet[d] && heavy && snowy && !storm[d]) t = 7;
        if (storm[d] && snowy) t = 8;
        if (fog[d] && !storm[d]) t = 4;
        Y.type[d] = t;
        Y.snow[d] = (snowy && wet[d]) ? 1 : 0;
    }
    Y.sailable.resize(n);
    const double smax = c.get("weather.sail_wind_max_ms");
    for (int d = 0; d < n; ++d) {
        const double uu = rng.uniform(0.0, 1.0);
        Y.sailable[d] = (!storm[d] && speed[d] < smax && uu < params[season[d]].p_calm) ? 1 : 0;
    }
    Y.day.resize(n);
    Y.season.resize(n);
    Y.month.resize(n);
    Y.day_of_month.resize(n);
    for (int d = 0; d < n; ++d) {
        Y.day[d] = d;
        Y.season[d] = season[d];
        Y.month[d] = d / dpm;
        Y.day_of_month[d] = d % dpm + 1;
    }
    Y.precip_mm = std::move(precip);
    Y.temp_c = std::move(temp);
    Y.wind_from_deg = std::move(wind_from);
    Y.wind_ms = std::move(speed);
    Y.storm_event = std::move(storm_id);
    Y.wet = std::move(wet);
    return Y;
}

void multi_year(const NodeInputs& inp, const Climate& clim, const Daily& daily, const std::vector<SeasonParams>& params, double rim_m,
                const Config& c, int years, std::vector<double>& P, std::vector<double>& F) {
    const int n_s = clim.cal.seasons;
    P.assign(static_cast<size_t>(years) * n_s, 0.0);
    F.assign(static_cast<size_t>(years) * n_s, 0.0);
    for (int y = 0; y < years; ++y) {
        Rng r = part_rng(inp, "weather:" + std::to_string(y));
        const WeatherYear Y = simulate_year(r, clim, daily, params, inp.height_m, c, rim_m, inp.lapse_c_per_km);
        for (int s = 0; s < n_s; ++s) {
            std::vector<double> pv;
            int64_t wn = 0, cnt = 0;
            for (size_t d = 0; d < Y.season.size(); ++d)
                if (Y.season[d] == s) {
                    pv.push_back(Y.precip_mm[d]);
                    wn += Y.wet[d];
                    ++cnt;
                }
            P[static_cast<size_t>(y) * n_s + s] = np_sum(pv.data(), pv.size());
            F[static_cast<size_t>(y) * n_s + s] = static_cast<double>(wn) / static_cast<double>(cnt);
        }
    }
}

Json climate_json(const Climate& C) {
    Json j = Json::obj();
    Json cal = Json::obj();
    cal.set("seasons", C.cal.seasons);
    cal.set("months_per_season", C.cal.months_per_season);
    cal.set("days_per_month", C.cal.days_per_month);
    cal.set("days_per_season", C.cal.days_per_season);
    cal.set("year_days", C.cal.year_days);
    cal.set("day_offset_solstice_n", C.cal.offset);
    j.set("calendar", cal);
    j.set("season_type", C.stype);
    j.set("season_type_code", C.type_code);
    j.set("season_names", Json::arr_of(C.names));
    Json sc = Json::obj();
    sc.set("temp", C.score_temp);
    sc.set("rain", C.score_rain);
    sc.set("storm", C.score_storm);
    j.set("scores", sc);
    Json an = Json::obj();
    an.set("temp_c", C.a_temp);
    an.set("temp_sea_c", C.a_temp_sea);
    an.set("precip_rel", C.a_precip_rel);
    an.set("precip_mm", C.a_precip_mm);
    an.set("storm", C.a_storm);
    an.set("window", C.a_window);
    an.set("season_range_c", C.a_range);
    an.set("season_range_sea_c", C.a_range_sea);
    an.set("temp_winter_c", C.a_winter);
    an.set("temp_summer_c", C.a_summer);
    an.set("temp_ref_height_m", C.a_ref_h);
    j.set("annual", an);
    Json th = Json::obj();
    th.set("continentality_sea", C.cont_sea);
    th.set("continentality_island", C.cont_isl);
    th.set("tau_sea_days", C.tau_sea);
    th.set("tau_island_days", C.tau_isl);
    th.set("amplitude_retained_sea", C.A_sea);
    th.set("amplitude_retained_island", C.A_isl);
    th.set("lag_sea_days", C.lag_sea_days);
    th.set("lag_island_days", C.lag_isl_days);
    th.set("k_shift", C.k_shift);
    th.set("band_shift_amp_deg", C.band_amp);
    j.set("thermal", th);
    Json ss = Json::arr();
    for (const SeasonRec& s : C.seasons) {
        Json e = Json::obj();
        e.set("index", s.index);
        e.set("name", s.name);
        e.set("days", Json::ipair(s.d0, s.d1));
        e.set("mid_day", s.mid_day);
        e.set("months", Json::arr_of(s.months));
        e.set("temp_c", s.temp_c);
        e.set("temp_sea_c", s.temp_sea_c);
        e.set("precip_rel", s.precip_rel);
        e.set("precip_mm", s.precip_mm);
        e.set("precip_mm_annual_rate", s.precip_rate);
        e.set("storm", s.storm);
        e.set("window", s.window);
        Json w = Json::obj();
        w.set("u", s.wu);
        w.set("v", s.wv);
        w.set("speed_ms", s.speed);
        w.set("from_deg", s.from_deg);
        e.set("wind", w);
        e.set("band_shift_deg", s.band_shift);
        e.set("lat_sampled", s.lat_sampled);
        ss.push(e);
    }
    j.set("seasons", ss);
    Json mc = Json::obj();
    mc.set("precip_rel", C.m_precip);
    mc.set("storm", C.m_storm);
    mc.set("window", C.m_window);
    mc.set("temp_c", C.m_temp);
    j.set("means_check", mc);
    return j;
}

}  // namespace skyisle::island
