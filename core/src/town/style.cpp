#include "skyisle/town/style.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "skyisle/grid.hpp"

namespace skyisle::town {

namespace {

constexpr double DEG = PI / 180.0;

Range range_of(const Config& c, const std::string& k) {
    auto it = c.vec.find(k);
    if (it != c.vec.end()) {
        if (it->second.size() != 2) throw std::invalid_argument("style: " + k + " should be [lo, hi]");
        return {it->second[0], it->second[1]};
    }
    const double v = c.get(k);   // 单个数 = 退化区间
    return {v, v};
}

Range range_or(const Config& c, const std::string& k, Range fb) {
    if (c.vec.count(k) || c.num.count(k)) return range_of(c, k);
    return fb;
}

bool flag(const Config& c, const std::string& k, bool fb) { return c.get(k, fb ? 1.0 : 0.0) != 0.0; }

// prefix 下一层的子键名（"style.village.operators." → fishbone、organic），三张表都看
std::vector<std::string> children(const Config& c, const std::string& prefix) {
    std::vector<std::string> out;
    auto scan = [&](const auto& m) {
        for (auto it = m.lower_bound(prefix); it != m.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it) {
            std::string rest = it->first.substr(prefix.size());
            const size_t dot = rest.find('.');
            if (dot != std::string::npos) rest = rest.substr(0, dot);
            if (std::find(out.begin(), out.end(), rest) == out.end()) out.push_back(rest);
        }
    };
    scan(c.num);
    scan(c.vec);
    scan(c.str);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::pair<std::string, double>> weights(const Config& c, const std::string& prefix) {
    std::vector<std::pair<std::string, double>> out;
    for (const std::string& k : children(c, prefix))
        if (c.num.count(prefix + k)) out.emplace_back(k, c.num.at(prefix + k));
    return out;
}

int side_of(const std::string& s) {
    if (s == "front") return SIDE_FRONT;
    if (s == "back") return SIDE_BACK;
    if (s == "left") return SIDE_LEFT;
    if (s == "right") return SIDE_RIGHT;
    throw std::invalid_argument("style: unknown side " + s);
}

int face_of(const std::string& s) {
    if (s == "front") return FACE_FRONT;
    if (s == "in") return FACE_IN;
    if (s == "out") return FACE_OUT;
    throw std::invalid_argument("style: unknown face " + s);
}

}  // namespace

int Range::sample_int(Rng& r, bool odd) const {
    const int lo_i = static_cast<int>(std::lround(lo)), hi_i = static_cast<int>(std::lround(hi));
    if (!odd) return hi_i > lo_i ? static_cast<int>(r.integers(lo_i, hi_i + 1)) : lo_i;
    std::vector<int> c;
    for (int k = lo_i; k <= hi_i; ++k)
        if (k % 2 != 0) c.push_back(k);
    if (c.empty()) return lo_i % 2 != 0 ? lo_i : std::max(1, lo_i - 1);
    return c.size() == 1 ? c[0] : c[static_cast<size_t>(r.integers(0, static_cast<int64_t>(c.size())))];
}

double FuncSpec::getn(const std::string& k, double fb) const {
    auto it = num.find(k);
    return it == num.end() ? fb : it->second;
}

Range FuncSpec::getr(const std::string& k, Range fb) const {
    auto it = range.find(k);
    if (it != range.end()) return it->second;
    auto jt = num.find(k);
    return jt == num.end() ? fb : Range{jt->second, jt->second};
}

std::string FuncSpec::gets(const std::string& k, const std::string& fb) const {
    auto it = str.find(k);
    return it == str.end() ? fb : it->second;
}

int FuncSpec::count(int households) const {
    if (households < count_min_hh) return 0;
    int n = count_base + (count_per_hh > 0 ? static_cast<int>(std::floor(households / count_per_hh)) : 0);
    return std::clamp(n, 0, count_max);
}

int Style::template_index(const std::string& tid) const {
    for (size_t k = 0; k < templates.size(); ++k)
        if (templates[k].id == tid) return static_cast<int>(k);
    return -1;
}

const FuncSpec* Style::func(const std::string& fid) const {
    for (const FuncSpec& f : funcs)
        if (f.id == fid) return &f;
    return nullptr;
}

std::vector<int> Style::house_templates() const {
    std::vector<int> out;
    for (size_t k = 0; k < templates.size(); ++k)
        if (templates[k].weight > 0.0) out.push_back(static_cast<int>(k));
    return out;
}

Style parse_style(const Config& c) {
    const std::string P = "style.";
    auto g = [&](const std::string& k) { return c.get(P + k); };
    auto r = [&](const std::string& k) { return range_of(c, P + k); };
    Style s;
    s.id = c.gets(P + "meta.id", "custom");

    s.center_search_m = g("site.center_search_m");
    s.water_scale_m = g("site.water_scale_m");
    s.edge_scale_m = g("site.edge_scale_m");
    s.w_flat = g("site.weights.flat");
    s.w_water = g("site.weights.water");
    s.w_dry = g("site.weights.dry");
    s.w_sun = g("site.weights.sun");
    s.w_farmland = g("site.weights.farmland");
    s.w_edge = g("site.weights.edge");
    s.w_shelter = g("site.weights.shelter");

    s.max_slope_deg = g("ground.max_slope_deg");
    s.max_cut_m = g("ground.max_cut_m");
    s.max_terrace_m = g("ground.max_terrace_m");
    s.allow_flood = flag(c, P + "ground.allow_flood", false);

    s.operators = weights(c, P + "village.operators.");
    s.hh_per_compound = r("village.households_per_compound");
    s.founders = r("village.founders");
    s.kin_p = g("village.kin_p");

    s.trunk_w = r("street.trunk_width_m");
    s.main_w = r("street.main_width_m");
    s.street_w = r("street.street_width_m");
    s.lane_w = r("street.lane_width_m");
    s.path_w = r("street.path_width_m");
    s.plot_gap = r("street.plot_gap_m");
    s.setback = r("street.setback_m");
    s.max_grade = g("street.max_grade");
    s.bridge_max_m = g("street.bridge_max_m");
    s.exits = static_cast<int>(g("street.exits"));

    s.fb_street_spacing = r("fishbone.street_spacing_m");
    s.fb_lane_spacing = r("fishbone.lane_spacing_m");
    s.fb_warp_m = g("fishbone.warp_m");
    s.fb_warp_wl = g("fishbone.warp_wavelength_m");
    s.fb_extent = g("fishbone.extent");
    s.fb_vacant_p = g("fishbone.vacant_p");
    s.fb_main_cross = flag(c, P + "fishbone.main_cross", true);
    s.fb_lane_extend = r("fishbone.lane_extend_m");
    s.fb_lane_extend_p = g("fishbone.lane_extend_p");

    s.og_candidates = static_cast<int>(g("organic.candidates"));
    s.og_gap = r("organic.gap_m");
    s.og_jitter = g("organic.jitter_deg") * DEG;
    s.og_stagger_m = g("organic.stagger_m");
    s.og_lane_reach_m = g("organic.lane_reach_m");
    s.og_main_street = flag(c, P + "organic.main_street", true);

    s.g_dist = g("growth.w_dist");
    s.g_net = g("growth.w_net");
    s.g_interest = g("growth.w_interest");
    s.g_kin = g("growth.w_kin");
    s.g_contact = g("growth.w_contact");
    s.g_noise = g("growth.noise");
    s.g_main_discount = g("growth.main_discount");

    s.snap = g("orient.snap_deg") * DEG;
    s.sun_offset = g("orient.sun_offset_deg") * DEG;
    const int nr = static_cast<int>(c.get(P + "orient.rule.n", 0.0));
    for (int k = 0; k < nr; ++k) {
        const std::string q = P + "orient.rule." + std::to_string(k) + ".";
        OrientRule o;
        o.kind = c.gets(q + "kind");
        o.weight = c.get(q + "weight", 1.0);
        o.tol = c.get(q + "tol_deg", 30.0) * DEG;
        o.angle = c.get(q + "angle_deg", 0.0) * DEG;
        s.rules.push_back(o);
    }
    s.gate_front = c.gets(P + "gate.front");
    s.gate_back = c.gets(P + "gate.back");
    s.gate_side = c.gets(P + "gate.side");
    s.gate_w = r("gate.width_m");

    s.wall_thickness_m = g("compound.wall_thickness_m");
    s.yard_min_m = g("compound.yard_min_m");
    const int nt = static_cast<int>(c.get(P + "compound.template.n", 0.0));
    for (int k = 0; k < nt; ++k) {
        const std::string q = P + "compound.template." + std::to_string(k) + ".";
        TemplateSpec t;
        t.id = c.gets(q + "id");
        t.name = c.gets(q + "name", t.id);
        t.kind = c.gets(q + "kind", "house");
        t.weight = c.get(q + "weight", 0.0);
        t.plot_w = range_of(c, q + "plot_w_m");
        t.plot_d = range_of(c, q + "plot_d_m");
        t.bay_m = range_or(c, q + "bay_m", {3.0, 3.6});
        t.wall = flag(c, q + "wall", true);
        t.gate_house = flag(c, q + "gate_house", false);
        t.gate = c.gets(q + "gate", "");
        const int nb = static_cast<int>(c.get(q + "b.n", 0.0));
        for (int j = 0; j < nb; ++j) {
            const std::string u = q + "b." + std::to_string(j) + ".";
            BuildingSpec b;
            b.role = c.gets(u + "role", "main");
            b.name = c.gets(u + "name", b.role);
            b.func = c.gets(u + "func", "dwelling");
            b.roof = c.gets(u + "roof", "gable");
            b.material = c.gets(u + "material", "");
            b.attach = c.gets(u + "attach", "");
            b.side = side_of(c.gets(u + "side", "back"));
            b.bays = range_or(c, u + "bays", {0, 0});
            b.odd = flag(c, u + "odd", false);
            b.len_frac = range_or(c, u + "len_frac", {1.0, 1.0});
            b.depth_m = range_of(c, u + "depth_m");
            b.pos = c.get(u + "pos", 0.5);
            b.face = face_of(c.gets(u + "face", "in"));
            b.prob = c.get(u + "prob", 1.0);
            b.storeys = static_cast<int>(c.get(u + "storeys", 1.0));
            b.eave_m = c.get(u + "eave_m", 3.0);
            b.pitch_deg = c.get(u + "pitch_deg", 30.0);
            b.plinth_m = c.get(u + "plinth_m", 0.3);
            t.b.push_back(b);
        }
        s.templates.push_back(t);
    }

    const int nf = static_cast<int>(c.get(P + "func.n", 0.0));
    for (int k = 0; k < nf; ++k) {
        const std::string q = P + "func." + std::to_string(k) + ".";
        FuncSpec f;
        f.id = c.gets(q + "id");
        f.name = c.gets(q + "name", f.id);
        f.mode = c.gets(q + "mode");
        f.tmpl = c.gets(q + "template", "");
        f.count_base = static_cast<int>(c.get(q + "count.base", 1.0));
        f.count_per_hh = c.get(q + "count.per_households", 0.0);
        f.count_min_hh = static_cast<int>(c.get(q + "count.min_households", 0.0));
        f.count_max = static_cast<int>(c.get(q + "count.max", 1.0));
        f.required = flag(c, q + "required", false);
        f.site = weights(c, q + "site.");
        for (const std::string& ch : children(c, q)) {
            const std::string key = q + ch;
            if (c.num.count(key)) f.num[ch] = c.num.at(key);
            if (c.vec.count(key) && c.vec.at(key).size() == 2) f.range[ch] = {c.vec.at(key)[0], c.vec.at(key)[1]};
            if (c.str.count(key)) f.str[ch] = c.str.at(key);
        }
        if (f.mode == "compound" && s.template_index(f.tmpl) < 0) throw std::invalid_argument("style: func " + f.id + " uses unknown template " + f.tmpl);
        s.funcs.push_back(f);
    }

    for (const std::string& k : children(c, P + "targets.")) {
        const std::string key = P + "targets." + k;
        if (c.vec.count(key) && c.vec.at(key).size() == 2) s.targets[k] = {c.vec.at(key)[0], c.vec.at(key)[1]};
    }
    if (s.house_templates().empty()) throw std::invalid_argument("style: no house template (weight > 0)");
    if (s.operators.empty()) throw std::invalid_argument("style: no village operator");
    return s;
}

}  // namespace skyisle::town
