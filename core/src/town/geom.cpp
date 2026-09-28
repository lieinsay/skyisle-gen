#include "skyisle/town/geom.hpp"

#include <algorithm>
#include <limits>

namespace skyisle::town {

namespace {
constexpr double PI = 3.141592653589793;
}

double wrap_pi(double a) {
    a = std::fmod(a + PI, 2.0 * PI);
    if (a <= 0.0) a += 2.0 * PI;
    return a - PI;
}

double angle_diff(double a, double b) { return std::fabs(wrap_pi(a - b)); }

std::array<V2, 4> corners(const Obb& o) {
    const V2 f = bearing_vec(o.facing) * o.hd;
    const V2 r = bearing_right(o.facing) * o.hw;
    return {o.c + f + r, o.c + f - r, o.c - f - r, o.c - f + r};
}

bool overlap(const Obb& a, const Obb& b, double clearance) {
    Obb A = a, B = b;
    const double e = 0.5 * clearance;   // 负数 = 各自内缩（贴边不算相交，留一点浮点余量）
    A.hw = std::max(0.0, A.hw + e), A.hd = std::max(0.0, A.hd + e), B.hw = std::max(0.0, B.hw + e), B.hd = std::max(0.0, B.hd + e);
    const std::array<V2, 4> ca = corners(A), cb = corners(B);
    const V2 axes[4] = {bearing_vec(A.facing), bearing_right(A.facing), bearing_vec(B.facing), bearing_right(B.facing)};
    for (const V2& ax : axes) {
        double amin = std::numeric_limits<double>::infinity(), amax = -amin, bmin = amin, bmax = -amin;
        for (const V2& p : ca) {
            const double d = dot(p, ax);
            amin = std::min(amin, d), amax = std::max(amax, d);
        }
        for (const V2& p : cb) {
            const double d = dot(p, ax);
            bmin = std::min(bmin, d), bmax = std::max(bmax, d);
        }
        if (amax <= bmin || bmax <= amin) return false;
    }
    return true;
}

bool contains(const Obb& o, V2 p) {
    const V2 d = p - o.c;
    return std::fabs(dot(d, bearing_vec(o.facing))) <= o.hd && std::fabs(dot(d, bearing_right(o.facing))) <= o.hw;
}

double polygon_area(const std::vector<V2>& p) {
    double s = 0.0;
    const size_t n = p.size();
    for (size_t i = 0; i < n; ++i) s += cross(p[i], p[(i + 1) % n]);
    return 0.5 * s;
}

V2 polygon_centroid(const std::vector<V2>& p) {
    const size_t n = p.size();
    double a = 0.0, cx = 0.0, cy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const V2 u = p[i], v = p[(i + 1) % n];
        const double w = cross(u, v);
        a += w, cx += (u.x + v.x) * w, cy += (u.y + v.y) * w;
    }
    if (std::fabs(a) < 1e-12) {
        V2 m;
        for (const V2& q : p) m = m + q;
        return n ? m * (1.0 / static_cast<double>(n)) : m;
    }
    return {cx / (3.0 * a), cy / (3.0 * a)};
}

bool point_in_polygon(const std::vector<V2>& p, V2 q) {
    bool in = false;
    const size_t n = p.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((p[i].y > q.y) != (p[j].y > q.y)) && (q.x < (p[j].x - p[i].x) * (q.y - p[i].y) / (p[j].y - p[i].y) + p[i].x)) in = !in;
    }
    return in;
}

double dist_point_segment(V2 p, V2 a, V2 b, double* t) {
    const V2 ab = b - a;
    const double L2 = dot(ab, ab);
    double s = L2 > 0.0 ? dot(p - a, ab) / L2 : 0.0;
    s = std::clamp(s, 0.0, 1.0);
    if (t) *t = s;
    return len(p - (a + ab * s));
}

double dist_point_polyline(V2 p, const std::vector<V2>& line, int* seg, double* t) {
    double best = std::numeric_limits<double>::infinity();
    if (line.size() == 1) {
        if (seg) *seg = 0;
        if (t) *t = 0.0;
        return len(p - line[0]);
    }
    for (size_t i = 0; i + 1 < line.size(); ++i) {
        double s = 0.0;
        const double d = dist_point_segment(p, line[i], line[i + 1], &s);
        if (d < best) {
            best = d;
            if (seg) *seg = static_cast<int>(i);
            if (t) *t = s;
        }
    }
    return best;
}

double polyline_length(const std::vector<V2>& line) {
    double s = 0.0;
    for (size_t i = 0; i + 1 < line.size(); ++i) s += len(line[i + 1] - line[i]);
    return s;
}

void chaikin(std::vector<V2>& line, std::vector<std::vector<double>>* attrs, int iterations) {
    for (int it = 0; it < iterations && line.size() >= 3; ++it) {
        std::vector<V2> out;
        std::vector<std::vector<double>> aout;
        out.reserve(line.size() * 2);
        out.push_back(line.front());
        if (attrs) aout.push_back(attrs->front());
        for (size_t i = 0; i + 1 < line.size(); ++i) {
            out.push_back(lerp(line[i], line[i + 1], 0.25));
            out.push_back(lerp(line[i], line[i + 1], 0.75));
            if (attrs) {
                const std::vector<double>& a = (*attrs)[i];
                const std::vector<double>& b = (*attrs)[i + 1];
                std::vector<double> q(a.size()), r(a.size());
                for (size_t k = 0; k < a.size(); ++k) q[k] = a[k] + 0.25 * (b[k] - a[k]), r[k] = a[k] + 0.75 * (b[k] - a[k]);
                aout.push_back(q);
                aout.push_back(r);
            }
        }
        out.push_back(line.back());
        if (attrs) aout.push_back(attrs->back());
        line.swap(out);
        if (attrs) attrs->swap(aout);
    }
}

namespace {
void dp_rec(const std::vector<V2>& line, size_t a, size_t b, double eps, std::vector<char>& keep) {
    if (b <= a + 1) return;
    double best = -1.0;
    size_t idx = a;
    for (size_t i = a + 1; i < b; ++i) {
        const double d = dist_point_segment(line[i], line[a], line[b]);
        if (d > best) best = d, idx = i;
    }
    if (best > eps) {
        keep[idx] = 1;
        dp_rec(line, a, idx, eps, keep);
        dp_rec(line, idx, b, eps, keep);
    }
}
}  // namespace

std::vector<V2> douglas_peucker(const std::vector<V2>& line, double eps) {
    if (line.size() < 3) return line;
    std::vector<char> keep(line.size(), 0);
    keep.front() = keep.back() = 1;
    dp_rec(line, 0, line.size() - 1, eps, keep);
    std::vector<V2> out;
    for (size_t i = 0; i < line.size(); ++i)
        if (keep[i]) out.push_back(line[i]);
    return out;
}

}  // namespace skyisle::town
