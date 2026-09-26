// 与 numpy 逐位一致的随机流（SeedSequence / PCG64 / Generator 的分布）。
#include "skyisle/rng.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace skyisle {

namespace {

#include "ziggurat_tables.inc"

constexpr double ZIG_NOR_R = 3.6541528853610087963519472518;
constexpr double ZIG_NOR_INV_R = 0.27366123732975827203338247596;
constexpr double ZIG_EXP_R = 7.69711747013104972;

// ---------------------------------------------------------------- 128 位运算
inline void mul64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
#if defined(_MSC_VER) && defined(_M_X64)
    lo = _umul128(a, b, &hi);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    hi = static_cast<uint64_t>(p >> 64);
    lo = static_cast<uint64_t>(p);
#else
    uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);
    lo = (mid << 32) | (p00 & 0xFFFFFFFFu);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}

inline U128 add(U128 a, U128 b) {
    U128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1 : 0);
    return r;
}

inline U128 mul(U128 a, U128 b) {
    U128 r;
    mul64(a.lo, b.lo, r.hi, r.lo);
    r.hi += a.hi * b.lo + a.lo * b.hi;
    return r;
}

const U128 PCG_MULT{0x2360ED051FC65DA4ULL, 0x4385DF649FCCF645ULL};

// ---------------------------------------------------------------- SeedSequence 常数
constexpr uint32_t INIT_A = 0x43b0d7e5u, MULT_A = 0x931e8875u, INIT_B = 0x8b51f9ddu, MULT_B = 0x58f38dedu;
constexpr uint32_t MIX_MULT_L = 0xca01f9ddu, MIX_MULT_R = 0x4973f715u;

inline uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= MULT_A;
    value *= hash_const;
    value ^= value >> 16;
    return value;
}

inline uint32_t mix(uint32_t x, uint32_t y) {
    uint32_t r = MIX_MULT_L * x - MIX_MULT_R * y;
    r ^= r >> 16;
    return r;
}

}  // namespace

uint32_t crc32(const std::string& s) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (unsigned char ch : s) c = table[(c ^ ch) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

SeedSequence::SeedSequence(const std::vector<uint64_t>& entropy) {
    std::vector<uint32_t> ent;
    for (uint64_t e : entropy) {
        if (e == 0) {
            ent.push_back(0);
            continue;
        }
        while (e > 0) {
            ent.push_back(static_cast<uint32_t>(e & 0xFFFFFFFFu));
            e >>= 32;
        }
    }
    uint32_t hc = INIT_A;
    for (size_t i = 0; i < pool_.size(); ++i) pool_[i] = hashmix(i < ent.size() ? ent[i] : 0u, hc);
    for (size_t s = 0; s < pool_.size(); ++s)
        for (size_t d = 0; d < pool_.size(); ++d)
            if (s != d) pool_[d] = mix(pool_[d], hashmix(pool_[s], hc));
    for (size_t s = pool_.size(); s < ent.size(); ++s)
        for (size_t d = 0; d < pool_.size(); ++d) pool_[d] = mix(pool_[d], hashmix(ent[s], hc));
}

std::vector<uint32_t> SeedSequence::generate_state_u32(size_t n_words) const {
    std::vector<uint32_t> out(n_words);
    uint32_t hc = INIT_B;
    for (size_t i = 0; i < n_words; ++i) {
        uint32_t v = pool_[i % pool_.size()];
        v ^= hc;
        hc *= MULT_B;
        v *= hc;
        v ^= v >> 16;
        out[i] = v;
    }
    return out;
}

std::vector<uint64_t> SeedSequence::generate_state_u64(size_t n_words) const {
    std::vector<uint32_t> w = generate_state_u32(2 * n_words);
    std::vector<uint64_t> out(n_words);
    for (size_t i = 0; i < n_words; ++i) out[i] = static_cast<uint64_t>(w[2 * i]) | (static_cast<uint64_t>(w[2 * i + 1]) << 32);
    return out;
}

Pcg64::Pcg64(const SeedSequence& ss) {
    std::vector<uint64_t> v = ss.generate_state_u64(4);
    U128 initstate{v[0], v[1]};
    U128 initseq{v[2], v[3]};
    inc_.hi = (initseq.hi << 1) | (initseq.lo >> 63);
    inc_.lo = (initseq.lo << 1) | 1u;
    state_ = U128{0, 0};
    step();
    state_ = add(state_, initstate);
    step();
}

void Pcg64::step() { state_ = add(mul(state_, PCG_MULT), inc_); }

uint64_t Pcg64::next_u64() {
    step();
    uint64_t x = state_.hi ^ state_.lo;
    unsigned rot = static_cast<unsigned>(state_.hi >> 58);
    return (x >> rot) | (x << ((64u - rot) & 63u));
}

uint32_t Pcg64::next_u32() {
    if (has_u32_) {
        has_u32_ = false;
        return u32_;
    }
    uint64_t next = next_u64();
    has_u32_ = true;
    u32_ = static_cast<uint32_t>(next >> 32);
    return static_cast<uint32_t>(next & 0xFFFFFFFFu);
}

void Rng::uniform_fill(double lo, double hi, double* out, size_t n) {
    const double range = hi - lo;
    for (size_t i = 0; i < n; ++i) out[i] = lo + range * bg_.next_double();
}

uint64_t Rng::bounded_u64(uint64_t off, uint64_t rng) {
    // random_bounded_uint64_fill（use_masked = false）
    if (rng == 0) return off;
    if (rng <= 0xFFFFFFFFULL) {
        if (rng == 0xFFFFFFFFULL) return off + bg_.next_u32();
        const uint32_t rng_excl = static_cast<uint32_t>(rng) + 1u;
        uint64_t m = static_cast<uint64_t>(bg_.next_u32()) * rng_excl;
        uint32_t leftover = static_cast<uint32_t>(m & 0xFFFFFFFFu);
        if (leftover < rng_excl) {
            const uint32_t threshold = (0xFFFFFFFFu - static_cast<uint32_t>(rng)) % rng_excl;
            while (leftover < threshold) {
                m = static_cast<uint64_t>(bg_.next_u32()) * rng_excl;
                leftover = static_cast<uint32_t>(m & 0xFFFFFFFFu);
            }
        }
        return off + (m >> 32);
    }
    if (rng == 0xFFFFFFFFFFFFFFFFULL) return off + bg_.next_u64();
    const uint64_t rng_excl = rng + 1;
    uint64_t hi, lo;
    mul64(bg_.next_u64(), rng_excl, hi, lo);
    if (lo < rng_excl) {
        const uint64_t threshold = (0xFFFFFFFFFFFFFFFFULL - rng) % rng_excl;
        while (lo < threshold) mul64(bg_.next_u64(), rng_excl, hi, lo);
    }
    return off + hi;
}

int64_t Rng::integers(int64_t lo, int64_t hi) {
    if (hi <= lo) throw std::invalid_argument("integers: high <= low");
    const uint64_t rng = static_cast<uint64_t>(hi - 1 - lo);
    return static_cast<int64_t>(bounded_u64(static_cast<uint64_t>(lo), rng));
}

double Rng::standard_normal() {
    for (;;) {
        uint64_t r = bg_.next_u64();
        const int idx = static_cast<int>(r & 0xff);
        r >>= 8;
        const int sign = static_cast<int>(r & 0x1);
        const uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
        double x = static_cast<double>(rabs) * ZIG_WI[idx];
        if (sign & 0x1) x = -x;
        if (rabs < ZIG_KI[idx]) return x;
        if (idx == 0) {
            for (;;) {
                const double xx = -ZIG_NOR_INV_R * std::log1p(-bg_.next_double());
                const double yy = -std::log1p(-bg_.next_double());
                if (yy + yy > xx * xx) return ((rabs >> 8) & 0x1) ? -(ZIG_NOR_R + xx) : ZIG_NOR_R + xx;
            }
        } else {
            if (((ZIG_FI[idx - 1] - ZIG_FI[idx]) * bg_.next_double() + ZIG_FI[idx]) < std::exp(-0.5 * x * x)) return x;
        }
    }
}

double Rng::standard_exponential() {
    // numpy 的 256 层 ziggurat（random_standard_exponential）：ri = 原始 >> 3，低 8 位是层号，其上 53 位；表是探出来的（probe_ziggurat.py）
    for (;;) {
        uint64_t ri = bg_.next_u64();
        ri >>= 3;
        const int idx = static_cast<int>(ri & 0xFF);
        ri >>= 8;
        const double x = static_cast<double>(ri) * ZIG_WE[idx];
        if (ri < ZIG_KE[idx]) return x;
        if (idx == 0) return ZIG_EXP_R - std::log1p(-bg_.next_double());
        if ((ZIG_FE[idx - 1] - ZIG_FE[idx]) * bg_.next_double() + ZIG_FE[idx] < std::exp(-x)) return x;
    }
}

int64_t Rng::poisson(double lam) {
    // random_poisson：λ ≥ 10 走 PTRS（Hörmann 1993），λ = 0 得 0，其余乘积法
    if (lam >= 10) {
        const double slam = std::sqrt(lam), loglam = std::log(lam);
        const double b = 0.931 + 2.53 * slam;
        const double a = -0.059 + 0.02483 * b;
        const double invalpha = 1.1239 + 1.1328 / (b - 3.4);
        const double vr = 0.9277 - 3.6224 / (b - 2);
        for (;;) {
            const double U = bg_.next_double() - 0.5;
            const double V = bg_.next_double();
            const double us = 0.5 - std::fabs(U);
            const int64_t k = static_cast<int64_t>(std::floor((2 * a / us + b) * U + lam + 0.43));
            if ((us >= 0.07) && (V <= vr)) return k;
            if ((k < 0) || ((us < 0.013) && (V > us))) continue;
            if ((std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b)) <= (-lam + static_cast<double>(k) * loglam - loggam(static_cast<double>(k + 1))))
                return k;
        }
    }
    if (lam == 0) return 0;
    const double enlam = std::exp(-lam);
    int64_t X = 0;
    double prod = 1.0;
    for (;;) {
        prod *= bg_.next_double();
        if (prod > enlam) X += 1;
        else return X;
    }
}

double Rng::loggam(double x) {
    // random_loggam（distributions.c）
    static const double a[10] = {8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04, -5.952380952380952e-04,
                                 8.417508417508418e-04, -1.917526917526918e-03, 6.410256410256410e-03, -2.955065359477124e-02,
                                 1.796443723688307e-01, -1.39243221690590e+00};
    if ((x == 1.0) || (x == 2.0)) return 0.0;
    int64_t n = 0;
    if (x < 7.0) n = static_cast<int64_t>(7 - x);
    double x0 = x + static_cast<double>(n);
    const double x2 = (1.0 / x0) * (1.0 / x0);
    const double lg2pi = 1.8378770664093453e+00;
    double gl0 = a[9];
    for (int k = 8; k >= 0; k--) {
        gl0 *= x2;
        gl0 += a[k];
    }
    double gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * std::log(x0) - x0;
    if (x < 7.0) {
        for (int64_t k = 1; k <= n; k++) {
            gl -= std::log(x0 - 1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

std::vector<int64_t> Rng::choice_noreplace(int64_t pop, int64_t size) {
    // Generator.choice(pop, size, replace=False, shuffle=True)：总体 > 10000 且 size > pop // 50 时尾部洗牌，否则 Floyd + 洗牌
    std::vector<int64_t> idx;
    if (size <= 0) return idx;
    if (size > pop) throw std::invalid_argument("choice: size > pop");
    if (pop > 10000 && size > pop / 50) {
        std::vector<int64_t> all(static_cast<size_t>(pop));
        for (int64_t i = 0; i < pop; ++i) all[i] = i;
        shuffle_int(all.data(), pop, std::max<int64_t>(pop - size, 1));
        idx.assign(all.begin() + (pop - size), all.end());
        return idx;
    }
    idx.resize(static_cast<size_t>(size));
    uint64_t mask = static_cast<uint64_t>(1.2 * static_cast<double>(size));
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    const uint64_t EMPTY = ~0ULL;
    std::vector<uint64_t> hs(static_cast<size_t>(mask + 1), EMPTY);
    for (int64_t j = pop - size; j < pop; ++j) {
        const uint64_t val = bounded_u64(0, static_cast<uint64_t>(j));
        uint64_t loc = val & mask;
        while (hs[loc] != EMPTY && hs[loc] != val) loc = (loc + 1) & mask;
        if (hs[loc] == EMPTY) {
            hs[loc] = val;
            idx[j - pop + size] = static_cast<int64_t>(val);
        } else {
            loc = static_cast<uint64_t>(j) & mask;
            while (hs[loc] != EMPTY) loc = (loc + 1) & mask;
            hs[loc] = static_cast<uint64_t>(j);
            idx[j - pop + size] = j;
        }
    }
    shuffle_int(idx.data(), size, 1);
    return idx;
}

void Rng::shuffle_int(int64_t* data, int64_t n, int64_t first) {
    for (int64_t i = n - 1; i >= first; --i) {
        const int64_t j = static_cast<int64_t>(bounded_u64(0, static_cast<uint64_t>(i)));
        std::swap(data[j], data[i]);
    }
}

double Rng::standard_gamma(double shape) {
    if (shape == 1.0) return standard_exponential();
    if (shape == 0.0) return 0.0;
    if (shape < 1.0) {
        for (;;) {
            const double U = bg_.next_double();
            const double V = standard_exponential();
            if (U <= 1.0 - shape) {
                const double X = std::pow(U, 1.0 / shape);
                if (X <= V) return X;
            } else {
                const double Y = -std::log((1 - U) / shape);
                const double X = std::pow(1.0 - shape + shape * Y, 1.0 / shape);
                if (X <= (V + Y)) return X;
            }
        }
    }
    const double b = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9 * b);
    for (;;) {
        double X, V;
        do {
            X = standard_normal();
            V = 1.0 + c * X;
        } while (V <= 0.0);
        V = V * V * V;
        const double U = bg_.next_double();
        if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return b * V;
        if (std::log(U) < 0.5 * X * X + b * (1.0 - V + std::log(V))) return b * V;
    }
}

double Rng::beta(double a, double b) {
    if (a <= 1.0 && b <= 1.0) {
        for (;;) {
            const double U = bg_.next_double();
            const double V = bg_.next_double();
            const double X = std::pow(U, 1.0 / a);
            const double Y = std::pow(V, 1.0 / b);
            const double XpY = X + Y;
            if (XpY <= 1.0 && U + V > 0.0) {
                if (XpY > 0) return X / XpY;
                double logX = std::log(U) / a;
                double logY = std::log(V) / b;
                const double logM = logX > logY ? logX : logY;
                logX -= logM;
                logY -= logM;
                return std::exp(logX - std::log(std::exp(logX) + std::exp(logY)));
            }
        }
    }
    const double Ga = standard_gamma(a);
    const double Gb = standard_gamma(b);
    return Ga / (Ga + Gb);
}

int64_t Rng::choice_p(const std::vector<double>& p) {
    std::vector<double> cdf(p.size());
    double acc = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        acc += p[i];
        cdf[i] = acc;
    }
    const double last = cdf.back();
    for (double& v : cdf) v /= last;
    const double u = bg_.next_double();
    // searchsorted side='right'：第一个 cdf[i] > u 的下标
    size_t lo = 0, hi = cdf.size();
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid + 1;
        else hi = mid;
    }
    return static_cast<int64_t>(lo);
}

Rng stage_rng(uint64_t seed, uint64_t stage) { return Rng(SeedSequence({seed, stage})); }

Rng entity_rng(uint64_t seed, uint64_t stage, const std::string& key) {
    return Rng(SeedSequence({seed, stage, static_cast<uint64_t>(crc32(key))}));
}

}  // namespace skyisle
