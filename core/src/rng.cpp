// 与 numpy 逐位一致的随机流（SeedSequence / PCG64 / Generator 的分布）。
#include "skyisle/rng.hpp"

#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace skyisle {

namespace {

#include "ziggurat_tables.inc"

constexpr double ZIG_NOR_R = 3.6541528853610087963519472518;
constexpr double ZIG_NOR_INV_R = 0.27366123732975827203338247596;

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
    // 第三层用不到指数分布的 ziggurat（gamma 形状 = 1 才走这里）；按反函数给出，不追 numpy 的位。
    return -std::log1p(-bg_.next_double());
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
