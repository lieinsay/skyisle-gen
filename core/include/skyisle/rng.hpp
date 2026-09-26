// 确定性随机流：与 numpy 的 SeedSequence + PCG64 + Generator 各分布逐位一致（docs/PLAN-CORE.md 第四节）。
// stage_rng(seed, k) = SeedSequence([seed, k])；entity_rng(seed, k, key) = SeedSequence([seed, k, crc32(key)])。
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace skyisle {

uint32_t crc32(const std::string& s);

// numpy.random.SeedSequence（池 4 字，uint32 运算）。entropy 里每个非负整数按 32 位小端拆字（0 → 一个 0 字）。
class SeedSequence {
public:
    explicit SeedSequence(const std::vector<uint64_t>& entropy);
    std::vector<uint32_t> generate_state_u32(size_t n_words) const;
    std::vector<uint64_t> generate_state_u64(size_t n_words) const;
    const std::array<uint32_t, 4>& pool() const { return pool_; }

private:
    std::array<uint32_t, 4> pool_{};
};

struct U128 {
    uint64_t hi = 0, lo = 0;
};

// numpy.random.PCG64：128 位 LCG，XSL-RR 输出；先步进后输出。next_u32 与 numpy 一样缓存半个 64 位字。
class Pcg64 {
public:
    explicit Pcg64(const SeedSequence& ss);
    uint64_t next_u64();
    uint32_t next_u32();
    double next_double() { return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

private:
    void step();
    U128 state_, inc_;
    bool has_u32_ = false;
    uint32_t u32_ = 0;
};

// numpy.random.Generator 用到的分布（numpy/random/src/distributions/distributions.c 同式）。
class Rng {
public:
    explicit Rng(const SeedSequence& ss) : bg_(ss) {}
    uint64_t raw() { return bg_.next_u64(); }
    double random() { return bg_.next_double(); }
    double uniform(double lo, double hi) { return lo + (hi - lo) * bg_.next_double(); }
    void uniform_fill(double lo, double hi, double* out, size_t n);
    // integers(lo, hi)：[lo, hi)，int64（numpy 默认 dtype）
    int64_t integers(int64_t lo, int64_t hi);
    double standard_normal();
    double normal(double loc, double scale) { return loc + scale * standard_normal(); }
    double standard_exponential();                  // numpy 的指数 ziggurat
    double standard_gamma(double shape);
    double gamma(double shape, double scale) { return scale * standard_gamma(shape); }
    double lognormal(double mean, double sigma) { return std::exp(normal(mean, sigma)); }
    int64_t poisson(double lam);                    // random_poisson：λ ≥ 10 走 PTRS，其余乘积法
    // choice(pop, size, replace=False)（shuffle=True）：Floyd + 洗牌；总体 > 10000 且 size > pop // 50 时尾部洗牌
    std::vector<int64_t> choice_noreplace(int64_t pop, int64_t size);
    void shuffle_int(int64_t* data, int64_t n, int64_t first);   // _shuffle_int：i = n−1 … first，与 [0, i] 里的一个换
    double beta(double a, double b);
    // choice(k, p=p)：cdf = cumsum(p) / cdf[-1]，u = random()，searchsorted(cdf, u, 'right')
    int64_t choice_p(const std::vector<double>& p);

    uint64_t bounded_u64(uint64_t off, uint64_t rng);   // random_bounded_uint64（Lemire，不用掩码）：[off, off + rng]
    static double loggam(double x);

private:
    Pcg64 bg_;
};

Rng stage_rng(uint64_t seed, uint64_t stage);
Rng entity_rng(uint64_t seed, uint64_t stage, const std::string& key);

}  // namespace skyisle
