// C++ 自检（ctest）：随机流对 numpy 的参考值、栅格公共件的小例子。不依赖 Python。
// 参考值来自 numpy 2.x：np.random.PCG64(np.random.SeedSequence([42, 21, zlib.crc32(b"island:2051:layout")])).random_raw(3) 等。
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "skyisle/flow.hpp"
#include "skyisle/grid.hpp"
#include "skyisle/rng.hpp"

using namespace skyisle;

#define REF_RAW0 4869714485165228175ULL
#define REF_RAW1 11149298506873232026ULL
#define REF_RAW2 14528220131426250776ULL
#define REF_NORMAL0 0x1.7787311499fd8p-3
#define REF_INT1 6
#define REF_BETA2 0x1.250d91b12be7bp-2

static int fails = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

int main() {
    CHECK(crc32("island:2051:layout") == 4019611777u);
    {
        Rng r = entity_rng(42, 21, "island:2051:layout");
        const uint64_t ref[3] = {REF_RAW0, REF_RAW1, REF_RAW2};
        for (uint64_t x : ref) CHECK(r.raw() == x);
    }
    {
        Rng r = entity_rng(42, 21, "island:2051:layout");
        CHECK(r.standard_normal() == REF_NORMAL0);
        CHECK(r.integers(0, 10) == REF_INT1);
        CHECK(r.beta(1.3, 2.2) == REF_BETA2);
    }
    {
        // 一个 5×5 的小盆：优先泛洪后中心被填到边缘 + eps 以上，D8 每格有下游或是出口
        GridD h(5, 5, 10.0);
        Mask m(5, 5, 1);
        h(2, 2) = 1.0;
        GridD f = priority_fill(h, m, 1e-3);
        CHECK(f(2, 2) > 10.0 && f(2, 2) < 10.01);
        FlowDir d = d8(f, m, 100.0);
        int outlets = 0;
        for (size_t k = 0; k < d.to_void.size(); ++k) outlets += d.to_void[k];
        CHECK(outlets == 16);
        CHECK(d.ri[2 * 5 + 2] >= 0);
        GridD A = accumulate(m, d);
        double tot = 0;
        for (int k = 0; k < 25; ++k)
            if (d.ri[k] < 0) tot += A.v[k];
        CHECK(tot == 25.0);
    }
    CHECK(pyround(2.675, 2) == 2.67);   // 二进制 2.675 = 2.67499999…
    CHECK(pyround(0.125, 2) == 0.12);   // 恰为一半：逢半取偶
    if (fails) {
        std::printf("%d 项失败\n", fails);
        return 1;
    }
    std::printf("core_selftest 全过\n");
    return 0;
}
