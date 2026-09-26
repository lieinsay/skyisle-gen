"""从 numpy 探出 Generator.standard_normal 与 standard_exponential 用的 256 层 ziggurat 表，写成 core/src/ziggurat_tables.inc。

numpy 的表（ziggurat_constants.h）不在发行包里。做法：把 PCG64 的状态设成「下一个原始输出 = 指定值」
（输出 = rotr(hi ^ lo, s >> 122)：取 s 的高 6 位为 0，lo = hi ^ 目标值；再用乘数的逆元倒推一步），
于是分布函数读到的 64 位完全由我们定。
正态（standard_normal）：低 8 位是层号 idx，第 8 位符号，其上 52 位是 rabs。
  wi[idx] = rabs 取 1 时的输出（x = rabs × wi，乘 1 精确）；
  ki[idx] = 二分找「快路径（只消耗一个原始数）/ 慢路径」的分界 rabs；
  fi[idx] = exp(−x²/2)，x = wi × 2^52（Marsaglia–Tsang 的定式；fi[0] = 1）——fi 只在 0.7% 的慢路径里做接受判断，差一位几乎不可能翻判断。
指数（standard_exponential，P6b：伽马形状 < 1 要用）：ri = 原始 >> 3，idx = ri & 0xFF，ri >>= 8（53 位）。
  we[idx] = ri 取 1 时的输出；ke[idx] = 快 / 慢路径的分界 ri；fe[idx] = exp(−we × 2^53)（fe[0] = 1）。
用法：python core/tools/probe_ziggurat.py（仓库根下；numpy 换大版本后重跑一遍看表变没变）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

M128 = (1 << 128) - 1
MULT = (0x2360ED051FC65DA4 << 64) | 0x4385DF649FCCF645
MINV = pow(MULT, -1, 1 << 128)
HI = 0x0123456789ABCDEF & ((1 << 58) - 1)


def main() -> None:
    bg = np.random.PCG64(1)
    inc = bg.state["state"]["inc"]
    gen = np.random.Generator(bg)

    def set_next(o: int) -> int:
        s1 = (HI << 64) | (HI ^ o)
        s0 = ((s1 - inc) * MINV) & M128
        bg.state = {"bit_generator": "PCG64", "state": {"state": s0, "inc": inc}, "has_uint32": 0, "uinteger": 0}
        return s1

    def probe(draw, encode, nbits: int):
        """draw() 抽一个；encode(idx, r) → 原始 64 位。返回 (w, k)：w = r 取 1 的输出，k = 快路径的上界。"""
        def with_(idx, r):
            s1 = set_next(encode(idx, r))
            x = draw()
            return x, bg.state["state"]["state"] == s1

        ws, ks = [], []
        for idx in range(256):
            x, fast = with_(idx, 1)
            if not fast:                     # k ≤ 1（最上一层恒走慢路径）：r = 1 时 x 极小，慢路径几乎必接受，多消耗一个原始数
                s1 = set_next(encode(idx, 1))
                x = draw()
                assert bg.state["state"]["state"] == (s1 * MULT + inc) & M128, idx
                _, f0 = with_(idx, 0)
                ws.append(x)
                ks.append(1 if f0 else 0)
                continue
            ws.append(x)
            lo, hi = 1, (1 << nbits) - 1
            if with_(idx, hi)[1]:
                ks.append(1 << nbits)
                continue
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if with_(idx, mid)[1]:
                    lo = mid
                else:
                    hi = mid
            ks.append(hi)
        return ws, ks

    set_next(0xDEADBEEF12345678)
    assert int(bg.random_raw()) == 0xDEADBEEF12345678
    wi, ki = probe(gen.standard_normal, lambda idx, r: idx | (r << 9), 52)
    fi = [1.0] + [math.exp(-0.5 * (w * 2.0 ** 52) ** 2) for w in wi[1:]]
    we, ke = probe(gen.standard_exponential, lambda idx, r: (idx << 3) | (r << 11), 53)
    fe = [1.0] + [math.exp(-(w * 2.0 ** 53)) for w in we[1:]]
    out = Path(__file__).resolve().parent.parent / "src" / "ziggurat_tables.inc"
    lines = ["// 由 core/tools/probe_ziggurat.py 从 numpy " + np.__version__ + " 探出，勿手改。",
             "// numpy Generator.standard_normal 的 256 层 ziggurat：ki（快路径上界）、wi（层宽 / 2^52）、fi（exp(-x^2/2)）；",
             "// standard_exponential 的：ke、we（层宽 / 2^53）、fe（exp(-x)）。", ""]
    for kname, karr, names in (("ZIG_KI", ki, (("ZIG_WI", wi), ("ZIG_FI", fi))), ("ZIG_KE", ke, (("ZIG_WE", we), ("ZIG_FE", fe)))):
        lines.append(f"static const uint64_t {kname}[256] = {{")
        lines += ["    " + ", ".join(f"{k}ULL" for k in karr[i:i + 4]) + "," for i in range(0, 256, 4)]
        lines.append("};")
        for name, arr in names:
            lines.append(f"static const double {name}[256] = {{")
            lines += ["    " + ", ".join(float(v).hex() for v in arr[i:i + 4]) + "," for i in range(0, 256, 4)]
            lines.append("};")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写出 {out}（wi[255]·2^52 = {wi[255] * 2.0 ** 52!r}，we[255]·2^53 = {we[255] * 2.0 ** 53!r}，ke[0..3] = {ke[:4]}）")


if __name__ == "__main__":
    main()
