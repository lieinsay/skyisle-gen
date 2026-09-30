"""确定性随机流（Python 这边只剩 island batch 的抽样用它；管线与第三层的随机数在 C++ 的 rng.hpp，与 numpy 逐位一致）。

实体级：entity_rng(seed, stage_idx, key) —— 以实体键派生，增删一个实体不扰动其它实体的随机数。
"""
from __future__ import annotations

import zlib

import numpy as np


def entity_rng(seed: int, stage_idx: int, key: str) -> np.random.Generator:
    crc = zlib.crc32(key.encode("utf-8"))
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(seed), int(stage_idx), int(crc)]))
    )
