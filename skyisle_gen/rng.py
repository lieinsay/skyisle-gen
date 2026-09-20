"""确定性随机流。

阶段级：stage_rng(seed, stage_idx) —— 每阶段一个独立子流，改后面阶段的参数不扰动前面阶段。
实体级：entity_rng(seed, stage_idx, key) —— 以实体键（特征 id、分量键等）派生，
增删一个实体不扰动其它实体的随机数。
"""
from __future__ import annotations

import zlib

import numpy as np


def stage_rng(seed: int, stage_idx: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(stage_idx)])))


def entity_rng(seed: int, stage_idx: int, key: str) -> np.random.Generator:
    crc = zlib.crc32(key.encode("utf-8"))
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(seed), int(stage_idx), int(crc)]))
    )
