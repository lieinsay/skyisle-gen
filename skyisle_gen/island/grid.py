"""岛群前端的栅格小工具：平移、膨胀（check 用）与 PNG 写出，全部 numpy、不引入 scipy。噪声、连通分量、形态学、重采样的算法
在 C++ 核心里（core/include/skyisle/grid.hpp；Python 参考版删于 2026-09-30，tag python-reference-final）。"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 形态学
def shift(a: np.ndarray, di: int, dj: int, fill):
    """整体平移（不回绕），空出的部分填 fill。"""
    out = np.full_like(a, fill)
    H, W = a.shape
    si0, si1 = max(0, di), min(H, H + di)
    sj0, sj1 = max(0, dj), min(W, W + dj)
    out[si0:si1, sj0:sj1] = a[si0 - di:si1 - di, sj0 - dj:sj1 - dj]
    return out


N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def binary_dilate(mask: np.ndarray, iterations: int = 1, connectivity: int = 8) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    nb = N8 if connectivity == 8 else N4
    for _ in range(iterations):
        acc = m.copy()
        for di, dj in nb:
            acc |= shift(m, di, dj, False)
        m = acc
    return m


# ---------------------------------------------------------------- PNG 写出（无 PIL 依赖）
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    c = struct.pack(">I", len(data)) + tag + data
    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png16(path: Path, arr: np.ndarray) -> None:
    """16 位灰度 PNG。"""
    a = np.ascontiguousarray(np.clip(arr, 0, 65535).astype(">u2"))
    H, W = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(H))
    ihdr = struct.pack(">IIBBBBB", W, H, 16, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def write_png8(path: Path, arr: np.ndarray, palette: list[tuple[int, int, int]] | None = None) -> None:
    """8 位 PNG；给调色板则写索引色（类型 3），否则灰度（类型 0）。"""
    a = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    H, W = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(H))
    ctype = 3 if palette is not None else 0
    ihdr = struct.pack(">IIBBBBB", W, H, 8, ctype, 0, 0, 0)
    chunks = _png_chunk(b"IHDR", ihdr)
    if palette is not None:
        pal = bytes(sum(([int(r), int(g), int(b)] for r, g, b in palette), []))
        chunks += _png_chunk(b"PLTE", pal)
    chunks += _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunks)
