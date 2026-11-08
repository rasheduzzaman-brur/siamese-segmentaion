"""Label Studio brush RLE codec (the `value.rle` of a brushlabels result).

The format is a bit stream: 32-bit pixel-value count, 5-bit word size - 1,
four 4-bit run-length field sizes - 1, then blocks of
    1 bit  repeat flag
    2 bits index of the run-length field size
    n bits run length - 1
    then one word (repeat) or run-length words (literal)
over the flattened H x W x 4 RGBA array, of which alpha marks the region.
"""
from __future__ import annotations

from typing import List

import numpy as np


def _bits(data: List[int]) -> np.ndarray:
    return np.unpackbits(np.asarray(data, dtype=np.uint8))


def decode(rle: List[int], width: int, height: int) -> np.ndarray:
    """-> boolean H x W mask."""
    bits = _bits(rle)
    pos = 0

    def read(n: int) -> int:
        nonlocal pos
        v = 0
        for b in bits[pos:pos + n]:
            v = (v << 1) | int(b)
        pos += n
        return v

    num = read(32)
    word = read(5) + 1
    sizes = [read(4) + 1 for _ in range(4)]
    out = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        repeat = read(1)
        j = i + 1 + read(sizes[read(2)])
        if repeat:
            out[i:j] = read(word)
            i = j
        else:
            while i < j:
                out[i] = read(word)
                i += 1
    if num != width * height * 4:
        raise ValueError(f"RLE holds {num} values, expected {width}x{height}x4")
    return out.reshape(height, width, 4)[:, :, 3] > 0


def encode(mask: np.ndarray) -> List[int]:
    """Boolean H x W mask -> RLE (RGBA, alpha = 255 inside), repeat blocks only."""
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask.astype(bool), 3] = 255
    flat = rgba.reshape(-1)
    sizes = [3, 4, 8, 16]
    parts = [format(len(flat), "032b"), format(8 - 1, "05b")] + [format(s - 1, "04b") for s in sizes]
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(flat)]])
    for s, e in zip(starts, ends):
        value, run = int(flat[s]), int(e - s)
        while run > 0:
            n = min(run, 1 << 16)
            idx = next(k for k, sz in enumerate(sizes) if n - 1 < (1 << sz))
            parts.append("1" + format(idx, "02b") + format(n - 1, f"0{sizes[idx]}b") + format(value, "08b"))
            run -= n
    bits = "".join(parts)
    bits += "0" * (-len(bits) % 8)
    return [int(bits[k:k + 8], 2) for k in range(0, len(bits), 8)]
