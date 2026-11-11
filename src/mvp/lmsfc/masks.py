from __future__ import annotations

from typing import List


def validate_masks(masks: List[int], d: int) -> None:
    if int(d) <= 0:
        raise ValueError("d must be > 0")
    if len(masks) != int(d):
        raise ValueError("masks length must equal d")

    used = 0
    for m in masks:
        if not isinstance(m, int):
            raise TypeError("mask must be int")
        if m < 0:
            raise ValueError("mask must be non-negative")
        if m & used:
            raise ValueError("masks must not overlap")
        used |= m

    if used.bit_length() > 64:
        raise ValueError("masks must fit within 64 bits")


def standard_morton_masks(d: int) -> List[int]:
    d = int(d)
    if d <= 0:
        raise ValueError("d must be > 0")

    masks: List[int] = []
    for i in range(d):
        m = 0
        for bit in range(i, 64, d):
            m |= 1 << bit
        masks.append(m)

    validate_masks(masks, d)
    return masks
