"""Per-element cosine on a 32x32 bf16 tile.

Despite the name, this kernel does not compose the full rotary embedding — it
computes ``y = cos(x)`` on a 32x32 bf16 tile (stored in VMEM as two 32x16
halves per the ``bf16_split_halves`` layout). Pair it with a sibling sin
kernel (or torch-side pre-computation) to form the full rotary transform.
"""

from __future__ import annotations

from typing import Any

from npu_model.software import Instruction

try:
    from .manifest_loader import load_kernel
except ImportError:
    from manifest_loader import load_kernel


def test() -> list[Instruction[Any]]:
    return load_kernel("rope_frequency")
