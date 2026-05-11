"""K-tiled matmul kernel variants that share a single MXU accumulator.

The base ``matmul`` kernel computes ``C = A @ B`` for one (M=32, K=32, N=32)
tile and immediately drains the accumulator. To support matmuls whose K
dimension exceeds 32, the compiler emits a sequence of these three variants
along a K-loop, all targeting the same accumulator state:

    matmul_acc_first  — vmatmul.mxu0       (overwrite accumulator, no drain)
    matmul_acc_mid    — vmatmul.acc.mxu0   (add to accumulator, no drain)
    matmul_acc_last   — vmatmul.acc.mxu0   (add then drain + DMA store)

The scalar/DMA prefix and the vmatpush.weight setup are identical across
variants — only the multiply mnemonic and the trailing pop/store block differ.
"""

from __future__ import annotations

from typing import Any

from npu_model.software import Instruction

try:
    from .manifest_loader import load_kernel
except ImportError:
    from manifest_loader import load_kernel


def first() -> list[Instruction[Any]]:
    return load_kernel("matmul_acc_first")


def mid() -> list[Instruction[Any]]:
    return load_kernel("matmul_acc_mid")


def last() -> list[Instruction[Any]]:
    return load_kernel("matmul_acc_last")
