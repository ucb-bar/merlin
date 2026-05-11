"""Multi-tile flash-attention kernel variants (compiler-side plumbing).

The compiler emits a sequence of three attention variants per query row-block,
chained by an outer K/V tile loop. The variants share an online-softmax
recurrence: ``_first`` initializes the running (max, denom, partial output),
``_mid`` advances them, and ``_last`` divides by the final denominator and
emits the result tile.

Status: the manifest-backed ISA bodies are CURRENTLY STUBS that clone the
single-tile ``attention`` kernel (npu_uk_attention). They produce the same
DMA in/out layout the eventual flash-attention kernels will use, so the
compiler's loop-emission, address-patching and stitching paths can be
exercised end-to-end. Replacing the bodies with the real online-softmax
recurrence is tracked as a follow-up.
"""

from __future__ import annotations

from typing import Any

from npu_model.software import Instruction

try:
    from .manifest_loader import load_kernel
except ImportError:
    from manifest_loader import load_kernel


def first() -> list[Instruction[Any]]:
    return load_kernel("attention_acc_first")


def mid() -> list[Instruction[Any]]:
    return load_kernel("attention_acc_mid")


def last() -> list[Instruction[Any]]:
    return load_kernel("attention_acc_last")
