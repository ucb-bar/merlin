"""Stage 6.A — torchao ``MXDynamicActivationMXWeightConfig`` factories.

mxGemmini block size is **16** (one ``ScaleFactorMem`` row), not 32
like SaturnNPU. The element dtype is approximated by the closest stock
torchao dtype:

    FP8_0 (E4M4 unsigned, ``±448``)  ≈  ``torch.float8_e4m3fn`` (signed E4M3)
    FP4   (E2M2 unsigned, ``±6``)    ≈  ``torch.float4_e2m1fn_x2`` (signed E2M1, NVFP4)

The mismatch is documented; Stage 6.B (``custom_dtype``) replaces these
with bit-exact subclasses for numerical fidelity.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

try:
    from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
except ImportError:  # pragma: no cover
    from torchao.prototype.mx_formats.inference_workflow import (
        MXDynamicActivationMXWeightConfig,
    )

try:
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference,
    )
except ImportError:  # pragma: no cover
    from torchao.quantization.quantize_.common import KernelPreference


# mxGemmini stores 16 activation x 16 weight scales per ScaleFactorMem
# row. See third_party/gemmini-mx/.../ScaleFactorMem.scala.
MX_BLOCK_SIZE = 16


class MxGemminiFormat(enum.StrEnum):
    """Hardware-supported act × wei combos in the default ``MxFPMul`` config.

    See ``third_party/gemmini-mx/src/main/scala/gemmini/MxFPMul.scala:18-21``
    for the enabled set; ``MxParameters.scala:124-130`` for the bit
    encoding (``CONFIG_EX rs1[11:10]/[13:12]/[15:14]``).
    """

    FP4 = "fp4"  # E2M2, encoding=0
    FP6_1 = "fp6_1"  # E3M3, encoding=3
    FP8_0 = "fp8_0"  # E4M4, encoding=2


@dataclass(frozen=True)
class MxGemminiSaturation:
    """Per-element saturation magnitudes from MxRequantizer.scala:7-44."""

    fp4_pmax: float = 6.0
    fp4_log2: int = 2
    fp6_pmax: float = 28.0
    fp6_log2: int = 4
    fp8_pmax: float = 448.0
    fp8_log2: int = 8
    bf16_pmax: float = 65024.0


def _resolve_kernel_preference(name: str | None) -> KernelPreference:
    """Default to ``EMULATED`` (not ``AUTO``).

    torchao's ``AUTO`` preference asserts ``block_size==32`` for both
    ``float8_e4m3fn`` and ``float4_e2m1fn_x2`` (see
    ``torchao/prototype/mx_formats/config.py::_validate_kernel_preference``).
    mxGemmini uses ``block_size=16``, so we have to fall back to
    ``EMULATED`` which has no such constraint.
    """
    if not name:
        return KernelPreference.EMULATED
    lookup = name.strip().upper()
    if hasattr(KernelPreference, lookup):
        return getattr(KernelPreference, lookup)
    raise ValueError(f"Unsupported KernelPreference '{name}'")


def make_mxgemmini_fp8_config(
    kernel_preference: str | None = None,
) -> MXDynamicActivationMXWeightConfig:
    """Stage 6.A — stock-torchao MX config approximating mxGemmini FP8_0.

    Element dtype: ``torch.float8_e4m3fn`` (signed E4M3, NOT the
    unsigned E4M4 mxGemmini actually uses). Block size: 16. Stage 6.B
    custom dtype is bit-exact; this stage trades fidelity for the
    ability to use torchao's stock export pipeline.
    """
    return MXDynamicActivationMXWeightConfig(
        block_size=MX_BLOCK_SIZE,
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=_resolve_kernel_preference(kernel_preference),
    )


def make_mxgemmini_fp4_config(
    kernel_preference: str | None = None,
) -> MXDynamicActivationMXWeightConfig:
    """Stage 6.A — stock-torchao MX config approximating mxGemmini FP4.

    Tries ``torch.float4_e2m1fn_x2`` (NVFP4, signed E2M1 packed two per
    byte) — closest stock torchao dtype to mxGemmini's unsigned E2M2.
    Block size: 16.

    Falls back to ``torch.float8_e4m3fn`` with a smaller block if
    NVFP4 is not available on this torchao build.
    """
    e2m1 = getattr(torch, "float4_e2m1fn_x2", None)
    if e2m1 is None:
        # Older torchao without NVFP4 dtype — keep FP8 fallback so the
        # mechanism still flows end-to-end. Stage 6.B replaces this.
        return make_mxgemmini_fp8_config(kernel_preference)
    return MXDynamicActivationMXWeightConfig(
        block_size=MX_BLOCK_SIZE,
        activation_dtype=e2m1,
        weight_dtype=e2m1,
        kernel_preference=_resolve_kernel_preference(kernel_preference),
    )
