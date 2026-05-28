"""mxGemmini-specific torchao quantization helpers.

Two stages, both available in this iteration:

    Stage 6.A — uses stock torchao MX-format dtypes:
        - FP8 path: ``torch.float8_e4m3fn`` (signed E4M3)
        - FP4 path: NVFP4 (signed E2M1, ``torch.float4_e2m1fn_x2``)
      with ``block_size=16`` to match the ScaleFactorMem row width that
      mxGemmini expects (NOT 32 like SaturnNPU). Lets the export →
      compile → run mechanism work end-to-end mechanically; the
      element-level encoding is approximate vs the hardware datapath.

    Stage 6.B — custom torchao Tensor subclasses
      (``MxGemminiE4M4Tensor`` for FP8_0, ``MxGemminiE2M2Tensor`` for
      FP4) implementing mxGemmini's *exact* bit layout: unsigned
      per-element, sign carried by the per-block shared scale.
      Numerical fidelity to the hardware datapath; intended for the
      final dialect-vs-simulator diff.

The user picks per-call which stage to use via the ``stage=`` argument
on :func:`safe_quantize_linears_`.

mxGemmini saturation values (from
``third_party/gemmini-mx/.../MxRequantizer.scala:7-44``):

    +---------+--------+----------+
    | format  | pmax   | log2     |
    +=========+========+==========+
    | FP4     | ±6     | 2        |
    | FP6     | ±28    | 4        |
    | FP8     | ±448   | 8        |
    | BF16    | ±65024 | (acc)    |
    +---------+--------+----------+

Public API mirrors
``third_party/Understanding-PI0/understanding_pi0/common/torchao_utils.py``
so callers used to that pattern can swap in.
"""

from __future__ import annotations

from .config import (
    MX_BLOCK_SIZE,
    MxGemminiFormat,
    MxGemminiSaturation,
    make_mxgemmini_fp4_config,
    make_mxgemmini_fp8_config,
)
from .custom_dtype import (
    E2M2_PMAX,
    E4M4_PMAX,
    MxGemminiE2M2Tensor,
    MxGemminiE4M4Tensor,
    quantize_to_e2m2,
    quantize_to_e4m4,
)
from .export import clone_and_rewrite_quantized_linears_for_export
from .quantize import (
    QuantizeResult,
    safe_quantize_linears_,
    summarize_results,
)

__all__ = [
    "E2M2_PMAX",
    "E4M4_PMAX",
    "MX_BLOCK_SIZE",
    "MxGemminiE2M2Tensor",
    "MxGemminiE4M4Tensor",
    "MxGemminiFormat",
    "MxGemminiSaturation",
    "QuantizeResult",
    "clone_and_rewrite_quantized_linears_for_export",
    "make_mxgemmini_fp4_config",
    "make_mxgemmini_fp8_config",
    "quantize_to_e2m2",
    "quantize_to_e4m4",
    "safe_quantize_linears_",
    "summarize_results",
]
