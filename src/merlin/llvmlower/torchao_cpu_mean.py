"""Optional source CPU reduction order for TorchAO W8A8 captures.

The captured PyTorch wheel (2.16.0.dev20261104+cpu, git 7f24a19c0967)
dispatches contiguous f32 ``aten.mean.dim`` to AVX2 ``cascade_sum`` and then
divides by K. An ordinary serial sum has different floating-point results and
can change a following signed-int8 quantizer at a rounding boundary.

This is a source-fidelity mode, default off. It is valid only when the caller
has checked that exact source CPU commit and AVX2 dispatch; the IR alone does
not certify those facts. The pass matches the source's already-quantized graph,
not any target name, and preserves the captured sum/division separation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .cpu_sum import TORCHAO_20261104_AVX2_POLICY, build_cpu_sum_lastdim

TORCH_SOURCE_COMMIT = "7f24a19c0967db04f2d5ba44c1e56eca489932ec"
MARKER = "merlin.torchao_cpu_mean"


def rewrite_torchao_cpu_avx2_means(
    module,
    *,
    source_commit: str,
    on_rewrite: Callable[[Any, tuple[Any, ...], Any], None] | None = None,
) -> int:
    """Replace eligible zero-seeded f32 mean sums by explicit AVX2 carry order."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import StringAttr, TensorType, f32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.rewriter import InsertPoint, Rewriter

    from .normalization_reassociation import _positive_zero_splat
    from .passes_xdsl import carry_provenance

    if source_commit != TORCH_SOURCE_COMMIT:
        raise ValueError("explicit exact captured Torch CPU source commit required")
    if getattr(module.attributes.get("prov.quantization"), "data", None) != "int8_dyn_act_int8_weight":
        raise ValueError("TorchAO source W8A8 capture scheme required")

    changed = 0
    for op in list(module.walk()):
        if not isinstance(op, linalg.ReduceOp):
            continue
        if getattr(op.attributes.get("prov.aten"), "data", None) != "aten.mean.dim":
            continue
        if len(op.operands) != 2 or len(op.results) != 1 or MARKER in op.attributes:
            continue
        value, init = op.operands
        if not isinstance(value.type, TensorType) or value.type.element_type != f32:
            continue
        shape = value.type.get_shape()
        if not shape or any(n <= 0 for n in shape) or shape[-1] != 2048:
            continue
        if tuple(op.dimensions.iter_values()) != (len(shape) - 1,):
            continue
        if op.results[0].type != TensorType(f32, shape[:-1]) or not _positive_zero_splat(init):
            continue
        body = list(op.regions[0].block.ops)
        if len(body) != 2 or not isinstance(body[0], arith.AddfOp) or not isinstance(body[1], linalg.YieldOp):
            continue
        if list(body[0].operands) != list(op.regions[0].block.args) or list(body[1].operands) != [body[0].result]:
            continue

        generated, result = build_cpu_sum_lastdim(
            value,
            backend_policy=TORCHAO_20261104_AVX2_POLICY,
            allow_reassociation=True,
            assume_finite_intermediates=True,
        )
        for added in generated:
            carry_provenance(added, op, "torchao_cpu_avx2_mean")
        generated[-1].attributes[MARKER] = StringAttr(TORCH_SOURCE_COMMIT)
        Rewriter.insert_op(generated, InsertPoint.before(op))
        if on_rewrite is not None:
            on_rewrite(op, tuple(generated), generated[-1])
        op.results[0].replace_all_uses_with(result)
        op.parent_block().detach_op(op)
        changed += 1
    return changed
