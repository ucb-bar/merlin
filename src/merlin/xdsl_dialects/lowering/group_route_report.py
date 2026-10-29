"""Report source-to-device format obligations without treating routing as code generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from merlin.common import mlir_query as mq


def lowering_obligation(group: Any) -> str:
    """Describe the next compiler obligation, never infer hardware impossibility.

    Family admission does not establish every operation in that family. In
    particular, ``no_device_form`` is missing implementation/evidence, not proof
    that arbitrary elementwise arithmetic can be offloaded.
    """
    from .compute_groups import HOST, VIEW, PAD, MOVEMENT, DEQUANTIZE

    if group.placement != HOST:
        decomposed = group.operand_sum or group.window_mean or getattr(group, "window_pool", None)
        return "selected_decomposition" if decomposed else "selected_device_form"
    if group.stages and all(stage == VIEW for stage in group.stages):
        return "metadata_only"
    if group.stages and all(stage in (VIEW, PAD, MOVEMENT) for stage in group.stages):
        return "data_movement"
    if DEQUANTIZE in group.stages and all(stage in (DEQUANTIZE, VIEW, MOVEMENT) for stage in group.stages):
        return "numerical_boundary"
    if group.refusal in ("no_device_form", "accumulate_exactness_unproved", "unplaced"):
        return "missing_device_form"
    if group.refusal in ("fused_only", "readout_requires_scale", "graph_structure"):
        return "missing_composition"
    if group.refusal in ("input_dtype", "weight_dtype", "operand_pair", "result_dtype", "float_epilogue_reassociation"):
        return "numerical_boundary"
    return "unresolved_evidence"


def input_format_changes(groups: Sequence[Any]) -> dict[str, int]:
    """Count device groups whose source operand types differ from their routed types."""
    from .compute_groups import HOST, _dtype_token

    changes: Counter[str] = Counter()
    for group in groups:
        if group.placement == HOST or group.root is None or not group.root.operands:
            continue
        observed = [_dtype_token(mq.type_shape_dtype(value.type)[1]) for value in group.root.operands[:2]]
        routed = [group.in_dtype, group.weight_dtype or group.in_dtype][: len(observed)]
        if any(src is not None and dst is not None and src != dst for src, dst in zip(observed, routed)):
            changes[f"{'+'.join(str(value) for value in observed)} -> {'+'.join(str(value) for value in routed)}"] += 1
    return dict(sorted(changes.items()))
