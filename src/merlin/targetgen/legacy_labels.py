"""Neutral spellings of labels earlier corpora and captures wrote, and the READ aliases that keep them.

Some labels a corpus carried named how a convolution happened to be lowered rather than what it
computes: an operation attribute ``semantic: conv2d_im2col`` and the capture provenance
``prov.conv_path = "im2col_matmul"`` / ``prov.op = "convolution_im2col_matmul"``. A capsule an agent
reads states the computation, never a lowering, so new writes use the neutral spellings below. Records,
retained capsules and captures written before keep verifying: every reader normalizes an old spelling
to the new one through these tables. Nothing here ever WRITES an old spelling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Operation ``semantic`` attribute: old spelling -> the one written now.
SEMANTIC_ALIASES = {"conv2d_im2col": "conv2d"}
#: ``prov.conv_path`` of a convolution the capture states as a gather of its windows and a contraction.
CONV_PATH_GATHERED = "gathered_matmul"
CONV_PATH_ALIASES = {"im2col_matmul": CONV_PATH_GATHERED}
#: ``prov.op`` / ``prov._pattern_hint`` of the same convolution.
CONV_OP_GATHERED = "convolution_gathered_matmul"
CONV_OP_ALIASES = {"convolution_im2col_matmul": CONV_OP_GATHERED}


def semantic(value: Any) -> Any:
    """An operation ``semantic`` label, old spellings read as the neutral one."""
    return SEMANTIC_ALIASES.get(value, value) if isinstance(value, str) else value


def conv_path(value: Any) -> Any:
    return CONV_PATH_ALIASES.get(value, value) if isinstance(value, str) else value


def conv_op(value: Any) -> Any:
    return CONV_OP_ALIASES.get(value, value) if isinstance(value, str) else value


def is_gathered_conv_path(value: Any) -> bool:
    return conv_path(value) == CONV_PATH_GATHERED


def is_gathered_conv_op(value: Any) -> bool:
    return conv_op(value) == CONV_OP_GATHERED


def operation_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """A capsule operation's attributes with its ``semantic`` label read through the aliases."""
    out = dict(attributes)
    if "semantic" in out:
        out["semantic"] = semantic(out["semantic"])
    return out


__all__ = [
    "CONV_OP_ALIASES",
    "CONV_OP_GATHERED",
    "CONV_PATH_ALIASES",
    "CONV_PATH_GATHERED",
    "SEMANTIC_ALIASES",
    "conv_op",
    "conv_path",
    "is_gathered_conv_op",
    "is_gathered_conv_path",
    "operation_attributes",
    "semantic",
]
