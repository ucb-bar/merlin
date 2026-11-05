"""Callable structural analysis of actual xDSL IR, independent of target policies."""

from __future__ import annotations

from xdsl.dialects import builtin

from .analysis_liveness import block_facts
from .analysis_memory import effect_facts, may_alias, value_facts
from .analysis_types import IRAnalysis, IRIndex


def analyze_ir(module: builtin.ModuleOp, *, index_bits: int | None = None) -> IRAnalysis:
    if index_bits is not None and (type(index_bits) is not int or index_bits <= 1):
        raise ValueError("index width must be explicit and positive when known")
    index = IRIndex(module)
    values, effects = value_facts(index, index_bits=index_bits), effect_facts(index)
    unknown = [f"value {value.id}: unknown memory aliasing" for value in values if value.alias_kind == "unknown"]
    unknown.extend(f"operation {effect.operation}: unknown effects" for effect in effects if effect.kind == "unknown")
    return IRAnalysis(
        index.source_sha256, index_bits, values, effects, block_facts(index, values, effects), tuple(unknown)
    )


def validate_analysis(module: builtin.ModuleOp, analysis: IRAnalysis) -> None:
    if analyze_ir(module, index_bits=analysis.index_bits) != analysis:
        raise ValueError("analysis does not describe the actual current IR")


__all__ = ["IRAnalysis", "IRIndex", "analyze_ir", "may_alias", "validate_analysis"]
