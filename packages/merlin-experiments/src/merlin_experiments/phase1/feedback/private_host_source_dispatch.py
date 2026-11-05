"""Route closed source-body admissions to their independent mandatory witnesses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.frontends.bucketize_source import STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA
from merlin.frontends.linalg_f32_maximum_patterns import STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
from merlin.frontends.linalg_reduction_source_body import STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA
from merlin.frontends.prepared_index_source_body import SCHEMA as PREPARED_INDEX_SOURCE_BODY_SCHEMA
from merlin_experiments.phase1.feedback import private_f32_maximum_support as maximum_support
from merlin_experiments.phase1.feedback import private_f32_reduction_support as f32_reduction_support
from merlin_experiments.phase1.feedback import private_index_host_support as index_support
from merlin_experiments.phase1.feedback import private_integer_reduction_support as integer_support
from merlin_experiments.phase1.feedback import private_linalg_support as linalg_support
from merlin_experiments.phase1.feedback.private_source_admission import (
    prepared_host_admission as prepared_host_admission,
)


def source_context(row: Mapping[str, Any], index: Mapping[str, Any], selected: Mapping[str, Any] | None) -> dict:
    """Only prepared index roots receive the independently proved trace premise."""
    context = {"selected_index_observation": selected}
    if row.get("frontend_op") == "aten.index.Tensor" and row.get("mlir_operation") in {
        "linalg.generic",
        "linalg.reduce",
    }:
        context["verified_index_source"] = index["source_proof"]
    return context


def record(
    linalg: dict,
    integer: dict,
    index: dict,
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
    bounded_control: Mapping[str, Any] | None,
    *,
    maximum: dict | None = None,
    f32_reductions: dict | None = None,
) -> None:
    """Never send a new body schema through the legacy pointwise-only witness.

    A reduction goes to the witness of its parsed element type: an f32 reduction
    to the f32 witness, every other reduction to the unchanged integer witness.
    """
    body = host_decision.get("source_body_proof")
    schema = body.get("schema") if isinstance(body, Mapping) else None
    if schema == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA:
        integer_support.record(integer, row, host_decision, parsed, source_rows)
        return
    if schema == STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA:
        if maximum is None:
            raise ValueError("f32 maximum source-body admission has no mandatory private witness")
        maximum_support.record(maximum, row, host_decision, parsed, source_rows)
        return
    if schema == PREPARED_INDEX_SOURCE_BODY_SCHEMA:
        index_support.record(index, row, host_decision, parsed, source_rows)
        return
    if schema == STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA:
        # The caller's mandatory bucketize witness independently records the
        # verified trace, boundary literal and selected linked source below.
        return
    linalg_support.record(linalg, row, host_decision, parsed, source_rows, control_proof=bounded_control)
    if f32_reduction_support.selects(row, parsed):
        if f32_reductions is None:
            raise ValueError("f32 reduction host admission has no mandatory private witness")
        f32_reduction_support.record(f32_reductions, row, host_decision, parsed, source_rows)
    else:
        integer_support.record(integer, row, host_decision, parsed, source_rows)
    index_support.record(index, row, host_decision, parsed, source_rows)
    if maximum is not None:
        maximum_support.record(maximum, row, host_decision, parsed, source_rows)
