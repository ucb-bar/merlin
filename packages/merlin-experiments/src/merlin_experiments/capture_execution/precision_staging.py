"""Validate an explicitly selected FP32 capture stage, not a host accuracy claim."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from merlin.targetgen.frontend_trace import loads_frontend_trace

_FLOAT_DTYPES = frozenset({"bf16", "f16", "f32", "f64"})
_EXACT_DTYPES = frozenset({"i1", "i8", "ui8", "i16", "i32", "i64"})
_HEX = frozenset("0123456789abcdef")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX for char in value)


STAGING_API_REQUIREMENTS = {
    "m2m/capture/trace.py": {
        "materialize_frontend_precision": {"dtype", "original_frontend_snapshot", "retarget_float_dtype_arguments"},
    },
}


def output_staging_error(output: Path, meta: dict, *, selected: bool, recipe: bool) -> str | None:
    """Check retained trace bytes with metadata; an unreadable trace never admits a stage."""
    trace = None
    if selected:
        try:
            trace = loads_frontend_trace((output / "frontend-trace.json").read_bytes())
        except (OSError, ValueError):
            return "FP32 staging retained frontend trace is unreadable"
    return staging_error(meta, selected=selected, recipe=recipe, trace=trace)


def staging_error(meta: dict[str, Any], *, selected: bool, recipe: bool, trace: dict | None = None) -> str | None:
    """Return a refusal reason when selected precision evidence and saved ABI disagree."""
    stage = meta.get("fp32_staging")
    if not selected:
        return "unselected FP32 staging evidence" if stage is not None else None
    conversion = meta.get("precision_conversion")
    if not isinstance(stage, dict) or not isinstance(conversion, dict):
        return "selected FP32 staging evidence is absent"
    original_sha, staged_sha = stage.get("original_graph_sha256"), stage.get("staged_graph_sha256")
    if (
        stage.get("status") != "observed"
        or stage.get("scope") != "original computation versus staged FP32; not accuracy equivalence"
        or not all(_sha256(value) for value in (original_sha, staged_sha))
        or conversion.get("original_graph_sha256") != original_sha
        or conversion.get("staged_graph_sha256") != staged_sha
        or conversion.get("graph_dtype_retargeting") != "schema_float_dtype_operands"
        or not isinstance(conversion.get("dtype_decisions"), list)
    ):
        return "FP32 staging lacks source-bound precision decisions"
    original_trace = ((trace or {}).get("graphs") or {}).get("original") if isinstance(trace, dict) else None
    if (
        not isinstance(original_trace, dict)
        or original_trace.get("status") != "complete"
        or original_trace.get("sha256") != original_sha
    ):
        return "FP32 staging original graph differs from the retained frontend trace"
    audit = conversion.get("staged_precision_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("status") != "complete"
        or audit.get("target_dtype") != "torch.float32"
        or audit.get("non_target_floating_values") != 0
        or type(audit.get("checked_floating_values")) is not int
        or audit["checked_floating_values"] < 1
    ):
        return "FP32 staging has no complete typed audit"
    if recipe:
        inventory = stage.get("source_layer_inventory")
        inventory_sha = stage.get("source_layer_inventory_sha256")
        plan_sha = stage.get("source_layer_plan_sha256")
        if (
            not isinstance(inventory, list)
            or any(not isinstance(row, dict) for row in inventory)
            or not isinstance(inventory_sha, str)
            or hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            != inventory_sha
            or not _sha256(plan_sha)
            or (meta.get("quantization_stats") or {}).get("plan_sha256") != plan_sha
        ):
            return "FP32 staging lacks its original source-layer placement plan"
    before_inputs = stage.get("original_input_abi")
    before_outputs = stage.get("original_output_abi")
    after_inputs, after_outputs = stage.get("staged_input_abi"), stage.get("staged_output_abi")
    metrics, count = stage.get("output_metrics"), stage.get("output_cardinality")
    if (
        any(
            not isinstance(rows, list) for rows in (before_inputs, before_outputs, after_inputs, after_outputs, metrics)
        )
        or meta.get("input_abi") != after_inputs
        or (not recipe and meta.get("output_abi") != after_outputs)
        or (recipe and (not isinstance(meta.get("output_abi"), list) or len(meta["output_abi"]) != len(after_outputs)))
        or type(count) is not int
        or count < 1
        or count != len(before_outputs)
        or count != len(after_outputs)
        or count != len(metrics)
        or len(before_inputs) != len(after_inputs)
    ):
        return "FP32 staging output cardinality or ABI differs"
    for old, new in [*zip(before_inputs, after_inputs, strict=True), *zip(before_outputs, after_outputs, strict=True)]:
        if not isinstance(old, dict) or not isinstance(new, dict) or old.get("shape") != new.get("shape"):
            return "FP32 staging changed a tensor shape"
        source_dtype, staged_dtype = old.get("dtype"), new.get("dtype")
        if source_dtype in _FLOAT_DTYPES:
            if staged_dtype != "f32":
                return "FP32 staging retained a non-FP32 floating tensor"
        elif source_dtype in _EXACT_DTYPES:
            if staged_dtype != source_dtype:
                return "FP32 staging changed an exact integer/bool tensor dtype"
        else:
            return "FP32 staging source tensor dtype is unsupported"
    for old, new, row in zip(before_outputs, after_outputs, metrics, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("shape") != old.get("shape")
            or row.get("original_dtype") != old.get("dtype")
            or row.get("staged_dtype") != new.get("dtype")
            or row.get("comparison")
            != ("observed_floating" if old.get("dtype") in _FLOAT_DTYPES else "exact_nonfloating")
            or any(
                type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0
                for key in ("max_abs", "max_rel")
            )
        ):
            return "FP32 staging lacks finite original-to-staged output metrics"
        if old["dtype"] in _EXACT_DTYPES and (row["max_abs"] != 0 or row["max_rel"] != 0):
            return "FP32 staging changed an exact integer/bool output"
    return None
