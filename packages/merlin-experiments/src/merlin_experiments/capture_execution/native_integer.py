"""Explicit unchanged integer source mode for the existing sealed CPU runner.

No quantization is selected or applied. This checks the saved integral source
domain and complete positional ABI, independently of accelerator admission.
The ordinary seal still binds and replays every source/runtime/output byte.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import numpy as np

from merlin.common import mlir_query as mq
from merlin.common import strict_json

MODE = "native_integer_v1"


def selected(options):
    return (options or {}).get("source_mode") == MODE


def require_selection(options, *, dtype, recipe):
    """The versioned mode cannot be combined with any numeric transformation."""
    if not selected(options):
        raise ValueError("native integer capture requires explicit native_integer_v1 source mode")
    if dtype != "int8" or recipe or set(options) - {"source_mode", "cpu_threads"}:
        raise ValueError("native integer capture admits int8 source without recipe, staging or tolerances")


def validate(output: Path, metadata: dict, plan: dict) -> dict:
    from .sealed_static import _file_digest

    if plan.get("native_integer_verifier_sha256") != _file_digest(Path(__file__)):
        raise ValueError("selected native integer verifier source changed")
    require_selection(plan.get("worker_options"), dtype=plan.get("dtype"), recipe=plan.get("recipe"))
    if (
        metadata.get("capture_quantization") != "already_materialized"
        or metadata.get("source_mode") != MODE
        or any(metadata.get(key) is not None for key in ("scheme", "recipe", "recipe_sha256", "precision_conversion"))
        or metadata.get("quantization_stats", {}).get("applied") is not False
        or metadata.get("ok") is not True
        or metadata.get("opaque") != 0
    ):
        raise ValueError("native integer capture selected transformations or incomplete source lowering")
    inputs, outputs = metadata.get("input_abi"), metadata.get("output_abi")
    trace = strict_json.loads((output / "frontend-trace.json").read_bytes())
    original = (trace.get("graphs") or {}).get("original") or {}
    if (
        trace.get("schema") != "m2m.frontend_trace.v1"
        or original.get("schema") != "m2m.frontend_graph.v1"
        or original.get("status") != "complete"
    ):
        raise ValueError("native integer source requires complete original frontend trace")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or not outputs:
        raise ValueError("native integer source requires complete nonempty typed ABI")
    for row in inputs + outputs:
        if (
            not isinstance(row, dict)
            or row.get("dtype") not in {"i8", "i32"}
            or not isinstance(row.get("shape"), list)
            or any(type(n) is not int or n < 1 for n in row["shape"])
        ):
            raise ValueError("native integer source mode refuses non-signed-i8/i32 or dynamic ABI")
    from xdsl.dialects import builtin

    module = mq.parse(output / "model.mlir")
    for operation in mq.walk(module):
        values = [*operation.operands, *operation.results]
        for region in operation.regions:
            values.extend(arg for block in region.blocks for arg in block.args)
        for value in values:
            typ = value.type
            element = typ.get_element_type() if isinstance(typ, builtin.ShapedType) else typ
            if isinstance(element, builtin.AnyFloat):
                raise ValueError("native integer source mode refuses floating intermediate arithmetic")
    golden = strict_json.loads((output / "golden.json").read_bytes())
    values = golden if len(outputs) > 1 else [golden]
    if not isinstance(values, list) or len(values) != len(outputs):
        raise ValueError("native integer capture omitted a complete golden output")
    for row, value in zip(outputs, values, strict=True):
        array = np.asarray(value)
        width = 8 if row["dtype"] == "i8" else 32
        if (
            array.dtype.kind not in "iu"
            or list(array.shape) != row["shape"]
            or array.size != prod(row["shape"])
            or np.any(array < -(2 ** (width - 1)))
            or np.any(array > 2 ** (width - 1) - 1)
        ):
            raise ValueError("native integer golden differs from its complete signed output ABI")
    with np.load(output / "inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"in{i}" for i in range(len(inputs))}:
            raise ValueError("native integer capture omitted complete input leaves")
        for i, row in enumerate(inputs):
            value = archive[f"in{i}"]
            if list(value.shape) != row["shape"] or value.dtype != np.dtype(
                "int8" if row["dtype"] == "i8" else "int32"
            ):
                raise ValueError("native integer captured input differs from its signed ABI")
    return {
        "source_mode": MODE,
        "numeric_contract": {"compare": "per_output", "outputs": [{**row, "compare": "exact_int"} for row in outputs]},
        "source_scope": "unchanged integral CPU source and complete outputs; no target/compiler qualification",
    }
