"""Closed semantic schema for independently selected minimal software specs.

Scheduling, model/capture/history, backend instructions, shapes and performance
metadata have no admitted field. This validator establishes content scope, not
human review, historical origin, target support or numerical correctness.
"""

from __future__ import annotations

from merlin.targetgen.software_spec import validate_software_spec

from .rtl_intake import RtlIntakeRefusal

_ROOT = {"schema", "target", "status", "numerical_semantics", "operations"}
_NUMERIC = {
    "model",
    "operand_dtype",
    "accumulator_dtype",
    "readout_dtype",
    "subnormal_operand_flush",
    "overflow",
    "rounding",
    "reduction_order",
    "reduction_cadence",
    "product_rounding",
    "input_domain",
    "output_domain",
}
_OPERATION = {"id", "ops", "families", "status", "hardware", "placement", "signature"}
_SIGNATURE = {
    "operand_dtypes",
    "ordered_operand_dtypes",
    "ordered_result_dtypes",
    "compute_dtypes",
    "accumulator_dtype",
    "readout_dtype",
    "broadcasting",
    "aliasing",
    "quantization_parameters",
}


def _dtype(value) -> None:
    if not isinstance(value, str) or not value.isascii():
        raise RtlIntakeRefusal("minimal numerical dtypes require explicit scalar type names")
    prefix = next((p for p in ("bfloat", "float", "uint", "int", "bf", "f", "u", "i") if value.startswith(p)), None)
    if prefix is None or not value[len(prefix) :].isdigit() or int(value[len(prefix) :]) <= 0:
        raise RtlIntakeRefusal("extended numerical types need a separately reviewed minimal semantic schema")


def _numeric(document) -> None:
    if not isinstance(document, dict) or set(document) - _NUMERIC:
        raise RtlIntakeRefusal("minimal numerical semantics refuse backend, model, scheduling or extra metadata")
    model = document.get("model")
    if (
        not isinstance(model, dict)
        or set(model) != {"engine"}
        or model["engine"]
        not in {
            "integer_reference",
            "specir_fp_reduce",
        }
    ):
        raise RtlIntakeRefusal("minimal numerical model must be a generic independent engine without source hooks")
    for key in ("operand_dtype", "accumulator_dtype", "readout_dtype"):
        _dtype(document.get(key))
    if "overflow" in document and document["overflow"] not in {"bounded_exact", "modular_wrap", "saturate"}:
        raise RtlIntakeRefusal("minimal overflow must be an explicit numerical choice")
    for key, allowed in (
        ("rounding", {"rne", "rmm", "rtz", "rdn", "rup"}),
        ("reduction_order", {"index_sequential", "tree", "pairwise"}),
        ("reduction_cadence", {"per_step", "single_final"}),
        ("product_rounding", {"accumulator_format"}),
    ):
        if key in document and document[key] not in allowed:
            raise RtlIntakeRefusal("minimal numerical choices cannot carry arbitrary descriptions")


def _signature(signature) -> None:
    if not isinstance(signature, dict) or set(signature) - _SIGNATURE:
        raise RtlIntakeRefusal("minimal operation signatures refuse shape, layout, schedule or descriptive metadata")
    for key, value in signature.items():
        if key.endswith("_dtypes"):
            if not isinstance(value, list) or not value:
                raise RtlIntakeRefusal("minimal dtype constraints require nonempty explicit lists")
            for dtype in value:
                _dtype(dtype)
        elif key.endswith("_dtype"):
            _dtype(value)
        elif key in {"broadcasting", "aliasing"} and (
            not isinstance(value, str) or value not in {"allow", "forbid", "none", "numpy", "elementwise"}
        ):
            raise RtlIntakeRefusal("minimal semantic constraints cannot carry arbitrary descriptions")


def validate_minimal_software(document, *, target: str) -> dict:
    """Validate raw bytes before legacy normalization can discard extra fields."""
    if not isinstance(document, dict) or set(document) != _ROOT or document["status"] != "reviewed":
        raise RtlIntakeRefusal("fresh software needs exactly the reviewed minimal semantic schema; legacy specs refuse")
    _numeric(document["numerical_semantics"])
    operations = document["operations"]
    rows = list(operations.values()) if isinstance(operations, dict) else operations
    if not isinstance(rows, list) or not rows:
        raise RtlIntakeRefusal("minimal software operations must be explicit nonempty declarations")
    for row in rows:
        if not isinstance(row, dict) or set(row) - _OPERATION:
            raise RtlIntakeRefusal(
                "minimal operations refuse evidence, descriptions, capture, backend or tuning metadata"
            )
        if "signature" in row:
            _signature(row["signature"])
        if "placement" in row and row["placement"] not in {"host", "accelerator", "fused_accelerator"}:
            raise RtlIntakeRefusal("minimal placement is a semantic permission, never a schedule description")
        if row.get("status", "reviewed") != "reviewed":
            raise RtlIntakeRefusal("minimal operation semantics need explicit protected review")
    try:
        return validate_software_spec(document, target=target)
    except (ValueError, TypeError) as error:
        raise RtlIntakeRefusal(str(error)) from error
