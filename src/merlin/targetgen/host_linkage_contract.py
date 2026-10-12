"""Closed, opt-in host library-supplier requirement; never a numerical grant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.common.digest import is_sha256
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags

SCHEMA = "merlin.host_linkage_contract.v1"
SUPPLIER = "selected_math_archive"


def validate_linkage_contract(value: object) -> dict[str, Any]:
    """Require an exact caller-selected archive digest and bounded symbol roster."""
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "supplier", "archive_sha256", "symbols"}
        or value.get("schema") != SCHEMA
        or value.get("supplier") != SUPPLIER
        or not is_sha256(value.get("archive_sha256"))
        or not isinstance(value.get("symbols"), list)
    ):
        raise ValueError("host linkage contract is not a closed selected archive roster")
    symbols = list(value["symbols"])
    trace_symbol_flags(symbols)
    if symbols != sorted(symbols):
        raise ValueError("host linkage symbols must be in canonical order")
    return {"schema": SCHEMA, "supplier": SUPPLIER, "archive_sha256": value["archive_sha256"], "symbols": symbols}


def required_composite_math_symbol(schema: object, operation: object) -> str | None:
    """Name an unresolved selected-library obligation, not an observed call."""
    from merlin.frontends.linalg_composite_math import STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA
    from merlin.frontends.linalg_mixed_patterns import SCHEMA as MIXED_SCHEMA

    if schema == MIXED_SCHEMA and type(operation) is str:
        return {"f64_math_sin": "sin", "f64_math_cos": "cos", "f64_arith_constant_math_powf": "pow"}.get(operation)
    if schema != STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA or type(operation) is not str:
        return None
    return {"literal_base_pow_f32": "powf", "tanh_gelu_f32": "tanhf"}.get(operation)


def validate_source_linkage_contract(schema: object, operation: object, value: object) -> dict[str, Any]:
    """Restrict new composite forms to their exact selected symbol requirement."""
    from merlin.frontends.linalg_composite_math import STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA
    from merlin.frontends.linalg_math_patterns import STATIC_F32_MATH_SOURCE_BODY_SCHEMA

    contract = validate_linkage_contract(value)
    if (
        schema == STATIC_F32_MATH_SOURCE_BODY_SCHEMA
        and type(operation) is str
        and operation in {"math.sin", "math.cos"}
    ):
        return contract  # Preserve the previously reviewed unary-math contract.
    symbol = required_composite_math_symbol(schema, operation)
    from merlin.frontends.linalg_mixed_patterns import SCHEMA as MIXED_SCHEMA

    if (
        schema not in {STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, MIXED_SCHEMA}
        or symbol is None
        or contract["symbols"] != [symbol]
    ):
        raise ValueError("linkage contract does not match the exact composite math source form")
    return contract
