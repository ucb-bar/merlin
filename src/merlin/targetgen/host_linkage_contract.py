"""Closed, opt-in host library-supplier requirement; never a numerical grant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.common.digest import is_sha256
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags

SCHEMA = "merlin.host_linkage_contract.v1"
SUPPLIER = "selected_math_archive"
RUNTIME_SCHEMA = "merlin.host_runtime_linkage_contract.v1"
RUNTIME_SUPPLIER = "selected_host_math_runtime_object"


def runtime_recipe(contract: Mapping[str, Any], policy: str) -> dict[str, Any]:
    """Match an explicit runtime selection to reviewed source and strict flags."""
    from hashlib import sha256

    from merlin.runtime.host_math import host_math_recipe

    selected = [row for row in contract["recipes"] if row["policy"] == policy]
    recipe = host_math_recipe(policy)
    actual = {
        "policy": policy,
        "source_sha256": sha256(recipe.source.encode()).hexdigest() if recipe.source is not None else None,
        "compile_flags": list(recipe.compile_flags),
        "link_flags": list(recipe.link_flags),
    }
    if len(selected) != 1 or selected[0] != actual:
        raise ValueError("reviewed runtime supplier differs from selected host math recipe/source/flags")
    wrappers = {flag.removeprefix("-Wl,--wrap=") for flag in recipe.link_flags if flag.startswith("-Wl,--wrap=")}
    if not set(contract["symbols"]) <= wrappers:
        raise ValueError("reviewed runtime supplier does not wrap every selected source symbol")
    return actual


def _runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        set(value) != {"schema", "supplier", "symbols", "recipes"}
        or value.get("schema") != RUNTIME_SCHEMA
        or value.get("supplier") != RUNTIME_SUPPLIER
        or not isinstance(value.get("symbols"), list)
        or not isinstance(value.get("recipes"), list)
        or not 1 <= len(value["recipes"]) <= 16
    ):
        raise ValueError("host linkage contract is not a closed selected runtime roster")
    symbols = list(value["symbols"])
    trace_symbol_flags(symbols)
    if symbols != sorted(symbols):
        raise ValueError("host linkage symbols must be in canonical order")
    recipes = []
    for row in value["recipes"]:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"policy", "source_sha256", "compile_flags", "link_flags"}
            or not isinstance(row.get("policy"), str)
            or not row["policy"]
            or not is_sha256(row.get("source_sha256"))
            or any(
                not isinstance(row.get(key), list)
                or not row[key]
                or any(not isinstance(flag, str) or not flag for flag in row[key])
                for key in ("compile_flags", "link_flags")
            )
        ):
            raise ValueError("selected runtime recipe identity is incomplete")
        recipes.append({key: list(item) if isinstance(item, list) else item for key, item in row.items()})
    policies = [row["policy"] for row in recipes]
    if policies != sorted(set(policies)):
        raise ValueError("selected runtime policies must be unique and canonically ordered")
    return {"schema": RUNTIME_SCHEMA, "supplier": RUNTIME_SUPPLIER, "symbols": symbols, "recipes": recipes}


def validate_linkage_contract(value: object) -> dict[str, Any]:
    """Require an exact caller-selected archive digest and bounded symbol roster."""
    if isinstance(value, Mapping) and value.get("supplier") == RUNTIME_SUPPLIER:
        return _runtime_contract(value)
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
    if contract["supplier"] == RUNTIME_SUPPLIER:
        unary = {"math.sin": "sinf", "math.cos": "cosf", "math.exp": "expf"}
        symbol = (
            unary.get(operation)
            if schema == STATIC_F32_MATH_SOURCE_BODY_SCHEMA and type(operation) is str
            else required_composite_math_symbol(schema, operation)
        )
        if symbol not in {"sinf", "cosf", "expf", "powf", "tanhf"} or contract["symbols"] != [symbol]:
            raise ValueError("runtime linkage contract does not match the exact f32 math source form")
        return contract
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
