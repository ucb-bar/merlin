"""Compare described instruction type/rank signatures with selected software records."""

from __future__ import annotations

from typing import Any

_CHECKED_SW_SIGNATURE_FIELDS = frozenset(
    {
        "operand_dtypes",
        "dtypes",
        "accumulator_dtype",
        "readout_dtype",
        "ranks",
        "ordered_operand_dtypes",
        "ordered_result_dtypes",
    }
)

_DTYPE_ALIASES = {"int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64", "fp32": "f32"}


def _mlir_dtype_rank(type_string: str) -> tuple[str, int] | None:
    """Read simple ranked MLIR tensor/vector types; unfamiliar syntax stays unknown."""
    source = type_string.strip()
    if source.startswith(("tensor<", "vector<")) and source.endswith(">"):
        body = source[source.index("<") + 1 : -1]
        if "," in body or body.startswith("*"):
            return None
        parts = body.split("x")
        if len(parts) < 2 or any(not part or (part != "?" and not part.isdecimal()) for part in parts[:-1]):
            return None
        return parts[-1], len(parts) - 1
    if "<" in source or ">" in source or not source:
        return None
    return source, 0


def _software_signature_unknowns(
    declaration: dict[str, Any], operands: list[dict[str, str]], results: list[dict[str, str]]
) -> list[str]:
    """Screen only type/rank axes we can compare without guessing numerics."""
    signature = declaration.get("signature")
    if not isinstance(signature, dict):
        signature = {name: declaration[name] for name in _CHECKED_SW_SIGNATURE_FIELDS if name in declaration}
    if not signature:
        return ["software_signature_missing"]
    unknowns = [
        f"software_signature_unchecked.{field}" for field in sorted(set(signature) - _CHECKED_SW_SIGNATURE_FIELDS)
    ]
    if any(field in declaration for field in ("ops", "families", "family")):
        unknowns.append("software_selector_unchecked")
    parsed_operands = [_mlir_dtype_rank(row["type"]) for row in operands]
    parsed_results = [_mlir_dtype_rank(row["type"]) for row in results]
    if any(item is None for item in parsed_operands + parsed_results):
        return [*unknowns, "software_signature_unparsed_mlir_type"]

    def _normalize_dtype(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return _DTYPE_ALIASES.get(value, value)

    def _check_types(rows: list[tuple[str, int]], *, ordered_field: str, allowed_fields: tuple[str, ...], role: str):
        ordered = signature.get(ordered_field)
        if ordered is not None:
            if not isinstance(ordered, list) or len(ordered) != len(rows):
                unknowns.append(f"software_signature_{role}_arity_mismatch")
                return
            for index, (observed, _) in enumerate(rows):
                if _normalize_dtype(ordered[index]) != observed:
                    unknowns.append(f"software_signature_{role}_dtype_mismatch.{index}")
            return
        allowed = set()
        for field in allowed_fields:
            declared = signature.get(field)
            if isinstance(declared, list):
                allowed.update(_normalize_dtype(item) for item in declared)
            elif declared is not None:
                allowed.add(_normalize_dtype(declared))
        if not allowed:
            unknowns.append(f"software_signature_{role}_dtype_unconstrained")
            return
        for index, (observed, _) in enumerate(rows):
            if observed not in allowed:
                unknowns.append(f"software_signature_{role}_dtype_mismatch.{index}")

    _check_types(
        parsed_operands,
        ordered_field="ordered_operand_dtypes",
        allowed_fields=("operand_dtypes", "dtypes"),
        role="operand",
    )
    _check_types(
        parsed_results,
        ordered_field="ordered_result_dtypes",
        allowed_fields=("readout_dtype", "accumulator_dtype", "dtypes"),
        role="result",
    )
    ranks = signature.get("ranks")
    if ranks is not None:
        if not isinstance(ranks, list) or any(type(rank) is not int or rank < 0 for rank in ranks):
            unknowns.append("software_signature_ranks_invalid")
        else:
            for role, rows in (("operand", parsed_operands), ("result", parsed_results)):
                for index, (_, rank) in enumerate(rows):
                    if rank not in ranks:
                        unknowns.append(f"software_signature_{role}_rank_mismatch.{index}")
    return unknowns
