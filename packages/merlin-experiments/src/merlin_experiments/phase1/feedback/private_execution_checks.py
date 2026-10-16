"""Revalidate the complete declared numerical checks of a private execution record.

These are structural checks over host-owned results, not a second numerical
oracle or an authority for candidate-supplied receipts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def _count(value):
    return type(value) is int and value >= 0


def _rows(value, indices):
    return (
        isinstance(value, list)
        and len(value) == len(indices)
        and all(
            isinstance(row, Mapping) and row.get("index") == index
            and type(row.get("index")) is int and row.get("passed") is True
            for row, index in zip(value, indices, strict=True)
        )
    )


def complete_checks(checks, entry):
    """Require the descriptor's checks and complete ordered result coverage."""
    required = {"completion", "integer_reference", "host_execution", "per_group_exactness"}
    sanity = entry.get("fp32_sanity")
    if sanity is not None:
        required.add("fp32_sanity")
    if not isinstance(checks, Mapping) or set(checks) != required:
        return False
    if any(not isinstance(checks[name], Mapping) or checks[name].get("passed") is not True
           for name in required - {"per_group_exactness"}):
        return False
    scope = checks["per_group_exactness"]
    if scope.get("status") != "not_observable" or "passed" in scope:
        return False
    completion = checks["completion"]
    results, elements = completion.get("results"), completion.get("elements")
    if not _count(results) or results == 0 or not _count(elements):
        return False
    integer, host = checks["integer_reference"], checks["host_execution"]
    integer_rows = integer.get("results", [])
    if integer.get("status") == "compared":
        if (
            integer.get("reference") not in entry["integer_reference"]
            or not isinstance(integer_rows, list) or not 0 < len(integer_rows) <= results
            or not _rows(integer_rows, range(len(integer_rows)))
        ):
            return False
    elif integer.get("status") == "not_available":
        if (
            entry["require_integer_reference"] is not False
            or integer.get("looked_for") != entry["integer_reference"]
            or integer_rows != []
        ):
            return False
    else:
        return False
    covered, counts = set(), {}
    for row in integer_rows:
        if row.get("status") == "unconfirmed_reference":
            if (
                row.get("reference_confirmed_by_host_execution") is not False
                or not _count(row.get("host_mismatched_elements"))
                or row["host_mismatched_elements"] == 0
            ):
                return False
            continue
        if (
            row.get("status") is not None or type(row.get("mismatched_elements")) is not int
            or row["mismatched_elements"] != 0 or not _count(row.get("of"))
        ):
            return False
        covered.add(row["index"])
        counts[row["index"]] = row["of"]
    judged = [index for index in range(results) if index not in covered]
    host_rows, tolerance = host.get("results"), entry.get("host_execution")
    if not _rows(host_rows, judged) or (judged and tolerance is None):
        return False
    for row in host_rows:
        if (
            not _count(row.get("of")) or type(row.get("within")) is not int
            or row["within"] != row["of"]
            or row.get("atol") != tolerance["atol"] or row.get("rtol") != tolerance["rtol"]
        ):
            return False
        counts[row["index"]] = row["of"]
    if len(counts) != results or sum(counts.values()) != elements:
        return False
    if sanity is None:
        return True
    check = checks["fp32_sanity"]
    digest = check.get("sha256")
    if (
        check.get("reference") != sanity["reference"]
        or type(check.get("receipt_bound")) is not bool
        or not isinstance(digest, str) or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or check.get("cosine_margin") != sanity["cosine_margin"]
        or not _rows(check.get("results"), range(results))
    ):
        return False
    from .private_full_model_execution import COSINE_DISTANCE_FLOOR

    for row in check["results"]:
        if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
               for key in ("cosine", "floor", "quantized_program_cosine")):
            return False
        expected_floor = 1.0 - sanity["cosine_margin"] * max(
            1.0 - row["quantized_program_cosine"], COSINE_DISTANCE_FLOOR,
        )
        if row["floor"] != expected_floor or row["cosine"] < row["floor"]:
            return False
    top1 = check.get("top1")
    if not isinstance(top1, Mapping):
        return False
    if not sanity["top1"]:
        return top1 == {"status": "not_declared"}
    return (
        top1.get("passed") is True
        and all(_count(top1.get(key)) for key in ("rows", "applicable_rows", "agreeing_rows"))
        and top1["agreeing_rows"] == top1["applicable_rows"] <= top1["rows"]
    )
