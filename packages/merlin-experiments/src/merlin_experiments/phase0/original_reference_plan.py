"""Closed original typed reference selections and preallocation accounting.

Counts are complete logical allocation/work bounds, not Python heap, framework
workspace, compiler resources or target cycles. No policy status grants them.
"""

from __future__ import annotations

import math
from dataclasses import fields

from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, OriginalReferencePolicy
from merlin.targetgen.original_reference_values import format_record, integer_project, round_f32

from . import component_execution_budget as E
from .original_call_sources import validate_budget

SCHEMA = "merlin.original_reference_selection.v1"
BATCH_SCHEMA = "merlin.original_reference_selection.v2"
POINTWISE_SCHEMA = "merlin.original_reference_selection.v3"
TRANSPOSE_SCHEMA = "merlin.original_reference_selection.v4"
COHORTS = ("functional_guard", "withheld_transfer")


def validate(selection):
    selection_fields = {
        "schema",
        "operator_schema_intake_sha256",
        "semantic_basis_sha256",
        "policies",
        "input_palettes",
        "cohorts",
        "source_budget",
        "execution_budget",
        "reference_budget",
        "byteorder",
    }
    if isinstance(selection, dict) and selection.get("schema") in {BATCH_SCHEMA, POINTWISE_SCHEMA, TRANSPOSE_SCHEMA}:
        selection_fields.add("native_observations")
    if (
        not isinstance(selection, dict)
        or set(selection) != selection_fields
        or selection["schema"] not in {SCHEMA, BATCH_SCHEMA, POINTWISE_SCHEMA, TRANSPOSE_SCHEMA}
        or (
            selection["schema"] in {BATCH_SCHEMA, POINTWISE_SCHEMA, TRANSPOSE_SCHEMA}
            and selection["native_observations"] != "batch.v1"
        )
    ):
        raise ValueError("original references require a closed independently selected contract")
    for key in ("operator_schema_intake_sha256", "semantic_basis_sha256"):
        value = selection[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("original reference selection needs exact live source identities")
    validate_budget(selection["source_budget"])
    E.validate(selection["execution_budget"])
    budget = selection["reference_budget"]
    if not isinstance(budget, dict) or set(budget) != {field.name for field in fields(OriginalReferenceBudget)}:
        raise ValueError("original reference requires complete explicit reference limits")
    OriginalReferenceBudget(**budget).verify()
    if selection["byteorder"] not in {"little", "big"}:
        raise ValueError("original reference requires explicitly selected binary byte order")
    cohorts = selection["cohorts"]
    if not isinstance(cohorts, dict) or set(cohorts) != set(COHORTS):
        raise ValueError("original references require complete guard and private cohort declarations")
    for extents in cohorts.values():
        if (
            not isinstance(extents, list)
            or not extents
            or any(type(n) is not int or n < 1 for n in extents)
            or len(set(extents)) != len(extents)
        ):
            raise ValueError("original reference cohorts need unique explicit positive fresh extents")
    policies, seen = selection["policies"], set()
    if not isinstance(policies, list):
        raise ValueError("original reference requires explicitly selected operation-local policies")
    for record in policies:
        selected = policy(
            record,
            pointwise=selection["schema"] in {POINTWISE_SCHEMA, TRANSPOSE_SCHEMA},
            transpose=selection["schema"] == TRANSPOSE_SCHEMA,
        )
        key = (selected.operation, selected.operand_dtypes, selected.readout_dtypes)
        if key in seen:
            raise ValueError("original reference policy selector is duplicated/ambiguous")
        seen.add(key)
    palettes, seen = selection["input_palettes"], set()
    if not isinstance(palettes, list):
        raise ValueError("original reference requires explicit typed signed input palettes")
    for row in palettes:
        if (
            not isinstance(row, dict)
            or set(row) != {"dtype", "values"}
            or not isinstance(row["dtype"], str)
            or row["dtype"] in seen
            or not isinstance(row["values"], list)
            or not row["values"]
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in row["values"])
            or not min(row["values"]) < 0 < max(row["values"])
        ):
            raise ValueError("original reference palettes need unique finite signed typed values")
        seen.add(row["dtype"])
    return selection


def transport(selection):
    return (
        "batch.v1"
        if validate(selection)["schema"] in {BATCH_SCHEMA, POINTWISE_SCHEMA, TRANSPOSE_SCHEMA}
        else "per_member"
    )


def policy(record, *, pointwise=False, transpose=False):
    from merlin.targetgen.original_operator_reference import POLICY_SCHEMA

    names = {field.name for field in fields(OriginalReferencePolicy)}
    if type(pointwise) is not bool or type(transpose) is not bool:
        raise ValueError("original reference policy version selection must be an explicit Boolean")
    if transpose:
        from merlin.targetgen.original_transpose_reference import POLICY_SCHEMA as TRANSPOSE_POLICY_SCHEMA
        from merlin.targetgen.original_transpose_reference import OriginalTransposeReferencePolicy

        if isinstance(record, dict) and record.get("schema") == TRANSPOSE_POLICY_SCHEMA:
            names = {field.name for field in fields(OriginalTransposeReferencePolicy)}
            if set(record) != {"schema", *names}:
                raise ValueError("transpose reference policy needs its complete separate storage contract")
            values = {key: record[key] for key in names}
            if any(type(values[key]) is not str for key in ("operation", "movement", "comparison")):
                raise ValueError("transpose reference policy needs exact string-valued storage choices")
            if type(values["finite_only"]) is not bool:
                raise ValueError("transpose reference policy needs an explicit finite domain choice")
            for key in ("operand_dtypes", "readout_dtypes"):
                if type(values[key]) is not list or not values[key] or any(type(t) is not str for t in values[key]):
                    raise ValueError("transpose reference policy needs complete ordered storage selectors")
                values[key] = tuple(values[key])
            return OriginalTransposeReferencePolicy(**values)
    schemas = {POLICY_SCHEMA}
    policy_type = OriginalReferencePolicy
    if pointwise:
        from merlin.targetgen.original_pointwise_reference import POLICY_SCHEMA as POINTWISE_POLICY_SCHEMA
        from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy

        schemas.add(POINTWISE_POLICY_SCHEMA)
        if isinstance(record, dict) and record.get("schema") == POINTWISE_POLICY_SCHEMA:
            policy_type = OriginalPointwiseReferencePolicy
    if not isinstance(record, dict) or set(record) != {"schema", *names} or record["schema"] not in schemas:
        raise ValueError("original reference policy must retain its complete explicit operation-local schema")
    values = {key: record[key] for key in names}
    if any(
        type(values[key]) is not str
        for key in names
        - {"operand_dtypes", "readout_dtypes", "finite_only", "subnormal_operand_flush", "atol", "rtol"}
    ):
        raise ValueError("original reference policy requires explicit string-valued semantics")
    if any(type(values[key]) is not bool for key in ("finite_only", "subnormal_operand_flush")):
        raise ValueError("original reference policy requires explicit boolean numerical choices")
    if any(
        type(values[key]) is not float or not math.isfinite(values[key]) or values[key] < 0 for key in ("atol", "rtol")
    ):
        raise ValueError("original reference policy requires explicit finite nonnegative tolerances")
    for key in ("operand_dtypes", "readout_dtypes"):
        if not isinstance(values[key], list) or not values[key] or any(type(t) is not str for t in values[key]):
            raise ValueError("original reference policy requires complete ordered dtype selectors")
        values[key] = tuple(values[key])
    # Unsupported numerical contracts remain required per-call unknown rows.
    # Structural selection validity does not call verify or mint implementation.
    return policy_type(**values)


def selected_policy(selection, form):
    rows = [
        policy(
            row,
            pointwise=selection["schema"] in {POINTWISE_SCHEMA, TRANSPOSE_SCHEMA},
            transpose=selection["schema"] == TRANSPOSE_SCHEMA,
        )
        for row in selection["policies"]
    ]
    rows = [
        row
        for row in rows
        if (row.operation, row.operand_dtypes, row.readout_dtypes)
        == (form["target"], tuple(form["operand_dtypes"]), tuple(form["result_dtypes"]))
    ]
    if len(rows) != 1:
        raise ValueError("original call lacks an exact independently selected operation/dtype numerical policy")
    rows[0].verify()
    return rows[0]


def palette(selection, dtype):
    rows = [row["values"] for row in selection["input_palettes"] if row["dtype"] == dtype]
    if len(rows) != 1:
        raise ValueError("original call lacks an explicit signed input palette in its original storage")
    fmt, values = format_record(dtype), rows[0]
    for value in values:
        if fmt["kind"] == "int_affine":
            integer_project(value, dtype, "bounded_exact")
        elif round_f32(value) != value:
            raise ValueError("original reference palette changes values when encoded in original storage")
    return values


def comparison_bits(selected):
    """Bound logical rational intermediates in the complete f32 comparison.

    The exact comparison converts f32 values and explicit binary64 tolerances
    to rationals. Bounds include numerator/denominator products and comparison
    cross products. Fraction normalization workspace and Python heap stay
    unqualified; this is not a process memory or arithmetic-time bound.
    """
    if selected.arithmetic != "finite_f32":
        return 0
    fmt = format_record(selected.readout_dtypes[0])
    numerator = 1 << (fmt["exp_bits"] - 1)
    denominator = numerator - 1 + fmt["mant_bits"]
    # These are bit lengths, including the denominator's leading one.
    denominator += 1
    ratios = [value.as_integer_ratio() for value in (selected.atol, selected.rtol)]
    (an, ad), (rn, rd) = [(abs(n).bit_length(), d.bit_length()) for n, d in ratios]
    pn, pd = rn + numerator, rd + denominator
    tn, td = max(an + pd, pn + ad) + 1, ad + pd
    dn, dd = numerator + denominator + 1, 2 * denominator
    return max(numerator, denominator, an, ad, rn, rd, pn, pd, tn, td, dn, dd, dn + td, tn + dd)


def measure(contract, selection):
    """Count both native execution and independent check before shaped data.

    Each named row counts one possible materialization, including raw/native
    copies, both reference evaluations (evaluate and compare), and full decoded
    comparisons. Scalar visits/products/prefixes use a conservative 64-bit
    logical element count; Python container/allocator bytes are not certified.
    """
    metadata = contract.verify()
    inputs, outputs = metadata["inputs"], metadata["outputs"]
    input_count = sum(math.prod(row["shape"]) for row in inputs)
    output_count = sum(math.prod(row["shape"]) for row in outputs)
    input_bytes = sum(math.prod(row["shape"]) * (format_record(row["dtype"])["element_bits"] // 8) for row in inputs)
    output_bytes = sum(math.prod(row["shape"]) * (format_record(row["dtype"])["element_bits"] // 8) for row in outputs)
    allocations = []

    def add(role, count, payload):
        allocations.append({"role": role, "elements": count, "payload_bytes": payload})

    for role in (
        "stimulus_values",
        "stimulus_encoding_values",
        "reference_input_decode_first",
        "reference_input_decode_comparison",
    ):
        add(role, input_count, 8 * input_count)
    for role in ("stimulus_raw", "native_raw_decode", "native_bytearray", "native_input_clone", "native_examples"):
        add(role, input_count, input_bytes)
    if metadata["target"] == "aten.transpose.int":
        for role in ("native_alias_input_bytes_before", "native_alias_input_bytes_after"):
            add(role, input_count, input_bytes)
        # Two rank-two stride vectors, offsets, indices, byte counts and the
        # storage-contact flag are explicit separate finite observation work.
        add("native_alias_geometry", 13, 13 * 8)
    for role in ("stimulus_hex", "native_input_hex"):
        add(role, input_count, 2 * input_bytes)
    for role in ("native_output", "native_contiguous_copy", "native_output_bytes", "parent_actual_bytes"):
        add(role, output_count, output_bytes)
    for run in ("first", "comparison"):
        for role in ("reference_result", "reference_projected", "reference_encoding"):
            add(role + "_" + run, output_count, 8 * output_count)
        for role in ("reference_element_bytes", "reference_raw"):
            add(role + "_" + run, output_count, output_bytes)
        add("reference_hex_" + run, output_count, 2 * output_bytes)
    for role in ("native_output_hex", "parent_actual_hex"):
        add(role, output_count, 2 * output_bytes)
    for role in ("comparison_reference_decode", "comparison_actual_decode", "mismatch_expected", "mismatch_actual"):
        add(role, output_count, 8 * output_count)
    from merlin.targetgen.original_operator_reference import _costs

    arithmetic = _costs(metadata)[2]
    for role in ("reference_intermediates_first", "reference_intermediates_comparison"):
        add(role, arithmetic, 8 * arithmetic)
    palette_values = sum(len(palette(selection, row["dtype"])) for row in inputs)
    add("palette_values", palette_values, 8 * palette_values)
    rational_bits = comparison_bits(contract.policy)
    if rational_bits:
        # Logical rational operands/results and comparison cross products.
        add("comparison_rational_parts", 32 * output_count, 32 * output_count * ((rational_bits + 7) // 8))
    return {
        "reference_work": 3 * arithmetic + sum(row["elements"] for row in allocations),
        "materialized_elements": sum(row["elements"] for row in allocations),
        "tensor_payload_bytes": sum(row["payload_bytes"] for row in allocations),
        "scalar_bits": max(64, format_record(contract.policy.accumulator_dtype)["element_bits"], rational_bits),
        "allocations": allocations,
    }
