"""Separate reviewed arithmetic declarations from observed storage/compute widths."""

from __future__ import annotations

from merlin.targetgen.operator_review import review_record


def integer_split_k_limit(
    numeric_policy: dict | None, *, tile_dim: int, full_k: int, operand_range: tuple[int, int] | None = None
) -> dict | None:
    """Derive the worst-case signed-integer primitive K from selected arithmetic.

    This is a target-neutral numerical obligation shared by capture selection and
    actual mesh dispatch. A full contraction may need several bounded primitive
    tiles, followed by exact i32 aggregation before its epilogue. The caller
    must separately establish that the declaration was reviewed and that a
    compiler performs the split; this calculation proves neither.
    """
    semantics = (numeric_policy or {}).get("numerical_semantics") or {}
    internal = semantics.get("internal_arithmetic") or {}
    if not internal:
        return None
    if internal.get("full_operation_overflow_policy") != "bounded_exact_requires_each_partial_sum":
        raise ValueError("integer mesh has no supported selected partial-sum policy")
    operand_bits = internal.get("signed_operand_bits")
    mac_bits = internal.get("mac_result_bits")
    if (
        type(operand_bits) is not int
        or type(mac_bits) is not int
        or not 2 <= operand_bits <= 32
        or not 2 <= mac_bits <= 64
        or semantics.get("accumulator_dtype") not in {"i32", "int32"}
        or internal.get("mac_result_overflow") != f"wrap_to_{mac_bits}_bits"
        or type(tile_dim) is not int
        or tile_dim < 1
        or type(full_k) is not int
        or full_k < 1
    ):
        raise ValueError("integer mesh has unsupported selected MAC/accumulator arithmetic")
    storage_lo, storage_hi = -(1 << (operand_bits - 1)), (1 << (operand_bits - 1)) - 1
    lo, hi = operand_range if operand_range is not None else (storage_lo, storage_hi)
    if type(lo) is not int or type(hi) is not int or not storage_lo <= lo <= hi <= storage_hi:
        raise ValueError("declared operand range exceeds selected signed storage")
    maximum_operand = max(abs(lo), abs(hi))
    maximum_product = maximum_operand * maximum_operand
    full_bound = full_k * maximum_product
    if full_bound > (1 << 31) - 1:
        raise ValueError("integer mesh full contraction may exceed exact i32 aggregation")
    limit = ((1 << (mac_bits - 1)) - 1) // maximum_product
    aligned = (limit // tile_dim) * tile_dim
    return {
        "max_true_k": limit,
        "max_padded_k": limit,
        "tile_k": aligned,
        "full_result_bound": full_bound,
        "mac_result_bits": mac_bits,
        "signed_operand_bits": operand_bits,
        "maximum_product": maximum_product,
        "operand_range": [lo, hi],
        "aggregation": "exact_integer_sum_before_epilogue",
    }


def integer_partial_sum_bound(
    semantics: dict, *, reduction_extent: int, lhs_values, rhs_values, initial_values=()
) -> dict:
    """Sufficient absolute bound for every partial sum of a concrete contraction.

    This is not a tensor execution model. The caller must select the actual
    reduction extent and include all initial accumulator/bias values. The bound
    deliberately ignores cancellation and conservatively uses the positive
    signed limit, so it cannot approve an overflowing internal MAC by relying on
    wider storage or a small final output.
    """
    if type(reduction_extent) is not int or reduction_extent < 1:
        raise ValueError("reduction_extent must be a positive integer")
    internal = semantics.get("internal_arithmetic") or {}
    bits, operand_bits = internal.get("mac_result_bits"), internal.get("signed_operand_bits")
    if type(bits) is not int or bits < 2 or type(operand_bits) is not int or operand_bits < 2:
        return {"status": "unknown", "reason": "selected internal signed operand/result widths are absent"}
    values = [list(lhs_values), list(rhs_values), list(initial_values)]
    if not values[0] or not values[1] or any(type(value) is not int for vector in values for value in vector):
        raise ValueError("partial-sum bounds require nonempty integral lhs/rhs values")
    lo, hi = -(1 << (operand_bits - 1)), (1 << (operand_bits - 1)) - 1
    if any(not lo <= value <= hi for vector in values[:2] for value in vector):
        raise ValueError("observed operands exceed the selected signed operand width")
    lhs, rhs, initial = (max((abs(value) for value in vector), default=0) for vector in values)
    bound = initial + reduction_extent * lhs * rhs
    limit = (1 << (bits - 1)) - 1
    return {
        "status": "proven_safe" if bound <= limit else "may_overflow",
        "scope": "all partial sums of the supplied contraction and initial addends; not full-kernel execution",
        "bound": bound,
        "signed_positive_limit": limit,
        "mac_result_bits": bits,
        "reduction_extent": reduction_extent,
        "maximum_absolute_operands": [lhs, rhs],
        "maximum_absolute_initial_addend": initial,
        "formula": "max_abs(initial) + reduction_extent * max_abs(lhs) * max_abs(rhs)",
    }


def _unknowns(value, path="semantics") -> list[str]:
    if isinstance(value, dict):
        return [unknown for key, member in value.items() for unknown in _unknowns(member, f"{path}.{key}")]
    if isinstance(value, list):
        return [unknown for index, member in enumerate(value) for unknown in _unknowns(member, f"{path}.{index}")]
    return (
        [path]
        if value is None or (isinstance(value, str) and (not value or value.lower().startswith("unknown")))
        else []
    )


def _explicit_contract(contract: dict | None) -> dict:
    if not isinstance(contract, dict):
        return {"status": "unknown", "reason": "no operation-scoped reviewed numerical contract"}
    semantics, evidence = contract.get("semantics"), contract.get("evidence")
    unknowns = _unknowns(semantics) if isinstance(semantics, dict) and semantics else ["semantics"]
    if not isinstance(evidence, dict) or not evidence:
        unknowns.append("evidence")
    review = review_record(contract, default="unreviewed")
    if not review["operator_accepted"]:
        unknowns.append("review_status")
    return {
        "status": "unknown" if unknowns else "resolved",
        "semantics": semantics,
        "unknowns": sorted(unknowns),
        "qualification": "accepted declaration, not measured numerical conformance",
        **review,
    }


def operation_numerical_contracts(entry: dict, software_spec: dict | None, host_capabilities: dict | None) -> dict:
    row = entry["observed_signature"]
    spec = software_spec or {}
    declarations = {declaration["id"]: declaration for declaration in spec.get("operations") or []}
    accelerator = []
    for admission in entry["software_admissions"]:
        if admission["placement"] not in {"accelerator", "fused_accelerator"} or admission["status"] != "admitted":
            continue
        declaration = declarations[admission["declaration"]]
        if "numerical_contract" in declaration:
            decision = _explicit_contract(declaration["numerical_contract"])
        elif row.get("semantic_family") == "contraction":
            semantics = spec.get("numerical_semantics") or {}
            unknowns = _unknowns(semantics)
            engine = (semantics.get("model") or {}).get("engine")
            required = {"operand_dtype", "accumulator_dtype", "readout_dtype", "subnormal_operand_flush"}
            if engine == "integer_reference":
                required.add("overflow")
            elif engine == "specir_fp_reduce":
                required.update({"rounding", "reduction_order", "reduction_cadence", "product_rounding"})
            else:
                unknowns.append("model.engine")
            unknowns.extend(sorted(required - set(semantics)))
            review = review_record(spec, default="unreviewed")
            if not review["operator_accepted"]:
                unknowns.append("software_spec_review")
            decision = {
                "status": "unknown" if unknowns else "resolved",
                "unknowns": sorted(set(unknowns)),
                "semantics": semantics,
                "basis": "selected contraction numerical semantics",
                "qualification": "accepted declaration, not measured numerical conformance",
                **review,
            }
        else:
            decision = _explicit_contract(None)
        accelerator.append({"declaration": declaration["id"], **decision})
    host = []
    for profile in entry["host_admission"].get("profiles") or []:
        if profile["status"] != "admitted" or not profile.get("operator_accepted", profile.get("reviewed")):
            continue
        selected = (host_capabilities or {}).get(profile["profile"]) or {}
        document = selected.get("capability_spec") or {}
        declarations = {declaration["id"]: declaration for declaration in document.get("operations") or []}
        for admission in profile.get("decisions") or []:
            if admission["status"] != "admitted":
                continue
            declaration = declarations[admission["declaration"]]
            host.append(
                {
                    "profile": profile["profile"],
                    "declaration": declaration["id"],
                    **_explicit_contract(declaration.get("numerical_contract")),
                }
            )

    def summarize(decisions):
        return {
            "status": "resolved" if any(decision["status"] == "resolved" for decision in decisions) else "unknown",
            "declarations": decisions,
            "scope": "operation-specific; widths and precision profile alone are insufficient",
        }

    return {"accelerator": summarize(accelerator), "host": summarize(host)}
