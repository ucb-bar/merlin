"""Data-owned conditions over observed operation semantics; absent evidence fails closed."""

from __future__ import annotations

import math
from collections.abc import Mapping

# Closed dispatch names implemented on decoded, retired commands in composition_audit.
INSTRUCTION_CHECKS = frozenset(
    {
        "unit_mvin_scale",
        "nonzero_dram",
        "pool_off",
        "transfer_rows",
        "movement_bytes",
        "nonnegative_stride",
        "weight_stationary",
        "full_i32",
        "raw_i32_exit",
        "max_columns",
        "pool_window",
        "mvout_scale",
        "mvout_relu",
        "mvout_pool_stride",
        "mvout_pool_size",
        "primitive_k_bound",
    }
)


def validate_source_conditions(condition):
    """Source admission cannot depend on an emitted-command observation."""
    validate_conditions(condition)
    if "path" in condition:
        if condition["path"].split(".")[0] == "execution":
            raise ValueError("execution conditions belong in emission_obligations")
        return
    for child in condition.get("all", condition.get("any", [])):
        validate_source_conditions(child)


def validate_emission_obligations(obligations):
    """Keep complete conditional programs, including alternatives, as data."""
    if not isinstance(obligations, dict) or not obligations:
        raise ValueError("emission_obligations must be a nonempty mapping")
    for name, obligation in obligations.items():
        if not isinstance(name, str) or not name or not isinstance(obligation, dict):
            raise ValueError("emission_obligations require named mappings")
        status = obligation.get("status")
        payload = set(obligation) & {"condition", "constraints"}
        allowed = {"status", *payload}
        if status == "instruction_audit":
            allowed.add("check")
            if obligation.get("check") not in INSTRUCTION_CHECKS or payload != {"constraints"}:
                raise ValueError("emission_obligations require a registered instruction audit check")
        elif status != "compiler_obligation":
            raise ValueError("emission_obligations require an instruction_audit or compiler_obligation status")
        if len(payload) != 1 or set(obligation) != allowed:
            raise ValueError("emission_obligations require exactly one condition or constraints payload")
        if "condition" in payload:
            validate_conditions(obligation["condition"])
        else:
            # Retain R2's scalar predicate spelling without a second evaluator.
            constraints = obligation["constraints"]
            if not isinstance(constraints, dict):
                raise ValueError("emission_obligations constraints must be a mapping")
            predicates = {("in" if k == "one_of" else k): v for k, v in constraints.items()}
            try:
                validate_conditions({"path": name, **predicates})
            except ValueError as error:
                raise ValueError(f"emission_obligations: {error}") from error


def obligation_condition(name, obligation):
    if "condition" in obligation:
        return obligation["condition"]
    return {"path": name, **{("in" if k == "one_of" else k): v for k, v in obligation["constraints"].items()}}


def screen_emission_obligations(obligations, observations):
    """Provider/host receipts may discharge duties; absence never blocks source eligibility.

    A compiler obligation stays recorded even when a diagnostic receipt matches it.
    This screen is deliberately separate from admit_operation.
    """
    validate_emission_obligations(obligations)
    return {
        name: {
            **obligation,
            "observation_status": screen_conditions(obligation_condition(name, obligation), observations)[0],
        }
        for name, obligation in obligations.items()
    }


#: Contracts keep the two distinct source rounding boundaries separate.
OPERAND_SUM_CONTRACT_ROUNDING = {
    "operand_sum_exhaustive_i8_v1": "single_rounding",
    "operand_sum_per_operand_exhaustive_i8_v1": "per_operand",
}


def _contraction_op(op):
    operator = op.removeprefix("aten.").partition(".")[0]
    return operator in {"mm", "bmm", "_int_mm", "addmm"} or "matmul" in op or "contraction" in op or "conv" in op


def binding_declaration(binding):
    """Select the source-bound declaration identically at admission and audit."""
    identity = binding.get("software_declaration")
    if identity is not None:
        return identity
    numeric = binding.get("numeric", {})
    family, form = numeric.get("family"), numeric.get("form")
    op = binding.get("op", "")
    if family == "F1":
        return "movement_bitwise"
    if family in {"F2", "F4", "F5"}:
        if form == "bilinear" or (form is None and "bilinear" in op):
            return "component_bilinear"
        if form == "linear" or (form is None and "linear" in op):
            return "component_linear"
        raise ValueError("UNKNOWN: component binding lacks a resolved numeric form")
    if _contraction_op(op):
        # The source op decides first: a numeric flag never moves a contraction
        # off its own row's duties.
        return "contraction"
    if numeric.get("scaled_operand_sum") is True:
        if numeric.get("operand_sum_rounding") == "per_operand":
            return "elementwise_map_after_per_operand_sum"
        return "elementwise_map_after_operand_sum"
    if "pool" in op:
        return "pooling_on_readout"
    if "movement" in op:
        return "movement"
    if "elementwise" in op:
        return "elementwise_map_standalone"
    raise ValueError("UNKNOWN: source binding lacks a software declaration")


def declaration_numerics(binding, identity, row):
    """Refuse a provider numeric route that the selected row's contract does not screen."""
    numeric = binding.get("numeric", {})
    if numeric.get("scaled_operand_sum") is not True:
        return
    contract = row.get("numerical_contract")
    rounding = OPERAND_SUM_CONTRACT_ROUNDING.get(contract) if isinstance(contract, str) else None
    if rounding is None or numeric.get("operand_sum_rounding") != rounding:
        raise ValueError(
            f"UNKNOWN: scaled operand-sum rounding {numeric.get('operand_sum_rounding')!r} is not screened by "
            f"declaration {identity} contract {contract!r}"
        )

    from merlin.targetgen.operator_review import review_accepted

    if not review_accepted(row):
        raise ValueError(f"UNKNOWN: declaration {identity} requires operator review acceptance")


def enforce_provider_emission_obligations(proof, spec):
    """Check available provider certificate observations, retaining absent ones.

    Only certificate fields actually supplied by the provider are projected.
    A source signature is never used as an execution receipt.
    """
    if not isinstance(proof, dict) or proof.get("status") != "proven":
        return proof
    declarations = {row["id"]: row for row in spec.get("operations", [])}
    for kernel in proof.get("kernels", []):
        bindings = [b for b in proof.get("bindings", []) if b.get("symbol") == kernel.get("symbol")]
        reports = {}
        for binding in bindings:
            identity = binding_declaration(binding)
            if identity not in declarations:
                raise ValueError(f"UNKNOWN: no selected software declaration {identity}")
            declaration_numerics(binding, identity, declarations[identity])
            obligations = declarations[identity].get("emission_obligations")
            if not obligations:
                continue
            execution = dict(kernel.get("execution", {}))
            bound = kernel.get("schedule", {}).get("bound_proof", {})
            if "max_mesh_partial_sum" in bound:
                execution["pe_partial_abs_bound"] = bound["max_mesh_partial_sum"]
            # A signed-interval (OS value-bound) proof states each column's lowest and highest
            # partial sum; project its extremes, never an absolute value of them.
            lower, upper = bound.get("column_lower"), bound.get("column_upper")
            if (
                isinstance(lower, list)
                and isinstance(upper, list)
                and lower
                and len(lower) == len(upper)
                and all(type(v) is int for v in (*lower, *upper))
            ):
                execution["pe_partial_min"] = min(lower)
                execution["pe_partial_max"] = max(upper)
            if type(bound.get("accumulator_min")) is int and type(bound.get("accumulator_max")) is int:
                execution["accumulator_partial_abs_bound"] = max(
                    abs(bound["accumulator_min"]), abs(bound["accumulator_max"])
                )
            observations = {"execution": execution}
            report = screen_emission_obligations(obligations, observations)
            for name, result in report.items():
                # Scalar audit duties have their separate retired-stream owner.
                # Compiler conditions can still contradict an available receipt.
                if result["status"] == "compiler_obligation" and result["observation_status"] == "refused":
                    raise ValueError(f"provider emission obligation {identity}.{name} violated")
            reports[identity] = report
        kernel["emission_obligations"] = reports
    return proof


def validate_conditions(condition):
    if not isinstance(condition, dict) or not condition:
        raise ValueError("admission conditions require a nonempty mapping")
    for connective in ("all", "any"):
        if connective in condition:
            children = condition[connective]
            if set(condition) != {connective} or not isinstance(children, list) or not children:
                raise ValueError("admission condition connective requires a nonempty list")
            for child in children:
                validate_conditions(child)
            return
    operators = set(condition) - {"path"}
    path = condition.get("path")
    if (
        not isinstance(path, str)
        or not all(path.split("."))
        or not operators
        or operators - {"eq", "in", "min", "max", "gt", "finite", "integer"}
    ):
        raise ValueError("admission condition requires a path and supported comparisons")
    if "in" in condition and (not isinstance(condition["in"], list) or not condition["in"]):
        raise ValueError("admission condition in requires nonempty choices")
    for value in ([condition["eq"]] if "eq" in condition else []) + condition.get("in", []):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("admission condition values must be finite")
    for key in operators & {"min", "max", "gt"}:
        if not _finite_number(condition[key]):
            raise ValueError("admission condition bounds must be finite numbers")
    if "finite" in condition and condition["finite"] is not True:
        raise ValueError("admission condition finite must be true")
    if "integer" in condition and condition["integer"] is not True:
        raise ValueError("admission condition integer must be true")


def _finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _equal(actual, expected):
    # Python's True == 1 must not turn a boolean assertion into a numerical bound.
    return type(actual) is type(expected) and actual == expected


def _matches(actual, operator, expected):
    if operator == "eq":
        return _equal(actual, expected)
    if operator == "in":
        return any(_equal(actual, choice) for choice in expected)
    if operator == "integer":
        return type(actual) is int
    return _finite_number(actual) and (
        operator == "finite"
        or (operator == "min" and actual >= expected)
        or (operator == "max" and actual <= expected)
        or (operator == "gt" and actual > expected)
    )


def _screen_leaf(condition, observations):
    path = condition["path"]
    actual = observations
    for member in path.split("."):
        if not isinstance(actual, Mapping) or member not in actual or actual[member] is None:
            return "unknown", [path]
        actual = actual[member]
    for operator, expected in condition.items():
        if operator == "path":
            continue
        if not _matches(actual, operator, expected):
            return "refused", [path]
    return "matched", []


def screen_conditions(condition, observations):
    """Return matched/unknown/refused plus the unresolved or violated semantic paths."""
    validate_conditions(condition)
    for connective in ("all", "any"):
        if connective not in condition:
            continue
        results = [screen_conditions(child, observations) for child in condition[connective]]
        states = [state for state, _ in results]
        priority = ("refused", "unknown", "matched") if connective == "all" else ("matched", "unknown", "refused")
        state = next(value for value in priority if value in states)
        return state, sorted({path for value, paths in results if value == state for path in paths})
    return _screen_leaf(condition, observations)
