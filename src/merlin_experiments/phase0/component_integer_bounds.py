"""Pure source interval proof for the selected integer DAG arithmetic contract.

No tensor stimulus or golden is allocated. Bounded exact arithmetic requires
every product, reduction prefix and node result to fit its selected width.
Intervals deliberately ignore cancellation/correlation and may refuse a safe
program. They establish source/declared numerical bounds, never hardware effects.
"""

from __future__ import annotations

import copy

from merlin.runtime.commandbuffer import STIMULUS_RANGE_KEY, stimulus_range
from merlin.targetgen.input_palette import pattern

from .component_execution_budget import source_for_capsule, source_for_entry
from .component_generation import digest

SCHEMA = "merlin.component_integer_bounds.v1"
_PARTIAL = "bounded_exact_requires_each_partial_sum"


def _bits(dtype):
    if not isinstance(dtype, str) or not dtype.startswith("i") or not dtype[1:].isdigit() or int(dtype[1:]) < 1:
        raise ValueError("component integer proof requires signed integer source types")
    return int(dtype[1:])


def _fits(interval, bits):
    """Avoid allocating a huge 1<<width merely to reject or check a format."""
    low, high = interval
    return (low >= 0 or (-low - 1).bit_length() < bits) and (high < 0 or high.bit_length() < bits)


def _require(interval, bits, *, owner, stage):
    if not _fits(interval, bits):
        raise ValueError(f"component bounded_exact {owner}: {stage} may overflow signed i{bits}")


def _contract(semantics):
    if not isinstance(semantics, dict) or not isinstance(semantics.get("model"), dict):
        raise ValueError("component DAG has no selected independent integer arithmetic contract")
    if (semantics.get("model") or {}).get("engine") != "integer_reference":
        raise ValueError("component DAG has no selected independent integer arithmetic contract")
    overflow = semantics.get("overflow")
    internal = semantics.get("internal_arithmetic") or {}
    if not isinstance(internal, dict):
        raise ValueError("component DAG has an unsupported declared partial-sum contract")
    partial = internal.get("full_operation_overflow_policy")
    if partial not in {None, _PARTIAL}:
        raise ValueError("component DAG has an unsupported declared partial-sum contract")
    if partial == _PARTIAL:
        for key in ("signed_operand_bits", "mac_result_bits"):
            if type(internal.get(key)) is not int or internal[key] < 2:
                raise ValueError("component bounded_exact partial-sum declaration has unknown signed widths")
        if overflow not in {"bounded_exact", "modular_wrap", "wrap_internal_mac"}:
            raise ValueError("component DAG has no compatible selected integer overflow contract")
        return "bounded_exact", internal
    if overflow not in {"bounded_exact", "modular_wrap"}:
        raise ValueError("component DAG requires explicit bounded_exact or modular_wrap arithmetic")
    return overflow, {}


def derive(source, semantics):
    """Prove selected source bounds using the actual canonical stimulus domains."""
    mode, internal = _contract(semantics)
    program = source["program"]
    if source["kind"] != "component_program":
        raise ValueError("component integer interval proof applies only to the checked DAG source")
    default = stimulus_range({"params": {STIMULUS_RANGE_KEY: source.get("stimulus_range")}})
    intervals, inputs, nodes = {}, [], []
    types = {row["name"]: row for row in program["inputs"] + program["nodes"]}
    for index, row in enumerate(program["inputs"]):
        selected = None
        if source.get("input_palette") is not None:
            try:
                selected = pattern(source["input_palette"], name=row["name"], dtype=row["dtype"], index=index)
            except (KeyError, OverflowError) as exc:
                raise ValueError("component integer input domain has an unsupported scalar format") from exc
        values = selected["values"] if selected is not None else default
        if any(type(value) is not int for value in values):
            raise ValueError("component integer interval proof requires an exact integral input domain")
        interval = min(values), max(values)
        _require(interval, _bits(row["dtype"]), owner=row["name"], stage="input domain")
        intervals[row["name"]] = interval
        inputs.append({"name": row["name"], "dtype": row["dtype"], "interval": list(interval)})
    if mode == "bounded_exact":
        for node in program["nodes"]:
            args = [intervals[name] for name in node["actual_inputs"]]
            bits, op = _bits(node["dtype"]), node["op"]
            proof = {"name": node["name"], "op": op, "result_bits": bits}
            if op == "matmul":
                if internal:
                    for arg in args:
                        _require(
                            arg, internal["signed_operand_bits"], owner=node["name"], stage="declared MAC operands"
                        )
                products = [lhs * rhs for lhs in args[0] for rhs in args[1]]
                product = min(products), max(products)
                lhs = types[node["actual_inputs"][0]]
                reduction = lhs["shape"][1]
                interval = product[0] * reduction, product[1] * reduction
                prefix = min(0, interval[0]), max(0, interval[1])
                widths = sorted({bits, internal["mac_result_bits"]} if internal else {bits})
                for width in widths:
                    _require(product, width, owner=node["name"], stage="product")
                    _require(prefix, width, owner=node["name"], stage="partial sum")
                proof.update(product_interval=list(product), partial_sum_interval=list(prefix), checked_mac_bits=widths)
            elif op in {"add", "update"}:
                interval = args[0][0] + args[1][0], args[0][1] + args[1][1]
            elif op == "reduce_sum":
                lhs = types[node["actual_inputs"][0]]
                reduction = lhs["shape"][node["axis"]]
                interval = args[0][0] * reduction, args[0][1] * reduction
                prefix = min(0, interval[0]), max(0, interval[1])
                _require(prefix, bits, owner=node["name"], stage="partial sum")
                proof.update(partial_sum_interval=list(prefix), reduction_axis=node["axis"], reduction_extent=reduction)
            elif op == "concat":
                interval = min(arg[0] for arg in args), max(arg[1] for arg in args)
            elif op in {"copy", "transpose", "reshape", "alias", "cast", "extract_slice"}:
                interval = args[0]
            else:
                raise ValueError("component integer interval proof has no semantics for the source node")
            _require(interval, bits, owner=node["name"], stage="node result")
            intervals[node["name"]] = interval
            nodes.append({**proof, "interval": list(interval)})
    return {
        "schema": SCHEMA,
        "status": "proven_safe" if mode == "bounded_exact" else "explicit_modular_wrap",
        "contract": mode,
        "source_sha256": digest(source),
        "numerical_semantics_sha256": digest(semantics),
        "input_intervals": inputs,
        "nodes": nodes,
        "hardware_arithmetic_status": "unverified",
        "scope": "all checked source node results/products/reduction prefixes; no hardware execution or physical proof",
    }


def preflight_entry(entry, *, binding):
    return derive(source_for_entry(entry, binding=binding), entry.get("numerical_semantics") or {})


def preflight_capsule(capsule):
    proof = derive(source_for_capsule(capsule), capsule.get("numerical_semantics") or {})
    if digest(capsule.get("integer_partial_sum_bound")) != digest(proof):
        raise ValueError("component integer reference source bound differs from its actual selected declaration")
    return proof


def verify_selected_capsule(capsule, *, semantics_sha256, require_bound):
    """Bind the replayed arithmetic choice to the protected selected SW policy."""
    if (capsule.get("operation") or {}).get("op") != "component_program":
        return
    proof = capsule.get("integer_partial_sum_bound") or {}
    if proof.get("schema") != SCHEMA:
        if require_bound:
            raise ValueError("component DAG lacks the independently derived integer source bound")
        return  # historical v1 generation, without a new numerical-bound claim
    semantics = copy.deepcopy(capsule.get("numerical_semantics") or {})
    if isinstance(semantics.get("model"), dict):
        semantics["model"].pop("source_bundle_sha256", None)
    if digest(semantics) != semantics_sha256:
        raise ValueError("component DAG arithmetic choice differs from protected selected software semantics")
    preflight_capsule(capsule)
