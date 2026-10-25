"""Composition truth from declared family semantics, schemes and source shapes.

Inputs are frozen operator-side declarations, never candidate plans or sampled
outputs. This module states placement and decoded-instruction obligations; it
does not execute an audit or qualify a target. Per-operation truth is unchanged.
"""

from __future__ import annotations

from fractions import Fraction

from merlin.targetgen.output_path import validate as validate_path
from merlin.targetgen.quant_methodology import scheme_geometry
from merlin.targetgen.software_spec import readout_exactness_gate, validate_compositions


def _constraints(chain):
    return {
        item if isinstance(item, str) else next(iter(item)): True
        if isinstance(item, str)
        else next(iter(item.values()))
        for item in chain["constraints"]
    }


def _readout_refusals(chain, scheme, case, path):
    stages = case["readout_stages"]
    order = [stage["kind"] for stage in path["readout"]]
    if case["exit"] not in chain.get("exits", ()) or path["exits"].get(case["exit"]) is not True:
        yield "exit is not declared and derived"
    if case["exit"] == "accumulator" and stages:
        yield "accumulator exit applies no readout stage"
    if len(set(stages)) != len(stages) or stages != [s for s in order if s in stages]:
        yield "readout stages violate the derived path order"
    declared = chain.get("readout_stages", [])
    if declared != "from_output_path" and not set(stages) <= set(declared):
        yield "readout stage is outside the declared chain"
    if case["exit"] == "requant" and "requant" not in stages:
        yield "narrow exit requires requantization"
    if "scale" in stages and "convert" not in stages:
        yield "scale requires the accumulator conversion"
    if "activation" in stages and any(
        stage.get("function") != "relu" for stage in path["readout"] if stage["kind"] == "activation"
    ):
        yield "activation instruction obligation is UNKNOWN"
    geometry = scheme_geometry(path, scheme["weight_scale_granularity"])
    if geometry.get("unknown") or any(v == "UNKNOWN" for v in geometry.values()):
        yield "scale geometry is not derived"
    if scheme.get("columns_per_scale", geometry.get("columns_per_scale")) != geometry.get("columns_per_scale"):
        yield "scheme width disagrees with the output path"
    if scheme["weight_scale_granularity"] == "column_block" and case["exit"] == "requant":
        columns, width = case.get("output_columns"), geometry.get("columns_per_scale")
        if type(columns) is not int or columns <= 0 or type(width) is not int:
            yield "block readout requires source columns and derived block width"
        elif len(case.get("multipliers", [])) != (columns + width - 1) // width:
            yield "block readout requires one multiplier per block, including the tail"


def _accumulator_refusals(chain, case, path):
    stages = case.get("accumulator_stages", [])
    declared = chain.get("accumulator_stages", {})
    available = {s["kind"] for s in path.get("accumulator_stages", [])}
    if not set(stages) <= set(declared) & available:
        yield "accumulator stage is not declared and derived"
    if any(mode == "required" and stage not in stages for stage, mode in declared.items()):
        yield "a required accumulator stage is absent"


def _scale_refusals(scheme, case):
    from merlin.runtime.exact_readout import prove_fp32_products

    strategy = scheme["readout_strategy"]
    if scheme["scale_value"] == "arbitrary_f32" and strategy != "f32_product_rne":
        yield "arbitrary scales require the f32 product definition"
    if case["exit"] == "accumulator":
        return
    scales = case.get("multipliers", [])
    if not scales:
        yield "narrow readout requires formed multipliers"
    if "activation" in case["readout_stages"] and any(Fraction(s) < 0 for s in scales):
        yield "activation before scale requires nonnegative multipliers"
    if strategy == "exact_rational_host":
        bounds = case.get("accumulator_range")
        if bounds is None:
            yield "exact rational readout lacks an f32 equality proof"
            return
        try:
            for scale in scales:
                prove_fp32_products(Fraction(scale), *bounds)
        except ValueError:
            yield "exact rational readout lacks an f32 equality proof"


def _pool_refusals(constraints, case, path, facts):
    if "pool" not in case["readout_stages"] and case.get("reduction_kind") != "max":
        return
    pools = [s for s in path["readout"] if s["kind"] == "pool"]
    if len(pools) != 1 or "max" not in pools[0].get("reductions", []):
        yield "store-path max pooling is not derived"
        return
    if case.get("indices_live") is not False:
        yield "max-pool indices must be dead"
    if "max_requires_store_path_pooling" in constraints and facts.get("max_requires_store_path_pooling") is not True:
        yield "movement-composed store pooling is not derived"
    if (
        "scratchpad_source_pooling_fenced" in constraints
        and (case["producer"] == "movement" or case.get("pool_source") == "scratchpad")
        and case.get("scratchpad_source_pooling_fenced") is not True
    ):
        yield "scratchpad-source pooling lacks its required fence"
    padding = case.get("padding")
    if not isinstance(padding, list) or not padding or any(type(value) is not int or value < 0 for value in padding):
        yield "pool padding is UNKNOWN"
    elif any(padding):
        pad = pools[0].get("padding")
        if isinstance(pad, str) and pad.lower() == "zero":
            pad = 0
        if type(pad) not in (int, float):
            yield "pool padding is UNKNOWN"
        elif not _padding_neutral(case, path, pad):
            yield "pool padding lacks a source-derived neutrality proof"
    limits = facts.get("pool_window_within_register_fields", {})
    for field, bits_key in (("pool_window", "size_bits"), ("pool_stride", "stride_bits")):
        bits, values = limits.get(bits_key), case.get(field)
        if (
            type(bits) is not int
            or bits <= 0
            or not values
            or any(type(v) is not int or not 0 < v < 1 << bits for v in values)
        ):
            yield f"{field} exceeds or lacks derived register bounds"


def _padding_neutral(case, path, pad):
    """Prove neutrality in the pool's input domain, after the selected readout."""
    import math

    from merlin.runtime.exact_readout import form_multiplier

    if not math.isfinite(pad):
        return False
    bound = case.get("producer_lower_bound")
    try:
        bound = Fraction(bound) if bound is not None else None
        for stage in path["readout"]:
            kind = stage["kind"]
            if kind == "pool":
                break
            if kind not in case["readout_stages"]:
                continue
            if kind in {"convert", "scale", "requant"} and stage.get("rounding") != "half_even":
                return False
            if kind == "activation" and stage.get("function") == "relu":
                bound = max(0, bound) if bound is not None else Fraction(0)
            elif kind == "convert" and bound is not None:
                bound = Fraction(float(form_multiplier(bound)))
            elif kind == "scale":
                scales = [Fraction(float(form_multiplier(scale))) for scale in case.get("multipliers", [])]
                bound = (
                    min(Fraction(float(form_multiplier(bound * scale))) for scale in scales)
                    if bound is not None and scales and min(scales) >= 0
                    else None
                )
            elif kind == "requant":
                clamp = stage.get("clamp")
                if not isinstance(clamp, list) or len(clamp) != 2:
                    return False
                bound = max(clamp[0], min(clamp[1], round(bound))) if bound is not None else clamp[0]
        return bound is not None and bound >= pad
    except (ValueError, TypeError, OverflowError):
        return False


def _sum_refusals(scheme, case):
    scales = case.get("operand_multipliers", [])
    if len(scales) != 2 or any(not 0 < Fraction(s) <= 1 for s in scales):
        yield "operand multipliers must lie in (0, 1]"
    if case.get("negates_load_minimum") is not False:
        yield "load minimum negation is unproved"
    if scheme["operand_sum_rounding"] == "single_rounding" and any(Fraction(s) != 1 for s in scales):
        yield "single-rounding operand sum needs an exhaustive exact proof"
    if scheme["operand_sum_rounding"] == "per_operand" and case.get("operand_requantization") != "round_saturate_each":
        yield "per-operand scheme requires independent round and saturation boundaries"


def _os_refusals(case, facts):
    weights, bits, operand_bits = (
        case.get("constant_weights"),
        facts.get("mesh_signed_bits"),
        facts.get("operand_signed_bits"),
    )
    if not weights or type(bits) is not int or type(operand_bits) is not int or min(bits, operand_bits) < 2:
        yield "OS requires literal weights and derived signed widths"
        return
    low, high = -(1 << (operand_bits - 1)), (1 << (operand_bits - 1)) - 1
    columns = len(weights[0])
    if not columns or any(
        len(row) != columns or any(type(w) is not int or not low <= w <= high for w in row) for row in weights
    ):
        yield "OS weights are outside the declared operand domain"
        return
    seed = case.get("seed_range")
    if seed is None:
        yield "OS requires a source initializer range"
        return
    for column in zip(*weights, strict=True):
        lower = seed[0] + sum(min(low * w, high * w) for w in column)
        upper = seed[1] + sum(max(low * w, high * w) for w in column)
        if lower < -(1 << (bits - 1)) or upper >= 1 << (bits - 1):
            yield "OS constant-weight partial sum exceeds the mesh bound"
            return


def _extension_refusals(case, facts):
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

    limits = facts.get("k_extension_within", {})
    try:
        lower_k_extension(
            case.get("coefficient"),
            source_k=case.get("source_k"),
            residual_width=case.get("residual_width"),
            digit_bits=facts.get("operand_signed_bits"),
            max_extended_k=limits.get("max_k"),
            accumulator_bits=limits.get("partial_sum_bits"),
            source_bound=case.get("source_partial_bound"),
            residual_range=case.get("residual_range", (None, None)),
        )
    except (ValueError, TypeError) as error:
        yield str(error)


def _shared_refusals(chain, case, facts):
    siblings = case.get("siblings", [])
    if len(siblings) < chain["multiplicity"]["min"] or case.get("immutable_shared_input") is not True:
        yield "shared-input fusion requires independent siblings and one immutable input"
    exits = set(chain.get("exits", ()))
    if any(s.get("exit") not in exits or not s.get("readout_independent") for s in siblings):
        yield "each sibling requires its independently declared readout"
    capacity = facts.get("shared_input_resident", {}).get("capacity_bytes")
    required = case.get("resident_bytes")
    if type(capacity) is not int or type(required) is not int or not 0 < required <= capacity:
        yield "shared input exceeds or lacks derived residency capacity"


def _family_refusals(chain, scheme, case, path, facts):
    constraints = _constraints(chain)
    if case.get("reduction_kind") not in [None, *chain.get("reduction_kinds", [])]:
        yield "reduction kind is outside the declared chain"
    yield from _pool_refusals(constraints, case, path, facts)
    if case["producer"] == "operand_sum":
        yield from _sum_refusals(scheme, case)
    if case.get("dataflow") == "OS":
        yield from _os_refusals(case, facts)
    if "dataflows" in chain and case.get("dataflow") not in facts.get("dataflows", []):
        yield "dataflow is not derived"
    if chain.get("extension"):
        yield from _extension_refusals(case, facts)
    if chain.get("multiplicity"):
        yield from _shared_refusals(chain, case, facts)


def _audit_modes(chain, case, path):
    modes = []
    if case["exit"] == "accumulator":
        modes.append("i32_exit")
    if "scale" in case["readout_stages"]:
        modes.append("readout_multipliers")
    if "activation" in case["readout_stages"]:
        modes.extend(stage.get("function", "UNKNOWN") for stage in path["readout"] if stage["kind"] == "activation")
    if "preload" in case.get("accumulator_stages", []):
        modes.append("accumulator_preload")
    if "accumulate" in case.get("accumulator_stages", []):
        modes.append("scaled_accumulate_load")
    if "pool" in case["readout_stages"] or case.get("reduction_kind") == "max":
        modes.append("pooled_store")
        if "scratchpad_source_pooling_fenced" in _constraints(chain) and (
            case["producer"] == "movement" or case.get("pool_source") == "scratchpad"
        ):
            modes.append("scratchpad_source_pooling_fenced")
    if case["producer"] == "operand_sum":
        modes.append("scaled_accumulate_load")
    if case.get("dataflow") == "OS":
        modes.append("output_stationary")
    if chain.get("extension"):
        modes.append("k_extension")
    if chain.get("multiplicity"):
        modes.append("shared_input")
    return list(dict.fromkeys(modes))


def _audit_parameters(scheme, case):
    import struct

    from merlin.runtime.exact_readout import form_multiplier

    def bits(values):
        return [struct.unpack("<I", struct.pack("<f", form_multiplier(value)))[0] for value in values]

    return dict(
        readout_multiplier_bits=bits(case.get("multipliers", [])),
        operand_multiplier_bits=bits(case.get("operand_multipliers", [])),
        pool_window=case.get("pool_window"),
        pool_stride=case.get("pool_stride"),
        dataflow=case.get("dataflow"),
        weight_layout=scheme.get("weight_layout"),
        siblings=case.get("siblings", []),
    )


def expectation(chain, scheme, case, *, facts):
    """Commit a lane for one family-chain × scheme × source shape instance.

    ``case`` holds source-semantic parameters and member identities. ``facts``
    holds the independently resolved output path and resource/width facts.
    Unknown applicable constraints refuse; unrelated optional stages do not.
    """
    validate_compositions(
        dict(
            schema="merlin.software_spec.compositions.v1",
            status="reviewed",
            grading="placement_and_instruction_audit",
            chains=[chain],
        )
    )
    path = validate_path(facts["output_path"])
    producers = [chain["producer"]] if isinstance(chain["producer"], str) else chain["producer"]
    refusals = []
    if case["producer"] not in producers:
        refusals.append("producer is outside the declared family chain")
    placement = facts.get("producer_placement", {}).get(case["producer"], [])
    if not set(placement) & {"accelerator", "fused_accelerator"}:
        refusals.append("accelerator producer placement is not derived")
    if scheme.get("status") != "resolved" or scheme.get("unknown") or scheme.get("graded") is not True:
        refusals.append("scheme is unresolved or outside graded block granularity")
    refusals.extend(_readout_refusals(chain, scheme, case, path))
    refusals.extend(_accumulator_refusals(chain, case, path))
    refusals.extend(_scale_refusals(scheme, case))
    refusals.extend(_family_refusals(chain, scheme, case, path, facts))
    members = case.get("members", [])
    if not members or len(set(members)) != len(members) or not case.get("shape_class"):
        raise ValueError("composition truth requires unique source members and a shape class")
    return dict(
        schema="merlin.functional_truth.compositions.v1",
        chain_id=chain["id"],
        scheme_id=scheme["id"],
        shape_class=case["shape_class"],
        lane="must_refuse" if refusals else "fused_device",
        outcome="refuse" if refusals else "pass",
        reasons=refusals,
        members=list(members),
        grading="placement_and_instruction_audit",
        required_modes=_audit_modes(chain, case, path),
        instruction_expectation=_audit_parameters(scheme, case),
        oracle_requirement=scheme.get("oracle_requirement"),
        exactness_gate=readout_exactness_gate(scheme["readout_strategy"]),
    )


def require_placement(commitment, *, member_kernels, instruction_audit):
    """Require all members in one kernel and all independent instruction audits.

    The caller supplies trusted frozen-placement and instruction-audit results.
    A numerical match, a compiler plan or a placement count supplies no credit.
    """
    if commitment["lane"] == "must_refuse":
        if member_kernels or instruction_audit.get("outcome") != "refuse":
            raise ValueError("composition truth requires refusal")
        return
    if commitment["lane"] != "fused_device":
        raise ValueError("unknown composition truth lane")
    if (
        set(member_kernels) != set(commitment["members"])
        or not all(member_kernels.values())
        or len(set(member_kernels.values())) != 1
    ):
        raise ValueError("every composition member must occupy the same accelerator kernel")
    if instruction_audit.get("kernel") != next(iter(member_kernels.values())):
        raise ValueError("instruction audit must bind the routed composition kernel")
    modes = instruction_audit.get("modes", {})
    if any(modes.get(mode) is not True for mode in commitment["required_modes"]):
        raise ValueError("composition instruction audit is missing or failed")
    counts = [instruction_audit.get(key) for key in ("outside_kernel_instructions", "host_intermediate_arithmetic")]
    if any(type(count) is not int or count != 0 for count in counts):
        raise ValueError("composition arithmetic escaped the routed kernel")
