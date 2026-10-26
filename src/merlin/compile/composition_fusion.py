"""Source-derived composition plans for target providers; no target emission.

Declarations select semantic families, never their arbitrary ids. Returned plans
retain source ordinals and numeric obligations. A provider must consume them in
its ordinary emitter and independently check actual effects and multiplier bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from merlin.transforms.certificates import ir_digest


@dataclass(frozen=True)
class CompositionPlan:
    route: str
    source_ir_sha256: str
    operation_ordinals: tuple[int, ...]
    decision: dict

    def record(self):
        return {
            "schema": "merlin.composition_plan.v1",
            "route": self.route,
            "source_ir_sha256": self.source_ir_sha256,
            "operation_ordinals": list(self.operation_ordinals),
            "decision": self.decision,
            "emission_obligation": "consume plan and check actual effects and decoded multiplier bits",
            "execution_qualification": "not_established",
        }


def plan_contraction_readout(module, root, *, path, scheme, facts, runtime_scale_facts=None):
    """Plan one source-bound readout for an ordinary provider catalog.

    Providers may request this without running fusion exploration. The same
    semantic proof and multiplier formation serve both entry points; emission
    remains an obligation of the consuming provider.
    """
    from merlin.targetgen.output_path import validate
    from merlin.verify.lean_contract import document_digest
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    path = validate(path)
    if scheme.get("readout_strategy") not in {"f32_product_rne", "exact_rational_host"}:
        raise ValueError("readout planning requires an explicit numerical strategy")
    chain = OF.source_chain(root, OF.Values())
    route = "epilogue" if chain.stages or chain.ops else "int32_readout"
    members = (root, *chain.ops)
    _closed_members(members)
    decision = _readout_plan(module, root, route, path, scheme, runtime_scale_facts=runtime_scale_facts)
    operations = tuple(module.walk())
    return CompositionPlan(
        route,
        ir_digest(module),
        tuple(operations.index(op) for op in members),
        {
            **decision,
            "scheme_sha256": document_digest(scheme),
            "facts_sha256": document_digest(facts),
            "output_path_sha256": document_digest(path),
        },
    )


def composition_routes(declaration):
    """Register routes by declared semantics, with no meaning assigned to ids."""
    producer = declaration["producer"]
    producers = {producer} if isinstance(producer, str) else set(producer)
    if declaration.get("multiplicity"):
        return ("shared_input",)
    if declaration.get("extension"):
        return ("k_concat",)
    if declaration.get("dataflows"):
        return ("output_stationary",)
    if "operand_sum" in producers:
        return ("residual_on_load",)
    if "sum" in declaration.get("reduction_kinds", ()):
        return ("avgpool",)
    if "max" in declaration.get("reduction_kinds", ()):
        return ("conv_pool",)
    exits = declaration.get("exits", ())
    return tuple(route for exit, route in (("requant", "epilogue"), ("accumulator", "int32_readout")) if exit in exits)


def _region_to_quantize(root):
    """A bounded single-use chain, including its dequantized input operands."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    members = []
    for operand in root.operands:
        owner = operand.owner
        stage = CG.classify(owner) if hasattr(owner, "name") else None
        if stage is not None and stage.kind == CG.DEQUANTIZE:
            members.append(owner)
    members.append(root)
    value = root.results[0]
    for _ in range(16):
        users = list(value.uses)
        if len(users) != 1:
            break
        op = users[0].operation
        stage = CG.classify(op)
        if stage is None or stage.kind not in (CG.SCALE, CG.RELU, CG.QUANTIZE):
            break
        members.append(op)
        if stage.kind == CG.QUANTIZE:
            break
        value = op.results[0]
    return tuple(members)


def composition_proposals(module, block, *, max_region_operations):
    """Yield resumable source opportunities, leaving all admission to realization."""
    from merlin.targetgen.software_spec import validate_compositions
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    declarations = validate_compositions(block)["chains"]
    operations = tuple(module.walk())
    values = OF.Values()
    for declaration in declarations:
        producer = declaration["producer"]
        producers = {producer} if isinstance(producer, str) else set(producer)
        for route in composition_routes(declaration):
            if route == "shared_input":
                from merlin.compile.scheduling.shared_input_groups import (
                    find_shared_input_groups,
                    grow_shared_input_group,
                )

                try:
                    groups = find_shared_input_groups(module)
                except ValueError:
                    groups = ()
                from xdsl.dialects import func

                function = next((op for op in module.walk() if isinstance(op, func.FuncOp)), None)
                if function is None:
                    continue
                body = tuple(function.body.block.ops)
                for group in groups:
                    for prefix in grow_shared_input_group(group):
                        if len(prefix.widths) >= declaration["multiplicity"]["min"]:
                            yield (
                                route,
                                tuple(body[i] for i in prefix.source_op_indices),
                                {**prefix.record(), "composition": declaration},
                                None,
                            )
                continue
            for op in operations:
                stage = CG.classify(op)
                if stage is None:
                    continue
                if (
                    stage.kind == CG.CONTRACTION
                    and "contraction" in producers
                    and route not in {"residual_on_load", "avgpool"}
                ):
                    chain = OF.source_chain(op, values)
                    members = (op,) if route in {"output_stationary", "int32_readout"} else (op, *chain.ops)
                elif (
                    route == "residual_on_load"
                    and stage.kind == CG.RESIDUAL_ADD
                    or route == "avgpool"
                    and op.name == "linalg.reduce"
                ):
                    members = _region_to_quantize(op)
                else:
                    continue
                if len(members) <= max_region_operations:
                    yield route, members, {"composition": declaration, "root_ordinal": operations.index(op)}, None


def _closed_members(members):
    selected = set(members)
    for op in members[:-1]:
        if any(use.operation not in selected for value in op.results for use in value.uses):
            raise ValueError("composition intermediate has an external observer")


def _readout_plan(module, root, route, path, scheme, *, runtime_scale_facts=None):
    from merlin.targetgen import model_coverage
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    values = OF.Values()
    chain = OF.source_chain(root, values)
    result = OF.decide(
        chain,
        path,
        values,
        model_coverage._contraction_extents(module),
        readout_strategy=scheme.get("readout_strategy"),
        # Transport is authenticated separately; keep facts_sha256 bound to raw facts.
        runtime_scale_facts=runtime_scale_facts,
    )
    expected = "accumulator" if route == "int32_readout" else "pool" if route == "conv_pool" else "requant"
    if result.cut != expected:
        raise ValueError(f"composition requires {expected} cut; got {result.cut}: {result.refusals}")
    if route == "int32_readout" and (chain.stages or chain.ops):
        raise ValueError("int32 exit must apply no readout stage")
    return result.record()


def _k_concat(module, root, path, scheme, facts):
    from xdsl.ir import Operation

    from merlin.targetgen import model_coverage
    from merlin.targetgen.native_mac_bound import signed_matmul_shape
    from merlin.xdsl_dialects.lowering import output_fusion as OF
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

    _, k, n = signed_matmul_shape(root)
    values = OF.Values()
    chain = OF.source_chain(root, values)
    residual = next((s for s in chain.stages if s.kind == "residual"), None)
    if residual is None:
        raise ValueError("K-concat requires a residual in the source chain")
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    operand = residual.detail["operand"]
    dequantize = operand.owner
    name = mq.op_name(dequantize) if isinstance(dequantize, Operation) else ""
    if name.rpartition(".")[2] not in {"dequantize_per_tensor", "dequantize_per_channel"}:
        raise ValueError("K-concat requires a directly dequantized residual")
    if mq.type_shape_dtype(dequantize.operands[0].type)[1] != "i8" or (
        name.endswith("dequantize_per_tensor") and GN._scale_source(dequantize).zero_point != 0
    ):
        raise ValueError("K-concat requires a zero-centered signed i8 residual")
    if tuple(dequantize.operands[0].type.get_shape()) != tuple(root.results[0].type.get_shape()):
        raise ValueError("K-concat residual shape differs from contraction result")
    table, scale = OF._residual_values(chain, residual, values, OF._numpy_float(chain.float_dtype))
    coefficients = OF._residual_coefficients(chain, residual, scale)
    if len(coefficients) != 1:
        raise ValueError("K-concat needs one exact integer residual coefficient")
    bound = OF._bound(chain, values, model_coverage._contraction_extents(module))
    from merlin.targetgen.native_mac_bound import constant_tensor

    constant_tensor(root.operands[2])
    source_bound = max(abs(v) for v in bound)
    if type(facts.get("digit_bits")) is not int or not 2 <= facts["digit_bits"] <= 8:
        raise ValueError("K-extension digit width must fit the source signed i8 operands")
    accumulator_bits = facts.get("accumulator_bits")
    if type(accumulator_bits) is not int or accumulator_bits < 2:
        raise ValueError("K-extension needs a declared accumulator width")
    extension = lower_k_extension(
        coefficients.pop(),
        source_k=k,
        residual_width=n,
        digit_bits=facts.get("digit_bits"),
        max_extended_k=facts.get("max_extended_k"),
        accumulator_bits=min(accumulator_bits, 32),
        source_bound=source_bound,
        residual_range=(min(table), max(table)),
    )
    # Admit the complete remaining boundary with the extended residual terms.
    # This cannot silently leave a sequential floating readout after moving the
    # residual into the contraction.
    readout = OF.decide(
        chain,
        path,
        values,
        model_coverage._contraction_extents(module),
        readout_strategy=scheme.get("readout_strategy"),
        integer_residual_extension=True,
    )
    if readout.cut not in {"requant", "pool"}:
        raise ValueError(f"K-extension readout is not admitted: {readout.refusals}")
    if readout.cut == "pool":
        _pool_limits(readout.record(), facts)
    extension["readout"] = readout.record()
    extension["residual_stage_ordinal"] = tuple(module.walk()).index(residual.op)
    extension["replaces"] = "complete selected contraction and readout chain"
    return extension


def _scalar_readout(path, quantize, *, relu):
    """Bind a tensor-wide f32 boundary to the actual output storage contract."""
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    if not path.get("exits", {}).get("requant") or path.get("accumulator", {}).get("dtype") != "i32":
        raise ValueError("composition requires an i32 accumulator and requant exit")
    readout = [s for s in path.get("readout", ()) if s["kind"] != "pool"]
    arithmetic = [s for s in readout if s["kind"] != "activation"]
    if (
        [s["kind"] for s in arithmetic] != ["convert", "scale", "requant"]
        or arithmetic[0].get("to") != "f32"
        or arithmetic[1].get("dtype") != "f32"
        or any(s.get("rounding") != "half_even" for s in arithmetic)
    ):
        raise ValueError("composition requires the declared f32 RNE readout sequence")
    if relu and not any(s["kind"] == "activation" and s.get("function") == "relu" for s in readout):
        raise ValueError("composition requires an available ReLU")
    _, dtype = mq.type_shape_dtype(quantize.results[0].type)
    narrow = arithmetic[-1]
    clamp = [GN._int_attr(quantize, "quant_min"), GN._int_attr(quantize, "quant_max")]
    if CG._dtype_token(dtype) != CG._dtype_token(narrow.get("to")) or narrow.get("clamp") != clamp:
        raise ValueError("composition output dtype or clamp differs from the declared readout")
    return {"accumulator": dict(path["accumulator"]), "readout_stages": readout}


def _sum_plan(members, scheme, path, facet, facts):
    import numpy as np

    from merlin.runtime.exact_readout import form_multiplier
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    stages = {id(op): CG.classify(op) for op in members}
    summed, why = CG._operand_sum_of(members, stages)
    if summed is None:
        raise ValueError(why or "source is not a closed dequantize/add/quantize operand sum")
    if not callable(getattr(facet, "operand_sum_refusal", None)):
        raise ValueError("scaled-load admission needs a selected readout facet")
    sources = [op for op in members if stages[id(op)].kind == CG.DEQUANTIZE]
    from merlin.common import mlir_query as mq

    operand_dtype = (getattr(facet, "operand_sum", None) or {}).get("operand_dtype")
    if operand_dtype is None or any(
        CG._dtype_token(mq.type_shape_dtype(op.operands[0].type)[1]) != CG._dtype_token(operand_dtype) for op in sources
    ):
        raise ValueError("source operand formats differ from selected load facet")
    quantize = next(op for op in members if stages[id(op)].kind == CG.QUANTIZE)
    output = GN._scale_source(quantize).value
    multipliers = [float(form_multiplier(GN._scale_source(op).value, output)) for op in sources]
    if any(not np.isfinite(m) or not 0 < m <= 1 for m in multipliers):
        raise ValueError("operand load multipliers must be in (0, 1]")
    reason = facet.operand_sum_refusal(multipliers)
    if reason:
        raise ValueError(reason)
    if scheme.get("operand_sum_rounding") not in {"single_rounding", "per_operand"}:
        raise ValueError("operand sum requires an explicit rounding scheme")
    # The facet, including the scheme-selected rounding, owns admission.
    if any(m != 1 for m in multipliers) and facts.get("operand_sum_rounding") != scheme["operand_sum_rounding"]:
        raise ValueError("selected load facts and operand-sum rounding scheme differ")
    if not path.get("exits", {}).get("requant"):
        raise ValueError("operand sum requires a requant exit")
    readout = _scalar_readout(path, quantize, relu=summed["relu"])
    return {
        **readout,
        "operand_multiplier_bits": np.asarray(multipliers, dtype=np.float32).view(np.uint32).tolist(),
        "operand_sum_rounding": scheme["operand_sum_rounding"],
        "multiplier_bits": [int(np.asarray(form_multiplier(1, 1)).view(np.uint32))],
        "relu": summed["relu"],
        "load_minimum_not_negated": True,
    }


def mean_realization(path, *, operand_dtype, accumulator_bits):
    """Static prerequisites of the existing ones-matmul average plan.

    Source window/divisor, all-prefix bounds and multiplier formation remain
    per-instance obligations of ``_mean_plan``.
    """
    if operand_dtype != "i8":
        raise ValueError("ones-matmul average requires signed i8 input")
    if type(accumulator_bits) is not int or accumulator_bits < 2:
        raise ValueError("ones-matmul average needs a declared signed accumulator width")
    if not path.get("exits", {}).get("requant"):
        raise ValueError("average requires a requant exit")
    scales = [s for s in path.get("readout", []) if s["kind"] == "scale"]
    if len(scales) != 1 or scales[0].get("dtype") != "f32" or scales[0].get("rounding") != "half_even":
        raise ValueError("average requires the derived f32 RNE scale path")
    return {
        "lowering": "ones_matmul",
        "definitional": "f32_product_rne",
        "accumulator_bits": accumulator_bits,
        "obligations": [
            "closed trailing window",
            "divisor equals window",
            "all-prefix sum fits accumulator",
            "per-case exhaustive f32 division equals reciprocal product after RNE",
        ],
    }


@lru_cache(maxsize=256)
def mean_division_matches_product(window):
    """Exhaust the signed-byte sum domain at the source's f32/RNE boundaries."""
    import numpy as np

    if type(window) is not int or window <= 0:
        raise ValueError("mean window must be a positive integer")
    divisor = np.float32(window)
    reciprocal = np.float32(1) / divisor
    # Bound temporary storage even for large, accumulator-safe windows.
    for start in range(-128 * window, 127 * window + 1, 65536):
        sums = np.arange(start, min(start + 65536, 127 * window + 1), dtype=np.int64).astype(np.float32)
        if not np.array_equal(np.rint(sums / divisor), np.rint(sums * reciprocal)):
            return False
    return True


def require_exact_mean_window(window):
    if not mean_division_matches_product(window):
        raise ValueError(f"mean window {window}: f32 division and reciprocal product disagree after RNE")


def _composed_mean(members, stages):
    """``(mean, None, relu)`` for the checked mean prefix followed by a proved tail to one quantize.

    The tail is closed and ordered: the divide's value reaches the quantize through views and at
    most one ReLU, and no other member exists. Any other stage -- an independent scale, a pad or a
    spatial pool -- is refused here: those need a source-derived readout/pool geometry and the
    provider's pooled drain, which this plan does not yet describe.
    The window, scales and exactness gates stay those of the strict route.
    """
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    prefix, why = CG.window_mean_prefix(members, stages)
    if prefix is None:
        return None, why, False
    quantizes = [op for op in members if stages[id(op)].kind == CG.QUANTIZE]
    tail = [
        op
        for op in members
        if op not in (prefix["dequantize"], prefix["reduce"], prefix["divide"], *quantizes)
        and stages[id(op)].kind != CG.VIEW
    ]
    if len(quantizes) != 1 or [stages[id(op)].kind for op in tail] not in ([], [CG.RELU]):
        return None, "mean tail is not views and at most one ReLU before one quantize", False
    reached = CG._through_views(quantizes[0].operands[0])
    if tail:
        if reached is not tail[0]:
            return None, "the quantize does not read the mean tail", False
        reached = CG._through_views(tail[0].operands[0])
    if reached is not prefix["divide"]:
        return None, "the mean tail does not read the division", False
    try:
        scales = [GN._scale_source(op) for op in (prefix["dequantize"], quantizes[0])]
    except GN.GroupNumericsError as exc:
        return None, str(exc), False
    if any(scale.value is None for scale in scales) or any(scale.zero_point for scale in scales) or not scales[1].value:
        return None, "a scale of the mean is not a compile-time number, or a zero point is carried", False
    mean = {
        "reduce": prefix["reduce"],
        "rows": prefix["rows"],
        "window": prefix["window"],
        "multiplier": scales[0].value / (float(prefix["count"]) * scales[1].value),
        "bound_lsb": 1,
        "exactness": "float_reassociation",
        "out_elements": prefix["out_elements"],
    }
    return mean, None, bool(tail)


def _mean_plan(members, scheme, path, facts):
    import numpy as np

    from merlin.runtime.exact_readout import form_multiplier
    from merlin.targetgen.exact_arithmetic import rational
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    stages = {id(op): CG.classify(op) for op in members}
    mean, why = CG._window_mean_of(members, stages)
    relu = False
    if mean is None and why is None:
        # Not the strict four-member mean: accept only the checked prefix plus a tail proved here.
        mean, why, relu = _composed_mean(members, stages)
    if mean is None:
        raise ValueError(why or "source is not a closed trailing-window average")
    if scheme.get("readout_strategy") != "f32_product_rne":
        raise ValueError("average reassociation requires the declared single f32 boundary")
    dequantize = next(op for op in members if stages[id(op)].kind == CG.DEQUANTIZE)
    quantize = next(op for op in members if stages[id(op)].kind == CG.QUANTIZE)
    from merlin.common import mlir_query as mq

    mean_realization(
        path,
        operand_dtype=mq.type_shape_dtype(dequantize.operands[0].type)[1],
        accumulator_bits=facts.get("accumulator_bits"),
    )
    divisor = next(op for op in members if stages[id(op)].kind == CG.SCALE)
    count = GN._constant_of(divisor.operands[1])
    if count != mean["window"]:
        raise ValueError("average divisor differs from its source window")
    require_exact_mean_window(mean["window"])
    bits = facts.get("accumulator_bits")
    if type(bits) is not int or bits < 2 or mean["window"] * 128 >= 1 << (min(bits, 32) - 1):
        raise ValueError("ones-matmul sum exceeds the declared accumulator")
    if not path.get("exits", {}).get("requant"):
        raise ValueError("average requires a requant exit")
    multiplier = form_multiplier(
        GN._scale_source(dequantize).value, rational(count) * rational(GN._scale_source(quantize).value)
    )
    # ReLU after the division commutes with the positive mean multiplier and RNE (both monotone,
    # both fix zero), so the source order divide -> ReLU -> Q equals the readout's scale/ReLU/RNE.
    if relu and not multiplier > 0:
        raise ValueError("a ReLU tail needs a positive mean multiplier to commute with the readout")
    readout = _scalar_readout(path, quantize, relu=relu)
    return {
        **readout,
        "relu": relu,
        "lowering": "ones_matmul",
        "rows": mean["rows"],
        "window": mean["window"],
        "constant_weight": 1,
        "multiplier_bits": [int(np.asarray(multiplier).view(np.uint32))],
        "definitional": "f32_product_rne",
    }


def _pool_limits(decision, facts):
    geometry = decision["proof"].get("pool_geometry")
    if geometry is None:
        raise ValueError("source pool geometry is not established")
    for key, limit_key in (("window", "pool_window_max"), ("stride", "pool_stride_max")):
        maxima = facts.get(limit_key)
        values = geometry[key]
        if (
            not isinstance(maxima, (list, tuple))
            or len(maxima) != len(values)
            or any(
                type(limit) is not int or not 0 < value <= limit for value, limit in zip(values, maxima, strict=True)
            )
        ):
            raise ValueError("pool window or stride exceeds declared register fields or is unknown")


def plan_composition(module, candidate, *, path, scheme, facts, readout_facet=None, runtime_scale_facts=None):
    """Return a provider-consumable semantic plan or raise a local premise refusal."""
    from merlin.targetgen.native_mac_bound import require_value_mac_bound
    from merlin.targetgen.output_path import validate
    from merlin.verify.lean_contract import document_digest

    path = validate(path)
    declaration = candidate["detail"]["composition"]
    for kind, requirement in declaration.get("accumulator_stages", {}).items():
        if requirement == "required" and not any(s["kind"] == kind for s in path.get("accumulator_stages", ())):
            raise ValueError(f"composition requires declared accumulator stage {kind}")
    requested = declaration.get("readout_stages", ())
    if requested != "from_output_path" and not set(requested) <= {s["kind"] for s in path.get("readout", ())}:
        raise ValueError("composition requires an unavailable readout stage")
    operations = tuple(module.walk())
    members = tuple(operations[i] for i in candidate["operation_ordinals"])
    _closed_members(members)
    root = operations[candidate["detail"]["root_ordinal"]]
    route = candidate["route"]
    if route not in composition_routes(declaration):
        raise ValueError("composition route differs from declared semantics")
    if route == "output_stationary":
        if "OS" not in facts.get("dataflows", ()):
            raise ValueError("OS dataflow is not established by selected facts; retain WS")
        decision = require_value_mac_bound(root, mesh_signed_bits=facts.get("mesh_signed_bits"))
        decision["fallback_dataflow"] = "WS"
    elif route == "k_concat":
        from merlin.xdsl_dialects.lowering.output_fusion import _Refused

        try:
            decision = _k_concat(module, root, path, scheme, facts)
        except _Refused as error:
            raise ValueError(error.reason) from error
    elif route == "residual_on_load":
        decision = _sum_plan(members, scheme, path, readout_facet, facts)
    elif route == "avgpool":
        decision = _mean_plan(members, scheme, path, facts)
    else:
        decision = _readout_plan(module, root, route, path, scheme, runtime_scale_facts=runtime_scale_facts)
        if decision.get("proof", {}).get("scale_operand"):
            from merlin.xdsl_dialects.lowering import output_fusion as OF

            # Composite generics count once; omitted or duplicate source members refuse.
            consumed = (root, *OF.source_chain(root, OF.Values()).ops)
            if members != consumed:
                raise ValueError("runtime plan operation ordinals must cover every consumed source member once")
        if route == "conv_pool":
            _pool_limits(decision, facts)
    return CompositionPlan(
        route,
        ir_digest(module),
        tuple(candidate["operation_ordinals"]),
        {
            **decision,
            "scheme_sha256": document_digest(scheme),
            "facts_sha256": document_digest(facts),
            "output_path_sha256": document_digest(path),
        },
    )
