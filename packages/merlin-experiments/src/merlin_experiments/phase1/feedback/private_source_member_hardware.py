"""Hardware admission of actual typed source members, never a group's stored dtype."""

from xdsl.dialects import builtin

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen import semantic_families as SF
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin_experiments.phase1.feedback.private_source_payload_roles import payload_roles


def _formats(values):
    formats, ranks = [], []
    for value in values:
        typ = value.type
        shape = typ.get_shape() if hasattr(typ, "get_shape") else None
        if shape is None:
            if not isinstance(typ, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)):
                continue
            formats.append(CG._dtype_token(str(typ)))
            ranks.append(0)
        else:
            formats.append(CG._dtype_token(str(getattr(typ, "element_type", ""))))
            ranks.append(len(shape))
    return tuple(formats), tuple(ranks)


def observe(member, stage, capability):
    """Observe one operation under the original selected hardware declaration.

    No form, precision, fusion, or layout is supplied by placement or provenance.
    A missing constrained observation remains unknown. In particular, a legal
    integer input does not license a floating dequantization result. Consumers
    must retain any admitted or unknown member in the source-demand obligation.
    """
    name = mq.op_name(member)
    actual = CG.classify(member)
    if actual is not None and actual.kind != stage:
        raise ValueError("source member stage disagrees with its original operation")
    if actual is None and not (name == "memref.copy" and stage == CG.MOVEMENT):
        return {"status": "unknown", "refusal": "unclassified_source_member",
                "basis": "actual_typed_source_and_selected_capability"}
    family = CG.FAMILY_OF_STAGE.get(stage)
    getter = getattr(member, "get_inputs", None)
    if callable(getter):
        inputs = tuple(getter())
    elif name.startswith("linalg."):
        inputs = tuple(member.operands[:CG._n_inputs(member)])
    else:
        inputs = tuple(member.operands)
    if name in {"quant_ext.quantize_per_tensor", "quant_ext.quantize_per_channel",
                "quant_ext.dequantize_per_tensor", "quant_ext.dequantize_per_channel"}:
        # These original source operations have explicit data, scale, zero-point
        # roles. Register/parameter formats are not tensor datapath formats.
        if len(inputs) != 3:
            return {"status": "unknown", "refusal": "missing_quantization_roles",
                    "basis": "actual_typed_source_and_selected_capability"}
        inputs = inputs[:1]
    roles = payload_roles(member, inputs)
    formats, ranks = _formats(roles["inputs"])
    outputs, result_ranks = _formats(member.results)
    # memref.copy has a destination operand instead of an SSA result.
    if name == "memref.copy" and len(formats) == 2:
        outputs, result_ranks = formats[1:], ranks[1:]
        formats, ranks = formats[:1], ranks[:1]
    form = SF.operation_form(name.rpartition(".")[2], carrier_op=name)
    shaped_result = len(member.results) == 1 and hasattr(member.results[0].type, "get_shape")
    rank = result_ranks[0] if shaped_result or name == "memref.copy" and len(result_ranks) == 1 else None
    record = {"basis": "actual_typed_source_and_selected_capability", "operation": name,
              "family": family, "input_formats": list(formats), "result_formats": list(outputs),
              "rank": rank, "form": form, "layout": None,
              "input_role_basis": roles["basis"], "control_operand_indices": list(roles["control_indices"])}
    if family is None or not formats or not all(formats):
        return {**record, "status": "unknown", "refusal": "missing_typed_source_observation"}
    caps = E.capability_map_from_contract(dict(capability))
    if family != "contraction":
        checked, _ = E._family_support(family, caps)
        # Every ins slot is actual data unless the closed SSA role proof above
        # establishes control. An unsupported required payload width excludes
        # the operation even when it has multiple results and their lowering
        # form is unknown. All selected capabilities must exclude that width.
        if checked and all(any(not E._dtype_ok(fmt, cap.dtypes) for fmt in formats)
                           for cap in checked):
            return {**record, "status": "unsupported", "refusal": "input_dtype"}
    if len(outputs) != 1 or not outputs[0]:
        return {**record, "status": "unknown", "refusal": "missing_typed_source_observation"}
    descriptor = E.RegionDescriptor(
        op=stage, family=family, in_dtype=formats[0],
        weight_dtype=formats[1] if family == "contraction" and len(formats) == 2 else None,
        captured_input_formats=formats, out_dtype=outputs[0], rank=rank, form=form,
    )
    verdict = E.is_eligible(descriptor, caps,
                            undetermined=E.undetermined_families_from_contract(dict(capability)))
    # is_eligible's legacy direct API does not distinguish an absent constrained
    # form from an observed unsupported form. Source accounting must distinguish it.
    if verdict.refusal == "form" and form is None:
        return {**record, "status": "unknown", "refusal": "missing_source_form"}
    if verdict.eligible:
        checked, _ = E._family_support(family, caps)
        if any(cap.layouts for cap in checked or ()):
            return {**record, "status": "unknown", "refusal": "missing_source_layout"}
        if any(cap.ranks for cap in checked or ()) and rank is None:
            return {**record, "status": "unknown", "refusal": "missing_source_rank"}
    status = "admitted" if verdict.eligible else "unknown" if verdict.undetermined else "unsupported"
    return {**record, "status": status, "refusal": verdict.refusal}
