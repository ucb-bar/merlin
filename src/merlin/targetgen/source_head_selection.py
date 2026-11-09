"""Resolve a head request from original eligible source identities only."""


def resolve_request(module, groups, eligible_groups, *, preferences, eligible_source_regions):
    from dataclasses import replace

    try:
        selected, record = resolve_heads(module, groups, eligible_groups,
                                         eligible_source_regions=eligible_source_regions)
        rows = preferences.get("shared_rhs_row_views", False)
        if type(rows) is not bool:
            raise ValueError("shared RHS row-view request requires a Boolean")
        if rows:
            selected = replace(selected, shared_rhs_row_views=True)
            record = {**record, "shared_rhs_row_views": "source_selected_not_qualified",
                      "effect_or_placement_grant": False}
        whole = preferences.get("whole_head_batches", False)
        if type(whole) is not bool:
            raise ValueError("whole head batch request requires a Boolean")
        if whole:
            selected = replace(selected, whole_batch=True, effects=replace(
                selected.effects, qualification_identity="owned host requires synchronous noescape/no-release "
                "whole-batch calls; every original product, ranked interface, actual bridge and object "
                "remain independently checked; no target issue or storage capability supplied"))
            record = {**record, "whole_head_batches": "source_selected_not_qualified",
                      "kernel_invocations": record["head_families"],
                      "matrix_products": record["matrix_products"],
                      "effect_or_placement_grant": False}
        if preferences["shared_input_siblings"]:
            try:
                return resolve_joint(module, groups, eligible_groups, selected, record,
                                     eligible_source_regions=eligible_source_regions)
            except ValueError as exc:
                record = {**record, "sibling_composition_unavailable": str(exc)}
        deferred = ["shared_input_siblings"] if preferences["shared_input_siblings"] else []
        return selected, {**record,
                          "deferred_requests": deferred,
                          "composition_order": ("whole_heads_first_siblings_deferred" if whole else
                                                "independent_heads_first_siblings_fallback_only")}
    except ValueError as exc:
        if not preferences["shared_input_siblings"]:
            return None, {"status": "not_applicable", "kind": "independent_heads", "reason": str(exc),
                          "baseline_retained": True}
    return None


def resolve_joint(module, groups, eligible_groups, selected, record, *, eligible_source_regions):
    """Resolve only disjoint remaining source roots after exact head refolding."""
    from dataclasses import replace

    from merlin_experiments.phase1.feedback.private_group_provenance import eligible_source_identity
    from xdsl.dialects import func

    from merlin.llvmlower.head_source_route import project_heads
    from merlin.llvmlower.sibling_components import prepare_sibling_components
    from merlin.llvmlower.sibling_source_route import _command_buffer, _text

    before = _text(module)
    plan = prepare_sibling_components(project_heads(module).module, effects=selected.effects)
    original = {eligible_source_identity(g.root): g.index for g in groups if g.placement != "host"}
    covered = []
    for component in plan.components:
        _, group = _command_buffer(component.core, "source_identity_only")
        function = next(op for op in component.core.body.block.ops if type(op) is func.FuncOp)
        operations = tuple(function.body.block.ops)
        for index in group.contraction_ordinals:
            identity = eligible_source_identity(operations[index])
            if identity not in original:
                raise ValueError("joint sibling member lost original source identity")
            covered.append(original[identity])
    heads = set(record["composed_source_groups"])
    if (not covered or len(set(covered)) != len(covered) or heads.intersection(covered)
            or not set(covered) <= set(eligible_groups)):
        raise ValueError("joint source needs disjoint admitted sibling components")
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp
                    and op.sym_name.data == "forward")
    positions = {id(op): i for i, op in enumerate(function.body.block.ops)}
    allowed = {(positions[id(g.root)], g.index, eligible_source_identity(g.root)[0], "contraction")
               for g in groups if g.index in covered}
    supplied = [(r.get("source_op_index"), r.get("source_group"), r.get("region_id"), r.get("semantic_family"))
                for r in eligible_source_regions if r.get("source_group") in covered]
    if len(supplied) != len(set(supplied)) or set(supplied) != allowed:
        raise ValueError("joint siblings need complete original contraction-only evidence")
    if _text(module) != before:
        raise ValueError("joint selection changed original source")
    all_covered = sorted(heads | set(covered))
    return replace(selected, shared_input_siblings=True), {
        **record, "kind": "independent_heads_and_shared_input_siblings",
        "composed_source_groups": all_covered, "sibling_source_groups": covered,
        "unmatched_source_groups": sorted(set(eligible_groups) - set(all_covered)),
        "component_count": len(plan.components), "deferred_requests": [],
        "composition_order": "checked_heads_then_disjoint_siblings_then_ordinary",
    }


def resolve_heads(module, groups, eligible_groups, *, eligible_source_regions=None):
    from merlin_experiments.phase1.feedback.private_group_provenance import eligible_source_identity
    from xdsl.dialects import func

    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.head_source_route import HeadBatchSourceRoute, project_heads
    from merlin.llvmlower.sibling_source_route import _sha, _text

    before = _text(module)
    projection = project_heads(module)
    original = {eligible_source_identity(g.root): g.index for g in groups if g.placement != "host"}
    if (len(original) != sum(g.placement != "host" for g in groups)
            or not set(eligible_groups) <= set(original.values())):
        raise ValueError("head source has ambiguous original admitted identities")
    covered = []
    for family in projection.members:
        for member in family["members"]:
            identity = member["source_region_id"], tuple(member["source_node_ids"])
            if identity not in original:
                raise ValueError("head source member is not an independently admitted original group")
            covered.append(original[identity])
    if not covered or len(set(covered)) != len(covered):
        raise ValueError("head source omits or duplicates original products")
    if not set(covered) <= set(eligible_groups):
        raise ValueError("head source includes a product outside independently admitted groups")
    if (type(eligible_source_regions) is not list or not eligible_source_regions
            or any(type(row) is not dict for row in eligible_source_regions)):
        raise ValueError("head source requires complete original eligible-operation evidence")
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp
                    and op.sym_name.data == "forward")
    positions = {id(op): i for i, op in enumerate(function.body.block.ops)}
    allowed = {(positions.get(id(g.root)), g.index, eligible_source_identity(g.root)[0], "contraction")
               for g in groups if g.index in covered}
    supplied = [(r.get("source_op_index"), r.get("source_group"), r.get("region_id"), r.get("semantic_family"))
                for r in eligible_source_regions if r.get("source_group") in covered]
    if any(row not in allowed for row in supplied):
        raise ValueError("raw head batching cannot replace an independently required supporting operation")
    if len(supplied) != len(set(supplied)) or set(supplied) != allowed:
        raise ValueError("head source omits or duplicates original eligible-operation evidence")
    if _text(module) != before:
        raise ValueError("head selection changed original source")
    selected = HeadBatchSourceRoute(_sha(before), BorrowedBufferEffects(
        "owned host requires synchronous noescape/no-release per-matrix calls; "
        "actual source, stock slice bridge, device work and final object remain independently checked",
        True, True, True))
    return selected, {"status": "source_selected_not_qualified", "kind": "independent_heads",
                      "source_ir_sha256": selected.source_ir_sha256, "composed_source_groups": covered,
                      "original_source_groups": sorted(eligible_groups),
                      "unmatched_source_groups": sorted(set(eligible_groups) - set(covered)),
                      "original_source_denominator_unchanged": True, "head_families": len(projection.members),
                      "batch_abi_calls": len(projection.members), "matrix_products": len(covered),
                      "accelerator_execution_count_reduced": False}
