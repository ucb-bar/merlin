"""Original-owner resolution of optional per-component source composition."""


def resolve_components(module, groups, eligible_groups, *, eligible_source_regions=None):
    from hashlib import sha256

    from merlin_experiments.phase1.feedback.private_group_provenance import eligible_source_identity
    from xdsl.dialects import func

    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.sibling_components import prepare_sibling_components
    from merlin.llvmlower.sibling_source_route import SiblingSourceRoute
    from merlin.xdsl_dialects._common import text

    before = text(module, generic=True)
    effects = BorrowedBufferEffects(
        "owned host requires synchronous noescape/no-release logical calls; "
        "actual implementation must independently pass dispatch and effect checks",
        True,
        True,
        True,
    )
    original = {eligible_source_identity(g.root): g.index for g in groups if g.placement != "host"}
    if len(original) != sum(g.placement != "host" for g in groups) or not set(eligible_groups) <= set(
        original.values()
    ):
        raise ValueError("source components have missing or ambiguous original eligibility")
    plan = prepare_sibling_components(module, effects=effects)
    covered = []
    for component in plan.components:
        from merlin.llvmlower.sibling_source_route import _command_buffer

        _, group = _command_buffer(component.core, "source_identity_only")
        function = next(op for op in component.core.body.block.ops if isinstance(op, func.FuncOp))
        ops = tuple(function.body.block.ops)
        for index in group.contraction_ordinals:
            identity = eligible_source_identity(ops[index])
            if identity not in original:
                raise ValueError("component member differs from original admitted source")
            covered.append(original[identity])
    if not covered or len(set(covered)) != len(covered):
        raise ValueError("source has no disjoint checked sibling components")
    if eligible_source_regions is not None:
        function = next(
            op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
        )
        indices = {id(op): i for i, op in enumerate(function.body.block.ops)}
        allowed = {
            (indices.get(id(g.root)), g.index, eligible_source_identity(g.root)[0], "contraction")
            for g in groups
            if g.index in covered
        }
        if any(
            (r.get("source_op_index"), r.get("source_group"), r.get("region_id"), r.get("semantic_family"))
            not in allowed
            for r in eligible_source_regions
            if r.get("source_group") in covered
        ):
            raise ValueError("component cannot replace independently required supporting source work")
    if text(module, generic=True) != before:
        raise ValueError("component resolution changed original source")
    return SiblingSourceRoute(sha256(before.encode()).hexdigest(), effects, partial=True), {
        "status": "source_selected_not_qualified",
        "source_ir_sha256": sha256(before.encode()).hexdigest(),
        "original_source_groups": sorted(eligible_groups),
        "composed_source_groups": covered,
        "unmatched_source_groups": sorted(set(eligible_groups) - set(covered)),
        "original_source_denominator_unchanged": True,
        "component_count": len(plan.components),
    }
