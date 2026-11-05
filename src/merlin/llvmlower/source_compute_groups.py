"""Complete source-group membership across all defined functions and control flow."""

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG


def form_source_groups(module, target, *, oracle=None, numeric_refusals=None, fusion=None):
    functions = [operation for operation in module.walk() if operation.name == "func.func" and operation.body.blocks]
    names = [function.sym_name.data for function in functions]
    if len(names) != len(set(names)):
        raise ValueError("source function identity is ambiguous")
    groups = []
    for name in names:
        groups.extend(
            CG.form_groups(
                module, target, oracle=oracle, function=name, numeric_refusals=numeric_refusals, fusion=fusion
            )
        )
    for index, group in enumerate(groups):
        group.index = index
    members = [id(member) for group in groups for member in group.members]
    if len(members) != len(set(members)):
        raise ValueError("source compute groups overlap")
    owned = set(members)
    for operation in module.walk():
        stage = CG.classify(operation)
        if stage is None or stage.kind in {CG.VIEW, CG.PAD, CG.MOVEMENT}:
            continue
        # Scalar address/control arithmetic is independently inventoried by
        # source admission; it is not a tensor compute-group definition.
        if not operation.name.startswith("linalg.") and not any(
            mq.type_shape_dtype(value.type)[0] for value in [*operation.operands, *operation.results]
        ):
            continue
        parent, nested = operation.parent_op(), False
        while parent is not None:
            if parent.name.startswith("linalg."):
                nested = True
                break
            parent = parent.parent_op()
        if not nested and id(operation) not in owned:
            raise ValueError("source computation outside the complete group membership remains unresolved")
    return groups


def form_exact_groups(module, target, capture, form, *, weight_args=None):
    """Form groups using the selected output path and capture-bound exactness proof.

    Both emission and source admission use this decision. A declared path owns
    its cuts; contracts without one retain the legacy float-transition proof.
    ``form`` supplies the caller's original function scope and target oracle.
    """
    from merlin.xdsl_dialects.lowering import group_exact_float as GEF
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    groups = form(module, target)
    path = OF.declared_path(target)
    if path is not None:
        decided = OF.decisions(groups, module, path, OF.Values(capture))
        if decided:
            groups = form(module, target, fusion=decided)
        return groups, sum(1 for decision in decided.values() if decision.refusals and decision.cut != "pool")
    refused = (
        GEF.refusals(groups, capture)
        if weight_args is None
        else GEF.refusals(groups, capture, weight_args=weight_args)
    )
    if refused:
        groups = form(module, target, numeric_refusals=refused)
    return groups, len(refused)
