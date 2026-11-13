"""Select a bounded first-valid partition from target-admitted tensor regions."""

from __future__ import annotations

from dataclasses import replace

from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, GlobalPlan, verify_global_plan

from .semantic_regions import outline_semantic_regions, semantic_dispatch


def select_application_plan(source, alternatives, *, max_states=10_000):
    """Choose fusion and placement without inventing numerical or cost facts.

    Providers enumerate legal regions, representations and placements. Merlin
    chooses a complete compatible cover, including checking noncontiguous
    fusions for cycles. Larger regions are tried first, with stable tie breaking.
    This is bounded first-valid selection, not a performance optimization.
    """
    if type(max_states) is not int or max_states <= 0:
        raise ValueError("application planning needs a positive work budget")
    program = semantic_dispatch(source, entry="application")
    choices = tuple(sorted(alternatives, key=lambda row: (-len(row.nodes), row.nodes, row.id)))
    if len({row.id for row in choices}) != len(choices):
        raise ValueError("application region alternatives need distinct identities")
    domain = frozenset(range(len(program.nodes)))
    if any(not row.nodes or not set(row.nodes) <= domain for row in choices):
        raise ValueError("application alternative references an invalid source node")
    stack = [((), domain)]
    visited = 0
    while stack:
        if visited >= max_states:
            raise NativeCompilationError("resource_limit", "application partition search work limit reached")
        selected, remaining = stack.pop()
        visited += 1
        if not remaining:
            plan = GlobalPlan(
                tuple(sorted(selected, key=lambda row: row.nodes)),
                (),
                CycleInterval.unknown("first-valid application partition; costs are not ranked"),
            )
            if verify_global_plan(program, plan):
                continue
            try:
                outline_semantic_regions(source, entry="application", plan=plan)
            except ValueError:
                continue
            return plan
            continue
        anchor = min(remaining)
        eligible = [row for row in choices if anchor in row.nodes and set(row.nodes) <= remaining]
        for row in reversed(eligible):
            stack.append(((*selected, row), remaining - set(row.nodes)))
    raise NativeCompilationError("unsupported_semantics", "no compatible application partition covers every node")


def _host_contraction(region):
    from .c_bf16_linear import emit_c

    try:
        emit_c(region, "host_contraction_check")
    except (KeyError, ValueError):
        return False
    return True


def fold_host_permutations(source, plan):
    """Read single-use host permutations through their host contraction.

    A selected host region holding only a pure, complete dimension permutation
    is merged into the one host region that reads it when that region, with
    and without the permutation, is a host contraction admitted by Merlin's
    exact host contraction emitter. The emitter then indexes the original
    operand instead of a materialized copy. Values, products and reduction
    order are unchanged, and the permuted tensor is no longer allocated.
    Permutations with other readers, published results, broadcasts, slices
    or non-contraction readers keep their own regions.
    """
    from .c_bf16_linear import permutation_axes
    from .semantic_regions import extract_semantic_region

    program = semantic_dispatch(source, entry="application")
    key_of = lambda index: program.nodes[index].outputs[0]
    owner = {key_of(index): item.id for item in plan.selected for index in item.nodes}
    selected = {item.id: item for item in plan.selected}
    folded, notes = set(), []
    for item in plan.selected:
        if item.placement != "host" or len(item.nodes) != 1 or item.id in folded:
            continue
        view = source.node(key_of(item.nodes[0]))
        base = source.node(view.inputs[0]) if len(view.inputs) == 1 else None
        readers = [node.id for node in source.nodes if view.id in node.inputs]
        if (base is None or view.id in source.outputs or len(readers) != 1
                or permutation_axes(base, view) is None or owner.get(readers[0]) in {None, item.id}):
            continue
        consumer = selected[owner[readers[0]]]
        if consumer.placement != "host":
            continue
        nodes = tuple(sorted((*item.nodes, *consumer.nodes)))
        region = extract_semantic_region(source, tuple(key_of(index) for index in nodes))
        if not _host_contraction(region) or not _host_contraction(
                extract_semantic_region(source, tuple(key_of(index) for index in consumer.nodes))):
            continue
        boundary = [node.id for node in region.nodes if node.effect in {"input", "constant"}]
        representations = {row.buffer: row for row in (*item.inputs, *consumer.inputs)}
        # The emitter indexes the base as the dense row-major tensor the
        # consumer used to read, so the base's whole representation (layout,
        # encoding, quantization, attributes, placement, dtype) must be the
        # consumer's former view input representation.
        former = [row.representation for row in consumer.inputs if row.buffer == view.id]
        bases = {row.representation for row in (*item.inputs, *consumer.inputs) if row.buffer == base.id}
        if (set(boundary) - set(representations) or len(former) != 1 or bases != {former[0]}
                or former[0].dtype != base.type.dtype):
            continue
        candidate = replace(plan, selected=tuple(
            replace(consumer, nodes=nodes, inputs=tuple(representations[key] for key in boundary))
            if row.id == consumer.id else row for row in plan.selected if row.id != item.id))
        if verify_global_plan(program, candidate):
            continue
        try:
            outline_semantic_regions(source, entry="application", plan=candidate)
        except ValueError:
            continue
        plan = candidate
        selected = {row.id: row for row in plan.selected}
        owner = {key_of(index): row.id for row in plan.selected for index in row.nodes}
        folded.add(item.id)
        notes.append(f"host permutation {view.id} folded into contraction region {consumer.id}")
    if notes:
        plan = replace(plan, notes=(*plan.notes, *notes))
    return plan


def application_plan(source, adapter):
    """Use shared selection for region-enumerating providers.

    Legacy providers may still supply a complete plan. A failed enumeration
    never falls back to that route or silently changes placement. Either plan
    then reads single-use host permutations through their host contraction.
    """
    enumerate_regions = getattr(adapter, "application_region_alternatives", None)
    if enumerate_regions is not None:
        return fold_host_permutations(source, select_application_plan(source, enumerate_regions(source)))
    return fold_host_permutations(source, adapter.plan(source))
