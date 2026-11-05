"""Select a bounded first-valid partition from target-admitted tensor regions."""

from __future__ import annotations

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


def application_plan(source, adapter):
    """Use shared selection for region-enumerating providers.

    Legacy providers may still supply a complete plan. A failed enumeration
    never falls back to that route or silently changes placement.
    """
    enumerate_regions = getattr(adapter, "application_region_alternatives", None)
    if enumerate_regions is not None:
        return select_application_plan(source, enumerate_regions(source))
    return adapter.plan(source)
