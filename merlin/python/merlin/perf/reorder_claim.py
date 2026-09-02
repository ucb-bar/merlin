"""Turn two orderings of one program into a `PC` verdict, or into a named reason there is none.

`PC` claims that issuing stage two's transfer before stage one's wait costs fewer cycles on identical
work. `workload_gen.plan_matmul(hoist_transfers=)` emits the pair; this is the driver that measures it.

WHAT MAKES THE COMPARISON SOUND, and each is CHECKED here rather than assumed:

* **the same instruction multiset.** The claim's comparand cancels the instruction multiset, so a pair
  that differs by an instruction is measuring that instruction, not the ordering. Refused, not warned.
* **one graph, two orders.** The dependence graph is built ONCE, from the unhoisted program, and both
  orderings are evaluated against it. Two graphs would have two edge sets, and a saving could then come
  from an edge one side happens not to have.
* **the hoisted order is a PERMUTATION of that graph's instructions**, recovered by matching the
  hoisted sequence back to indices. If it does not match, the pair is not two orders of one program.
* **identical unpriced demand.** `depgraph.demands_of` counts exposed edges over the whole graph, so
  two orderings expose the same unknowns and they cancel out of the difference.

⚠️ TWO MEASUREMENTS THIS CANNOT INVENT, and the verdict names whichever is absent:

``directions``
    A measured operand-direction model (`isa_direction.DirectionModel`, from probing instructions in
    isolation). Without it every instruction's effects are UNRESOLVED, the graph has no dependence
    edges, and a reorder across a dependence would look legal. So with no model the answer is "the
    reorder's legality is not established" -- never a cycle count.

``separations``
    The priced completion latency per structural role. With it unpriced, both orders cost the same and
    the comparison says so: that is the chain declining to credit a saving it has no basis for, not a
    null result. A driver that reported a win here would be pricing a quantity nobody measured.

So the outcome is always one of: a verdict with a cycle delta, or a refusal naming the missing
evidence. There is no third branch that guesses.

``established`` and ``saving`` are SEPARATE, because "we could not measure" and "we measured no
saving" read alike and only the second is a result. `established` says the evidence sufficed;
`saving` says what it showed. PC's declared falsifier -- ``hoisting_does_not_improve_any_declared_
separation_regime`` -- fires on ``established and not saving``, which is a verdict against the claim
rather than an absence of one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from merlin.perf import depgraph as DG
from merlin.perf import differential as DF
from merlin.perf.deps.liveness import Effects, Instruction


class ReorderClaimError(ValueError):
    """The pair cannot be compared at all -- a statement about the INPUT, not about the machine."""


@dataclass(frozen=True)
class ReorderVerdict:
    """What the measurement established, and on what evidence."""

    #: The measurement RAN on sufficient evidence. It says nothing about the outcome -- conflating the
    #: two is how "we could not measure" and "we measured no saving" come to read alike, and only the
    #: second is a result. The claim's falsifier fires on the second, so they must stay apart.
    established: bool
    #: True when the reordering is cheaper. ``None`` when nothing was established.
    saving: bool | None = None
    #: Cycles for each ordering, when the separations were priced enough to resolve them.
    baseline_cycles: float | None = None
    reordered_cycles: float | None = None
    #: Negative means the reorder is cheaper.
    delta: float | None = None
    #: The comparison's own justification, or the reason there is no comparison.
    reason: str = ""
    #: Evidence classes that were absent. Empty on an established verdict.
    missing: tuple[str, ...] = ()
    #: Separation classes the graph could not price -- carried so a reader sees what was unknown.
    unresolved: tuple[str, ...] = ()
    instructions_probed: int = 0
    instructions_total: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)


def _as_instructions(items: Sequence[tuple[str, dict]]) -> list[Instruction]:
    return [Instruction(index=i, mnemonic=m, operands=dict(o), branch_target=None, section="text")
            for i, (m, o) in enumerate(items)]


def permutation_of(baseline: Sequence[tuple[str, dict]],
                   reordered: Sequence[tuple[str, dict]]) -> list[int]:
    """Indices into ``baseline`` in the order ``reordered`` issues them.

    Matched on (mnemonic, operands) and STABLY -- the first unclaimed identical instruction wins -- so
    a program that issues the same instruction twice maps each occurrence to a distinct index rather
    than both to the first. Raises when the two are not permutations of each other, because that is the
    one thing the caller must not be allowed to compare.
    """
    if len(baseline) != len(reordered):
        raise ReorderClaimError(
            f"the two programs have {len(baseline)} and {len(reordered)} instructions; a reordering "
            f"cannot change the count, so these are not two orders of one program")
    a = Counter((m, tuple(sorted(o.items()))) for m, o in baseline)
    b = Counter((m, tuple(sorted(o.items()))) for m, o in reordered)
    if a != b:
        only_a = sorted(str(k) for k in (a - b))[:3]
        only_b = sorted(str(k) for k in (b - a))[:3]
        raise ReorderClaimError(
            f"the instruction multisets differ (only in baseline: {only_a}; only in reordered: "
            f"{only_b}); the claim cancels the multiset, so a difference here would be measuring the "
            f"instruction rather than the order")
    remaining: dict[tuple, list[int]] = {}
    for i, (m, o) in enumerate(baseline):
        remaining.setdefault((m, tuple(sorted(o.items()))), []).append(i)
    order: list[int] = []
    for m, o in reordered:
        order.append(remaining[(m, tuple(sorted(o.items())))].pop(0))
    return order


def compare_orderings(baseline: Sequence[tuple[str, dict]],
                      reordered: Sequence[tuple[str, dict]], *,
                      issue: DG.IssueModel,
                      stall_mnemonic: str,
                      roles: Mapping[str, str] | None = None,
                      directions: Any = None,
                      separations: Mapping[str, float] | None = None) -> ReorderVerdict:
    """Measure ``reordered`` against ``baseline`` as two orders of ONE dependence graph."""
    from merlin.perf.deps.liveness import effects_of

    order = permutation_of(baseline, reordered)
    instrs = _as_instructions(baseline)

    if directions is None:
        eff = [Effects(defs=(), uses=(), unresolved=("<no measured direction model>",),
                       observed=False) for _ in instrs]
    else:
        eff = [effects_of(i, directions) for i in instrs]
    probed = sum(1 for e in eff if e.observed)

    dag = DG.build_dag(instrs, eff, issue=issue, stall_mnemonic=stall_mnemonic, roles=roles,
                       resolved_separations=separations)
    demands = DG.demands_of(dag)
    base_cycles = DG.makespan(dag, list(range(len(instrs))))
    reorder_cycles = DG.makespan(dag, order)
    a = DG.to_composed(base_cycles, dag)
    b = DG.to_composed(reorder_cycles, dag)
    cmp = DF.compare(a, b, demands_a=demands, demands_b=demands,
                     label_a="not_hoisted", label_b="hoisted")

    missing: list[str] = []
    if not probed:
        # THE LEGALITY QUESTION, and it comes first: with no dependence edges every permutation looks
        # legal, so a cycle delta here would be a delta between two programs one of which may not
        # compute the same thing.
        missing.append("operand_direction_model: no instruction was probed, so no dependence edge "
                       "exists and the reorder's LEGALITY is not established")
    if not separations:
        missing.append("resolved_separations: no structural role carries a priced completion latency, "
                       "so there is nothing for overlapping two transfers to save")

    established = not missing
    return ReorderVerdict(
        established=established,
        # Only meaningful once the evidence is sufficient; before that a delta is an artifact of what
        # was not modelled. PC's falsifier fires on `saving is False` with `established` -- "hoisting
        # does not improve any declared separation regime" -- which is a verdict, not a null result.
        saving=(reorder_cycles < base_cycles) if established else None,
        baseline_cycles=base_cycles, reordered_cycles=reorder_cycles,
        delta=reorder_cycles - base_cycles,
        reason=cmp.reason if established else "; ".join(missing),
        missing=tuple(missing), unresolved=tuple(a.unresolved),
        instructions_probed=probed, instructions_total=len(instrs),
        detail={"comparison_reason": cmp.reason, "demands": dict(demands)},
    )


def hoist_verdict(plan_baseline, plan_hoisted, *, issue: DG.IssueModel,
                  roles: Mapping[str, str] | None = None, directions: Any = None,
                  separations: Mapping[str, float] | None = None) -> ReorderVerdict:
    """:func:`compare_orderings` for the two plans ``plan_matmul`` emits with and without the hoist."""
    return compare_orderings(
        plan_baseline.instructions, plan_hoisted.instructions, issue=issue,
        stall_mnemonic=plan_baseline.ops.stall, roles=roles, directions=directions,
        separations=separations)
