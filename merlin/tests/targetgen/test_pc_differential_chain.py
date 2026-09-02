"""The `PC` differential is reachable end to end, and this is the chain that reaches it.

`PC` claims that issuing stage two's transfer before stage one's wait costs fewer cycles on identical
work. The emitter exists (`workload_gen.plan_matmul(hoist_transfers=True)`); what `run_overnight`
declares unwired is the DRIVER that turns two programs into a verdict. This pins that the pieces
compose, so wiring it is assembly rather than rediscovery:

    Instruction[] + Effects[]
      -> depgraph.build_dag(roles=, resolved_separations=)
      -> depgraph.makespan(dag, order)            once per ordering
      -> depgraph.to_composed(cycles, dag)
      -> differential.compare(a, b, demands_a=, demands_b=)

Why the comparison is SOUND for this claim, which is the part worth pinning rather than the arithmetic:
`demands_of` counts exposed edges over the WHOLE graph, and two orderings of the same instructions
expose the same unpriced edges. So the unknowns are identical on both sides and cancel out of the
difference -- which is exactly the case `differential`'s own docstring names ("changing an overlap
policy changes how terms compose while leaving every demand alone").

⚠️ WITHOUT A PRICED TRANSFER LATENCY THE TWO ORDERS COST THE SAME, and the comparison says so. That is
not the chain failing; it is the chain refusing to credit a saving it has no basis for. Both directions
are asserted below, because a probe that only ever showed a difference could not tell the two apart.
"""
from __future__ import annotations

from merlin.perf import depgraph as DG
from merlin.perf import differential as DF
from merlin.perf.deps.liveness import Effects, Instruction

_MNEMS = ("DLOAD", "DWAIT", "DLOAD", "DWAIT", "MUL")
_UNHOISTED = [0, 1, 2, 3, 4]        # load, wait, load, wait, use  -- strictly serialized
_HOISTED = [0, 2, 1, 3, 4]          # load, load, wait, wait, use  -- both outstanding
_ROLES = {"DLOAD": "movement", "DWAIT": "movement", "MUL": "matmul"}


def _program():
    ins = [Instruction(index=i, mnemonic=m, operands={}, branch_target=None, section="text")
           for i, m in enumerate(_MNEMS)]
    eff = []
    for i, m in enumerate(_MNEMS):
        if m == "DLOAD":                        # a transfer defines the staging slot
            eff.append(Effects(defs=(f"slot{i}",), uses=(), unresolved=(), observed=True))
        elif m == "MUL":                        # the consumer uses both, so both must have landed
            eff.append(Effects(defs=("acc",), uses=("slot0", "slot2"), unresolved=(), observed=True))
        else:
            eff.append(Effects(defs=(), uses=(), unresolved=(), observed=True))
    return ins, eff


def _dag(separations):
    ins, eff = _program()
    # `stall_unit` is CYCLES PER UNIT of a stall instruction's immediate, not a label. This probe has
    # no stall instruction, so an earlier draft passed a string here and never noticed -- `cost_of`
    # only multiplies by it on the stall path.
    issue = DG.IssueModel(issue_cycles=1.0, stall_unit=1.0, tier="fixture",
                          provenance="pc-chain test")
    return DG.build_dag(ins, eff, issue=issue, stall_mnemonic="DELAY", roles=_ROLES,
                        resolved_separations=separations)


def test_the_two_orderings_expose_the_same_unknowns():
    """The soundness precondition: identical demands, so the unpriced parts cancel."""
    dag = _dag({"separation.movement": 10.0, "separation.matmul": 1.0})
    demands = DG.demands_of(dag)
    # One graph, two orders -- the demand count is a property of the graph, so it cannot differ.
    a = DG.to_composed(DG.makespan(dag, _UNHOISTED), dag)
    b = DG.to_composed(DG.makespan(dag, _HOISTED), dag)
    assert a.unresolved == b.unresolved, (
        "the two orderings leave different unknowns exposed, so their contributions cannot cancel "
        "and a difference of the resolved parts would be attributed to the wrong place")
    assert DF.comparable(a, b, demands_a=demands, demands_b=demands)


def test_a_priced_transfer_makes_the_hoist_measurably_cheaper():
    """With the completion latency priced, the hoisted order is shorter and the verdict is sound."""
    dag = _dag({"separation.movement": 10.0, "separation.matmul": 1.0})
    demands = DG.demands_of(dag)
    unhoisted = DG.makespan(dag, _UNHOISTED)
    hoisted = DG.makespan(dag, _HOISTED)
    assert hoisted < unhoisted, (
        f"hoisting did not reduce the makespan ({hoisted} vs {unhoisted}); with a priced transfer "
        f"latency, overlapping two transfers must cost less than serializing them")

    cmp = DF.compare(DG.to_composed(unhoisted, dag), DG.to_composed(hoisted, dag),
                     demands_a=demands, demands_b=demands,
                     label_a="not_hoisted", label_b="hoisted")
    assert "difference of the resolved parts" in cmp.reason, (
        f"the comparison did not license a difference: {cmp.reason!r}")


def test_an_unpriced_transfer_yields_no_saving_rather_than_a_guessed_one():
    """The mirror. A chain that always reported a win could not tell a real one from an assumed one."""
    dag = _dag(None)
    demands = DG.demands_of(dag)
    unhoisted = DG.makespan(dag, _UNHOISTED)
    hoisted = DG.makespan(dag, _HOISTED)
    assert hoisted == unhoisted, (
        "with the transfer latency UNPRICED the two orders must cost the same; a saving here would be "
        "credited to a quantity nobody measured")
    cmp = DF.compare(DG.to_composed(unhoisted, dag), DG.to_composed(hoisted, dag),
                     demands_a=demands, demands_b=demands)
    assert "equal" in cmp.reason
