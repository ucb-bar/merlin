"""The `PC` driver: two orderings of one program in, a verdict or a named absence out.

`workload_gen.plan_matmul(hoist_transfers=)` emits the pair; `perf.reorder_claim` measures it. What is
pinned here is not the arithmetic but the four soundness checks and the refusal to guess:

* the same instruction multiset (the claim cancels it, so a differing pair measures the instruction);
* one dependence graph, two orders (two graphs have two edge sets, and a saving could come from an
  edge one side happens to lack);
* the hoisted sequence really is a permutation of the baseline's;
* a verdict ONLY when both the reorder's legality and the separation's price are established.

And the distinction that matters most for reading a result: `established` says the evidence sufficed,
`saving` says what it showed. PC's falsifier fires on `established and not saving` -- a verdict against
the claim -- which is a different thing from having measured nothing.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf import depgraph as DG
from merlin.perf import reorder_claim as RC
from merlin.perf import workload_gen as WG
from merlin.targetgen import isa_direction as ID

sys.path.insert(0, str(merlin_dir() / "tests" / "targetgen"))
import test_layer_workload_gen as LW  # noqa: E402

_KW = dict(control_flow=LW.CF, settle=LW.SETTLE, subnormal_operand_flush=False)
_ISSUE = DG.IssueModel(issue_cycles=1.0, stall_unit=1.0, tier="fixture", provenance="driver test")
_ROLES = {LW.OPS.dma_load: "movement", LW.OPS.dma_wait: "movement",
          LW.OPS.contract: "matmul", LW.OPS.contract_accumulate: "matmul"}
_PRICES = {"separation.movement": 10.0, "separation.matmul": 1.0}


def _pair():
    a = WG.plan_matmul(LW.synthetic_facts(), LW.OPS, m=4, k=8, n=4, **_KW)
    b = WG.plan_matmul(LW.synthetic_facts(), LW.OPS, m=4, k=8, n=4, hoist_transfers=True, **_KW)
    return a, b


def _directions():
    """A probe-SHAPED direction model. Synthetic, and labelled as such in its provenance."""
    def od(mn, op, direction, sf="x"):
        return ID.OperandDirection(mnemonic=mn, operand=op, direction=direction, state_file=sf,
                                   written_slots=(0,), reason="synthetic")
    by: dict = {}
    for mn in (LW.OPS.add, LW.OPS.add_imm, LW.OPS.load_upper):
        by[mn] = {"rd": od(mn, "rd", ID.DEF), "rs1": od(mn, "rs1", ID.USE),
                  "rs2": od(mn, "rs2", ID.USE)}
    by[LW.OPS.dma_load] = {k: od(LW.OPS.dma_load, k, ID.USE) for k in ("rd", "rs1", "rs2")}
    for mn in (LW.OPS.dma_wait, LW.OPS.stall, LW.OPS.halt, LW.OPS.branch_ne, LW.OPS.dma_store):
        by[mn] = {}
    for mn in (LW.OPS.tile_load, LW.OPS.tile_store, LW.OPS.transpose, LW.OPS.weight_push,
               LW.OPS.contract, LW.OPS.contract_accumulate, LW.OPS.acc_read):
        by[mn] = {"vd": od(mn, "vd", ID.DEF, "v"), "vs1": od(mn, "vs1", ID.USE, "v"),
                  "vs2": od(mn, "vs2", ID.USE, "v"), "rs1": od(mn, "rs1", ID.USE)}
    return ID.DirectionModel(target="synthetic", by_mnemonic=by, refused={},
                             provenance="synthetic probe (test)")


def test_the_pair_is_a_permutation_of_one_program():
    a, b = _pair()
    order = RC.permutation_of(a.instructions, b.instructions)
    assert sorted(order) == list(range(len(a.instructions))), "not a permutation"
    assert order != list(range(len(a.instructions))), "the hoist emitted the baseline order"


def test_a_differing_multiset_is_refused_not_compared():
    """The claim cancels the instruction multiset, so a pair differing by one is measuring that one."""
    a, b = _pair()
    # A mnemonic the program does not contain -- replacing HALT with HALT changes nothing, which is
    # how this assertion first passed vacuously.
    assert not any(m == "NOT_AN_INSTRUCTION" for m, _ in a.instructions)
    with pytest.raises(RC.ReorderClaimError, match="multiset"):
        RC.permutation_of(a.instructions, b.instructions[:-1] + (("NOT_AN_INSTRUCTION", {}),))
    with pytest.raises(RC.ReorderClaimError, match="not two orders of one program"):
        RC.permutation_of(a.instructions, b.instructions[:-1])


def test_repeated_identical_instructions_map_to_distinct_indices():
    """Stable matching: a program issuing the same instruction twice must not map both to the first."""
    base = (("A", {"x": 1}), ("A", {"x": 1}), ("B", {}))
    order = RC.permutation_of(base, (("B", {}), ("A", {"x": 1}), ("A", {"x": 1})))
    assert sorted(order) == [0, 1, 2]
    assert order[0] == 2


def test_without_evidence_the_driver_names_what_is_missing():
    """No direction model and no prices: two named absences, and no cycle claim."""
    a, b = _pair()
    v = RC.hoist_verdict(a, b, issue=_ISSUE, roles=_ROLES)
    assert v.established is False
    assert v.saving is None, "a saving must not be reported before the evidence is established"
    assert v.instructions_probed == 0 and v.instructions_total > 0
    kinds = {m.split(":")[0] for m in v.missing}
    assert kinds == {"operand_direction_model", "resolved_separations"}


def test_prices_alone_are_not_enough_because_legality_is_unestablished():
    """With no dependence edges every permutation looks legal, so a delta would be meaningless."""
    a, b = _pair()
    v = RC.hoist_verdict(a, b, issue=_ISSUE, roles=_ROLES, separations=_PRICES)
    assert v.established is False
    assert {m.split(":")[0] for m in v.missing} == {"operand_direction_model"}


def test_with_both_evidences_the_driver_returns_a_verdict():
    """Established means the measurement ran; `saving` is what it showed. They are separate."""
    a, b = _pair()
    v = RC.hoist_verdict(a, b, issue=_ISSUE, roles=_ROLES, directions=_directions(),
                         separations=_PRICES)
    assert v.established is True and v.missing == ()
    assert v.instructions_probed > 0
    assert isinstance(v.saving, bool), "an established verdict must say whether it saved"
    assert v.baseline_cycles is not None and v.reordered_cycles is not None
    assert v.delta == v.reordered_cycles - v.baseline_cycles
    # PC's falsifier fires on established-and-not-saving. Either outcome is a RESULT; what must never
    # happen is `established` with `saving is None`.
    assert v.reason
