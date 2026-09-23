"""The cost plane decides something, and a too-slow program is what it decides against.

WHAT THESE PIN. Three facts, in the order they had to become true:

1. A capsule can RESOLVE a ceiling. 113 capsules wrote ``derived_at_preflight`` in their one ceiling
   slot; nothing read it, so ``assess()`` came back ``incomplete`` for every one of them and the plane
   could refute a physically impossible cycle count but never a merely slow one.
   :func:`merlin.perf.cost_plane.resolve_ceiling` derives it now -- from the array the target's own
   facts declare, the work the capsule's own command buffer counted, and a slack declared in
   ``gate_phases.yaml`` beside the gate's phase.
2. Something READS ``blocking``. The plane computed that field on every verdict and nothing in the
   tree consumed it, so a phase flip would have changed what an artifact says and not what happens.
   :func:`~merlin.perf.cost_plane.require_within` raises on it and
   :func:`~merlin.perf.cost_plane.apply_gate` turns the raise into a failed row.
3. Only THEN does the phase matter, and the phase is read from the tracked declaration rather than
   passed as a literal -- so reverting that one line turns these tests red.

MUTATION, NOT SMOKE. Every assertion below is paired with the mutation that must break it: a ceiling
lowered under the measurement must FAIL, the same measurement under a ceiling it meets must PASS, and
every path where a ceiling could not be resolved must be ``incomplete`` -- never a pass, because "not
measured" reading as "measured fine" is the failure this plane exists for.
"""

from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.perf import cost_plane as CP
from merlin.perf import cycle_bound as CB
from merlin.perf import gate_phase as GP
from merlin.perf import preflight as PF

#: A 16x16 array retires 256 MAC slots per cycle; a 64x64x64 contraction is 262,144 MACs, so the
#: issue floor is 1024 cycles. Both numbers are DERIVED here (the test passes the geometry and the
#: work, exactly as the grader does) rather than asserted about any particular device.
ARRAY_ROWS = 16
ARRAY_COLS = 16
MACS = 64 * 64 * 64
FLOOR = MACS / (ARRAY_ROWS * ARRAY_COLS)


def _capsule(cycles_slot, *, name: str = "T") -> dict:
    """A capsule that is ON the timing axis, with whatever it declares in its one ceiling slot."""
    performance: dict = {"acceptance": {"evidence": {"timing_tier": "L3"}}}
    if cycles_slot is not None:
        performance["cost"] = {"projected_cycles": cycles_slot}
    return {"name": name, "performance": performance}


def _assess(capsule: dict, cycles: int | None, *, macs: int | None = MACS, **kw) -> dict:
    return CP.assess(
        capsule,
        tiers={"L3": {"cycles": cycles}} if cycles is not None else {"L3": {}},
        macs=macs,
        array_rows=kw.pop("array_rows", ARRAY_ROWS),
        array_cols=kw.pop("array_cols", ARRAY_COLS),
        phase=kw.pop("phase", GP.configured_phase(CP.GATE)),
        **kw,
    )


# ---------------------------------------------------------------------------------------------------
# 1. the ceiling exists at all
# ---------------------------------------------------------------------------------------------------


def test_the_declared_slack_is_read_from_the_tracked_file_and_is_usable() -> None:
    """The one number the plane decides with. It is a POLICY, so it lives where changing it is a
    one-line reviewable diff -- not in Python, where it would silently choose which submissions pass."""
    slack, basis = CP.declared_slack()
    assert slack is not None and slack >= 1, basis
    assert "gate_phases.yaml" in basis and "ceiling_slack_over_issue_floor" in basis
    declared = yaml.safe_load((merlin_dir() / "contract" / "gate_phases.yaml").read_text(encoding="utf-8"))
    assert declared["policies"][CP.GATE]["ceiling_slack_over_issue_floor"] == slack


def test_the_promissory_word_now_resolves_to_a_number() -> None:
    """``derived_at_preflight`` was written by 113 capsules and read by nothing. It resolves now, and
    the number it resolves to is the derived floor times the declared slack -- not a literal."""
    slack, _ = CP.declared_slack()
    floor = CP.derived_floor(macs=MACS, array_rows=ARRAY_ROWS, array_cols=ARRAY_COLS)
    ceiling, basis = CP.resolve_ceiling(_capsule("derived_at_preflight"), floor)
    assert ceiling == int(FLOOR * slack)
    assert "derived_at_preflight" in basis and "declared slack" in basis
    # The word is still in the corpus's vocabulary and still says why no literal is typed.
    assert "derived_at_preflight" in CB.NO_CYCLE_BOUND
    assert "derived_at_preflight" in CB.DERIVED_AT_ASSESSMENT


def test_an_authors_literal_beats_the_derivation() -> None:
    """An explicit integer is a stronger statement than a policy multiple. Overriding it would make
    the field decorative in the other direction."""
    floor = CP.derived_floor(macs=MACS, array_rows=ARRAY_ROWS, array_cols=ARRAY_COLS)
    ceiling, basis = CP.resolve_ceiling(_capsule(7), floor)
    assert ceiling == 7 and "declared in performance.cost.projected_cycles" in basis


# ---------------------------------------------------------------------------------------------------
# 2. the mutation: too slow FAILS, fast enough PASSES
# ---------------------------------------------------------------------------------------------------


def test_a_program_over_its_derived_ceiling_is_a_decided_failure() -> None:
    slack, _ = CP.declared_slack()
    ceiling = int(FLOOR * slack)
    verdict = _assess(_capsule("derived_at_preflight"), ceiling + 1)
    assert verdict["status"] == CP.STATUS_OVER
    assert verdict["admitted"] is False
    assert verdict["ceiling_cycles"] == ceiling
    with pytest.raises(CP.CostCeilingExceeded):
        CP.require_within(verdict)


def test_the_same_program_inside_the_ceiling_passes() -> None:
    """The other half of the mutation. Without it, a plane that failed EVERYTHING would pass the test
    above and be indistinguishable from one that decides."""
    slack, _ = CP.declared_slack()
    verdict = _assess(_capsule("derived_at_preflight"), int(FLOOR * slack))
    assert verdict["status"] == CP.STATUS_WITHIN
    assert verdict["admitted"] is True
    CP.require_within(verdict)  # does not raise


def test_lowering_the_ceiling_under_a_passing_measurement_flips_it() -> None:
    """The mutation stated as a mutation: one capsule, one measurement, two declared ceilings."""
    measured = int(FLOOR * 2)
    assert _assess(_capsule(measured + 1), measured)["status"] == CP.STATUS_WITHIN
    assert _assess(_capsule(measured - 1), measured)["status"] == CP.STATUS_OVER


def test_a_count_below_the_arrays_own_issue_floor_is_still_refuted() -> None:
    """The half of the plane that needs no slack at all, and must survive the half that does."""
    verdict = _assess(_capsule("derived_at_preflight"), int(FLOOR) - 1)
    assert verdict["status"] == CP.STATUS_BELOW_FLOOR
    with pytest.raises(CP.CostCeilingExceeded):
        CP.require_within(verdict)


# ---------------------------------------------------------------------------------------------------
# 3. every unresolved path is INCOMPLETE, never a pass
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slot",
    ["unbounded", None],
    ids=["declares-no-bound", "carries-no-cost-block"],
)
def test_a_capsule_that_owes_no_ceiling_is_incomplete_and_never_a_pass(slot) -> None:
    verdict = _assess(_capsule(slot), 10**9)
    assert verdict["status"] == CP.STATUS_INCOMPLETE
    assert verdict["admitted"] is False and verdict["blocking"] is False
    CP.require_within(verdict)  # an undecided plane blocks at no phase


def test_an_unresolvable_floor_is_incomplete_even_though_the_word_promises_a_number() -> None:
    """No geometry, no work, no floor -- and therefore no ceiling. The promise is not a permission."""
    verdict = _assess(_capsule("derived_at_preflight"), 10**9, macs=None, array_rows=None, array_cols=None)
    assert verdict["status"] == CP.STATUS_INCOMPLETE
    assert "UNKNOWN" in verdict["ceiling_basis"]
    assert verdict["admitted"] is False


def test_a_law_member_is_never_given_a_ceiling_whatever_its_slot_says() -> None:
    """A law-fitting point exists to be MEASURED, including at the slow end. Budgeting one would refuse
    the very observations that identify the machine's own rate -- so the exemption is read off the
    capsule's own ``performance.member_class`` and cannot be lost by a profile that forgot to spell it."""
    capsule = _capsule("derived_at_preflight")
    capsule["performance"]["member_class"] = "LAW"
    verdict = _assess(capsule, 10**9)
    assert verdict["status"] == CP.STATUS_INCOMPLETE
    assert verdict["ceiling_cycles"] is None
    assert "member_class" in verdict["ceiling_basis"]
    CP.require_within(verdict)  # and it blocks nothing
    # The other classes still owe one, or the exemption would be a blanket.
    capsule["performance"]["member_class"] = "OBJECTIVE"
    assert _assess(capsule, 10**9)["status"] == CP.STATUS_OVER


def test_an_undeclared_slack_stops_the_plane_deciding_rather_than_starting_it_admitting(monkeypatch) -> None:
    """Deleting the policy line must make the plane go quiet, not permissive. This is the direction a
    missing declaration is allowed to fail in."""
    monkeypatch.setattr(CP, "declared_slack", lambda: (None, "nobody declared one"))
    verdict = _assess(_capsule("derived_at_preflight"), 10**9)
    assert verdict["status"] == CP.STATUS_INCOMPLETE
    assert verdict["ceiling_cycles"] is None
    assert "nobody declared one" in verdict["ceiling_basis"]


def test_the_ceiling_arithmetic_refuses_a_slack_below_one() -> None:
    """A slack under 1 would demand a program finish faster than its own sequencer's loop takes."""
    assert PF.projected_cycle_ceiling(1024.0, slack=1.0)[0] == 1024
    assert PF.projected_cycle_ceiling(1024.0, slack=0.5)[0] is None
    assert PF.projected_cycle_ceiling(0.0, slack=8)[0] is None
    # Rounds UP: at slack 1 a fractional floor must still admit the only schedule that could meet it.
    assert PF.projected_cycle_ceiling(1024.2, slack=1.0)[0] == 1025


# ---------------------------------------------------------------------------------------------------
# 4. the enforcement call site, and the phase it honours
# ---------------------------------------------------------------------------------------------------


def test_the_gate_is_declared_at_fail_in_the_tracked_file() -> None:
    """Step three of the rollout, pinned. Reverting the one-line flip turns this red, which is the
    point of the flip being a one-line diff."""
    assert GP.configured_phase(CP.GATE) == GP.PHASE_FAIL


def test_apply_gate_fails_the_slow_row_and_leaves_the_fast_one_alone() -> None:
    """The enforcement call site, end to end, at whatever phase the tracked file declares."""
    slack, _ = CP.declared_slack()
    ceiling = int(FLOOR * slack)
    rows = [
        {
            "capsule": "slow",
            "status": "pass",
            "tiers": {"L3": {"cycles": ceiling * 4}},
            "work_volume": {"known_macs": MACS},
        },
        {
            "capsule": "fast",
            "status": "pass",
            "tiers": {"L3": {"cycles": ceiling}},
            "work_volume": {"known_macs": MACS},
        },
    ]
    capsules = [_capsule("derived_at_preflight", name="slow"), _capsule("derived_at_preflight", name="fast")]
    judged = CP.apply_gate(rows, capsules, array_rows=ARRAY_ROWS, array_cols=ARRAY_COLS)

    assert {row["capsule"] for row in judged} == {"slow", "fast"}, "a gate that ran must say what it judged"
    assert rows[0]["status"] == "fail"
    assert rows[0]["failure"]["plane"] == CP.PLANE
    assert rows[0]["failure"]["ceiling_cycles"] == ceiling
    assert rows[1]["status"] == "pass" and "failure" not in rows[1]
    assert rows[0][CP.PLANE]["status"] == CP.STATUS_OVER


def test_apply_gate_blocks_nothing_at_the_report_phase() -> None:
    """Which is the whole point of landing a gate at ``report`` first, and what makes the flip above a
    decision rather than an accident."""
    rows = [
        {"capsule": "slow", "status": "pass", "tiers": {"L3": {"cycles": 10**9}}, "work_volume": {"known_macs": MACS}}
    ]
    CP.apply_gate(
        rows,
        [_capsule("derived_at_preflight", name="slow")],
        array_rows=ARRAY_ROWS,
        array_cols=ARRAY_COLS,
        phase=GP.PHASE_REPORT,
    )
    assert rows[0]["status"] == "pass"
    assert rows[0][CP.PLANE]["status"] == CP.STATUS_OVER, "it still DECIDED; it just did not block"


def test_a_row_off_the_timing_axis_is_not_judged_and_not_padded() -> None:
    """A capsule whose acceptance block names no timing rung owes no cycle count, so a verdict about
    it would be padding -- and a structural gate on result shape refuses padded rows."""
    rows = [{"capsule": "off", "status": "pass", "tiers": {"L3": {"cycles": 10**9}}}]
    judged = CP.apply_gate(rows, [{"name": "off"}], array_rows=ARRAY_ROWS, array_cols=ARRAY_COLS)
    assert judged == [] and CP.PLANE not in rows[0] and rows[0]["status"] == "pass"


def test_the_grader_calls_the_enforcing_entry_point_not_the_reporting_one() -> None:
    """THE CALLER, which is what was actually missing.

    Every assertion above tests :func:`~merlin.perf.cost_plane.apply_gate` directly, and all of them
    passed for months while `gate_phases.yaml` read ``cost_plane: fail`` and no submission could fail
    on cost -- because :mod:`merlin.targetgen.capsule_grade` called ``assess`` instead, recording the
    verdict in the score entry and never touching ``status``. A mechanism with a perfect unit test and
    the wrong caller is indistinguishable from one that does not work, and that is the failure mode
    this repo keeps paying for.

    Checked STRUCTURALLY over the module's AST rather than by searching its text: a substring search
    would be satisfied by the word appearing in a comment, and this file's whole subject is the
    difference between a thing being mentioned and a thing being called.
    """
    import ast
    import inspect

    from merlin.targetgen import capsule_grade

    tree = ast.parse(inspect.getsource(capsule_grade))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        # the alias `capsule_grade` imports the plane under
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_COST_PLANE"
    }
    assert "apply_gate" in called, (
        "capsule_grade does not call cost_plane.apply_gate, so the declared phase cannot reach a "
        "row's status: a submission a hundred times slower than the machine needs grades exactly "
        "like one that is not, whatever gate_phases.yaml says"
    )
    assert "assess" not in called, (
        "capsule_grade still calls cost_plane.assess. Two verdicts for one row can disagree the "
        "moment either call threads an argument differently, and only apply_gate's reaches status"
    )
