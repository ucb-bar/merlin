"""Every problem the scheduling-IR coverage gate claims to catch, caught.

Two gates in this repo measured something, printed it with the same marker as everything else, and were
structurally unable to fail on it: the number was decoration. The fix both times was to move the decision
into a pure function a test could reach, which is why `verdict` takes its whole world as arguments and
why nothing below needs the real register, a corpus or a simulator.

The register's own header promises that no axis is deleted to improve the ratio, that drift is caught in
BOTH directions, that the two-slice rule counts devices rather than names, and that it says which prior
systems its denominator was drawn from. These are the tests that make those promises checkable rather
than aspirational.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

GATE_PATH = repo_root() / "build_tools" / "scripts" / "check_schedule_ir_coverage.py"


def _gate():
    """Load the gate BY PATH, the way the other gate-regression tests do -- it is a script, not a
    module, and importing it as one would test a different thing than what runs in the hook."""
    spec = importlib.util.spec_from_file_location("_schedule_ir_coverage_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: A roster of FICTIONAL targets. `verdict` takes the roster as a parameter precisely so this suite
#: does not depend on which targets a checkout happens to have -- and naming real ones here would make
#: this file read as being ABOUT them, which it is not: it is about the gate.
ROSTER = {"t_alpha", "t_beta", "t_gamma"}


def _row(**overrides):
    """A row with nothing wrong with it, for a test to break in exactly one way."""
    row = {
        "id": "axis_under_test",
        "axis": "something a schedule must be able to say",
        "exo": "EXO_ABSENT",
        "exo_evidence": "verified by inspection of its grammar",
        "forced_by": ["t_alpha", "t_beta"],
        "generic": True,
        "owner": "sched",
        "surfaced_by": "a design note",
        "surfaced_on": "2026-09-15",
        "blocks": "what does not work today",
    }
    row.update(overrides)
    return row


def test_a_clean_register_is_clean():
    """The control. Without it, every test below passes for a gate that rejects everything."""
    gate = _gate()
    statuses, problems, rc = gate.verdict([_row()], set(), ROSTER)
    assert problems == [] and rc == 0
    assert statuses == {"axis_under_test": "UNEXPRESSED"}


@pytest.mark.parametrize(
    "broken,expect",
    [
        (_row(id=None), "no id"),
        (_row(owner=None), "missing required field 'owner'"),
        (_row(exo="EXO_MAYBE"), "not in ['EXO_HAS', 'EXO_PARTIAL', 'EXO_ABSENT']"),
        (_row(exo_evidence=None), "cites no evidence"),
        (_row(forced_by=["a_target_that_does_not_exist"]), "which no target registry declares"),
        (_row(forced_by=["t_alpha"], generic=True), "two slices need it"),
        (_row(proof="by_construction"), "An obligation is read from the primitive that carries it"),
        (_row(primitive="does_not_exist"), "which is not exported"),
        (_row(blocks=None), "computed UNEXPRESSED but declares no 'blocks'"),
    ],
)
def test_each_problem_class_alone_reaches_the_verdict(broken, expect):
    """One thing wrong, everything else clean, must produce debt.

    Parametrized rather than asserted in a lump so that a class dropped from the accumulation fails its
    own case instead of hiding behind another's.
    """
    gate = _gate()
    _statuses, problems, rc = gate.verdict([broken], set(), ROSTER)
    assert rc == 1, f"a row with {expect!r} produced no debt"
    assert any(expect in p for p in problems), f"expected {expect!r} in {problems}"


def test_a_primitive_with_no_row_is_flagged():
    """The direction a hand-maintained register loses.

    A row naming a vanished primitive is noticed the moment someone reads the register; a primitive
    nobody had to justify is invisible, and is how an IR acquires one per kernel.
    """
    gate = _gate()
    _s, problems, rc = gate.verdict([_row()], {"a_new_primitive"}, ROSTER)
    assert rc == 1
    assert any("is exported but no row declares it" in p for p in problems)


def test_a_row_whose_primitive_exists_is_expressed_and_then_owes_a_test():
    """Status is computed from the tree, so declaring a primitive changes the state and the duties."""
    gate = _gate()
    statuses, problems, _rc = gate.verdict(
        [_row(primitive="reorder")], {"reorder"}, ROSTER, {"reorder": "bounded_check"}
    )
    assert statuses["axis_under_test"] == "EXPRESSED"
    assert any("declares no 'test'" in p for p in problems)


def test_a_primitive_that_declares_no_obligation_is_flagged():
    """A primitive reaching the register unclassified would be coverage nobody reasoned about."""
    gate = _gate()
    _s, problems, rc = gate.verdict([_row(primitive="mystery", test="x")], {"mystery"}, ROSTER, {})
    assert rc == 1
    assert any("declares no obligation" in p for p in problems)


def test_a_status_cannot_be_authored():
    """Writing a status into the register must not change the computed one.

    This is the property that makes the register unfakeable: the only way to claim coverage is to make
    the tree support it.
    """
    gate = _gate()
    statuses, _p, _rc = gate.verdict([_row(status="EXERCISED")], set(), ROSTER)
    assert statuses["axis_under_test"] == "UNEXPRESSED", "an authored status was believed"


def test_the_gate_does_not_gate_on_the_ratio():
    """Gating a coverage score makes deleting the hard axis the rational response.

    Both halves matter: a register where nothing is expressed must still exit 0, and deleting a row must
    LOWER the denominator -- so the score cannot be raised by deletion either.
    """
    gate = _gate()
    rows = [_row(id=f"axis_{i}") for i in range(4)]
    statuses, problems, rc = gate.verdict(rows, set(), ROSTER)
    assert rc == 0 and problems == [], "a register with nothing expressed produced debt"
    assert len(statuses) == 4
    fewer, _p, _rc = gate.verdict(rows[:2], set(), ROSTER)
    assert len(fewer) == 2, "deleting a row did not lower the denominator"


def test_an_unreadable_register_is_a_refusal_not_a_pass():
    """A gate that could not run must never report success -- five recorded instances in this repo."""
    gate = _gate()
    original = gate.REGISTER
    try:
        gate.REGISTER = Path("/nonexistent/schedule_ir_coverage.yaml")
        assert gate.main([]) == 2, "a missing register exited 0 or 1; it must be 'cannot decide'"
    finally:
        gate.REGISTER = original


def test_the_refusal_reaches_a_stop_hook_in_its_own_dialect():
    """The Stop hook signals by JSON on stdout, not by exit code, so the refusal has to be spoken twice."""
    gate = _gate()
    original = gate.REGISTER
    try:
        gate.REGISTER = Path("/nonexistent/schedule_ir_coverage.yaml")
        assert gate.main(["--stop-hook"]) == 0
    finally:
        gate.REGISTER = original


def test_an_empty_roster_is_cannot_decide():
    """An unresolvable roster means forced_by cannot be checked at all; reporting clean would say the
    opposite of what happened."""
    gate = _gate()
    original = gate._roster
    try:
        gate._roster = lambda: set()
        assert gate.main([]) == 2
    finally:
        gate._roster = original


def test_the_real_register_is_consistent_with_the_live_tree():
    """The register as committed, against the tree as it is. Not a smoke test: it is the assertion that
    the two have not drifted, and it is the one that fires when a primitive lands without a row."""
    gate = _gate()
    assert gate.main([]) == 0, "the committed register has drifted from the live tree"


def test_a_key_stated_twice_in_one_entry_is_cannot_decide():
    """The failure this gate could not see: a row whose text and meaning disagree.

    A plain YAML load keeps the last of two identical keys and drops the first without a word, so a row
    reading `primitive: null` immediately above `primitive: [pipeline]` is scored on the second while a
    reviewer reads the first. In a register whose entire purpose is that a claim cannot outrun the code,
    that is the one failure that matters -- and it is invisible to a reader and to the gate alike.
    Measured once in the committed register, on `hazard_resolution_trait`.
    """
    import yaml

    gate = _gate()
    with pytest.raises(yaml.YAMLError, match="stated twice"):
        gate._load_strict(yaml, "entries:\n  - id: a\n    primitive: null\n    primitive: [p]\n")


def test_the_strict_load_still_accepts_the_same_key_in_different_entries():
    """The control: without it the refusal above would pass for a loader that rejects any repeat."""
    import yaml

    gate = _gate()
    doc = gate._load_strict(yaml, "entries:\n  - id: a\n    primitive: [p]\n  - id: b\n    primitive: [q]\n")
    assert [row["primitive"] for row in doc["entries"]] == [["p"], ["q"]]


def test_a_measurement_cannot_be_authored_either():
    """The hole the sibling test above did not cover, and the one that mattered.

    `test_a_status_cannot_be_authored` plants a literal `status:` key -- a door nobody would use. The
    door that was open was `evidence.measured_on`: `status_of` returned EXERCISED whenever that field
    held anything, so the register's strongest claim was conferred by a DATE A HUMAN TYPES, in a file
    whose header promises status is computed and never authored.
    """
    gate = _gate()
    row = _row(primitive=["p"], test="t.py::t", evidence={"corpus": "c", "measured_on": "2026-09-18"})
    statuses, problems, rc = gate.verdict([row], {"p"}, ROSTER)
    assert statuses["axis_under_test"] != "EXERCISED", "a typed date was believed as a measurement"
    assert any("cannot be checked and is not believed" in p for p in problems)
    assert rc != 0


def test_a_row_that_claims_no_measurement_is_not_accused_of_one():
    """The control: without it the test above passes for a gate that flags every row."""
    gate = _gate()
    row = _row(primitive=["p"], test="t.py::t", evidence={"corpus": "c"})
    _statuses, problems, _rc = gate.verdict([row], {"p"}, ROSTER)
    assert not any("measured_on" in p for p in problems)


# -- the two-slice rule counts DEVICES, not names --------------------------------------------------


def test_slice_basis_makes_several_names_one_slice():
    """The correction this mechanism exists for.

    `forced_by` names targets; the rule is about independent slices. An axis was promoted to core
    vocabulary on 2026-09-18 by counting two names that are one machine, and nothing could see it,
    because the gate counted the list. Declaring the grouping makes the claim checkable.
    """
    gate = _gate()
    basis = [{"slice": "one_device", "targets": ["t_alpha", "t_beta"], "why": "two names, one cluster"}]
    _, problems, rc = gate.verdict([_row(generic=True, slice_basis=basis)], set(), ROSTER)
    assert rc == 1
    assert any("1 independent slice(s)" in p for p in problems), problems


def test_a_row_without_slice_basis_still_counts_each_name_once():
    """The control, and the backward-compatibility guarantee: a row that declares no grouping is read
    exactly as before, so adding the mechanism re-classified nothing on its own."""
    gate = _gate()
    _, problems, rc = gate.verdict([_row(generic=True)], set(), ROSTER)
    assert problems == [] and rc == 0


ALL_THREE = ["t_alpha", "t_beta", "t_gamma"]
TWO = ["t_alpha", "t_beta"]


@pytest.mark.parametrize(
    "forced,basis,expect",
    [
        (TWO, [{"slice": "s", "targets": TWO}], "need a 'slice', a 'targets' list and a 'why'"),
        (TWO, [{"targets": TWO, "why": "x"}], "need a 'slice', a 'targets' list and a 'why'"),
        (TWO, [{"slice": "s", "targets": ["t_alpha"], "why": "x"}], "groups 1 target(s)"),
        (TWO, [{"slice": "s", "targets": ["t_alpha", "t_gamma"], "why": "x"}], "which is not in forced_by"),
        (
            ALL_THREE,
            [
                {"slice": "a", "targets": ["t_alpha", "t_beta"], "why": "x"},
                {"slice": "b", "targets": ["t_beta", "t_gamma"], "why": "y"},
            ],
            "is in two slices",
        ),
        (
            ALL_THREE,
            [
                {"slice": "a", "targets": ["t_alpha", "t_beta"], "why": "x"},
                {"slice": "a", "targets": ["t_gamma", "t_alpha"], "why": "y"},
            ],
            "declares 'a' twice",
        ),
    ],
    ids=["no why", "no slice name", "groups one", "not forced", "target in two", "duplicate name"],
)
def test_a_malformed_slice_basis_is_refused(forced, basis, expect):
    """A grouping is a claim, so a claim that does not parse must not quietly reduce the slice count --
    that would make the mechanism a way to weaken the rule rather than to state it honestly.

    Each case carries its OWN forcing list: the "not forced" case needs a slice naming a target the row
    does not force, which is unstateable if every roster target is already in `forced_by`. Sharing one
    list across the table made that case vacuous and it passed for the wrong reason.
    """
    gate = _gate()
    _, problems, _rc = gate.verdict([_row(forced_by=forced, slice_basis=basis)], set(), ROSTER)
    assert any(expect in p for p in problems), problems


@pytest.mark.target("gemmini")
@pytest.mark.target("mx_gemmini")
def test_two_names_for_one_design_must_be_declared_rather_than_merely_omitted():
    """`slice_basis` is a declaration, so on its own it can be left out -- and then two names count as
    two slices again, which is the original defect wearing a new hat. This is the half that can be
    decided without the RTL toolchain.

    The one test here that names REAL targets, and it has to: it asserts what `facts_alias` resolves,
    which is a property of those targets' own declarations. `mx_gemmini` declares `facts_target:
    gemmini` -- same generator, same mesh, same decoder -- so a row forced by both is one slice unless
    it says what makes them independent for that axis.
    """
    gate = _gate()
    roster = ROSTER | {"gemmini", "mx_gemmini"}
    _, problems, rc = gate.verdict([_row(forced_by=["gemmini", "mx_gemmini"])], set(), roster)
    assert rc == 1
    assert any("resolve to the design 'gemmini'" in p for p in problems), problems

    grouped = [{"slice": "gemmini_generator", "targets": ["gemmini", "mx_gemmini"], "why": "declared"}]
    _, ok_problems, _ = gate.verdict(
        [_row(forced_by=["gemmini", "mx_gemmini"], generic=False, slice_basis=grouped)], set(), roster
    )
    assert ok_problems == [], ok_problems


@pytest.mark.target("muon")
@pytest.mark.target("radiance")
def test_the_register_does_not_count_one_simt_cluster_as_two_slices():
    """Pins the 2026-09-19 demotion against the live register, so re-promoting the axis means deleting
    a test that says why it was wrong.

    `muon` and `radiance` are one `RadianceMuonConfig` cluster under two names -- kept apart so neither
    target's kernels are attributed to the other, which is a corpus-hygiene reason and not a second
    device. `facts_alias` cannot see it (they resolve to distinct designs), so nothing automatic will.
    """
    import yaml

    rows = yaml.safe_load((repo_root() / "merlin" / "contract" / "schedule_ir_coverage.yaml").read_text())
    row = next(r for r in rows["entries"] if r["id"] == "thread_identity_and_divergence")
    assert row["generic"] is False, "one SIMT cluster is one slice, so this axis is a target extension"
    grouped = {tuple(sorted(e["targets"])) for e in row.get("slice_basis") or ()}
    assert ("muon", "radiance") in grouped, f"muon and radiance must be declared one slice: {grouped}"


# -- the denominator names which systems it was drawn from -------------------------------------------


def _declared(*entries):
    return list(entries)


def test_a_system_declared_and_not_blanketed_must_be_scored_on_every_row():
    """The teeth. Declaring a system and letting rows omit it widens the denominator on paper and
    changes nothing -- the register would then look checked against three systems and be checked
    against one, which is the exact overstatement adding the column was meant to fix."""
    gate = _gate()
    systems, blanket, problems = gate._declared_systems(_declared({"id": "exo"}, {"id": "halide"}))
    assert problems == [] and blanket == {} and systems == {"exo", "halide"}
    _, verdict_problems, rc = gate.verdict([_row()], set(), ROSTER, systems=systems)
    assert rc == 1
    assert any("unscored for ['halide']" in p for p in verdict_problems), verdict_problems


def test_a_blanket_excuses_the_row_from_scoring_that_system():
    """One admission written once. Thirteen copies of "nobody looked" read as thirteen judgements."""
    gate = _gate()
    systems, blanket, problems = gate._declared_systems(
        _declared({"id": "exo"}, {"id": "halide", "blanket": "UNASSESSED", "why": "no pinned copy"})
    )
    assert problems == [] and systems == {"exo"} and blanket == {"halide": "no pinned copy"}
    _, verdict_problems, rc = gate.verdict([_row()], set(), ROSTER, systems=systems)
    assert verdict_problems == [] and rc == 0


@pytest.mark.parametrize(
    "entry,expect",
    [
        ({"id": "halide", "blanket": "UNASSESSED"}, "needs a 'why'"),
        ({"id": "halide", "blanket": "ABSENT", "why": "x"}, "may only be UNASSESSED"),
        ({"blanket": "UNASSESSED", "why": "x"}, "need an 'id'"),
    ],
    ids=["no why", "blanket is a verdict", "no id"],
)
def test_a_malformed_prior_art_declaration_is_refused(entry, expect):
    """A blanket is the one place a whole column can be waved through, so it is the one place worth
    making hard to write carelessly. `blanket: ABSENT` in particular would assert that a system lacks
    every axis -- a sweeping claim about each row, smuggled in as one line about the system."""
    gate = _gate()
    _systems, _blanket, problems = gate._declared_systems(_declared({"id": "exo"}, entry))
    assert any(expect in p for p in problems), problems


@pytest.mark.parametrize(
    "prior_art,evidence,expect",
    [
        ({"numpy": "HAS"}, {}, "does not declare"),
        ({"halide": "SORT_OF"}, {}, "not one of"),
        ({"halide": "ABSENT"}, {}, "cites no evidence"),
        ({"halide": "UNASSESSED"}, {}, "cites no evidence"),
    ],
    ids=["undeclared system", "bad state", "absent with no evidence", "unassessed with no reason"],
)
def test_a_prior_art_verdict_must_be_declared_and_evidenced(prior_art, evidence, expect):
    """UNASSESSED owes a reason exactly like ABSENT does. "That system cannot do this" and "nobody
    looked" are different claims and both are claims; a state with no reason is neither."""
    gate = _gate()
    row = _row(prior_art=prior_art, prior_art_evidence=evidence)
    _, problems, _rc = gate.verdict([row], set(), ROSTER, systems={"exo", "halide"})
    assert any(expect in p for p in problems), problems


def test_the_legacy_exo_spelling_still_counts_as_a_verdict():
    """Backward compatibility, and the reason the migration needed no row edits: the rows scored `exo`
    in a field named after it long before prior art was a list, and folding that spelling into the map
    means declaring the general mechanism re-classified nothing."""
    gate = _gate()
    states, evidence = gate._prior_art_of(_row(exo="EXO_ABSENT", exo_evidence="checked by grep"))
    assert states["exo"] == "ABSENT" and evidence["exo"] == "checked by grep"
    _, problems, rc = gate.verdict([_row()], set(), ROSTER, systems={"exo"})
    assert problems == [] and rc == 0


def test_the_real_register_declares_what_it_was_and_was_not_scored_against():
    """Pins the honest headline against the live file. 11/16 is a statement about the axes one system's
    absence suggested; two systems a reader would assume were in the denominator are blanketed."""
    import yaml

    doc = yaml.safe_load((repo_root() / "merlin" / "contract" / "schedule_ir_coverage.yaml").read_text())
    gate = _gate()
    systems, blanket, problems = gate._declared_systems(doc.get("prior_art") or [])
    assert problems == []
    assert systems == {"exo"}, f"only exo is verified against a vendored copy: {systems}"
    assert set(blanket) == {"halide", "tile_ir"}, f"the unscored columns must be named: {sorted(blanket)}"
    assert all(why.strip() for why in blanket.values()), "a blanket without a reason is just an omission"
