"""A capsule may decline the mesh demand, but not silently, and not on work the hardware admits.

`semantic.must_accelerate: true` demands a capsule's work reach the accelerator. Flipping it to
`false` is the cheapest way to make a failing capsule pass, and a regeneration does it by accident:
the generator emits its own default for `semantic`, the hand-authored assertion is not in its
vocabulary, and the field decays. This repo already paid for that once, when regeneration deleted
`pass_requirements` from the only two capsules that carried it.

Measured 2026-09-02, and the reason this test exists: six HAND-AUTHORED microscaling capsules went
from `must_accelerate: true, fallback_allowed: false` to `must_accelerate: false, eligible: auto` on
a target whose capability contract admits contraction at mxfp4, mxfp6 AND mxfp8. The work is
eligible, so the weaker value was not a judgement about the hardware. Three further hidden capsules
were weakened the same way and are gitignored, so no git diff could see them.

WHY AN INVARIANT AND NOT A DIFF. Diffing history catches only recent weakening and calls every
legitimate authoring change a violation. The property that actually holds is derived from the target:
if the contract admits a capsule's (family, dtype) then the accelerator can take that work, and a
capsule declining to demand it owes a `not_asserted_reason`. On a cell the contract does NOT admit,
`must_accelerate: false` is the correct answer and demanding otherwise would be the bug.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root


def _gate():
    p = repo_root() / "build_tools" / "scripts" / "check_mesh_assertion_not_weakened.py"
    spec = importlib.util.spec_from_file_location("_mesh_gate", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mesh_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_gate_examines_a_real_population():
    """A gate that checked nothing would pass forever."""
    gate = _gate()
    rep = gate.audit()
    assert rep["n_admitted_capsules_checked"] > 50, (
        f"only {rep['n_admitted_capsules_checked']} capsules sit on an admitted cell; this test "
        f"established almost nothing")


def test_every_reported_capsule_really_declines_admitted_work():
    """No false positives: each finding must be on an admitted cell and lack a stated reason."""
    from merlin.targetgen import conformance as CF

    gate = _gate()
    rep = gate.audit()
    for row in rep["weakened"]:
        adm, why = CF.admitted_with_reason(row["target"])
        assert why == "resolved", f"{row['capsule']}: {why}"
        # ONE SPELLING AUTHORITY, same as the gate: the manifest says `int8`, a capsule says `i8`.
        # Comparing raw made every int8 cell read as "not admitted" and hid 51 findings.
        dtypes = {CF.capsule_dtype(str(x)) for x in (adm.get(row["family"]) or ())}
        assert row["dtype"] in dtypes, (
            f"{row['capsule']} was reported, but {row['family']}/{row['dtype']} is NOT admitted on "
            f"{row['target']} (admits {sorted(dtypes)}) -- a fallback is correct there")
        assert row["must_accelerate"] is not True


def test_a_stated_reason_excuses_the_decline():
    """The remedy is a reason, not necessarily `true`: some paths genuinely cannot accelerate."""
    gate = _gate()
    rep = gate.audit()
    reported = {r["capsule"] for r in rep["weakened"]}
    # Any capsule carrying not_asserted_reason must NOT be reported.
    import yaml

    from merlin.common.paths import merlin_dir
    excused = 0
    for cy in (merlin_dir() / "contract" / "capsules").rglob("capsule.yaml"):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        sem = doc.get("semantic") or {}
        if sem.get("must_accelerate") is True:
            continue
        if doc.get("not_asserted_reason") or sem.get("not_asserted_reason"):
            excused += 1
            assert cy.parent.name not in reported, (
                f"{cy.parent.name} states a reason and must not be reported")
    if not excused:
        pytest.skip("no capsule currently states a not_asserted_reason")


def test_the_hand_authored_set_is_reported_distinctly():
    """The reference set the automation is being built to reproduce deserves its own count.

    A weakening in a generated capsule is a generator bug; a weakening in a hand-authored one also
    destroys the reference it was going to be checked against.
    """
    gate = _gate()
    rep = gate.audit()
    hand = [r for r in rep["weakened"] if r["hand_authored"]]
    for r in hand:
        assert r["source_role"] in ("handauthored_compiler_test", "uplifted_from_bareMetalC")
    # The gate must FAIL on them BY DEFAULT -- the exit code has to reflect the finding without an
    # opt-in flag. `--no-ratchet` is what the accepted-debt list is hiding; with the ratchet applied
    # the same population must be fully accounted for, never silently dropped.
    if hand:
        assert gate.main(["--no-ratchet"]) == 1
        assert gate.main(["--no-ratchet", "--advisory"]) == 0


def test_an_unresolvable_target_establishes_nothing():
    """Reported, never counted as clean -- the recurring failure this repo has paid for."""
    gate = _gate()
    rep = gate.audit()
    assert isinstance(rep["unresolved_targets"], dict)
    if rep["unresolved_targets"]:
        assert gate.main([]) == 2      # by default, not behind a flag
    # And the six corpus CATEGORIES (`isa`, `hidden`, `model`, ...) must never appear here: they are
    # not targets, and reading them as such put a sixth of the corpus in this bucket.
    import yaml

    from merlin.common.paths import merlin_dir
    categories = {d.name for d in (merlin_dir() / "contract" / "capsules").iterdir()
                  if d.is_dir() and not (d / "capsule.yaml").exists()}
    declared = set()
    tdir = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"
    for desc in tdir.glob("*/target_experiment.yaml"):
        doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
        declared.add(str(doc.get("target") or desc.parent.name))
        declared.add(desc.parent.name)
    for t in rep["unresolved_targets"]:
        assert t in declared, f"{t!r} is a corpus category, not a target"
