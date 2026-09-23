"""Every axis a coverage gate MEASURES must be able to change its verdict.

Two gates measured something, printed it with the same un-ratcheted ``*`` marker as everything else,
and then could not fail on it:

* ``check_conformance_coverage.py`` accumulated four of its five axes into ``bad``. Shape geometry --
  and with it "N% of real contraction MAC work sits in an untested aspect ratio" -- was decoration.
  ``build_tools/scripts/conformance_ratchet.txt`` still has no ``geometry:`` line, because nothing was
  ever forced to record one.
* ``check_claim_set_disjointness.py`` decided its exit from two literal status tuples, and
  ``claim_model_uncaptured`` -- the status its own documented check #2 raises -- was in neither. It was
  computed, printed, and structurally unable to produce a non-zero exit under any flag combination.

Both decisions now live in a pure function so a test can assert them directly instead of paying for a
full multi-target derivation, and so a sixth axis or a new status cannot go missing unnoticed.
"""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- conformance coverage

CC = _load("check_conformance_coverage")

#: THE PRODUCTION TABLE, not a copy of it. An earlier version of this file listed the axes itself, and
#: that list was written when four axes were wired -- so when `host_lane`, `epilogue` and
#: `conv_geometry` joined the gate and were not accumulated, every test here still passed. A test that
#: enumerates the axes it knows about cannot notice an axis nobody told it about, which is the same
#: shape of bug it exists to catch. Reading `CC.AXES` means adding an axis to the gate automatically
#: subjects it to every parametrised assertion below.
#:
#: `cell` is the odd one out and stays separate: it reports dicts under a bare `uncovered`, and its
#: ratchet key carries no axis tag.
_AXES = list(CC.AXES)


def _report(**axes) -> dict:
    """A clean `status: ok` report, with only the named axes carrying a gap."""
    r = {"target": "T", "status": "ok", "uncovered": []}
    for key, _tag in CC.AXES:
        r[key] = {"uncovered": []}
    r.update(axes)
    return r


@pytest.mark.parametrize("key,tag", _AXES)
def test_an_uncovered_gap_on_each_axis_reaches_the_verdict(key, tag):
    """A gap on ONE axis, everything else clean, must produce debt.

    Before the fix `shape_geometry` produced none, so --fail-on-uncovered returned 0 on it.
    """
    debt = CC.uncovered_debt([_report(**{key: {"uncovered": ["probe_gap"]}})], set())
    assert debt == [f"T {tag}:probe_gap"], (
        f"an uncovered {tag} gap did not reach the gate's verdict; --fail-on-uncovered cannot fail "
        f"on this axis. got: {debt}"
    )


def test_the_cell_axis_still_reaches_the_verdict():
    """The original axis, kept honest alongside the tagged ones."""
    assert CC.uncovered_debt([_report(uncovered=[{"cell": "c1"}])], set()) == ["T cell:c1"]


def test_every_axis_the_requirement_measures_is_named_in_the_gate_table():
    """THE TEST THAT MAKES THE REST MEAN SOMETHING.

    Every assertion above is parametrised over ``CC.AXES``, so all of them pass vacuously for an axis
    that is measured and simply absent from that table -- which is exactly how `host_lane`, `epilogue`
    and `conv_geometry` became decoration after `shape_geometry` had already been fixed once, and how
    `scope`, `groups` and `carried_state` were measured without ever being carried out of `audit()`.

    So this asks the requirement itself what it measures, rather than trusting either list. Any
    axis-shaped entry `conformance.uncovered` returns -- a dict carrying `uncovered` or `status` --
    must be named in the gate's table, or a gap on it cannot reach `--fail-on-uncovered` no matter how
    the accumulator is written. Adding an axis to the requirement and forgetting the gate is now a test
    failure that names the axis.

    Called with an empty spec and no corpus, so it is a pure shape question and costs no derivation.
    """
    from merlin.targetgen import conformance

    produced = {
        key
        for key, value in conformance.uncovered({}, []).items()
        if isinstance(value, dict) and ("uncovered" in value or "status" in value)
    }
    named = {key for key, _tag in CC.AXES}
    missing = produced - named
    assert not missing, (
        f"conformance.uncovered() measures {sorted(missing)}, which check_conformance_coverage.AXES "
        f"does not name. A gap on those axes is printed and cannot fail the gate. Add each to AXES "
        f"with its ratchet tag."
    )


def test_the_gate_table_names_no_axis_the_requirement_does_not_measure():
    """The other direction: a stale entry would reserve a ratchet tag nothing can ever produce, and
    would make the coverage test above look broader than it is."""
    from merlin.targetgen import conformance

    produced = {
        key
        for key, value in conformance.uncovered({}, []).items()
        if isinstance(value, dict) and ("uncovered" in value or "status" in value)
    }
    stale = {key for key, _tag in CC.AXES} - produced
    assert not stale, f"check_conformance_coverage.AXES names {sorted(stale)}, which nothing measures"


@pytest.mark.parametrize("key,tag", _AXES)
def test_a_ratcheted_gap_on_each_axis_is_forgiven(key, tag):
    """The ratchet must reach every axis too -- otherwise the axis is un-silenceable, not un-failable."""
    rep = _report(**{key: {"uncovered": ["probe_gap"]}})
    assert CC.uncovered_debt([rep], {f"T {tag}:probe_gap"}) == []


def test_an_unauditable_target_contributes_no_debt():
    """Documented behaviour, pinned so the axis loop cannot quietly start trusting a failed audit."""
    rep = _report(shape_geometry={"uncovered": ["probe_gap"]})
    rep["status"] = "no_contract"
    assert CC.uncovered_debt([rep], set()) == []


# ------------------------------------------------------------------------ claim-set disjointness

CS = _load("check_claim_set_disjointness")

#: Every status `audit()` can put on a row, with the verdict it must map to. `claim_model_uncaptured`
#: is the one that used to map to nothing at all.
_STATUS_VERDICTS = [
    ("circular", "violation"),
    ("overlap", "violation"),
    ("claim_model_uncaptured", "unmeasured"),
    ("no_requirement", "unmeasured"),
    ("contract_unresolved", "unmeasured"),
    ("ok", "clean"),
]


@pytest.mark.parametrize("status,want", _STATUS_VERDICTS)
def test_every_status_the_audit_emits_has_a_verdict(status, want):
    assert CS.verdict_bucket(status) == want, (
        f"status {status!r} does not map to {want!r}; a status that maps to nothing cannot change the "
        f"exit code, which is how check #2 (claim_model_uncaptured) came to be unfailable"
    )


def test_an_unknown_status_is_unmeasured_never_clean():
    """Closed by construction: a status added to audit() later must not arrive as silence."""
    assert CS.verdict_bucket("a_status_nobody_classified_yet") == "unmeasured"


def test_the_status_table_covers_what_the_audit_can_emit():
    """Guard the guard: the classification sets must not drift below what the module can produce."""
    classified = CS._VIOLATION | CS._UNMEASURED | CS._CLEAN
    for status, _ in _STATUS_VERDICTS:
        assert status in classified, f"{status} fell out of the classification sets"
