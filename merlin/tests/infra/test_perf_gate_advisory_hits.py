"""The performance gate must disqualify on answer-access VIOLATIONS, not on any audit hit at all.

The QA audit deliberately records two classes of event. A ``blocked_probe`` is a read the mask
REFUSED -- it is evidence the protection worked -- and a ``recon_probe`` returned filenames, not
content; both are advisory and leave the round ``answer_access_clean``. The campaign gate nonetheless
demanded ``audit_hits == []``, under an UNWAIVABLE predicate, so a round the audit itself judged clean
was unwaivably disqualified (measured: round 0 of merlincirct_g4p1_20260905 -- clean, two advisory
hits). These tests pin both directions: advisories pass, violations and malformed evidence fail.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.sandbox.answer_surfaces import AUDIT_ADVISORY_KINDS, AUDIT_VIOLATION_KINDS

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"


@pytest.fixture()
def gate():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import perf_campaign as PC  # noqa: PLC0415
    return PC


def _predicates(gate, rounds, finalize=None):
    """Run the conduct validator over a minimal record and return the predicate names it emitted."""
    summary = {"converged": True, "rounds": rounds,
               "finalize": finalize or {"answer_access_clean": True, "audit_hits": [],
                                        "regrade_all_pass": True}}
    return [d.predicate for d in gate._validate_clean_run({}, summary)]


# ------------------------------------------------------------------------- advisories must PASS
@pytest.mark.parametrize("kind", sorted(AUDIT_ADVISORY_KINDS))
def test_an_advisory_hit_does_not_disqualify_a_clean_round(gate, kind):
    rounds = [{"answer_access_clean": True, "audit_hits": [{"tool": "Bash", "kind": kind,
                                                            "token": "golden.yaml", "input": "cat x"}]}]
    assert "round_answer_access_unclean" not in _predicates(gate, rounds)


def test_the_measured_round_zero_shape_is_not_disqualified(gate):
    """The exact evidence shape the gate rejected: clean verdict, one recon + one blocked probe."""
    rounds = [{"answer_access_clean": True, "audit_hits": [
        {"tool": "Bash", "kind": "recon_probe", "token": "expected_command_buffer"},
        {"tool": "Bash", "kind": "blocked_probe", "token": "capsule_dram"}]}]
    assert "round_answer_access_unclean" not in _predicates(gate, rounds)


def test_advisory_hits_do_not_disqualify_the_finalize_turn(gate):
    finalize = {"answer_access_clean": True, "regrade_all_pass": True,
                "audit_hits": [{"tool": "Bash", "kind": "blocked_probe", "token": "golden.yaml"}]}
    preds = _predicates(gate, [{"answer_access_clean": True, "audit_hits": []}], finalize)
    assert "finalize_answer_access_unclean" not in preds


# ------------------------------------------------------------------------ violations must FAIL
@pytest.mark.parametrize("kind", sorted(AUDIT_VIOLATION_KINDS))
def test_a_violation_hit_still_disqualifies(gate, kind):
    rounds = [{"answer_access_clean": True, "audit_hits": [{"tool": "Bash", "kind": kind,
                                                            "token": "golden.yaml"}]}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_a_violation_disqualifies_even_next_to_advisories(gate):
    rounds = [{"answer_access_clean": True, "audit_hits": [
        {"kind": "blocked_probe"}, {"kind": "path_read", "token": "golden.yaml"}]}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_an_unclean_verdict_still_disqualifies_with_no_hits_recorded(gate):
    """``answer_access_clean`` is still an independent signal -- it is not superseded by the hits."""
    rounds = [{"answer_access_clean": False, "audit_hits": []}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_a_violation_disqualifies_the_finalize_turn(gate):
    finalize = {"answer_access_clean": True, "regrade_all_pass": True,
                "audit_hits": [{"kind": "oracle_use", "token": "merlin.runtime.reference"}]}
    preds = _predicates(gate, [{"answer_access_clean": True, "audit_hits": []}], finalize)
    assert "finalize_answer_access_unclean" in preds


# ------------------------------------------------------------------------------- FAIL CLOSED
@pytest.mark.parametrize("hits", [None, "", {}, 0, "path_read"])
def test_malformed_audit_hits_still_disqualify(gate, hits):
    """A round whose evidence is missing or not a list proves nothing and must not pass."""
    rounds = [{"answer_access_clean": True, "audit_hits": hits}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_a_malformed_round_row_still_disqualifies(gate):
    assert "round_answer_access_unclean" in _predicates(gate, ["not a row"])


def test_an_unrecognised_hit_kind_counts_as_a_violation(gate):
    """A hit kind nobody has declared advisory is disqualifying until someone deliberately does."""
    rounds = [{"answer_access_clean": True, "audit_hits": [{"kind": "some_future_kind"}]}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_a_hit_without_a_kind_counts_as_a_violation(gate):
    rounds = [{"answer_access_clean": True, "audit_hits": [{"tool": "Bash", "token": "golden.yaml"}]}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_a_non_mapping_hit_counts_as_a_violation(gate):
    rounds = [{"answer_access_clean": True, "audit_hits": ["blocked_probe"]}]
    assert "round_answer_access_unclean" in _predicates(gate, rounds)


def test_the_predicate_remains_unwaivable(gate):
    """Fixing WHEN it fires must not make the integrity predicate waivable."""
    assert "round_answer_access_unclean" in gate.UNWAIVABLE
    assert "finalize_answer_access_unclean" in gate.UNWAIVABLE
    assert "round_answer_access_unclean" not in gate._WAIVABLE_PREDICATES


def test_the_two_hit_classes_are_disjoint(gate):
    assert not (AUDIT_ADVISORY_KINDS & AUDIT_VIOLATION_KINDS)


def test_the_auditor_and_the_gate_share_one_vocabulary(monkeypatch):
    """The gate must not re-derive the split -- that is how it drifted from the audit in the first
    place. The QA auditor's advisory set IS the declared constant this gate consumes."""
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(
        merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"))
    harness = merlin_dir() / "experiments/capsule_bench/harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    import run_baseline_qa_loop as L  # noqa: PLC0415
    assert L._ADVISORY_KINDS is AUDIT_ADVISORY_KINDS
