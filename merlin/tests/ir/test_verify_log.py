"""A verification verdict is a measurement, and the gate that reads it must fail closed.

The capsule bench grades outcomes: a passing capsule says *this program* compiled correctly *this
time* on *this target*. It does not say the pass that tiled it is correct. The static
(lit/FileCheck) and formal (SMT refinement) layers answer that second question, and these tests
cover the join between those verdicts and the evidence system that already existed — the invocation
log beside it — rather than the layers themselves.

Three properties, each of which the repo has been bitten by the absence of:

* a verdict survives the round trip with the fields that make it citable (which pass, which
  requirement class, which target, by which method, with the evidence);
* a REFUTATION fails hard and no ratchet line can forgive it — accepted debt is for evidence we do
  not have yet, never for a disproof we do;
* a missing log exits 2 ("cannot decide"), never 0. A check that could not run and reported success
  is the failure mode this repo keeps re-encountering; `--fail-on-unverified` must spell it the same
  way `--fail-on-dead` already does.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from merlin.common.paths import repo_root
from merlin.xdsl_dialects.lowering import passes as PS

GATE = repo_root() / "build_tools" / "scripts" / "check_pass_obligations.py"

# Whatever the catalog happens to hold, not a name pinned here: a test that hardcodes a pass name
# starts failing for the wrong reason the day the pass is renamed, and stops testing the gate.
CATALOG = PS.catalog()
SUBJECT = CATALOG[0]
OTHER = CATALOG[-1]


def _gate():
    spec = importlib.util.spec_from_file_location("_check_pass_obligations", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_ambient_logs(monkeypatch):
    """An outer audit or a live capsule run must not change what these tests measure."""
    for var in (PS.PASS_LOG_ENV, PS.PASS_LOG_CAPSULE_ENV, PS.PASS_LOG_REQUIREMENTS_ENV,
                PS.VERIFY_LOG_ENV):
        monkeypatch.delenv(var, raising=False)


def _verdict(monkeypatch, path, item, *, verdict, method=None, target="target-under-test",
             capsule="capsule-under-test", evidence=None):
    """Write one verdict through the real recorder — never by hand-rolling the JSON."""
    monkeypatch.setenv(PS.VERIFY_LOG_ENV, str(path))
    PS.record_verification(item.name,
                           requirement_class=(item.required_by or (PS.UNKNOWN,))[0],
                           method=method or PS.METHOD_FILECHECK, verdict=verdict,
                           target=target, capsule=capsule,
                           evidence=evidence or {}, provenance={"tool": "test"})
    monkeypatch.delenv(PS.VERIFY_LOG_ENV, raising=False)


def _reach(monkeypatch, path, item):
    """Make ``item`` measurably REACHED: an install record plus one effective invocation.

    The invocation goes through the real recorder, so the attribution path (`pass_run_context` ->
    requirement classes -> `exercised`) is the one the gate reads. The install record is written
    directly rather than by `install_pass_recorder()` on purpose: wrapping the real entry points
    imports every pass module and turns an unrelated import failure into a `not_instrumented` status,
    which would make these tests flaky about something they are not testing. That whole-pipeline
    wrapping is covered over a real capture by `test_pass_exercise_evidence.py`; here the subject is
    the ledger.
    """
    monkeypatch.setenv(PS.PASS_LOG_ENV, str(path))
    PS._append({"kind": PS._LOG_INSTALL, "passes": {p.name: "instrumented" for p in CATALOG},
                "pid": 0, "t": 0.0})
    with PS.pass_run_context("capsule-under-test", item.required_by):
        PS.record_invocation(item.name, effect=PS.EFFECT_CHANGED, evidence={"subject_before": None})
    monkeypatch.delenv(PS.PASS_LOG_ENV, raising=False)


# --- the log itself -------------------------------------------------------------------------------

def test_verdict_round_trips_with_the_fields_that_make_it_citable(tmp_path, monkeypatch):
    log = tmp_path / "verify.jsonl"
    _verdict(monkeypatch, log, SUBJECT, verdict=PS.VERDICT_VERIFIED, method=PS.METHOD_SMT,
             evidence={"shape": {"m": 2, "k": 2, "n": 2}, "solver_status": "unsat"})

    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["kind"] == PS._LOG_VERDICT
    assert rec["pass"] == SUBJECT.name
    assert rec["capsule"] == "capsule-under-test"
    assert rec["requirement_class"] == SUBJECT.required_by[0]
    assert rec["target"] == "target-under-test"
    assert rec["method"] == PS.METHOD_SMT
    assert rec["verdict"] == PS.VERDICT_VERIFIED
    assert rec["evidence"]["solver_status"] == "unsat"
    assert rec["provenance"] == {"tool": "test"}

    report = PS.verification_report(CATALOG, logs=[log])
    row = report["per_pass"][SUBJECT.name]
    assert row["status"] == PS.VERDICT_VERIFIED
    assert row["verdicts"] == {PS.VERDICT_VERIFIED: 1}
    assert row["methods"] == [PS.METHOD_SMT]
    assert row["targets"] == ["target-under-test"]
    assert row["requirement_classes"] == [SUBJECT.required_by[0]]
    # Every other catalogued pass is `unverified`, not silently absent: the gate's whole job is to
    # name the passes nobody checked, and it can only do that if they appear in the report.
    assert {report["per_pass"][p.name]["status"] for p in CATALOG if p is not SUBJECT} \
        == {"unverified"}


def test_a_missing_target_is_recorded_unknown_rather_than_defaulted(tmp_path, monkeypatch):
    """The formal layer validates a target-independent plane; it must not borrow a target name."""
    monkeypatch.setenv(PS.VERIFY_LOG_ENV, str(tmp_path / "verify.jsonl"))
    PS.record_verification(SUBJECT.name, requirement_class=SUBJECT.required_by[0],
                           method=PS.METHOD_SMT, verdict=PS.VERDICT_VERIFIED)
    rec = json.loads((tmp_path / "verify.jsonl").read_text(encoding="utf-8").strip())
    assert rec["target"] == PS.UNKNOWN


def test_a_verdict_outside_the_vocabulary_raises_even_with_recording_off():
    """A typo that only surfaces once someone enables the log is a typo that ships."""
    assert PS.verify_log_path() is None
    with pytest.raises(ValueError):
        PS.record_verification(SUBJECT.name, requirement_class=SUBJECT.required_by[0],
                               method=PS.METHOD_SMT, verdict="passed")
    with pytest.raises(ValueError):
        PS.record_verification(SUBJECT.name, requirement_class=SUBJECT.required_by[0],
                               method="eyeball", verdict=PS.VERDICT_VERIFIED)


def test_solver_unknown_is_not_a_pass():
    """A timeout that reads as a proof is a check that could not run reporting success."""
    assert PS.solver_verdict("unsat") == PS.VERDICT_VERIFIED
    assert PS.solver_verdict("sat") == PS.VERDICT_REFUTED
    assert PS.solver_verdict("unknown") == PS.VERDICT_UNMEASURED


def test_refutation_dominates_and_keeps_its_counterexample(tmp_path, monkeypatch):
    """One disproof is not outvoted by any number of passing checks."""
    log = tmp_path / "verify.jsonl"
    _verdict(monkeypatch, log, SUBJECT, verdict=PS.VERDICT_VERIFIED)
    _verdict(monkeypatch, log, SUBJECT, verdict=PS.VERDICT_VERIFIED, method=PS.METHOD_SMT)
    _verdict(monkeypatch, log, SUBJECT, verdict=PS.VERDICT_REFUTED, method=PS.METHOD_SMT,
             evidence={"model": "a0 = 1, w0 = 0"})

    row = PS.verification_report(CATALOG, logs=[log])["per_pass"][SUBJECT.name]
    assert row["status"] == PS.VERDICT_REFUTED
    assert row["verdicts"] == {PS.VERDICT_VERIFIED: 2, PS.VERDICT_REFUTED: 1}
    # A refutation nobody can reproduce gets argued away rather than fixed.
    assert any(e["verdict"] == PS.VERDICT_REFUTED and e["evidence"]["model"]
               for e in row["evidence"])


def test_an_abstraction_is_recorded_but_is_not_verification(tmp_path, monkeypatch):
    """A check that could not be grounded is reported, never counted as a pass."""
    log = tmp_path / "verify.jsonl"
    _verdict(monkeypatch, log, SUBJECT, verdict=PS.VERDICT_ABSTRACTED,
             evidence={"reason": "the literal is not derivable from this target's facts"})
    assert PS.verification_report(CATALOG, logs=[log])["per_pass"][SUBJECT.name]["status"] \
        == PS.VERDICT_ABSTRACTED


def test_a_verdict_against_an_uncatalogued_name_is_surfaced(tmp_path, monkeypatch):
    """Evidence aimed at a renamed pass stops counting, which looks exactly like never running it."""
    log = tmp_path / "verify.jsonl"
    monkeypatch.setenv(PS.VERIFY_LOG_ENV, str(log))
    PS.record_verification("merlin-pass-that-was-renamed", requirement_class=PS.HOST_SEAM,
                           method=PS.METHOD_FILECHECK, verdict=PS.VERDICT_VERIFIED)
    monkeypatch.delenv(PS.VERIFY_LOG_ENV, raising=False)
    assert PS.verification_report(CATALOG, logs=[log])["unknown_passes"] == \
        ["merlin-pass-that-was-renamed"]


# --- the gate -------------------------------------------------------------------------------------

def test_absent_verify_log_cannot_decide_and_exits_two(capsys):
    """Mirrors --fail-on-dead verbatim: unmeasured is never spelled the same way as clean."""
    assert _gate().main(["--fail-on-unverified"]) == 2
    assert "CANNOT DECIDE" in capsys.readouterr().err


def test_a_verify_log_without_an_invocation_log_still_cannot_decide(tmp_path, monkeypatch, capsys):
    """The axis is "REACHED but unverified"; without the invocation log nothing is known reached."""
    log = tmp_path / "verify.jsonl"
    _verdict(monkeypatch, log, SUBJECT, verdict=PS.VERDICT_VERIFIED)
    assert _gate().main(["--fail-on-unverified", "--verify-log", str(log)]) == 2
    assert "invocation log" in capsys.readouterr().err


def test_refuted_fails_hard_even_when_ratcheted(tmp_path, monkeypatch, capsys):
    """A ratchet accepts evidence we do not have yet; it may never accept a disproof."""
    verify = tmp_path / "verify.jsonl"
    _verdict(monkeypatch, verify, SUBJECT, verdict=PS.VERDICT_REFUTED, method=PS.METHOD_SMT,
             evidence={"model": "a0 = 1, w0 = 0"})
    gate = _gate()

    # Every ratchet spelling that could plausibly be tried, including the one the unverified axis
    # legitimately uses for this same pass. None of them may silence a refutation.
    ratchet = tmp_path / "ratchet.txt"
    ratchet.write_text("\n".join([
        gate._debt(SUBJECT.name, "no-static-or-formal-verdict", "verification"),
        gate._debt(SUBJECT.name, PS.VERDICT_REFUTED, "verification"),
        gate._debt(SUBJECT.name, "no-capsule-runs-it", "exercise"),
    ]) + "\n", encoding="utf-8")

    rc = gate.main(["--verify-log", str(verify), "--ratchet", str(ratchet)])
    assert rc == 1, "a refutation must fail with no flag and no ratchet relief"
    assert "REFUTED" in capsys.readouterr().err

    # And it is not reported as ratchetable debt at all: the finding carries no debt key.
    rep = gate.audit([], [verify])
    found = gate.findings(rep, gate._load_ratchet(ratchet))
    assert [it["pass"] for it in found["refuted"]] == [SUBJECT.name]
    assert "debt" not in found["refuted"][0]
    assert "ratcheted" not in found["refuted"][0]


def test_reached_but_unverified_is_reported_and_is_ratchetable(tmp_path, monkeypatch, capsys):
    """Absent evidence — unlike a refutation — is exactly what a ratchet is for."""
    invoke, verify = tmp_path / "pass.jsonl", tmp_path / "verify.jsonl"
    _reach(monkeypatch, invoke, SUBJECT)
    _reach(monkeypatch, invoke, OTHER)
    _verdict(monkeypatch, verify, SUBJECT, verdict=PS.VERDICT_VERIFIED)
    gate = _gate()
    args = ["--fail-on-unverified", "--log", str(invoke), "--verify-log", str(verify)]

    assert gate.main(args) == 1
    err = capsys.readouterr().err
    assert "no static or formal layer has reached a verdict" in err

    found = gate.findings(gate.audit([invoke], [verify]), set())
    assert [it["pass"] for it in found["unverified"]] == [OTHER.name]
    assert found["unverified"][0]["verification"] == "unverified"
    assert found["unverified"][0]["exercise"].startswith("exercised")

    ratchet = tmp_path / "ratchet.txt"
    ratchet.write_text(gate._debt(OTHER.name, "no-static-or-formal-verdict", "verification") + "\n",
                       encoding="utf-8")
    assert gate.main(args + ["--ratchet", str(ratchet)]) == 0


def test_a_pass_nothing_reaches_is_not_charged_with_being_unverified(tmp_path, monkeypatch):
    """`dead` and `unverified` are different defects; charging a dead pass with both is noise."""
    invoke, verify = tmp_path / "pass.jsonl", tmp_path / "verify.jsonl"
    _reach(monkeypatch, invoke, SUBJECT)
    _verdict(monkeypatch, verify, SUBJECT, verdict=PS.VERDICT_VERIFIED)
    gate = _gate()
    found = gate.findings(gate.audit([invoke], [verify]), set())
    unreached = {p.name for p in CATALOG} - {SUBJECT.name}
    assert unreached, "the fixture must leave at least one pass unreached for this to mean anything"
    assert {it["pass"] for it in found["unverified"]}.isdisjoint(unreached)
    assert {it["pass"] for it in found["dead"]} == unreached


def test_the_new_axis_appears_in_json(tmp_path, monkeypatch, capsys):
    invoke, verify = tmp_path / "pass.jsonl", tmp_path / "verify.jsonl"
    _reach(monkeypatch, invoke, SUBJECT)
    _reach(monkeypatch, invoke, OTHER)
    _verdict(monkeypatch, verify, SUBJECT, verdict=PS.VERDICT_VERIFIED, method=PS.METHOD_SMT)
    _gate().main(["--json", "--log", str(invoke), "--verify-log", str(verify)])
    doc = json.loads(capsys.readouterr().out)

    assert doc["findings"]["unverified"] and doc["findings"]["refuted"] == []
    assert doc["findings"]["unverified"][0]["pass"] == OTHER.name
    report = doc["report"]
    assert report["verify_measured"] is True
    assert report["verify_logs_read"] == [str(verify)]
    assert report["verdicts_vocabulary"] == list(PS.VERDICTS)
    row = next(r for r in report["passes"] if r["name"] == SUBJECT.name)
    assert row["verification"] == PS.VERDICT_VERIFIED
    assert row["verify_methods"] == [PS.METHOD_SMT]


def test_json_says_unmeasured_rather_than_clean_when_no_verify_log_is_read(capsys):
    _gate().main(["--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["report"]["verify_measured"] is False
    assert doc["report"]["verify_logs_read"] == []
    # The load-bearing assertion: with nothing measured, no pass is claimed verified and none is
    # charged as unverified. An empty `unverified` list here is ignorance, and the exit code says so.
    assert doc["findings"]["unverified"] == []
    assert {r["verification"] for r in doc["report"]["passes"]} == {"unmeasured"}
