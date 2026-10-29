"""Coverage enforcement with synthetic worker verdicts; no Lean proof credit."""

import json
from types import SimpleNamespace

import pytest

from merlin.verify.compiler_coverage import CoverageJournal
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.speculative_gate import GateFailure, Job


def setup_job(tmp_path, *, middle=()):
    directory = tmp_path / "0001-actual"
    directory.mkdir()
    params = {"intermediates": list(middle)}
    job = Job("0001-actual", directory, "synthetic", "actual", params)
    request = dict(check=job.check, label=job.label, params=params, directory=str(directory))
    for side, raw in (("before", "A\r\n"), ("after", "B\r\n")):
        path = directory / f"{side}.mlir"
        path.write_bytes(raw.encode())
        request[side] = str(path)
    (directory / "request.json").write_text(json.dumps(request))
    gate = SimpleNamespace(directory=tmp_path, verdicts=[])
    journal = CoverageJournal(SimpleNamespace(gate=gate, passes=frozenset({"actual"})))
    return job, gate, journal


def accepted(job):
    return dict(job=job.id, check=job.check, label=job.label, verdict="accepted")


def test_unrelated_accepted_job_cannot_verify_a_different_boundary(tmp_path):
    job, gate, journal = setup_job(tmp_path)
    gate.verdicts.append(accepted(job))
    with pytest.raises(GateFailure, match="actual queued proof inputs"):
        journal.boundary("actual", "X", "Y", profile_id="test", job=job)


def test_same_id_from_another_gate_cannot_borrow_this_gate_verdict(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    own_job, gate, journal = setup_job(left)
    foreign_job, _, _ = setup_job(right)
    gate.verdicts.append(accepted(own_job))
    with pytest.raises(GateFailure, match="another compiler gate"):
        journal.boundary("actual", "A\r\n", "B\r\n", profile_id="test", job=foreign_job)


def test_exact_whole_chain_is_pending_until_the_actual_job_completes(tmp_path):
    job, gate, journal = setup_job(tmp_path, middle=("middle",))
    journal.boundary("unroll+tile", "A\r\n", "B\r\n", profile_id="test", job=job)
    assert journal.rows[0]["status"] == "pending"
    assert journal.rows[0]["proof_credit"] is False
    gate.verdicts.append(accepted(job))
    document = journal.finish(compiler_completed=True)
    assert document["verified_boundaries"] == 1
    assert document["boundaries"][0]["proof_credit"] is True


@pytest.mark.parametrize("mutation", ["before", "after", "request", "middle", "parameter", "duplicate"])
def test_changed_inputs_or_ambiguous_verdict_cannot_receive_credit(tmp_path, mutation):
    job, gate, journal = setup_job(tmp_path, middle=("middle",))
    journal.boundary("actual", "A\r\n", "B\r\n", profile_id="test", job=job)
    gate.verdicts.append(accepted(job))
    if mutation in ("before", "after"):
        (job.directory / f"{mutation}.mlir").write_text("changed")
    elif mutation == "request":
        with (job.directory / "request.json").open("a") as stream:
            stream.write(" ")
    elif mutation == "middle":
        job.params["intermediates"][0] = "other"
    elif mutation == "parameter":
        job.params["factor"] = 999
    else:
        gate.verdicts.append(accepted(job))
    with pytest.raises(GateFailure):
        journal.finish(compiler_completed=True)
    report = json.loads((tmp_path / "coverage.json").read_text())
    assert report["compiler_completed"] is False
    assert report["verified_boundaries"] == 0
    assert report["boundaries"][0]["proof_credit"] is False


def test_zero_jobs_with_exclusion_is_unverified_and_logged_only_at_finish(tmp_path, caplog):
    _, _, journal = setup_job(tmp_path)
    excluded = SourceOutsideProfile("bounded-v1", "type", "rank three is outside this model")
    journal.boundary("actual", "A", "B", profile_id=excluded.profile_id, exclusion=excluded)
    assert not caplog.records
    report = journal.finish(compiler_completed=True)
    assert report["verified_boundaries"] == 0
    assert report["unverified_boundaries"] == 1
    assert "rank three" in caplog.text
    assert report["boundaries"][0]["proof_credit"] is False


def test_exclusion_cannot_hide_a_proof_job(tmp_path):
    job, _, journal = setup_job(tmp_path)
    with pytest.raises(ValueError, match="both"):
        journal.boundary(
            "actual",
            "A",
            "B",
            profile_id="test",
            job=job,
            exclusion=SourceOutsideProfile("test", "type", "unsupported type"),
        )


def test_incomplete_ordinary_output_cannot_be_reported_as_success(tmp_path):
    _, _, journal = setup_job(tmp_path)
    journal.boundary(
        "actual",
        "A",
        None,
        profile_id="test",
        exclusion=SourceOutsideProfile("test", "type", "unsupported type"),
    )
    with pytest.raises(GateFailure):
        journal.finish(compiler_completed=True)


@pytest.mark.parametrize("verdict", ["not_established", "unsupported", "unavailable"])
def test_supported_job_failures_stay_blocking(tmp_path, verdict):
    job, gate, journal = setup_job(tmp_path)
    journal.boundary("actual", "A\r\n", "B\r\n", profile_id="test", job=job)
    gate.verdicts.append({**accepted(job), "verdict": verdict})
    with pytest.raises(GateFailure):
        journal.finish(compiler_completed=True)
    report = journal.finish(compiler_completed=False)
    assert report["boundaries"][0]["status"] == verdict
    assert report["unverified_boundaries"] == 0
