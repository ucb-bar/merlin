"""A sweep that skips a member must not throw away the members it paid for.

Measured 2026-09-04, and it cost a campaign: the tiered sweep walks members cheapest-first and stops
once a candidate is already losing, emitting a cell for each member it did not pay for. Every such
cell carries null correctness and null cycles, because an absent number is not a zero. The redacted
schema demanded a bool and a positive int on EVERY cell, so it rejected the first unmeasured one and
raised -- after the whole sweep had run. All 38 members completed, eighty minutes were spent, and the
entire result was discarded. Worse, the only thing recorded about the refusal was the exception's
TYPE: the receipt keeps a digest of the message, so recovering the cause meant brute-forcing
exception names against that digest.

So two things are pinned here: an unmeasured cell validates, and it still cannot be mistaken for a
measured one.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_agent_stage as PAS  # noqa: E402

SHA = "a" * 64


class _Member:
    family, capsule = "PX", "PX00_skipped"


def _measured_cell(family: str = "PK", capsule: str = "PK00_k16") -> dict:
    return {
        "family": family, "capsule": capsule,
        "baseline_correct": True, "candidate_correct": True,
        "baseline_gsim_cycles": 300, "candidate_gsim_cycles": 250,
        "candidate_minus_baseline_cycles": -50, "baseline_over_candidate": 300 / 250,
        "comparable": True, "declared_macs": 4096,
        "declared_work_basis": "declared operand shapes", "ideal_cycles_at_peak": 16.0,
        "baseline_utilization": 0.05, "candidate_utilization": 0.06,
        "baseline_share_of_achievable": 0.1, "candidate_share_of_achievable": 0.2,
        "verdict": "improved", "verdict_reason": "fewer cycles than the baseline",
        "measured": True, "skip_reason": None,
    }


def _document(cells: list[dict]) -> dict:
    measured = [c for c in cells if c.get("measured")]
    comparable = sum(1 for c in cells if c.get("comparable"))
    return {
        "schema_version": 1, "kind": "host_owned_tuning_gsim_feedback", "round": 0,
        "invocation": 1, "engine": "gsim",
        "tuning_corpus_sha256": SHA, "candidate_sha256": SHA, "certificate_sha256": SHA,
        "cells": cells,
        "summary": {"members": len(cells), "comparable": comparable,
                    "all_correct": comparable == len(measured),
                    "peak_macs_per_cycle": 256, "peak_basis": "facts-derived",
                    "achievable_macs_per_cycle": 80.0, "achievable_basis": "measured"},
        "stopping": {"status": "continue", "verdicts": [], "queries": 1,
                     "baseline_total_cycles": 300.0, "candidate_total_cycles": 250.0,
                     "best_total_cycles": 250.0, "previous_best_total_cycles": None,
                     "attainable_total_cycles": 100.0, "share_of_attainable": 0.4},
    }


def test_an_unmeasured_cell_has_the_same_field_set_as_a_measured_one():
    """The redacted schema is an EXACT key set, so a differently shaped cell is rejected outright."""
    skipped = PAS._unmeasured_cell(_Member(), reason="the sweep stopped before paying for it")
    assert set(skipped) == set(_measured_cell()), (
        "an unmeasured cell must differ from a measured one in its VALUES, never in its keys")


def test_a_sweep_that_skipped_a_member_still_validates():
    skipped = PAS._unmeasured_cell(_Member(), reason="the sweep stopped before paying for it")
    document = _document([_measured_cell(), skipped])
    PAS.validate_redacted_feedback(document)  # must not raise


def test_an_unmeasured_cell_may_not_claim_a_comparison():
    skipped = PAS._unmeasured_cell(_Member(), reason="stopped early")
    skipped["comparable"] = True
    with pytest.raises(PAS.StageGateError):
        PAS.validate_redacted_feedback(_document([_measured_cell(), skipped]))


def test_an_unmeasured_cell_may_not_carry_a_cycle_count():
    skipped = PAS._unmeasured_cell(_Member(), reason="stopped early")
    skipped["baseline_gsim_cycles"] = 300
    with pytest.raises(PAS.StageGateError):
        PAS.validate_redacted_feedback(_document([_measured_cell(), skipped]))


def test_an_unmeasured_cell_must_give_a_reason():
    skipped = PAS._unmeasured_cell(_Member(), reason="stopped early")
    skipped["skip_reason"] = None
    with pytest.raises(PAS.StageGateError):
        PAS.validate_redacted_feedback(_document([_measured_cell(), skipped]))


def test_all_correct_is_a_claim_about_what_was_measured():
    """A member nobody paid for is not a member that failed."""
    skipped = PAS._unmeasured_cell(_Member(), reason="stopped early")
    document = _document([_measured_cell(), skipped])
    assert document["summary"]["all_correct"] is True
    PAS.validate_redacted_feedback(document)


def test_a_host_refusal_is_recorded_where_the_host_can_read_it(tmp_path):
    """The agent gets a type; the host must get the reason, or an hour buys one word."""
    class _Evaluator:
        work_root = tmp_path

    class _Stage:
        feedback_evaluator = _Evaluator()
        feedback_round = 0

    PAS._record_host_refusal(_Stage(), PAS.StageGateError("cell 7 has invalid GSIM cycles"),
                             round_index=0, call_index=8)
    written = list((tmp_path / "host_refusals").glob("*.txt"))
    assert written, "the refusal left no host-side record"
    text = written[0].read_text()
    assert "invalid GSIM cycles" in text and "StageGateError" in text
    assert "Traceback" in text or "StageGateError" in text


# ---------------------------------------------------------------------------------------------------
# the measurement must survive the agent doing what it was told to do
# ---------------------------------------------------------------------------------------------------
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _capsule(tmp_path):
    source = tmp_path / "capsule"
    source.mkdir()
    (source / "capsule.yaml").write_text("name: PK00_k16\n", encoding="utf-8")
    return SimpleNamespace(
        family="PK", capsule="PK00_k16", source_dir=source,
        source_sha256="b" * 64,
        descriptor={"label": "dev", "required_oracle_tiers": ["L0", "L1", "L2", "L3"]})


def _raw_pass():
    return {"measurement": {
        "status": "pass", "numeric": "pass", "failure": None,
        "per_sim": {"spike": {"correct": True}, "gsim": {"correct": True, "cycles": 400}},
        "gsim_qualification": {"admitted": True,
                               "decision": {"selected_engine": "gsim", "certificate_sha256": SHA}},
    }}


def _evaluator(tmp_path, member):
    decision = SimpleNamespace(selected_engine="gsim", use_gsim=True, eligible=True,
                               admitted=True, certificate_sha256=SHA,
                               final_cycle_authority=False)
    return PAS.DevelopmentGsimFeedback(
        SimpleNamespace(sha256=SHA), SimpleNamespace(capsules=[member], capsules_sha256=SHA),
        Path("."), SHA, SimpleNamespace(target="t"), {}, tmp_path / "work",
        {(member.family, member.capsule): decision},
        peak_macs_per_cycle=256, peak_basis="test",
        achievable_macs_per_cycle=80.0, achievable_basis="test",
        tuning_call_budget=100,
        executor=lambda **kw: _raw_pass())


def test_a_note_written_while_the_sweep_runs_does_not_void_the_measurement(tmp_path):
    """The exact failure that cost a campaign.

    The sweep runs for over an hour while the agent keeps working, and the agent is TOLD to keep
    `iteration_notes.md`. Hashing the live tree at both ends and refusing on a difference threw away
    an eighty-minute measurement in which all 38 members ran and passed, because a note landed while
    the last member finished. Measuring a snapshot keeps the property that matters -- the cycles
    belong to the bytes that produced them -- without making a note fatal.
    """
    candidate = tmp_path / "submission"
    (candidate / "performance").mkdir(parents=True)
    (candidate / "compiler.py").write_text("# the compiler\n", encoding="utf-8")
    member = _capsule(tmp_path)
    evaluator = _evaluator(tmp_path, member)

    def edit_then_run(**kw):
        # what the agent does DURING the sweep: append to its own notes
        (candidate / "performance" / "iteration_notes.md").write_text("tried a thing\n",
                                                                      encoding="utf-8")
        return _raw_pass()

    evaluator.executor = edit_then_run
    document = evaluator.evaluate(candidate, round_index=0, call_index=0, timeout_s=600)
    assert document["cells"][0]["comparable"] is True
    assert document["cells"][0]["candidate_gsim_cycles"] == 400


def test_the_measured_bytes_are_the_ones_the_document_names(tmp_path):
    """The snapshot is the point: `candidate_sha256` must name what actually ran."""
    candidate = tmp_path / "submission2"
    (candidate / "performance").mkdir(parents=True)
    (candidate / "compiler.py").write_text("# the compiler\n", encoding="utf-8")
    member = _capsule(tmp_path)
    evaluator = _evaluator(tmp_path, member)

    seen: list[Path] = []

    def record(**kw):
        # only the candidate arm reads the snapshot; the baseline arm is the frozen submission
        if kw["arm"] == "candidate":
            seen.append(Path(kw["package"]))
        return _raw_pass()

    evaluator.executor = record
    document = evaluator.evaluate(candidate, round_index=0, call_index=1, timeout_s=600)
    assert seen, "no member ran"
    assert all(p != candidate for p in seen), (
        "the runs must read the snapshot, not the tree the agent is still editing")
    assert all(p == seen[0] for p in seen), "every member must read the SAME bytes"
    assert (seen[0] / "compiler.py").read_text() == (candidate / "compiler.py").read_text(), (
        "the snapshot must be a faithful copy of what the agent submitted")
    assert document["candidate_sha256"]
