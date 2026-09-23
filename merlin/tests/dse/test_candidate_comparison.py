"""The derivation that gives ``candidate_decision.compare`` phase 2's own evidence.

``compare`` encodes two measured incidents and refuses both by name. This file checks that the
ADAPTER hands it inputs on which those refusals actually fire, because a perfect instrument fed
dishonest axes decides exactly as badly as no instrument at all:

* two arms that emitted the SAME program cannot produce a speedup verdict, whatever the metrics say;
* an axis that did not move cannot carry the verdict -- the 1.642x host-instruction reading across
  byte-identical traffic must not come back as a win;
* an axis nothing observed is NOT declared as "did not move": ``accelerator_time`` and
  ``wall_cycles`` are never declared from an emission pair, because that analysis does not execute.

Every case here is built from the shape the phase-2 host analysis really produces (``arms``,
``target_artifact_activity``, ``machine_artifact_activity``), so a change to those shapes that
silently stops the decision from seeing them turns this file red.
"""

from __future__ import annotations

import pytest

from merlin.perf import candidate_decision as CD
from merlin.perf.candidate_comparison import (
    EMISSION_DECISION_SCHEMA,
    FUNCTIONAL_SIMULATOR_BLIND_AXES,
    decide_emitted_pair,
    decide_measured_totals,
)

BASE_LOWERED = "1" * 64
CAND_LOWERED = "2" * 64
BASE_BUFFER = "3" * 64
CAND_BUFFER = "4" * 64


def _emission(*, lowered=(BASE_LOWERED, CAND_LOWERED), buffers=(BASE_BUFFER, CAND_BUFFER)):
    return {
        "baseline_lowered_sha256": lowered[0],
        "candidate_lowered_sha256": lowered[1],
        "lowered_identical": lowered[0] == lowered[1],
        "baseline_command_buffer_sha256": buffers[0],
        "candidate_command_buffer_sha256": buffers[1],
        "command_buffer_identical": buffers[0] == buffers[1],
    }


def _arm(*, bytes_moved=None, lower_bound=False, issued=None):
    return {
        "status": "emitted",
        "movement": {
            "known_bytes": bytes_moved,
            "exact_bytes": None if lower_bound else bytes_moved,
            "is_lower_bound": lower_bound,
        },
    }


def _activity(issued):
    return {"status": "decoded", "issued": dict(issued)}


def _machine(sites, source_sha256):
    return {
        "status": "compiled",
        "source_sha256": source_sha256,
        "instruction_sites": {"schema": "encoded_instruction_sites_v1", "total": sites},
    }


ISSUED_BASE = {
    "movement_instructions": 40,
    "compute_instructions": 10,
    "configuration_instructions": 4,
    "loop_descriptor_instructions": 2,
    "synchronization_instructions": 3,
    "dma_instructions": 8,
}


def _diagnostics(*, base_bytes, cand_bytes, base_sites=None, cand_sites=None, base_issued=None, cand_issued=None):
    out = {
        "arms": {"baseline": _arm(bytes_moved=base_bytes), "candidate": _arm(bytes_moved=cand_bytes)},
        "target_artifact_activity": {
            "baseline": _activity(base_issued or ISSUED_BASE),
            "candidate": _activity(cand_issued or ISSUED_BASE),
        },
    }
    if base_sites is not None and cand_sites is not None:
        out["machine_artifact_activity"] = {
            "baseline": _machine(base_sites, BASE_LOWERED),
            "candidate": _machine(cand_sites, CAND_LOWERED),
        }
    return out


class TestTheSameProgramEndsTheComparison:
    def test_an_identical_command_buffer_is_IDENTICAL_EMISSION_not_a_speedup(self):
        """What the last campaign shipped a performance hypothesis on top of.

        MUTATION: stop binding the digests into `CandidateFacts` (pass empty strings) and this
        fails -- the traffic metric then reads 2x and the decision comes back BETTER on two arms
        that emitted the same bytes.
        """
        got = decide_emitted_pair(
            _diagnostics(base_bytes=169.52e6, cand_bytes=84.0e6),
            _emission(lowered=(BASE_LOWERED, CAND_LOWERED), buffers=(BASE_BUFFER, BASE_BUFFER)),
        )
        assert got["verdict"] == CD.IDENTICAL_EMISSION
        assert got["verdict"] != CD.BETTER
        assert "same emitted program" in got["why"]
        assert got["decided_by"] == ""

    def test_an_identical_lowered_module_ends_it_too(self):
        got = decide_emitted_pair(
            _diagnostics(base_bytes=1000.0, cand_bytes=10.0),
            _emission(lowered=(BASE_LOWERED, BASE_LOWERED)),
        )
        assert got["verdict"] == CD.IDENTICAL_EMISSION and "lowered module" in got["why"]

    def test_a_wholly_identical_emission_never_yields_a_ratio_verdict(self):
        got = decide_emitted_pair(
            _diagnostics(base_bytes=1000.0, cand_bytes=1.0, base_sites=900.0, cand_sites=1.0),
            _emission(lowered=(BASE_LOWERED, BASE_LOWERED), buffers=(BASE_BUFFER, BASE_BUFFER)),
        )
        assert got["verdict"] == CD.IDENTICAL_EMISSION
        assert got["verdict"] not in (CD.BETTER, CD.WORSE)


class TestAnAxisThatDidNotMoveCannotCarryTheVerdict:
    def test_the_1_642x_on_byte_identical_traffic_is_not_a_win(self):
        """THE MEASURED INCIDENT. Host instruction count improved 1.642x while traffic was
        byte-identical on every arm (169.52 MiB), and the axis under optimization was data movement.

        MUTATION: drop `blind_to` from the machine-site metric, or stop declaring the traffic axis,
        and this goes BETTER at 1.642x -- exactly the reading that was quoted.
        """
        got = decide_emitted_pair(
            _diagnostics(
                base_bytes=169.52e6,
                cand_bytes=169.52e6,
                base_sites=914191598.0,
                cand_sites=556761305.0,
            ),
            _emission(),
        )
        assert got["verdict"] != CD.BETTER
        assert "traffic" not in got["moved_axes"], "an axis that did not move may not be in the verdict"
        assert "host_work" in got["moved_axes"]

    def test_the_traffic_instrument_is_named_as_unable_to_see_the_host_change(self):
        got = decide_emitted_pair(
            _diagnostics(base_bytes=169.52e6, cand_bytes=169.52e6, base_sites=900.0, cand_sites=500.0),
            _emission(),
        )
        metrics = {row["name"]: row for row in got["decision"]["metrics"]}
        assert "host_work" in metrics["declared_movement_bytes"]["blind_to"]
        assert "traffic" in metrics["machine_instruction_sites"]["blind_to"]

    def test_nothing_moving_at_all_is_NO_EFFECT_rather_than_a_ratio(self):
        got = decide_emitted_pair(
            _diagnostics(base_bytes=1000.0, cand_bytes=1000.0, base_sites=50.0, cand_sites=50.0),
            _emission(lowered=(BASE_LOWERED, BASE_LOWERED[:-1] + "0"), buffers=(BASE_BUFFER, BASE_BUFFER[:-1] + "0")),
        )
        assert got["verdict"] == CD.NO_EFFECT

    def test_a_real_traffic_reduction_is_BETTER_and_names_the_axis(self):
        got = decide_emitted_pair(
            _diagnostics(base_bytes=169.52e6, cand_bytes=84.0e6),
            _emission(),
        )
        assert got["verdict"] == CD.BETTER
        assert got["decided_by"] == "command_buffer_movement_volume"
        assert "traffic" in got["moved_axes"]

    def test_two_instruments_disagreeing_is_UNKNOWN_never_the_flattering_one(self):
        got = decide_emitted_pair(
            _diagnostics(base_bytes=100.0, cand_bytes=400.0, base_sites=900.0, cand_sites=100.0),
            _emission(),
        )
        assert got["verdict"] == CD.UNKNOWN and got["decided_by"] == ""
        assert "the instruments disagree" in got["why"]


class TestAnUnobservedAxisIsNotADidNotMove:
    def test_the_two_timing_axes_are_never_declared_from_an_emission_pair(self):
        """MUTATION: declare `wall_cycles` with `moved=False` here and this fails -- an unexecuted
        pair would then read as a change that provably did not affect the elapsed window."""
        got = decide_emitted_pair(_diagnostics(base_bytes=100.0, cand_bytes=50.0), _emission())
        declared = {axis["name"] for axis in got["decision"]["axes"]}
        assert "wall_cycles" not in declared and "accelerator_time" not in declared
        undeclared = {row["axis"]: row["reason"] for row in got["undeclared_axes"]}
        assert "wall_cycles" in undeclared and "accelerator_time" in undeclared
        assert "without executing either" in undeclared["wall_cycles"]

    def test_a_lower_bound_movement_reading_is_not_compared(self):
        diagnostics = _diagnostics(base_bytes=100.0, cand_bytes=50.0)
        diagnostics["arms"]["candidate"] = _arm(bytes_moved=50.0, lower_bound=True)
        got = decide_emitted_pair(diagnostics, _emission())
        undeclared = {row["axis"] for row in got["undeclared_axes"]}
        assert "traffic" in undeclared
        assert any(row["metric"] == "declared_movement_bytes" for row in got["unavailable_instruments"])

    def test_an_unbound_machine_audit_is_refused_rather_than_read(self):
        """A stale audit reads exactly like a fresh one; only the source binding separates them."""
        diagnostics = _diagnostics(base_bytes=100.0, cand_bytes=100.0, base_sites=900.0, cand_sites=100.0)
        diagnostics["machine_artifact_activity"]["candidate"]["source_sha256"] = "9" * 64
        got = decide_emitted_pair(diagnostics, _emission())
        assert "host_work" not in got["moved_axes"]
        assert any(row["axis"] == "host_work" for row in got["undeclared_axes"])

    def test_an_incommensurate_instruction_site_total_is_refused(self):
        diagnostics = _diagnostics(base_bytes=100.0, cand_bytes=100.0, base_sites=900.0, cand_sites=100.0)
        diagnostics["machine_artifact_activity"]["baseline"]["instruction_sites"]["schema"] = "older_total_v0"
        got = decide_emitted_pair(diagnostics, _emission())
        assert "host_work" not in got["moved_axes"]

    def test_a_moved_axis_with_no_instrument_is_BLIND_not_NO_EFFECT(self):
        """An unmeasured change and an ineffective one are different findings."""
        diagnostics = {
            "arms": {"baseline": {"status": "declined"}, "candidate": {"status": "declined"}},
            "target_artifact_activity": {
                "baseline": _activity(ISSUED_BASE),
                "candidate": _activity({**ISSUED_BASE, "compute_instructions": 99}),
            },
        }
        got = decide_emitted_pair(diagnostics, _emission())
        assert got["verdict"] == CD.BLIND
        assert "unmeasured change, NOT an ineffective one" in got["why"]
        assert "offload" in got["moved_axes"]

    def test_no_derivable_axis_at_all_is_UNKNOWN_and_calls_nothing(self):
        got = decide_emitted_pair({}, {})
        assert got["verdict"] == CD.UNKNOWN and got["decision"] is None
        assert got["schema"] == EMISSION_DECISION_SCHEMA


class TestTheMeasuredTotalsDecision:
    def test_equal_totals_are_NO_EFFECT_not_a_1x_win(self):
        got = decide_measured_totals(instrument="gsim", baseline_cycles=1000, candidate_cycles=1000)
        assert got["verdict"] == CD.NO_EFFECT
        assert got["moved_axes"] == []

    def test_a_real_reduction_is_BETTER_on_the_certified_engine(self):
        got = decide_measured_totals(instrument="gsim", baseline_cycles=1000, candidate_cycles=500)
        assert got["verdict"] == CD.BETTER and got["decided_by"] == "gsim"

    def test_an_engine_blind_to_the_moved_axis_cannot_rank(self):
        """Never rank on a simulator that prices every accelerator command at one cycle."""
        got = decide_measured_totals(
            instrument="functional_isa_simulator",
            baseline_cycles=1000,
            candidate_cycles=500,
            axis="accelerator_time",
            blind_to=FUNCTIONAL_SIMULATOR_BLIND_AXES,
        )
        assert got["verdict"] == CD.BLIND and got["verdict"] != CD.BETTER

    def test_an_unnamed_engine_is_refused(self):
        got = decide_measured_totals(instrument="", baseline_cycles=1000, candidate_cycles=500)
        assert got["verdict"] == CD.UNKNOWN and got["decision"] is None

    def test_an_absent_total_is_not_a_zero_and_not_a_tie(self):
        got = decide_measured_totals(instrument="gsim", baseline_cycles=None, candidate_cycles=500)
        assert got["verdict"] == CD.UNKNOWN
        assert "not a zero and not a tie" in got["why"]

    def test_every_verdict_is_in_the_declared_vocabulary(self):
        cases = [
            decide_emitted_pair(_diagnostics(base_bytes=100.0, cand_bytes=50.0), _emission()),
            decide_emitted_pair(_diagnostics(base_bytes=100.0, cand_bytes=100.0), _emission()),
            decide_emitted_pair({}, {}),
            decide_measured_totals(instrument="gsim", baseline_cycles=10, candidate_cycles=20),
        ]
        for got in cases:
            assert got["verdict"] in CD.VERDICTS


def test_the_decision_never_claims_a_cycle_result_from_an_emission_pair() -> None:
    got = decide_emitted_pair(_diagnostics(base_bytes=169.52e6, cand_bytes=84.0e6), _emission())
    assert "never a cycle claim" in got["scope"]
    assert got["decision"]["decided_by"] != "gsim"


def test_a_pytest_approx_ratio_is_carried_for_the_reader() -> None:
    got = decide_emitted_pair(_diagnostics(base_bytes=169.52e6, cand_bytes=84.0e6), _emission())
    metric = {row["name"]: row for row in got["decision"]["metrics"]}["declared_movement_bytes"]
    assert metric["ratio"] == pytest.approx(169.52e6 / 84.0e6)
