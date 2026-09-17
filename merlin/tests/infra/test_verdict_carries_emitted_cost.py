"""The verdict must tell the agent what its OWN emitted program costs to execute.

MEASURED on the g3arm gemmini batch: the six capsules that failed the cert tier were the six heaviest
DRAM movers, the top two at 20,592 and 18,624 movement operations against a median of 10 across the
other 84. `SY_geometry_squareish_gemm` moved 540,672 bytes in 18,624 operations; the elaborated-RTL
engine spent its entire 900 s budget on that traffic and was killed. The agent's verdict said only
`tiers: {L3: fail}` with `failure_plane: null` and `failure_detail: null` -- an opaque failure with
nothing to act on -- while the harness had already computed the movement count, written it to
`generated/liveness_report.json` beside the result, and discarded it. The same capsule certifies in
0.022 s when lowered tile-wise.

The field is redaction-safe by construction: statistics of the agent's own program, plus hardware
capacities it is already granted through the ISA facts. These tests pin BOTH halves -- that the signal
arrives, and that nothing answer-bearing rides in with it.
"""

from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))
import qa_check as Q  # noqa: E402

PEAKS = {
    "dram_movements": 18624,
    "dram_unmapped": 0,
    "dram_unknown_provenance": 0,
    "scratchpad_rows_touched": 32,
    "scratchpad_rows_capacity": 16384,
    "accumulator_max_row": 0,
    "accumulator_rows_capacity": 1024,
    "closes_with_fence": True,
}


def _stage(tmp_path, peaks=PEAKS, *, write_report=True):
    cap = tmp_path / "runs" / "suite" / "SY_x"
    (cap / "generated").mkdir(parents=True)
    (cap / "capsule_result.json").write_text(json.dumps({"capsule": "SY_x", "status": "pass"}))
    if write_report:
        (cap / "generated" / "liveness_report.json").write_text(
            json.dumps({"target": "t", "program": "SY_x", "verdict": "unknown", "resource_peaks": peaks})
        )
    return cap / "capsule_result.json"


def test_the_movement_count_reaches_the_verdict(tmp_path):
    got = Q._emitted_cost(_stage(tmp_path))
    assert got is not None
    assert got["dram_movements"] == 18624
    assert got["movements_basis"] == "decoded_instruction_trace", "the basis must be named"


def test_capacities_ride_along_so_the_count_is_interpretable(tmp_path):
    got = Q._emitted_cost(_stage(tmp_path))
    assert got["scratchpad_rows_touched"] == 32
    assert got["scratchpad_rows_capacity"] == 16384


def test_no_answer_bearing_field_can_ride_in(tmp_path):
    """Only a numeric/bool allowlist survives; a golden smuggled into the report is dropped."""
    hostile = dict(PEAKS)
    hostile["expected_outputs"] = [1, 2, 3]
    hostile["golden"] = {"values": [4, 5]}
    hostile["reference_outputs"] = "secret"
    got = Q._emitted_cost(_stage(tmp_path, hostile))
    assert set(got) <= set(PEAKS) | {"movements_basis"}, f"unexpected field survived: {got}"
    for banned in ("expected_outputs", "golden", "reference_outputs"):
        assert banned not in got


def test_a_non_numeric_value_is_dropped(tmp_path):
    got = Q._emitted_cost(_stage(tmp_path, {"dram_movements": "lots", "scratchpad_rows_touched": 4}))
    assert "dram_movements" not in got
    assert got["scratchpad_rows_touched"] == 4


def test_absence_is_absence_not_a_crash(tmp_path):
    """An advisory screen that did not run must never break verdict production."""
    assert Q._emitted_cost(_stage(tmp_path, write_report=False)) is None


def test_an_unreadable_report_is_tolerated(tmp_path):
    cr = _stage(tmp_path)
    (cr.parent / "generated" / "liveness_report.json").write_text("{not json")
    assert Q._emitted_cost(cr) is None


def test_an_empty_peaks_block_yields_nothing(tmp_path):
    assert Q._emitted_cost(_stage(tmp_path, {})) is None
