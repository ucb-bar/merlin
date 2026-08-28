"""R6-enrich: framework attainment join (ceiling vs ours), honest not-measured handling."""
from __future__ import annotations

import json

import yaml

from merlin.kernels import attainment as at


def _setup(tmp_path):
    ceil = tmp_path / "ceiling.jsonl"
    ceil.write_text(json.dumps({"op": "matmul", "dtype": "f32", "M": 64, "N": 64, "K": 64,
                                "target": "spike", "cycles": 1000}) + "\n")
    rd = tmp_path / "runs" / "hand_v0_matmul_f32_64x64x64"
    rd.mkdir(parents=True)
    # The shape a REAL run record has (verified against out/runs/rvv_verify/*/results.yaml): the
    # top-level ``target`` is the BACKEND FAMILY, which declares the measurement authority, and each
    # entry of ``measurement`` carries the SUBSTRATE label, which is the join axis. The fixture used
    # to carry a bare ``cycles`` with neither, and the old code guessed the substrate was "spike"
    # unless a ``vlen`` field happened to be present -- so the test passed on a guess.
    (rd / "results.yaml").write_text(yaml.safe_dump(
        {"workload": "matmul_f32_64x64x64", "instruction_histogram": {}, "target": "rvv",
         "measurement": [{"target": "spike", "cycles": 2000, "cycle_accurate": False}]}))
    return ceil, tmp_path / "runs"


def test_attainment_computed(tmp_path):
    ceil, runs = _setup(tmp_path)
    res = at.compute(ceil, runs)
    m = next(r for r in res if r.attainment is not None)
    assert m.ceiling_cycles == 1000 and m.our_cycles == 2000
    assert m.attainment == 0.5            # expert is 2x faster -> 0.5 attainment


def test_ceiling_not_measured_is_honest(tmp_path):
    _, runs = _setup(tmp_path)
    # no ceiling file -> attainment None with reason, never fabricated
    res = at.compute(tmp_path / "missing.jsonl", runs)
    assert all(r.attainment is None and r.reason == "ceiling_not_measured" for r in res)


def test_key_stable():
    assert at.cca_key("matmul", "f32", (64, 64, 64), "spike") == "matmul|f32|64x64x64|spike"


def test_a_run_that_does_not_say_which_substrate_yields_no_number(tmp_path):
    """THE NEGATIVE CASE: a record with a bare ``cycles`` and no substrate is UNKNOWN, not "spike".

    Two substrates can each emit ``cycles`` while only one is authoritative and the other is a
    timer-derived estimate, so a guess here silently produces an attainment ratio against the wrong
    device. A missing ratio is visibly missing; a wrong one gets cited.
    """
    ceil = tmp_path / "ceiling.jsonl"
    ceil.write_text(json.dumps({"op": "matmul", "dtype": "f32", "M": 64, "N": 64, "K": 64,
                                "target": "spike", "cycles": 1000}) + "\n")
    rd = tmp_path / "runs" / "hand_v0_matmul_f32_64x64x64"
    rd.mkdir(parents=True)
    (rd / "results.yaml").write_text(yaml.safe_dump(
        {"workload": "matmul_f32_64x64x64", "instruction_histogram": {}, "cycles": 2000}))
    res = at.compute(ceil, tmp_path / "runs")
    assert all(r.attainment is None for r in res)
    assert any(r.reason == "ours_not_measured" for r in res), [r.reason for r in res]


def test_a_non_authoritative_substrate_is_not_borrowed(tmp_path):
    """NEGATIVE CASE: rvv declares spike authoritative for cycles. A record carrying only k1 cycles
    contributes nothing, rather than having k1's rdtime-derived estimate stand in for spike's."""
    ceil = tmp_path / "ceiling.jsonl"
    ceil.write_text(json.dumps({"op": "matmul", "dtype": "f32", "M": 64, "N": 64, "K": 64,
                                "target": "spike", "cycles": 1000}) + "\n")
    rd = tmp_path / "runs" / "hand_v0_matmul_f32_64x64x64"
    rd.mkdir(parents=True)
    (rd / "results.yaml").write_text(yaml.safe_dump(
        {"workload": "matmul_f32_64x64x64", "instruction_histogram": {}, "target": "rvv",
         "measurement": [{"target": "k1", "cycles": 2000}]}))
    res = at.compute(ceil, tmp_path / "runs")
    assert all(r.attainment is None for r in res)
