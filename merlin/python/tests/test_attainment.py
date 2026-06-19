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
    (rd / "results.yaml").write_text(yaml.safe_dump(
        {"workload": "matmul_f32_64x64x64", "instruction_histogram": {}, "cycles": 2000}))
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
