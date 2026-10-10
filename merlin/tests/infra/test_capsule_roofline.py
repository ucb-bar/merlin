"""Per-member ROOFLINE in the tuning feedback: a derived machine bound for each member's declared work.

The bound is the larger of the array's compute floor (from the RTL facts' array geometry) and the memory
path's movement floor (from the elaborated circuit's read/write widths), both derived; the feedback
cell states where each correct arm sits against it and refutes, never under-reports, a bound beaten.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase2 import corpus_feedback as CF
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2 import feedback_metrics as FM
from merlin_experiments.phase2.contracts import StageGateError

from merlin.perf import capsule_roofline as CR

_REPO = Path(__file__).resolve().parents[3]

_MACHINE = {
    "array_rows": 16,
    "array_cols": 16,
    "read_bytes_per_cycle": 16,
    "write_bytes_per_cycle": 16,
    "basis": {"compute": "test geometry", "movement": "test widths"},
    "unresolved": {},
}


def _matmul(m: int, k: int, n: int, *, out: str = "i32") -> dict:
    return {
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
            {"name": "A0", "role": "input", "shape": [m, k], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W", "output_dtype": out}},
    }


def test_compute_floor_is_block_issue_minimised_over_orientation():
    # 64 streamed rows through one 16-deep block per 16 columns: 4 blocks wide x 64 rows.
    assert CR.compute_floor(64, 16, 64, array_rows=16, array_cols=16)["cycles"] == 256
    # A short stream still pays each block's own entry depth.
    assert CR.compute_floor(1, 64, 16, array_rows=16, array_cols=16)["cycles"] == 64


def test_a_small_contraction_is_movement_bound_and_names_its_limiter():
    reads, writes, _ = FM.declared_capsule_operand_bytes(_matmul(16, 16, 16))
    assert (reads, writes) == (512, 1024)
    doc = FM.capsule_roofline(_matmul(16, 16, 16), _MACHINE)
    assert doc["compute_floor_cycles"] == 16
    assert doc["movement_floor_cycles"] == 64.0
    assert (doc["roofline_cycles"], doc["limiter"], doc["status"]) == (64, "movement", "derived")


def test_a_deep_contraction_is_compute_bound():
    doc = FM.capsule_roofline(_matmul(256, 256, 256), _MACHINE)
    assert doc["limiter"] == "compute"
    assert doc["roofline_cycles"] == doc["compute_floor_cycles"] == 16 * 16 * 256


def test_an_underivable_memory_path_drops_the_movement_term_with_its_reason():
    machine = dict(_MACHINE, read_bytes_per_cycle=None, write_bytes_per_cycle=None)
    machine["unresolved"] = {"movement": "no elaborated circuit was named"}
    doc = FM.capsule_roofline(_matmul(16, 16, 16), machine)
    assert doc["movement_floor_cycles"] is None and doc["limiter"] == "compute"
    assert "movement" in doc["unresolved"]


def test_no_machine_means_no_roofline_never_a_zero_one():
    cell = FM.roofline_cell(_matmul(16, 16, 16), None, baseline_cycles=100, candidate_cycles=90)
    assert cell["status"] == "unknown" and cell["roofline_cycles"] is None
    assert cell["baseline_over_roofline"] is None and cell["candidate_over_roofline"] is None


def test_positions_are_measured_over_bound_and_a_beaten_bound_is_refuted():
    cell = FM.roofline_cell(_matmul(16, 16, 16), _MACHINE, baseline_cycles=128, candidate_cycles=96)
    assert cell["baseline_over_roofline"] == 2.0 and cell["candidate_over_roofline"] == 1.5
    beaten = FM.roofline_cell(_matmul(16, 16, 16), _MACHINE, baseline_cycles=128, candidate_cycles=40)
    assert beaten["status"] == "refuted"
    assert beaten["baseline_over_roofline"] is None and beaten["candidate_over_roofline"] is None


def test_validator_refuses_a_position_below_the_bound_or_against_an_unknown_one():
    good = FM.roofline_cell(_matmul(16, 16, 16), _MACHINE, baseline_cycles=128, candidate_cycles=96)
    CF._validate_roofline_cell(good, index=0, measured=True)
    below = dict(good, candidate_over_roofline=0.5)
    with pytest.raises(StageGateError):
        CF._validate_roofline_cell(below, index=0, measured=True)
    unknown = FM.roofline_cell(_matmul(16, 16, 16), None)
    with pytest.raises(StageGateError):
        CF._validate_roofline_cell(dict(unknown, baseline_over_roofline=2.0), index=0, measured=True)
    with pytest.raises(StageGateError):
        CF._validate_roofline_cell(dict(good, extra=1), index=0, measured=True)
    # An unmeasured cell may carry the bound (it is the workload's) but never a position.
    with pytest.raises(StageGateError):
        CF._validate_roofline_cell(good, index=0, measured=False)


def test_the_roofline_block_never_trips_the_redaction_scan():
    cell = FM.roofline_cell(_matmul(16, 16, 16), _MACHINE, baseline_cycles=128, candidate_cycles=96)
    encoded = json.dumps({"roofline": cell, "machine": FM.roofline_machine_summary(_MACHINE)}).lower()
    for forbidden in ('"golden', '"output', '"shape', '"verilator', '"elf', '"path'):
        assert forbidden not in encoded


def test_every_priced_corpus_member_has_a_derived_roofline_consistent_with_its_price():
    descriptors = []
    for path in sorted(glob.glob(str(_REPO / "merlin/contract/capsules/**/capsule.yaml"), recursive=True)):
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(document, dict):
            descriptors.append(document)
    priced = 0
    for descriptor in descriptors:
        macs, _ = FM.declared_capsule_macs(descriptor)
        contractions, _ = FM.declared_capsule_contractions(descriptor)
        if macs is None:
            assert contractions is None
            continue
        priced += 1
        assert sum(m * k * n for m, k, n in contractions) == macs
        doc = FM.capsule_roofline(descriptor, _MACHINE)
        assert doc["status"] == "derived" and doc["compute_floor_cycles"] >= macs / 256
    assert priced > 0


def test_machine_bounds_read_geometry_from_facts_and_widths_from_the_pinned_elaboration(tmp_path):
    cmd = (
        "{ flip ready : UInt<1>, valid : UInt<1>, bits : { inst : { funct : UInt<7>, rs2 : UInt<5>, "
        "opcode : UInt<7>}, rs1 : UInt<64>}}"
    )
    edge = (
        "mem : { a : { flip ready : UInt<1>, valid : UInt<1>, bits : { opcode : UInt<3>, address : UInt<32>, "
        "mask : UInt<16>, data : UInt<128>}}, flip d : { flip ready : UInt<1>, valid : UInt<1>, bits : "
        "{ opcode : UInt<3>, data : UInt<256>}}}"
    )
    fir = tmp_path / "model.fir"
    fir.write_text(
        "FIRRTL version 4.0.0\ncircuit Top :\n"
        "  module Accel : @[gen/x.scala 1:1]\n"
        f"    output auto : {{ {edge}}} @[x.scala 1:1]\n"
        f"    output io : {{ flip cmd : {cmd}, busy : UInt<1>}} @[x.scala 1:1]\n"
        "    wire w : UInt<1>\n"
    )
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"facts": {"arrays": [{"name": "grid", "rows": 8, "cols": 4}]}}))

    class _Certificate:
        pins = {"gsim_firrtl": {"path": str(fir), "sha256": "0" * 64}}

    bounds = DF.derive_machine_bounds(facts, _Certificate(), "any_target")
    assert (bounds["array_rows"], bounds["array_cols"]) == (8, 4)
    assert (bounds["read_bytes_per_cycle"], bounds["write_bytes_per_cycle"]) == (32, 16)
    assert bounds["unresolved"] == {}

    class _Unpinned:
        pins = {}

    unpinned = DF.derive_machine_bounds(facts, _Unpinned(), "any_target")
    assert unpinned["read_bytes_per_cycle"] is None and "movement" in unpinned["unresolved"]


def test_a_result_reshaping_epilogue_leaves_the_write_volume_undeclared_never_overcharged():
    pooled = _matmul(64, 16, 16, out="i8")
    pooled["operation"]["attributes"]["epilogue"] = ["bias_add", "relu", "maxpool"]
    reads, writes, basis = FM.declared_capsule_operand_bytes(pooled)
    assert reads == 64 * 16 + 16 * 16 and writes is None and "maxpool" in basis
    plain = _matmul(64, 16, 16, out="i8")
    plain["operation"]["attributes"]["epilogue"] = ["bias_add", "acc_scale", "relu"]
    assert FM.declared_capsule_operand_bytes(plain)[1] == 64 * 16
    doc = FM.capsule_roofline(pooled, _MACHINE)
    assert doc["movement_floor_cycles"] == (64 * 16 + 16 * 16) / 16 and "write_bytes" in doc["unresolved"]


def test_a_full_sweep_carries_a_validated_roofline_per_cell(tmp_path):
    """evaluate() end to end with an injected executor: every cell carries the roofline block, the
    document passes the redacted schema, and positions are each correct arm's cycles over the bound."""
    from types import SimpleNamespace

    sha = "a" * 64
    members = [
        SimpleNamespace(family="PM", capsule="PM00", descriptor=_matmul(16, 16, 16), source_dir=tmp_path),
        SimpleNamespace(family="PM", capsule="PM01", descriptor=_matmul(256, 256, 256), source_dir=tmp_path),
    ]
    decision = SimpleNamespace(certificate_sha256=sha, selected_engine="gsim", use_gsim=True)
    cycles = {
        ("baseline", "PM00"): 200,
        ("candidate", "PM00"): 128,
        ("baseline", "PM01"): 200_000,
        ("candidate", "PM01"): 100_000,
    }

    def executor(*, arm, member, **_):
        return {
            "measurement": {
                "per_sim": {
                    "spike": {"correct": True},
                    "gsim": {"correct": True, "cycles": cycles[(arm, member.capsule)]},
                },
                "gsim_qualification": {
                    "admitted": True,
                    "decision": {"selected_engine": "gsim", "certificate_sha256": sha},
                },
                "numeric": "pass",
                "status": "pass",
            }
        }

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text("x: 1\n")
    feedback = DF.DevelopmentGsimFeedback(
        SimpleNamespace(sha256=sha),
        SimpleNamespace(capsules=members, capsules_sha256=sha),
        tmp_path,
        sha,
        SimpleNamespace(target="t"),
        {},
        tmp_path / "work",
        {(m.family, m.capsule): decision for m in members},
        peak_macs_per_cycle=256,
        peak_basis="test",
        machine_bounds=_MACHINE,
        executor=executor,
    )
    document = feedback.evaluate(candidate, round_index=0, call_index=1, timeout_s=600)
    assert CF.validate_redacted_feedback(document) == document
    by_capsule = {row["capsule"]: row["roofline"] for row in document["cells"]}
    assert by_capsule["PM00"]["limiter"] == "movement" and by_capsule["PM00"]["roofline_cycles"] == 64
    assert by_capsule["PM00"]["candidate_over_roofline"] == 2.0
    assert by_capsule["PM01"]["limiter"] == "compute"
    assert by_capsule["PM01"]["baseline_over_roofline"] == round(200_000 / 65_536, 4)
    assert document["summary"]["roofline_machine"]["read_bytes_per_cycle"] == 16
    # No program image was built by the injected executor, so what each arm executed is UNKNOWN, with why.
    executed = document["cells"][0]["executed_commands"]
    assert executed["candidate"]["status"] == "unknown" and "program images" in executed["candidate"]["why"]


def test_the_free_command_buffer_screen_states_a_movement_floor_when_widths_are_derived():
    from merlin_experiments.phase2 import emission_diagnostics as ED

    buffer = {
        "tensors": {
            "A": {"shape": [16, 16], "dtype": "i8", "role": "input"},
            "W": {"shape": [16, 16], "dtype": "i8", "role": "weight"},
            "Y": {"shape": [16, 16], "dtype": "i32", "role": "output"},
        },
        "commands": [{"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "W", "dst": "Y"}}],
    }
    bound = ED._demand_lower_bound(buffer, 256, _MACHINE)
    assert bound["movement_floor_cycles"] == 64.0 and bound["limiter"] == "movement"
    unbound = ED._demand_lower_bound(buffer, 256)
    assert unbound["movement_floor_cycles"] is None and unbound["limiter"] is None
