"""Mutating stage files must not let a proof of different bytes certify the recorded source."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from merlin.verify.lean import check, export

ADD = """func.func @forward(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  return %r : i32
}
"""
SUB = ADD.replace("arith.addi", "arith.subi")
UNSUPPORTED = """func.func private @external() -> i32
func.func @forward() -> i32 {
  %r = func.call @external() : () -> i32
  return %r : i32
}
"""
CONSTANT = """func.func @forward() -> i32 {
  %r = arith.constant 1 : i32
  return %r : i32
}
"""


@pytest.fixture
def standalone_round_trip(lean_checker, monkeypatch):
    # Exercise real Lean checking and MLIR parsing without requiring the torch frontend. Production
    # still requires the authoritative exporter; only these race tests select the secondary reader.
    opt = os.environ.get("MERLIN_VERIFY_MLIR_OPT") or shutil.which("mlir-opt")
    if not opt or not Path(opt).is_file():
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("standalone MLIR parser unavailable")
        pytest.skip("standalone MLIR parser unavailable")

    def trip(path, workdir, **kwargs):
        return export.round_trip(path, workdir, lean_exe=lean_checker, backend="xdsl", mlir_opt=Path(opt))

    # The old implementation exported again after round-tripping. Keep that code path executable
    # when running this regression against it, so a false verified result is an assertion failure.
    monkeypatch.setattr(
        check, "export_file", lambda src, dst, **kw: export.export_file(src, dst, backend="xdsl"), raising=False
    )
    return trip


def test_stage_rejects_the_original_file_race(tmp_path, monkeypatch, standalone_round_trip):
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text(ADD)
    after.write_text(SUB)
    (tmp_path / "meta.json").write_text(json.dumps({"boundary": "race"}))
    changed = False

    def race(path, workdir, **kwargs):
        nonlocal changed
        if not changed:
            before.write_text(SUB)
            changed = True
        return standalone_round_trip(path, workdir, **kwargs)

    monkeypatch.setattr(check, "round_trip", race)
    receipt = check.check_stage(tmp_path, m2m_python=Path(sys.executable))
    before.write_text(ADD)
    assert receipt["verdict"] == "abstracted"
    assert receipt["before"]["sha256"] == check._sha(before)


def test_support_rejects_the_original_file_race(tmp_path, monkeypatch, standalone_round_trip):
    program = tmp_path / "program.mlir"
    program.write_text(UNSUPPORTED)

    def race(path, workdir, **kwargs):
        program.write_text(CONSTANT)
        return standalone_round_trip(path, workdir, **kwargs)

    monkeypatch.setattr(check, "round_trip", race)
    receipt = check.check_support(program, "forward", tmp_path / "support", m2m_python=Path(sys.executable))
    program.write_text(UNSUPPORTED)
    assert receipt["verdict"] == "abstracted"
    assert receipt["subject"]["sha256"] == check._sha(program)


def test_checks_retain_separate_exports(tmp_path, monkeypatch, standalone_round_trip):
    for stem in ("before", "after"):
        (tmp_path / f"{stem}.mlir").write_text(ADD)
    (tmp_path / "meta.json").write_text(json.dumps({"boundary": "identity"}))
    used = []

    def trip(path, workdir, **kwargs):
        used.append(workdir)
        return standalone_round_trip(path, workdir, **kwargs)

    monkeypatch.setattr(check, "round_trip", trip)
    for _ in range(2):
        assert check.check_stage(tmp_path, m2m_python=Path(sys.executable))["verdict"] == "verified"
    assert used[0].parent != used[2].parent
    for workdir in used:
        assert (workdir / "export.json").is_file()


def test_input_changed_during_a_valid_check_is_not_published_as_verified(tmp_path, monkeypatch, standalone_round_trip):
    before = tmp_path / "before.mlir"
    before.write_text(ADD)
    (tmp_path / "after.mlir").write_text(ADD)
    (tmp_path / "meta.json").write_text(json.dumps({"boundary": "identity"}))

    def race(path, workdir, **kwargs):
        before.write_text(SUB)
        return standalone_round_trip(path, workdir, **kwargs)

    monkeypatch.setattr(check, "round_trip", race)
    receipt = check.check_stage(tmp_path, m2m_python=Path(sys.executable))
    assert receipt["verdict"] == "unmeasured"
    assert "changed or missing" in receipt["reason"]


def test_failed_publication_preserves_the_previous_result(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"verdict": "abstracted"}))
    replace = Path.replace

    def fail_publication(path, target):
        if target == receipt:
            raise OSError("publication interrupted")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="publication interrupted"):
        check._atomic_json(receipt, {"verdict": "verified"})
    assert json.loads(receipt.read_text()) == {"verdict": "abstracted"}
    assert list(tmp_path.iterdir()) == [receipt]


@pytest.fixture
def checked_chain(tmp_path, monkeypatch, standalone_round_trip):
    monkeypatch.setattr(check, "round_trip", standalone_round_trip)

    def make(after=ADD, bits=64):
        stage = tmp_path / "stage"
        stage.mkdir()
        (stage / "before.mlir").write_text(ADD)
        (stage / "after.mlir").write_text(after)
        (stage / "meta.json").write_text(json.dumps({"boundary": "identity"}))
        boundary = check.check_stage(stage, m2m_python=Path(sys.executable))
        support = tmp_path / "support"
        receipt = check.check_support(
            stage / "before.mlir", "forward", support, m2m_python=Path(sys.executable), index_bits=bits
        )
        assert receipt["verdict"] == "verified", receipt.get("reason")
        assert receipt["index_bits"] == bits
        return stage, support / "receipt.json", boundary

    return make


@pytest.mark.parametrize("bits", [32, 64])
def test_chain_replays_genuine_claims_without_overwriting_evidence(tmp_path, checked_chain, bits):
    import jsonschema

    stage, support, boundary = checked_chain(bits=bits)
    assert boundary["verdict"] == "verified"
    originals = {p: p.read_bytes() for p in (stage / "receipt.json", support)}
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "verified", result.get("reason")
    assert result["replayed"]
    schema = check.repo_root() / "merlin" / "contract" / "schemas"
    jsonschema.validate(result, json.loads((schema / "lean_chain.schema.json").read_text()))
    jsonschema.validate(json.loads(support.read_bytes()), json.loads((schema / "lean_receipt.schema.json").read_text()))
    assert all(p.read_bytes() == data for p, data in originals.items())


def test_chain_requires_replay(tmp_path, checked_chain):
    stage, support, _ = checked_chain()
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json")
    assert result["verdict"] == "abstracted" and not result["replayed"]
    assert "requires m2m_python" in result["reason"]


def test_chain_rejects_forged_verified_verdict(tmp_path, checked_chain):
    stage, support, boundary = checked_chain(after=SUB)
    assert boundary["verdict"] == "abstracted"
    boundary["verdict"] = "verified"
    (stage / "receipt.json").write_text(json.dumps(boundary))
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "abstracted"
    assert "checker replay returned abstracted" in result["reason"]


@pytest.mark.parametrize("field", ["checker", "theorem", "exporter", "axioms", "assumptions", "domain", "index_bits"])
def test_chain_rejects_changed_claim_metadata(tmp_path, checked_chain, field):
    stage, support, _ = checked_chain()
    path = support if field in ("domain", "index_bits") else stage / "receipt.json"
    receipt = json.loads(path.read_bytes())
    if field == "index_bits":
        receipt[field] = 32
    elif field in ("axioms", "assumptions"):
        receipt[field] = []
    elif field == "exporter":
        receipt[field]["executable_sha256"] = "0" * 64
    else:
        receipt[field] = "forged"
    path.write_text(json.dumps(receipt))
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "abstracted"
    assert "does not match the replay" in result["reason"]


def test_chain_rejects_legacy_support_without_recorded_width(tmp_path, checked_chain):
    stage, support, _ = checked_chain()
    receipt = json.loads(support.read_bytes())
    del receipt["index_bits"]
    support.write_text(json.dumps(receipt))
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "abstracted"
    assert "regenerate it" in result["reason"]


def test_chain_binds_replay_to_stage_files(tmp_path, checked_chain):
    stage, support, _ = checked_chain()
    # The stored reference remains genuine, but the actual pipeline stage now subtracts.
    original = tmp_path / "original.mlir"
    original.write_text(ADD)
    receipt = json.loads((stage / "receipt.json").read_bytes())
    receipt["after"] = check._file_ref(original)
    (stage / "receipt.json").write_text(json.dumps(receipt))
    (stage / "after.mlir").write_text(SUB)
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "abstracted"
    assert "after bytes do not match" in result["reason"]


def test_chain_rejects_receipt_replaced_during_replay(tmp_path, monkeypatch, checked_chain):
    stage, support, _ = checked_chain()
    path = stage / "receipt.json"
    digest = check._sha(path)
    replay = check._replay

    def race(*args, **kwargs):
        errors = replay(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return errors

    monkeypatch.setattr(check, "_replay", race)
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "abstracted"
    assert "receipt changed during composition" in result["reason"]
    assert result["links"][0]["receipt_sha256"] == digest


def test_chain_rechecks_actual_stage_after_replay(tmp_path, monkeypatch, checked_chain):
    stage, support, _ = checked_chain()
    original = tmp_path / "original.mlir"
    original.write_text(ADD)
    path = stage / "receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["after"] = check._file_ref(original)
    path.write_text(json.dumps(receipt))
    replay = check._replay

    def race(*args, **kwargs):
        errors = replay(*args, **kwargs)
        (stage / "after.mlir").write_text(SUB)
        return errors

    monkeypatch.setattr(check, "_replay", race)
    result = check.compose_chain([stage], support, "forward", tmp_path / "chain.json", m2m_python=Path(sys.executable))
    assert result["verdict"] == "abstracted"
    assert "live after changed during composition" in result["reason"]
