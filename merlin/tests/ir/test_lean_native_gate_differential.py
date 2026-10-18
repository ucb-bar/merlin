"""Actual held native prefixes need strict chains and independent MLIR execution.

The production suffix is exercised and compared with ordinary emitted bytes.
Native differentials concern the original and held tensor/scalar programs; they
establish empirical model fidelity on these samples, not a T4 lowering proof.
"""

import hashlib
import json
from pathlib import Path

import pytest
from test_lean_reshape_compose import _verify_mlir

from merlin.llvmlower import lower
from merlin.llvmlower.optional_passes import REGISTRY
from merlin.llvmlower.pipeline import PipelineError
from merlin.verify.lean import differential as diff
from merlin.verify.lean import native_gate, plans, preparation, proof_worker, stages
from merlin.verify.lean.handoff import Options

SOURCE = """func.func @forward(%x: i8) -> i32 {
  %extended = arith.extsi %x : i8 to i32
  %c = arith.constant 22 : i32
  %result = arith.addi %extended, %c : i32
  return %result : i32
}"""
INDEX_SOURCE = """func.func @forward(%x: index) -> index { return %x : index }"""


@pytest.fixture(autouse=True)
def ordinary_selection(monkeypatch):
    for key in (
        "MERLIN_PASSES",
        "MERLIN_VEC_RANK",
        "MERLIN_GENERALIZE_BEFORE_FUSE",
        "MERLIN_FUSE_POST",
        "MERLIN_NO_DEALLOC",
    ):
        monkeypatch.delenv(key, raising=False)
    for entry in REGISTRY:
        if entry.switch:
            monkeypatch.delenv(entry.switch, raising=False)
    monkeypatch.setenv("LEAN_NUM_THREADS", "2")


def _lower(tmp_path, tools, monkeypatch, *, bits=64, source=SOURCE, captured=None, gated=True):
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(tools.compiler_python))
    options = Options("forward", bits, tools.compiler_python)
    # These declared module layouts exercise explicit converter domains. The
    # scalar samples use no pointer arithmetic; no target hardware is asserted.
    return lower.lower_model(
        source,
        tmp_path,
        targets=(),
        features=frozenset(),
        static_arena=False,
        index_bits=bits,
        data_layout=f"e-p:{bits}:{bits}",
        shared_pass_verification=options if gated else None,
        shared_pass_preparation=captured,
        verify_native_prefix=gated,
    )


def _report(result):
    path = Path(result.stats["index_lowering"]["shared_pass_native_verification"])
    report = json.loads(path.read_bytes())
    assert report["verdict"] == "verified"
    assert "no native execution proof" in report["trusted_suffix"]
    assert report["held_sha256"] == hashlib.sha256((path.parent / "native/current.mlir").read_bytes()).hexdigest()
    assert json.loads((path.parent / "status.json").read_bytes())["verdict"] == "verified"
    return path, report


def _native_reference(program, samples, expected, work, tools, bits):
    diff.compare(program, samples, work, tools, index_bits=bits)
    assert json.loads((work / "native-results.json").read_bytes())["results"] == expected


@pytest.mark.parametrize("bits", [32, 64])
def test_actual_constant_hoist_is_checked_before_bufferization(tmp_path, fidelity_tools, monkeypatch, bits):
    result = _lower(tmp_path / "gated", fidelity_tools, monkeypatch, bits=bits)
    report_path, report = _report(result)
    assert report["boundary_count"] == len(plans.preprocessing().boundaries) + 5
    native = stages.completed(report_path.parent / "native/stages")
    assert [json.loads((p / "meta.json").read_bytes())["boundary"] for p in native] == [
        "native-parse",
        "native-canonicalize",
        "native-cse",
        "native-guarded-fusion",
        "native-generalize",
    ]
    before, after = (native[1] / "before.mlir").read_text(), (native[1] / "after.mlir").read_text()
    assert before.index("arith.extsi") < before.index("arith.constant")
    assert after.index("arith.constant") < after.index("arith.extsi")
    plain = _lower(tmp_path / "ordinary", fidelity_tools, monkeypatch, bits=bits, gated=False)
    assert (result.workdir / "model.ll").read_bytes() == (plain.workdir / "model.ll").read_bytes()
    samples = [[[x]] for x in (0, 1, 127, 128, 255)]
    expected = [[str(((x if x < 128 else x - 256) + 22) & 4294967295)] for x in (0, 1, 127, 128, 255)]
    for name, program in (("original", SOURCE), ("held", (report_path.parent / "native/current.mlir").read_text())):
        _native_reference(program, samples, expected, tmp_path / name, fidelity_tools, bits)


def test_actual_preparation_prefix_remains_bound_to_original_source(tmp_path, fidelity_tools, monkeypatch):
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(fidelity_tools.compiler_python))
    source = tmp_path / "original.mlir"
    source.write_text(SOURCE)
    captured = preparation.capture(source, tmp_path / "prepare", Options("forward", 64, fidelity_tools.compiler_python))
    result = _lower(
        tmp_path / "native", fidelity_tools, monkeypatch, source=captured.target.decode(), captured=captured
    )
    _, report = _report(result)
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["boundary_count"] == len(plans.preparation().boundaries) + len(plans.preprocessing().boundaries) + 5
    assert report["preparation_plan_sha256"] == plans.preparation().sha256


def test_index64_input_can_be_qualified_without_index_arithmetic(tmp_path, fidelity_tools, monkeypatch):
    result = _lower(tmp_path / "gated", fidelity_tools, monkeypatch, source=INDEX_SOURCE)
    path, report = _report(result)
    assert report["index_bits"] == 64
    samples = [[[x]] for x in (0, 5, 9223372036854775807)]
    expected = [[str(x)] for x in (0, 5, 9223372036854775807)]
    _native_reference(
        (path.parent / "native/current.mlir").read_text(), samples, expected, tmp_path / "held", fidelity_tools, 64
    )


def test_index32_input_refuses_before_native_tensor_passes(tmp_path, fidelity_tools, monkeypatch):
    with pytest.raises(PipelineError, match="index-typed"):
        _lower(tmp_path, fidelity_tools, monkeypatch, bits=32, source=INDEX_SOURCE)
    roots = list((tmp_path / "lean-native-prefix").iterdir())
    assert len(roots) == 1
    assert json.loads((roots[0] / "status.json").read_bytes())["verdict"] == "unmeasured"
    assert not (roots[0] / "native/request.json").exists()
    assert not (roots[0] / "native/ack.json").exists()
    assert not (tmp_path / "model.ll").exists()


def test_valid_wrong_native_boundary_never_releases_suffix(tmp_path, fidelity_tools, monkeypatch):
    original = native_gate.Session._verify_bounded
    changed = []

    def corrupt(self, request, owner):
        stage = owner / "stages/0001_native-canonicalize"
        target = stage / "after.mlir"
        text = target.read_text()
        assert text.count("22 : i32") == 1
        mutant = text.replace("22 : i32", "23 : i32")
        target.write_text(mutant)
        _verify_mlir(target, fidelity_tools.compiler_python)
        meta = stage / "meta.json"
        payload = json.loads(meta.read_bytes())
        payload["after_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        meta.write_text(json.dumps(payload))
        changed.append(mutant)
        return original(self, request, owner)

    monkeypatch.setattr(native_gate.Session, "_verify_bounded", corrupt)
    with pytest.raises((native_gate.VerificationError, proof_worker.WorkerFailure), match="native-canonicalize"):
        _lower(tmp_path / "gated", fidelity_tools, monkeypatch)
    assert len(changed) == 1
    roots = list((tmp_path / "gated/lean-native-prefix").iterdir())
    assert not (roots[0] / "native/ack.json").exists()
    assert not (tmp_path / "gated/model.ll").exists()
    # The valid mutant changes an observed output, independently of the refusal.
    _native_reference(changed[0], [[[0]], [[128]]], [["23"], ["4294967191"]], tmp_path / "mutant", fidelity_tools, 64)
