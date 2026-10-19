"""Changing structured boundaries qualify generalization and expose the fusion gap.

Native references observe initialized independent destinations. No nested captured
initialized destination is reused; the retained S4 exclusion remains in force.
"""

import json

import pytest
from test_lean_native_gate_differential import _lower, _native_reference, _report
from test_lean_native_gate_differential import ordinary_selection as ordinary_selection
from test_lean_parallel_fusion import source as _fusion_source
from test_lean_region_exact import NAMED_SOURCE

from merlin.verify.lean import native_gate, proof_worker

pytestmark = pytest.mark.usefixtures("ordinary_selection")

TRANSPOSE_SOURCE = """func.func @forward(%x: tensor<2x3xi32>, %init: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %r = linalg.transpose ins(%x : tensor<2x3xi32>) outs(%init : tensor<3x2xi32>) permutation = [1, 0]
  return %r : tensor<3x2xi32>
}"""

FUSION_SOURCE = """func.func @forward(%x: tensor<2xi32>, %pinit: tensor<2xi32>,
    %cinit: tensor<2xi32>) -> tensor<2xi32> {
  %p = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
  } ins(%x : tensor<2xi32>) outs(%pinit : tensor<2xi32>) {
  ^bb0(%a: i32, %old: i32):
    %v = arith.addi %a, %a : i32
    linalg.yield %v : i32
  } -> tensor<2xi32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
  } ins(%p : tensor<2xi32>) outs(%cinit : tensor<2xi32>) {
  ^bb0(%a: i32, %old: i32):
    %v = arith.muli %a, %a : i32
    linalg.yield %v : i32
  } -> tensor<2xi32>
  return %r : tensor<2xi32>
}"""
CAPTURED_FUSION_SOURCE = _fusion_source(captured=True)


def test_actual_named_matmul_changes_generalize_and_releases_proved_prefix(tmp_path, fidelity_tools, monkeypatch):
    result = _lower(tmp_path / "gated", fidelity_tools, monkeypatch, source=NAMED_SOURCE)
    path, report = _report(result)
    stage = path.parent / "native/stages/0004_native-generalize"
    before, after = (stage / "before.mlir").read_text(), (stage / "after.mlir").read_text()
    assert before != after
    assert before.count('"linalg.matmul"') == 1 and '"linalg.generic"' not in before
    assert after.count('"linalg.generic"') == 1 and '"linalg.matmul"' not in after
    chain = json.loads((path.parent / "chain.json").read_bytes())
    assert chain["verdict"] == "verified"
    assert report["boundary_count"] == 9
    plain = _lower(tmp_path / "ordinary", fidelity_tools, monkeypatch, source=NAMED_SOURCE, gated=False)
    assert (result.workdir / "model.ll").read_bytes() == (plain.workdir / "model.ll").read_bytes()
    samples = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [10, 20, 30, 40]],
        [[4294967295, 0, 2, 2147483648], [1, 3, 5, 7], [17, 19, 23, 29]],
    ]
    expected = [
        [
            str((initial[i * 2 + j] + sum(a[i * 2 + k] * b[k * 2 + j] for k in range(2))) & 4294967295)
            for i in range(2)
            for j in range(2)
        ]
        for a, b, initial in samples
    ]
    for label, program in (("original", NAMED_SOURCE), ("held", (path.parent / "native/current.mlir").read_text())):
        _native_reference(program, samples, expected, tmp_path / label, fidelity_tools, 64)


def test_actual_named_transpose_changes_generalize_and_preserves_ordinary_bytes(tmp_path, fidelity_tools, monkeypatch):
    result = _lower(tmp_path / "gated", fidelity_tools, monkeypatch, source=TRANSPOSE_SOURCE)
    path, report = _report(result)
    stage = path.parent / "native/stages/0004_native-generalize"
    before, after = (stage / "before.mlir").read_text(), (stage / "after.mlir").read_text()
    assert before != after
    assert before.count('"linalg.transpose"') == 1 and '"linalg.generic"' not in before
    assert after.count('"linalg.generic"') == 1 and '"linalg.transpose"' not in after
    assert json.loads((path.parent / "chain.json").read_bytes())["verdict"] == "verified"
    assert report["boundary_count"] == 9
    plain = _lower(tmp_path / "ordinary", fidelity_tools, monkeypatch, source=TRANSPOSE_SOURCE, gated=False)
    assert (result.workdir / "model.ll").read_bytes() == (plain.workdir / "model.ll").read_bytes()
    samples = [
        [[1, 2, 3, 4, 5, 6], [17, 19, 23, 29, 31, 37]],
        [[4294967295, 2147483648, 0, 7, 11, 13], [41, 43, 47, 53, 59, 61]],
    ]
    expected = [[str(data[i * 3 + j]) for j in range(3) for i in range(2)] for data, _ in samples]
    for label, program in (("original", TRANSPOSE_SOURCE), ("held", (path.parent / "native/current.mlir").read_text())):
        _native_reference(program, samples, expected, tmp_path / label, fidelity_tools, 64)


def test_actual_repeated_input_fusion_changes_ir_and_releases_proved_prefix(tmp_path, fidelity_tools, monkeypatch):
    result = _lower(tmp_path / "gated", fidelity_tools, monkeypatch, source=FUSION_SOURCE)
    path, report = _report(result)
    stage = path.parent / "native/stages/0003_native-guarded-fusion"
    before, after = (stage / "before.mlir").read_text(), (stage / "after.mlir").read_text()
    assert before != after
    assert before.count('"linalg.generic"') == 2 and after.count('"linalg.generic"') == 1
    assert '"tensor.empty"' in after
    assert json.loads((path.parent / "chain.json").read_bytes())["verdict"] == "verified"
    assert report["boundary_count"] == 9
    plain = _lower(tmp_path / "ordinary", fidelity_tools, monkeypatch, source=FUSION_SOURCE, gated=False)
    assert (result.workdir / "model.ll").read_bytes() == (plain.workdir / "model.ll").read_bytes()
    samples = [[[2, 3], [11, 13], [17, 19]], [[2147483648, 4294967295], [23, 29], [31, 37]]]
    expected = [[str((2 * value) ** 2 & 4294967295) for value in x] for x, _, _ in samples]
    for label, program in (("original", FUSION_SOURCE), ("held", (path.parent / "native/current.mlir").read_text())):
        _native_reference(program, samples, expected, tmp_path / label, fidelity_tools, 64)


def test_actual_captured_operand_fusion_remains_fail_closed_before_native_suffix(tmp_path, fidelity_tools, monkeypatch):
    gated = tmp_path / "gated"
    with pytest.raises((native_gate.VerificationError, proof_worker.WorkerFailure), match="native-guarded-fusion"):
        _lower(gated, fidelity_tools, monkeypatch, source=CAPTURED_FUSION_SOURCE)
    roots = list((gated / "lean-native-prefix").iterdir())
    assert len(roots) == 1
    root = roots[0]
    stage = root / "native/stages/0003_native-guarded-fusion"
    before, after = (stage / "before.mlir").read_text(), (stage / "after.mlir").read_text()
    assert before != after
    assert before.count('"linalg.generic"') == 2 and after.count('"linalg.generic"') == 1
    assert json.loads((stage / "receipt.json").read_bytes())["verdict"] == "abstracted"
    assert json.loads((root / "status.json").read_bytes())["verdict"] == "unmeasured"
    assert not (root / "native/ack.json").exists()
    assert not (gated / "model.ll").exists()
    prefixes = list((gated / "lean-shared-passes").iterdir())
    assert len(prefixes) == 1
    for name in ("support", "definedness"):
        assert json.loads((prefixes[0] / name / "receipt.json").read_bytes())["verdict"] == "verified"
    samples = [
        [[2, 3], [11, 13], [17, 19], [7]],
        [[1, 2], [23, 29], [31, 37], [4294967295]],
    ]
    expected = [
        [str(((value + capture[0]) & 4294967295) ** 2 & 4294967295) for value in x] for x, _, _, capture in samples
    ]
    # Equal sampled executions do not discharge the missing fusion theorem. The
    # actual compilation remains stopped, and no strict native-prefix claim exists.
    for label, program in (("original", CAPTURED_FUSION_SOURCE), ("held", (root / "native/current.mlir").read_text())):
        _native_reference(program, samples, expected, tmp_path / label, fidelity_tools, 64)
