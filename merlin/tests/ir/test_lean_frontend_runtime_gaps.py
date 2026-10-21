"""Qualify actual-export frozen-runtime obstruction proofs without changing semantics."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from test_lean_frontend_bool import _tools

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean.export import export_file
from merlin.xdsl_dialects._common import text as to_text


def _gaps(source: str, work: Path, function: str = "subject") -> dict:
    python, _ = _tools()
    work.mkdir(parents=True, exist_ok=True)
    path = work / "program.mlir"
    path.write_text(source)
    exported = work / "program.json"
    # The authoritative exporter parses and verifies the actual MLIR before Lean reads it.
    export_file(path, exported, backend="mlir", m2m_python=python)
    fixture = merlin_dir() / "tests/fixtures/lean_frontend_runtime_gaps.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(fixture), str(exported), function],
        cwd=merlin_dir() / "lean",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["success_theorem"] == "MerlinVerify.Rules.FrontendBoolRuntimeGaps.neverSucceeds_sound"
    (work / "refusals.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@pytest.mark.parametrize(
    "source,operations,successful_execution",
    [
        (
            "func.func @subject(%n: index) -> tensor<?xi64> {"
            "%e = tensor.empty(%n) : tensor<?xi64> return %e : tensor<?xi64> }",
            ["tensor.empty"],
            "proved impossible",
        ),
        (
            "func.func @subject(%b: i1) -> i64 {"
            "%z = arith.constant 0 : i64 %o = arith.constant 1 : i64 "
            "%r = scf.if %b -> i64 {scf.yield %o : i64} else {scf.yield %z : i64} "
            "return %r : i64 }",
            ["scf.if"],
            "proved impossible",
        ),
        (
            "func.func @subject(%n: index) -> index {"
            "%z = arith.constant 0 : index %o = arith.constant 1 : index "
            "%r = scf.for %i = %z to %n step %o iter_args(%s = %z) -> index {"
            "%v = arith.addi %s, %o : index scf.yield %v : index} return %r : index }",
            ["scf.for"],
            "proved impossible",
        ),
        (
            "func.func @subject(%x: tensor<0xf32>) -> tensor<0xi1> {"
            "%z = arith.constant 0 : index %o = arith.constant 1 : i1 "
            "%e = tensor.empty() : tensor<0xi1> "
            "%r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],"
            'iterator_types = ["parallel"]} ins(%x : tensor<0xf32>) outs(%e : tensor<0xi1>) {'
            "^bb0(%v: f32, %old: i1): %unused = tensor.empty(%z) : tensor<?xi64> "
            "linalg.yield %o : i1} -> tensor<0xi1> return %r : tensor<0xi1> }",
            ["tensor.empty"],
            # A nested witness alone does not prove that every successful path visits it.
            "unmeasured",
        ),
    ],
)
def test_compiler_valid_runtime_gap_witnesses(tmp_path, source, operations, successful_execution):
    result = _gaps("builtin.module {" + source + "}", tmp_path)
    assert [w["operation"] for w in result["witnesses"]] == operations
    assert result["successful_execution"] == successful_execution


def test_actual_smolvla_has_unconditional_frozen_runtime_obstruction(tmp_path):
    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    before = to_text(module, generic=True)
    assert fix_bool_fptosi(module) == 1
    after = to_text(module, generic=True)
    for side, source in (("before", before), ("after", after)):
        result = _gaps(source, tmp_path / side, "forward")
        assert result["successful_execution"] == "proved impossible"
        assert Counter(w["operation"] for w in result["witnesses"]) == {
            "tensor.empty": 1,
            "scf.for": 2,
            "scf.if": 2,
        }
