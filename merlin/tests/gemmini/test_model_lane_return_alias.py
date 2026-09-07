"""Regression for a source entry argument returned directly beside compiled work.

The generated compiler package is supplied explicitly because Phase-2 candidates are immutable
artifacts, not importable repository modules.  The test exercises the normal package CLI and then the
shared structural verifier; it does not inspect a model name or rely on a portfolio shape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan


SOURCE = '''builtin.module {
  func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>, %passthrough: tensor<2xi32>) -> (tensor<2x4xi32>, tensor<2xi32>) {
    %empty = tensor.empty() : tensor<2x4xi32>
    %zero = arith.constant 0 : i32
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x4xi32>) -> tensor<2x4xi32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%init : tensor<2x4xi32>) attrs = {prov.region_id = "matmul_0", prov.family = "contraction", prov.op = "matmul", prov.orig_dtype = "int8"} {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %product = arith.muli %xe, %ye : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<2x4xi32>
    func.return %result, %passthrough : tensor<2x4xi32>, tensor<2xi32>
  }
}'''


def test_directly_returned_argument_has_one_readwrite_physical_buffer(tmp_path: Path) -> None:
    value = os.environ.get("MERLIN_MODEL_LANE_CANDIDATE")
    if not value:
        pytest.skip("requires an explicit generated model-lane compiler package")
    package = Path(value).resolve(strict=True)
    source = tmp_path / "direct_return_alias.mlir"
    command_buffer = tmp_path / "command_buffer.json"
    source.write_text(SOURCE, encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(package / "gemmini-opt"), "--source-convolution",
        "--convert-iface-to-gemmini", f"--emit-command-buffer={command_buffer}",
        "--emit-target-artifact", str(source),
    ], capture_output=True, text=True, timeout=60,
       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    cb = json.loads(command_buffer.read_text(encoding="utf-8"))
    plan = cb["params"]["global_program_plan"]

    assert plan["entry_bindings"] == ["arg0", "arg1", "Y1"]
    assert plan["output_bindings"] == ["Y0", "Y1"]
    assert "arg2" not in cb["tensors"]
    assert next(arg for arg in cb["kernel_abi"]["args"]
                if arg["tensor"] == "Y1")["access"] == "readwrite"
    evidence = verify_compiler_global_plan(
        source_text=SOURCE, lowered_text=result.stdout, command_buffer=cb,
        candidate_sha256="a" * 64)
    assert evidence["status"] == "verified", evidence["problems"]
