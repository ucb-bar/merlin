"""Ordinary compiled objects are verified inputs, like explicitly prepared objects."""

import json
import os
from pathlib import Path

import pytest

from merlin.compile.outlined_program import build_outlined_program
from merlin.llvmlower import kernel_backend
from merlin.runtime.staged_dispatch import StagedRoles

SOURCE = """module {
func.func @forward(%a:tensor<4xi32>,%b:tensor<4xi32>)->tensor<4xi32> {
 %e=tensor.empty():tensor<4xi32>
 %r=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],
 iterator_types=["parallel"]} ins(%a,%b:tensor<4xi32>,tensor<4xi32>)outs(%e:tensor<4xi32>){
 ^bb0(%x:i32,%y:i32,%o:i32):
 %z=arith.addi %x,%y:i32
 linalg.yield %z:i32
 }->tensor<4xi32>
 func.return %r:tensor<4xi32>
}}
"""


@pytest.mark.parametrize("mutate", [False, True])
def test_actual_compiled_object_binding_and_changed_bytes_refusal(tmp_path, monkeypatch, mutate):
    if not os.environ.get("MERLIN_MLIR_INSTALL") or not os.environ.get("MERLIN_COMPILER_PYTHON"):
        pytest.skip("explicit MLIR compiler environment unavailable")
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    original = kernel_backend.compile_deployable_kernel

    def compile_then_change(*args, **kwargs):
        result = original(*args, **kwargs)
        obj = Path(result["object"])
        raw = obj.read_bytes()
        obj.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        return result

    if mutate:
        monkeypatch.setattr(kernel_backend, "compile_deployable_kernel", compile_then_change)
    output = tmp_path / "program"
    kwargs = dict(
        roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)),
        worker_slots=(0,),
        workspace_budget_bytes=1024,
        alignment_bytes=64,
        compute_cflags=("-march=rv64gcv",),
    )
    if mutate:
        with pytest.raises(ValueError, match="ordinary kernel object changed"):
            build_outlined_program(source, output, **kwargs)
        assert json.loads((output / "build.json").read_bytes())["status"] == "failed"
    else:
        report = build_outlined_program(source, output, **kwargs)
        assert report["status"] == "built" and len(report["kernel_builds"]) == 1
        for kernel in report["kernel_builds"]:
            assert report["input_sha256"][kernel["object"]] == kernel["object_sha256"]
        assert all(path in report["input_sha256"] for path in report["link_objects"])
