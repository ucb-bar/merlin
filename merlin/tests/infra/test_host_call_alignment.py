"""Host-owned address expressions, not kernel assumptions, prove call alignment."""

import subprocess

import pytest

from merlin.compile.model_execution_inputs import file_sha256
from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify.host_call_alignment import inspect, value_alignment


def test_round_down_alignment_requires_the_same_original_ssa_numerator():
    module = _parse_module("""module {
      llvm.func @test(%p: !llvm.ptr, %other: i64) {
        %n = llvm.ptrtoint %p : !llvm.ptr to i64
        %mask = llvm.mlir.constant(64) : i64
        %remainder = llvm.urem %n, %mask : i64
        %aligned = llvm.sub %n, %remainder : i64
        %changed = llvm.sub %other, %remainder : i64
        %a = llvm.inttoptr %aligned : i64 to !llvm.ptr
        %b = llvm.inttoptr %changed : i64 to !llvm.ptr
        llvm.return
      }
    }""")
    pointers = [op.results[0] for op in module.walk() if op.name == "llvm.inttoptr"]
    assert [value_alignment(value) for value in pointers] == [64, 1]


def _tools():
    from merlin.llvmlower import toolchain

    translator = toolchain.mlir_translate()
    compiler = toolchain.clang()
    if not all(path.is_file() for path in (translator, translator.with_name("llvm-extract"), compiler)):
        pytest.skip("selected LLVM tools unavailable; no hardware qualification")
    return compiler


def test_actual_host_import_binds_all_call_descriptors_and_rejects_source_drift(tmp_path):
    compiler = _tools()
    source, llvm, obj = (tmp_path / name for name in ("host.c", "host.ll", "host.o"))
    source.write_text("""
      typedef unsigned long U;
      extern void kernel(void*,void*,long,long,long,long,long,
                         void*,void*,long,long,long,long,long,
                         void*,void*,long,long,long,long,long);
      void forward(void *input, void *weight, void *output) {
        U a = (U)input + 63, b = (U)weight + 63, c = (U)output + 63;
        kernel(input,(void*)(a-a%64),0,2,2,2,1,
               weight,(void*)(b-b%64),0,2,2,2,1,
               output,(void*)(c-c%64),0,2,2,2,1);
      }
    """)
    subprocess.run([str(compiler), "-S", "-emit-llvm", "-O1", str(source), "-o", str(llvm)], check=True,
                   capture_output=True)
    subprocess.run([str(compiler), "-c", str(llvm), "-o", str(obj)], check=True, capture_output=True)
    records = {role: {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
               for role, path in (("llvm_ir", llvm), ("object", obj))}
    assigned = [{"symbol": "kernel"}]
    preparations = {"kernel": {"call_buffers": [
        {"shape": [2, 2], "dtype": dtype} for dtype in ("i8", "i8", "i32")
    ]}}
    actual, observation = inspect(records, assigned, preparations, diagnostics=tmp_path / "observations")
    assert actual == {"kernel": [64, 64, 64]}
    assert observation["host_artifacts"]["llvm_ir"] == records["llvm_ir"]["sha256"]
    assert len(observation["tools"]) == 3
    with pytest.raises(ValueError, match="call roster"):
        inspect(records, [*assigned, *assigned], preparations, diagnostics=tmp_path / "wrong-roster")
    llvm.write_text(llvm.read_text() + "\n; changed source\n")
    with pytest.raises(ValueError, match="actual compiled host artifact"):
        inspect(records, assigned, preparations, diagnostics=tmp_path / "changed")
