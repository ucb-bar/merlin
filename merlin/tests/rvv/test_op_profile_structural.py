"""Complete typed stage boundaries and real compiled marker/output behavior."""

from __future__ import annotations

import ctypes
import json
import subprocess
from io import StringIO

import pytest
from xdsl.dialects import func
from xdsl.printer import Printer

from merlin.llvmlower import op_profile, toolchain
from merlin.llvmlower.op_profile_structural import _parse_text as parse
from merlin.llvmlower.op_profile_structural import instrument_module

SOURCE = """module {
  func.func private @bump(%x: memref<4xi32>) {
    %i = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %v = memref.load %x[%i] : memref<4xi32>
    %w = arith.addi %v, %one : i32
    memref.store %w, %x[%i] : memref<4xi32>
    return
  }
  func.func @forward(%x: memref<4xi32>) attributes {llvm.emit_c_interface} {
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %seven = arith.constant 7 : i32
    memref.store %seven, %x[%i] {prov.region_id = "write", prov.family = "memory"} : memref<4xi32>
    func.call @bump(%x) {prov.region_id = "effect", prov.source_node_ids = ["original:effect"]} : (memref<4xi32>) -> ()
    %v = memref.load %x[%i] : memref<4xi32>
    %w = arith.addi %v, %seven : i32
    memref.store %w, %x[%j] : memref<4xi32>
    return
  }
}"""


def generic(source):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(parse(source))
    return stream.getvalue()


def test_generic_and_custom_forms_cover_stores_and_effect_only_calls():
    custom_text, custom = op_profile.instrument(SOURCE, structural=True)
    generic_text, table = op_profile.instrument(generic(SOURCE), structural=True)
    assert table == custom
    assert generic_text == custom_text
    assert [row["id"] for row in table] == list(range(8))
    assert [row["mlir_op"] for row in table] == [
        "arith.constant",
        "arith.constant",
        "arith.constant",
        "memref.store",
        "func.call",
        "memref.load",
        "arith.addi",
        "memref.store",
    ]
    assert table[3]["result_types"] == [] and table[3]["family"] == "memory"
    assert table[4]["callee"] == "@bump" and table[4]["result_type"] is None
    assert table[4]["source_node_ids"] == ["original:effect"]
    selected = parse(generic_text)
    selected.verify()
    markers = [
        op
        for op in selected.walk()
        if isinstance(op, func.CallOp) and op.callee.root_reference.data == op_profile.MARK_SYM
    ]
    assert len(markers) == len(table) + 1


def test_cached_input_module_is_never_mutated_and_existing_marker_refuses():
    module = parse(SOURCE)
    original = str(module)
    selected, _ = instrument_module(module)
    assert str(module) == original
    with pytest.raises(op_profile.OpProfileError, match="marker symbol already exists"):
        instrument_module(selected)
    assert str(module) == original


def test_named_typed_selection_keeps_legacy_positional_api_and_refuses_multiple():
    source = SOURCE.replace("func.func @forward(", "func.func @selected(")
    selected, table = op_profile.instrument(source, ["selected"], structural=True)
    assert [row["id"] for row in table] == list(range(8))
    module = parse(selected)
    markers = [
        op
        for op in module.walk()
        if isinstance(op, func.CallOp) and op.callee.root_reference.data == op_profile.MARK_SYM
    ]
    assert len(markers) == len(table) + 1
    legacy, legacy_table = op_profile.instrument(source, ["selected"])
    assert "function" not in legacy_table[0]
    assert f"func.func private @{op_profile.MARK_SYM}" in legacy
    with pytest.raises(op_profile.OpProfileError, match="exactly one selected function"):
        op_profile.instrument(source, ["selected", "bump"], structural=True)
    with pytest.raises(op_profile.OpProfileError, match="no function named"):
        op_profile.instrument(source, [], structural=True)


def test_a_module_the_context_cannot_parse_is_a_refusal_not_a_crash():
    """A debug image build asks for profiling and degrades on OpProfileError. A named linalg op the
    profiling context has no format for (seen: `linalg.reciprocal` in a per-channel dequant) used to
    escape as an xDSL ParseError and take the whole image build with it."""
    source = """module {
      func.func @forward(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
        %r = linalg.reciprocal ins(%a : tensor<4xf32>) outs(%b : tensor<4xf32>) -> tensor<4xf32>
        return %r : tensor<4xf32>
      }
    }"""
    with pytest.raises(op_profile.OpProfileError, match="cannot parse the module"):
        op_profile.instrument(source, ["forward"], structural=True)


def test_multi_block_control_flow_refuses_before_mutation():
    source = """module {
      func.func @forward() {
        cf.br ^done
      ^done:
        return
      }
    }"""
    module = parse(source)
    original = str(module)
    with pytest.raises(op_profile.OpProfileError, match="one function body block"):
        instrument_module(module)
    assert str(module) == original


def test_generic_c_interface_marks_public_definition_and_preserves_callbacks():
    from merlin.llvmlower.passes_xdsl import _attach_c_interface

    source = SOURCE.replace(" attributes {llvm.emit_c_interface}", "")
    selected, _ = op_profile.instrument(source, structural=True)
    result, count = _attach_c_interface(selected)
    assert count == 1
    module = parse(result)
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    assert "llvm.emit_c_interface" in functions["forward"].attributes
    assert "llvm.emit_c_interface" not in functions["bump"].attributes
    assert "llvm.emit_c_interface" not in functions[op_profile.MARK_SYM].attributes
    assert _attach_c_interface(result) == (result, 0)


def test_actual_compiled_outputs_and_effect_interval_marker_order(tmp_path):
    translate = toolchain.mlir_translate()
    opt = translate.with_name("mlir-opt")
    clang = toolchain.clang()
    if not all(tool.is_file() for tool in (translate, opt, clang)):
        pytest.skip("selected native MLIR toolchain unavailable")
    selected, table = op_profile.instrument(SOURCE, structural=True)
    support = tmp_path / "support.c"
    support.write_text("""#include <stdint.h>
int32_t marker_ids[64];
int32_t marker_count;
void merlin_prof_mark(int32_t id) { marker_ids[marker_count++] = id; }
struct descriptor { int32_t *allocated, *aligned; int64_t offset, sizes[1], strides[1]; };
extern void _mlir_ciface_forward(struct descriptor *);
void invoke(int32_t *values) {
  struct descriptor d = {values, values, 0, {4}, {1}};
  marker_count = 0;
  _mlir_ciface_forward(&d);
}
""")
    for name, source in [("original", SOURCE), ("selected", selected)]:
        input_path = tmp_path / (name + ".mlir")
        lowered = tmp_path / (name + ".llvm.mlir")
        llvm = tmp_path / (name + ".ll")
        so = tmp_path / (name + ".so")
        input_path.write_text(source)
        subprocess.run(
            [
                str(opt),
                str(input_path),
                "--convert-scf-to-cf",
                "--convert-arith-to-llvm",
                "--convert-index-to-llvm",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--reconcile-unrealized-casts",
                "-o",
                str(lowered),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [str(translate), "--mlir-to-llvmir", str(lowered), "-o", str(llvm)], check=True, capture_output=True
        )
        subprocess.run(
            [str(clang), "-O2", "-shared", "-fPIC", str(llvm), str(support), "-o", str(so)],
            check=True,
            capture_output=True,
        )
        library = ctypes.CDLL(str(so))
        values = (ctypes.c_int32 * 4)(1, 22, 33, 44)
        library.invoke.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        library.invoke(values)
        assert list(values) == [8, 15, 33, 44]
        count = ctypes.c_int32.in_dll(library, "marker_count").value
        ids = (ctypes.c_int32 * 64).in_dll(library, "marker_ids")
        assert list(ids)[:count] == (list(range(len(table) + 1)) if name == "selected" else [])


def test_ordinary_generic_model_build_profiles_and_executes_on_rv64gc(tmp_path):
    import numpy as np

    from merlin.runtime.backends import spike, spike_model

    if not spike.available() or not toolchain.available():
        pytest.skip("selected upstream and bare-metal toolchain unavailable")
    bundle, work = tmp_path / "bundle", tmp_path / "build"
    bundle.mkdir()
    source = """module {
      func.func @forward(%x: tensor<5xf32>) -> tensor<5xf32> {
        %empty = tensor.empty() : tensor<5xf32>
        %out = linalg.generic {
          indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]
        } ins(%x: tensor<5xf32>) outs(%empty: tensor<5xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %two = arith.constant 2.0 : f32
          %sum = arith.addf %value, %two : f32
          linalg.yield %sum : f32
        } -> tensor<5xf32>
        return %out : tensor<5xf32>
      }
    }"""
    (bundle / "model.mlir").write_text(generic(source))
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        op_profile=True,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )
    ran = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(ran["outputs"], values + np.float32(2))
    table = json.loads((work / "op_profile_table.json").read_text())
    assert [row["mlir_op"] for row in table] == ["tensor.empty", "linalg.generic"]
    assert table[1]["body_ops"] == ["arith.addf", "arith.constant"]
    measurements = op_profile.parse_prof_lines(ran["console"])
    assert set(measurements) == {0, 1}
    assert all(hits == 1 for ticks, hits in measurements.values())
    assert ran["metrics"]["prof_marks"] == 3
