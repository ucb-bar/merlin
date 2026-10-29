"""Complete typed stage boundaries and real compiled marker/output behavior."""

from __future__ import annotations

import ctypes
import json
import subprocess
from io import StringIO

import pytest
from xdsl.dialects import arith, builtin, func
from xdsl.printer import Printer

from merlin.llvmlower import op_profile, toolchain
from merlin.llvmlower.op_profile_structural import _parse_text as parse
from merlin.llvmlower.op_profile_structural import instrument_module, verify_instrumentation

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


@pytest.mark.parametrize("fault", ["id", "missing", "body", "callee", "abi", "table", "metadata", "extra"])
def test_actual_marker_and_unchanged_source_checker_rejects_mutations(fault):
    source = parse(SOURCE)
    selected, table = instrument_module(source)
    original_text, selected_text = generic(SOURCE), str(selected)
    verify_instrumentation(source, selected, table)
    assert str(selected) == selected_text and generic(SOURCE) == original_text
    fn = next(op for op in selected.body.block.ops
              if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    constant, marker = list(fn.body.block.ops)[:2]
    if fault == "id":
        constant.properties["value"] = builtin.IntegerAttr(4, builtin.i32)
    elif fault == "missing":
        fn.body.block.erase_op(marker)
        fn.body.block.erase_op(constant)
    elif fault == "body":
        addition = next(op for op in fn.walk() if isinstance(op, arith.AddiOp))
        addition.operands = [addition.operands[0], addition.operands[0]]
    elif fault == "callee":
        marker.properties["callee"] = builtin.SymbolRefAttr("bump")
        marker.operands = [fn.body.block.args[0]]
    elif fault == "abi":
        declaration = next(op for op in selected.body.block.ops
                          if isinstance(op, func.FuncOp) and op.sym_name.data == op_profile.MARK_SYM)
        declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    elif fault == "table":
        table[0]["result_type"] = "i7"
    elif fault == "metadata":
        fn.attributes["unknown_effects"] = builtin.UnitAttr()
    else:
        selected.body.block.add_op(func.FuncOp.external("unrecorded", [], []))
    with pytest.raises(op_profile.OpProfileError, match="profile"):
        verify_instrumentation(source, selected, table)


def test_profile_checker_retains_negative_zero_source_words():
    source = parse("module { func.func @forward() -> f32 { %z = arith.constant -0.0 : f32\n return %z : f32 } }")
    selected, table = instrument_module(source)
    constant = next(op for op in selected.walk()
                    if isinstance(op, arith.ConstantOp) and isinstance(op.value, builtin.FloatAttr))
    constant.properties["value"] = builtin.FloatAttr(0.0, builtin.f32)
    with pytest.raises(op_profile.OpProfileError, match="metadata"):
        verify_instrumentation(source, selected, table)


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
    from xdsl.dialects.builtin import UnitAttr

    broken = parse(selected)
    marker = next(op for op in broken.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == op_profile.MARK_SYM)
    marker.attributes["llvm.emit_c_interface"] = UnitAttr()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(broken)
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
    for name, source in [("original", SOURCE), ("selected", selected), ("broken_abi", stream.getvalue())]:
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
        linked = subprocess.run(
            [str(clang), "-O2", "-shared", "-fPIC", "-Wl,-z,defs", str(llvm), str(support), "-o", str(so)],
            capture_output=True,
        )
        if name == "broken_abi":
            assert linked.returncode != 0 and b"_mlir_ciface_merlin_prof_mark" in linked.stderr
            continue
        assert linked.returncode == 0, linked.stderr
        library = ctypes.CDLL(str(so))
        values = (ctypes.c_int32 * 4)(1, 22, 33, 44)
        library.invoke.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        library.invoke(values)
        assert list(values) == [8, 15, 33, 44]
        count = ctypes.c_int32.in_dll(library, "marker_count").value
        ids = (ctypes.c_int32 * 64).in_dll(library, "marker_ids")
        assert list(ids)[:count] == (list(range(len(table) + 1)) if name == "selected" else [])


@pytest.mark.parametrize("backend,padded", [("rvv", False), ("scalar", False), ("scalar", True)])
def test_ordinary_generic_model_build_profiles_and_executes_on_rv64gc(tmp_path, backend, padded):
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
    # The second complete result models carried state. Profiling must preserve
    # both results, not merely the printed prefix of the first tensor.
    source = source.replace("%x: tensor<5xf32>) -> tensor<5xf32>",
        "%x: tensor<5xf32>, %state: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)")
    source = source.replace("return %out : tensor<5xf32>", """
        %next = linalg.generic {
          indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]
        } ins(%state, %out: tensor<5xf32>, tensor<5xf32>) outs(%empty: tensor<5xf32>) {
        ^bb0(%old: f32, %value: f32, %unused: f32):
          %sum = arith.addf %old, %value : f32
          linalg.yield %sum : f32
        } -> tensor<5xf32>
        return %out, %next : tensor<5xf32>, tensor<5xf32>""")
    if padded:
        source = source.replace("%next = linalg.generic", """%v = arith.constant -7.0 : f32
        %pad = tensor.splat %v : tensor<8xf32>
        %next = linalg.generic""").replace(
            "return %out, %next : tensor<5xf32>, tensor<5xf32>", """
        %padded = "tensor.insert_slice"(%next,%pad) <{static_offsets=array<i64:1>,
          static_sizes=array<i64:5>,static_strides=array<i64:1>,operandSegmentSizes=array<i32:1,1,0,0,0>}>
          : (tensor<5xf32>,tensor<8xf32>) -> tensor<8xf32>
        return %out, %padded : tensor<5xf32>, tensor<8xf32>""").replace(
            "-> (tensor<5xf32>, tensor<5xf32>)", "-> (tensor<5xf32>, tensor<8xf32>)")
    (bundle / "model.mlir").write_text(generic(source))
    (bundle / "weights.safetensors.manifest.json").write_text(
        '{"0":{"kind":"input","name":"values"},"1":{"kind":"input","name":"state"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    state = np.array([3, 4, 5, 6, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values, in1=state)
    from merlin.compile.baremetal_model import read_full_outputs
    from merlin.runtime.operation_profile import summarize
    from merlin.runtime.out_bin import binary_console_diagnostics

    results = []
    for selected in (False, True):
        built = spike_model.build(
            bundle, work / str(selected), arena_mb=1, op_profile=selected, full_readback=True, backend=backend,
            cflags_override=[
                "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin",
            ],
        )
        raw = spike_model.run_raw(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
        outputs, _metrics = read_full_outputs(raw, bundle, built["build_hash"])
        np.testing.assert_array_equal(outputs[0], values + np.float32(2))
        expected_state = state + values + np.float32(2)
        if padded:
            expected_state = np.concatenate([[-7], expected_state, [-7, -7]]).astype(np.float32)
            destination = json.loads((work / str(selected) / "lower/automatic-host-destinations.json").read_text())
            assert destination["rewrites"] == 1
        np.testing.assert_array_equal(outputs[1], expected_state)
        text = binary_console_diagnostics(raw).decode()
        results.append(outputs)
        (work / str(selected) / "console.bin").write_bytes(raw)
        if not selected:
            assert not op_profile.parse_prof_lines(text)
            continue
        table = json.loads((work / str(selected) / "op_profile_table.json").read_text())
        assert (work / str(selected) / "op_profile_table.json").read_bytes() == (
            work / str(selected) / "lower/op_profile_table.json").read_bytes()
        report = summarize(table, text)
        (work / str(selected) / "operation_profile.json").write_text(json.dumps(report))
        assert report["executions"] == 1 and report["marks"] == len(table) + 1
        assert report["inter_invocation_counter_ticks"] == 0
        assert all(row["hits"] == 1 for row in report["operations"])
    for plain, profiled in zip(*results, strict=True):
        np.testing.assert_array_equal(plain, profiled)
    if not padded:
        assert [row["mlir_op"] for row in table] == ["tensor.empty", "linalg.generic", "linalg.generic"]
        assert table[1]["body_ops"] == ["arith.addf", "arith.constant"]
    else:
        assert any(row["mlir_op"] == "tensor.extract_slice" for row in table)
