"""Original dense-offset transport must preserve copies and ABI refusal gates."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.dialects import builtin

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.device_input_views import prepare, rewrite_readonly_dense_offsets, validate_selection


def source():
    return '''builtin.module {
      "func.func"() <{sym_name="writer", sym_visibility="private",
      function_type=(tensor<5x8xi8>,tensor<8x6xi8>,tensor<5x6xi32>)->tensor<5x6xi32>,
      arg_attrs=[{bufferization.access="read"},{bufferization.access="read"},
                 {bufferization.access="write"}]}> ({}) : () -> ()
      func.func @forward(%a:tensor<5x8xi8>,%w:tensor<8x6xi8>)->tensor<5x6xi32> {
        %e=tensor.empty():tensor<5x6xi32>
        %r=func.call @writer(%a,%w,%e):(tensor<5x8xi8>,tensor<8x6xi8>,tensor<5x6xi32>)->tensor<5x6xi32>
        return %r:tensor<5x6xi32>
      }
    }'''


def test_only_readonly_arguments_gain_dense_dynamic_offsets():
    module = parse_mlir_text(source())
    declaration = module.body.block.first_op
    original_output = declaration.arg_attrs.data[2]
    assert rewrite_readonly_dense_offsets(module, {"writer": (0, 1)}) == {"writer": [0, 1]}
    assert declaration.arg_attrs.data[2] == original_output
    for index, columns in ((0, 8), (1, 6)):
        assert declaration.arg_attrs.data[index].data == {
            "bufferization.access": builtin.StringAttr("read"),
            "bufferization.buffer_layout": builtin.StridedLayoutAttr([columns, 1], None),
        }


@pytest.mark.parametrize("case", ["choice", "backend", "effects", "catalog", "other_preparation"])
def test_original_route_limits_remain_explicit(case):
    from types import SimpleNamespace

    device = SimpleNamespace(granularity="group")
    choice, backend, unsupported = True, "scalar", ()
    if case == "choice":
        choice = 1
    elif case == "backend":
        backend = "rvv"
    elif case == "effects":
        device.descriptor_results = True
    elif case == "catalog":
        device.catalog_manifest = "a separately selected compiled catalog"
    else:
        unsupported = (object(),)
    with pytest.raises(ValueError):
        validate_selection(choice, backend=backend, device=device, unsupported=unsupported)
    validate_selection(False, backend=backend, device=device, unsupported=unsupported)


@pytest.mark.parametrize("case", ["write", "unknown", "duplicate", "rank", "dynamic", "old_layout"])
def test_unsupported_bindings_refuse_atomically(case):
    text = source()
    if case == "rank":
        text = text.replace("5x8xi8", "40xi8")
    if case == "dynamic":
        text = text.replace("5x8xi8", "?x8xi8")
    if case == "old_layout":
        text = text.replace('bufferization.access="read"',
                            'bufferization.access="read", bufferization.buffer_layout=strided<[8,1]> ', 1)
    module = parse_mlir_text(text)
    before = str(module)
    selected = {"writer": (0, 2) if case == "write" else (0, 0) if case == "duplicate" else (0, 1)}
    if case == "unknown":
        selected["missing"] = (0,)
    with pytest.raises(ValueError):
        rewrite_readonly_dense_offsets(module, selected)
    assert str(module) == before


def compiled_shim(tmp_path, monkeypatch):
    from logical_device_target import install

    from merlin.common import invocation_record as I
    from merlin.common.digest import sha256_file
    from merlin.llvmlower import device_shim as S
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.llvmlower.device_build import DeviceBuild
    from merlin.llvmlower.toolchain import clang
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    install(monkeypatch)
    cb = {"abi_version": "0.1", "target": "gemmini", "tensors": {
        "A": {"shape": [5, 8], "dtype": "i8", "role": "input"},
        "W": {"shape": [8, 6], "dtype": "i8", "role": "weight"},
    }, "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"},
         "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "R", "dst": "C"}},
        {"opcode": "COMMIT", "operands": {"src": "C", "dst": "Y"},
         "attributes": {"output_dtype": "i32", "epilogue": []}},
    ]}
    interface = emit_interface_mlir(cb)
    (tmp_path / "writer.iface.mlir").write_text(interface)
    buffers, reason = S.logical_buffers(interface)
    assert buffers is not None, reason
    call = [{"role": role, "shape": shape, "dtype": dtype} for role, shape, dtype in (
        ("input_0", [5, 8], "i8"), ("weight_0", [8, 6], "i8"), ("out_0", [5, 6], "i32"))]
    unit = S.emit_logical_translation_unit("gemmini", {"writer": (buffers, call)},
                                         kernel_symbol_for=lambda _: "original_kernel")
    shim, obj = tmp_path / "device_shim.c", tmp_path / "device_shim.o"
    shim.write_text(unit.text)
    result = I.run([clang(), "-O1", "-c", str(shim), "-o", str(obj)], directory=tmp_path / "observed",
                   stage="original_test_shim_compilation", inputs=(shim,), outputs=(obj,),
                   capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    invocation, = (tmp_path / "observed").rglob("invocation.json")
    built = DeviceBuild(device="gemmini", objects=(obj,), shim_object=obj,
                        kernels={"writer": "original_kernel"}, shim_unit=unit,
                        shim_invocation={"path": str(invocation), "sha256": sha256_file(invocation)})
    original = tmp_path / "source.mlir"
    original.write_text(source())
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps({"device": "gemmini", "granularity": "group",
                                  "call_buffers": {"writer": call},
                                  "arg_access": {"writer": ["read", "read", "write"]}}))
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    return original, sidecar, built, recipe


def test_actual_emitter_interface_and_compiled_source_join(tmp_path, monkeypatch):
    original, sidecar, built, recipe = compiled_shim(tmp_path, monkeypatch)
    selected = prepare(original, sidecar, built, tmp_path / "views", recipe)
    module = parse_mlir_text(selected.read_text())
    assert "bufferization.buffer_layout" in module.body.block.first_op.arg_attrs.data[0].data
    assert "bufferization.buffer_layout" not in module.body.block.first_op.arg_attrs.data[2].data
    assert original.read_text() == source()
    assert "merlin_disjoint" in built.shim_unit.text  # original output alias guard retained verbatim
    assert (tmp_path / "views/selection.json").is_file()
    assert any(pin["path"] == str(built.shim_object) for pin in recipe.record["preparation"].values())


@pytest.mark.parametrize("case", ["shim", "interface", "call_shape", "source_shape", "unit"])
def test_changed_source_or_abi_custody_refuses(tmp_path, monkeypatch, case):
    original, sidecar, built, recipe = compiled_shim(tmp_path, monkeypatch)
    if case == "shim":
        built.shim_object.with_suffix(".c").write_text("/* changed */")
    elif case == "interface":
        (tmp_path / "writer.iface.mlir").write_text("builtin.module {}")
    elif case == "call_shape":
        record = json.loads(sidecar.read_text())
        record["call_buffers"]["writer"][0]["shape"] = [1, 40]
        sidecar.write_text(json.dumps(record))
    elif case == "source_shape":
        original.write_text(source().replace("5x8xi8", "1x40xi8"))
    else:
        built = replace(built, shim_unit=replace(built.shim_unit, text="/* invented */"))
    with pytest.raises((ValueError, KeyError)):
        prepare(original, sidecar, built, tmp_path / "views", recipe)
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize("stride", [1, 2])
def test_actual_contiguous_view_reuses_input_and_strided_view_keeps_copy(tmp_path, monkeypatch, stride):
    """Compile the actual logical shim and original MLIR through normal tools.

    The C contraction is an independent small test oracle, never an authored
    target implementation. Both source versions retain every output check.
    """
    import ctypes
    import re

    import numpy as np

    from merlin.common import invocation_record as I
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import clang

    original, sidecar, built, recipe = compiled_shim(tmp_path, monkeypatch)
    reference = tmp_path / "reference.c"
    reference.write_text('''#include <stdint.h>
void original_kernel(const int8_t *a, const int8_t *b, int32_t *c) {
  for (int m=0;m<5;++m) for (int n=0;n<6;++n) {
    int32_t sum=0;
    for (int k=0;k<8;++k) sum+=(int32_t)a[m*8+k]*(int32_t)b[k*6+n];
    c[m*6+n]=sum;
  }
}''')
    library = tmp_path / "original_shim.so"
    linked = I.run([clang(), "-shared", "-fPIC", str(reference), str(built.shim_object), "-o", str(library)],
                   directory=tmp_path / "link", stage="public_test_reference_shim_link",
                   inputs=(reference, built.shim_object), outputs=(library,), capture_output=True, timeout=30)
    assert linked.returncode == 0, linked.stderr
    shim = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    assert shim is not None
    source_text = source().split("func.func @forward", 1)[0] + f'''
    func.func @forward(%a: tensor<4x5x{8 * stride}xi8>, %w: tensor<8x6xi8>)->tensor<5x6xi32> {{
      %s=tensor.extract_slice %a[1,0,0] [1,5,8] [1,1,{stride}]
        : tensor<4x5x{8 * stride}xi8> to tensor<1x5x8xi8>
      %f=tensor.collapse_shape %s [[0,1,2]] : tensor<1x5x8xi8> into tensor<40xi8>
      %v=tensor.expand_shape %f [[0,1]] output_shape [5,8] : tensor<40xi8> into tensor<5x8xi8>
      %e=tensor.empty():tensor<5x6xi32>
      %r=func.call @writer(%v,%w,%e):(tensor<5x8xi8>,tensor<8x6xi8>,tensor<5x6xi32>)->tensor<5x6xi32>
      return %r:tensor<5x6xi32>
    }} }}'''
    if stride == 2:
        # Keep the strided slice itself, rather than flattening a physically
        # noncontiguous memref through an unrelated upstream collapse path.
        start = source_text.index("        : tensor<4x5x16xi8> to tensor<1x5x8xi8>")
        end = source_text.index("      %e=", start)
        source_text = source_text[:start] + "        : tensor<4x5x16xi8> to tensor<5x8xi8>\n" + source_text[end:]
        source_text = source_text.replace("@writer(%v,%w,%e)", "@writer(%s,%w,%e)")
    source_text = source_text.replace("func.func @forward(", f"func.func @forward_{stride}(")
    source_text = source_text.replace(")->tensor<5x6xi32> {", ")->tensor<5x6xi32> attributes {llvm.emit_c_interface} {")
    original.write_text(source_text)
    selected = prepare(original, sidecar, built, tmp_path / "views", recipe)
    layout = json.loads((tmp_path / "views/selection.json").read_text())
    assert layout["arguments"]["writer"] == ([0, 1] if stride == 1 else [1])
    a = (np.arange(4 * 5 * 8 * stride) % 19 - 9).astype(np.int8).reshape(4, 5, 8 * stride)
    w = (np.arange(48) % 11 - 5).astype(np.int8).reshape(8, 6)
    expected = a[1, :, ::stride].astype(np.int64) @ w.astype(np.int64)
    counts = []
    for name, text in (("control", original.read_text()), ("views", selected.read_text())):
        result = lower_model(text, tmp_path / ("host_" + name))
        output = np.full((5, 6), -999, np.int32)
        model = HostModel.load(str(result.host_so), name=f"forward_{stride}",
                               image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
        model([(value.ctypes.data, value.shape) for value in (a, w, output)])
        np.testing.assert_array_equal(output, expected)
        llvm = result.ll_path.read_text()
        counts.append((len(re.findall(r"\bcall\b[^\n]*@malloc\(", llvm)),
                       len(re.findall(r"\bcall\b[^\n]*@llvm.memcpy", llvm))))
    if stride == 1:
        assert counts[1][0] < counts[0][0] and counts[1][1] < counts[0][1]
    else:
        assert counts[1] == counts[0]  # unproven physical contiguity retains ordinary copies


def test_one_strided_observer_keeps_original_layout_for_every_call(tmp_path, monkeypatch):
    """A callee ABI is shared by all original calls, including different source views."""
    original, sidecar, built, recipe = compiled_shim(tmp_path, monkeypatch)
    text = source().split("func.func @forward", 1)[0] + '''
      func.func @forward(%dense: tensor<5x8xi8>, %strided: tensor<5x16xi8>,
                         %w: tensor<8x6xi8>) -> (tensor<5x6xi32>, tensor<5x6xi32>) {
        %view = tensor.extract_slice %strided[0, 0] [5, 8] [1, 2]
          : tensor<5x16xi8> to tensor<5x8xi8>
        %e0 = tensor.empty() : tensor<5x6xi32>
        %e1 = tensor.empty() : tensor<5x6xi32>
        %r0 = func.call @writer(%dense, %w, %e0)
          : (tensor<5x8xi8>, tensor<8x6xi8>, tensor<5x6xi32>) -> tensor<5x6xi32>
        %r1 = func.call @writer(%view, %w, %e1)
          : (tensor<5x8xi8>, tensor<8x6xi8>, tensor<5x6xi32>) -> tensor<5x6xi32>
        return %r0, %r1 : tensor<5x6xi32>, tensor<5x6xi32>
      }
    }'''
    original.write_text(text)
    selected = prepare(original, sidecar, built, tmp_path / "views", recipe)
    report = json.loads((selected.parent / "selection.json").read_text())
    assert report["arguments"] == {"writer": [1]}
    declaration = parse_mlir_text(selected.read_text()).body.block.first_op
    assert "bufferization.buffer_layout" not in declaration.arg_attrs.data[0].data
    assert "bufferization.buffer_layout" not in declaration.arg_attrs.data[2].data
    assert original.read_text() == text
