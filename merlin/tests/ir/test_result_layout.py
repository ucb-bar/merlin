"""Native result descriptors preserve frontend storage semantics."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.result_layout import module_attribute
from merlin.llvmlower.toolchain import clang, llvm_install, m2m_python


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


def test_expanded_growing_mutation_preserves_prefix_in_fresh_storage(tmp_path):
    original = dict(shape=[2, 5], stride=[0, 1], storage_offset=0)
    grown = dict(
        shape=[3, 5], stride=[5, 1], storage_offset=0, alias_inputs=[0], role="user_input_mutation", input_index=0
    )
    source = (
        "builtin.module"
        + module_attribute(dict(inputs=[original], results=[grown]))
        + """ {
      func.func @forward(%a: tensor<2x5xf32>) -> tensor<3x5xf32>
          attributes {llvm.emit_c_interface} {
        %empty = tensor.empty() : tensor<3x5xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%a : tensor<2x5xf32>) outs(%empty : tensor<3x5xf32>) {
          ^bb0(%value: f32, %unused: f32):
            linalg.yield %value : f32
        } -> tensor<3x5xf32>
        func.return %result : tensor<3x5xf32>
      }
    }"""
    )
    lowered = lower_to_llvm_ir(source, workdir=tmp_path, preserve_result_descriptors=True)
    path, library = tmp_path / "model.ll", tmp_path / "model.so"
    path.write_text(lowered)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    values = np.array([1.25, -2.5, 7.0, 8.5, -9.75], np.float32)
    saved = values.copy()
    arg = Descriptor(values.ctypes.data, values.ctypes.data, 0, (ctypes.c_int64 * 2)(2, 5), (ctypes.c_int64 * 2)(0, 1))
    output = Descriptor()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(arg))
    assert output.allocated != values.ctypes.data
    assert list(output.sizes) == [3, 5] and list(output.strides) == [5, 1] and output.offset == 0
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    np.testing.assert_array_equal(np.ctypeslib.as_array(pointer, shape=(5,)), saved)
    pointer[0] = -17.0
    np.testing.assert_array_equal(values, saved)
    ctypes.CDLL(None).free(ctypes.c_void_p(output.allocated))


def execute(tmp_path, result, *, dynamic=False, extent=4, passthrough=False):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    import hashlib

    tmp_path = tmp_path / hashlib.sha256(str((result, extent)).encode()).hexdigest()[:12]
    tmp_path.mkdir()
    contract = dict(inputs=[dict(shape=[extent, 5], stride=[5, 1], storage_offset=0)], results=[result])
    shape = "?x5" if dynamic else "x".join(str(n) for n in result["shape"])
    body = (
        f"%b = tensor.cast %a : tensor<{extent}x5xf32> to tensor<?x5xf32>"
        if dynamic
        else f"%b = tensor.empty() : tensor<{shape}xf32>"
    )
    if passthrough:
        body = f"%b = tensor.cast %a : tensor<{extent}x5xf32> to tensor<{shape}xf32>"
    source = (
        "builtin.module"
        + module_attribute(contract)
        + f""" {{
      func.func @forward(%a: tensor<{extent}x5xf32>) -> tensor<{shape}xf32>
          attributes {{llvm.emit_c_interface}} {{
        {body}
        func.return %b : tensor<{shape}xf32>
      }}
    }}"""
    )
    lowered = lower_to_llvm_ir(source, workdir=tmp_path, preserve_result_descriptors=True)
    path, library = tmp_path / "model.ll", tmp_path / "model.so"
    path.write_text(lowered)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    values = np.arange(extent * 5, dtype=np.float32).reshape(extent, 5)
    arg = Descriptor(
        values.ctypes.data, values.ctypes.data, 0, (ctypes.c_int64 * 2)(extent, 5), (ctypes.c_int64 * 2)(5, 1)
    )
    output = Descriptor()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(arg))
    return values, output


@pytest.mark.parametrize("shape,stride,offset", [([2, 3], [5, 1], 6), ([5, 4], [1, 5], 0), ([3, 5], [0, 1], 0)])
def test_borrowed_native_view_shares_allocation(tmp_path, shape, stride, offset):
    values, output = execute(
        tmp_path, dict(shape=shape, stride=stride, storage_offset=offset, alias_inputs=[0], role="user_output")
    )
    assert output.allocated == values.ctypes.data
    assert output.aligned == values.ctypes.data
    assert output.offset == offset
    assert list(output.sizes) == shape
    assert list(output.strides) == stride
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    pointer[offset] = -17
    assert values.ravel()[offset] == -17  # A write through the result reaches the input.


@pytest.mark.parametrize("extent", [0, 1, 4])
def test_dynamic_fresh_result_is_physically_column_major(tmp_path, extent):
    values, output = execute(
        tmp_path,
        dict(shape=[-1, 5], stride=[1, {"shape_axes": [0]}], storage_offset=0, alias_inputs=[], role="user_output"),
        dynamic=True,
        extent=extent,
    )
    assert output.allocated != values.ctypes.data
    assert list(output.sizes) == [extent, 5]
    assert list(output.strides) == [1, extent]
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    actual = [[pointer[output.offset + i + extent * j] for j in range(5)] for i in range(extent)]
    np.testing.assert_array_equal(np.asarray(actual).reshape(extent, 5), values)
    ctypes.CDLL(None).free(ctypes.c_void_p(output.allocated))


def test_fresh_identity_result_cannot_borrow_caller_storage(tmp_path):
    values, output = execute(
        tmp_path,
        dict(shape=[4, 5], stride=[5, 1], storage_offset=0, alias_inputs=[], role="user_output"),
        passthrough=True,
    )
    assert output.allocated != values.ctypes.data
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    np.testing.assert_array_equal(np.ctypeslib.as_array(pointer, shape=(20,)).reshape(4, 5), values)
    pointer[0] = -17
    assert values[0, 0] == 0
    ctypes.CDLL(None).free(ctypes.c_void_p(output.allocated))


def test_fresh_transpose_result_cannot_borrow_caller_storage(tmp_path):
    source = """module {
      func.func @forward(%a: memref<2x3xf32>) -> memref<3x2xf32, strided<[1, 3]>>
          attributes {llvm.emit_c_interface} {
        %v = memref.transpose %a (d0, d1) -> (d1, d0)
          : memref<2x3xf32> to memref<3x2xf32, strided<[1, 3]>>
        func.return %v : memref<3x2xf32, strided<[1, 3]>>
      }
    }"""
    contract = dict(
        inputs=[dict(shape=[2, 3], stride=[3, 1], storage_offset=0)],
        results=[dict(shape=[3, 2], stride=[1, 3], storage_offset=0, alias_inputs=[])],
    )
    repaired = apply_memref_layout(source, contract)
    assert repaired.returncode == 0, repaired.stderr
    lowered = lower_to_llvm_ir(repaired.stdout, workdir=tmp_path, preserve_result_descriptors=True)
    path, library = tmp_path / "transpose.ll", tmp_path / "transpose.so"
    path.write_text(lowered)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    argument = Descriptor(
        values.ctypes.data, values.ctypes.data, 0, (ctypes.c_int64 * 2)(2, 3), (ctypes.c_int64 * 2)(3, 1)
    )
    output = Descriptor()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(argument))
    assert output.allocated != values.ctypes.data
    assert list(output.sizes) == [3, 2]
    assert list(output.strides) == [1, 3]
    pointer = ctypes.cast(output.aligned, ctypes.POINTER(ctypes.c_float))
    actual = np.array([[pointer[i + 3 * j] for j in range(2)] for i in range(3)])
    np.testing.assert_array_equal(actual, values.T)
    pointer[0] = -17
    assert values[0, 0] == 0
    ctypes.CDLL(None).free(ctypes.c_void_p(output.allocated))


def test_fresh_alias_class_covers_later_full_storage(tmp_path):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")

    class Pair(ctypes.Structure):
        _fields_ = [("partial", Descriptor), ("full", Descriptor)]

    contract = dict(
        inputs=[dict(shape=[4, 5], stride=[5, 1], storage_offset=0)],
        results=[
            dict(shape=[2, 5], stride=[5, 1], storage_offset=5, alias_inputs=[]),
            dict(shape=[4, 5], stride=[5, 1], storage_offset=0, alias_inputs=[], alias_results=[0]),
        ],
    )
    source = (
        "builtin.module"
        + module_attribute(contract)
        + """ {
      func.func @forward(%a: tensor<4x5xf32>) -> (tensor<2x5xf32>, tensor<4x5xf32>)
          attributes {llvm.emit_c_interface} {
        %b = "tensor.extract_slice"(%a) <{static_offsets = array<i64: 1, 0>,
          static_sizes = array<i64: 2, 5>, static_strides = array<i64: 1, 1>,
          operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<4x5xf32>) -> tensor<2x5xf32>
        func.return %b, %a : tensor<2x5xf32>, tensor<4x5xf32>
      }
    }"""
    )
    lowered = lower_to_llvm_ir(source, workdir=tmp_path, preserve_result_descriptors=True)
    path, library = tmp_path / "pair.ll", tmp_path / "pair.so"
    path.write_text(lowered)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    values = np.arange(20, dtype=np.float32)
    arg = Descriptor(values.ctypes.data, values.ctypes.data, 0, (ctypes.c_int64 * 2)(4, 5), (ctypes.c_int64 * 2)(5, 1))
    output = Pair()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Pair), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(arg))
    assert output.partial.allocated == output.full.allocated != values.ctypes.data
    assert output.partial.offset == 5 and output.full.offset == 0
    pointer = ctypes.cast(output.full.aligned, ctypes.POINTER(ctypes.c_float))
    np.testing.assert_array_equal([pointer[i] for i in range(20)], values)
    pointer[output.partial.offset] = -31
    assert ctypes.cast(output.partial.aligned, ctypes.POINTER(ctypes.c_float))[5] == -31
    ctypes.CDLL(None).free(ctypes.c_void_p(output.full.allocated))


def apply_memref_layout(source, contract, *, operation_owners=False):
    """Exercise the ownership rewrite in its native compiler interpreter."""
    import json

    code = """
import json, sys
from torch_mlir import ir
from torch_mlir.dialects import arith, memref
from merlin.llvmlower.result_layout import _apply_result_layout
source, contract, operation_owners = json.load(sys.stdin)
# Some compiler bindings expose bare Operation owners. Exercise that branch
# alongside native OpView owners without changing the parsed IR or its values.
if operation_owners:
    ir.Operation = (ir.Operation, ir.OpView)
with ir.Context() as ctx:
    module = ir.Module.parse(source)
    _apply_result_layout(module, contract)
    print(module)
"""
    return subprocess.run(
        [str(m2m_python()), "-c", code],
        input=json.dumps([source, contract, operation_owners]),
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("operation_owners", [False, True])
@pytest.mark.parametrize("producer", ["view", "call", "alloc", "prior_unknown", "prior_transpose"])
def test_fresh_result_reuse_requires_known_disjoint_allocation(producer, operation_owners):
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    target = "memref<2x3xf32>"
    declarations = f"""func.func private @alias(%x: {target}) -> {target} {{ return %x : {target} }}"""
    results = [dict(record, alias_inputs=[])]
    if producer == "view":
        input_type = "memref<24xi8>"
        inputs = [dict(shape=[24], stride=[1], storage_offset=0)]
        body = f"%zero = arith.constant 0 : index\n%value = memref.view %arg[%zero][] : {input_type} to {target}"
        returned, return_type = "%value", target
    else:
        input_type = target
        inputs = [record]
        if producer == "call":
            body = f"%value = func.call @alias(%arg) : ({target}) -> {target}"
            returned, return_type = "%value", target
        else:
            body = f"%owned = memref.alloc() : {target}"
            if producer == "alloc":
                returned, return_type = "%owned", target
            else:
                if producer == "prior_unknown":
                    live_type = target
                    body += f"\n%live = func.call @alias(%owned) : ({target}) -> {target}"
                else:
                    live_type = "memref<3x2xf32, strided<[1, 3]>>"
                    body += f"\n%live = memref.transpose %owned (d0, d1) -> (d1, d0) : {target} to {live_type}"
                returned, return_type = "%live, %owned", f"{live_type}, {target}"
                results.insert(0, None)
    source = f"""module {{
      {declarations}
      func.func @forward(%arg: {input_type}) -> ({return_type}) {{
        {body}
        return {returned} : {return_type}
      }}
    }}"""
    result = apply_memref_layout(source, dict(inputs=inputs, results=results), operation_owners=operation_owners)
    assert result.returncode == 0, result.stderr
    assert ("memref.copy" in result.stdout) is (producer != "alloc")
    if producer.startswith("prior_"):
        assert result.stdout.count("memref.alloc(") == 2
        assert "memref.dealloc" not in result.stdout


@pytest.mark.parametrize("operation_owners", [False, True])
@pytest.mark.parametrize("producer", ["transpose", "view", "alloc", "argument"])
def test_discarded_alloc_is_retained_when_live_root_is_unknown(producer, operation_owners):
    if producer == "view":
        input_type = "memref<24xi8>"
        live_type = "memref<2x3xf32>"
        body = (
            "%zero = arith.constant 0 : index\n%live = memref.view %owned[%zero][] : memref<24xi8> to memref<2x3xf32>"
        )
        input_shape, input_stride = [24], [1]
    else:
        input_type = "memref<2x3xf32>"
        input_shape, input_stride = [2, 3], [3, 1]
        if producer == "transpose":
            live_type = "memref<3x2xf32, strided<[1, 3]>>"
            body = "%live = memref.transpose %owned (d0, d1) -> (d1, d0) : memref<2x3xf32> to " + live_type
        elif producer == "alloc":
            live_type = input_type
            body = "%live = memref.alloc() : " + live_type
        else:
            live_type = input_type
            body = "%live = memref.cast %arg : memref<2x3xf32> to memref<2x3xf32>"
    source = f"""module {{
      func.func @forward(%arg: {input_type}) -> ({input_type}, {live_type}) {{
        %owned = memref.alloc() : {input_type}
        {body}
        return %owned, %live : {input_type}, {live_type}
      }}
    }}"""
    record = dict(shape=input_shape, stride=input_stride, storage_offset=0)
    result = apply_memref_layout(
        source, dict(inputs=[record], results=[dict(record, alias_inputs=[0]), None]), operation_owners=operation_owners
    )
    assert result.returncode == 0, result.stderr
    assert ("memref.dealloc" in result.stdout) == (producer in {"alloc", "argument"})
    # The original owned allocation remains the backing storage of the unknown view.
    if producer in {"transpose", "view"}:
        assert "memref.alloc" in result.stdout and "memref." + producer in result.stdout


@pytest.mark.parametrize("unknown", ["shape", "stride", "offset", "capacity", "compiled_result", "compiled_input"])
def test_borrowed_view_refuses_unproven_span(unknown):
    input_record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0, alias_inputs=[0])
    input_type = "memref<2x3xf32>"
    result_type = "memref<2x3xf32>"
    if unknown == "shape":
        record["shape"] = [-1, 3]
        result_type = "memref<?x3xf32>"
    elif unknown == "stride":
        record["stride"] = [{"shape_axes": [1]}, 1]
    elif unknown == "offset":
        record["storage_offset"] = {"shape_axes": [0]}
    elif unknown == "capacity":
        input_record["shape"] = [-1, 3]
        input_type = "memref<?x3xf32>"
    elif unknown == "compiled_result":
        result_type = "memref<?x3xf32>"
    else:
        input_type = "memref<?x3xf32>"
    source = f"""module {{
      func.func @forward(%arg: {input_type}, %value: {result_type}) -> {result_type} {{
        return %value : {result_type}
      }}
    }}"""
    other_input = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    result = apply_memref_layout(source, dict(inputs=[input_record, other_input], results=[record]))
    assert result.returncode != 0
    assert "cannot prove returned view fits" in result.stderr


def test_borrowed_view_refuses_static_out_of_bounds_span():
    source = """module {
      func.func @forward(%arg: memref<2x3xf32>, %value: memref<2x3xf32>) -> memref<2x3xf32> {
        return %value : memref<2x3xf32>
      }
    }"""
    inputs = [dict(shape=[2, 3], stride=[3, 1], storage_offset=0)] * 2
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=1, alias_inputs=[0])
    result = apply_memref_layout(source, dict(inputs=inputs, results=[record]))
    assert result.returncode != 0
    assert "exceeds its input storage envelope" in result.stderr


@pytest.mark.parametrize("dynamic", [True, False])
def test_view_of_mutation_result_requires_its_own_capacity_proof(dynamic):
    shape, result_type = ([-1, 3], "memref<?x3xf32>") if dynamic else ([4, 3], "memref<4x3xf32>")
    source = f"""module {{
      func.func @forward(%arg: memref<2x3xf32>, %value: {result_type}) -> (memref<2x3xf32>, {result_type}) {{
        return %arg, %value : memref<2x3xf32>, {result_type}
      }}
    }}"""
    original = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    mutation = dict(original, alias_inputs=[0], role="user_input_mutation", input_index=0)
    view = dict(shape=shape, stride=[3, 1], storage_offset=0, alias_inputs=[0], role="user_output")
    result = apply_memref_layout(source, dict(inputs=[original, dict(original, shape=shape)], results=[mutation, view]))
    assert result.returncode != 0
    assert ("cannot prove returned view fits" if dynamic else "exceeds its input storage envelope") in result.stderr


def test_dynamic_mutation_cannot_supply_an_unproven_borrowed_view():
    source = """module {
      func.func @forward(%arg: memref<2x3xf32>, %value: memref<?x3xf32>) -> (memref<?x3xf32>, memref<2x3xf32>) {
        return %value, %arg : memref<?x3xf32>, memref<2x3xf32>
      }
    }"""
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    mutation = dict(record, alias_inputs=[0], role="user_input_mutation", input_index=0)
    view = dict(record, alias_inputs=[0], role="user_output")
    result = apply_memref_layout(source, dict(inputs=[record, record], results=[mutation, view]))
    assert result.returncode != 0
    assert "cannot prove returned view fits" in result.stderr


def test_result_alias_cannot_bypass_borrowed_capacity_proof():
    source = """module {
      func.func @forward(%arg: memref<2x3xf32>, %value: memref<?x3xf32>) -> (memref<2x3xf32>, memref<?x3xf32>) {
        return %arg, %value : memref<2x3xf32>, memref<?x3xf32>
      }
    }"""
    record = dict(shape=[2, 3], stride=[3, 1], storage_offset=0)
    view = dict(record, shape=[-1, 3], alias_results=[0])
    result = apply_memref_layout(source, dict(inputs=[record, record], results=[dict(record, alias_inputs=[0]), view]))
    assert result.returncode != 0
    assert "cannot prove returned view fits" in result.stderr


@pytest.mark.parametrize("stride", [[1, 3], [0, 1]])
def test_materialized_borrowed_result_is_copied_before_return(stride):
    source = """module {
      func.func @forward(%arg: memref<2x3xi64>, %value: memref<3x2xi64>) -> memref<3x2xi64> {
        return %value : memref<3x2xi64>
      }
    }"""
    inputs = [dict(shape=[2, 3], stride=[3, 1], storage_offset=0), dict(shape=[3, 2], stride=[2, 1], storage_offset=0)]
    record = dict(shape=[3, 2], stride=stride, storage_offset=0, alias_inputs=[0], materialized_byte_selection=True)
    # A transpose needs a 3x2 descriptor with strides [1, 3]; zero stride tests
    # repeated addresses without asserting disjointness or changing the alias.
    result = apply_memref_layout(source, dict(inputs=inputs, results=[record]))
    assert result.returncode == 0, result.stderr
    assert "memref.copy" in result.stdout
    assert result.stdout.index("memref.copy") < result.stdout.index("return ")
    assert "memref.reinterpret_cast %arg0" in result.stdout


@pytest.mark.parametrize("strides,offset", [([5, 1], 0), ([1, 3], 2), ([0, 1], 1)])
def test_captured_inputs_with_output_parameters(tmp_path, strides, offset):
    """Out-params copy values while loads retain the captured physical layout."""
    tmp_path = tmp_path / ("layout-" + "-".join(map(str, [*strides, offset])))
    tmp_path.mkdir()
    input_record = dict(shape=[3, 5], stride=strides, storage_offset=offset)
    output_record = dict(shape=[3, 5], stride=[5, 1], storage_offset=0, alias_inputs=[])
    source = (
        "builtin.module"
        + module_attribute(dict(inputs=[input_record], results=[output_record]))
        + """ {
      func.func @forward(%a: tensor<3x5xf32>) -> tensor<3x5xf32>
          attributes {llvm.emit_c_interface} {
        %empty = tensor.empty() : tensor<3x5xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%a : tensor<3x5xf32>) outs(%empty : tensor<3x5xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %twice = arith.addf %value, %value : f32
            linalg.yield %twice : f32
        } -> tensor<3x5xf32>
        func.return %result : tensor<3x5xf32>
      }
    }"""
    )
    path, library = tmp_path / "model.ll", tmp_path / "model.so"
    path.write_text(lower_to_llvm_ir(source, workdir=tmp_path))
    subprocess.run(
        [str(clang()), "-shared", "-fPIC", "-O1", str(path), "-o", str(library)], check=True, capture_output=True
    )
    backing = np.arange(64, dtype=np.float32) - 7
    saved = backing.copy()
    output = np.full((3, 5), np.nan, np.float32)
    arg = Descriptor(
        backing.ctypes.data, backing.ctypes.data, offset, (ctypes.c_int64 * 2)(3, 5), (ctypes.c_int64 * 2)(*strides)
    )
    dest = Descriptor(output.ctypes.data, output.ctypes.data, 0, (ctypes.c_int64 * 2)(3, 5), (ctypes.c_int64 * 2)(5, 1))
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(arg), ctypes.byref(dest))
    expected = np.array(
        [[2 * saved[offset + i * strides[0] + j * strides[1]] for j in range(5)] for i in range(3)], np.float32
    )
    assert output.tobytes() == expected.tobytes()
    assert backing.tobytes() == saved.tobytes()
