"""Execute exact integer C against independent source semantics and fault cases."""

import ctypes
import os
import shutil
import subprocess
from dataclasses import replace
from itertools import count
from itertools import product as cartesian_product
from math import prod

import numpy as np
import pytest

from merlin.compile.c_integer import INTEGER_OPERATIONS, emit_integer_c
from merlin.compile.c_sources import c_source_pair
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def request_for(operation, shape=(2, 3, 5), axis=1, boolean=False):
    integer = TensorType(shape, "i64", "signed-i64")
    source = SemanticNode("x", "input", (), TensorType(shape, "i1", "boolean") if boolean else integer, effect="input")
    operands = [source]
    output = integer
    if operation == "i64_reduce_add":
        output = replace(integer, shape=shape[:axis] + shape[axis + 1 :])
        operands.append(SemanticNode("seed", "input", (), output, effect="input"))
        projection = IndexMap(
            len(shape),
            tuple(tuple(int(i == j) for j in range(len(shape))) for i in range(len(shape)) if i != axis),
            (0,) * (len(shape) - 1),
        )
        maps = (pointwise_identity_maps(len(shape), 1)[0], projection, projection)
        attrs = {
            "axis": axis,
            "minimum_index_bits": max(2, max(shape).bit_length() + 1),
            "numeric_semantics": "i64-add-modulo-2pow64-v1",
        }
    elif operation in {"i64_min_value", "i64_first_min_index", "i64_cumsum"}:
        if operation != "i64_cumsum":
            output = replace(integer, shape=shape[:axis] + shape[axis + 1:])
        maps = ()
        attrs = {
            "axis": axis,
            "minimum_index_bits": max(2, max(shape).bit_length() + 1),
            "numeric_semantics": ("signed-i64-prefix-sum-wrap-v1" if operation == "i64_cumsum"
                                  else "signed-i64-min-first-index-v1"),
        }
    else:
        if operation.startswith("signed_"):
            output = TensorType(shape, operation[-3:], "raw-bit-pattern")
            policy = f"signed-i64-to-{output.dtype}-rne-v1"
        else:
            operands.append(SemanticNode("y", "input", (), integer, effect="input"))
            policy = f"signed-i64-{operation[4:]}-wrap-v1"
        maps = pointwise_identity_maps(len(shape), len(operands))
        attrs = {"numeric_semantics": policy}
    node = SemanticNode(
        "out", operation, tuple(n.id for n in operands), output, tuple(sorted(attrs.items())), index_maps=maps
    )
    return KernelRequest(
        (*operands, node), ("out",), ("external",), tuple((n.id, "external") for n in operands), "synthetic"
    )


_libraries = count()


def compile_c(directory, files):
    # Passed-test directories are removed and reused while dlopen retains handles.
    library = f"integer_{next(_libraries)}.so"
    for name, text in files.items():
        (directory / name).write_text(text)
    command = [
        shutil.which("cc"),
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-shared",
        "-fPIC",
        "-fno-sanitize-recover=all",
        "-fsanitize=undefined",
        "integer.c",
        "-o",
        library,
    ]
    subprocess.run(command, cwd=directory, check=True, capture_output=True)
    if cross := os.environ.get("MERLIN_RISCV_GCC"):
        subprocess.run(
            [
                cross,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O2",
                "-march=rv64gcv",
                "-mabi=lp64d",
                "-mstrict-align",
                "-c",
                "integer.c",
                "-o",
                "integer.o",
            ],
            cwd=directory,
            check=True,
            capture_output=True,
        )
    function = ctypes.CDLL(str(directory / library)).integer
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    return function


@pytest.mark.parametrize("operation", INTEGER_OPERATIONS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_integer_execution_preserves_rounding_wrap_seeds_and_unaligned_guards(tmp_path, operation, axis):
    request = request_for(operation, axis=axis)
    _, files = emit_integer_c(request, "integer")
    function = compile_c(tmp_path, files)
    for panel in range(3):
        values = {}
        for operand in request.nodes[:-1]:
            rng = np.random.default_rng(200 + panel + len(values))
            data = rng.integers(-(1 << 63), (1 << 63) - 1, size=prod(operand.type.shape), dtype="i8")
            edges = [-(1 << 63), (1 << 63) - 1, -1, 0, 1]
            data[: min(len(edges), len(data))] = edges[: len(data)]
            values[operand.id] = TensorValue(operand.type, tuple(map(int, data)))
        expected = evaluate_graph(request, values)[0].elements
        check_call(function, request, values, expected)


def check_call(function, request, values, expected):
    payloads = [
        b"".join((v % (1 << 64)).to_bytes(8, "little") for v in values[n.id].elements) for n in request.nodes[:-1]
    ]
    storage = [ctypes.create_string_buffer(b"\xcd" + data + b"\xcd") for data in payloads]
    inputs = (ctypes.c_void_p * len(storage))(*(ctypes.addressof(item) + 1 for item in storage))
    width = 4 if request.nodes[-1].type.dtype == "f32" else 8
    encoded = b"".join((v % (1 << (width * 8))).to_bytes(width, "little") for v in expected)
    output = ctypes.create_string_buffer(b"\xa7" * (len(encoded) + 2))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
    assert function(None, inputs, outputs) == 0
    assert output.raw[1 : len(encoded) + 1] == encoded
    assert output.raw[0] == output.raw[len(encoded) + 1] == 0xA7
    for item, data in zip(storage, payloads, strict=True):
        assert item.raw[:-1] == b"\xcd" + data + b"\xcd"
    assert function(None, inputs, (ctypes.c_void_p * 1)(inputs[0])) == -1
    assert function(None, None, outputs) == -1


@pytest.mark.parametrize("dtype,fraction", [("f32", 23), ("f64", 52)])
def test_integer_conversion_covers_midpoints_neighbors_carry_and_both_signs(tmp_path, dtype, fraction):
    values = [-(1 << 63), (1 << 63) - 1, -1, 0, 1]
    for exponent in range(fraction + 1, 63):
        halfway = 1 << (exponent - fraction - 1)
        for low in [0, 1, (1 << fraction) - 1]:
            base = (1 << exponent) + (low << (exponent - fraction))
            for offset in [-1, 0, 1]:
                value = base + halfway + offset
                if value < 1 << 63:
                    values.extend([value, -value])
    request = request_for("signed_i64_to_" + dtype, shape=(len(values),))
    _, files = emit_integer_c(request, "integer")
    function = compile_c(tmp_path, files)
    arguments = {"x": TensorValue(request.nodes[0].type, tuple(values))}
    check_call(function, request, arguments, evaluate_graph(request, arguments)[0].elements)


def test_boolean_scan_checks_every_input_before_writing_and_resets_each_lane(tmp_path):
    request = request_for("i64_cumsum", boolean=True)
    _, files = emit_integer_c(request, "integer")
    function = compile_c(tmp_path, files)
    data = np.random.default_rng(24).integers(0, 2, size=(2, 3, 5), dtype="u1")
    source = ctypes.create_string_buffer(data.tobytes())
    output = ctypes.create_string_buffer(b"\xa7" * (30 * 8))
    inputs, outputs = (ctypes.c_void_p * 1)(ctypes.addressof(source)), (ctypes.c_void_p * 1)(ctypes.addressof(output))
    assert function(None, inputs, outputs) == 0
    assert output.raw[:-1] == np.cumsum(data, axis=1, dtype="<i8").tobytes()
    source[29] = b"\x02"
    output.raw = b"\xa7" * len(output)
    assert function(None, inputs, outputs) == -2
    assert output.raw == b"\xa7" * len(output)


def test_scalar_reduction_result_retains_nonzero_seed_and_modular_overflow(tmp_path):
    request = request_for("i64_reduce_add", shape=(3,), axis=0)
    _, files = emit_integer_c(request, "integer")
    function = compile_c(tmp_path, files)
    values = {
        "x": TensorValue(request.nodes[0].type, ((1 << 63) - 1, 1, -1)),
        "seed": TensorValue(request.nodes[1].type, (1,)),
    }
    check_call(function, request, values, (-(1 << 63),))


@pytest.mark.parametrize("operation", ["i64_min_value", "i64_first_min_index"])
def test_minimum_first_index_ties_and_bounded_exhaustive_rows(tmp_path, operation):
    rows = list(cartesian_product(range(-2, 3), repeat=4))
    rows += [((1 << 63) - 1, -(1 << 63), -(1 << 63), 0), (7, 7, 7, 7)]
    source = request_for(operation, shape=(len(rows), 4), axis=1)
    _, files = emit_integer_c(source, "integer")
    function = compile_c(tmp_path, files)
    values = {"x": TensorValue(source.nodes[0].type, tuple(value for row in rows for value in row))}
    # Independent finite oracle: all 625 length-four rows over {-2,-1,0,1,2}.
    expected = tuple(min(row) if operation == "i64_min_value" else row.index(min(row)) for row in rows)
    check_call(function, source, values, expected)


@pytest.mark.parametrize("operation", ["i64_min_value", "i64_first_min_index"])
def test_minimum_scalar_result_and_changed_attributes(tmp_path, operation):
    source = request_for(operation, shape=(3,), axis=0)
    _, files = emit_integer_c(source, "integer")
    function = compile_c(tmp_path, files)
    values = {"x": TensorValue(source.nodes[0].type, (-(1 << 63), -(1 << 63), (1 << 63) - 1))}
    check_call(function, source, values, (-(1 << 63) if operation == "i64_min_value" else 0,))
    for wrong in (replace(source.nodes[-1], attrs=()),
                  replace(source.nodes[-1], index_maps=pointwise_identity_maps(1, 1)),
                  replace(source.nodes[-1], type=TensorType((1,), "i64", "signed-i64"))):
        with pytest.raises(ValueError):
            emit_integer_c(replace(source, nodes=(*source.nodes[:-1], wrong)), "invalid")


def test_integer_codegen_rejects_changed_policy_axis_type_seed_and_address_map():
    source = request_for("i64_reduce_add")
    node = source.nodes[-1]
    for altered in [
        replace(node, attrs=node.attrs + (("extra", True),)),
        replace(node, attrs=tuple((key, True if key == "axis" else value) for key, value in node.attrs)),
        replace(node, index_maps=(node.index_maps[0],) * 3),
        replace(node, type=replace(node.type, numerical_policy="unsigned-i64")),
    ]:
        with pytest.raises(ValueError):
            emit_integer_c(replace(source, nodes=(*source.nodes[:-1], altered)), "integer")


def test_two_integer_regions_combine_without_private_symbol_collisions(tmp_path):
    _, first = emit_integer_c(request_for("signed_i64_to_f32", shape=(1,)), "load")
    _, second = emit_integer_c(request_for("signed_i64_to_f64", shape=(1,)), "store")
    files = c_source_pair(
        "integer",
        public_headers={"load.h": first["load.h"], "store.h": second["store.h"]},
        private_headers={},
        sources={"first.c": first["load.c"], "second.c": second["store.c"]},
        implementation="""int integer(void *context,const void *const *inputs,void *const *outputs) {
          int status=load(context,inputs,outputs);
          return status ? status : store(context,inputs,outputs+1);
        }""",
    )
    function = compile_c(tmp_path, files)
    source = ctypes.c_int64((1 << 24) + 1)
    first_out, second_out = ctypes.c_uint32(), ctypes.c_uint64()
    assert (
        function(
            None,
            (ctypes.c_void_p * 1)(ctypes.addressof(source)),
            (ctypes.c_void_p * 2)(ctypes.addressof(first_out), ctypes.addressof(second_out)),
        )
        == 0
    )
    assert first_out.value == 0x4B800000
    assert second_out.value == 0x4170000010000000
