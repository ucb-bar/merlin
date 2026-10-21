"""Compiled predicates preserve IEEE truth values, projections and failure atomicity."""

import ctypes
import os
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.c_predicates import emit_predicate_c
from merlin.semantic_compiler.model import (
    ConstantBinding,
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)


def compile_c(directory, files, symbol, tag):
    for filename, text in files.items():
        (directory / filename).write_text(text)
    library = f"{tag}.so"
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            symbol + ".c",
            "-o",
            library,
        ],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    if cross := os.environ.get("MERLIN_RISCV_GCC"):
        subprocess.run(
            [
                cross,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffreestanding",
                "-march=rv64gcv",
                "-mabi=lp64d",
                "-mstrict-align",
                "-c",
                symbol + ".c",
                "-o",
                tag + ".rv64.o",
            ],
            cwd=directory,
            check=True,
            capture_output=True,
        )
    function = getattr(ctypes.CDLL(str(directory / library)), symbol)
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    return function


def request_for(kind):
    truth = kind == "nonzero"
    integer = kind in {"eq", "ne", "slt", "sle", "sgt", "sge"}
    output = TensorType((3, 4), "i64" if kind == "extui" else "i1", "signed-i64" if kind == "extui" else "boolean")
    dtype, policy = ("f32", "raw-bit-pattern") if truth else ("i64", "signed-i64") if integer else ("i1", "boolean")
    operands = tuple(
        SemanticNode(f"x{i}", "input", (), TensorType(shape, dtype, policy), effect="input")
        for i, shape in enumerate(((3, 4),) if truth else ((3, 1),) if kind == "extui" else ((3, 1), (1, 4)))
    )
    maps = (
        pointwise_identity_maps(2, 1)
        if truth
        else (
            IndexMap(2, ((1, 0), (0, 0)), (0, 0)),
            *((IndexMap(2, ((0, 0), (0, 1)), (0, 0)),) if len(operands) == 2 else ()),
            pointwise_identity_maps(2, 1)[0],
        )
    )
    operation = "arith.cmpi" if integer else "arith." + kind
    attrs = (
        (("numeric_semantics", "f32-une-zero-v1"),)
        if truth
        else (("operation", operation), ("predicate", kind if integer else None))
    )
    node = SemanticNode(
        "out", "f32_nonzero" if truth else "mask", tuple(n.id for n in operands), output, attrs, index_maps=maps
    )
    return KernelRequest(
        (*operands, node), ("out",), ("external",), tuple((n.id, "external") for n in operands), "synthetic"
    )


@pytest.mark.parametrize("kind", ["nonzero", "eq", "ne", "slt", "sle", "sgt", "sge", "andi", "xori", "extui"])
def test_compiled_predicate_matches_independent_numpy_and_preserves_buffers(tmp_path, kind):
    request = request_for(kind)
    if kind == "nonzero":
        request = replace(request, nodes=tuple(replace(n, type=replace(n.type, shape=(17, 31))) for n in request.nodes))
    _, files = emit_predicate_c(request, "predicate")
    function = compile_c(tmp_path, files, "predicate", "predicate_" + kind)
    integer = kind in {"eq", "ne", "slt", "sle", "sgt", "sge"}
    for panel in range(3):
        if kind == "nonzero":
            edges = np.array(
                [
                    0,
                    0x80000000,
                    1,
                    0x80000001,
                    0x7F800000,
                    0xFF800000,
                    0x7FC12345,
                    0xFF812345,
                    0x3F800000,
                    0xBF800000,
                    0x00800000,
                    0x80800000,
                ],
                dtype="<u4",
            )
            random = np.random.default_rng(230 + panel).integers(0, 1 << 32, size=515, dtype="u4")
            bits = np.concatenate((edges, random)).reshape(17, 31)
            values = [bits]
            with np.errstate(invalid="ignore"):
                expected = bits.view("<f4") != 0
        elif integer:
            values = [
                np.roll(np.array([-(1 << 63), -1, 0], dtype="<i8"), panel).reshape(3, 1),
                np.roll(np.array([0, 1, (1 << 63) - 1, -(1 << 63)], dtype="<i8"), panel).reshape(1, 4),
            ]
            expected = {
                "eq": np.equal,
                "ne": np.not_equal,
                "slt": np.less,
                "sle": np.less_equal,
                "sgt": np.greater,
                "sge": np.greater_equal,
            }[kind](*values)
        else:
            values = [np.roll(np.array([0, 1, 0], dtype="u1"), panel).reshape(3, 1)]
            if kind == "extui":
                expected = np.broadcast_to(values[0], (3, 4))
            else:
                values.append(np.roll(np.array([0, 1, 1, 0], dtype="u1"), panel).reshape(1, 4))
                expected = (np.bitwise_and if kind == "andi" else np.bitwise_xor)(*values)
        expected = np.asarray(expected, dtype="<i8" if kind == "extui" else "u1").tobytes()
        storage = [ctypes.create_string_buffer(b"\xcd" + value.tobytes() + b"\xcd") for value in values]
        inputs = (ctypes.c_void_p * len(storage))(*(ctypes.addressof(value) + 1 for value in storage))
        output = ctypes.create_string_buffer(b"\xa7" * (len(expected) + 2))
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
        assert function(None, inputs, outputs) == 0
        assert output.raw[1 : len(expected) + 1] == expected
        assert output.raw[0] == output.raw[len(expected) + 1] == 0xA7
        for data, value in zip(storage, values, strict=True):
            assert data.raw[:-1] == b"\xcd" + value.tobytes() + b"\xcd"
        assert function(None, inputs, (ctypes.c_void_p * 1)(inputs[0])) == -1
        if not integer and kind != "nonzero":
            storage[-1][values[-1].nbytes] = b"\x02"
            output.raw = b"\xa7" * len(output)
            assert function(None, inputs, outputs) == -2
            assert output.raw == b"\xa7" * len(output)


def test_predicate_codegen_refuses_changed_semantics_policies_maps_and_physical_views():
    request = request_for("slt")
    original = request.nodes[-1]
    for attrs in ((("operation", "arith.cmpi"), ("predicate", "ult")), original.attrs + (("unknown", True),)):
        with pytest.raises(ValueError):
            emit_predicate_c(replace(request, nodes=(*request.nodes[:-1], replace(original, attrs=attrs))), "predicate")
    with pytest.raises(ValueError):
        changed = replace(request.nodes[0], type=replace(request.nodes[0].type, numerical_policy="unsigned-i64"))
        emit_predicate_c(replace(request, nodes=(changed, *request.nodes[1:])), "predicate")
    with pytest.raises(ValueError):
        changed = replace(
            original, index_maps=(replace(original.index_maps[0], offsets=(1, 0)), *original.index_maps[1:])
        )
        emit_predicate_c(replace(request, nodes=(*request.nodes[:-1], changed)), "predicate")
    with pytest.raises(ValueError):
        emit_predicate_c(
            replace(request, physical_extents=tuple((key, 1) for key, _ in request.input_storages)), "predicate"
        )


@pytest.mark.parametrize(
    "dtype,width", [("bf16", 2), ("fp8_e4m3", 1), ("f32", 4), ("f64", 8), ("i32", 4), ("i64", 8), ("i8", 1), ("i1", 1)]
)
def test_compiled_select_preserves_value_bits_and_checks_boolean_inputs(tmp_path, dtype, width):
    shape = (3, 4)
    policy = "signed-i64" if dtype == "i64" else "boolean" if dtype == "i1" else "raw-bit-pattern"
    type_ = TensorType(shape, dtype, policy)
    condition_type = TensorType(shape, "i1", "boolean")
    inputs = tuple(
        SemanticNode(f"x{i}", "input", (), type_, effect="input")
        for i, type_ in enumerate((condition_type, type_, type_))
    )
    result = SemanticNode("out", "select", tuple(n.id for n in inputs), type_, index_maps=pointwise_identity_maps(2, 3))
    request = KernelRequest(
        (*inputs, result), (result.id,), ("external",), tuple((n.id, "external") for n in inputs), "synthetic"
    )
    _, files = emit_predicate_c(request, "select_value")
    function = compile_c(tmp_path, files, "select_value", "select_" + dtype)
    a = np.arange(12, dtype=f"<u{width}").reshape(shape)
    if dtype == "i1":
        a %= 2
    else:
        a.flat[0] = (1 << (8 * width)) - 1
        a.flat[1] = 1 << (8 * width - 1)
    b = a[::-1].copy()
    for panel in range(3):
        flags = ((np.arange(12) + panel) % 2).astype("u1").reshape(shape)
        values = [flags, a, b]
        storage = [ctypes.create_string_buffer(b"\xcd" + value.tobytes() + b"\xcd") for value in values]
        pointers = (ctypes.c_void_p * 3)(*(ctypes.addressof(value) + 1 for value in storage))
        expected = np.where(flags, a, b).tobytes()
        output = ctypes.create_string_buffer(b"\xa7" * (len(expected) + 2))
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
        assert function(None, pointers, outputs) == 0
        assert output.raw[1 : len(expected) + 1] == expected
        assert output.raw[0] == output.raw[len(expected) + 1] == 0xA7
        for value, original in zip(storage, values, strict=True):
            assert value.raw[:-1] == b"\xcd" + original.tobytes() + b"\xcd"
        storage[0][12] = b"\x02"
        output.raw = b"\xa7" * len(output)
        assert function(None, pointers, outputs) == -2 and output.raw == b"\xa7" * len(output)
    encodings = {
        "bf16": "bf16-le-raw",
        "fp8_e4m3": "fp8_e4m3-raw",
        "f32": "f32-le-raw",
        "f64": "f64-le-raw",
        "i32": "i32-le",
        "i64": "i64-le",
        "i8": "i8",
        "i1": "i1-raw",
    }
    folded_source = replace(
        request,
        nodes=(*tuple(replace(n, op="constant", effect="constant") for n in inputs), result),
        input_storages=(),
        constants=tuple(
            ConstantBinding(n.id, encodings[n.type.dtype], value.tobytes().hex())
            for n, value in zip(inputs, values, strict=True)
        ),
    )
    folded, files = emit_predicate_c(folded_source, "select_constant")
    assert folded["inputs"] == [] and "constant_folding" in folded
    constant = compile_c(tmp_path, files, "select_constant", "constant_select_" + dtype)
    assert constant(None, None, outputs) == 0
    assert output.raw[1 : len(expected) + 1] == expected
    assert output.raw[0] == output.raw[len(expected) + 1] == 0xA7
    with pytest.raises(ValueError):
        bad = replace(inputs[-1], type=replace(type_, numerical_policy="different"))
        emit_predicate_c(replace(request, nodes=(*inputs[:-1], bad, result)), "select_value")


@pytest.mark.parametrize("kind", ["nonzero", "eq", "ne", "slt", "sle", "sgt", "sge", "andi", "xori", "extui"])
def test_declared_constant_predicates_fold_without_runtime_inputs(tmp_path, kind):
    from merlin.semantic_compiler.linalg_constants import read_tensor_bytes
    from merlin.semantic_compiler.reference import evaluate_graph

    original = request_for(kind)
    bindings, values, nodes = [], {}, []
    for operand in original.nodes[:-1]:
        dtype = operand.type.dtype
        count = np.prod(operand.type.shape)
        if dtype == "i64":
            payload = np.resize(np.array([-(1 << 63), -1, 0, (1 << 63) - 1], dtype="<i8"), count).tobytes()
            encoding = "i64-le"
        elif dtype == "f32":
            payload = np.resize(np.array([0, 0x80000000, 1, 0x7FC12345], dtype="<u4"), count).tobytes()
            encoding = "f32-le-raw"
        else:
            payload, encoding = bytes(i % 2 for i in range(count)), "i1-raw"
        nodes.append(replace(operand, op="constant", effect="constant"))
        bindings.append(ConstantBinding(operand.id, encoding, payload.hex()))
        values[operand.id] = read_tensor_bytes(operand.type, payload)[1]
    source = replace(original, nodes=(*nodes, original.nodes[-1]), input_storages=(), constants=tuple(bindings))
    record, files = emit_predicate_c(source, "folded_predicate")
    assert record["inputs"] == [] and "constant_folding" in record
    unfurled, _ = emit_predicate_c(source, "unfurled_predicate", constant_fold_limit=0)
    assert unfurled["inputs"] == list(source.nodes[-1].inputs) and "constant_folding" not in unfurled
    expected = evaluate_graph(source, {}, constants=values)[0]
    width = 8 if kind == "extui" else 1
    payload = b"".join(value.to_bytes(width, "little") for value in expected.elements)
    function = compile_c(tmp_path, files, "folded_predicate", "constant_" + kind)
    output = ctypes.create_string_buffer(b"\xa7" * (len(payload) + 2))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
    assert function(None, None, outputs) == 0
    assert output.raw[1 : len(payload) + 1] == payload
    assert output.raw[0] == output.raw[len(payload) + 1] == 0xA7


@pytest.mark.parametrize("dtype,width,fraction", [("f32", 4, 23), ("f64", 8, 52)])
@pytest.mark.parametrize("predicate", ["oeq", "one", "olt", "ole", "ogt", "oge"])
def test_ordered_float_predicates_and_constant_folding_match_ieee_comparisons(
    tmp_path, dtype, width, fraction, predicate
):
    original = request_for("slt")
    operands = tuple(
        replace(n, type=TensorType(shape, dtype, "raw-bit-pattern"))
        for n, shape in zip(original.nodes[:-1], [(47, 1), (1, 47)], strict=True)
    )
    node = replace(
        original.nodes[-1],
        type=TensorType((47, 47), "i1", "boolean"),
        attrs=(("operation", "arith.cmpf"), ("predicate", predicate)),
    )
    request = replace(original, nodes=(*operands, node))
    _, files = emit_predicate_c(request, "compare")
    function = compile_c(tmp_path, files, "compare", "compare_" + dtype + "_" + predicate)
    sign = 1 << (width * 8 - 1)
    infinity = sign - (1 << fraction)
    positive = [
        0,
        1,
        (1 << fraction) - 1,
        1 << fraction,
        infinity - 1,
        infinity,
        infinity | 1,
        infinity | (1 << (fraction - 1)),
        (127 if width == 4 else 1023) << fraction,
    ]
    edges = np.array([value | s for s in [0, sign] for value in positive], dtype=f"<u{width}")
    for panel in range(3):
        random = np.random.default_rng(470 + panel).integers(0, 1 << (width * 8), size=29, dtype=f"u{width}")
        bits = np.concatenate((edges, random))
        data = [bits.reshape(47, 1), np.roll(bits[::-1], panel).reshape(1, 47).copy()]
        a, b = (value.view(f"<f{width}") for value in data)
        with np.errstate(invalid="ignore"):
            compare = {
                "oeq": np.equal,
                "one": np.not_equal,
                "olt": np.less,
                "ole": np.less_equal,
                "ogt": np.greater,
                "oge": np.greater_equal,
            }[predicate]
            expected = ((~np.isnan(a)) & (~np.isnan(b)) & compare(a, b)).astype("u1").tobytes()
        storage = [ctypes.create_string_buffer(b"\xcd" + value.tobytes() + b"\xcd") for value in data]
        pointers = (ctypes.c_void_p * 2)(*(ctypes.addressof(value) + 1 for value in storage))
        output = ctypes.create_string_buffer(b"\xa7" * (len(expected) + 2))
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
        assert function(None, pointers, outputs) == 0
        assert output.raw[1 : len(expected) + 1] == expected
        assert output.raw[0] == output.raw[len(expected) + 1] == 0xA7
        for memory, before in zip(storage, data, strict=True):
            assert memory.raw[:-1] == b"\xcd" + before.tobytes() + b"\xcd"
        assert function(None, pointers, (ctypes.c_void_p * 1)(pointers[1])) == -1
    source = replace(
        request,
        nodes=(*tuple(replace(n, op="constant", effect="constant") for n in operands), node),
        input_storages=(),
        constants=tuple(
            ConstantBinding(n.id, dtype + "-le-raw", value.tobytes().hex())
            for n, value in zip(operands, data, strict=True)
        ),
    )
    record, files = emit_predicate_c(source, "folded")
    assert record["inputs"] == []
    folded = compile_c(tmp_path, files, "folded", "folded_" + dtype + "_" + predicate)
    output.raw = b"\xa7" * len(output)
    assert folded(None, None, outputs) == 0
    assert output.raw[1 : len(expected) + 1] == expected
    assert output.raw[0] == output.raw[len(expected) + 1] == 0xA7
    for attributes in [
        (("operation", "arith.cmpf"), ("predicate", "ult")),
        (("operation", "arith.cmpf"), ("predicate", "ueq")),
    ]:
        with pytest.raises(ValueError, match="no compiled implementation"):
            emit_predicate_c(replace(request, nodes=(*operands, replace(node, attrs=attributes))), "compare")
