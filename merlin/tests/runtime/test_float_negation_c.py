"""Source sign flips preserve signed zero, nonfinite encodings, and guards."""

import ctypes
import random
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.c_float_arithmetic import FLOAT_ARITHMETIC_OPERATIONS, emit_float_arithmetic_c
from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def request(dtype, count):
    type_ = TensorType((count,), dtype, "raw-bit-pattern")
    x = SemanticNode("x", "input", (), type_, effect="input")
    y = SemanticNode("y", dtype + "_neg", ("x",), type_,
                     (("numeric_semantics", dtype + "-sign-bit-flip-v1"),),
                     index_maps=pointwise_identity_maps(1, 1))
    return KernelRequest((x, y), ("y",), ("external",), (("x", "external"),), "synthetic")


@pytest.mark.parametrize("dtype,width", [("f32", 4), ("f64", 8)])
def test_all_encoding_classes_and_random_payloads(tmp_path, dtype, width):
    rng = random.Random(47)
    sign = 1 << (8 * width - 1)
    # Include both zero signs, subnormals, infinity, and signaling/quiet NaNs.
    edges = [0, 1, sign - 1, sign, (0x7f800000 if width == 4 else 0x7ff0000000000000)]
    words = [x | s for x in edges for s in (0, sign)] + [rng.getrandbits(8 * width) for _ in range(2048)]
    graph = request(dtype, len(words))
    record, files = emit_float_arithmetic_c(graph, "negate")
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    library_path = tmp_path / ("negate_" + dtype + ".so")
    subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(tmp_path / "negate.c"), "-o", str(library_path)], check=True, capture_output=True)
    function = ctypes.CDLL(str(library_path)).negate
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    raw = b"".join(word.to_bytes(width, "little") for word in words)
    a = ctypes.create_string_buffer(b"guard" + raw + b"tail")
    b = ctypes.create_string_buffer(b"guard" + bytes(len(raw)) + b"tail")
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(a) + 5)
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(b) + 5)
    assert function(None, inputs, outputs) == 0
    expected = b"".join((word ^ sign).to_bytes(width, "little") for word in words)
    assert b.raw[5:5 + len(raw)] == expected
    assert b.raw[:5] == b"guard" and b.raw[5 + len(raw):] == b"tail\0"
    assert a.raw == b"guard" + raw + b"tail\0"
    assert record["numeric_semantics"] == dtype + "-sign-bit-flip-v1"
    assert evaluate_graph(graph, {"x": TensorValue(graph.nodes[0].type, tuple(words))})[0].elements == tuple(w ^ sign for w in words)
    before = a.raw
    assert function(None, inputs, inputs) == -1 and a.raw == before
    assert function(None, None, outputs) == -1


def test_wrong_sign_flip_policy_is_rejected():
    graph = request("f32", 2)
    with pytest.raises(ValueError, match="policy differ"):
        emit_float_arithmetic_c(replace(graph, nodes=(graph.nodes[0], replace(graph.nodes[1], attrs=()))), "wrong")


@pytest.mark.parametrize("constant", [False, True])
def test_bf16_negation_preserves_every_bit_pattern_and_constant_byte(tmp_path, constant):
    words = tuple(range(65536))
    raw = b"".join(word.to_bytes(2, "little") for word in words)
    graph = request("bf16", len(words))
    if constant:
        graph = replace(graph, nodes=(replace(graph.nodes[0], op="constant", effect="constant"), graph.nodes[1]),
                        input_storages=(), constants=(ConstantBinding("x", "bf16-le-raw", raw.hex()),))
    original = graph.record()
    record, files = emit_float_arithmetic_c(graph, "negate_bf16")
    assert "bf16_neg" in FLOAT_ARITHMETIC_OPERATIONS
    assert record["numeric_semantics"] == "bf16-sign-bit-flip-v1"
    assert record["inputs"] == ["x"]
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    library_path = tmp_path / "negate_bf16.so"
    subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(tmp_path / "negate_bf16.c"),
                    "-o", str(library_path)], check=True, capture_output=True)
    function = ctypes.CDLL(str(library_path)).negate_bf16
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    source = ctypes.create_string_buffer(b"guard" + raw + b"tail")
    destination = ctypes.create_string_buffer(b"guard" + bytes(len(raw)) + b"tail")
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(source) + 5)
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(destination) + 5)
    assert function(None, inputs, outputs) == 0
    expected = b"".join((word ^ 0x8000).to_bytes(2, "little") for word in words)
    assert destination.raw == b"guard" + expected + b"tail\0"
    assert source.raw == b"guard" + raw + b"tail\0"
    assert graph.record() == original
    if constant:
        assert bytes.fromhex(graph.constants[0].data_hex) == raw
    for offset in (0, 4, 5, 6):
        overlapping = (ctypes.c_void_p * 1)(ctypes.addressof(source) + offset)
        before = source.raw
        assert function(None, inputs, overlapping) == -1
        assert source.raw == before
    before = destination.raw
    assert function(None, None, outputs) == -1
    assert function(None, inputs, None) == -1
    assert destination.raw == before


@pytest.mark.parametrize("fault", ["policy", "dtype", "maps"])
def test_bf16_negation_refuses_changed_source_contract(fault):
    graph = request("bf16", 2)
    node = graph.nodes[1]
    if fault == "policy":
        node = replace(node, attrs=(("numeric_semantics", "ieee754-bf16-neg-rne-v1"),))
    elif fault == "dtype":
        node = replace(node, type=replace(node.type, dtype="f32"))
    else:
        node = replace(node, index_maps=())
    with pytest.raises(ValueError, match="policy differ"):
        emit_float_arithmetic_c(replace(graph, nodes=(graph.nodes[0], node)), "wrong")
