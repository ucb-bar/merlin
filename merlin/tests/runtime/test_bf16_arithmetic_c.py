"""Check compiled BF16 source arithmetic against the rational IEEE reference."""

import ctypes
import itertools
import random
from dataclasses import replace

import pytest

from merlin.compile.c_float_arithmetic import emit_float_arithmetic_c
from merlin.semantic_compiler.model import ConstantBinding, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.tests.runtime.test_float_arithmetic_c import compile_c
from merlin.tests.runtime.test_float_arithmetic_c import request as f64_request


def request(shape, operation, shared=False):
    source = f64_request(shape, shared)
    return replace(source, nodes=tuple(
        replace(node, type=TensorType(shape, "bf16", "raw-bit-pattern"),
                **({"op": operation, "attrs": (("numeric_semantics", f"ieee754-bf16-{operation[5:]}-rne-v1"),)}
                   if node.effect == "pure" else {}))
        for node in source.nodes
    ))


def memory(codes):
    return ctypes.create_string_buffer(b"guard" + b"".join(x.to_bytes(2, "little") for x in codes) + b"end!")


def invoke(library, operands, output):
    inputs = (ctypes.c_void_p * len(operands))(*(ctypes.addressof(a) + 5 for a in operands))
    return library.multiply(None, inputs, (ctypes.c_void_p * 1)(ctypes.addressof(output) + 5))


@pytest.mark.parametrize("operation,shared", itertools.product(("bf16_add", "bf16_sub"), (False, True)))
def test_all_left_encodings_edges_seeded_pairs_and_rounding_modes(tmp_path, operation, shared):
    edges = [0, 1, 2, 127, 128, 129, 0x3B80, 0x3B81, 0x3F80, 0x3F81, 0x3F82, 0x7F7F, 0x7F80]
    edges += [code | 0x8000 for code in edges]
    rng = random.Random(20261020)
    pairs = list(itertools.product(edges, repeat=2))
    pairs += [(rng.randrange(65536), rng.randrange(65536)) for _ in range(4096)]
    # Every left code with a deterministic right value, not all 2^32 pairs.
    pairs += [(code, edges[code % len(edges)]) for code in range(65536)]
    pairs = [(a, a if shared else b) for a, b in pairs]
    pairs = [(a, b) for a, b in pairs if (a & 32767) <= 32640 and (b & 32767) <= 32640
             and not ((a & 32767) == (b & 32767) == 32640
                      and ((a ^ b ^ (32768 if operation == "bf16_sub" else 0)) & 32768))]
    source = request((len(pairs),), operation, shared)
    record, library = compile_c(tmp_path, source)
    arrays = [tuple(a for a, _ in pairs)] + ([] if shared else [tuple(b for _, b in pairs)])
    (expected,) = evaluate_graph(source, {
        key: TensorValue(source.node(key).type, codes) for key, codes in zip(record["inputs"], arrays, strict=True)
    })
    expected_bytes = b"".join(code.to_bytes(2, "little") for code in expected.elements)
    inputs = [memory(codes) for codes in arrays]
    saved = [a.raw for a in inputs]
    try:
        for mode in range(4):
            assert library.mode(mode) == 0
            output = memory([0xCDCD] * len(pairs))
            assert invoke(library, inputs, output) == 0
            assert output.raw[5:5 + len(expected_bytes)] == expected_bytes
            assert output.raw[:5] == b"guard" and output.raw[-5:] == b"end!\0"
            assert [a.raw for a in inputs] == saved
    finally:
        library.mode(0)


@pytest.mark.parametrize("operation", ["bf16_add", "bf16_sub"])
def test_invalid_late_lane_overlap_and_policy_fail_before_writes(tmp_path, operation):
    source = request((3,), operation)
    _, library = compile_c(tmp_path, source)
    bad_pairs = [(0x7FC1, 0x3F80), (0x3F80, 0xFFFF),
                 (0x7F80, 0xFF80 if operation == "bf16_add" else 0x7F80)]
    for a, b in bad_pairs:
        inputs = [memory([0x3F80, 0, a]), memory([0x3F80, 0, b])]
        output = memory([123, 456, 789])
        saved = output.raw
        assert invoke(library, inputs, output) == -2 and output.raw == saved
        saved = inputs[0].raw
        assert invoke(library, inputs, inputs[0]) == -1 and inputs[0].raw == saved
    assert library.multiply(None, None, None) == -1
    for wrong in (replace(source.nodes[-1], attrs=()), replace(source.nodes[-1], index_maps=())):
        with pytest.raises(ValueError):
            emit_float_arithmetic_c(replace(source, nodes=(*source.nodes[:-1], wrong)), "wrong")


@pytest.mark.parametrize("operation", ["bf16_add", "bf16_sub"])
@pytest.mark.parametrize("nan", [False, True])
def test_invalid_declared_constants_fail_during_compilation(operation, nan):
    source = request((1,), operation)
    a = 0x7FC1 if nan else 0x7F80
    b = 0xFF80 if operation == "bf16_add" else 0x7F80
    source = replace(source, input_storages=(), constants=(
        ConstantBinding("a", "bf16-le-raw", a.to_bytes(2, "little").hex()),
        ConstantBinding("b", "bf16-le-raw", b.to_bytes(2, "little").hex()),
    ), nodes=tuple(replace(node, op="constant", effect="constant") if node.effect == "input" else node
                   for node in source.nodes))
    with pytest.raises(ValueError, match="NaN|infinity cancellation"):
        emit_float_arithmetic_c(source, "invalid_constants")
