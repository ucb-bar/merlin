"""Compiled boundary counts preserve source ties and IEEE edge cases."""

import ctypes
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.c_bucketize import emit_bucketize_c
from merlin.compile.c_predicates import emit_predicate_c
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.tests.runtime.test_predicates_c import compile_c


def request_for(right, count=31, shape=(17, 31), *, limits=None):
    rank = len(shape)
    value = SemanticNode("values", "input", (), TensorType(shape, "f32", "raw-bit-pattern"), effect="input")
    boundary = SemanticNode(
        "limits",
        "input" if limits is None else "constant",
        (),
        TensorType((count,), "f32", "raw-bit-pattern"),
        effect="input" if limits is None else "constant",
    )
    projection = IndexMap(
        rank + 1, tuple(tuple(int(i == j) for j in range(rank + 1)) for i in range(rank)), (0,) * rank
    )
    node = SemanticNode(
        "out",
        "bucketize",
        (value.id, boundary.id),
        TensorType(shape, "i64", "signed-i64"),
        (("numeric_semantics", "f32-unordered-boundary-count-i64-v1"), ("right", right), ("boundary_count", count)),
        index_maps=(projection, IndexMap(rank + 1, ((0,) * rank + (1,),), (0,)), projection),
    )
    return KernelRequest(
        (value, boundary, node),
        (node.id,),
        ("external",),
        tuple((n.id, "external") for n in (value, boundary) if n.effect == "input"),
        "synthetic",
        constants=() if limits is None else (ConstantBinding(boundary.id, "f32-le-raw", limits.tobytes().hex()),),
    )


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("count", [1, 31, 257])
def test_compiled_bucketize_matches_numpy_for_random_bits_and_edge_cases(tmp_path, right, constant, count):
    rng = np.random.default_rng(1991 + count)
    boundaries = np.sort(rng.integers(-9, 10, count).astype("<f4"))
    if count > 1:
        boundaries[0], boundaries[-1] = -np.inf, np.inf
    if count == 31:
        boundaries = np.sort(np.array([-np.inf, -0.0, 0.0, np.inf] + [i / 4 for i in range(27)], dtype="<f4"))
    request = request_for(right, count, limits=boundaries if constant else None)
    record, files = emit_predicate_c(request, "buckets")
    assert record["algorithm"] == "checked-binary-search"
    function = compile_c(tmp_path, files, "buckets", f"buckets_{count}_{int(right)}_{int(constant)}")
    edges = np.array([0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC12345, 0xFF812345], dtype="<u4")
    for panel in range(3):
        bits = np.concatenate((edges, rng.integers(0, 1 << 32, 519, dtype="<u4")))
        values = bits.view("<f4")
        if panel == 1:
            values[:count] = boundaries
        with np.errstate(invalid="ignore"):
            expected = np.searchsorted(boundaries, values, side="right" if right else "left").astype("<i8").tobytes()
        storage = [ctypes.create_string_buffer(b"\xcd" + data.tobytes() + b"\xcd") for data in (values, boundaries)]
        inputs = (ctypes.c_void_p * 2)(*(ctypes.addressof(data) + 1 for data in storage))
        output = ctypes.create_string_buffer(b"\xa7" * (len(expected) + 2))
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
        assert function(None, inputs, outputs) == 0
        assert output.raw[1 : len(expected) + 1] == expected
        assert output.raw[0] == output.raw[len(expected) + 1] == 0xA7
        for data, original in zip(storage, (values, boundaries), strict=True):
            assert data.raw[:-1] == b"\xcd" + original.tobytes() + b"\xcd"


def test_invalid_boundary_data_buffers_and_overlap_fail_before_output_writes(tmp_path):
    request = request_for(True, 4, (8,))
    _, files = emit_bucketize_c(request, "buckets")
    function = compile_c(tmp_path, files, "buckets", "failures")
    values = ctypes.create_string_buffer(np.zeros(8, dtype="<f4").tobytes())
    output = ctypes.create_string_buffer(b"\xa7" * 64)
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output))
    for boundaries in ([0, 2, 1, 3], [0, 1, 2, np.nan], [-np.inf, np.nan, 1, np.inf]):
        limits = ctypes.create_string_buffer(np.array(boundaries, dtype="<f4").tobytes())
        inputs = (ctypes.c_void_p * 2)(ctypes.addressof(values), ctypes.addressof(limits))
        assert function(None, inputs, outputs) == -2
        assert output.raw[:-1] == b"\xa7" * 64
    for inputs in (
        None,
        (ctypes.c_void_p * 2)(None, ctypes.addressof(values)),
        (ctypes.c_void_p * 2)((1 << 64) - 1, ctypes.addressof(values)),
        (ctypes.c_void_p * 2)(ctypes.addressof(output), ctypes.addressof(values)),
        (ctypes.c_void_p * 2)(ctypes.addressof(values), ctypes.addressof(output)),
    ):
        assert function(None, inputs, outputs) == -1
        assert output.raw[:-1] == b"\xa7" * 64


def test_bucketize_rejects_changed_policies_maps_counts_and_invalid_literals():
    request = request_for(True)
    node = request.nodes[-1]
    for key, value in (
        ("numeric_semantics", "ordered-boundary-count"),
        ("right", 1),
        ("boundary_count", 32),
        ("unknown", True),
    ):
        attrs = dict(node.attrs) | {key: value}
        with pytest.raises(ValueError):
            emit_bucketize_c(
                replace(request, nodes=(*request.nodes[:-1], replace(node, attrs=tuple(attrs.items())))), "b"
            )
    with pytest.raises(ValueError):
        changed = replace(node, index_maps=(replace(node.index_maps[0], offsets=(1, 0)), *node.index_maps[1:]))
        emit_bucketize_c(replace(request, nodes=(*request.nodes[:-1], changed)), "b")
    with pytest.raises(ValueError):
        changed = replace(request.nodes[0], type=replace(request.nodes[0].type, numerical_policy="target-rounded"))
        emit_bucketize_c(replace(request, nodes=(changed, *request.nodes[1:])), "b")
    with pytest.raises(ValueError):
        emit_bucketize_c(replace(request, physical_extents=(("values", 1),)), "b")
    with pytest.raises(ValueError):
        emit_bucketize_c(request_for(True, 3, limits=np.array([0, 2, 1], dtype="<f4")), "b")
    with pytest.raises(ValueError):
        emit_bucketize_c(request_for(True, 3, limits=np.array([0, 1, np.nan], dtype="<f4")), "b")
