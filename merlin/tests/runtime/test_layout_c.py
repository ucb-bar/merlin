"""Compiled layouts preserve element bytes, policies, inputs and guard memory."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.layout_c import emit_layout_c
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


@pytest.fixture(
    params=[("f32", 4), ("bf16", 2), ("fp8_e4m3", 1), ("f64", 8), ("i64", 8), ("i32", 4), ("i8", 1), ("i1", 1)]
)
def element(request):
    dtype, width = request.param
    policy = "signed-i64" if dtype == "i64" else "boolean" if dtype == "i1" else "raw-bit-pattern"
    patterns = {
        1: [0, 0x80, 0x7F, 0xFF, 1, 0x38],
        2: [0, 0x8000, 0x7FC1, 0xFF81, 0x7F80, 0x3F80],
        4: [0, 0x80000000, 0x7FC12345, 0xFF812345, 0x7F800000, 0x3F800000],
        8: [0, 1 << 63, 0x7FF8000000001234, 0xFFF0000000001234, 0x7FF0000000000000, 0x3FF0000000000000],
    }
    bits = np.array([0, 1] if dtype == "i1" else patterns[width], dtype=f"<u{width}")
    return dtype, policy, bits


@pytest.mark.parametrize("kind", ["transpose", "reshape", "copy"])
def test_compiled_layout_preserves_bits_and_rejects_implicit_inplace_updates(tmp_path, kind, element):
    dtype, policy, patterns = element
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler required")
    source_type = TensorType((5, 7), dtype, policy)
    if kind == "transpose":
        shape = (7, 5)
        maps = (IndexMap(2, ((0, 1), (1, 0)), (0, 0)), IndexMap(2, ((1, 0), (0, 1)), (0, 0)))
    else:
        shape = (1, 35) if kind == "reshape" else (5, 7)
        maps = () if kind == "reshape" else pointwise_identity_maps(2, 1)
    result = SemanticNode("out", kind, ("in",), TensorType(shape, dtype, policy), index_maps=maps)
    request = KernelRequest(
        (SemanticNode("in", "input", (), source_type, effect="input"), result),
        ("out",),
        ("external",),
        (("in", "external"),),
        "synthetic",
    )
    _, files = emit_layout_c(request, "layout")
    for filename, text in files.items():
        (tmp_path / filename).write_text(text)
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "layout.c",
            "-o",
            f"layout_{dtype}_{kind}.so",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    # Passed-test directories can be reused by pytest's retention policy.
    # dlopen caches names even after deletion; keep each case's library distinct.
    lib = ctypes.CDLL(str(tmp_path / f"layout_{dtype}_{kind}.so"))
    lib.layout.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    bits = np.resize(patterns, 35).reshape(5, 7)
    source = ctypes.create_string_buffer(bits.tobytes())
    target = ctypes.create_string_buffer(bytes([0xA7]) * (bits.nbytes + 32))
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(source))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(target) + 16)
    assert lib.layout(None, inputs, outputs) == 0
    expected = bits.T.tobytes() if kind == "transpose" else bits.tobytes()
    assert target.raw[16 : 16 + bits.nbytes] == expected and source.raw[: bits.nbytes] == bits.tobytes()
    assert target.raw[:16] == target.raw[16 + bits.nbytes : 16 + bits.nbytes + 16] == bytes([0xA7]) * 16
    assert lib.layout(None, inputs, inputs) != 0
    with pytest.raises(ValueError):
        emit_layout_c(replace(request, nodes=(request.nodes[0], replace(result, attrs=(("unknown", True),)))), "layout")


def indexed_case(kind, dtype="f32", policy="raw-bit-pattern"):
    shape = (3, 5)
    if kind == "reverse":
        inputs = (shape,)
        op = "gather"
        maps = (IndexMap(2, ((1, 0), (0, -1)), (0, 4)), pointwise_identity_maps(2, 1)[0])

        def expected(values):
            return values[0][:, ::-1]
    elif kind == "broadcast":
        inputs = ((5,),)
        op = "gather"
        maps = (IndexMap(2, ((0, 1),), (0,)), pointwise_identity_maps(2, 1)[0])

        def expected(values):
            return np.broadcast_to(values[0], shape)
    elif kind == "im2col":
        inputs, shape, op = ((1, 2, 4, 5),), (2, 2, 2, 1, 3, 4), "gather"
        maps = (
            IndexMap(6, ((0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 1, 0), (0, 0, 1, 0, 0, 1)), (0, 0, 0, 0)),
            pointwise_identity_maps(6, 1)[0],
        )

        def expected(values):
            return np.lib.stride_tricks.sliding_window_view(values[0], (2, 2), axis=(2, 3)).transpose(1, 4, 5, 0, 2, 3)
    elif kind == "transpose3d":
        inputs, shape, op = ((2, 3, 5),), (5, 2, 3), "transpose"
        maps = (IndexMap(3, ((0, 1, 0), (0, 0, 1), (1, 0, 0)), (0, 0, 0)), pointwise_identity_maps(3, 1)[0])

        def expected(values):
            return values[0].transpose(2, 0, 1)
    elif kind in {"insert", "padding"}:
        inputs, op = (
            (((2, 2), shape), "insert_slice") if kind == "insert" else (((1, 2, 2, 3), (1, 2, 4, 5)), "insert_slice")
        )
        shape = inputs[1]
        identity = pointwise_identity_maps(len(shape), 1)[0]
        offsets = (1, 0) if kind == "insert" else (0, 0, 1, 1)
        strides = (1, 2) if kind == "insert" else (1, 1, 1, 1)
        placement = IndexMap(
            len(shape),
            tuple(tuple(strides[i] if d == i else 0 for d in range(len(shape))) for i in range(len(shape))),
            offsets,
        )
        maps = (identity, placement, placement)

        def expected(values):
            out = values[1].copy()
            if kind == "insert":
                out[1:3, 0:4:2] = values[0]
            else:
                out[:, :, 1:3, 1:4] = values[0]
            return out
    else:
        inputs, shape, op, maps = ((2, 1, 3), (2, 2, 3)), (2, 4, 3), "concat", ()

        def expected(values):
            return np.concatenate((values[0], values[1], values[0]), axis=1)

    nodes = tuple(
        SemanticNode(f"x{i}", "input", (), TensorType(s, dtype, policy), effect="input") for i, s in enumerate(inputs)
    )
    operands = tuple(node.id for node in nodes) + ((nodes[0].id,) if op == "concat" else ())
    result = SemanticNode(
        "out",
        op,
        operands,
        TensorType(shape, dtype, policy),
        (("axis", 1),) if op == "concat" else (),
        index_maps=maps,
    )
    return KernelRequest(
        (*nodes, result), (result.id,), ("external",), tuple((node.id, "external") for node in nodes), "synthetic"
    ), expected


@pytest.mark.parametrize("kind", ["reverse", "broadcast", "im2col", "transpose3d", "insert", "padding", "concat"])
def test_indexed_layout_preserves_bits_maps_guards_and_source_storage(tmp_path, kind, element):
    dtype, policy, codebook = element
    request, reference = indexed_case(kind, dtype, policy)
    record, files = emit_layout_c(request, "layout")
    for filename, text in files.items():
        (tmp_path / filename).write_text(text)
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
            "layout.c",
            "-o",
            f"indexed_{dtype}_{kind}.so",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(tmp_path / f"indexed_{dtype}_{kind}.so"))
    lib.layout.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    leaves = request.nodes[:-1]
    for panel in range(3):
        values = [
            np.resize(np.roll(codebook, panel + i), np.prod(node.type.shape)).reshape(node.type.shape)
            for i, node in enumerate(leaves)
        ]
        storage = [ctypes.create_string_buffer(value.tobytes()) for value in values]
        inputs = (ctypes.c_void_p * len(record["inputs"]))(
            *(ctypes.addressof(storage[int(key[1:])]) for key in record["inputs"])
        )
        expected = reference(values).tobytes()
        output = ctypes.create_string_buffer(bytes([0xA7]) * (len(expected) + 32))
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 16)
        assert lib.layout(None, inputs, outputs) == 0
        assert output.raw[16 : 16 + len(expected)] == expected
        assert output.raw[:16] == output.raw[16 + len(expected) : -1] == bytes([0xA7]) * 16
        for value, original in zip(storage, values, strict=True):
            assert value.raw[:-1] == original.tobytes()
        for original in storage:
            assert lib.layout(None, inputs, (ctypes.c_void_p * 1)(ctypes.addressof(original))) != 0
        assert lib.layout(None, inputs, (ctypes.c_void_p * 1)()) != 0


def test_invalid_indexed_layouts_fail_before_source_generation():
    for kind in ("reverse", "insert", "concat"):
        source, _ = indexed_case(kind)
        node = source.nodes[-1]
        bad = replace(node, attrs=(("unknown", True),))
        with pytest.raises(ValueError):
            emit_layout_c(replace(source, nodes=(*source.nodes[:-1], bad)), "layout")
        if kind == "concat":
            bad = replace(node, attrs=(("axis", True),))
        else:
            mapping = node.index_maps[0 if kind == "reverse" else 1]
            invalid = replace(mapping, offsets=tuple(n + 1 for n in node.type.shape))
            maps = (invalid, node.index_maps[-1]) if kind == "reverse" else (node.index_maps[0], invalid, invalid)
            bad = replace(node, index_maps=maps)
        with pytest.raises(ValueError):
            emit_layout_c(replace(source, nodes=(*source.nodes[:-1], bad)), "layout")
    source, _ = indexed_case("broadcast")
    with pytest.raises(ValueError):
        emit_layout_c(replace(source, physical_extents=((source.nodes[0].id, 1),)), "layout")
    mapping = IndexMap(2, ((2**63, 0),), (0,))
    bad = replace(source.nodes[-1], index_maps=(mapping, source.nodes[-1].index_maps[-1]))
    with pytest.raises(ValueError):
        emit_layout_c(replace(source, nodes=(*source.nodes[:-1], bad)), "layout")


@pytest.mark.parametrize("kind", ["reverse", "concat"])
def test_layout_c_rejects_changed_dtype_or_numerical_policy(kind):
    request, _ = indexed_case(kind)
    result = request.nodes[-1]
    for type_ in (replace(result.type, dtype="i32"), replace(result.type, numerical_policy="different")):
        with pytest.raises(ValueError, match="unchanged element types"):
            emit_layout_c(replace(request, nodes=(*request.nodes[:-1], replace(result, type=type_))), "layout")
    with pytest.raises((KeyError, ValueError)):
        request = replace(
            request, nodes=tuple(replace(n, type=replace(n.type, dtype="unknown")) for n in request.nodes)
        )
        emit_layout_c(request, "layout")


def test_compiled_embedding_preserves_table_encoding_and_checks_all_indices(tmp_path, element):
    dtype, policy, patterns = element
    indices_type = TensorType((2,), "i64", "signed-i64")
    table_type = TensorType((5, 3), dtype, policy)
    result = SemanticNode(
        "out",
        "embedding_lookup",
        ("indices", "table"),
        TensorType((2, 3), dtype, policy),
        (("index_domain", "nonnegative-below-table-rows"), ("minimum_index_bits", 4)),
    )
    request = KernelRequest(
        (
            SemanticNode("indices", "input", (), indices_type, effect="input"),
            SemanticNode("table", "input", (), table_type, effect="input"),
            result,
        ),
        ("out",),
        ("external",),
        (("indices", "external"), ("table", "external")),
        "synthetic",
    )
    _, files = emit_layout_c(request, "lookup")
    for name, text in files.items():
        (tmp_path / name).write_text(text)
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
            "lookup.c",
            "-o",
            f"lookup_{dtype}.so",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    entry = ctypes.CDLL(str(tmp_path / f"lookup_{dtype}.so")).lookup
    entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    table = np.resize(patterns, 15).reshape(5, 3)
    memory = ctypes.create_string_buffer(b"\xcd" + table.tobytes() + b"\xcd")
    for last in (0, 4, -1, 5, -(1 << 63), (1 << 63) - 1):
        positions = ctypes.create_string_buffer(b"\xcd" + np.array([4, last], dtype="<i8").tobytes())
        size = 6 * patterns.itemsize
        output = ctypes.create_string_buffer(b"\xa7" * (size + 2))
        inputs = (ctypes.c_void_p * 2)(ctypes.addressof(positions) + 1, ctypes.addressof(memory) + 1)
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
        status = entry(None, inputs, outputs)
        if last < 0 or last >= 5:
            assert status == -2 and output.raw[:-1] == b"\xa7" * (size + 2)
        else:
            assert status == 0 and output.raw[1 : size + 1] == table[[4, last]].tobytes()
            assert output.raw[0] == output.raw[size + 1] == 0xA7
        assert memory.raw[:-1] == b"\xcd" + table.tobytes() + b"\xcd"
