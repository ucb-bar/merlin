"""Check selected host BF16 dot scheduling with directed raw-bit fixtures."""
import ctypes
from dataclasses import replace
import shutil
import subprocess
import struct

import pytest

from merlin.compile.c_bf16_linear import SOURCE_POLICY, emit_c
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import (
    ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps)


def request(k):
    type_ = lambda shape: TensorType(shape, "bf16", "raw-bit-pattern")
    maps = (IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
            IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
            IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)))
    nodes = (SemanticNode("a", "input", (), type_((1, k)), effect="input"),
             SemanticNode("b", "input", (), type_((k, 1)), effect="input"),
             SemanticNode("seed", "input", (), type_((1, 1)), effect="input"),
             SemanticNode("out", "source_bf16_matmul_unqualified", ("a", "b", "seed"), type_((1, 1)),
                          (("numeric_semantics", "source-bf16-mul-add-unqualified-v1"), ("reduction_order", "unresolved")),
                          index_maps=(*maps, maps[-1])))
    return KernelRequest(nodes, ("out",), ("external",),
                         (("a", "external"), ("b", "external"), ("seed", "external")), "synthetic")


def compile_c(directory, k, *, policy=SOURCE_POLICY, source=None):
    source = source or request(k)
    record, files = emit_c(source, "dot", policy=policy)
    for name, text in files.items():
        (directory / name).write_text(text)
    (directory / "mode.c").write_text("""#include <fenv.h>
int mode(int downward) { return fesetround(downward?FE_DOWNWARD:FE_TONEAREST); }
""")
    # Passed tmp_path trees can be removed and their directory names reused.
    # The dynamic loader retains a path's first library, so bind it to shape.
    library_path = directory / f"dot-{k}-{policy}-{source.digest()[:12]}.so"
    subprocess.run(["nice", "-n", "10", shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-frounding-math", "-ffp-contract=off", "-shared", "-fPIC",
                    str(directory / "dot.c"), str(directory / "mode.c"), "-lm", "-o", str(library_path)],
                   check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    library.dot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    library.dot.restype = ctypes.c_int
    return record, library


def memory(words):
    return ctypes.create_string_buffer(b"guard" + b"".join(code.to_bytes(2, "little") for code in words) + b"end")


def call(library, arrays, output):
    inputs = (ctypes.c_void_p * 3)(*(ctypes.addressof(value) + 5 for value in arrays))
    return library.dot(None, inputs, (ctypes.c_void_p * 1)(ctypes.addressof(output) + 5))


def test_runtime_refusals_are_atomic(tmp_path):
    _, library = compile_c(tmp_path, 1)
    for a, b, seed in ((0x7fc0, 0x3f80, 0), (0x3f80, 0x7f80, 0), (0x7f7f, 0x7f7f, 0),
                       (0x3f80, 0x3f80, 0x8000)):
        output = memory([0xdead])
        assert call(library, [memory([a]), memory([b]), memory([seed])], output) == -2
        assert output.raw == memory([0xdead]).raw
    arrays, output = [memory([0x3f80]), memory([0x3f80]), memory([0])], memory([0xdead])
    assert library.mode(1) == 0
    try:
        assert call(library, arrays, output) == -1
    finally:
        assert library.mode(0) == 0
    assert output.raw == memory([0xdead]).raw
    assert call(library, arrays, arrays[0]) == -1


def test_source_contract_rejects_reinterpretation():
    source = request(1)
    changed = replace(source.nodes[-1], attrs=(("numeric_semantics", "ieee754-bf16-matmul-rne-v1"),))
    with pytest.raises(ValueError, match="source arithmetic"):
        emit_c(replace(source, nodes=(*source.nodes[:-1], changed)), "dot")
    operand = replace(source.nodes[0], attrs=(("hidden_scale", 2),))
    with pytest.raises(ValueError, match="source types or effects"):
        emit_c(replace(source, nodes=(operand, *source.nodes[1:])), "dot")


def widened_request(k):
    source = request(k)
    lhs, rhs, seed, out = source.nodes
    raw_f32 = lambda shape: TensorType(shape, "f32", "raw-bit-pattern")
    left = SemanticNode("left_f32", "bf16_to_f32", (lhs.id,), raw_f32(lhs.type.shape),
                        (("numeric_semantics", "bf16-to-f32-exact-v1"),), index_maps=pointwise_identity_maps(2, 1))
    right = replace(left, id="right_f32", inputs=(rhs.id,), type=raw_f32(rhs.type.shape))
    seed = replace(seed, type=raw_f32(out.type.shape))
    dot = replace(out, id="dot_f32", op="source_f32_matmul_unqualified", inputs=(left.id, right.id, seed.id),
                  type=seed.type, attrs=(("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                                         ("reduction_order", "unresolved")))
    result = SemanticNode("out", "f32_to_bf16", (dot.id,), out.type,
                          (("numeric_semantics", "f32-to-bf16-rne-v1"),), index_maps=pointwise_identity_maps(2, 1))
    return replace(source, nodes=(lhs, rhs, seed, left, right, dot, result))


@pytest.mark.parametrize("widened", [False, True])
@pytest.mark.parametrize("a,b,expected", [
    ([0x3f81], [0x3fc0], 0x3fc2),  # Exact BF16 product halfway rounds to even at the result.
    ([0x3f81, 0x3f80], [0x3fc0, 0xbfc0], 0x3c40),  # BF16-rounded products would give 0x3c80.
    ([0x3f80] * 4, [0x4b80, 0x3f80, 0xcb80, 0xbf80], 0xbf80),  # Ascending K, not a tree.
    ([0x3f80] * 2, [0x4380, 0x4040], 0x4382),  # Final BF16 RNE after binary32 accumulation.
    ([0x3f80] * 33, [0x3f80] + [0] * 31 + [0x3c40], 0x3f82),  # Single final RNE, no tile chop.
    ([1], [0x3f80], 1),  # Preserve a BF16 subnormal through the declared operations.
    ([1], [0x3fc0], 2),  # Subnormal final BF16 RNE, including tie-to-even.
    ([1], [1], 0),  # Binary32 product underflow follows source arithmetic.
    ([0x8000], [0x3f80], 0),
])
def test_source_f32_accumulation_preserves_products_and_order(tmp_path, widened, a, b, expected):
    source = widened_request(len(a)) if widened else request(len(a))
    record, library = compile_c(tmp_path, len(a), policy=SOURCE_POLICY, source=source)
    arrays = [memory(a), memory(b), memory([0, 0] if widened else [0])]
    output = memory([0xdead])
    before = [value.raw for value in arrays]
    assert call(library, arrays, output) == 0
    assert output.raw == memory([expected]).raw
    assert [value.raw for value in arrays] == before
    assert record["numeric_semantics"] == SOURCE_POLICY
    assert record["placement"] == "host"
    assert record["source_request_digest"] == source.digest()
    assert record["inputs"] == ["a", "b", "seed"]
    assert record["outputs"] == ["out"]


@pytest.mark.parametrize("k", [1, 31, 32, 33, 65])
def test_source_f32_accumulation_ordinary_matrix_and_tails(tmp_path, k):
    source = request(k)
    a, b, seed, out = source.nodes
    source = replace(source, nodes=(replace(a, type=replace(a.type, shape=(2, k))),
                                    replace(b, type=replace(b.type, shape=(k, 3))),
                                    replace(seed, type=replace(seed.type, shape=(2, 3))),
                                    replace(out, type=replace(out.type, shape=(2, 3)))))
    _, library = compile_c(tmp_path, k, policy=SOURCE_POLICY, source=source)
    arrays = [memory([0x3f80] * k + [0x4000] * k), memory([0x3f80, 0x4000, 0xbf80] * k), memory([0] * 6)]
    output = memory([0xdead] * 6)
    def bf16(value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    expected = [bf16(value) for value in [k, 2*k, -k, 2*k, 4*k, -2*k]]
    assert call(library, arrays, output) == 0
    assert output.raw == memory(expected).raw


def test_widened_source_seed_and_policy_refusals(tmp_path):
    source = widened_request(1)
    _, library = compile_c(tmp_path, 1, policy=SOURCE_POLICY, source=source)
    for words in ([0, 0x8000], [1, 0], [0, 1]):
        output = memory([0xdead])
        assert call(library, [memory([0x3f80]), memory([0x3f80]), memory(words)], output) == -2
        assert output.raw == memory([0xdead]).raw
    assert emit_c(source, "dot") == emit_c(source, "dot", policy=SOURCE_POLICY)
    for policy in ("unspecified", "bf16-product-rne-tree32-f32-rne-bf16-rne-tiles-ltr-chop-v1"):
        with pytest.raises(ValueError, match="unknown.*policy"):
            emit_c(request(1), "dot", policy=policy)
    changed_cast = replace(source.nodes[3], attrs=(("numeric_semantics", "other"),))
    with pytest.raises(ValueError, match="conversion"):
        emit_c(replace(source, nodes=(*source.nodes[:3], changed_cast, *source.nodes[4:])), "dot", policy=SOURCE_POLICY)


def permuted_head_source(m, k, n, *, extra_reader=False, published=False, broadcast=False,
                         lhs_view=False, rhs_view=True, widened=True):
    """Linear whose [K,M] activation and/or [N,K] weight is permuted before the contraction.

    ``widened`` selects the bf16_to_f32 source chain; otherwise the permuted
    operands feed the plain BF16 contraction directly.
    """
    bf16, f32 = (lambda shape: TensorType(shape, "bf16", "raw-bit-pattern"),
                 lambda shape: TensorType(shape, "f32", "raw-bit-pattern"))
    widen = (("numeric_semantics", "bf16-to-f32-exact-v1"),)
    identity = pointwise_identity_maps(2, 1)
    swap = IndexMap(2, ((0, 1), (1, 0)), (0, 0))
    weight_shape, read = ((k,), IndexMap(2, ((1, 0),), (0,))) if broadcast else ((n, k), swap)
    left, right = ("xt" if lhs_view else "x"), ("wt" if rhs_view else "w")
    nodes = [SemanticNode("x", "input", (), bf16((k, m) if lhs_view else (m, k)), effect="input"),
             SemanticNode("w", "input", (), bf16(weight_shape if rhs_view else (k, n)), effect="input")]
    if lhs_view:
        nodes.append(SemanticNode("xt", "transpose", ("x",), bf16((m, k)), index_maps=(swap, identity[-1])))
    if rhs_view:
        nodes.append(SemanticNode("wt", "gather" if broadcast else "transpose", ("w",), bf16((k, n)),
                                  index_maps=(read, identity[-1])))
    contraction_maps = (IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
                        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
                        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
                        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)))
    if widened:
        nodes += [
            SemanticNode("xf", "bf16_to_f32", (left,), f32((m, k)), widen, index_maps=identity),
            SemanticNode("wf", "bf16_to_f32", (right,), f32((k, n)), widen, index_maps=identity),
            SemanticNode("seed", "constant", (), f32((m, n)), effect="constant"),
            SemanticNode("dot", "source_f32_matmul_unqualified", ("xf", "wf", "seed"), f32((m, n)),
                         (("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                          ("reduction_order", "unresolved")), index_maps=contraction_maps),
            SemanticNode("out", "f32_to_bf16", ("dot",), bf16((m, n)), (("numeric_semantics", "f32-to-bf16-rne-v1"),),
                         index_maps=identity),
        ]
    else:
        nodes += [
            SemanticNode("seed", "constant", (), bf16((m, n)), effect="constant"),
            SemanticNode("out", "source_bf16_matmul_unqualified", (left, right, "seed"), bf16((m, n)),
                         (("numeric_semantics", "source-bf16-mul-add-unqualified-v1"),
                          ("reduction_order", "unresolved")), index_maps=contraction_maps),
        ]
    outputs = ["out"]
    if extra_reader:
        nodes.append(SemanticNode("other", "bf16_to_f32", ("wt",), f32((k, n)), widen, index_maps=identity))
        outputs.append("other")
    if published:
        outputs.append("wt")
    seed = (ConstantBinding("seed", "f32-le-raw", "00" * (4 * m * n)) if widened
            else ConstantBinding("seed", "bf16-le-raw", "00" * (2 * m * n)))
    return KernelRequest(tuple(nodes), tuple(outputs), ("external",) * len(outputs),
                         (("x", "external"), ("w", "external")), "synthetic", constants=(seed,))


VIEWS = {"xt", "wt"}


def contraction_region(source, *, with_permutation):
    keys = tuple(node.id for node in source.nodes if node.effect == "pure" and node.id != "other"
                 and (with_permutation or node.id not in VIEWS))
    return extract_semantic_region(source, keys)


def random_bf16(generator, count):
    # Finite codes only, including signed zeros and subnormals.
    return [(generator.getrandbits(1) << 15) | (generator.randint(0, 0x8e) << 7) | generator.getrandbits(7)
            for _ in range(count)]


@pytest.mark.parametrize("widened", [True, False])
@pytest.mark.parametrize("lhs_view,rhs_view", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (3, 37, 5), (2, 64, 9)])
def test_folded_weight_permutation_matches_materialized_transpose(tmp_path, m, k, n, lhs_view, rhs_view, widened):
    import random

    source = permuted_head_source(m, k, n, lhs_view=lhs_view, rhs_view=rhs_view, widened=widened)
    unfolded, folded = (contraction_region(source, with_permutation=flag) for flag in (False, True))
    folded_record, folded_library = compile_c(tmp_path, k, source=folded)
    unfolded_record, unfolded_library = compile_c(tmp_path, k, source=unfolded)
    assert unfolded_record["inputs"] == ["xt" if lhs_view else "x", "wt" if rhs_view else "w", "seed"]
    assert "folded_operand_permutations" not in unfolded_record
    assert folded_record["inputs"] == ["x", "w", "seed"]
    assert folded_record["folded_operand_permutations"] == [
        {"operand": side, "permutation": view, "source": base, "source_axis_reads": [1, 0]}
        for side, view, base, present in (("lhs", "xt", "x", lhs_view), ("rhs", "wt", "w", rhs_view)) if present]
    generator = random.Random(1000 * m + 10 * k + n)
    statuses = set()
    for _ in range(16):
        x, w = random_bf16(generator, m * k), random_bf16(generator, n * k)
        # Stored [K,M] / [N,K] operands and their materialized [M,K] / [K,N] views.
        x_view = [x[reduction * m + row] for row in range(m) for reduction in range(k)] if lhs_view else x
        w_view = [w[column * k + reduction] for reduction in range(k) for column in range(n)] if rhs_view else w
        seed = memory([0] * ((2 if widened else 1) * m * n))
        expected, actual = memory([0xdead] * (m * n)), memory([0xbeef] * (m * n))
        status = call(unfolded_library, [memory(x_view), memory(w_view), seed], expected)
        assert call(folded_library, [memory(x), memory(w), seed], actual) == status
        statuses.add(status)
        if status == 0:
            assert actual.raw == expected.raw
        else:
            assert actual.raw == memory([0xbeef] * (m * n)).raw
    assert 0 in statuses


def test_unfolded_emission_is_unchanged_and_permutation_refusals():
    source = permuted_head_source(2, 3, 4)
    _, files = emit_c(contraction_region(source, with_permutation=False), "dot")
    assert "a+2u*(row*3u+reduction)" in files["dot.c"] and "b+2u*(reduction*4u+column)" in files["dot.c"]
    _, files = emit_c(contraction_region(source, with_permutation=True), "dot")
    assert "b+2u*(reduction*1u+column*3u)" in files["dot.c"]
    for widened in (True, False):
        both = permuted_head_source(2, 3, 4, lhs_view=True, widened=widened)
        _, files = emit_c(contraction_region(both, with_permutation=True), "dot")
        assert "a+2u*(row*1u+reduction*2u)" in files["dot.c"] and "b+2u*(reduction*1u+column*3u)" in files["dot.c"]
    for changed in (permuted_head_source(2, 3, 4, extra_reader=True), permuted_head_source(2, 3, 4, published=True)):
        # The escaping permuted tensor is a second region result.
        with pytest.raises(ValueError, match="ordered inputs differ"):
            emit_c(contraction_region(changed, with_permutation=True), "dot")
    region = contraction_region(source, with_permutation=True)
    shared = replace(region, nodes=(*region.nodes, SemanticNode(
        "again", "bf16_to_f32", ("wt",), TensorType((3, 4), "f32", "raw-bit-pattern"),
        (("numeric_semantics", "bf16-to-f32-exact-v1"),), index_maps=pointwise_identity_maps(2, 1))))
    with pytest.raises(ValueError, match="single-use pure permutation"):
        emit_c(shared, "dot")
    with pytest.raises(ValueError, match="single-use pure permutation"):
        emit_c(contraction_region(permuted_head_source(2, 3, 4, broadcast=True), with_permutation=True), "dot")
    region = contraction_region(source, with_permutation=True)
    renamed = tuple(replace(node, op="bf16_neg") if node.id == "wt" else node for node in region.nodes)
    with pytest.raises(ValueError, match="single-use pure permutation"):
        emit_c(replace(region, nodes=renamed), "dot")
