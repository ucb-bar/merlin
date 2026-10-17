"""``layer-norm-chunked-sums``: both numerical permissions, a strict scope, and a default-off seam.

The pass is numerics-changing, so what is pinned here is what it may touch and when it runs, not bit
equality: it needs both explicit permissions, it leaves every reduction outside its scope byte-identical,
it re-applies to nothing, its rewritten sum equals the source sum on inputs whose partial sums are exact,
and the lowering's xDSL preprocessing runs it only when the pass is selected.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import optional_passes as OP
from merlin.llvmlower.normalization_reassociation import MARKER, chunk_layer_norm_sums
from merlin.llvmlower.passes_xdsl import preprocess_text, preprocess_text_with_transform_map
from merlin.runtime.linalg_numpy import evaluate
from merlin.xdsl_dialects._common import text

PERMITTED = {"allow_reassociation": True, "assume_finite_intermediates": True}


@pytest.fixture(autouse=True)
def _no_ambient_selection(monkeypatch):
    monkeypatch.delenv(OP.ENV, raising=False)
    monkeypatch.delenv("MERLIN_LAYER_NORM_CHUNKED_SUMS", raising=False)


def source(*, width=128, dtype="f32", initial="0.0", operation="arith.addf", provenance="layer_norm"):
    return f"""module {{
      func.func @forward(%x: tensor<2x{width}x{dtype}>) -> tensor<2x{dtype}> {{
        %zero = arith.constant {initial} : {dtype}
        %init = "tensor.splat"(%zero) : ({dtype}) -> tensor<2x{dtype}>
        %sum = "linalg.reduce"(%x, %init) <{{dimensions = array<i64: 1>}}> ({{
        ^bb0(%value: {dtype}, %acc: {dtype}):
          %next = "{operation}"(%value, %acc) : ({dtype}, {dtype}) -> {dtype}
          "linalg.yield"(%next) : ({dtype}) -> ()
        }}) {{prov.op = "{provenance}", prov.source_node_ids = ["norm:1"]}}
          : (tensor<2x{width}x{dtype}>, tensor<2x{dtype}>) -> tensor<2x{dtype}>
        func.return %sum : tensor<2x{dtype}>
      }}
    }}"""


def module(**kwargs):
    return parse_mlir_text(source(**kwargs))


@pytest.mark.parametrize("allow,finite", [(False, False), (False, True), (True, False)])
def test_requires_both_explicit_numerical_permissions(allow, finite):
    m = module()
    original = text(m)
    report = {}
    assert (
        chunk_layer_norm_sums(m, allow_reassociation=allow, assume_finite_intermediates=finite, report_out=report) == 0
    )
    assert not report["opted_in"]
    assert text(m) == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 63},
        {"width": 100},
        {"width": 32},
        {"dtype": "bf16"},
        {"dtype": "f64"},
        {"initial": "1.0"},
        {"initial": "-0.0"},
        {"operation": "arith.maximumf"},
        {"provenance": "softmax"},
    ],
)
def test_leaves_every_reduction_outside_its_scope_unchanged(kwargs):
    m = module(**kwargs)
    original = text(m)
    assert chunk_layer_norm_sums(m, **PERMITTED) == 0
    assert text(m) == original


def test_static_extents_provenance_and_idempotence():
    m = module(width=4096)
    report = {}
    assert chunk_layer_norm_sums(m, **PERMITTED, report_out=report) == 1
    assert report == {"rewritten": 1, "opted_in": True}
    m.verify()
    expanded = next(op for op in m.walk() if op.name == "tensor.expand_shape")
    assert expanded.results[0].type.get_shape() == (2, 128, 32)
    partial = next(op for op in m.walk() if op.name == "linalg.generic")
    assert partial.results[0].type.get_shape() == (2, 128)
    assert partial.attributes["prov.source_node_ids"].data[0].data == "norm:1"
    assert partial.attributes["prov.transforms"].data == "layer_norm_chunked_sum"
    assert MARKER in partial.attributes
    # Every loop extent is an operand dimension: no reduction size hides in a compound affine expression.
    assert partial.indexing_maps.data[0].data.is_minor_identity()
    reduce = next(op for op in m.walk() if op.name == "linalg.reduce")
    assert reduce.operands[0] is partial.results[0] and MARKER in reduce.attributes
    reparsed = parse_mlir_text(text(m))
    reparsed.verify()
    assert chunk_layer_norm_sums(reparsed, **PERMITTED) == 0


def test_rejects_an_invalid_chunk_size():
    for bad in (1, 0, 2.0):
        with pytest.raises(ValueError, match="at least two"):
            chunk_layer_norm_sums(module(), chunk_size=bad)


def test_rewritten_sum_equals_the_source_sum_where_every_partial_sum_is_exact():
    rng = np.random.default_rng(7)
    exact = rng.integers(-64, 64, size=(2, 256)).astype(np.float32)
    noisy = rng.standard_normal((2, 256)).astype(np.float32)
    rewritten = module(width=256)
    assert chunk_layer_norm_sums(rewritten, **PERMITTED) == 1
    for x in (exact, noisy):
        (want,) = evaluate(module(width=256), [x])
        (got,) = evaluate(rewritten, [x])
        if x is exact:
            # Small integers: every partial sum is representable, so the order cannot change a bit.
            assert np.array_equal(got, want) and np.array_equal(got, x.sum(axis=1))
        else:
            # A different association: close, and not claimed to be identical.
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_preprocessing_runs_it_only_when_selected(monkeypatch):
    raw = source(width=256)
    baseline, stats = preprocess_text(raw)
    assert MARKER not in baseline and "layer_norm_sums_chunked" not in stats

    monkeypatch.setenv(OP.ENV, "layer-norm-chunked-sums")
    selected, stats = preprocess_text(raw)
    assert stats["layer_norm_sums_chunked"] == 1 and MARKER in selected
    upstream, stats, transform_map = preprocess_text_with_transform_map(raw)
    assert upstream == selected and stats["layer_norm_sums_chunked"] == 1
    reduce = next(row for row in transform_map["operations"] if row["source_op_name"] == "linalg.reduce")
    # The reduction owns the five operations it added plus itself; nothing is left without an owner.
    assert len(reduce["preprocessed_op_indices"]) == 6 and len(reduce["result_map"]) == 1

    monkeypatch.setenv(OP.ENV, "-layer-norm-chunked-sums")
    monkeypatch.setenv("MERLIN_LAYER_NORM_CHUNKED_SUMS", "1")
    assert preprocess_text(raw)[0] == baseline  # an explicit deselection wins over the variable
    monkeypatch.delenv(OP.ENV)
    assert MARKER in preprocess_text(raw)[0]  # the variable alone still selects it, for A/B builds


@pytest.mark.parametrize("preprocess", [preprocess_text, preprocess_text_with_transform_map])
def test_selected_rewrite_has_a_contiguous_recorded_boundary(tmp_path, monkeypatch, preprocess):
    from merlin.verify.lean import stages

    monkeypatch.setenv(OP.ENV, "layer-norm-chunked-sums")
    raw = source(width=256)
    plain = preprocess(raw)
    monkeypatch.setenv(stages.ENV, str(tmp_path / "stages"))
    recorded = preprocess(raw)
    assert recorded == plain
    boundaries = stages.completed(tmp_path / "stages")
    assert [json.loads((p / "meta.json").read_text())["boundary"] for p in boundaries] == [
        "xdsl-parse",
        "xdsl-pruned",
        "xdsl-quant-lowered",
        "xdsl-layer-norm-chunked-sums",
        "xdsl-c-interface",
    ]
    for prev, nxt in zip(boundaries, boundaries[1:]):
        assert (prev / "after.mlir").read_bytes() == (nxt / "before.mlir").read_bytes()
    rewrite = boundaries[3]
    assert json.loads((rewrite / "meta.json").read_text())["changed"]
    assert MARKER not in (rewrite / "before.mlir").read_text()
    assert MARKER in (rewrite / "after.mlir").read_text()
    assert (boundaries[-1] / "after.mlir").read_text() == recorded[0]
