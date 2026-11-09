"""Ordered prefix scans keep separate output and accumulator rounding."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def source(total=6, stride=1, extent=3):
    return f"""module {{
      func.func @forward(%input: tensor<{total}xf32>) -> tensor<{total}xf32> {{
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %end = arith.constant {total} : index
        %stride = arith.constant {stride} : index
        %extent = arith.constant {extent} : index
        %fzero = arith.constant 0.0 : f64
        %empty = tensor.empty() : tensor<{total}xf32>
        %accempty = tensor.empty() : tensor<{total}xf64>
        %result:2 = scf.for %i = %zero to %end step %one
          iter_args(%output = %empty, %acc = %accempty) -> (tensor<{total}xf32>, tensor<{total}xf64>) {{
          %current = tensor.extract %input[%i] : tensor<{total}xf32>
          %wide = arith.extf %current : f32 to f64
          %quotient = arith.divui %i, %stride : index
          %lane = arith.remui %quotient, %extent : index
          %first = arith.cmpi eq, %lane, %zero : index
          %previous = scf.if %first -> (f64) {{
            scf.yield %fzero : f64
          }} else {{
            %back = arith.subi %i, %stride : index
            %prior = tensor.extract %acc[%back] : tensor<{total}xf64>
            scf.yield %prior : f64
          }}
          %sum = arith.addf %previous, %wide : f64
          %rounded = arith.truncf %sum : f64 to f32
          %stored = tensor.insert %rounded into %output[%i] : tensor<{total}xf32>
          %saved = tensor.insert %sum into %acc[%i] : tensor<{total}xf64>
          scf.yield %stored, %saved : tensor<{total}xf32>, tensor<{total}xf64>
        }}
        return %result#0 : tensor<{total}xf32>
      }}
    }}"""


def translate(text):
    return translate_linalg_text(text, entry="forward", target_identity="synthetic")


def test_scan_keeps_exact_source_identity():
    original = source()
    changed = "// Distinct source bytes, identical computation.\n" + original
    first, second = translate(original).request, translate(changed).request
    assert first.source_identity == sha256(original.encode()).hexdigest()
    assert second.source_identity == sha256(changed.encode()).hexdigest()
    assert first.source_identity != second.source_identity
    assert first.nodes == second.nodes and first.outputs == second.outputs
    assert first.digest() != second.digest()


def execute(translated, values):
    graph = translated.request
    return evaluate_graph(
        graph, {"arg0": TensorValue(graph.node("arg0").type, tuple(map(int, values)))}, constants=translated.constants
    )[0].elements


@pytest.mark.parametrize("outer,extent,stride", ((2, 3, 1), (2, 3, 2), (3, 1, 4), (1, 50, 1)))
def test_scan_matches_independent_wide_cumsum_on_all_lanes(outer, extent, stride):
    count = outer * extent * stride
    rng = np.random.default_rng(count)
    values = rng.normal(size=(outer, extent, stride)).astype(np.float32)
    values.reshape(-1)[:3] = [16777216.0, 1.0, -16777216.0]
    expected = np.cumsum(values, axis=1, dtype=np.float64).astype(np.float32)
    translated = translate(source(count, stride, extent))
    assert execute(translated, values.reshape(-1).view(np.uint32)) == tuple(
        map(int, expected.reshape(-1).view(np.uint32))
    )
    assert translated.request.nodes[-1].op == "ordered_f32_scan_f64"
    assert len(translated.request.nodes) == 2
    assert type(translated.request).from_record(translated.request.record()).digest() == translated.request.digest()
    if stride == 1 and extent == 3:
        assert execute(translated, values.reshape(-1).view(np.uint32))[2] == 0x3F800000
        assert np.cumsum(values, axis=1, dtype=np.float32).reshape(-1)[2] == 0


@pytest.mark.parametrize(
    "before,after",
    (
        ("%one = arith.constant 1", "%one = arith.constant 2"),
        ("%stride = arith.constant 1", "%stride = arith.constant 0"),
        ("%extent = arith.constant 3", "%extent = arith.constant 4"),
        ("%fzero = arith.constant 0.0", "%fzero = arith.constant -0.0"),
        ("arith.cmpi eq", "arith.cmpi ne"),
        ("arith.subi %i, %stride", "arith.addi %i, %stride"),
        ("%acc[%back]", "%acc[%i]"),
        ("%rounded into %output[%i]", "%current into %output[%i]"),
        ("%sum into %acc[%i]", "%wide into %acc[%i]"),
        ("scf.yield %prior : f64", "scf.yield %fzero : f64"),
    ),
)
def test_changed_control_initialization_or_rounding_is_rejected(before, after):
    with pytest.raises(ValueError, match="ordered floating scan"):
        translate(source().replace(before, after))


def test_reference_rechecks_policy_and_lane_geometry():
    translated = translate(source())
    for attributes in (
        (("stride", 0), ("extent", 3), ("numeric_semantics", "ordered-f32-prefix-f64-rne-v1")),
        (("stride", 1), ("extent", 3), ("numeric_semantics", "per-step-f32")),
    ):
        changed = replace(translated.request.nodes[-1], attrs=attributes)
        bad = replace(translated, request=replace(translated.request, nodes=(translated.request.nodes[0], changed)))
        with pytest.raises(ValueError, match="lane geometry or numerical policy"):
            execute(bad, (0,) * 6)


def test_scan_signed_zeros_and_nan_arithmetic_scope():
    translated = translate(source())
    assert execute(translated, (0x80000000,) * 6) == (0,) * 6
    with pytest.raises(ValueError, match="NaN"):
        execute(translated, (0x7FC00001,) + (0,) * 5)


def integer_source(dtype="i64"):
    cast = "%extended = arith.extui %value : i1 to i64" if dtype == "i1" else ""
    value = "%extended" if dtype == "i1" else "%value"
    return f"""module {{
      func.func @forward(%input: tensor<2x3x{dtype}>) -> tensor<2x3xi64> {{
        %zero = arith.constant 0 : i64
        %init = tensor.splat %zero : tensor<2x3xi64>
        %result = linalg.generic {{indexing_maps = [affine_map<(a,b,r)->(a,r)>,
          affine_map<(a,b,r)->(a,b)>], iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%input : tensor<2x3x{dtype}>) outs(%init : tensor<2x3xi64>) {{
        ^bb0(%value: {dtype}, %acc: i64):
          %i = linalg.index 1 : index
          %j = linalg.index 2 : index
          %included = arith.cmpi ule, %j, %i : index
          {cast}
          %selected = arith.select %included, {value}, %zero : i64
          %added = arith.addi %acc, %selected : i64
          linalg.yield %added : i64
        }} -> tensor<2x3xi64>
        return %result : tensor<2x3xi64>
      }}
    }}"""


@pytest.mark.parametrize(
    "dtype,values,expected",
    (
        ("i1", (1, 0, 1, 0, 1, 1), (1, 1, 2, 0, 1, 2)),
        ("i64", ((1 << 63) - 1, 1, -1, -2, 4, 1), ((1 << 63) - 1, -(1 << 63), (1 << 63) - 1, -2, 2, 3)),
    ),
)
def test_integer_prefix_sum_preserves_zero_extension_and_modular_arithmetic(dtype, values, expected):
    translated = translate(integer_source(dtype))
    assert execute(translated, values) == expected
    assert translated.request.nodes[-1].op == "i64_cumsum"
    assert dict(translated.request.nodes[-1].attrs)["minimum_index_bits"] == 3


@pytest.mark.parametrize(
    "before,after",
    (
        ("arith.constant 0 : i64", "arith.constant 1 : i64"),
        ("arith.cmpi ule", "arith.cmpi ult"),
        ("arith.extui", "arith.extsi"),
        ("linalg.index 2", "linalg.index 0"),
    ),
)
def test_integer_prefix_sum_changed_seed_or_mask_refuses(before, after):
    with pytest.raises(ValueError):
        translate(integer_source("i1").replace(before, after))
