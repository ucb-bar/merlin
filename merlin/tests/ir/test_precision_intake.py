"""Source-bound precision decisions must describe the parsed contractions."""

import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
from merlin.semantic_compiler.model import IndexMap, KernelRequest
from merlin.semantic_compiler.precision_constants import _trace_argument
from merlin.semantic_compiler.precision_intake import (
    check_source_bound_f32_batched_matmuls,
    check_source_bound_f32_matmuls,
    inventory_source_f32_batched_matmuls,
)
from merlin.semantic_compiler.precision_transform import (
    MatmulTileProfile,
    assemble_diagnostic_matmul_outputs,
    build_diagnostic_batched_matmul_tiles,
    build_diagnostic_matmul_tile,
    build_diagnostic_matmul_tiles,
)
from merlin.semantic_compiler.precision_values import FloatBoundarySpec

SOURCE = """module {
  func.func @work(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>,
                  %c: tensor<4x2xf32>) -> tensor<2x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty0 = tensor.empty() : tensor<2x4xf32>
    %init0 = linalg.fill ins(%zero : f32)
      outs(%empty0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %first = linalg.matmul {prov.region_id = "first"}
      ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
      outs(%init0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %empty1 = tensor.empty() : tensor<2x2xf32>
    %init1 = linalg.fill ins(%zero : f32)
      outs(%empty1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %second = linalg.matmul {prov.region_id = "second"}
      ins(%first, %c : tensor<2x4xf32>, tensor<4x2xf32>)
      outs(%init1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %second : tensor<2x2xf32>
  }
}"""

BATCHED_SOURCE = """module {
  func.func @work(%a: tensor<2x3x4xf32>, %b: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x3x5xf32>
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,
                       affine_map<(b,m,n,k)->(b,k,n)>,
                       affine_map<(b,m,n,k)->(b,m,n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3x4xf32>, tensor<2x4x5xf32>)
      outs(%init : tensor<2x3x5xf32>) {
      ^bb0(%av: f32, %bv: f32, %acc: f32):
        %product = arith.mulf %av, %bv : f32
        %sum = arith.addf %acc, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x3x5xf32>
    func.return %result : tensor<2x3x5xf32>
  }
}"""


def test_batched_source_inventory_keeps_geometry_order_and_value_identity():
    (region,) = inventory_source_f32_batched_matmuls(BATCHED_SOURCE, entry="work")
    assert region.region_id == "batch_matmul#0"
    assert region.batch_shape == (2,)
    assert (region.lhs_shape, region.rhs_shape, region.result_shape) == (
        (2, 3, 4),
        (2, 4, 5),
        (2, 3, 5),
    )
    assert region.scalar_add_order == "accumulator-product"
    assert (region.lhs_source.kind, region.lhs_source.index) == ("argument", 0)
    assert (region.rhs_source.kind, region.rhs_source.index) == ("argument", 1)
    assert (region.result_source.kind, region.result_source.index) == (
        "operation_result",
        region.operation_ordinal,
    )
    reversed_add = BATCHED_SOURCE.replace("%acc, %product", "%product, %acc")
    assert inventory_source_f32_batched_matmuls(reversed_add, entry="work")[0].scalar_add_order == (
        "product-accumulator"
    )


@pytest.mark.parametrize(
    "changed",
    (
        BATCHED_SOURCE.replace("0.000000e+00", "1.000000e+00"),
        BATCHED_SOURCE.replace("outs(%init : tensor<2x3x5xf32>) {", "outs(%empty : tensor<2x3x5xf32>) {"),
        BATCHED_SOURCE.replace("(b,k,n)", "(b,n,k)"),
        BATCHED_SOURCE.replace("arith.addf %acc, %product", "arith.subf %acc, %product"),
    ),
)
def test_batched_source_inventory_rejects_changed_reset_maps_or_arithmetic(changed):
    with pytest.raises(LinalgBridgeError):
        inventory_source_f32_batched_matmuls(changed, entry="work")


def _batched_policy(source: str = BATCHED_SOURCE) -> dict:
    return {
        "source_sha256": sha256(source.encode()).hexdigest(),
        "operand_encoding": "e4m3-rne",
        "matmul_regions": {},
        "batched_matmul_regions": {"batch_matmul#0": {"unit": 0}},
    }


def test_batched_precision_decisions_match_all_source_regions():
    policy = _batched_policy()
    assert [
        region.region_id
        for region in check_source_bound_f32_batched_matmuls(
            BATCHED_SOURCE,
            entry="work",
            policy=policy,
            original_source_sha256=policy["source_sha256"],
        )
    ] == ["batch_matmul#0"]
    with pytest.raises(LinalgBridgeError, match="missing"):
        check_source_bound_f32_batched_matmuls(
            BATCHED_SOURCE,
            entry="work",
            policy={**policy, "batched_matmul_regions": {}},
            original_source_sha256=policy["source_sha256"],
        )
    assert (
        check_source_bound_f32_matmuls(
            BATCHED_SOURCE,
            entry="work",
            policy=policy,
            original_source_sha256=policy["source_sha256"],
        )
        == ()
    )


def test_batched_diagnostic_tiles_preserve_batch_and_cover_all_results():
    encode = FloatBoundarySpec("f32-to-e4m3-rne", "fp8_e4m3", "encode")
    decode = FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")
    profile = replace(_tile_profile(), conversion_specs=(encode, encode, decode))
    args = dict(
        program=BATCHED_SOURCE,
        entry="work",
        policy_document=json.dumps(_batched_policy()).encode(),
        original_source_sha256=sha256(BATCHED_SOURCE.encode()).hexdigest(),
        region_id="batch_matmul#0",
        profile=profile,
        ordered_k_continuation=True,
    )
    tiles = build_diagnostic_batched_matmul_tiles(**args)
    assert len(tiles) == 8
    assert [(tile.batch_index, tile.tile_origin) for tile in tiles] == [
        (batch, origin) for batch in ((0,), (1,)) for origin in ((0, 0), (0, 4), (2, 0), (2, 4))
    ]
    assert len({tile.request.digest() for tile in tiles}) == 8
    assert all(tile.record()["schema"] == "merlin.diagnostic_precision_tile.v4" for tile in tiles)
    lhs = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    rhs = np.arange(1, 41, dtype=np.float32).reshape(2, 4, 5)
    assert not np.array_equal(
        tiles[0].inputs[0].convert_values(lhs),
        tiles[4].inputs[0].convert_values(lhs),
    )
    assert not np.array_equal(
        tiles[0].inputs[1].convert_values(rhs),
        tiles[4].inputs[1].convert_values(rhs),
    )
    windows = []
    for tile in tiles:
        codes = np.full(tile.output.target_type.shape, 0x4000, dtype=np.uint16)
        windows.append(tile.output.convert_values(codes))
    np.testing.assert_array_equal(
        assemble_diagnostic_matmul_outputs(tiles, windows),
        np.full((2, 3, 5), 2.0, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="unwritten"):
        assemble_diagnostic_matmul_outputs(tiles[:4], windows[:4])
    with pytest.raises(ValueError, match="overlap"):
        assemble_diagnostic_matmul_outputs((*tiles, tiles[0]), (*windows, windows[0]))
    with pytest.raises(ValueError, match="window lies outside"):
        replace(tiles[0].inputs[0], source_batch_index=(2,))
    with pytest.raises(LinalgBridgeError, match="batch index"):
        build_diagnostic_matmul_tile(**args, batch_index=(2,))


def _check(source=SOURCE, *, decisions=None, source_identity=None):
    return check_source_bound_f32_matmuls(
        source,
        entry="work",
        policy={
            "source_sha256": source_identity or sha256(source.encode()).hexdigest(),
            "matmul_regions": {"first": {}, "second": {}} if decisions is None else decisions,
        },
        original_source_sha256=sha256(source.encode()).hexdigest(),
    )


def test_matches_two_f32_matmuls_with_distinct_shapes_and_shared_producer():
    regions = _check()
    assert tuple(region.region_id for region in regions) == ("first", "second")
    assert regions[0].lhs_shape == (2, 3) and regions[0].rhs_shape == (3, 4)
    assert regions[1].lhs_shape == (2, 4) and regions[1].result_shape == (2, 2)
    assert regions[0].operation_ordinal < regions[1].operation_ordinal
    assert (regions[0].lhs_source.kind, regions[0].lhs_source.index) == ("argument", 0)
    assert (regions[1].lhs_source.kind, regions[1].lhs_source.index) == (
        "operation_result",
        regions[0].operation_ordinal,
    )


def _splat_initialized_source() -> str:
    source = SOURCE
    for index, shape in ((0, "2x4"), (1, "2x2")):
        old = (
            f"%empty{index} = tensor.empty() : tensor<{shape}xf32>\n"
            f"    %init{index} = linalg.fill ins(%zero : f32)\n"
            f"      outs(%empty{index} : tensor<{shape}xf32>) -> tensor<{shape}xf32>"
        )
        new = f"%init{index} = tensor.splat %zero : tensor<{shape}xf32>"
        assert old in source
        source = source.replace(old, new)
    return source


def test_positive_zero_tensor_splat_is_an_explicit_matmul_reset():
    regions = _check(_splat_initialized_source())
    assert [region.region_id for region in regions] == ["first", "second"]
    assert _build_tile(source=_splat_initialized_source()).request.node("result").op == "matmul"


@pytest.mark.parametrize("value", ("-0.000000e+00", "1.000000e+00"))
def test_tensor_splat_reset_rejects_non_positive_zero(value):
    with pytest.raises(LinalgBridgeError, match="positive-zero"):
        _check(_splat_initialized_source().replace("0.000000e+00", value))


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (SOURCE.replace("0.000000e+00", "-0.000000e+00"), "positive-zero"),
        (SOURCE.replace("0.000000e+00", "1.000000e+00"), "positive-zero"),
        (SOURCE.replace("%init0 : tensor<2x4xf32>", "%empty0 : tensor<2x4xf32>"), "zero-filled"),
        (SOURCE.replace('prov.region_id = "second"', 'prov.region_id = "first"'), "duplicate"),
    ],
)
def test_rejects_nonreset_or_ambiguous_source_matmul(source, error):
    with pytest.raises(LinalgBridgeError, match=error):
        _check(source)


def test_rejects_missing_extra_or_stale_source_decisions():
    with pytest.raises(LinalgBridgeError, match="missing=.*first"):
        _check(decisions={})
    with pytest.raises(LinalgBridgeError, match=r"missing=\['second'\]"):
        _check(decisions={"first": {}})
    with pytest.raises(LinalgBridgeError, match=r"extra=\['third'\]"):
        _check(decisions={"first": {}, "second": {}, "third": {}})
    with pytest.raises(LinalgBridgeError, match="captured source identity"):
        _check(source_identity="0" * 64)


def _tile_profile(*, transpose_rhs=True, maps=None):
    lhs = IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0))
    rhs = IndexMap(3, ((0, 1, 0), (0, 0, 1)) if transpose_rhs else ((0, 0, 1), (0, 1, 0)), (0, 0))
    result = IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0))
    return MatmulTileProfile(
        "synthetic-tile",
        (2, 4, 3),
        maps or (lhs, rhs, result),
        ("fp8_e4m3", "fp8_e4m3"),
        ("raw-bit-pattern", "raw-bit-pattern"),
        ("finite-normal", "finite-normal"),
        ("f32-to-e4m3-rne", "f32-to-e4m3-rne"),
        "bf16",
        "ordered-bf16",
        "bf16-to-f32-exact",
        "external",
        (1, 1),
        (("operand_encoding", "e4m3-rne"),),
        (("unit", 0),),
    )


def _build_tile(*, source=SOURCE, profile=None, policy=None, region_id="first", policy_document=None):
    policy = policy or {
        "source_sha256": sha256(source.encode()).hexdigest(),
        "operand_encoding": "e4m3-rne",
        "matmul_regions": {"first": {"unit": 0}, "second": {"unit": 0}},
    }
    return build_diagnostic_matmul_tile(
        source,
        entry="work",
        policy_document=(json.dumps(policy).encode() if policy_document is None else policy_document),
        original_source_sha256=sha256(source.encode()).hexdigest(),
        region_id=region_id,
        profile=profile or _tile_profile(),
    )


def test_diagnostic_tile_preserves_reindexing_and_conversion_obligations():
    plan = _build_tile()
    assert plan.request.lowering_policy == "diagnostic"
    assert plan.request.node("rhs").type.shape == (4, 3)
    assert plan.inputs[0].target_axes_from_source == (0, 1)
    assert plan.inputs[1].target_axes_from_source == (1, 0)
    assert plan.inputs[1].source_value.record() == {
        "kind": "argument",
        "index": 1,
        "result_index": 0,
    }
    assert plan.output.target_axes_from_source == (0, 1)
    assert plan.request.node("result").type.shape == (2, 4)
    assert plan.request.physical_extents == (("lhs", 1), ("rhs", 1))
    assert KernelRequest.from_record(plan.request.record()).digest() == plan.request.digest()
    assert all(item["status"] == "unmaterialized_boundary_obligation" for item in plan.record()["inputs"])
    assert plan.record()["quality_status"] == "unqualified"
    assert _build_tile(profile=_tile_profile(transpose_rhs=False)).inputs[1].target_axes_from_source == (0, 1)


def test_diagnostic_tile_identity_binds_policy_and_target_facts():
    baseline = _build_tile()
    changed_policy = _build_tile(
        policy_document=json.dumps(
            {
                "source_sha256": sha256(SOURCE.encode()).hexdigest(),
                "operand_encoding": "e4m3-rne",
                "matmul_regions": {"first": {"unit": 0}, "second": {"unit": 0}},
            },
            indent=2,
        ).encode()
    )
    assert baseline.request.digest() != changed_policy.request.digest()
    with pytest.raises(LinalgBridgeError, match="duplicate JSON fields"):
        _build_tile(policy_document=b'{"source_sha256":"one","source_sha256":"two"}')
    with pytest.raises(LinalgBridgeError, match="selected target precision field"):
        _build_tile(
            policy={
                "source_sha256": sha256(SOURCE.encode()).hexdigest(),
                "operand_encoding": "different",
                "matmul_regions": {"first": {"unit": 0}, "second": {"unit": 0}},
            }
        )
    with pytest.raises(LinalgBridgeError, match="missing or unknown fields"):
        _build_tile(
            policy={
                "source_sha256": sha256(SOURCE.encode()).hexdigest(),
                "operand_encoding": "e4m3-rne",
                "matmul_regions": {"first": {"unit": 0}, "second": {"unit": 0}},
                "unreviewed_arithmetic": "ignore-me",
            }
        )
    with pytest.raises(LinalgBridgeError, match="decision differs"):
        _build_tile(
            policy={
                "source_sha256": sha256(SOURCE.encode()).hexdigest(),
                "operand_encoding": "e4m3-rne",
                "matmul_regions": {"first": {"unit": 1}, "second": {"unit": 0}},
            }
        )
    with pytest.raises(LinalgBridgeError, match="shape exceeds"):
        _build_tile(region_id="second")


def test_diagnostic_tile_rejects_map_that_is_not_a_reindexing():
    profile = _tile_profile()
    malformed = IndexMap(3, ((0, 1, 0), (0, 0, 1)), (1, 0))
    with pytest.raises(LinalgBridgeError, match="zero-offset permutation"):
        _build_tile(profile=_tile_profile(maps=(profile.maps[0], malformed, profile.maps[2])))


SMALL_SOURCE = """module {
  func.func @work(%a: tensor<1x2xf32>, %b: tensor<2x2xf32>) -> tensor<1x2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x2xf32>
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %result = linalg.matmul {prov.region_id = "small"}
      ins(%a, %b : tensor<1x2xf32>, tensor<2x2xf32>)
      outs(%init : tensor<1x2xf32>) -> tensor<1x2xf32>
    func.return %result : tensor<1x2xf32>
  }
}"""


def _small_tile(*, profile=None):
    return _build_tile(
        source=SMALL_SOURCE,
        profile=profile,
        region_id="small",
        policy={
            "source_sha256": sha256(SMALL_SOURCE.encode()).hexdigest(),
            "operand_encoding": "e4m3-rne",
            "matmul_regions": {"small": {"unit": 0}},
        },
    )


def test_smaller_source_matmul_has_checked_padding_and_output_crop():
    encode = FloatBoundarySpec("f32-to-e4m3-rne", "fp8_e4m3", "encode")
    decode = FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")
    tile = _small_tile(
        profile=replace(
            _tile_profile(),
            conversion_specs=(encode, encode, decode),
        )
    )
    assert tile.request.node("lhs").type.shape == (2, 3)
    assert tile.request.node("rhs").type.shape == (4, 3)
    assert tile.request.node("result").type.shape == (2, 4)
    assert [item.target_offsets for item in tile.inputs] == [(0, 1), (0, 1)]
    assert tile.record()["inputs"][1]["target_offsets"] == [0, 1]
    assert tile.record()["inputs"][1]["padding_value"] == "positive_zero_f32"
    lhs = tile.inputs[0].convert_values(np.array([[1.0, 2.0]], dtype=np.float32))
    rhs = tile.inputs[1].convert_values(np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
    np.testing.assert_array_equal(lhs, np.array([[0, 56, 64], [0, 0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(
        rhs,
        np.array(
            [
                [0, 68, 74],
                [0, 72, 76],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    result_codes = np.full((2, 4), 0x7FC0, dtype=np.uint16)
    result_codes[0, :2] = (0x4140, 0x4180)
    np.testing.assert_array_equal(
        tile.output.convert_values(result_codes),
        np.array([[12.0, 16.0]], dtype=np.float32),
    )
    untransposed = _small_tile(
        profile=replace(
            _tile_profile(transpose_rhs=False),
            conversion_specs=(encode, encode, decode),
        )
    )
    assert untransposed.inputs[1].target_offsets == (1, 0)
    np.testing.assert_array_equal(
        untransposed.inputs[1].convert_values(np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)),
        np.array([[0, 0, 0, 0], [68, 72, 0, 0], [74, 76, 0, 0]], dtype=np.uint8),
    )


def test_boundary_window_rejects_out_of_bounds_or_invalid_offsets():
    tile = _small_tile()
    with pytest.raises(ValueError, match="window lies outside"):
        replace(tile.inputs[0], target_offsets=(0, 2))
    with pytest.raises(ValueError, match="window lies outside"):
        replace(tile.inputs[0], target_offsets=(-1, 1))
    with pytest.raises(ValueError, match="window lies outside"):
        replace(tile.inputs[0], target_axes_from_source=(0, 0))
    encode = FloatBoundarySpec("f32-to-e4m3-rne", "fp8_e4m3", "encode")
    typed = replace(tile.inputs[0], spec=encode)
    with pytest.raises(ValueError, match="dtype differs"):
        typed.convert_values(np.ones((1, 2), dtype=np.float64))


LARGE_MN_SOURCE = """module {
  func.func @work(%a: tensor<3x2xf32>, %b: tensor<2x5xf32>) -> tensor<3x5xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<3x5xf32>
    %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<3x5xf32>) -> tensor<3x5xf32>
    %result = linalg.matmul {prov.region_id = "large"}
      ins(%a, %b : tensor<3x2xf32>, tensor<2x5xf32>)
      outs(%init : tensor<3x5xf32>) -> tensor<3x5xf32>
    func.return %result : tensor<3x5xf32>
  }
}"""


def _large_policy() -> bytes:
    return json.dumps(
        {
            "source_sha256": sha256(LARGE_MN_SOURCE.encode()).hexdigest(),
            "operand_encoding": "e4m3-rne",
            "matmul_regions": {"large": {"unit": 0}},
        }
    ).encode()


def test_larger_mn_contraction_partitions_full_source_windows_without_overlaps():
    encode = FloatBoundarySpec("f32-to-e4m3-rne", "fp8_e4m3", "encode")
    decode = FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")
    profile = replace(_tile_profile(), conversion_specs=(encode, encode, decode))
    args = dict(
        program=LARGE_MN_SOURCE,
        entry="work",
        policy_document=_large_policy(),
        original_source_sha256=sha256(LARGE_MN_SOURCE.encode()).hexdigest(),
        region_id="large",
        profile=profile,
    )
    with pytest.raises(LinalgBridgeError, match="shape exceeds"):
        build_diagnostic_matmul_tile(**args)
    tiles = build_diagnostic_matmul_tiles(**args)
    assert [tile.tile_origin for tile in tiles] == [(0, 0), (0, 4), (2, 0), (2, 4)]
    assert len({tile.request.digest() for tile in tiles}) == 4
    assert [tile.output.source_window_shape for tile in tiles] == [
        (2, 4),
        (2, 1),
        (1, 4),
        (1, 1),
    ]
    lhs = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    rhs = np.arange(1, 11, dtype=np.float32).reshape(2, 5)
    last = tiles[-1]
    assert last.inputs[0].source_type.shape == lhs.shape
    assert last.inputs[0].source_offsets == (2, 0)
    assert last.inputs[1].source_type.shape == rhs.shape
    assert last.inputs[1].source_offsets == (0, 4)
    assert last.inputs[1].source_window_shape == (2, 1)
    np.testing.assert_array_equal(
        last.inputs[0].convert_values(lhs),
        np.array([[0, 74, 76], [0, 0, 0]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        last.inputs[1].convert_values(rhs),
        np.array([[0, 74, 82], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
    )
    windows = []
    for tile in tiles:
        valid_m, valid_n = tile.output.source_window_shape
        codes = np.zeros((2, 4), dtype=np.uint16)
        codes[:valid_m, :valid_n] = 0x4000  # BF16 2.0
        windows.append(tile.output.convert_values(codes))
    output = assemble_diagnostic_matmul_outputs(tiles, windows)
    np.testing.assert_array_equal(output, np.full((3, 5), 2.0, dtype=np.float32))
    assert tiles[-1].record()["output"]["source_offsets"] == [2, 4]

    with pytest.raises(ValueError, match="overlap"):
        assemble_diagnostic_matmul_outputs((*tiles, tiles[0]), (*windows, windows[0]))
    with pytest.raises(ValueError, match="unwritten"):
        assemble_diagnostic_matmul_outputs(tiles[:-1], windows[:-1])
    with pytest.raises(ValueError, match="different sources"):
        assemble_diagnostic_matmul_outputs(
            (tiles[0], replace(tiles[1], precision_policy_sha256="0" * 64)),
            windows[:2],
        )
    with pytest.raises(ValueError, match="origin"):
        assemble_diagnostic_matmul_outputs(
            (*tiles[:-1], replace(tiles[-1], tile_origin=(2, 0))),
            windows,
        )
    with pytest.raises(ValueError, match="shape or dtype"):
        assemble_diagnostic_matmul_outputs(tiles, (*windows[:-1], windows[-1].astype(np.float64)))


def test_mn_tiling_rejects_bad_origins_and_keeps_single_tile_identity():
    args = dict(
        program=LARGE_MN_SOURCE,
        entry="work",
        policy_document=_large_policy(),
        original_source_sha256=sha256(LARGE_MN_SOURCE.encode()).hexdigest(),
        region_id="large",
        profile=_tile_profile(),
    )
    with pytest.raises(LinalgBridgeError, match="tile origin"):
        build_diagnostic_matmul_tile(**args, tile_origin=(1, 0))
    with pytest.raises(LinalgBridgeError, match="tile origin"):
        build_diagnostic_matmul_tile(**args, tile_origin=(0, 8))
    assert (
        build_diagnostic_matmul_tiles(
            SMALL_SOURCE,
            entry="work",
            policy_document=json.dumps(
                {
                    "source_sha256": sha256(SMALL_SOURCE.encode()).hexdigest(),
                    "operand_encoding": "e4m3-rne",
                    "matmul_regions": {"small": {"unit": 0}},
                }
            ).encode(),
            original_source_sha256=sha256(SMALL_SOURCE.encode()).hexdigest(),
            region_id="small",
            profile=_tile_profile(),
        )[0].request.digest()
        == _small_tile().request.digest()
    )


def test_source_bound_k_continuation_preserves_order_and_head_pads_first_segment():
    source = LARGE_MN_SOURCE.replace("3x2xf32", "3x7xf32").replace("2x5xf32", "7x5xf32")
    policy = json.dumps(
        {
            "source_sha256": sha256(source.encode()).hexdigest(),
            "operand_encoding": "e4m3-rne",
            "matmul_regions": {"large": {"unit": 0}},
        }
    ).encode()
    args = dict(
        program=source,
        entry="work",
        policy_document=policy,
        original_source_sha256=sha256(source.encode()).hexdigest(),
        region_id="large",
        profile=_tile_profile(),
        tile_origin=(0, 4),
    )
    with pytest.raises(LinalgBridgeError, match="shape exceeds"):
        build_diagnostic_matmul_tile(**args)
    plan = build_diagnostic_matmul_tile(**args, ordered_k_continuation=True)
    assert plan.record()["schema"] == "merlin.diagnostic_precision_tile.v3"
    assert [item.source_role for item in plan.inputs] == [
        "lhs__k0",
        "rhs__k0",
        "lhs__k1",
        "rhs__k1",
        "lhs__k2",
        "rhs__k2",
    ]
    assert [item.source_offsets for item in plan.inputs] == [
        (0, 0),
        (0, 4),
        (0, 1),
        (1, 4),
        (0, 4),
        (4, 4),
    ]
    assert [item.window_shape for item in plan.inputs] == [
        (2, 1),
        (1, 1),
        (2, 3),
        (3, 1),
        (2, 3),
        (3, 1),
    ]
    assert [item.target_offsets for item in plan.inputs] == [
        (0, 2),
        (0, 2),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]
    assert [item["padding_value"] for item in plan.record()["inputs"]] == [
        "positive_zero_f32",
        "positive_zero_f32",
        None,
        "positive_zero_f32",
        None,
        "positive_zero_f32",
    ]
    assert [node.op for node in plan.request.nodes] == [
        *("input",) * 6,
        "matmul",
        "matmul_accumulate",
        "matmul_accumulate",
    ]
    assert plan.request.node("result").inputs[-1] == "result__k1"
    multi = build_diagnostic_matmul_tiles(
        source,
        entry="work",
        policy_document=policy,
        original_source_sha256=sha256(source.encode()).hexdigest(),
        region_id="large",
        profile=_tile_profile(),
        ordered_k_continuation=True,
    )
    assert len(multi) == 4
    assert all(tile.ordered_k_continuation for tile in multi)

    small = build_diagnostic_matmul_tile(
        SMALL_SOURCE,
        entry="work",
        policy_document=json.dumps(
            {
                "source_sha256": sha256(SMALL_SOURCE.encode()).hexdigest(),
                "operand_encoding": "e4m3-rne",
                "matmul_regions": {"small": {"unit": 0}},
            }
        ).encode(),
        original_source_sha256=sha256(SMALL_SOURCE.encode()).hexdigest(),
        region_id="small",
        profile=_tile_profile(),
        ordered_k_continuation=True,
    )
    assert small.request.digest() == _small_tile().request.digest()
    assert not small.ordered_k_continuation


def test_whole_contraction_keeps_full_source_windows_and_one_reduction():
    source = LARGE_MN_SOURCE.replace("3x2xf32", "3x7xf32").replace("2x5xf32", "7x5xf32")
    digest = sha256(source.encode()).hexdigest()
    encode = FloatBoundarySpec("f32-to-e4m3-rne", "fp8_e4m3", "encode")
    decode = FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")
    args = dict(
        program=source,
        entry="work",
        original_source_sha256=digest,
        region_id="large",
        profile=replace(_tile_profile(), conversion_specs=(encode, encode, decode)),
        policy_document=json.dumps(
            {
                "source_sha256": digest,
                "operand_encoding": "e4m3-rne",
                "matmul_regions": {"large": {"unit": 0}},
            }
        ).encode(),
        whole_contraction=True,
    )
    plan = build_diagnostic_matmul_tile(**args)
    assert [node.type.shape for node in plan.request.nodes] == [(3, 7), (5, 7), (3, 5)]
    assert [node.op for node in plan.request.nodes] == ["input", "input", "matmul"]
    assert plan.request.physical_extents == ()
    assert plan.record()["schema"] == "merlin.diagnostic_precision_contraction.v1"
    assert all(item.target_offsets == item.source_offsets == (0, 0) for item in (*plan.inputs, plan.output))
    rhs = np.arange(1, 36, dtype=np.float32).reshape(7, 5)
    np.testing.assert_array_equal(plan.inputs[1].convert_values(rhs), encode.encode(rhs.T))
    with pytest.raises(LinalgBridgeError, match="tile origin"):
        build_diagnostic_matmul_tile(**args, tile_origin=(0, 0))
    with pytest.raises(LinalgBridgeError, match="positive-zero"):
        build_diagnostic_matmul_tile(**{**args, "program": source.replace("0.000000e+00", "1.000000e+00")})


def test_boundary_conversion_reindexes_and_encodes_logical_values():
    profile = _tile_profile()
    encode = FloatBoundarySpec(
        "f32-to-e4m3-rne",
        "fp8_e4m3",
        "encode",
        max_normal_exponent_code=14,
        flush_below_normal=True,
        admitted_code_ranges=((0, 0), (8, 119), (136, 247)),
    )
    decode = FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode")
    profile = replace(profile, conversion_specs=(encode, encode, decode))
    tile = _build_tile(profile=profile)
    rhs = np.array(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 0.25, 0.125, 0.0625], [1.0625, 1.1875, 240.0, 1000.0]], dtype=np.float32
    )
    codes = tile.inputs[1].convert_values(rhs)
    assert codes.shape == (4, 3)
    np.testing.assert_array_equal(
        codes,
        np.array(
            [
                [56, 48, 56],
                [64, 40, 58],
                [68, 32, 119],
                [72, 24, 119],
            ],
            dtype=np.uint8,
        ),
    )
    assert tile.record()["inputs"][1]["conversion_spec"] == encode.record()
    assert tile.request.source_identity != _build_tile().request.source_identity
    bf16 = np.full((2, 4), 0x3F80, dtype=np.uint16)
    bf16[1, 2] = 0xC000
    result = tile.output.convert_values(bf16)
    assert result.dtype == np.float32 and result.shape == (2, 4)
    assert result[0, 0] == 1.0 and result[1, 2] == -2.0
    with pytest.raises(ValueError, match="shape differs"):
        tile.inputs[1].convert_values(rhs[:, :3])
    with pytest.raises(ValueError, match="in-range"):
        tile.output.convert_values(np.full((2, 4), 0x10000, dtype=np.uint32))
    with pytest.raises(ValueError, match="no matching executable specification"):
        _build_tile().inputs[0].convert_values(np.ones((2, 3), dtype=np.float32))


def test_boundary_conversion_refuses_disallowed_codes_and_nonfinite_source():
    full_finite = FloatBoundarySpec("full-finite", "fp8_e4m3", "encode")
    np.testing.assert_array_equal(
        full_finite.encode(np.array([448.0, 500.0], dtype=np.float32)),
        np.array([126, 126], dtype=np.uint8),
    )
    spec = FloatBoundarySpec(
        "restricted",
        "fp8_e4m3",
        "encode",
        max_normal_exponent_code=14,
        flush_below_normal=True,
        admitted_code_ranges=((0, 0), (56, 56), (64, 64)),
    )
    np.testing.assert_array_equal(
        spec.encode(np.array([1.0, 2.0, 0.001], dtype=np.float32)),
        np.array([56, 64, 0], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="input domain"):
        spec.encode(np.array([-0.001], dtype=np.float32))
    with pytest.raises(ValueError, match="input domain"):
        spec.encode(np.array([1.125], dtype=np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        spec.encode(np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="ordered"):
        FloatBoundarySpec("invalid", "fp8_e4m3", "encode", admitted_code_ranges=((8, 20), (20, 30)))


def test_raised_normal_exponent_floor_requires_flushing():
    # Without flushing, encode would still emit codes below the declared
    # normal floor (2**-6 as 0x08, 2**-9 as subnormal 0x01).
    for direction in ("encode", "decode"):
        with pytest.raises(ValueError, match="normal exponent floor"):
            FloatBoundarySpec("raised", "fp8_e4m3", direction, min_normal_exponent_code=3)
    spec = FloatBoundarySpec(
        "raised",
        "fp8_e4m3",
        "encode",
        min_normal_exponent_code=3,
        flush_below_normal=True,
    )
    np.testing.assert_array_equal(
        spec.encode(np.array([2.0**-6, 2.0**-9, 2.0**-4], dtype=np.float32)),
        np.array([0x00, 0x00, 0x18], dtype=np.uint8),
    )


def test_power_of_two_boundary_scale_is_explicit_and_domain_checked():
    common = dict(max_normal_exponent_code=14, flush_below_normal=True)
    activation = FloatBoundarySpec(
        "scaled-activation",
        "fp8_e4m3",
        "encode",
        admitted_code_ranges=((0, 0), (8, 119), (136, 247)),
        pre_encode_scale_exponent=5,
        **common,
    )
    weight = FloatBoundarySpec(
        "compensated-weight",
        "fp8_e4m3",
        "encode",
        admitted_code_ranges=((0, 0), (8, 119)),
        pre_encode_scale_exponent=-5,
        **common,
    )
    assert activation.record()["pre_encode_scale_exponent"] == 5
    assert weight.record()["pre_encode_scale_exponent"] == -5
    np.testing.assert_array_equal(
        activation.encode(np.array([0.5, 1.0, -0.5], dtype=np.float32)),
        np.array([0x58, 0x60, 0xD8], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        weight.encode(np.array([0.5, 1.0], dtype=np.float32)),
        np.array([0x08, 0x10], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="input domain"):
        weight.encode(np.array([-0.5], dtype=np.float32))
    with pytest.raises(ValueError, match="pre-encode scale"):
        FloatBoundarySpec("invalid", "bf16", "decode", pre_encode_scale_exponent=5)
    with pytest.raises(ValueError, match="scale exponent"):
        FloatBoundarySpec("invalid", "fp8_e4m3", "encode", pre_encode_scale_exponent=33)


def test_static_source_trace_follows_only_parsed_views_to_arguments():
    regions = _check()
    assert _trace_argument(SOURCE, "work", regions[0].rhs_source) == (1, ())
    assert _trace_argument(SOURCE, "work", regions[1].lhs_source) is None
    transposed = """module {
      func.func @work(%w: tensor<2x3xf32>) -> tensor<3x2xf32> {
        %empty = tensor.empty() : tensor<3x2xf32>
        %view = linalg.transpose ins(%w:tensor<2x3xf32>)
          outs(%empty:tensor<3x2xf32>) permutation = [1, 0]
        func.return %view : tensor<3x2xf32>
      }
    }"""
    from merlin.semantic_compiler.precision_intake import SourceValueRef

    assert _trace_argument(transposed, "work", SourceValueRef("operation_result", 1)) == (0, ((1, 0),))
    with pytest.raises(ValueError, match="out of range"):
        _trace_argument(transposed, "work", SourceValueRef("operation_result", 4))


def test_contraction_scale_restoration_preserves_units():
    from merlin.semantic_compiler.precision_values import contraction_restore_exponent

    # Exact dyadic operands isolate scale bookkeeping from quantization loss.
    lhs = np.array([[1.5, -0.25], [-2.0, 0.5]], dtype=np.float64)
    rhs = np.array([[0.5, 1.25], [-1.0, 0.25]], dtype=np.float64)
    for a in range(-5, 6):
        for b in range(-5, 6):
            physical = np.ldexp(lhs, a) @ np.ldexp(rhs, b)
            restored = np.ldexp(physical, contraction_restore_exponent(a, b))
            np.testing.assert_array_equal(restored, lhs @ rhs)
    for invalid in (True, 0.0, 33, -33):
        with pytest.raises(ValueError, match="encoding exponents"):
            contraction_restore_exponent(invalid, 0)
        with pytest.raises(ValueError, match="encoding exponents"):
            contraction_restore_exponent(0, invalid)
