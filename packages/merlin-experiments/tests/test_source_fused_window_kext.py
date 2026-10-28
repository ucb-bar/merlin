"""Soundness tests for the movement window chain and the K-extension proofs in ``source_fused_stages``."""

from __future__ import annotations

import pytest

from merlin_experiments.phase0.source_fused_stages import extension_digits, prove_chains, stage_rows

NUMERICS = {"acc_scale_rounding": "half_even", "narrowing": "saturate_to_declared_dtype"}
GAIN = "0.1666666716337204"


def window_program(
    *,
    rows=17,
    cols=17,
    window=2,
    stride=2,
    pad=0,
    high=None,
    relu=True,
    pad_value="0",
    init="-128",
    gains=None,
    second_use=False,
    channels=16,
):
    high = pad if high is None else high
    grid_rows, grid_cols = rows + pad + high, cols + pad + high
    pooled_rows = (grid_rows - window) // stride + 1
    pooled_cols = (grid_cols - window) // stride + 1
    m, p = rows * cols, pooled_rows * pooled_cols
    gains = gains or [GAIN] * channels
    relu_ops = (
        "      %zero = arith.constant 0.0 : f32\n      %act = arith.maximumf %f, %zero : f32\n" if relu else ""
    )
    scaled = "%act" if relu else "%f"
    if pad or high:
        view = f"""\
    %pool_spatial = tensor.expand_shape %readout [[0, 1], [2]] output_shape [{rows}, {cols}, {channels}] : tensor<{m}x{channels}xi8> into tensor<{rows}x{cols}x{channels}xi8>
    %pool_padded = tensor.pad %pool_spatial low[{pad}, {pad}, 0] high[{high}, {high}, 0] {{
    ^bb0(%y: index, %x: index, %c: index):
      %pad_zero = arith.constant {pad_value} : i8
      tensor.yield %pad_zero : i8
    }} : tensor<{rows}x{cols}x{channels}xi8> to tensor<{grid_rows}x{grid_cols}x{channels}xi8>
    %pool_flat = tensor.collapse_shape %pool_padded [[0, 1], [2]] : tensor<{grid_rows}x{grid_cols}x{channels}xi8> into tensor<{grid_rows * grid_cols}x{channels}xi8>
"""
        source, flat = "%pool_flat", grid_rows * grid_cols
    else:
        view, source, flat = "", "%readout", m
    taps = ", ".join(["0"] * (window * window))
    returns = f"%pool, %Image : tensor<{p}x{channels}xi8>, tensor<{m}x{channels}xi8>" if second_use else f"%pool : tensor<{p}x{channels}xi8>"
    result_types = f"tensor<{p}x{channels}xi8>, tensor<{m}x{channels}xi8>" if second_use else f"tensor<{p}x{channels}xi8>"
    return f"""\
builtin.module {{
  func.func @forward(%Image: tensor<{m}x{channels}xi8>) -> ({result_types}) {{
    %readout_gains = arith.constant dense<[{", ".join(gains)}]> : tensor<{channels}xf32>
    %readout_empty = tensor.empty() : tensor<{m}x{channels}xi8>
    %readout = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}} ins(%Image, %readout_gains : tensor<{m}x{channels}xi8>, tensor<{channels}xf32>) outs(%readout_empty : tensor<{m}x{channels}xi8>) {{
    ^bb0(%x0: i8, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : i8 to f32
{relu_ops}      %p = arith.mulf {scaled}, %x1 : f32
      %r = math.roundeven %p : f32
      %lo = arith.constant -128.0 : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    }} -> tensor<{m}x{channels}xi8>
{view}    %pool_window_shape = arith.constant dense<[{taps}]> : tensor<{window * window}xi8>
    %pool_empty = tensor.empty() : tensor<{p}x{channels}xi8>
    %pool_zero = arith.constant {init} : i8
    %pool_init = linalg.fill ins(%pool_zero : i8) outs(%pool_empty : tensor<{p}x{channels}xi8>) -> tensor<{p}x{channels}xi8>
    %pool = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2)->(((d0 floordiv {pooled_cols}) * {stride} + d2 floordiv {window}) * {grid_cols if (pad or high) else cols} + (d0 mod {pooled_cols}) * {stride} + d2 mod {window},d1)>, affine_map<(d0,d1,d2)->(d2)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins({source}, %pool_window_shape : tensor<{flat}x{channels}xi8>, tensor<{window * window}xi8>) outs(%pool_init : tensor<{p}x{channels}xi8>) {{
    ^bb0(%x0: i8, %x1: i8, %dest: i8):
      %v = arith.maxsi %x0, %dest : i8
      linalg.yield %v : i8
    }} -> tensor<{p}x{channels}xi8>
    func.return {returns}
  }}
}}
"""


def prove_window(tmp_path, fenced=True, strategy="f32_product_rne", **kw):
    path = tmp_path / "w.mlir"
    path.write_text(window_program(**kw))
    return prove_chains(path, numerics=NUMERICS, strategy=strategy, pooling_fenced=fenced)


def test_unpadded_window_chain_is_proven_with_its_pooled_store_fields(tmp_path):
    (chain,) = prove_window(tmp_path)
    readout, pool = chain["stages"]
    assert chain["producer"] is None
    assert readout["stages"] == ["relu", "acc_scale"] and pool["stages"] == ["maxpool"]
    assert pool["pooled_store"] == {
        "pool_size": 2, "pool_stride": 2, "orows": 17, "ocols": 17, "porows": 8, "pocols": 8,
        "upad": 0, "lpad": 0, "padded": False, "padding_value": None,
    }


def test_padded_window_after_relu_is_proven(tmp_path):
    (chain,) = prove_window(tmp_path, pad=1)
    fields = chain["stages"][1]["pooled_store"]
    assert (fields["upad"], fields["lpad"], fields["porows"], fields["padded"], fields["padding_value"]) == (1, 1, 9, True, 0)


def test_window_without_relu_and_without_padding_is_proven(tmp_path):
    (chain,) = prove_window(tmp_path, relu=False)
    assert chain["stages"][0]["stages"] == ["acc_scale"]


@pytest.mark.parametrize(
    "kw",
    [
        {"fenced": False},  # the Linalg cannot state the fence; only the declared fence admits
        {"pad": 1, "fenced": False},
        {"window": 4, "stride": 4, "rows": 16, "cols": 16},  # beyond the 2-bit size/stride fields
        {"window": 2, "stride": 4, "rows": 18, "cols": 18},
        {"window": 15, "stride": 15, "rows": 15, "cols": 15},
        {"pad": 1, "relu": False},  # a zero pad tap needs a non-negative producer
        {"pad": 1, "pad_value": "1"},
        {"pad": 1, "pad_value": "-1"},
        {"pad": 0, "high": 1, "rows": 17, "cols": 17},  # reads the high pad
        {"init": "-127"},
        {"init": "0"},
        {"gains": [GAIN] * 15 + ["0.5"]},  # per-column gains: tensor granularity only
        {"gains": ["-0.5"] * 16},  # relu before a negative gain
        {"second_use": True},
        {"strategy": "exact_rational_host"},  # 1/6 is no power of two
    ],
)
def test_window_variants_are_not_observed(tmp_path, kw):
    assert prove_window(tmp_path, **kw) == []


def test_window_chain_is_observed_only_under_the_f32_product_strategy(tmp_path):
    assert prove_window(tmp_path, strategy="exact_rational_host", gains=["0.25"] * 16) == []
    assert prove_window(tmp_path, strategy=None, gains=["0.25"] * 16) == []


def test_window_chain_channels_are_split_per_pooled_block_by_the_lowering(tmp_path):
    # Pool and readout act per column; the per-command column bound is an emission obligation.
    assert len(prove_window(tmp_path, channels=33)) == 1
    assert len(prove_window(tmp_path, channels=8)) == 1


def extension_program(*, coefficient="129 : i32", residual="tensor<16x16xi8>", second_use=False, float_path=False):
    rtype = residual.split("x")[-1].rstrip(">")
    if float_path:
        body = """\
      %rf = arith.sitofp %x1 : i8 to f32
      %c = arith.constant 0.5 : f32
      %pf = arith.mulf %rf, %c : f32
      %pr = math.roundeven %pf : f32
      %p = arith.fptosi %pr : f32 to i32
      %v = arith.addi %x0, %p : i32"""
    else:
        body = f"""\
      %r = arith.extsi %x1 : {rtype} to i32
      %c = arith.constant {coefficient}
      %p = arith.muli %r, %c : i32
      %v = arith.addi %x0, %p : i32"""
    returns = "%readout, %mac0 : tensor<16x16xi8>, tensor<16x16xi32>" if second_use else "%readout : tensor<16x16xi8>"
    result_types = "tensor<16x16xi8>, tensor<16x16xi32>" if second_use else "tensor<16x16xi8>"
    return f"""\
builtin.module {{
  func.func @forward(%A: tensor<16x16xi8>, %W0: tensor<16x16xi8>, %Residual: {residual}) -> ({result_types}) {{
    %mac0_empty = tensor.empty() : tensor<16x16xi32>
    %mac0_zero = arith.constant 0 : i32
    %mac0_init = linalg.fill ins(%mac0_zero : i32) outs(%mac0_empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %mac0 = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%A, %W0 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%mac0_init : tensor<16x16xi32>) {{
    ^bb0(%x0: i8, %x1: i8, %dest: i32):
      %a = arith.extsi %x0 : i8 to i32
      %b = arith.extsi %x1 : i8 to i32
      %p = arith.muli %a, %b : i32
      %v = arith.addi %p, %dest : i32
      linalg.yield %v : i32
    }} -> tensor<16x16xi32>
    %extension_empty = tensor.empty() : tensor<16x16xi32>
    %extension = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%mac0, %Residual : tensor<16x16xi32>, {residual}) outs(%extension_empty : tensor<16x16xi32>) {{
    ^bb0(%x0: i32, %x1: {rtype}, %dest: i32):
{body}
      linalg.yield %v : i32
    }} -> tensor<16x16xi32>
    %readout_gains = arith.constant dense<[{", ".join([GAIN] * 16)}]> : tensor<16xf32>
    %readout_empty = tensor.empty() : tensor<16x16xi8>
    %readout = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}} ins(%extension, %readout_gains : tensor<16x16xi32>, tensor<16xf32>) outs(%readout_empty : tensor<16x16xi8>) {{
    ^bb0(%x0: i32, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : i32 to f32
      %zero = arith.constant 0.0 : f32
      %act = arith.maximumf %f, %zero : f32
      %p = arith.mulf %act, %x1 : f32
      %r = math.roundeven %p : f32
      %lo = arith.constant -128.0 : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    }} -> tensor<16x16xi8>
    func.return {returns}
  }}
}}
"""


def prove_extension(tmp_path, max_k=131071, **kw):
    path = tmp_path / "e.mlir"
    path.write_text(extension_program(**kw))
    return prove_chains(path, numerics=NUMERICS, strategy="f32_product_rne", max_k=max_k)


@pytest.mark.parametrize("coefficient, digits", [(1, [1]), (127, [127]), (128, [127, 1]), (129, [127, 2]), (300, [127, 127, 46])])
def test_extension_digits_are_int8_and_sum_to_the_coefficient(coefficient, digits):
    assert extension_digits(coefficient) == digits


def test_integer_extension_is_proven_with_its_extended_k(tmp_path):
    (chain,) = prove_extension(tmp_path)
    extension, readout = chain["stages"]
    assert extension["stages"] == ["k_extension"] and readout["stages"] == ["relu", "acc_scale"]
    assert {k: extension["k_extension"][k] for k in ("coefficient", "digits", "extended_k")} == {
        "coefficient": 129, "digits": [127, 2], "extended_k": 48,
    }


@pytest.mark.parametrize(
    "kw",
    [
        {"coefficient": "0 : i32"},
        {"coefficient": "-1 : i32"},
        {"float_path": True},  # the non-integer twin's arithmetic
        {"residual": "tensor<16x16xi16>"},
        {"max_k": None},
        {"max_k": 47},  # K' = 16 + 2*16 = 48 exceeds it
        {"second_use": True},
    ],
)
def test_extension_variants_are_not_observed(tmp_path, kw):
    assert prove_extension(tmp_path, **kw) == []


def test_window_chain_rows_are_composed_with_movement(tmp_path):
    (chain,) = prove_window(tmp_path, pad=1)
    rows = stage_rows(chain)
    readout, pool = (rows[stage["member"]] for stage in chain["stages"])
    assert [row["operation"] for row in readout] == ["relu", "acc_scale"]
    assert [row["operation"] for row in pool] == ["maxpool"]
    for row in readout + pool:
        assert row["composed_observation"]["composed_with"] == ["movement"]
        assert row["composed_observation"]["scale_granularity"] == "tensor"
    assert pool[0]["reduction_kind"] == "max"
    assert pool[0]["_observation_extra"]["pooled_store"]["upad"] == 1


def test_extension_rows_carry_the_extended_contraction(tmp_path):
    (chain,) = prove_extension(tmp_path)
    rows = stage_rows(chain)
    (extension,) = rows[chain["stages"][0]["member"]]
    assert extension["semantic_family"] == "contraction"
    assert extension["contraction_shape"] == {"M": 16, "K": 48, "N": 16}
    assert extension["composed_observation"]["epilogues"] == ["k_extension"]
    assert extension["composed_observation"]["composed_with"] == ["contraction"]
