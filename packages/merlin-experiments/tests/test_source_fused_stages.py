"""Soundness tests for ``source_fused_stages.prove_chains``: only the exact written patterns are proven."""

from __future__ import annotations

import pytest

from merlin_experiments.phase0.source_fused_stages import prove_chains, stage_rows

NUMERICS = {"acc_scale_rounding": "half_even", "narrowing": "saturate_to_declared_dtype"}
GAIN = "0.1666666716337204"

_CONTRACTION = """\
    %mac0_empty = tensor.empty() : tensor<16x16xi32>
    %mac0_zero = arith.constant 0 : i32
    %mac0_init = linalg.fill ins(%mac0_zero : i32) outs(%mac0_empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %mac0 = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A, %W0 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%mac0_init : tensor<16x16xi32>) attrs = {prov.composition_stage = "mac0"} {
    ^bb0(%x0: i8, %x1: i8, %dest: i32):
      %a = arith.EXT %x0 : i8 to i32
      %b = arith.EXT %x1 : i8 to i32
      %p = arith.muli %a, %b : i32
      %v = arith.addi %p, %dest : i32
      linalg.yield %v : i32
    } -> tensor<16x16xi32>
"""
_BIAS = """\
    %bias_empty = tensor.empty() : tensor<16x16xi32>
    %bias = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, BIAS_MAP, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%mac0, %Bias : tensor<16x16xi32>, BIAS_TYPE) outs(%bias_empty : tensor<16x16xi32>) attrs = {prov.composition_stage = "bias"} {
    ^bb0(%x0: i32, %x1: i32, %dest: i32):
      %v = arith.addi BIAS_OPERANDS : i32
      linalg.yield %v : i32
    } -> tensor<16x16xi32>
"""
_GAINS = "    %readout_gains = arith.constant dense<[GAINS]> : tensor<16xf32>\n"
_READOUT = """\
    %readout_empty = tensor.empty() : tensor<16x16xOUT>
    %readout = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%SRC, %readout_gains : tensor<16x16xi32>, tensor<16xf32>) outs(%readout_empty : tensor<16x16xOUT>) attrs = {prov.composition_stage = "readout"} {
    ^bb0(%x0: i32, %x1: f32, %dest: iOUT_BITS):
      %f = arith.sitofp %x0 : i32 to f32
RELU_OPS      %p = arith.mulf %SCALED, %x1 FASTMATH : f32
      %r = math.REND %p : f32
      %lo = arith.constant LO : f32
      %hi = arith.constant HI : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to OUT
      linalg.yield %v : OUT
    } -> tensor<16x16xOUT>
"""
_RELU = "      %zero = arith.constant ZERO : f32\n      %act = arith.maximumf %f, %zero : f32\n"


#: Written bias forms: (indexing map, type, constant rows or None for a runtime argument).
_ROW = list(range(-8, 8))
BIAS_FORMS = {
    "column": ("affine_map<(d0, d1) -> (d1)>", "tensor<16xi32>", None),
    "matrix": ("affine_map<(d0, d1) -> (d0, d1)>", "tensor<16x16xi32>", None),
    "rows_equal": ("affine_map<(d0, d1) -> (d0, d1)>", "tensor<16x16xi32>", [_ROW] * 16),
    "rows_distinct": ("affine_map<(d0, d1) -> (d0, d1)>", "tensor<16x16xi32>", [_ROW] * 15 + [_ROW[::-1]]),
}


def build(*, bias=True, bias_form="column", relu=True, ext="extsi", bias_operands="%x0, %x1", gain=GAIN, gain_arg=False,
          fastmath="", rend="roundeven", lo="-128.0", hi="127.0", out="i8", zero="0.0", second_use=False):
    """The contraction -> bias -> readout template (a composition capsule's Linalg), with substitutions."""
    ret_type = "tensor<16x16xOUT>, tensor<16x16xi32>" if second_use else "tensor<16x16xOUT>"
    bias_map, bias_type, bias_rows = BIAS_FORMS[bias_form]
    args = "%A: tensor<16x16xi8>, %W0: tensor<16x16xi8>"
    if bias_rows is None:
        args += ", %Bias: " + bias_type
    if gain_arg:
        args += ", %readout_gains: tensor<16xf32>"
    gains = "" if gain_arg else _GAINS.replace("GAINS", ", ".join([gain] * 16))
    text = "builtin.module {\n  func.func @forward(" + args + ") -> (" + ret_type + ") {\n"
    text += _CONTRACTION.replace("EXT", ext)
    if bias and bias_rows is not None:
        dense = "[" + ", ".join("[" + ", ".join(map(str, row)) + "]" for row in bias_rows) + "]"
        text += f"    %Bias = arith.constant dense<{dense}> : {bias_type}\n"
    text += _BIAS.replace("BIAS_OPERANDS", bias_operands).replace("BIAS_MAP", bias_map).replace("BIAS_TYPE", bias_type) if bias else ""
    text += gains + _READOUT
    ret = "%readout, %mac0 : " if second_use else "%readout : "
    text += "    func.return " + ret + ret_type + "\n  }\n}\n"
    text = text.replace("RELU_OPS", _RELU.replace("ZERO", zero) if relu else "")
    text = text.replace("%SCALED", "%act" if relu else "%f").replace("%SRC", "%bias" if bias else "%mac0")
    text = text.replace("OUT_BITS", out[1:]).replace("OUT", out).replace("REND", rend)
    text = text.replace("FASTMATH", f"fastmath<{fastmath}>" if fastmath else "").replace("LO", lo).replace("HI", hi)
    return text


def prove(tmp_path, numerics=NUMERICS, strategy="f32_product_rne", full_tensor_seed=False, **kw):
    path = tmp_path / "capsule.interface.mlir"
    path.write_text(build(**kw))
    return prove_chains(path, numerics=numerics, strategy=strategy, full_tensor_seed=full_tensor_seed)


def stage_names(chains):
    return [name for chain in chains for stage in chain["stages"] for name in stage["stages"]]


def members(chains):
    return [(s["member"], s["stages"]) for c in chains for s in c["stages"]]


def test_template_proves_bias_then_relu_scale(tmp_path):
    chains = prove(tmp_path)
    assert [c["producer"] for c in chains] == [3]
    assert members(chains) == [(5, ["bias_add"]), (8, ["relu", "acc_scale"])]


def test_without_bias_proves_one_readout_stage(tmp_path):
    chains = prove(tmp_path, bias=False)
    assert [c["producer"] for c in chains] == [3]
    assert members(chains) == [(6, ["relu", "acc_scale"])]


def test_without_relu_proves_acc_scale_only(tmp_path):
    chains = prove(tmp_path, relu=False)
    assert members(chains) == [(5, ["bias_add"]), (8, ["acc_scale"])]


def test_negative_gain_without_relu_is_proven(tmp_path):
    assert stage_names(prove(tmp_path, relu=False, gain="-0.25"))[-1] == "acc_scale"


# Variants that must leave NO readout stage; the untouched bias member may still be proven.
NO_READOUT = {
    "floor_not_roundeven": {"rend": "floor"},
    "clamp_low_127": {"lo": "-127.0"},
    "clamp_high_255": {"hi": "255.0"},
    "negative_gain_with_relu": {"gain": "-0.25"},
    "fastmath_nnan": {"fastmath": "nnan"},
    "gain_is_function_argument": {"gain_arg": True},
    "i16_output_full_range": {"out": "i16", "lo": "-32768.0", "hi": "32767.0"},
    "relu_zero_negative": {"zero": "-0.0"},
    "relu_zero_one": {"zero": "1.0"},
}


@pytest.mark.parametrize("variant", NO_READOUT.values(), ids=NO_READOUT.keys())
def test_inexact_readout_yields_no_readout_stage(tmp_path, variant):
    chains = prove(tmp_path, **variant)
    assert "acc_scale" not in stage_names(chains), build(**variant)
    assert "relu" not in stage_names(chains)
    assert members(chains) in ([], [(5, ["bias_add"])])


def test_mac_result_with_second_use_yields_no_chain(tmp_path):
    assert prove(tmp_path, second_use=True) == []


def test_unsupported_numerics_yield_no_readout_stage(tmp_path):
    chains = prove(tmp_path, numerics={"acc_scale_rounding": "half_away"})
    assert "acc_scale" not in stage_names(chains)


def test_missing_narrowing_numerics_yield_no_readout_stage(tmp_path):
    chains = prove(tmp_path, numerics={"acc_scale_rounding": "half_even"})
    assert "acc_scale" not in stage_names(chains)


def test_unsigned_extension_in_contraction_yields_no_chain(tmp_path):
    assert prove(tmp_path, ext="extui") == []


def test_bias_added_to_itself_yields_no_bias_stage(tmp_path):
    chains = prove(tmp_path, bias_operands="%x0, %x0")
    assert "bias_add" not in stage_names(chains)
    assert "acc_scale" not in stage_names(chains)


def test_other_strategy_needs_power_of_two_gains_and_a_bounded_accumulator(tmp_path):
    # f32(acc) * gain equals an exact rational readout only for an exact sitofp and a power-of-two gain.
    assert stage_names(prove(tmp_path, strategy="exact_rational_host", bias=False)) == []  # gain 1/6
    assert stage_names(prove(tmp_path, strategy="exact_rational_host", bias=False, gain="0.0625")) == [
        "relu",
        "acc_scale",
    ]
    # A runtime i32 bias leaves the accumulator unbounded: no readout stage, the bias stage stands.
    assert stage_names(prove(tmp_path, strategy="exact_rational_host", gain="0.0625")) == ["bias_add"]
    assert stage_names(prove(tmp_path, strategy=None, bias=False, gain="0.0625")) == ["relu", "acc_scale"]


def test_large_runtime_contraction_under_exact_strategy_is_not_observed(tmp_path):
    # |acc| <= 128 * 128 * K exceeds 2**24 once K > 1024 (the shape of an OS runtime-weight twin).
    text = build(bias=False, gain="0.0625").replace("16x16xi8>", "16x2048xi8>", 1)
    text = text.replace("%W0: tensor<16x16xi8>", "%W0: tensor<2048x16xi8>")
    text = text.replace("ins(%A, %W0 : tensor<16x16xi8>, tensor<16x16xi8>)", "ins(%A, %W0 : tensor<16x2048xi8>, tensor<2048x16xi8>)")
    path = tmp_path / "big.mlir"
    path.write_text(text)
    assert stage_names(prove_chains(path, numerics=NUMERICS, strategy="exact_rational_host")) == []
    assert stage_names(prove_chains(path, numerics=NUMERICS, strategy="f32_product_rne")) == ["relu", "acc_scale"]


@pytest.mark.parametrize("dataflow", ["OS", "unknown_dataflow"])
def test_output_stationary_or_unknown_dataflow_is_not_observed(tmp_path, dataflow):
    # The mesh partial-sum bound an output-stationary contraction needs is not proven at Phase 0.
    path = tmp_path / "capsule.interface.mlir"
    path.write_text(build())
    assert prove_chains(path, numerics=NUMERICS, strategy="f32_product_rne", dataflow=dataflow) == []
    assert stage_names(prove_chains(path, numerics=NUMERICS, strategy="f32_product_rne", dataflow="WS")) == [
        "bias_add",
        "relu",
        "acc_scale",
    ]


def two_gain_chain(
    tmp_path, *, bias=False, relu=True, last="0.10000000149011612", first=GAIN, bias_form="column", **kwargs
):
    """A chain whose readout scales columns 0..14 by ``first`` and column 15 by ``last``."""
    text = build(bias=bias, bias_form=bias_form, relu=relu, gain=first)
    one = ", ".join([first] * 16)
    two = ", ".join([first] * 15 + [last])
    assert text.count(one) == 1
    path = tmp_path / "two_gain.mlir"
    path.write_text(text.replace(one, two))
    return prove_chains(path, numerics=NUMERICS, strategy="f32_product_rne", **kwargs)


def test_rows_state_the_chain_epilogue_in_commit_order(tmp_path):
    from merlin.runtime.commandbuffer import EPILOGUE_STAGES

    (chain,) = prove(tmp_path)
    rows = stage_rows(chain)
    epilogue = ["bias_add", "acc_scale", "relu"]
    assert epilogue == [name for name in EPILOGUE_STAGES if name in epilogue]
    assert [row["operation"] for member in sorted(rows) for row in rows[member]] == ["bias_add", "relu", "acc_scale"]
    # One scalar gain: tensor granularity, no column partition source.
    for member_rows in rows.values():
        for row in member_rows:
            assert row["composed_observation"]["scale_granularity"] == "tensor"
            assert "logical_output_column_source" not in row


def test_two_gains_partition_into_scalar_column_components(tmp_path):
    from merlin.targetgen.output_scale_partition import interface_components

    (chain,) = two_gain_chain(tmp_path)
    (member_rows,) = stage_rows(chain).values()
    assert [row["operation"] for row in member_rows] == ["relu", "acc_scale"]
    for row in member_rows:
        assert row["composed_observation"]["scale_granularity"] == "channel"
        source = row["logical_output_column_source"]
        assert source["attributes"]["epilogue"] == ["acc_scale", "relu"]
        partition = interface_components(source)
        assert [(part["start"], part["stop"]) for part in partition["parts"]] == [(0, 15), (15, 16)]
        assert [part["dimensions"] for part in partition["parts"]] == [
            {"M": 16, "K": 16, "N": 15},
            {"M": 16, "K": 16, "N": 1},
        ]


def test_two_gains_with_a_column_bias_partition_with_its_bias_lane(tmp_path):
    from merlin.targetgen.output_scale_partition import interface_components

    (chain,) = two_gain_chain(tmp_path, bias=True)
    for member_rows in stage_rows(chain).values():
        for row in member_rows:
            source = row["logical_output_column_source"]
            assert source["bias"] == {"dtype": "i32", "shape": [16]}
            assert interface_components(source) is not None


@pytest.mark.parametrize("form", ["matrix", "rows_distinct"])
def test_bias_not_constant_along_rows_leaves_no_chain(tmp_path, form):
    """The fused bias stage adds one value per column; a runtime or row-varying [M, N] bias is not that."""
    assert prove(tmp_path, bias_form=form) == []


@pytest.mark.parametrize("form", ["matrix", "rows_distinct"])
def test_full_tensor_bias_ends_the_chain_unless_the_seed_is_declared(tmp_path, form):
    assert prove(tmp_path, bias_form=form, full_tensor_seed=False) == []


@pytest.mark.parametrize(
    ("form", "source", "seed_member"), [("matrix", "runtime_input", 5), ("rows_distinct", "constant", 6)]
)
def test_declared_full_tensor_bias_is_an_accumulator_seed_not_bias_add(tmp_path, form, source, seed_member):
    """Under the reviewed ``accumulator_seed_tensor`` the bias is the contraction's own int32 accumulator seed:
    a ``seed`` member with no epilogue stage, no ``bias_add`` row and no bias lane in the readout's column source.
    """
    chains = prove(tmp_path, bias_form=form, full_tensor_seed=True)
    (chain,) = chains
    assert chain["producer"] == 3
    (seed,) = [stage for stage in chain["stages"] if "seed" in stage]
    assert seed["member"] == seed_member
    assert seed["seed"] == {"dtype": "i32", "shape": [16, 16], "source": source}
    assert seed["stages"] == []
    assert "bias" not in seed
    # No stage dict anywhere is the per-column bias stage.
    assert all("bias" not in stage and "bias_add" not in stage["stages"] for stage in chain["stages"])
    assert "bias_add" not in stage_names(chains)
    rows = stage_rows(chain)
    assert seed_member not in rows  # a seed carries no epilogue stage row of its own
    assert "bias_add" not in [row["operation"] for member_rows in rows.values() for row in member_rows]
    # The readout is still observed (relu, acc_scale) behind the seed.
    assert members(chains) == [(seed_member, []), (seed_member + 3, ["relu", "acc_scale"])]


@pytest.mark.parametrize("form", ["column", "rows_equal"])
def test_the_seed_declaration_does_not_change_a_per_column_bias(tmp_path, form):
    assert prove(tmp_path, bias_form=form, full_tensor_seed=True) == prove(tmp_path, bias_form=form)
    assert "bias_add" in stage_names(prove(tmp_path, bias_form=form, full_tensor_seed=True))


@pytest.mark.parametrize("form", ["matrix", "rows_distinct"])
def test_seeded_readout_column_source_states_no_bias_lane(tmp_path, form):
    from merlin.targetgen.output_scale_partition import interface_components

    (chain,) = two_gain_chain(tmp_path, bias=True, bias_form=form, full_tensor_seed=True)
    (member_rows,) = stage_rows(chain).values()
    for row in member_rows:
        source = row["logical_output_column_source"]
        assert source["bias"] is None
        assert interface_components(source) is not None


@pytest.mark.parametrize("form", ["column", "rows_equal"])
def test_per_column_bias_is_proven_as_a_column_vector(tmp_path, form):
    chains = prove(tmp_path, bias_form=form)
    assert members(chains) == [(5 if form == "column" else 6, ["bias_add"]), (8 if form == "column" else 9, ["relu", "acc_scale"])]
    bias = next(stage for stage in chains[0]["stages"] if "bias" in stage)
    assert bias["bias"] == {"dtype": "i32", "shape": [16]}


def test_row_constant_bias_bounds_the_accumulator(tmp_path):
    (chain,) = prove(tmp_path, bias_form="rows_equal")
    assert chain["carrier"]["bound"] == 128 * 128 * 16
    # The bias bound (8) is added to the exact-strategy accumulator bound, so the readout still proves.
    assert stage_names(prove(tmp_path, bias_form="rows_equal", strategy="exact_rational_host", gain="0.0625")) == [
        "bias_add",
        "relu",
        "acc_scale",
    ]


def test_two_gains_with_a_negative_gain_after_relu_have_no_readout_stage(tmp_path):
    assert stage_names(two_gain_chain(tmp_path, last="-0.10000000149011612")) == []
    assert stage_names(two_gain_chain(tmp_path, relu=False, last="-0.10000000149011612")) == ["acc_scale"]


def test_column_partition_needs_a_stated_acc_scale_epilogue_and_disjoint_output():
    from merlin.targetgen.output_scale_partition import interface_components

    source = {
        "operation": "matmul",
        "lhs": {"dtype": "i8", "shape": [16, 16]},
        "rhs": {"dtype": "i8", "shape": [16, 16]},
        "disjoint": True,
        "result": {"dtype": "i8", "shape": [16, 16]},
        "bias": None,
    }
    gains = [0.5] * 15 + [0.25]
    assert interface_components({**source, "attributes": {"acc_scale_by_column": gains, "output_dtype": "i8"}}) is None
    attributes = {"acc_scale_by_column": gains, "output_dtype": "i8", "epilogue": ["acc_scale"]}
    assert interface_components({**source, "attributes": attributes}) is not None
    assert interface_components({**source, "disjoint": False, "attributes": attributes}) is None
