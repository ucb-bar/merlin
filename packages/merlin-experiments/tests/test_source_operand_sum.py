"""Soundness tests for the source-side fused operand-sum proof (``source_fused_stages.operand_sum_chain``)."""

from __future__ import annotations

import pytest

from merlin_experiments.phase0.source_fused_stages import prove_operand_sums

NUMERICS = {"acc_scale_rounding": "half_even", "narrowing": "saturate_to_declared_dtype"}
SIXTH, TENTH = "0.1666666716337204", "0.10000000149011612"

_REQUANT = """\
    %NAME_gains = arith.constant dense<[GAINS]> : tensor<16xf32>
    %NAME_empty = tensor.empty() : tensor<16x16xi8>
    %NAME = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%SRC, %NAME_gains : tensor<16x16xi8>, tensor<16xf32>) outs(%NAME_empty : tensor<16x16xi8>) {
    ^bb0(%x0: i8, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : i8 to f32
RELU      %p = arith.mulf %SCALED, %x1 : f32
      %r = math.REND %p : f32
      %lo = arith.constant LO : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    } -> tensor<16x16xi8>
"""
_INT_SUM = """\
    %operand_sum_empty = tensor.empty() : tensor<16x16xi32>
    %operand_sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%IN0, %IN1 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%operand_sum_empty : tensor<16x16xi32>) {
    ^bb0(%x0: i8, %x1: i8, %dest: i32):
      %a = arith.EXT %x0 : i8 to i32
      %b = arith.EXT %x1 : i8 to i32
      %v = arith.addi ADD_OPERANDS : i32
      linalg.yield %v : i32
    } -> tensor<16x16xi32>
"""
_FLOAT_SUM = """\
    %operand_sum_empty = tensor.empty() : tensor<16x16xi32>
    %operand_sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%IN0, %IN1 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%operand_sum_empty : tensor<16x16xi32>) {
    ^bb0(%x0: i8, %x1: i8, %dest: i32):
      %a = arith.sitofp %x0 : i8 to f32
      %b = arith.sitofp %x1 : i8 to f32
      %s0 = arith.constant S0 : f32
      %s1 = arith.constant S1 : f32
      %p0 = arith.mulf %a, %s0 : f32
      %p1 = arith.mulf %SECOND, %s1 : f32
      %p = arith.addf %p0, %p1 : f32
      %r = math.roundeven %p : f32
      %v = arith.fptosi %r : f32 to i32
      linalg.yield %v : i32
    } -> tensor<16x16xi32>
"""
_READOUT = """\
    %readout_gains = arith.constant dense<[GAINS]> : tensor<16xf32>
    %readout_empty = tensor.empty() : tensor<16x16xi8>
    %readout = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%operand_sum, %readout_gains : tensor<16x16xi32>, tensor<16xf32>) outs(%readout_empty : tensor<16x16xi8>) {
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
    } -> tensor<16x16xi8>
"""
_RELU = "      %zero = arith.constant 0.0 : f32\n      %act = arith.maximumf %f, %zero : f32\n"


def _gains(value, second=None):
    values = [value] * 16 if second is None else [value] * 8 + [second] * 8
    return ", ".join(values)


def requant(name, src, gain=SIXTH, second=None, rend="roundeven", lo="-128.0", relu=False):
    text = _REQUANT.replace("NAME", name).replace("%SRC", src).replace("GAINS", _gains(gain, second))
    text = text.replace("RELU", _RELU if relu else "").replace("%SCALED", "%act" if relu else "%f")
    return text.replace("REND", rend).replace("LO", lo)


def per_operand(*, left=None, right=None, ext="extsi", add="%a, %b", ins=("%left", "%right"), readout_gain=SIXTH,
                readout_second=None, extra_return=""):
    body = (left if left is not None else requant("left", "%Left"))
    body += (right if right is not None else requant("right", "%Right", gain=TENTH))
    body += _INT_SUM.replace("EXT", ext).replace("ADD_OPERANDS", add).replace("%IN0", ins[0]).replace("%IN1", ins[1])
    body += _READOUT.replace("GAINS", _gains(readout_gain, readout_second))
    return _module(body, extra_return)


def unit_sum(*, s0="1.0", s1="1.0", second="%b", ins=("%Left", "%Right"), readout_gain=SIXTH, readout_second=None):
    body = _FLOAT_SUM.replace("S0", s0).replace("S1", s1).replace("%SECOND", second)
    body = body.replace("%IN0", ins[0]).replace("%IN1", ins[1])
    body += _READOUT.replace("GAINS", _gains(readout_gain, readout_second))
    return _module(body)


def _module(body, extra_return=""):
    ret_type = "tensor<16x16xi8>" + (", tensor<16x16xi8>" if extra_return else "")
    ret = "%readout" + (", " + extra_return if extra_return else "")
    return (
        "builtin.module {\n  func.func @forward(%Left: tensor<16x16xi8>, %Right: tensor<16x16xi8>) -> ("
        + ret_type + ") {\n" + body + f"    func.return {ret} : {ret_type}\n  }}\n}}\n"
    )


def prove(tmp_path, text, numerics=NUMERICS, strategy="f32_product_rne"):
    path = tmp_path / "capsule.interface.mlir"
    path.write_text(text)
    return prove_operand_sums(path, numerics=numerics, strategy=strategy)


def test_per_operand_chain_is_proven_with_its_exact_constants(tmp_path):
    [chain] = prove(tmp_path, per_operand())
    assert chain["rounding"] == "per_operand"
    assert chain["sum_members"] == [2, 5, 7] and chain["readout_member"] == 10
    assert chain["lhs_scale"] == pytest.approx(float(SIXTH), abs=0) and chain["rhs_scale"] == float(TENTH)
    assert chain["readout_scale"] == float(SIXTH) and chain["relu"] is True and chain["shape"] == [16, 16]


def test_float_single_rounding_sum_is_not_observed_even_at_unit_multipliers(tmp_path):
    # Exact at unit multipliers, but its realization binds the single-rounding contract, whose
    # witness (audit_i8_operand_sum) has no readout-gain parameter: the gain would go unscreened.
    assert prove(tmp_path, unit_sum()) == []


WRONG_PER_OPERAND = {
    "requant_floor": {"left": requant("left", "%Left", rend="floor")},
    "requant_clamp_127": {"left": requant("left", "%Left", lo="-127.0")},
    "requant_with_relu": {"left": requant("left", "%Left", relu=True)},
    "requant_two_gains": {"left": requant("left", "%Left", second=TENTH)},
    "requant_negative_gain": {"right": requant("right", "%Right", gain="-0.1")},
    "requant_zero_gain": {"right": requant("right", "%Right", gain="0.0")},
    "unsigned_extension": {"ext": "extui"},
    "sum_of_one_operand_twice": {"add": "%a, %a"},
    "same_requant_twice": {"ins": ("%left", "%left")},
    "both_requants_of_one_input": {"right": requant("right", "%Left", gain=TENTH)},
    "readout_two_gains": {"readout_gain": SIXTH, "readout_second": TENTH},
    "requant_result_escapes": {"extra_return": "%left"},
}


@pytest.mark.parametrize("variant", WRONG_PER_OPERAND.values(), ids=WRONG_PER_OPERAND.keys())
def test_wrong_per_operand_variants_are_not_proven(tmp_path, variant):
    assert prove(tmp_path, per_operand(**variant)) == []


def test_unsupported_numerics_or_strategy_prove_nothing(tmp_path):
    assert prove(tmp_path, per_operand(), numerics={"acc_scale_rounding": "half_away"}) == []
    # f32(x) * 1/6 is not an exact rational readout: another strategy needs power-of-two gains.
    assert prove(tmp_path, per_operand(), strategy="exact_rational_host") == []


class _Evidence:
    def __init__(self, facets, readout=NUMERICS, form="fused_operand_sum"):
        self.software_spec = {
            "operations": [{"id": "row", "derived_from_facts": {"form": form}}],
            "numerical_semantics": {"readout": dict(readout)},
        }
        self.readout_facets = facets


FACET = {
    "operand_sum": {"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True,
                    "scale_dtype": "f32"},
    "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
    "scale": {"granularities": ["tensor"], "dtype": "f32"},
    "rounding": "half_even",
    "clamp": [-128, 127],
    "accumulator_dtype": "i32",
}


def _witness(evidence, **kw):
    from merlin_experiments.phase0.program_admission import per_operand_readout_witness

    args = dict(lhs_scale=float(SIXTH), rhs_scale=float(TENTH), readout_scale=float(SIXTH), relu=True)
    args.update(kw)
    return per_operand_readout_witness(evidence=evidence, **args)


def test_readout_scale_witness_carries_the_source_gain():
    witness = _witness(_Evidence([FACET]))
    assert witness["form"] == "fused_operand_sum" and witness["status"] == "within_bound"
    assert witness["numerical_contract"] == "operand_sum_per_operand_exhaustive_i8_v1"
    assert witness["parameters"]["source"]["readout_scale"] == float(SIXTH)
    assert witness["parameters"]["operand_sum_rounding"] == "per_operand" and witness["parameters"]["bound_lsb"] == 0


def test_readout_scale_witness_fails_closed():
    assert _witness(_Evidence([FACET], form="standalone")) is None
    assert _witness(_Evidence([FACET, FACET]))["status"] == "unknown"
    assert _witness(_Evidence([]))["status"] == "unknown"
    assert _witness(_Evidence([FACET], readout={"acc_scale_rounding": "half_even"}))["status"] == "unknown"
    assert _witness(_Evidence([FACET]), lhs_scale=1.5)["status"] == "unknown"  # operand scale outside (0, 1]
    assert _witness(_Evidence([{**FACET, "clamp": [-127, 127]}]))["status"] == "unknown"


def test_audit_reference_is_the_source_program_arithmetic():
    """The witness's reference IS the proven source body: f32 products, RNE, i8 saturation, exact i32 sum."""
    import numpy as np

    from merlin.targetgen.operand_sum_numeric import _per_operand_reference

    lhs, rhs, gain = np.float32(SIXTH), np.float32(TENTH), np.float32(SIXTH)
    for a in range(-128, 128, 7):
        for b in range(-128, 128, 5):
            left = np.clip(np.rint(np.float32(a) * lhs), -128, 127)
            right = np.clip(np.rint(np.float32(b) * rhs), -128, 127)
            total = np.maximum(np.float32(int(left) + int(right)), np.float32(0.0))
            source = int(np.clip(np.rint(total * gain), -128, 127))
            assert source == _per_operand_reference(a, b, float(lhs), float(rhs), float(gain), True)


def test_fused_rows_are_the_interface_observation_with_the_source_witness(tmp_path):
    from merlin_experiments.phase0.source_fused_stages import operand_sum_rows

    [chain] = prove(tmp_path, per_operand())
    rows = operand_sum_rows(chain, evidence=_Evidence([FACET]))
    assert [row["operation"] for row in rows] == ["residual_add", "acc_scale", "relu"]
    carrier, *stages = rows
    assert carrier["ordinals"] == [2, 5, 7] and carrier["command_opcode"] == "RESIDUAL_ADD"
    assert carrier["composed_observation"]["composed_with"] == []
    for row in stages:
        assert row["ordinals"] == [10]
        assert row["composed_observation"] == {
            "epilogues": [row["operation"]],
            "composed_with": ["residual_add"],
            "scale_granularity": "tensor",
        }
        assert (row["tensor_operands"], row["expression"], row["broadcasting"]) == (2, "affine", "none")
    for row in rows:
        witness = row["numeric_screen"]
        assert witness["form"] == "fused_operand_sum" and witness["status"] == "within_bound"
        assert witness["parameters"]["source"] == {
            "lhs_scale": float(SIXTH), "rhs_scale": float(TENTH), "readout_scale": float(SIXTH), "relu": True
        }


def test_no_fused_operand_sum_row_yields_no_rows(tmp_path):
    from merlin_experiments.phase0.source_fused_stages import operand_sum_rows

    [chain] = prove(tmp_path, per_operand())
    assert operand_sum_rows(chain, evidence=_Evidence([FACET], form="standalone")) is None
