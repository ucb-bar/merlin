"""Soundness tests for the unit single-rounding source operand sum and its readout gain.

A composition program ``fptosi(roundeven(f32(a)*1.0 + f32(b)*1.0))`` read out by one uniform gain
is proven op by op, observed as one fused operand-sum command, and judged under the SAME
single-rounding contract its realization binds (``operand_sum_exhaustive_i8_v1``); the witness
carries the readout gain (``audit_i8_operand_sum(..., readout_scale=...)``).
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest
from merlin_experiments.phase0.source_fused_stages import operand_sum_rows, prove_operand_sums
from test_source_operand_sum import FACET, NUMERICS, _Evidence

SIXTH_TEXT = "0.16666667"


def f32(value) -> float:
    with np.errstate(over="ignore"):
        return float(np.float32(value))


_PROGRAM = """\
builtin.module {
  func.func @forward(%Left: tensor<MxNxi8>, %Right: tensor<MxNxi8>) -> (tensor<MxNxi8>RETTYPE) {
    %operand_sum_empty = tensor.empty() : tensor<MxNxRESULT>
    %operand_sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(IN0, IN1 : tensor<MxNxi8>, tensor<MxNxi8>) outs(%operand_sum_empty : tensor<MxNxRESULT>) {
    ^bb0(%x0: i8, %x1: i8, %dest: RESULT):
      %a = arith.sitofp %x0 : i8 to f32
      %b = arith.sitofp %x1 : i8 to f32
      %s0 = arith.constant S0 : f32
      %s1 = arith.constant S1 : f32
      %p0 = arith.mulf %a, %s0 FAST0 : f32
      %p1 = arith.mulf SECOND, %s1 : f32
      %p = arith.addf %p0, %p1 FAST1 : f32
ROUND      %v = arith.fptosi %r : f32 to RESULT
      linalg.yield %v : RESULT
    } -> tensor<MxNxRESULT>
    %readout_gains = arith.constant dense<[GAINS]> : tensor<Nxf32>
    %readout_empty = tensor.empty() : tensor<MxNxi8>
    %readout = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%operand_sum, %readout_gains : tensor<MxNxRESULT>, tensor<Nxf32>) outs(%readout_empty : tensor<MxNxi8>) {
    ^bb0(%x0: RESULT, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : RESULT to f32
RELU      %q = arith.mulf ACT, %x1 : f32
      %r = math.roundeven %q : f32
      %lo = arith.constant -128.0 : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    } -> tensor<MxNxi8>
    func.return %readout RETURN: tensor<MxNxi8>RETTYPE
  }
}
"""
_RELU = "      %zero = arith.constant 0.0 : f32\n      %act = arith.maximumf %f, %zero : f32\n"
_ROUND = "      %r = math.roundeven %p : f32\n"


def unit_sum_program(
    m=16,
    n=16,
    *,
    s0="1.0",
    s1="1.0",
    second="%b",
    ins=("%Left", "%Right"),
    gain=SIXTH_TEXT,
    gain2=None,
    relu=True,
    rounding=_ROUND,
    fast0="",
    fast1="",
    extra_return="",
    result="i32",
):
    gains = ", ".join([gain] * n if gain2 is None else [gain] * (n // 2) + [gain2] * (n - n // 2))
    text = _PROGRAM
    if rounding == "":  # the fptosi reads the sum itself
        text = text.replace("%v = arith.fptosi %r", "%v = arith.fptosi %p")
    text = (
        text.replace("ROUND", rounding)
        .replace("RELU", _RELU if relu else "")
        .replace("ACT", "%act" if relu else "%f")
        .replace("RETTYPE", f", tensor<{m}x{n}x{result}>" if extra_return else "")
        .replace("RETURN", f", {extra_return} " if extra_return else " ")
        .replace("RESULT", result)
        .replace("IN0", ins[0])
        .replace("IN1", ins[1])
        .replace("S0", s0)
        .replace("S1", s1)
        .replace("SECOND", second)
        .replace("FAST0", fast0)
        .replace("FAST1", fast1)
        .replace("GAINS", gains)
        .replace("MxN", f"{m}x{n}")
        .replace("Mx", f"{m}x")
        .replace("Nx", f"{n}x")
    )
    return text


def prove(tmp_path, text, numerics=NUMERICS, strategy="f32_product_rne"):
    path = tmp_path / "capsule.interface.mlir"
    path.write_text(text)
    return prove_operand_sums(path, numerics=numerics, strategy=strategy)


SHAPES = [(16, 16), (17, 16), (16, 33)]


@pytest.mark.parametrize("relu", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_unit_float_sum_with_a_uniform_gain_is_proven_as_single_rounding(tmp_path, shape, relu):
    [chain] = prove(tmp_path, unit_sum_program(*shape, relu=relu))
    assert chain["rounding"] == "single_rounding"
    assert chain["lhs_scale"] == 1.0 and chain["rhs_scale"] == 1.0
    assert f32(chain["readout_scale"]) == f32(1 / 6) and chain["relu"] is relu
    assert chain["shape"] == list(shape)
    # no requantization members: the sum is the one op, the readout follows the empty/gain constants
    assert chain["sum_members"] == [1] and chain["readout_member"] == 4
    assert "requants" not in chain and "reader" not in chain


@pytest.mark.parametrize("relu", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_rows_carry_a_v1_witness_with_the_readout_gain(tmp_path, shape, relu):
    [chain] = prove(tmp_path, unit_sum_program(*shape, relu=relu))
    rows = operand_sum_rows(chain, evidence=_Evidence([FACET]))
    assert [row["operation"] for row in rows] == ["residual_add", "acc_scale"] + (["relu"] if relu else [])
    assert rows[0]["ordinals"] == [1] and all(row["ordinals"] == [4] for row in rows[1:])
    for row in rows:
        screen = row["numeric_screen"]
        assert screen["form"] == "fused_operand_sum" and screen["command_index"] == 0
        assert screen["status"] == "within_bound" and screen["max_error_lsb"] == 0
        assert "numerical_contract" not in screen  # the v1 default: operand_sum_exhaustive_i8_v1
        parameters = screen["parameters"]
        assert parameters["readout_gain_f32"] == f32(1 / 6) and parameters["relu"] is relu
        assert parameters["lhs_scale_f32"] == 1.0 and parameters["rhs_scale_f32"] == 1.0
        assert parameters["readout_register_f32"] == f32(1 / 6) and parameters["bound_lsb"] == 0
        assert "operand_sum_rounding" not in parameters  # the legacy single-rounding parameters
    carrier = rows[0]
    assert carrier["command_opcode"] == "RESIDUAL_ADD"
    assert carrier["composed_observation"]["composed_with"] == []
    for row in rows[1:]:
        assert row["composed_observation"]["composed_with"] == ["residual_add"]


def test_rows_fail_closed_without_a_model_or_on_a_witness_mismatch(tmp_path, monkeypatch):
    [chain] = prove(tmp_path, unit_sum_program())
    assert operand_sum_rows(chain, evidence=_Evidence([FACET], form="standalone")) is None
    assert operand_sum_rows(chain, evidence=_Evidence([FACET, FACET])) is None
    assert operand_sum_rows(chain, evidence=_Evidence([{**FACET, "scale": {}}])) is None
    assert operand_sum_rows(chain, evidence=_Evidence([FACET], readout={"acc_scale_rounding": "half_even"})) is None
    for change in ({"lhs_scale": 0.5}, {"rhs_scale": 2.0}, {"rounding": "mystery"}, {"sum_members": [1, 2]}):
        assert operand_sum_rows({**chain, **change}, evidence=_Evidence([FACET])) is None
    # a gain the audit would refuse (outside f32 / 256) is no observation
    assert operand_sum_rows({**chain, "readout_scale": 1e38}, evidence=_Evidence([FACET])) is None

    # a witness describing other parameters than the source's constants is refused
    from merlin_experiments.phase0 import program_admission, source_fused_stages

    real = program_admission.single_rounding_readout_witness

    def tampered(**kwargs):
        witness = real(**kwargs)
        return {**witness, "parameters": {**witness["parameters"], "readout_gain_f32": f32(0.1)}}

    monkeypatch.setattr(program_admission, "single_rounding_readout_witness", tampered)
    assert source_fused_stages.operand_sum_rows(chain, evidence=_Evidence([FACET])) is None


REFUSALS = {
    "multiplier_1_0000001": dict(s0="1.0000001"),
    "second_multiplier_1_0000001": dict(s1="1.0000001"),
    "multiplier_half": dict(s0="0.5"),
    "second_multiplier_half": dict(s1="0.5"),
    "multiplier_two": dict(s0="2.0"),
    "multiplier_negative_unit": dict(s1="-1.0"),
    "same_operand_twice": dict(second="%a"),
    "same_input_twice": dict(ins=("%Left", "%Left")),
    "fast_math_on_a_product": dict(fast0="fastmath<fast>"),
    "fast_math_on_the_sum": dict(fast1="fastmath<reassoc>"),
    "no_roundeven": dict(rounding=""),
    "floor_instead_of_roundeven": dict(rounding="      %r = math.floor %p : f32\n"),
    "extra_use_of_the_sum": dict(extra_return="%operand_sum"),
    "two_readout_gains": dict(gain=SIXTH_TEXT, gain2="0.1"),
    "negative_gain": dict(gain="-0.16666667"),
    "negative_gain_without_relu": dict(gain="-0.16666667", relu=False),
    "zero_gain": dict(gain="0.0"),
    "narrow_sum_type": dict(result="i16"),
}


@pytest.mark.parametrize("variant", REFUSALS.values(), ids=REFUSALS.keys())
def test_wrong_unit_sum_variants_are_not_proven(tmp_path, variant):
    assert prove(tmp_path, unit_sum_program(**variant)) == []


def test_the_baseline_program_of_the_refusals_is_proven(tmp_path):
    assert len(prove(tmp_path, unit_sum_program())) == 1
    assert len(prove(tmp_path, unit_sum_program(gain="0.1"))) == 1


def test_unsupported_numerics_or_strategy_prove_nothing(tmp_path):
    assert prove(tmp_path, unit_sum_program(), numerics={"acc_scale_rounding": "half_away"}) == []
    assert prove(tmp_path, unit_sum_program(), strategy="exact_rational_host") == []


# ---- independent evaluation of the golden: sat_i8(RNE(f32(relu(a + b)) * f32(g))) -----------------------------

DOMAIN = np.arange(-128, 128)
SUMS = DOMAIN[:, None] + DOMAIN[None, :]
GAINS = [f32(1 / 6), f32(0.1), 0.5, 1.0]


def golden_numpy(g, relu):
    values = np.maximum(SUMS, 0) if relu else SUMS
    return np.clip(np.rint(values.astype(np.float32) * np.float32(g)), -128, 127).astype(np.int64)


def round_to_f32(x: Fraction) -> Fraction:
    """Exact round-to-nearest-even of a rational to a binary32 value (normal range)."""
    if x == 0:
        return x
    sign, magnitude = (1, x) if x > 0 else (-1, -x)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    quantum = Fraction(2) ** (exponent - 23)
    return sign * round(magnitude / quantum) * quantum  # round() on a Fraction is half-even


def golden_fractions(g, relu, *, product_in_f32=True):
    gain = Fraction(g)  # the exact binary value of the f32 gain
    table = {}
    for total in range(-256, 255):
        value = Fraction(max(total, 0) if relu else total)
        product = value * gain
        if product_in_f32:
            product = round_to_f32(product)
        table[total] = min(127, max(-128, round(product)))
    return np.vectorize(table.__getitem__)(SUMS).astype(np.int64)


@pytest.mark.parametrize("relu", [True, False])
@pytest.mark.parametrize("g", GAINS)
def test_golden_numpy_fractions_and_the_composition_readout_agree(g, relu):
    expected = golden_numpy(g, relu)
    assert (golden_fractions(g, relu) == expected).all()
    exact = golden_fractions(g, relu, product_in_f32=False)
    # An exact-rational product differs ONLY where the f32 product rounds onto/across an integer tie.
    gain = Fraction(g)
    for a, b in np.argwhere(exact != expected):
        value = max(int(SUMS[a, b]), 0) if relu else int(SUMS[a, b])
        product = value * gain
        tie = Fraction(round(product * 2), 2) if round(product * 2) % 2 else None
        assert tie is not None and abs(product - tie) < Fraction(1, 2**22) * max(1, abs(product))
    exact_readout = pytest.importorskip("merlin.runtime.exact_readout")
    composition = exact_readout.f32_readout(SUMS.astype(np.int32), np.float32(g), "i8", relu=relu)
    assert (np.asarray(composition).astype(np.int64) == expected).all()


@pytest.mark.parametrize("relu", [True, False])
@pytest.mark.parametrize("g", GAINS)
def test_the_witness_screens_exactly_the_golden(g, relu):
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    result = audit_i8_operand_sum(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=relu, facet=FACET, readout_scale=g)
    assert result["status"] == "within_bound" and result["max_error_lsb"] == 0 and result["n_over_bound"] == 0
    assert result["pairs_checked"] == 65536 and "numerical_contract" not in result
    assert result["parameters"]["readout_gain_f32"] == g and result["parameters"]["readout_register_f32"] == g
    # the digest binds the parameters, the reference and the device outputs: recompute it from the golden
    golden = golden_numpy(g, relu).astype(np.int8).tobytes()
    digest = hashlib.sha256(json.dumps(result["parameters"], sort_keys=True, separators=(",", ":")).encode())
    digest.update(golden)
    digest.update(golden)
    assert result["witness_sha256"] == digest.hexdigest()


def test_pinned_extremes_and_ties():
    sixth = golden_numpy(f32(1 / 6), False)
    sixth_relu = golden_numpy(f32(1 / 6), True)
    index = lambda value: value + 128  # noqa: E731
    assert sixth[index(-128), index(-128)] == -43 and sixth_relu[index(-128), index(-128)] == 0
    assert sixth[index(127), index(127)] == 42 and sixth_relu[index(127), index(127)] == 42
    assert sixth[index(127), index(-128)] == 0 and sixth_relu[index(127), index(-128)] == 0
    half = golden_numpy(0.5, False)
    # round-half-even on exact ties: 1/2 -> 0, 3/2 -> 2, -1/2 -> 0, -3/2 -> -2
    assert half[index(1), index(0)] == 0 and half[index(2), index(1)] == 2
    assert half[index(-1), index(0)] == 0 and half[index(-2), index(-1)] == -2
    assert golden_numpy(0.5, True)[index(-2), index(-1)] == 0


def test_the_witness_pins_the_same_extremes():
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    for relu, low in ((False, -43), (True, 0)):
        result = audit_i8_operand_sum(
            lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=relu, facet=FACET, readout_scale=f32(1 / 6)
        )
        assert result["status"] == "within_bound"
        # the screened corner (-128, -128) is the pinned golden value for this gain
        assert golden_numpy(f32(1 / 6), relu)[0, 0] == low


# ---- a non-unit scale pair: the documented reference and device models, evaluated independently ----------------


def independent_models(lhs, rhs, gain, relu):
    """The documented reference/device models of ``audit_i8_operand_sum(..., readout_scale=g)``, per pair."""
    lhs, rhs, gain = (np.float32(f32(x)) for x in (lhs, rhs, gain))
    factor = np.float32(max(np.float32(1), lhs, rhs))
    load_l, load_r = np.float32(lhs / factor), np.float32(rhs / factor)
    register = np.float32(factor * gain)
    reference = np.zeros((256, 256), dtype=np.int64)
    device = np.zeros((256, 256), dtype=np.int64)
    for i, a in enumerate(range(-128, 128)):
        for j, b in enumerate(range(-128, 128)):
            total = np.rint(np.float32(np.float32(a) * lhs) + np.float32(np.float32(b) * rhs))
            if relu:
                total = max(total, np.float32(0))
            reference[i, j] = min(127, max(-128, int(np.rint(np.float32(total) * gain))))
            la = min(127, max(-128, int(np.rint(np.float32(np.float32(a) * load_l)))))
            lb = min(127, max(-128, int(np.rint(np.float32(np.float32(b) * load_r)))))
            summed = max(la + lb, 0) if relu else la + lb
            device[i, j] = min(127, max(-128, int(np.rint(np.float32(summed) * register))))
    return reference, device


@pytest.mark.parametrize("relu", [True, False])
@pytest.mark.parametrize("scales", [(0.5, 0.25), (1 / 6, 0.1), (1.0, 1.0)])
def test_non_unit_scales_match_the_documented_models(scales, relu):
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    gain = 1 / 6
    reference, device = independent_models(*scales, gain, relu)
    error = np.abs(reference - device)
    for bound in (0, int(error.max()), 255):
        result = audit_i8_operand_sum(
            lhs_scale=scales[0], rhs_scale=scales[1], bound_lsb=bound, relu=relu, facet=FACET, readout_scale=gain
        )
        assert result["max_error_lsb"] == int(error.max())
        assert result["n_over_bound"] == int((error > bound).sum())
        assert result["status"] == ("within_bound" if error.max() <= bound else "exceeds_bound")
        digest = hashlib.sha256(json.dumps(result["parameters"], sort_keys=True, separators=(",", ":")).encode())
        digest.update(reference.astype(np.int8).tobytes())
        digest.update(device.astype(np.int8).tobytes())
        assert result["witness_sha256"] == digest.hexdigest()


# ---- the interface path ---------------------------------------------------------------------------------------

_INTERFACE = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<16x16xi8>
  %X1 = merlin_iface.tensor {name = "X1", role = "input"} : tensor<16x16xi8>
  %Y0 = merlin_iface.residual_add %X0, %X1 {name = "Y0", epilogue = EPILOGUE, lhs_scale = 1.0 : f32, rhs_scale = 0.5 : f32, bound_lsb = 0 : i64EXTRA, output_dtype = "i8"} : (tensor<16x16xi8>, tensor<16x16xi8>) -> tensor<16x16xi8>
}
"""


def interface_screen(epilogue, extra=""):
    from merlin_experiments.phase0 import program_admission

    text = _INTERFACE.replace("EPILOGUE", epilogue).replace("EXTRA", extra)
    [screen] = program_admission._operand_sum_numeric_screens_text(text, _Evidence([FACET]))
    return screen


def test_interface_single_rounding_acc_scale_is_unknown_not_screened_without_its_gain():
    """An interface RESIDUAL_ADD cannot state a readout gain: no screen may ignore one (formerly it did)."""
    screen = interface_screen('["acc_scale", "relu"]', ", acc_scale = 0.16666667 : f32")
    assert screen["status"] == "unknown" and "separate readout-scale witness" in screen["reason"]
    assert "parameters" not in screen


def test_interface_acc_scale_stage_without_a_numeric_scale_is_unknown():
    assert interface_screen('["acc_scale"]')["status"] == "unknown"


def test_interface_screen_without_acc_scale_is_the_legacy_screen():
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    screen = interface_screen('["relu"]')
    legacy = audit_i8_operand_sum(lhs_scale=1.0, rhs_scale=0.5, bound_lsb=0, relu=True, facet=FACET)
    assert screen == {"command_index": 0, "form": "fused_operand_sum", **legacy}
    assert "readout_gain_f32" not in screen["parameters"]
    plain = interface_screen("[]")
    assert plain["form"] == "standalone_operand_sum" and "readout_gain_f32" not in plain["parameters"]


def test_interface_per_operand_branch_is_unchanged():
    screen = interface_screen('["acc_scale"]', ', operand_sum_rounding = "per_operand", acc_scale = 0.5 : f32')
    assert screen["status"] == "unknown" and "separate readout-scale witness" in screen["reason"]


# ---- the witness' own gate ------------------------------------------------------------------------------------


def test_single_rounding_witness_gate_mirrors_the_per_operand_one():
    from merlin_experiments.phase0.program_admission import single_rounding_readout_witness

    def witness(evidence, **kw):
        args = dict(lhs_scale=1.0, rhs_scale=1.0, readout_scale=f32(1 / 6), relu=True)
        args.update(kw)
        return single_rounding_readout_witness(evidence=evidence, **args)

    good = witness(_Evidence([FACET]))
    assert good["form"] == "fused_operand_sum" and good["status"] == "within_bound"
    assert witness(_Evidence([FACET], form="standalone")) is None
    assert witness(_Evidence([FACET, FACET]))["status"] == "unknown"
    assert witness(_Evidence([]))["status"] == "unknown"
    assert witness(_Evidence([FACET], readout={"acc_scale_rounding": "half_even"}))["status"] == "unknown"
    for bad in (0.0, -1.0, float("nan"), float("inf"), 1e38, "x", True):
        assert witness(_Evidence([FACET]), readout_scale=bad)["status"] == "unknown", bad


def test_the_f32_product_is_not_the_exact_rational_product():
    """The golden IS the f32 composition: an exact-rational readout differs at near-ties (so it is not used)."""
    for g in (f32(1 / 6), f32(0.1)):
        assert (golden_fractions(g, False, product_in_f32=False) != golden_numpy(g, False)).any()
    for g in (0.5, 1.0):
        assert (golden_fractions(g, False, product_in_f32=False) == golden_numpy(g, False)).all()


# ---- the validator accepts a gain-carrying single-rounding witness only under the reviewed key -----------


def _single_spec(screens):
    row = dict(
        id="elementwise_map_after_operand_sum",
        status="reviewed",
        numerical_contract="operand_sum_exhaustive_i8_v1",
        families=["elementwise_map"],
        placement="fused_accelerator",
        signature=dict(dtypes=["int8"], composed_with=["residual_add"]),
    )
    if screens is not None:
        row["readout_gain_screens"] = screens
    return dict(status="reviewed", operations=[row])


def _single_signature(readout_scale):
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    extra = {} if readout_scale is None else {"readout_scale": readout_scale}
    witness = audit_i8_operand_sum(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=True, facet=FACET, **extra)
    return dict(
        family="elementwise_map",
        operand_dtype="i8",
        composed_with=["residual_add"],
        numeric_screen={"command_index": 0, "form": "fused_operand_sum", **witness},
    )


def _single_status(spec, signature):
    from merlin.targetgen.software_spec import admit_operation

    return admit_operation(spec, "acc_scale", signature, "fused_accelerator")["status"]


def test_validator_accepts_a_gain_witness_only_on_a_row_declaring_the_tensor_gain_screen():
    gained = _single_signature(f32(1 / 6))
    assert _single_status(_single_spec(None), gained) == "unknown"
    assert _single_status(_single_spec(["column_component"]), gained) == "unknown"
    assert _single_status(_single_spec(["tensor"]), gained) == "admitted"
    # The legacy (gain-free) witness is judged exactly as before, with or without the key.
    legacy = _single_signature(None)
    assert _single_status(_single_spec(None), legacy) == "admitted"
    assert _single_status(_single_spec(["tensor"]), legacy) == "admitted"
