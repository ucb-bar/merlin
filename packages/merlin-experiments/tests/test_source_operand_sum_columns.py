"""A fused operand sum whose readout gain differs by output column: column components, fail closed.

The requant gains stay uniform; only the readout's per-column gain may vary. It becomes rows only
behind the reviewed ``elementwise_map_after_per_operand_sum`` spec row declaring
``readout_gain_screens: column_component``, one exhaustive witness per run of equal gain.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from fractions import Fraction

import numpy as np
import pytest

from merlin_experiments.phase0.source_fused_stages import operand_sum_rows, prove_operand_sums

NUMERICS = {"acc_scale_rounding": "half_even", "narrowing": "saturate_to_declared_dtype"}
ROW = "elementwise_map_after_per_operand_sum"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


SIXTH, TENTH = f32(1 / 6), f32(0.1)

_MODULE = """\
builtin.module {
  func.func @forward(%Left: tensor<MxNxi8>, %Right: tensor<MxNxi8>) -> (tensor<MxNxi8>) {
    %left_gains = arith.constant dense<[LGAINS]> : tensor<Nxf32>
    %left_empty = tensor.empty() : tensor<MxNxi8>
    %left = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%Left, %left_gains : tensor<MxNxi8>, tensor<Nxf32>) outs(%left_empty : tensor<MxNxi8>) {
    ^bb0(%x0: i8, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : i8 to f32
      %p = arith.mulf %f, %x1 : f32
      %r = math.roundeven %p : f32
      %lo = arith.constant -128.0 : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    } -> tensor<MxNxi8>
    %right_gains = arith.constant dense<[RGAINS]> : tensor<Nxf32>
    %right_empty = tensor.empty() : tensor<MxNxi8>
    %right = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%Right, %right_gains : tensor<MxNxi8>, tensor<Nxf32>) outs(%right_empty : tensor<MxNxi8>) {
    ^bb0(%x0: i8, %x1: f32, %dest: i8):
      %f = arith.sitofp %x0 : i8 to f32
      %p = arith.mulf %f, %x1 : f32
      %r = math.roundeven %p : f32
      %lo = arith.constant -128.0 : f32
      %hi = arith.constant 127.0 : f32
      %l = arith.maximumf %r, %lo : f32
      %h = arith.minimumf %l, %hi : f32
      %v = arith.fptosi %h : f32 to i8
      linalg.yield %v : i8
    } -> tensor<MxNxi8>
    %operand_sum_empty = tensor.empty() : tensor<MxNxi32>
    %operand_sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%left, %right : tensor<MxNxi8>, tensor<MxNxi8>) outs(%operand_sum_empty : tensor<MxNxi32>) {
    ^bb0(%x0: i8, %x1: i8, %dest: i32):
      %a = arith.extsi %x0 : i8 to i32
      %b = arith.extsi %x1 : i8 to i32
      %v = arith.addi %a, %b : i32
      linalg.yield %v : i32
    } -> tensor<MxNxi32>
    %readout_gains = arith.constant dense<[OGAINS]> : tensor<Nxf32>
    %readout_empty = tensor.empty() : tensor<MxNxi8>
    %readout = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%operand_sum, %readout_gains : tensor<MxNxi32>, tensor<Nxf32>) outs(%readout_empty : tensor<MxNxi8>) {
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
    } -> tensor<MxNxi8>
    func.return %readout : tensor<MxNxi8>
  }
}
"""


def module(readout_gains, *, m=16):
    n = len(readout_gains)
    text = _MODULE.replace("LGAINS", ", ".join([repr(SIXTH)] * n)).replace("RGAINS", ", ".join([repr(TENTH)] * n))
    text = text.replace("OGAINS", ", ".join(repr(g) for g in readout_gains))
    return text.replace("MxN", f"{m}x{n}").replace("tensor<N", f"tensor<{n}")


def tail_gains(n=17):
    return [SIXTH] * (n - 1) + [TENTH]


FACET = {
    "operand_sum": {"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True,
                    "scale_dtype": "f32"},
    "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
    "scale": {"granularities": ["tensor"], "dtype": "f32"},
    "rounding": "half_even",
    "clamp": [-128, 127],
    "accumulator_dtype": "i32",
}


class Evidence:
    def __init__(self, *, screens=("tensor", "column_component"), status="reviewed", row_id=ROW):
        row = {"id": row_id, "derived_from_facts": {"form": "fused_operand_sum"}, "status": status}
        if screens is not None:
            row["readout_gain_screens"] = list(screens)
        self.software_spec = {"operations": [row], "numerical_semantics": {"readout": dict(NUMERICS)}}
        self.readout_facets = [FACET]


def chain_of(tmp_path, gains, *, m=16):
    path = tmp_path / "capsule.interface.mlir"
    path.write_text(module(gains, m=m))
    [chain] = prove_operand_sums(path, numerics=NUMERICS, strategy="f32_product_rne")
    return chain


def test_column_gain_chain_is_split_into_proven_components(tmp_path):
    chain = chain_of(tmp_path, tail_gains())
    assert chain["readout_scale"] is None and chain["readout_gains"] == tail_gains() and chain["shape"] == [16, 17]
    assert chain["lhs_scale"] == SIXTH and chain["rhs_scale"] == TENTH and chain["relu"] is True
    rows = operand_sum_rows(chain, evidence=Evidence())
    assert len(rows) == 6  # two components x (residual_add, acc_scale, relu)
    components = [row["output_column_component"] for row in rows]
    assert components == [
        {"start": 0, "stop": 16, "columns": 17, "readout_scale": SIXTH},
        *[{"start": 0, "stop": 16, "columns": 17, "readout_scale": SIXTH}] * 2,
        {"start": 16, "stop": 17, "columns": 17, "readout_scale": TENTH},
        *[{"start": 16, "stop": 17, "columns": 17, "readout_scale": TENTH}] * 2,
    ]
    for row in rows:
        witness = row["numeric_screen"]
        assert witness["status"] == "within_bound" and witness["max_error_lsb"] == 0
        assert witness["parameters"]["source"] == {
            "lhs_scale": SIXTH, "rhs_scale": TENTH, "readout_scale": row["output_column_component"]["readout_scale"],
            "relu": True,
        }
        assert row["ordinals"] == ([chain["readout_member"]] if row["operation"] != "residual_add" else chain["sum_members"])
        assert (row["tensor_operands"], row["expression"], row["broadcasting"]) == (2, "affine", "none")
    # The components partition [0, N) exactly once.
    spans = sorted({(c["start"], c["stop"]) for c in components})
    assert spans[0][0] == 0 and spans[-1][1] == 17 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_interleaved_gains_give_one_component_per_maximal_run(tmp_path):
    gains = [SIXTH, SIXTH, TENTH, SIXTH, TENTH, TENTH]
    rows = operand_sum_rows(chain_of(tmp_path, gains), evidence=Evidence())
    spans = []
    for row in rows:
        c = row["output_column_component"]
        if row["operation"] == "residual_add":
            spans.append((c["start"], c["stop"], c["readout_scale"]))
    assert spans == [(0, 2, SIXTH), (2, 3, TENTH), (3, 4, SIXTH), (4, 6, TENTH)]


def test_uniform_gain_rows_carry_no_component(tmp_path):
    chain = chain_of(tmp_path, [SIXTH] * 17)
    assert chain["readout_scale"] == SIXTH
    for evidence in (Evidence(), Evidence(screens=None)):
        rows = operand_sum_rows(chain, evidence=evidence)
        assert len(rows) == 3 and all("output_column_component" not in row for row in rows)
        assert all(row["numeric_screen"]["parameters"]["source"]["readout_scale"] == SIXTH for row in rows)


@pytest.mark.parametrize(
    "evidence",
    [
        Evidence(screens=None),
        Evidence(screens=["tensor"]),
        Evidence(status="unreviewed"),
        Evidence(row_id="another_row"),
    ],
    ids=["no_key", "tensor_only", "unreviewed", "row_absent"],
)
def test_column_gain_without_the_reviewed_declaration_yields_no_rows(tmp_path, evidence):
    assert operand_sum_rows(chain_of(tmp_path, tail_gains()), evidence=evidence) is None


def test_non_uniform_readout_needs_a_witness_model_and_a_within_bound_screen(tmp_path):
    chain = chain_of(tmp_path, tail_gains())
    no_facets = Evidence()
    no_facets.readout_facets = []
    assert operand_sum_rows(chain, evidence=no_facets) is None
    standalone = Evidence()
    standalone.software_spec["operations"][0]["derived_from_facts"]["form"] = "standalone"
    assert operand_sum_rows(chain, evidence=standalone) is None
    # A non-finite-product gain (the products would overflow f32) gives no within-bound witness.
    huge = copy.deepcopy(chain)
    huge["readout_gains"] = [SIXTH] * 16 + [3.0e38]
    assert operand_sum_rows(huge, evidence=Evidence()) is None


def test_chain_gains_must_match_the_shape(tmp_path):
    chain = chain_of(tmp_path, tail_gains())
    chain["readout_gains"] = chain["readout_gains"][:-1]
    assert operand_sum_rows(chain, evidence=Evidence()) is None


# ---- the validator binds a component row to its own witness -------------------------------------


def _admission_spec(screens):
    row = dict(
        id=ROW,
        status="reviewed",
        numerical_contract="operand_sum_per_operand_exhaustive_i8_v1",
        families=["elementwise_map"],
        placement="fused_accelerator",
        signature=dict(dtypes=["int8"], composed_with=["residual_add"]),
    )
    if screens is not None:
        row["readout_gain_screens"] = screens
    return dict(status="reviewed", operations=[row])


def _component_signature(witness_gain, component_gain, *, start=0, stop=16, columns=17):
    from merlin.targetgen.operand_sum_numeric import audit_i8_per_operand_sum

    witness = audit_i8_per_operand_sum(lhs_scale=SIXTH, rhs_scale=TENTH, readout_scale=witness_gain, relu=True, facet=FACET)
    witness = {"command_index": 0, "form": "fused_operand_sum", **witness}
    return dict(
        family="elementwise_map",
        operand_dtype="i8",
        composed_with=["residual_add"],
        numeric_screen=witness,
        output_column_component=dict(start=start, stop=stop, columns=columns, readout_scale=component_gain),
    )


def _status(spec, signature):
    from merlin.targetgen.software_spec import admit_operation

    return admit_operation(spec, "relu", signature, "fused_accelerator")["status"]


def test_validator_admits_a_component_only_for_its_own_witness_gain():
    spec = _admission_spec(["tensor", "column_component"])
    assert _status(spec, _component_signature(SIXTH, SIXTH)) == "admitted"
    assert _status(spec, _component_signature(TENTH, TENTH, start=16, stop=17)) == "admitted"
    # A witness for another gain can never admit this component's columns.
    assert _status(spec, _component_signature(SIXTH, TENTH)) == "unknown"
    assert _status(spec, _component_signature(TENTH, SIXTH)) == "unknown"


def test_validator_refuses_a_component_row_against_a_row_without_the_declaration():
    signature = _component_signature(SIXTH, SIXTH)
    for screens in (None, ["tensor"]):
        assert _status(_admission_spec(screens), signature) == "unknown"
    # A tensor-gain signature is unchanged by the new key's absence...
    plain = dict(signature)
    plain.pop("output_column_component")
    assert _status(_admission_spec(None), plain) == "admitted"
    assert _status(_admission_spec(["tensor", "column_component"]), plain) == "admitted"
    # ...but a row declaring only column components does not admit a whole-tensor signature.
    assert _status(_admission_spec(["column_component"]), plain) == "unknown"


def test_a_declaration_that_never_reads_the_witness_ignores_the_component_field():
    # The carrier command is admitted by a plain operand-sum declaration (no numerical contract); the
    # component field must not make that unrelated declaration unknown.
    spec = _admission_spec(None)
    row = spec["operations"][0]
    row.pop("numerical_contract")
    assert _status(spec, _component_signature(SIXTH, SIXTH)) == "admitted"
    row["numerical_contract"] = "operand_sum_exhaustive_i8_v1"  # but a single-rounding contract row is not exempt
    assert _status(spec, _component_signature(SIXTH, SIXTH)) == "unknown"


@pytest.mark.parametrize(
    "mangle",
    [
        {"start": 3, "stop": 3},
        {"start": 0, "stop": 18},
        {"start": -1},
        {"readout_scale": SIXTH + 1e-12},
        {"start": True},
        {"readout_scale": 1},
    ],
)
def test_validator_refuses_a_malformed_component(mangle):
    signature = _component_signature(SIXTH, SIXTH)
    signature["output_column_component"].update(mangle)
    assert _status(_admission_spec(["tensor", "column_component"]), signature) == "unknown"


# ---- independent evaluation of the per-operand source arithmetic ---------------------------------


def numpy_outputs(lhs, rhs, gain, relu):
    codes = np.arange(-128, 128, dtype=np.float32)
    left = np.clip(np.rint(codes * np.float32(lhs)), -128, 127).astype(np.int32)
    right = np.clip(np.rint(codes * np.float32(rhs)), -128, 127).astype(np.int32)
    total = left[:, None] + right[None, :]
    if relu:
        total = np.maximum(total, 0)
    return np.clip(np.rint(total.astype(np.float32) * np.float32(gain)), -128, 127).astype(np.int16)


def exact_outputs(lhs, rhs, gain, relu):
    """Pure Python: exact Fraction products, rounded once to binary32 via struct, banker's round()."""

    def product(integer, scale):
        return f32(float(Fraction(integer) * Fraction(scale)))

    def code(value):
        return min(127, max(-128, round(value)))

    out = np.zeros((256, 256), dtype=np.int16)
    for i, a in enumerate(range(-128, 128)):
        left = code(product(a, lhs))
        for j, b in enumerate(range(-128, 128)):
            total = left + code(product(b, rhs))
            total = max(total, 0) if relu else total
            out[i, j] = code(product(total, gain))
    return out


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("gain", [SIXTH, TENTH, 0.5])
def test_witness_reference_is_two_independent_evaluations(gain, relu):
    from merlin.targetgen.operand_sum_numeric import audit_i8_per_operand_sum

    witness = audit_i8_per_operand_sum(lhs_scale=SIXTH, rhs_scale=TENTH, readout_scale=gain, relu=relu, facet=FACET)
    assert witness["status"] == "within_bound" and witness["max_error_lsb"] == 0 and witness["n_over_bound"] == 0
    vectorized, exact = numpy_outputs(SIXTH, TENTH, gain, relu), exact_outputs(SIXTH, TENTH, gain, relu)
    assert np.array_equal(vectorized, exact)
    # The audit's digest covers its reference and device arrays: rebuild it from OUR arrays.
    digest = hashlib.sha256(json.dumps(witness["parameters"], sort_keys=True, separators=(",", ":")).encode())
    digest.update(vectorized.astype(np.int8).tobytes())
    digest.update(vectorized.astype(np.int8).tobytes())
    assert digest.hexdigest() == witness["witness_sha256"]


def test_pinned_extremes_and_ties():
    sixth_no_relu = numpy_outputs(1.0, 1.0, SIXTH, False)
    index = lambda code: code + 128  # noqa: E731
    assert sixth_no_relu[index(-128), index(-128)] == -43 and sixth_no_relu[index(127), index(127)] == 42
    assert exact_outputs(1.0, 1.0, SIXTH, False)[index(-128), index(-128)] == -43
    with_relu = numpy_outputs(1.0, 1.0, SIXTH, True)
    assert with_relu[index(-128), index(-128)] == 0 and with_relu[index(-1), index(0)] == 0
    assert with_relu[index(127), index(127)] == 42
    for relu in (False, True):
        half = numpy_outputs(1.0, 1.0, 0.5, relu)
        assert np.array_equal(half, exact_outputs(1.0, 1.0, 0.5, relu))
        # exact .5 ties go to the even code: 1 * 0.5 -> 0, 3 * 0.5 -> 2
        assert half[index(1), index(0)] == 0 and half[index(2), index(1)] == 2
    negative = numpy_outputs(1.0, 1.0, 0.5, False)
    assert negative[index(-1), index(0)] == 0 and negative[index(-2), index(-1)] == -2
