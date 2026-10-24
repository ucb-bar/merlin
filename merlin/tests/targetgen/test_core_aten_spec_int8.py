"""Exact per-site admission and static i32/f32 chain qualification."""

import copy

import numpy as np
import pytest
from targetgen_fixtures import measured_routing
from xdsl.parser import Parser

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import make_context
from merlin.targetgen.core_aten_spec_int8 import (
    DEQUANTIZE,
    QUANTIZE,
    admission_plan,
    host_signature,
    pipeline_refusals,
)
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import admit_operation, load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec


@pytest.fixture
def selection():
    base = repo_root() / "examples/gemmini/target"
    spec = load_software_spec(base / "software-spec.yaml")
    resolved, record = resolve_spec(spec, {})
    return dict(
        spec=resolved,
        host=validate_host_capabilities(
            load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")
        ),
        resolution=record,
        identities=dict(
            software_spec={"path": str(base / "software-spec.yaml")},
            host_capabilities={"path": str(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")},
        ),
    )


@pytest.mark.parametrize("rank,expected", [(rank, "admitted") for rank in range(6)] + [(6, "unsupported")])
def test_exact_dequantize_rank_and_dtype(selection, rank, expected):
    signature = host_signature(DEQUANTIZE, rank, dict(zero_point=0))
    for authority in ("spec", "host"):
        assert admit_operation(selection[authority], DEQUANTIZE, signature, "host")["status"] == expected
        for operand in (["i32", "f32", "i64"], ["i8", "f64", "i64"], ["i8", "f32", "i32"]):
            wrong = {**signature, "ordered_operand_dtypes": operand}
            assert admit_operation(selection[authority], DEQUANTIZE, wrong, "host")["status"] != "admitted"


def test_exact_quantize_parameters_and_typed_zero_point(selection):
    signature = host_signature(QUANTIZE, 3, dict(zero_point=0, quant_min=-128, quant_max=127))
    for authority in ("spec", "host"):
        assert admit_operation(selection[authority], QUANTIZE, signature, "host")["status"] == "admitted"
        for parameters in (
            dict(zero_point=1, quant_min=-128, quant_max=127),
            dict(zero_point=0, quant_min=-127, quant_max=127),
        ):
            wrong = {**signature, "quantization_parameters": parameters}
            assert admit_operation(selection[authority], QUANTIZE, wrong, "host")["status"] != "admitted"
        wrong = {**signature, "ordered_operand_dtypes": ["f32", "f32", "i32"]}
        assert admit_operation(selection[authority], QUANTIZE, wrong, "host")["status"] != "admitted"


def test_raw_i32_chain_does_not_require_narrowing_product_semantics(selection):
    before = copy.deepcopy(selection)
    assert pipeline_refusals(selection) == []
    assert selection == before
    proposed = copy.deepcopy(selection)
    proposed["host"]["status"] = "unreviewed"
    assert pipeline_refusals(proposed)


def test_rank3_plan_requires_flat_captured_readout_and_retains_mac_geometry(selection):
    source = """builtin.module {
      func.func @forward(%a: tensor<2x3x40xf32>, %b: tensor<2x40x4xf32>) -> tensor<2x3x4xf32> {
        %c = arith.constant 0.0 : f32
        %e = tensor.empty() : tensor<2x3x4xf32>
        %z = linalg.fill ins(%c : f32) outs(%e : tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
        %r = linalg.generic {
          indexing_maps = [affine_map<(b,m,n,k) -> (b,m,k)>,
                           affine_map<(b,m,n,k) -> (b,k,n)>, affine_map<(b,m,n,k) -> (b,m,n)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]
        } ins(%a, %b : tensor<2x3x40xf32>, tensor<2x40x4xf32>) outs(%z : tensor<2x3x4xf32>) {
          ^bb0(%x: f32, %y: f32, %z0: f32):
            %v = arith.mulf %x, %y : f32
            %w = arith.addf %z0, %v : f32
            linalg.yield %w : f32
        } -> tensor<2x3x4xf32>
        func.return %r : tensor<2x3x4xf32>
      }
    }"""
    module = Parser(make_context(), source).parse_module()
    before = str(module)
    plan = admission_plan(module, selection)
    assert str(module) == before
    assert plan["quantized_contractions"] == 0
    assert plan["calibration"]["performed"] is False
    assert plan["calibration"]["source"] == "declared_observer_domain"
    (site,) = plan["sites"]
    contraction = next(row for row in site["steps"] if row["step"] == 3)
    assert contraction["software"]["status"] == "admitted"
    assert contraction["signature"]["ordered_operand_dtypes"] == ["i8", "i8", "i32"]
    assert site["source_dtypes"] == ["f32", "f32", "f32"]
    assert site["status"] == "admitted"
    assert site["partial_sum_obligation"]["max_true_k"] == 31
    assert site["reasons"] == []
    assert contraction["signature"]["rank"] == 3
    assert next(row for row in site["steps"] if row["step"] == 5)["signature"]["rank"] == 2
    assert site["host_readout_bridge"]["status"] == "planned_static_geometry_requires_capture_proof"
    assert site["host_readout_bridge"]["flat_shape"] == [6, 4]


def test_dynamic_verdict_is_explicitly_diagnostic():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    report = dict(provenance={}, cases=[dict(overload="aten.mm.default", status="bundled", numeric_policy="int8_w8a8")])
    verdict = grade_core_aten_batch(report, execution_error="not run")["cases"]["aten.mm.default"]
    assert verdict["numeric_qualification"] == "diagnostic, not spec-conformant"


def test_spec_int8_host_pass_is_counted_only_as_original_host():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    case = dict(expected=dict(kind="tensor", dtype="float32", shape=[1], values=[2.0]), comparison="torch_close")
    report = dict(
        provenance={},
        cases=[
            dict(
                overload="aten.mm.default",
                status="bundled",
                case=case,
                numeric_policy="spec_int8",
                numeric_admission=dict(status="refused", quantized_contractions=0),
                output_indices=[0],
                output_abi=[dict(dtype="f32", shape=[1])],
            )
        ],
    )
    result = grade_core_aten_batch(report, [np.array([2.0], np.float32).tobytes()])
    verdict = result["cases"]["aten.mm.default"]
    assert verdict["status"] == verdict["numeric_status"] == "pass"
    assert verdict["numeric_execution_policy"] == "original_host"
    assert verdict["numeric_qualification"] == "original host fallback; no spec-int8 result"
    assert result["numeric_passed_by_lane_and_policy"]["host"]["original_host"] == 1
    assert result["numeric_passed_by_lane_and_policy"]["host"]["spec_int8"] == 0
    assert result["numeric_passed_by_lane_and_policy"]["device"]["spec_int8"] == 0
    assert result["spec_int8_passed_count"] == 0
    assert result["spec_int8_host_fallback_passed_count"] == 1


@pytest.mark.parametrize("evidence_valid", [False, True])
def test_capture_spec_grade_keeps_dynamic_gate_and_host_counts_separate(monkeypatch, evidence_valid):
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda _: evidence_valid)
    case = dict(expected=dict(kind="tensor", dtype="float32", shape=[1], values=[2.0]), comparison="torch_close")
    raw = np.array([2.0], np.float32).tobytes()
    record = dict(
        overload="aten.mm.default",
        status="bundled",
        case=case,
        fp32_case=copy.deepcopy(case),
        numeric_policy="spec_int8",
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1])],
        numeric_admission=dict(
            status="admitted",
            quantized_contractions=1,
            integer_reference=dict(status="prepared", engine="independent", expected_output_bytes={"0": raw.hex()}),
        ),
        routing=measured_routing("aten.mm.default"),
    )
    result = grade_core_aten_batch(dict(provenance={}, cases=[record]), [raw])
    row = result["cases"][record["overload"]]
    assert row["integer_reference"]["status"] == "pass"
    assert row["fp32_error"][0]["max_abs"] == 0
    assert "w8a8_policy" not in row
    assert row["status"] == ("pass" if evidence_valid else "execution_failed")
    assert result["spec_int8_passed_count"] == int(evidence_valid)
    assert result["spec_int8_host_fallback_passed_count"] == 0


def test_fp32_metrics_distinguish_zero_golden_and_zero_error():
    from merlin.targetgen.core_aten_numeric import fp32_error

    record = dict(
        output_indices=[0],
        output_abi=[dict(dtype="f32")],
        fp32_case=dict(expected=dict(kind="tensor", dtype="float32", shape=[2], values=[0.0, 1.0])),
    )
    (observed,) = fp32_error(record, [np.array([1.0, 1.0], np.float32).tobytes()])
    assert observed["max_rel"] == float("inf")
    assert observed["relative_error_at_zero"] == 1
    assert observed["rmse"] == pytest.approx(2**-0.5)
    assert observed["normalized_l2"] == 1.0
    assert observed["sqnr_db"] == 0.0


def rank2_module(k=40):
    """A 3xk by kx4 f32 matmul."""
    source = """builtin.module {
      func.func @forward(%a: tensor<3x40xf32>, %b: tensor<40x4xf32>) -> tensor<3x4xf32> {
        %c = arith.constant 0.0 : f32
        %e = tensor.empty() : tensor<3x4xf32>
        %z = linalg.fill ins(%c : f32) outs(%e : tensor<3x4xf32>) -> tensor<3x4xf32>
        %r = linalg.matmul ins(%a, %b : tensor<3x40xf32>, tensor<40x4xf32>)
          outs(%z : tensor<3x4xf32>) -> tensor<3x4xf32>
        func.return %r : tensor<3x4xf32>
      }
    }"""
    return Parser(make_context(), source.replace("40", str(k))).parse_module()


def test_rank2_chain_and_each_independent_host_rank_restriction(selection):
    module = rank2_module()
    plan = admission_plan(module, selection)
    (site,) = plan["sites"]
    assert site["status"] == "admitted", site["reasons"]
    assert {row["step"] for row in site["steps"]} == {1, 2, 3, 4, 5, 6}
    for identity, step in (
        ("host_quantize_per_tensor_int8", 1),
        ("host_i32_to_f32_cast", 5),
        ("host_f32_elementwise_exact", 6),
    ):
        restricted = copy.deepcopy(selection)
        row = next(row for row in restricted["host"]["operations"] if row["id"] == identity)
        row["signature"]["ranks"] = [3]
        (refused,) = admission_plan(module, restricted)["sites"]
        assert refused["status"] == "refused"
        assert all(f"step {step}" in r for r in refused["reasons"])


def test_integer_reference_compares_signed_zero_bits():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    record = dict(
        case_id="test",
        overload="test",
        status="bundled",
        numeric_policy="spec_int8",
        output_indices=[],
        output_abi=[],
        case=dict(expected=[]),
        numeric_admission=dict(
            integer_reference=dict(
                status="prepared",
                engine="independent",
                expected_output_bytes={"0": np.array([-0.0], np.float32).tobytes().hex()},
            )
        ),
    )
    result = grade_core_aten_batch(dict(provenance={}, cases=[record]), [np.array([0.0], np.float32).tobytes()])
    assert result["cases"]["test"]["integer_reference"]["status"] == "mismatch"
    assert result["cases"]["test"]["numeric_status"] == "mismatch"


def test_static_capture_freezes_original_float_intermediate_scales(tmp_path, selection):
    import os
    import subprocess

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("TorchAO capture interpreter is not configured")
    script = r"""
import sys
import torch
import m2m
from merlin.targetgen import _recipe_quantizer as quant
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
recipe = {
    "schema": "quant_recipe_v1", "status": "derived", "families": ["contraction"],
    "runtime_weight_contractions": True,
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
class Model(torch.nn.Module):
    def forward(self, a, b, c):
        return (a @ b) @ c
inputs = (
    torch.tensor([[1.25, -.91, .3], [.67, .15, -.79]]),
    torch.tensor([[.93, -1.1], [-.76, .23], [.41, .87]]),
    torch.tensor([[.7, -.1], [-.5, 1.1]]),
)
converted = quant.apply_recipe(Model().eval(), recipe, example_inputs=inputs)
reference = run_pt2e_reference(converted, inputs)
scale, observer = quant.observe_static_tensor(inputs[0] @ inputs[1], recipe["activation"])
assert reference.sites[1]["lhs"] == scale.item()
assert observer == "histogram"
integer, receipt = integerize_pt2e(converted, inputs, numerics="integer_readout")
assert torch.equal(integer(*inputs).view(torch.int32), reference.output.view(torch.int32))
text = m2m.convert(integer, inputs, backend="fx_importer").mlir_text
open(sys.argv[1], "w").write(text)
"""
    path = tmp_path / "captured.mlir"
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    # The scale-freeze observation is independent of a provider's implementation.
    # Exact readout placement requires source-bound provider coverage (tested on real captures).


@pytest.mark.parametrize("authority,key", [("spec", "host_f32_elementwise"), ("host", "host_f32_elementwise_exact")])
def test_readout_requires_authored_multiply_declaration(selection, authority, key):
    changed = copy.deepcopy(selection)
    row = next(r for r in changed[authority]["operations"] if r["id"] == key)
    row["ops"].remove("aten.mul.Tensor")
    site = admission_plan(rank2_module(), changed)["sites"][0]
    assert site["status"] == "refused"
    assert any("step 6" in reason for reason in site["reasons"])


def test_readout_requires_declared_binary32_rounding(selection):
    from merlin.targetgen.core_aten_spec_int8 import declaration_location

    changed = copy.deepcopy(selection)
    row = next(r for r in changed["host"]["operations"] if r["id"] == "host_f32_elementwise_exact")
    row["numerical_contract"]["semantics"].pop("rounding")
    location = declaration_location(selection, "host", row["id"], ("numerical_contract", "semantics", "rounding"))
    assert any(
        f"{location['path']}:{location['line']}" in reason
        for reason in admission_plan(rank2_module(), changed)["sites"][0]["reasons"]
    )


def test_reference_uses_exact_product_and_only_final_boundary_overflow():
    import os
    import subprocess

    script = r"""
import torch
import torchao.quantization.pt2e
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference

def model(sa, sb):
    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    x = graph.call_function(dequant, (a, sa, 0, -128, 127, torch.int8))
    y = graph.call_function(dequant, (b, sb, 0, -128, 127, torch.int8))
    result = graph.call_function(torch.ops.aten.mm.default, (x, y))
    graph.output(result)
    return torch.fx.GraphModule({}, graph)
inputs = (torch.tensor([[3]], dtype=torch.int8), torch.tensor([[5]], dtype=torch.int8))
sa, sb = 1.6571282148361206, 0.824306309223175
reference = run_pt2e_reference(model(sa, sb), inputs)
from merlin.targetgen.exact_arithmetic import rational,round_binary
expected = round_binary(15*rational(sa)*rational(sb))
eager = (torch.tensor(15, dtype=torch.float32) * sa) * sb
folded = torch.tensor(15, dtype=torch.float32) * torch.tensor(sa * sb, dtype=torch.float32)
assert eager.item() != folded.item()
assert reference.output.item() == expected
try:
    run_pt2e_reference(model(1e38, 1e38), inputs)
except ValueError as exc:
    assert "boundary overflows" in str(exc)
else:
    raise AssertionError("final boundary overflow must refuse")
assert torch.isfinite(run_pt2e_reference(model(1e38,1e-38),inputs).output).all()
"""
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("k,admitted", [(1024, True), (1025, False)])
def test_cast_selection_enforces_declared_exact_range(selection, k, admitted):
    (site,) = admission_plan(rank2_module(k), selection)["sites"]
    assert (site["status"] == "admitted") is admitted
    assert site["cast_exact_range_obligation"]["max_abs_bound"] == k * 128 * 128
    assert site["cast_exact_range_obligation"]["exact_limit"] == 2**24
    if not admitted:
        location = site["cast_exact_range_obligation"]["source"]
        expected = f"{location['path']}:{location['line']}"
        assert any(expected in reason and "exact_range" in reason for reason in site["reasons"])


def test_step6_checks_selected_declaration_instead_of_shadowed_exact_one(selection):
    changed = copy.deepcopy(selection)
    exact = next(r for r in changed["host"]["operations"] if r["id"] == "host_f32_elementwise_exact")
    shadow = copy.deepcopy(exact)
    shadow["id"] = "earlier_nonexact_multiply"
    shadow["numerical_contract"]["semantics"].pop("rounding")
    changed["host"]["operations"].insert(0, shadow)
    (site,) = admission_plan(rank2_module(), changed)["sites"]
    assert site["status"] == "refused"
    assert any("earlier_nonexact_multiply" in reason and "rounding" in reason for reason in site["reasons"])
    assert all(row["host"]["declaration"] == shadow["id"] for row in site["steps"] if row["step"] == 6)


def test_exact_byte_boundary_covers_integer_and_floating_results():
    from merlin.targetgen.core_aten_numeric import map_reference_byte_boundary

    values = [
        np.array([200], np.uint8),
        np.array([-0.0], np.float16),
        np.array([2**53 + 1], np.int64),
        np.array([1 + 2**-52], np.float64),
    ]
    raw = [value.tobytes().hex() for value in values]
    fresh = dict(output_indices=[3, 1, 0], semantic_boundary=dict(post_indices=[2]))
    reference = dict(output_bytes=raw[:3], post_input_bytes=raw[3:])
    assert map_reference_byte_boundary(fresh, reference, 4) == dict(zip(["3", "1", "0", "2"], raw))
    missing = {**fresh, "output_indices": [0, 1, 4]}
    with pytest.raises(ValueError, match="every exported result"):
        map_reference_byte_boundary(missing, reference, 4)


def test_channel_scale_error_metrics_remain_informational():
    from merlin.targetgen.core_aten_numeric import fp32_error

    expected = dict(kind="tensor", dtype="float32", shape=[1, 2], values=[[1.0, 2.0]])
    record = dict(
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1, 2])],
        fp32_case=dict(expected=expected),
        scales=[dict(lhs=0.5, rhs=[0.25, 0.5], product=[0.125, 0.25])],
    )
    (metrics,) = fp32_error(record, [np.array([[1.0, 2.0]], dtype=np.float32).tobytes()])
    assert metrics["max_abs"] == 0
    assert metrics["error_in_output_lsbs"] == [dict(site=0, lsb=[0.125, 0.25], max_abs=None, rmse=None)]


@pytest.mark.parametrize("rank", [-1, True, 1.5])
def test_software_rank_constraints_refuse_non_ranks(rank):
    from merlin.targetgen.software_spec import validate_software_spec

    document = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")
    document["operations"]["host_dequantize_per_tensor"]["ranks"] = [rank]
    with pytest.raises(ValueError, match="nonnegative integers"):
        validate_software_spec(document)


@pytest.mark.parametrize(
    "k,admitted", [(1024, True), (1025, True), (4096, True), (11008, True), (131071, True), (131072, False)]
)
def test_planned_integer_readout_uses_int32_bound(selection, k, admitted):
    (site,) = admission_plan(rank2_module(k), selection, planned_exact_readout=True)["sites"]
    assert (site["status"] == "admitted") is admitted
    assert site["cast_exact_range_obligation"] is None
    assert site["readout"] == "int32_host_exact_pending_capture"
    readout = next(row for row in site["steps"] if row["step"] == 5)
    assert readout["qualification"] == "pending_integer_capture_and_provider_proof"
    assert not any(row for row in site["steps"] if row["step"] == 6)
    if admitted:
        assert site["partial_sum_obligation"]["full_result_bound"] == k * 128 * 128
    else:
        assert any("i32 aggregation" in reason for reason in site["reasons"])


@pytest.mark.parametrize("field,value", [("status", "unreviewed"), ("rounding", "half_up"), ("api", "unknown")])
def test_planned_integer_readout_still_needs_reviewed_exact_api(selection, field, value):
    changed = copy.deepcopy(selection)
    changed["host"]["exact_readout"][field] = value
    (site,) = admission_plan(rank2_module(), changed, planned_exact_readout=True)["sites"]
    assert site["status"] == "refused"
    assert any("exact rational host readout" in reason for reason in site["reasons"])


def test_planned_flag_cannot_bypass_captured_float_chain(selection):
    (site,) = admission_plan(rank2_module(), selection, captured=True, planned_exact_readout=True)["sites"]
    assert site["status"] == "refused"
    assert any("floating readout lacks" in reason for reason in site["reasons"])
