"""Conditional packet 2 approval grants only the declared execution domains."""

import copy

import pytest

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.admission_conditions import screen_conditions, screen_emission_obligations
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.operator_review import PENDING, review_status
from merlin.targetgen.software_spec import admit_operation, load_software_spec


ROOT = repo_root() / "examples/gemmini"
PE = dict(pe_partial_abs_bound=262144, accumulator_partial_abs_bound=262144,
          pe_bound_scope="every_retained_partial_sum_including_total_os_k")
RAW = dict(readout_exit="accumulator_full", readout_stages=[], columns=16, pooling=False)
SEED = dict(seed_load="unscaled_full_i32_shrink0_once")
SELECTORS = dict(special_value_scope="source_cpu_certified_pointwise",
                 typed_boundaries="every_original_operation",
                 source_word_selectors="preserved_or_explicitly_reconstructed",
                 unproved_aggregate_or_competing_nan="refused")
EXECUTION = {
    "contraction": dict(**PE, **SEED, tail_initialization="explicit_zero",
                        constants="signed_int8_or_exact_decomposition"),
    "elementwise_map": RAW,
    "elementwise_map_standalone": dict(**PE, **SEED, **RAW, load_scale=1.0,
                                       affine_path="weighted_integer_mesh",
                                       weights="signed_int8_or_exact_decomposition"),
    "pooling_on_readout": dict(pool_dtype="signed_int8", pool_result="values_only", columns=16,
                               pool_size=3, pool_stride=2, pool_grid_max=255, readout_monotone=True,
                               valid_windows=True, padded=True, padding_value=0, producer_lower_bound=0),
    "host_exact_native_boundary": dict(**SELECTORS, scope="cpu_source_representation_reconstruction",
                                       source_integer_wrap="typed_boundaries",
                                       reduction_order_nan_theorem="not_a_reduction_or_separately_proved"),
    "host_f32_elementwise_exact": dict(scope="selected_cpu_source_primitives", fast_math=False,
                                       typed_boundaries="every_original_primitive",
                                       source_body_order="proved_including_relu_alpha_and_opmath",
                                       source_word_selectors="preserved_or_explicitly_reconstructed"),
    "host_f32_reduction": dict(scope="selected_pinned_cpu_lowering", reduction_order="fixed_pinned_lowering",
                               rounding_cadence="every_f32_primitive",
                               source_numerical_qualification="separately_proved"),
    "host_quantize_exact": dict(scope="declared_quantized_cpu_boundary", scale=0.1,
                                scale_form="retained_exact_binary_f64_scalar", input_domain="finite_f32_only",
                                quantize_program="exact_divide_single_RNE_then_clamp"),
    "host_dequantize_per_tensor": dict(scope="cpu_materialized_dequantize_output", scale=0.125,
                                       scale_form="retained_exact_binary_f32_scalar",
                                       dequantize_program="exact_centered_product_single_output_f32_RNE",
                                       output_overflow="refuse"),
}


def selection(name):
    host = name.startswith("host_")
    document = (validate_host_capabilities(load_yaml(ROOT / "platform/host-capabilities-core-aten.yaml"))
                if host else load_software_spec(ROOT / "target/software-spec.yaml"))
    row = next(row for row in document["operations"] if row["id"] == name)
    document["operations"] = [row]
    row.pop("hardware", None)
    placement = "host" if host else "fused_accelerator" if name in {"elementwise_map", "pooling_on_readout"} else "accelerator"
    row["placement"] = placement
    return document, row, placement


def observed(name):
    signature = dict(family="elementwise_map", operand_dtype="int8", accum_dtype="i32", rank=2,
                     layout="row_major_contiguous", aliasing="disjoint_inputs_outputs",
                     tails="zero_pad_valid_window", broadcasting="none", tensor_operands=2,
                     expression="affine", execution=copy.deepcopy(EXECUTION[name]))
    if name == "contraction":
        signature["family"] = "contraction"
    elif name == "pooling_on_readout":
        signature.update(family="reduction", kind="max")
    elif name == "host_exact_native_boundary":
        signature.update(numerical_family="F2", native_ieee_special_values=True)
    elif name in {"host_f32_elementwise_exact", "host_f32_reduction"}:
        signature.update(operand_dtype="f32", ordered_result_dtypes=["f32"])
    elif name == "host_quantize_exact":
        signature.update(operand_dtype="f32", ordered_operand_dtypes=["f32", "f64", "i64"], ordered_result_dtypes=["i8"],
                         quantization_parameters=dict(zero_point=0, quant_min=-128, quant_max=127))
    elif name == "host_dequantize_per_tensor":
        signature.update(ordered_operand_dtypes=["i8", "f32", "i64"], ordered_result_dtypes=["f32"],
                         quantization_parameters=dict(zero_point=0))
    return signature


def decision(name, signature):
    document, row, placement = selection(name)
    op = (row.get("ops") or [signature["family"]])[0]
    return admit_operation(document, op, signature, placement)


@pytest.mark.parametrize("name", EXECUTION)
def test_reviewed_domain_admits_source_without_execution_semantics(name, monkeypatch):
    monkeypatch.setenv("MERLIN_ACCEPT_AUTHORED_PENDING_OPERATOR_REVIEW", "0")
    signature = observed(name)
    assert decision(name, signature)["status"] == "admitted"
    signature.pop("execution")
    assert decision(name, signature)["status"] == "admitted"
    assert decision(name, signature)["emission_obligations"]


@pytest.mark.parametrize("name,field,value", [
    ("contraction", "pe_partial_abs_bound", 32 * 128 * 128),
    ("contraction", "accumulator_partial_abs_bound", 1 << 31),
    ("contraction", "pe_bound_scope", "single_instruction_only"),
    ("contraction", "tail_initialization", "short_mvin"),
    ("contraction", "seed_load", "scaled_int8"),
    ("elementwise_map", "readout_stages", ["relu"]),
    ("elementwise_map", "pooling", True),
    ("elementwise_map_standalone", "load_scale", -1.0),
    ("elementwise_map_standalone", "load_scale", 2.0),
    ("elementwise_map_standalone", "pe_partial_abs_bound", 524288),
    ("elementwise_map_standalone", "accumulator_partial_abs_bound", 2147483648),
    ("pooling_on_readout", "producer_lower_bound", -1),
    ("pooling_on_readout", "padding_value", -128),
    ("pooling_on_readout", "pool_dtype", "i32"),
    ("pooling_on_readout", "columns", 17),
    ("pooling_on_readout", "pool_size", 4),
    ("pooling_on_readout", "pool_stride", 0),
    ("pooling_on_readout", "pool_grid_max", 256),
    ("pooling_on_readout", "readout_monotone", False),
    ("host_exact_native_boundary", "typed_boundaries", "single_final_RNE"),
    ("host_exact_native_boundary", "source_word_selectors", "riscv_canonical_nan"),
    ("host_f32_elementwise_exact", "fast_math", True),
    ("host_f32_elementwise_exact", "source_body_order", "reassociated"),
    ("host_f32_reduction", "reduction_order", "order_independent"),
    ("host_f32_reduction", "rounding_cadence", "single_final"),
    ("host_quantize_exact", "scale", 0.0),
    ("host_quantize_exact", "scale", float("inf")),
    ("host_quantize_exact", "scale", float("nan")),
    ("host_quantize_exact", "input_domain", "all_f32_words"),
    ("host_quantize_exact", "quantize_program", "f32_reciprocal_multiply"),
    ("host_dequantize_per_tensor", "scale", -1.0),
    ("host_dequantize_per_tensor", "dequantize_program", "round_every_internal_DQ_node"),
    ("host_dequantize_per_tensor", "output_overflow", "unselected_infinity"),
])
def test_execution_violation_is_recorded_separately_from_source_admission(name, field, value):
    signature = observed(name)
    signature["execution"][field] = value
    assert decision(name, signature)["status"] == "admitted"
    _, row, _ = selection(name)
    result = screen_emission_obligations(row["emission_obligations"], signature)
    assert any(r["observation_status"] != "matched" for r in result.values() if r["status"] == "compiler_obligation")


def test_unit_load_and_selected_narrow_readout_are_separate_legal_paths():
    signature = observed("elementwise_map_standalone")
    signature["execution"].update(affine_path="unit_scale_accumulating_loads", weights="unit")
    assert decision("elementwise_map_standalone", signature)["status"] == "admitted"
    signature = observed("elementwise_map")
    signature["execution"] = dict(readout_exit="narrow_i8", readout_program="selected_scheme_order",
                                  readout_strategy="f32_product_rne",
                                  scale_delivery="scalar_per_command_partition",
                                  relu_commutation="nonnegative_multiplier_or_source_order_proof")
    assert decision("elementwise_map", signature)["status"] == "admitted"
    signature["execution"]["readout_strategy"] = "exact_rational_host"
    assert decision("elementwise_map", signature)["status"] == "admitted"
    _, row, _ = selection("elementwise_map")
    assert screen_emission_obligations(row["emission_obligations"], signature)["execution_program"]["observation_status"] != "matched"


def test_source_special_value_policy_review_is_an_admission_dependency(monkeypatch):
    monkeypatch.setenv("MERLIN_ACCEPT_AUTHORED_PENDING_OPERATOR_REVIEW", "0")
    document, row, placement = selection("host_exact_native_boundary")
    document["native_ieee_special_values"]["status"] = PENDING
    result = admit_operation(document, "elementwise_map", observed(row["id"]), placement)
    assert result["status"] == "unknown" and result["review_status"] == PENDING


def test_approved_rows_have_no_nested_pending_evidence():
    for name in EXECUTION:
        document, row, _ = selection(name)
        assert review_status(row) == "reviewed"
        for value in (row, row.get("numerical_contract", {}), document.get("native_ieee_special_values", {})):
            assert "pending operator review" not in str(value).lower()
            assert PENDING not in str(value)


def test_numeric_conditions_reject_boolean_and_nonfinite_bounds():
    rule = dict(path="bound", min=0, max=524287)
    for value in (True, float("nan"), float("inf"), "16"):
        assert screen_conditions(rule, dict(bound=value))[0] == "refused"


def test_deleted_software_quantizer_cannot_authorize_f32_reciprocal_q():
    from merlin.targetgen.core_aten_spec_int8 import QUANTIZE, host_signature
    from merlin.targetgen.spec_fact_drift import resolve_spec

    document = resolve_spec(load_software_spec(ROOT / "target/software-spec.yaml"), {})[0]
    assert "host_quantize" not in {row["id"] for row in document["operations"]}
    signature = host_signature(QUANTIZE, 3, dict(zero_point=0, quant_min=-128, quant_max=127))
    assert admit_operation(document, QUANTIZE, signature, "host")["status"] != "admitted"
