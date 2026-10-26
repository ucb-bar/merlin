"""Device declarations state explicit domains, and admission screens what they declare."""

import copy

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.software_spec import (
    DOMAIN_AXES,
    admit_operation,
    operation_domains,
    validate_software_spec,
)

SPEC = repo_root() / "examples/gemmini/target/software-spec.yaml"


@pytest.fixture
def document():
    return yaml.safe_load(SPEC.read_bytes())


def test_every_device_declaration_states_its_domain(document):
    domains = operation_domains(validate_software_spec(document))
    assert domains and "host_quantize" not in domains
    for identity, record in domains.items():
        assert record["arithmetic"], identity
        for axis in DOMAIN_AXES:
            assert record[axis] is not None, (identity, axis)
            if isinstance(record[axis], dict):
                assert record[axis]["unconstrained"].strip()


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda ops: ops["contraction"].pop("tails"), "undeclared"),
        (lambda ops: ops["movement"].pop("arithmetic"), "undeclared"),
        (lambda ops: ops["movement"].update(unconstrained={"tails": " "}), "written reason"),
        (lambda ops: ops["movement"].update(unconstrained={"tails": "x", "layouts": "y"}), "both constrains"),
        (lambda ops: ops["movement"].update(unconstrained={"dtypes": "x", "tails": "y"}), "not a domain axis"),
        (lambda ops: ops["reduction_standalone"].update(kinds=["max"]), "only sum"),
        (lambda ops: ops["contraction"].update(arithmetic="none"), "arithmetic none"),
        (lambda ops: ops["contraction"].update(arithmetic="integer_sum"), "requires constraints"),
        (lambda ops: ops["contraction"].update(arithmetic="floating"), "must be one of"),
    ],
)
def test_incomplete_or_inconsistent_domains_are_refused(document, edit, message):
    edit(document["operations"])
    with pytest.raises(ValueError, match=message):
        validate_software_spec(document)


def test_domains_are_required_only_when_the_spec_selects_them(document):
    document["operations"]["contraction"].pop("tails")
    with pytest.raises(ValueError):
        validate_software_spec(copy.deepcopy(document))
    document.pop("operation_domains")
    validate_software_spec(document)


def test_an_observed_arithmetic_class_must_match_the_declaration(document):
    spec = validate_software_spec(document)
    signature = dict(
        family="contraction",
        operand_dtype="i8",
        accum_dtype="i32",
        rank=2,
        layout="row_major_contiguous",
        tails="zero_pad_valid_window",
        broadcasting="none",
        aliasing="disjoint_inputs_outputs",
        execution=dict(
            pe_partial_abs_bound=262144,
            accumulator_partial_abs_bound=262144,
            pe_bound_scope="every_retained_partial_sum_including_total_os_k",
            seed_load="unscaled_full_i32_shrink0_once",
            tail_initialization="explicit_zero",
            constants="signed_int8_or_exact_decomposition",
        ),
    )
    assert admit_operation(spec, "contraction", signature, "accelerator")["status"] == "admitted"
    refused = admit_operation(spec, "contraction", {**signature, "arithmetic": "none"}, "accelerator")
    assert refused["status"] == "unsupported" and "arithmetic" in refused["reason"]
    outside = admit_operation(spec, "contraction", {**signature, "rank": 5}, "accelerator")
    assert outside["status"] == "unsupported" and "rank" in outside["reason"]


def _representation_signature(family="movement", numerical_family="F1"):
    return dict(
        family=family,
        numerical_family=numerical_family,
        operand_dtype="i8",
        accum_dtype="i32",
        readout_dtype="i32",
        rank=2,
        layout="row_major_contiguous",
        aliasing="disjoint_inputs_outputs",
    )


def _movement_execution():
    return dict(
        mvin_scale=1.0,
        minimum_dram_address=1,
        mvout_pool_stride=0,
        mvout_pool_size=0,
        maximum_rows_per_command=16,
        maximum_bytes_per_command=64,
        transpose=False,
        byte_order="little",
        minimum_stride=0,
        disjoint_inputs_outputs=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mvin_scale", 2.0),
        ("minimum_dram_address", 0),
        ("mvout_pool_stride", 1),
        ("mvout_pool_size", 1),
        ("maximum_rows_per_command", 17),
        ("maximum_bytes_per_command", 65),
        ("transpose", True),
        ("byte_order", "big"),
        ("minimum_stride", -1),
        ("disjoint_inputs_outputs", False),
    ],
)
def test_movement_emission_obligations_do_not_constrain_source_admission(document, field, value):
    spec = validate_software_spec(document)
    signature = _representation_signature()
    row = next(row for row in spec["operations"] if row["id"] == "movement_bitwise")
    assert "execution_constraints" not in row["signature"]
    assert row["emission_obligations"][field]["status"] in {"instruction_audit", "compiler_obligation"}
    # Neither an absent command stream nor unrelated source execution metadata
    # can establish or revoke an implementation's later emission witness.
    assert admit_operation(spec, "movement", signature, "accelerator")["status"] == "admitted"
    signature["execution"] = {**_movement_execution(), field: value}
    assert admit_operation(spec, "movement", signature, "accelerator")["status"] == "admitted"


@pytest.mark.parametrize("dataflow,expected", [("WS", "admitted"), ("OS", "admitted"), (None, "admitted")])
def test_limb_source_admission_records_weight_stationary_duty(document, dataflow, expected):
    spec = validate_software_spec(document)
    signature = _representation_signature("contraction", "F2")
    signature["execution"] = dict(
        readout="full_i32",
        mvout_scale=False,
        mvout_relu=False,
        maximum_mvout_columns=16,
        mvout_pool_stride=0,
        mvout_pool_size=0,
    )
    if dataflow is not None:
        signature["execution"]["dataflow"] = dataflow
    assert admit_operation(spec, "contraction", signature, "accelerator")["status"] == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("readout", "int8"),
        ("mvout_scale", True),
        ("mvout_relu", True),
        ("maximum_mvout_columns", 17),
        ("mvout_pool_stride", 1),
        ("mvout_pool_size", 1),
    ],
)
def test_bilinear_readout_does_not_constrain_source_eligibility(document, field, value):
    spec = validate_software_spec(document)
    signature = _representation_signature("contraction", "F2")
    signature["execution"] = dict(
        dataflow="WS",
        readout="full_i32",
        mvout_scale=False,
        mvout_relu=False,
        maximum_mvout_columns=16,
        mvout_pool_stride=0,
        mvout_pool_size=0,
    )
    signature["execution"][field] = value
    assert admit_operation(spec, "contraction", signature, "accelerator")["status"] == "admitted"


@pytest.mark.parametrize("scale,expected", [(1.0, "admitted"), (2.0, "admitted"), (None, "admitted")])
def test_linear_accumulate_on_load_requires_unit_scale(document, scale, expected):
    spec = validate_software_spec(document)
    signature = _representation_signature("elementwise_map", "F2")
    signature["execution"] = {"dataflow": "WS", **({"mvin_scale": scale} if scale is not None else {})}
    assert admit_operation(spec, "elementwise_map", signature, "accelerator")["status"] == expected


@pytest.mark.parametrize(
    "pending", ["calibration_domain", "quantization_nonfinite", "readout_specials", "f32_boundary_overflow"]
)
def test_held_numeric_policies_are_review_gated(document, monkeypatch, pending):
    from merlin.targetgen.native_precision_contract import proves_quantized_model
    from merlin.targetgen.operator_review import ACCEPT_ENV, PENDING
    from merlin.targetgen.software_spec import numerical_policy_review

    numeric = document["numerical_semantics"]
    numeric["calibration_domain"]["status"] = "reviewed"
    for review in numeric["policy_reviews"].values():
        review["status"] = "reviewed"
    selected = numeric["policy_reviews"][pending]
    selected["status"] = PENDING
    plan = dict(
        integer_reference={"engine": "independent_pt2e_exact_rational"},
        capture_recipe={
            "calibration_domain": copy.deepcopy(numeric["calibration_domain"]),
            "quantization_nonfinite": numeric["quantization_nonfinite"],
        },
    )
    spec = validate_software_spec(document)
    signature = _representation_signature("elementwise_map", "F2")
    signature["execution"] = dict(dataflow="WS", mvin_scale=1.0)
    for enabled in ("0", "1"):
        monkeypatch.setenv(ACCEPT_ENV, enabled)
        assert numerical_policy_review(spec)["operator_accepted"] is (enabled == "1")
        assert proves_quantized_model(plan, spec) is (enabled == "1")
        assert admit_operation(spec, "elementwise_map", signature, "accelerator")["status"] == (
            "admitted" if enabled == "1" else "unknown"
        )


def test_quantization_storage_narrowing_overrides_a_broader_format_row(document):
    document["operations"]["component_linear"]["signature"]["operand_dtypes"].append("uint8")
    spec = validate_software_spec(document)
    signature = {**_representation_signature("elementwise_map", "F2"), "operand_dtype": "uint8"}
    signature["execution"] = dict(dataflow="WS", mvin_scale=1.0)
    decision = admit_operation(spec, "elementwise_map", signature, "accelerator")
    assert decision["status"] == "unsupported" and "quantization_contract.storage" in decision["reason"]


@pytest.mark.parametrize(
    "constraints",
    [
        {},
        {"mvin_scale": {"approximately": 1.0}},
        {"dataflow": {"one_of": []}},
        {"maximum_rows_per_command": {"max": True}},
        {"mvin_scale": {"eq": float("nan")}},
    ],
)
def test_invalid_execution_constraints_fail_closed(document, constraints):
    document["operations"]["movement_bitwise"]["signature"]["execution_constraints"] = constraints
    with pytest.raises(ValueError, match="execution_constraints"):
        validate_software_spec(document)


@pytest.mark.parametrize("obligations", [
    {},
    {"scale": {"status": "reviewed", "constraints": {"eq": 1.0}}},
    {"scale": {"status": "compiler_obligation", "constraints": {"approximately": 1.0}}},
    {"scale": {"status": "compiler_obligation", "constraints": {"eq": float("nan")}}},
    {"scale": {"status": "compiler_obligation", "constraints": {"eq": 1.0}, "extra": True}},
])
def test_invalid_emission_obligations_fail_closed(document, obligations):
    document["operations"]["movement_bitwise"]["emission_obligations"] = obligations
    with pytest.raises(ValueError, match="emission_obligations|execution_constraints"):
        validate_software_spec(document)


def test_approved_policies_have_scoped_machine_visible_authority(document, monkeypatch):
    from merlin.targetgen.operator_review import ACCEPT_ENV, PENDING
    from merlin.targetgen.software_spec import numerical_policy_review

    monkeypatch.setenv(ACCEPT_ENV, "0")
    spec = validate_software_spec(document)
    numeric = spec["numerical_semantics"]
    assert numerical_policy_review(spec)["review_status"] == "reviewed"
    assert numerical_policy_review(spec)["operator_accepted"]
    assert numeric["policy_reviews"]["calibration_domain"]["scope"] == "op_validation_only"
    assert numeric["policy_reviews"]["calibration_domain"]["composition_observers"] == "independent_scheme_policy"
    assert numeric["policy_reviews"]["readout_specials"]["residual_coefficient"] == "nonnegative"
    assert numeric["quantized_native_fallback"]["status"] == "reviewed"
    assert numeric["quantized_native_fallback"]["operand_storage"] == ["f16", "bf16", "f32"]
    assert not {row["id"] for row in spec["operations"] if row.get("status") == PENDING}


@pytest.mark.parametrize("field,value", [("scope", "all_calibration"), ("composition_observers", "fixed_range")])
def test_op_validation_approval_cannot_override_composition_observers(document, field, value):
    document["numerical_semantics"]["policy_reviews"]["calibration_domain"][field] = value
    with pytest.raises(ValueError, match="independent composition observer"):
        validate_software_spec(document)
