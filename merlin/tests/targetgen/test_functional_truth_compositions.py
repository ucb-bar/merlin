"""Family-chain truth crosses schemes and refuses unknown applicable constraints."""

from fractions import Fraction

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import functional_truth_compositions as truth
from merlin.targetgen.quantization_spec import resolve_quantization_schemes
from merlin.targetgen.software_spec import load_software_spec


@pytest.fixture
def spec():
    return load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml")


@pytest.fixture
def facts():
    return dict(
        output_path=dict(
            schema="output_path_v1",
            accumulator={"dtype": "i32"},
            accumulator_stages=[{"kind": "preload", "dtype": "i32"}],
            readout=[
                {"kind": "activation", "function": "relu", "domain": "accumulator"},
                {"kind": "convert", "to": "f32", "rounding": "half_even"},
                {
                    "kind": "scale",
                    "dtype": "f32",
                    "rounding": "half_even",
                    "granularity": "per_command",
                    "columns_per_scale": 3,
                },
                {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
                {"kind": "pool", "reductions": ["max"], "padding": "unknown"},
            ],
            exits={"accumulator": True, "requant": True},
        ),
        dataflows=["WS", "OS"],
        producer_placement={family: ["accelerator"] for family in ("contraction", "movement", "operand_sum")},
        operand_signed_bits=8,
        mesh_signed_bits=12,
        pool_window_within_register_fields={"size_bits": 2, "stride_bits": 2},
        max_requires_store_path_pooling=True,
        k_extension_within={"max_k": 64, "partial_sum_bits": 20},
        shared_input_resident={"capacity_bytes": 4096},
        weight_panel_width=3,
        dtype="int8",
        max_k_extension=64,
    )


@pytest.fixture
def case():
    return dict(
        producer="contraction",
        readout_stages=["activation", "convert", "scale", "requant"],
        accumulator_stages=["preload"],
        exit="requant",
        dataflow="WS",
        output_columns=3,
        multipliers=[Fraction(1, 16)],
        accumulator_range=[-1000, 1000],
        shape_class="partial",
        members=["mac", "bias", "relu", "requant"],
    )


def selected(spec, facts, chain="contraction_readout", scheme="q_pow2_tensor"):
    chains = spec["compositions"]["chains"]
    schemes, _ = resolve_quantization_schemes(spec, facts)
    return next(c for c in chains if c["id"] == chain), next(s for s in schemes["schemes"] if s["id"] == scheme)


def expect(spec, facts, case, chain="contraction_readout", scheme="q_pow2_tensor"):
    declaration, selected_scheme = selected(spec, facts, chain, scheme)
    return truth.expectation(declaration, selected_scheme, case, facts=facts)


@pytest.mark.parametrize(
    "scheme", ["q_pow2_tensor", "q_f32_tensor", "q_f32_column_block", "q_f32_per_operand_sum", "q_f32_packed_panel"]
)
def test_bias_relu_requant_crosses_every_declared_scheme(spec, facts, case, scheme):
    commitment = expect(spec, facts, case, scheme=scheme)
    assert commitment["lane"] == "fused_device", commitment
    assert commitment["scheme_id"] == scheme
    assert commitment["required_modes"] == ["readout_multipliers", "relu", "accumulator_preload"]


def test_i32_exit_ignores_optional_pool_unknowns_and_applies_no_stages(spec, facts, case):
    case.update(exit="accumulator", readout_stages=[], members=["mac", "bias", "readout"])
    assert expect(spec, facts, case)["lane"] == "fused_device"
    assert expect(spec, facts, case)["required_modes"] == ["i32_exit", "accumulator_preload"]
    case["readout_stages"] = ["activation"]
    assert expect(spec, facts, case)["lane"] == "must_refuse"


def test_exact_readout_requires_proof_even_for_power_of_two(spec, facts, case):
    case["accumulator_range"] = [0, 2**25 + 1]
    assert expect(spec, facts, case)["lane"] == "must_refuse"
    assert expect(spec, facts, case, scheme="q_f32_tensor")["lane"] == "fused_device"


@pytest.mark.parametrize(
    "producer,chain", [("contraction", "contraction_readout"), ("movement", "window_reduction_readout")]
)
def test_max_pool_is_composition_only_and_padding_unknown_refuses(spec, facts, case, producer, chain):
    case.update(
        producer=producer,
        accumulator_stages=[],
        padding=[0, 0],
        pool_window=[2, 2],
        pool_stride=[2, 2],
        indices_live=False,
        scratchpad_source_pooling_fenced=True,
    )
    case["readout_stages"].append("pool")
    case["members"].append("max_values")
    assert expect(spec, facts, case, chain)["lane"] == "fused_device"
    case["padding"] = [1, 1]
    assert expect(spec, facts, case, chain)["lane"] == "must_refuse"
    case.update(padding=[0, 0], indices_live=True)
    assert expect(spec, facts, case, chain)["lane"] == "must_refuse"


@pytest.mark.parametrize("scales", [[1, 1], [Fraction(1, 2), Fraction(3, 10)]])
def test_per_operand_residual_is_offered_on_every_add(spec, facts, case, scales):
    case.update(
        producer="operand_sum",
        accumulator_stages=[],
        operand_multipliers=scales,
        operand_requantization="round_saturate_each",
        negates_load_minimum=False,
        members=["left_requant", "right_requant", "add", "requant"],
    )
    result = expect(spec, facts, case, "scaled_operand_sum", "q_f32_per_operand_sum")
    assert result["lane"] == "fused_device", result
    assert "scaled_accumulate_load" in result["required_modes"]
    case["operand_multipliers"] = [1, 2]
    assert expect(spec, facts, case, "scaled_operand_sum", "q_f32_per_operand_sum")["lane"] == "must_refuse"


def test_single_rounding_does_not_accept_lossy_loads(spec, facts, case):
    case.update(producer="operand_sum", accumulator_stages=[], operand_multipliers=[1, 1], negates_load_minimum=False)
    assert expect(spec, facts, case, "scaled_operand_sum")["lane"] == "fused_device"
    case["operand_multipliers"] = [Fraction(1, 2), Fraction(1, 2)]
    assert expect(spec, facts, case, "scaled_operand_sum")["lane"] == "must_refuse"


def test_window_mean_is_a_ones_contraction_with_scheme_readout(spec, facts, case):
    case.update(
        accumulator_stages=[],
        reduction_kind="sum",
        multipliers=[Fraction(1, 4)],
        members=["ones_contraction", "mean_requant"],
    )
    assert expect(spec, facts, case, "window_reduction_readout", "q_f32_tensor")["lane"] == "fused_device"


def test_os_uses_literal_weight_bound_with_adjacent_refusal_twin(spec, facts, case):
    rows = (1 << (facts["mesh_signed_bits"] - 1)) // (1 << (facts["operand_signed_bits"] - 1))
    case.update(dataflow="OS", constant_weights=[[1]] * rows, seed_range=[0, 0])
    assert expect(spec, facts, case)["lane"] == "fused_device"
    case["constant_weights"].append([1])
    assert expect(spec, facts, case)["lane"] == "must_refuse"
    case["constant_weights"] = None
    assert expect(spec, facts, case)["lane"] == "must_refuse"


@pytest.mark.parametrize("coefficient", [1, 129])
def test_integer_k_extension_is_required_and_noninteger_refuses(spec, facts, case, coefficient):
    case.update(
        accumulator_stages=[],
        coefficient=coefficient,
        source_k=3,
        residual_width=3,
        source_partial_bound=1000,
        residual_range=[-128, 127],
    )
    result = expect(spec, facts, case, "contraction_operand_extension")
    assert result["lane"] == "fused_device"
    assert "k_extension" in result["required_modes"]
    case["coefficient"] = Fraction(1, 2)
    assert expect(spec, facts, case, "contraction_operand_extension")["lane"] == "must_refuse"


def test_shared_input_keeps_independent_exits_and_derived_panel_width(spec, facts, case):
    case.update(
        accumulator_stages=[],
        immutable_shared_input=True,
        resident_bytes=100,
        siblings=[
            {"exit": "requant", "readout_independent": True},
            {"exit": "accumulator", "readout_independent": True},
        ],
    )
    chain, scheme = selected(spec, facts, "shared_input_contractions", "q_f32_packed_panel")
    assert scheme["weight_layout"] == {
        "kind": "packed_panel",
        "panel_width": 3,
        "logical_order": ["N/panel_width", "K", "panel_width"],
    }
    assert truth.expectation(chain, scheme, case, facts=facts)["lane"] == "fused_device"
    case["resident_bytes"] = facts["shared_input_resident"]["capacity_bytes"] + 1
    assert truth.expectation(chain, scheme, case, facts=facts)["lane"] == "must_refuse"


def test_renaming_ids_and_provenance_cannot_select_a_lane(spec, facts, case):
    chain, scheme = selected(spec, facts)
    original = truth.expectation(chain, scheme, case, facts=facts)
    chain["id"], scheme["id"] = "unrelated", "another_axis_point"
    case.update(model="unrelated", provenance="unrelated")
    assert truth.expectation(chain, scheme, case, facts=facts)["lane"] == original["lane"]


def test_unknown_scale_geometry_is_never_a_default(spec, facts, case):
    facts["output_path"]["readout"][2]["columns_per_scale"] = None
    assert expect(spec, facts, case, scheme="q_f32_column_block")["lane"] == "must_refuse"


def test_placement_requires_all_members_same_region_and_independent_audit(spec, facts, case):
    result = expect(spec, facts, case)
    placement = {member: "kernel" for member in result["members"]}
    audit = dict(
        kernel="kernel",
        modes={mode: True for mode in result["required_modes"]},
        outside_kernel_instructions=0,
        host_intermediate_arithmetic=0,
    )
    truth.require_placement(result, member_kernels=placement, instruction_audit=audit)
    for wrong in ({**placement, "relu": "host"}, {k: v for k, v in placement.items() if k != "relu"}):
        with pytest.raises(ValueError, match="every composition member"):
            truth.require_placement(result, member_kernels=wrong, instruction_audit=audit)
    for wrong in (
        {},
        {**audit, "modes": {}},
        {**audit, "outside_kernel_instructions": 1},
        {**audit, "host_intermediate_arithmetic": 1},
        {**audit, "kernel": "another_kernel"},
        {**audit, "outside_kernel_instructions": False},
    ):
        with pytest.raises(ValueError, match="audit|escaped"):
            truth.require_placement(result, member_kernels=placement, instruction_audit=wrong)


def test_refusal_is_an_outcome_not_permission_to_split_to_host(spec, facts, case):
    case.update(exit="accumulator", readout_stages=["activation"])
    result = expect(spec, facts, case)
    truth.require_placement(result, member_kernels={}, instruction_audit={"outcome": "refuse"})
    with pytest.raises(ValueError, match="requires refusal"):
        truth.require_placement(result, member_kernels={"mac": "kernel"}, instruction_audit={"outcome": "pass"})


def test_existing_truth_entry_point_dispatches_only_explicit_composition_cases(spec, facts, case):
    from merlin.targetgen.functional_truth import expectation, require_placement

    chain, scheme = selected(spec, facts)
    raw = {
        "semantic": {
            "generalization_axis": "composition",
            "composition_chain": chain,
            "quantization_scheme": scheme,
            "composition_instance": case,
            "composition_facts": facts,
        }
    }
    result = expectation(raw)
    assert result == truth.expectation(chain, scheme, case, facts=facts)
    with pytest.raises(ValueError, match="member placement and instruction audit"):
        require_placement(result, device=True)
    with pytest.raises(KeyError):
        expectation({"semantic": {"generalization_axis": "composition"}})


def test_multiblock_tail_requires_its_own_formed_multiplier(spec, facts, case):
    width = facts["output_path"]["readout"][2]["columns_per_scale"]
    case.update(
        output_columns=2 * width + 1,
        shape_class="multi_block",
        multipliers=[Fraction(1, 14), Fraction(3, 10), Fraction(1, 166)],
    )
    result = expect(spec, facts, case, scheme="q_f32_column_block")
    assert result["lane"] == "fused_device"
    assert len(set(result["instruction_expectation"]["readout_multiplier_bits"])) == 3
    case["multipliers"].pop()
    assert expect(spec, facts, case, scheme="q_f32_column_block")["lane"] == "must_refuse"


def test_unknown_producer_facts_cannot_be_supplied_by_a_software_request(spec, facts, case):
    facts["producer_placement"] = {}
    assert expect(spec, facts, case)["lane"] == "must_refuse"


@pytest.mark.parametrize("missing", ["relu", "accumulator_preload"])
def test_source_activation_and_bias_cannot_be_omitted_from_audit(spec, facts, case, missing):
    commitment = expect(spec, facts, case)
    audit = dict(
        kernel="kernel",
        modes={**dict.fromkeys(commitment["required_modes"], True), missing: False},
        outside_kernel_instructions=0,
        host_intermediate_arithmetic=0,
    )
    with pytest.raises(ValueError, match="missing or failed"):
        truth.require_placement(
            commitment, member_kernels=dict.fromkeys(commitment["members"], "kernel"), instruction_audit=audit
        )


def test_zero_padding_is_not_neutral_for_negative_pool_producer(spec, facts, case):
    # A window containing -2 and padding has source max -2, but device max 0.
    facts["output_path"]["readout"][-1]["padding"] = "zero"
    case.update(
        readout_stages=["convert", "scale", "requant", "pool"],
        accumulator_stages=[],
        padding=[1, 1],
        pool_window=[2, 2],
        pool_stride=[2, 2],
        indices_live=False,
        producer_lower_bound=-2,
        multipliers=[Fraction(1)],
    )
    assert expect(spec, facts, case)["lane"] == "must_refuse"


@pytest.mark.parametrize("fenced", [None, False, True])
def test_movement_pooling_requires_declared_scratchpad_fence(spec, facts, case, fenced):
    case.update(
        producer="movement",
        accumulator_stages=[],
        padding=[0, 0],
        pool_window=[2, 2],
        pool_stride=[2, 2],
        indices_live=False,
        scratchpad_source_pooling_fenced=fenced,
    )
    case["readout_stages"].append("pool")
    expected = "fused_device" if fenced is True else "must_refuse"
    assert expect(spec, facts, case, "window_reduction_readout")["lane"] == expected


def test_scratchpad_pool_fence_is_an_independent_instruction_obligation(spec, facts, case):
    case.update(
        producer="movement",
        accumulator_stages=[],
        padding=[0, 0],
        pool_window=[2, 2],
        pool_stride=[2, 2],
        indices_live=False,
        scratchpad_source_pooling_fenced=True,
    )
    case["readout_stages"].append("pool")
    commitment = expect(spec, facts, case, "window_reduction_readout")
    audit = dict(
        kernel="kernel",
        modes=dict.fromkeys(commitment["required_modes"], True),
        outside_kernel_instructions=0,
        host_intermediate_arithmetic=0,
    )
    audit["modes"]["scratchpad_source_pooling_fenced"] = False
    with pytest.raises(ValueError, match="missing or failed"):
        truth.require_placement(
            commitment, member_kernels=dict.fromkeys(commitment["members"], "kernel"), instruction_audit=audit
        )


@pytest.mark.parametrize("activation,lower_bound", [(True, None), (False, 0), (False, 1), (False, None)])
def test_padded_pool_requires_source_bound_or_relu_proof(spec, facts, case, activation, lower_bound):
    facts["output_path"]["readout"][-1]["padding"] = "zero"
    case.update(
        accumulator_stages=[],
        padding=[1, 1],
        pool_window=[2, 2],
        pool_stride=[2, 2],
        indices_live=False,
        producer_lower_bound=lower_bound,
    )
    case["readout_stages"].append("pool")
    if not activation:
        case["readout_stages"].remove("activation")
    expected = "fused_device" if activation or lower_bound is not None else "must_refuse"
    assert expect(spec, facts, case)["lane"] == expected
