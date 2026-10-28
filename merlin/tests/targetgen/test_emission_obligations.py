"""Source eligibility and retired execution duties have separate evidence owners."""

import copy

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import composition_audit as CA
from merlin.targetgen import native_instruction_audit as NA
from merlin.targetgen.admission_conditions import INSTRUCTION_CHECKS, validate_emission_obligations
from merlin.targetgen.software_spec import load_software_spec, validate_software_spec

SPEC = repo_root() / "examples/gemmini/target/software-spec.yaml"


def row(kind, **fields):
    return dict(kernel="retired_kernel", **{"class": kind}, decoded=fields)


def retired():
    return [
        row("CONFIG_LD", state_id=6, scale=1.0, stride=16),
        row("MVIN", dram=dict(kind="const", raw=4096), rows=3, cols=16),
        row("CONFIG_EX", dataflow=4),
        row("COMPUTE_PRELOADED", a_cols=16),
        row(
            "CONFIG_ST",
            acc_scale=1.0,
            acc_act=0,
            out_stride_bytes=64,
            pool_stride=0,
            pool_size=0,
            pool_out_dim=4,
            porows=4,
            pocols=4,
            orows=4,
            ocols=4,
        ),
        row("MVOUT", dram=dict(kind="const", raw=8192), rows=3, cols=16, readout="i32"),
    ]


# Every declared audit dispatch has an actual stream mutation that must fail.
MUTATIONS = {
    "unit_mvin_scale": (0, "scale", 2.0, dict(eq=1.0)),
    "nonzero_dram": (1, "dram", dict(kind="const", raw=0), dict(min=1)),
    "pool_off": (4, "pool_size", 1, dict(eq=True)),
    "transfer_rows": (1, "rows", 17, dict(min=1, max=16)),
    "movement_bytes": (1, "cols", 65, dict(min=1, max=64)),
    "nonnegative_stride": (0, "stride", -1, dict(min=0)),
    "weight_stationary": (2, "dataflow", 9, dict(eq="WS")),
    "full_i32": (5, "readout", "i8", dict(eq="full_i32")),
    "raw_i32_exit": (4, "acc_scale", 2.0, dict(eq=True)),
    "max_columns": (5, "cols", 17, dict(min=1, max=16)),
    "pool_window": (4, "pool_size", 4, dict(eq=True)),
    "mvout_scale": (4, "acc_scale", 2.0, dict(eq=False)),
    "mvout_relu": (4, "acc_act", 1, dict(eq=False)),
    "mvout_pool_stride": (4, "pool_stride", 1, dict(eq=0)),
    "mvout_pool_size": (4, "pool_size", 1, dict(eq=0)),
    "primitive_k_bound": (3, "a_cols", 32, dict(min=1, max=524287)),
}


@pytest.fixture
def hardware(monkeypatch):
    from merlin.targetgen import trace_check
    from merlin.targetgen.rtl import facts

    monkeypatch.setattr(CA, "_selected_hardware", lambda *_: dict(load_states={"MVIN": 6}))
    monkeypatch.setattr(trace_check, "_bound_trace_view", lambda *_: None)
    monkeypatch.setattr(
        facts,
        "load_facts",
        lambda _: dict(
            facts=dict(
                timing_structure=dict(
                    dataflow=dict(
                        status="derived", support="both", modes={"4": "weight_stationary", "9": "output_stationary"}
                    )
                )
            )
        ),
    )


@pytest.mark.parametrize("check", sorted(MUTATIONS))
def test_mutated_retired_stream_fails_native_audit(hardware, check):
    assert set(MUTATIONS) == INSTRUCTION_CHECKS
    index, field, value, constraints = MUTATIONS[check]
    instructions = retired()
    if check == "pool_window":
        instructions[4]["decoded"].update(pool_stride=2, pool_size=3)
        instructions[5]["decoded"]["readout"] = "i8"
    obligations = dict(duty=dict(status="instruction_audit", check=check, constraints=constraints))
    spec = dict(operations=[dict(id="selected", emission_obligations=obligations)])
    routed = [dict(symbol="retired_kernel", software_declaration="selected")]
    report = NA.audit_declared_emission(dict(instructions=instructions), routed, spec, target="selected")
    assert report["retired_kernel"]["obligations"]["duty"]["observation_status"] == "pass"
    instructions[index]["decoded"][field] = value
    with pytest.raises(ValueError, match="emission obligation duty failed"):
        NA.audit_declared_emission(dict(instructions=instructions), routed, spec, target="selected")


@pytest.mark.parametrize("field,value", [("pool_stride", 1), ("pool_size", 1), ("acc_act", 1)])
def test_raw_i32_exit_refuses_all_readout_stages(hardware, field, value):
    instructions = retired()
    instructions[4]["decoded"][field] = value
    with pytest.raises(ValueError, match="raw_i32_exit"):
        CA.check_emission_obligations(
            instructions,
            dict(raw=dict(status="instruction_audit", check="raw_i32_exit", constraints=dict(eq=True))),
            target="selected",
        )


def test_effective_load_port_and_missing_observation_fail_closed(hardware):
    obligations = dict(scale=dict(status="instruction_audit", check="unit_mvin_scale", constraints=dict(eq=1.0)))
    instructions = retired()
    instructions.insert(1, row("CONFIG_LD", state_id=3, scale=1.0, stride=16))
    instructions[0]["decoded"]["scale"] = 2.0
    with pytest.raises(ValueError, match="unit_mvin_scale"):
        CA.check_emission_obligations(instructions, obligations, target="selected")
    instructions = retired()[1:]
    with pytest.raises(ValueError, match="UNKNOWN"):
        CA.check_emission_obligations(instructions, obligations, target="selected")


def test_all_rows_validate_and_unwired_duties_are_recorded(hardware):
    spec = load_software_spec(SPEC)
    host = yaml.safe_load((SPEC.parents[1] / "platform/host-capabilities-core-aten.yaml").read_text())
    for declaration in [*spec["operations"], *host["operations"], host["native_ieee_special_values"]]:
        obligations = declaration.get("emission_obligations")
        if not obligations:
            continue
        validate_emission_obligations(obligations)
        unwired = {k: v for k, v in obligations.items() if v["status"] == "compiler_obligation"}
        if unwired:
            assert CA.check_emission_obligations([], unwired, target="selected") == unwired
        if declaration.get("id", "").startswith("host_"):
            assert set(v["status"] for v in obligations.values()) == {"compiler_obligation"}


@pytest.mark.parametrize(
    "bad",
    [
        dict(status="instruction_audit", check="invented", constraints=dict(eq=True)),
        dict(status="instruction_audit", check="full_i32", condition=dict(path="x", eq=True)),
        dict(status="compiler_obligation", condition=dict(any=[])),
        dict(status="compiler_obligation", condition=dict(path="execution.x", max=float("inf"))),
    ],
)
def test_malformed_obligation_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_emission_obligations(dict(duty=bad))


def test_source_admission_rejects_execution_preconditions():
    document = yaml.safe_load(SPEC.read_text())
    document["operations"]["contraction"]["admission_conditions"] = dict(path="execution.dataflow", eq="WS")
    with pytest.raises(ValueError, match="emission_obligations"):
        validate_software_spec(document)


def test_provider_certificate_bounds_are_enforced_without_source_execution():
    from merlin.targetgen.admission_conditions import enforce_provider_emission_obligations

    spec = load_software_spec(SPEC)
    proof = dict(
        status="proven",
        bindings=[dict(symbol="limb", op="limb_bilinear", numeric=dict(family="F2"))],
        kernels=[
            dict(
                symbol="limb",
                schedule=dict(
                    bound_proof=dict(max_mesh_partial_sum=262144, accumulator_min=-262144, accumulator_max=262144)
                ),
            )
        ],
    )
    result = enforce_provider_emission_obligations(copy.deepcopy(proof), spec)
    duties = result["kernels"][0]["emission_obligations"]["component_bilinear"]
    assert duties["partial_sum_bound"]["status"] == "compiler_obligation"
    assert duties["partial_sum_bound"]["observation_status"] == "unknown"
    proof["kernels"][0]["schedule"]["bound_proof"]["max_mesh_partial_sum"] = 524288
    with pytest.raises(ValueError, match="provider emission obligation"):
        enforce_provider_emission_obligations(proof, spec)


def test_native_audit_uses_normalized_original_decoder_projection(hardware, monkeypatch):
    from merlin.targetgen import trace_check

    instructions = retired()
    monkeypatch.setattr(trace_check, "_bound_trace_view", lambda *_: (dict(instructions=instructions), {}, [], True))
    spec = dict(
        operations=[
            dict(
                id="selected",
                emission_obligations=dict(
                    raw=dict(status="instruction_audit", check="full_i32", constraints=dict(eq="full_i32"))
                ),
            )
        ]
    )
    routed = [dict(symbol="retired_kernel", software_declaration="selected")]
    assert NA.audit_declared_emission(dict(instructions=[]), routed, spec, target="selected")
    instructions[-1]["decoded"]["readout"] = "i8"
    with pytest.raises(ValueError, match="full_i32"):
        NA.audit_declared_emission(dict(instructions=[]), routed, spec, target="selected")


def test_observed_native_audit_fails_on_retired_obligation_violation(hardware, monkeypatch):
    instructions = retired()
    monkeypatch.setattr(NA, "_function_ranges", lambda *_: [(0, 100, "retired_kernel")])
    monkeypatch.setattr(NA, "_retirement_counts", lambda *_: {})
    monkeypatch.setattr(NA, "_instruction_report", lambda *_: dict(status="pass", pass_=True))
    monkeypatch.setattr(NA, "decode_retirement_trace", lambda **_: dict(instructions=instructions))
    spec = dict(
        target="synthetic",
        operations=[
            dict(
                id="selected",
                emission_obligations=dict(
                    raw=dict(status="instruction_audit", check="full_i32", constraints=dict(eq="full_i32"))
                ),
            )
        ],
    )
    facts = dict(
        target="synthetic",
        facts=dict(interfaces=[dict(name="funct_decode_table", custom_opcode=43, names={"5": "WORK"})]),
    )
    arguments = dict(
        elf="elf",
        trace="trace",
        facts=facts,
        software_spec=spec,
        routed=[dict(symbol="retired_kernel", software_declaration="selected")],
    )
    assert NA.audit_case_instructions(**arguments)["pass_"] is True
    instructions[-1]["decoded"]["readout"] = "i8"
    result = NA.audit_case_instructions(**arguments)
    assert result["pass_"] is False
    assert "full_i32" in result["issues"][0]


def test_bilinear_form_selects_the_admitted_duties(hardware):
    from merlin.targetgen.admission_conditions import enforce_provider_emission_obligations
    from merlin.targetgen.software_spec import admit_operation

    spec = load_software_spec(SPEC)
    decision = admit_operation(
        spec,
        "contraction",
        dict(
            family="contraction",
            numerical_family="F2",
            operand_dtype="i8",
            accum_dtype="i32",
            readout_dtype="i32",
            rank=2,
            layout="row_major_contiguous",
            aliasing="disjoint_inputs_outputs",
        ),
        "accelerator",
    )
    assert decision["status"] == "admitted"
    binding = dict(symbol="retired_kernel", op="matmul", numeric=dict(family="F2", form="bilinear"))
    instructions = retired()
    result = NA.audit_declared_emission(dict(instructions=instructions), [binding], spec, target="selected")
    instructions[-1]["decoded"]["readout"] = "i8"
    with pytest.raises(ValueError, match="full_i32"):
        NA.audit_declared_emission(dict(instructions=instructions), [binding], spec, target="selected")
    assert result["retired_kernel"]["declaration"] == decision["declaration"]
    proof = enforce_provider_emission_obligations(
        dict(
            status="proven",
            bindings=[binding],
            kernels=[dict(symbol="retired_kernel")],
        ),
        spec,
    )
    assert set(proof["kernels"][0]["emission_obligations"]) == {decision["declaration"]}


@pytest.mark.parametrize("op", ["aten.mm.default", "aten.bmm.default", "aten._int_mm.default", "aten.addmm.default"])
def test_standard_contraction_bindings_select_their_declared_duties(hardware, op):
    from merlin.targetgen.admission_conditions import enforce_provider_emission_obligations

    spec = load_software_spec(SPEC)
    binding = dict(symbol="retired_kernel", op=op)
    proof = enforce_provider_emission_obligations(
        dict(
            status="proven",
            bindings=[binding],
            kernels=[dict(symbol="retired_kernel")],
        ),
        spec,
    )
    assert set(proof["kernels"][0]["emission_obligations"]) == {"contraction"}
    result = NA.audit_declared_emission(dict(instructions=retired()), [binding], spec, target="selected")
    assert result["retired_kernel"]["declaration"] == "contraction"


@pytest.mark.parametrize("wrapper", ["outer", "inner", "missing"])
def test_environment_spec_preserves_target_and_fails_closed(hardware, monkeypatch, tmp_path, wrapper):
    instructions = retired()
    monkeypatch.setattr(NA, "_function_ranges", lambda *_: [(0, 100, "retired_kernel")])
    monkeypatch.setattr(NA, "_retirement_counts", lambda *_: {})
    monkeypatch.setattr(NA, "_instruction_report", lambda *_: dict(status="pass", pass_=True))
    monkeypatch.setattr(NA, "decode_retirement_trace", lambda **_: dict(instructions=instructions))
    monkeypatch.setenv("MERLIN_CORE_ATEN_SOFTWARE_SPEC", str(SPEC))
    facts = dict(interfaces=[dict(name="funct_decode_table", custom_opcode=43, names={"5": "WORK"})])
    if wrapper == "inner":
        facts["target"] = load_software_spec(SPEC)["target"]
    facts = dict(facts=facts)
    if wrapper == "outer":
        facts["target"] = load_software_spec(SPEC)["target"]
    arguments = dict(
        elf="elf",
        trace="trace",
        facts=facts,
        routed=[dict(symbol="retired_kernel", software_declaration="component_bilinear")],
    )
    assert NA.audit_case_instructions(**arguments)["pass_"] is (wrapper != "missing")
    instructions[-1]["decoded"]["readout"] = "i8"
    result = NA.audit_case_instructions(**arguments)
    assert result["pass_"] is False
    assert "UNKNOWN" in result["issues"][0] or "full_i32" in result["issues"][0]


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("CONFIG_LD", dict(state_id=6, stride=16)),
        ("CONFIG_ST", dict(out_stride_bytes=64)),
    ],
)
def test_unused_stride_setup_has_no_retired_observation(hardware, kind, fields):
    with pytest.raises(ValueError, match="no retired observation"):
        CA.check_emission_obligations(
            [row(kind, **fields)],
            dict(stride=dict(status="instruction_audit", check="nonnegative_stride", constraints=dict(min=0))),
            target="selected",
        )


def test_stride_uses_only_the_consumed_load_port_and_store_configuration(hardware):
    instructions = retired()
    instructions.insert(1, row("CONFIG_LD", state_id=3, stride=-1))
    instructions.append(row("CONFIG_ST", out_stride_bytes=-1))
    result = CA.check_emission_obligations(
        instructions,
        dict(stride=dict(status="instruction_audit", check="nonnegative_stride", constraints=dict(min=0))),
        target="selected",
    )
    assert result["stride"]["observed_uses"] == 2


# A provider scaled_operand_sum binding: no numeric family and no declaration
# selector in its op, only the provider's numeric route.
SCALED_OPERAND_SUM = dict(
    symbol="retired_kernel",
    op="scaled_operand_sum",
    aten="",
    region="",
    numeric=dict(
        form="staged_sum",
        operand_sum_rounding="single_rounding",
        reference="source_f32_operations",
        scaled_operand_sum=True,
    ),
)


def scaled_loads(scale=0.5, readout="i8"):
    return [
        row("CONFIG_LD", state_id=6, scale=scale, stride=16),
        row("MVIN", dram=dict(kind="const", raw=4096), rows=16, cols=16),
        row("MVIN", dram=dict(kind="const", raw=4352), rows=16, cols=16, accumulate=True),
        row(
            "CONFIG_ST",
            acc_scale=0.25,
            acc_act=0,
            out_stride_bytes=16,
            pool_stride=0,
            pool_size=0,
            pool_out_dim=0,
            porows=0,
            pocols=0,
            orows=0,
            ocols=0,
        ),
        row("MVOUT", dram=dict(kind="const", raw=8192), rows=16, cols=16, readout=readout),
    ]


def test_scaled_operand_sum_binding_selects_its_declaration(hardware):
    from merlin.targetgen.admission_conditions import binding_declaration, enforce_provider_emission_obligations

    spec = load_software_spec(SPEC)
    binding = copy.deepcopy(SCALED_OPERAND_SUM)
    assert binding_declaration(binding) == "elementwise_map_after_operand_sum"
    rows = {r["id"]: r for r in spec["operations"]}
    assert rows["elementwise_map_after_operand_sum"]["status"] == "reviewed"
    report = NA.audit_declared_emission(dict(instructions=scaled_loads()), [binding], spec, target="selected")
    duties = report["retired_kernel"]["obligations"]
    assert report["retired_kernel"]["declaration"] == "elementwise_map_after_operand_sum"
    assert {d["observation_status"] for d in duties.values()} == {"pass"}
    proof = enforce_provider_emission_obligations(
        dict(
            status="proven",
            bindings=[binding],
            kernels=[dict(symbol="retired_kernel")],
        ),
        spec,
    )
    assert set(proof["kernels"][0]["emission_obligations"]) == {"elementwise_map_after_operand_sum"}


def test_single_rounding_realization_and_phase0_source_witness_share_one_row_and_contract(hardware):
    """The realization binds the row the Phase 0 witness (with its readout gain) is judged by."""
    from merlin.targetgen.admission_conditions import (
        OPERAND_SUM_CONTRACT_ROUNDING,
        binding_declaration,
        declaration_numerics,
    )
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    spec = load_software_spec(SPEC)
    rows = {r["id"]: r for r in spec["operations"]}
    binding = copy.deepcopy(SCALED_OPERAND_SUM)
    assert binding["numeric"]["operand_sum_rounding"] == "single_rounding"
    identity = binding_declaration(binding)
    assert identity == "elementwise_map_after_operand_sum"
    declaration_numerics(binding, identity, rows[identity])  # accepted: no refusal
    contract = rows[identity]["numerical_contract"]
    assert OPERAND_SUM_CONTRACT_ROUNDING[contract] == "single_rounding"
    facet = dict(
        rounding="half_even",
        clamp=[-128, 127],
        accumulator_dtype="i32",
        operand_sum=dict(
            operands=2, operand_dtype="i8", operand_rounding="half_even", operand_saturates=True, scale_dtype="f32"
        ),
        readouts=[dict(selector="i8", applies=["acc_scale", "relu"])],
        scale=dict(dtype="f32", granularities=["tensor"]),
    )
    witness = audit_i8_operand_sum(
        lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=True, facet=facet, readout_scale=0.16666667
    )
    assert witness["status"] == "within_bound" and "readout_gain_f32" in witness["parameters"]
    # the validator's default: a witness without a numerical_contract key is the row's v1 contract
    assert witness.get("numerical_contract", "operand_sum_exhaustive_i8_v1") == contract
    # the per-operand realization is a different row and contract
    other = copy.deepcopy(SCALED_OPERAND_SUM)
    other["numeric"]["operand_sum_rounding"] = "per_operand"
    assert binding_declaration(other) == "elementwise_map_after_per_operand_sum"
    assert rows[binding_declaration(other)]["numerical_contract"] != contract
    with pytest.raises(ValueError, match="UNKNOWN"):
        declaration_numerics(other, identity, rows[identity])


@pytest.mark.parametrize("scale", [1.0, 2.0**-7])
def test_scaled_operand_sum_admits_its_operand_multiplier_domain(hardware, scale):
    spec = load_software_spec(SPEC)
    NA.audit_declared_emission(dict(instructions=scaled_loads(scale)), [SCALED_OPERAND_SUM], spec, target="selected")


@pytest.mark.parametrize(
    "index,field,value,duty",
    [
        (0, "scale", 0.0, "load_scale"),
        (0, "scale", 1.5, "load_scale"),
        (0, "scale", -0.5, "load_scale"),
        (0, "stride", -16, "minimum_stride"),
        (3, "acc_act", 1, "raw_i32_exit"),
        (4, "cols", 17, "maximum_mvout_columns"),
    ],
)
def test_scaled_operand_sum_duties_refuse_mutated_streams(hardware, index, field, value, duty):
    spec = load_software_spec(SPEC)
    instructions = scaled_loads()
    if duty == "raw_i32_exit":
        instructions[4]["decoded"]["readout"] = "i32"
    instructions[index]["decoded"][field] = value
    with pytest.raises(ValueError, match=f"emission obligation {duty} failed"):
        NA.audit_declared_emission(dict(instructions=instructions), [SCALED_OPERAND_SUM], spec, target="selected")


def test_scaled_operand_sum_declaration_needs_the_provider_numeric_route():
    from merlin.targetgen.admission_conditions import binding_declaration

    for numeric in [dict(SCALED_OPERAND_SUM["numeric"], scaled_operand_sum=False), {}]:
        with pytest.raises(ValueError, match="lacks a software declaration"):
            binding_declaration(dict(SCALED_OPERAND_SUM, numeric=numeric))
    # An explicit declaration still wins over the numeric route.
    assert binding_declaration(dict(SCALED_OPERAND_SUM, software_declaration="other")) == "other"


def test_scaled_operand_sum_flag_never_moves_a_contraction_off_its_row():
    from merlin.targetgen.admission_conditions import binding_declaration

    for op in ("aten.mm.default", "aten.addmm.default", "aten.convolution.default", "contraction"):
        assert binding_declaration(dict(SCALED_OPERAND_SUM, op=op)) == "contraction"


@pytest.mark.parametrize("rounding", [None, "other"])
def test_scaled_operand_sum_refuses_rounding_outside_the_row_contract(hardware, rounding):
    from merlin.targetgen.admission_conditions import enforce_provider_emission_obligations

    spec = load_software_spec(SPEC)
    binding = copy.deepcopy(SCALED_OPERAND_SUM)
    binding["numeric"]["operand_sum_rounding"] = rounding
    with pytest.raises(ValueError, match="not screened by declaration elementwise_map_after_operand_sum"):
        NA.audit_declared_emission(dict(instructions=scaled_loads()), [binding], spec, target="selected")
    with pytest.raises(ValueError, match="not screened by declaration"):
        enforce_provider_emission_obligations(
            dict(
                status="proven",
                bindings=[binding],
                kernels=[dict(symbol="retired_kernel")],
            ),
            spec,
        )
    # A contraction carrying the flag gets the contraction row, which screens no operand sum.
    contraction = dict(copy.deepcopy(SCALED_OPERAND_SUM), op="aten.mm.default")
    with pytest.raises(ValueError, match="not screened by declaration contraction"):
        NA.audit_declared_emission(dict(instructions=scaled_loads()), [contraction], spec, target="selected")


@pytest.mark.parametrize("accepted", ["0", "1"])
def test_reviewed_per_operand_row_audits_independently_of_review_acceptance(hardware, monkeypatch, accepted):
    from merlin.targetgen.admission_conditions import binding_declaration, enforce_provider_emission_obligations
    from merlin.targetgen.operator_review import ACCEPT_ENV

    monkeypatch.setenv(ACCEPT_ENV, accepted)
    spec = load_software_spec(SPEC)
    binding = copy.deepcopy(SCALED_OPERAND_SUM)
    binding["numeric"]["operand_sum_rounding"] = "per_operand"
    identity = "elementwise_map_after_per_operand_sum"
    assert binding_declaration(binding) == identity
    selected = next(r for r in spec["operations"] if r["id"] == identity)
    assert selected["status"] == "reviewed"
    proof = dict(status="proven", bindings=[binding], kernels=[dict(symbol="retired_kernel")])
    report = NA.audit_declared_emission(dict(instructions=scaled_loads()), [binding], spec, target="selected")
    assert report["retired_kernel"]["review_status"] == "reviewed"
    checked = enforce_provider_emission_obligations(proof, spec)
    assert set(checked["kernels"][0]["emission_obligations"]) == {identity}
    # An explicit attempt to reuse the old single-rounding contract still fails.
    binding["software_declaration"] = "elementwise_map_after_operand_sum"
    with pytest.raises(ValueError, match="not screened by declaration"):
        NA.audit_declared_emission(dict(instructions=scaled_loads()), [binding], spec, target="selected")


def test_per_operand_contract_refuses_pooled_streams(hardware):
    binding = copy.deepcopy(SCALED_OPERAND_SUM)
    binding["numeric"]["operand_sum_rounding"] = "per_operand"
    instructions = scaled_loads()
    instructions[3]["decoded"].update(pool_stride=2, pool_size=2)
    with pytest.raises(ValueError, match="emission obligation pooling failed"):
        NA.audit_declared_emission(
            dict(instructions=instructions), [binding], load_software_spec(SPEC), target="selected"
        )
