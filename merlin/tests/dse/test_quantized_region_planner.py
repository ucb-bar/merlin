"""Target-neutral accuracy admission, global quant domains, and agent edit guidance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merlin.perf.agent_guidance import (
    guidance_for_quantized_region_plan,
    inspect_compiler_package,
)
from merlin.perf.quantization_contract import (
    POLICY_SCHEMA,
    AccuracyPolicy,
    ExactEpilogueProof,
    QuantParameter,
    QuantizedEpilogueCandidate,
    QuantizedEpilogueCapability,
    QuantizedEpilogueContract,
    admit_epilogue,
    evaluate_candidate,
    parse_accuracy_policy,
)
from merlin.perf.quantization_planner import (
    EpilogueSite,
    QuantizationDomain,
    QuantizedProgramEdge,
    QuantizedProgramNode,
    QuantizedRegionProblem,
    ResidualDomainProblem,
    ResidualDomainSite,
    plan_quantized_regions,
    plan_residual_domains,
)


SOURCE_SHA = "1" * 64
EVIDENCE_SHA = "2" * 64
DATASET_SHA = "3" * 64
SAMPLING_SHA = "4" * 64
SELECTION_SHA = "5" * 64
REFERENCE_SHA = "6" * 64
DOMAIN_SHA = "7" * 64


def _parameter(granularity: str, axis: int | None, dtype: str, binding: str) -> QuantParameter:
    return QuantParameter(granularity, axis, dtype, binding)


def _contract(*, channels: int = 2) -> QuantizedEpilogueContract:
    return QuantizedEpilogueContract(
        accumulator_dtype="i32",
        output_dtype="i8",
        channel_axis=1,
        channel_count=channels,
        activation_scale=_parameter("per_tensor", None, "f32", "activation_scale"),
        weight_scale=_parameter("per_axis", 1, "f32", "weight_scale"),
        output_scale_reciprocal=_parameter(
            "per_tensor", None, "f32", "output_scale_reciprocal"
        ),
        input_zero_point=_parameter("per_tensor", None, "i32", "input_zero_point"),
        weight_zero_point=_parameter("per_axis", 1, "i32", "weight_zero_point"),
        output_zero_point=_parameter("per_tensor", None, "i32", "output_zero_point"),
        bias_domain="real",
        bias=_parameter("per_axis", 1, "f32", "bias"),
        ordered_stages=(
            "scale_activation",
            "scale_weight_per_axis",
            "bias_real_per_axis",
            "relu",
            "scale_output_reciprocal",
            "round_to_nearest_even",
            "output_zero_point",
            "clamp",
        ),
        rounding="round_to_nearest_even",
        saturation=(-128, 127),
        activation="relu",
    )


def _candidate(*, scale: float = 0.5) -> QuantizedEpilogueCandidate:
    return QuantizedEpilogueCandidate(
        capability="scalar_scale_readout",
        scale_granularity="per_tensor",
        scale_axis=None,
        scales=(scale,),
        bias_domain="none",
        bias=(),
        output_zero_point=0,
        ordered_stages=("scale_f32", "round_to_nearest_even", "clamp", "relu"),
        rounding="round_to_nearest_even",
        saturation=(-128, 127),
        activation="relu",
    )


def _capability() -> QuantizedEpilogueCapability:
    return QuantizedEpilogueCapability(
        name="scalar_scale_readout",
        accumulator_dtypes=("i32",),
        output_dtypes=("i8",),
        scale_granularities=("per_tensor",),
        bias_domains=("none",),
        output_zero_points=(0,),
        roundings=("round_to_nearest_even",),
        saturations=((-128, 127),),
        ordered_stage_templates=((
            "scale_f32", "round_to_nearest_even", "clamp", "relu"
        ),),
        activations=("relu",),
    )


def _output_sha(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _policy_document(
    *, site_ids: tuple[str, ...] = ("ep0",), expected_delta: int = 0, opt_in: bool = True
) -> dict:
    accumulators = [-300, -3, -2, -1, 0, 1, 2, 3, 300]
    channels = [index % 2 for index in range(len(accumulators))]
    outputs = evaluate_candidate(_candidate(), accumulators, channels)
    if expected_delta:
        outputs[5] += expected_delta
    return {
        "schema": POLICY_SCHEMA,
        "opt_in": opt_in,
        "mode": "heldout_calibration_bounded",
        "budget": {
            "min_samples": len(accumulators),
            "min_channel_coverage_fraction": 1.0,
            "max_abs_error_lsb": 1,
            "max_accumulated_error_lsb": 1,
            "max_mean_abs_error_lsb": 0.2,
            "max_mismatch_fraction": 0.2,
            "max_saturation_mismatch_fraction": 0.0,
        },
        "corpora": [{
            "id": "holdout",
            "role": "holdout",
            "dataset_sha256": DATASET_SHA,
            "sampling_policy_sha256": SAMPLING_SHA,
            "selection_corpus_sha256": SELECTION_SHA,
            "independent_reference_sha256": REFERENCE_SHA,
        }],
        "sites": [{
            "site_id": site_id,
            "normalized_source_sha256": SOURCE_SHA,
            "source_operation_index": index,
            "semantic_contract_sha256": _contract().to_dict()["semantic_contract_sha256"],
            "corpus_id": "holdout",
            "candidate": _candidate().to_dict(),
            "samples": {
                "accumulators": accumulators,
                "channel_indices": channels,
                "reference_outputs": outputs,
                "reference_output_sha256": _output_sha(outputs),
            },
        } for index, site_id in enumerate(site_ids)],
    }


def _policy(**kwargs) -> AccuracyPolicy:
    return AccuracyPolicy.from_mapping(_policy_document(**kwargs))


def _exact(site_id: str, source_operation_index: int) -> tuple[
    QuantizedEpilogueCandidate, ExactEpilogueProof
]:
    candidate = _candidate()
    return candidate, ExactEpilogueProof(
        status="verified",
        method="symbolic source/candidate equivalence",
        normalized_source_sha256=SOURCE_SHA,
        source_operation_index=source_operation_index,
        semantic_contract_sha256=_contract().to_dict()["semantic_contract_sha256"],
        candidate_sha256=candidate.sha256,
        evidence_sha256=EVIDENCE_SHA,
    )


def test_contract_carries_all_quantization_semantics_and_stable_identity() -> None:
    receipt = _contract().to_dict()
    assert receipt["scales"]["weight"] == {
        "granularity": "per_axis", "axis": 1, "dtype": "f32", "binding": "weight_scale"
    }
    assert receipt["bias_domain"] == "real"
    assert receipt["rounding"] == "round_to_nearest_even"
    assert receipt["saturation"] == [-128, 127]
    assert receipt["ordered_stages"][0] == "scale_activation"
    assert len(receipt["semantic_contract_sha256"]) == 64
    with pytest.raises(ValueError, match="disagree with channel axis"):
        QuantizedEpilogueContract(
            **{
                **_contract().__dict__,
                "weight_scale": _parameter("per_axis", 0, "f32", "weight_scale"),
            }
        )


def test_policy_requires_explicit_distinct_holdout_corpus_provenance() -> None:
    parsed = parse_accuracy_policy(_policy_document())
    assert parsed.policy is not None
    assert parsed.receipt["status"] == "available"
    bad = _policy_document()
    bad["corpora"][0]["selection_corpus_sha256"] = DATASET_SHA
    refused = parse_accuracy_policy(bad)
    assert refused.policy is None
    assert refused.receipt["reason_codes"] == ["accuracy_policy_invalid"]
    missing = parse_accuracy_policy(None)
    assert missing.receipt["reason_codes"] == ["accuracy_policy_not_provided"]


def test_missing_policy_falls_back_to_exact_only_and_reports_all_gates() -> None:
    refused = admit_epilogue(
        site_id="ep0",
        normalized_source_sha256=SOURCE_SHA,
        source_operation_index=0,
        source_contract=_contract(),
        capability=_capability(),
    )
    assert not refused.selected
    assert set(refused.receipt["reason_codes"]) == {
        "exact_candidate_not_provided", "accuracy_policy_not_provided"
    }

    candidate, proof = _exact("ep0", 0)
    exact = admit_epilogue(
        site_id="ep0",
        normalized_source_sha256=SOURCE_SHA,
        source_operation_index=0,
        source_contract=_contract(),
        capability=_capability(),
        exact_candidate=candidate,
        exact_proof=proof,
    )
    assert exact.selected and exact.exact
    assert exact.max_observed_error_lsb == 0
    assert exact.receipt["accuracy_policy_consulted"] is False


def test_heldout_gate_recomputes_candidate_and_keeps_empirical_claim_boundary() -> None:
    admission = admit_epilogue(
        site_id="ep0",
        normalized_source_sha256=SOURCE_SHA,
        source_operation_index=0,
        source_contract=_contract(),
        capability=_capability(),
        policy=_policy(),
    )
    assert admission.selected and not admission.exact
    assert admission.receipt["proof_scope"] == "empirical_heldout_only"
    assert admission.receipt["exact_for_all_inputs"] is False
    assert admission.receipt["metrics"]["max_abs_error_lsb"] == 0
    assert admission.receipt["corpus"]["sampling_policy_sha256"] == SAMPLING_SHA
    assert "end-to-end accuracy" in admission.receipt["claim_boundary"]

    stale_document = _policy_document()
    stale_document["sites"][0]["normalized_source_sha256"] = "8" * 64
    stale = admit_epilogue(
        site_id="ep0",
        normalized_source_sha256=SOURCE_SHA,
        source_operation_index=0,
        source_contract=_contract(),
        capability=_capability(),
        policy=AccuracyPolicy.from_mapping(stale_document),
    )
    assert "accuracy_evidence_source_hash_mismatch" in stale.receipt["reason_codes"]


def test_accuracy_budget_failure_is_refusal_not_best_effort_selection() -> None:
    policy = _policy(expected_delta=2)
    decision = admit_epilogue(
        site_id="ep0",
        normalized_source_sha256=SOURCE_SHA,
        source_operation_index=0,
        source_contract=_contract(),
        capability=_capability(),
        policy=policy,
    )
    assert not decision.selected
    assert "heldout_accuracy_budget_exceeded" in decision.receipt["reason_codes"]
    assert "max_abs_error_lsb" in decision.receipt["failed_budget_dimensions"]


def _residual_problem(*, capacity: int | None = 1024, capabilities: bool = True) -> ResidualDomainProblem:
    required = ("wide_domain_add", "single_final_clamp", "resident_forwarding")
    return ResidualDomainProblem(
        domains=(
            QuantizationDomain("d0", 0.25, 0, -128, 127, DOMAIN_SHA),
            QuantizationDomain("d1", 0.5, 0, -128, 127, DOMAIN_SHA),
            QuantizationDomain("d2", 0.5, 0, -128, 127, DOMAIN_SHA),
        ),
        sites=(
            ResidualDomainSite(
                0, "r0", (10,), (11,), 64, "d0", None, 512, None, required, 800
            ),
            ResidualDomainSite(
                1, "r1", (20,), (21,), 64, "d1", 0, 256, 128, required, 512,
                saturation_scope="heldout_no_saturation", saturation_corpus_id="holdout",
            ),
            ResidualDomainSite(
                2, "r2", (30,), (31,), 64, "d2", 1, 4096, 4096, required, 512,
                saturation_scope="heldout_no_saturation", saturation_corpus_id="holdout",
            ),
        ),
        available_capabilities=frozenset(required if capabilities else ()),
        resident_capacity_bytes=capacity,
    )


def _single_residual_problem() -> ResidualDomainProblem:
    required = ("wide_domain_add", "single_final_clamp", "resident_forwarding")
    return ResidualDomainProblem(
        domains=(QuantizationDomain("d0", 0.25, 0, -128, 127, DOMAIN_SHA),),
        sites=(ResidualDomainSite(
            0, "r0", (10,), (11,), 64, "d0", None, 512, None, required, 512
        ),),
        available_capabilities=frozenset(required),
        resident_capacity_bytes=1024,
    )


def test_global_residual_solver_spends_one_budget_on_the_larger_downstream_edge() -> None:
    # Disable the independent resident-alignment lever so this isolates the cross-layer carry DP.
    report = plan_residual_domains(
        _residual_problem(capabilities=False), _policy(site_ids=())
    )
    by_id = {row["site_id"]: row for row in report["sites"]}
    assert not by_id["r1"]["carry_from_predecessor"]["selected"]
    assert by_id["r2"]["carry_from_predecessor"]["selected"]
    assert report["selection"]["selected_cross_layer_carries"] == 1
    assert report["cost"]["selected_materialization_bytes_eliminated"] >= 4096
    assert report["cost"]["cycle_prediction_claimed"] is False


def test_residual_capability_capacity_and_policy_are_independent_hard_gates() -> None:
    exact_default = plan_residual_domains(_residual_problem(), None)
    assert exact_default["selection"]["selected_residual_alignments"] == 0
    assert "accuracy_policy_not_provided" in exact_default["refusal_histogram"]

    missing_capability = plan_residual_domains(
        _residual_problem(capabilities=False), _policy(site_ids=())
    )
    assert missing_capability["selection"]["selected_residual_alignments"] == 0
    assert "missing_target_capability" in missing_capability["refusal_histogram"]

    over_capacity = plan_residual_domains(
        _residual_problem(capacity=100), _policy(site_ids=())
    )
    assert over_capacity["selection"]["selected_residual_alignments"] == 0
    assert "resident_capacity_exceeded" in over_capacity["refusal_histogram"]


def test_complete_program_plan_forms_regions_and_enumerates_every_host_refusal() -> None:
    exact0, proof0 = _exact("ep0", 0)
    exact1, proof1 = _exact("ep1", 1)
    problem = QuantizedRegionProblem(
        normalized_source_sha256=SOURCE_SHA,
        epilogues=(
            EpilogueSite("ep0", 0, _contract(), _capability(), exact0, proof0),
            EpilogueSite("ep1", 1, _contract(), _capability(), exact1, proof1),
        ),
        residuals=_single_residual_problem(),
        nodes=(
            QuantizedProgramNode(
                0, (0,), (100,), "enc_a", True, epilogue_site_id="ep0",
                resident_working_set_bytes=128,
            ),
            QuantizedProgramNode(
                1, (1,), (101,), "enc_a", True, epilogue_site_id="ep1",
                resident_working_set_bytes=128,
            ),
            QuantizedProgramNode(
                2, (10,), (11,), "enc_b", True, residual_site_index=0,
                resident_working_set_bytes=512,
            ),
            QuantizedProgramNode(
                3, (40,), (41,), None, False, base_refusal_reasons=("host_only_op",)
            ),
        ),
        edges=(
            QuantizedProgramEdge(0, 1, "enc_a", "enc_a", "v0", 64, 1),
            QuantizedProgramEdge(1, 2, "enc_a", "enc_b", "v1", 64, 1),
        ),
        resident_capacity_bytes=1024,
        unassigned_host_operation_ids=(99,),
    )
    report = plan_quantized_regions(problem, _policy(site_ids=()))
    regions = report["accelerator_region_formation"]
    assert [region["node_indices"] for region in regions["regions"]] == [[0, 1], [2]]
    assert regions["selected_source_operation_count"] == 3
    assert regions["direct_encoding_compatible_boundary_count"] == 1
    assert regions["logical_boundary_bytes_eliminated_if_lowered"] == 128
    assert regions["remaining_host_operation_ids"] == [41, 99]
    assert report["refusal_histogram"]["physical_encoding_transition_required"] == 1
    assert report["lowering_applied"] is False
    roles = {row["role"] for row in report["agent_edit_requirements"]}
    assert {
        "target_epilogue_emitter",
        "target_residual_region_emitter",
        "target_encoding_and_residency",
        "global_quant_domain_planner",
    } <= roles


def test_approximate_epilogue_needs_composed_bound_before_joining_residual_region() -> None:
    def problem(edge: QuantizedProgramEdge) -> QuantizedRegionProblem:
        return QuantizedRegionProblem(
            normalized_source_sha256=SOURCE_SHA,
            epilogues=(EpilogueSite("ep0", 0, _contract(), _capability()),),
            residuals=_single_residual_problem(),
            nodes=(
                QuantizedProgramNode(
                    0, (0,), (100,), "enc", True, epilogue_site_id="ep0",
                    resident_working_set_bytes=128,
                ),
                QuantizedProgramNode(
                    1, (10,), (11,), "enc", True, residual_site_index=0,
                    resident_working_set_bytes=512,
                ),
            ),
            edges=(edge,),
            resident_capacity_bytes=1024,
        )

    missing = plan_quantized_regions(
        problem(QuantizedProgramEdge(0, 1, "enc", "enc", "v", 64, 1)), _policy()
    )
    assert missing["nodes"][0]["selected_for_accelerator_region"] is False
    assert "downstream_error_composition_unproven" in missing["nodes"][0]["refusal_reasons"]
    assert missing["accelerator_region_formation"]["remaining_host_operation_ids"] == [100]

    bounded = plan_quantized_regions(
        problem(QuantizedProgramEdge(
            0, 1, "enc", "enc", "v", 64, 1,
            consumer_error_bound_lsb=0,
            error_bound_provenance_sha256=EVIDENCE_SHA,
        )),
        _policy(),
    )
    assert [region["node_indices"] for region in bounded[
        "accelerator_region_formation"
    ]["regions"]] == [[0, 1]]
    assert bounded["residual_domain_plan"]["sites"][0]["bounded_output_code_delta"] == 1


def _compiler_package(root: Path) -> Path:
    package = root / "compiler"
    (package / "lowering").mkdir(parents=True)
    (package / "lowering" / "planner.py").write_text(
        "def choose_domains():\n    return None\n\n"
        "def emit_epilogue():\n    return None\n",
        encoding="utf-8",
    )
    (package / "manifest.yaml").write_text(
        """components:
  emit: [lowering/]
optimization_surfaces:
  - id: domains
    scope: pass
    path: lowering/planner.py
    symbol: choose_domains
    effects: [quantization, encoding, residency, fusion]
    cca_axes: [communication.resident_across_calls, compute.epilogue]
    mechanism: select source-compatible quantization domains globally
    emitted_delta: remove representation boundaries across admitted regions
    validation: structural plan receipt plus reduced numeric witness
    abandonment: no legal region grows or warm cycles do not improve
  - id: epilogue-emission
    scope: codegen
    path: lowering/planner.py
    symbol: emit_epilogue
    effects: [quantization, encoding, fusion]
    cca_axes: [compute.epilogue, layout.operand_major]
    mechanism: emit a capability-supported quantized accumulator readout
    emitted_delta: replace host epilogue with narrow target readout
    validation: exact or policy-bound numeric witness and artifact decode
    abandonment: capability or numeric admission refuses the site
""",
        encoding="utf-8",
    )
    return package


def test_agent_guidance_returns_only_host_verified_contract_bound_surfaces(tmp_path) -> None:
    exact, proof = _exact("ep0", 0)
    plan = plan_quantized_regions(
        QuantizedRegionProblem(
            normalized_source_sha256=SOURCE_SHA,
            epilogues=(EpilogueSite("ep0", 0, _contract(), _capability(), exact, proof),),
            residuals=None,
            nodes=(QuantizedProgramNode(
                0, (0,), (1,), "enc", True, epilogue_site_id="ep0",
                resident_working_set_bytes=64,
            ),),
            edges=(),
            resident_capacity_bytes=128,
        )
    )
    inventory = inspect_compiler_package(_compiler_package(tmp_path))
    guidance = guidance_for_quantized_region_plan(plan, inventory)
    requirement = next(
        row for row in guidance["requirements"] if row["role"] == "target_epilogue_emitter"
    )
    assert requirement["status"] == "authorized"
    assert requirement["authorized_edit_surfaces"][0]["path"] == "lowering/planner.py"
    assert guidance["edit_contract_sha256"] == guidance["compiler_edit_contract"]["sha256"]
    assert guidance["candidate_manifest_grants_authority"] is False
    assert guidance["protected_inputs"][0]["editable_by_candidate"] is False

    injected = {**plan, "agent_edit_requirements": [{
        "role": "candidate_invented_path", "status": "required", "reason_codes": ["x"]
    }]}
    with pytest.raises(ValueError, match="unknown or repeated"):
        guidance_for_quantized_region_plan(injected, inventory)
