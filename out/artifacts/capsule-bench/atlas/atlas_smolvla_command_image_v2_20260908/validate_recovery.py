#!/usr/bin/env python3
"""Validate saved Atlas command-image and RTL evidence without rerunning GSIM."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import test_command_image as tests
import test_capture_bridge as bridge_tests
import test_compact_loops as compact_tests
import test_first_partition as partition_tests
import test_full_graph_inventory as graph_tests
import test_hybrid_runtime as hybrid_tests
import test_parser_compat as parser_tests
import test_partition_plan as plan_tests

ROOT = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


tests.test_commit_result_is_a_real_consumer_operand()
tests.test_saved_elaborated_rtl_results_are_exact()
tests.test_raw_smolvla_readback_has_no_expected_payload()
tests.test_oracle_controls_detect_instrument_mismatch_and_no_echo()
tests.test_declined_whole_model_cannot_emit_a_trivial_image()
compact_tests.test_rank2_k_and_n_tails_fit_and_have_valid_runtime_loops()
compact_tests.test_partial_n_tail_and_batch_count_reuse_one_body()
compact_tests.test_compact_epilogues_fail_closed_instead_of_emitting_ecall_only()
compact_tests.test_static_bias_precedes_relu_in_the_encoded_epilogue()
compact_tests.test_emitted_pair_writers_use_even_banks_and_validator_can_fail()
parser_tests.test_multi_result_normalization_preserves_ordered_result_types()
parser_tests.test_full_smolvla_capture_parses_without_mutating_the_capture()
graph_tests.test_full_capture_partition_inventory_is_fail_closed()
partition_tests.test_first_addmm_partition_contains_bias_and_fits_imem()
plan_tests.test_whole_capture_plan_compiles_only_real_structural_contractions()
plan_tests.test_dependencies_lifetimes_and_abis_are_stable_and_explicit()
plan_tests.test_no_host_region_is_silently_promoted_to_a_command_image()
plan_tests.test_plan_and_manifests_are_byte_stable_across_rebuilds()
bridge_tests.test_calibration_scale_equations_and_rounding_are_explicit_and_deterministic()
bridge_tests.test_bias_is_folded_in_quant_domain_and_output_scale_is_restored()
bridge_tests.test_gsim_dram_window_rejects_full_p0244_and_accepts_declared_n_tiles()
bridge_tests.test_gsim_dram_window_rejects_masked_overlap_even_below_total_bytes()
bridge_tests.test_dispatch_manifest_comes_from_planned_dependency_and_lifetime_abi()
bridge_tests.test_capture_bridge_fails_closed_on_semantic_or_value_drift()
bridge_tests.test_saved_real_capture_qualification_is_scoped_and_passes_fixed_tolerance()
bridge_tests.test_action_in_projection_binds_real_noise_and_independently_passes()
bridge_tests.test_action_time_mlp_in_three_dispatches_independently_reconstruct_full_result()
hybrid_tests.test_interval_allocator_respects_inclusive_lifetimes_and_reuses_storage()
hybrid_tests.test_saved_hybrid_schedule_is_complete_ordered_and_fail_closed()
hybrid_tests.test_bounded_real_chain_replays_host_semantics_and_retains_scoped_evidence()
hybrid_tests.test_hybrid_schedule_is_byte_stable_across_rebuilds()

full = load(ROOT / "full_capture_probe.json")
raw = load(ROOT / "cases/smolvla_tail_50_720_32/raw_readback.json")
state_proj = load(ROOT / "cases/smolvla_state_proj_1_32_960/gsim_result.json")
inventory = load(ROOT / "full_capture_partition_inventory.json")
partition = load(ROOT / "partitions/first_addmm_matmul_0/compile_receipt.json")
plan = load(ROOT / "whole_capture_plan/partition_plan.json")
hybrid = load(ROOT / "whole_capture_plan/hybrid_schedule_summary.json")
capture_qualifications = {
    "atlas_p0098": load(ROOT / "capture_semantics_state_proj/result.json"),
    "atlas_p0243": load(ROOT / "capture_semantics_action_in_proj/result.json"),
    "atlas_p0244": load(ROOT / "capture_semantics_action_time_mlp_in/result.json"),
}
raw_capture_receipts = {
    partition_id: load(ROOT / result["raw_gsim_receipt"])
    for partition_id, result in capture_qualifications.items()
    if "raw_gsim_receipt" in result
}
action_time_receipts = [
    load(ROOT / path)
    for path in capture_qualifications["atlas_p0244"]["raw_gsim_receipts"]
]
qualification_records = {}
for partition_id, result in capture_qualifications.items():
    record = {
        "fqn": result["fqn"],
        "capture_regions": result["capture_regions"],
        "cycles": result["cycles"],
        "source_f32_comparison": result["source_f32_comparison"],
        "quantized_domain_reference_comparison": result[
            "quantized_domain_reference_comparison"
        ],
        "acceptance": result["acceptance"],
        "device_output": result["device_output"],
    }
    if partition_id in raw_capture_receipts:
        receipt = raw_capture_receipts[partition_id]
        record["raw_gsim_receipt"] = {
            "path": result["raw_gsim_receipt"],
            "spec_sha256": receipt["spec_sha256"],
            "stdout_sha256": receipt["stdout_sha256"],
            "engine_sha256": receipt["engine_sha256"],
            "halted": receipt["halted"],
            "halt_reason": receipt["halt_reason"],
            "cycles": receipt["cycles"],
            "final_pc_available": receipt["final_pc_available"],
            "assertion_clean": receipt["assertion_clean"],
            "stderr_observation": receipt["stderr_observation"],
        }
    else:
        record["n_slice_dispatches"] = [
            {
                "path": path,
                "cycles": receipt["cycles"],
                "engine_sha256": receipt["engine_sha256"],
                "assertion_clean": receipt["assertion_clean"],
                "stderr_observation": receipt["stderr_observation"],
                "raw_output_sha256": receipt["raw_output_sha256"],
            }
            for path, receipt in zip(result["raw_gsim_receipts"], action_time_receipts)
        ]
    qualification_records[partition_id] = record
cases = {}
for case_dir in sorted((ROOT / "cases").iterdir()):
    result_path = case_dir / "gsim_result.json"
    if result_path.is_file():
        result = load(result_path)
        cases[case_dir.name] = {
            "instruction_words": result["instruction_words"],
            "cycles": result["cycles"],
            "commands": result["commands"],
            "outputs": result["kernel_outputs"],
            "all_outputs_bit_exact": result["all_outputs_bit_exact"],
            "kernel_sha256": result["kernel_sha256"],
            "command_buffer_sha256": result["command_buffer_sha256"],
        }

verdict = {
    "schema": "atlas_smolvla_command_image_validation_v3",
    "ok": True,
    "recovery_status": "representative_rtl_numeric",
    "backend_source_tree_sha256": tree_digest(ROOT / "submission"),
    "focused_tests": {"passed": 31, "failed": 0},
    "full_capture_structural_compile_coverage": plan["compile_coverage"],
    "rtl_numeric_smolvla_coverage": {
        "unique_contraction_shapes": 3,
        "unique_contraction_shapes_total": 28,
        "capture_semantics_physical_occurrences": 3,
        "physical_contraction_occurrences_total": 391,
        "outputs_checked": 74560,
        "cases": {
            "smolvla_tail_50_720_32": {
                "shape": [50, 720, 32],
                "mismatches": raw["comparisons"]["Y0"]["mismatches"],
                "cycles": raw["cycles"],
                "instruction_words": cases["smolvla_tail_50_720_32"]["instruction_words"],
                "raw_output_sha256": raw["comparisons"]["Y0"]["raw_sha256"],
            },
            "smolvla_state_proj_1_32_960": {
                "shape": [1, 32, 960],
                "mismatches": state_proj["comparisons"]["Y0"]["mismatches"],
                "cycles": cases["smolvla_state_proj_1_32_960"]["cycles"],
                "instruction_words": cases["smolvla_state_proj_1_32_960"]["instruction_words"],
            },
            "real_action_in_proj_50_32_720": {
                "shape": [50, 32, 720],
                "cycles": capture_qualifications["atlas_p0243"]["cycles"],
                "instruction_words": capture_qualifications["atlas_p0243"]["image"][
                    "instruction_words"
                ],
                "source_max_abs_error": capture_qualifications["atlas_p0243"][
                    "source_f32_comparison"
                ]["max_abs_error"],
                "source_cosine_similarity": capture_qualifications["atlas_p0243"][
                    "source_f32_comparison"
                ]["cosine_similarity"],
            },
            "real_action_time_mlp_in_50_1440_720": {
                "shape": [50, 1440, 720],
                "dispatch_n_tiles": [256, 256, 208],
                "cycles": capture_qualifications["atlas_p0244"]["cycles"],
                "source_max_abs_error": capture_qualifications["atlas_p0244"][
                    "source_f32_comparison"
                ]["max_abs_error"],
                "source_cosine_similarity": capture_qualifications["atlas_p0244"][
                    "source_f32_comparison"
                ]["cosine_similarity"],
            },
        },
    },
    "representative_cases": cases,
    "full_capture_probe": {
        "returncode": full["returncode"],
        "command_buffer_produced": full["command_buffer_produced"],
        "command_count": full["command_count"],
        "parser_compat_rewrites": inventory["capture"]["parser_compat_rewrites"],
        "first_blocker": full["declined"]["reason"],
    },
    "full_graph_inventory": {
        "logical_regions": inventory["logical_regions"],
        "physical_contractions": inventory["physical_contractions"],
        "host_required_breakdown": inventory["host_required_breakdown"],
        "effective_candidate_windows": inventory["candidate_island_count"],
    },
    "full_capture_partition_plan": {
        "structural_partitions": plan["partition_count"],
        "kernel_variants": plan["kernel_variant_count"],
        "accelerator_dependency_edges": len(plan["accelerator_dependency_edges"]),
        "maximal_accelerator_islands": plan["maximal_accelerator_island_count"],
        "capture_semantics_executable_partitions": plan["capture_semantics_executable_partition_count"],
    },
    "hybrid_capture_schedule": hybrid,
    "real_capture_semantics_qualification": {
        "qualified_partitions": 3,
        "structural_partitions_total": 391,
        "partitions": qualification_records,
        "preferred_50x720x32_blocker": {
            "partition_id": "atlas_p0390",
            "fqn": "model.action_out_proj",
            "geometry": {"M": 50, "K": 720, "N": 32},
            "activation_origin": "host region dtype_cast_471 via view_1321",
            "reason": "requires the unresolved preceding host prefix; no captured source activation is bound",
        },
    },
    "first_concrete_partition": {
        "capture_regions": partition["capture_regions"],
        "instruction_words": partition["instruction_words"],
        "imem_words": partition["imem_words"],
        "fits_imem": partition["fits_imem"],
        "command_count": partition["command_count"],
        "numeric_qualified": partition["device_quantization"]["qualified_against_full_model"],
    },
    "functional_model": {
        "usable_as_numeric_oracle_for_emitted_atlas_isa": False,
        "definitive_decoder_mismatch": "DMA_CONFIG funct7=0 in RTL/compiler, funct7=1 in functional ISA",
        "only_known_defect": False,
    },
    "unproven": [
        "whole SmolVLA image",
        "extraction/dispatch for every full-graph partition",
        "numeric qualification of first_addmm_matmul_0 (state_proj is qualified separately)",
        "arbitrary FP8 accumulation correctness",
        "performance of the full model",
        "capsule score of this changed package",
    ],
}
(ROOT / "validation.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

receipt = {
    "schema": "atlas_smolvla_command_image_receipt_v3",
    "status": verdict["recovery_status"],
    "baseline": "atlas_smolvla_compact_loops_v1_20260908 (committed 28/28 IMEM-fit package)",
    "integration_fix": {
        "path": "submission/mlir_oot/frontend.py",
        "mechanism": "register each merlin_iface.commit SSA result in the tensor-name map",
        "before_failure": "OpResult tensor<4x8xbf16> lookup while parsing the second matmul lhs",
    },
    "full_capture_parser_fix": {
        "path": "submission/mlir_oot/frontend.py",
        "mechanism": "remove only redundant tuple wrappers around multi-result region type lists before xDSL parsing",
        "rewrites": 8,
        "source_capture_mutated": False,
        "next_blocker": "calibrated quantization bridges plus host/device graph dispatch",
    },
    "full_capture_partition_planner": {
        "path": "submission/mlir_oot/planner.py",
        "partitions": plan["partition_count"],
        "kernel_variants": plan["kernel_variant_count"],
        "maximal_accelerator_islands": plan["maximal_accelerator_island_count"],
        "capture_semantics_executable_partitions": plan["capture_semantics_executable_partition_count"],
        "manifests": [
            "whole_capture_plan/dependency_manifest.json",
            "whole_capture_plan/lifetime_manifest.json",
            "whole_capture_plan/abi_manifest.json",
        ],
    },
    "hybrid_capture_schedule": {
        "implementation": "submission/mlir_oot/hybrid_runtime.py",
        "builder": "build_hybrid_schedule.py",
        "summary": "whole_capture_plan/hybrid_schedule_summary.json",
        "full_schedule": hybrid["full_schedule"],
        "status": hybrid["status"],
        "runnable_e2e": hybrid["runnable_e2e"],
        "coverage": hybrid["coverage"],
        "fail_closed": hybrid["fail_closed"],
        "device_activation_arena": hybrid["device_activation_arena"],
        "bounded_chain": hybrid["bounded_chain"],
    },
    "capture_semantics_bridges": {
        "qualified_partitions": 3,
        "structural_partitions_total": plan["partition_count"],
        "calibration_contract": "calibration_contract.json",
        "partitions": {
            "atlas_p0098": {
                "fqn": "model.state_proj",
                "dispatch_manifest": "capture_semantics_state_proj/dispatch_manifest.json",
                "device_output": "capture_semantics_state_proj/device_output.bf16.bin",
                "raw_gsim_receipt": "capture_semantics_state_proj/raw_gsim_receipt.json",
                "result": "capture_semantics_state_proj/result.json",
            },
            "atlas_p0243": {
                "fqn": "model.action_in_proj",
                "dispatch_manifest": "capture_semantics_action_in_proj/dispatch_manifest.json",
                "device_output": "capture_semantics_action_in_proj/device_output.bf16.bin",
                "raw_gsim_receipt": "capture_semantics_action_in_proj/raw_gsim_receipt.json",
                "result": "capture_semantics_action_in_proj/result.json",
            },
            "atlas_p0244": {
                "fqn": "model.action_time_mlp_in",
                "dispatch_manifest": "capture_semantics_action_time_mlp_in/dispatch_manifest.json",
                "device_output": "capture_semantics_action_time_mlp_in/device_output.bf16.bin",
                "raw_gsim_receipts": capture_qualifications["atlas_p0244"][
                    "raw_gsim_receipts"
                ],
                "result": "capture_semantics_action_time_mlp_in/result.json",
                "execution": "three alias-free N-slice dispatches [256,256,208]",
            },
        },
        "preferred_50x720x32_blocker": "host-produced activation from dtype_cast_471 is not captured",
        "claim_scope": "three real capture partitions, not whole-model execution",
    },
    "compact_bias_fix": {
        "path": "submission/mlir_oot/codegen.py",
        "before": "bias_add omitted from compact FP8 matmul; first VREDSUM broadcast fix permuted lanes",
        "after": "explicit BF16 two-register row layout; 960/960 state-projection outputs exact on GSIM",
        "epilogue_safety": "bias precedes ReLU; unsupported compact scale/ReLU combinations fail closed",
    },
    "pair_bank_fix": {
        "before": "VLI_ALL 63 encoded odd destination bank 31 and aborted assertion-enabled GSIM",
        "after": "reserved VLI_ALL pair 62/63 encodes even six-bit base bank 62",
        "field_width": "RTL ScalarDecoder uses instr[12:7] for six-bit vd/vs fields",
        "static_guard": "encoder.validate_pair_banks mirrors VPU pair reads/writes and MXU BF16-pop writes",
        "rtl_guard": "atlas_gsim_sim_assert returns zero with empty stderr",
    },
    "hypotheses_ranked_before_fix": [
        "missing commit-result SSA registration",
        "intermediate tensor allocation/role overlap",
        "mixed FP8-to-BF16 codegen selection failure",
        "runtime numeric failure",
    ],
    "hypothesis_outcome": "first fixed the compile failure; second and third refuted for bounded chain; fourth passes only for constrained exact fixtures",
    "validation": verdict,
}
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, indent=2, sort_keys=True))
