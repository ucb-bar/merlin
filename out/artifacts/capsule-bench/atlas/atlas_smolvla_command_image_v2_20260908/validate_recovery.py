#!/usr/bin/env python3
"""Validate saved Atlas command-image and RTL evidence without rerunning GSIM."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import test_command_image as tests
import test_compact_loops as compact_tests
import test_first_partition as partition_tests
import test_full_graph_inventory as graph_tests
import test_parser_compat as parser_tests

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "atlas_smolvla_compact_loops_v1_20260908"


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
parser_tests.test_multi_result_normalization_preserves_ordered_result_types()
parser_tests.test_full_smolvla_capture_parses_without_mutating_the_capture()
graph_tests.test_full_capture_partition_inventory_is_fail_closed()
partition_tests.test_first_addmm_partition_contains_bias_and_fits_imem()

parent = load(PARENT / "validation.json")
full = load(ROOT / "full_capture_probe.json")
raw = load(ROOT / "cases/smolvla_tail_50_720_32/raw_readback.json")
inventory = load(ROOT / "full_capture_partition_inventory.json")
partition = load(ROOT / "partitions/first_addmm_matmul_0/compile_receipt.json")
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
    "schema": "atlas_smolvla_command_image_validation_v2",
    "ok": True,
    "recovery_status": "representative_rtl_numeric",
    "backend_source_tree_sha256": tree_digest(ROOT / "submission"),
    "focused_tests": {"passed": 11, "failed": 0},
    "inherited_compile_coverage": {
        "unique_contraction_shapes_fitting_imem": parent["after"]["fitting_unique_shapes"],
        "unique_contraction_shapes_total": 28,
        "physical_contraction_layers_fitting_imem": parent["after"]["fitting_physical_layers"],
        "physical_contraction_layers_total": 391,
    },
    "rtl_numeric_smolvla_coverage": {
        "unique_contraction_shapes": 1,
        "unique_contraction_shapes_total": 28,
        "physical_contraction_occurrences": 1,
        "physical_contraction_occurrences_total": 391,
        "shape": [50, 720, 32],
        "outputs_checked": 1600,
        "mismatches": raw["comparisons"]["Y0"]["mismatches"],
        "cycles": raw["cycles"],
        "instruction_words": cases["smolvla_tail_50_720_32"]["instruction_words"],
        "raw_output_sha256": raw["comparisons"]["Y0"]["raw_sha256"],
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
        "numeric qualification of the first concrete partition",
        "arbitrary FP8 accumulation correctness",
        "performance of the full model",
        "capsule score of this changed package",
    ],
}
(ROOT / "validation.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

receipt = {
    "schema": "atlas_smolvla_command_image_receipt_v2",
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
        "next_blocker": "whole-model graph partition/dispatch lowering",
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
