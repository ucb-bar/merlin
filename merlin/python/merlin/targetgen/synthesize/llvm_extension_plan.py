"""Synthesize an llvm_extension_plan (validates against llvm_extension_plan.schema.yaml).

Required: target, requires_llvm_fork, initial_strategy, out_of_tree_candidates, fork_triggers.

Default posture is out-of-tree; a fork is only ever justified by ``fork_triggers``. No real
backend patches are produced here. Per-target defaults follow the spec:
  toy_npu / gemmini -> no fork (runtime calls / command buffers)
  saturn            -> maybe (RVV; custom extensions may later need TableGen/backend changes)
  radiance          -> no fork (command-processor packets / external SIMT toolchain)
"""
from __future__ import annotations

from typing import Any

from ..evidence.store import Evidence

_COMMON_OOT = ["td_fragments", "intrinsic_headers", "lit_tests", "patch_series"]
_COMMON_TRIGGERS = [
    "new_target_backend_registration",
    "new_register_classes",
    "new_instruction_selection_patterns",
    "mc_encoding_decoding",
    "assembler_disassembler_support",
]

_DEFAULTS: dict[str, dict[str, Any]] = {
    "toy_npu": {
        "requires_llvm_fork": False,
        "initial_strategy": "runtime_calls_or_command_buffer",
        "reason": "ToyNPU executes via the Merlin command buffer / simulator; no LLVM changes.",
    },
    "gemmini": {
        "requires_llvm_fork": False,
        "initial_strategy": "runtime_calls_or_command_buffer",
        "reason": "Emit C/RoCC wrapper calls first; patch LLVM only if custom-instruction "
                  "emission is later required.",
    },
    "saturn": {
        "requires_llvm_fork": "maybe",
        "initial_strategy": "rvv_intrinsics_or_existing_riscv_vector_path",
        "reason": "Saturn is RVV-oriented; custom extensions may eventually require LLVM "
                  "TableGen/backend changes.",
    },
    "radiance": {
        "requires_llvm_fork": False,
        "initial_strategy": "command_processor_packets_or_external_simt_toolchain",
        "reason": "Radiance can run via command-processor packets / standalone SIMT sim; "
                  "defer LLVM until a stable codegen story exists.",
    },
}


def synthesize_llvm_extension_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return an llvm_extension_plan dict for the contract's target."""
    target = target_contract.get("name", evidence.target)
    base = _DEFAULTS.get(target, {
        "requires_llvm_fork": False,
        "initial_strategy": "runtime_calls_or_command_buffer",
        "reason": "Default out-of-tree posture; confirm against target source.",
    })
    is_toy = target == "toy_npu"
    return {
        "target": target,
        "requires_llvm_fork": base["requires_llvm_fork"],
        "initial_strategy": base["initial_strategy"],
        "reason": base["reason"],
        "out_of_tree_candidates": list(_COMMON_OOT),
        "fork_triggers": list(_COMMON_TRIGGERS),
        "confidence": "high" if is_toy else "medium",
        "requires_human_review": not is_toy,
    }
