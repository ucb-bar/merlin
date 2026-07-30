"""Synthesize an llvm_extension_plan (validates against llvm_extension_plan.schema.yaml).

Required: target, requires_llvm_fork, initial_strategy, out_of_tree_candidates, fork_triggers.

Default posture is out-of-tree; a fork is only ever justified by ``fork_triggers``. No real backend
patches are produced here. The fork posture is DERIVED from the compute-unit family's codegen
ENDPOINT (``families.contract_endpoint_kind`` -> ``family_profile``), never from a target name:
  command_buffer / inline_asm_insn / external_backend -> no fork (runtime calls / stock-LLVM .insn /
    a device assembler)
  upstream_target (vector/scalar RVV/RISC-V) -> maybe (custom extensions may need TableGen/backend)
A contract with no compute_units (the neutral ``toy_npu`` example) resolves no family and falls back to
the FAMILY-DEFAULT seed (out-of-tree, runtime-calls/command-buffer, no fork).
"""
from __future__ import annotations

from typing import Any

from .. import families as _families
from ..evidence.store import Evidence

_COMMON_OOT = ["td_fragments", "intrinsic_headers", "lit_tests", "patch_series"]
_COMMON_TRIGGERS = [
    "new_target_backend_registration",
    "new_register_classes",
    "new_instruction_selection_patterns",
    "mc_encoding_decoding",
    "assembler_disassembler_support",
]

# Fork posture per codegen ENDPOINT (the family axis), not per target name. {endpoint -> (requires_fork,
# initial_strategy, reason)}.
_BY_ENDPOINT: dict[str, tuple[Any, str, str]] = {
    "command_buffer": (False, "runtime_calls_or_command_buffer",
                       "Executes via the target command buffer / simulator; no LLVM changes."),
    "inline_asm_insn": (False, "inline_asm_insn_on_stock_llvm",
                        "Emit the target dialect as llvm.inline_asm/.insn on STOCK LLVM; patch LLVM "
                        "only if custom-instruction emission is later required."),
    "external_backend": (False, "external_device_assembler",
                         "Emit a device kernel.S the target's own assembler builds; no host LLVM fork."),
    "upstream_target": ("maybe", "rvv_intrinsics_or_existing_riscv_vector_path",
                        "Vector/scalar lowers through the upstream LLVM RISC-V/RVV path; custom "
                        "extensions may eventually require LLVM TableGen/backend changes."),
}
# The FAMILY-DEFAULT seed (the neutral toy_npu example resolves no compute-unit family): out-of-tree,
# runtime-calls / command-buffer, no fork.
_DEFAULT = (False, "runtime_calls_or_command_buffer",
            "Default out-of-tree posture (family-default seed); confirm against target source.")


def synthesize_llvm_extension_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return an llvm_extension_plan dict for the contract's target, fork posture DERIVED from its
    compute-unit family endpoint (family-default seed when the contract resolves no family)."""
    target = target_contract.get("name", evidence.target)
    endpoint = _families.contract_endpoint_kind(target_contract)
    grounded = endpoint in _BY_ENDPOINT
    fork, strategy, reason = _BY_ENDPOINT[endpoint] if grounded else _DEFAULT
    return {
        "target": target,
        "requires_llvm_fork": fork,
        "initial_strategy": strategy,
        "reason": reason,
        "out_of_tree_candidates": list(_COMMON_OOT),
        "fork_triggers": list(_COMMON_TRIGGERS),
        # The family-default seed is the well-understood reference (high confidence, no review); a real
        # family-resolved hardware target is flagged for human review.
        "confidence": "medium" if grounded else "high",
        "requires_human_review": grounded,
    }
