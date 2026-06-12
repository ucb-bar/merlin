"""Emit an llvm_requirement dict (conforming to ``llvm_requirement.schema.yaml``).

The L8 record. The promotion rule is deliberately strict: evidence frequency NEVER
justifies touching LLVM — only Stages E/F/G (interface validation, target lowering,
measured exploitability) can. Kernel mining therefore always emits
``requires_llvm_fork: false`` with the explicit ``fork_triggers`` that would justify a
change later, keeping the L8 rung of the ladder honest rather than empty.
"""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas

_JUSTIFICATION = ("Stage F (target lowering) and Stage G (compiler exploitability vs "
                  "oracle) have not run; no machine-code support is justified by evidence "
                  "frequency alone.")

# What WOULD justify a fork, per known interface (cf. llvm_extension_plan fork_triggers).
FORK_TRIGGERS: dict[str, list[str]] = {
    "resident_packed_tensor": ["custom_riscv_pack_instruction",
                               "new_instruction_selection_patterns", "mc_encoding_decoding"],
    "accumulator_commit": ["custom_accumulator_register_class", "commit_instruction_encoding"],
    "async_pipeline": ["async_copy_intrinsics", "machine_scheduling_model_changes"],
}


def emit_llvm_requirement(
    source_abstraction: str,
    fork_triggers: Iterable[str] | None = None,
    extra: dict | None = None,
    validate: bool = True,
) -> dict:
    """Build a schema-shaped L8 requirement (always: fork not yet justified)."""
    triggers = (list(fork_triggers) if fork_triggers is not None
                else FORK_TRIGGERS.get(source_abstraction,
                                       ["new_instruction_selection_patterns"]))
    req = {
        "source_abstraction": source_abstraction,
        "requires_llvm_fork": False,
        "justification": _JUSTIFICATION,
        "fork_triggers": triggers,
        "status": "not_justified_pending_stage_F_G",
    }
    if extra:
        req.update(extra)
    if validate:
        schemas.validate_or_raise(req, "llvm_requirement")
    return req
