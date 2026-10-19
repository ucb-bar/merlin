"""Explicit shared compiler source owners; no discovery or import grants.

The export writer and ordinary pass receipts consume this same roster. Adding
an owner requires review of its import/resource closure and authoring boundary.
This roster does not attest loaded code or external dependency implementations.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

LIBRARY_MODULES_V6 = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_verification",
    "merlin.compile.scheduling.configuration",
    "merlin.compile.scheduling.configuration_ir",
    "merlin.compile.scheduling.copy_elimination",
    "merlin.compile.scheduling.dense_contracts",
    "merlin.compile.scheduling.dense_source",
    "merlin.compile.scheduling.device_spills",
    "merlin.compile.scheduling.fusion",
    "merlin.compile.scheduling.fusion_contracts",
    "merlin.compile.scheduling.fusion_semantics",
    "merlin.compile.scheduling.library_sources",
    "merlin.compile.scheduling.pointwise_source",
    "merlin.compile.scheduling.program_contracts",
    "merlin.compile.scheduling.program_limits",
    "merlin.compile.scheduling.row_contracts",
    "merlin.compile.scheduling.row_softmax",
    "merlin.compile.scheduling.row_softmax_verification",
    "merlin.compile.scheduling.schedule_pipeline",
    "merlin.compile.scheduling.selected_contracts",
    "merlin.compile.scheduling.spill_conformance",
    "merlin.compile.scheduling.spill_contracts",
    "merlin.compile.scheduling.storage_contracts",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.llvmlower",
    "merlin.llvmlower.cpu_softmax",
    "merlin.sched",
    "merlin.sched.codegen",
    "merlin.sched.ir",
    "merlin.sched.ir.expr",
    "merlin.sched.ir.integer_domain",
    "merlin.sched.ir.kernel",
    "merlin.sched.ir.parse",
    "merlin.sched.isa",
    "merlin.sched.isa.types",
    "merlin.semantic_compiler",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.egg_bridge",
    "merlin.semantic_compiler.extract",
    "merlin.semantic_compiler.model",
    "merlin.semantic_compiler.rules",
    "merlin.targetgen",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.isa_spec",
    "merlin.transforms",
    "merlin.transforms.certificates",
    "merlin.transforms.dense_ir",
    "merlin.transforms.dense_planning",
    "merlin.transforms.normalization",
    "merlin.transforms.pipeline_state",
    "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness",
    "merlin.transforms.unrolling",
    "merlin.verify",
    "merlin.verify.epilogue_applicability",
)

# Separate reviewed inventories: additions to V7 must never enlarge a V6
# source release or change the legacy receipt owner's import closure.
LIBRARY_MODULES_V7 = (*LIBRARY_MODULES_V6,)
LIBRARY_MODULES = LIBRARY_MODULES_V6


def library_modules(version=6):
    """Select a closed reviewed inventory; this grants no import access."""
    if type(version) is not int or version not in (6, 7):
        raise ValueError("shared compiler inventory requires version 6 or 7")
    return LIBRARY_MODULES_V6 if version == 6 else LIBRARY_MODULES_V7


def shared_library_sources(*, version=6):
    """Current source identities, equal for every treatment of one library."""
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in library_modules(version)
    }
