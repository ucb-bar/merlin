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
LIBRARY_MODULES_V7 = (
    *LIBRARY_MODULES_V6,
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
)
# Explicit prospective inventory; historical V6/V7 tuples remain unchanged.
LIBRARY_MODULES_V8 = (
    "merlin",
    "merlin.common",
    "merlin.common.dense_float_ir",
    "merlin.compile",
    "merlin.compile.scheduling",
    "merlin.compile.scheduling.block_schedule",
    "merlin.compile.scheduling.zero_initializer_groups",
    "merlin.compile.scheduling.shared_input_verification",
    "merlin.compile.scheduling.shared_input_schedule",
    "merlin.compile.scheduling.shared_input_groups",
    "merlin.compile.scheduling.panel_transfers",
    "merlin.compile.scheduling.copy_pipeline",
    "merlin.compile.scheduling.periodic_effects",
    "merlin.compile.scheduling.configuration_loops",
    "merlin.compile.scheduling.command_compaction",
    "merlin.compile.scheduling.compact_dense",
    "merlin.compile.scheduling.compact_loaded_sum",
    "merlin.compile.scheduling.spatial_gather",
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
    "merlin.compile.scheduling.llvm_program",
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
    "merlin.compile.scheduling.staged_dense",
    "merlin.compile.scheduling.staged_dense_verification",
    "merlin.compile.scheduling.temporary_storage",
    "merlin.compile.scheduling.window_transfer",
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
# Prospective selection only; existing reviewed inventories stay unchanged.
LIBRARY_MODULES_V9 = (*LIBRARY_MODULES_V8, "merlin.compile.scheduling.periodic_dense")
LIBRARY_MODULES_V10 = (
    *LIBRARY_MODULES_V9,
    "merlin.compile.scheduling.spatial_producer",
    "merlin.compile.scheduling.spatial_readout",
    "merlin.llvmlower.integer_readout",
    "merlin.llvmlower.integer_producer_range",
    "merlin.llvmlower.requantization",
)
LIBRARY_MODULES_V11 = (
    *LIBRARY_MODULES_V10,
    "merlin.llvmlower.spatial_scratch_abi",
    "merlin.llvmlower.fresh_tensor_writer",
    "merlin.llvmlower.private_workspace_pool",
    "merlin.llvmlower.borrowed_effects",
    "merlin.xdsl_dialects",
    "merlin.xdsl_dialects._common",
    "merlin.xdsl_dialects.fp8",
)
# Executable schedule dialect consumer; older inventories remain unchanged.
LIBRARY_MODULES_V12 = (
    *LIBRARY_MODULES_V11,
    "merlin.compile.scheduling.schedule_ir",
    "merlin.xdsl_dialects.schedule",
    "merlin.xdsl_dialects.contract",
)
LIBRARY_MODULES_V13 = (
    *LIBRARY_MODULES_V12,
    "merlin.compile.scheduling.loop_schedule",
    "merlin.sched.primitives",
)
LIBRARY_MODULES_V14 = (
    *LIBRARY_MODULES_V13,
    "merlin.compile.scheduling.dense_panel_pipeline",
    "merlin.compile.scheduling.scf_commands",
    "merlin.xdsl_dialects.runtime",
)
LIBRARY_MODULES_V15 = (
    *LIBRARY_MODULES_V14,
    "merlin.compile.scheduling.configuration_contexts",
    "merlin.compile.scheduling.configuration_loop_cleanup",
    "merlin.compile.scheduling.dense_panel_readout",
)
LIBRARY_MODULES_V16 = (
    *LIBRARY_MODULES_V15,
    "merlin.compile.scheduling.tile_load_coalescing",
    "merlin.compile.scheduling.panel_tile_loads",
    "merlin.compile.scheduling.dense_segmented_pipeline",
    "merlin.compile.scheduling.dense_array_retention",
    "merlin.compile.scheduling.api_catalog",
)
LIBRARY_MODULES_V17 = (
    *LIBRARY_MODULES_V16,
    "merlin.compile.scheduling.rolling_window",
    "merlin.compile.scheduling.rolling_spatial",
    "merlin.xdsl_dialects.rolling_window",
    "merlin.compile.scheduling.spatial_readout_cache",
)
LIBRARY_MODULES_V18 = (
    *LIBRARY_MODULES_V17,
    "merlin.compile.scheduling.dense_readout_overlap",
    "merlin.llvmlower.host_builder",
    "merlin.llvmlower.llvm_loop_metadata",
    "merlin.compile.scheduling.projection_views",
    "merlin.transforms.tensor_views",
    "merlin.llvmlower.source_region",
    "merlin.llvmlower.source_region_host",
    "merlin.compile.composition_catalog",
)
LIBRARY_MODULES_V19 = (*LIBRARY_MODULES_V18, "merlin.compile.scheduling.dense_ready_readout")
LIBRARY_MODULES = LIBRARY_MODULES_V6


def library_modules(version=6):
    """Select a closed reviewed inventory; this grants no import access."""
    if type(version) is not int or version not in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
        raise ValueError("shared compiler inventory requires version 6 through 19")
    return {
        6: LIBRARY_MODULES_V6,
        7: LIBRARY_MODULES_V7,
        8: LIBRARY_MODULES_V8,
        9: LIBRARY_MODULES_V9,
        10: LIBRARY_MODULES_V10,
        11: LIBRARY_MODULES_V11,
        12: LIBRARY_MODULES_V12,
        13: LIBRARY_MODULES_V13,
        14: LIBRARY_MODULES_V14,
        15: LIBRARY_MODULES_V15,
        16: LIBRARY_MODULES_V16,
        17: LIBRARY_MODULES_V17,
        18: LIBRARY_MODULES_V18,
        19: LIBRARY_MODULES_V19,
    }[version]


def shared_library_sources(*, version=6):
    """Current source identities, equal for every treatment of one library."""
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in library_modules(version)
    }
