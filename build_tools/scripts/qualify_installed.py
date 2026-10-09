"""Qualify committed core+experiments distributions outside the source checkout.

Usage: python build_tools/scripts/qualify_installed.py --ref HEAD --suite phase1
phase1 replays controller, CLI, RTL-feedback and private model-gate tests with core[xdsl].
runtime-admission checks cross-process simulator reservations without launching native simulators.
device-shim replays native C-interface ABI and numerical shim tests from the installed core.
phase0-inputs replays explicit recipe loading and declaration resolution, not hardware derivation.
compile-only checks ordinary source/object/link transport without tensor values or semantic authority.
Its optional --native-tool selections pin all three native executables and require zero test skips.
component-convergence admits the same tools for its declared Phase-1 compile-role transport tests;
other component tests retain their own prerequisites and their skips are reported separately.
host-arithmetic checks shared CPU arithmetic and admits an explicit host toolchain;
with that selection every test in its original roster must execute without skips.
invocation-record checks actual subprocess environment and executable provenance.
reviewed-corpus joins derivation, explicit review, installed Phase-1 authoring,
formal receipts and the Phase-2 checkpoint lifecycle against the same candidate
bytes. Agent transport, oracle results, measurement and OS isolation are synthetic;
this is not hardware qualification.
phase2-policy replays workflow, receipt, transcript and frozen-input contracts, not native engines.
phase2-lifecycle replays checkpoint execution/resume with synthetic external execution and admission.
source-snapshot checks explicit frozen source ownership and stdlib-only verifier loading.
revision-journal checks static revision publication and retained artifact history.
host-policy checks logical source ownership and portable frozen policy identities.
portfolio-checkpoint checks global candidate/authoring admission and scoped semantic evidence.
portfolio-resume checks archived verifier selection and cold synthetic checkpoint admission.
portfolio-evaluation checks frozen analytical evaluator configuration and serialized execution.
global-inputs checks staged frozen experiment admission and retained input integrity.
revision-session checks live revision admission and immutable checkpoint publication.
portfolio-analysis checks host-admitted portfolio analysis and exact revision reuse.
static-analysis-import checks pinned cross-run reuse without inheriting dynamic evidence.
portfolio-probes checks admitted probe lifecycle and accounting with synthetic execution.
global-experiment checks concrete installed experiment assembly with explicit input roots.
mechanism-program checks frozen host assignments and exact portfolio analysis bindings.
sandbox-inputs checks explicit sandbox configuration with mocked process execution.
qualification-policy checks frozen execution resources and private alias handling.
agent-sandbox-inputs checks captured agent/tool execution configuration without native launches.
portfolio-sandbox checks compiler policy assembly and exact cached-policy rebinding.
portfolio-authoring checks bounded authoring and checkpoint continuation with synthetic transport.
portfolio-launch-inputs checks installed launch admission and analytical-provider inputs without native execution.
portfolio-launch checks explicit deployment, lease/resource refusal and frozen worker supervision with fake processes.
portfolio-providers checks explicit backend capability assembly without executing runtime tools.
portfolio-worker checks installed worker admission, analysis-only results and terminal evidence without execution.
portfolio-cli checks installed deployment/catalog admission and frozen-launch input pins without execution.
performance-providers checks installed experiment-owned probe providers and private access identities.
initializer-admission checks bounded source-to-command initializer evidence, not numerical equivalence.
functional-qualification checks synthetic qualification execution/resume outside the checkout.
tensor-inspection checks dense payload storage, reconstruction and compact xDSL views without native compilers.
target-fetch checks published branch/tag acquisition using local Git repositories, not compiler execution.
lowering-inspection replays stage evidence and compact-view contracts; native compiler tests may skip.
Requires installed Merlin path helpers and uv. Dependency downloads may be needed; no network
services, hardware or paid agents are launched. This is packaging/functional regression evidence,
not numerical qualification, hardware certification or security isolation.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

SUITES = {
    "host-arithmetic": {
        "include_experiments": False,
        "native_tools": ("riscv-gcc",),
        "native_test_files": ("runtime/test_host_arithmetic.py",),
        "tests_root": "merlin/tests",
        "tests": ("runtime/test_host_arithmetic.py",),
        "core_extras": (),
        "probe_modules": ("merlin.runtime.host_arithmetic", "merlin.runtime.host_outward"),
        "required_modules": (),
    },
    "invocation-record": {
        "include_experiments": False,
        "tests_root": "merlin/tests",
        "tests": ("infra/test_invocation_record.py",),
        "core_extras": (),
        "probe_modules": ("merlin.common.invocation_record",),
        "required_modules": (),
    },
    "pinned-files": {
        "include_experiments": False,
        "tests_root": "merlin/tests",
        "tests": ("infra/test_pinned_files.py",),
        "core_extras": (),
        "probe_modules": ("merlin.common.pinned_files",),
        "required_modules": (),
    },
    "compile-only": {
        "native_tools": ("clang", "mlir-translate", "riscv-gcc"),
        "native_test_files": (
            "targetgen/test_compile_only_transport.py",
            "targetgen/test_shared_execution_deadline.py",
            "targetgen/test_frontend_use_def.py",
            "targetgen/test_stack_frame_preflight.py",
            "targetgen/test_explicit_execution_service.py",
            "infra/test_build_only_service.py",
            "targetgen/test_zero_input_abi.py",
        ),
        "tests_root": "merlin/tests",
        "tests": (
            "targetgen/test_compile_only_transport.py",
            "targetgen/test_shared_execution_deadline.py",
            "targetgen/test_frontend_use_def.py",
            "targetgen/test_stack_frame_preflight.py",
            "targetgen/test_explicit_execution_service.py",
            "infra/test_build_only_service.py",
            "targetgen/test_zero_input_abi.py",
            "infra/test_elf_build_cache.py",
            "targetgen/test_pointer_entry_abi.py",
            "targetgen/test_linalg_fill_inventory.py",
            "targetgen/test_linalg_constant_inventory.py",
            "targetgen/test_public_mixed_program_plan.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.compile_only_execution",
            "merlin.targetgen.contract.compile_only",
            "merlin.targetgen.contract.tensor_types",
            "merlin.targetgen.contract.compile",
            "merlin.targetgen.native_component_execution",
            "merlin.common.execution_deadline",
            "merlin.targetgen.frontend_use_def",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "component-cost": {
        "include_experiments": False,
        "tests_root": "merlin/tests/dse",
        "tests": (
            "test_component_cost.py",
            "test_global_planner.py",
            "test_fast_estimate_validation.py",
            "test_warm_profile_harness.py",
            "test_phase2_calibration_bundle.py",
            "test_phase2_feature_calibration.py",
            "test_perf_calibration_plan.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.perf.component_cost",
            "merlin.perf.component_screen",
            "merlin.perf.warm_profile_harness",
            "merlin.perf.phase2_calibration_bundle",
            "merlin.perf.fast_estimate_validation",
            "merlin.xdsl_dialects.lowering.global_plan",
        ),
        "required_modules": ("xdsl",),
    },
    "component-convergence": {
        "native_tools": ("clang", "mlir-translate", "riscv-gcc"),
        "native_test_files": (
            "test_component_compile_role_transport.py",
            "test_component_native_deadline.py",
            "test_component_pointer_entry.py",
        ),
        "tests": (
            "test_component_generation.py",
            "test_component_automatic.py",
            "test_component_source_binding.py",
            "test_component_operator_schemas.py",
            "test_component_zero_returns.py",
            "test_component_typed_add.py",
            "test_component_packing_sources.py",
            "test_component_hw_arithmetic.py",
            "test_component_automatic_composition.py",
            "test_independent_rtl_intake.py",
            "test_independent_command_intake.py",
            "test_independent_accessor_intake.py",
            "test_independent_source_predicate_intake.py",
            "test_independent_software_intake.py",
            "test_fresh_phase1_software_origin.py",
            "test_fresh_author_tools.py",
            "test_fresh_author_compiler_tools.py",
            "test_fresh_author_readiness.py",
            "test_runtime_dependency_projection.py",
            "test_fresh_phase1_generation_budget.py",
            "test_component_instruction_policy.py",
            "test_component_instruction_audit.py",
            "test_general_compiler_prompts.py",
            "test_phase1_task_staging.py",
            "test_component_origin.py",
            "test_component_lineage.py",
            "test_component_stage_lineage.py",
            "test_component_baseline.py",
            "test_component_runtime.py",
            "test_component_runtime_qualification.py",
            "test_component_runtime_support.py",
            "test_component_runtime_copy_controls.py",
            "test_component_native_deadline.py",
            "test_component_memory_transport.py",
            "test_component_decode_products.py",
            "test_component_pointer_entry.py",
            "test_component_input_projection.py",
            "test_component_runtime_controls.py",
            "test_component_measurement_qualification.py",
            "test_component_applicability.py",
            "test_component_observer.py",
            "test_rtl_engine_probe.py",
            "test_rtl_state_control.py",
            "test_rtl_native_memory.py",
            "test_component_coverage.py",
            "test_component_execution_budget.py",
            "test_component_integer_bounds.py",
            "test_component_graph_variants.py",
            "test_component_large_source_init.py",
            "test_component_compile_sources.py",
            "test_component_compile_graphs.py",
            "test_component_compile_admission.py",
            "test_component_compile_role_transport.py",
            "test_component_copy_proof.py",
            "test_component_input_palettes.py",
            "test_component_mechanisms.py",
            "test_component_minimal_spec.py",
            "test_component_semantic_basis.py",
            "test_component_workflow.py",
            "test_component_qualification.py",
            "test_component_package_execution.py",
            "test_component_container_context.py",
            "test_container_transport.py",
            "test_component_source_applicability.py",
            "test_component_launch.py",
            "test_component_launch_authority.py",
            "test_component_analytical.py",
            "test_component_screening.py",
            "test_supervised_feedback.py",
            "test_feedback_guardian.py",
            "test_component_normal_execution.py",
            "test_component_cca.py",
            "test_component_experiment.py",
            "test_fesvr_filesystem_containment.py",
            "test_component_final_policy.py",
            "test_numerical_readback.py",
            "test_protected_final_evaluation.py",
            "test_protected_verifier_qualification.py",
            "test_phase2_guard_link.py",
            "test_phase2_authoring_cli.py",
        ),
        # Only copied, committed test fixtures are added. The origin guard still
        # requires every compiler and orchestration import to come from wheels.
        "test_fixture_imports": True,
        "support_files": (
            "test_phase2_broker.py",
            "test_phase0_freeze.py",
            "component_baseline_fixture.py",
            "test_edit_authority.py",
            "reviewed_corpus_fixtures.py",
            "component_launch_fixture.py",
            "rtl_native_control_cases.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.component_coverage",
            "merlin_experiments.phase0.component_automatic",
            "merlin_experiments.phase0.component_automatic_plan",
            "merlin_experiments.phase0.arithmetic_intake",
            "merlin_experiments.phase0.component_arithmetic_obligations",
            "merlin.targetgen.rtl.hw_arithmetic",
            "merlin.targetgen.rtl.hw_packing",
            "merlin_experiments.phase0.packing_intake",
            "merlin_experiments.phase0.component_packing_sources",
            "merlin_experiments.phase0.operator_schema_intake",
            "merlin_experiments.phase0.tensor_argument_intake",
            "merlin_experiments.phase0.zero_return_intake",
            "merlin.targetgen.frontend_operator_effects",
            "merlin.targetgen.torch_schema_observer",
            "merlin.targetgen.torch_tensor_argument_observer",
            "merlin.targetgen.torch_zero_return_observer",
            "merlin.targetgen.frontend_typed_add",
            "merlin.targetgen.torch_schema_defaults_observer",
            "merlin_experiments.phase0.typed_add_sources",
            "merlin_experiments.phase0.component_execution_budget",
            "merlin_experiments.phase0.component_integer_bounds",
            "merlin_experiments.phase0.component_graph_variants",
            "merlin_experiments.phase0.component_graph_relations",
            "merlin_experiments.phase0.rtl_intake",
            "merlin_experiments.phase0.command_intake",
            "merlin_experiments.phase0.accessor_intake",
            "merlin_experiments.phase0.source_predicate_intake",
            "merlin_experiments.phase0.minimal_software",
            "merlin_experiments.phase0.software_intake",
            "merlin_experiments.phase1.component_origin",
            "merlin_experiments.phase1.component_generation_admission",
            "merlin_experiments.phase1.component_compile_admission",
            "merlin_experiments.phase1.component_compile_roles",
            "merlin_experiments.phase1.component_copy_proof",
            "merlin_experiments.phase1.component_pointer_storage",
            "merlin.targetgen.contract.pointer_storage",
            "merlin.llvmlower.counted_copy_check",
            "merlin.llvmlower.layout_observation",
            "merlin_experiments.phase1.component_lineage",
            "merlin_experiments.phase2.component_runtime_authority",
            "merlin_experiments.phase2.component_runtime_qualification",
            "merlin_experiments.phase2.component_runtime_support",
            "merlin_experiments.phase2.component_runtime_copy_controls",
            "merlin_experiments.phase2.component_runtime_copy_support",
            "merlin_experiments.phase2.component_runtime_controls",
            "merlin_experiments.phase2.component_instruction_policy",
            "merlin_experiments.phase2.component_instruction_audit",
            "merlin.targetgen.contract.elf_admission",
            "merlin.targetgen.generalization_prompt",
            "merlin.targetgen.generate_prompt",
            "merlin_experiments.phase1.task_staging",
            "merlin_experiments.phase2.component_measurement_qualification",
            "merlin_experiments.phase2.component_applicability",
            "merlin.perf.component_applicability",
            "merlin_experiments.phase2.component_baseline",
            "merlin_experiments.phase2.component_observer",
            "merlin_experiments.phase2.rtl_engine_protocol",
            "merlin_experiments.phase2.rtl_engine_probe",
            "merlin_experiments.phase2.rtl_state_control",
            "merlin_experiments.phase0.component_semantic_basis",
            "merlin_experiments.phase0.component_compile_plan",
            "merlin_experiments.phase0.component_compile_sources",
            "merlin_experiments.phase0.component_compile_graphs",
            "merlin.targetgen.input_palette",
            "merlin.targetgen.component_sources",
            "merlin_experiments.phase1.component_qualification",
            "merlin_experiments.phase1.component_qualification_evidence",
            "merlin_experiments.phase1.component_tool_readiness",
            "merlin_experiments.phase1.component_source_applicability",
            "merlin_experiments.phase2.component_launch",
            "merlin_experiments.phase2.component_stage",
            "merlin_experiments.phase2.component_final_qualification",
            "merlin_experiments.phase2.component_launch_probe",
            "merlin_experiments.phase2.component_launch_inputs",
            "merlin_experiments.phase2.component_analytical",
            "merlin_experiments.phase2.component_screening",
            "merlin_experiments.phase2.feedback_protocol",
            "merlin_experiments.phase2.supervised_feedback",
            "merlin_experiments.phase2.feedback_guardian",
            "merlin_experiments.execution.owned_children",
            "merlin_experiments.execution.container_image",
            "merlin_experiments.execution.container_policy",
            "merlin_experiments.execution.container_transport",
            "merlin_experiments.phase2.component_execution",
            "merlin_experiments.phase2.component_decode_products",
            "merlin_experiments.phase2.component_final_policy",
            "merlin_experiments.phase2.protected_final_evaluation",
            "merlin_experiments.phase2.protected_final_observation",
            "merlin_experiments.phase2.physical_final_admission",
            "merlin_experiments.phase2.protected_verifier_qualification",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "readback": {
        # The policy tests exercise core builds and experiments-owned oracle adapters.
        "include_experiments": True,
        "native_tools": ("clang",),
        "native_test_files": ("runtime/test_direct_kernel_invocation.py",),
        "tests_root": "merlin/tests",
        "tests": (
            "runtime/test_out_b64.py",
            "runtime/test_direct_kernel_harness.py",
            "runtime/test_direct_kernel_invocation.py",
            "targetgen/test_explicit_execution_service.py",
            "runtime/test_out_bin.py",
            "runtime/test_out_bin_bulk.py",
            "runtime/test_out_bin_memory.py",
            "runtime/test_out_packet.py",
            "runtime/test_out_b64_profile.py",
            "targetgen/test_invocation_readback_policy.py",
            "targetgen/test_memory_decode_observation.py",
            "infra/test_elf_build_cache.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.runtime.out_b64",
            "merlin.runtime.direct_kernel_harness",
            "merlin.runtime.direct_kernel_invocation",
            "merlin.targetgen.contract.execution_service",
            "merlin.targetgen.native_component_execution",
            "merlin.runtime.out_bin",
            "merlin.runtime.out_packet",
            "merlin.targetgen.contract.readback_policy",
        ),
        "required_modules": ("xdsl",),
    },
    "runtime-admission": {
        "include_experiments": False,
        "tests_root": "merlin/tests",
        "tests": ("targetgen/test_rtl_engine_policy.py", "targetgen/test_offload_census.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.rtl_engine_policy",
            "merlin.runtime.backends.base",
            "merlin.targetgen.offload_census",
            "merlin.targetgen.lowering_coverage",
            "merlin.targetgen.package_runtime",
        ),
        "required_modules": ("xdsl",),
    },
    "host-output": {
        "include_experiments": False,
        "tests_root": "merlin/tests",
        "tests": (
            "runtime/test_spike_model_exit.py",
            "rvv/test_scalar_host_qualification.py",
            "ir/test_quant_host_precision_policy.py",
            "ir/test_outline.py",
            "ir/test_quant_scope.py",
            "rvv/test_quant_passes.py",
            "runtime/test_compilation_recipe.py",
            "runtime/test_link_supplier_proof.py",
            "runtime/test_spike_libm_binding.py",
            "ir/test_linalg_composite_math.py",
            "ir/test_linalg_integer_reductions.py",
            "ir/test_linalg_extremum_patterns.py",
            "ir/test_linalg_f32_maximum_patterns.py",
            "ir/test_device_build.py",
            "infra/test_device_abi_resources.py",
            "ir/test_bucketize_source.py",
            "targetgen/test_host_capability_evidence.py",
            "targetgen/test_bucketize_host_source_body.py",
            "targetgen/test_f32_maximum_host_source_body.py",
            "runtime/test_whole_model_device_offload.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.compile.scalar_host_qualification",
            "merlin.llvmlower.c_runtime",
            "merlin.runtime.backends.spike_model",
            "merlin.llvmlower.link_supplier_trace",
            "merlin.llvmlower.device_build",
            "merlin.targetgen.contract.resident_interface_abi",
            "merlin.targetgen.host_linkage_contract",
            "merlin.frontends.linalg_composite_math",
            "merlin.frontends.bucketize_source",
            "merlin.frontends.linalg_f32_maximum_patterns",
            "merlin.frontends.linalg_reduction_source_body",
            "merlin.frontends.prepared_index_source_body",
        ),
        "required_modules": ("xdsl",),
    },
    "llvm-schema": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": (
            "test_public_llvm_metadata.py",
            "test_model_demand_canonical_family.py",
            "test_public_mixed_program_plan.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": ("merlin.targetgen.oot_starterkit.llvm_context", "merlin.targetgen.capsule_source"),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "device-abi": {
        "include_experiments": False,
        "tests_root": "merlin/tests/infra",
        "tests": ("test_device_abi_resources.py",),
        "core_extras": (),
        "probe_modules": ("merlin.llvmlower.device_shim", "merlin.targetgen.contract.schemas"),
        "required_modules": (),
    },
    "device-shim": {
        "include_experiments": False,
        "tests_root": "merlin/tests/ir",
        "tests": ("test_device_shim_abi.py",),
        "core_extras": (),
        "probe_modules": ("merlin.llvmlower.device_shim",),
        "required_modules": (),
    },
    "portfolio-cli": {
        "guarded_tests": True,
        "tests": ("test_portfolio_cli.py", "test_portfolio_catalog.py", "test_portfolio_launch.py"),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_cli", "merlin_experiments.portfolio_catalog"),
        "required_modules": ("xdsl",),
    },
    "portfolio-worker": {
        "guarded_tests": True,
        "tests": ("test_portfolio_worker.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_worker",),
        "required_modules": ("xdsl",),
    },
    "portfolio-providers": {
        "guarded_tests": True,
        "tests": ("test_portfolio_providers.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_providers",),
        "required_modules": ("xdsl",),
    },
    "portfolio-launch": {
        "guarded_tests": True,
        "tests": ("test_portfolio_launch.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_launch",),
        "required_modules": ("xdsl",),
    },
    "portfolio-launch-inputs": {
        "guarded_tests": True,
        "tests": ("test_portfolio_options.py", "test_fast_evaluation_installation.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase2.portfolio_options",
            "merlin_experiments.phase2.fast_evaluation_installation",
        ),
        "required_modules": ("xdsl",),
    },
    "rtl-protocol": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_rtl_check_providers.py", "test_rocc_provider_semantics.py", "test_hw_packing.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.rtl_checks",
            "merlin.targetgen.rtl_check_compiler",
            "merlin.targetgen.rtl_check_runner",
            "merlin.targetgen.circt_gate",
            "merlin.targetgen.rtl.hw_packing",
        ),
        "required_modules": ("xdsl",),
    },
    "phase1-provider-sources": {
        "tests": ("test_phase1_provider_sources.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase1.source_inputs",),
        "required_modules": ("xdsl",),
    },
    "target-fetch": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_oot_fetch.py",),
        "core_extras": (),
        "probe_modules": ("merlin.targetgen.oot_fetch",),
        "required_modules": (),
    },
    "measured-launch": {
        "tests": (
            "test_measured_claims_adapter.py",
            "test_checkpoint_child_environment.py",
            "test_chia_envelope.py",
            "test_chia_launch.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.measured_launch",
            "merlin_experiments.phase2.chia_envelope",
            "merlin_experiments.phase2.checkpoint_cli",
        ),
        "required_modules": ("xdsl",),
    },
    "chia-envelope": {
        "tests": ("test_chia_envelope.py", "test_chia_launch.py"),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.phase2.chia_envelope",
            "merlin_experiments.phase2.chia_envelope_cli",
        ),
        "required_modules": (),
    },
    "formal-handoff": {
        "tests": ("test_phase1_formal_handoff.py",),
        "support_files": ("phase1_feedback_fixtures.py",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase1.feedback.formal",
            "merlin_experiments.phase2.campaign",
            "merlin_experiments.phase2.global_inputs",
        ),
        "required_modules": ("xdsl",),
    },
    "broker-http": {
        "tests": ("test_phase2_broker.py", "test_broker_http_deadlines.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.broker",),
        "required_modules": (),
    },
    "codegen-declaration": {
        "include_experiments": False,
        "tests": ("test_preflight_codegen_declaration.py",),
        "core_extras": (),
        "probe_modules": ("merlin.targetgen.target_experiment",),
        "required_modules": (),
    },
    "codegen-smoke": {
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_codegen_smoke_fails_closed.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.targetgen.capsule_runner",),
        "required_modules": ("xdsl",),
    },
    "rocc-support": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_rocc_provider_semantics.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.targetgen.rocc.decode", "merlin.targetgen.rocc.asm"),
        "required_modules": ("xdsl",),
    },
    "phase0-inputs": {
        "tests": (
            "test_phase0_explicit_inputs.py",
            "test_phase0_declarations.py",
            "test_phase0_comparison.py",
            "test_certification_floor.py",
        ),
        "source_inputs": (
            "examples/*/experiment.yaml",
            "examples/*/target/descriptor.yaml",
            "examples/*/phase0/recipe.yaml",
        ),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.phase0.profiles",
            "merlin_experiments.phase0.generation",
            "merlin_experiments.phase0.declarations",
        ),
        "required_modules": (),
    },
    "capture-staging": {
        "tests": (
            "test_sealed_generation_capture.py",
            "test_sealed_m2m_capture.py",
            "test_phase0_capture_selection.py",
            "test_sealed_m2m_issuer_admission.py",
            "test_sealed_runtime_budget.py",
            "test_runtime_rehydrate.py",
        ),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.capture_execution.precision_staging",
            "merlin_experiments.capture_execution.sealed_m2m",
            "merlin_experiments.capture_execution.runtime_rehydrate",
            "merlin_experiments.phase0.sealed_generation",
        ),
        "required_modules": (),
    },
    "reviewed-corpus": {
        "tests": (
            "test_phase0_comparison_screen.py",
            "test_reviewed_corpus_phase1_handoff.py",
            "test_phase1_formal_handoff.py",
            "test_checkpoint_lifecycle.py",
            "test_corpus_release.py",
            "test_exact_offload_release_binding.py",
            "test_private_source_freeze.py",
            "test_private_full_models.py",
        ),
        "support_files": ("reviewed_corpus_fixtures.py", "phase1_feedback_fixtures.py"),
        "source_inputs": ("examples/*/target/descriptor.yaml",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase0",
            "merlin_experiments.corpus.release",
            "merlin_experiments.phase1.corpus_inputs",
            "merlin_experiments.phase1.feedback.private_source_freeze",
            "merlin_experiments.phase1.feedback.private_full_models",
            "merlin_experiments.phase1.feedback.formal",
            "merlin_experiments.phase2.functional_inputs",
        ),
        "required_modules": ("xdsl",),
        "required_entry_points": (
            "merlin.exact_offload_release:reviewed_phase0="
            "merlin_experiments.corpus.release:verify_exact_offload_binding",
        ),
    },
    "model-source-lifecycle": {
        "tests_root": "merlin/tests/targetgen",
        "tests": (
            "test_model_capsule_budget.py",
            "test_coverage_certificate.py",
            "test_public_mixed_program_plan.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.capsule_runner",
            "merlin.targetgen.capsule_grade",
            "merlin.targetgen.native_model_execution",
            "merlin.targetgen.coverage_certificate",
            "merlin.targetgen.oot_starterkit.plan",
        ),
        "required_modules": ("xdsl",),
    },
    "phase1": {
        "tests": (
            "test_candidate_selfcheck_feedback.py",
            "test_phase1_controller.py",
            "test_phase1_session.py",
            "test_phase1_readback_selection.py",
            "test_phase1_cli.py",
            "test_phase1_feedback.py",
            "test_phase1_codegen_scalability.py",
            "test_phase1_rtlchecks.py",
            "test_native_model_execution.py",
            "test_private_full_models.py",
            "test_private_pointwise_support.py",
            "test_private_linalg_support.py",
            "test_private_linkage_support.py",
            "test_private_literal_arange.py",
            "test_private_index_source.py",
            "test_private_integer_reduction_support.py",
            "test_private_f32_maximum_support.py",
            "test_private_ordered_scan_support.py",
            "test_private_bucketize_support.py",
            "test_private_source_freeze.py",
            "test_private_control_support.py",
            "test_private_pool_support.py",
            "test_private_pure_stage_support.py",
            "test_capsule_suite_dependencies.py",
            "test_public_caller_layout.py",
            "test_coherent_output_dump.py",
        ),
        "support_files": ("phase1_feedback_fixtures.py",),
        "source_inputs": ("examples/*/target/descriptor.yaml",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.contract.readback_policy",
            "merlin.targetgen.capsule_runner",
            "merlin.runtime.out_b64",
            "merlin.runtime.out_bin",
            "merlin.runtime.out_packet",
            "merlin.frontends.linalg_reduction_source_body",
            "merlin.frontends.prepared_index_source_body",
        )
        + tuple(
            "merlin_experiments.phase1." + tail
            for tail in (
                "authoring",
                "audit",
                "runtime_environment",
                "session",
                "controller",
                "task_staging",
                "workspace_transport",
                "feedback.certification",
                "feedback.qa",
                "feedback.selfcheck",
                "feedback.codegen_scalability",
                "feedback.rtlchecks",
                "feedback.private_full_models",
                "feedback.private_capture_roster",
                "feedback.private_pointwise_support",
                "feedback.private_linalg_support",
                "feedback.private_linkage_support",
                "feedback.private_literal_arange",
                "feedback.private_index_source",
                "feedback.private_index_host_support",
                "feedback.private_host_source_dispatch",
                "feedback.private_literal_arange_admission",
                "feedback.private_integer_reduction_support",
                "feedback.private_f32_maximum_support",
                "feedback.private_ordered_scan_support",
                "feedback.private_bucketize_support",
                "feedback.private_source_freeze",
                "feedback.private_pool_support",
                "feedback.private_pure_stage_support",
                "feedback.private_group_provenance",
                "feedback.private_device_audit",
                "feedback.caller_layout",
                "feedback.native_output_readback",
                "feedback.native_memory_readback",
                "feedback.native_packet_readback",
            )
        ),
        "required_modules": ("xdsl",),
    },
    "phase2-policy": {
        "tests": (
            "test_phase2_workflow_policy.py",
            "test_phase2_broker_evidence.py",
            "test_phase2_transcript_audit.py",
            "test_phase2_functional_inputs.py",
        ),
        "core_extras": (),
        "probe_modules": tuple(
            "merlin_experiments.phase2." + tail
            for tail in (
                "broker",
                "broker_policy",
                "broker_evidence",
                "corpus_feedback",
                "whole_model",
                "transcript_audit",
                "functional_inputs",
            )
        ),
        "required_modules": (),
    },
    "phase2-lifecycle": {
        "tests": ("test_checkpoint_lifecycle.py",),
        "core_extras": (),
        "probe_modules": tuple(
            "merlin_experiments.phase2." + tail
            for tail in (
                "checkpoint_admission",
                "checkpoint_controller",
                "checkpoint_cli",
                "paired_cli",
                "holdout_corpus",
                "chia_launch",
            )
        ),
        "required_modules": (),
    },
    "source-snapshot": {
        "tests": ("test_source_snapshot_ownership.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.source_snapshot", "merlin_experiments.frozen_python"),
        "required_modules": (),
    },
    "revision-journal": {
        "tests": ("test_revision_journal.py", "test_revision_artifacts.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.revision_journal",),
        "required_modules": (),
    },
    "static-reuse": {
        "tests": ("test_static_identity.py", "test_static_reuse.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.static_identity",),
        "required_modules": (),
    },
    "host-policy": {
        "tests": ("test_host_policy.py", "test_static_identity.py"),
        "support_files": ("host_policy_fixtures.py",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase2.host_policy",
            "merlin_experiments.phase2.static_identity",
        ),
        "required_modules": ("xdsl",),
    },
    "portfolio-checkpoint": {
        "tests": ("test_portfolio_checkpoint.py", "test_portfolio_checkpoint_semantics.py"),
        "support_files": ("portfolio_checkpoint_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_checkpoint",),
        "required_modules": (),
    },
    "portfolio-resume": {
        "tests": ("test_portfolio_resume.py", "test_portfolio_resume_frozen.py"),
        "support_files": ("portfolio_checkpoint_fixtures.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_resume",),
        "required_modules": ("xdsl",),
    },
    "portfolio-evaluation": {
        "tests": ("test_portfolio_evaluation.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_evaluation",),
        "required_modules": (),
    },
    "global-inputs": {
        "tests": ("test_global_inputs.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.global_inputs",),
        "required_modules": (),
    },
    "revision-session": {
        "tests": ("test_revision_session.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.revision_session",),
        "required_modules": (),
    },
    "portfolio-analysis": {
        "tests": ("test_portfolio_analysis.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_analysis",),
        "required_modules": (),
    },
    "static-analysis-import": {
        "tests": ("test_static_analysis_import.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.static_analysis_import",),
        "required_modules": (),
    },
    "portfolio-probes": {
        "tests": ("test_portfolio_probes.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_probes",),
        "required_modules": (),
    },
    "global-experiment": {
        "tests": ("test_global_experiment.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.global_experiment",),
        "required_modules": (),
    },
    "static-cache": {
        "tests": ("test_static_cache.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.static_cache",),
        "required_modules": (),
    },
    "edit-authority": {
        "tests": ("test_edit_authority.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.edit_authority",),
        "required_modules": (),
    },
    "mechanism-program": {
        "tests": ("test_mechanism_program.py", "test_mechanism_rounds.py"),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.phase2.mechanism_program",
            "merlin_experiments.phase2.mechanism_evidence",
            "merlin_experiments.phase2.mechanism_rounds",
        ),
        "required_modules": (),
    },
    "sandbox-inputs": {
        "tests": ("test_package_sandbox_inputs.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.campaign", "merlin.targetgen.sandbox.toolchain"),
        "required_modules": (),
    },
    "agent-sandbox-inputs": {
        "tests": ("test_agent_sandbox_inputs.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.agent_workspace", "merlin_experiments.phase2.broker"),
        "required_modules": (),
    },
    "portfolio-sandbox": {
        "tests": ("test_portfolio_sandbox.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_sandbox",),
        "required_modules": (),
    },
    "portfolio-authoring": {
        "tests": ("test_portfolio_authoring.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_authoring",),
        "required_modules": (),
    },
    "performance-providers": {
        "tests": ("test_performance_providers.py",),
        "core_extras": ("xdsl",),
        "probe_modules": tuple(
            "merlin.perf." + name
            for name in (
                "isolated_probe_provider",
                "controlled_context_provider",
                "paired_context_provider",
                "host_region_qualifier",
                "host_physical_transition_qualifier",
                "lane_migration_qualifier",
                "source_contraction_preparation",
                "source_convolution_preparation",
                "source_program_pair",
                "source_initializer_elision",
                "source_program_pair_provider",
            )
        ),
        "required_modules": ("xdsl",),
    },
    "initializer-admission": {
        "tests_root": "merlin/tests/dse",
        "tests": ("test_source_initializer_elision.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.perf.source_initializer_elision",),
        "required_modules": ("xdsl",),
    },
    "qualification-policy": {
        "tests": ("test_qualification_policy.py", "test_package_sandbox_inputs.py", "test_functional_qualification.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.qualification_policy",),
        "required_modules": (),
    },
    "functional-qualification": {
        "tests_root": "merlin/tests/infra",
        "tests": ("test_functional_gsim_qualification.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.functional_qualification",),
        "required_modules": (),
    },
    "tensor-inspection": {
        "include_experiments": False,
        "tests_root": "merlin/tests/ir",
        "tests": ("test_ir_audit_tensors.py", "test_inspection_tensor_payloads.py"),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.common.ir_audit", "merlin.xdsl_dialects.ir_inspection"),
        "required_modules": ("xdsl",),
    },
    "lowering-inspection": {
        "tests_root": "merlin/tests/ir",
        "tests": ("test_ir_audit.py", "test_ir_inspection.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.common.ir_audit",
            "merlin.compile_core",
            "merlin.llvmlower.ir_inspection",
        ),
        "required_modules": ("xdsl",),
    },
}
LIMITATIONS = [
    "Tool source is assumed static while Python initially imports it; observed hashes are rechecked at completion.",
    "Packaging and selected functional regressions only; no numerical/hardware certification.",
    "External venv and import-origin checks are not a security isolation guarantee.",
    "Dependency resolution uses configured indexes; exact resulting freeze is retained, not a lockfile replay.",
    "Unrecorded dependency overrides are removed from child environments; "
    "a local diagnostic override is not qualification.",
    "External venv is deliberately retained on success or failure; this command never deletes evidence.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_environment():
    excluded = {"UV_OVERRIDE", "UV_EXCLUDE", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT"}
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MERLIN", "PYTHON", "AET_", "CHIA_")) and k not in excluded
    }


NATIVE_TOOL_ENVIRONMENT = {
    "clang": "MERLIN_CLANG",
    "mlir-translate": "MERLIN_MLIR_TRANSLATE",
    "riscv-gcc": "MERLIN_TEST_RISCV_GCC",
}


def capture_native_tools(suite, selections):
    """Explicit executable selections, never ambient provider/env overrides."""
    selected = {}
    admitted = SUITES[suite].get("native_tools", ())
    for selection in selections:
        name, separator, supplied = selection.partition("=")
        if not separator or name not in admitted or name in selected:
            raise QualificationFailed("unknown, duplicate or suite-inadmissible native tool")
        path = Path(supplied)
        if not path.is_absolute():
            raise QualificationFailed("native tool must have an explicit absolute path")
        actual = path.resolve(strict=True)
        if not actual.is_file() or not os.access(actual, os.X_OK):
            raise QualificationFailed("native tool must resolve to an executable file")
        selected[name] = {
            "selected_path": str(path),
            "path": str(actual),
            "sha256": digest(actual),
            "environment_key": NATIVE_TOOL_ENVIRONMENT[name],
        }
    if selected and set(selected) != set(admitted):
        raise QualificationFailed("selected native suite needs its complete explicit tool roster")
    return selected


def verify_native_tools(selected):
    for tool in selected.values():
        if digest(tool["path"]) != tool["sha256"]:
            raise QualificationFailed("selected native executable changed")


def check_native_test_report(suite, path, report):
    """Require every declared native file to execute; distinguish other skips.

    The actual pytest child supplies xunit1 file identities relative to the
    explicit archived test root. This is packaging evidence, not native tool or
    compiler qualification. The retained XML preserves complete skip messages.
    """
    configured = SUITES[suite]
    required = configured.get("native_test_files", ())
    admitted = (*configured["tests"], *configured.get("support_files", ()))
    if not required or len(set(required)) != len(required) or not set(required) <= set(configured["tests"]):
        raise QualificationFailed("native suite has no closed declared test subset")
    cases = list(ET.parse(path).getroot().iter("testcase"))
    report["native_test_report"] = {"path": str(path), "sha256": digest(path)}
    report["native_test_files"] = list(required)
    report["native_zero_skip_scope"] = "declared_native_test_files"
    report["suite_test_counts"] = {"tests": len(cases), "skipped": 0}
    report["native_test_counts"] = {"tests": 0, "skipped": 0}
    report["other_test_counts"] = {"tests": 0, "skipped": 0}
    report["test_skips"] = {"native": [], "other": []}
    modules = {filename: Path(filename).with_suffix("").as_posix().replace("/", ".") for filename in admitted}
    observed, identities = set(), set()
    errors = []
    for case in cases:
        name, filename, classname = (case.get(key, "") for key in ("name", "file", "classname"))
        module = modules.get(filename)
        identity = (filename, classname, name)
        if (
            module is None
            or not name
            or identity in identities
            or not (classname == module or classname.startswith(module + "."))
        ):
            errors.append("testcase lacks a unique admitted archived file identity")
        identities.add(identity)
        category = "native" if filename in required else "other"
        report[category + "_test_counts"]["tests"] += 1
        if category == "native":
            observed.add(filename)
        skipped = case.find("skipped")
        if skipped is not None:
            report["suite_test_counts"]["skipped"] += 1
            report[category + "_test_counts"]["skipped"] += 1
            report["test_skips"][category].append(
                {
                    "file": filename,
                    "classname": classname,
                    "name": name,
                    "type": skipped.get("type", ""),
                    "message": skipped.get("message", ""),
                }
            )
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append("testcase has a failure or error")
    report["missing_native_test_files"] = sorted(set(required) - observed)
    if errors:
        raise QualificationFailed(errors[0])
    if report["missing_native_test_files"]:
        raise QualificationFailed("explicit native qualification did not execute every declared native test file")
    if report["native_test_counts"]["skipped"]:
        raise QualificationFailed("explicit native qualification requires declared native tests with zero skips")


def resolve_ref(root, ref):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=root,
        text=True,
        timeout=30,
    ).strip()


def reserve_output(base, label):
    if (
        not label
        or len(label) > 120
        or not label.isascii()
        or not label[0].isalnum()
        or any(not (character.isalnum() or character in "_.-") for character in label)
    ):
        raise ValueError("output label must be one safe path component")
    base = Path(base)
    if any(p.is_symlink() for p in (base, *base.parents)):
        raise ValueError("output ancestors may not be symlinks")
    base.mkdir(parents=True, exist_ok=True)
    output = base / label
    output.mkdir()  # Existing even-empty directories and symlinks are refused.
    return output


class QualificationFailed(RuntimeError):
    pass


class Recorder:
    def __init__(self, output, report, timeout):
        self.output, self.report, self.timeout = output, report, timeout
        self.environment = clean_environment()
        self.environment.update(
            (tool["environment_key"], tool["path"]) for tool in report.get("native_tools", {}).values()
        )

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def run(self, label, argv, cwd, *, stdout=None):
        verify_native_tools(self.report.get("native_tools", {}))
        argv = list(map(str, argv))
        log = self.output / (label + ".log")
        record = {
            "step": label,
            "argv": argv,
            "cwd": str(cwd),
            "timeout_s": self.timeout,
            "log": str(log),
            "returncode": None,
            "status": "running",
        }
        self.report["commands"].append(record)
        self.save()
        start = time.monotonic()
        try:
            with log.open("wb") as errors:
                stream = Path(stdout).open("wb") if stdout is not None else errors
                try:
                    with subprocess.Popen(
                        argv, cwd=cwd, env=self.environment, stdout=stream, stderr=errors, start_new_session=True
                    ) as child:
                        try:
                            record["returncode"] = child.wait(timeout=self.timeout)
                            record["status"] = "passed" if child.returncode == 0 else "failed"
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            record.update(status="timeout", returncode=child.returncode)
                        except KeyboardInterrupt:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            record.update(status="interrupted", returncode=child.returncode)
                            raise
                finally:
                    if stream is not errors:
                        stream.close()
        except OSError as exc:
            record.update(status="launch_failed", error=str(exc))
        finally:
            record["elapsed_s"] = time.monotonic() - start
            self.save()
        if record["status"] != "passed":
            raise QualificationFailed(f"{label}: {record['status']}; see {log}")


def projects(snapshot, extras, *, include_experiments=True):
    result = []
    roots = (Path("."), Path("packages/merlin-experiments")) if include_experiments else (Path("."),)
    for relative in roots:
        directory = snapshot / relative
        project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
        selected = extras if relative == Path(".") else ()
        for extra in selected:
            if extra not in project.get("optional-dependencies", {}):
                raise QualificationFailed(f"{project['name']} does not declare extra {extra}")
        result.append(
            {"name": project["name"], "version": project["version"], "path": str(directory), "extras": list(selected)}
        )
    return result


def only_artifact(directory, pattern):
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise QualificationFailed(f"expected one {pattern} in {directory}; got {files}")
    return files[0]


def source_input_archive_roots(patterns):
    """Archive literal owners; expand patterns only inside the selected commit."""
    roots = []
    for pattern in patterns:
        member = Path(pattern)
        if member.is_absolute() or ".." in member.parts or not member.parts:
            raise QualificationFailed(f"unsafe source input pattern: {pattern}")
        literal = []
        for part in member.parts:
            if any(char in part for char in "*?["):
                break
            literal.append(part)
        if not literal:
            raise QualificationFailed(f"source input pattern has no literal owner: {pattern}")
        roots.append(Path(*literal).as_posix())
    return tuple(dict.fromkeys(roots))


def selected_source_inputs(snapshot, patterns):
    source_input_archive_roots(patterns)
    names = set()
    for pattern in patterns:
        matched = sorted(snapshot.glob(pattern))
        if not matched:
            raise QualificationFailed(f"selected committed source input is missing: {pattern}")
        for source_path in matched:
            member = source_path.relative_to(snapshot)
            symlinked = any(
                snapshot.joinpath(*member.parts[:index]).is_symlink() for index in range(1, len(member.parts) + 1)
            )
            if not source_path.is_file() or symlinked or not source_path.resolve().is_relative_to(snapshot.resolve()):
                raise QualificationFailed(f"selected committed source input is unsafe: {member}")
            names.add(member.as_posix())
    return tuple(sorted(names))


def qualify(root, output, commit, suite, timeout, *, requested_ref=None, invocation=None, native_tools=()):
    own = Path(__file__).resolve()
    helper = own.with_name("installed_qualification_probe.py")
    tests_root = Path(SUITES[suite].get("tests_root", "packages/merlin-experiments/tests"))
    support_files = SUITES[suite].get("support_files", ())
    source_inputs = SUITES[suite].get("source_inputs", ())
    test_files = (*SUITES[suite]["tests"], *support_files)
    report = {
        "schema": "merlin.installed_qualification.v1",
        "ref": commit,
        "requested_ref": requested_ref,
        "invocation": invocation,
        "suite": suite,
        "status": "running",
        "commands": [],
        "limitations": LIMITATIONS,
        "tooling_revision": resolve_ref(root, "HEAD"),
        "tool_sources": {str(p.relative_to(root)): digest(p) for p in (own, helper)},
        "tool_source_note": "Actual executing bytes; hashes may include working-tree edits not in tooling_revision.",
        "selected_tests": list(SUITES[suite]["tests"]),
        "support_files": list(support_files),
        "test_fixture_imports": bool(SUITES[suite].get("test_fixture_imports")),
        "source_inputs": {},
        "tests_root": tests_root.as_posix(),
        "core_extras": list(SUITES[suite]["core_extras"]),
        "probe_modules": list(SUITES[suite]["probe_modules"]),
        "required_modules": list(SUITES[suite]["required_modules"]),
        "required_entry_points": list(SUITES[suite].get("required_entry_points", ())),
        "test_process_policy": (
            "deny_processes_and_listeners" if SUITES[suite].get("guarded_tests") else "suite_defined"
        ),
        "native_tools": {},
        "requested_native_tools": list(native_tools),
        "native_test_files": list(SUITES[suite].get("native_test_files", ())),
        "native_tool_policy": "Explicit executable bytes only; system dependencies are not a frozen toolchain closure.",
        "test_fixture_retention": "all; unique qualification-owned external temp root",
    }
    runner = Recorder(output, report, timeout)
    runner.save()
    try:
        report["native_tools"] = capture_native_tools(suite, native_tools)
        runner.environment.update((tool["environment_key"], tool["path"]) for tool in report["native_tools"].values())
        runner.save()
        copied_helper = output / "installed_qualification_probe.py"
        shutil.copyfile(helper, copied_helper)
        if digest(copied_helper) != report["tool_sources"][str(helper.relative_to(root))]:
            raise QualificationFailed("qualification helper changed during capture")
        runner.run(
            "resource-manifest",
            ["git", "show", commit + ":build_tools/package_resources.json"],
            root,
            stdout=output / "resources.json",
        )
        resources = json.loads((output / "resources.json").read_text())["files"]
        selected = [
            "src",
            "pyproject.toml",
            "setup.py",
            "README.md",
            "LICENSE",
            "MANIFEST.in",
            "build_tools/package_resources.json",
            "build_tools/extension_setup.py",
            "build_tools/scripts/check_distribution_layout.py",
            "packages/merlin-experiments/src",
            "packages/merlin-experiments/setup.py",
            "packages/merlin-experiments/pyproject.toml",
            *[(tests_root / n).as_posix() for n in test_files],
            *source_input_archive_roots(source_inputs),
            *resources,
        ]
        archive = output / "source.tar"
        runner.run("source-archive", ["git", "archive", "--format=tar", commit, "--", *selected], root, stdout=archive)
        snapshot = output / "snapshot"
        snapshot.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(snapshot, filter="data")
        for name in test_files:
            if not (snapshot / tests_root / name).is_file():
                raise QualificationFailed(f"selected committed test/support file is missing: {name}")
        source_inputs = selected_source_inputs(snapshot, source_inputs)
        report["source_archive_sha256"] = digest(archive)
        report["source_files"] = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*") if p.is_file()}
        report["projects"] = projects(
            snapshot,
            SUITES[suite]["core_extras"],
            include_experiments=SUITES[suite].get("include_experiments", True),
        )
        runner.save()
        wheels, install = [], []
        for project in report["projects"]:
            directory = output / "dist" / project["name"]
            directory.mkdir(parents=True)
            runner.run(
                project["name"] + "-sdist", ["uv", "build", "--sdist", "--out-dir", directory, project["path"]], output
            )
            sdist = only_artifact(directory, "*.tar.gz")
            runner.run(project["name"] + "-wheel", ["uv", "build", "--wheel", "--out-dir", directory, sdist], output)
            wheel = only_artifact(directory, "*.whl")
            wheels.append(wheel)
            install.append(str(wheel) + ("[" + ",".join(project["extras"]) + "]" if project["extras"] else ""))
            report.setdefault("artifacts", {}).update({str(p.relative_to(output)): digest(p) for p in (sdist, wheel)})
            runner.save()
        runner.run(
            "layout",
            [
                sys.executable,
                snapshot / "build_tools/scripts/check_distribution_layout.py",
                "--source-root",
                snapshot,
                *wheels,
            ],
            output,
        )
        external = Path(tempfile.mkdtemp(prefix="merlin-qualified-venv-"))
        report["external_venv"] = str(external)
        runner.save()
        if external.resolve().is_relative_to(root.resolve()):
            raise QualificationFailed("temporary venv must be outside the checkout; configure TMPDIR")
        runner.run("venv", ["uv", "venv", "--python", sys.executable, external], output)
        python = external / "bin/python"
        runner.run("install", ["uv", "pip", "install", "--python", python, *install], external)
        runner.run("dependencies", ["uv", "pip", "check", "--python", python], external)
        probe_args = [python, "-I", copied_helper]
        for module in SUITES[suite]["probe_modules"]:
            probe_args.extend(("--module", module))
        for module in SUITES[suite]["required_modules"]:
            probe_args.extend(("--require-module", module))
        for entry_point in SUITES[suite].get("required_entry_points", ()):
            probe_args.extend(("--require-entry-point", entry_point))
        runner.run("payload-probe", [*probe_args, *wheels], external)
        runner.run("pytest-install", ["uv", "pip", "install", "--python", python, "pytest"], external)
        runner.run("freeze", ["uv", "pip", "freeze", "--python", python], external, stdout=output / "dependencies.txt")
        report["dependency_freeze_sha256"] = digest(output / "dependencies.txt")
        tests = external / "qualification-tests"
        tests.mkdir()
        for name in test_files:
            (tests / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / tests_root / name, tests / name)
        for name in source_inputs:
            retained = tests / "source-inputs" / name
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / name, retained)
            source_sha256 = digest(snapshot / name)
            if digest(retained) != source_sha256:
                raise QualificationFailed(f"copied source input differs from committed archive: {name}")
            report["source_inputs"][name] = {"path": str(retained), "sha256": source_sha256}
        runner.save()
        shutil.copyfile(copied_helper, tests / "conftest.py")
        runner.run(
            "tests",
            [
                python,
                "-I",
                *([copied_helper, "--guarded-tests"] if SUITES[suite].get("guarded_tests") else ["-m", "pytest"]),
                "-q",
                "-c",
                "/dev/null",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                "-o",
                "tmp_path_retention_policy=all",
                *(["-o", "pythonpath=" + str(tests)] if SUITES[suite].get("test_fixture_imports") else []),
                # This venv is fresh and unique; pytest must not clean shared user temp roots.
                "--basetemp",
                external / "test-tmp",
                *(
                    [
                        "--rootdir",
                        tests,
                        "-o",
                        "junit_family=xunit1",
                        "--junitxml",
                        output / "tests.xml",
                    ]
                    if report["native_tools"]
                    else []
                ),
                tests,
            ],
            external,
        )
        verify_native_tools(report["native_tools"])
        if report["native_tools"]:
            check_native_test_report(suite, output / "tests.xml", report)
        if any(digest(root / name) != expected for name, expected in report["tool_sources"].items()):
            raise QualificationFailed("qualification tooling changed during execution")
        report["status"] = "passed"
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        runner.save()
    return report["status"] == "passed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git commit/ref to archive; never the dirty worktree")
    parser.add_argument("--suite", choices=SUITES, default="phase1")
    parser.add_argument("--label", help="New output directory name beneath build_dir()/python/qualified-installs")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per child command")
    parser.add_argument(
        "--native-tool",
        action="append",
        default=[],
        metavar="NAME=ABSOLUTE_PATH",
        help="Explicit suite-admitted executable; pin bytes and require zero skips in declared native test files",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    from merlin.common.paths import build_dir, repo_root

    root = repo_root()
    commit = resolve_ref(root, args.ref)
    label = args.label or (
        datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:12] + "-" + uuid.uuid4().hex[:8]
    )
    output = reserve_output(build_dir() / "python/qualified-installs", label)
    print(f"Qualification evidence: {output}", flush=True)
    return (
        0
        if qualify(
            root,
            output,
            commit,
            args.suite,
            args.timeout,
            requested_ref=args.ref,
            invocation=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            native_tools=args.native_tool,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
