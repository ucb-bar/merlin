"""Qualify committed core+experiments distributions outside the source checkout.

Usage: python build_tools/scripts/qualify_installed.py --ref HEAD --suite phase1
phase1 replays controller, CLI, RTL-feedback and private model-gate tests with core[xdsl].
runtime-admission checks cross-process simulator reservations without launching native simulators.
device-shim replays native C-interface ABI and numerical shim tests from the installed core.
phase0-inputs replays explicit recipe loading and declaration resolution, not hardware derivation.
source-preparation-qualification replays versioned source domain selection and real native
dependency/clone/output controls with synthetic author/runtime facets; it cannot qualify an experiment.
original-pointwise-host requires all seven original execution members with zero skips,
including without an optional native-tool inventory; missing compiler selections refuse.
original-candidate-members checks full typed transport and original qualification membership.
Every selected test is mandatory, including the actual source-only namespace check;
missing native tools or unavailable namespace isolation cannot qualify the suite.
host-ranked-descriptors requires all 46 selection controls and both ordinary
native descriptor cases with explicit compiler/source selections and zero skips.
This checks finite storage transport, not body, effects or runtime authority.
packing-memory-intake requires the complete 107 pure and 22 both-era native control roster.
Its explicit compiler pairs expose conditional structural facts, without packing/domain admission.
original-scalar-binary-sources requires the exact 110 source and six registered
conversion controls with explicit original public sources and native selections.
Construction/body correspondence does not admit numerical policy or coverage.
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
import importlib.util
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

_INPUT_SPEC = importlib.util.spec_from_file_location(
    "installed_native_inputs", Path(__file__).with_name("installed_native_inputs.py")
)
_INPUTS = importlib.util.module_from_spec(_INPUT_SPEC)
_INPUT_SPEC.loader.exec_module(_INPUTS)

SUITES = {
    "dispatch-placement": {
        "include_experiments": False,
        "tests_root": "merlin/tests/runtime",
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": ("test_dispatch_placement_execution.py",),
        "tests": ("test_dispatch_placement_execution.py",),
        "core_extras": (),
        "probe_modules": ("merlin.runtime.dispatch_runtime", "merlin.runtime.dispatch_placement"),
        "required_modules": ("numpy", "xdsl"),
        "guarded_tests": True,
    },
    "installed-test-origins": {
        "include_experiments": False,
        "tests_root": "merlin/tests/infra",
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": ("test_installed_test_origins.py",),
        "tests": ("test_installed_test_origins.py",),
        "source_inputs": ("build_tools/scripts/installed_qualification_probe.py",),
        "core_extras": (),
        "probe_modules": ("merlin.common.paths",),
        "required_modules": (),
        "guarded_tests": True,
    },
    "rtl-counter-timelines": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": (
            "test_hw_counter_intervals.py",
            "test_hw_counter_state_timelines.py",
            "test_hw_state_effects.py",
            "test_counter_source_tool.py",
        ),
        "tests": (
            "test_hw_counter_intervals.py",
            "test_hw_counter_state_timelines.py",
            "test_hw_state_effects.py",
            "test_counter_source_tool.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_counter_intervals",
            "merlin.targetgen.rtl.hw_counter_state_timelines",
            "merlin.targetgen.rtl.hw_state_effects",
            "merlin.targetgen.rtl.hw_combinational",
            "merlin.targetgen.rtl.hw_graph",
            "merlin.targetgen.tool_cli",
        ),
        "required_modules": ("xdsl",),
    },
    "packing-memory-intake": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("firtool", "circt-opt", "firtool-modern", "circt-opt-modern"),
        # All original pure and both-era native controls are mandatory. Source
        # fixtures and explicit reader budgets are retained in the same archive.
        "native_test_files": (
            "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
            "packages/merlin-experiments/tests/test_declared_phase0_run.py",
            "packages/merlin-experiments/tests/test_packing_memory_intake.py",
            "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
        ),
        "tests": (
            "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
            "packages/merlin-experiments/tests/test_declared_phase0_run.py",
            "packages/merlin-experiments/tests/test_packing_memory_intake.py",
            "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
        ),
        "native_test_cases": (
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_port_roster_and_distinct_intermediate_occurrences",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_original_concat_order_changes_exact_intervals_without_default_widening",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_partial_source_interval_does_not_fill_unproved_destination_bits",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_read_write_port_retains_mode_and_read_result_without_value_credit",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_intermediate_bits_remain_conditional_behind_original_stops[state-state_result]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_intermediate_bits_remain_conditional_behind_original_stops[opaque-opaque_instance_result]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_intermediate_bits_remain_conditional_behind_original_stops[read-memory_read_result]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_intermediate_bits_remain_conditional_behind_original_stops[unsupported-unsupported_result]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_intermediate_bits_remain_conditional_behind_original_stops[mux-combinational]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_parameterized_body_is_retained_as_a_stop_without_memory_port_invention",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_wire_budgets_precede_local_partition_expansion[operations]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_wire_budgets_precede_local_partition_expansion[extractions]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_wire_budgets_precede_local_partition_expansion[local_trace_work]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_wire_budgets_precede_local_partition_expansion[expression_depth]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_small_shared_dag_cannot_expand_an_exponential_local_trace_tree",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_join_and_materialization_budgets_refuse[partitions]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_join_and_materialization_budgets_refuse[slices]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_join_and_materialization_budgets_refuse[root_match_work]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_join_and_materialization_budgets_refuse[interval_pieces]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_join_and_materialization_budgets_refuse[cut_memberships]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_complete_join_and_materialization_budgets_refuse[traversal_work]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[operations]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[extractions]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[partitions]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[slices]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[local_trace_work]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[root_match_work]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[interval_pieces]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[cut_memberships]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[traversal_work]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_boolean_limits_cannot_replace_integer_budgets[expression_depth]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_original_named_binding_and_type_membership_is_required[missing]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_original_named_binding_and_type_membership_is_required[duplicate]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_original_named_binding_and_type_membership_is_required[reordered]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_original_named_binding_and_type_membership_is_required[type]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_malformed_original_extract_cannot_issue_bit_intervals[negative]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_malformed_original_extract_cannot_issue_bit_intervals[too_wide]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_malformed_original_extract_cannot_issue_bit_intervals[field_type]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_malformed_original_extract_cannot_issue_bit_intervals[signed]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_malformed_original_extract_cannot_issue_bit_intervals[unknown_field]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_malformed_original_extract_cannot_issue_bit_intervals[duplicate_field]",
            ),
            (
                "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
                "test_fresh_derivation_observes_source_changes_without_record_upgrade",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_request_cannot_import_authority_or_remove_construction_limits[saved_authority]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_request_cannot_import_authority_or_remove_construction_limits[missing_source_budget]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_request_cannot_import_authority_or_remove_construction_limits[missing_execution_budget]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_request_cannot_import_authority_or_remove_construction_limits[extra_command]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_pointwise_policy_request_keeps_both_limits_and_cannot_import_admission[None]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_pointwise_policy_request_keeps_both_limits_and_cannot_import_admission[source_budget]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_pointwise_policy_request_keeps_both_limits_and_cannot_import_admission[execution_budget]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_pointwise_policy_request_keeps_both_limits_and_cannot_import_admission[saved_status]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_changed_declared_bytes_refuse_before_issuer",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_forbidden_inputs_refuse_before_issuer",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_actual_issuer_failure_is_private_and_does_not_advance_phases",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_arbitrary_generator_error_cannot_be_a_completed_diagnostic[other_failure]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_arbitrary_generator_error_cannot_be_a_completed_diagnostic[complete]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_arbitrary_generator_error_cannot_be_a_completed_diagnostic[missing_manifest]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_arbitrary_generator_error_cannot_be_a_completed_diagnostic[no_unknown]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_request_preserves_closed_version_and_same_compiler[False]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_request_preserves_closed_version_and_same_compiler[True]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_declaration_cannot_import_authority_or_change_sdk_selection[bare_path]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_declaration_cannot_import_authority_or_change_sdk_selection[extra_authority]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_declaration_cannot_import_authority_or_change_sdk_selection[other_namespace]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_declaration_cannot_import_authority_or_change_sdk_selection[different_zero_compiler]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_declaration_cannot_import_authority_or_change_sdk_selection[missing_zero]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_declaration_cannot_import_authority_or_change_sdk_selection[extra_facet]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_compiler_membership_refuses_before_live_issuance[changed]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_compiler_membership_refuses_before_live_issuance[forbidden]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_compiler_membership_refuses_before_live_issuance[alias]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_bridge_compiler_membership_refuses_before_live_issuance[parent_component]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_normal_bridge_route_forwards_selection_to_existing_native_issuer[False]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_normal_bridge_route_forwards_selection_to_existing_native_issuer[True]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_changed_compiler_after_native_issuer_cannot_advance_to_rtl_observers",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_requirement_request_is_explicit_and_does_not_change_legacy_fields[source_diagnostic]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_requirement_request_is_explicit_and_does_not_change_legacy_fields[source_preparation]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_requirement_request_is_explicit_and_does_not_change_legacy_fields[performance_campaign]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_requirement_request_refuses_absent_purpose_or_saved_readiness[absent]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_requirement_request_refuses_absent_purpose_or_saved_readiness[unknown]",
            ),
            (
                "packages/merlin-experiments/tests/test_declared_phase0_run.py",
                "test_requirement_request_refuses_absent_purpose_or_saved_readiness[saved_ready]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_actual_typed_source_binding_retains_all_ports_and_undefined_memory_domains",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_selection_snapshot_does_not_alias_mutable_caller_budgets",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_complete_source_consumer_retains_state_read_opaque_and_unsupported_cuts[state_result]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_complete_source_consumer_retains_state_read_opaque_and_unsupported_cuts[memory_read_result]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_complete_source_consumer_retains_state_read_opaque_and_unsupported_cuts[opaque_instance_result]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_complete_source_consumer_retains_state_read_opaque_and_unsupported_cuts[unsupported_result]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[schema]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[saved_facts]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[root]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[source_bool]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[source_zero]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[missing_limits]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[missing_field]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[extra_field]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[bool_limit]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[float_limit]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_mapping_answers_and_incomplete_or_aliased_budgets_cannot_enter_selection[zero_limit]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_source_budget_precedes_parser_and_geometry_work",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_occurrence_budget_precedes_partition_expansion",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_ordinary_declared_caller_passes_only_explicit_v2_snapshot",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_new_declared_selector_preserves_every_legacy_requirement[None]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_new_declared_selector_preserves_every_legacy_requirement[missing]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_new_declared_selector_preserves_every_legacy_requirement[legacy]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_new_declared_selector_preserves_every_legacy_requirement[answer]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_new_declared_selector_preserves_every_legacy_requirement[missing_reference]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_new_declared_selector_preserves_every_legacy_requirement[missing_performance]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_exported_record_version_and_unknown_membership_refuse_before_native_replay[v1_extra]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_exported_record_version_and_unknown_membership_refuse_before_native_replay[v2_missing]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake.py",
                "test_exported_record_version_and_unknown_membership_refuse_before_native_replay[unknown_removed]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_same_source_v2_consumer_keeps_v1_partitions_and_every_memory_port[legacy]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_v2_data_does_not_change_existing_conditional_source_or_missing_mapping_roster[legacy]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-missing_port]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-false_as_zero]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-interval]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-unknown]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-source_owner]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-original_container]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-budget]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[legacy-new_facts_in_v1]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_exported_record_cannot_recreate_live_same_hardware_authority[legacy]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_same_source_v2_consumer_keeps_v1_partitions_and_every_memory_port[modern]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_v2_data_does_not_change_existing_conditional_source_or_missing_mapping_roster[modern]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-missing_port]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-false_as_zero]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-interval]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-unknown]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-source_owner]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-original_container]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-budget]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted[modern-new_facts_in_v1]",
            ),
            (
                "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
                "test_actual_exported_record_cannot_recreate_live_same_hardware_authority[modern]",
            ),
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_partition_memory_bindings",
            "merlin_experiments.phase0.packing_intake",
            "merlin_experiments.phase0.declared_run",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "original-scalar-binary-sources": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("operator-python", "firtool", "mlir-opt"),
        "native_python_entries": ("operator-python",),
        "native_sources": {"m2m": {"package": "m2m", "environment_key": "MERLIN_TEST_M2M_ROOT"}},
        "test_input_environment_keys": (
            "MERLIN_TEST_M2M_COMMIT",
            "MERLIN_TEST_OPERATOR_DECLARATIONS",
            "MERLIN_TEST_TORCH_SOURCE_ROOT",
        ),
        "native_test_files": (
            "merlin/tests/targetgen/test_original_scalar_binary_sources.py",
            "merlin/tests/targetgen/test_original_scalar_binary_correspondence.py",
            "packages/merlin-experiments/tests/test_original_scalar_binary_plan.py",
            "packages/merlin-experiments/tests/test_original_scalar_conversion_plan.py",
            "packages/merlin-experiments/tests/test_original_scalar_conversion_native.py",
        ),
        "tests": (
            "merlin/tests/targetgen/test_original_scalar_binary_sources.py",
            "merlin/tests/targetgen/test_original_scalar_binary_correspondence.py",
            "packages/merlin-experiments/tests/test_original_scalar_binary_plan.py",
            "packages/merlin-experiments/tests/test_original_scalar_conversion_plan.py",
            "packages/merlin-experiments/tests/test_original_scalar_conversion_native.py",
        ),
        "native_test_cases": tuple(
            (filename, method + suffix)
            for filename, methods in (
                (
                    "merlin/tests/targetgen/test_original_scalar_binary_sources.py",
                    (
                        (
                            "test_minimal_original_float_literal_call_has_one_input_and_complete_typed_result",
                            ("[aten.mul.Tensor-1.0]", "[aten.div.Tensor-2.23606797749979]"),
                        ),
                        (
                            "test_float_literal_kind_and_signed_zero_survive_without_tensorization_or_pre_rounding",
                            ("[0.0]", "[-0.0]", "[0.1]", "[-3.5]"),
                        ),
                        (
                            "test_unimplemented_literal_kinds_remain_required_unknown_forms",
                            ("[True]", "[1]", "[None]", "[1.0]", "[scalar4]"),
                        ),
                        (
                            "test_original_bindings_types_order_and_complete_result_roster_cannot_be_substituted",
                            (
                                "[literal]",
                                "[rank]",
                                "[storage]",
                                "[promotion]",
                                "[second_ssa]",
                                "[reversed]",
                                "[argument_schema]",
                                "[alias]",
                                "[result_roster]",
                                "[result_rank]",
                                "[dtype]",
                                "[default]",
                                "[arity_bool]",
                            ),
                        ),
                        (
                            "test_complete_logical_payload_and_rank_budgets_precede_geometry_allocation",
                            ("[2-2-11]", "[1000000000-1-100]", "[2-1000000000-100]"),
                        ),
                        (
                            "test_boolean_cannot_replace_an_explicit_positive_integer_budget",
                            ("[extent]", "[max_tensor_elements]"),
                        ),
                        (
                            "test_source_freezes_policy_literal_and_fresh_geometry_never_copies_original_extents",
                            ("",),
                        ),
                    ),
                ),
                (
                    "merlin/tests/targetgen/test_original_scalar_binary_correspondence.py",
                    (
                        (
                            "test_complete_literal_order_result_and_registered_construction_is_separate_from_admission",
                            (
                                "[1.0-aten.mul.Tensor]",
                                "[1.0-aten.div.Tensor]",
                                "[2.23606797749979-aten.mul.Tensor]",
                                "[2.23606797749979-aten.div.Tensor]",
                                "[-0.0-aten.mul.Tensor]",
                                "[-0.0-aten.div.Tensor]",
                                "[0.1-aten.mul.Tensor]",
                                "[0.1-aten.div.Tensor]",
                                "[1.0000000596046448-aten.mul.Tensor]",
                                "[1.0000000596046448-aten.div.Tensor]",
                                "[1.401298464324817e-45-aten.mul.Tensor]",
                                "[1.401298464324817e-45-aten.div.Tensor]",
                            ),
                        ),
                        (
                            "test_type_and_digest_equivalent_products_cannot_substitute_actual_scalar_body_relations",
                            (
                                "[coefficient]",
                                "[generic_order]",
                                "[empty_output]",
                                "[div_order]",
                                "[yield]",
                                "[return]",
                                "[map]",
                                "[iterator]",
                                "[fastmath]",
                                "[opaque_attr]",
                            ),
                        ),
                        (
                            "test_complete_actual_frontend_stage_semantics_must_match_the_original_call",
                            ("[literal]", "[kind]", "[second_ssa]", "[dtype]", "[schema]", "[result]", "[transition]"),
                        ),
                        (
                            "test_registry_source_and_actual_invocation_joins_cannot_be_replaced_by_matching_ids",
                            (
                                "[callable]",
                                "[source]",
                                "[literal]",
                                "[override]",
                                "[result]",
                                "[duplicate]",
                                "[missing]",
                            ),
                        ),
                        (
                            "test_complete_preallocation_and_source_roster_budgets_cannot_be_relaxed",
                            ("[max_tensor_elements-11]", "[max_operations-8]", "[max_source_bytes-100]"),
                        ),
                        (
                            "test_boolean_is_not_a_parser_or_logical_budget",
                            ("[max_nesting]", "[max_operations]", "[max_source_bytes]", "[max_tensor_elements]"),
                        ),
                        (
                            "test_literal_underflow_preserves_signed_zero_and_overflow_is_not_licensed",
                            ("",),
                        ),
                    ),
                ),
                (
                    "packages/merlin-experiments/tests/test_original_scalar_binary_plan.py",
                    (
                        (
                            "test_opt_in_observer_preserves_all_calls_cohorts_and_policy_without_new_admissions",
                            ("",),
                        ),
                        (
                            "test_legacy_v5_keeps_all_scalar_factory_gaps_and_exact_old_reader_roster",
                            ("",),
                        ),
                        (
                            "test_complete_original_roster_and_payload_budgets_retain_every_denied_slot",
                            ("[max_sources-8-0]", "[max_tensor_elements-11-2]", "[max_total_tensor_elements-79-5]"),
                        ),
                        (
                            "test_live_reconstruction_refuses_original_source_row_substitution",
                            (
                                "[cohort]",
                                "[literal]",
                                "[rank]",
                                "[storage]",
                                "[second_ssa]",
                                "[cost_bool]",
                                "[ordinal_bool]",
                                "[loader]",
                                "[member_drop]",
                                "[version]",
                            ),
                        ),
                        (
                            "test_automatic_new_version_preserves_closed_legacy_policy_fields_and_facets",
                            ("",),
                        ),
                    ),
                ),
                (
                    "packages/merlin-experiments/tests/test_original_scalar_conversion_plan.py",
                    (
                        (
                            "test_original_unsupported_slot_and_complete_three_cohorts_remain_with_no_admission",
                            ("",),
                        ),
                        (
                            "test_original_slot_denominator_and_old_vocabulary_cannot_be_changed",
                            ("[drop]", "[reorder]", "[cohort]", "[extent_bool]", "[legacy]"),
                        ),
                        (
                            "test_full_member_and_future_product_bounds_refuse_before_native_expansion",
                            ("[max_members-8]", "[max_total_source_bytes-1000000]"),
                        ),
                        (
                            "test_aggregate_logical_denial_preserves_original_request_slots",
                            ("",),
                        ),
                        (
                            "test_preflight_cannot_construct_a_substituted_original_source",
                            ("[literal]", "[storage]", "[loader]", "[metadata_alias]"),
                        ),
                        (
                            "test_closed_independent_selection_retains_exact_owners_and_budgets",
                            ("[source]", "[schema]", "[basis]", "[extra]", "[bool]", "[deadline]"),
                        ),
                        (
                            "test_saved_or_copied_records_cannot_mint_live_execution_authority",
                            ("",),
                        ),
                    ),
                ),
                (
                    "packages/merlin-experiments/tests/test_original_scalar_conversion_native.py",
                    (
                        (
                            "test_actual_public_schema_factory_registered_importer_complete_products_and_stock_parser",
                            ("",),
                        ),
                        (
                            "test_copied_live_object_and_modified_saved_receipt_cannot_issue_registered_conversion",
                            ("",),
                        ),
                        (
                            "test_actual_observed_product_drift_cannot_be_accepted",
                            ("[source]", "[trace]", "[registry]"),
                        ),
                        (
                            "test_selection_drift_cannot_change_original_members_or_native_bounds",
                            ("",),
                        ),
                    ),
                ),
            )
            for method, suffixes in methods
            for suffix in suffixes
        ),
        "support_files": ("packages/merlin-experiments/tests/original_scalar_conversion_fixtures.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.original_scalar_binary_sources",
            "merlin.targetgen.original_scalar_binary_correspondence",
            "merlin_experiments.phase0.original_call_sources",
            "merlin_experiments.phase0.original_scalar_conversion",
            "merlin_experiments.phase0.original_scalar_conversion_observer",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "original-broadcast-add-sources": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("operator-python", "firtool", "mlir-opt"),
        "native_python_entries": ("operator-python",),
        "native_sources": {"m2m": {"package": "m2m", "environment_key": "MERLIN_TEST_M2M_ROOT"}},
        "test_input_environment_keys": (
            "MERLIN_TEST_M2M_COMMIT",
            "MERLIN_TEST_OPERATOR_DECLARATIONS",
            "MERLIN_TEST_TORCH_SOURCE_ROOT",
        ),
        "native_test_files": (
            "merlin/tests/targetgen/test_original_broadcast_add.py",
            "packages/merlin-experiments/tests/test_original_broadcast_add_plan.py",
            "packages/merlin-experiments/tests/test_original_broadcast_add_flow.py",
        ),
        "tests": (
            "merlin/tests/targetgen/test_original_broadcast_add.py",
            "packages/merlin-experiments/tests/test_original_broadcast_add_plan.py",
            "packages/merlin-experiments/tests/test_original_broadcast_add_flow.py",
        ),
        "support_files": (
            "packages/merlin-experiments/tests/original_broadcast_add_fixtures.py",
            "packages/merlin-experiments/tests/original_reference_fixtures.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.original_broadcast_add_sources",
            "merlin.targetgen.original_operator_reference",
            "merlin_experiments.phase0.original_reference_roster",
            "merlin_experiments.phase0.original_reference_standard_ir",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "original-transpose-sources": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("operator-python", "firtool", "mlir-opt"),
        "native_python_entries": ("operator-python",),
        "native_sources": {"m2m": {"package": "m2m", "environment_key": "MERLIN_TEST_M2M_ROOT"}},
        "test_input_environment_keys": (
            "MERLIN_TEST_M2M_COMMIT",
            "MERLIN_TEST_OPERATOR_DECLARATIONS",
            "MERLIN_TEST_TORCH_SOURCE_ROOT",
        ),
        "native_test_files": (
            "merlin/tests/targetgen/test_original_transpose.py",
            "packages/merlin-experiments/tests/test_original_transpose_plan.py",
            "packages/merlin-experiments/tests/test_original_transpose_flow.py",
        ),
        "tests": (
            "merlin/tests/targetgen/test_original_transpose.py",
            "packages/merlin-experiments/tests/test_original_transpose_plan.py",
            "packages/merlin-experiments/tests/test_original_transpose_flow.py",
        ),
        "support_files": (
            "packages/merlin-experiments/tests/original_transpose_fixtures.py",
            "packages/merlin-experiments/tests/original_reference_fixtures.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.original_transpose_sources",
            "merlin.targetgen.original_transpose_reference",
            "merlin_experiments.phase0.original_reference_roster",
            "merlin_experiments.phase0.original_reference_standard_ir",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "host-ranked-descriptors": {
        "include_experiments": False,
        "tests_root": "merlin/tests/ir",
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("compiler-python", "llvm-llc", "clang"),
        "native_python_entries": ("compiler-python",),
        "native_sources": {"m2m": {"package": "m2m", "environment_key": "MERLIN_M2M_DIR"}},
        "native_test_files": ("test_host_descriptor_selection.py", "test_host_descriptor_native.py"),
        "native_test_cases": (
            *(
                ("test_host_descriptor_selection.py", method + suffix)
                for method, suffixes in (
                    ("test_original_complete_body_returns_and_storage_are_checked_before_lowering", ("",)),
                    (
                        "test_no_environment_default_duplicate_or_scalar_alias",
                        tuple(f"[environment{i}]" for i in range(4)),
                    ),
                    ("test_only_actual_calling_marker_may_differ_in_prepared_source", ("",)),
                    ("test_join_requires_same_actual_object_runtime_link_environment_and_image", ("",)),
                    (
                        "test_complete_raw_translation_copy_joins_actual_object_input_without_path_or_pin_shape_assumption",
                        tuple(
                            f"[{name}]"
                            for name in ("None", "changed_raw", "changed_final", "missing_raw", "stale_raw_pin")
                        ),
                    ),
                    ("test_compile_host_uses_selected_original_and_own_result_before_loading", ("",)),
                    (
                        "test_complete_ordered_original_fields_use_observed_i32_indices",
                        ("[shape0-i64]", "[shape1-i32]"),
                    ),
                    (
                        "test_changed_partial_or_aliased_caller_objects_refuse_before_descriptor_allocation",
                        tuple(
                            f"[{name}]"
                            for name in (
                                "missing",
                                "reordered",
                                "dtype",
                                "shape",
                                "stride",
                                "capacity",
                                "alignment",
                                "overlap",
                                "scalar",
                                "address",
                            )
                        ),
                    ),
                    ("test_constructed_transport_and_scalar_trampoline_paths_cannot_invoke", ("",)),
                    (
                        "test_unsupported_selected_image_entry_scalar_or_trampoline_refuses_before_loading",
                        tuple(f"[{name}]" for name in ("image", "entry", "wrapper", "scalar", "trampoline")),
                    ),
                    (
                        "test_actual_selected_model_route_checks_selection_and_entry_before_and_after_call",
                        tuple(
                            f"[{name}-{when}]"
                            for name in ("drop", "replace", "function", "restype", "argtypes", "scalar", "trampoline")
                            for when in ("before", "during")
                        ),
                    ),
                    ("test_absent_selection_keeps_ordinary_host_scalar_call_and_lowering_kwargs", ("",)),
                )
                for suffix in suffixes
            ),
            *(
                (
                    "test_host_descriptor_native.py",
                    "test_ordinary_original_complete_outputs_guards_and_actual_descriptor_call[" + case + "]",
                )
                for case in ("rank_zero_signed_zero", "tail_wide_repeated_results")
            ),
        ),
        "tests": ("test_host_descriptor_selection.py", "test_host_descriptor_native.py"),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.llvmlower.compiled_layout_query",
            "merlin.llvmlower.descriptor_contract",
            "merlin.llvmlower.descriptor_wrapper",
            "merlin.llvmlower.descriptor_layout",
            "merlin.llvmlower.host_descriptor_compile",
            "merlin.llvmlower.host_descriptor_call",
            "merlin.llvmlower.kernel_backend",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "integer-scalar-correspondence": {
        "include_experiments": False,
        "tests_root": "merlin/tests/ir",
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("compiler-python", "llvm-llc", "clang"),
        "native_python_entries": ("compiler-python",),
        "native_test_files": ("test_integer_scalar_native.py",),
        "tests": ("test_integer_scalar_correspondence.py", "test_integer_scalar_native.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.llvmlower.integer_scalar_contract",
            "merlin.llvmlower.integer_scalar_correspondence",
            "merlin.llvmlower.llvm_dialect_product",
        ),
        "required_modules": ("xdsl", "numpy"),
    },
    "capture-contraction-formats": {
        "include_experiments": False,
        "include_analysis": True,
        "tests_root": "merlin/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": (
            "targetgen/test_contraction_formats.py",
            "dse/test_capture_format_freezing.py",
            "dse/test_capture_format_policy.py",
        ),
        "tests": (
            "targetgen/test_contraction_formats.py",
            "dse/test_capture_format_freezing.py",
            "dse/test_capture_format_policy.py",
        ),
        "support_files": ("fixtures/capture_contraction_inputs.py",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.capture.contraction_formats",
            "merlin.compare.paper_measurement_freeze",
            "merlin.compare.freeze",
            "merlin.compare.capture_format_policy",
            "merlin.compare.paper",
            "merlin.compare.capture_workflow",
            "merlin.compare.cli",
        ),
        "required_modules": ("xdsl", "numpy", "yaml"),
    },
    "coherent-measurement": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("clang", "mlir-translate", "riscv-gcc", "readelf", "cpu-simulator"),
        "native_test_files": ("test_component_coherent_measurement.py", "test_component_measurement_execution.py"),
        "test_environment_record": "MERLIN_TEST_MEASUREMENT_ENV_SELECTION",
        "test_environment_defaults": {"MERLIN_TARGET_PATH": ""},
        "tests": (
            "test_component_coherent_measurement.py",
            "test_component_measurement_execution.py",
            "test_component_decode_products.py",
        ),
        "support_files": (
            "coherent_measurement_control.py",
            "coherent_measurement_runner.py",
            "measurement_execution_control.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.perf.component_coherent_measurement",
            "merlin_experiments.phase2.component_measurement_execution",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "serial-llvm-products": {
        "include_experiments": False,
        "tests_root": "merlin/tests/ir",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("compiler-python", "llvm-llc", "clang"),
        "native_python_entries": ("compiler-python",),
        "native_test_files": ("test_serial_llvm_dialect_product.py",),
        "tests": (
            "test_serial_llvm_dialect_product.py",
            "test_host_llc_selection.py",
            "test_ir_audit.py",
            "test_lowering_recipe.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.llvmlower.llvm_dialect_product",
            "merlin.llvmlower.pipeline",
            "merlin.llvmlower.lower",
            "merlin.llvmlower.kernel_backend",
        ),
        "required_modules": ("xdsl", "numpy"),
    },
    "original-pointwise-host": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("compiler-python", "mlir-translate", "llvm-llc"),
        "native_python_entries": ("compiler-python",),
        "native_sources": {"m2m": {"package": "m2m", "environment_key": "MERLIN_M2M_DIR"}},
        "native_test_files": ("test_component_original_pointwise_execution.py",),
        "native_test_cases": tuple(
            (
                "test_component_original_pointwise_execution.py",
                "test_original_pointwise_ordinary_host_values_and_source_applicability[" + member + "]",
            )
            for member in (
                "scalar_relu",
                "scalar_round",
                "scalar_integer",
                "scalar_integer_clamp",
                "round_tail",
                "clamp_rectangle",
                "integer_rectangle",
            )
        ),
        "tests": ("test_component_source_applicability.py", "test_component_original_pointwise_execution.py"),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase1.component_source_applicability",
            "merlin.llvmlower.kernel_backend",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "original-candidate-members": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("compiler-python", "mlir-translate", "llvm-llc", "clang", "bwrap", "firtool"),
        "native_python_entries": ("compiler-python",),
        "native_sources": {"m2m": {"package": "m2m", "environment_key": "MERLIN_M2M_DIR"}},
        "native_test_files": (
            "test_component_original_members.py",
            "test_component_original_member_transport.py",
            "test_component_original_member_host.py",
            "test_component_qualification.py",
            "test_component_runtime_support.py",
        ),
        "tests": (
            "test_component_original_members.py",
            "test_component_original_member_transport.py",
            "test_component_original_member_host.py",
            "test_component_qualification.py",
            "test_component_runtime_support.py",
        ),
        "support_files": ("test_component_original_pointwise_execution.py", "test_component_experiment.py"),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.common.access",
            "merlin.targetgen.native_component_inputs",
            "merlin.targetgen.native_component_execution",
            "merlin_experiments.phase1.component_original_members",
            "merlin_experiments.phase1.component_qualification_members",
            "merlin_experiments.phase1.component_qualification",
            "merlin_experiments.phase2.component_runtime_support",
        ),
        "required_modules": ("xdsl", "jsonschema", "numpy"),
    },
    "source-preparation-qualification": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "native_tools": ("clang",),
        "native_test_files": (
            "test_source_preparation_qualification.py",
            "test_component_qualification.py",
            "test_source_preparation_release.py",
        ),
        "tests": (
            "test_source_preparation_qualification.py",
            "test_component_qualification.py",
            "test_source_preparation_release.py",
        ),
        "support_files": (
            "test_source_requirement_ledger.py",
            "test_component_source_binding.py",
            "test_component_automatic.py",
            "test_component_generation.py",
            "test_component_minimal_spec.py",
            "test_component_execution_budget.py",
            "test_independent_rtl_intake.py",
            "test_component_coverage.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase1.component_qualification",
            "merlin_experiments.phase1.component_qualification_domain",
            "merlin_experiments.phase1.component_qualification_evidence",
            "merlin_experiments.phase0.source_preparation_release",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "source-preparation-release": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "tests": (
            "test_source_preparation_release.py",
            "test_source_preparation_semantic_join.py",
            "test_component_origin.py",
            "test_fresh_phase1_generation_budget.py",
        ),
        "support_files": (
            "test_source_requirement_ledger.py",
            "test_component_source_binding.py",
            "test_component_automatic.py",
            "test_component_generation.py",
            "test_component_minimal_spec.py",
            "test_component_execution_budget.py",
            "test_independent_rtl_intake.py",
            "test_component_coverage.py",
            "test_original_semantic_review.py",
            "test_original_reference_standard_ir.py",
            "test_original_reference_requirement_join.py",
            "original_reference_fixtures.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.source_preparation_release",
            "merlin_experiments.phase1.component_generation_admission",
            "merlin_experiments.phase1.component_origin",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "development-measurement-scheduling": {
        "tests_root": ".",
        "collect_selected_tests": True,
        "tests": (
            "merlin/tests/targetgen/test_component_measurement_plan.py",
            "packages/merlin-experiments/tests/test_component_measurement_scheduling.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.perf.component_measurement_plan",
            "merlin_experiments.phase2.component_measurement_scheduling",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "compiled-static-features": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "tests": (
            "merlin/tests/targetgen/test_compiled_static_features.py",
            "packages/merlin-experiments/tests/test_component_compiled_features.py",
        ),
        "support_files": ("packages/merlin-experiments/tests/compiled_feature_control.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.perf.compiled_static_features",
            "merlin_experiments.phase2.component_compiled_features",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "typed-index-ranges": {
        "tests_root": ".",
        "collect_selected_tests": True,
        "source_inputs": (
            "examples/*/target/docs/*.md",
            "examples/*/target/examples/*.mlir",
            "examples/*/target/contracts/*.yaml",
            "examples/*/target/evidence_concepts.yaml",
        ),
        "tests": (
            "merlin/tests/targetgen/test_hw_index_ranges.py",
            "packages/merlin-experiments/tests/test_index_range_intake.py",
            "merlin/tests/targetgen/test_targetgen_toy.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_index_ranges",
            "merlin_experiments.phase0.index_range_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "original-semantic-cases": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "tests": (
            "packages/merlin-experiments/tests/test_original_semantic_review.py",
            "packages/merlin-experiments/tests/test_original_pointwise_semantic_review.py",
            "merlin/tests/targetgen/test_original_reference_stress.py",
            "merlin/tests/targetgen/test_original_operator_reference.py",
        ),
        "support_files": (
            "packages/merlin-experiments/tests/test_original_pointwise_reference_flow.py",
            "packages/merlin-experiments/tests/original_pointwise_reference_fixtures.py",
            "packages/merlin-experiments/tests/test_original_reference_requirement_join.py",
            "packages/merlin-experiments/tests/test_original_reference_standard_ir.py",
            "packages/merlin-experiments/tests/original_reference_fixtures.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.original_semantic_review",
            "merlin_experiments.phase0.original_semantic_review_plan",
            "merlin_experiments.phase0.original_pointwise_semantic_probes",
            "merlin_experiments.phase0.original_pointwise_stress_observer",
            "merlin.targetgen.original_pointwise_stress",
            "merlin.targetgen.original_operator_reference",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "typed-array-inputs": {
        "tests_root": ".",
        "collect_selected_tests": True,
        "source_inputs": (
            "examples/*/target/docs/*.md",
            "examples/*/target/examples/*.mlir",
            "examples/*/target/contracts/*.yaml",
            "examples/*/target/evidence_concepts.yaml",
        ),
        "tests": (
            "merlin/tests/targetgen/test_hw_array_selection.py",
            "packages/merlin-experiments/tests/test_transition_array_connectivity.py",
            "packages/merlin-experiments/tests/test_transition_connectivity_intake.py",
            "merlin/tests/targetgen/test_targetgen_toy.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_array_selection",
            "merlin.targetgen.rtl.hw_transition_connectivity",
            "merlin_experiments.phase0.transition_connectivity_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "integer-comparison-inputs": {
        "tests_root": ".",
        "collect_selected_tests": True,
        "source_inputs": (
            "examples/*/target/docs/*.md",
            "examples/*/target/examples/*.mlir",
            "examples/*/target/contracts/*.yaml",
            "examples/*/target/evidence_concepts.yaml",
        ),
        "tests": (
            "merlin/tests/targetgen/test_hw_integer_comparisons.py",
            "merlin/tests/targetgen/test_hw_combinational_observations.py",
            "merlin/tests/targetgen/test_targetgen_toy.py",
            "packages/merlin-experiments/tests/test_transition_connectivity_intake.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_combinational",
            "merlin.targetgen.rtl.hw_transition_connectivity",
            "merlin_experiments.phase0.transition_connectivity_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "original-reference-flow": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "tests": (
            "test_declared_original_reference_flow.py",
            "test_original_reference_requirement_join.py",
        ),
        "support_files": (
            "test_declared_phase0_run.py",
            "original_reference_fixtures.py",
            "test_original_reference_standard_ir.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.original_reference_flow",
            "merlin_experiments.phase0.original_reference_requirements",
            "merlin_experiments.phase0.declared_run",
            "merlin_experiments.phase0.source_requirement_ledger",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "independent-feature-arms": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "tests": ("test_component_feature_arms.py",),
        "support_files": ("feature_arm_control.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase2.component_feature_inputs",
            "merlin_experiments.phase2.component_feature_arms",
            "merlin_experiments.phase2.component_analytical",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "transition-connectivity": {
        "tests_root": ".",
        "collect_selected_tests": True,
        "source_inputs": (
            "examples/*/target/docs/*.md",
            "examples/*/target/examples/*.mlir",
            "examples/*/target/contracts/*.yaml",
            "examples/*/target/evidence_concepts.yaml",
        ),
        "tests": (
            "packages/merlin-experiments/tests/test_transition_connectivity_intake.py",
            "merlin/tests/targetgen/test_targetgen_toy.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_transition_connectivity",
            "merlin_experiments.phase0.transition_connectivity_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "recorded-process-consumption": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_tools": ("clang", "readelf"),
        "native_test_files": ("test_prepared_process_readback.py",),
        "tests": (
            "test_recorded_process_execution.py",
            "test_explicit_execution_service.py",
            "test_prepared_process_readback.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.contract.process_execution",
            "merlin.targetgen.contract.prepared_process_readback",
            "merlin.targetgen.contract.execution_service",
            "merlin.targetgen.contract.compile",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "compiler-library-route": {
        "tests_root": ".",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "tests": (
            "merlin/tests/targetgen/test_compiler_library.py",
            "merlin/tests/targetgen/test_compile_only_transport.py",
            "packages/merlin-experiments/tests/test_component_compile_role_transport.py",
            "packages/merlin-experiments/tests/test_component_runtime_support.py",
            "packages/merlin-experiments/tests/test_component_origin.py",
            "packages/merlin-experiments/tests/test_component_copy_proof.py",
            "packages/merlin-experiments/tests/test_component_library_execution.py",
        ),
        "support_files": (
            "packages/merlin-experiments/tests/test_component_compile_sources.py",
            "packages/merlin-experiments/tests/test_independent_software_intake.py",
            "packages/merlin-experiments/tests/test_component_semantic_basis.py",
            "packages/merlin-experiments/tests/test_independent_rtl_intake.py",
            "packages/merlin-experiments/tests/test_component_coverage.py",
            "packages/merlin-experiments/tests/test_component_generation.py",
            "packages/merlin-experiments/tests/test_component_minimal_spec.py",
            "packages/merlin-experiments/tests/test_phase0_freeze.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.compiler_library",
            "merlin.targetgen.compile_only_execution",
            "merlin.targetgen.native_component_execution",
            "merlin_experiments.phase1.component_origin",
            "merlin_experiments.phase2.component_runtime_support",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "functional-callback-selection": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": (
            "test_explicit_execution_service.py",
            "test_linked_admission_selection.py",
        ),
        "tests": (
            "test_explicit_execution_service.py",
            "test_linked_admission_selection.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.contract.execution_service",
            "merlin.targetgen.contract.elf_admission",
            "merlin.targetgen.contract.compile",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "hw-discovery-inputs": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_hw_discovery_graph.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.targetgen.rtl.hw_graph",),
        "required_modules": ("xdsl",),
    },
    "original-standard-ir-inputs": {
        "tests": ("test_original_reference_standard_ir.py",),
        "support_files": ("original_reference_fixtures.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.original_reference_standard_ir",
            "merlin_experiments.phase0.original_standard_ir_plan",
            "merlin_experiments.phase0.original_standard_ir_products",
            "merlin_experiments.phase0.original_standard_ir_observer",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "renderer-source-owner": {
        "include_experiments": False,
        "tests_root": "merlin/tests/infra",
        "tests": ("test_build_only_service.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.targetgen.contract.build_service",),
        "required_modules": ("xdsl",),
    },
    "renderer-provider-inputs": {
        "tests": (
            "test_native_model_execution.py",
            "test_native_readback.py",
            "test_numerical_readback.py",
            "test_phase1_codegen_scalability.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": ("merlin.targetgen.native_model_execution",),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "address-transition-inputs": {
        "native_tools": ("firtool", "circt-opt"),
        "native_test_files": ("test_address_transition_intake.py",),
        "tests": ("test_address_transition_intake.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_address_transitions",
            "merlin_experiments.phase0.address_transition_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "logical-source-demand": {
        "include_experiments": False,
        "tests_root": "merlin/tests/dse",
        "tests": ("test_component_source_demand.py",),
        "core_extras": (),
        "probe_modules": ("merlin.perf.component_source_demand",),
        "required_modules": (),
    },
    "source-demand-inputs": {
        "native_tools": ("firtool",),
        "native_test_files": ("test_component_source_demand_contracts.py",),
        "tests": ("test_component_source_demand_contracts.py",),
        "test_fixture_imports": True,
        "support_files": (
            "test_component_source_performance.py",
            "test_component_source_binding.py",
            "test_component_automatic.py",
            "test_component_generation.py",
            "test_component_minimal_spec.py",
            "test_component_coverage.py",
            "test_component_execution_budget.py",
            "test_independent_rtl_intake.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.perf.component_source_demand",
            "merlin_experiments.phase0.component_source_demand",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "hierarchical-memory-inputs": {
        "native_tools": ("firtool", "circt-opt"),
        "native_test_files": ("test_hierarchical_memory_intake.py",),
        "tests": ("test_hierarchical_memory_intake.py", "test_memory_port_intake.py"),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_hierarchy_bindings",
            "merlin_experiments.phase0.hierarchical_memory_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "original-schema-batch": {
        "tests": ("test_original_reference_transfer.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.original_schema_batch",
            "merlin.targetgen.torch_schema_batch_observer",
            "merlin_experiments.phase0.original_schema_defaults",
            "merlin_experiments.phase0.original_reference_roster",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "source-performance-inputs": {
        "native_tools": ("firtool",),
        "native_test_files": (
            "test_component_source_performance.py",
            "test_declared_source_performance.py",
        ),
        "tests": ("test_component_source_performance.py", "test_declared_source_performance.py"),
        "test_fixture_imports": True,
        "support_files": (
            "test_declared_phase0_run.py",
            "test_component_source_binding.py",
            "test_component_automatic.py",
            "test_component_generation.py",
            "test_component_minimal_spec.py",
            "test_component_coverage.py",
            "test_component_execution_budget.py",
            "test_independent_rtl_intake.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.component_source_performance",
            "merlin_experiments.phase0.declared_run",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "memory-port-inputs": {
        "native_tools": ("firtool", "circt-opt"),
        "native_test_files": ("test_memory_port_intake.py",),
        "tests": ("test_memory_port_intake.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_memory_ports",
            "merlin_experiments.phase0.memory_port_intake",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "original-pointwise": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "tests": (
            "test_original_pointwise_sources.py",
            "test_original_pointwise_reference.py",
            "test_original_pointwise_stress.py",
        ),
        "core_extras": (),
        "probe_modules": (
            "merlin.targetgen.original_pointwise_sources",
            "merlin.targetgen.original_pointwise_reference",
            "merlin.targetgen.original_pointwise_stress",
        ),
        "required_modules": (),
    },
    "original-reference-roster": {
        # Full native controls separately require explicit framework/source
        # selections; clean packaging environments must not discover them.
        "tests": ("test_original_reference_transfer.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.original_reference_roster",
            "merlin_experiments.phase0.original_reference_plan",
            "merlin_experiments.phase0.original_reference_products",
            "merlin_experiments.phase0.original_reference_observer",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "source-requirement-ledger": {
        "tests": ("test_source_requirement_ledger.py", "test_declared_phase0_run.py"),
        "test_fixture_imports": True,
        "support_files": (
            "test_component_source_binding.py",
            "test_component_automatic.py",
            "test_component_generation.py",
            "test_component_minimal_spec.py",
            "test_component_coverage.py",
            "test_component_execution_budget.py",
            "test_independent_rtl_intake.py",
        ),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase0.declared_run",
            "merlin_experiments.phase0.source_requirement_ledger",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "original-reference": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_original_operator_reference.py",),
        "core_extras": (),
        "probe_modules": (
            "merlin.targetgen.original_operator_reference",
            "merlin.targetgen.original_reference_values",
        ),
        "required_modules": (),
    },
    "combinational-observations": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "native_tools": ("firtool", "iverilog", "vvp"),
        "native_test_files": ("test_hw_combinational_observations.py", "test_hw_instance_inputs.py"),
        "tests": (
            "test_hw_combinational_observations.py",
            "test_hw_instance_inputs.py",
            "test_hw_input_observations.py",
            "test_hw_packing.py",
            "test_hw_field_ownership.py",
            "test_mlir_source_admission.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.rtl.hw_combinational",
            "merlin.targetgen.rtl.hw_instance_inputs",
            "merlin.targetgen.rtl.hw_graph",
            "merlin.targetgen.contract.mlir_source_admission",
        ),
        "required_modules": ("xdsl",),
    },
    "direct-kernel-counters": {
        "include_experiments": False,
        "tests_root": "merlin/tests/runtime",
        "native_tools": ("clang",),
        "native_test_files": (
            "test_direct_kernel_counter.py",
            "test_direct_kernel_invocation.py",
            "test_direct_kernel_phases.py",
        ),
        "tests": (
            "test_direct_kernel_counter.py",
            "test_direct_kernel_invocation.py",
            "test_direct_kernel_harness.py",
            "test_direct_kernel_phases.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.runtime.direct_kernel_counter",
            "merlin.runtime.direct_kernel_harness",
            "merlin.runtime.direct_kernel_phases",
        ),
        "required_modules": ("xdsl",),
    },
    "emitted-dataflow": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "native_tools": ("mlir-opt",),
        "native_test_files": ("test_emitted_control_flow.py",),
        "tests": (
            "test_emitted_control_flow.py",
            "test_emitted_dataflow.py",
            "test_emitted_control_flow_bounds.py",
            "test_mlir_source_admission.py",
            "test_pointer_entry_abi.py",
            "test_source_observation.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.contract.emitted_control_flow",
            "merlin.targetgen.contract.emitted_dataflow",
            "merlin.targetgen.contract.mlir_source_admission",
            "merlin.targetgen.contract.source_observation",
        ),
        "required_modules": ("xdsl",),
    },
    "source-observation-context": {
        "native_tools": ("firtool",),
        "native_test_files": ("test_component_runtime_support.py",),
        "tests": (
            "test_component_runtime_source_selection.py",
            "test_component_runtime_support.py",
            "test_component_runtime_copy_controls.py",
            "test_component_runtime_stage_products.py",
            "test_component_decode_products.py",
        ),
        "test_fixture_imports": True,
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase2.component_runtime_source_selection",
            "merlin_experiments.phase2.component_runtime_support",
            "merlin.targetgen.contract.source_observation",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
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
    "selected-pin-replay": {
        "include_experiments": False,
        "tests_root": "merlin/tests/infra",
        "collect_selected_tests": True,
        "native_tools": ("circt-opt",),
        "native_tool_path_lists": {"MERLIN_TEST_PIN_REPLAY_TOOLS": ("circt-opt",)},
        "native_test_files": ("test_selected_pin_replay.py",),
        "tests": ("test_selected_pin_replay.py", "test_invocation_record.py"),
        "core_extras": (),
        "probe_modules": ("merlin.common.selected_pin_replay", "merlin.common.invocation_record"),
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
            "test_component_automatic_unified.py",
            "test_component_original_sources.py",
            "test_component_pointwise_sources.py",
            "test_declared_phase0_run.py",
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
            "test_component_runtime_source_selection.py",
            "test_component_runtime_stage_products.py",
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
            "test_component_original_pointwise_execution.py",
            "test_component_launch.py",
            "test_component_launch_authority.py",
            "test_component_analytical.py",
            "test_component_screening.py",
            "test_component_cost_interval_states.py",
            "test_component_held_group_partition.py",
            "test_component_unknown_coverage.py",
            "test_component_training_membership.py",
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
            "merlin_experiments.phase2.component_runtime_stage_products",
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
            "merlin_experiments.phase0.original_schema_defaults",
            "merlin_experiments.phase0.original_call_sources",
            "merlin_experiments.phase0.declared_run",
            "merlin.targetgen.frontend_original_call",
            "merlin.targetgen.original_operator_sources",
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
        "native_test_files": ("runtime/test_direct_kernel_invocation.py", "runtime/test_direct_kernel_phases.py"),
        "tests_root": "merlin/tests",
        "tests": (
            "runtime/test_out_b64.py",
            "runtime/test_direct_kernel_harness.py",
            "runtime/test_direct_kernel_invocation.py",
            "runtime/test_direct_kernel_phases.py",
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
            "merlin.runtime.direct_kernel_phases",
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
    "formal-instruction-selection": {
        "tests_root": "packages/merlin-experiments/tests",
        "test_fixture_imports": True,
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": (
            "test_formal_instruction_selection.py",
            "test_private_linked_elf_selection.py",
            "test_phase1_formal.py",
            "test_phase1_formal_handoff.py",
        ),
        "tests": (
            "test_formal_instruction_selection.py",
            "test_private_linked_elf_selection.py",
            "test_phase1_formal.py",
            "test_phase1_formal_handoff.py",
        ),
        "support_files": ("test_phase1_feedback.py", "phase1_feedback_fixtures.py"),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase1.feedback.formal",
            "merlin_experiments.phase1.feedback.private_instruction_declaration",
            "merlin_experiments.phase1.feedback.private_instruction_coordinator",
        ),
        "required_modules": ("xdsl", "jsonschema"),
    },
    "automatic-linked-elf-policy": {
        "tests_root": "packages/merlin-experiments/tests",
        "tests": ("test_private_linked_elf_selection.py",),
        "core_extras": ("xdsl", "targetgen"),
        "probe_modules": (
            "merlin_experiments.phase1.feedback.private_full_models",
            "merlin_experiments.phase1.feedback.private_linked_elf_selection",
        ),
        "required_modules": ("xdsl", "jsonschema"),
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
            "ir/test_device_build_arguments.py",
            "ir/test_device_routed_roster.py",
            "infra/test_device_abi_resources.py",
            "ir/test_bucketize_source.py",
            "targetgen/test_host_capability_evidence.py",
            "targetgen/test_bucketize_host_source_body.py",
            "targetgen/test_f32_maximum_host_source_body.py",
            "runtime/test_whole_model_device_offload.py",
            "runtime/test_device_linked_elf_admission.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.compile.scalar_host_qualification",
            "merlin.llvmlower.c_runtime",
            "merlin.runtime.backends.spike_model",
            "merlin.llvmlower.link_supplier_trace",
            "merlin.llvmlower.device_build",
            "merlin.llvmlower.device_offload",
            "merlin.targetgen.contract.elf_admission",
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


def retain_archived_test_sources(tests, names, output):
    """Pin exact copied test identities; production origin checks stay strict."""
    rows = []
    for name in dict.fromkeys(names):
        relative = Path(name.split("::", 1)[0])
        if relative.suffix != ".py" or not relative.stem.startswith("test_"):
            continue
        path = (tests / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(tests.resolve()):
            raise QualificationFailed("archived test identity escapes its selected root")
        rows.append({"module": ".".join(relative.with_suffix("").parts), "path": str(path), "sha256": digest(path)})
    path = output / "archived-test-sources.json"
    path.write_text(
        json.dumps({"schema": "merlin.installed_test_sources.v1", "files": rows}, sort_keys=True, indent=2) + "\n"
    )
    return path


def clean_environment():
    excluded = {"UV_OVERRIDE", "UV_EXCLUDE", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT"}
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MERLIN", "PYTHON", "AET_", "CHIA_")) and k not in excluded
    }


NATIVE_TOOL_ENVIRONMENT = {
    "bwrap": "MERLIN_TEST_BWRAP",
    "circt-opt": "MERLIN_TEST_CIRCT_OPT",
    "circt-opt-modern": "MERLIN_TEST_CIRCT_OPT_MODERN",
    "clang": "MERLIN_CLANG",
    "compiler-python": "MERLIN_COMPILER_PYTHON",
    "cpu-simulator": "MERLIN_TEST_STOCK_CPU_SIMULATOR",
    "firtool": "MERLIN_TEST_FIRTOOL",
    "firtool-modern": "MERLIN_TEST_FIRTOOL_MODERN",
    "iverilog": "MERLIN_TEST_IVERILOG",
    "mlir-opt": "MERLIN_TEST_MLIR_OPT",
    "mlir-translate": "MERLIN_MLIR_TRANSLATE",
    "llvm-llc": "MERLIN_LLVM_LLC",
    "operator-python": "MERLIN_TEST_TORCH_PYTHON",
    "readelf": "MERLIN_TEST_READELF",
    "riscv-gcc": "MERLIN_TEST_RISCV_GCC",
    "vvp": "MERLIN_TEST_VVP",
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
        if name in SUITES[suite].get("native_python_entries", ()):
            try:
                selected[name]["python_entry"] = _INPUTS.python_entry(path)
            except (OSError, ValueError) as exc:
                raise QualificationFailed(f"interpreter entry unavailable: {exc}") from exc
    if selected and set(selected) != set(admitted):
        raise QualificationFailed("selected native suite needs its complete explicit tool roster")
    return selected


def verify_native_tools(selected):
    try:
        for tool in selected.values():
            if digest(tool["path"]) != tool["sha256"]:
                raise QualificationFailed("selected native executable changed")
            if "python_entry" in tool and _INPUTS.python_entry(tool["selected_path"]) != tool["python_entry"]:
                raise QualificationFailed("selected interpreter entry or prefix changed")
    except (OSError, ValueError) as exc:
        raise QualificationFailed(f"selected native executable unavailable: {exc}") from exc


def capture_native_sources(suite, selections):
    """Closed caller-selected public source identities, not dependency authority."""
    admitted, selected = SUITES[suite].get("native_sources", {}), {}
    for selection in selections:
        name, separator, supplied = selection.partition("=")
        if not separator or name not in admitted or name in selected:
            raise QualificationFailed("unknown, duplicate or suite-inadmissible native source")
        path, separator, commit = supplied.rpartition("@")
        if not separator:
            raise QualificationFailed("native source needs an explicit path and full commit")
        try:
            identity = _INPUTS.source_record(path, commit, admitted[name]["package"], check_committed=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise QualificationFailed(f"native source unavailable: {exc}") from exc
        selected[name] = {"identity": identity, "environment_key": admitted[name]["environment_key"]}
    if selected and set(selected) != set(admitted):
        raise QualificationFailed("selected native suite needs its complete explicit source roster")
    return selected


def verify_native_inputs(report):
    verify_native_tools(report.get("native_tools", {}))
    selected_environment = report.get("native_test_environment")
    if selected_environment is not None:
        try:
            if digest(selected_environment["path"]) != selected_environment["sha256"]:
                raise QualificationFailed("selected native test environment changed")
        except OSError as exc:
            raise QualificationFailed("selected native test environment unavailable") from exc
    try:
        _INPUTS.verify_sources(report.get("native_sources", {}))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise QualificationFailed(f"selected native source changed or unavailable: {exc}") from exc
    selected_inputs = report.get("native_test_inputs")
    if selected_inputs:
        try:
            current = _INPUTS.test_inputs_record(selected_inputs["path"], selected_inputs["environment"])
        except (OSError, ValueError) as exc:
            raise QualificationFailed("selected native test inputs unavailable") from exc
        if current != selected_inputs:
            raise QualificationFailed("selected native test inputs changed")


def capture_native_test_inputs(suite, path):
    if path is None:
        return {}
    names = SUITES[suite].get("test_input_environment_keys", ())
    if not names or len(names) != len(set(names)):
        raise QualificationFailed("suite does not admit the selected native test inputs")
    try:
        return _INPUTS.test_inputs_record(path, names)
    except (OSError, ValueError) as exc:
        raise QualificationFailed("native test inputs require their complete closed suite mapping") from exc


def native_environment(report, *, suite=None):
    environment = {
        tool["environment_key"]: tool["selected_path"] if "python_entry" in tool else tool["path"]
        for tool in report.get("native_tools", {}).values()
    }
    environment.update(
        (source["environment_key"], source["identity"]["path"]) for source in report.get("native_sources", {}).values()
    )
    selected_inputs = report.get("native_test_inputs", {}).get("environment", {})
    if set(selected_inputs) & set(environment):
        raise QualificationFailed("native test inputs conflict with selected tools or sources")
    environment.update(selected_inputs)
    if suite is not None and report.get("native_tools"):
        selected = report["native_tools"]
        for key, names in SUITES[suite].get("native_tool_path_lists", {}).items():
            if key in environment or any(name not in selected for name in names):
                raise QualificationFailed("declared native tool list lacks its complete selections")
            environment[key] = json.dumps([selected[name]["path"] for name in names])
    return environment


def retain_test_environment(suite, output, environment, report):
    """Retain a declared child mapping for selected native replay, never recover one."""
    key = SUITES[suite].get("test_environment_record")
    if key is None:
        return dict(environment)
    if key in environment or "native_test_environment" in report:
        raise QualificationFailed("native test environment was already selected")
    selected = dict(environment)
    for name, value in SUITES[suite].get("test_environment_defaults", {}).items():
        if name in selected and selected[name] != value:
            raise QualificationFailed("native test environment conflicts with its declared default")
        selected[name] = value
    if any(type(name) is not str or type(value) is not str for name, value in selected.items()):
        raise QualificationFailed("native test environment requires declared string values")
    path = Path(output).resolve(strict=True) / "native-test-environment.json"
    selected[key] = str(path)
    # The caller owns this private output directory. The file contains actual
    # values, stays outside candidate inputs, and must not inherit broad modes.
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write((json.dumps(selected, sort_keys=True) + "\n").encode())
    report["native_test_environment"] = {"path": str(path), "sha256": digest(path), "environment_key": key}
    return selected


def native_test_report_required(suite, report):
    """A declared mandatory packaging roster is independent of tool selection."""
    policy = SUITES[suite].get("mandatory_test_report")
    if policy is not None and policy != "merlin.installed_mandatory_tests.v1":
        raise QualificationFailed("unknown mandatory test report policy")
    return policy is not None or bool(report["native_tools"])


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
    required_cases = configured.get("native_test_cases", ())
    if required_cases and (
        len(set(required_cases)) != len(required_cases)
        or {filename for filename, _ in required_cases} != set(required)
        or any(not name for _, name in required_cases)
    ):
        raise QualificationFailed("native suite has no closed declared member roster")
    cases = list(ET.parse(path).getroot().iter("testcase"))
    report["native_test_report"] = {"path": str(path), "sha256": digest(path)}
    report["native_test_files"] = list(required)
    report["native_zero_skip_scope"] = "declared_native_test_files"
    report["suite_test_counts"] = {"tests": len(cases), "skipped": 0}
    report["native_test_counts"] = {"tests": 0, "skipped": 0}
    report["other_test_counts"] = {"tests": 0, "skipped": 0}
    report["test_skips"] = {"native": [], "other": []}
    modules = {filename: Path(filename).with_suffix("").as_posix().replace("/", ".") for filename in admitted}
    observed, identities, observed_cases = set(), set(), set()
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
            if required_cases and (classname != module or (filename, name) in observed_cases):
                errors.append("testcase changed or repeated a declared top-level member identity")
            observed_cases.add((filename, name))
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
    if required_cases:
        report["native_test_cases"] = [list(member) for member in required_cases]
        report["missing_native_test_cases"] = [list(member) for member in sorted(set(required_cases) - observed_cases)]
        report["unexpected_native_test_cases"] = [
            list(member) for member in sorted(observed_cases - set(required_cases))
        ]
    if errors:
        raise QualificationFailed(errors[0])
    if report["missing_native_test_files"]:
        raise QualificationFailed("explicit native qualification did not execute every declared native test file")
    if required_cases and (report["missing_native_test_cases"] or report["unexpected_native_test_cases"]):
        raise QualificationFailed("explicit native qualification did not execute the exact original member roster")
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
        self.environment.update(native_environment(report, suite=report.get("suite")))

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def run(self, label, argv, cwd, *, stdout=None):
        verify_native_inputs(self.report)
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
        if (
            self.report.get("native_sources")
            or self.report.get("native_test_inputs")
            or any("python_entry" in tool for tool in self.report.get("native_tools", {}).values())
        ):
            try:
                verify_native_inputs(self.report)
            except QualificationFailed as exc:
                record.update(status="inputs_changed", error=str(exc))
                self.save()
                raise
        if record["status"] != "passed":
            raise QualificationFailed(f"{label}: {record['status']}; see {log}")


def projects(snapshot, extras, *, include_experiments=True, include_analysis=False):
    result = []
    roots = (Path("."), Path("packages/merlin-experiments")) if include_experiments else (Path("."),)
    if include_analysis:
        roots = (*roots, Path("packages/merlin-analysis"))
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


def qualify(
    root,
    output,
    commit,
    suite,
    timeout,
    *,
    requested_ref=None,
    invocation=None,
    native_tools=(),
    native_sources=(),
    test_inputs=None,
):
    own = Path(__file__).resolve()
    helper = own.with_name("installed_qualification_probe.py")
    input_helper = own.with_name("installed_native_inputs.py")
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
        "tool_sources": {str(p.relative_to(root)): digest(p) for p in (own, helper, input_helper)},
        "tool_source_note": "Actual executing bytes; hashes may include working-tree edits not in tooling_revision.",
        "selected_tests": list(SUITES[suite]["tests"]),
        "support_files": list(support_files),
        "test_fixture_imports": bool(SUITES[suite].get("test_fixture_imports")),
        "test_collection_policy": "selected_roster" if SUITES[suite].get("collect_selected_tests") else "copied_tree",
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
        "native_sources": {},
        "requested_native_sources": list(native_sources),
        "native_test_inputs": {},
        "native_test_input_policy": "Explicit closed suite input strings; no source, dependency or runtime authority.",
        "native_source_policy": (
            "Explicit package bytes and checkout identity only; import/dependency closure unproved."
        ),
        "native_test_files": list(SUITES[suite].get("native_test_files", ())),
        "native_tool_policy": "Explicit executable bytes only; system dependencies are not a frozen toolchain closure.",
        "test_fixture_retention": "all; unique qualification-owned external temp root",
    }
    runner = Recorder(output, report, timeout)
    runner.save()
    try:
        report["native_tools"] = capture_native_tools(suite, native_tools)
        report["native_sources"] = capture_native_sources(suite, native_sources)
        report["native_test_inputs"] = capture_native_test_inputs(suite, test_inputs)
        runner.environment.update(native_environment(report, suite=suite))
        report["native_environment"] = native_environment(report, suite=suite)
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
            *(
                ["packages/merlin-analysis/src", "packages/merlin-analysis/pyproject.toml"]
                if SUITES[suite].get("include_analysis")
                else []
            ),
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
            include_analysis=SUITES[suite].get("include_analysis", False),
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
        if source_inputs:
            input_root = str(tests / "source-inputs")
            runner.environment["MERLIN_TEST_SOURCE_INPUTS_ROOT"] = input_root
            report["source_input_root"] = input_root
        runner.save()
        shutil.copyfile(copied_helper, tests / "conftest.py")
        archived_tests = retain_archived_test_sources(tests, test_files, output)
        runner.environment["MERLIN_TEST_ARCHIVED_SOURCES"] = str(archived_tests)
        report["archived_test_sources"] = {"path": str(archived_tests), "sha256": digest(archived_tests)}
        runner.environment = retain_test_environment(suite, output, runner.environment, report)
        runner.save()
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
                *(
                    [
                        "-o",
                        "pythonpath="
                        + " ".join(dict.fromkeys((str(tests), *(str((tests / name).parent) for name in test_files)))),
                    ]
                    if SUITES[suite].get("test_fixture_imports")
                    else []
                ),
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
                    if native_test_report_required(suite, report)
                    else []
                ),
                *(
                    [tests / name for name in SUITES[suite]["tests"]]
                    if SUITES[suite].get("collect_selected_tests")
                    else [tests]
                ),
            ],
            external,
        )
        verify_native_inputs(report)
        if native_test_report_required(suite, report):
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
    parser.add_argument(
        "--native-source",
        action="append",
        default=[],
        metavar="NAME=ABSOLUTE_CHECKOUT@FULL_COMMIT",
        help="Explicit suite-admitted package checkout; pin tracked bytes and reject untracked package files",
    )
    parser.add_argument(
        "--test-inputs",
        metavar="ABSOLUTE_JSON_FILE",
        help="Explicit closed suite-declared native test input strings; pin and recheck before/after commands",
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
            native_sources=args.native_source,
            test_inputs=args.test_inputs,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
