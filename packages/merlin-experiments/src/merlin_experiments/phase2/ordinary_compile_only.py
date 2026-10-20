"""Queued no-data scale compilation on the exact original Phase 1 profile.

This diagnostic job does not execute Spike/RTL or produce an objective score.
The ordinary retained revision, resource pool and final promotion gate stay the
owners of source custody, job lifetime and complete numerical/performance tests.
"""

from __future__ import annotations

from pathlib import Path

from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compile_only as ORIGINAL

from . import campaign as PC
from .contracts import StageGateError, mapping_file

KIND = "corpus-compile-only-feedback"
SCHEMA = "merlin.ordinary_compile_only_feedback.v1"


def selection(policy):
    environment = mapping_file(policy._functional.run_dir / "environment.yaml", yaml_file=True)
    original = PC.verify_private_input_snapshot(policy._functional.run_dir, environment)
    profile = original["bundle"].get("compiler_tool_profile")
    if profile is None:
        raise StageGateError("compile-only was not admitted by the original Phase 1 tool profile")
    if profile != environment.get("compiler_tool_profile"):
        raise StageGateError("compile-only profile differs from its original Phase 1 selection")
    return profile


def available(policy):
    # Only original admission decides availability; candidate manifests do not.
    environment = mapping_file(policy._functional.run_dir / "environment.yaml", yaml_file=True)
    return environment.get("compiler_tool_profile") is not None


def evaluate_retained(policy, revision, *, work_root, timeout_s):
    policy.validate_search_inputs()
    revision.verify(authority=policy.edit_authority)
    profile = selection(policy)
    environment = mapping_file(policy._functional.run_dir / "environment.yaml", yaml_file=True)
    original = PC.verify_private_input_snapshot(policy._functional.run_dir, environment)
    common_service = CommonTransformService(
        workspace=original["workspace"], bundle_manifest=original["bundle_path"], repo=original["repo"]
    )
    if common_service.profile != profile:
        raise StageGateError("compile-only service differs from the original Phase 1 owner")
    members = tuple(sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule)))
    capsules = [{**member.descriptor, "__dir__": str(member.source_dir)} for member in members]
    result = ORIGINAL.observe(
        candidate=revision.source,
        target_experiment=policy.target_experiment,
        capsules=capsules,
        contract=policy._contract_root,
        facts_path=policy.feedback_evaluator.rtl_facts_path,
        common_service=common_service,
        work_root=Path(work_root),
        timeout_s=min(int(timeout_s), 600),
    )
    revision.verify(authority=policy.edit_authority)
    if selection(policy) != profile:
        raise StageGateError("compile-only original tool profile changed during its job")
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "candidate_sha256": revision.source_sha256,
        "compiler_tool_profile_sha256": profile["sha256"],
        "tuning_corpus_sha256": policy.search_corpus.capsules_sha256,
        "feedback": result,
        "numeric": "not_attempted",
        "rtl": "not_attempted",
        "timing_cycles": None,
        "promotion": "NONE",
    }


def validate_feedback(raw, *, policy, revision):
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "kind",
        "candidate_sha256",
        "compiler_tool_profile_sha256",
        "tuning_corpus_sha256",
        "feedback",
        "numeric",
        "rtl",
        "timing_cycles",
        "promotion",
    }:
        raise StageGateError("compile-only feedback has no original scoped result")
    profile = selection(policy)
    body = raw["feedback"]
    if (
        raw["schema"] != SCHEMA
        or raw["kind"] != KIND
        or raw["candidate_sha256"] != revision.source_sha256
        or raw["compiler_tool_profile_sha256"] != profile["sha256"]
        or raw["tuning_corpus_sha256"] != policy.search_corpus.capsules_sha256
        or raw["numeric"] != "not_attempted"
        or raw["rtl"] != "not_attempted"
        or raw["timing_cycles"] is not None
        or raw["promotion"] != "NONE"
        or not isinstance(body, dict)
        or body.get("schema") != ORIGINAL.SCHEMA
        or body.get("candidate_sha256") != revision.source_sha256
        or body.get("compiler_tool_profile_sha256") != profile["sha256"]
        or body.get("numeric") != "not_attempted"
        or body.get("rtl") != "not_attempted"
        or body.get("all_pass") is not False
        or body.get("timing_cycles") is not None
    ):
        raise StageGateError("compile-only result claims another compiler/profile or execution authority")
    if any(row.get("numeric") != "not_attempted" or row.get("rtl") != "not_attempted" for row in body.get("cases", [])):
        raise StageGateError("compile-only case cannot issue numerical or RTL credit")
    return raw
