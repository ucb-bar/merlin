"""Bind reduced-source comparisons to already verified complete-model artifacts.

This module selects neither a workload nor a runtime. It reuses exact cached
source/task proofs and the controller's compiler entrypoints; it never recompiles
or executes the full model while preparing a short witness.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def document_digest(value: Any) -> str:
    return text_digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def program_plan(command_buffer: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = (command_buffer.get("params") or {}).get("global_program_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("missing compiler source ownership plan")
    return plan


def source_owners(plan: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    owners = {}
    for task in plan["tasks"]:
        for index in task["source_op_indices"]:
            if type(index) is not int or index in owners:
                raise ValueError("ambiguous source ownership")
            owners[index] = task
    return owners


def artifact_record(artifact: Mapping[str, Any], *, baseline: bool = False) -> dict[str, Any]:
    return {"interface": Path(artifact["interface"]), "lowered_text": artifact["lowered_text"],
        "command_buffer": artifact["command_buffer"], "command_buffer_text": artifact["command_buffer_text"],
        "compiler_sha256": artifact["compiler_sha256" if baseline else "candidate_sha256"],
        "lowered_sha256": artifact["lowered_sha256" if baseline else "candidate_lowered_sha256"],
        "command_buffer_sha256": artifact["command_buffer_sha256" if baseline else "candidate_command_buffer_sha256"]}


def check_artifact_proof(artifact: Mapping[str, Any], proof: Mapping[str, Any],
                         source_sha: str, graph_sha: str) -> None:
    raw_buffer = artifact["command_buffer_text"]
    if (not isinstance(raw_buffer, str) or text_digest(raw_buffer) != artifact["command_buffer_sha256"]
            or document_digest(json.loads(raw_buffer)) != document_digest(artifact["command_buffer"])):
        raise ValueError("retained full command-buffer object differs from original hash-bound raw bytes")
    expected = {"candidate_sha256": artifact["compiler_sha256"], "source_sha256": source_sha,
        "candidate_lowered_sha256": artifact["lowered_sha256"],
        "candidate_command_buffer_sha256": artifact["command_buffer_sha256"],
        "logical_dispatch_digest": graph_sha,
        "plan_digest": document_digest(program_plan(artifact["command_buffer"]))}
    if (not isinstance(proof, Mapping) or proof.get("status") != "verified"
            or any(proof.get(key) != value for key, value in expected.items())
            or text_digest(artifact["lowered_text"]) != artifact["lowered_sha256"]):
        raise ValueError("missing or stale verified full-source ownership evidence")


@dataclass(frozen=True)
class SourceProgramPair:
    source: str
    source_sha256: str
    graph_sha256: str
    artifacts: Mapping[str, Mapping[str, Any]]
    owners: Mapping[str, Mapping[int, Mapping[str, Any]]]
    comparison_binding: Mapping[str, Any]
    compile_before: Callable[..., Any]


def bind_source_program_pair(*, candidate: Path, experiment: Any,
                             comparison_arm: str, max_source_bytes: int = 2_000_000,
                             portfolio_member: Mapping[str, Any] | None = None) -> SourceProgramPair:
    """Reuse both exact full-model proofs without granting numerical or timing authority.

    The caller owns the action deadline and must recheck the current controller
    binding after short compilation. An optimization baseline is never silently
    substituted with the preceding iteration, or vice versa.
    """
    if comparison_arm not in {"optimization_baseline", "previous"}:
        raise ValueError("select optimization_baseline or previous explicitly")
    if type(max_source_bytes) is not int or max_source_bytes <= 0:
        raise ValueError("source extraction requires a positive byte bound")
    selected = None
    if portfolio_member is not None:
        if comparison_arm != "previous":
            raise ValueError("portfolio source pairs currently require the preceding revision")
        if not callable(getattr(experiment, "selected_changed_portfolio_context", None)):
            raise ValueError("portfolio source pair requires member-aware experiment accessors")
        selected = experiment.selected_changed_portfolio_context(candidate, portfolio_member)
        before_context, after_context = selected["previous"], selected["current"]
        after_raw = after_context["artifacts"]
        diag = after_context["analysis"]["diagnostics"]
    else:
        after_raw = experiment.current_artifacts(candidate)
        diag = experiment.iterations[-1]["analysis"]["diagnostics"]
    graph_sha = diag["captured_logical_graph"]["logical_dispatch_digest"]
    if comparison_arm == "optimization_baseline":
        before_raw = experiment.optimization_baseline_artifacts(candidate)
        binding = experiment.optimization_baseline_artifact_binding(candidate)
        compile_before = experiment.compile_optimization_baseline_probe_candidate
        before_proof = diag.get("verified_baseline_global_plan_emission")
        proof_binding = diag.get("baseline_global_plan_evidence_binding")
        expected = {"source_sha256": binding["source_sha256"], "lowered_sha256": binding["lowered_sha256"],
            "command_buffer_sha256": binding["command_buffer_sha256"], "compiler_sha256": binding["compiler_sha256"],
            "host_verifier_policy_sha256": experiment.host_policy["sha256"],
            "evidence_sha256": document_digest(before_proof)}
        if (not isinstance(proof_binding, Mapping)
                or proof_binding.get("schema") != "baseline_global_plan_evidence_binding_v1"
                or any(proof_binding.get(key) != value for key, value in expected.items())):
            raise ValueError("optimization baseline lacks exact cached host-policy-bound plan proof; "
                             "run static proof preparation, not a probe-time full verification")
    else:
        if selected is None and len(experiment.iterations) < 2:
            raise ValueError("previous comparison requires a preceding analyzed iteration")
        before_raw = (before_context["artifacts"] if selected is not None
                      else experiment.previous_artifacts(candidate))
        binding = ({"schema": "selected_portfolio_source_program_pair_v1",
                    "selection": selected["selection"],
                    "previous": before_context["member_binding"],
                    "current": after_context["member_binding"]}
                   if selected is not None else experiment.previous_probe_binding(candidate).to_dict())
        compile_before = experiment.compile_previous_probe_candidate
        before_proof = ((before_context["analysis"].get("diagnostics") or {}).get(
            "verified_global_plan_emission") if selected is not None else
            experiment.iterations[-2]["analysis"]["diagnostics"].get("verified_global_plan_emission"))
    artifacts = {"before": artifact_record(before_raw, baseline=comparison_arm == "optimization_baseline"),
                 "after": artifact_record(after_raw)}
    source = artifacts["after"]["interface"].read_text()
    if len(source.encode()) > max_source_bytes or artifacts["before"]["interface"].read_text() != source:
        raise ValueError("source pair changed or exceeds bounded extraction input")
    source_sha = text_digest(source)
    for arm, proof in (("before", before_proof), ("after", diag.get("verified_global_plan_emission"))):
        check_artifact_proof(artifacts[arm], proof, source_sha, graph_sha)
    owners = {arm: source_owners(program_plan(artifact["command_buffer"]))
              for arm, artifact in artifacts.items()}
    return SourceProgramPair(source, source_sha, graph_sha, artifacts, owners, binding, compile_before)
