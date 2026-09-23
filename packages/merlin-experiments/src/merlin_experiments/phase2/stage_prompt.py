"""Canonical performance-stage prompt rendering from explicit frozen launch inputs."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from merlin.common.digest import sha256_bytes as _sha256
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS

from . import broker_policy as BP
from . import prompt as PP
from .claims import dispatch as CD
from .contracts import StageGateError


class E2ESentinel(Protocol):
    """Structural view of an existing frozen objective; no native type dependency."""

    @property
    def capsule(self) -> str: ...
    @property
    def capsule_path(self) -> str: ...
    @property
    def frozen_source_path(self) -> str: ...
    @property
    def capsule_sha256(self) -> str: ...
    @property
    def required_lanes(self) -> tuple[str, ...]: ...
    @property
    def required_tiers(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class PerformanceFamilyDeclaration:
    """Frozen family facts retained even when the shared prompt API is older."""

    family: str
    claim: str
    negative_control: str
    falsifier_observation: str
    differential_basis: str
    fitted_parameters: tuple[str, ...] = ()
    acceptance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StageHostLaneGrant:
    target: str
    package_id: str
    package_path: str
    package_sha256: str
    manifest_path: str
    integration_seam: str


@dataclass(frozen=True)
class StagePromptInputs:
    target: str
    approach: str
    functional_run_id: str
    functional_submission_sha256: str
    frozen_functional_path: str
    frozen_functional_sha256: str
    submission_path: str
    submission_initial_sha256: str
    functional_public_capsules: int
    functional_hidden_capsules: int
    functional_bundle_snapshot_manifest: str
    functional_bundle_snapshot_manifest_sha256: str
    functional_bundle_snapshot_sha256: str
    workload_root: str
    workload_manifest: str
    workload_manifest_sha256: str
    workload_capsules_sha256: str
    expected_cells: tuple[PP.PerfCell, ...]
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    formal_claim: Mapping[str, Any]
    smoke_replicates: int
    wall_budget_seconds: int
    rounds: int
    round_timeout_seconds: int
    max_tool_calls: int
    tool_timeout_seconds: int
    families: tuple[PerformanceFamilyDeclaration, ...]
    host_lane: StageHostLaneGrant
    e2e_sentinel: E2ESentinel
    tools: tuple[PP.ToolGrant, ...]
    allowed_paths: tuple[str, ...]
    execution_broker_path: str
    execution_broker_command: str
    broker_receipt_path: str


@dataclass(frozen=True)
class PromptArtifact:
    source_path: Path
    text: str
    sha256: str
    n_bytes: int


def load_prompt(path: Path) -> PromptArtifact:
    """Read the exact prompt artifact; no implicit default or strategy text is injected."""
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise StageGateError(f"an explicit, real performance prompt file is required: {raw}")
    path = raw.resolve()
    payload = path.read_bytes()
    if not payload or len(payload) > 2_000_000:
        raise StageGateError("performance prompt must be non-empty and at most 2 MB")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StageGateError("performance prompt must be UTF-8") from exc
    return PromptArtifact(path, text, _sha256(payload), len(payload))


def render_stage_prompt(inputs: StagePromptInputs) -> str:
    """Render through the current shared prompt, then bind richer stage facts."""
    for name, value in (
        ("replicates", inputs.replicates),
        ("smoke_replicates", inputs.smoke_replicates),
        ("wall_budget_seconds", inputs.wall_budget_seconds),
        ("rounds", inputs.rounds),
        ("round_timeout_seconds", inputs.round_timeout_seconds),
        ("max_tool_calls", inputs.max_tool_calls),
        ("tool_timeout_seconds", inputs.tool_timeout_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageGateError(f"performance prompt {name} must be a positive integer")
    if inputs.tool_timeout_seconds > ITERATION_MAX_SECONDS:
        raise StageGateError(
            f"performance prompt tool timeout exceeds the {ITERATION_MAX_SECONDS:g}s reduced-witness iteration limit"
        )
    if inputs.smoke_replicates >= inputs.replicates:
        raise StageGateError("smoke replicate count cannot masquerade as the formal cohort")
    expected_replicas = tuple(f"r{index:03d}" for index in range(inputs.replicates))
    if inputs.formal_replicate_identities != expected_replicas:
        raise StageGateError("formal replicate identities are not the exact canonical cohort")
    declaration = inputs.formal_claim.get("declaration")
    if not isinstance(declaration, Mapping) or inputs.formal_claim.get("status") != "READY":
        raise StageGateError("performance prompt formal claim is not preflight-ready")
    claim_family = str(inputs.formal_claim.get("family") or "")
    CD.verify_supported_acceptance(CD.declaration_module(declaration, claim_family), declaration, claim_family)
    if not inputs.e2e_sentinel.required_lanes or "L2" not in inputs.e2e_sentinel.required_tiers:
        raise StageGateError("performance prompt E2E objective lacks a declared lane and L2 screen")

    base_families = tuple(
        PP.PerfFamily(
            family.family,
            family.claim,
            family.negative_control,
            family.falsifier_observation,
            family.differential_basis,
            family.fitted_parameters,
        )
        for family in inputs.families
    )
    base_host = PP.HostLaneGrant(
        inputs.host_lane.target,
        inputs.host_lane.package_id,
        inputs.host_lane.package_path,
        inputs.host_lane.package_sha256,
        inputs.host_lane.manifest_path,
        inputs.host_lane.integration_seam,
    )
    base = PP.PerfPromptInputs(
        target=inputs.target,
        approach=inputs.approach,
        functional_run_id=inputs.functional_run_id,
        functional_submission_sha256=inputs.functional_submission_sha256,
        frozen_functional_path=inputs.frozen_functional_path,
        frozen_functional_sha256=inputs.frozen_functional_sha256,
        submission_path=inputs.submission_path,
        submission_initial_sha256=inputs.submission_initial_sha256,
        functional_public_capsules=inputs.functional_public_capsules,
        functional_hidden_capsules=inputs.functional_hidden_capsules,
        workload_root=inputs.workload_root,
        workload_manifest=inputs.workload_manifest,
        workload_manifest_sha256=inputs.workload_manifest_sha256,
        workload_capsules_sha256=inputs.workload_capsules_sha256,
        expected_cells=inputs.expected_cells,
        families=base_families,
        host_lane=base_host,
        tools=inputs.tools,
        allowed_paths=inputs.allowed_paths,
        execution_broker_path=inputs.execution_broker_path,
        execution_broker_command=inputs.execution_broker_command,
        broker_receipt_path=inputs.broker_receipt_path,
    )
    rendered = PP.render_initial_prompt(base).rstrip()
    family_acceptance = {
        family.family: copy.deepcopy(family.acceptance) for family in inputs.families if family.acceptance is not None
    }
    supplement = {
        "schema_version": 1,
        "formal_replicates": list(inputs.formal_replicate_identities),
        "formal_claim": copy.deepcopy(dict(inputs.formal_claim)),
        "smoke_replicates": inputs.smoke_replicates,
        "functional_bundle_snapshot": {
            "manifest": inputs.functional_bundle_snapshot_manifest,
            "manifest_sha256": inputs.functional_bundle_snapshot_manifest_sha256,
            "content_sha256": inputs.functional_bundle_snapshot_sha256,
        },
        "e2e_sentinel": {
            "capsule": inputs.e2e_sentinel.capsule,
            "capsule_path": inputs.e2e_sentinel.capsule_path,
            "frozen_source_path": inputs.e2e_sentinel.frozen_source_path,
            "capsule_sha256": inputs.e2e_sentinel.capsule_sha256,
            "required_lanes": list(inputs.e2e_sentinel.required_lanes),
            "required_tiers": list(inputs.e2e_sentinel.required_tiers),
        },
        "budgets": {
            "wall_budget_seconds": inputs.wall_budget_seconds,
            "rounds": inputs.rounds,
            "round_timeout_seconds": inputs.round_timeout_seconds,
            "max_tool_calls": inputs.max_tool_calls,
            "tool_timeout_seconds": inputs.tool_timeout_seconds,
        },
        "family_acceptance": family_acceptance,
    }
    return (
        rendered
        + "\n\n## Whole-model optimization loop (primary objective)\n\n"
        + (
            "Optimize the complete E2E sentinel and its global dataflow first. Capsules are reduced "
            "witnesses used to calibrate or refute a mechanism; they are not the objective and a "
            "capsule win is never evidence of an E2E win. At the start of each round, read "
            "`STAGE_CONTEXT.json` → `automatic_optimization_inventory`: it maps real Python AST "
            "symbols to the manifest commands that consume them and lists structurally verified, "
            "author-declared semantic edit surfaces. Use `inspect-optimization-surfaces` after source "
            "or manifest edits to refresh "
            "that inventory. A missing semantic mapping is UNKNOWN: declare the real surface in "
            "`manifest.yaml`, then let the host validate it; do not guess from a filename.\n\n"
            "Read `STAGE_CONTEXT.json` → `initial_whole_model_analysis` before choosing a lever, then "
            "invoke the required `analyze-whole-model` action once after the related compiler edits have "
            "stabilized; use the smallest affected witness while debugging those edits. Report-only edits "
            "may record that result without rerunning it, while any later compiler, manifest, or other "
            "execution-relevant edit requires a new final analysis. Both analysis paths emit "
            "frozen-baseline and live-candidate buffers for the fixed sentinel on the host, so there is "
            "no path substitution. "
            "Its `captured_logical_graph` is the complete unpruned dispatch graph, including scalar "
            "glue and constants, with source/dag digests and exact shared compiler entrypoints. "
            "`OutlinedGlobalPlanEmitter` can emit contiguous global fusion regions and prove the "
            "expanded before/after model IR structurally equivalent; integrate it in candidate code "
            "to expose multi-operation functions to target codegen. Its UNKNOWN cost is intentional: "
            "fewer function boundaries alone do not prove fewer physical transfers or faster cycles. "
            "The free command-buffer analysis reports declared representations, movement volume, "
            "barriers, placement boundaries, and structural findings. The whole-model action also "
            "decodes the lowered target artifact through the target-derived role table, lifts a "
            "program-scope CCA, runs the exact redundant-residency check, and joins each observed gap "
            "to declared AST edit surfaces. Read its `gap_coverage`: a row is not covered unless it has "
            "both evidence and a verified edit surface. If whole-model lowering is `declined` or the "
            "target stream is absent, fix that macro blocker and re-emit before spending a simulator "
            "call; an empty stream is UNKNOWN, never zero work. It intentionally leaves occupancy and "
            "contention UNKNOWN until an event adapter or warm counter establishes them. Then run the "
            "smallest shape preserving the same pressure/signature: one "
            "unmeasured warm invocation followed by exactly one measured invocation, capturing compute "
            "cycles and only the resource/movement counters needed to explain them. Invoke "
            f"`{BP.OCCUPANCY_PROFILE_ACTION}` when occupancy or overlap is the deciding UNKNOWN; its fixed "
            "witness and selection basis are recorded in `STAGE_CONTEXT.json` before candidate "
            "measurement. No simulator tool "
            f"may exceed {ITERATION_MAX_SECONDS:g}s. Never modify an evaluation or mixed-lane harness "
            "after seeing a candidate; keep the frozen declared E2E objective fixed. Never simulate the "
            "complete model or complete layer during search, even if it would finish quickly: only "
            "separate mechanism-equivalent probes calibrate costs. Full-size execution is an "
            "optional post-freeze validation, not a Phase-2 prerequisite. If FireSim is explicitly "
            "available and requested for that validation, use only the queue-owned `runworkload-full`; "
            "the queue must own exactly `firesim kill` -> `firesim infrasetup` -> "
            "`firesim runworkload` -> `firesim kill`, in that order.\n\n"
            "Use L3 sparsely: the broker permits at most one reduced occupancy profile and two tuning "
            "GSIM feedback calls per round. Treat the first tuning call, if used, as the sole exploratory "
            "promotion check and reserve the second for the exact final bytes. Iterate freely with "
            "`analyze-whole-model`, `analyze-command-buffers`, and "
            "`inspect-optimization-surfaces`; these launch no simulator.\n\n"
            "## Sealed authoring-stage supplement\n\n"
        )
        + (
            "The outer Codex control plane has only its isolated authentication mount. "
            "The inner execution plane has the live descriptor-derived toolchain, `--clearenv`, "
            "and no credentials. Network availability is not an isolation claim. The JSON below "
            "is immutable launch data; do not retune its acceptance rules after observing results.\n\n"
            "```json\n" + json.dumps(supplement, sort_keys=True, indent=2) + "\n```\n"
        )
    )


def materialize_canonical_prompt(inputs: StagePromptInputs, artifact_path: Path) -> PromptArtifact:
    """Render the sole accepted prompt, after every frozen launch fact is known."""
    text = render_stage_prompt(inputs)
    if not isinstance(text, str) or not text.strip():
        raise StageGateError("performance prompt renderer returned no instruction")
    payload = text.encode("utf-8")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as stream:
        stream.write(payload)
    return load_prompt(artifact_path)
