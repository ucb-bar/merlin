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
from merlin.targetgen.generalization_prompt import GENERAL_COMPILER_CONTRACT_V1

from . import broker_policy as BP
from . import prompt as PP
from .broker import BrokerAction
from .broker_evidence import _is_sha256
from .claims import dispatch as CD
from .contracts import StageGateError
from .search_guidance import ADAPTIVE_SCOPE

QUEUED_AUTHOR_CONTINUATION = (
    "Continue the same queued compiler search with the existing candidate and session history. "
    "Read search-status through the original broker, work on its lowest-attainment eligible capsule first, "
    "and use exploratory_capsule_priorities when only fixed-model estimates are available. "
    "Keep that estimated ordering explicitly exploratory; it cannot satisfy the measured target. "
    "author and freeze the next reusable compiler revision, evaluate affected members, and collect complete feedback. "
    "Keep every numerical, source, evaluator and resource check active. Unknown or refused evidence is not zero cost. "
    "Do not finalize or select while the host target remains pending and exploration admission is open. "
    "Finalize and drain only when original search-status declares its target or budget stopping reason."
)


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
    e2e_sentinel: E2ESentinel | None
    tools: tuple[PP.ToolGrant, ...]
    allowed_paths: tuple[str, ...]
    execution_broker_path: str
    execution_broker_command: str
    broker_receipt_path: str
    author_agents: int | None = None
    compiler_tool_profile: Mapping[str, Any] | None = None
    additional_compiler_library: Mapping[str, Any] | None = None
    compiler_pass_preparation: dict[str, Any] | None = None
    capsule_development: Mapping[str, Any] | None = None
    author_guidance: Mapping[str, str] | None = None
    continue_queued_author: bool = False
    search_session: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ComponentPromptInputs:
    """Agent-facing component facts, without model or launch permissions.

    Paths are supplied by the trusted materialized-view owner. This renderer
    neither approves those paths nor establishes session or runtime isolation.
    """

    corpus_manifest_path: str
    corpus_manifest_sha256: str
    corpus_sha256: str
    members: tuple[tuple[str, str], ...]
    candidate_path: str
    candidate_sha256: str
    actions: tuple[BrokerAction, ...]
    allowed_paths: tuple[str, ...]
    wall_budget_seconds: int
    max_tool_calls: int
    tool_timeout_seconds: int
    broker_path: str
    broker_receipt_path: str
    public_manifest_sha256: str | None = None
    author_agents: int | None = None


def render_component_prompt(inputs: ComponentPromptInputs) -> str:
    """Render explicit component capabilities without the legacy model supplement."""
    for value in (inputs.corpus_manifest_sha256, inputs.corpus_sha256, inputs.candidate_sha256):
        if not _is_sha256(value):
            raise StageGateError("component prompt requires exact source and candidate identities")
    if inputs.public_manifest_sha256 is not None and not _is_sha256(inputs.public_manifest_sha256):
        raise StageGateError("component prompt public projection requires an exact identity")
    if (
        not inputs.members
        or len(set(inputs.members)) != len(inputs.members)
        or any(
            len(member) != 2 or any(not isinstance(item, str) or not item for item in member)
            for member in inputs.members
        )
    ):
        raise StageGateError("component prompt member set is invalid")
    for value in (inputs.wall_budget_seconds, inputs.max_tool_calls, inputs.tool_timeout_seconds):
        if type(value) is not int or value <= 0:
            raise StageGateError("component prompt budgets require positive integers")
    if inputs.tool_timeout_seconds > ITERATION_MAX_SECONDS:
        raise StageGateError("component prompt tool timeout exceeds the reduced-component iteration limit")
    names = {action.name for action in inputs.actions}
    from .component_workflow import component_actions, evaluation_pool_actions, search_actions

    permitted = {action.name for action in component_actions()}
    if inputs.author_agents is not None:
        if type(inputs.author_agents) is not int or not 1 <= inputs.author_agents <= 3:
            raise StageGateError("component search prompt requires bounded native author concurrency")
        selected = (*evaluation_pool_actions(search=True), *search_actions())
        permitted.update(action.name for action in selected)
        if not {action.name for action in selected}.issubset(names):
            raise StageGateError("search prompt lacks its fixed queue and source-selection controls")
    if len(names) != len(inputs.actions) or any(
        name not in permitted and not name.startswith("candidate-") for name in names
    ):
        raise StageGateError("component prompt carries an action outside its selected profile")
    if any(not isinstance(path, str) or not path for path in inputs.allowed_paths):
        raise StageGateError("component prompt path declarations are invalid")
    if not {
        inputs.corpus_manifest_path,
        inputs.candidate_path,
        inputs.broker_path,
        inputs.broker_receipt_path,
    }.issubset(inputs.allowed_paths):
        raise StageGateError("component prompt paths do not belong to its declared view")
    document = {
        "schema": "merlin.component_prompt.v1",
        "workflow_id": BP.COMPONENT_ONLY_V1,
        "corpus": {
            "manifest": inputs.corpus_manifest_path,
            "manifest_sha256": inputs.corpus_manifest_sha256,
            **(
                {"public_manifest_sha256": inputs.public_manifest_sha256}
                if inputs.public_manifest_sha256 is not None
                else {}
            ),
            "sha256": inputs.corpus_sha256,
            "members": [{"family": family, "capsule": capsule} for family, capsule in inputs.members],
        },
        "candidate": {"path": inputs.candidate_path, "initial_sha256": inputs.candidate_sha256},
        "actions": [action.advertised() for action in inputs.actions],
        "allowed_paths": list(inputs.allowed_paths),
        "broker": {"path": inputs.broker_path, "receipt_path": inputs.broker_receipt_path},
        "budgets": {
            "wall_seconds": inputs.wall_budget_seconds,
            "max_calls": inputs.max_tool_calls,
            "tool_seconds": inputs.tool_timeout_seconds,
        },
        "launch_admission": "REQUIRED_SEPARATELY",
        "final_acceptance": "NOT_ESTABLISHED",
        **({"author_agents": inputs.author_agents} if inputs.author_agents is not None else {}),
    }
    action_inventory = "\n".join(
        f"- `{action.name}`: {action.purpose} "
        + ("AVAILABLE" if action.available else f"UNAVAILABLE: {action.unavailable_reason}")
        + ("; invocation required by this launch." if action.required else "; optional.")
        for action in inputs.actions
    )
    search_plan = (
        "## Bounded parallel search\n\n"
        f"Coordinate up to {inputs.author_agents} native authors on separate compiler symbols or files. "
        "Only the coordinator invokes the broker and merges their edits into this candidate. "
        "Authors propose reusable changes and return their hypothesis and scope; do not launch private "
        "simulators or write outside the candidate. Serialize edits to shared symbols. Freeze each "
        "integrated contender with revision-freeze, declaring hypothesis and edit_family; evaluations "
        "read that retained revision while authors may continue editing. Use analytical feedback for "
        "screening and measure the unchanged starting compiler once for the baseline fallback. Use predictions for "
        "screening only. Complete measured simulator cycles govern the conservative development "
        "ranking; complete host and whole-model costs remain unavailable when unmeasured. "
        "Use search-status to prioritize decisions and respect its stopping reason. Intermediate "
        "automatic-session checkpoints return a FINAL turn message without finalizing the search. "
        "At final campaign completion, join every author and stop editing, "
        "invoke search-finalize to finish or reap queued work, "
        "then search-select to restore the selected incumbent. Make no further edits after selection. "
        "The host checks observed final source and the original full correctness roster independently.\n\n"
        if inputs.author_agents is not None
        else ""
    )
    return (
        "# Generated component compiler optimization\n\n"
        "Edit the compiler library through its declared entrypoints and reusable transforms. "
        "Use generated components to test source semantics, representations, layout, packing, "
        "host and device work, resource ownership, and producer/consumer composition. Keep complete "
        "construction and consumption costs visible.\n\n"
        "Invoke only advertised available broker actions, using the supplied broker path. "
        "Calibrated intervals are estimates: UNKNOWN remains unavailable and cannot become zero. "
        "Component RTL feedback is optional and uses the unchanged certified workload gate. "
        "Spike establishes functional evidence and cannot supply hardware timing. Neither tier "
        "grants final acceptance. Whole-model graphs, measurements, holdout answers and FireSim "
        "services are absent from this training profile. Preserve the fixed grader and accuracy gates.\n\n"
        + search_plan
        + "This prompt grants no launch authority. The host must independently admit the approved "
        "minimal view, isolated runtime and zero-history session before any authoring starts.\n\n"
        + "## Phase 2 broker actions for this launch\n\n"
        + action_inventory
        + "\n\nUse the exact broker and argv bindings in the JSON below; an unavailable action "
        "cannot supply evidence. This inventory changes no execution authority.\n\n"
        + GENERAL_COMPILER_CONTRACT_V1
        + "\n```json\n"
        + json.dumps(document, sort_keys=True, indent=2)
        + "\n```\n"
    )


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


def bounded_search_text(author_agents: int | None) -> str:
    """Explain queue ownership and scale exploration to the measured target gap."""
    if author_agents is None:
        return ""
    if type(author_agents) is not int or not 1 <= author_agents <= 3:
        raise StageGateError("normal search requires one to three explicitly selected authors")
    return (
        "\n\n## Bounded parallel search\n\n"
        f"Coordinate up to {author_agents} native authors on separate compiler symbols or files. "
        "Only the coordinator invokes the broker and merges their edits into this candidate. "
        "Authors return a hypothesis and scope for reusable compiler changes. Serialize shared edits. "
        "Freeze each integrated contender with revision-freeze, declaring hypothesis and edit_family. "
        "Queued evaluations read that retained revision while authors may continue editing. "
        "Start distinct hypotheses from a recorded parent. Join authors before search-resume-parent "
        "restores that issued source. Check actual emitted artifacts before simulation and retain "
        "negative, unchanged and failed results. Queue corpus-compile-spike-feedback for cheap own-artifact "
        "diagnostics and bounded activity; Spike activity does not establish target timing. search-status reports "
        "a provisional shortlist separately from its measured incumbent. search-calibrate consumes "
        "completed own cheap/RTL jobs for comma-separated revision_ids; its public partition was frozen "
        "before measurements. It starts no simulator and refuses missing evidence. Only independently "
        "qualified in-domain intervals order experiments; unresolved costs stay UNKNOWN. "
        "When a roofline target is configured, assign authors and focused measurements to "
        "search-status.roofline_priorities.next in order: lowest worst-repeat attainment first. "
        "Recheck after each result; keep unknown bounds in the separate evidence queue. Otherwise "
        "prioritize absolute potential savings. Use bounded measurements to resolve uncertain "
        "comparisons. evaluation-submit-selected measures comma-separated public "
        "family/capsule members through the same retained source, numerical gates and RTL certificate. "
        "Unselected members remain unmeasured; this diagnostic kind cannot promote or establish a "
        "plateau. Use complete corpus-rtl-feedback before promotion. Use isolated variants when "
        "attribution helps; coherent pass combinations may be measured as their own hypothesis. Complete "
        "measured simulator results govern selection. Measure the unchanged starting compiler "
        "for the baseline fallback. "
        + ADAPTIVE_SCOPE
        + "Use search-status and respect its stopping reason. Its optional performance_target uses "
        "the complete measured incumbent and every repeat. Unknown or partial rooflines cannot "
        "satisfy that target. When stop_on_plateau is false, use a plateau to change hypotheses "
        "and continue within the declared budgets. Before returning, join every author, "
        "stop edits at final campaign completion, invoke search-finalize, then search-select. "
        "Intermediate automatic-session checkpoints return a FINAL turn message without finalizing the search. "
        "After selection make no further edits. Run the "
        "original required feedback and analysis actions on those selected bytes. The host then "
        "runs the complete final correctness roster independently.\n\n"
    )


def render_stage_prompt(inputs: StagePromptInputs) -> str:
    """Render through the current shared prompt, then bind richer stage facts."""
    if inputs.capsule_development is not None:
        return (render_capsule_development_prompt(inputs) + render_author_guidance(inputs.author_guidance)
                + render_queued_author_continuation(inputs.continue_queued_author))
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
        compiler_pass_preparation=inputs.compiler_pass_preparation,
    )
    rendered = PP.render_initial_prompt(base).rstrip()
    from .compiler_library_guidance import prompt_text

    rendered += prompt_text(inputs.additional_compiler_library)
    if inputs.compiler_tool_profile is not None:
        from merlin.targetgen.tool_profile import tool_profile_prompt

        profile = dict(inputs.compiler_tool_profile)
        rendered += tool_profile_prompt(
            {
                "arm": profile.get("arm"),
                "tools": profile.get("treatment_tools"),
                "compiler_tool_profile": profile,
            }
        )
        rendered += (
            "\nPhase 2 inherits this exact Phase 1 inventory and preservation policy; "
            + (
                "the explicit additional library is declared separately above.\n"
                if inputs.additional_compiler_library is not None
                else "it adds no authoring capabilities.\n"
            )
            + "Invoke inherited common transformations through the admitted `common-transform request=PATH` "
            "broker action. The public client accepts its declared broker transport command; "
            "the host owns the original proof toolchain and acceptance policy.\n"
        )
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
        **(
            {"compiler_tool_profile": copy.deepcopy(dict(inputs.compiler_tool_profile))}
            if inputs.compiler_tool_profile is not None
            else {}
        ),
        **(
            {"additional_compiler_library": copy.deepcopy(dict(inputs.additional_compiler_library))}
            if inputs.additional_compiler_library is not None
            else {}
        ),
    }
    return (
        rendered
        + bounded_search_text(inputs.author_agents)
        + ("\n\nAutomatic session checkpoints and fixed deadlines:\n"
           + json.dumps(dict(inputs.search_session), sort_keys=True, indent=2)
           if inputs.search_session is not None else "")
        + "\n\n## Public program composition and feedback\n\n"
        + (
            "Preserve the frozen public objective and inspect the complete public sentinel for "
            "producer/consumer interactions, preparation and readout. Reduced witnesses help isolate "
            "costs; their wins do not establish complete-program improvements. Normal queued selection "
            "uses the complete declared development roster, not unmeasured sentinel projections. "
            "At the start of each round, read "
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
            "after seeing a candidate; keep the frozen declared public objective fixed. Public complete "
            "programs and connected cuts may execute through explicitly available bounded tools. Use "
            "compile-only analysis for larger extents and property-preserving reduced witnesses for "
            "cost questions. Protected programs and FireSim execution remain after compiler freeze. "
            "Full-size execution is optional post-freeze validation. If FireSim is explicitly "
            "available and requested for that validation, use only the queue-owned `runworkload-full`; "
            "the queue must own exactly `firesim kill` -> `firesim infrasetup` -> "
            "`firesim runworkload` -> `firesim kill`, in that order.\n\n"
            "Use L3 sparsely: the broker permits at most one reduced occupancy profile and two tuning "
            "GSIM feedback calls per round through those legacy actions. Queued jobs have their separate "
            "explicit resource, job, time and final-reserve budgets shown by search-status; partial "
            "diagnostics cannot promote a contender. Reserve the original final tuning call for the "
            "exact selected bytes. Iterate with "
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
        + render_author_guidance(inputs.author_guidance)
        + render_queued_author_continuation(inputs.continue_queued_author)
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


def freeze_author_guidance(text: str | None) -> dict[str, str] | None:
    """Retain bounded public operator intent, without expanding any authority."""
    if text is None:
        return None
    if type(text) is not str or not text.strip() or len(text.encode("utf-8")) > 256 << 10 or "\0" in text:
        raise StageGateError("author guidance requires bounded nonempty public text")
    return {"schema": "merlin.public_author_guidance.v1", "text": text, "sha256": _sha256(text.encode("utf-8"))}


def render_queued_author_continuation(enabled: bool) -> str:
    if type(enabled) is not bool:
        raise StageGateError("queued author continuation requires explicit Boolean selection")
    if not enabled:
        return ""
    return "\n\n## Continuous queued author session\n\n" + QUEUED_AUTHOR_CONTINUATION + "\n"


def render_author_guidance(record: Mapping[str, str] | None) -> str:
    if record is None:
        return ""
    if not isinstance(record, Mapping) or set(record) != {"schema", "text", "sha256"}:
        raise StageGateError("author guidance lacks its frozen public text identity")
    expected = freeze_author_guidance(record["text"])
    if dict(record) != expected:
        raise StageGateError("author guidance differs from its frozen text identity")
    return (
        "\n\n## Public operator guidance\n\n"
        "This guidance supplies intent and reviewed public support. It does not change "
        "the compiler edit authority, tool grants, numerical checks, evaluator or resource limits. "
        f"Frozen guidance SHA256: {record['sha256']}.\n\n" + record["text"] + "\n"
    )


def render_capsule_development_prompt(inputs: StagePromptInputs) -> str:
    scope = inputs.capsule_development
    if (
        not isinstance(scope, Mapping)
        or scope.get("schema") != "merlin.capsule_qualified_ordinary_development.v1"
        or scope.get("allowed_execution") != "compile_spike_only"
    ):
        raise StageGateError("capsule source prompt requires its explicit original development scope")
    data = dict(
        functional_run_id=inputs.functional_run_id,
        original_compiler_sha256=inputs.functional_submission_sha256,
        original_public_capsules=inputs.functional_public_capsules,
        original_hidden_capsules_aggregate=inputs.functional_hidden_capsules,
        candidate_path=inputs.submission_path,
        candidate_sha256=inputs.submission_initial_sha256,
        frozen_corpus_manifest=inputs.workload_manifest,
        frozen_corpus_sha256=inputs.workload_capsules_sha256,
        complete_required_cohort=_capsule_cohort_summary(scope["required_cohort"]),
        cohort_pin={"sha256": scope["required_cohort_declaration"]["sha256"]},
        actions=[dict(name=t.name, command=t.command, purpose=t.purpose) for t in inputs.tools],
        allowed_paths=list(inputs.allowed_paths),
        max_tool_calls=inputs.max_tool_calls,
        wall_budget_seconds=inputs.wall_budget_seconds,
        tool_timeout_seconds=inputs.tool_timeout_seconds,
        **({"search_session": dict(inputs.search_session)} if inputs.search_session is not None else {}),
    )
    return (
        "Edit only the authorized compiler source and permitted functions. This is "
        "capsule-qualified source development. "
        "Read STAGE_CONTEXT.json for the frozen compiler edit authority, available "
        "reviewed pass modules and API previews. Derive policies from semantics, "
        "shapes, layouts and proven resources; never select by workload identities. "
        "Begin with an unchanged retained control, then evaluate a distinct compiler "
        "rule through the same queue and preserve complete output checks. "
        "The original complete functional capsule roster and the separately pinned "
        "complete required cohort remain obligations. "
        "Pending, refused and unselected cases are not dropped or scored as zero. "
        "Full-model and formal/final qualification remain pending. "
        "Use revision-freeze, then evaluation-submit-functional-selected, "
        "poll status, and finalize queue cleanup. Obtain original complete feedback on the exact final source bytes. "
        "All source, numerical, ISA, output, resource and timing-domain checks remain "
        "active; refusals and UNKNOWN are evidence. "
        "Only the listed compile/Spike queue controls are available. Do not request "
        "RTL/gSIM/FireSim or hardware-loop instructions/configurations. "
        "No private inputs, answers, saved success records or changed numerical policies may establish authority. "
        "This stage produces an unscored development checkpoint, not performance promotion "
        "or a final qualified compiler. "
        + ADAPTIVE_SCOPE
        + "\n\n```json\n" + json.dumps(data, sort_keys=True, indent=2) + "\n```\n"
    )


def _capsule_cohort_summary(cohort):
    members = cohort.get("members")
    if not isinstance(members, list) or not 1 <= len(members) <= 1024:
        raise StageGateError("capsule prompt needs its bounded complete pending member list")
    identities = []
    for member in members:
        identity = member if type(member) is str else member.get("identity") if type(member) is dict else None
        if type(identity) is not str or not identity or len(identity) > 128 or identity.startswith("/"):
            raise StageGateError("capsule prompt requires bounded public member identities")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise StageGateError("capsule prompt cannot drop or repeat pending member identities")
    # Role/readiness trees and answer-bearing historical receipts remain host-only.
    # All pending obligations are explicit; publication or execution is not timing admission.
    return {
        "member_count": len(identities),
        "identities": identities,
        "status": "all original obligations retained; unselected qualification remains pending",
        "roles": "unchanged in the separately supplied original answer-free role projection",
    }
