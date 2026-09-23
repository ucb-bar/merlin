"""Frozen authoring input selection and exact agent-facing prompt preparation.

Callers supply the source layout root and admitted snapshots explicitly. Selection
preserves recorded grant destinations; it does not discover a checkout or requalify
a functional run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.perf.external_objective import OBJECTIVE_DIRECTORY, ExternalObjective, objective_directory
from merlin.targetgen.target_experiment import TargetExperiment

from . import agent_workspace as AW
from . import candidate_record as RECORD
from . import contracts as CONTRACTS
from . import corpus as CORPUS
from . import prompt as PP
from . import stage_prompt as SP
from .broker import AGENT_CORPUS_MOUNT, BROKER_NAME, BROKER_RECEIPT_MOUNT, BrokerAction
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .functional_inputs import FrozenFunctionalInputs, StageFunctionalRun, _frozen_path_for_destination


@dataclass(frozen=True)
class FullModelSentinel:
    capsule: str
    source_dir: Path
    descriptor: dict[str, Any]
    source_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class StageE2ESentinel:
    capsule: str
    capsule_path: str
    frozen_source_path: str
    capsule_sha256: str
    required_lanes: tuple[str, ...]
    required_tiers: tuple[str, ...]


def sentinel_identity(sentinel: StageE2ESentinel, *, role: str) -> dict[str, Any]:
    """Stable public identity for one complete-model member of a global portfolio."""
    if role not in ("primary", "training"):
        raise ValueError("portfolio member role must be primary or training")
    return {
        "capsule": sentinel.capsule,
        "capsule_sha256": sentinel.capsule_sha256,
        "required_lanes": list(sentinel.required_lanes),
        "required_tiers": list(sentinel.required_tiers),
        "role": role,
        "analysis": "full_graph_compile_and_static_only",
        "full_model_simulation_allowed": False,
    }


def select_full_model_sentinel(
    functional: StageFunctionalRun,
    target_experiment: TargetExperiment,
    *,
    source_root: Path,
    objective_capsule: str | None = None,
) -> FullModelSentinel:
    """Choose an immutable public model; an explicit experiment choice overrides the default.

    Selection does not requalify Phase 1 or establish that a model-kind capsule represents a
    complete application rather than a seam. That scope remains the captured workload's evidence.
    Explicit selection only changes which frozen graph the new experiment optimizes.
    """
    if objective_capsule is not None:
        objective_capsule = CONTRACTS.safe_component(objective_capsule, label="global objective capsule")
    snapshot_repo = Path(functional.bundle_input_snapshot["path"]) / "repo"
    try:
        relative = Path(target_experiment.capsule_corpus).resolve().relative_to(source_root)
    except ValueError as exc:
        raise StageGateError("target corpus cannot be mapped into the functional snapshot") from exc
    parent = snapshot_repo / relative.parent
    candidates: list[FullModelSentinel] = []
    for descriptor_path in sorted(parent.glob("*/*/capsule.yaml")):
        source = descriptor_path.parent
        descriptor = CONTRACTS.mapping_file(descriptor_path, yaml_file=True)
        lanes = descriptor.get("lanes")
        required = lanes.get("require") if isinstance(lanes, Mapping) else None
        tiers = descriptor.get("required_oracle_tiers")
        if (
            descriptor.get("kind") != "model"
            or descriptor.get("label") != "public"
            or (objective_capsule is None and (not isinstance(required, list) or not required))
            or (
                required is not None
                and (not isinstance(required, list) or any(not isinstance(lane, str) or not lane for lane in required))
            )
            or not isinstance(tiers, list)
            or "L2" not in tiers
        ):
            continue
        name = CONTRACTS.safe_component(str(descriptor.get("name") or ""), label="E2E sentinel")
        if source.name != name:
            raise StageGateError("E2E sentinel directory/name mismatch")
        tree = CONTRACTS.exact_tree_record(source)
        candidates.append(
            FullModelSentinel(
                name, source.resolve(), descriptor, str(tree["sha256"]), int(tree["n_files"]), int(tree["n_bytes"])
            )
        )
    if not candidates:
        raise StageGateError("functional snapshot has no public model with a declared lane and L2 screen")
    if objective_capsule is not None:
        matches = [item for item in candidates if item.capsule == objective_capsule]
        if len(matches) != 1:
            raise StageGateError(
                "explicit global objective is not one public L2-screened model "
                f"in the immutable Phase 1 snapshot: {objective_capsule!r}"
            )
        return matches[0]
    objectives = [
        item
        for item in candidates
        if isinstance(item.descriptor.get("performance"), Mapping)
        and item.descriptor["performance"].get("global_objective") is True
    ]
    if len(objectives) > 1:
        raise StageGateError(
            "functional snapshot declares multiple performance.global_objective models: "
            f"{[item.capsule for item in objectives]}"
        )
    declared = getattr(target_experiment, "performance_global_objective", None)
    if objectives:
        if declared is not None and objectives[0].capsule != declared:
            raise StageGateError(
                "capsule and experiment declarations disagree about the performance global "
                f"objective: {objectives[0].capsule!r} != {declared!r}"
            )
        return objectives[0]
    if declared is None:
        raise StageGateError(
            "functional snapshot has no capsule-level performance.global_objective and the "
            "experiment declares no performance.global_objective_capsule"
        )
    matches = [item for item in candidates if item.capsule == declared]
    if len(matches) != 1:
        raise StageGateError(
            "declared performance.global_objective_capsule is not one public L2-screened model "
            f"in the immutable Phase 1 snapshot: {declared!r}"
        )
    return matches[0]


def family_declarations(
    capsules: Sequence[CORPUS.PerformanceCapsule], formal_claim: Mapping[str, Any]
) -> tuple[SP.PerformanceFamilyDeclaration, ...]:
    rows: dict[str, SP.PerformanceFamilyDeclaration] = {}
    for capsule in capsules:
        performance = capsule.descriptor["performance"]
        comparand, falsifier = performance["comparand"], performance["falsifier"]
        knobs = performance["emitter"]["knobs"]
        fitted: tuple[str, ...] = ()
        if performance["claim"] == "PREDICTS":
            axes = {
                str(value)
                for key, value in knobs.items()
                if isinstance(value, str) and ("axis" in str(key) or "parameter" in str(key))
            }
            fitted = tuple(sorted(axes or {str(key) for key in knobs}))
        differential = json.dumps(
            {
                "kind": comparand["kind"],
                "against": comparand["against"],
                "cancels": comparand["cancels"],
                "demand_equal": comparand["demand_equal"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        family = SP.PerformanceFamilyDeclaration(
            capsule.family,
            performance["claim"],
            str(falsifier["negative_control"]),
            str(falsifier["observation"]),
            differential,
            fitted,
            copy.deepcopy(performance.get("acceptance")),
        )
        previous = rows.get(capsule.family)
        if previous is not None and previous != family:
            raise StageGateError(f"performance family declaration drifts: {capsule.family}")
        rows[capsule.family] = family
    declared = str(formal_claim.get("family") or "")
    claiming = rows.get(declared)
    if claiming is None or _canonical_json(claiming.acceptance) != _canonical_json(formal_claim.get("declaration")):
        raise StageGateError(
            f"{declared or 'the claiming'} family declaration drifts from its formal preflight acceptance"
        )
    return tuple(rows[name] for name in sorted(rows))


def select_e2e_sentinel(
    functional: StageFunctionalRun,
    frozen: FrozenFunctionalInputs,
    target_experiment: TargetExperiment,
    *,
    source_root: Path,
    objective_capsule: str | None = None,
) -> StageE2ESentinel:
    """Select the declared frozen whole-model objective, with a legacy fallback."""
    selected = select_full_model_sentinel(
        functional, target_experiment, source_root=source_root, objective_capsule=objective_capsule
    )
    snapshot_repo = (frozen.root / "repo").resolve(strict=True)
    try:
        relative = selected.source_dir.resolve(strict=True).relative_to(snapshot_repo)
    except ValueError as exc:
        raise StageGateError("frozen full-model sentinel is outside the functional snapshot") from exc
    destination = (source_root / relative).absolute()
    # Prove the prompt destination is one of the exact frozen grant views.
    if _frozen_path_for_destination(frozen, destination).resolve() != selected.source_dir.resolve():
        raise StageGateError("full-model sentinel does not map to its frozen grant destination")
    # Older frozen complete models can have no lane declaration. Preserve that absence instead
    # of inventing host/device coverage or excluding them from compile-only optimization.
    lanes = selected.descriptor.get("lanes")
    required_lanes = lanes.get("require", ()) if isinstance(lanes, Mapping) else ()
    return StageE2ESentinel(
        selected.capsule,
        str(destination),
        str(selected.source_dir),
        selected.source_sha256,
        tuple(required_lanes or ()),
        tuple(selected.descriptor["required_oracle_tiers"]),
    )


def _host_lane_grant(functional: StageFunctionalRun) -> SP.StageHostLaneGrant:
    host = functional.model_host_lane_snapshot
    package = Path(functional.model_host_package)
    return SP.StageHostLaneGrant(
        str(host["target"]),
        str(host["run_id"]),
        str(package),
        str(host["package_sha256"]),
        str(package / "manifest.yaml"),
        "host-owned capsule/model runner consumes frozen schedule+knobs for declared host lanes; "
        "candidate package handles accelerator regions",
    )


def prepare_prompt_inputs(
    functional: StageFunctionalRun,
    frozen_functional: FrozenFunctionalInputs,
    frozen_corpus: CORPUS.FrozenPerformanceCorpus,
    agent_inputs: AW.AgentInputSnapshot,
    target_experiment: TargetExperiment,
    actions: Sequence[BrokerAction],
    *,
    source_root: Path,
    formal_claim: Mapping[str, Any],
    smoke_replicates: int,
    wall_budget_seconds: int,
    rounds: int,
    round_timeout_seconds: int,
    max_tool_calls: int,
    tool_timeout_seconds: int,
    candidate_path: str = "submission",
) -> SP.StagePromptInputs:
    declaration = formal_claim.get("declaration")
    if not isinstance(declaration, Mapping):
        raise StageGateError("the formal claim omits its frozen acceptance declaration")
    formal_identities = RECORD.preflight_cohort(formal_claim)
    replicates = len(formal_identities)
    if (
        isinstance(smoke_replicates, bool)
        or not isinstance(smoke_replicates, int)
        or smoke_replicates <= 0
        or smoke_replicates >= replicates
    ):
        raise StageGateError("smoke replicates must be positive and smaller than the formal cohort")
    evidence = declaration.get("evidence")
    if not isinstance(evidence, Mapping):
        raise StageGateError("the formal claim omits timing-engine evidence semantics")
    timing_simulator = str(evidence.get("timing_simulator"))
    cells = tuple(
        PP.PerfCell(row.family, row.capsule, row.simulator, row.replicate)
        for row in CORPUS.expected_perf_cells(frozen_corpus.capsules, replicates, timing_simulator)
    )
    families = family_declarations(frozen_corpus.capsules, formal_claim)
    sentinel = select_e2e_sentinel(functional, frozen_functional, target_experiment, source_root=source_root)
    host = _host_lane_grant(functional)
    tools = tuple(
        PP.ToolGrant(
            action.name,
            f"python3 {BROKER_NAME} {action.name}"
            + (" " + " ".join(f"{name}=PATH" for name in action.placeholders) if action.placeholders else ""),
            action.purpose,
            action.required,
        )
        for action in actions
    )
    # THE AGENT SEES SANDBOX PATHS, NOT HOST PATHS. `agent_inputs.root` is bound read-only at
    # AGENT_CORPUS_MOUNT (see inner_broker_policy / AW.outer_codex_policy), so inside bwrap the manifest
    # is at `/perf-corpus/agent_input_manifest.json`. Declaring `agent_inputs.manifest_path` here named
    # the HOST path, which is not bound -- and the prompt tells the agent to stop with NO-GO if any
    # declared path is absent. Measured 2026-09-03: the agent probed the set, found this one missing,
    # and correctly refused in 24 s ("the required declared mount is absent"), so the stage produced a
    # candidate with zero authoring rounds and no transcript to audit. Every other entry here is
    # already a mount-side path; this was the one host-side straggler.
    allowed = (
        str(AW.FUNCTIONAL_BASE_MOUNT),
        candidate_path,
        str(AGENT_CORPUS_MOUNT),
        str(AGENT_CORPUS_MOUNT / agent_inputs.manifest_path.name),
        host.package_path,
        host.manifest_path,
        sentinel.capsule_path,
        BROKER_NAME,
        str(BROKER_RECEIPT_MOUNT),
        str(AW.FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        str(AW.PERF_CORPUS_MANIFEST_MOUNT),
        *(str(grant.destination) for grant in frozen_functional.grants),
    )
    return SP.StagePromptInputs(
        target=target_experiment.target,
        approach="arm4",
        functional_run_id=functional.run_id,
        functional_submission_sha256=functional.digest,
        frozen_functional_path=str(AW.FUNCTIONAL_BASE_MOUNT),
        frozen_functional_sha256=functional.digest,
        submission_path=candidate_path,
        submission_initial_sha256=functional.digest,
        functional_public_capsules=functional.public_capsules,
        functional_hidden_capsules=functional.hidden_capsules,
        functional_bundle_snapshot_manifest=str(AW.FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        functional_bundle_snapshot_manifest_sha256=(
            frozen_functional.public_marker_sha256 or frozen_functional.marker_sha256
        ),
        functional_bundle_snapshot_sha256=(frozen_functional.public_content_sha256 or frozen_functional.content_sha256),
        workload_root=str(AGENT_CORPUS_MOUNT),
        workload_manifest=str(AW.PERF_CORPUS_MANIFEST_MOUNT),
        workload_manifest_sha256=frozen_corpus.manifest_sha256,
        workload_capsules_sha256=frozen_corpus.capsules_sha256,
        expected_cells=cells,
        replicates=replicates,
        formal_replicate_identities=formal_identities,
        formal_claim=copy.deepcopy(dict(formal_claim)),
        smoke_replicates=smoke_replicates,
        wall_budget_seconds=wall_budget_seconds,
        rounds=rounds,
        round_timeout_seconds=round_timeout_seconds,
        max_tool_calls=max_tool_calls,
        tool_timeout_seconds=tool_timeout_seconds,
        families=families,
        host_lane=host,
        e2e_sentinel=sentinel,
        tools=tools,
        allowed_paths=tuple(dict.fromkeys(allowed)),
        execution_broker_path=BROKER_NAME,
        execution_broker_command=f"python3 {BROKER_NAME}",
        broker_receipt_path=str(BROKER_RECEIPT_MOUNT),
    )


def select_external_e2e_sentinel(objective: ExternalObjective, inputs: AW.AgentInputSnapshot) -> StageE2ESentinel:
    """Select only the exact external source already sealed into the ordinary RO grant."""
    if type(objective) is not ExternalObjective:
        raise StageGateError("external objective requires host-loaded typed source bytes")
    AW.verify_answer_free_agent_inputs(inputs)
    manifest = CONTRACTS.mapping_file(inputs.manifest_path)
    portfolio = manifest.get("external_objectives")
    if isinstance(portfolio, list):
        matches = [row for row in portfolio if row.get("id") == objective.objective_id]
        if len(matches) != 1 or matches[0] != objective.record():
            raise StageGateError("external objective is not a member of the sealed portfolio")
        relative_root = objective_directory(objective.objective_id)
    else:
        relative_root = Path(OBJECTIVE_DIRECTORY)
    source = inputs.root / relative_root
    expected = dict(objective.files())
    if set(path.name for path in source.iterdir()) != set(expected) or any(
        (source / name).read_bytes() != payload for name, payload in expected.items()
    ):
        raise StageGateError("sealed external objective differs from host-pinned source")
    return StageE2ESentinel(
        objective.objective_id,
        str(AGENT_CORPUS_MOUNT / relative_root),
        str(source),
        CONTRACTS.exact_tree_record(source)["sha256"],
        (),
        (),
    )
