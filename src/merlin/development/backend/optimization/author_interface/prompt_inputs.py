"""Fail-closed initial task contract for an Arm4-only performance campaign.

The performance driver owns the measurements.  This module only renders the exact task shown to an
agent and refuses to render when launch facts are incomplete or internally inconsistent.  Keeping the
prompt as a deterministic artifact makes its promises auditable alongside the frozen compiler and
workload digests.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.development.backend.optimization.contracts import StageGateError
from merlin.development.targetgen.authoring.compiler_contract import append_general_compiler_contract

# ---- prompt ------------------------------------------------------------------------------------


class PerfPromptContractError(ValueError):
    """The proposed performance task is not safe or complete enough to serve."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerfPromptContractError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in ("\n", "\r", "\0")):
        raise PerfPromptContractError(f"{label} must be a single safe text line")
    return result


def _simple_name(value: object, *, label: str) -> str:
    name = _text(value, label=label)
    if PurePosixPath(name).name != name or name in (".", ".."):
        raise PerfPromptContractError(f"{label} must be a simple path component, got {name!r}")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if any(character not in allowed for character in name):
        raise PerfPromptContractError(f"{label} contains an unsafe character: {name!r}")
    return name


def _sha256(value: object, *, label: str) -> str:
    digest = _text(value, label=label)
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PerfPromptContractError(f"{label} must be an explicit lowercase SHA-256")
    return digest


def _unique_nonempty(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    rows = tuple(_text(value, label=label) for value in values)
    if not rows:
        raise PerfPromptContractError(f"{label} must be non-empty")
    if len(set(rows)) != len(rows):
        raise PerfPromptContractError(f"{label} contains duplicate entries")
    return rows


@dataclass(frozen=True, order=True)
class PerfCell:
    """The exact identity of one result the campaign must produce."""

    family: str
    capsule: str
    simulator: str
    replicate: str

    def validate(self) -> None:
        _simple_name(self.family, label="cell family")
        _simple_name(self.capsule, label="cell capsule")
        _simple_name(self.replicate, label="cell replicate")
        if self.simulator not in ("spike", "gsim", "verilator"):
            raise PerfPromptContractError(f"cell simulator must be spike, gsim, or verilator, got {self.simulator!r}")

    @property
    def label(self) -> str:
        return "/".join((self.family, self.capsule, self.simulator, self.replicate))


CANONICAL_CLAIMS = ("RECOVERS", "PREDICTS", "DIFFERENTIAL", "EMITS")


@dataclass(frozen=True)
class PerfFamily:
    """The declaration that gives one generated family a falsifiable interpretation."""

    family: str
    claim: str
    negative_control: str
    falsifier_observation: str
    differential_basis: str
    fitted_parameters: tuple[str, ...] = ()

    def validate(self) -> None:
        _simple_name(self.family, label="performance family")
        if self.claim not in CANONICAL_CLAIMS:
            raise PerfPromptContractError(f"family {self.family!r} has a non-canonical claim {self.claim!r}")
        _text(self.negative_control, label=f"family {self.family} negative control")
        _text(self.falsifier_observation, label=f"family {self.family} falsifier observation")
        _text(self.differential_basis, label=f"family {self.family} differential basis")
        fitted = tuple(
            _simple_name(value, label=f"family {self.family} fitted parameter") for value in self.fitted_parameters
        )
        if len(set(fitted)) != len(fitted):
            raise PerfPromptContractError(f"family {self.family!r} repeats a fitted parameter")
        if self.claim == "PREDICTS" and not fitted:
            raise PerfPromptContractError(f"PREDICTS family {self.family!r} must declare fitted parameters")


@dataclass(frozen=True)
class ToolGrant:
    """One sandbox-visible tool and whether evidence must prove that the agent invoked it."""

    name: str
    command: str
    purpose: str
    required: bool = False

    def validate(self) -> None:
        _simple_name(self.name, label="tool name")
        _text(self.command, label=f"tool {self.name} command")
        _text(self.purpose, label=f"tool {self.name} purpose")
        if not isinstance(self.required, bool):
            raise PerfPromptContractError(f"tool {self.name} required flag must be boolean")


@dataclass(frozen=True)
class HostLaneGrant:
    """The pinned host/RVV compiler package used at accelerator boundaries."""

    target: str
    package_id: str
    package_path: str
    package_sha256: str
    manifest_path: str
    entrypoint: str

    def validate(self) -> None:
        _simple_name(self.target, label="host-lane target")
        _simple_name(self.package_id, label="host-lane package id")
        _text(self.package_path, label="host-lane package path")
        _sha256(self.package_sha256, label="host-lane package digest")
        _text(self.manifest_path, label="host-lane manifest path")
        _text(self.entrypoint, label="host-lane entrypoint")


@dataclass(frozen=True)
class PerfPromptInputs:
    """All launch facts that are interpolated into the initial agent task."""

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
    workload_root: str
    workload_manifest: str
    workload_manifest_sha256: str
    workload_capsules_sha256: str
    expected_cells: tuple[PerfCell, ...]
    families: tuple[PerfFamily, ...]
    host_lane: HostLaneGrant
    tools: tuple[ToolGrant, ...]
    allowed_paths: tuple[str, ...]
    execution_broker_path: str
    execution_broker_command: str
    broker_receipt_path: str
    sandbox_engine: str = "bwrap"
    broker_clearenv: bool = True
    direct_candidate_execution_forbidden: bool = True
    credentials_absent: bool = True
    answer_surfaces_masked: bool = True
    declared_files_only: bool = True
    functional_highest_tier: str = "L3"
    functional_integrity_status: str = "clean"
    compiler_pass_preparation: dict[str, Any] | None = None


def validate_prompt_inputs(inputs: PerfPromptInputs) -> None:
    """Refuse a prompt whose launch facts could license a misleading experiment."""
    if not isinstance(inputs, PerfPromptInputs):
        raise PerfPromptContractError("performance prompt inputs must be PerfPromptInputs")
    _simple_name(inputs.target, label="target")
    if inputs.approach != "arm4":
        raise PerfPromptContractError(
            "the performance campaign retains the frozen functional compiler approach "
            "(frozen approach='arm4' compatibility key)"
        )
    _simple_name(inputs.functional_run_id, label="functional run id")
    functional_digest = _sha256(inputs.functional_submission_sha256, label="functional submission digest")
    baseline_digest = _sha256(inputs.frozen_functional_sha256, label="frozen functional digest")
    seed_digest = _sha256(inputs.submission_initial_sha256, label="submission initial digest")
    if inputs.compiler_pass_preparation is None:
        if baseline_digest != functional_digest:
            raise PerfPromptContractError("frozen functional bytes do not match the certified submission")
        if seed_digest != functional_digest:
            raise PerfPromptContractError("performance submission did not start as the exact functional fork")
    else:
        from merlin.development.backend.optimization.campaign.pass_preparation import (
            validate_prompt_projection,
        )
        from merlin.development.backend.optimization.contracts import StageGateError

        try:
            validate_prompt_projection(
                inputs.compiler_pass_preparation,
                original_sha256=functional_digest,
                baseline_sha256=baseline_digest,
                candidate_seed_sha256=seed_digest,
            )
        except StageGateError as error:
            raise PerfPromptContractError(str(error)) from error
    for value, label in (
        (inputs.frozen_functional_path, "frozen functional path"),
        (inputs.submission_path, "performance submission path"),
        (inputs.workload_root, "workload root"),
        (inputs.workload_manifest, "workload manifest"),
    ):
        _text(value, label=label)
    _sha256(inputs.workload_manifest_sha256, label="workload manifest digest")
    _sha256(inputs.workload_capsules_sha256, label="workload capsule digest")
    for value, label in (
        (inputs.functional_public_capsules, "functional public capsule count"),
        (inputs.functional_hidden_capsules, "functional hidden capsule count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PerfPromptContractError(f"{label} must be a positive integer")
    if inputs.functional_highest_tier != "L3":
        raise PerfPromptContractError("the functional fork must have passed through L3")
    if inputs.functional_integrity_status != "clean":
        raise PerfPromptContractError("the functional fork must have a clean integrity verdict")

    if (
        inputs.sandbox_engine != "bwrap"
        or inputs.credentials_absent is not True
        or inputs.answer_surfaces_masked is not True
        or inputs.declared_files_only is not True
        or inputs.broker_clearenv is not True
        or inputs.direct_candidate_execution_forbidden is not True
    ):
        raise PerfPromptContractError(
            "performance execution requires broker-only bwrap with --clearenv, no candidate "
            "credentials, masked answers, and a declared-files-only view"
        )
    _text(inputs.execution_broker_path, label="execution broker path")
    _text(inputs.execution_broker_command, label="execution broker command")
    _text(inputs.broker_receipt_path, label="broker receipt path")

    cells = tuple(inputs.expected_cells)
    if not cells:
        raise PerfPromptContractError("performance completion set contains zero cells")
    for cell in cells:
        if not isinstance(cell, PerfCell):
            raise PerfPromptContractError("every expected performance cell must be a PerfCell")
        cell.validate()
    if len(set(cells)) != len(cells):
        raise PerfPromptContractError("performance completion set repeats a cell identity")
    by_replica: dict[tuple[str, str, str], set[str]] = {}
    for cell in cells:
        key = (cell.family, cell.capsule, cell.replicate)
        by_replica.setdefault(key, set()).add(cell.simulator)
    timing_simulators = {cell.simulator for cell in cells if cell.simulator != "spike"}
    if len(timing_simulators) != 1:
        raise PerfPromptContractError("the exact completion matrix must select one elaborated-RTL timing engine")
    timing_simulator = next(iter(timing_simulators))
    incomplete = sorted(key for key, simulators in by_replica.items() if simulators != {"spike", timing_simulator})
    if incomplete:
        raise PerfPromptContractError(f"every replica requires both the L2 and L3 cells; incomplete: {incomplete}")
    replicates_by_capsule: dict[tuple[str, str], set[str]] = {}
    for family, capsule, replicate in by_replica:
        replicates_by_capsule.setdefault((family, capsule), set()).add(replicate)
    replicate_sets = {tuple(sorted(values)) for values in replicates_by_capsule.values()}
    if len(replicate_sets) != 1:
        raise PerfPromptContractError("every performance capsule must declare the same exact replicate set")

    families = tuple(inputs.families)
    if not families:
        raise PerfPromptContractError("performance corpus has no falsifiable family declarations")
    for family in families:
        if not isinstance(family, PerfFamily):
            raise PerfPromptContractError("every performance family must be a PerfFamily")
        family.validate()
    names = [family.family for family in families]
    if len(set(names)) != len(names):
        raise PerfPromptContractError("performance family declarations contain duplicates")
    cell_families = {cell.family for cell in cells}
    if set(names) != cell_families:
        raise PerfPromptContractError("performance family declarations must exactly cover the expected cell families")

    inputs.host_lane.validate()
    tools = tuple(inputs.tools)
    if not tools:
        raise PerfPromptContractError("performance sandbox grants no tools")
    for tool in tools:
        if not isinstance(tool, ToolGrant):
            raise PerfPromptContractError("every tool grant must be a ToolGrant")
        tool.validate()
    tool_names = [tool.name for tool in tools]
    if len(set(tool_names)) != len(tool_names):
        raise PerfPromptContractError("performance tool grants contain duplicate names")
    if not any(tool.required for tool in tools):
        raise PerfPromptContractError("at least one granted tool must be enforced as required")
    broker_command = _text(inputs.execution_broker_command, label="execution broker command")
    outside_broker = [
        tool.name
        for tool in tools
        if not (tool.command == broker_command or tool.command.startswith(broker_command + " "))
    ]
    if outside_broker:
        raise PerfPromptContractError(f"tool commands bypass the required execution broker: {outside_broker}")

    allowed = _unique_nonempty(inputs.allowed_paths, label="sandbox allowed path")
    required_paths = {
        inputs.frozen_functional_path,
        inputs.submission_path,
        inputs.workload_root,
        inputs.workload_manifest,
        inputs.host_lane.package_path,
        inputs.host_lane.manifest_path,
        inputs.execution_broker_path,
        inputs.broker_receipt_path,
    }
    absent = sorted(required_paths - set(allowed))
    if absent:
        raise PerfPromptContractError(f"sandbox allowed paths omit launch inputs required by the prompt: {absent}")


def _bullet_paths(paths: tuple[str, ...]) -> str:
    return "\n".join(f"- `{path}`" for path in paths)


def _tool_table(tools: tuple[ToolGrant, ...]) -> str:
    rows = ["| Tool | Command | Enforcement | Purpose |", "|---|---|---|---|"]
    for tool in tools:
        enforcement = "required; invocation receipt checked" if tool.required else "allowed"
        rows.append(f"| `{tool.name}` | `{tool.command}` | {enforcement} | {tool.purpose} |")
    return "\n".join(rows)


def _family_table(families: tuple[PerfFamily, ...]) -> str:
    rows = [
        "| Family | Claim | Fit axes | Negative control | Falsifier observation | Differential basis |",
        "|---|---|---|---|---|---|",
    ]
    for family in families:
        fitted = ", ".join(family.fitted_parameters) or "none"
        rows.append(
            f"| `{family.family}` | `{family.claim}` | `{fitted}` | `{family.negative_control}` | "
            f"{family.falsifier_observation} | {family.differential_basis} |"
        )
    return "\n".join(rows)


def _final_cells_only_note() -> str:
    """Stated only when the stage declares an authoring gSIM budget; neutral about the members it names."""
    from merlin.development.backend.optimization.measurement.development_feedback import (
        authoring_gsim_floor_budget,
    )

    budget = authoring_gsim_floor_budget()
    if budget is None:
        return ""
    return (
        "\n   Tuning members whose roofline floor exceeds this stage's authoring gSIM budget "
        f"({budget} cycles) are measured on gSIM only in the final measurement cells. For those members "
        "`tuning-gsim-feedback` returns an unmeasured cell that says so, with its roofline; Spike "
        "correctness, the static analyses and the reduced witnesses are the feedback available for them."
    )


# ---- metric_scope ------------------------------------------------------------------------------
# What a Phase 2 performance score may contain: accelerator work, never Merlin-supplied host compute.
#
# Phase 2 tunes what decides ACCELERATOR performance: tiling and blocking, the dataflow choice, DMA
# scheduling and double-buffering, scratchpad and accumulator reuse, epilogue fusion and command-queue
# pressure. Host code counts only where it changes accelerator behaviour or codegen -- the layout and
# packing that feed the accelerator, im2col, command issue. Pure host compute is Merlin-supplied
# (``docs/design/merlin_supplied_host_placement.md``): Merlin's own host package compiles it, so it is
# neither the candidate's work nor something a candidate may be scored on.
#
# The split is the frozen Phase 0 placement (:mod:`merlin.development.targetgen.execution.host.frozen_placement`), never the
# candidate's output:
#
# * operations Phase 0 placed on the **accelerator** are the candidate's scored work;
# * operations placed on the **host** are Merlin-supplied and excluded;
# * **support** operations (packing, residency, readout bookkeeping, shape plumbing) are deliberately
#   unplaced by the Phase 0 screen. They are what the candidate emits around its device groups, so they
#   count, as accelerator-affecting host code.
#
# Two consequences for measurement:
#
# 1. A whole-program capsule whose frozen placement keeps compute on the host may not take the
#    candidate-whole-program route, where the candidate would emit (and could tune) that host code. It
#    takes the owned-host split instead, where Merlin compiles the host part.
# 2. A cycle count measured on such a program still contains Merlin's host stage. The absolute score is
#    the measured count minus a measured Merlin host stage; without that measurement it is UNKNOWN. A
#    paired delta is admissible without it only when both arms link byte-identical Merlin host code,
#    because the host stage then cancels.

SCHEMA = "merlin.phase2.metric_scope.v1"

ACCELERATOR_ONLY = "accelerator_only"

MERLIN_HOST_SPLIT = "merlin_host_split"

ACCELERATOR_AFFECTING_HOST = (
    "unplaced support operations the candidate emits around its device groups: layout and packing "
    "feeding the accelerator, im2col, residency and readout bookkeeping, command issue"
)


class MetricScopeRefusal(ValueError):
    """The scored scope cannot be established, so no score may be issued."""


def _placements(declaration: Mapping[str, Any]) -> Mapping[str, Any]:
    screen = declaration.get("software_screen")
    placements = screen.get("placements") if isinstance(screen, Mapping) else None
    if not isinstance(placements, Mapping):
        raise MetricScopeRefusal("the capsule carries no frozen Phase 0 placement, so its scored scope is unknown")
    for lane, count in placements.items():
        if type(count) is not int or count < 0:
            raise MetricScopeRefusal(f"the frozen placement count for lane {lane!r} is malformed")
    if not placements.get("accelerator"):
        raise MetricScopeRefusal("the frozen placement puts no operation on the accelerator; nothing is scorable")
    return placements


def declared_scope(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """The scope a declaration's frozen placement states, without reading program bytes.

    The route this selects re-binds the placement to the program bytes (:func:`bound_scope`, or the
    owned-host route's own ``frozen_placement.read``), which fails closed.
    """
    placements = _placements(declaration)
    host = int(placements.get("host") or 0)
    return {
        "schema": SCHEMA,
        "authority": "frozen Phase 0 software_screen",
        "kind": MERLIN_HOST_SPLIT if host else ACCELERATOR_ONLY,
        "placements": dict(sorted(placements.items())),
        "scored": "accelerator-placed operations plus " + ACCELERATOR_AFFECTING_HOST,
        "excluded": (
            "host-placed operations: Merlin-supplied host compute, compiled by Merlin's host package"
            if host
            else "none: the frozen placement keeps no compute on the host"
        ),
        "candidate_whole_program_admissible": not host,
    }


def bound_scope(declaration: Mapping[str, Any], directory: str | Path) -> dict[str, Any]:
    """:func:`declared_scope` bound to the capsule's program bytes, with per-operation lanes."""
    from merlin.development.targetgen.execution.host import frozen_placement

    try:
        frozen = frozen_placement.read(declaration, directory)
    except frozen_placement.FrozenPlacementRefusal as exc:
        raise MetricScopeRefusal(str(exc)) from exc
    scope = declared_scope(declaration)
    scope.update(
        program_sha256=frozen["program_sha256"],
        scored_operations=frozen["accelerator_operations"],
        excluded_operations=frozen["host_operations"],
    )
    return scope


def scored_cycles(
    scope: Mapping[str, Any], measured_cycles: int, *, merlin_host_cycles: int | None = None
) -> dict[str, Any]:
    """The cycles a candidate is scored on, from a complete measured count.

    An accelerator-only program scores its whole measured count. A Merlin-host split scores the
    measured count minus a measured Merlin host stage, and is UNKNOWN without one: the whole-program
    count is never silently promoted to a candidate score.
    """
    if isinstance(measured_cycles, bool) or not isinstance(measured_cycles, int) or measured_cycles <= 0:
        raise MetricScopeRefusal("a scored count needs a positive measured cycle count")
    kind = scope.get("kind")
    if kind == ACCELERATOR_ONLY:
        if merlin_host_cycles not in (None, 0):
            raise MetricScopeRefusal("an accelerator-only program has no Merlin host stage to subtract")
        return {"status": "scored", "cycles": measured_cycles, "basis": "complete call; no Merlin host stage"}
    if kind != MERLIN_HOST_SPLIT:
        raise MetricScopeRefusal(f"unknown metric scope kind {kind!r}")
    if merlin_host_cycles is None:
        return {
            "status": "unknown",
            "cycles": None,
            "measured_cycles": measured_cycles,
            "basis": (
                "the measured count contains Merlin-supplied host compute and no Merlin host stage was "
                "measured; only a paired delta over byte-identical Merlin host code is admissible"
            ),
        }
    if isinstance(merlin_host_cycles, bool) or not isinstance(merlin_host_cycles, int) or merlin_host_cycles < 0:
        raise MetricScopeRefusal("a Merlin host stage must be a nonnegative measured cycle count")
    if merlin_host_cycles >= measured_cycles:
        raise MetricScopeRefusal("the Merlin host stage is not smaller than the complete measured count")
    return {
        "status": "scored",
        "cycles": measured_cycles - merlin_host_cycles,
        "measured_cycles": measured_cycles,
        "excluded_merlin_host_cycles": merlin_host_cycles,
        "basis": "complete call minus the measured Merlin-supplied host stage",
    }


def paired_delta(
    scope: Mapping[str, Any],
    *,
    candidate_cycles: int,
    baseline_cycles: int,
    candidate_host_sha256: str | None,
    baseline_host_sha256: str | None,
) -> dict[str, Any]:
    """Candidate minus baseline, admissible when Merlin's host stage provably cancels.

    For a Merlin-host split both arms must link the same Merlin host code, bound by digest; a differing
    or missing digest means the delta could contain host work and it is refused.
    """
    for value in (candidate_cycles, baseline_cycles):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MetricScopeRefusal("a paired delta needs positive measured counts for both arms")
    if scope.get("kind") == MERLIN_HOST_SPLIT:
        if not candidate_host_sha256 or candidate_host_sha256 != baseline_host_sha256:
            return {
                "status": "refused",
                "delta_cycles": None,
                "basis": "the arms do not provably link byte-identical Merlin host code",
            }
        basis = "Merlin host code is byte-identical in both arms, so its stage cancels"
    elif scope.get("kind") == ACCELERATOR_ONLY:
        basis = "accelerator-only program"
    else:
        raise MetricScopeRefusal(f"unknown metric scope kind {scope.get('kind')!r}")
    return {"status": "scored", "delta_cycles": candidate_cycles - baseline_cycles, "basis": basis}


# ---- sweep_policy ------------------------------------------------------------------------------
# The authoring gSIM sweep's declared environment: its fan-out and its member budget.

SWEEP_WORKERS_ENV = "MERLIN_PERF_SWEEP_WORKERS"

AUTHORING_GSIM_FLOOR_BUDGET_ENV = "MERLIN_PERF_AUTHORING_GSIM_FLOOR_BUDGET"


def authoring_gsim_floor_budget() -> int | None:
    """The declared authoring gSIM budget (roofline-floor cycles), or None when every member is swept."""
    raw = (os.environ.get(AUTHORING_GSIM_FLOOR_BUDGET_ENV) or "").strip()
    if not raw:
        return None
    try:
        budget = int(raw)
    except ValueError:
        raise StageGateError(f"{AUTHORING_GSIM_FLOOR_BUDGET_ENV}={raw!r} is not an integer cycle budget") from None
    if budget <= 0:
        raise StageGateError(f"{AUTHORING_GSIM_FLOOR_BUDGET_ENV}={budget} is not a positive cycle budget")
    return budget


def sweep_workers() -> int:
    """The declared sweep fan-out, or 1. Refuses a value it cannot read rather than guessing one."""
    raw = (os.environ.get(SWEEP_WORKERS_ENV) or "").strip()
    if not raw:
        return 1
    try:
        workers = int(raw)
    except ValueError:
        raise StageGateError(f"{SWEEP_WORKERS_ENV}={raw!r} is not an integer; refusing to guess a fan-out") from None
    if workers < 1:
        raise StageGateError(f"{SWEEP_WORKERS_ENV}={workers} is not a positive fan-out")
    return workers


# ---- search_guidance ---------------------------------------------------------------------------
# Author guidance for scaling a search to its measured performance deficit.
#
# This changes exploration advice, never the evidence used for ranking or stopping.

ADAPTIVE_SCOPE = (
    "Scale the scope of changes to the gap from the configured target. For each valid measured "
    "attainment, compute target attainment / current attainment. Above 4x, prioritize replacing "
    "costly lowering strategies, restructuring whole loop nests, changing tiling, loop order, "
    "data reuse, operation grouping and coherent combinations of "
    "admitted passes; estimate the recoverable dominant cost before spending a measurement. "
    "Above 1.25x through 4x, explore pass combinations and broader parameter sweeps. At 1.25x "
    "or less, prioritize granular tuning and regression control. These are exploration heuristics, "
    "not predicted speedups or evidence of convergence. Unknown or partial bounds stay UNKNOWN; "
    "do not assume those members are close to the target. Continue high-impact improvements "
    "using admitted semantics while resolving their missing measurements. Preserve all numerical "
    "and source gates at every scale. "
    "Use available queue capacity for distinct frozen contenders instead of waiting after each "
    "submission. Give each contender a recorded parent and hypothesis, serialize shared source "
    "edits, and keep evidence attached to its immutable revision. Reconsider hypotheses as "
    "results arrive. Combine retained improvements and measure the combination; individual "
    "gains do not establish the combined gain. "
)
