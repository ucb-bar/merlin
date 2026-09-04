"""Fail-closed initial task contract for an Arm4-only performance campaign.

The performance driver owns the measurements.  This module only renders the exact task shown to an
agent and refuses to render when launch facts are incomplete or internally inconsistent.  Keeping the
prompt as a deterministic artifact makes its promises auditable alongside the frozen compiler and
workload digests.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable


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
    if (len(digest) != 64 or digest.lower() != digest
            or any(character not in "0123456789abcdef" for character in digest)):
        raise PerfPromptContractError(f"{label} must be an explicit lowercase SHA-256")
    return digest


def _unique_nonempty(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    rows = tuple(_text(value, label=label) for value in values)
    if not rows:
        raise PerfPromptContractError(f"{label} must be non-empty")
    if len(set(rows)) != len(rows):
        raise PerfPromptContractError(f"{label} contains duplicate entries")
    return rows


#: The disjoint claim vocabulary the performance template declares.  Named once so the authoring
#: stage can admit a family's claim by MEMBERSHIP rather than by naming one family's claim word.
PERF_CLAIM_VOCABULARY = ("RECOVERS", "PREDICTS", "DIFFERENTIAL")


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
        if self.simulator not in ("spike", "verilator"):
            raise PerfPromptContractError(
                f"cell simulator must be spike or verilator, got {self.simulator!r}")

    @property
    def label(self) -> str:
        return "/".join((self.family, self.capsule, self.simulator, self.replicate))


@dataclass(frozen=True)
class PerfFamily:
    """The declaration that gives one generated family a falsifiable interpretation."""

    family: str
    claim: str
    negative_control: str
    falsifier_observation: str
    differential_basis: str
    fitted_parameters: tuple[str, ...] = ()
    acceptance: Mapping[str, Any] | None = None

    def validate(self) -> None:
        _simple_name(self.family, label="performance family")
        if self.claim not in PERF_CLAIM_VOCABULARY:
            raise PerfPromptContractError(
                f"family {self.family!r} has a non-canonical claim {self.claim!r}")
        _text(self.negative_control, label=f"family {self.family} negative control")
        _text(self.falsifier_observation, label=f"family {self.family} falsifier observation")
        _text(self.differential_basis, label=f"family {self.family} differential basis")
        fitted = tuple(_simple_name(value, label=f"family {self.family} fitted parameter")
                       for value in self.fitted_parameters)
        if len(set(fitted)) != len(fitted):
            raise PerfPromptContractError(
                f"family {self.family!r} repeats a fitted parameter")
        if self.claim == "PREDICTS" and not fitted:
            raise PerfPromptContractError(
                f"PREDICTS family {self.family!r} must declare fitted parameters")
        if self.claim == "PREDICTS" and not isinstance(self.acceptance, Mapping):
            raise PerfPromptContractError(
                f"PREDICTS family {self.family!r} must carry its frozen acceptance contract")
        if self.acceptance is not None:
            try:
                json.dumps(self.acceptance, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise PerfPromptContractError(
                    f"family {self.family!r} acceptance contract is not canonical JSON") from exc


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
    integration_seam: str

    def validate(self) -> None:
        _simple_name(self.target, label="host-lane target")
        _simple_name(self.package_id, label="host-lane package id")
        _text(self.package_path, label="host-lane package path")
        _sha256(self.package_sha256, label="host-lane package digest")
        _text(self.manifest_path, label="host-lane manifest path")
        _text(self.integration_seam, label="host-lane integration seam")


@dataclass(frozen=True)
class E2ESentinel:
    """Smallest frozen public whole-model admission check; never a timing claim."""

    capsule: str
    capsule_path: str
    frozen_source_path: str
    capsule_sha256: str
    required_lanes: tuple[str, ...]
    required_tiers: tuple[str, ...]

    def validate(self) -> None:
        _simple_name(self.capsule, label="E2E sentinel capsule")
        _text(self.capsule_path, label="E2E sentinel path")
        _text(self.frozen_source_path, label="E2E sentinel frozen source path")
        _sha256(self.capsule_sha256, label="E2E sentinel digest")
        if set(self.required_lanes) != {"on_mesh", "scalar_rvv_lane"}:
            raise PerfPromptContractError(
                "E2E sentinel must require both on_mesh and scalar_rvv_lane")
        if "L3" not in self.required_tiers:
            raise PerfPromptContractError("E2E sentinel must require L3")


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
    functional_bundle_snapshot_manifest: str
    functional_bundle_snapshot_manifest_sha256: str
    functional_bundle_snapshot_sha256: str
    workload_root: str
    workload_manifest: str
    workload_manifest_sha256: str
    workload_capsules_sha256: str
    expected_cells: tuple[PerfCell, ...]
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    formal_claim: Mapping[str, Any]
    smoke_replicates: int
    wall_budget_seconds: int
    rounds: int
    round_timeout_seconds: int
    max_tool_calls: int
    tool_timeout_seconds: int
    families: tuple[PerfFamily, ...]
    host_lane: HostLaneGrant
    e2e_sentinel: E2ESentinel
    tools: tuple[ToolGrant, ...]
    allowed_paths: tuple[str, ...]
    execution_broker_path: str
    execution_broker_command: str
    broker_receipt_path: str
    sandbox_engine: str = "bwrap"
    broker_clearenv: bool = True
    direct_candidate_execution_forbidden: bool = True
    outer_codex_auth_exception: bool = True
    inner_execution_credentials_absent: bool = True
    answer_surfaces_masked: bool = True
    declared_files_only: bool = True
    functional_highest_tier: str = "L3"
    functional_integrity_status: str = "clean"


def validate_prompt_inputs(inputs: PerfPromptInputs) -> None:
    """Refuse a prompt whose launch facts could license a misleading experiment."""
    if not isinstance(inputs, PerfPromptInputs):
        raise PerfPromptContractError("performance prompt inputs must be PerfPromptInputs")
    _simple_name(inputs.target, label="target")
    if inputs.approach != "arm4":
        raise PerfPromptContractError("the performance campaign evaluates only approach='arm4'")
    _simple_name(inputs.functional_run_id, label="functional run id")
    functional_digest = _sha256(
        inputs.functional_submission_sha256, label="functional submission digest")
    if _sha256(inputs.frozen_functional_sha256, label="frozen functional digest") != functional_digest:
        raise PerfPromptContractError("frozen functional bytes do not match the certified submission")
    if _sha256(inputs.submission_initial_sha256, label="submission initial digest") != functional_digest:
        raise PerfPromptContractError("performance submission did not start as the exact functional fork")
    for value, label in (
            (inputs.frozen_functional_path, "frozen functional path"),
            (inputs.submission_path, "performance submission path"),
            (inputs.workload_root, "workload root"),
            (inputs.workload_manifest, "workload manifest")):
        _text(value, label=label)
    _sha256(inputs.workload_manifest_sha256, label="workload manifest digest")
    _sha256(inputs.workload_capsules_sha256, label="workload capsule digest")
    _text(inputs.functional_bundle_snapshot_manifest,
          label="functional bundle snapshot manifest")
    _sha256(inputs.functional_bundle_snapshot_manifest_sha256,
            label="functional bundle snapshot manifest digest")
    _sha256(inputs.functional_bundle_snapshot_sha256,
            label="functional bundle snapshot content digest")
    for value, label in ((inputs.functional_public_capsules, "functional public capsule count"),
                         (inputs.functional_hidden_capsules, "functional hidden capsule count")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PerfPromptContractError(f"{label} must be a positive integer")
    if inputs.functional_highest_tier != "L3":
        raise PerfPromptContractError("the functional fork must have passed through L3")
    if inputs.functional_integrity_status != "clean":
        raise PerfPromptContractError("the functional fork must have a clean integrity verdict")

    if (inputs.sandbox_engine != "bwrap" or inputs.outer_codex_auth_exception is not True
            or inputs.inner_execution_credentials_absent is not True
            or inputs.answer_surfaces_masked is not True
            or inputs.declared_files_only is not True or inputs.broker_clearenv is not True
            or inputs.direct_candidate_execution_forbidden is not True):
        raise PerfPromptContractError(
            "performance execution requires broker-only bwrap with --clearenv, an explicit outer "
            "Codex auth exception, no inner execution credentials, masked answers, and a "
            "declared-files-only view")
    _text(inputs.execution_broker_path, label="execution broker path")
    _text(inputs.execution_broker_command, label="execution broker command")
    _text(inputs.broker_receipt_path, label="broker receipt path")

    cells = tuple(inputs.expected_cells)
    if isinstance(inputs.replicates, bool) or not isinstance(inputs.replicates, int) \
            or inputs.replicates <= 0:
        raise PerfPromptContractError("performance replicates must be a positive integer")
    # Diagnose a formal-contract mismatch before the derived cell-count check.  Otherwise a caller
    # that requests two replicas against the frozen three-replica acceptance receives a generic
    # identity error and can mistake the formal cohort for a configurable run knob.
    early_formal = inputs.formal_claim
    early_declaration = (early_formal.get("declaration")
                         if isinstance(early_formal, Mapping) else None)
    early_replicates = (early_declaration.get("replicates")
                        if isinstance(early_declaration, Mapping) else None)
    if isinstance(early_replicates, Mapping):
        # A family predeclaring its identities is held to them; one whose declaration leaves them to
        # the run is held to the schedule its own preflight returned.  Reading the cohort this way
        # keeps the diagnostic available to every family, not only to one that names an exact count.
        early_identities = tuple(early_replicates.get("identities")
                                 or early_formal.get("replicates") or ())
        early_exact = early_replicates.get("exact_count")
        if (early_identities != tuple(inputs.formal_replicate_identities)
                or len(early_identities) != inputs.replicates
                or (isinstance(early_exact, int) and not isinstance(early_exact, bool)
                    and early_exact != inputs.replicates)):
            raise PerfPromptContractError(
                "formal replicates must exactly match the frozen acceptance cohort")
        if (isinstance(inputs.smoke_replicates, bool)
                or not isinstance(inputs.smoke_replicates, int)
                or inputs.smoke_replicates <= 0
                or inputs.smoke_replicates >= inputs.replicates):
            raise PerfPromptContractError(
                "smoke replicates must be a smaller non-claim diagnostic count")
    for value, label in (
            (inputs.wall_budget_seconds, "wall budget"), (inputs.rounds, "round count"),
            (inputs.round_timeout_seconds, "round timeout"),
            (inputs.max_tool_calls, "tool-call budget"),
            (inputs.tool_timeout_seconds, "tool timeout")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PerfPromptContractError(f"performance {label} must be a positive integer")
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
    incomplete = sorted(key for key, simulators in by_replica.items()
                        if simulators != {"spike", "verilator"})
    if incomplete:
        raise PerfPromptContractError(
            f"every replica requires both the L2 and L3 cells; incomplete: {incomplete}")
    replicates_by_capsule: dict[tuple[str, str], set[str]] = {}
    for family, capsule, replicate in by_replica:
        replicates_by_capsule.setdefault((family, capsule), set()).add(replicate)
    replicate_sets = {tuple(sorted(values)) for values in replicates_by_capsule.values()}
    if len(replicate_sets) != 1:
        raise PerfPromptContractError(
            "every performance capsule must declare the same exact replicate set")
    if next(iter(replicate_sets), ()) != tuple(f"r{index:03d}" for index in range(inputs.replicates)):
        raise PerfPromptContractError("expected cells do not match the declared replicate count")

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
        raise PerfPromptContractError(
            "performance family declarations must exactly cover the expected cell families")

    # The claim-bearing family is the one the preflight names for ITSELF.  A family literal here
    # would refuse every family but one, including a newly declared family whose own acceptance
    # block names the analyzer that decides it.
    formal = inputs.formal_claim
    claim_family = formal.get("family") if isinstance(formal, Mapping) else None
    if (not isinstance(formal, Mapping) or formal.get("schema_version") != 1
            or not isinstance(claim_family, str) or not claim_family
            or formal.get("claim") not in PERF_CLAIM_VOCABULARY
            or formal.get("status") != "READY" or formal.get("refusal_reasons") != []):
        raise PerfPromptContractError("formal claim preflight is absent or not READY")
    declaration = formal.get("declaration")
    cohort = formal.get("cohort")
    expected_identities = formal.get("expected_identities")
    claim_families = [family for family in families if family.family == claim_family]
    if len(claim_families) != 1 or not isinstance(declaration, Mapping):
        raise PerfPromptContractError("formal claim lacks one frozen acceptance declaration")
    if json.dumps(claim_families[0].acceptance, sort_keys=True, separators=(",", ":")) != json.dumps(
            declaration, sort_keys=True, separators=(",", ":")):
        raise PerfPromptContractError("formal acceptance drifts from the frozen family declaration")
    replicate_contract = declaration.get("replicates")
    if not isinstance(replicate_contract, Mapping):
        raise PerfPromptContractError("formal acceptance omits its exact replicate contract")
    # Predeclared identities bind the run; their absence means the declaration leaves the schedule
    # to the run, and the preflight's own returned schedule is then the cohort.
    identities = tuple(replicate_contract.get("identities") or formal.get("replicates") or ())
    exact_count = replicate_contract.get("exact_count")
    minimum_count = replicate_contract.get("minimum_count")
    if (not identities or any(not isinstance(value, str) for value in identities)
            or identities != tuple(inputs.formal_replicate_identities)
            or inputs.replicates != len(identities)
            or identities != tuple(f"r{index:03d}" for index in range(len(identities)))
            or (isinstance(exact_count, int) and not isinstance(exact_count, bool)
                and exact_count != len(identities))
            or (isinstance(minimum_count, int) and not isinstance(minimum_count, bool)
                and len(identities) < minimum_count)
            or not (isinstance(exact_count, int) and not isinstance(exact_count, bool)
                    or isinstance(minimum_count, int) and not isinstance(minimum_count, bool))):
        raise PerfPromptContractError(
            "formal replicates must exactly match the frozen acceptance cohort")
    if (isinstance(inputs.smoke_replicates, bool) or not isinstance(inputs.smoke_replicates, int)
            or inputs.smoke_replicates <= 0 or inputs.smoke_replicates >= inputs.replicates):
        raise PerfPromptContractError(
            "smoke replicates must be a smaller non-claim diagnostic count")
    declared_cohort = cohort.get("replicates") if isinstance(cohort, Mapping) else None
    if (not isinstance(cohort, Mapping)
            or (declared_cohort is not None and tuple(declared_cohort) != identities)
            or not isinstance(expected_identities, list) or not expected_identities):
        raise PerfPromptContractError("formal preflight omits its exact cohort or identities")
    formal_cells: set[tuple[str, str, str, str]] = set()
    for row in expected_identities:
        if not isinstance(row, Mapping):
            raise PerfPromptContractError("formal preflight has a malformed identity")
        simulator = row.get("simulator")
        tier = row.get("tier")
        if ((simulator, tier) not in (("spike", "L2"), ("verilator", "L3"))
                or row.get("family") != claim_family):
            raise PerfPromptContractError("formal preflight changes L2/L3 evidence semantics")
        identity = (str(row.get("family")), str(row.get("capsule")), str(simulator),
                    str(row.get("replicate")))
        if identity in formal_cells:
            raise PerfPromptContractError("formal preflight repeats an identity")
        formal_cells.add(identity)
    prompt_claim_cells = {(cell.family, cell.capsule, cell.simulator, cell.replicate)
                          for cell in cells if cell.family == claim_family}
    if formal_cells != prompt_claim_cells:
        raise PerfPromptContractError(
            "formal preflight identities drift from the exact prompt completion cells")

    inputs.host_lane.validate()
    inputs.e2e_sentinel.validate()
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
    outside_broker = [tool.name for tool in tools
                      if not (tool.command == broker_command
                              or tool.command.startswith(broker_command + " "))]
    if outside_broker:
        raise PerfPromptContractError(
            f"tool commands bypass the required execution broker: {outside_broker}")

    allowed = _unique_nonempty(inputs.allowed_paths, label="sandbox allowed path")
    required_paths = {
        inputs.frozen_functional_path,
        inputs.submission_path,
        inputs.workload_root,
        inputs.workload_manifest,
        inputs.host_lane.package_path,
        inputs.host_lane.manifest_path,
        inputs.e2e_sentinel.capsule_path,
        inputs.functional_bundle_snapshot_manifest,
        inputs.execution_broker_path,
        inputs.broker_receipt_path,
    }
    absent = sorted(required_paths - set(allowed))
    if absent:
        raise PerfPromptContractError(
            f"sandbox allowed paths omit launch inputs required by the prompt: {absent}")


def _bullet_paths(paths: tuple[str, ...]) -> str:
    return "\n".join(f"- `{path}`" for path in paths)


def _tool_table(tools: tuple[ToolGrant, ...]) -> str:
    rows = ["| Tool | Command | Enforcement | Purpose |", "|---|---|---|---|"]
    for tool in tools:
        enforcement = "required; invocation receipt checked" if tool.required else "allowed"
        rows.append(f"| `{tool.name}` | `{tool.command}` | {enforcement} | {tool.purpose} |")
    return "\n".join(rows)


def _family_table(families: tuple[PerfFamily, ...]) -> str:
    rows = ["| Family | Claim | Fit axes | Negative control | Falsifier observation | Differential basis |",
            "|---|---|---|---|---|---|"]
    for family in families:
        fitted = ", ".join(family.fitted_parameters) or "none"
        rows.append(
            f"| `{family.family}` | `{family.claim}` | `{fitted}` | `{family.negative_control}` | "
            f"{family.falsifier_observation} | {family.differential_basis} |")
    return "\n".join(rows)


def _acceptance_blocks(families: tuple[PerfFamily, ...]) -> str:
    rows: list[str] = []
    for family in families:
        if family.acceptance is not None:
            rendered = json.dumps(family.acceptance, sort_keys=True, indent=2)
            rows.append(f"### `{family.family}` frozen acceptance\n\n```json\n{rendered}\n```")
    return "\n\n".join(rows)


def render_initial_prompt(inputs: PerfPromptInputs) -> str:
    """Render deterministic TASK.md bytes after validating every launch prerequisite."""
    validate_prompt_inputs(inputs)
    cells = tuple(sorted(inputs.expected_cells))
    capsules = {(cell.family, cell.capsule) for cell in cells}
    replicas = {(cell.family, cell.capsule, cell.replicate) for cell in cells}
    required_tools = [tool.name for tool in inputs.tools if tool.required]
    identity_lines = "\n".join(f"- `{cell.label}`" for cell in cells)
    prompt = f"""# {inputs.target} Arm4-only performance task

This is a performance continuation of one already-certified functional compiler, not a fresh backend
task and not a cross-approach race. **Only the Arm4 submission is evaluated.** Do not build, run, copy,
or report a golden, IREE, baseline, prior-agent, or alternate compiler arm.

## Immutable fork point

- Functional run: `{inputs.functional_run_id}`
- Certified functional submission SHA-256: `{inputs.functional_submission_sha256}`
- Read-only functional baseline: `{inputs.frozen_functional_path}`
- Arm4 submission under evaluation: `{inputs.submission_path}`
- Submission initial SHA-256: `{inputs.submission_initial_sha256}` (must equal the certified digest)
- Functional evidence: {inputs.functional_public_capsules} public/dev and
  {inputs.functional_hidden_capsules} held-out capsules passed through L3; integrity status `clean`.
- Frozen functional authoring/tool snapshot: `{inputs.functional_bundle_snapshot_manifest}`
  (manifest `{inputs.functional_bundle_snapshot_manifest_sha256}`, content
  `{inputs.functional_bundle_snapshot_sha256}`).

The frozen baseline is the attribution boundary. Never modify it. The evaluated submission must begin as
its byte-exact copy; do not replace the package, change its interface, or import implementation from
another compiler. Any permitted optimization belongs only in the Arm4 submission, must remain a general
compiler transformation, and must preserve the complete functional grade.

## Exact generated workload and completion set

- Frozen generated corpus: `{inputs.workload_root}`
- Structured corpus manifest: `{inputs.workload_manifest}`
- Manifest SHA-256: `{inputs.workload_manifest_sha256}`
- Capsule-tree SHA-256: `{inputs.workload_capsules_sha256}`
- Non-vacuous completion target: **{len(cells)} cells** = {len(capsules)} capsules x the declared
  simulators x **{inputs.replicates}** replicate(s) per capsule.

The manifest and every capsule byte are immutable campaign inputs. Read shapes, dtypes, operations,
performance declarations, and knobs from them. Do not hand-author replacement workloads, specialize on a
capsule name, bake in observed outputs/timings, or add undeclared cases after seeing a result.

Every one of these exact `(family, capsule, simulator, replicate)` identities is required exactly once:

{identity_lines}

No aggregate count can hide a missing, duplicate, failed, or unexpected identity.

## Two tiers with different meanings

- **L2 / Spike is a correctness screen only.** It must establish correctness for its exact cell. Spike
  cycles must be recorded as `null`/absent and are never citable as performance.
- **L3 / Verilator is performance certification.** Its exact cell must be correct and produce a positive
  integer cycle count. Only L3 Verilator timing may enter a performance result.

A matching L2 and L3 cell must exist for every family/capsule/replicate. Passing L2 never substitutes for
L3, and a Verilator number from an incorrect result is not a measurement.

## Host/RVV/accelerator boundary: use the granted lane

The harness supplies a pinned, read-only host lane for code outside accelerator regions:

- Target/package: `{inputs.host_lane.target}` / `{inputs.host_lane.package_id}`
- Package: `{inputs.host_lane.package_path}`
- Package SHA-256: `{inputs.host_lane.package_sha256}`
- Manifest: `{inputs.host_lane.manifest_path}`
- Integration seam: `{inputs.host_lane.integration_seam}`

Use the existing compiler interface naturally: the Arm4 target dialect and passes own accelerator
regions; the pinned host/RVV package owns host islands and accelerator boundaries; the harness passes the
two packages through the declared model/compiler interface. Do not modify, vendor, clone, reimplement, or
silently bypass the host lane. Do not move accelerator work onto the host merely to pass. Do not invent a
hidden host lowering, ABI, RVV dialect, runtime, or boundary shim: use the granted package and manifest.
The RVV package is schedule/knob data consumed by the host-owned model runner; it is not an executable
agent CLI and this task does not fabricate one.

## Frozen full-model admission sentinel

`{inputs.e2e_sentinel.capsule}` at `{inputs.e2e_sentinel.capsule_path}` (SHA-256
`{inputs.e2e_sentinel.capsule_sha256}`) is the smallest frozen public `kind: model` capsule requiring
both `on_mesh` and `scalar_rvv_lane` and L3. The harness regrades it through L2+L3 after optimization.
It is a high-fidelity correctness admission sentinel, not a performance cell and not a source of a
cycle or speedup claim.

## Enforced tooling

{_tool_table(inputs.tools)}

The commands below are broker/control entrypoints, not permission to invoke the candidate package or a
simulator directly. Use only the declared tools and inputs. The harness checks invocation receipts for every required tool:
`{', '.join(required_tools)}`. Mentioning a command in a plan is not use; the corresponding broker/tool
receipt must exist. A missing required receipt is NO-GO even if output files happen to exist. Tools may
inspect your own artifacts and return redacted verdicts; they may not reveal or reconstruct withheld
answers. Do not substitute a hand-written parser, simulator, timing model, host compiler, or target
backend for a provided tool.

## Sandbox and integrity boundary

Network is available to the agent and to brokered execution. Network isolation is neither claimed nor required for this campaign.
The integrity boundary instead comes from explicit mounts and tools,
credential hygiene, answer/oracle masking, immutable inputs, broker-only execution, and receipts.

The outer Codex control plane necessarily has an isolated authentication exception and network access.
Only inner brokered execution is credential-free (`--clearenv` with no credential mounts). Do not claim
that the entire run is credential-free.

**All candidate/compiler/tool/simulator/measurement execution must go through**
`{inputs.execution_broker_command}` at `{inputs.execution_broker_path}`. It records harness-authored,
append-only invocation receipts at `{inputs.broker_receipt_path}`. The broker launches
mount-scoped `bwrap` with `--clearenv`, removes credentials, masks answer surfaces, hides undeclared host
paths, and mounts package/inputs read-only except for explicit Arm4 work/output areas. You may not invoke
the candidate or any underlying tool/simulator outside this broker. If the broker, its credential/mount/
mask enforcement, or its receipt channel is absent, stop with **NO-GO**. The complete declared path set is:

{_bullet_paths(inputs.allowed_paths)}

- Do not access credentials, hidden/withheld capsules or goldens, prior run outputs, other submissions,
  undeclared files, `/proc` escape surfaces, or host state outside the mounts. Network availability does
  not relax the answer/oracle mask, declared-tool boundary, or evidence requirements.
- Do not loosen mounts/permissions, follow or create links across the boundary, launch a nested sandbox,
  or smuggle data through another process. Do not edit the runner, grader, oracle, corpus, result files,
  tool receipts, frozen compiler, host lane, or provenance records.
- Keep generated build/cache state ephemeral. Before ending every round, remove `build/`, `__pycache__/`,
  and `.git/` from the candidate; these paths are excluded from the measurement digest and retaining one
  makes the candidate unsealable rather than silently unhashed.
- Compute and optimization must remain compiler-generated. No hand kernel, library answer, hardcoded
  output, capsule-specific dispatch, authored timing, or fabricated evidence.
- Official correctness, cycles, provenance, falsifier, differential, and GO/NO-GO records are written by
  the harness. Never author or patch a passing verdict yourself.

## Falsifiable family contract

{_family_table(inputs.families)}

The following `performance.acceptance` block is frozen descriptor data, not an after-the-fact tuning
knob. Its analyzer, cohort, replicate identities, evidence tiers, and thresholds must match exactly:

{_acceptance_blocks(inputs.families)}

The formal claim uses exactly **{inputs.replicates}** replicates with identities
`{', '.join(inputs.formal_replicate_identities)}`. The authoring smoke allowance is
**{inputs.smoke_replicates}** replicate(s) only. Smoke work is a cheap diagnostic: it cannot populate an
official result row, satisfy an expected identity, complete the frozen cohort, or support a formal
performance claim regardless of its outcome. Only the harness-owned exact L2/L3 campaign can do that.

For every family, the run must record: the declared traits and emitter status; exact correctness and L3
timing rows; the measured negative-control result; whether the declared falsifier fired and why; and a
`merlin.perf.differential` comparability decision proving that unpriced demands cancel (or explicitly
refusing the comparison). A capsule declaration may state what would falsify a claim, but only run-authored
evidence may state whether it fired. Correctness alone is never a performance falsifier on an interlocked
machine.

## Work loop

The authoring budget is exactly {inputs.rounds} round(s), {inputs.round_timeout_seconds} seconds per
round, {inputs.wall_budget_seconds} seconds total wall time, at most {inputs.max_tool_calls} broker
calls, and {inputs.tool_timeout_seconds} seconds per broker action. Stop honestly when any limit is
reached; do not route around it.

1. Read this file, the frozen performance manifest, the Arm4 submission manifest/report, the host-lane
   manifest, and all granted tool documentation. First write `submission/performance/PLAN.md`, covering
   the whole corpus, derived performance levers, invariants, negative controls, and cheapest-to-L3
   validation order. Append each change, observed redacted verdict, and next hypothesis to
   `submission/performance/iteration_notes.md`; fresh rounds must read both files before editing.
2. Establish the exact functional fork and input digests before changing or measuring anything. If any
   digest, mount, tool, host lane, family declaration, expected identity, or sandbox guarantee is absent
   or mismatched, stop and report **NO-GO**. Do not improvise around a failed prerequisite.
3. Use required tools through their declared commands. Iterate on the smallest affected generated member;
   preserve generality and re-run the relevant correctness screen after every compiler change.
4. Before declaring ready, re-check the full functional contract, frozen-input digests, required tool
   receipts, and exact expected identity set. Summarize scope and honest limitations in
   `submission/performance/REPORT.md`. The harness then owns L2/L3 execution and final evidence.

## Completion and claims

`GO` requires all of the following, with no exception: every exact identity reported once; all L2 and L3
cells correct; every L2 cycle absent; every L3 cycle a positive integer; compiler, corpus, host-lane,
simulator, and tool provenance present and digest-verified before and after execution; clean sandbox and
anti-cheat audits; every required tool receipt; measured negative-control/falsifier evidence; and a valid
differential-comparability decision for every promoted claim. Missing, duplicate, stale, incomparable,
unverifiable, incorrect, or mutable evidence makes the result `NO-GO` or an explicit family-level
`REFUSED`/`INERT` outcome—never an assumed pass.

Even a complete measurement does **not** claim: performance from Spike; superiority over an unevaluated
arm; speedup without an explicit identical-work comparand; functional or performance generalization beyond
the graded held-out set and this frozen generated corpus; end-to-end model speedup from kernel-only cells;
or a promoted family when its falsifier/differential evidence is absent, inert, or incomparable.

End your own report with exactly one honest line:

1. `Arm4 performance candidate is ready for the exact frozen campaign.`
2. `Arm4 performance candidate is not ready; remaining blockers and affected identities are listed.`
3. `Arm4 performance result is non-comparable because the fork, integrity, or evidence boundary was violated.`
"""
    return prompt.rstrip() + "\n"


def prompt_sha256(inputs: PerfPromptInputs) -> str:
    """Content identity archived with the task actually served to the agent."""
    return hashlib.sha256(render_initial_prompt(inputs).encode("utf-8")).hexdigest()
