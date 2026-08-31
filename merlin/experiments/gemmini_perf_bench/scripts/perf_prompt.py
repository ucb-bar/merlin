"""Fail-closed initial task contract for an Arm4-only performance campaign.

The performance driver owns the measurements.  This module only renders the exact task shown to an
agent and refuses to render when launch facts are incomplete or internally inconsistent.  Keeping the
prompt as a deterministic artifact makes its promises auditable alongside the frozen compiler and
workload digests.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable


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

    def validate(self) -> None:
        _simple_name(self.family, label="performance family")
        if self.claim not in ("RECOVERS", "PREDICTS", "DIFFERENTIAL"):
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
    for value, label in ((inputs.functional_public_capsules, "functional public capsule count"),
                         (inputs.functional_hidden_capsules, "functional hidden capsule count")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PerfPromptContractError(f"{label} must be a positive integer")
    if inputs.functional_highest_tier != "L3":
        raise PerfPromptContractError("the functional fork must have passed through L3")
    if inputs.functional_integrity_status != "clean":
        raise PerfPromptContractError("the functional fork must have a clean integrity verdict")

    if (inputs.sandbox_engine != "bwrap" or inputs.credentials_absent is not True
            or inputs.answer_surfaces_masked is not True
            or inputs.declared_files_only is not True or inputs.broker_clearenv is not True
            or inputs.direct_candidate_execution_forbidden is not True):
        raise PerfPromptContractError(
            "performance execution requires broker-only bwrap with --clearenv, no candidate "
            "credentials, masked answers, and a declared-files-only view")
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
  simulators x {len(replicas) // len(capsules)} replicate(s) per capsule.

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
- Declared interface: `{inputs.host_lane.entrypoint}`

Use the existing compiler interface naturally: the Arm4 target dialect and passes own accelerator
regions; the pinned host/RVV package owns host islands and accelerator boundaries; the harness passes the
two packages through the declared model/compiler interface. Do not modify, vendor, clone, reimplement, or
silently bypass the host lane. Do not move accelerator work onto the host merely to pass. Do not invent a
hidden host lowering, ABI, RVV dialect, runtime, or boundary shim: use the granted package and manifest.

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
- Compute and optimization must remain compiler-generated. No hand kernel, library answer, hardcoded
  output, capsule-specific dispatch, authored timing, or fabricated evidence.
- Official correctness, cycles, provenance, falsifier, differential, and GO/NO-GO records are written by
  the harness. Never author or patch a passing verdict yourself.

## Falsifiable family contract

{_family_table(inputs.families)}

For every family, the run must record: the declared traits and emitter status; exact correctness and L3
timing rows; the measured negative-control result; whether the declared falsifier fired and why; and a
`merlin.perf.differential` comparability decision proving that unpriced demands cancel (or explicitly
refusing the comparison). A capsule declaration may state what would falsify a claim, but only run-authored
evidence may state whether it fired. Correctness alone is never a performance falsifier on an interlocked
machine.

## Work loop

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
