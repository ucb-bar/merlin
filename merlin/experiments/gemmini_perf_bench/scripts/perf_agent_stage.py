#!/usr/bin/env python3
"""Create one auditable performance candidate from an exact functional fork.

This stage deliberately does not grade or promote anything.  It gives Codex a fresh copy of a
functionally certified package and an answer-free view of the generated performance contracts.  Any
compiler/tool execution requested by the agent goes through a second, credential-free
bwrap broker.  When the bounded authoring rounds end, the candidate is copied to a read-only snapshot
and described by a content-addressed record for :mod:`run_perf_bench` to consume.

There are two distinct filesystem and credential boundaries.  The outer Codex control plane gets one
isolated ``CODEX_HOME`` plus the explicit authentication mount and the functional run's frozen authoring
grants, but no live descriptor-derived target toolchain.  The inner execution plane gets that live
descriptor-derived toolchain and the writable candidate, but has ``--clearenv``
and receives no credential bind.  Network is available in both planes and is explicitly not claimed as
an isolation property; the experiment's protection comes from exact mounts, masks, and audited routes.
"""
from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

import perf_campaign as PC
import perf_prompt as PP
from merlin.benchharness import hash_tree, runs_root
from merlin.common.paths import merlin_dir, repo_root
from merlin.perf import claim_reach as CLAIM
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_tokens
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment


SCHEMA_VERSION = 2
AGENT_CORPUS_MOUNT = Path("/perf-corpus")
FUNCTIONAL_BASE_MOUNT = Path("/perf-functional-base")
FUNCTIONAL_INPUT_MANIFEST_MOUNT = Path("/perf-functional-inputs/snapshot.json")
PERF_CORPUS_MANIFEST_MOUNT = Path("/perf-corpus-manifest.json")
BROKER_NAME = "/perf-control/perf_tool.py"
BROKER_RECEIPT_MOUNT = Path("/perf-control/receipts.jsonl")
_HEX = frozenset("0123456789abcdef")
_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


class StageGateError(RuntimeError):
    """The candidate cannot be launched or admitted without weakening an experiment gate."""


@dataclass(frozen=True)
class AgentSandboxPolicy:
    argv: tuple[str, ...]
    answer_surface_gap: tuple[str, ...]
    network: str
    clear_environment: bool
    candidate_writable: bool
    corpus_read_only: bool


@dataclass(frozen=True)
class AgentInputSnapshot:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class PromptArtifact:
    source_path: Path
    text: str
    sha256: str
    n_bytes: int


@dataclass(frozen=True)
class FrozenGrant:
    declared_path: str
    destination: Path
    source: Path
    source_sha256: str


@dataclass(frozen=True)
class FrozenFunctionalInputs:
    root: Path
    marker: Path
    marker_sha256: str
    content_sha256: str
    grants: tuple[FrozenGrant, ...]


@dataclass(frozen=True)
class BrokerAction:
    name: str
    argv_template: tuple[str, ...]
    placeholders: tuple[str, ...]
    purpose: str
    required: bool

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv_template": list(self.argv_template),
                "placeholders": list(self.placeholders), "purpose": self.purpose,
                "required": self.required}


@dataclass(frozen=True)
class VerifiedCandidateHandoff:
    """Narrow, stable measurement boundary produced only from a validated record."""

    record_path: Path
    record_sha256: str
    candidate_path: Path
    candidate_sha256: str
    candidate_initial_sha256: str
    functional_run_id: str
    functional_submission_sha256: str
    functional_base_path: Path
    functional_bundle_snapshot_sha256: str
    functional_bundle_manifest: Path
    functional_bundle_manifest_sha256: str
    target_descriptor: Path
    target_descriptor_sha256: str
    corpus_root: Path
    corpus_manifest: Path
    corpus_manifest_sha256: str
    corpus_sha256: str
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    formal_claim: dict[str, Any]
    smoke_replicates: int
    expected_cells: tuple[dict[str, str], ...]
    families: tuple[dict[str, Any], ...]
    host_lane: dict[str, Any]
    e2e_sentinel: dict[str, Any]
    prompt_sha256: str
    prompt_facts_sha256: str
    prompt_path: Path
    transcript_path: Path
    transcript_sha256: str
    transcript_audit: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    required_actions: tuple[str, ...]
    tool_evidence: dict[str, Any]
    sandbox_evidence: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"content-addressed evidence is absent or linked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and value.lower() == value
            and all(character in _HEX for character in value))


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode("utf-8")


def _tree_files(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StageGateError(f"candidate tree contains a symlink: {path}")
        if path.is_file():
            rows[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not rows:
        raise StageGateError(f"candidate tree contains zero files: {root}")
    return rows


def assert_candidate_sealable(root: Path) -> None:
    """Keep the authoring/measurement digest domains identical."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            raise StageGateError(
                f"performance candidate retains digest-excluded ephemeral state: {path}")


def candidate_delta(base: Path, candidate: Path) -> dict[str, Any]:
    """Describe changed package bytes and reject documentation-only authoring as vacuous."""
    before, after = _tree_files(base), _tree_files(candidate)
    changed = sorted(path for path in before.keys() | after.keys()
                     if before.get(path) != after.get(path))
    execution_relevant = [path for path in changed
                          if "docs" not in Path(path).parts and Path(path).suffix.lower() != ".md"]
    return {"changed_files": changed, "changed_file_count": len(changed),
            "execution_relevant_changed_files": execution_relevant,
            "execution_relevant_changed_file_count": len(execution_relevant)}


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(document))


def _safe_component(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise StageGateError(f"{label} must be a simple non-empty path component")
    return value


def _require_real_directory(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise StageGateError(f"{label} is absent or linked: {raw}")
    return raw.resolve()


def _require_executable(name_or_path: str, *, label: str) -> Path:
    found = shutil.which(name_or_path)
    path = Path(found or name_or_path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise StageGateError(f"required {label} executable is absent: {name_or_path}")
    return path.resolve()


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


def materialize_canonical_prompt(inputs: PP.PerfPromptInputs,
                                 artifact_path: Path) -> PromptArtifact:
    """Render the sole accepted prompt, after every frozen launch fact is known."""
    text = PP.render_initial_prompt(inputs)
    if not isinstance(text, str) or not text.strip():
        raise StageGateError("performance prompt renderer returned no instruction")
    payload = text.encode("utf-8")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as stream:
        stream.write(payload)
    return load_prompt(artifact_path)


def load_frozen_functional_inputs(functional: PC.FunctionalRun) -> FrozenFunctionalInputs:
    """Reify the exact v2 grant table already verified by ``inspect_functional_run``."""
    root = Path(functional.bundle_input_snapshot["path"])
    marker = root / "snapshot.json"
    if root.is_symlink() or not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise StageGateError("functional v2 input snapshot or marker is absent")
    document = json.loads(marker.read_text(encoding="utf-8"))
    if (document.get("version") != 2
            or document.get("content_sha256") != functional.bundle_input_snapshot.get(
                "content_sha256")):
        raise StageGateError("functional v2 input snapshot marker identity changed")
    records = document.get("grants")
    if not isinstance(records, list) or not records:
        raise StageGateError("functional v2 input snapshot has no exact grant table")
    grants: list[FrozenGrant] = []
    resolved_root = root.resolve(strict=True)
    repo = repo_root().absolute()
    for row in records:
        if not isinstance(row, Mapping):
            raise StageGateError("functional v2 input snapshot contains a malformed grant")
        declared = str(row.get("path") or "")
        destination = Path(str(row.get("destination") or ""))
        relative = Path(str(row.get("snapshot") or ""))
        if (not declared or not destination.is_absolute() or relative.is_absolute()
                or ".." in relative.parts):
            raise StageGateError("functional v2 input snapshot contains an unsafe grant")
        allowed_destinations = {(repo / declared).absolute(),
                                (repo / "merlin" / declared).absolute()}
        if destination not in allowed_destinations:
            raise StageGateError(f"functional frozen grant has foreign destination: {declared}")
        source = root / relative
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise StageGateError(f"functional frozen grant is absent: {declared}") from exc
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            raise StageGateError(f"functional frozen grant escapes snapshot: {declared}")
        if source.is_symlink():
            raise StageGateError(f"functional frozen grant is linked: {declared}")
        digest = (PC._exact_tree_record(source)["sha256"] if source.is_dir()
                  else _sha256_file(source))
        grants.append(FrozenGrant(declared, destination, source, str(digest)))
    host_source = Path(functional.model_host_package)
    if host_source.is_symlink() or not host_source.is_dir() \
            or resolved_root not in host_source.resolve(strict=True).parents:
        raise StageGateError("functional frozen host lane is absent from the v2 input snapshot")
    grants.append(FrozenGrant(
        "__model_host_lane_snapshot__", host_source, host_source,
        str(PC._exact_tree_record(host_source)["sha256"])))
    return FrozenFunctionalInputs(root, marker, _sha256_file(marker),
                                  str(document["content_sha256"]), tuple(grants))


def frozen_grant_mounts(inputs: FrozenFunctionalInputs) -> list[str]:
    argv: list[str] = []
    for grant in inputs.grants:
        argv += ["--ro-bind", str(grant.source), str(grant.destination)]
    return argv


def _frozen_path_for_destination(inputs: FrozenFunctionalInputs, destination: Path) -> Path:
    destination = destination.absolute()
    candidates: list[tuple[int, Path]] = []
    for grant in inputs.grants:
        if destination == grant.destination or grant.destination in destination.parents:
            candidates.append((len(grant.destination.parts),
                               grant.source / destination.relative_to(grant.destination)))
    if not candidates:
        raise StageGateError(f"functional snapshot did not grant required path: {destination}")
    frozen = max(candidates, key=lambda row: row[0])[1]
    if frozen.is_symlink() or not frozen.exists():
        raise StageGateError(f"functional frozen required path is absent: {destination}")
    return frozen


def _run_authored_identities(count: int) -> tuple[str, ...]:
    """The replicate schedule a run authors for itself, in the corpus' one identity spelling.

    ``PC.expected_perf_cells`` mints exactly these names, so a family whose declaration leaves its
    identities to the run is handed the same cohort the completion set will be enumerated over.
    """
    return tuple(f"r{index:03d}" for index in range(count))


def _declared_analyzer(capsules: Sequence[PC.PerformanceCapsule]) -> tuple[Any, Any]:
    """Resolve the decision procedure the frozen family declares FOR ITSELF.

    Nothing here knows a family name.  The module, function and version come out of the capsules'
    own ``performance.acceptance.analyzer``, so a family that declares an analyzer is handed off with
    no edit to this stage, and one that declares none is REFUSED by name rather than authored
    against some other family's procedure.  Deliberately the same resolution
    ``run_perf_bench._declared_analyzer`` performs, so the authoring boundary and the measurement
    boundary cannot disagree about who decides.
    """
    identities: set[Any] = set()
    for capsule in capsules:
        descriptor = capsule.descriptor if isinstance(capsule.descriptor, Mapping) else {}
        performance = descriptor.get("performance")
        if not isinstance(performance, Mapping):
            raise StageGateError(
                f"frozen capsule {capsule.capsule!r} carries no performance declaration")
        try:
            identity = CLAIM.analyzer_identity(performance)
        except ValueError as exc:
            raise StageGateError(
                f"frozen family {performance.get('family')!r} declares an unusable "
                f"acceptance.analyzer: {exc}") from exc
        if identity is None:
            raise StageGateError(
                f"frozen family {performance.get('family')!r} declares no acceptance.analyzer, so "
                "nothing computes its verdict from its rows; a candidate handoff is refused rather "
                "than authored against another family's analyzer")
        identities.add(identity)
    if len(identities) != 1:
        raise StageGateError(
            f"the frozen performance corpus declares {len(identities)} claim analyzers; exactly "
            "one is required")
    identity = identities.pop()
    return identity, _analyzer_module(identity)


def _analyzer_module(identity: Any) -> Any:
    try:
        return importlib.import_module(identity.module)
    except Exception as exc:
        raise StageGateError(
            f"declared claim analyzer {identity.declared!r} is unavailable: {exc}") from exc


def _preflight_entry(module: Any) -> Any:
    """The module's preflight entry point, found by the shape every claim analyzer publishes."""
    names = sorted(name for name in dir(module)
                   if name.startswith("preflight_") and callable(getattr(module, name, None)))
    if len(names) != 1:
        raise StageGateError(
            f"declared claim analyzer module {module.__name__!r} publishes {len(names)} preflight "
            "entry points; exactly one is required")
    return getattr(module, names[0])


def _analyzer_kwargs(entry: Any, providers: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Only the run facts ``entry``'s OWN signature asks for; an unsatisfied one fails closed."""
    parameters = inspect.signature(entry).parameters
    kwargs: dict[str, Any] = {}
    for name, provide in providers.items():
        if name not in parameters:
            continue
        value = provide()
        if value is None:
            raise StageGateError(
                f"{label} declares a {name!r} parameter and this stage could not derive one; a "
                "handoff authored without it would rest on an assumed fact")
        kwargs[name] = value
    unmet = sorted(name for name, parameter in parameters.items()
                   if parameter.default is inspect.Parameter.empty
                   and parameter.kind is inspect.Parameter.KEYWORD_ONLY
                   and name not in kwargs)
    if unmet:
        raise StageGateError(
            f"{label} requires run facts this stage cannot supply: {', '.join(unmet)}")
    return kwargs


def _declared_family(capsules: Sequence[PC.PerformanceCapsule]) -> str:
    """The one family name the frozen descriptors declare, read from the descriptors themselves."""
    names = {str(capsule.descriptor["performance"].get("family")) for capsule in capsules}
    if len(names) != 1 or not next(iter(names)):
        raise StageGateError(
            f"the frozen performance corpus declares {len(names)} families; exactly one is required")
    return names.pop()


def _declared_replicate_contract(
        capsules: Sequence[PC.PerformanceCapsule]) -> Mapping[str, Any]:
    """The replicate contract the frozen family declares, read BEFORE any preflight runs.

    Some families' preflights take the run's replica schedule as an argument, so the admissible
    count has to come out of the declaration first; reading it here keeps that count a declared fact
    rather than a stage default.
    """
    contracts: dict[bytes, Mapping[str, Any]] = {}
    for capsule in capsules:
        performance = capsule.descriptor.get("performance")
        acceptance = performance.get("acceptance") if isinstance(performance, Mapping) else None
        contract = acceptance.get("replicates") if isinstance(acceptance, Mapping) else None
        if not isinstance(contract, Mapping):
            raise StageGateError(
                f"frozen capsule {capsule.capsule!r} declares no acceptance.replicates contract")
        contracts[_canonical_json(contract)] = contract
    if len(contracts) != 1:
        raise StageGateError(
            "the frozen performance corpus declares more than one replicate contract")
    return next(iter(contracts.values()))


def _admitted_replicate_cohort(
        contract: Mapping[str, Any], requested: int | None, *,
        family: str) -> tuple[int, tuple[str, ...]]:
    """Admit this run's cohort against the family's OWN replicate contract.

    ``exact_count`` is an exact count and ``minimum_count`` a floor -- the same two admissible
    shapes ``run_perf_bench._pk_preflight`` holds a launch to, applied here where the count must
    also be CHOSEN.  Declared ``identities`` bind the run; their absence means the declaration
    leaves the schedule to the run (``identities_authored_by: run``), which is a contract, not a
    gap.  A declaration stating neither count is refused: an unstated cohort is not an open one.
    """
    exact, minimum = contract.get("exact_count"), contract.get("minimum_count")
    if isinstance(exact, int) and not isinstance(exact, bool):
        if requested is not None and (
                isinstance(requested, bool) or not isinstance(requested, int)
                or requested != exact):
            raise StageGateError(
                f"formal replicate override must equal frozen {family} exact_count={exact}")
        count = exact
    elif isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0:
        if requested is None:
            count = minimum
        elif (isinstance(requested, bool) or not isinstance(requested, int)
                or requested < minimum):
            raise StageGateError(
                f"formal replicate count must be at least frozen {family} "
                f"minimum_count={minimum}")
        else:
            count = requested
    else:
        raise StageGateError(
            f"frozen {family} acceptance states neither an exact nor a minimum replicate cohort, "
            "so this run's replica count cannot be admitted")
    identities = contract.get("identities")
    if identities is None:
        return count, _run_authored_identities(count)
    if (not isinstance(identities, list) or any(not isinstance(value, str) for value in identities)
            or len(identities) != count
            or tuple(identities) != _run_authored_identities(count)):
        raise StageGateError(f"frozen {family} acceptance has an invalid exact replicate cohort")
    return count, tuple(identities)


def prepare_formal_pk_claim(
        capsules: Sequence[PC.PerformanceCapsule],
        requested_replicates: int | None = None) -> dict[str, Any]:
    """Admit the exact frozen declaration and derive its formal result cohort.

    Which analyzer decides, and what replica cohort it is entitled to, are read out of the FAMILY'S
    OWN ``performance.acceptance`` -- never out of a family name or a table here -- so a newly
    declared family that names an analyzer produces a candidate handoff with no edit to this stage.
    The descriptors handed to the preflight are the WHOLE frozen corpus, exactly the set
    ``run_perf_bench`` preflights, so the runner's comparison of its own preflight against this
    handoff compares two computations of the same thing.
    """
    identity, module = _declared_analyzer(capsules)
    family = _declared_family(capsules)
    descriptors = [capsule.descriptor for capsule in capsules]
    count, cohort = _admitted_replicate_cohort(
        _declared_replicate_contract(capsules), requested_replicates, family=family)
    entry = _preflight_entry(module)
    kwargs = _analyzer_kwargs(
        entry, {"replicates": lambda: list(cohort)},
        label=f"{identity.module}.{entry.__name__}")
    preflight = entry(descriptors, **kwargs)
    if preflight.get("status") != "READY":
        reasons = preflight.get("refusal_reasons")
        detail = "; ".join(str(value) for value in reasons) if isinstance(reasons, list) else "unknown"
        raise StageGateError(f"frozen {family} formal claim preflight refused: {detail}")
    declaration = preflight.get("declaration")
    if (_canonical_json(declaration) != _canonical_json(_supported_acceptance(module, identity))
            or not isinstance(declaration, Mapping)):
        raise StageGateError(
            f"frozen {family} acceptance differs from the supported claim contract")
    replicate_contract = declaration.get("replicates")
    if not isinstance(replicate_contract, Mapping):
        raise StageGateError(f"frozen {family} acceptance omits its replicate contract")
    if tuple(_preflight_cohort(preflight)) != cohort:
        raise StageGateError(
            f"frozen {family} preflight scheduled a cohort other than the admitted one")
    expected = preflight.get("expected_identities")
    if not isinstance(expected, list) or len(expected) != len(descriptors) * count * 2:
        raise StageGateError(
            f"frozen {family} preflight did not produce its exact L2/L3 identity cohort")
    return copy.deepcopy(preflight)


def _supported_acceptance(module: Any, identity: Any) -> Mapping[str, Any]:
    """The frozen contract the declared analyzer says it is the only decider of."""
    supported = getattr(module, "supported_acceptance", None)
    if not callable(supported):
        raise StageGateError(
            f"declared claim analyzer {identity.declared!r} publishes no supported_acceptance")
    return supported()


def _preflight_cohort(formal_claim: Mapping[str, Any]) -> tuple[str, ...]:
    """The replicate schedule a formal preflight is entitled to.

    The declaration's own ``identities`` where it predeclares them, otherwise the schedule the
    preflight itself returned -- the same two cases, resolved in the same order, that
    ``run_perf_bench`` uses when it compares its preflight against this handoff.
    """
    declaration = formal_claim.get("declaration")
    contract = declaration.get("replicates") if isinstance(declaration, Mapping) else None
    identities = contract.get("identities") if isinstance(contract, Mapping) else None
    if identities is None:
        identities = formal_claim.get("replicates")
    if (not isinstance(identities, list) or not identities
            or any(not isinstance(value, str) for value in identities)
            or tuple(identities) != _run_authored_identities(len(identities))):
        raise StageGateError(
            "the formal claim states no admissible replicate schedule, so no cohort can be "
            "handed off")
    return tuple(identities)


def _family_declarations(
        capsules: Sequence[PC.PerformanceCapsule],
        formal_claim: Mapping[str, Any]) -> tuple[PP.PerfFamily, ...]:
    rows: dict[str, PP.PerfFamily] = {}
    for capsule in capsules:
        performance = capsule.descriptor["performance"]
        comparand, falsifier = performance["comparand"], performance["falsifier"]
        knobs = performance["emitter"]["knobs"]
        fitted: tuple[str, ...] = ()
        if performance["claim"] == "PREDICTS":
            axes = {str(value) for key, value in knobs.items()
                    if isinstance(value, str) and ("axis" in str(key) or "parameter" in str(key))}
            fitted = tuple(sorted(axes or {str(key) for key in knobs}))
        differential = json.dumps({
            "kind": comparand["kind"], "against": comparand["against"],
            "cancels": comparand["cancels"], "demand_equal": comparand["demand_equal"],
        }, sort_keys=True, separators=(",", ":"))
        family = PP.PerfFamily(
            capsule.family, performance["claim"], str(falsifier["negative_control"]),
            str(falsifier["observation"]), differential, fitted,
            copy.deepcopy(performance.get("acceptance")))
        previous = rows.get(capsule.family)
        if previous is not None and previous != family:
            raise StageGateError(f"performance family declaration drifts: {capsule.family}")
        rows[capsule.family] = family
    # The claim-bearing family is the one the preflight names for itself, not a name written here.
    claim_family = formal_claim.get("family")
    declared = rows.get(claim_family) if isinstance(claim_family, str) else None
    if (declared is None or _canonical_json(declared.acceptance) != _canonical_json(
            formal_claim.get("declaration"))):
        raise StageGateError(
            f"{claim_family} family declaration drifts from its formal preflight acceptance")
    return tuple(rows[name] for name in sorted(rows))


def select_e2e_sentinel(functional: PC.FunctionalRun, frozen: FrozenFunctionalInputs,
                        target_experiment: TargetExperiment) -> PP.E2ESentinel:
    """Select the smallest public capsule DECLARING the whole-model L3 cross-lane shape.

    Not "already-passed": nothing here consults the functional run's pass list, and measured, nothing
    could -- a real gemmini public grade is 20/20 over kernel capsules and contains no `kind: model`
    capsule at all. The sentinel is graded by the performance runner's admission gate instead; see
    ``PC.select_full_model_sentinel``.
    """
    selected = PC.select_full_model_sentinel(functional, target_experiment)
    snapshot_repo = (frozen.root / "repo").resolve(strict=True)
    try:
        relative = selected.source_dir.resolve(strict=True).relative_to(snapshot_repo)
    except ValueError as exc:
        raise StageGateError("frozen full-model sentinel is outside the functional snapshot") from exc
    destination = (repo_root() / relative).absolute()
    # Prove the prompt destination is one of the exact frozen grant views.
    if _frozen_path_for_destination(frozen, destination).resolve() != selected.source_dir.resolve():
        raise StageGateError("full-model sentinel does not map to its frozen grant destination")
    return PP.E2ESentinel(
        selected.capsule, str(destination), str(selected.source_dir), selected.source_sha256,
        tuple(selected.descriptor["lanes"]["require"]),
        tuple(selected.descriptor["required_oracle_tiers"]))


_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")


def build_action_registry(candidate: Path,
                          target_experiment: TargetExperiment) -> tuple[BrokerAction, ...]:
    """Create named candidate-manifest actions; no caller-selected executable is accepted."""
    manifest_path = candidate / "manifest.yaml"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StageGateError("functional candidate has no real manifest.yaml")
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries, commands = document.get("entrypoints"), document.get("commands")
    if not isinstance(entries, Mapping) or not isinstance(commands, Mapping) or not commands:
        raise StageGateError("functional candidate manifest has no brokerable commands")
    tool = entries.get("tool")
    if not isinstance(tool, str) or not tool:
        raise StageGateError("functional candidate manifest has no tool entrypoint")
    tool_path = candidate / tool
    if tool_path.is_symlink() or not tool_path.is_file():
        raise StageGateError("functional candidate tool entrypoint is absent or linked")
    actions: list[BrokerAction] = []
    for command_name in sorted(commands):
        row = commands[command_name]
        raw = row.get("argv") if isinstance(row, Mapping) else None
        if not isinstance(raw, list) or not raw or any(not isinstance(v, str) for v in raw):
            raise StageGateError(f"candidate manifest command {command_name!r} has malformed argv")
        argv = tuple(str(tool_path) if value == "{tool}" else value for value in raw)
        placeholders = tuple(sorted({match for value in argv for match in _PLACEHOLDER.findall(value)}))
        if "tool" in placeholders:
            raise StageGateError(f"candidate manifest command {command_name!r} has embedded tool token")
        actions.append(BrokerAction(
            f"candidate-{command_name.replace('_', '-')}", argv, placeholders,
            f"candidate manifest command {command_name}", True))
    for probe in TC.required_tool_probes(target_experiment):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", probe.label).strip("-").lower()
        if not slug:
            raise StageGateError("required target tool probe has no safe action name")
        actions.append(BrokerAction(f"probe-{slug}", ("bash", "-c", probe.cmd), (),
                                    f"descriptor-derived probe for {probe.label}", False))
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise StageGateError("broker action registry contains duplicate names")
    return tuple(actions)


def action_registry_contract(actions: Sequence[BrokerAction], candidate: Path) -> list[dict[str, Any]]:
    """Normalize only the per-round candidate root; every manifest argument remains pinned."""
    root = str(candidate)
    return [{**action.as_dict(), "argv_template": [
        value.replace(root, "{candidate}", 1) if value == root or value.startswith(root + os.sep)
        else value for value in action.argv_template]}
            for action in actions]


def actions_from_registry_contract(rows: object, candidate: Path) -> tuple[BrokerAction, ...]:
    """Rebuild only the sealed action identities needed to replay transcript admission."""
    if not isinstance(rows, list) or not rows:
        raise StageGateError("sealed broker action registry is absent")
    actions: list[BrokerAction] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StageGateError(f"sealed broker action {index} is malformed")
        name, argv = row.get("name"), row.get("argv_template")
        placeholders, purpose, required = (
            row.get("placeholders"), row.get("purpose"), row.get("required"))
        if (not isinstance(name, str) or not name or Path(name).name != name
                or not isinstance(argv, list) or not argv
                or any(not isinstance(value, str) or not value for value in argv)
                or not isinstance(placeholders, list)
                or any(not isinstance(value, str) or not value for value in placeholders)
                or len(placeholders) != len(set(placeholders))
                or not isinstance(purpose, str) or not purpose
                or not isinstance(required, bool)):
            raise StageGateError(f"sealed broker action {index} violates the registry schema")
        found_placeholders = {match for value in argv for match in _PLACEHOLDER.findall(value)}
        if found_placeholders - {"candidate"} != set(placeholders):
            raise StageGateError(f"sealed broker action {index} changes its binding contract")
        expanded = tuple(
            str(candidate) + value.removeprefix("{candidate}")
            if value == "{candidate}" or value.startswith("{candidate}/") else value
            for value in argv)
        actions.append(BrokerAction(
            name, expanded, tuple(placeholders), purpose, required))
    if len({action.name for action in actions}) != len(actions):
        raise StageGateError("sealed broker action registry repeats an action")
    return tuple(actions)


def _host_lane_grant(functional: PC.FunctionalRun) -> PP.HostLaneGrant:
    host = functional.model_host_lane_snapshot
    package = Path(functional.model_host_package)
    return PP.HostLaneGrant(
        str(host["target"]), str(host["run_id"]), str(package), str(host["package_sha256"]),
        str(package / "manifest.yaml"),
        "host-owned capsule/model runner consumes frozen schedule+knobs for scalar/RVV islands; "
        "candidate package handles accelerator regions")


def prepare_prompt_inputs(
        functional: PC.FunctionalRun, frozen_functional: FrozenFunctionalInputs,
        frozen_corpus: PC.FrozenPerformanceCorpus, agent_inputs: AgentInputSnapshot,
        target_experiment: TargetExperiment, actions: Sequence[BrokerAction], *,
        formal_claim: Mapping[str, Any], smoke_replicates: int,
        wall_budget_seconds: int, rounds: int, round_timeout_seconds: int,
        max_tool_calls: int, tool_timeout_seconds: int,
        candidate_path: str = "submission") -> PP.PerfPromptInputs:
    declaration = formal_claim.get("declaration")
    if not isinstance(declaration, Mapping) or not isinstance(declaration.get("replicates"), Mapping):
        raise StageGateError("formal claim omits its frozen replicate declaration")
    formal_identities = _preflight_cohort(formal_claim)
    replicates = len(formal_identities)
    if (isinstance(smoke_replicates, bool) or not isinstance(smoke_replicates, int)
            or smoke_replicates <= 0 or not isinstance(replicates, int)
            or smoke_replicates >= replicates):
        raise StageGateError(
            "smoke replicates must be positive and smaller than the formal cohort")
    cells = tuple(PP.PerfCell(row.family, row.capsule, row.simulator, row.replicate)
                  for row in PC.expected_perf_cells(frozen_corpus.capsules, replicates))
    families = _family_declarations(frozen_corpus.capsules, formal_claim)
    sentinel = select_e2e_sentinel(functional, frozen_functional, target_experiment)
    host = _host_lane_grant(functional)
    tools = tuple(PP.ToolGrant(
        action.name, f"python3 {BROKER_NAME} {action.name}" +
        (" " + " ".join(f"{name}=PATH" for name in action.placeholders)
         if action.placeholders else ""), action.purpose, action.required) for action in actions)
    # The agent-input snapshot is bound at AGENT_CORPUS_MOUNT, so the prompt must name the
    # manifest by its in-sandbox path; advertising the host stage path makes the agent refuse
    # an "absent" mount.  Derived from the same bind the sandbox policies emit; fail closed.
    try:
        agent_manifest_mount = AGENT_CORPUS_MOUNT / agent_inputs.manifest_path.relative_to(
            agent_inputs.root)
    except ValueError as exc:
        raise StageGateError(
            "agent input manifest lies outside the mounted answer-free corpus") from exc
    allowed = (str(FUNCTIONAL_BASE_MOUNT), candidate_path, str(AGENT_CORPUS_MOUNT),
               str(agent_manifest_mount), host.package_path, host.manifest_path,
               sentinel.capsule_path, BROKER_NAME, str(BROKER_RECEIPT_MOUNT),
               str(FUNCTIONAL_INPUT_MANIFEST_MOUNT), str(PERF_CORPUS_MANIFEST_MOUNT),
               *(str(grant.destination) for grant in frozen_functional.grants))
    return PP.PerfPromptInputs(
        target=target_experiment.target, approach="arm4", functional_run_id=functional.run_id,
        functional_submission_sha256=functional.digest,
        frozen_functional_path=str(FUNCTIONAL_BASE_MOUNT),
        frozen_functional_sha256=functional.digest, submission_path=candidate_path,
        submission_initial_sha256=functional.digest,
        functional_public_capsules=functional.public_capsules,
        functional_hidden_capsules=functional.hidden_capsules,
        functional_bundle_snapshot_manifest=str(FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        functional_bundle_snapshot_manifest_sha256=frozen_functional.marker_sha256,
        functional_bundle_snapshot_sha256=frozen_functional.content_sha256,
        workload_root=str(AGENT_CORPUS_MOUNT), workload_manifest=str(PERF_CORPUS_MANIFEST_MOUNT),
        workload_manifest_sha256=frozen_corpus.manifest_sha256,
        workload_capsules_sha256=frozen_corpus.capsules_sha256,
        expected_cells=cells, replicates=replicates,
        formal_replicate_identities=formal_identities,
        formal_claim=copy.deepcopy(dict(formal_claim)), smoke_replicates=smoke_replicates,
        wall_budget_seconds=wall_budget_seconds,
        rounds=rounds, round_timeout_seconds=round_timeout_seconds,
        max_tool_calls=max_tool_calls, tool_timeout_seconds=tool_timeout_seconds,
        families=families, host_lane=host,
        e2e_sentinel=sentinel, tools=tools, allowed_paths=tuple(dict.fromkeys(allowed)),
        execution_broker_path=BROKER_NAME,
        execution_broker_command=f"python3 {BROKER_NAME}",
        broker_receipt_path=str(BROKER_RECEIPT_MOUNT))


def _path_is_answer(path: Path, surfaces: Sequence) -> bool:
    resolved = path.resolve()
    for surface in surfaces:
        answer = Path(surface.path).resolve()
        if resolved == answer or (surface.kind == "dir" and answer in resolved.parents):
            return True
    return False


def build_answer_free_agent_inputs(
        corpus: PC.FrozenPerformanceCorpus, target_experiment: TargetExperiment,
        destination: Path) -> AgentInputSnapshot:
    """Copy only non-answer capsule bytes into the read-only view exposed to the agent.

    The complete frozen corpus remains host-only.  Filtering is derived from the same answer-surface
    registry used by the bwrap coverage proof; no golden filename allow/deny list is maintained here.
    """
    raw_destination = Path(destination)
    if raw_destination.exists() or raw_destination.is_symlink():
        raise StageGateError(f"agent input snapshot already exists: {raw_destination}")
    destination = raw_destination.resolve()
    surfaces = answer_surfaces(target_experiment)
    rows: list[dict[str, Any]] = []
    destination.mkdir(parents=True)
    for member in corpus.capsules:
        original = Path(target_experiment.capsule_corpus).resolve().parent / member.source_relative_path
        for frozen_file in sorted(path for path in member.source_dir.rglob("*") if path.is_file()):
            relative_in_capsule = frozen_file.relative_to(member.source_dir)
            original_file = original / relative_in_capsule
            if _path_is_answer(original_file, surfaces):
                continue
            relative = Path(member.source_relative_path) / relative_in_capsule
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = frozen_file.read_bytes()
            output.write_bytes(payload)
            rows.append({"path": relative.as_posix(), "sha256": _sha256(payload),
                         "n_bytes": len(payload)})
    if not rows:
        raise StageGateError("answer-free performance input view contains zero files")
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
    content_sha = aggregate.hexdigest()
    manifest = {
        "schema_version": 1,
        "source_performance_manifest_sha256": corpus.manifest_sha256,
        "source_performance_corpus_sha256": corpus.capsules_sha256,
        "answer_surface_registry": "merlin.targetgen.sandbox.answer_surfaces",
        "files": rows,
        "content_sha256": content_sha,
        "n_files": len(rows),
        "n_bytes": sum(int(row["n_bytes"]) for row in rows),
    }
    manifest_path = destination / "agent_input_manifest.json"
    payload = _canonical_json(manifest)
    manifest_path.write_bytes(payload)
    for path in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return AgentInputSnapshot(destination, manifest_path, _sha256(payload), content_sha,
                              len(rows), int(manifest["n_bytes"]))


def verify_answer_free_agent_inputs(snapshot: AgentInputSnapshot) -> None:
    if _sha256(snapshot.manifest_path.read_bytes()) != snapshot.manifest_sha256:
        raise StageGateError("answer-free performance input manifest changed")
    document = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    rows = document.get("files")
    if not isinstance(rows, list) or len(rows) != snapshot.n_files or not rows:
        raise StageGateError("answer-free performance input manifest is incomplete")
    aggregate = hashlib.sha256()
    total = 0
    for row in rows:
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise StageGateError("answer-free performance input path escapes its snapshot")
        path = snapshot.root / relative
        if path.is_symlink() or not path.is_file():
            raise StageGateError(f"answer-free performance input is absent or linked: {relative}")
        payload = path.read_bytes()
        if _sha256(payload) != row.get("sha256") or len(payload) != row.get("n_bytes"):
            raise StageGateError(f"answer-free performance input changed: {relative}")
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["sha256"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
        total += len(payload)
    if aggregate.hexdigest() != snapshot.content_sha256 or total != snapshot.n_bytes:
        raise StageGateError("answer-free performance input aggregate digest changed")


def _strip_claude_home(argv: Sequence[str]) -> list[str]:
    """Remove the broad Claude credential/history bind inherited from the shared bwrap base."""
    claude_home = Path.home() / ".claude"
    output: list[str] = []
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and index + 2 < len(argv):
            destination = Path(argv[index + 2])
            if destination == claude_home or claude_home in destination.parents:
                index += 3
                continue
        if option == "--tmpfs" and index + 1 < len(argv):
            destination = Path(argv[index + 1])
            if destination == claude_home or claude_home in destination.parents:
                index += 2
                continue
        output.append(option)
        index += 1
    return output


def inner_execution_policy(target_experiment: TargetExperiment, candidate: Path,
                           agent_inputs: AgentInputSnapshot,
                           frozen_functional: FrozenFunctionalInputs | None = None,
                           functional_base: Path | None = None,
                           frozen_corpus_manifest: Path | None = None) -> AgentSandboxPolicy:
    """Build the credential-free, answer-masked sandbox used by the local tool broker."""
    candidate = _require_real_directory(candidate, label="performance candidate")
    verify_answer_free_agent_inputs(agent_inputs)
    argv = _strip_claude_home(BW.base_argv(candidate, {}, _policy_test_live_inputs=True))
    argv += ["--clearenv", "--setenv", "HOME", "/tmp",
             "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += TC.toolchain_binds(target_experiment)
    if frozen_functional is not None:
        argv += frozen_grant_mounts(frozen_functional)
        argv += ["--ro-bind", str(frozen_functional.marker),
                 str(FUNCTIONAL_INPUT_MANIFEST_MOUNT)]
    if functional_base is not None:
        argv += ["--ro-bind", str(functional_base), str(FUNCTIONAL_BASE_MOUNT)]
    if frozen_corpus_manifest is not None:
        argv += ["--ro-bind", str(frozen_corpus_manifest), str(PERF_CORPUS_MANIFEST_MOUNT)]
    argv += ["--ro-bind", str(agent_inputs.root), str(AGENT_CORPUS_MOUNT)]
    surfaces = answer_surfaces(target_experiment)
    argv = BW.apply_answer_masks(argv, surfaces)
    if "--unshare-net" in argv:
        raise StageGateError("inner execution policy unexpectedly disables required network availability")
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise StageGateError(f"inner performance-tool sandbox exposes answer surfaces: {gaps}")
    joined = " ".join(argv)
    for token in (".codex/auth.json", "AWS_ACCESS_KEY_ID", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if token in joined:
            raise StageGateError(f"inner performance-tool sandbox exposes credential token {token!r}")
    return AgentSandboxPolicy(
        tuple(argv), gaps, "available_not_an_isolation_claim", True, True, True)


def outer_codex_policy(workspace: Path, agent_inputs: AgentInputSnapshot,
                       runtime_binds: Sequence[str], target_experiment: TargetExperiment,
                       frozen_functional: FrozenFunctionalInputs | None = None,
                       functional_base: Path | None = None,
                       control_dir: Path | None = None,
                       frozen_corpus_manifest: Path | None = None) -> AgentSandboxPolicy:
    """Build Codex's filesystem boundary; network/auth exceptions are explicit and recorded."""
    workspace = _require_real_directory(workspace, label="Codex round workspace")
    verify_answer_free_agent_inputs(agent_inputs)
    argv = _strip_claude_home(BW.base_argv(workspace, {}, _policy_test_live_inputs=True))
    argv += ["--clearenv", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
             "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += list(runtime_binds)
    if frozen_functional is not None:
        argv += frozen_grant_mounts(frozen_functional)
        argv += ["--ro-bind", str(frozen_functional.marker),
                 str(FUNCTIONAL_INPUT_MANIFEST_MOUNT)]
    if functional_base is not None:
        argv += ["--ro-bind", str(functional_base), str(FUNCTIONAL_BASE_MOUNT)]
    if control_dir is not None:
        argv += ["--ro-bind", str(control_dir), "/perf-control"]
    if frozen_corpus_manifest is not None:
        argv += ["--ro-bind", str(frozen_corpus_manifest), str(PERF_CORPUS_MANIFEST_MOUNT)]
    argv += ["--ro-bind", str(agent_inputs.root), str(AGENT_CORPUS_MOUNT)]
    surfaces = answer_surfaces(target_experiment)
    argv = BW.apply_answer_masks(argv, surfaces)
    if "--unshare-net" in argv:
        raise StageGateError("outer Codex policy unexpectedly disables required network availability")
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise StageGateError(f"outer Codex sandbox exposes answer surfaces: {gaps}")
    return AgentSandboxPolicy(
        tuple(argv), gaps, "available_not_an_isolation_claim", True, True, True)


def inner_command(policy: AgentSandboxPolicy, target_experiment: TargetExperiment,
                  candidate: Path, argv: Sequence[str], timeout_s: int) -> list[str]:
    """Construct one shell-free payload for the inner broker."""
    if policy.network != "available_not_an_isolation_claim" or not policy.clear_environment:
        raise StageGateError("inner command requires the explicit clear-environment policy")
    if (not argv or any(not isinstance(value, str) or not value or "\0" in value for value in argv)
            or len(argv) > 256 or sum(len(value) for value in argv) > 131_072):
        raise StageGateError("inner tool argv is empty, malformed, or too large")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s <= 0:
        raise StageGateError("inner tool timeout must be a positive integer")
    environment = TC.sandbox_env(target_experiment, candidate)
    return [*policy.argv, "--chdir", str(candidate), "bash", "-c",
            environment + 'exec "$@"', "perf-tool", *argv]


def run_required_tool_probes(policy: AgentSandboxPolicy, target_experiment: TargetExperiment,
                             candidate: Path, *, timeout_s: int = 60) -> list[dict[str, Any]]:
    probes = TC.required_tool_probes(target_experiment)
    if not probes:
        raise StageGateError("inner execution policy derives zero required tool probes")
    rows: list[dict[str, Any]] = []
    for probe in probes:
        command = [*policy.argv, "--chdir", str(candidate), "bash", "-c",
                   TC.sandbox_env(target_experiment, candidate) + probe.cmd]
        proc = subprocess.run(command, cwd=str(repo_root()), capture_output=True, text=True,
                              timeout=timeout_s)
        row = {"label": probe.label, "returncode": proc.returncode,
               "command": probe.cmd, "bind": probe.bind,
               "stdout": (proc.stdout or "")[-400:], "stderr": (proc.stderr or "")[-400:]}
        rows.append(row)
        if proc.returncode != 0:
            # Report the EVIDENCE the gate refused on. A bare rc names nothing: the sandbox failures
            # that actually happen here (a mask whose destination is absent from the bound tree, a
            # missing bind) are diagnosable only from bwrap's own stderr, and swallowing it turns a
            # one-line cause into an all-day hunt.
            raise StageGateError(
                f"required inner-sandbox tool probe {probe.label!r} failed with rc={proc.returncode}"
                f"; command={probe.cmd!r}; stderr={row['stderr'].strip()!r}"
                f"; stdout={row['stdout'].strip()!r}")
    return rows


class _Broker:
    """A bounded localhost bridge from Codex to the credential-free inner bwrap."""

    def __init__(self, policy: AgentSandboxPolicy, target_experiment: TargetExperiment,
                 candidate: Path, actions: Sequence[BrokerAction], receipt_path: Path, *,
                 deadline: float, max_calls: int, max_tool_seconds: int):
        self.policy = policy
        self.target_experiment = target_experiment
        self.candidate = candidate
        self.deadline = deadline
        self.max_calls = max_calls
        self.max_tool_seconds = max_tool_seconds
        self.actions = {action.name: action for action in actions}
        if not self.actions or len(self.actions) != len(actions):
            raise StageGateError("broker requires a non-empty unique action registry")
        self.receipt_path = receipt_path
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if self.receipt_path.exists() or self.receipt_path.is_symlink():
            raise StageGateError(f"broker receipt path is not fresh: {self.receipt_path}")
        self.receipt_path.touch(mode=0o600)
        self.token = secrets.token_urlsafe(32)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        action_name, bindings = request.get("action"), request.get("bindings", {})
        if not isinstance(action_name, str) or action_name not in self.actions:
            raise StageGateError("broker request names an undeclared action")
        if not isinstance(bindings, Mapping):
            raise StageGateError("broker request bindings must be a mapping")
        action = self.actions[action_name]
        if set(bindings) != set(action.placeholders):
            raise StageGateError(
                f"broker action {action_name!r} requires exact bindings {action.placeholders}")
        rendered: dict[str, str] = {}
        for name, value in bindings.items():
            if not isinstance(value, str) or not value or "\0" in value or len(value) > 8192:
                raise StageGateError(f"broker binding {name!r} is malformed")
            path = Path(value)
            if path.is_absolute():
                allowed_roots = (self.candidate, AGENT_CORPUS_MOUNT)
                if not any(path == root or root in path.parents for root in allowed_roots):
                    raise StageGateError(f"broker binding {name!r} escapes declared inputs")
            elif ".." in path.parts:
                raise StageGateError(f"broker binding {name!r} escapes the candidate")
            if name.startswith("output") and path.is_absolute() \
                    and not (path == self.candidate or self.candidate in path.parents):
                raise StageGateError(f"broker output binding {name!r} is not candidate-scoped")
            rendered[name] = value
        raw_argv = [
            _PLACEHOLDER.sub(lambda match: rendered[match.group(1)], value)
            for value in action.argv_template
        ]
        with self._lock:
            if len(self.calls) >= self.max_calls:
                raise StageGateError("inner tool-call budget is exhausted")
            remaining = int(self.deadline - time.monotonic())
            requested = request.get("timeout_s", self.max_tool_seconds)
            if isinstance(requested, bool) or not isinstance(requested, int):
                raise StageGateError("broker timeout must be an integer")
            timeout_s = min(requested, self.max_tool_seconds, remaining)
            if timeout_s <= 0:
                raise StageGateError("performance stage wall-clock budget is exhausted")
            command = inner_command(
                self.policy, self.target_experiment, self.candidate, raw_argv, timeout_s)
            call_index = len(self.calls)
            self.calls.append({"index": call_index, "action": action_name,
                               "bindings": dict(sorted(rendered.items())),
                               "argv_sha256": _sha256(_canonical_json(raw_argv)),
                               "timeout_s": timeout_s, "state": "running"})
        started = time.monotonic()
        try:
            proc = subprocess.run(command, cwd=str(repo_root()), capture_output=True, text=True,
                                  timeout=timeout_s)
            result = {"returncode": proc.returncode, "stdout": (proc.stdout or "")[-1_000_000:],
                      "stderr": (proc.stderr or "")[-1_000_000:],
                      "elapsed_s": round(time.monotonic() - started, 3)}
        except subprocess.TimeoutExpired as exc:
            result = {"returncode": 124, "stdout": str(exc.stdout or "")[-1_000_000:],
                      "stderr": str(exc.stderr or "")[-1_000_000:], "timed_out": True,
                      "elapsed_s": round(time.monotonic() - started, 3)}
        with self._lock:
            self.calls[call_index].update({key: value for key, value in result.items()
                                           if key not in ("stdout", "stderr")})
            self.calls[call_index]["stdout_sha256"] = _sha256(result["stdout"].encode("utf-8"))
            self.calls[call_index]["stderr_sha256"] = _sha256(result["stderr"].encode("utf-8"))
            self.calls[call_index]["state"] = "complete"
            receipt = dict(self.calls[call_index])
            receipt["receipt_schema_version"] = 1
            receipt["bindings_command_sha256"] = _sha256(_canonical_json(
                [f"{key}={value}" for key, value in sorted(rendered.items())]))
            payload = _canonical_json(receipt)
            with self.receipt_path.open("ab", buffering=0) as stream:
                stream.write(payload)
                os.fsync(stream.fileno())
        return result

    @contextlib.contextmanager
    def serving(self) -> Iterator[tuple[str, int]]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                try:
                    if self.path != "/execute" or self.headers.get("X-Perf-Token") != owner.token:
                        self.send_error(403)
                        return
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0 or length > 1_000_000:
                        self.send_error(400)
                        return
                    request = json.loads(self.rfile.read(length))
                    if not isinstance(request, dict):
                        raise StageGateError("broker request must be a JSON mapping")
                    response = owner.execute(request)
                    payload = _canonical_json(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (StageGateError, ValueError, json.JSONDecodeError) as exc:
                    payload = _canonical_json({"error": f"{type(exc).__name__}: {exc}"})
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        try:
            yield "127.0.0.1", int(self._server.server_address[1])
        finally:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)


_BROKER_SHIM = '''#!/usr/bin/env python3
import json, pathlib, sys, urllib.error, urllib.request
cfg = json.loads((pathlib.Path(__file__).parent / ".perf_broker.json").read_text())
argv = sys.argv[1:]
if not argv or argv[0] not in cfg["actions"]:
    raise SystemExit("usage: python3 /perf-control/perf_tool.py ACTION [NAME=VALUE ...]")
action, bindings = argv[0], {}
for item in argv[1:]:
    if "=" not in item:
        raise SystemExit("broker action arguments must be exact NAME=VALUE bindings")
    name, value = item.split("=", 1)
    if not name or name in bindings:
        raise SystemExit("broker action binding is empty or repeated")
    bindings[name] = value
request = urllib.request.Request(
    cfg["url"], data=json.dumps({"action": action, "bindings": bindings,
                                 "timeout_s": cfg["tool_timeout_s"]}).encode(),
    headers={"Content-Type": "application/json", "X-Perf-Token": cfg["token"]}, method="POST")
try:
    with urllib.request.urlopen(request, timeout=cfg["tool_timeout_s"] + 10) as response:
        result = json.load(response)
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode(errors="replace"))
    raise SystemExit(125)
sys.stdout.write(result.get("stdout") or "")
sys.stderr.write(result.get("stderr") or "")
raise SystemExit(int(result.get("returncode", 125)))
'''


def stage_broker_shim(control_dir: Path, *, host: str, port: int, token: str,
                      tool_timeout_s: int, actions: Sequence[BrokerAction]) -> Path:
    if control_dir.is_symlink() or (control_dir.exists() and not control_dir.is_dir()):
        raise StageGateError(f"broker control directory is unsafe: {control_dir}")
    control_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in control_dir.iterdir() if path.name != "receipts.jsonl"]
    if unexpected:
        raise StageGateError(f"broker control directory is not fresh: {control_dir}")
    shim = control_dir / Path(BROKER_NAME).name
    shim.write_text(_BROKER_SHIM, encoding="utf-8")
    shim.chmod(0o555)
    config = control_dir / ".perf_broker.json"
    _write_json(config, {
        "url": f"http://{host}:{port}/execute", "token": token,
        "tool_timeout_s": tool_timeout_s,
        "actions": sorted(action.name for action in actions),
    })
    config.chmod(0o444)
    return shim


def audit_codex_transcript(path: Path, target_experiment: TargetExperiment,
                           candidate: Path,
                           actions: Sequence[BrokerAction] = ()) -> dict[str, Any]:
    """Reject answer reconnaissance and direct execution in translated or native Codex JSONL.

    This audit is a second line of defence.  Answer bytes are absent from the outer mount table, and
    target tools are available only to the inner broker; an audit hit therefore makes the candidate
    non-consumable even if the attempted command failed.  Native Codex emits both ``item.started`` and
    ``item.completed`` for one command, so those envelopes are validated independently but counted once.
    """
    tokens = audit_tokens(target_experiment)
    answer_tokens = tuple(value.lower() for values in tokens.values() for value in values if value)
    entry_tokens: set[str] = set()
    manifest = candidate / "manifest.yaml"
    if manifest.is_file():
        document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        entries = document.get("entrypoints") if isinstance(document, dict) else {}
        if isinstance(entries, dict):
            entry_tokens.update(Path(str(value)).name.lower() for value in entries.values()
                                if isinstance(value, str))
    entry_tokens -= {"python", "python3", "bash", "sh", "env"}
    target_tool_tokens: set[str] = set()
    ordinary_shell = {"awk", "basename", "bash", "cat", "command", "cut", "dirname", "echo",
                      "env", "false", "find", "grep", "head", "ls", "printf", "pwd", "python",
                      "python3", "readlink", "sed", "sh", "tail", "test", "true", "wc", "which"}
    for probe in TC.required_tool_probes(target_experiment):
        first = probe.cmd.strip().split(maxsplit=1)[0] if probe.cmd.strip() else ""
        if first and Path(first).name.lower() not in ordinary_shell:
            target_tool_tokens.add(Path(first).name.lower())
        label_token = re.match(r"[A-Za-z0-9+_.-]+", probe.label)
        if label_token and label_token.group(0).lower() not in ordinary_shell:
            target_tool_tokens.add(label_token.group(0).lower())
    hits: list[dict[str, str]] = []
    broker_invocations: list[dict[str, Any]] = []
    action_names = {action.name for action in actions}
    commands_seen = 0
    current_thread = "no-thread"
    native_commands: dict[tuple[str, str], str] = {}
    known_native_items = {
        "agent_message", "command_execution", "error", "file_change", "mcp_tool_call",
        "reasoning", "web_search",
    }

    def audit_command(command: str, line_number: int) -> None:
        nonlocal commands_seen
        commands_seen += 1
        lowered = command.lower()
        command_words = {
            Path(word.strip("'\";|&()[]{}<>")).name
            for word in lowered.replace("\n", " ").split()
        }
        try:
            outer_words = shlex.split(command)
        except ValueError:
            hits.append({"kind": "malformed_shell_command", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
            outer_words = []
        words: list[str] = []
        payload_text = command
        if (len(outer_words) == 3 and Path(outer_words[0]).name in ("bash", "sh")
                and outer_words[1] in ("-c", "-lc")):
            payload_text = outer_words[2]
        try:
            lexer = shlex.shlex(payload_text, posix=True, punctuation_chars=";&|<>()")
            lexer.whitespace_split = True
            lexer.commenters = ""
            words = list(lexer)
        except ValueError:
            words = []
            hits.append({"kind": "malformed_shell_command", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        brokered = False
        if len(words) >= 3 and words[0] in ("python", "python3") and words[1] == BROKER_NAME:
            action = words[2]
            bindings = words[3:]
            binding_names = [value.split("=", 1)[0] for value in bindings if "=" in value]
            brokered = (action in action_names and len(binding_names) == len(bindings)
                and len(binding_names) == len(set(binding_names)) and all(binding_names)
                and not any(value and all(character in ";&|<>()" for character in value)
                            for value in words)
                and not any(character in payload_text for character in ("`", "$", "\n", "\r")))
            if brokered:
                broker_invocations.append({
                    "line": line_number, "action": action,
                    "bindings_sha256": _sha256(_canonical_json(sorted(bindings))),
                })
            else:
                hits.append({"kind": "invalid_broker_invocation", "line": str(line_number),
                             "command_sha256": _sha256(command.encode("utf-8"))})
        elif BROKER_NAME.lower() in lowered:
            hits.append({"kind": "invalid_broker_invocation", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        if any(token in lowered for token in answer_tokens):
            hits.append({"kind": "answer_reconnaissance", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})

        candidate_root = candidate.resolve(strict=True)

        def candidate_path(value: str, *, must_exist: bool) -> Path | None:
            raw = Path(value)
            possible_paths = ([raw] if raw.is_absolute()
                              else [candidate.parent / raw, candidate / raw])
            for possible in possible_paths:
                try:
                    resolved = possible.resolve(strict=must_exist)
                    resolved.relative_to(candidate_root)
                except (OSError, ValueError):
                    continue
                if not must_exist or resolved.exists():
                    return resolved
            return None

        # A candidate copied to an untracked host path and executed from there would evade the
        # ordinary "path is under candidate" check.  Reject the copy-out itself, regardless of whether
        # the later segment ran successfully.
        segments: list[list[str]] = [[]]
        for value in words:
            if value and all(character in ";&|" for character in value):
                segments.append([])
            else:
                segments[-1].append(value)
        for segment in segments:
            if not segment or Path(segment[0]).name not in ("cp", "install", "mv"):
                continue
            operands = [value for value in segment[1:] if not value.startswith("-")]
            if len(operands) < 2:
                continue
            destination = operands[-1]
            if (any(candidate_path(source, must_exist=True) is not None
                    for source in operands[:-1])
                    and candidate_path(destination, must_exist=False) is None):
                hits.append({"kind": "candidate_code_copied_outside", "line": str(line_number),
                             "command_sha256": _sha256(command.encode("utf-8"))})
                break

        # ``python -c`` has no script argv for the structural execution check below.  Inspect its AST
        # and reject code that opens candidate bytes, including exec(open(...).read()).
        interpreter_c_reads_candidate = False
        if words and Path(words[0]).name.lower() in ("python", "python3") and "-c" in words:
            code_index = words.index("-c") + 1
            if code_index >= len(words):
                interpreter_c_reads_candidate = True
            else:
                try:
                    tree = ast.parse(words[code_index])
                except (SyntaxError, ValueError):
                    tree = None
                if tree is None:
                    interpreter_c_reads_candidate = True
                else:
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                and node.func.id == "open" and node.args
                                and isinstance(node.args[0], ast.Constant)
                                and isinstance(node.args[0].value, str)
                                and candidate_path(node.args[0].value, must_exist=True) is not None):
                            interpreter_c_reads_candidate = True
                            break
            if interpreter_c_reads_candidate:
                hits.append({"kind": "candidate_execution_outside_broker",
                             "line": str(line_number),
                             "command_sha256": _sha256(command.encode("utf-8"))})
        execution_token: str | None = None
        if words:
            executable = Path(words[0]).name.lower()
            if executable in ("python", "python3", "bash", "sh"):
                for value in words[1:]:
                    if value.startswith("-"):
                        if value in ("-c", "-lc", "-m"):
                            break
                        continue
                    execution_token = value
                    break
            else:
                execution_token = words[0]
        direct_candidate = interpreter_c_reads_candidate
        if execution_token:
            token_path = Path(execution_token)
            resolved_execution = candidate_path(execution_token, must_exist=True)
            if resolved_execution is not None and resolved_execution.is_file():
                direct_candidate = True
            if Path(execution_token).name.lower() in entry_tokens:
                direct_candidate = True
        if not brokered and direct_candidate and not interpreter_c_reads_candidate:
            hits.append({"kind": "candidate_execution_outside_broker",
                         "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        if not brokered and target_tool_tokens & command_words:
            hits.append({"kind": "target_tool_outside_broker", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            record = json.loads(line)
        except ValueError:
            hits.append({"kind": "malformed_transcript", "line": str(line_number)})
            continue
        if not isinstance(record, dict):
            hits.append({"kind": "malformed_transcript", "line": str(line_number)})
            continue
        event_type = record.get("type")
        if event_type == "codex_unparsed":
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
        if event_type == "codex_summary":
            unknown_types = record.get("unknown_types")
            if not isinstance(unknown_types, list):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            elif unknown_types:
                hits.append({"kind": "unknown_command_event", "line": str(line_number)})
        if event_type == "thread.started":
            thread_id = record.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            else:
                current_thread = thread_id

        item = record.get("item")
        if event_type in ("item.started", "item.completed"):
            if not isinstance(item, dict):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            elif item.get("type") not in known_native_items:
                hits.append({"kind": "unknown_command_event", "line": str(line_number)})
            elif item.get("type") == "command_execution":
                item_id, command = item.get("id"), item.get("command")
                if (not isinstance(item_id, str) or not item_id
                        or not isinstance(command, str) or not command.strip()):
                    hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                else:
                    key = (current_thread, item_id)
                    previous = native_commands.get(key)
                    if previous is not None and previous != command:
                        hits.append({"kind": "conflicting_command_event", "line": str(line_number)})
                    elif previous is None:
                        native_commands[key] = command
                        audit_command(command, line_number)
        elif isinstance(item, dict) and item.get("type") == "command_execution":
            hits.append({"kind": "unknown_command_event", "line": str(line_number)})

        message = record.get("message")
        if message is not None and not isinstance(message, dict):
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            continue
        content = message.get("content") if isinstance(message, dict) else []
        if content is None:
            content = []
        if not isinstance(content, list):
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                continue
            command = tool_input.get("command")
            if command is None:
                # File edits and other non-command tools legitimately have no command field.
                continue
            if not isinstance(command, str) or not command.strip():
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                continue
            audit_command(command, line_number)
    if commands_seen <= 0:
        hits.append({"kind": "no_command_evidence", "line": "0"})
    return {"clean": not hits, "hits": hits, "commands_seen": commands_seen,
            "candidate": str(candidate), "broker_required": BROKER_NAME,
            "broker_invocations": broker_invocations}


def verify_broker_receipts(path: Path, actions: Sequence[BrokerAction],
                           audit: Mapping[str, Any]) -> dict[str, Any]:
    """Join transcript broker invocations to host-owned append-only completion receipts."""
    if path.is_symlink() or not path.is_file():
        raise StageGateError("host-owned broker receipt stream is absent or linked")
    by_name = {action.name: action for action in actions}
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise StageGateError(f"broker receipt {line_number} is malformed") from exc
        if (not isinstance(row, dict) or row.get("receipt_schema_version") != 1
                or row.get("state") != "complete" or row.get("action") not in by_name
                or not isinstance(row.get("index"), int) or row["index"] != len(rows)
                or not _is_sha256(row.get("argv_sha256"))
                or not _is_sha256(row.get("bindings_command_sha256"))
                or not _is_sha256(row.get("stdout_sha256"))
                or not _is_sha256(row.get("stderr_sha256"))):
            raise StageGateError(f"broker receipt {line_number} violates the action schema")
        rows.append(row)
    invocations = audit.get("broker_invocations")
    if not isinstance(invocations, list):
        raise StageGateError("transcript audit omitted exact broker invocations")
    observed = [(row["action"], row["bindings_command_sha256"]) for row in rows]
    claimed = [(row.get("action"), row.get("bindings_sha256")) for row in invocations
               if isinstance(row, Mapping)]
    if observed != claimed:
        raise StageGateError("host broker receipts do not exactly match transcript invocations")
    successful = {str(row["action"]) for row in rows if row.get("returncode") == 0}
    required = {action.name for action in actions if action.required}
    missing = sorted(required - successful)
    if missing:
        raise StageGateError(f"required broker actions lack successful receipts: {missing}")
    return {"path": str(path), "sha256": _sha256_file(path), "count": len(rows),
            "successful_actions": sorted(successful), "required_actions": sorted(required),
            "all_required_succeeded": True}


def _make_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_symlink():
            raise StageGateError(f"candidate copy contains a symlink: {path}")
        path.chmod(0o755 if path.is_dir() else (path.stat().st_mode | stat.S_IWUSR))
    root.chmod(0o755)


def fresh_round_workspace(source_submission: Path, workspace: Path,
                          expected_sha256: str) -> Path:
    """Create a new round workspace from exactly the previous candidate bytes."""
    raw_workspace = Path(workspace)
    if raw_workspace.exists() or raw_workspace.is_symlink():
        raise StageGateError(f"round workspace is not fresh: {raw_workspace}")
    workspace = raw_workspace.resolve()
    submission = workspace / "submission"
    workspace.mkdir(parents=True)
    shutil.copytree(source_submission, submission, symlinks=False)
    _make_writable(submission)
    observed = hash_tree(submission)["sha256"]
    if observed != expected_sha256:
        raise StageGateError(
            f"fresh round candidate digest {observed} does not match its input {expected_sha256}")
    return submission


def _validate_formal_claim_facts(
        formal: object, replicates: object, formal_replicate_identities: object,
        smoke_replicates: object, cells: object, families: object) -> None:
    # The claim-bearing family and its claim vocabulary are read OFF the record; the record is then
    # held to its own declaration, to the analyzer that declaration names, and to the cells and
    # family rows it was sealed beside.  A family literal here would refuse every family but one.
    family = formal.get("family") if isinstance(formal, Mapping) else None
    if (not isinstance(formal, Mapping) or formal.get("schema_version") != 1
            or not isinstance(family, str) or not family
            or formal.get("claim") not in PP.PERF_CLAIM_VOCABULARY
            or formal.get("status") != "READY" or formal.get("refusal_reasons") != []):
        raise StageGateError("performance candidate omits a READY frozen formal claim")
    declaration = formal.get("declaration")
    if not isinstance(declaration, Mapping):
        raise StageGateError(f"performance candidate {family} acceptance contract drifted")
    identity = _recorded_analyzer_identity(declaration)
    if _canonical_json(declaration) != _canonical_json(
            _supported_acceptance(_analyzer_module(identity), identity)):
        raise StageGateError(f"performance candidate {family} acceptance contract drifted")
    replicate_contract = declaration.get("replicates")
    if not isinstance(replicate_contract, Mapping):
        raise StageGateError(f"performance candidate {family} acceptance omits replicates")
    identities = list(_preflight_cohort(formal))
    exact_count = replicate_contract.get("exact_count")
    minimum_count = replicate_contract.get("minimum_count")
    if (formal_replicate_identities != identities or replicates != len(identities)
            or (isinstance(exact_count, int) and not isinstance(exact_count, bool)
                and exact_count != len(identities))
            or (isinstance(minimum_count, int) and not isinstance(minimum_count, bool)
                and len(identities) < minimum_count)
            or not (isinstance(exact_count, int) and not isinstance(exact_count, bool)
                    or isinstance(minimum_count, int) and not isinstance(minimum_count, bool))):
        raise StageGateError(
            f"performance candidate formal replicates drift from {family} acceptance")
    if (isinstance(smoke_replicates, bool) or not isinstance(smoke_replicates, int)
            or smoke_replicates <= 0 or smoke_replicates >= len(identities)):
        raise StageGateError("performance candidate smoke replicas could masquerade as formal evidence")
    if not isinstance(families, list):
        raise StageGateError("performance candidate formal families are malformed")
    claim_families = [row for row in families
                      if isinstance(row, Mapping) and row.get("family") == family]
    if (len(claim_families) != 1
            or _canonical_json(claim_families[0].get("acceptance")) != _canonical_json(declaration)):
        raise StageGateError(f"performance candidate family omits its exact {family} acceptance")
    cohort = formal.get("cohort")
    expected = formal.get("expected_identities")
    if (not isinstance(cohort, Mapping)
            or (cohort.get("replicates") is not None and cohort.get("replicates") != identities)
            or not isinstance(expected, list) or not expected):
        raise StageGateError(f"performance candidate {family} preflight omits its exact cohort")
    expected_cells: list[dict[str, str]] = []
    for row in expected:
        if not isinstance(row, Mapping):
            raise StageGateError(f"performance candidate {family} preflight has a malformed identity")
        simulator, tier = row.get("simulator"), row.get("tier")
        if ((simulator, tier) not in (("spike", "L2"), ("verilator", "L3"))
                or row.get("family") != family):
            raise StageGateError(f"performance candidate {family} preflight changes L2/L3 semantics")
        expected_cells.append({key: str(row.get(key))
                               for key in ("family", "capsule", "simulator", "replicate")})
    if len({tuple(row.items()) for row in expected_cells}) != len(expected_cells):
        raise StageGateError(f"performance candidate {family} preflight repeats a formal identity")
    if not isinstance(cells, list):
        raise StageGateError("performance candidate formal cells are malformed")
    recorded_claim = [row for row in cells
                      if isinstance(row, Mapping) and row.get("family") == family]
    if sorted(expected_cells, key=_canonical_json) != sorted(
            (dict(row) for row in recorded_claim), key=_canonical_json):
        raise StageGateError(
            f"performance candidate {family} formal identities drift from expected cells")


def _recorded_analyzer_identity(declaration: Mapping[str, Any]) -> Any:
    """The analyzer a RECORDED acceptance block names, resolved from that block alone."""
    try:
        identity = CLAIM.analyzer_identity({"acceptance": declaration})
    except ValueError as exc:
        raise StageGateError(
            f"performance candidate acceptance names an unusable analyzer: {exc}") from exc
    if identity is None:
        raise StageGateError(
            "performance candidate acceptance names no analyzer, so nothing computes the recorded "
            "family's verdict")
    return identity


def validate_candidate_record(document: Mapping[str, Any], *, require_consumable: bool = True) -> dict:
    """Pure schema/boundary validator intended for ``run_perf_bench`` integration."""
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise StageGateError("performance candidate record has an unsupported schema")
    if document.get("kind") != "arm4_performance_candidate":
        raise StageGateError("performance candidate record has a foreign kind")
    target = document.get("target")
    base = document.get("base_functional")
    candidate = document.get("candidate")
    prompt = document.get("prompt")
    corpus = document.get("performance_corpus")
    sandbox = document.get("sandbox")
    broker = document.get("broker")
    agent = document.get("agent")
    admission = document.get("admission")
    if not all(isinstance(value, Mapping)
               for value in (target, base, candidate, prompt, corpus, sandbox, broker, agent,
                             admission)):
        raise StageGateError("performance candidate record omits a required evidence mapping")
    for label, value in (
            ("functional submission", base.get("submission_sha256")),
            ("candidate initial", candidate.get("initial_sha256")),
            ("candidate final", candidate.get("sha256")),
            ("prompt", prompt.get("sha256")),
            ("prompt facts", prompt.get("facts_sha256")),
            ("prompt renderer", prompt.get("renderer_sha256")),
            ("performance manifest", corpus.get("manifest_sha256")),
            ("performance corpus", corpus.get("capsules_sha256")),
            ("agent-input manifest", corpus.get("agent_input_manifest_sha256")),
            ("agent-input content", corpus.get("agent_input_sha256")),
            ("target descriptor", target.get("descriptor_sha256")),
            ("Codex binary", agent.get("codex_binary_sha256")),
            ("transcript", agent.get("transcript_sha256")),
            ("broker registry", broker.get("registry_sha256")),
            ("broker receipt manifest", broker.get("receipt_manifest_sha256"))):
        if not _is_sha256(value):
            raise StageGateError(f"performance candidate record has no valid {label} SHA-256")
    if candidate.get("initial_sha256") != base.get("submission_sha256"):
        raise StageGateError("performance candidate was not forked byte-for-byte from functional")
    if (not isinstance(target.get("name"), str) or not target.get("name")
            or not isinstance(target.get("descriptor"), str) or not target.get("descriptor")):
        raise StageGateError("performance candidate record omits its target descriptor identity")
    _safe_component(str(base.get("run_id") or ""), label="base functional run id")
    bundle_snapshot = base.get("bundle_input_snapshot")
    host_lane = base.get("model_host_lane")
    sentinel = base.get("e2e_sentinel")
    if (not isinstance(bundle_snapshot, Mapping) or not _is_sha256(
            bundle_snapshot.get("content_sha256")) or not _is_sha256(
                bundle_snapshot.get("manifest_sha256"))
            or not isinstance(bundle_snapshot.get("grants"), list)
            or not bundle_snapshot["grants"]):
        raise StageGateError("performance candidate omits the frozen functional grant snapshot")
    if (not isinstance(host_lane, Mapping) or not _is_sha256(host_lane.get("package_sha256"))
            or not isinstance(host_lane.get("integration_seam"), str)):
        raise StageGateError("performance candidate omits its exact functional host lane")
    if (not isinstance(sentinel, Mapping) or not _is_sha256(sentinel.get("capsule_sha256"))
            or set(sentinel.get("required_lanes") or []) != {"on_mesh", "scalar_rvv_lane"}
            or "L3" not in (sentinel.get("required_tiers") or [])):
        raise StageGateError("performance candidate omits its frozen full-model E2E sentinel")
    for evidence, label in ((candidate.get("path"), "sealed candidate path"),
                            (prompt.get("staged_path"), "staged prompt path"),
                            (corpus.get("path"), "frozen performance corpus path"),
                            (agent.get("codex_binary"), "Codex binary path"),
                            (agent.get("transcript"), "combined transcript path")):
        if not isinstance(evidence, str) or not evidence:
            raise StageGateError(f"performance candidate record omits the {label}")
    for field in ("agent_input_files", "agent_input_bytes"):
        value = corpus.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageGateError(f"performance candidate record has no non-vacuous {field}")
    replicates = corpus.get("replicates")
    cells, families = corpus.get("expected_cells"), corpus.get("families")
    formal_claim = corpus.get("formal_claim")
    formal_replicate_identities = corpus.get("formal_replicate_identities")
    smoke_replicates = corpus.get("smoke_replicates")
    prompt_facts = prompt.get("facts")
    if (isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0
            or not isinstance(cells, list) or not cells or not isinstance(families, list)
            or not families or not isinstance(prompt_facts, Mapping)
            or prompt_facts.get("expected_cells") != cells
            or prompt_facts.get("families") != families
            or prompt_facts.get("replicates") != replicates
            or prompt_facts.get("formal_claim") != formal_claim
            or prompt_facts.get("formal_replicate_identities") != formal_replicate_identities
            or prompt_facts.get("smoke_replicates") != smoke_replicates):
        raise StageGateError("performance candidate omits exact prompt cells/families/replicates")
    _validate_formal_claim_facts(
        formal_claim, replicates, formal_replicate_identities, smoke_replicates, cells, families)
    if (candidate.get("read_only") is not True
            or candidate.get("base_submission_overwritten") is not False):
        raise StageGateError("performance candidate record does not prove a separate read-only snapshot")
    delta = candidate.get("delta")
    if (not isinstance(delta, Mapping) or not isinstance(delta.get("changed_files"), list)
            or not isinstance(delta.get("execution_relevant_changed_files"), list)
            or delta.get("changed_file_count") != len(delta["changed_files"])
            or delta.get("execution_relevant_changed_file_count") != len(
                delta["execution_relevant_changed_files"])):
        raise StageGateError("performance candidate record has no structured candidate delta")
    outer = sandbox.get("outer_codex_control_plane")
    inner = sandbox.get("inner_execution_plane")
    if not isinstance(outer, Mapping) or not isinstance(inner, Mapping):
        raise StageGateError("performance candidate record collapses its two sandbox boundaries")
    network_record = "available_not_an_isolation_claim"
    if (outer.get("network") != network_record or outer.get("clear_environment") is not True
            or outer.get("auth_exception") != "isolated_codex_home_explicit_auth_mount"
            or outer.get("session_history_mounted") is not False
            or outer.get("live_target_toolchain_mounted") is not False
            or outer.get("frozen_functional_grants_mounted") is not True
            or outer.get("frozen_grant_manifest_sha256") != bundle_snapshot.get("manifest_sha256")
            or outer.get("answer_surface_gap") != []):
        raise StageGateError("outer Codex control-plane evidence is incomplete")
    if not isinstance(outer.get("bwrap_binary"), str) or not outer.get("bwrap_binary"):
        raise StageGateError("outer Codex control-plane evidence omits the bwrap binary path")
    for evidence, label in ((outer.get("bwrap_binary_sha256"), "bwrap binary"),
                            (outer.get("policy_sha256"), "outer bwrap policy"),
                            (inner.get("policy_sha256"), "inner bwrap policy")):
        if not _is_sha256(evidence):
            raise StageGateError(f"performance candidate record omits the {label} digest")
    if (inner.get("network") != network_record or inner.get("clear_environment") is not True
            or inner.get("credentials") != "none" or inner.get("answer_surface_gap") != []
            or inner.get("candidate_writable") is not True
            or inner.get("corpus_read_only") is not True
            or inner.get("frozen_functional_grants_mounted") is not True
            or inner.get("frozen_grant_manifest_sha256") != bundle_snapshot.get(
                "manifest_sha256")):
        raise StageGateError("inner execution-plane evidence is not credential-free and answer-masked")
    probes = inner.get("tool_probe_results")
    probe_recheck = inner.get("tool_probe_recheck_results")
    if (not isinstance(probes, list) or not probes
            or any(not isinstance(row, Mapping) or row.get("returncode") != 0
                   or not isinstance(row.get("label"), str) or not row.get("label")
                   or not isinstance(row.get("command"), str) or not row.get("command")
                   for row in probes)):
        raise StageGateError("performance candidate record lacks passing inner tool probes")
    if probe_recheck != probes:
        raise StageGateError("performance candidate tool evidence changed between preflight and recheck")
    registry, receipt_rows = broker.get("registry"), broker.get("round_receipts")
    if (not isinstance(registry, list) or not registry
            or _sha256(_canonical_json(registry)) != broker.get("registry_sha256")
            or not isinstance(receipt_rows, list)
            or any(not isinstance(row, Mapping) or not isinstance(row.get("path"), str)
                   or not _is_sha256(row.get("sha256"))
                   or row.get("all_required_succeeded") is not True for row in receipt_rows)
            or broker.get("control_owned_by_harness") is not True
            or broker.get("control_writable_by_agent") is not False
            or not isinstance(broker.get("receipt_manifest"), str)
            or not broker.get("receipt_manifest")):
        raise StageGateError("performance candidate lacks immutable broker registry/receipts")
    required_actions = sorted(str(row.get("name")) for row in registry
                              if isinstance(row, Mapping) and row.get("required") is True)
    if not required_actions or broker.get("required_actions") != required_actions:
        raise StageGateError("performance candidate broker required-action contract is incomplete")
    budget_facts = prompt_facts.get("budgets")
    expected_budget_facts = {
        "wall_budget_seconds": agent.get("wall_budget_seconds"),
        "rounds": agent.get("rounds_requested"),
        "round_timeout_seconds": agent.get("round_timeout_seconds"),
        "max_tool_calls": agent.get("max_tool_calls"),
        "tool_timeout_seconds": agent.get("tool_timeout_seconds"),
    }
    prompt_host = prompt_facts.get("host_lane")
    host_fact_keys = {"target", "package_id", "package_path", "package_sha256",
                      "manifest_path", "integration_seam"}
    if (budget_facts != expected_budget_facts or prompt_facts.get("tools") != registry
            or prompt_facts.get("e2e_sentinel") != sentinel
            or not isinstance(prompt_host, Mapping)
            or set(prompt_host) != host_fact_keys
            or any(prompt_host.get(key) != host_lane.get(key) for key in host_fact_keys)
            or prompt_facts.get("mount_destinations") != outer.get("mount_destinations")):
        raise StageGateError("canonical prompt facts drift from recorded launch enforcement")
    round_rows = agent.get("rounds")
    rounds_requested = agent.get("rounds_requested")
    if (agent.get("driver") != "codex" or not isinstance(round_rows, list) or not round_rows
            or isinstance(rounds_requested, bool) or not isinstance(rounds_requested, int)
            or rounds_requested < len(round_rows)
            or any(not isinstance(row, Mapping) or isinstance(row.get("agent_exit_code"), bool)
                   or not isinstance(row.get("agent_exit_code"), int)
                   or not isinstance(row.get("transcript"), str) or not row.get("transcript")
                   or not isinstance(row.get("audit"), Mapping)
                   or not isinstance(row["audit"].get("clean"), bool)
                   or not isinstance(row["audit"].get("hits"), list)
                   or row["audit"].get("broker_required") != BROKER_NAME
                   or not isinstance(row["audit"].get("broker_invocations"), list)
                   or isinstance(row["audit"].get("commands_seen"), bool)
                   or not isinstance(row["audit"].get("commands_seen"), int)
                   or not _is_sha256(row.get("transcript_sha256")) for row in round_rows)):
        raise StageGateError("performance candidate was not produced by the bounded Codex driver")
    audit = agent.get("audit")
    if (not isinstance(audit, Mapping) or not isinstance(audit.get("clean"), bool)
            or not isinstance(audit.get("hits"), list)
            or audit.get("broker_required") != BROKER_NAME
            or not isinstance(audit.get("broker_invocations"), list)
            or isinstance(audit.get("commands_seen"), bool)
            or not isinstance(audit.get("commands_seen"), int)):
        raise StageGateError("performance candidate lacks structured agent audit evidence")
    if (admission.get("evaluation_performed_by_stage") is not False
            or admission.get("success_declared_by_stage") is not False
            or admission.get("consumer") != "run_perf_bench.py"):
        raise StageGateError("performance authoring stage crossed the evaluation boundary")
    if require_consumable:
        if (admission.get("consumable") is not True or document.get("state") != "sealed"
                or rounds_requested != len(round_rows) or audit.get("clean") is not True
                or audit.get("hits") != []
                or audit.get("commands_seen", 0) <= 0
                or any(row["audit"].get("commands_seen", 0) <= 0
                       or row["audit"].get("clean") is not True for row in round_rows)
                or delta.get("execution_relevant_changed_file_count", 0) <= 0
                or broker.get("all_required_succeeded") is not True
                or len(receipt_rows) != rounds_requested
                or any(row.get("agent_exit_code") != 0 for row in round_rows)):
            raise StageGateError(f"performance candidate is not consumable: {admission.get('refusal')}")
    return dict(document)


def verify_candidate_record(path: Path, *, require_consumable: bool = True,
                            verify_authoring_tools: bool = False,
                            target_experiment: TargetExperiment | None = None) -> dict:
    """Re-hash immutable run artifacts; optionally require the live authoring tools to remain pinned."""
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise StageGateError(f"performance candidate record is absent or linked: {raw_path}")
    path = raw_path.resolve()
    document = validate_candidate_record(json.loads(path.read_text(encoding="utf-8")),
                                         require_consumable=require_consumable)
    if verify_authoring_tools:
        for binary_path, expected, label in (
                (document["target"]["descriptor"], document["target"]["descriptor_sha256"],
                 "target descriptor"),
                (document["agent"]["codex_binary"], document["agent"]["codex_binary_sha256"],
                 "Codex binary"),
                (document["sandbox"]["outer_codex_control_plane"]["bwrap_binary"],
                 document["sandbox"]["outer_codex_control_plane"]["bwrap_binary_sha256"],
                 "bwrap binary")):
            binary = Path(binary_path)
            if binary.is_symlink() or not binary.is_file() or _sha256_file(binary) != expected:
                raise StageGateError(f"{label} bytes do not match the performance candidate record")
    candidate = Path(document["candidate"]["path"])
    if candidate.is_symlink() or not candidate.is_dir():
        raise StageGateError("sealed performance candidate path is absent or linked")
    if hash_tree(candidate)["sha256"] != document["candidate"]["sha256"]:
        raise StageGateError("sealed performance candidate bytes do not match their record")
    for member in (candidate, *candidate.rglob("*")):
        if member.is_symlink():
            raise StageGateError(f"sealed performance candidate contains a symlink: {member}")
        if member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise StageGateError(f"sealed performance candidate is writable: {member}")
    transcript = Path(document["agent"]["transcript"])
    if transcript.is_symlink() or not transcript.is_file():
        raise StageGateError("performance candidate transcript is absent or linked")
    if _sha256(transcript.read_bytes()) != document["agent"]["transcript_sha256"]:
        raise StageGateError("performance candidate transcript bytes do not match their record")
    if target_experiment is not None:
        recorded_actions = actions_from_registry_contract(
            document["broker"]["registry"], candidate)
        observed_audit = audit_codex_transcript(
            transcript, target_experiment, candidate, recorded_actions)
        recorded_audit = document["agent"]["audit"]
        for field in ("clean", "hits", "commands_seen", "broker_required"):
            if observed_audit.get(field) != recorded_audit.get(field):
                raise StageGateError(
                    f"combined Codex transcript audit field {field!r} does not match its record")
    for row in document["agent"]["rounds"]:
        round_transcript = Path(str(row.get("transcript") or ""))
        if (round_transcript.is_symlink() or not round_transcript.is_file()
                or _sha256(round_transcript.read_bytes()) != row["transcript_sha256"]):
            raise StageGateError(
                f"performance candidate round {row.get('round')} transcript bytes changed")
    staged_prompt = Path(document["prompt"]["staged_path"])
    if (staged_prompt.is_symlink() or not staged_prompt.is_file()
            or _sha256(staged_prompt.read_bytes()) != document["prompt"]["sha256"]):
        raise StageGateError("staged performance prompt bytes do not match their record")
    if (_sha256(_canonical_json(document["prompt"]["facts"]))
            != document["prompt"]["facts_sha256"]):
        raise StageGateError("canonical performance prompt facts changed")
    renderer = Path(document["prompt"]["renderer_path"])
    if (renderer.is_symlink() or not renderer.is_file()
            or _sha256_file(renderer) != document["prompt"]["renderer_sha256"]):
        raise StageGateError("performance prompt renderer bytes changed")
    corpus = document["performance_corpus"]
    frozen_manifest = Path(corpus["manifest"])
    if (frozen_manifest.is_symlink() or not frozen_manifest.is_file()
            or _sha256(frozen_manifest.read_bytes()) != corpus["manifest_sha256"]):
        raise StageGateError("frozen performance manifest bytes do not match their record")
    frozen_root = Path(corpus["path"])
    try:
        frozen_loaded = PC.load_frozen_performance_corpus(
            frozen_root, manifest_sha256=str(corpus["manifest_sha256"]),
            capsules_sha256=str(corpus["capsules_sha256"]),
            expected_target=str(document["target"]["name"]))
    except PC.CampaignGateError as exc:
        raise StageGateError(f"frozen performance corpus verification failed: {exc}") from exc
    observed_formal_claim = prepare_formal_pk_claim(
        frozen_loaded.capsules, int(corpus["replicates"]))
    if _canonical_json(observed_formal_claim) != _canonical_json(corpus["formal_claim"]):
        raise StageGateError("frozen performance descriptors changed their formal preflight")
    agent_manifest = Path(corpus["agent_input_manifest"])
    if (agent_manifest.is_symlink() or not agent_manifest.is_file()
            or _sha256(agent_manifest.read_bytes()) != corpus["agent_input_manifest_sha256"]):
        raise StageGateError("answer-free agent input manifest bytes do not match their record")
    verify_answer_free_agent_inputs(AgentInputSnapshot(
        Path(corpus["agent_input_path"]), agent_manifest,
        str(corpus["agent_input_manifest_sha256"]), str(corpus["agent_input_sha256"]),
        int(corpus["agent_input_files"]), int(corpus["agent_input_bytes"])))
    base = Path(document["base_functional"]["snapshot"])
    if base.is_symlink() or not base.is_dir() or hash_tree(base)["sha256"] != document[
            "base_functional"]["submission_sha256"]:
        raise StageGateError("frozen functional base bytes do not match their record")
    bundle = document["base_functional"]["bundle_input_snapshot"]
    bundle_marker = Path(bundle["manifest"])
    if (bundle_marker.is_symlink() or not bundle_marker.is_file()
            or _sha256_file(bundle_marker) != bundle["manifest_sha256"]):
        raise StageGateError("frozen functional grant marker bytes changed")
    PC.verify_functional_host_lane_snapshot(document["base_functional"]["model_host_lane"])
    sentinel = document["base_functional"]["e2e_sentinel"]
    sentinel_source = Path(sentinel["frozen_source_path"])
    if (sentinel_source.is_symlink() or not sentinel_source.is_dir()
            or PC._exact_tree_record(sentinel_source)["sha256"] != sentinel["capsule_sha256"]):
        raise StageGateError("frozen full-model E2E sentinel bytes changed")
    receipt_manifest = Path(document["broker"]["receipt_manifest"])
    if (receipt_manifest.is_symlink() or not receipt_manifest.is_file()
            or _sha256_file(receipt_manifest) != document["broker"]["receipt_manifest_sha256"]):
        raise StageGateError("broker receipt manifest bytes changed")
    receipt_document = json.loads(receipt_manifest.read_text(encoding="utf-8"))
    if (receipt_document.get("schema_version") != 1
            or receipt_document.get("rounds") != document["broker"]["round_receipts"]):
        raise StageGateError("broker receipt manifest disagrees with the candidate record")
    for row in document["broker"]["round_receipts"]:
        receipt_path = Path(str(row.get("path") or ""))
        if (receipt_path.is_symlink() or not receipt_path.is_file()
                or not _is_sha256(row.get("sha256"))
                or _sha256_file(receipt_path) != row["sha256"]
                or row.get("all_required_succeeded") is not True):
            raise StageGateError("host-owned per-round broker receipt bytes changed")
    return document


def verify_candidate_handoff(
        path: Path, *, verify_authoring_tools: bool = False,
        target_experiment: TargetExperiment | None = None) -> VerifiedCandidateHandoff:
    """Return the only normalized stage-to-measurement API after full byte verification."""
    document = verify_candidate_record(
        path, require_consumable=True, verify_authoring_tools=verify_authoring_tools,
        target_experiment=target_experiment)
    base, candidate = document["base_functional"], document["candidate"]
    corpus, prompt = document["performance_corpus"], document["prompt"]
    broker, agent = document["broker"], document["agent"]
    bundle = base["bundle_input_snapshot"]
    return VerifiedCandidateHandoff(
        record_path=Path(path).resolve(), record_sha256=_sha256_file(Path(path).resolve()),
        candidate_path=Path(candidate["path"]), candidate_sha256=str(candidate["sha256"]),
        candidate_initial_sha256=str(candidate["initial_sha256"]),
        functional_run_id=str(base["run_id"]),
        functional_submission_sha256=str(base["submission_sha256"]),
        functional_base_path=Path(base["snapshot"]),
        functional_bundle_snapshot_sha256=str(bundle["content_sha256"]),
        functional_bundle_manifest=Path(bundle["manifest"]),
        functional_bundle_manifest_sha256=str(bundle["manifest_sha256"]),
        target_descriptor=Path(document["target"]["descriptor"]),
        target_descriptor_sha256=str(document["target"]["descriptor_sha256"]),
        corpus_root=Path(corpus["path"]), corpus_manifest=Path(corpus["manifest"]),
        corpus_manifest_sha256=str(corpus["manifest_sha256"]),
        corpus_sha256=str(corpus["capsules_sha256"]), replicates=int(corpus["replicates"]),
        formal_replicate_identities=tuple(
            str(value) for value in corpus["formal_replicate_identities"]),
        formal_claim=copy.deepcopy(dict(corpus["formal_claim"])),
        smoke_replicates=int(corpus["smoke_replicates"]),
        expected_cells=tuple(dict(row) for row in corpus["expected_cells"]),
        families=tuple(dict(row) for row in corpus["families"]),
        host_lane=dict(base["model_host_lane"]), e2e_sentinel=dict(base["e2e_sentinel"]),
        prompt_sha256=str(prompt["sha256"]),
        prompt_facts_sha256=str(prompt["facts_sha256"]),
        prompt_path=Path(prompt["staged_path"]),
        transcript_path=Path(agent["transcript"]),
        transcript_sha256=str(agent["transcript_sha256"]),
        transcript_audit=dict(agent["audit"]),
        receipt_path=Path(broker["receipt_manifest"]),
        receipt_sha256=str(broker["receipt_manifest_sha256"]),
        required_actions=tuple(str(value) for value in broker["required_actions"]),
        tool_evidence={"registry_sha256": broker["registry_sha256"],
                       "round_receipts": broker["round_receipts"],
                       "tool_probe_results": document["sandbox"]["inner_execution_plane"][
                           "tool_probe_results"],
                       "tool_probe_recheck_results": document["sandbox"][
                           "inner_execution_plane"]["tool_probe_recheck_results"]},
        sandbox_evidence={plane: {
            "network": document["sandbox"][plane]["network"],
            "clear_environment": document["sandbox"][plane]["clear_environment"],
            "policy_sha256": document["sandbox"][plane]["policy_sha256"],
        } for plane in ("outer_codex_control_plane", "inner_execution_plane")})


def _import_codex_driver():
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import codex_agent  # noqa: PLC0415
    import run_baseline_qa_loop  # noqa: PLC0415
    return codex_agent, run_baseline_qa_loop


# Linux caps ONE argv entry at MAX_ARG_STRLEN (32 pages = 131072 bytes); ``bash -c <cmd>`` hands the
# whole bwrap policy over as a single entry. The derived answer mask is one ``--ro-bind /dev/null <path>``
# triple per answer surface, so a corpus of a few hundred capsules pushes the inlined policy past the cap
# and the round dies at spawn with "OSError: [Errno 7] Argument list too long: 'bash'" -- a message that
# names nothing in this file. Spill the policy to a NUL-separated file read through ``bwrap --args FD``
# instead, which keeps the command line short no matter how large the mask grows.
_ARG_STRLEN_MAX = 131_072
_ARGS_SPILL_FD = 42            # a fixed high fd; bash opens it read-only via the `FD< file` redirect
_ARGS_SPILL_MARGIN = 4_096     # headroom for the payload the caller appends after the policy


def _spill_bwrap_argv(argv: Sequence[str], spill_dir: Path) -> Path:
    """Write ``argv[1:]`` NUL-separated for ``bwrap --args``, returning the fresh 0600 file."""
    if not argv or argv[0] != "bwrap":
        raise StageGateError("bwrap policy argv does not start with the sandbox executable")
    tail = [str(value) for value in argv[1:]]
    if any("\0" in value for value in tail):
        raise StageGateError("bwrap policy argv contains a NUL and cannot be spilled to --args")
    spill_dir.mkdir(parents=True, exist_ok=True)
    path = spill_dir / f"policy_{secrets.token_hex(16)}.args"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, b"\0".join(value.encode("utf-8") for value in tail))
    finally:
        os.close(fd)
    return path


def bwrap_launch_command(argv: Sequence[str], payload: str, spill_dir: Path,
                         spilled: list[Path]) -> str:
    """The shell command that runs ``argv`` with ``payload`` appended, spilling only when it must.

    Below the single-argument cap the inlined join is returned unchanged, so the small-policy behaviour
    (and everything that reads the command back) is byte-identical. Above it the policy moves into a
    file that ``bwrap --args`` reads from a redirected fd; the caller unlinks every path appended to
    ``spilled`` once the round is over.
    """
    joined = " ".join(str(value) for value in argv)
    if len(joined) + len(payload) + _ARGS_SPILL_MARGIN <= _ARG_STRLEN_MAX:
        return joined + payload
    path = _spill_bwrap_argv(argv, spill_dir)
    spilled.append(path)
    return (f"{argv[0]} --args {_ARGS_SPILL_FD}{payload} "
            f"{_ARGS_SPILL_FD}<{shlex.quote(str(path))}")


def _codex_round(
        workspace: Path, stage_root: Path, prompt: PromptArtifact, target_experiment: TargetExperiment,
        agent_inputs: AgentInputSnapshot, frozen_functional: FrozenFunctionalInputs,
        functional_base: Path, frozen_corpus_manifest: Path, control_dir: Path, *,
        model: str, effort: str, round_index: int,
        timeout_s: int, codex_binary: Path) -> tuple[int, Path, AgentSandboxPolicy]:
    CA, loop = _import_codex_driver()
    from merlin.common import artifacts as artifact_paths  # noqa: PLC0415

    original_bwrap = loop.bwrap_cmd
    original_cache_dir = artifact_paths.cache_dir
    original_codex_bin = os.environ.get("CODEX_BIN")
    captured: dict[str, AgentSandboxPolicy] = {}
    spilled_argv: list[Path] = []

    def stage_local_cache(namespace: str) -> Path:
        if namespace == "codex_home":
            path = stage_root / "codex_homes"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return original_cache_dir(namespace)

    def exact_bwrap(inner: str, ws: Path, _bundle: dict,
                    extra_binds: list[str] | None = None) -> str:
        policy = outer_codex_policy(
            ws, agent_inputs, extra_binds or (), target_experiment, frozen_functional,
            functional_base, control_dir, frozen_corpus_manifest)
        captured["policy"] = policy
        # ``inner`` is already shell-quoted by codex_agent.  Quote only the outer payload boundary.
        payload = " bash -c '" + inner.replace("'", "'\\''") + "'"
        return bwrap_launch_command(policy.argv, payload, stage_root / "_bwrap_argv", spilled_argv)

    loop.bwrap_cmd = exact_bwrap
    artifact_paths.cache_dir = stage_local_cache
    os.environ["CODEX_BIN"] = str(codex_binary)
    try:
        rc, transcript = CA.run_round(
            workspace, stage_root, model, {}, target_experiment, "bwrap", round_index, timeout_s,
            effort=effort, prompt=prompt.text)
    finally:
        loop.bwrap_cmd = original_bwrap
        artifact_paths.cache_dir = original_cache_dir
        if original_codex_bin is None:
            os.environ.pop("CODEX_BIN", None)
        else:
            os.environ["CODEX_BIN"] = original_codex_bin
        for spill in spilled_argv:
            spill.unlink(missing_ok=True)
    policy = captured.get("policy")
    if policy is None:
        raise StageGateError("Codex driver did not construct the required outer bwrap policy")
    return rc, transcript, policy


def run_stage(
        *, functional_runs_root: Path, functional_run_id: str, functional_submission_sha256: str,
        target_experiment: TargetExperiment, stage_root: Path,
        model: str, effort: str, wall_budget_seconds: int, rounds: int, round_timeout_seconds: int,
        max_tool_calls: int, tool_timeout_seconds: int, replicates: int | None = None,
        smoke_replicates: int = 1, families: str = "all",
        capsules: str = "all", codex_binary: str = "codex") -> Path:
    """Run bounded authoring rounds and return the sealed candidate-record path."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in
           (wall_budget_seconds, rounds, round_timeout_seconds, max_tool_calls, tool_timeout_seconds,
            smoke_replicates)):
        raise StageGateError("all performance stage budgets must be positive integers")
    if not model.strip():
        raise StageGateError("an explicit Codex model is required")
    bwrap_binary = _require_executable("bwrap", label="bwrap")
    codex_path = _require_executable(codex_binary, label="Codex")
    descriptor_path = Path(target_experiment.path).resolve()
    bwrap_sha256 = _sha256_file(bwrap_binary)
    codex_sha256 = _sha256_file(codex_path)
    descriptor_sha256 = _sha256_file(descriptor_path)
    if bwrap_binary.name != "bwrap":
        raise StageGateError("the sandbox executable does not resolve to bwrap")
    raw_stage_root = Path(stage_root)
    if raw_stage_root.exists() or raw_stage_root.is_symlink():
        raise StageGateError(f"performance agent stage must use a fresh directory: {raw_stage_root}")
    stage_root = raw_stage_root.resolve()

    functional = PC.inspect_functional_run(
        functional_runs_root, functional_run_id, functional_submission_sha256)
    discovered = PC.discover_performance_corpus(
        target_experiment, families=families, capsules=capsules)
    stage_root.mkdir(parents=True)
    base = PC.materialize_perf_workspace(functional, stage_root / "_frozen_functional")
    frozen_corpus = PC.freeze_performance_corpus(discovered, stage_root / "_frozen_corpus")
    formal_claim = prepare_formal_pk_claim(frozen_corpus.capsules, replicates)
    # The cohort the admitted preflight actually scheduled -- an exact declared count where the
    # family states one, the run-authored schedule where it leaves them to the run.
    replicates = len(_preflight_cohort(formal_claim))
    agent_inputs = build_answer_free_agent_inputs(
        frozen_corpus, target_experiment, stage_root / "_agent_inputs")
    frozen_functional = load_frozen_functional_inputs(functional)
    prepared_actions = build_action_registry(base, target_experiment)
    prepared_action_contract = action_registry_contract(prepared_actions, base)
    prompt_inputs = prepare_prompt_inputs(
        functional, frozen_functional, frozen_corpus, agent_inputs, target_experiment,
        prepared_actions, formal_claim=formal_claim, smoke_replicates=smoke_replicates,
        wall_budget_seconds=wall_budget_seconds,
        rounds=rounds, round_timeout_seconds=round_timeout_seconds,
        max_tool_calls=max_tool_calls, tool_timeout_seconds=tool_timeout_seconds)
    staged_prompt = stage_root / "prompt.txt"
    prompt = materialize_canonical_prompt(prompt_inputs, staged_prompt)
    staged_prompt.chmod(0o444)
    fork = PC.functional_fork(functional)
    fork_check = PC.check_fork(fork, base)
    if fork_check.ok is not True:
        raise StageGateError(f"functional fork is invalid before performance authoring: {fork_check.reason}")

    deadline = time.monotonic() + wall_budget_seconds
    previous_submission = base
    previous_digest = functional.digest
    round_records: list[dict[str, Any]] = []
    transcript_paths: list[Path] = []
    probe_results: list[dict[str, Any]] | None = None
    last_outer: AgentSandboxPolicy | None = None
    refusal: str | None = None
    total_calls = 0
    receipt_records: list[dict[str, Any]] = []
    last_actions: tuple[BrokerAction, ...] = ()
    last_inner: AgentSandboxPolicy | None = None
    for round_index in range(rounds):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            refusal = "performance stage wall-clock budget expired before the next round"
            break
        workspace = stage_root / "agent_workspaces" / f"round_{round_index:02d}"
        candidate = fresh_round_workspace(previous_submission, workspace, previous_digest)
        (workspace / "TASK.md").write_text(prompt.text, encoding="utf-8")
        _write_json(workspace / "STAGE_CONTEXT.json", {
            "functional_run_id": functional.run_id,
            "functional_submission_sha256": functional.digest,
            "performance_manifest_sha256": frozen_corpus.manifest_sha256,
            "performance_corpus_sha256": frozen_corpus.capsules_sha256,
            "agent_corpus_mount": str(AGENT_CORPUS_MOUNT),
            "candidate": "submission",
            "tool_command": f"python3 {BROKER_NAME} ACTION [NAME=VALUE ...]",
            "broker_actions": [action.name for action in prepared_actions],
            "formal_replicates": replicates,
            "formal_replicate_identities": list(
                prompt_inputs.formal_replicate_identities),
            "smoke_replicates": smoke_replicates,
            "round": round_index,
            "rounds": rounds,
            "remaining_wall_budget_seconds": remaining,
        })
        actions = build_action_registry(candidate, target_experiment)
        if action_registry_contract(actions, candidate) != prepared_action_contract:
            refusal = f"candidate manifest action contract drifted before round {round_index}"
            break
        inner = inner_execution_policy(
            target_experiment, candidate, agent_inputs, frozen_functional, base,
            frozen_corpus.manifest_path)
        if probe_results is None:
            probe_results = run_required_tool_probes(inner, target_experiment, candidate)
        control_dir = stage_root / "control" / f"round_{round_index:02d}"
        receipt_path = control_dir / "receipts.jsonl"
        broker = _Broker(inner, target_experiment, candidate, actions, receipt_path,
                         deadline=deadline, max_calls=max_tool_calls - total_calls,
                         max_tool_seconds=tool_timeout_seconds)
        round_timeout = min(round_timeout_seconds, remaining)
        try:
            with broker.serving() as (host, port):
                stage_broker_shim(control_dir, host=host, port=port, token=broker.token,
                                  tool_timeout_s=tool_timeout_seconds, actions=actions)
                rc, transcript, outer = _codex_round(
                    workspace, stage_root, prompt, target_experiment, agent_inputs,
                    frozen_functional, base, frozen_corpus.manifest_path, control_dir, model=model,
                    effort=effort, round_index=round_index, timeout_s=round_timeout,
                    codex_binary=codex_path)
        finally:
            broker_config = control_dir / ".perf_broker.json"
            if broker_config.is_file() and not broker_config.is_symlink():
                broker_config.chmod(0o600)
                broker_config.unlink()
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt_path.chmod(0o444)
        total_calls += len(broker.calls)
        transcript_paths.append(transcript)
        audit = audit_codex_transcript(transcript, target_experiment, candidate, actions)
        try:
            receipt_evidence = verify_broker_receipts(receipt_path, actions, audit)
        except StageGateError as exc:
            receipt_evidence = {"path": str(receipt_path), "error": str(exc),
                                "all_required_succeeded": False}
            refusal = f"Codex round {round_index} failed broker receipt enforcement: {exc}"
        receipt_records.append(receipt_evidence)
        audit_path = stage_root / "rounds" / f"round_{round_index:02d}.audit.json"
        _write_json(audit_path, audit)
        assert_candidate_sealable(candidate)
        observed = hash_tree(candidate)["sha256"]
        round_record = {
            "round": round_index, "workspace": str(workspace), "candidate_sha256": observed,
            "agent_exit_code": rc, "transcript": str(transcript),
            "transcript_sha256": _sha256(transcript.read_bytes()), "audit": audit,
            "broker_calls": broker.calls, "broker_receipts": receipt_evidence,
            "broker_registry_sha256": _sha256(_canonical_json(
                [action.as_dict() for action in actions])), "budget_seconds": round_timeout,
        }
        round_records.append(round_record)
        last_outer = outer
        last_inner = inner
        last_actions = actions
        previous_submission, previous_digest = candidate, observed
        if rc != 0:
            refusal = f"Codex round {round_index} exited with rc={rc}"
            break
        if refusal is not None:
            break
        if not audit["clean"]:
            refusal = f"Codex round {round_index} failed the answer/tool-access audit"
            break

    if (not round_records or not transcript_paths or last_outer is None or last_inner is None
            or probe_results is None or not last_actions):
        raise StageGateError(refusal or "performance stage completed no auditable Codex round")
    probe_recheck_results: list[dict[str, Any]] = []
    try:
        probe_recheck_results = run_required_tool_probes(
            last_inner, target_experiment, previous_submission)
        if probe_recheck_results != probe_results:
            refusal = "inner sandbox tool probe evidence changed during performance authoring"
    except StageGateError as exc:
        refusal = f"inner sandbox tool recheck failed: {exc}"
    if (_sha256_file(bwrap_binary) != bwrap_sha256 or _sha256_file(codex_path) != codex_sha256
            or _sha256_file(descriptor_path) != descriptor_sha256):
        refusal = "descriptor or agent/sandbox executable bytes changed during performance authoring"
    PC.verify_frozen_performance_corpus(frozen_corpus)
    verify_answer_free_agent_inputs(agent_inputs)
    sealed = stage_root / "sealed_candidate" / "submission"
    assert_candidate_sealable(previous_submission)
    sealed_sha = PC.materialize_readonly_tree(previous_submission, sealed)
    if sealed_sha != previous_digest:
        raise StageGateError("sealed performance candidate changed during final copy")
    after = PC.check_fork(fork, base)
    if after.ok is not True:
        refusal = f"functional base fork changed during authoring: {after.reason}"
    exits_clean = all(row["agent_exit_code"] == 0 for row in round_records)
    delta = candidate_delta(base, previous_submission)
    if not delta["execution_relevant_changed_files"]:
        refusal = "Codex produced no execution-relevant candidate change"
    combined_transcript = stage_root / "rounds" / "combined.transcript.jsonl"
    with combined_transcript.open("wb") as stream:
        for path in transcript_paths:
            payload = path.read_bytes()
            stream.write(payload)
            if payload and not payload.endswith(b"\n"):
                stream.write(b"\n")
    combined_transcript.chmod(0o444)
    combined_audit = audit_codex_transcript(
        combined_transcript, target_experiment, previous_submission, prepared_actions)
    round_audits_clean = all(
        row["audit"]["clean"] and row["audit"]["commands_seen"] > 0 for row in round_records)
    audits_clean = round_audits_clean and combined_audit["clean"]
    if combined_audit["commands_seen"] <= 0:
        refusal = "combined Codex transcript contains zero command evidence"
    elif not combined_audit["clean"]:
        refusal = "combined Codex transcript failed the answer/tool-access audit"
    receipts_clean = (len(receipt_records) == rounds and all(
        row.get("all_required_succeeded") is True for row in receipt_records))
    if not receipts_clean:
        refusal = refusal or "required host-owned broker receipt evidence is incomplete"
    consumable = (refusal is None and audits_clean and exits_clean and receipts_clean
                  and len(round_records) == rounds)
    receipt_manifest = stage_root / "control" / "receipt_manifest.json"
    _write_json(receipt_manifest, {"schema_version": 1, "rounds": receipt_records})
    receipt_manifest.chmod(0o444)
    expected_cells = [{"family": cell.family, "capsule": cell.capsule,
                       "simulator": cell.simulator, "replicate": cell.replicate}
                      for cell in prompt_inputs.expected_cells]
    family_facts = [{"family": family.family, "claim": family.claim,
                     "negative_control": family.negative_control,
                     "falsifier_observation": family.falsifier_observation,
                     "differential_basis": family.differential_basis,
                     "fitted_parameters": list(family.fitted_parameters),
                     "acceptance": copy.deepcopy(family.acceptance)}
                    for family in prompt_inputs.families]
    host_lane = {
        "target": prompt_inputs.host_lane.target,
        "package_id": prompt_inputs.host_lane.package_id,
        "package_path": prompt_inputs.host_lane.package_path,
        "package_sha256": prompt_inputs.host_lane.package_sha256,
        "manifest_path": prompt_inputs.host_lane.manifest_path,
        "integration_seam": prompt_inputs.host_lane.integration_seam,
    }
    model_host_record = dict(functional.model_host_lane_snapshot)
    model_host_record.update(host_lane)
    e2e_sentinel = {
        "capsule": prompt_inputs.e2e_sentinel.capsule,
        "capsule_path": prompt_inputs.e2e_sentinel.capsule_path,
        "frozen_source_path": prompt_inputs.e2e_sentinel.frozen_source_path,
        "capsule_sha256": prompt_inputs.e2e_sentinel.capsule_sha256,
        "required_lanes": list(prompt_inputs.e2e_sentinel.required_lanes),
        "required_tiers": list(prompt_inputs.e2e_sentinel.required_tiers),
        "purpose": "functional_L2_L3_admission_not_performance_measurement",
    }
    frozen_grants = [{"declared_path": grant.declared_path,
                      "destination": str(grant.destination), "source": str(grant.source),
                      "source_sha256": grant.source_sha256}
                     for grant in frozen_functional.grants]
    prompt_facts = {
        "replicates": replicates,
        "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
        "formal_claim": copy.deepcopy(formal_claim),
        "smoke_replicates": smoke_replicates,
        "expected_cells": expected_cells,
        "budgets": {"wall_budget_seconds": wall_budget_seconds, "rounds": rounds,
                    "round_timeout_seconds": round_timeout_seconds,
                    "max_tool_calls": max_tool_calls,
                    "tool_timeout_seconds": tool_timeout_seconds},
        "families": family_facts, "host_lane": host_lane, "e2e_sentinel": e2e_sentinel,
        "tools": prepared_action_contract,
        "mount_destinations": list(prompt_inputs.allowed_paths),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "arm4_performance_candidate",
        "state": "sealed" if consumable else "refused",
        "target": {
            "name": target_experiment.target,
            "descriptor": str(descriptor_path),
            "descriptor_sha256": descriptor_sha256,
        },
        "base_functional": {
            "run_id": functional.run_id, "submission_sha256": functional.digest,
            "snapshot": str(base), "fork_before": fork_check.to_dict(), "fork_after": after.to_dict(),
            "bundle_input_snapshot": {
                "path": str(frozen_functional.root),
                "content_sha256": frozen_functional.content_sha256,
                "manifest": str(frozen_functional.marker),
                "manifest_sha256": frozen_functional.marker_sha256,
                "grants": frozen_grants,
            },
            "model_host_lane": model_host_record,
            "e2e_sentinel": e2e_sentinel,
        },
        "candidate": {
            "path": str(sealed), "initial_sha256": functional.digest, "sha256": sealed_sha,
            "rounds_completed": len(round_records), "read_only": True,
            "base_submission_overwritten": False,
            "delta": delta,
        },
        "prompt": {"renderer_path": str(Path(PP.__file__).resolve()),
                   "renderer_sha256": _sha256_file(Path(PP.__file__).resolve()),
                   "staged_path": str(staged_prompt), "sha256": prompt.sha256,
                   "n_bytes": prompt.n_bytes, "facts": prompt_facts,
                   "facts_sha256": _sha256(_canonical_json(prompt_facts))},
        "performance_corpus": {
            "path": str(frozen_corpus.root), "manifest": str(frozen_corpus.manifest_path),
            "manifest_sha256": frozen_corpus.manifest_sha256,
            "capsules_sha256": frozen_corpus.capsules_sha256,
            "agent_input_path": str(agent_inputs.root),
            "agent_input_manifest": str(agent_inputs.manifest_path),
            "agent_input_manifest_sha256": agent_inputs.manifest_sha256,
            "agent_input_sha256": agent_inputs.content_sha256,
            "agent_input_files": agent_inputs.n_files,
            "agent_input_bytes": agent_inputs.n_bytes,
            "replicates": replicates,
            "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
            "formal_claim": copy.deepcopy(formal_claim),
            "smoke_replicates": smoke_replicates,
            "expected_cells": expected_cells,
            "families": family_facts,
        },
        "sandbox": {
            "outer_codex_control_plane": {
                "engine": "bwrap", "network": last_outer.network,
                "clear_environment": last_outer.clear_environment,
                "auth_exception": "isolated_codex_home_explicit_auth_mount",
                "session_history_mounted": False,
                "live_target_toolchain_mounted": False,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": frozen_functional.marker_sha256,
                "mount_destinations": list(prompt_inputs.allowed_paths),
                "answer_surface_gap": list(last_outer.answer_surface_gap),
                "bwrap_binary": str(bwrap_binary),
                "bwrap_binary_sha256": bwrap_sha256,
                "policy_sha256": _sha256(_canonical_json(list(last_outer.argv))),
            },
            "inner_execution_plane": {
                "engine": "bwrap", "network": last_inner.network,
                "clear_environment": last_inner.clear_environment, "credentials": "none",
                "candidate_writable": last_inner.candidate_writable,
                "corpus_read_only": last_inner.corpus_read_only,
                "answer_surface_gap": list(last_inner.answer_surface_gap),
                "required_tools": [probe.label for probe in TC.required_tool_probes(target_experiment)],
                "tool_probe_results": probe_results,
                "tool_probe_recheck_results": probe_recheck_results,
                "broker_calls": total_calls,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": frozen_functional.marker_sha256,
                "policy_sha256": _sha256(_canonical_json(list(last_inner.argv))),
            },
        },
        "broker": {
            "shim_mount": BROKER_NAME, "shim_sha256": _sha256(
                _BROKER_SHIM.encode("utf-8")),
            "registry": prepared_action_contract,
            "registry_sha256": _sha256(_canonical_json(prepared_action_contract)),
            "receipt_manifest": str(receipt_manifest),
            "receipt_manifest_sha256": _sha256_file(receipt_manifest),
            "round_receipts": receipt_records,
            "required_actions": sorted(action.name for action in prepared_actions if action.required),
            "all_required_succeeded": receipts_clean,
            "control_owned_by_harness": True,
            "control_writable_by_agent": False,
        },
        "agent": {
            "driver": "codex", "model": model, "effort": effort,
            "codex_binary": str(codex_path), "codex_binary_sha256": codex_sha256,
            "wall_budget_seconds": wall_budget_seconds, "round_timeout_seconds": round_timeout_seconds,
            "max_tool_calls": max_tool_calls, "tool_timeout_seconds": tool_timeout_seconds,
            "rounds_requested": rounds, "rounds": round_records,
            "transcript": str(combined_transcript),
            "transcript_sha256": _sha256(combined_transcript.read_bytes()),
            "audit": combined_audit,
        },
        "admission": {
            "consumable": consumable,
            "refusal": refusal,
            "evaluation_performed_by_stage": False,
            "success_declared_by_stage": False,
            "consumer": "run_perf_bench.py",
        },
    }
    record_path = stage_root / "performance_candidate.json"
    _write_json(record_path, record)
    record_path.chmod(0o444)
    validate_candidate_record(record, require_consumable=consumable)
    if consumable:
        verify_candidate_record(
            record_path, verify_authoring_tools=True, target_experiment=target_experiment)
    return record_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument("--run-id", required=True,
                        help="fresh directory name under the Arm4 performance run root")
    parser.add_argument("--model", required=True, help="explicit Codex model slug")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--wall-budget-seconds", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--replicates", type=int, default=None,
        help="optional assertion; must equal the frozen formal acceptance exact_count")
    parser.add_argument(
        "--smoke-replicates", type=int, default=1,
        help="non-claim diagnostic count; must be smaller than the formal cohort")
    parser.add_argument("--round-timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=100)
    parser.add_argument("--tool-timeout-seconds", type=int, default=900)
    parser.add_argument("--families", default="all")
    parser.add_argument("--capsules", default="all")
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--descriptor", type=Path, default=(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"))
    args = parser.parse_args(argv)
    run_id = _safe_component(args.run_id, label="performance stage run id")
    target_experiment = load_target_experiment(args.descriptor)
    functional_root = runs_root(target_experiment.target, "capsule-bench")
    stage_root = runs_root(target_experiment.target, "perf-bench") / "agent_stages" / run_id
    try:
        record = run_stage(
            functional_runs_root=functional_root,
            functional_run_id=args.functional_run_id,
            functional_submission_sha256=args.functional_submission_sha256,
            target_experiment=target_experiment, stage_root=stage_root,
            model=args.model, effort=args.effort, wall_budget_seconds=args.wall_budget_seconds,
            rounds=args.rounds, round_timeout_seconds=args.round_timeout_seconds,
            replicates=args.replicates,
            smoke_replicates=args.smoke_replicates,
            max_tool_calls=args.max_tool_calls, tool_timeout_seconds=args.tool_timeout_seconds,
            families=args.families, capsules=args.capsules, codex_binary=args.codex_binary)
    except (StageGateError, PC.CampaignGateError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 2
    document = json.loads(record.read_text(encoding="utf-8"))
    if document["admission"]["consumable"] is not True:
        print(f"NO-GO: {document['admission']['refusal']}\nrecord: {record}", file=sys.stderr)
        return 2
    print(f"SEALED: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
