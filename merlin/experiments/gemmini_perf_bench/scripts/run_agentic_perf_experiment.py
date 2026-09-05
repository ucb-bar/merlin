#!/usr/bin/env python3
"""Resume-safe, three-trial orchestrator for the arms-rigorous performance experiment.

The orchestrator owns ordering and admission; existing modules own every substantive gate.  It commits
the hidden holdout before authoring, seals three identically configured agent trials, regrades each
candidate on the complete public+hidden functional L3 suite, reveals the holdout, predeclares all
paired measurements, and evaluates every GSIM cell without best-of selection or failed-cell dropping.
Paid agents and simulators are launched only through an injected command runner.

The authoring trials and the paired measurement matrix are independent children, so a launch may
declare how many of them may run at once with ``MERLIN_PERF_CAMPAIGN_FANOUT`` (default 1, the fully
serial campaign; 6 gives both phases their full width).  Concurrency changes only the wall clock:
the checkpoint chain, the trial evidence and the measurement matrix are recorded on the main thread
in fixed order, so the record does not depend on which child finishes first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import yaml

import _pbcommon as PB
import perf_agent_stage as PAS
import perf_experiment_stats as STATS
import perf_gsim_gate as GATE
import perf_holdout_corpus as HOLDOUT
import heldout_gsim_qualification as HQUAL
import run_paired_perf_bench as PAIRED
from merlin.benchharness import hash_tree, runs_root
from merlin.common.paths import build_dir
from merlin.targetgen import capsule_runner as CAPSULES
from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.contract.materialize import public_capsules_for
from merlin.targetgen.target_experiment import load_target_experiment

HERE = Path(__file__).resolve().parent
TRIALS = ("trial_00", "trial_01", "trial_02")
REPLICATES = ("r000", "r001", "r002")
SCHEMA = "merlin.agentic-performance-experiment.v1"


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    experiment_id: str
    root: Path
    functional_run_id: str
    functional_submission_sha256: str
    descriptor: Path
    rtl_facts: Path
    perf_profile: Path
    gsim_certificate: Path
    gsim_certificate_sha256: str
    model: str
    effort: str
    wall_budget_seconds: int
    rounds: int
    round_timeout_seconds: int
    max_tool_calls: int
    tool_timeout_seconds: int
    smoke_replicates: int
    holdout_count: int
    measurement_timeout: int
    gsim_max_cycles: int | None = None
    codex_binary: str = "codex"
    hardware_counters: bool = False
    functional_gsim_certificate: Path | None = None
    functional_gsim_certificate_sha256: str | None = None
    heldout_qualification_timeout: int = 3600
    generalization_count: int = 4
    telemetry_price_table: Path | None = None
    chia_python: Path | None = None
    # Named completeness gaps in the functional baseline that this campaign accepts. Empty means the
    # gate must pass on its own terms. Recorded in the manifest so a result never loses the condition
    # it was produced under.
    waive_functional_gate: tuple[str, ...] = ()
    # Exact performance members to measure, or "all". Named when a member cannot be CERTIFIED --
    # the reference simulator cannot execute it, so no equivalence evidence exists for it -- which
    # would otherwise let one unrunnable capsule block the entire corpus. Recorded in the manifest,
    # so what a result was measured over is never in doubt.
    perf_capsules: str = "all"
    perf_families: str = "all"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CommandResult]


@dataclass(frozen=True)
class FunctionalCapsule:
    """One exact descriptor admitted by the canonical formal grading policy."""

    name: str
    kind: str
    manifest: Path
    manifest_sha256: str
    workload_sha256: str


@dataclass(frozen=True)
class FunctionalGradeCohort:
    """The public and hidden descriptors that actually enter the scored denominator."""

    public: tuple[FunctionalCapsule, ...]
    hidden: tuple[FunctionalCapsule, ...]
    public_source_count: int
    hidden_source_count: int


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _is_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _verify_chia_launch_receipt() -> dict[str, Any]:
    """Verify CHIA's runtime resource assignment before any campaign mutation or paid turn."""
    plan_sha256 = os.environ.get("MERLIN_CHIA_ENVELOPE_PLAN_SHA256")
    receipt_sha256 = os.environ.get("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256")
    receipt_value = os.environ.get("MERLIN_CHIA_LAUNCH_RECEIPT")
    if not (_is_sha(plan_sha256) and _is_sha(receipt_sha256) and receipt_value):
        raise ExperimentError(
            "actual campaign requires a content-addressed CHIA launch receipt, not only a plan id")
    receipt_path = Path(receipt_value)
    if (receipt_path.is_symlink() or not receipt_path.is_file()
            or receipt_path.stat().st_mode & 0o222
            or _sha_file(receipt_path) != receipt_sha256):
        raise ExperimentError("CHIA launch receipt is absent, linked, writable, or has changed")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ExperimentError("CHIA launch receipt is malformed") from exc
    plan = receipt.get("plan") or {}
    unhashed_plan = {key: value for key, value in plan.items() if key != "sha256"}
    required = receipt.get("required_resources") or {}
    assigned = receipt.get("assigned_resources") or {}
    current_command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    recorded_command = receipt.get("command")
    command_matches = (
        isinstance(recorded_command, list) and len(recorded_command) == len(current_command)
        and Path(str(recorded_command[0])).resolve() == Path(current_command[0]).resolve()
        and Path(str(recorded_command[1])).resolve() == Path(current_command[1]).resolve()
        and recorded_command[2:] == current_command[2:])
    wrapper_record = receipt.get("wrapper") or {}
    chia_record = receipt.get("chia_trace") or {}
    command_artifacts = receipt.get("command_artifacts") or []
    expected_command_artifacts = [
        {"index": index, "path": str(Path(current_command[index]).resolve()),
         "sha256": _sha_file(Path(current_command[index]))}
        for index in (0, 1)]
    wrapper = (HERE / "chia_agentic_perf_experiment.py").resolve()
    chia_path = Path(str(chia_record.get("path") or ""))
    if (receipt.get("schema") != "merlin.chia-agentic-perf-launch.v1"
            or receipt.get("status") != "assigned_before_coordinator"
            or receipt.get("plan_sha256") != plan_sha256
            or plan.get("sha256") != plan_sha256
            or _sha_bytes(_canonical(unhashed_plan)) != plan_sha256
            or plan.get("command") != recorded_command
            or required != {"codex_slots": 1, "gsim_slots": 1}
            or any(not isinstance(assigned.get(name), (int, float))
                   or isinstance(assigned.get(name), bool)
                   or float(assigned[name]) < 1.0 for name in required)
            or not command_matches
            or command_artifacts != expected_command_artifacts
            or plan.get("command_artifacts") != command_artifacts
            or wrapper_record.get("path") != str(wrapper)
            or wrapper_record.get("sha256") != _sha_file(wrapper)
            or plan.get("wrapper") != wrapper_record
            or chia_path.is_symlink() or not chia_path.is_file()
            or chia_record.get("sha256") != _sha_file(chia_path)
            or plan.get("chia_trace") != chia_record):
        raise ExperimentError("CHIA launch receipt does not attest this exact assigned invocation")
    return {"path": str(receipt_path.resolve()), "sha256": receipt_sha256,
            "plan_sha256": plan_sha256, "required_resources": dict(required),
            "assigned_resources": dict(assigned), "wrapper": dict(wrapper_record),
            "chia_trace": dict(chia_record), "command": list(recorded_command),
            "command_artifacts": [dict(row) for row in command_artifacts]}


def _verify_resume_chia_identity(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse a campaign resume under a different predeclared command or source stack.

    Receipt location, receipt digest, and runtime assignment are deliberately per-dispatch evidence.
    The plan and content-addressed source identities are the stable campaign treatment.
    """
    stable_keys = ("plan_sha256", "required_resources", "wrapper", "chia_trace",
                   "command", "command_artifacts")
    if any(saved.get(key) != current.get(key) for key in stable_keys):
        raise ExperimentError(
            "CHIA resume command/source identity differs from the saved predeclaration")


def _verify_resume_declaration(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Keep every predeclared treatment/gate byte stable, not only the CHIA envelope."""
    saved_declaration = saved.get("declaration")
    if not isinstance(saved_declaration, Mapping) or dict(saved_declaration) != dict(current):
        raise ExperimentError("campaign declaration differs from the saved predeclaration")


def _telemetry_treatment_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Project the exact executable and telemetry implementation bytes into the trial contract."""
    sources = preflight.get("sources") or {}
    if (not isinstance(sources, Mapping)
            or set(sources) != PAS.TELEMETRY_TREATMENT_SOURCES
            or any(not isinstance(source, Mapping) or not _is_sha(source.get("sha256"))
                   for source in sources.values())):
        raise ExperimentError("agent telemetry preflight lacks its complete treatment source identity")
    source_sha256 = {str(name): str(source["sha256"])
                     for name, source in sorted(sources.items())}
    model_resolution = preflight.get("model_resolution") or {}
    if (not isinstance(model_resolution, Mapping)
            or not isinstance(model_resolution.get("requested_model"), str)
            or not model_resolution.get("requested_model")
            or not isinstance(model_resolution.get("resolved_model"), str)
            or not model_resolution.get("resolved_model")
            or not isinstance(model_resolution.get("codex_model_map"), str)):
        raise ExperimentError("agent telemetry preflight lacks its exact model resolution identity")
    return {
        "telemetry_preflight_sha256": _sha_bytes(_canonical(preflight)),
        "codex_binary_sha256": source_sha256["codex_binary"],
        "authoring_stage_sha256": source_sha256["performance_authoring_stage"],
        "telemetry_source_sha256": source_sha256,
        "requested_model": str(model_resolution["requested_model"]),
        "resolved_model": str(model_resolution["resolved_model"]),
        "codex_model_map": str(model_resolution["codex_model_map"]),
    }


def _verify_trial_treatments(
        handoffs: Mapping[str, PAS.VerifiedCandidateHandoff],
        expected: Mapping[str, Any]) -> None:
    """Require all three trials to use the one predeclared executable/telemetry treatment."""
    observed = {}
    for trial, handoff in handoffs.items():
        identity = dict(handoff.agent_contract.get("treatment_identity") or {})
        if identity != dict(expected):
            raise ExperimentError(f"{trial} agent treatment differs from predeclaration")
        observed[trial] = identity
    if len(handoffs) != len(TRIALS) or set(handoffs) != set(TRIALS):
        raise ExperimentError("agent treatment verification requires all three declared trials")
    if len({_sha_bytes(_canonical(identity)) for identity in observed.values()}) != 1:
        raise ExperimentError("three agent trials did not use one identical treatment")


def _verify_live_agent_treatment(config: Config, expected: Mapping[str, Any]) -> None:
    """Re-attest treatment bytes immediately before spending the next paid agent trial."""
    try:
        current = PAS.telemetry_preflight(
            model=config.model, price_table=config.telemetry_price_table,
            codex_binary=config.codex_binary)
        identity = _telemetry_treatment_identity(current)
    except PAS.StageGateError as exc:
        raise ExperimentError(f"live agent treatment preflight failed: {exc}") from exc
    if identity != dict(expected):
        raise ExperimentError("live agent treatment differs from the saved predeclaration")


def _verify_trial_contract(
        trial: str, handoff: PAS.VerifiedCandidateHandoff,
        expected: Mapping[str, Any]) -> None:
    """Reject a complete/stale child unless it exactly implements its declared paid trial."""
    if dict(handoff.agent_contract) != dict(expected):
        raise ExperimentError(f"{trial} agent stage differs from its predeclared trial contract")


def _safe(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise ExperimentError(f"{label} must be one safe path component")
    return value


def _config_document(config: Config) -> dict[str, Any]:
    document = asdict(config)
    for key in ("root", "descriptor", "rtl_facts", "perf_profile", "gsim_certificate"):
        document[key] = str(document[key])
    if document["functional_gsim_certificate"] is not None:
        document["functional_gsim_certificate"] = str(document["functional_gsim_certificate"])
    for key in ("telemetry_price_table", "chia_python"):
        if document[key] is not None:
            document[key] = str(document[key])
    document["trials"] = list(TRIALS)
    document["replicates"] = list(REPLICATES)
    document["selection"] = "all_three_trials_no_best_of_no_failed_cell_dropping"
    return document


class Checkpoints:
    """Append-only, content-addressed state; resume discovers and validates the unique chain."""

    def __init__(self, root: Path, config_sha256: str):
        self.root, self.config_sha256 = Path(root), config_sha256
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("checkpoint.*.json")):
            raw = path.read_bytes()
            digest = _sha_bytes(raw)
            if path.name != f"checkpoint.{len(rows):04d}.{digest}.json":
                raise ExperimentError(f"checkpoint name/order is invalid: {path}")
            row = json.loads(raw)
            if (row.get("config_sha256") != self.config_sha256
                    or row.get("index") != len(rows)
                    or row.get("previous_sha256") != (rows[-1]["sha256"] if rows else None)):
                raise ExperimentError(f"checkpoint chain is invalid: {path}")
            rows.append({**row, "sha256": digest, "path": str(path.resolve())})
        return rows

    def append(self, stage: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        rows = self.load()
        if any(row["stage"] == stage for row in rows):
            raise ExperimentError(f"checkpoint stage is duplicated: {stage}")
        body = {"schema": SCHEMA, "index": len(rows), "stage": stage,
                "config_sha256": self.config_sha256,
                "previous_sha256": rows[-1]["sha256"] if rows else None,
                "evidence": dict(evidence)}
        payload, digest = _canonical(body), _sha_bytes(_canonical(body))
        path = self.root / f"checkpoint.{len(rows):04d}.{digest}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return {**body, "sha256": digest, "path": str(path.resolve())}

    def evidence(self, stage: str) -> dict[str, Any] | None:
        found = [row for row in self.load() if row["stage"] == stage]
        if len(found) > 1:
            raise ExperimentError(f"checkpoint stage is duplicated: {stage}")
        return dict(found[0]["evidence"]) if found else None


def subprocess_runner(argv: Sequence[str], cwd: Path,
                      environment: Mapping[str, str]) -> CommandResult:
    completed = subprocess.run(list(argv), cwd=cwd, env=dict(environment),
                               capture_output=True, text=True)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def child_environment(config: Config, certificate: GATE.CertificateRecord) -> dict[str, str]:
    """Pin GSIM selection for child stages; ambient engine paths/cycle caps never decide a run."""
    environment = {**os.environ, "MERLIN_TARGET_EXPERIMENT": str(config.descriptor.resolve()),
                   "MERLIN_GEMMINI_GSIM_EMU": certificate.pins["gsim_binary"]["path"],
                   # The binary pin alone makes GSIM available but does not force the equal-fidelity
                   # engine policy to choose it when another RTL engine is also installed. This run's
                   # certificate is specifically about the pinned GSIM build, so bind both operator L3
                   # and model tile dispatch to that engine.
                   "MERLIN_REQUIRED_RTL_ENGINE": "gsim"}
    if config.gsim_max_cycles is None:
        environment.pop("MERLIN_GEMMINI_GSIM_MAXCYCLES", None)
    else:
        environment["MERLIN_GEMMINI_GSIM_MAXCYCLES"] = str(config.gsim_max_cycles)
    if config.telemetry_price_table is None:
        raise ExperimentError("the child environment lacks a pinned telemetry price table")
    environment["AET_PRICE_TABLE"] = str(Path(config.telemetry_price_table).resolve())
    return environment


def snapshot_contract_inputs(config: Config) -> tuple[Config, dict[str, Any]]:
    """Copy mutable tracked facts/contracts into immutable content-addressed experiment evidence."""
    destination_root = Path(config.root) / "preflight_inputs"
    destination_root.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {}
    replacements: dict[str, Path] = {}
    price_source = config.telemetry_price_table
    if price_source is None:
        try:
            from merlin.common.paths import _dotenv  # noqa: PLC0415
            raw = (os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or "")
        except Exception:  # noqa: BLE001
            raw = os.environ.get("AET_PRICE_TABLE", "")
        price_source = Path(raw) if raw else None
    if price_source is None:
        raise ExperimentError("AET_PRICE_TABLE is required for content-addressed telemetry pricing")
    for field, source in (("rtl_facts", config.rtl_facts),
                          ("perf_profile", config.perf_profile),
                          ("telemetry_price_table", price_source)):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ExperimentError(f"{field} snapshot source is absent, linked, or non-regular: {source}")
        payload = source.read_bytes()
        digest = _sha_bytes(payload)
        suffix = source.suffix if source.suffix else ".evidence"
        destination = destination_root / f"{field}.{digest}{suffix}"
        if destination.exists():
            if (not destination.is_file() or destination.is_symlink()
                    or destination.read_bytes() != payload):
                raise ExperimentError(f"content-addressed {field} snapshot is inconsistent")
        else:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
        destination.chmod(0o444)
        replacements[field] = destination.resolve()
        evidence[field] = {"source": str(source.resolve()), "snapshot": str(destination.resolve()),
                           "sha256": digest}
    return replace(config, **replacements), evidence


def _functional_capsule(cap: Mapping[str, Any]) -> FunctionalCapsule:
    name = cap.get("name")
    kind = cap.get("kind")
    directory = cap.get("__dir__")
    if (not isinstance(name, str) or not name or not isinstance(kind, str) or not kind
            or not isinstance(directory, str) or not directory):
        raise ExperimentError("functional capsule discovery returned a malformed name/kind/directory")
    manifest = (Path(directory) / "capsule.yaml").resolve()
    if not manifest.is_file():
        raise ExperimentError(f"functional capsule descriptor is absent: {manifest}")
    workload = PAIRED.CERTPROD.derive_workload(manifest)
    return FunctionalCapsule(
        name=name, kind=kind, manifest=manifest, manifest_sha256=_sha_file(manifest),
        workload_sha256=GATE.workload_sha256(workload))


def _functional_grade_cohort(target: object) -> FunctionalGradeCohort:
    """Reproduce the official grader's descriptor-driven public and hidden admission.

    Public grading consumes ``public_capsules_for(target)``: that canonical materializer discovers the
    descriptor's graded roots with the public/dev label policy and applies ``graded_exclude``.  Hidden
    grading consumes the descriptor's hidden roots and asks ``capsule_grade.grade`` for capability
    admission, whose selector is ``capsule_runner._split_ineligible`` over non-model capsules.  Derive
    those same decisions here without materializing a cache, so preflight remains read-only.
    """
    contract = PB.REPO / "merlin/contract"
    public_source = discover_capsules(
        target.graded_roots(), labels={"public", "dev"}, contract=contract)
    public_names = [str(cap.get("name")) for cap in public_source]
    if len(public_names) != len(set(public_names)):
        raise ExperimentError("functional public roots contain duplicate capsule names")
    excluded = set(getattr(target, "graded_exclude", ()) or ())
    unknown = sorted(excluded - set(public_names))
    if unknown:
        raise ExperimentError(f"functional public exclusions name absent capsules: {unknown}")
    public_selected = [cap for cap in public_source if str(cap.get("name")) not in excluded]

    hidden_source = discover_capsules(
        target.hidden_roots(), labels={"hidden"}, contract=contract)
    hidden_names = [str(cap.get("name")) for cap in hidden_source]
    if len(hidden_names) != len(set(hidden_names)):
        raise ExperimentError("functional hidden roots contain duplicate capsule names")
    hidden_ops = [cap for cap in hidden_source if cap.get("kind") != "model"]
    _eligible, hidden_ineligible = CAPSULES._split_ineligible(hidden_ops, target.target)
    hidden_excluded = {str(row.get("capsule")) for row in hidden_ineligible}
    hidden_selected = [cap for cap in hidden_source
                       if str(cap.get("name")) not in hidden_excluded]

    expected = (
        ("public source", len(public_source),
         getattr(target, "graded_expected_source_capsules", None)),
        ("public admitted", len(public_selected),
         getattr(target, "graded_expected_admitted_capsules", None)),
        ("hidden source", len(hidden_source),
         getattr(target, "hidden_expected_source_capsules", None)),
        ("hidden admitted", len(hidden_selected),
         getattr(target, "hidden_expected_admitted_capsules", None)),
    )
    drift = [f"{label}={observed}, expected={declared}"
             for label, observed, declared in expected
             if declared is not None and observed != declared]
    if drift:
        raise ExperimentError("functional grade cohort drifted from descriptor: " + "; ".join(drift))
    if not public_selected or not hidden_selected:
        raise ExperimentError("full functional grade needs nonempty admitted public and hidden cohorts")
    return FunctionalGradeCohort(
        public=tuple(_functional_capsule(cap) for cap in public_selected),
        hidden=tuple(_functional_capsule(cap) for cap in hidden_selected),
        public_source_count=len(public_source), hidden_source_count=len(hidden_source))


def _functional_gsim_cases(cohort: FunctionalGradeCohort) -> tuple[FunctionalCapsule, ...]:
    """Exact admitted descriptors eligible for strict one-ELF GSIM/Verilator capture.

    An operator capsule lowers to one ELF, so the pre-authoring certificate can run that identical ELF
    on both engines. A model capsule is an ordered host/accelerator program whose formal evidence is a
    dynamic ledger of many tile ELFs; pretending it has one descriptor-level ELF would certify an
    artifact that does not exist. Models therefore remain in ``FunctionalGradeCohort`` and the complete
    post-candidate functional regrade, where the pinned GSIM runs every emitted tile and the model
    execution checker validates their ledger. They are excluded only from this *prebuilt one-ELF*
    certificate envelope.
    """
    return tuple(capsule for capsule in (*cohort.public, *cohort.hidden)
                 if capsule.kind != "model")


def _verify_functional_certificate(certificate: GATE.CertificateRecord,
                                   cohort: FunctionalGradeCohort) -> dict[str, Any]:
    """Bind a strict same-ELF GSIM certificate to every admitted non-model descriptor."""
    descriptors = _functional_gsim_cases(cohort)
    identities: dict[str, list[str]] = {}
    for capsule in descriptors:
        identities.setdefault(capsule.workload_sha256, []).append(str(capsule.manifest))
    missing = sorted(set(identities) - set(certificate.members))
    extras = sorted(set(certificate.members) - set(identities))
    if missing or extras:
        detail = []
        if missing:
            detail.append("missing=" + ", ".join(
                f"{identities[identity]}={identity}" for identity in missing))
        if extras:
            detail.append("extras=" + ", ".join(extras))
        raise ExperimentError(
            "functional GSIM certificate is not the exact admitted public+hidden cohort ("
            + "; ".join(detail) + ")")
    return {
        "public_source_descriptors": cohort.public_source_count,
        "public_descriptors": len(cohort.public),
        "hidden_source_descriptors": cohort.hidden_source_count,
        "hidden_descriptors": len(cohort.hidden),
        "same_elf_certificate_descriptors": len(descriptors),
        "dynamic_model_regrade_descriptors": (
            len(cohort.public) + len(cohort.hidden) - len(descriptors)),
        "same_elf_certificate_scope": "admitted_non_model_descriptors",
        "model_certificate_scope": "full_regrade_dynamic_tile_gsim_execution_ledger",
        "distinct_workload_sha256": sorted(identities),
    }


def _functional_regrade_inputs(target: object,
                               cohort: FunctionalGradeCohort) -> tuple[str, str]:
    """Build the canonical public grade view and bind both CLI inputs to ``cohort``.

    The public CLI input is already admission-filtered and carries the materializer's source/admitted
    record.  The hidden CLI input deliberately remains the full source root: ``grade_agent_run`` applies
    the same post-freeze capability selector used by :func:`_functional_grade_cohort`, preserving the
    honest hidden source/admitted counts without exposing its excluded names.
    """
    current = _functional_grade_cohort(target)
    if current != cohort:
        raise ExperimentError("functional grade cohort changed after preflight")
    public_root = public_capsules_for(target, tier_ceiling="L3")
    materialized_public = discover_capsules(
        public_root, labels={"public", "dev"}, contract=PB.REPO / "merlin/contract")
    if {str(cap.get("name")) for cap in materialized_public} != {
            capsule.name for capsule in cohort.public}:
        raise ExperimentError("materialized public regrade cohort differs from certificate cohort")
    hidden_roots = ",".join(str(path) for path in target.hidden_roots())
    if not hidden_roots:
        raise ExperimentError("full functional regrade needs nonempty hidden source roots")
    return str(public_root), hidden_roots


def _verify_tuning_certificate(certificate: GATE.CertificateRecord,
                               target: object,
                               capsules: str = "all",
                               families: str = "all") -> dict[str, Any]:
    """Require the initial certificate to be exactly the descriptor-derived tuning corpus.

    The corpus is narrowed by the SAME selection the trials will measure. Several families measure the
    same workload under a different lever, so the full phase contains repeated workload identities
    by design; the campaign measures one member per identity, and validating the unselected corpus
    here would refuse a launch over members it was never going to run.
    """
    corpus = PAS.discover_performance_corpus(
        target,
        capsules=None if capsules in (None, "", "all") else capsules,
        families=None if families in (None, "", "all") else families)
    identities: dict[str, str] = {}
    for member in corpus.capsules:
        identity = GATE.workload_sha256(
            PAIRED.CERTPROD.derive_workload(member.source_dir / "capsule.yaml"))
        if identity in identities:
            raise ExperimentError(
                f"tuning corpus repeats exact workload identity in {identities[identity]} and "
                f"{member.family}/{member.capsule}")
        identities[identity] = f"{member.family}/{member.capsule}"
    missing = sorted(set(identities) - set(certificate.members))
    extras = sorted(set(certificate.members) - set(identities))
    if missing or extras:
        raise ExperimentError(
            f"tuning GSIM certificate is not the exact derived corpus "
            f"(missing={missing}, extras={extras})")
    return {"workload_sha256": sorted(identities), "members": len(identities)}


def _require_same_gsim_build(reference: GATE.CertificateRecord,
                             certificate: GATE.CertificateRecord, *, label: str) -> None:
    """Keep every qualification phase on one exact RTL/simulator build."""
    changed = sorted(name for name in GATE.REQUIRED_PINS
                     if certificate.pins[name]["sha256"] != reference.pins[name]["sha256"])
    if changed:
        raise ExperimentError(f"{label} changed pinned build artifacts: {changed}")


def _content_addressed_document(root: Path, stem: str) -> tuple[Path, str, dict[str, Any]]:
    paths = sorted(root.glob(f"{stem}.*.json"))
    if len(paths) != 1:
        raise ExperimentError(f"functional qualification has no unique {stem} receipt")
    path = paths[0]
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(f"functional qualification {stem} receipt is absent or linked")
    digest = _sha_file(path)
    if path.name != f"{stem}.{digest}.json":
        raise ExperimentError(
            f"functional qualification {stem} receipt is not content-addressed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"functional qualification {stem} receipt is unreadable") from exc
    if not isinstance(document, dict):
        raise ExperimentError(f"functional qualification {stem} receipt is not a mapping")
    return path.resolve(), digest, document


def _verify_functional_certificate_provenance(
        certificate: GATE.CertificateRecord, tuning: GATE.CertificateRecord,
        functional_submission_sha256: str) -> dict[str, Any]:
    """Bind the qualification envelope to the exact sealed compiler and source certificate."""
    certificate_path = Path(certificate.path)
    if certificate_path.is_symlink() or not certificate_path.is_file():
        raise ExperimentError("functional GSIM certificate is absent or linked")
    root = certificate_path.resolve().parent
    if certificate_path.name != f"functional-certificate.{certificate.sha256}.json":
        raise ExperimentError("functional GSIM certificate is not producer-content-addressed")
    declaration_path, declaration_sha, declaration = _content_addressed_document(
        root, "declaration")
    completion_path, completion_sha, completion = _content_addressed_document(root, "completion")

    source = declaration.get("source_certificate") or {}
    baseline = declaration.get("functional_baseline") or {}
    target_descriptor = declaration.get("target_descriptor") or {}
    cases = declaration.get("cases")
    declared_pins = source.get("pins") if isinstance(source, Mapping) else None
    source_path = Path(str(source.get("path") or "")) if isinstance(source, Mapping) else Path()
    try:
        source_path = source_path.resolve(strict=True)
        tuning_path = Path(tuning.path).resolve(strict=True)
        descriptor_path = Path(str(target_descriptor.get("path") or "")).resolve(strict=True)
    except OSError as exc:
        raise ExperimentError(
            "functional qualification source certificate or target descriptor is unavailable") from exc
    expected_pins = {name: tuning.pins[name]["sha256"] for name in sorted(GATE.REQUIRED_PINS)}
    declared_workloads = ({str(row.get("workload_sha256")) for row in cases}
                          if isinstance(cases, list)
                          and all(isinstance(row, Mapping) for row in cases) else set())
    if (declaration.get("schema") != "merlin.functional-gsim-qualification.v1"
            or declaration.get("policy")
            != "formal-public-plus-hidden-admission-distinct-workloads.v1"
            or declaration.get("target") != certificate.target
            or baseline.get("sha256") != functional_submission_sha256
            or target_descriptor.get("sha256") != _sha_file(descriptor_path)
            or source.get("sha256") != tuning.sha256
            or source_path != tuning_path
            or declared_pins != expected_pins
            or declared_workloads != set(certificate.members)):
        raise ExperimentError(
            "functional qualification declaration is not bound to the exact sealed functional "
            "submission and tuning certificate")

    completed_source = completion.get("source_certificate") or {}
    completed_certificate = completion.get("functional_certificate") or {}
    selected_path = Path(str(completed_certificate.get("path") or ""))
    completed_source_path = Path(str(completed_source.get("path") or ""))
    try:
        selected_path = selected_path.resolve(strict=True)
        completed_source_path = completed_source_path.resolve(strict=True)
    except OSError as exc:
        raise ExperimentError("functional qualification completion selects an absent certificate") from exc
    if (completion.get("schema") != "merlin.functional-gsim-qualification.v1"
            or completion.get("status") != "complete"
            or completion.get("declaration_sha256") != declaration_sha
            or completed_source.get("sha256") != tuning.sha256
            or completed_source_path != tuning_path
            or selected_path != certificate_path.resolve()
            or completed_certificate.get("sha256") != certificate.sha256
            or set(completed_certificate.get("workload_sha256") or ())
            != set(certificate.members)):
        raise ExperimentError(
            "functional qualification completion does not select this exact certificate")
    return {
        "root": str(root),
        "declaration": str(declaration_path), "declaration_sha256": declaration_sha,
        "completion": str(completion_path), "completion_sha256": completion_sha,
        "functional_submission_sha256": functional_submission_sha256,
        "source_certificate_sha256": tuning.sha256,
        "target_descriptor": {"path": str(descriptor_path),
                              "sha256": target_descriptor.get("sha256")},
    }


def _resolve_chia_python(explicit: Path | None) -> Path | None:
    candidates = [explicit, Path(os.environ["MERLIN_CHIA_PYTHON"])
                  if os.environ.get("MERLIN_CHIA_PYTHON") else None,
                  build_dir() / "chia-venv/bin/python",
                  PB.REPO / "build/chia-venv/bin/python"]
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate).absolute()
    return None


def _chia_canary(python: Path | None) -> dict[str, Any]:
    """Read-only external-env import canary; never initializes Ray or launches work."""
    resolved = _resolve_chia_python(python)
    if resolved is None or not resolved.is_file():
        raise ExperimentError("a CHIA venv Python is required for the orchestration capability canary")
    bridge = (PB.REPO / "merlin/python/merlin/benchharness/chia_bridge.py").resolve()
    wrapper = (HERE / "chia_agentic_perf_experiment.py").resolve()
    program = (
        "import json,chia.trace,ray;"
        "from merlin.benchharness.chia_bridge import chia_available;"
        "assert chia_available();"
        "print(json.dumps({'chia_trace':chia.trace.__file__,'ray':ray.__version__}))")
    environment = dict(os.environ)
    python_path = str(PB.REPO / "merlin/python")
    environment["PYTHONPATH"] = python_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    completed = subprocess.run([str(resolved), "-c", program], cwd=PB.REPO,
                               env=environment, capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise ExperimentError(f"CHIA venv import canary failed: {completed.stderr[-500:]}")
    try:
        probe = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise ExperimentError("CHIA venv import canary returned malformed evidence") from exc
    trace_path = Path(str(probe.get("chia_trace") or "")).resolve()
    if not trace_path.is_file():
        raise ExperimentError("CHIA import canary did not identify its source bytes")
    return {
        "available": True,
        "python": str(resolved), "python_resolved": str(resolved.resolve()),
        "python_sha256": _sha_file(resolved),
        "chia_trace": str(trace_path), "chia_trace_sha256": _sha_file(trace_path),
        "ray_version": str(probe.get("ray") or ""),
        "bridge": str(bridge), "bridge_sha256": _sha_file(bridge),
        "required_entrypoint": str(wrapper), "required_entrypoint_sha256": _sha_file(wrapper),
        "launch_envelope_required": True,
        "campaign_scheduler": "chia_single_task_envelope_over_resume_safe_content_addressed_host_chain",
        "chia_role": "logical_resource_assignment_and_profiler_envelope_without_protocol_reordering",
        "driver_parity_claim": False,
        "reason": "all three predeclared trials use one Codex driver; Claude-vs-Codex is not an arm",
    }


def preflight(config: Config, *, heldout_certificate_provider_available: bool = False) -> dict[str, Any]:
    """Read-only validation and exact top-level declaration. Launches nothing and writes nothing."""
    _safe(config.experiment_id, label="experiment id")
    if len(TRIALS) != 3 or len(set(TRIALS)) != 3:
        raise ExperimentError("experiment requires exactly three independent trial identities")
    if tuple(REPLICATES) != ("r000", "r001", "r002"):
        raise ExperimentError("measurement identities must be exactly r000-r002")
    integers = (config.wall_budget_seconds, config.rounds, config.round_timeout_seconds,
                config.max_tool_calls, config.tool_timeout_seconds, config.smoke_replicates,
                config.holdout_count, config.measurement_timeout)
    integers = (*integers, config.heldout_qualification_timeout, config.generalization_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integers):
        raise ExperimentError("all budgets/counts must be positive integers")
    if not config.model.strip() or not config.effort.strip():
        raise ExperimentError("model and effort must be explicit")
    if (config.gsim_max_cycles is not None
            and (isinstance(config.gsim_max_cycles, bool)
                 or not isinstance(config.gsim_max_cycles, int)
                 or config.gsim_max_cycles <= 0)):
        raise ExperimentError("predeclared GSIM max cycles must be a positive integer")
    target = load_target_experiment(config.descriptor)
    functional_cohort = _functional_grade_cohort(target)
    functional = PAS.inspect_stage_functional_run(
        runs_root(target.target, "capsule-bench"), config.functional_run_id,
        config.functional_submission_sha256,
        waive=frozenset(config.waive_functional_gate or ()))
    certificate = GATE.load_certificate(
        config.gsim_certificate, expected_sha256=config.gsim_certificate_sha256)
    if certificate.target != target.target:
        raise ExperimentError("GSIM certificate target differs from the experiment target")
    tuning_coverage = _verify_tuning_certificate(
        certificate, target, config.perf_capsules, config.perf_families)
    pinned_gsim = Path(certificate.pins["gsim_binary"]["path"])
    if not pinned_gsim.is_file() or _sha_file(pinned_gsim) != certificate.pins[
            "gsim_binary"]["sha256"]:
        raise ExperimentError("the tuning certificate's pinned GSIM binary is unavailable")
    blockers = []
    price_path = config.telemetry_price_table
    if price_path is None:
        try:
            from merlin.common.paths import _dotenv  # noqa: PLC0415
            raw_price = (os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or "")
        except Exception:  # noqa: BLE001
            raw_price = os.environ.get("AET_PRICE_TABLE", "")
        price_path = Path(raw_price) if raw_price else None
    snapshot_candidates = [config.rtl_facts, config.perf_profile]
    if price_path is not None:
        snapshot_candidates.append(price_path)
    snapshot_needed = [str(path) for path in snapshot_candidates
                       if Path(path).is_file() and Path(path).stat().st_mode & 0o222]
    if snapshot_needed:
        blockers.append(
            "RTL facts/performance profile/telemetry price table include writable inputs and must "
            "first be copied into "
            "the experiment's content-addressed 0444 preflight_inputs evidence: "
            + ", ".join(snapshot_needed))
        domain = {"status": "requires_immutable_snapshot"}
        rtl_provenance = None
    else:
        rtl_provenance = HOLDOUT.verify_rtl_facts_provenance(
            config.rtl_facts, target=target.target)
        domain = HOLDOUT.derive_domain(config.rtl_facts, config.perf_profile, target=target.target)
    telemetry_preflight = None
    telemetry_treatment = None
    try:
        telemetry_preflight = PAS.telemetry_preflight(
            model=config.model, price_table=price_path, codex_binary=config.codex_binary)
        telemetry_treatment = _telemetry_treatment_identity(telemetry_preflight)
    except (PAS.StageGateError, ExperimentError) as exc:
        blockers.append(f"agent telemetry preflight failed: {exc}")
    chia_preflight = None
    try:
        chia_preflight = _chia_canary(config.chia_python)
    except ExperimentError as exc:
        blockers.append(f"CHIA orchestration capability preflight failed: {exc}")
    functional_certificate = None
    functional_coverage = None
    functional_provenance = None
    if (config.functional_gsim_certificate is None
            or config.functional_gsim_certificate_sha256 is None):
        blockers.append(
            "full public+hidden functional-suite GSIM certificate is required before agent launch")
    else:
        functional_certificate = GATE.load_certificate(
            config.functional_gsim_certificate,
            expected_sha256=config.functional_gsim_certificate_sha256)
        if functional_certificate.target != target.target:
            raise ExperimentError("functional GSIM certificate target differs from experiment target")
        _require_same_gsim_build(
            certificate, functional_certificate, label="functional GSIM certificate")
        functional_provenance = _verify_functional_certificate_provenance(
            functional_certificate, certificate, functional.digest)
        functional_coverage = _verify_functional_certificate(
            functional_certificate, functional_cohort)
    telemetry_sha = (_sha_bytes(_canonical(telemetry_preflight))
                     if telemetry_preflight is not None else None)
    trial_contract = {trial: {"model": config.model,
                              "resolved_model": ((telemetry_treatment or {}).get(
                                  "resolved_model")),
                              "effort": config.effort,
                              "wall_budget_seconds": config.wall_budget_seconds,
                              "rounds": config.rounds,
                              "round_timeout_seconds": config.round_timeout_seconds,
                              "max_tool_calls": config.max_tool_calls,
                              "tool_timeout_seconds": config.tool_timeout_seconds,
                              "smoke_replicates": config.smoke_replicates,
                              "measurement_replicates": len(REPLICATES),
                              "functional_run_id": functional.run_id,
                              "functional_submission_sha256": functional.digest,
                              "telemetry_required": True,
                              "telemetry_preflight_sha256": telemetry_sha,
                              "treatment_identity": telemetry_treatment}
                      for trial in TRIALS}
    if not heldout_certificate_provider_available:
        blockers.append(
            "post-seal held-out GSIM extension certificate provider is unavailable: "
            "produce_gsim_certificate.py can validate supplied captures but has no CLI/API that "
            "discovers the newly revealed capsules, builds their exact ELFs, captures same-ELF "
            "GSIM/Verilator output evidence, and emits an extension certificate; the tuning "
            "certificate cannot qualify them")
    declaration = {"schema": SCHEMA,
                   "status": "GO" if not blockers else "NO_GO", "blockers": blockers,
                   "experiment_id": config.experiment_id,
                   "target": target.target, "trial_contracts": trial_contract,
                   "trials": list(TRIALS), "replicates": list(REPLICATES),
                   "holdout_commit_before_authoring": True,
                   "holdout_domain_sha256": _sha_bytes(_canonical(domain)),
                   "rtl_facts_provenance": rtl_provenance,
                   "gsim_certificate_sha256": certificate.sha256,
                   "tuning_gsim_coverage": tuning_coverage,
                   "functional_gsim_certificate_sha256": (
                       functional_certificate.sha256 if functional_certificate else None),
                   "functional_gsim_coverage": functional_coverage,
                   "functional_gsim_provenance": functional_provenance,
                   "agent_telemetry": telemetry_preflight,
                   "agent_treatment": telemetry_treatment,
                   "orchestration": chia_preflight,
                   "gsim_runtime": {"binary": str(pinned_gsim.resolve()),
                                    "binary_sha256": certificate.pins["gsim_binary"]["sha256"],
                                    "max_cycles": config.gsim_max_cycles,
                                    "ambient_selection_forbidden": True},
                   "heldout_certificate_phase": "required_after_all_candidates_seal_and_reveal",
                   "functional_regrade": "full_public_plus_hidden_L3_before_performance",
                   "measurement_phases": ["tuning", "held_out"],
                   "measurement_engine_policy": {
                       "semantic_screen": "spike_no_timing",
                       "rtl_execution_backends": ["gsim"],
                       "timing_authority": "gsim",
                       "verilator": "prelaunch_certificate_qualification_only",
                   },
                   "selection": "all_trials_all_cells_no_best_of_no_drop"}
    return {**declaration, "declaration_sha256": _sha_bytes(_canonical(declaration))}


def _run_checked(runner: CommandRunner, argv: Sequence[str], *, environment: Mapping[str, str]) -> None:
    result = runner(argv, PB.REPO, environment)
    if result.returncode:
        raise ExperimentError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr[-1000:]}")


FANOUT_ENVIRONMENT_VARIABLE = "MERLIN_PERF_CAMPAIGN_FANOUT"


def declared_fanout(environment: Mapping[str, str] | None = None) -> int:
    """How many independent children one phase may have in flight at once.

    The width is DECLARED by the launch, never inferred: unset (or blank) means one, which is the
    fully serial campaign, so a launch that says nothing behaves exactly as before.  A value that
    cannot be read as a positive integer is refused rather than rounded to a default -- a 40-hour
    campaign silently run at the wrong width is not recoverable after the fact.
    """
    source = os.environ if environment is None else environment
    raw = source.get(FANOUT_ENVIRONMENT_VARIABLE)
    if raw is None or not raw.strip():
        return 1
    text = raw.strip()
    if any(character not in "0123456789" for character in text) or int(text) < 1:
        raise ExperimentError(
            f"{FANOUT_ENVIRONMENT_VARIABLE} must be a positive integer, got {raw!r}")
    return int(text)


@dataclass(frozen=True)
class ChildStage:
    """One independent child: a blocking launch, then the commit that writes the record."""

    name: str
    # None when the child's final artifact already exists and is merely adopted.
    launch: Callable[[], None] | None
    commit: Callable[[], Any]


def run_child_stages(stages: Sequence[ChildStage], *, workers: int) -> list[Any]:
    """Launch independent children with bounded concurrency, then commit in DECLARED order.

    Only `launch` may leave this thread.  Every `commit` runs on the calling thread, in `stages`
    order, because the checkpoint chain is a strict linear hash chain (it asserts its own index and
    links the previous digest): concurrent appends would corrupt it, and a completion-ordered record
    would not reproduce.  So the record is identical whatever order the children finish in.

    With `workers` at 1 the launches stay on this thread and the first failure aborts the phase,
    exactly as the serial campaign does.  Above 1 every launch is joined before anything is raised,
    and the failure names every child that failed rather than only the first.  Nothing is committed
    when any launch failed, so a resumed campaign re-derives the same chain in the same order --
    children that did finish are adopted from their completed artifacts instead of being rerun.
    """
    if workers < 1:
        raise ExperimentError("child stage concurrency must be at least one")
    pending = [stage for stage in stages if stage.launch is not None]
    if workers == 1 or len(pending) <= 1:
        for stage in pending:
            assert stage.launch is not None
            stage.launch()
    else:
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(pending)),
                                thread_name_prefix="perf-campaign") as pool:
            launched = [(stage.name, pool.submit(stage.launch)) for stage in pending]
            for name, future in launched:
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - report every failure, not just the first
                    failures.append(f"{name}: {error}")
        if failures:
            raise ExperimentError(
                f"{len(failures)} of {len(pending)} concurrent child stages failed; "
                + "; ".join(failures))
    return [stage.commit() for stage in stages]


def _uncheckpointed_state(root: Path, final_artifact: Path, *, label: str) -> str:
    """Classify a child attempt without mutating it, so resume never reruns in place."""
    if final_artifact.is_file():
        return "complete"
    if root.exists() or root.is_symlink():
        raise ExperimentError(f"uncheckpointed {label} is partial; refusing in-place rerun: {root}")
    return "absent"


def _verify_saved_file(saved: Mapping[str, Any], expected: Path, *, label: str) -> Path:
    path = Path(str(saved.get("path") or ""))
    if (not path.is_file() or path.resolve() != expected.resolve()
            or _sha_file(path) != saved.get("sha256")):
        raise ExperimentError(f"{label} evidence changed across resume")
    return path


def _handoff(path: Path, target: object) -> PAS.VerifiedCandidateHandoff:
    return PAS.verify_candidate_handoff(path, verify_authoring_tools=False,
                                        target_experiment=target)


def _verify_regrade(run_dir: Path, handoff: PAS.VerifiedCandidateHandoff) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.is_file():
        raise ExperimentError(f"functional regrade manifest is absent: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if (manifest.get("submission_sha256") != handoff.candidate_sha256
            or (manifest.get("completion") or {}).get("formal_grade_complete") is not True
            or (manifest.get("public_dev") or {}).get("formal_complete") is not True
            or (manifest.get("hidden") or {}).get("formal_complete") is not True):
        raise ExperimentError("candidate did not pass the full public+hidden functional L3 regrade")
    if str(hash_tree(handoff.candidate_path)["sha256"]) != handoff.candidate_sha256:
        raise ExperimentError("sealed candidate changed during functional regrade")
    return {"path": str(manifest_path.resolve()), "sha256": _sha_file(manifest_path),
            "submission_sha256": handoff.candidate_sha256}


def _prepare_regrade(run_dir: Path, handoff: PAS.VerifiedCandidateHandoff) -> None:
    if run_dir.exists():
        return
    run_dir.mkdir(parents=True)
    shutil.copytree(handoff.candidate_path, run_dir / "submission",
                    ignore=shutil.ignore_patterns("build", "__pycache__"))
    copied = run_dir / "submission"
    for path in (copied, *copied.rglob("*")):
        if path.is_symlink():
            raise ExperimentError("functional regrade copy contains a symlink")
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
    if str(hash_tree(copied)["sha256"]) != handoff.candidate_sha256:
        raise ExperimentError("functional regrade copy differs from the sealed candidate")


def _paired_rows(manifest_path: Path, trial: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "GO":
        raise ExperimentError(f"paired measurement did not reach GO: {manifest_path}")
    results = manifest.get("raw_results") or {}
    path = Path(str(results.get("paired_cells") or ""))
    if not path.is_file() or _sha_file(path) != results.get("paired_cells_sha256"):
        raise ExperimentError("paired result cells are absent or changed")
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells") or []
    rows = []
    for cell in cells:
        if cell.get("simulator") != "gsim":
            continue
        provenance = cell.get("provenance") or {}
        rows.append({"identity": {"trial": trial, "subject": cell.get("arm"),
                                  "family": f"{cell.get('phase')}:{cell.get('family')}",
                                  "capsule": cell.get("capsule"), "simulator": "gsim",
                                  "replicate": cell.get("replicate")},
                     "tier": provenance.get("tier"), "correct": cell.get("correct"),
                     "cycle_accurate": provenance.get("cycle_accurate"),
                     "cycles": cell.get("cycles"),
                     "oracle": {"kind": provenance.get("oracle_kind"),
                                "derived_from_rtl": provenance.get("derived_from_rtl")}})
    return rows


def _verify_measurement_manifest(
        manifest_path: Path, *, phase: str, functional_run_id: str,
        functional_submission_sha256: str, handoff: PAS.VerifiedCandidateHandoff,
        corpus_manifest_sha256: str, corpus_capsules_sha256: str,
        certificate_sha256: str) -> dict[str, Any]:
    """Validate a completed child artifact before checkpointing or adopting it after a crash."""
    _paired_rows(manifest_path, "verification")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    certificate = manifest.get("gsim_certificate") or {}
    corpus = manifest.get("frozen_corpus") or {}
    completion = manifest.get("completion") or {}
    engine_policy = manifest.get("engine_policy") or {}
    results_path = Path(str((manifest.get("raw_results") or {}).get("paired_cells") or ""))
    raw_cells = json.loads(results_path.read_text(encoding="utf-8")).get("cells") or []
    plan = manifest.get("measurement_plan") or {}
    if (manifest.get("measurement_plan_sha256")
            != PAIRED._sha256_bytes(PAIRED._canonical_bytes(plan))):
        raise ExperimentError("paired measurement plan digest is invalid")
    try:
        expected_results = tuple(PAIRED.ResultIdentity(**row)
                                 for row in plan.get("expected_results") or [])
        recomputed_completion = PAIRED.completion_report(raw_cells, expected_results)
    except Exception as exc:
        raise ExperimentError(f"paired raw evidence cannot be revalidated: {exc}") from exc
    if recomputed_completion != completion:
        raise ExperimentError("paired completion does not match raw evidence")
    expected = {
        "phase": phase,
        "functional_run_id": functional_run_id,
        "functional_submission_sha256": functional_submission_sha256,
        "candidate_record_sha256": handoff.record_sha256,
        "candidate_sha256": handoff.candidate_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ExperimentError("paired measurement manifest identity differs from its trial declaration")
    if (certificate.get("sha256") != certificate_sha256
            or corpus.get("manifest_sha256") != corpus_manifest_sha256
            or corpus.get("capsules_sha256") != corpus_capsules_sha256
            or corpus.get("visibility") != phase):
        raise ExperimentError("paired measurement certificate/corpus identity differs from declaration")
    expected_cells = completion.get("expected")
    if (isinstance(expected_cells, bool) or not isinstance(expected_cells, int)
            or expected_cells <= 0 or completion.get("reported") != expected_cells
            or completion.get("passed") != expected_cells
            or completion.get("failed") != 0 or completion.get("missing") != 0
            or completion.get("complete") is not True
            or engine_policy.get("rtl_execution_backends") != ["gsim"]
            or engine_policy.get("timing_authority") != "gsim"
            or engine_policy.get("verilator") != "prelaunch_certificate_qualification_only"
            or manifest.get("identity_before") != manifest.get("identity_after")
            or (manifest.get("fork_before") or {}).get("ok") is not True
            or (manifest.get("fork_after") or {}).get("ok") is not True):
        raise ExperimentError("paired measurement lacks complete GSIM-only identity/fork evidence")
    return {"path": str(manifest_path.resolve()), "sha256": _sha_file(manifest_path)}


def _statistics_trials(trial_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Project sealed agent evidence onto the deliberately narrow predeclaration API."""
    return [{"trial": str(row["trial"]), "agent_run_id": str(row["agent_run_id"])}
            for row in trial_evidence]


def _seal_final(root: Path, document: Mapping[str, Any]) -> Path:
    payload, digest = _canonical(document), _sha_bytes(_canonical(document))
    path = root / f"experiment_manifest.{digest}.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise ExperimentError("final manifest digest collision")
        return path
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def _verify_extension_certificate(
        tuning: GATE.CertificateRecord, extension: GATE.CertificateRecord,
        holdout_manifest: Path, *, manifest_sha256: str | None = None,
        corpus_sha256: str | None = None, target: str | None = None) -> dict[str, Any]:
    """Require a true same-build envelope extension covering every exact revealed descriptor."""
    changed_pins = sorted(name for name in GATE.REQUIRED_PINS
                          if extension.pins[name]["sha256"] != tuning.pins[name]["sha256"])
    if changed_pins:
        raise ExperimentError(f"held-out extension changed pinned build artifacts: {changed_pins}")
    missing_tuning = sorted(set(tuning.members) - set(extension.members))
    if missing_tuning:
        raise ExperimentError("held-out certificate is not an extension of the tuning envelope")
    members = HQUAL.load_revealed_members(
        holdout_manifest, expected_manifest_sha256=manifest_sha256,
        expected_corpus_sha256=corpus_sha256, expected_target=target)
    identities = {member.workload_sha256: member.name for member in members}
    missing = sorted(identity for identity in identities if identity not in extension.members)
    if missing:
        raise ExperimentError(
            "held-out extension does not cover every revealed workload: "
            + ", ".join(f"{identities[identity]}={identity}" for identity in missing))
    extras = sorted(set(extension.members) - set(tuning.members) - set(identities))
    if extras:
        raise ExperimentError(
            "held-out extension contains workloads outside the exact tuning-plus-reveal envelope: "
            + ", ".join(extras))
    return {"holdout_workload_sha256": sorted(identities),
            "tuning_workloads_retained": len(tuning.members),
            "heldout_workloads_covered": len(identities),
            "pins_unchanged": True}


def _qualify_heldout_with_config(
        reveal_manifest: Path, qualification_root: Path, tuning: GATE.CertificateRecord, *,
        functional_base: Path, functional_base_sha256: str,
        reveal_manifest_sha256: str, reveal_corpus_sha256: str,
        config: Config, target: object) -> tuple[Path, str]:
    """Thread declared runtime bounds into the in-process post-reveal provider."""
    return HQUAL.qualify_revealed_holdout(
        reveal_manifest, qualification_root, tuning,
        functional_base=functional_base,
        functional_base_sha256=functional_base_sha256,
        reveal_manifest_sha256=reveal_manifest_sha256,
        reveal_corpus_sha256=reveal_corpus_sha256,
        timeout=config.heldout_qualification_timeout,
        gsim_max_cycles=config.gsim_max_cycles,
        target_experiment=target)


def _author_candidates(
        config: Config, state: Checkpoints, target: object, declaration: Mapping[str, Any], *,
        environment: Mapping[str, str], expected_treatment: Mapping[str, Any],
        command_runner: CommandRunner, workers: int
) -> tuple[dict[str, PAS.VerifiedCandidateHandoff], list[dict[str, Any]]]:
    """Seal one candidate per trial, optionally with several agent stages in flight at once.

    The trials share nothing mutable: the hidden holdout is committed before this phase, each stage
    writes its own run directory, the frozen tuning corpus is proven byte-identical across trials
    afterwards, and independence is enforced by distinct evidence hashes.  The one order-sensitive
    act is the checkpoint append, which `run_child_stages` keeps on this thread in TRIALS order.
    """
    stage_root_base = runs_root(target.target, "perf-bench") / "agent_stages"
    stage_names = {trial: f"{config.experiment_id}__{trial}" for trial in TRIALS}
    records = {trial: stage_root_base / name / "performance_candidate.json"
               for trial, name in stage_names.items()}

    stages: list[ChildStage] = []
    for trial in TRIALS:
        if state.evidence(f"candidate:{trial}") is not None:
            continue
        stage_dir = stage_root_base / stage_names[trial]
        record = records[trial]
        launch: Callable[[], None] | None = None
        if _uncheckpointed_state(stage_dir, record, label=f"agent stage {trial}") == "absent":
            _verify_live_agent_treatment(config, expected_treatment)
            command = [sys.executable, str(HERE / "perf_agent_stage.py"),
                       "--functional-run-id", config.functional_run_id,
                       "--functional-submission-sha256", config.functional_submission_sha256,
                       "--run-id", stage_names[trial], "--model", config.model,
                       "--effort", config.effort,
                       "--wall-budget-seconds", str(config.wall_budget_seconds),
                       "--rounds", str(config.rounds),
                       "--round-timeout-seconds", str(config.round_timeout_seconds),
                       "--max-tool-calls", str(config.max_tool_calls),
                       "--tool-timeout-seconds", str(config.tool_timeout_seconds),
                       "--smoke-replicates", str(config.smoke_replicates),
                       "--replicates", "3", "--codex-binary", config.codex_binary,
                       "--gsim-certificate", str(config.gsim_certificate),
                       "--gsim-certificate-sha256", config.gsim_certificate_sha256,
                       "--rtl-facts", str(config.rtl_facts),
                       "--telemetry-price-table", str(config.telemetry_price_table),
                       "--descriptor", str(config.descriptor)]
            # A MEMBER THE ORACLE CANNOT EXECUTE MUST NOT HOLD THE WHOLE CORPUS HOSTAGE.
            # The stage refuses any member outside the certificate, and a member the reference
            # simulator cannot run cannot be certified -- so without a way to name it, one
            # unrunnable capsule blocks every other one. The selection is recorded in the run
            # manifest, so what was measured is never in doubt.
            if config.perf_capsules and config.perf_capsules != "all":
                command += ["--capsules", config.perf_capsules]
            if config.perf_families and config.perf_families != "all":
                command += ["--families", config.perf_families]
            launch = partial(_run_checked, command_runner, command, environment=environment)

        def commit(trial: str = trial, record: Path = record) -> dict[str, Any]:
            handoff = _handoff(record, target)
            _verify_trial_contract(trial, handoff, declaration["trial_contracts"][trial])
            saved = {"record": str(record.resolve()), "record_sha256": handoff.record_sha256,
                     "candidate_sha256": handoff.candidate_sha256}
            state.append(f"candidate:{trial}", saved)
            return saved

        stages.append(ChildStage(f"candidate:{trial}", launch, commit))
    run_child_stages(stages, workers=workers)

    handoffs: dict[str, PAS.VerifiedCandidateHandoff] = {}
    trial_evidence: list[dict[str, Any]] = []
    for trial in TRIALS:
        saved = state.evidence(f"candidate:{trial}")
        if saved is None:
            raise ExperimentError(f"agent stage evidence is missing after it ran: {trial}")
        handoff = _handoff(Path(saved["record"]), target)
        if handoff.record_sha256 != saved["record_sha256"]:
            raise ExperimentError(f"candidate record changed across resume: {trial}")
        _verify_trial_contract(trial, handoff, declaration["trial_contracts"][trial])
        if handoff.telemetry_evidence.get("preflight_sha256") != _sha_bytes(
                _canonical(declaration["agent_telemetry"])):
            raise ExperimentError(f"candidate telemetry stack differs from predeclaration: {trial}")
        handoffs[trial] = handoff
        trial_evidence.append({"trial": trial, "agent_run_id": stage_names[trial],
                               "agent_evidence_sha256": handoff.record_sha256})
    return handoffs, trial_evidence


@dataclass(frozen=True)
class _MeasurementCell:
    """One (trial, phase) paired-bench cell, addressed by its own fresh run directory."""

    trial: str
    phase: str
    handoff: PAS.VerifiedCandidateHandoff
    corpus_root: Path
    corpus_manifest: Path
    corpus_manifest_sha256: str
    corpus_capsules_sha256: str
    certificate: GATE.CertificateRecord
    stage: str
    run_id: str
    output: Path


def _measurement_cells(config: Config,
                       handoffs: Mapping[str, PAS.VerifiedCandidateHandoff],
                       revealed: Mapping[str, Any], *,
                       tuning_certificate: GATE.CertificateRecord,
                       heldout_certificate: GATE.CertificateRecord) -> list[_MeasurementCell]:
    """The complete paired matrix in fixed order: every trial x every phase, no cell dropped."""
    cells: list[_MeasurementCell] = []
    for trial in TRIALS:
        handoff = handoffs[trial]
        for phase, corpus_root, corpus_manifest, manifest_sha, capsules_sha, certificate in (
                ("tuning", handoff.corpus_root,
                 handoff.corpus_root / "performance_corpus_manifest.json",
                 handoff.corpus_manifest_sha256, handoff.corpus_sha256, tuning_certificate),
                ("held_out", Path(revealed["root"]), Path(revealed["manifest"]),
                 revealed["manifest_sha256"], revealed["capsules_sha256"], heldout_certificate)):
            run_id = f"{config.experiment_id}__{trial}__{phase}"
            cells.append(_MeasurementCell(
                trial=trial, phase=phase, handoff=handoff, corpus_root=corpus_root,
                corpus_manifest=corpus_manifest, corpus_manifest_sha256=manifest_sha,
                corpus_capsules_sha256=capsules_sha, certificate=certificate,
                stage=f"measurement:{trial}:{phase}", run_id=run_id,
                output=PB.RUNS / run_id / "campaign_manifest.json"))
    return cells


def _measure_cells(cells: Sequence[_MeasurementCell], config: Config, state: Checkpoints, *,
                   command_runner: CommandRunner, workers: int) -> list[tuple[str, Path]]:
    """Measure every cell, optionally several at once; each writes a disjoint fresh run directory.

    Adoption, verification and checkpointing all happen on this thread in `cells` order, so the
    recorded matrix is the same whichever cell finishes first.
    """
    def verify(cell: _MeasurementCell, path: Path) -> dict[str, Any]:
        return _verify_measurement_manifest(
            path, phase=cell.phase, functional_run_id=config.functional_run_id,
            functional_submission_sha256=config.functional_submission_sha256,
            handoff=cell.handoff, corpus_manifest_sha256=cell.corpus_manifest_sha256,
            corpus_capsules_sha256=cell.corpus_capsules_sha256,
            certificate_sha256=cell.certificate.sha256)

    stages: list[ChildStage] = []
    for cell in cells:
        if state.evidence(cell.stage) is not None:
            continue
        launch: Callable[[], None] | None = None
        if _uncheckpointed_state(
                cell.output.parent, cell.output,
                label=f"paired measurement {cell.trial}/{cell.phase}") == "absent":
            command = [sys.executable, str(HERE / "run_paired_perf_bench.py"),
                       "--functional-run-id", config.functional_run_id,
                       "--functional-submission-sha256", config.functional_submission_sha256,
                       "--candidate-record", str(cell.handoff.record_path),
                       "--corpus-root", str(cell.corpus_root),
                       "--corpus-manifest", str(cell.corpus_manifest),
                       "--corpus-manifest-sha256", cell.corpus_manifest_sha256,
                       "--corpus-capsules-sha256", cell.corpus_capsules_sha256,
                       "--phase", cell.phase,
                       "--gsim-certificate", str(cell.certificate.path),
                       "--gsim-certificate-sha256", cell.certificate.sha256,
                       "--rtl-facts", str(config.rtl_facts), "--run-id", cell.run_id,
                       "--timeout", str(config.measurement_timeout),
                       "--hardware-counters" if config.hardware_counters
                       else "--no-hardware-counters"]
            launch = partial(_run_checked, command_runner, command,
                             environment=child_environment(config, cell.certificate))

        def commit(cell: _MeasurementCell = cell) -> dict[str, Any]:
            saved = verify(cell, cell.output)
            state.append(cell.stage, saved)
            return saved

        stages.append(ChildStage(cell.stage, launch, commit))
    run_child_stages(stages, workers=workers)

    manifests: list[tuple[str, Path]] = []
    for cell in cells:
        saved = state.evidence(cell.stage)
        if saved is None:
            raise ExperimentError(
                f"paired measurement evidence is missing after it ran: {cell.stage}")
        saved_path = _verify_saved_file(saved, cell.output, label=f"measurement {cell.stage}")
        verify(cell, saved_path)
        manifests.append((cell.trial, Path(saved["path"])))
    return manifests


def run(config: Config, *, command_runner: CommandRunner = subprocess_runner,
        dry_run: bool = False,
        commit_holdout: Callable[..., HOLDOUT.HoldoutPaths] = HOLDOUT.commit_holdout,
        reveal_holdout: Callable[..., Path] = HOLDOUT.reveal_and_materialize,
        heldout_certificate_provider: Callable[[Path, Path, GATE.CertificateRecord],
                                               tuple[Path, str]] | None = None
        ) -> Path | dict[str, Any]:
    # Read the declared width before anything is launched, so an unreadable declaration fails now
    # rather than 20 hours in, at the phase that would have used it.
    fanout = declared_fanout()
    input_snapshots = None
    chia_launch = None if dry_run else _verify_chia_launch_receipt()
    if not dry_run:
        config, input_snapshots = snapshot_contract_inputs(config)
    declaration = preflight(config, heldout_certificate_provider_available=True)
    if dry_run:
        return declaration
    if declaration["status"] != "GO":
        raise ExperimentError("; ".join(declaration["blockers"]))
    assert chia_launch is not None
    orchestration = declaration.get("orchestration") or {}
    if (orchestration.get("required_entrypoint_sha256") != chia_launch["wrapper"]["sha256"]
            or orchestration.get("chia_trace_sha256") != chia_launch["chia_trace"]["sha256"]):
        raise ExperimentError("CHIA launch receipt differs from the preflighted orchestration stack")
    config_doc = _config_document(config)
    config_sha = _sha_bytes(_canonical(config_doc))
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    state = Checkpoints(root / "state", config_sha)
    predeclared = state.evidence("predeclared")
    if predeclared is None:
        state.append("predeclared", {"declaration": declaration, "config": config_doc,
                                     "input_snapshots": input_snapshots,
                                     "chia_launch_receipt": chia_launch})
    else:
        saved_launch = predeclared.get("chia_launch_receipt")
        if not isinstance(saved_launch, Mapping):
            raise ExperimentError("saved predeclaration lacks its CHIA launch identity")
        _verify_resume_chia_identity(saved_launch, chia_launch)
        _verify_resume_declaration(predeclared, declaration)
    target = load_target_experiment(config.descriptor)
    functional = PAS.inspect_stage_functional_run(
        runs_root(target.target, "capsule-bench"), config.functional_run_id,
        config.functional_submission_sha256,
        waive=frozenset(config.waive_functional_gate or ()))
    tuning_certificate = GATE.load_certificate(
        config.gsim_certificate, expected_sha256=config.gsim_certificate_sha256)
    assert config.functional_gsim_certificate is not None
    assert config.functional_gsim_certificate_sha256 is not None
    functional_certificate = GATE.load_certificate(
        config.functional_gsim_certificate,
        expected_sha256=config.functional_gsim_certificate_sha256)
    _require_same_gsim_build(
        tuning_certificate, functional_certificate, label="functional GSIM certificate")
    _verify_functional_certificate_provenance(
        functional_certificate, tuning_certificate, functional.digest)
    functional_cohort = _functional_grade_cohort(target)
    _verify_functional_certificate(functional_certificate, functional_cohort)
    environment = child_environment(config, tuning_certificate)
    expected_treatment = declaration.get("agent_treatment")
    if not isinstance(expected_treatment, Mapping):
        raise ExperimentError("predeclaration lacks an exact agent treatment identity")
    public_dir, private_dir = root / "agent_visible", root / "host_private"
    public_dir.mkdir(exist_ok=True)
    holdout = state.evidence("holdout_committed")
    if holdout is None:
        paths = commit_holdout(
            public_dir / "holdout_commitment.json", private_dir,
            rtl_facts_path=config.rtl_facts, perf_profile_path=config.perf_profile,
            target=target.target, candidate_ids=TRIALS, count=config.holdout_count,
            generalization_count=config.generalization_count,
            agent_view_root=public_dir)
        holdout = {"public": str(paths.public_commitment),
                   "public_sha256": _sha_file(paths.public_commitment),
                   "private": str(paths.host_private_dir)}
        state.append("holdout_committed", holdout)
    elif _sha_file(Path(holdout["public"])) != holdout["public_sha256"]:
        raise ExperimentError("holdout commitment changed across resume")

    handoffs, trial_evidence = _author_candidates(
        config, state, target, declaration, environment=environment,
        expected_treatment=expected_treatment, command_runner=command_runner, workers=fanout)
    _verify_trial_treatments(handoffs, expected_treatment)
    if len({row["agent_evidence_sha256"] for row in trial_evidence}) != 3:
        raise ExperimentError("three independent agent trials need distinct evidence hashes")
    if len({handoff.corpus_manifest_sha256 for handoff in handoffs.values()}) != 1 \
            or len({handoff.corpus_sha256 for handoff in handoffs.values()}) != 1:
        raise ExperimentError("independent trials did not receive one identical frozen tuning corpus")
    for trial, handoff in handoffs.items():
        workloads = PAIRED.CERTPROD.derive_frozen_corpus_workloads(
            handoff.corpus_root, manifest_sha256=handoff.corpus_manifest_sha256,
            capsules_sha256=handoff.corpus_sha256, expected_target=target.target)
        identities = {GATE.workload_sha256(workload) for workload in workloads.values()}
        if identities != set(tuning_certificate.members):
            raise ExperimentError(
                f"{trial} frozen tuning corpus differs from the exact tuning certificate envelope")

    # Passing target.graded_roots() directly would re-admit the policy-excluded descriptors. Build the
    # exact official public view and recheck the hidden selector against the certificate cohort after the
    # potentially long authoring phase.
    public_roots, hidden_roots = _functional_regrade_inputs(target, functional_cohort)
    regrades = {}
    for trial, handoff in handoffs.items():
        saved = state.evidence(f"functional_regrade:{trial}")
        grade_dir = root / "functional_regrades" / trial
        if saved is None:
            manifest_path = grade_dir / "run_manifest.yaml"
            if _uncheckpointed_state(
                    grade_dir, manifest_path, label=f"functional regrade {trial}") == "absent":
                _prepare_regrade(grade_dir, handoff)
                command = [
                    sys.executable,
                    str(PB.REPO / "merlin/experiments/capsule_bench/harness/grade_agent_run.py"),
                    "--run-dir", str(grade_dir), "--arm", "merlin_assisted",
                    "--model", config.model, "--capsules", public_roots,
                    "--hidden-capsules", hidden_roots]
                regrade_environment = child_environment(config, functional_certificate)
                _run_checked(command_runner, command, environment=regrade_environment)
            saved = _verify_regrade(grade_dir, handoff)
            state.append(f"functional_regrade:{trial}", saved)
        _verify_saved_file(saved, grade_dir / "run_manifest.yaml",
                           label=f"functional regrade {trial}")
        regrades[trial] = _verify_regrade(grade_dir, handoff)

    revealed = state.evidence("holdout_revealed")
    if revealed is None:
        manifest = reveal_holdout(
            Path(holdout["public"]), Path(holdout["private"]), root / "held_out_corpus",
            candidate_seals={trial: handoffs[trial].record_path for trial in TRIALS})
        document = json.loads(manifest.read_text(encoding="utf-8"))
        revealed = {"root": str(manifest.parent), "manifest": str(manifest),
                    "manifest_sha256": _sha_file(manifest),
                    "capsules_sha256": document["corpus"]["sha256"]}
        state.append("holdout_revealed", revealed)
    elif _sha_file(Path(revealed["manifest"])) != revealed["manifest_sha256"]:
        raise ExperimentError("held-out reveal changed across resume")

    extension = state.evidence("heldout_gsim_certificate")
    if extension is None:
        qualification_root = root / "heldout_gsim_qualification"
        if qualification_root.exists() or qualification_root.is_symlink():
            extension_path, extension_sha = HQUAL.load_completed_qualification(
                qualification_root, tuning=tuning_certificate,
                reveal_manifest_sha256=revealed["manifest_sha256"],
                reveal_corpus_sha256=revealed["capsules_sha256"],
                functional_base_sha256=functional.digest,
                gsim_max_cycles=config.gsim_max_cycles)
        elif heldout_certificate_provider is not None:
            extension_path, extension_sha = heldout_certificate_provider(
                Path(revealed["manifest"]), qualification_root, tuning_certificate)
        else:
            extension_path, extension_sha = _qualify_heldout_with_config(
                Path(revealed["manifest"]), qualification_root, tuning_certificate,
                functional_base=functional.submission_dir,
                functional_base_sha256=functional.digest,
                reveal_manifest_sha256=revealed["manifest_sha256"],
                reveal_corpus_sha256=revealed["capsules_sha256"],
                config=config, target=target)
        try:
            Path(extension_path).resolve(strict=True).relative_to(qualification_root.resolve())
        except ValueError as exc:
            raise ExperimentError("held-out certificate provider wrote outside its fresh host root") from exc
        heldout_certificate = GATE.load_certificate(
            extension_path, expected_sha256=extension_sha)
        if heldout_certificate.target != target.target:
            raise ExperimentError("held-out GSIM extension certificate names a different target")
        coverage = _verify_extension_certificate(
            tuning_certificate, heldout_certificate, Path(revealed["manifest"]),
            manifest_sha256=revealed["manifest_sha256"],
            corpus_sha256=revealed["capsules_sha256"], target=target.target)
        extension = {"path": str(Path(extension_path).resolve()), "sha256": extension_sha,
                     "certificate": heldout_certificate.to_dict(),
                     "coverage": coverage,
                     "produced_after_checkpoint": state.load()[-1]["sha256"]}
        state.append("heldout_gsim_certificate", extension)
    heldout_certificate = GATE.load_certificate(
        extension["path"], expected_sha256=extension["sha256"])
    coverage = _verify_extension_certificate(
        tuning_certificate, heldout_certificate, Path(revealed["manifest"]),
        manifest_sha256=revealed["manifest_sha256"],
        corpus_sha256=revealed["capsules_sha256"], target=target.target)
    if coverage != extension.get("coverage"):
        raise ExperimentError("held-out certificate coverage changed across resume")

    # Declare the statistics denominator before any performance subprocess is launched.
    stats_saved = state.evidence("statistics_predeclared")
    if stats_saved is None:
        capsules: set[tuple[str, str]] = set()
        for trial, handoff in handoffs.items():
            for phase, corpus_args, certificate in (
                    ("tuning", (handoff.corpus_root, handoff.corpus_manifest_sha256,
                                handoff.corpus_sha256, handoff.corpus_root /
                                "performance_corpus_manifest.json"), tuning_certificate),
                    ("held_out", (Path(revealed["root"]), revealed["manifest_sha256"],
                                  revealed["capsules_sha256"], Path(revealed["manifest"])),
                     heldout_certificate)):
                inputs = PAIRED.load_paired_inputs(
                    handoff.record_path, config.functional_run_id,
                    config.functional_submission_sha256, target, corpus_root=corpus_args[0],
                    corpus_manifest_sha256=corpus_args[1], corpus_capsules_sha256=corpus_args[2],
                    corpus_manifest=corpus_args[3], phase=phase,
                    gsim_certificate=certificate.path,
                    gsim_certificate_sha256=certificate.sha256)
                plan = PAIRED.build_measurement_plan(inputs)
                capsules.update((f"{phase}:{spec.family}", spec.capsule)
                                for spec in plan.schedule)
        declaration_stats = STATS.predeclare(
            trials=_statistics_trials(trial_evidence),
            capsules=[{"family": family, "capsule": capsule}
                      for family, capsule in sorted(capsules)],
            replicates=REPLICATES, primary_simulator="gsim")
        stats_path = root / "statistics_predeclaration.json"
        stats_payload = _canonical(declaration_stats)
        if stats_path.exists():
            if not stats_path.is_file() or stats_path.read_bytes() != stats_payload:
                raise ExperimentError("uncheckpointed statistics predeclaration differs from plan")
        else:
            stats_path.write_bytes(stats_payload)
            stats_path.chmod(0o444)
        stats_saved = {"path": str(stats_path.resolve()), "sha256": _sha_file(stats_path)}
        state.append("statistics_predeclared", stats_saved)
    stats_path = _verify_saved_file(
        stats_saved, root / "statistics_predeclaration.json",
        label="statistics predeclaration")
    declaration_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    measurement_manifests = _measure_cells(
        _measurement_cells(config, handoffs, revealed,
                           tuning_certificate=tuning_certificate,
                           heldout_certificate=heldout_certificate),
        config, state, command_runner=command_runner, workers=fanout)

    all_rows = [row for trial, path in measurement_manifests for row in _paired_rows(path, trial)]
    result = STATS.evaluate(declaration_stats, all_rows, trial_evidence=trial_evidence)
    if result.get("status") != "admitted":
        raise ExperimentError(f"all-cell performance statistic refused: {result.get('issues')}")
    final = {"schema": SCHEMA, "status": "GO", "declaration": declaration,
             "config_sha256": config_sha, "holdout": revealed,
             "chia_launch_receipt": chia_launch,
             "heldout_gsim_certificate": extension,
             "trials": trial_evidence, "functional_regrades": regrades,
             "agent_telemetry": {trial: {
                 "preflight_sha256": handoff.telemetry_evidence["preflight_sha256"],
                 "raw_trace": handoff.telemetry_evidence["artifacts"]["combined_raw"],
                 "aet_trajectory": handoff.telemetry_evidence["artifacts"]["trajectory"],
                 "aet_metrics_log": handoff.telemetry_evidence["artifacts"]["aet_metrics_log"],
                 "cost_time_toolcalls": handoff.telemetry_evidence["artifacts"][
                     "cost_time_toolcalls"],
                 "activity_share": handoff.telemetry_evidence["artifacts"]["activity_share"],
                 "tool_call_count": handoff.telemetry_evidence["tool_call_count"],
                 "subagent_tool_calls_tracked": handoff.telemetry_evidence[
                     "subagent_tool_calls_tracked"],
             } for trial, handoff in handoffs.items()},
             "measurement_manifests": [{"trial": trial, "path": str(path),
                                         "sha256": _sha_file(path)}
                                        for trial, path in measurement_manifests],
             "statistics_predeclaration_sha256": stats_saved["sha256"],
             "statistics": result,
             "selection": "all_three_trials_all_predeclared_cells_no_best_of_no_drop"}
    return _seal_final(root, final)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument("--perf-capsules", default="all",
                        help="comma-separated performance capsules to measure, or 'all'")
    parser.add_argument("--perf-families", default="all",
                        help="comma-separated performance families to measure, or 'all'")
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument("--waive-functional-gate", action="append", default=[],
                        metavar="PREDICATE",
                        help="accept a NAMED completeness gap in the functional baseline instead of "
                             "refusing (repeatable). Integrity predicates cannot be waived; asking is "
                             "an error. Recorded in the manifest and marks the run not gate-clean.")
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--perf-profile", type=Path, required=True)
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument("--functional-gsim-certificate", type=Path)
    parser.add_argument("--functional-gsim-certificate-sha256")
    parser.add_argument("--heldout-qualification-timeout", type=int, default=3600)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--wall-budget-seconds", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-timeout-seconds", type=int, required=True)
    parser.add_argument("--max-tool-calls", type=int, required=True)
    parser.add_argument("--tool-timeout-seconds", type=int, required=True)
    parser.add_argument("--smoke-replicates", type=int, default=1)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument("--generalization-count", type=int, default=4)
    parser.add_argument("--measurement-timeout", type=int, default=3600)
    parser.add_argument("--gsim-max-cycles", type=int)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--telemetry-price-table", type=Path)
    parser.add_argument("--chia-python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _raw = {key: value for key, value in vars(args).items() if key != "dry_run"}
    _raw["waive_functional_gate"] = tuple(_raw.pop("waive_functional_gate", ()) or ())
    config = Config(**_raw)
    try:
        outcome = run(config, dry_run=args.dry_run)
    except Exception as exc:
        print(f"NO-GO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome if isinstance(outcome, dict) else {"manifest": str(outcome)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
