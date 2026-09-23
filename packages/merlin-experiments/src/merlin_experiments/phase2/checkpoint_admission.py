#!/usr/bin/env python3
"""Resume-safe, three-trial orchestrator for the arms-rigorous performance experiment.

The orchestrator owns ordering and admission; existing modules own every substantive gate.  It commits
the hidden holdout before authoring, seals three identically configured agent trials, regrades each
candidate on the complete public+hidden functional L3 suite, reveals the holdout, predeclares all
paired measurements, and evaluates every GSIM cell without best-of selection or failed-cell dropping.
Paid agents and simulators are launched only through an injected command runner.

The authoring trials and the paired measurement matrix are independent children, so a launch may
declare how many of them may run at once with ``MERLIN_PERF_CAMPAIGN_FANOUT`` (default 1, the fully
serial campaign; 6 gives both phases their full width).  Independently of launch width,
the checkpoint chain, the trial evidence and the measurement matrix are recorded on the main thread
in fixed order, so the record does not depend on which child finishes first.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from merlin.common.paths import module_source_path
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.contract.materialize import materialize_public_cohort
from merlin.targetgen.sandbox import preflight as SANDBOX_PREFLIGHT
from merlin.targetgen.sandbox.answer_surfaces import dropped_declarations
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase2 import authoring as AUTHORING
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import corpus as P2_CORPUS
from merlin_experiments.phase2 import functional_cohort as FC
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import gsim_workload as WORKLOAD
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT
from merlin_experiments.phase2 import telemetry as TEL
from merlin_experiments.phase2.chia_launch import PYTHON_SOURCE_ENVIRONMENT_KEYS
from merlin_experiments.phase2.contracts import PerformanceExperimentError as ExperimentError

TRIALS = ("trial_00", "trial_01", "trial_02")
REPLICATES = ("r000", "r001")
SCHEMA = "merlin.agentic-performance-experiment.v1"
MEASUREMENT_CACHE_CONDITION = "warm"


@dataclass(frozen=True)
class ExecutionContext:
    """Explicit experiment resources and the exact invocation admitted by Chia."""

    source_root: Path
    contract_root: Path
    functional_runs_root: Path
    stage_root: Path
    measurement_root: Path
    holdout_sources: HOLDOUT.HoldoutSourceContext
    chia_wrapper: Path
    invocation: tuple[str, ...]
    suite: str

    def __post_init__(self) -> None:
        for name in (
            "source_root",
            "contract_root",
            "functional_runs_root",
            "stage_root",
            "measurement_root",
            "chia_wrapper",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ExperimentError(f"execution context {name} must be an absolute Path")
        if self.holdout_sources.source_root != self.source_root:
            raise ExperimentError("holdout and execution source roots differ")
        if (
            not self.invocation
            or not isinstance(self.invocation, tuple)
            or any(not isinstance(value, str) or not value for value in self.invocation)
        ):
            raise ExperimentError("execution context requires an exact nonempty invocation tuple")
        if not self.suite.strip():
            raise ExperimentError("execution context requires an explicit telemetry suite")


@dataclass(frozen=True)
class Config:
    context: ExecutionContext
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
    # Launch WITHOUT the public+hidden functional cross-validation certificate, accepting GSIM's
    # functional verdicts on the same terms phase 1 already accepted them.
    #
    waive_functional_gsim_certificate: bool = False
    heldout_qualification_timeout: int = 600
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
    # Declared execution width per measurement cell, recorded in measured evidence.
    # One by default, because a formal campaign's width is DECLARED rather than inferred.
    sim_workers: int = 1


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CommandResult]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _verify_resume_chia_identity(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Refuse a campaign resume under a different predeclared command or source stack.

    Receipt location, receipt digest, and runtime assignment are deliberately per-dispatch evidence.
    The plan and content-addressed source identities are the stable campaign treatment.
    """
    stable_keys = (
        "plan_sha256",
        "required_resources",
        "wrapper",
        "chia_trace",
        "command",
        "command_artifacts",
        "launch_policy",
    )
    if any(saved.get(key) != current.get(key) for key in stable_keys):
        raise ExperimentError("CHIA resume command/source identity differs from the saved predeclaration")


def _verify_resume_declaration(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Keep every predeclared treatment/gate byte stable, not only the CHIA envelope."""
    saved_declaration = saved.get("declaration")
    if not isinstance(saved_declaration, Mapping) or dict(saved_declaration) != dict(current):
        raise ExperimentError("campaign declaration differs from the saved predeclaration")


def _verify_trial_treatments(
    handoffs: Mapping[str, RECORD.VerifiedCandidateHandoff], expected: Mapping[str, Any]
) -> None:
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
        current = TEL.prepare(
            authoring_stage=Path(AUTHORING.__file__).resolve(),
            model=config.model,
            price_table=config.telemetry_price_table,
            codex_binary=config.codex_binary,
        )
        identity = TEL.treatment_identity(current)
    except P2_CONTRACTS.StageGateError as exc:
        raise ExperimentError(f"live agent treatment preflight failed: {exc}") from exc
    if identity != dict(expected):
        raise ExperimentError("live agent treatment differs from the saved predeclaration")


def _verify_trial_contract(trial: str, handoff: RECORD.VerifiedCandidateHandoff, expected: Mapping[str, Any]) -> None:
    """Reject a complete/stale child unless it exactly implements its declared paid trial."""
    if dict(handoff.agent_contract) != dict(expected):
        raise ExperimentError(f"{trial} agent stage differs from its predeclared trial contract")


def _safe(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise ExperimentError(f"{label} must be one safe path component")
    return value


def _config_document(config: Config) -> dict[str, Any]:
    def json_value(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        return value

    document = asdict(config)
    document["context"] = json_value(document["context"])
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
            if (
                row.get("config_sha256") != self.config_sha256
                or row.get("index") != len(rows)
                or row.get("previous_sha256") != (rows[-1]["sha256"] if rows else None)
            ):
                raise ExperimentError(f"checkpoint chain is invalid: {path}")
            rows.append({**row, "sha256": digest, "path": str(path.resolve())})
        return rows

    def append(self, stage: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        rows = self.load()
        if any(row["stage"] == stage for row in rows):
            raise ExperimentError(f"checkpoint stage is duplicated: {stage}")
        body = {
            "schema": SCHEMA,
            "index": len(rows),
            "stage": stage,
            "config_sha256": self.config_sha256,
            "previous_sha256": rows[-1]["sha256"] if rows else None,
            "evidence": dict(evidence),
        }
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


def subprocess_runner(argv: Sequence[str], cwd: Path, environment: Mapping[str, str]) -> CommandResult:
    from merlin_experiments.frozen_python import inherited_python_command

    completed = subprocess.run(
        inherited_python_command(argv), cwd=cwd, env=dict(environment), capture_output=True, text=True
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def child_environment(config: Config, certificate: GATE.CertificateRecord) -> dict[str, str]:
    """Pin GSIM selection for child stages; ambient engine paths/cycle caps never decide a run."""
    from merlin.runtime.backends import base as backends

    backend = backends.get_backend(certificate.target)
    configure = getattr(backend, "runtime_environment", None)
    if not callable(configure):
        raise ExperimentError("selected backend lacks pure runtime environment configuration")
    binaries = {}
    for engine in ("gsim", "verilator"):
        pin = certificate.pins[f"{engine}_binary"]
        path = Path(pin["path"])
        if path.is_symlink() or not path.is_file() or _sha_file(path) != pin["sha256"]:
            raise ExperimentError(f"runtime {engine} binary differs from the certificate pin")
        binaries[engine] = path.resolve(strict=True)
    # Backend runtime selection may configure tools, not replace the admitted
    # interpreter's source selection. Bind absence as well as present values, and
    # retain a separate copy so a callback cannot mutate its own comparison base.
    inherited = dict(os.environ)
    configured = configure(binaries=binaries, gsim_max_cycles=config.gsim_max_cycles, environment=dict(inherited))
    if not isinstance(configured, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in configured.items()
    ):
        raise ExperimentError("selected backend returned a malformed runtime environment")
    if any(configured.get(key) != inherited.get(key) for key in PYTHON_SOURCE_ENVIRONMENT_KEYS):
        raise ExperimentError("selected backend changed admitted Python source selection")
    environment = {
        **configured,
        "MERLIN_TARGET_EXPERIMENT": str(config.descriptor.resolve()),
        "MERLIN_CACHE_STATE": MEASUREMENT_CACHE_CONDITION,
        # The binary pin alone makes GSIM available but does not force the equal-fidelity
        # engine policy to choose it when another RTL engine is also installed. This run's
        # certificate is specifically about the pinned GSIM build, so bind both operator L3
        # and model tile dispatch to that engine.
        "MERLIN_REQUIRED_RTL_ENGINE": "gsim",
    }
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

            raw = os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or ""
        except Exception:  # noqa: BLE001
            raw = os.environ.get("AET_PRICE_TABLE", "")
        price_source = Path(raw) if raw else None
    if price_source is None:
        raise ExperimentError("AET_PRICE_TABLE is required for content-addressed telemetry pricing")
    for field, source in (
        ("rtl_facts", config.rtl_facts),
        ("perf_profile", config.perf_profile),
        ("telemetry_price_table", price_source),
    ):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ExperimentError(f"{field} snapshot source is absent, linked, or non-regular: {source}")
        payload = source.read_bytes()
        digest = _sha_bytes(payload)
        suffix = source.suffix if source.suffix else ".evidence"
        destination = destination_root / f"{field}.{digest}{suffix}"
        if destination.exists():
            if not destination.is_file() or destination.is_symlink() or destination.read_bytes() != payload:
                raise ExperimentError(f"content-addressed {field} snapshot is inconsistent")
        else:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
        destination.chmod(0o444)
        replacements[field] = destination.resolve()
        evidence[field] = {"source": str(source.resolve()), "snapshot": str(destination.resolve()), "sha256": digest}
    return replace(config, **replacements), evidence


def _verify_functional_certificate(
    certificate: GATE.CertificateRecord, cohort: FC.FunctionalGradeCohort
) -> dict[str, Any]:
    """Independently derive exact or predeclared sampled functional coverage."""
    from merlin_experiments.phase2 import functional_coverage as COVERAGE

    descriptors = FC.functional_gsim_cases(cohort)
    identities: dict[str, list[str]] = {}
    for capsule in descriptors:
        identities.setdefault(capsule.workload_sha256, []).append(str(capsule.manifest))
    metadata = certificate.document.get("functional_coverage")
    if "functional_coverage" in certificate.document and metadata is None:
        raise ExperimentError("functional coverage metadata is null rather than absent")
    try:
        coverage = COVERAGE.verify(
            metadata,
            [(capsule.workload_sha256, capsule.manifest) for capsule in descriptors],
            set(certificate.members),
            pins=certificate.pins,
        )
    except COVERAGE.CoverageError as exc:
        raise ExperimentError(str(exc)) from exc
    return {
        "public_source_descriptors": cohort.public_source_count,
        "public_descriptors": len(cohort.public),
        "hidden_source_descriptors": cohort.hidden_source_count,
        "hidden_descriptors": len(cohort.hidden),
        "same_elf_certificate_descriptors": len(descriptors),
        "dynamic_model_regrade_descriptors": (len(cohort.public) + len(cohort.hidden) - len(descriptors)),
        "same_elf_certificate_scope": "admitted_non_model_descriptors",
        "model_certificate_scope": "full_regrade_dynamic_tile_gsim_execution_ledger",
        "distinct_workload_sha256": sorted(identities),
        "coverage_mode": coverage["mode"],
        "independently_verified_workload_sha256": coverage["selected"],
        "unsampled_workload_sha256": coverage["unsampled"],
    }


def _functional_regrade_inputs(
    target: object, cohort: FC.FunctionalGradeCohort, *, contract_root: Path, public_destination: Path
) -> tuple[str, str]:
    """Build the canonical public grade view and bind both CLI inputs to ``cohort``.

    The public CLI input is already admission-filtered and carries the materializer's source/admitted
    record.  The hidden CLI input deliberately remains the full source root: ``grade_agent_run`` applies
    the same post-freeze capability selector used by :func:`_functional_grade_cohort`, preserving the
    honest hidden source/admitted counts without exposing its excluded names.
    """
    current = FC.functional_grade_cohort(target, contract_root=contract_root)
    if current != cohort:
        raise ExperimentError("functional grade cohort changed after preflight")
    public_root = materialize_public_cohort(target, tier_ceiling="L3", destination=public_destination)
    materialized_public = discover_capsules(public_root, labels={"public", "dev"}, contract=contract_root)
    if {str(cap.get("name")) for cap in materialized_public} != {capsule.name for capsule in cohort.public}:
        raise ExperimentError("materialized public regrade cohort differs from certificate cohort")
    hidden_roots = ",".join(str(path) for path in target.hidden_roots())
    if not hidden_roots:
        raise ExperimentError("full functional regrade needs nonempty hidden source roots")
    return str(public_root), hidden_roots


def _functional_qualification_descriptor(
    certificate: GATE.CertificateRecord, cohort: FC.FunctionalGradeCohort
) -> tuple[Path, dict[str, Any]]:
    """Resolve the frozen target descriptor and prove the adjacent qualification binds this cert."""
    root = certificate.path.parent.resolve()
    declarations = sorted(root.glob("declaration.*.json"))
    completions = sorted(root.glob("completion.*.json"))
    if len(declarations) != 1 or len(completions) != 1:
        raise ExperimentError("functional certificate lacks one sealed qualification chain")
    declaration_path, completion_path = declarations[0], completions[0]
    declaration_sha = _sha_file(declaration_path)
    if declaration_path.name != f"declaration.{declaration_sha}.json":
        raise ExperimentError("functional qualification declaration is not content-addressed")
    try:
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperimentError("functional qualification chain is unreadable") from exc
    target = declaration.get("target_descriptor") or {}
    descriptor = Path(str(target.get("path") or ""))
    try:
        descriptor = descriptor.resolve(strict=True)
        descriptor.relative_to(root / "inputs")
    except (OSError, ValueError) as exc:
        raise ExperimentError("functional qualification target descriptor escapes its snapshot") from exc
    functional_record = completion.get("functional_certificate") or {}
    declared_cohort = declaration.get("cohort") or {}
    if (
        target.get("sha256") != _sha_file(descriptor)
        or target.get("sha256") != cohort.admission_descriptor_sha256
        or descriptor.stat().st_mode & 0o222
        or completion.get("declaration_sha256") != declaration_sha
        or functional_record.get("sha256") != certificate.sha256
        or Path(str(functional_record.get("path") or "")).resolve() != certificate.path.resolve()
        or set(functional_record.get("workload_sha256") or ()) != set(certificate.members)
        or declaration.get("functional_coverage") != certificate.document.get("functional_coverage")
        or declared_cohort.get("public_source_descriptors") != cohort.public_source_count
        or declared_cohort.get("public_descriptors") != len(cohort.public)
        or declared_cohort.get("hidden_source_descriptors") != cohort.hidden_source_count
        or declared_cohort.get("hidden_descriptors") != len(cohort.hidden)
    ):
        raise ExperimentError("functional qualification chain differs from its frozen cohort")
    return descriptor, {
        "path": str(descriptor),
        "sha256": str(target["sha256"]),
        "declaration": str(declaration_path.resolve()),
        "declaration_sha256": declaration_sha,
        "completion": str(completion_path.resolve()),
        "completion_sha256": _sha_file(completion_path),
    }


def _frozen_functional_regrade_inputs(root: Path, cohort: FC.FunctionalGradeCohort) -> tuple[str, str, Path]:
    """Bind the regrade CLI to the exact Phase-1 public and hidden snapshot inputs."""
    if (
        cohort.frozen_contract is None
        or not cohort.frozen_hidden_source_dirs
        or not isinstance(cohort.public_admission, Mapping)
    ):
        raise ExperimentError("functional cohort lacks its frozen regrade inputs")
    contract = Path(cohort.frozen_contract)
    if contract.is_symlink() or not contract.is_dir():
        raise ExperimentError("frozen functional contract is absent or linked")
    for capsule in (*cohort.public, *cohort.hidden):
        if (
            capsule.manifest.is_symlink()
            or not capsule.manifest.is_file()
            or _sha_file(capsule.manifest) != capsule.manifest_sha256
            or GATE.workload_sha256(WORKLOAD.derive_workload(capsule.manifest)) != capsule.workload_sha256
        ):
            raise ExperimentError(f"frozen functional capsule changed: {capsule.name}")
    hidden = discover_capsules(list(cohort.frozen_hidden_source_dirs), labels={"hidden"}, contract=contract)
    hidden_names = tuple(str(cap.get("name")) for cap in hidden)
    FC.score_cohort_boundary(
        {
            "per_capsule": [{"capsule": capsule.name} for capsule in cohort.hidden],
            "n_capsules": len(cohort.hidden),
            "cohort_admission": {
                "n_source_capsules": cohort.hidden_source_count,
                "n_admitted_capsules": len(cohort.hidden),
                "n_capability_excluded": cohort.hidden_source_count - len(cohort.hidden),
                "n_resource_excluded": 0,
                "admitted_name_set_sha256": FC.name_set_sha256(tuple(capsule.name for capsule in cohort.hidden)),
                "excluded_name_set_sha256": FC.name_set_sha256(
                    tuple(set(hidden_names) - {capsule.name for capsule in cohort.hidden})
                ),
            },
        },
        label="hidden regrade",
        source_names=hidden_names,
        recorded_count=len(cohort.hidden),
    )
    admission_root = Path(root) / "functional_regrade_inputs/public_admission"
    admission_root.mkdir(parents=True, exist_ok=True)
    admission_path = admission_root / ".cohort_admission.json"
    payload = _canonical(dict(cohort.public_admission))
    if admission_path.exists():
        if admission_path.is_symlink() or admission_path.read_bytes() != payload:
            raise ExperimentError("functional public admission snapshot changed across resume")
    else:
        descriptor = os.open(admission_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    admission_path.chmod(0o444)
    admission_root.chmod(0o555)
    public_roots = tuple(dict.fromkeys(str(capsule.manifest.parent) for capsule in cohort.public)) + (
        str(admission_root),
    )
    hidden_roots = tuple(dict.fromkeys(str(path) for path in cohort.frozen_hidden_source_dirs))
    return ",".join(public_roots), ",".join(hidden_roots), contract.resolve()


def _verify_tuning_certificate(
    certificate: GATE.CertificateRecord, target: object, capsules: str = "all", families: str = "all"
) -> dict[str, Any]:
    """Require the initial certificate to be exactly the descriptor-derived tuning corpus.

    The corpus is narrowed by the SAME selection the trials will measure. Several families measure the
    same workload under a different lever, so the full phase contains repeated workload identities
    by design; the campaign measures one member per identity, and validating the unselected corpus
    here would refuse a launch over members it was never going to run.
    """
    capsule_selection = None if capsules in (None, "", "all") else capsules
    family_selection = None if families in (None, "", "all") else families
    selected = capsule_selection is not None or family_selection is not None
    corpus = P2_CORPUS.discover_performance_corpus(target, capsules=capsule_selection, families=family_selection)
    # MANY MEMBERS MAY SHARE ONE IDENTITY, exactly as the docstring above describes. This grouped
    # form replaces a 1:1 map that raised on the first repeat -- which contradicted the documented
    # design and refused a launch over it. Measured 2026-09-06: PK00_k16, PM00_m16n16 and
    # PR00_fits_double_k16 are all 16x16x16, the shared anchor of three sweeps, and every one of the
    # three families REQUIRES its anchor (PK needs exactly four descriptors, PR needs three depths in
    # its `fits_double` band), so no capsule could be dropped to satisfy the 1:1 assumption.
    #
    # What the certificate must cover is the identity SET, and that comparison is unchanged below --
    # this is the same many-to-one grouping `functional_gsim_qualification.derive_cases` already
    # applies on the functional side.
    identities: dict[str, list[str]] = {}
    for member in corpus.capsules:
        identity = GATE.workload_sha256(WORKLOAD.derive_workload(member.source_dir / "capsule.yaml"))
        identities.setdefault(identity, []).append(f"{member.family}/{member.capsule}")
    # A NARROWED RUN MAY USE A WIDER CERTIFICATE. `missing` is the load-bearing half: every member
    # this campaign will measure must be certified. `extras` guards a different thing -- a
    # certificate carrying a workload this corpus does not contain at all -- so it is checked against
    # the FULL corpus, not the selection. Checking it against the selection instead refused a
    # 16-member PM launch against the 36-workload corpus certificate, which is exactly the situation
    # the selection exists to create (measured 2026-09-06).
    full_identities = set(identities)
    if selected:
        full_identities = {
            GATE.workload_sha256(WORKLOAD.derive_workload(member.source_dir / "capsule.yaml"))
            for member in P2_CORPUS.discover_performance_corpus(target).capsules
        }
    missing = sorted(set(identities) - set(certificate.members))
    extras = sorted(set(certificate.members) - full_identities)
    if missing or extras:
        raise ExperimentError(
            f"tuning GSIM certificate does not cover the derived corpus (missing={missing}, extras={extras})"
        )
    # `members` stays the DISTINCT-workload count it always was, and the capsule count is recorded
    # beside it so a reader can see where the two differ instead of inferring one from the other.
    return {
        "workload_sha256": sorted(identities),
        "members": len(identities),
        "corpus_capsules": sum(len(names) for names in identities.values()),
        "shared_identities": {
            identity: sorted(names) for identity, names in sorted(identities.items()) if len(names) > 1
        },
    }


def _require_same_gsim_build(
    reference: GATE.CertificateRecord, certificate: GATE.CertificateRecord, *, label: str
) -> None:
    """Keep every qualification phase on one exact RTL/simulator build."""
    changed = sorted(
        name for name in GATE.REQUIRED_PINS if certificate.pins[name]["sha256"] != reference.pins[name]["sha256"]
    )
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
        raise ExperimentError(f"functional qualification {stem} receipt is not content-addressed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"functional qualification {stem} receipt is unreadable") from exc
    if not isinstance(document, dict):
        raise ExperimentError(f"functional qualification {stem} receipt is not a mapping")
    return path.resolve(), digest, document


def _verify_functional_certificate_provenance(
    certificate: GATE.CertificateRecord, tuning: GATE.CertificateRecord, functional_submission_sha256: str
) -> dict[str, Any]:
    """Bind the qualification envelope to the exact sealed compiler and source certificate."""
    from merlin_experiments.phase2 import functional_qualification as qualification

    certificate_path = Path(certificate.path)
    if certificate_path.is_symlink() or not certificate_path.is_file():
        raise ExperimentError("functional GSIM certificate is absent or linked")
    root = certificate_path.resolve().parent
    if certificate_path.name != f"functional-certificate.{certificate.sha256}.json":
        raise ExperimentError("functional GSIM certificate is not producer-content-addressed")
    declaration_path, declaration_sha, declaration = _content_addressed_document(root, "declaration")
    completion_path, completion_sha, completion = _content_addressed_document(root, "completion")
    try:
        qualification.validate_contract_snapshot(root, declaration)
        if "execution_policy" in declaration:
            qualification.validate_execution_policy(root, declaration)
    except qualification.FunctionalQualificationError as exc:
        raise ExperimentError(str(exc)) from exc

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
            "functional qualification source certificate or target descriptor is unavailable"
        ) from exc
    expected_pins = {name: tuning.pins[name]["sha256"] for name in sorted(GATE.REQUIRED_PINS)}
    declared_workloads = (
        {str(row.get("workload_sha256")) for row in cases}
        if isinstance(cases, list) and all(isinstance(row, Mapping) for row in cases)
        else set()
    )
    metadata = declaration.get("functional_coverage")
    if metadata != certificate.document.get("functional_coverage"):
        raise ExperimentError("functional certificate coverage differs from its sealed declaration")
    if metadata is not None:
        from merlin_experiments.phase2 import functional_coverage as COVERAGE

        try:
            if not isinstance(cases, list) or not cases:
                raise COVERAGE.CoverageError("functional declaration has no cases")
            declared_cases = []
            for row in cases:
                manifest = Path(row["representative_manifest"])
                if manifest.is_symlink() or _sha_file(manifest) != row["representative_manifest_sha256"]:
                    raise COVERAGE.CoverageError("functional sample descriptor changed after declaration")
                declared_cases.append((row["workload_sha256"], manifest))
            verified = COVERAGE.verify(metadata, declared_cases, set(certificate.members), pins=certificate.pins)
            declared_workloads = set(verified["selected"])
        except (COVERAGE.CoverageError, KeyError, TypeError, OSError) as exc:
            raise ExperimentError(f"invalid functional coverage declaration: {exc}") from exc
    if (
        declaration.get("schema") != qualification.SCHEMA
        or declaration.get("policy") != "formal-public-plus-hidden-admission-distinct-workloads.v1"
        or declaration.get("target") != certificate.target
        or baseline.get("sha256") != functional_submission_sha256
        or target_descriptor.get("sha256") != _sha_file(descriptor_path)
        or source.get("sha256") != tuning.sha256
        or source_path != tuning_path
        or declared_pins != expected_pins
        or declared_workloads != set(certificate.members)
    ):
        raise ExperimentError(
            "functional qualification declaration is not bound to the exact sealed functional "
            "submission and tuning certificate"
        )

    completed_source = completion.get("source_certificate") or {}
    completed_certificate = completion.get("functional_certificate") or {}
    selected_path = Path(str(completed_certificate.get("path") or ""))
    completed_source_path = Path(str(completed_source.get("path") or ""))
    try:
        selected_path = selected_path.resolve(strict=True)
        completed_source_path = completed_source_path.resolve(strict=True)
    except OSError as exc:
        raise ExperimentError("functional qualification completion selects an absent certificate") from exc
    if (
        completion.get("schema") != qualification.SCHEMA
        or completion.get("status") != "complete"
        or completion.get("declaration_sha256") != declaration_sha
        or completed_source.get("sha256") != tuning.sha256
        or completed_source_path != tuning_path
        or selected_path != certificate_path.resolve()
        or completed_certificate.get("sha256") != certificate.sha256
        or set(completed_certificate.get("workload_sha256") or ()) != set(certificate.members)
    ):
        raise ExperimentError("functional qualification completion does not select this exact certificate")
    return {
        "root": str(root),
        "declaration": str(declaration_path),
        "declaration_sha256": declaration_sha,
        "completion": str(completion_path),
        "completion_sha256": completion_sha,
        "functional_submission_sha256": functional_submission_sha256,
        "source_certificate_sha256": tuning.sha256,
        "target_descriptor": {"path": str(descriptor_path), "sha256": target_descriptor.get("sha256")},
    }


def _resolve_chia_python(explicit: Path | None) -> Path | None:
    from merlin.benchharness.chia_bridge import chia_python

    try:
        return Path(chia_python(explicit)).absolute()
    except RuntimeError as exc:
        raise ExperimentError(str(exc)) from exc


def _chia_canary(python: Path | None, *, context: ExecutionContext) -> dict[str, Any]:
    """Read-only external-env import canary; never initializes Ray or launches work."""
    resolved = _resolve_chia_python(python)
    if resolved is None or not resolved.is_file():
        raise ExperimentError("a CHIA venv Python is required for the orchestration capability canary")
    bridge = module_source_path("merlin.benchharness.chia_bridge").resolve()
    wrapper = context.chia_wrapper
    program = (
        "import json,chia.trace,ray;"
        "from merlin.benchharness.chia_bridge import chia_available;"
        "assert chia_available();"
        "print(json.dumps({'chia_trace':chia.trace.__file__,'ray':ray.__version__}))"
    )
    environment = dict(os.environ)
    python_path = os.pathsep.join(
        str(path)
        for path in dict.fromkeys(
            (
                context.holdout_sources.core_package_root.parent,
                context.holdout_sources.experiments_package_root.parent,
                context.holdout_sources.experiments_namespace_root.parent,
            )
        )
    )
    environment["PYTHONPATH"] = python_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    from merlin_experiments.frozen_python import inherited_python_command

    completed = subprocess.run(
        inherited_python_command([str(resolved), "-c", program]),
        cwd=context.source_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
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
        "python": str(resolved),
        "python_resolved": str(resolved.resolve()),
        "python_sha256": _sha_file(resolved),
        "chia_trace": str(trace_path),
        "chia_trace_sha256": _sha_file(trace_path),
        "ray_version": str(probe.get("ray") or ""),
        "bridge": str(bridge),
        "bridge_sha256": _sha_file(bridge),
        "required_entrypoint": str(wrapper),
        "required_entrypoint_sha256": _sha_file(wrapper),
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
    if tuple(REPLICATES) != tuple(f"r{index:03d}" for index in range(len(REPLICATES))):
        raise ExperimentError("measurement identities must be a dense r000.. sequence")
    integers = (
        config.wall_budget_seconds,
        config.rounds,
        config.round_timeout_seconds,
        config.max_tool_calls,
        config.tool_timeout_seconds,
        config.smoke_replicates,
        config.holdout_count,
        config.measurement_timeout,
    )
    integers = (*integers, config.heldout_qualification_timeout, config.generalization_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integers):
        raise ExperimentError("all budgets/counts must be positive integers")
    bounded_simulator_deadlines = {
        "tool_timeout_seconds": config.tool_timeout_seconds,
        "measurement_timeout": config.measurement_timeout,
        "heldout_qualification_timeout": config.heldout_qualification_timeout,
    }
    oversized = {name: value for name, value in bounded_simulator_deadlines.items() if value > ITERATION_MAX_SECONDS}
    if oversized:
        raise ExperimentError(
            f"performance-loop simulator deadlines exceed {ITERATION_MAX_SECONDS:g}s: "
            f"{oversized}; reduce the witness shape"
        )
    if not config.model.strip() or not config.effort.strip():
        raise ExperimentError("model and effort must be explicit")
    if config.gsim_max_cycles is not None and (
        isinstance(config.gsim_max_cycles, bool)
        or not isinstance(config.gsim_max_cycles, int)
        or config.gsim_max_cycles <= 0
    ):
        raise ExperimentError("predeclared GSIM max cycles must be a positive integer")
    target = load_target_experiment(config.descriptor, source_root=config.context.source_root)
    functional = FI.inspect_stage_functional_run(
        config.context.functional_runs_root,
        config.functional_run_id,
        config.functional_submission_sha256,
        waive=frozenset(config.waive_functional_gate or ()),
    )
    functional_cohort = FC.functional_grade_cohort_from_run(target, functional, source_root=config.context.source_root)
    # WHAT THIS SUBMISSION DECLINED, from the grade that recorded it. The names travel on the cohort so
    # the certificate BUILDER and the certificate VERIFIER derive one case set from one function;
    # deriving them separately is how the two come to disagree about what was certified.
    functional_cohort = replace(functional_cohort, declined=FC.declined_names(functional))
    certificate = GATE.load_certificate(config.gsim_certificate, expected_sha256=config.gsim_certificate_sha256)
    if certificate.target != target.target:
        raise ExperimentError("GSIM certificate target differs from the experiment target")
    tuning_coverage = _verify_tuning_certificate(certificate, target, config.perf_capsules, config.perf_families)
    pinned_gsim = Path(certificate.pins["gsim_binary"]["path"])
    if not pinned_gsim.is_file() or _sha_file(pinned_gsim) != certificate.pins["gsim_binary"]["sha256"]:
        raise ExperimentError("the tuning certificate's pinned GSIM binary is unavailable")
    blockers = []
    sandbox_probe = SANDBOX_PREFLIGHT.probe_sandbox()
    if not sandbox_probe.usable:
        blockers.append("the agent isolation sandbox cannot be built on this host: " + sandbox_probe.describe())
    declaration_drops = dropped_declarations(target)
    if declaration_drops:
        blockers.append(
            "declared answer-surface rules matched nothing and masked nothing: "
            + "; ".join(drop.describe() for drop in declaration_drops)
        )
    price_path = config.telemetry_price_table
    if price_path is None:
        try:
            from merlin.common.paths import _dotenv  # noqa: PLC0415

            raw_price = os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or ""
        except Exception:  # noqa: BLE001
            raw_price = os.environ.get("AET_PRICE_TABLE", "")
        price_path = Path(raw_price) if raw_price else None
    snapshot_candidates = [config.rtl_facts, config.perf_profile]
    if price_path is not None:
        snapshot_candidates.append(price_path)
    snapshot_needed = [
        str(path) for path in snapshot_candidates if Path(path).is_file() and Path(path).stat().st_mode & 0o222
    ]
    if snapshot_needed:
        blockers.append(
            "RTL facts/performance profile/telemetry price table include writable inputs and must "
            "first be copied into "
            "the experiment's content-addressed 0444 preflight_inputs evidence: " + ", ".join(snapshot_needed)
        )
        domain = {"status": "requires_immutable_snapshot"}
        rtl_provenance = None
    else:
        rtl_provenance = HOLDOUT.verify_rtl_facts_provenance(config.rtl_facts, target=target.target)
        domain = HOLDOUT.derive_domain(
            config.rtl_facts, config.perf_profile, target=target.target, context=config.context.holdout_sources
        )
    telemetry_preflight = None
    telemetry_treatment = None
    try:
        telemetry_preflight = TEL.prepare(
            authoring_stage=Path(AUTHORING.__file__).resolve(),
            model=config.model,
            price_table=price_path,
            codex_binary=config.codex_binary,
        )
        telemetry_treatment = TEL.treatment_identity(telemetry_preflight)
    except (P2_CONTRACTS.StageGateError, ExperimentError) as exc:
        blockers.append(f"agent telemetry preflight failed: {exc}")
    chia_preflight = None
    try:
        chia_preflight = _chia_canary(config.chia_python, context=config.context)
    except ExperimentError as exc:
        blockers.append(f"CHIA orchestration capability preflight failed: {exc}")
    functional_certificate = None
    functional_coverage = None
    functional_provenance = None
    functional_descriptor_binding = None
    functional_certificate_waiver = None
    if config.functional_gsim_certificate is None or config.functional_gsim_certificate_sha256 is None:
        if not config.waive_functional_gsim_certificate:
            blockers.append("full public+hidden functional-suite GSIM certificate is required before agent launch")
        else:
            # RECORDED, NOT SILENT. The waiver travels in the predeclaration so a result can never be
            # read without the condition it was produced under, and the run is not gate-clean. A
            # waiver that only removed a blocker would be indistinguishable from a certificate that
            # was checked and passed -- the failure this repo keeps paying for.
            functional_certificate_waiver = {
                "waived": True,
                "requirement": ("full public+hidden functional-suite GSIM equivalence certificate before agent launch"),
                "claim_withdrawn": (
                    "the functional regrade's pass/fail verdicts are GSIM-only and "
                    "uncorroborated by a second elaborated-RTL engine on the same "
                    "ELF"
                ),
                "claim_retained": (
                    "timing authority is unaffected: it is pinned by the TUNING "
                    "certificate, which is verified here and is not waivable"
                ),
                "same_standard_as": (
                    "phase 1, which graded this submission on GSIM at L3 and required no equivalence certificate"
                ),
                "gate_clean": False,
            }
            if getattr(target, "descriptor_sha256", None) != functional_cohort.admission_descriptor_sha256:
                blockers.append(
                    "functional certificate waiver cannot bind the changed live descriptor to the "
                    "frozen Phase-1 cohort; supply the qualification certificate"
                )
    else:
        functional_certificate = GATE.load_certificate(
            config.functional_gsim_certificate, expected_sha256=config.functional_gsim_certificate_sha256
        )
        if functional_certificate.target != target.target:
            raise ExperimentError("functional GSIM certificate target differs from experiment target")
        _require_same_gsim_build(certificate, functional_certificate, label="functional GSIM certificate")
        functional_provenance = _verify_functional_certificate_provenance(
            functional_certificate, certificate, functional.digest
        )
        functional_coverage = _verify_functional_certificate(functional_certificate, functional_cohort)
        _functional_descriptor, functional_descriptor_binding = _functional_qualification_descriptor(
            functional_certificate, functional_cohort
        )
    telemetry_sha = _sha_bytes(_canonical(telemetry_preflight)) if telemetry_preflight is not None else None
    trial_contract = {
        trial: {
            "model": config.model,
            "resolved_model": ((telemetry_treatment or {}).get("resolved_model")),
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
            "treatment_identity": telemetry_treatment,
        }
        for trial in TRIALS
    }
    if not heldout_certificate_provider_available:
        blockers.append(
            "post-seal held-out GSIM extension certificate provider is unavailable: "
            "produce_gsim_certificate.py can validate supplied captures but has no CLI/API that "
            "discovers the newly revealed capsules, builds their exact ELFs, captures same-ELF "
            "GSIM/Verilator output evidence, and emits an extension certificate; the tuning "
            "certificate cannot qualify them"
        )
    declaration = {
        "schema": SCHEMA,
        "status": "GO" if not blockers else "NO_GO",
        "blockers": blockers,
        "sandbox_preflight": sandbox_probe.as_record(),
        "dropped_answer_surface_declarations": [drop.as_record() for drop in declaration_drops],
        "experiment_id": config.experiment_id,
        "target": target.target,
        "trial_contracts": trial_contract,
        "trials": list(TRIALS),
        "replicates": list(REPLICATES),
        "holdout_commit_before_authoring": True,
        "holdout_domain_sha256": _sha_bytes(_canonical(domain)),
        "rtl_facts_provenance": rtl_provenance,
        "gsim_certificate_sha256": certificate.sha256,
        "tuning_gsim_coverage": tuning_coverage,
        "functional_gsim_certificate_sha256": (functional_certificate.sha256 if functional_certificate else None),
        "functional_gsim_coverage": functional_coverage,
        "functional_gsim_provenance": functional_provenance,
        "functional_descriptor_binding": functional_descriptor_binding,
        "functional_gsim_certificate_waiver": functional_certificate_waiver,
        "agent_telemetry": telemetry_preflight,
        "agent_treatment": telemetry_treatment,
        "orchestration": chia_preflight,
        "gsim_runtime": {
            "binary": str(pinned_gsim.resolve()),
            "binary_sha256": certificate.pins["gsim_binary"]["sha256"],
            "max_cycles": config.gsim_max_cycles,
            "ambient_selection_forbidden": True,
        },
        "heldout_certificate_phase": "required_after_all_candidates_seal_and_reveal",
        "functional_regrade": "full_public_plus_hidden_L3_before_performance",
        "measurement_phases": ["tuning", "held_out"],
        "measurement_engine_policy": {
            "semantic_screen": "spike_no_timing",
            "rtl_execution_backends": ["gsim"],
            "timing_authority": "gsim",
            "verilator": "prelaunch_certificate_qualification_only",
        },
        "selection": "all_trials_all_cells_no_best_of_no_drop",
    }
    return {**declaration, "declaration_sha256": _sha_bytes(_canonical(declaration))}
