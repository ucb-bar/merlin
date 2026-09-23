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

import json
import os
import shutil
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from merlin.benchharness import hash_tree
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import candidate_verification as VERIFY
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import chia_launch as CHIA
from merlin_experiments.phase2 import functional_cohort as FC
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import gsim_workload as WORKLOAD
from merlin_experiments.phase2 import heldout_qualification as HQUAL
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2 import revealed_corpus as RC
from merlin_experiments.phase2 import statistics as STATS
from merlin_experiments.phase2.contracts import PerformanceExperimentError as ExperimentError


def _run_checked(
    runner: AD.CommandRunner, argv: Sequence[str], *, environment: Mapping[str, str], context: AD.ExecutionContext
) -> None:
    result = runner(argv, context.source_root, environment)
    if result.returncode:
        raise ExperimentError(f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr[-1000:]}")


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
        raise ExperimentError(f"{FANOUT_ENVIRONMENT_VARIABLE} must be a positive integer, got {raw!r}")
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
        with ThreadPoolExecutor(max_workers=min(workers, len(pending)), thread_name_prefix="perf-campaign") as pool:
            launched = [(stage.name, pool.submit(stage.launch)) for stage in pending]
            for name, future in launched:
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - report every failure, not just the first
                    failures.append(f"{name}: {error}")
        if failures:
            raise ExperimentError(
                f"{len(failures)} of {len(pending)} concurrent child stages failed; " + "; ".join(failures)
            )
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
    if not path.is_file() or path.resolve() != expected.resolve() or AD._sha_file(path) != saved.get("sha256"):
        raise ExperimentError(f"{label} evidence changed across resume")
    return path


def _handoff(path: Path, target: object) -> RECORD.VerifiedCandidateHandoff:
    return VERIFY.verify_candidate_handoff(path, verify_authoring_tools=False, target_experiment=target)


def _verify_regrade(run_dir: Path, handoff: RECORD.VerifiedCandidateHandoff) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.is_file():
        raise ExperimentError(f"functional regrade manifest is absent: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if (
        manifest.get("submission_sha256") != handoff.candidate_sha256
        or (manifest.get("completion") or {}).get("formal_grade_complete") is not True
        or (manifest.get("public_dev") or {}).get("formal_complete") is not True
        or (manifest.get("hidden") or {}).get("formal_complete") is not True
    ):
        raise ExperimentError("candidate did not pass the full public+hidden functional L3 regrade")
    if str(hash_tree(handoff.candidate_path)["sha256"]) != handoff.candidate_sha256:
        raise ExperimentError("sealed candidate changed during functional regrade")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": AD._sha_file(manifest_path),
        "submission_sha256": handoff.candidate_sha256,
    }


def _prepare_regrade(run_dir: Path, handoff: RECORD.VerifiedCandidateHandoff) -> None:
    if run_dir.exists():
        return
    run_dir.mkdir(parents=True)
    shutil.copytree(
        handoff.candidate_path, run_dir / "submission", ignore=shutil.ignore_patterns("build", "__pycache__")
    )
    copied = run_dir / "submission"
    for path in (copied, *copied.rglob("*")):
        if path.is_symlink():
            raise ExperimentError("functional regrade copy contains a symlink")
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
    if str(hash_tree(copied)["sha256"]) != handoff.candidate_sha256:
        raise ExperimentError("functional regrade copy differs from the sealed candidate")


def _verify_measurement_manifest(
    manifest_path: Path,
    *,
    phase: str,
    functional_run_id: str,
    functional_submission_sha256: str,
    handoff: RECORD.VerifiedCandidateHandoff,
    corpus_manifest_sha256: str,
    corpus_capsules_sha256: str,
    certificate_sha256: str,
) -> dict[str, Any]:
    """Adapt coordinator admission inputs to the shared measurement verifier."""
    expected = ME.MeasurementBinding(
        phase,
        functional_run_id,
        functional_submission_sha256,
        handoff.record_sha256,
        handoff.candidate_sha256,
        corpus_manifest_sha256,
        corpus_capsules_sha256,
        certificate_sha256,
    )
    try:
        return ME.verify_paired_measurement(manifest_path, expected=expected).receipt()
    except ME.MeasurementEvidenceError as exc:
        raise ExperimentError(str(exc)) from exc


def _statistics_trials(trial_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Project sealed agent evidence onto the deliberately narrow predeclaration API."""
    return [{"trial": str(row["trial"]), "agent_run_id": str(row["agent_run_id"])} for row in trial_evidence]


def _seal_final(root: Path, document: Mapping[str, Any]) -> Path:
    payload, digest = AD._canonical(document), AD._sha_bytes(AD._canonical(document))
    path = root / f"experiment_manifest.{digest}.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise ExperimentError("final manifest digest collision")
        return path
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def _verify_extension_certificate(
    tuning: GATE.CertificateRecord,
    extension: GATE.CertificateRecord,
    holdout_manifest: Path,
    *,
    manifest_sha256: str | None = None,
    corpus_sha256: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """Require a true same-build envelope extension covering every exact revealed descriptor."""
    changed_pins = sorted(
        name for name in GATE.REQUIRED_PINS if extension.pins[name]["sha256"] != tuning.pins[name]["sha256"]
    )
    if changed_pins:
        raise ExperimentError(f"held-out extension changed pinned build artifacts: {changed_pins}")
    missing_tuning = sorted(set(tuning.members) - set(extension.members))
    if missing_tuning:
        raise ExperimentError("held-out certificate is not an extension of the tuning envelope")
    members = RC.load_revealed_members(
        holdout_manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_corpus_sha256=corpus_sha256,
        expected_target=target,
    )
    identities = {member.workload_sha256: member.name for member in members}
    missing = sorted(identity for identity in identities if identity not in extension.members)
    if missing:
        raise ExperimentError(
            "held-out extension does not cover every revealed workload: "
            + ", ".join(f"{identities[identity]}={identity}" for identity in missing)
        )
    extras = sorted(set(extension.members) - set(tuning.members) - set(identities))
    if extras:
        raise ExperimentError(
            "held-out extension contains workloads outside the exact tuning-plus-reveal envelope: " + ", ".join(extras)
        )
    return {
        "holdout_workload_sha256": sorted(identities),
        "tuning_workloads_retained": len(tuning.members),
        "heldout_workloads_covered": len(identities),
        "pins_unchanged": True,
    }


def _qualify_heldout_with_config(
    reveal_manifest: Path,
    qualification_root: Path,
    tuning: GATE.CertificateRecord,
    *,
    functional_base: Path,
    functional_base_sha256: str,
    reveal_manifest_sha256: str,
    reveal_corpus_sha256: str,
    config: AD.Config,
    target: object,
) -> tuple[Path, str]:
    """Thread declared runtime bounds into the in-process post-reveal provider."""

    return HQUAL.qualify_revealed_holdout(
        reveal_manifest,
        qualification_root,
        tuning,
        functional_base=functional_base,
        functional_base_sha256=functional_base_sha256,
        reveal_manifest_sha256=reveal_manifest_sha256,
        reveal_corpus_sha256=reveal_corpus_sha256,
        contract_root=config.context.contract_root,
        timeout=config.heldout_qualification_timeout,
        gsim_max_cycles=config.gsim_max_cycles,
        target_experiment=target,
    )


def _author_candidates(
    config: AD.Config,
    state: AD.Checkpoints,
    target: object,
    declaration: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    expected_treatment: Mapping[str, Any],
    command_runner: AD.CommandRunner,
    workers: int,
) -> tuple[dict[str, RECORD.VerifiedCandidateHandoff], list[dict[str, Any]]]:
    """Seal one candidate per trial, optionally with several agent stages in flight at once.

    The trials share nothing mutable: the hidden holdout is committed before this phase, each stage
    writes its own run directory, the frozen tuning corpus is proven byte-identical across trials
    afterwards, and independence is enforced by distinct evidence hashes.  The one order-sensitive
    act is the checkpoint append, which `run_child_stages` keeps on this thread in TRIALS order.
    """
    stage_root_base = config.context.stage_root
    stage_names = {trial: f"{config.experiment_id}__{trial}" for trial in AD.TRIALS}
    records = {trial: stage_root_base / name / "performance_candidate.json" for trial, name in stage_names.items()}

    stages: list[ChildStage] = []
    for trial in AD.TRIALS:
        if state.evidence(f"candidate:{trial}") is not None:
            continue
        stage_dir = stage_root_base / stage_names[trial]
        record = records[trial]
        launch: Callable[[], None] | None = None
        if _uncheckpointed_state(stage_dir, record, label=f"agent stage {trial}") == "absent":
            AD._verify_live_agent_treatment(config, expected_treatment)
            command = [
                sys.executable,
                "-m",
                "merlin_experiments.phase2.authoring_cli",
                "--suite",
                config.context.suite,
                "--functional-runs-root",
                str(config.context.functional_runs_root),
                "--stage-root",
                str(stage_dir),
                "--source-root",
                str(config.context.source_root),
                "--contract-root",
                str(config.context.contract_root),
                "--functional-run-id",
                config.functional_run_id,
                "--functional-submission-sha256",
                config.functional_submission_sha256,
                "--model",
                config.model,
                "--effort",
                config.effort,
                "--wall-budget-seconds",
                str(config.wall_budget_seconds),
                "--rounds",
                str(config.rounds),
                "--round-timeout-seconds",
                str(config.round_timeout_seconds),
                "--max-tool-calls",
                str(config.max_tool_calls),
                "--tool-timeout-seconds",
                str(config.tool_timeout_seconds),
                "--smoke-replicates",
                str(config.smoke_replicates),
                "--replicates",
                str(len(AD.REPLICATES)),
                "--codex-binary",
                config.codex_binary,
                "--gsim-certificate",
                str(config.gsim_certificate),
                "--gsim-certificate-sha256",
                config.gsim_certificate_sha256,
                "--rtl-facts",
                str(config.rtl_facts),
                "--telemetry-price-table",
                str(config.telemetry_price_table),
                "--descriptor",
                str(config.descriptor),
            ]
            # A MEMBER THE ORACLE CANNOT EXECUTE MUST NOT HOLD THE WHOLE CORPUS HOSTAGE.
            # The stage refuses any member outside the certificate, and a member the reference
            # simulator cannot run cannot be certified -- so without a way to name it, one
            # unrunnable capsule blocks every other one. The selection is recorded in the run
            # manifest, so what was measured is never in doubt.
            if config.perf_capsules and config.perf_capsules != "all":
                command += ["--capsules", config.perf_capsules]
            if config.perf_families and config.perf_families != "all":
                command += ["--families", config.perf_families]
            # FORWARD THE WAIVERS. The coordinator admitted this functional baseline under exactly
            # these named gaps; a trial that re-checks it without them refuses a run its own campaign
            # already accepted.
            for predicate in config.waive_functional_gate or ():
                command += ["--waive-functional-gate", predicate]
            launch = partial(_run_checked, command_runner, command, environment=environment, context=config.context)

        def commit(trial: str = trial, record: Path = record) -> dict[str, Any]:
            handoff = _handoff(record, target)
            AD._verify_trial_contract(trial, handoff, declaration["trial_contracts"][trial])
            saved = {
                "record": str(record.resolve()),
                "record_sha256": handoff.record_sha256,
                "candidate_sha256": handoff.candidate_sha256,
            }
            state.append(f"candidate:{trial}", saved)
            return saved

        stages.append(ChildStage(f"candidate:{trial}", launch, commit))
    run_child_stages(stages, workers=workers)

    handoffs: dict[str, RECORD.VerifiedCandidateHandoff] = {}
    trial_evidence: list[dict[str, Any]] = []
    for trial in AD.TRIALS:
        saved = state.evidence(f"candidate:{trial}")
        if saved is None:
            raise ExperimentError(f"agent stage evidence is missing after it ran: {trial}")
        handoff = _handoff(Path(saved["record"]), target)
        if handoff.record_sha256 != saved["record_sha256"]:
            raise ExperimentError(f"candidate record changed across resume: {trial}")
        AD._verify_trial_contract(trial, handoff, declaration["trial_contracts"][trial])
        if handoff.telemetry_evidence.get("preflight_sha256") != AD._sha_bytes(
            AD._canonical(declaration["agent_telemetry"])
        ):
            raise ExperimentError(f"candidate telemetry stack differs from predeclaration: {trial}")
        handoffs[trial] = handoff
        trial_evidence.append(
            {"trial": trial, "agent_run_id": stage_names[trial], "agent_evidence_sha256": handoff.record_sha256}
        )
    return handoffs, trial_evidence


@dataclass(frozen=True)
class _MeasurementCell:
    """One (trial, phase) paired-bench cell, addressed by its own fresh run directory."""

    trial: str
    phase: str
    handoff: RECORD.VerifiedCandidateHandoff
    corpus_root: Path
    corpus_manifest: Path
    corpus_manifest_sha256: str
    corpus_capsules_sha256: str
    certificate: GATE.CertificateRecord
    stage: str
    run_id: str
    output: Path


def _measurement_cells(
    config: AD.Config,
    handoffs: Mapping[str, RECORD.VerifiedCandidateHandoff],
    revealed: Mapping[str, Any],
    *,
    tuning_certificate: GATE.CertificateRecord,
    heldout_certificate: GATE.CertificateRecord,
) -> list[_MeasurementCell]:
    """The complete paired matrix in fixed order: every trial x every phase, no cell dropped."""
    cells: list[_MeasurementCell] = []
    for trial in AD.TRIALS:
        handoff = handoffs[trial]
        for phase, corpus_root, corpus_manifest, manifest_sha, capsules_sha, certificate in (
            (
                "tuning",
                handoff.corpus_root,
                handoff.corpus_root / "performance_corpus_manifest.json",
                handoff.corpus_manifest_sha256,
                handoff.corpus_sha256,
                tuning_certificate,
            ),
            (
                "held_out",
                Path(revealed["root"]),
                Path(revealed["manifest"]),
                revealed["manifest_sha256"],
                revealed["capsules_sha256"],
                heldout_certificate,
            ),
        ):
            run_id = f"{config.experiment_id}__{trial}__{phase}"
            cells.append(
                _MeasurementCell(
                    trial=trial,
                    phase=phase,
                    handoff=handoff,
                    corpus_root=corpus_root,
                    corpus_manifest=corpus_manifest,
                    corpus_manifest_sha256=manifest_sha,
                    corpus_capsules_sha256=capsules_sha,
                    certificate=certificate,
                    stage=f"measurement:{trial}:{phase}",
                    run_id=run_id,
                    output=config.context.measurement_root / run_id / "campaign_manifest.json",
                )
            )
    return cells


def baseline_lead_prefix(stages: Sequence[ChildStage], phases: Sequence[str]) -> int:
    """How many leading cells must finish before the rest may fan out. Zero when nothing launches.

    THE BASELINE ARM IS THE SAME BYTES IN EVERY CELL OF A CAMPAIGN. It is the frozen functional
    submission; the trials fork CANDIDATES from it and never touch it. So for one corpus member the
    baseline emits one program, and its cycle count on the pinned engine is a constant of the whole
    campaign -- measured once, it is known for every trial that will ask for it. The measurement
    store already returns it rather than re-deriving it, and the serial campaign therefore pays for
    the baseline arm once and reads it twice.

    That saving is FORFEITED the moment the cells fan out: three trials launched together all miss
    the store at the same instant and all three pay, because none of them has finished yet. Letting
    one cell per corpus land first restores it without changing a single measured number -- every
    later cell still asks the engine for its own row, and gets back the number the engine returned
    for those very bytes.

    A PREFIX, deliberately, and not a filtered subset: `run_child_stages` commits on the calling
    thread in the order it is handed, and the checkpoint chain is a strict linear hash chain. A
    prefix split preserves that order exactly, so the recorded matrix is identical to the serial
    one. The declared cell order interleaves the phases, so this prefix is short (one cell per
    corpus, two in the full matrix).
    """
    wanted = {phase for stage, phase in zip(stages, phases) if stage.launch is not None}
    if not wanted:
        return 0
    seen: set[str] = set()
    for index, (stage, phase) in enumerate(zip(stages, phases)):
        if stage.launch is None:
            continue
        seen.add(phase)
        if seen == wanted:
            return index + 1
    return len(stages)


def _measure_cells(
    cells: Sequence[_MeasurementCell],
    config: AD.Config,
    state: AD.Checkpoints,
    *,
    command_runner: AD.CommandRunner,
    workers: int,
) -> list[tuple[str, Path]]:
    """Measure every cell, optionally several at once; each writes a disjoint fresh run directory.

    Adoption, verification and checkpointing all happen on this thread in `cells` order, so the
    recorded matrix is the same whichever cell finishes first.
    """

    def verify(cell: _MeasurementCell, path: Path) -> dict[str, Any]:
        return _verify_measurement_manifest(
            path,
            phase=cell.phase,
            functional_run_id=config.functional_run_id,
            functional_submission_sha256=config.functional_submission_sha256,
            handoff=cell.handoff,
            corpus_manifest_sha256=cell.corpus_manifest_sha256,
            corpus_capsules_sha256=cell.corpus_capsules_sha256,
            certificate_sha256=cell.certificate.sha256,
        )

    stages: list[ChildStage] = []
    stage_phases: list[str] = []
    for cell in cells:
        if state.evidence(cell.stage) is not None:
            continue
        launch: Callable[[], None] | None = None
        if (
            _uncheckpointed_state(
                cell.output.parent, cell.output, label=f"paired measurement {cell.trial}/{cell.phase}"
            )
            == "absent"
        ):
            command = [
                sys.executable,
                "-m",
                "merlin_experiments.phase2.paired_cli",
                "--source-root",
                str(config.context.source_root),
                "--measurement-root",
                str(config.context.measurement_root),
                "--functional-runs-root",
                str(config.context.functional_runs_root),
                "--contract-root",
                str(config.context.contract_root),
                "--functional-run-id",
                config.functional_run_id,
                "--functional-submission-sha256",
                config.functional_submission_sha256,
                "--descriptor",
                str(config.descriptor),
                "--candidate-record",
                str(cell.handoff.record_path),
                "--corpus-root",
                str(cell.corpus_root),
                "--corpus-manifest",
                str(cell.corpus_manifest),
                "--corpus-manifest-sha256",
                cell.corpus_manifest_sha256,
                "--corpus-capsules-sha256",
                cell.corpus_capsules_sha256,
                "--phase",
                cell.phase,
                "--gsim-certificate",
                str(cell.certificate.path),
                "--gsim-certificate-sha256",
                cell.certificate.sha256,
                "--rtl-facts",
                str(config.rtl_facts),
                "--run-id",
                cell.run_id,
                "--timeout",
                str(config.measurement_timeout),
                "--sim-workers",
                str(config.sim_workers),
                "--hardware-counters" if config.hardware_counters else "--no-hardware-counters",
            ]
            for predicate in config.waive_functional_gate or ():
                command += ["--waive-functional-gate", predicate]
            launch = partial(
                _run_checked,
                command_runner,
                command,
                environment=AD.child_environment(config, cell.certificate),
                context=config.context,
            )

        def commit(cell: _MeasurementCell = cell) -> dict[str, Any]:
            saved = verify(cell, cell.output)
            state.append(cell.stage, saved)
            return saved

        stages.append(ChildStage(cell.stage, launch, commit))
        stage_phases.append(cell.phase)
    lead = baseline_lead_prefix(stages, stage_phases) if workers > 1 else len(stages)
    if 0 < lead < len(stages):
        run_child_stages(stages[:lead], workers=workers)
        run_child_stages(stages[lead:], workers=workers)
    else:
        run_child_stages(stages, workers=workers)

    manifests: list[tuple[str, Path]] = []
    for cell in cells:
        saved = state.evidence(cell.stage)
        if saved is None:
            raise ExperimentError(f"paired measurement evidence is missing after it ran: {cell.stage}")
        saved_path = _verify_saved_file(saved, cell.output, label=f"measurement {cell.stage}")
        verify(cell, saved_path)
        manifests.append((cell.trial, Path(saved["path"])))
    return manifests


def run(
    config: AD.Config,
    *,
    command_runner: AD.CommandRunner = AD.subprocess_runner,
    dry_run: bool = False,
    commit_holdout: Callable[..., HOLDOUT.HoldoutPaths] = HOLDOUT.commit_holdout,
    reveal_holdout: Callable[..., Path] = HOLDOUT.reveal_and_materialize,
    heldout_certificate_provider: Callable[[Path, Path, GATE.CertificateRecord], tuple[Path, str]] | None = None,
) -> Path | dict[str, Any]:
    # Read the declared width before anything is launched, so an unreadable declaration fails now
    # rather than 20 hours in, at the phase that would have used it.
    fanout = declared_fanout()
    input_snapshots = None
    chia_launch = (
        None
        if dry_run
        else CHIA.verify_launch_receipt(
            command=config.context.invocation, wrapper=config.context.chia_wrapper, environment=os.environ
        )
    )
    if not dry_run:
        config, input_snapshots = AD.snapshot_contract_inputs(config)
    declaration = AD.preflight(config, heldout_certificate_provider_available=True)
    if dry_run:
        return declaration
    if declaration["status"] != "GO":
        raise ExperimentError("; ".join(declaration["blockers"]))
    assert chia_launch is not None
    orchestration = declaration.get("orchestration") or {}
    if (
        orchestration.get("required_entrypoint_sha256") != chia_launch["wrapper"]["sha256"]
        or orchestration.get("chia_trace_sha256") != chia_launch["chia_trace"]["sha256"]
    ):
        raise ExperimentError("CHIA launch receipt differs from the preflighted orchestration stack")
    config_doc = AD._config_document(config)
    config_sha = AD._sha_bytes(AD._canonical(config_doc))
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    state = AD.Checkpoints(root / "state", config_sha)
    predeclared = state.evidence("predeclared")
    if predeclared is None:
        state.append(
            "predeclared",
            {
                "declaration": declaration,
                "config": config_doc,
                "input_snapshots": input_snapshots,
                "chia_launch_receipt": chia_launch,
            },
        )
    else:
        saved_launch = predeclared.get("chia_launch_receipt")
        if not isinstance(saved_launch, Mapping):
            raise ExperimentError("saved predeclaration lacks its CHIA launch identity")
        AD._verify_resume_chia_identity(saved_launch, chia_launch)
        AD._verify_resume_declaration(predeclared, declaration)
    target = load_target_experiment(config.descriptor, source_root=config.context.source_root)
    functional = FI.inspect_stage_functional_run(
        config.context.functional_runs_root,
        config.functional_run_id,
        config.functional_submission_sha256,
        waive=frozenset(config.waive_functional_gate or ()),
    )
    functional_cohort = FC.functional_grade_cohort_from_run(target, functional, source_root=config.context.source_root)
    functional_cohort = replace(functional_cohort, declined=FC.declined_names(functional))
    tuning_certificate = GATE.load_certificate(config.gsim_certificate, expected_sha256=config.gsim_certificate_sha256)
    functional_certificate = None
    functional_descriptor = None
    if config.functional_gsim_certificate is not None and config.functional_gsim_certificate_sha256 is not None:
        functional_certificate = GATE.load_certificate(
            config.functional_gsim_certificate, expected_sha256=config.functional_gsim_certificate_sha256
        )
        AD._require_same_gsim_build(tuning_certificate, functional_certificate, label="functional GSIM certificate")
        AD._verify_functional_certificate_provenance(functional_certificate, tuning_certificate, functional.digest)
        AD._verify_functional_certificate(functional_certificate, functional_cohort)
        functional_descriptor, _functional_descriptor_binding = AD._functional_qualification_descriptor(
            functional_certificate, functional_cohort
        )
    elif not config.waive_functional_gsim_certificate:
        # Unreachable through preflight, which already refuses. Kept because this function is also
        # the resume entrypoint: a resumed run must not acquire a waiver the predeclaration lacks.
        raise ExperimentError("full public+hidden functional-suite GSIM certificate is required before agent launch")
    environment = AD.child_environment(config, tuning_certificate)
    expected_treatment = declaration.get("agent_treatment")
    if not isinstance(expected_treatment, Mapping):
        raise ExperimentError("predeclaration lacks an exact agent treatment identity")
    public_dir, private_dir = root / "agent_visible", root / "host_private"
    public_dir.mkdir(exist_ok=True)
    holdout = state.evidence("holdout_committed")
    if holdout is None:
        paths = commit_holdout(
            public_dir / "holdout_commitment.json",
            private_dir,
            rtl_facts_path=config.rtl_facts,
            perf_profile_path=config.perf_profile,
            context=config.context.holdout_sources,
            target=target.target,
            candidate_ids=AD.TRIALS,
            count=config.holdout_count,
            generalization_count=config.generalization_count,
            agent_view_root=public_dir,
        )
        holdout = {
            "public": str(paths.public_commitment),
            "public_sha256": AD._sha_file(paths.public_commitment),
            "private": str(paths.host_private_dir),
        }
        state.append("holdout_committed", holdout)
    elif AD._sha_file(Path(holdout["public"])) != holdout["public_sha256"]:
        raise ExperimentError("holdout commitment changed across resume")

    handoffs, trial_evidence = _author_candidates(
        config,
        state,
        target,
        declaration,
        environment=environment,
        expected_treatment=expected_treatment,
        command_runner=command_runner,
        workers=fanout,
    )
    AD._verify_trial_treatments(handoffs, expected_treatment)
    if len({row["agent_evidence_sha256"] for row in trial_evidence}) != 3:
        raise ExperimentError("three independent agent trials need distinct evidence hashes")
    if (
        len({handoff.corpus_manifest_sha256 for handoff in handoffs.values()}) != 1
        or len({handoff.corpus_sha256 for handoff in handoffs.values()}) != 1
    ):
        raise ExperimentError("independent trials did not receive one identical frozen tuning corpus")
    for trial, handoff in handoffs.items():
        workloads = WORKLOAD.derive_frozen_corpus_workloads(
            handoff.corpus_root,
            manifest_sha256=handoff.corpus_manifest_sha256,
            capsules_sha256=handoff.corpus_sha256,
            expected_target=target.target,
        )
        identities = {GATE.workload_sha256(workload) for workload in workloads.values()}
        if identities != set(tuning_certificate.members):
            raise ExperimentError(f"{trial} frozen tuning corpus differs from the exact tuning certificate envelope")

    # Passing target.graded_roots() directly would re-admit the policy-excluded descriptors. Build the
    # exact official public view and recheck the hidden selector against the certificate cohort after the
    # potentially long authoring phase.
    if functional_descriptor is None:
        if getattr(target, "descriptor_sha256", None) != functional_cohort.admission_descriptor_sha256:
            raise ExperimentError("waived functional certificate cannot recover the frozen Phase-1 target descriptor")
        functional_descriptor = Path(config.descriptor).resolve()
    public_roots, hidden_roots, frozen_contract = AD._frozen_functional_regrade_inputs(root, functional_cohort)
    regrades = {}
    for trial, handoff in handoffs.items():
        saved = state.evidence(f"functional_regrade:{trial}")
        grade_dir = root / "functional_regrades" / trial
        if saved is None:
            manifest_path = grade_dir / "run_manifest.yaml"
            if _uncheckpointed_state(grade_dir, manifest_path, label=f"functional regrade {trial}") == "absent":
                _prepare_regrade(grade_dir, handoff)
                command = [
                    sys.executable,
                    "-m",
                    "merlin_experiments.phase1.feedback.formal",
                    "--descriptor",
                    str(functional_descriptor),
                    "--repo",
                    str(config.context.source_root),
                    "--run-dir",
                    str(grade_dir),
                    "--arm",
                    "merlin_assisted",
                    "--model",
                    config.model,
                    "--capsules",
                    public_roots,
                    "--hidden-capsules",
                    hidden_roots,
                    "--contract",
                    str(frozen_contract),
                ]
                # The certificate is used here only to pin the GSIM build the regrade must execute on.
                # When it is waived, the TUNING certificate pins the same build -- that equality is
                # what `_require_same_gsim_build` asserts whenever both exist -- so the regrade runs
                # on the identical engine either way.
                regrade_environment = AD.child_environment(config, functional_certificate or tuning_certificate)
                regrade_environment["MERLIN_TARGET_EXPERIMENT"] = str(functional_descriptor)
                _run_checked(command_runner, command, environment=regrade_environment, context=config.context)
            saved = _verify_regrade(grade_dir, handoff)
            state.append(f"functional_regrade:{trial}", saved)
        _verify_saved_file(saved, grade_dir / "run_manifest.yaml", label=f"functional regrade {trial}")
        regrades[trial] = _verify_regrade(grade_dir, handoff)

    revealed = state.evidence("holdout_revealed")
    if revealed is None:
        manifest = reveal_holdout(
            Path(holdout["public"]),
            Path(holdout["private"]),
            root / "held_out_corpus",
            candidate_seals={trial: handoffs[trial].record_path for trial in AD.TRIALS},
            context=config.context.holdout_sources,
        )
        document = json.loads(manifest.read_text(encoding="utf-8"))
        revealed = {
            "root": str(manifest.parent),
            "manifest": str(manifest),
            "manifest_sha256": AD._sha_file(manifest),
            "capsules_sha256": document["corpus"]["sha256"],
        }
        state.append("holdout_revealed", revealed)
    elif AD._sha_file(Path(revealed["manifest"])) != revealed["manifest_sha256"]:
        raise ExperimentError("held-out reveal changed across resume")

    extension = state.evidence("heldout_gsim_certificate")
    if extension is None:
        qualification_root = root / "heldout_gsim_qualification"
        if qualification_root.exists() or qualification_root.is_symlink():
            extension_path, extension_sha = HQUAL.load_completed_qualification(
                qualification_root,
                tuning=tuning_certificate,
                reveal_manifest_sha256=revealed["manifest_sha256"],
                reveal_corpus_sha256=revealed["capsules_sha256"],
                functional_base_sha256=functional.digest,
                gsim_max_cycles=config.gsim_max_cycles,
            )
        elif heldout_certificate_provider is not None:
            extension_path, extension_sha = heldout_certificate_provider(
                Path(revealed["manifest"]), qualification_root, tuning_certificate
            )
        else:
            extension_path, extension_sha = _qualify_heldout_with_config(
                Path(revealed["manifest"]),
                qualification_root,
                tuning_certificate,
                functional_base=functional.submission_dir,
                functional_base_sha256=functional.digest,
                reveal_manifest_sha256=revealed["manifest_sha256"],
                reveal_corpus_sha256=revealed["capsules_sha256"],
                config=config,
                target=target,
            )
        try:
            Path(extension_path).resolve(strict=True).relative_to(qualification_root.resolve())
        except ValueError as exc:
            raise ExperimentError("held-out certificate provider wrote outside its fresh host root") from exc
        heldout_certificate = GATE.load_certificate(extension_path, expected_sha256=extension_sha)
        if heldout_certificate.target != target.target:
            raise ExperimentError("held-out GSIM extension certificate names a different target")
        coverage = _verify_extension_certificate(
            tuning_certificate,
            heldout_certificate,
            Path(revealed["manifest"]),
            manifest_sha256=revealed["manifest_sha256"],
            corpus_sha256=revealed["capsules_sha256"],
            target=target.target,
        )
        extension = {
            "path": str(Path(extension_path).resolve()),
            "sha256": extension_sha,
            "certificate": heldout_certificate.to_dict(),
            "coverage": coverage,
            "produced_after_checkpoint": state.load()[-1]["sha256"],
        }
        state.append("heldout_gsim_certificate", extension)
    heldout_certificate = GATE.load_certificate(extension["path"], expected_sha256=extension["sha256"])
    coverage = _verify_extension_certificate(
        tuning_certificate,
        heldout_certificate,
        Path(revealed["manifest"]),
        manifest_sha256=revealed["manifest_sha256"],
        corpus_sha256=revealed["capsules_sha256"],
        target=target.target,
    )
    if coverage != extension.get("coverage"):
        raise ExperimentError("held-out certificate coverage changed across resume")

    # Declare the statistics denominator before any performance subprocess is launched.
    stats_saved = state.evidence("statistics_predeclared")
    if stats_saved is None:
        capsules: set[tuple[str, str]] = set()
        for trial, handoff in handoffs.items():
            for phase, corpus_args, certificate in (
                (
                    "tuning",
                    (
                        handoff.corpus_root,
                        handoff.corpus_manifest_sha256,
                        handoff.corpus_sha256,
                        handoff.corpus_root / "performance_corpus_manifest.json",
                    ),
                    tuning_certificate,
                ),
                (
                    "held_out",
                    (
                        Path(revealed["root"]),
                        revealed["manifest_sha256"],
                        revealed["capsules_sha256"],
                        Path(revealed["manifest"]),
                    ),
                    heldout_certificate,
                ),
            ):
                inputs = PI.load_paired_inputs(
                    handoff.record_path,
                    config.functional_run_id,
                    config.functional_submission_sha256,
                    target,
                    corpus_root=corpus_args[0],
                    corpus_manifest_sha256=corpus_args[1],
                    corpus_capsules_sha256=corpus_args[2],
                    corpus_manifest=corpus_args[3],
                    phase=phase,
                    gsim_certificate=certificate.path,
                    gsim_certificate_sha256=certificate.sha256,
                    waive_functional_gate=tuple(config.waive_functional_gate or ()),
                    functional_runs_root=config.context.functional_runs_root,
                )
                plan = PME.build_measurement_plan(inputs)
                capsules.update((f"{phase}:{spec.family}", spec.capsule) for spec in plan.schedule)
        declaration_stats = STATS.predeclare(
            trials=_statistics_trials(trial_evidence),
            capsules=[{"family": family, "capsule": capsule} for family, capsule in sorted(capsules)],
            replicates=AD.REPLICATES,
            primary_simulator="gsim",
        )
        stats_path = root / "statistics_predeclaration.json"
        stats_payload = AD._canonical(declaration_stats)
        if stats_path.exists():
            if not stats_path.is_file() or stats_path.read_bytes() != stats_payload:
                raise ExperimentError("uncheckpointed statistics predeclaration differs from plan")
        else:
            stats_path.write_bytes(stats_payload)
            stats_path.chmod(0o444)
        stats_saved = {"path": str(stats_path.resolve()), "sha256": AD._sha_file(stats_path)}
        state.append("statistics_predeclared", stats_saved)
    stats_path = _verify_saved_file(
        stats_saved, root / "statistics_predeclaration.json", label="statistics predeclaration"
    )
    declaration_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    measurement_cells = _measurement_cells(
        config, handoffs, revealed, tuning_certificate=tuning_certificate, heldout_certificate=heldout_certificate
    )
    measurement_manifests = _measure_cells(
        measurement_cells,
        config,
        state,
        command_runner=command_runner,
        workers=fanout,
    )

    pinned_measurements = [
        {"trial": trial, "path": str(path), "sha256": state.evidence(cell.stage)["sha256"]}
        for cell, (trial, path) in zip(measurement_cells, measurement_manifests, strict=True)
    ]
    all_rows = [
        row
        for receipt in pinned_measurements
        for row in ME.read_statistics_rows(
            Path(receipt["path"]), trial=receipt["trial"], manifest_sha256=receipt["sha256"]
        )
    ]
    result = STATS.evaluate(declaration_stats, all_rows, trial_evidence=trial_evidence)
    if result.get("status") != "admitted":
        raise ExperimentError(f"all-cell performance statistic refused: {result.get('issues')}")
    final = {
        "schema": AD.SCHEMA,
        "status": "GO",
        "declaration": declaration,
        "config_sha256": config_sha,
        "holdout": revealed,
        "chia_launch_receipt": chia_launch,
        "heldout_gsim_certificate": extension,
        "trials": trial_evidence,
        "functional_regrades": regrades,
        "agent_telemetry": {
            trial: {
                "preflight_sha256": handoff.telemetry_evidence["preflight_sha256"],
                "raw_trace": handoff.telemetry_evidence["artifacts"]["combined_raw"],
                "aet_trajectory": handoff.telemetry_evidence["artifacts"]["trajectory"],
                "aet_metrics_log": handoff.telemetry_evidence["artifacts"]["aet_metrics_log"],
                "cost_time_toolcalls": handoff.telemetry_evidence["artifacts"]["cost_time_toolcalls"],
                "activity_share": handoff.telemetry_evidence["artifacts"]["activity_share"],
                "tool_call_count": handoff.telemetry_evidence["tool_call_count"],
                "subagent_tool_calls_tracked": handoff.telemetry_evidence["subagent_tool_calls_tracked"],
            }
            for trial, handoff in handoffs.items()
        },
        "measurement_manifests": pinned_measurements,
        "statistics_predeclaration_sha256": stats_saved["sha256"],
        "statistics": result,
        "selection": "all_three_trials_all_predeclared_cells_no_best_of_no_drop",
    }
    return _seal_final(root, final)
