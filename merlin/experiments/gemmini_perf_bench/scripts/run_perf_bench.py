#!/usr/bin/env python3
"""Run the Gemmini performance corpus on one frozen, functionally complete Arm-4 compiler.

The runner deliberately has no "latest submission" discovery and no alternate learned/compiler arm.
The caller supplies the exact functional run ID, submission SHA-256, and sealed candidate handoff.
The candidate is mounted read-only in a credential-free, mount-scoped bwrap and checked against its
functional fork before and after the corpus.  An explicit measurement-only smoke campaign is GO only
when every
expected Arm-4 capsule replica passes its L2 correctness screen and its L3 certification reports
positive cycles.  That does not by itself establish the capsule's performance hypothesis.  Formal
claim mode separately preflights the frozen family declaration and applies the fixed
quantitative decision procedure that declaration itself names to the sealed exact-cell rows.  Measurement-smoke mode never runs or promotes
that claim decision.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import inspect
import json
import multiprocessing
import stat
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _pbcommon as PB
import perf_campaign as PC
from merlin.benchharness import runs_root as _runs_root
from merlin.perf import claim_reach as CLAIM
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.target_experiment import load_target_experiment


_FUNCTIONAL_RUNS = _runs_root(PB.TARGET, "capsule-bench")
_DESCRIPTOR = (PB.REPO / "merlin/experiments/capsule_bench/targets" / PB.TARGET
               / "target_experiment.yaml")


@dataclass(frozen=True)
class MeasurementCandidate:
    handoff: Any
    package: Path
    package_sha256: str
    corpus: PC.FrozenPerformanceCorpus
    functional_base: Path
    contract_root: Path
    contract_sha256: str
    sentinel: PC.FullModelSentinel


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement_mode(mode: str, preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Make claim intent explicit and require a READY frozen-declaration preflight."""
    if mode == "measurement-smoke":
        return {
            "experiment_mode": "measurement_smoke_only",
            "claim_launch_status": "NOT_REQUESTED",
            "claim_launch_blocker": PC.SMOKE_CLAIM_NONCLAIM,
            "claim_preflight": None,
        }
    if mode != "claim":
        raise PC.CampaignGateError(f"unsupported performance measurement mode: {mode!r}")
    reasons = preflight.get("refusal_reasons") if isinstance(preflight, Mapping) else None
    family = preflight.get("family") if isinstance(preflight, Mapping) else None
    # The family is NOT checked against a literal here.  Which family may make a claim is decided by
    # whether its own declaration names a decision procedure, which _declared_analyzer resolved
    # before this preflight ran; a second, hardcoded opinion here would refuse families that the
    # declaration already admitted.
    if (not isinstance(preflight, Mapping) or preflight.get("status") != "READY"
            or not preflight.get("family") or not preflight.get("claim")):
        detail = "; ".join(str(reason) for reason in reasons or []) or "preflight is absent"
        raise PC.CampaignGateError(
            f"claim-bearing performance launch is NO-GO: {family or 'declared family'} "
            f"preflight refused: {detail}")
    return {
        "experiment_mode": "formal_claim",
        "claim_launch_status": "GO",
        "claim_launch_blocker": None,
        "claim_preflight": dict(preflight),
    }


def _declared_analyzer(capsules: Sequence[PC.PerformanceCapsule]) -> tuple[Any, Any]:
    """Resolve the decision procedure the frozen family declares FOR ITSELF.

    Nothing here knows a family name.  The module, function and version come out of the capsules'
    own ``performance.acceptance.analyzer``, so a family that declares an analyzer is dispatched
    with no edit to this runner, and one that declares none is REFUSED by name rather than decided
    by some other family's procedure.
    """
    identities: set[Any] = set()
    for capsule in capsules:
        descriptor = capsule.descriptor if isinstance(capsule.descriptor, Mapping) else {}
        performance = descriptor.get("performance")
        if not isinstance(performance, Mapping):
            raise PC.CampaignGateError(
                f"frozen capsule {capsule.capsule!r} carries no performance declaration")
        try:
            identity = CLAIM.analyzer_identity(performance)
        except ValueError as exc:
            raise PC.CampaignGateError(
                f"frozen family {performance.get('family')!r} declares an unusable "
                f"acceptance.analyzer: {exc}") from exc
        if identity is None:
            raise PC.CampaignGateError(
                f"frozen family {performance.get('family')!r} declares no acceptance.analyzer, so "
                "nothing computes its verdict from its rows; a claim-bearing launch is refused "
                "rather than decided by another family's analyzer")
        identities.add(identity)
    if len(identities) != 1:
        raise PC.CampaignGateError(
            f"the frozen performance corpus declares {len(identities)} claim analyzers; exactly "
            "one is required")
    identity = identities.pop()
    try:
        module = importlib.import_module(identity.module)
    except Exception as exc:
        raise PC.CampaignGateError(
            f"declared claim analyzer {identity.declared!r} is unavailable: {exc}") from exc
    return identity, module


def _preflight_entry(module: Any) -> Any:
    """The module's preflight entry point, found by the shape every claim analyzer publishes."""
    names = sorted(name for name in dir(module)
                   if name.startswith("preflight_") and callable(getattr(module, name, None)))
    if len(names) != 1:
        raise PC.CampaignGateError(
            f"declared claim analyzer module {module.__name__!r} publishes {len(names)} preflight "
            "entry points; exactly one is required")
    return getattr(module, names[0])


def _analyze_entry(module: Any, identity: Any) -> Any:
    """The analyze entry point the declaration NAMES, never one guessed from the module."""
    entry = getattr(module, identity.function, None)
    if not callable(entry):
        raise PC.CampaignGateError(
            f"declared claim analyzer {identity.declared!r} names no callable "
            f"{identity.function!r} in {identity.module!r}")
    return entry


def _run_fact_providers(replicas: int | None, target: str | None) -> dict[str, Any]:
    """Facts this RUN can supply, keyed by the parameter name an analyzer uses to ask for one.

    The providers are lazy so a fact is derived only when the analyzer's own signature declares it:
    a family needing no hardware counters never reads a header, and one that needs them gets them
    DERIVED from the target rather than assumed.
    """
    def replicates() -> Any:
        return [f"r{index:03d}" for index in range(replicas)] if replicas else None

    def counters() -> Any:
        if not target:
            return None
        from merlin.perf.hw_counters import counters_for_target
        derived = counters_for_target(target)
        return derived.get("counters") if derived.get("status") == "derived" else None

    return {"replicates": replicates, "counters": counters}


def _analyzer_kwargs(entry: Any, providers: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Only the run facts ``entry``'s OWN signature asks for; an underivable one fails closed."""
    parameters = inspect.signature(entry).parameters
    kwargs: dict[str, Any] = {}
    for name, provide in providers.items():
        if name not in parameters:
            continue
        value = provide()
        if value is None:
            raise PC.CampaignGateError(
                f"{label} declares a {name!r} parameter and this run could not derive one; a "
                "claim decided without it would rest on an assumed fact")
        kwargs[name] = value
    return kwargs


def _boundary_record(module: Any, identity: Any, family: str | None,
                     decision: Mapping[str, Any] | None) -> dict[str, Any]:
    """The campaign-manifest boundary, written by the analyzer itself where it publishes one."""
    own = getattr(module, "decision_boundary", None)
    if callable(own):
        record = dict(own(decision))
        record["module"] = identity.module
        return record
    bridge = f"{identity.function}(frozen_descriptors,sealed_result_rows)"
    if decision is None:
        return {
            "module": identity.module,
            "identity_bridge": bridge,
            "promotion_integration": "integrated",
            "promotion_status": "PENDING",
            "reason": (f"the predeclared {family} analyzer has not yet consumed the sealed "
                       "result rows"),
        }
    established = decision.get("status") == "ESTABLISHED"
    return {
        "module": identity.module,
        "identity_bridge": bridge,
        "promotion_integration": "integrated",
        "promotion_status": "PROMOTED" if established else "BLOCKED",
        "reason": (f"the predeclared {family} quantitative decision was established" if established
                   else f"the predeclared {family} quantitative decision was refuted"),
    }


def _pk_preflight(capsules: Sequence[PC.PerformanceCapsule], replicas: int,
                  expected: Sequence[PC.PerfCellIdentity], *,
                  target: str | None = None) -> dict[str, Any]:
    """Preflight only the authenticated frozen descriptor objects and exact cell schedule."""
    identity, module = _declared_analyzer(capsules)
    entry = _preflight_entry(module)
    kwargs = _analyzer_kwargs(entry, _run_fact_providers(replicas, target),
                              label=f"{identity.module}.{entry.__name__}")
    preflight = entry([capsule.descriptor for capsule in capsules], **kwargs)
    if preflight.get("status") != "READY":
        return preflight
    declaration = preflight.get("declaration")
    contract = ((declaration or {}).get("replicates") or {})
    # The admissible replica count comes from the declaration's OWN replicate contract, so a family
    # that predeclares an exact cohort is held to it and one that predeclares a floor is held to
    # that floor.  A family declaring neither is refused: an unstated cohort is not an open one.
    exact = contract.get("exact_count")
    minimum = contract.get("minimum_count")
    if isinstance(exact, int) and not isinstance(exact, bool):
        if replicas != exact:
            raise PC.CampaignGateError(
                f"claim-bearing {preflight.get('family')} launch requires exactly the predeclared "
                f"{exact} replicates")
    elif isinstance(minimum, int) and not isinstance(minimum, bool):
        if replicas < minimum:
            raise PC.CampaignGateError(
                f"claim-bearing {preflight.get('family')} launch requires at least the predeclared "
                f"{minimum} replicates")
    else:
        raise PC.CampaignGateError(
            f"frozen {preflight.get('family')} declaration states no replicate contract, so this "
            "run's replica count cannot be admitted")
    declared_cells = tuple({key: row[key] for key in (
        "family", "capsule", "simulator", "replicate")}
        for row in preflight.get("expected_identities") or [])
    actual_cells = tuple(identity.as_dict() for identity in expected)
    if declared_cells != actual_cells:
        raise PC.CampaignGateError(
            "PK preflight identities differ from the frozen exact-cell schedule")
    return preflight


def _claim_decision(capsules: Sequence[PC.PerformanceCapsule], rows: Sequence[Mapping], *,
                    replicates: int | None = None, target: str | None = None) -> dict:
    """Return the declared analyzer's result verbatim for sealing into the campaign manifest."""
    identity, module = _declared_analyzer(capsules)
    entry = _analyze_entry(module, identity)
    kwargs = _analyzer_kwargs(entry, _run_fact_providers(replicates, target),
                              label=f"{identity.module}.{identity.function}")
    return entry([capsule.descriptor for capsule in capsules], rows, **kwargs)


def load_measurement_candidate(
        record_path: Path, functional: PC.FunctionalRun,
        target_experiment) -> MeasurementCandidate:
    """Admit the sealed authoring handoff; never rediscover candidate or corpus bytes."""
    raw = Path(record_path)
    if raw.is_symlink() or not raw.is_file() or raw.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PC.CampaignGateError("candidate record is absent, linked, or writable")
    record_path = raw.resolve(strict=True)
    try:
        import perf_agent_stage as agent_stage  # lazy: the measurement CLI owns this dependency
        handoff = agent_stage.verify_candidate_handoff(
            record_path, verify_authoring_tools=False, target_experiment=target_experiment)
    except Exception as exc:
        raise PC.CampaignGateError(f"performance candidate record is not consumable: {exc}") from exc

    if (handoff.functional_run_id != functional.run_id
            or handoff.functional_submission_sha256 != functional.digest):
        raise PC.CampaignGateError(
            "candidate record names a different functional run or submission digest")
    expected_descriptor = Path(target_experiment.path).resolve(strict=True)
    if (handoff.target_descriptor.resolve(strict=True) != expected_descriptor
            or _file_sha256(expected_descriptor) != handoff.target_descriptor_sha256):
        raise PC.CampaignGateError(
            "candidate record target descriptor does not match the measurement descriptor")

    package = handoff.candidate_path.resolve(strict=True)
    package_sha = handoff.candidate_sha256
    if (package == functional.submission_dir
            or package == handoff.functional_base_path.resolve(strict=True)
            or handoff.candidate_initial_sha256 != functional.digest):
        raise PC.CampaignGateError(
            "candidate is not a separate exact fork of the functional submission")
    expected_bundle_manifest = (
        Path(functional.bundle_input_snapshot["path"]) / "snapshot.json").resolve(strict=True)
    if (handoff.functional_bundle_snapshot_sha256 !=
            functional.bundle_input_snapshot["content_sha256"]
            or handoff.functional_bundle_manifest.resolve(strict=True) != expected_bundle_manifest
            or _file_sha256(expected_bundle_manifest) !=
            handoff.functional_bundle_manifest_sha256):
        raise PC.CampaignGateError(
            "candidate record names a different functional bundle-input snapshot")
    recorded_host = handoff.host_lane
    for key, value in functional.model_host_lane_snapshot.items():
        if recorded_host.get(key) != value:
            raise PC.CampaignGateError(
                "candidate record host lane differs from the functional run snapshot")

    corpus_root = handoff.corpus_root.resolve(strict=True)
    if handoff.corpus_manifest.resolve(strict=True) != (
            corpus_root / "performance_corpus_manifest.json"):
        raise PC.CampaignGateError("candidate record corpus manifest escapes its frozen corpus")
    corpus = PC.load_frozen_performance_corpus(
        corpus_root,
        manifest_sha256=handoff.corpus_manifest_sha256,
        capsules_sha256=handoff.corpus_sha256,
        expected_target=target_experiment.target,
    )
    exact_cells = tuple(identity.as_dict() for identity in PC.expected_perf_cells(
        corpus.capsules, handoff.replicates))
    if tuple(handoff.expected_cells) != exact_cells:
        raise PC.CampaignGateError(
            "candidate prompt cells differ from the frozen corpus and replicate count")
    if {str(row.get("family")) for row in handoff.families} != {
            capsule.family for capsule in corpus.capsules}:
        raise PC.CampaignGateError(
            "candidate prompt families differ from the frozen performance corpus")

    contract_destination = (PB.REPO / "merlin/contract").absolute()
    contract_root = PC.frozen_bundle_grant_path(
        handoff.functional_bundle_manifest, handoff.functional_bundle_manifest_sha256,
        contract_destination, label="measurement contract")
    expected_contract = (Path(functional.bundle_input_snapshot["path"])
                         / "repo/merlin/contract").resolve(strict=True)
    if contract_root != expected_contract:
        raise PC.CampaignGateError(
            "candidate handoff contract does not resolve inside the functional snapshot")
    contract_sha256 = str(PC._exact_tree_record(contract_root)["sha256"])

    sentinel_record = handoff.e2e_sentinel
    sentinel_source = Path(str(sentinel_record.get("frozen_source_path") or "")).resolve(strict=True)
    try:
        sentinel_source.relative_to(
            (Path(functional.bundle_input_snapshot["path"]) / "repo").resolve(strict=True))
    except ValueError as exc:
        raise PC.CampaignGateError(
            "candidate E2E sentinel is outside the functional snapshot") from exc
    sentinel_tree = PC._exact_tree_record(sentinel_source)
    if (sentinel_record.get("capsule_sha256") != sentinel_tree["sha256"]
            or set(sentinel_record.get("required_lanes") or []) != {
                "on_mesh", "scalar_rvv_lane"}
            or not {"L2", "L3"}.issubset(sentinel_record.get("required_tiers") or [])):
        raise PC.CampaignGateError("candidate E2E sentinel identity is incomplete or changed")
    sentinel = PC.FullModelSentinel(
        str(sentinel_record.get("capsule") or ""), sentinel_source,
        PC._mapping_file(sentinel_source / "capsule.yaml", yaml_file=True),
        str(sentinel_tree["sha256"]), int(sentinel_tree["n_files"]),
        int(sentinel_tree["n_bytes"]))
    if sentinel.descriptor.get("name") != sentinel.capsule:
        raise PC.CampaignGateError("candidate E2E sentinel descriptor/name identity changed")
    return MeasurementCandidate(
        handoff, package, package_sha, corpus, handoff.functional_base_path.resolve(strict=True),
        contract_root, contract_sha256, sentinel)


def _candidate_stage_record(candidate: MeasurementCandidate) -> dict[str, Any]:
    return PC.candidate_handoff_record(candidate.handoff)


def _result_rows(identity_base: tuple[str, str, str], *, error: str | None = None,
                 traceback_text: str | None = None) -> list[dict]:
    family, capsule, replicate = identity_base
    rows: list[dict] = []
    for simulator, tier, purpose, citable in (
            ("spike", "L2", "correctness_screen", False),
            ("verilator", "L3", "performance_certification", True)):
        row = {
            "identity": {"family": family, "capsule": capsule,
                         "simulator": simulator, "replicate": replicate},
            "approach": "arm4",
            "tier": tier,
            "purpose": purpose,
            "citable": citable,
            "correct": False,
            "cycles": None,
        }
        if error is not None:
            row["error"] = error
            row["traceback"] = traceback_text
        rows.append(row)
    return rows


def _verify_frozen_contract(contract_root: Path, expected_sha256: str) -> None:
    observed = PC._exact_tree_record(contract_root)["sha256"]
    if observed != expected_sha256:
        raise PC.CampaignGateError(
            "frozen measurement contract bytes differ from the functional handoff")


def run_arm4(package: Path, member: PC.PerformanceCapsule, replicate: str,
             capsule_runs: Path, timeout: int, target: str, contract_root: Path,
             contract_sha256: str) -> list[dict]:
    """Run one frozen generated capsule; Spike screens and Verilator certifies the same replica."""
    rows = _result_rows((member.family, member.capsule, replicate))
    _verify_frozen_contract(contract_root, contract_sha256)
    capsule = CR.load_capsule(member.source_dir, contract=str(contract_root))
    capsule = dict(capsule)
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    try:
        grade = CR.run_capsule(
            capsule,
            str(package),
            runs_root=str(capsule_runs),
            run_id=f"arm4_{member.family}_{member.capsule}_{replicate}",
            contract=str(contract_root),
            oracle_adapters=CR.default_adapters(),
            timeout=timeout,
            target=target,
            workers=1,
        )
    except Exception as exc:  # one failed cell is recorded; the global completion gate still refuses
        return _result_rows(
            (member.family, member.capsule, replicate),
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
            traceback_text=traceback.format_exc()[-1600:])
    tiers = grade.get("tiers") or {}
    grade_status = grade.get("status")
    numeric = grade.get("numeric")
    numeric_status = numeric.get("status") if isinstance(numeric, dict) else numeric
    grade_failure = grade.get("failure")
    for row in rows:
        tier = str(row["tier"])
        tier_result = tiers.get(tier) or {}
        status = tier_result.get("status") if isinstance(tier_result, dict) else tier_result
        cycles = tier_result.get("cycles") if isinstance(tier_result, dict) else None
        row["tier_status"] = status
        row["correct"] = (status == "pass" and grade_status == "pass"
                          and numeric_status == "pass" and not grade_failure)
        # Spike is a correctness screen.  Its cycle count is deliberately not copied into the result
        # row, so downstream reporting cannot accidentally cite it as a performance measurement.
        row["cycles"] = cycles if row["citable"] else None
        row["grade_status"] = grade_status
        row["numeric_status"] = numeric_status
    if grade_failure:
        failure = {key: grade_failure.get(key) for key in ("plane", "category", "detail")}
        for row in rows:
            row["failure"] = failure
    _verify_frozen_contract(contract_root, contract_sha256)
    return rows


def run_full_model_admission(
        package: Path, sentinel: PC.FullModelSentinel, workspace: Path, timeout: int,
        target_experiment, contract_root: Path, contract_sha256: str) -> dict:
    """Require one frozen mixed-lane whole model at the target's citable RTL tier before perf cells.

    The tier is DERIVED, never written down. This gate used to force ``["L2", "L3"]`` and demand both
    be ``pass``; a whole model emits exactly ONE execution tier (``CR.model_citable_rtl_tier`` -- the
    last declared tier the target's manifest counts as RTL), so on a target whose RTL tiers begin at
    L3 the L2 clause read ``None`` on a flawless run and the campaign refused every candidate before
    measuring a single cell. Fail closed: a declaration with no citable RTL tier is refused outright,
    and a run that reached its RTL tier without executing on the accelerator still fails (the tier
    itself reports ``skipped``/``fail``/``unavailable`` in exactly those cases).
    """
    _verify_frozen_contract(contract_root, contract_sha256)
    capsule = dict(CR.load_capsule(sentinel.source_dir, contract=str(contract_root)))
    declared = [str(value) for value in (capsule.get("required_oracle_tiers")
                                        or sentinel.descriptor.get("required_oracle_tiers") or [])]
    cert_tier = CR.model_citable_rtl_tier(declared, target_experiment.target)
    if cert_tier is None:
        raise PC.CampaignGateError(
            f"full-model sentinel {sentinel.capsule!r} declares no tier "
            f"{target_experiment.target!r} counts as RTL (declared={declared}); there is no citable "
            "whole-model verdict to admit a performance campaign on")
    capsule["required_oracle_tiers"] = [cert_tier]
    capsule_runs = workspace / "capsule_runs"
    capsule_runs.mkdir()
    policy = PC.package_sandbox_policy(target_experiment, workspace, package)
    with PC.boxed_entrypoints(policy):
        grade = CR.run_capsule(
            capsule, str(package), runs_root=str(capsule_runs),
            run_id=f"arm4_full_model_{sentinel.capsule}", contract=str(contract_root),
            oracle_adapters=CR.default_adapters(), timeout=timeout,
            target=target_experiment.target, workers=1)
    tiers = grade.get("tiers") or {}

    def status(tier: str) -> object:
        value = tiers.get(tier)
        return value.get("status") if isinstance(value, dict) else value

    numeric = grade.get("numeric")
    numeric_status = numeric.get("status") if isinstance(numeric, dict) else numeric
    passed = (grade.get("status") == "pass" and numeric_status == "pass"
              and status(cert_tier) == "pass")
    _verify_frozen_contract(contract_root, contract_sha256)
    return {
        "capsule": sentinel.capsule,
        "kind": "model",
        "source": "functional_bundle_input_snapshot_v2",
        "source_sha256": sentinel.source_sha256,
        "n_files": sentinel.n_files,
        "n_bytes": sentinel.n_bytes,
        "lanes_required": ["on_mesh", "scalar_rvv_lane"],
        "required_tiers": [cert_tier],
        "tier_status": {cert_tier: status(cert_tier)},
        # The INPUTS to the tier choice, recorded so a reader (and the reporting gate) can re-derive
        # it without this target's capability manifest in hand -- the fact is the manifest's, the
        # arithmetic is auditable, and neither is a literal anyone had to write down.
        "tier_derivation": {
            "declared": declared,
            "rtl_tiers": sorted(CR.rtl_tiers_of(target_experiment.target)),
            "cert_tier": cert_tier,
            "rule": "the last declared tier the target's capability manifest counts as RTL",
        },
        "grade_status": grade.get("status"),
        "numeric_status": numeric_status,
        "passed": passed,
        "cycles_recorded": False,
        "role": "correctness_admission_not_performance_claim",
    }


def _run_cell_task(
        task: tuple[Path, PC.PerformanceCapsule, str, Path, int, Any, Path, str]) -> list[dict]:
    """Run one pair in an isolated process/workspace; failures remain exact result rows."""
    (package, member, replicate, cell_workspace, timeout, target_experiment, contract_root,
     contract_sha256) = task
    try:
        capsule_runs = cell_workspace / "capsule_runs"
        policy = PC.package_sandbox_policy(target_experiment, cell_workspace, package)
        with PC.boxed_entrypoints(policy):
            return run_arm4(
                package, member, replicate, capsule_runs, timeout, target_experiment.target,
                contract_root, contract_sha256)
    except Exception as exc:
        return _result_rows(
            (member.family, member.capsule, replicate),
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
            traceback_text=traceback.format_exc()[-1600:])


def _write_json(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-run-id", required=True,
                        help="exact completed Arm-4 functional run directory name")
    parser.add_argument("--functional-submission-sha256", required=True,
                        help="exact frozen functional submission SHA-256")
    parser.add_argument("--candidate-record", type=Path, required=True,
                        help="sealed performance_candidate.json produced by perf_agent_stage.py")
    parser.add_argument(
        "--replicates", type=int,
        help=("exact replica count for both L2 and L3; defaults to the sealed mode-specific "
              "candidate count"))
    parser.add_argument("--workers", type=int, default=1,
                        help="independent capsule/replicate processes (deterministic output order)")
    parser.add_argument("--approach", choices=("arm4",), default="arm4",
                        help="only the Arm-4 compiler lane is admitted in this campaign")
    parser.add_argument(
        "--mode", choices=("claim", "measurement-smoke"), default="claim",
        help="formal claim launch is fail-closed; smoke mode measures tooling/cells only")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--run-id", default="perf_0001")
    args = parser.parse_args(argv)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise PC.CampaignGateError("performance run id must be a simple directory name")
    if args.timeout <= 0:
        raise PC.CampaignGateError("performance cell timeout must be positive")
    if args.replicates is not None and args.replicates <= 0:
        raise PC.CampaignGateError("performance replicas must be positive")
    if args.workers <= 0:
        raise PC.CampaignGateError("performance workers must be positive")

    functional = PC.inspect_functional_run(
        _FUNCTIONAL_RUNS, args.functional_run_id, args.functional_submission_sha256)
    target_experiment = load_target_experiment(_DESCRIPTOR)
    candidate = load_measurement_candidate(
        args.candidate_record, functional, target_experiment)
    required_replicates = (candidate.handoff.replicates if args.mode == "claim"
                           else candidate.handoff.smoke_replicates)
    if args.replicates is None:
        args.replicates = required_replicates
    if args.replicates != required_replicates:
        raise PC.CampaignGateError(
            f"{args.mode} --replicates differs from the sealed candidate prompt")
    frozen_corpus = candidate.corpus
    sentinel = candidate.sentinel
    expected = PC.expected_perf_cells(frozen_corpus.capsules, args.replicates)
    claim_identity, claim_module = _declared_analyzer(frozen_corpus.capsules)
    preflight = (_pk_preflight(frozen_corpus.capsules, args.replicates, expected,
                               target=target_experiment.target)
                 if args.mode == "claim" else None)
    claim_family = str(preflight.get("family")) if isinstance(preflight, Mapping) else None
    if args.mode == "claim" and preflight != candidate.handoff.formal_claim:
        raise PC.CampaignGateError(
            f"runner {claim_family} preflight differs from the verified candidate handoff")
    if args.mode == "claim":
        # A family predeclaring its replicate identities is held to them; one whose declaration
        # says the run authors them is held to the schedule its own preflight returned.
        declared_identities = preflight["declaration"]["replicates"].get("identities")
        if declared_identities is None:
            declared_identities = preflight.get("replicates") or []
        if tuple(declared_identities) != tuple(candidate.handoff.formal_replicate_identities):
            raise PC.CampaignGateError(
                "formal replicate identities differ from the verified candidate handoff")
    mode_record = measurement_mode(args.mode, preflight)
    out_dir = PB.RUNS / args.run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise PC.CampaignGateError(
            f"performance run directory already exists; choose a fresh --run-id: {out_dir}")

    out_dir.mkdir(parents=True)
    snapshot = candidate.package
    fork = PC.functional_fork(functional)
    functional_base = candidate.functional_base
    before = PC.check_fork(fork, functional_base)
    if before.ok is not True:
        raise PC.CampaignGateError(f"functional fork does not hold before performance: {before.reason}")
    fork_record = fork.to_dict()
    fork_record.update({"functional_run_id": functional.run_id,
                        "functional_submission_sha256": functional.digest,
                        "functional_base_snapshot": str(functional_base),
                        "candidate_path": str(snapshot),
                        "candidate_sha256": candidate.package_sha256})
    _write_json(out_dir / "functional_fork.json", fork_record)

    probe_workspace = out_dir / "_probe_workspace"
    probe_workspace.mkdir()
    probe_policy = PC.package_sandbox_policy(target_experiment, probe_workspace, snapshot)
    campaign = {
        "status": "NO_GO",
        "measurement_status": "NO_GO",
        "claim_status": "PENDING" if args.mode == "claim" else "NOT_ESTABLISHED",
        "claim_decision": None,
        **mode_record,
        "approach": args.approach,
        "functional_run_id": functional.run_id,
        "functional_submission_sha256": functional.digest,
        "functional_public_capsules": functional.public_capsules,
        "functional_hidden_capsules": functional.hidden_capsules,
        "snapshot": str(snapshot),
        "snapshot_sha256": candidate.package_sha256,
        "candidate_stage": _candidate_stage_record(candidate),
        "model_host_lane_snapshot": candidate.handoff.host_lane,
        "frozen_contract": {
            "path": str(candidate.contract_root), "sha256": candidate.contract_sha256,
            "source": "functional_bundle_input_snapshot_v2",
        },
        "full_model_admission": {
            "capsule": sentinel.capsule, "source_sha256": sentinel.source_sha256,
            "status": "not_run", "role": "correctness_admission_not_performance_claim",
        },
        "workload_snapshot": str(frozen_corpus.root),
        "workload_manifest": str(frozen_corpus.manifest_path),
        "workload_manifest_sha256": frozen_corpus.manifest_sha256,
        "workload_sha256": frozen_corpus.capsules_sha256,
        "expected_identities": [identity.as_dict() for identity in expected],
        "simulator_semantics": {
            "spike": {"tier": "L2", "purpose": "correctness_screen", "citable": False},
            "verilator": {"tier": "L3", "purpose": "performance_certification", "citable": True},
        },
        "decision_boundary": (
            _boundary_record(claim_module, claim_identity, claim_family, None)
            if args.mode == "claim" else {
            "module": claim_identity.module,
            "identity_bridge": "not_invoked_in_measurement_smoke",
            "promotion_integration": "blocked",
            "promotion_status": "BLOCKED",
            "reason": PC.SMOKE_CLAIM_NONCLAIM,
        }),
        "fork_before": before.to_dict(),
        "fork_after": None,
        "candidate_before": {"state": "held", "ok": True,
                             "sha256": candidate.package_sha256},
        "candidate_after": None,
        "sandbox": {
            "engine": "bwrap",
            "network": "available",
            "package_read_only": True,
            "answer_surface_coverage_gap": list(probe_policy.coverage_gap),
            "required_tool_probes": [probe.label for probe in probe_policy.required_tools],
            "tool_probe_results": [],
        },
        "completion": PC.completion_report([], expected),
        "results": None,
        "workers": args.workers,
        "refusal": "campaign has not completed",
    }
    _write_json(out_dir / "campaign_manifest.json", campaign)

    results: list[dict] = []
    refusal: str | None = None
    try:
        cells_root = out_dir / "_cell_workspaces"
        cells_root.mkdir()
        tasks: list[
            tuple[Path, PC.PerformanceCapsule, str, Path, int, Any, Path, str]] = []
        for member in frozen_corpus.capsules:
            for replica_index in range(args.replicates):
                replicate = f"r{replica_index:03d}"
                # Each replica gets a fresh writable mount. The package cannot inspect oracle/result
                # files from an earlier cell; capsule_runner copies this cell's interface MLIR into
                # generated/ before the first boxed entrypoint and keeps the corpus outside the mount.
                cell_name = f"{member.family}__{member.capsule}__{replicate}"
                cell_workspace = cells_root / cell_name
                cell_workspace.mkdir()
                capsule_runs = cell_workspace / "capsule_runs"
                capsule_runs.mkdir()
                tasks.append((snapshot, member, replicate, cell_workspace,
                              args.timeout, target_experiment, candidate.contract_root,
                              candidate.contract_sha256))

        with PC.functional_host_lane(functional):
            campaign["sandbox"]["tool_probe_results"] = PC.run_tool_probes(probe_policy)
            sentinel_workspace = out_dir / "_full_model_admission"
            sentinel_workspace.mkdir()
            admission = run_full_model_admission(
                snapshot, sentinel, sentinel_workspace, args.timeout, target_experiment,
                candidate.contract_root, candidate.contract_sha256)
            campaign["full_model_admission"] = admission
            _write_json(out_dir / "campaign_manifest.json", campaign)
            if admission["passed"] is not True:
                raise PC.CampaignGateError(
                    f"full-model admission sentinel {sentinel.capsule!r} did not pass its citable "
                    f"RTL tier: {admission['tier_status']}")
            if args.workers == 1:
                task_results = [_run_cell_task(task) for task in tasks]
            else:
                # Each worker owns its oot_runner monkeypatch and writable build copy. Threads would
                # race those process globals, so only process isolation is admitted here.
                context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=args.workers, mp_context=context) as pool:
                    task_results = list(pool.map(_run_cell_task, tasks))

        for task, rows in zip(tasks, task_results, strict=True):
            (_package, member, replicate, _workspace, _timeout, _target_experiment,
             _contract_root, _contract_sha256) = task
            print(f"\n=== Arm-4 performance {member.family}/{member.capsule}/{replicate} "
                  "(L2 screen + L3 certification) ===", flush=True)
            results.extend(rows)
            cell_name = f"{member.family}__{member.capsule}__{replicate}"
            _write_json(out_dir / f"{cell_name}.json", rows)
            campaign["completion"] = PC.completion_report(results, expected)
            _write_json(out_dir / "campaign_manifest.json", campaign)
            summary = {row["identity"]["simulator"]:
                       (row.get("cycles"), row.get("correct")) for row in rows}
            print(f"  [arm4] {summary}", flush=True)
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            PC.verify_frozen_performance_corpus(frozen_corpus)
        except PC.CampaignGateError as exc:
            refusal = f"frozen performance corpus changed during execution: {exc}"
        try:
            result_record = PC.write_immutable_json(out_dir / "perf_results.json", results)
            verified_results = PC.verify_immutable_json(
                out_dir / "perf_results.json", result_record["sha256"])
            if verified_results != results:
                raise PC.CampaignGateError("immutable results did not round-trip exactly")
            result_record.update({"records": len(results), "verified": True, "immutable": True})
            digest_record = PC.write_immutable_json(
                out_dir / "perf_results.digest.json", result_record)
            result_record["digest_record_sha256"] = digest_record["sha256"]
            campaign["results"] = result_record
        except PC.CampaignGateError as exc:
            refusal = f"performance results could not be sealed: {exc}"
        after = PC.check_fork(fork, functional_base)
        campaign["fork_after"] = after.to_dict()
        if after.ok is not True:
            refusal = f"functional fork changed during performance: {after.reason}"
        try:
            functional_after = PC.inspect_functional_run(
                _FUNCTIONAL_RUNS, args.functional_run_id, args.functional_submission_sha256)
            candidate_after = load_measurement_candidate(
                args.candidate_record, functional_after, target_experiment)
            if (candidate_after.handoff.record_sha256 != candidate.handoff.record_sha256
                    or candidate_after.package_sha256 != candidate.package_sha256
                    or candidate_after.corpus.manifest_sha256 != frozen_corpus.manifest_sha256
                    or candidate_after.corpus.capsules_sha256 != frozen_corpus.capsules_sha256
                    or candidate_after.contract_sha256 != candidate.contract_sha256
                    or candidate_after.sentinel != candidate.sentinel
                    or functional_after.model_host_lane_snapshot !=
                    functional.model_host_lane_snapshot):
                raise PC.CampaignGateError(
                    "candidate, corpus, or functional host-lane identity changed during measurement")
            campaign["candidate_after"] = {
                "state": "held", "ok": True, "sha256": candidate_after.package_sha256}
        except PC.CampaignGateError as exc:
            campaign["candidate_after"] = {"state": "unknown", "ok": None,
                                           "reason": str(exc)}
            refusal = f"candidate handoff changed during performance: {exc}"
        try:
            campaign["completion"] = PC.completion_report(results, expected)
            if refusal is None and not campaign["completion"]["complete"]:
                counts = campaign["completion"]
                refusal = (f"Arm-4 performance reported {counts['reported']} of "
                           f"{counts['expected']} expected cells; {counts['failed']} reported "
                           "cell(s) failed correctness or positive-cycle measurement")
        except PC.CampaignGateError as exc:
            if refusal is None:
                refusal = str(exc)
        if refusal is None and args.mode == "claim":
            try:
                decision = _claim_decision(frozen_corpus.capsules, results,
                                           replicates=args.replicates,
                                           target=target_experiment.target)
                campaign["claim_decision"] = decision
                campaign["claim_status"] = decision.get("status")
                campaign["decision_boundary"] = _boundary_record(
                    claim_module, claim_identity, claim_family, decision)
                if decision.get("status") == "REFUSED":
                    refusal = (f"{claim_family} claim decision refused sealed evidence: "
                               + "; ".join(decision.get("refusal_reasons") or []))
            except Exception as exc:
                refusal = (f"{claim_family} claim decision could not be evaluated: "
                           f"{type(exc).__name__}: {exc}")
        elif args.mode == "measurement-smoke":
            campaign["claim_status"] = "NOT_ESTABLISHED"
        campaign["refusal"] = refusal
        campaign["measurement_status"] = "GO" if refusal is None else "NO_GO"
        # Back-compatible top-level status is explicitly the measurement status.  It must never be
        # read as a promoted performance claim; claim_status and decision_boundary are mandatory in
        # the reporting gate.
        campaign["status"] = campaign["measurement_status"]
        _write_json(out_dir / "campaign_manifest.json", campaign)

    try:
        PC.seal_campaign_manifest(out_dir / "campaign_manifest.json", campaign)
    except PC.CampaignGateError as exc:
        print(f"\nNO-GO: campaign manifest could not be sealed: {exc}\n"
              f"run: {out_dir}", flush=True)
        return 2
    if refusal is not None:
        print(f"\nNO-GO: {refusal}\nmanifest: {out_dir / 'campaign_manifest.json'}", flush=True)
        return 2
    claim_message = (f"{claim_family} CLAIM {campaign['claim_status']}"
                     if args.mode == "claim" else "CLAIM NOT ESTABLISHED")
    print(f"\nMEASUREMENT GO; {claim_message}: completed "
          f"{campaign['completion']['expected']} Arm-4 cells; "
          f"manifest: {out_dir / 'campaign_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
