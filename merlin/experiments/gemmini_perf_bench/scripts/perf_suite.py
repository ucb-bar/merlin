#!/usr/bin/env python3
"""Prepare, run/resume, and inspect one frozen multi-claim performance suite.

The input JSON is an orchestrator Config plus ``members`` (explicit capsule names), ``suite_id``,
and optional ``campaign_workers`` / ``trial_workers``. One declared family is one campaign; its
analyzer, cohort and replicate schedule are derived before any authoring starts. All campaigns use
the same private source snapshot. Run this entry point with the repository Python environment.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import fields
from pathlib import Path

import _pbcommon  # noqa: F401 - standalone bootstrap
import perf_snapshot as SNAP
import perf_agent_stage as STAGE
import perf_gsim_gate as GATE
import run_agentic_perf_experiment as ORCH
import run_paired_perf_bench as PAIRED
from merlin.common.paths import artifacts_dir, out_dir, repo_root
from merlin.benchharness import runs_root
from merlin.targetgen.target_experiment import load_target_experiment

SCHEMA = "merlin.performance-suite.v1"
SCRIPTS = Path("merlin/experiments/gemmini_perf_bench/scripts")
PATH_FIELDS = {
    "descriptor", "rtl_facts", "perf_profile", "gsim_certificate", "functional_gsim_certificate",
    "telemetry_price_table", "chia_python",
}
DEFAULTS = dict(model="gpt-5.6-sol", effort="high", wall_budget_seconds=43200, rounds=1,
                round_timeout_seconds=39600, max_tool_calls=400, tool_timeout_seconds=600,
                smoke_replicates=1, holdout_count=4, measurement_timeout=600,
                heldout_qualification_timeout=600, sim_workers=8)
EVALUATION_FILES = {
    "backend_contract": Path("merlin/contract/mlir_oot_backend_contract.yaml"),
    "command_buffer_schema": Path("merlin/contract/schemas/command_buffer.schema.json"),
    "gemmini_harness": Path("merlin/targets/gemmini/backend/gemmini.py"),
    "paired_runner": SCRIPTS / "run_paired_perf_bench.py",
}


class SuiteError(RuntimeError):
    pass


def evaluation_boundary(source: Path) -> dict:
    """Exact evaluator bytes fixed by the suite snapshot before any candidate is authored."""
    files = {}
    for label, relative in EVALUATION_FILES.items():
        path = source / relative
        if not path.is_file():
            raise SuiteError(f"evaluation boundary file is absent: {relative}")
        files[label] = {"path": relative.as_posix(), "sha256": SNAP.sha_file(path)}
    return {"frozen_before_authoring": True, "files": files}


def claim_groups(descriptor: Path, certificate: Path, certificate_sha256: str,
                 members: list[str]) -> list[dict]:
    if not members or len(set(members)) != len(members):
        raise SuiteError("members must be explicit, nonempty and unique")
    corpus = STAGE.discover_performance_corpus(load_target_experiment(descriptor))
    by_name = {member.capsule: member for member in corpus.capsules}
    if set(members) - set(by_name):
        raise SuiteError(f"undiscovered members: {sorted(set(members) - set(by_name))}")
    cert = GATE.load_certificate(certificate, expected_sha256=certificate_sha256)
    groups = defaultdict(list)
    for name in members:
        member = by_name[name]
        decision = GATE.plan_evaluation(cert, PAIRED._gsim_workload(member),
                                        phase="development_correctness", gsim_available=True)
        if not (decision.admitted and decision.eligible and decision.use_gsim
                and decision.selected_engine == "gsim"):
            raise SuiteError(f"member has no tuning certificate authority: {name}")
        groups[member.family].append(member)
    result = []
    # Dict insertion order preserves the owner's explicit member order.  Campaigns are often
    # diagnostic dependencies (for example residency before synchronization before depth), so an
    # alphabetical regrouping silently changes the predeclared experiment sequence.
    for family, capsules in groups.items():
        capsules.sort(key=lambda member: member.capsule)
        formal = STAGE.prepare_formal_claim(capsules)
        result.append({"family": family, "members": [c.capsule for c in capsules],
                       "formal_claim": formal,
                       "sources": {c.capsule: c.source_sha256 for c in capsules}})
    return result


def config_from(document: dict) -> ORCH.Config:
    data = dict(document)
    for key in PATH_FIELDS | {"root"}:
        if data.get(key) is not None:
            data[key] = Path(data[key])
    if "waive_functional_gate" in data:
        data["waive_functional_gate"] = tuple(data["waive_functional_gate"])
    return ORCH.Config(**data)


def _input_path(value: str | Path, *, preserve_entrypoint: bool = False) -> Path:
    """Return an existing absolute input, retaining a virtualenv's executable symlink when asked."""
    path = Path(value).absolute()
    if not path.exists():
        raise SuiteError(f"suite input is absent: {path}")
    # Invoking ``venv/bin/python`` activates that environment by path. Resolving the final symlink
    # invokes the base interpreter instead and silently drops the venv's installed packages.
    return path if preserve_entrypoint else path.resolve(strict=True)


def coordinator_args(config: dict) -> list[str]:
    argv = []
    for field in fields(ORCH.Config):
        if field.name not in config or config[field.name] is None:
            continue
        value = config[field.name]
        flag = "--" + field.name.replace("_", "-")
        if field.name == "waive_functional_gate":
            for waiver in value:
                argv.extend((flag, waiver))
        elif isinstance(value, bool):
            if field.name == "hardware_counters":
                argv.append(flag if value else "--no-hardware-counters")
            elif value:
                argv.append(flag)
        else:
            argv.extend((flag, str(value)))
    return argv


def environment(root: Path, manifest: dict) -> dict[str, str]:
    snapshot = root / "source"
    llvm_install = snapshot / "third_party/llvm-install"
    return {**os.environ, "MERLIN_REPO_ROOT": str(snapshot),
            "MERLIN_OUT_ROOT": manifest["output_root"], "PYTHONDONTWRITEBYTECODE": "1",
            # Never inherit a mutable external install path. One PR run passed its launch probe,
            # then the external host-merlin build disappeared before formal measurement and every
            # cell failed uniformly. The suite snapshot already contains the exact LLVM toolchain;
            # bind both the host runner and the inner sandbox to that immutable copy.
            "MERLIN_CLANG_INSTALL": str(llvm_install),
            "MERLIN_CLANG": str(llvm_install / "bin/clang-23"),
            # runtime_preflight executes the paired adapter in this process rather than through
            # ORCH.child_environment. Pin the same warm measured-only protocol here so preflight and
            # campaign cannot validate different cache conditions.
            "MERLIN_CACHE_STATE": ORCH.MEASUREMENT_CACHE_CONDITION,
            "PYTHONPATH": os.pathsep.join(str(snapshot / item) for item in (
                "merlin/python", str(SCRIPTS), "merlin/experiments/capsule_bench/harness")),
            "MERLIN_PERF_CAMPAIGN_FANOUT": str(manifest["trial_workers"]),
            "MERLIN_PERF_SWEEP_WORKERS": str(manifest["sim_workers"]),
            "MERLIN_REQUIRED_RTL_ENGINE": "gsim"}


def prepare(root: Path, specification: dict) -> Path:
    spec = dict(specification)
    suite_id = str(spec.pop("suite_id"))
    ORCH._safe(suite_id, label="suite id")
    members = spec.pop("members")
    campaign_workers, trial_workers = spec.pop("campaign_workers", 1), spec.pop("trial_workers", 3)
    for name, count in (("campaign_workers", campaign_workers), ("trial_workers", trial_workers)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SuiteError(f"{name} must be a positive integer")
    if trial_workers > len(ORCH.TRIALS):
        raise SuiteError("trial_workers exceeds the three predeclared trials")
    spec = {**DEFAULTS, **spec}
    if (isinstance(spec["sim_workers"], bool) or not isinstance(spec["sim_workers"], int)
            or spec["sim_workers"] < 1):
        raise SuiteError("sim_workers must be a positive integer")
    for key in PATH_FIELDS:
        if spec.get(key) is not None:
            spec[key] = str(_input_path(
                spec[key], preserve_entrypoint=key == "chia_python"))
    if spec.get("waive_functional_gsim_certificate") or not spec.get("functional_gsim_certificate"):
        raise SuiteError("this suite requires exact or stratified functional evidence, not a waiver")
    for key in ("gsim_certificate", "functional_gsim_certificate"):
        digest = SNAP.sha_file(Path(spec[key]))
        if spec.get(key + "_sha256", digest) != digest:
            raise SuiteError(f"{key} digest mismatch")
        spec[key + "_sha256"] = digest
    # Fail cheaply before copying the source/corpus or starting orchestration.
    groups = claim_groups(Path(spec["descriptor"]), Path(spec["gsim_certificate"]),
                          spec["gsim_certificate_sha256"], members)
    root = root.absolute()
    root.relative_to(artifacts_dir().resolve())
    if root.exists() or root.is_symlink():
        raise SuiteError("prepare requires a new suite root; use run to resume a sealed suite")
    root.mkdir(parents=True, mode=0o700)
    snapshot = root / "source"
    receipt = SNAP.create(
        repo_root(), snapshot, output_root=out_dir(),
        target_name=load_target_experiment(Path(spec["descriptor"])).target)
    # Only corpus/source inputs move into the snapshot. Certificates retain their evidence roots:
    # their provenance verifier follows their sealed declarations and exact capture receipts.
    for key in ("descriptor", "rtl_facts", "perf_profile", "telemetry_price_table"):
        relative = Path(spec[key]).relative_to(repo_root())
        spec[key] = str(snapshot / relative)
    campaigns = []
    for index, group in enumerate(groups):
        identity = f"{suite_id}__claim_{index:02d}"
        config = {**spec, "experiment_id": identity, "root": str(root / identity),
                  "perf_capsules": ",".join(group["members"]), "perf_families": "all"}
        config_from(config)  # reject misspelled/unhandled configuration, before sealing
        campaigns.append({**group, "config": config})
    manifest = {"schema": SCHEMA, "suite_id": suite_id, "output_root": str(out_dir()),
                "source_snapshot_sha256": SNAP.sha_file(receipt), "campaigns": campaigns,
                "evaluation_boundary": evaluation_boundary(snapshot),
                "campaign_workers": campaign_workers, "trial_workers": trial_workers,
                "sim_workers": spec["sim_workers"], "trials": list(ORCH.TRIALS),
                "measurement": {
                    "requested_cache_condition": ORCH.MEASUREMENT_CACHE_CONDITION,
                    "cache_protocol": "one_unmeasured_predecessor",
                    "cycle_window": "gemmini_region",
                    "recorded_metric": "cycles",
                },
                "selection": "all campaigns, all trials; separate claims, no best-of"}
    return SNAP.seal(root, "suite", manifest)


def load(root: Path) -> dict:
    _path, manifest = SNAP.load_seal(root, "suite")
    if manifest.get("schema") != SCHEMA:
        raise SuiteError("unknown suite schema")
    SNAP.verify(root / "source")
    receipt, _ = SNAP.load_seal(root / "source", "snapshot")
    if SNAP.sha_file(receipt) != manifest["source_snapshot_sha256"]:
        raise SuiteError("suite snapshot identity changed")
    return manifest


def completion(campaign: dict) -> dict | None:
    root = Path(campaign["config"]["root"])
    if not list(root.glob("experiment_manifest.*.json")):
        return None
    path, result = SNAP.load_seal(root, "experiment_manifest")
    if (result.get("schema") != ORCH.SCHEMA or result.get("status") != "GO"
            or result.get("declaration", {}).get("experiment_id")
            != campaign["config"]["experiment_id"]):
        raise SuiteError("campaign completion has a foreign identity or non-admitted status")
    return {"path": str(path), "sha256": SNAP.sha_file(path)}


def run(root: Path) -> list[dict]:
    # The open descriptor is inherited by each wrapper, retaining exclusion if this parent dies.
    with (root / "launcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuiteError("another launcher owns this suite") from exc
        manifest = load(root)
        env = environment(root, manifest)
        snapshot = root / "source"
        # In a FRESH process: never let modules already imported from the live tree pass preflight.
        check = subprocess.run([sys.executable, str(snapshot / SCRIPTS / "perf_suite.py"),
                                "preflight", "--root", str(root)], cwd=snapshot, env=env)
        if check.returncode:
            raise SuiteError("snapshot preflight refused; no campaigns launched")

        def launch(campaign):
            done = completion(campaign)
            if done:
                return done
            config = campaign["config"]
            identity = config["experiment_id"]
            attempts = root / "launcher_attempts" / identity
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = attempts / f"attempt-{len(list(attempts.iterdir())):03d}"
            attempt.mkdir()
            python = config.get("chia_python")
            if not python:
                raise SuiteError("chia_python must be explicit for a suite launch")
            argv = [python, str(snapshot / SCRIPTS / "chia_agentic_perf_experiment.py"),
                    "--orchestration-run-id", identity + "__chia",
                    "--codex-slots", "1", "--gsim-slots", str(config["sim_workers"]),
                    "--", *coordinator_args(config)]
            SNAP.seal(attempt, "launch", {"argv": argv, "suite_id": manifest["suite_id"]})
            with (attempt / "console.log").open("xb") as log:
                result = subprocess.run(argv, cwd=snapshot, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            SNAP.seal(attempt, "exit", {"returncode": result.returncode})
            if result.returncode or not (done := completion(campaign)):
                raise SuiteError(f"{identity} did not complete; inspect {attempt / 'console.log'}")
            return done

        # Submit only the declared concurrency window. A failed claim must not start every
        # queued campaign as executor.map would while its caller is unwinding the exception.
        results = {}
        campaigns = iter(enumerate(manifest["campaigns"]))
        with ThreadPoolExecutor(max_workers=manifest["campaign_workers"]) as pool:
            pending = {}
            for _ in range(manifest["campaign_workers"]):
                item = next(campaigns, None)
                if item is not None:
                    index, campaign = item
                    pending[pool.submit(launch, campaign)] = index
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    results[pending.pop(future)] = future.result()
                for _ in finished:
                    item = next(campaigns, None)
                    if item is not None:
                        index, campaign = item
                        pending[pool.submit(launch, campaign)] = index
        results = [results[i] for i in range(len(manifest["campaigns"]))]
        if not list(root.glob("completion.*.json")):
            SNAP.seal(root, "completion", {"schema": SCHEMA, "campaigns": results,
                                           "status": "all_campaigns_sealed"})
        return results


def runtime_preflight(config: ORCH.Config, root: Path) -> Path:
    """Exercise every pre-authoring program arm the frozen claim actually owns.

    This is host-private execution evidence, not tuning feedback or a performance claim. Each call
    uses fresh immutable inputs and keeps failed attempts; no passing receipt bypasses a later run.
    The adapter may reuse only its own certificate-bound execution cache. A candidate-only claim has
    no candidate before authoring; its certificate/adapter/capability gates are resolved here and its
    execution is explicitly deferred to the first candidate feedback call. Running the functional
    ancestor and calling its expected capability failure a suite failure would test a program the
    claim does not measure.
    """
    root.mkdir(parents=True, exist_ok=False)
    record = {"schema": "merlin.performance-runtime-preflight.v1", "status": "NO_GO",
              "functional_submission_sha256": config.functional_submission_sha256,
              "certificate_sha256": config.gsim_certificate_sha256, "cells": []}
    try:
        target = load_target_experiment(config.descriptor)
        functional = STAGE.inspect_stage_functional_run(
            runs_root(target.target, "capsule-bench"), config.functional_run_id,
            config.functional_submission_sha256,
            waive=frozenset(config.waive_functional_gate or ()))
        baseline = STAGE.PC.materialize_perf_workspace(functional, root / "baseline")
        discovered = STAGE.discover_performance_corpus(
            target, families=config.perf_families, capsules=config.perf_capsules)
        corpus = STAGE.freeze_performance_corpus(discovered, root / "corpus")
        if not corpus.capsules:
            raise SuiteError("runtime preflight has no baseline members")
        record["corpus_sha256"] = corpus.capsules_sha256
        formal = STAGE.prepare_formal_claim(corpus.capsules)
        declaration = formal.get("declaration") if isinstance(formal, dict) else None
        if not isinstance(declaration, dict):
            raise SuiteError("runtime preflight has no frozen formal declaration")
        program_arm = declaration.get("program_arm", "baseline")
        if program_arm not in ("baseline", "candidate"):
            raise SuiteError(f"runtime preflight does not understand program arm {program_arm!r}")
        stated = {row.get("program_arm") for row in formal.get("expected_identities", [])
                  if isinstance(row, dict) and row.get("program_arm") is not None}
        if stated and stated != {program_arm}:
            raise SuiteError("formal measurement identities disagree with their declared program arm")
        if declaration.get("program_arm") is not None and not stated:
            raise SuiteError("formal declaration names a program arm but its identities omit it")
        record["program_arm"] = program_arm
        record["formal_claim_sha256"] = STAGE._document_sha256(formal)
        evaluator = STAGE.prepare_development_feedback(
            certificate_path=config.gsim_certificate,
            certificate_sha256=config.gsim_certificate_sha256,
            rtl_facts_path=config.rtl_facts, corpus=corpus, baseline=baseline,
            baseline_sha256=functional.digest, target_experiment=target,
            work_root=root / "measurements", functional_run_dir=functional.run_dir)
        for index, member in enumerate(corpus.capsules):
            decision = evaluator.decisions[(member.family, member.capsule)]
            if program_arm == "candidate":
                record["cells"].append({
                    "family": member.family, "capsule": member.capsule,
                    "source_sha256": member.source_sha256, "arm": "candidate",
                    "execution_status": "gated_until_candidate_exists",
                    "correct": None, "gsim_cycles": None,
                    "qualification": decision.to_dict(),
                })
                continue
            raw = evaluator._execute(
                arm="baseline", package=baseline, package_sha256=functional.digest,
                member=member, decision=decision, workspace=root / f"measurement_{index:03d}",
                timeout_s=config.measurement_timeout)
            evidence = SNAP.seal(root, f"execution.{index:03d}", raw)
            row = evaluator._redact_execution(
                raw, decision, arm="baseline", family=member.family, capsule=member.capsule,
                required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()))
            record["cells"].append({"family": member.family, "capsule": member.capsule,
                                    "source_sha256": member.source_sha256,
                                    "evidence": str(evidence), **row})
            if row["correct"] is not True:
                raise SuiteError(f"runtime preflight baseline is incorrect: {member.capsule}")
        if STAGE.hash_tree(baseline)["sha256"] != functional.digest:
            raise SuiteError("runtime preflight mutated its frozen baseline")
        record["live_execution"] = (
            "completed" if program_arm == "baseline"
            else "required_on_first_candidate_feedback_before_any_measurement_can_pass")
        record["status"] = "GO"
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
        SNAP.seal(root, "runtime_preflight", record)
        raise
    return SNAP.seal(root, "runtime_preflight", record)


def preflight(root: Path) -> None:
    manifest = load(root)
    if repo_root().resolve() != (root / "source").resolve():
        raise SuiteError("suite preflight must run from its sealed source snapshot")
    for name, module in tuple(sys.modules.items()):
        if name == "merlin" or name.startswith("merlin.") or name in (
                "_pbcommon", "_common", "perf_agent_stage", "run_agentic_perf_experiment"):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root / "source"):
                raise SuiteError(f"live module leaked into snapshot preflight: {name}")
    pending = []
    for campaign in manifest["campaigns"]:
        config = config_from(campaign["config"])
        groups = claim_groups(config.descriptor, config.gsim_certificate,
                              config.gsim_certificate_sha256, campaign["members"])
        if len(groups) != 1 or groups[0] != {key: campaign[key] for key in groups[0]}:
            raise SuiteError("claim cohort/contract changed since suite sealing")
        result = ORCH.preflight(config, heldout_certificate_provider_available=True)
        print(json.dumps({"family": campaign["family"], "status": result["status"],
                          "blockers": result["blockers"]}), flush=True)
        if result["status"] != "GO":
            raise SuiteError("campaign preflight refused")
        if completion(campaign) is None:
            pending.append(config)
    # Finish all cheap gates before paying for any baseline, then finish all baseline gates before
    # starting any authoring. Keep each invocation separate so a refusal never overwrites evidence.
    if pending:
        attempts = root / "runtime_preflight"
        attempts.mkdir(exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
        for index, config in enumerate(pending):
            receipt = runtime_preflight(config, attempt / f"claim_{index:02d}")
            print(json.dumps({"experiment_id": config.experiment_id,
                              "runtime_preflight": str(receipt), "status": "GO"}), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "preflight"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    try:
        if args.action == "prepare":
            if args.config is None:
                parser.error("prepare requires --config")
            print(prepare(root, json.loads(args.config.read_text())))
        elif args.action == "run":
            print(json.dumps(run(root)))
        elif args.action == "preflight":
            preflight(root)
        else:
            manifest = load(root)
            print(json.dumps([{"family": c["family"], "completion": completion(c)}
                              for c in manifest["campaigns"]], indent=2))
    except (SuiteError, SNAP.SnapshotError, ORCH.ExperimentError, STAGE.StageGateError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
