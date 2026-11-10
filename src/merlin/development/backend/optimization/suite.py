#!/usr/bin/env python3
"""Prepare, run/resume, and inspect one frozen multi-claim performance suite.

The input JSON is an orchestrator Config plus ``members`` (explicit capsule names), ``suite_id``,
and optional ``campaign_workers`` / ``trial_workers``. One declared family is one campaign; its
analyzer, cohort and replicate schedule are derived before any authoring starts. All campaigns use
the same private source snapshot. The installed module owns orchestration; retained native scripts are compatibility launchers.
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

from merlin.benchharness import chia_bridge, runs_root
from merlin.common.paths import artifacts_dir, out_dir, repo_root
from merlin.development.backend.optimization import campaign as PC  # noqa: F401 -- historical patch lookup
from merlin.development.backend.optimization import candidate_record as RECORD
from merlin.development.backend.optimization import checkpoint_admission as AD
from merlin.development.backend.optimization import contracts as P2_CONTRACTS
from merlin.development.backend.optimization import corpus as P2_CORPUS
from merlin.development.backend.optimization import development_feedback as DF  # noqa: F401 -- historical patch lookup
from merlin.development.backend.optimization import functional_inputs as FI  # noqa: F401 -- historical patch lookup
from merlin.development.backend.optimization import gsim_gate as GATE
from merlin.development.backend.optimization import holdout_corpus as HOLDOUT
from merlin.development.backend.optimization import paired_measurement as PME
from merlin.development.backend.optimization.source_layout import IMPLEMENTATION_ROOT, STUDY_POLICY, snapshot_layout
from merlin.development.backend.optimization.suite_runtime import SuiteError, require_managed_wrapper, runtime_preflight
from merlin.experiments import source_snapshot as SNAP
from merlin.experiments.frozen_python import python_command
from merlin.targetgen.target_experiment import load_target_experiment

SCHEMA = "merlin.performance-suite.v1"

PATH_FIELDS = {
    "descriptor",
    "rtl_facts",
    "perf_profile",
    "gsim_certificate",
    "functional_gsim_certificate",
    "telemetry_price_table",
    "chia_python",
}
CONTEXT_PATH_FIELDS = (
    "source_root",
    "contract_root",
    "functional_runs_root",
    "stage_root",
    "measurement_root",
    "holdout_catalog",
    "core_package_root",
    "experiments_package_root",
    "experiments_namespace_root",
    "chia_wrapper",
)
DEFAULTS = dict(
    model="gpt-5.6-sol",
    effort="high",
    wall_budget_seconds=43200,
    rounds=1,
    round_timeout_seconds=39600,
    max_tool_calls=400,
    tool_timeout_seconds=600,
    smoke_replicates=1,
    holdout_count=4,
    measurement_timeout=600,
    heldout_qualification_timeout=600,
    sim_workers=8,
)
EVALUATION_FILES = {
    "backend_contract": Path("src/merlin/resources/contracts/mlir_oot_backend_contract.yaml"),
    "command_buffer_schema": Path("src/merlin/resources/contracts/schemas/command_buffer.schema.json"),
    "paired_runner": IMPLEMENTATION_ROOT / "paired_cli.py",
}


def evaluation_boundary(source: Path, receipt: dict) -> dict:
    """Exact evaluator bytes fixed by the suite snapshot before any candidate is authored."""
    files = {}
    for label, relative in EVALUATION_FILES.items():
        path = source / relative
        if not path.is_file():
            raise SuiteError(f"evaluation boundary file is absent: {relative}")
        files[label] = {"path": relative.as_posix(), "sha256": SNAP.sha_file(path)}
    provider = receipt.get("selected_provider")
    if provider is None:
        raise SuiteError("suite evaluator lacks sealed selected-provider ownership")
    prefix = provider["root"] + "/"
    for relative, digest in receipt["files"].items():
        if relative.startswith(prefix):
            files["provider:" + relative[len(prefix) :]] = {"path": relative, "sha256": digest}
    return {"frozen_before_authoring": True, "files": files, "selected_provider": provider}


def claim_groups(descriptor: Path, certificate: Path, certificate_sha256: str, members: list[str]) -> list[dict]:
    if not members or len(set(members)) != len(members):
        raise SuiteError("members must be explicit, nonempty and unique")
    corpus = P2_CORPUS.discover_performance_corpus(load_target_experiment(descriptor))
    by_name = {member.capsule: member for member in corpus.capsules}
    if set(members) - set(by_name):
        raise SuiteError(f"undiscovered members: {sorted(set(members) - set(by_name))}")
    cert = GATE.load_certificate(certificate, expected_sha256=certificate_sha256)
    groups = defaultdict(list)
    for name in members:
        member = by_name[name]
        decision = GATE.plan_evaluation(
            cert, PME.gsim_workload(member), phase="development_correctness", gsim_available=True
        )
        if not (decision.admitted and decision.eligible and decision.use_gsim and decision.selected_engine == "gsim"):
            raise SuiteError(f"member has no tuning certificate authority: {name}")
        groups[member.family].append(member)
    result = []
    # Dict insertion order preserves the owner's explicit member order.  Campaigns are often
    # diagnostic dependencies (for example residency before synchronization before depth), so an
    # alphabetical regrouping silently changes the predeclared experiment sequence.
    for family, capsules in groups.items():
        capsules.sort(key=lambda member: member.capsule)
        formal = RECORD.prepare_formal_claim(capsules)
        result.append(
            {
                "family": family,
                "members": [c.capsule for c in capsules],
                "formal_claim": formal,
                "sources": {c.capsule: c.source_sha256 for c in capsules},
            }
        )
    return result


def config_from(document: dict) -> AD.Config:
    data = dict(document)
    if any(not data.get(key) for key in (*CONTEXT_PATH_FIELDS, "suite", "driver_python")):
        raise SuiteError("suite config lacks explicit execution resources; prepare a newly frozen suite")
    resources = {key: Path(data.pop(key)) for key in CONTEXT_PATH_FIELDS}
    suite = data.pop("suite")
    driver = data.pop("driver_python")
    if not isinstance(driver, str) or not Path(driver).is_absolute():
        raise SuiteError("driver_python must be an absolute interpreter entrypoint")
    context = AD.ExecutionContext(
        **{key: resources[key] for key in CONTEXT_PATH_FIELDS[:5]},
        holdout_sources=HOLDOUT.HoldoutSourceContext(
            source_root=resources["source_root"],
            catalog_path=resources["holdout_catalog"],
            core_package_root=resources["core_package_root"],
            experiments_package_root=resources["experiments_package_root"],
            experiments_namespace_root=resources["experiments_namespace_root"],
        ),
        chia_wrapper=resources["chia_wrapper"],
        suite=suite,
        invocation=(
            driver,
            "-m",
            "merlin.development.backend.optimization.checkpoint_cli",
            *coordinator_args(document),
        ),
    )
    for key in PATH_FIELDS | {"root"}:
        if data.get(key) is not None:
            data[key] = Path(data[key])
    if "waive_functional_gate" in data:
        data["waive_functional_gate"] = tuple(data["waive_functional_gate"])
    return AD.Config(context=context, **data)


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
    for field in fields(AD.Config):
        if field.name == "context":
            continue
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
    for name in (*CONTEXT_PATH_FIELDS, "suite"):
        if config.get(name) is not None:
            argv.extend(("--" + name.replace("_", "-"), str(config[name])))
    return argv


def environment(root: Path, manifest: dict) -> dict[str, str]:
    snapshot = root / "source"
    llvm_install = snapshot / "third_party/llvm-install"
    drivers = {campaign["config"].get("driver_python") for campaign in manifest["campaigns"]}
    if len(drivers) != 1 or not all(isinstance(value, str) and Path(value).is_absolute() for value in drivers):
        raise SuiteError("suite requires one sealed driver_python; prepare a newly frozen suite")
    return {
        **os.environ,
        **SNAP.provider_environment(snapshot, SNAP.verify(snapshot)),
        "MERLIN_REPO_ROOT": str(snapshot),
        "MERLIN_OUT_ROOT": manifest["output_root"],
        "MERLIN_EXPERIMENT_PYTHON": next(iter(drivers)),
        "PYTHONDONTWRITEBYTECODE": "1",
        # Never inherit a mutable external install path. One PR run passed its launch probe,
        # then the external host-merlin build disappeared before formal measurement and every
        # cell failed uniformly. The suite snapshot already contains the exact LLVM toolchain;
        # bind both the host runner and the inner sandbox to that immutable copy.
        "MERLIN_CLANG_INSTALL": str(llvm_install),
        "MERLIN_CLANG": str(llvm_install / "bin/clang-23"),
        # runtime_preflight executes the paired adapter in this process rather than through
        # AD.child_environment. Pin the same warm measured-only protocol here so preflight and
        # campaign cannot validate different cache conditions.
        "MERLIN_CACHE_STATE": AD.MEASUREMENT_CACHE_CONDITION,
        "PYTHONPATH": str(snapshot / "src"),
        "MERLIN_PERF_CAMPAIGN_FANOUT": str(manifest["trial_workers"]),
        "MERLIN_PERF_SWEEP_WORKERS": str(manifest["sim_workers"]),
        "MERLIN_REQUIRED_RTL_ENGINE": "gsim",
    }


def prepare(root: Path, specification: dict) -> Path:
    spec = dict(specification)
    suite_id = str(spec.pop("suite_id"))
    AD._safe(suite_id, label="suite id")
    members = spec.pop("members")
    campaign_workers, trial_workers = spec.pop("campaign_workers", 1), spec.pop("trial_workers", 3)
    for name, count in (("campaign_workers", campaign_workers), ("trial_workers", trial_workers)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SuiteError(f"{name} must be a positive integer")
    if trial_workers > len(AD.TRIALS):
        raise SuiteError("trial_workers exceeds the three predeclared trials")
    spec = {**DEFAULTS, **spec}
    # Resolve outside the Chia environment once, preserving the virtualenv entrypoint.
    spec["driver_python"] = str(_input_path(chia_bridge.driver_python(), preserve_entrypoint=True))
    if isinstance(spec["sim_workers"], bool) or not isinstance(spec["sim_workers"], int) or spec["sim_workers"] < 1:
        raise SuiteError("sim_workers must be a positive integer")
    for key in PATH_FIELDS:
        if spec.get(key) is not None:
            spec[key] = str(_input_path(spec[key], preserve_entrypoint=key == "chia_python"))
    if spec.get("waive_functional_gsim_certificate") or not spec.get("functional_gsim_certificate"):
        raise SuiteError("this suite requires exact or stratified functional evidence, not a waiver")
    for key in ("gsim_certificate", "functional_gsim_certificate"):
        digest = SNAP.sha_file(Path(spec[key]))
        if spec.get(key + "_sha256", digest) != digest:
            raise SuiteError(f"{key} digest mismatch")
        spec[key + "_sha256"] = digest
    # Fail cheaply before copying the source/corpus or starting orchestration.
    groups = claim_groups(
        Path(spec["descriptor"]), Path(spec["gsim_certificate"]), spec["gsim_certificate_sha256"], members
    )
    root = root.absolute()
    root.relative_to(artifacts_dir().resolve())
    if root.exists() or root.is_symlink():
        raise SuiteError("prepare requires a new suite root; use run to resume a sealed suite")
    root.mkdir(parents=True, mode=0o700)
    snapshot = root / "source"
    target_name = load_target_experiment(Path(spec["descriptor"])).target
    provider = SNAP.selected_provider(target_name)
    receipt = SNAP.create(
        repo_root(),
        snapshot,
        output_root=out_dir(),
        target_name=target_name,
        provider=provider,
        **snapshot_layout(repo_root(), target_name=target_name, provider=provider),
        declared_inputs={
            key: Path(spec[key]) for key in ("descriptor", "rtl_facts", "perf_profile", "telemetry_price_table")
        },
    )
    snapshot_record = SNAP.verify(snapshot)
    # Only corpus/source inputs move into the snapshot. Certificates retain their evidence roots:
    # their provenance verifier follows their sealed declarations and exact capture receipts.
    for key in ("descriptor", "rtl_facts", "perf_profile", "telemetry_price_table"):
        spec[key] = str(SNAP.remap_input(snapshot, snapshot_record, Path(spec[key]), name=key))
    campaigns = []
    for index, group in enumerate(groups):
        identity = f"{suite_id}__claim_{index:02d}"
        config = {
            **spec,
            "experiment_id": identity,
            "root": str(root / identity),
            "perf_capsules": ",".join(group["members"]),
            "perf_families": "all",
            "source_root": str(snapshot),
            "contract_root": str(snapshot / "src/merlin/resources/contracts"),
            "functional_runs_root": str(runs_root(target_name, "capsule-bench")),
            "stage_root": str(runs_root(target_name, "perf-bench") / "agent_stages"),
            "measurement_root": str(runs_root(target_name, "perf-bench")),
            "holdout_catalog": str(snapshot / "experiments/catalog.yaml"),
            "core_package_root": str(snapshot / "src/merlin"),
            "experiments_package_root": str(snapshot / "src/merlin/experiments"),
            "experiments_namespace_root": str(snapshot / "src/merlin"),
            "chia_wrapper": str(snapshot / IMPLEMENTATION_ROOT / "chia_envelope.py"),
            "suite": STUDY_POLICY["suite"],
        }
        config_from(config)  # reject misspelled/unhandled configuration, before sealing
        campaigns.append({**group, "config": config})
    manifest = {
        "schema": SCHEMA,
        "suite_id": suite_id,
        "output_root": str(out_dir()),
        "source_snapshot_sha256": SNAP.sha_file(receipt),
        "campaigns": campaigns,
        "evaluation_boundary": evaluation_boundary(snapshot, snapshot_record),
        "campaign_workers": campaign_workers,
        "trial_workers": trial_workers,
        "sim_workers": spec["sim_workers"],
        "trials": list(AD.TRIALS),
        "measurement": {
            "requested_cache_condition": AD.MEASUREMENT_CACHE_CONDITION,
            "cache_protocol": "one_unmeasured_predecessor",
            "cycle_window": STUDY_POLICY["cycle_window"],
            "recorded_metric": "cycles",
        },
        "selection": "all campaigns, all trials; separate claims, no best-of",
    }
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
    if (
        result.get("schema") != AD.SCHEMA
        or result.get("status") != "GO"
        or result.get("declaration", {}).get("experiment_id") != campaign["config"]["experiment_id"]
    ):
        raise SuiteError("campaign completion has a foreign identity or non-admitted status")
    return {"path": str(path), "sha256": SNAP.sha_file(path)}


def run(root: Path, *, managed_native_endpoint: Path | None = None) -> list[dict]:
    if managed_native_endpoint is None:
        raise SuiteError(
            "run requires --managed-native-endpoint for an explicitly provisioned same-snapshot supervisor"
        )
    endpoint = managed_native_endpoint.absolute()
    # The open descriptor is inherited by each wrapper, retaining exclusion if this parent dies.
    with (root / "launcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuiteError("another launcher owns this suite") from exc
        manifest = load(root)
        env = environment(root, manifest)
        snapshot = root / "source"
        from merlin.development.backend.providers.agent_bridge import bind_frozen_proxy_config

        receipt = SNAP.verify(snapshot)
        require_managed_wrapper(snapshot)
        seal, _ = SNAP.load_seal(snapshot, "snapshot")
        from merlin.experiments.execution.chia_native import Session

        # Admission only: each wrapper owns its own task reservations and receipts.
        # This deployment address is not part of the scientific campaign configuration.
        with Session(endpoint, expected_source={"path": str(seal.absolute()), "sha256": SNAP.sha_file(seal)}):
            pass
        proxy_config_identity = bind_frozen_proxy_config(snapshot, receipt, env)
        # In a FRESH process: never let modules already imported from the live tree pass preflight.
        check = subprocess.run(
            python_command(
                snapshot,
                [
                    env["MERLIN_EXPERIMENT_PYTHON"],
                    "-m",
                    "merlin.development.backend.optimization.suite",
                    "preflight",
                    "--root",
                    str(root),
                ],
                verifier_source=Path(SNAP.__file__),
            ),
            cwd=snapshot,
            env=env,
        )
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
            argv = [
                python,
                "-m",
                "merlin.development.backend.optimization.chia_envelope_cli",
                "--driver-python",
                config["driver_python"],
                "--cwd",
                str(snapshot),
                "--suite",
                config["suite"],
                "--target",
                load_target_experiment(config["descriptor"]).target,
                "--managed-native-endpoint",
                str(endpoint),
                "--orchestration-run-id",
                identity + "__chia",
                "--codex-slots",
                "1",
                "--gsim-slots",
                str(config["sim_workers"]),
                "--",
                *coordinator_args(config),
            ]
            transport = python_command(snapshot, argv, verifier_source=Path(SNAP.__file__))
            SNAP.seal(
                attempt,
                "launch",
                {
                    "argv": argv,
                    "transport_argv": transport,
                    "suite_id": manifest["suite_id"],
                    "proxy_config": proxy_config_identity,
                },
            )
            with (attempt / "console.log").open("xb") as log:
                result = subprocess.run(
                    transport, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),)
                )
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
            SNAP.seal(root, "completion", {"schema": SCHEMA, "campaigns": results, "status": "all_campaigns_sealed"})
        return results


def preflight(root: Path) -> None:
    manifest = load(root)
    if repo_root().resolve() != (root / "source").resolve():
        raise SuiteError("suite preflight must run from its sealed source snapshot")
    for name, module in tuple(sys.modules.items()):
        if (
            name == "merlin"
            or name.startswith("merlin.")
            or name == "merlin.experiments"
            or name.startswith("merlin.experiments.")
            or name in ("_pbcommon", "_common", "perf_agent_stage", "run_agentic_perf_experiment")
        ):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root / "source"):
                raise SuiteError(f"live module leaked into snapshot preflight: {name}")
    pending = []
    for campaign in manifest["campaigns"]:
        config = config_from(campaign["config"])
        groups = claim_groups(
            config.descriptor, config.gsim_certificate, config.gsim_certificate_sha256, campaign["members"]
        )
        if len(groups) != 1 or groups[0] != {key: campaign[key] for key in groups[0]}:
            raise SuiteError("claim cohort/contract changed since suite sealing")
        result = AD.preflight(config, heldout_certificate_provider_available=True)
        print(
            json.dumps({"family": campaign["family"], "status": result["status"], "blockers": result["blockers"]}),
            flush=True,
        )
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
            print(
                json.dumps({"experiment_id": config.experiment_id, "runtime_preflight": str(receipt), "status": "GO"}),
                flush=True,
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "preflight"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--managed-native-endpoint",
        type=Path,
        help="run only: provisioned local supervisor started from this suite's sealed source",
    )
    args = parser.parse_args(argv)
    root = args.root.absolute()
    try:
        if args.action == "prepare":
            if args.config is None:
                parser.error("prepare requires --config")
            print(prepare(root, json.loads(args.config.read_text())))
        elif args.action == "run":
            print(json.dumps(run(root, managed_native_endpoint=args.managed_native_endpoint)))
        elif args.action == "preflight":
            preflight(root)
        else:
            manifest = load(root)
            print(
                json.dumps(
                    [{"family": c["family"], "completion": completion(c)} for c in manifest["campaigns"]], indent=2
                )
            )
    except (SuiteError, SNAP.SnapshotError, AD.ExperimentError, P2_CONTRACTS.StageGateError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
