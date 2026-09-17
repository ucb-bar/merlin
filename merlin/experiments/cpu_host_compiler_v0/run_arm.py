#!/usr/bin/env python3
"""Run one CPU-host compiler arm through Chia/Codex and reconcile it into AET.

No paid work occurs without ``--live``. A live arm requires a fully GO preflight, including the
external deterministic grader and K1 probe. Codex is placed inside a deny-by-default outer bwrap;
the Ray/Chia driver remains outside so raw telemetry survives a sandbox or agent failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from merlin.benchharness.chia_bridge import chia_get, chia_run, driver_python, require_chia
from merlin.benchharness.host_agent import (
    audit_staged_inputs,
    create_compiler_seal,
    prepare_isolated_codex_home,
    record_codex_trajectory,
    stage_host_workspace,
    verify_trusted_search,
    write_codex_bwrap_wrapper,
)
from merlin.common.paths import build_dir, repo_root
from merlin.compare.host_experiment import HostExperimentSpec


DEFAULT_SPEC = repo_root() / "merlin/experiments/cpu_host_compiler_v0/experiment.yaml"


def _authorization_cell(spec: HostExperimentSpec, spec_path: Path, arm_id: str,
                        seed: int, run_id: str | None,
                        claim_path: Path | None) -> tuple[Path, Path]:
    if run_id is None or claim_path is None:
        raise ValueError("live arm requires its launcher-bound run ID and protocol claim")
    expected_claim = spec._repo_path(spec.telemetry["output_layout"]).resolve() / \
        ".protocol_claims" / f"{spec.freeze['protocol_inputs_sha256']}.json"
    claim_path = claim_path.resolve()
    if claim_path != expected_claim or not claim_path.is_file():
        raise ValueError("live arm protocol claim is absent or outside the canonical claim path")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    campaign = str(claim.get("campaign_run_id", ""))
    exclusion_path = expected_claim.parent.parent / ".campaign_exclusions" / f"{campaign}.json"
    if exclusion_path.is_file():
        exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
        if (exclusion.get("authority") != "controller_campaign_exclusion_v1" or
                exclusion.get("campaign_run_id") != campaign or
                exclusion.get("protocol_inputs_sha256") !=
                spec.freeze.get("protocol_inputs_sha256") or
                exclusion.get("excluded_from_arm_outcomes") is not True or
                exclusion.get("excluded_from_promotion") is not True):
            raise ValueError("campaign exclusion tombstone is malformed; refusing live arm")
        raise ValueError("live arm belongs to a controller-excluded protocol-invalid campaign")
    cell = next((row for row in spec.agent["launch_plan"]
                 if row["arm"] == arm_id and int(row["seed"]) == seed), None)
    if cell is None:
        raise ValueError("live arm/seed is not one frozen launch cell")
    expected_run_id = (f"{campaign}__{arm_id}__r{int(cell['repeat']):02d}__seed{seed:03d}")
    if (claim.get("version") != 1 or claim.get("status") != "bound" or not campaign or
            claim.get("protocol_inputs_sha256") != spec.freeze.get("protocol_inputs_sha256") or
            claim.get("environment_manifest_sha256") != spec.environment.get("sha256") or
            claim.get("analysis_plan_sha256") != spec.analysis.get("sha256") or
            claim.get("spec_path") != str(spec_path.resolve()) or
            claim.get("spec_sha256") != hashlib.sha256(spec_path.read_bytes()).hexdigest() or
            run_id != expected_run_id):
        raise ValueError("live arm identity does not match the one-shot frozen protocol claim")
    cells = claim_path.with_name(f"{claim_path.stem}.cells")
    authorized = cells / f"{int(cell['ordinal']):02d}.authorized.json"
    consumed = cells / f"{int(cell['ordinal']):02d}.consumed.json"
    if not authorized.is_file() or consumed.exists():
        raise ValueError("frozen arm cell was not authorized or has already been consumed")
    receipt = json.loads(authorized.read_text(encoding="utf-8"))
    expected_receipt = {
        "version": 1, "status": "authorized",
        "protocol_inputs_sha256": spec.freeze["protocol_inputs_sha256"],
        "environment_manifest_sha256": spec.environment["sha256"],
        "analysis_plan_sha256": spec.analysis["sha256"],
        "campaign_run_id": campaign, "ordinal": int(cell["ordinal"]), "arm": arm_id,
        "repeat": int(cell["repeat"]), "seed": seed, "run_id": run_id,
    }
    if receipt != expected_receipt:
        raise ValueError("frozen arm cell authorization bytes differ from the launch plan")
    return authorized, consumed


def _consume_authorization_cell(authorized: Path, consumed: Path) -> None:
    """Atomically linearize cell consumption against controller exclusion.

    The receipt rename is the consumption linearization point.  Cooperating controller cancellation
    code must take ``_authorization_lifecycle_lock`` while cancelling receipts and publishing its
    exclusion tombstone.  The post-rename check additionally makes a non-cooperating tombstone that
    races between the first check and ``os.replace`` fail closed: the receipt is quarantined as
    cancelled and is never returned to the authorized state.
    """
    with _authorization_lifecycle_lock(authorized):
        receipt = json.loads(authorized.read_text(encoding="utf-8"))
        campaign = str(receipt.get("campaign_run_id", ""))
        exclusion = (
            authorized.parent.parent.parent / ".campaign_exclusions" / f"{campaign}.json"
        )
        if exclusion.exists() or exclusion.is_symlink():
            raise ValueError(
                "campaign was controller-excluded before cell authorization consumption")
        try:
            os.replace(authorized, consumed)
        except FileNotFoundError as exc:
            raise ValueError("frozen arm cell authorization was consumed concurrently") from exc
        if exclusion.exists() or exclusion.is_symlink():
            cancelled = authorized.with_name(
                authorized.name.removesuffix(".authorized.json") + ".cancelled.json")
            try:
                os.link(consumed, cancelled)
            except FileExistsError:
                if (not cancelled.is_file() or
                        hashlib.sha256(cancelled.read_bytes()).digest() !=
                        hashlib.sha256(consumed.read_bytes()).digest()):
                    raise ValueError(
                        "concurrent campaign cancellation produced a conflicting cell receipt")
            consumed.unlink(missing_ok=True)
            raise ValueError(
                "campaign was controller-excluded during cell authorization consumption")


@contextmanager
def _authorization_lifecycle_lock(authorized: Path):
    """Serialize one protocol claim's receipt/exclusion lifecycle across processes."""
    cells = authorized.parent
    if not cells.name.endswith(".cells"):
        raise ValueError("authorization receipt is outside a protocol claim cells directory")
    claim_stem = cells.name.removesuffix(".cells")
    lock_root = cells.parent / ".lifecycle_locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_root / f"{claim_stem}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield lock_path
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _dispatch_codex_prompt(llm, prompt: str):
    """Dispatch through Chia's explicit-self remote method contract.

    ``@ChiaFunction`` methods intentionally do not bind the instance for ``chia_remote``; Chia's
    documented call form is ``method.chia_remote(instance, ...)``.
    """
    return llm.prompt.chia_remote(llm, prompt, tools=[])


def _verify_staged_workspace_identity(preflight, arm_id: str, staged) -> None:
    expected = preflight.evidence.get("arm_workspace_inputs", {}).get(arm_id, {}).get(
        "input_lock_sha256")
    if expected != staged.input_lock_sha256:
        raise RuntimeError(
            f"live staged workspace for {arm_id} differs from frozen preflight: "
            f"expected={expected!r} actual={staged.input_lock_sha256!r}")


def _prompt(spec: HostExperimentSpec, arm) -> str:
    task = spec._repo_path(spec.task).read_text(encoding="utf-8").rstrip()
    capabilities = "\n".join(f"- {name}" for name in sorted(arm.capabilities))
    return f"""{task}

## Experimental arm

Arm: {arm.id}
Treatment: {arm.treatment}

Available capabilities:
{capabilities}

If `deterministic_candidate_search` is available, run `/usr/bin/python3 -B policy/beam_search.py`
with output fixed at workspace-local `scratch/search_work` (outside `submission/`) and evaluator
`/usr/bin/python3 -B policy/trusted_evaluator.py`. Use `scratch/` for every transient build or test
artifact. After convergence, create `submission/search/` containing exactly byte-copies of
`scratch/search_work/search_record.json` and `scratch/search_work/selected_policy.json`; copy the same
selected policy byte-for-byte to the manifest-declared policy. Do not place build/source dependencies
under `submission/search`. The driver independently replays the private K1 ledger; a fabricated record,
another evaluator, or a changed compiler cannot pass.

Work only inside this workspace. Preserve the supplied contracts and public corpus. Put the complete
compiler package and machine-readable policy under `submission/`. You may use `train.jsonl` to choose
candidates and `validation.jsonl` only for deterministic promotion. Do not seek files outside the
workspace. Do not encode capsule IDs, shapes, or operation-instance identities as dispatch keys.
Before package-level tests, run your descriptor parser against all six immutable examples in
`contracts/descriptor_fixtures/`; `contracts/capsule_descriptor.py` is the exact public renderer.
"""


def _run_grader(spec: HostExperimentSpec, workspace: Path, run_dir: Path,
                *, trusted_search_seal: Path | None = None,
                compiler_seal: Path | None = None) -> dict:
    grader = spec._repo_path(spec.grading["grader"])
    corpus = spec._repo_path(spec.development_corpus["materialized_capsules"]).resolve()
    output = run_dir / "metrics" / "grader_result.json"
    command = [
        driver_python(), str(grader), "--submission", str(workspace / "submission"),
        "--target-contract", str(spec._repo_path(spec.target_contract)),
        "--dialect-plan", str(spec._repo_path(spec.dialect_plan)),
        "--train", str(corpus / "public" / "train.jsonl"),
        "--validation", str(corpus / "public" / "validation.jsonl"),
        "--heldout", str(corpus / "sealed" / "heldout.jsonl"),
        "--output", str(output),
    ]
    if trusted_search_seal is not None:
        command += ["--trusted-search-seal", str(trusted_search_seal)]
    if compiler_seal is not None:
        command += ["--compiler-seal", str(compiler_seal)]
    started = time.monotonic()
    proc = subprocess.run(command, cwd=repo_root(), capture_output=True, text=True, timeout=7200)
    elapsed = time.monotonic() - started
    (run_dir / "logs" / "grader.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "logs" / "grader.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if output.is_file():
        result = json.loads(output.read_text(encoding="utf-8"))
    else:
        result = {"version": 1, "status": "error", "returncode": proc.returncode,
                  "reason": "grader did not produce its required JSON result"}
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # The grader's own result is the durable authority for the driver-observed elapsed time; the
    # campaign finalizer rejects a summary that differs from this value.
    result["wall_seconds"] = elapsed
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"wall_seconds": elapsed, "returncode": proc.returncode, "result": result,
            "command": command}


def _promotion_not_verified(run_dir: Path, reason: str, evidence: dict) -> dict:
    output = run_dir / "metrics" / "grader_result.json"
    result = {"version": 1, "status": "not_run", "wall_seconds": 0.0,
              "reason": reason, "promotion_evidence": evidence}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"wall_seconds": 0.0, "returncode": 2, "result": result, "command": []}


def _classify_terminal_outcome(*, agent_success: bool, input_audit_ok: bool,
                               aet_reconciled: bool, search_required: bool,
                               search_status: str, search_failure_class: str | None,
                               compiler_seal_status: str,
                               compiler_seal_failure_class: str | None,
                               agent_failure_class: str | None,
                               grader_returncode: int, grader_status: str | None,
                               grader_failure_class: str | None) -> str:
    """Separate treatment outcomes from failures of the measurement controller itself."""
    if not aet_reconciled or not input_audit_ok:
        return "harness_invalid"
    if (search_required and search_status != "pass" and
            search_failure_class == "harness_invalid"):
        return "harness_invalid"
    if not agent_success:
        return ("treatment_agent_fail" if agent_failure_class == "treatment_agent_fail"
                else "harness_invalid")
    if search_required and search_status != "pass":
        return (search_failure_class if search_failure_class in {
            "treatment_search_fail", "treatment_build_fail", "treatment_agent_fail"}
            else "harness_invalid")
    if compiler_seal_status != "sealed":
        return ("treatment_agent_fail" if
                compiler_seal_failure_class == "treatment_agent_fail" else "harness_invalid")
    if grader_returncode == 0 and grader_status == "pass":
        return "graded_pass"
    if grader_returncode == 1 and grader_status == "fail":
        return "graded_fail"
    if (grader_returncode == 1 and grader_status in {
            "treatment_build_fail", "treatment_agent_fail"} and
            grader_failure_class == grader_status):
        return grader_status
    return "harness_invalid"


def _agent_failure_class(run_result) -> str | None:
    """Classify only reconciled Codex process outcomes; telemetry authority is checked separately."""
    if getattr(run_result, "status", None) == "completed":
        return None
    attempts = list(getattr(run_result, "attempts", ()) or ())
    failure_classes = {str(getattr(attempt, "failure_class", "") or "")
                       for attempt in attempts}
    controller_or_backend = {
        "AuthenticationError", "BillingError", "RateLimitError", "ServerError",
        "InvalidRequestError",
    }
    if failure_classes & controller_or_backend:
        return "harness_invalid"
    sandbox_bootstrap_markers = (
        "bwrap: setting up uid map:",
        "bwrap: Creating new namespace failed:",
        "bwrap: No permissions to create new namespace",
    )
    if any(any(marker in str(getattr(attempt, "retry_reason", "") or "")
               for marker in sandbox_bootstrap_markers)
           for attempt in attempts):
        return "harness_invalid"
    return "treatment_agent_fail" if attempts else "harness_invalid"


def _preview(spec: HostExperimentSpec, arm_id: str) -> dict:
    arm = next((candidate for candidate in spec.arms if candidate.id == arm_id), None)
    if arm is None:
        raise ValueError(f"unknown arm {arm_id!r}; choose one of {[a.id for a in spec.arms]}")
    preflight = spec.preflight(
        check_environment=True, probe_board=False, require_frozen=True)
    return {"mode": "preview", "paid_work": False, "arm": arm.id,
            "treatment": arm.treatment, "capabilities": sorted(arm.capabilities),
            "preflight": preflight.to_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--authorization-claim", type=Path)
    parser.add_argument("--live", action="store_true", help="authorize one paid Codex arm")
    args = parser.parse_args(argv)

    spec = HostExperimentSpec.from_yaml(args.spec)
    arm = next((candidate for candidate in spec.arms if candidate.id == args.arm), None)
    if arm is None:
        parser.error(f"unknown arm {args.arm!r}; choose one of {[a.id for a in spec.arms]}")
    if not args.live:
        print(json.dumps(_preview(spec, arm.id), indent=2))
        return 0
    if spec.status != "protocol_frozen":
        print("REFUSING paid run: live arm requires status exactly protocol_frozen")
        return 2
    try:
        authorized_cell, consumed_cell = _authorization_cell(
            spec, args.spec, arm.id, args.seed, args.run_id, args.authorization_claim)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSING paid run: {exc}")
        return 2

    require_chia()
    preflight = spec.preflight(
        check_environment=True, probe_board=True, require_frozen=True)
    if not preflight.ready:
        print(json.dumps(preflight.to_dict(), indent=2))
        print("REFUSING paid run: CPU-host experiment preflight is NO_GO")
        return 2
    try:
        _consume_authorization_cell(authorized_cell, consumed_cell)
    except ValueError as exc:
        print(f"REFUSING paid run: {exc}")
        return 2

    from chia.models.codex import CodexLLM

    # This driver-only monotonic interval is the authority for total wall time.  It is written as
    # start/end ticks before the mutable summary is updated, so promotion cannot inflate wall time.
    wall_start_ns = time.monotonic_ns()
    with chia_run(
        suite="cpu-host-compiler", method=arm.id, target="k1_cpu", seed=args.seed,
        run_id=args.run_id,
        extra={"experiment": spec.label, "arm": arm.id, "model": spec.agent["model"],
               "billing_mode": spec.agent["billing"],
               "environment_manifest_sha256": str(spec.environment["sha256"]),
               "analysis_plan_sha256": str(spec.analysis["sha256"]),
               "provider_sampling_seeded": False,
               "spec": str(Path(args.spec).resolve())},
        ray_resources={"codex_slots": 1, "k1": 1},
    ) as run:
        (run.run_dir / "contracts" / "preflight.json").write_text(
            json.dumps(preflight.to_dict(), indent=2) + "\n", encoding="utf-8")
        workspace_root = build_dir() / "cpu-host-compiler" / run.run_id / "workspace"
        corpus = spec._repo_path(spec.development_corpus["materialized_capsules"]).resolve()
        staged = stage_host_workspace(
            workspace_root, task_path=spec._repo_path(spec.task),
            target_contract_path=spec._repo_path(spec.target_contract),
            dialect_plan_path=spec._repo_path(spec.dialect_plan), public_corpus_dir=corpus / "public",
            search_space_path=spec._repo_path(spec.search["space"]),
            search_runner_path=spec._repo_path(spec.search["runner"]),
            trusted_evaluator_path=spec._repo_path(spec.search["trusted_evaluator"]),
            submission_contract_path=spec._repo_path(spec.grading["submission_contract"]),
            arm_id=arm.id, capabilities=arm.capabilities, treatment=arm.treatment)
        _verify_staged_workspace_identity(preflight, arm.id, staged)
        prompt = _prompt(spec, arm)
        (run.run_dir / "contracts" / "prompt.md").write_text(prompt, encoding="utf-8")
        (run.run_dir / "contracts" / "workspace_input_lock.json").write_text(
            json.dumps(staged.input_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        agent_dir = run.run_dir / "agent"
        raw_dir = agent_dir / "raw"
        raw_dir.mkdir(parents=True)
        codex_home, auth = prepare_isolated_codex_home(agent_dir / "codex_home")
        codex_binary = shutil.which("codex")
        if codex_binary is None or auth is None:
            raise RuntimeError("Codex binary or subscription credential disappeared after preflight")
        wrapper = write_codex_bwrap_wrapper(
            build_dir() / "cpu-host-compiler" / run.run_id / "codex-bwrap",
            workspace=staged.path, codex_home=codex_home, codex_binary=codex_binary)

        llm = CodexLLM(
            model=str(spec.agent["model"]),
            reasoning_effort=str(spec.agent["reasoning_effort"]),
            timeout_seconds=int(spec.agent["active_wall_seconds_per_arm"]), retries=1,
            codex_bin=str(wrapper), work_dir=str(staged.path), raw_event_dir=str(raw_dir),
            capture_arrival_timestamps=True, resume_session=False,
            dangerously_bypass_approvals_and_sandbox=True, external_sandbox=True,
            skip_git_repo_check=True,
        )
        search_required = "deterministic_candidate_search" in arm.capabilities
        broker = None
        broker_logs = []
        ledger = run.run_dir / "metrics" / "trusted_search_ledger"
        if search_required:
            stdout = (run.run_dir / "logs" / "trusted_search_broker.stdout.log").open("w")
            stderr = (run.run_dir / "logs" / "trusted_search_broker.stderr.log").open("w")
            broker_logs = [stdout, stderr]
            broker = subprocess.Popen([
                driver_python(), str(spec._repo_path(spec.search["trusted_broker"])),
                "--workspace", str(staged.path), "--public", str(corpus / "public"),
                "--space", str(spec._repo_path(spec.search["space"])),
                "--runner", str(spec._repo_path(spec.search["runner"])),
                "--grader", str(spec._repo_path(spec.grading["grader"])),
                "--ledger", str(ledger),
            ], cwd=repo_root(), stdout=stdout, stderr=stderr)
            ready = staged.path / ".trusted_search_channel" / "READY"
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not ready.is_file() and broker.poll() is None:
                time.sleep(0.1)
            if not ready.is_file():
                if broker.poll() is None:
                    broker.terminate()
                    broker.wait(timeout=10)
                for stream in broker_logs:
                    stream.close()
                raise RuntimeError("trusted search broker failed to become ready")
        try:
            result = chia_get(_dispatch_codex_prompt(llm, prompt))
        finally:
            if broker is not None:
                channel = staged.path / ".trusted_search_channel"
                channel.mkdir(parents=True, exist_ok=True)
                (channel / "STOP").write_text("stop\n", encoding="utf-8")
                try:
                    broker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    broker.terminate()
                    broker.wait(timeout=10)
                channel_archive = run.run_dir / "logs" / "trusted_search_channel"
                shutil.copytree(channel, channel_archive, symlinks=False)
                (run.run_dir / "contracts" / "trusted_search_broker_terminal.json").write_text(
                    json.dumps({
                        "version": 1, "authority": "driver_process_wait",
                        "returncode": broker.returncode,
                        "channel_archive": str(channel_archive.resolve()),
                        "ledger": str(ledger.resolve()),
                    }, indent=2) + "\n", encoding="utf-8")
            for stream in broker_logs:
                stream.close()
        (agent_dir / "final.md").write_text(result.result or "", encoding="utf-8")
        (agent_dir / "run_result.json").write_text(
            json.dumps(result.run_result.as_dict(), indent=2) + "\n", encoding="utf-8")

        aet = record_codex_trajectory(
            run_result=result.run_result, run_id=run.run_id, model=str(spec.agent["model"]),
            run_dir=run.run_dir, logger=run.handle.logger)
        input_audit = audit_staged_inputs(staged)
        (run.run_dir / "contracts" / "workspace_input_audit.json").write_text(
            json.dumps(input_audit, indent=2) + "\n", encoding="utf-8")
        if search_required:
            search_seal = verify_trusted_search(
                workspace=staged.path, ledger=ledger,
                space_path=spec._repo_path(spec.search["space"]),
                runner_path=spec._repo_path(spec.search["runner"]),
                replay_path=spec._repo_path(spec.search["trusted_replay"]),
                train_path=corpus / "public" / "train.jsonl",
                validation_path=corpus / "public" / "validation.jsonl")
        else:
            search_seal = {"version": 1, "status": "not_required", "arm": arm.id}
        search_seal_path = run.run_dir / "contracts" / "trusted_search_seal.json"
        search_seal_path.write_text(
            json.dumps(search_seal, indent=2) + "\n", encoding="utf-8")
        if not result.success:
            compiler_seal = {
                "version": 1, "status": "not_run",
                "failure_class": "treatment_agent_fail",
                "reason": "reconciled Codex attempt did not complete",
            }
        elif search_seal["status"] in {"pass", "not_required"}:
            try:
                compiler_seal = create_compiler_seal(
                    workspace=staged.path, search_seal=search_seal)
            except Exception as exc:
                compiler_seal = {"version": 1, "status": "fail",
                                 "failure_class": "treatment_agent_fail",
                                 "reason": f"{type(exc).__name__}: {exc}"}
        else:
            compiler_seal = {"version": 1, "status": "not_run",
                             "reason": "trusted search did not verify"}
        compiler_seal_path = run.run_dir / "contracts" / "compiler_seal.json"
        compiler_seal_path.write_text(
            json.dumps(compiler_seal, indent=2) + "\n", encoding="utf-8")
        compiler_archive = run.run_dir / "artifacts" / "compiler_submission"
        if compiler_seal["status"] == "sealed":
            shutil.copytree(staged.path / "submission", compiler_archive, symlinks=False)
        grader = (_run_grader(
                      spec, staged.path, run.run_dir,
                      trusted_search_seal=search_seal_path if search_required else None,
                      compiler_seal=compiler_seal_path)
                  if compiler_seal["status"] == "sealed"
                  else _promotion_not_verified(
                      run.run_dir, "trusted search/compiler seal did not verify",
                      {"search": search_seal, "compiler": compiler_seal}))
        run.metrics.log_scalar("agent/active_wall_s", result.run_result.active_wall_s, 0)
        run.metrics.log_scalar(
            "search/trusted_wall_s", search_seal.get("trusted_broker_wall_ns", 0) / 1e9, 0)
        run.metrics.log_scalar("grader/wall_s", grader["wall_seconds"], 0)
        wall_end_ns = time.monotonic_ns()
        total_wall_seconds = (wall_end_ns - wall_start_ns) / 1e9
        timing_path = run.run_dir / "metrics" / "driver_wall_timing.json"
        timing_path.write_text(json.dumps({
            "version": 1, "authority": "driver_monotonic_ns",
            "start_monotonic_ns": wall_start_ns, "end_monotonic_ns": wall_end_ns,
            "wall_seconds": total_wall_seconds,
        }, indent=2) + "\n", encoding="utf-8")
        run.metrics.log_scalar("run/wall_s", total_wall_seconds, 0)
        terminal_class = _classify_terminal_outcome(
            agent_success=bool(result.success), input_audit_ok=bool(input_audit["ok"]),
            aet_reconciled=bool(aet["reconciliation"]["ok"]),
            agent_failure_class=_agent_failure_class(result.run_result),
            search_required=search_required, search_status=str(search_seal["status"]),
            search_failure_class=search_seal.get("failure_class"),
            compiler_seal_status=str(compiler_seal["status"]),
            compiler_seal_failure_class=compiler_seal.get("failure_class"),
            grader_returncode=int(grader["returncode"]),
            grader_status=grader["result"].get("status"),
            grader_failure_class=grader["result"].get("failure_class"))
        outcome = {
            "version": 1, "run_id": run.run_id, "arm": arm.id,
            "terminal_class": terminal_class,
            "paper_evidence_eligible": terminal_class in {"graded_pass", "graded_fail"},
            "promotion_eligible": terminal_class == "graded_pass",
            "checks": {
                "agent_success": bool(result.success),
                "agent_failure_class": _agent_failure_class(result.run_result),
                "workspace_input_audit": bool(input_audit["ok"]),
                "aet_reconciled": bool(aet["reconciliation"]["ok"]),
                "trusted_search_status": search_seal["status"],
                "compiler_seal_status": compiler_seal["status"],
                "compiler_seal_failure_class": compiler_seal.get("failure_class"),
                "grader_returncode": grader["returncode"],
                "grader_status": grader["result"].get("status"),
                "grader_failure_class": grader["result"].get("failure_class"),
            },
        }
        (run.run_dir / "contracts" / "terminal_outcome.json").write_text(
            json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
        run.summary.update({
            "arm": arm.id, "agent_success": bool(result.success),
            "agent_failure_class": _agent_failure_class(result.run_result),
            "workspace_inputs_unchanged": input_audit["ok"],
            "aet_reconciled": aet["reconciliation"]["ok"],
            "active_wall_seconds": result.run_result.active_wall_s,
            "grader_wall_seconds": grader["wall_seconds"],
            "grader_returncode": grader["returncode"],
            "grader_status": grader["result"].get("status"),
            "trusted_search_status": search_seal["status"],
            "compiler_seal_status": compiler_seal["status"],
            "compiler_archive": str(compiler_archive.resolve())
            if compiler_archive.is_dir() else None,
            "trusted_search_wall_seconds": (
                search_seal.get("trusted_broker_wall_ns", 0) / 1e9),
            "wall_seconds": total_wall_seconds,
            "billing_mode": spec.agent["billing"],
            "terminal_class": terminal_class,
        })
        print(f"run_dir: {run.run_dir}")
        return 0 if terminal_class == "graded_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
