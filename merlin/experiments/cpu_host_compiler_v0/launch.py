#!/usr/bin/env python3
"""Preflight and launch all four CPU-host compiler arms sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from merlin.common.artifacts import finish_run, start_run
from merlin.common.paths import repo_root
from merlin.compare.host_experiment import HostExperimentSpec


HERE = repo_root() / "merlin/experiments/cpu_host_compiler_v0"


def _requalify_block_boundary(spec: HostExperimentSpec, *, block: int,
                              first_ordinal: int) -> tuple[object, dict]:
    """Enforce and retain the washout that makes cross-block transitions out of scope."""
    environment = dict(spec.search_space_config()["board_environment"])
    attempts = int(environment["settle_attempts"])
    interval = float(environment["settle_interval_seconds"])
    # A real elapsed quiet interval is mandatory even when the first state probe is already ready.
    time.sleep(interval)
    gates = []
    last = None
    for attempt in range(attempts):
        last = spec.preflight(check_environment=True, probe_board=True, require_frozen=True)
        gates.append(last.to_dict())
        if last.ready:
            break
        if attempt + 1 < attempts:
            time.sleep(interval)
    assert last is not None
    receipt = {
        "version": 1,
        "authority": "frozen_k1_board_environment_gate",
        "block": block,
        "first_ordinal": first_ordinal,
        "mandatory_washout_seconds": interval,
        "stabilization_attempt_limit": attempts,
        "board_environment": environment,
        "attempts": gates,
        "qualifying_attempt_index": len(gates) - 1 if last.ready else None,
        "ready": last.ready,
    }
    return last, receipt


def _claim_protocol_once(claim_root: Path, *, protocol_sha256: str,
                         environment_manifest_sha256: str,
                         analysis_plan_sha256: str, spec_path: Path) -> Path:
    """Atomically consume one frozen protocol; a failed/negative campaign is never relaunched."""
    claim_root = claim_root.resolve() / ".protocol_claims"
    claim_root.mkdir(parents=True, exist_ok=True)
    claim = claim_root / f"{protocol_sha256}.json"
    payload = {
        "version": 1, "status": "reserved", "protocol_inputs_sha256": protocol_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "analysis_plan_sha256": analysis_plan_sha256,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"frozen protocol is already claimed and terminal outcomes cannot be retried: {claim}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return claim


def _bind_protocol_claim(claim: Path, campaign_run_id: str, plan: list[dict]) -> None:
    payload = json.loads(claim.read_text(encoding="utf-8"))
    if payload.get("status") != "reserved" or "campaign_run_id" in payload:
        raise ValueError("protocol claim is not an unused reservation")
    payload.update({"status": "bound", "campaign_run_id": campaign_run_id})
    temporary = claim.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, claim)
    cells = claim.with_name(f"{claim.stem}.cells")
    cells.mkdir(mode=0o700)
    for row in plan:
        run_id = (f"{campaign_run_id}__{row['arm']}__r{int(row['repeat']):02d}"
                  f"__seed{int(row['seed']):03d}")
        cell = {"version": 1, "status": "authorized",
                "protocol_inputs_sha256": payload["protocol_inputs_sha256"],
                "environment_manifest_sha256": payload["environment_manifest_sha256"],
                "analysis_plan_sha256": payload["analysis_plan_sha256"],
                "campaign_run_id": campaign_run_id, "ordinal": int(row["ordinal"]),
                "arm": row["arm"], "repeat": int(row["repeat"]),
                "seed": int(row["seed"]), "run_id": run_id}
        target = cells / f"{int(row['ordinal']):02d}.authorized.json"
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(cell, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _execute_live_plan(live_plan: list[dict], command_for, preflight_for=None) -> list[dict]:
    """Execute every predeclared attempt exactly once, retaining non-passing outcomes.

    A grader failure is an experimental observation, not a reason to censor the remaining arms.  This
    helper deliberately has no stop-on-failure branch.  The completion controller later distinguishes a
    complete negative outcome from missing/corrupt evidence and independently gates compiler promotion.
    """
    results: list[dict] = []
    for index, item in enumerate(live_plan):
        if preflight_for is not None:
            gate = preflight_for(item)
            if gate is not None:
                # Protocol drift is not a treatment outcome.  Do not invoke additional paid cells;
                # retain every remaining planned identity explicitly as not started.
                for pending in live_plan[index:]:
                    results.append({**pending, "attempted": False, "executed": False,
                                    "cell_status": "not_started",
                                    "terminal_class": "harness_invalid",
                                    "reason": "campaign preflight invalidated before cell start",
                                    "preflight_receipt": gate, "returncode": None,
                                    "run_identity_ok": False})
                break
        proc = subprocess.run(command_for(item), cwd=repo_root(), text=True)
        run_dir = Path(item["run_dir"])
        record_path = run_dir / "run_record.json"
        identity_ok = False
        terminal_class = "harness_invalid"
        if record_path.is_file():
            try:
                identity_ok = json.loads(record_path.read_text(encoding="utf-8")).get(
                    "run_id") == item["run_id"]
            except (OSError, json.JSONDecodeError):
                pass
        terminal_path = run_dir / "contracts" / "terminal_outcome.json"
        if identity_ok and terminal_path.is_file():
            try:
                declared = json.loads(terminal_path.read_text(encoding="utf-8"))
                if (declared.get("run_id") == item["run_id"] and
                        declared.get("terminal_class") in {
                            "graded_pass", "graded_fail", "treatment_search_fail",
                            "treatment_build_fail", "treatment_agent_fail",
                            "harness_invalid"}):
                    terminal_class = declared["terminal_class"]
            except (OSError, json.JSONDecodeError):
                pass
        results.append({**item, "attempted": True, "executed": identity_ok,
                        "cell_status": "executed" if identity_ok else "not_started",
                        "terminal_class": terminal_class,
                        "returncode": proc.returncode, "run_identity_ok": identity_ok})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=HERE / "experiment.yaml")
    parser.add_argument("--live", action="store_true", help="authorize the sequential paid campaign")
    args = parser.parse_args(argv)

    spec = HostExperimentSpec.from_yaml(args.spec)
    # Preview reports the exact live authorization decision too. A green machinery-only check must never
    # make a draft or under-repeated paper campaign look launchable.
    preflight = spec.preflight(
        check_environment=True, probe_board=args.live, require_frozen=True)
    plan = [dict(row) for row in spec.agent["launch_plan"]]
    if not args.live:
        print(json.dumps({"mode": "preview", "paid_work": False, "sequential": True,
                          "plan": plan, "preflight": preflight.to_dict()}, indent=2))
        return 0
    if spec.status != "protocol_frozen":
        print("REFUSING paid campaign: live launch requires status exactly protocol_frozen")
        return 2
    if not preflight.ready:
        print(json.dumps(preflight.to_dict(), indent=2))
        print("REFUSING paid campaign: CPU-host experiment preflight is NO_GO")
        return 2

    claim_root = spec._repo_path(spec.telemetry["output_layout"])
    claim_path = _claim_protocol_once(
        claim_root, protocol_sha256=str(spec.freeze["protocol_inputs_sha256"]),
        environment_manifest_sha256=str(spec.environment["sha256"]),
        analysis_plan_sha256=str(spec.analysis["sha256"]),
        spec_path=args.spec)
    handle = start_run(
        suite="cpu-host-compiler", method="launcher", target="k1_cpu",
        seed=int(spec.agent["launch_seed"]),
        extra={"experiment": spec.label, "spec": str(args.spec.resolve()), "plan": plan,
               "protocol_claim": str(claim_path),
               "environment_manifest_sha256": str(spec.environment["sha256"]),
               "analysis_plan_sha256": str(spec.analysis["sha256"]),
               "provider_sampling_seeded": False})
    # ``launch_seed`` is AET campaign metadata only; CodexLLM receives no provider sampling seed.
    _bind_protocol_claim(claim_path, handle.run_id, plan)
    status = "error"
    results: list[dict] = []
    try:
        chia_python = spec._repo_path(spec.telemetry["chia_python"])
        runner = spec._repo_path(spec.agent["runner"])
        live_plan = []
        for item in plan:
            run_id = (f"{handle.run_id}__{item['arm']}__r{item['repeat']:02d}"
                      f"__seed{item['seed']:03d}")
            live_plan.append({**item, "run_id": run_id,
                              "run_dir": str((handle.run_dir.parent / run_id).resolve())})
        def command_for(item):
            return [str(chia_python), str(runner), "--spec", str(args.spec.resolve()),
                    "--arm", item["arm"], "--seed", str(item["seed"]),
                    "--run-id", item["run_id"], "--authorization-claim", str(claim_path),
                    "--live"]

        preflight_receipts = handle.run_dir / "contracts" / "cell_preflights"
        preflight_receipts.mkdir()
        boundary_receipts = handle.run_dir / "contracts" / "block_boundaries"
        boundary_receipts.mkdir()
        block_boundaries = []

        def preflight_for(item):
            block = int(item["repeat"])
            if int(item["ordinal"]) == block * len(spec.arms):
                gate, boundary = _requalify_block_boundary(
                    spec, block=block, first_ordinal=int(item["ordinal"]))
                boundary_path = boundary_receipts / f"{block:02d}.json"
                boundary_path.write_text(
                    json.dumps(boundary, indent=2) + "\n", encoding="utf-8")
                block_boundaries.append({
                    "block": block, "first_ordinal": int(item["ordinal"]),
                    "path": str(boundary_path.resolve()),
                    "sha256": hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
                })
            else:
                gate = spec.preflight(
                    check_environment=True, probe_board=True, require_frozen=True)
            receipt = preflight_receipts / f"{int(item['ordinal']):02d}.json"
            receipt.write_text(json.dumps(gate.to_dict(), indent=2) + "\n", encoding="utf-8")
            return None if gate.ready else str(receipt.resolve())

        results = _execute_live_plan(live_plan, command_for, preflight_for=preflight_for)
        all_accounted = len(results) == len(live_plan)
        executed = [row for row in results if row["executed"]]
        invalidated = [row for row in results if row["terminal_class"] == "harness_invalid"]
        status = "ok" if all_accounted and not invalidated else "error"
        # Non-zero arm codes are retained outcomes.  The finalizer validates their full evidence before
        # declaring the campaign complete, so the launcher succeeds once all exact identities exist.
        return 0 if status == "ok" else 1
    finally:
        record = {"version": 3, "sequential": True, "terminal_failure_policy": "record_and_continue",
                  "retry_terminal_outcomes": False, "campaign_run_id": handle.run_id,
                  "launch_seed": int(spec.agent["launch_seed"]),
                  "launch_seed_role": spec.agent["launch_seed_role"],
                  "provider_sampling_seeded": False,
                  "run_id_scheme": "{campaign_run_id}__{arm}__r{repeat:02d}__seed{seed:03d}",
                  "runs_root": str(handle.run_dir.parent.resolve()),
                  "authorization_claim": str(claim_path),
                  "authorization_claim_sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
                  "protocol_inputs_sha256": spec.freeze.get("protocol_inputs_sha256"),
                  "environment_manifest_sha256": spec.environment.get("sha256"),
                  "analysis_plan_sha256": spec.analysis.get("sha256"),
                  "block_boundaries": locals().get("block_boundaries", []),
                  "planned": locals().get("live_plan", plan), "results": results}
        launch_path = handle.run_dir / "contracts" / "launch.json"
        temporary = launch_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, launch_path)
        finish_run(handle, status=status,
                   summary={"planned_cells": len(plan),
                            "accounted_cells": len(results),
                            "attempted_cells": sum(row.get("attempted") is True for row in results),
                            "executed_cells": sum(row.get("executed") is True for row in results),
                            "not_started_cells": sum(row.get("cell_status") == "not_started"
                                                     for row in results),
                            "successful_cells": sum(row.get("returncode") == 0 and
                                                    row.get("executed") is True for row in results),
                            "nonpassing_executed_cells": sum(
                                row.get("executed") is True and row.get("returncode") != 0
                                for row in results),
                            "harness_invalid_cells": sum(
                                row.get("terminal_class") == "harness_invalid" for row in results),
                            "continued_after_treatment_failure": True})


if __name__ == "__main__":
    raise SystemExit(main())
