"""Qualify one frozen fresh compiler without an author, under its original deadline.

This separate operator route preserves the old run and never upgrades its grade
or claims authoring convergence. Native formal grading and the selected complete
model chains start together; all-declared-output audits remain mandatory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import launch_phase1 as P
from merlin_experiments.phase1 import launch_config as L

from merlin.common import owned_children
from merlin.common.digest import sha256_file


class DeadlineReached(RuntimeError):
    """The original campaign budget has expired, including during offline audit."""


def qualifier_argv(launch, submission: Path) -> list[str]:
    """Keep selected grading inputs and remove every author/client option."""
    keep = {
        "--run-id", "--descriptor", "--repo", "--level", "--arm", "--treatment",
        "--bundle", "--bundle-manifest", "--corpus-seal", "--oracle-timing",
        "--private-full-model-spec", "--driver", "--model", "--sandbox",
        "--experiment", "--schedule", "--qa-timeout", "--model-budget-s",
        "--sim-max-jobs", "--sim-per-capsule-timeout", "--instruction-selection",
        "--native-result-format-selection", "--hidden-capsule-mode", "--readback-policy",
    }
    argv = list(launch.argv[:3])
    for index, item in enumerate(launch.argv[3:], start=3):
        if item in keep:
            argv.extend((item, launch.argv[index + 1]))
    return [*argv, "--qualify-submission", str(submission)]


def admit_handoff(F, path: Path, expected_sha256: str, submission: Path, deadline: float):
    """Reopen root's immutable native-OOT handoff; a result string is insufficient."""
    if submission.is_symlink() or any(p.is_symlink() for p in submission.rglob("*")):
        raise ValueError("frozen compiler must be an ordinary self-contained directory")
    if sha256_file(path) != expected_sha256:
        raise ValueError("fresh author handoff differs")
    proof = F.read(path)
    package = F.package_identity(submission)
    original = Path(proof["original_run_dir"])
    original_deadline = float(proof["original_hard_deadline_unix"])
    if (
        proof.get("schema") != "merlin.operator.fresh_author_handoff.v1"
        or proof.get("qualification_only") is not True
        or proof.get("authoring_converged") is not False
        or proof.get("frozen_submission") != package
        or not math.isfinite(original_deadline) or deadline > original_deadline
        or F.oot_repo.tree_digest(original / "oot", proof["native_oot_commit"])
        != package["hash_tree_sha256"]
    ):
        raise ValueError("fresh author handoff/candidate/deadline correspondence failed")
    if sha256_file(path) != expected_sha256:
        raise ValueError("fresh author handoff changed while reopening")
    return package


def formal_complete(F, native, run_dir: Path, package: dict) -> bool:
    path = run_dir / "submission_qualification.json"
    if native["process"].poll() != 0 or not path.is_file():
        return False
    value = F.read(path)
    return (
        value.get("schema") == "merlin.phase1.submission_qualification.v1"
        and value.get("qualification_only") is True
        and value.get("authoring_converged") is False
        and value.get("selected_source_tree_sha256") == package["hash_tree_sha256"]
        and value.get("graded_source_sha256") == package["hash_tree_sha256"]
        and value.get("official_grade", {}).get("complete") is True
    )


def run(launch, prep: Path, submission: Path, proof: Path, proof_sha256: str,
        deadline: float, *, poll_s: float = 1) -> int:
    """A dedicated process owns/reaps all children, including escaped sessions."""
    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("original campaign deadline already expired")
    F = P.operator_module("full_model_verification")
    S = P.operator_module("launch_supervisor")
    processes = S.Processes()
    owned_children.become_child_subreaper()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    state = {"schema": "merlin.operator.submission_qualification_supervisor.v1",
             "qualification_only": True, "authoring_converged": False,
             "phase1_complete": False, "status": "preparing", "models": {},
             "deadline_unix": deadline, "author_proof": {"path": str(proof), "sha256": proof_sha256}}
    prep.mkdir(parents=True, exist_ok=True)
    if (prep / "status.json").exists():
        raise ValueError("qualification supervisor output already exists")

    def publish():
        state["updated_unix"] = time.time()
        P.save(prep / "status.json", state)

    def interrupted(signum, _frame):
        raise DeadlineReached(f"qualification stopped by signal {signum}")

    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("qualification requires its own dedicated deadline timer")
    previous = {sig: signal.signal(sig, interrupted) for sig in signals}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.time()))
    code, children = 2, {}
    try:
        with P.configured_environment(launch):
            package = admit_handoff(F, proof, proof_sha256, submission, deadline)
            state.update(original_hard_deadline_unix=F.read(proof)["original_hard_deadline_unix"],
                         deadline_reset=False)
            run_dir = Path(launch.variables["RUN"])
            repo = launch.path(launch.document["inputs"]["repo"])
            workspace = (repo / "out/build/agent-workspaces/phase1"
                         / launch.document["identity"]["target"] / launch.document["identity"]["arm"]
                         / launch.run_id / "workspace")
            if run_dir.exists() or workspace.exists():
                raise ValueError("qualification requires a distinct unused run ID and workspace")
            policy = launch.document.get("full_model_verification") or {}
            if policy.get("output_scope") != "all_declared" or "all_output_plan" not in policy:
                raise ValueError("qualification requires selected all-declared-output evidence")
            # The ordinary launcher performs source/config/pin, bundle-seal and
            # positive infrastructure readiness admission; it starts no author.
            readiness = P.prepare(launch, prep)
            P.save(prep / "launch-readiness.json", readiness)
            selected, identity = F.selection(launch), F.selected_identity(launch)
            if "native" in selected["names"] or not selected["names"]:
                raise ValueError("selected model roster is empty or uses reserved native role")
            output = F.output_verification()
            context = output.validate(launch, identity=identity, models=selected["models"])
            if context is None:
                raise ValueError("qualification has no all-output context")
            assigned = S.assigned_cpus(launch)
            remaining = int(deadline - time.time())
            if remaining <= 0:
                raise DeadlineReached("preparation exhausted original campaign budget")
            env = dict(os.environ)
            jobs = {"native": (qualifier_argv(launch, submission), None, None)}
            for name in selected["names"]:
                report = prep / f"{name}.json"
                work = repo / "out/build/model_check/phase1-launch" / launch.run_id / "qualification" / name
                if work.exists():
                    raise ValueError("qualification model work path already exists")
                jobs[name] = (F.model_command(launch, name, submission, report, work, remaining), report, work)
            for name, (argv, report, work) in jobs.items():
                row = processes.start(argv, prep / f"{name}.log", role=name, cwd=repo, env=env, cpus=assigned)
                pid = row["process"].pid
                row.update(report=report, pin=prep / f"{name}-pin.json", input_id=policy["input"])
                P.save(row["pin"], {"selected_identity": identity, "submission": package,
                       "argv": argv, "work": str(work) if work else None, "timeout_s": remaining,
                       "pid": pid, "start_ticks": processes.known.get(pid, (None, None))[0],
                       "boot_id": boot, "process_group": pid, "deadline_unix": deadline})
                children[name] = row
            state.update(status="formal_and_full_models", selected_identity=identity, submission=package)
            publish()
            while True:
                if time.time() >= deadline:
                    raise DeadlineReached("original campaign deadline expired")
                records = processes.owned()
                charged, available = P.memory(records)
                if (charged > int(launch.variables["SUPERVISOR_PSS_MAX_BYTES"])
                        or available < int(launch.variables["SUPERVISOR_HOST_HEADROOM_BYTES"])):
                    raise RuntimeError("qualification memory guard")
                for name in selected["names"]:
                    row = children[name]
                    judged = False
                    if row["process"].poll() is not None and row["report"].is_file():
                        digest = sha256_file(row["report"])
                        if "completed_report_sha256" not in row:
                            row["completed_report_sha256"] = digest
                            row["judged_complete"] = row["process"].poll() == 0 and F.report_complete(
                                F.read(row["report"]), selected["models"].model(name), policy, package)
                        if digest != row["completed_report_sha256"]:
                            raise ValueError("completed model report changed")
                        judged = row["judged_complete"]
                    audit = output.audit(context, name=name, child=row, package=package,
                                         models_path=selected["models"].path) if judged else {"status": "pending"}
                    state["models"][name] = {"exit_code": row["process"].poll(),
                        "judged_full_chain_complete": judged, "all_declared_outputs": audit,
                        "full_chain_complete": judged and audit["status"] == "pass"}
                state.update(owned_processes=len(records), pss_bytes=charged, host_available_bytes=available,
                             remaining_s=max(0, deadline - time.time()))
                if all(row["process"].poll() is not None for row in children.values()):
                    native = formal_complete(F, children["native"], run_dir, package)
                    models = all(row["full_chain_complete"] for row in state["models"].values())
                    final = (output.finalize(context, {n: children[n] for n in selected["names"]})
                             if models else {"status": "incomplete"})
                    unchanged = (F.package_identity(submission) == package
                                 and F.selected_identity(launch) == identity
                                 and sha256_file(proof) == proof_sha256)
                    state.update(native_formal_complete=native, all_outputs_finalization=final,
                                 phase1_complete=native and models and final["status"] == "pass" and unchanged,
                                 status="finished", frozen_inputs_unchanged=unchanged)
                    code = 0 if state["phase1_complete"] else 1
                    break
                publish()
                time.sleep(poll_s)
    except BaseException as error:
        state.update(status="stopped_incomplete", termination_reason=f"{type(error).__name__}: {error}",
                     phase1_complete=False)
        if not isinstance(error, (DeadlineReached, ValueError, OSError, RuntimeError)):
            raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Ignore a second operator stop while closing the actual ownership tree.
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        try:
            processes.close()
        except BaseException as error:
            state.update(phase1_complete=False, status="cleanup_unknown", cleanup_error=str(error))
            code = 2
        cleanup = owned_children.reap_owned_children(None, 6)
        state["cleanup"] = cleanup
        state["job_exit_codes"] = {name: row["process"].poll() for name, row in children.items()}
        if not cleanup["complete"]:
            state.update(phase1_complete=False, status="cleanup_unknown")
            code = 2
        publish()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "task-root", "prep", "submission", "author-proof"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--author-proof-sha256", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--var", action="append", default=[])
    parser.add_argument("--run", action="store_true", help="otherwise print argv without starting jobs")
    args = parser.parse_args()
    launch = L.build(L.load(args.config), task_root=args.task_root, prep=args.prep,
                     run_id=args.run_id, overrides=L._overrides(args.var))
    if not args.run:
        print(json.dumps({"qualification_only": True, "authoring_converged": False,
                          "argv": qualifier_argv(launch, args.submission), "deadline_unix": args.deadline_unix}))
        return 0
    return run(launch, args.prep, args.submission, args.author_proof,
               args.author_proof_sha256, args.deadline_unix)


if __name__ == "__main__":
    raise SystemExit(main())
