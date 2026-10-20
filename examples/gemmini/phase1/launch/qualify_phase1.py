"""Qualify one frozen fresh compiler without an author, under an explicit budget.

This separate operator route preserves the old run and never upgrades its grade
or claims authoring convergence. Native formal grading and the selected complete
model chains start together under full scope. Explicit capsule scope starts only
native capsule qualification; complete external model validation remains pending.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from hashlib import sha256
from pathlib import Path

import launch_phase1 as P
import yaml
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
        "--native-result-format-selection", "--hidden-capsule-mode", "--readback-policy", "--validation-scope",
    }
    argv = list(launch.argv[:3])
    for index, item in enumerate(launch.argv[3:], start=3):
        if item in keep:
            argv.extend((item, launch.argv[index + 1]))
    return [*argv, "--qualify-submission", str(submission)]


def admit_budget_extension(path, expected_sha256, *, proof_path, proof_sha256, proof, package, deadline):
    """Bind a separately selected operator continuation; never rewrite the author budget."""
    if path is None and expected_sha256 is None:
        return None
    path = Path(path) if path is not None else None
    keys = {"schema", "scope", "qualification_only", "new_author_allowed", "original_run_dir",
            "original_handoff_sha256", "original_hard_deadline_unix", "submission_sha256",
            "authorized_at_unix", "wall_budget_s", "absolute_deadline_unix", "authorization"}
    if (path is None or not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256):
        raise ValueError("qualification budget extension has no exact selected ordinary receipt")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("qualification budget extension snapshot differs")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("qualification budget extension has an invalid closed schema")
    numbers = (value["authorized_at_unix"], value["absolute_deadline_unix"],
               value["original_hard_deadline_unix"])
    if any(type(n) not in (int, float) or not math.isfinite(n) for n in numbers):
        raise ValueError("qualification budget extension needs finite explicit times")
    authorized, extended, original = numbers
    authority = value["authorization"]
    if not isinstance(authority, dict) or set(authority) != {"path", "sha256"}:
        raise ValueError("qualification budget extension has no selected operator authorization")
    authority_path = Path(authority["path"])
    if not authority_path.is_absolute() or authority_path.is_symlink() or not authority_path.is_file():
        raise ValueError("qualification budget authorization changed")
    authority_raw = authority_path.read_bytes()
    if sha256(authority_raw).hexdigest() != authority["sha256"]:
        raise ValueError("qualification budget authorization changed")
    authorization = json.loads(authority_raw)
    if (value["schema"] != "merlin.operator.qualification_budget_extension.v1"
            or value["scope"] != "qualification_of_existing_frozen_compiler_only"
            or value["qualification_only"] is not True or value["new_author_allowed"] is not False
            or value["original_run_dir"] != proof["original_run_dir"]
            or value["original_handoff_sha256"] != proof_sha256
            or sha256_file(proof_path) != proof_sha256
            or original != float(proof["original_hard_deadline_unix"])
            or value["submission_sha256"] != package["hash_tree_sha256"]
            or type(value["wall_budget_s"]) is not int or value["wall_budget_s"] <= 0
            or not 0 < authorized <= time.time() or not original < deadline <= extended
            or not authorized < extended <= authorized + value["wall_budget_s"]
            or not isinstance(authorization, dict)
            or not isinstance(authorization.get("authorized_by_user_text"), str)
            or not authorization["authorized_by_user_text"].strip()
            or authorization.get("new_author_allowed") is not False
            or authorization.get("started_unix") != authorized
            or authorization.get("absolute_deadline_unix") != extended):
        raise ValueError("qualification budget extension differs from selected author, operator or deadline")
    if (path.read_bytes() != raw or sha256_file(path) != expected_sha256
            or sha256_file(authority_path) != authority["sha256"]):
        raise ValueError("qualification budget extension changed while reopening")
    return {"path": str(path), "sha256": expected_sha256, "authorization": authority,
            "original_hard_deadline_unix": original, "qualification_deadline_unix": deadline,
            "original_campaign_budget_preserved": True, "new_author_allowed": False}


def admit_handoff(F, path: Path, expected_sha256: str, submission: Path, deadline: float, *,
                  budget_extension=None, budget_extension_sha256=None):
    """Reopen the operator's immutable native-OOT handoff; a result string is insufficient."""
    if submission.is_symlink() or any(p.is_symlink() for p in submission.rglob("*")):
        raise ValueError("frozen compiler must be an ordinary self-contained directory")
    if sha256_file(path) != expected_sha256:
        raise ValueError("fresh author handoff differs")
    proof = F.read(path)
    package = F.package_identity(submission)
    continuation = admit_budget_extension(
        budget_extension, budget_extension_sha256, proof_path=path, proof_sha256=expected_sha256,
        proof=proof, package=package, deadline=deadline)
    original = Path(proof["original_run_dir"])
    original_deadline = float(proof["original_hard_deadline_unix"])
    mode = proof.get("freeze_mode", "native_oot")
    if mode == "native_oot":
        checkpoint_digest = F.oot_repo.tree_digest(original / "oot", proof["native_oot_commit"])
    elif mode == "terminal_author_operator_checkpoint":
        checkpoint = Path(proof["operator_oot_path"])
        source = Path(proof["terminal_authored_source_path"])
        environment = yaml.safe_load((original / "environment.yaml").read_text())
        source_revision = proof["original_source_revision"]
        if (
            not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_dir()
            or checkpoint.resolve().is_relative_to(original.resolve())
            or not source.is_absolute() or source.is_symlink() or not source.is_dir()
            or source.resolve() != (Path(environment["workspace_path"]) / "submission").resolve()
            or environment.get("run_id") != original.name
            or proof.get("original_empty_origin") is not True
            or len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision)
            or any(proof[key] != package["hash_tree_sha256"] for key in (
                "actual_terminal_source_tree_sha256", "source_sha256_before", "source_sha256_after"))
            or F.hash_tree(source).get("sha256") != package["hash_tree_sha256"]
        ):
            raise ValueError("terminal authored source/operator checkpoint correspondence failed")
        checkpoint_digest = F.oot_repo.tree_digest(checkpoint, proof["operator_oot_commit"])
    else:
        raise ValueError("unknown fresh compiler freeze mode")
    if (
        proof.get("schema") != "merlin.operator.fresh_author_handoff.v1"
        or proof.get("qualification_only") is not True
        or proof.get("authoring_converged") is not False
        or proof.get("frozen_submission") != package
        or not math.isfinite(original_deadline) or (deadline > original_deadline and continuation is None)
        or checkpoint_digest != package["hash_tree_sha256"]
    ):
        raise ValueError("fresh author handoff/candidate/deadline correspondence failed")
    if sha256_file(path) != expected_sha256:
        raise ValueError("fresh author handoff changed while reopening")
    return package


def formal_complete(F, native, run_dir: Path, package: dict, *, validation_scope="full", descriptor=None) -> bool:
    path = run_dir / "submission_qualification.json"
    if native["process"].poll() != 0 or not path.is_file():
        return False
    value = F.read(path)
    complete = (
        value.get("schema") == "merlin.phase1.submission_qualification.v1"
        and value.get("qualification_only") is True
        and value.get("authoring_converged") is False
        and value.get("selected_source_tree_sha256") == package["hash_tree_sha256"]
        and value.get("graded_source_sha256") == package["hash_tree_sha256"]
        and value.get("validation_scope", "full") == validation_scope
        and value.get("official_grade", {}).get("complete") is True
    )
    if not complete or validation_scope == "full":
        return complete
    if validation_scope != "capsules" or descriptor is None:
        return False
    from merlin_experiments.phase1.feedback import capsule_certification

    try:
        verified = capsule_certification.verify(run_dir, descriptor=descriptor)
        return (verified["submission_sha256"] == package["hash_tree_sha256"]
                and verified["capsule_complete"] is True
                and value["official_grade"].get("validation_scope") == "capsules"
                and value["official_grade"].get("capsule_certification") == verified["receipt"])
    except (ValueError, KeyError, OSError):
        return False


def run(launch, prep: Path, submission: Path, proof: Path, proof_sha256: str,
        deadline: float, *, poll_s: float = 1, budget_extension=None,
        budget_extension_sha256=None) -> int:
    """A dedicated process owns/reaps all children, including escaped sessions."""
    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("original campaign deadline already expired")
    scope = launch.document.get("validation_scope", "full")
    if scope not in {"full", "capsules"}:
        raise ValueError("unknown qualification validation scope")
    capsule_only = scope == "capsules"
    if capsule_only and deadline - time.time() > 3600:
        raise ValueError("capsule-only requalification requires a finite budget of at most one hour")
    started = time.time()
    F = P.operator_module("full_model_verification")
    S = P.operator_module("launch_supervisor")
    processes = S.Processes()
    owned_children.become_child_subreaper()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    state = {"schema": "merlin.operator.submission_qualification_supervisor.v1",
             "qualification_only": True, "authoring_converged": False,
             "validation_scope": scope, "capsule_complete": False,
             "phase1_complete": False, "external_full_models_complete": False,
             "overall_release_qualified": False, "status": "preparing", "models": {},
             "qualification_started_unix": started, "qualification_budget_s": deadline - started,
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
        publish()
        with P.configured_environment(launch):
            continuation_args = ({"budget_extension": budget_extension,
                                  "budget_extension_sha256": budget_extension_sha256}
                                 if budget_extension is not None or budget_extension_sha256 is not None else {})
            capsule_proof = F.read(proof).get("schema") == "merlin.phase1.capsule_certification.v1"
            if capsule_proof:
                intake = P.operator_module("capsule_handoff_intake")
                package, budget = intake.admit(
                    F, proof, proof_sha256, submission, deadline,
                    budget_path=budget_extension, budget_sha256=budget_extension_sha256)
                descriptor = launch.path(launch.document["inputs"]["descriptor"])
                if F.read(proof)["descriptor"]["sha256"] != sha256_file(descriptor):
                    raise ValueError("external model job selects a different capsule descriptor")
                state.update(capsule_complete=not capsule_only, external_validation_only=not capsule_only,
                             phase1_clock_preserved=True, deadline_reset=False,
                             qualification_budget_extended=False, budget_extension=budget)
            else:
                package = admit_handoff(F, proof, proof_sha256, submission, deadline, **continuation_args)
                state.update(original_hard_deadline_unix=F.read(proof)["original_hard_deadline_unix"],
                             deadline_reset=False, qualification_budget_extended=bool(continuation_args))
                if continuation_args:
                    state["budget_extension"] = admit_budget_extension(
                        budget_extension, budget_extension_sha256, proof_path=proof, proof_sha256=proof_sha256,
                        proof=F.read(proof), package=package, deadline=deadline)
            run_dir = Path(launch.variables["RUN"])
            repo = launch.path(launch.document["inputs"]["repo"])
            workspace = (repo / "out/build/agent-workspaces/phase1"
                         / launch.document["identity"]["target"] / launch.document["identity"]["arm"]
                         / launch.run_id / "workspace")
            if run_dir.exists() or workspace.exists():
                raise ValueError("qualification requires a distinct unused run ID and workspace")
            policy = launch.document.get("full_model_verification") or {}
            if not capsule_only and (policy.get("output_scope") != "all_declared" or "all_output_plan" not in policy):
                raise ValueError("qualification requires selected all-declared-output evidence")
            # The ordinary launcher performs source/config/pin, bundle-seal and
            # positive infrastructure readiness admission; it starts no author.
            from merlin_experiments.phase1.feedback import private_numerical_selection

            source_delegated = (not capsule_only and "private_full_model_spec" in launch.document["inputs"]
                and private_numerical_selection.read(launch.path(
                    launch.document["inputs"]["private_full_model_spec"])) is not None)
            readiness = (P.prepare(launch, prep, qualification_only=True) if source_delegated
                         else P.prepare(launch, prep))
            P.save(prep / "launch-readiness.json", readiness)
            selected = {"names": []} if capsule_only else F.selection(launch)
            identity = F.selected_identity(launch)
            if not capsule_only and ("native" in selected["names"] or not selected["names"]):
                raise ValueError("selected model roster is empty or uses reserved native role")
            output = None if capsule_only else F.output_verification()
            context = None if capsule_only else output.validate(launch, identity=identity, models=selected["models"])
            if not capsule_only and context is None:
                raise ValueError("qualification has no all-output context")
            chain = None if capsule_only else P.operator_module("selected_chain_supervision")
            native_chain = (chain.selected(launch, run_dir, deadline)
                            if not capsule_only and "private_full_model_spec" in launch.document["inputs"] else None)
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
                pin = prep / f"{name}-pin.json"
                original_argv = argv
                if native_chain is not None and name != "native":
                    P.save(pin, {"selected_identity": identity, "submission": package,
                           "argv": original_argv, "work": str(work), "timeout_s": remaining,
                           "deadline_unix": deadline})
                    argv = chain.worker_argv(native_chain, launch, name, submission, pin,
                                             prep / "native-pin.json")
                row = processes.start(argv, prep / f"{name}.log", role=name, cwd=repo, env=env, cpus=assigned)
                pid = row["process"].pid
                row.update(report=report, pin=prep / f"{name}-pin.json",
                           input_id=None if capsule_only else policy["input"])
                process_pin = (prep / f"{name}-process.json"
                               if native_chain is not None and name != "native" else row["pin"])
                P.save(process_pin, {"selected_identity": identity, "submission": package,
                       "argv": argv, "work": str(work) if work else None, "timeout_s": remaining,
                       "pid": pid, "start_ticks": processes.known.get(pid, (None, None))[0],
                       "boot_id": boot, "process_group": pid, "deadline_unix": deadline})
                children[name] = row
            state.update(status="capsule_qualification" if capsule_only else "formal_and_full_models",
                         selected_identity=identity, submission=package)
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
                if native_chain is not None:
                    chain.publish_static(native_chain, selected["names"])
                    chain.publish_numerical(native_chain, launch, children=children, names=selected["names"],
                        state=state, identity=identity, package=package, output_context=context, output=output)
                failed = [name for name, row in children.items()
                          if row["process"].poll() not in (None, 0)]
                if failed:
                    code = 1
                    raise RuntimeError("qualification children failed: " + ", ".join(failed))
                state.update(owned_processes=len(records), pss_bytes=charged, host_available_bytes=available,
                             remaining_s=max(0, deadline - time.time()))
                if all(row["process"].poll() is not None for row in children.values()):
                    native = (formal_complete(F, children["native"], run_dir, package,
                              validation_scope=scope, descriptor=launch.path(launch.document["inputs"]["descriptor"]))
                              if capsule_only else formal_complete(F, children["native"], run_dir, package))
                    models = not capsule_only and all(row["full_chain_complete"] for row in state["models"].values())
                    final = (output.finalize(context, {n: children[n] for n in selected["names"]})
                             if models else {"status": "not_selected" if capsule_only else "incomplete"})
                    unchanged = (F.package_identity(submission) == package
                                 and F.selected_identity(launch) == identity
                                 and sha256_file(proof) == proof_sha256)
                    if continuation_args:
                        budget = state["budget_extension"]
                        unchanged = (unchanged and sha256_file(Path(budget["path"])) == budget["sha256"]
                                     and sha256_file(Path(budget["authorization"]["path"]))
                                     == budget["authorization"]["sha256"])
                    state.update(native_formal_complete=native if not capsule_only else False,
                                 capsule_complete=native and unchanged if capsule_only else False,
                                 native_capsule_complete=native if capsule_only else False,
                                 all_outputs_finalization=final,
                                 phase1_complete=native and models and final["status"] == "pass" and unchanged,
                                 status="finished", frozen_inputs_unchanged=unchanged,
                                 qualification_completed_unix=time.time(),
                                 qualification_wall_s=time.time() - started)
                    if capsule_proof and not capsule_only:
                        from merlin_experiments.phase1.feedback import capsule_certification
                        final_publication = capsule_certification.verify_phase2_handoff(
                            proof.parent, descriptor=launch.path(launch.document["inputs"]["descriptor"]))
                        unchanged = (unchanged and final_publication["receipt"]
                                     == state["budget_extension"]["phase2_publication"])
                        state["phase1_complete"] = state["phase1_complete"] and unchanged
                        state.update(external_full_models_complete=state["phase1_complete"],
                                     external_validation_completed_unix=time.time(),
                                     overall_release_qualified=state["phase1_complete"])
                    state["external_validation_status"] = "pending" if capsule_only else (
                        "pass" if state["phase1_complete"] else "incomplete")
                    code = 0 if (state["capsule_complete"] if capsule_only else state["phase1_complete"]) else 1
                    break
                publish()
                time.sleep(poll_s)
    except BaseException as error:
        state.update(status="stopped_incomplete", termination_reason=f"{type(error).__name__}: {error}",
                     phase1_complete=False, external_full_models_complete=False, overall_release_qualified=False)
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
            state.update(phase1_complete=False, external_full_models_complete=False,
                         overall_release_qualified=False, status="cleanup_unknown", cleanup_error=str(error))
            code = 2
        cleanup = owned_children.reap_owned_children(None, 6)
        state["cleanup"] = cleanup
        state["job_exit_codes"] = {name: row["process"].poll() for name, row in children.items()}
        if not cleanup["complete"]:
            state.update(phase1_complete=False, external_full_models_complete=False,
                         overall_release_qualified=False, status="cleanup_unknown")
            code = 2
        if capsule_only and code != 0:
            state["capsule_complete"] = False
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
    parser.add_argument("--budget-extension", type=Path,
                        help="explicit operator continuation receipt; original handoff stays unchanged")
    parser.add_argument("--budget-extension-sha256")
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
               args.author_proof_sha256, args.deadline_unix,
               budget_extension=args.budget_extension, budget_extension_sha256=args.budget_extension_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
