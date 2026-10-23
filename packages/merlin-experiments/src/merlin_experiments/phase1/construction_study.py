"""Explicit offline author continuation using the ordinary sandbox and brokers.

A study contains operator-authored public commands, not a model transcript or a
fresh generated compiler achievement. The same staged action runner is available
to automatic authors. Native admission and independent feedback remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from . import construction_actions as actions

CLIENT = "construction_actions.py"
PLAN = "CONSTRUCTION_ACTIONS.json"


def selection(options):
    value = getattr(options, "offline_actions", "")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("offline actions require an explicit canonical public JSON plan")
    plan, digest = actions.load(path)
    return {
        "schema": actions.SCHEMA,
        "sha256": digest,
        "actions": len(plan["actions"]),
        "origin": "operator-authored public construction commands",
        "provider_started": False,
    }


def validate(options):
    if not getattr(options, "offline_actions", ""):
        return
    if (
        options.resume
        or options.joint_epoch
        or options.preflight_only
        or options.codex_canary
        or options.qualify_submission
        or options.seal_current
        or options.allow_unsandboxed
        or options.sandbox != "bwrap"
        or options.no_oracle
        or options.skip_hidden
        or options.seed_submission
        or not options.fresh_backend
    ):
        raise ValueError(
            "offline actions require --fresh-backend and ordinary isolation without seeds, bypasses or resume"
        )
    if options.max_wall_s <= 0:
        raise ValueError("offline actions require a positive --max-wall-s budget")
    selection(options)


def stage(workspace, run_dir, options):
    record = selection(options)
    if record is None:
        return
    data = Path(options.offline_actions).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("offline action plan changed during selection")
    for root in (Path(workspace), Path(run_dir)):
        client = root / CLIENT
        expected = Path(actions.__file__).read_bytes()
        if client.exists():
            if client.is_symlink() or client.read_bytes() != expected:
                raise ValueError("staged construction action client differs")
        else:
            client.write_bytes(expected)
            client.chmod(0o444)
        plan_path = root / PLAN
        if plan_path.exists():
            if plan_path.is_symlink() or plan_path.read_bytes() != data:
                raise ValueError("staged construction plan differs")
        else:
            plan_path.write_bytes(data)
            plan_path.chmod(0o444)


def execute(prepared, *, timing_file):
    from merlin.targetgen.sandbox import bwrap as BW

    from . import joint_control
    from .feedback import lifecycle as FL
    from .providers import execution as EX

    options, context = prepared.request.options, prepared.request.context
    validate(options)
    prepared.verify_inputs()
    ws, run_dir, bundle = prepared.workspace, prepared.run_dir, prepared.bundle
    expected = prepared.environment["run_config"]["construction_study"]
    _, digest = actions.load(ws / PLAN)
    if digest != expected["sha256"]:
        raise RuntimeError("served construction actions differ from native preparation")
    # Match ordinary authoring: start the admitted joint clock immediately
    # before launching broker-backed author work. The study still has its own
    # smaller wall budget; start() does not reset an already-started clock.
    joint_control.start()
    broker = FL.start_brokers(
        ws,
        FL.BrokerConfig(
            context,
            tuple(prepared.request.resolved_tools()),
            timing_file,
            max(0, int(options.sim_max_jobs)),
            prepared.public_root,
            prepared.policy_root,
            prepared.contract_root,
            BW.frozen_selected_rtl_facts(ws, bundle, repo=context.repo),
            run_dir / "input_bundle_manifest.yaml" if bundle.get("compiler_tool_profile") is not None else None,
            sim_per_capsule_timeout=options.sim_per_capsule_timeout,
            raw_evidence_root=(run_dir / "_qa_work" / "selfchecks").resolve(),
        ),
    )
    process, primary_error = None, None
    started = time.time_ns()
    try:
        argv = ["python3", CLIENT, PLAN, "--out", "construction_study", "--timeout", str(options.max_wall_s)]
        command = EX.sandbox_command(shlex.join(argv), ws, bundle, context=context, private_run_dir=run_dir)
        with (
            (run_dir / "construction.stdout.log").open("w") as out,
            (run_dir / "construction.stderr.log").open("w") as err,
        ):
            process = subprocess.Popen(["bash", "-c", command], cwd=ws, stdout=out, stderr=err, start_new_session=True)
            try:
                returncode = process.wait(timeout=options.max_wall_s + 5)
            except subprocess.TimeoutExpired:
                returncode = 124
        prepared.verify_inputs()
        result = {
            **expected,
            "returncode": returncode,
            "started_ns": started,
            "finished_ns": time.time_ns(),
            "workspace": str(ws),
            "status": "actions_completed" if returncode == 0 else "actions_incomplete",
            "qualification_authority": False,
            "result": str(ws / "construction_study/result.json"),
            "note": "Ordinary broker evidence only; not a provider run or formal compiler qualification.",
        }
        (run_dir / "construction_study_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return returncode
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        FL.stop_brokers(ws, broker, primary_error=primary_error)
