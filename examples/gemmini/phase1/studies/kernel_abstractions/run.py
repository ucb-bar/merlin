"""Minimal isolated offline study using the existing provider, broker and evaluator.

Launch is explicit. Preparation never starts an author. Each process and its evaluator inherit one
selected CPU. Host-only configuration, models and oracle answers remain outside the author mount.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def boot(config, *, context_setup=True):
    global C, WS, PRIVATE, F, ENV, CH, END, CHECK_LOCK, CHILDREN, CASES, shim
    global prepare_codex_home, build_cmd, build_resume_cmd, reference_analysis, reference_experiments
    global request_label, pinned_source, _decorate_response, base_argv, selection_path, selection_sha, context
    C = json.loads(Path(config).read_bytes())
    C["config"] = str(Path(config).resolve())
    WS, PRIVATE, F = Path(C["workspace"]), Path(C["private"]), Path(C["framework"])
    ENV = json.loads((PRIVATE / "evaluator-environment.json").read_bytes())
    sys.path[:0] = ENV["PYTHONPATH"].split(":")
    os.environ.update(ENV)
    if C.get("full_compiler"):
        os.environ["MERLIN_FULL_COMPILER_SELECTION"] = json.dumps(C["full_compiler"], sort_keys=True)
    # An inherited operator selection must never enable the off arm.
    for key in tuple(os.environ):
        if key.startswith("MERLIN_REFERENCE"):
            os.environ.pop(key)
    if C["reference_guidance"] == "on" and C.get("reference_map"):
        os.environ["MERLIN_REFERENCE_MAP"] = json.dumps(C["reference_map"])
    from merlin_experiments.phase1.brokers.selfcheck import _decorate_response
    from merlin_experiments.phase1.brokers.selfcheck_sources import pinned_source, request_label
    from merlin_experiments.phase1.feedback import reference_analysis, reference_experiments
    from merlin_experiments.phase1.providers.codex_agent import build_cmd, build_resume_cmd, prepare_codex_home

    from merlin.targetgen.sandbox.bwrap import base_argv

    spec = importlib.util.spec_from_file_location("study_selfcheck_shim", WS / "agent_selfcheck.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    CH, END = WS / ".qa_channel", time.monotonic() + C["wall_s"]
    CHECK_LOCK, CHILDREN, CASES = threading.Lock(), set(), set(C["training_cases"])
    context = dict(
        compiler_sha256=shim._submission_digest(WS / "submission"),
        source_sha256=hashlib.sha256(
            "".join(digest(p) for p in sorted((WS / "public-cases").glob("*.mlir"))).encode()
        ).hexdigest(),
        environment_sha256=digest(PRIVATE / "evaluator-environment.json"),
        reference_sha256=digest(WS / "reference-kernels/reference-manifest.json"),
        evaluator=dict(identity="existing-offline-selfcheck", sha256=digest(__file__)),
        visibility="reference_guided",
        baseline={"submission": "submission"},
        capsules=[],
    )
    inputs = json.loads(Path(C["inputs"]).read_bytes())
    for row in inputs["cases"]:
        if row["split"] == "train" and row["case"] in CASES:
            context["capsules"].append(
                dict(
                    id=row["case"],
                    split="training",
                    goals=row.get("goals", []),
                    source_sha256=row["source"]["sha256"],
                    selfcheck_argv=[
                        "python3",
                        "agent_selfcheck.py",
                        "--submission",
                        "submission",
                        "--suite",
                        "scalability",
                        "--sim",
                        "spike",
                        "--tiers",
                        "L0,L1,L2",
                        "--capsules",
                        row["case"],
                        "--workers",
                        "1",
                    ],
                )
            )
    selection_path, selection_sha = None, None
    context["assistance"] = {
        "cca": C["reference_guidance"] == "on",
        "library_guidance": C.get("library_guidance", True),
        "schedule_guidance": C.get("schedule_guidance", True),
    }
    if context_setup and C["reference_guidance"] == "on":
        from merlin.mining.reconstruction_experiments import build_experiment_packet

        packet = PRIVATE / "packet.json"
        write(packet, build_experiment_packet({}, context=context, history=()))
        selection_path = PRIVATE / "cca-selection.json"
        write(
            selection_path,
            dict(
                schema="merlin.reference_experiment_selection.v1",
                packet=dict(path=str(packet), sha256=digest(packet)),
                current_context=context,
                context_pins=[
                    dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                    dict(path=str(PRIVATE / "evaluator-environment.json"), sha256=context["environment_sha256"]),
                ],
            ),
        )
        selection_sha = digest(selection_path)


def selected():
    return (
        reference_experiments.invocation(selection_path, selection_sha, tasks=False)
        if selection_path
        else nullcontext()
    )


def evaluate(config, submission, capsules, output, timeout, tiers="L0,L1,L2"):
    """The original evaluator CLI, with one worker and the selected original runtime."""
    boot(config, context_setup=False)
    from merlin_experiments.phase1 import full_compiler_workspace as full
    from merlin_experiments.phase1.context import load_context
    from merlin_experiments.phase1.feedback.public_scalability import install
    from merlin_experiments.phase1.feedback.selfcheck import main as selfcheck

    from merlin.targetgen.target_experiment import load_target_experiment

    ctx = load_context(C["descriptor"], repo=C["framework"])
    install(load_target_experiment(C["descriptor"]), Path(C["suite"]))
    args = [
        "--suite",
        "scalability",
        "--sim",
        "spike",
        "--submission",
        submission,
        "--capsules",
        capsules,
        "--workers",
        "1",
        "--tiers",
        tiers,
        "--timeout",
        str(timeout),
        "--out",
        output,
    ]
    contract = C.get("library_contract")
    if contract:
        args += [
            "--compiler-library-root",
            C["library"],
            "--compiler-library-contract",
            contract["path"],
            "--compiler-library-contract-sha256",
            C["library_sha"],
        ]
    owner = full.select(**C["full_compiler"]) if C.get("full_compiler") else None
    with full.scoped_full_compiler(owner):
        return selfcheck(args, context=ctx)


def terminate(process, timeout=5):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass


def completion_status(*, failed, budget_stop):
    return "FAILED" if failed else "BUDGET_COMPLETE" if budget_stop else "FINISHED"


def verify_quota(percent, *, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    """The author and every evaluator must inherit this unit's actual finite cgroup limit."""
    groups = [line.split(":", 2)[2] for line in proc.read_text().splitlines() if line.startswith("0::")]
    if len(groups) != 1:
        raise ValueError("launch requires one unified cgroup")
    maximum, period = (root / groups[0].lstrip("/") / "cpu.max").read_text().split()
    if maximum == "max" or int(period) <= 0 or int(maximum) * 100 != int(period) * percent:
        raise ValueError("actual service CPU quota differs from prepared study budget")


def serve(path, *, timeout=900):
    rid, started, source_sha, req = path.name[4:-5], time.time_ns(), None, {}
    try:
        if path.is_symlink() or path.stat().st_size > 65536:
            raise ValueError("invalid request file")
        req = json.loads(path.read_bytes())
        label = request_label(req)
        with pinned_source(WS, label) as source:
            source_sha = shim._submission_digest(source)
        if req.get("submission_sha256") != source_sha:
            raise ValueError("source changed since request; submit current bytes")
        (CH / ("claimed_" + rid)).write_text("offline host\n")
        with selected():
            if req.get("action") == "reference-analysis":
                if C["reference_guidance"] != "on":
                    raise ValueError("CCA/reference-analysis endpoint is disabled for this arm")
                response = reference_analysis.broker_response(
                    req,
                    workspace=WS,
                    facts=Path(C["facts"]),
                    compiler_sha256=source_sha,
                    rid=rid,
                    started_ns=started,
                    descriptor=Path(C["descriptor"]),
                    target="gemmini",
                    repo=F,
                )
            else:
                if req.get("suite") != "scalability" or req.get("sim") not in (None, "spike"):
                    raise ValueError("offline checks use --suite scalability --sim spike")
                from merlin_experiments.phase1.feedback.public_scalability import request_policy

                _, tiers = request_policy(req.get("sim"), req.get("tiers", ""))
                names = req.get("capsules", "").split(",")
                if not 1 <= len(names) <= 6 or any(n not in CASES for n in names):
                    raise ValueError("choose 1 to 6 explicit training case names")
                with CHECK_LOCK:
                    if time.monotonic() >= END:
                        raise TimeoutError("study budget ended before evaluator slot")
                    job = PRIVATE / "checks" / rid
                    job.mkdir(parents=True)
                    with pinned_source(WS, label) as source:
                        before = shim._submission_digest(source)
                        shutil.copytree(
                            source,
                            job / "submission",
                            symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".git", "build", "selfcheck_out"),
                        )
                        if any(p.is_symlink() for p in (job / "submission").rglob("*")):
                            raise ValueError("source links are not accepted")
                        if before != source_sha or shim._submission_digest(source) != before:
                            raise ValueError("source changed during private snapshot; retry")
                    if shim._submission_digest(job / "submission") != source_sha:
                        raise ValueError("private source snapshot differs")
                    from merlin_experiments.phase1.brokers import selfcheck_tuning

                    req = selfcheck_tuning.apply_request(job / "submission", req, digest=shim._submission_digest)
                    output = job / "result.json"
                    argv = [
                        C["python"],
                        "-B",
                        str(Path(__file__).resolve()),
                        "evaluate",
                        "--config",
                        C["config"],
                        "--submission",
                        str(job / "submission"),
                        "--capsules",
                        ",".join(names),
                        "--out",
                        str(output),
                        "--timeout",
                        str(timeout),
                        "--tiers",
                        ",".join(tiers),
                    ]
                    with (job / "worker.log").open("wb") as log:
                        process = subprocess.Popen(
                            argv, env=ENV, cwd=WS, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                        )
                        CHILDREN.add(process.pid)
                        try:
                            process.wait(timeout=max(1, min(timeout + 30, END - time.monotonic())))
                        finally:
                            terminate(process)
                            CHILDREN.discard(process.pid)
                    if not output.is_file():
                        raise RuntimeError("existing evaluator produced no result; inspect private worker log")
                    response = json.loads(output.read_bytes())
                    response.update(
                        offline_mode="existing_public_functional_and_provisional_model", timing_qualified=False
                    )
                    response.update(selfcheck_tuning.response_fields(req))
            response = _decorate_response(
                json.dumps(response), req, rid, source_sha, started_ns=started, completed_ns=time.time_ns()
            )
    except Exception as error:
        response = dict(error=str(error), all_pass=False, status="offline_request_failed")
    response.update(
        protocol=3,
        selfcheck_request_id=rid,
        selfcheck_submission=req.get("submission", "submission"),
        submission_sha256=source_sha,
        official=False,
    )
    write(CH / ("resp_" + rid + ".json"), response)
    (CH / ("done_" + rid)).write_text("ok\n")


def sandbox(payload, home):
    bwrap = "/work/merlin/hw/bubblewrap-0.11.0/bin/bwrap"
    argv = base_argv(WS, {}, repo=F, include_claude_home=False, inherit_environment=False)
    argv[0] = bwrap
    readonly = [
        "/usr",
        "/bin",
        "/lib",
        "/lib64",
        "/etc",
        str(Path(C["python"]).parent.parent),
        "/home/ubuntu/.local/share/uv/python",
        "/work/merlin/toolchains/llvm-a47bddcc/install",
        *C.get("read_only_dependencies", []),
    ]
    for path in readonly[5:]:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    binary = "/home/ubuntu/.local/share/codex-0.162.0/bin/codex"
    argv += [
        "--ro-bind",
        binary,
        binary,
        "--ro-bind",
        binary,
        str(Path(binary).with_name("codex-linux-sandbox")),
        "--ro-bind",
        str(Path(binary).with_name("codex-code-mode-host")),
        str(Path(binary).with_name("codex-code-mode-host")),
        "--bind",
        str(WS),
        str(WS),
        "--bind",
        str(home),
        str(home),
        "--bind",
        "/home/ubuntu/.codex/auth.json",
        str(home / "auth.json"),
        "--ro-bind",
        str(WS / "reference-kernels"),
        str(WS / "reference-kernels"),
        *(["--ro-bind", str(WS / "cca-evidence"), str(WS / "cca-evidence")] if C.get("cca_lifting") else []),
        "--ro-bind",
        str(WS / "baseline"),
        str(WS / "baseline"),
        "--ro-bind",
        str(WS / "rtl_facts"),
        str(WS / "rtl_facts"),
        "--ro-bind",
        str(WS / "public-cases"),
        str(WS / "public-cases"),
        "--ro-bind",
        str(WS / "target_contract.yaml"),
        str(WS / "target_contract.yaml"),
        "--ro-bind",
        str(WS / "agent_selfcheck.py"),
        str(WS / "agent_selfcheck.py"),
        *(
            ["--ro-bind", str(WS / "submission/compiler_library"), str(WS / "submission/compiler_library")]
            if C.get("sdk_readonly", C.get("tooling") == "agent_only")
            else []
        ),
        "--ro-bind",
        "/work/merlin/hw/bubblewrap-0.11.0/bin/bwrap",
        "/usr/bin/bwrap",
        "--setenv",
        "CODEX_HOME",
        str(home),
        "--setenv",
        "HOME",
        str(WS / "home"),
        "--setenv",
        "TMPDIR",
        str(WS / "tmp"),
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "PATH",
        str(Path(C["python"]).parent) + ":/work/merlin/toolchains/llvm-a47bddcc/install/bin:/usr/bin:/bin",
        "--setenv",
        "PYTHONPATH",
        str(WS / "submission/compiler_library"),
        "--chdir",
        str(WS),
        "--",
        *payload,
    ]
    return argv, (*tuple(p for p in readonly if Path(p).exists()), str(Path(binary).parent))


def launch(*, preflight=False):
    global END
    if C.get("task_sha256") and digest(WS / "TASK.md") != C["task_sha256"]:
        raise ValueError("prepared study task changed")
    home = PRIVATE / "codex-home"
    _, read_paths = sandbox(["/usr/bin/true"], home)
    prepare_codex_home(
        home,
        model=C.get("model", "gpt-6.1-sol"),
        effort=C.get("effort", "high"),
        workspace=WS,
        candidate_read_paths=read_paths,
        auth_source=Path("/home/ubuntu/.codex/auth.json"),
        max_concurrent_subagents=1,
    )
    # Validate outer sandbox mounts before spending an author request.
    probe, _ = sandbox(["/usr/bin/test", "!", "-e", str(PRIVATE / "config.json")], home)
    subprocess.run(probe, check=True, timeout=20)
    mode = ["!", "-w"] if C.get("sdk_readonly", C.get("tooling") == "agent_only") else ["-w"]
    probe, _ = sandbox(["/usr/bin/test", *mode, str(WS / "submission/compiler_library")], home)
    subprocess.run(probe, check=True, timeout=20)
    # Exercise the provider's native command sandbox, including credential denial.
    probe_script = (
        'test ! -r "$CODEX_HOME/auth.json" && test ! -w "$CODEX_HOME/auth.json" && '
        'for proc_auth in /proc/[0-9]*/root"$CODEX_HOME/auth.json"; '
        'do test ! -r "$proc_auth" || exit 1; done && command -v python3 && python3 --version'
    )
    probe, _ = sandbox(
        [
            "/home/ubuntu/.local/share/codex-0.162.0/bin/codex",
            "sandbox",
            "--permission-profile",
            "merlin-candidate",
            "-C",
            str(WS),
            "--",
            "/bin/sh",
            "-c",
            probe_script,
        ],
        home,
    )
    subprocess.run(probe, check=True, timeout=40)
    if preflight:
        write(PRIVATE / "preflight.json", dict(status="PASS", source=context["compiler_sha256"], author_launched=False))
        return
    END = time.monotonic() + C["wall_s"]
    prompt = (WS / "TASK.md").read_text()
    seen, pending = set(), []
    thread_id = None
    budget_stop = False
    write(
        PRIVATE / "status.json",
        dict(
            status="RUNNING",
            started_unix=time.time(),
            deadline_unix=time.time() + C["wall_s"],
            cpus=sorted(os.sched_getaffinity(0)),
            wall_s=C["wall_s"],
            source=context["compiler_sha256"],
            harness="existing provider and selfcheck",
            reference_guidance=C["reference_guidance"],
        ),
    )
    # The original two dispatch slots allow analysis during the single locked numerical check.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
        early = C.get("early_slot")
        if early and early["case"] in CASES:
            rid = "early-largest-dense"
            req = dict(
                suite="scalability",
                sim="spike",
                capsules=early["case"],
                submission="submission",
                submission_sha256=context["compiler_sha256"],
            )
            write(CH / ("req_" + rid + ".json"), req)
            seen.add("req_" + rid + ".json")
            pending.append(workers.submit(serve, CH / ("req_" + rid + ".json"), timeout=C["early_timeout_s"]))
        turn = 0
        try:
            while time.monotonic() < END - 30:
                turn += 1
                kwargs = dict(
                    model=C.get("model", "gpt-6.1-sol"),
                    effort=C.get("effort", "high"),
                    final_path=WS / "qa" / f"turn-{turn}.md",
                    sandbox="bwrap",
                    codex_bin="/home/ubuntu/.local/share/codex-0.162.0/bin/codex",
                )
                payload = build_resume_cmd(WS, thread_id=thread_id, **kwargs) if thread_id else build_cmd(WS, **kwargs)
                argv, _ = sandbox(payload, home)
                events = PRIVATE / f"author-{turn}.jsonl"
                with events.open("wb") as log, (PRIVATE / f"author-{turn}.stderr").open("wb") as errors:
                    process = subprocess.Popen(
                        argv,
                        cwd=WS,
                        env={"PATH": ENV["PATH"], "LANG": "C.UTF-8", "USER": "ubuntu"},
                        stdin=subprocess.PIPE,
                        stdout=log,
                        stderr=errors,
                        start_new_session=True,
                    )
                    CHILDREN.add(process.pid)
                    process.stdin.write(prompt.encode())
                    process.stdin.close()
                    while process.poll() is None and time.monotonic() < END:
                        write(
                            CH / "broker_health.json",
                            dict(ordinary_capacity=1, variant_capacity=1, dispatch_capacity=2),
                        )
                        for request in CH.glob("req_*.json"):
                            if request.name not in seen:
                                seen.add(request.name)
                                pending.append(workers.submit(serve, request))
                        time.sleep(0.25)
                    if process.poll() is None:
                        budget_stop = time.monotonic() >= END
                        terminate(process)
                    CHILDREN.discard(process.pid)
                for line in events.read_text(errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "thread.started":
                        thread_id = event["thread_id"]
                if process.returncode != 0 and not budget_stop:
                    raise RuntimeError(f"author exited {process.returncode}; see {events} and stderr")
                if not thread_id and not budget_stop:
                    raise RuntimeError("author returned no session identity")
                prompt = (
                    f"Continue the same offline optimization task. {int(END - time.monotonic())} seconds remain. "
                    "Inspect feedback and improve another mechanism or validate a neighboring shape. "
                    "Do not rerun a baseline correctness suite."
                )
        finally:
            for pid in tuple(CHILDREN):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            write(
                PRIVATE / "status.json",
                dict(
                    status=completion_status(
                        failed=sys.exc_info()[0] is not None, budget_stop=budget_stop or time.monotonic() >= END - 30
                    ),
                    stopped_unix=time.time(),
                    thread_id=thread_id,
                    requests=len(seen),
                    author_turns=turn,
                    source=shim._submission_digest(WS / "submission"),
                    physical_timing=False,
                ),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "preflight", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--submission")
    parser.add_argument("--capsules")
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tiers", default="L0,L1,L2")
    args = parser.parse_args()
    if args.action == "evaluate":
        return evaluate(args.config, args.submission, args.capsules, args.out, args.timeout, args.tiers)
    boot(args.config)
    if set(os.sched_getaffinity(0)) != set(C["cpus"]):
        raise ValueError("launch/preflight must inherit selected CPUs; use prepared taskset argv and quota wrapper")
    if digest(__file__) != C["runner"]["sha256"]:
        raise ValueError("runner bytes differ from prepared pin")
    if args.action == "launch" and len(C["cpus"]) > 1:
        verify_quota(C["cpu_quota_percent"])
    return launch(preflight=args.action == "preflight")


if __name__ == "__main__":
    raise SystemExit(main())
