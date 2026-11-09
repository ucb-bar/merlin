"""Minimal isolated offline study using the existing provider, broker and evaluator.

Launch is explicit. Preparation never starts an author. Each process and its evaluator inherit one
selected CPU. Host-only configuration, models and oracle answers remain outside the author mount.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
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
    if C["reference_guidance"] == "on" and C.get("cca_lifting"):
        os.environ["MERLIN_REFERENCE_CAPTURE_SELECTION"] = json.dumps(C["cca_lifting"], sort_keys=True)
    from merlin.experiments.phase1.brokers.selfcheck import _decorate_response
    from merlin.experiments.phase1.brokers.selfcheck_sources import pinned_source, request_label
    from merlin.experiments.phase1.feedback import reference_analysis, reference_experiments
    from merlin.experiments.phase1.providers.codex_agent import build_cmd, build_resume_cmd, prepare_codex_home
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
                    goals=row.get("goals") or C.get("reference_goals", []),
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


def abstraction_feedback():
    """The prepared advisory feedback selection; absent for configurations that predate it."""
    selected = C.get("abstraction_feedback") or {}
    if selected and selected.get("schema") != "private.kernel_abstractions.abstraction_feedback.v1":
        raise ValueError("unknown abstraction feedback selection")
    return selected


def seed_selection():
    """A private copy of the starting compiler, verified against the prepared initial digest."""
    if not abstraction_feedback().get("seed_fallback"):
        return None
    from merlin.experiments.phase1.feedback import abstraction_feedback as AF

    root = PRIVATE / "seed-emission"
    source = root / "source"
    if not source.is_dir():
        staging = root / "source.next"
        shutil.rmtree(staging, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            WS / "baseline",
            staging,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", ".git", "build", "selfcheck_out"),
        )
        if any(p.is_symlink() for p in staging.rglob("*")):
            raise ValueError("seed source links are not accepted")
        if shim._submission_digest(staging) != C["initial_submission_sha256"]:
            raise ValueError("baseline differs from the prepared initial compiler")
        staging.rename(source)
    if shim._submission_digest(source) != C["initial_submission_sha256"]:
        raise ValueError("private seed copy differs from the prepared initial compiler")
    return AF.select_seed(source, cache=root / "cache")


def sweep_request(request):
    """Validate an author-named coverage sweep against the prepared selection and training cases."""
    from merlin.experiments.phase1.feedback.abstraction_feedback import parse_shapes

    selected = abstraction_feedback().get("coverage_sweep")
    if not selected:
        raise ValueError("coverage sweeps are not enabled for this run")
    if request.get("suite") != "scalability" or request.get("sim") not in (None, "spike"):
        raise ValueError("coverage sweeps use --suite scalability --sim spike")
    names = [v for v in str(request.get("capsules", "")).split(",") if v]
    if len(names) != 1 or names[0] not in CASES:
        raise ValueError("coverage sweep needs exactly one public training case as its family template")
    shapes = parse_shapes(request.get("coverage_sweep"), max_shapes=int(selected.get("max_shapes", 6)))
    numerical = request.get("coverage_sweep_numerical") is True
    if numerical and not selected.get("numerical"):
        raise ValueError("functional checks of sweep shapes are not enabled for this run")
    return names[0], shapes, numerical


def resize_case(cb, shape):
    """The study's existing case-resizing path for one author-named shape of the template's family."""
    spec = importlib.util.spec_from_file_location("kernel_abstraction_inputs", Path(__file__).with_name("inputs.py"))
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    resize_dense, same_conv_case = recipe.resize_dense, recipe.same_conv_case
    opcodes = {command["opcode"] for command in cb["commands"]}
    if "MATMUL_RESIDENT" in opcodes and set(shape) <= {"m", "k", "n"}:
        products = [c for c in cb["commands"] if c["opcode"] == "MATMUL_RESIDENT"]
        n = shape.get("n")
        return resize_dense(
            cb,
            m=shape.get("m", cb["tensors"][products[0]["operands"]["lhs"]]["shape"][0]),
            k=shape.get("k"),
            ns=None if n is None else [n] * len(products),
        )
    if "CONV2D" in opcodes and set(shape) <= {"batch", "spatial", "ci", "co"}:
        conv = next(c for c in cb["commands"] if c["opcode"] == "CONV2D")
        ifm = cb["tensors"][conv["operands"]["ifm"]]["shape"]
        kernel, padding = conv["attributes"]["kernel"], conv["attributes"].get("padding") or [0, 0, 0, 0]
        spatial = ifm[1] - kernel[0] + 1 + padding[0] + padding[1]
        return same_conv_case(
            cb,
            batch=shape.get("batch", ifm[0]),
            spatial=shape.get("spatial", spatial),
            input_channels=shape.get("ci", kernel[2]),
            output_channels=shape.get("co", kernel[3]),
        )
    raise ValueError("this template family accepts m/k/n (dense) or batch/spatial/ci/co (SAME K3 convolution)")


def sweep(config, submission, template, shapes, output, timeout, *, numerical=False):
    """Compile author-named shapes of one public template's family with the original package path."""
    boot(config, context_setup=False)
    from merlin.experiments.phase1 import full_compiler_workspace as full
    from merlin.experiments.phase1.context import load_context
    from merlin.experiments.phase1.feedback import abstraction_feedback as AF

    ctx = load_context(C["descriptor"], repo=C["framework"])
    owner = full.select(**C["full_compiler"]) if C.get("full_compiler") else None
    seed = seed_selection()
    selected = abstraction_feedback().get("coverage_sweep") or {}
    with full.scoped_full_compiler(owner), AF.seeded(seed):
        result = AF.coverage_sweep(
            submission=Path(submission),
            public_cases=Path(C["suite"]) / "cases",
            allowed_templates=CASES,
            template=template,
            shapes=json.loads(shapes),
            resize=resize_case,
            output=Path(output).with_suffix(".sweep"),
            contract=ctx.repo / "src/merlin/resources/contracts",
            context=ctx,
            timeout=timeout,
            numerical=numerical,
            max_shapes=int(selected.get("max_shapes", 6)),
        )
    write(Path(output), result)
    return 0


def evaluation_request(request):
    """Keep source diagnostics distinct from ordinary numerical checks."""
    from merlin.experiments.phase1.brokers.selfcheck import FORWARDED_PROBES
    from merlin.experiments.phase1.feedback.public_scalability import request_policy
    from merlin.experiments.phase1.feedback.public_scalability_inspection import requested

    if request.get("suite") != "scalability" or request.get("sim") not in (None, "spike"):
        raise ValueError("offline checks use --suite scalability --sim spike")
    mode = requested(request)
    if mode == "compile_only":
        raise ValueError(
            "offline compile-only requires an original native compiler-bundle owner; no numerical check was substituted"
        )
    other_probes = bool(
        request.get("caller_layout")
        or any(request.get(key) for key in FORWARDED_PROBES if key != "candidate_diagnostics")
    )
    _, tiers = request_policy(request.get("sim"), request.get("tiers", ""), inspection=other_probes)
    return tiers, ["--candidate-diagnostics"] if mode == "candidate_diagnostics" else []


def evaluate(config, submission, capsules, output, timeout, tiers="L0,L1,L2", *, candidate_diagnostics=False):
    """The original evaluator CLI, with one worker and the selected original runtime."""
    boot(config, context_setup=False)
    from merlin.experiments.phase1 import full_compiler_workspace as full
    from merlin.experiments.phase1.context import load_context
    from merlin.experiments.phase1.feedback.public_scalability import install
    from merlin.experiments.phase1.feedback.selfcheck import main as selfcheck
    from merlin.targetgen.target_experiment import load_target_experiment

    if type(candidate_diagnostics) is not bool:
        raise ValueError("candidate-diagnostics selection must be boolean")
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
    if candidate_diagnostics:
        args.append("--candidate-diagnostics")
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
    from merlin.experiments.phase1.feedback.abstraction_feedback import seeded

    try:
        seed = seed_selection()
    except (OSError, ValueError):
        seed = None  # the per-case field then says no seed emission is selected
    with full.scoped_full_compiler(owner), seeded(seed):
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
            elif req.get("coverage_sweep") is not None:
                template, shapes, numerical = sweep_request(req)
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
                    output = job / "sweep.json"
                    argv = [
                        C["python"],
                        "-B",
                        str(Path(__file__).resolve()),
                        "sweep",
                        "--config",
                        C["config"],
                        "--submission",
                        str(job / "submission"),
                        "--capsules",
                        template,
                        "--shapes",
                        json.dumps(shapes),
                        "--out",
                        str(output),
                        "--timeout",
                        str(timeout),
                    ] + (["--numerical"] if numerical else [])
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
                        raise RuntimeError("coverage sweep produced no result; inspect private worker log")
                    response = json.loads(output.read_bytes())
                    response.update(offline_mode="author_named_shape_sweep", timing_qualified=False)
            else:
                tiers, inspection_argv = evaluation_request(req)
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
                    from merlin.experiments.phase1.brokers import selfcheck_tuning

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
                    ] + inspection_argv
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
    if req.get("action") == "reference-analysis" and response.get("schema") == "merlin.reference_analysis.v1":
        response = reference_analysis._bounded_response(response, workspace=WS, pretty=True)
    write(CH / ("resp_" + rid + ".json"), response)
    (CH / ("done_" + rid)).write_text("ok\n")


def deployment_paths():
    """Resolve operator-selected tools without embedding an account or install root."""
    from merlin.experiments.phase1.providers.codex_agent import real_codex_home

    def executable(key, variable, name):
        value = C.get(key) or os.environ.get(variable) or shutil.which(name)
        if not value:
            raise ValueError(f"select {key} in the study configuration or {variable}")
        path = Path(value).resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"selected {key} is not executable: {path}")
        return path

    llvm = Path(C.get("llvm_install") or ENV["MERLIN_MLIR_INSTALL"]).resolve(strict=True)
    auth = Path(C.get("codex_auth_source") or real_codex_home() / "auth.json").resolve(strict=True)
    return dict(
        bwrap=executable("bwrap", "MERLIN_BWRAP", "bwrap"),
        codex=executable("codex_binary", "CODEX_BIN", "codex"),
        llvm=llvm,
        auth=auth,
        python_runtime=Path(C["python"]).resolve(strict=True).parent.parent,
    )


def sandbox(payload, home):
    paths = deployment_paths()
    bwrap = str(paths["bwrap"])
    argv = base_argv(WS, {}, repo=F, include_claude_home=False, inherit_environment=False)
    argv[0] = bwrap
    readonly = [
        "/usr",
        "/bin",
        "/lib",
        "/lib64",
        "/etc",
        str(Path(C["python"]).parent.parent),
        str(paths["python_runtime"]),
        str(paths["llvm"]),
        *C.get("read_only_dependencies", []),
    ]
    for path in readonly[5:]:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    binary = str(paths["codex"])
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
        str(paths["auth"]),
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
        bwrap,
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
        os.pathsep.join((str(Path(C["python"]).parent), str(paths["llvm"] / "bin"), "/usr/bin", "/bin")),
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
    paths = deployment_paths()
    version = subprocess.run(
        [str(paths["codex"]), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    expected_version = C.get("codex_version", "0.162.0")
    if version != "codex-cli " + expected_version:
        raise ValueError(f"selected Codex version {version!r} differs from {expected_version!r}")
    if C.get("task_sha256") and digest(WS / "TASK.md") != C["task_sha256"]:
        raise ValueError("prepared study task changed")
    # Copy the starting compiler privately before any author edit can reach baseline/.
    seed_selection()
    home = PRIVATE / "codex-home"
    _, read_paths = sandbox(["/usr/bin/true"], home)
    prepare_codex_home(
        home,
        model=C.get("model", "gpt-6.1-sol"),
        effort=C.get("effort", "high"),
        workspace=WS,
        candidate_read_paths=read_paths,
        auth_source=paths["auth"],
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
            str(paths["codex"]),
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
                    codex_bin=str(paths["codex"]),
                )
                payload = build_resume_cmd(WS, thread_id=thread_id, **kwargs) if thread_id else build_cmd(WS, **kwargs)
                argv, _ = sandbox(payload, home)
                events = PRIVATE / f"author-{turn}.jsonl"
                with events.open("wb") as log, (PRIVATE / f"author-{turn}.stderr").open("wb") as errors:
                    process = subprocess.Popen(
                        argv,
                        cwd=WS,
                        env={"PATH": ENV["PATH"], "LANG": "C.UTF-8", "USER": getpass.getuser()},
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
    parser.add_argument("action", choices=("launch", "preflight", "evaluate", "sweep"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--submission")
    parser.add_argument("--capsules")
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tiers", default="L0,L1,L2")
    parser.add_argument("--candidate-diagnostics", action="store_true")
    parser.add_argument("--shapes", help="sweep: JSON list of author-named shapes")
    parser.add_argument("--numerical", action="store_true", help="sweep: also run the functional Spike check")
    args = parser.parse_args()
    if args.action == "sweep":
        return sweep(
            args.config, args.submission, args.capsules, args.shapes, args.out, args.timeout, numerical=args.numerical
        )
    if args.action == "evaluate":
        return evaluate(
            args.config,
            args.submission,
            args.capsules,
            args.out,
            args.timeout,
            args.tiers,
            candidate_diagnostics=args.candidate_diagnostics,
        )
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
