"""Host-only broker for the visible split of Claude's model-check contract."""

from __future__ import annotations

import argparse
import json
import itertools
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from merlin.model_check.bundle import Bundle
from merlin.model_check.models import default_models_path, target_from_contract
from merlin_experiments.phase1.run_inputs import strip_build_state
from merlin_experiments.phase1.compiler_library import child_arguments, selected_child, library_thread, join_background
from merlin_experiments.phase1.context import add_context_arguments, context_argv, resolve_context

PROGRAMS = {"resnet50": ("model",),
            "smolvla": ("prefix_encode", "flow_denoise", "action_decode"),
            "tiny_llama": ("prefill", "decode")}
_GROUP = re.compile(r"[A-Za-z0-9_.@-]+\Z")
REPORT_SCHEMA = "merlin.model_check.report.v1"


def _safe_report(report: object) -> bool:
    text = json.dumps(report, ensure_ascii=True).lower()
    return not any(token in text for token in (
        '"sealed"', '"split": "sealed"', '/private/', 'private_corpus', 'sealed_inputs',
    ))


def _require_compiler_selection(snapshot, manifest, context, *, legacy_library):
    """Both compiler modes require explicit original host selection and inputs."""
    from .. import full_compiler_workspace as full
    from ..feedback.snapshot_execution import selected

    implementation = full.selected() if selected() else None
    bound = legacy_library is not None or implementation is not None
    if bound and (manifest is None or context is None):
        raise ValueError("model child lost its original compiler profile context")
    fork = snapshot / "compiler_library"
    if not bound and fork.is_dir() and any(fork.rglob("*.py")):
        raise ValueError("model child lost its selected compiler library")
    if implementation is not None:
        full.candidate_sources(implementation, snapshot)


def execute_visible(ws: Path, request: dict, *, timeout_s: int = 7200, stop=None,
                    snapshot_source: Path | None = None, context=None,
                    compiler_bundle_manifest: Path | None = None) -> dict:
    """Run one fixed CLI request on a host-only immutable source snapshot."""
    action, model = request.get("action"), request.get("model")
    if action not in {"check", "replay-kernel"} or model not in PROGRAMS or request.get("split") != "visible":
        return {"status": "rejected", "error": "only declared visible model checks are admitted"}
    group, engine, program = request.get("group"), request.get("engine"), request.get("program")
    if action == "replay-kernel" and (
        program not in PROGRAMS[model] or not isinstance(group, str) or len(group) > 200
        or not _GROUP.fullmatch(group) or engine != "spike"
    ):
        return {"status": "rejected", "error": "replay requires a declared program, group, and the spike engine"}
    if action == "check" and (group is not None or engine is not None or program is not None):
        return {"status": "rejected", "error": "check accepts only a model"}
    mode = request.get("mode", "fast")
    if mode not in {"fast", "full"} or (action != "check" and mode != "fast"):
        return {"status": "rejected", "error": "unknown model-check mode"}
    if mode == "full":
        from ..joint_control import require_full_model, selected as joint_campaign

        if joint_campaign() is None:
            return {"status": "rejected", "error": "full checks require a selected joint campaign"}
        try:
            timeout_s = min(timeout_s, require_full_model(engine="spike"))
        except (ValueError, RuntimeError) as exc:
            return {"status": "deferred", "error": str(exc)}
    source = snapshot_source if snapshot_source is not None else ws / "submission"
    if source.is_symlink() or not (source / "manifest.yaml").is_file():
        return {"status": "pending", "error": "no stable submission manifest"}
    if any(path.is_symlink() for path in source.rglob("*")):
        return {"status": "rejected", "error": "model-check source contains a symlink"}
    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir or not Path(tmpdir).is_absolute():
        return {"status": "error", "error": "host TMPDIR is not task-owned"}
    with tempfile.TemporaryDirectory(prefix="model-check-", dir=tmpdir) as scratch:
        snapshot = Path(scratch) / "submission" if snapshot_source is None else snapshot_source
        from merlin.common.tree_hash import hash_tree

        if snapshot_source is None:
            before = hash_tree(source)["sha256"]
            shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
            strip_build_state(snapshot)
            if hash_tree(source)["sha256"] != before:
                return {"status": "pending", "error": "submission changed while taking model-check snapshot"}

        digest = hash_tree(snapshot)["sha256"]
        report_path = Path(scratch) / "report.json"
        target = target_from_contract()
        bundle = Bundle.default(target).root
        python, models = sys.executable, default_models_path(target)
        if mode == "full":
            from merlin.integrations.model2mlir import capture_python

            python = str(capture_python())
            models = bundle / "configuration" / "models.yaml"
            if not models.is_file():
                return {"status": "error", "error": "full model bundle has no bound models configuration"}
        # A subprocess cannot inherit context-local compiler owners. Carry the
        # original reviewed library and profile explicitly, as selfcheck does.
        from merlin.targetgen.compiler_library import active_compiler_library
        from ..feedback.compiler_profile_runtime import active_profile_service
        service = active_profile_service() if compiler_bundle_manifest is None else None
        if service is not None:
            compiler_bundle_manifest = service.bundle_manifest
            if context is None:
                from ..context import load_context
                context = load_context(os.environ["MERLIN_TARGET_EXPERIMENT"], repo=service.repo)
        _require_compiler_selection(snapshot, compiler_bundle_manifest, context,
                                    legacy_library=active_compiler_library())
        argv = [python, "-m", "merlin_experiments.phase1.brokers.model_check_worker", action, "--submission", str(snapshot),
                "--model", model, "--split", "visible", "--agent-view", "--out", str(report_path),
                "--work", str(Path(scratch) / "work"), "--timeout-s", str(timeout_s),
                "--bundle", str(bundle), "--models", str(models)]
        if action == "check":
            argv += ["--mode", mode, "--engine", "spike"]
            if mode == "full":
                argv += ["--full-comparison", "quantized-reference"]
        else:
            argv += ["--program", program, "--group", group, "--engine", engine]
        argv += list(child_arguments())
        if compiler_bundle_manifest is not None:
            if context is None:
                raise ValueError("model compiler profile requires explicit invocation context")
            argv += ["--compiler-bundle-manifest", str(compiler_bundle_manifest), *context_argv(context)]
        stdout_path, stderr_path = Path(scratch) / "stdout", Path(scratch) / "stderr"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(argv, cwd=ws, stdout=stdout_file, stderr=stderr_file,
                                       start_new_session=True)
            deadline = time.monotonic() + timeout_s
            while process.poll() is None:
                if (stop is not None and stop()) or time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return {"status": "cancelled" if stop is not None and stop() else "timeout",
                            "model": model, "source_sha256": digest}
                time.sleep(0.25)
        if hash_tree(snapshot)["sha256"] != digest:
            return {"status": "error", "error": "model-check compiler snapshot changed during execution",
                    "model": model, "source_sha256": digest}
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
        if report is not None and (
            not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA
            or report.get("command") != action or report.get("model") != model
            or report.get("split") != "visible" or report.get("agent_view") is not True
            or report.get("status") not in {"pass", "fail", "error"}
        ):
            return {"status": "error", "error": "model-check report identity or schema mismatch",
                    "model": model, "source_sha256": digest}
        if report is not None and not _safe_report(report):
            return {"status": "error", "error": "model-check report contained sealed or private data",
                    "model": model, "source_sha256": digest}
        status = "unavailable" if report is None or report.get("status") == "error" else report["status"]
        if report is not None and report.get("exit_code") != process.returncode:
            status = "error"
        return {"status": status,
                "model": model, "action": action, "source_sha256": digest,
                "returncode": process.returncode, "report": report,
                **({"error": stderr[-2000:]} if process.returncode and report is None else {})}


DEFAULT_MAX_ACTIVE_REQUESTS = 4
MAX_QUEUED_REQUESTS = 32
MAX_REQUEST_BYTES = 64 << 10


def _request(path):
    # Request contents are data, never execution/profile/library selection.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("model-check request exceeds metadata limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("model-check request must be an object")
    return value


def _publish(channel, job_id, result):
    with tempfile.NamedTemporaryFile(mode="w", prefix=".model-response-", dir=channel,
                                     delete=False, encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, channel / f"resp_{job_id}.json")


def serve(ws, *, poll, stop, context=None, compiler_bundle_manifest=None,
          max_active=DEFAULT_MAX_ACTIVE_REQUESTS):
    """Concurrent independently snapshotted requests share the existing native pool.

    There is no executor submission queue or per-model CPU reservation. The
    bounded file-channel waiting set is descriptive; excess requests receive a
    deferred response. Original library scopes remain alive until every worker
    is joined, including STOP and owner loss. Existing exact caches are unchanged.
    """
    if type(max_active) is not int or not 1 <= max_active <= MAX_QUEUED_REQUESTS:
        raise ValueError("model-check max-active must be1..32")
    channel = ws / ".model_check_channel"
    channel.mkdir(exist_ok=True)
    completed = {path.stem.removeprefix("resp_") for path in channel.glob("resp_*.json")}
    active = {}
    shutdown = threading.Event()

    def stopping():
        return shutdown.is_set() or stop()

    def pending():
        candidates = (path for path in channel.glob("req_*.json")
                      if path.stem.removeprefix("req_") not in completed
                      and path.stem.removeprefix("req_") not in active)
        return sorted(itertools.islice(candidates, MAX_QUEUED_REQUESTS + 1))

    def work(request, box):
        try:
            box['result'] = execute_visible(ws, request, context=context,
                compiler_bundle_manifest=compiler_bundle_manifest, stop=stopping)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            box['result'] = {"status":"error", "error":f"{type(exc).__name__}: {exc}"}

    try:
        while True:
            for job_id, (thread, box) in list(active.items()):
                if thread.is_alive():
                    continue
                try:
                    join_background(thread)
                    result = box.get('result', {"status":"error", "error":"model worker returned no report"})
                except (ValueError, RuntimeError) as exc:
                    result = {"status":"error", "error":f"original model worker custody failed: {exc}"}
                _publish(channel, job_id, result)
                completed.add(job_id)
                del active[job_id]
            waiting = pending()
            if stopping():
                for path in waiting:
                    job_id = path.stem.removeprefix("req_")
                    _publish(channel, job_id, {"status":"cancelled", "error":"broker stopped before admission"})
                    completed.add(job_id)
                if not active:
                    return 0
            else:
                if len(waiting) > MAX_QUEUED_REQUESTS:
                    path = waiting.pop()
                    job_id = path.stem.removeprefix("req_")
                    _publish(channel, job_id, {"status":"deferred", "error":"model-check pending queue is full"})
                    completed.add(job_id)
                for path in waiting[:max_active - len(active)]:
                    if stopping():
                        break
                    job_id = path.stem.removeprefix("req_")
                    try:
                        request = _request(path)
                    except (OSError, ValueError, TypeError) as exc:
                        _publish(channel, job_id, {"status":"error", "error":f"{type(exc).__name__}: {exc}"})
                        completed.add(job_id)
                        continue
                    box = {}
                    # library_thread transfers only the original live library
                    # and shared-pass custody; profile/context are explicit.
                    thread = library_thread(lambda request=request, box=box: work(request, box),
                                            name="model-check-" + job_id)
                    active[job_id] = (thread, box)
                    thread.start()
            time.sleep(poll)
    finally:
        # Also join if publication or channel I/O fails; never leave model
        # children running after their original library lifetime has exited.
        shutdown.set()
        failures = []
        for thread, _box in active.values():
            try:
                join_background(thread)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError("model broker worker custody failed during drain") from failures[0]


@selected_child
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    add_context_arguments(parser)
    parser.add_argument("--compiler-bundle-manifest", type=Path)
    parser.add_argument("--ws", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--max-active", type=int, help="Concurrent model requests, up to shared CPU capacity; default min(4,capacity) bounds concurrent model working sets")
    args = parser.parse_args(argv)
    if not 0.01 <= args.poll <= 10:
        parser.error("poll interval must be0.01..10seconds")
    from merlin.common.cpu_pool import compute_capacity
    capacity = compute_capacity() or len(os.sched_getaffinity(0))
    from ..feedback import snapshot_execution
    default_active = capacity if snapshot_execution.selected() else min(DEFAULT_MAX_ACTIVE_REQUESTS, capacity)
    max_active = default_active if args.max_active is None else args.max_active
    if not 1 <= max_active <= min(capacity, MAX_QUEUED_REQUESTS):
        parser.error("max-active must fit the existing shared CPU capacity and bounded request window")
    context = resolve_context(args, parser) if args.compiler_bundle_manifest is not None else None
    ws = args.ws.resolve(strict=True)
    parent = os.getppid()
    return serve(ws, poll=args.poll, context=context, max_active=max_active,
                 compiler_bundle_manifest=args.compiler_bundle_manifest,
                 stop=lambda: (ws / ".model_check_channel/STOP").exists() or os.getppid() != parent)


if __name__ == "__main__":
    from merlin.common.cpu_pool import AUTHOR, cpu_priority

    with cpu_priority(AUTHOR, export=True):  # Author-requested model checks lease first.
        raise SystemExit(main())
