"""Bounded host analysis with candidate execution through an existing sandbox policy.

This module does not invent a sandbox. The host supplies command prefixes built by its
existing inner-command policy, binding only the immutable compiler and dedicated scratch.
Only JSON crosses the worker boundary; submitted compiler Python is never imported here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


_RESULT_TRANSPORT_GRACE_SECONDS = 5.0


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_output(path: Path) -> str:
    """Never follow a compiler-created link into a host-only answer or control file."""
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=parent_fd)
        except FileNotFoundError:
            return ""
    finally:
        os.close(parent_fd)
    with os.fdopen(fd) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("compiler artifact is not a regular file")
        return stream.read()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"non-JSON worker result: {type(value).__name__}")


def run_sandboxed_entrypoint(package: Any, name: str, interface: Path,
                            output_json: Path | None = None, *,
                            sandbox: Mapping[str, Any], timeout_s: float,
                            own_process_group: bool = True) -> subprocess.CompletedProcess:
    """Invoke manifest argv inside a host-created inner policy; never import the package.

    The same helper can serve short-probe compilation. Inside an analysis worker, children
    share its process group so its single wall-clock deadline also terminates compilers.
    """
    from merlin.targetgen import oot_runner as OR

    if timeout_s <= 0:
        raise TimeoutError("compiler wall-clock budget exhausted")
    prefix = list(sandbox["command_prefix"])
    if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
        raise ValueError("compiler execution requires the existing clear-environment bwrap policy")
    if Path(package.directory).resolve() != Path(sandbox["package_path"]).resolve():
        raise ValueError("sandbox compiler identity does not match submitted package")
    argv = OR._resolve_argv(package, name, Path(interface).resolve(),
                            Path(output_json).resolve() if output_json else None)
    if OR._needs_interpreter(package, argv):
        argv = [sys.executable, *argv]
    command = [*prefix, *argv]
    process = subprocess.Popen(command, cwd=str(package.directory), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=own_process_group)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if own_process_group:
            _kill_group(process.pid)
        elif process.poll() is None:
            process.kill()
        process.wait()


class IsolatedAnalysisWorker:
    """Analyzer-compatible host callable with one subprocess deadline for all analysis.

    ``sandbox_factory(baseline, candidate, scratch)`` returns ``baseline`` and ``candidate``
    records containing ``package_path`` and ``command_prefix``. Prefixes are the existing
    host ``inner_command(..., argv=[marker])`` result without its last marker. Both policies
    must mount ``scratch`` read-write and compiler/dependency inputs read-only. Request,
    result and logs remain outside that scratch, inaccessible to compiler subprocesses.
    """
    def __init__(self, *, stage_path: Path, sandbox_factory: Callable[..., Mapping[str, Any]],
                 output: Path):
        self.stage_path = Path(stage_path).resolve()
        self.sandbox_factory = sandbox_factory
        self.output = Path(output).resolve()
        self.completed_sandboxes: Mapping[str, Any] | None = None

    def __call__(self, baseline: Path, candidate: Path, sentinel: Any, *, timeout_s: float,
                 artifact_sink: Callable[..., Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        from tempfile import mkdtemp

        if not 0 < timeout_s <= 600:
            raise ValueError("analysis worker requires a positive wall-clock budget at most 600s")
        started = time.monotonic()
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(mkdtemp(prefix="analysis_", dir=self.output))
        scratch = work / "compiler_scratch"
        scratch.mkdir()
        verifier = kwargs.pop("global_plan_verifier", None)
        if verifier is not None:
            if "<locals>" in verifier.__qualname__:
                raise ValueError("worker verifier must be a host importable function")
            kwargs["verifier_import"] = [verifier.__module__, verifier.__qualname__]
        sandboxes = self.sandbox_factory(Path(baseline), Path(candidate), scratch)
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("analysis preparation exhausted its wall-clock budget")
        # The child must stop before the caller's deadline so it can atomically serialize a
        # potentially large whole-model result and the parent can read/rebind it.  Cap the reserve
        # so short unit/probe budgets still spend most of their time doing useful analysis.
        result_grace = min(_RESULT_TRANSPORT_GRACE_SECONDS, max(0.05, remaining * 0.1))
        analysis_timeout = remaining - result_grace
        if analysis_timeout <= 0:
            raise TimeoutError("analysis preparation left no child execution budget")
        request = {
            "stage_path": str(self.stage_path), "baseline": str(Path(baseline).resolve()),
            "candidate": str(Path(candidate).resolve()), "sentinel": dataclasses.asdict(sentinel),
            "timeout_s": analysis_timeout, "kwargs": kwargs,
            "sandboxes": sandboxes,
            "scratch": str(scratch), "result": str(work / "result.json"),
        }
        request_path = work / "request.json"
        request_path.write_text(json.dumps(request, default=_json_default))
        request_path.chmod(0o600)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            str(Path(value or ".").resolve())
            for value in environment.get("PYTHONPATH", "").split(os.pathsep))
        remaining = timeout_s - (time.monotonic() - started)
        process = None
        status = "failed"
        try:
            if remaining <= 0:
                raise TimeoutError("analysis preparation exhausted its wall-clock budget")
            with (work / "worker.stdout").open("wb") as stdout, (work / "worker.stderr").open("wb") as stderr:
                process = subprocess.Popen(
                    [sys.executable, "-m", __name__, str(request_path)], env=environment,
                    stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    status = "timeout"
                    raise TimeoutError(f"whole-model analysis exceeded {timeout_s}s; worker receipt: {work}") from exc
            if process.returncode != 0 or not (work / "result.json").is_file():
                raise RuntimeError(f"analysis worker failed with rc={process.returncode}; artifacts: {work}")
            result = json.loads((work / "result.json").read_text())
            if result.get("failure"):
                raise RuntimeError(f"{result['failure']}; worker artifacts: {work}")
            if artifact_sink is not None:
                artifact_sink(result["artifacts"])
            self.completed_sandboxes = request["sandboxes"]
            status = "completed"
            return result["analysis"]
        finally:
            if process is not None:
                _kill_group(process.pid)
                process.wait()
            (work / "receipt.json").write_text(json.dumps({
                "schema": "bounded_host_analysis_worker_v1", "status": status,
                "wall_seconds": time.monotonic() - started, "budget_seconds": timeout_s,
                "analysis_budget_seconds": analysis_timeout,
                "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                "worker_pid": process.pid if process else None,
                "process_group_cleanup": process is not None,
                "candidate_execution": "existing_inner_bwrap_policy", "simulation": False,
            }, indent=2))


def _worker(request_path: Path) -> int:
    request = json.loads(request_path.read_text())
    stage_path = Path(request["stage_path"])
    sys.path.insert(0, str(stage_path.parent))
    spec = importlib.util.spec_from_file_location(stage_path.stem, stage_path)
    if spec is None or spec.loader is None:
        raise ValueError("host analyzer module is unavailable")
    stage = importlib.util.module_from_spec(spec)
    sys.modules[stage_path.stem] = stage
    spec.loader.exec_module(stage)
    started = time.monotonic()
    deadline = started + request["timeout_s"]
    retained: dict[str, Any] = {}

    def emit_pair(package: Any, interface: Path, unused_scratch: Path, tag: str,
                  timeout_s: int) -> tuple[int, str, str]:
        from merlin.targetgen import oot_runner as OR

        scratch = Path(request["scratch"]) / tag
        scratch.mkdir()
        source = scratch / "interface.mlir"
        source.write_bytes(interface.read_bytes())
        output = scratch / "command_buffer.json"
        rows = []
        results = []

        def persist():
            diagnostics = {"schema": "compiler_emission_diagnostics_v1", "arm": tag, "entrypoints": rows}
            (request_path.parent / f"{tag}_emission.json").write_text(json.dumps(diagnostics))
            (unused_scratch / f"emission_{tag}.json").write_text(json.dumps(diagnostics))

        entrypoints = OR.analysis_emission_entrypoints(package)
        for name in entrypoints:
            destination = output if name in ("emit_command_buffer", "emit_analysis_bundle") else None
            try:
                result = run_sandboxed_entrypoint(
                    package, name, source, destination, sandbox=request["sandboxes"][tag],
                    timeout_s=min(timeout_s, deadline - time.monotonic()), own_process_group=False)
            except (subprocess.TimeoutExpired, TimeoutError) as exc:
                stderr = getattr(exc, "stderr", None) or ""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                rows.append({"command": name, "returncode": None,
                             "exception": type(exc).__name__, "stderr_tail": stderr[-4096:]})
                (request_path.parent / f"{tag}_{name}.stderr").write_text(stderr)
                persist()
                raise
            results.append(result)
            rows.append({"command": name, "returncode": result.returncode,
                         "stderr_tail": (result.stderr or "")[-4096:]})
            (request_path.parent / f"{tag}_{name}.stderr").write_text(result.stderr or "")
            persist()
            if result.returncode != 0:
                # A failed compiler may leave a malformed/linked partial artifact. Preserve
                # its first error without parsing that artifact or invoking the next emitter.
                return result.returncode, "", ""
        target_result = results[0] if entrypoints == ("emit_analysis_bundle",) else results[1]
        (request_path.parent / f"{tag}_lowered.mlir").write_text(target_result.stdout or "")
        return (next((result.returncode for result in results if result.returncode), 0),
                target_result.stdout or "", _read_output(output))

    def machine_audit(lowered_text: str, *, arm: str, timeout_s: float) -> dict[str, Any]:
        from merlin.runtime.backends.base import get_backend
        backend = get_backend(str(request["kwargs"]["target"]))
        analyzer = getattr(backend, "analyze_machine_artifact", None)
        if analyzer is None:
            raise ValueError("registered target has no compiled-machine activity adapter")
        prefix = list(request["sandboxes"][arm]["command_prefix"])
        if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
            raise ValueError("machine assembly requires the existing answer-masked inner policy")

        def run_command(argv: list[str], *, timeout_s: float) -> subprocess.CompletedProcess:
            remaining = min(timeout_s, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("machine assembly exceeded whole-model analysis deadline")
            command = [*prefix, *map(str, argv)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)  # Same worker process group; parent owns cleanup.
            try:
                stdout, stderr = process.communicate(timeout=remaining)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()

        return analyzer(lowered_text, workdir=Path(request["scratch"]) / f"machine_{arm}",
                        run_command=run_command, timeout_seconds=min(timeout_s, deadline-time.monotonic()))

    try:
        kwargs = request["kwargs"]
        verifier_import = kwargs.pop("verifier_import", None)
        if verifier_import:
            verifier = importlib.import_module(verifier_import[0])
            for name in verifier_import[1].split("."):
                verifier = getattr(verifier, name)
            kwargs["global_plan_verifier"] = verifier
        from merlin.runtime.backends.base import get_backend
        try:
            backend = get_backend(str(kwargs.get("target", "")))
            identity_provider = getattr(backend, "machine_artifact_policy_identity", None)
            build_policy = identity_provider() if identity_provider is not None else None
        except (ImportError, KeyError, ValueError):
            build_policy = None  # Optional compiled-machine evidence cannot invent target support.
        analysis = stage.analyze_whole_model_emission(
            Path(request["baseline"]), Path(request["candidate"]),
            stage.StageE2ESentinel(**request["sentinel"]), timeout_s=request["timeout_s"],
            artifact_sink=retained.update, emit_pair_runner=emit_pair,
            machine_artifact_auditor=machine_audit, machine_build_policy_identity=build_policy, **kwargs)
        retained.pop("parsed_lowered_module", None)
        result = {"analysis": analysis, "artifacts": retained}
    except Exception as exc:
        result = {"failure": {"type": type(exc).__name__, "reason": str(exc)[:20000]}}
        result["failure"]["emission_diagnostics"] = {
            tag: json.loads(path.read_text()) for tag in ("baseline", "candidate")
            if (path := request_path.parent / f"{tag}_emission.json").is_file()}
    Path(request["result"]).write_text(json.dumps(result, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker(Path(sys.argv[1])))
