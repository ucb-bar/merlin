"""Plan, preflight, and execute a frozen version-2 paper comparison."""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.baselines import bundle
from merlin.common.artifacts import finish_run, new_product, start_run
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml, write_yaml

from .freeze import sha256_paths
from .paper import MatrixCell, PaperStudySpec, Preflight, validate_paper_result
from .paper_report import build_paper_report, render_markdown, seal_results_document
from .session import validate_capture_session, validate_paper_input_binding


class StudyNotReady(RuntimeError):
    def __init__(self, message: str, output_dir: Path):
        super().__init__(message)
        self.output_dir = output_dir


_EXTERNAL_MEASURED_SECTIONS = frozenset({
    "lifecycle", "correctness", "quality", "timing", "memory", "execution", "provenance",
})
_EXTERNAL_IDENTITY = frozenset({
    "schema_version", "study_label", "target", "model", "checkpoint", "artifact_sha256",
    "backend", "runtime", "precision", "core_count",
})
_EXECUTORCH_COMMAND_FIELDS = frozenset({
    "model", "variant", "cores", "framework_package", "framework_package_sha256",
    "warmups", "observations", "measurement_repeats", "quality_metric", "quality_min",
    "framework_source_sha256",
})
_MIN_CELL_TIMEOUT_SECONDS = 1
_MAX_CELL_TIMEOUT_SECONDS = 86_400


def _cell_timeout_seconds(cell: MatrixCell) -> int:
    value = cell.backend.options.get("timeout")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("timeout must be an integer whole-cell budget in seconds")
    if not _MIN_CELL_TIMEOUT_SECONDS <= value <= _MAX_CELL_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout must be in [{_MIN_CELL_TIMEOUT_SECONDS}, "
            f"{_MAX_CELL_TIMEOUT_SECONDS}] seconds")
    return value


def _remaining_seconds(deadline_ns: int) -> int:
    remaining_ns = int(deadline_ns) - time.monotonic_ns()
    if remaining_ns <= 0:
        raise TimeoutError("paper cell exhausted its frozen whole-cell deadline")
    return max(1, math.ceil(remaining_ns / 1_000_000_000))


def execution_matrix(spec: PaperStudySpec) -> tuple[MatrixCell, ...]:
    """Return the frozen, drift-resistant cell order without changing matrix membership.

    Backends for the same model/precision/core block run close together, while their order and the
    block order are content-stably randomized from the committed seed. This prevents the canonical
    YAML/backend order from being confounded with board time or temperature, yet remains exactly
    replayable and auditable without relying on Python's process-randomized ``hash()``.
    """
    matrix = spec.matrix()
    keys = [cell.key for cell in matrix]
    if len(keys) != len(set(keys)):
        raise ValueError("paper matrix contains duplicate cell identities")
    policy = spec.reporting["execution_order"]
    seed = str(policy["seed_sha256"])
    blocks: dict[tuple[str, str, int], list[MatrixCell]] = {}
    for cell in matrix:
        blocks.setdefault((cell.model.name, cell.precision, cell.core_count), []).append(cell)

    def digest(label: str) -> str:
        return hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()

    ordered: list[MatrixCell] = []
    for block in sorted(blocks, key=lambda value: digest("block:" + ":".join(map(str, value)))):
        cells = sorted(
            blocks[block],
            key=lambda cell: digest(
                "backend:" + ":".join(map(str, block)) + ":" + cell.backend.name))
        ordered.extend(cells)
    if (len(ordered) != len(matrix)
            or {cell.key for cell in ordered} != set(keys)):
        raise ValueError("execution ordering changed paper matrix membership")
    return tuple(ordered)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


@dataclass(frozen=True)
class _PinnedPython:
    source: Path
    sha256: str
    argv0: Path


def _sha256_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(fd, 8 * 1024 * 1024):
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _external_python(cell: MatrixCell) -> _PinnedPython:
    """Resolve and verify the exact Python executable frozen for an external adapter.

    External measurements must not inherit whichever ``python`` happens to be first on ``PATH``.
    This check is deliberately called during preflight and again immediately before execution.
    """
    value = str(cell.backend.options.get("python_executable", ""))
    expected = str(cell.backend.options.get("python_executable_sha256", ""))
    if not value:
        raise ValueError("python_executable is absent")
    path = _resolve_path(value).absolute()
    if not path.is_absolute():  # Defensive: _resolve_path currently makes relative paths absolute.
        raise ValueError("python_executable does not resolve to an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"python_executable cannot be resolved safely: {exc}") from exc
    # Reject a symlink in the final component *or any parent component*, and reject lexical '..'
    # escapes.  The digest is not permission to redirect the open to a different inode.
    if resolved != path:
        raise ValueError("python_executable path contains an escape or symlink")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"python_executable is absent or not executable: {path}")
    if not _is_sha256(expected):
        raise ValueError("python_executable_sha256 is unresolved")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not (opened.st_mode & 0o111):
            raise ValueError("python_executable is not a regular executable file")
        if Path(f"/proc/self/fd/{fd}").resolve() != path:
            raise ValueError("python_executable changed while it was opened")
        actual = _sha256_fd(fd)
    finally:
        os.close(fd)
    if actual != expected:
        raise ValueError(
            f"python executable digest mismatch: expected={expected} actual={actual}")
    argv0_value = str(cell.backend.options.get("python_venv_argv0", ""))
    if not argv0_value:
        raise ValueError("python_venv_argv0 is absent")
    argv0 = _resolve_path(argv0_value).absolute()
    try:
        argv0.parent.resolve(strict=True).relative_to(repo_root().resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("python_venv_argv0 parent escapes the repository") from exc
    if argv0.exists() or argv0.is_symlink():
        raise ValueError("python_venv_argv0 must be a nonexistent, non-symlink semantic argv0")
    if not (argv0.parent.parent / "pyvenv.cfg").is_file():
        raise ValueError("python_venv_argv0 does not name a validated virtual-environment bin dir")
    return _PinnedPython(path, expected, argv0)


def _stage_external_python(cell: MatrixCell, private_dir: Path) -> tuple[_PinnedPython, Path, int]:
    """Copy verified interpreter bytes privately and return an open executable fd.

    The source is revalidated here, after matrix preflight.  Execution uses ``/proc/self/fd/N``
    with the fd inherited by the child, so swapping either the source or staged pathname after this
    function returns cannot redirect ``execve``.  argv[0] remains next to the frozen venv's
    ``pyvenv.cfg`` to preserve the interpreter's package-environment semantics.
    """
    pinned = _external_python(cell)
    private_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    staged = private_dir / "python"
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(pinned.source, source_flags)
    destination_fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o500)
    try:
        with os.fdopen(os.dup(source_fd), "rb") as source, \
                os.fdopen(os.dup(destination_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    os.chmod(staged, 0o500)
    if hashlib.sha256(staged.read_bytes()).hexdigest() != pinned.sha256:
        raise ValueError("private Python staging digest differs from the pinned interpreter")
    # This CPython build does not expose os.memfd_create/fcntl seal constants, so call the Linux
    # libc API directly.  Constants are from linux/memfd.h and linux/fcntl.h.
    libc = ctypes.CDLL(None, use_errno=True)
    memfd_create = getattr(libc, "memfd_create", None)
    if memfd_create is None:
        raise ValueError("sealed memfd execution is unavailable for the pinned Python adapter")
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    execute_fd = int(memfd_create(b"merlin-paper-python", 0x0001 | 0x0002))
    if execute_fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    with staged.open("rb") as source, os.fdopen(os.dup(execute_fd), "wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    os.fchmod(execute_fd, 0o500)
    fcntl.fcntl(execute_fd, 1033, 0x0008 | 0x0002 | 0x0004 | 0x0001)
    if _sha256_fd(execute_fd) != pinned.sha256:
        os.close(execute_fd)
        raise ValueError("private Python bytes changed before execution")
    return pinned, staged, execute_fd


def _framework_package(cell: MatrixCell) -> tuple[str, str]:
    """Return the cell's prebuilt external package path/digest without fallback lookup."""
    packages = cell.backend.options.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("external runtime packages must be a model-keyed mapping")
    by_precision = packages.get(cell.model.name)
    if not isinstance(by_precision, dict):
        raise ValueError(f"external runtime package is absent for model {cell.model.name!r}")
    row = by_precision.get(cell.precision)
    if not isinstance(row, dict):
        raise ValueError(
            f"external runtime package is absent for {cell.model.name}/{cell.precision}")
    path, digest = str(row.get("path", "")), str(row.get("sha256", ""))
    if not path or not digest:
        raise ValueError(
            f"external runtime package path/digest is absent for "
            f"{cell.model.name}/{cell.precision}")
    return path, digest


def _audit_framework_package(cell: MatrixCell, package_value: str,
                             package_digest: str) -> list[str]:
    """Validate one unique prebuilt package (shared by the 1-core and 8-core cells)."""
    errors: list[str] = []
    package_path = _resolve_path(package_value)
    if package_value == "unresolved" or not package_path.is_dir():
        return [f"frozen session package is absent for {cell.model.name}/{cell.precision}: "
                f"{package_value}"]
    if not _is_sha256(package_digest):
        return [f"session package sha256 is unresolved for {cell.model.name}/{cell.precision}"]
    try:
        from merlin.baselines.executorch_session import (
            capture_session_identity, load_session_package, session_identity_sha256,
        )
        from merlin.common.yaml import load_yaml
        frozen = load_session_package(package_path, expected_sha256=package_digest)
        artifact_digest = cell.model.artifacts[cell.precision]["sha256"]
        if (frozen.model != cell.model.capture
                or frozen.variant != cell.model.artifacts[cell.precision]["variant"]):
            errors.append(
                f"session package model/variant differs for {cell.model.name}/{cell.precision}")
        if _is_sha256(artifact_digest) and frozen.capture_sha256 != artifact_digest:
            errors.append(
                f"session package capture digest differs for {cell.model.name}/{cell.precision}")
        capture_path = Path(str(cell.model.artifacts[cell.precision].get("path", "")))
        if not capture_path.is_absolute():
            capture_path = _resolve_path(str(capture_path))
        expected_identity = session_identity_sha256(capture_session_identity(
            load_yaml(capture_path / "session_contract.yaml")))
        if frozen.capture_session_identity_sha256 != expected_identity:
            errors.append(
                f"session package loader/checkpoint/input identity differs for "
                f"{cell.model.name}/{cell.precision}")
        expected_source = str(cell.backend.options.get("framework_source_sha256", ""))
        if (_is_sha256(expected_source)
                and frozen.framework_source_sha256 != expected_source):
            errors.append(
                f"session package source digest differs for {cell.model.name}/{cell.precision}")
        package_rows = cell.backend.options.get("packages", {})
        model_rows = package_rows.get(cell.model.name, {}) if isinstance(package_rows, dict) else {}
        package_row = model_rows.get(cell.precision, {}) if isinstance(model_rows, dict) else {}
        expected_environment = str(package_row.get("build_environment_sha256", ""))
        if (not _is_sha256(expected_environment)
                or frozen.build_environment_sha256 != expected_environment):
            errors.append(
                f"session package build-environment digest differs for "
                f"{cell.model.name}/{cell.precision}")
    except Exception as exc:  # noqa: BLE001 — report every preflight blocker
        errors.append(
            f"invalid frozen session package for {cell.model.name}/{cell.precision}: {exc}")
    return errors


def _capture_contract(spec: PaperStudySpec, cell: MatrixCell, *, verify_digest: bool = True) \
        -> tuple[Path, dict, list[str]]:
    artifact = cell.model.artifacts[cell.precision]
    capture = (Path(artifact["path"]) if artifact.get("path") else
               bundle.resolve(cell.model.capture, artifact["variant"]).root)
    errors: list[str] = []
    if not capture.is_dir():
        return capture, {}, [f"capture directory is absent: {capture}"]
    if verify_digest:
        actual = sha256_paths([capture])
        if actual != artifact["sha256"]:
            errors.append(f"capture digest mismatch: spec={artifact['sha256']} actual={actual}")
    session, session_errors = validate_capture_session(
        capture, cell.model.session,
        expected_provenance=cell.model.expected_provenance)
    provenance = session.get("provenance", {}) if isinstance(session, dict) else {}
    captured_checkpoint = provenance.get("checkpoint") if isinstance(provenance, dict) else None
    if session.get("paper_ready") is True and captured_checkpoint != cell.model.checkpoint:
        errors.append(
            f"session checkpoint differs: study={cell.model.checkpoint!r} "
            f"capture={captured_checkpoint!r}")
    return capture, session, [*errors, *session_errors]


def environment_preflight(spec: PaperStudySpec) -> Preflight:
    """Verify frozen bytes and session inputs without compiling or touching the board."""
    base = spec.preflight()
    errors, blockers, warnings = list(base.errors), list(base.blockers), list(base.warnings)
    paper_inputs = _resolve_path(str(spec.paper_inputs.get("path", "unresolved")))
    if not paper_inputs.is_dir():
        blockers.append(f"paper input bundle is absent: {paper_inputs}")
    elif _is_sha256(spec.paper_inputs.get("sha256")):
        actual_inputs = sha256_paths([paper_inputs])
        if actual_inputs != spec.paper_inputs["sha256"]:
            errors.append(
                "paper input bundle digest mismatch: "
                f"spec={spec.paper_inputs['sha256']} actual={actual_inputs}")
        else:
            errors.extend(validate_paper_input_binding(paper_inputs, spec.models))
    compiler_paths = [Path(value) for value in spec.freeze.get("compiler_source_paths", ()) or ()]
    if not compiler_paths:
        blockers.append("frozen compiler source paths are absent")
    else:
        missing = [str(path) for path in compiler_paths if not path.exists()]
        if missing:
            blockers.append(f"frozen compiler source paths are absent: {missing}")
        elif _is_sha256(spec.freeze.get("compiler_source_sha256")):
            if sha256_paths(compiler_paths) != spec.freeze["compiler_source_sha256"]:
                blockers.append("frozen compiler source digest mismatch")
    authority_value = str(spec.freeze.get("toolchain_authority_path", ""))
    authority_digest = str(spec.freeze.get("toolchain_authority_sha256", ""))
    authority_path = _resolve_path(authority_value) if authority_value else Path()
    if not authority_value or not authority_path.is_file():
        blockers.append(f"frozen toolchain authority is absent: {authority_value}")
    elif _is_sha256(authority_digest):
        try:
            from .paper_toolchain_authority import load_toolchain_authority
            load_toolchain_authority(
                authority_path, expected_sha256=authority_digest,
                expected_target=spec.target)
        except ValueError as exc:
            blockers.append(f"frozen toolchain authority is invalid: {exc}")
    registration_value = str(spec.freeze.get("external_package_registration_path", ""))
    registration_digest = str(spec.freeze.get("external_package_registration_sha256", ""))
    if registration_value:
        registration_path = _resolve_path(registration_value)
        if not registration_path.is_file():
            blockers.append(f"frozen external package registration is absent: {registration_path}")
        elif _is_sha256(registration_digest):
            if hashlib.sha256(registration_path.read_bytes()).hexdigest() != registration_digest:
                blockers.append("frozen external package registration digest mismatch")
    checked_captures: set[tuple[str, str]] = set()
    checked_framework_packages: set[tuple[str, str, str]] = set()
    for cell in spec.matrix():
        capture_key = (cell.model.name, cell.precision)
        if capture_key not in checked_captures:
            artifact_digest = str(cell.model.artifacts[cell.precision].get("sha256", ""))
            digest_ready = _is_sha256(artifact_digest)
            # Draft audits still validate the session semantics/provenance when a capture exists, but
            # skip hashing multi-GB trees until a real digest has been registered.
            _, _, problems = _capture_contract(spec, cell, verify_digest=digest_ready)
            blockers.extend(f"{cell.model.name}/{cell.precision}: {problem}" for problem in problems)
            checked_captures.add(capture_key)
        options = cell.backend.options
        try:
            _cell_timeout_seconds(cell)
        except ValueError as exc:
            blockers.append(f"{cell.backend.name}: invalid whole-cell timeout: {exc}")
        if cell.backend.adapter == "merlin_compile":
            package_value = str(options.get("package", ""))
            package = _resolve_path(package_value) if package_value else Path()
            if not package_value or not package.exists():
                blockers.append(f"{cell.backend.name}: compiler package is absent: {package_value}")
            elif (_is_sha256(options.get("package_sha256"))
                  and sha256_paths([package]) != options["package_sha256"]):
                blockers.append(f"{cell.backend.name}: compiler package digest mismatch")
        elif cell.backend.adapter == "executorch":
            command = options.get("command")
            if not isinstance(command, list) or not command:
                blockers.append(
                    f"{cell.backend.name}: no stateful ExecuTorch session command is frozen")
            elif not all(isinstance(part, str) and part for part in command):
                blockers.append(f"{cell.backend.name}: command must be a list of non-empty strings")
            else:
                if command[0] != "{python_executable}":
                    blockers.append(
                        f"{cell.backend.name}: command must start with the pinned "
                        "{python_executable} placeholder")
                if ("merlin.baselines.executorch_session" not in command or "run" not in command
                        or "build" in command or "--session-contract" in command):
                    blockers.append(
                        f"{cell.backend.name}: paper command must use the frozen-package "
                        "ExecuTorch run-only adapter")
                present = {field for field in _EXECUTORCH_COMMAND_FIELDS
                           if any(f"{{{field}}}" in part for part in command)}
                missing_fields = sorted(_EXECUTORCH_COMMAND_FIELDS - present)
                if missing_fields:
                    blockers.append(
                        f"{cell.backend.name}: stateful command omits placeholders {missing_fields}")
            try:
                _external_python(cell)
            except ValueError as exc:
                blockers.append(f"{cell.backend.name}: invalid pinned Python interpreter: {exc}")
            try:
                package_value, package_digest = _framework_package(cell)
            except ValueError as exc:
                blockers.append(f"{cell.backend.name}: {exc}")
            else:
                package_key = (cell.backend.name, cell.model.name, cell.precision)
                if package_key not in checked_framework_packages:
                    checked_framework_packages.add(package_key)
                    blockers.extend(
                        f"{cell.backend.name}: {problem}" for problem in
                        _audit_framework_package(cell, package_value, package_digest))
        else:
            blockers.append(f"{cell.backend.name}: unknown adapter {cell.backend.adapter!r}")

        source_values = options.get("source_paths", ()) or ()
        digest_key = ("kernel_source_sha256" if cell.backend.kind in {
                          "kernel_swap", "frozen_baseline"}
                      else "framework_source_sha256" if cell.backend.kind == "external_runtime"
                      else None)
        if digest_key is not None:
            if not isinstance(source_values, list) or not source_values:
                message = f"{cell.backend.name}: frozen source_paths are absent"
                if cell.backend.kind == "frozen_baseline":
                    warnings.append(message + "; causal attribution unavailable")
                else:
                    blockers.append(message)
            else:
                source_paths = [_resolve_path(str(value)) for value in source_values]
                missing_sources = [str(path) for path in source_paths if not path.exists()]
                if missing_sources:
                    blockers.append(
                        f"{cell.backend.name}: frozen source paths are absent: {missing_sources}")
                elif _is_sha256(options.get(digest_key)):
                    if sha256_paths(source_paths) != options[digest_key]:
                        blockers.append(f"{cell.backend.name}: {digest_key} mismatch")
    try:
        from merlin.mining import k1
        if not k1.available():
            blockers.append("K1 board/toolchain is unavailable")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"K1 availability check failed: {exc}")
    return Preflight(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(blockers)),
                     tuple(dict.fromkeys(warnings)))


def _quality_from_trajectory(metric: str, trajectory: dict, observations: int) -> tuple[bool, float | None]:
    if trajectory.get("scope") != "trajectory" or int(trajectory.get("steps") or 0) != observations:
        return False, None
    if metric.endswith("cosine"):
        value = trajectory.get("min_cosine")
    elif metric == "top1_agreement":
        value = trajectory.get("top1_agreement")
    else:
        value = None
    return value is not None, value


def _board_conditions_locked(value: object) -> bool:
    """Whether both untimed endpoint probes establish the frozen K1 frequency regime."""
    if not isinstance(value, dict):
        return False
    for endpoint in ("before", "after"):
        observed = value.get(endpoint)
        if not isinstance(observed, dict) or observed.get("governor") != "performance":
            return False
        try:
            current = int(observed.get("current_khz") or 0)
            maximum = int(observed.get("max_khz") or 0)
            thermal = int(observed.get("max_thermal_millic") or 0)
        except (TypeError, ValueError):
            return False
        if current <= 0 or current != maximum or thermal <= 0:
            return False
    return True


def _base_result(spec: PaperStudySpec, cell: MatrixCell, run_id: str, timestamp: str,
                 git_sha: str) -> dict:
    artifact = cell.model.artifacts[cell.precision]
    stages = list(cell.model.session.stages)
    timed_stages = list(cell.model.session.parameters.get("timed_stages", stages))
    excluded_stages = [stage for stage in stages if stage not in timed_stages]
    timing_scope = "end_to_end" if timed_stages == stages else "stage_subset"
    session_identities = spec.freeze.get("capture_session_identity_sha256", {}) or {}
    model_session_identities = session_identities.get(cell.model.name, {}) \
        if isinstance(session_identities, dict) else {}
    capture_session_identity = model_session_identities.get(cell.precision) \
        if isinstance(model_session_identities, dict) else None
    return {
        "schema_version": 2, "run_id": run_id, "timestamp": timestamp, "git_sha": git_sha,
        "study_label": spec.label, "target": spec.target, "model": cell.model.name,
        "checkpoint": cell.model.checkpoint, "artifact_sha256": artifact["sha256"],
        "fidelity": cell.model.fidelity, "backend": cell.backend.name,
        "runtime": cell.backend.runtime, "precision": cell.precision,
        "quantization": cell.backend.quantization, "core_count": cell.core_count,
        "session": cell.model.session.to_dict(),
        "lifecycle": {"built": False, "ran": False, "status": "not_run", "reason": None},
        "correctness": {"gate_ok": False}, "quality": {"gate_ok": False,
                                                               "metric": cell.model.quality["metric"],
                                                               "value": None},
        "timing": {"unit": "ns", "sample_unit": "complete_session", "scope": timing_scope,
                   "timed_stages": timed_stages, "excluded_stages": excluded_stages,
                   "samples": [], "stage_samples": {}, "median": None, "p95": None},
        "memory": {"policy": None, "peak_rss_bytes": None},
        "execution": {"mode": None, "requested_mode": None, "fallback_used": False},
        "provenance": {"study_sha256": spec.sha256(),
                       "compiler_policy_sha256": spec.freeze["policy_sha256"],
                       "compiler_source_sha256": spec.freeze.get("compiler_source_sha256"),
                       "runtime_sha256": spec.freeze["runtime_sha256"],
                       "capture_session_identity_sha256": capture_session_identity,
                       "vlen_bits": None, "vlen_source": None},
    }


def _merge_external_result(result: dict, external: object, cell: MatrixCell) -> None:
    """Merge only measured evidence; a foreign adapter cannot rewrite matrix identity/protocol."""
    if not isinstance(external, dict):
        raise ValueError("external adapter output must be a JSON object")
    unknown = sorted(set(external) - _EXTERNAL_MEASURED_SECTIONS - _EXTERNAL_IDENTITY)
    if unknown:
        raise ValueError(f"external adapter returned forbidden fields {unknown}")
    for name in _EXTERNAL_IDENTITY & set(external):
        if external[name] != result[name]:
            raise ValueError(
                f"external adapter identity mismatch for {name}: "
                f"expected={result[name]!r} got={external[name]!r}")
    missing = sorted(_EXTERNAL_MEASURED_SECTIONS - set(external))
    if missing:
        raise ValueError(f"external adapter omitted measured sections {missing}")
    for name in _EXTERNAL_MEASURED_SECTIONS - {"provenance"}:
        if not isinstance(external[name], dict):
            raise ValueError(f"external adapter section {name} must be a mapping")
        result[name] = dict(external[name])
    provenance = external["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("external adapter provenance must be a mapping")
    protected = {"study_sha256", "compiler_policy_sha256", "compiler_source_sha256",
                 "runtime_sha256", "capture_session_identity_sha256"}
    overlap = sorted(protected & set(provenance))
    if overlap:
        raise ValueError(f"external adapter cannot overwrite frozen provenance {overlap}")
    expected_framework = cell.backend.options.get("framework_source_sha256")
    if provenance.get("framework_source_sha256") != expected_framework:
        raise ValueError("external adapter framework source digest differs from the frozen backend")
    _, expected_package = _framework_package(cell)
    if provenance.get("framework_package_sha256") != expected_package:
        raise ValueError("external adapter package digest differs from the frozen backend")
    result["provenance"].update(provenance)


def execute_cell(spec: PaperStudySpec, cell: MatrixCell, run_id: str, timestamp: str,
                 git_sha: str, *, deadline_ns: int | None = None,
                 evidence_sink: dict | None = None, private_dir: Path | None = None) -> dict:
    """Diagnostic adapter result; live paper authority requires a standalone controller contract."""
    result = _base_result(spec, cell, run_id, timestamp, git_sha)
    timeout_seconds = _cell_timeout_seconds(cell)
    deadline_ns = (int(deadline_ns) if deadline_ns is not None
                   else time.monotonic_ns() + timeout_seconds * 1_000_000_000)
    evidence = evidence_sink if evidence_sink is not None else {}
    evidence.update(adapter_output=None, stdout="", stderr="",
                    adapter_command=["paper-cell-adapter", "--cell", cell.key])
    capture, session_contract, capture_errors = _capture_contract(spec, cell)
    if capture_errors:
        result["lifecycle"]["reason"] = "; ".join(capture_errors)
        evidence.update(adapter_output=None, stdout="", stderr="",
                        adapter_command=["capture-preflight", cell.key])
        validate_paper_result(result)
        return result
    validated_v2_session = int(session_contract.get("version", 0) or 0) == 2
    if not validated_v2_session:
        result["lifecycle"]["reason"] = (
            "paper measurement requires an executed, validated version-2 session contract")
        evidence.update(adapter_output=None, stdout="", stderr="",
                        adapter_command=["session-contract", "--required-version", "2"])
        validate_paper_result(result)
        return result
    backend_identity_ok = True
    try:
        if cell.backend.adapter == "merlin_compile":
            from merlin.compile_cli import compile_rvv
            options = cell.backend.options
            evidence["adapter_command"] = ["merlin.compile_rvv", "--cell", cell.key]
            stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buffer), \
                        contextlib.redirect_stderr(stderr_buffer):
                    compiled = compile_rvv(
                        cell.model.capture, "int8" if cell.precision == "w8a8" else "fp32",
                        run="k1", verify=True,
                        package=str(_resolve_path(str(options["package"]))),
                        auto_capture=False, timeout=_remaining_seconds(deadline_ns),
                        deadline_ns=deadline_ns,
                        harts=cell.core_count, iters=cell.model.session.observations,
                        warmup=cell.model.session.warmups,
                        session_repeats=cell.model.session.measurement_repeats,
                        kernel_backend=options.get("kernel_backend"), fallback_policy="forbid",
                        bundle_path=capture)
            finally:
                evidence.update(stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue())
            evidence.update(adapter_output=compiled, stdout=stdout_buffer.getvalue(),
                            stderr=stderr_buffer.getvalue(),
                            adapter_command=["merlin.compile_rvv", "--cell", cell.key])
            status = compiled.get("status")
            result["lifecycle"]["built"] = bool(compiled.get("binary"))
            result["lifecycle"]["ran"] = status in {"ran", "verified", "run_mismatch"}
            result["lifecycle"]["reason"] = compiled.get("reason") or compiled.get("error")
            verify = dict(compiled.get("verify", {}) or {})
            result["correctness"] = verify or {"gate_ok": False}
            trajectory = dict(compiled.get("trajectory_quality", {}) or {})
            quality_observed, quality_value = _quality_from_trajectory(
                str(cell.model.quality["metric"]), trajectory, cell.model.session.observations)
            quality_min = float(cell.model.quality.get("cosine_min", 0.0))
            quality_ok = quality_observed and quality_value is not None and quality_value >= quality_min
            result["quality"] = {"gate_ok": quality_ok,
                                 "metric": cell.model.quality["metric"], "value": quality_value,
                                 "scope": trajectory.get("scope"),
                                 "steps": trajectory.get("steps")}
            samples = list(compiled.get("iter_wall_ns", ()) or ())
            stage_samples = dict(compiled.get("stage_wall_ns", {}) or {})
            wall_stats = dict(compiled.get("sustained_wall_ns", {}) or {})
            timing_contract = result["timing"]
            result["timing"] = {"unit": "ns", "sample_unit": "complete_session",
                                "scope": timing_contract["scope"],
                                "timed_stages": timing_contract["timed_stages"],
                                "excluded_stages": timing_contract["excluded_stages"],
                                "samples": samples,
                                "stage_samples": stage_samples,
                                "median": wall_stats.get("median"), "p95": wall_stats.get("p95"),
                                "drift": wall_stats.get("drift")}
            result["memory"] = {"policy": compiled.get("memory_policy"),
                                "peak_rss_bytes": compiled.get("peak_rss_bytes")}
            raw_execution = dict(compiled.get("execution", {}) or {})
            expected_backend = options.get("kernel_backend")
            backend_identity_ok = (
                "kernel_backend" in raw_execution
                and raw_execution.get("kernel_backend") == expected_backend)
            if not backend_identity_ok and not result["lifecycle"].get("reason"):
                result["lifecycle"]["reason"] = (
                    "raw Merlin kernel-backend evidence differs from the frozen cell: "
                    f"expected={expected_backend!r} "
                    f"observed={raw_execution.get('kernel_backend')!r}")
            result["execution"] = raw_execution
            # ``compile_rvv`` retains this in its raw output as backend evidence.  The paper result
            # schema is intentionally closed and backend identity already lives at the cell level.
            result["execution"].pop("kernel_backend", None)
            result["provenance"].update({"vlen_bits": compiled.get("vlen"),
                                          "vlen_source": compiled.get("vlen_source"),
                                          "board_conditions": compiled.get("board_conditions"),
                                          "binary": compiled.get("binary"),
                                          "package_sha256": options.get("package_sha256")})
            if cell.backend.kind in {"kernel_swap", "frozen_baseline"}:
                result["provenance"]["kernel_source_sha256"] = options.get(
                    "kernel_source_sha256")
        elif cell.backend.adapter == "executorch":
            framework_package, framework_package_sha256 = _framework_package(cell)
            evidence["adapter_command"] = ["executorch-session", "--cell", cell.key]
            if private_dir is None:
                private_dir = Path(tempfile.mkdtemp(prefix="merlin_paper_external_"))
            pinned, staged, execute_fd = _stage_external_python(
                cell, Path(private_dir) / "python-runtime")
            command = [str(part).format(python_executable=str(pinned.argv0),
                                        model=cell.model.capture,
                                        variant=cell.model.artifacts[cell.precision]["variant"],
                                        cores=cell.core_count,
                                        framework_package=str(_resolve_path(framework_package)),
                                        framework_package_sha256=framework_package_sha256,
                                        warmups=cell.model.session.warmups,
                                        observations=cell.model.session.observations,
                                        measurement_repeats=cell.model.session.measurement_repeats,
                                        quality_metric=cell.model.quality["metric"],
                                        quality_min=cell.model.quality["cosine_min"],
                                        framework_source_sha256=(
                                            cell.backend.options["framework_source_sha256"]))
                       for part in cell.backend.options["command"]]
            if command[0] != str(pinned.argv0) or not Path(command[0]).is_absolute():
                raise ValueError("external adapter did not resolve to its pinned absolute Python")
            try:
                if _sha256_fd(execute_fd) != pinned.sha256:
                    raise ValueError("private Python fd changed immediately before execution")
                try:
                    proc = subprocess.run(
                        command, executable=f"/proc/self/fd/{execute_fd}",
                        pass_fds=(execute_fd,), capture_output=True, text=True,
                        timeout=_remaining_seconds(deadline_ns))
                except subprocess.TimeoutExpired as exc:
                    evidence.update(
                        stdout=(exc.stdout or ""), stderr=(exc.stderr or ""),
                        adapter_command=command, staged_python=str(staged),
                        staged_python_sha256=pinned.sha256)
                    raise
            finally:
                os.close(execute_fd)
            evidence.update(
                adapter_output=(json.loads(proc.stdout) if proc.returncode == 0 else None),
                stdout=proc.stdout, stderr=proc.stderr, adapter_command=command,
                staged_python=str(staged), staged_python_sha256=pinned.sha256)
            if proc.returncode != 0:
                result["lifecycle"]["reason"] = f"external adapter failed: {proc.stderr[-1000:]}"
            else:
                external = evidence["adapter_output"]
                _merge_external_result(result, external, cell)
        else:
            result["lifecycle"]["reason"] = f"unknown adapter {cell.backend.adapter!r}"
    except Exception as exc:  # noqa: BLE001 — one failed cell must not erase the full matrix
        result["lifecycle"].update(status="error", reason=f"{type(exc).__name__}: {exc}")

    execution = result["execution"]
    routed = execution.get("n_routed")
    eligible = execution.get("n_eligible")
    candidates = execution.get("n_candidates")
    treatment_applied = (
        cell.backend.kind != "kernel_swap"
        or (isinstance(routed, int) and not isinstance(routed, bool)
            and isinstance(eligible, int) and not isinstance(eligible, bool)
            and isinstance(candidates, int) and not isinstance(candidates, bool)
            and eligible > 0 and routed == eligible and candidates >= eligible))
    if (cell.backend.kind == "kernel_swap" and not treatment_applied
            and not result["lifecycle"].get("reason")):
        result["lifecycle"]["reason"] = (
            f"{cell.backend.name} did not route the complete nonempty eligible GEMM set "
            f"(routed={routed!r}, eligible={eligible!r}, candidates={candidates!r}); "
            "this is not a complete kernel-swap measurement")
    affinity_ok = (
        int(execution.get("core_count") or 0) == cell.core_count
        and int(execution.get("requested_core_count") or 0) == cell.core_count
        and execution.get("affinity_source") == "sched_getaffinity")
    if result["lifecycle"].get("ran") and not affinity_ok and not result["lifecycle"].get("reason"):
        result["lifecycle"]["reason"] = (
            "on-device CPU affinity evidence is absent or differs from the requested core count")
    expected_worker_source = ("extension_threadpool_no_pool_guard" if cell.core_count == 1
                              else "extension_threadpool")
    worker_threads_ok = (
        cell.backend.kind != "external_runtime"
        or (int(execution.get("worker_threads") or 0) == cell.core_count
            and execution.get("worker_thread_source") == expected_worker_source))
    if (result["lifecycle"].get("ran") and not worker_threads_ok
            and not result["lifecycle"].get("reason")):
        result["lifecycle"]["reason"] = (
            "ExecuTorch worker-thread configuration differs from the requested core count")
    board_conditions_ok = _board_conditions_locked(
        result["provenance"].get("board_conditions"))
    if (result["lifecycle"].get("ran") and not board_conditions_ok
            and not result["lifecycle"].get("reason")):
        result["lifecycle"]["reason"] = (
            "K1 board-condition endpoints do not establish the performance governor, "
            "current=max frequency lock, and a positive thermal observation")
    semantic_session_ok = (
        validated_v2_session
        and execution.get("semantic_session") is True
        and execution.get("same_input_repetition") is False)
    if (result["lifecycle"].get("ran") and not semantic_session_ok
            and not result["lifecycle"].get("reason")):
        result["lifecycle"]["reason"] = (
            "execution evidence does not prove a version-2 semantic session with distinct "
            "per-observation inputs")
    passed = (result["lifecycle"]["ran"] and result["correctness"].get("gate_ok") is True
              and result["quality"].get("gate_ok") is True and result["timing"].get("samples")
              and execution.get("fallback_used") is False
              and treatment_applied
              and execution.get("mode") == execution.get("requested_mode")
              and backend_identity_ok
              and affinity_ok
              and worker_threads_ok
              and semantic_session_ok
              and board_conditions_ok
              and result["provenance"].get("vlen_source") == "csr")
    if result["lifecycle"]["status"] != "error":
        result["lifecycle"]["status"] = "pass" if passed else (
            "fail" if result["lifecycle"]["ran"] else "not_run")
    validate_paper_result(result)
    return result


def _execute_live_cell(spec: PaperStudySpec, cell: MatrixCell, *, index: int,
                       out_dir: Path, parent_handle) -> dict:
    """Build and execute one exact standalone session through the trusted controller."""
    from .paper_measurement_controller import normalize_receipt, produce_receipt

    run_id = f"{parent_handle.run_id}__cell{index:03d}"
    child_dir = (out_dir / "cell-runs" /
                 f"{index:03d}_{cell.model.name}_{cell.backend.name}_{cell.precision}_"
                 f"{cell.core_count}c")
    staging_dir = (out_dir / "controller-contracts" /
                   f"{index:03d}_{cell.model.name}_{cell.backend.name}_{cell.precision}_"
                   f"{cell.core_count}c")
    from .paper_contract_registry import build_registered_contract

    expected = _base_result(
        spec, cell, run_id, parent_handle.timestamp, parent_handle.git_sha)
    contract_path = build_registered_contract(
        spec, cell, run_id=run_id, timestamp=parent_handle.timestamp,
        git_sha=parent_handle.git_sha, staging_dir=staging_dir, base_result=expected)
    if not contract_path.is_file():
        raise ValueError("registered backend did not materialize a measurement contract")
    receipt_path = produce_receipt(contract_path, child_dir)
    result = normalize_receipt(receipt_path)
    identity_fields = {
        "schema_version", "run_id", "timestamp", "git_sha", "study_label", "target",
        "model", "checkpoint", "artifact_sha256", "fidelity", "backend", "runtime",
        "precision", "quantization", "core_count", "session",
    }
    if any(result.get(field) != expected[field] for field in identity_fields):
        raise ValueError("controller-normalized result identity differs from the frozen study cell")
    frozen_provenance = {key: value for key, value in expected["provenance"].items()
                         if key not in {"vlen_bits", "vlen_source", "board_conditions"}}
    if any(result["provenance"].get(key) != value
           for key, value in frozen_provenance.items()):
        raise ValueError("controller-normalized provenance differs from the frozen study")
    return result


def _write_plan(out_dir: Path, spec: PaperStudySpec, preflight: Preflight) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(out_dir / "study.yaml", spec.canonical_dict())
    write_yaml(out_dir / "preflight.yaml", preflight.to_dict())
    ordered_cells = execution_matrix(spec)
    matrix = [{"execution_index": index, "key": c.key, "model": c.model.name,
               "backend": c.backend.name,
               "precision": c.precision, "core_count": c.core_count,
               "session": c.model.session.kind} for index, c in enumerate(ordered_cells)]
    write_yaml(out_dir / "matrix.yaml", {"n_cells": len(matrix), "cells": matrix})
    lines = [f"# {spec.label}", "", f"Study SHA-256: `{spec.sha256()}`", "",
             f"Preflight: **{'READY' if preflight.ready else 'BLOCKED'}**", "",
             f"Planned cells: {len(matrix)}", "",
             "Execution order: deterministic block-randomized (frozen seed); backends for each "
             "model/precision/core block remain contiguous.", ""]
    if preflight.blockers:
        lines += ["## Blockers", "", *[f"- {v}" for v in preflight.blockers], ""]
    if preflight.warnings:
        lines += ["## Warnings", "", *[f"- {v}" for v in preflight.warnings], ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(spec: PaperStudySpec, *, live: bool = False, environment_audit: bool = False,
        out_dir: Path | None = None) -> Path:
    """Plan a study or execute only contracts selected by the closed backend registry."""
    handle = None
    product = None
    explicit_out = Path(out_dir).resolve() if out_dir is not None else None
    if live:
        handle = start_run(suite="paper-study", method="frozen-compiler", target=spec.target,
                           extra={"study_sha256": spec.sha256(), "n_cells": len(spec.matrix()),
                                  "results_dir": (str(explicit_out) if explicit_out else
                                                  "canonical_aet_run")})
        if out_dir is None:
            out_dir = handle.run_dir
    elif out_dir is None:
        product = new_product("paper-study", version=2, target=spec.target,
                              sources=[str(spec.source_path)] if spec.source_path else [])
        out_dir = product.path
    out_dir = Path(out_dir)
    parent_status = "fail"
    parent_summary = {"preflight_ready": False, "n_cells": 0, "n_pass": 0}
    try:
        if handle is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_yaml(out_dir / "aet-parent.yaml", {
                "run_id": handle.run_id, "canonical_run_dir": str(handle.run_dir.resolve()),
                "run_record": str((handle.run_dir / "run_record.json").resolve()),
                "results_dir": str(out_dir.resolve()),
            })
        preflight = environment_preflight(spec) if (live or environment_audit) else spec.preflight()
        _write_plan(out_dir, spec, preflight)
        if not live:
            if product:
                for rel in ("study.yaml", "preflight.yaml", "matrix.yaml", "report.md"):
                    product.add_artifact(rel)
                product.write_manifest()
            return out_dir
        if not preflight.ready:
            raise StudyNotReady("paper study preflight is blocked", out_dir)
        if handle is None:  # pragma: no cover - guarded by the live branch above
            raise RuntimeError("live paper study did not acquire an AET run handle")

        results = []
        for index, cell in enumerate(execution_matrix(spec)):
            result = _execute_live_cell(
                spec, cell, index=index, out_dir=out_dir, parent_handle=handle)
            results.append(result)
            cell_path = (out_dir / "measurements" / cell.model.name / cell.backend.name /
                         f"{cell.precision}_{cell.core_count}c.yaml")
            write_yaml(cell_path, result)
        # Explanations are produced only after both sides of a comparator pair are present.  The
        # attribution module reads pre-freeze evidence and never timing samples to construct text.
        from .paper_attribution import attach_causal_attribution
        attach_causal_attribution(spec, results)
        for result in results:
            validate_paper_result(result)
            cell_path = (out_dir / "measurements" / result["model"] / result["backend"] /
                         f"{result['precision']}_{result['core_count']}c.yaml")
            write_yaml(cell_path, result)
        results_document = seal_results_document(spec, results)
        write_yaml(out_dir / "results.yaml", results_document)
        paper_report = build_paper_report(spec, results_document)
        write_yaml(out_dir / "paper-results.yaml", paper_report)
        (out_dir / "paper-results.md").write_text(
            render_markdown(paper_report), encoding="utf-8")
        n_pass = sum(r["lifecycle"]["status"] == "pass" for r in results)
        parent_status = "ok" if n_pass == len(results) else "fail"
        parent_summary = {"preflight_ready": True, "n_cells": len(results), "n_pass": n_pass}
        return out_dir
    finally:
        if handle is not None:
            finish_run(handle, parent_status, parent_summary)
