"""One bounded, independently replayable CPU Model2MLIR capture.

This is a *process and byte-closure* proof for a selected CPU workload, not a
general Python purity theorem and not Phase 0 admission.  The guest has an
empty filesystem root apart from copied runtime/source, private devices and
temporary/output mounts; it has no network or host checkout mount.  Reviewers
must separately qualify the workload, framework numerics and Phase 0 policy.
"""

from __future__ import annotations

import ast
import json as json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import record_formats as strict_json
from merlin.experiments.capture_execution import (
    readonly_runtime,
)
from merlin.experiments.capture_execution import (
    sealed_python as sealed_failure,
)
from merlin.experiments.capture_execution import (
    sealed_static as native_integer,
)
from merlin.experiments.capture_execution.m2m_origin import (
    M2MOriginError,
    git_origin,
    verify_frozen_receipt,
    verify_frozen_selector,
)
from merlin.experiments.capture_execution.precision_staging import STAGING_API_REQUIREMENTS as STAGING_API_REQUIREMENTS
from merlin.experiments.capture_execution.precision_staging import output_staging_error as output_staging_error
from merlin.experiments.capture_execution.python_preflight import _loader_env_reads
from merlin.experiments.capture_execution.sealed_python import (
    _FLAGS,
    _TIMEOUT_SECONDS,
    _snapshot_rows,
    _verify_replayed_output,
    _verify_staged_selection,
)
from merlin.experiments.capture_execution.sealed_static import (
    _bwrap_binary,
    _canonical_path,
    _digest,
    _file_digest,
    _json,
    _tree,
)

# ---- sealed_m2m --------------------------------------------------------------------------------

SCHEMA_V1 = "merlin.sealed_m2m_fp32.v1"


SCHEMA = "merlin.sealed_m2m_cpu.v2"


SCHEMA_V3 = "merlin.sealed_m2m_cpu.v3"


_V1_ISSUER_SHA256 = "f8ca017999a5cb40d44ed29bc9412bef842fe8f3edd8d85170879d1df15d1dd6"


_HISTORICAL_V2_ISSUER_SHA256 = "36aa1528481a9630e2e31394f45fdde62a0bfea738b8e1855085029f9574344b"


_PRE_V3_ISSUER_SHA256 = "596e8828727b3835f384a264fe9fb4a3ee1c975e869ed2130d7d20022ed06a2a"


_STRICT_JSON_SELECTED_ISSUER_SHA256 = "73f2463303b3a7274c666e44c99845dbe44953663aff3bfd346633fc7d391fea"


_STRICT_JSON_PREDECESSOR_V2_ISSUER_SHA256 = "f1f36bc57807fbc4e93360757e9b0f891f38a70326ffc10c8067bfb0b5b11920"


_PRE_FROZEN_ORIGIN_ISSUER_SHA256 = "16aa775b37791164c1546b987245f8e359231acd3bd3ad2e4b8662d1834465d5"


_ARCHIVED_V3_ISSUER_SHA256 = "ff12b040ac7f98fb54f704e00f3c9764b1fc923072d2c9b1f35a00d5d4f6161f"


_PRE_CACHE_NORMALIZATION_V3_ISSUER_SHA256 = "03e1d28ebd4cc356928355490a13e329e5b73d134cdffa827f4aa60b29ab8b78"


_PRE_INVENTORY_REUSE_ISSUER_SHA256 = "b78a8d7f13785c9d0263e4149e38a4de0d9bfd43052e552213003d143fac7ba3"


_ARCHIVED_SELECTED_V2_ISSUER_SHA256 = "83af01552173c2ddd24d78996838bfb9d72b79abecc61b9a0e79690bf12ef4d5"


_PRE_CONCURRENT_REPLAY_ISSUER_SHA256 = "6c60fb10e4a2b2e05ff67a8b215f86317242346a7cd72b3865a603fc1d2813c4"


_PRE_EXTERNAL_REPLAY_SCRATCH_ISSUER_SHA256 = "fe672aaf62efcf1f2c5cec499af646d59ca0f458107e459a2fbdc59271c4586b"


_PRE_SELECTED_THREADS_ISSUER_SHA256 = "ba57b6e1bd00734b95b7cb4a64c30f636f715cc313aa048986b3812da04b5fb8"


_PRE_NATIVE_INTEGER_ISSUER_SHA256 = "7eb45b2169dac5b2984ab444c183155855b1ecd9de81c6c25890d12e0ce39889"


_PRE_FAILURE_DIAGNOSTICS_ISSUER_SHA256 = "7eb45b2169dac5b2984ab444c183155855b1ecd9de81c6c25890d12e0ce39889"


_PRE_REPLAY_DIAGNOSTICS_ISSUER_SHA256 = "4356dafad28f9b81ba77988074c44214d1b52c304cdb29bae63108b4e3276343"


_PRE_PUBLICATION_DIAGNOSTICS_ISSUER_SHA256 = sealed_failure.PRE_RETENTION_ISSUER_SHA256


_PRE_DYNAMIC_REFERENCE_ISSUER_SHA256 = "b1a7e83ea8daabb5f7e708118c674116c2d817d7ff912f34c99e031c11bd141f"


_V1_SCOPE = "isolated selected FP32 CPU M2M capture; no Phase 0 admission"


_V2_SCOPE = "isolated selected CPU M2M capture; no Phase 0 admission"


_V3_SCOPE = "isolated selected CPU M2M capture with declared model inputs; no Phase 0 admission"


_MAX_SNAPSHOT_BYTES = 16_000_000_000


_MAX_FULL_CAPTURE_SECONDS = 43_200


_MAX_SELECTED_CAPTURE_SECONDS = 14_400


_VALIDATION_SOURCE_SHA256 = "0a6f340c90d8cff92e15282441603034e5d2430a5c6033555fade5f6b3673628"


def _validation_functions():
    import hashlib
    from importlib import import_module

    source = Path(__file__).with_name("sealed_m2m_validation.py")
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != _VALIDATION_SOURCE_SHA256:
        raise SealedM2MError("selected materialization verifier source changed")
    implementation = import_module(".sealed_m2m_validation", __package__)
    if Path(implementation.__file__).resolve() != source.resolve():
        raise SealedM2MError("selected materialization verifier import differs")
    return implementation


def _selected_timeout_valid(schema: str, value: Any) -> bool:
    """Whether a plan's selected execution timeout is in its schema's bounds."""
    if schema == SCHEMA_V3:
        return type(value) is int and _TIMEOUT_SECONDS <= value <= _MAX_FULL_CAPTURE_SECONDS
    return value is None or (type(value) is int and _TIMEOUT_SECONDS <= value <= _MAX_SELECTED_CAPTURE_SECONDS)


_LAUNCH_PREFIX = (
    "import runpy,sys;"
    "sys.path[:0]=['/source/m2m-src','/opt/capture-venv/lib/python3.12/site-packages'];"
    "sys.argv=['/source/worker.py','--m2m-dir','/source/m2m-src','--loader',"
    "'/source/workload/loader.py','--dtype','fp32','--seed','0',"
    "'--materialize-bundle','--out',"
)


_LAUNCH_SUFFIX = (
    "];import structlog;"
    "structlog.configure(processors=[structlog.processors.KeyValueRenderer(sort_keys=True)]);"
    "runpy.run_path('/source/worker.py',run_name='__main__')"
)


def _command(output_mount: Path) -> tuple[str, ...]:
    return (
        "/opt/capture-venv/bin/python",
        "-I",
        "-S",
        "-B",
        "-c",
        _LAUNCH_PREFIX + repr(str(output_mount)) + _LAUNCH_SUFFIX,
    )


_FLOAT_DTYPES = frozenset({"fp32", "f32"})


def _worker_options(options: Any) -> dict[str, Any]:
    """Validate the worker options a plan may select; nothing outside this vocabulary is accepted."""
    if options is None:
        return {}
    if not isinstance(options, dict) or set(options) - {
        "agreement_tolerance",
        "stage_fp32",
        "cpu_threads",
        "source_mode",
    }:
        raise SealedM2MError("sealed capture worker options are unsupported")
    if "source_mode" in options and options["source_mode"] != native_integer.MODE:
        raise SealedM2MError("sealed capture source_mode must select native_integer_v1")
    if "stage_fp32" in options and type(options["stage_fp32"]) is not bool:
        raise SealedM2MError("sealed capture stage_fp32 must be a boolean")
    if "cpu_threads" in options and (type(options["cpu_threads"]) is not int or not 1 <= options["cpu_threads"] <= 256):
        raise SealedM2MError("sealed capture cpu_threads must be an integer from 1 to 256")
    tolerance = options.get("agreement_tolerance")
    if tolerance is not None and (
        not isinstance(tolerance, list)
        or len(tolerance) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in tolerance)
        or any(value != value or value in (float("inf"),) for value in tolerance)
    ):
        raise SealedM2MError("sealed capture agreement tolerance must be two finite nonnegative numbers")
    return {key: value for key, value in options.items() if value is not None}


_WORKER_LAYOUTS = (
    "targetgen/_m2m_capture_worker.py",
    "capture/preparation/worker.py",
)
_WORKER_GUEST_MEMBERS = tuple("merlin-src/merlin/" + member for member in _WORKER_LAYOUTS)


def _worker_member(plan: dict[str, Any]) -> str:
    """Select a closed guest layout from the original recorded source paths.

    The worker and complete package tree are already pinned by the plan. Old
    plans retain their original command and filename; current packages execute
    the worker at its actual package depth without a synthetic source alias.
    """
    root, worker = plan.get("merlin_root"), plan.get("worker")
    if isinstance(root, str) and isinstance(worker, str):
        for layout, member in zip(_WORKER_LAYOUTS, _WORKER_GUEST_MEMBERS, strict=True):
            if worker == str(Path(root) / layout):
                return member
    raise SealedM2MError("selected worker must belong to a supported Merlin source package layout")


def _command_v2(
    output_mount: Path,
    *,
    dtype: str,
    recipe: bool,
    options: dict[str, Any] | None = None,
    worker_member: str = _WORKER_GUEST_MEMBERS[0],
) -> tuple[str, ...]:
    options = _worker_options(options)
    if native_integer.selected(options):
        try:
            native_integer.require_selection(options, dtype=dtype, recipe=recipe)
        except ValueError as error:
            raise SealedM2MError(str(error)) from error
    elif not ((dtype in _FLOAT_DTYPES and not recipe) or (dtype == "int8" and recipe)):
        raise SealedM2MError("CPU capture requires fp32 without a recipe or int8 with a selected recipe")
    if options.get("stage_fp32") and dtype not in _FLOAT_DTYPES and (dtype, recipe) != ("int8", True):
        raise SealedM2MError("FP32 staging requires a float capture or selected int8 recipe")
    if worker_member not in _WORKER_GUEST_MEMBERS:
        raise SealedM2MError("unsupported sealed worker guest layout")
    worker = "/source/" + worker_member
    argv = [
        worker,
        "--m2m-dir",
        "/source/m2m-src",
        "--loader",
        "/source/workload/loader.py",
        "--dtype",
        dtype,
        "--seed",
        "0",
        "--materialize-bundle",
        "--out",
        str(output_mount),
    ]
    if recipe:
        argv += ["--recipe", "/source/inputs/quant_recipe.json"]
    if native_integer.selected(options):
        argv += ["--already-quantized", "--native-integer-source"]
    if options.get("agreement_tolerance") is not None:
        atol, rtol = options["agreement_tolerance"]
        argv += ["--agreement-atol", repr(float(atol)), "--agreement-rtol", repr(float(rtol))]
    if options.get("stage_fp32"):
        argv.append("--stage-fp32")
    if options.get("cpu_threads") is not None:
        # Reference bytes depend on the intra-op thread count, so the selected count is part of
        # the command, its policy digest and the replay; absent, the worker runs one thread.
        argv += ["--cpu-threads", str(options["cpu_threads"])]
    program = (
        "import runpy,sys;"
        "sys.path[:0]=['/source/m2m-src','/source/merlin-src',"
        "'/opt/capture-venv/lib/python3.12/site-packages'];"
        "sys.argv=" + repr(argv) + ";import structlog;"
        "structlog.configure(processors=[structlog.processors.KeyValueRenderer(sort_keys=True)]);"
        "runpy.run_path(" + repr(worker) + ",run_name='__main__')"
    )
    return ("/opt/capture-venv/bin/python", "-I", "-S", "-B", "-c", program)


def _recipe_selection(path: Path | None, *, dtype: str) -> dict[str, Any] | None:
    if dtype in _FLOAT_DTYPES:
        if path is not None:
            raise SealedM2MError("fp32 capture must not select a quantization recipe")
        return None
    if dtype != "int8" or path is None:
        raise SealedM2MError("CPU capture supports only fp32 or int8 with an explicit recipe")
    path = _canonical_path(path, exists=True)
    if not path.is_file() or path.is_symlink():
        raise SealedM2MError("selected quantization recipe is absent or indirect")
    try:
        recipe = strict_json.loads(path.read_bytes())
        from merlin.compiler.quantization.recipe import digest as recipe_digest

        valid = (
            isinstance(recipe, dict)
            and recipe.get("schema") == "quant_recipe_v1"
            and recipe.get("status") == "derived"
            and recipe.get("recipe_sha256") == recipe_digest(recipe)
            and recipe.get("software_numerical_engine") == "integer_reference"
            and all((recipe.get(part) or {}).get("dtype") == "int8" for part in ("activation", "weight"))
            and (recipe.get("activation") or {}).get("mode") == "static"
        )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise SealedM2MError("selected int8 recipe is unreadable or malformed") from exc
    if not valid:
        raise SealedM2MError("selected int8 recipe lacks a derived static W8A8 integer-reference contract")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_digest(path),
        "recipe_sha256": recipe["recipe_sha256"],
    }


_REPLAYABLE_LOG_ENV = (
    ("TORCH_CPP_LOG_LEVEL", "ERROR"),
    ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
    ("TQDM_DISABLE", "1"),
)


def _guest_env(
    output_mount: Path, *, replayable_logs: bool, loader_env: dict[str, str | None] | None = None
) -> dict[str, str]:
    env = {"USER": "capture", "LOGNAME": "capture", "XDG_CACHE_HOME": str(output_mount / "cache")}
    if replayable_logs:
        env.update(_REPLAYABLE_LOG_ENV)
    for name, value in (loader_env or {}).items():
        if value is not None:
            env[name] = value
    return env


def _policy(
    command: tuple[str, ...],
    output_mount: Path,
    *,
    replayable_logs: bool = False,
    loader_env: dict[str, str | None] | None = None,
    timeout_seconds: int = _TIMEOUT_SECONDS,
    runtime_reuse: dict | None = None,
) -> str:
    return _digest(
        _json(
            {
                "flags": _FLAGS,
                "guest_env": _guest_env(output_mount, replayable_logs=replayable_logs, loader_env=loader_env),
                "command": command,
                "output_mount": str(output_mount),
                "mounts": ["guest-root:ro", "source:ro", "capture:rw", "tmp:tmpfs", "dev:private"],
                "timeout_seconds": timeout_seconds,
                **({"runtime_reuse": runtime_reuse} if runtime_reuse is not None else {}),
            }
        )
    )


def _sandbox_argv(
    bwrap: Path,
    runtime: Path,
    source: Path,
    output: Path,
    command: tuple[str, ...],
    output_mount: Path,
    *,
    replayable_logs: bool = False,
    loader_env: dict[str, str | None] | None = None,
    runtime_reuse: dict | None = None,
) -> list[str]:
    setenv = [
        item
        for name, value in _guest_env(output_mount, replayable_logs=replayable_logs, loader_env=loader_env).items()
        for item in ("--setenv", name, value)
    ]
    return [
        str(bwrap),
        *_FLAGS,
        *setenv,
        "--ro-bind",
        str(runtime),
        "/",
        *(readonly_runtime.mounts(runtime_reuse, runtime) if runtime_reuse is not None else []),
        "--ro-bind",
        str(source),
        "/source",
        "--bind",
        str(output),
        str(output_mount),
        "--tmpfs",
        "/tmp",
        "--dev",
        "/dev",
        "--",
        *command,
    ]


def _process_record(
    returncode: int, stdout: bytes, stderr: bytes, *, failure_dir: Path | None = None
) -> dict[str, Any]:
    if failure_dir is not None:
        sealed_failure.process(failure_dir, returncode, stdout, stderr)
    if returncode:
        diagnostic = ""
        if failure_dir is not None:
            diagnostic = f"; full process diagnostics: {failure_dir}"
        detail = stderr.decode("utf-8", errors="replace")[-4000:]
        tail = stdout.decode("utf-8", errors="replace")[-4000:]
        raise SealedM2MError(f"sandboxed M2M exited {returncode}{diagnostic}: stderr={detail}; stdout={tail}")
    return {
        "returncode": 0,
        "stdout": {"bytes": len(stdout), "sha256": _digest(stdout)},
        "stderr": {"bytes": len(stderr), "sha256": _digest(stderr)},
    }


def _execute(
    bwrap: Path,
    runtime: Path,
    source: Path,
    output: Path,
    command: tuple[str, ...],
    output_mount: Path,
    *,
    replayable_logs: bool = False,
    loader_env: dict[str, str | None] | None = None,
    timeout_seconds: int = _TIMEOUT_SECONDS,
    runtime_reuse: dict | None = None,
) -> dict[str, Any]:
    argv = _sandbox_argv(
        bwrap,
        runtime,
        source,
        output,
        command,
        output_mount,
        replayable_logs=replayable_logs,
        loader_env=loader_env,
        **({"runtime_reuse": runtime_reuse} if runtime_reuse is not None else {}),
    )
    try:
        result = subprocess.run(
            argv, env={}, cwd="/", stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout_seconds
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealedM2MError(f"sandbox execution failed: {type(exc).__name__}") from exc
    return _process_record(
        result.returncode, result.stdout, result.stderr, failure_dir=output.parent / (output.name + "-failure")
    )


def _execute_beside(
    bwrap: Path,
    runtime: Path,
    source: Path,
    outputs: tuple[Path, Path],
    command: tuple[str, ...],
    output_mount: Path,
    *,
    cpus: tuple[frozenset[int], frozenset[int]],
    replayable_logs: bool = False,
    loader_env: dict[str, str | None] | None = None,
    timeout_seconds: int = _TIMEOUT_SECONDS,
    runtime_reuse: dict | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the identical sandboxed command twice at once, each into its own fresh output.

    Each execution is its own sandbox: its own namespaces, the same read-only source and guest
    root, its own ``/tmp`` and an empty output directory bound at the same guest path, so neither
    can read the other's output. Each is pinned to its own CPU set (a thread's affinity, which its
    sandbox inherits). If either fails, the other is stopped and the issue fails.
    """
    import threading

    argvs = [
        _sandbox_argv(
            bwrap,
            runtime,
            source,
            out,
            command,
            output_mount,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            **({"runtime_reuse": runtime_reuse} if runtime_reuse is not None else {}),
        )
        for out in outputs
    ]
    processes: list[subprocess.Popen | None] = [None, None]
    results: list[Any] = [None, None]
    failures: list[SealedM2MError] = []
    failed = threading.Event()
    lock = threading.Lock()

    def run(index: int) -> None:
        try:
            os.sched_setaffinity(0, cpus[index])
            with lock:
                if failed.is_set():
                    return
                processes[index] = subprocess.Popen(
                    argvs[index],
                    env={},
                    cwd="/",
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            stdout, stderr = processes[index].communicate(timeout=timeout_seconds)
            results[index] = _process_record(
                processes[index].returncode,
                stdout,
                stderr,
                failure_dir=outputs[index].parent / (outputs[index].name + "-failure"),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            results[index] = SealedM2MError(f"sandbox execution failed: {type(exc).__name__}")
        except SealedM2MError as exc:
            results[index] = exc
        if isinstance(results[index], Exception):
            failed.set()
            with lock:
                if not failures:  # the cause, not a sibling this failure stopped
                    failures.append(results[index])
                for process in processes:
                    if process is not None and process.poll() is None:
                        process.kill()

    threads = [threading.Thread(target=run, args=(index,), daemon=True) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for process in processes:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    if failures:
        raise failures[0]
    if any(result is None for result in results):
        raise SealedM2MError("sandbox execution failed: an execution did not start")
    return results[0], results[1]


def replay_cpu_sets(cpus: Any) -> tuple[frozenset[int], frozenset[int]]:
    """Split ``cpus`` into two equal disjoint halves: lower for issuance, upper for its replay."""
    ordered = sorted({int(cpu) for cpu in cpus})
    if len(ordered) < 2:
        raise SealedM2MError("a concurrent replay needs at least two CPUs")
    half = len(ordered) // 2
    return frozenset(ordered[:half]), frozenset(ordered[half : 2 * half])


def _probe_sandbox(bwrap: Path) -> None:
    """Fail before copying a large runtime if this host cannot create the required namespaces.

    The probe runs only the host's ``true`` under the same namespace flags. It is
    readiness evidence, not an attested capture and not a weaker execution policy.
    The real capture still runs with its selected, private guest-root mounts.
    """
    argv = [
        str(bwrap),
        *_FLAGS,
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        "/tmp",
        "--dev",
        "/dev",
        "--",
        "/bin/true",
    ]
    try:
        result = subprocess.run(argv, env={}, cwd="/", stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealedM2MError(f"sealed M2M sandbox unavailable before snapshot: {type(exc).__name__}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise SealedM2MError(f"sealed M2M sandbox unavailable before snapshot: {detail}")


class SealedM2MError(ValueError):
    """The proposed capture or replay does not satisfy this narrow policy."""


def _capture_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._capture_api_missing(m2m_root)


def _frontend_trace_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._frontend_trace_api_missing(m2m_root)


def _fp32_stage_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._fp32_stage_api_missing(m2m_root)


def _static_integer_reference_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._static_integer_reference_api_missing(m2m_root)


def _source_api_missing(m2m_root: Path, required: dict[str, dict[str, set[str]]]) -> tuple[str, ...]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._source_api_missing(m2m_root, required)
