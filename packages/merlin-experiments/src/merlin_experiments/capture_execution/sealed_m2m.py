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

from merlin.common import strict_json

from . import native_integer, readonly_runtime, sealed_failure
from .m2m_origin import M2MOriginError, git_origin, verify_frozen_receipt, verify_frozen_selector
from .precision_staging import STAGING_API_REQUIREMENTS as STAGING_API_REQUIREMENTS
from .precision_staging import output_staging_error as output_staging_error
from .python_preflight import _loader_env_reads
from .sealed_inventory import _snapshot_rows, _verify_replayed_output, _verify_staged_selection
from .sealed_python import _FLAGS, _TIMEOUT_SECONDS
from .sealed_static import (
    SealedCaptureError,
    _BASE_ROOT_XATTR,
    _base_link_root,
    _base_symlink,
    _bwrap_binary,
    _canonical_path,
    _dangling_symlink,
    _digest,
    _directory_symlink,
    _file_digest,
    _json,
    _omitted_members,
    _record_omitted,
    _unreadable_file,
)
from .sealed_static import _tree as _strict_tree

SCHEMA_V1 = "merlin.sealed_m2m_fp32.v1"
SCHEMA = "merlin.sealed_m2m_cpu.v2"
SCHEMA_V3 = "merlin.sealed_m2m_cpu.v3"
# The only historical v1 issuer whose policy this verifier knows. New captures
# use v2; an old unsigned receipt remains a diagnostic replay claim only.
_V1_ISSUER_SHA256 = "f8ca017999a5cb40d44ed29bc9412bef842fe8f3edd8d85170879d1df15d1dd6"
_HISTORICAL_V2_ISSUER_SHA256 = "36aa1528481a9630e2e31394f45fdde62a0bfea738b8e1855085029f9574344b"
_PRE_V3_ISSUER_SHA256 = "596e8828727b3835f384a264fe9fb4a3ee1c975e869ed2130d7d20022ed06a2a"
# Exact selected v2/v3 issuer that used strict JSON reads before the separate
# historical-issuer addition below. Only replay accepts its existing receipts.
_STRICT_JSON_SELECTED_ISSUER_SHA256 = "73f2463303b3a7274c666e44c99845dbe44953663aff3bfd346633fc7d391fea"
# Byte-exact selected v2 issuer before strict JSON reads replaced json.loads.
# Its already-issued receipts still undergo this verifier's strict parsing,
# selected-policy, source/runtime/output identity, and fresh replay checks.
_STRICT_JSON_PREDECESSOR_V2_ISSUER_SHA256 = "f1f36bc57807fbc4e93360757e9b0f891f38a70326ffc10c8067bfb0b5b11920"
# Exact pre-frozen-origin issuer from 0e66019b. Its already-issued v2/v3
# snapshots may be replayed, but no new selection/issue may use those bytes.
_PRE_FROZEN_ORIGIN_ISSUER_SHA256 = "16aa775b37791164c1546b987245f8e359231acd3bd3ad2e4b8662d1834465d5"
# Exact v3 producer archived by commit 2cdc08c27ea9dec74f25a20a530c7179230c505b.
# Accept its already-issued pending receipts for fresh byte-for-byte replay; the
# selected receipt must still bind this same digest and every source/output byte.
_ARCHIVED_V3_ISSUER_SHA256 = "ff12b040ac7f98fb54f704e00f3c9764b1fc923072d2c9b1f35a00d5d4f6161f"
# Exact issuer of an already selected, executed and independently replayed
# full v3 capture before transient Merlin bytecode caches were excluded.
# This is replay-only: issue() still requires the current plan and issuer, and
# the selection, staged source, runtime, inputs and output must still match.
_PRE_CACHE_NORMALIZATION_V3_ISSUER_SHA256 = "03e1d28ebd4cc356928355490a13e329e5b73d134cdffa827f4aa60b29ab8b78"
# Exact predecessor before checkpoint inventory reuse. Replay still verifies
# its selected policy and bytes, with a fresh inventory after execution.
_PRE_INVENTORY_REUSE_ISSUER_SHA256 = "b78a8d7f13785c9d0263e4149e38a4de0d9bfd43052e552213003d143fac7ba3"
# Exact selected v2 issuer archived at a8eb1da93, before inventory reuse.
# Replay retains its fixed command/policy and every selected snapshot check;
# this does not authorize new issuance with historical producer bytes.
_ARCHIVED_SELECTED_V2_ISSUER_SHA256 = "83af01552173c2ddd24d78996838bfb9d72b79abecc61b9a0e79690bf12ef4d5"
# Exact issuer before the attestation replay could run beside issuance. Its
# receipts are byte-compatible with the current issuer's; replay still checks
# every selected policy, source, runtime and output byte.
_PRE_CONCURRENT_REPLAY_ISSUER_SHA256 = "6c60fb10e4a2b2e05ff67a8b215f86317242346a7cd72b3865a603fc1d2813c4"
# Moving replay scratch outside protected captures changes no selected guest
# path, command, policy or byte comparison. Admit exactly the previous issuer.
_PRE_EXTERNAL_REPLAY_SCRATCH_ISSUER_SHA256 = "fe672aaf62efcf1f2c5cec499af646d59ca0f458107e459a2fbdc59271c4586b"
# Exact issuer before a plan could select the worker's CPU thread count. Its plans
# cannot select one, so their command, policy and replay are unchanged.
_PRE_SELECTED_THREADS_ISSUER_SHA256 = "ba57b6e1bd00734b95b7cb4a64c30f636f715cc313aa048986b3812da04b5fb8"
# Exact predecessor; new source mode never retroactively interprets its plans.
_PRE_NATIVE_INTEGER_ISSUER_SHA256 = "7eb45b2169dac5b2984ab444c183155855b1ecd9de81c6c25890d12e0ce39889"
# Exact issuer at 4acf64171c21, before failure-only process diagnostics were
# retained outside the capture. Successful commands, policies and process
# records are unchanged. Replay still checks every selected byte and executes
# the original snapshot afresh; historical bytes cannot issue a new capture.
_PRE_FAILURE_DIAGNOSTICS_ISSUER_SHA256 = "7eb45b2169dac5b2984ab444c183155855b1ecd9de81c6c25890d12e0ce39889"
# Replay-only predecessor; failure detail changed, successful policy and comparisons did not.
_PRE_REPLAY_DIAGNOSTICS_ISSUER_SHA256 = "4356dafad28f9b81ba77988074c44214d1b52c304cdb29bae63108b4e3276343"
_PRE_PUBLICATION_DIAGNOSTICS_ISSUER_SHA256 = sealed_failure.PRE_RETENTION_ISSUER_SHA256
_V1_SCOPE = "isolated selected FP32 CPU M2M capture; no Phase 0 admission"
_V2_SCOPE = "isolated selected CPU M2M capture; no Phase 0 admission"
_V3_SCOPE = "isolated selected CPU M2M capture with declared model inputs; no Phase 0 admission"
_MAX_SNAPSHOT_BYTES = 16_000_000_000
_MAX_FULL_CAPTURE_SECONDS = 43_200
#: Upper bound for an operator-selected timeout on a checkpoint-free (v2) capture. Absent, a v2
#: capture keeps the historical fixed timeout and its plan bytes stay unchanged.
_MAX_SELECTED_CAPTURE_SECONDS = 14_400


# This literal is part of the selected issuer identity. Refuse changes to the
# extracted verifier; historical issuer admission remains unchanged.
_VALIDATION_SOURCE_SHA256 = "b5b6c46235ea873015013f9a4a314d91680dfbfad43592f7027f8fcfa87b9664"


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


#: The worker's float tokens. A float capture selects no recipe; an int8 capture must select one.
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


def _command_v2(
    output_mount: Path, *, dtype: str, recipe: bool, options: dict[str, Any] | None = None
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
    worker = "/source/merlin-src/merlin/targetgen/_m2m_capture_worker.py"
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
        from merlin.targetgen.quant_recipe import digest as recipe_digest

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


def _declared_cpu_capability() -> str:
    """Select the dispatch recorded by the m2m/native CPU proofs, independent of this host."""
    from merlin.targetgen import native_ieee_certificate

    # Only read the declaration: the full numerical certificate loader also
    # requires the original proof host's library and header paths to exist.
    certificate = strict_json.loads(Path(native_ieee_certificate.__file__).with_suffix(".json").read_bytes())
    capability = certificate["host"]["capability"]
    if not isinstance(capability, str) or not capability:
        raise SealedM2MError("native CPU certificate lacks a declared CPU capability")
    return capability.lower()


def _guest_env(
    output_mount: Path, *, replayable_logs: bool, loader_env: dict[str, str | None] | None = None
) -> dict[str, str]:
    if "ATEN_CPU_CAPABILITY" in (loader_env or {}):
        raise SealedM2MError("loader environment cannot override recorded ATEN_CPU_CAPABILITY")
    env = {"USER": "capture", "LOGNAME": "capture", "XDG_CACHE_HOME": str(output_mount / "cache")}
    env["ATEN_CPU_CAPABILITY"] = _declared_cpu_capability()
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


def _source_tree(
    root: Path,
    *,
    skip_lib64: bool = False,
    skip_python_cache: bool = False,
    readonly_modes: bool = False,
    allow_dangling_symlinks: bool = False,
) -> dict[str, Any]:
    """Digest normalized file bytes, names and modes without retaining a huge manifest.

    The selected Python package may exclude validated transient bytecode caches;
    the venv may skip its known directory alias. Relative internal directory
    aliases retain their link text without traversing the target twice. File
    symlinks are dereferenced by copying and thus by this inventory; outside targets are
    permitted only for the three CPython executable aliases, whose selected
    base interpreter is independently snapshotted. The opt-in readonly view
    removes only write bits to predict the exact mode transformation at freeze.
    Only the broad base inventory permits opaque links, chains and unreadable metadata; selected
    interpreter/startup, import, dependency and data trees remain strict.
    """
    if not root.is_dir() or root.is_symlink():
        raise SealedM2MError(f"selected tree is absent or indirect: {root}")
    records: list[tuple[Any, ...]] = []
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        here = Path(current)
        relative = here.relative_to(root).as_posix()
        if skip_python_cache:
            cache = here / "__pycache__"
            if "__pycache__" in directories:
                if cache.is_symlink() or any(
                    not member.is_file() or member.is_symlink() or member.suffix not in {".pyc", ".pyo"}
                    for member in cache.iterdir()
                ):
                    raise SealedM2MError(f"selected Python cache has an unsupported member: {cache}")
            directories[:] = [name for name in directories if name != "__pycache__"]
        if relative == "." and skip_lib64:
            if (here / "lib64").is_symlink() and (here / "lib64").resolve() == (here / "lib").resolve():
                directories.remove("lib64")
            elif (here / "lib64").exists():
                raise SealedM2MError("venv /lib64 is not the expected /lib alias")
        for name in directories:
            path = here / name
            if path.is_symlink():
                try:
                    link = _base_symlink(path, root) if allow_dangling_symlinks else None
                    if link is None:
                        link = _directory_symlink(path, root)
                except SealedCaptureError as exc:
                    raise SealedM2MError(str(exc)) from exc
                records.append((path.relative_to(root).as_posix(), *sorted(link.items())))
        try:
            omitted = _omitted_members(here)
        except SealedCaptureError as exc:
            raise SealedM2MError(str(exc)) from exc
        if omitted and not allow_dangling_symlinks:
            raise SealedM2MError(f"unreadable source member: {relative}")
        directory_mode = stat.S_IMODE(here.stat().st_mode)
        records.append(
            (
                relative,
                *sorted(
                    {
                        "kind": "directory",
                        "mode": directory_mode & ~0o222 if readonly_modes else directory_mode,
                        "members": sorted([*directories, *files, *omitted]),
                    }.items()
                ),
            )
        )
        for name, row in omitted.items():
            records.append(((here / name).relative_to(root).as_posix(), *sorted(row.items())))
        directories[:] = [name for name in directories if not (here / name).is_symlink()]
        for name in sorted(files):
            path = here / name
            member = path.relative_to(root).as_posix()
            if path.is_symlink():
                try:
                    link = _base_symlink(path, root) if allow_dangling_symlinks else _dangling_symlink(path)
                except SealedCaptureError as exc:
                    raise SealedM2MError(str(exc)) from exc
                if link is not None:
                    if not allow_dangling_symlinks:
                        raise SealedM2MError(f"dangling source link: {member}")
                    records.append((member, *sorted(link.items())))
                    continue
                try:
                    resolved = path.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise SealedM2MError(f"dangling source link: {member}") from exc
                except (OSError, RuntimeError) as exc:
                    raise SealedM2MError(f"unresolvable source link: {member}") from exc
                if not resolved.is_file():
                    raise SealedM2MError(f"non-file link in source: {member}")
                if not resolved.is_relative_to(root) and not (
                    skip_lib64 and member in {"bin/python", "bin/python3", "bin/python3.12"}
                ):
                    raise SealedM2MError(f"external source link: {member}")
            if not path.is_file():
                raise SealedM2MError(f"non-regular source member: {member}")
            unreadable = _unreadable_file(path)
            if unreadable is not None:
                if not allow_dangling_symlinks:
                    raise SealedM2MError(f"unreadable source member: {member}")
                records.append((member, *sorted(unreadable.items())))
                continue
            info = path.stat()
            total += info.st_size
            file_mode = stat.S_IMODE(info.st_mode)
            records.append(
                (
                    member,
                    *sorted(
                        {
                            "kind": "file",
                            "mode": file_mode & ~0o222 if readonly_modes else file_mode,
                            "bytes": info.st_size,
                            "sha256": _file_digest(path),
                        }.items()
                    ),
                )
            )
    return {"members": len(records), "bytes": total, "sha256": _digest(_json(sorted(records)))}


def _copy_source_tree(
    root: Path, destination: Path, *, ignore=None, allow_dangling_symlinks: bool = False
) -> None:
    """Keep directory/opaque aliases, dereference internal files and copy modes last.

    Under the base-only opt-in, unreadable files are omitted from the copy and
    recorded only as metadata markers on their parent directory.
    """

    def copy(directory: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=False)
        if directory == root and allow_dangling_symlinks:
            _, guest_base = _base_link_root(root / ".base-boundary", root)
            os.setxattr(target, _BASE_ROOT_XATTR, os.fsencode(guest_base), follow_symlinks=False)
        names = os.listdir(directory)
        omitted = _omitted_members(directory)
        if omitted and not allow_dangling_symlinks:
            raise SealedM2MError(f"unreadable source member: {directory}")
        for name, row in omitted.items():
            _record_omitted(target, name, row)
        ignored = set() if ignore is None else set(ignore(str(directory), names))
        for name in names:
            if name in ignored:
                continue
            source, copied = directory / name, target / name
            if source.is_symlink():
                link = _base_symlink(source, root) if allow_dangling_symlinks else _dangling_symlink(source)
                if link is not None:
                    copied.symlink_to(link["target"])
                    continue
                if source.is_dir():
                    copied.symlink_to(_directory_symlink(source, root)["target"], target_is_directory=True)
                    continue
            if source.is_dir():
                copy(source, copied)
            else:
                unreadable = _unreadable_file(source)
                if unreadable is not None:
                    if not allow_dangling_symlinks:
                        raise SealedM2MError(f"unreadable source member: {source}")
                    # Never materialized: the guest sees no such member at all.
                    _record_omitted(target, name, unreadable)
                    continue
                shutil.copy2(source, copied)
        shutil.copystat(directory, target)

    copy(root, destination)


def _tree(root: Path, *, allow_dangling_symlinks: bool = False) -> dict[str, dict[str, Any]]:
    """Inventory sealed M2M aliases without changing the static capture policy."""
    return _strict_tree(root, allow_directory_symlinks=True, allow_dangling_symlinks=allow_dangling_symlinks)


def _snapshot_tree(root: Path, *, allow_dangling_symlinks: bool = False) -> dict[str, Any]:
    # File identities are checked while hashing; directory aliases retain their text.
    return _snapshot_rows(_tree(root, allow_dangling_symlinks=allow_dangling_symlinks))


def _declared_loader_env(
    source: str, selected: dict[str, str | None] | None
) -> tuple[dict[str, str | None], list[str]]:
    """Bind literal loader reads, including intentionally absent values, before execution.

    The guest starts with an empty host environment. Dynamic loader reads cannot
    be selected exactly and are therefore outside this policy.
    """
    if not isinstance(selected, dict):
        raise SealedM2MError("v3 capture requires a declared loader environment mapping")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SealedM2MError("selected loader is not parseable Python") from exc
    reads: set[str] = set()
    recognized: set[int] = set()

    def literal(node: ast.AST) -> str:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or not node.value.isidentifier():
            raise SealedM2MError("dynamic loader environment read is not selectable")
        return node.value

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "os" and alias.asname for alias in node.names):
            raise SealedM2MError("aliased loader environment module is not selectable")
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(alias.name in {"environ", "getenv"} for alias in node.names)
        ):
            raise SealedM2MError("loader must read environment through literal os calls")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "os"
        ):
            raise SealedM2MError("indirect loader environment access is not selectable")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in {"putenv", "unsetenv"}
            ):
                raise SealedM2MError("loader environment mutation is not selectable")
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "getenv":
                reads.add(literal(node.args[0]) if node.args else literal(node))
            if (
                node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.func.value.attr == "environ"
            ):
                recognized.add(id(node.func.value))
                reads.add(literal(node.args[0]) if node.args else literal(node))
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            if not isinstance(node.ctx, ast.Load):
                raise SealedM2MError("loader environment mutation is not selectable")
            recognized.add(id(node.value))
            reads.add(literal(node.slice))
    if any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
        and id(node) not in recognized
        for node in ast.walk(tree)
    ):
        raise SealedM2MError("indirect loader environment read is not selectable")
    protected = {"USER", "LOGNAME", "XDG_CACHE_HOME", "TORCH_CPP_LOG_LEVEL", "HOME", "PATH", "ATEN_CPU_CAPABILITY"}
    for name, value in selected.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name.upper() != name
            or name in protected
            or name.startswith(("PYTHON", "LD_", "DYLD_"))
        ):
            raise SealedM2MError("selected loader environment has an unsafe name")
        if value is not None and (not isinstance(value, str) or "\x00" in value or len(value) > 4096):
            raise SealedM2MError("selected loader environment has an unsafe value")
    if not reads <= set(selected):
        raise SealedM2MError(
            "loader environment reads lack explicit present-or-absent selection: "
            + ", ".join(sorted(reads - set(selected)))
        )
    return dict(sorted(selected.items())), sorted(reads)


def _input_selection(source: Path, guest_member: str) -> dict[str, Any]:
    """Select one file or self-contained tree under the guest's read-only input root."""
    if not isinstance(guest_member, str):
        raise SealedM2MError("selected guest input member is unsafe")
    member = PurePosixPath(guest_member)
    if (
        not guest_member
        or guest_member == "."
        or member.is_absolute()
        or ".." in member.parts
        or member.as_posix() != guest_member
        or "\\" in guest_member
        or "\x00" in guest_member
    ):
        raise SealedM2MError("selected guest input member is unsafe")
    source = _canonical_path(source, exists=True)
    if source.is_symlink():
        raise SealedM2MError("selected checkpoint or input must not be an indirect root")
    if source.is_file():
        if _unreadable_file(source) is not None:
            raise SealedM2MError("selected input file is unreadable")
        return {
            "source": str(source),
            "guest_member": guest_member,
            "kind": "file",
            "bytes": source.stat().st_size,
            "sha256": _file_digest(source),
        }
    if source.is_dir():
        return {"source": str(source), "guest_member": guest_member, "kind": "tree", "tree": _source_tree(source)}
    raise SealedM2MError("selected checkpoint or input is not a file or directory")


def _selected_inputs(
    checkpoint: Path | None,
    checkpoint_guest_member: str | None,
    extra_inputs: dict[str, Path] | None,
) -> list[dict[str, Any]]:
    if (checkpoint is None) != (checkpoint_guest_member is None):
        raise SealedM2MError("checkpoint and its guest member must be selected together")
    if extra_inputs is not None and not isinstance(extra_inputs, dict):
        raise SealedM2MError("extra selected inputs must map guest members to source paths")
    rows = []
    if checkpoint is not None:
        rows.append({"role": "checkpoint", **_input_selection(checkpoint, checkpoint_guest_member)})
    for member, path in sorted((extra_inputs or {}).items()):
        rows.append({"role": "input", **_input_selection(path, member)})
    members = [PurePosixPath(row["guest_member"]) for row in rows]
    if any(
        a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(members) for b in members[i + 1 :]
    ):
        raise SealedM2MError("selected guest input members overlap")
    return rows


def _verify_selected_input(row: dict[str, Any], path: Path) -> None:
    if row["kind"] == "file":
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or _file_digest(path) != row["sha256"]
        ):
            raise SealedM2MError("selected input file bytes differ")
    elif row["kind"] == "tree":
        if _snapshot_tree(path) != row["tree"]:
            raise SealedM2MError("selected input tree bytes or membership differ")
    else:
        raise SealedM2MError("selected input kind is unsupported")


def _venv_home(venv: Path) -> Path:
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        raise SealedM2MError("selected interpreter has no pyvenv.cfg")
    homes = [
        line.partition("=")[2].strip()
        for line in cfg.read_text().splitlines()
        if line.partition("=")[0].strip() == "home"
    ]
    if len(homes) != 1 or not Path(homes[0]).is_absolute() or ".." in Path(homes[0]).parts:
        raise SealedM2MError("unsupported venv home")
    base = Path(homes[0]).parent
    if not (base / "bin/python3.12").resolve().is_file():
        raise SealedM2MError("venv base CPython is absent")
    try:
        selected_python = _canonical_path(base / "bin/python3.12", exists=True)
        if _unreadable_file(selected_python) is not None:
            raise SealedCaptureError("selected base CPython is unreadable")
    except SealedCaptureError as exc:
        raise SealedM2MError(str(exc)) from exc
    return base


def _ldd_library_path(line: str) -> Path | None:
    """Parse the two ordinary `ldd` dependency forms without matching line text."""
    fields = line.partition("(")[0].split()
    if len(fields) == 3 and fields[1] == "=>":
        raw = fields[2]
    elif len(fields) == 1:
        raw = fields[0]
    else:
        return None
    if raw.startswith(("/lib", "/usr/lib")):
        return Path(raw)
    return None


def _require_plain_base_resolution(path: Path, base: Path) -> None:
    """Refuse a selected closure path that depends on base metadata-only entries.

    Every symlink hop the kernel would take is replayed lexically. A hop inside
    the base that the base inventory records as external, dangling or a
    directory chain, or an unreadable final base file, is refused; such an
    entry is never content in the guest, so the closure would be unbound.
    """
    pending = list(Path(path).parts)
    if not pending or pending[0] != "/":
        raise SealedM2MError(f"selected closure path is not absolute: {path}")
    pending.pop(0)
    current, hops = Path("/"), 0
    try:
        while pending:
            part = pending.pop(0)
            if part == "..":
                current = current.parent
                continue
            candidate = current / part
            if not candidate.is_symlink():
                current = candidate
                continue
            hops += 1
            if hops > 40:
                raise SealedM2MError(f"selected closure path has a link loop: {path}")
            if candidate.is_relative_to(base):
                row = _base_symlink(candidate, base)
                if row is not None:
                    if row["kind"] != "symlink":
                        raise SealedM2MError(f"selected closure path resolves through a {row['kind']}: {path}")
                    _directory_symlink(candidate, base)  # refuses directory chains
            text = os.readlink(candidate)
            pending = list(Path(text).parts) + pending
            if text.startswith("/"):
                pending.pop(0)
                current = Path("/")
        if current.is_relative_to(base) and _unreadable_file(current) is not None:
            raise SealedM2MError(f"selected closure path is an unreadable base file: {path}")
    except SealedCaptureError as exc:
        raise SealedM2MError(f"selected closure path depends on base metadata: {path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise SealedM2MError(f"selected closure path is absent: {path}") from exc


def _system_libs(interpreter: Path, torch_so: Path, numpy_so: Path) -> tuple[Path, ...]:
    external: set[Path] = set()
    for binary in (interpreter, torch_so, numpy_so):
        result = subprocess.run(
            ["/usr/bin/ldd", str(binary)],
            capture_output=True,
            text=True,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            timeout=30,
        )
        if result.returncode or "not found" in result.stdout:
            raise SealedM2MError(f"ELF dependencies unavailable for {binary.name}")
        for line in result.stdout.splitlines():
            path = _ldd_library_path(line)
            if path is not None:
                if not path.is_file() or not path.resolve().is_file():
                    raise SealedM2MError(f"unavailable system ELF library: {path}")
                external.add(path)
    return tuple(sorted(external))


def prepare_plan(
    *,
    m2m_root: Path,
    frozen_origin: dict[str, str] | None = None,
    workload_root: Path,
    worker: Path,
    venv: Path,
    schemas_root: Path,
    dtype: str = "fp32",
    recipe: Path | None = None,
    max_snapshot_bytes: int = _MAX_SNAPSHOT_BYTES,
    worker_options: dict[str, Any] | None = None,
    checkpoint: Path | None = None,
    checkpoint_guest_member: str | None = None,
    extra_inputs: dict[str, Path] | None = None,
    loader_env: dict[str, str | None] | None = None,
    execution_timeout_seconds: int | None = None,
    runtime_reuse: dict | None = None,
) -> dict[str, Any]:
    """Read-only selection and space bound; no capture or admission claim."""
    options = _worker_options(worker_options)
    if type(max_snapshot_bytes) is not int or not 0 < max_snapshot_bytes <= _MAX_SNAPSHOT_BYTES:
        raise SealedM2MError("snapshot cap must be a positive bound no larger than 16 GB")
    m2m_root = _canonical_path(m2m_root, exists=True)
    workload_root = _canonical_path(workload_root, exists=True)
    worker = _canonical_path(worker, exists=True)
    venv = _canonical_path(venv, exists=True)
    schemas_root = _canonical_path(schemas_root, exists=True)
    if native_integer.selected(options):
        try:
            native_integer.require_selection(options, dtype=dtype, recipe=recipe)
        except ValueError as error:
            raise SealedM2MError(str(error)) from error
        selected_recipe = None
    else:
        selected_recipe = _recipe_selection(recipe, dtype=dtype)
    version = SCHEMA_V3 if checkpoint is not None or extra_inputs is not None or loader_env is not None else SCHEMA
    if native_integer.selected(options) and (
        checkpoint is not None or checkpoint_guest_member is not None or extra_inputs or loader_env
    ):
        raise SealedM2MError(
            "native integer source mode supports checkpoint-free, environment-free source captures only"
        )
    if frozen_origin is not None and version != SCHEMA:
        raise SealedM2MError("frozen Phase 0 M2M origin is only supported for v2 captures")
    if version == SCHEMA_V3:
        if (
            type(execution_timeout_seconds) is not int
            or not _TIMEOUT_SECONDS <= execution_timeout_seconds <= _MAX_FULL_CAPTURE_SECONDS
        ):
            raise SealedM2MError("v3 capture requires a selected timeout between 120 and 43200 seconds")
    elif not _selected_timeout_valid(version, execution_timeout_seconds):
        raise SealedM2MError(
            f"v2 capture timeout must be absent (historical {_TIMEOUT_SECONDS} s) or between "
            f"{_TIMEOUT_SECONDS} and {_MAX_SELECTED_CAPTURE_SECONDS} seconds"
        )
    merlin_root = worker.parents[1]
    if worker != merlin_root / "targetgen/_m2m_capture_worker.py" or not (merlin_root / "__init__.py").is_file():
        raise SealedM2MError("selected worker must belong to the selected Merlin source package")
    if schemas_root not in {merlin_root / "_data/schemas", merlin_root.parent.parent / "merlin/schemas"}:
        raise SealedM2MError("selected schemas must belong to the selected Merlin package or source checkout")
    if not schemas_root.is_dir() or any(
        not (schemas_root / name).is_file() for name in ("quant_formats.registry.yaml", "quant_format.schema.yaml")
    ):
        raise SealedM2MError("selected Merlin schema tree lacks the quant-format registry and validator")
    if not (m2m_root / "m2m/api.py").is_file() or not (workload_root / "loader.py").is_file():
        raise SealedM2MError("M2M package or workload loader is absent")
    missing_api = _capture_api_missing(m2m_root)
    if missing_api:
        raise SealedM2MError(
            "selected Model2MLIR lacks same-conversion materialization/receipt API: " + ", ".join(missing_api)
        )
    if options.get("stage_fp32"):
        _command_v2(Path("/capture-out"), dtype=dtype, recipe=selected_recipe is not None, options=options)
        missing_stage_api = _fp32_stage_api_missing(m2m_root)
        if missing_stage_api:
            raise SealedM2MError("selected Model2MLIR lacks FP32 staging API: " + ", ".join(missing_stage_api))
    loader_source = (workload_root / "loader.py").read_text()
    if version == SCHEMA_V3:
        selected_env, loader_reads = _declared_loader_env(loader_source, loader_env)
        input_rows = _selected_inputs(checkpoint, checkpoint_guest_member, extra_inputs)
        if not native_integer.selected(options) and not any(row["role"] == "checkpoint" for row in input_rows):
            raise SealedM2MError("v3 full-model capture requires an explicit checkpoint input")
    else:
        if _loader_env_reads(loader_source):
            raise SealedM2MError("this first sealed policy rejects environment-reading loaders")
        selected_env, loader_reads, input_rows = {}, [], []
    if not worker.is_file() or worker.suffix != ".py":
        raise SealedM2MError("worker must be a selected Python source file")
    try:
        origin = git_origin(m2m_root, clean=version == SCHEMA) if frozen_origin is None else None
    except M2MOriginError as exc:
        raise SealedM2MError(str(exc)) from exc
    base = _venv_home(venv)
    interpreter = venv / "bin/python"
    if not interpreter.is_file() or interpreter.resolve() != (base / "bin/python3.12").resolve():
        raise SealedM2MError("venv interpreter does not resolve to selected base CPython")
    site = venv / "lib/python3.12/site-packages"
    torch_so = next(site.glob("torch/_C*.so"), None)
    numpy_so = next(site.glob("numpy/_core/_multiarray_umath*.so"), None)
    if torch_so is None or numpy_so is None:
        raise SealedM2MError("selected torch/NumPy ELF roots are absent")
    libs = _system_libs(interpreter, torch_so, numpy_so)
    for library in libs:
        _require_plain_base_resolution(library, base.resolve())
    # Inventory *before* copying: directory aliases retain their topology and
    # file aliases are dereferenced. These exact normalized bytes and topology are
    # part of the plan and are rechecked at issuance, before any 9 GB copy.
    selected_trees = {
        "venv": _source_tree(venv, skip_lib64=True),
        "base": _source_tree(base.resolve(), allow_dangling_symlinks=True),
        # The guest receives only the selected package source (see _stage_source),
        # never these transient caches. -B prevents guest bytecode writes; it does
        # not by itself disable reading existing bytecode.
        "m2m": _source_tree(m2m_root / "m2m", skip_python_cache=True),
        "workload": _source_tree(workload_root),
        "merlin": _source_tree(merlin_root, skip_python_cache=True),
        "schemas": _source_tree(schemas_root),
    }
    if frozen_origin is not None:
        try:
            selected = verify_frozen_selector(frozen_origin, m2m_root, selected_trees["m2m"])
        except M2MOriginError as exc:
            raise SealedM2MError(str(exc)) from exc
        dirty = ""
    else:
        assert origin is not None
        selected, dirty = origin["commit"], origin["worktree_status"]
    estimate = sum(row["bytes"] for row in selected_trees.values())
    if frozen_origin is not None:
        estimate += Path(frozen_origin["path"]).stat().st_size
    estimate += sum(path.stat().st_size for path in libs)
    if selected_recipe is not None:
        estimate += selected_recipe["bytes"]
    estimate += sum(row.get("bytes", (row.get("tree") or {}).get("bytes", 0)) for row in input_rows)
    if estimate > max_snapshot_bytes:
        raise SealedM2MError(f"normalized snapshot bytes {estimate} exceed selected cap")
    plan = {
        "schema": version,
        "status": "plan_only",
        "m2m_root": str(m2m_root),
        "m2m_commit": selected,
        **({"frozen_origin": frozen_origin} if frozen_origin is not None else {}),
        **(
            {"m2m_worktree_status": dirty, "m2m_commit_role": "origin_hint_not_execution_authority"}
            if version == SCHEMA_V3
            else {}
        ),
        "workload_root": str(workload_root),
        "worker": str(worker),
        "merlin_root": str(merlin_root),
        "schemas_root": str(schemas_root),
        "venv": str(venv),
        "base": str(base),
        "dtype": dtype,
        "recipe": selected_recipe,
        "worker_sha256": _file_digest(worker),
        "loader_sha256": _file_digest(workload_root / "loader.py"),
        "system_libs": [str(path) for path in libs],
        "estimate_bytes": estimate,
        "selected_trees": selected_trees,
        "max_snapshot_bytes": max_snapshot_bytes,
        "command_template_sha256": _digest(
            _command_v2(Path("/capture-out"), dtype=dtype, recipe=selected_recipe is not None, options=options)[
                -1
            ].encode()
        ),
        "scope": (
            "one selected CPU capture; declared loader environment and checkpoint source closure"
            if version == SCHEMA_V3
            else "one selected CPU capture; no env-reading loader or unselected checkpoint"
        ),
        **(
            {"selected_inputs": input_rows, "loader_env": selected_env, "loader_env_reads": loader_reads}
            if version == SCHEMA_V3
            else {}
        ),
        # V3 always selects a timeout; v2 records one only when the operator selected it, so an
        # unselected v2 plan keeps its historical bytes and fixed timeout.
        **(
            {"execution_timeout_seconds": execution_timeout_seconds}
            if version == SCHEMA_V3 or execution_timeout_seconds is not None
            else {}
        ),
        # Present only when selected, so a plan without options keeps its historical bytes.
        **({"worker_options": options} if options else {}),
        **(
            {"native_integer_verifier_sha256": _file_digest(Path(native_integer.__file__))}
            if native_integer.selected(options)
            else {}
        ),
    }
    if runtime_reuse is not None:
        readonly_runtime.verify(plan, runtime_reuse, workload_root.parent)
        plan["runtime_reuse"] = runtime_reuse
        plan["readonly_runtime_verifier_sha256"] = _file_digest(Path(readonly_runtime.__file__))
    return plan


def _validate_snapshots(
    source: Path, runtime: Path, output_mount: Path, *, schema: str = SCHEMA_V1, runtime_reuse: dict | None = None
) -> None:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._validate_snapshots(
        source, runtime, output_mount, schema=schema, runtime_reuse=runtime_reuse
    )


def _materialized(
    output: Path, source: Path, output_mount: Path, *, worker_member: str = "worker.py"
) -> dict[str, Any]:
    """Verify materialization and require the dispatch selected by the recorded policy."""
    result = _validation_functions()._materialized(output, source, output_mount, worker_member=worker_member)
    declared = _declared_cpu_capability()
    try:
        record = strict_json.loads((output / "meta.json").read_bytes())
        dispatch = record["determinism"]["cpu_dispatch"]
        capability = dispatch["capability"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SealedM2MError("capture lacks an observed CPU dispatch capability") from exc
    if not isinstance(capability, str) or capability.lower() != declared or dispatch.get("status") != "observed":
        raise SealedM2MError(f"capture CPU dispatch {capability!r} does not match declared capability {declared!r}")
    return result


def _materialized_v2(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._materialized_v2(output, source, output_mount, plan)


def _dynamic_integer_census(meta: dict, stats: dict, stage: Path, emitted: int, exported: int) -> int:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._dynamic_integer_census(meta, stats, stage, emitted, exported)


def _materialized_v3(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._materialized_v3(output, source, output_mount, plan)


def _stage_source(plan: dict[str, Any], source: Path) -> None:
    """Delegate to the source-pinned capture validation implementation."""
    return _validation_functions()._stage_source(plan, source)


def issue(
    plan: dict[str, Any],
    run_dir: Path,
    *,
    bwrap_binary: Path | None = None,
    capture_selection_sha256: str | None = None,
    selected_system_libraries: list[dict[str, Any]] | None = None,
    selected_bwrap_sha256: str | None = None,
) -> Path:
    """Make one private snapshot and capture; receipt remains pending replay."""
    return _issue(
        plan,
        run_dir,
        bwrap_binary=bwrap_binary,
        capture_selection_sha256=capture_selection_sha256,
        selected_system_libraries=selected_system_libraries,
        selected_bwrap_sha256=selected_bwrap_sha256,
    )[0]


def issue_and_replay(
    plan: dict[str, Any],
    run_dir: Path,
    *,
    cpus: tuple[frozenset[int], frozenset[int]],
    bwrap_binary: Path | None = None,
    capture_selection_sha256: str | None = None,
    selected_system_libraries: list[dict[str, Any]] | None = None,
    selected_bwrap_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Issue one capture and its independent replay at once, on two disjoint CPU sets.

    The replay is the same fresh execution :func:`replay_verify` performs after issuance: the
    identical sandboxed command on the same sealed snapshot, into its own empty output directory
    bound at the same guest path. It only starts with, not after, the issued execution. Once the
    pending receipt is written, :func:`replay_verify` checks it exactly as a later replay would and
    compares this replay's process record, materialized receipt and output tree with it.
    """
    issue_cpus, replay_cpus = (frozenset(int(cpu) for cpu in part) for part in cpus)
    if not issue_cpus or not replay_cpus or issue_cpus & replay_cpus:
        raise SealedM2MError("a concurrent replay needs two nonempty disjoint CPU sets")
    receipt, replay = _issue(
        plan,
        run_dir,
        bwrap_binary=bwrap_binary,
        capture_selection_sha256=capture_selection_sha256,
        selected_system_libraries=selected_system_libraries,
        selected_bwrap_sha256=selected_bwrap_sha256,
        replay_cpus=(issue_cpus, replay_cpus),
    )
    return receipt, replay


def _issue(
    plan: dict[str, Any],
    run_dir: Path,
    *,
    bwrap_binary: Path | None = None,
    capture_selection_sha256: str | None = None,
    selected_system_libraries: list[dict[str, Any]] | None = None,
    selected_bwrap_sha256: str | None = None,
    replay_cpus: tuple[frozenset[int], frozenset[int]] | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    if plan.get("schema") not in {SCHEMA, SCHEMA_V3} or plan.get("status") != "plan_only":
        raise SealedM2MError("unsupported M2M plan")
    if capture_selection_sha256 is not None and (
        not isinstance(capture_selection_sha256, str)
        or len(capture_selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in capture_selection_sha256)
    ):
        raise SealedM2MError("capture selection requires an exact SHA-256 identity")
    if capture_selection_sha256 is not None and (selected_system_libraries is None or selected_bwrap_sha256 is None):
        raise SealedM2MError("selected capture requires antecedent system-library and bubblewrap bytes")
    if selected_system_libraries is not None and (
        not isinstance(selected_system_libraries, list)
        or [row.get("path") for row in selected_system_libraries] != plan.get("system_libs")
    ):
        raise SealedM2MError("selected system-library roster differs from the plan")
    selected = prepare_plan(
        m2m_root=Path(plan["m2m_root"]),
        frozen_origin=plan.get("frozen_origin"),
        workload_root=Path(plan["workload_root"]),
        worker=Path(plan["worker"]),
        venv=Path(plan["venv"]),
        schemas_root=Path(plan["schemas_root"]),
        dtype=plan["dtype"],
        recipe=Path(plan["recipe"]["path"]) if plan.get("recipe") else None,
        max_snapshot_bytes=plan["max_snapshot_bytes"],
        worker_options=plan.get("worker_options"),
        **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
        **(
            {
                "checkpoint": next(
                    (Path(row["source"]) for row in plan["selected_inputs"] if row["role"] == "checkpoint"), None
                ),
                "checkpoint_guest_member": next(
                    (row["guest_member"] for row in plan["selected_inputs"] if row["role"] == "checkpoint"), None
                ),
                "extra_inputs": {
                    row["guest_member"]: Path(row["source"])
                    for row in plan["selected_inputs"]
                    if row["role"] == "input"
                },
                "loader_env": plan["loader_env"],
                "execution_timeout_seconds": plan["execution_timeout_seconds"],
            }
            if plan["schema"] == SCHEMA_V3
            else (
                {"execution_timeout_seconds": plan["execution_timeout_seconds"]}
                if "execution_timeout_seconds" in plan
                else {}
            )
        ),
    )
    if selected != plan:
        raise SealedM2MError("selected M2M plan changed")
    run_dir = _canonical_path(run_dir, exists=False)
    # The source venv is read-only.  The copy and capture live on the selected
    # run filesystem, so only that destination's capacity is relevant here.
    from .runtime_store import snapshot_space_requirement

    reuse = plan.get("runtime_reuse")
    if reuse is not None:
        readonly_runtime.verify(plan, reuse, run_dir.parent)
        required_bytes, verified_cache = plan["estimate_bytes"] - reuse["runtime_bytes"] + 64 * 1024**2, None
    else:
        required_bytes, verified_cache = snapshot_space_requirement(plan, run_dir.parent, selected_system_libraries)
    if required_bytes > shutil.disk_usage(run_dir.parent).free:
        raise SealedM2MError("normalized snapshot bytes exceed selected run filesystem free space")
    inputs = [
        Path(plan[key])
        for key in ("m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base")
    ]
    if plan.get("recipe"):
        inputs.append(Path(plan["recipe"]["path"]))
    if reuse is not None:
        inputs.append(Path(reuse["entry"]))
    if plan["schema"] == SCHEMA_V3:
        inputs.extend(Path(row["source"]) for row in plan["selected_inputs"])
    if any(run_dir == path or run_dir.is_relative_to(path) or path.is_relative_to(run_dir) for path in inputs):
        raise SealedM2MError("run directory overlaps a selected input")
    bwrap = _bwrap_binary(bwrap_binary)
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes differ from the pre-execution selection")
    _probe_sandbox(bwrap)
    run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source = run_dir / "snapshots/source"
    runtime = run_dir / "snapshots/guest-root"
    source.mkdir(parents=True)
    runtime.mkdir()
    _stage_source(plan, source)
    # The selected runtime (venv, base interpreter, system libraries) is identical across captures, so
    # it is materialized once into a content-addressed store and hard-linked into this private guest
    # root. Every byte is still re-verified against the plan before anything executes.
    from .runtime_store import link_runtime

    if reuse is not None:
        readonly_runtime.stage(reuse, runtime)
    else:
        link_runtime(plan, runtime, verified_cache=verified_cache, selected_system_libraries=selected_system_libraries)
    output = run_dir / "capture"
    for name in ("source", "capture-out", "tmp", "dev"):
        (runtime / name).mkdir()
    (runtime / output.relative_to("/")).mkdir(parents=True)
    _validate_snapshots(
        source, runtime, output, schema=plan["schema"], **({"runtime_reuse": reuse} if reuse is not None else {})
    )
    # Rehash the copied bytes against the antecedent selection, not mutable
    # host paths: a source change after prepare_plan must never be executed.
    source_rows = _tree(source)
    runtime_rows = readonly_runtime.effective_rows(reuse) if reuse else _tree(runtime, allow_dangling_symlinks=True)
    selected_schemas = _verify_staged_selection(
        plan, source, Path(reuse["entry"]) if reuse else runtime, source_rows=source_rows, runtime_rows=runtime_rows
    )
    if _file_digest(source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py") != plan["worker_sha256"]:
        raise SealedM2MError("worker snapshot differs from selected bytes")
    if plan.get("recipe") and _file_digest(source / "inputs/quant_recipe.json") != plan["recipe"]["sha256"]:
        raise SealedM2MError("selected recipe snapshot differs from selected bytes")
    for index, name in enumerate(plan["system_libs"]):
        member = Path(name).relative_to("/").as_posix()
        copied = readonly_runtime.file_path(reuse, member) if reuse else runtime / member
        expected = selected_system_libraries[index] if selected_system_libraries is not None else None
        if expected is not None:
            if copied.stat().st_size != expected.get("bytes") or _file_digest(copied) != expected.get("sha256"):
                raise SealedM2MError(f"system ELF snapshot differs from the pre-execution selection: {name}")
        elif _file_digest(copied) != _file_digest(Path(name)):
            raise SealedM2MError(f"system ELF snapshot differs from selected bytes: {name}")
    source_digest = _snapshot_rows(source_rows)
    runtime_digest = _snapshot_tree(runtime, allow_dangling_symlinks=True) if reuse else _snapshot_rows(runtime_rows)
    del source_rows, runtime_rows
    if source_digest["bytes"] + runtime_digest["bytes"] > plan["max_snapshot_bytes"]:
        raise SealedM2MError("actual snapshot bytes exceed selected cap")
    output.mkdir()
    command = _command_v2(
        output, dtype=plan["dtype"], recipe=plan.get("recipe") is not None, options=plan.get("worker_options")
    )
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes changed before sandbox execution")
    replayable_logs = capture_selection_sha256 is not None
    loader_env = plan.get("loader_env") if plan["schema"] == SCHEMA_V3 else None
    if replay_cpus is None:
        process = _execute(
            bwrap,
            runtime,
            source,
            output,
            command,
            output,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": reuse} if reuse is not None else {}),
        )
        with sealed_failure.publication(run_dir, replay=None, plan=plan, selection_sha256=capture_selection_sha256):
            return _write_pending(
                plan=plan,
                run_dir=run_dir,
                source=source,
                runtime=runtime,
                output=output,
                command=command,
                process=process,
                bwrap=bwrap,
                source_digest=source_digest,
                runtime_digest=runtime_digest,
                selected_schemas=selected_schemas,
                replayable_logs=replayable_logs,
                loader_env=loader_env,
                capture_selection_sha256=capture_selection_sha256,
                selected_bwrap_sha256=selected_bwrap_sha256,
            ), None
    # The replay's own empty output, made as replay_verify makes it: inside the run, with the
    # issued output's mode, bound at the same guest path.
    replay_output = Path(tempfile.mkdtemp(prefix="m2m-replay-", dir=run_dir))
    with sealed_failure.publication(
        run_dir, replay=replay_output, plan=plan, selection_sha256=capture_selection_sha256
    ):
        replay_output.chmod(output.stat().st_mode & 0o777)
        process, replay_process = _execute_beside(
            bwrap,
            runtime,
            source,
            (output, replay_output),
            command,
            output,
            cpus=replay_cpus,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": reuse} if reuse is not None else {}),
        )
        receipt = _write_pending(
            plan=plan,
            run_dir=run_dir,
            source=source,
            runtime=runtime,
            output=output,
            command=command,
            process=process,
            bwrap=bwrap,
            source_digest=source_digest,
            runtime_digest=runtime_digest,
            selected_schemas=selected_schemas,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            capture_selection_sha256=capture_selection_sha256,
            selected_bwrap_sha256=selected_bwrap_sha256,
        )
        return receipt, replay_verify(run_dir, bwrap_binary=bwrap, _executed=(replay_process, replay_output))


def _write_pending(
    *,
    plan: dict[str, Any],
    run_dir: Path,
    source: Path,
    runtime: Path,
    output: Path,
    command: tuple[str, ...],
    process: dict[str, Any],
    bwrap: Path,
    source_digest: dict[str, Any],
    runtime_digest: dict[str, Any],
    selected_schemas: Any,
    replayable_logs: bool,
    loader_env: dict[str, str | None] | None,
    capture_selection_sha256: str | None,
    selected_bwrap_sha256: str | None,
) -> Path:
    """Check the executed capture and write its pending receipt (shared by both issue modes)."""
    materialized = (
        _materialized_v3(output, source, output, plan)
        if plan["schema"] == SCHEMA_V3
        else _materialized_v2(output, source, output, plan)
    )
    if (_snapshot_tree(source), _snapshot_tree(runtime, allow_dangling_symlinks=True)) != (
        source_digest,
        runtime_digest,
    ):
        raise SealedM2MError("sealed source or runtime changed during capture")
    if plan.get("runtime_reuse") is not None:
        readonly_runtime.verify(plan, plan["runtime_reuse"], runtime)
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes changed during sandbox execution")
    receipt = run_dir / "sealed_m2m_pending.json"
    payload = {
        "schema": plan["schema"],
        "status": "pending_replay",
        "issuer_sha256": _file_digest(Path(__file__)),
        "nonce": secrets.token_hex(16),
        "plan": plan,
        "command": list(command),
        "policy_sha256": _policy(
            command,
            output,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
        ),
        "source": source_digest,
        "guest_root": runtime_digest,
        "output": _snapshot_tree(output),
        "schemas": selected_schemas,
        "process": process,
        "materialized": materialized,
        "bwrap_sha256": _file_digest(bwrap),
        "scope": _V3_SCOPE if plan["schema"] == SCHEMA_V3 else _V2_SCOPE,
    }
    if capture_selection_sha256 is not None:
        payload["capture_selection_sha256"] = capture_selection_sha256
    with receipt.open("xb") as stream:
        stream.write(_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt.chmod(0o444)
    return receipt


def replay_verify(
    run_dir: Path,
    *,
    bwrap_binary: Path | None = None,
    _executed: tuple[dict[str, Any], Path] | None = None,
) -> dict[str, Any]:
    """Independent replay from exact private bytes; no Phase 0 status is granted.

    ``_executed`` is only for :func:`issue_and_replay`: the process record and fresh output of a
    replay that ran beside issuance, compared here exactly as a replay run now would be.
    """
    run_dir = _canonical_path(run_dir, exists=True)
    receipt = run_dir / "sealed_m2m_pending.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise SealedM2MError("pending M2M receipt is absent or indirect")
    try:
        doc = strict_json.loads(receipt.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        raise SealedM2MError("pending M2M receipt is unreadable") from exc
    plan = doc.get("plan") or {}
    schema = doc.get("schema")
    if schema == SCHEMA_V1:
        expected_issuer = _V1_ISSUER_SHA256
        expected_template = _digest((_LAUNCH_PREFIX + _LAUNCH_SUFFIX).encode())
        expected_command = _command(run_dir / "capture")
        expected_scope = _V1_SCOPE
    elif schema in {SCHEMA, SCHEMA_V3}:
        dtype, recipe = plan.get("dtype"), plan.get("recipe")
        is_native = native_integer.selected(plan.get("worker_options"))
        if is_native:
            try:
                native_integer.require_selection(plan["worker_options"], dtype=dtype, recipe=recipe)
            except ValueError as error:
                raise SealedM2MError(str(error)) from error
        elif dtype not in _FLOAT_DTYPES | {"int8"} or (recipe is not None) != (dtype == "int8"):
            raise SealedM2MError("pending M2M receipt has an unsupported dtype or recipe selection")
        if not _selected_timeout_valid(schema, plan.get("execution_timeout_seconds")):
            raise SealedM2MError("pending capture has no selected bounded execution timeout")
        selected_digest = doc.get("capture_selection_sha256")
        if selected_digest is not None and (
            not isinstance(selected_digest, str)
            or len(selected_digest) != 64
            or any(character not in "0123456789abcdef" for character in selected_digest)
        ):
            raise SealedM2MError("pending M2M receipt has a malformed capture selection identity")
        expected_issuer = _file_digest(Path(__file__))
        expected_template = _digest(
            _command_v2(
                Path("/capture-out"), dtype=dtype, recipe=recipe is not None, options=plan.get("worker_options")
            )[-1].encode()
        )
        expected_command = _command_v2(
            run_dir / "capture", dtype=dtype, recipe=recipe is not None, options=plan.get("worker_options")
        )
        expected_scope = _V3_SCOPE if schema == SCHEMA_V3 else _V2_SCOPE
    else:
        raise SealedM2MError("pending M2M receipt has an unsupported schema")
    replayable_logs = schema in {SCHEMA, SCHEMA_V3} and doc.get("capture_selection_sha256") is not None
    loader_env = plan.get("loader_env") if schema == SCHEMA_V3 else None
    supported_issuer = (
        {expected_issuer, _PRE_V3_ISSUER_SHA256, _HISTORICAL_V2_ISSUER_SHA256}
        if schema == SCHEMA and doc.get("capture_selection_sha256") is None
        else (
            {expected_issuer, _PRE_V3_ISSUER_SHA256, _STRICT_JSON_PREDECESSOR_V2_ISSUER_SHA256}
            if schema == SCHEMA
            else (
                {expected_issuer, _ARCHIVED_V3_ISSUER_SHA256, _PRE_CACHE_NORMALIZATION_V3_ISSUER_SHA256}
                if schema == SCHEMA_V3
                else {expected_issuer}
            )
        )
    )
    if schema in {SCHEMA, SCHEMA_V3} and doc.get("capture_selection_sha256") is not None:
        supported_issuer.add(_STRICT_JSON_SELECTED_ISSUER_SHA256)
    if schema == SCHEMA and doc.get("capture_selection_sha256") is not None:
        supported_issuer.add(_ARCHIVED_SELECTED_V2_ISSUER_SHA256)
    if schema in {SCHEMA, SCHEMA_V3}:
        supported_issuer.add(_PRE_INVENTORY_REUSE_ISSUER_SHA256)
        supported_issuer.add(_PRE_CONCURRENT_REPLAY_ISSUER_SHA256)
        supported_issuer.add(_PRE_EXTERNAL_REPLAY_SCRATCH_ISSUER_SHA256)
        supported_issuer.add(_PRE_SELECTED_THREADS_ISSUER_SHA256)
        if not native_integer.selected(plan.get("worker_options")):
            supported_issuer.add(_PRE_NATIVE_INTEGER_ISSUER_SHA256)
        supported_issuer.add(_PRE_FAILURE_DIAGNOSTICS_ISSUER_SHA256)
        supported_issuer.add(_PRE_REPLAY_DIAGNOSTICS_ISSUER_SHA256)
        supported_issuer.add(_PRE_PUBLICATION_DIAGNOSTICS_ISSUER_SHA256)
    if schema in {SCHEMA, SCHEMA_V3} and "frozen_origin" not in plan:
        supported_issuer.add(_PRE_FROZEN_ORIGIN_ISSUER_SHA256)
    if native_integer.selected(plan.get("worker_options")) or "runtime_reuse" in plan:
        supported_issuer = {expected_issuer}
    if (
        doc.get("status") != "pending_replay"
        or doc.get("issuer_sha256") not in supported_issuer
        or not isinstance(doc.get("nonce"), str)
        or len(doc["nonce"]) != 32
        or plan.get("schema") != schema
        or plan.get("status") != "plan_only"
        or plan.get("command_template_sha256") != expected_template
        or doc.get("command") != list(expected_command)
        or doc.get("policy_sha256")
        != _policy(
            expected_command,
            run_dir / "capture",
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
        )
        or doc.get("scope") != expected_scope
    ):
        raise SealedM2MError("pending M2M receipt has an unsupported policy")
    source, runtime, output = (run_dir / "snapshots/source", run_dir / "snapshots/guest-root", run_dir / "capture")
    reuse = plan.get("runtime_reuse")
    if reuse is not None:
        if plan.get("readonly_runtime_verifier_sha256") != _file_digest(Path(readonly_runtime.__file__)):
            raise SealedM2MError("selected read-only runtime verifier source changed")
        readonly_runtime.verify(plan, reuse, runtime)
    _validate_snapshots(
        source, runtime, output, schema=schema, **({"runtime_reuse": reuse} if reuse is not None else {})
    )
    source_rows, runtime_rows = _tree(source), _tree(runtime, allow_dangling_symlinks=True)
    if (_snapshot_rows(source_rows), _snapshot_rows(runtime_rows), _snapshot_tree(output)) != (
        doc.get("source"),
        doc.get("guest_root"),
        doc.get("output"),
    ):
        raise SealedM2MError("sealed M2M snapshot or capture bytes differ")
    if schema in {SCHEMA, SCHEMA_V3}:
        runtime_view = Path(reuse["entry"]) if reuse else runtime
        view_rows = readonly_runtime.effective_rows(reuse) if reuse else runtime_rows
        selected_trees = plan.get("selected_trees") or {}
        if not isinstance(selected_trees, dict):
            raise SealedM2MError("v2 plan lacks exact selected source/runtime tree identities")
        commit = plan.get("m2m_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise SealedM2MError("v2 plan lacks a pinned M2M revision identity")
        selected_roots = {
            "venv": runtime_view / "opt/capture-venv",
            "m2m": source / "m2m-src/m2m",
            "workload": source / "workload",
            "schemas": source / "merlin-src/merlin/_data/schemas",
        }
        base = plan.get("base")
        if (
            not isinstance(base, str)
            or not base.startswith("/")
            or base == "/"
            or Path(base).as_posix() != base
            or ".." in Path(base).parts
        ):
            raise SealedM2MError("selected base Python path is absent or unsafe")
        selected_roots["base"] = runtime_view / base.lstrip("/")
        if set(selected_trees) != set(selected_roots) | {"merlin"}:
            raise SealedM2MError("v2 plan lacks exact selected source/runtime tree identities")
        if plan.get("frozen_origin") is not None:
            if schema != SCHEMA:
                raise SealedM2MError("frozen M2M origin is not a v2 source selection")
            try:
                if (
                    verify_frozen_receipt(
                        plan["frozen_origin"],
                        (source / "m2m-origin.json").read_bytes(),
                        Path(plan["m2m_root"]),
                        selected_trees["m2m"],
                    )
                    != commit
                ):
                    raise SealedM2MError("frozen M2M origin revision differs from selected plan")
            except (M2MOriginError, OSError) as exc:
                raise SealedM2MError("frozen M2M origin receipt differs from replayed source") from exc
        # The external-schema policy adds the selected schema tree to the
        # Merlin package after its original selected-tree digest was taken.
        if plan.get("schemas_root") == str(Path(str(plan.get("merlin_root"))) / "_data/schemas"):
            selected_roots["merlin"] = source / "merlin-src/merlin"
        for name, path in selected_roots.items():
            root, rows = (runtime_view, view_rows) if path.is_relative_to(runtime_view) else (source, source_rows)
            if _snapshot_rows(rows, path.relative_to(root).as_posix()) != selected_trees[name]:
                raise SealedM2MError(f"selected {name} bytes differ from the v2 plan")
        if _file_digest(source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py") != plan.get("worker_sha256"):
            raise SealedM2MError("selected Merlin worker bytes differ from the v2 plan")
        if selected_trees["schemas"] != doc.get("schemas"):
            raise SealedM2MError("selected Merlin schema bytes differ from the v2 receipt")
        if schema == SCHEMA_V3:
            if not isinstance(plan.get("loader_env"), dict) or not isinstance(plan.get("selected_inputs"), list):
                raise SealedM2MError("v3 plan lacks declared environment or checkpoint inputs")
            declared, reads = _declared_loader_env((source / "workload/loader.py").read_text(), plan["loader_env"])
            if declared != plan["loader_env"] or reads != plan.get("loader_env_reads"):
                raise SealedM2MError("v3 loader environment declaration changed")
            selected_rows = plan["selected_inputs"]
            if native_integer.selected(plan.get("worker_options")):
                if selected_rows or plan["loader_env"] or reads:
                    raise SealedM2MError(
                        "native integer session requires no checkpoint, extra input or loader environment"
                    )
            elif sum(row.get("role") == "checkpoint" for row in selected_rows) != 1:
                raise SealedM2MError("v3 plan must select exactly one checkpoint")
            for row in selected_rows:
                _verify_selected_input(row, source / "inputs" / row["guest_member"])
    materialized = (
        _materialized(output, source, output)
        if schema == SCHEMA_V1
        else (
            _materialized_v3(output, source, output, plan)
            if schema == SCHEMA_V3
            else _materialized_v2(output, source, output, plan)
        )
    )
    if materialized != doc.get("materialized"):
        raise SealedM2MError("materialized M2M receipt differs")
    bwrap = _bwrap_binary(bwrap_binary)
    del source_rows, runtime_rows
    if _file_digest(bwrap) != doc.get("bwrap_sha256"):
        raise SealedM2MError("bubblewrap bytes differ from M2M receipt")

    def compare(process: dict[str, Any], replay: Path) -> None:
        if process != doc.get("process"):
            raise SealedM2MError("fresh M2M process output differs")
        _verify_replayed_output(replay, source, output, doc, schema, plan)

    if _executed is not None:
        process, replay = _executed
        if replay.parent != run_dir or not replay.name.startswith("m2m-replay-") or replay.is_symlink():
            raise SealedM2MError("concurrent M2M replay output is not this run's fresh replay directory")
        compare(process, replay)
    else:
        # The host scratch directory is mounted at the original output path.
        # Keep it outside chattr-protected captures when the operator selects TMPDIR.
        with tempfile.TemporaryDirectory(prefix="m2m-replay-", dir=os.environ.get("TMPDIR") or run_dir) as temporary:
            replay = Path(temporary)
            replay.chmod(output.stat().st_mode & 0o777)
            compare(
                _execute(
                    bwrap,
                    runtime,
                    source,
                    replay,
                    expected_command,
                    output,
                    replayable_logs=replayable_logs,
                    loader_env=loader_env,
                    timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
                    **({"runtime_reuse": reuse} if reuse is not None else {}),
                ),
                replay,
            )
    if (_snapshot_tree(source), _snapshot_tree(runtime, allow_dangling_symlinks=True), _snapshot_tree(output)) != (
        doc["source"],
        doc["guest_root"],
        doc["output"],
    ):
        raise SealedM2MError("sealed M2M evidence changed during replay")
    if reuse is not None:
        readonly_runtime.verify(plan, reuse, runtime)
    result = {
        "schema": schema,
        "status": "verified_sandbox_replay",
        "sealed_source_closure_replayed": True,
        "scope": (
            "selected FP32 CPU workload in copied empty-root Python runtime only"
            if schema == SCHEMA_V1
            else "selected CPU workload in copied empty-root Python runtime only"
        ),
        "phase0_admission": "not_granted",
        "receipt_sha256": _file_digest(receipt),
    }
    if schema in {SCHEMA, SCHEMA_V3}:
        result["capture_dtype"] = plan["dtype"]
    return result
