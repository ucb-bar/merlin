"""One bounded, independently replayable CPU Model2MLIR capture.

This is a *process and byte-closure* proof for a selected CPU workload, not a
general Python purity theorem and not Phase 0 admission.  The guest has an
empty filesystem root apart from copied runtime/source, private devices and
temporary/output mounts; it has no network or host checkout mount.  Reviewers
must separately qualify the workload, framework numerics and Phase 0 policy.
"""

from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import strict_json

from .m2m_origin import M2MOriginError, git_origin, verify_frozen_receipt, verify_frozen_selector
from .precision_staging import STAGING_API_REQUIREMENTS, output_staging_error
from .python_preflight import _loader_env_reads
from .sealed_python import _FLAGS, _TIMEOUT_SECONDS
from .sealed_static import _bwrap_binary, _canonical_path, _digest, _file_digest, _json, _tree

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
_V1_SCOPE = "isolated selected FP32 CPU M2M capture; no Phase 0 admission"
_V2_SCOPE = "isolated selected CPU M2M capture; no Phase 0 admission"
_V3_SCOPE = "isolated selected CPU M2M capture with declared model inputs; no Phase 0 admission"
_MAX_SNAPSHOT_BYTES = 16_000_000_000
_MAX_FULL_CAPTURE_SECONDS = 43_200
#: Upper bound for an operator-selected timeout on a checkpoint-free (v2) capture. Absent, a v2
#: capture keeps the historical fixed timeout and its plan bytes stay unchanged.
_MAX_SELECTED_CAPTURE_SECONDS = 14_400


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
    if not isinstance(options, dict) or set(options) - {"agreement_tolerance", "stage_fp32"}:
        raise SealedM2MError("sealed capture worker options are unsupported")
    if "stage_fp32" in options and type(options["stage_fp32"]) is not bool:
        raise SealedM2MError("sealed capture stage_fp32 must be a boolean")
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
    if not ((dtype in _FLOAT_DTYPES and not recipe) or (dtype == "int8" and recipe)):
        raise SealedM2MError("CPU capture requires fp32 without a recipe or int8 with a selected recipe")
    options = _worker_options(options)
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
    if options.get("agreement_tolerance") is not None:
        atol, rtol = options["agreement_tolerance"]
        argv += ["--agreement-atol", repr(float(atol)), "--agreement-rtol", repr(float(rtol))]
    if options.get("stage_fp32"):
        argv.append("--stage-fp32")
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
            }
        )
    )


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
) -> dict[str, Any]:
    setenv = [
        item
        for name, value in _guest_env(output_mount, replayable_logs=replayable_logs, loader_env=loader_env).items()
        for item in ("--setenv", name, value)
    ]
    argv = [
        str(bwrap),
        *_FLAGS,
        *setenv,
        "--ro-bind",
        str(runtime),
        "/",
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
    try:
        result = subprocess.run(
            argv, env={}, cwd="/", stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout_seconds
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealedM2MError(f"sandbox execution failed: {type(exc).__name__}") from exc
    if result.returncode:
        raise SealedM2MError(
            f"sandboxed M2M exited {result.returncode}: {result.stderr.decode('utf-8', errors='replace')[-8000:]}"
        )
    return {
        "returncode": 0,
        "stdout": {"bytes": len(result.stdout), "sha256": _digest(result.stdout)},
        "stderr": {"bytes": len(result.stderr), "sha256": _digest(result.stderr)},
    }


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
    """Check the selected source API without importing its heavyweight runtime.

    This is only a compatibility gate. The fresh sandbox execution and replay,
    not source signatures, establish whether a selected implementation works.
    """
    return _source_api_missing(
        m2m_root,
        {
            "m2m/api.py": {
                "convert": {"backend", "quantization", "quantization_preapplied", "level", "func_name", "weights_path"}
            },
            "m2m/capture/bundle.py": {"write_bundle": {"source_path", "capture_trace", "conversion_result"}},
            "m2m/capture/provenance.py": {"write_capture_receipt": {"source_path"}},
        },
    )


def _frontend_trace_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Report the exact optional APIs needed for frontend-op and precision evidence."""
    return _source_api_missing(
        m2m_root,
        {
            "m2m/api.py": {"convert": {"capture_trace", "original_frontend_snapshot"}},
            "m2m/capture/trace.py": {
                "capture_frontend_snapshot": {"stage"},
                "materialize_frontend_precision": {"dtype", "original_frontend_snapshot"},
            },
        },
    )


def _fp32_stage_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Require the opt-in typed-retarget API before a sealed stage is selected."""
    return _source_api_missing(m2m_root, STAGING_API_REQUIREMENTS)


def _static_integer_reference_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Report APIs needed before a static W8A8 capture can claim integer arithmetic."""
    return _source_api_missing(
        m2m_root,
        {
            "m2m/capture/pt2e_integerize.py": {"integerize_pt2e": set()},
            "m2m/capture/pt2e_integer_reference.py": {"run_pt2e_integer_reference": set()},
        },
    )


def _source_api_missing(m2m_root: Path, required: dict[str, dict[str, set[str]]]) -> tuple[str, ...]:
    """Inspect selected source signatures only; neither execution nor provenance proof."""
    missing: list[str] = []
    for member, functions in required.items():
        source = m2m_root / member
        if not source.is_file() or source.is_symlink():
            missing.append(member)
            continue
        try:
            module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError):
            missing.append(f"{member}: readable Python source")
            continue
        top_level = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        for function, parameters in functions.items():
            node = top_level.get(function)
            if node is None:
                missing.append(f"{member}:{function}")
                continue
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            declared = {argument.arg for argument in arguments}
            missing.extend(f"{member}:{function}({name})" for name in sorted(parameters - declared))
    return tuple(missing)


def _source_tree(
    root: Path,
    *,
    skip_lib64: bool = False,
    skip_python_cache: bool = False,
    readonly_modes: bool = False,
) -> dict[str, Any]:
    """Digest normalized file bytes, names and modes without retaining a huge manifest.

    The selected Python package may exclude validated transient bytecode caches;
    the venv may skip its known directory alias. File symlinks are
    dereferenced by copytree and thus by this inventory; outside targets are
    permitted only for the three CPython executable aliases, whose selected
    base interpreter is independently snapshotted. The opt-in readonly view
    removes only write bits to predict the exact mode transformation at freeze.
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
            if (here / name).is_symlink():
                raise SealedM2MError(f"directory link is outside the supported snapshot policy: {here / name}")
        directory_mode = stat.S_IMODE(here.stat().st_mode)
        records.append(
            (
                relative,
                *sorted(
                    {
                        "kind": "directory",
                        "mode": directory_mode & ~0o222 if readonly_modes else directory_mode,
                        "members": sorted([*directories, *files]),
                    }.items()
                ),
            )
        )
        for name in sorted(files):
            path = here / name
            member = path.relative_to(root).as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if not resolved.is_file():
                    raise SealedM2MError(f"non-file link in source: {member}")
                if not resolved.is_relative_to(root) and not (
                    skip_lib64 and member in {"bin/python", "bin/python3", "bin/python3.12"}
                ):
                    raise SealedM2MError(f"external source link: {member}")
            if not path.is_file():
                raise SealedM2MError(f"non-regular source member: {member}")
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


def _snapshot_tree(root: Path) -> dict[str, Any]:
    # _tree rejects every symlink and checks file identity during hashing.
    rows = _tree(root)
    compact = [(name, *sorted(info.items())) for name, info in rows.items()]
    return {
        "members": len(rows),
        "bytes": sum(row.get("bytes", 0) for row in rows.values()),
        "sha256": _digest(_json(compact)),
    }


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
    protected = {"USER", "LOGNAME", "XDG_CACHE_HOME", "TORCH_CPP_LOG_LEVEL", "HOME", "PATH"}
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
    selected_recipe = _recipe_selection(recipe, dtype=dtype)
    version = SCHEMA_V3 if checkpoint is not None or extra_inputs is not None or loader_env is not None else SCHEMA
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
        if not any(row["role"] == "checkpoint" for row in input_rows):
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
    # Inventory *before* copying: `du` does not dereference a directory link,
    # whereas copytree does.  These exact normalized bytes and topology are
    # part of the plan and are rechecked at issuance, before any 9 GB copy.
    selected_trees = {
        "venv": _source_tree(venv, skip_lib64=True),
        "base": _source_tree(base.resolve()),
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
    return {
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
    }


def _validate_snapshots(source: Path, runtime: Path, output_mount: Path, *, schema: str = SCHEMA_V1) -> None:
    for name in ("source", "capture-out", "tmp", "dev"):
        path = runtime / name
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise SealedM2MError(f"guest mount point /{name} is not empty")
    executable = runtime / "opt/capture-venv/bin/python"
    if executable.is_symlink() or not executable.is_file() or executable.read_bytes()[:4] != b"\x7fELF":
        raise SealedM2MError("snapshotted interpreter is not a direct ELF")
    worker = (
        source / "worker.py" if schema == SCHEMA_V1 else source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    )
    if not worker.is_file() or not (source / "workload/loader.py").is_file():
        raise SealedM2MError("selected source entrypoints are absent")
    if schema in {SCHEMA, SCHEMA_V3} and any(
        not (source / "merlin-src/merlin/_data/schemas" / name).is_file()
        for name in ("quant_formats.registry.yaml", "quant_format.schema.yaml")
    ):
        raise SealedM2MError("selected Merlin package data is absent from the snapshot")
    mounted = runtime / output_mount.relative_to("/")
    if mounted.is_symlink() or not mounted.is_dir() or any(mounted.iterdir()):
        raise SealedM2MError("host-resolvable guest output mount point is not empty")


def _materialized(
    output: Path, source: Path, output_mount: Path, *, worker_member: str = "worker.py"
) -> dict[str, Any]:
    from merlin.targetgen.application_inventory import verify_capture_receipt

    result = verify_capture_receipt(output / "model.mlir")
    if result.get("status") != "verified_materialized":
        raise SealedM2MError(f"M2M materialized receipt is unverified: {result.get('errors')}")
    mlir = (output / "model.mlir").read_text()
    pointer = "prov.weights_file = " + json.dumps(str(output_mount / "weights.safetensors"))
    if mlir.count("prov.weights_file") != 1 or pointer not in mlir:
        raise SealedM2MError("saved MLIR does not identify host-resolvable, receipt-bound weights")
    payload = strict_json.loads((output / "capture_receipt.json").read_bytes())
    loader = payload.get("source") or {}
    entry = (payload.get("tool") or {}).get("executed_entrypoint") or {}
    worker = source / worker_member
    if (
        loader.get("path") != "/source/workload/loader.py"
        or loader.get("sha256") != _file_digest(source / "workload/loader.py")
        or entry.get("path") != "/source/" + worker_member
        or entry.get("sha256") != _file_digest(worker)
    ):
        raise SealedM2MError("M2M receipt does not bind the snapshotted entrypoints")
    sources = (payload.get("tool") or {}).get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or (payload.get("tool") or {}).get("source_inventory_status") != "complete"
    ):
        raise SealedM2MError("M2M receipt lacks its complete direct source inventory")
    for name, digest in sources.items():
        if (
            not isinstance(name, str)
            or not name.startswith("m2m/")
            or PurePosixPath(name).as_posix() != name
            or ".." in PurePosixPath(name).parts
            or "\\" in name
            or "\x00" in name
        ):
            raise SealedM2MError(f"unsafe M2M source member: {name!r}")
        path = source / "m2m-src" / name
        if not path.is_file() or _file_digest(path) != digest:
            raise SealedM2MError(f"M2M source digest differs: {name}")
    return {"status": result["status"], "receipt_sha256": result["receipt_sha256"]}


def _materialized_v2(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    result = _materialized(
        output, source, output_mount, worker_member="merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    )
    metadata = strict_json.loads((output / "meta.json").read_bytes())
    if not isinstance(metadata, dict) or metadata.get("dtype") != plan.get("dtype"):
        raise SealedM2MError("capture metadata does not identify the selected dtype")
    stage_reason = output_staging_error(
        output,
        metadata,
        selected=_worker_options(plan.get("worker_options")).get("stage_fp32", False),
        recipe=plan.get("recipe") is not None,
    )
    if stage_reason is not None:
        raise SealedM2MError(stage_reason)
    recipe = plan.get("recipe")
    if plan.get("dtype") in _FLOAT_DTYPES:
        if recipe is not None or metadata.get("recipe_sha256") is not None:
            raise SealedM2MError("fp32 capture unexpectedly selected a quantization recipe")
    elif plan.get("dtype") == "int8":
        selected = source / "inputs/quant_recipe.json"
        if not isinstance(recipe, dict) or not selected.is_file():
            raise SealedM2MError("int8 capture did not retain the selected recipe bytes")
        observed = _recipe_selection(selected, dtype="int8")
        if any(observed[key] != recipe.get(key) for key in ("bytes", "sha256", "recipe_sha256")):
            raise SealedM2MError("int8 recipe snapshot differs from the selected plan")
        stats = metadata.get("quantization_stats") or {}
        agreement = (metadata.get("integerization_receipt") or {}).get("golden_agreement") or {}
        if (
            metadata.get("scheme") != "int8_static_act_int8_weight"
            or metadata.get("recipe_sha256") != recipe.get("recipe_sha256")
            or stats.get("recipe_sha256") != recipe.get("recipe_sha256")
            or agreement.get("status") != "passed"
            or agreement.get("reference") != "pt2e_integer"
            or metadata.get("software_numerical_engine") not in (None, "integer_reference")
        ):
            raise SealedM2MError("int8 capture lacks selected recipe and independent integer-reference agreement")
    else:
        raise SealedM2MError("unsupported selected capture dtype")
    return result


def _materialized_v3(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Verify the complete saved ABI, including every declared session program."""
    from merlin.capture.integerization import contraction_partition

    def integer_partition(meta: dict) -> tuple[int, dict[str, int]]:
        if not isinstance(meta, dict):
            raise SealedM2MError("capture lacks verified integer contractions: invalid metadata")
        receipt = meta.get("integerization_receipt") or {}
        try:
            partition = contraction_partition(receipt)
        except ValueError as exc:
            raise SealedM2MError(f"capture lacks verified integer contractions: {exc}") from exc
        agreement = receipt.get("golden_agreement") or {}
        stats, recipe = meta.get("quantization_stats"), meta.get("recipe")
        if not all(isinstance(value, dict) for value in (agreement, stats, recipe)):
            raise SealedM2MError("capture lacks verified integer contractions: malformed numerical evidence")
        executed = agreement.get("executed_contractions") or {}
        outputs = agreement.get("outputs")
        numerical_rows = [agreement, *outputs] if isinstance(outputs, list) and outputs else []
        by_kind = receipt["quantized_by_kind"]
        count = receipt.get("exported_integer_mm_count")
        emitted = receipt.get("integer_mm_emitted")
        max_k = receipt.get("max_reduction_k")
        annotated = stats.get("annotated_contractions")
        engine = recipe.get("software_numerical_engine")
        if (
            engine != "integer_reference"
            or type(annotated) is not int
            or annotated != partition["seen"]
            or agreement.get("status") != "passed"
            or agreement.get("reference") != "pt2e_integer"
            or type(agreement.get("samples")) is not int
            or agreement["samples"] < 1
            or not numerical_rows
            or any(
                not isinstance(result, dict)
                or result.get("finite") is not True
                or (result is not agreement and result.get("within_tolerance") is not True)
                or any(
                    type(result.get(key)) not in (int, float) or result[key] != 0.0
                    for key in ("atol", "rtol", "max_abs", "max_rel")
                )
                for result in numerical_rows
            )
            or not isinstance(executed, dict)
            or any(
                type(executed.get(key)) is not int
                for key in ("total", "selected", "observed", "linear", "conv2d", "matmul")
            )
            or executed.get("total") != partition["integerized"]
            or executed.get("selected") != partition["seen"]
            or executed.get("observed") != partition["seen"]
            or any(executed.get(kind) != by_kind[kind]["integerized"] for kind in ("linear", "conv2d", "matmul"))
            or type(count) is not int
            or count <= 0
            or type(emitted) is not int
            or emitted < partition["integerized"]
            or emitted != count
            or receipt.get("accumulator_bound_checked") is not True
            or type(max_k) is not int
            or max_k < 1
            or max_k * 128 * 128 > (1 << 31) - 1
        ):
            raise SealedM2MError(
                "capture lacks independently verified integer contractions or complete precision census"
            )
        return count, partition

    report_path = output / "session-receipt.json"
    contract_path = output / "session_contract.yaml"
    if not report_path.exists():
        if report_path.is_symlink():
            raise SealedM2MError("single-program capture has an indirect session receipt")
        if (output / "stages").exists() or (output / "stages").is_symlink():
            raise SealedM2MError("single-program capture has an undeclared stages directory")
        single_contract = None
        if contract_path.exists() or contract_path.is_symlink():
            if contract_path.is_symlink() or not contract_path.is_file():
                raise SealedM2MError("single-program session contract is indirect or absent")
            try:
                import yaml

                contract = yaml.safe_load(contract_path.read_bytes())
            except (OSError, ValueError, yaml.YAMLError) as exc:
                raise SealedM2MError("single-program session contract is unreadable") from exc
            stages = contract.get("stages") if isinstance(contract, dict) else None
            schedule = contract.get("stage_schedule") if isinstance(contract, dict) else None
            if (
                not isinstance(contract, dict)
                or contract.get("version") != 1
                or not isinstance(stages, list)
                or len(stages) != 1
                or not isinstance(stages[0], str)
                or PurePosixPath(stages[0]).name != stages[0]
                or stages[0] in {"", ".", ".."}
                or "\\" in stages[0]
                or "\x00" in stages[0]
                or not isinstance(schedule, list)
                or len(schedule) != 1
                or not isinstance(schedule[0], dict)
                or schedule[0].get("name") != stages[0]
                or any(
                    contract.get(label) is not None and not isinstance(contract[label], dict)
                    for label in ("correctness", "quality")
                )
            ):
                raise SealedM2MError("single-program session contract has an invalid stage roster")
            for member in (
                contract.get("inputs"),
                (contract.get("correctness") or {}).get("golden"),
                (contract.get("quality") or {}).get("golden"),
            ):
                if member is None:
                    continue
                if (
                    not isinstance(member, str)
                    or not member
                    or PurePosixPath(member).is_absolute()
                    or PurePosixPath(member).as_posix() != member
                    or ".." in PurePosixPath(member).parts
                    or "\\" in member
                    or "\x00" in member
                    or (output / member).is_symlink()
                    or not (output / member).is_file()
                ):
                    raise SealedM2MError("single-program session contract references an absent or unsafe file")
            single_contract = _file_digest(contract_path)
        verified = _materialized_v2(output, source, output_mount, plan)
        integer_work = 0
        if plan["dtype"] == "int8":
            meta = strict_json.loads((output / "meta.json").read_bytes())
            exported, partition = integer_partition(meta)
            # Preserve the exact historical sealed-v3 materialized summary:
            # this legacy field counts exported integer operations. The new
            # ledger distinguishes source contractions only for mixed captures,
            # which the old verifier never admitted.
            integer_work = exported
        return {
            "kind": "single",
            **verified,
            "integer_contractions": integer_work,
            **(
                {"exported_integer_operations": exported, "contraction_partition": partition}
                if plan["dtype"] == "int8" and partition["preserved"]
                else {}
            ),
            **({"session_contract_sha256": single_contract} if single_contract is not None else {}),
        }
    quantized = plan["dtype"] == "int8"
    if quantized:
        recipe = plan.get("recipe")
        selected_recipe = source / "inputs/quant_recipe.json"
        if not isinstance(recipe, dict) or not selected_recipe.is_file():
            raise SealedM2MError("multi-program int8 capture lacks the selected recipe")
        observed_recipe = _recipe_selection(selected_recipe, dtype="int8")
        if any(observed_recipe[key] != recipe.get(key) for key in ("bytes", "sha256", "recipe_sha256")):
            raise SealedM2MError("multi-program int8 recipe snapshot differs from the selected plan")
    elif plan["dtype"] not in _FLOAT_DTYPES or plan.get("recipe") is not None:
        raise SealedM2MError("multi-program capture has an unsupported dtype or recipe")
    if any(path.is_symlink() or not path.is_file() for path in (report_path, contract_path)):
        raise SealedM2MError("multi-program session receipt or contract is absent or indirect")
    try:
        import yaml

        report = strict_json.loads(report_path.read_bytes())
        contract = yaml.safe_load(contract_path.read_bytes())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SealedM2MError("multi-program session evidence is unreadable") from exc
    if not isinstance(report, dict) or not isinstance(contract, dict):
        raise SealedM2MError("multi-program session evidence is malformed")
    selected_stage = _worker_options(plan.get("worker_options")).get("stage_fp32", False)
    if selected_stage and report.get("source_state_unchanged") is not True:
        raise SealedM2MError("FP32 staging lacks unchanged shared source state")
    names = contract.get("stages")
    programs = contract.get("programs")
    rows = report.get("programs")
    if (
        contract.get("version") != 2
        or report.get("schema") != "merlin.model_session_capture.v1"
        or report.get("session_contract_sha256") != _file_digest(contract_path)
        or report.get("recipe_sha256") != (plan["recipe"]["recipe_sha256"] if quantized else None)
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
        or not isinstance(programs, list)
        or not isinstance(rows, list)
        or len(programs) != len(names)
        or len(rows) != len(names)
        or any(not isinstance(row, dict) for row in [*programs, *rows])
    ):
        raise SealedM2MError("multi-program session roster or contract digest differs")
    for index, name in enumerate(names):
        if (
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).name != name
            or name in {".", ".."}
            or "\\" in name
            or "\x00" in name
            or programs[index].get("name") != name
            or programs[index].get("bundle") != f"stages/{name}"
            or rows[index].get("name") != name
        ):
            raise SealedM2MError("multi-program session contains an unsafe or mismatched program")
    stages_root = output / "stages"
    if (
        stages_root.is_symlink()
        or not stages_root.is_dir()
        or sorted(path.name for path in stages_root.iterdir()) != sorted(names)
    ):
        raise SealedM2MError("multi-program stage directory membership differs from the contract")
    verified = []
    integer_work = 0
    for name, row in zip(names, rows):
        stage = stages_root / name
        if stage.is_symlink() or not stage.is_dir():
            raise SealedM2MError("multi-program stage is indirect or absent")
        try:
            meta = strict_json.loads((stage / "meta.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise SealedM2MError("multi-program stage metadata is unreadable") from exc
        if not isinstance(meta, dict):
            raise SealedM2MError("multi-program stage metadata is malformed")
        stage_reason = output_staging_error(
            stage,
            meta,
            selected=selected_stage,
            recipe=quantized and row.get("precision_selection") == "recipe",
        )
        if stage_reason is not None:
            raise SealedM2MError(stage_reason)
        mode = row.get("precision_selection")
        if quantized and mode == "recipe":
            count, partition = integer_partition(meta)
            if (
                meta.get("dtype") != "int8"
                or meta.get("scheme") != "int8_static_act_int8_weight"
                or meta.get("recipe_sha256") != recipe["recipe_sha256"]
                or (meta.get("quantization_stats") or {}).get("recipe_sha256") != recipe["recipe_sha256"]
            ):
                raise SealedM2MError("multi-program recipe stage lacks verified integer contractions")
            integer_work += count
        elif quantized and mode == "no_recipe_work":
            if (
                meta.get("dtype") not in _FLOAT_DTYPES
                or meta.get("recipe_sha256") is not None
                or meta.get("integerization_receipt") is not None
            ):
                raise SealedM2MError("multi-program host-only stage unexpectedly claims recipe precision")
        elif not quantized and mode == "untransformed":
            if meta.get("dtype") not in _FLOAT_DTYPES or meta.get("recipe_sha256") is not None:
                raise SealedM2MError("multi-program FP32 stage unexpectedly claims recipe precision")
        else:
            raise SealedM2MError("multi-program stage has an unsupported precision selection")
        result = _materialized(
            stage,
            source,
            output_mount / "stages" / name,
            worker_member="merlin-src/merlin/targetgen/_m2m_capture_worker.py",
        )
        stage_receipt = strict_json.loads((stage / "capture_receipt.json").read_bytes())
        if (
            row.get("receipt_sha256") != result["receipt_sha256"]
            or (row.get("materialized_abi") or {}).get("complete") is not True
            or row.get("materialized_abi") != stage_receipt.get("materialized_abi")
        ):
            raise SealedM2MError("multi-program stage receipt differs from the session receipt")
        verified.append(
            {
                "name": name,
                "model_sha256": _file_digest(stage / "model.mlir"),
                "receipt_sha256": result["receipt_sha256"],
                "precision_selection": mode,
                **(
                    {"contraction_partition": partition, "exported_integer_operations": count}
                    if quantized and mode == "recipe" and partition["preserved"]
                    else {}
                ),
            }
        )
    if quantized and integer_work == 0:
        raise SealedM2MError("multi-program int8 capture contains no verified integer contractions")
    return {
        "kind": "session",
        "session_contract_sha256": _file_digest(contract_path),
        "session_receipt_sha256": _file_digest(report_path),
        "programs": verified,
        "integer_contractions": integer_work,
    }


def _stage_source(plan: dict[str, Any], source: Path) -> None:
    """Copy exactly the source members named by the selected v2 plan."""
    shutil.copytree(
        Path(plan["m2m_root"]) / "m2m",
        source / "m2m-src/m2m",
        symlinks=False,
        ignore=lambda _directory, names: {name for name in names if name == "__pycache__"},
    )
    if plan.get("frozen_origin") is not None:
        shutil.copy2(plan["frozen_origin"]["path"], source / "m2m-origin.json")
    shutil.copytree(Path(plan["workload_root"]), source / "workload", symlinks=False)
    shutil.copytree(
        Path(plan["merlin_root"]),
        source / "merlin-src/merlin",
        symlinks=False,
        ignore=lambda _directory, names: {name for name in names if name == "__pycache__"},
    )
    if _snapshot_tree(source / "merlin-src/merlin") != plan["selected_trees"]["merlin"]:
        raise SealedM2MError("Merlin package snapshot differs from selected bytes")
    bundled_schemas = source / "merlin-src/merlin/_data/schemas"
    if not bundled_schemas.exists():
        shutil.copytree(Path(plan["schemas_root"]), bundled_schemas, symlinks=False)
    if plan.get("recipe"):
        selected_recipe = source / "inputs/quant_recipe.json"
        selected_recipe.parent.mkdir()
        shutil.copy2(plan["recipe"]["path"], selected_recipe)
    if plan.get("schema") == SCHEMA_V3:
        for row in plan["selected_inputs"]:
            destination = source / "inputs" / row["guest_member"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if row["kind"] == "file":
                shutil.copy2(row["source"], destination)
            else:
                shutil.copytree(row["source"], destination, symlinks=False)


def _verify_staged_selection(plan: dict[str, Any], source: Path, runtime: Path) -> dict[str, Any]:
    """Reject bytes copied after selection changed, before executing any guest code."""
    roots = {
        "venv": runtime / "opt/capture-venv",
        "base": runtime / Path(plan["base"]).relative_to("/"),
        "m2m": source / "m2m-src/m2m",
        "workload": source / "workload",
        "schemas": source / "merlin-src/merlin/_data/schemas",
    }
    selected_trees = plan["selected_trees"]
    if plan.get("frozen_origin") is not None:
        try:
            origin_commit = verify_frozen_receipt(
                plan["frozen_origin"],
                (source / "m2m-origin.json").read_bytes(),
                Path(plan["m2m_root"]),
                selected_trees["m2m"],
            )
        except (M2MOriginError, OSError) as exc:
            raise SealedM2MError("staged frozen M2M origin differs from selection") from exc
        if origin_commit != plan["m2m_commit"]:
            raise SealedM2MError("staged frozen M2M origin revision differs from selected plan")
    for name, path in roots.items():
        if _snapshot_tree(path) != selected_trees[name]:
            raise SealedM2MError(f"staged {name} bytes differ from the pre-execution selection")
    if plan.get("schema") == SCHEMA_V3:
        for row in plan["selected_inputs"]:
            _verify_selected_input(row, source / "inputs" / row["guest_member"])
    # External schemas are inserted after the original Merlin package snapshot.
    if plan["schemas_root"] == str(Path(plan["merlin_root"]) / "_data/schemas"):
        if _snapshot_tree(source / "merlin-src/merlin") != selected_trees["merlin"]:
            raise SealedM2MError("staged Merlin bytes differ from the pre-execution selection")
    return selected_trees["schemas"]


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
        **(
            {
                "checkpoint": next(
                    Path(row["source"]) for row in plan["selected_inputs"] if row["role"] == "checkpoint"
                ),
                "checkpoint_guest_member": next(
                    row["guest_member"] for row in plan["selected_inputs"] if row["role"] == "checkpoint"
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

    required_bytes, verified_cache = snapshot_space_requirement(plan, run_dir.parent, selected_system_libraries)
    if required_bytes > shutil.disk_usage(run_dir.parent).free:
        raise SealedM2MError("normalized snapshot bytes exceed selected run filesystem free space")
    inputs = [
        Path(plan[key])
        for key in ("m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base")
    ]
    if plan.get("recipe"):
        inputs.append(Path(plan["recipe"]["path"]))
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

    link_runtime(plan, runtime, verified_cache=verified_cache, selected_system_libraries=selected_system_libraries)
    output = run_dir / "capture"
    for name in ("source", "capture-out", "tmp", "dev"):
        (runtime / name).mkdir()
    (runtime / output.relative_to("/")).mkdir(parents=True)
    _validate_snapshots(source, runtime, output, schema=plan["schema"])
    # Rehash the copied bytes against the antecedent selection, not mutable
    # host paths: a source change after prepare_plan must never be executed.
    selected_schemas = _verify_staged_selection(plan, source, runtime)
    if _file_digest(source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py") != plan["worker_sha256"]:
        raise SealedM2MError("worker snapshot differs from selected bytes")
    if plan.get("recipe") and _file_digest(source / "inputs/quant_recipe.json") != plan["recipe"]["sha256"]:
        raise SealedM2MError("selected recipe snapshot differs from selected bytes")
    for index, name in enumerate(plan["system_libs"]):
        copied = runtime / Path(name).relative_to("/")
        expected = selected_system_libraries[index] if selected_system_libraries is not None else None
        if expected is not None:
            if copied.stat().st_size != expected.get("bytes") or _file_digest(copied) != expected.get("sha256"):
                raise SealedM2MError(f"system ELF snapshot differs from the pre-execution selection: {name}")
        elif _file_digest(copied) != _file_digest(Path(name)):
            raise SealedM2MError(f"system ELF snapshot differs from selected bytes: {name}")
    source_digest = _snapshot_tree(source)
    runtime_digest = _snapshot_tree(runtime)
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
    )
    materialized = (
        _materialized_v3(output, source, output, plan)
        if plan["schema"] == SCHEMA_V3
        else _materialized_v2(output, source, output, plan)
    )
    if (_snapshot_tree(source), _snapshot_tree(runtime)) != (source_digest, runtime_digest):
        raise SealedM2MError("sealed source or runtime changed during capture")
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


def replay_verify(run_dir: Path, *, bwrap_binary: Path | None = None) -> dict[str, Any]:
    """Independent replay from exact private bytes; no Phase 0 status is granted."""
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
        if dtype not in _FLOAT_DTYPES | {"int8"} or (recipe is not None) != (dtype == "int8"):
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
    if schema in {SCHEMA, SCHEMA_V3} and "frozen_origin" not in plan:
        supported_issuer.add(_PRE_FROZEN_ORIGIN_ISSUER_SHA256)
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
        )
        or doc.get("scope") != expected_scope
    ):
        raise SealedM2MError("pending M2M receipt has an unsupported policy")
    source, runtime, output = (run_dir / "snapshots/source", run_dir / "snapshots/guest-root", run_dir / "capture")
    _validate_snapshots(source, runtime, output, schema=schema)
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc.get("source"),
        doc.get("guest_root"),
        doc.get("output"),
    ):
        raise SealedM2MError("sealed M2M snapshot or capture bytes differ")
    if schema in {SCHEMA, SCHEMA_V3}:
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
            "venv": runtime / "opt/capture-venv",
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
        selected_roots["base"] = runtime / base.lstrip("/")
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
            if _snapshot_tree(path) != selected_trees[name]:
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
            if sum(row.get("role") == "checkpoint" for row in selected_rows) != 1:
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
    if _file_digest(bwrap) != doc.get("bwrap_sha256"):
        raise SealedM2MError("bubblewrap bytes differ from M2M receipt")
    with tempfile.TemporaryDirectory(prefix="m2m-replay-", dir=run_dir) as temporary:
        replay = Path(temporary)
        replay.chmod(output.stat().st_mode & 0o777)
        if _execute(
            bwrap,
            runtime,
            source,
            replay,
            expected_command,
            output,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
        ) != doc.get("process"):
            raise SealedM2MError("fresh M2M process output differs")
        replay_materialized = (
            _materialized(replay, source, output)
            if schema == SCHEMA_V1
            else (
                _materialized_v3(replay, source, output, plan)
                if schema == SCHEMA_V3
                else _materialized_v2(replay, source, output, plan)
            )
        )
        if replay_materialized != doc.get("materialized") or _snapshot_tree(replay) != doc["output"]:
            raise SealedM2MError("fresh M2M materialized bytes differ")
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc["source"],
        doc["guest_root"],
        doc["output"],
    ):
        raise SealedM2MError("sealed M2M evidence changed during replay")
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
