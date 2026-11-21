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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    return _snapshot_rows(_tree(root))


def _declared_loader_env(
    source: str, selected: dict[str, str | None] | None
) -> tuple[dict[str, str | None], list[str]]:
    """Bind literal loader reads, including intentionally absent values, before execution.

    The guest starts with an empty host environment. Dynamic loader reads cannot
    be selected exactly and are therefore outside this policy.
    """
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

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
    from merlin.experiments.capture_execution.m2m_sandbox import (
        _MAX_FULL_CAPTURE_SECONDS,
        _MAX_SELECTED_CAPTURE_SECONDS,
        _MAX_SNAPSHOT_BYTES,
        SCHEMA,
        SCHEMA_V3,
        SealedM2MError,
        _capture_api_missing,
        _command_v2,
        _fp32_stage_api_missing,
        _recipe_selection,
        _selected_timeout_valid,
        _worker_member,
        _worker_options,
    )

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
    merlin_root = worker.parents[2] if worker.name == "worker.py" else worker.parents[1]
    worker_member = _worker_member({"worker": str(worker), "merlin_root": str(merlin_root)})
    if not (merlin_root / "__init__.py").is_file():
        raise SealedM2MError("selected worker must belong to the selected Merlin source package")
    if schemas_root not in {
        merlin_root / "resources/schemas",
        merlin_root / "_data/schemas",
        merlin_root.parent.parent / "src/merlin/resources/schemas",
    }:
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
        _command_v2(
            Path("/capture-out"),
            dtype=dtype,
            recipe=selected_recipe is not None,
            options=options,
            worker_member=worker_member,
        )
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
            _command_v2(
                Path("/capture-out"),
                dtype=dtype,
                recipe=selected_recipe is not None,
                options=options,
                worker_member=worker_member,
            )[-1].encode()
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
    from merlin.experiments.capture_execution.m2m_sandbox import SCHEMA_V1, _validation_functions

    return _validation_functions()._validate_snapshots(
        source,
        runtime,
        output_mount,
        schema=schema,
        runtime_reuse=runtime_reuse,
        **({"worker_member": worker_member} if worker_member is not None else {}),
    )


def _materialized(
    output: Path, source: Path, output_mount: Path, *, worker_member: str = "worker.py"
) -> dict[str, Any]:
    """Delegate to the source-pinned capture validation implementation."""
    from merlin.experiments.capture_execution.m2m_sandbox import _validation_functions

    return _validation_functions()._materialized(output, source, output_mount, worker_member=worker_member)


def _materialized_v2(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the source-pinned capture validation implementation."""
    from merlin.experiments.capture_execution.m2m_sandbox import _validation_functions

    return _validation_functions()._materialized_v2(output, source, output_mount, plan)


def _dynamic_integer_census(meta: dict, stats: dict, stage: Path, emitted: int, exported: int) -> int:
    """Delegate to the source-pinned capture validation implementation."""
    from merlin.experiments.capture_execution.m2m_sandbox import _validation_functions

    return _validation_functions()._dynamic_integer_census(meta, stats, stage, emitted, exported)


def _materialized_v3(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the source-pinned capture validation implementation."""
    from merlin.experiments.capture_execution.m2m_sandbox import _validation_functions

    return _validation_functions()._materialized_v3(output, source, output_mount, plan)


def _stage_source(plan: dict[str, Any], source: Path) -> None:
    """Delegate to the source-pinned capture validation implementation."""
    from merlin.experiments.capture_execution.m2m_sandbox import _validation_functions

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
    from merlin.experiments.capture_execution.sealed_m2m import _issue

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
    from merlin.experiments.capture_execution.m2m_sandbox import SealedM2MError

    issue_cpus, replay_cpus = (frozenset(int(cpu) for cpu in part) for part in cpus)
    if not issue_cpus or not replay_cpus or issue_cpus & replay_cpus:
        raise SealedM2MError("a concurrent replay needs two nonempty disjoint CPU sets")
    from merlin.experiments.capture_execution.sealed_m2m import _issue

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
