"""Fail-closed SSH/systemd orchestration for an already frozen K1 paper matrix.

The prepare action delegates construction to the closed production contract registry and never
executes a cell.  Plan/run/finalize never edit a measurement contract: a local plan binds complete
contract directories and the controller source closure by digest.  Execution stages those exact
bytes into content-addressed board directories, serializes the single board with systemd + flock,
and atomically retrieves each controller receipt and detached issuance root.  A locally validated
terminal cell is skipped on resume; a board-terminal cell missing locally is retrieved without
rerunning its measurement.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import platform
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import yaml

from merlin.common.artifacts import finish_run, start_run
from merlin.common.paths import artifacts_dir, repo_root

from .paper import validate_paper_result
from .paper_measurement_controller import (
    _contract,
    issuance_fingerprint,
    normalize_receipt,
    produce_receipt,
)


_PLAN_KIND = "paper_k1_frozen_contract_matrix_v2"
_PREPARED_KIND = "paper_k1_prepared_contract_matrix_v1"
_ENVIRONMENT_KIND = "paper_k1_environment_preflight_v1"
_FINALIZATION_KIND = "paper_k1_results_finalization_v1"
_TERMINAL_KIND = "paper_k1_remote_terminal_v1"
_STATE_KIND = "paper_k1_local_terminal_v1"
_NOTARY_KIND = "paper_external_issuance_notary_v1"
#: A lowercase hex sha256, as a length plus a character set. Spelled structurally because the repo
#: forbids regex in library code -- and because this predicate decides whether a frozen study and its
#: matrix are the ones a result is attributed to, which is the wrong place for a validator nobody can
#: check by eye.
_SHA256_HEX_LEN = 64
_HEX_LOWER = "0123456789abcdef"

#: Characters a rendered path segment may keep; every other character collapses to one underscore.
_SAFE_EXTRA = "_.-"
#: The prefix a systemd cell unit must carry, and the characters its suffix may use.
_UNIT_PREFIX = "merlin-paper-"
_UNIT_SUFFIX_MAX = 120

# All of these paths are invoked by the transport or trusted measurement controller.  The remote
# preflight binds their bytes rather than accepting a successful PATH lookup as evidence.
_REMOTE_TOOLS = {
    "cc": "/usr/bin/cc",
    "cut": "/usr/bin/cut",
    "env": "/usr/bin/env",
    "flock": "/usr/bin/flock",
    "install": "/usr/bin/install",
    "mv": "/usr/bin/mv",
    "openssl": "/usr/bin/openssl",
    "rm": "/usr/bin/rm",
    "sha256sum": "/usr/bin/sha256sum",
    "systemd_run": "/usr/bin/systemd-run",
    "tar": "/usr/bin/tar",
    "sh": "/bin/sh",
}
_REMOTE_MODULES = {
    "pyyaml": "yaml",
    "aet": "aet",
    "aet_run_logger": "aet.tracking.run_logger",
}
_ELF_MACHINE_RISCV = 243


def _is_sha256_hex(value: str) -> bool:
    return len(value) == _SHA256_HEX_LEN and all(c in _HEX_LOWER for c in value)


def _is_safe_unit_name(name: str) -> bool:
    """A systemd unit identity: the fixed prefix plus 1..120 name-safe characters."""
    if not name.startswith(_UNIT_PREFIX):
        return False
    tail = name[len(_UNIT_PREFIX):]
    if not (1 <= len(tail) <= _UNIT_SUFFIX_MAX):
        return False
    return all(c.isascii() and (c.isalnum() or c in _SAFE_EXTRA) for c in tail)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii"))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(value)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_tree(root: Path, *, controller_source: bool = False) -> list[dict[str, Any]]:
    unresolved = Path(root)
    if unresolved.is_symlink():
        raise ValueError(f"staged tree cannot be a symlink: {unresolved}")
    root = unresolved.resolve()
    if not root.is_dir():
        raise ValueError(f"staged tree is not a regular directory: {root}")
    rows: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in [*names, *files]:
            if (directory_path / name).is_symlink():
                raise ValueError(f"staged tree contains a symlink: {directory_path / name}")
        for name in sorted(files):
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if controller_source and ("__pycache__" in path.parts or path.suffix == ".pyc"):
                continue
            if (controller_source and path.suffix != ".py" and relative not in {
                    "compare/paper_model_abi_runner.c",
                    "compare/paper_k1_board_probe.c",
            }):
                continue
            if not path.is_file():
                raise ValueError(f"staged tree contains a non-regular file: {path}")
            stat = path.stat()
            rows.append({
                "path": relative, "sha256": _sha_file(path), "size": stat.st_size,
                "mode": stat.st_mode & 0o777,
            })
    if not rows:
        raise ValueError(f"staged tree is empty: {root}")
    return sorted(rows, key=lambda row: row["path"])


def _controller_rows(root: Path) -> list[dict[str, Any]]:
    unresolved = Path(root)
    if unresolved.is_symlink():
        raise ValueError("controller source root cannot be a symlink")
    root = unresolved.resolve()
    prefixes = ((Path("merlin/python/merlin"), True), (Path("merlin/schemas"), False))
    rows: list[dict[str, Any]] = []
    for prefix, source_filter in prefixes:
        source = root / prefix
        for row in _regular_tree(source, controller_source=source_filter):
            rows.append({**row, "path": (prefix / row["path"]).as_posix()})
    required = {
        "merlin/python/merlin/compare/paper_measurement_controller.py",
        "merlin/python/merlin/compare/paper_k1_orchestrator.py",
        "merlin/python/merlin/compare/paper_model_abi_runner.c",
        "merlin/python/merlin/compare/paper_k1_board_probe.c",
    }
    if not required <= {row["path"] for row in rows}:
        raise ValueError("controller staging closure omits required paper controller sources")
    return sorted(rows, key=lambda row: row["path"])


def _tree_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha(list(rows))


def _safe_name(value: str) -> str:
    # Collapse every run of non-name-safe characters to ONE underscore, matching the pattern this
    # replaces: a per-character map would turn "a  b" into "a__b" and change existing path names.
    out, prev_bad = [], False
    for ch in str(value):
        if ch.isascii() and (ch.isalnum() or ch in _SAFE_EXTRA):
            out.append(ch)
            prev_bad = False
        elif not prev_bad:
            out.append("_")
            prev_bad = True
    rendered = "".join(out).strip("._")
    if not rendered:
        raise ValueError("matrix identity cannot be rendered as a safe path")
    return rendered[:120]


def _elf_machine(path: Path) -> int:
    """Read ``e_machine`` without executing an authority-selected build tool."""
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF" or header[4] not in {1, 2}:
        raise ValueError(f"paper build tool is not a supported ELF: {path}")
    if header[5] == 1:
        byteorder = "little"
    elif header[5] == 2:
        byteorder = "big"
    else:
        raise ValueError(f"paper build tool has an invalid ELF byte order: {path}")
    return int.from_bytes(header[18:20], byteorder=byteorder)


def _contract_build_tool(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    build = contract.get("build")
    reference = build.get("tool") if isinstance(build, Mapping) else None
    if (not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}
            or not _is_sha256_hex(str(reference.get("sha256", "")))):
        raise ValueError("K1 measurement contract has no closed build-tool reference")
    relative = PurePosixPath(str(reference["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("K1 measurement build tool escapes its contract tree")
    path = root.joinpath(*relative.parts).resolve()
    if (not path.is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file()
            or not os.access(path, os.X_OK) or _sha_file(path) != reference["sha256"]):
        raise ValueError("K1 measurement build tool is absent, unsafe, or changed")
    machine = _elf_machine(path)
    if machine != _ELF_MACHINE_RISCV:
        raise ValueError(
            "K1 controller rebuild requires a RISC-V-native build tool; the frozen contract "
            f"selects ELF e_machine={machine}. A host SpacemiT cross-compiler cannot execute "
            "inside the board-local measurement controller")
    return {"sha256": reference["sha256"], "size": path.stat().st_size,
            "elf_machine": machine}


def _runtime_requirements(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tools = {(row["build_tool"]["sha256"], row["build_tool"]["size"],
              row["build_tool"]["elf_machine"]) for row in cells}
    required_cores = max(int(row["cell"]["core_count"]) for row in cells)
    return {
        "required_core_count": required_cores,
        "build_tools": [
            {"sha256": sha, "size": size, "elf_machine": machine}
            for sha, size, machine in sorted(tools)
        ],
    }


def prepare_contract_matrix(study_path: str | Path, *,
                            output_dir: str | Path | None = None) -> Path:
    """Materialize every frozen contract without executing a paper cell.

    The parent AET identity is allocated exactly once and becomes the prefix of every child run
    identity.  Contracts follow :func:`study.execution_matrix` verbatim; the resulting transport
    plan revalidates that order and all retained bytes before publication.
    """
    from .paper import PaperStudySpec
    from .paper_contract_registry import build_registered_contract
    from .study import _base_result, execution_matrix

    unresolved_study = Path(study_path)
    if unresolved_study.is_symlink():
        raise ValueError("frozen paper study cannot be a symlink")
    source = unresolved_study.resolve()
    if not source.is_file():
        raise ValueError("frozen paper study is absent")
    source_sha = _sha_file(source)
    spec = PaperStudySpec.from_yaml(source)
    if _sha_file(source) != source_sha:
        raise ValueError("frozen paper study changed while it was loaded")
    preflight = spec.preflight()
    if spec.target != "k1" or not preflight.ready:
        raise ValueError(
            "contract-only K1 preparation requires a preflight-ready frozen study: "
            + "; ".join([*preflight.errors, *preflight.blockers]))
    explicit = Path(output_dir) if output_dir is not None else None
    if explicit is not None and (explicit.exists() or explicit.is_symlink()):
        raise FileExistsError(f"prepared matrix output already exists: {explicit}")

    handle = start_run(
        suite="paper-study", method="frozen-contract-preparation", target="k1",
        extra={"study_sha256": spec.sha256(), "study_path": str(source),
               "contract_only": True, "n_cells": len(spec.matrix())},
    )
    destination = (explicit.resolve() if explicit is not None else handle.run_dir.resolve())
    status = "fail"
    summary: dict[str, Any] = {"prepared": False, "n_cells": 0}
    try:
        destination.mkdir(parents=True, exist_ok=explicit is None)
        retained_study = destination / "study.frozen.yaml"
        _atomic_bytes(retained_study, source.read_bytes())
        if _sha_file(retained_study) != source_sha:
            raise ValueError("retained frozen study differs from preparation input")
        _atomic_yaml(destination / "aet-parent.yaml", {
            "run_id": handle.run_id,
            "canonical_run_dir": str(handle.run_dir.resolve()),
            "run_record": str((handle.run_dir / "run_record.json").resolve()),
            "prepared_matrix_dir": str(destination),
        })
        contract_paths: list[Path] = []
        rows: list[dict[str, Any]] = []
        for index, cell in enumerate(execution_matrix(spec)):
            run_id = f"{handle.run_id}__cell{index:03d}"
            name = _safe_name(
                f"{index:03d}_{cell.model.name}_{cell.backend.name}_{cell.precision}_"
                f"{cell.core_count}c")
            root = destination / "controller-contracts" / name
            base = _base_result(spec, cell, run_id, handle.timestamp, handle.git_sha)
            contract = build_registered_contract(
                spec, cell, run_id=run_id, timestamp=handle.timestamp,
                git_sha=handle.git_sha, staging_dir=root, base_result=base)
            if contract.resolve() != (root / "measurement_contract.yaml").resolve():
                raise ValueError("registered backend did not emit the canonical K1 contract name")
            contract_paths.append(contract)
            rows.append({
                "execution_index": index, "run_id": run_id, "cell_key": cell.key,
                "contract": str(contract.resolve()), "contract_sha256": _sha_file(contract),
            })
        plan_path = create_matrix_plan(
            contract_paths, destination / "k1-matrix-plan.json", controller_root=repo_root())
        plan = _load_plan(plan_path)
        if ([row["run_id"] for row in plan["cells"]] != [row["run_id"] for row in rows]
                or [row["execution_index"] for row in plan["cells"]]
                != list(range(len(rows)))):
            raise ValueError("transport plan changed the frozen study execution order")
        _atomic_yaml(destination / "prepared-matrix.yaml", {
            "schema_version": 1, "kind": _PREPARED_KIND, "status": "complete",
            "study": {"path": str(retained_study), "file_sha256": source_sha,
                      "canonical_sha256": spec.sha256()},
            "aet_parent": {"run_id": handle.run_id, "timestamp": handle.timestamp,
                           "git_sha": handle.git_sha,
                           "run_record": str((handle.run_dir / "run_record.json").resolve()),
                           "run_record_sha256": _sha_file(handle.run_dir / "run_record.json")},
            "matrix_plan": {"path": str(plan_path), "sha256": _sha_file(plan_path),
                            "matrix_sha256": plan["matrix_sha256"]},
            "cells": rows,
        })
        status = "ok"
        summary = {"prepared": True, "n_cells": len(rows),
                   "matrix_sha256": plan["matrix_sha256"]}
        return destination
    finally:
        finish_run(handle, status, summary)


def create_matrix_plan(contract_paths: Sequence[str | Path], output_path: str | Path, *,
                       controller_root: str | Path | None = None) -> Path:
    """Freeze the exact already-materialized contract roots and controller source closure."""
    if not contract_paths:
        raise ValueError("K1 matrix plan requires at least one frozen contract")
    source_unresolved = Path(controller_root or repo_root())
    if source_unresolved.is_symlink():
        raise ValueError("controller source root cannot be a symlink")
    source_root = source_unresolved.resolve()
    controller_rows = _controller_rows(source_root)
    cells: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seen_cells: set[tuple[Any, ...]] = set()
    study_sha: str | None = None
    for index, raw_path in enumerate(contract_paths):
        contract_unresolved = Path(raw_path)
        if contract_unresolved.is_symlink():
            raise ValueError(f"measurement contract cannot be a symlink: {contract_unresolved}")
        contract_path = contract_unresolved.resolve()
        if not contract_path.is_file():
            raise ValueError(f"measurement contract is absent or a symlink: {contract_path}")
        contract, cell = _contract(contract_path)
        if contract["target"] != "k1":
            raise ValueError("K1 matrix plan accepts only target=k1 contracts")
        run_id = str(contract["run_id"])
        cell_key = (cell["model"], cell["backend"], cell["precision"], cell["core_count"])
        if run_id in seen_run_ids or cell_key in seen_cells:
            raise ValueError("K1 matrix plan contains duplicate run or cell identities")
        if study_sha is None:
            study_sha = str(contract["study_sha256"])
        elif study_sha != contract["study_sha256"]:
            raise ValueError("K1 matrix contracts do not share one frozen study identity")
        root = contract_path.parent
        if contract_path.relative_to(root).as_posix() != "measurement_contract.yaml":
            raise ValueError("frozen K1 contract must use the canonical measurement_contract.yaml")
        rows = _regular_tree(root)
        build_tool = _contract_build_tool(root, contract)
        cells.append({
            "execution_index": index, "run_id": run_id, "cell": dict(cell),
            "contract_root": str(root),
            "contract_path": contract_path.relative_to(root).as_posix(),
            "contract_sha256": _sha_file(contract_path), "tree_sha256": _tree_sha(rows),
            "tree_files": len(rows), "build_tool": build_tool,
        })
        seen_run_ids.add(run_id)
        seen_cells.add(cell_key)
    runtime_requirements = _runtime_requirements(cells)
    identity = {
        "study_sha256": study_sha,
        "controller_tree_sha256": _tree_sha(controller_rows),
        "runtime_requirements": runtime_requirements,
        "cells": [{key: row[key] for key in (
            "execution_index", "run_id", "cell", "contract_sha256", "tree_sha256",
            "build_tool")}
                  for row in cells],
    }
    plan = {
        "schema_version": 2, "kind": _PLAN_KIND, "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_sha256": study_sha, "matrix_sha256": _canonical_sha(identity),
        "runtime_requirements": runtime_requirements,
        "controller": {
            "root": str(source_root), "tree_sha256": identity["controller_tree_sha256"],
            "tree_files": len(controller_rows),
        },
        "cells": cells,
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    _atomic_json(output, plan)
    return output


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
            "schema_version", "kind", "status", "created_at_utc", "study_sha256",
            "matrix_sha256", "runtime_requirements", "controller", "cells"}:
        raise ValueError("K1 matrix plan is not a closed version-2 document")
    if (value["schema_version"] != 2 or value["kind"] != _PLAN_KIND or
            value["status"] != "frozen" or not _is_sha256_hex(str(value["study_sha256"])) or
            not _is_sha256_hex(str(value["matrix_sha256"])) or
            not isinstance(value["runtime_requirements"], dict) or
            not isinstance(value["controller"], dict) or
            set(value["controller"]) != {"root", "tree_sha256", "tree_files"} or
            not isinstance(value["cells"], list) or not value["cells"]):
        raise ValueError("K1 matrix plan identity is invalid")
    controller_root = Path(str(value["controller"]["root"])).resolve()
    controller_rows = _controller_rows(controller_root)
    if (value["controller"]["tree_sha256"] != _tree_sha(controller_rows) or
            value["controller"]["tree_files"] != len(controller_rows)):
        raise ValueError("controller source closure changed after matrix planning")
    seen_runs: set[str] = set()
    seen_cells: set[tuple[Any, ...]] = set()
    identity_cells: list[dict[str, Any]] = []
    verified_cells: list[dict[str, Any]] = []
    for index, row in enumerate(value["cells"]):
        fields = {"execution_index", "run_id", "cell", "contract_root", "contract_path",
                  "contract_sha256", "tree_sha256", "tree_files", "build_tool"}
        if not isinstance(row, dict) or set(row) != fields or row["execution_index"] != index:
            raise ValueError("K1 matrix plan cell schema/order differs")
        root = Path(str(row["contract_root"])).resolve()
        relative = PurePosixPath(str(row["contract_path"]))
        if relative.as_posix() != "measurement_contract.yaml":
            raise ValueError("planned K1 contract does not use its canonical filename")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("planned contract path escapes its frozen root")
        contract_path = root.joinpath(*relative.parts).resolve()
        if (not contract_path.is_relative_to(root) or contract_path.is_symlink() or
                not contract_path.is_file()):
            raise ValueError("planned contract is absent or escapes its frozen root")
        contract, cell = _contract(contract_path)
        rows = _regular_tree(root)
        build_tool = _contract_build_tool(root, contract)
        key = (cell["model"], cell["backend"], cell["precision"], cell["core_count"])
        if (contract["target"] != "k1" or contract["study_sha256"] != value["study_sha256"] or
                contract["run_id"] != row["run_id"] or dict(cell) != row["cell"] or
                build_tool != row["build_tool"] or
                _sha_file(contract_path) != row["contract_sha256"] or
                _tree_sha(rows) != row["tree_sha256"] or len(rows) != row["tree_files"] or
                row["run_id"] in seen_runs or key in seen_cells):
            raise ValueError("planned K1 contract tree changed or has duplicate identity")
        identity_cells.append({key_name: row[key_name] for key_name in (
            "execution_index", "run_id", "cell", "contract_sha256", "tree_sha256",
            "build_tool")})
        verified_cells.append({"cell": dict(cell), "build_tool": build_tool})
        seen_runs.add(str(row["run_id"]))
        seen_cells.add(key)
    identity = {
        "study_sha256": value["study_sha256"],
        "controller_tree_sha256": value["controller"]["tree_sha256"],
        "runtime_requirements": value["runtime_requirements"],
        "cells": identity_cells,
    }
    if (_runtime_requirements(verified_cells) != value["runtime_requirements"]
            or _canonical_sha(identity) != value["matrix_sha256"]):
        raise ValueError("K1 matrix plan digest differs from its cells")
    return value


def _write_tar(archive: Path, root: Path, rows: Sequence[Mapping[str, Any]], *,
               prefix: str) -> str:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as stream:
        for row in rows:
            source = root / str(row["path"])
            stat = source.stat()
            if (_sha_file(source) != row["sha256"] or stat.st_size != row["size"] or
                    stat.st_mode & 0o777 != row["mode"]):
                raise ValueError(f"staged source changed before archiving: {source}")
            info = tarfile.TarInfo(f"{prefix}/{row['path']}")
            info.size = int(row["size"])
            info.mode = int(row["mode"])
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as input_stream:
                stream.addfile(info, input_stream)
            stat = source.stat()
            if (_sha_file(source) != row["sha256"] or stat.st_size != row["size"] or
                    stat.st_mode & 0o777 != row["mode"]):
                raise ValueError(f"staged source changed while archiving: {source}")
    return _sha_file(archive)


@dataclass(frozen=True)
class SSHConfig:
    host: str
    user: str = "root"
    port: int = 22
    key: Path | None = None
    remote_root: str = "/var/lib/merlin-paper-k1"
    remote_python: str = "/usr/bin/python3"

    def __post_init__(self) -> None:
        remote_root = PurePosixPath(self.remote_root)
        remote_python = PurePosixPath(self.remote_python)
        if (not self.host or self.user != "root" or not 1 <= self.port <= 65535 or
                not remote_root.is_absolute() or remote_root.as_posix() != self.remote_root or
                len(remote_root.parts) < 3 or ".." in remote_root.parts or
                not remote_python.is_absolute() or
                remote_python.as_posix() != self.remote_python or ".." in remote_python.parts):
            raise ValueError(
                "K1 SSH execution requires root, absolute remote paths, and a valid port")
        if self.key is not None and (not self.key.is_file() or self.key.is_symlink()):
            raise ValueError("K1 SSH private key is absent or a symlink")


class RemoteTransport(Protocol):
    def stage(self, archive: Path, *, archive_sha256: str, tree_sha256: str,
              kind: str) -> str: ...
    def environment_preflight(self, *, remote_controller: str, remote_output: str,
                              matrix_sha256: str, controller_tree_sha256: str,
                              runtime_requirements_sha256: str,
                              required_core_count: int) -> bytes: ...
    def terminal_exists(self, remote_run_root: str) -> bool: ...
    def run_cell(self, *, remote_controller: str, remote_contract: str,
                 remote_run_root: str, contract_tree_sha256: str, unit_name: str,
                 timeout_seconds: int) -> dict[str, Any]: ...
    def retrieve(self, remote_run_root: str, destination: Path) -> str: ...


class K1SSHSystemdTransport:
    """Minimal non-interactive SSH/SCP transport; every remote mutation is content-addressed."""

    def __init__(self, config: SSHConfig, *,
                 runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run):
        self.config = config
        self.runner = runner

    def _ssh_base(self) -> list[str]:
        argv = ["ssh", "-o", "BatchMode=yes", "-p", str(self.config.port)]
        if self.config.key is not None:
            argv += ["-i", str(self.config.key.resolve())]
        return [*argv, f"{self.config.user}@{self.config.host}"]

    def _scp_base(self) -> list[str]:
        argv = ["scp", "-q", "-o", "BatchMode=yes", "-P", str(self.config.port)]
        if self.config.key is not None:
            argv += ["-i", str(self.config.key.resolve())]
        return argv

    def _run(self, argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        completed = self.runner(argv, capture_output=True, text=True, timeout=timeout, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"remote command failed rc={completed.returncode}: {completed.stderr[-2000:]}")
        return completed

    def _ssh(self, argv: Sequence[str], *,
             timeout: float = 120) -> subprocess.CompletedProcess[str]:
        return self._run([*self._ssh_base(), shlex.join(list(argv))], timeout=timeout)

    def stage(self, archive: Path, *, archive_sha256: str, tree_sha256: str,
              kind: str) -> str:
        if kind not in {"controller", "contract"} or not _is_sha256_hex(archive_sha256) or (
                not _is_sha256_hex(tree_sha256) or _sha_file(archive) != archive_sha256):
            raise ValueError("remote stage request has an invalid content identity")
        base = PurePosixPath(self.config.remote_root)
        incoming_dir = base / "incoming"
        incoming = incoming_dir / f"{archive_sha256}.{uuid.uuid4().hex}.tar"
        destination = base / "cache" / kind / tree_sha256
        self._ssh(["install", "-d", "-m", "0700", str(incoming_dir),
                   str(destination.parent)], timeout=30)
        remote = f"{self.config.user}@{self.config.host}:{incoming}"
        self._run([*self._scp_base(), str(archive), remote], timeout=1800)
        marker = json.dumps({"archive_sha256": archive_sha256, "tree_sha256": tree_sha256},
                            sort_keys=True, separators=(",", ":"))
        script = """
set -eu
actual=$(/usr/bin/sha256sum "$1" | /usr/bin/cut -d ' ' -f1)
test "$actual" = "$2"
if test -d "$3"; then
  test "$(/usr/bin/cat "$3/.merlin-stage.json")" = "$5"
  /usr/bin/rm -f -- "$1"
  exit 0
fi
tmp="$3.tmp.$4"
/usr/bin/install -d -m 0700 "$tmp"
/usr/bin/tar -xf "$1" -C "$tmp" --no-same-owner --no-same-permissions
/usr/bin/printf '%s' "$5" > "$tmp/.merlin-stage.json"
/usr/bin/mv -T "$tmp" "$3"
/usr/bin/rm -f -- "$1"
""".strip()
        self._ssh(["/bin/sh", "-c", script, "stage", str(incoming), archive_sha256,
                   str(destination), uuid.uuid4().hex, marker], timeout=1800)
        return str(destination)

    def terminal_exists(self, remote_run_root: str) -> bool:
        completed = self.runner(
            [*self._ssh_base(), shlex.join(["test", "-f", f"{remote_run_root}/terminal.json"])],
            capture_output=True, text=True, timeout=30, check=False)
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"cannot inspect remote terminal: {completed.stderr[-1000:]}")
        return completed.returncode == 0

    def environment_preflight(self, *, remote_controller: str, remote_output: str,
                              matrix_sha256: str, controller_tree_sha256: str,
                              runtime_requirements_sha256: str,
                              required_core_count: int) -> bytes:
        pythonpath = f"{remote_controller}/repo/merlin/python"
        argv = [
            "/usr/bin/env", f"PYTHONPATH={pythonpath}", "PYTHONDONTWRITEBYTECODE=1",
            self.config.remote_python, "-m", "merlin.compare.paper_k1_orchestrator",
            "remote-preflight", "--output", remote_output,
            "--matrix-sha256", matrix_sha256,
            "--controller-tree-sha256", controller_tree_sha256,
            "--runtime-requirements-sha256", runtime_requirements_sha256,
            "--required-core-count", str(required_core_count),
            "--expected-python", self.config.remote_python,
        ]
        self._ssh(argv, timeout=300)
        return self._ssh(["/usr/bin/cat", remote_output], timeout=30).stdout.encode("utf-8")

    def run_cell(self, *, remote_controller: str, remote_contract: str,
                 remote_run_root: str, contract_tree_sha256: str, unit_name: str,
                 timeout_seconds: int) -> dict[str, Any]:
        if not _is_safe_unit_name(unit_name) or timeout_seconds <= 0:
            raise ValueError("systemd cell unit identity is invalid")
        pythonpath = f"{remote_controller}/repo/merlin/python"
        argv = [
            "/usr/bin/systemd-run", "--quiet", "--wait", "--collect", "--pipe",
            f"--unit={unit_name}", "--property=Type=oneshot",
            f"--property=TimeoutStartSec={2 * timeout_seconds + 240}",
            f"--property=RuntimeMaxSec={2 * timeout_seconds + 240}",
            f"--working-directory={remote_contract}/contract",
            f"--setenv=PYTHONPATH={pythonpath}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
            "/usr/bin/flock", "-w", str(timeout_seconds + 120),
            "/run/lock/merlin-paper-k1.lock",
            self.config.remote_python, "-m", "merlin.compare.paper_k1_orchestrator",
            "remote-cell", "--contract", f"{remote_contract}/contract/measurement_contract.yaml",
            "--output-root", remote_run_root,
            "--contract-tree-sha256", contract_tree_sha256,
        ]
        started = time.monotonic_ns()
        completed = self._ssh(argv, timeout=2 * timeout_seconds + 300)
        return {
            "started_monotonic_ns": started, "ended_monotonic_ns": time.monotonic_ns(),
            "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:],
            "unit": unit_name,
        }

    def retrieve(self, remote_run_root: str, destination: Path) -> str:
        archive = f"{remote_run_root}/retrieval.tar"
        script = """
set -eu
root="$1"
test -f "$root/terminal.json"
/usr/bin/rm -f -- "$root/retrieval.tar"
/usr/bin/tar -cf "$root/retrieval.tar" -C "$root" \
  terminal.json result.yaml output .paper-controller-issuance-v1
/usr/bin/sha256sum "$root/retrieval.tar" | /usr/bin/cut -d ' ' -f1
""".strip()
        completed = self._ssh(["/bin/sh", "-c", script, "retrieve", remote_run_root],
                              timeout=300)
        digest = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        if not _is_sha256_hex(digest):
            raise RuntimeError("remote retrieval archive has no SHA-256 receipt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        remote = f"{self.config.user}@{self.config.host}:{archive}"
        self._run([*self._scp_base(), remote, str(destination)], timeout=1800)
        if _sha_file(destination) != digest:
            raise ValueError("atomically retrieved result archive differs from remote SHA-256")
        return digest


def _safe_extract(archive: Path, destination: Path) -> None:
    allowed = {"terminal.json", "result.yaml", "output", ".paper-controller-issuance-v1"}
    with tarfile.open(archive, "r") as stream:
        members = stream.getmembers()
        if not members:
            raise ValueError("retrieval archive is empty")
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts or
                    path.parts[0] not in allowed or member.issym() or member.islnk() or
                    not (member.isfile() or member.isdir())):
                raise ValueError("retrieval archive contains an unsafe or unexpected entry")
        stream.extractall(destination, members=members, filter="data")


def _terminal_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema_version", "kind", "status", "run_id", "contract_tree_sha256",
              "contract_sha256", "receipt", "result", "issuance_fingerprint",
              "completed_at_utc"}
    if (not isinstance(value, dict) or set(value) != fields or value["schema_version"] != 1 or
            value["kind"] != _TERMINAL_KIND or value["status"] != "terminal" or
            not _is_sha256_hex(str(value["contract_tree_sha256"])) or
            not _is_sha256_hex(str(value["contract_sha256"])) or
            not _is_sha256_hex(str(value["issuance_fingerprint"]))):
        raise ValueError("remote terminal marker is malformed")
    for name in ("receipt", "result"):
        ref = value[name]
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or
                not _is_sha256_hex(str(ref["sha256"]))):
            raise ValueError(f"remote terminal {name} reference is malformed")
    return value


def _contained(root: Path, relative: object, *, label: str) -> Path:
    raw = PurePosixPath(str(relative))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} escapes its transport bundle")
    path = root.joinpath(*raw.parts).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink()):
        raise ValueError(f"{label} is absent from its transport bundle")
    return path


def _validate_retrieved(transport_root: Path, planned: Mapping[str, Any], *,
                        retrieval_sha256: str, final_dir: Path) -> dict[str, Any]:
    terminal = _terminal_document(transport_root / "terminal.json")
    if (terminal["run_id"] != planned["run_id"] or
            terminal["contract_tree_sha256"] != planned["tree_sha256"] or
            terminal["contract_sha256"] != planned["contract_sha256"]):
        raise ValueError("retrieved terminal marker differs from frozen matrix cell")
    receipt = _contained(transport_root, terminal["receipt"]["path"], label="retrieved receipt")
    remote_result = _contained(
        transport_root, terminal["result"]["path"], label="retrieved result")
    if (_sha_file(receipt) != terminal["receipt"]["sha256"] or
            _sha_file(remote_result) != terminal["result"]["sha256"]):
        raise ValueError("retrieved terminal files differ from their remote digests")
    fingerprint = issuance_fingerprint(receipt)
    if fingerprint != terminal["issuance_fingerprint"]:
        raise ValueError("retrieved issuance fingerprint differs from remote terminal marker")
    result = yaml.safe_load(remote_result.read_text(encoding="utf-8"))
    if (not isinstance(result, dict) or result.get("run_id") != planned["run_id"] or
            any(result.get(name) != value for name, value in planned["cell"].items())):
        raise ValueError("retrieved result identity differs from frozen matrix cell")
    localized = dict(result)
    localized["measurement_receipt"] = dict(localized["measurement_receipt"])
    local_receipt = final_dir / "transport" / terminal["receipt"]["path"]
    localized["measurement_receipt"]["path"] = str(local_receipt)
    if localized["measurement_receipt"].get("sha256") != _sha_file(receipt):
        raise ValueError("retrieved result does not bind its controller receipt")
    validate_paper_result(localized)
    return {
        "terminal": terminal, "localized_result": localized,
        "issuance_fingerprint": fingerprint, "receipt_sha256": _sha_file(receipt),
        "remote_result_sha256": _sha_file(remote_result),
        "retrieval_archive_sha256": retrieval_sha256,
    }


def _validate_local_terminal(cell_dir: Path, planned: Mapping[str, Any]) -> dict[str, Any]:
    if cell_dir.is_symlink() or not cell_dir.is_dir():
        raise ValueError("local terminal cell must be a regular directory")
    state_path = cell_dir / "terminal-state.json"
    if not state_path.is_file() or state_path.is_symlink():
        raise ValueError(f"existing cell directory is not terminal: {cell_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    fields = {"schema_version", "kind", "status", "run_id", "cell", "contract_sha256",
              "contract_tree_sha256", "receipt_sha256", "result_sha256",
              "remote_result_sha256", "retrieval_archive_sha256", "issuance_fingerprint"}
    if (not isinstance(state, dict) or set(state) != fields or state["schema_version"] != 1 or
            state["kind"] != _STATE_KIND or state["status"] != "terminal" or
            state["run_id"] != planned["run_id"] or state["cell"] != planned["cell"] or
            state["contract_sha256"] != planned["contract_sha256"] or
            state["contract_tree_sha256"] != planned["tree_sha256"] or
            any(not _is_sha256_hex(str(state[name])) for name in (
                "receipt_sha256", "result_sha256", "remote_result_sha256",
                "retrieval_archive_sha256", "issuance_fingerprint"))):
        raise ValueError("local terminal state differs from frozen matrix cell")
    result_path = cell_dir / "result.yaml"
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    if (result_path.is_symlink() or not isinstance(result, dict) or
            _sha_file(result_path) != state["result_sha256"] or
            result.get("run_id") != planned["run_id"]):
        raise ValueError("localized terminal result changed after retrieval")
    receipt_raw = Path(str(result["measurement_receipt"]["path"]))
    receipt = receipt_raw.resolve()
    if (receipt_raw.is_symlink() or
            not receipt.is_relative_to((cell_dir / "transport").resolve()) or
            not receipt.is_file() or _sha_file(receipt) !=
            state["receipt_sha256"] or issuance_fingerprint(receipt) !=
            state["issuance_fingerprint"]):
        raise ValueError("local terminal receipt/issuance changed after retrieval")
    validate_paper_result(result)
    return state


def _remote_file_identity(path: Path, *, label: str) -> dict[str, Any]:
    configured = path
    try:
        resolved = configured.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"remote dependency {label} is absent") from error
    if not resolved.is_file():
        raise ValueError(f"remote dependency {label} is not a regular file")
    return {"path": str(configured), "resolved_path": str(resolved),
            "sha256": _sha_file(resolved), "size": resolved.stat().st_size}


def _remote_module_identity(public_name: str, import_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(import_name)
    except Exception as error:
        raise ValueError(f"required remote Python module {import_name} cannot be imported") from error
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"required remote Python module {import_name} has no source identity")
    identity = _remote_file_identity(Path(raw), label=import_name)
    return {"name": public_name, "import_name": import_name, **identity}


def _mapped_runtime_libraries() -> list[dict[str, Any]]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise ValueError("remote runtime lacks /proc/self/maps")
    paths: set[Path] = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[-1].startswith("/"):
            continue
        raw = fields[-1].removesuffix(" (deleted)")
        name = Path(raw).name
        if ".so" in name or name.startswith("ld-linux"):
            path = Path(raw).resolve()
            if path.is_file():
                paths.add(path)
    rows = [{"path": str(path), "sha256": _sha_file(path), "size": path.stat().st_size}
            for path in sorted(paths)]
    names = {Path(row["path"]).name for row in rows}
    if (not any(name.startswith("libc.so") for name in names)
            or not any(name.startswith("ld-linux") for name in names)):
        raise ValueError("remote runtime cannot bind both libc and the ELF dynamic loader")
    return rows


def _openssl_ed25519_selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="merlin-paper-openssl-") as temporary:
        root = Path(temporary)
        private, public = root / "private.pem", root / "public.pem"
        message, signature = root / "message", root / "signature"
        message.write_bytes(b"merlin-paper-k1-environment-preflight-v1")
        commands = [
            ["/usr/bin/openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
            ["/usr/bin/openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
            ["/usr/bin/openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin",
             "-in", str(message), "-out", str(signature)],
            ["/usr/bin/openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(public),
             "-rawin", "-in", str(message), "-sigfile", str(signature)],
        ]
        for argv in commands:
            completed = subprocess.run(
                argv, capture_output=True, timeout=30, check=False,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
            if completed.returncode:
                raise ValueError("remote OpenSSL lacks the controller's ED25519 operation set")


def create_remote_environment_receipt(output_path: str | Path, *, matrix_sha256: str,
                                      controller_tree_sha256: str,
                                      runtime_requirements_sha256: str,
                                      required_core_count: int,
                                      expected_python: str) -> Path:
    """Board-only dependency and runtime proof, completed before the first paper cell."""
    for digest in (matrix_sha256, controller_tree_sha256, runtime_requirements_sha256):
        if not _is_sha256_hex(digest):
            raise ValueError("remote environment preflight identity is not SHA-256 bound")
    if required_core_count <= 0 or not PurePosixPath(expected_python).is_absolute():
        raise ValueError("remote environment preflight requirements are invalid")
    if not platform.machine().lower().startswith("riscv"):
        raise ValueError("K1 environment preflight must execute on RISC-V")
    output = Path(output_path)
    if output.is_symlink() or not PurePosixPath(output.as_posix()).is_absolute():
        raise ValueError("remote environment receipt requires an absolute non-symlink path")
    if output.exists():
        value = _validate_environment_receipt_bytes(
            output.read_bytes(), matrix_sha256=matrix_sha256,
            controller_tree_sha256=controller_tree_sha256,
            runtime_requirements_sha256=runtime_requirements_sha256,
            required_core_count=required_core_count)
        if value["python"]["path"] != expected_python:
            raise ValueError("existing remote environment receipt used another Python path")
        return output

    expected_python_path = Path(expected_python)
    python_identity = _remote_file_identity(expected_python_path, label="configured Python")
    if Path("/proc/self/exe").resolve() != Path(python_identity["resolved_path"]):
        raise ValueError("remote preflight is not running under the configured Python bytes")
    python_identity.update({
        "version": platform.python_version(), "implementation": platform.python_implementation(),
    })
    modules = {public: _remote_module_identity(public, imported)
               for public, imported in _REMOTE_MODULES.items()}
    if yaml.safe_load("ready: true") != {"ready": True}:
        raise ValueError("remote PyYAML semantic self-test failed")
    run_logger = importlib.import_module("aet.tracking.run_logger")
    if not hasattr(run_logger, "EvalRunLogger"):
        raise ValueError("remote AET installation omits EvalRunLogger")
    tools = {name: _remote_file_identity(Path(path), label=name)
             for name, path in _REMOTE_TOOLS.items()}
    if any(not os.access(Path(row["resolved_path"]), os.X_OK) for row in tools.values()):
        raise ValueError("remote environment contains a non-executable required tool")
    if not Path("/run/systemd/system").is_dir():
        raise ValueError("remote systemd manager is not active")
    systemd_version = subprocess.run(
        ["/usr/bin/systemd-run", "--version"], capture_output=True, timeout=15, check=False)
    if systemd_version.returncode:
        raise ValueError("remote systemd-run identity query failed")
    _openssl_ed25519_selftest()

    from .paper_measurement_controller import _compile_probe, _frequency_rows, _probe
    source = Path(__file__).with_name("paper_k1_board_probe.c")
    with tempfile.TemporaryDirectory(prefix="merlin-paper-k1-preflight-") as temporary:
        probe_executable = Path(temporary) / "board-probe"
        _compile_probe(source, "k1", probe_executable, 60)
        _raw_probe, probe = _probe(probe_executable, "k1", 30)
        probe_sha = _sha_file(probe_executable)
    core_ids = list(range(required_core_count))
    frequencies = _frequency_rows("k1", core_ids, probe)
    affinity = sorted(os.sched_getaffinity(0))
    if not set(core_ids) <= set(affinity):
        raise ValueError("remote preflight process cannot address every frozen matrix CPU")
    if not Path("/proc/self/task").is_dir():
        raise ValueError("remote runtime lacks controller-required procfs task state")

    document = {
        "schema_version": 1, "kind": _ENVIRONMENT_KIND, "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": matrix_sha256,
        "controller_tree_sha256": controller_tree_sha256,
        "runtime_requirements_sha256": runtime_requirements_sha256,
        "machine": platform.machine().lower(),
        "python": python_identity, "modules": modules, "tools": tools,
        "runtime": {
            "required_core_count": required_core_count, "available_affinity": affinity,
            "procfs_task_state": True, "systemd_manager": True,
            "systemd_version_sha256": _sha_bytes(systemd_version.stdout + systemd_version.stderr),
            "openssl_ed25519": True, "board_probe_source_sha256": _sha_file(source),
            "board_probe_executable_sha256": probe_sha, "board_probe": probe,
            "core_frequencies": frequencies,
            "mapped_libraries": _mapped_runtime_libraries(),
        },
    }
    _validate_environment_receipt_bytes(
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        matrix_sha256=matrix_sha256, controller_tree_sha256=controller_tree_sha256,
        runtime_requirements_sha256=runtime_requirements_sha256,
        required_core_count=required_core_count)
    _atomic_bytes(
        output, (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return output


def _validate_environment_receipt_bytes(value: bytes, *, matrix_sha256: str,
                                        controller_tree_sha256: str,
                                        runtime_requirements_sha256: str,
                                        required_core_count: int) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("remote environment preflight receipt is not JSON") from error
    fields = {"schema_version", "kind", "status", "created_at_utc", "matrix_sha256",
              "controller_tree_sha256", "runtime_requirements_sha256", "machine", "python",
              "modules", "tools", "runtime"}
    if (not isinstance(document, dict) or set(document) != fields
            or document["schema_version"] != 1 or document["kind"] != _ENVIRONMENT_KIND
            or document["status"] != "ready" or document["matrix_sha256"] != matrix_sha256
            or document["controller_tree_sha256"] != controller_tree_sha256
            or document["runtime_requirements_sha256"] != runtime_requirements_sha256
            or not str(document["machine"]).startswith("riscv")):
        raise ValueError("remote environment receipt differs from the frozen K1 matrix")
    try:
        created = datetime.fromisoformat(str(document["created_at_utc"]))
    except ValueError as error:
        raise ValueError("remote environment receipt timestamp is invalid") from error
    if created.tzinfo is None:
        raise ValueError("remote environment receipt timestamp has no timezone")
    file_fields = {"path", "resolved_path", "sha256", "size"}
    python = document["python"]
    if (not isinstance(python, dict) or set(python) != file_fields | {"version", "implementation"}
            or not _is_sha256_hex(str(python.get("sha256", "")))
            or not isinstance(python.get("size"), int) or python["size"] <= 0):
        raise ValueError("remote Python identity is incomplete")
    modules = document["modules"]
    if not isinstance(modules, dict) or set(modules) != set(_REMOTE_MODULES):
        raise ValueError("remote Python module roster is incomplete")
    for name, row in modules.items():
        if (not isinstance(row, dict)
                or set(row) != file_fields | {"name", "import_name"}
                or row["name"] != name or row["import_name"] != _REMOTE_MODULES[name]
                or not _is_sha256_hex(str(row.get("sha256", "")))
                or not isinstance(row.get("size"), int) or row["size"] <= 0):
            raise ValueError("remote Python module identity is malformed")
    tools = document["tools"]
    if not isinstance(tools, dict) or set(tools) != set(_REMOTE_TOOLS):
        raise ValueError("remote executable dependency roster is incomplete")
    for name, row in tools.items():
        if (not isinstance(row, dict) or set(row) != file_fields
                or row["path"] != _REMOTE_TOOLS[name]
                or not _is_sha256_hex(str(row.get("sha256", "")))
                or not isinstance(row.get("size"), int) or row["size"] <= 0):
            raise ValueError("remote executable dependency identity is malformed")
    runtime = document["runtime"]
    runtime_fields = {"required_core_count", "available_affinity", "procfs_task_state",
                      "systemd_manager", "systemd_version_sha256", "openssl_ed25519",
                      "board_probe_source_sha256", "board_probe_executable_sha256",
                      "board_probe", "core_frequencies", "mapped_libraries"}
    if (not isinstance(runtime, dict) or set(runtime) != runtime_fields
            or runtime["required_core_count"] != required_core_count
            or runtime["procfs_task_state"] is not True or runtime["systemd_manager"] is not True
            or runtime["openssl_ed25519"] is not True
            or any(not _is_sha256_hex(str(runtime[name])) for name in (
                "systemd_version_sha256", "board_probe_source_sha256",
                "board_probe_executable_sha256"))
            or runtime["available_affinity"] != sorted(set(runtime["available_affinity"]))
            or not set(range(required_core_count)) <= set(runtime["available_affinity"])):
        raise ValueError("remote controller runtime proof is incomplete")
    probe = runtime["board_probe"]
    if (not isinstance(probe, dict) or probe.get("kind") != "merlin_board_probe_v1"
            or probe.get("vlen_source") != "csr" or probe.get("governor") != "performance"):
        raise ValueError("remote environment receipt lacks a trusted K1 board probe")
    frequencies = runtime["core_frequencies"]
    if (not isinstance(frequencies, list) or len(frequencies) != required_core_count
            or [row.get("core_id") for row in frequencies] != list(range(required_core_count))
            or any(row.get("governor") != "performance"
                   or row.get("current_khz") != row.get("max_khz") for row in frequencies)):
        raise ValueError("remote environment receipt does not lock every requested core")
    libraries = runtime["mapped_libraries"]
    if (not isinstance(libraries, list) or not libraries
            or any(not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}
                   or not _is_sha256_hex(str(row.get("sha256", "")))
                   or not isinstance(row.get("size"), int) or row["size"] <= 0
                   for row in libraries)):
        raise ValueError("remote dynamic-runtime dependency identity is incomplete")
    return document


def _notary(plan: Mapping[str, Any], fingerprints: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1, "kind": _NOTARY_KIND,
        "study_sha256": plan["study_sha256"],
        "fingerprints": dict(sorted(fingerprints.items())),
    }


def _default_output(plan: Mapping[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (artifacts_dir() / "paper-k1-matrix" / "k1" /
            f"{stamp}_{str(plan['matrix_sha256'])[:8]}")


def _cell_label(planned: Mapping[str, Any]) -> str:
    cell = planned["cell"]
    return _safe_name(
        f"{planned['execution_index']:03d}_{cell['model']}_{cell['backend']}_"
        f"{cell['precision']}_{cell['core_count']}c")


def run_matrix(plan_path: str | Path, *, transport: RemoteTransport,
               output_dir: str | Path | None = None, resume: bool = False,
               remote_root: str = "/var/lib/merlin-paper-k1") -> Path:
    """Run/retrieve the frozen matrix sequentially; never rerun a validated terminal cell."""
    unresolved_plan = Path(plan_path)
    if unresolved_plan.is_symlink():
        raise ValueError("K1 matrix plan cannot be a symlink")
    plan_path = unresolved_plan.resolve()
    plan_file_sha256 = _sha_file(plan_path)
    plan = _load_plan(plan_path)
    if _sha_file(plan_path) != plan_file_sha256:
        raise ValueError("K1 matrix plan changed while it was being validated")
    unresolved_output = Path(output_dir) if output_dir is not None else _default_output(plan)
    if unresolved_output.is_symlink():
        raise ValueError("K1 matrix output cannot be a symlink")
    output = unresolved_output.resolve()
    if output.exists() and not resume:
        raise FileExistsError(f"matrix output exists; pass resume=True: {output}")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".matrix.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns this K1 matrix output") from error
        retained_plan = output / "matrix-plan.json"
        if retained_plan.exists():
            if _sha_file(retained_plan) != plan_file_sha256:
                raise ValueError("resume plan differs from retained matrix plan")
        else:
            temporary_plan = output / f".matrix-plan.{uuid.uuid4().hex}.tmp"
            temporary_plan.write_bytes(plan_path.read_bytes())
            if _sha_file(temporary_plan) != plan_file_sha256:
                raise ValueError("retained matrix plan copy differs")
            with temporary_plan.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_plan, retained_plan)
            descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        staging = output / "staging"
        controller_root = Path(plan["controller"]["root"])
        controller_rows = _controller_rows(controller_root)
        remote_controller: str | None = None

        requirements_sha = _canonical_sha(plan["runtime_requirements"])
        environment_path = output / "environment-preflight.json"
        existing_cells = output / "cells"
        if environment_path.exists():
            if environment_path.is_symlink() or not environment_path.is_file():
                raise ValueError("retained K1 environment receipt is unsafe")
            environment_bytes = environment_path.read_bytes()
            _validate_environment_receipt_bytes(
                environment_bytes, matrix_sha256=plan["matrix_sha256"],
                controller_tree_sha256=plan["controller"]["tree_sha256"],
                runtime_requirements_sha256=requirements_sha,
                required_core_count=plan["runtime_requirements"]["required_core_count"])
        else:
            if existing_cells.exists() and any(existing_cells.iterdir()):
                raise ValueError(
                    "existing terminal cells have no retained prerequisite environment receipt")
            try:
                controller_archive = staging / (
                    f"controller-{plan['controller']['tree_sha256']}.tar")
                controller_archive_sha = _write_tar(
                    controller_archive, controller_root, controller_rows, prefix="repo")
                if (_tree_sha(_controller_rows(controller_root)) !=
                        plan["controller"]["tree_sha256"]):
                    raise ValueError("controller source changed during environment preflight staging")
                remote_controller = transport.stage(
                    controller_archive, archive_sha256=controller_archive_sha,
                    tree_sha256=plan["controller"]["tree_sha256"], kind="controller")
                remote_matrix = str(
                    PurePosixPath(remote_root) / "runs" / plan["matrix_sha256"])
                environment_bytes = transport.environment_preflight(
                    remote_controller=remote_controller,
                    remote_output=f"{remote_matrix}/environment-preflight.json",
                    matrix_sha256=plan["matrix_sha256"],
                    controller_tree_sha256=plan["controller"]["tree_sha256"],
                    runtime_requirements_sha256=requirements_sha,
                    required_core_count=plan["runtime_requirements"]["required_core_count"])
                _validate_environment_receipt_bytes(
                    environment_bytes, matrix_sha256=plan["matrix_sha256"],
                    controller_tree_sha256=plan["controller"]["tree_sha256"],
                    runtime_requirements_sha256=requirements_sha,
                    required_core_count=plan["runtime_requirements"]["required_core_count"])
                _atomic_bytes(environment_path, environment_bytes)
            except Exception as exc:
                _atomic_json(output / "environment-preflight-failure.json", {
                    "schema_version": 1, "status": "blocked_before_first_cell",
                    "matrix_sha256": plan["matrix_sha256"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                _atomic_json(output / "matrix-state.json", {
                    "schema_version": 1, "status": "incomplete",
                    "matrix_sha256": plan["matrix_sha256"], "terminal_cells": 0,
                    "expected_cells": len(plan["cells"]),
                    "environment_preflight": "failed",
                })
                raise
        environment_sha = _sha_file(environment_path)

        fingerprints: dict[str, str] = {}
        for planned in plan["cells"]:
            label = _cell_label(planned)
            cell_dir = output / "cells" / label
            if cell_dir.exists():
                state = _validate_local_terminal(cell_dir, planned)
                fingerprints[planned["run_id"]] = state["issuance_fingerprint"]
                _atomic_yaml(output / "issuance-notary.partial.yaml", _notary(plan, fingerprints))
                continue
            remote_run = str(PurePosixPath(remote_root) / "runs" / plan["matrix_sha256"] /
                             f"{planned['execution_index']:03d}_{_safe_name(planned['run_id'])}")
            attempt: dict[str, Any] = {
                "schema_version": 1, "run_id": planned["run_id"],
                "cell": planned["cell"], "remote_run_root": remote_run,
                "environment_preflight_sha256": environment_sha,
                "board_terminal_before": transport.terminal_exists(remote_run),
            }
            try:
                if not attempt["board_terminal_before"]:
                    if remote_controller is None:
                        controller_archive = staging / (
                            f"controller-{plan['controller']['tree_sha256']}.tar")
                        controller_archive_sha = _write_tar(
                            controller_archive, controller_root, controller_rows, prefix="repo")
                        if (_tree_sha(_controller_rows(controller_root)) !=
                                plan["controller"]["tree_sha256"]):
                            raise ValueError("controller source changed while it was archived")
                        remote_controller = transport.stage(
                            controller_archive, archive_sha256=controller_archive_sha,
                            tree_sha256=plan["controller"]["tree_sha256"], kind="controller")
                    contract_root = Path(planned["contract_root"])
                    contract_rows = _regular_tree(contract_root)
                    contract_archive = staging / f"contract-{planned['tree_sha256']}.tar"
                    contract_archive_sha = _write_tar(
                        contract_archive, contract_root, contract_rows, prefix="contract")
                    if _tree_sha(_regular_tree(contract_root)) != planned["tree_sha256"]:
                        raise ValueError("frozen contract changed while it was archived")
                    remote_contract = transport.stage(
                        contract_archive, archive_sha256=contract_archive_sha,
                        tree_sha256=planned["tree_sha256"], kind="contract")
                    contract, _ = _contract(
                        contract_root / PurePosixPath(planned["contract_path"]))
                    attempt["systemd"] = transport.run_cell(
                        remote_controller=remote_controller, remote_contract=remote_contract,
                        remote_run_root=remote_run,
                        contract_tree_sha256=planned["tree_sha256"],
                        unit_name=("merlin-paper-" + plan["matrix_sha256"][:10] + "-" +
                                   f"{planned['execution_index']:03d}"),
                        timeout_seconds=int(contract["timeout_seconds"]),
                    )
                incoming = output / ".incoming" / f"{label}.{uuid.uuid4().hex}"
                incoming.mkdir(parents=True, exist_ok=False)
                archive = incoming / "retrieval.tar"
                retrieval_sha = transport.retrieve(remote_run, archive)
                transport_root = incoming / "transport"
                transport_root.mkdir()
                _safe_extract(archive, transport_root)
                archive.unlink()
                verified = _validate_retrieved(
                    transport_root, planned, retrieval_sha256=retrieval_sha,
                    final_dir=cell_dir)
                result_path = incoming / "result.yaml"
                _atomic_yaml(result_path, verified["localized_result"])
                state = {
                    "schema_version": 1, "kind": _STATE_KIND, "status": "terminal",
                    "run_id": planned["run_id"], "cell": planned["cell"],
                    "contract_sha256": planned["contract_sha256"],
                    "contract_tree_sha256": planned["tree_sha256"],
                    "receipt_sha256": verified["receipt_sha256"],
                    "result_sha256": _sha_file(result_path),
                    "remote_result_sha256": verified["remote_result_sha256"],
                    "retrieval_archive_sha256": verified["retrieval_archive_sha256"],
                    "issuance_fingerprint": verified["issuance_fingerprint"],
                }
                _atomic_json(incoming / "terminal-state.json", state)
                cell_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(incoming, cell_dir)
                descriptor = os.open(cell_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                fingerprints[planned["run_id"]] = verified["issuance_fingerprint"]
                attempt["status"] = "terminal_retrieved"
                attempt["retrieval_archive_sha256"] = retrieval_sha
                _atomic_yaml(output / "issuance-notary.partial.yaml", _notary(plan, fingerprints))
            except Exception as exc:
                attempt["status"] = "not_terminal_locally_manual_recovery_may_be_required"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                _atomic_json(output / "attempts" / f"{label}_{time.time_ns()}.json", attempt)
                _atomic_json(output / "matrix-state.json", {
                    "schema_version": 1, "status": "incomplete", "matrix_sha256":
                    plan["matrix_sha256"], "terminal_cells": len(fingerprints),
                    "expected_cells": len(plan["cells"]),
                    "environment_preflight_sha256": environment_sha,
                })
                raise
            _atomic_json(output / "attempts" / f"{label}_{time.time_ns()}.json", attempt)

        if set(fingerprints) != {str(row["run_id"]) for row in plan["cells"]}:
            raise ValueError("K1 matrix completed without an exact issuance fingerprint roster")
        final_notary = _notary(plan, fingerprints)
        _atomic_yaml(output / "issuance-notary.yaml", final_notary)
        _atomic_json(output / "matrix-state.json", {
            "schema_version": 1, "status": "complete", "matrix_sha256":
            plan["matrix_sha256"], "terminal_cells": len(fingerprints),
            "expected_cells": len(plan["cells"]),
            "environment_preflight_sha256": environment_sha,
            "issuance_notary_sha256": _sha_file(output / "issuance-notary.yaml"),
        })
    return output


def finalize_matrix(plan_path: str | Path, run_dir: str | Path, study_path: str | Path, *,
                    output_path: str | Path | None = None) -> Path:
    """Ingest one exact terminal matrix and publish the canonical sealed ``results.yaml``."""
    from .paper import PaperStudySpec
    from .paper_attribution import attach_causal_attribution
    from .paper_report import build_paper_report, load_issuance_notary, seal_results_document
    from .study import execution_matrix

    unresolved_plan = Path(plan_path)
    unresolved_run = Path(run_dir)
    unresolved_study = Path(study_path)
    if any(path.is_symlink() for path in (unresolved_plan, unresolved_run, unresolved_study)):
        raise ValueError("K1 finalization inputs cannot be symlinks")
    plan_path = unresolved_plan.resolve()
    run = unresolved_run.resolve()
    study_path = unresolved_study.resolve()
    if not run.is_dir() or not study_path.is_file():
        raise ValueError("K1 finalization run/study input is absent")
    plan_file_sha = _sha_file(plan_path)
    plan = _load_plan(plan_path)
    if _sha_file(plan_path) != plan_file_sha:
        raise ValueError("K1 matrix plan changed during finalization")
    retained_plan = run / "matrix-plan.json"
    if (not retained_plan.is_file() or retained_plan.is_symlink()
            or _sha_file(retained_plan) != plan_file_sha):
        raise ValueError("K1 run did not retain the exact finalized matrix plan")
    study_file_sha = _sha_file(study_path)
    spec = PaperStudySpec.from_yaml(study_path)
    if _sha_file(study_path) != study_file_sha:
        raise ValueError("frozen K1 study changed while it was loaded")
    if spec.status != "frozen" or spec.target != "k1" or spec.sha256() != plan["study_sha256"]:
        raise ValueError("K1 finalization study differs from the frozen matrix identity")
    frozen_order = [{"model": cell.model.name, "backend": cell.backend.name,
                     "precision": cell.precision, "core_count": cell.core_count}
                    for cell in execution_matrix(spec)]
    if [row["cell"] for row in plan["cells"]] != frozen_order:
        raise ValueError("K1 matrix plan order/membership differs from the frozen study")

    state_path = run / "matrix-state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("K1 matrix completion state is absent or unsafe")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_fields = {"schema_version", "status", "matrix_sha256", "terminal_cells",
                    "expected_cells", "environment_preflight_sha256",
                    "issuance_notary_sha256"}
    if (not isinstance(state, dict) or set(state) != state_fields
            or state["schema_version"] != 1 or state["status"] != "complete"
            or state["matrix_sha256"] != plan["matrix_sha256"]
            or state["terminal_cells"] != len(plan["cells"])
            or state["expected_cells"] != len(plan["cells"])):
        raise ValueError("K1 matrix state is partial or differs from the frozen plan")
    environment_path = run / "environment-preflight.json"
    if environment_path.is_symlink() or not environment_path.is_file():
        raise ValueError("K1 prerequisite environment receipt is absent or unsafe")
    environment_bytes = environment_path.read_bytes()
    if _sha_bytes(environment_bytes) != state["environment_preflight_sha256"]:
        raise ValueError("K1 prerequisite environment receipt changed after execution")
    _validate_environment_receipt_bytes(
        environment_bytes, matrix_sha256=plan["matrix_sha256"],
        controller_tree_sha256=plan["controller"]["tree_sha256"],
        runtime_requirements_sha256=_canonical_sha(plan["runtime_requirements"]),
        required_core_count=plan["runtime_requirements"]["required_core_count"])

    notary_path = run / "issuance-notary.yaml"
    if (not notary_path.is_file() or notary_path.is_symlink()
            or _sha_file(notary_path) != state["issuance_notary_sha256"]):
        raise ValueError("K1 final issuance notary changed after matrix completion")
    fingerprints = load_issuance_notary(
        notary_path, expected_study_sha256=spec.sha256())
    expected_run_ids = [str(row["run_id"]) for row in plan["cells"]]
    if set(fingerprints) != set(expected_run_ids):
        raise ValueError("K1 issuance notary does not contain the exact frozen run roster")

    cells_root = run / "cells"
    if cells_root.is_symlink() or not cells_root.is_dir():
        raise ValueError("K1 terminal cell directory is absent or unsafe")
    expected_labels = [_cell_label(row) for row in plan["cells"]]
    actual_entries = list(cells_root.iterdir())
    if (len(set(expected_labels)) != len(expected_labels)
            or {entry.name for entry in actual_entries} != set(expected_labels)
            or any(entry.is_symlink() or not entry.is_dir() for entry in actual_entries)):
        raise ValueError("K1 terminal cell roster is partial, duplicated, extra, or unsafe")

    results: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_cells: set[tuple[Any, ...]] = set()
    terminal_refs: list[dict[str, Any]] = []
    identity_fields = {
        "timestamp", "git_sha", "study_label", "target", "model", "checkpoint",
        "fidelity", "backend", "runtime", "precision", "quantization", "core_count",
    }
    for planned, label in zip(plan["cells"], expected_labels):
        cell_dir = cells_root / label
        cell_entries = {entry.name: entry for entry in cell_dir.iterdir()}
        if (set(cell_entries) != {"terminal-state.json", "result.yaml", "transport"}
                or any(entry.is_symlink() for entry in cell_entries.values())
                or not cell_entries["terminal-state.json"].is_file()
                or not cell_entries["result.yaml"].is_file()
                or not cell_entries["transport"].is_dir()):
            raise ValueError(f"K1 terminal cell {label} has an unexpected top-level artifact")
        terminal = _validate_local_terminal(cell_dir, planned)
        run_id = str(planned["run_id"])
        if terminal["issuance_fingerprint"] != fingerprints[run_id]:
            raise ValueError("K1 terminal issuance differs from the final external notary")
        result_path = cell_dir / "result.yaml"
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("K1 terminal result is not a mapping")
        contract_path = Path(str(planned["contract_root"])) / str(planned["contract_path"])
        contract, contract_cell = _contract(contract_path)
        identity = contract.get("result_identity")
        provenance = result.get("provenance")
        key = (result.get("model"), result.get("backend"), result.get("precision"),
               result.get("core_count"))
        if (not isinstance(identity, Mapping) or set(identity) != identity_fields
                or any(result.get(name) != value for name, value in identity.items())
                or dict(contract_cell) != planned["cell"] or key != (
                    planned["cell"]["model"], planned["cell"]["backend"],
                    planned["cell"]["precision"], planned["cell"]["core_count"])
                or result.get("run_id") != run_id
                or result.get("artifact_sha256") != contract.get("artifact_sha256")
                or result.get("session") != contract.get("session")
                or not isinstance(provenance, Mapping)
                or any(provenance.get(name) != value
                       for name, value in contract.get("frozen_provenance", {}).items())):
            raise ValueError("K1 terminal result identity differs from its frozen contract")
        validate_paper_result(result)
        if run_id in seen_runs or key in seen_cells:
            raise ValueError("K1 terminal results contain duplicate run or cell identities")
        seen_runs.add(run_id)
        seen_cells.add(key)
        results.append(result)
        terminal_refs.append({
            "execution_index": planned["execution_index"], "run_id": run_id,
            "cell": dict(planned["cell"]),
            "terminal_state_sha256": _sha_file(cell_dir / "terminal-state.json"),
            "result_sha256": _sha_file(result_path),
            "issuance_fingerprint": fingerprints[run_id],
        })
    if [str(result["run_id"]) for result in results] != expected_run_ids:
        raise ValueError("K1 terminal results differ from the frozen execution order")

    attach_causal_attribution(spec, results)
    for result in results:
        validate_paper_result(result)
    sealed = seal_results_document(
        spec, results, trusted_issuance_fingerprints=fingerprints)
    # Re-enter the public report API before publication.  This proves the exact document being
    # emitted is acceptable to the same fresh-process verifier used by figures.
    build_paper_report(
        spec, sealed, trusted_issuance_fingerprints=fingerprints)

    unresolved_output = Path(output_path) if output_path is not None else run / "results.yaml"
    if unresolved_output.is_symlink():
        raise ValueError("canonical K1 results output cannot be a symlink")
    output = unresolved_output.resolve()
    finalization_path = output.with_name("results-finalization.json")
    if (output.is_symlink() or output.exists() or finalization_path.exists()
            or finalization_path.is_symlink()):
        raise FileExistsError("canonical K1 results/finalization output already exists")
    results_bytes = yaml.safe_dump(sealed, sort_keys=True).encode("utf-8")
    finalization = {
        "schema_version": 1, "kind": _FINALIZATION_KIND, "status": "complete",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_file_sha256": study_file_sha, "study_sha256": spec.sha256(),
        "plan_file_sha256": plan_file_sha, "matrix_sha256": plan["matrix_sha256"],
        "environment_preflight_sha256": state["environment_preflight_sha256"],
        "issuance_notary_sha256": state["issuance_notary_sha256"],
        "terminal_cells": terminal_refs,
        "results_sha256": _sha_bytes(results_bytes),
        "results_content_seal_sha256": sealed["content_seal"]["seal_sha256"],
    }
    _atomic_json(finalization_path, finalization)
    _atomic_bytes(output, results_bytes)  # canonical completion marker is published last
    return output


def _require_riscv_k1(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, cell = _contract(contract_path)
    if contract["target"] != "k1" or not platform.machine().lower().startswith("riscv"):
        raise ValueError("remote paper cell must execute locally on a RISC-V K1 host")
    _contract_build_tool(contract_path.parent, contract)
    return contract, cell


def _validate_remote_terminal(root: Path, *, contract: Mapping[str, Any],
                              contract_tree_sha256: str,
                              contract_sha256: str) -> dict[str, Any]:
    terminal = _terminal_document(root / "terminal.json")
    if (terminal["run_id"] != contract["run_id"] or
            terminal["contract_tree_sha256"] != contract_tree_sha256 or
            terminal["contract_sha256"] != contract_sha256):
        raise ValueError("existing remote terminal belongs to another frozen contract")
    receipt = _contained(root, terminal["receipt"]["path"], label="remote receipt")
    result = _contained(root, terminal["result"]["path"], label="remote result")
    if (_sha_file(receipt) != terminal["receipt"]["sha256"] or
            _sha_file(result) != terminal["result"]["sha256"] or
            issuance_fingerprint(receipt) != terminal["issuance_fingerprint"]):
        raise ValueError("existing remote terminal evidence changed")
    result_value = yaml.safe_load(result.read_text(encoding="utf-8"))
    if (not isinstance(result_value, Mapping) or
            result_value.get("run_id") != contract["run_id"] or
            any(result_value.get(name) != value for name, value in contract["cell"].items())):
        raise ValueError("existing remote result differs from the frozen contract cell")
    return terminal


def run_remote_cell(contract_path: str | Path, output_root: str | Path, *,
                    contract_tree_sha256: str) -> Path:
    """Board-only systemd payload. Existing valid terminal evidence is never measured again."""
    if not _is_sha256_hex(contract_tree_sha256):
        raise ValueError("remote cell requires the staged contract tree SHA-256")
    contract_unresolved = Path(contract_path)
    if contract_unresolved.is_symlink():
        raise ValueError("remote staged contract cannot be a symlink")
    contract_path = contract_unresolved.resolve()
    contract, cell = _require_riscv_k1(contract_path)
    root_unresolved = Path(output_root)
    pure_root = PurePosixPath(root_unresolved.as_posix())
    if (root_unresolved.is_symlink() or not pure_root.is_absolute() or
            ".." in pure_root.parts or "runs" not in pure_root.parts or len(pure_root.parts) < 5):
        raise ValueError("remote output root must be a deep absolute matrix runs path")
    root = root_unresolved.resolve()
    terminal_path = root / "terminal.json"
    if terminal_path.exists():
        _validate_remote_terminal(
            root, contract=contract, contract_tree_sha256=contract_tree_sha256,
            contract_sha256=_sha_file(contract_path))
        return terminal_path
    if root.exists():
        raise FileExistsError("non-terminal remote cell directory exists; refusing to rerun")
    root.mkdir(parents=True, exist_ok=False)
    receipt = produce_receipt(contract_path, root / "output")
    result = normalize_receipt(receipt)
    if (result.get("run_id") != contract["run_id"] or
            any(result.get(name) != value for name, value in cell.items())):
        raise ValueError("controller-normalized remote result differs from contract cell")
    result_path = root / "result.yaml"
    _atomic_yaml(result_path, result)
    fingerprint = issuance_fingerprint(receipt)
    terminal = {
        "schema_version": 1, "kind": _TERMINAL_KIND, "status": "terminal",
        "run_id": contract["run_id"], "contract_tree_sha256": contract_tree_sha256,
        "contract_sha256": _sha_file(contract_path),
        "receipt": {"path": receipt.relative_to(root).as_posix(), "sha256": _sha_file(receipt)},
        "result": {"path": result_path.relative_to(root).as_posix(),
                   "sha256": _sha_file(result_path)},
        "issuance_fingerprint": fingerprint,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(terminal_path, terminal)
    return terminal_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin-paper-k1-matrix", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser(
        "prepare", help="materialize the frozen matrix contracts without executing them")
    prepare.add_argument("--study", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path)
    plan = actions.add_parser("plan", help="freeze already-materialized contract directories")
    plan.add_argument("--contract", action="append", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--controller-root", type=Path)
    finalize = actions.add_parser(
        "finalize", help="ingest a complete retrieved matrix and seal canonical results")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--study", type=Path, required=True)
    finalize.add_argument("--output", type=Path)
    run = actions.add_parser("run", help="stage, execute, and retrieve the frozen matrix")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--host", required=True)
    run.add_argument("--user", default="root")
    run.add_argument("--port", type=int, default=22)
    run.add_argument("--key", type=Path)
    run.add_argument("--remote-root", default="/var/lib/merlin-paper-k1")
    run.add_argument("--remote-python", default="/usr/bin/python3")
    remote = actions.add_parser("remote-cell", help="board-only systemd payload")
    remote.add_argument("--contract", type=Path, required=True)
    remote.add_argument("--output-root", type=Path, required=True)
    remote.add_argument("--contract-tree-sha256", required=True)
    remote_preflight = actions.add_parser(
        "remote-preflight", help="board-only dependency/environment preflight")
    remote_preflight.add_argument("--output", type=Path, required=True)
    remote_preflight.add_argument("--matrix-sha256", required=True)
    remote_preflight.add_argument("--controller-tree-sha256", required=True)
    remote_preflight.add_argument("--runtime-requirements-sha256", required=True)
    remote_preflight.add_argument("--required-core-count", type=int, required=True)
    remote_preflight.add_argument("--expected-python", required=True)
    arguments = parser.parse_args(argv)
    if arguments.action == "prepare":
        print(prepare_contract_matrix(arguments.study, output_dir=arguments.output_dir))
        return 0
    if arguments.action == "plan":
        print(create_matrix_plan(
            arguments.contract, arguments.output, controller_root=arguments.controller_root))
        return 0
    if arguments.action == "finalize":
        print(finalize_matrix(
            arguments.plan, arguments.run_dir, arguments.study,
            output_path=arguments.output))
        return 0
    if arguments.action == "remote-cell":
        print(run_remote_cell(
            arguments.contract, arguments.output_root,
            contract_tree_sha256=arguments.contract_tree_sha256))
        return 0
    if arguments.action == "remote-preflight":
        print(create_remote_environment_receipt(
            arguments.output, matrix_sha256=arguments.matrix_sha256,
            controller_tree_sha256=arguments.controller_tree_sha256,
            runtime_requirements_sha256=arguments.runtime_requirements_sha256,
            required_core_count=arguments.required_core_count,
            expected_python=arguments.expected_python))
        return 0
    config = SSHConfig(
        host=arguments.host, user=arguments.user, port=arguments.port, key=arguments.key,
        remote_root=arguments.remote_root, remote_python=arguments.remote_python)
    transport = K1SSHSystemdTransport(config)
    print(run_matrix(
        arguments.plan, transport=transport, output_dir=arguments.output_dir,
        resume=arguments.resume, remote_root=arguments.remote_root))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
