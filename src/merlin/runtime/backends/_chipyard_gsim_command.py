"""Trusted, non-executing GSIM command preparation for a bounded sandbox runner.

Describes how to execute an already compiled short ELF on a backend's pinned GSIM engine. It neither
admits the workload or wall time nor grants paths to a sandbox: the caller validates the program's
scope, provides an approved environment, kills the whole process group on deadline, then
revalidates the descriptor. Parameterized by the BACKEND module that owns the engine, so no target
or engine path is named here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from merlin.targetgen import gsim_emulator


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _document(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require(condition, message) -> None:
    if not condition:
        raise ValueError(message)


def _sha_value(value) -> bool:
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def current_engine(backend: ModuleType) -> dict:
    """The backend's GSIM engine citation, as the shared emulator resolver states it."""
    return gsim_emulator.citation(backend.TARGET_NAME, env_var=backend.GSIM_EMU_ENV)


@dataclass(frozen=True)
class GsimCommand:
    """``argv`` suits the existing argv-only native-probe boundary.

    Its isolated Python prelude applies the backend's stack policy then *execs* the emulator: it
    creates no process group and swallows no exit status. ``emulator_argv`` is the backend's exact
    GSIM vector. Required paths are obligations, not mount or execution authorization.
    """

    argv: tuple[str, ...]
    emulator_argv: tuple[str, ...]
    env_overrides: tuple[tuple[str, str], ...]
    required_readonly_paths: tuple[str, ...]
    evidence_json: str
    backend: ModuleType

    def to_evidence(self) -> dict:
        return json.loads(self.evidence_json)

    def sandbox_dependency_fields(self) -> dict:
        """The one-engine capability as plain fields (for the optional sandbox's
        ``HostExecutableDependencies``); the enclosing host still admits the workload."""
        self.revalidate()
        evidence = self.to_evidence()
        return {
            "argv": self.argv,
            "executable_path": self.emulator_argv[0],
            "artifact_path": self.emulator_argv[1],
            "engine_receipt_path": evidence["engine_provenance"]["receipt"]["receipt_path"],
            "file_pins": tuple(evidence["file_pins"].items()),
            "command_revalidator": self.revalidate,
        }

    def revalidate(self) -> dict:
        """Call immediately before and after the bounded runner consumes argv."""
        evidence = self.to_evidence()
        _require(
            _document(current_engine(self.backend)) == _document(evidence["engine_provenance"]),
            "GSIM engine provenance changed after command preparation",
        )
        for path, digest in evidence["file_pins"].items():
            _require(_sha(path) == digest, "GSIM command input/tool/source pin changed: " + path)
        _require(
            tuple(evidence["argv"]) == self.argv and tuple(evidence["emulator_argv"]) == self.emulator_argv,
            "GSIM command vector differs from its evidence",
        )
        return {
            "status": "unchanged",
            "executed": False,
            "command_sha256": hashlib.sha256(self.evidence_json.encode()).hexdigest(),
        }


def prepare_gsim_command(
    backend: ModuleType,
    elf,
    *,
    expected_elf_sha256: str,
    expected_engine_provenance: Mapping,
    max_cycles: int | None = None,
    python_executable: str | Path | None = None,
) -> GsimCommand:
    """Resolve and pin only; never run GSIM or widen the existing sandbox.

    ``max_cycles`` is a host-selected hang bound, not a runtime estimate. Omitting it keeps the
    backend's configured plusarg. The host runner keeps a separate hard wall-clock deadline.
    """
    _require(_sha_value(expected_elf_sha256), "explicit expected ELF SHA256 required")
    _require(
        isinstance(expected_engine_provenance, Mapping) and expected_engine_provenance,
        "explicit expected engine provenance required",
    )
    path = Path(elf).absolute()
    _require(path.is_file() and _sha(path) == expected_elf_sha256, "ELF differs from expected bytes")
    engine = current_engine(backend)
    _require(
        engine.get("available") is True
        and engine.get("refused") is False
        and engine.get("flavour") == "binary"
        and engine.get("receipt_status") == "bound",
        "GSIM requires an available binary with a bound build receipt",
    )
    _require(_document(engine) == _document(dict(expected_engine_provenance)), "unexpected GSIM engine provenance")
    configured = backend.gsim_max_cycles() if max_cycles is None else max_cycles
    if isinstance(configured, str):
        _require(configured.isdecimal(), "GSIM max cycles must be a positive decimal integer")
        configured = int(configured)
    _require(
        type(configured) is int and 0 < configured < 1 << 63, "GSIM max cycles must fit a positive bounded host integer"
    )
    direct = tuple(backend._gsim_argv(path, max_cycles=configured))
    emulator = Path(direct[0])
    _require(
        emulator.is_absolute() and emulator.is_file() and os.access(emulator, os.X_OK),
        "GSIM command requires an absolute executable engine path",
    )
    _require(
        str(emulator) == engine["path"] and _sha(emulator) == engine["binary_sha256"],
        "GSIM argument vector does not name the pinned engine bytes",
    )
    interpreter = Path(python_executable or sys.executable)
    _require(
        interpreter.is_absolute() and interpreter.is_file() and os.access(interpreter, os.X_OK),
        "stack prelude requires an absolute host-approved Python executable",
    )
    # -I -S excludes candidate cwd/PYTHONPATH/site hooks; imports use only the approved interpreter.
    interpreter = interpreter.resolve()
    stack_source = inspect.getsource(backend._unlimited_stack)
    prelude = (
        "import os, resource, sys\n" + stack_source + "\n_unlimited_stack()\nos.execv(sys.argv[1], sys.argv[1:])\n"
    )
    argv = (str(interpreter), "-I", "-S", "-c", prelude, *direct)
    receipt = Path(engine["receipt"]["receipt_path"])
    _require(
        receipt.is_file() and _sha(receipt) == engine["receipt"]["receipt_sha256"],
        "GSIM build receipt changed during command preparation",
    )
    paths = (
        path,
        emulator,
        receipt,
        interpreter,
        Path(__file__).resolve(),
        Path(backend.__file__).resolve(),
        Path(gsim_emulator.__file__).resolve(),
    )
    pins = {str(item): _sha(item) for item in paths}
    required = tuple(dict.fromkeys((str(path), str(emulator), str(interpreter))))
    evidence = {
        "schema": "trusted_gsim_command_v1",
        "argv": list(argv),
        "emulator_argv": list(direct),
        "engine_provenance": engine,
        "elf_sha256": expected_elf_sha256,
        "max_cycles": configured,
        "max_cycles_source": "backend_configuration" if max_cycles is None else "explicit_host_hang_bound",
        "file_pins": pins,
        "stack_prelude_sha256": hashlib.sha256(prelude.encode()).hexdigest(),
        "environment_overrides": {},
        "environment_scope": "inherit existing approved sandbox environment only",
        "interpreter_mode": "isolated, no site initialization",
        "required_readonly_paths": list(required),
        "runtime_loader_and_libraries": "must already be available in approved sandbox; not granted here",
        "mount_authorization": False,
        "wall_time_admission": False,
        "workload_scope_admitted": False,
        "warm_protocol_verified": False,
        "target_executed": False,
        "expected_returncode": 0,
        "process_group_policy": "exec preserves the bounded parent runner's killable process group",
        "scope": "exact trusted command construction only; no correctness or timing verdict",
    }
    result = GsimCommand(argv, direct, (), required, _document(evidence), backend)
    result.revalidate()
    return result
