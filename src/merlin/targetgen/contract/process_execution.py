"""Explicit native process consumption without simulator or runtime authority.

The fixed owner invokes the selected tool itself and joins its requested artifact,
argv and captured stream to the actual closed invocation. A matching command
operand is not proof of loader semantics, complete dependencies or isolation.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.common import invocation_record as I

from .build_service import file_digest
from .prepared_process_readback import OPERANDS, PreparedProcessReadbackPlan, validate_request

_EXECUTED = WeakKeyDictionary()
_ELF = "{elf}"
_SCOPE = (
    "actual direct tool/argv/artifact/raw-stream attribution only; loader/ISA semantics, "
    "transitive dependencies, isolation, runtime/effects/hardware/timing unqualified"
)


def _plain(path):
    if not isinstance(path, Path) or not path.is_absolute() or path.resolve() != path:
        raise ValueError("process consumption requires a canonical explicit path")
    if any(member.is_symlink() for member in (path, *path.parents)):
        raise ValueError("process consumption cannot follow linked members")
    return path


def _pin(path):
    path = _plain(path)
    if not path.is_file():
        raise ValueError("process consumption requires a regular selected member")
    return {"path": str(path), "sha256": file_digest(path)}


@dataclass(frozen=True)
class _Executed:
    record: Path
    record_sha256: str
    elf_path: Path
    elf_sha256: str
    selection_json: str
    request: dict | None
    request_json: str | None
    input_pins: tuple
    argv: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class RecordedProcessExecution:
    """An explicit process choice; no callback, saved receipt or default engine."""

    executable: Path
    argv_template: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]
    record_root: Path
    stream: str
    source_pins: tuple[tuple[str, str], ...]
    prepared_readback: PreparedProcessReadbackPlan | None = None

    def verify(self):
        if (
            type(self) is not RecordedProcessExecution
            or type(self.argv_template) is not tuple
            or not self.argv_template
            or self.argv_template.count(_ELF) != 1
            or self.prepared_readback is not None
            and type(self.prepared_readback) is not PreparedProcessReadbackPlan
            or any(
                type(token) is not str
                or "\0" in token
                or token not in (_ELF, *OPERANDS)
                and ("{" in token or "}" in token)
                for token in self.argv_template
            )
            or any(
                self.argv_template.count(token) != (1 if self.prepared_readback is not None else 0)
                for token in OPERANDS
            )
            or type(self.environment) is not tuple
            or any(
                type(row) is not tuple or len(row) != 2 or any(type(v) is not str for v in row)
                for row in self.environment
            )
            or len(dict(self.environment)) != len(self.environment)
            or self.stream not in {"stdout", "combined"}
            or type(self.source_pins) is not tuple
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("recorded process requires an exact explicit command/environment/stream selection")
        pins = dict(self.source_pins)
        owner = Path(__file__).resolve()
        if str(self.executable) not in pins or str(owner) not in pins:
            raise ValueError("recorded process omits its selected tool or fixed source owner")
        executable = _pin(self.executable)
        if not os.access(self.executable, os.X_OK) or not _plain(self.cwd).is_dir():
            raise ValueError("recorded process selected tool or working directory is unavailable")
        root = _plain(self.record_root)
        if root.exists() and (not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077):
            raise ValueError("recorded process evidence root must be private and owned")
        required = {str(self.executable): executable["sha256"], str(owner): file_digest(owner)}
        if self.prepared_readback is not None:
            required.update((str(path), file_digest(path)) for path in self.prepared_readback.source_paths())
        if any(pins.get(path) != digest for path, digest in required.items()):
            raise ValueError("recorded process omits its selected tool or fixed source owner")
        for path, expected in self.source_pins:
            if _pin(Path(path))["sha256"] != expected:
                raise ValueError("recorded process selected source/tool changed")
        record = {
            "executable": executable,
            "argv_template": list(self.argv_template),
            "cwd": str(self.cwd),
            "environment": I.environment_identity(dict(self.environment)),
            "record_root": str(root),
            "stream": self.stream,
            "source_pins": [{"path": path, "sha256": digest} for path, digest in self.source_pins],
            "scope": _SCOPE,
        }
        if self.prepared_readback is not None:
            record["prepared_readback"] = self.prepared_readback.record()
        return record

    def run_elf(self, elf, *, timeout, capture_bytes=False, **kwargs):
        kwargs.pop("simulator", None)  # Service verifies its selected identity; this owner never selects by name.
        request = kwargs.pop("memory_readback", None) if self.prepared_readback is not None else None
        if kwargs or type(capture_bytes) is not bool or type(timeout) not in (int, float):
            raise ValueError("recorded process received unsupported invocation options")
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("recorded process requires a bounded native deadline")
        selection = self.verify()
        artifact = _pin(Path(elf))
        _EXECUTED.pop(self, None)
        inputs, outputs = (Path(elf),), ()
        operands = {_ELF: str(elf)}
        request_json = None
        if self.prepared_readback is not None:
            _, paths = validate_request(self.prepared_readback, request, Path(elf), completed=False)
            inputs += paths[:2]
            outputs = paths[2:]
            operands.update(zip(OPERANDS, (str(paths[1]), str(paths[2])), strict=True))
            request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
        input_pins = tuple(sorted((_pin(path) for path in inputs), key=lambda row: Path(row["path"])))
        if not self.record_root.exists():
            self.record_root.mkdir(mode=0o700)
        self.verify()
        call = self.record_root / uuid.uuid4().hex
        call.mkdir(mode=0o700)
        argv = [str(self.executable), *(operands.get(token, token) for token in self.argv_template)]
        result = I.run(
            argv,
            directory=call,
            stage="recorded_functional_process",
            cwd=self.cwd,
            env=dict(self.environment),
            inputs=inputs,
            outputs=outputs,
            dependencies=tuple(Path(path) for path, _ in self.source_pins),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if self.stream == "combined" else subprocess.PIPE,
            timeout=timeout,
        )
        result.check_returncode()
        records = tuple(call.glob("invocations/*/invocation.json"))
        if len(records) != 1:
            raise ValueError("recorded process omitted its unique actual native invocation")
        _EXECUTED[self] = _Executed(
            records[0],
            file_digest(records[0]),
            Path(elf),
            artifact["sha256"],
            json.dumps(selection, sort_keys=True, separators=(",", ":")),
            request,
            request_json,
            input_pins,
            tuple(argv),
        )
        console = result.stdout if capture_bytes else result.stdout.decode("utf-8")
        self.consumption(elf=Path(elf), console=console)
        return console

    def consumption(self, *, elf, console):
        """Reopen this owner's actual process, never a supplied receipt or PASS."""
        executed = _EXECUTED.get(self)
        if executed is None:
            raise ValueError("recorded process has no actual completed native execution")
        selection = self.verify()
        if json.dumps(selection, sort_keys=True, separators=(",", ":")) != executed.selection_json:
            raise ValueError("recorded process selection changed after native execution")
        expected = {"path": str(executed.elf_path), "sha256": executed.elf_sha256}
        if Path(elf) != executed.elf_path or _pin(Path(elf)) != expected:
            raise ValueError("recorded process consumed a different or changed requested ELF")
        record_pin = _pin(executed.record)
        if record_pin["sha256"] != executed.record_sha256 or not executed.record.is_relative_to(self.record_root):
            raise ValueError("recorded process actual invocation changed or escaped")
        prepared = None
        if self.prepared_readback is not None:
            if json.dumps(executed.request, sort_keys=True, separators=(",", ":")) != executed.request_json:
                raise ValueError("recorded process prepared request changed after native execution")
            wire, paths = validate_request(self.prepared_readback, executed.request, Path(elf), completed=True)
            prepared = {
                "request": _pin(paths[1]),
                "output": _pin(paths[2]),
                "objects": wire["objects"],
                "payload_bytes": wire["payload_bytes"],
                "product_bytes": wire["product_bytes"],
                "scope": wire["plan"]["scope"],
            }
        observed = I.verify(executed.record)
        if (
            observed["kind"] != "subprocess"
            or observed["argv"] != list(executed.argv)
            or observed["inputs"] != list(executed.input_pins)
            or observed["outputs"] != ([] if prepared is None else [prepared["output"]])
            or observed["executable"] != selection["executable"]
            or observed["cwd"] != str(self.cwd)
            or observed["dependencies"] != sorted(selection["source_pins"], key=lambda row: Path(row["path"]))
        ):
            raise ValueError("recorded process actual tool/argv/input differs from its original selection")
        I.require_environment(executed.record, environment=dict(self.environment))
        stdout, stderr = (_plain(executed.record.parent / name) for name in ("stdout.bin", "stderr.bin"))
        if observed["stdout"] != _pin(stdout) or observed["stderr"] != _pin(stderr):
            raise ValueError("recorded process actual captured stream changed")
        data = console if type(console) is bytes else console.encode("utf-8") if type(console) is str else None
        if data is None or data != stdout.read_bytes() or self.stream == "combined" and stderr.read_bytes():
            raise ValueError("recorded process returned console differs from its actual selected captured stream")
        result = {
            "record": record_pin,
            "elf": expected,
            "executable": selection["executable"],
            "stdout": observed["stdout"],
            "stderr": observed["stderr"],
            "stream": self.stream,
            "scope": _SCOPE,
        }
        if prepared is not None:
            result["prepared_readback"] = prepared
        return result
