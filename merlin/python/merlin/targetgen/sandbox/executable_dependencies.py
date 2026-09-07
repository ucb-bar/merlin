"""One exact trusted raw-engine grant, not a general executable/mount API."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

from .build_dependencies import _hash, _path, _overlap, _coverage_gaps


@dataclass(frozen=True)
class HostExecutableDependencies:
    """Host target adapter binds an exact command to an engine and workload.

    Only ``executable_path`` is granted, read-only at its identical path. All
    other pins (ELF, engine receipt/configuration, interpreter and adapter code)
    are revalidation obligations, never implicit grants. The enclosing provider
    owns bounded-work admission, warm/correctness checks and iteration charging.
    """
    argv: tuple[str, ...]
    executable_path: str
    artifact_path: str
    engine_receipt_path: str
    file_pins: tuple[tuple[str, str], ...]
    command_revalidator: Callable

    def revalidate(self, argv):
        if (type(self) is not HostExecutableDependencies or tuple(map(str, argv)) != self.argv
                or not self.argv or not callable(self.command_revalidator)):
            raise ValueError("runtime capability belongs to another trusted command")
        pins = dict(self.file_pins)
        if len(pins) != len(self.file_pins):
            raise ValueError("runtime command pins must be unique")
        required = (self.argv[0], self.executable_path, self.artifact_path, self.engine_receipt_path)
        if any(path not in pins for path in required):
            raise ValueError("runtime command lacks explicit engine/ELF/receipt/interpreter pins")
        if self.executable_path not in self.argv or self.artifact_path not in self.argv:
            raise ValueError("runtime argv does not consume its exact engine and artifact")
        for name, expected in self.file_pins:
            path = Path(name)
            if not path.is_absolute() or not path.is_file() or _hash(path) != expected:
                raise ValueError("runtime input/tool/configuration pin changed: " + name)
        engine = _path(self.executable_path)
        if not os.access(engine, os.X_OK):
            raise ValueError("runtime engine is not executable")
        for path in (engine, Path(self.artifact_path)):
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError("raw runtime engine and target artifact must be ELF files")
        self.command_revalidator()

    def extend(self, sandbox, argv):
        from .answer_surfaces import AnswerSurface
        self.revalidate(argv)
        prefix = list(sandbox["command_prefix"])
        if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
            raise ValueError("runtime engine needs the existing clear-environment policy")
        surfaces = [AnswerSurface(str(row.get("label", "answer")), _path(row["path"]),
                    row["kind"], str(row.get("origin", "oracle"))) for row in sandbox["answer_surfaces"]]
        engine = _path(self.executable_path)
        if not surfaces or _coverage_gaps(prefix, surfaces):
            raise ValueError("runtime policy has missing or exposed answer masks")
        if any(_overlap(engine, surface.path) for surface in surfaces):
            raise ValueError("runtime engine grant overlaps an answer surface")
        masks = {str(surface.path) for surface in surfaces}
        insertion = next((i for i, flag in enumerate(prefix)
            if (flag == "--tmpfs" and i+1 < len(prefix) and prefix[i+1] in masks)
            or (flag == "--ro-bind" and i+2 < len(prefix) and prefix[i+1] == "/dev/null"
                and prefix[i+2] in masks)), None)
        if insertion is None:
            raise ValueError("runtime policy has no explicit answer-mask boundary")
        result = [*prefix[:insertion], "--ro-bind", str(engine), str(engine), *prefix[insertion:]]
        if _coverage_gaps(result, surfaces):
            raise ValueError("runtime executable grant exposed an answer")
        self.revalidate(argv)
        return result
