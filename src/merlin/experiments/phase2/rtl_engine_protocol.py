"""Private, source-pinned minimal RTL controls; never target cycle authority.

Declarations select actual subprocesses and original complete outputs. They are
not receipts or capabilities. The evaluator executes them before issuing a live
diagnostic admission, which remains separate from a qualified target runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import StageGateError, document_sha256, sha256_file

RTL_CONTRACTS = ("input_grammar", "memory_semantics", "clock_reset", "program_loading")


def canonical_file(value):
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file():
        raise StageGateError("RTL engine selections require canonical regular files")
    return path


def relative_member(value):
    if type(value) is not str:
        raise StageGateError("RTL probe member requires a relative string path")
    path = Path(value)
    if not value or path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise StageGateError("RTL probe member escapes its private workspace")
    if str(path) != value:
        raise StageGateError("RTL probe member is not canonical")
    return path


@dataclass(frozen=True)
class RtlEngineSelection:
    tools: tuple[Path, ...]
    source_files: tuple[Path, ...]
    dependencies: tuple[Path, ...]
    pins: tuple[tuple[Path, str], ...]
    producer_records: tuple[tuple[Path, str], ...]

    def verify(self):
        from merlin.common import invocation_record

        for path, digest in (*self.pins, *self.producer_records):
            if sha256_file(canonical_file(path)) != digest:
                raise StageGateError("RTL engine source, tool or producer changed")
        for path, _ in self.producer_records:
            invocation_record.verify(path)
        return self.sha256

    @property
    def sha256(self):
        return document_sha256({
            "tools": [str(path) for path in self.tools],
            "sources": [str(path) for path in self.source_files],
            "dependencies": [str(path) for path in self.dependencies],
            "pins": [(str(path), digest) for path, digest in self.pins],
            "producer_records": [(str(path), digest) for path, digest in self.producer_records],
        })


def prepare_rtl_engine_selection(*, tools, source_files, dependencies=(), producer_records=()):
    """Pin explicitly selected support. Build records do not prove RTL equivalence."""
    tools = tuple(canonical_file(path) for path in tools)
    sources = tuple(canonical_file(path) for path in source_files)
    dependencies = tuple(canonical_file(path) for path in dependencies)
    records = tuple(canonical_file(path) for path in producer_records)
    if not tools or not sources or len(set(tools)) != len(tools):
        raise StageGateError("RTL engine probes need explicit tools and independent support sources")
    selection = RtlEngineSelection(
        tools, sources, dependencies,
        tuple((path, sha256_file(path)) for path in sorted(set((*tools, *sources, *dependencies)))),
        tuple((path, sha256_file(path)) for path in records),
    )
    selection.verify()
    return selection


@dataclass(frozen=True)
class RtlProbeStep:
    id: str
    argv: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    timeout_seconds: float = 30
    output_directories: tuple[str, ...] = ()


@dataclass(frozen=True)
class RtlProbeControl:
    id: str
    contract: str
    files: tuple[tuple[str, bytes], ...]
    steps: tuple[RtlProbeStep, ...]
    output_columns: tuple[str, ...] = ()
    expected_rows: tuple[tuple[int, ...], ...] = ()

    def verify(self):
        if (type(self.id) is not str or self.contract not in RTL_CONTRACTS
            or not self.id or not self.id.replace("_", "").isalnum()
            or type(self.files) is not tuple or type(self.steps) is not tuple or not self.files or not self.steps
            or any(type(step) is not RtlProbeStep for step in self.steps)
            or len({step.id for step in self.steps}) != len(self.steps)):
            raise StageGateError("RTL probe control has an incomplete or duplicate declaration")
        names = [name for name, _ in self.files]
        if len(set(names)) != len(names):
            raise StageGateError("RTL probe source members are duplicated")
        for name, payload in self.files:
            relative_member(name)
            if type(payload) is not bytes:
                raise StageGateError("RTL probe controls retain exact original source bytes")
        for step in self.steps:
            if (type(step.id) is not str or not step.id or not step.id.replace("_", "").isalnum()
                or type(step.argv) is not tuple or not step.argv
                or any(type(value) is not tuple for value in (step.inputs, step.outputs, step.output_directories))
                or type(step.timeout_seconds) not in (int, float) or not 0 < step.timeout_seconds <= 60):
                raise StageGateError("RTL probe command or bounded deadline is unavailable")
            for member in (*step.inputs, *step.outputs):
                relative_member(member)
            if not set(step.output_directories) <= set(step.outputs):
                raise StageGateError("RTL probe output directories require declared product ownership")
            if any(type(arg) is not str or not arg or "\x00" in arg for arg in step.argv):
                raise StageGateError("RTL probe command requires exact argv tokens")
        if type(self.output_columns) is not tuple or type(self.expected_rows) is not tuple:
            raise StageGateError("RTL probe original output roster requires closed tuples")
        if self.contract != "input_grammar" and not self.expected_rows:
            raise StageGateError("semantic RTL controls require complete original numeric outputs")
        if self.expected_rows:
            if (not self.output_columns or len(set(self.output_columns)) != len(self.output_columns)
                or any(type(name) is not str or not name or "," in name or "\n" in name for name in self.output_columns)
                or any(type(row) is not tuple or len(row) != len(self.output_columns)
                       or any(type(item) is not int for item in row)
                       for row in self.expected_rows)):
                raise StageGateError("RTL probe original output roster is incomplete")
        elif self.output_columns:
            raise StageGateError("RTL probe output roster has no original samples")
        return self.sha256

    @property
    def sha256(self):
        import hashlib

        return document_sha256({
            "id": self.id, "contract": self.contract,
            "sources": [(name, hashlib.sha256(data).hexdigest()) for name, data in self.files],
            "steps": [{"id": step.id, "argv": step.argv, "inputs": step.inputs,
                       "outputs": step.outputs, "timeout_seconds": step.timeout_seconds,
                       "output_directories": step.output_directories} for step in self.steps],
            "output_columns": self.output_columns, "expected_rows": self.expected_rows,
        })


def check_original_outputs(stdout, control):
    """Check every declared sample/column; reject labels, missing and extra data."""
    try:
        lines = stdout.decode("ascii").splitlines()
        if not lines or tuple(lines[0].split(",")) != control.output_columns:
            raise ValueError("original output roster differs")
        rows = tuple(tuple(int(item) for item in line.split(",")) for line in lines[1:])
        if rows != control.expected_rows:
            raise ValueError("complete original samples differ")
    except (UnicodeError, ValueError) as error:
        raise StageGateError("RTL probe complete original output check failed") from error
    return rows
