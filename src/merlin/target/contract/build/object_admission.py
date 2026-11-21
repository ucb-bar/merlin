"""Explicit source-bound linked-artifact checks before simulator dispatch.

This transport validates unchanged artifact/report bytes and callback ownership.
Its caller supplies the independently evaluated policy. It grants no physical,
instruction-effect, numerical or timing authority and never resolves a backend.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import digest as _mdigest
from merlin.common.record_formats import loads
from merlin.target.contract.build.execution_service import _callback_selection, _same_callback
from merlin.target.contract.build.services import file_digest

# ---- elf_admission -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkedElfAdmissionService:
    target: str
    evaluator: Callable
    source_pins: tuple[tuple[str, str], ...]
    prerequisite: LinkedElfAdmissionService | None = None
    result_validator: Callable | None = None
    _evaluator_selection: tuple | None = field(init=False, repr=False, compare=False)
    _prerequisite_selection: object = field(init=False, repr=False, compare=False)
    _result_validator_selection: tuple | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_evaluator_selection", _callback_selection(self.evaluator))
        object.__setattr__(self, "_prerequisite_selection", self.prerequisite)
        object.__setattr__(self, "_result_validator_selection", _callback_selection(self.result_validator))

    def verify(self, target):
        return self._verify(target, ())

    def _verify(self, target, ancestors):
        if id(self) in ancestors or len(ancestors) >= 32:
            raise ValueError("linked ELF admission prerequisite chain is cyclic or exceeds 32 services")
        if self.prerequisite is not self._prerequisite_selection:
            raise ValueError("linked ELF admission selected prerequisite changed")
        if (
            type(self) is not LinkedElfAdmissionService
            or target != self.target
            or type(self.source_pins) is not tuple
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("linked ELF admission requires an exact target and source selection")
        for path, digest in self.source_pins:
            member = Path(path)
            if (
                not member.is_absolute()
                or any(parent.is_symlink() for parent in (member, *member.parents))
                or not member.is_file()
                or file_digest(member) != digest
            ):
                raise ValueError("linked ELF admission source pin changed: " + str(path))
        for label, callback, expected in (
            ("evaluator", self.evaluator, self._evaluator_selection),
            ("result validator", self.result_validator, self._result_validator_selection),
        ):
            if label == "result validator" and callback is None and expected is None:
                continue
            current = _callback_selection(callback)
            if current is None:
                raise ValueError(f"linked ELF admission {label} must be an actual Python function or bound method")
            if not _same_callback(current, expected):
                raise ValueError(f"linked ELF admission selected {label} implementation or partial bindings changed")
            owner = inspect.getsourcefile(current[1])
            if owner is None or (str(Path(owner).resolve()), file_digest(Path(owner))) not in self.source_pins:
                raise ValueError(f"linked ELF admission {label} has no pinned inspected source owner")
        selection = {"target": target, "source_pins": self.source_pins}
        if self.prerequisite is not None:
            if type(self.prerequisite) is not LinkedElfAdmissionService:
                raise ValueError("linked ELF admission prerequisite must be an exact selected service")
            selection["prerequisite"] = self.prerequisite._verify(target, (*ancestors, id(self)))
        return selection

    def evaluate(self, *, elf, target, evidence_root):
        before = self.verify(target)
        artifact = Path(elf)
        kwargs = {}
        if self.prerequisite is not None:
            Path(evidence_root).mkdir(parents=True, exist_ok=True)
            parent = self.prerequisite.evaluate(
                elf=artifact,
                target=target,
                evidence_root=Path(evidence_root) / "prerequisite",
            )
            kwargs["prerequisite_result"] = _receipt(parent)
            digest = parent["elf_sha256"]
        else:
            digest = file_digest(artifact)
        prerequisite_receipt = _receipt(kwargs["prerequisite_result"]) if kwargs else None
        result = self.evaluator(elf=artifact, evidence_root=Path(evidence_root), **kwargs)
        if type(result) is not dict or result.get("status") not in ("accepted", "refused"):
            raise ValueError("linked ELF admission returned no actual accepted/refused observation")
        if self.verify(target) != before:
            raise ValueError("linked ELF admission selection changed during evaluation")
        if kwargs and result.get("prerequisite") != prerequisite_receipt:
            raise ValueError("linked ELF admission omitted or replaced its actual prerequisite receipt")
        self.revalidate(elf=artifact, result=result, target=target)
        if result["elf_sha256"] != digest:
            raise ValueError("linked ELF changed during admission evaluation")
        return result

    def revalidate(self, *, elf, result, target):
        self.verify(target)
        self._revalidate_report(result, target)
        self.verify(target)
        # One final streamed ELF read for the whole chain. Reports and source
        # pins are reopened on every call; no verdict or digest is cached.
        if file_digest(Path(elf)) != result.get("elf_sha256"):
            raise ValueError("linked ELF changed after admission")
        return result["status"]

    def _revalidate_report(self, result, target):
        if type(result) is not dict or result.get("status") not in ("accepted", "refused"):
            raise ValueError("linked ELF admission result has no accepted/refused status")
        report = Path(result.get("report_path", ""))
        if (
            not report.is_absolute()
            or any(parent.is_symlink() for parent in (report, *report.parents))
            or not report.is_file()
            or file_digest(report) != result.get("report_sha256")
        ):
            raise ValueError("linked ELF admission report has no unchanged actual file")
        stored = loads(report.read_text())
        if type(stored) is not dict or any(stored.get(key) != result.get(key) for key in ("status", "elf_sha256")):
            raise ValueError("linked ELF admission report disagrees with its evaluated result")
        if self.prerequisite is not None:
            parent = result.get("prerequisite")
            if type(parent) is not dict or stored.get("prerequisite") != parent:
                raise ValueError("linked ELF admission report lost its prerequisite receipt")
            self.prerequisite._revalidate_report(parent, target)
            if parent["elf_sha256"] != result["elf_sha256"]:
                raise ValueError("linked ELF admission prerequisite checked different ELF bytes")
            if parent["status"] == "refused" and result["status"] != "refused":
                raise ValueError("linked ELF admission cannot override a refused prerequisite")
        if self.result_validator is not None:
            # Policy-specific context is checked against the authenticated file,
            # including when this service is consumed through a stripped parent
            # receipt. Unknown evaluator state is never inspected here.
            validator = self.result_validator
            if not _same_callback(_callback_selection(validator), self._result_validator_selection):
                raise ValueError("linked ELF admission result validator changed during validation")
            validator(stored)


def _receipt(result, depth=0):
    """Retain custody references, never copy a prerequisite's protected findings."""
    if depth >= 32:
        raise ValueError("linked ELF admission receipt exceeds 32 services")
    receipt = {key: result[key] for key in ("status", "elf_sha256", "report_path", "report_sha256")}
    if "prerequisite" in result:
        receipt["prerequisite"] = _receipt(result["prerequisite"], depth + 1)
    return receipt


# ---- stack_usage -------------------------------------------------------------------------------
# Fail-closed parsing and recording of compiler-produced static stack-frame measurements.


class StackFramePreflightError(ValueError):
    """A compiler stack report cannot prove that the declared entrypoint fits its budget."""

    def __init__(self, message: str, *, measurement: StackFrameMeasurement | None = None):
        super().__init__(message)
        self.measurement = measurement


@dataclass(frozen=True)
class StackUsageRow:
    """One strict ``-fstack-usage`` row."""

    source_identity: str
    function: str
    frame_bytes: int
    allocation: str


@dataclass(frozen=True)
class StackFrameMeasurement:
    """The admitted entrypoint frame and the policy it was checked against."""

    entry_symbol: str
    frame_bytes: int
    max_static_bytes: int
    report_rows: int

    @property
    def headroom_bytes(self) -> int:
        return self.max_static_bytes - self.frame_bytes


_sha256 = _mdigest.sha256_file


def parse_stack_usage(text: str) -> tuple[StackUsageRow, ...]:
    """Parse clang's tab-separated report, rejecting every ambiguous or malformed row.

    The identity field is intentionally split from the right: source paths may contain colons, while
    the final component is the C ABI symbol clang reports.  No partial report is accepted.  If a
    compiler changes this format, the build stops with an actionable diagnostic rather than silently
    disabling the safety gate.
    """
    if not isinstance(text, str):
        raise StackFramePreflightError("stack-usage report is not text")
    rows: list[StackUsageRow] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise StackFramePreflightError(f"stack-usage report line {number} does not have three tab-separated fields")
        identity, byte_text, allocation = fields
        source_identity, separator, function = identity.rpartition(":")
        if not separator or not source_identity or not function:
            raise StackFramePreflightError(f"stack-usage report line {number} has no source/function identity")
        if not byte_text.isascii() or not byte_text.isdecimal():
            raise StackFramePreflightError(f"stack-usage report line {number} has a non-decimal frame size")
        frame_bytes = int(byte_text)
        if allocation not in {"static", "dynamic", "dynamic,bounded"}:
            raise StackFramePreflightError(
                f"stack-usage report line {number} has unknown allocation kind {allocation!r}"
            )
        rows.append(StackUsageRow(source_identity, function, frame_bytes, allocation))
    if not rows:
        raise StackFramePreflightError("stack-usage report is empty")
    return tuple(rows)


def measure_entrypoint(
    report_path: Path, *, llvm_path: Path, entry_symbol: str, max_static_bytes: int
) -> StackFrameMeasurement:
    """Require one fresh, source-bound, fully static entrypoint row within the target's budget."""
    report_path, llvm_path = Path(report_path), Path(llvm_path)
    measurement_view = report_path.with_suffix(".measured.su")
    measurement_view.unlink(missing_ok=True)
    if report_path.is_symlink() or not report_path.is_file():
        raise StackFramePreflightError(f"compiler produced no regular stack-usage report at {report_path}")
    try:
        rows = parse_stack_usage(report_path.read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise StackFramePreflightError("stack-usage report is not valid UTF-8") from exc
    for row in rows:
        if Path(row.source_identity).name != llvm_path.name:
            raise StackFramePreflightError(
                f"stack row for {row.function!r} names source {row.source_identity!r}, expected {llvm_path.name!r}"
            )
    # Keep the actual compiler product byte-identical to its completed invocation.
    # Canonical path spelling belongs to a separate measurement view. Counts,
    # classifications, symbols and row order remain exactly compiler-produced.
    canonical = "".join(f"{llvm_path.name}:{row.function}\t{row.frame_bytes}\t{row.allocation}\n" for row in rows)
    measurement_view.write_text(canonical, encoding="utf-8")
    # A dynamic helper beneath a static entrypoint is still an unbounded stack use.  Reject every
    # dynamic row in the translation unit, not merely a dynamic spelling on the entry row.
    dynamic = [row.function for row in rows if row.allocation != "static"]
    if dynamic:
        raise StackFramePreflightError("stack-usage report contains dynamic allocation for: " + ", ".join(dynamic))
    matches = [row for row in rows if row.function == entry_symbol]
    if len(matches) != 1:
        raise StackFramePreflightError(
            f"stack-usage report contains {len(matches)} rows for entrypoint {entry_symbol!r}; exactly one is required"
        )
    row = matches[0]
    if type(max_static_bytes) is not int or max_static_bytes <= 0:
        raise StackFramePreflightError("stack-frame budget is not a positive integer")
    measurement = StackFrameMeasurement(
        entry_symbol=entry_symbol, frame_bytes=row.frame_bytes, max_static_bytes=max_static_bytes, report_rows=len(rows)
    )
    if measurement.frame_bytes > measurement.max_static_bytes:
        raise StackFramePreflightError(
            f"entrypoint {entry_symbol!r} needs {measurement.frame_bytes} static stack bytes, "
            f"exceeding the target-declared {measurement.max_static_bytes}-byte budget by "
            f"{measurement.frame_bytes - measurement.max_static_bytes} bytes",
            measurement=measurement,
        )
    return measurement


def write_receipt(
    path: Path,
    *,
    status: str,
    llvm_path: Path,
    object_path: Path,
    report_path: Path,
    entry_symbol: str,
    max_static_bytes: int,
    measurement: StackFrameMeasurement | None = None,
    diagnostic: str | None = None,
    repair: dict | None = None,
) -> Path:
    """Write the content-bound stack assessment beside the compiler outputs."""
    llvm_path, object_path, report_path = map(Path, (llvm_path, object_path, report_path))
    record = {
        "schema": "merlin_kernel_stack_frame_preflight_v1",
        "status": status,
        "entry_symbol": entry_symbol,
        "max_static_bytes": max_static_bytes,
        "frame_bytes": measurement.frame_bytes if measurement is not None else None,
        "headroom_bytes": measurement.headroom_bytes if measurement is not None else None,
        "report_rows": measurement.report_rows if measurement is not None else None,
        "llvm_ir_sha256": _sha256(llvm_path) if llvm_path.is_file() else None,
        "object_sha256": _sha256(object_path) if object_path.is_file() else None,
        "stack_usage_report_sha256": _sha256(report_path) if report_path.is_file() else None,
        "stack_usage_measurement_view": (
            {
                "name": report_path.with_suffix(".measured.su").name,
                "sha256": _sha256(report_path.with_suffix(".measured.su")),
                "raw_source_sha256": _sha256(report_path),
            }
            if report_path.is_file() and report_path.with_suffix(".measured.su").is_file()
            else None
        ),
        "diagnostic": diagnostic,
        # What was DONE to make an over-budget frame fit, if anything, and what the first
        # measurement said before it. A frame that passes only after a transform must not
        # be indistinguishable from one that passed exactly as emitted.
        "repair": dict(repair) if repair else None,
    }
    destination = Path(path)
    destination.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return destination


# ---- object_preflight --------------------------------------------------------------------------
# Selected emitted-module ABI and measured entry-frame preflight.


def _module_target_abi(llvm_text: str) -> str | None:
    """Read the LLVM module's target-abi flag, without treating IR as ABI authority."""
    references: list[str] = []
    definitions: dict[str, str] = {}
    for line in llvm_text.splitlines():
        line = line.strip()
        if line.startswith("!llvm.module.flags = !{"):
            if references:
                raise ValueError("LLVM module has duplicate module-flag tables")
            payload = line.partition("!{")[2].removesuffix("}")
            references = [item.strip() for item in payload.split(",")]
        elif line.startswith("!") and " = !{" in line:
            name, _, payload = line.partition(" = !{")
            definitions[name] = payload.removesuffix("}")
    found: list[str] = []
    for reference in references:
        operands = [part.strip() for part in definitions.get(reference, "").split(",")]
        if len(operands) > 1 and operands[1] == '!"target-abi"':
            if len(operands) != 3 or not operands[2].startswith('!"') or not operands[2].endswith('"'):
                raise ValueError("LLVM module has malformed target-abi flag")
            found.append(operands[2][2:-1])
    if len(found) > 1:
        raise ValueError("LLVM module has ambiguous target-abi flags")
    return found[0] if found else None


def _abi_receipt(workdir: Path, obj: Path, abi: str) -> None:
    """Bind the object to the ABI selected for its harness in a paired build."""
    (workdir / "kernel.abi.json").write_text(
        json.dumps(
            {
                "schema": "merlin_kernel_abi_v1",
                "abi": abi,
                "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _repair_oversized_frame(
    llvm_path, object_path, *, recipe, policy, extra, remaining: Callable[[], float | None] | None = None
):
    """Seat the entry frame's static temporaries in one arena and RE-MEASURE.

    Returns ``(object, measurement, llvm_path, report_path, report)``, or ``None`` when the repair
    is unavailable or did not actually fit. The measurement is taken on the REBUILT object; a repair
    that is believed rather than measured is how an over-budget frame reaches a device.

    On ``None`` the caller's original refusal stands, and it is deliberately the diagnostic the
    caller sees: the emitted program is what was asked about, and a second diagnostic describing a
    repaired variant would name a program nobody requested.
    """
    from merlin.compiler.lower.host.storage import StackArenaError, bind_stack_arena
    from merlin.compiler.lower.model import codegen as _codegen

    llvm_path, object_path = Path(llvm_path), Path(object_path)
    try:
        rewritten, report = bind_stack_arena(llvm_path.read_text(encoding="utf-8"), entry_symbol=policy.entry_symbol)
    except StackArenaError:
        return None
    if not report.n_bound:
        return None
    arena_llvm = llvm_path.with_suffix(".arena.ll")
    arena_object = object_path.with_suffix(".arena.o")
    arena_su = arena_object.with_suffix(".su")
    arena_llvm.write_text(rewritten, encoding="utf-8")
    for stale in (arena_object, arena_su, arena_object.with_suffix(".measured.su")):
        stale.unlink(missing_ok=True)
    compile_kwargs = {"extra_flags": (*extra, "-fstack-usage")}
    if remaining is not None:
        compile_kwargs["timeout_s"] = remaining()
    rebuilt = Path(_codegen.compile_ll(arena_llvm, arena_object, "riscv", **compile_kwargs))
    try:
        measurement = measure_entrypoint(
            arena_su, llvm_path=arena_llvm, entry_symbol=policy.entry_symbol, max_static_bytes=policy.max_static_bytes
        )
    except StackFramePreflightError:
        return None
    # The arena's bytes are NOT free: they are `.bss` in the same image, and a large one narrows the
    # PC-relative window a medany build has to work in. Reported through the receipt, never absorbed.
    return rebuilt, measurement, arena_llvm, arena_su, report.to_dict()


# Preserve historical callable and record pickle lookup identities.
_module_target_abi.__module__ = "merlin.development.targetgen.execution.runner.compile"
_abi_receipt.__module__ = "merlin.development.targetgen.execution.runner.compile"
_repair_oversized_frame.__module__ = "merlin.development.targetgen.execution.runner.compile"
