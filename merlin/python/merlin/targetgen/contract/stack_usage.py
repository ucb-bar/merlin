"""Fail-closed parsing and recording of compiler-produced static stack-frame measurements."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


class StackFramePreflightError(ValueError):
    """A compiler stack report cannot prove that the declared entrypoint fits its budget."""

    def __init__(self, message: str, *, measurement: "StackFrameMeasurement | None" = None):
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise StackFramePreflightError(
                f"stack-usage report line {number} does not have three tab-separated fields")
        identity, byte_text, allocation = fields
        source_identity, separator, function = identity.rpartition(":")
        if not separator or not source_identity or not function:
            raise StackFramePreflightError(
                f"stack-usage report line {number} has no source/function identity")
        if not byte_text.isascii() or not byte_text.isdecimal():
            raise StackFramePreflightError(
                f"stack-usage report line {number} has a non-decimal frame size")
        frame_bytes = int(byte_text)
        if allocation not in {"static", "dynamic", "dynamic,bounded"}:
            raise StackFramePreflightError(
                f"stack-usage report line {number} has unknown allocation kind {allocation!r}")
        rows.append(StackUsageRow(source_identity, function, frame_bytes, allocation))
    if not rows:
        raise StackFramePreflightError("stack-usage report is empty")
    return tuple(rows)


def measure_entrypoint(report_path: Path, *, llvm_path: Path, entry_symbol: str,
                       max_static_bytes: int) -> StackFrameMeasurement:
    """Require one fresh, source-bound, fully static entrypoint row within the target's budget."""
    report_path, llvm_path = Path(report_path), Path(llvm_path)
    if report_path.is_symlink() or not report_path.is_file():
        raise StackFramePreflightError(
            f"compiler produced no regular stack-usage report at {report_path}")
    try:
        rows = parse_stack_usage(report_path.read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise StackFramePreflightError("stack-usage report is not valid UTF-8") from exc
    for row in rows:
        if Path(row.source_identity).name != llvm_path.name:
            raise StackFramePreflightError(
                f"stack row for {row.function!r} names source {row.source_identity!r}, "
                f"expected {llvm_path.name!r}")
    # Clang writes the absolute input path into each otherwise deterministic row.  Canonicalize that
    # one workdir-dependent component after validating it, so two byte-identical compilations yield
    # the same evidence artifact and build-cache receipt.  Counts, classifications, symbols and row
    # order remain exactly compiler-produced.
    canonical = "".join(
        f"{llvm_path.name}:{row.function}\t{row.frame_bytes}\t{row.allocation}\n"
        for row in rows)
    report_path.write_text(canonical, encoding="utf-8")
    # A dynamic helper beneath a static entrypoint is still an unbounded stack use.  Reject every
    # dynamic row in the translation unit, not merely a dynamic spelling on the entry row.
    dynamic = [row.function for row in rows if row.allocation != "static"]
    if dynamic:
        raise StackFramePreflightError(
            "stack-usage report contains dynamic allocation for: " + ", ".join(dynamic))
    matches = [row for row in rows if row.function == entry_symbol]
    if len(matches) != 1:
        raise StackFramePreflightError(
            f"stack-usage report contains {len(matches)} rows for entrypoint {entry_symbol!r}; "
            "exactly one is required")
    row = matches[0]
    if type(max_static_bytes) is not int or max_static_bytes <= 0:
        raise StackFramePreflightError("stack-frame budget is not a positive integer")
    measurement = StackFrameMeasurement(
        entry_symbol=entry_symbol, frame_bytes=row.frame_bytes,
        max_static_bytes=max_static_bytes, report_rows=len(rows))
    if measurement.frame_bytes > measurement.max_static_bytes:
        raise StackFramePreflightError(
            f"entrypoint {entry_symbol!r} needs {measurement.frame_bytes} static stack bytes, "
            f"exceeding the target-declared {measurement.max_static_bytes}-byte budget by "
            f"{measurement.frame_bytes - measurement.max_static_bytes} bytes",
            measurement=measurement)
    return measurement


def write_receipt(path: Path, *, status: str, llvm_path: Path, object_path: Path,
                  report_path: Path, entry_symbol: str, max_static_bytes: int,
                  measurement: StackFrameMeasurement | None = None,
                  diagnostic: str | None = None,
                  repair: dict | None = None) -> Path:
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
        "diagnostic": diagnostic,
        # What was DONE to make an over-budget frame fit, if anything, and what the first
        # measurement said before it. A frame that passes only after a transform must not
        # be indistinguishable from one that passed exactly as emitted.
        "repair": dict(repair) if repair else None,
    }
    destination = Path(path)
    destination.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return destination
