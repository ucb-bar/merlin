"""Original caller-owned borrowed ABI premises for native lifetime checking.

These data transport a separately qualified implementation contract. Neither an
identity, matching signature, read/write annotation nor successful numerical
test establishes the supplied effects. No permission is read from candidate IR.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file

SCHEMA = "merlin.borrowed_buffer_calls.v1"
ROSTER = "borrowed-buffer-calls.json"
CHECKPOINT = "borrowed-buffer-lifetime.mlir"
REPORT = "borrowed-buffer-lifetime.json"
_MAX_BYTES = 1 << 20


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class BorrowedBufferEffects:
    """Explicit original implementation premises, not a derived qualification."""

    qualification_identity: str
    no_retained_aliases: bool
    no_releases: bool
    complete_before_return: bool

    def __post_init__(self):
        if (
            type(self.qualification_identity) is not str
            or not self.qualification_identity.strip()
            or len(self.qualification_identity) > 4096
            or any(value is not True for value in (
                self.no_retained_aliases, self.no_releases, self.complete_before_return,
            ))
        ):
            raise ValueError("borrowed ABI requires explicit noescape, no-release and synchronous effects")


@dataclass(frozen=True)
class BorrowedBufferCall:
    wrapper_symbol: str
    borrowed_symbol: str
    argument_types: tuple[str, ...]
    result_argument: int
    effects: BorrowedBufferEffects

    def __post_init__(self):
        if (
            any(type(s) is not str or not s or len(s) > 4096 for s in (
                self.wrapper_symbol, self.borrowed_symbol,
            ))
            or self.wrapper_symbol == self.borrowed_symbol
            or type(self.argument_types) is not tuple or not self.argument_types
            or len(self.argument_types) > 128
            or any(type(t) is not str or not t or len(t) > 4096 for t in self.argument_types)
            or type(self.result_argument) is not int
            or not 0 <= self.result_argument < len(self.argument_types)
            or type(self.effects) is not BorrowedBufferEffects
        ):
            raise ValueError("borrowed ABI requires bounded original symbols, positional memref types and effects")
        self.effects.__post_init__()


def validate_effects(effects):
    if effects is not None:
        if type(effects) is not BorrowedBufferEffects:
            raise ValueError("explicit typed borrowed ABI implementation effects required")
        effects.__post_init__()


def validate_calls(calls):
    if type(calls) is not tuple or len(calls) > 1024 or any(type(c) is not BorrowedBufferCall for c in calls):
        raise ValueError("immutable bounded original borrowed-call roster required")
    symbols = []
    for call in calls:
        call.__post_init__()
        symbols.extend((call.wrapper_symbol, call.borrowed_symbol))
    if len(symbols) != len(set(symbols)):
        raise ValueError("borrowed-call symbols must be distinct across the complete original roster")
    size = 0
    for chunk in json.JSONEncoder().iterencode([asdict(c) for c in calls]):
        size += len(chunk.encode())
        if size > _MAX_BYTES:
            raise ValueError("borrowed-call roster exceeds its byte budget")


def read_borrowed_calls(record):
    """Read typed DATA; the caller must establish original authority and custody."""
    if (type(record) is not dict or record.get("schema") != SCHEMA
            or type(record.get("calls")) is not list or len(record["calls"]) > 1024):
        raise ValueError("versioned original borrowed-call record required")
    calls = []
    for item in record["calls"]:
        if (type(item) is not dict or set(item) != {
            "wrapper_symbol", "borrowed_symbol", "argument_types", "result_argument", "effects",
        } or type(item["argument_types"]) is not list or len(item["argument_types"]) > 128
                or type(item["effects"]) is not dict):
            raise ValueError("complete positional original borrowed-call record required")
        if set(item["effects"]) != {
            "qualification_identity", "no_retained_aliases", "no_releases", "complete_before_return",
        }:
            raise ValueError("complete original borrowed ABI effects required")
        calls.append(BorrowedBufferCall(
            item["wrapper_symbol"], item["borrowed_symbol"], tuple(item["argument_types"]),
            item["result_argument"], BorrowedBufferEffects(**item["effects"]),
        ))
    result = tuple(calls)
    validate_calls(result)
    return result


@dataclass(frozen=True)
class BorrowedBufferCalls:
    source_path: Path
    source_sha256: str
    input_sha256: str
    calls: tuple[BorrowedBufferCall, ...]
    pins: tuple[tuple[Path, str], ...]
    validate_original: Callable[[], None] = field(repr=False, compare=False)
    observe_native: Callable[[tuple[Path, ...], bool], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if (
            not isinstance(self.source_path, Path)
            or not is_sha256(self.source_sha256) or not is_sha256(self.input_sha256)
            or type(self.pins) is not tuple
            or any(type(p) is not tuple or len(p) != 2 or not isinstance(p[0], Path) or not is_sha256(p[1])
                   for p in self.pins)
            or not callable(self.validate_original)
            or self.observe_native is not None and not callable(self.observe_native)
        ):
            raise ValueError("borrowed calls require original source, immutable pins and invocation validation")
        validate_calls(self.calls)

    def revalidate(self, source=None):
        self.__post_init__()
        self.validate_original()
        for path, digest in ((self.source_path, self.source_sha256), *self.pins):
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("original borrowed-call source or contracts changed; reselect and replan")
        if source is not None and _sha(source) != self.input_sha256:
            raise ValueError("borrowed-call selection is stale for the current source; reselect and replan")

    def after_preprocessing(self, before, after):
        self.revalidate(before)
        return replace(self, input_sha256=_sha(after))

    def record(self):
        return dict(
            schema=SCHEMA, calls=[{**asdict(c), "argument_types": list(c.argument_types)} for c in self.calls],
            original_source=dict(path=str(self.source_path), sha256=self.source_sha256),
            current_input_sha256=self.input_sha256,
            original_pins=[dict(path=str(p), sha256=d) for p, d in self.pins],
            scope="original conditional ABI effects; implementation qualification and final link remain independent",
        )


def from_writers(source, writers, pins, validate_original, observe_native=None):
    """Freeze selected writer premises, never discover permissions in a module."""
    from xdsl.dialects import builtin, func

    from merlin.common import mlir_query as mq

    source = Path(source).resolve()
    module = mq.parse(source)
    module.verify()
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    calls = []
    for writer in writers:
        if writer.borrowed_effects is None:
            continue
        declaration = declarations.get(writer.borrowed_symbol)
        if (
            writer.declaration_abi != "ranked_c" or writer.private_workspaces
            or writer.fully_written_arguments != (writer.result_argument,)
            or declaration is None or declaration.body.blocks or declaration.function_type.outputs
            or any(not isinstance(t, builtin.MemRefType) for t in declaration.function_type.inputs)
        ):
            raise ValueError("borrowed lifetime proof supports one fresh ranked result and no private workspace")
        calls.append(BorrowedBufferCall(
            writer.wrapper_symbol or writer.symbol, writer.borrowed_symbol,
            tuple(str(t) for t in declaration.function_type.inputs), writer.result_argument, writer.borrowed_effects,
        ))
    if not calls:
        return None
    digest = sha256_file(source)
    result = BorrowedBufferCalls(source, digest, digest, tuple(calls), tuple(pins), validate_original, observe_native)
    result.revalidate()
    return result


def bind_native(source, calls=(), *, output=None, record=None):
    """Bind immutable original data into the trusted runner, without IR attributes."""
    validate_calls(calls)
    marker = "_OBL_CALL_ROSTER = ()"
    if source.count(marker) != 1:
        raise ValueError("native lifetime checker lost its original borrowed-call binding seam")
    packet = [asdict(c) for c in calls]
    bound = source.replace(marker, "_OBL_CALL_ROSTER = " + repr(packet), 1)
    if output is not None:
        marker = "_OBL_CALL_OUTPUT = None"
        if bound.count(marker) != 1:
            raise ValueError("native lifetime checker lost its selected observation seam")
        bound = bound.replace(marker, "_OBL_CALL_OUTPUT = " + repr(dict(
            checkpoint=str(output / CHECKPOINT), report=str(output / REPORT), original=record,
        )), 1)
    return bound


def prepare_native(selection, source, runner, work):
    if type(selection) is not BorrowedBufferCalls or not selection.calls:
        raise ValueError("nonempty original borrowed-call selection required")
    selection.revalidate(source)
    record = selection.record()
    path = work / ROSTER
    path.write_text(json.dumps(record, sort_keys=True))
    for name in (REPORT, CHECKPOINT):
        (work / name).unlink(missing_ok=True)
    return bind_native(runner, selection.calls, output=work, record=record), path


def verify_native(selection, source, work, *, native_stdout=None, native_runner=None):
    """Authenticate a returned native observation, or reopen its retained pins."""
    selection.revalidate(source)
    original = json.loads((work / ROSTER).read_text())
    report = json.loads((work / REPORT).read_text())
    allowed = {symbol for call in selection.calls for symbol in (call.wrapper_symbol, call.borrowed_symbol)}
    required = {call.borrowed_symbol for call in selection.calls}
    accepted = report.get("accepted_symbols")
    if (
        set(report) != {"schema", "status", "original", "violations", "checkpoint_sha256", "accepted_symbols", "scope"}
        or original != selection.record() or report.get("original") != original
        or report.get("schema") != "merlin.borrowed_buffer_lifetime.v1"
        or report.get("status") != "checked" or report.get("violations") != []
        or report.get("checkpoint_sha256") != sha256_file(work / CHECKPOINT)
        or type(accepted) is not list or any(type(s) is not str for s in accepted)
        or accepted != sorted(set(accepted)) or not required <= set(accepted) <= allowed
        or report.get("scope") != "conditional original borrowed effects and current typed forwarding bodies"
    ):
        raise ValueError("borrowed-call native consumption or current checkpoint is missing or changed")
    paths = tuple((work / name).resolve() for name in (ROSTER, CHECKPOINT, REPORT))
    if native_stdout is not None:
        lines = [line.split() for line in native_stdout.splitlines() if line.startswith("OK borrowed_buffer_lifetime ")]
        if lines != [["OK", "borrowed_buffer_lifetime", sha256_file(work / REPORT), sha256_file(work / CHECKPOINT)]]:
            raise ValueError("native borrowed lifetime output token is absent or does not match actual files")
        if native_runner is not None:
            runner = Path(native_runner)
            recipe = json.loads((work / "lowering_recipe.json").read_text())
            identity = recipe["sources"]["runner"]
            if (runner.is_symlink() or Path(identity["path"]) != runner.resolve()
                    or sha256_file(runner) != identity["sha256"]):
                raise ValueError("actual borrowed lifetime runner changed after native execution")
            paths = (*paths, runner.resolve())
    elif selection.observe_native is None:
        raise ValueError("original native borrowed lifetime consumption token or retained observation required")
    if selection.observe_native is not None:
        selection.observe_native(paths, native_stdout is not None)
    return report
