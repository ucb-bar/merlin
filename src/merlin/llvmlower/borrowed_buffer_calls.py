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
RESULTS_SCHEMA = "merlin.borrowed_buffer_calls.v2"
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
            or any(
                value is not True
                for value in (
                    self.no_retained_aliases,
                    self.no_releases,
                    self.complete_before_return,
                )
            )
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
            any(
                type(s) is not str or not s or len(s) > 4096
                for s in (
                    self.wrapper_symbol,
                    self.borrowed_symbol,
                )
            )
            or self.wrapper_symbol == self.borrowed_symbol
            or type(self.argument_types) is not tuple
            or not self.argument_types
            or len(self.argument_types) > 128
            or any(type(t) is not str or not t or len(t) > 4096 for t in self.argument_types)
            or type(self.result_argument) is not int
            or not 0 <= self.result_argument < len(self.argument_types)
            or type(self.effects) is not BorrowedBufferEffects
        ):
            raise ValueError("borrowed ABI requires bounded original symbols, positional memref types and effects")
        self.effects.__post_init__()


@dataclass(frozen=True)
class BorrowedBufferResults:
    """One original borrowed call with every ordered fresh result position.

    This is a lifetime/ABI premise, not a full-write or numerical witness.
    The native checker independently checks current forwarding and owners.
    """

    wrapper_symbol: str
    borrowed_symbol: str
    argument_types: tuple[str, ...]
    result_arguments: tuple[int, ...]
    effects: BorrowedBufferEffects

    def __post_init__(self):
        # Validate the original bounded symbol/type/effect fields first.
        BorrowedBufferCall(self.wrapper_symbol, self.borrowed_symbol, self.argument_types, 0, self.effects)
        if (
            type(self.result_arguments) is not tuple
            or not 2 <= len(self.result_arguments) <= 128
            or any(type(i) is not int or not 0 <= i < len(self.argument_types) for i in self.result_arguments)
            or len(set(self.result_arguments)) != len(self.result_arguments)
        ):
            raise ValueError("plural borrowed ABI requires distinct ordered result positions")
        # Reuse exactly the original symbol, type and effects validation.
        BorrowedBufferCall(
            self.wrapper_symbol, self.borrowed_symbol, self.argument_types, self.result_arguments[0], self.effects
        )


def _schema(calls):
    return RESULTS_SCHEMA if any(type(call) is BorrowedBufferResults for call in calls) else SCHEMA


def validate_effects(effects):
    if effects is not None:
        if type(effects) is not BorrowedBufferEffects:
            raise ValueError("explicit typed borrowed ABI implementation effects required")
        effects.__post_init__()


def validate_calls(calls):
    if (
        type(calls) is not tuple
        or len(calls) > 1024
        or any(type(c) not in (BorrowedBufferCall, BorrowedBufferResults) for c in calls)
    ):
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
    if (
        type(record) is not dict
        or record.get("schema") not in (SCHEMA, RESULTS_SCHEMA)
        or type(record.get("calls")) is not list
        or len(record["calls"]) > 1024
    ):
        raise ValueError("versioned original borrowed-call record required")
    calls = []
    for item in record["calls"]:
        plural = type(item) is dict and "result_arguments" in item
        result_key = "result_arguments" if plural else "result_argument"
        if (
            type(item) is not dict
            or set(item)
            != {
                "wrapper_symbol",
                "borrowed_symbol",
                "argument_types",
                result_key,
                "effects",
            }
            or type(item["argument_types"]) is not list
            or len(item["argument_types"]) > 128
            or type(item["effects"]) is not dict
            or plural
            and (
                record["schema"] != RESULTS_SCHEMA or type(item[result_key]) is not list or len(item[result_key]) > 128
            )
        ):
            raise ValueError("complete positional original borrowed-call record required")
        if set(item["effects"]) != {
            "qualification_identity",
            "no_retained_aliases",
            "no_releases",
            "complete_before_return",
        }:
            raise ValueError("complete original borrowed ABI effects required")
        row_type = BorrowedBufferResults if plural else BorrowedBufferCall
        calls.append(
            row_type(
                item["wrapper_symbol"],
                item["borrowed_symbol"],
                tuple(item["argument_types"]),
                tuple(item[result_key]) if plural else item[result_key],
                BorrowedBufferEffects(**item["effects"]),
            )
        )
    result = tuple(calls)
    validate_calls(result)
    if record["schema"] != _schema(result):
        raise ValueError("borrowed-call version must match its actual result roster")
    return result


def _callback_binding(callback):
    """Retain identity as well as references; user equality grants no custody."""
    if callback is None:
        return ((id(None), None),) * 4
    target = getattr(callback, "__func__", callback)
    implementation = target if hasattr(target, "__code__") else getattr(type(callback), "__call__", None)
    code = getattr(implementation, "__code__", None)
    owner = getattr(callback, "__self__", None)
    return tuple((id(value), value) for value in (callback, owner, implementation, code))


def _plural_state(selection):
    """Callback-free snapshot of original v2 data and selected reader code."""
    callbacks = (selection.validate_original, selection.observe_native)
    functions = (
        _plural_state,
        _callback_binding,
        _check_plural_state,
        _schema,
        validate_calls,
        validate_effects,
        BorrowedBufferEffects.__post_init__,
        BorrowedBufferCall.__post_init__,
        BorrowedBufferResults.__post_init__,
        BorrowedBufferCalls.__post_init__,
        BorrowedBufferCalls.revalidate,
        BorrowedBufferCalls.after_preprocessing,
        BorrowedBufferCalls.record,
        from_writers,
        bind_native,
        prepare_native,
        verify_native,
        read_borrowed_calls,
        sha256_file,
    )
    return (
        selection.source_path,
        selection.source_sha256,
        selection.input_sha256,
        json.dumps([asdict(c) for c in selection.calls], sort_keys=True),
        selection.pins,
        selection._plural_sources,
        tuple(_callback_binding(callback) for callback in callbacks),
        tuple(_callback_binding(function) for function in functions),
    )


def _check_plural_state(selection):
    binding = selection._plural_binding
    if binding is not None:
        state, reader, code, checker, checker_code, callback_reader, callback_code = binding
        if (
            _check_plural_state is not checker
            or checker.__code__ is not checker_code
            or _plural_state is not reader
            or reader.__code__ is not code
            or _callback_binding is not callback_reader
            or callback_reader.__code__ is not callback_code
            or reader(selection) != state
        ):
            raise ValueError("original plural borrowed selection or selected code changed; reselect and replan")
        for path, digest in (
            (selection.source_path, selection.source_sha256),
            *selection.pins,
            *selection._plural_sources,
        ):
            if path.is_symlink() or not path.is_file() or path.resolve() != path or sha256_file(path) != digest:
                raise ValueError("original plural borrowed source or implementation changed; reselect and replan")


@dataclass(frozen=True)
class BorrowedBufferCalls:
    source_path: Path
    source_sha256: str
    input_sha256: str
    calls: tuple[BorrowedBufferCall | BorrowedBufferResults, ...]
    pins: tuple[tuple[Path, str], ...]
    validate_original: Callable[[], None] = field(repr=False, compare=False)
    observe_native: Callable[[tuple[Path, ...], bool], None] | None = field(default=None, repr=False, compare=False)
    _plural_binding: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _plural_sources: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if (
            not isinstance(self.source_path, Path)
            or not is_sha256(self.source_sha256)
            or not is_sha256(self.input_sha256)
            or type(self.pins) is not tuple
            or any(
                type(p) is not tuple or len(p) != 2 or not isinstance(p[0], Path) or not is_sha256(p[1])
                for p in self.pins
            )
            or not callable(self.validate_original)
            or self.observe_native is not None
            and not callable(self.observe_native)
        ):
            raise ValueError("borrowed calls require original source, immutable pins and invocation validation")
        validate_calls(self.calls)
        if _schema(self.calls) == RESULTS_SCHEMA and self._plural_binding is None:
            owner = Path(__file__).resolve(strict=True)
            sources = (owner, owner.with_name("dealloc_lifetime.py"))
            object.__setattr__(self, "_plural_sources", tuple((path, sha256_file(path)) for path in sources))
            object.__setattr__(
                self,
                "_plural_binding",
                (
                    _plural_state(self),
                    _plural_state,
                    _plural_state.__code__,
                    _check_plural_state,
                    _check_plural_state.__code__,
                    _callback_binding,
                    _callback_binding.__code__,
                ),
            )

    def revalidate(self, source=None):
        # New plural selections freeze all rows before the first original
        # callback. Existing v1 record/selection semantics remain unchanged.
        binding = self._plural_binding
        checker, code = binding[3:5] if binding is not None else (_check_plural_state, _check_plural_state.__code__)
        if _check_plural_state is not checker or checker.__code__ is not code:
            raise ValueError("original plural borrowed selection reader changed")
        checker(self)
        self.__post_init__()
        self.validate_original()
        if self._plural_binding is not binding or (_schema(self.calls) == RESULTS_SCHEMA) != (binding is not None):
            raise ValueError("original plural borrowed selection mode changed")
        if binding is not None:
            if _check_plural_state is not checker or checker.__code__ is not code:
                raise ValueError("original plural borrowed selection reader changed")
            checker(self)
        for path, digest in ((self.source_path, self.source_sha256), *self.pins):
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("original borrowed-call source or contracts changed; reselect and replan")
        if source is not None and _sha(source) != self.input_sha256:
            raise ValueError("borrowed-call selection is stale for the current source; reselect and replan")

    def after_preprocessing(self, before, after):
        self.revalidate(before)
        return replace(self, input_sha256=_sha(after))

    def record(self):
        binding = self._plural_binding
        if binding is not None:
            checker, code = binding[3:5]
            if _check_plural_state is not checker or checker.__code__ is not code:
                raise ValueError("original plural borrowed selection reader changed")
            checker(self)
        record = dict(
            schema=_schema(self.calls),
            calls=[
                {
                    **asdict(c),
                    "argument_types": list(c.argument_types),
                    **({"result_arguments": list(c.result_arguments)} if type(c) is BorrowedBufferResults else {}),
                }
                for c in self.calls
            ],
            original_source=dict(path=str(self.source_path), sha256=self.source_sha256),
            current_input_sha256=self.input_sha256,
            original_pins=[dict(path=str(p), sha256=d) for p, d in self.pins],
            scope="original conditional ABI effects; implementation qualification and final link remain independent",
        )
        if binding is not None:
            record["implementation_sources"] = [dict(path=str(p), sha256=d) for p, d in self._plural_sources]
        return record


def from_writers(source, writers, pins, validate_original, observe_native=None):
    """Freeze selected writer premises, never discover permissions in a module."""
    from xdsl.dialects import builtin, func

    from merlin.common import mlir_query as mq

    writers = tuple(writers)
    if len(writers) > 1024:
        raise ValueError("borrowed writer roster exceeds its call budget")
    plural_selection = any(bool(getattr(writer, "result_arguments", ())) for writer in writers)
    source = Path(source).resolve()
    original_digest = sha256_file(source)
    writer_state = json.dumps([asdict(writer) for writer in writers], sort_keys=True) if plural_selection else None
    module = mq.parse(source)
    module.verify()
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    calls = []
    for writer in writers:
        plural = bool(getattr(writer, "result_arguments", ()))
        if plural and writer.borrowed_effects is None:
            raise ValueError("plural borrowed lifetime proof requires original effects")
        if writer.borrowed_effects is None:
            continue
        declaration = declarations.get(writer.borrowed_symbol)
        if (
            writer.declaration_abi != "ranked_c"
            or writer.private_workspaces
            or (not plural and writer.fully_written_arguments != (writer.result_argument,))
            or (
                plural
                and (
                    type(writer.result_arguments) is not tuple
                    or writer.result_arguments[0] != writer.result_argument
                    or set(writer.result_arguments) != set(writer.fully_written_arguments)
                )
            )
            or declaration is None
            or declaration.body.blocks
            or declaration.function_type.outputs
            or any(not isinstance(t, builtin.MemRefType) for t in declaration.function_type.inputs)
        ):
            raise ValueError("borrowed lifetime proof requires complete fresh ranked results and no private workspace")
        row_type = BorrowedBufferResults if plural else BorrowedBufferCall
        calls.append(
            row_type(
                writer.wrapper_symbol or writer.symbol,
                writer.borrowed_symbol,
                tuple(str(t) for t in declaration.function_type.inputs),
                writer.result_arguments if plural else writer.result_argument,
                writer.borrowed_effects,
            )
        )
    if not calls:
        return None
    digest = sha256_file(source)
    validator = validate_original
    if plural_selection:
        callback_reader, callback_code = _callback_binding, _callback_binding.__code__
        validator_binding = callback_reader(validator)

        def validate_writer_selection():
            def unchanged():
                if _callback_binding is not callback_reader or callback_reader.__code__ is not callback_code:
                    raise ValueError("original plural writer callback reader changed")
                if (
                    json.dumps([asdict(writer) for writer in writers], sort_keys=True) != writer_state
                    or callback_reader(validator) != validator_binding
                ):
                    raise ValueError("original plural writer selection changed; reselect and replan")

            unchanged()
            validator()
            unchanged()

        if original_digest != digest:
            raise ValueError("original plural writer source changed during binding")
        validate_original = validate_writer_selection
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
        bound = bound.replace(
            marker,
            "_OBL_CALL_OUTPUT = "
            + repr(
                dict(
                    checkpoint=str(output / CHECKPOINT),
                    report=str(output / REPORT),
                    original=record,
                )
            ),
            1,
        )
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
        or original != selection.record()
        or report.get("original") != original
        or report.get("schema")
        != (
            "merlin.borrowed_buffer_lifetime.v2"
            if _schema(selection.calls) == RESULTS_SCHEMA
            else "merlin.borrowed_buffer_lifetime.v1"
        )
        or report.get("status") != "checked"
        or report.get("violations") != []
        or report.get("checkpoint_sha256") != sha256_file(work / CHECKPOINT)
        or type(accepted) is not list
        or any(type(s) is not str for s in accepted)
        or accepted != sorted(set(accepted))
        or not required <= set(accepted) <= allowed
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
            if (
                runner.is_symlink()
                or Path(identity["path"]) != runner.resolve()
                or sha256_file(runner) != identity["sha256"]
            ):
                raise ValueError("actual borrowed lifetime runner changed after native execution")
            paths = (*paths, runner.resolve())
    elif selection.observe_native is None:
        raise ValueError("original native borrowed lifetime consumption token or retained observation required")
    products = tuple((path, sha256_file(path)) for path in paths) if selection._plural_binding is not None else ()
    if selection.observe_native is not None:
        selection.observe_native(paths, native_stdout is not None)
    if products:
        selection.revalidate(source)
        for path, digest in products:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("actual plural borrowed lifetime observation changed after callback")
    return report
