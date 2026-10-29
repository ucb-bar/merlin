"""Original-host scheduling of finite, declared target instruction MLIR.

This is an adapter to the existing schedule compiler, outside the public SDK.
It supplies no target implementation, execution qualification or import grant.
Unknown/dynamic instructions preserve the original fragment. Actual emitted
code must still be checked by the independently qualified target consumer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from xdsl.dialects import arith, builtin

from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection, _bounded_text, _call, _constant, _names
from merlin.compile.scheduling.program_limits import ProgramLimits
from merlin.compile.scheduling.schedule_pipeline import (
    ScheduleEncoding,
    _rewrite_snapshot,
    compile_schedule,
)
from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageOptimizationUnavailable, TemporaryObject
from merlin.sched.ir import Call, Kernel, evaluate
from merlin.targetgen.isa_dialect_gen import build_dialect, encode_instruction
from merlin.transforms.certificates import ir_digest

from .source_multioutput import _callback, _check_callback


def _functions():
    """Current direct owners, checked separately from retained callback objects."""
    return (
        instruction_kernel,
        instruction_module,
        compile_schedule,
        _rewrite_snapshot,
        _callback,
        _check_callback,
        _callbacks,
        _record,
        _json,
        _bounded_text,
        _call,
        _constant,
        _names,
        ir_digest,
        build_dialect,
        encode_instruction,
        ConfigurationIRSelection.parse,
        SelectedSchedulingContracts.revalidate,
        SelectedSchedulingContracts.record,
        TargetScheduleSelection.parse,
        TargetScheduleSelection.apply,
        TargetScheduleSelection.revalidate,
        TargetScheduleSelection.record,
        TargetScheduleSelection._state,
        TargetScheduleResult.__post_init__,
        TargetScheduleResult.revalidate,
        TargetScheduleResult._images,
    )


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _callbacks(value):
    if callable(value):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _callbacks(getattr(value, item.name))
    elif type(value) is dict:
        for key in sorted(value):
            yield from _callbacks(value[key])
    elif type(value) in (tuple, list):
        for item in value:
            yield from _callbacks(item)


def _record(value):
    """Describe exact original choices; a record establishes no authority."""
    if value is None or type(value) in (str, int, bool, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if type(value) is bytes:
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if callable(value):
        _, function, _, path, digest = _callback(value)
        return {"function": function.__qualname__, "path": str(path), "sha256": digest}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__qualname__,
            "fields": {f.name: _record(getattr(value, f.name)) for f in fields(value)},
        }
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("target scheduling options require string keys")
        return {key: _record(item) for key, item in value.items()}
    if type(value) in (tuple, list, set, frozenset):
        rows = [_record(item) for item in value]
        return sorted(rows, key=_json) if type(value) in (set, frozenset) else rows
    raise ValueError("target scheduling requires explicit typed original parameters")


def instruction_kernel(module, selected):
    """Read one closed finite ISA fragment without discarding opaque work."""
    if type(module) is not builtin.ModuleOp or module.attributes or module.properties or len(module.body.blocks) != 1:
        raise StorageOptimizationUnavailable("target scheduling needs one unannotated instruction module")
    names = _names(selected.isa)
    calls = []
    for operation in module.body.block.ops:
        if _constant(operation):
            continue
        call = _call(operation, selected, names)
        if call.instr in names:
            raise StorageOptimizationUnavailable("unknown, dynamic, resultful or control-flow target operation")
        calls.append(call)
    if not calls:
        raise StorageOptimizationUnavailable("target instruction fragment has no declared work")
    return Kernel("target_program", (), tuple(calls))


def instruction_module(kernel, selected):
    """Rebuild declared ISA calls, retaining every operand and exact encoding."""
    if type(kernel) is not Kernel or kernel.args or kernel.attrs:
        raise ValueError("target instruction writer needs a closed finite kernel")
    selected.revalidate()
    operations, classes = [], {op.name: op for op in build_dialect(selected.isa).operations}
    for call in kernel.body:
        if type(call) is not Call or any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes)):
            raise ValueError("target instruction writer cannot erase execution annotations")
        instruction = selected.isa.instruction(call.instr)
        values = {key: evaluate(expression, {}) for key, expression in call.args}
        if len(values) != len(call.args) or instruction.results:
            raise ValueError("target instruction writer needs exact scalar bindings and no SSA results")
        encode_instruction(selected.isa, call.instr, values)
        operands, attributes = [], {}
        for argument in (*instruction.operands, *instruction.immediates):
            if argument.type.kind != "integer":
                raise ValueError("target instruction writer requires declared integer instruction operands")
            value = values[argument.name]
            if argument in instruction.operands:
                constant = arith.ConstantOp.from_int_and_width(value, argument.type.bits)
                operations.append(constant)
                operands.append(constant.result)
            else:
                attributes[argument.name] = builtin.IntegerAttr(value, builtin.IntegerType(argument.type.bits))
        operations.append(
            classes[selected.isa.namespace + "." + call.instr].create(operands=operands, attributes=attributes)
        )
    result = builtin.ModuleOp(operations)
    result.verify()
    if instruction_kernel(result, selected).body != kernel.body:
        raise ValueError("reconstructed target IR differs from actual checked schedule")
    return result


@dataclass(frozen=True)
class TargetScheduleSelection:
    """An original host's existing adapters, resources and bounded choices.

    Target callbacks and qualification identifiers retain their independent
    original ownership. This class pins their code and current declarations;
    it neither qualifies their effects nor discovers a target implementation.
    ``parameters`` forwards existing compile_schedule keyword arguments only.
    """

    selected_contracts: SelectedSchedulingContracts
    objects: tuple[TemporaryObject, ...]
    adapter: StorageAdapter
    encoding: ScheduleEncoding
    limits: ProgramLimits
    parameters: dict = field(default_factory=dict)
    max_operations: int = 4096
    max_ir_bytes: int = 8 << 20
    _snapshot: object = field(init=False, repr=False)
    _selected_owner: object = field(init=False, repr=False)
    _owners: tuple = field(init=False, repr=False)
    _record: str = field(init=False, repr=False)
    _guards: tuple = field(init=False, repr=False)
    _bindings: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.objects) is not tuple
            or any(type(obj) is not TemporaryObject for obj in self.objects)
            or type(self.adapter) is not StorageAdapter
            or type(self.encoding) is not ScheduleEncoding
            or type(self.limits) is not ProgramLimits
            or type(self.parameters) is not dict
            or any(type(n) is not int or not 0 < n <= 1 << 30 for n in (self.max_operations, self.max_ir_bytes))
        ):
            raise ValueError("target scheduling needs original typed contracts, adapters and finite bounds")
        allowed = {
            name
            for name, parameter in inspect.signature(compile_schedule).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        } - {"selected_contracts", "pass_validation"}
        if set(self.parameters) - allowed:
            raise ValueError("target scheduling cannot override positional inputs or selected contracts")
        if self.selected_contracts.storage is None:
            raise ValueError("target scheduling lacks selected software/RTL storage effects")
        self.selected_contracts.revalidate()
        object.__setattr__(self, "_selected_owner", self.selected_contracts)
        state = self._state()
        snapshot = _rewrite_snapshot(state)
        functions = (
            _functions,
            *_functions(),
            self.selected_contracts.revalidate,
            self.selected_contracts.record,
            *_callbacks(state),
        )
        owners = tuple((callback, _callback(callback)) for callback in functions)
        record = _json(
            {
                "schema": "merlin.target_schedule_selection.v1",
                "selected": self.selected_contracts.record(),
                "parameters": _record(state),
                "implementation_sources": {str(row[3]): row[4] for _, row in owners},
                "qualification": "original independent adapter/runtime qualification remains required",
            }
        )
        if len(record.encode()) > 2 << 20:
            raise ValueError("target schedule selection exceeds its metadata budget")
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_owners", owners)
        object.__setattr__(
            self, "_guards", tuple((f, f.__code__) for f in (_check_callback, _rewrite_snapshot, _functions))
        )
        object.__setattr__(self, "_bindings", tuple((f, f.__code__) for f in _functions()))
        object.__setattr__(self, "_record", record)
        self.revalidate()

    def _state(self):
        return (
            self.selected_contracts,
            self.objects,
            self.adapter,
            self.encoding,
            self.limits,
            self.parameters,
            self.max_operations,
            self.max_ir_bytes,
        )

    def revalidate(self):
        for actual, (original, code) in zip(
            (_check_callback, _rewrite_snapshot, _functions), self._guards, strict=True
        ):
            if actual is not original or actual.__code__ is not code:
                raise ValueError("original target scheduling checker changed; replan")
        current = _functions()
        if len(current) != len(self._bindings) or any(
            actual is not original or actual.__code__ is not code
            for actual, (original, code) in zip(current, self._bindings, strict=True)
        ):
            raise ValueError("original target scheduling implementation binding changed; replan")
        if self.selected_contracts is not self._selected_owner:
            raise ValueError("original target scheduling selection owner changed; reselect")
        for callback, snapshot in self._owners:
            _check_callback(callback, snapshot)
        self.selected_contracts.revalidate()
        if _rewrite_snapshot(self._state()) != self._snapshot:
            raise ValueError("original target scheduling parameters or callbacks changed; replan")
        if self.selected_contracts.record() != json.loads(self._record)["selected"]:
            raise ValueError("original target scheduling contracts changed; reselect")

    def record(self):
        self.revalidate()
        return json.loads(self._record)

    def parse(self, text):
        self.revalidate()
        result = ConfigurationIRSelection(
            self.selected_contracts, None, max_operations=self.max_operations, max_ir_bytes=self.max_ir_bytes
        ).parse(text)
        self.revalidate()
        return result

    def apply(self, module, *, pass_validation=None):
        self.revalidate()
        _bounded_text(module, self.max_ir_bytes)
        module.verify()
        source = module.clone()
        source_digest = ir_digest(module)
        try:
            if sum(1 for _ in module.walk()) > self.max_operations:
                raise StorageOptimizationUnavailable("target scheduling operation budget exceeded")
            kernel = instruction_kernel(module, self.selected_contracts)
            compilation = compile_schedule(
                kernel,
                self.objects,
                None,
                self.adapter,
                self.encoding,
                self.limits,
                selected_contracts=self.selected_contracts,
                pass_validation=pass_validation,
                **self.parameters,
            )
            # Textual emission cannot erase a segmented launch/continuation ABI.
            if len(compilation.program.segments) != 1:
                raise StorageOptimizationUnavailable("target IR scheduling needs one checked launch segment")
            actual = instruction_module(compilation.kernel, self.selected_contracts)
            report = {"status": "checked", "schedule": compilation.report, "timing": "UNKNOWN"}
        except StorageOptimizationUnavailable as error:
            actual, compilation = source.clone(), None
            report = {"status": "fallback", "reason": str(error), "timing": "UNKNOWN"}
        if source_digest != ir_digest(module):
            raise ValueError("original target instruction IR changed during scheduling")
        self.revalidate()
        return TargetScheduleResult(source, actual, self, source_digest, ir_digest(actual), compilation, report)


@dataclass(frozen=True)
class TargetScheduleResult:
    source: builtin.ModuleOp
    module: builtin.ModuleOp
    selection: TargetScheduleSelection
    source_sha256: str
    output_sha256: str
    compilation: object
    report: dict
    _report_sha256: str = field(init=False, repr=False)
    _image_sha256: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_report_sha256", hashlib.sha256(_json(self.report).encode()).hexdigest())
        object.__setattr__(self, "_image_sha256", self._images())
        self.revalidate()

    def _images(self):
        return (
            ()
            if self.compilation is None
            else tuple(hashlib.sha256(image).hexdigest() for image in self.compilation.program.segments)
        )

    def revalidate(self):
        self.selection.revalidate()
        if (
            ir_digest(self.source) != self.source_sha256
            or ir_digest(self.module) != self.output_sha256
            or hashlib.sha256(_json(self.report).encode()).hexdigest() != self._report_sha256
            or self._images() != self._image_sha256
        ):
            raise ValueError("checked target schedule source, selected IR or report changed; replan")
        if self.compilation is None:
            if not self.source.is_structurally_equivalent(self.module):
                raise ValueError("unavailable target schedule changed original IR")
        else:
            actual = instruction_kernel(self.module, self.selection.selected_contracts)
            if actual != self.compilation.kernel:
                raise ValueError("actual target IR lost its independently checked schedule")
            decoded = tuple(self.selection.encoding.decode_calls(image) for image in self.compilation.program.segments)
            if tuple(call for segment in decoded for call in segment) != actual.body:
                raise ValueError("actual target IR differs from decoded checked program")
