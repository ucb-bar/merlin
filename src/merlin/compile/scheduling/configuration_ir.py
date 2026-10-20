"""Persistent configuration cleanup on actual, declared instruction MLIR.

The existing schedule algorithm owns deletion legality. This adapter reads the
selected generated ISA schema, preserves every other SSA operation and checks
the realized IR independently. It interprets neither opaque backend artifacts
nor target instruction names. Execution qualification is an original caller
premise, separate from structural facts, declarations and these checks.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field

from xdsl.dialects import arith, builtin, func, memref, scf

from merlin.sched.ir import Call, Const, Kernel
from merlin.targetgen.isa_dialect_gen import build_dialect, decode_instruction, encode_instruction
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order

from .configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    ConfigurationUnavailable,
    _effects,
    eliminate_redundant_configuration,
    verify_configuration_rewrite,
)
from .library_sources import shared_library_sources
from .selected_contracts import SelectedSchedulingContracts, _parse


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _bounded_text(module, maximum):
    from merlin.common.dense_float_ir import DenseFloatPrinter

    class LimitedText(io.StringIO):
        size = 0

        def write(self, value):
            self.size += len(value.encode("utf-8"))
            if self.size > maximum:
                raise ValueError("configuration IR exceeds its explicit input byte budget")
            return super().write(value)

    output = LimitedText()
    DenseFloatPrinter(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def _selected(selected):
    if type(selected) is not SelectedSchedulingContracts:
        raise ValueError("configuration IR requires an original typed scheduling selection")
    selected.revalidate()
    isa, effects, policy, byteorder, *_ = _parse(selected.facts_bytes, selected.software_bytes, selected.target)
    if (isa, effects, policy, byteorder) != (
        selected.isa,
        selected.effects_json,
        selected.numerical_policy,
        selected.instruction_byteorder,
    ):
        raise ValueError("configuration IR selection differs from original declarations; reselect")
    return dict(selected=selected.record(), effects=effects)


@dataclass(frozen=True)
class ConfigurationIRSelection:
    """Original-owned input; a qualification identifier does not establish support.

    Missing qualification, unavailable effects or exceeded analysis budgets
    retain unchanged IR. The caller freezes this object before invoking a
    compiler package and separately qualifies its instruction implementation.
    """

    selected_contracts: SelectedSchedulingContracts
    qualification_identity: str | None
    enabled: bool = True
    max_operations: int = 4096
    max_fields: int = 256
    max_ir_bytes: int = 8 << 20
    _original: str = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.enabled) is not bool
            or any(type(n) is not int or not 0 < n <= 1 << 20 for n in (self.max_operations, self.max_fields))
            or type(self.max_ir_bytes) is not int
            or not 0 < self.max_ir_bytes <= 1 << 30
            or self.qualification_identity is not None
            and (type(self.qualification_identity) is not str or not self.qualification_identity.strip())
        ):
            raise ValueError("configuration IR needs explicit qualification and bounded typed controls")
        object.__setattr__(self, "_original", _json(self._current()))

    def _current(self):
        return dict(
            schema="merlin.configuration_ir_selection.v1",
            **_selected(self.selected_contracts),
            qualification_identity=self.qualification_identity,
            enabled=self.enabled,
            max_operations=self.max_operations,
            max_fields=self.max_fields,
            max_ir_bytes=self.max_ir_bytes,
            library=shared_library_sources(),
        )

    def revalidate(self):
        if _json(self._current()) != self._original:
            raise ValueError("original configuration IR selection or library changed; reselect")

    def record(self):
        self.revalidate()
        return json.loads(self._original)

    def parse(self, text):
        """Parse the original generated ISA; opaque operations remain barriers."""
        from xdsl.context import Context

        from merlin.common.dense_float_ir import DenseFloatParser

        self.revalidate()
        if type(text) is not str or len(text) > self.max_ir_bytes or len(text.encode("utf-8")) > self.max_ir_bytes:
            raise ValueError("configuration IR exceeds its explicit input byte budget")
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, func.Func, arith.Arith, scf.Scf, memref.MemRef):
            context.load_dialect(dialect)
        context.load_dialect(build_dialect(self.selected_contracts.isa))
        module = DenseFloatParser(context, text).parse_module()
        module.verify()
        self.revalidate()
        return module

    def apply(self, module):
        return _eliminate(module, self)


def _constant(op):
    return (
        type(op) is arith.ConstantOp
        and not op.attributes
        and set(op.properties) == {"value"}
        and not op.operands
        and len(op.results) == 1
        and not op.regions
        and not op.successors
        and type(op.value) is builtin.IntegerAttr
        and type(op.results[0].type) in (builtin.IntegerType, builtin.IndexType)
        and op.value.type == op.results[0].type
    )


def _scope(op):
    """Opaque region semantics do not establish sequential block execution."""
    if type(op) is builtin.ModuleOp:
        return len(op.regions) == 1 and len(op.body.blocks) == 1
    if type(op) is func.FuncOp:
        return len(op.body.blocks) <= 1
    return type(op) in (scf.ForOp, scf.IfOp) and not op.attributes and not op.properties


def _blocks(op):
    if _scope(op):
        for region in op.regions:
            for block in region.blocks:
                yield block
                for child in block.ops:
                    yield from _blocks(child)


def _names(isa):
    names = {instruction.name for instruction in isa.instructions}
    opaque, scalar = "__merlin_ir_unknown", "__merlin_ir_constant"
    while opaque in names:
        opaque += "_"
    while scalar in names:
        scalar += "_"
    return opaque, scalar


def _call(op, selected, names):
    opaque, scalar = names
    if _constant(op):
        return Call(scalar, ())
    instructions = {selected.isa.namespace + "." + ins.name: ins for ins in selected.isa.instructions}
    ins = instructions.get(op.name)
    if (
        ins is None
        or ins.results
        or op.results
        or op.regions
        or op.successors
        or op.properties
        or len(op.operands) != len(ins.operands)
        or set(op.attributes) != {a.name for a in ins.immediates}
        or any(a.type.kind != "integer" for a in (*ins.operands, *ins.immediates))
        or {f.argument for f in ins.fields if f.argument is not None} != {a.name for a in ins.arguments()}
    ):
        return Call(opaque, ())
    values = {}
    for argument, value in zip(ins.operands, op.operands, strict=True):
        if value.type != builtin.IntegerType(argument.type.bits) or not _constant(value.owner):
            return Call(opaque, ())
        integer = value.owner.value.value.data
        values[argument.name] = integer if argument.type.signed else integer & ((1 << argument.type.bits) - 1)
    for argument in ins.immediates:
        value = op.attributes[argument.name]
        if type(value) is not builtin.IntegerAttr or value.type != builtin.IntegerType(argument.type.bits):
            return Call(opaque, ())
        values[argument.name] = (
            value.value.data if argument.type.signed else value.value.data & ((1 << argument.type.bits) - 1)
        )
    word = encode_instruction(selected.isa, ins.name, values)
    name, decoded = decode_instruction(selected.isa, word)
    if name != ins.name or decoded != values:
        raise ValueError("configuration IR instruction has no exact original encoding")
    return Call(name, tuple((key, Const(value)) for key, value in values.items()))


def _adapter(selection, names):
    selected = selection.selected_contracts
    original = selected.configuration_adapter(selection.qualification_identity)

    def project(call):
        if call.instr == names[1]:
            return ConfigurationEffects(ordered=True)
        if call.instr == names[0]:
            return None
        return original.project(call)

    return ConfigurationAdapter(project, original.configuration_identity, original.qualification_identity)


def _unavailable(module, selection):
    if not selection.enabled:
        return "disabled"
    if selection.qualification_identity is None:
        return "independent execution qualification unavailable"
    if not json.loads(selection.selected_contracts.effects_json):
        return "configuration effects unavailable"
    for count, _ in enumerate(module.walk(), 1):
        if count > selection.max_operations:
            return "operation analysis budget exceeded"
    names = _names(selection.selected_contracts.isa)
    adapter = _adapter(selection, names)
    try:
        for block in _blocks(module):
            calls = tuple(_call(op, selection.selected_contracts, names) for op in block.ops)
            _effects(Kernel("block", (), calls), adapter, selection.max_operations, selection.max_fields)
    except ConfigurationUnavailable:
        return "configuration field analysis budget exceeded"
    return None


def _header(left, right, mapping):
    return (
        left.name == right.name
        and left.attributes == right.attributes
        and left.properties == right.properties
        and tuple(v.type for v in left.results) == tuple(v.type for v in right.results)
        and tuple(mapping.get(v, v) for v in left.operands) == tuple(right.operands)
        and tuple(mapping.get(b, b) for b in left.successors) == tuple(right.successors)
        and len(left.regions) == len(right.regions)
    )


def _opaque(left, right, mapping):
    """Preserve unknown nested SSA exactly, including types omitted by structural equivalence."""
    before, after, local = tuple(left.walk()), tuple(right.walk()), dict(mapping)
    if len(before) != len(after):
        return False
    for a, b in zip(before, after, strict=True):
        if len(a.results) != len(b.results) or len(a.regions) != len(b.regions):
            return False
        local.update(zip(a.results, b.results, strict=True))
        for ar, br in zip(a.regions, b.regions, strict=True):
            if len(ar.blocks) != len(br.blocks):
                return False
            for ab, bb in zip(ar.blocks, br.blocks, strict=True):
                if tuple(v.type for v in ab.args) != tuple(v.type for v in bb.args) or len(tuple(ab.ops)) != len(
                    tuple(bb.ops)
                ):
                    return False
                local[ab] = bb
                local.update(zip(ab.args, bb.args, strict=True))
    return all(_header(a, b, local) for a, b in zip(before, after, strict=True))


def verify_configuration_ir(original, actual, *, selection):
    """Check actual omissions and every retained SSA operation, not a chosen plan."""
    if type(selection) is not ConfigurationIRSelection:
        raise ValueError("configuration IR verification requires an original typed selection")
    selection.revalidate()
    _bounded_text(original, selection.max_ir_bytes)
    _bounded_text(actual, selection.max_ir_bytes)
    original.verify()
    actual.verify()
    unavailable = _unavailable(original, selection)
    if unavailable is not None:
        if not _opaque(original, actual, {}):
            raise ValueError("configuration IR unavailable route changed original program")
        rows = []
    else:
        names, rows, mapping = _names(selection.selected_contracts.isa), [], {}
        adapter = _adapter(selection, names)

        def operation(left, right):
            if not _header(left, right, mapping):
                raise ValueError("configuration IR changes a retained operation or SSA binding")
            mapping.update(zip(left.results, right.results, strict=True))
            if not _scope(left):
                if not _opaque(left, right, mapping):
                    raise ValueError("configuration IR changes an opaque region")
                return
            for before_region, after_region in zip(left.regions, right.regions, strict=True):
                if len(before_region.blocks) != len(after_region.blocks):
                    raise ValueError("configuration IR changes control-flow blocks")
                for before_block, after_block in zip(before_region.blocks, after_region.blocks, strict=True):
                    if tuple(v.type for v in before_block.args) != tuple(v.type for v in after_block.args):
                        raise ValueError("configuration IR changes block argument types")
                    mapping[before_block] = after_block
                    mapping.update(zip(before_block.args, after_block.args, strict=True))
                    before, after = tuple(before_block.ops), tuple(after_block.ops)
                    source_calls = tuple(_call(op, selection.selected_contracts, names) for op in before)
                    actual_calls = tuple(_call(op, selection.selected_contracts, names) for op in after)
                    proof = verify_configuration_rewrite(
                        Kernel("block", (), source_calls),
                        Kernel("block", (), actual_calls),
                        adapter,
                        max_calls=selection.max_operations,
                        max_fields=selection.max_fields,
                    )
                    rows.append(proof)
                    cursor = 0
                    for op, call in zip(before, source_calls, strict=True):
                        if cursor < len(after) and _header(op, after[cursor], mapping):
                            operation(op, after[cursor])
                            cursor += 1
                        elif (effect := adapter.project(call)) is None or not effect.pure_write:
                            raise ValueError("configuration IR omits a retained operation")
                    if cursor != len(after):
                        raise ValueError("configuration IR inserts unreported operations")

        operation(original, actual)
    selection.revalidate()
    return dict(
        schema="merlin.configuration_ir_rewrite.v1",
        source_sha256=ir_digest(original),
        output_sha256=ir_digest(actual),
        removed_operations=sum(row["removed_calls"] for row in rows),
        blocks=rows,
        fallback=unavailable,
        selection=selection.record(),
        execution_qualification="not_established",
        timing="UNKNOWN",
        scope="declared instruction IR, original codec and retained SSA; native emission and execution unqualified",
    )


@dataclass(frozen=True)
class ConfigurationIRResult:
    source: builtin.ModuleOp
    module: builtin.ModuleOp
    selection: ConfigurationIRSelection
    source_sha256: str
    output_sha256: str
    selection_identity: str
    report: dict
    _report_identity: str

    def revalidate(self):
        self.selection.revalidate()
        _bounded_text(self.source, self.selection.max_ir_bytes)
        _bounded_text(self.module, self.selection.max_ir_bytes)
        if (
            ir_digest(self.source) != self.source_sha256
            or ir_digest(self.module) != self.output_sha256
            or _hash(self.selection.record()) != self.selection_identity
            or _hash(self.report) != self._report_identity
            or verify_configuration_ir(self.source, self.module, selection=self.selection)
            != self.report["verification"]
        ):
            raise ValueError("checked configuration IR or original selection changed; replan")


def _eliminate(module, selection):
    from xdsl.rewriter import Rewriter

    selection.revalidate()
    _bounded_text(module, selection.max_ir_bytes)
    module.verify()
    original, current = module.clone(), module.clone()
    context = PipelineContext(
        hardware=selection.selected_contracts.hardware_identity,
        numerical_policy=selection.selected_contracts.numerical_policy,
        execution=_hash(selection.record()),
        library=_hash(shared_library_sources()),
    )
    session = PipelineSession(current, context=context, ir_level="target_instruction")
    specs = (
        PassSpec(
            "configuration-ir-source", 0, produces=("configuration",), mutates_ir=False, ir_level="target_instruction"
        ),
        PassSpec(
            "configuration-ir-cleanup",
            1,
            requires=("configuration",),
            produces=("realized",),
            invalidates=("*",),
            ir_level="target_instruction",
        ),
        PassSpec(
            "configuration-ir-conformance",
            1,
            requires=("realized",),
            produces=("checked",),
            mutates_ir=False,
            ir_level="target_instruction",
        ),
    )
    validate_pass_order(specs)
    unavailable = _unavailable(original, selection)
    names = _names(selection.selected_contracts.isa)
    adapter = None if unavailable else _adapter(selection, names)

    def analyze():
        return {
            "configuration": ()
            if unavailable
            else tuple(
                (block, tuple(block.ops), tuple(_call(op, selection.selected_contracts, names) for op in block.ops))
                for block in _blocks(current)
            )
        }

    def transform():
        for _, operations, calls in session.analysis("configuration"):
            kept = eliminate_redundant_configuration(
                Kernel("block", (), calls), adapter, max_calls=selection.max_operations, max_fields=selection.max_fields
            ).kernel.body
            cursor = 0
            for operation, call in zip(operations, calls, strict=True):
                if cursor < len(kept) and call == kept[cursor]:
                    cursor += 1
                else:
                    Rewriter.erase_op(operation)
        return {"realized": True}

    for spec, action in zip(
        specs,
        (analyze, transform, lambda: {"checked": verify_configuration_ir(original, current, selection=selection)}),
        strict=True,
    ):
        selection.revalidate()
        session.run(spec, action)
        selection.revalidate()
    report = dict(verification=session.analysis("checked"), passes=[asdict(row) for row in session.records])
    return ConfigurationIRResult(
        original,
        current,
        selection,
        ir_digest(original),
        ir_digest(current),
        _hash(selection.record()),
        report,
        _hash(report),
    )


def eliminate_configuration_ir(
    module,
    *,
    selected_contracts,
    qualification_identity,
    enabled=True,
    max_operations=4096,
    max_fields=256,
    max_ir_bytes=8 << 20,
):
    return ConfigurationIRSelection(
        selected_contracts, qualification_identity, enabled, max_operations, max_fields, max_ir_bytes
    ).apply(module)
