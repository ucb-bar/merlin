"""Bind a whole-source plan to current CPU/device operation identities.

This leaf preserves source IR. It supplies original/current operation objects to
an existing offload rewrite, never instruction selection, capability admission or
performance qualification. Only singleton implementations with the unchanged
logical tensor ABI are realizable here. Unknown effects and structured control
remain barriers; no implicit fusion, residency or representation transfer exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from xdsl.dialects import arith, builtin, func, linalg, math, tensor
from xdsl.dialects.linalg.ops import LinalgStructuredOperation

from merlin.transforms.certificates import ir_digest
from merlin.transforms.dense_ir import recognize_signed_integer_matmul
from merlin.transforms.pipeline_state import (
    AnalysisStamp,
    PassSpec,
    PipelineContext,
    PipelineSession,
    validate_pass_order,
)
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program, verify_program
from merlin.xdsl_dialects.lowering.global_plan import (
    GLOBAL_PLAN_SCHEMA,
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
    verify_global_plan,
)
from merlin.xdsl_dialects.lowering.global_plan_emission import _component_boundary, dispatch_digest
from merlin.xdsl_dialects.lowering.outline import outline_dispatches

from .prepared_operand_effects import validate_pure_tensor_function


class OffloadPlanUnavailable(ValueError):
    """This source has no supported flat semantic binding; retain its legal route."""


def _require(claim, message):
    if not claim:
        raise ValueError("source offload plan: " + message)


def _record(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context(context):
    _require(type(context) is PipelineContext, "an original pipeline context is required")
    context.__post_init__()
    return _record(asdict(context))


def _read(module, entry, max_operations):
    if type(module) is not builtin.ModuleOp or type(entry) is not str or not entry:
        raise OffloadPlanUnavailable("source offload plan requires a typed module and explicit entry")
    if type(max_operations) is not int or not 0 < max_operations <= 100_000:
        raise ValueError("source offload plan requires a bounded positive operation budget")
    walked = []
    for operation in module.walk():
        if len(walked) >= max_operations:
            raise OffloadPlanUnavailable("source offload operation budget exceeded")
        walked.append(operation)
    module.verify()
    functions = tuple(module.body.block.ops)
    entries = [op for op in functions if type(op) is func.FuncOp and op.sym_name.data == entry]
    if any(type(op) is not func.FuncOp for op in functions) or len(entries) != 1:
        raise OffloadPlanUnavailable("source offload requires one exact root entry without opaque module operations")
    function = entries[0]
    if len(function.body.blocks) != 1 or type(function.body.block.last_op) is not func.ReturnOp:
        raise OffloadPlanUnavailable("source offload requires a complete flat single-block entry")
    operations = tuple(function.body.block.ops)[:-1]
    for typ in (*function.function_type.inputs, *function.function_type.outputs):
        if type(typ) is builtin.TensorType and (
            type(typ.encoding) is not builtin.NoneAttr or any(d <= 0 for d in typ.get_shape())
        ):
            raise OffloadPlanUnavailable("source offload requires ordinary static entry tensor extents")
    registered = set(
        (*arith.Arith.operations, *math.Math.operations, *tensor.Tensor.operations, *linalg.Linalg.operations)
    )
    for operation in function.walk():
        if operation is function or type(operation) is func.ReturnOp:
            continue
        if (
            type(operation) not in registered
            or operation.successors
            or operation.regions
            and not isinstance(operation, LinalgStructuredOperation)
            or any(not key.startswith("prov.") for key in operation.attributes)
            or "library_call" in operation.properties
        ):
            raise OffloadPlanUnavailable("source offload has unknown effects, required placement or nested control")
        if isinstance(operation, LinalgStructuredOperation):
            if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
                raise OffloadPlanUnavailable("source offload requires one pure structured scalar region")
            if any(sub.regions or sub.successors for sub in operation.regions[0].walk()):
                raise OffloadPlanUnavailable("source offload scalar region contains nested control")
        for value in (*operation.operands, *operation.results):
            if type(value.type) is builtin.TensorType and (
                type(value.type.encoding) is not builtin.NoneAttr or any(d <= 0 for d in value.type.get_shape())
            ):
                raise OffloadPlanUnavailable("source offload requires ordinary static positive tensor extents")
    if set(function.properties) - {"sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs"}:
        raise OffloadPlanUnavailable("source offload entry carries unknown execution properties")
    if any(key != "llvm.emit_c_interface" and not key.startswith("prov.") for key in function.attributes):
        raise OffloadPlanUnavailable("source offload entry carries unknown or required execution placement")
    if module.properties or any(not key.startswith("prov.") for key in module.attributes):
        raise OffloadPlanUnavailable("source offload module carries unknown execution attributes")
    try:
        validate_pure_tensor_function(function, {entry: function}, allow_named_linalg=True)
    except ValueError as error:
        raise OffloadPlanUnavailable("source offload effects are not established: " + str(error)) from error
    outlined = outline_dispatches(module, forward=entry)
    program = build_dispatch_program(outlined, entry=entry)
    problems = verify_program(program)
    _require(not problems, "invalid actual source graph: " + "; ".join(problems))
    _require(len(program.nodes) == len(operations), "outlining changed source top-operation correspondence")
    ordinals = {op: index for index, op in enumerate(walked)}
    return operations, tuple(ordinals[op] for op in operations), outlined, program


def _cpu_alternatives(program):
    result = []
    order = {name: index for index, name in enumerate(program.buffers)}
    for index, node in enumerate(program.nodes):
        inputs, outputs = _component_boundary(program, (index,))

        def representations(names):
            return tuple(
                BufferRepresentation(name, ValueRepresentation("tensor_ssa", "logical", program.buffers[name].dtype))
                for name in sorted(names, key=order.__getitem__)
            )

        result.append(
            RegionAlternative(
                id=f"cpu_{index}",
                nodes=(index,),
                implementation=node.op,
                placement="host",
                cycles=CycleInterval.unknown("CPU cost is not supplied by source binding"),
                inputs=representations(inputs),
                outputs=representations(outputs),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class OffloadSource:
    module: builtin.ModuleOp
    entry: str
    operations: tuple
    operation_ordinals: tuple[int, ...]
    program: object
    cpu_alternatives: tuple[RegionAlternative, ...]
    stamp: AnalysisStamp
    max_operations: int
    _session: PipelineSession
    _original_context: PipelineContext
    _context_record: str

    def revalidate(self, *, context=None):
        current = self._original_context if context is None else context
        _require(_context(current) == self._context_record, "original context changed; replan")
        _require(_context(self._session.context) == self._context_record, "bound context changed; replan")
        _require(self._session.module is self.module, "bound source owner changed; replan")
        saved = self._session.analysis("source_graph")
        _require(
            saved == (self.stamp, self.operations, self.operation_ordinals, self.max_operations),
            "source binding changed; replan",
        )
        operations, ordinals, _, graph = _read(self.module, self.entry, self.max_operations)
        _require(
            operations == self.operations and ordinals == self.operation_ordinals,
            "source operation ownership changed; replan",
        )
        _require(ir_digest(self.module) == self.stamp.source_sha256, "source bytes changed; replan")
        _require(dispatch_digest(graph) == dispatch_digest(self.program), "source graph changed; replan")
        _require(self.cpu_alternatives == _cpu_alternatives(graph), "original CPU alternatives changed; replan")


def bind_source(module, *, entry, context, max_operations=4096):
    """Read every current entry operation, including glue; never prune the graph."""
    context_record = _context(context)
    frozen_context = PipelineContext(**asdict(context))
    operations, ordinals, _, program = _read(module, entry, max_operations)
    session = PipelineSession(module, context=frozen_context, ir_level="linalg", max_repairs=0)
    stamp = session.stamp()
    spec = PassSpec("offload-source", 0, produces=("source_graph",), mutates_ir=False, ir_level="linalg")
    session.run(spec, lambda: {"source_graph": (stamp, operations, ordinals, max_operations)})
    result = OffloadSource(
        module,
        entry,
        operations,
        ordinals,
        program,
        _cpu_alternatives(program),
        stamp,
        max_operations,
        session,
        context,
        context_record,
    )
    result.revalidate()
    return result


def _structural_alternative(alternative):
    record = alternative.to_dict()
    for key in ("cycles", "demands", "occupancy"):
        del record[key]
    return record


def _checked_choice(source, plan, alternatives):
    _require(type(plan) is GlobalPlan, "a typed original global plan is required")
    _require(
        type(alternatives) is tuple and all(type(a) is RegionAlternative for a in alternatives),
        "an immutable original device alternative roster is required",
    )
    _require(
        type(plan.selected) is tuple
        and type(plan.transitions) is tuple
        and type(plan.cycles) is CycleInterval
        and plan.schema == GLOBAL_PLAN_SCHEMA,
        "global plan has unsupported fields or schema",
    )
    plan.cycles.__post_init__()
    for alternative in (*alternatives, *plan.selected):
        _require(
            type(alternative) is RegionAlternative
            and type(alternative.nodes) is tuple
            and all(type(index) is int and index >= 0 for index in alternative.nodes)
            and type(alternative.cycles) is CycleInterval,
            "original alternatives require typed nonnegative singleton indices and costs",
        )
        alternative.cycles.__post_init__()
        alternative.__post_init__()
    problems = verify_global_plan(source.program, plan)
    _require(not problems, "invalid global plan: " + "; ".join(problems))
    _require(
        not plan.transitions and not plan.arena_bytes and not plan.arena_offsets,
        "representation transitions or planned allocation have no physical realization",
    )
    placements = {alternative.placement for alternative in alternatives}
    _require(
        len(placements) <= 1 and "host" not in placements, "device alternatives require one distinct device endpoint"
    )
    by_id = {}
    for alternative in alternatives:
        _require(alternative.id not in by_id, "original device alternatives repeat an identity")
        _require(
            len(alternative.nodes) == 1 and 0 <= alternative.nodes[0] < len(source.operations),
            "device alternative is not a source singleton",
        )
        index = alternative.nodes[0]
        operation = source.operations[index]
        try:
            dtypes = tuple(str(value.type.element_type) for value in (*operation.inputs, *operation.outputs))
            recognize_signed_integer_matmul(operation, dtypes)
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "source offload plan: device alternative does not bind exact signed contraction semantics"
            ) from error
        expected = source.cpu_alternatives[index]
        _require(
            alternative.inputs == expected.inputs and alternative.outputs == expected.outputs,
            "device alternative changes the original logical host-buffer ABI",
        )
        _require(
            not alternative.resident_inputs and not alternative.resident_outputs and not alternative.scratch_bytes,
            "device alternative requires unrealized residency or scratch",
        )
        by_id[alternative.id] = alternative
    selected = set()
    for alternative in plan.selected:
        _require(
            type(alternative) is RegionAlternative and len(alternative.nodes) == 1,
            "offload realization requires original singleton implementations",
        )
        index = alternative.nodes[0]
        expected = source.cpu_alternatives[index]
        if alternative.placement == "host":
            _require(
                _structural_alternative(alternative) == _structural_alternative(expected),
                "CPU implementation differs from its original source identity",
            )
        else:
            _require(
                alternative.id in by_id and alternative == by_id[alternative.id],
                "selected device implementation is not an original admitted alternative",
            )
            selected.add(index)
    return tuple(index for index in range(len(source.operations)) if index in selected)


@dataclass(frozen=True)
class OffloadPlanSelection:
    source: OffloadSource
    plan: GlobalPlan
    device_alternatives: tuple[RegionAlternative, ...]
    selected_ops: tuple
    selected_ordinals: tuple[int, ...]
    retained_cpu_ops: tuple
    stamp: AnalysisStamp
    _session: PipelineSession
    _plan_record: str
    _alternatives_record: str

    @property
    def entry(self):
        return self.source.entry

    @property
    def covered_ordinals(self):
        return self.source.operation_ordinals

    def revalidate(self, *, context=None):
        self.source.revalidate(context=context)
        _require(_record(self.plan.to_dict()) == self._plan_record, "original plan changed; replan")
        _require(
            _record([a.to_dict() for a in self.device_alternatives]) == self._alternatives_record,
            "original device alternatives changed; replan",
        )
        _require(
            _context(self._session.context) == self.source._context_record, "selection pipeline context changed; replan"
        )
        saved = self._session.analysis("checked_selection")
        _require(
            saved == (self._plan_record, self._alternatives_record, self.stamp), "selection context changed; replan"
        )
        indices = _checked_choice(self.source, self.plan, self.device_alternatives)
        _require(self.stamp == self.source.stamp, "selected source stamp changed; replan")
        _require(
            self.selected_ops == tuple(self.source.operations[index] for index in indices)
            and self.selected_ordinals == tuple(self.source.operation_ordinals[index] for index in indices)
            and self.retained_cpu_ops
            == tuple(op for index, op in enumerate(self.source.operations) if index not in indices),
            "realized source membership changed; replan",
        )

    def select_for(self, module, *, context=None):
        """Rebind exact equal source to current objects, never by shape or provenance."""
        self.revalidate(context=context)
        if module is self.source.module:
            return self.selected_ops
        operations, ordinals, _, graph = _read(module, self.entry, self.source.max_operations)
        _require(
            ir_digest(module) == self.stamp.source_sha256
            and ordinals == self.covered_ordinals
            and dispatch_digest(graph) == dispatch_digest(self.source.program),
            "current source or graph differs from original plan; replan",
        )
        selected = set(self.selected_ordinals)
        return tuple(op for op, ordinal in zip(operations, ordinals, strict=True) if ordinal in selected)

    def record(self):
        self.revalidate()
        return dict(
            schema="merlin.source_offload_plan.v1",
            source_sha256=self.stamp.source_sha256,
            graph_sha256=dispatch_digest(self.source.program),
            plan_sha256=self.plan.digest,
            device_alternatives=[alternative.to_dict() for alternative in self.device_alternatives],
            context=asdict(self.stamp.context),
            entry=self.entry,
            covered_ordinals=list(self.covered_ordinals),
            selected_ordinals=list(self.selected_ordinals),
            retained_cpu_ordinals=[
                ordinal for ordinal in self.covered_ordinals if ordinal not in self.selected_ordinals
            ],
            passes=[asdict(record) for record in self._session.records],
            scope="source/graph/implementation correspondence only; no runtime, capability or cost qualification",
        )


def realize_offload_plan(source, plan, *, device_alternatives, context):
    """Check original alternatives and return a selection for the existing rewrite."""
    _require(type(source) is OffloadSource, "an original typed source binding is required")
    source.revalidate(context=context)
    _require(
        type(plan) is GlobalPlan
        and type(device_alternatives) is tuple
        and all(type(alternative) is RegionAlternative for alternative in device_alternatives),
        "typed original plan and immutable device alternatives required",
    )
    plan_record = _record(plan.to_dict())
    alternatives_record = _record([a.to_dict() for a in device_alternatives])
    session = PipelineSession(
        source.module, context=PipelineContext(**asdict(source.stamp.context)), ir_level="linalg", max_repairs=0
    )
    specs = (
        PassSpec("offload-source", 0, produces=("graph",), mutates_ir=False, ir_level="linalg"),
        PassSpec("offload-plan", 1, requires=("graph",), produces=("selection",), mutates_ir=False, ir_level="linalg"),
        PassSpec(
            "offload-conformance",
            2,
            requires=("selection",),
            produces=("checked_selection",),
            mutates_ir=False,
            ir_level="linalg",
        ),
    )
    validate_pass_order(specs)
    session.run(specs[0], lambda: {"graph": source.program})
    session.run(specs[1], lambda: {"selection": _checked_choice(source, plan, device_alternatives)})
    indices = session.analysis("selection")
    session.run(specs[2], lambda: {"checked_selection": (plan_record, alternatives_record, source.stamp)})
    result = OffloadPlanSelection(
        source,
        plan,
        device_alternatives,
        tuple(source.operations[i] for i in indices),
        tuple(source.operation_ordinals[i] for i in indices),
        tuple(op for i, op in enumerate(source.operations) if i not in indices),
        source.stamp,
        session,
        plan_record,
        alternatives_record,
    )
    result.revalidate(context=context)
    return result
