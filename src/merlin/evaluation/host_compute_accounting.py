"""Account for tensor-data arithmetic in emitted host code independently of placement claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.analysis.optimization.candidates import gate_phase
from merlin.evaluation.route_admission import SCHEMA, STATUS_REPORTED

# --------------------------------------------------------------------------------------------
# Tier 2 — host compute inside an accepted region
# --------------------------------------------------------------------------------------------


@dataclass
class TaskCompute:
    """One program task's host arithmetic, split by what the arithmetic reaches."""

    task: str
    route: str
    #: Arithmetic whose operand chain reaches a load from a tensor buffer. The defect, in an ``A``.
    on_tensor: int = 0
    #: Arithmetic that reaches only induction variables, constants and pointer roots. Addressing.
    on_addressing: int = 0
    #: Arithmetic whose chain reaches something this walk cannot classify (a call, inline assembly).
    undecided: int = 0
    #: Operation indices of the first few tensor-reaching findings, so a reader can open the IR.
    witnesses: list[int] = field(default_factory=list)
    #: What made a chain undecided, deduplicated.
    undecided_causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "route": self.route,
            "arithmetic_on_tensor_data": self.on_tensor,
            "arithmetic_on_addressing": self.on_addressing,
            "arithmetic_undecided": self.undecided,
            "witness_operation_indices": self.witnesses[:8],
            "undecided_causes": sorted(set(self.undecided_causes))[:8],
        }


@dataclass(frozen=True)
class HostComputeReport:
    gate: str
    status: str
    tasks: tuple[TaskCompute, ...]
    causes: tuple[str, ...] = ()

    @property
    def findings(self) -> tuple[TaskCompute, ...]:
        """Accepted tasks that compute on tensor data on the host. The defect."""
        return tuple(t for t in self.tasks if t.route == "A" and t.on_tensor)

    @property
    def coverage(self) -> int:
        """Host arithmetic on tensor data in ``H`` tasks. Where host work is SUPPOSED to be."""
        return sum(t.on_tensor for t in self.tasks if t.route == "H")

    def blocks(self, phase: str | None = None) -> bool:
        resolved = gate_phase.configured_phase(self.gate) if phase is None else phase
        return gate_phase.blocks(resolved, self.status, failing=(STATUS_REPORTED,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "gate": self.gate,
            "status": self.status,
            "causes": list(self.causes),
            "host_coverage_arithmetic_on_tensor_data": self.coverage,
            "accepted_tasks_computing_on_tensor_data": len(self.findings),
            "tasks": [t.to_dict() for t in self.tasks],
        }


class HostComputeViolation(RuntimeError):
    """An accepted accelerator task performs known tensor arithmetic on the host."""


class HostComputeUnverified(RuntimeError):
    """The host-compute report cannot prove the accepted tasks clean."""


def require_clean_host_compute(report: HostComputeReport) -> HostComputeReport:
    """Require a decided, clean host-compute result for an offload claim.

    This is a stricter admission guard than :meth:`HostComputeReport.blocks`:
    the latter retains the configured report phase for legacy reporting.
    An incomplete or malformed result never grants offload qualification.
    """
    from merlin.evaluation.route_admission import GATE_HOST_COMPUTE, STATUS_OK

    if not isinstance(report, HostComputeReport) or report.gate != GATE_HOST_COMPUTE:
        raise HostComputeUnverified("a host-compute report for the selected route is required")
    findings = report.findings
    if findings:
        tasks = ", ".join(row.task for row in findings[:8])
        raise HostComputeViolation(f"accepted task(s) {tasks} compute on tensor data on the host")
    if report.status != STATUS_OK or report.causes:
        undecided = [cause for row in report.tasks if row.route == "A" for cause in row.undecided_causes]
        why = "; ".join((*report.causes, *undecided)) or f"status is {report.status!r}"
        raise HostComputeUnverified(f"host compute is not fully verified: {why}")
    return report


def route_by_task(command_buffer: Mapping[str, Any]) -> tuple[dict[str, str], tuple[str, ...]]:
    """``({task id: "A"|"H"}, causes)`` from the program plan's own task kinds.

    A task the plan calls a host task is route ``H``; every other kind names device work the plan
    committed to emitting, which is route ``A``. The split is the PLAN's, not a lane vocabulary
    invented here, so a target whose device kinds are spelled differently needs no edit.
    """
    from merlin.evaluation.route_admission import _HOST_TASK_KIND

    params = command_buffer.get("params")
    params = params if isinstance(params, Mapping) else {}
    plan = params.get("global_program_plan")
    if not isinstance(plan, Mapping):
        return {}, ("params.global_program_plan is absent, so no host IR operation can be attributed to a route",)
    tasks = plan.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        return {}, ("the program plan declares no task list, so no host IR operation can be attributed to a route",)
    out: dict[str, str] = {}
    causes: list[str] = []
    for entry in tasks:
        if not isinstance(entry, Mapping):
            causes.append("a program-plan task entry is not a mapping and was not attributed")
            continue
        index = entry.get("task_index")
        kind = entry.get("kind")
        if index is None or kind is None:
            causes.append(f"a program-plan task declares no {'task_index' if index is None else 'kind'}")
            continue
        out[str(index)] = "H" if str(kind) == _HOST_TASK_KIND else "A"
    return out, tuple(causes)


def _tensor_arguments(command_buffer: Mapping[str, Any]) -> frozenset[int]:
    """Entry-block argument positions that carry a tensor buffer, from the kernel ABI itself."""
    abi = command_buffer.get("kernel_abi")
    args = abi.get("args") if isinstance(abi, Mapping) else None
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        return frozenset()
    return frozenset(i for i, a in enumerate(args) if isinstance(a, Mapping) and a.get("tensor"))


def _pointer_root(value: Any, entry: Any, op_index: Mapping[Any, int]) -> str:
    """The SSA root of a pointer: an entry argument, a stack slot, a module buffer, or UNKNOWN.

    A SECOND pointer-root walk, deliberately. :mod:`merlin.analysis.programs.cfg_activity` has one, but it
    is a closure and it reports a module-level buffer as ``UNKNOWN``; teaching it to resolve one
    would change the numbers that instrument has already published. Widening an instrument is a
    scheduled change, not a side effect of landing a check, so this walk is local and says so.
    """
    from xdsl.ir import Block

    seen: set[Any] = set()
    while value not in seen:
        seen.add(value)
        owner = getattr(value, "owner", None)
        if isinstance(owner, Block):
            return f"arg:{value.index}" if owner is entry else "UNKNOWN"
        if owner is None:
            return "UNKNOWN"
        name = _real_op_name(owner)
        if name == "llvm.alloca":
            return f"alloca:{op_index.get(owner, -1)}"
        if name in {"llvm.mlir.addressof", "llvm.mlir.addrspacecast"}:
            symbol = owner.properties.get("global_name") or owner.attributes.get("global_name")
            return f"global:{getattr(symbol, 'root_reference', symbol)}"
        if name in {"llvm.getelementptr", "llvm.bitcast"} and owner.operands:
            value = owner.operands[0]
            continue
        return "UNKNOWN"
    return "UNKNOWN"


def _real_op_name(operation: Any) -> str:
    from merlin.analysis.programs.cfg_activity import _real_name

    return _real_name(operation)


def _category_of(operation: Any) -> str:
    from merlin.analysis.programs.cfg_activity import _category

    return _category(_real_op_name(operation))


def _sibling_llvm_functions(function: Any) -> dict[str, Any] | None:
    """Resolve direct callees only from the selected function's own parsed module.

    A detached function, nested symbol or duplicate definition has no trustworthy local callee
    evidence. The call scanner treats that as unverified rather than guessing from a name.
    """
    block = getattr(function, "parent", None)
    region = getattr(block, "parent", None)
    module = getattr(region, "parent", None)
    if module is None or _real_op_name(module) != "builtin.module":
        return None
    symbols: dict[str, Any] = {}
    for op in module.body.block.ops:
        if _real_op_name(op) != "llvm.func":
            continue
        name = getattr(getattr(op, "sym_name", None), "data", None)
        if not isinstance(name, str) or name in symbols:
            return None
        symbols[name] = op
    return symbols


def host_compute(
    command_buffer: Mapping[str, Any],
    *,
    function: Any | None = None,
    absent_cause: str | None = None,
) -> HostComputeReport:
    """Find arithmetic on tensor data inside tasks the program routed ``A``.

    ``function`` is the parsed ``llvm.func`` of the package's host IR. Absent, the report is
    ``incomplete`` carrying ``absent_cause`` — never ``ok``, because a package with no host IR is
    one nobody looked inside.
    """
    from merlin.evaluation.route_admission import (
        _COMPUTE_CATEGORIES,
        GATE_HOST_COMPUTE,
        STATUS_INCOMPLETE,
        STATUS_OK,
        STATUS_REPORTED,
    )

    routes, causes = route_by_task(command_buffer)
    causes = list(causes)
    if function is None:
        causes.append(absent_cause or "the package carries no parsed host IR, so nothing was looked inside")
        return HostComputeReport(GATE_HOST_COMPUTE, STATUS_INCOMPLETE, (), tuple(causes))
    if not routes:
        return HostComputeReport(GATE_HOST_COMPUTE, STATUS_INCOMPLETE, (), tuple(causes))

    from merlin.analysis.programs.host_cfg_index import prepare_host_cfg

    cfg = prepare_host_cfg(function)
    if not cfg.blocks:
        causes.append("the host IR function has no block, so it carries no operation to classify")
        return HostComputeReport(GATE_HOST_COMPUTE, STATUS_INCOMPLETE, (), tuple(causes))
    siblings = _sibling_llvm_functions(function)
    tensor_args = _tensor_arguments(command_buffer)
    if not tensor_args:
        causes.append(
            "the buffer's kernel_abi names no tensor argument, so a load's pointer root cannot be "
            "told from a tensor buffer"
        )

    rows: dict[str, TaskCompute] = {}
    unattributed = 0

    def scan_function(
        func: Any,
        *,
        inherited_task: str | None,
        scalar_args: Mapping[int, tuple[bool, str | None]],
        pointer_args: Mapping[int, tuple[bool, str | None]],
        call_stack: frozenset[Any],
    ) -> tuple[bool, str | None]:
        """Scan one concrete call site; return whether its returned value reaches tensor data."""
        nonlocal unattributed
        blocks = tuple(func.body.blocks)
        if not blocks:
            return False, "a called llvm.func has no body"
        func_cfg = cfg if func is function else prepare_host_cfg(func)
        entry = blocks[0]
        operations = list(func.walk())
        op_index = {op: i for i, op in enumerate(operations)}
        stored_into: dict[str, list[Any]] = {}
        for op in operations:
            if _category_of(op) == "store" and len(op.operands) >= 2:
                stored_into.setdefault(_pointer_root(op.operands[1], entry, op_index), []).append(op.operands[0])
        value_verdict: dict[Any, tuple[bool, str | None]] = {}
        call_verdict: dict[Any, tuple[bool, str | None]] = {}

        def task_of(op: Any) -> str | None:
            owner = op.attributes.get("merlin.global_task")
            task = getattr(getattr(owner, "value", None), "data", None)
            if inherited_task is not None:
                if task is not None and str(task) != inherited_task:
                    return None  # a helper cannot silently change the caller's task attribution
                return inherited_task
            return str(task) if task is not None else None

        def row_for(op: Any) -> TaskCompute | None:
            nonlocal unattributed
            task = task_of(op)
            if task is None or task not in routes:
                unattributed += 1
                return None
            return rows.setdefault(task, TaskCompute(task=task, route=routes[task]))

        def scan_call(op: Any, task: str) -> tuple[bool, str | None]:
            if op in call_verdict:
                return call_verdict[op]
            name = _real_op_name(op)
            symbol = getattr(op, "callee", None) if name == "llvm.call" else None
            references = getattr(getattr(symbol, "nested_references", None), "data", ())
            callee_name = getattr(getattr(symbol, "root_reference", None), "data", None)
            callee = siblings.get(callee_name) if siblings is not None and not references else None
            if callee is None or not tuple(callee.body.blocks):
                result = (False, f"{name} has no resolved local callee body ({callee_name or 'indirect/opaque'})")
            elif callee in call_stack:
                result = (False, f"recursive call to {callee_name!r} has no finite helper proof")
            else:
                scalar = {i: reaches_tensor(arg, frozenset()) for i, arg in enumerate(op.args)}
                pointer: dict[int, tuple[bool, str | None]] = {}
                for i, arg in enumerate(op.args):
                    root = _pointer_root(arg, entry, op_index)
                    if root.startswith("arg:"):
                        position = int(root.partition(":")[2])
                        pointer[i] = pointer_args.get(position, (position in tensor_args, None))
                    elif root.startswith("global:"):
                        pointer[i] = (True, None)
                    elif root.startswith("alloca:"):
                        values = [reaches_tensor(stored, frozenset()) for stored in stored_into.get(root, ())]
                        pointer[i] = (
                            any(hit for hit, _ in values),
                            next((cause for _, cause in values if cause), None),
                        )
                    else:
                        pointer[i] = (False, "a called helper's pointer argument has no derivable root")
                result = scan_function(
                    callee,
                    inherited_task=task,
                    scalar_args=scalar,
                    pointer_args=pointer,
                    call_stack=call_stack | {callee},
                )
            call_verdict[op] = result
            if result[1] is not None:
                row = rows.setdefault(task, TaskCompute(task=task, route=routes[task]))
                row.undecided += 1
                row.undecided_causes.append(result[1])
            return result

        def reaches_tensor(value: Any, seen: frozenset) -> tuple[bool, str | None]:
            """``(reaches tensor data, undecided cause)`` for a value in this invocation."""
            hit, cause, _ = trace_value(value, seen)
            return hit, cause

        def trace_value(value: Any, seen: frozenset) -> tuple[bool, str | None, bool]:
            """Trace a value; the last bit marks a loop edge not yet proved by an entry path.

            A cycle is not a clean value by itself. A block argument with a known clean incoming
            seed and only clean transfer operations can close that cycle as clean; a cycle with no
            independently proved seed remains unverified. Cyclic provisional results are not
            memoized, so visiting a loop backedge before its seed cannot cache a false verdict.
            """
            from xdsl.ir import Block

            if value in value_verdict:
                hit, cause = value_verdict[value]
                return hit, cause, False
            if value in seen:
                return False, None, True
            owner = getattr(value, "owner", None)
            if isinstance(owner, Block):
                if owner is entry:
                    hit, cause = scalar_args.get(value.index, (False, None))
                    return hit, cause, False
                incoming: list[Any] = []
                for predecessor in func_cfg.predecessors.get(owner, ()):
                    terminal = predecessor.last_op
                    name = _real_op_name(terminal) if terminal is not None else ""
                    if name == "llvm.br" and terminal.successor is owner:
                        edges = (terminal.arguments,)
                    elif name == "llvm.cond_br":
                        edges = tuple(
                            arguments
                            for successor, arguments in (
                                (terminal.then_block, terminal.then_arguments),
                                (terminal.else_block, terminal.else_arguments),
                            )
                            if successor is owner
                        )
                    else:
                        edges = ()
                    for arguments in edges:
                        if len(arguments) != len(owner.args):
                            return False, "a CFG edge has mismatched block-argument arity", False
                        incoming.append(arguments[value.index])
                if not incoming:
                    return False, "a block argument has no proved incoming CFG value", False
                saw_independent = False
                worst: str | None = None
                for source in incoming:
                    hit, cause, cyclic = trace_value(source, seen | {value})
                    if hit:
                        value_verdict[value] = (True, None)
                        return True, None, False
                    if cause is not None:
                        worst = cause
                    saw_independent |= not cyclic
                if worst is not None:
                    value_verdict[value] = (False, worst)
                    return False, worst, False
                if not saw_independent:
                    return False, "a cyclic block argument has no independently proved seed", False
                value_verdict[value] = (False, None)
                return False, None, False
            if owner is None:
                return False, None, False
            name = _real_op_name(owner)
            category = _category_of(owner)
            if name in {"llvm.call", "llvm.call_intrinsic"}:
                task = task_of(owner)
                if task is None or task not in routes:
                    return False, "a called value has no program-plan task attribution", False
                hit, cause = scan_call(owner, task)
                return hit, cause, False
            if category == "load":
                root = _pointer_root(owner.operands[0], entry, op_index) if owner.operands else "UNKNOWN"
                if root.startswith("arg:"):
                    position = int(root.partition(":")[2])
                    out = pointer_args.get(position, (position in tensor_args, None))
                elif root.startswith("global:"):
                    out = (True, None)
                elif root.startswith("alloca:"):
                    out = (False, None)
                    saw_cycle = False
                    for stored in stored_into.get(root, ()):
                        hit, cause, cyclic = trace_value(stored, seen | {value})
                        if hit:
                            out = (True, None)
                            break
                        if cause is not None:
                            out = (False, cause)
                        saw_cycle |= cyclic
                    if saw_cycle and out == (False, None):
                        return False, "a stack value has only cyclic unproved provenance", False
                else:
                    out = (False, f"a load's pointer root is not derivable ({name})")
                value_verdict[value] = out
                return out[0], out[1], False
            if category in {"opaque_inline_asm", "other"}:
                return False, f"an operand chain reaches {name!r}, whose result this walk cannot classify", False
            if category == "constant":
                return False, None, False
            worst: str | None = None
            cyclic = False
            for operand in owner.operands:
                hit, cause, edge_cycle = trace_value(operand, seen | {value})
                if hit:
                    value_verdict[value] = (True, None)
                    return True, None, False
                if cause is not None:
                    worst = cause
                cyclic |= edge_cycle
            if not cyclic:
                value_verdict[value] = (False, worst)
            return False, worst, cyclic

        returned: list[tuple[bool, str | None]] = []
        for op in operations:
            category = _category_of(op)
            name = _real_op_name(op)
            if category in _COMPUTE_CATEGORIES:
                row = row_for(op)
                if row is None:
                    continue
                hit, cause = reaches_tensor(op.results[0], frozenset()) if op.results else (False, None)
                if hit:
                    row.on_tensor += 1
                    if len(row.witnesses) < 8:
                        row.witnesses.append(op_index[op])
                elif cause is not None:
                    row.undecided += 1
                    row.undecided_causes.append(cause)
                else:
                    row.on_addressing += 1
            elif name in {"llvm.call", "llvm.call_intrinsic"}:
                row = row_for(op)
                if row is not None:
                    scan_call(op, row.task)
            elif name == "llvm.return" and op.operands:
                returned.append(reaches_tensor(op.operands[0], frozenset()))
        if any(hit for hit, _ in returned):
            return True, None
        return False, next((cause for _, cause in returned if cause), None)

    scan_function(function, inherited_task=None, scalar_args={}, pointer_args={}, call_stack=frozenset({function}))

    if unattributed:
        causes.append(
            f"{unattributed} host arithmetic operation(s) carry no program-plan task attribution, so "
            f"their route is unknown and they were neither credited nor reported"
        )
    ordered = tuple(sorted(rows.values(), key=lambda r: (r.route, int(r.task))))
    findings = [r for r in ordered if r.route == "A" and r.on_tensor]
    undecided_in_accepted = any(r.route == "A" and r.undecided for r in ordered)
    if findings:
        status = STATUS_REPORTED
    elif undecided_in_accepted or causes:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_OK
    return HostComputeReport(GATE_HOST_COMPUTE, status, ordered, tuple(causes))


def format_host_compute_report(report: HostComputeReport, label: str = "") -> str:
    out = [f"== route host compute {label}".rstrip(), f"status: {report.status}"]
    out.append(f"accepted tasks computing on tensor data: {len(report.findings)}")
    out.append(f"host-route coverage (arithmetic on tensor data in H): {report.coverage}")
    for row in report.findings[:8]:
        out.append(
            f"  task {row.task} (A): {row.on_tensor} arithmetic op(s) on tensor data, "
            f"{row.on_addressing} on addressing; witnesses {row.witnesses[:4]}"
        )
    for cause in report.causes:
        out.append(f"  incomplete: {cause}")
    return "\n".join(out)
