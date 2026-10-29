"""Prospective v1 bridge from counted schedules to ordinary xDSL SCF/SSA.

SCF transforms operate on the actual command-bearing IR consumed by emission.
Existing runtime events carry dependencies; existing Stage annotations retain
private-row lifetimes. This conditional compiler mechanism grants no target
completion semantics, ownership, instruction support or execution qualification.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict

from xdsl.dialects import arith, builtin, func, scf
from xdsl.ir import Block, Region

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Select, Stage, Sum, Var, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain

from .scf_carried_events import CARRIED_EVENTS

SCHEMA = "merlin.scf_commands.v1"
ISSUE_GROUP = "merlin.scf_commands.issue_group.v1"
PERSISTENT_STATE = "merlin.scf_commands.persistent_state.v1"


def _event_type():
    # Synchronous candidates need no new dialect import from a frozen SDK.
    from merlin.xdsl_dialects.runtime import EventType

    return EventType()


def _event(value):
    return value.type.name == "runtime.event"


def _text(module):
    from xdsl.printer import Printer

    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _digest(module):
    return hashlib.sha256(_text(module).encode()).hexdigest()


def _constant(value):
    if type(value) is not int or not -(1 << 63) <= value < 1 << 63:
        raise ValueError("command indices require bounded signed i64 constants")
    return arith.ConstantOp.from_int_and_width(value, 64)


def _metadata(call, allow_persistent_state=False):
    if (call.sets or call.assumes) and not allow_persistent_state:
        raise ValueError("SCF command bridge does not yet lower persistent configuration annotations")
    return builtin.StringAttr(
        json.dumps(
            {
                "instruction": call.instr,
                "operands": [n for n, _ in call.args],
                "unit": call.unit,
                "produces": call.produces,
                "awaits": call.awaits,
                "stages": [asdict(s) for s in call.stages],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _state_pairs(pairs):
    """Freeze only closed, unambiguous signed-i64 scalar declarations."""
    if type(pairs) is not tuple:
        raise ValueError("persistent state requires unique named scalar constants")
    encoded = []
    keys = set()
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("persistent state requires unique named scalar constants")
        key, value = pair
        if type(key) is not str or not key or key in keys:
            raise ValueError("persistent state requires unique named scalar constants")
        keys.add(key)
        if (type(value) is not Const or type(value.value) is not int
                or not -(1 << 63) <= value.value < 1 << 63):
            raise ValueError("persistent state admits closed signed-i64 constants only")
        encoded.append([key, {"kind": "const", "value": value.value}])
    return encoded


def _state_annotations(op, allow_persistent_state):
    raw = op.attributes.get(PERSISTENT_STATE)
    if raw is None:
        return (), ()
    if not allow_persistent_state:
        raise ValueError("persistent configuration annotations require explicit opt-in")
    if type(raw) is not builtin.StringAttr:
        raise ValueError("persistent state requires its explicit typed payload")

    def unique_object(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("duplicate persistent state metadata field")
            data[key] = value
        return data

    data = json.loads(raw.data, object_pairs_hook=unique_object)
    if type(data) is not dict or set(data) != {"sets", "assumes"}:
        raise ValueError("unknown persistent state metadata")
    result = []
    for name in ("sets", "assumes"):
        if type(data[name]) is not list:
            raise ValueError("malformed persistent state declarations")
        pairs = []
        for pair in data[name]:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("malformed persistent state declarations")
            key, record = pair
            if (type(record) is not dict or set(record) != {"kind", "value"}
                    or record["kind"] != "const"
                    or type(record["value"]) is not int):
                raise ValueError("malformed persistent state scalar type")
            pairs.append((key, Const(record["value"])))
        pairs = tuple(pairs)
        if _state_pairs(pairs) != data[name]:
            raise ValueError("persistent state scalar declarations changed")
        result.append(pairs)
    return tuple(result)


def _command(op, allow_persistent_state=False):
    raw = op.attributes.get(SCHEMA)
    if type(raw) is not builtin.StringAttr:
        raise ValueError("every command call requires its explicit v1 payload")
    data = json.loads(raw.data)
    if set(data) != {"instruction", "operands", "unit", "produces", "awaits", "stages"}:
        raise ValueError("unknown SCF command metadata")
    if (
        type(data["instruction"]) is not str
        or not data["instruction"]
        or type(data["operands"]) is not list
        or len(set(data["operands"])) != len(data["operands"])
        or any(type(n) is not str or not n for n in data["operands"])
        or data["unit"] is not None
        and (type(data["unit"]) is not str or not data["unit"])
        or data["produces"] is not None
        and (type(data["produces"]) is not str or not data["produces"])
        or type(data["awaits"]) is not list
        or any(type(n) is not str or not n for n in data["awaits"])
        or len(set(data["awaits"])) != len(data["awaits"])
    ):
        raise ValueError("malformed command identity or event annotations")
    stages = tuple(Stage(**s) for s in data["stages"])
    for stage in stages:
        if (
            type(stage.memory) is not str
            or not stage.memory
            or any(type(x) is not int or x < 0 for x in (stage.row, stage.rows, stage.depth))
            or not stage.rows
            or not stage.depth
            or type(stage.writes) is not bool
            or stage.bank is not None
            and (type(stage.bank) is not int or stage.bank < 0)
        ):
            raise ValueError("invalid retained private-row lifetime")
    names = data["operands"]
    operands = tuple(op.arguments)
    if (
        len(operands) != len(names) + len(data["awaits"])
        or any(v.type != builtin.i64 for v in operands[: len(names)])
        or any(not _event(v) for v in operands[len(names) :])
        or len(op.results) != int(bool(data["produces"]))
        or any(not _event(v) for v in op.results)
    ):
        raise ValueError("command SSA indices/events differ from the retained payload")
    sets, assumes = _state_annotations(op, allow_persistent_state)
    return Call(data["instruction"], (), data["unit"], data["produces"], tuple(data["awaits"]), stages,
                sets, assumes), names


def materialize_scf_commands(kernel, *, allow_persistent_state=False):
    """Keep counted loops and real command operands in standard SCF/arith/func.

    Integer-index calls only. Loop-carried events require explicit inputs,
    yields and results; other events close within each lexical scope. Tensor
    arguments stay with the target ABI; this module computes no tensor payload.
    Persistent declarations require explicit opt-in and closed i64 constants.
    They remain declarations: actual state visibility/clobbers/readout effects
    must still be proved by the selected target adapter.
    """
    if type(allow_persistent_state) is not bool:
        raise ValueError("persistent state opt-in must be an explicit Boolean")
    if type(kernel) is not Kernel or check_structure(kernel) or kernel.attrs:
        raise ValueError("SCF command bridge requires a checked closed counted schedule")
    from merlin.sched.ir.events import has_carried_events

    carried_mode = has_carried_events(kernel.body)
    check_integer_domain(kernel, 64)
    declarations = []
    block = Block()

    def expr(value, env, out):
        if type(value) is Const:
            op = _constant(value.value)
        elif type(value) is Var:
            return env[value.name]
        elif type(value) is Scaled:
            op = arith.MuliOp(expr(value.expr, env, out), expr(Const(value.factor), env, out))
        elif type(value) is Sum:
            result = expr(Const(0), env, out)
            for term in value.terms:
                op = arith.AddiOp(result, expr(term, env, out))
                out.add_op(op)
                result = op.result
            return result
        elif type(value) is Select:
            compare = arith.CmpiOp(env[value.var], expr(Const(value.equals), env, out), "eq")
            out.add_op(compare)
            op = arith.SelectOp(compare.result, expr(value.then, env, out), expr(value.other, env, out))
        else:
            raise ValueError("SCF command bridge admits integer indices only, never payload arithmetic")
        out.add_op(op)
        return op.results[0]

    def body(statements, env, out, events):
        for statement in statements:
            if type(statement) is Loop:
                lb, ub, step = (_constant(v) for v in (0, statement.extent, 1))
                out.add_ops((lb, ub, step))
                initial = [events[name] for _, name in statement.iter_args]
                inner = Block(arg_types=[builtin.i64, *(_event_type() for _ in initial)])
                # Preserve legacy completed-token capture exactly when no new
                # lexical carry contract was requested. Opted-in programs use
                # only explicitly declared block inputs, including captures.
                local_events = (dict(zip((n for n, _ in statement.iter_args), inner.args[1:], strict=True))
                                if carried_mode else dict(events))
                body(statement.body, {**env, statement.var: inner.args[0]}, inner, local_events)
                inner.add_op(scf.YieldOp(*(local_events[name] for name in statement.yields)))
                op = scf.ForOp(lb.result, ub.result, step.result, initial, Region(inner))
                if initial:
                    op.attributes[CARRIED_EVENTS] = builtin.StringAttr(json.dumps(
                        {"inputs": [n for n, _ in statement.iter_args], "results": statement.results}, sort_keys=True))
                out.add_op(op)
                for _, name in statement.iter_args:
                    events.pop(name)
                events.update(zip(statement.results, op.results, strict=True))
            elif type(statement) is Call:
                operands = [expr(v, env, out) for _, v in statement.args]
                try:
                    operands.extend(events[n] for n in statement.awaits)
                except KeyError as error:
                    raise ValueError("event use must follow its dominating original launch") from error
                types = [_event_type()] if statement.produces else []
                symbol = f"command_{len(declarations)}"
                declarations.append(func.FuncOp.external(symbol, [v.type for v in operands], types))
                op = func.CallOp(symbol, operands, types)
                op.attributes[SCHEMA] = _metadata(statement, allow_persistent_state)
                if statement.sets or statement.assumes:
                    op.attributes[PERSISTENT_STATE] = builtin.StringAttr(json.dumps(
                        {"sets": _state_pairs(statement.sets), "assumes": _state_pairs(statement.assumes)},
                        sort_keys=True, separators=(",", ":"),
                    ))
                out.add_op(op)
                if statement.produces:
                    if statement.produces in events:
                        raise ValueError("event names must be unique within a counted iteration")
                    events[statement.produces] = op.results[0]
            else:
                raise ValueError("unknown counted schedule statement")

    body(kernel.body, {}, block, {})
    block.add_op(func.ReturnOp())
    entry = func.FuncOp("command_program", ([], []), Region(block))
    module = builtin.ModuleOp([*declarations, entry], attributes={SCHEMA: builtin.StringAttr(kernel.digest())})
    if carried_mode:
        module.attributes[CARRIED_EVENTS] = builtin.UnitAttr()
    verify_scf_commands(module, allow_persistent_state=allow_persistent_state)
    return module


def _entry(module):
    if type(module) is not builtin.ModuleOp or type(module.attributes.get(SCHEMA)) is not builtin.StringAttr:
        raise ValueError("SCF command program requires its explicit source binding")
    if any(type(op) is not func.FuncOp for op in module.body.block.ops):
        raise ValueError("unknown operation outside the command entry")
    entries = [op for op in module.body.block.ops if type(op) is func.FuncOp and op.body.blocks]
    if len(entries) != 1 or entries[0].sym_name.data != "command_program" or len(entries[0].body.blocks) != 1:
        raise ValueError("one closed SCF command entry required")
    if entries[0].body.block.args or entries[0].function_type.outputs:
        raise ValueError("command entry cannot introduce a tensor ABI or escape events")
    return entries[0]


def _bounds(op):
    values = []
    from .scf_carried_events import loop_event_names
    loop_event_names(op)
    for value in (op.lb, op.ub, op.step):
        owner = value.owner
        if type(owner) is not arith.ConstantOp or type(owner.value) is not builtin.IntegerAttr:
            raise ValueError("command loops need explicit static bounds")
        values.append(owner.value.value.data)
    low, high, step = values
    if not 0 <= low <= high < 1 << 63 or not 0 < step < 1 << 63 or high + step >= 1 << 63:
        raise ValueError("command loop bounds exceed ascending signed i64 control")
    return low, high, step


def _overlap(left, right):
    return (
        left.memory == right.memory
        and (left.writes or right.writes)
        and left.row < right.row + right.rows * right.depth
        and right.row < left.row + left.rows * left.depth
    )


def _declared_state_summary(block):
    """Compose constant transfers once per static block, without unfolding loops.

    A transfer requires values read before its first write and leaves its last
    constant writes live. Repetition is valid exactly when those final writes
    satisfy the next iteration's incoming requirements. All subsequent trips
    then have the same transfer. Zero trips contribute neither reads nor writes.
    """
    required, written = {}, {}

    def advance(reads, writes):
        for key, wanted in reads.items():
            if key in written:
                if written[key] != wanted:
                    raise ValueError(f"declared persistent state {key!r} differs from its required value")
            elif key in required and required[key] != wanted:
                raise ValueError(f"conflicting incoming persistent state requirements for {key!r}")
            else:
                required[key] = wanted
        written.update(writes)

    for op in block.ops:
        if type(op) is func.CallOp:
            metadata, _ = _command(op, True)
            reads, writes = ({key: value.value for key, value in pairs}
                             for pairs in (metadata.assumes, metadata.sets))
            advance(reads, writes)
        elif type(op) is scf.ForOp:
            low, high, step = _bounds(op)
            trips = (high - low + step - 1) // step
            if not trips:
                continue
            reads, writes = _declared_state_summary(op.body.block)
            if trips >= 2:
                for key, wanted in reads.items():
                    if key in writes and writes[key] != wanted:
                        raise ValueError(f"persistent state loop reentry does not preserve {key!r}")
            advance(reads, writes)
    return required, written


def verify_scf_commands(module, *, max_operations=65536, allow_persistent_state=False):
    """Check closed events and declared staging hazards before callbacks run.

    Rows in the same named memory conservatively alias even across bank tags.
    This checks only declared staging: no physical ownership or target state is
    inferred, and no arbitrary SCF transform receives a semantic certificate.
    Opted-in persistent declarations are checked with initial state unknown,
    including nested counted-loop reentry, without expanding dynamic trips.
    """
    if type(allow_persistent_state) is not bool:
        raise ValueError("persistent state opt-in must be an explicit Boolean")
    module.verify()
    entry = _entry(module)
    if sum(1 for _ in module.walk()) > max_operations:
        raise ValueError("SCF command program exceeds its explicit operation budget")
    declarations = {
        op.sym_name.data: op for op in module.body.block.ops if type(op) is func.FuncOp and not op.body.blocks
    }
    known = set()

    if CARRIED_EVENTS in module.attributes:
        from .scf_carried_events import verify_carried_events
        carried_report = verify_carried_events(module, max_operations=max_operations,
                                               allow_persistent_state=allow_persistent_state)
    else:
        carried_report = None

    def body(block):
        pending = {}
        for op in block.ops:
            if PERSISTENT_STATE in op.attributes and type(op) is not func.CallOp:
                raise ValueError("persistent state declarations belong to actual command calls only")
            if type(op) is scf.ForOp:
                if pending:
                    raise ValueError("events carried into loops require an explicit iter_arg contract")
                low, high, step = _bounds(op)
                if ISSUE_GROUP in op.attributes:
                    marker = op.attributes[ISSUE_GROUP]
                    if (
                        type(marker) is not builtin.DictionaryAttr
                        or set(marker.data) != {"max_issues", "contract_sha256"}
                        or type(marker.data["max_issues"]) is not builtin.IntegerAttr
                        or type(marker.data["contract_sha256"]) is not builtin.StringAttr
                        or len(marker.data["contract_sha256"].data) != 64
                        or any(c not in "0123456789abcdef" for c in marker.data["contract_sha256"].data)
                        or marker.data["max_issues"].type != builtin.i64
                        or not 1 <= (high - low + step - 1) // step <= marker.data["max_issues"].value.data <= 64
                        or any(type(child) is scf.ForOp for child in op.body.walk())
                    ):
                        raise ValueError("malformed or non-innermost completion group")
                body(op.body.block)
            elif type(op) is func.CallOp:
                metadata, names = _command(op, allow_persistent_state)
                declaration = declarations.get(op.callee.root_reference.data)
                if (
                    declaration is None
                    or tuple(declaration.function_type.inputs) != tuple(v.type for v in op.arguments)
                    or tuple(declaration.function_type.outputs) != tuple(v.type for v in op.results)
                ):
                    raise ValueError("command declaration and actual SSA call disagree")
                for token_name, event in zip(metadata.awaits, op.arguments[len(names) :], strict=True):
                    if (type(event.owner) is not func.CallOp
                            or _command(event.owner, allow_persistent_state)[0].produces != token_name):
                        raise ValueError("event SSA edge differs from its original named producer")
                    if event not in known:
                        raise ValueError("unknown or non-dominating completion event")
                    pending.pop(event, None)
                if any(_overlap(a, b) for stages in pending.values() for a in stages for b in metadata.stages):
                    raise ValueError("private rows reused before their actual event dependency")
                if metadata.produces:
                    known.add(op.results[0])
                    pending[op.results[0]] = metadata.stages
            elif type(op) not in (
                arith.ConstantOp,
                arith.AddiOp,
                arith.MuliOp,
                arith.CmpiOp,
                arith.SelectOp,
                scf.YieldOp,
                func.ReturnOp,
            ):
                raise ValueError("unsupported operation in command control program")
            if type(op) in (arith.ConstantOp, arith.AddiOp, arith.MuliOp, arith.CmpiOp, arith.SelectOp):
                if (
                    any(v.type not in (builtin.i64, builtin.i1) for v in (*op.operands, *op.results))
                    or type(op) is arith.CmpiOp
                    and op.predicate.value.data != 0
                ):
                    raise ValueError("unsupported command index type or predicate")
        if pending:
            raise ValueError("every launched event must complete before loop/entry exit")

    if carried_report is None:
        body(entry.body.block)
    report = {
        "schema": SCHEMA,
        "source_schedule_sha256": module.attributes[SCHEMA].data,
        "command_ir_sha256": _digest(module),
        "target_execution": "NOT_ESTABLISHED",
    }
    if carried_report is not None:
        report["loop_carried_completion"] = carried_report
    if allow_persistent_state:
        required, written = _declared_state_summary(entry.body.block)
        if required:
            raise ValueError(f"persistent state has no earlier setter: {sorted(required)!r}")
        report["declared_persistent_state"] = {
            "final_keys": sorted(written),
            "initial_state": "unknown",
            "validation": "closed i64 declarations; static transfer composition and counted-loop reentry",
            "target_state_visibility_and_clobber_law": "NOT_ESTABLISHED",
        }
    return report


def unroll_small_command_loops(
    module, *, max_extent=2, max_operations=65536, preserve_issue_groups=False, allow_persistent_state=False,
):
    """Apply xDSL's existing SCF unroller to bounded innermost command loops.

    Work on a clone; failures preserve the caller's original module. No command
    is synthesized or reordered within an iteration, and SSA event edges clone
    with their producers/consumers. This is a code size policy, not overlap proof.
    Explicit preservation leaves completion regions, their ancestors and their
    contents untouched while transforming independent loops. The default
    continues refusing already bound target regions.
    """
    from xdsl.pattern_rewriter import PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
    from xdsl.transforms.scf_for_loop_unroll import UnrollLoopPattern

    verify_scf_commands(module, max_operations=max_operations, allow_persistent_state=allow_persistent_state)
    if type(preserve_issue_groups) is not bool:
        raise ValueError("issue-group preservation must be an explicit Boolean")
    if not preserve_issue_groups and any(ISSUE_GROUP in op.attributes for op in module.walk()):
        raise ValueError("unroll before binding completion groups; do not erase target region hooks")
    if type(max_extent) is not int or not 1 <= max_extent <= 64:
        raise ValueError("unroll extent must be an explicit bounded policy")
    selected = set()
    estimate = sum(1 for _ in module.walk())
    for op in module.walk():
        if type(op) is scf.ForOp:
            if preserve_issue_groups:
                parent, protected = op, any(ISSUE_GROUP in child.attributes for child in op.walk())
                while parent is not None:
                    protected |= ISSUE_GROUP in parent.attributes
                    parent = parent.parent_op()
                if protected:
                    continue
            low, high, step = _bounds(op)
            extent = (high - low + step - 1) // step
            if extent <= max_extent and not any(type(inner) is scf.ForOp for inner in op.body.walk()):
                selected.add(op)
                estimate += extent * sum(1 for _ in op.body.walk())
    if estimate > max_operations:
        raise ValueError("unrolling would exceed the explicit operation budget")
    actual = module.clone()
    # Match only the original innermost loop paths, never newly exposed parents.
    original_paths = {i for i, op in enumerate(module.walk()) if op in selected}
    selected_clones = {op for i, op in enumerate(actual.walk()) if i in original_paths}
    original = UnrollLoopPattern()

    class BoundedUnroll(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: scf.ForOp, rewriter):
            low, high, step = _bounds(op)
            if op in selected_clones:
                original.match_and_rewrite(op, rewriter)

    PatternRewriteWalker(BoundedUnroll()).rewrite_module(actual)
    if CARRIED_EVENTS in actual.attributes:
        from .scf_carried_events import rename_ssa_events
        rename_ssa_events(actual)
    verify_scf_commands(actual, max_operations=max_operations, allow_persistent_state=allow_persistent_state)
    return actual


def emit_scf_commands(
    module, builder, lower_call, *, await_events=None, enter_issue_group=None, exit_issue_group=None,
    allow_persistent_state=False,
):
    """Emit the actual transformed SCF program through integer/target callbacks.

    lower_call(instruction, index_operands, metadata) implements real target
    commands and returns a handle for an asynchronous launch. await_events gets
    the actual SSA-produced handles before its consumer issues. It must implement
    real target completion; a callback name or this report supplies no proof.
    Selected issue-group hooks surround an actual counted loop, not its body:
    exit_issue_group must realize completion before any subsequent operation.
    """
    before = verify_scf_commands(module, allow_persistent_state=allow_persistent_state)
    entry = _entry(module)
    if any(ISSUE_GROUP in op.attributes for op in module.walk()) and not all(
        callable(callback) for callback in (enter_issue_group, exit_issue_group)
    ):
        raise ValueError("issue groups require actual target entry/completion hooks")
    if not callable(lower_call) or any(
        not callable(getattr(builder, n, None)) for n in ("op", "cmp", "select", "loop")
    ):
        raise ValueError("SCF emission needs the ordinary index builder and target callback")
    if any(_event(v) for op in module.walk() for v in op.results) and not callable(await_events):
        raise ValueError("explicit events require an actual target completion callback")
    if any(type(op) is scf.ForOp and op.iter_args for op in module.walk()) and not callable(
        getattr(builder, "loop_carried", None)
    ):
        raise ValueError("loop-carried completion requires actual loop input/yield/result lowering")

    def emit(block, inherited):
        env = dict(inherited)
        for op in block.ops:
            if type(op) is arith.ConstantOp:
                env[op.result] = op.value.value.data
            elif type(op) in (arith.AddiOp, arith.MuliOp):
                env[op.result] = builder.op("add" if type(op) is arith.AddiOp else "mul", env[op.lhs], env[op.rhs])
            elif type(op) is arith.CmpiOp:
                env[op.result] = builder.cmp("eq", env[op.lhs], env[op.rhs])
            elif type(op) is arith.SelectOp:
                env[op.result] = builder.select(env[op.cond], env[op.lhs], env[op.rhs])
            elif type(op) is scf.ForOp:
                low, high, step = _bounds(op)
                grouped = ISSUE_GROUP in op.attributes
                if grouped:
                    enter_issue_group(op)
                if op.iter_args:
                    def iteration(index, carried):
                        if len(carried) != len(op.iter_args):
                            raise ValueError("actual builder changed completion input arity")
                        return emit(op.body.block, {**env, op.body.block.args[0]: index,
                                    **dict(zip(op.body.block.args[1:], carried, strict=True))})
                    result = builder.loop_carried(low, high, step, tuple(env[v] for v in op.iter_args), iteration)
                    if not isinstance(result, (tuple, list)) or len(result) != len(op.results):
                        raise ValueError("actual builder changed completion result arity")
                    env.update(zip(op.results, result, strict=True))
                else:
                    builder.loop(low, high, step,
                                 lambda index: emit(op.body.block, {**env, op.body.block.args[0]: index}))
                if grouped:
                    exit_issue_group(op)
            elif type(op) is func.CallOp:
                metadata, names = _command(op, allow_persistent_state)
                if metadata.awaits:
                    await_events(tuple(env[v] for v in op.arguments[len(names) :]))
                result = lower_call(
                    metadata.instr,
                    dict(zip(names, (env[v] for v in op.arguments[: len(names)]), strict=True)),
                    metadata,
                )
                if metadata.produces:
                    if result is None:
                        raise ValueError("asynchronous target launch did not return a completion handle")
                    env[op.results[0]] = result
            elif type(op) is scf.YieldOp:
                return tuple(env[v] for v in op.arguments)

    emit(entry.body.block, {})
    if verify_scf_commands(module, allow_persistent_state=allow_persistent_state) != before:
        raise ValueError("command-bearing IR changed during actual emission")
    return {**before, "scope": "actual transformed SCF/index emission; target qualification remains independent"}


def group_scf_command_loops(
    module, select_loop, *, max_issues, contract_sha256, max_operations=65536, allow_persistent_state=False,
):
    """Strip-mine selected innermost command loops into bounded issue regions.

    This changes only control partitioning, never command order/operands. The
    caller's separate contract must admit issuing every region before a real
    completion hook. Queue entry issue/release is never treated as completion.
    Existing per-command completion remains the default in the target callback.
    This is not admission of arithmetic, aliasing, ownership or fence removal.
    """
    from xdsl.pattern_rewriter import PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

    before = verify_scf_commands(module, max_operations=max_operations, allow_persistent_state=allow_persistent_state)
    if (
        not callable(select_loop)
        or type(max_issues) is not int
        or not 2 <= max_issues <= 64
        or type(contract_sha256) is not str
        or len(contract_sha256) != 64
        or any(c not in "0123456789abcdef" for c in contract_sha256)
    ):
        raise ValueError("issue groups need a selected contract and bounded issue count")
    selected = set()
    for index, op in enumerate(module.walk()):
        if type(op) is scf.ForOp and select_loop(op):
            if (
                ISSUE_GROUP in op.attributes
                or any(type(inner) is scf.ForOp for inner in op.body.walk())
                or any(_event(v) for inner in op.body.walk() for v in (*inner.operands, *inner.results))
            ):
                raise ValueError("issue grouping needs an innermost event-free command loop")
            low, high, step = _bounds(op)
            if (high - low + step - 1) // step > 1:
                selected.add(index)
    if verify_scf_commands(
        module, max_operations=max_operations, allow_persistent_state=allow_persistent_state
    ) != before:
        raise ValueError("loop selection changed the original command-bearing program")
    # A loop body is cloned at most twice (whole groups and a remainder).
    if 3 * sum(1 for _ in module.walk()) + 32 * len(selected) > max_operations:
        raise ValueError("issue grouping would exceed the explicit static operation budget")
    actual = module.clone()
    selected_clones = {op for i, op in enumerate(actual.walk()) if i in selected}
    marker = builtin.DictionaryAttr(
        {
            "max_issues": builtin.IntegerAttr(max_issues, builtin.i64),
            "contract_sha256": builtin.StringAttr(contract_sha256),
        }
    )

    class Group(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: scf.ForOp, rewriter):
            if op not in selected_clones:
                return
            low, high, step = _bounds(op)
            extent = (high - low + step - 1) // step
            if extent <= max_issues:
                op.attributes[ISSUE_GROUP] = marker
                return
            whole, tail = divmod(extent, max_issues)
            zero, count, one, chunk = (_constant(v) for v in (0, whole, 1, max_issues))
            outer = Block(arg_types=[builtin.i64])
            inner = Block(arg_types=[builtin.i64])
            multiply = arith.MuliOp(outer.args[0], chunk.result)
            add_index = arith.AddiOp(multiply.result, inner.args[0])
            stride, offset = _constant(step), _constant(low)
            scale = arith.MuliOp(add_index.result, stride.result)
            index = arith.AddiOp(scale.result, offset.result)
            inner.add_ops((multiply, add_index, stride, offset, scale, index))
            mapping = {op.body.block.args[0]: index.result}
            for original in op.body.block.ops:
                if type(original) is not scf.YieldOp:
                    inner.add_op(original.clone(mapping))
            inner.add_op(scf.YieldOp())
            grouped = scf.ForOp(zero.result, chunk.result, one.result, [], Region(inner))
            grouped.attributes[ISSUE_GROUP] = marker
            outer.add_ops((grouped, scf.YieldOp()))
            loop = scf.ForOp(zero.result, count.result, one.result, [], Region(outer))
            replacements = [zero, count, one, chunk, loop]
            if tail:
                start, end, increment = (_constant(v) for v in (low + whole * max_issues * step, high, step))
                remainder = Block(arg_types=[builtin.i64])
                mapping = {op.body.block.args[0]: remainder.args[0]}
                for original in op.body.block.ops:
                    if type(original) is not scf.YieldOp:
                        remainder.add_op(original.clone(mapping))
                remainder.add_op(scf.YieldOp())
                tail_loop = scf.ForOp(start.result, end.result, increment.result, [], Region(remainder))
                tail_loop.attributes[ISSUE_GROUP] = marker
                replacements.extend((start, end, increment, tail_loop))
            rewriter.replace(op, replacements)

    PatternRewriteWalker(Group(), apply_recursively=False).rewrite_module(actual)
    verify_scf_commands(actual, max_operations=max_operations, allow_persistent_state=allow_persistent_state)
    return actual
