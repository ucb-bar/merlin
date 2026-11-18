"""Exact finite transfer analysis for lexical SCF completion-token ownership.

Static stage footprints form the abstract state. Repeated states close counted
loops without enumerating trips or guessing device latency. These checks cover
declared RAW/WAR/WAW lifetimes only; target completion callbacks still need the
selected target's independent source/effect contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from xdsl.dialects import arith, builtin, func, scf

from merlin.compiler.kernel_ir.ir import Call, Kernel, Loop, Stage, Var, check_integer_domain, check_structure
from merlin.compiler.kernel_ir.loops import _allocator, _clone
from merlin.compiler.kernel_ir.primitives import (
    Cursor,
    NotApplicable,
    _loop_at,
    _map_calls,
    _rebuild,
    _resolve,
    _subst_call,
)
from merlin.compiler.schedule.loops.configuration import ConfigurationEffects

# ---- scf_carried_events ------------------------------------------------------------------------

CARRIED_EVENTS = "merlin.scf_commands.carried_events.v1"


def loop_event_names(op):
    marker = op.attributes.get(CARRIED_EVENTS)
    if marker is None:
        if op.iter_args or op.results or len(op.body.block.args) != 1:
            raise ValueError("carried completion requires its explicit lexical loop bindings")
        return (), ()
    if type(marker) is not builtin.StringAttr:
        raise ValueError("malformed lexical completion loop bindings")
    data = json.loads(marker.data)
    if type(data) is not dict or set(data) != {"inputs", "results"}:
        raise ValueError("unknown lexical completion loop binding fields")
    inputs, results = data["inputs"], data["results"]
    if (
        type(inputs) is not list
        or type(results) is not list
        or not inputs
        or len(inputs) != len(results)
        or len(inputs) != len(op.iter_args)
        or len(op.results) != len(inputs)
        or len(op.body.block.args) != len(inputs) + 1
        or any(type(n) is not str or not n for n in (*inputs, *results))
        or len(set(inputs)) != len(inputs)
        or len(set(results)) != len(results)
        or any(v.type.name != "runtime.event" for v in (*op.iter_args, *op.results, *op.body.block.args[1:]))
    ):
        raise ValueError("completion loop input/result types or arities differ")
    return tuple(inputs), tuple(results)


def verify_carried_events(module, *, max_operations, allow_persistent_state):
    from merlin.compiler.schedule.loops import scf_commands as C

    if type(module.attributes.get(CARRIED_EVENTS)) is not builtin.UnitAttr:
        raise ValueError("loop-carried completion requires the explicit versioned program marker")
    entry = C._entry(module)
    declarations = {op.sym_name.data: op for op in module.body.block.ops if not op.body.blocks}
    work = 0
    summaries = 0

    def body(block, incoming=(), input_names=(), *, yielding=False):
        nonlocal work, summaries
        values = dict(zip(block.args[1:], incoming, strict=True)) if yielding else {}
        names = dict(zip(block.args[1:], input_names, strict=True)) if yielding else {}
        pending = {v: state for v, state in values.items() if state is not None}
        # None is completed; an empty tuple is still an outstanding event with
        # no declared private rows. Completion can never be inferred from it.
        for op in block.ops:
            work += 1
            if work > max_operations:
                raise ValueError("completion-state transfer proof exceeds its explicit operation budget")
            if C.PERSISTENT_STATE in op.attributes and type(op) is not func.CallOp:
                raise ValueError("persistent state declarations belong to actual command calls only")
            if type(op) is scf.ForOp:
                local_names, result_names = loop_event_names(op)
                low, high, step = C._bounds(op)
                if C.ISSUE_GROUP in op.attributes:
                    raise ValueError("issue groups and carried completion need a separately composed target contract")
                initial = tuple(op.iter_args)
                if len(set(initial)) != len(initial) or any(v not in values for v in initial):
                    raise ValueError("loop completion inputs are aliased or do not dominate the loop")
                if set(pending) - set(initial):
                    raise ValueError("live completion crossing a loop must be an explicit iter_arg")
                states = tuple(pending.get(v) for v in initial)
                trips = (high - low + step - 1) // step
                seen, iteration = {}, 0
                if not trips:
                    # Even unreachable blocks must be well typed and closed;
                    # conservative footprint refusal is permitted, never a pass
                    # for an unchecked hidden command region.
                    body(op.body.block, states, local_names, yielding=True)
                while iteration < trips:
                    if states in seen:
                        period = iteration - seen[states]
                        skip = (trips - iteration) // period
                        if skip:
                            iteration += skip * period
                            continue
                    seen[states] = iteration
                    states = body(op.body.block, states, local_names, yielding=True)
                    iteration += 1
                summaries += 1
                for value in initial:
                    pending.pop(value, None)
                    values.pop(value)
                    names.pop(value)
                for value, state, name in zip(op.results, states, result_names, strict=True):
                    values[value], names[value] = state, name
                    if state is not None:
                        pending[value] = state
            elif type(op) is func.CallOp:
                metadata, operands = C._command(op, allow_persistent_state)
                declaration = declarations.get(op.callee.root_reference.data)
                if (
                    declaration is None
                    or tuple(declaration.function_type.inputs) != tuple(v.type for v in op.arguments)
                    or tuple(declaration.function_type.outputs) != tuple(v.type for v in op.results)
                ):
                    raise ValueError("command declaration and actual SSA call disagree")
                for name, event in zip(metadata.awaits, op.arguments[len(operands) :], strict=True):
                    if event not in values or names[event] != name:
                        raise ValueError("event SSA edge differs from its dominating lexical binding")
                    pending.pop(event, None)
                    values[event] = None
                if any(C._overlap(a, b) for stages in pending.values() for a in stages for b in metadata.stages):
                    raise ValueError("private rows reused before their actual event dependency")
                if metadata.produces:
                    if metadata.produces in names.values():
                        raise ValueError("completion producer shadows a live lexical binding")
                    value = op.results[0]
                    values[value] = pending[value] = metadata.stages
                    names[value] = metadata.produces
            elif type(op) is scf.YieldOp:
                if not yielding or op is not block.last_op:
                    raise ValueError("completion yields require their lexical loop terminator")
                yielded = tuple(op.arguments)
                if (
                    len(yielded) != len(incoming)
                    or len(set(yielded)) != len(yielded)
                    or any(v not in values for v in yielded)
                    or set(pending) - set(yielded)
                ):
                    raise ValueError("every live event must be yielded once or completed before loop exit")
                return tuple(pending.get(v) for v in yielded)
            elif type(op) is func.ReturnOp:
                if yielding or op is not block.last_op or op.arguments:
                    raise ValueError("completion cannot escape the closed command entry")
            elif type(op) not in (arith.ConstantOp, arith.AddiOp, arith.MuliOp, arith.CmpiOp, arith.SelectOp):
                raise ValueError("unsupported operation in command control program")
            if type(op) in (arith.ConstantOp, arith.AddiOp, arith.MuliOp, arith.CmpiOp, arith.SelectOp):
                if (
                    any(v.type not in (builtin.i64, builtin.i1) for v in (*op.operands, *op.results))
                    or type(op) is arith.CmpiOp
                    and op.predicate.value.data != 0
                ):
                    raise ValueError("unsupported command index type or predicate")
        if pending:
            raise ValueError("every launched event must complete before entry exit")
        if yielding:
            raise ValueError("completion loop requires its explicit yield terminator")
        return ()

    body(entry.body.block)
    return {
        "state_transfer_operations": work,
        "counted_loop_summaries": summaries,
        "method": "exact declared-stage transfer cycles; no dynamic trip expansion or device latency",
    }


def rename_ssa_events(module):
    """Alpha-rename metadata from actual SSA edges after a supported SCF clone.

    This changes no SSA operand, instruction, staging or completion effect.
    An edge not present in the lexical environment refuses; metadata never
    fabricates an edge. Callers must run the complete verifier afterwards.
    """
    from merlin.compiler.schedule.loops import scf_commands as C

    counter = 0

    def fresh():
        nonlocal counter
        result = f"completion_{counter}"
        counter += 1
        return result

    def body(block, names):
        names = dict(names)
        for op in block.ops:
            if type(op) is scf.ForOp:
                local = tuple(fresh() for _ in op.iter_args)
                body(op.body.block, dict(zip(op.body.block.args[1:], local, strict=True)))
                results = tuple(fresh() for _ in op.results)
                if local:
                    op.attributes[CARRIED_EVENTS] = builtin.StringAttr(
                        json.dumps({"inputs": local, "results": results}, sort_keys=True)
                    )
                for value in op.iter_args:
                    names.pop(value, None)
                names.update(zip(op.results, results, strict=True))
            elif type(op) is func.CallOp:
                data = json.loads(op.attributes[C.SCHEMA].data)
                try:
                    data["awaits"] = [names[v] for v in op.arguments[len(data["operands"]) :]]
                except KeyError as error:
                    raise ValueError("cloned completion edge escaped its lexical scope") from error
                if op.results:
                    data["produces"] = fresh()
                    names[op.results[0]] = data["produces"]
                op.attributes[C.SCHEMA] = builtin.StringAttr(json.dumps(data, sort_keys=True, separators=(",", ":")))

    body(C._entry(module).body.block, {})


# ---- completion_streams ------------------------------------------------------------------------
# Fuse independent counted completion recurrences without inventing events.
#
# The selected projector must cover ALL memory, persistent state and externally
# observable effects over each actual call's counted domain. Footprints are
# conservative envelopes in canonical backing coordinates, not a declaration that
# opcodes with different names commute. This module checks disjointness, preserves
# each stream's ordered calls/drains and rebuilds actual event inputs/yields.
# The projector's source/target qualification, numerical policy and finite-queue
# completion implementation remain independent obligations, never supplied here.


@dataclass(frozen=True)
class CompletionEffectEnvelope:
    """Complete conservative effects across a supplied counted index domain.

    Stage.memory names a selected canonical backing/coordinate system; bank
    labels never establish nonaliasing. The projector includes external tensor
    reads/writes as well as private storage, using one unit per backing. No
    physical layout, hidden effect or absence of observers is inferred from IR.
    Pure completion may be declared observer-free only under its independently
    selected full completion contract; every such original call is retained.
    """

    memory: tuple[Stage, ...]
    configuration: ConfigurationEffects
    complete: bool = False
    observers: str = "unknown"

    def __post_init__(self):
        if (
            type(self.memory) is not tuple
            or any(
                type(s) is not Stage
                or type(s.row) is not int
                or type(s.rows) is not int
                or type(s.depth) is not int
                or s.row < 0
                or s.rows < 1
                or s.depth < 1
                or type(s.writes) is not bool
                or type(s.memory) is not str
                or not s.memory
                for s in self.memory
            )
            or type(self.configuration) is not ConfigurationEffects
            or type(self.complete) is not bool
            or self.observers not in ("none", "unknown")
        ):
            raise ValueError("completion commutation requires typed complete backing/state envelopes")


@dataclass(frozen=True)
class CompletionEffectsAdapter:
    """Independently selected actual-call/domain projector; names alone grant nothing.

    project(call, ((index, extent), ...)) returns a CompletionEffectEnvelope or
    None. Envelopes must cover every actual instance, including tails. The
    identity binds its external qualification; a matching string is not proof.
    """

    project: object
    qualification_identity: str
    numerical_policy_identity: str
    storage_identity: str

    def __post_init__(self):
        if not callable(self.project) or any(
            type(v) is not str or not v
            for v in (self.qualification_identity, self.numerical_policy_identity, self.storage_identity)
        ):
            raise ValueError("completion effects need an explicit selected projector and source identities")


def _passive(loop):
    """Follow actual identity passthrough through nested counted bindings.

    New producers have unknown lineage. A nonidentity permutation in a repeated
    child conservatively makes all its incoming channels active; this helper
    never guesses a fixed point or equates two independently produced events.
    """
    used = set()

    def body(statements, incoming):
        aliases = dict(incoming)
        for op in statements:
            if type(op) is Loop:
                initials = tuple(aliases.get(n) for _, n in op.iter_args)
                child = body(op.body, dict(zip((n for n, _ in op.iter_args), initials, strict=True)))
                outputs = tuple(child.get(n) for n in op.yields)
                if op.extent > 1 and any(
                    v is not None and v != initial for v, initial in zip(outputs, initials, strict=True)
                ):
                    used.update(v for v in initials if v is not None)
                    outputs = (None,) * len(outputs)
                for _, n in op.iter_args:
                    aliases.pop(n, None)
                aliases.update(zip(op.results, outputs, strict=True))
            else:
                used.update(aliases[n] for n in op.awaits if aliases.get(n) is not None)
                if op.produces:
                    aliases[op.produces] = None
        return aliases

    aliases = body(loop.body, {n: i for i, (n, _) in enumerate(loop.iter_args)})
    return tuple(aliases.get(n) == i and i not in used for i, n in enumerate(loop.yields))


def _overlap(a, b):
    return (
        a.memory == b.memory
        and (a.writes or b.writes)
        and a.row < b.row + b.rows * b.depth
        and b.row < a.row + a.rows * a.depth
    )


def _commute(a, b):
    if any(_overlap(x, y) for x in a.memory for y in b.memory):
        raise NotApplicable("completion stream memory envelopes have a possible RAW/WAR/WAW dependence")

    def state(effect):
        c = effect.configuration
        return ({key for key, _ in c.reads}, {key for key, _ in c.writes} | set(c.clobbers))

    ar, aw = state(a)
    br, bw = state(b)
    if aw & (br | bw) or bw & (ar | aw):
        raise NotApplicable("completion streams share a written or clobbered persistent state field")


def _effects(body, domains, adapter, limit):
    effects = []

    def visit(body, domains):
        for op in body:
            if type(op) is Loop:
                visit(op.body, (*domains, (op.var, op.extent)))
                continue
            if type(op) is not Call or len(effects) >= limit:
                raise NotApplicable(
                    "completion fusion exceeds its static call proof budget or has an unknown statement"
                )
            effect = adapter.project(op, domains)
            if (
                type(effect) is not CompletionEffectEnvelope
                or not effect.complete
                or effect.observers != "none"
                or not effect.configuration.ordered
            ):
                raise NotApplicable("completion stream has unknown, incomplete or externally observed effects")
            effects.append(effect)

    visit(body, domains)
    return tuple(effects)


@dataclass(frozen=True)
class CompletionStreamFusion:
    kernel: Kernel
    report: dict

    def revalidate(self, original, adapter):
        actual = fuse_completion_streams(
            original,
            tuple(self.report["path"]),
            adapter,
            max_static_calls=self.report["max_static_calls"],
            max_commutation_pairs=self.report["max_commutation_pairs"],
            allow_persistent_state=self.report["allow_persistent_state"],
        )
        if actual.kernel != self.kernel or actual.report != self.report:
            raise ValueError("completion fusion source, actual fields or selected effect proof changed")
        return actual.report


def fuse_completion_streams(
    kernel, path, adapter, *, max_static_calls=4096, max_commutation_pairs=65536, allow_persistent_state=False
):
    """Zip two adjacent independent recurrences and verify actual SCF lifetimes.

    The second loop must consume a bijection of all first-loop results. Each
    channel is active in exactly one loop and unused identity passthrough in
    the other. Thus fusion preserves each channel's recurrence and each source
    stream's ordered command subsequence. Cross-stream motion additionally
    requires complete independently qualified effect envelopes to commute.
    No drain is deleted, hardware ticket fabricated, or fixed delay assumed.
    """
    from merlin.compiler.schedule.loops.scf_commands import materialize_scf_commands

    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(adapter) is not CompletionEffectsAdapter
        or type(path) is not tuple
        or not path
        or any(type(n) is not str for n in path)
        or type(max_static_calls) is not int
        or not 1 <= max_static_calls <= 4096
        or type(max_commutation_pairs) is not int
        or not 1 <= max_commutation_pairs <= 65536
        or type(allow_persistent_state) is not bool
    ):
        raise ValueError("completion fusion requires checked source, a selected effects adapter and bounded proof work")
    check_integer_domain(kernel, 64)
    original_digest = kernel.digest()
    materialize_scf_commands(kernel, allow_persistent_state=allow_persistent_state)
    first = _loop_at(kernel.body, path)
    siblings, _ = _resolve(kernel.body, Cursor(loops=path[:-1]))
    position = next(i for i, op in enumerate(siblings) if op is first)
    if position + 1 >= len(siblings) or type(siblings[position + 1]) is not Loop:
        raise NotApplicable("completion fusion requires an immediately adjacent following loop")
    second = siblings[position + 1]
    if (
        first.extent != second.extent
        or not first.iter_args
        or len(first.iter_args) != len(second.iter_args)
        or len(set(v for _, v in second.iter_args)) != len(first.results)
        or {v for _, v in second.iter_args} != set(first.results)
    ):
        raise NotApplicable("completion streams need equal counted extents and an exact result-to-input bijection")
    routing = tuple(first.results.index(v) for _, v in second.iter_args)
    ap, bp = _passive(first), _passive(second)
    if not any(ap) or all(ap) or any(ap[a] == bp[b] for b, a in enumerate(routing)):
        raise NotApplicable("each completion channel must be active in exactly one stream and passive in the other")
    domains = tuple((name, _loop_at(kernel.body, path[: i + 1]).extent) for i, name in enumerate(path[:-1]))
    left = _effects(first.body, (*domains, (first.var, first.extent)), adapter, max_static_calls)
    right = _effects(second.body, (*domains, (second.var, second.extent)), adapter, max_static_calls)
    if len(left) * len(right) > max_commutation_pairs:
        raise NotApplicable("completion commutation exceeds its static pair proof budget")
    for a in left:
        for b in right:
            _commute(a, b)
    if kernel.digest() != original_digest:
        raise ValueError("completion effect projection mutated its original source")
    fresh = _allocator(kernel)
    bindings = {local: first.yields[a] for (local, _), a in zip(second.iter_args, routing, strict=True)}
    body, names = _clone(second.body, bindings, fresh)
    body = _map_calls(body, lambda call: _subst_call(call, second.var, Var(first.var)))
    yielded, results = list(first.yields), list(first.results)
    for b, a in enumerate(routing):
        yielded[a] = names.get(second.yields[b], second.yields[b])
        results[a] = second.results[b]
    fused = replace(first, body=(*first.body, *body), yields=tuple(yielded), results=tuple(results))
    actual = replace(
        kernel, body=_rebuild(kernel.body, path[:-1], (*siblings[:position], fused, *siblings[position + 2 :]))
    )
    check_integer_domain(actual, 64)
    checked = materialize_scf_commands(actual, allow_persistent_state=allow_persistent_state)
    proof = hashlib.sha256(json.dumps([asdict(x) for x in (*left, *right)], sort_keys=True).encode()).hexdigest()
    report = dict(
        schema="merlin.completion_stream_fusion.v1",
        path=path,
        source_kernel_sha256=kernel.digest(),
        actual_kernel_sha256=actual.digest(),
        max_static_calls=max_static_calls,
        max_commutation_pairs=max_commutation_pairs,
        allow_persistent_state=allow_persistent_state,
        static_commutation_pairs=len(left) * len(right),
        counted_extent=first.extent,
        effect_envelopes_sha256=proof,
        qualification_identity=adapter.qualification_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
        storage_identity=adapter.storage_identity,
        completion="actual independent recurrences, lexical SCF edges and all original drain calls retained",
        source_mapping="(stream,iteration,call-site) -> (fused-iteration,original stream subsequence)",
        actual_scf_sha256=hashlib.sha256(str(checked).encode()).hexdigest(),
        target_execution="NOT_ESTABLISHED",
    )
    return CompletionStreamFusion(actual, report)
