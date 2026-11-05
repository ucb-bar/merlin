"""Fuse independent counted completion recurrences without inventing events.

The selected projector must cover ALL memory, persistent state and externally
observable effects over each actual call's counted domain. Footprints are
conservative envelopes in canonical backing coordinates, not a declaration that
opcodes with different names commute. This module checks disjointness, preserves
each stream's ordered calls/drains and rebuilds actual event inputs/yields.
The projector's source/target qualification, numerical policy and finite-queue
completion implementation remain independent obligations, never supplied here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from merlin.sched.ir import Call, Kernel, Loop, Stage, Var, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import Cursor, NotApplicable, _loop_at, _map_calls, _rebuild, _resolve, _subst_call
from merlin.sched.primitives.event_loops import _allocator, _clone

from .configuration import ConfigurationEffects
from .scf_commands import materialize_scf_commands


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
