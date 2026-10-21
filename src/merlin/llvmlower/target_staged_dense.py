"""Original-owned LLVM adapters for existing finite staged contraction passes.

The adapter must exhaustively account for configuration, global, opaque and
memory effects. Its explicit completeness premise is not qualification evidence.
This leaf checks actual projected operations, not arbitrary LLVM equivalence;
target execution, resource assignment and effect qualification remain original
owner obligations. No target discovery or new public SDK grant is supplied.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FunctionType

from xdsl.dialects.builtin import ModuleOp

from merlin.compile.scheduling import staged_dense as staged
from merlin.compile.scheduling import staged_dense_verification as verifier
from merlin.compile.scheduling.configuration_ir import _bounded_text
from merlin.compile.scheduling.schedule_pipeline import _rewrite_snapshot
from merlin.transforms.certificates import ir_digest

from . import target_staged_dense_check as completion
from .source_multioutput import _callback, _check_callback


@dataclass(frozen=True)
class StagedDenseTargetAdapter:
    """Original Python callbacks; unknown instructions must return None.

    project reads current actual operands and accounts for the whole module.
    emit must return a new module, preserving all effects outside the selected
    staged sequence. The independent reproject must reject altered unrepresented
    configuration/globals/other work. A completeness label supplies no such proof.
    """

    project: Callable
    emit: Callable
    complete_effects: bool = field(kw_only=True)

    def __post_init__(self):
        if type(self.complete_effects) is not bool:
            raise ValueError("staged target adapter needs an explicit complete-effects Boolean")
        for callback in (self.project, self.emit):
            _callback(callback)


def _functions():
    return (
        _functions,
        _module_state,
        _guard,
        _project,
        _bounds,
        rewrite_target_staged_dense,
        verify_target_staged_dense_emission,
        StagedDenseTargetAdapter.__post_init__,
        TargetStagedDenseResult.revalidate,
        _callback,
        _check_callback,
        _rewrite_snapshot,
        _bounded_text,
        ir_digest,
        *(
            value
            for value in vars(staged).values()
            if type(value) is FunctionType and value.__module__ == staged.__name__
        ),
        *(
            value
            for value in vars(completion).values()
            if type(value) is FunctionType and value.__module__ == completion.__name__
        ),
        verifier.verify_staged_dense,
        staged.StagedDenseContract.__post_init__,
        staged.StagedDenseProblem.__post_init__,
        staged.StagedDenseRewrite.__post_init__,
    )


def _bounds(max_commands, max_products, max_ir_bytes, max_operations):
    limits = (max_commands, max_products, max_ir_bytes, max_operations)
    if any(type(n) is not int or not 0 < n <= 1 << 30 for n in limits):
        raise ValueError("staged target verification requires positive bounded integer ceilings")
    return limits


def _module_state(module, bounds):
    if type(module) is not ModuleOp:
        raise ValueError("staged target adapter needs an actual builtin.ModuleOp")
    operations = []
    for op in module.walk():
        operations.append(op)
        if len(operations) > bounds[3]:
            raise ValueError("staged target module exceeds its operation ceiling")
    _bounded_text(module, bounds[2])
    module.verify()
    return (
        ir_digest(module),
        tuple(operations),
        tuple(
            (
                op.parent,
                tuple(op.operands),
                tuple(op.results),
                tuple(tuple((u.operation, u.index) for u in result.uses) for result in op.results),
            )
            for op in operations
        ),
    )


def _guard(modules, problem, contract, adapter, rewrites, bounds):
    """Retain readers, code and detached data before any original callback."""
    if type(problem) is not staged.StagedDenseProblem or type(contract) is not staged.StagedDenseContract:
        raise ValueError("staged target rewriting needs original typed problem and contract")
    if type(adapter) is not StagedDenseTargetAdapter or type(rewrites) is not tuple or len(rewrites) > 128:
        raise ValueError("staged target rewriting needs its typed adapter and bounded ordered rewrite tuple")
    if any(type(r) is not staged.StagedDenseRewrite for r in rewrites):
        raise ValueError("staged target rewrites must use the existing explicit rewrite records")
    staged.StagedDenseProblem.__post_init__(problem)
    staged.StagedDenseContract.__post_init__(contract)
    StagedDenseTargetAdapter.__post_init__(adapter)
    state = _rewrite_snapshot((problem, contract, adapter, rewrites, bounds))
    reader, reader_code = _functions, _functions.__code__
    functions = _functions()
    callbacks = (*functions, adapter.project, adapter.emit, *(r.implementation for r in rewrites))
    owners = tuple(
        (
            callback,
            _callback(callback),
            _rewrite_snapshot(
                (
                    getattr(getattr(callback, "__func__", callback), "__defaults__", None),
                    getattr(getattr(callback, "__func__", callback), "__kwdefaults__", None),
                )
            ),
        )
        for callback in callbacks
    )
    module_reader, snapshot_reader, callback_reader = _module_state, _rewrite_snapshot, _check_callback
    fixed = tuple((f, f.__code__) for f in (module_reader, snapshot_reader, callback_reader))
    snapshots = tuple(module_reader(module, bounds) for module in modules)

    def fresh():
        if _functions is not reader or reader.__code__ is not reader_code:
            raise ValueError("staged target implementation reader changed; replan")
        if any(
            actual is not old or old.__code__ is not code
            for actual, (old, code) in zip((_module_state, _rewrite_snapshot, _check_callback), fixed, strict=True)
        ):
            raise ValueError("staged target snapshot/checker implementation changed; replan")
        current = reader()
        if len(current) != len(functions) or any(a is not b for a, b in zip(current, functions, strict=True)):
            raise ValueError("staged target direct implementation binding changed; replan")
        for callback, snapshot, defaults in owners:
            callback_reader(callback, snapshot)
            function = getattr(callback, "__func__", callback)
            if snapshot_reader((function.__defaults__, function.__kwdefaults__)) != defaults:
                raise ValueError("staged target callback defaults changed; replan")
        if snapshot_reader((problem, contract, adapter, rewrites, bounds)) != state:
            raise ValueError("staged target original contract, adapter or choices changed; replan")
        if tuple(module_reader(module, bounds) for module in modules) != snapshots:
            raise ValueError("staged target original module/SSA changed during callback; replan")

    fresh()
    return fresh


def _project(module, adapter, fresh, bounds):
    fresh()
    result = adapter.project(module)
    fresh()
    if result is not None and (type(result) is not tuple or len(result) > bounds[0]):
        raise ValueError("actual staged projection must be a bounded immutable operation tuple or None")
    return result


def verify_target_staged_dense_emission(
    source,
    actual,
    *,
    problem,
    contract,
    adapter,
    selected_operations,
    max_commands=1_000_000,
    max_products=4 << 20,
    max_ir_bytes=8 << 20,
    max_operations=65536,
):
    """Reproject actual emitted LLVM independently; never accept a plan as proof."""
    bounds = _bounds(max_commands, max_products, max_ir_bytes, max_operations)
    fresh = _guard((source, actual), problem, contract, adapter, (), bounds)
    if not adapter.complete_effects:
        raise ValueError("actual staged emission lacks original exhaustive effect accounting")
    if type(selected_operations) is not tuple or len(selected_operations) > bounds[0]:
        raise ValueError("selected staged effects must be an explicitly bounded immutable tuple")
    selected_state = _rewrite_snapshot(selected_operations)
    before = _project(source, adapter, fresh, bounds)
    after = _project(actual, adapter, fresh, bounds)
    if before is None or after is None:
        raise ValueError("actual staged source/emission has unknown or unaccounted effects")
    if _rewrite_snapshot(selected_operations) != selected_state or after != selected_operations:
        raise ValueError("actual emitted staged effects differ from the checked selected sequence")
    preservation = completion.verify_staged_completion_preservation(before, after, max_commands=max_commands)
    checks = tuple(
        verifier.verify_staged_dense(
            ops, problem=problem, contract=contract, max_commands=max_commands, max_products=max_products
        )
        for ops in (before, after)
    )
    fresh()
    return {
        "source": checks[0],
        "emitted": checks[1],
        "completion_preservation": preservation,
        "qualification": "original target owner required",
    }


_ISSUED = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class TargetStagedDenseResult:
    source: ModuleOp
    module: ModuleOp
    source_operations: tuple | None
    selected_operations: tuple | None
    report: dict
    _check: Callable = field(repr=False)

    def revalidate(self):
        issued = _ISSUED.get(self)
        if issued is None or issued[0]() is not self._check or self._check.__code__ is not issued[1]:
            raise ValueError("staged target result is unissued or its original guard changed")
        check = self._check
        check()
        if self._check is not check or check.__code__ is not issued[1]:
            raise ValueError("staged target result guard changed during validation")


def rewrite_target_staged_dense(
    module,
    *,
    problem,
    contract,
    adapter,
    rewrites=(),
    max_commands=1_000_000,
    max_products=4 << 20,
    max_ir_bytes=8 << 20,
    max_operations=65536,
):
    """Run existing ordered staged proposals and check the actual emitted module.

    Missing exhaustive permission or unknown source projection retains an exact
    original clone. Invalid known effects/emission refuse; they never downgrade
    into a successful fallback. No automatic profitability decision is made.
    """
    bounds = _bounds(max_commands, max_products, max_ir_bytes, max_operations)
    fresh = _guard((module,), problem, contract, adapter, rewrites, bounds)
    source = module.clone()
    source_fresh = _guard((source,), problem, contract, adapter, rewrites, bounds)
    before = _project(module, adapter, fresh, bounds) if adapter.complete_effects else None
    if before is None:
        selected, actual = None, source.clone()
        report = {"status": "fallback", "reason": "missing exhaustive effects or unknown source projection"}
    else:
        before_state = _rewrite_snapshot(before)
        selected, proposal = staged.rewrite_staged_dense(
            before,
            problem=problem,
            contract=contract,
            rewrites=rewrites,
            max_commands=max_commands,
            max_products=max_products,
        )
        completion.verify_staged_completion_preservation(before, selected, max_commands=max_commands)
        fresh()
        source_fresh()
        if _rewrite_snapshot(before) != before_state:
            raise ValueError("original projected effects changed during staged proposal")
        selected_state = _rewrite_snapshot(selected)
        fresh()
        actual = adapter.emit(module, before, selected)
        fresh()
        source_fresh()
        if actual is module or type(actual) is not ModuleOp:
            raise ValueError("staged target emitter must return a new actual module")
        if _rewrite_snapshot(before) != before_state or _rewrite_snapshot(selected) != selected_state:
            raise ValueError("staged target emitter changed original or selected effects")
        emitted = verify_target_staged_dense_emission(
            source,
            actual,
            problem=problem,
            contract=contract,
            adapter=adapter,
            selected_operations=selected,
            max_commands=max_commands,
            max_products=max_products,
            max_ir_bytes=max_ir_bytes,
            max_operations=max_operations,
        )
        fresh()
        source_fresh()
        report = {"status": "checked", "proposal": proposal, "emission": emitted}
    retained = _guard((module, source, actual), problem, contract, adapter, rewrites, bounds)
    result = TargetStagedDenseResult(source, actual, before, selected, report, lambda: None)
    state = _rewrite_snapshot((before, selected, report))
    identities = (source, actual, before, selected, report)

    def check():
        fresh()
        source_fresh()
        retained()
        if (
            any(
                a is not b
                for a, b in zip(
                    (result.source, result.module, result.source_operations, result.selected_operations, result.report),
                    identities,
                    strict=True,
                )
            )
            or _rewrite_snapshot((before, selected, report)) != state
        ):
            raise ValueError("staged target result or retained effects/report changed")

    object.__setattr__(result, "_check", check)
    _ISSUED[result] = (weakref.ref(check), check.__code__)
    result.revalidate()
    return result
