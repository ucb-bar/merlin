"""Bounded iteration checking for counted, independently drained memory blocks.

This checks every dynamic block, not a sampled prefix or affine recurrence. It
preserves counted IR for emission and establishes only the explicitly projected
memory/completion obligations. It supplies no instruction semantics or schedule.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json

from merlin.sched.ir import Call, Const, Kernel, Loop, Ptr, evaluate, check_structure
from merlin.sched.ir.expr import Expr
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.targetgen.isa_spec import resolve_fact_integer

from .temporary_storage import StorageAdapter, StorageOptimizationUnavailable, _project, verify_temporary_storage


@dataclass(frozen=True)
class DrainedPeriodAdmission:
    concurrent_pairs: tuple[tuple[str, str], ...]
    max_pending: int
    hardware_identity: str
    storage_qualification_identity: str
    qualification_identity: str
    facts_sha256: str
    declaration_sha256: str

    def __post_init__(self):
        if (type(self.concurrent_pairs) is not tuple or len(self.concurrent_pairs) > 4096
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(name) is not str or not name for name in pair) for pair in self.concurrent_pairs)
                or len(set(self.concurrent_pairs)) != len(self.concurrent_pairs)
                or type(self.max_pending) is not int or self.max_pending <= 0
                or any(type(value) is not str or not value for value in
                       (self.hardware_identity, self.storage_qualification_identity, self.qualification_identity))
                or any(type(value) is not str or len(value) != 64
                       or any(c not in "0123456789abcdef" for c in value)
                       for value in (self.facts_sha256, self.declaration_sha256))):
            raise ValueError("drained period requires bounded explicit assumptions and identities")


def resolve_drained_period_admission(declaration, facts, *, adapter, qualification_identity):
    """Resolve explicit assumptions, not a certificate or target qualification."""
    if type(adapter) is not StorageAdapter or type(qualification_identity) is not str or not qualification_identity:
        raise ValueError("drained period requires its explicit effect adapter and qualification identity")
    if (type(declaration) is not dict or set(declaration) != {"schema", "concurrent_pairs", "max_pending", "effects"}
            or declaration["schema"] != "merlin.drained_period.v1"
            or declaration["effects"] != "independent_memory_and_explicit_completion"):
        raise ValueError("drained period requires explicit concurrency and memory/completion assumptions")
    pairs = declaration["concurrent_pairs"]
    if (type(pairs) is not list or len(pairs) > 4096
            or any(type(pair) is not list or len(pair) != 2
                   or any(type(name) is not str or not name for name in pair) for pair in pairs)
            or len({tuple(pair) for pair in pairs}) != len(pairs)):
        raise ValueError("drained period concurrency pairs must be bounded, named and unique")
    digest = lambda value: sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return DrainedPeriodAdmission(
        tuple(tuple(pair) for pair in pairs),
        resolve_fact_integer(declaration["max_pending"], facts, label="drained period max pending", minimum=1),
        adapter.configuration_identity, adapter.qualification_identity, qualification_identity,
        digest(facts), digest(declaration),
    )


def _closed(call, variable, iteration):
    def value(argument):
        if isinstance(argument, Expr):
            return Const(evaluate(argument, {variable: iteration}))
        if isinstance(argument, Ptr):
            return replace(argument, offset=Const(evaluate(argument.offset, {variable: iteration})))
        return argument
    return replace(call, args=tuple((name, value(argument)) for name, argument in call.args))


def _pending_peak(calls, effects, admission):
    pending, peak = {}, 0
    for call, effect in zip(calls, effects, strict=True):
        for token in effect.waits:
            del pending[token]  # Existing completion checker already established exact token validity.
        if effect.completion == "wait":
            continue
        if any((previous, call.instr) not in admission.concurrent_pairs for previous in pending.values()):
            raise StorageOptimizationUnavailable("simultaneous instructions lack the explicit concurrency assumption")
        if effect.completion == "asynchronous":
            pending[effect.token] = call.instr
            peak = max(peak, len(pending))
            if peak > admission.max_pending:
                raise ValueError("drained period exceeds its selected pending-work bound")
    return peak


def _identity(kernel, objects, banks, adapter, admission, options):
    return (kernel.digest(), deepcopy((objects, banks, admission, options)),
            (adapter.project, adapter.rewrite, adapter.configuration_identity,
             adapter.numerical_policy_identity, adapter.qualification_identity))


def _check(kernel, objects, banks, adapter, admission, options):
    if (type(kernel) is not Kernel or check_structure(kernel) or kernel.attrs
            or len(kernel.body) != 1 or type(kernel.body[0]) is not Loop
            or any(type(call) is not Call for call in kernel.body[0].body)):
        raise StorageOptimizationUnavailable("drained period requires exactly one counted loop of actual calls")
    if (type(adapter) is not StorageAdapter or type(admission) is not DrainedPeriodAdmission
            or admission.hardware_identity != adapter.configuration_identity
            or admission.storage_qualification_identity != adapter.qualification_identity):
        raise ValueError("drained period admission differs from its selected effect adapter")
    loop = kernel.body[0]
    budget = options["max_dynamic_calls"]
    count = loop.extent * len(loop.body)
    if type(budget) is not int or budget <= 0 or type(loop.extent) is not int or loop.extent <= 0 or not loop.body:
        raise ValueError("drained period requires positive explicit iteration/call bounds")
    if count > budget:
        raise StorageOptimizationUnavailable(f"drained period needs {count} dynamic call checks; budget is {budget}")
    check_integer_domain(kernel, options["index_bits"])
    identity = _identity(kernel, objects, banks, adapter, admission, options)
    digest, peak = sha256(), 0
    check_options = {key: value for key, value in options.items() if key not in ("max_dynamic_calls", "index_bits")}
    for iteration in range(loop.extent):
        block = replace(kernel, body=tuple(_closed(call, loop.var, iteration) for call in loop.body))
        # Reuse actual effect, initialization, alias, reservation, window and
        # completion checks. No allocation or instruction order is changed.
        verify_temporary_storage(block, block, objects, banks, adapter,
                                 max_calls=len(loop.body), **check_options)
        effects = _project(block, adapter, len(loop.body))
        peak = max(peak, _pending_peak(block.body, effects, admission))
        digest.update(repr(effects).encode())
        if _identity(kernel, objects, banks, adapter, admission, options) != identity:
            raise ValueError("drained period source, adapter or selected assumptions changed during checking")
    return {"schema": "merlin.checked_drained_period.v1", "kernel_sha256": kernel.digest(),
            "iterations_checked": loop.extent, "dynamic_calls_checked": count,
            "static_calls": len(loop.body), "maximum_pending": peak,
            "effect_observations_sha256": digest.hexdigest(), "admission": asdict(admission),
            "scope": "bounded iteration memory/completion checks under explicit adapter assumptions",
            "affine_recurrence_proven": False, "target_execution_qualified": False,
            "numerical_equivalence_proven": False, "configuration_invariants_proven": False,
            "external_output_coverage_proven": False, "timing_authority": "NONE"}


@dataclass(frozen=True)
class CheckedDrainedPeriod:
    kernel: Kernel
    objects: tuple
    banks: tuple
    adapter: StorageAdapter
    admission: DrainedPeriodAdmission
    _options: dict
    _identity: tuple
    _checked: dict

    def revalidate(self):
        if _identity(self.kernel, self.objects, self.banks, self.adapter, self.admission, self._options) != self._identity:
            raise ValueError("drained period source, adapter or assumptions changed; recheck")
        if _check(self.kernel, self.objects, self.banks, self.adapter, self.admission, self._options) != self._checked:
            raise ValueError("drained period actual effect projection changed; recheck")

    def report(self):
        """Historical observation bound to its recorded kernel; no fresh recheck."""
        report = deepcopy(self._checked)
        report["freshness"] = "historical_check_only; current effects not reopened"
        return report

    def emit_c(self, instruction_set, *, symbol):
        """Consume unchanged counted IR; target rendering still needs execution checks."""
        from merlin.sched.codegen import emit_c_function
        self.revalidate()
        source = emit_c_function(self.kernel, instruction_set, symbol=symbol, index_bits=self._options["index_bits"])
        self.revalidate()
        return source


def check_drained_period(kernel, objects, banks, adapter, admission, *, max_dynamic_calls,
                         index_bits=64, external_windows=(), private_ownership=False,
                         reservations=(), max_objects=256):
    """Check every bounded iteration, retaining the caller's ordinary route on refusal.

    Each iteration must independently initialize its complete private objects and
    drain all actual asynchronous commands. There is no carried pending work or
    private live-in state. Tails are checked exactly if expressible under those
    same requirements; partial initialization and unbounded domains refuse.
    """
    if index_bits not in (32, 64) or type(index_bits) is not int:
        raise ValueError("drained period requires explicitly selected 32 or 64 bit indices")
    options = dict(max_dynamic_calls=max_dynamic_calls, index_bits=index_bits,
                   external_windows=external_windows, private_ownership=private_ownership,
                   reservations=reservations, max_objects=max_objects)
    checked = _check(kernel, objects, banks, adapter, admission, options)
    return CheckedDrainedPeriod(kernel, objects, banks, adapter, admission, options,
                                _identity(kernel, objects, banks, adapter, admission, options), checked)
