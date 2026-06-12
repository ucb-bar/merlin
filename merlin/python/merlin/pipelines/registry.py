"""Registry of named compilation passes. A strategy references passes by name, so a
compilation approach is *data* (a comma-separated pass spec), not code.

A pass is a pure function ``plan -> plan`` over a cost-model *plan* (the interface flag-set
consumed by ``merlin.dse.cost_model.evaluate_cost``: ``pack_count, weight_loads,
per_step_intermediate, dispatch_count, resident_setup, accumulator_setup``). Effect passes
(``hoist-pack``, ``make-resident``, ``defer-commit``, ``batch-dispatch``) carry the cost
semantics of exposing an interface; lowering passes (the ``merlin-*`` / ``*-lower`` names that
will become real xDSL rewrites on the stable plane) are registered as identity here so a real
xDSL pass spec still loads and runs. When the xDSL dialects land, these identities are swapped
for real rewrites without touching the strategy / harness / search layers.
"""
from __future__ import annotations

from typing import Callable

Plan = dict
PassFn = Callable[[Plan], Plan]

_REGISTRY: dict[str, PassFn] = {}


def register_pass(name: str, fn: PassFn | None = None):
    """Register a pass. Usable directly or as a decorator (``@register_pass("name")``)."""
    if fn is not None:
        _REGISTRY[name] = fn
        return fn

    def _decorator(f: PassFn) -> PassFn:
        _REGISTRY[name] = f
        return f
    return _decorator


def get_pass(name: str) -> PassFn | None:
    return _REGISTRY.get(name)


def has_pass(name: str) -> bool:
    return name in _REGISTRY


def list_passes() -> list[str]:
    return sorted(_REGISTRY)


def _set(plan: Plan, **updates) -> Plan:
    out = dict(plan)
    out.update(updates)
    return out


# --- effect passes (carry cost semantics) ---------------------------------------------------
@register_pass("hoist-pack")
def _hoist_pack(plan: Plan) -> Plan:
    """Pack the immutable weight once instead of every step."""
    return _set(plan, pack_count=1)


@register_pass("make-resident")
def _make_resident(plan: Plan) -> Plan:
    """Keep the packed weight resident: load once, pay the resident make/evict setup."""
    return _set(plan, pack_count=1, weight_loads=1, resident_setup=True)


@register_pass("hw-cache")
def _hw_cache(plan: Plan) -> Plan:
    """Hardware-managed reuse: the loaded weight is cached (load once) but not exposed/hoisted."""
    return _set(plan, weight_loads=1)


@register_pass("defer-commit")
def _defer_commit(plan: Plan) -> Plan:
    """Keep the accumulator live across the epilogue; commit once (no i32 intermediate)."""
    return _set(plan, per_step_intermediate=False, accumulator_setup=True)


@register_pass("batch-dispatch")
def _batch_dispatch(plan: Plan) -> Plan:
    """Coalesce the per-step dispatches into a single command submission."""
    return _set(plan, dispatch_count=1)


# --- lowering passes (identity until the xDSL stable plane exists) ---------------------------
for _name in ("merlin-contract", "merlin-schedule", "interface-lower", "toynpu-lower",
              "runtime-lower"):
    register_pass(_name, lambda plan: plan)
