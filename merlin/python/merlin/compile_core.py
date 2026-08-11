"""Route a generic-MLIR payload to the compiler path that can actually carry it.

Merlin has two lowering paths, and which one applies is decided by the PAYLOAD, not by the target:

* the **staged** path (``xdsl_dialects.lowering.lower_module``) — contract -> schedule -> interface
  -> generated target dialect -> runtime -> command buffer. It materializes matmul-family
  computation against a target's own dialect, and every reference target takes it, CPU-ish ones
  included.
* the **LLVM** path (``llvmlower.lower_model``) — upstream MLIR passes to LLVM IR to an object.
  This is the general fallback: it compiles computation for which the target declares no
  accelerated op.

So "accelerator uses the staged path, CPU uses LLVM" is the wrong model. A vector add has no
matmul, so it takes the LLVM path *even on an accelerator target*; a matmul takes the staged path
*even on a CPU-class target*. That is exactly the property a kernel frontend needs, because a
target that accelerates only some ops must still be able to compile the rest.

Nothing here is target-specific: coverage is read from the resolved target's dialect plan, so a
newly generated target routes correctly with no edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The interface-level op classes the staged pipeline can actually MATERIALIZE today. This is a
# property of `interface_lowering.lower_to_interface` (it rebuilds a body as
# resident_pack/matmul/commit/evict), NOT of any target: a plan may legitimately declare coverage
# for an op the interface layer cannot yet build — several generated plans declare `elementwise`,
# for which there is no `interface.elementwise`. Routing on the intersection is what keeps such a
# target from being handed a payload the staged path would silently drop.
STAGED_MATERIALIZABLE: frozenset[str] = frozenset({"matmul"})

# Payload op classes, keyed by the linalg op that expresses them. Matmul-family only, matching
# `input_workload.find_matmuls`, which is what the contract/interface stages actually look for.
_MATMUL_OPS: frozenset[str] = frozenset({
    "linalg.matmul", "linalg.quantized_matmul", "linalg.batch_matmul",
})


class RoutingError(RuntimeError):
    """The payload cannot be routed (no target resolvable, or an empty module)."""


@dataclass(frozen=True)
class Route:
    """Which path was chosen, and the evidence for it."""

    kind: str                                  # "staged" | "llvm"
    reason: str
    payload: tuple[str, ...] = ()              # op classes found in the module
    covered: tuple[str, ...] = ()              # op classes the target's plan declares
    materializable: tuple[str, ...] = ()       # covered AND buildable by the interface layer

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "payload": list(self.payload),
                "covered": list(self.covered), "materializable": list(self.materializable)}


@dataclass
class CoreCompileResult:
    """The route taken plus whichever path's result object was produced."""

    route: Route
    target: str
    staged: Any | None = None                  # xdsl_dialects.lowering.LoweringResult
    llvm: Any | None = None                    # llvmlower.lower.LowerResult
    notes: list[str] = field(default_factory=list)


def payload_classes(module) -> tuple[str, ...]:
    """The op classes present in ``module``, in a target-independent vocabulary.

    Only classes the routing decision turns on are named; everything else is reported as
    ``"generic"``, which is what sends a payload down the LLVM path.

    Read from the entry function's own block rather than a full ``walk()``: named linalg ops carry a
    hidden body region (a ``linalg.yield``), and counting those made every matmul module look like
    it contained generic computation too.

    What counts as "generic" is not a list of op names kept here. It is whatever interface
    materialization would neither rebuild nor safely drop, asked of the interface layer itself
    (``unaccounted_ops``), so the routing decision and the guard that fires later cannot disagree —
    they did when each kept its own list, and a zeroed accumulator was enough to split them.
    """
    from merlin.xdsl_dialects.lowering.interface_lowering import unaccounted_ops

    block = _entry_block(module)
    if block is None:
        return ()
    payload = [op for op in block.ops if op.name in _MATMUL_OPS]
    found: list[str] = ["matmul"] if payload else []
    if unaccounted_ops(block, payload):
        found.append("generic")
    return tuple(found)


def _entry_block(module):
    """The entry function's block — the level the staged pipeline reasons at."""
    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns or not fns[0].body.blocks:
        return None
    return fns[0].body.blocks[0]


def plan_interface_ops(plan: dict[str, Any] | None) -> tuple[str, ...]:
    """The interface-level op classes a dialect plan declares coverage for.

    Handles both spellings that occur in committed plans, structurally rather than by pattern:
    ``{from: interface.matmul, to: <t>.matmul}`` (the curated reference plans) and
    ``{op: matmul, to: <t>.matmul}`` (the generated-from-compute_units plans).
    """
    if not plan:
        return ()
    out: set[str] = set()
    for entry in plan.get("lowering", []) or []:
        if not isinstance(entry, dict):
            continue
        src = entry.get("from")
        if isinstance(src, str) and src:
            # "interface.matmul" -> "matmul"; a bare name stays itself.
            _, _, leaf = src.rpartition(".")
            out.add(leaf or src)
            continue
        op = entry.get("op")
        if isinstance(op, str) and op:
            out.add(op)
    return tuple(sorted(out))


def _resolve_plan(target: str | None, target_package: Any | None,
                  dialect_plan: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """(target name, dialect plan) from whichever of the three inputs was supplied."""
    if dialect_plan is not None and target is not None:
        return target, dialect_plan
    if target_package is not None:
        return target_package.name, dialect_plan or target_package.dialect_plan()
    if target is None:
        raise RoutingError("compile_core_mlir needs a target= or target_package=")
    if dialect_plan is not None:
        return target, dialect_plan
    from merlin.targetgen.target_registry import resolve
    try:
        info = resolve(target)
    except Exception as exc:  # noqa: BLE001 — an unresolvable target is a routing failure
        raise RoutingError(f"cannot resolve target {target!r}: {exc}") from exc
    try:
        return info.name, info.load_dialect_plan()
    except Exception as exc:  # noqa: BLE001
        # FAIL CLOSED. An unreadable plan must never be treated as "this target covers nothing":
        # that silently demotes an accelerator to the generic LLVM path and looks like a successful
        # compile. It is also the common case rather than an exotic one — a target whose plan lives
        # in its out-of-tree package has no in-tree plan to read, and must be passed as
        # target_package=/dialect_plan= instead of by name.
        raise RoutingError(
            f"target {info.name!r} resolved but its dialect plan is unreadable "
            f"({type(exc).__name__}: {exc}). Routing cannot tell 'this target accelerates nothing' "
            f"from 'the plan is somewhere else', and guessing would silently compile the payload as "
            f"generic computation. Pass target_package= (merlin.targetgen.registry.load_target) or "
            f"an explicit dialect_plan=.") from exc


def choose_route(module, *, target: str | None = None, target_package: Any | None = None,
                 dialect_plan: dict[str, Any] | None = None) -> Route:
    """Decide the path WITHOUT compiling — the inspectable half of :func:`compile_core_mlir`."""
    name, plan = _resolve_plan(target, target_package, dialect_plan)
    payload = payload_classes(module)
    if not payload:
        raise RoutingError("module has no recognizable payload to compile")
    covered = plan_interface_ops(plan)
    materializable = tuple(sorted(set(covered) & STAGED_MATERIALIZABLE))

    unstaged = [p for p in payload if p not in materializable]
    if unstaged:
        return Route(
            kind="llvm", payload=payload, covered=covered, materializable=materializable,
            reason=(f"target {name!r} cannot materialize {', '.join(unstaged)} through the staged "
                    f"pipeline (declared coverage: {list(covered) or 'none'}; interface-buildable: "
                    f"{list(materializable) or 'none'}) — compiling as generic computation"))
    return Route(
        kind="staged", payload=payload, covered=covered, materializable=materializable,
        reason=f"target {name!r} materializes {', '.join(payload)} through its own dialect")


def compile_core_mlir(module, *, target: str | None = None, target_package: Any | None = None,
                      dialect_plan: dict[str, Any] | None = None,
                      target_contract: dict[str, Any] | None = None,
                      backend: str | None = None,
                      workdir: str | Path | None = None,
                      **llvm_kwargs: Any) -> CoreCompileResult:
    """Compile a generic-MLIR module for ``target`` down whichever path can carry it.

    ``workdir`` is required only for the LLVM path (it emits files). Keyword arguments are forwarded
    to :func:`llvmlower.lower.lower_model` when that path is chosen.
    """
    route = choose_route(module, target=target, target_package=target_package,
                         dialect_plan=dialect_plan)
    name = target_package.name if target_package is not None else target
    assert name is not None  # _resolve_plan already rejected the both-None case

    if route.kind == "staged":
        from merlin.xdsl_dialects.lowering import lower_module
        res = lower_module(module, target=name, target_contract=target_contract,
                           dialect_plan=dialect_plan, backend=backend,
                           target_package=target_package)
        return CoreCompileResult(route=route, target=name, staged=res)

    if workdir is None:
        raise RoutingError(
            f"{route.reason}. The LLVM path writes artifacts, so compile_core_mlir needs workdir=")
    from merlin.llvmlower.lower import lower_model
    from merlin.xdsl_dialects._common import text
    res = lower_model(text(module), workdir, **llvm_kwargs)
    return CoreCompileResult(route=route, target=name, llvm=res)
