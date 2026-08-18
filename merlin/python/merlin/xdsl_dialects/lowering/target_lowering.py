"""``interface`` -> target dialect (merlin-interface-to-target stage).

Driven by the target's dialect plan lowering table (interface.<op> -> <target>.<op>).
The one BUILT-IN reference target is ``toynpu`` (NPU with real resident storage, the sanctioned neutral
example). Other reference target dialects (e.g. ``saturn``, an RVV CPU where residency is a packed
weight kept live in memory) are DISCOVERED: their target package contributes the dialect spec as a
plugin (``plugin.dialect`` in the contract), self-registered into the same registry ``_specs()`` builds.
All share the pack / matmul / commit / evict op shape, so one rebuild loop serves them via a
:class:`TargetSpec`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._common import HAS_XDSL
from .interface_lowering import LoweringError

# --- discovered target-package dialect specs -------------------------------------------------------
# A reference target's dialect is DATA its own package contributes (``plugin.dialect``), not a hardcoded
# entry here. The package's dialect module self-registers its TargetSpec (and its target-op -> opcode
# map) at import via :func:`register_dialect_spec`; discovery imports those modules so ``_specs()`` and
# ``runtime_lowering`` pick them up without shared code naming a target. Mirrors the backend /
# sim-oracle plugin seams (merlin.runtime.backends.base).
_PLUGIN_SPECS: dict[str, Any] = {}
_PLUGIN_OPCODES: dict[str, dict[str, str]] = {}
_dialect_env_seen: str | None = None


def register_dialect_spec(spec: Any, opcodes: dict[str, str] | None = None) -> None:
    """A target package's dialect module calls this at import to contribute its :class:`TargetSpec`
    (and optional target-op -> command-buffer opcode map) to the reference lowering registry
    (idempotent per ``spec.name``)."""
    _PLUGIN_SPECS[spec.name] = spec
    if opcodes:
        _PLUGIN_OPCODES[spec.name] = dict(opcodes)


def _ensure_dialects_discovered() -> None:
    """Import the dialect module every target package declares via its contract ``plugin.dialect`` — a
    module that calls :func:`register_dialect_spec` at import — through the SAME OOT/reference plugin
    discovery the runtime backends and sim oracles use. Re-scans only when ``MERLIN_TARGET_PATH``
    changes; registration is idempotent, so repeated scans are harmless. Best-effort: a broken/optional
    plugin must not break lowering."""
    global _dialect_env_seen
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _dialect_env_seen:
        return
    _dialect_env_seen = key
    try:
        from ...runtime.backends import base as _bk
        for name, path in _bk._oot_plugin_modules("dialect"):
            _bk._load_oot_backend(name, path, ns="merlin._oot_dialects")
    except Exception:  # noqa: BLE001 — discovery is best-effort; a broken plugin must not break lowering
        pass


def plugin_opcodes() -> dict[str, dict[str, str]]:
    """Discovered target-op -> opcode maps, keyed by target name (runs discovery first). Consumed by
    ``runtime_lowering.lower_to_runtime`` so a discovered dialect's encoding rides with its package."""
    _ensure_dialects_discovered()
    return dict(_PLUGIN_OPCODES)

# The interface ops a tensor-resident target must lower (used to check coverage for both
# built-in reference targets and isolated/generated target packages).
EXPECTED_INTERFACE_OPS = ("interface.resident_pack", "interface.matmul",
                          "interface.commit", "interface.resident_evict")


def load_lowering_table(dialect_plan: dict[str, Any] | None = None,
                        target: str = "toy_npu") -> dict[str, str]:
    """{interface op name: target op name} from a dialect_plan dict, or the target's committed plan
    (via the target registry — no hardcoded per-target table)."""
    if dialect_plan is None:
        from merlin.targetgen.target_registry import load_dialect_plan
        dialect_plan = load_dialect_plan(target)
    return {rule["from"]: rule["to"] for rule in dialect_plan.get("lowering", [])}


def load_dialect_plan(target: str, repo_root: str | Path | None = None) -> dict[str, Any]:
    """The committed in-tree dialect plan for a reference target (via the target registry)."""
    import yaml

    if repo_root is not None:   # explicit-root override (tests) keeps the direct read
        path = Path(repo_root) / f"merlin/targets/{target}/contracts/dialect_plan.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    from merlin.targetgen.target_registry import load_dialect_plan as _reg
    return _reg(target)


def load_toy_dialect_plan(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Back-compat alias for the toy_npu dialect plan."""
    return load_dialect_plan("toy_npu", repo_root)


if HAS_XDSL:

    @dataclass(frozen=True)
    class TargetSpec:
        """The op/type classes a reference target contributes to the rebuild loop.

        ``op_properties`` lets a target dialect require facts that only its own contract can
        supply. The two in-tree reference targets need none, but a SIMT target does: Radiance
        carries ``compiler_obligations: [must_map_to_warps]`` and
        ``capabilities.simt.lanes_per_warp``, and an obligation with nowhere to be recorded is not
        an obligation. The rebuild loop merges whatever the package declared and never interprets
        it, so core stays target-blind: it is the package that reads its contract and fails closed
        when the fact is missing.
        """

        name: str
        dialect_module: Any
        pack_op: type
        matmul_op: type
        commit_op: type
        evict_op: type
        resident_type: type
        accumulator_type: type
        # Optional non-matmul vector-lane ops — None when the target lowers neither.
        vector_map_op: type | None = None
        vector_reduce_op: type | None = None
        op_properties: dict[str, dict[str, Any]] | None = None
        # Optional: only targets whose dialect plan covers interface.elementwise supply one.
        elementwise_op: type | None = None

        def extra(self, kind: str) -> dict[str, Any]:
            """Contract-derived properties this target requires on the ``kind`` op ('matmul', …)."""
            return dict((self.op_properties or {}).get(kind, {}))

    def _specs() -> dict[str, "TargetSpec"]:
        # The ONE built-in REFERENCE target is toy_npu (the sanctioned neutral example). Other reference
        # target dialects (e.g. saturn) are DISCOVERED — their package contributes the spec via
        # ``plugin.dialect`` and self-registers it (see _ensure_dialects_discovered), so no target name is
        # hardcoded here. Generated targets (e.g. gemmini) are not registered here at all — they load from
        # isolated per-run packages via merlin.targetgen.registry and pass their spec through the ``spec``
        # argument of lower_to_target.
        from ..targets import toynpu as toy

        specs = {
            "toy_npu": TargetSpec("toy_npu", toy, toy.ResPackOp, toy.MatmulOp,
                                  toy.CommitOp, toy.EvictOp, toy.ResidentTensorType,
                                  toy.AccumulatorType,
                                  vector_map_op=getattr(toy, "VectorMapOp", None),
                                  vector_reduce_op=getattr(toy, "VectorReduceOp", None)),
        }
        _ensure_dialects_discovered()   # load any target-package-contributed dialect specs (e.g. saturn)
        specs.update(_PLUGIN_SPECS)
        return specs


def lower_to_target(module, dialect_plan: dict[str, Any] | None = None,
                    target: str = "toy_npu", spec=None):
    """Rebuild the interface module in the target dialect.

    ``spec`` (a :class:`TargetSpec`) overrides the built-in reference lookup — this is how an
    isolated/generated target package (loaded via merlin.targetgen.registry) supplies its own
    dialect, without the target being hardcoded in this module. When ``spec`` is given,
    ``dialect_plan`` carries the package's interface->target lowering table.
    """
    if not HAS_XDSL:
        return module
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
    from xdsl.dialects.func import FuncOp, ReturnOp

    from .. import interface as i

    if spec is None:
        specs = _specs()
        if target not in specs:
            raise LoweringError(f"no in-tree reference target for {target!r}; pass a loaded "
                                f"target package's spec (merlin.targetgen.registry.load_target)")
        spec = specs[target]
    table = load_lowering_table(dialect_plan, target if dialect_plan is None else None)
    missing = [op for op in EXPECTED_INTERFACE_OPS if op not in table]
    if missing:
        raise LoweringError("dialect plan does not lower: %s" % ", ".join(missing))

    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise LoweringError("no func.func in module")
    fn = fns[0]
    src_block = fn.body.blocks[0]
    arg_types = [a.type for a in src_block.args]
    blk = Block(arg_types=arg_types)
    value_map = dict(zip(src_block.args, blk.args))

    ops = []
    ret_op = None
    for op in src_block.ops:
        if isinstance(op, i.ResidentPackOp):
            props = {"layout": StringAttr(op.layout.data.value)}
            scale = value_map[op.scale] if op.scale is not None else None
            if op.dequant_axis is not None:
                props["dequant_axis"] = op.dequant_axis
            props.update(spec.extra("pack"))
            # A target's pack op may or may not carry the optional dequant-scale operand (generated
            # dialects predating it have only `src`). Match its arity; a dequant pack against a
            # scale-less pack op is an honest LoweringError, not a silent drop.
            pack_operands = [n for n, _ in spec.pack_op.get_irdl_definition().operands]
            if "scale" in pack_operands:
                operands = [value_map[op.src], scale]
            elif scale is not None:
                raise LoweringError(f"target {spec.name!r} pack op has no dequant scale operand")
            else:
                operands = [value_map[op.src]]
            # Mirror the interface resident element type (f32 for a dequant pack, else the src type).
            new = spec.pack_op(operands=operands,
                               result_types=[spec.resident_type(op.res.type.element)],
                               properties=props)
            value_map[op.res] = new.res
        elif isinstance(op, i.MatmulOp):
            new = spec.matmul_op(
                operands=[value_map[op.lhs], value_map[op.rhs]],
                result_types=[spec.accumulator_type(op.acc.type.element)],
                properties=spec.extra("matmul"))
            value_map[op.acc] = new.acc
        elif isinstance(op, i.CommitOp):
            props = {"epilogue": op.epilogue}
            for key in ("bias", "requant_shift", "output_dtype"):
                val = getattr(op, key)
                if val is not None:
                    props[key] = val
            props.update(spec.extra("commit"))
            new = spec.commit_op(operands=[value_map[op.acc]],
                                 result_types=[op.out.type], properties=props)
            value_map[op.out] = new.out
        elif isinstance(op, i.ResidentEvictOp):
            new = spec.evict_op(operands=[value_map[op.handle]],
                                properties=spec.extra("evict"))
        elif isinstance(op, i.VectorMapOp):
            if spec.vector_map_op is None:
                raise LoweringError(f"target {spec.name!r} does not lower interface.vector_map")
            props = {"combine": op.combine}
            if op.activation is not None:
                props["activation"] = op.activation
            new = spec.vector_map_op(operands=[value_map[op.lhs], value_map[op.rhs]],
                                     result_types=[op.out.type], properties=props)
            value_map[op.out] = new.out
        elif isinstance(op, i.VectorReduceOp):
            if spec.vector_reduce_op is None:
                raise LoweringError(f"target {spec.name!r} does not lower interface.vector_reduce")
            new = spec.vector_reduce_op(operands=[value_map[op.src]],
                                        result_types=[op.out.type],
                                        properties={"reduce": op.reduce})
            value_map[op.out] = new.out
        elif isinstance(op, i.ElementwiseOp):
            if "interface.elementwise" not in table:
                raise LoweringError(
                    f"target {target!r}'s dialect plan does not lower interface.elementwise, so this "
                    "payload cannot descend to it. Coverage is read from the plan, never assumed "
                    "from the dialect happening to have the op.")
            if spec.elementwise_op is None:
                raise LoweringError(
                    f"target {target!r} has no elementwise op, so interface.elementwise cannot be "
                    "lowered. A dialect plan that maps interface.elementwise must come with a "
                    "target op for it (SPEC_OPS['elementwise']) — a declared capability the dialect "
                    "cannot express is the gap this check exists to surface.")
            props = {"combine": op.combine}
            if op.activation is not None:
                props["activation"] = op.activation
            props.update(spec.extra("elementwise"))
            new = spec.elementwise_op(
                operands=[value_map[op.lhs], value_map[op.rhs]],
                result_types=[op.out.type], properties=props)
            value_map[op.out] = new.out
        elif op.name == "func.return":
            ret_op = op
            continue
        else:
            raise LoweringError(f"no {target} lowering for {op.name}")
        ops.append(new)
    # Return exactly what the interface function returned — NOT every commit (a chained
    # layer's committed output is an intermediate, not a result).
    outs, out_types = [], []
    if ret_op is not None:
        for operand in ret_op.operands:
            outs.append(value_map[operand])
            out_types.append(value_map[operand].type)
    ops.append(ReturnOp(*outs))
    blk.add_ops(ops)
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists(arg_types, out_types),
                    Region([blk]))
    return ModuleOp([new_fn])
