"""``interface`` -> target dialect (merlin-interface-to-target stage).

Driven by the target's dialect plan lowering table (interface.<op> -> <target>.<op>).
In-tree reference targets: ``toynpu`` (NPU with real resident storage) and ``saturn``
(RVV CPU where residency is a packed weight kept live in memory). Both share the
pack / matmul / commit / evict op shape, so one rebuild loop serves both via a
:class:`TargetSpec`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._common import HAS_XDSL
from .interface_lowering import LoweringError

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
        op_properties: dict[str, dict[str, Any]] | None = None
        # Optional: only targets whose dialect plan covers interface.elementwise supply one.
        elementwise_op: type | None = None

        def extra(self, kind: str) -> dict[str, Any]:
            """Contract-derived properties this target requires on the ``kind`` op ('matmul', …)."""
            return dict((self.op_properties or {}).get(kind, {}))

    def _specs() -> dict[str, "TargetSpec"]:
        # Built-in REFERENCE targets only. Generated targets (e.g. gemmini) are NOT hardcoded
        # here — they load from isolated per-run packages via merlin.targetgen.registry and are
        # passed in through the ``spec`` argument of lower_to_target.
        from ..targets import saturn as sat
        from ..targets import toynpu as toy

        return {
            "toy_npu": TargetSpec("toy_npu", toy, toy.ResPackOp, toy.MatmulOp,
                                  toy.CommitOp, toy.EvictOp, toy.ResidentTensorType,
                                  toy.AccumulatorType),
            "saturn": TargetSpec("saturn", sat, sat.PackOp, sat.MatmulOp,
                                 sat.CommitOp, sat.ReleaseOp, sat.PackedTensorType,
                                 sat.AccumulatorType),
        }


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
    outs = []
    out_types = []
    for op in src_block.ops:
        if isinstance(op, i.ResidentPackOp):
            props = {"layout": StringAttr(op.layout.data.value)}
            props.update(spec.extra("pack"))
            new = spec.pack_op(
                operands=[value_map[op.src]],
                result_types=[spec.resident_type(op.src.type)],
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
            outs.append(new.out)
            out_types.append(op.out.type)
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
            outs.append(new.out)
            out_types.append(op.out.type)
        elif isinstance(op, i.ResidentEvictOp):
            new = spec.evict_op(operands=[value_map[op.handle]],
                                properties=spec.extra("evict"))
        elif op.name == "func.return":
            continue
        else:
            raise LoweringError(f"no {target} lowering for {op.name}")
        ops.append(new)
    ops.append(ReturnOp(*outs))
    blk.add_ops(ops)
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists(arg_types, out_types),
                    Region([blk]))
    return ModuleOp([new_fn])
