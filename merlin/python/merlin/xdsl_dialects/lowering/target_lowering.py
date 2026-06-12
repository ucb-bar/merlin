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

# interface op -> target op, mirroring merlin/targets/<t>/contracts/dialect_plan.yaml.
LOWERING_TABLES = {
    "toy_npu": {
        "interface.resident_pack": "toynpu.res_pack",
        "interface.matmul": "toynpu.matmul",
        "interface.commit": "toynpu.commit",
        "interface.resident_evict": "toynpu.evict",
    },
    "saturn": {
        "interface.resident_pack": "saturn.pack",
        "interface.matmul": "saturn.matmul",
        "interface.commit": "saturn.commit",
        "interface.resident_evict": "saturn.release",
    },
}


def load_lowering_table(dialect_plan: dict[str, Any] | None = None,
                        target: str = "toy_npu") -> dict[str, str]:
    """{interface op name: target op name} from a dialect_plan dict (or built-in)."""
    if dialect_plan is None:
        return dict(LOWERING_TABLES[target])
    return {rule["from"]: rule["to"] for rule in dialect_plan.get("lowering", [])}


def load_dialect_plan(target: str, repo_root: str | Path | None = None) -> dict[str, Any]:
    """The committed in-tree dialect plan for a reference target."""
    import yaml

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[5]
    path = root / f"merlin/targets/{target}/contracts/dialect_plan.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_toy_dialect_plan(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Back-compat alias for the toy_npu dialect plan."""
    return load_dialect_plan("toy_npu", repo_root)


if HAS_XDSL:

    @dataclass(frozen=True)
    class TargetSpec:
        """The op/type classes a reference target contributes to the rebuild loop."""

        name: str
        dialect_module: Any
        pack_op: type
        matmul_op: type
        commit_op: type
        evict_op: type
        resident_type: type
        accumulator_type: type

    def _specs() -> dict[str, "TargetSpec"]:
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
                    target: str = "toy_npu"):
    """Rebuild the interface module in the target dialect."""
    if not HAS_XDSL:
        return module
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
    from xdsl.dialects.func import FuncOp, ReturnOp

    from .. import interface as i

    specs = _specs()
    if target not in specs:
        raise LoweringError(f"no in-tree reference target dialect for {target!r}")
    spec = specs[target]
    table = load_lowering_table(dialect_plan, target)
    missing = [op for op in LOWERING_TABLES[target] if op not in table]
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
            new = spec.pack_op(
                operands=[value_map[op.src]],
                result_types=[spec.resident_type(op.src.type)],
                properties={"layout": StringAttr(op.layout.data.value)})
            value_map[op.res] = new.res
        elif isinstance(op, i.MatmulOp):
            new = spec.matmul_op(
                operands=[value_map[op.lhs], value_map[op.rhs]],
                result_types=[spec.accumulator_type(op.acc.type.element)])
            value_map[op.acc] = new.acc
        elif isinstance(op, i.CommitOp):
            props = {"epilogue": op.epilogue}
            for key in ("bias", "requant_shift", "output_dtype"):
                val = getattr(op, key)
                if val is not None:
                    props[key] = val
            new = spec.commit_op(operands=[value_map[op.acc]],
                                 result_types=[op.out.type], properties=props)
            value_map[op.out] = new.out
            outs.append(new.out)
            out_types.append(op.out.type)
        elif isinstance(op, i.ResidentEvictOp):
            new = spec.evict_op(operands=[value_map[op.handle]])
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
