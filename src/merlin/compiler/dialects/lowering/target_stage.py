"""Parametric builder for a tensor-resident target dialect (xDSL).

Every tensor-resident target (the toy_npu reference and each accelerator) shares ONE dialect shape — four ops
(pack / matmul / commit / evict) over two types (a resident/packed tensor + an accumulator) — that
differ only in op/type NAMES and two small structural knobs. Those used to be hand-copied per target
legacy per-target dialect modules were ~90% identical. This factory synthesizes the IRDL op
and type classes dynamically from the target's committed ``dialect_plan.yaml`` (op/type names + the
interface→target lowering roles), so a reference target module is now data + one ``build_dialect``
call, and a generated target can be built the same way from its plan.

The two knobs the plan does not (yet) encode are passed explicitly by the caller:
``matmul_rhs_typed`` (NPU requires the RHS to be resident; a CPU accepts a plain tensor) and
``matmul_vl_policy`` (vector targets carry an optional vl-policy prop). Neither changes the emitted
MLIR for valid inputs — only op verification — so the built dialects reproduce the hand-written ones
byte-for-byte (golden-checked in tests/ir/test_target_factory.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException

from merlin.compiler.dialects.interface import missing_pool_props
from merlin.compiler.dialects.lowering.interface_stage import LoweringError
from merlin.compiler.dialects.xdsl_context import HAS_XDSL
from merlin.runtime.engine.command_buffer import EPILOGUE_STAGE_SET as KNOWN_EPILOGUE  # noqa: E402
from merlin.target.capabilities.families import DEFAULT_EXAMPLE_TARGET

# ---- factory -----------------------------------------------------------------------------------

# The pooling-geometry requirement is SHARED with the interface dialect rather than re-typed here:
# two copies of "which properties a maxpool stage needs" is exactly the drift this module was written
# to remove (the per-target dialects it replaced were ~90% hand-copied).

# The epilogue vocabulary is DERIVED from the one definition the ABI, its JSON validator and the three
# engines share -- not restated here (this copy had drifted from both of the other two).

# Must stay aligned with interface.KNOWN_{COMBINE,ACTIVATION,REDUCE} and the runtime engine.
KNOWN_COMBINE = {"add", "mul", "identity"}
KNOWN_ACTIVATION = {"relu"}
KNOWN_REDUCE = {"sum"}


@dataclass(frozen=True)
class BuiltDialect:
    """A dynamically-built tensor-resident dialect + the handles the lowering loop needs."""

    dialect: Dialect
    spec: Any  # a target_lowering.TargetSpec
    pack_op: type
    matmul_op: type
    commit_op: type
    evict_op: type
    resident_type: type
    accumulator_type: type
    # Optional non-matmul vector-lane ops — built only when the plan declares a lowering rule
    # for interface.vector_map / interface.vector_reduce (a target may support neither).
    vector_map_op: type | None = None
    vector_reduce_op: type | None = None


def _roles(plan: dict[str, Any]) -> dict[str, str]:
    """interface role (resident_pack/matmul/commit/resident_evict) -> target op name."""
    out = {}
    for rule in plan.get("lowering", []):
        out[rule["from"].split(".", 1)[1]] = rule["to"].split(".", 1)[1]
    return out


def build_dialect(
    target: str, *, plan: dict[str, Any] | None = None, matmul_rhs_typed: bool = False, matmul_vl_policy: bool = False
) -> BuiltDialect:
    """Build the tensor-resident dialect for ``target``. ``plan`` overrides the committed
    dialect_plan (e.g. a freshly TargetGen-generated plan that isn't on disk yet); otherwise the
    target's committed plan is loaded via the registry."""

    if plan is None:
        from merlin.target.registry.identity import load_dialect_plan

        plan = load_dialect_plan(target)
    dname = plan["dialect_name"]
    roles = _roles(plan)
    pack_n, mm_n = roles["resident_pack"], roles["matmul"]
    commit_n, evict_n = roles["commit"], roles["resident_evict"]
    resident_tn, acc_tn = (plan["types"][0]["name"], plan["types"][1]["name"])

    def _mk_type(tn: str) -> type:
        cls = type(
            f"{dname}_{tn}",
            (ParametrizedAttribute, TypeAttribute),
            {
                "name": f"{dname}.{tn}",
                "__annotations__": {"element_type": Attribute},
                "__module__": __name__,
                "__doc__": f"!{dname}.{tn}<element_type>",
            },
        )
        return irdl_attr_definition(cls)

    resident_type = _mk_type(resident_tn)
    accumulator_type = _mk_type(acc_tn)

    def _op(clsname: str, ns: dict) -> type:
        ns = {**ns, "__module__": __name__}
        return irdl_op_definition(type(clsname, (IRDLOperation,), ns))

    pack_op = _op(
        f"{dname}_{pack_n}",
        {
            "name": f"{dname}.{pack_n}",
            "src": operand_def(),
            "scale": opt_operand_def(),
            "layout": prop_def(StringAttr),
            "dequant_axis": opt_prop_def(IntegerAttr),
            "res": result_def(resident_type),
        },
    )

    mm_ns = {
        "name": f"{dname}.{mm_n}",
        "lhs": operand_def(),
        "rhs": operand_def(resident_type) if matmul_rhs_typed else operand_def(),
        "acc": result_def(accumulator_type),
    }
    if matmul_vl_policy:
        mm_ns["vl_policy"] = opt_prop_def(StringAttr)
    matmul_op = _op(f"{dname}_{mm_n}", mm_ns)

    def _commit_verify(self) -> None:
        stages = []
        for entry in self.epilogue:
            stage = entry.data if isinstance(entry, StringAttr) else None
            if stage not in KNOWN_EPILOGUE:
                raise VerifyException(f"{dname}.{commit_n} epilogue stage {stage!r} not in {sorted(KNOWN_EPILOGUE)}")
            stages.append(stage)
        if ("bias" in stages or "bias_add" in stages) and self.bias is None:
            raise VerifyException(f"{dname}.{commit_n} epilogue has a bias stage but no `bias` tensor name")
        missing = missing_pool_props(stages, self.properties)
        if missing:
            raise VerifyException(
                f"{dname}.{commit_n} epilogue has `maxpool` but no "
                f"{', '.join(repr(k) for k in missing)}; a pooling stage with no window cannot be "
                f"executed, and defaulting one would commit a tensor of the wrong extent"
            )

    commit_op = _op(
        f"{dname}_{commit_n}",
        {
            "name": f"{dname}.{commit_n}",
            "acc": operand_def(accumulator_type),
            "epilogue": prop_def(ArrayAttr),
            "bias": opt_prop_def(StringAttr),
            "requant_shift": opt_prop_def(IntegerAttr),
            "output_dtype": opt_prop_def(StringAttr),
            "out": result_def(),
            "verify_": _commit_verify,
        },
    )

    evict_op = _op(f"{dname}_{evict_n}", {"name": f"{dname}.{evict_n}", "handle": operand_def(resident_type)})

    # Non-matmul vector-lane ops are OPTIONAL: synthesized only when the plan lowers them, so a
    # target (or a generated package predating them) that supports neither is unaffected.
    vector_map_op = vector_reduce_op = None
    if "vector_map" in roles:
        vmap_n = roles["vector_map"]

        def _vmap_verify(self) -> None:
            if self.combine.data not in KNOWN_COMBINE:
                raise VerifyException(f"{dname}.{vmap_n} combine {self.combine.data!r} not in {sorted(KNOWN_COMBINE)}")
            if self.activation is not None:
                for entry in self.activation:
                    act = entry.data if isinstance(entry, StringAttr) else None
                    if act not in KNOWN_ACTIVATION:
                        raise VerifyException(f"{dname}.{vmap_n} activation {act!r} not in {sorted(KNOWN_ACTIVATION)}")

        vector_map_op = _op(
            f"{dname}_{vmap_n}",
            {
                "name": f"{dname}.{vmap_n}",
                "lhs": operand_def(),
                "rhs": operand_def(),
                "combine": prop_def(StringAttr),
                "activation": opt_prop_def(ArrayAttr),
                "out": result_def(),
                "verify_": _vmap_verify,
            },
        )
    if "vector_reduce" in roles:
        vred_n = roles["vector_reduce"]

        def _vred_verify(self) -> None:
            if self.reduce.data not in KNOWN_REDUCE:
                raise VerifyException(f"{dname}.{vred_n} op {self.reduce.data!r} not in {sorted(KNOWN_REDUCE)}")

        vector_reduce_op = _op(
            f"{dname}_{vred_n}",
            {
                "name": f"{dname}.{vred_n}",
                "src": operand_def(),
                "reduce": prop_def(StringAttr, prop_name="op"),
                "out": result_def(),
                "verify_": _vred_verify,
            },
        )

    extra_ops = [o for o in (vector_map_op, vector_reduce_op) if o is not None]
    dialect = Dialect(dname, [pack_op, matmul_op, commit_op, evict_op, *extra_ops], [resident_type, accumulator_type])
    spec = TargetSpec(
        target,
        None,
        pack_op,
        matmul_op,
        commit_op,
        evict_op,
        resident_type,
        accumulator_type,
        vector_map_op=vector_map_op,
        vector_reduce_op=vector_reduce_op,
    )
    return BuiltDialect(
        dialect,
        spec,
        pack_op,
        matmul_op,
        commit_op,
        evict_op,
        resident_type,
        accumulator_type,
        vector_map_op=vector_map_op,
        vector_reduce_op=vector_reduce_op,
    )


# ---- target_lowering ---------------------------------------------------------------------------
# ``interface`` -> target dialect (merlin-interface-to-target stage).
#
# Driven by the target's dialect plan lowering table (interface.<op> -> <target>.<op>).
# Authored examples can opt into the generic xDSL dialect factory through their dialect plan;
# executable OOT target dialects contribute a selected plugin (``plugin.dialect`` in the contract).
# All share the pack / matmul / commit / evict op shape, so one rebuild loop serves them via a
# :class:`TargetSpec`.

# --- discovered target-package dialect specs -------------------------------------------------------
# A reference target's dialect is DATA its own package contributes (``plugin.dialect``), not a hardcoded
# entry here. The package's dialect module self-registers its TargetSpec (and its target-op -> opcode
# map) at import via :func:`register_dialect_spec`; discovery imports those modules so ``_specs()`` and
# ``runtime_lowering`` pick them up without shared code naming a target. Mirrors the backend /
# sim-oracle plugin seams (merlin.runtime.backends.registry).
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
    from merlin.runtime.backends import registry as _bk

    _bk._assert_oot_plugin_ownership()
    key = os.environ.get("MERLIN_TARGET_PATH", "")
    if key == _dialect_env_seen:
        return
    try:
        for name, path in _bk._oot_plugin_modules("dialect"):
            _bk._load_oot_backend(name, path, ns="merlin._oot_dialects")
    except _bk.PluginOwnershipError:
        raise
    except Exception:  # noqa: BLE001 — discovery is best-effort; a broken plugin must not break lowering
        pass
    _dialect_env_seen = key


def plugin_opcodes() -> dict[str, dict[str, str]]:
    """Discovered target-op -> opcode maps, keyed by target name (runs discovery first). Consumed by
    ``runtime_lowering.lower_to_runtime`` so a discovered dialect's encoding rides with its package."""
    _ensure_dialects_discovered()
    return {**_reference_opcodes(), **_PLUGIN_OPCODES}


def _reference_plans():
    """Author-selected checkout examples; installed core has none unless support is selected."""
    from merlin.target.registry.identity import explicit_targets, load_dialect_plan, reference_targets

    for name, root in {**reference_targets(), **explicit_targets()}.items():
        if not (root / "contracts/dialect_plan.yaml").is_file():
            continue
        plan = load_dialect_plan(name)
        if isinstance(plan.get("construction"), dict):
            yield name, plan


def _reference_opcodes() -> dict[str, dict[str, str]]:
    mappings = {}
    for name, plan in _reference_plans():
        declared = plan.get("runtime_opcodes")
        if not isinstance(declared, dict) or not declared:
            raise LoweringError(f"{name}: construction plan needs runtime_opcodes")
        prefix = plan["dialect_name"] + "."
        mappings[name] = {prefix + key: value for key, value in declared.items()}
    return mappings


# The interface ops a tensor-resident target must lower (used to check coverage for both
# built-in reference targets and isolated/generated target packages).
EXPECTED_INTERFACE_OPS = ("interface.resident_pack", "interface.matmul", "interface.commit", "interface.resident_evict")


def load_lowering_table(
    dialect_plan: dict[str, Any] | None = None, target: str = DEFAULT_EXAMPLE_TARGET
) -> dict[str, str]:
    """{interface op name: target op name} from a dialect_plan dict, or the target's committed plan
    (via the target registry — no hardcoded per-target table)."""
    if dialect_plan is None:
        from merlin.target.registry.identity import load_dialect_plan

        dialect_plan = load_dialect_plan(target)
    return {rule["from"]: rule["to"] for rule in dialect_plan.get("lowering", [])}


def load_dialect_plan(target: str, repo_root: str | Path | None = None) -> dict[str, Any]:
    """The committed in-tree dialect plan for a reference target (via the target registry)."""
    import yaml

    if repo_root is not None:  # explicit-root override (tests) keeps the direct read
        path = Path(repo_root) / f"examples/{target}/target/contracts/dialect_plan.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    from merlin.target.registry.identity import load_dialect_plan as _reg

    return _reg(target)


def load_toy_dialect_plan(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Back-compat alias for the toy_npu dialect plan."""
    return load_dialect_plan(DEFAULT_EXAMPLE_TARGET, repo_root)


if HAS_XDSL:

    @dataclass(frozen=True)
    class TargetSpec:
        """The op/type classes a reference target contributes to the rebuild loop.

        ``op_properties`` lets a target dialect require facts that only its own contract can
        supply. The two in-tree reference targets need none, but a SIMT target does: its contract
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

        specs = {}
        for name, plan in _reference_plans():
            options = plan["construction"]
            for key in ("matmul_rhs_typed", "matmul_vl_policy"):
                if key in options and not isinstance(options[key], bool):
                    raise LoweringError(f"{name}: construction.{key} must be boolean")
            built = build_dialect(
                name,
                plan=plan,
                matmul_rhs_typed=options.get("matmul_rhs_typed", False),
                matmul_vl_policy=options.get("matmul_vl_policy", False),
            )
            module = SimpleNamespace(get_dialect=lambda dialect=built.dialect: dialect)
            specs[name] = replace(built.spec, dialect_module=module)
        _ensure_dialects_discovered()
        specs.update(_PLUGIN_SPECS)
        return specs


def lower_to_target(
    module, dialect_plan: dict[str, Any] | None = None, target: str = DEFAULT_EXAMPLE_TARGET, spec=None
):
    """Rebuild the interface module in the target dialect.

    ``spec`` (a :class:`TargetSpec`) overrides the built-in reference lookup — this is how an
    isolated/generated target package supplies its own
    dialect, without the target being hardcoded in this module. When ``spec`` is given,
    ``dialect_plan`` carries the package's interface->target lowering table.
    """
    if not HAS_XDSL:
        return module
    from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from merlin.compiler.dialects import interface as i

    if spec is None:
        specs = _specs()
        if target not in specs:
            raise LoweringError(f"no in-tree reference target for {target!r}; pass a loaded target package's spec")
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
            new = spec.pack_op(
                operands=operands, result_types=[spec.resident_type(op.res.type.element)], properties=props
            )
            value_map[op.res] = new.res
        elif isinstance(op, i.MatmulOp):
            new = spec.matmul_op(
                operands=[value_map[op.lhs], value_map[op.rhs]],
                result_types=[spec.accumulator_type(op.acc.type.element)],
                properties=spec.extra("matmul"),
            )
            value_map[op.acc] = new.acc
        elif isinstance(op, i.CommitOp):
            if op.readout is not None or len(op.readout_params):
                raise LoweringError(
                    "legacy target lowering does not implement structured Interface readout; "
                    "use the selected backend's canonical Interface conversion"
                )
            props = {"epilogue": op.epilogue}
            for key in ("bias", "requant_shift", "output_dtype"):
                val = getattr(op, key)
                if val is not None:
                    props[key] = val
            props.update(spec.extra("commit"))
            new = spec.commit_op(operands=[value_map[op.acc]], result_types=[op.out.type], properties=props)
            value_map[op.out] = new.out
        elif isinstance(op, i.ResidentEvictOp):
            new = spec.evict_op(operands=[value_map[op.handle]], properties=spec.extra("evict"))
        elif isinstance(op, i.VectorMapOp):
            if spec.vector_map_op is None:
                raise LoweringError(f"target {spec.name!r} does not lower interface.vector_map")
            props = {"combine": op.combine}
            if op.activation is not None:
                props["activation"] = op.activation
            new = spec.vector_map_op(
                operands=[value_map[op.lhs], value_map[op.rhs]], result_types=[op.out.type], properties=props
            )
            value_map[op.out] = new.out
        elif isinstance(op, i.VectorReduceOp):
            if spec.vector_reduce_op is None:
                raise LoweringError(f"target {spec.name!r} does not lower interface.vector_reduce")
            new = spec.vector_reduce_op(
                operands=[value_map[op.src]], result_types=[op.out.type], properties={"reduce": op.reduce}
            )
            value_map[op.out] = new.out
        elif isinstance(op, i.ElementwiseOp):
            if "interface.elementwise" not in table:
                raise LoweringError(
                    f"target {target!r}'s dialect plan does not lower interface.elementwise, so this "
                    "payload cannot descend to it. Coverage is read from the plan, never assumed "
                    "from the dialect happening to have the op."
                )
            if spec.elementwise_op is None:
                raise LoweringError(
                    f"target {target!r} has no elementwise op, so interface.elementwise cannot be "
                    "lowered. A dialect plan that maps interface.elementwise must come with a "
                    "target op for it (SPEC_OPS['elementwise']) — a declared capability the dialect "
                    "cannot express is the gap this check exists to surface."
                )
            props = {"combine": op.combine}
            if op.activation is not None:
                props["activation"] = op.activation
            props.update(spec.extra("elementwise"))
            new = spec.elementwise_op(
                operands=[value_map[op.lhs], value_map[op.rhs]], result_types=[op.out.type], properties=props
            )
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
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists(arg_types, out_types), Region([blk]))
    return ModuleOp([new_fn])
