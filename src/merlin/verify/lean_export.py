"""Structural export of a deliberately small, exact integer xDSL domain.

The source and target use identical argument bindings and return identities.
Unknown operations, poison flags, memory effects and floats abstain. Primitive
declarations are frozen caller inputs, never inferred from instruction names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class UnsupportedLeanIR(ValueError):
    pass


@dataclass(frozen=True)
class ScalarFunction:
    name: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    bindings: tuple[str, ...]
    values: tuple[str, ...]

    def term(self) -> str:
        body = "\n".join(self.bindings)
        returned = "(" + ", ".join(self.values) + ")" if len(self.values) > 1 else self.values[0]
        return (body + "\n" if body else "") + returned


def parse_ir(text: str):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, memref, scf
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def _width(typ) -> int:
    from xdsl.dialects.builtin import IntegerType, Signedness

    if not isinstance(typ, IntegerType) or typ.signedness.data != Signedness.SIGNLESS or not 1 <= typ.width.data <= 256:
        raise UnsupportedLeanIR("formal scalar domain requires integer widths 1..256")
    return typ.width.data


def _no_flags(op) -> None:
    from xdsl.dialects.arith import IntegerOverflowAttr

    flags = op.properties.get("overflowFlags")
    if flags is not None and flags != IntegerOverflowAttr("none"):
        raise UnsupportedLeanIR("poison-generating arithmetic flags are outside the wrapping model")


def export_scalar(text: str, entry: str, registry: Mapping[str, dict] | None = None) -> ScalarFunction:
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import IntegerAttr

    module = parse_ir(text)
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    if len(functions) != 1 or len(tuple(module.body.block.ops)) != 1 or entry not in functions:
        raise UnsupportedLeanIR("formal scalar input requires exactly one defined entry function")
    selected = functions[entry]
    if (
        module.attributes
        or module.properties
        or selected.attributes
        or set(selected.properties) - {"sym_name", "function_type", "sym_visibility"}
    ):
        raise UnsupportedLeanIR("module or function metadata is outside the modeled domain")
    if len(selected.body.blocks) != 1:
        raise UnsupportedLeanIR("formal scalar input requires a single-block function")
    block = selected.body.block
    inputs = tuple(_width(arg.type) for arg in block.args)
    outputs = tuple(_width(t) for t in selected.function_type.outputs)
    if not outputs:
        raise UnsupportedLeanIR("formal scalar input requires complete integer outputs")
    env = {arg: f"v{i}" for i, arg in enumerate(block.args)}
    bindings: list[str] = []
    returned: tuple[str, ...] | None = None
    binary = {arith.AddiOp: "add", arith.SubiOp: "sub", arith.MuliOp: "mul"}
    for op in block.ops:
        if returned is not None:
            raise UnsupportedLeanIR("operations after return are outside the formal domain")
        if isinstance(op, func.ReturnOp):
            if op.attributes or op.properties:
                raise UnsupportedLeanIR("return metadata is outside the modeled domain")
            returned = tuple(env[value] for value in op.arguments)
            if tuple(_width(v.type) for v in op.arguments) != outputs:
                raise UnsupportedLeanIR("return types do not match complete function outputs")
            continue
        if len(op.results) != 1 or op.regions or op.successors:
            raise UnsupportedLeanIR("formal scalar operations require one result and no regions or successors")
        width = _width(op.results[0].type)
        args = [env[v] for v in op.operands]
        from xdsl.dialects.builtin import UnregisteredOp

        name = op.op_name.data if isinstance(op, UnregisteredOp) else op.name
        allowed_attributes = {"op_name__"} if isinstance(op, UnregisteredOp) else set()
        allowed_properties = (
            {"value"} if isinstance(op, arith.ConstantOp) else {"overflowFlags"} if type(op) in binary else set()
        )
        if set(op.attributes) - allowed_attributes or set(op.properties) - allowed_properties:
            raise UnsupportedLeanIR("unmodeled operation flags, immediates or metadata")
        if isinstance(op, arith.ConstantOp):
            value = op.value
            if not isinstance(value, IntegerAttr):
                raise UnsupportedLeanIR("formal scalar constant must be an integer")
            expr = f"BitVec.ofInt {width} ({value.value.data})"
        elif type(op) in binary:
            _no_flags(op)
            if len(args) != 2 or any(_width(v.type) != width for v in op.operands):
                raise UnsupportedLeanIR("formal binary operation requires two equal-width operands")
            expr = f"Merlin.bv{binary[type(op)].capitalize()} {width} {args[0]} {args[1]}"
        elif isinstance(op, arith.ExtSIOp):
            expr = f"BitVec.signExtend {width} {args[0]}"
        elif isinstance(op, arith.ExtUIOp):
            expr = f"BitVec.setWidth {width} {args[0]}"
        elif isinstance(op, arith.TruncIOp):
            expr = f"BitVec.setWidth {width} {args[0]}"
        else:
            declaration = (registry or {}).get(name)
            if declaration is None or declaration.get("effects") != {"reads": [], "writes": []}:
                raise UnsupportedLeanIR("operation has no admitted pure primitive semantics")
            semantic = declaration.get("semantics") or {}
            primitive = semantic.get("primitive")
            if primitive not in {"integer.add", "integer.sub", "integer.mul"}:
                raise UnsupportedLeanIR("declared primitive is outside the formal scalar model")
            if len(args) != 2 or any(_width(v.type) != width for v in op.operands):
                raise UnsupportedLeanIR("declared integer primitive has an unsupported signature")
            expected = declaration.get("operand_types"), declaration.get("result_types")
            if declaration.get("immediates") != [] or not all(isinstance(types, list) for types in expected):
                raise UnsupportedLeanIR("primitive requires its complete frozen typed signature")
            for declared, actual in zip(expected, (tuple(op.operands), tuple(op.results)), strict=True):
                if len(declared) != len(actual) or any(
                    row.get("kind") != "integer" or row.get("bits") != _width(v.type)
                    for row, v in zip(declared, actual, strict=True)
                ):
                    raise UnsupportedLeanIR("actual operation differs from its frozen instruction signature")
            # Binding order is semantic: an instruction may encode its rhs first.
            operand_names = declaration.get("operand_names")
            mapping = semantic.get("bindings")
            if (
                not isinstance(operand_names, list)
                or len(operand_names) != 2
                or len(set(operand_names)) != 2
                or not isinstance(mapping, dict)
            ):
                raise UnsupportedLeanIR("primitive requires explicit operand names and lhs/rhs bindings")
            if set(mapping) != {"lhs", "rhs"} or set(mapping.values()) != set(operand_names):
                raise UnsupportedLeanIR("primitive operand bindings are incomplete")
            parameters = semantic.get("parameters") or {}
            if parameters != {"width": width, "overflow": "wrap"}:
                raise UnsupportedLeanIR("primitive requires its exact width and wrapping policy")
            lhs, rhs = (args[operand_names.index(mapping[key])] for key in ("lhs", "rhs"))
            expr = f"Merlin.bv{primitive.partition('.')[2].capitalize()} {width} {lhs} {rhs}"
        result_name = f"t{len(bindings)}"
        bindings.append(f"let {result_name} : BitVec {width} := {expr}")
        env[op.results[0]] = result_name
    if returned is None or len(returned) != len(outputs):
        raise UnsupportedLeanIR("formal scalar program has incomplete outputs")
    return ScalarFunction(entry, inputs, outputs, tuple(bindings), returned)


def equivalence_query(before: ScalarFunction, after: ScalarFunction, *, tactic: str = "simp") -> str:
    if before.inputs != after.inputs or before.outputs != after.outputs:
        raise UnsupportedLeanIR("source and target have different complete typed signatures")
    if tactic not in {"simp", "rfl", "bv_decide"}:
        raise ValueError("proof strategy is outside the admitted tactic vocabulary")
    args = " ".join(f"(v{i} : BitVec {w})" for i, w in enumerate(before.inputs))
    proof = (
        "simp"
        if tactic == "simp"
        else "simp only [Merlin.bvAdd_eq, Merlin.bvSub_eq, Merlin.bvMul_eq]; bv_decide"
        if tactic == "bv_decide"
        else tactic
    )
    return (
        "import Merlin.Foundation\nset_option warningAsError true\nset_option linter.unusedSimpArgs false\n"
        f"theorem merlin_transform {args} :\n"
        f"  ({before.term()}) = ({after.term()}) := by\n  {proof}\n"
        "#print axioms merlin_transform\n"
    )
