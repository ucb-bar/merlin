"""Closed effect check for capture-bound source specialization proofs."""


def single_forward(directory, report):
    """(record, original bytes, module, forward) of a one-record capsule with one forward function, else None."""
    from merlin.frontends.linalg_mlir import parse_mlir_file

    records = [r for r in report.get("cases", ()) if r.get("status") == "bundled"]
    if len(records) != 1:
        return None
    original = (directory / "model.mlir").read_bytes()
    module = parse_mlir_file(directory / "model.mlir")
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1 or functions[0].sym_name.data != "forward":
        return None
    return records[0], original, module, functions[0]


def pure_tensor_source(function):
    from xdsl.dialects.builtin import IntegerType, TensorType
    from xdsl.traits import Pure

    from merlin.llvmlower._ir import input_literal as literal
    from merlin.targetgen.closed_scalar import predicate_holds

    def true_assert(value):
        known = literal(value)
        if known is not None:
            return bool(known)
        owner = value.owner
        if getattr(owner, "name", None) not in {"arith.cmpi", "arith.cmpf"}:
            return False
        a, b = [literal(v) for v in owner.operands]
        if a is None or b is None:
            return False
        floating = owner.name == "arith.cmpf"
        if floating:
            import math

            if owner.fastmath.data:
                return False
            return predicate_holds(
                owner.predicate.value.data, (a > b) - (a < b), math.isnan(a) or math.isnan(b), floating=True
            )
        predicate = owner.predicate.value.data
        typ = owner.operands[0].type
        if isinstance(typ, IntegerType):
            width = typ.width.data
            a, b = int(a) % (1 << width), int(b) % (1 << width)
            if predicate in {2, 3, 4, 5}:
                a = a - (1 << width) if a >= 1 << (width - 1) else a
                b = b - (1 << width) if b >= 1 << (width - 1) else b
        elif not (0 <= a < 2**31 and 0 <= b < 2**31):
            return False
        return predicate_holds(predicate, (a > b) - (a < b))

    for op in function.body.block.walk():
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if name == "cf.assert":
            if len(op.operands) == 1 and true_assert(op.operands[0]):
                continue
            return False
        if name in {"func.return", "linalg.yield", "scf.yield"}:
            continue
        # Registered tensor operations are functional values, including their
        # private allocations. Unknown names do not acquire this permission.
        if type(op).__module__ == "xdsl.dialects.tensor":
            if any("memref" in str(v.type) for v in (*op.operands, *op.results)):
                return False
            continue
        if name == "linalg.index" and type(op).__module__ == "xdsl.dialects.linalg.ops":
            continue
        if name.startswith("linalg.") and op.regions:
            if all(isinstance(v.type, TensorType) for v in (*op.operands, *op.results)):
                continue
            return False
        if name in {"scf.if", "scf.for"}:
            if any("memref" in str(v.type) for v in (*op.operands, *op.results)):
                return False
            continue  # Every nested operation is checked by the same walk.
        implementation = type(op).__module__
        if name in {"arith.divui", "arith.remui"} and implementation == "xdsl.dialects.arith":
            divisor = literal(op.operands[1])
            if isinstance(divisor, int) and divisor > 0:
                continue  # The only unsigned division effect is a zero divisor.
        if implementation.startswith(
            ("xdsl.dialects.arith", "xdsl.dialects.math", "xdsl.dialects.tensor", "xdsl.dialects.linalg")
        ) and op.has_trait(Pure):
            continue
        # This standard scalar primitive is not registered in this xDSL build.
        # Its closed type/effect signature does not admit an arbitrary carrier.
        if (
            name == "arith.remf"
            and len(op.operands) == 2
            and len(op.results) == 1
            and not op.regions
            and not op.successors
        ):
            types = [str(v.type) for v in (*op.operands, *op.results)]
            if types[0] in {"f16", "bf16", "f32", "f64"} and len(set(types)) == 1:
                continue
        return False
    return True
