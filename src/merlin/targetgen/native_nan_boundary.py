"""Prove a storage epilogue to be identity outside NaN representations.

The complete scalar body is interpreted as bit vectors. No numerical floating
operation, unknown carrier, literal lookup or additional effect is admitted.
The device transports the word; the retained epilogue changes only NaN bits.
"""

import hashlib


def prove(op):
    try:
        import z3
    except ImportError:
        return None
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType
    from xdsl.ir.affine import AffineMap

    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    if op.name != "linalg.generic" or len(op.results) != 1 or len(op.outputs) != 1:
        return None
    element = op.results[0].type.element_type
    if str(element) not in FORMAT or any(it.data.value != "parallel" for it in op.get_iterator_types()):
        return None
    identity = AffineMap.identity(len(op.results[0].type.get_shape()))
    if any(m.data != identity for m in op.get_indexing_maps()) or op.body.block.args[-1].uses:
        return None
    runtime = [i for i, v in enumerate(op.inputs) if _constant_of(v) is None]
    if len(runtime) != 1 or op.inputs[runtime[0]].type != op.results[0].type:
        return None
    fraction, exponent, _ = FORMAT[str(element)]
    width = 1 + fraction + exponent
    original = z3.BitVec("original_storage_word", width)

    def literal(value, typ):
        if isinstance(typ, IntegerType):
            return z3.BitVecVal(int(value), typ.width.data)
        if str(typ) not in FORMAT:
            raise ValueError("unknown literal storage format")
        return z3.BitVecVal(int.from_bytes(typ.pack([float(value)]), "little"), sum(FORMAT[str(typ)][:2]) + 1)

    def nan(value):
        if value.size() != width:
            raise ValueError("NaN classifier has a different storage format")
        return z3.And(
            z3.Extract(width - 2, fraction, value) == (1 << exponent) - 1, z3.Extract(fraction - 1, 0, value) != 0
        )

    env = {}
    for i, v in enumerate(op.inputs):
        if i == runtime[0]:
            env[op.body.block.args[i]] = original
        else:
            try:
                env[op.body.block.args[i]] = literal(_constant_of(v), v.type.element_type)
            except (TypeError, ValueError, OverflowError):
                return None
    result = None
    classified = False
    try:
        for child in op.body.block.ops:
            if child.name == "linalg.yield":
                result = env[child.operands[0]]
                break
            if child.regions or len(child.results) != 1:
                return None
            if child.name == "arith.constant":
                attr = child.value
                if not isinstance(attr, (FloatAttr, IntegerAttr)):
                    return None
                value = literal(attr.value.data, child.results[0].type)
            else:
                args = [env[v] for v in child.operands]
                if child.name == "arith.bitcast" and child.operands[0].type != child.results[0].type:
                    typ = child.results[0].type
                    n = typ.width.data if isinstance(typ, IntegerType) else sum(FORMAT[str(typ)][:2]) + 1
                    if args[0].size() != n:
                        return None
                    value = args[0]
                elif child.name in {"arith.ori", "arith.andi", "arith.xori"}:
                    value = {
                        "arith.ori": lambda: args[0] | args[1],
                        "arith.andi": lambda: args[0] & args[1],
                        "arith.xori": lambda: args[0] ^ args[1],
                    }[child.name]()
                elif (
                    child.name == "arith.cmpf"
                    and child.operands[0] is child.operands[1]
                    and child.predicate.value.data in {7, 14}
                    and not child.fastmath.data
                ):
                    value = nan(args[0])
                    if child.predicate.value.data == 7:
                        value = z3.Not(value)
                    classified = True
                elif child.name == "arith.select" and z3.is_bool(args[0]):
                    value = z3.If(*args)
                else:
                    return None
            env[child.results[0]] = value
    except (KeyError, TypeError, AttributeError, ValueError, z3.Z3Exception):
        return None
    if result is None or not classified or z3.is_bool(result) or result.size() != width:
        return None
    solver = z3.Solver()
    solver.set(timeout=2000)
    # This catches a changed finite value, a lost signed zero, NaN-to-number,
    # or a newly introduced signaling NaN for every possible input word.
    invalid = z3.Or(
        z3.And(z3.Not(nan(original)), result != original),
        z3.And(
            nan(original),
            z3.Or(z3.Not(nan(result)), z3.And(result != original, z3.Extract(fraction - 1, fraction - 1, result) == 0)),
        ),
    )
    solver.add(invalid)
    if solver.check() != z3.unsat:
        return None
    return dict(
        method="complete_storage_bitvector_nan_boundary",
        input_domain="all_storage_bit_patterns",
        non_nan_storage="identity_including_signed_zeros_and_infinities",
        runtime_input=runtime[0],
        nan_storage="original_or_quiet_nan_only",
        runtime_input_observations=False,
        scalar_source_sha256=hashlib.sha256(str(op).encode()).hexdigest(),
    )


def expose_device_transport(module):
    """Make every proved word boundary consume real native byte movement."""
    from merlin.llvmlower._ir import copy

    proofs = []
    for op in list(module.walk()):
        proof = prove(op)
        if proof is None:
            continue
        operations = []
        operands = list(op.operands)
        operands[proof["runtime_input"]] = copy(op.inputs[proof["runtime_input"]], operations)
        op.parent_block().insert_ops_before(operations, op)
        op.operands = operands
        proofs.append(proof)
    return proofs
