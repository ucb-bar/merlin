"""Typed scalar proof adapter for bounded-RNE tensor map scheduling.

This module recognizes arithmetic, not provenance labels. It shares the complete
numeric graph proof with host LLVM legalization. The optional tensor scheduler
additionally proves index maps, extents, immutable tensor dependencies and all
tail accesses before grouping memory operations.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import SSAValue

from .late_quant_rne import _identity, _Instruction, _match


@dataclass(frozen=True)
class ScalarRneProof:
    raw_input: SSAValue
    result: SSAValue
    integer_bits: int
    bounds: tuple[int, int]


def prove_scalar_bounded_rne(block) -> ScalarRneProof | None:
    """Accept the complete typed arithmetic ties-even graph, else refuse.

    Unknown operations, nested regions, fast FP and integer overflow promises
    refuse the entire body. Constants are interpreted from typed attributes.
    No model, symbol, source-node or transform attribute selects acceptance.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType, f32

    ops = list(block.ops)
    if not ops or ops[-1].name not in ("linalg.yield", "func.return") or len(ops[-1].operands) != 1:
        return None
    values = {v: f"%arg{i}" for i, v in enumerate(block.args)}
    reverse = {_identity(n): v for v, n in values.items()}
    definitions = {}

    def dtype(t):
        return "float" if t == f32 else str(t) if isinstance(t, IntegerType) else None

    binary = {"arith.addi": "add", "arith.andi": "and", "arith.ori": "or", "arith.subf": "fsub", "arith.mulf": "fmul"}
    for serial, op in enumerate(ops[:-1]):
        if op.regions or len(op.results) != 1 or any(x not in values for x in op.operands):
            return None
        for key in ("fastmath", "overflowFlags"):
            flag = op.properties.get(key)
            if flag is not None and (
                not isinstance(flag, (arith.FastMathFlagsAttr, arith.IntegerOverflowAttr)) or flag.data
            ):
                return None
        result = op.results[0]
        if isinstance(op, arith.ConstantOp):
            value = op.value
            if isinstance(value, (FloatAttr, IntegerAttr)):
                values[result] = repr(value.value.data)
                continue
            return None
        name = f"%v{serial}"
        operands = tuple(values[v] for v in op.operands)
        predicate = callee = output_type = ""
        if op.name in binary and len(operands) == 2:
            opcode, typ = binary[op.name], dtype(result.type)
        elif op.name == "arith.negf" and len(operands) == 1:
            opcode, typ = "fneg", dtype(result.type)
        elif op.name in ("arith.minimumf", "arith.maximumf") and len(operands) == 2:
            opcode, typ = "call", dtype(result.type)
            callee = "llvm.minimum.f32" if op.name == "arith.minimumf" else "llvm.maximum.f32"
        elif op.name in ("arith.fptosi", "arith.sitofp") and len(operands) == 1:
            opcode = "fptosi" if op.name == "arith.fptosi" else "sitofp"
            typ, output_type = dtype(op.operands[0].type), dtype(result.type)
        elif op.name in ("arith.cmpf", "arith.cmpi") and len(operands) == 2:
            opcode = "fcmp" if op.name == "arith.cmpf" else "icmp"
            predicates = {1: "oeq", 2: "ogt", 4: "olt"} if opcode == "fcmp" else {1: "ne"}
            predicate = predicates.get(op.predicate.value.data)
            if predicate is None:
                return None
            typ = dtype(op.operands[0].type)
        elif op.name == "arith.select" and len(operands) == 3:
            opcode, typ = "select", dtype(result.type)
        else:
            return None
        if typ is None or output_type is None:
            return None
        values[result] = name
        reverse[_identity(name)] = result
        definitions[_identity(name)] = _Instruction(
            name, opcode, typ, operands, serial, serial + 1, predicate, callee, output_type
        )
    yielded = ops[-1].operands[0]
    token = values.get(yielded, "")
    final = definitions.get(_identity(token))
    proof = _match(final, definitions) if final is not None else None
    if proof is None:
        return None
    raw = reverse.get(_identity(proof["raw_input"]))
    if raw is None:
        return None
    return ScalarRneProof(raw, yielded, int(proof["integer_dtype"][1:]), tuple(proof["bounds"]))


def schedule_bounded_rne_maps(module, *, lanes=4, symbol_prefix="bounded_rne_packet"):
    """Stripmine proved pure tensor maps; retain original arithmetic in helpers.

    This explicit transform changes no numeric contract. Inputs are immutable
    tensor values and the output is carried through SCF; upstream bufferization
    remains responsible for any copy required by live aliases. Full packets and
    static tails are separate, so no load or store relies on masking. Affine
    permutation inputs can be traced through one or more tensor transposes.
    """
    from xdsl.dialects import arith, func, scf, tensor
    from xdsl.dialects.builtin import IndexType, NoneAttr, StringAttr, TensorType
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(lanes) is not int or not 1 <= lanes <= 8:
        raise ValueError("packet width must be an integer in [1,8]")
    if (
        not isinstance(symbol_prefix, str)
        or not symbol_prefix
        or symbol_prefix[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in symbol_prefix)
    ):
        raise ValueError("explicit valid symbol prefix required")
    module.verify()
    # Do not turn a dead pure tensor branch into opaque helper calls. Unknown
    # effects, terminators and cycles conservatively count as observable uses.
    from xdsl.transforms.dead_code_elimination import would_be_trivially_dead

    def has_observable_use(value):
        pending = [use.operation for use in value.uses]
        seen = set()
        while pending:
            user = pending.pop()
            if user in seen:
                continue
            seen.add(user)
            if not would_be_trivially_dead(user):
                return True
            pending.extend(use.operation for result in user.results for use in result.uses)
        return False

    plans = []
    for op in list(module.walk()):
        if not isinstance(op, linalg.GenericOp) or len(op.outputs) != 1 or len(op.results) != 1:
            continue
        if not has_observable_use(op.results[0]):
            continue
        output = op.outputs[0]
        typ = output.type
        if not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr):
            continue
        shape = typ.get_shape()
        rank = len(shape)
        if not rank or any(n <= 0 for n in shape) or op.results[0].type != typ:
            continue
        if len(op.body.blocks) != 1 or op.body.block.args[-1].uses:
            continue
        if tuple(x.data for x in op.iterator_types) != (linalg.IteratorType.PARALLEL,) * rank:
            continue
        maps = tuple(x.data for x in op.indexing_maps)
        if len(maps) != len(op.inputs) + 1 or maps[-1] != AffineMap.identity(rank):
            continue
        proof = prove_scalar_bounded_rne(op.body.block)
        if proof is None:
            continue
        sources = []
        legal = True
        for value, amap in zip(op.inputs, maps[:-1]):
            if amap.num_dims != rank or amap.num_symbols or any(not isinstance(e, AffineDimExpr) for e in amap.results):
                legal = False
                break
            dims = tuple(e.position for e in amap.results)
            if len(set(dims)) != len(dims):
                legal = False
                break
            if isinstance(value.type, TensorType):
                if not isinstance(value.type.encoding, NoneAttr) or value.type.get_shape() != tuple(
                    shape[d] for d in dims
                ):
                    legal = False
                    break
                while isinstance(value.owner, linalg.TransposeOp):
                    transpose = value.owner
                    perm = tuple(transpose.permutation.get_values())
                    if sorted(perm) != list(range(len(dims))) or len(transpose.inputs) != 1:
                        legal = False
                        break
                    original = transpose.inputs[0]
                    if (
                        not isinstance(original.type, TensorType)
                        or not isinstance(original.type.encoding, NoneAttr)
                        or original.type.get_shape()
                        != tuple(value.type.get_shape()[perm.index(d)] for d in range(len(dims)))
                    ):
                        legal = False
                        break
                    dims = tuple(dims[perm.index(d)] for d in range(len(dims)))
                    value = original
                if not legal:
                    break
            elif dims:
                legal = False
                break
            sources.append((value, dims))
        if not legal or not any(dims for _, dims in sources):
            continue
        # Follow a contiguous source dimension rather than guessing a model layout.
        axis = next(dims[-1] for _, dims in sources if dims)
        plans.append((op, proof, shape, sources, axis))
    names = {op.sym_name.data for op in module.walk() if isinstance(op, func.FuncOp)}
    reserved = {
        attr.data
        for op in module.walk()
        if isinstance(attr := op.properties.get("sym_name", op.attributes.get("sym_name")), StringAttr)
    }
    reports = []
    for op, proof, shape, sources, axis in plans:
        output = op.outputs[0]
        typ = output.type
        body = op.body.block
        rank = len(shape)
        widths = {lanes}
        if shape[axis] % lanes:
            widths.add(shape[axis] % lanes)
        helpers = {}
        for width in sorted(widths):
            serial = len(names)
            name = f"{symbol_prefix}_{serial}_{width}"
            while name in names or name in reserved:
                serial += 1
                name = f"{symbol_prefix}_{serial}_{width}"
            names.add(name)
            block = Block(arg_types=[arg.type for _ in range(width) for arg in body.args[:-1]])
            results = []
            for lane in range(width):
                mapping = dict(zip(body.args[:-1], block.args[lane * len(sources) : (lane + 1) * len(sources)]))
                for old in list(body.ops)[:-1]:
                    block.add_op(old.clone(mapping))
                results.append(mapping[body.last_op.operands[0]])
            block.add_op(func.ReturnOp(*results))
            helper = func.FuncOp(
                name, ([x.type for x in block.args], [proof.result.type] * width), Region(block), visibility="private"
            )
            module.body.block.add_op(helper)
            helpers[width] = name
        constants = {
            n: arith.ConstantOp.from_int_and_width(n, IndexType())
            for n in {0, 1, lanes, *shape, *range(lanes), (shape[axis] // lanes) * lanes}
        }

        def c(n):
            return constants[n].result

        root = Block()
        uniform = {}
        for i, (source, dims) in enumerate(sources):
            if not dims and isinstance(source.type, TensorType):
                extract = tensor.ExtractOp(source, [], source.type.element_type)
                root.add_op(extract)
                uniform[i] = extract.result
            elif not dims:
                uniform[i] = source

        def packet(parent, current, indices, width):
            all_values = []
            all_indices = []
            for lane in range(width):
                ix = list(indices)
                if lane:
                    add = arith.AddiOp(ix[axis], c(lane))
                    parent.add_op(add)
                    ix[axis] = add.result
                all_indices.append(ix)
                for i, (source, dims) in enumerate(sources):
                    if i in uniform:
                        all_values.append(uniform[i])
                        continue
                    extract = tensor.ExtractOp(source, [ix[d] for d in dims], source.type.element_type)
                    parent.add_op(extract)
                    all_values.append(extract.result)
            call = func.CallOp(helpers[width], all_values, [proof.result.type] * width)
            parent.add_op(call)
            for value, ix in zip(call.results, all_indices):
                insert = tensor.InsertOp(value, current, ix)
                parent.add_op(insert)
                current = insert.result
            return current

        order = [d for d in range(rank) if d != axis]

        def loops(parent, depth, current, indices):
            if depth < len(order):
                d = order[depth]
                bb = Block(arg_types=[IndexType(), typ])
                ix = list(indices)
                ix[d] = bb.args[0]
                value = loops(bb, depth + 1, bb.args[1], ix)
                bb.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(shape[d]), c(1), [current], Region(bb))
                parent.add_op(loop)
                return loop.results[0]
            full = (shape[axis] // lanes) * lanes
            if full:
                bb = Block(arg_types=[IndexType(), typ])
                ix = list(indices)
                ix[axis] = bb.args[0]
                value = packet(bb, bb.args[1], ix, lanes)
                bb.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(full), c(lanes), [current], Region(bb))
                parent.add_op(loop)
                current = loop.results[0]
            tail = shape[axis] - full
            if tail:
                ix = list(indices)
                ix[axis] = c(full)
                current = packet(parent, current, ix, tail)
            return current

        value = loops(root, 0, output, [c(0)] * rank)
        detached = [*constants.values(), *[root.detach_op(x) for x in list(root.ops)]]
        parent = op.parent_block()
        parent.insert_ops_before(detached, op)
        op.results[0].replace_all_uses_with(value)
        parent.erase_op(op)
        reports.append(
            {
                "shape": list(shape),
                "packet_axis": axis,
                "lanes": lanes,
                "tail": shape[axis] % lanes,
                "integer_bits": proof.integer_bits,
                "bounds": list(proof.bounds),
                "helpers": helpers,
                "numeric_contract": "original scalar body cloned; no reassociation",
                "memory_contract": (
                    "immutable input tensors, carried output tensor; upstream bufferization owns alias copies"
                ),
            }
        )
    module.verify()
    return reports
