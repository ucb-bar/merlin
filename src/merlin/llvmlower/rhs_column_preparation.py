"""Native source for exact immutable RHS preparation, independent of arithmetic."""

SOURCE = r"""
def _rc_preparation_anchor(operation, rhs):
    # Only a single-block function with the exact SSA value defined before the
    # containing root operation admits cross-consumer sharing. Never use pointer
    # identity, alias annotations, equal shapes or equal source names as a join.
    anchor = operation
    while anchor.parent is not None and anchor.parent.name != "func.func":
        anchor = anchor.parent
    function = anchor.parent
    if (
        function is None
        or len(function.regions) != 1
        or len(function.regions[0].blocks) != 1
    ):
        return None
    block = function.regions[0].blocks[0]
    roots = list(block.operations)
    if not any(anchor == root.operation for root in roots):
        return None
    if any(rhs == argument for argument in block.arguments):
        return anchor
    owner = _rc_owner(rhs)
    before = []
    for root in roots:
        if root.operation == anchor:
            break
        before.append(root.operation)
    return (
        anchor if owner is not None and any(owner == root for root in before) else None
    )


def _rc_prepare_columns(rhs, extents, anchor=None):
    from torch_mlir import ir
    import contextlib

    with ir.InsertionPoint(anchor) if anchor is not None else contextlib.nullcontext():
        index = ir.IndexType.get()
        i1 = ir.IntegerType.get_signless(1)
        i32 = ir.IntegerType.get_signless(32)
        f32 = ir.F32Type.get()

        def create(name, operands=(), results=(), attrs=None, regions=0):
            return ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=attrs or {},
                regions=regions,
            )

        def const(n, typ=None):
            typ = typ or index
            return create(
                "arith.constant",
                results=[typ],
                attrs={"value": ir.IntegerAttr.get(typ, n)},
            ).results[0]

        zero, one, yes = const(0), const(1), const(1, i1)
        K, N = const(extents[2]), const(extents[1])

        def cmp(a, b, p):
            return create(
                "arith.cmpi",
                [a, b],
                [i1],
                {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), p)},
            ).results[0]

        def loop(stop, initial, body):
            op = create(
                "scf.for",
                [zero, stop, one, *initial],
                [x.type for x in initial],
                regions=1,
            )
            block = ir.Block.create_at_start(
                op.regions[0], [index, *[x.type for x in initial]]
            )
            with ir.InsertionPoint(block):
                create("scf.yield", body(block.arguments[0], list(block.arguments[1:])))
            return list(op.results)

        def choose(test, types, then, otherwise):
            op = create("scf.if", [test], types, regions=2)
            for n, fn in enumerate((then, otherwise)):
                block = ir.Block.create_at_start(op.regions[n], [])
                with ir.InsertionPoint(block):
                    create("scf.yield", fn())
            return list(op.results)

        def alloc(shape, typ):
            return create(
                "memref.alloca",
                results=[ir.MemRefType.get(shape, typ)],
                attrs={"operandSegmentSizes": ir.DenseI32ArrayAttr.get([0, 0])},
            ).results[0]

        representatives, mapping, hashes = (
            alloc([extents[1]], index),
            alloc([extents[1]], index),
            alloc([extents[1]], i32),
        )

        def load(mem, pos):
            return create(
                "memref.load", [mem, pos], [ir.MemRefType(mem.type).element_type]
            ).results[0]

        def store(value, mem, pos):
            create("memref.store", [value, mem, pos])

        def word(k, n):
            return create(
                "arith.bitcast",
                [create("tensor.extract", [rhs, k, n], [f32]).results[0]],
                [i32],
            ).results[0]

        offset, prime = const(2166136261, i32), const(16777619, i32)

        def column(n, state):
            count = state[0]

            def hash_k(k, state):
                mixed = create("arith.xori", [state[0], word(k, n)], [i32]).results[0]
                return [create("arith.muli", [mixed, prime], [i32]).results[0]]

            signature = loop(K, [offset], hash_k)[0]

            def probe(slot, state):
                prior = load(representatives, slot)

                def exact():
                    def equal_k(k, state):
                        return [
                            create(
                                "arith.andi",
                                [state[0], cmp(word(k, n), word(k, prior), 0)],
                                [i1],
                            ).results[0]
                        ]

                    return loop(K, [yes], equal_k)

                equal = choose(
                    cmp(signature, load(hashes, slot), 0),
                    [i1],
                    exact,
                    lambda: [const(0, i1)],
                )[0]
                return [
                    create("arith.select", [equal, prior, state[0]], [index]).results[0]
                ]

            representative = loop(count, [n], probe)[0]
            store(representative, mapping, n)

            def append():
                store(n, representatives, count)
                store(signature, hashes, count)
                return [create("arith.addi", [count, one], [index]).results[0]]

            return choose(cmp(representative, n, 0), [index], append, lambda: [count])

        count = loop(N, [zero], column)[0]
        packed = alloc([extents[2], extents[1]], f32)

        def pack():
            def pack_column(slot, state):
                n = load(representatives, slot)

                def pack_k(k, state):
                    x = create("tensor.extract", [rhs, k, n], [f32]).results[0]
                    create("memref.store", [x, packed, k, slot])
                    return []

                loop(K, [], pack_k)
                return []

            return loop(count, [], pack_column)

        choose(cmp(count, N, 6), [], pack, lambda: [])
        return count, representatives, mapping, packed
"""
