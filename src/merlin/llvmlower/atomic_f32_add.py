"""Closed standard f32 IEEE/FTZ arithmetic and its exact native CAS loop.

This support extends the existing atomic-word feature. It recognizes actual
scalar bit arithmetic, not source names or provenance, and retains the typed
borrowing boundary through ownership before restoring its complete native body.
"""

SUPPORT = r'''
def _f32_add_match(op):
    if (op.name != 'memref.generic_atomic_rmw' or len(op.results) != 1
            or str(op.results[0].type) != 'f32' or len(op.operands) != 2
            or str(op.operands[0].type) != 'memref<?xf32>' or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1 or str(block.arguments[0].type) != 'f32':
        return None
    used = set()
    def owner(value, name, count):
        if not isinstance(value, ir.OpResult):
            return None
        item = value.owner.operation
        if item.name != name or len(item.operands) != count or len(item.results) != 1 or item.regions:
            return None
        if item.block == block:
            used.add(item)
        return item
    def constant(value, number):
        item = owner(value, 'arith.constant', 0)
        return (item is not None and str(value.type) == 'i32' and 'value' in item.attributes
                and isinstance(item.attributes['value'], ir.IntegerAttr)
                and ir.IntegerAttr(item.attributes['value']).value & 0xffffffff == number)
    def masked(value, original, mask):
        item = owner(value, 'arith.andi', 2)
        return item is not None and ((item.operands[0] == original and constant(item.operands[1], mask))
                                    or (item.operands[1] == original and constant(item.operands[0], mask)))
    def flush_word(value):
        selected = owner(value, 'arith.select', 3)
        if selected is None:
            return None
        original = selected.operands[2]
        equal = owner(selected.operands[0], 'arith.cmpi', 2)
        if (str(original.type) != 'i32' or equal is None or ir.IntegerAttr(equal.attributes['predicate']).value != 0
                or not masked(selected.operands[1], original, 0x80000000)):
            return None
        if not any(constant(equal.operands[i], 0) and masked(equal.operands[1-i], original, 0x7f800000) for i in (0, 1)):
            return None
        return original
    def floating(value):
        cast = owner(value, 'arith.bitcast', 1)
        return cast.operands[0] if cast is not None and str(value.type) == 'i32' and str(cast.operands[0].type) == 'f32' else None
    def flush(value):
        cast = owner(value, 'arith.bitcast', 1)
        if cast is None or str(value.type) != 'f32':
            return None
        word = flush_word(cast.operands[0])
        return floating(word) if word is not None else None
    def quiet_word(value):
        selected = owner(value, 'arith.select', 3)
        if selected is None:
            return None
        original = selected.operands[2]
        updated = owner(selected.operands[1], 'arith.ori', 2)
        test = owner(selected.operands[0], 'arith.andi', 2)
        if updated is None or test is None or not any(updated.operands[i] == original and constant(updated.operands[1-i], 0x00400000) for i in (0, 1)):
            return None
        for i in (0, 1):
            equal = owner(test.operands[i], 'arith.cmpi', 2)
            nonzero = owner(test.operands[1-i], 'arith.cmpi', 2)
            if (equal is not None and nonzero is not None
                    and ir.IntegerAttr(equal.attributes['predicate']).value == 0
                    and ir.IntegerAttr(nonzero.attributes['predicate']).value == 1
                    and any(constant(equal.operands[j], 0x7f800000) and masked(equal.operands[1-j], original, 0x7f800000) for j in (0, 1))
                    and any(constant(nonzero.operands[j], 0) and masked(nonzero.operands[1-j], original, 0x007fffff) for j in (0, 1))):
                return original
        return None
    terminal = tuple(block.operations)[-1].operation
    if terminal.name != 'memref.atomic_yield' or len(terminal.operands) != 1:
        return None
    returned = owner(terminal.operands[0], 'arith.bitcast', 1)
    quieted = quiet_word(returned.operands[0]) if returned is not None else None
    checkpoint = set(used)
    result_word = flush_word(quieted) if quieted is not None else None
    ftz = result_word is not None
    if not ftz:
        used.clear()
        used.update(checkpoint)
        result_word = quieted
    updated = floating(result_word) if result_word is not None else None
    add = owner(updated, 'arith.addf', 2) if updated is not None else None
    if add is None and result_word is not None:
        original = result_word
        for _ in range(len(tuple(block.operations))):
            current = floating(original)
            if current is not None and current == block.arguments[0]:
                if used | {terminal} == {item.operation for item in block.operations}:
                    op.verify()
                    return (*op.operands, None, 32, 'ftz-identity' if ftz else 'ieee-quiet-identity')
                break
            if current is not None:
                inverse = owner(current, 'arith.bitcast', 1)
                if inverse is not None and str(inverse.operands[0].type) == 'i32':
                    original = inverse.operands[0]
                    continue
            original = flush_word(original)
            if original is None:
                break
    if (add is None or add.block != block or 'fastmath' in add.attributes
            and add.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>', context=op.context)):
        return None
    value = None
    for i in (0, 1):
        checkpoint = set(used)
        flushed = flush(add.operands[i]) if ftz else add.operands[i]
        if flushed is not None and flushed == block.arguments[0]:
            value = add.operands[1-i]
            break
        used.clear()
        used.update(checkpoint)
    if (value is None or str(value.type) != 'f32' or value == block.arguments[0]
            or isinstance(value, ir.OpResult) and value.owner.operation.block == block
            or used | {terminal} != {item.operation for item in block.operations}):
        return None
    op.verify()
    return (*op.operands, value, 32, 'fadd-ftz' if ftz else 'fadd-ieee-quiet')


def _f32_add_source(symbol, identity=False, ieee=False):
    import json
    source = """module {
      func.func private @SYMBOL(%buffer: memref<?xf32>, %index: index, %value: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
        %bytes = arith.constant 4 : index
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xf32> -> index
        %offset = arith.muli %index, %bytes : index
        %address = arith.addi %base, %offset : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %exponent = arith.constant 2139095040 : i32
        %sign = arith.constant -2147483648 : i32
        %zero = arith.constant 0 : i32
        %fraction = arith.constant 8388607 : i32
        %quiet = arith.constant 4194304 : i32
        %true = arith.constant true
        %start = llvm.load %pointer atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
        %old = scf.while (%current = %start) : (i32) -> i32 {
          %field = arith.andi %current, %exponent : i32
          %small = arith.cmpi eq, %field, %zero : i32
          %signedzero = arith.andi %current, %sign : i32
          %flushed = arith.select %small, %signedzero, %current : i32
          %floating = arith.bitcast %flushed : i32 to f32
          %sum = arith.addf %floating, %value : f32
          %sumword = arith.bitcast %sum : f32 to i32
          %sumfield = arith.andi %sumword, %exponent : i32
          %sumsmall = arith.cmpi eq, %sumfield, %zero : i32
          %sumsignedzero = arith.andi %sumword, %sign : i32
          %desired = arith.select %sumsmall, %sumsignedzero, %sumword : i32
          %nanfield = arith.andi %desired, %exponent : i32
          %all_exponent = arith.cmpi eq, %nanfield, %exponent : i32
          %mantissa = arith.andi %desired, %fraction : i32
          %nonzero = arith.cmpi ne, %mantissa, %zero : i32
          %nan = arith.andi %all_exponent, %nonzero : i1
          %quietword = arith.ori %desired, %quiet : i32
          %stored = arith.select %nan, %quietword, %desired : i32
          %pair = llvm.cmpxchg %pointer, %current, %stored acq_rel acquire : !llvm.ptr, i32
          %observed = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i1)>
          %success = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i1)>
          %retry = arith.xori %success, %true : i1
          scf.condition(%retry) %observed : i32
        } do {
        ^bb0(%next: i32):
          scf.yield %next : i32
        }
        %result = arith.bitcast %old : i32 to f32
        func.return %result : f32
      }
    }"""
    if ieee:
        omitted = ('%field =', '%small =', '%signedzero =', '%flushed =', '%sumfield =', '%sumsmall =', '%sumsignedzero =', '%desired =')
        source = '\n'.join(line for line in source.splitlines() if not line.strip().startswith(omitted))
        source = source.replace('%flushed', '%current').replace('%desired', '%sumword')
    if identity:
        source = source.replace(', %value: f32', '').replace('%sum = arith.addf %floating, %value : f32', '').replace('%sumword = arith.bitcast %sum : f32 to i32', '').replace('%sumword', '%current' if ieee else '%flushed')
    return source.replace('SYMBOL', json.dumps(symbol))


def _f32_add_helper_equal(op, symbol, module, identity=False, ieee=False):
    from torch_mlir import passmanager as _ftz_pm
    with module.context:
        expected = ir.Module.parse(_f32_add_source(symbol, identity, ieee))
        _ftz_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(expected.operation)
    reference = tuple(expected.body.operations)[0].operation
    return op.get_asm(print_generic_op_form=True, use_local_scope=True) == reference.get_asm(print_generic_op_form=True, use_local_scope=True)
'''
