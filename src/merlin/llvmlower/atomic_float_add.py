"""Closed IEEE f16/BF16/f32/f64 and FTZ f32 arithmetic with native CAS loops.

This support extends the existing atomic-word feature. It recognizes actual
scalar bit arithmetic, not source names or provenance, and retains the typed
borrowing boundary through ownership before restoring its complete native body.
"""

SUPPORT = r'''
def _float_add_match(op):
    if (op.name != 'memref.generic_atomic_rmw' or len(op.results) != 1
            or str(op.results[0].type) not in ('f16', 'bf16', 'f32', 'f64') or len(op.operands) != 2
            or str(op.operands[0].type) != 'memref<?x' + str(op.results[0].type) + '>' or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    block = op.regions[0].blocks[0]
    floating_type = str(op.results[0].type)
    width = 16 if floating_type == 'bf16' else int(floating_type[1:])
    integer_type = 'i' + str(width)
    exponent_mask, fraction_bits = {'f16': (0x7c00, 10), 'bf16': (0x7f80, 7),
        'f32': (0x7f800000, 23), 'f64': (0x7ff0000000000000, 52)}[floating_type]
    if len(block.arguments) != 1 or str(block.arguments[0].type) != floating_type:
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
        return (item is not None and str(value.type) == integer_type and 'value' in item.attributes
                and isinstance(item.attributes['value'], ir.IntegerAttr)
                and ir.IntegerAttr(item.attributes['value']).value & ((1 << width) - 1) == number)
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
        if (str(original.type) != integer_type or equal is None or ir.IntegerAttr(equal.attributes['predicate']).value != 0
                or not masked(selected.operands[1], original, 1 << (width - 1))):
            return None
        if not any(constant(equal.operands[i], 0) and masked(equal.operands[1-i], original, exponent_mask) for i in (0, 1)):
            return None
        return original
    def floating(value):
        cast = owner(value, 'arith.bitcast', 1)
        return cast.operands[0] if cast is not None and str(value.type) == integer_type and str(cast.operands[0].type) == floating_type else None
    def flush(value):
        cast = owner(value, 'arith.bitcast', 1)
        if cast is None or str(value.type) != floating_type:
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
        if updated is None or test is None or not any(updated.operands[i] == original and constant(updated.operands[1-i], 1 << (fraction_bits - 1)) for i in (0, 1)):
            return None
        for i in (0, 1):
            equal = owner(test.operands[i], 'arith.cmpi', 2)
            nonzero = owner(test.operands[1-i], 'arith.cmpi', 2)
            if (equal is not None and nonzero is not None
                    and ir.IntegerAttr(equal.attributes['predicate']).value == 0
                    and ir.IntegerAttr(nonzero.attributes['predicate']).value == 1
                    and any(constant(equal.operands[j], exponent_mask) and masked(equal.operands[1-j], original, exponent_mask) for j in (0, 1))
                    and any(constant(nonzero.operands[j], 0) and masked(nonzero.operands[1-j], original, (1 << fraction_bits) - 1) for j in (0, 1))):
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
    if ftz and width != 32:
        return None
    if not ftz:
        used.clear()
        used.update(checkpoint)
        result_word = quieted
    updated = floating(result_word) if result_word is not None else None
    if floating_type == 'bf16' and updated is not None:
        narrowed = owner(updated, 'arith.truncf', 1)
        if narrowed is not None and (str(narrowed.operands[0].type) != 'f32'
                or 'roundingmode' in narrowed.attributes and ir.IntegerAttr(narrowed.attributes['roundingmode']).value != 0
                or 'fastmath' in narrowed.attributes and narrowed.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>', context=op.context)):
            return None
        if narrowed is not None:
            updated = narrowed.operands[0]
    add = owner(updated, 'arith.addf', 2) if updated is not None else None
    if add is None and result_word is not None:
        original = result_word
        for _ in range(len(tuple(block.operations))):
            current = floating(original)
            if current is not None and current == block.arguments[0]:
                if used | {terminal} == {item.operation for item in block.operations}:
                    op.verify()
                    kind = 'ieee-quiet-identity-' + floating_type if width == 16 else 'ftz-identity' if ftz else 'ieee-quiet-identity'
                    return (*op.operands, None, width, kind)
                break
            if current is not None:
                inverse = owner(current, 'arith.bitcast', 1)
                if inverse is not None and str(inverse.operands[0].type) == integer_type:
                    original = inverse.operands[0]
                    continue
            original = flush_word(original)
            if original is None:
                break
    if (add is None or add.block != block
            or 'roundingmode' in add.attributes and ir.IntegerAttr(add.attributes['roundingmode']).value != 0
            or 'fastmath' in add.attributes
            and add.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>', context=op.context)):
        return None
    value = None
    def operand(value):
        if floating_type != 'bf16':
            return value
        extended = owner(value, 'arith.extf', 1)
        if (extended is None or str(value.type) != 'f32' or str(extended.operands[0].type) != 'bf16'
                or 'fastmath' in extended.attributes and extended.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>', context=op.context)):
            return None
        return extended.operands[0]
    for i in (0, 1):
        checkpoint = set(used)
        flushed = flush(add.operands[i]) if ftz else operand(add.operands[i])
        if flushed is not None and flushed == block.arguments[0]:
            value = operand(add.operands[1-i])
            break
        used.clear()
        used.update(checkpoint)
    if (value is None or str(value.type) != floating_type or value == block.arguments[0]
            or isinstance(value, ir.OpResult) and value.owner.operation.block == block
            or used | {terminal} != {item.operation for item in block.operations}):
        return None
    op.verify()
    kind = 'fadd-ieee-' + floating_type if width == 16 else 'fadd-ftz' if ftz else 'fadd-ieee-quiet'
    return (*op.operands, value, width, kind)


def _float_add_source(symbol, identity=False, ieee=False, width=32):
    import json
    if width not in (32, 64) or width == 64 and not ieee:
        raise ValueError('floating atomic helpers require IEEE f32/f64 or FTZ f32')
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
    if width == 64:
        source = source.replace('f32', 'f64').replace('i32', 'i64')
        for old, new in (('2139095040', '9218868437227405312'), ('-2147483648', '-9223372036854775808'),
                         ('8388607', '4503599627370495'), ('4194304', '2251799813685248'),
                         ('arith.constant 4 : index', 'arith.constant 8 : index'), ('alignment = 4', 'alignment = 8')):
            source = source.replace(old, new)
    return source.replace('SYMBOL', json.dumps(symbol))


def _float_add_helper_equal(op, symbol, module, identity=False, ieee=False, width=32):
    from torch_mlir import passmanager as _ftz_pm
    with module.context:
        expected = ir.Module.parse(_float_add_source(symbol, identity, ieee, width))
        _ftz_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(expected.operation)
    reference = tuple(expected.body.operations)[0].operation
    return op.get_asm(print_generic_op_form=True, use_local_scope=True) == reference.get_asm(print_generic_op_form=True, use_local_scope=True)


def _half_add_source(symbol, floating_type, identity=False):
    if floating_type not in ('f16', 'bf16'):
        raise ValueError('half atomic addition requires f16 or bf16 storage')
    source = _float_add_source(symbol, identity=identity, ieee=True)
    source = source.replace('f32', floating_type).replace('i32', 'i16')
    exponent, fraction, quiet = (31744, 1023, 512) if floating_type == 'f16' else (32640, 127, 64)
    for old, new in (('2139095040', str(exponent)), ('-2147483648', '-32768'),
                     ('8388607', str(fraction)), ('4194304', str(quiet)),
                     ('arith.constant 4 : index', 'arith.constant 2 : index'), ('alignment = 4', 'alignment = 2')):
        source = source.replace(old, new)
    source = source.replace(') -> ' + floating_type + ' attributes', ') -> i16 attributes')
    source = source.replace('%result = arith.bitcast %old : i16 to ' + floating_type, '')
    source = source.replace('func.return %result : ' + floating_type, 'func.return %old : i16')
    if not identity:
        source = source.replace('%value: ' + floating_type, '%valueword: i16')
        source = source.replace('%bytes =', '%value = arith.bitcast %valueword : i16 to ' + floating_type + '\n        %bytes =')
        if floating_type == 'bf16':
            source = source.replace('%sum = arith.addf %floating, %value : bf16', """%wide_old = arith.extf %floating : bf16 to f32
          %wide_value = arith.extf %value : bf16 to f32
          %wide_sum = arith.addf %wide_old, %wide_value : f32
          %sum = arith.truncf %wide_sum : f32 to bf16""")
    return source


def _half_add_helper_equal(op, symbol, module):
    signature = ir.FunctionType(ir.TypeAttr(op.attributes['function_type']).value)
    if (len(signature.results) != 1 or str(signature.results[0]) != 'i16'
            or len(signature.inputs) not in (2, 3)
            or str(signature.inputs[0]) not in ('memref<?xf16>', 'memref<?xbf16>')
            or str(signature.inputs[1]) != 'index'
            or len(signature.inputs) == 3 and str(signature.inputs[2]) != 'i16'):
        return False
    from torch_mlir import passmanager as _half_pm
    floating_type = 'bf16' if str(signature.inputs[0]) == 'memref<?xbf16>' else 'f16'
    with module.context:
        expected = ir.Module.parse(_half_add_source(symbol, floating_type, identity=len(signature.inputs) == 2))
        _half_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(expected.operation)
    reference = tuple(expected.body.operations)[0].operation
    return op.get_asm(print_generic_op_form=True, use_local_scope=True) == reference.get_asm(print_generic_op_form=True, use_local_scope=True)
'''
