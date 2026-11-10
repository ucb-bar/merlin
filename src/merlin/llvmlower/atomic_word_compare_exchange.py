"""Exact integer compare/exchange on f16/f32/f64 logical storage words.

Integer helper results avoid scalar half floating ABI conversions. Portable
input retains the original floating storage and closed standard atomic body.
"""

SUPPORT = r'''
def _word_cas_match(op):
    if (op.name != 'memref.generic_atomic_rmw' or len(op.operands) != 2 or len(op.results) != 1
            or str(op.results[0].type) not in ('f16', 'f32', 'f64')
            or str(op.operands[0].type) != 'memref<?x' + str(op.results[0].type) + '>'
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return None
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1 or block.arguments[0].type != op.results[0].type:
        return None
    width = int(str(op.results[0].type)[1:])
    integer = ir.IntegerType.get_signless(width, context=op.context)
    operations = tuple(item.operation for item in block.operations)
    constants = {item for item in operations if item.name == 'arith.constant' and not item.operands
                 and len(item.results) == 1 and item.results[0].type == integer
                 and isinstance(item.attributes['value'], ir.IntegerAttr)}
    body = tuple(item for item in operations if item not in constants)
    if len(body) != 5:
        return None
    viewed, equal, selected, stored, terminal = body
    if (viewed.name != 'arith.bitcast' or tuple(viewed.operands) != (block.arguments[0],)
            or len(viewed.results) != 1 or viewed.results[0].type != integer
            or equal.name != 'arith.cmpi' or len(equal.operands) != 2 or len(equal.results) != 1
            or selected.name != 'arith.select' or len(selected.operands) != 3
            or tuple(selected.operands)[:1] != tuple(equal.results)
            or len(selected.results) != 1 or selected.results[0].type != integer
            or stored.name != 'arith.bitcast' or tuple(stored.operands) != tuple(selected.results)
            or len(stored.results) != 1 or stored.results[0].type != block.arguments[0].type
            or terminal.name != 'memref.atomic_yield' or tuple(terminal.operands) != tuple(stored.results)):
        return None
    predicate = ir.IntegerAttr(equal.attributes['predicate']).value
    if predicate not in (0, 1):
        return None
    current = viewed.results[0]
    if selected.operands[2 if predicate == 0 else 1] != current:
        return None
    comparison = equal.operands[1] if equal.operands[0] == current else equal.operands[0] if equal.operands[1] == current else None
    desired = selected.operands[1 if predicate == 0 else 2]
    used = set()
    def outside(value):
        if value is None or value.type != integer or value == current:
            raise ValueError('word CAS requires original outside comparison/value words')
        if isinstance(value, ir.OpResult) and value.owner.operation.block == block:
            constant = value.owner.operation
            if constant not in constants:
                raise ValueError('word CAS captured unaccounted inside-region computation')
            used.add(constant)
            return constant
        return value
    values = (outside(comparison), outside(desired))
    if constants != used:
        raise ValueError('word CAS contains unused or unaccounted constants')
    op.verify()
    return (*op.operands, values, width, 'cas-word')


def _word_cas_source(symbol, width, identity=False):
    import json
    if width not in (16, 32, 64):
        raise ValueError('word CAS requires qualified 16/32/64-bit storage')
    source = """module {
      func.func private @SYMBOL(%buffer: memref<?xfWIDTH>, %index: index, %comparison: iWIDTH, %value: iWIDTH) -> iWIDTH attributes {llvm.linkage = #llvm.linkage<internal>} {
        %bytes = arith.constant BYTES : index
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xfWIDTH> -> index
        %offset = arith.muli %index, %bytes : index
        %address = arith.addi %base, %offset : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %pair = llvm.cmpxchg %pointer, %comparison, %value acq_rel acquire : !llvm.ptr, iWIDTH
        %old = llvm.extractvalue %pair[0] : !llvm.struct<(iWIDTH, i1)>
        func.return %old : iWIDTH
      }
    }"""
    if identity:
        source = source.replace(', %comparison: iWIDTH, %value: iWIDTH', '').replace(
            '%pair = llvm.cmpxchg %pointer, %comparison, %value acq_rel acquire : !llvm.ptr, iWIDTH\n        %old = llvm.extractvalue %pair[0] : !llvm.struct<(iWIDTH, i1)>',
            '%zero = arith.constant 0 : iWIDTH\n        %old = llvm.atomicrmw _or %pointer, %zero acq_rel : !llvm.ptr, iWIDTH')
    return source.replace('WIDTH', str(width)).replace('BYTES', str(width // 8)).replace('SYMBOL', json.dumps(symbol))


def _word_cas_helper_equal(op, symbol, module):
    signature = ir.FunctionType(ir.TypeAttr(op.attributes['function_type']).value)
    if len(signature.results) != 1 or str(signature.results[0]) not in ('i16', 'i32', 'i64'):
        return False
    width = int(str(signature.results[0])[1:])
    if (len(signature.inputs) not in (2, 4) or str(signature.inputs[0]) != 'memref<?xf' + str(width) + '>'
            or str(signature.inputs[1]) != 'index'
            or any(value != signature.results[0] for value in tuple(signature.inputs)[2:])):
        return False
    from torch_mlir import passmanager as _word_cas_pm
    with module.context:
        expected = ir.Module.parse(_word_cas_source(symbol, width, identity=len(signature.inputs) == 2))
        _word_cas_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(expected.operation)
    function = tuple(expected.body.operations)[0].operation
    return op.get_asm(print_generic_op_form=True, use_local_scope=True) == function.get_asm(print_generic_op_form=True, use_local_scope=True)
'''
