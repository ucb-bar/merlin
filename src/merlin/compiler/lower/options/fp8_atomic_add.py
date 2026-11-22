"""Source-width FP8 addition using exact casts and checked byte CAS loops.

Reuse the qualified FP8 cast expansions rather than a second encoder. The
portable update stays standard arithmetic. Native helper recognition compares
the complete independently reconstructed body, including pointer arithmetic,
integer old-value transport, retry condition and rounding inside each retry.
"""

from merlin.compiler.lower.options.fp8_narrowing import SUPPORT as _NARROW_SUPPORT
from merlin.compiler.lower.options.fp8_widening import SUPPORT as _WIDEN_SUPPORT

SUPPORT = _NARROW_SUPPORT + _WIDEN_SUPPORT + r'''
_F8A_FORMATS = ('f8E4M3FN','f8E4M3FNUZ','f8E5M2','f8E5M2FNUZ')


def _fp8_add_source(symbol, storage, identity=False, wide=False):
    import json
    if storage not in _F8A_FORMATS:
        raise ValueError('FP8 atomic addition requires a qualified storage format')
    arguments = '' if identity else ', %rhs: f64' if wide else ', %valueword: i8'
    if identity:
        if storage == 'f8E5M2':
            update = """%exponent = arith.constant 124 : i8
          %fraction = arith.constant 3 : i8
          %quiet = arith.constant 2 : i8
          %zero = arith.constant 0 : i8
          %field = arith.andi %current, %exponent : i8
          %all = arith.cmpi eq, %field, %exponent : i8
          %mantissa = arith.andi %current, %fraction : i8
          %nonzero = arith.cmpi ne, %mantissa, %zero : i8
          %nan = arith.andi %all, %nonzero : i1
          %quieted = arith.ori %current, %quiet : i8
          %desired = arith.select %nan, %quieted, %current : i8"""
        else:
            update = '%desired = arith.ori %current, %startzero : i8'
    else:
        update = """%oldvalue = arith.bitcast %current : i8 to STORAGE
          %value = arith.bitcast %valueword : i8 to STORAGE
          %lhs = arith.extf %oldvalue : STORAGE to f64
          %rhs = arith.extf %value : STORAGE to f64
          %sum = arith.addf %lhs, %rhs : f64
          %rounded = arith.truncf %sum : f64 to STORAGE
          %desired = arith.bitcast %rounded : STORAGE to i8"""
        if wide:
            update = '\n'.join(line for line in update.splitlines()
                               if not line.strip().startswith(('%value =', '%rhs =')))
    source = """module {
      func.func private @SYMBOL(%buffer: memref<?xSTORAGE>, %index: indexARGUMENTS) -> i8
          attributes {llvm.linkage = #llvm.linkage<internal>} {
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xSTORAGE> -> index
        %address = arith.addi %base, %index : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %true = arith.constant true
        %startzero = arith.constant 0 : i8
        %start = llvm.load %pointer atomic acquire {alignment = 1 : i64} : !llvm.ptr -> i8
        %old = scf.while (%current = %start) : (i8) -> i8 {
          UPDATE
          %pair = llvm.cmpxchg %pointer, %current, %desired acq_rel acquire : !llvm.ptr, i8
          %observed = llvm.extractvalue %pair[0] : !llvm.struct<(i8, i1)>
          %success = llvm.extractvalue %pair[1] : !llvm.struct<(i8, i1)>
          %retry = arith.xori %success, %true : i1
          scf.condition(%retry) %observed : i8
        } do {
        ^bb0(%next: i8):
          scf.yield %next : i8
        }
        func.return %old : i8
      }
    }""".replace('UPDATE', update).replace('ARGUMENTS', arguments).replace(
        'STORAGE', storage).replace('SYMBOL', json.dumps(symbol))
    # These are ordinary native scalar casts, outside any atomic region.
    # Expand before helper suspension so the whole-body check sees precisely
    # the executable arithmetic later restored after ownership lowering.
    reference = ir.Module.parse(source)
    _fp8_narrow(reference)
    _fp8_widen(reference)
    reference.operation.verify()
    return str(reference)


def _fp8_add_helper_equal(op, symbol, module):
    signature = ir.FunctionType(ir.TypeAttr(op.attributes['function_type']).value)
    if (len(signature.results) != 1 or str(signature.results[0]) != 'i8'
            or len(signature.inputs) not in (2,3)
            or str(signature.inputs[1]) != 'index'
            or len(signature.inputs) == 3 and str(signature.inputs[2]) not in ('i8','f64')):
        return False
    storage = next((name for name in _F8A_FORMATS
                    if str(signature.inputs[0]) == 'memref<?x'+name+'>'), None)
    if storage is None:
        return False
    from torch_mlir.passmanager import PassManager as _f8a_pm
    with module.context:
        reference = ir.Module.parse(_fp8_add_source(symbol, storage, len(signature.inputs)==2,
            len(signature.inputs)==3 and str(signature.inputs[2])=='f64'))
        _f8a_pm.parse('builtin.module(canonicalize,cse)').run(reference.operation)
    expected = tuple(reference.body.operations)[0].operation
    return op.get_asm(print_generic_op_form=True, use_local_scope=True) == expected.get_asm(
        print_generic_op_form=True, use_local_scope=True)
'''
