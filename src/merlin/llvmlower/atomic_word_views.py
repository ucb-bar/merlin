"""Exact integer RMW regions on floating logical storage, default-off lowering.

The exported program keeps standard atomic regions. Native legalization performs
one integer LLVM atomic on the same storage word, preserving NaN/zero bits and
avoiding whole-buffer conversion or an unnecessarily expanded compare/swap loop.
"""

FEATURE = "lower_atomic_word_views"
MARKER = "__merlin_atomic_word_views__"

SUPPORT = r'''
_WORD_OPS = {'arith.addi': 'add', 'arith.andi': 'and', 'arith.ori': 'or', 'arith.xori': 'xor',
             'arith.maxsi': 'max', 'arith.minsi': 'min', 'arith.maxui': 'umax', 'arith.minui': 'umin'}


def _word_match(op):
    if op.name != 'memref.generic_atomic_rmw':
        return None
    if (len(op.operands) != 2 or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        raise ValueError('word atomics require one typed rank-one atomic region')
    floating = str(op.results[0].type)
    if floating not in ('f32', 'f64') or str(op.operands[0].type) != 'memref<?x' + floating + '>':
        raise ValueError('word atomics require identity-layout default-space f32/f64 storage')
    width = int(floating[1:])
    integer = ir.IntegerType.get_signless(width, context=op.context)
    block = op.regions[0].blocks[0]
    operations = tuple(item.operation for item in block.operations)
    if len(block.arguments) != 1 or block.arguments[0].type != op.results[0].type:
        raise ValueError('word atomics require the actual floating current storage word')
    def cast(operation, source, destination):
        return (operation.name == 'arith.bitcast' and tuple(operation.operands) == (source,)
                and len(operation.results) == 1 and operation.results[0].type == destination)
    if len(operations) == 4:
        viewed, updated, stored, terminal = operations
        if (not cast(viewed, block.arguments[0], integer) or updated.name not in _WORD_OPS
                or len(updated.operands) != 2 or len(updated.results) != 1 or updated.results[0].type != integer
                or not cast(stored, updated.results[0], block.arguments[0].type)):
            raise ValueError('word atomic region differs from a closed integer update and storage bitcasts')
        current = viewed.results[0]
        value = updated.operands[1] if updated.operands[0] == current else updated.operands[0] if updated.operands[1] == current else None
        kind = _WORD_OPS[updated.name]
        if (updated.name == 'arith.addi' and 'overflowFlags' in updated.attributes
                and updated.attributes['overflowFlags'] != ir.Attribute.parse('#arith.overflow<none>', context=op.context)):
            raise ValueError('integer word addition must retain modular overflow semantics')
    elif len(operations) == 2:
        stored, terminal = operations
        value = stored.operands[0] if len(stored.operands) == 1 else None
        if value is None or not cast(stored, value, block.arguments[0].type):
            raise ValueError('word exchange requires one outside word and its storage bitcast')
        kind = 'xchg'
    else:
        raise ValueError('word atomics require an exact update/bitcast/yield body')
    if (value is None or value.type != integer or isinstance(value, ir.OpResult) and value.owner.operation.block == block
            or value == block.arguments[0] or terminal.name != 'memref.atomic_yield'
            or tuple(terminal.operands) != tuple(stored.results)):
        raise ValueError('word atomic value must be an original outside-region integer scalar')
    op.verify()
    return (*op.operands, value, width, kind)


def _word_source(symbol, width, kind):
    import json
    if width not in (32, 64) or kind not in (*_WORD_OPS.values(), 'xchg'):
        raise ValueError('unqualified native word atomic operation or width')
    return """module {
      func.func private @SYMBOL(%buffer: memref<?xfWIDTH>, %index: index, %value: iWIDTH) -> fWIDTH attributes {llvm.linkage = #llvm.linkage<internal>} {
        %bytes = arith.constant BYTES : index
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xfWIDTH> -> index
        %offset = arith.muli %index, %bytes : index
        %address = arith.addi %base, %offset : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %old = llvm.atomicrmw KIND %pointer, %value acq_rel : !llvm.ptr, iWIDTH
        %floating = arith.bitcast %old : iWIDTH to fWIDTH
        func.return %floating : fWIDTH
      }
    }""".replace('WIDTH', str(width)).replace('BYTES', str(width // 8)).replace('KIND', '_' + kind if kind in ('and', 'or', 'xor') else kind).replace('SYMBOL', json.dumps(symbol))


def _word_helper_equal(op, symbol, module):
    # This grants synchronous borrowing only. Reconstruct from the actual
    # primitive/type, never the symbol spelling, source tag or saved plan.
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 3 or str(block.arguments[2].type) not in ('i32', 'i64'):
        return False
    atomics = [item for item in _cas_walk(op) if item.name == 'llvm.atomicrmw']
    if len(atomics) != 1 or 'bin_op' not in atomics[0].attributes:
        return False
    from torch_mlir import passmanager as _word_pm
    for kind in (*_WORD_OPS.values(), 'xchg'):
        with module.context:
            reference = ir.Module.parse(_word_source(symbol, int(str(block.arguments[2].type)[1:]), kind))
            primitive = next(item for item in _cas_walk(reference.operation) if item.name == 'llvm.atomicrmw')
            # LLVM's native binding encodes this enum as an integer attribute.
            # Obtain its value from the registered parser, not a duplicated enum.
            if primitive.attributes['bin_op'] != atomics[0].attributes['bin_op']:
                continue
            _word_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(reference.operation)
        expected = tuple(reference.body.operations)[0].operation
        return op.get_asm(print_generic_op_form=True, use_local_scope=True) == expected.get_asm(print_generic_op_form=True, use_local_scope=True)
    return False

'''


def edit_pipeline(passes):
    from .atomic_compare_exchange import edit_pipeline as edit_cas, MARKER as CAS_MARKER
    if CAS_MARKER in passes:
        at = passes.index(CAS_MARKER)
        if (passes.count(CAS_MARKER) != 1 or MARKER in passes or at + 1 == len(passes)
                or not passes[at + 1].startswith("one-shot-bufferize{")):
            raise ValueError("word atomics require one unmodified bufferization seam")
        return [MARKER if name == CAS_MARKER else name for name in passes]
    return [MARKER if name == CAS_MARKER else name for name in edit_cas(passes)]


def ensure_registered():
    from .impr_features import ImprFeature, known, register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE, action_class="PASS", edit_pipeline=edit_pipeline,
            description="Lower exact integer word updates on floating storage and integer CAS to checked native atomics; default off."))
    return FEATURE
