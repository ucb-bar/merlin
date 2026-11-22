"""Exact atomic regions on floating logical storage, default-off lowering.

The exported program keeps standard atomic regions. Native legalization performs
one integer LLVM RMW for word updates, preserving NaN/zero bits. Closed floating
addition uses a checked atomic-load/CAS retry loop retaining the scalar numeric
body. Both paths avoid whole-buffer conversion and keep borrowing/lifetime checks.
"""

from merlin.compiler.lower.options.atomic_float_add import SUPPORT as _FLOAT_SUPPORT
from merlin.compiler.lower.options.atomic_word_compare_exchange import SUPPORT as _WORD_CAS_SUPPORT

FEATURE = "lower_atomic_word_views"
MARKER = "__merlin_atomic_word_views__"


SUPPORT = (
    _FLOAT_SUPPORT
    + _WORD_CAS_SUPPORT
    + (
        r"""
_WORD_OPS = {'arith.addi': 'add', 'arith.andi': 'and', 'arith.ori': 'or', 'arith.xori': 'xor',
             'arith.maxsi': 'max', 'arith.minsi': 'min', 'arith.maxui': 'umax', 'arith.minui': 'umin'}


def _word_match(op):
    if op.name != 'memref.generic_atomic_rmw':
        return None
    if (floating_add := _float_add_match(op)) is not None:
        return floating_add
    if (word_cas := _word_cas_match(op)) is not None:
        return word_cas
    if (len(op.operands) != 2 or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        raise ValueError('word atomics require one typed rank-one atomic region')
    floating = str(op.results[0].type)
    if floating not in ('f16', 'f32', 'f64') or str(op.operands[0].type) != 'memref<?x' + floating + '>':
        raise ValueError('word atomics require identity-layout default-space f16/f32/f64 storage')
    width = int(floating[1:])
    integer = ir.IntegerType.get_signless(width, context=op.context)
    block = op.regions[0].blocks[0]
    operations = tuple(item.operation for item in block.operations)
    if len(block.arguments) != 1 or block.arguments[0].type != op.results[0].type:
        raise ValueError('word atomics require the actual floating current storage word')
    def cast(operation, source, destination):
        return (operation.name == 'arith.bitcast' and tuple(operation.operands) == (source,)
                and len(operation.results) == 1 and operation.results[0].type == destination)
    if len(operations) == 1 and operations[0].name == 'memref.atomic_yield':
        value, = operations[0].operands
        if value == block.arguments[0]:
            op.verify()
            return (*op.operands, None, width, 'identity')
"""
        "        if width != 16 and value.type == block.arguments[0].type and n"
        "ot (isinstance(value, ir.OpResult) and value.owner.operation.block == "
        "block):\n"
        r"""            op.verify()
            return (*op.operands, value, width, 'exchange-float')
        raise ValueError('atomic assignment must retain an outside floating scalar')
"""
        "    if (width != 16 and len(operations) == 2 and operations[0].name =="
        " 'arith.constant' and not operations[0].operands\n"
        "            and len(operations[0].results) == 1 and operations[0].resu"
        "lts[0].type == block.arguments[0].type\n"
        r"""            and operations[1].name == 'memref.atomic_yield'
            and tuple(operations[1].operands) == tuple(operations[0].results)):
        op.verify()
        return (*op.operands, operations[0], width, 'constant-float')
    constant = None
    if len(operations) in (3, 5):
        constants = [item for item in operations if item.name == 'arith.constant']
        if (len(constants) == 1 and not constants[0].operands and len(constants[0].results) == 1
                and constants[0].results[0].type == integer):
            constant = constants[0]
            operations = tuple(item for item in operations if item != constant)
    if len(operations) == 3:
        viewed, stored, terminal = operations
"""
        "        if (cast(viewed, block.arguments[0], integer) and cast(stored,"
        " viewed.results[0], block.arguments[0].type)\n"
        "                and terminal.name == 'memref.atomic_yield' and tuple(t"
        "erminal.operands) == tuple(stored.results)):\n"
        r"""            op.verify()
            return (*op.operands, None, width, 'identity')
        raise ValueError('atomic identity requires exact inverse current-word bitcasts')
    if len(operations) == 4:
        viewed, updated, stored, terminal = operations
        if (not cast(viewed, block.arguments[0], integer) or updated.name not in _WORD_OPS
"""
        "                or len(updated.operands) != 2 or len(updated.results) "
        "!= 1 or updated.results[0].type != integer\n"
        r"""                or not cast(stored, updated.results[0], block.arguments[0].type)):
"""
        "            raise ValueError('word atomic region differs from a closed"
        " integer update and storage bitcasts')\n"
        r"""        current = viewed.results[0]
"""
        "        value = updated.operands[1] if updated.operands[0] == current "
        "else updated.operands[0] if updated.operands[1] == current else None\n"
        r"""        kind = _WORD_OPS[updated.name]
        if (updated.name == 'arith.addi' and 'overflowFlags' in updated.attributes
"""
        "                and updated.attributes['overflowFlags'] != ir.Attribut"
        "e.parse('#arith.overflow<none>', context=op.context)):\n"
        r"""            raise ValueError('integer word addition must retain modular overflow semantics')
    elif len(operations) == 2:
        stored, terminal = operations
        value = stored.operands[0] if len(stored.operands) == 1 else None
        if value is None or not cast(stored, value, block.arguments[0].type):
            raise ValueError('word exchange requires one outside word and its storage bitcast')
        kind = 'xchg'
    else:
"""
        "        raise ValueError('word atomics require an exact update/bitcast"
        "/yield body; found ' + ', '.join(item.name for item in operations))\n"
        "    if (value is None or value.type != integer or isinstance(value, ir"
        ".OpResult) and value.owner.operation.block == block and (constant is N"
        "one or value.owner.operation != constant)\n"
        r"""            or value == block.arguments[0] or terminal.name != 'memref.atomic_yield'
            or tuple(terminal.operands) != tuple(stored.results)):
        raise ValueError('word atomic value must be an original outside-region integer scalar')
    op.verify()
    if constant is not None:
        if value != constant.results[0]:
            raise ValueError('atomic word constant must be the actual update operand')
        value, kind = constant, 'constant-int:' + kind
    return (*op.operands, value, width, kind)


def _word_source(symbol, width, kind):
"""
        "    if width == 16 and kind in ('fadd-ieee-f16', 'fadd-ieee-bf16', 'ie"
        "ee-quiet-identity-f16', 'ieee-quiet-identity-bf16'):\n"
        "        return _half_add_source(symbol, kind.rsplit('-', 1)[-1], ident"
        "ity=kind.startswith('ieee-quiet-identity'))\n"
        r'''    import json
    if kind == 'cas-word' or width == 16 and kind == 'identity':
        return _word_cas_source(symbol, width, identity=kind == 'identity')
    if width == 32 and kind == 'fadd-ftz':
        return _float_add_source(symbol)
    if width == 32 and kind == 'ftz-identity':
        return _float_add_source(symbol, identity=True)
    if width in (32, 64) and kind in ('ieee-quiet-identity', 'fadd-ieee-quiet'):
        return _float_add_source(symbol, identity=kind == 'ieee-quiet-identity', ieee=True, width=width)
    if width not in (16, 32, 64) or kind not in (*_WORD_OPS.values(), 'xchg', 'identity', 'exchange-float'):
        raise ValueError('unqualified native word atomic operation or width')
    if width == 16 and kind == 'exchange-float':
        raise ValueError('f16 word exchange requires an integer bit-pattern argument')
    result = """module {
'''
        "      func.func private @SYMBOL(%buffer: memref<?xfWIDTH>, %index: ind"
        "ex, %value: iWIDTH) -> fWIDTH attributes {llvm.linkage = #llvm.linkage"
        "<internal>} {\n"
        r'''        %bytes = arith.constant BYTES : index
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xfWIDTH> -> index
        %offset = arith.muli %index, %bytes : index
        %address = arith.addi %base, %offset : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %old = llvm.atomicrmw KIND %pointer, %value acq_rel : !llvm.ptr, iWIDTH
        %floating = arith.bitcast %old : iWIDTH to fWIDTH
        func.return %floating : fWIDTH
      }
    }"""
    if kind == 'identity':
        result = result.replace(', %value: iWIDTH', '').replace(
'''
        "            '%old = llvm.atomicrmw KIND %pointer, %value', '%zero = ar"
        "ith.constant 0 : iWIDTH\\n        %old = llvm.atomicrmw _or %pointer, %"
        "zero')\n"
        r"""    elif kind == 'exchange-float':
        result = result.replace('%value: iWIDTH', '%value: fWIDTH').replace(
            'llvm.atomicrmw KIND %pointer, %value acq_rel : !llvm.ptr, iWIDTH',
            'llvm.atomicrmw xchg %pointer, %value acq_rel : !llvm.ptr, fWIDTH').replace(
"""
        "            '%floating = arith.bitcast %old : iWIDTH to fWIDTH', '').r"
        "eplace('func.return %floating', 'func.return %old')\n"
        r"""    if width == 16:
        result = result.replace(') -> fWIDTH', ') -> iWIDTH').replace(
            '%floating = arith.bitcast %old : iWIDTH to fWIDTH', '').replace(
            'func.return %floating : fWIDTH', 'func.return %old : iWIDTH')
    return result.replace('WIDTH', str(width)).replace('BYTES', str(width // 8)).replace(
"""
        "        'KIND', '_' + kind if kind in ('and', 'or', 'xor') else kind)."
        "replace('SYMBOL', json.dumps(symbol))\n"
        r"""

def _word_helper_equal(op, symbol, module):
    # This grants synchronous borrowing only. Reconstruct from the actual
    # primitive/type, never the symbol spelling, source tag or saved plan.
    if _word_cas_helper_equal(op, symbol, module):
        return True
    if _half_add_helper_equal(op, symbol, module):
        return True
    block = op.regions[0].blocks[0]
    signature = ir.FunctionType(ir.TypeAttr(op.attributes['function_type']).value)
    if len(signature.results) != 1:
        return False
    result = str(signature.results[0])
    if result == 'i16' and block.arguments and str(block.arguments[0].type) == 'memref<?xf16>':
        width = 16
    elif result in ('f32', 'f64'):
        width = int(result[1:])
    else:
        return False
    if len(block.arguments) == 2:
        kinds = ('identity',)
"""
        "        if any(_float_add_helper_equal(op, symbol, module, identity=Tr"
        "ue, ieee=ieee, width=width) for ieee in ((False, True) if width == 32 "
        "else (True,))):\n"
        r"""            return True
    elif len(block.arguments) == 3 and str(block.arguments[2].type) == 'i' + str(width):
        kinds = (*_WORD_OPS.values(), 'xchg')
    elif len(block.arguments) == 3 and block.arguments[2].type == signature.results[0]:
        kinds = ('exchange-float',)
"""
        "        if any(_float_add_helper_equal(op, symbol, module, ieee=ieee, "
        "width=width) for ieee in ((False, True) if width == 32 else (True,))):"
        "\n"
        r"""            return True
    else:
        return False
    atomics = [item for item in _cas_walk(op) if item.name == 'llvm.atomicrmw']
    if len(atomics) != 1 or 'bin_op' not in atomics[0].attributes:
        return False
    from torch_mlir import passmanager as _word_pm
    for kind in kinds:
        with module.context:
            reference = ir.Module.parse(_word_source(symbol, width, kind))
"""
        "            primitive = next(item for item in _cas_walk(reference.oper"
        "ation) if item.name == 'llvm.atomicrmw')\n"
        r"""            # LLVM's native binding encodes this enum as an integer attribute.
            # Obtain its value from the registered parser, not a duplicated enum.
            if primitive.attributes['bin_op'] != atomics[0].attributes['bin_op']:
                continue
            _word_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(reference.operation)
        expected = tuple(reference.body.operations)[0].operation
"""
        "        return op.get_asm(print_generic_op_form=True, use_local_scope="
        "True) == expected.get_asm(print_generic_op_form=True, use_local_scope="
        "True)\n"
        r"""    return False

"""
    )
)


def edit_pipeline(passes):
    from merlin.compiler.lower.options.atomic_compare_exchange import MARKER as CAS_MARKER
    from merlin.compiler.lower.options.atomic_compare_exchange import edit_pipeline as edit_cas
    from merlin.compiler.lower.options.atomic_integer_regions import MARKER as REGION_MARKER

    if REGION_MARKER in passes:
        return edit_cas(passes)
    if CAS_MARKER in passes:
        at = passes.index(CAS_MARKER)
        if (
            passes.count(CAS_MARKER) != 1
            or MARKER in passes
            or at + 1 == len(passes)
            or not passes[at + 1].startswith("one-shot-bufferize{")
        ):
            raise ValueError("word atomics require one unmodified bufferization seam")
        return [MARKER if name == CAS_MARKER else name for name in passes]
    return [MARKER if name == CAS_MARKER else name for name in edit_cas(passes)]


def ensure_registered():
    from merlin.compiler.optimizations.registry import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description=(
                    "Lower exact word updates, IEEE f32/f64 and FTZ f32 addition, and integ"
                    "er CAS to checked native atomics; default off."
                ),
            )
        )
    return FEATURE
