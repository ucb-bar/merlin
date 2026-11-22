"""Pure integer atomic regions through the checked native borrowing seam."""

FEATURE = "lower_integer_atomic_regions"
MARKER = "__merlin_integer_atomic_regions__"

SUPPORT = r"""
_REGION_TYPES = ('i1', 'i4', 'i8', 'i16', 'i32', 'i64')
_REGION_OPS = ('arith.constant', 'arith.addi', 'arith.subi', 'arith.muli',
    'arith.andi', 'arith.ori', 'arith.xori', 'arith.shli', 'arith.shrui',
    'arith.shrsi', 'arith.cmpi', 'arith.select', 'arith.maxsi', 'arith.minsi',
    'arith.maxui', 'arith.minui', 'arith.trunci', 'arith.extui', 'arith.extsi')
_SMALL_RMW = {1: 'arith.addi', 2: 'arith.andi', 3: 'assign', 6: 'arith.maxsi',
              7: 'arith.maxui', 10: 'arith.minsi', 11: 'arith.minui', 13: 'arith.muli',
              14: 'arith.ori', 15: 'arith.xori'}


def _small_memory(memory, element):
    if not isinstance(memory.type, ir.MemRefType):
        raise ValueError('subbyte atomics require an actual ranked memref')
    typed = ir.MemRefType(memory.type)
    with element.context, ir.Location.unknown():
        default = ir.MemRefType.get(typed.shape, element)
    if len(typed.shape) != 1 or memory.type != default:
        raise ValueError('subbyte atomics require identity-layout default-space rank-one cells')


def _small_rmw_match(op):
    if op.name != 'memref.atomic_rmw' or len(op.results) != 1 or str(op.results[0].type) not in ('i1', 'i4'):
        return None
    if len(op.operands) != 3 or op.regions:
        raise ValueError('subbyte RMW requires one actual value, memory and index')
    value, memory, index = op.operands
    _small_memory(memory, op.results[0].type)
    names = [item if isinstance(item, str) else item.name for item in op.attributes]
    if (value.type != op.results[0].type or str(index.type) != 'index'
            or 'kind' not in names
            or any(name != 'kind' and not name.startswith('prov.') for name in names)):
        raise ValueError('subbyte RMW changed its complete typed operation')
    kind = ir.IntegerAttr(op.attributes['kind']).value
    if kind not in _SMALL_RMW:
        raise ValueError('subbyte RMW needs a defined integer update')
    op.verify()
    return memory, index, value, int(str(value.type)[1:]), _SMALL_RMW[kind]


def _small_rmw_update(ctx, symbol, width, kind):
    import json
    word = 'i' + str(width)
    if width not in (1, 4) or kind not in _SMALL_RMW.values():
        raise ValueError('subbyte update has an unqualified integer prototype')
    body = 'return %value : ' + word if kind == 'assign' else (
        '%changed = ' + kind + ' %current, %value : ' + word + '\nreturn %changed : ' + word)
    module = ir.Module.parse('module {func.func private @' + json.dumps(symbol)
        + '(%current: ' + word + ', %value: ' + word + ') -> ' + word
        + ' attributes {llvm.linkage = #llvm.linkage<internal>} {' + body + '}}', context=ctx)
    function = tuple(module.body.operations)[0].operation
    _region_body(function.regions[0].blocks[0], 'func.return', function.regions[0].blocks[0].arguments[0].type,
                 capture=False)
    function.detach_from_parent()
    return function


def _region_body(block, terminal_name, result_type, *, capture):
    operations = tuple(item.operation for item in block.operations)
    if not operations or len(operations) > 512:
        raise ValueError('integer atomic region exceeds its closed body budget')
    terminal = operations[-1]
    if (terminal.name != terminal_name or len(terminal.operands) != 1
            or terminal.operands[0].type != result_type):
        raise ValueError('integer atomic region must yield its exact word type')
    known, captures = list(block.arguments), []
    for op in operations[:-1]:
        if (op.name not in _REGION_OPS or op.regions or len(op.results) != 1
                or str(op.results[0].type) not in _REGION_TYPES):
            raise ValueError('integer atomic regions require qualified pure scalar arithmetic')
        if ('overflowFlags' in op.attributes
                and op.attributes['overflowFlags'] != ir.Attribute.parse('#arith.overflow<none>', context=op.context)):
            raise ValueError('integer atomic arithmetic requires modular overflow semantics')
        for value in op.operands:
            if str(value.type) not in _REGION_TYPES:
                raise ValueError('integer atomic regions cannot capture addresses or noninteger values')
            if value not in known:
                if not capture:
                    raise ValueError('integer atomic update function has a foreign SSA capture')
                if value not in captures:
                    captures.append(value)
        known.extend(op.results)
    if terminal.operands[0] not in known:
        if not capture or str(terminal.operands[0].type) not in _REGION_TYPES:
            raise ValueError('integer atomic yield has an unknown value')
        if terminal.operands[0] not in captures:
            captures.append(terminal.operands[0])
    if len(captures) > 32:
        raise ValueError('integer atomic region exceeds its scalar capture budget')
    needed = [terminal.operands[0]]
    for op in reversed(operations[:-1]):
        if not any(value in needed for value in op.results):
            raise ValueError('integer atomic region contains unaccounted arithmetic')
        needed.extend(value for value in op.operands if value not in needed)
    return tuple(captures)


def _region_match(op):
    if (len(op.operands) != 2 or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        raise ValueError('integer atomic regions require one rank-one scalar update')
    word = str(op.results[0].type)
    if word in ('i1', 'i4'):
        _small_memory(op.operands[0], op.results[0].type)
    elif word not in ('i8', 'i16', 'i32', 'i64') or str(op.operands[0].type) != 'memref<?x' + word + '>':
        raise ValueError('integer atomic regions require identity-layout default-space word storage')
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1 or block.arguments[0].type != op.results[0].type:
        raise ValueError('integer atomic region current value differs from its word type')
    captures = _region_body(block, 'memref.atomic_yield', op.results[0].type, capture=True)
    op.verify()
    return (*op.operands, captures, int(word[1:]))


def _region_source(symbol, width, update, types):
    import json
    word = 'i' + str(width)
    args = ''.join(', %v' + str(i) + ': ' + type_ for i, type_ in enumerate(types))
    values = ''.join(', %v' + str(i) for i in range(len(types)))
    typed = ''.join(', ' + type_ for type_ in types)
    declaration = 'func.func private @' + json.dumps(update) + '(' + ', '.join((word, *types)) + ') -> ' + word
    body = '''module {
      DECLARATION
      func.func private @SYMBOL(%buffer: memref<?xWORD>, %index: indexARGS) -> WORD
          attributes {llvm.linkage = #llvm.linkage<internal>} {
        %bytes = arith.constant BYTES : index
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xWORD> -> index
        %offset = arith.muli %index, %bytes : index
        %address = arith.addi %base, %offset : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %true = arith.constant true
        %start = llvm.load %pointer atomic acquire {alignment = BYTES : i64} : !llvm.ptr -> PHYSICAL
        %old = scf.while (%current = %start) : (PHYSICAL) -> PHYSICAL {
          INPUT
          %desired = func.call @UPDATE(CURRENTVALUES) : (WORDTYPES) -> WORD
          OUTPUT
          %pair = llvm.cmpxchg %pointer, %current, REPLACEMENT acq_rel acquire : !llvm.ptr, PHYSICAL
          %observed = llvm.extractvalue %pair[0] : !llvm.struct<(PHYSICAL, i1)>
          %success = llvm.extractvalue %pair[1] : !llvm.struct<(PHYSICAL, i1)>
          %retry = arith.xori %success, %true : i1
          scf.condition(%retry) %observed : PHYSICAL
        } do {
        ^bb0(%next: PHYSICAL):
          scf.yield %next : PHYSICAL
        }
        RESULT
      }
    }'''
    subbyte = width < 8
    for key, value in (('DECLARATION', declaration), ('SYMBOL', json.dumps(symbol)),
                       ('ARGS', args), ('VALUES', values), ('TYPES', typed),
                       ('UPDATE', json.dumps(update)), ('BYTES', str(max(1, width // 8))),
                       ('INPUT', '%logical = arith.trunci %current : i8 to WORD' if subbyte else ''),
                       ('CURRENT', '%logical' if subbyte else '%current'),
                       ('OUTPUT', '%physical = arith.extui %desired : WORD to i8' if subbyte else ''),
                       ('REPLACEMENT', '%physical' if subbyte else '%desired'),
                       ('RESULT', '%result = arith.trunci %old : i8 to WORD\nfunc.return %result : WORD'
                        if subbyte else 'func.return %old : WORD'),
                       ('PHYSICAL', 'i8' if subbyte else word), ('WORD', word)):
        body = body.replace(key, value)
    return body


def _region_update(ctx, op, symbol, captures):
    import json
    types = (op.results[0].type, *(value.type for value in captures))
    args = ', '.join('%a' + str(i) + ': ' + str(type_) for i, type_ in enumerate(types))
    source = ('module { func.func private @' + json.dumps(symbol) + '(' + args + ') -> '
        + str(types[0]) + ' attributes {llvm.linkage = #llvm.linkage<internal>} {'
        + '%zero = arith.constant 0 : ' + str(types[0]) + ' return %zero : ' + str(types[0]) + '}}')
    helper = ir.Module.parse(source, context=ctx)
    function = tuple(helper.body.operations)[0].operation
    block = function.regions[0].blocks[0]
    for item in reversed(tuple(block.operations)):
        item.operation.erase()
    original = op.regions[0].blocks[0]
    mapping = dict(zip((original.arguments[0], *captures), block.arguments))
    with ctx, ir.Location.unknown(), ir.InsertionPoint(block):
        for item in tuple(original.operations)[:-1]:
            primitive = item.operation
            cloned = primitive.clone()
            for i, value in enumerate(primitive.operands):
                cloned.operands[i] = mapping[value]
            mapping.update(zip(primitive.results, cloned.results))
        terminal = tuple(original.operations)[-1].operation
        ir.Operation.create('func.return', operands=[mapping[terminal.operands[0]]])
    function.verify()
    function.detach_from_parent()
    return function


def _region_helper_equal(op, symbol, module, functions, duplicates):
    # Reconstruct the complete borrowing template and verify its actual scalar
    # callee. A saved plan, marker or symbol prefix grants no no-escape premise.
    block = op.regions[0].blocks[0]
    types = tuple(str(value.type) for value in block.arguments)
    signature = ir.FunctionType(ir.TypeAttr(op.attributes['function_type']).value)
    if (len(types) < 2 or len(signature.results) != 1
            or str(signature.results[0]) not in _REGION_TYPES
            or types[:2] != ('memref<?x' + str(signature.results[0]) + '>', 'index')
            or any(type_ not in _REGION_TYPES for type_ in types[2:])):
        return False
    calls = [item for item in _cas_walk(op) if item.name == 'func.call']
    if len(calls) != 1:
        return False
    update = ir.FlatSymbolRefAttr(calls[0].attributes['callee']).value
    scalar = functions.get(update)
    if (scalar is None or update in duplicates or len(scalar.regions) != 1
            or len(scalar.regions[0].blocks) != 1
            or 'sym_visibility' not in scalar.attributes or str(scalar.attributes['sym_visibility']) != '"private"'
            or 'llvm.linkage' not in scalar.attributes
            or str(scalar.attributes['llvm.linkage']) != '#llvm.linkage<internal>'):
        return False
    scalar_block = scalar.regions[0].blocks[0]
    if tuple(str(value.type) for value in scalar_block.arguments) != (str(signature.results[0]), *types[2:]):
        return False
    try:
        _region_body(scalar_block, 'func.return', signature.results[0], capture=False)
        scalar.verify()
    except (ValueError, RuntimeError):
        return False
    from torch_mlir import passmanager as _region_pm
    with module.context:
        reference = ir.Module.parse(_region_source(symbol, int(str(signature.results[0])[1:]), update, types[2:]))
        _region_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(reference.operation)
    expected = tuple(reference.body.operations)[-1].operation
    actual_text = op.get_asm(print_generic_op_form=True, use_local_scope=True)
    expected_text = expected.get_asm(print_generic_op_form=True, use_local_scope=True)
    return actual_text == expected_text
"""


def edit_pipeline(passes):
    from merlin.compiler.lower.options.atomic_compare_exchange import MARKER as CAS
    from merlin.compiler.lower.options.atomic_compare_exchange import edit_pipeline as edit_cas
    from merlin.compiler.lower.options.atomic_word_views import MARKER as WORD

    points = list(passes)
    positions = [i for i, name in enumerate(points) if name.startswith("one-shot-bufferize{")]
    if MARKER in points or len(positions) != 1:
        raise ValueError("integer atomic regions require one unmodified bufferization seam")
    markers = [name for name in (CAS, WORD) if name in points]
    if markers:
        if len(markers) != 1 or points.count(markers[0]) != 1 or points.index(markers[0]) != positions[0] - 1:
            raise ValueError("integer atomic regions require one unmodified bufferization seam")
        return [MARKER if name == markers[0] else name for name in points]
    return [MARKER if name == CAS else name for name in edit_cas(points)]


def ensure_registered():
    from merlin.compiler.optimizations.registry import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description=(
                    "Lower closed pure integer atomic regions through checked atomic-load/CAS helpers; default off."
                ),
            )
        )
    return FEATURE
