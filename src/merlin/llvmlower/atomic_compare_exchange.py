"""Exact integer compare/exchange legalization before native ownership analysis.

Portable input retains standard memref.generic_atomic_rmw. The pinned ownership
pass requires RegionBranchOpInterface for scalar-result regions, which MemRef's
atomic region does not implement. A typed private synchronous helper keeps the
memref live across one LLVM cmpxchg, with no escaping pointer or allocation.
"""

import ast

FEATURE = "lower_integer_compare_exchange"
MARKER = "__merlin_integer_compare_exchange__"

SUPPORT = r"""
_CAS_GENERATED = []
_CAS_PENDING = []


def _cas_native_policy(pipeline):
    import re, struct
    if struct.calcsize('P') != 8:
        raise ValueError('native compare/exchange requires the qualified 64-bit host descriptor ABI')
    widths = re.findall(r'index-bitwidth\s*=\s*(?:"(\d+)"|(\d+))', pipeline)
    if len(widths) != pipeline.count('index-bitwidth') or any(int(a or b) not in (0, 64) for a, b in widths):
        raise ValueError('native compare/exchange requires the qualified default 64-bit index ABI')

def _cas_walk(operation):
    for region in operation.regions:
        for block in region.blocks:
            for item in block.operations:
                yield item.operation
                yield from _cas_walk(item.operation)


def _cas_source(symbol, width):
    import json
    return '''module {
      func.func private @SYMBOL(%buffer: memref<?xiWIDTH>, %index: index, %comparison: iWIDTH, %value: iWIDTH) -> iWIDTH attributes {llvm.linkage = #llvm.linkage<internal>} {
        %bytes = arith.constant BYTES : index
        %base = memref.extract_aligned_pointer_as_index %buffer : memref<?xiWIDTH> -> index
        %offset = arith.muli %index, %bytes : index
        %address = arith.addi %base, %offset : index
        %integer = arith.index_cast %address : index to i64
        %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
        %pair = llvm.cmpxchg %pointer, %comparison, %value acq_rel acquire : !llvm.ptr, iWIDTH
        %old = llvm.extractvalue %pair[0] : !llvm.struct<(iWIDTH, i1)>
        func.return %old : iWIDTH
      }
    }'''.replace('WIDTH', str(width)).replace('BYTES', str(width // 8)).replace('SYMBOL', json.dumps(symbol))


def _cas_root_function(op, module):
    scopes = ('scf.for', 'scf.if', 'scf.while', 'scf.execute_region', 'scf.parallel', 'scf.forall',
              'scf.forall.in_parallel', 'linalg.generic', 'linalg.fill', 'affine.for', 'affine.parallel')
    parent = op.block.owner.operation
    while parent.name in scopes:
        parent = parent.block.owner.operation
    return parent if parent.name == 'func.func' and parent.block == module.body else None


def _cas_match(op):
    if op.name != 'memref.generic_atomic_rmw':
        return None
    if (len(op.operands) != 2 or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        raise ValueError('compare/exchange requires one typed rank-one atomic region')
    width = str(op.results[0].type)
    if width not in ('i16', 'i32', 'i64') or str(op.operands[0].type) != 'memref<?x' + width + '>':
        raise ValueError('compare/exchange requires identity-layout default-space i16/i32/i64 memory')
    block = op.regions[0].blocks[0]
    operations = tuple(item.operation for item in block.operations)
    if len(block.arguments) != 1 or block.arguments[0].type != op.results[0].type or len(operations) != 3:
        raise ValueError('compare/exchange requires its closed comparison/select/yield body')
    equal, selected, terminal = operations
    current = block.arguments[0]
    if (equal.name != 'arith.cmpi' or len(equal.operands) != 2 or ir.IntegerAttr(equal.attributes['predicate']).value != 0
            or selected.name != 'arith.select' or tuple(selected.operands)[::2] != (equal.results[0], current)
            or terminal.name != 'memref.atomic_yield' or tuple(terminal.operands) != tuple(selected.results)):
        raise ValueError('atomic region differs from exact integer compare/exchange')
    comparison = equal.operands[1] if equal.operands[0] == current else equal.operands[0] if equal.operands[1] == current else None
    value = selected.operands[1]
    if (comparison is None or any(v == current or v.type != current.type or
            isinstance(v, ir.OpResult) and v.owner.operation.block == block for v in (comparison, value))):
        raise ValueError('compare/exchange comparison/value must be original outside-region scalars')
    op.verify()
    return (*op.operands, comparison, value, int(width[1:]))


def _cas_rewrite(ctx, module):
    if _CAS_PENDING:
        raise ValueError('previous atomic helpers were not restored')
    _CAS_GENERATED.clear()
    operations = tuple(_cas_walk(module.operation))
    plans = [(op, match) for op in operations if (match := _cas_match(op)) is not None]
    if any(_cas_root_function(op, module) is None for op, _ in plans):
        raise ValueError('compare/exchange requires a direct root function and closed standard control scopes')
    if plans:
        _cas_native_policy('')
        owners = {module.operation, *(_cas_root_function(op, module) for op, _ in plans)}
        for owner in owners:
            names = [item if isinstance(item, str) else item.name for item in owner.attributes]
            if any(name.startswith('dlti.') or name in ('llvm.data_layout', 'llvm.target_triple') for name in names):
                raise ValueError('native compare/exchange requires the default compiler data layout')
    symbols = {ir.StringAttr(op.attributes['sym_name']).value for op in module.body.operations if 'sym_name' in op.attributes}
    helpers = {}
    for op, (memory, index, comparison, value, width) in plans:
        if width not in helpers:
            ordinal = 0
            while (symbol := '__merlin_atomic_cas_' + str(width) + '_' + str(ordinal)) in symbols:
                ordinal += 1
            symbols.add(symbol)
            helper = ir.Module.parse(_cas_source(symbol, width), context=ctx)
            function = tuple(helper.body.operations)[0].operation
            function.detach_from_parent()
            module.body.append(function)
            _CAS_GENERATED.append(function)
            helpers[width] = symbol
        with ctx, ir.Location.unknown(), ir.InsertionPoint(op):
            attributes = {}
            for item in op.attributes:
                name = item if isinstance(item, str) else item.name
                if name.startswith('prov.'):
                    attributes[name] = op.attributes[name]
            call = ir.Operation.create('func.call', operands=[memory, index, comparison, value],
                results=[op.results[0].type], attributes={'callee': ir.FlatSymbolRefAttr.get(helpers[width]),
                    **attributes})
        op.results[0].replace_all_uses_with(call.results[0])
        op.erase()
    module.operation.verify()
    print('OK integer_compare_exchange', len(plans), flush=True)
    return len(plans)


def _cas_suspend(module):
    # Ownership analysis sees the typed borrowing call boundary. The complete
    # generated allocation-free body remains retained as an actual IR object,
    # and is restored before liveness checking and LLVM conversion. This avoids
    # asking ownership analysis to interpret LLVM scalar memory primitives.
    if _CAS_PENDING:
        raise ValueError('atomic helper suspension cannot be nested')
    for function in _CAS_GENERATED:
        attributes = {}
        for item in function.attributes:
            name = item if isinstance(item, str) else item.name
            attributes[name] = function.attributes[name]
        function.detach_from_parent()
        with module.context, ir.Location.unknown(), ir.InsertionPoint(module.body):
            declaration = ir.Operation.create('func.func', attributes=attributes, regions=1)
        _CAS_PENDING.append((function, declaration))
    module.operation.verify()


def _cas_restore(module):
    count = len(_CAS_PENDING)
    for function, declaration in _CAS_PENDING:
        def attributes(op):
            names = [item if isinstance(item, str) else item.name for item in op.attributes]
            return {name: str(op.attributes[name]) for name in names}
        if (declaration.block != module.body or declaration.regions[0].blocks
                or attributes(declaration) != attributes(function)):
            raise ValueError('atomic borrowing declaration changed before body restoration')
        declaration.erase()
        module.body.append(function)
    _CAS_PENDING.clear()
    module.operation.verify()
    print('OK integer_compare_exchange_helpers_restored', count, flush=True)


def _cas_checked_calls(module):
    # Reconstruct the complete current typed helper body. A symbol spelling,
    # saved record or provenance tag supplies no borrowing/no-escape grant.
    from torch_mlir import passmanager as _cas_pm
    functions, duplicates = {}, set()
    for item in module.body.operations:
        op = item.operation
        if op.name == 'func.func':
            symbol = ir.StringAttr(op.attributes['sym_name']).value
            if symbol in functions:
                duplicates.add(symbol)
            functions[symbol] = op
    accepted = set()
    for symbol, op in functions.items():
        if symbol in duplicates or not op.regions or len(op.regions[0].blocks) != 1:
            continue
        block = op.regions[0].blocks[0]
        if len(block.arguments) != 4 or not any(item.operation.name == 'llvm.cmpxchg' for item in block.operations):
            continue
        type_ = str(block.arguments[2].type)
        if type_ not in ('i16', 'i32', 'i64'):
            continue
        with module.context:
            reference = ir.Module.parse(_cas_source(symbol, int(type_[1:])))
            _cas_pm.PassManager.parse('builtin.module(canonicalize,cse)').run(reference.operation)
        expected = tuple(reference.body.operations)[0].operation
        if op.get_asm(print_generic_op_form=True, use_local_scope=True) == expected.get_asm(print_generic_op_form=True, use_local_scope=True):
            accepted.add(symbol)
    if not accepted:
        return set()
    result = set()
    for op in _cas_walk(module.operation):
        if op.name != 'func.call' or 'callee' not in op.attributes:
            continue
        try:
            symbol = ir.FlatSymbolRefAttr(op.attributes['callee']).value
        except ValueError:
            continue
        if symbol in accepted:
            function = functions[symbol]
            signature = ir.FunctionType(ir.TypeAttr(function.attributes['function_type']).value)
            if (_cas_root_function(op, module) is not None and tuple(v.type for v in op.operands) == tuple(signature.inputs)
                    and tuple(v.type for v in op.results) == tuple(signature.results)):
                result.add(op.operation)
    return result
"""

PRELUDE = SUPPORT + r'''
_CAS_PREVIOUS_RUN_STAGES = _run_stages
_CAS_CONSUMED = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    marker = '__merlin_integer_compare_exchange__'
    if marker not in pipeline:
        return _CAS_PREVIOUS_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if pipeline.count(marker) != 1:
        raise ValueError('integer compare/exchange requires one pre-bufferization marker')
    _cas_native_policy(pipeline)
    head, _, tail = pipeline.partition(marker)
    head, tail = head.rstrip(','), tail.lstrip(',')
    if not tail.startswith('one-shot-bufferize{') or 'one-shot-bufferize' in head:
        raise ValueError('integer compare/exchange must immediately precede bufferization')
    if head:
        _CAS_PREVIOUS_RUN_STAGES(ctx, module, head, 0, (), (), (), pre_generalize)
    _cas_rewrite(ctx, module)
    _cas_suspend(module)
    global _CAS_CONSUMED
    _CAS_CONSUMED = True
    passes = tail.split(',')
    position = next((i for i, p in enumerate(passes) if p == 'bufferization-lower-deallocations'), -1)
    if position < 0 or any('optimize-allocation-liveness' in p for p in passes[:position + 1]):
        raise ValueError('atomic helpers require ownership lowering before liveness verification')
    _CAS_PREVIOUS_RUN_STAGES(ctx, module, ','.join(passes[:position + 1]), erase, mid, (), (), ())
    _cas_restore(module)
    return _CAS_PREVIOUS_RUN_STAGES(ctx, module, ','.join(passes[position + 1:]), 0, (), late, post_openmp, ())
'''


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [node for node in ast.parse(source).body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == "_run_stages"]
    if len(calls) != 1:
        raise ValueError("compare/exchange requires one native stage runner call")
    lines = source.splitlines(keepends=True)
    at = calls[0].lineno - 1
    after = calls[0].end_lineno
    return ("".join(lines[:at]) + PRELUDE + "\n" + "".join(lines[at:after])
            + "\nif not _CAS_CONSUMED:\n    raise ValueError('integer compare/exchange marker was not consumed')\n"
            + "".join(lines[after:]))


def edit_pipeline(passes):
    positions = [i for i, name in enumerate(passes) if name.startswith("one-shot-bufferize{")]
    if len(positions) != 1 or MARKER in passes:
        raise ValueError("compare/exchange requires one unmodified bufferization seam")
    at = positions[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE, action_class="PASS", edit_pipeline=edit_pipeline,
            description="Lower exact integer compare/exchange regions to checked synchronous native helpers; default off."))
    return FEATURE
