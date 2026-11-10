"""Lower standard literal assertions without a NUL-terminated message ABI.

The upstream CF conversion uses puts. At the same late native boundary, keep
the complete UTF-8 message plus linefeed in a private constant and pass its
actual byte count to an internal byte loop. No source pointer, allocation or
borrowed buffer crosses this helper. The C runtime supplies putchar and abort.
"""

import ast

FEATURE = "lower_length_aware_diagnostics"
MARKER = "__merlin_length_aware_diagnostics__"

SUPPORT = r"""
def _diag_walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _diag_walk(child.operation)


def _diag_lower(module):
    with module.context:
        return _diag_lower_in_context(module)


def _diag_lower_in_context(module):
    module.operation.verify()
    operations = tuple(_diag_walk(module.operation))
    assertions = [op for op in operations if op.name == 'cf.assert']
    if not assertions:
        return 0
    if any(op.name.split('.')[0] not in ('builtin', 'llvm', 'cf') for op in operations):
        raise ValueError('literal diagnostics require the late LLVM/CF boundary')
    for op in assertions:
        if (len(op.operands) != 1 or str(op.operands[0].type) != 'i1'
                or op.results or op.regions or 'msg' not in op.attributes
                or not isinstance(op.attributes['msg'], ir.StringAttr)):
            raise ValueError('literal diagnostics require a scalar standard assertion')
    symbols = {ir.StringAttr(op.attributes['sym_name']).value: op
               for op in module.body.operations if 'sym_name' in op.attributes}
    declarations = ir.Module.parse('module { llvm.func @putchar(i32) -> i32 llvm.func @abort() }')
    for declaration in declarations.body.operations:
        name = ir.StringAttr(declaration.attributes['sym_name']).value
        if name in symbols and str(symbols[name]).strip() != str(declaration).strip():
            raise ValueError('literal diagnostic runtime symbol has an incompatible definition: ' + name)
    # All refusals above precede edits. Fresh internal symbols avoid collisions
    # without interpreting an existing symbol name as an implementation grant.
    def fresh(stem):
        for index in range(len(symbols) + len(assertions) + 3):
            candidate = stem + str(index)
            if candidate not in symbols:
                symbols[candidate] = None
                return candidate
        raise ValueError('literal diagnostic symbol budget exhausted')
    helper = fresh('__merlin_assert_bytes_')
    messages = {}
    for op in assertions:
        message = ir.StringAttr(op.attributes['msg']).value + '\n'
        if message not in messages:
            messages[message] = fresh('__merlin_diagnostic_')
    helper_ir = '''
module {
  llvm.func internal @HELPER(%condition: i1, %message: !llvm.ptr, %length: i64) {
    llvm.cond_br %condition, ^done, ^start
  ^start:
    %zero = llvm.mlir.constant(0 : i64) : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^loop(%zero : i64)
  ^loop(%index: i64):
    %more = llvm.icmp "ult" %index, %length : i64
    llvm.cond_br %more, ^byte, ^fail
  ^byte:
    %address = llvm.getelementptr %message[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %value = llvm.load %address : !llvm.ptr -> i8
    %character = llvm.zext %value : i8 to i32
    %written = llvm.call @putchar(%character) : (i32) -> i32
    %next = llvm.add %index, %one : i64
    llvm.br ^loop(%next : i64)
  ^fail:
    llvm.call @abort() : () -> ()
    llvm.unreachable
  ^done:
    llvm.return
  }
  llvm.func @putchar(i32) -> i32
  llvm.func @abort()
}
'''
    prepared = ir.Module.parse(helper_ir.replace('HELPER', helper))
    template = next(op for op in _diag_walk(prepared.operation) if op.name == 'llvm.call')
    call_attributes = {name: template.attributes[name] for name in template.attributes}
    for op in tuple(prepared.body.operations):
        name = ir.StringAttr(op.attributes['sym_name']).value
        if name not in ('putchar', 'abort') or name not in symbols:
            op.operation.detach_from_parent()
            module.body.append(op.operation)
    for message, name in messages.items():
        count = len(message.encode('utf-8'))
        literal = str(ir.StringAttr.get(message))
        glob = ir.Module.parse('module { llvm.mlir.global private constant @' + name
            + '(' + literal + ') : !llvm.array<' + str(count) + ' x i8> }')
        operation = glob.body.operations[0].operation
        operation.detach_from_parent()
        module.body.append(operation)
    pointer, i64 = ir.Type.parse('!llvm.ptr'), ir.IntegerType.get_signless(64)
    for op in assertions:
        message = ir.StringAttr(op.attributes['msg']).value + '\n'
        attributes = dict(call_attributes)
        attributes.update({name: op.attributes[name] for name in op.attributes if name.startswith('prov.')})
        attributes['callee'] = ir.FlatSymbolRefAttr.get(helper)
        attributes['operandSegmentSizes'] = ir.DenseI32ArrayAttr.get([3, 0])
        with ir.InsertionPoint(op), op.location:
            address = ir.Operation.create('llvm.mlir.addressof', results=[pointer],
                attributes={'global_name': ir.FlatSymbolRefAttr.get(messages[message])})
            count = ir.Operation.create('llvm.mlir.constant', results=[i64],
                attributes={'value': ir.IntegerAttr.get(i64, len(message.encode('utf-8')))})
            ir.Operation.create('llvm.call', operands=[op.operands[0], address.results[0], count.results[0]],
                attributes=attributes)
        op.erase()
    module.operation.verify()
    return len(assertions)
"""

PRELUDE = SUPPORT + r'''
_DIAG_PREVIOUS_RUN_STAGES = _run_stages
_DIAG_CONSUMED = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    marker = '__merlin_length_aware_diagnostics__'
    if marker not in pipeline:
        return _DIAG_PREVIOUS_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if pipeline.count(marker) != 1:
        raise ValueError('literal diagnostics require one late CF conversion seam')
    head, _, tail = pipeline.partition(marker)
    head, tail = head.rstrip(','), tail.lstrip(',')
    if not tail.startswith('convert-cf-to-llvm'):
        raise ValueError('literal diagnostics must immediately precede CF conversion')
    result = _DIAG_PREVIOUS_RUN_STAGES(ctx, module, head, erase, mid, late, post_openmp, pre_generalize)
    _diag_lower(module)
    _DIAG_PREVIOUS_RUN_STAGES(ctx, module, tail, 0, (), (), (), ())
    global _DIAG_CONSUMED
    _DIAG_CONSUMED = True
    return result
'''


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [node for node in ast.parse(source).body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "_run_stages"]
    if len(calls) != 1:
        raise ValueError("literal diagnostics require one native stage runner call")
    lines = source.splitlines(keepends=True)
    at, after = calls[0].lineno - 1, calls[0].end_lineno
    return ("".join(lines[:at]) + PRELUDE + "\n" + "".join(lines[at:after])
            + "\nif not _DIAG_CONSUMED:\n    raise ValueError('literal diagnostic marker was not consumed')\n"
            + "".join(lines[after:]))


def edit_pipeline(passes):
    if passes.count("convert-cf-to-llvm") != 1 or MARKER in passes:
        raise ValueError("literal diagnostics require one unmodified CF conversion seam")
    at = passes.index("convert-cf-to-llvm")
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE, action_class="PASS", edit_pipeline=edit_pipeline,
            description="Preserve complete literal assertion bytes through native stdio; default off."))
    return FEATURE
