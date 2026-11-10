"""Explicit host materialization of standard scalar/string debug output.

VectorOps.td defines human-readable stdout effects. Triton's portable event
profile selects signed decimal i64 and fixed-six-place f64; word formatting
and unsigned-64 conversion remain executable standard frontend arithmetic.
"""

from .length_aware_diagnostics import SUPPORT as ASSERT_SUPPORT
from .length_aware_diagnostics import bind_runner as _bind, edit_pipeline as _edit, MARKER as ASSERT_MARKER

FEATURE = "lower_scalar_diagnostics"
MARKER = "__merlin_scalar_diagnostics__"

PRINT_SUPPORT = r"""
def _scalar_diag_lower(module):
    with module.context:
        return _scalar_diag_lower_in_context(module)


def _scalar_diag_lower_in_context(module):
    module.operation.verify()
    operations = tuple(_diag_walk(module.operation))
    prints = [op for op in operations if op.name == 'vector.print']
    if any(op.name.split('.')[0] not in ('builtin','llvm','cf') and op.name != 'vector.print' for op in operations):
        raise ValueError('scalar diagnostics require the late LLVM/CF boundary')
    if not prints:
        return {'prints':0,'assertions':_diag_lower(module)}
    punctuation = {ir.Attribute.parse('#vector.punctuation<' + key + '>'): value
        for key,value in [('no_punctuation',''),('newline','\n'),('open','( '),('close',' )'),('comma',', ')]}
    plans = []
    for op in prints:
        if op.results or op.regions or len(op.operands) > 1:
            raise ValueError('scalar diagnostics require a standard scalar/string print')
        string = ir.StringAttr(op.attributes['stringLiteral']).value if 'stringLiteral' in op.attributes else None
        punct = op.attributes.get('punctuation',ir.Attribute.parse('#vector.punctuation<newline>'))
        if punct not in punctuation or string is not None and (op.operands or punctuation[punct] != '\n'):
            raise ValueError('scalar diagnostics have incompatible string/punctuation fields')
        if op.operands and str(op.operands[0].type) not in ('i1','i8','i16','i32','i64','f32','f64'):
            raise ValueError('scalar diagnostics require signless integers or widened f32/f64 values')
        # Decode all source strings before edits; non-UTF-8 diagnostics have
        # a different contract. NUL remains an ordinary byte within UTF-8.
        if string is not None:
            string.encode('utf-8')
        plans.append((op,string,punctuation[punct]))
    for op in operations:
        if op.name == 'cf.assert':
            ir.StringAttr(op.attributes['msg']).value.encode('utf-8')
    symbols = {ir.StringAttr(op.attributes['sym_name']).value:op for op in module.body.operations if 'sym_name' in op.attributes}
    declarations = ir.Module.parse('module {llvm.func @putchar(i32)->i32 llvm.func @printf(!llvm.ptr,...)->i32 llvm.func @abort()}')
    for declaration in declarations.body.operations:
        name = ir.StringAttr(declaration.attributes['sym_name']).value
        if name in symbols and str(symbols[name]).strip() != str(declaration).strip():
            raise ValueError('scalar diagnostic runtime symbol has an incompatible definition: '+name)
    def fresh(stem):
        for index in range(len(symbols)+len(plans)*2+10):
            name = stem+str(index)
            if name not in symbols:
                symbols[name] = None
                return name
        raise ValueError('scalar diagnostic symbol budget exhausted')
    byte_helper = fresh('__merlin_print_bytes_')
    integer_helper = fresh('__merlin_print_i64_')
    float_helper = fresh('__merlin_print_f64_')
    integer_format,float_format = fresh('__merlin_print_integer_format_'),fresh('__merlin_print_float_format_')
    strings = {}
    for _,string,punct in plans:
        literal = string if string is not None else punct
        if literal and literal not in strings:
            strings[literal] = fresh('__merlin_print_literal_')
    helper_ir = r'''
module {
  llvm.mlir.global private constant @INTEGER_FORMAT("%lld\00") : !llvm.array<5 x i8>
  llvm.mlir.global private constant @FLOAT_FORMAT("%f\00") : !llvm.array<3 x i8>
  llvm.func internal @BYTE_HELPER(%message:!llvm.ptr,%length:i64) {
    %zero=llvm.mlir.constant(0:i64):i64
    %one=llvm.mlir.constant(1:i64):i64
    llvm.br ^loop(%zero:i64)
  ^loop(%index:i64):
    %more=llvm.icmp "ult" %index,%length:i64
    llvm.cond_br %more,^byte,^done
  ^byte:
    %address=llvm.getelementptr %message[%index]:(!llvm.ptr,i64)->!llvm.ptr,i8
    %value=llvm.load %address : !llvm.ptr -> i8
    %character=llvm.zext %value:i8 to i32
    %written=llvm.call @putchar(%character):(i32)->i32
    %next=llvm.add %index,%one:i64
    llvm.br ^loop(%next:i64)
  ^done:
    llvm.return
  }
  llvm.func internal @INTEGER_HELPER(%value:i64) {
    %format=llvm.mlir.addressof @INTEGER_FORMAT:!llvm.ptr
    %written=llvm.call @printf(%format,%value) vararg(!llvm.func<i32 (ptr,...)>):(!llvm.ptr,i64)->i32
    llvm.return
  }
  llvm.func internal @FLOAT_HELPER(%value:f64) {
    %format=llvm.mlir.addressof @FLOAT_FORMAT:!llvm.ptr
    %written=llvm.call @printf(%format,%value) vararg(!llvm.func<i32 (ptr,...)>):(!llvm.ptr,f64)->i32
    llvm.return
  }
  llvm.func @putchar(i32)->i32
  llvm.func @printf(!llvm.ptr,...)->i32
  llvm.func @abort()
}
'''
    for key,value in [('INTEGER_FORMAT',integer_format),('FLOAT_FORMAT',float_format),('BYTE_HELPER',byte_helper),
                       ('INTEGER_HELPER',integer_helper),('FLOAT_HELPER',float_helper)]:
        helper_ir = helper_ir.replace(key,value)
    prepared = ir.Module.parse(helper_ir)
    template = next(op for op in _diag_walk(prepared.operation) if op.name == 'llvm.call' and str(op.attributes['callee']) == '@putchar')
    call_attributes = {name:template.attributes[name] for name in template.attributes}
    literal_modules = []
    for value,name in strings.items():
        literal_modules.append(ir.Module.parse('module {llvm.mlir.global private constant @'+name+'('
            +str(ir.StringAttr.get(value))+'):!llvm.array<'+str(len(value.encode('utf-8')))+' x i8>}'))
    # Full source/runtime/schema preflight and helper parsing precede edits.
    for op in tuple(prepared.body.operations):
        name = ir.StringAttr(op.attributes['sym_name']).value
        if name not in ('putchar','printf','abort') or name not in symbols:
            op.operation.detach_from_parent()
            module.body.append(op.operation)
    for literal_module in literal_modules:
        op = literal_module.body.operations[0].operation
        op.detach_from_parent()
        module.body.append(op)
    i64,f64,pointer = ir.IntegerType.get_signless(64),ir.F64Type.get(),ir.Type.parse('!llvm.ptr')
    for op,string,punct in plans:
        attributes = dict(call_attributes)
        attributes.update({name:op.attributes[name] for name in op.attributes if name.startswith('prov.')})
        def call(name,args):
            selected = dict(attributes,callee=ir.FlatSymbolRefAttr.get(name),
                operandSegmentSizes=ir.DenseI32ArrayAttr.get([len(args),0]))
            ir.Operation.create('llvm.call',operands=args,attributes=selected)
        with ir.InsertionPoint(op),op.location:
            if op.operands:
                value = op.operands[0]
                if str(value.type) in ('i1','i8','i16','i32'):
                    instruction = 'llvm.zext' if str(value.type) == 'i1' else 'llvm.sext'
                    value = ir.Operation.create(instruction,operands=[value],results=[i64]).results[0]
                if str(value.type) == 'f32':
                    value = ir.Operation.create('llvm.fpext',operands=[value],results=[f64]).results[0]
                call(float_helper if value.type == f64 else integer_helper,[value])
            value = string if string is not None else punct
            if value:
                address = ir.Operation.create('llvm.mlir.addressof',results=[pointer],
                    attributes={'global_name':ir.FlatSymbolRefAttr.get(strings[value])}).results[0]
                count = ir.Operation.create('llvm.mlir.constant',results=[i64],
                    attributes={'value':ir.IntegerAttr.get(i64,len(value.encode('utf-8')))}).results[0]
                call(byte_helper,[address,count])
        op.erase()
    assertions = _diag_lower(module)
    module.operation.verify()
    return {'prints':len(plans),'assertions':assertions}
"""
SUPPORT = ASSERT_SUPPORT + PRINT_SUPPORT


def edit_pipeline(passes):
    if ASSERT_MARKER in passes or MARKER in passes:
        raise ValueError("scalar diagnostics already include literal assertion lowering")
    return [MARKER if item == ASSERT_MARKER else item for item in _edit(passes)]


def bind_runner(source, *, selected):
    if not selected:
        return source
    bound = _bind(source,selected=True)
    # Reuse the checked single native seam, replacing only the embedded
    # implementation and its one driver call.
    return bound.replace(ASSERT_SUPPORT,SUPPORT,1).replace(ASSERT_MARKER,MARKER).replace(
        "    _diag_lower(module)\n","    _scalar_diag_lower(module)\n",1)


def ensure_registered():
    from .impr_features import ImprFeature,known,register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE,action_class="PASS",edit_pipeline=edit_pipeline,
            description="Materialize standard scalar/string debug effects with fixed numeric formatting; default off."))
    return FEATURE
