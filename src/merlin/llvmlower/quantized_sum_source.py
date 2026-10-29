"""Bind a sole-use quantized producer to its exact following residual call.

The two numerical source entries remain separate and unchanged. This matcher
owns only the current tensor edge and observers; target implementation, private
workspace, numerical scalar proof and actual completion remain independent.
"""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class QuantizedSumSource:
    module: object
    rewrite: object
    producer: object
    consumer: object
    receipt: dict
    identity: str

    def revalidate(self):
        current = plan_quantized_sum_source(self.module, self.rewrite, self.producer, self.consumer)
        if self.identity != current.identity or self.receipt != current.receipt:
            raise ValueError('quantized sum original source, stages or observers changed')


def discover_quantized_sum_sources(module, rewrite):
    """Find legal current producer/residual edges without selecting a backend.

    Results are source-bound proposals. The caller can inspect the unchanged
    numerical stages and shape, try its target adapter, and select only admitted
    alternatives. Shared producer symbols are omitted because the ordinary
    grouping route identifies a producer by its unique current call. Source
    legality supplies no target effects, storage assignment or speedup claim.
    """
    from collections import Counter
    from xdsl.dialects import func
    from .device_rewrite import DeviceRewrite

    if type(rewrite) is not DeviceRewrite:
        raise ValueError('quantized discovery requires the current ordinary source rewrite')
    module.verify()
    calls = tuple(op for op in module.walk() if isinstance(op, func.CallOp))
    counts = Counter(op.callee.root_reference.data for op in calls)
    proposals = []
    for producer in calls:
        symbol = producer.callee.root_reference.data
        if (counts[symbol] != 1 or len(producer.res) != 1
                or rewrite.entries.get(symbol, {}).get('op') not in ('conv2d', 'matmul')):
            continue
        uses = tuple(producer.res[0].uses)
        if len(uses) != 1 or not isinstance(uses[0].operation, func.CallOp):
            continue
        consumer = uses[0].operation
        if rewrite.entries.get(consumer.callee.root_reference.data, {}).get('op') != 'residual_add':
            continue
        try:
            proposal = plan_quantized_sum_source(module, rewrite, producer, consumer)
        except ValueError:
            # A nonmatching source remains on its original unfused route.
            continue
        proposals.append(proposal)
    return tuple(proposals)


def plan_quantized_sum_source(module, rewrite, producer, consumer):
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import TensorType
    from .device_rewrite import DeviceRewrite
    from merlin.xdsl_dialects._common import text

    if type(rewrite) is not DeviceRewrite or not isinstance(producer, func.CallOp) or not isinstance(consumer, func.CallOp):
        raise ValueError('quantized sum requires its current original calls and ordinary source rewrite')
    module.verify()
    operations = tuple(module.walk())
    if (producer not in operations or consumer not in operations or producer.parent is not consumer.parent
            or len(producer.res) != 1 or len(consumer.res) != 1):
        raise ValueError('quantized sum calls must be current same-block single-result stages')
    ps, cs = (op.callee.root_reference.data for op in (producer, consumer))
    pe, ce = (deepcopy(rewrite.entries.get(symbol, {})) for symbol in (ps, cs))
    if (pe.get('op') not in ('conv2d', 'matmul') or ce.get('op') != 'residual_add'
            or pe.get('operand_dtype') != 'int8' or ce.get('operand_dtype') != 'int8'
            or pe.get('epilogue') not in (['bias_add','acc_scale'], ['bias_add','acc_scale','relu'], ['acc_scale'], ['acc_scale','relu'])
            or ce.get('epilogue') not in ([], ['relu'])):
        raise ValueError('quantized sum retains an original signed-i8 producer Q1 and residual Q2')
    for symbol, entry in ((ps,pe),(cs,ce)):
        program = rewrite.programs.get(symbol, {})
        if program.get('entry') != {k:v for k,v in entry.items() if k in program.get('entry', {})} or not program.get('entry'):
            raise ValueError('quantized sum numerical stages differ from original source programs')
    access = rewrite.arg_access
    if (access.get(ps) not in (('read','read','write'), ('read','read','read','write'))
            or access.get(cs) != ('read','read','write') or len(producer.arguments) != len(access[ps])
            or len(consumer.arguments) != 3):
        raise ValueError('quantized sum requires original borrowed inputs and fresh complete destinations')
    uses = list(producer.res[0].uses)
    if (len(uses) != 1 or uses[0].operation is not consumer or uses[0].index not in (0,1)
            or consumer.arguments[uses[0].index] is not producer.res[0]):
        raise ValueError('quantized producer intermediate has an unaccounted observer or different consumer')
    side = uses[0].index
    skip = consumer.arguments[1-side]
    output_type = producer.res[0].type
    if (not isinstance(output_type,TensorType) or str(output_type.element_type) != 'i8'
            or len(output_type.get_shape()) != 2 or any(v <= 0 for v in output_type.get_shape())
            or skip.type != output_type or consumer.res[0].type != output_type
            or producer.arguments[-1].type != output_type or consumer.arguments[-1].type != output_type):
        raise ValueError('quantized sum edge, skip and original outputs must have the same complete signed-i8 matrix type')
    for call in (producer,consumer):
        if not isinstance(call.arguments[-1].owner,tensor.EmptyOp) or len(list(call.arguments[-1].uses)) != 1:
            raise ValueError('quantized sum original destination must be fresh and unobserved')
    cursor=producer.next_op
    while cursor is not None and cursor is not consumer:
        if not isinstance(cursor,tensor.EmptyOp):
            raise ValueError('quantized sum cannot cross an intervening operation or effect')
        cursor=cursor.next_op
    if cursor is None:
        raise ValueError('quantized sum consumer does not follow its original producer')
    declarations={op.sym_name.data:op for op in module.body.block.ops if isinstance(op,func.FuncOp)}
    if any(symbol not in declarations or declarations[symbol].body.blocks
           or getattr(declarations[symbol].sym_visibility,'data',None) != 'private' for symbol in (ps,cs)):
        raise ValueError('quantized sum requires original private borrowed device declarations')
    for call,symbol in ((producer,ps),(consumer,cs)):
        buffers=rewrite.call_buffers.get(symbol,())
        if len(buffers) != len(call.arguments) or any(
            not isinstance(value.type,TensorType) or buffer.get('shape') != list(value.type.get_shape())
            or buffer.get('dtype') != str(value.type.element_type)
            for buffer,value in zip(buffers,call.arguments,strict=True)):
            raise ValueError('quantized sum original logical buffer bindings changed')
    record=dict(schema='merlin.quantized_sum_source.v1', source_host_sha256=hashlib.sha256(text(module,generic=True).encode()).hexdigest(),
        producer_ordinal=operations.index(producer),consumer_ordinal=operations.index(consumer),
        components=[ps,cs],entries=[pe,ce],producer_result_operand=side,shape=list(output_type.get_shape()),
        intermediate_observers=[],intermediate_uses=1,quantization='original producer Q1 then original residual sum/Q2/activation',
        source_programs=[deepcopy(rewrite.programs[ps]),deepcopy(rewrite.programs[cs])],
        effect_or_target_admission=False)
    return QuantizedSumSource(module,rewrite,producer,consumer,record,_digest(record))


@dataclass(frozen=True)
class QuantizedSumCommandSource:
    """The exact ordinary connected request, independent of host SSA custody."""
    command_buffer: dict
    receipt: dict
    identity: str

    def revalidate(self):
        current=plan_quantized_sum_commands(self.command_buffer)
        if self.receipt!=current.receipt or self.identity!=current.identity:
            raise ValueError('original connected command source or numerical stages changed')


def plan_quantized_sum_commands(command_buffer):
    """Check the complete ordinary request edge; retain both original stages.

    This establishes source dataflow only. The host's current SSA grouping,
    selected target numerical implementation, private byte layout and actual
    completion remain independently checked. No source is inferred from output.
    """
    if type(command_buffer)is not dict or type(command_buffer.get('commands'))is not list or type(command_buffer.get('tensors'))is not dict:
        raise ValueError('complete original connected command program required')
    commands=command_buffer['commands'];opcodes=[c.get('opcode')for c in commands]
    convolution=opcodes==['RES_PACK','CONV2D','EVICT','RESIDUAL_ADD']
    if not convolution and opcodes!=['RES_PACK','MATMUL_RESIDENT','COMMIT','EVICT','RESIDUAL_ADD']:
        raise ValueError('quantized request needs one original contraction and one ordered residual')
    packed,product=commands[:2];commit=product if convolution else commands[2]
    evict,residual=commands[-2:];po,co,ro=product['operands'],commit['operands'],residual['operands']
    handle=packed['operands']['dst'];weight=packed['operands']['src']
    if (packed.get('attributes',{})!={'layout':'packed_conv_rhs'if convolution else'packed_rhs'} or evict['operands']!={'handle':handle}
            or po.get('weight'if convolution else'rhs')!=handle
            or not convolution and co.get('src')!=po.get('dst') or set(ro)!={'lhs','rhs','dst'}):
        raise ValueError('connected request changed original weight residency or source edge')
    edge,output=co['dst'],ro['dst'];positions=[i for i,k in enumerate(('lhs','rhs'))if ro[k]==edge]
    if len(positions)!=1 or output==edge:raise ValueError('one sole intermediate operand and a distinct final destination required')
    side=positions[0];skip=ro[('rhs','lhs')[side]];tensors=command_buffer['tensors'];attrs=commit.get('attributes',{})
    if edge not in tensors or skip not in tensors or output not in tensors or skip==output:
        raise ValueError('connected source edge, skip and output must be explicit distinct tensors')
    shape=tensors[edge].get('shape')
    if (type(shape)is not list or len(shape)!=2 or any(type(v)is not int or v<=0 for v in shape)
            or any(tensors[n].get('shape')!=shape or tensors[n].get('dtype')!='i8'for n in (edge,skip,output))):
        raise ValueError('connected source must retain its complete same-shaped signed-i8 edge')
    epilogue=attrs.get('epilogue');sum_attrs=residual.get('attributes',{})
    if (epilogue not in (['acc_scale'],['acc_scale','relu'],['bias_add','acc_scale'],['bias_add','acc_scale','relu'])
            or attrs.get('output_dtype')!='i8' or sum_attrs.get('output_dtype')!='i8'
            or sum_attrs.get('epilogue')not in ([],['relu'])
            or type(sum_attrs.get('source_scales'))is not list or len(sum_attrs['source_scales'])!=3):
        raise ValueError('connected request must retain original Q1 and ordered residual Q2/activation')
    lhs=po['ifm'if convolution else'lhs'];inputs={lhs,weight}
    if'bias'in attrs:inputs.add(attrs['bias'])
    if (set(tensors)!=inputs|{edge,skip,output} or {edge,output}&inputs
            or any(n not in tensors for n in inputs)):
        raise ValueError('connected source has an unaccounted external tensor or intermediate observer')
    pe=dict(deepcopy(attrs),op='conv2d'if convolution else'matmul',operand_dtype='int8')
    ce=dict(deepcopy(sum_attrs),op='residual_add',operand_dtype='int8')
    record=dict(schema='merlin.quantized_sum_command_source.v1',source_command_buffer_sha256=_digest(command_buffer),
        source_kind='original ordinary connected command request; host current-SSA binding is separate',
        entries=[pe,ce],producer_result_operand=side,shape=deepcopy(shape),intermediate=edge,skip=skip,output=output,
        quantization='original producer Q1 then original ordered sum/Q2/activation',effect_or_target_admission=False)
    return QuantizedSumCommandSource(command_buffer,record,_digest(record))


def split_quantized_sum_commands(source):
    """Select the two literal original statements; no contraction is rebuilt."""
    if type(source)is not QuantizedSumCommandSource:raise ValueError('current checked connected command source required')
    source.revalidate();cb=source.command_buffer;commands=cb['commands'];record=source.receipt
    producer_names=set(cb['tensors'])-{record['skip'],record['output']}
    consumer_names={record['intermediate'],record['skip'],record['output']}
    common={k:deepcopy(cb[k])for k in ('abi_version','target')}
    first=dict(common,tensors={n:deepcopy(v)for n,v in cb['tensors'].items()if n in producer_names},commands=deepcopy(commands[:-1]))
    second=dict(common,tensors={n:deepcopy(v)for n,v in cb['tensors'].items()if n in consumer_names},commands=deepcopy(commands[-1:]))
    # The retained edge is an input only in the second original statement.
    second['tensors'][record['intermediate']]['role']='input'
    source.revalidate()
    return first,second
