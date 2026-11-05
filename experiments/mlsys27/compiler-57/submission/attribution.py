"""Source-bound task inventory and LLVM operation ownership, not a certificate."""
from io import StringIO
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from llvm_context import make_llvm_context


def prepare(cb):
    params=cb['params']
    inv=params.get('source_inventory')
    if not inv:return None
    payload=params['linalg_inventory']['ops']
    source_indices=params['payload_source_op_indices']
    temporary_names={t['tensor'] for t in params.get('compiler_temporaries',[])}
    # Source constants materialized as pointer-ABI tensors need an actual
    # source-owned producer. A module constant alone does not initialize a
    # mutable argument supplied by the caller. Give each such value an
    # explicit initialization task before its consumers.
    constants=[]
    for name, value in params.get('constant_tensors',{}).items():
        if name not in temporary_names:
            constants.append({'opcode':'DEVICE_CONSTANT','operands':{'dst':name},
                              'attributes':{'source_op_index':value['source_op_index']}})
    cb['commands']=sorted(constants+cb['commands'],
                          key=lambda c:c.get('attributes',{}).get('source_op_index',-1))
    values=[]
    for rec in payload:
        for j,spec in enumerate(rec['results']):
            name=f'value_{rec["id"]}_{j}'
            for k,ret in enumerate(params['linalg_inventory']['returns']):
                if ret['source']==['op',rec['id']] and ret.get('result_index',0)==j:
                    name=cb['kernel_abi']['outputs'][k]
            if name in cb['tensors']:
                values.append({'op_index':int(source_indices.get(rec['id'],source_indices.get(str(rec['id'])))),'result_index':j,'tensor':name})
    if params.get('source_values'):values=params['source_values']
    existing={a['tensor'] for a in cb['kernel_abi']['args']}
    cb['kernel_abi']['args'].extend({'tensor':n,'access':'write'} for n in cb['tensors'] if n not in existing and n not in temporary_names)
    ordered=sorted({int(c.get('attributes',{}).get('source_op_index',-1)) for c in cb['commands']} - {-1})
    if not ordered:raise ValueError('source has no implemented tasks')
    tasks=[]
    previous=-1
    for i,s in enumerate(ordered):
        end=s if i+1<len(ordered) else inv['source_op_count']-1
        indices=list(range(previous+1,end+1));previous=end
        commands=[c for c in cb['commands'] if c.get('attributes',{}).get('source_op_index')==s]
        reads=[];writes=[]
        for c in commands:
            for k,n in c.get('operands',{}).items():
                if n in cb['tensors'] and n not in temporary_names:
                    target=writes if k.startswith('dst') else reads
                    if n not in target:target.append(n)
        tasks.append({'task_index':i,'kind':'host' if any(c['opcode']=='HOST_LINALG' for c in commands) else 'device',
                      'source_op_indices':indices,'source_region_ids':sorted({inv['operations'][j]['region_id'] for j in indices if inv['operations'][j]['region_id']}),
                      'instruction_start':i,'instruction_end':i+1,'reads':reads,'writes':writes,'payload_source_index':s})
    plan={'schema':'mixed_program_plan_v1','source_sha256':inv['source_sha256'],'source_op_count':inv['source_op_count'],
          'tasks':tasks,'schedule_instruction_count':len(tasks),'prologue_instruction_range':[0,0],
          'epilogue_instruction_range':[len(tasks),len(tasks)],'entry_bindings':[f'arg_{i}' for i in range(len(inv['arguments']))],
          'source_values':values,'output_bindings':cb['kernel_abi']['outputs'],'compiler_temporaries':[]}
    params['global_program_plan']=plan
    return {t['payload_source_index']:t['task_index'] for t in tasks}


def marker(g,task,source):
    # This comment is removed after LLVM import; it only transports task order
    # through the stock LLVM importer without relying on discarded LLVM metadata.
    g.emit(f'call void asm sideeffect "# merlin.task {task} {source}", "~{{memory}}"()')


def annotate(text,cb):
    module=Parser(make_llvm_context(),text).parse_module()
    module.verify()
    for operation in module.walk():
        if operation.name in ('llvm.fpext','llvm.sitofp','llvm.uitofp','llvm.fptrunc'):
            flags=operation.properties.get('fastmathFlags')
            if flags is not None and str(flags)=='#llvm.fastmath<none>':
                operation.properties.pop('fastmathFlags')
    current=-1;source=None
    for fn in module.body.block.ops:
        if fn.name!='llvm.func' or fn.sym_name.data!='kernel_entry':continue
        for block in fn.body.blocks:
            for op in list(block.ops):
                asm=getattr(op,'asm_string',None)
                if asm and asm.data.startswith('# merlin.task '):
                    _,_,task,src=asm.data.split()
                    current=int(task);source=int(src)
                    block.erase_op(op)
                    continue
                op.attributes['merlin.global_task']=IntegerAttr(current,64)
                if current>=0:
                    op.attributes['merlin.source_op_index']=IntegerAttr(source,64)
                    region=cb['params']['source_inventory']['operations'][source]['region_id']
                    if region:op.attributes['prov.region_id']=StringAttr(region)
    module.verify()
    out=StringIO();Printer(stream=out,print_generic_format=True).print_op(module)
    return out.getvalue()+'\n'
