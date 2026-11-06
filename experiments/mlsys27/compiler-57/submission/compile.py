#!/usr/bin/env python3
"""Out-of-tree compiler entrypoints."""
import sys
import json
from pathlib import Path
from emitter import Emitter
from planbuilder import prepare,finish


def main():
    args=sys.argv[1:]
    sources=[x for x in args if not x.startswith('-')]
    if not sources: raise ValueError('expected input MLIR path')
    text=Path(sources[-1]).read_text()
    from frontend import parse_source
    cb=parse_source(text)
    if '--verify-diagnostics' in args: return
    from normalize_affine import normalize as normalize_affine
    from normalize_pooling import normalize as normalize_pooling
    from normalize_reduction import normalize as normalize_reduction
    from normalize_extension import normalize as normalize_extension
    from normalize_seeds import normalize as normalize_seeds
    from normalize_residency import normalize as normalize_residency
    cb=normalize_residency(normalize_seeds(normalize_pooling(normalize_affine(normalize_reduction(normalize_extension(cb))))))
    ordered=[];dirty=set()
    for command in cb.get('commands',[]):
        operands=command.get('operands',{})
        attrs=command.get('attributes',{})
        reads={v for k,v in operands.items() if not k.startswith('dst')}
        reads.update(v for k,v in attrs.items() if k in ('bias','accumulator_seed') and isinstance(v,str))
        if reads & dirty:
            fence={'opcode':'FENCE'}
            if '_source_op_index' in command:fence['_source_op_index']=command['_source_op_index']
            ordered.append(fence);dirty.clear()
        ordered.append(command)
        if command['opcode']=='FENCE':dirty.clear()
        dirty.update(v for k,v in operands.items() if k.startswith('dst') and v in cb['tensors'])
    cb['commands']=ordered
    from frontend import attach_source_metadata
    cb=attach_source_metadata(cb)
    plan=prepare(cb,text)
    e=Emitter(cb)
    ranges={}
    state={}
    from importlib import import_module
    lowerers=[]
    for module in ('integer_reduction','integer_contraction','movement','contraction','readout','affine_reduction','pooling','quantized_float','integer_arithmetic','promoted_integer','host_lowering'):
        try: lowerers.append(import_module(module).lower)
        except ModuleNotFoundError: pass
    for c in cb.get('commands',[]) if not cb.get('declined') else []:
        e.current_task=c.get('_task_index')
        e.current_source=c.get('_source_op_index')
        e.current_region=c.get('_source_region_id')
        e.current_stage=c.get('_composition_stage')
        begin=e.schedule_count
        if c['opcode']=='FENCE':
            e.fence()
        else:
            try:
                handled=any(lower(e,c,state) for lower in lowerers)
            except ValueError as error:
                cb['declined']={'op':c['opcode'],'reason':str(error) or 'Typed lowering has no supported representation'}
                cb['commands']=[]
                break
            if not handled:
                cb['declined']={'op':c['opcode'],'reason':'Compiler lowering not yet implemented for this typed operation'}
                cb['commands']=[]
                break
        if c['opcode']=='HOST_REGION':e.fence()
        if c['opcode']=='FILL':
            value=c.get('attributes',{}).get('value_code',0)
            state.setdefault('value_bounds',{})[c['operands']['dst']]=(value,value)
        if c['opcode']=='MOVEMENT' and c.get('attributes',{}).get('form','identity') in ('identity','reshape','permute'):
            src,dst=c['operands']['src'],c['operands']['dst']
            if e.tensors[src]['dtype']=='i8':state.setdefault('value_bounds',{})[dst]=(-128,127)
            elif src in state.get('value_bounds',{}):state['value_bounds'][dst]=state['value_bounds'][src]
        if e.current_task is not None:
            tid=e.current_task
            ranges[tid]=[ranges.get(tid,[begin,begin])[0],e.schedule_count]
    if cb.get('declined'):
        cb.setdefault('params',{})['lane_placement']=[]
        cb['params'].pop('global_program_plan',None)
        cb.pop('kernel_abi',None)
        e=Emitter(cb)
    e.current_task=None
    e.current_source=None
    e.current_region=None
    e.current_stage=None
    lowered=e.render()
    if not cb.get('declined'):finish(plan,e,ranges)
    cb.setdefault('params',{}).setdefault('lane_placement',[{'lane':'systolic_mesh','region':'program'}] if not cb.get('declined') else [])
    cb['commands']=[{k:v for k,v in c.items() if not k.startswith('_')} for c in cb['commands'] if c['opcode']!='HOST_REGION']
    cb={k:v for k,v in cb.items() if not k.startswith('_')}
    for a in args:
        if a.startswith('--emit-command-buffer='):
            Path(a.split('=',1)[1]).write_text(json.dumps(cb,indent=2))
    if '--emit-target-artifact' in args:
        print(lowered)
    elif not any(x.startswith('--emit-command-buffer=') for x in args):
        print(lowered)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'compiler error: {exc}',file=sys.stderr)
        sys.exit(1)
