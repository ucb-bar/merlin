"""Explicit current-source Q1 producer/sum grouping for ordinary logical shims.

Both original entries and numerical programs travel intact. Target code owns
its temporary and device effects; grouping supplies neither qualification nor
an allocation. A shared residual callee retains all unrelated call instances.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
import hashlib,json
from pathlib import Path


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class QuantizedSumRoute:
    source_pairs: tuple[tuple[str,str], ...]

    def record(self):
        if (type(self.source_pairs) is not tuple or not self.source_pairs or len(self.source_pairs)>128
                or any(type(p) is not tuple or len(p)!=2 or any(type(s) is not str or not s for s in p)
                       or p[0]==p[1] for p in self.source_pairs)
                or len({p[0]for p in self.source_pairs})!=len(self.source_pairs)):
            raise ValueError('quantized sum needs distinct explicit original producer/consumer source pairs')
        return dict(schema='merlin.quantized_sum_route.v1',source_pairs=[list(p)for p in self.source_pairs])


def freeze_route(device):
    selected=getattr(device,'quantized_sum',None)
    if selected is None:return None
    if type(selected) is not QuantizedSumRoute or getattr(device,'granularity',None)!='group' or any(
            getattr(device,k,None) is not None for k in ('connected_spatial','sibling_source','exact_selection',
                'catalog_manifest','catalog_object','catalog_builder','prepared_transform','post_offload_transform','offload_selection')):
        raise ValueError('quantized sum requires its original typed group route without opaque source/ABI rewrites')
    return selected,selected.record()


def require_route(device,frozen):
    current=freeze_route(device)
    if (current is None)!=(frozen is None) or current is not None and (
            current[0] is not frozen[0] or current[1]!=frozen[1]):
        raise ValueError('quantized sum original source selection changed')
    return None if current is None else current[0]


@dataclass(frozen=True)
class QuantizedSumSelection:
    module: object
    rewrite: object
    receipt: dict


def group_quantized_sum(module,rewrite,route):
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,StringAttr,UnitAttr
    from xdsl.rewriter import Rewriter
    from merlin.llvmlower.device_offload import DeviceRewrite,Routed
    from merlin.xdsl_dialects._common import text
    from .quantized_sum_source import plan_quantized_sum_source

    if type(rewrite) is not DeviceRewrite or type(route) is not QuantizedSumRoute:
        raise ValueError('original ordinary source rewrite and quantized sum route required')
    route.record();module.verify();before=text(module,generic=True);selected=module.clone()
    maps={name:deepcopy(getattr(rewrite,name))for name in
        ('signatures','programs','entries','arg_access','call_buffers','expected_interfaces')}
    records=[];added=[];removed=set();remaining_routed=list(rewrite.routed)
    for ps,cs in route.source_pairs:
        calls=[o for o in selected.walk()if isinstance(o,func.CallOp)]
        producers=[o for o in calls if o.callee.root_reference.data==ps]
        if len(producers)!=1:raise ValueError('selected quantized producer must name one current source call')
        producer=producers[0];uses=list(producer.res[0].uses)
        if len(uses)!=1 or not isinstance(uses[0].operation,func.CallOp) or uses[0].operation.callee.root_reference.data!=cs:
            raise ValueError('selected quantized producer lost its exact original sum consumer')
        consumer=uses[0].operation
        originals=[]
        for call,symbol in ((producer,ps),(consumer,cs)):
            current_calls=[o for o in calls if o.callee.root_reference.data==symbol]
            inventory=[r for r in remaining_routed if r.symbol==symbol]
            if len(inventory)!=len(current_calls):
                raise ValueError('quantized routing records do not cover every current call instance')
            originals.append(inventory[current_calls.index(call)])
        for row in originals:remaining_routed.remove(row)
        source=plan_quantized_sum_source(selected,rewrite,producer,consumer);source.revalidate()
        if ps in removed or cs in removed:raise ValueError('quantized source pairs overlap')
        pe,ce=source.receipt['entries'];side=source.receipt['producer_result_operand'];skip=1-side
        entry=dict(schema='connected_quantized_sum_entry_v1',kind='model_slice',op='quantized_producer_sum',
            operand_dtype='int8',source_role=pe.get('source_role','model_derived'),
            source_reference=pe.get('source_reference','explicit original quantized call edge'),
            components=deepcopy([pe,ce]),producer_result_operand=side)
        binding=_sha(json.dumps(dict(entry=entry,source=source.receipt),sort_keys=True))
        symbol='merlin_quantized_sum_'+binding[:16]
        if any(isinstance(o,func.FuncOp)and o.sym_name.data==symbol for o in selected.body.block.ops):
            raise ValueError('quantized source group symbol collision')
        operands=[*producer.arguments[:-1],consumer.arguments[skip],consumer.arguments[-1]]
        access=(*rewrite.arg_access[ps][:-1],'read','write')
        declaration=func.FuncOp.external(symbol,[v.type for v in operands],[consumer.res[0].type])
        declaration.attributes['llvm.emit_c_interface']=UnitAttr()
        declaration.properties['arg_attrs']=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(a)})for a in access])
        selected.body.block.add_op(declaration)
        replacement=func.CallOp(symbol,operands,[consumer.res[0].type]);Rewriter.replace_op(consumer,replacement)
        destination=producer.arguments[-1].owner;Rewriter.erase_op(producer);Rewriter.erase_op(destination)
        entry['name']=symbol
        maps['signatures'][symbol]=();maps['entries'][symbol]=entry
        maps['programs'][symbol]=dict(schema='connected_quantized_sum_program_v1',components=deepcopy([rewrite.programs[ps],rewrite.programs[cs]]),producer_result_operand=side)
        maps['arg_access'][symbol]=access
        maps['call_buffers'][symbol]=[*deepcopy(rewrite.call_buffers[ps][:-1]),deepcopy(rewrite.call_buffers[cs][skip]),deepcopy(rewrite.call_buffers[cs][-1])]
        added.append(Routed(symbol,(),(),('i8','i8','i8'),source_node_ids=tuple(n for r in originals for n in r.source_node_ids)))
        # Retire declarations only after their last actual current call leaves.
        remaining={o.callee.root_reference.data for o in selected.walk()if isinstance(o,func.CallOp)}
        for old in (ps,cs):
            if old not in remaining:
                removed.add(old)
                for mapping in maps.values():mapping.pop(old,None)
                decl=next(o for o in selected.body.block.ops if isinstance(o,func.FuncOp)and o.sym_name.data==old)
                Rewriter.erase_op(decl)
        records.append(dict(symbol=symbol,components=[ps,cs],source_binding_sha256=binding,
            original_source=source.receipt,logical_argument_types=[str(v.type)for v in operands],
            result_argument=len(operands)-1,quantization='original producer Q1, original ordered sum/Q2/activation',
            external_intermediate_observers=[]))
    updated=replace(rewrite,**maps,routed=tuple(remaining_routed)+tuple(added))
    selected.verify()
    if text(module,generic=True)!=before:raise ValueError('quantized grouping changed original source')
    return QuantizedSumSelection(selected,updated,dict(schema='merlin.quantized_sum_selection.v1',
        source_host_sha256=_sha(before),selected_host_sha256=_sha(text(selected,generic=True)),groups=records,
        target_execution=False,numerical_evaluation=False))


def prepare_host(device,route,prepared,rewrite,directory):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects._common import text
    selected=group_quantized_sum(parse_mlir_file(prepared),rewrite,route)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'host.mlir';path.write_text(text(selected.module,generic=True)+'\n')
    (directory/'receipt.json').write_text(json.dumps(selected.receipt,sort_keys=True,indent=2)+'\n')
    return path,selected.rewrite


def quantized_sum_interface(entry,binder,*,source_reference):
    from .whole_program import group_interface
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir,parse_interface_mlir
    if entry.get('schema')!='connected_quantized_sum_entry_v1' or entry.get('producer_result_operand')not in (0,1):
        raise ValueError('two unchanged original quantized source entries and operand order required')
    parts=entry.get('components')
    if type(parts)is not list or len(parts)!=2:raise ValueError('exactly two original quantized source stages required')
    first,second=[deepcopy(parse_interface_mlir(group_interface(e,binder,source_reference=source_reference)))for e in parts]
    if (first['target'],first['abi_version'])!=(second['target'],second['abi_version']):raise ValueError('quantized source target or ABI differs')
    if [c['opcode']for c in first['commands']]not in (['RES_PACK','CONV2D','EVICT'],['RES_PACK','MATMUL_RESIDENT','COMMIT','EVICT']) or [c['opcode']for c in second['commands']]!=['RESIDUAL_ADD']:
        raise ValueError('quantized source must retain one original contraction and one loaded sum')
    produced=next(c for c in first['commands']if c['opcode']in ('CONV2D','COMMIT'))
    residual=second['commands'][0];edge=produced['operands']['dst'];key=('lhs','rhs')[entry['producer_result_operand']];source=residual['operands'][key]
    a,b=first['tensors'][edge],second['tensors'][source]
    if a.get('dtype')!='i8' or any(a.get(k)!=b.get(k)for k in ('shape','dtype')):raise ValueError('quantized source edge shape or format differs')
    for prefix,cb in [('producer_',first),('sum_',second)]:
        cb['tensors']={prefix+n:s for n,s in cb['tensors'].items()}
        for c in cb['commands']:
            c['operands']={k:prefix+v for k,v in c['operands'].items()}
            if'bias'in c.get('attributes',{}):c['attributes']['bias']=prefix+c['attributes']['bias']
    second['tensors'].pop('sum_'+source);residual['operands'][key]='producer_'+edge
    return emit_interface_mlir(dict(abi_version=first['abi_version'],target=first['target'],
        tensors={**first['tensors'],**second['tensors']},commands=[*first['commands'],*second['commands']]))
