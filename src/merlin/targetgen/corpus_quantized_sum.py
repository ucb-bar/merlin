"""Two original quantized source operations under the ordinary capsule binder."""
from copy import deepcopy


def build(entry,binding):
    from merlin.llvmlower.quantized_sum_route import quantized_sum_interface
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from .corpus_spatial_pair import _composition_modes
    components=entry.get('components')
    if type(components)is not list or len(components)!=2 or components[0].get('op')not in ('conv2d','matmul') or components[1].get('op')!='residual_add':
        raise ValueError('quantized pair requires its original contraction and ordered residual entries')
    originals=[CS.build({**e,'source_reference':entry['source_reference']},binding)[0]for e in components]
    interface=quantized_sum_interface(entry,lambda _:binding,source_reference=entry['source_reference'])
    cb=parse_interface_mlir(interface);out=cb['commands'][-1]['operands']['dst'];side=entry['producer_result_operand']
    consumed=('lhs','rhs')[side];skip=cb['commands'][-1]['operands'][('rhs','lhs')[side]]
    inputs=[{**deepcopy(row),'name':'producer_'+row['name']}for row in originals[0]['inputs']]
    original_skip=originals[1]['operation']['attributes'].get(('rhs','lhs')[side])
    for row in originals[1]['inputs']:
        if row['name']==original_skip:inputs.append({**deepcopy(row),'name':skip})
    if len(inputs)!=len(originals[0]['inputs'])+1:raise ValueError('original residual skip input is not explicit')
    producer,residual=[deepcopy(o['operation'])for o in originals]
    for key in ('ifm','lhs','weight','bias','out'):
        if key in producer['attributes']:producer['attributes'][key]='producer_'+producer['attributes'][key]
    for key in ('lhs','rhs','out'):
        residual['attributes'][key]='sum_'+residual['attributes'][key]
    internal=cb['commands'][-1]['operands'][consumed]
    residual['attributes'][consumed]=internal
    return dict(name=entry['name'],kind=entry['kind'],label=entry.get('label','model'),
        source_role=entry['source_role'],source_reference=entry['source_reference'],interface_mlir='capsule.interface.mlir',
        inputs=inputs,operation=dict(op='quantized_producer_sum',attributes=dict(
            producer=producer,residual=residual,producer_result_operand=side,
            intermediate_shape=cb['tensors'][internal]['shape'],intermediate_dtype=cb['tensors'][internal]['dtype'],out=out)),
        numeric_policy=deepcopy(originals[1]['numeric_policy']),expected=dict(
            instruction_classes=list(dict.fromkeys(c for o in originals for c in o['expected']['instruction_classes'])),
            modes=_composition_modes(*(o['expected']['modes']for o in originals))),
        required_oracle_tiers=list(binding.tiers),vcs='optional',firesim='optional'),interface
