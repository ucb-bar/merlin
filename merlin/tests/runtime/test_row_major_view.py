"""Logical view preserves independent exact values, shape and ownership."""
from copy import deepcopy
import pytest
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate


def program():
    return {'abi_version':'0.1','target':'t','tensors':{
        'A':{'shape':[2,3],'dtype':'i8','role':'input'},
        'B':{'shape':[3,2],'dtype':'i8','role':'weight'},
        'Y':{'shape':[4,1],'dtype':'i32','role':'output'}},'commands':[
        {'opcode':'MATMUL','operands':{'lhs':'A','rhs':'B','dst':'acc'}},
        {'opcode':'COMMIT','operands':{'src':'acc','dst':'Y'},'attributes':{'epilogue':[],'output_dtype':'i32'}},
        {'opcode':'ROW_MAJOR_VIEW','operands':{'src':'Y','dst':'Y'},'attributes':{'source_shape':[2,2],'shape':[4,1]}}]}


def test_exact_shape_values_and_no_added_traffic():
    p=program();inputs={'A':[[1,2,3],[-1,0,2]],'B':[[1,4],[2,5],[3,6]]}
    expected={'Y':[[14],[32],[5],[8]]}
    assert reference_outputs(p,inputs)==expected
    result=simulate(p,inputs)
    assert result['outputs']==expected
    raw=deepcopy(p);raw['commands'].pop();raw['tensors']['Y']['shape']=[2,2]
    unviewed=simulate(raw,inputs)
    assert {k:v for k,v in result['metrics'].items()if k!='command_count'}=={k:v for k,v in unviewed['metrics'].items()if k!='command_count'}


@pytest.mark.parametrize('change', ['extent','dtype','source','newslot','input','extra','bool'])
def test_unsupported_view_refuses(change):
    p=program();cmd=p['commands'][-1]
    if change=='extent':cmd['attributes']['shape']=[3,1]
    elif change=='dtype':p['tensors']['Y']['dtype']='i8'
    elif change=='source':cmd['attributes']['source_shape']=[1,4]
    elif change=='newslot':cmd['operands']['dst']='Z'
    elif change=='input':cmd['operands']={'src':'A','dst':'A'}
    elif change=='extra':cmd['attributes']['transpose']=True
    elif change=='bool':cmd['attributes']['shape']=[True,4]
    for engine in (reference_outputs,simulate):
        with pytest.raises(ValueError):engine(p)
