"""Reject stale mathematical metadata before source placement or golden work."""
import copy

import pytest

from merlin.targetgen.contract.interface_emit import emit_interface_mlir
from merlin_experiments.corpus.scalability_intake import check_convolution_metadata


@pytest.fixture
def written(tmp_path):
    attrs={'kernel':[2,3,8,5],'stride':[1,2],'padding':[1,0,0,1],
           'dilation':[1,1],'layout':'nhwc','epilogue':['bias_add'],'output_dtype':'i8','bias':'B'}
    tensors={'X':{'shape':[2,4,5,8],'dtype':'i8','role':'input'},
             'W':{'shape':[48,5],'dtype':'i8','role':'weight'},
             'B':{'shape':[5],'dtype':'i32','role':'bias'},
             'Y':{'shape':[16,5],'dtype':'i8','role':'output'}}
    cb={'abi_version':'0.1','target':'toy','tensors':tensors,'commands':[
        {'opcode':'RES_PACK','operands':{'src':'W','dst':'R'},'attributes':{'layout':'packed_conv_rhs'}},
        {'opcode':'CONV2D','operands':{'ifm':'X','weight':'R','dst':'Y'},'attributes':attrs},
        {'opcode':'EVICT','operands':{'handle':'R'}}]}
    source=tmp_path/'source.mlir';source.write_text(emit_interface_mlir(cb))
    cap={'operation':{'op':'conv2d','attributes':dict(attrs,kh=2,kw=3,ci=8,co=5)}}
    return cap,source


def test_agreeing_and_absent_duplicate_dimensions_preserve_source(written):
    cap,path=written;before=path.read_bytes();saved=copy.deepcopy(cap)
    check_convolution_metadata(cap,path)
    assert cap==saved and path.read_bytes()==before
    for key in ('kh','kw','ci','co'):cap['operation']['attributes'].pop(key)
    check_convolution_metadata(cap,path)


@pytest.mark.parametrize('key,value',[('ci',4),('kh',1),('kw',1),('co',4),
    ('kernel',[2,3,4,5]),('stride',[2,2]),('padding',[0,0,0,0]),
    ('dilation',[2,1]),('layout','nchw'),('epilogue',[]),('output_dtype','i32'),('bias','other')])
def test_source_metadata_conflicts_refuse_without_repair(written,key,value):
    cap,path=written;cap['operation']['attributes'][key]=value;saved=copy.deepcopy(cap)
    with pytest.raises(ValueError,match=key):check_convolution_metadata(cap,path)
    assert cap==saved
