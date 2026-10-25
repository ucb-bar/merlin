"""Ordinary public source owner binds metadata through actual compiler use."""
import json
import pytest
import yaml
from merlin.targetgen.public_source_abi import public_abi_source
from merlin.common.invocation_record import verify

SOURCE='''module attributes {prov.level = "linalg-on-tensors"} {func.func @forward(%x:tensor<2x3xi32>)->tensor<2x3xi32>{func.return %x:tensor<2x3xi32>}}'''


def fixture(tmp_path):
    cap={'label':'public','interface_mlir':'capsule.interface.mlir','inputs':[{'name':'A','shape':[2,3],'dtype':'i32','role':'input'}],
         'component_program':{'schema':'merlin.component_program.v2','outputs':[{'name':'Y','shape':[2,3],'dtype':'i32'}]}}
    source=tmp_path/'capsule.interface.mlir';source.write_text(SOURCE)
    (tmp_path/'capsule.yaml').write_text(yaml.safe_dump(cap));cap['__dir__']=str(tmp_path)
    return cap,source,tmp_path/'generated'


def test_actual_source_transition_is_pinned(tmp_path):
    cap,source,out=fixture(tmp_path)
    with public_abi_source(cap,source,out)as normalized:
        assert source.read_text()==SOURCE and normalized!=source
        assert 'merlin.logical_outputs = ["Y"]'in normalized.read_text()
        record=json.loads((normalized.parent/'source_binding.json').read_text())
        assert record['body_transformations']==0 and record['original_source']==str(source)
    observation=next(out.rglob('invocation.json'))
    assert verify(observation)['outcome']=='returned'


@pytest.mark.parametrize('mutation',['metadata','source','normalized','in_memory'])
def test_transition_mutation_refuses(tmp_path,mutation):
    cap,source,out=fixture(tmp_path)
    with pytest.raises(ValueError):
        with public_abi_source(cap,source,out)as normalized:
            if mutation=='metadata':(tmp_path/'capsule.yaml').write_text('label: public\n')
            elif mutation=='source':source.write_text(SOURCE+'\n')
            elif mutation=='normalized':normalized.write_text(normalized.read_text()+'\n')
            else:cap['inputs'][0]['name']='other'


def test_wrong_loaded_metadata_and_type_refuse(tmp_path):
    cap,source,out=fixture(tmp_path);cap['inputs'][0]['name']='fake'
    with pytest.raises(ValueError,match='loaded'): 
        with public_abi_source(cap,source,out):pass
    cap,source,out=fixture(tmp_path);cap['inputs'][0]['shape']=[3,2]
    (tmp_path/'capsule.yaml').write_text(yaml.safe_dump({k:v for k,v in cap.items()if k!='__dir__'}))
    with pytest.raises(ValueError,match='shape'):
        with public_abi_source(cap,source,out):pass


def test_private_or_undeclared_program_is_unchanged(tmp_path):
    cap,source,out=fixture(tmp_path);cap['label']='hidden'
    with public_abi_source(cap,source,out)as selected:assert selected==source
    assert not out.exists()


def test_ordinary_entrypoint_route_uses_the_live_owner(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen import capsule_common
    cap,source,out=fixture(tmp_path);out.mkdir();seen=[]
    def lower(pkg,selected,generated,**kwargs):
        seen.append(selected)
        assert selected!=source and 'merlin.logical_io'in selected.read_text()
        return {'commands':[]},'artifact'
    monkeypatch.setattr(capsule_common,'lower_interface',lower)
    pkg=SimpleNamespace(tool=source)
    assert capsule_common.run_entrypoints(pkg,tmp_path,cap,SimpleNamespace(generated=out),contract=None,timeout=1,fourth_output_name='result.mlir')==(pkg,{'commands':[]},'artifact')
    assert len(seen)==1
