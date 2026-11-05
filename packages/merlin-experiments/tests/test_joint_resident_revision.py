import copy
import hashlib
import json
import pytest
import yaml
from merlin_experiments.phase1.joint_resident_revision import verify
from merlin_experiments.phase2.contracts import document_sha256


def fixture(tmp_path, m=16):
    text=f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"}} {{
 %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<32x16xi8>
 %A = merlin_iface.tensor {{name = "A", role = "input"}} : tensor<{m}x32xi8>
 %R = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<32x16xi8>) -> !merlin_iface.resident
 %acc = merlin_iface.matmul %A, %R : (tensor<{m}x32xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
 %Y = merlin_iface.commit %acc {{name = "Y", epilogue = [], output_dtype = "i32"}} : (!merlin_iface.acc<i32>) -> tensor<{m}x16xi32>
 merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}}'''
    def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    old=dict(name='control',interface_mlir='program.mlir',expected={'modes':{'resident_reuse':True}},
             software_screen={'status':'admitted','program_sha256':hashlib.sha256(text.encode()).hexdigest()},
             numeric_policy={'compare':'exact_int'},required_oracle_tiers=['L0','L1','L2','L3'])
    new=copy.deepcopy(old);new['expected']['modes']['resident_reuse']=False
    ev=dict(schema='merlin.single_use_resident_review.v1',name='control',array='mesh')
    for role,doc in [('before',old),('after',new)]:
        root=tmp_path/role;root.mkdir();(root/'program.mlir').write_text(text)
        (root/'capsule.yaml').write_text(yaml.safe_dump(doc));ev[role]=pin(root/'capsule.yaml')
    f=tmp_path/'facts.json';f.write_text(json.dumps({'facts':{'arrays':[dict(name='mesh',rows=16,cols=16,corroborated=True)]}}));ev['facts']=pin(f)
    return ev,{'control':{'capsule_sha256':document_sha256(old)}},new


def test_single_output_tile_correction(tmp_path):
    ev,members,new=fixture(tmp_path)
    assert verify(ev,facts_pin=ev['facts'],members=members)==('control',document_sha256(new))


def test_one_logical_matmul_with_four_output_tiles_still_requires_reuse(tmp_path):
    ev,members,_=fixture(tmp_path,64)
    with pytest.raises(ValueError,match='multi-output-tile'):
        verify(ev,facts_pin=ev['facts'],members=members)


def test_single_use_review_cannot_change_numeric_or_tier_contract(tmp_path):
    ev,members,new=fixture(tmp_path);new['required_oracle_tiers']=['L2']
    from pathlib import Path
    p=Path(ev['after']['path']);p.write_text(yaml.safe_dump(new));ev['after']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='another source obligation'):
        verify(ev,facts_pin=ev['facts'],members=members)
