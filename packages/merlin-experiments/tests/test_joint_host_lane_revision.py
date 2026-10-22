import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin_experiments.phase1.joint_host_lane_revision import verify
from merlin_experiments.phase2.contracts import document_sha256

SOURCE = '''module {
 func.func @forward(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>, %c: tensor<16x16xi32>) -> tensor<16x16xi32> {
 %r = linalg.generic {indexing_maps = [affine_map<(i,j,k)->(i,k)>, affine_map<(i,j,k)->(k,j)>, affine_map<(i,j,k)->(i,j)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<16x16xi8>, tensor<16x16xi8>) outs(%c : tensor<16x16xi32>) {
 ^bb0(%x: i8, %y: i8, %z: i32):
 %xx = arith.extsi %x : i8 to i32
 %yy = arith.extsi %y : i8 to i32
 %p = arith.muli %xx, %yy : i32
 %s = arith.addi %p, %z : i32
 linalg.yield %s : i32
 } -> tensor<16x16xi32>
 return %r : tensor<16x16xi32>
 }
}'''


def fixture(tmp_path):
    cap = dict(name='connected',kind='model_slice',interface_mlir='source.mlir',linalg_mlir='source.mlir',
               operation={'op':'model','attributes':{'dtype':'int8','compile_dtype':'fp32'}},
               software_screen={'status':'admitted','program_sha256':hashlib.sha256(SOURCE.encode()).hexdigest()},
               numeric_policy={'compare':'exact_int'},required_oracle_tiers=['L0','L1','L2','L3'])
    new=copy.deepcopy(cap);new['operation']['attributes']['compile_dtype']='int8'
    ev={'schema':'merlin.captured_host_lane_review.v1','name':'connected'}
    for role,doc in [('before',cap),('after',new)]:
        root=tmp_path/role;root.mkdir();(root/'source.mlir').write_text(SOURCE)
        p=root/'capsule.yaml';p.write_text(yaml.safe_dump(doc));ev[role]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return ev,{'connected':{'capsule_sha256':document_sha256(cap)}},new


def test_review_changes_only_package_selection(tmp_path):
    ev,members,new=fixture(tmp_path)
    assert verify(ev,members=members)==('connected',document_sha256(new))


@pytest.mark.parametrize('mutation',['numeric','tiers','source'])
def test_review_cannot_change_typed_source_or_obligations(tmp_path,mutation):
    ev,members,new=fixture(tmp_path)
    p=Path(ev['after']['path'])
    if mutation=='numeric':new['numeric_policy']['compare']='approximate'
    elif mutation=='tiers':new['required_oracle_tiers']=['L2']
    else:(p.parent/'source.mlir').write_text(SOURCE.replace('arith.extsi','arith.extui'))
    p.write_text(yaml.safe_dump(new));ev['after']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='changes source|another source'):
        verify(ev,members=members)


def test_unsupported_placement_stays_unsupported(tmp_path):
    ev,members,new=fixture(tmp_path)
    for role in ['before','after']:
        p=Path(ev[role]['path']);cap=yaml.safe_load(p.read_text())
        cap['software_screen']['status']='unsupported'
        p.write_text(yaml.safe_dump(cap));ev[role]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        if role=='before':members['connected']['capsule_sha256']=document_sha256(cap)
        else:new=cap
    assert verify(ev,members=members)==('connected',document_sha256(new))
    assert new['software_screen']['status']=='unsupported'
