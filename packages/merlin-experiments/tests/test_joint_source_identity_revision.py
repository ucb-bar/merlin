import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin_experiments.phase1.joint_source_identity_revision import verify
from merlin_experiments.phase2.contracts import document_sha256

SOURCE = '''module {
 func.func @forward() -> i32 attributes {prov.region_id = "r"} {
 %x = arith.constant 1 : i32
 return %x : i32
 }
}'''


def fixture(tmp_path, mutation=None):
    new_source = SOURCE.replace('prov.region_id = "r"',
        'prov.region_id = "r", prov.source_node_ids = ["direct_ir.r"]')
    if mutation == 'math':
        new_source = new_source.replace('constant 1', 'constant 2')
    if mutation == 'identity':
        new_source = new_source.replace('direct_ir.r', 'captured.framework_node')
    cap = dict(name='connected', kind='model_slice', interface_mlir='source.mlir',
               linalg_mlir='source.mlir', numeric_policy={'compare': 'exact_int'},
               software_screen={'status': 'admitted', 'placements': {'accelerator': 1}})
    evidence = dict(schema='merlin.direct_ir_identity_review.v1', name='connected')
    docs = {}
    for role, source in [('before', SOURCE), ('after', new_source)]:
        folder = tmp_path / role
        folder.mkdir()
        (folder / 'source.mlir').write_text(source)
        (folder / 'golden.bin').write_bytes(b'changed' if role == 'after' and mutation == 'golden' else b'original')
        doc = copy.deepcopy(cap)
        doc['software_screen']['program_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        if role == 'after' and mutation == 'placement':
            doc['software_screen']['placements'] = {'host': 1}
        p = folder / 'capsule.yaml'
        p.write_text(yaml.safe_dump(doc))
        evidence[role] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        docs[role] = doc
    return evidence, {'connected': {'capsule_sha256': document_sha256(docs['before'])}}, docs['after']


def test_identity_only_revision_keeps_exact_math_and_contract(tmp_path):
    evidence, members, new = fixture(tmp_path)
    assert verify(evidence, members=members) == ('connected', document_sha256(new))


@pytest.mark.parametrize('mutation', ['math', 'identity', 'golden', 'placement'])
def test_identity_revision_rejects_other_changes(tmp_path, mutation):
    evidence, members, _ = fixture(tmp_path, mutation)
    with pytest.raises(ValueError):
        verify(evidence, members=members)
