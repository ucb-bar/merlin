import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from merlin_experiments.phase1.joint_constant_closure_revision import verify
from merlin_experiments.phase2.contracts import document_sha256


def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(root):
    before, after = root / 'before', root / 'after'
    before.mkdir(); after.mkdir()
    for directory, text in ((before, 'reviewed original source'), (after, 'reviewed literal closure')):
        (directory / 'source.mlir').write_text(text)
        (directory / 'golden.npz').write_bytes(b'unchanged retained array payload')
    old_hash, new_hash = pin(before / 'source.mlir')['sha256'], pin(after / 'source.mlir')['sha256']
    old = dict(name='cut', kind='model_slice', interface_mlir='source.mlir', linalg_mlir='source.mlir',
        inputs=[dict(name='input_0', shape=[2], dtype='i8'), dict(name='input_1', shape=[2], dtype='i32')],
        operation={'attributes': {'arg_order': ['input_0', 'input_1', 'output_0'], 'outs': ['output_0']}},
        numeric_policy={'compare': 'exact'}, required_oracle_tiers=['L2', 'L3'],
        source_reference='original', software_screen={'status': 'unsupported', 'program_sha256': old_hash})
    new = copy.deepcopy(old); new['inputs'] = old['inputs'][:1]
    new['operation']['attributes']['arg_order'] = ['input_0', 'output_0']
    new['source_reference'] = 'reviewed closed source'
    new['software_screen'] = {'status': 'admitted', 'program_sha256': new_hash}
    base_gold = {'outputs': {'output_0': [1, 2]}, 'oracle_provenance': {
        'inputs': {'input_0': {'decoded': [1, 2]}, 'input_1': {'decoded': [0, 0]}},
        'arg_order': old['operation']['attributes']['arg_order'], 'scope': 'original'}}
    new_gold = copy.deepcopy(base_gold)
    new_gold['oracle_provenance'].update(inputs={'input_0': {'decoded': [1, 2]}},
        arg_order=new['operation']['attributes']['arg_order'], source_sha256='restored-source',
        region_sha256=new_hash, scope='new source projection')
    for directory, cap, gold in ((before, old, base_gold), (after, new, new_gold)):
        (directory / 'capsule.yaml').write_text(yaml.safe_dump(cap))
        (directory / 'golden.yaml').write_text(yaml.safe_dump(gold))
    def write(name, doc):
        path = root / name; path.write_text(json.dumps(doc)); return pin(path)
    source = {'path': '/reviewed/prepared/source', 'sha256': 'prepared-source'}
    record = write('record.json', {'source_projection': {'sha256': old_hash},
        'source': source, 'all_live_outputs_retained': True})
    preparation = write('preparation.json', {'schema': 'operator.prepared_source_reproduction.v1',
        'old': source, 'new': {'sha256': 'restored-source'}, 'attribute_only': True,
        'changes': [{'additions_only': True, 'provenance_only': True}],
        'projected': {'sha256': new_hash, 'inputs': 1, 'outputs': 1}})
    projection = write('projection.json', {'schema': 'operator.connected_boundary_projection.v1',
        'preparation_review': preparation, 'original_record': record, 'projected_sha256': new_hash,
        'source_sha256': 'restored-source', 'outputs_same_source_values': True, 'output_count': 1,
        'new_to_original_inputs': [0], 'removed_literal_inputs': [
            {'original_input': 1, 'literal_bytes_equal': True, 'array_sha256': 'zero-array'}]})
    evidence = {'schema': 'merlin.captured_constant_closure_review.v1', 'name': 'cut',
        'before': pin(before / 'capsule.yaml'), 'after': pin(after / 'capsule.yaml'),
        'projection': projection, 'preparation': preparation}
    return evidence, {'cut': {'capsule_sha256': document_sha256(old)}}, new


def test_reviewed_boundary_rebind_preserves_all_other_obligations(tmp_path):
    evidence, members, new = fixture(tmp_path)
    assert verify(evidence, members=members) == ('cut', document_sha256(new))


def test_source_review_chain_remains_bounded():
    from merlin_experiments.phase1.joint_source_revision import normalized
    with pytest.raises(ValueError, match='ancestry'):
        normalized({'members': {}}, {'source_binding_revisions': [{}] * 33})


@pytest.mark.parametrize('mutation', ['source', 'golden', 'tiers', 'input', 'literal'])
def test_review_does_not_admit_unreviewed_changes(tmp_path, mutation):
    evidence, members, new = fixture(tmp_path)
    directory = Path(evidence['after']['path']).parent
    if mutation == 'source':
        (directory / 'source.mlir').write_text('different code')
    elif mutation == 'golden':
        (directory / 'golden.npz').write_bytes(b'changed expected values')
    elif mutation in {'tiers', 'input'}:
        if mutation == 'tiers':
            new['required_oracle_tiers'] = ['L2']
        else:
            new['inputs'][0]['shape'] = [3]
        path = directory / 'capsule.yaml'; path.write_text(yaml.safe_dump(new)); evidence['after'] = pin(path)
    else:
        path = Path(evidence['projection']['path']); doc = json.loads(path.read_text())
        doc['removed_literal_inputs'][0]['literal_bytes_equal'] = False
        path.write_text(json.dumps(doc)); evidence['projection'] = pin(path)
    with pytest.raises(ValueError):
        verify(evidence, members=members)
