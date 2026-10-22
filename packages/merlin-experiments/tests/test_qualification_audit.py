"""Regress missing broker products and prohibit unsafe cross-source credit."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from merlin_experiments.tracking.qualification_audit import Reader, raw_paths, issued_scalability_rows, boundary_coverage
from merlin_experiments.phase1.feedback.tier_boundary_join import _COMMON, join


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_private_ordinary_selfcheck_is_collected_alongside_automatic(tmp_path):
    a = tmp_path / '_qa_work/fruns_a/runs/suite/case/capsule_result.json'
    b = tmp_path / '_qa_work/selfchecks/selfcheck_a/runs/suite/case/capsule_result.json'
    for p in (a, b): write(p, {})
    assert set(raw_paths(tmp_path)) == {a, b}


def test_issued_scalability_uses_executed_source_and_pin_not_request_fingerprint(tmp_path):
    channel = tmp_path / '.qa_channel'
    out = channel / '.resp_a.scalability'
    source = 'a' * 64
    issuer = write(channel / 'issuers/invocations/native/invocation.json', {'status': 'complete'})
    result = write(out / 'qualification.json', {'schema': 'merlin.phase1.public_scalability.v1',
        'candidate_sha256': source, 'score': {'per_capsule': [
            {'capsule': 'case', 'status': 'pass', 'numeric': 'pass'}]}})
    write(out / 'issued.json', {'issuer': issuer, 'result': result})
    raw = out / 'runs/runs/suite/case/capsule_result.json'
    row = {'capsule': 'case', 'status': 'pass', 'numeric': {'status': 'pass'},
           'compiler_source': {'candidate_sha256': source}}
    write(raw, row)
    assert list(issued_scalability_rows(tmp_path, Reader())) == [(row, raw, source)]
    row['compiler_source']['candidate_sha256'] = 'b' * 64
    write(raw, row)
    with pytest.raises(ValueError, match='source differs'):
        list(issued_scalability_rows(tmp_path, Reader()))
    write(Path(result['path']), {})
    with pytest.raises(ValueError, match='pin differs'):
        list(issued_scalability_rows(tmp_path, Reader()))


def observations():
    common = {key: 'd' * 64 for key in _COMMON}
    common.update(capsule='case', compiler_sha256='a'*64, submission_digest='old',
                  required_tiers=['L0', 'L1', 'L2', 'L3'])
    def tier(name, char):
        return {'schema':'merlin.original_tier_boundary.v1', 'tier':name,
            'engine':{'L2':'spike','L3':'gsim'}[name], 'common':common.copy(),
            **{k:char*64 for k in ['execution_digest','elf_sha256','harness_sha256','result_sha256']}}
    return tier('L2', '2'), tier('L3', '3')


def test_native_join_reuse_requires_current_l2_and_exact_semantic_boundary():
    l2, l3 = observations()
    proof = join(l2, l3)
    cert = {'status':'pass','digest':'old','instrument':'judge','execution_digest':l3['execution_digest'],
            'tier_boundary':l3,'boundary_join':proof}
    loop = {'status':'pass','digest':'old','execution_digest':l2['execution_digest'],'tier_boundary':l2}
    state = {'case':{'<certs>':{'L2':{'loop':loop},'L3':{'cert':cert}}}}
    assert boundary_coverage(state,source='a'*64,instrument='judge')['case']['kind'] == 'same_source_native_boundary'
    assert not boundary_coverage(state,source='b'*64,instrument='judge')
    new = copy.deepcopy(loop);new['digest']='new'
    new['tier_boundary']['common'].update(compiler_sha256='b'*64,submission_digest='new')
    state['case']['<certs>']['L2']['new']=new
    assert boundary_coverage(state,source='b'*64,instrument='judge')['case']['kind'] == 'compatible_native_certificate_reuse'
    assert not boundary_coverage(state,source='b'*64,instrument='another-judge')
    new['tier_boundary']['common']['kernel_object_sha256']='e'*64
    assert not boundary_coverage(state,source='b'*64,instrument='judge')
