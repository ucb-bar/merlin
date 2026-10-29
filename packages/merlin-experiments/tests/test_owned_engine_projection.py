"""Engine metadata follows reopened model evidence without granting new tiers."""
import json

import numpy as np
import pytest
import yaml

from merlin.targetgen import capsule_grade


@pytest.fixture
def candidate():
    return capsule_grade


@pytest.mark.parametrize('status', ['pass', 'unverified', 'fail'])
@pytest.mark.parametrize('metadata', [
    {},
    {'engine': 'spike', 'fidelity': 'functional_model'},
    {'engine': '', 'fidelity': None},
    {'engine': 3, 'fidelity': False},
])
def test_outer_tier_uses_only_reopened_audit_metadata(candidate, monkeypatch, status, metadata):
    audit = {'status': status, **metadata}
    check = {
        'kind': 'owned_host_model_execution',
        'status': status if status in {'pass','fail'} else 'incomplete',
        'numeric': {'status': 'pass'} if status == 'pass' else {},
        'candidate_required_tiers': {'required_tiers': ['L2'], 'tiers': {'L2': audit}},
    }
    monkeypatch.setattr(candidate, 'model_execution_check', lambda *args, **kwargs: check)
    row = {'kind': 'model', 'tiers': {'L2': {'engine': 'untrusted-old-engine'}}}
    result = candidate.enforce_model_execution_check(row, {}, target='synthetic')
    tier = result['tiers']['L2']
    assert tier['status'] == status
    assert tier['candidate_evidence'] == audit
    assert tier.get('engine') == (metadata.get('engine') if isinstance(metadata.get('engine'),str) and metadata['engine'] else None)
    assert tier.get('fidelity') == (metadata.get('fidelity') if isinstance(metadata.get('fidelity'),str) and metadata['fidelity'] else None)
    assert tier['cycle_accurate'] is False
    assert tier['derived_from_rtl'] is False


@pytest.mark.parametrize('engine', ['spike', 'other_functional_engine', None])
def test_score_projection_preserves_audited_engine_without_inventing_one(candidate, engine):
    audit = {'status': 'pass', 'fidelity': 'functional_model', 'derived_from_rtl': False, 'cycle_accurate': False}
    if engine is not None:
        audit['engine'] = engine
    row = {'kind': 'model', 'numeric': {'status': 'pass'}, 'model_execution_check': {
        'kind': 'owned_host_model_execution', 'status': 'pass',
        'candidate_required_tiers': {'required_tiers': ['L2'], 'tiers': {'L2': audit}},
    }}
    view = candidate._candidate_score_view(row)
    assert view['tiers']['L2'].get('engine') == engine
    assert view['tiers']['L2']['fidelity'] == 'functional_model'
    assert view['tiers']['L2']['cycle_accurate'] is False


@pytest.mark.parametrize('mutation',['none','mismatch','missing','extra','duplicate','short_extent','wrong_order','missing_done','duplicate_done'])
def test_owned_numeric_reopens_complete_frozen_output_roster(tmp_path,monkeypatch,mutation):
    from merlin.compile import baremetal_model
    from merlin.targetgen.owned_host_model_execution import _numeric_full_output

    capsule=tmp_path/'capsule';capsule.mkdir()
    runtime=tmp_path/'runtime';runtime.mkdir()
    (capsule/'capsule.yaml').write_text(yaml.safe_dump({'numeric_policy':{'compare':'exact_int','dtype':'i32'}}))
    (capsule/'golden.yaml').write_text(yaml.safe_dump({'golden_source':'independent_fixture','outputs':{'Y0':[1,2],'Y1':[3,-4]}}))
    order=['Y0','Y1'];values=[np.array([1,2],dtype=np.int32),np.array([3,-4],dtype=np.int32)]
    console=b'DONE\n'
    if mutation=='mismatch':values[1][1]=-3
    if mutation=='missing':order.pop();values.pop()
    if mutation=='extra':order.append('Y2');values.append(np.array([9],dtype=np.int32))
    if mutation=='duplicate':order[1]='Y0'
    if mutation=='short_extent':values[1]=np.array([3],dtype=np.int32)
    if mutation=='wrong_order':order.reverse()
    if mutation=='missing_done':console=b''
    if mutation=='duplicate_done':console=b'DONE\nDONE\n'
    (runtime/'output_order.json').write_text(json.dumps(order))
    record={'runtime_inputs':str(runtime),'build':{'output':{'build_hash':'selected-linked-build'}}}
    calls=[]
    def transport(actual_console,capture,build_hash):
        assert actual_console==console and capture==runtime and build_hash=='selected-linked-build'
        calls.append(True)
        return values,{'build_hash':build_hash}
    # Only byte transport is substituted. Frozen golden loading, complete
    # roster/extents and the numerical comparator remain the original code.
    monkeypatch.setattr(baremetal_model,'read_full_outputs',transport)
    if mutation in {'none','mismatch','wrong_order'}:
        numeric,_=_numeric_full_output({'__dir__':str(capsule)},record,console)
        assert numeric['status']==('pass'if mutation=='none'else'fail')
        assert set(numeric['per_output'])=={'Y0','Y1'}
        assert all(row['n_elements']==2 for row in numeric['per_output'].values())
        assert numeric['mismatch_count']==(0 if mutation=='none'else 1 if mutation=='mismatch'else 4)
    else:
        with pytest.raises(ValueError):_numeric_full_output({'__dir__':str(capsule)},record,console)
    assert bool(calls)==(mutation not in {'missing_done','duplicate_done'})
