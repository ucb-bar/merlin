"""Background completion retains native products without timing or dispatch."""
import json
from pathlib import Path

import pytest

from merlin.benchharness import hash_tree
from merlin_experiments.phase1.feedback import completed_calibration as C


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import joint_performance as P
    from merlin_experiments.phase2 import census_launch
    from merlin.targetgen import capsule_common
    from merlin.runtime import _native_files as F

    compiler = tmp_path / 'compiler'
    compiler.mkdir()
    (compiler / 'compiler.py').write_text('pass\n')
    root = tmp_path / 'runs' / 'public_case'
    generated = root / 'generated'
    generated.mkdir(parents=True)
    elf = generated / 'fixture.elf'
    elf.write_bytes(b'fixture original producer; not executable')
    result = {'capsule': 'public_case', 'kind': 'layer', 'label': 'public',
        'numeric': {'status': 'pass'},
        'tiers': {'L2': {'status': 'pass', 'engine': 'spike'}, 'L3': {'status': 'unavailable'}},
        'compiler_source': {'status': 'stable', 'candidate_sha256': hash_tree(compiler)['sha256']}}
    path = root / 'capsule_result.json'
    path.write_text(json.dumps(result))
    cap = {'name': 'public_case', 'kind': 'layer', 'label': 'public'}
    selection = {'members': {'public_case': {'capsule_sha256': P.capsule_identity(cap)}},
                 'facts': {'sha256': 'facts'}}
    profile = {'performance_selection': {'sha256': 'selection'}, 'members': {}}
    state = {'policy': {'native_calibration': {'sha256': 'profile'}}}
    monkeypatch.setenv(joint_control.ENV, str(tmp_path / 'state.json'))
    monkeypatch.setattr(joint_control, 'selected', lambda: state)
    monkeypatch.setattr(P, 'selected', lambda: ({'sha256': 'selection'}, selection))
    monkeypatch.setattr(P, '_read', lambda pin: profile if pin['sha256'] == 'profile' else selection)
    monkeypatch.setattr(P, 'original_executable', lambda root: elf)
    monkeypatch.setattr(P, '_observe', lambda *a, **kw: pytest.fail('passive intake must not simulate'))
    monkeypatch.setattr(capsule_common, 'load_capsule', lambda *a, **kw: cap)
    # Native link custody itself has independent tests; this fixture has no executable.
    monkeypatch.setattr(census_launch, '_producer', lambda elf, limits: F.pin(elf, max_bytes=1024))
    kwargs = dict(result_path=path, compiler=compiler, capsules_root=tmp_path/'capsules', contract=None)
    return result, kwargs, cap, profile


def test_completed_background_product_enters_public_index_without_fit_authority(completed, tmp_path):
    result, kwargs, _, _ = completed
    receipt = C.register(result, **kwargs)
    assert receipt['calibration_candidate_intake']['status'] == 'pending_review'
    entries = list((tmp_path/'calibration_candidate_index').glob('*.json'))
    assert len(entries) == 1
    value = json.loads(entries[0].read_text())
    assert value['native_result'] == result
    assert value['protocol_status'] == 'unknown_or_incomplete'
    assert not value['fit_authorized'] and not value['dispatch_authorized']
    assert not (tmp_path/'calibration_registry').exists()
    assert C.register(result, **kwargs) == receipt
    assert len(list((tmp_path/'calibration_candidate_index').glob('*.json'))) == 1


@pytest.mark.parametrize('damage', ['durable', 'source', 'private', 'selected_source'])
def test_changed_or_unselected_completed_inputs_never_enter_index(completed, tmp_path, damage):
    result, kwargs, cap, _ = completed
    if damage == 'durable':
        kwargs['result_path'].write_text('{}')
    elif damage == 'source':
        (kwargs['compiler']/'compiler.py').write_text('changed\n')
    elif damage == 'private':
        cap['label'] = 'hidden'
    else:
        cap['kind'] = 'isa'
    with pytest.raises(ValueError):
        C.register(result, **kwargs)
    assert not (tmp_path/'calibration_candidate_index').exists()


def test_failed_numeric_or_unrun_spike_does_not_register(completed, tmp_path):
    result, kwargs, _, _ = completed
    result['numeric']['status'] = 'fail'
    assert C.register(result, **kwargs) is None
    result['numeric']['status'] = 'pass'
    result['tiers']['L2']['status'] = 'unavailable'
    assert C.register(result, **kwargs) is None
    assert not (tmp_path/'calibration_candidate_index').exists()


def test_only_existing_reviewed_profile_member_enters_fit_registry(completed, tmp_path):
    result, kwargs, _, profile = completed
    profile['members']['public_case'] = {'source_group_sha256': 'reviewed-independent-group'}
    C.register(result, **kwargs)
    entries = list((tmp_path/'calibration_registry').glob('*/entry.json'))
    assert len(entries) == 1
    value = json.loads(entries[0].read_text())
    assert value['source_group_sha256'] == 'reviewed-independent-group'
    assert value['estimate_status'] is None  # No timing invented from numerical pass.
