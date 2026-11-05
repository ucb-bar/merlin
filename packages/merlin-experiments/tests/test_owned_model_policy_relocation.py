"""Only exact retained capsule-local policy bytes may move between grade stages."""
import copy
import hashlib

import pytest

from merlin.targetgen.owned_host_model_execution import _same_owned_frozen_policy


def policies(tmp_path):
    result = []
    for folder in ('retained', 'selected'):
        root = tmp_path / folder
        root.mkdir()
        policy = {'descriptor': {'path': '/trusted/descriptor', 'sha256': 'owner'},
                  'manifest': {'path': '/trusted/manifest', 'sha256': 'owner'},
                  'required_tiers': ['L0', 'L1', 'L2', 'L3'], 'prohibited_instruction_roles': ['loop_descriptor']}
        for role in ('capsule_declaration', 'instruction_coverage'):
            path = root / role
            path.write_text('exact reviewed ' + role)
            policy[role] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'size_bytes': path.stat().st_size}
        result.append(policy)
    return result


def test_owned_policy_joins_verified_byte_identical_local_relocation(tmp_path):
    recorded, selected = policies(tmp_path)
    assert recorded != selected
    assert _same_owned_frozen_policy(recorded, selected)
    assert _same_owned_frozen_policy(recorded, recorded)


@pytest.mark.parametrize('change', ['original_content', 'selected_content', 'missing_original', 'tier', 'owner'])
def test_owned_policy_refuses_changed_missing_or_different_owner(tmp_path, change):
    from pathlib import Path
    recorded, selected = policies(tmp_path)
    selected = copy.deepcopy(selected)
    if change == 'original_content':
        Path(recorded['capsule_declaration']['path']).write_text('changed')
    elif change == 'selected_content':
        Path(selected['instruction_coverage']['path']).write_text('changed')
    elif change == 'missing_original':
        Path(recorded['instruction_coverage']['path']).unlink()
    elif change == 'tier':
        selected['required_tiers'] = ['L2']
    else:
        selected['descriptor']['path'] = '/different/owner'
    try:
        matched = _same_owned_frozen_policy(recorded, selected)
    except ValueError:
        matched = False
    assert not matched
