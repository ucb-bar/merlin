"""Authenticated loader configuration survives publication without recreating captures."""
import hashlib
import json
import shutil

import pytest

from merlin.targetgen import _capsule_bundle_worker as worker
from merlin.targetgen.capsule_source import M2MUnavailable, freeze_materialized_loader_profile


@pytest.mark.parametrize('profile', [{'seq': 12, 'decode_step': False},
                                   {'seq': 10, 'decode_step': True, 'cache_len': 9, 'kv_heads': 2}])
def test_exact_selected_profile_reaches_adjacent_loader_for_iteration_role(tmp_path, profile):
    selected = tmp_path / 'selected'
    selected.mkdir()
    loader = selected / 'loader.py'
    loader.write_text('import json\nfrom pathlib import Path\n'
                      'def get_model_and_inputs():\n'
                      ' return None, json.loads(Path(__file__).with_name("profile.json").read_text())\n')
    source = selected / 'profile.json'
    raw = json.dumps(profile).encode()
    source.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    published = tmp_path / 'capsule'
    published.mkdir()
    attributes = freeze_materialized_loader_profile(
        {'workload_role': 'iteration', 'loader_path': str(loader),
         'profile_path': str(source), 'profile_sha256': digest}, published,
    )
    frozen_loader = published / 'capsule.pytorch.py'
    shutil.copyfile(loader, frozen_loader)
    assert attributes == {'loader_profile': 'profile.json', 'loader_profile_sha256': digest}
    record = {'path': str(published / attributes['loader_profile']), 'sha256': digest}
    worker._verify_loader_profile(frozen_loader, record)
    assert worker._load_module(frozen_loader).get_model_and_inputs()[1] == profile
    (published / 'profile.json').unlink()
    with pytest.raises(RuntimeError, match='missing, indirect or changed'):
        worker._verify_loader_profile(frozen_loader, record)


def test_profile_pins_refuse_changed_source_and_unbound_adjacent_file(tmp_path):
    loader = tmp_path / 'loader.py'
    loader.write_text('def get_model_and_inputs(): return None, ()')
    profile = tmp_path / 'profile.json'
    profile.write_text('{}')
    output = tmp_path / 'out'
    output.mkdir()
    with pytest.raises(M2MUnavailable, match='changed'):
        freeze_materialized_loader_profile({'loader_path': str(loader), 'profile_path': str(profile),
                                            'profile_sha256': '0' * 64}, output)
    assert not (output / 'profile.json').exists()
    with pytest.raises(RuntimeError, match='no selected immutable binding'):
        worker._verify_loader_profile(loader, None)
