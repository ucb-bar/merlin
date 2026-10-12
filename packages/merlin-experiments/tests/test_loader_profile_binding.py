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


def test_iteration_requirement_producer_retains_receipt_bound_profile(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from merlin_experiments.phase0.requirements import _materialized_iteration_capsules
    from merlin.targetgen import capsule_source

    source = tmp_path / 'model.mlir'
    source.write_text('original source')
    loader = tmp_path / 'loader.py'
    loader.write_text('original loader')
    profile = tmp_path / 'profile.json'
    profile.write_text('{"selected":"original"}')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = tmp_path / 'capture_receipt.json'
    receipt.write_text(json.dumps({'artifacts': {'model.mlir': {'bytes': source.stat().st_size, 'sha256': sha(source)}}}))
    monkeypatch.setattr(capsule_source, 'materialized_model_artifacts', lambda *a: SimpleNamespace(
        dtype='int8', meta={'loader_provenance': {'profile_sha256': sha(profile)}},
    ))
    entries, outputs = _materialized_iteration_capsules({'applications': {'public_program': {
        'capture_source_path': str(source), 'capture_sha256': sha(source),
        'capture_receipt': {'receipt_sha256': sha(receipt)}, 'n_operations': 1,
    }}}, 'inventory', loader_snapshots={'public_program': (loader, sha(loader))})
    binding = entries[0]['materialized_capture']
    assert outputs[binding['profile_path']] == profile.read_bytes()
    assert binding['profile_sha256'] == sha(profile)
    assert binding['workload_role'] == 'iteration'
