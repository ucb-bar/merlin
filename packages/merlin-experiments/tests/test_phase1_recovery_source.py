import hashlib
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import capsule_timing_archive as A
from merlin_experiments.phase1.feedback import public_owned_host_selection as S

from merlin.targetgen import capsule_runner as R


@pytest.fixture(autouse=True)
def _private_content_store(tmp_path, monkeypatch):
    """Captured compiler source shares bytes through the content store; keep this test's private."""
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))


def test_host_only_selection_is_installed_without_instruction_declaration(tmp_path, monkeypatch):
    monkeypatch.setenv(S.ENV, 'untrusted inherited')
    model = {'board': 'fixture_board', 'arena_mb': 32}
    for field in ('board_catalog', 'host_dts', 'rtl_facts'):
        path = tmp_path / field
        path.write_text(field)
        model[field] = str(path)
        model[field + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = tmp_path / 'spec.yaml'
    spec.write_text(yaml.safe_dump({'target': 'fixture', 'models': [model]}))
    record = S.install_for_session(spec, None, target='fixture', run_dir=tmp_path / 'run')
    projected = json.loads(Path(record['path']).read_text())
    assert set(projected) == {'board_catalog', 'dts', 'rtl_facts', 'board', 'arena_mb'}
    assert json.loads(__import__('os').environ[S.ENV]) == record
    (tmp_path / 'rtl_facts').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        S.install_for_session(spec, None, target='fixture', run_dir=tmp_path / 'run')
    assert S.ENV not in __import__('os').environ


def prepare(tmp_path):
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'compiler.py').write_text('print("compiler one")\n')
    generated = tmp_path / 'generated'
    generated.mkdir()
    captured = R._capture_compiler_source(package, generated)
    artifacts = {}
    for name, key in [('package_kernel.elf', 'elf'), ('harness.c', 'harness'),
                      ('input.interface.mlir', 'input_interface'), ('command_buffer.json', 'command_buffer')]:
        path = generated / name
        path.write_text(name)
        artifacts[key] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return package, generated, captured, artifacts


def test_producer_retains_compiler_and_archive_verifies_artifact_join(tmp_path):
    package, generated, captured, artifacts = prepare(tmp_path)
    source = R._finish_compiler_source(captured, generated)
    document = {'compiler_source': source, 'original_compiler_child': {'candidate_sha256': 'f' * 64}}
    assert A._compiler_candidate(document, generated, artifacts) == captured['candidate_sha256']
    # Later author edits don't change the original retained compilation source.
    (package / 'compiler.py').write_text('print("compiler two")\n')
    assert A._compiler_candidate(document, generated, artifacts) == captured['candidate_sha256']
    artifacts['elf']['sha256'] = '0' * 64
    assert A._compiler_candidate(document, generated, artifacts) is None


def test_source_movement_during_execution_does_not_authorize_timing(tmp_path):
    package, generated, captured, artifacts = prepare(tmp_path)
    (package / 'compiler.py').write_text('changed')
    source = R._finish_compiler_source(captured, generated)
    assert source['status'] == 'source_changed'
    assert A._compiler_candidate({'compiler_source': source}, generated, artifacts) is None


def test_retained_snapshot_tampering_refuses_timing(tmp_path):
    _, generated, captured, artifacts = prepare(tmp_path)
    source = R._finish_compiler_source(captured, generated)
    retained = generated / captured['snapshot'] / 'compiler.py'
    retained.unlink()  # retained source is read-only; replacing the name is how it could change
    retained.write_text('changed')
    assert A._compiler_candidate({'compiler_source': source}, generated, artifacts) is None


def test_legacy_unattributed_results_stay_unattributed(tmp_path):
    _, generated, _, artifacts = prepare(tmp_path)
    assert A._compiler_candidate({}, generated, artifacts) is None
