"""Separate host-pinned objectives never extend frozen functional qualification."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.perf.external_objective import (load_external_objective, OBJECTIVE_DIRECTORY,
                                             objective_directory)

sys.path.insert(0, str(repo_root() / 'merlin/experiments/gemmini_perf_bench/scripts'))
PAS = importlib.import_module('perf_agent_stage')

SOURCE = '''builtin.module {
  func.func @forward(%x: tensor<2xi16>) -> tensor<2xi16> {
    %y = tensor.cast %x : tensor<2xi16> to tensor<2xi16>
    func.return %y : tensor<2xi16>
  }
}'''


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def specification(tmp_path, *, source_text=SOURCE, **changes):
    source = tmp_path / 'normalized.mlir'
    source.write_text(source_text)
    spec = {'schema': 'external_full_model_objective_spec_v1', 'id': 'host-model', 'entry': 'forward',
        'normalized_source': str(source), 'source_sha256': sha(source_text.encode()),
        'provenance_pins': {'capture_identity': sha(b'host capture identity')},
        'already_normalized': True, 'complete_model': True}
    spec.update(changes)
    path = tmp_path / 'objective-spec.json'
    payload = json.dumps(spec).encode()
    path.write_bytes(payload)
    return path, sha(payload), source


def load(path, digest):
    return load_external_objective(path, spec_sha256=digest, max_source_bytes=4096)


def test_exact_source_and_declared_completeness_without_arithmetic_execution(tmp_path):
    path, digest, source = specification(tmp_path)
    objective = load(path, digest)
    record = objective.record()
    assert objective.source_bytes == source.read_bytes()
    assert record['complete_model'] == 'host_declared_not_independently_proved'
    assert record['phase1_qualification_applies_to_external_model'] is False
    assert record['numerical_equivalence'] == 'UNPROVEN'
    assert not record['full_model_execution_allowed']
    assert {name for name, _ in objective.files()} == {'capsule.yaml', 'capsule.interface.mlir', 'objective.json'}
    assert str(source) not in objective.record_bytes.decode()
    source.write_text('changed after host load')
    assert objective.source_bytes == SOURCE.encode()


@pytest.mark.parametrize('change', ['spec_pin', 'source_pin', 'source_drift', 'relative', 'traversal',
    'symlink', 'parent_symlink', 'unknown_key', 'unqualified', 'missing_pins', 'path_pin', 'wrong_entry', 'size'])
def test_stale_pins_and_path_or_authority_surprises_refuse(tmp_path, change):
    changes = {}
    if change == 'source_pin': changes['source_sha256'] = sha(b'wrong')
    elif change == 'relative': changes['normalized_source'] = 'normalized.mlir'
    elif change == 'traversal': changes['normalized_source'] = str(tmp_path / '..' / tmp_path.name / 'normalized.mlir')
    elif change == 'unknown_key': changes['normalization_callback'] = 'do_not_execute'
    elif change == 'unqualified': changes['complete_model'] = False
    elif change == 'missing_pins': changes['provenance_pins'] = {}
    elif change == 'path_pin': changes['provenance_pins'] = {'/secret/reference': sha(b'x')}
    elif change == 'wrong_entry': changes['entry'] = 'missing'
    path, digest, source = specification(tmp_path, **changes)
    if change == 'spec_pin': digest = sha(b'other spec')
    elif change == 'source_drift': source.write_text(SOURCE+'\n')
    elif change == 'symlink':
        source.rename(tmp_path / 'real.mlir')
        source.symlink_to(tmp_path / 'real.mlir')
    elif change == 'parent_symlink':
        linked = tmp_path / 'link'
        linked.symlink_to(tmp_path, target_is_directory=True)
        spec = json.loads(path.read_text())
        spec['normalized_source'] = str(linked / 'normalized.mlir')
        path.write_text(json.dumps(spec))
        digest = sha(path.read_bytes())
    with pytest.raises(ValueError):
        load_external_objective(path, spec_sha256=digest, max_source_bytes=1 if change == 'size' else 4096)


def corpus(tmp_path, monkeypatch):
    frozen = tmp_path / 'frozen' / 'public' / 'capsule'
    frozen.mkdir(parents=True)
    (frozen / 'capsule.yaml').write_text('id: original\n')
    (frozen / 'capsule.interface.mlir').write_text(SOURCE)
    member = PAS.PerformanceCapsule('family', 'capsule', frozen, 'public/capsule', {}, sha(b'a'), 2, 1)
    result = PAS.FrozenPerformanceCorpus(tmp_path / 'frozen', tmp_path / 'frozen', tmp_path / 'manifest',
                                       sha(b'b'), sha(b'c'), (member,))
    target = SimpleNamespace(capsule_corpus=tmp_path / 'original' / 'public')
    monkeypatch.setattr(PAS, 'answer_surfaces', lambda _: [])
    return result, target


def test_external_sealed_under_existing_readonly_grant_and_legacy_unchanged(tmp_path, monkeypatch):
    path, digest, source = specification(tmp_path)
    objective = load(path, digest)
    frozen, target = corpus(tmp_path, monkeypatch)
    legacy = PAS.build_answer_free_agent_inputs(frozen, target, tmp_path / 'legacy')
    legacy_record = PAS._exact_tree_record(legacy.root)
    inputs = PAS.build_answer_free_agent_inputs(frozen, target, tmp_path / 'inputs', external_objective=objective)
    sentinel = PAS.select_external_e2e_sentinel(objective, inputs)
    assert sentinel.capsule_path == str(PAS.AGENT_CORPUS_MOUNT / OBJECTIVE_DIRECTORY)
    assert sentinel.frozen_source_path == str(inputs.root / OBJECTIVE_DIRECTORY)
    assert sentinel.required_tiers == sentinel.required_lanes == ()
    for file in Path(sentinel.frozen_source_path).iterdir():
        assert file.stat().st_mode & 0o222 == 0
    assert (Path(sentinel.frozen_source_path) / 'capsule.interface.mlir').read_bytes() == source.read_bytes()
    assert PAS._exact_tree_record(legacy.root) == legacy_record
    assert 'external_objective' not in json.loads(legacy.manifest_path.read_bytes())
    assert not (legacy.root / OBJECTIVE_DIRECTORY).exists()
    assert inputs.n_files == legacy.n_files+3
    PAS.verify_answer_free_agent_inputs(inputs)
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    monkeypatch.setattr(PAS.BW, 'base_argv', lambda *a, **k: ['bwrap', '--bind', str(candidate), str(candidate)])
    monkeypatch.setattr(PAS.TC, 'toolchain_binds', lambda _: [])
    inner = PAS.inner_execution_policy(target, candidate, inputs)
    outer = PAS.outer_codex_policy(candidate, inputs, [], target)
    for policy in (inner, outer):
        assert f'--ro-bind {inputs.root} {PAS.AGENT_CORPUS_MOUNT}' in ' '.join(policy.argv)
        assert str(source) not in policy.argv and str(path) not in policy.argv
    tampered = inputs.root / OBJECTIVE_DIRECTORY / 'capsule.interface.mlir'
    tampered.chmod(0o644)
    tampered.write_text(SOURCE+'\n')
    with pytest.raises(PAS.StageGateError, match='changed'):
        PAS.select_external_e2e_sentinel(objective, inputs)


def test_multiple_external_models_receive_distinct_readonly_grants(tmp_path, monkeypatch):
    first_dir, second_dir = tmp_path / 'first', tmp_path / 'second'
    first_dir.mkdir()
    second_dir.mkdir()
    first_path, first_digest, _ = specification(first_dir, id='vision-model')
    second_source = SOURCE.replace('2xi16', '3xi16')
    second_path, second_digest, _ = specification(
        second_dir, id='language-model', source_text=second_source)
    objectives = [load(first_path, first_digest), load(second_path, second_digest)]
    frozen, target = corpus(tmp_path, monkeypatch)

    inputs = PAS.build_answer_free_agent_inputs(
        frozen, target, tmp_path / 'inputs', external_objectives=objectives)
    sentinels = [PAS.select_external_e2e_sentinel(objective, inputs)
                 for objective in objectives]
    assert [Path(row.frozen_source_path).relative_to(inputs.root) for row in sentinels] == [
        objective_directory('vision-model'), objective_directory('language-model')]
    assert len({row.capsule_sha256 for row in sentinels}) == 2
    manifest = json.loads(inputs.manifest_path.read_text())
    assert [row['id'] for row in manifest['external_objectives']] == [
        'vision-model', 'language-model']
    PAS.verify_answer_free_agent_inputs(inputs)

    with pytest.raises(PAS.StageGateError, match='distinct'):
        PAS.build_answer_free_agent_inputs(
            frozen, target, tmp_path / 'duplicate', external_objectives=[objectives[0], objectives[0]])


def test_launcher_requires_pin_and_mutually_exclusive_objective_selection(tmp_path):
    launcher = importlib.import_module('launch_global_agent_experiment')
    base = ['--campaign-config', str(tmp_path/'config'), '--candidate', str(tmp_path/'candidate'),
            '--output', str(tmp_path/'out')]
    for options in (['--external-objective', '/model.json'],
                    ['--external-objective-sha256', sha(b'pin')],
                    ['--external-objective', '/model.json', '--objective-capsule', 'legacy']):
        with pytest.raises(SystemExit) as error:
            launcher.main(base+options)
        assert error.value.code == 2


def test_source_snapshot_worker_receives_exact_external_spec_pin(tmp_path, monkeypatch):
    launcher = importlib.import_module('launch_global_agent_experiment')
    snapshot = importlib.import_module('perf_snapshot')
    descriptor = tmp_path / 'target.yaml'
    descriptor.write_text('target: fixture\n')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'descriptor': str(descriptor)}))
    path, digest, _ = specification(tmp_path)
    monkeypatch.setattr(snapshot, 'create', lambda *a, **k: None)
    seen = {}
    class WorkerIntercept(Exception):
        pass
    lease = (tmp_path / 'host-resource.lease').open('a+')
    monkeypatch.setattr(launcher, '_acquire_host_resource_lease', lambda stage_root: lease)
    def execute(command, environment, **kwargs):
        seen.update(command=command, environment=environment)
        raise WorkerIntercept
    monkeypatch.setattr(launcher, '_run_resource_guarded_worker', execute)
    with pytest.raises(WorkerIntercept):
        launcher.main(['--campaign-config', str(config), '--candidate', str(tmp_path/'candidate'),
            '--output', str(tmp_path/'out'), '--external-objective', str(path),
            '--external-objective-sha256', digest])
    command = seen['command']
    assert command[command.index('--external-objective')+1] == str(path)
    assert command[command.index('--external-objective-sha256')+1] == digest
    assert '--source-worker' in command


def test_source_snapshot_worker_receives_ordered_portfolio_and_external_member(tmp_path, monkeypatch):
    launcher = importlib.import_module('launch_global_agent_experiment')
    snapshot = importlib.import_module('perf_snapshot')
    descriptor = tmp_path / 'target.yaml'
    descriptor.write_text('target: fixture\n')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'descriptor': str(descriptor)}))
    first_dir, second_dir = tmp_path / 'first', tmp_path / 'second'
    first_dir.mkdir()
    second_dir.mkdir()
    path, digest, _ = specification(first_dir, id='external-one')
    path2, digest2, _ = specification(
        second_dir, id='external-two', source_text=SOURCE.replace('2xi16', '3xi16'))
    monkeypatch.setattr(snapshot, 'create', lambda *a, **k: None)
    lease = (tmp_path / 'portfolio-resource.lease').open('a+')
    monkeypatch.setattr(launcher, '_acquire_host_resource_lease', lambda stage_root: lease)
    seen = {}
    class WorkerIntercept(Exception):
        pass
    def execute(command, environment, **kwargs):
        seen['command'] = command
        raise WorkerIntercept
    monkeypatch.setattr(launcher, '_run_resource_guarded_worker', execute)

    with pytest.raises(WorkerIntercept):
        launcher.main([
            '--campaign-config', str(config), '--candidate', str(tmp_path/'candidate'),
            '--output', str(tmp_path/'out'), '--objective-capsule', 'resnet',
            '--portfolio-capsule', 'tiny-llama', '--portfolio-capsule', 'lstmnetvit',
            '--portfolio-external-objective', str(path),
            '--portfolio-external-objective-sha256', digest,
            '--portfolio-external-objective', str(path2),
            '--portfolio-external-objective-sha256', digest2])
    command = seen['command']
    members = [command[index + 1] for index, value in enumerate(command)
               if value == '--portfolio-capsule']
    assert members == ['tiny-llama', 'lstmnetvit']
    paths = [command[index + 1] for index, value in enumerate(command)
             if value == '--portfolio-external-objective']
    digests = [command[index + 1] for index, value in enumerate(command)
               if value == '--portfolio-external-objective-sha256']
    assert paths == [str(path), str(path2)]
    assert digests == [digest, digest2]
