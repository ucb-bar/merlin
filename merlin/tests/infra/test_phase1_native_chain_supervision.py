"""Selected native build coordination never invents completed model evidence."""
import importlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / 'examples/gemmini/phase1/launch'))
    return importlib.import_module('selected_chain_supervision'), importlib.import_module('native_full_chain')


def test_model_argv_supports_interpreter_flags_and_rejects_other_module(modules):
    _, worker = modules
    assert worker.model_arguments(['python', '-u', '-P', '-m', 'merlin.model_check', 'check']) == ['check']
    with pytest.raises(ValueError):
        worker.model_arguments(['python', '-m', 'arbitrary.module', 'check'])


def test_owner_wait_rejects_dead_or_reused_process_before_files(modules, tmp_path):
    _, worker = modules
    pin = {'pid': os.getpid(), 'start_ticks': -1, 'boot_id': ''}
    with pytest.raises(ValueError, match='original native owner exited'):
        worker.wait_owner(tmp_path, pin, time.time() + 3)


def test_static_index_requires_every_completed_partition_marker(modules, tmp_path):
    supervisor, _ = modules
    context = {'root': tmp_path, 'run_dir': tmp_path / 'run'}
    for name in ['one', 'two']:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'native-static.json').write_text('{}')
    (tmp_path / 'one/partition-ready.json').write_text('{}')
    supervisor.publish_static(context, ['one', 'two'])
    assert not (tmp_path / 'static-index.json').exists()
    (tmp_path / 'two/partition-ready.json').write_text('{}')
    supervisor.publish_static(context, ['one', 'two'])
    assert set(json.loads((tmp_path / 'static-index.json').read_text())['records']) == {'one', 'two'}


def test_numerical_receipt_cannot_publish_partial_or_failed_finalization(modules, tmp_path):
    supervisor, _ = modules
    context = {'root': tmp_path}
    args = dict(children={}, names=['one', 'two'], identity={}, package={}, output_context={},
                output=SimpleNamespace(finalize=lambda *a: {'status': 'fail'}))
    state = {'models': {'one': {'full_chain_complete': True}}}
    supervisor.publish_numerical(context, None, state=state, **args)
    assert not (tmp_path / 'full-chain-qualification.json').exists()
    state['models']['two'] = {'full_chain_complete': True}
    args['children'] = {'one': {}, 'two': {}}
    with pytest.raises(ValueError, match='finalization failed'):
        supervisor.publish_numerical(context, None, state=state, **args)
    assert not (tmp_path / 'full-chain-qualification.json').exists()


@pytest.mark.parametrize('mode', ['paid', 'qualified', 'missing_selection'])
def test_only_explicit_frozen_qualification_delegates_source_precheck(modules, monkeypatch, tmp_path, mode):
    from merlin_experiments.phase1.feedback import private_numerical_selection as selection
    supervisor, _ = modules
    called = []
    monkeypatch.setattr(supervisor.P, 'source_preparation', lambda *a: called.append('paid') or {'checked': True})
    monkeypatch.setattr(selection, 'read', lambda *a: None if mode == 'missing_selection' else {'mode': selection.MODE})
    if mode == 'missing_selection':
        with pytest.raises(ValueError, match='canonical native full-chain'):
            supervisor.P.source_preparation_for(None, tmp_path, tmp_path / 'spec', qualification_only=True)
        assert not called
    else:
        value = supervisor.P.source_preparation_for(
            None, tmp_path, tmp_path / 'spec', qualification_only=mode == 'qualified')
        if mode == 'paid':
            assert called == ['paid'] and value['checked'] is True
        else:
            assert not called and value['source_preflight_passed'] is False
            assert value['candidate_qualified'] is False and value['author_allowed'] is False
            assert value['status'] == 'delegated_to_canonical_native_static_gate'


def test_fresh_attempt_preserves_failed_metadata_and_rejects_completed_intake(modules, monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import private_numerical_selection as selection
    supervisor, _ = modules
    config = tmp_path / 'config'
    config.write_text('{}')
    root = tmp_path / 'static'
    (root / 'old-failed-model').mkdir(parents=True)
    previous = root / 'old-failed-model/record.json'
    previous.write_text('original-failure')
    value = {'static_root': str(root), 'config': str(config), 'deadline_unix': 123}
    monkeypatch.setattr(selection, 'read', lambda _: value)
    launch = SimpleNamespace(path=lambda p: p, config=SimpleNamespace(path=config),
        document={'inputs': {'private_full_model_spec': str(tmp_path / 'spec')}}, run_id='new-run')
    context = supervisor.selected(launch, tmp_path / 'run', 123)
    assert context['partition_root'] == root / 'new-run'
    assert previous.read_text() == 'original-failure'
    with pytest.raises(ValueError, match='attempt must be fresh'):
        supervisor.selected(launch, tmp_path / 'run', 123)
    (root / 'static-index.json').write_text('{}')
    launch.run_id='third-run'
    with pytest.raises(ValueError, match='completed intake'):
        supervisor.selected(launch, tmp_path / 'run', 123)
