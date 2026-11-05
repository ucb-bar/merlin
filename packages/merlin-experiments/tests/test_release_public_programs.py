"""Release review includes selected programs without exposing answer artifacts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import preparation as P
from merlin_experiments.corpus import public_program_suite as S
from merlin_experiments.corpus import release as R
from merlin_experiments.spec import SpecError

from merlin.common.digest import sha256_file


def selected_source(tmp_path, monkeypatch):
    source = tmp_path / 'selected/corpus/_perf/program'
    source.mkdir(parents=True)
    (source / 'capsule.interface.mlir').write_text('module {}')
    (source / 'golden.yaml').write_text('private expected values')
    (source / 'proof.npz').write_bytes(b'private dependency')
    cap = {'name': 'program', 'interface_mlir': 'capsule.interface.mlir', 'materialized_capture': {}}
    (source / 'capsule.yaml').write_text(yaml.safe_dump(cap))
    selection = tmp_path / 'selected/selection.json'
    selection.write_text(json.dumps({'schema': S.SCHEMA, 'original': True}))
    pin = {'path': str(selection), 'sha256': sha256_file(selection)}
    reopened = []

    def read(path, *, target):
        assert Path(path) == selection and target == 'fixture'
        reopened.append(path)
        return json.loads(selection.read_bytes()), {'program': source}

    monkeypatch.setattr(S, 'read', read)
    return pin, reopened


def test_source_view_is_bound_before_admission_and_has_no_answers(tmp_path, monkeypatch):
    pin, reopened = selected_source(tmp_path, monkeypatch)
    experiment = tmp_path / 'release/payload/experiment'
    experiment.mkdir(parents=True)
    document = {'workload_spec': {'existing': 'unchanged'}}
    view = P._stage_public_program_view(document, pin, experiment, target='fixture')
    assert document['workload_spec'] == {'existing': 'unchanged', S.KEY: pin}
    assert Path(view['path']) == experiment / 'public-program-sources'
    assert {p.name for p in Path(view['path']).rglob('*') if p.is_file()} == {
        'suite.json', 'capsule.interface.mlir'}
    assert len(reopened) == 2
    pin['sha256'] = 'f' * 64
    assert document['workload_spec'][S.KEY]['sha256'] != pin['sha256']


def test_changed_source_selection_pin_refuses_before_view(tmp_path, monkeypatch):
    pin, _ = selected_source(tmp_path, monkeypatch)
    Path(pin['path']).write_text('{}')
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    with pytest.raises(ValueError, match='changed'):
        P._stage_public_program_view({}, pin, experiment, target='fixture')
    assert not list(experiment.iterdir())


def test_unselected_refresh_preserves_legacy_and_refuses_inherited_pointer(tmp_path):
    assert P._stage_public_program_view({}, None, tmp_path, target='fixture') is None
    with pytest.raises(SpecError, match='explicit selected roster'):
        P._stage_public_program_view({'workload_spec': {S.KEY: {'path': 'old'}}}, None,
                                     tmp_path, target='fixture')


def test_review_reopens_private_selection_and_public_view(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment as T

    pin, reopened = selected_source(tmp_path, monkeypatch)
    root = tmp_path / 'release'
    experiment = root / 'payload/experiment'
    experiment.mkdir(parents=True)
    document = {}
    P._stage_public_program_view(document, pin, experiment, target='fixture')
    te = SimpleNamespace(workload_spec=document['workload_spec'], target='fixture')
    monkeypatch.setattr(T, 'load_target_experiment', lambda path: te)
    prepared = {'assembly': {'public_programs': dict(pin)}}
    R._verify_public_programs(root, prepared)
    assert len(reopened) >= 4
    leaf = experiment / 'public-program-sources/program/capsule.interface.mlir'
    leaf.unlink()
    leaf.write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        R._verify_public_programs(root, prepared)
    te.workload_spec[S.KEY] = {'path': pin['path'], 'sha256': 'f' * 64}
    with pytest.raises(SpecError, match='prepared source selection'):
        R._verify_public_programs(root, prepared)
