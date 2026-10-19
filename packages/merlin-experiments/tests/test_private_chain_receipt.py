"""Private numerical receipts reject altered selection before accepting model evidence."""
import json

import pytest
from merlin_experiments.phase1.feedback import private_chain_receipt as R

from merlin.common.digest import sha256_file


def test_files_reopen_changed_bytes_and_reject_indirection(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('{}')
    files = R._Files()
    files.pin(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='bytes changed'):
        files.unchanged()
    indirect = tmp_path / 'alias'
    indirect.symlink_to(path)
    with pytest.raises(ValueError, match='indirect'):
        R._Files().pin(indirect)


@pytest.mark.parametrize('key,value', [
    ('passed', False), ('run_dir', '/other/run'), ('candidate_tree_sha256', 'other'),
    ('private_spec_sha256', 'other'), ('schema', 'arbitrary-pass-record'),
])
def test_foreign_or_partial_receipt_cannot_reach_model_comparison(tmp_path, monkeypatch, key, value):
    from merlin_experiments.phase1.feedback import private_numerical_selection as S
    spec = tmp_path / 'spec.yaml'
    spec.write_text('selected: true')
    receipt_path = tmp_path / 'full-chain-qualification.json'
    record = {'schema': R.SCHEMA, 'passed': True, 'run_dir': str(tmp_path / 'run'),
              'private_spec_sha256': sha256_file(spec), 'candidate_tree_sha256': 'candidate'}
    record[key] = value
    receipt_path.write_text(json.dumps(record))
    monkeypatch.setattr(S, 'read', lambda _: {'static_root': str(tmp_path)})
    with pytest.raises(ValueError, match='selected native run/compiler/specification'):
        R.verify(receipt_path, run_dir=tmp_path / 'run', private_spec_path=spec, candidate_sha256='candidate')


def test_outside_selected_producer_directory_refuses(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_numerical_selection as S
    spec = tmp_path / 'spec.yaml'
    spec.write_text('selected: true')
    record = tmp_path / 'wrong.json'
    record.write_text('{}')
    monkeypatch.setattr(S, 'read', lambda _: {'static_root': str(tmp_path)})
    with pytest.raises(ValueError, match='outside the selected producer'):
        R.verify(record, run_dir=tmp_path / 'run', private_spec_path=spec, candidate_sha256='candidate')


@pytest.mark.parametrize('mutation', [None, 'structural', 'other_partition', 'other_image', 'other_program'])
def test_numerical_image_must_be_the_same_independently_admitted_static_image(tmp_path, mutation):
    from types import SimpleNamespace as N

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value))
        return path

    original = save('native.json', {'output': {'elf_sha256': 'image', 'build_hash': 'build'}})
    partition = save('native-static.json', {'instruction_selection': {'issuer': 'original'},
        'receipts': {'program': {'path': str(original), 'sha256': sha256_file(original)}}})
    qualification = {'partition': {'path': str(partition), 'sha256': sha256_file(partition)},
                     'program': 'program', 'instruction_selection': {'issuer': 'original'}}
    build = {'key': {'native_qualification': qualification}, 'native_qualification': qualification,
             'elf_sha256': 'image', 'build_hash': 'build'}
    if mutation == 'structural':
        build.pop('native_qualification')
    elif mutation == 'other_partition':
        qualification['partition']['path'] = str(tmp_path / 'other.json')
    elif mutation == 'other_image':
        build['elf_sha256'] = 'other'
    elif mutation == 'other_program':
        qualification['program'] = 'other'
    save('work/programs/program/build.json', build)
    report = {'programs': [{'program': 'program', 'build': build}]}
    kwargs = dict(partition_path=partition, model=N(programs=[N(name='program')]),
                  report=report, work=tmp_path / 'work/runs')
    if mutation is None:
        R._native_images(R._Files(), **kwargs)
    else:
        with pytest.raises(ValueError, match='independently admitted'):
            R._native_images(R._Files(), **kwargs)
