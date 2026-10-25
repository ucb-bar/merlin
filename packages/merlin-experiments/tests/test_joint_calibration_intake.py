import json
import pytest

from merlin_experiments.phase1 import joint_calibration_intake as intake
from test_joint_calibration_eligibility import valid_grade


def test_scope_and_structural_limits_remain_independent():
    kwargs = dict(capsule={'name': 'public'}, result={'numeric': {'status': 'pass'}},
                  performance={'structural_qualification': {'unavailable_fields': ['reuse']}},
                  source_sha256='source', compiler_sha256='compiler', elf={'sha256': 'elf'},
                  selection_pin={'sha256': 'selection'}, facts_pin={'sha256': 'facts'})
    report = {'scope_binding': {'elf_sha256': 'elf', 'input_sha256': 'input', 'scope_sha256': 'scope'},
              'feature_observation': {'complete': True}}
    row = intake.project(**kwargs, report=report)
    assert row['protocol_status'] == 'complete_observation'
    assert row['structural_qualification']['unavailable_fields'] == ['reuse']
    assert row['source_group_sha256'] is None  # A new shape does not prove independence.
    assert not row['eligible_candidate'] and not row['fit_authorized']
    assert not row['dispatch_authorized']
    report['scope_binding']['elf_sha256'] = 'foreign'
    row = intake.project(**kwargs, report=report)
    assert row['scope_binding'] is None
    assert row['protocol_status'] == 'unknown_or_incomplete'


def test_public_intake_is_bounded_and_separate_from_fit_registry(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import joint_performance as perf
    from merlin_experiments.phase2 import census_launch
    from merlin.runtime import _native_files as files

    root = tmp_path / 'product'
    root.mkdir()
    elf = root / 'program.elf'
    elf.write_bytes(b'fixture only; no execution')
    compiler = tmp_path / 'compiler'
    compiler.mkdir()
    (compiler / 'candidate.py').write_text('pass')
    capsule = {'name': 'new_public', 'kind': 'layer'}
    selection = {'members': {'new_public': {'capsule_sha256': perf.capsule_identity(capsule)}},
                 'facts': {'sha256': 'facts'}}
    monkeypatch.setattr(perf, '_read', lambda pin: selection)
    monkeypatch.setattr(perf, 'selected_capsule', lambda selection, capsule: capsule, raising=False)
    monkeypatch.setattr(perf, 'original_executable', lambda root: elf, raising=False)
    # Only custody verification is replaced; no subprocess or simulator is invoked.
    monkeypatch.setattr(census_launch, '_producer', lambda elf, limits: files.pin(elf, max_bytes=1024))
    monkeypatch.setenv(joint_control.ENV, str(tmp_path / 'state.json'))
    profile = {'performance_selection': {'sha256': 'selected'}, 'members': {}}
    state = {'policy': {}}
    result = valid_grade()
    kwargs = dict(result=result, compiler=compiler, capsule=capsule, state=state, profile=profile)
    pin = intake.register({}, root, **kwargs)
    record = json.loads(open(pin['path']).read())
    assert record['numerical_status'] == 'pass'
    assert record['source_group_sha256'] is None
    assert not (tmp_path / 'calibration_registry').exists()
    assert intake.register({}, root, **kwargs) == pin
    assert intake.register({}, root, **dict(kwargs, capsule={'name': 'hidden', 'kind': 'layer'})) is None
    assert intake.register({}, root, **dict(kwargs, capsule={'name': 'new_public', 'kind': 'model'})) is None
    assert intake.register({}, root, **dict(kwargs, result={'numeric': {'status': 'fail'}})) is None
    monkeypatch.setattr(intake, 'MAX_ENTRIES', 1)
    (compiler / 'candidate.py').write_text('pass # changed')
    with pytest.raises(ValueError, match='bounded capacity'):
        intake.register({}, root, **kwargs)


def test_unknown_original_producer_cannot_enter_index(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import joint_performance as perf
    from merlin_experiments.phase2 import census_launch
    monkeypatch.setattr(perf, '_read', lambda pin: {'members': {'public': {}}, 'facts': {}})
    monkeypatch.setattr(perf, 'selected_capsule', lambda selection, capsule: capsule, raising=False)
    monkeypatch.setattr(perf, 'original_executable', lambda root: root / 'program.elf', raising=False)
    def refuse(elf, limits):
        raise ValueError('no original completed link producer')
    monkeypatch.setattr(census_launch, '_producer', refuse)
    monkeypatch.setenv(joint_control.ENV, str(tmp_path / 'state.json'))
    with pytest.raises(ValueError, match='original completed link'):
        intake.register({}, tmp_path, result=valid_grade(),
                        compiler=tmp_path, capsule={'name': 'public', 'kind': 'layer'},
                        state={'policy': {}}, profile={'performance_selection': {}, 'members': {}})
    assert not (tmp_path / 'calibration_candidate_index').exists()
