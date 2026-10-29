"""Development queue coverage never replaces full-suite or physical qualification."""
import json
from pathlib import Path

import pytest

from merlin_experiments.phase1.feedback import joint_curriculum as C
from merlin_experiments.phase1.feedback import joint_test_scope as S
from merlin_experiments.phase1.feedback import rolling_scalability as R


def case(root, name, label='public'):
    path = root / name
    path.mkdir(parents=True)
    (path / 'capsule.yaml').write_text(json.dumps({
        'name': name, 'label': label, 'inputs': [{'shape': [16, 16]}],
        'operation': {'op': 'matmul'}}))
    return path


def test_one_250_case_queue_with_original_priority_and_eventual_scale(tmp_path, monkeypatch):
    core = tmp_path / 'core'
    original = [f'original{i:03}' for i in range(153)]
    for name in original + [f'calibration{i:02}' for i in range(20)]:
        case(core, name)
    case(core, 'hidden', label='hidden')
    scale = {f'scale{i:02}': case(tmp_path / 'scale', f'scale{i:02}') for i in range(77)}
    policy = {'correctness_curriculum': {'original_capsules': original}}
    monkeypatch.setattr(S.joint_control, 'selected', lambda: {'policy': policy})
    monkeypatch.delenv('MERLIN_JOINT_GRADE_MILESTONE', raising=False)
    p = C.Progress(tmp_path / 'progress', identity=None, runs_root=tmp_path / 'runs')
    paths, report = S.select(core, submission=tmp_path / 'candidate', curriculum=p, additional_cases=scale)
    assert len(paths) == len(set(paths)) == report['required_development'] == 250
    assert report['cohorts'] == {'core_and_calibration': 173, 'public_scalability': 77}
    assert len(report['pending']) == 250
    assert set(path.name for path in paths[:153]) == set(original)
    # Admit and drain without successful evidence. Untouched cases move ahead
    # of repeated failures; there is no fixed original-cohort barrier.
    for name in original:
        p.started({'name': name})
    p.release()
    paths, report = S.select(core, submission=tmp_path / 'candidate', curriculum=p, additional_cases=scale)
    assert set(path.name for path in paths[:97]) == set(scale) | {f'calibration{i:02}' for i in range(20)}
    assert not report['curriculum']['certified_complete']
    assert not report['curriculum']['exact_correctness_reused']


@pytest.mark.parametrize('bad', ['duplicate', 'private', 'wrong_name'])
def test_cross_suite_ambiguity_or_nonpublic_member_refuses(tmp_path, monkeypatch, bad):
    core = tmp_path / 'core'
    case(core, 'original')
    name = 'original' if bad == 'duplicate' else 'scale'
    path = case(tmp_path / 'scale', name, label='hidden' if bad == 'private' else 'public')
    if bad == 'wrong_name':
        doc = json.loads((path / 'capsule.yaml').read_text())
        doc['name'] = 'another'
        (path / 'capsule.yaml').write_text(json.dumps(doc))
    monkeypatch.setattr(S.joint_control, 'selected', lambda: {'policy': {}})
    with pytest.raises(ValueError):
        S.select(core, submission=tmp_path / 'candidate', additional_cases={name: path})


def test_frozen_suite_identity_reopens_before_reuse(tmp_path, monkeypatch):
    root = tmp_path / 'frozen'
    record = {'cases': {'scale': 'a' * 64}, 'source': {'policy': 'spike-only'}}
    selected = [root, record]
    monkeypatch.setattr(R.public_scalability, 'selected', lambda _: tuple(selected))
    selection = R.Selection(object())
    assert selection.cases == {'scale': root / 'cases' / 'scale'}
    assert selection.identity == {'root': str(root), 'record': record}
    selection.verify()
    selected[1] = {**record, 'cases': {'scale': 'b' * 64}}
    with pytest.raises(ValueError, match='selection changed'):
        selection.verify()


def test_scalability_callback_does_not_promote_to_gsim(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen import capsule_common
    from merlin_experiments.phase1.feedback import loop_grading as L, promotion

    monkeypatch.setattr(L, 'load_target_experiment', lambda _: SimpleNamespace(
        target='test', effective_exclusions=lambda _: ()))
    monkeypatch.setattr(capsule_common, 'discover_capsules', lambda *a, **k: [{'name': 'original'}])
    monkeypatch.setattr(promotion, 'promote', lambda *a, **k: pytest.fail('scale reached physical promotion'))
    inputs = SimpleNamespace(context=SimpleNamespace(descriptor=tmp_path / 'descriptor'),
        public_root=tmp_path / 'core', contract=None, promotion_root=tmp_path / 'core')
    complete = L._completed_loop_promoter(tmp_path, tmp_path, tmp_path, 'L2', 'L3', {'original'}, inputs=inputs)
    complete({'capsule': 'scale', 'tiers': {'L2': {'status': 'pass'}}})
