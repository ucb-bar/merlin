import json
from pathlib import Path

from merlin_experiments.phase1.feedback import joint_test_scope as S


def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(S.joint_control, 'selected', lambda: {'policy': {'final_only_capsules': ['final']}})
    monkeypatch.delenv('MERLIN_JOINT_GRADE_MILESTONE', raising=False)
    root = tmp_path / 'corpus'
    for name, family, extent, label in [('m1', 'matmul', 1, 'public'), ('m2', 'matmul', 2, 'dev'),
                                        ('m3', 'matmul', 3, 'public'), ('conv', 'conv', 2, 'public'),
                                        ('hidden', 'matmul', 1, 'hidden'), ('final', 'model', 1, 'public')]:
        path = root / name
        path.mkdir(parents=True)
        (path / 'capsule.yaml').write_text(json.dumps({'label': label, 'operation': {'op': family},
                                                     'inputs': [{'shape': [extent]}]}))
    candidate = tmp_path / 'submission'
    candidate.mkdir()
    return root, candidate


def test_unknown_impact_smoke_never_claims_full_coverage(tmp_path, monkeypatch):
    root, candidate = corpus(tmp_path, monkeypatch)
    roots, scope = S.select(root, submission=candidate)
    assert {p.name for p in roots} == {'m1', 'm3', 'conv'}
    verdict = S.annotate({'all_pass': True, 'n_capsules': 3, 'n_passed': 3}, scope)
    assert verdict['all_pass'] is None
    assert verdict['n_required_development'] == 4
    assert verdict['n_pending_development'] == 1
    assert verdict['n_capsules'] == 3


def test_explicit_family_and_milestone(tmp_path, monkeypatch):
    root, candidate = corpus(tmp_path, monkeypatch)
    (candidate / 'joint-test-scope.json').write_text('{"families":["conv"]}')
    roots, scope = S.select(root, submission=candidate)
    assert [p.name for p in roots] == ['conv']
    assert len(scope['pending']) == 3
    monkeypatch.setenv('MERLIN_JOINT_GRADE_MILESTONE', '1')
    roots, scope = S.select(root, submission=candidate)
    assert len(roots) == 4 and not scope['pending']
    monkeypatch.setattr(S.joint_control, 'selected', lambda: None)
    assert S.select(root, submission=candidate) == (root, None)


def test_package_failure_keeps_every_obligation_pending_and_actionable(tmp_path, monkeypatch):
    root, candidate = corpus(tmp_path, monkeypatch)
    _, scope = S.select(root, submission=candidate)
    verdict = S.annotate({'all_pass': None, 'n_capsules': 0, 'n_passed': 0, 'per_capsule': [],
                         'package_failure': {'plane': 'contract', 'detail': 'shared_pass_contract missing fields'}}, scope)
    assert verdict['all_pass'] is False
    assert verdict['n_pending_development'] == verdict['n_required_development'] == 4
    assert verdict['n_selected_development'] == 3
    assert verdict['feedback_brief']['first_actionable_failure']['detail'] == 'shared_pass_contract missing fields'
    assert verdict['n_capsules'] == 0
