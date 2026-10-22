"""Native cohort membership separates optimization advice without changing grades."""
import copy

from merlin_experiments.phase1.feedback import joint_performance_priority as P
from merlin_experiments.phase1.feedback import joint_test_scope as T


def row(name, headroom=10, status='PROVISIONAL_DIAGNOSTIC'):
    return {'capsule': name, 'numeric': {'status': 'pass'},
            'performance': {'search_priority': {'status': status, 'diagnostic_headroom': headroom,
                            'gap_ratio': 2, 'comparison_group': {'model_sha256': 'same'}}}}


def names(result):
    return [r['capsule'] for g in result['comparison_groups']
            for lane in ('qualified', 'provisional_diagnostics') for r in g[lane]]


def test_primary_ranking_ignores_large_microcase_headroom_but_retains_observations():
    rows = [row('large'), row('small', 999999), row('calibration', 99999),
            row('small_failure', status='CORRECTNESS_FIRST')]
    before = copy.deepcopy(rows)
    result = P.summarize(rows, cohorts={'primary': ['large'],
        'correctness_regression': ['small', 'small_failure'], 'calibration_diagnostics': ['calibration']})
    assert names(result) == ['large']
    assert names(result['correctness_regression']) == ['small']
    assert names(result['calibration_diagnostics']) == ['calibration']
    assert [r['capsule'] for r in result['correctness_first']] == ['small_failure']
    assert rows == before
    assert names(P.summarize(rows)) == ['small', 'calibration', 'large']
    assert result['promotion_authority'] is False


def test_missing_unknown_and_overlapping_members_are_not_primary():
    rows = [row('large'), row('unknown', 9000), row('overlap', 8000)]
    rows[1]['cohort'] = 'primary'  # An authored label confers no membership.
    result = P.summarize(rows, cohorts={'primary': ['large', 'overlap'],
        'correctness_regression': ['overlap'], 'calibration_diagnostics': []})
    assert names(result) == ['large']
    assert {r['capsule'] for r in result['unclassified']} == {'unknown', 'overlap'}
    assert result['cohort_selection_status'] == 'invalid_or_overlapping'
    missing = P.summarize(rows, cohorts={})
    assert not names(missing) and len(missing['unclassified']) == 3
    assert not names(P.summarize(rows, cohorts={'primary': ['large', 'large'],
        'correctness_regression': [], 'calibration_diagnostics': []}))


def test_native_selection_uses_suite_and_policy_not_candidate_hint(tmp_path, monkeypatch):
    root, submission = tmp_path/'core', tmp_path/'compiler'
    submission.mkdir()
    (submission/'joint-test-scope.json').write_text('{"capsules":["small"],"performance_cohorts":{"primary":["small"]}}')
    paths = {}
    for name, parent in [('small', root), ('diagnostic', root), ('large', tmp_path/'scale')]:
        path = parent/name; path.mkdir(parents=True)
        (path/'capsule.yaml').write_text(f'name: {name}\nlabel: public\nkind: operator\ninputs: []\n')
        paths[name] = path
    state = {'policy': {'correctness_curriculum': {'original_capsules': ['small']},
                        'gsim': {'witness_capsules': ['diagnostic']}, 'final_only_capsules': []}}
    monkeypatch.setattr(T.joint_control, 'selected', lambda: state)
    monkeypatch.delenv('MERLIN_JOINT_GRADE_MILESTONE', raising=False)
    _, selection = T.select(root, submission=submission, additional_cases={'large': paths['large']})
    assert selection['performance_cohorts'] == {'primary': ['large'],
        'correctness_regression': ['small'], 'calibration_diagnostics': ['diagnostic']}
    assert selection['required_development'] == 3
    verdict = {'per_capsule': [row('small'), row('large')], 'feedback_brief': {}}
    T.annotate(verdict, selection)
    assert names(verdict['feedback_brief']['performance_priority']) == ['large']
    assert verdict['per_capsule'][0]['numeric']['status'] == 'pass'
