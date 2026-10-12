from merlin_experiments.phase1.feedback import qa, selfcheck

from merlin.targetgen.capsule_grade import _candidate_score_view


def owned_row():
    tiers = {t: {'status': 'pass'} for t in ('L0', 'L1', 'L2')}
    tiers['L2']['numeric'] = {'status': 'pass', 'mismatch_count': 0}
    tiers['L3'] = {'status': 'unverified'}
    check = {
        'kind': 'owned_host_model_execution', 'status': 'incomplete', 'violations': [],
        'candidate_source_placement': {'status': 'clean'},
        'candidate_completed_dispatch': {'status': 'verified'},
        'candidate_source_coverage': {'status': 'verified', 'n_source_operations': 8,
                                      'n_eligible': 2, 'n_completed_eligible': 2},
        'candidate_required_tiers': {'status': 'incomplete', 'required_tiers': list(tiers),
                                     'pending_tiers': ['L3'], 'tiers': tiers},
    }
    return {'kind': 'model', 'status': 'incomplete', 'model_execution_check': check}


def project(row, barrier='L2'):
    summary = qa._candidate_native_feedback(row)
    return selfcheck._candidate_selfcheck_row(
        row, {'candidate_native_verification': summary}, name='model', barrier_tier=barrier)


def test_pending_native_tier_retains_screen_without_final_credit():
    row = owned_row()
    screened, certified = project(row)
    assert screened['pass'] and screened['screened_only']
    assert not certified and not screened['certified_complete']
    assert not project(row, 'L3')[0]['pass']
    scored = _candidate_score_view(row)
    assert scored['numeric']['status'] == 'pass'
    assert scored['tiers']['L2']['status'] == 'pass'
    assert scored['tiers']['L3']['status'] == 'unverified'
    assert scored['status'] == 'incomplete'


def test_pending_native_never_hides_bad_source_or_numeric():
    for field, value in [('candidate_source_placement', {'status': 'unverified'}),
                         ('candidate_source_coverage', {'status': 'unverified'}),
                         ('candidate_completed_dispatch', {'status': 'unverified'})]:
        row = owned_row()
        row['model_execution_check'][field] = value
        assert not project(row)[0]['pass']
    row = owned_row()
    row['model_execution_check']['candidate_required_tiers']['tiers']['L2']['status'] = 'fail'
    assert not project(row)[0]['pass']


def test_raw_owned_runtime_receipt_cannot_authorize_screen():
    assert qa._candidate_native_feedback({'owned_host_native_execution': owned_row()}) is None


def test_enforcement_preserves_only_audited_tiers(monkeypatch):
    from merlin.targetgen import capsule_grade
    row = owned_row()
    check = row['model_execution_check']
    row['tiers'] = {t: {'status': 'pass'} for t in ('L0', 'L1', 'L2', 'L3')}
    monkeypatch.setattr(capsule_grade, 'model_execution_check', lambda *a, **k: check)
    capsule_grade.enforce_model_execution_check(row, {}, target='target')
    assert row['tiers']['L2']['status'] == 'pass'
    assert row['tiers']['L3']['status'] == 'unverified'
    assert row['status'] == 'incomplete'
    refusal = {'kind': 'owned_host_model_execution', 'status': 'incomplete',
               'violations': ['candidate_completed_dispatch_unverified']}
    monkeypatch.setattr(capsule_grade, 'model_execution_check', lambda *a, **k: refusal)
    capsule_grade.enforce_model_execution_check(row, {}, target='target')
    assert all(t['status'] == 'unverified' for t in row['tiers'].values())
    assert not project(row)[0]['pass']
