"""Whole-model cases reach L3: promoted from their Spike pass, kept from a self-check, closed finally.

Measured on a fresh four-hour run: all four whole-model cases passed L0-L3 on gSIM in the author's own
self-checks, yet none of 394 promotion requests was a model (their rows carry the grader's native-tier
gap, which the physical-correctness admission never recognised), the self-check certificates were not
kept, the grade showed them L3 "unverified", and final coverage filed them as unexplained skips that
withheld every native core pass. Promotion stays fail-open; these tests are what makes it audible.
"""
import io
import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import pytest

from merlin.common.tree_hash import hash_tree
from merlin.targetgen import capsule_grade as CG
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1 import joint_final_coverage as F
from merlin_experiments.phase1 import joint_final_model_l3 as M
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import joint_physical_correctness as Q
from merlin_experiments.phase1.feedback import promotion as P
from merlin_experiments.phase1.feedback import selfcheck_certificates as S
from merlin_experiments.phase1.feedback import snapshot_execution
from test_current_source_promotions import workspace

INSTRUMENT = 'a' * 64
GAP = {'plane': 'candidate_model_execution', 'category': 'NOT_RUN_IS_NOT_PASS',
       'detail': 'mandatory native tier has not completed'}


@pytest.fixture
def ws(tmp_path, monkeypatch):
    policy = {'gsim_correctness': {'capsules': ['MODEL', 'OP'], 'max_jobs': 20, 'per_job_s': 1200,
                                   'worker_budget_s': 864000}}
    monkeypatch.setattr(joint_control, 'selected', lambda: {'policy': policy})
    monkeypatch.setattr(P, 'cert_sim', lambda *a, **k: 'gsim')
    monkeypatch.setattr(P, 'cert_instrument', lambda *a, **k: INSTRUMENT)
    root = workspace(tmp_path / 'ws')
    (root / '.qa_channel').mkdir()
    return root


def model_row(source, **changes):
    row = {'capsule': 'MODEL', 'kind': 'model', 'status': 'incomplete', 'failure': dict(GAP),
           'numeric': {'status': 'pass', 'mismatch_count': 0, 'golden_source': 'host_torch_eager',
                       'per_output': {'Y0': {'status': 'pass', 'n_elements': 1024}}},
           'tiers': {'L0': {'status': 'pass', 'mandatory': True}, 'L1': {'status': 'pass', 'mandatory': True},
                     'L2': {'status': 'pass', 'mandatory': True, 'engine': 'spike'},
                     'L3': {'status': 'unverified', 'mandatory': True}},
           'model_execution_check': {'status': 'incomplete', 'violations': []},
           'snapshot_external_model_child': {'compiler_sha256': hash_tree(source)['sha256']}}
    row.update(changes)
    return row


def promote(ws, row):
    return P.promote(ws, ws / '.qa_channel', {'per_capsule': [{'capsule': row['capsule'], 'pass': True}]},
                     'L2', 'L3', set(), io.StringIO(), context=None, closed_loop_rows={row['capsule']: row})


def report(ws):
    return json.loads((ws / 'qa/joint_physical_correctness.json').read_text())


def test_spike_passing_model_is_promoted_to_its_physical_l3(ws, tmp_path, monkeypatch):
    assert promote(ws, model_row(ws / 'submission')) == ['MODEL']
    request = json.loads(next((ws / '.qa_channel').glob('simreq*.json')).read_text())
    assert request['sim'] == 'gsim' and request['tiers'] == 'L3' and request['joint_physical_correctness']
    assert Q.selected_request(request)
    assert joint_control.physical_policy('MODEL')['per_job_s'] == 1200  # the correctness ledger's budget
    # The broker launches a model's certificate with its own L2 (snapshot execution admits by request).
    caps = tmp_path / 'caps'
    (caps / 'MODEL').mkdir(parents=True)
    (caps / 'MODEL/capsule.yaml').write_text('kind: model\n')
    monkeypatch.setenv(snapshot_execution.ENV, snapshot_execution.MODE)
    assert simjob._promoted_model_continuation(ws, ['MODEL'], caps, 'L3', digest=request['submission_digest'],
                                               context=None, contract=None, cert_tier='L3') == ('L2,L3', [])
    native = {'per_capsule': [{'capsule': 'MODEL', 'pass': True, 'barrier_tier': 'L3', 'barrier_status': 'pass'}]}
    assert P.record_cert(ws, native, 'L3', identity=request['identity'], source_identity_verified=True) == [
        'MODEL=pass']
    assert report(ws)['obligations']['MODEL']['status'] == 'pass'


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(kind='op'),
    lambda r: r.update(failure={**GAP, 'detail': 'linked group 1 host-compute audit is unknown'}),
    lambda r: r['tiers']['L3'].update(status='fail'),
    lambda r: r['tiers']['L2'].update(engine='gsim'),
    lambda r: r['tiers']['L1'].update(status='unknown'),
    lambda r: r['numeric'].update(status='fail'),
    lambda r: r['numeric']['per_output']['Y0'].update(n_elements=0),
    lambda r: r['model_execution_check'].update(violations=['source placement differs']),
    lambda r: r['snapshot_external_model_child'].update(compiler_sha256='f' * 64),
    lambda r: r.update(capsule='UNLISTED'),
], ids=['not_model', 'other_failure', 'l3_failed', 'l2_not_spike', 'structural_open', 'numeric_fail',
        'empty_output', 'violation', 'other_compiler', 'outside_roster'])
def test_only_the_exact_native_l3_gap_of_this_snapshot_is_promoted(ws, mutate):
    row = model_row(ws / 'submission')
    mutate(row)
    assert promote(ws, row) == []
    assert not list((ws / '.qa_channel').glob('simreq*.json'))


def selfcheck_verdict(**row_changes):
    row = {'capsule': 'MODEL', 'pass': True, 'certified_complete': True, 'barrier_tier': 'L3',
           'barrier_declared': 'L3', 'barrier_status': 'pass', 'execution_digest': None, 'screened_only': False}
    row.update(row_changes)
    return {'all_pass': True, 'oracle_tiers': ['L2', 'L3'], 'tier_engines': {'L2': 'spike', 'L3': 'gsim'},
            'per_capsule': [row]}


def snapshot(ws, tmp_path):
    source = tmp_path / 'snapshot'
    shutil.copytree(ws / 'submission', source / 'submission')
    return source


def test_self_check_l3_is_kept_against_the_snapshot_it_graded(ws, tmp_path):
    source = snapshot(ws, tmp_path)
    (ws / 'submission/compiler.py').write_text('def main(): return 2\n')  # edited after the check
    log = io.StringIO()
    assert S.import_executed(ws, selfcheck_verdict(), loop_tier='L2', cert_tier='L3', source_ws=source,
                             context=None, log=log) == ['MODEL=pass']
    digest = P.submission_digests(source)[0]
    assert digest != P._submission_digest(ws)
    slots = P._slots_ro(P._tier_state(ws), 'MODEL', 'L3')
    assert list(slots) == [f'submission:{digest}']
    entry = slots[f'submission:{digest}']
    assert entry['status'] == 'pass' and entry['instrument'] == INSTRUMENT and entry['origin'] == 'author_selfcheck'
    Q.project(ws, P._tier_state(ws), source=digest, instrument=INSTRUMENT)
    assert report(ws)['obligations']['MODEL']['status'] == 'pass'
    assert 'certificates kept' in log.getvalue()
    # The same bytes are not imported twice, and a later promotion for them is not enqueued.
    assert S.import_executed(ws, selfcheck_verdict(), loop_tier='L2', cert_tier='L3', source_ws=source,
                             context=None) == []


@pytest.mark.parametrize('change', [
    {'row': {'certified_complete': False}}, {'row': {'barrier_status': 'unknown'}}, {'row': {'pass': False}},
    {'row': {'capsule': 'UNLISTED'}}, {'row': {'screened_only': True}},
    {'verdict': {'tier_engines': {'L2': 'spike', 'L3': 'verilator'}}},
    {'verdict': {'oracle_tiers': ['L2']}}, {'verdict': {'error': 'aborted'}},
    {'instrument': None}, {'joint': None},
], ids=lambda c: next(iter(c)) + ':' + json.dumps(next(iter(c.values())))[:40])
def test_self_check_l3_needs_a_certified_barrier_on_the_promotion_engine_and_judge(ws, tmp_path, monkeypatch,
                                                                                  change):
    verdict = selfcheck_verdict(**change.get('row', {}))
    verdict.update(change.get('verdict', {}))
    if 'instrument' in change:
        monkeypatch.setattr(P, 'cert_instrument', lambda *a, **k: None)
    if 'joint' in change:
        monkeypatch.setattr(joint_control, 'selected', lambda: None)
    assert S.import_executed(ws, verdict, loop_tier='L2', cert_tier='L3', source_ws=snapshot(ws, tmp_path),
                             context=None) == []
    assert P._slots_ro(P._tier_state(ws), 'MODEL', 'L3') == {}


def test_an_unrecordable_self_check_leaves_nothing_outstanding(ws, tmp_path, monkeypatch):
    refusing = SimpleNamespace(__wrapped__=lambda *a, **k: [])
    monkeypatch.setattr(P, 'record_cert', refusing)
    assert S.import_executed(ws, selfcheck_verdict(), loop_tier='L2', cert_tier='L3',
                             source_ws=snapshot(ws, tmp_path), context=None) == []
    assert P._slots_ro(P._tier_state(ws), 'MODEL', 'L3') == {}


def test_the_selfcheck_broker_keeps_an_already_executed_certificate():
    from pathlib import Path

    from merlin_experiments.phase1.brokers import selfcheck as broker

    source = Path(broker.__file__).read_text()
    branch = source[source.index('if cert_already_executed(_v, _cert):'):]
    branch = branch[:branch.index('elif isinstance(_v, dict)')]
    assert 'import_executed(' in branch and 'source_ws=promotion_source' in branch


# -- final coverage on the Spike-only route ---------------------------------------------------------
def _final(tmp_path, ws, *, cert=None):
    frozen = tmp_path / 'frozen'
    shutil.copytree(ws / 'submission', frozen)
    runs = tmp_path / 'core'
    result = runs / 'runs/suite/MODEL/capsule_result.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(model_row(frozen)))
    if cert is not None:
        state = P._tier_state(ws)
        P._record(state, 'MODEL', 'L3', dict({'status': 'pass', 'digest': P.submission_tree_digest(frozen),
                                              'instrument': INSTRUMENT}, **cert))
        P._save_tier_state(ws, state)
    score = {'per_capsule': [{'capsule': 'MODEL', 'kind': 'model', 'status': 'incomplete'},
                             {'capsule': 'OP', 'status': 'pass'}]}
    return M.close(score, runs=runs, workspace=ws, frozen=frozen, compiler_sha256=hash_tree(frozen)['sha256'],
                   context=None)


def test_final_model_row_takes_the_exact_byte_l3_certificate(ws, tmp_path):
    closed = _final(tmp_path, ws, cert={})
    assert closed['MODEL']['certificate']['instrument'] == INSTRUMENT
    assert set(closed) == {'MODEL'}


@pytest.mark.parametrize('cert', [None, {'digest': 'other'}, {'instrument': 'b' * 64}, {'status': 'pending'},
                                  {'status': 'fail'}], ids=['none', 'other_bytes', 'other_judge', 'pending', 'fail'])
def test_final_model_row_without_that_certificate_says_l3_evidence_missing(ws, tmp_path, cert):
    closed = _final(tmp_path, ws, cert=cert)
    assert 'certificate' not in closed['MODEL']
    assert closed['MODEL']['missing'].startswith('L3 evidence missing')


def _score(names, model=None):
    rows = [{'capsule': n, 'status': 'pass', 'numeric': 'pass', 'tiers': {}} for n in names]
    if model:
        rows.append({'capsule': model, 'kind': 'model', 'status': 'incomplete', 'numeric': 'pass',
                     'tiers': {'L0': 'pass', 'L1': 'pass', 'L2': 'pass', 'L3': 'unverified'}})
    every = names + ([model] if model else [])
    return {'gradeable': True, 'n_capsules': len(every), 'n_passed': len(names), 'n_incomplete': int(bool(model)),
            'cohort_admission': {'n_admitted_capsules': len(every),
                                 'admitted_name_set_sha256': CG._name_set_sha256(every)},
            'per_capsule': rows, 'numeric_all_exact': True, 'structural_evidence_all_pass': True,
            'functional_public_evidence': {'claim': 'functional_corroboration'}}


def test_a_certified_model_row_closes_the_core_cohort():
    group = F._capsule_group([{'name': 'a'}, {'name': 'm'}])
    F._apply_score(group, _score(['a'], 'm'), {'m': {'certificate': {'identity': 'submission:x'}}})
    rollup = F._rollup({'groups': {'core': group}})
    assert group['obligations']['m']['status'] == 'pass' and group['obligations']['m']['l3_certificate']
    assert group['native_cohort_complete'] is False and group['passed'] is True
    assert rollup['passed'] is True and rollup['n_l3_evidence_missing'] == 0


def test_a_model_row_without_l3_evidence_is_reported_not_unfinished():
    group = F._capsule_group([{'name': 'a'}, {'name': 'm'}])
    F._apply_score(group, _score(['a'], 'm'), {'m': {'missing': 'L3 evidence missing: none recorded'}})
    rollup = F._rollup({'groups': {'core': group}})
    assert group['obligations']['m'] == {**group['obligations']['m'], 'status': M.MISSING,
                                         'reason': 'L3 evidence missing: none recorded'}
    assert group['model_l3'] == {'certified': [], 'missing': ['m']}
    assert rollup['n_l3_evidence_missing'] == 1 and rollup['n_unfinished'] == 0
    assert rollup['passed'] is False and group['passed'] is False
    # Without any closure the native veto is unchanged: the row stays an unexplained skip.
    group = F._capsule_group([{'name': 'a'}, {'name': 'm'}])
    F._apply_score(group, _score(['a'], 'm'))
    assert group['obligations']['m']['status'] == 'skipped' and group['passed'] is False


def test_final_coverage_closes_core_model_rows(tmp_path, monkeypatch):
    from test_joint_final_coverage import _native_fixture

    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    seen = []

    def close(score, **kwargs):
        seen.append(kwargs['runs'].name)
        assert kwargs['frozen'] == frozen and kwargs['compiler_sha256'] == digest
        return {'core0': {'missing': 'L3 evidence missing: test'}}

    monkeypatch.setattr(M, 'close', close)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert seen == ['core']  # only the core group carries whole-model rows
    assert report['groups']['core']['obligations']['core0']['status'] == M.MISSING
    assert report['n_l3_evidence_missing'] == 1 and report['passed'] is False


def test_model_l3_predicate_matches_the_grader_gap_text():
    from pathlib import Path

    source = Path(CG.__file__).read_text()
    assert repr(GAP['detail'])[1:-1] in source and Q._MODEL_NATIVE_TIER_GAP == GAP
    assert deepcopy(GAP) == Q._MODEL_NATIVE_TIER_GAP
