"""Read-only counts preserve source, engine, refusal and missing-data boundaries."""
import hashlib
import json

from merlin_experiments.tracking.terminal_capsules import Capsules, project

A, B = 'a' * 64, 'b' * 64
ROSTERS = {'functional': {'case', 'refusal'}, 'calibration': {'cal'}, 'scalability': {'scale'}}


def row(name, *, tiers=('L0', 'L1', 'L2'), status='pass', numeric='pass'):
    return {'capsule': name, 'status': status, 'numeric': {'status': numeric}, 'candidate_native_execution': {'status': 'recorded'},
            'tiers': {t: {'status': 'pass', 'mandatory': True, 'engine': {'L2': 'spike', 'L3': 'gsim'}.get(t)} for t in tiers}}


def report(source, rows):
    return {'graded_source_sha256': source, 'per_capsule': rows,
            'snapshot_current': True, 'updated_at': '2026-10-22T17:00:00Z'}


def cert(source, digest, tier, name='case'):
    return {'status': 'pass', 'digest': digest, 'instrument': 'instrument', 'tier_boundary': {'tier': tier, 'engine': {'L2': 'spike', 'L3': 'gsim'}[tier],
            'common': {'capsule': name, 'compiler_sha256': source, 'submission_digest': digest}}}


def test_engine_counts_never_join_different_sources_and_refusal_is_not_numeric():
    refuse = row('refusal', tiers=(), numeric='skipped')
    refuse['expected_refusal'] = {'status': 'met'}
    refuse['declined'] = {'reason': 'declared negative contract'}
    l2 = row('case', status='incomplete')
    l2['tiers']['L3'] = {'status': 'unavailable', 'mandatory': True}
    physical = {'schema': 'merlin.joint_physical_correctness.v1', 'instrument_sha256': 'instrument', 'source_sha256': 'shortB',
                'obligations': {'case': {'status': 'pass'}}}
    ledger = {'case': {'L3': cert(B, 'shortB', 'L3')}}
    result = project(ROSTERS, report(A, [l2, refuse]), ledger, physical, [])
    assert result['total'] == {'total': 4, 'all': 1, 'L2': 1, 'L3': 0, 'refusal': 1}
    assert result['historical_union']['L3'] == 1
    assert result['historical_union']['all'] == 1  # Cross-source L2+L3 never complete the case.
    assert result['source_current'] is None  # Stale boolean is not author-current proof.
    assert result['completed_perf_evaluations'] is None


def test_native_physical_projection_uses_explicit_digest_bridge_and_conflicts_remain_visible():
    ledger = {'case': {'L2': cert(A, 'shortA', 'L2'), 'L3': cert(A, 'shortA', 'L3')}}
    physical = {'schema': 'merlin.joint_physical_correctness.v1', 'instrument_sha256': 'instrument', 'source_sha256': 'shortA',
                'obligations': {'case': {'status': 'pass'}}}
    result = project(ROSTERS, report(A, []), ledger, physical, [])
    assert result['total']['L2'] == result['total']['L3'] == 1
    assert result['total']['all'] == 0  # Physical-only projection cannot create a full-row verdict.
    # Same alias associated with two sources is not resolvable.
    ledger['scale'] = {'L2': cert(B, 'shortA', 'L2', 'scale')}
    result = project(ROSTERS, report(A, []), ledger, physical, [])
    assert result['total']['all'] == 0
    result = project(ROSTERS, report(A, [row('case', status='fail', numeric='fail')]),
                     {'case': {'L2': cert(A, 'shortA', 'L2')}}, physical, [])
    assert result['total']['all'] == 0


def test_missing_rosters_and_unclassified_names_are_not_prefix_guessed():
    result = project({'functional': {'case'}, 'calibration': None, 'scalability': None},
                     report(A, [row('SY_unclassified')]), {}, {}, [])
    assert result['total']['total'] is None and result['total']['L2'] is None
    assert result['unclassified_observed'] == 1
    assert result['cohorts'][1]['total'] is None
    result = project(ROSTERS, {}, {}, {}, [])
    assert result['total']['L2'] is None


def test_perf_evaluation_count_requires_matched_metric_and_deduplicates_document():
    from test_terminal_metadata import performance
    first, second = row('case'), row('scale')
    first['performance'] = performance(); second['performance'] = performance()
    result = project(ROSTERS, report(A, []), {}, {}, [(report(A, [first, second]), 'response-one')])
    assert result['completed_perf_evaluations'] == 1
    second['performance']['scope_sha256'] = 'wrong'
    result = project(ROSTERS, report(A, []), {}, {}, [(report(A, [second]), 'response-two')])
    assert result['completed_perf_evaluations'] is None


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_reader_native_rosters_cache_changed_source_and_byte_bounds(tmp_path):
    run, ws, scale = tmp_path / 'run', tmp_path / 'workspace', tmp_path / 'scale'
    cal = tmp_path / 'cal.json'
    write(cal, {'members': {'cal': {}}})
    write(run / 'joint_campaign.json', {'policy': {'gsim_correctness': {'capsules': ['case', 'refusal']},
          'native_calibration': {'path': str(cal), 'sha256': hashlib.sha256(cal.read_bytes()).hexdigest()}}})
    write(run / 'environment.yaml', {'public_corpus_input': {'public_scalability': {'path': str(scale)}}})
    write(scale / 'selection.json', {'schema': 'merlin.corpus.public_scalability.staged.v1', 'cases': {'scale': 'pin'}})
    write(ws / 'qa/progress.json', report(A, [row('case')]))
    reader = Capsules(interval=0)
    first = reader.sample(run, ws)
    assert [r['total'] for r in first['cohorts']] == [2, 1, 1]
    assert first['total']['L2'] == 1 and first['bytes_read'] > 0
    assert reader.sample(run, ws)['bytes_read'] == 0
    write(ws / 'qa/progress.json', report(B, []))
    second = reader.sample(run, ws)
    assert second['source_sha256'] == B and second['total']['L2'] == 0
    bounded = Capsules(interval=0, max_bytes=10).sample(run, ws)
    assert bounded['partial'] and bounded['total']['L2'] is None
    write(cal, {'members': {'unreviewed': {}}})
    assert reader.sample(run, ws)['cohorts'][1]['total'] is None


def test_native_whole_program_nonapplicability_counts_but_unexplained_skips_do_not():
    item = row('scale')
    item['tiers']['L0'] = {'status': 'skipped', 'mandatory': True, 'not_applicable': True,
        'reason': 'command-buffer reference cannot interpret scalar regions of a whole-program ELF'}
    item['tiers']['L1'] = {'status': 'skipped', 'mandatory': True, 'not_applicable': True,
        'reason': 'command-buffer simulate cannot execute scalar regions of a whole-program ELF'}
    item['numeric'].update(per_output={'Y0': {'status': 'pass', 'n_elements': 4}},
                           missing_outputs=[], mismatch_count=0, golden_source='native selected capsule')
    assert project(ROSTERS, report(A, [item]), {}, {}, [])['total']['all'] == 1
    item['tiers']['L0']['reason'] = 'arbitrary skip'
    assert project(ROSTERS, report(A, [item]), {}, {}, [])['total']['all'] == 0
    item = row('case')  # Functional policy requires L3 even if this request only declared L2.
    assert project(ROSTERS, report(A, [item]), {}, {}, [])['total']['all'] == 0
