"""Compact feedback preserves all public denominators without freezing optimization."""
import json

from merlin_experiments.phase1.feedback import brief
from merlin_experiments.phase1.feedback.capability_brief import render


def inventory():
    names = [('original_correctness', 153), ('calibration', 20), ('scalability_performance', 77)]
    return {'schema': 'merlin.public_capability_inventory.v1', 'members': {
        f'{cohort}-{index}': {'cohort': cohort, 'family': f'capability-{index % 7}'}
        for cohort, count in names for index in range(count)}}


def test_all_denominators_bounded_failures_and_no_private_payload():
    inv = inventory()
    rows = [{'capsule': name, 'status': 'fail', 'failure_plane': 'emission',
             'stderr': 'DO_NOT_COPY_RAW_OUTPUT' * 10000} for name in inv['members']]
    rows.append({'capsule': 'PRIVATE_HIDDEN_NAME', 'status': 'fail', 'failure_plane': 'DO_NOT_COPY'})
    text = render(inv, {'per_capsule': rows, 'graded_candidate_sha256': 'a' * 64})
    assert '| original_correctness | 153 / 153 |' in text
    assert '| calibration | 20 / 20 |' in text
    assert '| scalability_performance | 77 / 77 |' in text
    assert '7 failure families total' in text
    assert text.count('cases; examples') == 3
    assert 'DO_NOT_COPY' not in text and 'PRIVATE_HIDDEN_NAME' not in text
    assert len(text) < 6000
    assert 'Passing code may be improved' in text
    assert 'not automatically to current edits' in text


def test_missing_and_distinct_tier_counts():
    inv = inventory()
    row = {'capsule': 'original_correctness-0', 'status': 'pass',
           'tiers': {'L2': {'status': 'pass'}, 'L3': {'status': 'unavailable'}}}
    text = render(inv, {'per_capsule': [row]})
    assert '| original_correctness | 1 / 153 | 1 | 1 | 0 |' in text
    assert '| calibration | 0 / 20 | 0 | 0 | 0 |' in text
    assert 'unobserved' in text


def test_joint_brief_handles_continuous_keys_and_keeps_errata(tmp_path):
    run, ws = tmp_path / 'run', tmp_path / 'ws'
    history = run / 'qa_history'
    history.mkdir(parents=True)
    ws.mkdir()
    (ws / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps(inventory()))
    (run / 'ERRATA.md').write_text('Review this important correction.')
    for index, status in enumerate(('pass', 'fail')):
        doc = {'per_capsule': [{'capsule': 'scalability_performance-0', 'status': status}]}
        (history / f'verdict_round_r0000_t{index:06d}.json').write_text(json.dumps(doc))
    text = brief.build(run, ws, 2)
    assert 'Reported snapshot regressions: **1**' in text
    assert 'important correction' in text
    assert 'Do NOT re-work' not in text and 'Passing (locked)' not in text
    assert 'Compiler approach ledger' in text


def test_legacy_brief_selection_unchanged(tmp_path):
    (tmp_path / 'ws').mkdir()
    (tmp_path / 'run').mkdir()
    assert brief.build(tmp_path / 'run', tmp_path / 'ws', 0).startswith('# Round brief')
