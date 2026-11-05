"""Synthetic policy/completion controls; no hidden corpus or hardware evidence."""

from copy import deepcopy

import pytest
import yaml
from merlin_experiments.phase1.feedback import certification, formal
from merlin_experiments.phase1.feedback import hidden_capsule_policy as H
from merlin_experiments.phase1.options import parse_options


def clean_score():
    return {
        'n_capsules': 2, 'n_passed': 2, 'functional_pass': 1, 'gradeable': True,
        'integrity_status': 'clean', 'numeric_all_exact': True, 'trace_all_pass': True,
        'model_execution_all_pass': None, 'structural_evidence_all_pass': True,
        'structural_evidence_scope': {'n_instruction_trace_capsules': 2, 'n_model_execution_capsules': 0},
        'cohort_admission': {'policy': 'all_discovered', 'n_source_capsules': 2, 'n_admitted_capsules': 2,
                             'n_capability_excluded': 0, 'n_resource_excluded': 0,
                             'excluded_name_set_sha256': '0' * 64, 'admitted_name_set_sha256': '1' * 64},
        **{field: 0 for field in ('n_not_graded_ineligible', 'n_gated_deferred', 'n_screened_only',
                                  'n_budget_exhausted', 'n_incomplete', 'n_not_gradeable_no_oracle')},
        'tier_reached': {'L2': 2, 'L3': 0}, 'pass_evidence': {'rtl_backed': 0},
        'functional_hidden_evidence': {'tier': 'L2', 'engine': 'spike', 'fidelity': 'functional_model',
                                       'claim': 'functional_corroboration', 'n_admitted_capsules': 2,
                                       'n_spike_passed': 2, 'n_validated_refusals': 0,
                                       'n_higher_tier_measured': 0},
    }


def test_operator_and_descriptor_selection_are_explicit_and_immutable(tmp_path):
    descriptor = tmp_path / 'descriptor.yaml'
    descriptor.write_text('target: fixture\n')
    assert H.resolve(descriptor) == H.record('rtl')
    assert H.resolve(descriptor, 'functional_spike') == H.record('functional_spike')
    descriptor.write_text('grading:\n  hidden_capsule_mode: functional_spike\n')
    assert H.resolve(descriptor) == H.record('functional_spike')
    with pytest.raises(ValueError, match='conflicts'):
        H.resolve(descriptor, 'rtl')
    with pytest.raises(ValueError, match='frozen run'):
        H.verify_record(H.record('functional_spike'), H.record('rtl'))
    with pytest.raises(ValueError, match='frozen run'):
        H.verify_record(H.record('functional_spike'), None)
    H.verify_record(H.record('rtl'), None)
    absent = tmp_path / 'absent.yaml'
    assert H.resolve(absent) == H.record('rtl')
    with pytest.raises(ValueError, match='selected descriptor'):
        H.resolve(absent, 'functional_spike')


def test_mode_cli_is_closed_and_does_not_change_author_budget():
    baseline = parse_options(['--run-id', 'fixture'], environ={})
    selected = parse_options(['--run-id', 'fixture', '--hidden-capsule-mode', 'functional_spike'], environ={})
    assert selected.hidden_capsule_mode == 'functional_spike'
    assert (selected.max_wall_s, selected.round_timeout) == (baseline.max_wall_s, baseline.round_timeout)
    with pytest.raises(SystemExit):
        parse_options(['--run-id', 'fixture', '--hidden-capsule-mode', 'skip'], environ={})


def test_functional_hidden_complete_is_not_public_RTL_completion():
    score = clean_score()
    assert formal.phase_completion(score, required_tier='L2', functional_hidden=True) == (True, [])
    public, failures = formal.phase_completion(score)
    assert not public and 'not_all_capsules_reached_L3' in failures and 'not_all_capsules_rtl_backed' in failures
    phase = formal._phase_manifest(score, required_tier='L2', functional_hidden=True)
    assert phase['claim'] == 'functional_corroboration' and phase['rtl_certified'] is False


@pytest.mark.parametrize('field,value', [
    ('numeric_all_exact', False), ('trace_all_pass', None), ('structural_evidence_all_pass', False),
    ('integrity_status', 'dirty'), ('n_incomplete', 1), ('n_not_gradeable_no_oracle', 1),
    ('n_capsules', 0), ('n_passed', True),
])
def test_missing_applicable_evidence_or_unmeasured_case_refuses(field, value):
    score = clean_score()
    score[field] = value
    assert formal.phase_completion(score, required_tier='L2', functional_hidden=True)[0] is False


@pytest.mark.parametrize('field,value', [
    ('engine', 'gsim'), ('fidelity', 'elaborated_rtl'), ('n_spike_passed', 1),
    ('n_spike_passed', True), ('n_admitted_capsules', 1), ('n_higher_tier_measured', 1),
    ('n_validated_refusals', 1),
])
def test_only_actual_complete_Spike_evidence_can_satisfy_hidden_policy(field, value):
    score = clean_score()
    score['functional_hidden_evidence'][field] = value
    assert formal.phase_completion(score, required_tier='L2', functional_hidden=True)[0] is False


def test_launcher_requires_same_functional_hidden_selection_and_preserves_public_L3(tmp_path):
    score = clean_score()
    hidden = formal._phase_manifest(score, required_tier='L2', functional_hidden=True)
    manifest = {'public_dev': deepcopy(hidden), 'hidden': hidden,
                'hidden_capsule_validation': H.record('functional_spike'),
                'completion': {'formal_grade_complete': True, 'required_tier': 'L3',
                               'hidden_required_tier': 'L2', 'hidden_validation_mode': 'functional_spike'}}
    (tmp_path / 'run_manifest.yaml').write_text(yaml.safe_dump(manifest))
    chosen = certification._official_grade_result(0, tmp_path, hidden_capsule_mode='functional_spike')
    assert 'public_dev:not_all_capsules_reached_L3' in chosen['failures']
    assert not any(reason.startswith('hidden:') for reason in chosen['failures'])
    unchosen = certification._official_grade_result(0, tmp_path)
    assert 'hidden_capsule_validation_mismatch' in unchosen['failures']
    assert 'hidden:not_all_capsules_reached_L3' in unchosen['failures']


def refusal_score():
    score = clean_score()
    score['expected_refusal_scope'] = {
        'binding': 'original_capsule_declaration_and_canonical_result_v1',
        'n_admitted_capsules': 2, 'n_declared_refusals': 1,
        'n_validated_refusals': 1, 'n_executable_capsules': 1,
    }
    score['structural_evidence_scope']['n_instruction_trace_capsules'] = 1
    score['tier_reached']['L2'] = 1
    score['functional_hidden_evidence'].update(n_spike_passed=1, n_validated_refusals=1)
    return score


def test_refusal_and_execution_have_separate_positive_evidence(tmp_path):
    score = refusal_score()
    assert formal.phase_completion(score, required_tier='L2', functional_hidden=True) == (True, [])
    score['tier_reached']['L3'] = score['pass_evidence']['rtl_backed'] = 1
    assert formal.phase_completion(score) == (True, [])
    phase = formal._phase_manifest(score, required_tier='L3')
    (tmp_path / 'run_manifest.yaml').write_text(yaml.safe_dump({
        'completion': {'formal_grade_complete': True, 'required_tier': 'L3'},
        'public_dev': phase, 'hidden': phase,
    }))
    status = certification._official_grade_result(0, tmp_path)
    assert not any(reason.startswith(('public_dev:', 'hidden:')) for reason in status['failures'])
    score['tier_reached']['L3'] = 0
    assert formal.phase_completion(score)[0] is False  # The executable still requires actual L3.


@pytest.mark.parametrize('field,value', [
    ('n_validated_refusals', 0), ('n_validated_refusals', True), ('n_executable_capsules', 0),
    ('n_declared_refusals', 2), ('n_admitted_capsules', 1), ('binding', 'claimed_by_candidate'),
])
def test_refusal_applicability_cannot_hide_unmeasured_or_executable_cases(field, value):
    score = refusal_score()
    score['expected_refusal_scope'][field] = value
    assert not formal.phase_completion(score, required_tier='L2', functional_hidden=True)[0]


def test_all_refusals_are_measured_without_fake_spike_or_rtl_passes():
    score = refusal_score()
    score['expected_refusal_scope'].update(n_declared_refusals=2, n_validated_refusals=2, n_executable_capsules=0)
    score['structural_evidence_scope']['n_instruction_trace_capsules'] = 0
    score['trace_all_pass'] = None
    score['tier_reached'] = {'L2': 0, 'L3': 0}
    score['functional_hidden_evidence'].update(n_spike_passed=0, n_validated_refusals=2)
    assert formal.phase_completion(score, required_tier='L2', functional_hidden=True) == (True, [])
    assert formal.phase_completion(score) == (True, [])


def test_checkpoint_freshly_revalidates_refusal_without_native_credit():
    raw = {'refusal': {'status': 'pass', 'tiers': {}, 'expected_refusal': {'status': 'met'}},
           'exec': {'status': 'pass', 'tiers': {'L3': 'pass'}}}
    proof = {'capsule': 'refusal', 'validated_refusal': True, 'qualification': 'expected_refusal',
             'native_executed': False, 'expected_refusal': {'status': 'met'}}
    eligible, held = certification._l3_promotion({'per_capsule': [
        {'capsule': 'refusal', 'status': 'pass', 'tiers': {}},
        {'capsule': 'exec', 'status': 'pass', 'tiers': {'L2': 'pass'}},
    ]}, 'L2')
    assert eligible == ['refusal', 'exec'] and held == []  # Both get the CURRENT checkpoint.
    joined = certification._checkpoint_refusal_rows(raw, {'per_capsule': [proof]})
    rows, passed, total = certification._l3_attempt_tally(joined, [], 'L3')
    assert (passed, total) == (2, 2)
    refused = next(row for row in rows if row['capsule'] == 'refusal')
    assert refused['l3_status'] == 'not_applicable_expected_refusal' and refused['native_executed'] is False
    assert sum(row['l3_status'] == 'pass' for row in rows) == 1
    assert 'validated_refusal' not in raw['refusal']


def test_raw_or_previous_loop_refusal_cannot_complete_checkpoint():
    raw = {'refusal': {'status': 'pass', 'tiers': {}, 'validated_refusal': True,
                       'qualification': 'expected_refusal', 'native_executed': False,
                       'expected_refusal': {'status': 'met'}}}
    for score in (None, {}, {'per_capsule': []}):
        joined = certification._checkpoint_refusal_rows(raw, score)
        assert certification._l3_attempt_tally(joined, [], 'L3')[1:] == (0, 1)
    assert certification._l3_attempt_tally({}, [{'capsule': 'refusal', **raw['refusal']}], 'L3')[1:] == (0, 1)
