"""Only native pre-candidate package gates defer an entire selected cohort."""
from pathlib import Path

import pytest

from merlin.targetgen import capsule_grade as CG, capsule_runner as CR
from merlin.targetgen.package_runtime import CertFailure, Package
from merlin_experiments.phase1.feedback import joint_test_scope as J


@pytest.mark.parametrize('stage', ['load_package', 'integrity_scan', 'build_package'])
def test_native_precheck_scope_excludes_arbitrary_build_failure(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(CG, 'source_experiment_env', lambda target: {})
    def refuse(*args, **kwargs):
        raise CertFailure('build' if stage == 'build_package' else 'contract', 'structural_invariant_violation', 'original refusal')
    monkeypatch.setattr(CG, 'load_package', lambda root, **kwargs: Package(Path(root), {}, Path(root)/'compiler.py'))
    monkeypatch.setattr(CG, 'integrity_scan', lambda *args, **kwargs: None)
    monkeypatch.setattr(CG, 'build_package', lambda *args, **kwargs: None)
    monkeypatch.setattr(CG, stage, refuse)
    monkeypatch.setattr(CR, 'discover_capsules', lambda *a, **k: pytest.fail('precheck failure dispatched discovery'))
    monkeypatch.setattr(CR, 'run_suite', lambda *a, **k: pytest.fail('precheck failure dispatched candidate'))
    score = CG.grade(tmp_path, capsules_root=tmp_path/'caps', runs_root=tmp_path/'runs', target='fixture')
    assert score['failure']['detail'] == 'original refusal'
    assert score['n_capsules'] == score['n_passed'] == 0
    if stage == 'build_package':
        assert 'package_precheck_refusal' not in score
    else:
        assert score['package_precheck_refusal']['stage'] == stage
        assert score['package_precheck_refusal']['candidate_execution_started'] is False
        assert score['package_precheck_refusal']['reuse_authority'] is False


def selection():
    return {'scope': 'rolling', 'required_development': 4, 'selected': ['A', 'B'], 'pending': ['C', 'D']}


def verdict(source='sourceA'):
    return {'package_failure': {'plane': 'contract', 'category': 'structural_invariant_violation', 'detail': 'invalid manifest'},
        'n_passed': 0, 'n_capsules': 0, 'per_capsule': [],
        'package_precheck_refusal': {'schema': 'merlin.package_precheck_refusal.v1', 'scope': 'selected_suite',
            'stage': 'load_package', 'candidate_execution_started': False, 'source_sha256': source}}


def test_pending_rows_share_diagnostic_without_inventing_execution_or_passes():
    out = J.annotate(verdict(), selection())
    assert out['n_required_development'] == out['n_pending_development'] == 4
    assert out['n_capsules'] == 2 and out['n_executed_capsules'] == 0 and out['n_passed'] == 0
    assert out['all_pass'] is False and out['certified_complete'] is False
    ref = out['shared_package_diagnostic']['sha256']
    assert out['shared_package_diagnostic']['reuse_authority'] is False
    for row in out['per_capsule']:
        assert row['shared_package_diagnostic_sha256'] == ref
        assert row['execution_status'] == 'not_started' and row['correctness'] == 'unknown'
        assert row['tiers'] == {} and row['status'] == 'gated'
    changed = J.annotate(verdict('different-source'), selection())
    assert changed['shared_package_diagnostic']['sha256'] != ref


def test_same_error_text_or_case_failure_cannot_issue_suite_scope():
    original = verdict()
    original.pop('package_precheck_refusal')
    out = J.annotate(original, selection())
    assert out['per_capsule'] == [] and 'shared_package_diagnostic' not in out
    case = verdict()
    case['per_capsule'] = [{'capsule': 'A', 'status': 'fail', 'failure_detail': 'invalid manifest'}]
    out = J.annotate(case, selection())
    assert len(out['per_capsule']) == 1 and 'shared_package_diagnostic' not in out
