"""Streaming capsule feedback uses the ordinary redactor, never a final verdict."""
import json

from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import qa


def test_progress_is_redacted_before_final_grade_and_honors_worker_bound(tmp_path, monkeypatch):
    context = InvocationContext(tmp_path, tmp_path/'descriptor', tmp_path, 'fixture',
                                tmp_path, tmp_path, tmp_path, ())
    monkeypatch.setattr(qa, '_loop_target_sim_via', lambda _: ('fixture', ''))
    monkeypatch.setattr(qa.CR, 'qa_checkpoint_adapters', lambda *a, **k: {'L2': object()})
    received = []

    def grade(*args, **kwargs):
        assert kwargs['max_workers'] == 4
        raw = {'capsule': 'case', 'status': 'pass', 'numeric': {
            'status': 'pass', 'mismatch_count': 0,
            'first_mismatch': {'expected': 'PRIVATE_ANSWER_SENTINEL'},
        }, 'tiers': {'L2': {'status': 'pass'}},
            'private_path': '/operator/PRIVATE_ANSWER_SENTINEL'}
        destination = tmp_path/'runs/fixture-suite/case/capsule_result.json'
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(raw))
        kwargs['on_capsule_complete'](raw)
        assert len(received) == 1
        assert 'PRIVATE_ANSWER_SENTINEL' not in json.dumps(received)
        assert received[0]['numeric_status'] == 'pass'
        assert received[0]['tiers'] == {'L2': 'pass'}
        return {'n_capsules': 1, 'n_passed': 1, 'per_capsule': [
            {'capsule': 'case', 'status': 'pass', 'tiers': {'L2': 'pass'}},
        ]}

    monkeypatch.setattr(qa.CG, 'grade', grade)
    final = qa.run('submission', str(tmp_path), tmp_path, {'public'}, False, 1,
                   context=context, on_capsule_complete=received.append, max_workers=4)
    assert final['all_pass'] is False  # Missing structural/cohort closure is not certified.
    assert final['n_passed'] == 1


def test_background_diagnostic_deferral_keeps_mandatory_shape_gate(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import loop_grading as G
    from merlin_experiments.phase1.feedback import promotion

    from merlin.targetgen import lowering_coverage

    ws, run = tmp_path/'workspace', tmp_path/'run'
    (ws/'submission').mkdir(parents=True)
    (ws/'submission/manifest.yaml').write_text('language: python\n')
    context = InvocationContext(tmp_path, tmp_path/'descriptor', tmp_path, 'fixture',
                                tmp_path, tmp_path, tmp_path, ())
    inputs = G.GradingInputs(context, 'raw_baseline', tmp_path/'public', (), None)
    monkeypatch.setenv('MERLIN_BACKGROUND_CODEGEN_SCALABILITY', '0')
    monkeypatch.setattr(lowering_coverage, 'sweep', lambda *a, **k: {'all_covered': False, 'corners': []})
    monkeypatch.setattr(G, '_attach_codegen_scalability', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('background advisory probe should be deferred')))
    monkeypatch.setattr(G, '_write_stage_ledger', lambda *a, **k: None)
    monkeypatch.setattr(G, '_record_plateau', lambda *a: None)
    monkeypatch.setattr(G.FL, 'record_channel_health', lambda *a: None)
    monkeypatch.setattr(promotion, 'resolve_tiers', lambda *a, **k: (None, None, None))
    result = G.grade(ws, run, 1, False, 5, inputs=inputs, advisory_background=True,
        qa_runner=lambda *a, **k: {'all_pass': True, 'n_capsules': 1, 'n_passed': 1, 'per_capsule': []})
    assert result['shape_coverage']['ran'] is True
    assert result['shape_coverage']['all_covered'] is False
    assert result['all_pass'] is False
    assert result['codegen_scalability']['ran'] is False
